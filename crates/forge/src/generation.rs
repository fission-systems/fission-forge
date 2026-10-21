//! Deterministic, bounded byte generation with detached incremental KV inference.
use crate::{
    data,
    model::{Decoder, ModelConfig},
    pcode, training,
};
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use serde::Serialize;
use std::{fs, path::Path};

#[derive(Debug, Serialize)]
pub struct Completion {
    pub tokens: Vec<u32>,
    pub bytes: Vec<u8>,
    pub stop_reason: &'static str,
}

fn prefix(
    prompt: &str,
    context: usize,
    max_new_tokens: usize,
    codec: &crate::tokenizer::Codec,
) -> Result<Vec<u32>> {
    ensure!(!prompt.is_empty(), "empty generation prompt");
    ensure!(
        max_new_tokens > 0 && max_new_tokens <= 4096,
        "generation budget must be 1..4096 tokens"
    );
    let mut ids = vec![data::BOS];
    ids.extend(codec.encode(prompt.as_bytes()));
    ids.push(data::SEP);
    ensure!(
        ids.len() <= context,
        "prompt exceeds context; truncation is forbidden"
    );
    Ok(ids)
}

pub fn decode(
    prompt: &str,
    context: usize,
    max_new_tokens: usize,
    logits: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<Completion> {
    decode_with_codec(
        prompt,
        context,
        max_new_tokens,
        &crate::tokenizer::Codec::new(None)?,
        logits,
    )
}
fn decode_with_codec(
    prompt: &str,
    context: usize,
    max_new_tokens: usize,
    codec: &crate::tokenizer::Codec,
    mut logits: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<Completion> {
    let mut ids = prefix(prompt, context, max_new_tokens, codec)?;
    let mut result = Completion {
        tokens: Vec::new(),
        bytes: Vec::new(),
        stop_reason: "max_new_tokens",
    };
    for _ in 0..max_new_tokens {
        if ids.len() > context {
            result.stop_reason = "context_limit";
            break;
        }
        let scores = logits(&ids)?;
        ensure!(
            scores.len() == codec.vocab_size() && scores.iter().all(|v| v.is_finite()),
            "invalid generation logits"
        );
        // Only EOS and byte tokens are legal in a completion. Stable low-ID tie break.
        let mut best = data::EOS as usize;
        for candidate in 4..codec.vocab_size() {
            if scores[candidate] > scores[best] {
                best = candidate;
            }
        }
        let token = best as u32;
        if token == data::EOS {
            result.tokens.push(token);
            result.stop_reason = "eos";
            break;
        }
        let piece = codec.piece(token)?;
        if result.bytes.len() + piece.len() > max_new_tokens {
            result.stop_reason = "byte_limit";
            break;
        }
        result.tokens.push(token);
        result.bytes.extend_from_slice(piece);
        ids.push(token);
    }
    Ok(result)
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub evidence_scope: &'static str,
    pub representation: String,
    pub checkpoint_representations: Vec<String>,
    pub checkpoint_sha256: String,
    pub checkpoint_state_sha256: String,
    pub prompt_sha256: String,
    pub prompt_bytes: usize,
    pub max_new_tokens: usize,
    pub tokenizer: String,
    pub tokenizer_sha256: Option<String>,
    pub max_output_bytes: usize,
    pub decoding: &'static str,
    pub model: ModelConfig,
    pub backend: String,
    pub completion: Completion,
    pub text: Option<String>,
}

pub fn run(
    checkpoint: &Path,
    prompt: &str,
    representation: &str,
    max_new_tokens: usize,
    output: &Path,
    device: &Device,
) -> Result<Report> {
    ensure!(
        !output.exists(),
        "output already exists: {}",
        output.display()
    );
    let state_bytes = fs::read(checkpoint.join("state.json"))?;
    let state: training::Checkpoint = serde_json::from_slice(&state_bytes)?;
    ensure!(
        (1..=5).contains(&state.schema_version),
        "unsupported checkpoint schema/tokenizer"
    );
    let codec = crate::tokenizer::Codec::new(state.tokenizer_spec.clone())?;
    ensure!(
        state.tokenizer == codec.name()
            && state.model.vocab_size == codec.vocab_size()
            && ((state.schema_version == 5) == codec.spec.is_some()),
        "checkpoint codec mismatch"
    );
    ensure!(
        matches!(
            representation,
            "assembly-v0"
                | pcode::REPRESENTATION
                | pcode::cfg::REPRESENTATION
                | pcode::cfg::slice::REPRESENTATION
        ),
        "unknown input representation"
    );
    ensure!(
        state.schema_version < 3 || state.representations.iter().any(|r| r == representation),
        "checkpoint was not trained on requested input representation"
    );
    // Check context/budget before loading weights or allocating a decoder.
    prefix(prompt, state.model.max_seq_len, max_new_tokens, &codec)?;
    let (vars, digest) = load_weights(checkpoint, &state, device)?;
    let model = Decoder::for_eval(&state.model, &vars, device)?;
    let mut cache: Option<crate::model::DecodeCache> = None;
    let query_chunk = state
        .corpus
        .as_ref()
        .map_or(256, |s| s.settings.attention_chunk);
    let completion = decode_with_codec(
        prompt,
        state.model.max_seq_len,
        max_new_tokens,
        &codec,
        |ids| {
            let logits = if let Some(cache) = &mut cache {
                ensure!(
                    cache.position() + 1 == ids.len(),
                    "generation cache length mismatch"
                );
                model.decode_step(&Tensor::new(&[[*ids.last().unwrap()]], device)?, cache)?
            } else {
                let (logits, prefix_cache) = model.prefill(
                    &Tensor::from_slice(ids, (1, ids.len()), device)?,
                    query_chunk,
                )?;
                cache = Some(prefix_cache);
                logits
            };
            Ok(logits.get(0)?.get(0)?.to_vec1::<f32>()?)
        },
    )?;
    let text = String::from_utf8(completion.bytes.clone()).ok();
    let report = Report {
        evidence_scope: "bounded_model_generation_only",
        representation: representation.into(),
        checkpoint_representations: state.representations,
        checkpoint_sha256: digest,
        checkpoint_state_sha256: pcode::sha256(&state_bytes),
        prompt_sha256: pcode::sha256(prompt.as_bytes()),
        prompt_bytes: prompt.len(),
        max_new_tokens,
        tokenizer: codec.name().into(),
        tokenizer_sha256: codec.sha256()?,
        max_output_bytes: max_new_tokens,
        decoding: if codec.spec.is_some() {
            "greedy_eos_and_bpe_tokens_kv_cache_byte_budget"
        } else {
            "greedy_eos_and_bytes_only_kv_cache"
        },
        model: state.model,
        backend: format!("{:?}", device.location()),
        completion,
        text,
    };
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    fs::write(output.join("completion.bin"), &report.completion.bytes)?;
    if let Some(text) = &report.text {
        fs::write(output.join("completion.txt"), text)?;
    }
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

pub(crate) fn load_weights(
    checkpoint: &Path,
    state: &training::Checkpoint,
    device: &Device,
) -> Result<(candle_nn::VarMap, String)> {
    let bytes = fs::read(checkpoint.join("training.safetensors"))?;
    let digest = pcode::sha256(&bytes);
    if state.schema_version >= 4 {
        ensure!(
            state.corpus.is_some() && state.checkpoint_sha256.as_deref() == Some(&digest),
            "corpus checkpoint integrity mismatch"
        );
    }
    let tensors = candle_core::safetensors::SliceSafetensors::new(&bytes)?;
    let vars = candle_nn::VarMap::new();
    let _ = Decoder::new(&state.model, &vars, device)?;
    let expected = crate::model::named_vars(&vars);
    let model_names: std::collections::BTreeSet<_> = tensors
        .tensors()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.starts_with("model."))
        .collect();
    ensure!(
        model_names.len() == expected.len(),
        "unexpected inference model tensor count"
    );
    for (name, variable) in expected {
        let tensor = tensors.load(&format!("model.{name}"), device)?;
        ensure!(
            tensor.shape() == variable.shape() && tensor.dtype() == variable.dtype(),
            "inference tensor shape/dtype mismatch: {name}"
        );
        variable.set(&tensor)?;
    }
    // Drop optimizer payload bytes after loading only model tensors onto the device.
    drop(bytes);
    Ok((vars, digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bpe_generation_preserves_pieces_and_stops_before_byte_budget() -> Result<()> {
        let codec = crate::tokenizer::Codec::new(Some(crate::tokenizer::Spec {
            algorithm: "lexical-byte-bpe-v1".into(),
            train_split_sha256: "a".repeat(64),
            source_weight: 8,
            merges: vec![[b'A' as u32 + 4, b'B' as u32 + 4]],
        }))?;
        let result = decode_with_codec("x", 16, 3, &codec, |_| {
            let mut scores = vec![0.; 261];
            scores[260] = 1.;
            Ok(scores)
        })?;
        assert_eq!(result.bytes, b"AB");
        assert_eq!(result.tokens, vec![260]);
        assert_eq!(result.stop_reason, "byte_limit");
        let mut step = 0;
        let result = decode_with_codec("x", 16, 4, &codec, |_| {
            let mut scores = vec![0.; 261];
            scores[if step == 0 { 260 } else { 3 }] = 1.;
            step += 1;
            Ok(scores)
        })?;
        assert_eq!(result.bytes, b"AB");
        assert_eq!(result.tokens, vec![260, 3]);
        assert_eq!(result.stop_reason, "eos");
        Ok(())
    }
    #[test]
    fn decoding_preserves_bytes_masks_prompt_tokens_and_reports_limits() -> Result<()> {
        let mut count = 0;
        let result = decode("x", 8, 4, |_| {
            let mut logits = vec![0.; 260];
            logits[0] = 100.;
            logits[1] = 100.;
            logits[2] = 100.;
            logits[if count == 0 { 259 } else { 3 }] = 1.;
            count += 1;
            Ok(logits)
        })?;
        assert_eq!(result.bytes, vec![255]);
        assert_eq!(result.stop_reason, "eos");
        let result = decode("x", 3, 3, |_| {
            let mut logits = vec![0.; 260];
            logits[69] = 1.;
            Ok(logits)
        })?;
        assert_eq!(result.bytes, b"A");
        assert_eq!(result.stop_reason, "context_limit");
        assert!(decode("xx", 3, 1, |_| unreachable!()).is_err());
        assert!(decode("x", 8, 0, |_| unreachable!()).is_err());
        assert!(decode("x", 8, 1, |_| Ok(vec![f32::NAN; 260])).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::training::Options;
    #[test]
    fn actual_return_slice_checkpoint_reaches_generation() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "forge-slice-generation-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let raw: pcode::RawDump = serde_json::from_str(include_str!(
            "../../../fixtures/pcode-cfg/forge_diamond.raw.json"
        ))?;
        let profile: pcode::Profile =
            serde_json::from_str(include_str!("../../../configs/lifter/fission-x86_64.json"))?;
        let source = "unsigned int f(unsigned int a, unsigned int b) { return (a > b ? a + 3u : b - 2u) + 1u; }";
        let pair = pcode::Pair {
            schema_version: 1,
            id: "software-generation-probe".into(),
            input: pcode::Input {
                representation: pcode::cfg::slice::REPRESENTATION.into(),
                profile,
                raw: raw.clone(),
            },
            source: source.into(),
            provenance: pcode::Provenance {
                binary_sha256: "0".repeat(64),
                raw_sha256: pcode::sha256(&serde_json::to_vec(&raw)?),
                source_sha256: pcode::sha256(source.as_bytes()),
                extraction: "software_test_not_a_functional_score".into(),
            },
        };
        let config = root.join("config.json");
        let dataset = root.join("pair.jsonl");
        let checkpoint = root.join("checkpoint");
        fs::write(
            &config,
            r#"{"vocab_size":260,"hidden_size":8,"heads":1,"layers":1,"feed_forward_size":16,"max_seq_len":8192,"rotary_positions":true}"#,
        )?;
        fs::write(&dataset, serde_json::to_vec(&pair)?)?;
        training::run(
            &config,
            &dataset,
            &checkpoint,
            &Device::Cpu,
            Options {
                steps: 1,
                learning_rate: 0.001,
                seed: 42,
                resume: None,
            },
        )?;
        let report = run(
            &checkpoint,
            &pair.prompt()?,
            pcode::cfg::slice::REPRESENTATION,
            1,
            &root.join("generation"),
            &Device::Cpu,
        )?;
        assert_eq!(report.representation, pcode::cfg::slice::REPRESENTATION);
        assert!(root.join("generation/completion.bin").exists());
        assert!(
            run(
                &checkpoint,
                &pair.prompt()?,
                pcode::cfg::REPRESENTATION,
                1,
                &root.join("wrong-lane"),
                &Device::Cpu
            )
            .is_err()
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn real_checkpoint_generates_memorized_c_and_checks_lane_before_loading_weights() -> Result<()>
    {
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = Temporary(
            std::env::temp_dir().join(format!("forge-generate-{}-{nonce}", std::process::id())),
        );
        fs::create_dir(&root.0)?;
        let config = root.0.join("config.json");
        let dataset = root.0.join("pairs.jsonl");
        let checkpoint = root.0.join("checkpoint");
        fs::write(
            &config,
            r#"{"vocab_size":260,"hidden_size":16,"heads":2,"layers":1,"feed_forward_size":32,"max_seq_len":64}"#,
        )?;
        let source = "int f(void){return 1;}";
        fs::write(
            &dataset,
            serde_json::to_vec(
                &serde_json::json!({"id":"generation-software-fixture","assembly":"ret","source":source}),
            )?,
        )?;
        training::run(
            &config,
            &dataset,
            &checkpoint,
            &Device::Cpu,
            Options {
                steps: 250,
                learning_rate: 0.01,
                seed: 42,
                resume: None,
            },
        )?;
        let report = run(
            &checkpoint,
            "ret",
            "assembly-v0",
            32,
            &root.0.join("generation"),
            &Device::Cpu,
        )?;
        assert_eq!(report.text.as_deref(), Some(source));
        assert_eq!(report.completion.stop_reason, "eos");
        let rejected = root.0.join("wrong-lane");
        assert!(
            run(
                &checkpoint,
                "ret",
                pcode::cfg::REPRESENTATION,
                32,
                &rejected,
                &Device::Cpu
            )
            .is_err()
        );
        assert!(!rejected.exists());
        assert!(
            run(
                &checkpoint,
                &"x".repeat(64),
                "assembly-v0",
                32,
                &root.0.join("too-long"),
                &Device::Cpu
            )
            .is_err()
        );
        assert!(!root.0.join("too-long").exists());
        Ok(())
    }
}

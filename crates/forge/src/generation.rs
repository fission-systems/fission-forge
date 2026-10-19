//! Deterministic, bounded byte generation. Full-prefix inference; no KV cache.
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

fn prefix(prompt: &str, context: usize, max_new_tokens: usize) -> Result<Vec<u32>> {
    ensure!(!prompt.is_empty(), "empty generation prompt");
    ensure!(
        max_new_tokens > 0 && max_new_tokens <= 4096,
        "generation budget must be 1..4096 tokens"
    );
    let mut ids = vec![data::BOS];
    ids.extend(prompt.bytes().map(|b| u32::from(b) + 4));
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
    mut logits: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<Completion> {
    let mut ids = prefix(prompt, context, max_new_tokens)?;
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
            scores.len() == 260 && scores.iter().all(|v| v.is_finite()),
            "invalid generation logits"
        );
        // Only EOS and byte tokens are legal in a completion. Stable low-ID tie break.
        let mut best = data::EOS as usize;
        for candidate in 4..260 {
            if scores[candidate] > scores[best] {
                best = candidate;
            }
        }
        let token = best as u32;
        result.tokens.push(token);
        if token == data::EOS {
            result.stop_reason = "eos";
            break;
        }
        result.bytes.push((token - 4) as u8);
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
        (1..=3).contains(&state.schema_version) && state.tokenizer == "byte-v1-260",
        "unsupported checkpoint schema/tokenizer"
    );
    ensure!(
        matches!(
            representation,
            "assembly-v0" | pcode::REPRESENTATION | pcode::cfg::REPRESENTATION
        ),
        "unknown input representation"
    );
    ensure!(
        state.schema_version < 3 || state.representations.iter().any(|r| r == representation),
        "checkpoint was not trained on requested input representation"
    );
    // Check context/budget before loading weights or allocating a decoder.
    prefix(prompt, state.model.max_seq_len, max_new_tokens)?;
    let bytes = fs::read(checkpoint.join("training.safetensors"))?;
    let digest = pcode::sha256(&bytes);
    let tensors = candle_core::safetensors::SliceSafetensors::new(&bytes)?;
    let vars = candle_nn::VarMap::new();
    let model = Decoder::new(&state.model, &vars, device)?;
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
    let completion = decode(prompt, state.model.max_seq_len, max_new_tokens, |ids| {
        let ids = Tensor::from_slice(ids, (1, ids.len()), device)?;
        Ok(model
            .forward(&ids)?
            .get(0)?
            .get(ids.dim(1)? - 1)?
            .to_vec1::<f32>()?)
    })?;
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
        decoding: "greedy_eos_and_bytes_only",
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

#[cfg(test)]
mod tests {
    use super::*;
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

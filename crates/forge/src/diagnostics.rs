//! Frozen-checkpoint input-conditioning diagnostic; no training or generation.
use crate::{data, generation, model::Decoder, pcode, tokenizer::Codec, training::Checkpoint};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use serde_json::json;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
const HEADER: &str = "unsigned int f(unsigned int a, unsigned int b) { return ";

fn splice(
    prompt: &[u32],
    start: usize,
    target: &[u32],
    target_start: usize,
) -> Result<(Vec<u32>, usize)> {
    ensure!(
        start >= 2 && prompt[0] == data::BOS && prompt[start - 1] == data::SEP,
        "invalid donor prefix"
    );
    ensure!(
        target_start < target.len() && target.last() == Some(&data::EOS),
        "invalid target completion"
    );
    let mut ids = prompt[..start].to_vec();
    ids.extend_from_slice(&target[target_start..]);
    Ok((ids, start))
}
fn token_losses(
    model: &Decoder,
    ids: Vec<u32>,
    start: usize,
    context: usize,
    chunk: usize,
    device: &Device,
) -> Result<Vec<f32>> {
    let batch = data::encoded_batch(
        &[(ids, start)],
        vec![pcode::cfg::REPRESENTATION.into()],
        context,
        device,
    )?;
    let logits = model.forward_chunked(&batch.input, chunk)?;
    let (_, _, vocab) = logits.dims3()?;
    let logp = candle_nn::ops::log_softmax(&logits.reshape(((), vocab))?, 1)?;
    let selected = logp
        .gather(&batch.target.flatten_all()?.unsqueeze(1)?, 1)?
        .squeeze(1)?
        .neg()?
        .to_vec1::<f32>()?;
    let losses = selected[start - 1..].to_vec();
    ensure!(
        losses.len() == batch.completion_tokens && losses.iter().all(|x| x.is_finite()),
        "invalid likelihood vector"
    );
    Ok(losses)
}
pub fn run(
    root: &Path,
    checkpoint: &Path,
    output: &Path,
    device: &Device,
) -> Result<serde_json::Value> {
    ensure!(!output.exists(), "output exists");
    crate::corpus::run(root, false)?;
    let state_bytes = fs::read(checkpoint.join("state.json"))?;
    let state: Checkpoint = serde_json::from_slice(&state_bytes)?;
    ensure!(
        (4..=5).contains(&state.schema_version),
        "requires corpus checkpoint"
    );
    let resume = state.corpus.as_ref().context("missing corpus state")?;
    ensure!(
        pcode::sha256(&fs::read(root.join("release/lock.json"))?) == resume.corpus_lock_sha256,
        "checkpoint corpus changed"
    );
    ensure!(
        state.representations == [pcode::cfg::REPRESENTATION],
        "unsupported diagnostic representation"
    );
    let codec = Codec::new(state.tokenizer_spec.clone())?;
    ensure!(
        codec.name() == state.tokenizer && codec.vocab_size() == state.model.vocab_size,
        "codec mismatch"
    );
    if let Some(spec) = &codec.spec {
        ensure!(
            spec.train_split_sha256 == pcode::sha256(&fs::read(root.join("release/train.jsonl"))?),
            "tokenizer split changed"
        );
    }
    let pairs: Vec<pcode::Pair> =
        BufReader::new(fs::File::open(root.join("release/validation.jsonl"))?)
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect::<Result<_>>()?;
    ensure!(
        pairs.len() == 130,
        "diagnostic is frozen to the 130-row composition validation"
    );
    let encoded: Vec<_> = pairs
        .iter()
        .map(|p| Ok(codec.pair(&p.prompt()?, &p.source)))
        .collect::<Result<_>>()?;
    let header = codec.encode(HEADER.as_bytes());
    for (p, (tokens, start)) in pairs.iter().zip(&encoded) {
        ensure!(
            p.source.starts_with(HEADER) && tokens[*start..].starts_with(&header),
            "outside header contract"
        );
    }
    let (vars, digest) = generation::load_weights(checkpoint, &state, device)?;
    let model = Decoder::for_eval(&state.model, &vars, device)?;
    let mut records = Vec::new();
    let mut totals = [[0f64; 3]; 2];
    let mut counts = [0usize; 3];
    for (i, pair) in pairs.iter().enumerate() {
        let donor = (i + 17) % pairs.len();
        ensure!(
            pair.id[..8] != pairs[donor].id[..8],
            "control donor must have a different template family"
        );
        let (target, start) = &encoded[i];
        let (other, other_start) = &encoded[donor];
        let control = splice(other, *other_start, target, *start)?;
        let mut row = Vec::new();
        for (arm, (ids, first)) in [(target.clone(), *start), control].into_iter().enumerate() {
            let nll = token_losses(
                &model,
                ids,
                first,
                state.model.max_seq_len,
                resume.settings.attention_chunk,
                device,
            )?;
            let sums = [
                nll.iter().map(|x| f64::from(*x)).sum(),
                nll[..header.len()].iter().map(|x| f64::from(*x)).sum(),
                nll[header.len()..].iter().map(|x| f64::from(*x)).sum(),
            ];
            for (j, sum) in sums.iter().enumerate() {
                totals[arm][j] += sum;
            }
            if arm == 0 {
                counts[0] += nll.len();
                counts[1] += header.len();
                counts[2] += nll.len() - header.len();
            }
            row.push(json!({"sum_nll":sums[0],"completion_tokens":nll.len(),"header_sum_nll":sums[1],"body_and_eos_sum_nll":sums[2]}));
        }
        records.push(json!({"id":pair.id,"donor_id":pairs[donor].id,"correct_prompt_sha256":pcode::sha256(pair.prompt()?.as_bytes()),"control_prompt_sha256":pcode::sha256(pairs[donor].prompt()?.as_bytes()),"correct":row[0],"wrong_prompt":row[1]}));
    }
    let arms:Vec<_>=totals.iter().map(|s|json!({"nll":s[0]/counts[0] as f64,"header_nll":s[1]/counts[1] as f64,"body_and_eos_nll":s[2]/counts[2] as f64})).collect();
    let report = json!({"schema_version":1,"scope":"teacher_forced_wrong_prompt_control_not_generated_C_accuracy","checkpoint_sha256":digest,"checkpoint_state_sha256":pcode::sha256(&state_bytes),"step":state.step,"corpus_lock_sha256":resume.corpus_lock_sha256,"validation_sha256":pcode::sha256(&fs::read(root.join("release/validation.jsonl"))?),"tokenizer_sha256":codec.sha256()?,"backend":format!("{:?}",device.location()),"model":state.model,"examples":pairs.len(),"test_model_examples":0,"optimizer_updates":0,"control":"cyclic_rotation_17_different_family_same_target_C","completion_tokens":counts[0],"header_tokens":counts[1],"body_and_eos_tokens":counts[2],"correct":arms[0],"wrong_prompt":arms[1],"wrong_minus_correct_nll":(totals[1][0]-totals[0][0])/counts[0] as f64,"records":records});
    fs::create_dir_all(output)?;
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
    fn wrong_prompt_control_preserves_the_complete_target_and_eos() -> Result<()> {
        let codec = Codec::new(None)?;
        let (a, sa) = codec.pair("correct", "C");
        let (b, sb) = codec.pair("wrong longer input", "different target");
        let (ids, start) = splice(&b, sb, &a, sa)?;
        assert_eq!(&ids[..start], &b[..sb]);
        assert_eq!(&ids[start..], &a[sa..]);
        assert_eq!(ids.last(), Some(&data::EOS));
        Ok(())
    }
}

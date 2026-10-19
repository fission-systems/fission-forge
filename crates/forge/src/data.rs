//! Byte-level conditional next-token batches. This is not a learned tokenizer.
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use serde::Deserialize;

pub const PAD: u32 = 0;
pub const BOS: u32 = 1;
pub const SEP: u32 = 2;
pub const EOS: u32 = 3;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssemblyPair {
    pub id: String,
    pub assembly: String,
    pub source: String,
}

pub struct Batch {
    pub input: Tensor,
    pub target: Tensor,
    pub mask: Tensor,
    pub examples: usize,
    pub sequence_length: usize,
    pub completion_tokens: usize,
    pub representations: Vec<String>,
    pub input_sha256: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Pair {
    Assembly(AssemblyPair),
    Pcode(Box<crate::pcode::Pair>),
}
impl Pair {
    fn representation(&self) -> &str {
        match self {
            Self::Assembly(_) => "assembly-v0",
            Self::Pcode(p) => &p.input.representation,
        }
    }
    fn parts(&self) -> Result<(&str, String, &str)> {
        match self {
            Self::Assembly(p) => Ok((&p.id, p.assembly.clone(), &p.source)),
            Self::Pcode(p) => Ok((&p.id, p.prompt()?, &p.source)),
        }
    }
}

pub fn encode(prompt: &str, source: &str) -> (Vec<u32>, usize) {
    let mut ids = vec![BOS];
    ids.extend(prompt.bytes().map(|b| b as u32 + 4));
    ids.push(SEP);
    let completion_start = ids.len();
    ids.extend(source.bytes().map(|b| b as u32 + 4));
    ids.push(EOS);
    (ids, completion_start)
}

pub fn batch(text: &str, context: usize, device: &Device) -> Result<Batch> {
    let pairs = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str::<Pair>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !pairs.is_empty() && pairs.len() <= 32,
        "smoke batch requires 1..32 examples"
    );
    let mut seen = std::collections::HashSet::new();
    let mut encoded = Vec::new();
    let mut representations = std::collections::BTreeSet::new();
    for pair in &pairs {
        let (id, prompt, source) = pair.parts()?;
        representations.insert(pair.representation().to_string());
        ensure!(
            !id.is_empty() && seen.insert(id),
            "empty or duplicate pair ID"
        );
        ensure!(
            !prompt.is_empty() && !source.is_empty(),
            "empty input or source for {}",
            id
        );
        let (tokens, start) = encode(&prompt, source);
        ensure!(
            tokens.len() - 1 <= context,
            "pair {} exceeds context; truncation is forbidden",
            id
        );
        encoded.push((tokens, start));
    }
    let time = encoded.iter().map(|(t, _)| t.len() - 1).max().unwrap();
    let mut input = vec![PAD; pairs.len() * time];
    let mut target = input.clone();
    let mut mask = vec![0f32; input.len()];
    for (b, (tokens, start)) in encoded.iter().enumerate() {
        for i in 0..tokens.len() - 1 {
            input[b * time + i] = tokens[i];
            target[b * time + i] = tokens[i + 1];
            if i + 1 >= *start {
                mask[b * time + i] = 1.;
            }
        }
    }
    let completion_tokens = mask.iter().filter(|m| **m == 1.).count();
    let shape = (pairs.len(), time);
    let input_sha256 = crate::pcode::sha256(&serde_json::to_vec(&(shape, &input, &target, &mask))?);
    Ok(Batch {
        representations: representations.into_iter().collect(),
        input_sha256,
        input: Tensor::from_vec(input, shape, device)?,
        target: Tensor::from_vec(target, shape, device)?,
        mask: Tensor::from_vec(mask, shape, device)?,
        examples: pairs.len(),
        sequence_length: time,
        completion_tokens,
    })
}

pub fn completion_loss(logits: &Tensor, batch: &Batch) -> Result<Tensor> {
    let (_, _, vocab) = logits.dims3()?;
    let log_probs = candle_nn::ops::log_softmax(&logits.reshape(((), vocab))?, 1)?;
    let selected = log_probs
        .gather(&batch.target.flatten_all()?.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    let masked = (selected.neg()? * batch.mask.flatten_all()?)?;
    masked
        .sum_all()?
        .affine(1. / batch.completion_tokens as f64, 0.)
        .context("completion-only cross entropy")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_and_padding_logits_do_not_contribute_to_loss() -> Result<()> {
        let text = "{\"id\":\"a\",\"assembly\":\"long input\",\"source\":\"C\"}\n{\"id\":\"b\",\"assembly\":\"x\",\"source\":\"D\"}";
        let batch = batch(text, 32, &Device::Cpu)?;
        let shape = (batch.examples, batch.sequence_length, 260);
        let baseline = Tensor::zeros(shape, candle_core::DType::F32, &Device::Cpu)?;
        let mut changed = vec![0f32; batch.examples * batch.sequence_length * 260];
        for (row, mask) in batch
            .mask
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .enumerate()
        {
            if *mask == 0. {
                changed[row * 260 + 5] = 100.;
            }
        }
        let changed = Tensor::from_vec(changed, shape, &Device::Cpu)?;
        let a = completion_loss(&baseline, &batch)?.to_scalar::<f32>()?;
        let b = completion_loss(&changed, &batch)?.to_scalar::<f32>()?;
        assert!((a - b).abs() < 1e-6);
        assert!((a - (260f32).ln()).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn shift_scores_first_source_byte_and_eos_but_not_prompt_or_padding() -> Result<()> {
        let text = "{\"id\":\"a\",\"assembly\":\"ab\",\"source\":\"C\"}\n{\"id\":\"b\",\"assembly\":\"x\",\"source\":\"DEF\"}";
        let batch = batch(text, 16, &Device::Cpu)?;
        let mask = batch.mask.to_vec2::<f32>()?;
        assert_eq!(mask[0], vec![0., 0., 0., 1., 1., 0.]);
        assert_eq!(mask[1], vec![0., 0., 1., 1., 1., 1.]);
        assert_eq!(batch.target.to_vec2::<u32>()?[0][3], b'C' as u32 + 4);
        assert_eq!(batch.completion_tokens, 6);
        assert!(super::batch(text, 2, &Device::Cpu).is_err());
        Ok(())
    }
}

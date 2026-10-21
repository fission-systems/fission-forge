//! Reversible lexical byte-pair encoding. Vocabulary learning reads the train split only.
use crate::{data, pcode};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub algorithm: String,
    pub train_split_sha256: String,
    pub source_weight: usize,
    pub merges: Vec<[u32; 2]>,
}
#[derive(Clone)]
pub struct Codec {
    pub spec: Option<Spec>,
    pieces: Vec<Vec<u8>>,
    ranks: BTreeMap<[u32; 2], u32>,
}
fn segments(text: &[u8]) -> Vec<Vec<u32>> {
    fn class(b: u8) -> u8 {
        if b.is_ascii_alphabetic() || b == b'_' {
            1
        } else if b.is_ascii_digit() {
            2
        } else if b == b' ' || b == b'\t' {
            3
        } else {
            0
        }
    }
    let mut result = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let kind = class(text[start]);
        let mut end = start + 1;
        if kind != 0 {
            while end < text.len() && class(text[end]) == kind {
                end += 1;
            }
        }
        result.push(text[start..end].iter().map(|b| u32::from(*b) + 4).collect());
        start = end;
    }
    result
}
fn merge(ids: &[u32], pair: [u32; 2], token: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i..i + 2] == pair {
            out.push(token);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}
impl Codec {
    pub fn new(spec: Option<Spec>) -> Result<Self> {
        let mut pieces = vec![Vec::new(); 4];
        pieces.extend((0..=255).map(|b| vec![b]));
        if let Some(s) = &spec {
            ensure!(
                s.algorithm == "lexical-byte-bpe-v1"
                    && s.source_weight == 8
                    && s.train_split_sha256.len() == 64
                    && s.train_split_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid BPE specification"
            );
            ensure!(s.merges.len() <= 3836, "BPE vocabulary too large");
            let mut seen = std::collections::BTreeSet::new();
            for &[left, right] in &s.merges {
                ensure!(
                    left >= 4
                        && right >= 4
                        && (left as usize) < pieces.len()
                        && (right as usize) < pieces.len()
                        && seen.insert([left, right]),
                    "invalid BPE merge"
                );
                let mut piece = pieces[left as usize].clone();
                piece.extend_from_slice(&pieces[right as usize]);
                ensure!(piece.len() <= 256, "BPE piece too long");
                pieces.push(piece);
            }
        }
        let ranks = spec
            .as_ref()
            .map(|s| {
                s.merges
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (*p, 260 + i as u32))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            spec,
            pieces,
            ranks,
        })
    }
    pub fn read(path: Option<&Path>) -> Result<Self> {
        Self::new(
            path.map(|p| {
                fs::read(p)
                    .map_err(anyhow::Error::from)
                    .and_then(|b| Ok(serde_json::from_slice(&b)?))
            })
            .transpose()?,
        )
    }
    pub fn name(&self) -> &str {
        if self.spec.is_some() {
            "lexical-byte-bpe-v1"
        } else {
            "byte-v1-260"
        }
    }
    pub fn sha256(&self) -> Result<Option<String>> {
        Ok(self
            .spec
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()?
            .map(|b| pcode::sha256(&b)))
    }
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub fn piece(&self, id: u32) -> Result<&[u8]> {
        ensure!(
            id >= 4 && (id as usize) < self.pieces.len(),
            "invalid completion token"
        );
        Ok(&self.pieces[id as usize])
    }
    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        if self.spec.is_some() {
            segments(bytes)
                .into_iter()
                .flat_map(|mut ids| {
                    while let Some((token, pair)) = ids
                        .windows(2)
                        .filter_map(|p| self.ranks.get(&[p[0], p[1]]).map(|id| (*id, [p[0], p[1]])))
                        .min_by_key(|(id, _)| *id)
                    {
                        ids = merge(&ids, pair, token);
                    }
                    ids
                })
                .collect()
        } else {
            bytes.iter().map(|b| u32::from(*b) + 4).collect()
        }
    }
    pub fn pair(&self, prompt: &str, source: &str) -> (Vec<u32>, usize) {
        let mut ids = vec![data::BOS];
        ids.extend(self.encode(prompt.as_bytes()));
        ids.push(data::SEP);
        let start = ids.len();
        ids.extend(self.encode(source.as_bytes()));
        ids.push(data::EOS);
        (ids, start)
    }
}
pub fn train(corpus: &Path, vocab_size: usize, output: &Path) -> Result<serde_json::Value> {
    ensure!(
        (260..=4096).contains(&vocab_size) && !output.exists(),
        "invalid vocabulary/output"
    );
    crate::corpus::run(corpus, false)?;
    let bytes = fs::read(corpus.join("release/train.jsonl"))?;
    let mut words: BTreeMap<Vec<u32>, u64> = BTreeMap::new();
    for line in std::str::from_utf8(&bytes)?.lines() {
        let row: pcode::Pair = serde_json::from_str(line)?;
        for (text, weight) in [(row.prompt()?, 1), (row.source, 8)] {
            for word in segments(text.as_bytes()) {
                *words.entry(word).or_default() += weight;
            }
        }
    }
    let mut merges = Vec::new();
    while 260 + merges.len() < vocab_size {
        let mut counts: BTreeMap<[u32; 2], u64> = BTreeMap::new();
        for (word, weight) in &words {
            for pair in word.windows(2) {
                *counts.entry([pair[0], pair[1]]).or_default() += weight;
            }
        }
        // Highest frequency, then lexicographically smallest pair gives stable ties.
        let Some((pair, count)) = counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        else {
            break;
        };
        if count < 2 {
            break;
        }
        let token = 260 + merges.len() as u32;
        words = words
            .into_iter()
            .map(|(word, weight)| (merge(&word, pair, token), weight))
            .collect();
        merges.push(pair);
    }
    let codec = Codec::new(Some(Spec {
        algorithm: "lexical-byte-bpe-v1".into(),
        train_split_sha256: pcode::sha256(&bytes),
        source_weight: 8,
        merges,
    }))?;
    fs::create_dir_all(output)?;
    fs::write(
        output.join("tokenizer.json"),
        serde_json::to_vec_pretty(&codec.spec)?,
    )?;
    let stats = stats(corpus, &codec)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&stats)?,
    )?;
    Ok(stats)
}
pub fn stats(corpus: &Path, codec: &Codec) -> Result<serde_json::Value> {
    let bytes = fs::read(corpus.join("release/train.jsonl"))?;
    if let Some(s) = &codec.spec {
        ensure!(
            s.train_split_sha256 == pcode::sha256(&bytes),
            "tokenizer was learned on a different train split"
        );
    }
    let mut records = Vec::new();
    for line in std::str::from_utf8(&bytes)?.lines() {
        let row: pcode::Pair = serde_json::from_str(line)?;
        let prompt = row.prompt()?;
        let (ids, start) = codec.pair(&prompt, &row.source);
        records.push(serde_json::json!({"id":row.id,"sequence_length":ids.len()-1,"completion_tokens":ids.len()-start,"byte_sequence_length":prompt.len()+row.source.len()+2}));
    }
    Ok(
        serde_json::json!({"tokenizer":codec.name(),"tokenizer_sha256":codec.sha256()?,"vocab_size":codec.vocab_size(),"training_only":true,"source_weight":codec.spec.as_ref().map(|s|s.source_weight),"records":records}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reversible_bytes_specials_and_checkpoint_spec() -> Result<()> {
        let s = Spec {
            algorithm: "lexical-byte-bpe-v1".into(),
            train_split_sha256: "a".repeat(64),
            source_weight: 8,
            merges: vec![[b'u' as u32 + 4, b'n' as u32 + 4], [260, b's' as u32 + 4]],
        };
        let codec = Codec::new(Some(s.clone()))?;
        let raw: Vec<u8> = (0..=255).chain(*b"unsigned 123\n").collect();
        let ids = codec.encode(&raw);
        let decoded: Vec<u8> = ids
            .iter()
            .flat_map(|id| codec.piece(*id).unwrap().iter().copied())
            .collect();
        assert_eq!(decoded, raw);
        assert_eq!(codec.encode(b"unsigned")[0], 261);
        let restored = Codec::new(Some(serde_json::from_slice(&serde_json::to_vec(&s)?)?))?;
        assert_eq!(codec.sha256()?, restored.sha256()?);
        assert_eq!(Codec::new(None)?.pair("x", "A"), data::encode("x", "A"));
        let mut bad = s;
        bad.merges.push([262, 4]);
        assert!(Codec::new(Some(bad)).is_err());
        assert!(codec.piece(3).is_err());
        Ok(())
    }
}

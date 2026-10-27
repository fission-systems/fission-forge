//! Freeze a family-disjoint synthetic bootstrap corpus, retaining failed attempts.
use crate::pcode::{self, Pair};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    pair: Pair,
    metadata: Metadata,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    id: String,
    program_id: String,
    family: String,
    split: String,
    optimization: String,
    binary_path: String,
    address: u64,
    code_size: u64,
    machine_code_sha256: String,
    ast_sha256: String,
    shape_sha256: String,
    behavior_sha256: String,
    prompt_sha256: String,
    sequence_length: usize,
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn lines<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .with_context(|| format!("missing string {field}"))
}
fn bounded(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = Path::new(relative);
    ensure!(
        !p.as_os_str().is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe artifact path"
    );
    let path = root.join(p).canonicalize()?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "artifact escapes corpus root"
    );
    Ok(path)
}
fn digest(value: &Value) -> Result<String> {
    Ok(pcode::sha256(&serde_json::to_vec(value)?))
}
fn number(v: &Value, f: &str) -> Result<usize> {
    v[f].as_u64()
        .map(|n| n as usize)
        .with_context(|| format!("missing integer {f}"))
}
fn elf_code(data: &[u8], address: u64, size: u64) -> Result<&[u8]> {
    fn word(d: &[u8], o: usize, n: usize) -> Result<u64> {
        let mut b = [0; 8];
        b[..n].copy_from_slice(
            d.get(o..o.checked_add(n).context("ELF overflow")?)
                .context("truncated ELF")?,
        );
        Ok(u64::from_le_bytes(b))
    }
    ensure!(
        data.get(..6) == Some(b"\x7fELF\x02\x01") && word(data, 18, 2)? == 62 && size > 0,
        "expected ELF64 LE x86-64 symbol"
    );
    let table = word(data, 40, 8)?;
    let width = word(data, 58, 2)?;
    ensure!(width >= 64, "invalid ELF section width");
    let end = address.checked_add(size).context("symbol overflow")?;
    for i in 0..word(data, 60, 2)? {
        let o = table
            .checked_add(i.checked_mul(width).context("ELF overflow")?)
            .context("ELF overflow")? as usize;
        let start = word(data, o + 16, 8)?;
        let length = word(data, o + 32, 8)?;
        if word(data, o + 8, 8)? & 4 != 0
            && start <= address
            && end <= start.checked_add(length).context("section overflow")?
        {
            let begin = word(data, o + 24, 8)?
                .checked_add(address - start)
                .context("offset overflow")?;
            let finish = begin.checked_add(size).context("size overflow")?;
            return data
                .get(usize::try_from(begin)?..usize::try_from(finish)?)
                .context("truncated symbol");
        }
    }
    anyhow::bail!("symbol outside executable section")
}
// Union whole generator families on any shared fingerprint before selecting rows.
fn quarantine(
    assignments: &BTreeMap<String, String>,
    keys: &[(String, String)],
) -> BTreeSet<String> {
    let mut groups: BTreeMap<String, BTreeSet<String>> = assignments
        .keys()
        .map(|f| (f.clone(), BTreeSet::from([f.clone()])))
        .collect();
    let mut owners = BTreeMap::<&str, &str>::new();
    for (key, family) in keys {
        if let Some(other) = owners.insert(key, family) {
            let mut merged = groups[family].clone();
            merged.extend(groups[other].iter().cloned());
            for member in &merged {
                groups.insert(member.clone(), merged.clone());
            }
        }
    }
    assignments
        .keys()
        .filter(|f| {
            groups[*f]
                .iter()
                .map(|m| &assignments[m])
                .collect::<BTreeSet<_>>()
                .len()
                > 1
        })
        .cloned()
        .collect()
}
#[derive(Serialize)]
struct Exclusion {
    id: String,
    reason: String,
}

pub fn run(root: &Path, finalize: bool) -> Result<Value> {
    let root = root.canonicalize()?;
    let manifest = read(&root.join("manifest.json"))?;
    ensure!(
        manifest["schema_version"] == 1
            && manifest["origin"] == "owned_synthetic_generator"
            && manifest["split_unit"] == "generator_family"
            && manifest["benchmark_sources_used"] == false
            && manifest["external_evaluation_overlap"] == "not_measured",
        "unsupported corpus contract"
    );
    let artifacts = manifest["artifacts"]
        .as_object()
        .context("missing artifact hashes")?;
    for required in [
        "config.json",
        "profile.json",
        "programs.jsonl",
        "probes.json",
        "native-verification.json",
        "pool.jsonl",
        "attempts.jsonl",
    ] {
        ensure!(
            artifacts.contains_key(required),
            "unbound required artifact {required}"
        );
    }
    for (path, hash) in artifacts {
        ensure!(
            pcode::sha256(&fs::read(bounded(&root, path)?)?)
                == hash.as_str().context("invalid artifact digest")?,
            "artifact hash mismatch: {path}"
        );
    }
    let cfg = read(&root.join("config.json"))?;
    ensure!(
        cfg["schema_version"] == 1
            && [
                pcode::cfg::REPRESENTATION,
                pcode::cfg::slice::REPRESENTATION
            ]
            .contains(&cfg["representation"].as_str().unwrap_or(""))
            && cfg["representation"] == manifest["representation"],
        "invalid corpus representation"
    );
    ensure!(
        artifacts["config.json"] == manifest["config_sha256"]
            && artifacts["profile.json"] == manifest["profile_sha256"],
        "configuration digest mismatch"
    );
    let profile: pcode::Profile = serde_json::from_value(read(&root.join("profile.json"))?)?;
    let profile_value = serde_json::to_value(&profile)?;
    let assignments: BTreeMap<String, String> = serde_json::from_value(cfg["families"].clone())?;
    ensure!(
        assignments
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == BTreeSet::from(["train", "validation", "test"]),
        "three explicit splits required"
    );
    let variants = number(&cfg, "variants_per_family")?;
    ensure!((1..=256).contains(&variants), "invalid family size");
    let extra: BTreeMap<String, usize> = match cfg.get("extra_variants") {
        Some(v) => serde_json::from_value(v.clone())?,
        None => BTreeMap::new(),
    };
    ensure!(
        extra.iter().all(|(family, n)| {
            assignments.get(family).map(String::as_str) == Some("train")
                && (1..=256 - variants).contains(n)
        }),
        "extra variants are limited to explicit train families"
    );
    let opts: Vec<String> = serde_json::from_value(cfg["optimizations"].clone())?;
    ensure!(
        !opts.is_empty()
            && opts.iter().collect::<BTreeSet<_>>().len() == opts.len()
            && opts
                .iter()
                .all(|o| ["O0", "O1", "O2", "O3"].contains(&o.as_str())),
        "invalid optimization list"
    );
    let programs: Vec<Value> = lines(&root.join("programs.jsonl"))?;
    let probes = read(&root.join("probes.json"))?;
    let probe_count = probes.as_array().context("invalid probes")?.len();
    ensure!(probe_count > 0, "empty probes");
    let mut by_id = BTreeMap::new();
    let mut keys = Vec::new();
    for p in &programs {
        let id = string(p, "id")?;
        let family = string(p, "family")?;
        ensure!(
            assignments.get(family).map(String::as_str) == Some(string(p, "split")?),
            "source split assignment mismatch"
        );
        ensure!(by_id.insert(id, p).is_none(), "duplicate program id");
        ensure!(
            p["oracle_outputs"]
                .as_array()
                .context("missing oracle outputs")?
                .len()
                == probe_count,
            "oracle length mismatch"
        );
        for (value, hash) in [
            ("canonical_ast", "ast_sha256"),
            ("shape_ast", "shape_sha256"),
            ("oracle_outputs", "behavior_sha256"),
        ] {
            ensure!(
                !p[value].is_null() && digest(&p[value])? == string(p, hash)?,
                "program fingerprint mismatch"
            );
            keys.push((format!("{hash}:{}", string(p, hash)?), family.to_owned()));
        }
        keys.push((
            format!("source:{}", pcode::sha256(string(p, "source")?.as_bytes())),
            family.to_owned(),
        ));
    }
    for family in assignments.keys() {
        ensure!(
            programs.iter().filter(|p| p["family"] == *family).count()
                == variants + extra.get(family).copied().unwrap_or(0),
            "family program count mismatch"
        );
    }
    let native = read(&root.join("native-verification.json"))?;
    ensure!(
        native["status"] == "passed"
            && number(&native, "programs")? == programs.len()
            && number(&native, "probes_per_program")? == probe_count
            && number(&native, "comparisons")? == programs.len() * probe_count,
        "source oracle verification incomplete"
    );
    let attempts: Vec<Value> = lines(&root.join("attempts.jsonl"))?;
    let mut planned = BTreeMap::new();
    for p in &programs {
        for opt in &opts {
            planned.insert(format!("{}-{opt}", string(p, "id")?), (p, opt));
        }
    }
    let mut supported = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut failures = BTreeMap::<String, usize>::new();
    for a in &attempts {
        let id = string(a, "id")?;
        let (p, opt) = planned.get(id).context("unexpected attempt")?;
        ensure!(seen.insert(id), "duplicate attempt");
        ensure!(
            a["program_id"] == p["id"]
                && a["family"] == p["family"]
                && a["split"] == p["split"]
                && a["optimization"] == **opt,
            "attempt assignment mismatch"
        );
        match string(a, "status")? {
            "supported" => {
                supported.insert(id.to_owned());
            }
            "unsupported" => {
                ensure!(
                    !string(a, "diagnostic")?.is_empty(),
                    "missing failure diagnostic"
                );
                *failures.entry(string(a, "split")?.to_owned()).or_default() += 1;
            }
            _ => anyhow::bail!("invalid attempt status"),
        }
    }
    ensure!(seen.len() == planned.len(), "missing attempted variants");
    let mut rows: Vec<Row> = lines(&root.join("pool.jsonl"))?;
    rows.sort_by(|a, b| a.metadata.id.cmp(&b.metadata.id));
    let mut row_ids = BTreeSet::new();
    let mut prompt_labels = BTreeMap::<String, BTreeSet<String>>::new();
    for row in &rows {
        let m = &row.metadata;
        let p = by_id
            .get(m.program_id.as_str())
            .context("unknown row program")?;
        ensure!(
            row_ids.insert(m.id.clone()) && supported.contains(&m.id),
            "unexpected/duplicate supported row"
        );
        ensure!(
            m.id == row.pair.id
                && m.id == format!("{}-{}", m.program_id, m.optimization)
                && opts.contains(&m.optimization)
                && p["family"] == m.family
                && p["split"] == m.split,
            "row assignment mismatch"
        );
        ensure!(
            row.pair.source == string(p, "source")?
                && serde_json::to_value(&row.pair.input.profile)? == profile_value
                && row.pair.input.representation == cfg["representation"].as_str().unwrap_or(""),
            "row label/profile mismatch"
        );
        for (actual, field) in [
            (&m.ast_sha256, "ast_sha256"),
            (&m.shape_sha256, "shape_sha256"),
            (&m.behavior_sha256, "behavior_sha256"),
        ] {
            ensure!(actual == string(p, field)?, "row fingerprint mismatch");
        }
        let prompt = row.pair.prompt()?;
        ensure!(
            pcode::sha256(prompt.as_bytes()) == m.prompt_sha256
                && prompt.len() + row.pair.source.len() + 2 == m.sequence_length,
            "prompt digest/length mismatch"
        );
        ensure!(artifacts.contains_key(&m.binary_path), "unbound row binary");
        let binary = fs::read(bounded(&root, &m.binary_path)?)?;
        ensure!(
            pcode::sha256(&binary) == row.pair.provenance.binary_sha256
                && pcode::sha256(elf_code(&binary, m.address, m.code_size)?)
                    == m.machine_code_sha256,
            "binary/symbol digest mismatch"
        );
        ensure!(
            u64::from_str_radix(
                row.pair.input.raw.entry_address.trim_start_matches("0x"),
                16
            )? == m.address,
            "raw entry mismatch"
        );
        keys.push((
            format!("machine:{}", m.machine_code_sha256),
            m.family.clone(),
        ));
        keys.push((format!("prompt:{}", m.prompt_sha256), m.family.clone()));
        prompt_labels
            .entry(m.prompt_sha256.clone())
            .or_default()
            .insert(row.pair.provenance.source_sha256.clone());
    }
    ensure!(row_ids == supported, "supported attempts/pool mismatch");
    let quarantined = quarantine(&assignments, &keys);
    let mut exclusions = Vec::new();
    let mut duplicates = BTreeMap::new();
    let mut splits: BTreeMap<&str, Vec<&Row>> = ["train", "validation", "test"]
        .into_iter()
        .map(|s| (s, Vec::new()))
        .collect();
    for row in &rows {
        let m = &row.metadata;
        let reason = if quarantined.contains(&m.family) {
            Some("cross_split_fingerprint_family".to_owned())
        } else if prompt_labels[&m.prompt_sha256].len() > 1 {
            Some("ambiguous_prompt_labels".to_owned())
        } else {
            let key = (
                m.split.clone(),
                m.prompt_sha256.clone(),
                row.pair.provenance.source_sha256.clone(),
            );
            if let Some(first) = duplicates.get(&key) {
                Some(format!("duplicate_of:{first}"))
            } else {
                duplicates.insert(key, m.id.clone());
                None
            }
        };
        if let Some(reason) = reason {
            exclusions.push(Exclusion {
                id: m.id.clone(),
                reason,
            });
        } else {
            splits
                .get_mut(m.split.as_str())
                .context("invalid split")?
                .push(row);
        }
    }
    ensure!(
        splits.values().all(|r| !r.is_empty()),
        "leakage filtering leaves an empty split; release refused"
    );
    let mut statistics = BTreeMap::new();
    let mut release = BTreeMap::<String, Vec<u8>>::new();
    for (split, items) in &splits {
        let mut bytes = Vec::new();
        for row in items {
            serde_json::to_writer(&mut bytes, &row.pair)?;
            bytes.push(b'\n');
        }
        release.insert(format!("{split}.jsonl"), bytes);
        let eligible: BTreeMap<String, usize> = [1024, 2048, 4096, 8192, 16384]
            .into_iter()
            .map(|c| {
                (
                    c.to_string(),
                    items
                        .iter()
                        .filter(|r| r.metadata.sequence_length <= c)
                        .count(),
                )
            })
            .collect();
        statistics.insert(*split,json!({"rows":items.len(),"programs":items.iter().map(|r|&r.metadata.program_id).collect::<BTreeSet<_>>().len(),"families":items.iter().map(|r|&r.metadata.family).collect::<BTreeSet<_>>(),"context_eligible":eligible,"min_sequence":items.iter().map(|r|r.metadata.sequence_length).min(),"max_sequence":items.iter().map(|r|r.metadata.sequence_length).max()}));
    }
    let report = json!({"schema_version":1,"scope":"owned_synthetic_family_disjoint_not_real_project_generalization","attempted":attempts.len(),"supported":rows.len(),"unsupported":attempts.len()-rows.len(),"unsupported_by_split":failures,"quarantined_families":quarantined,"exclusions":exclusions,"splits":statistics,"native_source_comparisons":programs.len()*probe_count,"x86_ir_execution_parity":"not_measured","external_evaluation_overlap":"not_measured","truncation":false});
    release.insert("audit.json".into(), serde_json::to_vec_pretty(&report)?);
    let hashes: BTreeMap<String, String> = release
        .iter()
        .map(|(name, b)| (name.clone(), pcode::sha256(b)))
        .collect();
    let lock = json!({"schema_version":1,"manifest_sha256":pcode::sha256(&fs::read(root.join("manifest.json"))?),"files":hashes});
    release.insert("lock.json".into(), serde_json::to_vec_pretty(&lock)?);
    let destination = root.join("release");
    if finalize {
        ensure!(!destination.exists(), "release already exists");
        fs::create_dir(&destination)?;
        for (name, bytes) in release {
            fs::write(destination.join(name), bytes)?;
        }
    } else {
        for (name, bytes) in release {
            ensure!(
                fs::read(bounded(&root, &format!("release/{name}"))?)? == bytes,
                "frozen release mismatch: {name}"
            );
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transitive_clone_quarantines_whole_families() {
        let a = [
            ("a", "train"),
            ("b", "train"),
            ("c", "test"),
            ("d", "validation"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
        let keys = [
            ("source:x", "a"),
            ("source:x", "b"),
            ("machine:y", "b"),
            ("machine:y", "c"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect::<Vec<_>>();
        assert_eq!(
            quarantine(&a, &keys),
            BTreeSet::from(["a".into(), "b".into(), "c".into()])
        );
    }
    #[test]
    fn elf_bounds_fail_closed() {
        assert!(elf_code(b"truncated", 0, 10).is_err());
    }
}

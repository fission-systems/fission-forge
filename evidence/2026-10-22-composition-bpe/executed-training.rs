//! Bounded fixed-batch training probe with serializable AdamW state.
//! No dropout or stochastic sampler: resume has no hidden RNG/sampler state.
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    data,
    model::{self, Decoder, ModelConfig},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdamConfig {
    pub(crate) lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}
impl AdamConfig {
    pub(crate) fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

struct Slot {
    name: String,
    weight: Var,
    m: Tensor,
    v: Tensor,
}
pub(crate) struct Adam {
    slots: Vec<Slot>,
    pub(crate) config: AdamConfig,
    pub(crate) step: u64,
}
impl Adam {
    pub(crate) fn new(vars: &VarMap, config: AdamConfig) -> Result<Self> {
        ensure!(
            config.lr.is_finite()
                && config.lr > 0.
                && (0. ..1.).contains(&config.beta1)
                && (0. ..1.).contains(&config.beta2)
                && config.eps.is_finite()
                && config.eps > 0.
                && config.weight_decay.is_finite()
                && config.weight_decay >= 0.,
            "invalid AdamW settings"
        );
        let slots = model::named_vars(vars)
            .into_iter()
            .map(|(name, weight)| {
                Ok(Slot {
                    m: weight.zeros_like()?,
                    v: weight.zeros_like()?,
                    name,
                    weight,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            slots,
            config,
            step: 0,
        })
    }
    pub(crate) fn update(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step += 1;
        let c = &self.config;
        let scale_m = 1. / (1. - c.beta1.powf(self.step as f64));
        let scale_v = 1. / (1. - c.beta2.powf(self.step as f64));
        for slot in &mut self.slots {
            let g = grads
                .get(&slot.weight)
                .with_context(|| format!("missing gradient: {}", slot.name))?;
            slot.m = ((&slot.m * c.beta1)? + (g * (1. - c.beta1))?)?.detach();
            slot.v = ((&slot.v * c.beta2)? + (g.sqr()? * (1. - c.beta2))?)?.detach();
            let direction = ((&slot.m * scale_m)? / ((&slot.v * scale_v)?.sqrt()? + c.eps)?)?;
            let next =
                ((slot.weight.as_tensor() * (1. - c.lr * c.weight_decay))? - (direction * c.lr)?)?;
            slot.weight.set(&next.detach())?;
        }
        Ok(())
    }
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        for slot in &self.slots {
            tensors.insert(
                format!("model.{}", slot.name),
                slot.weight.as_tensor().clone(),
            );
            tensors.insert(format!("adam.m.{}", slot.name), slot.m.clone());
            tensors.insert(format!("adam.v.{}", slot.name), slot.v.clone());
        }
        Ok(candle_core::safetensors::save(&tensors, path)?)
    }
    pub(crate) fn load(&mut self, path: &Path, step: u64, device: &Device) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, device)?;
        ensure!(
            tensors.len() == self.slots.len() * 3,
            "unexpected checkpoint tensor count"
        );
        for slot in &mut self.slots {
            let fetch = |prefix: &str| -> Result<Tensor> {
                let key = format!("{prefix}.{}", slot.name);
                let t = tensors
                    .get(&key)
                    .with_context(|| format!("checkpoint missing {key}"))?;
                ensure!(
                    t.shape() == slot.weight.shape() && t.dtype() == slot.weight.dtype(),
                    "shape/dtype mismatch for {key}"
                );
                Ok(t.clone())
            };
            let (weight, m, v) = (fetch("model")?, fetch("adam.m")?, fetch("adam.v")?);
            slot.weight.set(&weight)?;
            slot.m = m;
            slot.v = v;
        }
        self.step = step;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Checkpoint {
    pub(crate) schema_version: u32,
    pub(crate) model: ModelConfig,
    pub(crate) optimizer: AdamConfig,
    pub(crate) step: u64,
    pub(crate) seed: u64,
    pub(crate) dataset_sha256: String,
    pub(crate) tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer_spec: Option<crate::tokenizer::Spec>,
    #[serde(default)]
    pub(crate) input_sha256: Option<String>,
    #[serde(default)]
    pub(crate) representations: Vec<String>,
    #[serde(default)]
    pub(crate) corpus: Option<crate::corpus_training::ResumeState>,
    #[serde(default)]
    pub(crate) checkpoint_sha256: Option<String>,
}

pub struct Options {
    pub steps: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub resume: Option<PathBuf>,
}

#[derive(Serialize)]
pub struct TrainReport {
    pub evidence_scope: &'static str,
    pub backend: String,
    pub dtype: &'static str,
    pub model: ModelConfig,
    pub parameters: usize,
    pub initialization: &'static str,
    pub dataset_sha256: String,
    pub input_sha256: String,
    pub representations: Vec<String>,
    pub examples: usize,
    pub sequence_length: usize,
    pub initial_step: u64,
    pub completed_steps: u64,
    pub measured_steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub training_seconds: f64,
    pub padded_tokens_per_second: f64,
    pub completion_tokens_per_second: f64,
    pub checkpoint_logit_max_error: f32,
    pub resumed_adamw_step_max_error: f32,
}

fn max_error(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

pub fn run(
    config: &Path,
    dataset: &Path,
    output: &Path,
    device: &Device,
    options: Options,
) -> Result<TrainReport> {
    ensure!(
        options.steps > 0 && options.seed != 0,
        "steps and seed must be positive"
    );
    ensure!(
        !output.exists(),
        "output already exists: {}",
        output.display()
    );
    let config: ModelConfig = serde_json::from_slice(&fs::read(config)?)?;
    config.validate()?;
    ensure!(
        config.vocab_size == 260,
        "smoke trainer requires byte codec"
    );
    let dataset_bytes = fs::read(dataset)?;
    let digest = format!("{:x}", Sha256::digest(&dataset_bytes));
    let batch = data::batch(
        std::str::from_utf8(&dataset_bytes)?,
        config.max_seq_len,
        device,
    )?;
    let vars = VarMap::new();
    let model = Decoder::new(&config, &vars, device)?;
    let mut optimizer = Adam::new(&vars, AdamConfig::new(options.learning_rate))?;
    if let Some(resume) = &options.resume {
        let state: Checkpoint = serde_json::from_slice(&fs::read(resume.join("state.json"))?)?;
        ensure!(
            (((state.schema_version == 2
                || (state.schema_version == 3 && state.representations == batch.representations))
                && state.input_sha256.as_deref() == Some(&batch.input_sha256))
                || (state.schema_version == 1 && batch.representations == ["assembly-v0"]))
                && state.model == config
                && state.dataset_sha256 == digest
                && state.tokenizer == "byte-v1-260"
                && state.seed == options.seed,
            "checkpoint provenance/configuration mismatch"
        );
        ensure!(
            state.optimizer.lr == options.learning_rate,
            "resume learning rate differs"
        );
        optimizer = Adam::new(&vars, state.optimizer)?;
        optimizer.load(&resume.join("training.safetensors"), state.step, device)?;
    } else {
        model::initialize(&vars, options.seed)?;
    }
    let parameters = model::named_vars(&vars)
        .iter()
        .map(|(_, v)| v.elem_count())
        .sum();
    let initial_step = optimizer.step;
    let initial_loss =
        data::completion_loss(&model.forward(&batch.input)?, &batch)?.to_scalar::<f32>()?;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    let timer = Instant::now();
    for i in 0..options.steps {
        let loss = data::completion_loss(&model.forward(&batch.input)?, &batch)?;
        let value = loss.to_scalar::<f32>()?; // Synchronizes device before reporting.
        ensure!(
            value.is_finite(),
            "nonfinite loss at step {}",
            optimizer.step
        );
        optimizer.update(&loss)?;
        if i == 0 || (i + 1) % 10 == 0 {
            eprintln!("step={} loss={value:.6}", optimizer.step);
        }
    }
    let logits = model.forward(&batch.input)?;
    let final_loss = data::completion_loss(&logits, &batch)?.to_scalar::<f32>()?;
    let training_seconds = timer.elapsed().as_secs_f64();
    ensure!(final_loss.is_finite(), "nonfinite final loss");
    let completed_steps = optimizer.step;
    let state = Checkpoint {
        schema_version: 3,
        corpus: None,
        checkpoint_sha256: None,
        representations: batch.representations.clone(),
        input_sha256: Some(batch.input_sha256.clone()),
        model: config.clone(),
        optimizer: optimizer.config.clone(),
        step: completed_steps,
        seed: options.seed,
        dataset_sha256: digest.clone(),
        tokenizer: "byte-v1-260".into(),
        tokenizer_spec: None,
    };
    optimizer.save(&output.join("training.safetensors"))?;
    fs::write(
        output.join("state.json"),
        serde_json::to_vec_pretty(&state)?,
    )?;
    let restored_vars = VarMap::new();
    let restored = Decoder::new(&config, &restored_vars, device)?;
    let mut restored_optimizer = Adam::new(&restored_vars, optimizer.config.clone())?;
    restored_optimizer.load(
        &output.join("training.safetensors"),
        completed_steps,
        device,
    )?;
    let checkpoint_logit_max_error = max_error(&logits, &restored.forward(&batch.input)?)?;
    ensure!(
        checkpoint_logit_max_error <= 1e-5,
        "checkpoint logits changed: {checkpoint_logit_max_error}"
    );
    optimizer.update(&data::completion_loss(
        &model.forward(&batch.input)?,
        &batch,
    )?)?;
    restored_optimizer.update(&data::completion_loss(
        &restored.forward(&batch.input)?,
        &batch,
    )?)?;
    let mut resumed_adamw_step_max_error = 0f32;
    for (a, b) in optimizer.slots.iter().zip(&restored_optimizer.slots) {
        for (x, y) in [
            (a.weight.as_tensor(), b.weight.as_tensor()),
            (&a.m, &b.m),
            (&a.v, &b.v),
        ] {
            resumed_adamw_step_max_error = resumed_adamw_step_max_error.max(max_error(x, y)?);
        }
    }
    ensure!(
        resumed_adamw_step_max_error <= 1e-5,
        "resumed AdamW step diverged: {resumed_adamw_step_max_error}"
    );
    let report = TrainReport {
        evidence_scope: "fixed_batch_transformer_training_probe_only",
        backend: format!("{:?}", device.location()),
        dtype: "f32",
        model: config,
        parameters,
        initialization: if options.resume.is_some() {
            "resumed_checkpoint"
        } else {
            "seeded_from_scratch"
        },
        dataset_sha256: digest,
        input_sha256: batch.input_sha256.clone(),
        representations: batch.representations.clone(),
        examples: batch.examples,
        sequence_length: batch.sequence_length,
        initial_step,
        completed_steps,
        measured_steps: options.steps,
        initial_loss,
        final_loss,
        training_seconds,
        padded_tokens_per_second: (options.steps * batch.examples * batch.sequence_length) as f64
            / training_seconds,
        completion_tokens_per_second: (options.steps * batch.completion_tokens) as f64
            / training_seconds,
        checkpoint_logit_max_error,
        resumed_adamw_step_max_error,
    };
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Optimizer, ParamsAdamW, VarBuilder};
    #[test]
    fn adamw_matches_candle_reference_over_multiple_steps() -> Result<()> {
        let a = VarMap::new();
        let b = VarMap::new();
        let make = |v: &VarMap| {
            VarBuilder::from_varmap(v, candle_core::DType::F32, &Device::Cpu).get_with_hints(
                3,
                "w",
                candle_nn::Init::Const(0.7),
            )
        };
        let wa = make(&a)?;
        let wb = make(&b)?;
        let mut ours = Adam::new(&a, AdamConfig::new(0.003))?;
        let mut reference = candle_nn::AdamW::new(
            b.all_vars(),
            ParamsAdamW {
                lr: 0.003,
                ..Default::default()
            },
        )?;
        for _ in 0..4 {
            ours.update(&wa.sqr()?.sum_all()?)?;
            reference.backward_step(&wb.sqr()?.sum_all()?)?;
            assert!(max_error(&wa, &wb)? < 1e-6);
        }
        Ok(())
    }
}

#[cfg(test)]
mod ir_checkpoint_tests {
    use super::*;

    #[test]
    fn ir_resume_matches_uninterrupted_and_rejects_changed_input_fingerprint() -> Result<()> {
        struct Temporary(PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = Temporary(
            std::env::temp_dir().join(format!("forge-ir-resume-{}-{nonce}", std::process::id())),
        );
        fs::create_dir(&root.0)?;
        let config = root.0.join("config.json");
        fs::write(
            &config,
            r#"{"vocab_size":260,"hidden_size":8,"heads":2,"layers":1,"feed_forward_size":16,"max_seq_len":512}"#,
        )?;
        let dataset = root.0.join("pairs.jsonl");
        fs::write(
            &dataset,
            include_str!("../../../fixtures/pcode/pairs.jsonl")
                .lines()
                .next()
                .unwrap(),
        )?;
        let first = root.0.join("first");
        let resumed = root.0.join("resumed");
        let full = root.0.join("full");
        let options = |steps, resume| Options {
            steps,
            learning_rate: 0.003,
            seed: 42,
            resume,
        };
        run(&config, &dataset, &first, &Device::Cpu, options(2, None))?;
        run(
            &config,
            &dataset,
            &resumed,
            &Device::Cpu,
            options(1, Some(first.clone())),
        )?;
        run(&config, &dataset, &full, &Device::Cpu, options(3, None))?;
        assert_eq!(
            fs::read(resumed.join("training.safetensors"))?,
            fs::read(full.join("training.safetensors"))?
        );
        let mut state: Checkpoint = serde_json::from_slice(&fs::read(first.join("state.json"))?)?;
        state.schema_version = 2;
        state.representations.clear();
        fs::write(first.join("state.json"), serde_json::to_vec(&state)?)?;
        let schema2 = root.0.join("schema2-resume");
        run(
            &config,
            &dataset,
            &schema2,
            &Device::Cpu,
            options(1, Some(first.clone())),
        )?;
        assert_eq!(
            fs::read(schema2.join("training.safetensors"))?,
            fs::read(full.join("training.safetensors"))?
        );
        state.input_sha256 = Some("0".repeat(64));
        fs::write(first.join("state.json"), serde_json::to_vec(&state)?)?;
        let rejected = root.0.join("rejected");
        let error = run(
            &config,
            &dataset,
            &rejected,
            &Device::Cpu,
            options(1, Some(first.clone())),
        )
        .err()
        .context("expected fingerprint rejection")?;
        assert!(
            error
                .to_string()
                .contains("provenance/configuration mismatch")
        );
        assert!(!rejected.exists());
        state.schema_version = 1;
        state.input_sha256 = None;
        fs::write(first.join("state.json"), serde_json::to_vec(&state)?)?;
        assert!(
            run(
                &config,
                &dataset,
                &rejected,
                &Device::Cpu,
                options(1, Some(first))
            )
            .is_err()
        );
        assert!(!rejected.exists());
        Ok(())
    }
}

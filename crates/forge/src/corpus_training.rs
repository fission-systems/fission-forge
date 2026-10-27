//! Train-only epoch sampling, resumable AdamW, and token-weighted held-out NLL.
use crate::{
    corpus, data,
    model::{self, Decoder, ModelConfig},
    pcode,
    training::{Adam, AdamConfig, Checkpoint},
};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub batch_size: usize,
    pub attention_chunk: usize,
    /// Zero evaluates every validation row; otherwise the fixed file-order prefix.
    pub validation_limit: usize,
    pub evaluate_every: u64,
    pub checkpoint_every: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Sampler {
    pub seed: u64,
    pub epoch: u64,
    pub cursor: usize,
}
impl Sampler {
    fn order(&self, count: usize) -> Vec<usize> {
        let mut result: Vec<_> = (0..count).collect();
        let digest =
            pcode::sha256(format!("forge-fisher-yates-v1:{}:{}", self.seed, self.epoch).as_bytes());
        let mut state = u64::from_str_radix(&digest[..16], 16).unwrap().max(1);
        for i in (1..count).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state % (i as u64 + 1)) as usize;
            result.swap(i, j);
        }
        result
    }
    fn next(&mut self, count: usize) -> Result<usize> {
        ensure!(
            count > 0 && self.cursor <= count,
            "invalid sampler position"
        );
        if self.cursor == count {
            self.epoch = self.epoch.checked_add(1).context("epoch overflow")?;
            self.cursor = 0;
        }
        let index = self.order(count)[self.cursor];
        self.cursor += 1;
        Ok(index)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Best {
    pub step: u64,
    pub nll: f64,
    pub checkpoint: Option<PathBuf>,
    pub weights_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeState {
    pub schema_version: u32,
    pub sampler_algorithm: String,
    pub sampler: Sampler,
    pub settings: Settings,
    pub corpus_lock_sha256: String,
    pub train_encoded_sha256: String,
    pub validation_encoded_sha256: String,
    pub best: Option<Best>,
}
pub struct Options {
    pub steps: usize,
    pub learning_rate: f64,
    pub override_learning_rate: bool,
    pub seed: u64,
    pub resume: Option<PathBuf>,
    pub settings: Settings,
    pub tokenizer: Option<PathBuf>,
}
struct Example {
    id: String,
    tokens: Vec<u32>,
    start: usize,
}
struct Dataset {
    rows: Vec<Example>,
    representation: String,
    effective_sha256: String,
}
impl Dataset {
    fn load(path: &Path, context: usize, codec: &crate::tokenizer::Codec) -> Result<Self> {
        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        let mut representation = None;
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            ensure!(!line.trim().is_empty(), "blank corpus row");
            let pair: pcode::Pair = serde_json::from_str(&line)?;
            ensure!(
                [
                    pcode::cfg::REPRESENTATION,
                    pcode::cfg::slice::REPRESENTATION,
                    pcode::cfg::slice::COMPACT,
                    pcode::cfg::slice::COMPACT_DECIMAL
                ]
                .contains(&pair.input.representation.as_str())
                    && seen.insert(pair.id.clone()),
                "invalid corpus lane or duplicate ID"
            );
            ensure!(
                representation
                    .as_ref()
                    .is_none_or(|r| r == &pair.input.representation),
                "mixed corpus representations"
            );
            representation = Some(pair.input.representation.clone());
            let prompt = pair.prompt()?;
            ensure!(
                !prompt.is_empty() && !pair.source.is_empty(),
                "empty corpus example"
            );
            let (tokens, start) = codec.pair(&prompt, &pair.source);
            ensure!(
                tokens.len() - 1 <= context,
                "corpus row {} exceeds context; truncation is forbidden",
                pair.id
            );
            rows.push(Example {
                id: pair.id,
                tokens,
                start,
            });
        }
        ensure!(!rows.is_empty(), "empty corpus split");
        let fingerprints: Vec<_> = rows.iter().map(|e| (&e.id, &e.tokens, e.start)).collect();
        let effective_sha256 = pcode::sha256(&serde_json::to_vec(&fingerprints)?);
        Ok(Self {
            rows,
            representation: representation.context("missing corpus representation")?,
            effective_sha256,
        })
    }
    fn batch(&self, indices: &[usize], context: usize, device: &Device) -> Result<data::Batch> {
        let encoded = indices
            .iter()
            .map(|i| (self.rows[*i].tokens.clone(), self.rows[*i].start))
            .collect::<Vec<_>>();
        data::encoded_batch(&encoded, vec![self.representation.clone()], context, device)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Validation {
    pub step: u64,
    pub examples: usize,
    pub total_examples: usize,
    pub completion_tokens: usize,
    pub nll: f64,
    pub perplexity: f64,
    pub selection: String,
    pub selected_ids: Vec<String>,
}
fn evaluate(
    vars: &VarMap,
    config: &ModelConfig,
    validation: &Dataset,
    settings: &Settings,
    step: u64,
    device: &Device,
) -> Result<Validation> {
    // Detached snapshots cannot populate gradients or update the optimizer/sampler.
    let model = Decoder::for_eval(config, vars, device)?;
    let count = if settings.validation_limit == 0 {
        validation.rows.len()
    } else {
        settings.validation_limit.min(validation.rows.len())
    };
    let mut weighted = 0f64;
    let mut tokens = 0;
    // One example at a time avoids validation batch padding and long-row spikes.
    for i in 0..count {
        let batch = validation.batch(&[i], config.max_seq_len, device)?;
        let loss = data::completion_loss(
            &model.forward_chunked(&batch.input, settings.attention_chunk)?,
            &batch,
        )?
        .to_scalar::<f32>()?;
        ensure!(loss.is_finite(), "nonfinite validation loss");
        weighted += f64::from(loss) * batch.completion_tokens as f64;
        tokens += batch.completion_tokens;
    }
    let nll = weighted / tokens as f64;
    let perplexity = nll.exp();
    ensure!(perplexity.is_finite(), "nonfinite validation perplexity");
    Ok(Validation {
        step,
        examples: count,
        total_examples: validation.rows.len(),
        completion_tokens: tokens,
        nll,
        perplexity,
        selection: if count == validation.rows.len() {
            "full_split"
        } else {
            "fixed_file_order_prefix_probe"
        }
        .into(),
        selected_ids: validation.rows[..count]
            .iter()
            .map(|e| e.id.clone())
            .collect(),
    })
}
fn append(log: &mut File, event: &Value) -> Result<()> {
    serde_json::to_writer(&mut *log, event)?;
    writeln!(log)?;
    log.flush()?;
    Ok(())
}
fn snapshot(
    output: &Path,
    optimizer: &Adam,
    architecture: (&ModelConfig, &crate::tokenizer::Codec, &str),
    seed: u64,
    train_sha: &str,
    state: &mut ResumeState,
    validation: Option<&Validation>,
) -> Result<PathBuf> {
    let (config, codec, representation) = architecture;
    let name = format!("step-{:08}", optimizer.step);
    let destination = output.join("checkpoints").join(&name);
    let staging = output.join(format!(".{name}.partial"));
    fs::create_dir(&staging)?;
    optimizer.save(&staging.join("training.safetensors"))?;
    let weights_sha256 = pcode::sha256(&fs::read(staging.join("training.safetensors"))?);
    if let Some(v) = validation
        && state.best.as_ref().is_none_or(|b| v.nll < b.nll)
    {
        state.best = Some(Best {
            step: optimizer.step,
            nll: v.nll,
            checkpoint: Some(destination.clone()),
            weights_sha256: weights_sha256.clone(),
        });
    }
    let checkpoint = Checkpoint {
        schema_version: if codec.spec.is_some() { 5 } else { 4 },
        model: config.clone(),
        optimizer: optimizer.config.clone(),
        step: optimizer.step,
        seed,
        dataset_sha256: train_sha.into(),
        tokenizer: codec.name().into(),
        tokenizer_spec: codec.spec.clone(),
        input_sha256: Some(state.train_encoded_sha256.clone()),
        representations: vec![representation.into()],
        corpus: Some(state.clone()),
        checkpoint_sha256: Some(weights_sha256),
    };
    fs::write(
        staging.join("state.json"),
        serde_json::to_vec_pretty(&checkpoint)?,
    )?;
    // Rename only complete snapshots; partial directories are never resume targets.
    ensure!(!destination.exists(), "checkpoint already exists");
    fs::rename(staging, &destination)?;
    Ok(destination)
}

pub fn run(
    config_path: &Path,
    root: &Path,
    output: &Path,
    device: &Device,
    options: Options,
) -> Result<Value> {
    ensure!(
        options.steps > 0
            && options.seed > 0
            && options.learning_rate.is_finite()
            && options.learning_rate > 0.,
        "invalid training options"
    );
    ensure!(
        !options.override_learning_rate || options.resume.is_some(),
        "learning-rate override requires resume"
    );
    let settings = &options.settings;
    ensure!(
        (1..=32).contains(&settings.batch_size)
            && settings.attention_chunk > 0
            && settings.evaluate_every > 0
            && settings.checkpoint_every > 0,
        "invalid corpus settings"
    );
    ensure!(!output.exists(), "output already exists");
    let audit = corpus::run(root, false).context("corpus release audit failed")?;
    let config: ModelConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    config.validate()?;
    let codec = crate::tokenizer::Codec::read(options.tokenizer.as_deref())?;
    ensure!(
        config.vocab_size == codec.vocab_size(),
        "model/tokenizer vocabulary mismatch"
    );
    if let Some(spec) = &codec.spec {
        ensure!(
            spec.train_split_sha256 == pcode::sha256(&fs::read(root.join("release/train.jsonl"))?),
            "tokenizer train split mismatch"
        );
    }
    let release = root.join("release");
    let lock_sha = pcode::sha256(&fs::read(release.join("lock.json"))?);
    let train_sha = pcode::sha256(&fs::read(release.join("train.jsonl"))?);
    let train = Dataset::load(&release.join("train.jsonl"), config.max_seq_len, &codec)?;
    let validation = Dataset::load(
        &release.join("validation.jsonl"),
        config.max_seq_len,
        &codec,
    )?;
    ensure!(
        train.representation == validation.representation,
        "split lane mismatch"
    );
    // No test split is decoded into model batches. The audit only checks its integrity.
    let mut state = ResumeState {
        schema_version: 1,
        sampler_algorithm: "forge-fisher-yates-v1".into(),
        sampler: Sampler {
            seed: options.seed,
            epoch: 0,
            cursor: 0,
        },
        settings: settings.clone(),
        corpus_lock_sha256: lock_sha,
        train_encoded_sha256: train.effective_sha256.clone(),
        validation_encoded_sha256: validation.effective_sha256.clone(),
        best: None,
    };
    let mut checkpoint = None;
    let mut learning_rate_transition = None;
    if let Some(path) = &options.resume {
        let previous: Checkpoint = serde_json::from_slice(&fs::read(path.join("state.json"))?)?;
        let prior = previous
            .corpus
            .as_ref()
            .context("resume requires corpus checkpoint schema 4")?;
        ensure!(
            previous.schema_version == (if codec.spec.is_some() { 5 } else { 4 })
                && previous.model == config
                && previous.seed == options.seed
                && previous.tokenizer == codec.name()
                && previous.tokenizer_spec == codec.spec
                && previous.dataset_sha256 == train_sha
                && (previous.optimizer.lr == options.learning_rate
                    || options.override_learning_rate)
                && previous.representations == [train.representation.as_str()]
                && previous.input_sha256.as_deref() == Some(&state.train_encoded_sha256),
            "corpus checkpoint provenance/configuration mismatch"
        );
        ensure!(
            prior.schema_version == 1
                && prior.sampler_algorithm == state.sampler_algorithm
                && prior.sampler.seed == options.seed
                && prior.settings == state.settings
                && prior.corpus_lock_sha256 == state.corpus_lock_sha256
                && prior.train_encoded_sha256 == state.train_encoded_sha256
                && prior.validation_encoded_sha256 == state.validation_encoded_sha256,
            "corpus/sampler/validation settings changed"
        );
        let consumed = previous
            .step
            .checked_mul(settings.batch_size as u64)
            .context("sample count overflow")?;
        let position = prior
            .sampler
            .epoch
            .checked_mul(train.rows.len() as u64)
            .and_then(|v| v.checked_add(prior.sampler.cursor as u64))
            .context("sampler overflow")?;
        ensure!(
            prior.sampler.cursor <= train.rows.len() && consumed == position,
            "sampler position does not match optimizer step"
        );
        ensure!(
            previous.checkpoint_sha256.as_deref()
                == Some(&pcode::sha256(&fs::read(
                    path.join("training.safetensors")
                )?)),
            "checkpoint weights digest mismatch"
        );
        if let Some(best) = &prior.best {
            ensure!(
                best.step <= previous.step && best.nll.is_finite() && best.nll >= 0.,
                "best checkpoint metadata mismatch"
            );
        }
        state = prior.clone();
        if let Some(best) = &mut state.best {
            if best.step == previous.step {
                best.checkpoint = Some(path.canonicalize()?);
            }
            if let Some(location) = &best.checkpoint {
                if location.join("training.safetensors").exists() {
                    ensure!(
                        pcode::sha256(&fs::read(location.join("training.safetensors"))?)
                            == best.weights_sha256,
                        "best checkpoint provenance mismatch"
                    );
                } else {
                    // Historical best weights are separate artifacts; moving only the latest
                    // checkpoint still permits training resume, but cannot recover the best model.
                    best.checkpoint = None;
                }
            }
        }

        if previous.optimizer.lr != options.learning_rate {
            learning_rate_transition = Some(
                json!({"policy":"explicit_resume_lr_override_preserve_adam_moments_and_step", "previous_lr":previous.optimizer.lr,"new_lr":options.learning_rate,"parent_checkpoint_sha256":previous.checkpoint_sha256,"parent_step":previous.step}),
            );
        }
        checkpoint = Some(previous);
    }
    let vars = VarMap::new();
    let model = Decoder::new(&config, &vars, device)?;
    let mut optimizer = Adam::new(&vars, AdamConfig::new(options.learning_rate))?;
    if let Some(c) = checkpoint {
        let mut resumed_config = c.optimizer;
        if options.override_learning_rate {
            resumed_config.lr = options.learning_rate;
        }
        optimizer = Adam::new(&vars, resumed_config)?;
        optimizer.load(
            &options
                .resume
                .as_ref()
                .unwrap()
                .join("training.safetensors"),
            c.step,
            device,
        )?;
    } else {
        model::initialize(&vars, options.seed)?;
    }
    let initial_step = optimizer.step;
    let final_step = initial_step
        .checked_add(options.steps as u64)
        .context("step overflow")?;
    let initial_validation = evaluate(&vars, &config, &validation, settings, initial_step, device)?;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    let output = output.canonicalize()?;
    fs::create_dir(output.join("checkpoints"))?;
    let mut log = File::create(output.join("metrics.jsonl"))?;
    append(
        &mut log,
        &json!({"event":"validation_initial","validation":initial_validation}),
    )?;
    let timer = Instant::now();
    let mut trained_tokens = 0;
    let mut trained_examples = 0;
    let mut max_sequence = 0;
    let mut last_validation = initial_validation.clone();
    let mut final_checkpoint = PathBuf::new();
    for _ in 0..options.steps {
        let mut next_sampler = state.sampler.clone();
        let indices = (0..settings.batch_size)
            .map(|_| next_sampler.next(train.rows.len()))
            .collect::<Result<Vec<_>>>()?;
        let batch = train.batch(&indices, config.max_seq_len, device)?;
        let value = {
            let loss = data::completion_loss(
                &model.forward_chunked(&batch.input, settings.attention_chunk)?,
                &batch,
            )?;
            let value = loss.to_scalar::<f32>()?;
            ensure!(value.is_finite(), "nonfinite training loss");
            optimizer.update(&loss)?;
            value
        };
        state.sampler = next_sampler;
        trained_tokens += batch.completion_tokens;
        trained_examples += batch.examples;
        max_sequence = max_sequence.max(batch.sequence_length);
        append(
            &mut log,
            &json!({"event":"train","step":optimizer.step,"nll":value,"sample_ids":indices.iter().map(|i|&train.rows[*i].id).collect::<Vec<_>>(),"sampler":state.sampler,"sequence_length":batch.sequence_length,"completion_tokens":batch.completion_tokens}),
        )?;
        eprintln!(
            "corpus step={} loss={value:.6} sequence={}",
            optimizer.step, batch.sequence_length
        );
        let eval_due =
            optimizer.step.is_multiple_of(settings.evaluate_every) || optimizer.step == final_step;
        if eval_due {
            last_validation = evaluate(
                &vars,
                &config,
                &validation,
                settings,
                optimizer.step,
                device,
            )?;
            append(
                &mut log,
                &json!({"event":"validation","validation":last_validation}),
            )?;
        }
        if eval_due
            || optimizer.step.is_multiple_of(settings.checkpoint_every)
            || optimizer.step == final_step
        {
            final_checkpoint = snapshot(
                &output,
                &optimizer,
                (&config, &codec, &train.representation),
                options.seed,
                &train_sha,
                &mut state,
                if eval_due {
                    Some(&last_validation)
                } else {
                    None
                },
            )?;
            append(
                &mut log,
                &json!({"event":"checkpoint","step":optimizer.step,"path":final_checkpoint,"best_validation":state.best}),
            )?;
        }
    }
    let seconds = timer.elapsed().as_secs_f64();
    let report = json!({"schema_version":1,"tokenizer":codec.name(),"tokenizer_sha256":codec.sha256()?,"learning_rate_transition":learning_rate_transition,"evidence_scope":"bounded_corpus_training_and_teacher_forced_validation_not_decompilation_accuracy","backend":format!("{:?}",device.location()),"dtype":"f32","model":config,"parameters":model::named_vars(&vars).iter().map(|(_,v)|v.elem_count()).sum::<usize>(),"settings":settings,"initial_step":initial_step,"completed_step":optimizer.step,"measured_updates":options.steps,"train_examples":train.rows.len(),"validation_examples":validation.rows.len(),"test_model_examples":0,"trained_examples":trained_examples,"trained_completion_tokens":trained_tokens,"max_training_sequence":max_sequence,"training_and_periodic_validation_seconds":seconds,"completion_tokens_per_second_including_periodic_validation":trained_tokens as f64/seconds,"initial_validation":initial_validation,"final_validation":last_validation,"best_validation":state.best,"sampler":state.sampler,"corpus_lock_sha256":state.corpus_lock_sha256,"train_encoded_sha256":train.effective_sha256,"validation_encoded_sha256":validation.effective_sha256,"corpus_audit":audit["splits"],"final_checkpoint":final_checkpoint,"attention_memory":"query_tiled_exact_causal_attention_still_quadratic_training_activations"});
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
    fn sampler_visits_each_epoch_once_and_resumes_at_boundary() -> Result<()> {
        let mut sampler = Sampler {
            seed: 42,
            epoch: 0,
            cursor: 0,
        };
        let first = (0..17)
            .map(|_| sampler.next(17))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            first.iter().copied().collect::<BTreeSet<_>>(),
            (0..17).collect()
        );
        let mut restored: Sampler = serde_json::from_slice(&serde_json::to_vec(&sampler)?)?;
        let next = (0..35)
            .map(|_| sampler.next(17))
            .collect::<Result<Vec<_>>>()?;
        let resumed = (0..35)
            .map(|_| restored.next(17))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(next, resumed);
        assert_ne!(first, &next[..17]);
        assert_eq!(sampler, restored);
        Ok(())
    }
    #[test]
    fn validation_aggregates_completion_tokens_and_leaves_weights_unchanged() -> Result<()> {
        let config = ModelConfig {
            vocab_size: 260,
            hidden_size: 8,
            heads: 1,
            layers: 1,
            feed_forward_size: 16,
            max_seq_len: 64,
            rotary_positions: false,
        };
        let vars = VarMap::new();
        let model = Decoder::new(&config, &vars, &Device::Cpu)?;
        model::initialize(&vars, 42)?;
        let rows = [("a", "x", "A"), ("b", "longer prompt", "BBBBB")]
            .into_iter()
            .map(|(id, p, s)| {
                let (tokens, start) = data::encode(p, s);
                Example {
                    id: id.into(),
                    tokens,
                    start,
                }
            })
            .collect();
        let validation = Dataset {
            representation: pcode::cfg::REPRESENTATION.into(),
            rows,
            effective_sha256: String::new(),
        };
        let settings = Settings {
            batch_size: 1,
            attention_chunk: 3,
            validation_limit: 0,
            evaluate_every: 1,
            checkpoint_every: 1,
        };
        let before = model::named_vars(&vars)
            .into_iter()
            .map(|(n, v)| Ok((n, v.flatten_all()?.to_vec1::<f32>()?)))
            .collect::<Result<Vec<_>>>()?;
        let result = evaluate(&vars, &config, &validation, &settings, 0, &Device::Cpu)?;
        let mut numerator = 0.;
        let mut denominator = 0;
        for i in 0..2 {
            let b = validation.batch(&[i], 64, &Device::Cpu)?;
            let l = data::completion_loss(&model.forward(&b.input)?, &b)?.to_scalar::<f32>()?;
            numerator += f64::from(l) * b.completion_tokens as f64;
            denominator += b.completion_tokens;
        }
        assert!((result.nll - numerator / denominator as f64).abs() < 1e-6);
        assert_eq!(result.completion_tokens, 8);
        assert_eq!(result.selection, "full_split");
        let after = model::named_vars(&vars)
            .into_iter()
            .map(|(n, v)| Ok((n, v.flatten_all()?.to_vec1::<f32>()?)))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(before, after);
        Ok(())
    }
}

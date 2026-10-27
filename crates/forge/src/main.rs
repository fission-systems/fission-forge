use std::path::PathBuf;

use anyhow::{Result, bail};
use candle_core::Device;
use clap::{Parser, Subcommand, ValueEnum};
use fission_forge::{corpus, corpus_training, generation, pcode, probe, tokenizer, training};

#[derive(Parser)]
#[command(
    name = "forge",
    version,
    about = "Rust neural-decompiler research tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare frozen validation likelihood under correct and rotated wrong IR prompts.
    CorpusDiagnose {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
    },
    /// Audit raw P-code and SSA against owned pure-u32 corpus reference probes.
    CorpusSemantics {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Learn a reversible BPE vocabulary using only the audited training split.
    TokenizerTrain {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 772)]
        vocab_size: usize,
        /// Keep every ASCII digit as its own token (digits are never merged).
        #[arg(long)]
        isolate_digits: bool,
    },
    /// Measure training row lengths under a selected codec without allocating a model.
    TokenizerStats {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    /// Train on a frozen corpus release, with separate validation and resumable epoch sampling.
    TrainCorpus {
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 0.0003)]
        learning_rate: f64,
        /// Explicitly change LR on resume while retaining Adam moments and step.
        #[arg(long)]
        override_learning_rate: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 256)]
        attention_chunk: usize,
        #[arg(long, default_value_t = 0)]
        validation_limit: usize,
        #[arg(long, default_value_t = 100)]
        evaluate_every: u64,
        #[arg(long, default_value_t = 100)]
        checkpoint_every: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
    },
    /// Verify corpus provenance, remove cross-split clones, and optionally freeze a new release.
    CorpusAudit {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        finalize: bool,
    },
    /// Generate raw completion bytes from a trained checkpoint with bounded greedy decoding.
    Generate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: PathBuf,
        #[arg(long, default_value = "assembly-v0")]
        representation: String,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
    },
    /// Validate a raw P-code dump and emit a versioned SSA prompt.
    PcodeCompile {
        #[arg(long)]
        raw: PathBuf,
        #[arg(long, default_value = "pcode-plus-ssa-v0")]
        representation: String,
        #[arg(long)]
        profile: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Train a from-scratch causal decoder on one fixed JSONL batch (a probe only).
    TrainSmoke {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 0.003)]
        learning_rate: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
    },
    /// Verify tensor training and weight restoration on a synthetic fixture.
    BackendProbe {
        /// A new directory for weights and a JSON report; existing paths are refused.
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
    Metal,
}

fn device(backend: Backend) -> Result<Device> {
    match backend {
        Backend::Cpu => Ok(Device::Cpu),
        Backend::Cuda => {
            if !cfg!(feature = "cuda") {
                bail!("CUDA requires a build with --features cuda");
            }
            Ok(Device::new_cuda(0)?)
        }
        Backend::Metal => {
            if !cfg!(feature = "metal") {
                bail!("Metal requires a build with --features metal");
            }
            Ok(Device::new_metal(0)?)
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::CorpusDiagnose {
            corpus,
            checkpoint,
            output,
            device: backend,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&fission_forge::diagnostics::run(
                    &corpus,
                    &checkpoint,
                    &output,
                    &device(backend)?
                )?)?
            );
        }
        Command::CorpusSemantics { corpus, output } => {
            let report = pcode::cfg::audit::run(&corpus, &output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report["failures"] == 0,
                "IR/source parity audit failed; see saved report"
            );
        }
        Command::TokenizerTrain {
            corpus,
            output,
            vocab_size,
            isolate_digits,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&tokenizer::train(
                    &corpus,
                    vocab_size,
                    &output,
                    isolate_digits
                )?)?
            );
        }
        Command::TokenizerStats { corpus, tokenizer } => {
            let codec = tokenizer::Codec::read(tokenizer.as_deref())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&tokenizer::stats(&corpus, &codec)?)?
            );
        }
        Command::TrainCorpus {
            tokenizer,
            config,
            corpus,
            output,
            steps,
            learning_rate,
            override_learning_rate,
            seed,
            batch_size,
            attention_chunk,
            validation_limit,
            evaluate_every,
            checkpoint_every,
            resume,
            device: backend,
        } => {
            let report = corpus_training::run(
                &config,
                &corpus,
                &output,
                &device(backend)?,
                corpus_training::Options {
                    tokenizer,
                    steps,
                    learning_rate,
                    override_learning_rate,
                    seed,
                    resume,
                    settings: corpus_training::Settings {
                        batch_size,
                        attention_chunk,
                        validation_limit,
                        evaluate_every,
                        checkpoint_every,
                    },
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::CorpusAudit { root, finalize } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&corpus::run(&root, finalize)?)?
            );
        }
        Command::Generate {
            checkpoint,
            prompt,
            representation,
            max_new_tokens,
            output,
            device: backend,
        } => {
            let prompt = std::fs::read_to_string(prompt)?;
            let report = generation::run(
                &checkpoint,
                &prompt,
                &representation,
                max_new_tokens,
                &output,
                &device(backend)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::PcodeCompile {
            raw,
            representation,
            profile,
            output,
        } => {
            anyhow::ensure!(
                !output.exists(),
                "output already exists: {}",
                output.display()
            );
            let raw: pcode::RawDump = serde_json::from_slice(&std::fs::read(raw)?)?;
            let profile: pcode::Profile = serde_json::from_slice(&std::fs::read(profile)?)?;
            let report = pcode::Input {
                representation,
                profile,
                raw,
            }
            .compile()?;
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::create_dir(&output)?;
            std::fs::write(output.join("prompt.txt"), &report.prompt)?;
            std::fs::write(
                output.join("report.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::TrainSmoke {
            config,
            dataset,
            output,
            steps,
            learning_rate,
            seed,
            resume,
            device: backend,
        } => {
            let report = training::run(
                &config,
                &dataset,
                &output,
                &device(backend)?,
                training::Options {
                    steps,
                    learning_rate,
                    seed,
                    resume,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::BackendProbe {
            output,
            device: backend,
        } => {
            let report = probe::run(&output, &device(backend)?)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

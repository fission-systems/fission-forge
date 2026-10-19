use std::path::PathBuf;

use anyhow::{Result, bail};
use candle_core::Device;
use clap::{Parser, Subcommand, ValueEnum};
use fission_forge::{pcode, probe, training};

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
    /// Validate a straight-line raw P-code dump and emit a canonical SSA prompt.
    PcodeCompile {
        #[arg(long)]
        raw: PathBuf,
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
        Command::PcodeCompile {
            raw,
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
            let report = pcode::compile(&raw, &profile)?;
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

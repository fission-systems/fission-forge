//! A synthetic backend acceptance probe, not a language model or decompiler.

use std::{fs, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, Optimizer, SGD, VarBuilder, VarMap};
use serde::Serialize;

const STEPS: usize = 64;
const LEARNING_RATE: f64 = 0.5;
const TOLERANCE: f32 = 1e-6;

#[derive(Debug, Serialize)]
pub struct ProbeReport {
    pub schema_version: u32,
    pub evidence_scope: &'static str,
    pub backend: String,
    pub dtype: &'static str,
    pub optimizer: &'static str,
    pub training_steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub reload_max_abs_error: f32,
    pub resumed_step_max_abs_error: f32,
    pub elapsed_seconds: f64,
}

fn weight(vars: &VarMap, device: &Device) -> candle_core::Result<Tensor> {
    VarBuilder::from_varmap(vars, DType::F32, device).get_with_hints(
        (4, 4),
        "weight",
        Init::Const(0.0),
    )
}

fn loss(input: &Tensor, weight: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    (input.matmul(weight)? - target)?.sqr()?.mean_all()
}

fn max_error(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
}

pub fn run(output: &Path, device: &Device) -> Result<ProbeReport> {
    let started = Instant::now();
    let input = Tensor::new(
        &[
            [1f32, 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        device,
    )?;
    let target = Tensor::new(
        &[
            [0f32, 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
        ],
        device,
    )?;
    let vars = VarMap::new();
    let w = weight(&vars, device)?;
    let initial_loss = loss(&input, &w, &target)?.to_scalar::<f32>()?;
    let mut optimizer = SGD::new(vars.all_vars(), LEARNING_RATE)?;

    // Reserve the output atomically; never overwrite another run's checkpoint.
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output).with_context(|| format!("create new run at {}", output.display()))?;

    for _ in 0..STEPS {
        optimizer.backward_step(&loss(&input, &w, &target)?)?;
    }
    let final_loss = loss(&input, &w, &target)?.to_scalar::<f32>()?;
    ensure!(
        final_loss.is_finite() && final_loss < initial_loss * 0.01,
        "training did not reduce loss sufficiently: {initial_loss} -> {final_loss}"
    );
    let prediction = input.matmul(&w)?;
    let checkpoint = output.join("model.safetensors");
    vars.save(&checkpoint)?;

    let mut restored_vars = VarMap::new();
    let restored_w = weight(&restored_vars, device)?;
    restored_vars.load(&checkpoint)?;
    let reload_max_abs_error = max_error(&prediction, &input.matmul(&restored_w)?)?;
    ensure!(
        reload_max_abs_error <= TOLERANCE,
        "weight reload changed predictions"
    );

    // Plain SGD has no momentum or adaptive moments. Reloaded weights plus the
    // learning rate fully restore this optimizer; this does NOT test AdamW resume.
    let mut restored_optimizer = SGD::new(restored_vars.all_vars(), LEARNING_RATE)?;
    optimizer.backward_step(&loss(&input, &w, &target)?)?;
    restored_optimizer.backward_step(&loss(&input, &restored_w, &target)?)?;
    let resumed_step_max_abs_error = max_error(&w, &restored_w)?;
    ensure!(
        resumed_step_max_abs_error <= TOLERANCE,
        "resumed SGD step diverged"
    );

    let report = ProbeReport {
        schema_version: 1,
        evidence_scope: "synthetic_linear_backend_probe_only",
        backend: format!("{:?}", device.location()),
        dtype: "f32",
        optimizer: "sgd_without_momentum",
        training_steps: STEPS,
        initial_loss,
        final_loss,
        reload_max_abs_error,
        resumed_step_max_abs_error,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

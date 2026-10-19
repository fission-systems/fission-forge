//! First experimental decoder: pre-norm, causal attention, GELU, learned positions.
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap, embedding, linear_no_bias};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub heads: usize,
    pub layers: usize,
    pub feed_forward_size: usize,
    pub max_seq_len: usize,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.vocab_size == 260,
            "this baseline requires the 260-token byte codec"
        );
        ensure!(
            self.hidden_size > 0 && self.heads > 0 && self.hidden_size.is_multiple_of(self.heads),
            "hidden size must be divisible by positive head count"
        );
        ensure!(
            self.layers > 0 && self.feed_forward_size > 0 && self.max_seq_len >= 2,
            "invalid model dimensions"
        );
        Ok(())
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}
impl Norm {
    fn new(size: usize, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?,
            bias: vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Primitive ops retain an explicit autodiff path on every backend.
        let centered = x.broadcast_sub(&x.mean_keepdim(candle_core::D::Minus1)?)?;
        let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        centered
            .broadcast_div(&(variance + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
}
impl Block {
    fn new(c: &ModelConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            norm1: Norm::new(h, vb.pp("norm1"))?,
            norm2: Norm::new(h, vb.pp("norm2"))?,
            qkv: linear_no_bias(h, h * 3, vb.pp("qkv"))?,
            out: linear_no_bias(h, h, vb.pp("out"))?,
            up: linear_no_bias(h, c.feed_forward_size, vb.pp("up"))?,
            down: linear_no_bias(c.feed_forward_size, h, vb.pp("down"))?,
        })
    }
    fn forward(&self, x: &Tensor, mask: &Tensor, c: &ModelConfig) -> candle_core::Result<Tensor> {
        let (batch, time, hidden) = x.dims3()?;
        let width = hidden / c.heads;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?;
        let split = |offset| {
            qkv.narrow(2, offset, hidden)?
                .reshape((batch, time, c.heads, width))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(hidden)?, split(hidden * 2)?);
        let scores =
            (q.matmul(&k.transpose(2, 3)?)? / (width as f64).sqrt())?.broadcast_add(mask)?;
        let attention = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let values = attention
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, hidden))?;
        let x = (x + self.out.forward(&values)?)?;
        let ff = self
            .down
            .forward(&self.up.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        x + ff
    }
}

pub struct Decoder {
    config: ModelConfig,
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    norm: Norm,
    head: Linear,
}
impl Decoder {
    pub fn new(c: &ModelConfig, vars: &VarMap, device: &Device) -> Result<Self> {
        c.validate()?;
        let vb = VarBuilder::from_varmap(vars, DType::F32, device);
        Ok(Self {
            config: c.clone(),
            tokens: embedding(c.vocab_size, c.hidden_size, vb.pp("tokens"))?,
            positions: embedding(c.max_seq_len, c.hidden_size, vb.pp("positions"))?,
            blocks: (0..c.layers)
                .map(|i| Block::new(c, vb.pp(format!("blocks.{i}"))))
                .collect::<candle_core::Result<_>>()?,
            norm: Norm::new(c.hidden_size, vb.pp("final_norm"))?,
            head: linear_no_bias(c.hidden_size, c.vocab_size, vb.pp("head"))?,
        })
    }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, time) = ids.dims2()?;
        ensure!(
            time > 0 && time <= self.config.max_seq_len,
            "sequence exceeds model context"
        );
        let positions = Tensor::arange(0u32, time as u32, ids.device())?;
        let mut x = self
            .tokens
            .forward(ids)?
            .broadcast_add(&self.positions.forward(&positions)?.unsqueeze(0)?)?;
        let mask: Vec<f32> = (0..time)
            .flat_map(|i| (0..time).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, time, time), ids.device())?;
        for block in &self.blocks {
            x = block.forward(&x, &mask, &self.config)?;
        }
        Ok(self.head.forward(&self.norm.forward(&x)?)?)
    }
}

/// Stable initialization across processes without relying on backend RNG state.
pub fn initialize(vars: &VarMap, mut seed: u64) -> Result<()> {
    ensure!(seed != 0, "seed must be nonzero");
    let vars = named_vars(vars);
    for (name, var) in vars {
        let values = (0..var.elem_count())
            .map(|_| {
                if name.ends_with("bias") {
                    0.
                } else if name.contains("norm") {
                    1.
                } else {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / 16_777_216. - 0.5) * 0.04
                }
            })
            .collect::<Vec<_>>();
        var.set(&Tensor::from_vec(values, var.shape(), var.device())?)?;
    }
    Ok(())
}

pub fn named_vars(vars: &VarMap) -> Vec<(String, candle_core::Var)> {
    let mut entries = vars
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), v.clone()))
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_tokens_cannot_change_prefix_logits() -> Result<()> {
        let c = ModelConfig {
            vocab_size: 260,
            hidden_size: 16,
            heads: 2,
            layers: 1,
            feed_forward_size: 32,
            max_seq_len: 8,
        };
        let vars = VarMap::new();
        let model = Decoder::new(&c, &vars, &Device::Cpu)?;
        initialize(&vars, 42)?;
        let a = model.forward(&Tensor::new(&[[1u32, 5, 6, 7]], &Device::Cpu)?)?;
        let b = model.forward(&Tensor::new(&[[1u32, 5, 99, 100]], &Device::Cpu)?)?;
        let delta = (a.narrow(1, 0, 2)? - b.narrow(1, 0, 2)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(delta <= 1e-6, "future-token leakage: {delta}");
        Ok(())
    }
}

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
            (260..=4096).contains(&self.vocab_size),
            "vocabulary must contain 260..4096 tokens"
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

struct LayerCache {
    key: Tensor,
    value: Tensor,
}

pub struct DecodeCache {
    layers: Vec<LayerCache>,
    position: usize,
    batch: usize,
    config: ModelConfig,
}
impl DecodeCache {
    pub fn position(&self) -> usize {
        self.position
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
    fn forward(
        &self,
        x: &Tensor,
        c: &ModelConfig,
        query_chunk: usize,
    ) -> candle_core::Result<Tensor> {
        Ok(self.prefill(x, c, query_chunk)?.0)
    }
    fn prefill(
        &self,
        x: &Tensor,
        c: &ModelConfig,
        query_chunk: usize,
    ) -> candle_core::Result<(Tensor, LayerCache)> {
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
        // Query tiling avoids a full T*T mask and skips future key blocks.
        // Backprop still retains attention activations: this is not FlashAttention.
        let chunk = if query_chunk == 0 {
            time
        } else {
            query_chunk.min(time)
        };
        let mut pieces = Vec::new();
        for start in (0..time).step_by(chunk) {
            let count = chunk.min(time - start);
            let end = start + count;
            let mask: Vec<f32> = (start..end)
                .flat_map(|i| (0..end).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, count, end), x.device())?;
            let scores = (q
                .narrow(2, start, count)?
                .matmul(&k.narrow(2, 0, end)?.transpose(2, 3)?)?
                / (width as f64).sqrt())?
            .broadcast_add(&mask)?;
            let attention = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
            pieces.push(attention.matmul(&v.narrow(2, 0, end)?)?);
        }
        let values = Tensor::cat(&pieces, 2)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, hidden))?;
        let x = (x + self.out.forward(&values)?)?;
        let ff = self
            .down
            .forward(&self.up.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        Ok((
            (x + ff)?,
            LayerCache {
                key: k.detach(),
                value: v.detach(),
            },
        ))
    }
    fn step(
        &self,
        x: &Tensor,
        c: &ModelConfig,
        cache: &mut LayerCache,
    ) -> candle_core::Result<Tensor> {
        let (batch, _, hidden) = x.dims3()?;
        let width = hidden / c.heads;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?;
        let split = |offset| {
            qkv.narrow(2, offset, hidden)?
                .reshape((batch, 1, c.heads, width))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(hidden)?, split(hidden * 2)?);
        let key = Tensor::cat(&[&cache.key, &k], 2)?;
        let value = Tensor::cat(&[&cache.value, &v], 2)?;
        let scores = (q.matmul(&key.transpose(2, 3)?)? / (width as f64).sqrt())?;
        let values = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, hidden))?;
        let x = (x + self.out.forward(&values)?)?;
        let ff = self
            .down
            .forward(&self.up.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        cache.key = key.detach();
        cache.value = value.detach();
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
        Self::from_builder(c, VarBuilder::from_varmap(vars, DType::F32, device))
    }
    /// A weight snapshot with no variable leaves; validation creates no gradient graph.
    pub fn for_eval(c: &ModelConfig, vars: &VarMap, device: &Device) -> Result<Self> {
        let tensors = named_vars(vars)
            .into_iter()
            .map(|(n, v)| (n, v.detach()))
            .collect();
        Self::from_builder(c, VarBuilder::from_tensors(tensors, DType::F32, device))
    }
    fn from_builder(c: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        c.validate()?;
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
    /// Full causal prefix once, retaining detached keys/values for incremental inference.
    pub fn prefill(&self, ids: &Tensor, query_chunk: usize) -> Result<(Tensor, DecodeCache)> {
        let (batch, time) = ids.dims2()?;
        ensure!(
            batch > 0 && time > 0 && time <= self.config.max_seq_len,
            "invalid cached prefix/context"
        );
        let positions = Tensor::arange(0u32, time as u32, ids.device())?;
        let mut x = self
            .tokens
            .forward(ids)?
            .broadcast_add(&self.positions.forward(&positions)?.unsqueeze(0)?)?;
        let mut layers = Vec::new();
        for block in &self.blocks {
            let (next, cache) = block.prefill(&x, &self.config, query_chunk)?;
            x = next;
            layers.push(cache);
        }
        let logits = self
            .head
            .forward(&self.norm.forward(&x.narrow(1, time - 1, 1)?)?)?;
        Ok((
            logits,
            DecodeCache {
                layers,
                position: time,
                batch,
                config: self.config.clone(),
            },
        ))
    }
    pub fn decode_step(&self, token: &Tensor, cache: &mut DecodeCache) -> Result<Tensor> {
        ensure!(
            token.dims2()? == (cache.batch, 1)
                && cache.config == self.config
                && cache.layers.len() == self.blocks.len()
                && cache.position < self.config.max_seq_len,
            "invalid decode cache/token/context"
        );
        let position = Tensor::new(&[cache.position as u32], token.device())?;
        let mut x = self
            .tokens
            .forward(token)?
            .broadcast_add(&self.positions.forward(&position)?.unsqueeze(0)?)?;
        for (block, layer) in self.blocks.iter().zip(&mut cache.layers) {
            x = block.step(&x, &self.config, layer)?;
        }
        cache.position += 1;
        Ok(self.head.forward(&self.norm.forward(&x)?)?)
    }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.forward_chunked(ids, 0)
    }
    pub fn forward_chunked(&self, ids: &Tensor, query_chunk: usize) -> Result<Tensor> {
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
        for block in &self.blocks {
            x = block.forward(&x, &self.config, query_chunk)?;
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

#[cfg(test)]
mod tiled_tests {
    use super::*;
    #[test]
    fn tiled_attention_matches_full_logits_and_parameter_gradients() -> Result<()> {
        let c = ModelConfig {
            vocab_size: 260,
            hidden_size: 8,
            heads: 2,
            layers: 1,
            feed_forward_size: 16,
            max_seq_len: 32,
        };
        let vars = VarMap::new();
        let model = Decoder::new(&c, &vars, &Device::Cpu)?;
        initialize(&vars, 42)?;
        let ids = Tensor::new(&[[1u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], &Device::Cpu)?;
        let full = model.forward(&ids)?;
        let tiled = model.forward_chunked(&ids, 3)?;
        assert!(
            (&full - &tiled)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?
                < 1e-6
        );
        let a = full.sqr()?.mean_all()?.backward()?;
        let b = tiled.sqr()?.mean_all()?.backward()?;
        for (_, v) in named_vars(&vars) {
            let delta = (a.get(&v).unwrap() - b.get(&v).unwrap())?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(delta < 1e-5, "gradient mismatch {delta}");
        }
        let inference = Decoder::for_eval(&c, &vars, &Device::Cpu)?;
        let loss = inference.forward_chunked(&ids, 3)?.sqr()?.mean_all()?;
        let detached = loss.backward()?;
        for (_, v) in named_vars(&vars) {
            assert!(
                detached.get(&v).is_none(),
                "validation retains variable graph"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    #[test]
    fn cached_logits_match_full_prefix_across_layers_and_stop_at_context() -> Result<()> {
        let c = ModelConfig {
            vocab_size: 260,
            hidden_size: 16,
            heads: 2,
            layers: 2,
            feed_forward_size: 32,
            max_seq_len: 16,
        };
        let vars = VarMap::new();
        let _ = Decoder::new(&c, &vars, &Device::Cpu)?;
        initialize(&vars, 42)?;
        let model = Decoder::for_eval(&c, &vars, &Device::Cpu)?;
        let mut tokens = vec![1u32, 5, 6, 7, 8];
        let (prefill, mut cache) = model.prefill(
            &Tensor::from_slice(&tokens, (1, tokens.len()), &Device::Cpu)?,
            3,
        )?;
        let last = model
            .forward(&Tensor::from_slice(
                &tokens,
                (1, tokens.len()),
                &Device::Cpu,
            )?)?
            .narrow(1, tokens.len() - 1, 1)?;
        assert!(
            (&prefill - &last)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?
                < 1e-5
        );
        for token in 20..31 {
            tokens.push(token);
            let step = model.decode_step(&Tensor::new(&[[token]], &Device::Cpu)?, &mut cache)?;
            let expected = model
                .forward(&Tensor::from_slice(
                    &tokens,
                    (1, tokens.len()),
                    &Device::Cpu,
                )?)?
                .narrow(1, tokens.len() - 1, 1)?;
            assert!(
                (&step - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?
                    < 1e-5
            );
        }
        assert_eq!(cache.position(), 16);
        assert!(
            model
                .decode_step(&Tensor::new(&[[42u32]], &Device::Cpu)?, &mut cache)
                .is_err()
        );
        Ok(())
    }
}

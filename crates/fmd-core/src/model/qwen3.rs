//! Qwen3 text backbone with a batched KV cache.
//!
//! Two things here are load-bearing for speed and for correctness under
//! batching, and both are easy to get subtly wrong:
//!
//! * **Chunked prefill.** A 10-minute recording produces an ~8000-token prompt.
//!   Materialising an 8000x8000 attention matrix per head would need gigabytes,
//!   so queries are processed in blocks against the whole key history.
//!
//! * **Left padding with shared positions.** Sequences in a batch are
//!   left-padded to a common length and given positions `0..len` regardless of
//!   where their real tokens start. RoPE scores depend only on the *difference*
//!   between two positions, so a constant per-sequence shift is invisible to
//!   attention; only the padding mask has to distinguish the sequences.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops::softmax_last_dim;

use crate::config::TextConfig;
use crate::error::Result;
use crate::model::linear::{FusedLinear, Linear, Loader, RmsNorm};

/// Query rows per prefill block. 256 keeps the transient attention matrix in the
/// tens of megabytes even for very long prompts, while still being wide enough
/// to saturate the tensor cores.
const PREFILL_BLOCK: usize = 256;

/// Per-layer key/value cache for a whole batch.
///
/// Shape `(batch, kv_heads, capacity, head_dim)`. Capacity is fixed at
/// construction: the caller knows the prompt length and the generation budget
/// up front, and growing would mean copying gigabytes mid-decode.
pub struct KvCache {
    k: Tensor,
    v: Tensor,
    len: usize,
    capacity: usize,
}

impl KvCache {
    pub fn new(
        batch: usize,
        kv_heads: usize,
        capacity: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros((batch, kv_heads, capacity, head_dim), dtype, device)?,
            v: Tensor::zeros((batch, kv_heads, capacity, head_dim), dtype, device)?,
            len: 0,
            capacity,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes one layer's cache occupies, for capacity planning.
    pub fn bytes_per_layer(
        batch: usize,
        kv_heads: usize,
        capacity: usize,
        head_dim: usize,
        dtype: DType,
    ) -> usize {
        2 * batch * kv_heads * capacity * head_dim * dtype.size_in_bytes()
    }

    /// Write `k`/`v` (shape `(b, kv_heads, t, head_dim)`) at the current offset
    /// and return views over the whole history.
    ///
    /// The write is in place. Reallocating the cache each step -- which is what
    /// `slice_assign` would do -- would copy well over a gigabyte per layer per
    /// token on long prompts and dominate the whole decode.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = k.dim(2)?;
        if self.len + t > self.capacity {
            return Err(candle_core::Error::Msg(format!(
                "kv cache overflow: {} + {t} exceeds capacity {}",
                self.len, self.capacity
            ))
            .into());
        }
        self.k.slice_set(&k.contiguous()?, 2, self.len)?;
        self.v.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len += t;
        Ok((
            self.k.narrow(2, 0, self.len)?,
            self.v.narrow(2, 0, self.len)?,
        ))
    }

    /// Drop everything but the first `len` positions, so a cache can be reused.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

/// Precomputed RoPE tables.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(
        head_dim: usize,
        theta: f64,
        max_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;
        let positions = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to `(b, heads, t, head_dim)` starting at absolute `offset`.
    fn apply(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let t = xs.dim(2)?;
        let cos = self.cos.narrow(0, offset, t)?;
        let sin = self.sin.narrow(0, offset, t)?;
        Ok(candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)?)
    }
}

/// Expand grouped key/value heads to match the query head count.
///
/// The engine does *not* use this on the hot path -- see [`Attention::forward`],
/// which regroups the queries instead. It is kept because it defines the head
/// ordering that regrouping has to reproduce, and the tests check the two agree.
#[cfg_attr(not(test), allow(dead_code))]
fn repeat_kv(xs: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(xs.clone());
    }
    let (b, kv_heads, t, d) = xs.dims4()?;
    Ok(xs
        .unsqueeze(2)?
        .expand((b, kv_heads, repeat, t, d))?
        .reshape((b, kv_heads * repeat, t, d))?)
}

struct Attention {
    qkv: FusedLinear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    repeat: usize,
    scale: f64,
}

impl Attention {
    fn load(loader: &Loader, cfg: &TextConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            qkv: loader.fused_linear(
                h,
                &[
                    ("self_attn.q_proj", cfg.q_dim()),
                    ("self_attn.k_proj", cfg.kv_dim()),
                    ("self_attn.v_proj", cfg.kv_dim()),
                ],
                cfg.attention_bias,
                true,
            )?,
            o: loader.linear(cfg.q_dim(), h, "self_attn.o_proj", false)?,
            // Qwen3 normalises each head's q and k over head_dim before RoPE.
            q_norm: RmsNorm::load(loader, cfg.head_dim, "self_attn.q_norm", cfg.rms_norm_eps)?,
            k_norm: RmsNorm::load(loader, cfg.head_dim, "self_attn.k_norm", cfg.rms_norm_eps)?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            repeat: cfg.kv_repeat(),
            scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    /// `mask` is `(b, 1, q_len, kv_len)` additive, or `None` for unmasked.
    fn forward(
        &self,
        xs: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        offset: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;

        let parts = self.qkv.forward(xs)?;
        let q = parts[0].reshape((b, t, self.heads, self.head_dim))?;
        let k = parts[1].reshape((b, t, self.kv_heads, self.head_dim))?;
        let v = parts[2].reshape((b, t, self.kv_heads, self.head_dim))?;

        // Head-wise RMS norm happens on the (.., heads, head_dim) layout, before
        // the transpose into attention order.
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        let (k_all, v_all) = cache.append(&k, &v)?;

        // Grouped-query attention without expanding the cache.
        //
        // The obvious implementation broadcasts each key/value head out to the
        // query head count, but that copies the entire cache twice per layer per
        // token -- gigabytes of traffic that dominated everything else. Instead
        // the *queries* are regrouped onto the key/value head axis, which is a
        // free reshape, and the cache is read in place as a strided view.
        //
        // `(b, heads, t, d) -> (b, kv_heads, repeat * t, d)` keeps query head
        // `kv * repeat + r` adjacent to its own key head, matching the layout
        // `repeat_kv` would have produced.
        let kv_heads = self.kv_heads;
        let grouped = q.reshape((b, kv_heads, self.repeat * t, self.head_dim))?;

        let scores = (grouped.matmul(&k_all.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last_dim(&scores)?;

        // `(b, kv_heads, repeat * t, d)` back to `(b, t, heads * d)`, restoring
        // head-major order for the output projection.
        let ctx = probs
            .matmul(&v_all)?
            .reshape((b, kv_heads, self.repeat, t, self.head_dim))?
            .permute((0, 3, 1, 2, 4))?
            .contiguous()?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o.forward(&ctx)
    }
}

struct Mlp {
    gate_up: FusedLinear,
    down: Linear,
}

impl Mlp {
    fn load(loader: &Loader, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            gate_up: loader.fused_linear(
                cfg.hidden_size,
                &[
                    ("mlp.gate_proj", cfg.intermediate_size),
                    ("mlp.up_proj", cfg.intermediate_size),
                ],
                false,
                true,
            )?,
            down: loader.linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                "mlp.down_proj",
                false,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let parts = self.gate_up.forward(xs)?;
        let gated = candle_nn::ops::silu(&parts[0])?;
        self.down.forward(&(gated * &parts[1])?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

impl Layer {
    fn load(loader: &Loader, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(loader, cfg)?,
            mlp: Mlp::load(loader, cfg)?,
            input_norm: RmsNorm::load(
                loader,
                cfg.hidden_size,
                "input_layernorm",
                cfg.rms_norm_eps,
            )?,
            post_attn_norm: RmsNorm::load(
                loader,
                cfg.hidden_size,
                "post_attention_layernorm",
                cfg.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        offset: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let h = self
            .attn
            .forward(&self.input_norm.forward(xs)?, rope, cache, offset, mask)?;
        let xs = (residual + h)?;

        let residual = &xs;
        let h = self.mlp.forward(&self.post_attn_norm.forward(&xs)?)?;
        Ok((residual + h)?)
    }
}

pub struct Qwen3 {
    embed: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    pub cfg: TextConfig,
    dtype: DType,
    device: Device,
}

impl Qwen3 {
    pub fn load(loader: &Loader, cfg: &TextConfig, max_len: usize) -> Result<Self> {
        let lm = loader.sub("model.language_model");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(&lm.sub(&format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        let embed = lm.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?;

        // The checkpoint ties `lm_head` to the input embedding and therefore
        // stores no separate tensor; reuse the embedding matrix as the output
        // projection. Quantising it is a large win: at 151936 x 1024 it is a
        // sixth of the model and is touched on every decode step.
        let lm_head = if lm.contains("lm_head.weight") {
            lm.linear(cfg.hidden_size, cfg.vocab_size, "lm_head", false)?
        } else if loader.contains("lm_head.weight") {
            loader.linear(cfg.hidden_size, cfg.vocab_size, "lm_head", false)?
        } else {
            build_tied_head(loader, cfg)?
        };

        Ok(Self {
            layers,
            norm: RmsNorm::load(&lm, cfg.hidden_size, "norm", cfg.rms_norm_eps)?,
            rope: Rope::new(
                cfg.head_dim,
                cfg.rope_theta,
                max_len,
                loader.dtype(),
                loader.device(),
            )?,
            lm_head,
            embed,
            cfg: cfg.clone(),
            dtype: loader.dtype(),
            device: loader.device().clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Look up token embeddings for `(b, t)` ids.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        Ok(self.embed.index_select(&ids.flatten_all()?, 0)?.reshape((
            b,
            t,
            self.cfg.hidden_size,
        ))?)
    }

    /// Allocate one cache per layer.
    pub fn new_caches(&self, batch: usize, capacity: usize) -> Result<Vec<KvCache>> {
        (0..self.cfg.num_hidden_layers)
            .map(|_| {
                KvCache::new(
                    batch,
                    self.cfg.num_key_value_heads,
                    capacity,
                    self.cfg.head_dim,
                    self.dtype,
                    &self.device,
                )
            })
            .collect()
    }

    /// Total KV cache bytes for a batch, across all layers.
    pub fn cache_bytes(&self, batch: usize, capacity: usize) -> usize {
        self.cfg.num_hidden_layers
            * KvCache::bytes_per_layer(
                batch,
                self.cfg.num_key_value_heads,
                capacity,
                self.cfg.head_dim,
                self.dtype,
            )
    }

    /// Run the stack over `(b, t, hidden)` embeddings, returning the final
    /// hidden states. `mask` is additive of shape `(b, 1, t, offset + t)`.
    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        caches: &mut [KvCache],
        offset: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = embeds.clone();
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            xs = layer.forward(&xs, &self.rope, cache, offset, mask)?;
        }
        self.norm.forward(&xs)
    }

    /// Project hidden states to vocabulary logits.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        // Logits are always produced in f32: sampling and argmax over 152k
        // classes are noticeably noisier in f16.
        Ok(self.lm_head.forward(hidden)?.to_dtype(DType::F32)?)
    }

    /// Hidden states for the final position only, which is all generation needs.
    pub fn last_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        let t = hidden.dim(1)?;
        Ok(hidden.i((.., t - 1..t, ..))?)
    }

    /// Prefill the cache with a full prompt, returning the last hidden state.
    ///
    /// Queries are consumed in blocks of [`PREFILL_BLOCK`] so peak memory stays
    /// bounded no matter how long the prompt is.
    pub fn prefill(
        &self,
        embeds: &Tensor,
        caches: &mut [KvCache],
        pad_lengths: &[usize],
    ) -> Result<Tensor> {
        let (b, total, _) = embeds.dims3()?;
        let mut last = None;

        let mut start = 0;
        while start < total {
            let len = PREFILL_BLOCK.min(total - start);
            let block = embeds.narrow(1, start, len)?.contiguous()?;
            let mask = self.build_mask(b, start, len, start + len, pad_lengths)?;
            let hidden = self.forward_embeds(&block, caches, start, Some(&mask))?;
            last = Some(hidden);
            start += len;
        }

        let hidden = last.ok_or_else(|| candle_core::Error::Msg("empty prompt".into()))?;
        self.last_hidden(&hidden)
    }

    /// One decode step for `(b, 1)` token ids.
    pub fn decode_step(
        &self,
        ids: &Tensor,
        caches: &mut [KvCache],
        offset: usize,
        pad_lengths: &[usize],
    ) -> Result<Tensor> {
        let embeds = self.embed(ids)?;
        let b = embeds.dim(0)?;
        let mask = self.build_mask(b, offset, 1, offset + 1, pad_lengths)?;
        let hidden = self.forward_embeds(&embeds, caches, offset, Some(&mask))?;
        self.logits(&hidden)
    }

    /// Build the additive attention mask, laid out for the regrouped queries.
    ///
    /// Two things are masked: positions after the query (causality) and the
    /// left-padding region of each sequence. The mask does not depend on the
    /// head, so the `q_len` rows are simply tiled `repeat` times to line up with
    /// the `(b, kv_heads, repeat * q_len, kv_len)` scores that
    /// [`Attention::forward`] produces; it then broadcasts over `kv_heads`.
    ///
    /// Building it here, once per forward pass rather than once per layer, keeps
    /// the tiling off the hot path.
    fn build_mask(
        &self,
        batch: usize,
        q_offset: usize,
        q_len: usize,
        kv_len: usize,
        pad_lengths: &[usize],
    ) -> Result<Tensor> {
        let repeat = self.cfg.kv_repeat();
        let neg = f32::NEG_INFINITY;
        let rows = repeat * q_len;
        let mut data = vec![0f32; batch * rows * kv_len];

        for (bi, &pad) in pad_lengths.iter().enumerate().take(batch) {
            for qi in 0..q_len {
                let abs_q = q_offset + qi;
                // Query rows that lie inside the padding get plain causal
                // masking instead of the padding rule.
                //
                // Applying the padding rule to them would mask every key, and
                // a softmax over an all -inf row is NaN. That NaN reaches the
                // real tokens even though they never attend to a pad position,
                // because the value matmul computes `0 * NaN` for the masked
                // entries and the sum poisons the whole row. Their outputs are
                // discarded, so any well-defined value will do.
                let padding_applies = abs_q >= pad;
                let first = (bi * rows + qi) * kv_len;
                for ki in 0..kv_len {
                    if ki > abs_q || (padding_applies && ki < pad) {
                        data[first + ki] = neg;
                    }
                }
                for r in 1..repeat {
                    let dst = (bi * rows + r * q_len + qi) * kv_len;
                    data.copy_within(first..first + kv_len, dst);
                }
            }
        }
        Ok(Tensor::from_vec(data, (batch, 1, rows, kv_len), &self.device)?.to_dtype(self.dtype)?)
    }
}

/// Reconstruct the output projection from the tied input embedding.
fn build_tied_head(loader: &Loader, cfg: &TextConfig) -> Result<Linear> {
    let name = "model.language_model.embed_tokens";
    if loader.precision().is_quantized() {
        // Go through the same f32 source the other quantised layers use.
        let sub = loader.sub("model.language_model");
        sub.linear(cfg.hidden_size, cfg.vocab_size, "embed_tokens", false)
    } else {
        Ok(Linear::Dense {
            weight: loader.get((cfg.vocab_size, cfg.hidden_size), &format!("{name}.weight"))?,
            bias: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TextConfig {
        TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
            attention_bias: false,
        }
    }

    #[test]
    fn cache_appends_and_reports_length() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(2, 2, 8, 4, DType::F32, &dev)?;
        assert!(cache.is_empty());

        let k = Tensor::ones((2, 2, 3, 4), DType::F32, &dev)?;
        let (k_all, v_all) = cache.append(&k, &k)?;
        assert_eq!(cache.len(), 3);
        assert_eq!(k_all.dims(), &[2, 2, 3, 4]);
        assert_eq!(v_all.dims(), &[2, 2, 3, 4]);

        let (k_all, _) = cache.append(&k, &k)?;
        assert_eq!(cache.len(), 6);
        assert_eq!(k_all.dims(), &[2, 2, 6, 4]);
        Ok(())
    }

    #[test]
    fn cache_refuses_to_overflow() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(1, 1, 4, 2, DType::F32, &dev)?;
        let k = Tensor::zeros((1, 1, 5, 2), DType::F32, &dev)?;
        assert!(cache.append(&k, &k).is_err());
        Ok(())
    }

    #[test]
    fn cache_bytes_match_the_allocation() {
        // 2 tensors x 4 batch x 8 heads x 1000 positions x 128 dims x 2 bytes.
        let bytes = KvCache::bytes_per_layer(4, 8, 1000, 128, DType::F16);
        assert_eq!(bytes, 2 * 4 * 8 * 1000 * 128 * 2);
    }

    #[test]
    fn repeat_kv_expands_grouped_heads() -> Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 1, 2), &dev)?;
        let out = repeat_kv(&xs, 2)?;
        assert_eq!(out.dims(), &[1, 4, 1, 2]);
        // Each kv head is duplicated in place, not interleaved across heads.
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            vec![1., 2., 1., 2., 3., 4., 3., 4.]
        );
        assert_eq!(repeat_kv(&xs, 1)?.dims(), xs.dims());
        Ok(())
    }

    #[test]
    fn rope_is_shift_invariant() -> Result<()> {
        // The engine relies on this: left-padded sequences share one position
        // range, so attention must only see relative offsets.
        let dev = Device::Cpu;
        let rope = Rope::new(4, 10_000.0, 32, DType::F32, &dev)?;
        let q = Tensor::randn(0f32, 1f32, (1, 1, 1, 4), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (1, 1, 1, 4), &dev)?;

        let score_at = |qp: usize, kp: usize| -> Result<f32> {
            let qr = rope.apply(&q, qp)?;
            let kr = rope.apply(&k, kp)?;
            Ok(qr
                .matmul(&kr.transpose(2, 3)?)?
                .flatten_all()?
                .to_vec1::<f32>()?[0])
        };

        // Same relative distance (3) at two different absolute offsets.
        let a = score_at(5, 2)?;
        let b = score_at(20, 17)?;
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        Ok(())
    }

    /// The grouped-query path must be arithmetically identical to the naive one.
    ///
    /// This is the optimisation the decode loop depends on: queries are
    /// regrouped onto the key/value head axis so the cache is never expanded.
    /// If the reshape/permute algebra were wrong the model would still run and
    /// still produce fluent text, just attending through the wrong heads -- so
    /// it is pinned here against the obvious implementation.
    #[test]
    fn grouped_attention_matches_expanded_attention() -> Result<()> {
        let dev = Device::Cpu;
        let (b, kv_heads, repeat, t, l, d) = (2usize, 3usize, 2usize, 4usize, 7usize, 5usize);
        let heads = kv_heads * repeat;

        let q = Tensor::randn(0f32, 1f32, (b, heads, t, d), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, kv_heads, l, d), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, kv_heads, l, d), &dev)?;

        // Reference: expand key/value heads to the query head count.
        let k_full = repeat_kv(&k, repeat)?;
        let v_full = repeat_kv(&v, repeat)?;
        let expected = softmax_last_dim(&q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?)?
            .matmul(&v_full)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, heads * d))?;

        // Engine path: regroup the queries, leave key/value alone.
        let grouped = q.reshape((b, kv_heads, repeat * t, d))?;
        let got = softmax_last_dim(&grouped.matmul(&k.transpose(D::Minus2, D::Minus1)?)?)?
            .matmul(&v)?
            .reshape((b, kv_heads, repeat, t, d))?
            .permute((0, 3, 1, 2, 4))?
            .contiguous()?
            .reshape((b, t, heads * d))?;

        assert_eq!(got.dims(), expected.dims());
        let diff = (got - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "grouped path diverged by {diff}");
        Ok(())
    }

    /// No mask row may be entirely masked.
    ///
    /// Regression test. A fully masked row makes `softmax` produce NaN, and that
    /// NaN spreads from the padding into the real tokens through the `0 * NaN`
    /// terms of the value matmul -- which showed up as a batched short clip
    /// decoding to a wall of "!" while the same clip alone decoded correctly.
    #[test]
    fn no_query_row_is_fully_masked() -> Result<()> {
        let dev = Device::Cpu;
        let model = bare_model(&dev)?;

        // A heavily padded sequence next to an unpadded one, which is exactly
        // the shape that batching a short clip with a long one produces.
        let (q_len, kv_len) = (6, 6);
        let mask = model.build_mask(2, 0, q_len, kv_len, &[4, 0])?;
        let m = mask.i((.., 0, .., ..))?.to_vec3::<f32>()?;

        for (bi, seq) in m.iter().enumerate() {
            for (ri, row) in seq.iter().enumerate() {
                assert!(
                    row.iter().any(|v| v.is_finite()),
                    "batch {bi} row {ri} is fully masked"
                );
            }
        }

        // The real tokens of the padded sequence must still not see the padding.
        // Query 4 is the first real token, so only key 4 is visible.
        assert!(m[0][4][..4].iter().all(|v| v.is_infinite()));
        assert_eq!(m[0][4][4], 0.0);
        assert!(m[0][4][5].is_infinite());
        Ok(())
    }

    #[test]
    fn softmax_over_the_mask_never_produces_nan() -> Result<()> {
        let dev = Device::Cpu;
        let model = bare_model(&dev)?;
        let mask = model.build_mask(2, 0, 6, 6, &[4, 0])?;
        // Zero scores plus the mask is the worst case for the padded rows.
        let scores = (Tensor::zeros((2, 1, 12, 6), DType::F32, &dev)? + &mask)?;
        let probs = softmax_last_dim(&scores)?;
        let total = probs.sum_all()?.to_scalar::<f32>()?;
        assert!(total.is_finite(), "softmax produced NaN or inf: {total}");
        Ok(())
    }

    #[test]
    fn mask_is_tiled_for_the_regrouped_queries() -> Result<()> {
        let dev = Device::Cpu;
        let model = bare_model(&dev)?;
        // cfg() has 4 query heads over 2 kv heads, so repeat = 2.
        let mask = model.build_mask(1, 0, 3, 3, &[0])?;
        assert_eq!(mask.dims(), &[1, 1, 6, 3], "rows must be repeat * q_len");

        let m = mask.i((0, 0))?.to_vec2::<f32>()?;
        // Row r * q_len + q carries the mask for query q, for every r.
        for q in 0..3 {
            assert_eq!(m[q], m[3 + q], "tile {q} must match");
        }
        Ok(())
    }

    /// A model with no layers, for exercising mask construction alone.
    fn bare_model(dev: &Device) -> Result<Qwen3> {
        Ok(Qwen3 {
            embed: Tensor::zeros((32, 8), DType::F32, dev)?,
            layers: vec![],
            norm: RmsNorm {
                weight: Tensor::ones(8, DType::F32, dev)?,
                eps: 1e-6,
            },
            lm_head: Linear::Dense {
                weight: Tensor::zeros((32, 8), DType::F32, dev)?,
                bias: None,
            },
            rope: Rope::new(4, 10_000.0, 64, DType::F32, dev)?,
            cfg: cfg(),
            dtype: DType::F32,
            device: dev.clone(),
        })
    }

    #[test]
    fn mask_blocks_future_and_padding() -> Result<()> {
        let dev = Device::Cpu;
        let model = bare_model(&dev)?;

        // Sequence 0 has 2 pad tokens on the left, sequence 1 has none.
        let mask = model.build_mask(2, 0, 4, 4, &[2, 0])?;
        let m = mask.i((.., 0, .., ..))?.to_vec3::<f32>()?;

        // Row 0 of the padded sequence can see nothing (all its visible
        // positions are pads), row 2 can see exactly position 2.
        assert!(m[0][2][0].is_infinite() && m[0][2][1].is_infinite());
        assert_eq!(m[0][2][2], 0.0);
        assert!(m[0][2][3].is_infinite(), "future must stay masked");

        // The unpadded sequence sees the full causal triangle.
        assert_eq!(m[1][2][0], 0.0);
        assert_eq!(m[1][2][1], 0.0);
        assert_eq!(m[1][2][2], 0.0);
        assert!(m[1][2][3].is_infinite());
        Ok(())
    }
}

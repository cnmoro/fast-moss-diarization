//! Whisper-Medium audio encoder (encoder half only).
//!
//! Every chunk fed here is exactly 30 s of log-mel, so the sequence length is a
//! constant 1500 after the stride-2 convolution and there is no padding mask to
//! carry: attention is dense and bidirectional over the whole window.

use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax_last_dim;

use crate::config::AudioConfig;
use crate::error::Result;
use crate::model::linear::{LayerNorm, Linear, Loader};

/// Whisper's fixed 1e-5 layer-norm epsilon.
const LN_EPS: f64 = 1e-5;

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(loader: &Loader, cfg: &AudioConfig) -> Result<Self> {
        let d = cfg.d_model;
        let head_dim = cfg.head_dim();
        Ok(Self {
            // The encoder is compute-bound rather than bandwidth-bound and runs
            // once per 30 s of audio, so it is left dense even in int8 mode.
            q: loader.linear_dense(d, d, "self_attn.q_proj", true)?,
            // Whisper's k_proj uniquely has no bias.
            k: loader.linear_dense(d, d, "self_attn.k_proj", false)?,
            v: loader.linear_dense(d, d, "self_attn.v_proj", true)?,
            out: loader.linear_dense(d, d, "self_attn.out_proj", true)?,
            heads: cfg.encoder_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.25),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let split = |x: Tensor| -> Result<Tensor> {
            Ok(x.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };

        // Whisper folds the 1/sqrt(d) into both q and k as d^-1/4, matching the
        // reference implementation's numerics rather than scaling the product.
        let q = split((self.q.forward(xs)? * self.scale)?)?;
        let k = split((self.k.forward(xs)? * self.scale)?)?;
        let v = split(self.v.forward(xs)?)?;

        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.out.forward(&ctx)
    }
}

struct EncoderLayer {
    attn: Attention,
    attn_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(loader: &Loader, cfg: &AudioConfig) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(loader, cfg)?,
            attn_norm: LayerNorm::load(loader, cfg.d_model, "self_attn_layer_norm", LN_EPS)?,
            fc1: loader.linear_dense(cfg.d_model, cfg.encoder_ffn_dim, "fc1", true)?,
            fc2: loader.linear_dense(cfg.encoder_ffn_dim, cfg.d_model, "fc2", true)?,
            final_norm: LayerNorm::load(loader, cfg.d_model, "final_layer_norm", LN_EPS)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-norm residual blocks.
        let residual = xs;
        let h = self.attn.forward(&self.attn_norm.forward(xs)?)?;
        let xs = (residual + h)?;

        let residual = &xs;
        let h = self.final_norm.forward(&xs)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        Ok((residual + h)?)
    }
}

pub struct WhisperEncoder {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    positions: Tensor,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    /// Output frames per chunk (1500 for a 30 s window).
    pub out_frames: usize,
    pub d_model: usize,
}

impl WhisperEncoder {
    pub fn load(loader: &Loader, cfg: &AudioConfig) -> Result<Self> {
        let d = cfg.d_model;
        let enc = loader.sub("model.whisper_encoder");

        let layers = (0..cfg.encoder_layers)
            .map(|i| EncoderLayer::load(&enc.sub(&format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            conv1_w: enc.get((d, cfg.num_mel_bins, 3), "conv1.weight")?,
            conv1_b: enc.get(d, "conv1.bias")?,
            conv2_w: enc.get((d, d, 3), "conv2.weight")?,
            conv2_b: enc.get(d, "conv2.bias")?,
            positions: enc.get((cfg.max_source_positions, d), "embed_positions.weight")?,
            layers,
            norm: LayerNorm::load(&enc, d, "layer_norm", LN_EPS)?,
            out_frames: cfg.max_source_positions,
            d_model: d,
        })
    }

    pub fn device(&self) -> &Device {
        self.positions.device()
    }

    /// Encode a batch of log-mel chunks `(batch, n_mels, frames)` into
    /// `(batch, frames / 2, d_model)`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // conv1 keeps the frame rate; conv2 halves it, taking 3000 mel frames
        // down to the 1500 positions the encoder is trained on.
        let xs = mel
            .conv1d(&self.conv1_w, 1, 1, 1, 1)?
            .broadcast_add(&self.conv1_b.reshape((1, (), 1))?)?
            .gelu_erf()?;
        let xs = xs
            .conv1d(&self.conv2_w, 1, 2, 1, 1)?
            .broadcast_add(&self.conv2_b.reshape((1, (), 1))?)?
            .gelu_erf()?;

        // (b, d, t) -> (b, t, d), then add the learned sinusoidal positions.
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let t = xs.dim(1)?;
        let mut xs = xs.broadcast_add(&self.positions.narrow(0, 0, t)?)?;

        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::AudioConfig;

    #[test]
    fn head_dim_divides_the_model_width() {
        let cfg = AudioConfig {
            num_mel_bins: 80,
            d_model: 1024,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            max_source_positions: 1500,
        };
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.head_dim() * cfg.encoder_attention_heads, cfg.d_model);
    }
}

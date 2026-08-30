use serde::Deserialize;

/// Qwen3 text backbone hyper-parameters.
///
/// Note `head_dim` is explicit and is *not* `hidden_size / num_attention_heads`:
/// the 0.6B backbone uses 16 heads of 128 dims over a 1024-wide residual stream,
/// so the q/o projections change width.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_head_dim() -> usize {
    128
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_max_position_embeddings() -> usize {
    131_072
}

impl TextConfig {
    /// Width of the concatenated query projection.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Width of each of the key and value projections.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// How many query heads share one key/value head.
    pub fn kv_repeat(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Whisper encoder hyper-parameters. Only the encoder half is ever loaded.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub num_mel_bins: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub max_source_positions: usize,
}

impl AudioConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
}

/// Top-level `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    pub audio_config: AudioConfig,
    pub audio_token_id: u32,
    pub audio_merge_size: usize,
    pub adaptor_input_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

/// `generation_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

fn default_max_new_tokens() -> usize {
    5120
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            bos_token_id: Some(151_643),
            eos_token_id: Some(151_645),
            pad_token_id: Some(151_643),
            max_new_tokens: default_max_new_tokens(),
        }
    }
}

/// `preprocessor_config.json` — the Whisper log-mel frontend.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureConfig {
    #[serde(default = "default_feature_size")]
    pub feature_size: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_n_samples")]
    pub n_samples: usize,
    #[serde(default = "default_nb_max_frames")]
    pub nb_max_frames: usize,
}

fn default_feature_size() -> usize {
    80
}
fn default_sampling_rate() -> usize {
    16_000
}
fn default_hop_length() -> usize {
    160
}
fn default_n_fft() -> usize {
    400
}
fn default_n_samples() -> usize {
    480_000
}
fn default_nb_max_frames() -> usize {
    3_000
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            feature_size: default_feature_size(),
            sampling_rate: default_sampling_rate(),
            hop_length: default_hop_length(),
            n_fft: default_n_fft(),
            n_samples: default_n_samples(),
            nb_max_frames: default_nb_max_frames(),
        }
    }
}

/// `processor_config.json` — how the audio placeholder span is built.
#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    #[serde(default = "default_tokens_per_second")]
    pub audio_tokens_per_second: f64,
    #[serde(default = "default_merge_size")]
    pub audio_merge_size: usize,
    #[serde(default = "default_time_marker_every")]
    pub time_marker_every_seconds: usize,
    #[serde(default = "default_enable_time_marker")]
    pub enable_time_marker: bool,
}

fn default_tokens_per_second() -> f64 {
    12.5
}
fn default_merge_size() -> usize {
    4
}
fn default_time_marker_every() -> usize {
    5
}
fn default_enable_time_marker() -> bool {
    true
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            audio_tokens_per_second: default_tokens_per_second(),
            audio_merge_size: default_merge_size(),
            time_marker_every_seconds: default_time_marker_every(),
            enable_time_marker: default_enable_time_marker(),
        }
    }
}

/// The stride, in raw samples, that one post-merge audio token covers.
///
/// The Whisper conv stack downsamples by 2 and the adaptor merges 4 frames, so
/// one token spans `hop_length * 2 * merge_size` samples (1280 at 16 kHz, i.e.
/// 12.5 tokens per second).
pub const WHISPER_ENCODER_STRIDE: usize = 2;

pub fn audio_token_stride(hop_length: usize, merge_size: usize) -> usize {
    hop_length * WHISPER_ENCODER_STRIDE * merge_size
}

/// Number of audio tokens a chunk of `num_samples` raw samples expands to.
pub fn audio_token_length(num_samples: usize, hop_length: usize, merge_size: usize) -> usize {
    if num_samples == 0 {
        return 0;
    }
    let stride = audio_token_stride(hop_length, merge_size);
    (num_samples - 1) / stride + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_chunk_is_375_tokens() {
        // 30 s at 16 kHz through the 1280-sample stride: 12.5 tokens/second.
        assert_eq!(audio_token_length(480_000, 160, 4), 375);
        assert_eq!(audio_token_stride(160, 4), 1280);
    }

    #[test]
    fn partial_chunks_round_up() {
        assert_eq!(audio_token_length(0, 160, 4), 0);
        assert_eq!(audio_token_length(1, 160, 4), 1);
        assert_eq!(audio_token_length(1280, 160, 4), 1);
        assert_eq!(audio_token_length(1281, 160, 4), 2);
    }

    #[test]
    fn text_config_widths() {
        let cfg = TextConfig {
            vocab_size: 151_936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 131_072,
            tie_word_embeddings: true,
            attention_bias: false,
        };
        // 16 heads x 128 dims = 2048, wider than the 1024 residual stream.
        assert_eq!(cfg.q_dim(), 2048);
        assert_eq!(cfg.kv_dim(), 1024);
        assert_eq!(cfg.kv_repeat(), 2);
    }
}

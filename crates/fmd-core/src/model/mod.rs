//! The assembled multimodal model: Whisper encoder -> adaptor -> Qwen3.

pub mod adaptor;
pub mod linear;
pub mod qwen3;
pub mod whisper_encoder;

use candle_core::{DType, Device, Tensor};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::precision::Precision;

pub use adaptor::VqAdaptor;
pub use linear::{Linear, Loader};
pub use qwen3::{KvCache, Qwen3};
pub use whisper_encoder::WhisperEncoder;

/// How many 30 s chunks to push through the encoder at once.
///
/// The encoder's activations are ~1500 x 4096 per chunk per layer, so a very
/// large batch buys little and risks an out-of-memory on smaller cards.
pub const DEFAULT_ENCODER_BATCH: usize = 16;

pub struct MossModel {
    pub encoder: WhisperEncoder,
    pub adaptor: VqAdaptor,
    pub lm: Qwen3,
    pub cfg: ModelConfig,
    pub precision: Precision,
    device: Device,
    dtype: DType,
}

impl MossModel {
    pub fn load(
        weights: &[std::path::PathBuf],
        cfg: &ModelConfig,
        precision: Precision,
        device: &Device,
        max_len: usize,
    ) -> Result<Self> {
        precision.validate_for(device)?;
        // SAFETY: the checkpoint files are not mutated for the process lifetime.
        let loader = unsafe { Loader::from_safetensors(weights, precision, device)? };

        let encoder = WhisperEncoder::load(&loader, &cfg.audio_config)?;
        let adaptor = VqAdaptor::load(
            &loader,
            cfg.adaptor_input_dim,
            cfg.text_config.hidden_size,
            cfg.audio_merge_size,
            cfg.text_config.rms_norm_eps,
        )?;
        let lm = Qwen3::load(&loader, &cfg.text_config, max_len)?;

        Ok(Self {
            encoder,
            adaptor,
            lm,
            cfg: cfg.clone(),
            precision,
            device: device.clone(),
            dtype: precision.activation_dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encode all mel chunks and reassemble them into one audio-embedding
    /// sequence per source file.
    ///
    /// `mel` is `(n_chunks, n_mels, frames)`; chunks from different files are
    /// interleaved according to `chunk_mapping`, which is what makes a single
    /// batched encoder pass possible.
    pub fn encode_audio(
        &self,
        mel: &Tensor,
        chunk_token_lengths: &[usize],
        chunk_mapping: &[usize],
        num_audios: usize,
        encoder_batch: usize,
    ) -> Result<Vec<Tensor>> {
        let n_chunks = mel.dim(0)?;
        debug_assert_eq!(n_chunks, chunk_token_lengths.len());

        let mut frames: Vec<Tensor> = Vec::with_capacity(n_chunks);
        let mut start = 0;
        while start < n_chunks {
            let take = encoder_batch.max(1).min(n_chunks - start);
            let batch = mel.narrow(0, start, take)?.contiguous()?;
            let encoded = self.encoder.forward(&batch)?;
            for i in 0..take {
                frames.push(encoded.narrow(0, i, 1)?);
            }
            start += take;
        }

        // Each chunk contributes `tokens * merge_size` encoder frames; a partial
        // final chunk contributes fewer, and the rest is padding to discard.
        let merge = self.adaptor.merge_size();
        let mut per_audio: Vec<Vec<Tensor>> = vec![Vec::new(); num_audios];
        for (idx, chunk) in frames.into_iter().enumerate() {
            let keep = chunk_token_lengths[idx] * merge;
            let trimmed = chunk.narrow(1, 0, keep.min(chunk.dim(1)?))?;
            per_audio[chunk_mapping[idx]].push(trimmed);
        }

        per_audio
            .into_iter()
            .map(|parts| {
                let joined = if parts.len() == 1 {
                    parts.into_iter().next().expect("length checked")
                } else {
                    Tensor::cat(&parts, 1)?
                };
                self.adaptor.merge_and_project(&joined)
            })
            .collect()
    }

    /// Build the prompt embedding for one sequence by splicing audio embeddings
    /// into the positions marked by the audio placeholder id.
    ///
    /// The placeholder positions are not contiguous: the prompt builder
    /// interleaves numeric time markers into the span, so the audio embeddings
    /// are scattered rather than pasted as one block.
    pub fn splice_audio(
        &self,
        token_ids: &[u32],
        audio_embeds: &Tensor,
        audio_token_id: u32,
    ) -> Result<Tensor> {
        let hidden = self.lm.hidden_size();
        let audio = audio_embeds.reshape(((), hidden))?.to_dtype(self.dtype)?;
        let n_audio = audio.dim(0)?;
        let n_slots = token_ids.iter().filter(|&&id| id == audio_token_id).count();
        if n_slots != n_audio {
            return Err(candle_core::Error::Msg(format!(
                "prompt has {n_slots} audio placeholders but the encoder produced \
                 {n_audio} embeddings"
            ))
            .into());
        }

        if n_slots == 0 {
            let ids = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), &self.device)?;
            return self.lm.embed(&ids);
        }

        // Walk the prompt as alternating runs of text and audio placeholders,
        // embedding text runs and slicing audio runs, then concatenate once.
        let mut parts: Vec<Tensor> = Vec::new();
        let mut audio_used = 0usize;
        let mut i = 0usize;
        while i < token_ids.len() {
            let is_audio = token_ids[i] == audio_token_id;
            let mut j = i;
            while j < token_ids.len() && (token_ids[j] == audio_token_id) == is_audio {
                j += 1;
            }
            if is_audio {
                let run = j - i;
                parts.push(audio.narrow(0, audio_used, run)?.unsqueeze(0)?);
                audio_used += run;
            } else {
                let ids = Tensor::from_vec(token_ids[i..j].to_vec(), (1, j - i), &self.device)?;
                parts.push(self.lm.embed(&ids)?);
            }
            i = j;
        }

        Ok(Tensor::cat(&parts, 1)?)
    }
}

//! Audio frontend: decode -> 30 s chunks -> log-mel features.

pub mod decode;
pub mod mel;

use std::path::Path;

use rayon::prelude::*;

use crate::config::{audio_token_length, FeatureConfig};
use crate::error::Result;

pub use decode::{decode_file, Waveform};
pub use mel::MelFrontend;

/// Mel features for a set of waveforms, flattened into one encoder batch.
///
/// Every chunk is exactly 30 s, so chunks from *different* audio files stack
/// into a single uniform tensor. That is what lets the Whisper encoder run one
/// large batched pass over an entire request set instead of one pass per file.
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// `[n_chunks * n_mels * n_frames]`, row-major.
    pub mel: Vec<f32>,
    pub n_chunks: usize,
    pub n_mels: usize,
    pub n_frames: usize,
    /// Post-merge audio tokens each chunk contributes (the last chunk of a file
    /// is usually partial).
    pub chunk_token_lengths: Vec<usize>,
    /// Which input waveform each chunk came from.
    pub chunk_mapping: Vec<usize>,
    /// Total audio tokens per input waveform.
    pub tokens_per_audio: Vec<usize>,
}

impl AudioFeatures {
    pub fn is_empty(&self) -> bool {
        self.n_chunks == 0
    }
}

/// Split waveforms into padded 30 s chunks and compute log-mel features.
///
/// `merge_size` is the adaptor's temporal merge factor and only affects the
/// reported token counts, not the mel maths.
pub fn extract_features(
    waveforms: &[Waveform],
    cfg: &FeatureConfig,
    frontend: &MelFrontend,
    merge_size: usize,
) -> AudioFeatures {
    let n_samples = cfg.n_samples;
    let n_frames = cfg.nb_max_frames;
    let n_mels = cfg.feature_size;

    // Plan the chunk layout first so the mel work can be done in parallel into
    // a preallocated buffer.
    struct Chunk<'a> {
        audio: usize,
        samples: &'a [f32],
        tokens: usize,
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut tokens_per_audio = vec![0usize; waveforms.len()];

    for (audio_idx, wav) in waveforms.iter().enumerate() {
        let total = wav.samples.len().max(1);
        let mut start = 0;
        while start < total {
            let end = (start + n_samples).min(wav.samples.len());
            let slice = &wav.samples[start..end];
            let tokens = audio_token_length(slice.len(), cfg.hop_length, merge_size);
            tokens_per_audio[audio_idx] += tokens;
            chunks.push(Chunk {
                audio: audio_idx,
                samples: slice,
                tokens,
            });
            start += n_samples;
        }
    }

    let n_chunks = chunks.len();
    let stride = n_mels * n_frames;
    let mut mel = vec![0f32; n_chunks * stride];

    mel.par_chunks_mut(stride)
        .zip(chunks.par_iter())
        .for_each(|(slot, chunk)| {
            // Every chunk is zero-padded out to the full 30 s window, exactly as
            // the reference processor does before calling the feature extractor.
            let mut padded = vec![0f32; n_samples];
            padded[..chunk.samples.len()].copy_from_slice(chunk.samples);
            let computed = frontend.log_mel(&padded);
            debug_assert_eq!(computed.len(), slot.len());
            slot.copy_from_slice(&computed);
        });

    AudioFeatures {
        mel,
        n_chunks,
        n_mels,
        n_frames,
        chunk_token_lengths: chunks.iter().map(|c| c.tokens).collect(),
        chunk_mapping: chunks.iter().map(|c| c.audio).collect(),
        tokens_per_audio,
    }
}

/// Decode several files in parallel.
pub fn decode_files<P>(paths: &[P], target_rate: u32) -> Vec<Result<Waveform>>
where
    P: AsRef<Path> + Sync,
{
    paths
        .par_iter()
        .map(|p| decode_file(p.as_ref(), target_rate))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32) -> Waveform {
        let n = (16_000.0 * seconds) as usize;
        Waveform {
            samples: (0..n)
                .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.3)
                .collect(),
            sample_rate: 16_000,
        }
    }

    #[test]
    fn short_audio_is_one_padded_chunk() {
        let cfg = FeatureConfig::default();
        let front = MelFrontend::new(&cfg);
        let feats = extract_features(&[tone(3.0)], &cfg, &front, 4);

        assert_eq!(feats.n_chunks, 1);
        assert_eq!(feats.mel.len(), 80 * 3000);
        // 3 s at 12.5 tokens/s = 37.5, rounded up to 38.
        assert_eq!(feats.chunk_token_lengths, vec![38]);
        assert_eq!(feats.tokens_per_audio, vec![38]);
        assert_eq!(feats.chunk_mapping, vec![0]);
    }

    #[test]
    fn long_audio_splits_into_full_and_partial_chunks() {
        let cfg = FeatureConfig::default();
        let front = MelFrontend::new(&cfg);
        // 70 s -> two full 30 s chunks plus a 10 s remainder.
        let feats = extract_features(&[tone(70.0)], &cfg, &front, 4);

        assert_eq!(feats.n_chunks, 3);
        assert_eq!(feats.chunk_token_lengths, vec![375, 375, 125]);
        assert_eq!(feats.tokens_per_audio, vec![875]);
        assert_eq!(feats.chunk_mapping, vec![0, 0, 0]);
    }

    #[test]
    fn chunks_from_several_files_stack_into_one_batch() {
        let cfg = FeatureConfig::default();
        let front = MelFrontend::new(&cfg);
        let feats = extract_features(&[tone(35.0), tone(5.0), tone(31.0)], &cfg, &front, 4);

        // 2 + 1 + 2 chunks, all the same shape, ready for a single encoder pass.
        assert_eq!(feats.n_chunks, 5);
        assert_eq!(feats.chunk_mapping, vec![0, 0, 1, 2, 2]);
        assert_eq!(feats.mel.len(), 5 * 80 * 3000);
        assert_eq!(feats.tokens_per_audio.len(), 3);
        assert_eq!(feats.tokens_per_audio[0], 375 + 63);
    }

    #[test]
    fn chunk_boundaries_do_not_bleed_between_files() {
        let cfg = FeatureConfig::default();
        let front = MelFrontend::new(&cfg);
        let loud = tone(1.0);
        let silent = Waveform {
            samples: vec![0f32; 16_000],
            sample_rate: 16_000,
        };
        let feats = extract_features(&[loud, silent], &cfg, &front, 4);

        let stride = 80 * 3000;
        let first = &feats.mel[..stride];
        let second = &feats.mel[stride..2 * stride];
        // Silence collapses to a constant floor; a tone does not.
        let second_flat = second.iter().all(|v| (v - second[0]).abs() < 1e-6);
        let first_flat = first.iter().all(|v| (v - first[0]).abs() < 1e-6);
        assert!(second_flat, "silent input should be flat");
        assert!(!first_flat, "tone input should not be flat");
    }
}

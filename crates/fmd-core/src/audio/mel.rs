//! Whisper log-mel frontend.
//!
//! This reproduces `transformers.WhisperFeatureExtractor` bit-for-bit closely
//! enough that encoder outputs match: a periodic Hann window, a centred STFT
//! with reflect padding, power spectrum, a Slaney-normalised mel filterbank,
//! then the log10 / dynamic-range-clamp / rescale tail.
//!
//! Getting this wrong is the classic way an otherwise correct port produces
//! plausible-but-wrong transcripts, so the filterbank maths is written out
//! explicitly rather than approximated.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use crate::config::FeatureConfig;

/// One triangular mel filter, stored as its non-zero span.
///
/// Mel filters are contiguous triangles over a few FFT bins each, so skipping
/// the zeros turns the filterbank projection from an 80x201 dense matmul into a
/// handful of multiply-adds per output.
#[derive(Debug, Clone)]
struct MelFilter {
    start: usize,
    weights: Vec<f32>,
}

pub struct MelFrontend {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    n_freq: usize,
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for MelFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelFrontend")
            .field("n_fft", &self.n_fft)
            .field("hop_length", &self.hop_length)
            .field("n_mels", &self.n_mels)
            .finish()
    }
}

impl MelFrontend {
    pub fn new(cfg: &FeatureConfig) -> Self {
        let n_freq = cfg.n_fft / 2 + 1;
        let filters = mel_filter_bank(
            n_freq,
            cfg.feature_size,
            0.0,
            cfg.sampling_rate as f32 / 2.0,
            cfg.sampling_rate as f32,
        );
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            n_fft: cfg.n_fft,
            hop_length: cfg.hop_length,
            n_mels: cfg.feature_size,
            n_freq,
            window: hann_periodic(cfg.n_fft),
            filters,
            fft: planner.plan_fft_forward(cfg.n_fft),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Number of frames `log_mel` will emit for a waveform of `n` samples.
    ///
    /// The centred STFT yields `n / hop + 1` frames and Whisper drops the last
    /// one, leaving exactly `n / hop`.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        num_samples / self.hop_length
    }

    /// Compute the log-mel spectrogram, returned row-major as `[n_mels][frames]`.
    pub fn log_mel(&self, samples: &[f32]) -> Vec<f32> {
        let n_frames = self.num_frames(samples.len());
        // The STFT is centred, so frame `t` is taken from `samples` starting at
        // `t * hop - n_fft/2`; reflect-pad rather than materialising a copy.
        let half = self.n_fft / 2;

        let mut out = vec![0f32; self.n_mels * n_frames];
        let mut frame = vec![0f32; self.n_fft];
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0f32; self.n_freq];
        let mut scratch = self.fft.make_scratch_vec();

        for t in 0..n_frames {
            let origin = (t * self.hop_length) as isize - half as isize;
            for (i, slot) in frame.iter_mut().enumerate() {
                let idx = reflect_index(origin + i as isize, samples.len());
                *slot = samples[idx] * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .expect("rfft buffer sizes are fixed at construction");

            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }

            for (m, filter) in self.filters.iter().enumerate() {
                let mut acc = 0f32;
                for (k, w) in filter.weights.iter().enumerate() {
                    acc += w * power[filter.start + k];
                }
                out[m * n_frames + t] = acc;
            }
        }

        // log10 with a floor, then Whisper's 80 dB dynamic-range clamp and the
        // affine rescale into roughly [-1, 1].
        let mut peak = f32::NEG_INFINITY;
        for v in out.iter_mut() {
            *v = v.max(1e-10).log10();
            if *v > peak {
                peak = *v;
            }
        }
        let floor = peak - 8.0;
        for v in out.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        out
    }
}

/// Index into `len` samples as if the signal were reflect-padded on both ends.
///
/// `np.pad(mode="reflect")` mirrors without repeating the edge sample, so index
/// -1 maps to 1 and index `len` maps to `len - 2`.
fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let mut i = idx.rem_euclid(period);
    if i >= len as isize {
        i = period - i;
    }
    i as usize
}

/// Periodic Hann window, matching `scipy.signal.get_window("hann", n, fftbins=True)`.
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (0.5 - 0.5 * x.cos()) as f32
        })
        .collect()
}

/// Slaney mel scale: linear below 1 kHz, logarithmic above.
fn hz_to_mel(freq: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep
    } else {
        freq / F_SP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Build the Slaney-normalised triangular mel filterbank.
///
/// Equivalent to `librosa.filters.mel(..., htk=False, norm="slaney")` and to
/// `transformers.audio_utils.mel_filter_bank(mel_scale="slaney", norm="slaney")`.
fn mel_filter_bank(
    n_freq: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    sample_rate: f32,
) -> Vec<MelFilter> {
    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|i| i as f32 * (sample_rate / 2.0) / (n_freq - 1) as f32)
        .collect();

    // n_mels + 2 band edges, evenly spaced on the mel scale.
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hz(mel)
        })
        .collect();

    (0..n_mels)
        .map(|m| {
            let (left, center, right) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
            // Slaney normalisation: unit *area* per filter, so wide
            // high-frequency bands are not louder than narrow low ones.
            let enorm = 2.0 / (right - left);

            let mut start = None;
            let mut weights = Vec::new();
            for (k, &f) in fft_freqs.iter().enumerate() {
                let lower = (f - left) / (center - left);
                let upper = (right - f) / (right - center);
                let w = lower.min(upper).max(0.0) * enorm;
                if w > 0.0 {
                    if start.is_none() {
                        start = Some(k);
                    }
                    weights.push(w);
                } else if start.is_some() {
                    // Triangles are contiguous; once we leave the support we are done.
                    break;
                }
            }
            MelFilter {
                start: start.unwrap_or(0),
                weights,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_round_trips() {
        for hz in [0.0f32, 100.0, 999.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "{hz} -> {back}");
        }
    }

    #[test]
    fn mel_breakpoint_is_continuous() {
        // The Slaney scale switches from linear to log at 1 kHz; the two
        // branches must agree there or the filterbank develops a seam.
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-4);
        assert!((hz_to_mel(999.999) - hz_to_mel(1000.0)).abs() < 1e-3);
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_edges() {
        // np.pad([0,1,2,3,4], 2, mode="reflect") -> [2,1,0,1,2,3,4,3,2]
        let len = 5;
        assert_eq!(reflect_index(-2, len), 2);
        assert_eq!(reflect_index(-1, len), 1);
        assert_eq!(reflect_index(0, len), 0);
        assert_eq!(reflect_index(4, len), 4);
        assert_eq!(reflect_index(5, len), 3);
        assert_eq!(reflect_index(6, len), 2);
    }

    #[test]
    fn hann_window_is_periodic_not_symmetric() {
        let w = hann_periodic(4);
        // Periodic Hann over 4 points: [0, 0.5, 1, 0.5]. A symmetric window
        // would end at 0 instead.
        assert!((w[0] - 0.0).abs() < 1e-6);
        assert!((w[1] - 0.5).abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
        assert!((w[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn filterbank_has_expected_shape_and_support() {
        let filters = mel_filter_bank(201, 80, 0.0, 8000.0, 16000.0);
        assert_eq!(filters.len(), 80);
        // Every filter must have some support, and they must march upward.
        let mut prev_start = 0;
        for f in &filters {
            assert!(!f.weights.is_empty());
            assert!(f.start >= prev_start);
            prev_start = f.start;
        }
        assert!(filters.last().unwrap().start + filters.last().unwrap().weights.len() <= 201);
    }

    #[test]
    fn frame_count_matches_whisper() {
        let front = MelFrontend::new(&FeatureConfig::default());
        // 30 s at 16 kHz -> exactly 3000 frames after dropping the trailing one.
        assert_eq!(front.num_frames(480_000), 3000);
        assert_eq!(front.n_mels(), 80);
    }

    #[test]
    fn log_mel_is_bounded_and_correctly_shaped() {
        let front = MelFrontend::new(&FeatureConfig::default());
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.5)
            .collect();
        let mel = front.log_mel(&samples);
        assert_eq!(mel.len(), 80 * 100);

        // The 80 dB clamp puts the floor exactly 8 log10-units below the peak,
        // which the /4 rescale turns into a fixed span of 2.0.
        let peak = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            (peak - floor - 2.0).abs() < 1e-4,
            "peak {peak} floor {floor}"
        );
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gain_shifts_the_spectrogram_by_a_constant() {
        // Because the tail is log10 then an affine rescale, multiplying the
        // waveform by g must shift every bin by 2*log10(g)/4 -- and the clamp,
        // being relative to the peak, must not disturb that.
        let front = MelFrontend::new(&FeatureConfig::default());
        let base: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.1)
            .collect();
        let loud: Vec<f32> = base.iter().map(|v| v * 10.0).collect();

        let a = front.log_mel(&base);
        let b = front.log_mel(&loud);
        let expected = 2.0 * 10f32.log10() / 4.0;
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((y - x - expected).abs() < 1e-3, "{x} -> {y}");
        }
    }

    /// Byte-level parity against `transformers.WhisperFeatureExtractor`.
    ///
    /// Regenerate the fixtures with `python scripts/dump_reference_mel.py testdata/`.
    /// Ignored by default so the suite stays runnable without them.
    #[test]
    #[ignore = "requires fixtures from scripts/dump_reference_mel.py"]
    fn matches_the_transformers_reference() {
        fn load(name: &str) -> Vec<f32> {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata");
            let bytes = std::fs::read(root.join(name))
                .unwrap_or_else(|e| panic!("missing fixture {name}: {e}"));
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        let input = load("mel_input.f32");
        let expected = load("mel_reference.f32");
        assert_eq!(input.len(), 480_000);
        assert_eq!(expected.len(), 80 * 3000);

        let got = MelFrontend::new(&FeatureConfig::default()).log_mel(&input);
        assert_eq!(got.len(), expected.len());

        let mut worst = 0f32;
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            if d > worst {
                worst = d;
            }
            assert!(d < 2e-3, "bin {i}: rust {g} vs reference {e}");
        }
        eprintln!("max absolute deviation from reference: {worst:e}");
    }

    #[test]
    fn silence_produces_a_flat_floor() {
        let front = MelFrontend::new(&FeatureConfig::default());
        let mel = front.log_mel(&vec![0f32; 16_000]);
        // All-zero input clamps to the 1e-10 floor everywhere, so every bin is equal.
        let first = mel[0];
        assert!(mel.iter().all(|v| (v - first).abs() < 1e-6));
    }
}

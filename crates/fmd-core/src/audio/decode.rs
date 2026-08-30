//! Container/codec decoding down to mono f32 at the model's sample rate.

use std::fs::File;
use std::path::Path;

use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::sample::SampleFormat;
use symphonia::core::audio::{Audio, GenericAudioBufferRef};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};

/// A decoded waveform: mono, f32, at `sample_rate`.
#[derive(Debug, Clone)]
pub struct Waveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Waveform {
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Decode any container symphonia understands, downmix to mono, and resample to
/// `target_rate`.
pub fn decode_file(path: &Path, target_rate: u32) -> Result<Waveform> {
    let fail = |reason: String| Error::AudioDecode {
        path: path.display().to_string(),
        reason,
    };

    let file = File::open(path).map_err(|e| fail(format!("cannot open file: {e}")))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| fail(format!("unrecognised container: {e}")))?;

    let track = format
        .default_track(symphonia::core::formats::TrackType::Audio)
        .ok_or_else(|| fail("file has no audio track".into()))?;
    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| fail("audio track has no codec parameters".into()))?
        .clone();

    let source_rate = codec_params
        .sample_rate
        .ok_or_else(|| fail("audio track does not declare a sample rate".into()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &AudioDecoderOptions::default())
        .map_err(|e| fail(format!("no decoder for this codec: {e}")))?;

    let mut mono = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(fail(format!("read error: {e}"))),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => append_mono(&buf, &mut mono),
            // A corrupt packet mid-stream should not abandon the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(fail(format!("decode error: {e}"))),
        }
    }

    if mono.is_empty() {
        return Err(fail("decoded zero audio samples".into()));
    }

    let samples = if source_rate == target_rate {
        mono
    } else {
        resample(&mono, source_rate, target_rate).map_err(|e| fail(e.to_string()))?
    };

    Ok(Waveform {
        samples,
        sample_rate: target_rate,
    })
}

/// Average all channels into a single mono track, converting to f32.
fn append_mono(buf: &GenericAudioBufferRef<'_>, out: &mut Vec<f32>) {
    macro_rules! mix {
        ($b:expr, $conv:expr) => {{
            let b = $b;
            let channels = b.spec().channels().count().max(1);
            let frames = b.frames();
            out.reserve(frames);
            let conv = $conv;
            for f in 0..frames {
                let mut acc = 0f32;
                for c in 0..channels {
                    acc += conv(b.plane(c).map(|p| p[f]).unwrap_or_default());
                }
                out.push(acc / channels as f32);
            }
        }};
    }

    match buf {
        GenericAudioBufferRef::F32(b) => mix!(b, |v: f32| v),
        GenericAudioBufferRef::F64(b) => mix!(b, |v: f64| v as f32),
        GenericAudioBufferRef::S32(b) => mix!(b, |v: i32| v as f32 / i32::MAX as f32),
        GenericAudioBufferRef::U32(b) => {
            mix!(b, |v: u32| (v as f32 - 2147483648.0) / 2147483648.0)
        }
        GenericAudioBufferRef::S24(b) => {
            mix!(b, |v: symphonia::core::audio::sample::i24| v.inner() as f32
                / 8388608.0)
        }
        GenericAudioBufferRef::U24(b) => {
            mix!(b, |v: symphonia::core::audio::sample::u24| (v.inner()
                as f32
                - 8388608.0)
                / 8388608.0)
        }
        GenericAudioBufferRef::S16(b) => mix!(b, |v: i16| v as f32 / 32768.0),
        GenericAudioBufferRef::U16(b) => mix!(b, |v: u16| (v as f32 - 32768.0) / 32768.0),
        GenericAudioBufferRef::S8(b) => mix!(b, |v: i8| v as f32 / 128.0),
        GenericAudioBufferRef::U8(b) => mix!(b, |v: u8| (v as f32 - 128.0) / 128.0),
    }
}

/// Resample caller-supplied samples, for audio that did not come from a file.
pub fn resample_public(input: &[f32], from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    resample(input, from, to)
}

/// Band-limited resampling via an FFT-based polyphase filter.
fn resample(input: &[f32], from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    // A chunk of ~1 s balances FFT efficiency against peak memory.
    let chunk = from as usize;
    let mut resampler = FftFixedIn::<f32>::new(from as usize, to as usize, chunk, 2, 1)?;

    let mut out = Vec::with_capacity(input.len() * to as usize / from as usize + chunk);
    let mut pos = 0;
    let mut buf = vec![0f32; chunk];
    while pos < input.len() {
        let take = chunk.min(input.len() - pos);
        buf[..take].copy_from_slice(&input[pos..pos + take]);
        // The final partial chunk is zero-filled; the trailing silence is
        // trimmed below by the exact output-length calculation.
        buf[take..].fill(0.0);
        let done = resampler.process(&[&buf], None)?;
        out.extend_from_slice(&done[0]);
        pos += take;
    }

    let expected = (input.len() as u64 * to as u64 / from as u64) as usize;
    out.truncate(expected.min(out.len()));
    Ok(out)
}

/// Whether symphonia is likely to handle this extension. Used only for nicer
/// error messages; decoding is still attempted for unknown extensions.
pub fn is_known_extension(path: &Path) -> bool {
    const KNOWN: &[&str] = &[
        "wav", "wave", "mp3", "flac", "ogg", "oga", "opus", "m4a", "mp4", "aac", "caf", "mkv",
        "webm", "mov", "aiff", "aif", "alac",
    ];
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| KNOWN.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Sample formats symphonia can hand back, listed for documentation purposes.
pub(crate) const _SUPPORTED_FORMATS: &[SampleFormat] = &[
    SampleFormat::U8,
    SampleFormat::S16,
    SampleFormat::S32,
    SampleFormat::F32,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_sniffing() {
        assert!(is_known_extension(Path::new("a.wav")));
        assert!(is_known_extension(Path::new("a.MP3")));
        assert!(!is_known_extension(Path::new("a.txt")));
        assert!(!is_known_extension(Path::new("noext")));
    }

    #[test]
    fn resampling_changes_length_proportionally() {
        let input: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = resample(&input, 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn resampling_preserves_a_pure_tone() {
        // A 440 Hz tone downsampled 48k -> 16k should keep roughly its amplitude.
        let input: Vec<f32> = (0..48_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin())
            .collect();
        let out = resample(&input, 48_000, 16_000).unwrap();
        // Ignore filter warm-up at the very start.
        let peak = out[2000..14_000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.9 && peak < 1.1, "peak was {peak}");
    }

    #[test]
    fn missing_file_reports_the_path() {
        let err = decode_file(Path::new("/nonexistent/audio.wav"), 16_000).unwrap_err();
        assert!(err.to_string().contains("/nonexistent/audio.wav"));
    }
}

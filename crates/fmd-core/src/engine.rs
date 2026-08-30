//! The batching inference engine.
//!
//! Throughput here comes from batching at two separate places, because the two
//! halves of the model have opposite shapes:
//!
//! * The **audio encoder** sees fixed 30 s windows, so chunks from *different*
//!   files stack into one uniform tensor. A hundred short clips become a single
//!   large encoder pass.
//! * The **decoder** sees variable-length prompts, so requests are sorted by
//!   prompt length and cut into micro-batches. Grouping similar lengths keeps
//!   both the padding waste and the KV cache -- which is sized by the longest
//!   member of a batch -- small.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::audio::{self, MelFrontend, Waveform};
use crate::config::{FeatureConfig, GenerationConfig, ModelConfig, ProcessorConfig};
use crate::error::{Error, Result};
use crate::hub::ModelSource;
use crate::model::{MossModel, DEFAULT_ENCODER_BATCH};
use crate::precision::{default_precision, Precision};
use crate::prompt::PromptBuilder;
use crate::transcript::{self, Segment};

/// Which device to run on.
#[derive(Debug, Clone)]
pub enum DeviceSpec {
    /// First CUDA device if one exists, otherwise CPU.
    Auto,
    Cpu,
    Cuda(usize),
}

impl DeviceSpec {
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" | "" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda(0)),
            other => match other
                .strip_prefix("cuda:")
                .or_else(|| other.strip_prefix("gpu:"))
            {
                Some(idx) => idx
                    .parse()
                    .map(Self::Cuda)
                    .map_err(|_| Error::config(format!("bad device index in {spec:?}"))),
                None => Err(Error::config(format!("unrecognised device {spec:?}"))),
            },
        }
    }

    pub fn resolve(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(i) => Ok(Device::new_cuda(*i)?),
            Self::Auto => match Device::new_cuda(0) {
                Ok(d) => Ok(d),
                Err(_) => Ok(Device::Cpu),
            },
        }
    }
}

/// Engine construction options.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub source: ModelSource,
    pub device: DeviceSpec,
    /// `None` picks bf16 on GPU and fp32 on CPU.
    pub precision: Option<Precision>,
    /// Generation budget per request.
    pub max_new_tokens: usize,
    /// Upper bound on sequences decoded together. Raise for throughput, lower
    /// if the KV cache does not fit.
    pub max_batch_size: usize,
    /// 30 s windows pushed through the audio encoder at once.
    pub encoder_batch: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::default(),
            device: DeviceSpec::Auto,
            precision: None,
            max_new_tokens: 4096,
            max_batch_size: 8,
            encoder_batch: DEFAULT_ENCODER_BATCH,
        }
    }
}

/// Where a request's audio comes from.
#[derive(Debug, Clone)]
pub enum AudioSource {
    Path(PathBuf),
    /// Mono f32 samples at a known rate; resampled if it is not 16 kHz.
    Samples {
        samples: Vec<f32>,
        sample_rate: u32,
    },
}

/// One transcription request.
#[derive(Debug, Clone)]
pub struct Request {
    /// Caller-chosen identifier, echoed back on the response.
    pub id: String,
    pub source: AudioSource,
    /// Overrides the default transcribe-and-diarize instruction.
    pub prompt: Option<String>,
}

impl Request {
    pub fn from_path(id: impl Into<String>, path: impl AsRef<Path>) -> Self {
        Self {
            id: id.into(),
            source: AudioSource::Path(path.as_ref().to_path_buf()),
            prompt: None,
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted its end-of-turn token.
    Stop,
    /// The `max_new_tokens` budget ran out; the transcript may be truncated.
    Length,
}

/// A completed transcription.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Response {
    pub id: String,
    pub text: String,
    pub segments: Vec<Segment>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub audio_seconds: f64,
    pub finish_reason: FinishReason,
}

/// Timings for a whole `transcribe` call.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BatchStats {
    pub requests: usize,
    pub decode_seconds: f64,
    pub feature_seconds: f64,
    pub encode_seconds: f64,
    pub generate_seconds: f64,
    pub total_seconds: f64,
    pub audio_seconds: f64,
    pub generated_tokens: usize,
    pub micro_batches: usize,
}

impl BatchStats {
    /// Seconds of audio processed per wall-clock second.
    pub fn realtime_factor(&self) -> f64 {
        if self.total_seconds > 0.0 {
            self.audio_seconds / self.total_seconds
        } else {
            0.0
        }
    }
}

pub struct Engine {
    model: MossModel,
    prompt: PromptBuilder,
    mel: MelFrontend,
    feature_cfg: FeatureConfig,
    generation: GenerationConfig,
    cfg: EngineConfig,
    eos_id: u32,
    audio_token_id: u32,
}

impl Engine {
    /// Load the model, downloading the checkpoint if it is not cached.
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let files = cfg.source.resolve()?;
        let device = cfg.device.resolve()?;
        let precision = cfg.precision.unwrap_or_else(|| default_precision(&device));

        let read = |p: &Path| -> Result<String> {
            std::fs::read_to_string(p).map_err(|source| Error::Io {
                path: p.to_path_buf(),
                source,
            })
        };
        let parse = |what: &str, text: &str| -> Result<serde_json::Value> {
            serde_json::from_str(text).map_err(|source| Error::Json {
                what: what.to_string(),
                source,
            })
        };

        let model_cfg: ModelConfig =
            serde_json::from_value(parse("config.json", &read(&files.config)?)?).map_err(
                |source| Error::Json {
                    what: "config.json".into(),
                    source,
                },
            )?;
        let feature_cfg: FeatureConfig = serde_json::from_str(&read(&files.preprocessor)?)
            .map_err(|source| Error::Json {
                what: "preprocessor_config.json".into(),
                source,
            })?;
        let processor_cfg: ProcessorConfig = match &files.processor {
            Some(p) => serde_json::from_str(&read(p)?).map_err(|source| Error::Json {
                what: "processor_config.json".into(),
                source,
            })?,
            None => ProcessorConfig::default(),
        };
        let generation: GenerationConfig = match &files.generation {
            Some(p) => serde_json::from_str(&read(p)?).map_err(|source| Error::Json {
                what: "generation_config.json".into(),
                source,
            })?,
            None => GenerationConfig::default(),
        };

        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", files.tokenizer.display())))?;
        let prompt = PromptBuilder::new(tokenizer, processor_cfg)?;

        // RoPE tables must span the longest sequence the engine can ever build.
        let max_len = model_cfg.text_config.max_position_embeddings.min(
            // A 30-minute recording is ~22.5k prompt tokens; leave generous room
            // without allocating a 131k-row table.
            65_536,
        );

        let model = MossModel::load(&files.weights, &model_cfg, precision, &device, max_len)?;
        let eos_id = generation.eos_token_id.unwrap_or(151_645);
        let audio_token_id = model_cfg.audio_token_id;

        Ok(Self {
            mel: MelFrontend::new(&feature_cfg),
            model,
            prompt,
            feature_cfg,
            generation,
            cfg,
            eos_id,
            audio_token_id,
        })
    }

    pub fn precision(&self) -> Precision {
        self.model.precision
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// Transcribe one file.
    pub fn transcribe_file(&mut self, path: impl AsRef<Path>) -> Result<Response> {
        let req = Request::from_path("0", path);
        let (mut out, _) = self.transcribe(vec![req])?;
        out.remove(0)
    }

    /// Transcribe a batch.
    ///
    /// Results come back in the same order as `requests`, each carrying the
    /// caller's `id`. One failed request does not sink the batch: its slot holds
    /// the error and the rest still run.
    pub fn transcribe(
        &mut self,
        requests: Vec<Request>,
    ) -> Result<(Vec<Result<Response>>, BatchStats)> {
        let started = Instant::now();
        let mut stats = BatchStats {
            requests: requests.len(),
            ..Default::default()
        };
        let mut results: Vec<Option<Result<Response>>> =
            (0..requests.len()).map(|_| None).collect();
        if requests.is_empty() {
            return Ok((Vec::new(), stats));
        }

        // --- 1. Decode audio (parallel, CPU) --------------------------------
        let t = Instant::now();
        let target_rate = self.feature_cfg.sampling_rate as u32;
        let decoded = self.decode_all(&requests, target_rate);
        stats.decode_seconds = t.elapsed().as_secs_f64();

        let mut live: Vec<usize> = Vec::new();
        let mut waveforms: Vec<Waveform> = Vec::new();
        for (i, outcome) in decoded.into_iter().enumerate() {
            match outcome {
                Ok(w) => {
                    stats.audio_seconds += w.duration_seconds();
                    live.push(i);
                    waveforms.push(w);
                }
                Err(e) => results[i] = Some(Err(e)),
            }
        }
        if live.is_empty() {
            return Ok((finish(results), stats));
        }

        // --- 2. Log-mel features (parallel, CPU) ----------------------------
        let t = Instant::now();
        let feats = audio::extract_features(
            &waveforms,
            &self.feature_cfg,
            &self.mel,
            self.model.cfg.audio_merge_size,
        );
        stats.feature_seconds = t.elapsed().as_secs_f64();

        // --- 3. One batched encoder pass over every chunk -------------------
        let t = Instant::now();
        let mel = Tensor::from_vec(
            feats.mel.clone(),
            (feats.n_chunks, feats.n_mels, feats.n_frames),
            self.model.device(),
        )?
        .to_dtype(self.model.dtype())?;
        let audio_embeds = self.model.encode_audio(
            &mel,
            &feats.chunk_token_lengths,
            &feats.chunk_mapping,
            live.len(),
            self.cfg.encoder_batch,
        )?;
        stats.encode_seconds = t.elapsed().as_secs_f64();

        // --- 4. Build prompts ----------------------------------------------
        struct Pending {
            slot: usize,
            tokens: Vec<u32>,
            embeds: Tensor,
            audio_seconds: f64,
        }
        let mut pending: Vec<Pending> = Vec::new();
        for (k, &slot) in live.iter().enumerate() {
            let tokens = match self
                .prompt
                .build(feats.tokens_per_audio[k], requests[slot].prompt.as_deref())
            {
                Ok(t) => t,
                Err(e) => {
                    results[slot] = Some(Err(e));
                    continue;
                }
            };
            match self
                .model
                .splice_audio(&tokens, &audio_embeds[k], self.audio_token_id)
            {
                Ok(embeds) => pending.push(Pending {
                    slot,
                    tokens,
                    embeds,
                    audio_seconds: waveforms[k].duration_seconds(),
                }),
                Err(e) => results[slot] = Some(Err(e)),
            }
        }

        // --- 5. Decode in length-bucketed micro-batches ---------------------
        // Sorting by prompt length is what keeps padding and cache size down.
        let t = Instant::now();
        pending.sort_by_key(|p| p.tokens.len());

        for group in pending.chunks(self.cfg.max_batch_size.max(1)) {
            stats.micro_batches += 1;
            let prompts: Vec<&[u32]> = group.iter().map(|p| p.tokens.as_slice()).collect();
            let embeds: Vec<&Tensor> = group.iter().map(|p| &p.embeds).collect();

            match self.generate_batch(&prompts, &embeds) {
                Ok(outputs) => {
                    for (p, out) in group.iter().zip(outputs) {
                        stats.generated_tokens += out.tokens.len();
                        let text = match self.prompt.decode(&out.tokens) {
                            Ok(t) => t.trim().to_string(),
                            Err(e) => {
                                results[p.slot] = Some(Err(e));
                                continue;
                            }
                        };
                        results[p.slot] = Some(Ok(Response {
                            id: requests[p.slot].id.clone(),
                            segments: transcript::parse(&text),
                            text,
                            prompt_tokens: p.tokens.len(),
                            generated_tokens: out.tokens.len(),
                            audio_seconds: p.audio_seconds,
                            finish_reason: out.finish_reason,
                        }));
                    }
                }
                Err(e) => {
                    // A batch-level failure (usually an allocation) is reported
                    // against every request in that batch.
                    for p in group {
                        results[p.slot] =
                            Some(Err(Error::config(format!("batch generation failed: {e}"))));
                    }
                }
            }
        }
        stats.generate_seconds = t.elapsed().as_secs_f64();
        stats.total_seconds = started.elapsed().as_secs_f64();

        Ok((finish(results), stats))
    }

    fn decode_all(&self, requests: &[Request], target_rate: u32) -> Vec<Result<Waveform>> {
        use rayon::prelude::*;
        requests
            .par_iter()
            .map(|req| match &req.source {
                AudioSource::Path(p) => audio::decode_file(p, target_rate),
                AudioSource::Samples {
                    samples,
                    sample_rate,
                } => {
                    if *sample_rate == target_rate {
                        Ok(Waveform {
                            samples: samples.clone(),
                            sample_rate: target_rate,
                        })
                    } else {
                        // Route through the same resampler the file path uses.
                        crate::audio::decode::resample_public(samples, *sample_rate, target_rate)
                            .map(|samples| Waveform {
                                samples,
                                sample_rate: target_rate,
                            })
                            .map_err(|e| Error::AudioDecode {
                                path: format!("<{} samples in memory>", samples.len()),
                                reason: e.to_string(),
                            })
                    }
                }
            })
            .collect()
    }

    /// Greedy generation over a left-padded batch.
    fn generate_batch(&self, prompts: &[&[u32]], embeds: &[&Tensor]) -> Result<Vec<Generated>> {
        let batch = prompts.len();
        let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let budget = self
            .cfg
            .max_new_tokens
            .min(self.generation.max_new_tokens.max(1));
        let capacity = max_prompt + budget;

        let device = self.model.device();
        let dtype = self.model.dtype();
        let hidden = self.model.lm.hidden_size();

        // Left-pad every prompt to a common length. Positions stay shared
        // across the batch; only the mask distinguishes real tokens from pads.
        let pad_lengths: Vec<usize> = prompts.iter().map(|p| max_prompt - p.len()).collect();
        let padded: Vec<Tensor> = embeds
            .iter()
            .zip(&pad_lengths)
            .map(|(e, &pad)| -> Result<Tensor> {
                if pad == 0 {
                    Ok((*e).clone())
                } else {
                    let filler = Tensor::zeros((1, pad, hidden), dtype, device)?;
                    Ok(Tensor::cat(&[&filler, *e], 1)?)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let batched = Tensor::cat(&padded, 0)?.contiguous()?;

        let mut caches = self.model.lm.new_caches(batch, capacity)?;
        let hidden_last = self.model.lm.prefill(&batched, &mut caches, &pad_lengths)?;
        let mut logits = self.model.lm.logits(&hidden_last)?;

        let mut outputs: Vec<Generated> = (0..batch)
            .map(|_| Generated {
                tokens: Vec::new(),
                finish_reason: FinishReason::Length,
            })
            .collect();
        let mut done = vec![false; batch];

        for step in 0..budget {
            let next = argmax_rows(&logits)?;

            let mut all_done = true;
            for (i, &tok) in next.iter().enumerate() {
                if done[i] {
                    continue;
                }
                if tok == self.eos_id {
                    done[i] = true;
                    outputs[i].finish_reason = FinishReason::Stop;
                } else {
                    outputs[i].tokens.push(tok);
                    all_done = false;
                }
            }
            if all_done || step + 1 == budget {
                break;
            }

            // Finished sequences keep stepping so the batch stays rectangular,
            // but they are fed padding and their output is discarded.
            let feed: Vec<u32> = next
                .iter()
                .enumerate()
                .map(|(i, &t)| if done[i] { self.eos_id } else { t })
                .collect();
            let ids = Tensor::from_vec(feed, (batch, 1), device)?;
            let offset = caches[0].len();
            logits = self
                .model
                .lm
                .decode_step(&ids, &mut caches, offset, &pad_lengths)?;
        }

        Ok(outputs)
    }
}

struct Generated {
    tokens: Vec<u32>,
    finish_reason: FinishReason,
}

/// Greedy pick per row of a `(b, 1, vocab)` or `(b, vocab)` logit tensor.
fn argmax_rows(logits: &Tensor) -> Result<Vec<u32>> {
    let flat = if logits.rank() == 3 {
        let (b, t, v) = logits.dims3()?;
        logits.narrow(1, t - 1, 1)?.reshape((b, v))?
    } else {
        logits.clone()
    };
    let ids = flat.to_dtype(DType::F32)?.argmax(1)?;
    Ok(ids.to_vec1::<u32>()?)
}

fn finish(results: Vec<Option<Result<Response>>>) -> Vec<Result<Response>> {
    results
        .into_iter()
        .map(|r| {
            r.unwrap_or_else(|| {
                Err(Error::config(
                    "request produced no result (internal scheduling bug)",
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_spec_parsing() {
        assert!(matches!(
            DeviceSpec::parse("auto").unwrap(),
            DeviceSpec::Auto
        ));
        assert!(matches!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu));
        assert!(matches!(
            DeviceSpec::parse("cuda").unwrap(),
            DeviceSpec::Cuda(0)
        ));
        assert!(matches!(
            DeviceSpec::parse("cuda:1").unwrap(),
            DeviceSpec::Cuda(1)
        ));
        assert!(matches!(
            DeviceSpec::parse("GPU:3").unwrap(),
            DeviceSpec::Cuda(3)
        ));
        assert!(DeviceSpec::parse("tpu").is_err());
        assert!(DeviceSpec::parse("cuda:x").is_err());
    }

    #[test]
    fn argmax_picks_the_largest_logit_per_row() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.9, 0.2, /* row 2 */ 5.0, 1.0, 2.0],
            (2, 3),
            &dev,
        )?;
        assert_eq!(argmax_rows(&logits)?, vec![1, 0]);

        // The 3-D form must read the final position only.
        let logits3 = Tensor::from_vec(vec![9f32, 0., 0., 0., 0., 7.], (1, 2, 3), &dev)?;
        assert_eq!(argmax_rows(&logits3)?, vec![2]);
        Ok(())
    }

    #[test]
    fn realtime_factor_reports_speedup() {
        let stats = BatchStats {
            audio_seconds: 600.0,
            total_seconds: 30.0,
            ..Default::default()
        };
        assert_eq!(stats.realtime_factor(), 20.0);
        assert_eq!(BatchStats::default().realtime_factor(), 0.0);
    }

    #[test]
    fn missing_results_become_errors_not_panics() {
        let out = finish(vec![None, Some(Err(Error::config("boom")))]);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_err());
        assert!(out[1].is_err());
    }
}

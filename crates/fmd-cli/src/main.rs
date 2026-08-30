use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use fmd_core::{transcript, DeviceSpec, Engine, EngineConfig, ModelSource, Precision, Request};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    /// Human-readable segments.
    Text,
    /// One JSON document with every result.
    Json,
    /// SubRip subtitles (only meaningful for a single input).
    Srt,
}

#[derive(Parser, Debug)]
#[command(
    name = "fmd",
    about = "Fast GPU transcription and diarization for MOSS-Transcribe-Diarize",
    version
)]
struct Args {
    /// Audio or video files. Several inputs are batched together.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Hugging Face repo id or a local checkpoint directory.
    #[arg(long, default_value = fmd_core::DEFAULT_MODEL_ID)]
    model: String,

    /// Checkpoint revision, when loading from the hub.
    #[arg(long)]
    revision: Option<String>,

    /// fp32, fp16, bf16 or int8. Defaults to bf16 on GPU and fp32 on CPU.
    #[arg(long)]
    dtype: Option<String>,

    /// auto, cpu, cuda, or cuda:N.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Generation budget per input.
    #[arg(long, default_value_t = 4096)]
    max_new_tokens: usize,

    /// Sequences decoded together. Lower this if the KV cache will not fit.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// 30 s windows pushed through the audio encoder at once.
    #[arg(long, default_value_t = 16)]
    encoder_batch: usize,

    /// Override the transcribe-and-diarize instruction.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Print timings to stderr.
    #[arg(long)]
    stats: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let precision = args
        .dtype
        .as_deref()
        .map(Precision::parse)
        .transpose()
        .context("bad --dtype")?;

    let cfg = EngineConfig {
        source: ModelSource::parse(&args.model).with_revision(args.revision.clone()),
        device: DeviceSpec::parse(&args.device).context("bad --device")?,
        precision,
        max_new_tokens: args.max_new_tokens,
        max_batch_size: args.batch_size,
        encoder_batch: args.encoder_batch,
    };

    eprintln!("loading {} ...", args.model);
    let load_started = std::time::Instant::now();
    let mut engine = Engine::new(cfg).context("failed to load the model")?;
    eprintln!(
        "ready on {:?} in {} at {:.1}s",
        engine.device(),
        engine.precision(),
        load_started.elapsed().as_secs_f64()
    );

    let requests: Vec<Request> = args
        .inputs
        .iter()
        .map(|path| Request {
            id: path.display().to_string(),
            source: fmd_core::AudioSource::Path(path.clone()),
            prompt: args.prompt.clone(),
        })
        .collect();

    let (results, stats) = engine.transcribe(requests)?;

    match args.format {
        Format::Text => {
            for result in &results {
                match result {
                    Ok(r) => {
                        println!("=== {} ===", r.id);
                        for seg in &r.segments {
                            println!(
                                "[{:>8.2} - {:>8.2}] {}: {}",
                                seg.start, seg.end, seg.speaker, seg.text
                            );
                        }
                        if r.segments.is_empty() {
                            println!("{}", r.text);
                        }
                        println!();
                    }
                    Err(e) => eprintln!("error: {e}"),
                }
            }
        }
        Format::Json => {
            let payload: Vec<serde_json::Value> = results
                .iter()
                .map(|r| match r {
                    Ok(r) => serde_json::to_value(r).unwrap_or(serde_json::Value::Null),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "results": payload,
                    "stats": stats,
                }))?
            );
        }
        Format::Srt => {
            for result in &results {
                match result {
                    Ok(r) => print!("{}", transcript::to_srt(&r.segments, true)),
                    Err(e) => eprintln!("error: {e}"),
                }
            }
        }
    }

    if args.stats {
        eprintln!(
            "\n{} request(s) in {} micro-batch(es)\n\
             decode {:.2}s | features {:.2}s | encoder {:.2}s | generate {:.2}s | total {:.2}s\n\
             {:.1}s of audio, {} tokens generated, {:.1}x realtime",
            stats.requests,
            stats.micro_batches,
            stats.decode_seconds,
            stats.feature_seconds,
            stats.encode_seconds,
            stats.generate_seconds,
            stats.total_seconds,
            stats.audio_seconds,
            stats.generated_tokens,
            stats.realtime_factor(),
        );
    }

    // A failed request is a failed run, even if others succeeded.
    if results.iter().any(|r| r.is_err()) {
        std::process::exit(1);
    }
    Ok(())
}

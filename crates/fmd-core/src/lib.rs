//! Fast GPU inference engine for MOSS-Transcribe-Diarize.
//!
//! See [`engine::Engine`] for the entry point.

pub mod audio;
pub mod config;
pub mod engine;
pub mod error;
pub mod hub;
pub mod model;
pub mod precision;
pub mod prompt;
pub mod transcript;

pub use engine::{
    AudioSource, BatchStats, DeviceSpec, Engine, EngineConfig, FinishReason, Request, Response,
};
pub use error::{Error, Result};
pub use hub::{ModelSource, DEFAULT_MODEL_ID};
pub use precision::Precision;
pub use prompt::DEFAULT_PROMPT;
pub use transcript::Segment;

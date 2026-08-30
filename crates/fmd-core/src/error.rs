use std::path::PathBuf;

/// Errors surfaced by the engine.
///
/// Variants are deliberately coarse: callers almost always either retry the
/// whole request or give up, and the Python binding flattens everything into a
/// single exception type anyway.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor op failed: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to download {file} from {repo}: {source}")]
    Hub {
        repo: String,
        file: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("could not parse {what}: {source}")]
    Json {
        what: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("audio decode failed for {path}: {reason}")]
    AudioDecode { path: String, reason: String },

    #[error("unsupported dtype {0:?}; expected one of fp32, fp16, bf16, int8")]
    UnsupportedDtype(String),

    #[error("{0}")]
    Config(String),

    #[error("request {id} exceeded the {limit} token generation budget")]
    LengthLimit { id: String, limit: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    pub(crate) fn tokenizer(err: impl std::fmt::Display) -> Self {
        Error::Tokenizer(err.to_string())
    }
}

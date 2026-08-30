use std::path::{Path, PathBuf};

use hf_hub::HFClientSync;

use crate::error::{Error, Result};

/// The upstream checkpoint this engine targets.
pub const DEFAULT_MODEL_ID: &str = "OpenMOSS-Team/MOSS-Transcribe-Diarize";

/// Files the engine needs. `model.safetensors.index.json` is absent on
/// single-shard checkpoints, so it is fetched opportunistically.
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "preprocessor_config.json"];

const OPTIONAL_FILES: &[&str] = &[
    "generation_config.json",
    "processor_config.json",
    "model.safetensors.index.json",
];

/// Every file of a resolved checkpoint, on local disk.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub root: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub preprocessor: PathBuf,
    pub generation: Option<PathBuf>,
    pub processor: Option<PathBuf>,
    /// Every safetensors shard, in index order.
    pub weights: Vec<PathBuf>,
}

/// Where to get the weights from.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// A Hugging Face repo id, downloaded on demand into the shared HF cache.
    Hub {
        repo_id: String,
        revision: Option<String>,
    },
    /// A directory that already holds an unpacked checkpoint.
    Local(PathBuf),
}

impl Default for ModelSource {
    fn default() -> Self {
        Self::Hub {
            repo_id: DEFAULT_MODEL_ID.to_string(),
            revision: None,
        }
    }
}

impl ModelSource {
    /// Treat the string as a local directory when one exists at that path, and
    /// as a hub repo id otherwise. This is what users expect from `--model`.
    pub fn parse(spec: &str) -> Self {
        let path = Path::new(spec);
        if path.is_dir() {
            Self::Local(path.to_path_buf())
        } else {
            Self::Hub {
                repo_id: spec.to_string(),
                revision: None,
            }
        }
    }

    pub fn with_revision(mut self, revision: Option<String>) -> Self {
        if let Self::Hub { revision: slot, .. } = &mut self {
            *slot = revision;
        }
        self
    }

    /// Resolve to local paths, downloading anything missing.
    ///
    /// Downloads land in the standard `HF_HOME` cache, so a checkpoint already
    /// pulled by `transformers` is reused rather than fetched twice.
    pub fn resolve(&self) -> Result<ModelFiles> {
        match self {
            Self::Local(dir) => resolve_local(dir),
            Self::Hub { repo_id, revision } => resolve_hub(repo_id, revision.as_deref()),
        }
    }
}

fn resolve_local(dir: &Path) -> Result<ModelFiles> {
    let need = |name: &str| -> Result<PathBuf> {
        let p = dir.join(name);
        if p.is_file() {
            Ok(p)
        } else {
            Err(Error::config(format!(
                "{} is missing from the checkpoint at {}",
                name,
                dir.display()
            )))
        }
    };
    let maybe = |name: &str| -> Option<PathBuf> {
        let p = dir.join(name);
        p.is_file().then_some(p)
    };

    let mut weights: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    weights.sort();

    if weights.is_empty() {
        return Err(Error::config(format!(
            "no .safetensors weights found in {}",
            dir.display()
        )));
    }

    Ok(ModelFiles {
        root: dir.to_path_buf(),
        config: need("config.json")?,
        tokenizer: need("tokenizer.json")?,
        preprocessor: need("preprocessor_config.json")?,
        generation: maybe("generation_config.json"),
        processor: maybe("processor_config.json"),
        weights,
    })
}

fn resolve_hub(repo_id: &str, revision: Option<&str>) -> Result<ModelFiles> {
    let (owner, name) = repo_id.split_once('/').ok_or_else(|| {
        Error::config(format!(
            "model id {repo_id:?} must be in `owner/name` form, or point at a local directory"
        ))
    })?;

    let client = HFClientSync::new().map_err(|source| Error::Hub {
        repo: repo_id.to_string(),
        file: "<client>".into(),
        source: Box::new(source),
    })?;
    let repo = client.model(owner, name);

    let fetch = |filename: &str| -> Result<PathBuf> {
        repo.download_file()
            .filename(filename)
            .maybe_revision(revision.map(str::to_string))
            .send()
            .map_err(|source| Error::Hub {
                repo: repo_id.to_string(),
                file: filename.to_string(),
                source: Box::new(source),
            })
    };
    let fetch_opt = |filename: &str| -> Option<PathBuf> { fetch(filename).ok() };

    let mut paths = Vec::new();
    for name in REQUIRED_FILES {
        paths.push(fetch(name)?);
    }
    let index = fetch_opt("model.safetensors.index.json");

    // Single-shard checkpoints publish one of a small set of conventional
    // names; sharded ones list their shards in the index.
    let weights = match &index {
        Some(index_path) => {
            let shards = shard_names_from_index(index_path)?;
            shards
                .iter()
                .map(|s| fetch(s))
                .collect::<Result<Vec<_>>>()?
        }
        None => {
            let candidates = ["model.safetensors", "model-00000-of-00001.safetensors"];
            let found: Vec<PathBuf> = candidates.iter().filter_map(|c| fetch_opt(c)).collect();
            if found.is_empty() {
                return Err(Error::config(format!(
                    "could not locate safetensors weights in {repo_id}; \
                     tried an index and {candidates:?}"
                )));
            }
            found
        }
    };

    let root = paths[0]
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    Ok(ModelFiles {
        root,
        config: paths[0].clone(),
        tokenizer: paths[1].clone(),
        preprocessor: paths[2].clone(),
        generation: fetch_opt(OPTIONAL_FILES[0]),
        processor: fetch_opt(OPTIONAL_FILES[1]),
        weights,
    })
}

/// Read the distinct shard filenames out of a `model.safetensors.index.json`.
fn shard_names_from_index(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let index: serde_json::Value = serde_json::from_str(&text).map_err(|source| Error::Json {
        what: "model.safetensors.index.json".into(),
        source,
    })?;
    let map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::config("safetensors index has no weight_map object"))?;

    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str())
        .map(str::to_string)
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_without_slash_is_not_a_hub_repo() {
        let src = ModelSource::parse("not-a-repo-id");
        // Parsed as a hub id (no such directory exists), but resolution refuses it.
        assert!(matches!(src, ModelSource::Hub { .. }));
        assert!(src.resolve().is_err());
    }

    #[test]
    fn existing_directory_parses_as_local() {
        let dir = std::env::temp_dir();
        assert!(matches!(
            ModelSource::parse(dir.to_str().unwrap()),
            ModelSource::Local(_)
        ));
    }

    #[test]
    fn default_targets_the_upstream_checkpoint() {
        match ModelSource::default() {
            ModelSource::Hub { repo_id, revision } => {
                assert_eq!(repo_id, DEFAULT_MODEL_ID);
                assert!(revision.is_none());
            }
            other => panic!("unexpected default: {other:?}"),
        }
    }

    #[test]
    fn index_shards_are_deduped_and_sorted() {
        let dir = std::env::temp_dir().join("fmd-index-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors.index.json");
        std::fs::write(
            &path,
            r#"{"weight_map":{"a":"model-00002.safetensors","b":"model-00001.safetensors","c":"model-00001.safetensors"}}"#,
        )
        .unwrap();
        let shards = shard_names_from_index(&path).unwrap();
        assert_eq!(
            shards,
            vec!["model-00001.safetensors", "model-00002.safetensors"]
        );
        std::fs::remove_file(&path).ok();
    }
}

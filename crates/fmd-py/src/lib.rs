//! Python bindings.
//!
//! The Rust engine is wrapped rather than reimplemented: this module's job is
//! argument marshalling, releasing the GIL around the compute, and presenting
//! results in a shape Python callers expect.

use std::path::PathBuf;
use std::sync::Mutex;

use fmd_core::{
    AudioSource, DeviceSpec, Engine as CoreEngine, EngineConfig, FinishReason, ModelSource,
    Precision, Request,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

/// One diarised utterance.
#[pyclass(module = "fast_moss_diarization", frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
}

#[pymethods]
impl Segment {
    fn __repr__(&self) -> String {
        format!(
            "Segment(start={:.2}, end={:.2}, speaker={:?}, text={:?})",
            self.start, self.end, self.speaker, self.text
        )
    }

    /// Length of the utterance in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("start", self.start)?;
        d.set_item("end", self.end)?;
        d.set_item("speaker", &self.speaker)?;
        d.set_item("text", &self.text)?;
        Ok(d)
    }
}

/// A completed transcription.
#[pyclass(module = "fast_moss_diarization", frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct Result {
    /// The identifier supplied with the request; the path, by default.
    pub id: String,
    /// The raw model output, before parsing.
    pub text: String,
    pub segments: Vec<Segment>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub audio_seconds: f64,
    /// `"stop"` if the model finished, `"length"` if it hit the token budget.
    pub finish_reason: String,
}

#[pymethods]
impl Result {
    fn __repr__(&self) -> String {
        format!(
            "Result(id={:?}, segments={}, generated_tokens={}, finish_reason={:?})",
            self.id,
            self.segments.len(),
            self.generated_tokens,
            self.finish_reason
        )
    }

    /// True when the token budget cut the transcript short.
    #[getter]
    fn truncated(&self) -> bool {
        self.finish_reason == "length"
    }

    /// The transcript as SubRip subtitles.
    #[pyo3(signature = (show_speaker = true))]
    fn to_srt(&self, show_speaker: bool) -> String {
        let segs: Vec<fmd_core::Segment> = self
            .segments
            .iter()
            .map(|s| fmd_core::Segment {
                start: s.start,
                end: s.end,
                speaker: s.speaker.clone(),
                text: s.text.clone(),
            })
            .collect();
        fmd_core::transcript::to_srt(&segs, show_speaker)
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("id", &self.id)?;
        d.set_item("text", &self.text)?;
        let segs = PyList::empty(py);
        for s in &self.segments {
            segs.append(s.as_dict(py)?)?;
        }
        d.set_item("segments", segs)?;
        d.set_item("prompt_tokens", self.prompt_tokens)?;
        d.set_item("generated_tokens", self.generated_tokens)?;
        d.set_item("audio_seconds", self.audio_seconds)?;
        d.set_item("finish_reason", &self.finish_reason)?;
        Ok(d)
    }
}

/// Timings for one `transcribe_batch` call.
#[pyclass(module = "fast_moss_diarization", frozen, get_all, from_py_object)]
#[derive(Clone)]
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

#[pymethods]
impl BatchStats {
    /// Seconds of audio handled per wall-clock second.
    #[getter]
    fn realtime_factor(&self) -> f64 {
        if self.total_seconds > 0.0 {
            self.audio_seconds / self.total_seconds
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchStats(requests={}, total_seconds={:.2}, realtime_factor={:.1})",
            self.requests,
            self.total_seconds,
            self.realtime_factor()
        )
    }
}

/// A request that failed, returned in place of a [`Result`] when
/// `raise_on_error=False`.
#[pyclass(module = "fast_moss_diarization", frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct Failure {
    pub id: String,
    pub error: String,
}

#[pymethods]
impl Failure {
    fn __repr__(&self) -> String {
        format!("Failure(id={:?}, error={:?})", self.id, self.error)
    }
}

/// The inference engine.
///
/// Loading is expensive, so build one engine and reuse it. Instances are safe to
/// share between threads; calls are serialised internally.
#[pyclass(module = "fast_moss_diarization")]
pub struct Engine {
    inner: Mutex<CoreEngine>,
    precision: String,
    device: String,
}

#[pymethods]
impl Engine {
    /// Load the model, downloading the checkpoint on first use.
    ///
    /// `model` is a Hugging Face repo id or a local directory. `dtype` is one of
    /// `"fp32"`, `"fp16"`, `"bf16"` or `"int8"`; the default picks bf16 on GPU
    /// and fp32 on CPU. `device` is `"auto"`, `"cpu"`, `"cuda"` or `"cuda:N"`.
    #[new]
    #[pyo3(signature = (
        model = fmd_core::DEFAULT_MODEL_ID,
        *,
        dtype = None,
        device = "auto",
        revision = None,
        max_new_tokens = 4096,
        batch_size = 8,
        encoder_batch = 16,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &str,
        dtype: Option<&str>,
        device: &str,
        revision: Option<String>,
        max_new_tokens: usize,
        batch_size: usize,
        encoder_batch: usize,
    ) -> PyResult<Self> {
        let precision = dtype
            .map(Precision::parse)
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let device_spec =
            DeviceSpec::parse(device).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let cfg = EngineConfig {
            source: ModelSource::parse(model).with_revision(revision),
            device: device_spec,
            precision,
            max_new_tokens,
            max_batch_size: batch_size,
            encoder_batch,
        };

        // Downloading and loading weights takes seconds; do not hold the GIL.
        let engine = py
            .detach(|| CoreEngine::new(cfg))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            precision: engine.precision().to_string(),
            device: format!("{:?}", engine.device()),
            inner: Mutex::new(engine),
        })
    }

    /// The precision actually in use.
    #[getter]
    fn dtype(&self) -> &str {
        &self.precision
    }

    /// The device actually in use.
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    /// Transcribe a single input and return its [`Result`].
    #[pyo3(signature = (audio, *, prompt = None))]
    fn transcribe(
        &self,
        py: Python<'_>,
        audio: &Bound<'_, PyAny>,
        prompt: Option<String>,
    ) -> PyResult<Result> {
        let list = PyList::new(py, [audio])?;
        let (mut results, _) = self.run(py, list.as_any(), prompt, true)?;
        match results.remove(0) {
            Outcome::Ok(r) => Ok(r),
            Outcome::Err(f) => Err(PyRuntimeError::new_err(f.error)),
        }
    }

    /// Transcribe many inputs in one batched pass.
    ///
    /// `inputs` may be a sequence of paths or waveforms, a sequence of
    /// `(id, audio)` pairs, or a mapping of `id -> audio`. Results come back in
    /// the same order (for a mapping, in iteration order) and each carries its
    /// `id`, so callers can match results to inputs either way.
    ///
    /// With `raise_on_error=True` (the default) the first failure raises. Set it
    /// to False to receive a `Failure` in that slot instead and keep the rest.
    ///
    /// Returns `(results, stats)`.
    #[pyo3(signature = (inputs, *, prompt = None, raise_on_error = true))]
    fn transcribe_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        prompt: Option<String>,
        raise_on_error: bool,
    ) -> PyResult<(Bound<'py, PyList>, BatchStats)> {
        let (outcomes, stats) = self.run(py, inputs, prompt, raise_on_error)?;
        let list = PyList::empty(py);
        for outcome in outcomes {
            match outcome {
                Outcome::Ok(r) => list.append(r.into_pyobject(py)?)?,
                Outcome::Err(f) => list.append(f.into_pyobject(py)?)?,
            }
        }
        Ok((list, stats))
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(dtype={:?}, device={:?})",
            self.precision, self.device
        )
    }
}

enum Outcome {
    Ok(Result),
    Err(Failure),
}

impl Engine {
    fn run(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        prompt: Option<String>,
        raise_on_error: bool,
    ) -> PyResult<(Vec<Outcome>, BatchStats)> {
        let requests = build_requests(inputs, prompt)?;
        if requests.is_empty() {
            return Ok((
                Vec::new(),
                BatchStats {
                    requests: 0,
                    decode_seconds: 0.0,
                    feature_seconds: 0.0,
                    encode_seconds: 0.0,
                    generate_seconds: 0.0,
                    total_seconds: 0.0,
                    audio_seconds: 0.0,
                    generated_tokens: 0,
                    micro_batches: 0,
                },
            ));
        }
        let ids: Vec<String> = requests.iter().map(|r| r.id.clone()).collect();

        // The whole batch runs without the GIL, so other Python threads keep
        // working while the GPU does.
        let (results, stats) = py
            .detach(|| {
                let mut engine = self
                    .inner
                    .lock()
                    .map_err(|_| fmd_core::Error::Config("engine mutex was poisoned".into()))?;
                engine.transcribe(requests)
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut outcomes = Vec::with_capacity(results.len());
        for (id, result) in ids.into_iter().zip(results) {
            match result {
                Ok(r) => outcomes.push(Outcome::Ok(Result {
                    id: r.id,
                    text: r.text,
                    segments: r
                        .segments
                        .into_iter()
                        .map(|s| Segment {
                            start: s.start,
                            end: s.end,
                            speaker: s.speaker,
                            text: s.text,
                        })
                        .collect(),
                    prompt_tokens: r.prompt_tokens,
                    generated_tokens: r.generated_tokens,
                    audio_seconds: r.audio_seconds,
                    finish_reason: match r.finish_reason {
                        FinishReason::Stop => "stop".into(),
                        FinishReason::Length => "length".into(),
                    },
                })),
                Err(e) => {
                    if raise_on_error {
                        return Err(PyRuntimeError::new_err(format!("{id}: {e}")));
                    }
                    outcomes.push(Outcome::Err(Failure {
                        id,
                        error: e.to_string(),
                    }));
                }
            }
        }

        Ok((
            outcomes,
            BatchStats {
                requests: stats.requests,
                decode_seconds: stats.decode_seconds,
                feature_seconds: stats.feature_seconds,
                encode_seconds: stats.encode_seconds,
                generate_seconds: stats.generate_seconds,
                total_seconds: stats.total_seconds,
                audio_seconds: stats.audio_seconds,
                generated_tokens: stats.generated_tokens,
                micro_batches: stats.micro_batches,
            },
        ))
    }
}

/// Turn the many accepted input shapes into engine requests.
fn build_requests(inputs: &Bound<'_, PyAny>, prompt: Option<String>) -> PyResult<Vec<Request>> {
    // A mapping of id -> audio.
    if let Ok(dict) = inputs.cast::<PyDict>() {
        return dict
            .iter()
            .map(|(k, v)| {
                let id = k
                    .extract::<String>()
                    .or_else(|_| k.str().map(|s| s.to_string_lossy().into_owned()))?;
                Ok(Request {
                    id,
                    source: audio_source(&v)?,
                    prompt: prompt.clone(),
                })
            })
            .collect();
    }

    // A bare path or waveform, not a sequence of them.
    if inputs.is_instance_of::<PyString>() || is_buffer(inputs) {
        return Ok(vec![Request {
            id: default_id(inputs, 0),
            source: audio_source(inputs)?,
            prompt,
        }]);
    }

    let mut out = Vec::new();
    let iter = inputs.try_iter().map_err(|_| {
        PyTypeError::new_err("inputs must be a path, a waveform, a sequence, or a mapping")
    })?;
    for (index, item) in iter.enumerate() {
        let item = item?;
        // A 2-tuple is ambiguous: it can be `(id, audio)` or
        // `(waveform, sample_rate)`. Only the former names its first element
        // with a string, and only the latter has an integer second element, so
        // requiring both settles it.
        if let Ok(pair) = item.cast::<PyTuple>() {
            if pair.len() == 2 && !is_waveform_pair(pair)? {
                let id = pair.get_item(0)?;
                if id.is_instance_of::<PyString>() {
                    let audio = pair.get_item(1)?;
                    out.push(Request {
                        id: id.extract::<String>()?,
                        source: audio_source(&audio)?,
                        prompt: prompt.clone(),
                    });
                    continue;
                }
            }
        }
        out.push(Request {
            id: default_id(&item, index),
            source: audio_source(&item)?,
            prompt: prompt.clone(),
        });
    }
    Ok(out)
}

/// Paths identify themselves; in-memory waveforms fall back to their position.
fn default_id(obj: &Bound<'_, PyAny>, index: usize) -> String {
    obj.extract::<PathBuf>()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| index.to_string())
}

fn is_buffer(obj: &Bound<'_, PyAny>) -> bool {
    !obj.is_instance_of::<PyString>() && PyBuffer::<f32>::get(obj).is_ok()
}

/// Whether a 2-tuple is `(waveform, sample_rate)` rather than `(id, audio)`.
fn is_waveform_pair(pair: &Bound<'_, PyTuple>) -> PyResult<bool> {
    if pair.len() != 2 {
        return Ok(false);
    }
    let rate_is_int = pair.get_item(1)?.extract::<u32>().is_ok();
    Ok(rate_is_int && is_buffer(&pair.get_item(0)?))
}

/// Accept a path, or any float32 buffer (a numpy array, say) as
/// `(samples, sample_rate)`.
fn audio_source(obj: &Bound<'_, PyAny>) -> PyResult<AudioSource> {
    // (waveform, sample_rate)
    if let Ok(pair) = obj.cast::<PyTuple>() {
        if pair.len() == 2 {
            if let Ok(rate) = pair.get_item(1)?.extract::<u32>() {
                let buf = PyBuffer::<f32>::get(&pair.get_item(0)?).map_err(|_| {
                    PyTypeError::new_err(
                        "waveform must be a contiguous float32 buffer, such as a numpy array",
                    )
                })?;
                return Ok(AudioSource::Samples {
                    samples: buffer_to_vec(&buf, pair.py())?,
                    sample_rate: rate,
                });
            }
        }
    }

    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if !obj.is_instance_of::<PyString>() {
            // A bare waveform is assumed to already be at the model's rate.
            return Ok(AudioSource::Samples {
                samples: buffer_to_vec(&buf, obj.py())?,
                sample_rate: 16_000,
            });
        }
    }

    let path: PathBuf = obj.extract().map_err(|_| {
        PyTypeError::new_err(
            "expected a path, a float32 waveform, or a (waveform, sample_rate) pair",
        )
    })?;
    Ok(AudioSource::Path(path))
}

fn buffer_to_vec(buf: &PyBuffer<f32>, py: Python<'_>) -> PyResult<Vec<f32>> {
    buf.to_vec(py)
        .map_err(|e| PyValueError::new_err(format!("could not read the waveform buffer: {e}")))
}

#[pymodule]
fn _fast_moss_diarization(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<Result>()?;
    m.add_class::<Segment>()?;
    m.add_class::<BatchStats>()?;
    m.add_class::<Failure>()?;
    m.add("DEFAULT_MODEL_ID", fmd_core::DEFAULT_MODEL_ID)?;
    m.add("DEFAULT_PROMPT", fmd_core::DEFAULT_PROMPT)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

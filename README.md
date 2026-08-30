# fast-moss-diarization

A GPU inference engine for [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize),
written in Rust with Python bindings. It does joint transcription and speaker
diarization of long-form audio, and it is built for throughput: many files go
through one batched pass rather than one at a time.

The whole pipeline is Rust — audio decoding, the log-mel frontend, the Whisper
encoder, and the Qwen3 decoder — on top of [candle](https://github.com/huggingface/candle).
There is no PyTorch dependency.

## What it does

* **fp32, fp16, bf16 and int8** precision.
* **Weights download on demand** into the standard Hugging Face cache, so a
  checkpoint already pulled by `transformers` is reused rather than fetched again.
* **Batched inference over many files at once**, with results returned in input
  order and tagged with the caller's own ids.
* **Any container ffmpeg-ish** — wav, mp3, flac, ogg, opus, m4a, mp4, mkv,
  webm — decoded and resampled in Rust.

## Install

Needs a CUDA toolkit (`nvcc` on `PATH`) and a recent Rust toolchain.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

pip install maturin
maturin develop --release
```

For a CPU-only build (much slower; fp32 only):

```bash
maturin develop --release --no-default-features
```

## Python

```python
from fast_moss_diarization import Engine

engine = Engine(dtype="fp16")            # bf16 on GPU by default
result = engine.transcribe("meeting.wav")

for seg in result.segments:
    print(f"[{seg.start:6.2f} - {seg.end:6.2f}] {seg.speaker}: {seg.text}")
```

### Batching

Batching is the reason this engine exists. Passing several files to one call is
far faster than looping, because both halves of the model batch:

```python
results, stats = engine.transcribe_batch([
    "interview.wav",
    "standup.mp3",
    "lecture.m4a",
])

for r in results:                        # same order as the inputs
    print(r.id, len(r.segments), r.finish_reason)

print(f"{stats.realtime_factor:.0f}x realtime")
```

Results carry an `id` so they can be matched to inputs by name instead of by
position. Any of these input shapes work:

```python
engine.transcribe_batch(["a.wav", "b.wav"])              # id = the path
engine.transcribe_batch({"call-1": "a.wav", "call-2": "b.wav"})
engine.transcribe_batch([("call-1", "a.wav"), ("call-2", "b.wav")])
engine.transcribe_batch([(samples, 44100)])              # numpy float32 array
```

By default a failed input raises. To let the rest of the batch through, pass
`raise_on_error=False` and check for `Failure` objects:

```python
from fast_moss_diarization import Failure

results, _ = engine.transcribe_batch(paths, raise_on_error=False)
for r in results:
    if isinstance(r, Failure):
        print(f"{r.id} failed: {r.error}")
```

### Options

```python
Engine(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",  # repo id or local directory
    dtype="fp16",          # fp32 | fp16 | bf16 | int8
    device="auto",         # auto | cpu | cuda | cuda:N
    max_new_tokens=4096,   # generation budget per input
    batch_size=8,          # sequences decoded together
    encoder_batch=16,      # 30 s windows encoded together
)
```

## Command line

```bash
fmd meeting.wav --dtype fp16 --stats
fmd clips/*.wav --batch-size 8 --format json
fmd talk.mp4 --format srt > talk.srt
```

## Choosing a precision

| dtype  | Weights | Notes |
|--------|--------:|-------|
| `bf16` |  ~1.8 GB | Matches the checkpoint exactly. The safe default. |
| `fp16` |  ~1.8 GB | Same memory, usually a little faster than bf16. |
| `int8` |  ~0.6 GB | Q8_0 weights, 16-bit activations. Smallest and fastest. |
| `fp32` |  ~3.6 GB | Reference precision; the only option on CPU. |

`int8` quantizes the Qwen3 decoder and the output projection, which is where
both the parameters and the per-token memory traffic are. The audio encoder and
the adaptor stay in floating point on purpose: the encoder runs once per 30 s of
audio and is compute-bound rather than bandwidth-bound, so quantizing it would
cost accuracy and buy almost no time.

## Memory

The KV cache, not the weights, is what usually runs a card out of memory. It
costs roughly

```
2 x layers x kv_heads x head_dim x (prompt + max_new_tokens) x batch x bytes
= 2 x 28 x 8 x 128 x tokens x batch x 2   bytes
```

which is about **1.3 GB per sequence** for a 10-minute recording with a
4096-token budget. Lower `batch_size` if allocation fails. Requests are sorted by
length and grouped, so a batch is sized by its own longest member rather than by
the longest in the whole call.

## How the batching works

The two halves of the model have opposite shapes, so they are batched
differently:

* The **audio encoder** always sees 30-second windows. Windows from *different*
  files stack into one uniform tensor, so a hundred short clips become a single
  large encoder pass.
* The **decoder** sees variable-length prompts. Requests are sorted by prompt
  length and cut into micro-batches, which keeps both the padding waste and the
  KV cache small, then decoded together with a shared cache and per-sequence
  end-of-turn tracking.

## Verifying against the reference

Two fixtures pin this implementation to the Python one. Generate them with the
reference `transformers` environment, then run the ignored tests:

```bash
python scripts/dump_reference_mel.py testdata/
python scripts/dump_reference_prompt.py testdata/
cargo test -p fmd-core --release -- --ignored
```

The log-mel frontend matches `WhisperFeatureExtractor` to about `1e-5`, and the
prompt builder reproduces the reference token ids exactly — including the time
markers interleaved into the audio span, where a single token of drift would
shift every timestamp the model emits.

## Licence

Apache-2.0, matching the upstream model repository.

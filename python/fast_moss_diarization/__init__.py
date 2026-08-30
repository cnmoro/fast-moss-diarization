"""Fast GPU transcription and diarization for MOSS-Transcribe-Diarize.

The engine loads the model once and reuses it, so build one and keep it:

    from fast_moss_diarization import Engine

    engine = Engine(dtype="fp16")
    result = engine.transcribe("meeting.wav")
    for seg in result.segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.speaker}: {seg.text}")

Batching several files into one call is much faster than looping, because the
audio encoder and the decoder both run over the whole batch at once:

    results, stats = engine.transcribe_batch(["a.wav", "b.wav", "c.wav"])
    print(f"{stats.realtime_factor:.0f}x realtime")

Results come back in input order and each carries an ``id``, so they can be
matched to inputs either positionally or by name.
"""

from ._fast_moss_diarization import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    BatchStats,
    Engine,
    Failure,
    Result,
    Segment,
    __version__,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROMPT",
    "BatchStats",
    "Engine",
    "Failure",
    "Result",
    "Segment",
    "__version__",
]

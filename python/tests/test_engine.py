"""Tests for the Python bindings.

These load the real model, so they are slow and need the checkpoint. Run with:

    pytest python/tests -v

Set FMD_TEST_AUDIO to a directory of wav files to use your own fixtures;
otherwise short tones are synthesised, which exercises the plumbing without
asserting anything about transcription quality.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from fast_moss_diarization import BatchStats, Engine, Failure, Result, Segment

DTYPE = os.environ.get("FMD_TEST_DTYPE", "fp16")


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine(dtype=DTYPE, batch_size=4, max_new_tokens=128)


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> list[Path]:
    """Three short wav files."""
    supplied = os.environ.get("FMD_TEST_AUDIO")
    if supplied:
        found = sorted(Path(supplied).glob("*.wav"))[:3]
        if len(found) >= 3:
            return found

    out_dir = tmp_path_factory.mktemp("clips")
    paths = []
    for i, freq in enumerate((220.0, 440.0, 880.0)):
        t = np.arange(16_000 * 2) / 16_000.0
        wave_data = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
        path = out_dir / f"tone_{i}.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16_000)
            fh.writeframes(wave_data.tobytes())
        paths.append(path)
    return paths


def test_engine_reports_its_configuration(engine: Engine) -> None:
    assert engine.dtype == DTYPE
    assert "Cuda" in engine.device or "Cpu" in engine.device


def test_transcribe_returns_a_result(engine: Engine, clips) -> None:
    result = engine.transcribe(str(clips[0]))
    assert isinstance(result, Result)
    assert result.id == str(clips[0])
    assert result.finish_reason in {"stop", "length"}
    assert result.prompt_tokens > 0
    assert result.audio_seconds > 0
    assert all(isinstance(s, Segment) for s in result.segments)


def test_batch_preserves_input_order_and_ids(engine: Engine, clips) -> None:
    paths = [str(p) for p in clips]
    results, stats = engine.transcribe_batch(paths)

    assert isinstance(stats, BatchStats)
    assert stats.requests == len(paths)
    # The engine reorders internally to bucket by length; callers must not see it.
    assert [r.id for r in results] == paths


def test_batch_accepts_a_mapping(engine: Engine, clips) -> None:
    mapping = {f"call-{i}": str(p) for i, p in enumerate(clips)}
    results, _ = engine.transcribe_batch(mapping)
    assert [r.id for r in results] == list(mapping)


def test_batch_accepts_id_audio_pairs(engine: Engine, clips) -> None:
    pairs = [("first", str(clips[0])), ("second", str(clips[1]))]
    results, _ = engine.transcribe_batch(pairs)
    assert [r.id for r in results] == ["first", "second"]


def test_accepts_a_numpy_waveform(engine: Engine, clips) -> None:
    import soundfile as sf

    samples, rate = sf.read(str(clips[0]), dtype="float32")
    result = engine.transcribe((samples, rate))
    assert isinstance(result, Result)
    assert result.audio_seconds == pytest.approx(len(samples) / rate, abs=0.1)


def test_a_waveform_at_another_rate_is_resampled(engine: Engine) -> None:
    # 44.1 kHz in, 16 kHz internally: the duration must survive the conversion.
    seconds = 2.0
    t = np.arange(int(44_100 * seconds)) / 44_100.0
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    result = engine.transcribe((samples, 44_100))
    assert result.audio_seconds == pytest.approx(seconds, abs=0.1)


def test_a_bad_input_raises_by_default(engine: Engine) -> None:
    with pytest.raises(RuntimeError, match="no_such_file"):
        engine.transcribe_batch(["no_such_file.wav"])


def test_a_bad_input_can_be_tolerated(engine: Engine, clips) -> None:
    inputs = [str(clips[0]), "no_such_file.wav", str(clips[1])]
    results, _ = engine.transcribe_batch(inputs, raise_on_error=False)

    assert [r.id for r in results] == inputs
    assert isinstance(results[0], Result)
    assert isinstance(results[1], Failure)
    assert isinstance(results[2], Result)
    assert "no_such_file" in results[1].error


def test_an_empty_batch_is_not_an_error(engine: Engine) -> None:
    results, stats = engine.transcribe_batch([])
    assert results == []
    assert stats.requests == 0
    assert stats.realtime_factor == 0.0


def test_bad_options_are_rejected_early() -> None:
    with pytest.raises(ValueError):
        Engine(dtype="fp8")
    with pytest.raises(ValueError):
        Engine(device="tpu")


def test_result_converts_to_plain_python(engine: Engine, clips) -> None:
    result = engine.transcribe(str(clips[0]))
    payload = result.as_dict()
    assert set(payload) >= {"id", "text", "segments", "finish_reason"}
    assert isinstance(payload["segments"], list)
    assert isinstance(result.to_srt(), str)


def test_batching_matches_running_one_at_a_time(engine: Engine, clips) -> None:
    """Batched and unbatched decoding must agree.

    Left-padded sequences share one position range and are separated only by the
    attention mask, so a mask bug shows up here as a padded sequence decoding to
    something unrelated to its solo run.

    The comparison is a similarity threshold rather than equality on purpose:
    batched and unbatched matmuls reduce in different orders, and that is enough
    to flip an occasional near-tie in the greedy argmax -- typically one capital
    letter or one timestamp in the last decimal. The failures this guards
    against are not subtle; the original masking bug produced a wall of "!".
    """
    from difflib import SequenceMatcher

    paths = [str(p) for p in clips]
    solo = [engine.transcribe(p).text for p in paths]
    batched, _ = engine.transcribe_batch(paths)

    for path, alone, together in zip(paths, solo, batched):
        ratio = SequenceMatcher(None, alone, together.text).ratio()
        assert ratio > 0.95, (
            f"{path} decoded differently in a batch (similarity {ratio:.3f})\n"
            f"  alone:   {alone[:200]}\n"
            f"  batched: {together.text[:200]}"
        )


def test_a_short_clip_batched_with_a_long_one(engine: Engine, clips) -> None:
    """Heavy left padding must not corrupt the short sequence.

    Regression test. When a batch mixes very different lengths, the short
    sequence is padded far to the left, and every query row inside that padding
    used to be fully masked. Softmax over an all -inf row is NaN, and the NaN
    reached the real tokens through the `0 * NaN` terms of the value matmul, so
    the short clip decoded to garbage while decoding fine on its own.
    """
    from difflib import SequenceMatcher

    short = np.tile(
        (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(16_000 * 2) / 16_000.0)),
        1,
    ).astype(np.float32)
    long = np.tile(short, 30)  # 60 s against 2 s: ~28x the prompt length

    alone = engine.transcribe((short, 16_000)).text
    batched, _ = engine.transcribe_batch([(short, 16_000), (long, 16_000)])

    ratio = SequenceMatcher(None, alone, batched[0].text).ratio()
    assert ratio > 0.95, (
        f"padded short clip diverged (similarity {ratio:.3f})\n"
        f"  alone:   {alone[:200]}\n"
        f"  batched: {batched[0].text[:200]}"
    )

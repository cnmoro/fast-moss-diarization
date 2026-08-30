"""Dump reference Whisper log-mel features for the Rust parity test.

Run this with the reference `transformers` environment, then run
`cargo test -p fmd-core --release -- --ignored` to compare.

    python scripts/dump_reference_mel.py testdata/

The waveform is deterministic (seeded), so the test needs no audio fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor


def reference_waveform(num_samples: int = 480_000) -> np.ndarray:
    """A deterministic signal with tones, noise and silence.

    Mixing the three matters: pure tones alone would not exercise the 80 dB
    dynamic-range clamp, and pure noise would not expose filterbank misalignment.
    """
    rng = np.random.default_rng(0)
    t = np.arange(num_samples, dtype=np.float64) / 16_000.0
    wave = (
        0.4 * np.sin(2 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2 * np.pi * 1337.0 * t)
        + 0.1 * np.sin(2 * np.pi * 6000.0 * t)
        + 0.02 * rng.standard_normal(num_samples)
    )
    # A silent stretch, so the clamp floor is actually reached somewhere.
    wave[100_000:140_000] = 0.0
    # A short loud transient, to move the per-chunk peak.
    wave[200_000:200_400] *= 8.0
    return wave.astype(np.float32)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")
    out_dir.mkdir(parents=True, exist_ok=True)

    wave = reference_waveform()
    extractor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16_000,
        hop_length=160,
        n_fft=400,
        chunk_length=30,
        dither=0.0,
    )
    feats = extractor(
        [wave],
        sampling_rate=16_000,
        padding="max_length",
        return_tensors="np",
    )["input_features"][0]

    assert feats.shape == (80, 3000), feats.shape

    (out_dir / "mel_input.f32").write_bytes(wave.astype("<f4").tobytes())
    (out_dir / "mel_reference.f32").write_bytes(
        feats.astype("<f4").ravel(order="C").tobytes()
    )
    print(f"wrote {out_dir}/mel_input.f32 ({wave.size} samples)")
    print(f"wrote {out_dir}/mel_reference.f32 ({feats.shape[0]}x{feats.shape[1]})")
    print(f"reference range: [{feats.min():.4f}, {feats.max():.4f}]")


if __name__ == "__main__":
    main()

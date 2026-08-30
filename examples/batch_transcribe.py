"""Transcribe a directory of audio files in one batched pass.

    python examples/batch_transcribe.py /path/to/audio --dtype fp16 --batch-size 8

Prints a diarized transcript per file and a throughput summary. Compare the
reported realtime factor at `--batch-size 1` against `--batch-size 8` to see what
batching buys on your hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fast_moss_diarization import Engine, Failure

AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a",
    ".mp4", ".mkv", ".webm", ".aac", ".aiff",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="An audio file or a directory of them.")
    parser.add_argument("--dtype", default="fp16", help="fp32, fp16, bf16 or int8.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    if args.path.is_dir():
        files = sorted(
            p for p in args.path.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES
        )
    else:
        files = [args.path]

    if not files:
        print(f"no audio files found under {args.path}", file=sys.stderr)
        return 1

    print(f"loading the model ...", file=sys.stderr)
    engine = Engine(
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"ready: {engine!r}", file=sys.stderr)

    # One call, not a loop: the engine batches the encoder across every file and
    # decodes them together.
    results, stats = engine.transcribe_batch(
        [str(p) for p in files], raise_on_error=False
    )

    if args.json:
        print(json.dumps(
            {
                "results": [
                    {"id": r.id, "error": r.error} if isinstance(r, Failure)
                    else r.as_dict()
                    for r in results
                ],
                "realtime_factor": stats.realtime_factor,
                "total_seconds": stats.total_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ))
    else:
        for result in results:
            print(f"\n=== {result.id} ===")
            if isinstance(result, Failure):
                print(f"  failed: {result.error}")
                continue
            for seg in result.segments:
                print(f"[{seg.start:8.2f} - {seg.end:8.2f}] {seg.speaker}: {seg.text}")
            if result.truncated:
                print("  (truncated: raise --max-new-tokens)")

    failures = sum(isinstance(r, Failure) for r in results)
    print(
        f"\n{stats.requests} file(s), {stats.audio_seconds:.0f}s of audio in "
        f"{stats.total_seconds:.1f}s -> {stats.realtime_factor:.0f}x realtime "
        f"({stats.micro_batches} micro-batch(es), {failures} failed)",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

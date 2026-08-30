"""Dump reference prompt token ids for the Rust parity test.

    python scripts/dump_reference_prompt.py testdata/

Writes one JSON file holding, for several audio lengths, the exact `input_ids`
the reference processor produces. The Rust builder must match them element for
element: a single token of drift shifts every timestamp the model emits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoProcessor

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

# Covers: no markers at all, exactly one 30 s chunk, a partial chunk, and a
# 10-minute recording where marker digits grow to three characters.
AUDIO_TOKEN_COUNTS = [0, 12, 50, 62, 63, 375, 438, 875, 7500]

CASES = [
    ("default", DEFAULT_PROMPT),
    ("custom", "Transcribe the audio with speaker labels."),
]


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # The method is public in the repo copy and private in the published remote
    # code; accept either so this script works against both.
    expand = getattr(processor, "expand_audio_token", None) or processor._expand_audio_token

    messages_text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "dummy.wav"},
                    {"type": "text", "text": "PROMPT_PLACEHOLDER"},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    out = {
        "audio_token_id": processor.audio_token_id,
        "rendered_template": messages_text,
        "cases": [],
    }
    for name, instruction in CASES:
        text = messages_text.replace("PROMPT_PLACEHOLDER", instruction)
        for n in AUDIO_TOKEN_COUNTS:
            ids = expand(text, n, max_length=131072)
            out["cases"].append(
                {
                    "name": name,
                    "instruction": instruction,
                    "audio_tokens": n,
                    "input_ids": ids,
                }
            )

    path = out_dir / "prompt_reference.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {path}")
    print(f"audio_token_id = {out['audio_token_id']}")
    for case in out["cases"]:
        print(f"  {case['name']:8} audio={case['audio_tokens']:5} -> {len(case['input_ids'])} ids")


if __name__ == "__main__":
    main()

"""Format neutral/styled pairs as a fine-tuning-ready JSONL (build plan step 4).

Emits the common ``{"messages": [...]}`` chat format most fine-tuning APIs
accept (OpenAI-compatible, and easily adaptable to others): neutral text in
as the user turn, styled text out as the assistant turn to learn.
"""
import argparse
import json
from pathlib import Path

TASK_PROMPT = "Rewrite the following neutral text in the target author's voice."


def to_example(pair: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": TASK_PROMPT},
            {"role": "user", "content": pair["neutral"]},
            {"role": "assistant", "content": pair["styled"]},
        ]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.infile.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            fout.write(json.dumps(to_example(pair), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} fine-tuning examples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

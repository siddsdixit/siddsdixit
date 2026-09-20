"""Strip styled text down to flat, neutral phrasing (build plan step 2).

Two modes:
  --mock       pure-Python heuristic neutralization; no API calls. Exercises
               the rest of the pipeline (pairing, dataset building,
               fine-tune formatting) without cost or a key.
  (default)    calls the Anthropic Messages API to do the actual
               style-stripping. Requires ANTHROPIC_API_KEY.

The mock mode is not a substitute for the real model-based step when
actually building a training set — it's there so the pipeline plumbing is
testable offline.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

NEUTRALIZE_SYSTEM_PROMPT = (
    "You flatten text to plain, neutral phrasing for a style-transfer "
    "training pipeline. Preserve every fact, claim, and piece of "
    "information exactly. Remove voice: no rhetorical questions, no "
    "idioms, no emphasis, no personality, no rhythm, no varied sentence "
    "length. Use short, plain, declarative sentences. Do not add "
    "commentary, headers, or quotation marks around the output. Output "
    "only the neutralized text."
)

_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "i'm": "I am", "it's": "it is", "that's": "that is",
    "there's": "there is", "you're": "you are", "we're": "we are",
    "they're": "they are", "i've": "I have", "we've": "we have",
    "i'll": "I will", "you'll": "you will",
}


def neutralize_mock(text: str) -> str:
    """Cheap heuristic stand-in for the LLM neutralization call."""
    out = text
    for contraction, expanded in _CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(contraction)}\b", expanded, out, flags=re.IGNORECASE)
    out = re.sub(r"[!?]+", ".", out)           # drop emphasis/rhetorical punctuation
    out = re.sub(r"[—–]|--", ",", out)  # em/en dash -> comma
    out = re.sub(r"\.{2,}", ".", out)
    out = re.sub(r"[ \t]+", " ", out).strip()
    return out


def neutralize_llm(text: str, model: str) -> str:
    import anthropic  # imported lazily so --mock never needs the dependency

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=NEUTRALIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", type=Path, required=True,
                         help="JSONL with 'id' and 'text' fields (styled source)")
    parser.add_argument("--out", type=Path, required=True,
                         help="Output JSONL of {id, styled, neutral, source} records")
    parser.add_argument("--mock", action="store_true", help="Use the offline heuristic instead of calling an LLM")
    parser.add_argument("--model", default="claude-sonnet-5", help="Model to use for real neutralization")
    parser.add_argument("--limit", type=int, help="Only process the first N records")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between live API calls")
    args = parser.parse_args(argv)

    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is not set. Pass --mock to test without an API key.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.infile.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit and processed >= args.limit:
                break
            record = json.loads(line)
            styled = record["text"]
            neutral = neutralize_mock(styled) if args.mock else neutralize_llm(styled, args.model)
            fout.write(json.dumps({
                "id": record["id"],
                "styled": styled,
                "neutral": neutral,
                "source": {k: v for k, v in record.items() if k not in ("id", "text")},
            }, ensure_ascii=False) + "\n")
            processed += 1
            if not args.mock:
                time.sleep(args.sleep)
    print(f"Neutralized {processed} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

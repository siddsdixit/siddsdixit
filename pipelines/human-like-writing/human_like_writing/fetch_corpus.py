"""Pull a small pre-2022 text sample via the Hugging Face datasets-server API.

No heavy ML dependencies (no ``datasets``/``pyarrow``): just HTTP GET
against the public, paginated datasets-server "rows" endpoint. This is the
"pull a small test sample first" step of the build plan, before committing
to a full download.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

# Each source is pre-2022 (or date-filterable to be), per the write-up's
# corpus table. Add more sources here as they're vetted.
SOURCES = {
    "c4": {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "train",
        "text_field": "text",
        "extra_fields": ["timestamp", "url"],
    },
}


def fetch_rows(source: str, limit: int, offset: int = 0, sleep: float = 0.2) -> Iterator[dict]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Known: {sorted(SOURCES)}")
    spec = SOURCES[source]
    fetched = 0
    while fetched < limit:
        length = min(PAGE_SIZE, limit - fetched)
        params = {
            "dataset": spec["dataset"],
            "config": spec["config"],
            "split": spec["split"],
            "offset": offset + fetched,
            "length": length,
        }
        resp = requests.get(DATASETS_SERVER, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
        if not rows:
            break
        for row in rows:
            record = row["row"]
            yield {
                "id": f"{source}-{row['row_idx']}",
                "text": record[spec["text_field"]],
                **{f: record.get(f) for f in spec["extra_fields"]},
            }
        fetched += len(rows)
        if len(rows) < length:
            break
        time.sleep(sleep)  # be polite to the shared public endpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="c4", choices=sorted(SOURCES))
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to pull")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=50, help="Skip samples shorter than this many words")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for record in fetch_rows(args.source, args.limit, args.offset):
            if len(record["text"].split()) < args.min_words:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} samples to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

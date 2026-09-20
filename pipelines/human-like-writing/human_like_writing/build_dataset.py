"""Pair neutral/styled records into a train/val split (build plan step 3)."""
import argparse
import json
import random
from pathlib import Path


def load_pairs(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    all_pairs = list(load_pairs(args.infile))
    pairs = [p for p in all_pairs if p["neutral"].strip() and p["neutral"] != p["styled"]]
    random.Random(args.seed).shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_fraction)) if pairs else 0
    val, train = pairs[:n_val], pairs[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for pair in split:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    dropped = len(all_pairs) - len(pairs)
    print(f"train={len(train)} val={len(val)} (dropped {dropped} no-op/empty pairs) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

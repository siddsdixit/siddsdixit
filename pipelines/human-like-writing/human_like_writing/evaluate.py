"""Check generated text against real human samples (build plan step 5).

Compares each generated sample's stylometric feature vector to the
distribution of that feature across real human reference text, using
z-scores. This is the harder bar the write-up asks for: not "does this
fool a detector" but "does its feature distribution actually overlap with
real pre-2022 human writing."
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from .stylometry import FEATURE_NAMES, extract_features


def _load_texts(path: Path, field: str):
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)[field])
    return texts


def reference_stats(texts):
    vectors = [extract_features(t) for t in texts]
    stats = {}
    for name in FEATURE_NAMES:
        values = [v[name] for v in vectors]
        stats[name] = {"mean": mean(values), "stdev": pstdev(values) if len(values) > 1 else 1e-9}
    return stats


def z_scores(text: str, stats: dict) -> dict:
    features = extract_features(text)
    return {
        name: (features[name] - stats[name]["mean"]) / (stats[name]["stdev"] or 1e-9)
        for name in FEATURE_NAMES
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True, help="JSONL of real human reference samples")
    parser.add_argument("--human-field", default="text")
    parser.add_argument("--generated", type=Path, required=True, help="JSONL of model-generated samples to check")
    parser.add_argument("--generated-field", default="text")
    parser.add_argument("--within", type=float, default=1.0, help="Std-dev band counted as 'overlapping' human range")
    args = parser.parse_args(argv)

    human_texts = _load_texts(args.human, args.human_field)
    if len(human_texts) < 2:
        parser.error("Need at least 2 human reference samples to estimate a distribution")
    stats = reference_stats(human_texts)

    generated_texts = _load_texts(args.generated, args.generated_field)
    header = f"{'sample':>8}  " + "  ".join(f"{n:>20}" for n in FEATURE_NAMES)
    print(header)
    overlap_counts = []
    for i, text in enumerate(generated_texts):
        z = z_scores(text, stats)
        overlap = sum(1 for v in z.values() if abs(v) <= args.within)
        overlap_counts.append(overlap)
        row = f"{i:>8}  " + "  ".join(f"{z[n]:>20.2f}" for n in FEATURE_NAMES)
        print(row + f"   overlap={overlap}/{len(FEATURE_NAMES)}")

    if overlap_counts:
        print(f"\nAverage feature overlap with human range: {mean(overlap_counts):.1f}/{len(FEATURE_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

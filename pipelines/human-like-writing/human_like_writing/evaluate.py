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


_MIN_INFORMATIVE_STDEV = 1e-6


def reference_stats(texts):
    """Per-feature mean/stdev over the human reference sample.

    A feature whose reference stdev is ~0 (e.g. every reference sample has
    zero em-dashes) carries no discriminating signal: any generated text
    with the same near-zero value would trivially "match," so such
    features are marked non-informative rather than floored to a fake
    stdev that manufactures a match.
    """
    vectors = [extract_features(t) for t in texts]
    stats = {}
    for name in FEATURE_NAMES:
        values = [v[name] for v in vectors]
        stdev = pstdev(values) if len(values) > 1 else 0.0
        stats[name] = {
            "mean": mean(values),
            "stdev": stdev,
            "informative": stdev >= _MIN_INFORMATIVE_STDEV,
        }
    return stats


def z_scores(text: str, stats: dict) -> dict:
    """Per-feature z-score, or None for features with no reference variance."""
    features = extract_features(text)
    result = {}
    for name in FEATURE_NAMES:
        if not stats[name]["informative"]:
            result[name] = None
            continue
        result[name] = (features[name] - stats[name]["mean"]) / stats[name]["stdev"]
    return result


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

    n_informative = sum(1 for s in stats.values() if s["informative"])
    skipped = [name for name, s in stats.items() if not s["informative"]]
    if skipped:
        print(f"(skipping non-informative features with ~zero human variance: {', '.join(skipped)})\n")

    generated_texts = _load_texts(args.generated, args.generated_field)
    header = f"{'sample':>8}  " + "  ".join(f"{n:>20}" for n in FEATURE_NAMES)
    print(header)
    overlap_counts = []
    for i, text in enumerate(generated_texts):
        z = z_scores(text, stats)
        overlap = sum(1 for v in z.values() if v is not None and abs(v) <= args.within)
        overlap_counts.append(overlap)
        row = f"{i:>8}  " + "  ".join(
            f"{'n/a':>20}" if z[n] is None else f"{z[n]:>20.2f}" for n in FEATURE_NAMES
        )
        print(row + f"   overlap={overlap}/{n_informative}")

    if overlap_counts and n_informative:
        print(f"\nAverage feature overlap with human range: {mean(overlap_counts):.1f}/{n_informative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json

from human_like_writing.build_dataset import main


def _write_pairs(path, n):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": str(i), "styled": f"styled {i}!", "neutral": f"neutral {i}."}) + "\n")
        # one no-op pair that should be dropped
        f.write(json.dumps({"id": "noop", "styled": "same text", "neutral": "same text"}) + "\n")


def test_split_drops_noop_pairs_and_respects_fraction(tmp_path):
    infile = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "dataset"
    _write_pairs(infile, 20)

    main(["--in", str(infile), "--out-dir", str(out_dir), "--val-fraction", "0.2", "--seed", "1"])

    train = out_dir.joinpath("train.jsonl").read_text(encoding="utf-8").splitlines()
    val = out_dir.joinpath("val.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(train) + len(val) == 20  # the no-op pair was dropped
    assert len(val) == 4  # 20% of 20
    all_ids = {json.loads(line)["id"] for line in train + val}
    assert "noop" not in all_ids

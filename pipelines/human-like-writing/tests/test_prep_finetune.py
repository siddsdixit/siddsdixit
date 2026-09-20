import json

from human_like_writing.prep_finetune import main, to_example


def test_to_example_shape():
    example = to_example({"neutral": "flat text", "styled": "voicey text"})
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][1]["content"] == "flat text"
    assert example["messages"][2]["content"] == "voicey text"


def test_main_writes_one_example_per_pair(tmp_path):
    infile = tmp_path / "pairs.jsonl"
    outfile = tmp_path / "finetune.jsonl"
    pairs = [{"id": "1", "neutral": "n1", "styled": "s1"}, {"id": "2", "neutral": "n2", "styled": "s2"}]
    infile.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")

    main(["--in", str(infile), "--out", str(outfile)])

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["messages"][1]["content"] == "n1"

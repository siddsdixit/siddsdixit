import json

from human_like_writing.filter_domains import domain_matches, main


def test_domain_matches_exact_and_subdomain():
    domains = ["substack.com"]
    assert domain_matches("https://substack.com/post/1", domains)
    assert domain_matches("https://myblog.substack.com/post/1", domains)
    assert not domain_matches("https://example.com/post/1", domains)


def test_domain_matches_empty_url():
    assert not domain_matches("", ["substack.com"])


def test_main_filters_jsonl(tmp_path):
    infile = tmp_path / "in.jsonl"
    outfile = tmp_path / "out.jsonl"
    records = [
        {"id": "1", "text": "a", "url": "https://someblog.wordpress.com/p"},
        {"id": "2", "text": "b", "url": "https://news.example.com/p"},
    ]
    infile.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    main(["--in", str(infile), "--out", str(outfile)])

    kept = [json.loads(line) for line in outfile.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1
    assert kept[0]["id"] == "1"

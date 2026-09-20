import json

from human_like_writing import fetch_corpus


class _FakeResponse:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {"rows": self._rows}


def test_fetch_rows_paginates_and_stops_on_short_page(monkeypatch):
    pages = [
        [{"row_idx": 0, "row": {"text": "a", "timestamp": "t0", "url": "u0"}},
         {"row_idx": 1, "row": {"text": "b", "timestamp": "t1", "url": "u1"}}],
        [{"row_idx": 2, "row": {"text": "c", "timestamp": "t2", "url": "u2"}}],
    ]
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        page = pages[calls["n"]]
        calls["n"] += 1
        return _FakeResponse(page)

    monkeypatch.setattr(fetch_corpus.requests, "get", fake_get)
    monkeypatch.setattr(fetch_corpus, "PAGE_SIZE", 2)

    records = list(fetch_corpus.fetch_rows("c4", limit=5, sleep=0))
    assert [r["text"] for r in records] == ["a", "b", "c"]
    assert records[0]["id"] == "c4-0"


def test_main_writes_jsonl_and_applies_min_words(tmp_path, monkeypatch):
    rows = [
        {"row_idx": 0, "row": {"text": "short", "timestamp": "t", "url": "u"}},
        {"row_idx": 1, "row": {"text": " ".join(["word"] * 60), "timestamp": "t", "url": "u"}},
    ]

    def fake_get(url, params, timeout):
        return _FakeResponse(rows if params["offset"] == 0 else [])

    monkeypatch.setattr(fetch_corpus.requests, "get", fake_get)

    out = tmp_path / "out.jsonl"
    fetch_corpus.main(["--source", "c4", "--limit", "2", "--min-words", "10", "--out", str(out)])

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "c4-1"

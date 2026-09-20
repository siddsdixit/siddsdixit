"""Filter a fetched corpus down to blog/newsletter-platform domains.

Getting closer to a LinkedIn-shaped voice starts with narrowing general web
text to personal/professional blogging platforms rather than using general
web text as-is (build plan step 6).
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_DOMAINS = [
    "blogspot.com",
    "wordpress.com",
    "medium.com",
    "substack.com",
    "typepad.com",
    "tumblr.com",
]


def domain_matches(url: str, domains) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", action="append", dest="domains", help="Additional allowed domain (repeatable)")
    args = parser.parse_args(argv)

    domains = DEFAULT_DOMAINS + (args.domains or [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with args.infile.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            if domain_matches(record.get("url", ""), domains):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Kept {kept}/{total} records matching blog-platform domains -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

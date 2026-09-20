# human-like-writing pipeline

Runnable implementation of the build plan from
[`writing/human-like-ai-writing-corpus-and-finetuning-plan.md`](../../writing/human-like-ai-writing-corpus-and-finetuning-plan.md):
pull a pre-2022 corpus sample, strip it to neutral phrasing, pair
neutral/styled text into a training set, format it for fine-tuning, and
stylometrically check generated output against real human writing.

## Setup

```bash
cd pipelines/human-like-writing
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # only needed for the real (non --mock) neutralize step
```

## Stages

Each stage is a CLI subcommand under `human_like_writing.cli`, matching a
numbered step in the write-up's build plan:

| Command | Build plan step | What it does |
| --- | --- | --- |
| `fetch-corpus` | 1 | Pulls a small pre-2022 text sample via the public Hugging Face datasets-server API (currently wired to `allenai/c4`'s 2019 snapshot). No `datasets`/`pyarrow` dependency. |
| `filter-domains` | 6 | Narrows a fetched corpus down to blog/newsletter-platform domains (Blogspot, WordPress, Medium, Substack, ...) for a more LinkedIn/professional-blog-shaped source. Optional, run before `neutralize`. |
| `neutralize` | 2 | Strips styled text to flat, neutral phrasing. `--mock` runs an offline heuristic (contraction expansion, dropped emphasis punctuation) for testing the pipeline without an API key; the default calls the Anthropic Messages API to do the real neutralization. |
| `build-dataset` | 3 | Pairs neutral/styled records, drops empty or no-op pairs, and writes a shuffled train/val split. |
| `prep-finetune` | 4 | Formats pairs into a `{"messages": [...]}` chat JSONL (system/user/assistant), the format most fine-tuning APIs expect. Fine-tuning itself is kicked off with whichever provider's own tooling accepts this file. |
| `evaluate` | 5 | Computes lightweight stylometric features (sentence-length variability, type-token ratio, function-word rate, punctuation rates, etc.) for a set of generated samples and reports each one's z-score distance from a real human reference distribution, plus how many features fall within it. |

## End-to-end example

```bash
python -m human_like_writing.cli fetch-corpus \
  --source c4 --limit 500 --min-words 60 --out data/raw.jsonl

# Optional: narrow to blog/newsletter platforms first (step 6)
python -m human_like_writing.cli filter-domains \
  --in data/raw.jsonl --out data/raw.blogs.jsonl

python -m human_like_writing.cli neutralize \
  --in data/raw.jsonl --out data/pairs.jsonl   # add --mock to test without an API key

python -m human_like_writing.cli build-dataset \
  --in data/pairs.jsonl --out-dir data/dataset --val-fraction 0.1

python -m human_like_writing.cli prep-finetune \
  --in data/dataset/train.jsonl --out data/finetune_train.jsonl
python -m human_like_writing.cli prep-finetune \
  --in data/dataset/val.jsonl --out data/finetune_val.jsonl

# After fine-tuning, check generated samples (one JSON object per line, {"text": "..."})
# against real human reference text:
python -m human_like_writing.cli evaluate \
  --human data/raw.jsonl --generated data/generated_samples.jsonl
```

## Tests

```bash
pytest
```

Tests cover the stylometric feature extractor, the mock neutralizer, the
dataset split/dedup logic, fine-tune formatting, the evaluation report, and
`fetch-corpus` pagination (mocked HTTP, no network needed). They don't hit
the real Anthropic or Hugging Face APIs — run the end-to-end example above
for that.

## Notes / limitations

- `neutralize`'s `--mock` mode is a cheap heuristic (contraction expansion,
  dropped `!`/`?`, em-dash → comma) meant only to exercise the rest of the
  pipeline offline. It is not a substitute for the real LLM-based
  neutralization when actually building a training set.
- `evaluate`'s stylometric features are intentionally lightweight (no
  spaCy/StyloMetrix dependency) so they run anywhere with no setup. For a
  more rigorous check, feed generated samples through a dedicated tool
  like [StyloMetrix](https://github.com/ZILiAT-NASK/StyloMetrix) or the
  authorship-attribution methods the
  ["Catch Me If You Can? Not Yet"](https://arxiv.org/abs/2509.14543) paper
  uses.
- Actually running a fine-tuning job is provider-specific and isn't
  automated here — `prep-finetune`'s output is a generic chat-format JSONL
  meant to be handed to whichever fine-tuning API you're targeting.

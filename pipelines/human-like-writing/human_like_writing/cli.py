"""Single entry point for the human-like-writing pipeline.

Maps 1:1 onto the build plan in
``writing/human-like-ai-writing-corpus-and-finetuning-plan.md``:

  fetch-corpus    step 1  pull a small pre-2022 test sample
  filter-domains  step 6  narrow to blog/newsletter platforms (optional)
  neutralize      step 2  strip styled text to flat neutral phrasing
  build-dataset   step 3  pair + split into train/val
  prep-finetune   step 4  format pairs as fine-tuning-ready JSONL
  evaluate        step 5  stylometric check of generated vs. real human text

Run ``python -m human_like_writing.cli <command> --help`` for a command's
own arguments.
"""
import sys

from . import build_dataset, evaluate, fetch_corpus, filter_domains, neutralize, prep_finetune

COMMANDS = {
    "fetch-corpus": fetch_corpus.main,
    "filter-domains": filter_domains.main,
    "neutralize": neutralize.main,
    "build-dataset": build_dataset.main,
    "prep-finetune": prep_finetune.main,
    "evaluate": evaluate.main,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print(f"Usage: python -m human_like_writing.cli <{'|'.join(COMMANDS)}> [args...]")
        return 1
    command, rest = argv[0], argv[1:]
    return COMMANDS[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())

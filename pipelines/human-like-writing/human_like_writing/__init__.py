"""Neutral-to-styled paired fine-tuning pipeline.

Implements the build plan from
``writing/human-like-ai-writing-corpus-and-finetuning-plan.md`` as runnable
stages: fetch a pre-2022 corpus sample, optionally narrow it to
blog/newsletter-platform domains, neutralize it, pair it into a
train/val split, format it for fine-tuning, and stylometrically evaluate
generated output against real human samples.
"""

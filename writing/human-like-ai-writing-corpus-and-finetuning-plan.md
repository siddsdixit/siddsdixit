# Human-Like AI Writing: Corpus and Fine-Tuning Plan

*Sep 19, 2026 · [@Sid](https://github.com/siddsdixit)*

> The build plan below is implemented as a runnable pipeline in
> [`pipelines/human-like-writing/`](../pipelines/human-like-writing/) —
> fetch, neutralize, pair, prep-finetune, and evaluate stages, each mapped
> to a numbered step here.

## Why prompting alone doesn't work

A 2025 research paper, ["Catch Me If You Can? Not Yet"](https://arxiv.org/abs/2509.14543), tested whether LLMs can imitate an individual author's implicit writing style using few-shot prompting — feeding the model a handful of the author's own samples. It used forensic linguistics techniques including authorship attribution, authorship verification, and stylometric distance analysis to check the results rigorously. The finding: LLMs still struggle to imitate the implicit writing styles of everyday authors. Few-shot prompting is a known weak point across the field, not a failure specific to any one approach.

Separately, AI-generated text carries detectable linguistic fingerprints regardless of topic: consistent overuse of specific words and phrases, formulaic and repetitive sentence structure, and a tendency toward emotional hedging and cognitive-sounding phrasing that reads as smooth but generic. "Humanizer" tools that add noise or swap words can fool detectors without fixing this underlying flatness.

## The approach: neutral-to-styled paired fine-tuning

The method that shows real results is fine-tuning on a paired corpus rather than prompting:

1. Take source text written in the target voice (pre-2022 human writing)
2. Strip each sample down to flat, neutral phrasing — removing tone, rhythm, and personality
3. Pair the neutral version with the original styled version
4. Fine-tune a model to map neutral text back to the styled version

This produces more consistent tone matching across long-form content than either prompting or single-pass rewriting, because the model learns the transformation pattern rather than imitating a handful of examples.

## Pre-2022 corpus sources

The goal is human writing from before LLM chatbots went mainstream (pre-2022), so the source text isn't already contaminated with AI-flavored patterns.

### General web text, date-filterable

| Source | What it is | Link |
| --- | --- | --- |
| Common Crawl | Monthly snapshots with CDX indices; specific months from 2018-2021 can be pulled directly. WET files are text-only, already extracted. | [commoncrawl.org](https://commoncrawl.org) |
| C4 (allenai/c4) | Built from the April 2019 Common Crawl snapshot, already cleaned of gibberish and non-English text. Loadable via the Hugging Face `datasets` library or Git LFS. | [huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4) |

### Blog and long-form personal writing

| Source | What it is | Link |
| --- | --- | --- |
| Blog Authorship Corpus | Real personal blog posts with metadata (age, date, occupation), usable for research purposes. | [huggingface.co/datasets/barilan/blog_authorship_corpus](https://huggingface.co/datasets/barilan/blog_authorship_corpus) |
| Common Corpus (PleIAs) | Large curated public-domain collection including newspapers and books; useful as a style-neutral baseline. | [huggingface.co/datasets/PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus) |

**Caveat:** none of these are LinkedIn-shaped — they're general web or personal-blog voice, not professional-post voice. Getting closer to a LinkedIn-specific target likely means filtering these snapshots down to blog and newsletter platform domains (Blogspot, WordPress, Medium, Substack) using the WAT metadata files, or separately sourcing archived professional blogs from that era.

## Build plan

1. Pull a small test sample first (a few hundred posts from the Blog Authorship Corpus, or a filtered C4 slice) before committing to a full download — validate the neutral-to-styled pairing approach at small scale
2. Build the neutralization step: use a model to strip a sample of pre-2022 posts down to flat, neutral phrasing
3. Pair neutral and original versions into a training set
4. Fine-tune on the pairs: neutral in, styled out
5. Hold out a test set and run stylometric checks (the same authorship attribution and verification methods the research paper used) on generated output, rather than eyeballing it
6. If targeting LinkedIn voice specifically, filter source corpora down to blog-platform domains before step 2, rather than using general web text as-is

## Further research: implementations and detection metrics

### Existing style-transfer implementations

Most open code targets formality or bias-neutralization rather than pure authorial voice, but a couple are directly relevant: TinyStyler does few-shot text style transfer using authorship embeddings, and Fast Forward Labs' text-style-transfer includes reusable modules for style intensity classification and content preservation scoring — useful for checking that your fine-tuned output kept the meaning while changing the voice.

### Stylometric detection tools for validating output

Stylometry — quantifying vocabulary, syntax, and punctuation patterns to fingerprint a writer — is the standard way to check whether generated text actually reads as human. Recent research found that while AI-generated text is stylistically uniform, human writing is marked by variability and individuality, which is exactly the gap a fine-tuned model needs to close. On the tooling side: StyloMetrix is an open-source library used in recent stylometry research to extract linguistic, grammatical, and syntactic features as numeric scores. For quick checks, GitHub's stylometry topic page lists several transformer-based and n-gram tools built specifically to compare human versus LLM writing.

The practical workflow: fine-tune, generate a sample, run it through a stylometric feature extractor alongside real pre-2022 human samples, and check whether the feature distributions overlap — that's a harder bar to clear than fooling a detector, and closer to what you're actually after.

## Testable hypothesis: do voice-trained models write more human-like text?

The hypothesis: models built or tuned for conversational voice interaction may produce more human-sounding written text than text-only frontier models, because they're trained on real spoken exchanges rather than purely written corpora. Research on spoken dialogue systems supports part of this — real speech conversation data can provide more natural, spontaneous behaviors such as pauses and interruptions that don't show up in written training data — but that work targets naturalness of spoken output, not written prose, so whether it carries over to text on the page is unconfirmed and worth testing directly.

**Proposed test:** run the same writing prompt (a LinkedIn-style piece, ideally one already used to test other models) through a voice/conversational-tuned model and compare the output against the standard text-frontier models already in scope, using the same stylometric checks described above (feature-distribution overlap against real pre-2022 human samples, plus a read for disfluency-style looseness that plain text models tend to lack).

**Candidate model:** Gemini 3.8 Live. Google's voice-specific model (distinct from the text/coding-focused Gemini 3.8 Flash) is built for natural spoken conversation — it processes visual input in near real-time and automatically detects and transitions between 97 supported languages mid-conversation. The Extended Thinking variant took the number one spot on Artificial Analysis's Speech to Speech Quality Index at 82.6, a score built from conversational coherence, natural prosodic cadence, acoustic clarity, and resilience. Important caveat: that score measures how it sounds when speaking, not how its written text reads on a page — there's no existing benchmark for its written human-likeness, which is exactly the gap this test would fill.

Sid's hypothesis: if a model sounds more human in speech, there's no obvious reason its written output shouldn't also read more human — the same underlying training on real human conversational exchange should shape both the acoustic delivery and the word choice, phrasing, and rhythm of what it writes. Worth testing directly rather than assuming either way.

**Status:** not yet run — flagged as the next experiment once a candidate voice model is confirmed and accessible for text-generation testing.

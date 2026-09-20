"""Lightweight stylometric feature extraction (no heavy NLP dependencies).

Implements the kind of vocabulary/syntax/punctuation fingerprinting the
write-up calls "stylometry": cheap enough to run on every sample, and
enough to surface the uniformity-vs-variability gap between AI and human
text that the research it cites found.
"""
import re
from collections import Counter
from statistics import mean, pstdev

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

FUNCTION_WORDS = {
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "for",
    "it", "with", "as", "on", "by", "at", "this", "but", "from", "or",
}


def _sentences(text: str):
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _words(text: str):
    return [w.lower() for w in _WORD_RE.findall(text)]


def extract_features(text: str) -> dict:
    words = _words(text)
    sentences = _sentences(text) or [text]
    sentence_lengths = [len(_words(s)) or 1 for s in sentences]
    n_words = len(words) or 1
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)

    return {
        "avg_sentence_len": mean(sentence_lengths),
        "sentence_len_stdev": pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        "avg_word_len": mean(len(w) for w in words) if words else 0.0,
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": hapax / n_words,
        "function_word_rate": sum(counts[w] for w in FUNCTION_WORDS) / n_words,
        "comma_rate": text.count(",") / n_words * 1000,
        "em_dash_rate": (text.count("—") + text.count("--")) / n_words * 1000,
        "exclamation_rate": text.count("!") / n_words * 1000,
        "semicolon_rate": text.count(";") / n_words * 1000,
    }


FEATURE_NAMES = tuple(extract_features("Sample text used only to discover feature names.").keys())

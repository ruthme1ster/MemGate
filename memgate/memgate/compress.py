"""Extractive compression — choosing WHICH words to keep, not just how many.

The compressor was head-truncation: keep the first N words, drop the rest. On
natural dialogue that is close to the worst possible rule. The head of a turn is
usually greeting and phatic filler --

    "[Caroline] Hey Mel! Good to see you! How have you been?"

-- while the durable content (dates, names, numbers, places) sits later in the
sentence. Head-truncation therefore spends its whole budget on the least
informative part of the turn. It is why fixing consolidation raised the
write-path ceiling from 9.4% to 25.5% but converted only ~1.4 points of it:
the gists retained turn *ids* while discarding the turn *content*.

This module keeps the most informative words instead, in their original order,
so the result still reads as a compressed sentence. It is the same
rate-distortion question the project is about, applied at word level: given a
token budget, which words preserve the most downstream utility?

NOT abstractive. A real summariser rewrites; this only selects. Abstractive
re-summarisation by the base LLM remains the Phase 2 target and needs an LLM
that is not available offline -- but this establishes the extractive baseline it
has to beat, which is what makes that a measurable experiment.
"""
import re
from typing import List

from .utils import count_tokens

# Words that almost never carry durable information on their own.
_STOP = {
    "a", "an", "the", "and", "or", "but", "so", "if", "then", "than", "as",
    "at", "by", "for", "from", "in", "into", "of", "on", "to", "with", "about",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "there", "here", "what", "which", "who", "how", "when",
    "very", "really", "just", "quite", "also", "too", "some", "any", "much",
    "many", "more", "most", "well", "get", "got", "like", "know", "think",
    "oh", "ah", "yeah", "yes", "no", "ok", "okay", "hi", "hey", "hello",
    "thanks", "thank", "please", "sure", "good", "great", "nice", "cool",
}

_MONTH = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_HAS_DIGIT = re.compile(r"\d")
_WORDY = re.compile(r"[A-Za-z0-9']")
_SPEAKER = re.compile(r"^\s*\[([^\]]{1,40})\]\s*")


def _score_word(w: str, pos: int, n: int, first_word: bool) -> float:
    """Higher = more worth keeping."""
    bare = w.strip(".,!?;:'\"()").lower()
    if not bare:
        return -1.0

    s = 1.0
    if _HAS_DIGIT.search(bare):
        s += 4.0                       # dates, amounts, counts: the answers
    if _MONTH.match(bare):
        s += 3.0
    # proper noun: capitalised but not merely sentence-initial
    if w[:1].isupper() and not first_word:
        s += 2.5
    if bare in _STOP:
        s -= 2.0
    if len(bare) >= 7:
        s += 0.5                       # long words tend to be contentful
    elif len(bare) <= 2:
        s -= 0.5
    # mild preference for earlier words when otherwise tied, so the compressed
    # sentence keeps its subject rather than drifting to the tail
    s += 0.25 * (1.0 - pos / max(1, n))
    return s


def compress(text: str, max_tokens: int) -> str:
    """Keep the most informative words of `text` within `max_tokens`.

    Order is preserved, and the "[Speaker]" tag is always kept -- attribution is
    what makes a gist interpretable later, and it costs 2-3 tokens.
    """
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    m = _SPEAKER.match(text)
    prefix, body = (m.group(0).strip(), text[m.end():]) if m else ("", text)

    budget = max_tokens - (count_tokens(prefix) if prefix else 0)
    if budget <= 0:
        return prefix

    words = body.split()
    if not words:
        return prefix

    scored = [(_score_word(w, i, len(words), i == 0), i, w)
              for i, w in enumerate(words)]
    scored = [t for t in scored if t[0] >= 0]
    scored.sort(key=lambda t: -t[0])

    kept, used = [], 0
    for s, i, w in scored:
        c = count_tokens(w)
        if used + c > budget:
            continue
        kept.append((i, w))
        used += c
        if used >= budget:
            break
    if not kept:
        return prefix

    kept.sort()
    out = " ".join(w for _, w in kept)
    return f"{prefix} {out}".strip() if prefix else out


def informative_head(text: str, max_words: int) -> str:
    """Budget expressed in words rather than tokens, for the merge path."""
    return compress(text, max_tokens=max(1, int(max_words * 1.4)))

"""Scoring policies — the Adaptive Memory Decision Engine.

P1 (heuristic) is implemented. P2 (LLM-judged) and P3 (distilled) share the
same `Scorer` interface so they drop in without touching the router.
"""
import re
from typing import Tuple
from .utils import words

# --- signal vocabularies ----------------------------------------------------
DURABLE_HINTS = {
    "deadline", "budget", "due", "name", "called", "prefer", "prefers", "must",
    "should", "need", "needs", "requirement", "decided", "decision", "chose",
    "using", "email", "phone", "address", "goal", "target", "constraint",
    "remember", "always", "never", "born", "lives", "works",
}
# Correction cues -- the marker of a genuine supersession ("the old value is now
# wrong"). Deliberately does NOT include "instead": in "I decided to use Django
# for the board instead of Rails" the word is *contrastive phrasing inside one
# new statement*, not a correction of an earlier one. Treating it as an update
# cue mislabelled 6 turns on a set that plants exactly 1 supersession, and since
# those decision sentences are near-identical in wording they then retired each
# other -- a cascading false supersession that cost ~10 recall points.
CORRECTION_HINTS = {"actually", "changed", "update", "updated", "correction",
                    "revised", "moved", "postponed", "rescheduled", "push",
                    "pushed"}
# Multi-word cues can't be matched against a set of single words.
CORRECTION_PHRASES = ("no longer", "not anymore", "scratch that", "i meant")
FILLER_PAT = re.compile(
    r"^(ok(ay)?|thanks?|thank you|cool|nice|great|got it|sure|yes|no|yeah|"
    r"hi|hey|hello|good morning|good evening|bye|see you|np|sounds good)[\s!.,]*$",
    re.I,
)
NUM_PAT = re.compile(r"\d")
MONTH_PAT = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
MONEY_PAT = re.compile(r"[$£€₹]\s?\d|\b\d+\s?(k|rs|usd|inr|dollars|rupees)\b", re.I)
PROPER_PAT = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}")


class Scorer:
    """Interface. Returns (utility in [0,1], type_label)."""
    name = "base"

    def score(self, text: str, speaker: str = "user") -> Tuple[float, str]:
        raise NotImplementedError


class RecencyScorer(Scorer):
    """P0 — no content judgement at all. Everything scores equally."""
    name = "P0-recency"

    def score(self, text, speaker="user"):
        return 0.0, "unscored"


# Every signal the heuristic uses, individually switchable so each one can be
# ablated. "Which component earns the gain?" is only answerable if components
# can be turned off one at a time.
SIGNALS = ("filler", "durable", "numeric", "temporal", "proper", "question",
           "speaker", "update")


class HeuristicScorer(Scorer):
    """P1 — rules + shallow NER, no LLM calls.

    Signals: filler detection, durability hints, numerals, temporal/money
    patterns, proper nouns, question down-weight, speaker down-weight, and
    correction detection.

    `signals` selects which are active; omitting one is that signal's ablation.
    """
    name = "P1-heuristic"

    def __init__(self, signals=SIGNALS):
        self.signals = frozenset(signals)

    def _on(self, s):
        return s in self.signals

    def score(self, text, speaker="user"):
        t = text.strip()
        if self._on("filler") and (FILLER_PAT.match(t) or len(words(t)) <= 2):
            return 0.02, "filler"

        w = set(words(t))
        score = 0.15                      # base
        label = "context"

        if self._on("durable"):
            hits = len(w & DURABLE_HINTS)
            if hits:
                score += min(0.35, 0.18 * hits)
                label = "fact"

        if self._on("numeric") and NUM_PAT.search(t):
            score += 0.20
            label = "fact"
        if self._on("temporal") and (MONEY_PAT.search(t) or MONTH_PAT.search(t)):
            score += 0.15
            label = "fact"
        if self._on("proper") and PROPER_PAT.search(t):
            score += 0.12
            label = "fact"

        # questions are usually not durable facts themselves
        if self._on("question") and t.endswith("?"):
            score -= 0.12
            if label != "fact":
                label = "query"

        # assistant chatter is less likely to be a durable user fact
        if self._on("speaker") and speaker == "assistant":
            score -= 0.10

        if self._on("update"):
            tl = t.lower()
            corrected = (bool(w & CORRECTION_HINTS)
                         or any(p in tl for p in CORRECTION_PHRASES))
            if corrected and (NUM_PAT.search(t) or MONTH_PAT.search(t)):
                label = "update"
                score += 0.10

        return max(0.0, min(1.0, score)), label


class LLMJudgeScorer(Scorer):
    """P2 — stub. Phase 3.

    Prompt plan: give the model the turn plus recent context, ask for
    {"importance": 0-1, "type": fact|decision|preference|task|filler}.
    Cache by turn hash; call only on eviction, never per turn.
    """
    name = "P2-llm-judge"

    def __init__(self, client=None):
        self.client = client

    def score(self, text, speaker="user"):
        raise NotImplementedError("P2 lands in Phase 3 — see build plan step 3.")


class DistilledScorer(Scorer):
    """P3 — stub. Trained on P2's labels to reach P2 quality at P1 cost."""
    name = "P3-distilled"

    def __init__(self, model=None):
        self.model = model

    def score(self, text, speaker="user"):
        raise NotImplementedError("P3 lands in Phase 3 — trained on P2 labels.")

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
    """P3 — stub. Trained on P2's labels to reach P2 quality at P1 cost.

    Still a stub because P2 is blocked: distilling P2 requires P2's labels, and
    P2 requires a generative model this project does not have offline. See
    `LearnedScorer` for the same slot filled by a different supervision source.
    """
    name = "P3-distilled"

    def __init__(self, model=None):
        self.model = model

    def score(self, text, speaker="user"):
        raise NotImplementedError("P3 lands in Phase 3 — trained on P2 labels.")


# --------------------------------------------------------------- P3-learned
# Feature names, in the order `features()` emits them. Kept explicit so the
# fitted coefficients can be read back and reported -- a learned scorer that
# cannot be inspected is a worse research object than the heuristic it replaces.
FEATURE_NAMES = (
    "n_words", "n_chars", "has_digit", "has_month", "has_money", "has_proper",
    "is_question", "ends_period", "durable_hits", "correction_hint",
    "is_filler", "speaker_assistant", "frac_stop", "n_caps", "has_time",
)

TIME_PAT = re.compile(r"\b\d{1,2}\s?(am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)


def features(text: str, speaker: str = "user"):
    """Hand features for one turn. Cheap, inspectable, and label-free.

    These are the SAME signals the P1 heuristic uses, but handed to the model as
    evidence rather than combined by hand-chosen weights. That is the whole
    experiment: P1 fixes the weights by intuition, P3-learned fits them. If the
    fitted version does no better, the heuristic's weights were already fine and
    the scorer is not the bottleneck after all.
    """
    t = text.strip()
    w = words(t)
    ws = set(w)
    n = max(1, len(w))
    return [
        len(w),
        len(t),
        1.0 if NUM_PAT.search(t) else 0.0,
        1.0 if MONTH_PAT.search(t) else 0.0,
        1.0 if MONEY_PAT.search(t) else 0.0,
        1.0 if PROPER_PAT.search(t) else 0.0,
        1.0 if t.endswith("?") else 0.0,
        1.0 if t.endswith(".") else 0.0,
        float(len(ws & DURABLE_HINTS)),
        1.0 if (ws & CORRECTION_HINTS) else 0.0,
        1.0 if (FILLER_PAT.match(t) or len(w) <= 2) else 0.0,
        1.0 if speaker == "assistant" else 0.0,
        sum(1 for x in w if len(x) <= 3) / n,
        sum(1 for c in t if c.isupper()),
        1.0 if TIME_PAT.search(t) else 0.0,
    ]


class LearnedScorer(Scorer):
    """P3-learned — a fitted salience model in place of hand-tuned weights.

    Fills the P3 slot ("a scorer trained rather than written") from a different
    supervision source than the build plan assumed. P2's LLM labels are
    unavailable offline, so the target here is DISTANT SUPERVISION from the
    benchmark: was this turn ever cited as evidence by a question?

    TRAIN/TEST DISCIPLINE -- read before trusting any number this produces.

    The label is derived from ground truth, so it may only ever be seen for
    conversations the model is not evaluated on. Training is leave-one-
    conversation-out: to score conv-26, the model is fitted on the other nine
    and has never seen a single conv-26 turn or label. At inference `score()`
    sees text and speaker and nothing else -- no fact_id, no evidence set, no
    question -- so the label-invariance test in test_memgate.py covers this
    scorer exactly as it covers P1, and it must keep passing.

    WHAT IT DOES AND DOES NOT SHOW. A deployed agent has no future questions, so
    this is not a drop-in component; it is a measurement of the HEADROOM a
    learned scorer has over a hand-written one, given good supervision and no
    test-time leakage. That is the §4 research question -- how much of the gap a
    better decision policy can recover -- answered for the strongest scorer this
    project can build offline. Treat it as an informed upper bound on P2/P3,
    strictly below the Oracle bound and strictly above P1.
    """
    name = "P3-learned"

    def __init__(self, model=None, use_embedding=True, embedder=None):
        self.model = model                # fitted sklearn estimator
        self.use_embedding = use_embedding
        self._embed = embedder            # injected to avoid a circular import

    def score(self, text, speaker="user"):
        if self.model is None:
            raise RuntimeError("LearnedScorer is unfitted — see train_scorer.py")
        x = features(text, speaker)
        if self.use_embedding and self._embed is not None:
            x = list(x) + list(self._embed(text))
        p = float(self.model.predict_proba([x])[0][1])
        # Reuse P1's label vocabulary so routing, supersession and the type
        # counters behave identically -- only the utility number changes, which
        # is what makes this a clean single-component swap.
        t = text.strip()
        w = set(words(t))
        if FILLER_PAT.match(t) or len(words(t)) <= 2:
            label = "filler"
        elif (w & CORRECTION_HINTS) and (NUM_PAT.search(t) or MONTH_PAT.search(t)):
            label = "update"
        elif t.endswith("?"):
            label = "query"
        else:
            label = "fact" if p >= 0.5 else "context"
        return max(0.0, min(1.0, p)), label

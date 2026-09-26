"""Online learning of the decision policy — the part that makes it deployable.

WHY THIS EXISTS
---------------
`LearnedScorer` (P3) is the strongest scorer this project can fit, and its
docstring is explicit that it is NOT a component: it is supervised by "was this
turn ever cited as evidence", which requires the future questions. A live agent
does not have them. P3 therefore bounds the headroom and cannot be shipped, and
every deliverable since §24 has listed online learning as the only route that
closes that gap.

THE SUPERVISION PROBLEM, AND THE SIGNAL THIS USES
-------------------------------------------------
The tempting online label is "did the agent answer correctly", but offline that
correctness comes from the gold answer, which would smuggle the evaluation label
back into the write path -- §13's leak wearing a different coat.

So this learns from something an agent genuinely observes at runtime and that
contains no ground truth at all: WHICH STORED ITEMS RETRIEVAL ACTUALLY PULLED
INTO CONTEXT. Every query ranks the store against itself; items that keep
surfacing are, empirically, the kind of thing this user asks about. Items that
sit in the store for many queries and are never retrieved are not.

    positive  features of items retrieved into the assembled context
    negative  features of items held but not retrieved

No question's gold evidence is consulted, no answer text is read, and the update
happens after the fact from the policy's own behaviour. A deployed agent can run
this loop unchanged, which is the whole point.

WHAT IT CAN AND CANNOT SHOW
---------------------------
This does not learn what is TRUE, it learns what is ASKED ABOUT -- retrieval
feedback is a proxy, and a self-reinforcing one: an item that is never retrieved
can never become positive. The honest framing is that it tests whether a
label-free runtime signal recovers any of the gap between the hand-written
heuristic (P1) and the fitted upper bound (P3), not whether it reaches P3.

COLD START
----------
Starting from random weights would make the first thousand turns worse than P1
for reasons having nothing to do with learning, and the comparison would then be
measuring the warm-up. The score is therefore a blend that starts as pure
heuristic and hands over to the model as evidence accumulates:

    u = (1 - a) * heuristic(text) + a * model(text),   a = n / (n + half_life)

so the policy is never worse than P1 by construction at n=0, and the experiment
asks whether `a` rising is an improvement or a regression.
"""
import math
from typing import Sequence, Tuple

import numpy as np

from .scoring import (Scorer, HeuristicScorer, features, FEATURE_NAMES,
                      FILLER_PAT, CORRECTION_HINTS, NUM_PAT, MONTH_PAT)
from .utils import words


class OnlineScorer(Scorer):
    """A salience scorer updated from retrieval feedback during the run.

    Interface is `Scorer`, so it drops into the same router as P0/P1/P2/P3 and
    the single-component-swap discipline holds: policy, budgets, retrieval and
    compression stay pinned and only the scorer varies.

    It never sees `fact_id`, `is_filler`, an evidence set or a question's
    answer -- `score()` takes text and speaker exactly as P1 does, and
    `feedback()` takes item TEXT only. The label-invariance test therefore
    covers it unchanged, which is asserted in test_memgate.py rather than
    claimed here.
    """

    name = "P4-online"

    def __init__(self, lr: float = 0.08, half_life: int = 400,
                 l2: float = 1e-4, max_negatives: int = 8, seed: int = 20260921):
        self.w = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.half_life = half_life
        self.l2 = l2
        self.max_negatives = max_negatives
        self.updates = 0
        self.positives = 0
        self.negatives = 0
        self._prior = HeuristicScorer()
        self._rng = np.random.default_rng(seed)
        # Running feature standardisation. Raw features differ by orders of
        # magnitude (n_chars in the hundreds, has_digit in {0,1}), and an
        # unscaled SGD step then moves almost entirely along the longest axis --
        # the model would effectively learn "length" and nothing else, which is
        # the one thing train_scorer.py already showed a fitted model finds.
        self._mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self._var = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        self._seen = 0

    # ------------------------------------------------------------ internals
    def _observe_stats(self, x: np.ndarray):
        """Welford update, so standardisation needs no second pass."""
        self._seen += 1
        d = x - self._mean
        self._mean += d / self._seen
        self._var += d * (x - self._mean)

    def _z(self, x: np.ndarray) -> np.ndarray:
        if self._seen < 2:
            return np.zeros_like(x)
        sd = np.sqrt(np.maximum(self._var / (self._seen - 1), 1e-9))
        return np.clip((x - self._mean) / sd, -6.0, 6.0)

    def _p(self, x: np.ndarray) -> float:
        z = float(np.dot(self.w, self._z(x)) + self.b)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def _alpha(self) -> float:
        """How much of the score the learned model owns, in [0, 1)."""
        return self.updates / (self.updates + self.half_life)

    # ------------------------------------------------------------ scoring
    def score(self, text: str, speaker: str = "user") -> Tuple[float, str]:
        prior, label = self._prior.score(text, speaker)
        x = np.asarray(features(text, speaker), dtype=np.float64)
        self._observe_stats(x)
        a = self._alpha()
        if a <= 0.0:
            return prior, label
        u = (1.0 - a) * prior + a * self._p(x)
        # Labels stay with the heuristic: "update"/"filler" detection is
        # regex-reliable and routing depends on it, so only the UTILITY number
        # is learned. Same split MLXJudgeScorer uses, and for the same reason.
        return max(0.0, min(1.0, u)), label

    # ------------------------------------------------------------ learning
    def feedback(self, retrieved: Sequence[str], held: Sequence[str],
                 speaker: str = "user"):
        """One learning step from a query's retrieval outcome.

        `retrieved` is the text of items the read path actually pulled into the
        assembled context; `held` is the text of items the store was holding and
        did not. Both are plain strings -- no ids, no labels, nothing the policy
        could not see at runtime.

        Negatives are subsampled because a store holds far more unretrieved
        items than retrieved ones, and an unbalanced step would simply drive
        every score to zero.
        """
        pos = [t for t in dict.fromkeys(retrieved) if t]
        if not pos:
            return
        seen = set(pos)
        neg_pool = [t for t in held if t and t not in seen]
        if neg_pool:
            k = min(self.max_negatives, len(neg_pool), max(1, len(pos)))
            idx = self._rng.choice(len(neg_pool), size=k, replace=False)
            neg = [neg_pool[i] for i in idx]
        else:
            neg = []

        for text, y in [(t, 1.0) for t in pos] + [(t, 0.0) for t in neg]:
            x = np.asarray(features(text, speaker), dtype=np.float64)
            self._observe_stats(x)
            z = self._z(x)
            p = self._p(x)
            g = (p - y)
            self.w -= self.lr * (g * z + self.l2 * self.w)
            self.b -= self.lr * g
            self.updates += 1
            if y > 0.5:
                self.positives += 1
            else:
                self.negatives += 1

    # ------------------------------------------------------------ reporting
    def weights(self):
        """Fitted weights by feature name, largest magnitude first.

        A learned decision policy that cannot be inspected is a worse research
        object than the heuristic it replaces (scoring.py makes the same point
        about P3), so this is part of the interface rather than a debug aid.
        """
        order = np.argsort(-np.abs(self.w))
        return [(FEATURE_NAMES[i], float(self.w[i])) for i in order]

    def stats(self):
        return {"updates": self.updates, "positives": self.positives,
                "negatives": self.negatives, "alpha": self._alpha(),
                "bias": self.b}

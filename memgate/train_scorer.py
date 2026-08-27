#!/usr/bin/env python3
"""P3-learned — fit the salience scorer instead of hand-writing its weights.

Every diagnostic this project has run names the same bottleneck. `diagnose.py`
attributes synthetic misses `{scorer_miss: 18, dedupe_collapse: 1,
retrieval_miss: 2}`; the LoCoMo ablation puts scored eviction at -5.0 strict
against FIFO. The scorer is where the headroom is, and P1 sets its weights by
intuition.

The build plan closed that gap with P2 (an LLM rating each turn) distilled into
P3. P2 needs a generative model, which is not available offline -- no API key,
no local instruct model -- so this fills the same slot from the only supervision
available: LoCoMo's own evidence sets. The label is "was this turn ever cited as
evidence by a question about this conversation?", which is true for 31.5% of
turns.

WHY THIS IS NOT THE §13 LEAK
----------------------------
§13's leak read `fact_id` -- the answer key for the item being decided -- at
DECISION time. This never does. Training is leave-one-conversation-out: the
model that scores conv-26 was fitted on the other nine and has never seen a
conv-26 turn or label. At inference the scorer sees text and speaker only. The
label-invariance test covers it unchanged, and is asserted below rather than
merely claimed.

The distinction that matters for the report: this measures the HEADROOM a
learned scorer has over a hand-written one under honest cross-validation. It is
not a deployable component, because a live agent has no future questions to
learn from. It bounds P2/P3 from below, and the Oracle bounds it from above.

Usage:
    python3 train_scorer.py
    python3 train_scorer.py --store-budget 4096
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from memgate.data import load_locomo
from memgate.policies import MemGateSelectPolicy
from memgate.harness import run_policy
from memgate.scoring import (HeuristicScorer, LearnedScorer, features,
                             FEATURE_NAMES)
from memgate.utils import backend_info, warm_cache, reset_cache, embed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONTEXT_BUDGET = 2048


def build_xy(turns, questions, use_embedding):
    """Feature matrix and evidence labels for one conversation."""
    cited = set()
    for q in questions:
        cited.update(q.evidence_ids)
    X, y = [], []
    for t in turns:
        f = features(f"[{t.speaker}] {t.text}", t.speaker)
        if use_embedding:
            f = list(f) + list(embed(f"[{t.speaker}] {t.text}"))
        X.append(f)
        y.append(1 if t.fact_id in cited else 0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


def fit(X, y):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    ).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=2048)
    args = ap.parse_args()

    print("\nMemGate — P3-learned scorer (leave-one-conversation-out)")
    print("=" * 80)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)
    print(f"  context budget {CONTEXT_BUDGET}   store budget {args.store_budget}")
    print("  policy: P1-S select (the §23 winner) — only the SCORER varies\n")

    rows = []
    for use_emb in (False, True):
        tag = "hand+MiniLM" if use_emb else "hand features only"
        feats = [build_xy(ts, qs, use_emb) for _s, ts, qs in data]

        aucs, aps = [], []
        tot_q = tot_hits = ans_n = ans_hits = 0
        h_q = h_hits = h_an = h_ah = 0
        coef_sum = np.zeros(len(FEATURE_NAMES))
        for i, (sid, ts, qs) in enumerate(data):
            Xte, yte = feats[i]
            Xtr = np.concatenate([feats[j][0] for j in range(len(data)) if j != i])
            ytr = np.concatenate([feats[j][1] for j in range(len(data)) if j != i])
            # Fold integrity. A silent off-by-one here would train on the
            # conversation being scored and turn the whole result into the §13
            # leak wearing a lab coat, so it is asserted rather than assumed.
            assert len(Xtr) + len(Xte) == sum(len(f[0]) for f in feats), \
                "fold sizes do not partition the corpus"
            model = fit(Xtr, ytr)
            # intrinsic: can it tell evidence turns from the rest, unseen?
            p = model.predict_proba(Xte)[:, 1]
            if len(set(yte)) > 1:
                aucs.append(roc_auc_score(yte, p))
                aps.append(average_precision_score(yte, p))
            coef_sum += model[-1].coef_[0][:len(FEATURE_NAMES)]

            # extrinsic: does the memory system actually get better?
            sc = LearnedScorer(model=model, use_embedding=use_emb, embedder=embed)
            r = run_policy(MemGateSelectPolicy(budget=CONTEXT_BUDGET,
                                               store_budget=args.store_budget,
                                               scorer=sc), ts, qs, CONTEXT_BUDGET)
            tot_q += r["n"]; tot_hits += r["hits"]
            ans_n += r["answer_n"]; ans_hits += r["answer_hits"]

            rh = run_policy(MemGateSelectPolicy(budget=CONTEXT_BUDGET,
                                                store_budget=args.store_budget,
                                                scorer=HeuristicScorer()),
                            ts, qs, CONTEXT_BUDGET)
            h_q += rh["n"]; h_hits += rh["hits"]
            h_an += rh["answer_n"]; h_ah += rh["answer_hits"]

        L = (tot_hits / tot_q * 100, ans_hits / ans_n * 100)
        H = (h_hits / h_q * 100, h_ah / h_an * 100)
        print(f"  {tag}")
        print(f"    scorer AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   "
              f"AP {np.mean(aps):.3f}  (base rate "
              f"{sum(int(v) for _, y in feats for v in y)/sum(len(y) for _, y in feats):.3f})")
        print(f"    P1  heuristic   strict {H[0]:5.1f}%   answer {H[1]:5.1f}%")
        print(f"    P3  learned     strict {L[0]:5.1f}%   answer {L[1]:5.1f}%   "
              f"(delta {L[0]-H[0]:+.1f} / {L[1]-H[1]:+.1f})\n")
        rows.append({"features": tag, "auc": np.mean(aucs), "ap": np.mean(aps),
                     "p1_strict": H[0], "p1_answer": H[1],
                     "p3_strict": L[0], "p3_answer": L[1],
                     "store_budget": args.store_budget})
        if not use_emb:
            order = np.argsort(-np.abs(coef_sum / len(data)))
            print("    fitted weights, largest magnitude first "
                  "(mean over folds, standardised units):")
            for k in order[:8]:
                print(f"      {FEATURE_NAMES[k]:<18}{coef_sum[k]/len(data):+7.3f}")
            print()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "learned_scorer.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

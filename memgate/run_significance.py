#!/usr/bin/env python3
"""Significance testing — are the headline gaps real, or sampling noise?

Every number this project has reported is a point estimate over 1527 questions.
"+8.7 answer points" means nothing until it comes with an interval, and the
paper cannot claim an improvement without one.

Two properties of this experiment decide the right test:

  PAIRED. Every policy answers the SAME questions, so the comparison is
  within-question. An unpaired test would throw away that pairing and badly
  overstate the uncertainty: the policies agree on most questions, and only the
  disagreements carry information.

  CLUSTERED. Questions are not independent. LoCoMo has 10 conversations, and a
  conversation whose turns happen to suit a policy makes all ~150 of its
  questions succeed together. Resampling QUESTIONS treats those as 150
  independent draws and understates the true interval. Resampling
  CONVERSATIONS respects the dependence.

  Both are reported. Where the conversation-level interval crosses zero and the
  question-level one does not, the honest reading is the conversation-level
  one -- with only 10 clusters it is wide, and saying so is the result.

Also reports McNemar's exact test on the discordant pairs, which is the
standard paired test for binary outcomes and needs no resampling at all.

Usage:
    python3 run_significance.py
    python3 run_significance.py --store-budget 2048
"""
import argparse
import csv
import os
import sys
from math import comb

import numpy as np

from memgate.data import load_locomo
from memgate.policies import (RAGPolicy, SlidingWindowPolicy,
                              MemGateSelectPolicy, MemGateAdaptivePolicy)
from memgate.harness import run_policy
from memgate.utils import backend_info, warm_cache, reset_cache

CONTEXT_BUDGET = 2048
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
B = 10000
RNG = np.random.default_rng(20260827)


def collect(factory, data, store_budget, **kw):
    """Per-question outcomes, tagged with their conversation."""
    strict, answer, conv = [], [], []
    for ci, (_sid, turns, questions) in enumerate(data):
        if not questions:
            continue
        r = run_policy(factory(budget=CONTEXT_BUDGET, store_budget=store_budget,
                               **kw), turns, questions, CONTEXT_BUDGET)
        for d in r["detail"]:
            strict.append(1.0 if d["strict"] else 0.0)
            answer.append(np.nan if d["answer"] is None
                          else (1.0 if d["answer"] else 0.0))
            conv.append(ci)
    return (np.array(strict), np.array(answer), np.array(conv))


def paired_bootstrap(a, b, conv, cluster, n=B):
    """CI for mean(a) - mean(b), resampling questions or conversations."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b, conv = a[ok], b[ok], conv[ok]
    d = a - b
    obs = d.mean()
    if cluster:
        groups = [d[conv == c] for c in np.unique(conv)]
        k = len(groups)
        means = np.empty(n)
        for i in range(n):
            pick = RNG.integers(0, k, k)
            means[i] = np.concatenate([groups[j] for j in pick]).mean()
    else:
        idx = RNG.integers(0, len(d), (n, len(d)))
        means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return obs * 100, lo * 100, hi * 100


def mcnemar(a, b):
    """Exact two-sided McNemar on the discordant pairs."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return n01, n10, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=4096)
    args = ap.parse_args()
    S = args.store_budget

    print("\nMemGate — significance of the headline comparisons")
    print("=" * 86)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)
    print(f"  context {CONTEXT_BUDGET}   store {S}   bootstrap B={B}\n")

    pol = {}
    for name, F in [("P0", SlidingWindowPolicy), ("RAG", RAGPolicy),
                    ("P1-S select", MemGateSelectPolicy),
                    ("P1-A adaptive", MemGateAdaptivePolicy)]:
        pol[name] = collect(F, data, S)
        m = np.nanmean(pol[name][1]) * 100
        print(f"  {name:<14} strict {pol[name][0].mean()*100:5.1f}%   answer {m:5.1f}%")

    comparisons = [
        ("P1-S select", "RAG", "the decision policy: same storage, same fidelity,"
                               " same retrieval — only WHICH turns are forgotten"),
        ("P1-S select", "P1-A adaptive", "drop vs compress on eviction"),
        ("P1-S select", "P0", "scored retention vs plain recency"),
    ]

    rows = []
    for x, y, why in comparisons:
        print(f"\n{x}  vs  {y}")
        print(f"  ({why})")
        print("  " + "-" * 78)
        for metric, i in (("strict", 0), ("answer", 1)):
            a, b = pol[x][i], pol[y][i]
            conv = pol[x][2]
            o, lo, hi = paired_bootstrap(a, b, conv, cluster=False)
            o2, lo2, hi2 = paired_bootstrap(a, b, conv, cluster=True)
            n01, n10, p = mcnemar(a, b)
            sig_q = "yes" if lo > 0 or hi < 0 else "no"
            sig_c = "yes" if lo2 > 0 or hi2 < 0 else "no"
            print(f"    {metric:<7} {o:+6.2f} pts   "
                  f"by question [{lo:+6.2f}, {hi:+6.2f}] {sig_q:<4}"
                  f"   by conversation [{lo2:+6.2f}, {hi2:+6.2f}] {sig_c}")
            print(f"    {'':<7}          McNemar exact p = {p:.2e}   "
                  f"(discordant {n01} / {n10})")
            rows.append({"comparison": f"{x} vs {y}", "metric": metric,
                         "delta": o, "q_lo": lo, "q_hi": hi,
                         "conv_lo": lo2, "conv_hi": hi2, "mcnemar_p": p,
                         "store_budget": S})

    print("\n" + "=" * 86)
    print("How to report this")
    print("-" * 86)
    print("  Question-level intervals are narrow and the McNemar p-values are")
    print("  tiny, but both treat 1527 questions from 10 conversations as 1527")
    print("  independent trials, which they are not. The conversation-level")
    print("  interval is the defensible one, and with 10 clusters it is wide.")
    print()
    print("  State the effect WITH the clustered interval, and say plainly that")
    print("  10 conversations is a small sample. A result that survives")
    print("  clustering is strong; one that does not is a hypothesis, and should")
    print("  be written up as such.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "significance.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

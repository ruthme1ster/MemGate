#!/usr/bin/env python3
"""Abstractive compression — does rewriting beat selecting words?

THE CLAIM UNDER TEST
--------------------
§23 found that compressing an evicted turn loses 6.71 answer-recall points
against dropping it outright, and every deliverable since has carried the same
caveat: that result is stated for EXTRACTIVE compression only. `compress.py`
picks the most informative words of a turn and keeps them in order; it cannot
rephrase, so whatever it drops is gone. A rewriting summariser is not
restricted to a subsequence and may carry the same content in fewer tokens.

If compression ever pays for itself, this is where it should. If it still
loses, the negative result stops being about one implementation of compression
and starts being about compression.

THE DESIGN — ONE VARIABLE
-------------------------
A single-component swap, the same discipline `run_judge_eval.py` used for the
scorer. Policy, storage budget, retrieval, thresholds and eviction order are
pinned; only the function that turns a stored item into a shorter stored item
changes. `ThreeTierStore._gist` is the one site both compressors pass through,
so demotion under pressure and Tier-2 consolidation move together.

    P1-A extractive   compress on eviction, compress.py       (the §23 arm)
    P1-A abstractive  compress on eviction, a local LLM       (the swap)
    P1-S select       drop on eviction                        (the §23 winner)
    RAG store-all     drop oldest-first                       (the no-policy control)

Both metrics are reported, because this is exactly the comparison on which they
disagree (§19): a gist keeps the evidence ids of a turn whose text it discarded,
so evidence recall credits a pointer that answer recall does not. Where they
disagree, believe the answer metric.

COST IS PART OF THE RESULT
--------------------------
The extractive rule runs in microseconds. A rewrite is ~20 generated tokens per
compressed item on a local 1.5B model. The run reports how many generations it
actually made, so "abstractive wins" and "abstractive wins enough to justify
this" stay separable claims.

Usage:
    python3 run_abstractive.py                    # S=4096, the headline point
    python3 run_abstractive.py --store-budget 2048
    python3 run_abstractive.py --limit 3          # first 3 conversations, quick
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from memgate.abstractive import AbstractiveCompressor
from memgate.data import load_locomo
from memgate.harness import run_policy
from memgate.policies import (RAGPolicy, MemGateAdaptivePolicy,
                              MemGateSelectPolicy)
from memgate.utils import backend_info, warm_cache, reset_cache

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONTEXT_BUDGET = 2048
STORE_BUDGET = 4096
B = 10000
RNG = np.random.default_rng(20260921)


def clustered_ci(diff, conv, b=B):
    """Paired bootstrap resampled by CONVERSATION, not by question.

    1527 questions drawn from 10 conversations are not 1527 independent trials
    (§24.4). Every interval this project reports is clustered, and the one claim
    that did not survive clustering is reported as not significant -- so a new
    comparison arriving with an unclustered interval would not be comparable to
    any of them.
    """
    diff, conv = np.asarray(diff, dtype=float), np.asarray(conv)
    if not len(diff):
        return 0.0, 0.0, 0.0
    groups = [diff[conv == c] for c in np.unique(conv)]
    groups = [g for g in groups if len(g)]
    boot = np.array([
        np.concatenate([groups[j] for j in RNG.integers(0, len(groups),
                                                        len(groups))]).mean()
        for _ in range(b)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.mean() * 100, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=STORE_BUDGET)
    ap.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N conversations (quick check)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    S, CTX = args.store_budget, args.context_budget

    print("\nMemGate — abstractive vs extractive compression")
    print("=" * 92)
    data = load_locomo()
    if args.limit:
        data = data[:args.limit]
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")

    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    print(f"  warmed embedding cache: {warm_cache(texts)} vectors")

    comp = AbstractiveCompressor(args.model)
    print(f"  summariser: {os.path.basename(comp.model_name)}   "
          f"cache {len(comp._cache)} rewrites")
    print(f"  context {CTX}  store {S}  conversations {len(data)}\n")

    # Only the summariser differs between the first two arms. Everything else --
    # policy class, budgets, retrieval, thresholds -- is constructed identically.
    arms = [
        ("RAG store-all", lambda: RAGPolicy(budget=CTX, store_budget=S)),
        ("P1-A extractive", lambda: MemGateAdaptivePolicy(budget=CTX,
                                                          store_budget=S)),
        ("P1-A abstractive", lambda: MemGateAdaptivePolicy(budget=CTX,
                                                           store_budget=S,
                                                           summariser=comp)),
        ("P1-S select", lambda: MemGateSelectPolicy(budget=CTX, store_budget=S)),
    ]

    per_arm = {}
    for name, factory in arms:
        t0 = time.time()
        strict, answer, conv_ids = [], [], []
        hits = n = ans_hits = ans_n = 0
        stored = []
        for ci, (_sid, turns, questions) in enumerate(data):
            if not questions:
                continue
            r = run_policy(factory(), turns, questions, CTX)
            hits += r["hits"]; n += r["n"]
            ans_hits += r["answer_hits"]; ans_n += r["answer_n"]
            stored.append(r["stored_tokens"])
            for d in r["detail"]:
                strict.append(1.0 if d["strict"] else 0.0)
                conv_ids.append(ci)
                answer.append(np.nan if d["answer"] is None
                              else (1.0 if d["answer"] else 0.0))
        per_arm[name] = {
            "strict": np.array(strict), "answer": np.array(answer),
            "conv": np.array(conv_ids),
            "strict_recall": hits / n * 100 if n else 0.0,
            "answer_recall": ans_hits / ans_n * 100 if ans_n else 0.0,
            "stored_tokens": float(np.mean(stored)) if stored else 0.0,
            "n": n, "answer_n": ans_n, "secs": time.time() - t0,
        }
        if name == "P1-A abstractive":
            comp.save_cache()
        r_ = per_arm[name]
        print(f"  {name:<18} strict {r_['strict_recall']:5.1f}%   "
              f"answer {r_['answer_recall']:5.1f}%   "
              f"stored {r_['stored_tokens']:6.0f}   {r_['secs']:5.1f}s")

    cs = comp.stats()
    print(f"\n  summariser cost: {cs['generated']} generations, "
          f"{cs['cache_hits']} cache hits, {cs['fallbacks']} fallbacks")

    # ---- the comparison this script exists for -----------------------------
    print("\nSingle-component swap: abstractive minus extractive")
    print("-" * 92)
    rows = []
    base = per_arm["P1-A extractive"]
    for name in ("P1-A abstractive", "P1-S select", "RAG store-all"):
        cur = per_arm[name]
        for metric in ("strict", "answer"):
            a, b_ = cur[metric], base[metric]
            m = min(len(a), len(b_))
            ok = ~(np.isnan(a[:m]) | np.isnan(b_[:m]))
            d, lo, hi = clustered_ci(a[:m][ok] - b_[:m][ok], cur["conv"][:m][ok])
            sig = "significant" if (lo > 0 or hi < 0) else "n.s."
            rows.append({"arm": name, "vs": "P1-A extractive", "metric": metric,
                         "delta": d, "ci_lo": lo, "ci_hi": hi,
                         "significant": sig == "significant",
                         "strict_recall": cur["strict_recall"],
                         "answer_recall": cur["answer_recall"],
                         "stored_tokens": cur["stored_tokens"],
                         "store_budget": S, "context_budget": CTX,
                         "n": cur["n"], "answer_n": cur["answer_n"],
                         "generations": cs["generated"],
                         "fallbacks": cs["fallbacks"],
                         "model": cs["model"]})
            print(f"  {name:<18} {metric:<7} {d:+6.2f}  "
                  f"95% CI [{lo:+6.2f}, {hi:+6.2f}]  {sig}")

    print("\nReading it")
    print("-" * 92)
    ab = per_arm["P1-A abstractive"]["answer_recall"]
    ex = per_arm["P1-A extractive"]["answer_recall"]
    sel = per_arm["P1-S select"]["answer_recall"]
    if ab > ex:
        print(f"  Rewriting beats selecting words by {ab-ex:+.1f} answer points, so the")
        print("  §23 negative result was partly about the compressor rather than")
        print("  about compression.")
    else:
        print(f"  Rewriting does NOT beat selecting words ({ab-ex:+.1f} answer points).")
        print("  The §23 finding survives a stronger compressor, which widens it from")
        print("  a statement about extractive methods to one about compression.")
    if sel > ab:
        print(f"  Dropping still beats both ({sel:.1f}% vs {ab:.1f}%): at this scale the")
        print("  optimum remains at the vertex -- a subset at full fidelity.")
    else:
        print(f"  Compression overtakes selection ({ab:.1f}% vs {sel:.1f}%) -- the")
        print("  crossover §24.1 could not find on the length axis.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "abstractive.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""LongMemEval — the retention claim on conversations that are genuinely long.

WHAT THIS REPLACES
------------------
§24.1 tested the rate-distortion crossover by concatenating up to ten LoCoMo
conversations to 186K tokens, and the caveat was stated in the paper rather than
buried: a concatenated stream is not a natural long conversation. Each question
concerns one constituent and the others act as distractors, so it stresses
retention but is not a long-dialogue benchmark.

LongMemEval supplies the real thing. Each item is one question against its own
haystack of 38-62 sessions and ~400-600 turns of single-thread history. If the
§23 finding -- that choosing WHICH turns to forget beats forgetting oldest-first
at identical storage -- holds here, it holds on length nobody manufactured.

WHY THE INTERVALS ARE BETTER HERE
---------------------------------
LoCoMo gives ten clusters, which is why every interval in this project is wide
and why one claim did not survive clustering at all (§24.4). LongMemEval is one
haystack per question, so each item is an independent cluster: a subsample of
100 items is 100 clusters against LoCoMo's 10.

SCALE, STATED HONESTLY
----------------------
The full release is 500 items averaging ~538 turns, i.e. ~269,000 turns to embed
per arm. That is hours, so the default is a deterministic prefix of the corpus
(--limit) and the item count is written into the CSV and printed in the header.
A subsample is a subsample and is reported as one; it is not described as the
benchmark.

Usage:
    python3 run_longmemeval.py                 # 100 items, S=4096
    python3 run_longmemeval.py --limit 500     # the whole release (hours)
    python3 run_longmemeval.py --store-budget 8192
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from memgate.data import load_longmemeval
from memgate.harness import run_policy
from memgate.policies import (SlidingWindowPolicy, RAGPolicy,
                              MemGateAdaptivePolicy, MemGateSelectPolicy)
from memgate.utils import backend_info, warm_cache, reset_cache

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONTEXT_BUDGET = 2048
STORE_BUDGET = 4096
B = 10000
RNG = np.random.default_rng(20260921)


def clustered_ci(diff, cluster, b=B):
    """Paired bootstrap resampled by ITEM -- here, one item is one haystack."""
    diff, cluster = np.asarray(diff, dtype=float), np.asarray(cluster)
    if not len(diff):
        return 0.0, 0.0, 0.0
    groups = [diff[cluster == c] for c in np.unique(cluster)]
    groups = [g for g in groups if len(g)]
    boot = np.array([
        np.concatenate([groups[j] for j in RNG.integers(0, len(groups),
                                                        len(groups))]).mean()
        for _ in range(b)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.mean() * 100, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100,
                    help="number of items (deterministic prefix); 500 = all")
    ap.add_argument("--store-budget", type=int, default=STORE_BUDGET)
    ap.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    args = ap.parse_args()
    S, CTX = args.store_budget, args.context_budget

    print("\nMemGate — LongMemEval (genuinely long single conversations)")
    print("=" * 92)
    data = load_longmemeval(limit=args.limit)
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")

    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    t0 = time.time()
    print(f"  warming embedding cache over {len(texts)} texts...", flush=True)
    print(f"  warmed {warm_cache(texts)} vectors in {time.time()-t0:.0f}s")
    n_turns = sum(len(t) for _, t, _ in data)
    print(f"  context {CTX}  store {S}  items {len(data)}  turns {n_turns}\n")

    arms = [
        ("P0 sliding-window", lambda: SlidingWindowPolicy(budget=CTX,
                                                          store_budget=S)),
        ("RAG store-all", lambda: RAGPolicy(budget=CTX, store_budget=S)),
        ("P1-A adaptive", lambda: MemGateAdaptivePolicy(budget=CTX,
                                                        store_budget=S)),
        ("P1-S select", lambda: MemGateSelectPolicy(budget=CTX, store_budget=S)),
    ]

    per_arm = {}
    for name, factory in arms:
        t0 = time.time()
        strict, answer, cluster = [], [], []
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
                answer.append(np.nan if d["answer"] is None
                              else (1.0 if d["answer"] else 0.0))
                cluster.append(ci)
        per_arm[name] = {
            "strict": np.array(strict), "answer": np.array(answer),
            "cluster": np.array(cluster),
            "strict_recall": hits / n * 100 if n else 0.0,
            "answer_recall": ans_hits / ans_n * 100 if ans_n else 0.0,
            "stored_tokens": float(np.mean(stored)) if stored else 0.0,
            "n": n, "answer_n": ans_n,
        }
        r_ = per_arm[name]
        print(f"  {name:<20} strict {r_['strict_recall']:5.1f}%   "
              f"answer {r_['answer_recall']:5.1f}%   "
              f"stored {r_['stored_tokens']:6.0f}   {time.time()-t0:5.1f}s",
              flush=True)

    # ---- the same three one-variable contrasts as on LoCoMo ---------------
    print("\nOne variable per contrast, clustered by item")
    print("-" * 92)
    contrasts = [("P1-S select", "RAG store-all", "which turns are forgotten"),
                 ("P1-S select", "P1-A adaptive", "fidelity: drop vs compress"),
                 ("RAG store-all", "P0 sliding-window", "read path")]
    rows = []
    for a_name, b_name, what in contrasts:
        A, Bm = per_arm[a_name], per_arm[b_name]
        for metric in ("strict", "answer"):
            a, b_ = A[metric], Bm[metric]
            m = min(len(a), len(b_))
            ok = ~(np.isnan(a[:m]) | np.isnan(b_[:m]))
            d, lo, hi = clustered_ci(a[:m][ok] - b_[:m][ok], A["cluster"][:m][ok])
            sig = (lo > 0 or hi < 0)
            rows.append({"arm": a_name, "vs": b_name, "isolates": what,
                         "metric": metric, "delta": d, "ci_lo": lo, "ci_hi": hi,
                         "significant": sig,
                         "strict_recall": A["strict_recall"],
                         "answer_recall": A["answer_recall"],
                         "stored_tokens": A["stored_tokens"],
                         "store_budget": S, "context_budget": CTX,
                         "items": len(data), "n": A["n"],
                         "answer_n": A["answer_n"],
                         "embedder": info["embedder"]})
            print(f"  {a_name:<16} vs {b_name:<18} {metric:<7} {d:+6.2f}  "
                  f"95% CI [{lo:+6.2f}, {hi:+6.2f}]  "
                  f"{'significant' if sig else 'n.s.'}")

    print("\nReading it")
    print("-" * 92)
    sel = per_arm["P1-S select"]["answer_recall"]
    rag = per_arm["RAG store-all"]["answer_recall"]
    ada = per_arm["P1-A adaptive"]["answer_recall"]
    print(f"  selection vs FIFO      {sel-rag:+.1f} answer points")
    print(f"  dropping vs compressing{sel-ada:+.1f} answer points")
    print("  On LoCoMo these were +8.67 and +6.71. Agreement on a benchmark with")
    print("  real long histories and ~50x the clusters is what would make the")
    print("  §23 finding a property of the decision rather than of LoCoMo.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "longmemeval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

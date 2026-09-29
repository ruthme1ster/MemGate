#!/usr/bin/env python3
"""§30.4 — does the Tier-1 recency reservation explain P1-S's LongMemEval loss?

`MemGatePolicy.build_context` reserves `split[0]` of the context for the most
recent turns BEFORE retrieval is consulted; `RAGPolicy` reserves nothing and
spends the whole budget on retrieved items. So "P1-S vs RAG isolates which turns
are forgotten" (§23, and this runner's sibling) is not strictly true: they also
differ in guaranteed recency share.

That costs little on LoCoMo -- 25% of 2048 buys ~16 of its 32-token turns, and
questions often concern recent sessions. It should cost a great deal on
LongMemEval, where the same 512 tokens buys 2-3 of its 210-token turns and the
answer sits in one of 38-62 sessions chosen without regard to recency.

This runs the identical comparison with the reservation removed. If P1-S
recovers, the §30 headline is a confound and must not be published as it
stands. If it does not, the scorer transfer failure of §30.3 is the whole story.

Usage:
    python3 run_lme_split.py                    # S=23000 (LoCoMo-matched), 94 items
    python3 run_lme_split.py --store-budget 4096
"""
import argparse, csv, os, sys, time
import numpy as np

from memgate.data import load_longmemeval
from memgate.harness import run_policy
from memgate.policies import RAGPolicy, MemGateSelectPolicy
from memgate.utils import backend_info, warm_cache, reset_cache

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
B = 10000
RNG = np.random.default_rng(20260929)


def clustered_ci(diff, cluster, b=B):
    """Paired bootstrap resampled by ITEM -- one haystack is one cluster."""
    diff, cluster = np.asarray(diff, dtype=float), np.asarray(cluster)
    if not len(diff):
        return 0.0, 0.0, 0.0
    groups = [diff[cluster == c] for c in np.unique(cluster)]
    groups = [g for g in groups if len(g)]
    boot = np.array([
        np.concatenate([groups[j] for j in RNG.integers(0, len(groups),
                                                        len(groups))]).mean()
        for _ in range(b)]) * 100
    return diff.mean() * 100, *np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=23000)
    ap.add_argument("--context-budget", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    S, CTX = a.store_budget, a.context_budget

    print("\nMemGate — §30.4: the Tier-1 recency reservation, removed")
    print("=" * 92)
    data = load_longmemeval(limit=a.limit)
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    t0 = time.time()
    print(f"  warming {len(texts)} texts...", flush=True)
    print(f"  warmed {warm_cache(texts)} vectors in {time.time()-t0:.0f}s")
    print(f"  context {CTX}  store {S}  items {len(data)}\n")

    # Only split[0] differs between the first two arms. Same class, same scorer,
    # same budgets, same retrieval depth.
    arms = [
        ("P1-S split .25", lambda: MemGateSelectPolicy(budget=CTX, store_budget=S,
                                                       split=(0.25, 0.25, 0.50))),
        ("P1-S split 0",   lambda: MemGateSelectPolicy(budget=CTX, store_budget=S,
                                                       split=(0.0, 0.30, 0.70))),
        ("RAG store-all",  lambda: RAGPolicy(budget=CTX, store_budget=S)),
    ]
    per = {}
    for name, fac in arms:
        t0 = time.time()
        strict, answer, cluster = [], [], []
        hits = n = ah = an = 0
        for ci, (_sid, turns, qs) in enumerate(data):
            if not qs:
                continue
            r = run_policy(fac(), turns, qs, CTX)
            hits += r["hits"]; n += r["n"]; ah += r["answer_hits"]; an += r["answer_n"]
            for d in r["detail"]:
                strict.append(1.0 if d["strict"] else 0.0)
                answer.append(np.nan if d["answer"] is None else (1.0 if d["answer"] else 0.0))
                cluster.append(ci)
        per[name] = {"strict": np.array(strict), "answer": np.array(answer),
                     "cluster": np.array(cluster),
                     "sr": hits / n * 100 if n else 0.0,
                     "ar": ah / an * 100 if an else 0.0, "n": n, "an": an}
        print(f"  {name:<16} strict {per[name]['sr']:5.1f}%   "
              f"answer {per[name]['ar']:5.1f}%   {time.time()-t0:6.1f}s", flush=True)

    print("\nContrasts, clustered by item")
    print("-" * 92)
    rows = []
    for aN, bN, what in (("P1-S split 0", "P1-S split .25", "the recency reservation alone"),
                         ("P1-S split 0", "RAG store-all", "which turns are forgotten, unconfounded"),
                         ("P1-S split .25", "RAG store-all", "which turns are forgotten, as in §30")):
        A, Bm = per[aN], per[bN]
        for metric in ("strict", "answer"):
            x, y = A[metric], Bm[metric]
            m = min(len(x), len(y))
            ok = ~(np.isnan(x[:m]) | np.isnan(y[:m]))
            d, lo, hi = clustered_ci(x[:m][ok] - y[:m][ok], A["cluster"][:m][ok])
            sig = lo > 0 or hi < 0
            rows.append({"arm": aN, "vs": bN, "isolates": what, "metric": metric,
                         "delta": d, "ci_lo": lo, "ci_hi": hi, "significant": sig,
                         "strict_recall": A["sr"], "answer_recall": A["ar"],
                         "store_budget": S, "context_budget": CTX,
                         "items": len(data), "n": A["n"], "answer_n": A["an"]})
            print(f"  {aN:<15} vs {bN:<15} {metric:<7} {d:+6.2f}  "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {'significant' if sig else 'n.s.'}")

    out = os.path.join(RESULTS, f"longmemeval_split_S{S}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Online learning of the eviction policy — P3's headroom, without P3's oracle.

THE GAP THIS CLOSES
-------------------
P3-learned reaches 48.3% answer recall against P1's 36.8% at S=4096, and is
explicitly not a component: it is fitted on "was this turn ever cited as
evidence", which needs the future questions. A live agent does not have them, so
that +11.5 is a headroom bound and nothing more.

P4 asks whether any of that headroom is reachable from a signal an agent
actually observes while running.

THE SIGNAL, AND WHY IT IS NOT A LEAK
------------------------------------
The obvious online label -- "did the agent answer correctly" -- is unavailable
offline without reading the gold answer, which would put the evaluation label
back into the write path. This uses retrieval feedback instead: which stored
items the read path pulled into context when the agent was queried. No gold
evidence, no answer text, no fact_id.

WHERE THE QUERIES COME FROM, AND WHY NOT THE EVALUATION QUESTIONS
-----------------------------------------------------------------
The benchmark's questions are all asked at the end, so driving feedback with
them would mean the scorer adapts to the very queries it is about to be scored
on -- test-time adaptation on the evaluation set. It would probably look good
and would mean nothing.

Instead the pseudo-queries are USER TURNS FROM THE CONVERSATION ITSELF, which is
what a deployed agent retrieves against: every user message triggers a retrieval,
and the outcome of that retrieval is the feedback. The evaluation questions are
touched only at scoring time, after the conversation has been replayed.

THE COMPARISON
--------------
Single-component swap at the §23 winner (P1-S select, context 2048, store 4096):
policy, budgets, retrieval and compression pinned, only the scorer varying.

    P0 recency    no content judgement at all       (the floor)
    P1 heuristic  hand-written weights              (the incumbent)
    P4 online     heuristic prior + retrieval SGD   (this)
    P3 learned    fitted on future questions        (the bound, from §24)

Usage:
    python3 run_online.py
    python3 run_online.py --every 20 --store-budget 2048
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from memgate.data import load_locomo
from memgate.online import OnlineScorer
from memgate.policies import MemGateSelectPolicy
from memgate.scoring import HeuristicScorer, RecencyScorer
from memgate.utils import (backend_info, warm_cache, reset_cache,
                           answer_tokens)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONTEXT_BUDGET = 2048
STORE_BUDGET = 4096
B = 10000
RNG = np.random.default_rng(20260921)


def clustered_ci(diff, cluster, b=B):
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


def replay(policy, turns, budget, scorer=None, every=0):
    """Ingest the conversation, with retrieval feedback every `every` turns.

    The feedback loop is deliberately the one a deployed agent would run:
      1. a user turn arrives and is stored
      2. periodically, that turn is used as a query against the store
      3. whatever retrieval returned is positive evidence; a sample of what the
         store held and did not return is negative

    Nothing here reads a question, an answer or an evidence id.
    """
    for i, t in enumerate(turns):
        policy.observe(t)
        if not (scorer is not None and every and (i + 1) % every == 0):
            continue
        if not t.text.strip():
            continue
        ctx = policy.build_context(t.text, budget)
        retrieved = [it.text for it in ctx.items]
        held = [it.text for it in policy.store.long] + \
               [it.text for it in policy.store.working]
        scorer.feedback(retrieved, held)


def evaluate(policy, questions, budget):
    """Score an ALREADY-REPLAYED policy. Mirrors harness.run_policy's metrics.

    The harness ingests and then scores in one call, which cannot express "train
    during ingest, then score", so the metric loop is repeated here rather than
    the policy being replayed twice.
    """
    strict, answer = [], []
    for q in questions:
        ev = set(q.evidence_ids)
        ctx = policy.build_context(q.text, budget)
        got = ctx.fact_ids
        strict.append(1.0 if (ev and ev <= got) else 0.0)
        if q.answer_recoverable:
            ok = set(answer_tokens(q.answer)) <= set(answer_tokens(ctx.text))
            answer.append(1.0 if ok else 0.0)
        else:
            answer.append(np.nan)
    return np.array(strict), np.array(answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=STORE_BUDGET)
    ap.add_argument("--context-budget", type=int, default=CONTEXT_BUDGET)
    ap.add_argument("--every", type=int, default=25,
                    help="run a feedback query every N turns")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    S, CTX = args.store_budget, args.context_budget

    print("\nMemGate — P4, online learning from retrieval feedback")
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
    print(f"  context {CTX}  store {S}  feedback every {args.every} turns")
    print("  policy pinned at P1-S select — only the SCORER varies\n")

    arms = {
        "P0 recency": lambda: RecencyScorer(),
        "P1 heuristic": lambda: HeuristicScorer(),
        "P4 online": lambda: OnlineScorer(),
    }

    per_arm, weights_report = {}, None
    for name, mk in arms.items():
        t0 = time.time()
        strict, answer, cluster = [], [], []
        last_scorer = None
        for ci, (_sid, turns, questions) in enumerate(data):
            sc = mk()
            pol = MemGateSelectPolicy(budget=CTX, store_budget=S, scorer=sc)
            online = sc if isinstance(sc, OnlineScorer) else None
            replay(pol, turns, CTX, scorer=online, every=args.every)
            s, a = evaluate(pol, questions, CTX)
            strict.extend(s); answer.extend(a); cluster.extend([ci] * len(s))
            last_scorer = sc
        s_arr, a_arr = np.array(strict), np.array(answer)
        per_arm[name] = {"strict": s_arr, "answer": a_arr,
                         "cluster": np.array(cluster),
                         "strict_recall": float(np.nanmean(s_arr)) * 100,
                         "answer_recall": float(np.nanmean(a_arr)) * 100,
                         "n": len(s_arr),
                         "answer_n": int(np.sum(~np.isnan(a_arr)))}
        r = per_arm[name]
        extra = ""
        if isinstance(last_scorer, OnlineScorer):
            st = last_scorer.stats()
            extra = (f"   updates {st['updates']} "
                     f"(alpha {st['alpha']:.2f})")
            weights_report = last_scorer.weights()
        print(f"  {name:<14} strict {r['strict_recall']:5.1f}%   "
              f"answer {r['answer_recall']:5.1f}%   {time.time()-t0:5.1f}s{extra}")

    print("\nAgainst the hand-written heuristic, clustered by conversation")
    print("-" * 92)
    rows = []
    base = per_arm["P1 heuristic"]
    for name in ("P4 online", "P0 recency"):
        cur = per_arm[name]
        for metric in ("strict", "answer"):
            a, b_ = cur[metric], base[metric]
            m = min(len(a), len(b_))
            ok = ~(np.isnan(a[:m]) | np.isnan(b_[:m]))
            d, lo, hi = clustered_ci(a[:m][ok] - b_[:m][ok], cur["cluster"][:m][ok])
            sig = (lo > 0 or hi < 0)
            rows.append({"scorer": name, "vs": "P1 heuristic", "metric": metric,
                         "delta": d, "ci_lo": lo, "ci_hi": hi,
                         "significant": sig,
                         "strict_recall": cur["strict_recall"],
                         "answer_recall": cur["answer_recall"],
                         "store_budget": S, "context_budget": CTX,
                         "feedback_every": args.every,
                         "n": cur["n"], "answer_n": cur["answer_n"]})
            print(f"  {name:<12} {metric:<7} {d:+6.2f}  "
                  f"95% CI [{lo:+6.2f}, {hi:+6.2f}]  "
                  f"{'significant' if sig else 'n.s.'}")

    if weights_report:
        print("\n  what the online scorer learned (largest magnitude first)")
        for fname, w in weights_report[:8]:
            print(f"    {fname:<18}{w:+7.3f}")

    print("\nReading it")
    print("-" * 92)
    p4 = per_arm["P4 online"]["answer_recall"]
    p1 = per_arm["P1 heuristic"]["answer_recall"]
    print(f"  P4 {p4:.1f}%  vs  P1 {p1:.1f}%   ({p4-p1:+.1f})")
    print("  P3-learned reached 48.3% at this budget using the future questions.")
    print("  P4 uses only retrieval feedback, so the distance between P4 and P3 is")
    print("  the price of not having an oracle -- and the distance between P4 and")
    print("  P1 is what a label-free runtime signal actually bought.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "online.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

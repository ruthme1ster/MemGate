#!/usr/bin/env python3
"""P2 evaluation — does an LLM judge beat hand-written rules at deciding what to keep?

This is the build plan's Step 3, finally runnable. It is a SINGLE-COMPONENT
swap: policy, storage budget, retrieval and compression are held fixed at the
§23 winner (P1-S select), and only the scorer changes. Any difference is the
scorer.

Four scorers on the same footing:

  P0-recency    no content judgement at all (every turn scores equally, so
                eviction falls back to recency) -- the floor
  P1-heuristic  rules + shallow NER, weights chosen by hand
  P2-llm-judge  Qwen2.5-0.5B-Instruct rates each turn 0-9, read as a continuous
                expectation over digit-token logits (memgate/judge.py)
  P3-learned    logistic model fitted on OTHER conversations' evidence labels,
                leave-one-conversation-out

The comparison that matters is P1 vs P2: an LLM judge costs ~15 minutes of GPU
per corpus against a regex's microseconds, so it has to earn that. P3 is the
informed upper bound -- it is not deployable (a live agent has no future
questions), so it brackets how much a better scorer could possibly be worth.

Requires the judge cache: run `python3 precompute_judge.py` first.

Usage:
    python3 run_judge_eval.py
    python3 run_judge_eval.py --store-budget 2048
"""
import argparse
import csv
import os
import sys

import numpy as np

from memgate.data import load_locomo
from memgate.policies import MemGateSelectPolicy
from memgate.harness import run_policy
from memgate.scoring import HeuristicScorer, RecencyScorer, LearnedScorer, features
from memgate.judge import LocalLLMJudge, LLMJudgeScorer, rank_normalise, CACHE_PATH
from memgate.utils import backend_info, warm_cache, reset_cache, embed

CONTEXT_BUDGET = 2048
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(data, scorer_for, store_budget):
    """Micro-averaged over questions. `scorer_for(i)` may vary per conversation
    (P3 needs a different fold-model for each).

    Also returns the PER-QUESTION outcomes tagged by conversation. Aggregates
    alone cannot support a paired test -- two scorers at 36.8% and 30.9% may
    agree on nearly every question or disagree on all of them, and the interval
    differs enormously between those cases (§24.4)."""
    tot_q = tot_hits = ans_n = ans_hits = 0
    answer, conv = [], []
    for i, (_sid, turns, questions) in enumerate(data):
        if not questions:
            continue
        r = run_policy(MemGateSelectPolicy(budget=CONTEXT_BUDGET,
                                           store_budget=store_budget,
                                           scorer=scorer_for(i)),
                       turns, questions, CONTEXT_BUDGET)
        tot_q += r["n"]; tot_hits += r["hits"]
        ans_n += r["answer_n"]; ans_hits += r["answer_hits"]
        for d in r["detail"]:
            answer.append(np.nan if d["answer"] is None
                          else (1.0 if d["answer"] else 0.0))
            conv.append(i)
    return (tot_hits / tot_q * 100 if tot_q else 0.0,
            ans_hits / ans_n * 100 if ans_n else 0.0,
            np.asarray(answer), np.asarray(conv))


RNG = np.random.default_rng(20260827)


def clustered_ci(a, b, conv, n=10000):
    """95% CI for mean(a)-mean(b), resampling CONVERSATIONS.

    Clustered rather than per-question for the reason §24.4 established: a
    conversation that happens to suit a scorer makes all ~150 of its questions
    succeed together, so resampling questions treats 150 dependent draws as
    independent and understates the interval. It was this correction that
    withdrew the evidence-recall claim against RAG, so it applies here too.
    """
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    c = conv[ok]
    groups = [d[c == k] for k in np.unique(c)]
    groups = [g for g in groups if len(g)]
    means = np.empty(n)
    for i in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100


def build_p3(data, use_emb=True):
    """Leave-one-conversation-out models, one per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = []
    for _sid, turns, questions in data:
        cited = set()
        for q in questions:
            cited.update(q.evidence_ids)
        X, y = [], []
        for t in turns:
            txt = f"[{t.speaker}] {t.text}"
            f = features(txt, t.speaker)
            if use_emb:
                f = list(f) + list(embed(txt))
            X.append(f); y.append(1 if t.fact_id in cited else 0)
        feats.append((np.asarray(X, dtype=np.float32),
                      np.asarray(y, dtype=np.int8)))
    models = []
    for i in range(len(data)):
        Xtr = np.concatenate([feats[j][0] for j in range(len(data)) if j != i])
        ytr = np.concatenate([feats[j][1] for j in range(len(data)) if j != i])
        models.append(make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-budget", type=int, default=4096)
    args = ap.parse_args()
    S = args.store_budget

    print("\nMemGate — P2 LLM judge vs the alternatives")
    print("=" * 78)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)

    judge = LocalLLMJudge()
    n = judge.load_cache()
    if n == 0:
        print(f"\n  ERROR: no judge cache at {CACHE_PATH}")
        print("  run `python3 precompute_judge.py` first (~15 min, one time).")
        return 1
    print(f"  judge cache: {n} scored turns")
    print(f"  policy P1-S select, context {CONTEXT_BUDGET}, store {S}")
    print("  ONLY the scorer varies\n")

    turn_texts = sorted({f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts})
    raw = judge.score_many(turn_texts)
    calib = rank_normalise(raw)
    print(f"  judge raw range {min(raw):.3f}–{max(raw):.3f} "
          f"(rank-normalised for routing; ranking is unchanged)\n")

    p3models = build_p3(data)

    rows, outcomes = [], {}
    print(f"{'Scorer':<26}{'Strict':>9}{'Answer':>9}{'vs P1 (answer)':>17}")
    print("-" * 62)
    base_ans = None
    for name, factory in [
        ("P0 recency (no scoring)", lambda i: RecencyScorer()),
        ("P1 heuristic (hand)", lambda i: HeuristicScorer()),
        ("P2 LLM judge (Qwen-0.5B)", lambda i: LLMJudgeScorer(judge, calib)),
        ("P3 learned (LOCO, bound)", lambda i: LearnedScorer(
            model=p3models[i], use_embedding=True, embedder=embed)),
    ]:
        st, an, av, cv = evaluate(data, factory, S)
        if name.startswith("P1"):
            base_ans = an
        delta = "" if base_ans is None or name.startswith("P1") else f"{an-base_ans:+.1f}"
        print(f"{name:<26}{st:>8.1f}%{an:>8.1f}%{delta:>17}")
        outcomes[name] = (av, cv)
        rows.append({"scorer": name, "strict": st, "answer": an,
                     "store_budget": S, "context_budget": CONTEXT_BUDGET})

    print("\nAre these gaps real? (95% CI, resampled by conversation)")
    print("-" * 62)
    p1n = next(r["scorer"] for r in rows if r["scorer"].startswith("P1"))
    for name in outcomes:
        if name == p1n:
            continue
        o, lo, hi = clustered_ci(outcomes[name][0], outcomes[p1n][0],
                                 outcomes[name][1])
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"  {name:<26}{o:+6.1f} pts  [{lo:+6.1f}, {hi:+6.1f}]  {verdict}")
        for r in rows:
            if r["scorer"] == name:
                r["vs_p1"], r["ci_lo"], r["ci_hi"] = o, lo, hi
    for r in rows:
        r.setdefault("vs_p1", 0.0); r.setdefault("ci_lo", 0.0); r.setdefault("ci_hi", 0.0)

    print("\nReading it")
    print("-" * 62)
    p1 = next(r for r in rows if r["scorer"].startswith("P1"))
    p2 = next(r for r in rows if r["scorer"].startswith("P2"))
    p3 = next(r for r in rows if r["scorer"].startswith("P3"))
    p0 = next(r for r in rows if r["scorer"].startswith("P0"))
    print(f"  scoring at all is worth {p1['answer']-p0['answer']:+.1f} answer points "
          f"(P1 over unscored recency)")
    if p2["answer"] > p1["answer"]:
        print(f"  the LLM judge earns its cost: {p2['answer']-p1['answer']:+.1f} over the heuristic")
    else:
        print(f"  the LLM judge does NOT earn its cost: {p2['answer']-p1['answer']:+.1f} vs the")
        print("  heuristic, for ~15 min of GPU against a regex's microseconds.")
        print("  A 0.5B model rating isolated turns is a weak salience signal;")
        print("  report this plainly rather than burying it.")
    print(f"  headroom (P3 bound, not deployable): {p3['answer']-p1['answer']:+.1f}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "judge_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""LoCoMo evaluation — the real benchmark (build-plan Step 1).

Replaces the synthetic sanity set with LoCoMo (Maharana et al., 2024):
10 conversations, 5882 turns, 1527 scoreable questions.

Each conversation is evaluated independently and results are micro-averaged
over questions, so a long conversation with many probes carries proportionate
weight. Conversations are never concatenated -- see load_locomo().

Usage:
    python3 run_locomo.py                      # default backends (auto)
    MEMGATE_EMBEDDER=hash python3 run_locomo.py    # hashing-embedder ablation
"""
import csv
import os
import sys
from collections import defaultdict

from memgate.data import load_locomo, CATEGORY_NAMES
from memgate.policies import (FullContextPolicy, OraclePolicy,
                              SlidingWindowPolicy, MemGatePolicy)
from memgate.harness import run_policy
from memgate.utils import backend_info, warm_cache, reset_cache

BUDGETS = [512, 1024, 2048, 4096]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(factory, data, budget):
    """Run one policy at one budget across every conversation, micro-averaged."""
    tot_q = tot_hits = ans_n = ans_hits = 0
    soft_sum = tok_sum = 0.0
    p95s = []
    cats = defaultdict(lambda: {"n": 0, "strict": 0, "soft": 0.0})
    for _sid, turns, questions in data:
        if not questions:
            continue
        # budget is passed to the constructor too: a policy may size its store
        # from it (see MemGatePolicy). Baselines ignore it via **kw.
        r = run_policy(factory(budget=budget), turns, questions, budget)
        n = r["n"]
        tot_q += n
        tot_hits += r["hits"]
        soft_sum += r["soft_recall"] * n
        tok_sum += r["avg_tokens"] * n
        ans_n += r["answer_n"]
        ans_hits += r["answer_hits"]
        p95s.append(r["p95_latency_ms"])
        for c, v in r["by_category"].items():
            cats[c]["n"] += v["n"]
            cats[c]["strict"] += v["strict"]
            cats[c]["soft"] += v["soft"] * v["n"]
    return {
        "n": tot_q,
        "recall": tot_hits / tot_q if tot_q else 0.0,
        "soft_recall": soft_sum / tot_q if tot_q else 0.0,
        "avg_tokens": tok_sum / tot_q if tot_q else 0.0,
        "answer_recall": ans_hits / ans_n if ans_n else 0.0,
        "answer_n": ans_n,
        "p95_latency_ms": max(p95s) if p95s else 0.0,
        "by_category": {c: {"n": v["n"], "recall": v["strict"] / v["n"],
                            "soft": v["soft"] / v["n"]}
                        for c, v in cats.items()},
    }


def main():
    print("\nMemGate — LoCoMo evaluation")
    print("=" * 74)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")

    # Batch-embed every turn up front. The store embeds one string at a time on
    # the write path; with a transformer that is the dominant cost, so warming
    # the cache in one batched pass turns thousands of forward passes into a
    # handful. Purely a speed optimisation -- the vectors are identical.
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, turns, _q in data for t in turns]
    texts += [q.text for _s, _t, qs in data for q in qs]
    n_new = warm_cache(texts)
    print(f"  warmed embedding cache: {n_new} vectors\n")

    rows = []
    print(f"{'Policy':<22}{'Budget':>8}{'Strict':>9}{'Soft':>8}{'Answer':>9}"
          f"{'AvgTok':>9}{'p95 ms':>9}")
    print("-" * 74)

    r = evaluate(FullContextPolicy, data, 10 ** 9)
    r.update(policy="Full-context", budget=0)
    rows.append(r)
    print(f"{'Full-context':<22}{'unlim':>8}{r['recall']*100:>8.1f}%"
          f"{r['soft_recall']*100:>7.1f}%{r['answer_recall']*100:>8.1f}%"
          f"{r['avg_tokens']:>9.0f}{r['p95_latency_ms']:>9.2f}")
    print(f"  (answer-presence scored over {r['answer_n']} questions whose "
          f"answer is recoverable from evidence text)\n")

    for budget in BUDGETS:
        for name, factory in [("Oracle", OraclePolicy),
                              ("P0 sliding-window", SlidingWindowPolicy),
                              ("P1 MemGate", MemGatePolicy)]:
            r = evaluate(factory, data, budget)
            r.update(policy=name, budget=budget)
            rows.append(r)
            print(f"{name:<22}{budget:>8}{r['recall']*100:>8.1f}%"
                  f"{r['soft_recall']*100:>7.1f}%{r['answer_recall']*100:>8.1f}%"
                  f"{r['avg_tokens']:>9.0f}{r['p95_latency_ms']:>9.2f}")
        print()

    # per-category, at the middle budget
    print("Per-category strict recall @ 2048 tokens")
    print("-" * 74)
    mid = [r for r in rows if r["budget"] == 2048]
    cats = sorted({c for r in mid for c in r["by_category"]})
    print(f"{'Policy':<22}" + "".join(
        f"{CATEGORY_NAMES.get(c, c)[:11]:>13}" for c in cats))
    for r in mid:
        cells = ""
        for c in cats:
            v = r["by_category"].get(c)
            cells += f"{v['recall']*100:>12.1f}%" if v else f"{'--':>13}"
        print(f"{r['policy']:<22}{cells}")
    print(f"{'(n questions)':<22}" + "".join(
        f"{mid[0]['by_category'][c]['n']:>13}" for c in cats))

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "locomo_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "budget", "strict_recall", "soft_recall",
                    "answer_recall", "answer_n", "avg_tokens",
                    "p95_latency_ms", "n_questions", "embedder", "tokenizer"])
        for r in rows:
            w.writerow([r["policy"], r["budget"], f"{r['recall']:.4f}",
                        f"{r['soft_recall']:.4f}", f"{r['answer_recall']:.4f}",
                        r["answer_n"], f"{r['avg_tokens']:.1f}",
                        f"{r['p95_latency_ms']:.3f}", r["n"],
                        info["embedder"], info["tokenizer"]])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

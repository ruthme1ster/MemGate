#!/usr/bin/env python3
"""Ablation study (build-plan Step 4) — which component actually earns the gain?

Every row turns off exactly ONE thing and reports the delta against the full
system on LoCoMo. A component that costs nothing when removed is not earning its
place; a component whose removal collapses the result is the one carrying the
contribution.

Two families:

  ARCHITECTURE  the changes this project made -- consolidation merging, the
                informative compressor, density packing, retrieval, Tier 2,
                supersession. "Off" restores the behaviour that was replaced,
                so each row is a like-for-like before/after.

  SCORING       the eight signals inside the P1 heuristic, removed one at a
                time. `diagnose.py` attributes nearly every remaining miss to
                the scorer, so this is where the headroom for P2/P3 is.

Run at a single budget: the budget sweep is already in run_locomo.py, and the
question here is component attribution, not the cost-quality frontier.

Usage:
    python3 run_ablations.py                 # default budget 2048
    python3 run_ablations.py --budget 4096
    MEMGATE_EMBEDDER=hash python3 run_ablations.py    # faster, no model
"""
import argparse
import csv
import os
import sys

from memgate.data import load_locomo
from memgate.policies import MemGatePolicy
from memgate.harness import run_policy
from memgate.scoring import HeuristicScorer, SIGNALS
from memgate.utils import backend_info, warm_cache, reset_cache

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# name -> kwargs disabling exactly one architectural component
ARCH = [
    ("full system", {}),
    ("- consolidation merge", {"merge_on_consolidate": False}),
    ("- informative compressor", {"informative_compress": False}),
    ("- density packing", {"pack_by_density": False}),
    ("- retrieval (tier 3)", {"use_retrieval": False}),
    ("- working memory (tier 2)", {"use_working": False}),
    ("- supersession", {"supersede": False}),
]

# parameter sensitivity: not ablations, but the same question asked of numbers
PARAMS = [
    ("buffer N=2", {"short_capacity": 2}),
    ("buffer N=12", {"short_capacity": 12}),
    ("tau_fact=0.30", {"tau_fact": 0.30}),
    ("tau_fact=0.60", {"tau_fact": 0.60}),
    ("chunk_tokens=32", {"chunk_tokens": 32}),
    ("chunk_tokens=128", {"chunk_tokens": 128}),
]


def evaluate(data, budget, store_budget=None, write_mode="threshold", **kw):
    """One config across every conversation, micro-averaged over questions."""
    tot_q = tot_hits = ans_n = ans_hits = 0
    soft_sum = tok_sum = stored_sum = 0.0
    n_conv = 0
    for _sid, turns, questions in data:
        if not questions:
            continue
        r = run_policy(MemGatePolicy(budget=budget, store_budget=store_budget,
                                     write_mode=write_mode, **kw),
                       turns, questions, budget)
        stored_sum += r["stored_tokens"]
        n_conv += 1
        n = r["n"]
        tot_q += n
        tot_hits += r["hits"]
        soft_sum += r["soft_recall"] * n
        tok_sum += r["avg_tokens"] * n
        ans_n += r["answer_n"]
        ans_hits += r["answer_hits"]
    return {
        "strict": tot_hits / tot_q if tot_q else 0.0,
        "soft": soft_sum / tot_q if tot_q else 0.0,
        "answer": ans_hits / ans_n if ans_n else 0.0,
        "tokens": tok_sum / tot_q if tot_q else 0.0,
        "stored": stored_sum / n_conv if n_conv else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048)
    # §22.2 found the compression machinery to be a net loss against keeping
    # everything -- measured with the STORE unbounded, where forgetting can only
    # lose. These re-run the same battery under the constraint §22.3 identified
    # as missing, which is the accounting under which the question is meaningful.
    ap.add_argument("--store-budget", type=int, default=None,
                    help="cap total stored tokens (default: unbounded, as in §22)")
    ap.add_argument("--adaptive", action="store_true",
                    help="store verbatim and compress on eviction, rather than "
                         "routing by fixed thresholds at ingest")
    args = ap.parse_args()
    common = {"store_budget": args.store_budget,
              "write_mode": "adaptive" if args.adaptive else "threshold"}
    if args.store_budget is not None:
        # Only meaningful when the store is capped: with unbounded storage
        # nothing is ever evicted, so both rows would be no-ops.
        ARCH.extend([("- scored eviction (FIFO)", {"scored_eviction": False}),
                     ("- demotion on pressure", {"demote_on_pressure": False})])

    print("\nMemGate — ablation study")
    print("=" * 78)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    print(f"  context budget: {args.budget}   store budget: "
          f"{args.store_budget or 'unbounded'}   write mode: {common['write_mode']}")

    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, turns, _q in data for t in turns]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)

    rows = []
    base = evaluate(data, args.budget, **common)
    rows.append(("architecture", "full system", base, 0.0))

    def show_header(title):
        print(f"\n{title}")
        print("-" * 78)
        print(f"{'Configuration':<30}{'Strict':>9}{'Δ':>8}{'Soft':>9}"
              f"{'Answer':>9}{'Tokens':>9}")

    show_header("ARCHITECTURE — remove one component")
    print(f"{'full system':<30}{base['strict']*100:>8.1f}%{'--':>8}"
          f"{base['soft']*100:>8.1f}%{base['answer']*100:>8.1f}%"
          f"{base['tokens']:>9.0f}")
    for name, kw in ARCH[1:]:
        r = evaluate(data, args.budget, **common, **kw)
        d = (r["strict"] - base["strict"]) * 100
        rows.append(("architecture", name, r, d))
        print(f"{name:<30}{r['strict']*100:>8.1f}%{d:>+8.1f}"
              f"{r['soft']*100:>8.1f}%{r['answer']*100:>8.1f}%{r['tokens']:>9.0f}")

    show_header("SCORING SIGNALS — remove one signal")
    for sig in SIGNALS:
        kw = {"scorer": HeuristicScorer(signals=[s for s in SIGNALS if s != sig])}
        r = evaluate(data, args.budget, **common, **kw)
        d = (r["strict"] - base["strict"]) * 100
        rows.append(("scoring", f"- {sig}", r, d))
        print(f"{'- ' + sig:<30}{r['strict']*100:>8.1f}%{d:>+8.1f}"
              f"{r['soft']*100:>8.1f}%{r['answer']*100:>8.1f}%{r['tokens']:>9.0f}")

    show_header("PARAMETER SENSITIVITY")
    for name, kw in PARAMS:
        r = evaluate(data, args.budget, **common, **kw)
        d = (r["strict"] - base["strict"]) * 100
        rows.append(("parameter", name, r, d))
        print(f"{name:<30}{r['strict']*100:>8.1f}%{d:>+8.1f}"
              f"{r['soft']*100:>8.1f}%{r['answer']*100:>8.1f}%{r['tokens']:>9.0f}")

    # what matters most
    arch = [r for r in rows if r[0] == "architecture" and r[3] != 0.0]
    score = [r for r in rows if r[0] == "scoring"]
    print("\n" + "=" * 78)
    if arch:
        worst = min(arch, key=lambda r: r[3])
        print(f"  Most load-bearing component : {worst[1][2:]}  ({worst[3]:+.1f} pts)")
    if score:
        ws = min(score, key=lambda r: r[3])
        print(f"  Most load-bearing signal    : {ws[1][2:]}  ({ws[3]:+.1f} pts)")
        free = [r for r in score if r[3] >= 0]
        if free:
            print(f"  Signals that cost nothing   : "
                  f"{', '.join(r[1][2:] for r in free)}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"ablations_{args.budget}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "configuration", "strict_recall", "delta_pts",
                    "soft_recall", "answer_recall", "avg_tokens", "budget",
                    "embedder", "tokenizer"])
        for fam, name, r, d in rows:
            w.writerow([fam, name, f"{r['strict']:.4f}", f"{d:+.2f}",
                        f"{r['soft']:.4f}", f"{r['answer']:.4f}",
                        f"{r['tokens']:.1f}", args.budget,
                        info["embedder"], info["tokenizer"]])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Step 0 experiment: baselines vs MemGate on the synthetic multi-session set.

Usage:
    python run_experiment.py                # full run + CSV + plot
    python run_experiment.py --quick        # table only
"""
import argparse
import os
from memgate import (build_dataset, run_policy, format_table, to_csv,
                     FullContextPolicy, OraclePolicy, SlidingWindowPolicy,
                     MemGatePolicy)

BUDGETS = [200, 400, 800, 1600, 3200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    turns, questions = build_dataset(args.sessions, args.turns)
    total_tokens = sum(len(t.text.split()) for t in turns)
    print(f"\nDataset: {len(turns)} turns across {args.sessions} sessions, "
          f"{len(questions)} probe questions (~{total_tokens} words).")
    print("Probes are asked at the very end, so early facts are maximally "
          "distant from where they were stated.\n")

    rows = []

    # --- fixed-cost baselines (budget-independent) --------------------------
    rows.append(run_policy(FullContextPolicy(), turns, questions, budget=10**9))
    rows.append(run_policy(OraclePolicy(), turns, questions, budget=3200))

    # --- budget sweep -------------------------------------------------------
    for b in BUDGETS:
        # budget reaches the constructor as well, so a policy can size its
        # store to the budget it will be asked to fill (see MemGatePolicy)
        rows.append(run_policy(SlidingWindowPolicy(budget=b), turns, questions, b))
        rows.append(run_policy(MemGatePolicy(budget=b), turns, questions, b))

    print(format_table(rows))

    # --- headline comparison ------------------------------------------------
    full = rows[0]
    mg = [r for r in rows if r["policy"].startswith("P1") and r["budget"] == 800][0]
    p0 = [r for r in rows if r["policy"].startswith("P0") and r["budget"] == 800][0]
    print("\n" + "=" * 62)
    print("HEADLINE (at 800-token budget)")
    print("=" * 62)
    print(f"  Full-context : recall {full['recall']*100:5.1f}%   "
          f"avg tokens {full['avg_tokens']:.0f}")
    print(f"  P0 window    : recall {p0['recall']*100:5.1f}%   "
          f"avg tokens {p0['avg_tokens']:.0f}")
    print(f"  P1 MemGate   : recall {mg['recall']*100:5.1f}%   "
          f"avg tokens {mg['avg_tokens']:.0f}")
    if mg["avg_tokens"]:
        print(f"\n  MemGate keeps {mg['recall']/max(full['recall'],1e-9)*100:.0f}% "
              f"of full-context recall using "
              f"{full['avg_tokens']/mg['avg_tokens']:.1f}x fewer tokens.")
    print(f"  Routing: {mg['stats']}")

    if args.quick:
        return

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "step0_results.csv")
    to_csv(rows, csv_path)
    print(f"\nSaved {csv_path}")

    try:
        plot(rows, os.path.join(args.outdir, "frontier.png"), full)
        print(f"Saved {os.path.join(args.outdir, 'frontier.png')}")
    except ImportError:
        print("matplotlib not installed — skipped plot (pip install matplotlib)")


def plot(rows, path, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for prefix, label, colour, marker in [
        ("P0", "P0 sliding window", "#C07C22", "s"),
        ("P1", "P1 MemGate", "#0E7C86", "o"),
    ]:
        pts = sorted([(r["avg_tokens"], r["recall"] * 100)
                      for r in rows if r["policy"].startswith(prefix)])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=marker, color=colour, label=label, linewidth=2)

    ax.axhline(full["recall"] * 100, ls="--", color="#5C6675", linewidth=1.2)
    ax.annotate(f"full-context ceiling — {full['recall']*100:.0f}% at "
                f"{full['avg_tokens']:.0f} tokens",
                xy=(0.97, full["recall"] * 100 - 7),
                xycoords=("axes fraction", "data"),
                fontsize=9, color="#5C6675", ha="right")

    # mark where MemGate first reaches the ceiling
    mg = sorted([(r["avg_tokens"], r["recall"])
                 for r in rows if r["policy"].startswith("P1")])
    for tok, rec in mg:
        if rec >= full["recall"] - 1e-9:
            ax.annotate(f"same recall,\n{full['avg_tokens']/tok:.1f}x fewer tokens",
                        xy=(tok, rec * 100), xytext=(tok + 450, 74),
                        fontsize=9, color="#0E7C86",
                        arrowprops=dict(arrowstyle="->", color="#0E7C86", lw=1.2))
            break

    ax.set_xlabel("Average tokens per query  (cost)")
    ax.set_ylabel("Context recall %  (quality)")
    ax.set_title("MemGate Step 0 — accuracy-per-token frontier\n"
                 "(synthetic sanity set; LoCoMo replaces this in Phase 1)",
                 fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()

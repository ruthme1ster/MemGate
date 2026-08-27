#!/usr/bin/env python3
"""Publication figures.

Design notes, so they are reproducible rather than a matter of taste:

* Four categorical hues in FIXED order, validated with the dataviz palette
  checker (lightness band, chroma floor, CVD separation, normal-vision floor).
  Worst adjacent pair is deutan dE 9.2 -- above the 8 floor.
* Every series also carries a distinct MARKER SHAPE. That is secondary encoding,
  so identity never rests on colour alone: it survives colour-blind readers,
  grayscale printing and a photocopied panel handout.
* One y-axis per plot. Answer recall is the primary metric throughout, because
  strict recall credits a compressed gist for evidence ids whose text it dropped
  (§19), and every figure here compares policies that differ in exactly that.
* Grid and spines are recessive; no value labels on every point.

Usage:  python3 make_figures.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figures")

# validated categorical slots (light surface), in fixed assignment order
C = {"rag": "#2a78d6", "compress": "#eb6834", "select": "#1baf7a",
     "p0": "#4a3aa7", "oracle": "#52514e"}
M = {"rag": "o", "compress": "s", "select": "D", "p0": "^", "oracle": "v"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  skip: {name} not found — run its script first")
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_storage():
    rows = load("storage_sweep.csv")
    if not rows:
        return
    series = [("P1-S MemGate select", "select", "MemGate select (drop)"),
              ("P1-A MemGate adaptive", "compress", "MemGate adaptive (compress)"),
              ("RAG store-all", "rag", "RAG store-all (keep newest)"),
              ("P0 sliding-window", "p0", "P0 sliding window")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, metric, lab in ((axes[0], "answer_recall", "Answer recall (%)"),
                            (axes[1], "strict_recall", "Evidence recall (%)")):
        for pol, key, label in series:
            pts = [(int(r["store_budget"]), float(r[metric]) * 100)
                   for r in rows if r["policy"] == pol and int(r["store_budget"]) > 0]
            pts.sort()
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker=M[key],
                    color=C[key], linewidth=2, markersize=5, label=label,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_xscale("log", base=2)
        style(ax, "Storage budget S (tokens, log₂)", lab, "")
    axes[0].set_title("Answer recall — did the answer text survive?",
                      loc="left", fontsize=9.5, weight="bold", pad=8)
    axes[1].set_title("Evidence recall — generous: credits ids only",
                      loc="left", fontsize=9.5, weight="bold", pad=8)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Under a storage budget, selection pays and compression does not",
                 x=0.005, ha="left", fontsize=11, weight="bold")
    fig.text(0.005, -0.06,
             "LoCoMo, 1527 questions, context budget fixed at 2048 tokens. "
             "The two panels disagree by construction: compression keeps a turn's "
             "evidence id\nwhile discarding its text, so it leads on the right and "
             "trails on the left. Answer recall is the metric to believe.",
             ha="left", fontsize=7, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_storage_frontier")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    plt.close(fig); print("  fig1_storage_frontier")


def fig_scaling():
    rows = load("scaling.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for pol, key, label in [("P1-S select (drop)", "select", "select (drop)"),
                            ("P1-A adaptive (compress)", "compress", "adaptive (compress)"),
                            ("RAG store-all", "rag", "RAG store-all")]:
        pts = sorted((float(r["ratio"]), float(r["answer"]))
                     for r in rows if r["policy"] == pol)
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker=M[key],
                color=C[key], linewidth=2, markersize=5, label=label,
                markeredgecolor="white", markeredgewidth=0.6, linestyle="none"
                if False else "-")
    ax.set_xscale("log")
    style(ax, "Compression ratio (conversation tokens ÷ stored tokens, log)",
          "Answer recall (%)",
          "No crossover: compression never overtakes")
    ax.legend(fontsize=8)
    fig.text(0.0, -0.10,
             "Streams of 1–10 concatenated LoCoMo conversations at three storage\n"
             "budgets. The rate–distortion prediction is that compression must win\n"
             "once the store is tight enough; it does not, up to 182×.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_scaling")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    plt.close(fig); print("  fig2_scaling")


def fig_scorer():
    rows = load("learned_scorer.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels, p1, p3 = [], [], []
    for r in rows:
        labels.append(r["features"].replace("hand+MiniLM", "hand + MiniLM"))
        p1.append(float(r["p1_answer"])); p3.append(float(r["p3_answer"]))
    x = range(len(labels))
    w = 0.34
    ax.bar([i - w/2 for i in x], p1, w, color=MUTED, label="P1 heuristic (hand-tuned)")
    ax.bar([i + w/2 for i in x], p3, w, color=C["select"], label="P3 learned (fitted)")
    for i, (a, b) in enumerate(zip(p1, p3)):
        ax.text(i + w/2, b + 0.6, f"+{b-a:.1f}", ha="center", fontsize=8,
                color=INK, weight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    style(ax, "", "Answer recall (%)",
          "A fitted scorer beats hand-tuned weights")
    ax.legend(fontsize=8)
    fig.text(0.0, -0.10,
             "Leave-one-conversation-out: the model scoring a conversation never\n"
             "saw it in training. Bounds the headroom of a learned decision policy;\n"
             "not deployable, since a live agent has no future questions.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_scorer")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    plt.close(fig); print("  fig3_scorer")


def fig_cost():
    rows = load("cost_model.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for pol, key, label in [("P1-S select (drop)", "select", "select (drop)"),
                            ("RAG store-all", "rag", "RAG store-all"),
                            ("P1-A adaptive (compress)", "compress", "adaptive (compress)"),
                            ("P0 sliding-window", "p0", "P0 (no index)")]:
        pts = sorted((int(r["store_budget"]) / 1024, float(r["answer"]))
                     for r in rows if r["policy"] == pol)
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker=M[key],
                color=C[key], linewidth=2, markersize=5, label=label,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log", base=2)
    style(ax, "Storage budget (KiB, log₂)", "Answer recall (%)",
          "Charging for the index changes who wins")
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.0, -0.10,
             "Budget in bytes: utf-8 text + 1536 B per embedded item. P0 keeps no\n"
             "vectors, so it pays no index tax: it leads at 32 KiB and is overtaken\n"
             "before 64 KiB. Below that crossing an embedding index cannot earn its\n"
             "own storage — a regime invisible when the budget counts text alone.",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_cost_model")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    plt.close(fig); print("  fig4_cost_model")


def main():
    os.makedirs(FIGS, exist_ok=True)
    print(f"\nwriting figures to {FIGS}")
    fig_storage(); fig_scaling(); fig_scorer(); fig_cost()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

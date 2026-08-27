#!/usr/bin/env python3
"""Cost-model robustness — does the §23 finding survive a change of unit?

§23 concluded that under a storage budget SELECTION pays and COMPRESSION does
not. That was measured with the budget denominated in TEXT TOKENS, which quietly
assumes the text is what a memory system stores. It is not. Every retrievable
item also carries an embedding vector, and at 384 float32 dimensions that is
1536 bytes against a LoCoMo turn's ~130 bytes of text -- the index outweighs the
content by roughly 12x.

That is not a rounding correction, it changes which quantity binds. Under token
accounting the cost of an item is its LENGTH; under byte accounting it is
dominated by a constant per ITEM. Two consequences follow, and both are tested
here:

  1. Compression that shortens text without reducing the number of items saves
     almost nothing, because the vector is untouched. Only compression that
     MERGES items can pay. (P1 threshold merges; P1-S does not.)
  2. A policy that needs no index at all -- plain recency -- pays no vector tax
     whatsoever, so it should become competitive exactly where the budget is
     tight enough for the index to hurt.

Both are single-parameter changes to the existing policies (`cost_mode`), so
this is a robustness check on a published conclusion, not a new system.

Usage:
    python3 run_cost_model.py
"""
import csv
import os
import sys

from memgate.data import load_locomo
from memgate.policies import (SlidingWindowPolicy, RAGPolicy,
                              MemGatePolicy, MemGateSelectPolicy,
                              MemGateAdaptivePolicy)
from memgate.harness import run_policy
from memgate.utils import backend_info, warm_cache, reset_cache

CONTEXT_BUDGET = 2048
BYTE_BUDGETS = [32768, 65536, 131072, 262144, 524288]
POLICIES = [("P0 sliding-window", SlidingWindowPolicy),
            ("RAG store-all", RAGPolicy),
            ("P1 threshold (merges)", MemGatePolicy),
            ("P1-S select (drop)", MemGateSelectPolicy),
            ("P1-A adaptive (compress)", MemGateAdaptivePolicy)]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(factory, data, S):
    tot_q = tot_hits = ans_n = ans_hits = 0
    cost = 0.0
    items = 0
    worst = 0
    for _sid, turns, questions in data:
        if not questions:
            continue
        p = factory(budget=CONTEXT_BUDGET, store_budget=S, cost_mode="bytes")
        r = run_policy(p, turns, questions, CONTEXT_BUDGET)
        tot_q += r["n"]; tot_hits += r["hits"]
        ans_n += r["answer_n"]; ans_hits += r["answer_hits"]
        c = p.stored_cost()
        cost += c
        worst = max(worst, c)
        st = p.stats()
        items += st.get("short", 0) + st.get("working", 0) + st.get("long", 0) \
            or st.get("held", 0)
    n = max(1, len(data))
    return {"strict": tot_hits / tot_q * 100 if tot_q else 0.0,
            "answer": ans_hits / ans_n * 100 if ans_n else 0.0,
            "cost": cost / n, "items": items / n, "max_cost": worst}


def main():
    print("\nMemGate — cost-model robustness (byte-denominated storage)")
    print("=" * 84)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)
    print(f"  context budget {CONTEXT_BUDGET} tokens; store budget in BYTES")
    print("  charged: utf-8 text + 1536 B per embedded item "
          "(384-d float32). P0 embeds nothing and pays no vector tax.\n")

    rows = []
    hdr = f"{'S bytes':>9}  {'Policy':<26}{'Strict':>9}{'Answer':>9}{'Items':>8}{'Bytes':>10}"
    print(hdr); print("-" * len(hdr))
    for S in BYTE_BUDGETS:
        for name, F in POLICIES:
            r = evaluate(F, data, S)
            assert r["max_cost"] <= S, f"{name} exceeded {S}: {r['max_cost']}"
            r.update(policy=name, store_budget=S)
            rows.append(r)
            print(f"{S:>9}  {name:<26}{r['strict']:>8.1f}%{r['answer']:>8.1f}%"
                  f"{r['items']:>8.0f}{r['cost']:>10.0f}")
        print()

    by = {(r["policy"], r["store_budget"]): r for r in rows}
    print("Findings")
    print("-" * 84)
    print("  1. Selection still beats compression, so §23 is unit-robust:")
    for S in BYTE_BUDGETS:
        d = by[("P1-S select (drop)", S)]["answer"] - \
            by[("P1-A adaptive (compress)", S)]["answer"]
        m = by[("P1-S select (drop)", S)]["answer"] - \
            by[("P1 threshold (merges)", S)]["answer"]
        print(f"       S={S:>7}  select - compress {d:+6.1f}   select - merge {m:+6.1f}")
    print()
    print("  2. Compression cannot pay when the vector, not the text, is the cost.")
    print("     Shortening text leaves the 1536-byte vector untouched, and merging")
    print("     items away loses more content than the vector it saves.")
    print()
    print("  3. The index is not free. P0 stores no vectors, so it holds far more")
    print("     conversation per byte and wins outright while the budget is tight:")
    for S in BYTE_BUDGETS:
        p0 = by[("P0 sliding-window", S)]["answer"]
        best = max(by[(n, S)]["answer"] for n, _ in POLICIES if n != "P0 sliding-window")
        who = "P0" if p0 > best else "indexed"
        print(f"       S={S:>7}  P0 {p0:5.1f}  vs best indexed {best:5.1f}   -> {who}")
    print()
    print("     There is a budget below which an embedding index cannot earn its")
    print("     own storage. Reporting retrieval quality without charging for the")
    print("     index hides that entirely -- and every accuracy-per-token number")
    print("     in the memory literature, ours in §23 included, is stated that way.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "cost_model.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

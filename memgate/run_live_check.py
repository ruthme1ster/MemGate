#!/usr/bin/env python3
"""Re-run the headline comparison live and check it against the committed CSV.

    python3 run_live_check.py            # S = 4096, the slide 11 / 12 row
    python3 run_live_check.py --store 2048

Replays all ten LoCoMo conversations through RAG store-all and P1-S select at
one storage budget, with the context pinned at 2048 tokens, and prints the
answer recall each policy reaches as it goes. At the end it compares both
numbers with `results/storage_sweep.csv` -- the file every slide is generated
from -- and says whether they match.

This is for showing a panel that the numbers on the slides come out of the
code, not out of a spreadsheet. The pipeline is deterministic (fixed embedder,
exact tokenizer, no sampling), so a match is exact, not approximate.

It is the same `evaluate` path as run_storage_sweep.py, conversation by
conversation so progress is visible; nothing is re-implemented.
"""
import argparse
import csv
import os
import sys
import time

from memgate.data import load_locomo
from memgate.harness import run_policy
from memgate.policies import RAGPolicy, MemGateSelectPolicy
from memgate.utils import backend_info, warm_cache, reset_cache

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_CSV = os.path.join(HERE, "results", "storage_sweep.csv")
CONTEXT_BUDGET = 2048
ARMS = [("RAG store-all", RAGPolicy), ("P1-S MemGate select", MemGateSelectPolicy)]


def committed(store_budget):
    """{policy: answer_recall} for this budget, from the committed sweep."""
    out = {}
    with open(SWEEP_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["store_budget"]) == store_budget:
                out[r["policy"]] = float(r["answer_recall"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=int, default=4096,
                    help="storage budget S in tokens (default 4096)")
    args = ap.parse_args()
    S = args.store
    t0 = time.time()

    print(f"MemGate live check — LoCoMo, store budget S = {S}, "
          f"context {CONTEXT_BUDGET} tokens", flush=True)
    data = [d for d in load_locomo() if d[2]]
    info = backend_info()
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, turns, _q in data for t in turns]
    texts += [q.text for _s, _t, qs in data for q in qs]
    n_vec = warm_cache(texts)
    info = backend_info()
    print(f"  {len(data)} conversations, embedded {n_vec} texts "
          f"[{info['embedder']}, {info['tokenizer']}]", flush=True)

    measured = {}
    for name, factory in ARMS:
        hits = n = 0
        for i, (_sid, turns, questions) in enumerate(data, 1):
            p = factory(budget=CONTEXT_BUDGET, store_budget=S)
            r = run_policy(p, turns, questions, CONTEXT_BUDGET)
            hits += r["answer_hits"]
            n += r["answer_n"]
            print(f"  {name:<20} conversation {i:>2}/{len(data)}   "
                  f"answer recall so far {hits / n * 100:5.1f}%  ({hits}/{n})",
                  flush=True)
        measured[name] = hits / n

    ref = committed(S)
    print(f"\nRESULT  (S = {S}, {time.time() - t0:.0f} s)", flush=True)
    ok = True
    for name, _f in ARMS:
        got, want = measured[name], ref.get(name)
        same = want is not None and round(got, 4) == round(want, 4)
        ok &= same
        print(f"  {name:<20} live {got * 100:5.1f}%   committed "
              f"{'—' if want is None else f'{want * 100:5.1f}%'}   "
              f"{'MATCH' if same else 'DIFFERS'}", flush=True)
    gap = (measured["P1-S MemGate select"] - measured["RAG store-all"]) * 100
    print(f"  selection over FIFO: {gap:+.1f} answer-recall points", flush=True)
    if not ok and "minilm" not in info["embedder"].lower():
        print("  note: the committed numbers use MiniLM; this run used "
              f"{info['embedder']}, so they are not expected to match.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

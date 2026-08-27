#!/usr/bin/env python3
"""Scaling study — where does compression finally start to pay?

§23 established that under a storage budget, SELECTION beats keep-everything
while COMPRESSION loses: dropping an evicted turn beats compressing it on
answer-presence at every budget tested. The rate-distortion prediction is that
this must reverse eventually -- once the store is squeezed hard enough that even
the selected subset will not fit verbatim, a lossy record has to beat no record.

Sweeping S downward on LoCoMo cannot test that prediction. By S=256 every policy
is at 0.2% strict / ~2.4% answer, so the regime where compression should win is
below the floor where anything works at all, and the comparison is noise.

The missing axis is not a smaller store, it is a LONGER CONVERSATION. This
builds one by concatenating k LoCoMo conversations into a single stream and
pooling their questions, holding S fixed. Compression ratio (stream tokens per
stored token) then runs from ~9x at k=1 to ~91x at k=10, and the crossover -- if
it exists -- has room to appear.

CAVEAT, to be stated wherever these numbers are used: a concatenated stream is
NOT a real 186K-token conversation. Its questions each concern one constituent
conversation, so the other k-1 act as pure distractors, and session timestamps
interleave incoherently. It is a stress test of RETENTION UNDER PRESSURE, which
is the variable under study, not a natural long-dialogue benchmark. Turn ids are
already namespaced `<sample_id>/<dia_id>` (§14), so merging cannot alias
evidence across conversations.

Usage:
    python3 run_scaling.py
"""
import csv
import os
import sys

from memgate.data import load_locomo
from memgate.policies import (RAGPolicy, MemGateSelectPolicy,
                              MemGateAdaptivePolicy)
from memgate.harness import run_policy
from memgate.utils import (backend_info, warm_cache, reset_cache, count_tokens)

CONTEXT_BUDGET = 2048
STORE_BUDGETS = [1024, 2048, 4096]
SCALES = [1, 2, 5, 10]
POLICIES = [("RAG store-all", RAGPolicy),
            ("P1-S select (drop)", MemGateSelectPolicy),
            ("P1-A adaptive (compress)", MemGateAdaptivePolicy)]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_streams(data, k):
    """Concatenate conversations k at a time; pool their questions."""
    streams = []
    for i in range(0, len(data), k):
        group = data[i:i + k]
        if len(group) < k:
            break                     # only whole groups, so k is exact
        turns, questions = [], []
        for _sid, ts, qs in group:
            turns.extend(ts)
            questions.extend(qs)
        streams.append((f"x{k}", turns, questions))
    return streams


def evaluate(factory, streams, store_budget):
    tot_q = tot_hits = ans_n = ans_hits = 0
    stored = 0.0
    for _sid, turns, questions in streams:
        r = run_policy(factory(budget=CONTEXT_BUDGET, store_budget=store_budget),
                       turns, questions, CONTEXT_BUDGET)
        tot_q += r["n"]
        tot_hits += r["hits"]
        ans_n += r["answer_n"]
        ans_hits += r["answer_hits"]
        stored += r["stored_tokens"]
    return (tot_hits / tot_q * 100 if tot_q else 0.0,
            ans_hits / ans_n * 100 if ans_n else 0.0,
            stored / len(streams))


def main():
    print("\nMemGate — scaling study: when does compression start to pay?")
    print("=" * 84)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    print(f"  warmed embedding cache: {warm_cache(texts)} vectors")
    print(f"  context budget fixed at {CONTEXT_BUDGET}\n")

    rows = []
    for S in STORE_BUDGETS:
        print(f"Store budget S = {S}")
        print(f"  {'stream':>7}{'conv tok':>10}{'ratio':>8}   "
              f"{'RAG':>14}{'select(drop)':>16}{'adapt(compress)':>18}")
        print("  " + "-" * 76)
        for k in SCALES:
            streams = make_streams(data, k)
            if not streams:
                continue
            conv_tok = sum(count_tokens(f"[{t.speaker}] {t.text}")
                           for _s, ts, _q in streams for t in ts) / len(streams)
            cells, res = [], {}
            for name, F in POLICIES:
                st, an, sto = evaluate(F, streams, S)
                res[name] = (st, an)
                cells.append(f"{st:6.1f} {an:6.1f}")
                rows.append({"store_budget": S, "scale": k, "policy": name,
                             "conv_tokens": conv_tok, "ratio": conv_tok / S,
                             "strict": st, "answer": an, "stored": sto,
                             "n_streams": len(streams)})
            drop = res["P1-S select (drop)"][1]
            comp = res["P1-A adaptive (compress)"][1]
            flag = "  <== compress wins" if comp > drop else ""
            print(f"  {'x'+str(k):>7}{conv_tok:>10.0f}{conv_tok/S:>7.0f}x   "
                  f"{cells[0]:>14}{cells[1]:>16}{cells[2]:>18}{flag}")
        print()

    print("Reading it")
    print("-" * 84)
    print("  Each cell is `strict answer`, both in %. `ratio` is conversation")
    print("  tokens per stored token -- how hard the store is being squeezed.")
    print("  The prediction under test: compression must overtake dropping once")
    print("  the ratio is high enough that the selected subset cannot fit whole.")
    wins = [r for r in rows if r["policy"] == "P1-A adaptive (compress)"]
    best = None
    for w in wins:
        d = next(r for r in rows if r["store_budget"] == w["store_budget"]
                 and r["scale"] == w["scale"] and r["policy"] == "P1-S select (drop)")
        if w["answer"] > d["answer"] and (best is None or w["ratio"] < best[0]):
            best = (w["ratio"], w["store_budget"], w["scale"])
    print()
    if best:
        print(f"  Crossover found: compression first wins on answer recall at a")
        print(f"  compression ratio of ~{best[0]:.0f}x (S={best[1]}, stream=x{best[2]}).")
        print(f"  Below that ratio, keeping a selected subset verbatim is better.")
    else:
        print("  NO crossover anywhere in the range tested. Dropping beats")
        print("  compressing at every compression ratio up to "
              f"{max(r['ratio'] for r in rows):.0f}x. Report that as the result:")
        print("  extractive compression of dialogue turns does not pay at any")
        print("  storage pressure this benchmark can reach.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "scaling.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

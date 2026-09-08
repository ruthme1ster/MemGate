#!/usr/bin/env python3
"""End-task accuracy — what a real model actually answers from the context.

Every number this project has reported so far is *context recall*: did the
evidence reach the prompt? That is the CEILING on accuracy, not accuracy. §26
named this the single most valuable remaining experiment, and the vendored
local model makes it reachable offline.

The question is narrow and worth stating exactly:

    The decision policy is worth +8.7 answer-recall points over FIFO at
    identical storage (§23, §24.4). How many of those points does a reader
    model actually convert into correct answers?

Design — one variable, everything else pinned:

  * Same context budget (2048) and store budget (4096) as §23/§24, so the
    contexts are the ones whose recall was already measured.
  * Same question subset the answer metric is defined over: the
    `answer_recoverable` questions, where the answer text is literally present
    in the evidence turns. Elsewhere "did the answer survive" is undefined, so
    an end-task number there would measure the benchmark's phrasing.
  * Same reader for every arm: Qwen2.5-1.5B-Instruct-4bit, greedy, 32 tokens.
    The reader is held fixed so any difference is the memory policy.

Two controls the comparison cannot be read without:

  closed-book   the question with NO memory at all. A model that answers from
                parametric knowledge would make every policy look good, and
                without this row we could not tell. It is the floor that the
                memory system has to beat to have done anything.
  oracle        perfect selection of exactly the cited evidence. The ceiling.

Accuracy is NOT bounded by answer recall, which is why the run reports the
decomposition rather than a single number: a model can guess, so accuracy
splits into `answer present in context` (can it USE what it was given) and
`answer absent` (did it guess anyway). The first is what a memory policy can
influence; the second is what it cannot.

Scoring uses the SAME normaliser as the answer-presence metric
(`utils.answer_tokens`), so "the answer reached the context" and "the answer
reached the output" are the same predicate applied at two points in the
pipeline, and the ceiling comparison is exact rather than approximate.

Generations are cached on disk by (model, prompt) hash and checkpointed, so an
interrupted run resumes -- the same discipline as the P2 judge cache (§25).

Usage:
    python3 run_endtask.py --limit 20          # smoke test, measures throughput
    python3 run_endtask.py                     # full run (resumable)
    python3 run_endtask.py --sample 300        # stratified subsample
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

from memgate.data import load_locomo
from memgate.policies import (RAGPolicy, SlidingWindowPolicy, OraclePolicy,
                              MemGateSelectPolicy, MemGateAdaptivePolicy)
from memgate.utils import backend_info, warm_cache, reset_cache, answer_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE_PATH = os.path.join(RESULTS, "endtask_cache.json")
CONTEXT_BUDGET = 2048
STORE_BUDGET = 4096
MAX_NEW = 32
RNG = np.random.default_rng(20260827)

SYSTEM = ("You answer questions about a conversation using only the notes "
          "provided. Answer in as few words as possible -- a name, a date, a "
          "number or a short phrase. If the notes do not contain the answer, "
          "say: unknown.")

WITH_CTX = ("Notes from the conversation:\n{ctx}\n\n"
            "Question: {q}\nShort answer:")
NO_CTX = "Question: {q}\nShort answer:"


# ------------------------------------------------------------------ reader
class Reader:
    """Greedy short-answer generation from a local MLX model, cached to disk.

    Greedy rather than sampled: the experiment compares memory policies, and a
    sampling temperature would add variance that has nothing to do with them.
    Two runs of the same arm must give the same answer.
    """

    def __init__(self, model_name=None, max_new=MAX_NEW):
        from memgate.llm_judge import default_model
        self.model_name = model_name or default_model()
        self.max_new = max_new
        self._model = self._tok = None
        self.cache = {}
        self.calls = self.hits = 0

    def load_cache(self):
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH) as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}
        return len(self.cache)

    def save_cache(self):
        os.makedirs(RESULTS, exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
        os.replace(tmp, CACHE_PATH)      # atomic; a killed run cannot truncate it

    def _load(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_name)

    def _key(self, prompt):
        h = hashlib.sha1(f"{os.path.basename(self.model_name)}\x00{prompt}"
                         .encode("utf-8")).hexdigest()
        return h[:16]

    def answer(self, prompt: str) -> str:
        k = self._key(prompt)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self._load()
        from mlx_lm import generate
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        out = generate(self._model, self._tok, prompt=text,
                       max_tokens=self.max_new, verbose=False)
        out = out.strip().split("\n")[0].strip()
        self.calls += 1
        self.cache[k] = out
        return out


# ------------------------------------------------------------------ scoring
def score(pred: str, gold: str):
    """(containment, F1) on the answer-presence normaliser.

    `containment` -- every content token of the gold answer appears in the
    prediction -- is the primary metric, and deliberately the same predicate
    the answer-recall metric applies to the context. F1 is reported alongside
    because containment is unforgiving of a correct answer stated at length,
    and a metric that punishes verbosity would flatter the arms whose contexts
    are too thin to be verbose about.
    """
    g = [t for t in answer_tokens(gold)]
    p = set(answer_tokens(pred))
    if not g:
        return None, None
    contains = float(set(g) <= p)
    gs = set(g)
    inter = len(gs & p)
    if inter == 0:
        return contains, 0.0
    prec, rec = inter / len(p), inter / len(gs)
    return contains, 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ contexts
def build_contexts(data, arms, qsel):
    """Replay each conversation once per arm and collect (context, present).

    `present` is the answer-presence flag recomputed here rather than imported
    from the harness, so that the ceiling and the end-task number are produced
    by the same code path over the same context string.
    """
    out = {a: [] for a in arms}
    for ci, (_sid, turns, questions) in enumerate(data):
        keep = [q for q in questions if id(q) in qsel]
        if not keep:
            continue
        for arm, factory in arms.items():
            if arm == "closed-book":
                for q in keep:
                    out[arm].append({"conv": ci, "q": q, "ctx": "", "present": False})
                continue
            pol = factory()
            for t in turns:
                pol.observe(t)
            for q in keep:
                if isinstance(pol, OraclePolicy):
                    ctx = pol.build_context(q.text, CONTEXT_BUDGET,
                                            gold_ids=set(q.evidence_ids))
                else:
                    ctx = pol.build_context(q.text, CONTEXT_BUDGET)
                present = set(answer_tokens(q.answer)) <= set(answer_tokens(ctx.text))
                out[arm].append({"conv": ci, "q": q, "ctx": ctx.text,
                                 "present": present, "tokens": ctx.tokens})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only this many questions, first come")
    ap.add_argument("--sample", type=int, default=0,
                    help="random subsample of questions (seeded)")
    ap.add_argument("--store-budget", type=int, default=STORE_BUDGET)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    S = args.store_budget

    print("\nMemGate — end-task accuracy through a real reader model")
    print("=" * 86)
    data = load_locomo()
    info = backend_info()
    print(f"  backends: embedder={info['embedder']}  tokenizer={info['tokenizer']}")
    reset_cache()
    texts = [f"[{t.speaker}] {t.text}" for _s, ts, _q in data for t in ts]
    texts += [q.text for _s, _t, qs in data for q in qs]
    warm_cache(texts)

    # The answer-recoverable subset: the only questions where "did the answer
    # survive into the context" is defined, so the only ones where end-task
    # accuracy can be read against its own ceiling.
    pool = [q for _s, _t, qs in data for q in qs if q.answer_recoverable]
    if args.sample and args.sample < len(pool):
        idx = RNG.choice(len(pool), args.sample, replace=False)
        pool = [pool[i] for i in sorted(idx)]
    if args.limit:
        pool = pool[:args.limit]
    qsel = {id(q) for q in pool}
    print(f"  {len(pool)} answer-recoverable questions "
          f"(of {sum(len(qs) for _s,_t,qs in data)} scoreable)")

    arms = {
        "closed-book": None,
        "P0 sliding-window": lambda: SlidingWindowPolicy(
            budget=CONTEXT_BUDGET, store_budget=S),
        "RAG store-all": lambda: RAGPolicy(budget=CONTEXT_BUDGET, store_budget=S),
        "P1-A adaptive": lambda: MemGateAdaptivePolicy(
            budget=CONTEXT_BUDGET, store_budget=S),
        "P1-S select": lambda: MemGateSelectPolicy(
            budget=CONTEXT_BUDGET, store_budget=S),
        "Oracle": lambda: OraclePolicy(budget=CONTEXT_BUDGET),
    }
    print(f"  context {CONTEXT_BUDGET}  store {S}  arms {len(arms)}")

    t0 = time.time()
    ctxs = build_contexts(data, arms, qsel)
    print(f"  contexts assembled in {time.time()-t0:.1f}s\n")

    reader = Reader(args.model)
    have = reader.load_cache()
    print(f"  reader: {os.path.basename(reader.model_name)}   "
          f"cache {have} generations\n")

    rows, per_q = [], {}
    total = sum(len(v) for v in ctxs.values())
    done = 0
    t0 = time.time()
    for arm, recs in ctxs.items():
        acc, f1s, n = 0.0, 0.0, 0
        pres_hit = pres_n = abs_hit = abs_n = 0
        outcomes, convs, presents = [], [], []
        for r in recs:
            q = r["q"]
            prompt = (NO_CTX.format(q=q.text) if arm == "closed-book"
                      else WITH_CTX.format(ctx=r["ctx"], q=q.text))
            pred = reader.answer(prompt)
            c, f1 = score(pred, q.answer)
            done += 1
            if c is None:
                continue
            acc += c; f1s += f1; n += 1
            outcomes.append(c); convs.append(r["conv"]); presents.append(r["present"])
            if r["present"]:
                pres_n += 1; pres_hit += int(c)
            else:
                abs_n += 1; abs_hit += int(c)
            if reader.calls and reader.calls % 100 == 0:
                reader.save_cache()
                el = time.time() - t0
                rate = reader.calls / max(el, 1e-9)
                print(f"    {done}/{total}  {rate:.2f} gen/s  "
                      f"eta {(total-done)/max(rate,1e-9)/60:.0f} min", flush=True)
        reader.save_cache()
        ceiling = sum(1 for r in recs if r["present"]) / max(len(recs), 1) * 100
        rows.append({
            "arm": arm, "n": n,
            "accuracy": acc / n * 100 if n else 0.0,
            "f1": f1s / n * 100 if n else 0.0,
            "answer_recall": ceiling,
            "acc_when_present": pres_hit / pres_n * 100 if pres_n else 0.0,
            "n_present": pres_n,
            "acc_when_absent": abs_hit / abs_n * 100 if abs_n else 0.0,
            "n_absent": abs_n,
            "store_budget": S, "context_budget": CONTEXT_BUDGET,
            "model": os.path.basename(reader.model_name),
        })
        per_q[arm] = (np.array(outcomes), np.array(convs), np.array(presents))

    print(f"\n{'Arm':<20}{'Accuracy':>10}{'F1':>8}{'Ceiling':>10}"
          f"{'acc|present':>13}{'acc|absent':>12}")
    print("-" * 86)
    for r in rows:
        print(f"{r['arm']:<20}{r['accuracy']:>9.1f}%{r['f1']:>7.1f}%"
              f"{r['answer_recall']:>9.1f}%"
              f"{r['acc_when_present']:>12.1f}%{r['acc_when_absent']:>11.1f}%")

    # ---- the comparison the project is about, with a clustered interval
    print("\nDoes the decision-policy gap survive a real reader?")
    print("-" * 86)
    if "P1-S select" in per_q and "RAG store-all" in per_q:
        a, conv, _ = per_q["P1-S select"]
        b = per_q["RAG store-all"][0]
        m = min(len(a), len(b))
        d = a[:m] - b[:m]
        obs = d.mean() * 100
        groups = [d[conv[:m] == c] for c in np.unique(conv[:m])]
        groups = [g for g in groups if len(g)]
        boot = np.array([np.concatenate(
            [groups[j] for j in RNG.integers(0, len(groups), len(groups))]).mean()
            for _ in range(10000)]) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ceil_s = next(r for r in rows if r["arm"] == "P1-S select")["answer_recall"]
        ceil_r = next(r for r in rows if r["arm"] == "RAG store-all")["answer_recall"]
        print(f"  ceiling gap  (answer recall) : {ceil_s-ceil_r:+.1f} pts")
        print(f"  realised gap (accuracy)      : {obs:+.1f} pts   "
              f"95% CI by conversation [{lo:+.1f}, {hi:+.1f}]"
              f"   {'significant' if lo > 0 or hi < 0 else 'NOT significant'}")
        conv_rate = obs / (ceil_s - ceil_r) * 100 if abs(ceil_s - ceil_r) > 1e-9 else 0
        print(f"  conversion                   : {conv_rate:.0f}% of the ceiling gap")
    cb = next((r for r in rows if r["arm"] == "closed-book"), None)
    if cb:
        print(f"\n  closed-book floor: {cb['accuracy']:.1f}% — what the reader")
        print("  answers with NO memory at all. Every arm must beat this to have")
        print("  contributed anything; the margin above it is the memory system.")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "endtask.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    print(f"  {reader.calls} generations, {reader.hits} cache hits, "
          f"{(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

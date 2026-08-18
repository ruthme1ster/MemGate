"""Evaluation harness.

Metric: CONTEXT RECALL @ BUDGET -- did the assembled context actually contain
the gold evidence, given a token budget?

Why this metric for Step 0: it isolates the variable we are studying (the
memory decision policy) from the model's reasoning ability, and needs no API
key, so it is fast, deterministic and reproducible. Phase 3 adds end-task
accuracy by feeding this same context to a real LLM -- context recall is the
ceiling on that accuracy, so it stays meaningful.
"""
import time
import statistics
from typing import List, Dict
from .types import Turn, Question
from .policies import OraclePolicy
from .utils import answer_tokens


def run_policy(policy, turns: List[Turn], questions: List[Question],
               budget: int) -> Dict:
    """Replay the conversation, then answer every probe.

    Two recall numbers, because with multi-evidence questions they diverge:

      strict  fraction of questions where EVERY evidence turn reached the
              context. This is the answerable rate and the headline metric --
              retrieving 18 of 19 required turns answers nothing.
      soft    mean fraction of evidence turns retrieved. Partial credit, which
              shows whether a policy is close or hopeless on the hard ones.

    On single-evidence questions the two coincide, which is why Step 0 only
    needed one of them.
    """
    t0 = time.perf_counter()
    for turn in turns:
        policy.observe(turn)
    ingest_ms = (time.perf_counter() - t0) * 1000

    hits, soft_sum, token_counts, latencies = 0, 0.0, [], []
    ans_n = ans_hits = 0
    by_cat = {}
    for q in questions:
        ev = set(q.evidence_ids)
        t1 = time.perf_counter()
        if isinstance(policy, OraclePolicy):
            ctx = policy.build_context(q.text, budget, gold_ids=ev)
        else:
            ctx = policy.build_context(q.text, budget)
        latencies.append((time.perf_counter() - t1) * 1000)
        token_counts.append(ctx.tokens)

        got = ctx.fact_ids
        strict = bool(ev) and ev <= got
        soft = (len(ev & got) / len(ev)) if ev else 0.0
        hits += int(strict)
        soft_sum += soft
        c = by_cat.setdefault(q.category, {"n": 0, "strict": 0, "soft": 0.0})
        c["n"] += 1
        c["strict"] += int(strict)
        c["soft"] += soft

        # Answer-presence: did the answer TEXT survive into the context, not
        # merely a pointer to the turn that once held it? A Tier 2 summary is
        # truncated but keeps its fact_id, so it scores a strict hit even when
        # the answer was cut. This is the check that catches that.
        if q.answer_recoverable:
            ans_n += 1
            if set(answer_tokens(q.answer)) <= set(answer_tokens(ctx.text)):
                ans_hits += 1

    n = len(questions)
    lat_sorted = sorted(latencies)
    for c in by_cat.values():
        c["recall"] = c["strict"] / c["n"]
        c["soft"] = c["soft"] / c["n"]
    return {
        "policy": policy.name,
        "budget": budget,
        "recall": hits / n if n else 0.0,
        "soft_recall": soft_sum / n if n else 0.0,
        "answer_recall": ans_hits / ans_n if ans_n else 0.0,
        "answer_n": ans_n,
        "answer_hits": ans_hits,
        "hits": hits,
        "n": n,
        "avg_tokens": statistics.mean(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "p95_latency_ms": lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0,
        "ingest_ms": ingest_ms,
        "by_category": by_cat,
        "stats": policy.stats(),
    }


def sweep(policy_factory, turns, questions, budgets) -> List[Dict]:
    """Fresh policy per budget -- state must never leak between runs.

    The budget reaches the constructor as well as build_context, so a policy
    can size its store to the budget it will be asked to fill.
    """
    out = []
    for b in budgets:
        out.append(run_policy(policy_factory(budget=b), turns, questions, b))
    return out


def format_table(rows: List[Dict]) -> str:
    hdr = f"{'Policy':<22}{'Budget':>8}{'Recall':>9}{'AvgTok':>9}{'Tok/hit':>9}{'p95 ms':>9}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        tph = (r["avg_tokens"] * r["n"] / r["hits"]) if r["hits"] else float("inf")
        tph_s = f"{tph:>9.0f}" if r["hits"] else f"{'--':>9}"
        lines.append(
            f"{r['policy']:<22}{r['budget']:>8}{r['recall']*100:>8.1f}%"
            f"{r['avg_tokens']:>9.0f}{tph_s}{r['p95_latency_ms']:>9.2f}"
        )
    return "\n".join(lines)


def to_csv(rows: List[Dict], path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "budget", "recall", "hits", "n",
                    "avg_tokens", "max_tokens", "p95_latency_ms"])
        for r in rows:
            w.writerow([r["policy"], r["budget"], f"{r['recall']:.4f}",
                        r["hits"], r["n"], f"{r['avg_tokens']:.1f}",
                        r["max_tokens"], f"{r['p95_latency_ms']:.3f}"])

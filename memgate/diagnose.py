#!/usr/bin/env python3
"""Diagnostic: WHY does P1 miss the facts it misses?

This is the most useful thing to show a reviewer: it quantifies exactly how
much headroom the P2/P3 scorers have, and localises the failure to one
component (the heuristic scorer) rather than the architecture.
"""
from memgate import build_dataset, MemGatePolicy
from memgate.scoring import HeuristicScorer

BUDGET = 400

turns, questions = build_dataset()
pol = MemGatePolicy()
for t in turns:
    pol.observe(t)

stored = {i.fact_id for i in pol.store.long if i.fact_id and i.is_active}
stored |= {i.fact_id for i in pol.store.working if i.fact_id}

missed, recovered = [], []
for q in questions:
    ctx = pol.build_context(q.text, BUDGET)
    (recovered if q.gold_fact_id in ctx.fact_ids else missed).append(q)

print(f"\nProbes: {len(questions)}   recovered: {len(recovered)}   missed: {len(missed)}")
print(f"Facts that reached a durable tier: {len(stored)}/{len(questions)}\n")

# classify each miss
scorer = HeuristicScorer()
fact_text = {}
for t in turns:
    if t.fact_id and t.fact_id not in fact_text:
        fact_text[t.fact_id] = t.text

print("MISS ANALYSIS")
print("-" * 78)
causes = {"scorer_miss": 0, "dedupe_collapse": 0, "retrieval_miss": 0}
if not missed:
    print("  (no misses at this budget)")

for q in missed:
    txt = fact_text.get(q.gold_fact_id, "")
    u, label = scorer.score(txt, "user")
    if q.gold_fact_id in stored:
        cause = "retrieval_miss"
        why = "reached a durable tier, but retrieval ranked it below the budget cut"
    elif u < pol.tau_fact:
        cause = "scorer_miss"
        why = f"u={u:.2f} < tau_fact={pol.tau_fact} -> P1 did not see it as durable"
    else:
        cause = "dedupe_collapse"
        why = (f"u={u:.2f} passed tau_fact, so it was ROUTED but then absorbed by "
               f"the novelty check against a near-identical earlier item")
    causes[cause] += 1
    print(f"  {q.gold_fact_id:<18} u={u:4.2f} {label:<8} {cause}")
    print(f"      fact : {txt[:70]}")
    print(f"      why  : {why}")

print("-" * 78)
print("CAUSE SUMMARY:", causes)
print("""
Interpretation
  * scorer_miss      -> P1's heuristic failed to recognise a durable fact.
                        This is exactly the error an LLM judge (P2) and its
                        distilled student (P3) are meant to remove, so it
                        quantifies the headroom available in Phase 3.
  * dedupe_collapse  -> the crude hashing embedder cannot separate two facts
                        with similar surface form. Fixed by the real embedder
                        (sentence-transformers) in Phase 1.
  * retrieval_miss   -> a ranking problem, not a memory-policy problem.

Each cause is localised to one swappable component, which is what makes the
next phase a measurable experiment rather than a guess.
""")

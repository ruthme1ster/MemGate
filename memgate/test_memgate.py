#!/usr/bin/env python3
"""Tests. Run: python test_memgate.py"""
import sys
from memgate import (build_dataset, MemGatePolicy, SlidingWindowPolicy,
                     FullContextPolicy, run_policy)
from memgate.scoring import HeuristicScorer
from memgate.store import ThreeTierStore
from memgate.types import MemoryItem, SHORT_TERM
from memgate.utils import count_tokens, embed, cosine

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("\n-- utils --")
from memgate.utils import backend_info
print(f"  backends: {backend_info()}")
check("tokenizer non-zero", count_tokens("hello world") > 0)
check("tokenizer empty", count_tokens("") == 0)


def _same_vec(a, b):
    """Backend-agnostic equality: the embedder returns numpy arrays when numpy
    is present and plain lists otherwise."""
    return len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))


check("embedding deterministic", _same_vec(embed("the deadline is March 15"),
                                           embed("the deadline is March 15")))
check("cosine self ~ 1", abs(cosine(embed("abc def"), embed("abc def")) - 1.0) < 1e-6)
check("embedding is normalised",
      abs(cosine(embed("a longer sentence here"),
                 embed("a longer sentence here")) - 1.0) < 1e-6)
check("cosine separates unrelated text",
      cosine(embed("budget 40000 rupees"), embed("hello there friend")) < 0.9)

print("\n-- scorer --")
s = HeuristicScorer()
check("filler scores low", s.score("ok thanks")[0] < 0.1)
check("greeting scores low", s.score("hey")[0] < 0.1)
check("fact scores high", s.score("The deadline is March 15.")[0] > 0.45)
check("budget fact scores high", s.score("My budget is 40000 rupees.")[0] > 0.45)
check("fact labelled", s.score("My budget is 40000 rupees.")[1] == "fact")
check("chatter below fact threshold",
      s.score("That makes sense to me overall.")[0] < 0.45)

print("\n-- store --")
st = ThreeTierStore(short_capacity=3)
ev = [st.add_short(MemoryItem(text=f"turn {i}", tier=SHORT_TERM, kind="turn",
                              session=0, turn_index=i)) for i in range(5)]
check("buffer respects capacity", len(st.short) == 3)
check("evicts oldest first", ev[3] is not None and ev[3].text == "turn 0")
check("no eviction while under capacity", ev[0] is None and ev[2] is None)

st2 = ThreeTierStore()
src = MemoryItem(text="x", tier=SHORT_TERM, kind="turn", session=0, turn_index=0)
st2.add_long("The deadline is March 15.", src)
st2.add_long("The deadline is March 15.", src)      # exact duplicate
check("dedupe blocks duplicate", len(st2.long) == 1, f"got {len(st2.long)}")
st2.add_long("Ravi joined as a developer.", src)
check("distinct fact stored", len(st2.long) == 2, f"got {len(st2.long)}")

print("\n-- dataset integrity --")
turns, questions = build_dataset(n_sessions=8, turns_per_session=20, n_facts=16)
stmts = [t.text for t in turns if t.fact_id]
check("facts planted", len(stmts) > 0)
check("probes unique", len({q.text for q in questions}) == len(questions))
check("every probe is planted",
      {q.gold_fact_id for q in questions} <= {t.fact_id for t in turns if t.fact_id})

print("\n-- budget discipline --")
turns, questions = build_dataset()
for budget in (100, 400, 1200):
    p = MemGatePolicy()
    for t in turns:
        p.observe(t)
    ctx = p.build_context(questions[0].text, budget)
    check(f"MemGate respects budget {budget}", ctx.tokens <= budget,
          f"used {ctx.tokens}")

    p0 = SlidingWindowPolicy()
    for t in turns:
        p0.observe(t)
    c0 = p0.build_context(questions[0].text, budget)
    check(f"P0 respects budget {budget}", c0.tokens <= budget, f"used {c0.tokens}")

print("\n-- end-to-end behaviour --")
r_full = run_policy(FullContextPolicy(), turns, questions, 10**9)
r_mg = run_policy(MemGatePolicy(), turns, questions, 800)
r_p0 = run_policy(SlidingWindowPolicy(), turns, questions, 800)
check("full-context recalls everything", r_full["recall"] == 1.0)
check("MemGate beats sliding window at equal budget",
      r_mg["recall"] > r_p0["recall"], f"{r_mg['recall']} vs {r_p0['recall']}")
check("MemGate far cheaper than full-context",
      r_mg["avg_tokens"] < r_full["avg_tokens"] / 3)
# Leak-free baseline. This was 0.8 when consolidation promoted on the
# ground-truth fact_id; the honest P1 number at this budget is ~0.50, and the
# write path (only ~half the planted facts clear tau_fact) is the ceiling.
# Raise this as the scorer improves -- do not relax it.
check("MemGate recall clears leak-free baseline",
      r_mg["recall"] >= 0.45, f"{r_mg['recall']}")

print("\n-- compressor keeps content, not the greeting --")
from memgate.compress import compress
_dlg = "[Caroline] Hey Mel! Good to see you! How have you been? I finally " \
       "signed up for the LGBTQ support group on 7 May 2023."
_c = compress(_dlg, 14)
check("compression respects the token budget", count_tokens(_c) <= 14,
      f"{count_tokens(_c)} tokens: {_c!r}")
# The whole point: head-truncation would keep "Hey Mel! Good to see you! How"
# and lose the date, which IS the answer to "when did Caroline go...".
check("keeps the date over the greeting", "2023" in _c, _c)
check("drops phatic filler", "Hey" not in _c, _c)
check("keeps the speaker tag", _c.startswith("[Caroline]"), _c)

_fact = "[user] My budget for the analytics dashboard is 40000 rupees."
_cf = compress(_fact, 12)
check("keeps numbers and entities",
      "40000" in _cf and "analytics" in _cf, _cf)
check("short text passes through untouched",
      compress("[user] Deadline is March 15.", 99) == "[user] Deadline is March 15.")
check("compression preserves word order",
      [w for w in _cf.split() if w in _fact.split()] ==
      [w for w in _fact.split() if w in _cf.split()], _cf)

print("\n-- consolidation re-summarises rather than discards --")
# Tier 2 used to be a death chamber: routing only sends items scoring below
# tau_fact (0.45) there, but survival required clearing promote_threshold (0.5),
# so nothing could ever get out. Consolidation now merges the remainder into
# coarser gists instead of dropping them.
_ct, _cq = build_dataset(n_sessions=10, turns_per_session=40, n_facts=20)
_cp = MemGatePolicy(budget=2048)
for t in _ct:
    _cp.observe(t)
_merged = [i for i in _cp.store.working if i.fragments]
check("consolidation produced merged gists", _cp.store.merges > 0,
      f"merges={_cp.store.merges}")
check("merged gists survive in tier 2", len(_merged) > 0)

# The honest-accounting invariant. A merged gist may only claim evidence ids
# whose TEXT it still physically contains -- recursive merging otherwise
# accumulates ids while truncating the text away, and one item ended up
# claiming 373 turns in 221 tokens, reporting a write-path ceiling it could not
# support. A fragment is kept whole with its ids, or dropped with them.
_bad = [i for i in _merged
        if set(i.covered_ids) != {x for ids, _ in i.fragments for x in ids}]
check("merged gist claims only ids it still holds text for", not _bad,
      f"{len(_bad)} items claim ids with no surviving fragment")

# and every fragment's text must actually be in the item's text
_lost = [i for i in _merged if any(t not in i.text for _, t in i.fragments)]
check("every claimed fragment is present in the text", not _lost,
      f"{len(_lost)} items lost fragment text")

# granularity: merging must not collapse tier 2 into one atomic block, which
# either fits the tier-2 share whole or is skipped whole (cost -1.2 pts @2048)
_toks = [count_tokens(i.text) for i in _merged]
check("merged gists stay chunked, not one block",
      not _toks or max(_toks) <= _cp.store.chunk_tokens * 2,
      f"largest merged item is {max(_toks) if _toks else 0} tokens")

print("\n-- supersession --")
# Staleness handling: "Actually, we pushed the bakery module deadline to April 2"
# must retire the record it corrects, found by SEMANTIC match (the fact_id
# version was an oracle -- it was handed the record to invalidate).
_sup_turns, _ = build_dataset()
_upd = [t for t in _sup_turns if t.text.startswith("Actually,")]
check("dataset plants exactly one update", len(_upd) == 1, f"got {len(_upd)}")

_p = MemGatePolicy()
for t in _sup_turns:
    _p.observe(t)
_dead = [i for i in _p.store.long if not i.is_active]

# Invariant that must hold under ANY embedder: one update retires at most one
# record. Superseding every match above the bar once cascaded through the
# near-identical decision facts and cost ~10 recall points.
check("an update retires at most one record", len(_dead) <= 1,
      f"retired {len(_dead)}")

# Correctness depends on embedding quality, so it is asserted only where it is
# achievable. With hashing bag-of-words the nearest competitor outranks the true
# target (0.583 vs 0.500), so the ranking is inverted and this cannot pass --
# which is precisely the point of keeping the ablation.
_is_minilm = backend_info()["embedder"].startswith("all-MiniLM")
if _is_minilm:
    check("supersession retires the CORRECT record",
          len(_dead) == 1 and _dead[0].fact_id == _upd[0].fact_id,
          f"retired {[d.fact_id for d in _dead]}, expected {_upd[0].fact_id}")
else:
    print("  SKIP  supersession correctness (needs MiniLM; hashing inverts the "
          "ranking, 0.583 competitor vs 0.500 true target)")

print("\n-- no ground-truth leakage --")
# `fact_id` and `is_filler` are EVALUATION LABELS (types.py). If any routing,
# promotion, dedupe or supersession decision reads one, the reported recall is
# partly the answer key rather than the policy. Pin it: strip the labels and the
# store must come out bit-identical.
#
# This is not hypothetical. Promotion on `it.fact_id` in consolidation was worth
# +45 to +47.5 points here before it was removed.
from memgate.types import Turn


def _tier_fingerprint(ts, **kw):
    p = MemGatePolicy(**kw)
    for t in ts:
        p.observe(t)
    return ([i.text for i in p.store.short],
            [i.text for i in p.store.working],
            [(i.text, i.is_active) for i in p.store.long])


blind = [Turn(t.session, t.index, t.speaker, t.text) for t in turns]
check("routing is invariant to ground-truth labels",
      _tier_fingerprint(turns) == _tier_fingerprint(blind),
      "a policy decision is reading fact_id/is_filler")

# The storage budget added two more decision points that could read the answer
# key -- which item to EVICT and which to DEMOTE -- so the same invariant has to
# hold with the cap engaged, in both write modes. Eviction ranks on `utility`
# and recency only (store._forget_key).
for _mode in ("threshold", "adaptive"):
    check(f"eviction under a storage budget is label-blind ({_mode})",
          _tier_fingerprint(turns, store_budget=600, write_mode=_mode)
          == _tier_fingerprint(blind, store_budget=600, write_mode=_mode),
          "an eviction or demotion decision is reading fact_id/is_filler")

print("\n-- storage budget --")
# §22.3: the experiment capped the CONTEXT but never the STORE, so "keep
# everything and retrieve" was charged nothing for holding the whole
# conversation and forgetting could only ever lose. These pin the cap itself.
from memgate.policies import (SlidingWindowPolicy, RAGPolicy,
                              MemGateAdaptivePolicy)

for _name, _F in [("P0", SlidingWindowPolicy), ("RAG", RAGPolicy),
                  ("P1", MemGatePolicy), ("P1-A", MemGateAdaptivePolicy)]:
    _worst = []
    for _S in (256, 512, 1024, 2048):
        _p = _F(budget=1024, store_budget=_S)
        for _t in turns:
            _p.observe(_t)
        _worst.append((_S, _p.stored_tokens()))
    check(f"{_name} never exceeds its storage budget",
          all(v <= _S for _S, v in _worst),
          f"held more than allowed: {_worst}")

# A storage budget must actually BITE, or the sweep is measuring nothing. The
# unconstrained store holds the whole conversation; a capped one cannot.
_free = MemGateAdaptivePolicy(budget=1024, store_budget=None)
_tight = MemGateAdaptivePolicy(budget=1024, store_budget=512)
for _t in turns:
    _free.observe(_t)
    _tight.observe(_t)
check("a tight storage budget forces real loss",
      _tight.stored_tokens() < _free.stored_tokens(),
      f"tight={_tight.stored_tokens()} free={_free.stored_tokens()}")

# Sub-threshold items are dropped at ingest in threshold mode. With a storage
# budget that is waste while the store still has room, so `fill_store` keeps
# them instead and lets eviction decide later. It never fires on LoCoMo (whose
# turns all clear tau_low) but does on the synthetic set, so it is pinned here.
_drop = MemGatePolicy(budget=800, store_budget=1200, fill_store=None)
_fill = MemGatePolicy(budget=800, store_budget=1200, fill_store="long")
for _t in turns:
    _drop.observe(_t)
    _fill.observe(_t)
check("fill_store keeps sub-threshold turns while the store has room",
      _fill.routed["filled"] > 0 and _drop.routed["dropped"] > 0
      and _fill.stored_tokens() > _drop.stored_tokens(),
      f"fill={_fill.routed} {_fill.stored_tokens()} "
      f"drop={_drop.routed} {_drop.stored_tokens()}")

# Adaptive mode's whole claim is that the compression RATE follows storage
# pressure. With room to spare nothing should be demoted at all.
check("adaptive mode compresses nothing when storage is free",
      _free.store.stats()["demoted"] == 0,
      f"demoted {_free.store.stats()['demoted']} with an unbounded store")
check("adaptive mode compresses under pressure",
      _tight.store.stats()["demoted"] > 0,
      "storage pressure did not trigger any demotion")

print("\n-- learned scorer --")
# P3-learned replaces P1's hand-chosen weights with a fitted model. It is
# trained on OTHER conversations' evidence labels, but at inference it must see
# text and speaker only -- otherwise it is the §13 leak with extra steps. The
# invariance test therefore has to cover this scorer too, not just P1.
try:
    import numpy as _np
    from sklearn.linear_model import LogisticRegression as _LR
    from memgate.scoring import LearnedScorer, features as _feat
    from memgate.utils import embed as _emb

    _X = _np.array([_feat(t.text, t.speaker) for t in turns], dtype=float)
    _y = _np.array([1 if t.fact_id else 0 for t in turns])
    _m = _LR(max_iter=1000, class_weight="balanced").fit(_X, _y)
    _sc = LearnedScorer(model=_m, use_embedding=False)

    check("learned scorer is label-blind at inference",
          _tier_fingerprint(turns, scorer=_sc, store_budget=600,
                            write_mode="adaptive")
          == _tier_fingerprint(blind, scorer=_sc, store_budget=600,
                               write_mode="adaptive"),
          "the learned scorer path is reading fact_id/is_filler")

    _a, _l = _sc.score("We moved the bakery deadline to April 2.")
    check("learned scorer returns a bounded utility and a label",
          0.0 <= _a <= 1.0 and isinstance(_l, str))
except ImportError:
    print("  SKIP  sklearn not installed")

print("\n-- offline reproducibility --")
# A run that can silently fetch a model is a run whose inputs are not pinned,
# and this project has already lost a results table to a half-downloaded
# MiniLM (§16.3). Two things must hold: importing the package pins the loaders
# offline, and the embedder still resolves to MiniLM once it is pinned. The
# second is the one that matters -- if offline mode made MiniLM unloadable the
# fallback would quietly relabel every result as hashing bag-of-words.
import os as _os
from memgate.utils import backend_info as _bi
check("import pins the HF loaders offline",
      _os.environ.get("HF_HUB_OFFLINE") == "1",
      "set MEMGATE_ALLOW_HUB=1 only when deliberately adding a model")
check("MiniLM still resolves with the Hub pinned off",
      _bi()["embedder"].startswith("all-MiniLM"),
      f"fell back to {_bi()['embedder']} -- results would be mislabelled")

print("\n-- reproducibility --")
a = run_policy(MemGatePolicy(), turns, questions, 800)
b = run_policy(MemGatePolicy(), turns, questions, 800)
check("runs are deterministic", a["recall"] == b["recall"] and
      a["avg_tokens"] == b["avg_tokens"])

print(f"\n{'='*46}\n  {passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)

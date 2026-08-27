"""Policies under comparison.

All expose the same two methods so the harness can treat them identically:
    observe(turn)                 -> ingest one conversation turn
    build_context(query, budget)  -> ContextBundle within the token budget

Policies:
    FullContextPolicy  -- upper bound on recall, worst cost
    OraclePolicy       -- upper bound: only the gold evidence (diagnostic)
    SlidingWindowPolicy(P0) -- recency only
    MemGatePolicy(P1)  -- three-tier store + Adaptive Memory Decision Engine
"""
from typing import List
from .types import (Turn, MemoryItem, ContextBundle,
                    SHORT_TERM, WORKING, LONG_TERM)
from .utils import count_tokens, embed, cosine
from .compress import compress
from .store import ThreeTierStore
from .scoring import HeuristicScorer


def _pack(items: List[MemoryItem], budget: int) -> ContextBundle:
    """Greedily pack items into the budget, in the order given."""
    kept, used = [], 0
    for it in items:
        c = count_tokens(it.text)
        if used + c > budget:
            continue
        kept.append(it)
        used += c
    text = "\n".join(i.text for i in kept)
    return ContextBundle(text=text, items=kept, tokens=used, budget=budget)


def _by_density(items: List[MemoryItem]) -> List[MemoryItem]:
    """Order Tier 2 by conversation-per-token, densest first.

    A merged gist holding 38 turns in 543 tokens carries far more of the
    conversation per token than a single 19-token turn, but strict recency
    ordering puts the oldest merged item last, so it never fitted the Tier 2
    share and was always skipped. Packing by density is the rate-distortion
    assembly this project argues for, applied to the read path.

    Density uses `len(fragments)` -- how many turns were folded in -- which is a
    structural property of the store. It must NOT use `covered_ids`: those are
    evidence ids, i.e. the evaluation labels, and ranking by them would smuggle
    the answer key back into the policy (the invariance test would catch it).
    Recency breaks ties, so unmerged items keep their previous ordering.
    """
    order = {id(it): n for n, it in enumerate(items)}
    return sorted(
        items,
        key=lambda i: (-(max(1, len(i.fragments)) / max(1, count_tokens(i.text))),
                       -order[id(i)]),
    )


def _item_cost(it: MemoryItem, cost_mode: str, vector_bytes: int = 1536) -> int:
    """Storage cost of one item, in the budget's unit. See store._cost.

    Under "bytes" an item pays for its embedding vector as well as its text.
    A policy that does not retrieve (P0) stores no vectors and honestly pays
    nothing for them -- recency needs no index.
    """
    if cost_mode == "bytes":
        vec = vector_bytes if it.embedding is not None else 0
        return len(it.text.encode("utf-8")) + vec
    return count_tokens(it.text)


def _as_item(turn: Turn) -> MemoryItem:
    return MemoryItem(
        text=f"[{turn.speaker}] {turn.text}",
        tier=SHORT_TERM, kind="turn",
        session=turn.session, turn_index=turn.index,
        fact_id=turn.fact_id,
    )


# --------------------------------------------------------------- baselines
class FullContextPolicy:
    """Send the entire history every turn. Accuracy ceiling, cost ceiling."""
    name = "Full-context"
    is_baseline = True

    def __init__(self, **kw):
        self.items: List[MemoryItem] = []

    def observe(self, turn: Turn):
        self.items.append(_as_item(turn))

    def build_context(self, query: str, budget: int) -> ContextBundle:
        # Deliberately ignores the budget: that is the point of this baseline.
        text = "\n".join(i.text for i in self.items)
        return ContextBundle(text=text, items=list(self.items),
                             tokens=count_tokens(text), budget=budget)

    def stored_tokens(self) -> int:
        # Deliberately unbounded, like the context: this is the reference row
        # that shows what the whole conversation costs to hold verbatim.
        return sum(count_tokens(i.text) for i in self.items)

    def stats(self):
        return {"stored": len(self.items), "stored_tokens": self.stored_tokens()}


class OraclePolicy:
    """Diagnostic upper bound: perfect selection of only the needed evidence."""
    name = "Oracle"
    is_baseline = True

    def __init__(self, **kw):
        self.by_fact = {}

    def observe(self, turn: Turn):
        if turn.fact_id:
            self.by_fact.setdefault(turn.fact_id, []).append(_as_item(turn))

    def build_context(self, query: str, budget: int, gold_ids=None) -> ContextBundle:
        # Multi-evidence aware: a LoCoMo multi-hop question needs every cited
        # turn, so the bound must supply all of them, not just one.
        items = []
        for gid in (gold_ids or ()):
            items.extend(self.by_fact.get(gid, [])[-1:])
        return _pack(items, budget)

    def stored_tokens(self) -> int:
        return sum(count_tokens(i.text)
                   for v in self.by_fact.values() for i in v)

    def stats(self):
        return {"facts": len(self.by_fact), "stored_tokens": self.stored_tokens()}


class SlidingWindowPolicy:
    """P0 — spend the entire budget on the most recent turns.

    Fairness note: the window is budget-driven, not a fixed turn count. P0 is
    allowed to fill exactly the same token budget as MemGate, so the only
    difference between them is *which* turns are chosen, never how many tokens
    they may spend. A fixed small window would make this a strawman.
    """
    name = "P0 sliding-window"
    is_baseline = False

    def __init__(self, max_retained: int = 2000, store_budget: int = None,
                 cost_mode: str = "tokens", vector_bytes: int = 1536, **kw):
        self.max_retained = max_retained
        self.store_budget = store_budget
        self.cost_mode = cost_mode
        self.vector_bytes = vector_bytes
        self.items: List[MemoryItem] = []
        self._tokens = 0
        self.dropped = 0

    def observe(self, turn: Turn):
        it = _as_item(turn)
        self.items.append(it)
        self._tokens += self._cost(it)
        if len(self.items) > self.max_retained:
            self._tokens -= self._cost(self.items.pop(0))
        # Storage budget: retain the most recent S tokens. For a recency policy
        # the cap IS the policy -- there is nothing else it could drop -- which
        # is what makes it the honest floor for the storage sweep.
        while self.store_budget is not None and self._tokens > self.store_budget \
                and len(self.items) > 1:
            self._tokens -= self._cost(self.items.pop(0))
            self.dropped += 1

    def _cost(self, it):
        return _item_cost(it, self.cost_mode, self.vector_bytes)

    def build_context(self, query: str, budget: int) -> ContextBundle:
        # most recent first, packed until the budget is exhausted
        return _pack(list(reversed(self.items)), budget)

    def stored_tokens(self) -> int:
        return sum(count_tokens(i.text) for i in self.items)

    def stored_cost(self) -> int:
        return sum(self._cost(i) for i in self.items)

    def stats(self):
        return {"held": len(self.items), "dropped": self.dropped,
                "stored_tokens": self.stored_tokens()}


class RAGPolicy:
    """Store every turn verbatim, retrieve top-k. No compression, no tiering.

    §22.2 found this configuration -- reachable in MemGate by setting
    `tau_fact = 0`, so that nothing is ever compressed, merged or dropped --
    scored 59.2% strict on LoCoMo against the fully tuned system's 14.4%, and
    landed within one point of the best tuned variant. Retrieval contributed
    ~51 of those points; merging and compression together contributed ~1.

    That result is why the storage budget exists, so the configuration is
    promoted here from a footnote to a named baseline: it is the thing
    compression actually has to beat.

    Under a storage cap it evicts FIFO. It has no scorer, and that is the
    point -- it is the no-policy control. The three retention regimes then
    isolate one variable each at a fixed store size S:

        P0  vs  RAG   same retention, different READ path
                      (recency vs embedding retrieval)
        RAG vs  P1    same storage budget, different WRITE path
                      (verbatim-and-forget vs compress-and-keep)

    If compression is worth anything, P1 holds more of the conversation per
    stored token than RAG and overtakes it as S is squeezed. If it never
    overtakes, that is a real negative result about tiered memory rather than
    an artefact of unbounded storage.
    """
    name = "RAG store-all"
    is_baseline = False

    def __init__(self, store_budget: int = None, retrieve_k: int = 60,
                 split=(0.25, 0.75), budget: int = None,
                 cost_mode: str = "tokens", vector_bytes: int = 1536, **kw):
        self.store_budget = store_budget
        self.cost_mode = cost_mode
        self.vector_bytes = vector_bytes
        self.retrieve_k = retrieve_k
        self.split = split
        self.items: List[MemoryItem] = []
        self._tokens = 0
        self.dropped = 0

    def observe(self, turn: Turn):
        it = _as_item(turn)
        it.embedding = embed(it.text)
        self.items.append(it)
        self._tokens += self._cost(it)
        while self.store_budget is not None and self._tokens > self.store_budget \
                and len(self.items) > 1:
            self._tokens -= self._cost(self.items.pop(0))
            self.dropped += 1

    def _cost(self, it):
        return _item_cost(it, self.cost_mode, self.vector_bytes)

    def build_context(self, query: str, budget: int) -> ContextBundle:
        # Same read-path shape as MemGate -- a recent share plus a retrieved
        # share -- so the only difference between them is what the write path
        # chose to keep. Recent turns are excluded from the retrieval pool so
        # the two shares never pay twice for the same turn.
        b_recent = int(budget * self.split[0])
        recent = _pack(list(reversed(self.items)), b_recent)
        seen = {id(i) for i in recent.items}
        q = embed(query)
        scored = [(cosine(q, i.embedding), i)
                  for i in self.items if id(i) not in seen]
        scored.sort(key=lambda x: -x[0])
        hits = [i for sc, i in scored[:self.retrieve_k] if sc > 0]
        for i in hits:
            i.access_count += 1
        long_ = _pack(hits, budget - recent.tokens)
        items = recent.items + long_.items
        text = "\n".join(i.text for i in items)
        return ContextBundle(text=text, items=items,
                             tokens=recent.tokens + long_.tokens, budget=budget)

    def stored_tokens(self) -> int:
        return sum(count_tokens(i.text) for i in self.items)

    def stored_cost(self) -> int:
        return sum(self._cost(i) for i in self.items)

    def stats(self):
        return {"held": len(self.items), "dropped": self.dropped,
                "stored_tokens": self.stored_tokens(),
                "stored_cost": self.stored_cost()}


# --------------------------------------------------------------- MemGate
class MemGatePolicy:
    """P1 — three-tier store driven by the Adaptive Memory Decision Engine.

    Routing happens on eviction from the short-term buffer:
        u >= tau_fact                -> extract to Tier 3 (long-term)
        tau_low <= u < tau_fact      -> summarise into Tier 2 (working)
        u <  tau_low                 -> drop from context
    """
    name = "P1 MemGate"
    is_baseline = False

    def __init__(self, scorer=None, short_capacity: int = 6,
                 tau_fact: float = 0.45, tau_low: float = 0.15,
                 working_capacity_tokens: int = None,
                 split=(0.25, 0.25, 0.50), retrieve_k: int = 60,
                 promote_threshold: float = 0.5,
                 supersede_threshold: float = 0.60,
                 budget: int = None,
                 # --- ablation switches (off = the behaviour we replaced) ---
                 merge_on_consolidate: bool = True,
                 informative_compress: bool = True,
                 pack_by_density: bool = True,
                 use_retrieval: bool = True,
                 use_working: bool = True,
                 supersede: bool = True,
                 store_budget: int = None,
                 demote_on_pressure: bool = True,
                 scored_eviction: bool = True,
                 fill_store: str = "long",
                 write_mode: str = "threshold",
                 retrieve_working: bool = None,
                 cost_mode: str = "tokens", vector_bytes: int = 1536, **kw):
        # Size the working tier from the token budget rather than pinning it at
        # a constant. A fixed 400-token Tier 2 meant MemGate could not fill a
        # 4096-token budget however well it scored (it used 1336), so the
        # comparison against P0 -- which is allowed to spend the whole budget --
        # was no longer apples-to-apples. This is the same strawman that was
        # fixed for P0 in Step 0, pointed the other way.
        #
        # The store may hold more than one context-window's worth, since only
        # the slice that survives retrieval is actually spent.
        if working_capacity_tokens is None:
            working_capacity_tokens = (max(400, int(budget * split[1] * 2))
                                       if budget else 400)
        # Under a storage cap, Tier 2 cannot be sized from the context budget
        # alone -- a 4096-token context share is meaningless when the entire
        # store is capped at 1024. Take the tighter of the two.
        if store_budget is not None:
            working_capacity_tokens = min(working_capacity_tokens,
                                          max(64, int(store_budget * split[1])))
        # In adaptive mode Tier 2 is a demotion destination under storage
        # pressure, not a share of the context window, so it is sized from the
        # STORE budget. Sized from the context budget it stayed pinned at 1024
        # tokens however large the store was, and MemGate flatlined at 2148
        # stored tokens on a 16384-token allowance.
        if write_mode == "adaptive" and store_budget is not None:
            working_capacity_tokens = max(64, int(store_budget * 0.5))
        self.scorer = scorer or HeuristicScorer()
        self.store = ThreeTierStore(short_capacity, working_capacity_tokens,
                                    promote_threshold=promote_threshold,
                                    supersede_threshold=supersede_threshold,
                                    merge_on_consolidate=merge_on_consolidate,
                                    informative_compress=informative_compress,
                                    supersede=supersede,
                                    store_budget=store_budget,
                                    demote_on_pressure=demote_on_pressure,
                                    scored_eviction=scored_eviction,
                                    cost_mode=cost_mode,
                                    vector_bytes=vector_bytes)
        self.informative_compress = informative_compress
        self.pack_by_density = pack_by_density
        self.use_retrieval = use_retrieval
        self.use_working = use_working
        self.tau_fact = tau_fact
        self.tau_low = tau_low
        # With a storage cap, dropping a sub-threshold item while the store
        # still has room is pure waste -- the same strawman as §13.5, pointed
        # the other way: there MemGate could not fill the CONTEXT it was given,
        # here it cannot fill the STORE. It flatlined at 2148 stored tokens on
        # a 16384-token allowance, so the S >= 4096 comparisons were measuring
        # MemGate's refusal to use its budget rather than the value of
        # compression.
        #
        # Under a cap the thresholds should ORDER the queue, not force a drop:
        # the budget is the constraint, and `enforce_store_budget` already
        # decides what actually goes when pressure arrives.
        #     "long"    keep verbatim, retrievable; compress only under pressure
        #     "working" keep as a compressed gist immediately
        #     None      drop outright (the pre-storage-budget behaviour)
        # Only ever active when a store budget is set -- with an unbounded store
        # "fill it" would mean "keep everything", which is the RAG baseline, and
        # would silently change every previously reported number.
        self.fill_store = fill_store if store_budget is not None else None
        self.store_budget = store_budget
        # write_mode -- WHERE the keep/compress/forget decision is taken.
        #
        #   "threshold" (default, unchanged)
        #       tau_fact / tau_low decide at INGEST. An item scoring below
        #       tau_low is compressed or dropped there and then, whatever the
        #       storage situation. Every result in §15-§22 was produced this way
        #       and stays reproducible.
        #
        #   "adaptive"
        #       Everything is stored VERBATIM while the store has room, and the
        #       scores only ORDER the eviction queue; compression happens on
        #       demotion, when the budget actually binds. The compression rate
        #       therefore adapts to storage pressure instead of being fixed in
        #       advance -- which is the rate-distortion claim the project makes,
        #       and the form §22.3 asked for.
        #
        # Threshold mode compresses every turn to a ~14-word gist on the way in,
        # so it cannot store more than ~4735 tokens of a 19K conversation no
        # matter how large the budget. That is why it lost to plain retrieval at
        # every store size above 2048.
        self.write_mode = write_mode
        # Demoted gists live in Tier 2, so in adaptive mode Tier 2 MUST be
        # retrievable or the demotion destination is a black hole.
        self.retrieve_working = (retrieve_working if retrieve_working is not None
                                 else write_mode == "adaptive")
        if write_mode == "adaptive":
            self.name = "P1-A MemGate adaptive"
        self.split = split          # (short, working, long) budget shares
        self.retrieve_k = retrieve_k
        self.routed = {"long": 0, "working": 0, "dropped": 0, "filled": 0}

    # ---------------------------------------------------------- write path
    def observe(self, turn: Turn):
        item = _as_item(turn)
        evicted = self.store.add_short(item)
        if evicted is not None:
            self._route(evicted, turn.speaker)

    def _route(self, item: MemoryItem, speaker: str = "user"):
        u, label = self.scorer.score(item.text, speaker)
        item.utility, item.type_label = u, label

        if self.write_mode == "adaptive":
            # Keep it whole and retrievable. `u` is not discarded -- it rides on
            # the item and sets its place in the eviction queue (_forget_key),
            # so the scorer still decides what is forgotten first, just later
            # and against a real budget rather than a guessed threshold.
            self.store.add_long(item.text, item, kind="turn")
            self.routed["long"] += 1
            return

        if u >= self.tau_fact:
            self.store.add_long(item.text, item, kind="fact")
            self.routed["long"] += 1
            # a durable fact also contributes its gist to working memory
            if u >= self.tau_fact + 0.2:
                self.store.add_working(self._summarise(item.text), item)
        elif u >= self.tau_low:
            self.store.add_working(self._summarise(item.text), item)

            self.routed["working"] += 1
        elif self.fill_store == "long":
            # Verbatim while there is headroom. Demotion to a gist happens on
            # pressure, so the compression RATE adapts to how tight the store
            # is instead of being fixed in advance -- which is the
            # rate-distortion claim the project is actually making.
            self.store.add_long(item.text, item, kind="turn")
            self.routed["filled"] += 1
        elif self.fill_store == "working":
            self.store.add_working(self._summarise(item.text), item)
            self.routed["filled"] += 1
        else:
            self.routed["dropped"] += 1
            self.store.dropped += 1

    def _summarise(self, text: str, max_words: int = 14) -> str:
        """Compress a turn to its most informative words.

        Was head-truncation (`" ".join(w[:max_words])`), which on dialogue
        spends the whole budget on the greeting and drops the dates and names
        that are actually the answers. See compress.py.
        `informative_compress=False` restores head-truncation as the ablation.

        SWAP-IN (Phase 2): base-LLM abstractive summarisation, which must beat
        this extractive baseline to earn its cost.
        """
        if not self.informative_compress:
            w = text.split()
            return text if len(w) <= max_words else " ".join(w[:max_words]) + " ..."
        return compress(text, max_tokens=max(1, int(max_words * 1.4)))

    # ---------------------------------------------------------- read path
    def build_context(self, query: str, budget: int) -> ContextBundle:
        b_short = int(budget * self.split[0])
        b_work = int(budget * self.split[1])
        b_long = budget - b_short - b_work

        short = _pack(list(reversed(self.store.short)), b_short)

        if self.use_working:
            order = (_by_density(self.store.working) if self.pack_by_density
                     else list(reversed(self.store.working)))
            work = _pack(order, b_work)
        else:
            work = _pack([], b_work)

        if self.retrieve_working:
            # One relevance-ranked pool over Tier 2 + Tier 3 rather than a
            # positional Tier 2 share plus a retrieved Tier 3 share. Splitting
            # them fixes in advance how much of the context each tier may use,
            # which is a guess; ranking them together lets the QUESTION decide.
            retrieved = (self.store.retrieve(query, k=self.retrieve_k,
                                             include_working=True)
                         if self.use_retrieval else [])
            rest = _pack(retrieved, budget - short.tokens)
            items = short.items + rest.items
            text = "\n".join(i.text for i in items)
            return ContextBundle(text=text, items=items,
                                 tokens=short.tokens + rest.tokens, budget=budget)

        retrieved = (self.store.retrieve(query, k=self.retrieve_k)
                     if self.use_retrieval else [])
        long_ = _pack(retrieved, b_long)

        items = short.items + work.items + long_.items
        text = "\n".join(i.text for i in items)
        return ContextBundle(text=text, items=items,
                             tokens=short.tokens + work.tokens + long_.tokens,
                             budget=budget)

    def stored_tokens(self) -> int:
        return self.store.stored_tokens()

    def stored_cost(self) -> int:
        return self.store.stored_cost()

    def stats(self):
        # Namespaced: store.stats() and `routed` both define long/working/dropped,
        # so a plain update() silently replaced the tier SIZES with the routing
        # COUNTERS (working showed 361 items against a 400-token cap).
        s = self.store.stats()
        s.update({f"routed_{k}": v for k, v in self.routed.items()})
        return s


def MemGateAdaptivePolicy(**kw):
    """MemGate with the decision moved from ingest thresholds to eviction.

    Stores verbatim while the store has room and DEMOTES to a compressed gist
    under pressure, so the compression rate follows the storage constraint.
    """
    kw.setdefault("write_mode", "adaptive")
    return MemGatePolicy(**kw)


def MemGateSelectPolicy(**kw):
    """Selection only: keep the high-utility turns verbatim, forget the rest.

    Identical to the adaptive policy except that an evicted item is DROPPED
    rather than compressed into a gist. It is therefore the same storage
    budget, the same verbatim fidelity and the same retrieval as RAG store-all,
    differing in one thing only -- WHICH turns are forgotten when the cap
    binds. RAG evicts FIFO; this evicts by scored utility. The gap between them
    is the decision policy and nothing else, which is the isolation the whole
    project is aimed at.

    It exists because compression measurably loses. On answer-presence -- the
    metric that checks whether the answer TEXT survived, not merely a pointer
    to the turn that held it -- dropping beats demoting at every storage budget
    tested (36.8% vs 30.1% at S=4096), while demotion wins on evidence recall
    by keeping ids whose text it has thrown away. That is the §19 failure mode,
    and it is why both metrics are always reported together.

    The rate-distortion reading: at LoCoMo dialogue scale the optimum sits at
    the vertex -- a subset at full fidelity beats everything at reduced
    fidelity. Compression only starts to pay when the store is squeezed far
    enough that even the selected subset will not fit verbatim.
    """
    kw.setdefault("write_mode", "adaptive")
    kw.setdefault("demote_on_pressure", False)
    return MemGatePolicy(**kw)


POLICIES = {
    "full": FullContextPolicy,
    "oracle": OraclePolicy,
    "p0": SlidingWindowPolicy,
    "rag": RAGPolicy,
    "p1": MemGatePolicy,
    "p1a": MemGateAdaptivePolicy,
    "p1s": MemGateSelectPolicy,
}

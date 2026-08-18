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
from .utils import count_tokens, embed
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

    def stats(self):
        return {"stored": len(self.items)}


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

    def stats(self):
        return {"facts": len(self.by_fact)}


class SlidingWindowPolicy:
    """P0 — spend the entire budget on the most recent turns.

    Fairness note: the window is budget-driven, not a fixed turn count. P0 is
    allowed to fill exactly the same token budget as MemGate, so the only
    difference between them is *which* turns are chosen, never how many tokens
    they may spend. A fixed small window would make this a strawman.
    """
    name = "P0 sliding-window"
    is_baseline = False

    def __init__(self, max_retained: int = 2000, **kw):
        self.max_retained = max_retained
        self.items: List[MemoryItem] = []

    def observe(self, turn: Turn):
        self.items.append(_as_item(turn))
        if len(self.items) > self.max_retained:
            self.items.pop(0)

    def build_context(self, query: str, budget: int) -> ContextBundle:
        # most recent first, packed until the budget is exhausted
        return _pack(list(reversed(self.items)), budget)

    def stats(self):
        return {"held": len(self.items)}


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
                 supersede: bool = True, **kw):
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
        self.scorer = scorer or HeuristicScorer()
        self.store = ThreeTierStore(short_capacity, working_capacity_tokens,
                                    promote_threshold=promote_threshold,
                                    supersede_threshold=supersede_threshold,
                                    merge_on_consolidate=merge_on_consolidate,
                                    informative_compress=informative_compress,
                                    supersede=supersede)
        self.informative_compress = informative_compress
        self.pack_by_density = pack_by_density
        self.use_retrieval = use_retrieval
        self.use_working = use_working
        self.tau_fact = tau_fact
        self.tau_low = tau_low
        self.split = split          # (short, working, long) budget shares
        self.retrieve_k = retrieve_k
        self.routed = {"long": 0, "working": 0, "dropped": 0}

    # ---------------------------------------------------------- write path
    def observe(self, turn: Turn):
        item = _as_item(turn)
        evicted = self.store.add_short(item)
        if evicted is not None:
            self._route(evicted, turn.speaker)

    def _route(self, item: MemoryItem, speaker: str = "user"):
        u, label = self.scorer.score(item.text, speaker)
        item.utility, item.type_label = u, label

        if u >= self.tau_fact:
            self.store.add_long(item.text, item, kind="fact")
            self.routed["long"] += 1
            # a durable fact also contributes its gist to working memory
            if u >= self.tau_fact + 0.2:
                self.store.add_working(self._summarise(item.text), item)
        elif u >= self.tau_low:
            self.store.add_working(self._summarise(item.text), item)

            self.routed["working"] += 1
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

        retrieved = (self.store.retrieve(query, k=self.retrieve_k)
                     if self.use_retrieval else [])
        long_ = _pack(retrieved, b_long)

        items = short.items + work.items + long_.items
        text = "\n".join(i.text for i in items)
        return ContextBundle(text=text, items=items,
                             tokens=short.tokens + work.tokens + long_.tokens,
                             budget=budget)

    def stats(self):
        # Namespaced: store.stats() and `routed` both define long/working/dropped,
        # so a plain update() silently replaced the tier SIZES with the routing
        # COUNTERS (working showed 361 items against a 400-token cap).
        s = self.store.stats()
        s.update({f"routed_{k}": v for k, v in self.routed.items()})
        return s


POLICIES = {
    "full": FullContextPolicy,
    "oracle": OraclePolicy,
    "p0": SlidingWindowPolicy,
    "p1": MemGatePolicy,
}

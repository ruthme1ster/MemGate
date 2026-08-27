"""The three-tier memory store.

Tier 1 short-term : raw recent turns, FIFO, criterion = WHEN
Tier 2 working    : compressed summaries, criterion = GIST
Tier 3 long-term  : atomic facts + embeddings, criterion = WHAT

Tiers are a lifecycle, not a partition: everything enters Tier 1 and the
routing decision happens on eviction.

INVARIANT — no ground-truth leakage
-----------------------------------
`MemoryItem.fact_id` is an *evaluation label* (see types.py): the harness scores
a policy by checking whether the gold fact_id reached the assembled context.
Therefore no routing, promotion, dedupe or supersession decision in this module
may read it. It is carried on the item purely so the harness can score, the way
a sample id rides along with a feature vector without being a feature.

An earlier version violated this in two places -- promotion on `it.fact_id` in
consolidation, and supersession keyed on `source.fact_id`. That single promotion
clause was worth +45 to +47.5 recall points on the synthetic set, i.e. most of
the apparent gain was the answer key leaking into the policy. `test_memgate.py`
now pins this with an invariance test: routing must be bit-identical when every
fact_id is stripped.
"""
from typing import List, Optional
from .types import MemoryItem, SHORT_TERM, WORKING, LONG_TERM  # noqa: F401
from .utils import embed, cosine, count_tokens
from .compress import informative_head


class ThreeTierStore:
    def __init__(self, short_capacity: int = 6, working_capacity_tokens: int = 400,
                 promote_threshold: float = 0.5,
                 supersede_threshold: float = 0.60,
                 merge_head_words: int = 8, chunk_tokens: int = 64,
                 merge_on_consolidate: bool = True,
                 informative_compress: bool = True,
                 supersede: bool = True,
                 store_budget: Optional[int] = None,
                 demote_on_pressure: bool = True,
                 scored_eviction: bool = True,
                 cost_mode: str = "tokens", vector_bytes: int = 1536):
        self.short_capacity = short_capacity
        self.working_capacity_tokens = working_capacity_tokens
        self.promote_threshold = promote_threshold
        self.supersede_threshold = supersede_threshold
        self.merge_head_words = merge_head_words
        self.chunk_tokens = chunk_tokens
        # Ablation switches. Off = the behaviour this project replaced, so each
        # flag isolates exactly one change.
        self.merge_on_consolidate = merge_on_consolidate   # off = discard (§13.2)
        self.informative_compress = informative_compress   # off = head-truncate
        self.supersede = supersede
        # --- storage budget (§22.3) ---
        self.store_budget = store_budget
        self.demote_on_pressure = demote_on_pressure   # off = drop instead
        self.scored_eviction = scored_eviction         # off = FIFO ablation
        # What the storage budget is DENOMINATED in.
        #
        #   "tokens"  text only -- what the sweep in §23 charges for
        #   "bytes"   text bytes + one embedding vector per retained ITEM
        #
        # The second is not a refinement, it can invert the ranking. A LoCoMo
        # turn averages ~32 tokens (~130 bytes of text) but carries a 384-d
        # float32 vector at 1536 bytes, so under byte accounting the embedding
        # outweighs the text by roughly 12x and the binding cost becomes the
        # NUMBER OF ITEMS, not their length. A policy that merges many turns
        # into one gist pays one vector for all of them; a policy that keeps
        # turns whole pays one vector each. Token accounting cannot see that.
        self.cost_mode = cost_mode
        self.vector_bytes = vector_bytes
        self._enforcing = False
        self.evicted = 0
        self.demoted = 0
        self.reclaimed = 0
        self.merges = 0
        self.short: List[MemoryItem] = []
        self.working: List[MemoryItem] = []
        self.long: List[MemoryItem] = []
        self.dropped = 0
        self.consolidations = 0

    # ------------------------------------------------------------ tier 1
    def add_short(self, item: MemoryItem) -> Optional[MemoryItem]:
        """Append to the buffer. Returns an item if one was evicted."""
        item.tier = SHORT_TERM
        self.short.append(item)
        if len(self.short) > self.short_capacity:
            return self.short.pop(0)
        return None

    # ------------------------------------------------------------ tier 2
    def add_working(self, text: str, source: MemoryItem):
        it = MemoryItem(
            text=text, tier=WORKING, kind="summary",
            session=source.session, turn_index=source.turn_index,
            utility=source.utility, type_label=source.type_label,
            fact_id=source.fact_id, embedding=embed(text),
            source_ids=[source.id],
        )
        self.working.append(it)
        self._consolidate_if_needed()
        self.enforce_store_budget()

    def _consolidate_if_needed(self):
        """Tier 2 cannot grow forever: over budget, RE-SUMMARISE the oldest half.

        What happens to the oldest half:
          * `utility >= promote_threshold`  -> promoted to Tier 3 verbatim
          * everything else                 -> MERGED into one coarser gist that
                                               stays in Tier 2

        The merge is the point. The previous version *discarded* the remainder,
        which made Tier 2 a death chamber: routing only sends items scoring
        below `tau_fact` (0.45) here, yet survival required clearing
        `promote_threshold` (0.5), so nothing could ever get out. On LoCoMo that
        retained just 30 of 419 turns (7.2%) and capped recall at 9.4% no matter
        how good retrieval was.

        Merging is the rate-distortion move the project is actually about: pay
        fewer tokens for a lossier representation, rather than paying zero and
        losing the information entirely.

        Promotion is on `utility` alone -- never `fact_id`, which is the
        evaluation label (see the module docstring).
        """
        guard = 0
        while (sum(count_tokens(i.text) for i in self.working)
               > self.working_capacity_tokens and len(self.working) > 1):
            guard += 1
            if guard > 32:                      # cannot happen; never loop forever
                break
            self.consolidations += 1
            half = max(1, len(self.working) // 2)
            oldest, rest = self.working[:half], self.working[half:]

            keep = []
            for it in oldest:
                if it.utility >= self.promote_threshold:
                    self.add_long(it.text, it, kind="fact")
                elif self.merge_on_consolidate:
                    keep.append(it)
                else:
                    self.dropped += 1          # ablation: the old discard path

            # Merge into SEVERAL bounded gists, not one big block. A single
            # merged item is atomic at assembly time: it either fits the Tier 2
            # share whole or is skipped whole. Collapsing 27 packable items into
            # one 512-token block cost more in granularity than density won --
            # measurably, -1.2 strict recall at 2048. Chunking keeps both.
            merged = self._merge_chunked(
                keep, self.working_capacity_tokens // 2) if keep else []
            self.working = merged + rest

            # No progress (merge did not shrink anything) -> stop rather than spin.
            if len(merged) >= half and half == len(oldest):
                break

    def _merge_chunked(self, items: List[MemoryItem],
                       total_target: int) -> List[MemoryItem]:
        """Re-summarise into several gists, each at most `chunk_tokens`.

        Keeps the newest fragments up to `total_target` (older ones are dropped
        with their ids, as in _merge), then packs them into bounded chunks so
        the assembler can still choose at a useful granularity.
        """
        frags = []
        for it in items:
            if it.fragments:
                frags.extend(it.fragments)
            else:
                head = (informative_head(it.text, self.merge_head_words)
                        if self.informative_compress
                        else " ".join(it.text.split()[:self.merge_head_words]))
                frags.append((tuple(it.evidence_ids), head))

        kept, used = [], 0
        for ids, txt in reversed(frags):
            c = count_tokens(txt)
            if used + c > total_target:
                self.dropped += 1
                continue
            kept.append((ids, txt))
            used += c
        kept.reverse()
        if not kept:
            return []

        chunks, cur, cur_tok = [], [], 0
        for ids, txt in kept:
            c = count_tokens(txt)
            if cur and cur_tok + c > self.chunk_tokens:
                chunks.append(cur)
                cur, cur_tok = [], 0
            cur.append((ids, txt))
            cur_tok += c
        if cur:
            chunks.append(cur)

        newest = items[-1]
        out = []
        for ch in chunks:
            text = " ; ".join(t for _, t in ch)
            self.merges += 1
            out.append(MemoryItem(
                text=text, tier=WORKING, kind="summary",
                session=newest.session, turn_index=newest.turn_index,
                utility=max(i.utility for i in items),
                type_label="merged", fact_id=None, embedding=embed(text),
                source_ids=[i.id for i in items],
                covered_ids=sorted({i for ids, _ in ch for i in ids}),
                fragments=ch,
            ))
        return out

    def _merge(self, items: List[MemoryItem],
               target_tokens: int) -> Optional[MemoryItem]:
        """Fold several gists into one coarser gist, under a token target.

        The honest part is the accounting. Each input contributes (ids, text)
        fragments; fragments are kept newest-first until `target_tokens` is
        exhausted, and a fragment that does not fit is dropped **together with
        its evidence ids**. A merged item therefore only ever claims turns whose
        text it still physically contains.

        Without this, recursive merging inflates `covered_ids` while truncating
        the text, and the store reports a 100% write-path ceiling it cannot
        support -- coverage on paper, nothing in the tokens. Dropping ids with
        their text is what makes this a real rate-distortion trade (fewer tokens
        for a lossier but *honest* representation) instead of free credit.

        SWAP-IN (Phase 2): abstractive re-summarisation by the base LLM, which
        should retain far more content per token than head-truncation.
        """
        if not items:
            return None

        frags = []
        for it in items:
            if it.fragments:
                frags.extend(it.fragments)          # already fragment-accounted
            else:
                head = (informative_head(it.text, self.merge_head_words)
                        if self.informative_compress
                        else " ".join(it.text.split()[:self.merge_head_words]))
                frags.append((tuple(it.evidence_ids), head))

        kept, used = [], 0
        for ids, txt in reversed(frags):            # newest first
            c = count_tokens(txt)
            if used + c > target_tokens:
                self.dropped += 1                   # honest loss, ids go too
                continue
            kept.append((ids, txt))
            used += c
        kept.reverse()
        if not kept:
            return None

        text = " ; ".join(t for _, t in kept)
        covered = sorted({i for ids, _ in kept for i in ids})
        newest = items[-1]
        self.merges += 1
        return MemoryItem(
            text=text, tier=WORKING, kind="summary",
            session=newest.session, turn_index=newest.turn_index,
            utility=max(i.utility for i in items),
            type_label="merged",
            fact_id=None,
            embedding=embed(text),
            source_ids=[i.id for i in items],
            covered_ids=covered,
            fragments=kept,
        )

    # ------------------------------------------------------------ tier 3
    def add_long(self, text: str, source: MemoryItem, kind: str = "fact",
                 novelty_threshold: float = 0.92):
        """Write an atomic fact, with dedupe and supersession."""
        e = embed(text)
        for existing in self.long:
            if not existing.is_active:
                continue
            sim = cosine(e, existing.embedding)
            if sim >= novelty_threshold:
                # near-duplicate: reinforce rather than store twice
                existing.access_count += 1
                return existing
        it = MemoryItem(
            text=text, tier=LONG_TERM, kind=kind,
            session=source.session, turn_index=source.turn_index,
            utility=source.utility, type_label=source.type_label,
            fact_id=source.fact_id, embedding=e, source_ids=[source.id],
        )
        # Supersession: a newer fact about the same slot invalidates the old one.
        #
        # Keyed on SEMANTIC similarity, not fact_id. The old fact_id version was
        # an oracle -- it was handed the exact record to invalidate. A real system
        # has to find it, so this retires the single closest active record above
        # `supersede_threshold`. Only the best match, never every match above the
        # bar, or one update could retire half the store.
        #
        # This is a genuinely fallible mechanism and that is the point: its
        # accuracy now depends on embedding quality, which makes it a real
        # ablation target rather than a free win.
        if self.supersede and source.type_label == "update":
            best, best_sim = None, self.supersede_threshold
            for existing in self.long:
                if not existing.is_active:
                    continue
                sim = cosine(e, existing.embedding)
                if sim >= best_sim:
                    best, best_sim = existing, sim
            if best is not None:
                best.superseded_by = it.id
        self.long.append(it)
        self.enforce_store_budget()
        return it

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k: int = 5,
                 include_working: bool = False) -> List[MemoryItem]:
        """Rank stored items against the query.

        `include_working` puts Tier 2 into the retrieval pool as well. Tier 2 is
        otherwise read POSITIONALLY -- packed by density, never by relevance to
        the question being asked -- so anything routed there is unreachable no
        matter how much of it is stored. That is why growing Tier 2 alone made
        recall WORSE (12.1% vs 24.2% at S=16384): more stored, none of it
        addressable. Tier 2 items already carry embeddings, so making them
        retrievable costs nothing but the ranking.
        """
        q = embed(query)
        pool = [i for i in self.long if i.is_active]
        if include_working:
            pool += list(self.working)
        scored = [(cosine(q, i.embedding), i) for i in pool]
        scored.sort(key=lambda x: -x[0])
        out = [i for s, i in scored[:k] if s > 0]
        for i in out:
            i.access_count += 1
        return out

    # ------------------------------------------------- storage budget (§22.3)
    def stored_tokens(self) -> int:
        """Tokens of text this store physically holds, across all three tiers.

        This is the quantity the whole project had never constrained. Every
        experiment up to §22 capped the CONTEXT assembled per query but left
        the STORE unbounded, so "keep everything and retrieve top-k" was
        charged nothing for holding all 419 turns of a conversation. Under that
        accounting forgetting can only ever lose information -- there is no
        budget it saves -- and the ablation duly showed the compression
        machinery to be a net loss against keeping everything.

        Retired (superseded) records are excluded: they are reclaimed on the
        next enforcement pass, and a retired record is not storage a real
        system keeps paying for.
        """
        return sum(count_tokens(i.text) for i in self._held())

    def _held(self):
        """Every item the store is physically paying for."""
        return (list(self.short) + list(self.working)
                + [i for i in self.long if i.is_active])

    def _cost(self, it: MemoryItem) -> int:
        """What one item costs against the storage budget."""
        if self.cost_mode == "bytes":
            # Tier 1 is a raw turn buffer with no vector; Tiers 2 and 3 are
            # embedded and so pay for one.
            vec = self.vector_bytes if it.embedding is not None else 0
            return len(it.text.encode("utf-8")) + vec
        return count_tokens(it.text)

    def stored_cost(self) -> int:
        """Total held, in whatever unit the budget is denominated in."""
        return sum(self._cost(i) for i in self._held())

    def _forget_key(self, it: MemoryItem):
        """Eviction order: the item with the lowest key is forgotten first.

        Deliberately minimal -- `utility` from the scorer, oldest breaking
        ties. §22.2 is the cautionary tale here: one untuned threshold
        (`tau_fact`) turned out to dominate every carefully engineered
        component in the system. Stacking heuristics into the eviction rule
        would repeat that mistake with more knobs, so this stays at one signal
        that is already measured, already ablated, and cheap to reason about.
        `scored_eviction=False` gives plain FIFO as the control.

        NEVER reads `fact_id` or `covered_ids`. Those are evaluation labels,
        and choosing what to forget is precisely a policy decision -- the same
        class of leak that inflated the Step 0 headline by 45 points. The
        invariance test in test_memgate.py covers this path too.
        """
        if not self.scored_eviction:
            return (0.0, it.turn_index)
        return (it.utility, it.turn_index)

    def _demote(self, it: MemoryItem) -> bool:
        """Compress a Tier 3 fact into a Tier 2 gist. True if it shrank.

        This is the cache -> RAM -> disk demotion the design (§5) always
        described but which only ever ran on Tier-2 consolidation. Under
        storage pressure it is the cheap loss: pay fewer tokens for a lossier
        record rather than lose the record outright.

        The gist keeps the evidence ids of what it came from, exactly as a
        merged gist does. That is generous to evidence-recall -- a truncated
        gist still claims its id -- which is why the answer-presence metric
        (§19) exists and is reported alongside.
        """
        gist = (informative_head(it.text, self.merge_head_words)
                if self.informative_compress
                else " ".join(it.text.split()[:self.merge_head_words]))
        if self.cost_mode == "bytes":
            # A demoted gist still carries its own vector, so shortening the
            # text only pays if the text was the expensive part. Under byte
            # accounting on short turns it usually is not, and demotion then
            # costs MORE than it saves -- which the check has to catch, or
            # enforcement loops without making progress.
            shrank = (len(gist.encode("utf-8")) + self.vector_bytes
                      < len(it.text.encode("utf-8")) + self.vector_bytes)
        else:
            shrank = count_tokens(gist) < count_tokens(it.text)
        if not shrank:
            return False                      # nothing gained; let it go
        self.demoted += 1
        self.working.append(MemoryItem(
            text=gist, tier=WORKING, kind="summary",
            session=it.session, turn_index=it.turn_index,
            utility=it.utility, type_label="demoted", fact_id=None,
            embedding=embed(gist), source_ids=[it.id],
            covered_ids=sorted(it.evidence_ids),
            fragments=[(tuple(it.evidence_ids), gist)],
        ))
        return True

    def enforce_store_budget(self):
        """Hold total stored tokens at or below `store_budget`.

        Cheapest loss first:
          1. reclaim superseded records   -- already logically retired
          2. demote the lowest-utility Tier 3 fact into a Tier 2 gist
          3. drop from Tier 2 when even the gist will not fit

        Tier 1 is never evicted: it is the live turn buffer, it is what makes
        the next few turns coherent, and it is bounded at `short_capacity`
        items anyway. A budget small enough that Tier 1 alone breaches it is
        reported rather than silently violated -- see `stored_tokens`.

        Each iteration strictly reduces `stored_tokens` (a demoted gist is
        smaller than its source by construction, and a drop removes tokens
        outright), so the loop terminates.
        """
        if self.store_budget is None or self._enforcing:
            return
        self._enforcing = True
        try:
            if any(not i.is_active for i in self.long):
                n = len(self.long)
                self.long = [i for i in self.long if i.is_active]
                self.reclaimed += n - len(self.long)

            self._consolidate_if_needed()

            guard = 0
            while self.stored_cost() > self.store_budget:
                guard += 1
                if guard > 8192:              # cannot happen; never spin
                    break
                if self.long:
                    victim = min(self.long, key=self._forget_key)
                    self.long.remove(victim)
                    self.evicted += 1
                    if not (self.demote_on_pressure and self._demote(victim)):
                        self.dropped += 1
                    continue
                if self.working:
                    victim = min(self.working, key=self._forget_key)
                    self.working.remove(victim)
                    self.dropped += 1
                    continue
                break                          # only Tier 1 left; see docstring
        finally:
            self._enforcing = False

    def stats(self):
        return {
            "short": len(self.short),
            "working": len(self.working),
            "long": len(self.long),
            "dropped": self.dropped,
            "consolidations": self.consolidations,
            "merges": self.merges,
            "evicted": self.evicted,
            "demoted": self.demoted,
            "reclaimed": self.reclaimed,
            "stored_tokens": self.stored_tokens(),
            "stored_cost": self.stored_cost(),
            "cost_mode": self.cost_mode,
        }

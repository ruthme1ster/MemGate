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
                 supersede: bool = True):
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
        return it

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k: int = 5) -> List[MemoryItem]:
        q = embed(query)
        scored = [
            (cosine(q, i.embedding), i)
            for i in self.long if i.is_active
        ]
        scored.sort(key=lambda x: -x[0])
        out = [i for s, i in scored[:k] if s > 0]
        for i in out:
            i.access_count += 1
        return out

    def stats(self):
        return {
            "short": len(self.short),
            "working": len(self.working),
            "long": len(self.long),
            "dropped": self.dropped,
            "consolidations": self.consolidations,
            "merges": self.merges,
        }

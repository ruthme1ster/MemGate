"""Core data types for MemGate."""
from dataclasses import dataclass, field
from typing import Optional, List
import itertools

_ids = itertools.count(1)

# Tier constants
SHORT_TERM = 1   # raw recent turns, verbatim
WORKING = 2      # compressed summaries
LONG_TERM = 3    # atomic facts, retrieved on demand

TIER_NAMES = {SHORT_TERM: "short_term", WORKING: "working", LONG_TERM: "long_term"}


@dataclass
class Turn:
    """One message in the conversation."""
    session: int
    index: int
    speaker: str
    text: str
    fact_id: Optional[str] = None   # ground truth: planted fact, if any
    is_filler: bool = False         # ground truth: known filler

    @property
    def uid(self) -> str:
        return f"s{self.session}t{self.index}"


@dataclass
class MemoryItem:
    """A unit of stored memory.

    Note `superseded_by`: this is how we handle memory staleness, i.e. a stored
    fact silently becoming wrong when circumstances change later.
    """
    text: str
    tier: int
    kind: str                       # turn | summary | fact
    session: int
    turn_index: int
    utility: float = 0.0
    type_label: str = "unknown"
    fact_id: Optional[str] = None
    embedding: Optional[object] = None
    source_ids: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    access_count: int = 0
    id: str = ""
    # Evidence ids folded into this item when several were merged during
    # consolidation. A merged gist stands in for all of them, so scoring must
    # see all of them -- otherwise re-summarising would look like data loss.
    # Note this makes evidence-recall MORE generous, which is exactly why the
    # answer-presence metric exists to check whether the content really survived.
    covered_ids: List[str] = field(default_factory=list)
    # (ids, text) pairs a merged gist still physically contains. Merging is
    # recursive, so without per-fragment accounting `covered_ids` accumulates
    # while the text is re-truncated away -- one item ended up claiming 373
    # turns in 221 tokens. A fragment is kept whole or dropped with its ids;
    # never kept as a bare id whose text is gone.
    fragments: List[tuple] = field(default_factory=list)

    @property
    def evidence_ids(self) -> List[str]:
        out = list(self.covered_ids)
        if self.fact_id:
            out.append(self.fact_id)
        return out

    def __post_init__(self):
        if not self.id:
            self.id = f"m{next(_ids)}"

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None


@dataclass
class Question:
    """An evaluation probe.

    The synthetic set plants exactly one evidence turn per probe, so
    `gold_fact_id` is enough. LoCoMo does not: a multi-hop question can cite up
    to 19 evidence turns, and a policy that retrieves 18 of them still cannot
    answer. `gold_evidence` carries the full set and `evidence_ids` is the one
    accessor every scorer should use.

    `category` is the LoCoMo question type: 1 multi-hop, 2 temporal,
    3 open-domain, 4 single-hop, 5 adversarial. (Verified against the data, not
    assumed: category 1 averages 3.13 evidence turns to category 4's 1.07, and
    all 446 category-5 items carry an `adversarial_answer` field.)
    """
    text: str
    gold_fact_id: str = ""
    asked_after_session: int = 0
    kind: str = "single_hop"
    gold_evidence: List[str] = field(default_factory=list)
    category: Optional[int] = None
    answer: str = ""
    # True when every token of `answer` appears in the evidence turns, i.e. the
    # answer is literally recoverable from retrieved text rather than inferred.
    # Only 41.7% of LoCoMo questions qualify (single-hop 61%, temporal 8%,
    # open-domain 4%) because temporal answers are dates derived from session
    # timestamps and open-domain answers are commonsense inferences. The
    # answer-presence metric is scored ONLY over this subset -- elsewhere it
    # would measure the benchmark's phrasing, not the memory policy.
    answer_recoverable: bool = False

    @property
    def evidence_ids(self) -> List[str]:
        if self.gold_evidence:
            return self.gold_evidence
        return [self.gold_fact_id] if self.gold_fact_id else []


@dataclass
class ContextBundle:
    """The context a policy assembled for one query."""
    text: str
    items: List[MemoryItem]
    tokens: int
    budget: int

    @property
    def fact_ids(self):
        out = set()
        for i in self.items:
            out.update(i.evidence_ids)
        return out

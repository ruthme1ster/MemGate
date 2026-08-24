# MemGate — Session Record

**Capstone Project** · SVKM's NMIMS, Indore Campus
Simar Singh Khanuja & Yash Ramchandani
Session 1: 12 August 2026 · Session 2: 17 August 2026 · Session 3: 24 August 2026

A complete record of what was discussed, decided, built and found in these
working sessions. Written so it can be picked up cold later.

---

> ## ⚠ READ FIRST — Session 2 corrected the Step 0 headline numbers
>
> Session 2 found an **evaluation leak**: the memory policy was reading
> `fact_id`, which is the *ground-truth answer key*, when deciding what to keep.
> The Step 0 results in §6 and the "95% vs 10%" claim in the **Review 1 deck**
> are inflated by it and **must not be presented again as they stand**.
>
> | Synthetic set @ 800 tokens | As presented | Honest |
> |---|---|---|
> | P1 MemGate | 95.0% | **50.0%** |
> | P0 sliding window | 10.0% | 10.0% |
>
> The leak is fixed, pinned by a regression test, and the project now has real
> LoCoMo numbers. Full detail in **§13**.
>
> ## ⚠ ALSO READ — Session 3 changed the headline claim
>
> §22 concluded that the compression machinery was a **net loss** against
> simply keeping everything and retrieving. Session 3 imposed the missing
> **storage budget** and re-ran it. The corrected finding:
>
> * **Selection wins.** Forgetting by scored utility rather than oldest-first
>   is worth **+8.7 answer-recall points** at identical storage and identical
>   fidelity — that gap is the decision policy and nothing else.
> * **Compression still loses.** Compressing on eviction wins *evidence*
>   recall and loses *answer* recall, because a gist keeps the ids of a turn
>   whose text it discarded.
>
> Do not present tiered compression as validated. Full detail in **§23**.

---

## 1. What the project is

**Original topic:** AI Memory Compression for LLM Agents — "an operating system
for AI memory".

**Working name:** **MemGate** (just a name; can be changed freely).

**One-line definition:**
> MemGate is a software layer that wraps an existing LLM (open model or hosted
> API) and controls what enters the prompt on each turn. It decides what to
> **keep**, **compress**, **forget** and **recall**, so long multi-session
> conversations stay inside a fixed token budget without losing what matters.

It treats the base model as a **black box**. We train nothing.

**Explicitly out of scope:** re-training/fine-tuning the base model, image or
audio memory, a polished product or UI, production/multi-user serving.

---

## 2. Key decisions taken this session

| # | Decision | Rationale |
|---|---|---|
| 1 | Deck must use the **NMIMS template** (red bands, logo, serif) | Faculty requirement; matched the EcoHazNet reference deck exactly |
| 2 | Build **real editable diagrams**, not AI-image placeholders | More original than the reference deck, and editable |
| 3 | **Base paper = MemGPT** (Packer et al., 2023) | Matches the "LLM as an OS" framing; our work extends its missing decision policy |
| 4 | Scope = **P0–P3 only**; **P4 (interactive) on hold** | Pending faculty approval; keeps the core project safe from UX risk |
| 5 | Rename "Importance Scoring" → **Adaptive Memory Decision Engine** | Clearer, and names the actual contribution |
| 6 | Replace "learned scoring" → **"adaptive importance scoring policy"** | More precise wording in the literature-review gaps |
| 7 | **Sharpen the positioning** (see §4) | Tiered memory alone is no longer novel — production systems ship it |
| 8 | Step 0 metric = **context recall @ budget** | Isolates the memory policy from model reasoning; no API key needed |
| 9 | **Build the evaluation harness before the clever parts** | Makes every later idea a measurable experiment, not a guess |

---

## 3. The four panel objections, and the answers

### Q1 — Is this project really required in the current phase?

**Yes.** The objection assumes large context windows solved this. Evidence says
they relocated the problem rather than removing it.

* **Context rot** — Chroma Research (July 2025) tested 18 frontier models
  (GPT-4.1, Claude 4, Gemini 2.5, Qwen3): accuracy degrades **30–50% well before
  the documented context limit**. A 200K window can show serious loss at ~50K
  tokens. It is an *architectural property of attention*, not a training gap —
  so bigger windows don't fix it, curation does.
* **Cost and latency** — on LoCoMo, full-context ≈ 72.9% accuracy at ~26,000
  tokens and 9.87 s median / 17.12 s p95 latency; selective memory ≈ 66.9% at
  ~1,800 tokens and 0.71 s / 1.44 s. Roughly **14× cheaper, ~91% lower p95**.
* **Live industry problem** — a whole product category exists: Letta/MemGPT,
  Mem0, Zep, Cognee. Independent comparisons report up to **15-point accuracy
  gaps between memory architectures** on temporal queries.

### Q2 — Is it feasible?

**Yes**, because we train nothing.

* Inference only — a single GPU or Colab runs a 7–8B open model; the large model
  is reached by API.
* Public benchmarks already exist (LoCoMo, LongMemEval, LongBench) — no data
  collection.
* Every heavy component is mature open-source (embeddings, vector search,
  summarisation).
* Staged phases, each independently demoable.
* Only recurring cost is limited API calls.

### Q3 — Is more research required in this area?

**This is the one the panel was right about.** Tiered memory with forgetting is
*already shipped* by Letta, Mem0 and Zep. Presented as "a memory layer with
three tiers", the honest verdict is that it has been built.

The novelty must sit one level deeper — in the **decision policy** — and that is
exactly where the field says the gap is:

* A 2026 review states current systems **"do not solve the fundamental
  challenge: deciding what to remember and what to forget"**.
* **Adaptive compression is still heuristic**; **relevance scoring lacks
  theoretical guarantees**.
* Memory **staleness** and cross-session consolidation are named open problems.
* Temporal reasoning is the hardest category (~25% loss when scaling 10×).

**The measurable gap:**
* LongMemEval — oracle retrieval ≈ **92%**, same model interactive ≈ **58%**:
  a **34-point gap** that is purely memory-selection failure, not reasoning.
* LoCoMo — selective memory is ~14× cheaper but still **~6 points behind**
  full-context.

**Theory now exists, but no system:** *"What to Keep, What to Forget: A
Rate–Distortion View of Memory Compaction in LLMs and Agents"* (Colaco &
Lahjouji, arXiv:2607.08032, July 2026) argues all memory-compaction decisions
are one problem — a rate–distortion decision about what to retain, at what
fidelity, under a budget, to preserve downstream utility. It is a *framework,
not a system*. That is our opening.

### Q4 — What is the end product?

1. **MemGate library** — installable Python package wrapping any LLM.
2. **Evaluation harness** — reports accuracy *and* tokens *and* latency together.
3. **Empirical results** — policy comparison + ablations + frontier plots.
4. **Report + demo** — paper-shaped report, plus a live long-conversation demo.

---

## 4. Sharpened project statement (use from Review 2 onward)

> MemGate is not "another memory layer". It is a study of **the memory decision
> policy itself**: we treat keep / compress / forget as a **rate–distortion
> problem under a fixed token budget**, implement several scoring policies
> (heuristic, LLM-judged salience, and a distilled scorer), and measure them on
> a common **accuracy-per-token frontier** against full-context and existing
> memory systems.
>
> **Research question:** how much of the 6-point cost–quality gap and the
> 34-point oracle gap can a better decision policy recover, and at what token
> budget?

---

## 5. Technical design

### The three tiers — a lifecycle, not a partition

Everything enters Tier 1. The routing decision happens **on eviction** from
Tier 1 (cheaper: each item is scored once, after the point where it's still
trivially available).

| Tier | Holds | Criterion | In context? |
|---|---|---|---|
| **1 — Short-term** | raw verbatim turns | **When** — recent? (FIFO, no scoring) | Always |
| **2 — Working** | compressed summaries | **Gist** — continuity worth keeping? | Always (small) |
| **3 — Long-term** | atomic facts + embeddings | **What** — durable, retrievable? | Only on retrieval match |

Tiers are **not mutually exclusive** — one evicted segment can drop a fact into
Tier 3 *and* contribute a line to the Tier 2 summary.

### Routing rule

```
u >= tau_fact   -> Tier 3 (durable fact), embed + dedupe
u >= tau_low    -> Tier 2 (compressed gist)
otherwise       -> dropped from context
```

Defaults used: `tau_fact = 0.45`, `tau_low = 0.15`, buffer `N = 6`,
budget split `(short 25%, working 25%, long 50%)`.

### Four scoring signals

1. **Durability** — entities, numbers, dates, decisions, preferences, constraints
2. **Novelty** — cosine distance to nearest existing memory (dedupe)
3. **Reference-likelihood** — is this the kind of thing asked about later?
4. **Filler detection** — greetings, acknowledgements

### Memory staleness

Handled by `superseded_by` on the record, plus `last_accessed`, `access_count`
and `source_ids`. A later "Actually, push the deadline to April 2" marks the
earlier March 15 fact as superseded. This is one of the open problems named in
the 2026 literature.

### Tier 2 consolidation

Tier 2 cannot grow forever. Over budget → recursively re-summarise the oldest
half; anything carrying a durable fact demotes to Tier 3, the rest is dropped.
That is the cache → RAM → disk demotion path.

### The four policies

| | Policy | Scoring | Status |
|---|---|---|---|
| P0 | Sliding window | recency only, **budget-driven** | done |
| P1 | MemGate heuristic | rules + shallow NER + novelty | done |
| P2 | LLM-judged salience | small LLM rates importance 0–1 + type | Phase 3 |
| P3 | Distilled scorer | trained on P2's labels — P2 quality at P1 cost | Phase 3 |

### P4 — Interactive (ON HOLD, pending faculty)

Discussed but deferred. The idea: ask the user whether to save an item.

* Naive version fails — friction/consent fatigue, doesn't scale, users can't
  predict future needs, **and it breaks offline benchmark evaluation**.
* Workable version: **confidence-gated active learning** — auto-route when
  confident, ask *only* in the uncertainty band. Plus explicit commands
  ("remember that…"), batched end-of-session review, passive correction.
* Upside: every user answer is a **training label** for P3, and it adds a second
  axis — *accuracy per user question*.
* Can be evaluated offline with a **simulated user** (benchmark QA pairs reveal
  which facts were needed later).

---

## 6. Step 0 — what was built and what it showed

### Built (~1,150 lines, 30/30 tests passing, zero dependencies)

```
memgate/
  memgate/
    types.py     MemoryItem, Turn, Question (incl. superseded_by)
    utils.py     tokenizer + hashing embedder   [SWAP-IN marked]
    store.py     three-tier store, dedupe, supersession, consolidation
    scoring.py   P0/P1 implemented; P2/P3 stubs on the same interface
    policies.py  baselines + MemGate, routing + budget-aware assembly
    data.py      synthetic generator + integrity assertions + LoCoMo stub
    harness.py   metrics, sweep, table/CSV output
  run_experiment.py    diagnose.py    test_memgate.py    README.md
```

Runs with **no API key, no GPU, no downloads**.

### Results (synthetic set: 600 turns / 20 sessions / 40 probes)

> **⚠ SUPERSEDED — these numbers contain the evaluation leak. See §13.**

| Policy | Budget | Recall | Avg tokens |
|---|---|---|---|
| Full-context | unlimited | 100.0% | 5451 |
| Oracle (diagnostic bound) | 3200 | 100.0% | 16 |
| P0 sliding window | 800 | 10.0% | 800 |
| **P1 MemGate** | 800 | **95.0%** | 626 |
| P0 sliding window | 3200 | 57.5% | 3198 |
| **P1 MemGate** | 1600 | **100.0%** | 976 |

**Headline:** MemGate reaches the same **100% recall as full-context using
5.6× fewer tokens**; at an equal 800-token budget it recalls **95% vs P0's 10%**.

### Honest caveat to state before anyone asks

This is a **synthetic sanity set, not a research result**. It proves the
pipeline and metrics are correct. Real numbers need LoCoMo. The embedder is a
hashing bag-of-words and the summariser is extractive — both marked `SWAP-IN`.

---

## 7. Bugs found and fixed (worth knowing — they show method)

1. **P0 was a strawman.** First version gave the sliding window a fixed 12-turn
   window → 0% everywhere. Changed to **budget-driven**: P0 now spends exactly
   the same token budget as MemGate, so the only difference is *which* turns are
   chosen. Its score rose to 57.5% and MemGate still dominates — a defensible
   comparison.
2. **Duplicate fact statements.** Two different `fact_id`s rendered *identical
   text*; the deduper correctly collapsed them, making one permanently
   unanswerable. Looked like a memory failure but was a **dataset artefact**.
   Fixed by giving each fact a unique subject (100 unique combinations) and
   adding `_assert_wellformed()` so it can't recur silently.
3. **Duplicate probe questions** — caught by the new assertion immediately after
   fixing #2 (the "project" template's probe didn't reference the unique subject).
4. **`retrieve_k` bound before the budget did**, flattening the frontier. Raised
   k so the *budget* is the binding constraint.
5. **Deck:** references slide rendered `[object Object]` (text-run nesting);
   Gantt legend collided with the footer; long titles wrapped and overlapped
   body text in the NMIMS template (fixed by forcing placeholder geometry and
   disabling autofit at the XML level).

> Lesson worth repeating in the report: building the **harness first** is what
> surfaced #1–#4. A measurement artefact that looks like a model failure is the
> most dangerous kind of bug.

---

## 8. Deliverables produced

| File | What it is |
|---|---|
| `MemGate_Capstone_Review1_NMIMS_v3.pptx` | 17-slide Review 1 deck in the NMIMS template |
| `MemGate_Defence_and_Build_Plan_FINAL.pdf` | 6-page evidence-backed answers to the four objections + full build plan |
| `MemGate_Technology_Stack.pdf` | One-page tech stack + architecture mapping |
| `memgate/` | Step 0 source code, tests, harness |
| `SESSION_RECORD.md` | This document |

**Deck structure (17 slides):** Title → Introduction → Problem Statement →
Objectives → Scope → Need → Proposed System → System Architecture →
Implementation Workflow → Tools & Technology → Feasibility → Literature Review
(1–2) → Gantt Chart → References → Thank You → *Backup: Implementation Roadmap*.

---

## 9. Literature (10 verified citations)

Author lists, venues and years were each verified against source pages.

1. **Packer et al. (2023)** — *MemGPT: Towards LLMs as Operating Systems*, arXiv:2310.08560 ← **base paper**
2. Zhong et al. (2024) — *MemoryBank*, AAAI 38(17), 19724–19731
3. Park et al. (2023) — *Generative Agents*, UIST '23
4. Xiao et al. (2024) — *StreamingLLM / Attention Sinks*, ICLR
5. Zhang et al. (2023) — *H2O: Heavy-Hitter Oracle*, NeurIPS
6. Jiang et al. (2023) — *LLMLingua*, EMNLP, 13358–13376
7. Mu, Li & Goodman (2023) — *Learning to Compress Prompts with Gist Tokens*, NeurIPS
8. Chevalier et al. (2023) — *Adapting Language Models to Compress Contexts*, EMNLP
9. Zhang et al. (2024) — *Survey on the Memory Mechanism of LLM-Based Agents*, arXiv:2404.13501
10. Xu et al. (2025) — *A-MEM: Agentic Memory for LLM Agents*, NeurIPS, arXiv:2502.12110

**Additional 2025–2026 sources** used for the defence document: Chroma Research
*Context Rot* (2025); Colaco & Lahjouji, arXiv:2607.08032 (2026); LongMemEval
(arXiv:2410.10813); LoCoMo (Maharana et al., 2024); 2026 agent-memory benchmark
and framework surveys.

> Note: several 2026 performance figures come from **vendor blogs** and should
> be treated as indicative. Re-measuring them under one harness is itself part
> of the contribution.

---

## 10. Build plan — remaining steps

| Step | Work | Done when |
|---|---|---|
| ~~0~~ | ~~Harness + baselines~~ | ✅ **complete** |
| 1 | Wire `load_locomo()`; swap in `sentence-transformers` | Real benchmark numbers reproduced |
| 2 | Real abstractive summariser in the compressor | Budget never exceeded, quality holds |
| 3 | P2 LLM-judge; then distil into P3 | P3 approaches P2 accuracy at far lower cost |
| 4 | Full sweep, ablations, frontier plots, second model | Frontier chart complete |
| 5 | Report, demo, packaging | Submitted |

**Ablations planned:** remove each scoring signal; vary token budget; vary buffer
size N; disable retrieval fallback — to show which component earns the gain.

---

## 11. Open items

* **P4 (interactive memory)** — on hold pending faculty opinion.
* **Project name** — "MemGate" is provisional.
* **Real-data demo** — running MemGate over *this session's own transcript* was
  proposed as a second, non-synthetic demo (dogfooding). Not yet done.
* **Blocker:** the sandbox needs free space on the **internal** drive to run
  code; an external volume doesn't provide its scratch space. P2/P3 build is
  waiting on this.

---

## 12. How to run

```bash
cd ~/Documents/Capstone\ Project/memgate
python3 test_memgate.py      # expect: 30 passed, 0 failed
python3 run_experiment.py    # table + results/step0_results.csv + frontier.png
python3 diagnose.py          # miss analysis by cause
```

Python 3.8+. `matplotlib` only for the plot; everything else is standard library.

---

# Session 2 — 17 August 2026

## 13. The evaluation leak, and what it was hiding

### 13.1 What was found

`store.py` consulted `MemoryItem.fact_id` in two decisions:

```python
if it.fact_id or it.utility >= 0.5:      # consolidation: promote to Tier 3
if source.type_label == "update" and source.fact_id:   # supersession target
```

`fact_id` is the **ground-truth evaluation label** (`types.py`) — the harness
scores a policy by asking whether the gold `fact_id` reached the context. So the
policy was reading the answer key to decide what to keep.

Isolating the single promotion clause on the synthetic set:

| Budget | P1 as presented | P1 leak-free | P0 |
|---|---|---|---|
| 400 | 92.5% | **45.0%** | 5.0% |
| 800 | **95.0%** | **50.0%** | 10.0% |
| 1600 | 100.0% | **52.5%** | 27.5% |
| 3200 | 100.0% | **52.5%** | **57.5%** |

Roughly **45–47 points** of the headline was the leak. The Review 1 claim
"95% vs 10% at 800 tokens" does not survive; the honest figure is **50% vs 10%**,
and at 3200 tokens **P0 overtakes MemGate**.

### 13.2 What the leak was hiding — a dead consolidation path

Removing it exposed a structural flaw. Routing sends items scoring
`u >= tau_fact (0.45)` straight to Tier 3, and `u >= tau_low (0.15)` to Tier 2.
Consolidation then promotes out of Tier 2 only if `u >= 0.5`. But **every item in
Tier 2 scored below 0.45 by construction**, so nothing can ever clear 0.5.

Tier 2 was a death chamber: everything entering it was eventually discarded.
`it.fact_id` had been silently performing the retention that the consolidation
policy was supposed to perform. On LoCoMo this shows up starkly — of 419 turns in
`conv-26`, only **30 (7.2%)** are retained anywhere.

This is the design in §5 ("recursively re-summarise the oldest half") not
actually being implemented — consolidation *discards* rather than re-summarises.
**Implementing it properly is now the main technical task**, and it is exactly
the rate–distortion question the project claims as its contribution.

### 13.3 Guard against recurrence

`test_memgate.py` pins it: the store must come out **bit-identical** when every
`fact_id`/`is_filler` is stripped from the input turns.

```
PASS  routing is invariant to ground-truth labels
```

Any future decision that reads a label fails this test. Same discipline as
`_assert_wellformed()` for dataset artefacts.

### 13.4 A second scorer bug found the same way

`UPDATE_HINTS` contained `"instead"`, so *"I decided to use Django for the board
**instead of** Rails"* was classed as a correction. 6 turns were labelled
`update` on a set that plants exactly **1**. Because those decision sentences are
near-identical in wording, semantic supersession then had them **retire each
other** — a cascade costing ~10 recall points. Fixed by separating genuine
correction cues (`actually`, `no longer`, `revised`, …) from contrastive
phrasing.

### 13.5 Fairness fix: budget-aware store sizing

`working_capacity_tokens` was pinned at 400 regardless of budget, so MemGate
could not fill a 4096-token budget however well it scored — it spent 1336. That
is the **same strawman that was fixed for P0 in Step 0**, pointed the other way.
Tier 2 is now sized from the budget. Effect on LoCoMo: 8.9% → **10.8%** at 2048,
9.1% → **15.1%** at 4096.

---

## 14. LoCoMo is wired (Step 1)

`load_locomo()` is implemented. 10 conversations, **5882 turns**, **1527
scoreable questions**.

**Two data hazards, both silent if unhandled:**

1. **`dia_id` is unique only within a conversation** — 5882 turns share just
   1033 distinct ids; `D1:1` exists in every conversation. Ids are namespaced
   `<sample_id>/<dia_id>`, otherwise evidence matches turns from unrelated
   conversations.
2. **9 questions cite evidence turns that do not exist**, plus 4 with empty
   evidence — unanswerable by construction, so dropped rather than left to
   depress every policy equally.

**Category mapping was verified from the data, not assumed:** category 1
averages 3.13 evidence turns (max 19) vs category 4's 1.07, and all 446
category-5 items carry an `adversarial_answer` field.
→ 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.

**Adversarial (category 5) is excluded from the headline metric.** Those
questions *do* carry evidence, but the correct behaviour is to decline to
answer, so retrieving the evidence does not indicate success.

**The metric is now multi-evidence** — a multi-hop question needs up to 19
turns, and retrieving 18 of 19 answers nothing:
* **strict recall** — every evidence turn present (headline)
* **soft recall** — mean fraction present (partial credit)

They coincide on single-evidence questions, which is why Step 0 needed only one.

## 15. First real LoCoMo numbers

**MiniLM 384d + tiktoken (`cl100k_base`)**, budget-aware sizing.
`strict` = every evidence turn present; `soft` = mean fraction present.

| Policy | Budget | Strict | Soft | Avg tokens |
|---|---|---|---|---|
| Full-context | unlimited | 100.0% | 100.0% | 18 959 |
| Oracle | 512 | 99.9% | 100.0% | 65 |
| P0 sliding window | 512 | 1.6% | 1.9% | 509 |
| **P1 MemGate** | 512 | **7.0%** | 8.7% | 487 |
| P0 sliding window | 1024 | 4.3% | 4.9% | 1022 |
| **P1 MemGate** | 1024 | **8.5%** | 10.5% | 899 |
| P0 sliding window | 2048 | 10.1% | 11.4% | 2047 |
| **P1 MemGate** | 2048 | **11.0%** | 13.3% | 1432 |
| P0 sliding window | 4096 | **18.8%** | 21.8% | 4094 |
| P1 MemGate | 4096 | 15.3% | 18.2% | 2058 |

### 15.1 The embedder buys almost nothing — as predicted

| Budget | P1 hashing | P1 MiniLM | Δ |
|---|---|---|---|
| 512 | 6.0% | 7.0% | +1.0 |
| 1024 | 8.1% | 8.5% | +0.4 |
| 2048 | 10.8% | 11.0% | +0.2 |
| 4096 | 15.1% | 15.3% | +0.2 |

**This was predicted before the run and is the strongest methodological result
of the session.** The diagnostic showed retrieval reaching 6.7% against a
write-path ceiling of 9.4% — nearly saturated — so a better embedder had almost
nothing left to recover. Swapping a bag-of-words hash for a 384-dimension
transformer moves the headline by **0.2 points at the working budget**.

The corollary matters for planning: **effort spent on retrieval, re-ranking or
a larger embedding model is effort wasted** until the write path stops
discarding 93% of the conversation. The embedder does earn its place — but on
supersession (§16.1), not on recall.

### 15.2 Per-category, at 2048 tokens

| Policy | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|
| Oracle | 100.0% | 100.0% | 100.0% | 100.0% |
| P0 sliding window | **2.2%** | 9.4% | 7.9% | 13.2% |
| P1 MemGate | 0.7% | **13.4%** | **12.4%** | **13.3%** |
| *(n questions)* | *278* | *320* | *89* | *840* |

MemGate wins on temporal, open-domain and single-hop, and is **crushed on
multi-hop (0.7% vs 2.2%)**. That is arithmetic, not a separate defect: multi-hop
questions cite 3.13 evidence turns on average, strict recall demands *all* of
them, and only ~7% of turns are retained. Retention failure compounds
multiplicatively with evidence count. Fixing the write path should lift
multi-hop fastest.

**Read this honestly: on real data the current heuristic does not work.** It
wins at tight budgets (6.0% vs 1.6% at 512) and at 2048, but P0 overtakes it at
4096. Multi-hop recall is ~0%.

The diagnosis is unambiguous and it is **not** retrieval:

```
conv-26:  turns retained anywhere        30/419  (7.2%)
          WRITE-PATH CEILING              9.4%
          read path (k=60, no budget cap) 6.7%
```

Retrieval reaches 6.7% against a ceiling of 9.4% — it is nearly saturated. **The
write path throws away 93% of the conversation.** No retrieval or embedding
improvement can beat 9.4% until Tier 2 stops discarding.

This is a real, defensible result: the P1 heuristic was tuned on synthetic data
where facts are explicit declaratives ("The deadline is March 15"). Natural
dialogue carries durable information diffusely, and a keyword-and-regex scorer
does not see it. **That is precisely the gap P2/P3 exist to close**, and it is now
measured rather than assumed.

`diagnose.py` attributes the leak-free synthetic misses the same way:

```
CAUSE SUMMARY: {'scorer_miss': 18, 'dedupe_collapse': 1, 'retrieval_miss': 2}
```

**The scorer is the bottleneck by an order of magnitude.** Both datasets agree,
and each cause is localised to one swappable component — which is what makes
Phase 3 a measurable experiment rather than a guess.

## 16. Component swaps

| Component | Was | Now |
|---|---|---|
| Tokenizer | `words × 1.3` | **tiktoken `cl100k_base`** ✅ |
| Embedder | hashing bag-of-words | **all-MiniLM-L6-v2, 384d** ✅ |

The weights are vendored at `models/all-MiniLM-L6-v2/` (90 868 376 bytes,
verified) because the Hub download failed repeatedly on this connection — Xet
`IncompleteBody`, consistency-check failures, and curl restarting instead of
resuming. `_local_model_ok()` checks the exact byte size, since a truncated file
loads as `InvalidHeaderDeserialization`.

The embedder is pluggable (`MEMGATE_EMBEDDER=auto|hash|minilm`) and the hashing
backend is retained deliberately as the ablation that shows how much of the
result depends on embedding quality.

**`backend_info()` is recorded into every results row.** This is not
bookkeeping: a resumed download produced a corrupt `model.safetensors`, the
`auto` path silently fell back to hashing, and only that provenance field
prevented a table being labelled MiniLM that was actually hashing. Fallback is
now a loud warning.

### 16.1 Supersession — the first clean component ablation

A prediction was made before the swap: with hashing, the genuine supersession
scored only **0.101** cosine against its target, so semantic supersession was
non-functional and MiniLM should fix it. **Confirmed, and sharper than expected.**

| Embedder | → true target | → nearest competitor | Outcome |
|---|---|---|---|
| **MiniLM 384d** | **0.865** | 0.741 | **Correct**, at every threshold 0.50–0.70 |
| Hashing BoW | 0.500 | **0.583** | **Wrong** — ranking inverted |

Under hashing the nearest competitor *outranks* the true target, so the
mechanism either retires the wrong record or stays silent. Under MiniLM it
retires the right one with a 0.124 margin and is insensitive to the threshold.

This is the first result attributing a gain to a single swapped component,
which is exactly the ablation structure the build plan calls for. It is also the
one place so far where the embedder is demonstrably load-bearing — on the
synthetic recall numbers MiniLM and hashing are near-identical (50.0% either
way), consistent with the scorer, not the embedder, being the bottleneck.

### 16.2 A dataset artefact found on the way

The result above was only obtainable after fixing the probe. The planted update
read *"Actually, we pushed **that deadline** to April 2"* — a bare anaphor with
no antecedent, fired ten sessions after its target, competing with four other
deadline facts. No mechanism could resolve it, and semantic supersession duly
retired the wrong record; the apparent "failure" was the dataset, not the policy.

It now names its subject (*"we pushed the bakery module deadline…"*), and
`_assert_wellformed()` refuses to build a set where an update does not reference
the fact it corrects. Third artefact of this class caught by an assertion —
see §7.

### 16.3 Provenance is not bookkeeping

`backend_info()` is recorded into every results row. A resumed download produced
a corrupt `model.safetensors`, the `auto` path silently fell back to hashing,
and only that field prevented a table being labelled MiniLM that was actually
produced by the baseline. Fallback is now a loud stderr warning.

## 17. Where this leaves the project

The story is **stronger**, not weaker — but it must be told accurately.

* The panel's Q3 objection ("has this not been built?") is now answerable with
  evidence: tiered memory is shipped, and **it does not work well** — a
  well-formed three-tier system scores 15% on LoCoMo where full-context scores
  100%.
* The 34-point oracle gap cited from LongMemEval is reproduced in our own
  harness: **Oracle 100% at 65 tokens vs P1 15% at 2055**. The gap is entirely
  memory selection.
* Building the harness first is what surfaced all of this — the lesson from §7,
  now with a much sharper example.

**Do not re-present the Review 1 numbers.** Use §15 and frame the write-path
ceiling as the identified research problem.

### Immediate next steps

1. **Fix consolidation** — re-summarise on eviction instead of discarding, so
   Tier 2 stops being a death chamber. Highest-value change; lifts the 9.4%
   ceiling directly.
2. Finish the MiniLM download; re-run and quantify the embedder's contribution.
3. P2 (LLM-judge) — the heuristic is the demonstrated bottleneck.
4. Consider an **answer-presence** metric: a Tier 2 summary is truncated to 14
   words but still carries its `fact_id`, so it counts as a hit even when the
   answer was cut. LoCoMo ships gold answer strings, so this is checkable and
   the current metric is generous to MemGate.

## 19. The answer-presence metric

Evidence recall asks "did the gold turn's **id** reach the context". That is
generous: a Tier 2 gist is truncated but keeps its id, so it scores a hit even
when the answer text was cut. LoCoMo ships gold answers, so this is checkable.

**It cannot simply replace the headline metric.** Only **41.7%** of
non-adversarial answers appear verbatim in their own evidence turns:

| Category | answer present in evidence |
|---|---|
| single-hop | 61% |
| multi-hop | 34% |
| temporal | 8% |
| open-domain | 4% |

Temporal answers are dates derived from session timestamps and open-domain
answers are commonsense inferences — neither is in the turn text. A global
answer-presence metric would cap even a perfect oracle at 41.7% and would
measure the benchmark's phrasing, not the memory policy.

So it is scored **only over the 715 questions whose answer is recoverable from
evidence text** (`Question.answer_recoverable`). On that subset it is exact, and
validated: **Full-context and Oracle both score 100.0%**, which is the check
that the subsetting is right. Stopwords are dropped, so "the deadline is March
15" is not judged present because "the" and "is" survived.

**It immediately earned its place.** During the consolidation work, strict
recall rose with budget (8.1% → 14.8%) while answer recall stayed pinned at
12.8%. The extra "recall" was pure id bookkeeping from merged gists with no
answer content behind it. Evidence recall alone would have reported that as
progress.

## 20. Fixing consolidation — and two wrong turns on the way

The target: Tier 2 must re-summarise on eviction instead of discarding, so it
stops being a death chamber (§13.2).

**Attempt 1 — merge the sub-threshold remainder into one gist.** Retention went
7.2% → 100% and the write-path ceiling 9.4% → 100%. Both were **fictitious**.
Merging is recursive, so `covered_ids` accumulated while the text was
re-truncated each pass; one item claimed **373 turns in 221 tokens**. Coverage
on paper, nothing in the tokens. Answer recall did not move, correctly refusing
to credit it.

**Fix — per-fragment accounting.** A merged gist now carries `(ids, text)`
fragments and may claim only ids whose text it still physically contains; a
fragment that does not fit is dropped **together with its ids**. Honest numbers:
retention **18.1%**, ceiling **25.5%**. That is the real rate–distortion trade —
fewer tokens for a lossier but truthful representation — rather than free credit.

**Attempt 2 — one merged block.** Ceiling up, but realized recall went *down*
(−1.2 strict at 2048). Diagnosis: a merged item is **atomic at assembly time**;
it fits the Tier 2 share whole or is skipped whole. Collapsing ~27 individually
packable items into one 512-token block lost more in granularity than density
won.

**Fix — chunked merging** (`chunk_tokens = 64`), plus Tier 2 packing ordered by
**conversation-per-token** rather than recency, so a dense gist is no longer
always last in line. Density is computed from `len(fragments)`, a structural
property — deliberately **not** from `covered_ids`, which are evidence ids and
would smuggle the answer key back into the policy.

### Result — FINAL, MiniLM + tiktoken, all fixes in

| Policy | Budget | Strict | Soft | Answer | Avg tokens |
|---|---|---|---|---|---|
| Full-context | unlimited | 100.0% | 100.0% | 100.0% | 18 959 |
| Oracle | 512 | 99.9% | 99.9% | 100.0% | 65 |
| P0 sliding window | 512 | 1.6% | 1.9% | 4.9% | 509 |
| **P1 MemGate** | 512 | **7.7%** | **9.3%** | **12.0%** | 484 |
| P0 sliding window | 1024 | 4.3% | 4.9% | 9.1% | 1022 |
| **P1 MemGate** | 1024 | **9.4%** | **11.6%** | **16.1%** | 897 |
| P0 sliding window | 2048 | 10.1% | 11.4% | 18.3% | 2047 |
| **P1 MemGate** | 2048 | **12.4%** | **15.0%** | **19.2%** | 1430 |
| P0 sliding window | 4096 | **18.8%** | **21.8%** | **31.3%** | 4094 |
| P1 MemGate | 4096 | 16.6% | 20.3% | 21.1% | 2062 |

Gain from the consolidation fix (MiniLM, like-for-like against §15):

| Budget | P1 before | P1 after | Δ |
|---|---|---|---|
| 512 | 7.0% | **7.7%** | +0.7 |
| 1024 | 8.5% | **9.4%** | +0.9 |
| 2048 | 11.0% | **12.4%** | +1.4 |
| 4096 | 15.3% | **16.6%** | +1.3 |

**MemGate now leads P0 on every metric at every budget up to 2048** — including
answer recall (19.2% vs 18.3%), i.e. on content actually present, not just ids —
while spending 1430 tokens to P0's 2047. At 4096 P0 still wins outright.

Per-category strict recall at 2048:

| Policy | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|
| P0 sliding window | **2.2%** | 9.4% | 7.9% | 13.2% |
| P1 MemGate | 1.1% | **15.3%** | **10.1%** | **15.4%** |

Multi-hop improved (0.7% → 1.1%) but still trails P0. It needs *all* ~3.13 cited
turns, so retention failure compounds multiplicatively — it will be the last
category to move and the clearest signal that the write path is genuinely fixed.

**The honest reading:** the ceiling rose 9.4% → 25.5% but only ~1.4 points of it
converted. Head-truncation is too lossy a summariser to exploit the headroom it
creates. **This is precisely the Step 2 dependency** — a real abstractive
summariser retains far more content per token, and the ceiling is now there for
it to claim.

## 21. The compressor — choosing *which* words to keep

§20 left the diagnosis: the ceiling rose to 25.5% but only ~1.4 points
converted, because the gists retained turn **ids** while discarding turn
**content**. The cause was the compression rule itself. It was head-truncation
— keep the first N words — which on natural dialogue is close to the worst
possible choice:

```
"[Caroline] Hey Mel! Good to see you! How have you been?
 I finally signed up for the LGBTQ support group on 7 May 2023."

head-truncation (14 tok):  "[Caroline] Hey Mel! Good to see you! How"
informative    (14 tok):   "[Caroline] Mel! Good 7 May 2023."
```

The head of a turn is greeting and phatic filler; the durable content — dates,
names, numbers, places — sits later. Head-truncation spent the entire budget on
the least informative part of every turn. `7 May 2023` **is** the answer to
"When did Caroline go to the LGBTQ support group?", and truncation threw it away
while faithfully keeping "Hey Mel!".

`memgate/compress.py` scores each word (digits +4, month names +3, mid-sentence
capitalisation +2.5, stopwords −2, with a mild preference for earlier words to
break ties) and keeps the highest-scoring ones **in original order**, always
retaining the `[Speaker]` tag. It is the project's own rate–distortion question
applied at word level: given a token budget, which words preserve the most
downstream utility?

This is **extractive, not abstractive** — it selects, it does not rewrite.
Abstractive re-summarisation by the base LLM remains the Phase 2 target and
needs a generative model that is not available offline (no API key, no local
model). What this establishes is the **extractive baseline that an abstractive
summariser must beat to justify its cost**, which turns Phase 2 into a
measurable experiment rather than an assumption.

### Result — FINAL (MiniLM + tiktoken, all fixes in)

| Policy | Budget | Strict | Soft | Answer | Avg tokens |
|---|---|---|---|---|---|
| Full-context | unlimited | 100.0% | 100.0% | 100.0% | 18 959 |
| Oracle | 512 | 99.9% | 100.0% | 100.0% | 65 |
| P0 sliding window | 512 | 1.6% | 1.9% | 4.9% | 509 |
| **P1 MemGate** | 512 | **8.0%** | **9.8%** | **12.9%** | 484 |
| P0 sliding window | 1024 | 4.3% | 4.9% | 9.1% | 1022 |
| **P1 MemGate** | 1024 | **10.6%** | **12.8%** | **16.9%** | 900 |
| P0 sliding window | 2048 | 10.1% | 11.4% | 18.3% | 2047 |
| **P1 MemGate** | 2048 | **14.4%** | **17.3%** | **20.6%** | 1423 |
| P0 sliding window | 4096 | 18.8% | 21.8% | **31.3%** | 4094 |
| **P1 MemGate** | 4096 | **20.3%** | **24.6%** | 25.0% | 2051 |

**MemGate now beats the sliding window on strict recall at EVERY budget,
including 4096 (20.3% vs 18.8%) on half the tokens.** Every earlier version lost
that budget outright. P0 still wins answer recall at 4096 (31.3% vs 25.0%) —
with 4094 tokens it simply carries more raw text, and that is worth stating
plainly.

Progression of P1 strict recall across this session:

| Budget | leak-free start | + consolidation | + compressor |
|---|---|---|---|
| 512 | 7.0% | 7.7% | **8.0%** |
| 1024 | 8.5% | 9.4% | **10.6%** |
| 2048 | 11.0% | 12.4% | **14.4%** |
| 4096 | 15.3% | 16.6% | **20.3%** |

**+5.0 points at 4096 from the two write-path fixes.** The gain grows with
budget, which is the signature of a write-path fix: retained content only pays
off once there is room to put it in.

Per-category at 2048: temporal **17.2%** vs 9.4%, single-hop **17.9%** vs 13.2%,
open-domain **12.4%** vs 7.9%. Multi-hop remains the one loss, 1.4% vs 2.2% — it
needs *all* ~3.13 cited turns, so it is the last thing to move and the sharpest
remaining test of the write path.

## 22. Ablation study (Step 4) — and the uncomfortable result

`run_ablations.py` turns off exactly one component at a time and reports the
delta against the full system on LoCoMo at a 2048-token budget. Three families:
architecture, the eight scoring signals, and parameter sensitivity.

### 22.1 What each component is worth

MiniLM + tiktoken, budget 2048:

| Removed | Strict | Δ |
|---|---|---|
| *(full system)* | 14.4% | — |
| working memory (Tier 2) | 7.1% | **−7.3** |
| retrieval (Tier 3) | 7.3% | **−7.1** |
| consolidation merge | 12.4% | −2.0 |
| informative compressor | 12.4% | −2.0 |
| density packing | 12.6% | −1.8 |
| supersession | 14.3% | **−0.1** |

| Signal removed | Δ | | Signal removed | Δ |
|---|---|---|---|---|
| durable | **−3.9** | | filler | −0.3 |
| numeric | **−3.5** | | question | −0.1 |
| temporal | −2.4 | | update | −0.1 |
| proper | −2.2 | | speaker | **+0.0** |

`chunk_tokens` (32/64/128) makes no difference at all, and buffer `N` is nearly
flat (N=12 is +0.5). Those knobs are not where the behaviour lives.

Two components earn nothing on this benchmark. **Supersession** costs 0.0 —
LoCoMo has almost no explicit corrections, so the mechanism never fires (it did
earn its place on the synthetic set, §16.1). **The speaker down-weight** costs
0.0 because both LoCoMo speakers are humans, so the `speaker == "assistant"`
branch is dead. Both are honest "this signal is not doing anything here"
results, not bugs.

### 22.2 The finding that matters — the tiering does not earn its place

Parameter sensitivity turned up an effect far larger than any component:

| `tau_fact` | Strict (MiniLM) | Answer (MiniLM) | Strict (hash) | Tokens |
|---|---|---|---|---|
| **0.45** *(default)* | 14.4% | 20.6% | 14.1% | 1423 |
| 0.30 | 29.1% | 40.7% | 22.5% | 1690 |
| 0.20 | 57.7% | 70.6% | 39.2% | 1692 |
| **0.15** | **60.2%** | **71.2%** | 40.5% | 1693 |

**+45.8 points** from one untuned threshold — for reference, P0 at this budget
scores 10.1% and full-context scores 100% at 18 959 tokens.

Nearly **triple the recall at the same token cost**, from one threshold that was
set by intuition in Step 0 and never tuned.

But the reason is not that 0.15 is a better threshold. At `tau_fact = tau_low`
the Tier 2 branch can never fire, and the routing confirms it:

```
tau_fact=0.45 -> {long: 14,  working: 399, merges: 294}
tau_fact=0.15 -> {long: 413, working: 0,   merges: 0}
```

**Every item goes straight to Tier 3. Nothing is compressed, merged, or
dropped.** The 40.5% is plain retrieval over the entire conversation — a RAG
baseline, with the tiering machinery bypassed entirely. Three checks confirm it:

| Config @ 2048 (MiniLM) | Strict | Answer |
|---|---|---|
| MemGate `tau=0.15` | 60.2% | 71.2% |
| **store-everything (`tau=0.0`)** | **59.2%** | 69.8% |
| …without retrieval | **9.0%** | 8.8% |
| …without merging | 59.1% | 70.5% |
| …without compressor | 59.3% | 69.4% |

Retrieval contributes **~51 points**. Merging and compression together
contribute ~1. And "keep everything, retrieve top-k" scores within one point of
the tuned system.

### 22.2a This corrects §15.1 about the embedder

§15.1 concluded the MiniLM swap "buys almost nothing" (+0.2 at 2048). That was
true **and the reasoning was right for the regime it was measured in** —
retrieval was saturated against a 9.4% write-path ceiling, so a better embedder
had nothing left to recover.

Once the ceiling is removed, the same swap is worth **+19.7 points**
(hash 40.5% → MiniLM 60.2%). The embedder was never weak; its contribution was
masked by a write path that discarded 93% of the conversation before retrieval
ever saw it.

The general lesson is worth keeping for the report: **a component's measured
value is conditional on the bottleneck being elsewhere.** Ablations run against
a broken pipeline will systematically undervalue exactly the components that
would matter once it is fixed.

**Stated plainly: on LoCoMo, at a 2048-token context budget, the compression
machinery this project is about is a net loss against simply keeping everything
and retrieving.** All of §20 and §21's hard-won gains (+1.4, +3.6) are gains
*within a regime that should not have been entered*.

### 22.3 Why — and the methodological gap it exposes

The experiment constrains the **context** but never the **store**. Full-context
is charged 18 959 tokens per query; "store everything and retrieve" is charged
nothing for holding all 413 turns. Under that accounting, forgetting can only
ever lose information — there is no budget it saves.

A LoCoMo conversation is ~19K tokens. A 19K-token memory is free. Compression
cannot pay for itself at that scale, and our own ablation proves it.

This does not sink the project; it locates it. Keep/compress/forget only has
value where the store itself is bounded — and that constraint was never
imposed. **The fix is a storage budget**: cap the store at S tokens, sweep S,
and measure accuracy per *stored* token as well as per context token. That is
the axis on which compression must win, and if it does not win there either,
that is a publishable negative result about tiered memory.

**Do not present the tiered architecture as validated.** Present the frontier
honestly: at LoCoMo scale, retrieval dominates; the open question is where the
crossover lies as the store is squeezed.

## 18. How to run (updated)

```bash
cd ~/Documents/Capstone\ Project/memgate
python3 test_memgate.py       # 35 passed, 0 failed (34 under MEMGATE_EMBEDDER=hash)
python3 run_experiment.py     # synthetic
python3 run_locomo.py         # LoCoMo — the real benchmark

MEMGATE_EMBEDDER=hash python3 run_locomo.py    # embedder ablation
```

Data: `data/locomo/locomo10.json` (2.7 MB, from `snap-research/locomo`).

---

# Session 3 — 24 August 2026

## 23. The storage budget — and what it says about compression

§22.3 ended with the project's own ablation showing the compression machinery
to be a **net loss** against simply keeping everything and retrieving, and
located the cause precisely: the experiment constrained the **context** but
never the **store**. Full-context was charged 18 959 tokens per query while
"store everything and retrieve top-k" was charged *nothing* for holding all 419
turns. Under that accounting forgetting can only lose information — there is no
budget it saves — so compression cannot pay for itself and the ablation was
right to say so.

This session imposes the missing constraint and re-asks the question.

### 23.1 What was built

* **`store_budget` (S)** — a cap on total tokens held across all three tiers,
  enforced on every write (`ThreeTierStore.enforce_store_budget`). Cheapest
  loss first: reclaim superseded records → demote the lowest-utility Tier 3
  fact to a Tier 2 gist → drop from Tier 2. Tier 1 is never evicted.
* **`stored_tokens()`** on every policy, recorded in every results row, so
  accuracy per **stored** token is reportable alongside accuracy per context
  token.
* **`RAG store-all`** — §22.2's "keep everything and retrieve" configuration
  promoted from a footnote to a first-class baseline, because it is the thing
  compression has to beat. FIFO eviction under the cap: it has no scorer, which
  is the point.
* **Two new MemGate variants** (defaults untouched, so every §15–§22 number
  still reproduces exactly — verified):
  * **P1-A adaptive** — store verbatim while there is room; **compress on
    eviction**, so the compression rate follows storage pressure instead of
    being fixed at ingest.
  * **P1-S select** — identical, except an evicted item is **dropped rather
    than compressed**.
* **`run_storage_sweep.py`**, and `--store-budget` / `--adaptive` on
  `run_ablations.py` so the whole §22 battery re-runs under the corrected
  accounting.

The three retention regimes isolate one variable each at a fixed store size S:

| comparison | held constant | varies |
|---|---|---|
| P0 vs RAG | retention (same turns, same size) | **read path** — recency vs retrieval |
| RAG vs P1-S | storage, fidelity, retrieval | **which turns are forgotten** |
| P1-S vs P1-A | storage, selection | **fidelity** — drop vs compress |

### 23.2 Two under-fill bugs, both the §13.5 strawman pointed the other way

The first sweep showed MemGate **flatlining at 2148 stored tokens on a
16 384-token allowance** — it lost at every large S while refusing to use 87% of
the storage it was given. Same class of defect as §13.5, where MemGate could
not fill the *context* it was allowed; here it could not fill the *store*.

1. **Tier 2 was sized from the context budget**, not the store
   (`budget × split × 2` = 1024), so no storage allowance could make it grow.
2. **Tier 2 is not query-addressable.** Routing sends 399 of 413 items there,
   but Tier 2 is read *positionally* — packed by density — never ranked against
   the question. Only Tier 3 is retrievable, and it held 14 items.

(2) is the more interesting one, and it produced the session's sharpest
diagnostic: enlarging Tier 2 alone made recall **worse** (24.2% → 12.1% at
S=16 384). More stored, none of it reachable. **Stored content that is not
addressable earns nothing**, and a storage budget spent on it is worse than
wasted. Tier 2 items already carried embeddings, so making them retrievable
cost only the ranking.

A third, deeper limit remained: threshold mode compresses every turn to a
~14-word gist *on the way in*, so it cannot hold more than ~4735 tokens of a
19K conversation however large the budget. The compression rate was fixed in
advance rather than adapted to pressure — which is what P1-A exists to fix.

### 23.3 Result — LoCoMo, context budget fixed at 2048, store budget swept

MiniLM + tiktoken, 10 conversations, 1527 questions. `stored` is what the
policy still holds after ingest; `ctx` is what it spends per query.

| S | Policy | Strict | **Answer** | Stored | Ctx |
|---|---|---|---|---|---|
| 1024 | RAG store-all | 4.3% | 9.1% | 1005 | 987 |
| 1024 | P1-A adaptive | **9.1%** | 10.9% | 1002 | 984 |
| 1024 | **P1-S select** | 5.6% | **15.7%** | 1006 | 963 |
| 2048 | RAG store-all | 10.1% | 18.3% | 2031 | 1983 |
| 2048 | P1-A adaptive | **19.6%** | 23.4% | 2022 | 1758 |
| 2048 | **P1-S select** | 10.9% | **26.3%** | 2025 | 1932 |
| 4096 | RAG store-all | 17.8% | 28.1% | 4068 | 2018 |
| 4096 | P1-A adaptive | **30.5%** | 30.1% | 4042 | 1470 |
| 4096 | **P1-S select** | 19.8% | **36.8%** | 4077 | 2006 |

Delta against RAG store-all **at the same storage budget**:

| S | P1 threshold | P1-A adaptive | **P1-S select** |
| | strict / answer | strict / answer | strict / answer |
|---|---|---|---|
| 512 | +1.0 / +1.0 | +2.8 / +0.6 | +0.2 / **+2.5** |
| 1024 | +1.8 / +2.4 | +4.8 / +1.8 | +1.2 / **+6.6** |
| 2048 | +2.2 / −0.3 | +9.5 / +5.0 | +0.8 / **+8.0** |
| 4096 | −3.7 / −7.8 | +12.7 / +2.0 | +2.0 / **+8.7** |
| 8192 | −16.2 / −23.8 | +12.9 / **−6.9** | +0.7 / **+3.9** |
| 16384 | −38.9 / −45.9 | +5.0 / +0.0 | +5.4 / **+5.9** |
| unlim | −47.4 / −54.3 | +0.5 / +0.4 | +0.5 / +0.4 |

**§22.2's verdict is reversed, but not in the way the project expected.**

### 23.4 The finding: selection wins, compression loses

Read the two metrics against each other, as §19 requires. **P1-A wins strict
recall and loses answer recall** — at S=8192, +12.9 strict but **−6.9 answer**.
A demoted gist keeps the evidence ids of the turn it came from while the answer
text has been compressed away, so evidence recall credits it and
answer-presence refuses to. This is exactly the failure mode the answer metric
was built for in §19, and it fires here on the project's own headline variant.

Turning demotion **off** makes it unambiguous. Same storage, same selection,
the only change being drop-instead-of-compress:

| S | RAG answer | P1-A (compress) | **P1-S (drop)** |
|---|---|---|---|
| 1024 | 9.1% | 10.9% | **15.7%** |
| 2048 | 18.3% | 23.4% | **26.3%** |
| 4096 | 28.1% | 30.1% | **36.8%** |
| 8192 | 44.3% | 37.5% | **48.3%** |
| 16384 | 66.4% | 66.4% | **72.3%** |

**Compression is a net negative on the honest metric at every storage budget
tested.** What earns the gain is (a) imposing the cap at all, and (b) *scored*
eviction — choosing which turns to keep whole.

That second point is the cleanest result the project has produced. **RAG and
P1-S hold the same verbatim turns, in the same space, retrieved the same way,
and differ in one respect only: which turns are forgotten when the cap binds.**
RAG forgets oldest-first; P1-S forgets lowest-utility-first. The gap between
them — **+8.7 answer points at S=4096** — is the decision policy and nothing
else. That is precisely the quantity §4 set out to measure.

**The rate–distortion reading:** at LoCoMo dialogue scale the optimum sits at
the **vertex** — a *subset at full fidelity* beats *everything at reduced
fidelity*. Compression should only begin to pay once the store is squeezed far
enough that even the selected subset will not fit verbatim. The sweep bottoms
out at S=512 before reaching that regime (all policies are near the floor
there), so **where that second crossover lies is now the open question**, and
it needs either a tighter S or a longer conversation than LoCoMo provides.

### 23.5 Ablation under the corrected accounting

`run_ablations.py --budget 2048 --store-budget 4096 --adaptive`:

| Removed | Strict | Δ | Answer |
|---|---|---|---|
| *(full system)* | 30.5% | — | 30.1% |
| retrieval (Tier 3) | 0.2% | **−30.3** | 2.4% |
| demotion on pressure | 19.8% | −10.7 | **36.8** ↑ |
| scored eviction (→ FIFO) | 25.5% | **−5.0** | 23.4% |
| informative compressor | 26.7% | −3.8 | 27.8% |
| consolidation merge | 28.0% | −2.6 | 25.0% |
| supersession | 30.3% | −0.2 | 29.9% |
| density packing | 30.5% | +0.0 | 30.1% |
| working memory (Tier 2) | 30.5% | +0.0 | 30.1% |

Three things to state plainly:

* **Retrieval still dominates everything** (−30.3), as in §22.2. That has been
  true in every regime measured.
* **Scored eviction earns −5.0 strict and −6.7 answer** against FIFO. The
  decision engine is doing real work — but only where the budget binds. At
  S=8192 FIFO actually *beat* it by 3.4 strict, i.e. under light pressure
  recency is the better forgetting rule.
* **Density packing and Tier 2 read +0.0 because adaptive mode bypasses that
  path entirely** — the positional Tier 2 share is replaced by one ranked pool.
  They are no-ops here, not free components. Their §22.1 numbers still stand
  for threshold mode.

### 23.6 Guards added

`test_memgate.py` is at **57 passed, 0 failed**. New:

* no policy ever exceeds its storage budget (P0 / RAG / P1 / P1-A, four values
  of S each);
* **the label-invariance test now covers eviction and demotion** in both write
  modes. The storage budget introduced two new decision points — *which item to
  evict* and *which to demote* — and either could have read the answer key the
  way consolidation did in §13. Eviction ranks on `utility` and recency only;
  `_forget_key` is documented as label-blind and pinned by the test.
* a tight budget must force real loss, and adaptive mode must demote **nothing**
  when storage is free and something when it is not — otherwise the sweep would
  be measuring a constraint that never binds.

`count_tokens` is now memoised (pure function of its input); the cap re-counts
held items on every write, which is otherwise O(n²) per conversation.

### 23.7 Where this leaves the project

The Review-2 story is now a real experimental narrative rather than a claim:

1. Tiered memory with forgetting is already shipped by Letta/Mem0/Zep (§3, Q3).
2. Measured honestly, a well-formed three-tier system scores **14.4%** on
   LoCoMo where full-context scores 100% (§21).
3. Its own ablation then showed the compression machinery to be a **net loss**
   against keeping everything (§22.2) — and correctly diagnosed why: storage
   was never budgeted (§22.3).
4. With the store budgeted, **the decision policy is worth +8.7 answer points
   over forgetting oldest-first, at identical storage and identical fidelity**
   — but **compression itself still loses**. Keep a scored subset whole; do not
   compress everything.

That is a sharper and more defensible contribution than "we built a memory
layer", and (4) is a genuine negative result about tiered compression at this
scale, arrived at through the project's own metrics.

**Do not present tiered compression as validated.** Present the storage-budget
frontier: selection pays, compression does not (yet), and the crossover where
it should is the next thing to find.

### 23.8 Immediate next steps

1. **Push S below 512 and/or use a longer benchmark** to find the second
   crossover — the point where the selected subset no longer fits verbatim and
   compression must start paying. LongMemEval is the natural candidate; its
   conversations are far longer than LoCoMo's ~19K tokens.
2. **P2 (LLM-judge)** is now the highest-value scorer work, and its value is
   quantified in advance: scored eviction is worth −5.0 against FIFO, so a
   better scorer attacks a component that is *already demonstrably load-bearing*
   — unlike §15.1's embedder, which was masked by a broken write path.
3. **Charge for the embeddings.** `stored_tokens` counts text only; a 384-d
   float32 vector per item is real storage that RAG and P1-S pay on every
   retained turn and a compressed gist pays once. Under a byte-denominated
   budget the comparison may shift.
4. Report **accuracy per stored token** as a headline axis alongside accuracy
   per context token — the sweep already emits it (`strict_per_kstored`).

### 23.9 How to run (updated)

```bash
cd ~/Documents/Capstone\ Project/memgate
python3 test_memgate.py            # 57 passed, 0 failed
python3 run_locomo.py              # §21 table, unchanged
python3 run_storage_sweep.py       # §23 — the storage-budget frontier
python3 run_ablations.py --budget 2048 --store-budget 4096 --adaptive
```

Results: `results/storage_sweep.csv` (adds `stored_tokens`, `max_stored`,
`strict_per_kstored`).

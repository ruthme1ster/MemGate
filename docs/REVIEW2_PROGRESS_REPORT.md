# MemGate — Capstone Progress Report

**Prepared for the faculty progress review (pre–Review 2)**

| | |
|---|---|
| **Project** | MemGate: An Adaptive Memory Layer for Long-Running LLM Agents |
| **Students** | Simar Singh Khanuja (70562200088) · Yash Ramchandani (70562300097) |
| **Institution** | SVKM's NMIMS, Indore Campus |
| **Report date** | 8 September 2026 |
| **Covers** | Review 1 (25 July 2026) → present |
| **Working sessions** | 12, 17, 24, 27 August 2026 |
| **Repository state** | commit `34df12e`, 61 tests passing, 5,430 lines of Python |
| **Companion documents** | `REPORT.md` (paper-ready writeup) · `SESSION_RECORD.md` (full working record) · `frontend/index.html` (interactive results dashboard) |

---

## 1. Status at a glance

**The measurement track is complete.** Every claim the project makes is produced
by a script in this repository, carries a 95% confidence interval clustered by
conversation, and is pinned by a regression test. Nine experiments run end to
end and their outputs are committed as CSVs.

| Workstream | State | Evidence |
|---|---|---|
| Harness, three-tier store, policies P0/P1 | **Complete** | `memgate/`, 5,430 lines |
| LoCoMo benchmark wired, evaluation leak removed | **Complete** | `run_locomo.py`, 3 label-blindness tests |
| Ablation study (23 configurations) | **Complete** | `results/ablations_2048.csv` |
| Storage budget — the corrected accounting | **Complete** | `results/storage_sweep.csv`, 35 configurations |
| Statistical significance (bootstrap + McNemar) | **Complete** | `results/significance.csv` |
| Scaling to 182× compression | **Complete** | `results/scaling.csv` |
| Byte accounting (the embedding index charged for) | **Complete** | `results/cost_model.csv` |
| Learned scorer, leave-one-conversation-out | **Complete** | `results/learned_scorer.csv` |
| P2 LLM judge, built **and evaluated** | **Complete** | `results/judge_eval.csv` |
| Publication figures | **Complete** | `results/figures/fig1–fig5` |
| End-task accuracy through a real reader | **Pilot only** | pipeline + cache built; 12-question smoke run |
| Abstractive compression | Not started | scoped in §9 |
| LongMemEval (genuinely long conversations) | Not started | scoped in §9 |
| Online learning of the eviction policy | Not started | scoped in §9 |

### The headline, in one sentence

Under a bounded store, **choosing which turns to forget is worth +8.7
answer-recall points** over forgetting oldest-first — at identical storage,
identical fidelity and identical retrieval, so the gap is the decision policy
and nothing else (95% CI [+4.8, +14.3] clustered by conversation, McNemar
p = 1.1×10⁻⁴). **Compressing an evicted turn instead of dropping it loses 6.7
points.**

---

## 2. What changed since Review 1

Review 1 presented a design and a synthetic proof-of-concept. Three things
changed materially since, and the panel should hear all three.

### 2.1 A Review 1 number was withdrawn

The Review 1 deck reported **95% recall for MemGate against 10% for a sliding
window** on the synthetic set. That number was inflated by an **evaluation
leak**: the memory policy was reading `fact_id`, which is the ground-truth
answer key, when deciding what to keep. The honest number for the same
configuration is **50%**.

| Synthetic set @ 800 tokens | As presented at Review 1 | Honest |
|---|---|---|
| P1 MemGate | 95.0% | **50.0%** |
| P0 sliding window | 10.0% | 10.0% |

The leak was worth 45–47 recall points. It is fixed, and it is now pinned by a
regression test rather than a comment: strip every evaluation label from the
input and the store must come out **bit-identical**. Each new decision point
added since — eviction under a storage budget, and demotion — was added to that
test *before* any number from it was quoted. This is reported here in full
because a withdrawn result that a panel discovers is worse than one the students
raise themselves.

### 2.2 The project moved from a synthetic set to a real benchmark

Review 1's evidence was a generated conversation. The project now runs on
**LoCoMo** (Maharana et al., 2024): 10 real multi-session conversations, 5,882
turns, 1,527 scoreable questions. The synthetic set is retained, but only as a
sanity harness that proves the pipeline and the metrics are correct — it is no
longer quoted as a result.

### 2.3 The claim itself changed, twice, and each change is measured

* Under the accounting used throughout this literature — context bounded,
  storage unbounded — our own ablation found the **tiering machinery to be a net
  loss** against simply keeping every turn and retrieving. That result was
  correct, and it is reported.
* Imposing the missing **storage budget** inverted it: selection now wins
  decisively, while compression still loses. The project's contribution is
  partly this correction to how the comparison is set up.
* A claim that did not survive the statistics is reported as such: the
  select-vs-RAG gain on *evidence* recall (+1.96 points) does **not** survive
  clustering (p = 0.16) and is **not** claimed. Only the answer-recall claim
  stands.

---

## 3. What the system is

MemGate sits between an agent and a **frozen** LLM and controls what enters the
prompt. Nothing is trained; the base model is a black box.

```
Agent
  │
  ▼
MemGate ─── Adaptive Memory Decision Engine   scores each evicted turn
        ├── Compressor / Summariser           tier-2 gists
        ├── Retriever                         semantic search over tier 3
        └── Context Assembler                 packs to the token budget
  │
  ├── Tier 1  short-term   raw recent turns        criterion: WHEN
  ├── Tier 2  working      compressed gists        criterion: GIST
  └── Tier 3  long-term    atomic facts + vectors  criterion: WHAT
  │
  ▼
Frozen LLM
```

The tiers are a **lifecycle, not a partition**. Everything enters Tier 1; the
routing decision happens once, on eviction, when the budget binds:

```
u ≥ τ_fact   → Tier 3 (durable fact)
u ≥ τ_low    → Tier 2 (compressed gist)
otherwise    → dropped
```

### Policies compared

| | Policy | Retention rule | Fidelity |
|---|---|---|---|
| P0 | Sliding window | recency, budget-driven | verbatim |
| RAG | Store-all + retrieve | FIFO — the *no-policy* control | verbatim |
| P1 | MemGate threshold | scored at ingest, τ-routed | compressed |
| **P1-A** | MemGate adaptive | score orders eviction | **compress** on eviction |
| **P1-S** | MemGate select | score orders eviction | **drop** on eviction |
| P2 | LLM-judged salience | Qwen2.5-0.5B rates each turn | drop |
| P3 | Learned scorer | fitted, cross-validated | drop |
| Oracle | — | perfect selection (upper bound) | verbatim |

Each comparison is constructed to isolate exactly **one** variable:

| Comparison | Held constant | Varies |
|---|---|---|
| P0 vs RAG | retention, storage | **read path** — recency vs retrieval |
| RAG vs P1-S | storage, fidelity, retrieval | **which turns are forgotten** |
| P1-S vs P1-A | storage, selection | **fidelity** — drop vs compress |

### Metrics, and why there are two

* **Evidence (strict) recall** — every gold evidence turn reached the context.
* **Answer recall** — the answer *text* survived, over the 715 questions whose
  answer is recoverable from the evidence text. **This is the metric to
  believe.**

The two metrics *disagree* on exactly the comparison the project is about, and
that disagreement is a finding, not an inconvenience: evidence recall credits a
compressed gist for the ids of a turn whose text it discarded; answer recall
checks the content.

---

## 4. Results completed

All figures below are read from committed CSVs in `memgate/results/`.

### 4.1 Selection pays; compression does not (§3.2)

Context budget fixed at 2048 tokens, storage budget swept. **Answer recall:**

| S (tokens) | RAG store-all | P1-A adaptive *(compress)* | **P1-S select** *(drop)* |
|---|---|---|---|
| 1024 | 9.1% | 10.9% | **15.7%** |
| 2048 | 18.3% | 23.4% | **26.3%** |
| 4096 | 28.1% | 30.1% | **36.8%** |
| 8192 | 44.3% | 37.5% | **48.3%** |
| 16384 | 66.4% | 66.4% | **72.3%** |

P1-A *wins* evidence recall (+12.9 at S=8192) while *losing* answer recall
(−6.9). Reporting only the former would have shown compression winning.

### 4.2 Statistical significance (§3.3)

Paired bootstrap, B = 10,000, resampled **by conversation**, with exact McNemar.

| Comparison | Metric | Δ | 95% CI (clustered) | McNemar p | |
|---|---|---|---|---|---|
| P1-S vs RAG | **answer** | **+8.67** | **[+4.77, +14.27]** | 1.1×10⁻⁴ | significant |
| P1-S vs RAG | evidence | +1.96 | [−1.48, +5.52] | 0.16 | **not significant** |
| P1-S vs P1-A | answer | +6.71 | [+4.24, +9.54] | 3.6×10⁻⁵ | significant |
| P1-S vs P1-A | evidence | −10.74 | [−13.65, −7.90] | 4.7×10⁻²⁰ | significant |
| P1-S vs P0 | answer | +18.46 | [+14.68, +23.56] | 3.8×10⁻¹⁷ | significant |
| P1-S vs P0 | evidence | +9.69 | [+6.85, +12.83] | 3.6×10⁻¹⁴ | significant |

### 4.3 The rate–distortion prediction fails (§3.4)

Theory predicts compression must eventually beat selection. Holding S fixed and
concatenating up to 10 conversations (186K tokens) to raise the compression
ratio, **answer recall at S = 2048:**

| Ratio | RAG | P1-A *(compress)* | **P1-S** *(drop)* |
|---|---|---|---|
| 9× | 18.3% | 23.4% | **26.3%** |
| 18× | 10.8% | 14.7% | **18.2%** |
| 45× | 7.8% | 11.2% | **11.7%** |
| 91× | 4.6% | 6.0% | **8.7%** |

**No crossover to 182×**, and the selection margin over RAG *widens* with
pressure.

### 4.4 The index is not free (§3.5)

A 384-d float32 vector is 1,536 B against a turn's ~130 B of text — the index
outweighs the content ~12×. **Answer recall under byte accounting:**

| Budget | P0 *(no index)* | RAG | P1-S select |
|---|---|---|---|
| 32 KiB | **18.3%** | 5.0% | 15.1% |
| 64 KiB | 18.3% | 11.0% | **22.5%** |
| 256 KiB | 18.3% | 33.0% | **39.0%** |

The selection finding is unit-robust. And **below ~40 KiB an embedding index
cannot earn its own storage** — a policy with no index at all wins outright, a
regime that is invisible when the budget counts text alone.

### 4.5 The scorer is the bottleneck, and it is learnable (§3.6)

Ablation at S = 4096, evidence recall:

| Component removed | Δ |
|---|---|
| retrieval (Tier 3) | **−30.3** |
| demotion on pressure | −10.7 *(but answer recall **+6.7**)* |
| scored eviction → FIFO | **−5.0** |
| informative compressor | −3.8 |
| consolidation merge | −2.6 |
| supersession | −0.2 |

Replacing the hand-chosen weights with a fitted model, under
leave-one-conversation-out (S = 2048):

| Scorer | AUC | Answer recall |
|---|---|---|
| P1 heuristic (hand-tuned) | — | 26.3% |
| P3 learned, hand features | 0.746 | 27.7% |
| **P3 learned, + MiniLM** | **0.793** | **31.5%** |

The fitted weights say something the heuristic never encoded: **turn length is
the strongest single predictor of evidence-worthiness.**

### 4.6 An LLM judge is the wrong instrument for salience (§3.7)

A single-component swap — policy, storage budget, retrieval and compression
pinned at the §3.2 winner, **only the scorer varying**:

| Scorer | Evidence | Answer | Δ vs P1 | 95% CI (clustered) |
|---|---|---|---|---|
| P0 recency (no content judgement) | 12.6% | 25.6% | −11.19 | [−17.24, −6.19] |
| **P1 heuristic** (regex + shallow NER) | 19.8% | **36.8%** | — | — |
| P2 LLM judge (Qwen2.5-0.5B) | 14.0% | 30.9% | **−5.87** | [−10.77, −1.76] |
| P3 learned (LOCO, bound) | 33.4% | 48.3% | +11.47 | [+4.90, +17.86] |

**The judge loses to the regex by 5.87 points and the interval excludes zero** —
a real negative result, for ~15 minutes of GPU per corpus against microseconds.
The diagnosis is what the judge is *shown*: it rates each turn **in isolation**,
and salience is not a property of a turn on its own. A phone number matters
because something later asks for it.

### 4.7 End-task accuracy — pipeline built, **pilot only**

The single most valuable remaining experiment. Six arms through one fixed reader
(Qwen2.5-1.5B-Instruct-4bit, greedy, 32 new tokens), scored with the same
normaliser as the answer-presence metric, with two controls the comparison
cannot be read without: **closed-book** (the floor every arm must beat to have
contributed anything) and **oracle** (the ceiling).

| Arm | Accuracy | F1 | Answer present |
|---|---|---|---|
| closed-book | 8.3% | 12.5 | 0% |
| P0 sliding-window | 8.3% | 9.7 | 0% |
| RAG store-all | 0.0% | 12.5 | 0% |
| P1-A adaptive | 25.0% | 25.3 | 33.3% |
| **P1-S select** | **33.3%** | **44.7** | 41.7% |
| Oracle | 66.7% | 76.9 | 100% |

> **This is a 12-question smoke run, not a result.** It is committed because the
> pipeline, the two controls and the generation cache are what matter, and they
> work end to end. It is far too small to support any claim. The full
> 715-question run is several hours of local generation and has not been made.

---

## 5. Findings about measurement

Three of the project's most useful results are about *measurement itself*, and
each was caught by a guard rather than by inspection. They are worth presenting
because they are transferable — any group evaluating a memory system can hit
all three.

| | Finding | What it cost |
|---|---|---|
| **5.1** | **A policy that read its own answer key.** An early consolidation rule promoted items on `fact_id`, the evaluation label. | +45–47 recall points. Invalidated the Review 1 headline. |
| **5.2** | **A component's measured value is conditional on the bottleneck being elsewhere.** The MiniLM embedder measured at +0.2 points — true in the regime measured, where retrieval was saturated against a write path discarding 93% of the conversation. Once that ceiling lifted, the same swap was worth **+19.7**. | Ablations run against a broken pipeline systematically undervalue exactly the components that would matter once it is fixed. |
| **5.3** | **A metric that credits pointers.** Evidence recall counts a turn as retrieved if its *id* reaches the context; a gist keeps the id and discards the text. Evidence recall once rose 8.1% → 14.8% while answer recall stayed pinned at 12.8%. | The entire gain was bookkeeping. Answer-presence scoring is what separated selection from compression. |

A fourth, smaller instance: batched judge scoring returned `NaN` for exactly the
short turns. The instinct was numerical precision, and float32 did not fix it;
the cause was **left-padding creating fully-masked attention rows**. Without the
NaN guard, every filler turn would have scored NaN and quietly inverted the
eviction order.

---

## 6. Technology used

### 6.1 What was actually used

| Layer | Component | Version / detail | Role in the project |
|---|---|---|---|
| **Language** | Python | 3.10 | Entire implementation; no framework dependency |
| | NumPy | — | Vector maths, cosine retrieval, bootstrap resampling |
| | `dataclasses`, `unittest` | stdlib | Typed records; the 61-test suite |
| **Tokenisation** | `tiktoken` | `cl100k_base` | Exact token accounting — budgets are real tokens, not a word-count proxy |
| **Embeddings** | `sentence-transformers` | all-MiniLM-L6-v2, 384-d | Retrieval ranking and semantic supersession |
| | PyTorch | MPS backend | Local inference on Apple silicon |
| **Language models** | Qwen2.5-0.5B-Instruct | local weights | The P2 salience judge — read as a continuous expectation over digit-token logits at one forward pass |
| | Qwen2.5-1.5B-Instruct-4bit | local weights | The end-task reader, greedy decoding |
| **Benchmark** | LoCoMo | Maharana et al., 2024 | 10 conversations, 5,882 turns, 1,527 scoreable questions |
| **Statistics** | Paired bootstrap | B = 10,000 | Confidence intervals, resampled by conversation |
| | Exact McNemar | — | Paired discordant-pair test over the same questions |
| | scikit-learn | logistic regression | The P3 learned scorer, leave-one-conversation-out |
| **Figures** | matplotlib | 3.10 | fig1–fig5 (PNG + PDF) |
| **Deliverables** | python-pptx | 1.0.2 | Generates the Review 2 deck from the committed CSVs |
| | HTML / SVG / vanilla JS | no framework | The interactive results dashboard (`frontend/`) |
| **Discipline** | `HF_HUB_OFFLINE` | set on import | A run that can silently fetch a model is a run whose inputs are not pinned |
| | Generation caches | keyed by (model, prompt) hash | Judge and reader runs are resumable and deterministic |
| | git | — | Every results CSV committed alongside the code that produced it |

### 6.2 Proposed at Review 1 versus used — and why

| Area | Proposed (Review 1) | Actually used | Why it changed |
|---|---|---|---|
| Base model | Llama 3 / Mistral via Hugging Face; OpenAI/Anthropic API for comparison | Qwen2.5-0.5B and 1.5B-4bit, **run locally** | No API key, no rate limit, no per-run cost, and bit-reproducible offline |
| Agent scaffolding | LangChain / LlamaIndex | **None** — a direct harness | A framework's own retrieval and prompt assembly would confound the exact variable under study |
| Vector store | FAISS / Chroma | NumPy cosine over an in-process store | At 10 conversations an ANN index adds approximation error, not speed |
| Evaluation | End-task F1 on a long-conversation benchmark | Context recall + answer presence, **then** end-task | Recall isolates the memory policy from the reader's reasoning, is deterministic, and is the *ceiling* on end-task accuracy |
| Hosted API | OpenAI / Anthropic for comparison | Not used | Cost, and non-determinism across a 35-configuration sweep |

This is not scope reduction — it is the opposite. Running everything locally
and free is what made **35 storage-budget configurations, 23 ablations, a
10,000-sample bootstrap and a 36-point scaling sweep** affordable to re-run on
every change.

---

## 7. Repository inventory

```
Capstone Project/
├── REPORT.md                     paper-ready writeup (the scientific document)
├── SESSION_RECORD.md             full working record, §1–§28, cold-start readable
├── docs/
│   ├── REVIEW2_PROGRESS_REPORT.md          this document
│   ├── make_review2_deck.py                builds the deck from the CSVs
│   └── MemGate_Capstone_Review2_NMIMS.pptx the Review 2 presentation
├── frontend/
│   ├── index.html                interactive results dashboard (opens offline)
│   ├── data.js                   GENERATED — every figure on the page
│   └── build_data.py             regenerates data.js from results/*.csv
└── memgate/
    ├── memgate/                  the library — 2,022 lines
    │   ├── types.py              MemoryItem, Turn, Question (multi-evidence)
    │   ├── utils.py              tiktoken + MiniLM, with fallbacks and provenance
    │   ├── store.py              three-tier store, dedupe, supersession, consolidation
    │   ├── scoring.py            P0/P1 scorers on one interface
    │   ├── policies.py           baselines + MemGate, routing, budget-aware assembly
    │   ├── compress.py           the extractive compressor
    │   ├── judge.py              the P2 LLM judge (logit-expectation scoring)
    │   ├── llm_judge.py          local model loading, offline-pinned
    │   ├── data.py               synthetic generator + LoCoMo loader + integrity assertions
    │   └── harness.py            recall metrics, per-category breakdown, sweep, CSV
    ├── run_locomo.py             context-budget frontier
    ├── run_storage_sweep.py      §3.2 storage budget — 35 configurations
    ├── run_scaling.py            §3.4 compression ratio to 182×
    ├── run_cost_model.py         §3.5 byte accounting
    ├── run_significance.py       §3.3 bootstrap + McNemar
    ├── train_scorer.py           §3.6 learned scorer, LOCO
    ├── precompute_judge.py       P2 judge scores (~15 min, cached)
    ├── run_judge_eval.py         §3.7 scorer comparison with clustered CIs
    ├── run_endtask.py            §3.8 end-task accuracy (resumable)
    ├── run_ablations.py          23-configuration component attribution
    ├── run_experiment.py         synthetic sanity set
    ├── diagnose.py               attributes every miss to one swappable component
    ├── make_figures.py           fig1–fig5
    ├── test_memgate.py           61 tests
    └── results/                  every CSV and figure quoted in this report
```

**Not in the repository, by design:** the LoCoMo benchmark (2.7 MB) and the
model weights (~1 GB). Both are reproducible from source, and
`memgate/README.md` gives the exact fetch commands. The results CSVs *are*
committed, so a reviewer can check every number in this report without a
nine-minute benchmark run.

---

## 8. Verification and engineering discipline

| Guard | What it prevents |
|---|---|
| **61 tests** (`test_memgate.py`) | Regression across the store, scorers, policies, compressor and harness |
| **3 label-blindness invariants** | The §5.1 leak, re-armed. Routing, eviction *and* demotion must produce a bit-identical store when every evaluation label is stripped |
| **Offline pinning** on import | A silently fetched or half-downloaded model relabelling a whole results table — this already happened once, with MiniLM |
| **Backend provenance** (`backend_info()`) | Reporting a MiniLM number that was actually produced by the hashed bag-of-words fallback |
| **NaN guard** in the judge | Fillers scoring NaN and inverting the eviction order |
| **Generation caches** keyed by (model, prompt) hash | Non-reproducible LLM output; also makes long runs resumable |
| **Results committed with the code** | A figure in a report that no longer matches the code that made it |

---

## 9. Limitations

Stated in full, because each one bounds a claim above.

1. **Ten conversations.** Clustered intervals are wide because the cluster count
   is small. Effects are reported with those intervals, and the one that does
   not survive clustering is reported as not significant.
2. **Context recall, not end-task accuracy.** Recall is the *ceiling* on
   accuracy. Closing this needs the full generative run (§4.7).
3. **Extractive compression only.** P1-A selects words; it does not rewrite. The
   negative result about compression is stated for the extractive case.
4. **Concatenated streams are synthetic.** In §4.3 each question concerns one
   constituent conversation, so the others act as distractors. It is a stress
   test of retention under pressure, not a long-dialogue benchmark.
5. **P3-learned is a headroom bound, not a component.** A live agent has no
   future questions to learn from.
6. **Single embedder, single judge model.** The §4.6 negative is evidence about
   *a 0.5B model prompted per turn*, not about LLM judging in general.

---

## 10. Remaining work and plan to the final review

| # | Work | Why it matters | Effort | Target |
|---|---|---|---|---|
| 1 | **Full 715-question end-task run** | Converts the whole project from a ceiling to an accuracy claim. Pipeline, controls and cache are already built and validated on 12 questions | ~4–6 h of local generation, resumable | September 2026 |
| 2 | **Abstractive compression** | The compression negative is stated for extractive methods only; a rewriting summariser may retain more content per token, and would either overturn or strengthen the finding | 1 working session | October 2026 |
| 3 | **LongMemEval** | Replaces the concatenated-stream stress test with genuinely long single conversations | 1–2 sessions | October 2026 |
| 4 | **Online learning of the eviction policy** | The only route that makes a learned scorer deployable — P3 currently needs future questions | 2 sessions | November 2026 |
| 5 | **Final report and defence** | — | — | December 2026 |

---

## 11. Conclusion as it stands

Presented as "a memory layer with three tiers", this work would already have
been done by Letta, Mem0 and Zep. Measured honestly, the more useful result is
what *doesn't* work:

1. With storage unbounded — the standard accounting — tiered compression is a
   **net loss** against keeping everything and retrieving.
2. With storage bounded, the decision policy is worth **+8.7 answer-recall
   points**, while compression still **loses 6.7**.
3. The rate–distortion crossover at which compression should pay is **not
   reachable** on this benchmark — to 182×, keeping a selected subset whole
   beats compressing everything.
4. Charging for the embedding index **inverts the ranking** below ~40 KiB.
5. An LLM judge, the obvious way to improve the scorer, **loses to a regex** —
   and the reason is diagnosable: it rates turns in isolation.

**Practical guidance: keep a scored subset whole; do not compress everything.
And denominate the budget in bytes, including the index, or the comparison is
not the one you think you are making.**

---

## 12. Reproducing every number in this report

```bash
cd memgate
python3 test_memgate.py          # 61 tests
python3 run_locomo.py            # context-budget frontier
python3 run_storage_sweep.py     # §4.1 storage budget
python3 run_significance.py      # §4.2 bootstrap + McNemar
python3 run_scaling.py           # §4.3 compression ratio to 182×
python3 run_cost_model.py        # §4.4 byte accounting
python3 train_scorer.py          # §4.5 learned scorer
python3 precompute_judge.py      # P2 judge scores (~15 min, cached)
python3 run_judge_eval.py        # §4.6 scorer comparison + CIs
python3 run_endtask.py           # §4.7 end-task accuracy (resumable)
python3 make_figures.py          # fig1–fig5

cd ..
python3 frontend/build_data.py   # refresh the dashboard from results/
python3 docs/make_review2_deck.py  # rebuild the Review 2 deck
```

Results land in `memgate/results/`; figures in `memgate/results/figures/`.
The benchmark and model weights are fetched separately — see `memgate/README.md`.

---

## 13. References

**Base paper.** Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating
Systems.* arXiv:2310.08560.

1. Zhong, W., et al. (2024). *MemoryBank: Enhancing Large Language Models with
   Long-Term Memory.* AAAI 38(17), 19724–19731.
2. Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of
   Human Behavior.* UIST '23.
3. Xiao, G., et al. (2024). *Efficient Streaming Language Models with Attention
   Sinks.* ICLR.
4. Zhang, Z., et al. (2023). *H2O: Heavy-Hitter Oracle for Efficient Generative
   Inference of Large Language Models.* NeurIPS.
5. Jiang, H., et al. (2023). *LLMLingua: Compressing Prompts for Accelerated
   Inference of Large Language Models.* EMNLP, 13358–13376.
6. Mu, J., Li, X. L., & Goodman, N. (2023). *Learning to Compress Prompts with
   Gist Tokens.* NeurIPS.
7. Chevalier, A., et al. (2023). *Adapting Language Models to Compress
   Contexts.* EMNLP.
8. Zhang, Z., et al. (2024). *A Survey on the Memory Mechanism of Large Language
   Model based Agents.* arXiv:2404.13501.
9. Xu, W., et al. (2025). *A-MEM: Agentic Memory for LLM Agents.* NeurIPS,
   arXiv:2502.12110.
10. Maharana, A., et al. (2024). *Evaluating Very Long-Term Conversational
    Memory of LLM Agents (LoCoMo).* ACL.

**Additional sources.** Chroma Research, *Context Rot* (2025); Colaco &
Lahjouji, arXiv:2607.08032 (2026); Wu et al., *LongMemEval*, arXiv:2410.10813.

> Several 2026 performance figures in the motivation come from **vendor blogs**
> and are treated as indicative. Re-measuring them under one harness is itself
> part of this project's contribution.

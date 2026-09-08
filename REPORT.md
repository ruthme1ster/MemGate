# MemGate — What to Keep, What to Forget

**A study of the memory decision policy under a bounded store**

Simar Singh Khanuja & Yash Ramchandani
Capstone Project · SVKM's NMIMS, Indore Campus

---

## Abstract

Long-running LLM agents cannot keep every turn of a conversation in context, so
a memory layer must decide what to **keep**, **compress**, **forget** and
**recall**. Production systems (Letta/MemGPT, Mem0, Zep) ship tiered memory with
compression, and the 2026 literature names the decision policy itself as the
open problem. We build MemGate, a policy-agnostic harness that measures memory
policies on a common accuracy-per-token frontier, and use it to ask which part
of a tiered memory system actually earns its cost.

Our central finding is **negative about compression and positive about
selection**. Measured with only the context window bounded — the accounting
used throughout this literature — the tiering machinery is a net loss against
simply keeping every turn and retrieving. We show this is an artefact of leaving
storage unbounded, impose a storage budget, and re-run. Under the corrected
accounting, **choosing which turns to forget is worth +8.7 answer-recall points
over forgetting oldest-first at identical storage, fidelity and retrieval**
(95% CI [+4.8, +14.3] clustered by conversation, McNemar p = 1.1×10⁻⁴), while
**compressing an evicted turn instead of dropping it loses 6.7 points**. The
rate–distortion prediction that compression must eventually win does not hold
anywhere we can reach: we test compression ratios to 182× and the selection
margin *widens* with pressure.

We further show that charging for the embedding index — normally given away
free — inverts the ranking below 64 KiB, where a policy with no index at all
beats every retrieval-based method.

---

## 1. Problem

A 200K-token context window did not solve agent memory; it relocated the
problem.

* **Context rot.** Chroma Research (2025) tested 18 frontier models: accuracy
  degrades 30–50% well before the documented context limit. It is a property of
  attention, not a training gap, so larger windows do not fix it — curation does.
* **Cost and latency.** On LoCoMo, full context is ~26,000 tokens and 9.87 s
  median latency; selective memory reaches similar accuracy at ~1,800 tokens and
  0.71 s — roughly 14× cheaper.
* **The measurable gap.** On LongMemEval, oracle retrieval scores ~92% where the
  same model interactively scores ~58%. That 34-point gap is memory *selection*
  failure, not reasoning failure.

Tiered memory with forgetting is already shipped. The open problem, named
explicitly in a 2026 review, is that current systems *"do not solve the
fundamental challenge: deciding what to remember and what to forget."*
Colaco & Lahjouji (2026) frame all such decisions as one rate–distortion
problem — what to retain, at what fidelity, under a budget — but supply a
framework, not a system. **That gap is this project.**

### Research question

> How much of the memory-selection gap can a better decision policy recover,
> and under what budget does compression pay for itself?

---

## 2. Method

MemGate wraps a frozen LLM and controls what enters the prompt. Nothing is
trained; the base model is a black box.

### 2.1 Architecture

Three tiers, as a **lifecycle rather than a partition** — everything enters
Tier 1, and the routing decision happens on eviction, when each item is scored
exactly once:

| Tier | Holds | Criterion | In context? |
|---|---|---|---|
| 1 short-term | raw verbatim turns | **when** — recent? | always |
| 2 working | compressed gists | **gist** — continuity? | always (small) |
| 3 long-term | atomic facts + vectors | **what** — durable? | on retrieval match |

### 2.2 Policies compared

| | Policy | Retention rule | Fidelity |
|---|---|---|---|
| P0 | Sliding window | recency, budget-driven | verbatim |
| RAG | Store-all + retrieve | FIFO (the no-policy control) | verbatim |
| P1 | MemGate threshold | score at ingest, τ-routed | compressed |
| **P1-A** | MemGate adaptive | score orders eviction | **compress** on eviction |
| **P1-S** | MemGate select | score orders eviction | **drop** on eviction |
| P3 | Learned scorer | fitted, cross-validated | drop |
| Oracle | — | perfect selection (bound) | verbatim |

The comparisons are constructed so each isolates **one** variable:

| Comparison | Held constant | Varies |
|---|---|---|
| P0 vs RAG | retention, storage | **read path** — recency vs retrieval |
| RAG vs P1-S | storage, fidelity, retrieval | **which turns are forgotten** |
| P1-S vs P1-A | storage, selection | **fidelity** — drop vs compress |

### 2.3 Metrics

* **Evidence (strict) recall** — every gold evidence turn reached the context.
  A multi-hop question cites up to 19 turns and 18 of 19 answers nothing.
* **Answer recall** — the answer *text* survived, scored over the 715 questions
  whose answer is recoverable from evidence text. **This is the metric to
  believe.** Evidence recall credits a compressed gist for ids whose text it
  discarded; answer recall does not.
* **Stored tokens / bytes** — what the policy still holds, as opposed to what it
  spends per query.

Reporting both is essential rather than decorative: the two metrics *disagree*
on exactly the comparison the project is about, and the disagreement is the
finding (§3.2).

### 2.4 Benchmark

LoCoMo (Maharana et al., 2024): 10 conversations, 5,882 turns, 1,527 scoreable
questions. Adversarial items (category 5) are excluded — they carry evidence,
but the correct behaviour is to decline, so retrieving it is not success.
Embedder: all-MiniLM-L6-v2 (384-d). Tokenizer: tiktoken `cl100k_base`.
P2 judge: Qwen2.5-0.5B-Instruct, run locally.

---

## 3. Results

### 3.1 Storage was never budgeted, and that determined the answer

Every experiment in this literature, and every one of ours through §22 of the
working record, caps the **context** per query and leaves the **store**
unbounded. Under that accounting "keep everything and retrieve top-k" is
charged nothing for holding all 419 turns of a conversation, so forgetting can
only ever lose information. Our own ablation duly reported the compression
machinery as a **net loss** against keeping everything — and it was right to.

Imposing a storage budget S changes the question from "what fits in context?"
to "what is worth keeping at all?"

### 3.2 Selection pays; compression does not

Context budget fixed at 2048 tokens; storage budget swept. Answer recall:

| S (tokens) | RAG store-all | P1-A adaptive *(compress)* | **P1-S select** *(drop)* |
|---|---|---|---|
| 1024 | 9.1% | 10.9% | **15.7%** |
| 2048 | 18.3% | 23.4% | **26.3%** |
| 4096 | 28.1% | 30.1% | **36.8%** |
| 8192 | 44.3% | 37.5% | **48.3%** |
| 16384 | 66.4% | 66.4% | **72.3%** |

**The two metrics disagree, and that is the result.** P1-A *wins* evidence
recall (+12.9 at S=8192) while *losing* answer recall (−6.9). A demoted gist
keeps the evidence ids of a turn whose text it has thrown away. Evidence recall
credits the pointer; answer recall checks the content. Reporting only the
former would have shown compression winning.

**RAG and P1-S hold the same verbatim turns, in the same space, retrieved the
same way, and differ in one respect only: which turns they forget when the cap
binds.** RAG forgets oldest-first; P1-S forgets lowest-utility-first. That gap
is the decision policy and nothing else.

### 3.3 Statistical significance

Paired bootstrap (B=10,000), clustered by conversation, with exact McNemar.
Clustering matters: 1,527 questions from 10 conversations are not 1,527
independent trials.

| Comparison | Metric | Δ | 95% CI (clustered) | McNemar p |
|---|---|---|---|---|
| P1-S vs RAG | **answer** | **+8.67** | **[+4.77, +14.27]** | 1.1×10⁻⁴ |
| P1-S vs RAG | evidence | +1.96 | [−1.48, +5.52] | 0.16 **(n.s.)** |
| P1-S vs P1-A | answer | +6.71 | [+4.24, +9.54] | 3.6×10⁻⁵ |
| P1-S vs P1-A | evidence | −10.74 | [−13.65, −7.90] | 4.7×10⁻²⁰ |
| P1-S vs P0 | answer | +18.46 | [+14.68, +23.56] | 3.8×10⁻¹⁷ |

**We do not claim an evidence-recall improvement over RAG** — it does not
survive clustering. The answer-recall claim does.

### 3.4 The rate–distortion prediction fails

Theory predicts compression must eventually beat selection: squeeze the store
hard enough and a lossy record must beat no record. Sweeping S downward cannot
test this — by S=256 every policy is at ~0.2% evidence recall, below the floor
where anything works. The missing axis is conversation **length**, so we
concatenate up to 10 LoCoMo conversations (186K tokens) and hold S fixed.

Answer recall at S=2048, by compression ratio:

| Ratio | RAG | P1-A *(compress)* | **P1-S** *(drop)* |
|---|---|---|---|
| 9× | 18.3% | 23.4% | **26.3%** |
| 18× | 10.8% | 14.7% | **18.2%** |
| 45× | 7.8% | 11.2% | **11.7%** |
| 91× | 4.6% | 6.0% | **8.7%** |

**No crossover to 182×**, and the selection margin over RAG *widens* with
pressure. At LoCoMo scale the optimum sits at the vertex: a subset at full
fidelity beats everything at reduced fidelity.

*Caveat:* a concatenated stream is not a natural long conversation. Each
question concerns one constituent conversation, so the others act as
distractors. It is a stress test of retention under pressure — the variable
under study — not a long-dialogue benchmark.

### 3.5 The index is not free

Storage is normally denominated in text tokens, which silently gives the
embedding index away. A 384-d float32 vector is 1,536 B against a turn's ~130 B
of text: the index outweighs the content ~12×, and the binding cost becomes the
**number of items**, not their length.

Answer recall under byte accounting:

| Budget | P0 *(no index)* | RAG | P1-S select |
|---|---|---|---|
| 32 KiB | **18.3%** | 5.0% | 15.1% |
| 64 KiB | 18.3% | 11.0% | **22.5%** |
| 256 KiB | 18.3% | 33.0% | **39.0%** |

Two results. The §3.2 conclusion is **unit-robust** — selection still beats
compression, by +7.8 to +34.0. And **below ~40 KiB an embedding index cannot
earn its own storage**: plain recency, which keeps no vectors, wins outright.
That regime is invisible when the budget counts text alone.

### 3.6 The scorer is the bottleneck, and it is learnable

Ablation at S=4096 (adaptive mode), evidence recall:

| Removed | Δ |
|---|---|
| retrieval (Tier 3) | **−30.3** |
| demotion on pressure | −10.7 *(but answer recall **+6.7**)* |
| scored eviction → FIFO | **−5.0** |
| informative compressor | −3.8 |
| consolidation merge | −2.6 |
| supersession | −0.2 |

Replacing P1's hand-chosen weights with a fitted model, under
leave-one-conversation-out (the model scoring a conversation never saw it in
training):

| Scorer | AUC | Answer recall |
|---|---|---|
| P1 heuristic (hand-tuned) | — | 26.3% |
| P3 learned, hand features | 0.746 | 27.7% |
| **P3 learned, + MiniLM** | **0.793** | **31.5%** |

The fitted weights say something the heuristic never encoded: **turn length is
the strongest single predictor of evidence-worthiness**, ahead of every
hand-designed cue.

P3-learned is **not deployable** — a live agent has no future questions to
learn from. It bounds the headroom of a learned decision policy, with the
Oracle above it and P1 below.

### 3.7 An LLM judge is the wrong instrument for salience

The obvious way to improve the scorer is to ask a language model. We ran it as
a **single-component swap**: policy, storage budget, retrieval and compression
pinned at the §3.2 winner (P1-S, context 2048, store 4096), only the scorer
varying. The judge is Qwen2.5-0.5B-Instruct, read as a continuous expectation
over digit-token logits at one forward pass rather than by generating and
parsing a digit — faster, unparseable-by-construction, and continuous, so
eviction ranking has no ties.

| Scorer | Evidence | Answer | Δ vs P1 | 95% CI (clustered) |
|---|---|---|---|---|
| P0 recency (no content judgement) | 12.6% | 25.6% | −11.19 | [−17.24, −6.19] |
| **P1 heuristic** (regex + shallow NER) | 19.8% | **36.8%** | — | — |
| P2 LLM judge (Qwen2.5-0.5B) | 14.0% | 30.9% | **−5.87** | [−10.77, −1.76] |
| P3 learned (LOCO, bound) | 33.4% | 48.3% | +11.47 | [+4.90, +17.86] |

**The judge loses to the regex by 5.87 points, and the interval excludes
zero.** This is a genuine negative result rather than a null one, and it is not
a matter of the model being small in the abstract: the same table shows the
headroom is real (+11.47 to the learned bound), so the ceiling P2 failed to
reach genuinely exists.

The diagnosis is in what the judge is shown. It rates each turn **in
isolation**, and salience is not a property of a turn on its own — a phone
number matters because something later asks for it. The learned scorer, given
the same isolated text, does better precisely because it was fitted to which
turns *turned out* to be cited, absorbing the corpus-level base rates the judge
has no way to see. That an LLM prompted per-turn underperforms a regex, at
~15 minutes of GPU per corpus against microseconds, is worth stating plainly.

---

## 4. Methodological findings

Three of this project's most useful results are about *measurement*, and each
was caught by a guard rather than by inspection.

**4.1 A policy that reads its own answer key.** `fact_id` is an evaluation
label. An early consolidation rule promoted items on `it.fact_id`, i.e. the
policy consulted the answer key when deciding what to keep. It was worth
**+45 to +47 recall points** and invalidated the first review's headline
(95% → 50%). The fix is a test, not a comment: strip every label from the input
and the store must come out bit-identical. Each new decision point — eviction
and demotion included — is added to that test before its numbers are quoted.

**4.2 A component's measured value is conditional on the bottleneck being
elsewhere.** We measured the MiniLM embedder as worth +0.2 points and concluded
it barely mattered. That was true *in the regime measured*: retrieval was
saturated against a write path discarding 93% of the conversation. Once that
ceiling was removed, the same swap was worth **+19.7**. Ablations run against a
broken pipeline systematically undervalue exactly the components that would
matter once it is fixed.

**4.3 A metric that credits pointers.** Evidence recall counts a turn as
retrieved if its *id* reaches the context. A compressed gist keeps the id and
discards the text, so it scores a hit with none of the answer behind it. During
the consolidation work, evidence recall rose 8.1% → 14.8% while answer recall
stayed pinned at 12.8% — the entire gain was bookkeeping. Answer-presence
scoring is what separated selection from compression in §3.2.

A fourth, smaller instance: batched judge scoring returned NaN for exactly the
short turns. The instinct was precision, and float32 did not fix it; the cause
was left-padding creating fully-masked attention rows. A NaN guard surfaced it.
Without that guard every filler turn would have scored NaN and quietly inverted
the eviction order.

---

## 5. Limitations

* **10 conversations.** Clustered intervals are wide because the cluster count
  is small. Effects are reported with those intervals, and the one that does
  not survive clustering is reported as not significant.
* **Context recall, not end-task accuracy.** We measure whether the needed
  evidence reached the context. That is the *ceiling* on end-task accuracy, not
  accuracy itself. Closing this needs a generative model in the loop.
* **Extractive compression only.** P1-A selects words; it does not rewrite.
  Abstractive re-summarisation may retain more content per token, and the
  negative result about compression is stated for the extractive case.
* **Concatenated streams are synthetic** (§3.4).
* **P3-learned is a headroom bound, not a component** (§3.6).
* **Single embedder and single judge model.** Both are ablated but neither is
  varied across families. The §3.7 negative is therefore evidence about *a 0.5B
  model prompted per-turn*, not about LLM judging in general; a larger model, or
  one shown the surrounding turns, is untested.

---

## 6. Conclusion

Presented as "a memory layer with three tiers", this work would have been
already done. Measured honestly, the more useful result is what *doesn't* work:

1. With storage unbounded — the standard accounting — tiered compression is a
   net loss against keeping everything and retrieving.
2. With storage bounded, **the decision policy is worth +8.7 answer-recall
   points at identical storage, fidelity and retrieval**, while **compression
   still loses 6.7 points**.
3. The rate–distortion crossover at which compression should pay is not
   reachable on this benchmark — to 182× compression, keeping a selected subset
   whole beats compressing everything.
4. Charging for the embedding index inverts the ranking below ~40 KiB.

**Practical guidance: keep a scored subset whole; do not compress everything.**
And denominate the budget in bytes, including the index, or the comparison is
not the one you think you are making.

### Future work

* End-task accuracy with a generative model reading the assembled context.
* Abstractive compression, to test whether §3.4's negative result is specific to
  extractive methods.
* A benchmark with genuinely long single conversations (LongMemEval).
* Online learning of the eviction policy from interaction, which would make a
  learned scorer deployable.

---

## Reproducing

```bash
cd memgate
python3 test_memgate.py          # 61 tests
python3 run_locomo.py            # context-budget frontier
python3 run_storage_sweep.py     # §3.2 storage budget
python3 run_scaling.py           # §3.4 compression ratio to 182x
python3 run_cost_model.py        # §3.5 byte accounting
python3 run_significance.py      # §3.3 bootstrap + McNemar
python3 train_scorer.py          # §3.6 learned scorer
python3 precompute_judge.py      # P2 judge scores (local model, ~15 min, cached)
python3 run_judge_eval.py        # §3.7 scorer comparison + CIs
python3 run_endtask.py           # §3.8 end-task accuracy (~50 min, resumable)
python3 make_figures.py          # figures
```

Results land in `results/`; figures in `results/figures/`. The benchmark and
model weights are fetched separately — see `memgate/README.md`.

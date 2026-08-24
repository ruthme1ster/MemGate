# MemGate — an adaptive memory layer for long-running LLM agents

Capstone project · SVKM's NMIMS, Indore Campus
Simar Singh Khanuja & Yash Ramchandani

MemGate sits between an agent and a frozen LLM and decides, per turn, what to
**keep**, **compress**, **forget**, and **recall** — so long multi-session
conversations stay inside a fixed token budget without losing what matters.

---

## Status: Step 4 — storage-budget frontier

> **Earlier Step 0 numbers were withdrawn.** The policy was reading `fact_id`,
> the ground-truth evaluation label, when deciding what to keep. That leak was
> worth **45–47 recall points**. It is fixed and pinned by a regression test.
> See `SESSION_RECORD.md` §13.

```bash
cd memgate
python3 test_memgate.py           # 57 tests
python3 run_locomo.py             # LoCoMo — the real benchmark
python3 run_storage_sweep.py      # the storage-budget frontier (§23)
python3 run_experiment.py         # synthetic sanity set
python3 diagnose.py               # why the policy misses what it misses

python3 run_ablations.py --budget 2048 --store-budget 4096 --adaptive
MEMGATE_EMBEDDER=hash python3 run_locomo.py    # embedder ablation
```

Python 3.8+. `tiktoken` for exact token counts and `sentence-transformers` for
the real embedder; both degrade gracefully to stdlib fallbacks, and
`backend_info()` records which actually ran.

### Setup — fetching the ignored artefacts

The benchmark and the model weights are **not in the repo** (90 MB + 2.7 MB,
reproducible from source). Fetch both:

```bash
pip install sentence-transformers tiktoken

# LoCoMo (Maharana et al., 2024)
mkdir -p data/locomo
curl -L -o data/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# all-MiniLM-L6-v2 — vendored locally because the Hub download is unreliable
# on some networks and a resumed/partial file loads as a corrupt safetensors.
B=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main
mkdir -p models/all-MiniLM-L6-v2/1_Pooling
for f in config.json config_sentence_transformers.json modules.json \
         sentence_bert_config.json special_tokens_map.json tokenizer.json \
         tokenizer_config.json vocab.txt 1_Pooling/config.json; do
  curl -sSL --retry 10 -o "models/all-MiniLM-L6-v2/$f" "$B/$f"
done
curl -L --retry 20 -C - -o models/all-MiniLM-L6-v2/model.safetensors \
  "$B/model.safetensors"   # must end up exactly 90868376 bytes
```

Without them everything still runs: the loader errors only if you call
`load_locomo()`, and the embedder falls back to hashing bag-of-words with a
warning on stderr.

## Results — LoCoMo (10 conversations, 5882 turns, 1527 questions)

MiniLM 384d + tiktoken. **Strict** = every evidence turn present (a multi-hop
question cites up to 19, and 18 of 19 answers nothing). **Answer** = the answer
*text* survived into the context, scored over the 715 questions where the answer
is recoverable from evidence.

| Policy | Budget | Strict | Soft | Answer | Avg tokens |
|---|---|---|---|---|---|
| Full-context | unlimited | 100.0% | 100.0% | 100.0% | 18 959 |
| Oracle (diagnostic bound) | 512 | 99.9% | 100.0% | 100.0% | 65 |
| P0 sliding window | 512 | 1.6% | 1.9% | 4.9% | 509 |
| **P1 MemGate** | 512 | **8.0%** | **9.8%** | **12.9%** | 484 |
| P0 sliding window | 2048 | 10.1% | 11.4% | 18.3% | 2047 |
| **P1 MemGate** | 2048 | **14.4%** | **17.3%** | **20.6%** | 1423 |
| P0 sliding window | 4096 | 18.8% | 21.8% | **31.3%** | 4094 |
| **P1 MemGate** | 4096 | **20.3%** | **24.6%** | 25.0% | 2051 |

**MemGate beats the sliding window on strict recall at every budget**, at 4096
on half the tokens (2051 vs 4094). P0 still wins answer recall at 4096 — with
4094 tokens of raw text it simply carries more verbatim content.

Per-category at 2048:

| Policy | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|
| P0 sliding window | **2.2%** | 9.4% | 7.9% | 13.2% |
| P1 MemGate | 1.4% | **17.2%** | **12.4%** | **17.9%** |

Multi-hop is the one loss: strict recall demands *all* ~3.13 cited turns, so
retention failure compounds multiplicatively. It will be the last to move.

**Read this honestly: on real dialogue the current heuristic does not work.** It
wins at tight budgets and at 2048, but plain recency overtakes it at 4096.

**The bottleneck is the write path, not retrieval.** Tier 2 used to promote at
`utility >= 0.5` while only ever containing items that scored below
`tau_fact = 0.45` — nothing could survive, so consolidation discarded everything:

```
                              before   after
turns retained anywhere         7.2%   18.1%
WRITE-PATH CEILING              9.4%   25.5%
read path (k=60, no budget)     6.7%       —
```

Consolidation now **re-summarises** instead of discarding. That more than
doubled the ceiling, but only ~1.4 points converted into recall — head
truncation is too lossy to exploit the headroom it creates. That headroom is
what the Step 2 abstractive summariser exists to claim.

The gap this project exists to close is now measured in our own harness:
**Oracle 100% at 65 tokens vs P1 15% at 2055** — entirely memory selection, not
reasoning.

## Results — the storage budget, and why it changes the answer

Every result above caps the **context** assembled per query but leaves the
**store** unbounded. Under that accounting "keep everything and retrieve top-k"
is charged nothing for holding all 419 turns of a conversation, so forgetting
can only ever lose information — and our own ablation duly found the
compression machinery to be a **net loss** against keeping everything.

`run_storage_sweep.py` imposes the missing constraint: the store is capped at
**S** tokens, S is swept, and accuracy is reported per *stored* token as well as
per context token. Three retention regimes isolate one variable each:

| comparison | held constant | varies |
|---|---|---|
| P0 vs RAG | retention | **read path** — recency vs retrieval |
| RAG vs P1-S | storage, fidelity, retrieval | **which turns are forgotten** |
| P1-S vs P1-A | storage, selection | **fidelity** — drop vs compress |

Context budget fixed at 2048; **answer** recall is the honest metric:

| S | Policy | Strict | **Answer** | Stored |
|---|---|---|---|---|
| 2048 | RAG store-all | 10.1% | 18.3% | 2031 |
| 2048 | P1-A adaptive *(compress)* | **19.6%** | 23.4% | 2022 |
| 2048 | **P1-S select** *(drop)* | 10.9% | **26.3%** | 2025 |
| 4096 | RAG store-all | 17.8% | 28.1% | 4068 |
| 4096 | P1-A adaptive *(compress)* | **30.5%** | 30.1% | 4042 |
| 4096 | **P1-S select** *(drop)* | 19.8% | **36.8%** | 4077 |

**Selection wins; compression loses.** P1-A wins strict recall and *loses*
answer recall (−6.9 at S=8192) because a compressed gist keeps the evidence ids
of a turn whose text it has thrown away — the exact failure the answer metric
exists to catch. Dropping instead of compressing beats both at every budget.

The cleanest number the project has produced: **RAG and P1-S hold the same
verbatim turns, in the same space, retrieved the same way, and differ only in
which turns they forget** — oldest-first versus lowest-utility-first. That gap
is **+8.7 answer points at S=4096**, and it is the decision policy and nothing
else.

**Rate–distortion reading:** at LoCoMo scale the optimum sits at the *vertex* —
a subset at full fidelity beats everything at reduced fidelity. Compression
should only start paying once even the selected subset will not fit verbatim,
and LoCoMo's ~19K-token conversations do not reach that regime. Finding that
second crossover needs a tighter S or a longer benchmark.

## Results — synthetic sanity set (leak-free)

| Policy | Budget | Recall | Avg tokens |
|---|---|---|---|
| Full-context | unlimited | 100.0% | 5825 |
| P0 sliding window | 800 | 10.0% | 800 |
| **P1 MemGate** | 800 | **50.0%** | 505 |
| P0 sliding window | 3200 | **57.5%** | 3198 |
| P1 MemGate | 3200 | 55.0% | 1113 |

### Honest caveats

* The synthetic set is a **sanity harness, not a result** — it proves the
  pipeline and metrics are correct. LoCoMo is the real number.
* The metric is **context recall @ budget**, not end-task accuracy. It isolates
  the memory policy from the model's reasoning and needs no API key, and it is
  the *ceiling* on end-task accuracy.
* A Tier 2 summary is truncated to 14 words but keeps its `fact_id`, so it
  counts as a hit even if the answer was cut. **The metric is currently generous
  to MemGate.** LoCoMo ships gold answer strings, so an answer-presence check is
  the planned tightening.
* Adversarial questions (LoCoMo category 5) are excluded: they carry evidence,
  but the correct behaviour is to decline, so retrieving it is not success.

## Metric rationale

`context recall @ budget` answers: *given B tokens, did the policy put the
needed evidence in front of the model?* Choosing it for Step 0 means the
harness is deterministic, instant, free, and measures exactly the variable
under study — the decision policy — with nothing else confounding it.

## Architecture

```
Agent
  |
  v
MemGate
  Adaptive Memory Decision Engine   scores each evicted turn
  Compressor / Summariser           tier 2 summaries
  Retriever                         semantic search over tier 3
  Context Assembler                 packs to the token budget
  |
  +-- Tier 1 short-term   raw recent turns          criterion: WHEN
  +-- Tier 2 working      compressed summaries      criterion: GIST
  +-- Tier 3 long-term    atomic facts + vectors    criterion: WHAT
  |
  v
Frozen LLM (open model or hosted API)
```

Tiers are a **lifecycle, not a partition**: everything enters Tier 1, and the
routing decision happens *on eviction*:

```
u >= tau_fact   -> Tier 3 (durable fact)
u >= tau_low    -> Tier 2 (compressed gist)
otherwise       -> dropped from context
```

## Policies

| | Policy | Scoring | Status |
|---|---|---|---|
| P0 | Sliding window | recency only, budget-driven | done |
| P1 | MemGate heuristic | rules + shallow NER + novelty | done |
| P1-A | MemGate adaptive | verbatim until full, **compress** on eviction | done |
| P1-S | MemGate select | verbatim until full, **drop** on eviction | done |
| RAG | store-all + retrieve | none — FIFO eviction (the control) | done |
| P2 | LLM-judged salience | small LLM rates importance | Phase 3 |
| P3 | Distilled scorer | trained on P2 labels | Phase 3 |

P0 is given the **same token budget** as MemGate, so the only difference is
*which* turns are chosen — never how many tokens may be spent. A fixed small
window would have made it a strawman.

## What the diagnostic shows

`diagnose.py` attributes every miss to one swappable component:

* `scorer_miss` — P1 didn't recognise a durable fact → the headroom P2/P3 exist to close
* `dedupe_collapse` — hashing embedder can't separate similar surface forms → fixed by a real embedder
* `retrieval_miss` — ranking, not memory policy

On the leak-free synthetic set the attribution is
`{scorer_miss: 18, dedupe_collapse: 1, retrieval_miss: 2}` — **the scorer is the
bottleneck by an order of magnitude**, which is exactly the headroom P2/P3 exist
to recover. LoCoMo says the same thing more harshly: retrieval is saturated
against a 9.4% write-path ceiling.

## Layout

```
memgate/
  memgate/
    types.py     MemoryItem, Turn, Question (multi-evidence, superseded_by)
    utils.py     tiktoken + MiniLM, with stdlib fallbacks and provenance
    store.py     three-tier store, dedupe, semantic supersession, consolidation
    scoring.py   P0/P1 implemented, P2/P3 stubs on the same interface
    policies.py  baselines + MemGate, routing and budget-aware assembly
    data.py      synthetic generator + LoCoMo loader + integrity assertions
    harness.py   strict/soft recall, per-category breakdown, sweep, CSV
  run_locomo.py        LoCoMo evaluation
  run_storage_sweep.py storage-budget frontier
  run_ablations.py     component attribution
  run_experiment.py    synthetic sanity set
  diagnose.py          miss analysis by cause
  test_memgate.py      57 tests
  data/locomo/         locomo10.json (2.7 MB)
  models/              vendored all-MiniLM-L6-v2
```

## No ground-truth leakage

`fact_id` and `is_filler` are **evaluation labels**. No routing, promotion,
dedupe or supersession decision may read one. This is enforced, not just
documented — the store must come out bit-identical when the labels are stripped:

```
PASS  routing is invariant to ground-truth labels
PASS  eviction under a storage budget is label-blind (threshold)
PASS  eviction under a storage budget is label-blind (adaptive)
```

The test exists because the earlier version *did* read `fact_id`, and it was
worth 45–47 recall points. The storage budget added two more decision points
that could read the answer key — *which item to evict* and *which to demote* —
so the same invariant is pinned there too. Eviction ranks on `utility` and
recency only.

## Embedder ablation

Swapping the hashing bag-of-words for MiniLM moves LoCoMo recall by **+0.2
points** at 2048 tokens (10.8% → 11.0%). This was predicted before the run:
retrieval was already saturated at 6.7% against a 9.4% write-path ceiling, so a
better embedder had nothing left to recover.

**Retrieval, re-ranking and larger embedding models are not worth effort until
the write path is fixed.** The embedder does earn its place, but on
supersession, not recall — it is the difference between retiring the correct
record and retiring the wrong one:

| Embedder | → true target | → nearest competitor | Outcome |
|---|---|---|---|
| MiniLM 384d | **0.865** | 0.741 | Correct, thresholds 0.50–0.70 |
| Hashing BoW | 0.500 | **0.583** | Wrong — ranking inverted |

## Next

1. **Find the second crossover** — push S below 512, or move to a benchmark
   with longer conversations than LoCoMo's ~19K tokens (LongMemEval), to reach
   the regime where the selected subset no longer fits verbatim and compression
   must start paying.
2. **P2 (LLM judge)** — now the highest-value work, and its payoff is
   quantified in advance: scored eviction is already worth −5.0 strict against
   FIFO, so a better scorer attacks a component demonstrably load-bearing.
3. **Charge for the embeddings.** `stored_tokens` counts text only; a 384-d
   float32 vector per retained item is real storage that verbatim policies pay
   on every turn. A byte-denominated budget may shift the comparison.
4. End-task accuracy through a real model — context recall is its ceiling.

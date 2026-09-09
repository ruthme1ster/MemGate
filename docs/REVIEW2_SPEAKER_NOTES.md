# MemGate — Review 2 speaker's guide

**Companion to `MemGate_Capstone_Review2_NMIMS.pptx` (28 slides).**
Simar Singh Khanuja & Yash Ramchandani · SVKM's NMIMS, Indore Campus

This document has three parts:

* **Part 1** — the project explained end to end, in plain terms. Read this
  first; it is what you need to hold in your head, not the slides.
* **Part 2** — slide-by-slide notes. For each slide: what is on it, what to
  say, the one number to land, and how to move on.
* **Part 3** — the question bank, the timing plan, and the failure modes.

---

# Part 1 — The project in plain terms

## 1.1 What problem are we solving?

An LLM agent that runs for weeks — a coding assistant, a research agent, a
personal assistant — accumulates a conversation far longer than any context
window. Something has to decide what goes into the prompt on each turn.

The obvious answer, "use a bigger context window", does not work:

* **Context rot.** Chroma Research (2025) tested 18 frontier models and found
  accuracy degrading 30–50% *well before* the documented context limit. This is
  a property of how attention behaves over long inputs, not a training gap, so
  bigger windows do not fix it.
* **Cost.** On LoCoMo, full context is ~26,000 tokens per query. Selective
  memory reaches comparable accuracy at ~1,800 tokens — roughly 14× cheaper.
* **The gap is measurable.** On LongMemEval, *oracle* retrieval scores ~92%
  where the same model retrieving for itself scores ~58%. That 34-point gap is
  not the model failing to reason. It is the memory layer failing to select.

So the real question is not "how much can we fit?" but **"what is worth
keeping?"** — and that is a decision, made per turn, that nobody has measured
properly.

## 1.2 Why is this not already done?

It half is. Letta (formerly MemGPT), Mem0 and Zep all ship tiered memory with
compression and forgetting. If we presented MemGate as "a memory layer with
three tiers", the honest response would be *that already exists*.

What does **not** exist is a measurement of the decision itself. A 2026 review
of agent-memory systems says exactly this: current systems *"do not solve the
fundamental challenge: deciding what to remember and what to forget."* Colaco &
Lahjouji (2026) frame every such decision as one rate–distortion problem — what
to retain, at what fidelity, under a budget — but they supply a framework, not
a system.

**That gap is this project.** MemGate is a policy-agnostic harness that puts
competing memory policies on one accuracy-per-token frontier, with each
comparison built so that exactly one variable moves.

## 1.3 How the system works

MemGate sits between the agent and a **frozen** LLM. Nothing is trained. The
base model is a black box. All MemGate does is control what enters the prompt.

**Three tiers, but as a lifecycle rather than a partition.** This is the design
point people get wrong. Everything enters Tier 1. The routing decision happens
*once*, on eviction, when the budget binds:

```
u ≥ τ_fact   → Tier 3   long-term    atomic fact + vector   (recalled on match)
u ≥ τ_low    → Tier 2   working      compressed gist        (always in context)
otherwise    → dropped
```

| Tier | Holds | The question it answers | In context? |
|---|---|---|---|
| 1 short-term | raw verbatim turns | **when** — is this recent? | always |
| 2 working | compressed gists | **gist** — is continuity needed? | always (small) |
| 3 long-term | atomic facts + vectors | **what** — is this durable? | on retrieval match |

**Write path.** A turn arrives → the decision engine scores it → on eviction it
is routed to a tier or dropped.

**Read path.** A query arrives → embed it → retrieve semantically from Tier 3 →
assemble a context that fits the token budget → call the frozen LLM.

## 1.4 The policies we compare

| | Policy | How it decides what to forget | Fidelity |
|---|---|---|---|
| P0 | Sliding window | recency — drop the oldest | verbatim |
| RAG | Store-all + retrieve | FIFO — **the no-policy control** | verbatim |
| P1 | MemGate threshold | scored at ingest, τ-routed | compressed |
| **P1-A** | MemGate adaptive | score orders eviction | **compress** on eviction |
| **P1-S** | MemGate select | score orders eviction | **drop** on eviction |
| P2 | LLM-judged | a 0.5B model rates each turn | drop |
| P3 | Learned scorer | fitted on which turns were cited | drop |
| Oracle | — | perfect selection (upper bound) | verbatim |

**The comparisons are the design.** Each isolates one variable:

| Comparison | Held identical | The one thing that varies |
|---|---|---|
| P0 vs RAG | retention, storage | **read path** — recency vs retrieval |
| RAG vs P1-S | storage, fidelity, retrieval | **which turns are forgotten** |
| P1-S vs P1-A | storage, selection | **fidelity** — drop vs compress |

P0 gets the *same token budget* as MemGate. A fixed small window would be a
strawman, and the panel would be right to say so.

## 1.5 How we measure

Two metrics, and the fact that there are two is itself a result.

* **Evidence (strict) recall** — did every gold evidence turn reach the
  context? A multi-hop question can cite 19 turns, of which 18 answer nothing.
* **Answer recall** — did the answer *text* survive into the context? Scored
  over the 715 questions (of 1,527) whose answer is literally recoverable from
  the evidence. **This is the metric to believe.**

Why both: **evidence recall credits a pointer.** A compressed gist keeps the
evidence *id* of a turn whose text it threw away, so it scores a hit with none
of the answer behind it. During our consolidation work, evidence recall rose
8.1% → 14.8% while answer recall stayed pinned at 12.8% — the entire gain was
bookkeeping.

**Benchmark.** LoCoMo (Maharana et al., 2024): 10 real multi-session
conversations, 5,882 turns, 1,527 scoreable questions. Adversarial items
(category 5) are excluded — they carry evidence, but the correct behaviour is
to *decline*, so retrieving it is not success.

## 1.6 The five findings, in order of importance

**1. Storage was never budgeted, and that pre-decided the answer.**
Every paper in this literature — and every experiment of ours up to a point —
caps the *context* per query and leaves the *store* unbounded. Under that
accounting, "keep everything and retrieve top-k" is charged nothing for holding
all 419 turns of a conversation, so forgetting can only ever lose information.
Our own ablation duly reported the compression machinery as a **net loss**, and
it was right to. Impose a storage budget S and the question changes from *what
fits in context?* to *what is worth keeping at all?*

**2. Selection pays; compression does not.**
At identical storage, identical fidelity and identical retrieval, forgetting by
scored utility rather than oldest-first is worth **+8.67 answer-recall points**
(95% CI [+4.77, +14.27], McNemar p = 1.1×10⁻⁴). Compressing an evicted turn
instead of dropping it **loses 6.71 points**. RAG and P1-S hold the same turns,
in the same space, retrieved the same way — the only difference is *which* ones
they forget. That gap is the decision policy and nothing else.

**3. The rate–distortion prediction fails.**
Theory says compression must eventually win: squeeze hard enough and a lossy
record must beat no record. We cannot test that by shrinking S — everything
hits the floor first — so we raise the *compression ratio* by concatenating up
to 10 conversations (186K tokens) at fixed S. **No crossover to 182×**, and the
selection margin over RAG *widens* under pressure.

**4. The embedding index is not free.**
Storage is normally counted in text tokens, which silently gives the index
away. A 384-d float32 vector is 1,536 bytes against a turn's ~130 bytes of
text — the index outweighs the content ~12×. Count bytes and the finding
survives, but **below ~40 KiB an embedding index cannot earn its own storage**:
plain recency, which keeps no vectors at all, wins outright.

**5. An LLM judge loses to a regex.**
The obvious way to improve the scorer is to ask a language model. Held as a
single-component swap — everything pinned, only the scorer varying — a
Qwen2.5-0.5B judge scores 30.9% against the hand-written heuristic's 36.8%:
**−5.87 points, interval excluding zero**, for ~15 minutes of GPU per corpus
against microseconds. The diagnosis matters: the judge rates each turn **in
isolation**, and salience is not a property of a turn on its own. A phone
number matters because something *later* asks for it.

## 1.7 What is NOT done

* **End-task accuracy.** Everything above is *context recall* — did the
  evidence reach the prompt? That is the **ceiling** on accuracy, not accuracy.
  The pipeline, both controls and the generation cache are built and validated
  on a 12-question smoke run. The full 715-question run is 4–6 hours of local
  generation and has not been made. **This is the critical path.**
* Abstractive compression (our negative result is stated for *extractive* only).
* A genuinely long benchmark (LongMemEval) rather than concatenated streams.
* Online learning, which is what would make a learned scorer deployable.

---

# Part 2 — Slide by slide

Legend: **On screen** = what the panel sees · **Say** = your words ·
**Land** = the one number that must register · **Next** = the transition.

---

### Slide 1 — Title

**On screen.** MemGate / *What to keep, what to forget* / both names and roll
numbers / NMIMS Indore / Progress review, September 2026.

**Say.** "MemGate is a memory layer for long-running LLM agents. The subtitle is
the actual research question — what to keep and what to forget — and most of
what we'll show you today is measurement, including several results that came
out negative."

**Next.** "Before the results, one correction we owe you."

*(Do not linger. 20 seconds.)*

---

### Slide 2 — Where the project stands

**On screen.** A 10-row status table (workstream / state / evidence) and the
headline band.

**Say.** "The measurement track is complete. Nine experiments run end to end,
every claim carries a confidence interval clustered by conversation, and every
number on these slides is generated from a committed CSV — no slide has a
hand-typed figure. One row is a pilot, three are scoped extensions."

**Land.** +8.7 answer-recall points.

**Next.** "One thing has to come first, though."

---

### Slide 3 — First: a Review 1 number is withdrawn

**This is the most important slide in the deck. Do not soften it.**

**On screen.** The leak explained, and the 95% → 50% table.

**Say.** "At Review 1 we reported 95% recall for MemGate against 10% for a
sliding window. That number was wrong, and we found out why. The policy was
reading `fact_id` — the ground-truth evaluation label — when deciding what to
keep. It was consulting the answer key. Stripping the leak, the honest number
for the same configuration is 50%. The leak was worth 45 to 47 recall points.

The fix is a test, not a comment: strip every evaluation label from the input,
and the store must come out bit-identical. Every decision point we've added
since — eviction under a storage budget, and demotion — went into that test
*before* we quoted any number from it."

**Land.** 95% → 50%. Say both numbers out loud.

**Why it leads.** A withdrawn result the students raise is worth more than one
the panel discovers. After this slide the panel stops auditing and starts
listening.

**Next.** "Everything after this is measured on a real benchmark, leak-free."

---

### Slide 4 — The problem

**On screen.** Three bullets — context rot, cost/latency, the measurable gap —
plus the 2026 review quote.

**Say.** "A 200K context window did not solve agent memory; it relocated the
problem. Chroma tested 18 frontier models and found 30–50% degradation well
before the documented limit — that's a property of attention, so bigger windows
don't fix it. On LongMemEval, oracle retrieval scores 92% where the same model
retrieving for itself scores 58%. That 34-point gap is not reasoning failure.
It's selection failure."

**Land.** The 34-point gap.

**Next.** "So why isn't this already solved?"

---

### Slide 5 — The question, and what is new

**On screen.** The research question in a band, then three bullets on
positioning.

**Say.** "Our research question: how much of the memory-selection gap can a
better decision policy recover, and under what budget does compression pay for
itself?

We want to be honest about novelty. Tiered memory with forgetting is *already
shipped* — Letta, Mem0, Zep. If we presented this as 'a memory layer with three
tiers', it would already be done. What isn't done is measuring the decision
itself. The 2026 framework paper treats it as a rate–distortion problem but
supplies a framework, not a system. MemGate is the harness that makes it
measurable."

**Land.** "Our contribution is a measurement, and several of its results are
negative."

**Next.** "Here's the system it runs on."

---

### Slide 6 — System architecture

**On screen.** Write path (turn → decision engine → three routing rules →
tiered store), read path (query → retrieve → assemble → frozen LLM → response).

**Say.** "MemGate wraps a frozen LLM. Nothing is trained — the base model is a
black box, and all we control is what enters the prompt.

The design point that matters: the tiers are a **lifecycle, not a partition**.
Everything enters Tier 1. The routing decision happens once, on eviction, when
the budget binds. Above τ_fact it becomes a durable fact with a vector; above
τ_low it becomes a compressed gist; otherwise it's dropped. That single
decision point is the entire subject of this project."

**Land.** Scored exactly once, on eviction.

**Next.** "Here's what we compare it against."

---

### Slide 7 — Policies compared

**On screen.** The 8-policy table, then the three one-variable comparisons.

**Say.** "Eight arms. The one to notice is RAG — store everything and retrieve
top-k. That's the *no-policy control*: FIFO eviction, no scoring. It's the
thing we have to beat for the decision policy to have earned anything.

The lower table is the actual experimental design. Each comparison holds
everything constant except one variable. RAG versus P1-S holds storage,
fidelity and retrieval identical, and varies only *which turns are forgotten*.

And P0 gets the same token budget as MemGate — the only difference is which
turns are chosen, never how many tokens may be spent. A fixed small window
would be a strawman."

**Land.** RAG is the no-policy control.

**Next.** "One more setup point, and it's the one that produced our sharpest
result."

---

### Slide 8 — Two metrics, and why both are reported

**Set this up carefully — slide 11 pays it off.**

**On screen.** Three metric definitions, the "why two" band, two more bullets.

**Say.** "Evidence recall asks: did every gold evidence turn reach the context?
Answer recall asks: did the answer *text* survive? Answer recall is the metric
to believe.

Why report both? Because they **disagree**, on exactly the comparison this
project is about — and the disagreement is the finding. Evidence recall credits
a *pointer*: a compressed gist keeps the evidence id of a turn whose text it
threw away, and scores a hit with none of the answer behind it. During our
consolidation work, evidence recall went from 8.1% to 14.8% while answer recall
stayed pinned at 12.8%. The entire gain was bookkeeping."

**Land.** 8.1 → 14.8 while answer stayed at 12.8.

**Next.** "The setup, briefly."

---

### Slide 9 — Experimental setup

**On screen.** Element/choice/why table — benchmark, exclusions, embedder,
tokenizer, judge, reader, statistics.

**Say.** "LoCoMo — 10 real multi-session conversations, 5,882 turns, 1,527
scoreable questions. We exclude adversarial items: they carry evidence, but the
correct behaviour is to decline, so retrieving it isn't success.

Everything runs locally and offline. tiktoken for exact token counts, because a
budget measured in words isn't a budget.

The statistical point matters more than it looks: 1,527 questions drawn from 10
conversations are **not** 1,527 independent trials. Every interval in this deck
is resampled at the conversation level."

**Land.** Clustered by conversation, not by question.

**Next.** "Now the result that reorganised the whole project."

---

### Slide 10 — Storage was never budgeted

**The intellectual core of the project.**

**On screen.** Three bullets, fig1 (the storage frontier), the correction band.

**Say.** "Every experiment in this literature caps the context assembled per
query and leaves the store unbounded — and so did every experiment of ours up
to this point. Under that accounting, 'keep everything and retrieve top-k' is
charged nothing for holding all 419 turns of a conversation. So forgetting can
only ever lose information.

Our own ablation duly reported the compression machinery as a net loss against
just keeping everything. And it was *right to* — given how we were counting.

Imposing a storage budget changes the question from 'what fits in context?' to
'what is worth keeping at all?' And when we did that, the ranking inverted."

**Land.** The accounting pre-decided the answer.

**Next.** "Here's what the corrected accounting shows."

---

### Slide 11 — Selection pays; compression does not

**The headline. This is where the live demo goes.**

**On screen.** Storage sweep table (S from 1024 to 16384; RAG / P1-A / P1-S /
delta), one bullet, the "cleanest comparison" band.

**Say.** "Context fixed at 2048 tokens, storage swept. P1-S — select, which
drops the lowest-utility turn — beats both at every budget. At S=4096 it's
36.8% against RAG's 28.1%.

RAG and P1-S hold the same verbatim turns, in the same space, retrieved the
same way. They differ in one respect: RAG forgets oldest-first, P1-S forgets
lowest-utility-first. That gap is the decision policy and nothing else."

**► LIVE DEMO — 60 seconds.** Switch to `frontend/index.html`, scroll to
*Storage budget*, and flip the metric toggle in the top right.

> "On answer recall, dropping beats compressing at every budget. Watch what
> happens when I switch to evidence recall." *(flip)* "The lines cross —
> compression now appears to win, by 12.9 points at S=8192. It doesn't. The
> gist kept the evidence *id* of a turn whose text it threw away, and evidence
> recall credits the pointer. If we'd reported only this metric, we would have
> concluded that compression works."

Flip back and return to the deck.

**Land.** +8.7 points, same storage, same fidelity, same retrieval.

**Next.** "Which raises the question of whether that's real."

---

### Slide 12 — What survives clustering

**On screen.** Six comparisons with Δ, clustered 95% CI, McNemar p, and a
verdict column. One row reads *not significant*.

**Say.** "Paired bootstrap, ten thousand resamples, clustered by conversation,
with an exact McNemar alongside.

The answer-recall gain over RAG is +8.67 with the interval clear of zero,
p=1.1×10⁻⁴. That claim stands.

The **evidence**-recall gain over RAG is +1.96 — and it does **not** survive
clustering. p = 0.16. So we do not claim it. It's on this slide, marked, rather
than dropped quietly."

**Land.** We do not claim an evidence-recall improvement over RAG.

**Next.** "Theory makes a prediction we could test, so we tested it."

---

### Slide 13 — The rate–distortion prediction fails

**On screen.** Two bullets, the compression-ratio table (9× to 91×), fig2, the
result band, and the caveat.

**Say.** "Rate–distortion theory says compression must eventually beat
selection — squeeze hard enough and a lossy record has to beat no record.

We can't test that by shrinking storage; by S=256 everything is at 0.2%, below
the floor where anything works. So the missing axis is *length*. We concatenate
up to 10 conversations — 186,000 tokens — and hold storage fixed, which pushes
the compression ratio to 182×.

No crossover. And the selection margin actually *widens* under pressure. At
this scale the optimum sits at the vertex: a subset at full fidelity beats
everything at reduced fidelity."

**Caveat — say it yourself.** "A concatenated stream is not a natural long
conversation. Each question concerns one constituent conversation, so the
others act as distractors. It's a stress test of retention, not a long-dialogue
benchmark."

**Land.** Falsified to 182×.

---

### Slide 14 — The index is not free

**On screen.** One bullet, the byte-budget table (32/64/256 KiB), fig4, the
two-results band.

**Say.** "Storage is normally denominated in text tokens — which silently gives
the embedding index away for free. A 384-dimension float32 vector is 1,536
bytes. A turn is about 130 bytes of text. The index outweighs the content
twelve to one, which means the binding cost is the *number* of items, not their
length.

Two results. First, our selection finding is unit-robust — it holds when you
count bytes. Second, below about 40 KiB an embedding index cannot earn its own
storage: plain recency, which keeps no vectors at all, wins outright. That
regime is completely invisible if you count text alone."

**Land.** Below ~40 KiB, no-index recency wins.

---

### Slide 15 — The scorer is the bottleneck, and it is learnable

**On screen.** Ablation table (retrieval −30.3 … supersession −0.2), the
learned-scorer table, fig3.

**Say.** "Remove one component at a time and re-run. Retrieval is worth 30
points — nothing else is close. Scored eviction is worth 5. Note the demotion
row: removing it *costs* 10.7 evidence points but *gains* 6.7 answer points.
That's the compression finding in miniature.

Then we replaced our hand-tuned weights with a fitted model, under
leave-one-conversation-out — the model scoring a conversation never saw it in
training. AUC 0.79, and answer recall goes from 26.3% to 31.5%.

The fitted weights told us something our heuristic never encoded: **turn length
is the strongest single predictor of evidence-worthiness**, ahead of every cue
we designed by hand."

**Be first to say it.** "P3 is not deployable — a live agent has no future
questions to learn from. It's a headroom bound, not a component."

**Land.** Turn length beat every hand-designed cue.

---

### Slide 16 — An LLM judge is the wrong instrument for salience

**On screen.** Four scorers with evidence, answer, Δ vs P1 and clustered CI;
three bullets; the diagnosis band.

**Say.** "The obvious way to improve the scorer is to ask a language model. We
built it and ran it as a single-component swap — policy, storage, retrieval and
compression all pinned, only the scorer varying, so anything that moves is the
scorer.

Three readings, all significant. One: scoring *at all* is worth 11.2 points
over unscored recency, so the decision point is load-bearing. Two: the LLM
judge loses to our regex by 5.87 points, and the interval excludes zero — that's
a real loss, not a wash, for fifteen minutes of GPU per corpus against
microseconds. Three: the headroom is real, +11.5 to the learned bound, so P2
wasn't defeated by a missing ceiling.

The diagnosis is what the judge is *shown*. It rates each turn in isolation, and
salience is not a property of a turn on its own. A phone number matters because
something later asks for it."

**Land.** The LLM judge loses to a regex by 5.87 points.

---

### Slide 17 — End-task accuracy: built, and honestly labelled

**On screen.** Six-arm table, fig5, the two controls, and a bordered warning
band.

**Say — the warning first, before any number.** "Read the flag before the
table. This is a **12-question smoke run. It is not a result.**

What it demonstrates is that the pipeline works end to end: both controls, the
generation cache, the scoring. Every number so far in this deck is *context
recall* — did the evidence reach the prompt — which is the ceiling on accuracy,
not accuracy. This arm sends the assembled context through a real reader and
scores the output with the same normaliser, so 'the answer reached the context'
and 'the answer reached the output' are one predicate applied at two points.

Two controls the comparison can't be read without. Closed-book is the floor — a
model answering from parametric knowledge would make every policy look good.
Oracle is the ceiling.

The full 715-question run is four to six hours of local generation. It's the
critical path to the final review."

**Land.** Not a result. A validated pipeline.

---

### Slide 18 — Three findings about measurement

**On screen.** A table of three findings with what each cost, plus the NaN
footnote.

**Say.** "Three of our most useful results are about measurement itself, and
none was found by reading the code — each was found because a guard fired.

One: the policy that read its own answer key. That's the leak from slide 3,
worth 45 points.

Two: a component's measured value is conditional on the bottleneck being
elsewhere. We measured our embedder at +0.2 points and concluded it barely
mattered. That was true *in the regime we measured* — retrieval was saturated
against a write path throwing away 93% of the conversation. Once we fixed that,
the same swap was worth **+19.7**. Ablations run against a broken pipeline
systematically undervalue exactly the components that would matter once it's
fixed.

Three: a metric that credits pointers — that's the 8.1 to 14.8 with answer
recall pinned.

And a fourth, smaller one: our judge returned NaN for exactly the short turns.
The instinct is precision, and float32 didn't fix it. The cause was left-padding
creating fully-masked attention rows. Without the NaN guard, every filler turn
would have scored NaN and silently inverted our eviction order."

**Land.** Every one was caught by a guard, not by inspection.

---

### Slide 19 — Tools and technology

**On screen.** Proposed at Review 1 / actually used / why it changed.

**Say.** "At Review 1 we proposed a hosted API, Llama 3 or Mistral, LangChain,
and FAISS. We used none of it.

Everything runs locally: Qwen2.5 models we vendored ourselves, a direct harness
with no framework, NumPy cosine instead of an ANN index, tiktoken for exact
token counts, and a paired bootstrap for the statistics.

This is not scope reduction — it's the opposite. Running everything locally and
free is what made 35 storage configurations, 23 ablations, a ten-thousand-sample
bootstrap and a 36-point scaling sweep affordable to *re-run on every change*.
A framework would also have confounded the exact variable we're studying,
because it brings its own retrieval and its own prompt assembly."

**Land.** Local and free is what made the sweep affordable.

---

### Slide 20 — Verification and engineering discipline

**On screen.** Guard / what it prevents.

**Say.** "Seven guards, and each exists because the failure it prevents already
happened to us once.

61 tests. Three of them are label-blindness invariants — strip every evaluation
label and the store must come out bit-identical. That's the leak, re-armed.

Offline pinning on import, because a run that can silently fetch a model is a
run whose inputs aren't pinned — we already lost a results table once to a
half-downloaded MiniLM.

And the results CSVs are committed alongside the code that produced them, so a
figure in this deck can't drift from the code that made it."

**Land.** The fix for a leak is a test, not a comment.

---

### Slide 21 — Deliverables

**On screen.** Seven items: REPORT.md, the progress report, SESSION_RECORD.md,
the dashboard, the library, the results, the deck generator.

**Say.** "Everything is in the repository. Note the last one: this deck is
*generated* from the same CSVs as the report and the dashboard. No slide here
has a hand-typed number, because a hand-typed number goes stale the moment an
experiment is re-run and nobody notices until a reviewer does.

You can check every number in this deck without running the benchmark — the
results are committed, and each slide names the file it came from."

---

### Slide 22 — Timeline

**On screen.** Gantt: filled bars complete, hollow bars pending, four
milestones.

**Say.** "Filled bars are done. The critical path to the final review is the
full end-task run — the pipeline is built and validated, what it needs is
machine time. Then abstractive compression and LongMemEval in October, online
learning in November, and the final report in December."

---

### Slide 23 — Limitations, and what remains

**On screen.** Two columns: six limitations, four remaining items with effort
estimates.

**Say.** "Six limitations, stated because each one bounds a claim we made.

Ten conversations, so intervals are wide — which is exactly why the one effect
that doesn't survive clustering is reported as not significant. Context recall
is a ceiling, not accuracy. Our compression is extractive, so the negative
result is stated for that case only. The concatenated streams are synthetic. And
our judge result is evidence about *a 0.5B model prompted per turn*, not about
LLM judging in general.

Four things remain, with effort estimates attached."

**Land.** Every limitation maps to a claim we deliberately didn't overstate.

---

### Slide 24 — Conclusion as it stands

**On screen.** Five findings, the practical-guidance band, the closing line.

**Say.** "Five findings. With storage unbounded — the standard accounting —
tiered compression is a net loss. With storage bounded, the decision policy is
worth +8.7 points and compression still loses 6.7. The rate–distortion crossover
isn't reachable. Charging for the index inverts the ranking below 40 KiB. And
an LLM judge loses to a regex.

Practical guidance: **keep a scored subset whole; do not compress everything.**
And denominate the budget in bytes, including the index, or the comparison isn't
the one you think you're making.

Presented as 'a memory layer with three tiers', this work would already have
been done. Measured honestly, the more useful result is what doesn't work."

---

### Slide 25 — References

**On screen.** 13 references; MemGPT flagged as the base paper, LoCoMo as the
benchmark.

**Say.** "MemGPT is our base paper, LoCoMo the benchmark. Author lists, venues
and years were each verified against source pages. A few 2026 performance
figures in our motivation come from vendor blogs — we treat those as
indicative, and re-measuring them under one harness is itself part of what we're
contributing."

---

### Slide 26 — Thank you

**Say.** "Thank you. Questions welcome — including on the results that didn't
work."

---

### Slides 27–28 — Backup (on request only)

* **27** — the full storage sweep: every policy at every S, with stored and
  context tokens. Use it if the panel challenges a specific figure.
* **28** — the repository layout. Use it if asked "where does this live?" or
  "what's actually committed?"

---

# Part 3 — Running the session

## 3.1 Timing — 20 minutes plus questions

| Slides | Content | Minutes |
|---|---|---|
| 1–3 | Title, status, **the withdrawal** | 3 |
| 4–5 | Problem and positioning | 2 |
| 6–9 | Architecture, policies, metrics, setup | 3 |
| 10–11 | The accounting error and the headline **+ live demo** | 5 |
| 12 | Significance | 1.5 |
| 13–16 | Rate–distortion, bytes, scorer, the judge | 3 |
| 17 | End task, labelled as a pilot | 1.5 |
| 18–24 | Method findings, tools, discipline, limits, conclusion | 3 |

**If you are cut to 10 minutes,** keep slides **3, 10, 11, 12, 17** and the
demo. Those five carry the entire argument. Drop 13, 14, 18 first.

## 3.2 Splitting between two presenters

A clean split, if you want one:

* **Presenter A** — slides 1–11 (framing, system, the accounting error, the
  headline) and runs the live demo.
* **Presenter B** — slides 12–24 (statistics, the four supporting results,
  method findings, tools, plan).

Whoever does *not* hold slide 3 should be the one to say "we found it
ourselves" — it lands better as a shared admission than a solo one.

## 3.3 The question bank

| Question | Answer | Slide |
|---|---|---|
| "Isn't this just RAG?" | RAG is one of our arms — the no-policy control. RAG and P1-S hold identical turns, retrieved identically. The only difference is which get forgotten, worth +8.7 points. | 7, 11 |
| "Why not GPT-4 / a bigger model?" | Determinism and cost — 35 configs × 23 ablations × 10,000 bootstrap samples has to be re-runnable. And we *did* test an LLM scorer; it lost. | 16, 19 |
| "Ten conversations is small." | Agreed. That's why every interval is clustered by conversation, and why one of our claims is reported as not significant. | 9, 12 |
| "Your accuracy numbers look low." | They're context recall at a deliberately tight budget, not accuracy. Oracle at the same budget is 100% — that's the point: the gap is selection, not reasoning. | 8, 17 |
| "What's actually novel here?" | Not the three tiers. The measurement — the storage-budget correction, and four negative results the field currently reports the other way. | 5, 24 |
| "Why did the tech stack change?" | It didn't shrink. Local and free is what made the sweep affordable, and a framework would confound the variable under study. | 19 |
| "Is the leak definitely gone?" | It's pinned by three regression tests: strip every label and the store is bit-identical. Eviction and demotion were added to that test before we quoted them. | 3, 20 |
| "Why does compression fail? It shouldn't." | It fails on *answer* recall while winning *evidence* recall, because a gist keeps the id and drops the text. And we tested the theoretical prediction to 182× — no crossover. | 8, 11, 13 |
| "Can you show it working live?" | Yes — the dashboard. *(Only offer this if you have time; the metric toggle is the demo.)* | — |
| "When will it be finished?" | The evaluation is finished. The end-task run is 4–6 hours of machine time and is the critical path; three extensions are scoped with estimates. | 22, 23 |

## 3.4 Things not to do

* **Don't** open with the architecture diagram. Open with the withdrawal.
* **Don't** let the 12-question end-task table be read as a result. Say
  "smoke run" out loud even though it's printed on the slide.
* **Don't** quote evidence recall and answer recall interchangeably. If you
  quote a strict-recall number, name it as strict.
* **Don't** claim the evidence-recall improvement over RAG. It's p = 0.16.
* **Don't** call P3-learned a component. It's a headroom bound.
* **Don't** browse the dashboard live beyond the metric toggle. One
  interaction, 60 seconds, back to the deck.

## 3.5 Pre-review checklist

- [ ] Open the .pptx once and page through all 28 slides.
- [ ] Open `frontend/index.html` and practise the metric toggle twice.
- [ ] Decide the presenter split and mark it on a printed slide list.
- [ ] Have `docs/REVIEW2_PROGRESS_REPORT.md` ready to send afterwards.
- [ ] If the panel wants repo access: the repo is **private** — add them as
      collaborators, or `gh repo edit ruthme1ster/ruthme1ster --visibility public`.
- [ ] Know these five numbers cold, without looking:
      **+8.67** · **−6.71** · **182×** · **−5.87** · **95% → 50%**.

# MemGate — Review 2 speech

**The spoken script for `MemGate_Capstone_Review2_NMIMS.pptx` (28 main slides + 2 backup).**
Simar Singh Khanuja & Yash Ramchandani · SVKM's NMIMS, Indore Campus · Review 2, October 2026

Each slide's section below is also loaded into that slide's speaker notes by
`docs/make_review2_deck.py`, so it shows in PowerPoint's Presenter View. Edit
it here and rebuild the deck. Don't edit the notes in PowerPoint.

**Timing.** About 3,000 words, so roughly **20 minutes** at a normal speaking
pace. If the slot is 15 minutes, keep slides 3, 10, 11, 17, 18 and 26 in full
and give slides 13, 14, 19, 20 and 21 one sentence each.

**Wording.** Read for meaning, not word for word. The numbers are exact, so
quote them as written. For the background behind any slide, and the question
bank, see `docs/REVIEW2_SPEAKER_NOTES.md`.

---

## Slide 1 — Title  ·  ~30 s

Good morning. We are Simar Singh Khanuja and Yash Ramchandani, and this is
Review 2 of our capstone project, MemGate.

At Review 1 we proposed an adaptive memory layer for long-running LLM agents.
Since then the project has become a measurement. When an agent's memory is
full, which turns should it keep and which should it forget? And does that
choice actually matter? We'll show you what we found, including the results
that went against us.

## Slide 2 — Where the project stands  ·  ~50 s

Every workstream on this table is complete. We ran eleven experiments end to
end on the LoCoMo benchmark and then tested the main result on a second
benchmark, LongMemEval. Every claim has a confidence interval clustered by
conversation. Every number on these slides is generated from a committed
results file, so nothing is typed by hand.

The headline: choosing which turns to forget, instead of forgetting the oldest
first, is worth **8.7 answer-recall points**. Storage, fidelity and retrieval
are held identical. We'll also be upfront that on the second benchmark this
result does not hold. Slides 18 and 19 show why.

*Next:* But one thing has to come first.

## Slide 3 — First: a Review 1 number is withdrawn  ·  ~60 s

At Review 1 we showed 95% recall for MemGate against 10% for a sliding window.
That number was wrong.

The memory policy was reading a field called fact_id, which is the
ground-truth evaluation label, when deciding what to keep. In effect, it was
reading the answer key. The leak was worth 45 to 47 points. The honest number
on the same test set is **50%, not 95%**.

We fixed it with a test, not a comment. If you strip every evaluation label
from the input, the memory store must come out bit-identical. Every decision
point we added later went under that test before we quoted any number from it.

Everything after this slide is measured on a real benchmark, with no leak. We
would rather tell you this ourselves than have you find it.

## Slide 4 — The problem  ·  ~50 s

Why does agent memory still matter when models have 200,000-token context
windows? There are three reasons.

First, context rot. Chroma Research tested 18 frontier models, and accuracy
drops 30 to 50 percent well before the advertised limit. A bigger window
doesn't fix that. Choosing what goes into it does.

Second, cost. On LoCoMo, the full context is about 26,000 tokens and nearly 10
seconds per answer. Selective memory reaches similar accuracy with about 1,800
tokens in 0.7 seconds, roughly **14 times cheaper**.

Third, on LongMemEval, a model given the right evidence scores about 92%, but
the same model working interactively scores about 58%. That 34-point gap is a
failure to select memory, not a failure to reason. A 2026 survey names it
directly: deciding what to remember and what to forget is the unsolved
problem.

## Slide 5 — The question, and what is new  ·  ~45 s

So our research question is: how much of that gap can a better decision policy
recover, and under what budget does compression pay for itself?

We want to be honest about novelty. Tiered memory with forgetting already
ships, in Letta/MemGPT, Mem0 and Zep. If we presented "a memory layer with
three tiers", that work would already be done.

What isn't done is measuring the decision itself. Colaco and Lahjouji frame it
as a rate–distortion problem, but they give a framework, not a system. MemGate
is the harness that makes the decision measurable. Different policies plug
into the same system, and each comparison changes only one variable at a time.

Our contribution is a measurement, and several of its results are negative.

## Slide 6 — System architecture  ·  ~55 s

MemGate sits between the agent and a frozen LLM. Nothing is trained. We only
control what goes into the prompt.

On the **write path**, every new turn enters Tier 1, the short-term buffer, as
raw text. When the buffer is full, the Adaptive Memory Decision Engine scores
each evicted turn exactly once. High-scoring turns become durable facts in
Tier 3. Middle ones become compressed summaries in Tier 2. The rest are
dropped.

On the **read path**, a new query is matched against Tier 3. We retrieve the
closest items, pack them into the token budget, and send that context to the
LLM.

The key point is that the tiers are a lifecycle, not a partition. Everything
enters Tier 1, and the decision happens once, on eviction, when the budget is
full. That one decision is what this project measures.

## Slide 7 — Policies compared  ·  ~45 s

These are the policies we compare. P0 is a sliding window. RAG stores
everything and retrieves; once its storage is full it forgets first-in,
first-out, so it is our no-policy control. P1 is MemGate: P1-A compresses
what it evicts, P1-S drops it. P2 uses a small LLM to judge importance, P3 is
a learned scorer, and Oracle is perfect selection, the upper bound.

The bottom table matters most: every comparison changes exactly one thing.
RAG against P1-S holds storage, fidelity and retrieval the same, so the only
difference is which turns get forgotten. P1-S against P1-A isolates dropping
versus compressing. P0 gets the same token budget as MemGate, so it is never a
strawman.

## Slide 8 — Two metrics, and why both are reported  ·  ~45 s

We report two metrics because they disagree.

Evidence recall asks whether every gold evidence turn reached the context.
Answer recall asks whether the actual answer text survived. We score it on
the 715 questions whose answer can be recovered from the evidence, and it is
the metric to believe.

The reason is that evidence recall gives credit for a pointer. A compressed
summary keeps the ID of a turn whose text it threw away, and still counts as a
hit. During one round of development, evidence recall rose from 8.1 to 14.8
percent while answer recall stayed at 12.8 percent. The whole gain was
bookkeeping.

That disagreement turns out to be one of our findings.

## Slide 9 — Experimental setup  ·  ~40 s

The benchmark is LoCoMo: 10 long multi-session conversations, 5,882 turns and
1,527 scorable questions with gold evidence. We exclude the adversarial
category, where the right behaviour is to decline to answer.

Embeddings use MiniLM, budgets are counted in exact tokens, the judge is a
0.5-billion-parameter Qwen model, and the reader for end-task accuracy is a
1.5-billion-parameter Qwen model, held fixed across every arm. Everything runs
locally and offline, so every experiment can be re-run after every change.

One statistical point matters most. 1,527 questions from 10 conversations are
**not** 1,527 independent trials. Every interval in this deck is clustered by
conversation.

## Slide 10 — Storage was never budgeted  ·  ~60 s

This is the turning point of the project.

Every experiment in this literature, including ours up to this point, limits
the context per query but leaves the store unlimited. Under that accounting,
"keep everything and retrieve" pays nothing for holding all 419 turns of a
conversation, so forgetting can only lose. Our own ablation correctly reported
compression as a net loss.

But a real agent's memory isn't free. Once we impose a storage budget, S, the
question changes from *"what fits in the context?"* to *"what is worth keeping
at all?"*

The figure shows the result. With the context fixed at 2048 tokens and the
store swept, the ranking inverts. What looked like a loss for the decision
policy came from never charging for storage.

## Slide 11 — Selection pays; compression does not  ·  ~55 s

Here are the numbers: context fixed at 2048 tokens, storage swept, answer
recall.

P1-S, scored selection that drops whole turns, beats RAG store-all at every
budget, by 3.9 to 8.7 points. At S = 4096 it is **36.8 against 28.1**.
Compression, P1-A, sits in between.

Notice that the metrics disagree again. P1-A wins evidence recall by 12.9
points at 8192 while losing answer recall by 6.9, because its summaries keep
the IDs but not the content.

RAG against P1-S is the cleanest comparison in the project: the same verbatim
turns in the same space, retrieved the same way. The only difference is which
turns are forgotten, oldest first or lowest utility first.

## Slide 12 — What survives clustering  ·  ~45 s

Does that survive proper statistics? This is a paired bootstrap with 10,000
resamples, clustered by conversation, plus exact McNemar tests, at S = 4096.

Selection over RAG on answer recall: **+8.67 points**, interval 4.8 to 14.3,
p around 10 to the minus 4. That is significant.

On evidence recall, though, the gain over RAG is only 1.96 points, the
interval crosses zero, and p is 0.16. So we do **not** claim an
evidence-recall improvement over RAG. The comparisons against compression and
against recency are all significant.

We report the non-significant result ourselves, so the panel doesn't have to
find it.

## Slide 13 — The rate–distortion prediction fails  ·  ~45 s

Theory says that under enough pressure, compression must eventually win:
keeping a blurry version of everything should beat keeping a sharp version of
a little. We tried to find that crossover.

Shrinking the budget can't test it. By S = 256 every policy is near zero. So
we stretched the length instead. We concatenated up to 10 conversations,
186,000 tokens, and held storage fixed, pushing the compression ratio to 182
times.

There is no crossover anywhere, and selection's margin over RAG actually
widens as the pressure rises. One caveat: concatenated streams are synthetic.
This is a stress test, not a natural long conversation.

## Slide 14 — The index is not free  ·  ~45 s

There is another accounting problem. Counting storage in text tokens hides the
cost of the embedding index. A 384-dimensional vector is 1,536 bytes, while a
typical LoCoMo turn is about 130 bytes of text. The index is about **12 times
larger** than the content it indexes.

When we budget in bytes, two things happen. Our selection result still holds:
P1-S beats RAG at every byte budget. And below roughly 40 kilobytes, plain
recency wins outright, because it keeps no vectors at all. At 32 KiB, P0 gets
18.3% against 5% for RAG. That regime is invisible if you only count text.

## Slide 15 — The scorer is the bottleneck, and it is learnable  ·  ~50 s

Which component actually matters? On the left is an ablation that removes one
component at a time. Retrieval is the largest, at minus 30 points. Scored
eviction is worth about 5.

So is the scorer the limit? We trained a learned scorer, evaluated
leave-one-conversation-out, so it never sees the conversation it is scoring.
With hand-built features it reaches an AUC of 0.746. Adding MiniLM embeddings
takes it to 0.793 and lifts answer recall from 26.3 to 31.5 percent.

The interesting part is what it learned. **Turn length** is the single
strongest predictor of whether a turn is evidence, and our hand-tuned
heuristic never encoded it.

## Slide 16 — An LLM judge is the wrong instrument for salience  ·  ~50 s

The obvious way to improve the scorer is to ask an LLM. We tested that with a
single-component swap: everything else stays fixed and only the scorer
changes.

First, scoring matters. The heuristic beats unscored recency by 11.2 points.

But the LLM judge loses to our simple regex heuristic by **5.9 points**, and
the interval excludes zero. It also costs about 15 minutes of GPU time per
corpus, where the heuristic takes microseconds. And there is room to improve:
the learned bound is 11.5 points above the heuristic.

The reason is diagnosable. The judge sees each turn in isolation, and
importance isn't a property of a turn on its own. A phone number matters
because something later asks for it.

## Slide 17 — End-task accuracy: the gap survives a real reader  ·  ~50 s

Everything so far measures whether the evidence reached the prompt. Does it
change real answers?

We ran a fixed reader model on all 715 answer-recoverable questions.
Closed-book, with no memory, gets 1.7%, so the model is not answering from its
own knowledge. Oracle gets 67.3%, the ceiling.

P1-S reaches **18.5% accuracy against 12.4% for RAG**. That means 6 points of
the 8.7-point recall gap turn into correct answers, with a clustered interval
of 3.3 to 9.1, so about 69% of the gap converts. Because closed-book is so
low, nearly all of that margin comes from the memory system.

## Slide 18 — We tested our own headline on a second benchmark, and it failed  ·  ~60 s

Then we tested our own headline on a second benchmark, and it failed.

LoCoMo is only ten conversations. LongMemEval has 94 independent haystacks of
38 to 62 sessions each. At storage matched to LoCoMo's retention, about 22%,
selection **loses to FIFO by 21 answer points**, where LoCoMo gave plus 8.7.

Why? The mechanism didn't fail, the scorer did. Our heuristic scorer separates
evidence from non-evidence turns in opposite directions on the two corpora:
positive on LoCoMo, negative on LongMemEval. Eviction is ordered by that
score, so on LongMemEval it throws evidence away first. That is worse than no
ranking at all.

The scorer was written for LoCoMo's short 32-token lines, while LongMemEval's
turns are 210-token prose.

## Slide 19 — The obvious alternative explanation, ruled out  ·  ~45 s

Before accepting a result against our own headline, we tried to break it.

The suspicion was that MemGate reserves 25% of its context for recent turns
and RAG reserves nothing. On LongMemEval the answer sits in a random session,
so that reserve could be wasted.

So we removed it, keeping the same code, scorer, budgets and retrieval depth;
only that one setting changed. Answer recall moved by exactly zero: **19.4%
either way**.

The confound is real but not load-bearing. Selection still loses by 21
points, so the scorer-transfer failure explains the whole result.

## Slide 20 — Three findings about measurement  ·  ~50 s

Three of our findings are about measurement itself, and each was caught by an
automated check, not by reading the code.

One: the answer-key leak from slide 3, worth 45 to 47 points.

Two: a component's value depends on where the bottleneck is. The MiniLM
embedder measured at plus 0.2 points while the write path was throwing away
93% of the conversation. Once that ceiling lifted, the same swap was worth
19.7.

Three: a metric that credits pointers. Evidence recall rose while answer
recall stayed flat.

And a smaller fourth: the judge returned NaN, "not a number", for short turns
because of left-padding. Without a guard, every filler turn would have
silently inverted the eviction order.

## Slide 21 — Tools and technology  ·  ~45 s

This slide compares what we proposed at Review 1 with what we actually used.

Instead of Llama or a hosted API, we ran Qwen 2.5 models locally: no keys, no
cost, and bit-reproducible results. We used no LangChain or LlamaIndex,
because a framework's own retrieval would interfere with the very variable we
are studying. We used plain NumPy cosine search instead of FAISS, since at ten
conversations an approximate index only adds error. And we added exact
tokenisation and clustered statistics.

This isn't scope reduction. Running locally and for free is what made 35
storage configurations, 23 ablations and a 10,000-sample bootstrap
affordable to re-run after every change.

## Slide 22 — Verification and engineering discipline  ·  ~40 s

Every guard on this slide exists because the failure it prevents already
happened once.

There are 68 regression tests, plus three label-blindness checks that re-run
the leak test whenever we add a decision point. Offline pinning stops a
half-downloaded model from silently relabelling a results table; that
happened once with MiniLM. Then there is provenance logging, the NaN guard,
caches keyed by model and prompt so long runs are reproducible and resumable,
and results committed alongside the code.

The rule we adopted: the fix for a leak is a test, not a comment.

## Slide 23 — Deliverables  ·  ~30 s

Everything is in the repository: the paper-ready report, this review's
progress report, a complete working record, an interactive results dashboard,
the memgate library of about 6,500 lines with 17 runnable experiments, and a
6-page IEEE-format paper.

Every table in the paper and every number in this deck is generated from
committed results files, including this deck itself. You can check any number
without re-running the benchmark.

## Slide 24 — Timeline  ·  ~35 s

On the timeline, the filled bars are complete. Every planned track was closed
by the end of September: the literature review in July, the harness and the
LoCoMo evaluation, the storage and significance work, LongMemEval, abstractive
compression, online learning and the compiled paper.

We are at Review 2 now, in early October. Before the **final review in the
first week of November**, two things remain. The first is a scorer that
transfers across corpora, which LongMemEval made the live question. The
second is the final report and defence preparation.

## Slide 25 — Limitations, and what remains  ·  ~55 s

Here are our limitations.

Ten conversations means wide intervals. Context recall is a ceiling on
accuracy, not accuracy itself. Compression is now tested in both forms:
rewriting beats word-selection by 4.1 points but still loses to dropping, at
1,850 times the cost. Concatenated streams are synthetic. We used one embedder
and one judge model. P3 is an upper bound, not something an agent could run
live. And the scorer is tuned to LoCoMo, so plus 8.7 is what selection
recovers *given a scorer that suits the corpus*.

On the right is what remains. First and most important is a scorer that
transfers across corpora. The obvious first try is to run our online learner
on LongMemEval. After that come a larger summariser and a second reader
model.

## Slide 26 — Conclusion as it stands  ·  ~50 s

To conclude:

With storage unlimited, tiered compression is a net loss. With storage
bounded, the decision policy is worth **8.7 answer-recall points**, and
compression still loses. The rate–distortion crossover is not reachable even
at 182 times compression. Charging for the index flips the ranking below about
40 KiB. An LLM judge loses to a regex, for a reason we can diagnose. And on a
second benchmark our headline does not replicate, because of the scorer, not
the mechanism.

The practical guidance is this: keep a scored subset whole, don't compress
everything, and budget in bytes, including the index.

## Slide 27 — References  ·  ~10 s

These are our main references. MemGPT is the base paper and LoCoMo is the
benchmark. We checked every citation against its source.

*(Don't read the list. Move on.)*

## Slide 28 — Thank you  ·  ~10 s

Thank you. We're happy to take questions, including on the results that did
not work.

## Slide 29 — Backup: the full storage sweep

*(Show only if asked.)* This is every storage configuration we ran: all five
policies at every budget from 512 to 16,384 tokens, with the context fixed at
2048. The rows behind slide 11 are here.

## Slide 30 — Backup: repository layout

*(Show only if asked.)* This is how the repository is organised. Every
experiment has its own script, and every result it produces is committed
under memgate/results.

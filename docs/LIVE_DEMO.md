# MemGate — live demo guide

There are two live pieces, and both run on one page:

1. **Chat through MemGate.** You type, the local model answers, and the three
   tiers update on screen. Each card shows the message's score, what was kept,
   compressed or forgotten, and the exact context the model was given.
2. **Benchmark check.** This re-runs the slide 11/12 comparison on all of
   LoCoMo in about 20 seconds and checks the result against the committed CSV.
   The expected output is RAG 28.1%, P1-S 36.8%, both **MATCH**, +8.7 points.

Everything runs offline on the laptop. No internet and no API key are needed.

## Starting it

```bash
cd memgate
python3 live_demo.py          # loads MiniLM + Qwen2.5-1.5B (~10 s), then:
# open http://localhost:8765
```

* `python3 live_demo.py --no-llm` is the fallback if the model won't load. The
  memory still works and the page shows what the model *would* have been sent.
* `python3 run_live_check.py` runs the benchmark check from a terminal on its
  own, without the page.
* Start it **before** the review and leave it running. Then close other heavy
  apps. The machine has 8 GB of memory and the demo uses about 2–3 GB.

## What's on screen

| Area | What it shows |
|---|---|
| **Chat** (left) | The conversation. The model sees *only* what MemGate sends, never the history. |
| **Tier 1** | The last 4 messages, verbatim. Every message enters here. |
| **Tier 2** | Compressed gists. |
| **Tier 3** | Kept facts with their vectors, retrieved by meaning. A struck-through card has been superseded by a newer fact. |
| **Decisions** | Every routing decision: the score *u*, its label, and where the message went and why. |
| **Context sent** | The exact notes put in the prompt for the last message, tagged by tier. |
| **Mode** | **P1**: threshold routing, exactly slide 6. **P1-S**: the headline policy, which forgets the lowest score first once the store is full. **P1-A**: demotes to a gist instead of forgetting. |

The demo uses small sizes so decisions happen within a few messages instead
of a few hundred. Tier 1 holds 4 messages, the context is 512 tokens, and the
store budget is S = 150. Those are the only settings that differ from the
experiments. Every decision on screen comes from the same `observe()` and
`build_context()` code the results came from.

## A 3-minute script

1. **Mode P1, then *Load sample chat*.** "This is 18 messages of a normal
   conversation. Watch where each one went." Point at three things:
   * "Hi!" and "Sure." were **forgotten**: they're filler, with u ≈ 0.02.
   * The deadline went to **Tier 3** as a durable fact (u = 0.80).
   * The assistant's "Got it: report due 2 November" is **struck through**.
     The correction "Actually the report deadline moved to 4 November"
     superseded it. Supersession retires only the *single closest* old fact,
     so the user's original sentence with the old date is still in Tier 3. If
     asked, say so: it's a deliberate design choice (store.py) so that one
     update can't wipe out half the store.
2. **Ask: "When is my report due?"** The model answers 4 November. Point at
   *Context sent*: "The model never saw the conversation, only these notes,
   tagged by the tier they came from. The correction and its higher score are
   in there, so the model takes the newer date."
3. **Switch to P1-S** (S = 150), *Load sample chat* again. "This is the
   headline policy under a storage budget. Watch the store fill up, then watch
   it forget the lowest-scored messages first." The Decisions log shows
   *store over budget: lowest-score fact forgotten*.
4. **Use the scorer's weakness, honestly.** Under P1-S, "My guide wants the
   slides in the NMIMS template" scores only 0.15, the same as "lol yes", and
   is forgotten first because it's older. "This is slide 15 happening live: the hand-written scorer
   doesn't know that a preference matters. That's why the scorer is the
   bottleneck, and why a scorer that transfers is our remaining work." The
   question still gets answered, because the assistant's restatement of it
   survived. Say that too.
5. **Benchmark check → Run.** "These are the numbers on slide 11, recomputed
   now from the raw benchmark: 1,527 questions, about 20 seconds." End on the
   two **MATCH** badges.

Let the panel type their own message too. That is the strongest moment of the
demo.

## What it can't show, and how to answer if asked

* **Accuracy on the panel's own chat.** There's no answer key for a live
  conversation, so there's nothing to score against. That is what the
  benchmark check is for.
* **The model is small (1.5B).** Its replies are plain, and it can be wrong
  even when the right note is in its context. That is the model reading
  badly, not the memory failing, and the context panel shows the
  difference. The demo is about the memory decisions, not the chatbot.
* **End-task accuracy (slide 17) is not re-run live.** It takes hours. The
  benchmark check reproduces context recall, which is the ceiling it is
  measured against.

## If something breaks

* **The page shows an error:** press *Reset* and continue.
* **The model hangs:** restart with `--no-llm`. The tiers, decisions and
  context panel all still work.
* **The benchmark check says DIFFERS:** check the info line at the top right.
  If the embedder is `hash-bow` rather than `all-MiniLM-L6-v2`, MiniLM failed
  to load and the numbers are not expected to match. Restart the demo.
* **Keep a screen recording** of one clean run of the script above as a
  backup.

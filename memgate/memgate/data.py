"""Synthetic multi-session dataset for Step 0.

This is a SANITY HARNESS, not a result. It exists so the pipeline and metrics
can be validated with zero downloads. Phase 1 replaces it with LoCoMo /
LongMemEval via `load_locomo()`.

Design goals (so the experiment is actually informative):
  * many durable facts, so at small budgets a policy MUST choose between them
    -> produces a real accuracy-per-token curve rather than a flat line
  * facts planted across all sessions and probed at the end, so recency alone
    recovers only the most recent few -> P0 gets partial, not zero, credit
  * heavy filler and generic chatter, the realistic signal-to-noise case
  * one fact update, to exercise supersession
"""
import ast
import json
import os
import random
import re
from typing import List, Tuple
from .types import Turn, Question
from .utils import answer_tokens

# --- fact templates: (category, statement, probe) ---------------------------
TEMPLATES = [
    ("project", "The {thing} is a {adj} app for {who}, we called it {name}.",
     "What did we call the {thing}?"),
    ("deadline", "The deadline for the {thing} is {month} {day}.",
     "When is the deadline for the {thing}?"),
    ("budget", "My budget for the {thing} is {num} rupees.",
     "What is my budget for the {thing}?"),
    ("decision", "I decided to use {tech} for the {thing} instead of {alt}.",
     "Which technology did I choose for the {thing}?"),
    ("person", "{name} is handling the {thing} for us.",
     "Who is handling the {thing}?"),
    ("pref", "I always prefer {pref} when working on the {thing}.",
     "What do I prefer when working on the {thing}?"),
    ("contact", "You can reach {name} at {name2}@example.com about the {thing}.",
     "What is the contact email for the {thing}?"),
    ("location", "The {thing} review happens in {place} every {month}.",
     "Where does the {thing} review happen?"),
]

NAMES = ["Priya", "Ravi", "Ananya", "Karthik", "Meera", "Arjun", "Divya",
         "Rohan", "Sneha", "Vikram", "Nisha", "Aditya", "Kavya", "Manish"]
_THING_ADJ = ["bakery", "billing", "mobile", "analytics", "payment",
              "onboarding", "search", "admin", "notification", "report"]
_THING_NOUN = ["site", "module", "client", "board", "flow", "page",
               "feature", "panel", "service", "export"]
# 100 unique combinations. Each fact MUST get a unique subject, otherwise two
# different fact_ids can render identical text, the deduper correctly collapses
# them, and the second becomes permanently unanswerable — a dataset artefact
# that would otherwise be misread as a memory failure.
THINGS = [f"{a} {n}" for a in _THING_ADJ for n in _THING_NOUN]
TECH = ["Postgres", "Redis", "FastAPI", "Django", "Next.js", "Kotlin",
        "Flutter", "GraphQL", "Kafka", "SQLite", "Vue", "Svelte"]
ALT = ["MongoDB", "Memcached", "Flask", "Rails", "Nuxt", "Java", "Ionic",
       "REST", "RabbitMQ", "MySQL", "Angular", "Ember"]
ADJ = ["React", "Android", "internal", "offline-first", "dashboard"]
WHO = ["my dad's bakery", "our college fest", "a local clinic", "my uncle's shop"]
NAMEZ = ["Sweet Crumb", "FestTrack", "ClinicLog", "ShopMate", "QuickTally"]
MONTHS = ["March", "April", "June", "July", "September", "November"]
PREFS = ["dark mode", "typed interfaces", "small commits", "written specs",
         "morning standups", "automated tests"]
PLACES = ["the Indore campus", "the annexe lab", "the third floor room",
          "the online call", "the library block"]

FILLER = ["ok", "thanks", "cool", "great", "got it", "sure", "hi", "hey",
          "good morning", "sounds good", "nice", "yeah", "no problem", "bye",
          "right", "understood", "perfect", "alright"]

CHATTER = [
    "Let me think about that for a moment.",
    "That makes sense to me overall.",
    "I was reading about this yesterday evening.",
    "We can look at that a little later if you want.",
    "It has been a fairly busy week for everyone.",
    "I am not entirely sure what the best approach here is.",
    "Let us come back to this topic quite soon.",
    "That is an interesting way of putting it honestly.",
    "I might need to review the whole thing once more.",
    "We should probably keep moving along for now.",
    "There is quite a lot going on this month.",
    "I have been meaning to bring that up actually.",
]

ASSISTANT = [
    "Understood, I can definitely help with that.",
    "Sure, what would you like to do next here?",
    "That sounds like a reasonable plan to me.",
    "Noted. Anything else on your mind today?",
    "Happy to help you work through all of it.",
    "Got it, I will keep that in mind going forward.",
]


def build_dataset(n_sessions: int = 20, turns_per_session: int = 30,
                  n_facts: int = 40, seed: int = 7) -> Tuple[List[Turn], List[Question]]:
    rng = random.Random(seed)
    facts = []          # (fact_id, statement, probe)

    for i in range(n_facts):
        cat, stmt, probe = TEMPLATES[i % len(TEMPLATES)]
        slots = {
            "adj": rng.choice(ADJ), "who": rng.choice(WHO),
            "name": rng.choice(NAMES), "name2": rng.choice(NAMES).lower(),
            "thing": THINGS[i % len(THINGS)],   # unique per fact
            "month": rng.choice(MONTHS), "day": str(rng.randint(2, 28)),
            "num": str(rng.choice([15000, 22000, 40000, 65000, 90000])),
            "tech": TECH[i % len(TECH)], "alt": ALT[i % len(ALT)],
            "pref": rng.choice(PREFS), "place": rng.choice(PLACES),
        }
        if cat == "project":
            slots["name"] = rng.choice(NAMEZ)
        facts.append((f"f{i:02d}_{cat}", stmt.format(**slots),
                      probe.format(**slots), slots["thing"]))

    # spread facts evenly across sessions
    per_session = max(1, n_facts // n_sessions)
    plant = {}
    for i in range(n_facts):
        plant.setdefault(min(i // max(1, per_session), n_sessions - 1), []).append(i)

    turns: List[Turn] = []
    for s in range(n_sessions):
        idx = 0
        for fi in plant.get(s, []):
            fid, stmt, _, _ = facts[fi]
            turns.append(Turn(s, idx, "user", stmt, fact_id=fid)); idx += 1
            turns.append(Turn(s, idx, "assistant", rng.choice(ASSISTANT))); idx += 1

        # One supersession event, mid-run.
        #
        # The update NAMES ITS SUBJECT. It used to read "we pushed that deadline
        # to April 2" -- an anaphor with no antecedent, fired ten sessions after
        # its target, competing with four other deadline facts. No mechanism
        # could resolve that, and semantic supersession duly retired the wrong
        # record. An unresolvable probe measures nothing, so it named the
        # subject instead; `_assert_wellformed` now enforces it.
        if s == n_sessions // 2 and facts:
            fid, _, _, thing = facts[1]
            turns.append(Turn(s, idx, "user",
                              f"Actually, we pushed the {thing} deadline "
                              f"to April 2 instead.",
                              fact_id=fid)); idx += 1

        while idx < turns_per_session:
            r = rng.random()
            if r < 0.40:
                turns.append(Turn(s, idx, "user", rng.choice(FILLER), is_filler=True))
            elif r < 0.72:
                turns.append(Turn(s, idx, "user", rng.choice(CHATTER)))
            else:
                turns.append(Turn(s, idx, "assistant", rng.choice(ASSISTANT)))
            idx += 1

    # probe every fact at the very end: maximum distance from where it was said
    questions = [Question(text=p, gold_fact_id=fid,
                          asked_after_session=n_sessions - 1)
                 for fid, _, p, _ in facts]

    _assert_wellformed(facts, turns, questions)
    return turns, questions


def _assert_wellformed(facts, turns, questions):
    """Guard against dataset artefacts that would be misread as model failures."""
    stmts = [s for _, s, _, _ in facts]
    if len(set(stmts)) != len(stmts):
        raise AssertionError(
            "duplicate fact statements: two fact_ids share identical text, so "
            "the deduper will collapse them and one becomes unanswerable")
    probes = [p for _, _, p, _ in facts]
    if len(set(probes)) != len(probes):
        raise AssertionError("duplicate probe questions: gold answer is ambiguous")

    # An update must name the subject it corrects. A bare anaphor ("that
    # deadline") has no resolvable antecedent when several facts share the
    # category, so supersession cannot be scored -- it silently measures which
    # distractor happens to sit closest in embedding space.
    upd = [t for t in turns if t.text.startswith("Actually,")]
    for t in upd:
        thing = next((th for fid, _, _, th in facts if fid == t.fact_id), None)
        if thing and thing not in t.text:
            raise AssertionError(
                f"update turn does not name its subject {thing!r}: {t.text!r} "
                "-- unresolvable, so supersession cannot be evaluated")
    planted = {t.fact_id for t in turns if t.fact_id}
    missing = {q.gold_fact_id for q in questions} - planted
    if missing:
        raise AssertionError(f"probed but never planted: {sorted(missing)}")
    return True


# --------------------------------------------------------------------- LoCoMo

DEFAULT_LOCOMO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "locomo", "locomo10.json")

_SESSION_KEY = re.compile(r"^session_(\d+)$")

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain",
                  4: "single_hop", 5: "adversarial"}
ADVERSARIAL = 5


def load_locomo(path: str = DEFAULT_LOCOMO_PATH, include_adversarial: bool = False):
    """Load LoCoMo (Maharana et al., 2024) into Turn/Question.

    Returns [(sample_id, turns, questions), ...] -- one entry per conversation.
    The ten conversations are NOT concatenated: they have different speakers and
    no shared referents, so joining them would invent cross-conversation
    distractors that the benchmark never intended.

    Two data hazards, both found by inspection and both silent if unhandled:

    1. `dia_id` is unique only WITHIN a conversation. Across the file there are
       5882 turns but just 1033 distinct ids -- "D1:1" exists in every
       conversation. Ids are namespaced as "<sample_id>/<dia_id>" so evidence
       can never match a turn from a different conversation.

    2. Nine evidence ids reference turns that do not exist in their
       conversation. Those questions are unanswerable by construction, so they
       are dropped rather than left to depress every policy's recall equally --
       the same class of dataset artefact that Step 0's `_assert_wellformed`
       was added to catch.

    Adversarial questions (category 5) are excluded by default. They do carry
    evidence, but the correct behaviour is to decline to answer, so retrieving
    the evidence does not indicate success and mixing them into a retrieval
    metric makes it meaningless. Pass include_adversarial=True to study them.

    Speaker names are kept verbatim ("Caroline", "Melanie") rather than mapped
    onto user/assistant. LoCoMo is two humans talking; labelling one of them
    "assistant" would trip the scorer's assistant down-weight arbitrarily.
    """
    with open(path) as f:
        raw = json.load(f)

    out, dropped_dangling, dropped_empty, dropped_adv = [], 0, 0, 0
    for sample in raw:
        sid = sample["sample_id"]
        conv = sample["conversation"]

        turns: List[Turn] = []
        known_ids = set()
        by_id = {}
        sessions = sorted(
            ((int(m.group(1)), k) for k in conv
             for m in [_SESSION_KEY.match(k)] if m))
        for s_pos, (_, key) in enumerate(sessions):
            for idx, t in enumerate(conv[key]):
                uid = f"{sid}/{t['dia_id']}"
                known_ids.add(uid)
                by_id[uid] = t.get("text", "")
                turns.append(Turn(session=s_pos, index=idx,
                                  speaker=t.get("speaker", "user"),
                                  text=t.get("text", ""), fact_id=uid))

        last_session = len(sessions) - 1
        questions: List[Question] = []
        for q in sample.get("qa", []):
            cat = int(q["category"])
            if cat == ADVERSARIAL and not include_adversarial:
                dropped_adv += 1
                continue
            ev = q.get("evidence") or []
            if isinstance(ev, str):              # defensive: some dumps stringify it
                ev = ast.literal_eval(ev)
            ev = [f"{sid}/{e}" for e in ev]
            if not ev:
                dropped_empty += 1
                continue
            if any(e not in known_ids for e in ev):
                dropped_dangling += 1
                continue
            ans = str(q.get("answer", "") or "")
            ev_text = " ".join(by_id.get(e, "") for e in ev)
            aw = answer_tokens(ans)
            recoverable = bool(aw) and set(aw) <= set(answer_tokens(ev_text))
            questions.append(Question(
                text=q["question"], gold_fact_id=ev[0], gold_evidence=ev,
                asked_after_session=last_session,
                kind=CATEGORY_NAMES.get(cat, "unknown"), category=cat,
                answer=ans, answer_recoverable=recoverable))

        _assert_locomo_wellformed(sid, turns, questions)
        out.append((sid, turns, questions))

    if not out:
        raise ValueError(f"no conversations parsed from {path}")
    print(f"  LoCoMo: {len(out)} conversations, "
          f"{sum(len(t) for _, t, _ in out)} turns, "
          f"{sum(len(q) for _, _, q in out)} questions "
          f"(dropped {dropped_dangling} dangling-evidence, "
          f"{dropped_empty} empty-evidence"
          f"{'' if include_adversarial else f', {dropped_adv} adversarial'})")
    return out


def _assert_locomo_wellformed(sid, turns, questions):
    ids = [t.fact_id for t in turns]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{sid}: duplicate turn ids after namespacing")
    known = set(ids)
    for q in questions:
        missing = set(q.evidence_ids) - known
        if missing:
            raise AssertionError(f"{sid}: evidence not in conversation: {sorted(missing)}")
    return True

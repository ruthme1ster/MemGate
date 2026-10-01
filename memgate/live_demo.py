#!/usr/bin/env python3
"""Live demo: chat through MemGate and watch what each tier holds.

    python3 live_demo.py                 # then open http://localhost:8765
    python3 live_demo.py --no-llm        # no model: replies show the context only
    python3 live_demo.py --port 9000

Every message -- yours and the model's reply -- goes through the same
`policy.observe()` the experiments use, and every question is answered from
the same `policy.build_context()`: the model sees ONLY the memory MemGate
assembled, never the raw history. The page shows the three tiers as they are,
an event log of each routing decision (score, label, destination), and the
exact context sent with each question.

The benchmark panel runs `run_live_check.py` in a subprocess and streams its
output: the LoCoMo answer-recall numbers re-computed in front of the panel and
compared with the committed CSV.

Nothing here is a mock-up. The demo deliberately uses small capacities so that
eviction happens within a few messages instead of a few hundred; those are
the only settings that differ from the experiments.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memgate.types import Turn, SHORT_TERM, WORKING, LONG_TERM
from memgate.policies import (MemGatePolicy, MemGateAdaptivePolicy,
                              MemGateSelectPolicy)
from memgate.utils import backend_info, count_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "demo", "index.html")
TIER_KEY = {SHORT_TERM: "t1", WORKING: "t2", LONG_TERM: "t3"}

# Demo-scale settings. The experiments use the same classes at LoCoMo scale.
SHORT_CAPACITY = 4          # Tier 1 holds the last 4 messages (2 exchanges)
CONTEXT_BUDGET = 512        # tokens of memory sent with each question
DEFAULT_STORE = 150         # store budget S for P1-S / P1-A, in tokens

MODES = {
    "p1": "P1 MemGate · threshold routing (slide 6)",
    "p1s": "P1-S select · scored eviction under a store budget",
    "p1a": "P1-A adaptive · demote to a gist under pressure",
}

SYSTEM = ("You are a helpful assistant in a long-running chat. You do not see "
          "the conversation history. You only see the memory notes below, "
          "which a memory system selected for this message. Use them to "
          "answer. If the notes do not contain what is asked, say you don't "
          "remember it. Keep replies short: one or two sentences.")

SAMPLE = [  # A scripted opening, loaded instantly, for asking questions about.
    ("user", "Hi!"),
    ("assistant", "Hello! How can I help you today?"),
    ("user", "My name is Simar and I'm a final-year student at NMIMS Indore."),
    ("assistant", "Nice to meet you, Simar. What are you working on?"),
    ("user", "My capstone final review is on 6 November and the report is due on 2 November."),
    ("assistant", "Got it: report due 2 November, final review on 6 November."),
    ("user", "ok thanks"),
    ("assistant", "You're welcome!"),
    ("user", "My guide wants the slides in the NMIMS template."),
    ("assistant", "Understood, the slides should use the NMIMS template."),
    ("user", "lol yes"),
    ("assistant", "Sure."),
    ("user", "Actually the report deadline moved to 4 November."),
    ("assistant", "Updated: the report is now due on 4 November."),
    ("user", "My teammate Yash runs the LongMemEval experiments."),
    ("assistant", "Noted, Yash is running LongMemEval."),
    ("user", "cool"),
    ("assistant", "Anything else?"),
]


# ------------------------------------------------------------------ model
class ChatModel:
    """The local reader the end-task run uses, sampled greedily."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.name = "none (--no-llm)"
        self._model = self._tok = None
        self.lock = threading.Lock()
        if enabled:
            from memgate.llm_judge import default_model
            self.path = default_model()
            self.name = os.path.basename(self.path)

    def load(self):
        if self.enabled and self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.path)

    def reply(self, notes: str, message: str) -> str:
        if not self.enabled:
            return "(no model loaded: the memory notes above are what it would see)"
        with self.lock:
            self.load()
            from mlx_lm import generate
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                     f"Memory notes:\n{notes or '(empty)'}\n\nMessage: {message}"}]
            prompt = self._tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            out = generate(self._model, self._tok, prompt=prompt,
                           max_tokens=100, verbose=False)
        return " ".join(out.strip().split())


# ---------------------------------------------------------------- session
def _item(it):
    return {"id": it.id, "text": it.text, "tier": TIER_KEY[it.tier],
            "kind": it.kind, "utility": round(float(it.utility), 2),
            "label": it.type_label, "turn": it.turn_index,
            "tokens": count_tokens(it.text),
            "superseded": it.superseded_by is not None}


class Session:
    def __init__(self, mode="p1", store_budget=DEFAULT_STORE):
        self.mode = mode if mode in MODES else "p1"
        self.store_budget = int(store_budget)
        kw = dict(short_capacity=SHORT_CAPACITY, budget=CONTEXT_BUDGET)
        if self.mode == "p1":
            self.policy = MemGatePolicy(**kw)
        elif self.mode == "p1s":
            self.policy = MemGateSelectPolicy(store_budget=self.store_budget, **kw)
        else:
            self.policy = MemGateAdaptivePolicy(store_budget=self.store_budget, **kw)
        self.n = 0
        self.log = []          # every routing event, newest last
        self.lock = threading.Lock()

    # -- what the store holds right now
    def _held(self):
        st = self.policy.store
        return ({i.id: i for i in st.short}, {i.id: i for i in st.working},
                {i.id: i for i in st.long})

    def state(self):
        st = self.policy.store
        s = self.policy.stats()
        return {
            "mode": self.mode, "mode_label": MODES[self.mode],
            "store_budget": self.store_budget if self.mode != "p1" else None,
            "short_capacity": SHORT_CAPACITY, "context_budget": CONTEXT_BUDGET,
            "tau_fact": self.policy.tau_fact, "tau_low": self.policy.tau_low,
            "t1": [_item(i) for i in st.short],
            "t2": [_item(i) for i in st.working],
            "t3": [_item(i) for i in st.long],
            "stored_tokens": s["stored_tokens"],
            "counts": dict({k: s[k] for k in ("dropped", "evicted", "demoted",
                                               "merges", "consolidations")},
                           compressed=self.policy.routed["working"] + s["demoted"]),
            "messages": self.n,
        }

    # -- the write path, with every decision it made written to the log
    def add(self, speaker, text):
        text = " ".join(text.split())
        if not text:
            return []
        p, st = self.policy, self.policy.store
        s0, w0, l0 = self._held()
        live0 = {i.id for i in st.long if i.is_active}
        c0 = dict(p.routed, dropped=st.dropped, demoted=st.demoted,
                  merges=st.merges, reclaimed=st.reclaimed)

        turn = Turn(session=1, index=self.n, speaker=speaker, text=text)
        self.n += 1
        p.observe(turn)                      # exactly the experiments' code path

        s1, w1, l1 = self._held()
        new = next(i for i in st.short if i.id not in s0)
        ev = [{"type": "enter", "tier": "t1", "text": new.text,
               "msg": f"entered Tier 1 (short-term buffer, holds the last "
                      f"{SHORT_CAPACITY} messages)"}]

        out = [s0[k] for k in s0 if k not in s1]
        if out:
            e = out[0]
            u, lab = e.utility, e.type_label
            score = f"score u = {u:.2f} ({lab})"
            went = [i for i in list(w1.values()) + list(l1.values())
                    if e.id in i.source_ids]
            if any(i.tier == LONG_TERM for i in went) and self.mode == "p1":
                why = f"u ≥ τ_fact = {p.tau_fact}"
                ev.append({"type": "t3", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1 → Tier 3 as a durable fact. "
                                  f"{score}, {why}"})
                if any(i.tier == WORKING for i in went):
                    ev.append({"type": "t2", "tier": "t2", "text": e.text,
                               "msg": "very high score, so its gist also goes to Tier 2"})
            elif any(i.tier == LONG_TERM for i in went):
                ev.append({"type": "t3", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1 → kept verbatim in Tier 3. "
                                  f"{score}; the score sets its place in the "
                                  f"forgetting queue"})
            elif any(i.tier == WORKING for i in went):
                g = next(i for i in went if i.tier == WORKING)
                ev.append({"type": "t2", "tier": "t2", "text": g.text,
                           "msg": f"evicted from Tier 1 → compressed into Tier 2. "
                                  f"{score}, between τ_low = {p.tau_low} and "
                                  f"τ_fact = {p.tau_fact}"})
            elif p.routed["long"] > c0["long"] or p.routed["filled"] > c0["filled"]:
                ev.append({"type": "dup", "tier": "t3", "text": e.text,
                           "msg": f"evicted from Tier 1; near-duplicate of a fact "
                                  f"already in Tier 3, which is reinforced instead. "
                                  f"{score}"})
            else:
                ev.append({"type": "drop", "tier": None, "text": e.text,
                           "msg": f"evicted from Tier 1 → forgotten. {score}, "
                                  f"below τ_low = {p.tau_low}"})

        # what storage pressure, consolidation and supersession did meanwhile
        retired = {k for k, it in l0.items()
                   if k in live0 and it.superseded_by is not None}
        for k in retired:
            ev.append({"type": "supersede", "tier": "t3", "text": l0[k].text,
                       "msg": "superseded: a newer fact about the same thing "
                              "replaced it" + (", and its space was reclaimed"
                                               if k not in l1 else "")})
        for k, it in l0.items():
            if k not in l1 and k in live0 and k not in retired:
                demoted = [g for g in w1.values() if k in g.source_ids]
                if demoted:
                    ev.append({"type": "demote", "tier": "t2", "text": demoted[0].text,
                               "msg": f"store over budget: lowest-score fact "
                                      f"(u = {it.utility:.2f}) demoted from Tier 3 "
                                      f"to a shorter gist in Tier 2"})
                else:
                    ev.append({"type": "drop", "tier": None, "text": it.text,
                               "msg": f"store over budget (S = {self.store_budget}): "
                                      f"lowest-score fact forgotten, u = "
                                      f"{it.utility:.2f}"})
        for k, it in w1.items():
            if k not in w0 and it.type_label == "merged":
                ev.append({"type": "merge", "tier": "t2", "text": it.text,
                           "msg": f"Tier 2 full: {len(it.source_ids)} older gists "
                                  f"merged into one"})
        promoted = [i for k, i in l1.items() if k not in l0
                    and any(s in w0 for s in i.source_ids)]
        for it in promoted:
            ev.append({"type": "t3", "tier": "t3", "text": it.text,
                       "msg": "Tier 2 full: a high-score gist promoted to Tier 3"})
        for k, it in w0.items():
            if (k not in w1 and it.type_label != "merged"
                    and not any(k in g.source_ids for g in w1.values())
                    and not any(k in f.source_ids for f in l1.values())):
                ev.append({"type": "drop", "tier": None, "text": it.text,
                           "msg": "store over budget: gist dropped from Tier 2"})

        for e in ev:
            e["n"] = turn.index
            e["speaker"] = speaker
        self.log.extend(ev)
        return ev

    # -- the read path
    def context(self, query):
        ctx = self.policy.build_context(query, CONTEXT_BUDGET)
        return ctx, [_item(i) for i in ctx.items]


# ----------------------------------------------------------------- bench
class Bench:
    """run_live_check.py in a subprocess, its output kept for the page."""

    def __init__(self):
        self.lines, self.proc, self.code = [], None, None
        self.lock = threading.Lock()

    def start(self, store):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False
            self.lines, self.code = [], None
            self.proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "run_live_check.py"),
                 "--store", str(int(store))],
                cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line and not line.lstrip().startswith(("Warning", "warnings.")):
                self.lines.append(line)
        self.code = self.proc.wait()

    def state(self):
        running = bool(self.proc) and self.proc.poll() is None
        return {"running": running, "lines": self.lines[-40:], "code": self.code}


# ---------------------------------------------------------------- server
def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, app.full_state())
            if self.path == "/api/bench":
                return self._send(200, app.bench.state())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            b = self._body()
            try:
                if self.path == "/api/chat":
                    return self._send(200, app.chat(str(b.get("text", ""))))
                if self.path == "/api/sample":
                    return self._send(200, app.sample())
                if self.path == "/api/reset":
                    return self._send(200, app.reset(b.get("mode", "p1"),
                                                     b.get("store_budget", DEFAULT_STORE)))
                if self.path == "/api/bench":
                    app.bench.start(b.get("store", 4096))
                    return self._send(200, app.bench.state())
            except Exception as exc:          # show it on the page, keep serving
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            self._send(404, {"error": "not found"})
    return H


class App:
    def __init__(self, model):
        self.model = model
        self.session = Session()
        self.bench = Bench()

    def full_state(self):
        s = self.session.state()
        s.update(log=self.session.log[-80:], model=self.model.name,
                 backends=backend_info())
        return s

    def reset(self, mode, store_budget):
        self.session = Session(mode, store_budget)
        return self.full_state()

    def sample(self):
        with self.session.lock:
            for spk, txt in SAMPLE:
                self.session.add(spk, txt)
        return self.full_state()

    def chat(self, text):
        sess = self.session
        with sess.lock:
            events = sess.add("user", text)
            ctx, items = sess.context(text)
            reply = self.model.reply(ctx.text, text)
            events += sess.add("assistant", reply)
        out = self.full_state()
        out.update(reply=reply, context=items, context_tokens=ctx.tokens,
                   events=events)
        return out


def main():
    ap = argparse.ArgumentParser(description="MemGate live demo")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-llm", action="store_true",
                    help="run without the chat model (memory still works)")
    args = ap.parse_args()

    model = ChatModel(enabled=not args.no_llm)
    print("MemGate live demo")
    print("  warming the embedder ...", flush=True)
    from memgate.utils import embed
    embed("warm up")
    print(f"  embedder: {backend_info()['embedder']}", flush=True)
    if model.enabled:
        print(f"  loading {model.name} ...", flush=True)
        model.load()
    app = App(model)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"\n  open http://localhost:{args.port}   (Ctrl+C to stop)\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

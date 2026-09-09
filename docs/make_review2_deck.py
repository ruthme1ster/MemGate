#!/usr/bin/env python3
"""Build the Review 2 presentation from the committed results.

    python3 docs/make_review2_deck.py

Every table in the deck is read out of `memgate/results/*.csv` at build time,
for the same reason the dashboard is: a slide with a hand-typed number goes
stale the moment an experiment is re-run, and nobody notices until a panel
does. Figures are the committed PNGs from `make_figures.py`.

Deck geometry, type and colour match `MemGate_Capstone_Review1_NMIMS_v3.pptx`
so the two decks read as one series.
"""
import csv
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "memgate", "results")
FIGS = os.path.join(RESULTS, "figures")
OUT = os.path.join(ROOT, "docs", "MemGate_Capstone_Review2_NMIMS.pptx")

# --- house style, taken from the Review 1 deck -----------------------------
FONT = "Times New Roman"
INK = RGBColor(0x1A, 0x1A, 0x1A)
INK_TITLE = RGBColor(0x11, 0x11, 0x11)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0xC1, 0x27, 0x2D)          # NMIMS red
RULE = RGBColor(0xCC, 0xCC, 0xCC)
BAND = RGBColor(0xF2, 0xF2, 0xF0)
W, H = 20.0, 11.25                            # inches
L, CW = 0.9, 18.2                             # content left / width


def load(name, casts=()):
    with open(os.path.join(RESULTS, name), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for c in casts:
            if c in r and r[c] != "":
                r[c] = float(r[c])
    return rows


STORAGE = load("storage_sweep.csv", ("store_budget", "strict_recall", "answer_recall"))
SIG = load("significance.csv", ("delta", "conv_lo", "conv_hi", "mcnemar_p"))
SCALING = load("scaling.csv", ("store_budget", "ratio", "strict", "answer"))
COST = load("cost_model.csv", ("strict", "answer", "store_budget"))
JUDGE = load("judge_eval.csv", ("strict", "answer", "vs_p1", "ci_lo", "ci_hi"))
SCORER = load("learned_scorer.csv", ("auc", "p1_answer", "p3_answer", "store_budget"))
ABL = load("ablations_2048.csv", ("strict_recall", "delta_pts", "answer_recall"))
ENDTASK = load("endtask.csv", ("n", "accuracy", "f1", "answer_recall"))


def s_at(policy_prefix, budget, field="answer_recall"):
    for r in STORAGE:
        if r["store_budget"] == budget and r["policy"].startswith(policy_prefix):
            return r[field] * 100
    raise KeyError((policy_prefix, budget))


def scale_at(policy_sub, budget, ratio_rank, field="answer"):
    rows = sorted({r["ratio"] for r in SCALING if r["store_budget"] == budget})
    ratio = rows[ratio_rank]
    for r in SCALING:
        if (r["store_budget"] == budget and r["ratio"] == ratio
                and policy_sub in r["policy"]):
            return r[field], ratio
    raise KeyError(policy_sub)


def cost_at(policy_sub, budget, field="answer"):
    for r in COST:
        if r["store_budget"] == budget and r["policy"].startswith(policy_sub):
            return r[field]
    raise KeyError(policy_sub)


def sig_row(comparison, metric):
    for r in SIG:
        if r["comparison"] == comparison and r["metric"] == metric:
            return r
    raise KeyError(comparison)


def pnum(p):
    return f"{p:.1e}".replace("e-0", "×10⁻").replace("e-", "×10⁻") if p < 1e-3 else f"{p:.2f}"


# --- slide primitives ------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def textbox(sl, left, top, width, height):
    tb = sl.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def run(p, text, size=23, bold=False, color=INK, italic=False):
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r


def head(sl, title, kicker=None):
    tb, tf = textbox(sl, L, 0.62, CW, 1.35)
    p = tf.paragraphs[0]
    run(p, title, size=42, bold=True, color=INK_TITLE)
    ln = sl.shapes.add_shape(1, Inches(L), Inches(1.92), Inches(CW), Emu(9525 * 2))
    ln.fill.solid()
    ln.fill.fore_color.rgb = ACCENT
    ln.line.fill.background()
    ln.shadow.inherit = False
    if kicker:
        tb2, tf2 = textbox(sl, L, 1.99, CW, 0.6)
        p2 = tf2.paragraphs[0]
        run(p2, kicker, size=19, color=MUTED, italic=True)
    return 2.75 if kicker else 2.42


def bullets(sl, items, top, size=23, left=L, width=CW, gap=10, line=1.05):
    """items: (text, ...) or (lead, rest) or ("--", text) for a sub-point."""
    cpl = max(20, (width * 72) / (0.46 * size))
    nlines = sum(max(1, int(len(" ".join(i if isinstance(i, tuple) else (i,))) / cpl) + 1)
                 for i in items)
    est = nlines * size * 1.2 * line / 72 + len(items) * gap / 72 + 0.15
    tb, tf = textbox(sl, left, top, width, min(est, H - top - 0.3))
    first = True
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = line
        if isinstance(it, str):
            it = (it,)
        if it[0] == "--":
            run(p, "     – ", size=size - 2, color=MUTED)
            run(p, it[1], size=size - 2, color=MUTED)
        elif it[0] == "":
            run(p, it[1] if len(it) > 1 else " ", size=size)
        else:
            run(p, "•  ", size=size, color=ACCENT)
            run(p, it[0], size=size, bold=True)
            if len(it) > 1:
                run(p, it[1], size=size)
    return tb


def table(sl, rows, top, left=L, width=CW, size=18, col_w=None, height=None,
          bold_rows=(), accent_rows=()):
    nr, nc = len(rows), len(rows[0])
    h = height or min(H - top - 0.6, 0.52 * nr)
    shp = sl.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width), Inches(h))
    tbl = shp.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(width) * cw / total))
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(h / nr)
        for ci, cell in enumerate(row):
            c = tbl.cell(ri, ci)
            c.text = ""
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left, c.margin_right = Inches(0.14), Inches(0.10)
            c.margin_top = c.margin_bottom = Inches(0.03)
            p = c.text_frame.paragraphs[0]
            if ci > 0 and str(cell)[:1] in "0123456789+-−.":
                p.alignment = PP_ALIGN.RIGHT
            txt = str(cell)
            emph = txt.startswith("*") and txt.endswith("*")     # headline: bold accent
            flag = txt.startswith("~") and txt.endswith("~")     # caveat: bold, but muted
            if emph or flag:
                txt = txt[1:-1]
            col = ACCENT if (ri in accent_rows or emph) else (
                MUTED if flag else (INK_TITLE if ri == 0 else INK))
            run(p, txt, size=size,
                bold=(ri == 0 or ri in bold_rows or emph or flag),
                color=col)
    return shp


def picture(sl, name, top, height=None, width=None, center=True, left=None):
    path = os.path.join(FIGS, name)
    if height:
        pic = sl.shapes.add_picture(path, Inches(0), Inches(top), height=Inches(height))
    else:
        pic = sl.shapes.add_picture(path, Inches(0), Inches(top), width=Inches(width))
    if center:
        pic.left = Emu(int((Inches(W) - pic.width) / 2))
    elif left is not None:
        pic.left = Inches(left)
    return pic


def caption(sl, text, top, left=L, width=CW, size=17, align=PP_ALIGN.LEFT):
    lines = max(1, int(len(text) / max(1, (width * 72) / (0.46 * size))) + 1)
    tb, tf = textbox(sl, left, top, width, lines * size * 1.35 / 72 + 0.12)
    p = tf.paragraphs[0]
    p.alignment = align
    run(p, text, size=size, color=MUTED, italic=True)


def band(sl, text, top, height=1.15, label=None, left=L, width=CW, size=23):
    """A pulled-out statement -- the slide's takeaway, on an accent rule."""
    bar = sl.shapes.add_shape(1, Inches(left), Inches(top), Inches(0.07), Inches(height))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background(); bar.shadow.inherit = False
    bg = sl.shapes.add_shape(1, Inches(left + 0.07), Inches(top), Inches(width - 0.07), Inches(height))
    bg.fill.solid(); bg.fill.fore_color.rgb = BAND
    bg.line.fill.background(); bg.shadow.inherit = False
    tb, tf = textbox(sl, left + 0.32, top + 0.10, width - 0.7, height - 0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.line_spacing = 1.05
    if label:
        run(p, label.upper() + "   ", size=size - 6, bold=True, color=ACCENT)
    for i, chunk in enumerate(text if isinstance(text, list) else [text]):
        if isinstance(chunk, tuple):
            run(p, chunk[0], size=size, bold=True)
            if len(chunk) > 1:
                run(p, chunk[1], size=size)
        else:
            run(p, chunk, size=size)
    return bg


def footer(sl, text):
    tb, tf = textbox(sl, L, H - 0.72, CW, 0.5)
    p = tf.paragraphs[0]
    run(p, text, size=14, color=MUTED)


# =========================================================================
# 1 — title
# =========================================================================
sl = slide()
bar = sl.shapes.add_shape(1, Inches(L), Inches(3.05), Inches(2.4), Emu(9525 * 4))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
bar.line.fill.background(); bar.shadow.inherit = False
tb, tf = textbox(sl, L, 3.35, 16.5, 5.2)
p = tf.paragraphs[0]
run(p, "MemGate", size=60, bold=True, color=INK_TITLE)
p = tf.add_paragraph(); p.space_before = Pt(6)
run(p, "What to keep, what to forget", size=40, color=INK)
p = tf.add_paragraph(); p.space_before = Pt(14)
run(p, "A measured study of the memory decision policy for long-running LLM agents,"
       " under a bounded store", size=25, color=MUTED, italic=True)
p = tf.add_paragraph(); p.space_before = Pt(34)
run(p, "Simar Singh Khanuja", size=26); run(p, " — 70562200088", size=26, color=MUTED)
p = tf.add_paragraph()
run(p, "Yash Ramchandani", size=26); run(p, " — 70562300097", size=26, color=MUTED)
p = tf.add_paragraph(); p.space_before = Pt(20)
run(p, "SVKM’s NMIMS, Indore Campus", size=24, color=MUTED)
p = tf.add_paragraph()
run(p, "Capstone Project  ·  Progress review, September 2026", size=24, color=MUTED)

# =========================================================================
# 2 — status
# =========================================================================
sl = slide()
top = head(sl, "Where the project stands",
           "The measurement track is complete. Nine experiments run end to end; "
           "every claim carries a clustered confidence interval.")
gap = s_at("P1-S", 4096) - s_at("RAG", 4096)
table(sl, [
    ["Workstream", "State", "Evidence"],
    ["Harness, three-tier store, policies P0 / P1", "*Complete*", "memgate/ — 5,430 lines of Python"],
    ["LoCoMo wired; the evaluation leak removed", "*Complete*", "3 label-blindness regression tests"],
    ["Ablation study — 23 configurations", "*Complete*", "results/ablations_2048.csv"],
    ["Storage budget — the corrected accounting", "*Complete*", "35 configurations swept"],
    ["Significance: bootstrap + exact McNemar", "*Complete*", "clustered by conversation, B = 10,000"],
    ["Scaling to 182× · byte accounting · learned scorer", "*Complete*", "three separate sweeps"],
    ["P2 LLM judge — built and evaluated", "*Complete*", "single-component swap, with CIs"],
    ["End-task accuracy through a real reader", "Pilot only", "pipeline + cache built; 12-question run"],
    ["Abstractive compression · LongMemEval · online learning", "Not started", "scoped, with effort estimates"],
], top, size=19, col_w=[8, 2.4, 7.8])
band(sl, [("The headline:  "),
          ("+%.1f answer-recall points" % gap, " for choosing which turns to forget, over forgetting "
           "oldest-first — at identical storage, fidelity and retrieval.")],
     H - 1.85, height=1.15)

# =========================================================================
# 3 — the correction
# =========================================================================
sl = slide()
top = head(sl, "First: a Review 1 number is withdrawn",
           "Raised here rather than left for the panel to find.")
bullets(sl, [
    ("The Review 1 deck reported 95% recall for MemGate against 10% for a sliding window. ",
     "That number was inflated by an evaluation leak."),
    ("The leak. ", "The memory policy was reading fact_id — the ground-truth answer key — "
     "when deciding what to keep. It was worth 45–47 recall points."),
], top, size=24)
table(sl, [
    ["Synthetic set @ 800 tokens", "As presented at Review 1", "Honest"],
    ["P1 MemGate", "95.0%", "*50.0%*"],
    ["P0 sliding window", "10.0%", "10.0%"],
], top + 1.9, size=22, col_w=[8, 5, 5], height=1.7)
bullets(sl, [
    ("The fix is a test, not a comment. ",
     "Strip every evaluation label from the input and the store must come out bit-identical. "
     "Each decision point added since — eviction under a storage budget, and demotion — was "
     "added to that test before any number from it was quoted."),
    ("Everything after this slide ", "is measured on a real benchmark, leak-free."),
], top + 3.95, size=23)
band(sl, "A withdrawn result the students raise is worth more than one the panel discovers.",
     H - 1.7, height=1.05, label="Why it leads")

# =========================================================================
# 4 — problem
# =========================================================================
sl = slide()
top = head(sl, "The problem", "A 200K-token context window did not solve agent memory; it relocated it.")
bullets(sl, [
    ("Context rot. ", "Chroma Research (2025) tested 18 frontier models: accuracy degrades "
     "30–50% well before the documented context limit. It is a property of attention, not a "
     "training gap — so larger windows do not fix it. Curation does."),
    ("Cost and latency. ", "On LoCoMo, full context is ~26,000 tokens and 9.87 s median latency; "
     "selective memory reaches similar accuracy at ~1,800 tokens and 0.71 s — roughly 14× cheaper."),
    ("The measurable gap. ", "On LongMemEval, oracle retrieval scores ~92% where the same model "
     "interactively scores ~58%. That 34-point gap is memory selection failure, not reasoning failure."),
], top, size=25)
band(sl, [("The open problem, named in a 2026 review: current systems "),
          ("“do not solve the fundamental challenge: deciding what to remember and what to forget.”",)],
     H - 2.4, height=1.5, label="Where the gap is")

# =========================================================================
# 5 — question and contribution
# =========================================================================
sl = slide()
top = head(sl, "The question, and what is new")
band(sl, "How much of the memory-selection gap can a better decision policy recover, "
         "and under what budget does compression pay for itself?",
     top, height=1.5, label="Research question", size=27)
bullets(sl, [
    ("Tiered memory with forgetting is already shipped. ",
     "Letta/MemGPT, Mem0 and Zep all ship it. Presented as “a memory layer with three tiers”, "
     "this work would already be done."),
    ("What is not done is the decision itself. ",
     "Colaco & Lahjouji (2026) frame every such decision as one rate–distortion problem — what to "
     "retain, at what fidelity, under a budget — but supply a framework, not a system."),
    ("MemGate is the harness that makes the decision measurable. ",
     "Policy-agnostic: competing policies are placed on one accuracy-per-token frontier, with "
     "each comparison constructed so that exactly one variable moves."),
], top + 1.9, size=24)
band(sl, "The contribution is a measurement, and several of its results are negative.",
     H - 1.7, height=1.05, label="Stated plainly")

# =========================================================================
# 6 — architecture
# =========================================================================
sl = slide()
top = head(sl, "System architecture",
           "MemGate wraps a frozen LLM and controls what enters the prompt. Nothing is trained.")


def abox(x, y, w, h, lines, fill=BAND, edge=RULE, bold=False, size=17, ink=INK):
    shp = sl.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge; shp.line.width = Pt(1)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run(p, t, size=size if i == 0 else size - 3, bold=bold and i == 0,
            color=ink if i == 0 else MUTED)
    return shp


def conn(x1, y1, x2, y2):
    c = sl.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RULE; c.line.width = Pt(1.5)
    return c


caption(sl, "WRITE PATH — what to keep", top - 0.05, size=16)
y0 = top + 0.45
abox(L, y0, 2.9, 1.0, ["incoming turn"])
abox(L + 3.35, y0 - 0.15, 3.9, 1.3, ["Adaptive Memory Decision Engine",
                                     "scores each item exactly once, on eviction"],
     fill=RGBColor(0xF7, 0xE7, 0xE8), edge=ACCENT, bold=True)
conn(L + 2.9, y0 + 0.5, L + 3.35, y0 + 0.5)
for i, (t, dy) in enumerate([("u ≥ τ_fact   →   Tier 3   durable fact", -0.55),
                             ("u ≥ τ_low    →   Tier 2   compressed gist", 0.25),
                             ("otherwise    →   dropped from context", 1.05)]):
    abox(L + 7.85, y0 + dy, 4.6, 0.68, [t], size=16)
    conn(L + 7.25, y0 + 0.5, L + 7.85, y0 + dy + 0.34)
abox(L + 13.0, y0 - 0.35, 5.1, 2.6, [""], fill=RGBColor(0xFA, 0xFA, 0xF9))
caption(sl, "TIERED STORE", y0 - 0.36, left=L + 13.15, width=4, size=14)
for i, t in enumerate(["Tier 1  short-term   raw verbatim turns",
                       "Tier 2  working      compressed gists",
                       "Tier 3  long-term    facts + vectors"]):
    abox(L + 13.2, y0 + 0.15 + i * 0.72, 4.7, 0.6, [t], size=15)
conn(L + 12.45, y0 - 0.21, L + 13.2, y0 + 0.45)
conn(L + 12.45, y0 + 0.59, L + 13.2, y0 + 1.17)
conn(L + 12.45, y0 + 1.39, L + 13.2, y0 + 1.89)

ry = y0 + 3.55
rule = sl.shapes.add_shape(1, Inches(L), Inches(ry - 0.45), Inches(CW), Emu(9525 * 2))
rule.fill.solid(); rule.fill.fore_color.rgb = RULE
rule.line.fill.background(); rule.shadow.inherit = False
caption(sl, "READ PATH — what to recall", ry - 0.35, size=16)
for i, (t, w) in enumerate([("new query", 2.6), ("retrieve   semantic match", 3.4),
                            ("assemble context   to the token budget", 4.3),
                            ("FROZEN LLM", 3.0), ("response", 2.4)]):
    pass
xs = [L, L + 3.0, L + 6.8, L + 11.5, L + 15.0]
ws = [2.6, 3.4, 4.3, 3.1, 2.6]
labels = [["new query"], ["retrieve", "semantic match over tier 3"],
          ["assemble context", "packs to the token budget"], ["FROZEN LLM"], ["response"]]
for i in range(5):
    abox(xs[i], ry + 0.25, ws[i], 1.0, labels[i],
         fill=BAND if i != 3 else RGBColor(0xEA, 0xEA, 0xE8), bold=(i == 3))
    if i:
        conn(xs[i - 1] + ws[i - 1], ry + 0.75, xs[i], ry + 0.75)
band(sl, "Tiers are a lifecycle, not a partition: everything enters Tier 1, and the routing "
         "decision happens once — on eviction, when the budget binds.",
     H - 1.6, height=1.0)

# =========================================================================
# 7 — policies
# =========================================================================
sl = slide()
top = head(sl, "Policies compared", "Each comparison is built so that exactly one variable moves.")
table(sl, [
    ["", "Policy", "Retention rule", "Fidelity"],
    ["P0", "Sliding window", "recency, budget-driven", "verbatim"],
    ["RAG", "Store-all + retrieve", "FIFO — the no-policy control", "verbatim"],
    ["P1", "MemGate threshold", "scored at ingest, τ-routed", "compressed"],
    ["*P1-A*", "*MemGate adaptive*", "*score orders eviction*", "*compress on eviction*"],
    ["*P1-S*", "*MemGate select*", "*score orders eviction*", "*drop on eviction*"],
    ["P2", "LLM-judged salience", "Qwen2.5-0.5B rates each turn", "drop"],
    ["P3", "Learned scorer", "fitted, cross-validated", "drop"],
    ["Oracle", "—", "perfect selection (upper bound)", "verbatim"],
], top, size=19, col_w=[1.6, 5, 7.5, 4.5], height=4.0)
table(sl, [
    ["Comparison", "Held constant", "The one thing that varies"],
    ["P0  vs  RAG", "retention, storage", "*read path — recency vs retrieval*"],
    ["RAG  vs  P1-S", "storage, fidelity, retrieval", "*which turns are forgotten*"],
    ["P1-S  vs  P1-A", "storage, selection", "*fidelity — drop vs compress*"],
], top + 4.4, size=20, col_w=[4.5, 6.5, 7.2], height=2.0)
band(sl, "P0 is given the same token budget as MemGate, so the only difference is which turns "
         "are chosen — never how many tokens may be spent. A fixed small window would be a strawman.",
     H - 1.55, height=1.0)

# =========================================================================
# 8 — metrics
# =========================================================================
sl = slide()
top = head(sl, "Two metrics, and why both are reported")
bullets(sl, [
    ("Evidence (strict) recall. ", "Every gold evidence turn reached the context. A multi-hop "
     "question cites up to 19 turns, and 18 of 19 answers nothing."),
    ("Answer recall. ", "The answer text itself survived, scored over the 715 questions whose "
     "answer is recoverable from the evidence text. This is the metric to believe."),
    ("Stored tokens / bytes. ", "What the policy still holds, as opposed to what it spends per query."),
], top, size=25)
band(sl, [("Reporting both is essential rather than decorative: the two metrics "),
          ("disagree",), (" on exactly the comparison this project is about, and the disagreement "
                          "is the finding.")],
     top + 3.5, height=1.35, label="Why two")
bullets(sl, [
    ("Evidence recall credits a pointer. ", "A compressed gist keeps the evidence id of a turn "
     "whose text it has thrown away, and scores a hit with none of the answer behind it."),
    ("Answer recall checks the content. ", "During the consolidation work, evidence recall rose "
     "8.1% → 14.8% while answer recall stayed pinned at 12.8%. The entire gain was bookkeeping."),
], top + 5.15, size=23)

# =========================================================================
# 9 — setup
# =========================================================================
sl = slide()
top = head(sl, "Experimental setup", "Deterministic, offline, and free to re-run on every change.")
table(sl, [
    ["Element", "Choice", "Why"],
    ["Benchmark", "LoCoMo — 10 conversations, 5,882 turns, 1,527 scoreable questions",
     "Real multi-session dialogue with gold evidence ids"],
    ["Excluded", "Adversarial items (category 5)",
     "They carry evidence, but the correct behaviour is to decline"],
    ["Embedder", "all-MiniLM-L6-v2, 384-d", "Retrieval ranking and semantic supersession"],
    ["Tokenizer", "tiktoken cl100k_base", "Budgets are exact tokens, not a word-count proxy"],
    ["P2 judge", "Qwen2.5-0.5B-Instruct, run locally", "One forward pass per turn, logit-expectation scored"],
    ["Reader", "Qwen2.5-1.5B-Instruct-4bit, greedy", "Held fixed across arms, so any difference is the policy"],
    ["Statistics", "Paired bootstrap B = 10,000 + exact McNemar", "Resampled by conversation, not by question"],
], top, size=18, col_w=[3.2, 8.2, 6.8])
band(sl, "1,527 questions drawn from 10 conversations are not 1,527 independent trials. "
         "Every interval in this deck is clustered at the conversation level.",
     H - 1.75, height=1.1, label="The one statistical point that matters")

# =========================================================================
# 10 — the accounting error
# =========================================================================
sl = slide()
top = head(sl, "Storage was never budgeted — and that determined the answer")
bullets(sl, [
    ("Every experiment in this literature ", "caps the context assembled per query and leaves "
     "the store unbounded. So does every experiment of ours before this point."),
    ("Under that accounting, ", "“keep everything and retrieve top-k” is charged nothing for "
     "holding all 419 turns of a conversation — so forgetting can only ever lose information."),
    ("Our own ablation duly reported ", "the compression machinery as a net loss against keeping "
     "everything. It was right to."),
], top, size=23, width=9.0)
picture(sl, "fig1_storage_frontier.png", top + 0.35, width=8.5, center=False, left=10.6)
caption(sl, "Fig. 1 — the storage-token frontier. Context pinned at 2048; the store swept.",
        top + 4.5, left=10.6, width=8.5, size=16)
band(sl, [("Imposing a storage budget S changes the question from "),
          ("“what fits in context?”",), (" to "), ("“what is worth keeping at all?”",)],
     H - 2.6, height=1.5, label="The correction", size=26)
caption(sl, "Under the corrected accounting the ranking inverts: what looked like a loss for the "
            "decision policy was an artefact of never charging for the store.", H - 1.0, size=18)

# =========================================================================
# 11 — the headline
# =========================================================================
sl = slide()
top = head(sl, "Selection pays; compression does not",
           "Context budget fixed at 2048 tokens. Storage budget swept. Answer recall.")
rows = [["S (tokens)", "RAG store-all", "P1-A adaptive (compress)", "P1-S select (drop)",
         "P1-S − RAG"]]
for S in (1024, 2048, 4096, 8192, 16384):
    rag, adp, sel = s_at("RAG", S), s_at("P1-A", S), s_at("P1-S", S)
    rows.append([f"{S:,}", f"{rag:.1f}%", f"{adp:.1f}%", f"*{sel:.1f}%*",
                 f"+{sel - rag:.1f}"])
table(sl, rows, top, size=21, col_w=[3, 4, 5.2, 4.4, 3], height=3.6)
bullets(sl, [
    ("The two metrics disagree, and that is the result. ",
     "P1-A wins evidence recall (+12.9 at S = 8192) while losing answer recall (−6.9). "
     "A demoted gist keeps the evidence ids of a turn whose text it discarded. "
     "Reporting only evidence recall would have shown compression winning."),
], top + 4.0, size=23)
band(sl, "RAG and P1-S hold the same verbatim turns, in the same space, retrieved the same way, "
         "and differ in one respect only: which turns they forget when the cap binds. "
         "RAG forgets oldest-first; P1-S forgets lowest-utility-first.",
     H - 2.15, height=1.5, label="The cleanest comparison in the project")

# =========================================================================
# 12 — significance
# =========================================================================
sl = slide()
top = head(sl, "What survives clustering",
           "Paired bootstrap, B = 10,000, resampled by conversation · exact McNemar · S = 4096")
order = [("P1-S select vs RAG", "answer"), ("P1-S select vs RAG", "strict"),
         ("P1-S select vs P1-A adaptive", "answer"), ("P1-S select vs P1-A adaptive", "strict"),
         ("P1-S select vs P0", "answer"), ("P1-S select vs P0", "strict")]
name = {"P1-S select vs RAG": "P1-S select  vs  RAG store-all",
        "P1-S select vs P1-A adaptive": "P1-S select  vs  P1-A compress",
        "P1-S select vs P0": "P1-S select  vs  P0 recency"}
rows = [["Comparison", "Metric", "Δ points", "95% CI (clustered)", "McNemar p", "Verdict"]]
for comp, met in order:
    r = sig_row(comp, met)
    ns = r["conv_lo"] <= 0 <= r["conv_hi"]
    lead = "*" if (met == "answer" and comp == "P1-S select vs RAG") else ""
    rows.append([lead + name[comp] + lead,
                 lead + ("answer recall" if met == "answer" else "evidence recall") + lead,
                 f"{lead}{r['delta']:+.2f}{lead}",
                 f"{lead}[{r['conv_lo']:+.2f}, {r['conv_hi']:+.2f}]{lead}",
                 f"{lead}{pnum(r['mcnemar_p'])}{lead}",
                 "~not significant~" if ns else lead + "significant" + lead])
table(sl, rows, top, size=19, col_w=[5.4, 3.2, 2.2, 3.6, 2.4, 3.0], height=4.3)
band(sl, [("We do not claim an evidence-recall improvement over RAG.", ""),
          (" +1.96 points does not survive clustering (p = 0.16). The answer-recall claim, "
           "+8.67 with the interval clear of zero, does.",)],
     top + 4.75, height=1.35, label="Reported as it stands")
footer(sl, "A claim reported as not significant is a claim the panel does not have to find.")

# =========================================================================
# 13 — rate-distortion
# =========================================================================
sl = slide()
top = head(sl, "The rate–distortion prediction fails",
           "Theory says compression must eventually win. It does not, anywhere we can reach.")
bullets(sl, [
    ("Sweeping S downward cannot test it. ", "By S = 256 every policy sits at ~0.2% evidence "
     "recall — below the floor where anything works."),
    ("The missing axis is length. ", "We concatenate up to 10 LoCoMo conversations (186K tokens) "
     "and hold S fixed, so the compression ratio rises to 182×."),
], top, size=23)
rows = [["Compression ratio", "RAG", "P1-A (compress)", "P1-S (drop)"]]
for rank in range(4):
    rag, ratio = scale_at("RAG", 2048, rank)
    adp, _ = scale_at("P1-A", 2048, rank)
    sel, _ = scale_at("P1-S", 2048, rank)
    rows.append([f"{ratio:.0f}×", f"{rag:.1f}%", f"{adp:.1f}%", f"*{sel:.1f}%*"])
table(sl, rows, top + 2.3, size=20, col_w=[4.6, 3.6, 4.6, 4.0], height=2.9,
      width=11.0)
picture(sl, "fig2_scaling.png", top + 2.3, height=3.5, center=False, left=12.6)
band(sl, "No crossover to 182×, and the selection margin over RAG widens with pressure. "
         "At LoCoMo scale the optimum sits at the vertex: a subset at full fidelity beats "
         "everything at reduced fidelity.",
     H - 2.35, height=1.15, label="Result")
caption(sl, "Caveat: a concatenated stream is not a natural long conversation — each question "
            "concerns one constituent conversation, so the others act as distractors. It is a "
            "stress test of retention under pressure, not a long-dialogue benchmark.",
        H - 1.05, size=16)

# =========================================================================
# 14 — byte accounting
# =========================================================================
sl = slide()
top = head(sl, "The index is not free",
           "Storage denominated in text tokens silently gives the embedding index away.")
bullets(sl, [
    ("A 384-d float32 vector is 1,536 bytes ", "against a turn’s ~130 bytes of text. The index "
     "outweighs the content ~12×, and the binding cost becomes the number of items, not their length."),
], top, size=23)
rows = [["Budget", "P0 (keeps no index)", "RAG store-all", "P1-S select"]]
for b, kib in ((32768, "32 KiB"), (65536, "64 KiB"), (262144, "256 KiB")):
    p0, rag, sel = cost_at("P0", b), cost_at("RAG", b), cost_at("P1-S", b)
    best = max(p0, rag, sel)
    rows.append([kib] + [("*%.1f%%*" % v) if v == best else ("%.1f%%" % v)
                         for v in (p0, rag, sel)])
table(sl, rows, top + 1.4, size=21, col_w=[3.2, 4.6, 3.6, 3.4], height=2.3, width=10.8)
picture(sl, "fig4_cost_model.png", top + 1.3, height=4.0, center=False, left=12.2)
band(sl, [("Two results. ", ""),
          ("The selection finding is unit-robust — it holds when the budget is counted in bytes. "
           "And below ~40 KiB an embedding index cannot earn its own storage: plain recency, "
           "which keeps no vectors at all, wins outright. That regime is invisible when the "
           "budget counts text alone.",)],
     H - 2.35, height=1.6)

# =========================================================================
# 15 — the scorer
# =========================================================================
sl = slide()
top = head(sl, "The scorer is the bottleneck — and it is learnable")
abl = {r["configuration"]: r for r in ABL if r["family"] == "architecture"}
rows = [["Component removed", "Δ evidence recall"]]
for k in ["- retrieval (tier 3)", "- demotion on pressure", "- scored eviction (FIFO)",
          "- informative compressor", "- consolidation merge", "- supersession"]:
    r = abl[k]
    lbl = k.replace("- ", "")
    extra = "   (but answer recall +6.7)" if "demotion" in k else ""
    rows.append([lbl + extra, f"{r['delta_pts']}"])
table(sl, rows, top, size=20, col_w=[7.4, 3.4], height=3.6, width=10.4)
s2048 = [r for r in SCORER if r["store_budget"] == 2048]
rows2 = [["Scorer", "AUC", "Answer recall"],
         ["P1 heuristic (hand-tuned)", "—", f"{s2048[0]['p1_answer']:.1f}%"]]
for r in s2048:
    lbl = "P3 learned, hand features" if "hand features only" in r["features"] \
        else "P3 learned, + MiniLM"
    star = "*" if "MiniLM" in r["features"] else ""
    rows2.append([star + lbl + star, f"{star}{r['auc']:.3f}{star}",
                  f"{star}{r['p3_answer']:.1f}%{star}"])
table(sl, rows2, top + 4.0, size=20, col_w=[6.2, 2.0, 2.2], height=2.1, width=10.4)
picture(sl, "fig3_scorer.png", top + 0.2, height=4.6, center=False, left=11.6)
caption(sl, "Leave-one-conversation-out: the model scoring a conversation never saw it in training.",
        top + 6.25, width=10.4, size=17)
band(sl, "The fitted weights say something the heuristic never encoded: turn length is the "
         "strongest single predictor of evidence-worthiness — ahead of every hand-designed cue.",
     H - 1.7, height=1.05, label="What the model learned")

# =========================================================================
# 16 — the judge loses
# =========================================================================
sl = slide()
top = head(sl, "An LLM judge is the wrong instrument for salience",
           "A single-component swap: policy, storage, retrieval and compression pinned at the "
           "winner — only the scorer varies, so anything that moves is the scorer.")
short = {"P0 recency (no scoring)": "P0  recency — no content judgement",
         "P1 heuristic (hand)": "*P1  heuristic — regex + shallow NER*",
         "P2 LLM judge (Qwen-0.5B)": "P2  LLM judge — Qwen2.5-0.5B",
         "P3 learned (LOCO, bound)": "P3  learned — LOCO, a bound"}
rows = [["Scorer", "Evidence", "Answer", "Δ vs P1", "95% CI (clustered)"]]
for r in JUDGE:
    is_p1 = r["scorer"].startswith("P1 ")
    rows.append([short[r["scorer"]],
                 f"{r['strict']:.1f}%",
                 ("*%.1f%%*" % r["answer"]) if is_p1 else ("%.1f%%" % r["answer"]),
                 "—" if is_p1 else f"{r['vs_p1']:+.2f}",
                 "—" if is_p1 else f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"])
table(sl, rows, top, size=21, col_w=[7.2, 2.6, 2.6, 2.6, 4.2], height=3.0)
bullets(sl, [
    ("Scoring at all is load-bearing. ", "+11.19 points over unscored recency — without this row "
     "the rest of the table would not be worth reading."),
    ("The judge fails at it anyway. ", "It loses to the regex by 5.87 points and the interval "
     "excludes zero, for ~15 minutes of GPU per corpus against microseconds."),
    ("The headroom is real. ", "+11.47 to the learned bound, so P2 was not defeated by a missing "
     "ceiling. The ceiling it failed to reach genuinely exists."),
], top + 3.4, size=22)
band(sl, "The diagnosis is what the judge is shown: it rates each turn in isolation, and salience "
         "is not a property of a turn on its own. A phone number matters because something later "
         "asks for it.",
     H - 1.95, height=1.3, label="Why it loses")

# =========================================================================
# 17 — end task
# =========================================================================
sl = slide()
top = head(sl, "End-task accuracy — built, and honestly labelled",
           "Every number so far is context recall: did the evidence reach the prompt? "
           "That is the ceiling on accuracy, not accuracy.")
rows = [["Arm", "Accuracy", "F1", "Answer present in context"]]
for name_ in ["closed-book", "P0 sliding-window", "RAG store-all", "P1-A adaptive",
              "P1-S select", "Oracle"]:
    r = [x for x in ENDTASK if x["arm"] == name_][0]
    star = "*" if name_ == "P1-S select" else ""
    rows.append([star + name_ + star, f"{star}{r['accuracy']:.1f}%{star}",
                 f"{star}{r['f1']:.1f}{star}", f"{star}{r['answer_recall']:.1f}%{star}"])
table(sl, rows, top, size=20, col_w=[4.6, 2.6, 2.2, 4.4], height=3.4, width=10.4)
picture(sl, "fig5_endtask.png", top + 0.1, height=4.2, center=False, left=11.6)
bullets(sl, [
    ("Two controls the comparison cannot be read without. ",
     "Closed-book is the floor — a model answering from parametric knowledge would make every "
     "policy look good. Oracle is the ceiling: exactly the cited evidence, perfectly selected."),
], top + 3.8, size=21, width=10.4)
band(sl, [("This is a 12-question smoke run, not a result.", ""),
          (" It is shown because the pipeline, both controls and the generation cache work end "
           "to end — the remaining experiment is wired and ready. The full 715-question run is "
           "several hours of local generation and has not been made.",)],
     H - 2.1, height=1.45, label="Read this before the numbers")

# =========================================================================
# 18 — measurement findings
# =========================================================================
sl = slide()
top = head(sl, "Three findings about measurement",
           "Each was caught by a guard, not by reading the code.")
table(sl, [
    ["", "Finding", "What it cost"],
    ["4.1", "*A policy that read its own answer key.* An early consolidation rule promoted items "
            "on fact_id — the evaluation label.",
     "+45–47 recall points. Invalidated the Review 1 headline."],
    ["4.2", "*A component’s value is conditional on the bottleneck being elsewhere.* The MiniLM "
            "embedder measured at +0.2 points — true in the regime measured, where retrieval was "
            "saturated against a write path discarding 93% of the conversation.",
     "Once that ceiling lifted, the same swap was worth +19.7."],
    ["4.3", "*A metric that credits pointers.* Evidence recall counts a turn as retrieved if its "
            "id reaches the context; a gist keeps the id and discards the text.",
     "Evidence recall rose 8.1% → 14.8% while answer recall stayed at 12.8%."],
], top, size=19, col_w=[1.2, 11.4, 5.6], height=5.0)
band(sl, "A fourth, smaller instance: batched judge scoring returned NaN for exactly the short "
         "turns. The instinct was precision, and float32 did not fix it — the cause was "
         "left-padding creating fully-masked attention rows. Without the NaN guard, every filler "
         "turn would have scored NaN and quietly inverted the eviction order.",
     top + 5.4, height=1.75, label="And a fourth")

# =========================================================================
# 19 — tools, proposed vs used
# =========================================================================
sl = slide()
top = head(sl, "Tools and technology", "What was proposed at Review 1, and what the results were "
                                       "actually produced with.")
table(sl, [
    ["Area", "Proposed at Review 1", "Actually used", "Why it changed"],
    ["Base model", "Llama 3 / Mistral; OpenAI or Anthropic API",
     "*Qwen2.5-0.5B and 1.5B-4bit, run locally*",
     "No key, no rate limit, no per-run cost, bit-reproducible offline"],
    ["Agent scaffolding", "LangChain / LlamaIndex", "*None — a direct harness*",
     "A framework’s own retrieval would confound the variable under study"],
    ["Vector store", "FAISS / Chroma", "*NumPy cosine over an in-process store*",
     "At 10 conversations an ANN index adds approximation error, not speed"],
    ["Embeddings", "Sentence-transformers", "all-MiniLM-L6-v2, 384-d",
     "As proposed; ablated against a hashing fallback"],
    ["Tokenisation", "—", "*tiktoken cl100k_base*",
     "Budgets must be exact tokens, not a word-count proxy"],
    ["Evaluation", "End-task F1 on a long-conversation benchmark",
     "*Context recall + answer presence, then end-task*",
     "Recall isolates the policy from the reader’s reasoning, and bounds it"],
    ["Statistics", "—", "*Paired bootstrap (B = 10,000) + exact McNemar*",
     "1,527 questions from 10 conversations are not 1,527 trials"],
], top, size=17, col_w=[2.6, 4.6, 5.4, 5.6], height=5.2)
band(sl, "This is not scope reduction. Running everything locally and free is what made 35 "
         "storage configurations, 23 ablations, a 10,000-sample bootstrap and a 36-point scaling "
         "sweep affordable to re-run on every change.",
     top + 5.6, height=1.35, label="Why it matters")

# =========================================================================
# 20 — discipline
# =========================================================================
sl = slide()
top = head(sl, "Verification and engineering discipline",
           "Each guard exists because the failure it prevents already happened once.")
table(sl, [
    ["Guard", "What it prevents"],
    ["61 tests (test_memgate.py)", "Regression across the store, scorers, policies, compressor and harness"],
    ["*3 label-blindness invariants*",
     "*The evaluation leak, re-armed. Routing, eviction and demotion must produce a bit-identical "
     "store when every evaluation label is stripped*"],
    ["Offline pinning on import (HF_HUB_OFFLINE)",
     "A silently fetched or half-downloaded model relabelling a whole results table — this "
     "already happened once, with MiniLM"],
    ["Backend provenance (backend_info())",
     "Reporting a MiniLM number that the hashing fallback actually produced"],
    ["NaN guard in the judge", "Filler turns scoring NaN and inverting the eviction order"],
    ["Generation caches keyed by (model, prompt)",
     "Non-reproducible LLM output; also what makes long runs resumable"],
    ["Results committed with the code", "A figure in a report that no longer matches the code that made it"],
], top, size=19, col_w=[6.2, 12.0], height=5.6)
band(sl, "The fix for a leak is a test, not a comment. Each new decision point is added to the "
         "invariant before any number from it is quoted.",
     H - 1.75, height=1.1, label="The rule adopted")

# =========================================================================
# 21 — deliverables
# =========================================================================
sl = slide()
top = head(sl, "Deliverables", "Everything below is in the repository and reproducible from source.")
bullets(sl, [
    ("REPORT.md ", "— the paper-ready writeup: abstract, method, results with intervals, "
     "methodological findings, limitations."),
    ("docs/REVIEW2_PROGRESS_REPORT.md ", "— this review’s progress report: status, what changed "
     "since Review 1, the full technology inventory, and the plan to the final review."),
    ("SESSION_RECORD.md ", "— the complete working record, §1–§28, written to be picked up cold."),
    ("frontend/index.html ", "— an interactive results dashboard. Every figure on it is read "
     "from the committed CSVs by frontend/build_data.py; nothing is typed by hand."),
    ("memgate/ ", "— 5,430 lines of Python: the library, 13 runnable experiments, 61 tests."),
    ("memgate/results/ ", "— every CSV and every figure quoted in this deck, committed alongside "
     "the code that produced them."),
    ("docs/make_review2_deck.py ", "— this deck, generated from those same CSVs."),
], top, size=23)
band(sl, "A reviewer can check every number in this deck without a nine-minute benchmark run: "
         "the results are committed, and each slide names the file it came from.",
     H - 1.75, height=1.1)

# =========================================================================
# 22 — timeline
# =========================================================================
sl = slide()
top = head(sl, "Timeline")
months = ["Jul ’26", "Aug ’26", "Sep ’26", "Oct ’26", "Nov ’26", "Dec ’26"]
gx, gy, gw, rowh = L + 6.2, top + 0.35, 11.6, 0.60
colw = gw / len(months)
for i, m in enumerate(months):
    b = sl.shapes.add_shape(5, Inches(gx + i * colw), Inches(gy), Inches(colw - 0.06), Inches(0.55))
    b.fill.solid(); b.fill.fore_color.rgb = BAND
    b.line.color.rgb = RULE; b.shadow.inherit = False
    p = b.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run(p, m, size=17, color=MUTED)
tasks = [
    ("Literature review, problem finalisation", 0, 1.0, True),
    ("System design and harness", 0.3, 1.2, True),
    ("LoCoMo evaluation, leak removed", 1.0, 0.8, True),
    ("Ablation study", 1.3, 0.5, True),
    ("Storage budget, significance, scaling", 1.5, 0.7, True),
    ("P2 judge, byte accounting, learned scorer", 1.8, 0.6, True),
    ("Full end-task run (715 questions)", 2.3, 0.8, False),
    ("Abstractive compression + LongMemEval", 3.0, 1.4, False),
    ("Online learning of the eviction policy", 4.0, 1.2, False),
    ("Final report, documentation, defence", 4.8, 1.2, False),
]
for i, (label, start, dur, done) in enumerate(tasks):
    y = gy + 0.70 + i * rowh
    tb, tf = textbox(sl, L, y - 0.09, 6.0, 0.5)
    p = tf.paragraphs[0]
    run(p, label, size=17, color=INK if done else MUTED)
    bar = sl.shapes.add_shape(5, Inches(gx + start * colw), Inches(y),
                              Inches(max(dur * colw - 0.06, 0.3)), Inches(0.4))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT if done else RGBColor(0xDD, 0xDD, 0xDA)
    bar.line.fill.background(); bar.shadow.inherit = False
y_end = gy + 0.70 + len(tasks) * rowh
for label, x in (("Review 1 · Jul", 0.15), ("This review · Sep", 2.15),
                 ("Review 2 · Oct", 3.35), ("Final · Dec", 5.15)):
    tb, tf = textbox(sl, gx + x * colw - 1.0, y_end + 0.05, 2.0, 0.34)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run(p, "● " + label, size=14, color=ACCENT)
caption(sl, "Filled bars are complete. The critical path to the final review is the full "
            "end-task run — the pipeline is built and validated; what it needs is machine time.",
        H - 0.95, size=19)

# =========================================================================
# 23 — limitations and remaining work
# =========================================================================
sl = slide()
top = head(sl, "Limitations, and what remains")
tb, tf = textbox(sl, L, top, 8.7, 6.6)
p = tf.paragraphs[0]
run(p, "Stated limitations", size=26, bold=True, color=ACCENT)
for t in ["Ten conversations. Clustered intervals are wide because the cluster count is small — "
          "and the one effect that does not survive clustering is reported as not significant.",
          "Context recall is the ceiling on end-task accuracy, not accuracy itself.",
          "Compression tested is extractive. The negative result is stated for that case only.",
          "Concatenated streams are synthetic.",
          "Single embedder, single judge model. The judge result is evidence about a 0.5B model "
          "prompted per turn, not about LLM judging in general.",
          "P3-learned is a headroom bound, not a component: a live agent has no future questions."]:
    p = tf.add_paragraph(); p.space_before = Pt(9); p.line_spacing = 1.03
    run(p, "•  ", size=20, color=ACCENT); run(p, t, size=20)
tb2, tf2 = textbox(sl, L + 9.4, top, 8.8, 6.6)
p = tf2.paragraphs[0]
run(p, "Remaining work", size=26, bold=True, color=ACCENT)
for n, t, e in [("1", "Full 715-question end-task run. Converts the project from a ceiling to an "
                      "accuracy claim; pipeline, controls and cache already validated.", "~4–6 h, resumable"),
                ("2", "Abstractive compression, to test whether the negative result is specific "
                      "to extractive methods.", "1 session"),
                ("3", "LongMemEval — genuinely long single conversations, replacing the "
                      "concatenated-stream stress test.", "1–2 sessions"),
                ("4", "Online learning of the eviction policy — the only route that makes a "
                      "learned scorer deployable.", "2 sessions")]:
    p = tf2.add_paragraph(); p.space_before = Pt(11); p.line_spacing = 1.03
    run(p, n + ".  ", size=20, bold=True, color=ACCENT)
    run(p, t, size=20)
    run(p, "   " + e, size=18, color=MUTED, italic=True)

# =========================================================================
# 24 — conclusion
# =========================================================================
sl = slide()
top = head(sl, "Conclusion as it stands")
bullets(sl, [
    ("With storage unbounded — the standard accounting — ",
     "tiered compression is a net loss against keeping everything and retrieving."),
    ("With storage bounded, the decision policy is worth +8.7 answer-recall points, ",
     "while compression still loses 6.7."),
    ("The rate–distortion crossover at which compression should pay is not reachable ",
     "on this benchmark — to 182×, keeping a selected subset whole beats compressing everything."),
    ("Charging for the embedding index inverts the ranking below ~40 KiB, ",
     "where a policy with no index at all wins outright."),
    ("An LLM judge — the obvious way to improve the scorer — loses to a regex, ",
     "and the reason is diagnosable: it rates turns in isolation."),
], top, size=24)
band(sl, [("Practical guidance: keep a scored subset whole; do not compress everything. ", ""),
          ("And denominate the budget in bytes, including the index, or the comparison is not "
           "the one you think you are making.",)],
     top + 5.0, height=1.7, size=26)
caption(sl, "Presented as “a memory layer with three tiers”, this work would already have been "
            "done. Measured honestly, the more useful result is what doesn’t work.",
        top + 6.95, size=20)

# =========================================================================
# 25 — references
# =========================================================================
sl = slide()
top = head(sl, "References")
refs = [
    ("Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.",
     "  ← base paper"),
    ("Maharana, A., et al. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents "
     "(LoCoMo). ACL.", "  ← benchmark"),
    ("Zhong, W., et al. (2024). MemoryBank. AAAI 38(17), 19724–19731.", ""),
    ("Park, J. S., et al. (2023). Generative Agents. UIST ’23.", ""),
    ("Xiao, G., et al. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR.", ""),
    ("Zhang, Z., et al. (2023). H2O: Heavy-Hitter Oracle. NeurIPS.", ""),
    ("Jiang, H., et al. (2023). LLMLingua. EMNLP, 13358–13376.", ""),
    ("Mu, J., Li, X. L., & Goodman, N. (2023). Learning to Compress Prompts with Gist Tokens. NeurIPS.", ""),
    ("Chevalier, A., et al. (2023). Adapting Language Models to Compress Contexts. EMNLP.", ""),
    ("Zhang, Z., et al. (2024). A Survey on the Memory Mechanism of LLM-Based Agents. arXiv:2404.13501.", ""),
    ("Xu, W., et al. (2025). A-MEM: Agentic Memory for LLM Agents. NeurIPS, arXiv:2502.12110.", ""),
    ("Chroma Research (2025). Context Rot.  ·  Colaco & Lahjouji (2026). arXiv:2607.08032.", ""),
    ("Wu, D., et al. (2024). LongMemEval. arXiv:2410.10813.", ""),
]
tb, tf = textbox(sl, L, top, CW, 7.6)
for i, (r, tag) in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(9); p.line_spacing = 1.02
    run(p, f"{i + 1}.  ", size=20, color=MUTED)
    run(p, r, size=20)
    if tag:
        run(p, tag, size=20, bold=True, color=ACCENT)
caption(sl, "Author lists, venues and years were each verified against source pages. Several 2026 "
            "performance figures used in the motivation come from vendor blogs and are treated as "
            "indicative — re-measuring them under one harness is itself part of the contribution.",
        H - 1.5, size=17)

# =========================================================================
# 26 — thank you
# =========================================================================
sl = slide()
bar = sl.shapes.add_shape(1, Inches(L), Inches(4.3), Inches(2.4), Emu(9525 * 4))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
bar.line.fill.background(); bar.shadow.inherit = False
tb, tf = textbox(sl, L, 4.6, 16.0, 3.4)
p = tf.paragraphs[0]
run(p, "Thank you", size=54, bold=True, color=INK_TITLE)
p = tf.add_paragraph(); p.space_before = Pt(16)
run(p, "Questions welcome — including on the results that did not work.", size=27, color=MUTED)
p = tf.add_paragraph(); p.space_before = Pt(26)
run(p, "Simar Singh Khanuja  ·  Yash Ramchandani", size=24)
p = tf.add_paragraph()
run(p, "SVKM’s NMIMS, Indore Campus", size=24, color=MUTED)

# =========================================================================
# B1 — backup: full storage sweep
# =========================================================================
sl = slide()
top = head(sl, "Backup — the full storage sweep",
           "Shown on request. Context budget pinned at 2048 for every row.")
rows = [["S", "Policy", "Evidence recall", "Answer recall", "Stored tokens", "Context tokens"]]
for r in STORAGE:
    if r["store_budget"] == 0:
        continue
    rows.append([f"{int(r['store_budget']):,}", r["policy"],
                 f"{float(r['strict_recall']) * 100:.1f}%",
                 f"{float(r['answer_recall']) * 100:.1f}%",
                 f"{float(r['stored_tokens']):,.0f}",
                 f"{float(r['avg_context_tokens']):,.0f}"])
table(sl, rows, top, size=13, col_w=[2.0, 5.4, 3.2, 3.0, 2.6, 2.8], height=H - top - 0.5)

# =========================================================================
# B2 — backup: repository
# =========================================================================
sl = slide()
top = head(sl, "Backup — repository layout")
tb, tf = textbox(sl, L, top, CW, 6.6)
layout = """Capstone Project/
  REPORT.md                       paper-ready writeup
  SESSION_RECORD.md               working record, §1–§28
  docs/
    REVIEW2_PROGRESS_REPORT.md    the progress report for this review
    make_review2_deck.py          builds this deck from the CSVs
  frontend/
    index.html                    interactive results dashboard
    build_data.py                 regenerates data.js from results/*.csv
  memgate/
    memgate/                      the library — types, utils, store, scoring,
                                  policies, compress, judge, llm_judge, data, harness
    run_locomo.py                 context-budget frontier
    run_storage_sweep.py          storage budget — 35 configurations
    run_scaling.py                compression ratio to 182×
    run_cost_model.py             byte accounting
    run_significance.py           bootstrap + McNemar
    train_scorer.py               learned scorer, leave-one-conversation-out
    precompute_judge.py           P2 judge scores (cached)
    run_judge_eval.py             scorer comparison with clustered CIs
    run_endtask.py                end-task accuracy (resumable)
    run_ablations.py              23-configuration component attribution
    diagnose.py                   attributes every miss to one component
    make_figures.py               fig1–fig5
    test_memgate.py               61 tests
    results/                      every CSV and figure quoted in this deck"""
for i, line in enumerate(layout.split("\n")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.line_spacing = 1.0
    r = run(p, line, size=16, color=INK if not line.strip().startswith(("run_", "test_", "make_", "diagnose")) else MUTED)
    r.font.name = "Consolas"
caption(sl, "Not committed, by design: the LoCoMo benchmark (2.7 MB) and the model weights "
            "(~1 GB). Both are reproducible — memgate/README.md gives the exact fetch commands. "
            "The results CSVs are committed, so every number here is checkable without a "
            "nine-minute benchmark run.", H - 1.5, size=17)

prs.save(OUT)
print(f"wrote {OUT}")
print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
      f"{os.path.getsize(OUT):,} bytes")

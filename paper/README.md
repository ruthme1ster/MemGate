# MemGate — IEEE conference paper

```
paper/
  main.tex          the paper (hand-written prose)
  refs.bib          19 references
  tables/*.tex      GENERATED — do not edit by hand
  figures/*.pdf     vector figures, copied from memgate/results/figures/
  make_tables.py    regenerates tables/ from memgate/results/*.csv
  check.py          structural checks (no TeX installed on the dev machine)
```

## Compiling

**Tectonic is installed locally** (`brew install tectonic`, v0.17.0), so the
paper compiles here:

```sh
cd paper && tectonic -X compile main.tex
```

First run downloads the TeX resource bundle package-by-package and takes a
while; afterwards it is cached in
`~/Library/Caches/TectonicProject.Tectonic/` and recompiles take seconds.
Tectonic runs BibTeX and the rerun passes itself, so `[?]` citations resolve
without a manual second pass.

An earlier note here claimed the install was impossible because Homebrew could
not reach `ghcr.io`. **That was wrong.** `ghcr.io` answers normally; the link
was just slow enough (6–40 KB/s) that the portable-Ruby fetch looked like a
hang. It completed in about 50 minutes. If it appears stuck, measure the
`.incomplete` file in `~/Library/Caches/Homebrew/` before concluding anything.

Overleaf still works as a fallback:

```sh
sh paper/bundle.sh        # regenerates tables, runs check.py, writes the zip
```

Then: **overleaf.com → New Project → Upload Project**, pick
`MemGate_paper_overleaf.zip`, and set **Compiler: pdfLaTeX**, **Main document:
main.tex**. Overleaf ships `IEEEtran.cls` and `IEEEtran.bst`, so nothing else is
needed. If citations render as `[?]`, recompile once more — BibTeX needs the
second pass.

The zip itself is gitignored: everything in it is already tracked, so
committing it would duplicate the repository.

Once a TeX distribution is available:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before every commit

```bash
python3 paper/make_tables.py   # refresh tables from the result CSVs
python3 paper/check.py         # environments, braces, \input, \cite, \ref
```

`check.py` catches what a failed compile would otherwise surface at submission
time: unbalanced environments, missing `\input` or figure targets, `\cite` keys
with no bib entry, and `\ref` with no `\label`. It is not a substitute for
compiling once on Overleaf.

## Length

**6 pages exactly**, at 4,225 words with 7 tables and 1 figure — measured, not
estimated (`Output written on main.xdv (6 pages...)` in `main.log`). It was 8
pages at 4,975 words, 8 tables and 5 figures before the cut.

### What the cut actually removed

Every result, number and interval survived. What went was duplication and
vertical scaffolding:

* **Four of five figures.** `fig:cost` and `fig:scorer` had *no prose reference
  at all*. `fig:scaling` duplicated `tab:scaling`; `fig:frontier` was a
  full-width `figure*` duplicating `tab:storage`. Only `fig:endtask` remains.
* **`tab:policies`**, a descriptive table, folded into one sentence of prose.
* **The roadmap paragraph** ("the rest of this paper is organised as follows").
* **The contributions `enumerate`**, rewritten as a running paragraph.
* **Prose compression** in Related Work, Metrics, Setup, Scorer analysis,
  Threats, Limitations and Conclusion.

Protected and untouched: `sec:accounting`, `tab:storage`, `tab:significance`,
`sec:endtask`, `tab:endtask`, `fig:endtask`.

### What we learned about cutting, which contradicts the old advice here

The prioritized cut list that used to sit in this section was written without a
compiler. **Applied in full, it removed zero pages.** Dropping the redundant
cost and scaling figures, deleting the restating Threats subsection and merging
the Related Work headings left the document at 8 pages, because the text simply
reflowed into the freed space.

What actually moved the page count was *vertical structure*, not word count:
removing the full-width `figure*` took it 8 to 7, and converting the seven-item
contributions list into a paragraph took it 7 to 6 — the latter worth only 7
words but a large amount of list leading and inter-item space.

A useful diagnostic: compile with `\bibliography` stripped. The body alone was
6 pages while the full document was 7, which located the overflow in the
19-entry bibliography and turned "cut two pages" into the much smaller job of
freeing one column.

Cut by measuring. Each compile takes seconds once the bundle is cached.

## What is deliberately not in the paper

The rule: **a pilot-scale run does not enter the paper.** A number that cannot
carry an interval is worth less than an honest statement that the experiment is
scoped and pending, because the first invites a question the paper cannot
answer and the second does not.

*(End-task accuracy used to be listed here, on a 12-question pilot. The full
715-question run completed on 17 September and it is now
Section~\ref{sec:endtask}, with the corresponding limitation replaced by the
single-reader bound. The rule below is what kept it out until it was real.)*

**Abstractive compression.** `run_abstractive.py` exists and runs, but the only
result on disk is **one conversation** (n=149). With a single cluster the
paired bootstrap has nothing to resample, so every interval in
`results/abstractive.csv` is degenerate — `ci_lo == ci_hi == delta` — and its
`significant` column is an artifact of that, not a finding. Needs all ten
conversations before it is quotable. Until then the Limitations paragraph
*"Extractive compression only"* stands as written.

**Online learning.** `run_online.py`, likewise pilot-scale at **two
conversations** (n=230). Two clusters is an interval in form only.

**LongMemEval.** `run_longmemeval.py` and the data are in place. Note when
reading its output that the item count and the answer-recoverable count differ
(94 and 62 on the default 100-item prefix): strict recall clusters at the
former, answer recall at the latter, and answer recall is the one the project's
claims rest on. A subsample is reported as a subsample and is never described
as the benchmark.

## Revising the prose

The numbers, figures, tables and experimental design come from this
repository. The prose is a draft and should be revised into the authors' own
voice before submission. Two reasons: it will read better, and a paper
submitted under your names should be written in your words.

The sections that most need rewriting, in priority order:

1. **Abstract** — the part reviewers read first and the part most obviously
   drafted rather than written. Rewrite from scratch after the rest is settled.
2. **Introduction, paragraphs 1–3** — the motivation is standard and currently
   reads that way. Your own framing of why this problem matters is better than
   a generic one.
3. **Section~\ref{sec:threats} (Threats to Validity)** — this section describes
   things that happened to you. First-hand narration will read as first-hand.
4. **Conclusion** — currently a summary of the results section. Say what you
   actually think follows from the work.

Sections that need less attention: the method, setup and results subsections,
where the content constrains the phrasing.

A practical check: read a paragraph aloud. Anything you would not say out loud
in a lab meeting should be rewritten.

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

There is **no LaTeX distribution on this machine**, so the paper has never been
compiled here — only structurally checked. Compile it on Overleaf:

1. New Project → Upload Project → zip the whole `paper/` directory.
2. Overleaf ships `IEEEtran.cls` and `IEEEtran.bst`, so nothing else is needed.
3. Set the compiler to **pdfLaTeX** and the main document to `main.tex`.

Locally, once a TeX distribution is installed:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Two passes after `bibtex` are required or the citations render as `[?]`.

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

About 4,500 words with 7 tables and 4 figures, which lands near **8 pages** in
the two-column IEEE conference style. Many venues cap the main text at 6 pages
with two extra pages available at a charge.

To reach 6 pages, cut in this order:

1. `Table \ref{tab:cost}` **or** `Fig. \ref{fig:cost}`, not both — they carry
   the same result (saves ~0.4 page).
2. `Table \ref{tab:scaling}` **or** `Fig. \ref{fig:scaling}`, same reason.
3. Section~\ref{sec:threats} subsection 3, which restates
   Section~\ref{sec:metrics}.
4. The Related Work subsection headings, merged into running paragraphs.

Do **not** cut Section~\ref{sec:accounting} (the storage-accounting
correction), Table~\ref{tab:storage}, or Table~\ref{tab:significance}. Those
three carry the argument.

## What is deliberately not in the paper

**End-task accuracy.** The pipeline is implemented with a closed-book floor and
an oracle ceiling, but has only been validated on a 12-question pilot. Twelve
questions cannot support a claim, so no end-task numbers appear in the results.
The work is described in Limitations as implemented and pending. When the full
715-question run completes, it becomes a new results subsection and the
corresponding limitation is removed.

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

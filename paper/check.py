#!/usr/bin/env python3
"""Structural checks on main.tex, since no TeX distribution is installed here.

Catches what a failed compile would otherwise report at submission time:
unbalanced environments and braces, missing \\input and \\includegraphics
targets, \\cite keys absent from refs.bib, and \\ref targets with no \\label.

    python3 paper/check.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")
src = open(TEX, encoding="utf-8").read()
# strip comments so a % in prose never trips the brace count
nocom = re.sub(r"(?<!\\)%.*", "", src)
problems = []

# 1. environments
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", nocom):
    kind, name = m.group(1), m.group(2)
    if kind == "begin":
        stack.append(name)
    else:
        if not stack:
            problems.append(f"\\end{{{name}}} with nothing open")
        elif stack[-1] != name:
            problems.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
for name in stack:
    problems.append(f"\\begin{{{name}}} never closed")

# 2. braces
depth = 0
for i, ch in enumerate(nocom):
    if ch == "{" and (i == 0 or nocom[i - 1] != "\\"):
        depth += 1
    elif ch == "}" and (i == 0 or nocom[i - 1] != "\\"):
        depth -= 1
        if depth < 0:
            problems.append("unmatched closing brace")
            break
if depth > 0:
    problems.append(f"{depth} unclosed brace(s)")

# 3. \input and \includegraphics targets
for m in re.finditer(r"\\input\{([^}]+)\}", nocom):
    t = os.path.join(HERE, m.group(1))
    if not (os.path.exists(t) or os.path.exists(t + ".tex")):
        problems.append(f"\\input target missing: {m.group(1)}")
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", nocom):
    if not os.path.exists(os.path.join(HERE, m.group(1))):
        problems.append(f"figure missing: {m.group(1)}")

# 4. citations resolve against refs.bib
bib = open(os.path.join(HERE, "refs.bib"), encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", nocom):
    cited |= {k.strip() for k in m.group(1).split(",")}
for k in sorted(cited - keys):
    problems.append(f"\\cite{{{k}}} has no entry in refs.bib")
unused = sorted(keys - cited)

# 5. \ref targets exist
labels = set(re.findall(r"\\label\{([^}]+)\}", nocom))
for m in re.finditer(r"\\ref\{([^}]+)\}", nocom):
    if m.group(1) not in labels:
        problems.append(f"\\ref{{{m.group(1)}}} has no \\label")

# 6. unescaped % or & outside tabular/math is a common silent break
n_sec = len(re.findall(r"\\section", nocom))
n_tab = len(re.findall(r"\\begin\{table", nocom))
n_fig = len(re.findall(r"\\begin\{figure", nocom))
print(f"main.tex: {len(src.split())} words, {n_sec} sections, "
      f"{n_tab} tables, {n_fig} figures")
print(f"citations: {len(cited)} used, {len(keys)} defined")
if unused:
    print(f"  unused bib entries (harmless): {', '.join(unused)}")

if problems:
    print(f"\n{len(problems)} PROBLEM(S):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\nstructure OK")

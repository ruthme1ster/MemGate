#!/bin/sh
# Rebuild the Overleaf upload bundle.
#
# There is no TeX distribution on the development machine and Homebrew cannot
# reach ghcr.io from this network, so the paper is compiled on Overleaf rather
# than locally. This produces the zip to upload.
#
#   sh paper/bundle.sh
#
# Then: overleaf.com -> New Project -> Upload Project -> pick the zip,
# set Compiler = pdfLaTeX and Main document = main.tex.
set -e
cd "$(dirname "$0")"
python3 make_tables.py
python3 check.py
rm -f ../MemGate_paper_overleaf.zip
zip -qr ../MemGate_paper_overleaf.zip \
    main.tex refs.bib tables figures -x '*.DS_Store'
echo "wrote MemGate_paper_overleaf.zip ($(du -h ../MemGate_paper_overleaf.zip | cut -f1))"

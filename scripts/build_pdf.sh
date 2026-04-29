#!/bin/bash
# Compile report.tex to report.pdf using tectonic.
# The .tex file is the source of truth; edit it directly.
# To regenerate report.tex from report.md, run scripts/md_to_tex.sh.
set -e
cd "$(dirname "$0")/.."

tectonic --keep-logs --outdir report report/report.tex

echo "Generated report/report.pdf"
ls -la report/report.pdf

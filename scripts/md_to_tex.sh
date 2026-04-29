#!/bin/bash
# Regenerate report.tex from report.md (only needed if you want to re-sync from markdown).
# Normal workflow: edit report.tex directly, then run scripts/build_pdf.sh.
set -e
cd "$(dirname "$0")/.."

pandoc report/report.md \
  --resource-path="report:.:figures" \
  --variable=geometry:"margin=2cm" \
  --variable=fontsize:11pt \
  --variable=linestretch:1.5 \
  --variable=classoption:onecolumn \
  --variable=colorlinks:true \
  --variable=linkcolor:blue \
  --variable=urlcolor:blue \
  --standalone \
  --toc=false \
  -o report/report.tex

echo "Regenerated report/report.tex from report.md"
ls -la report/report.tex

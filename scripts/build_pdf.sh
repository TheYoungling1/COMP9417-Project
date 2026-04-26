#!/bin/bash
# Convert report.md to report.pdf using pandoc + tectonic
set -e
cd "$(dirname "$0")/.."

# Resolve image paths relative to report/
pandoc report/report.md \
  --pdf-engine=tectonic \
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
  -o report/report.pdf

echo "Generated report/report.pdf"
ls -la report/report.pdf

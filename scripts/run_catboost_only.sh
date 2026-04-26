#!/bin/bash
# Launch catboost experiments for all 5 datasets in parallel via Modal
set -e
cd "$(dirname "$0")/.."

DATASETS=(seoul_bike appliances_energy hcc_survival ida2016 crop_mapping)
for ds in "${DATASETS[@]}"; do
    echo "Launching catboost on $ds..."
    .venv/bin/modal run --detach modal_app.py::run_single --dataset-name "$ds" --model-name catboost > "/tmp/catboost_$ds.log" 2>&1 &
    sleep 1  # stagger so modal deploy doesn't duplicate
done
wait
echo "All catboost launches submitted."

# xRFM Comparison Benchmark

Graduate-level comparison of xRFM (Beaglehole et al., 2025; arXiv:2508.10053) against XGBoost, Random Forest, CatBoost, and TabPFN-v2 on 5 tabular datasets.

## Datasets (all UCI, verified absent from TALENT benchmark + xRFM meta-test)

| # | Dataset | UCI ID | n | d | Task | Feature Types |
|---|---------|--------|---|---|------|---------------|
| 1 | Seoul Bike Sharing Demand | 560 | 8,760 | 12 | Regression | 9 num + 3 cat |
| 2 | Appliances Energy Prediction | 374 | 19,735 | 27 | Regression | All numeric (incl. rv1, rv2 random noise) |
| 3 | HCC Survival | 423 | 165 | 49 | Binary | 22 num + 27 cat (heavy missingness) |
| 4 | IDA2016 Scania APS Failures | 414 | 60,000 | 170 | Binary (imbalanced) | All numeric (heavy missingness) |
| 5 | Crop Mapping (Winnipeg) | 525 | 325,834 (subsampled to 50k) | 174 | Multiclass (7 classes) | All numeric (SAR+optical) |

## Models

- **xRFM** (Beaglehole et al., 2025) — tree-partitioned recursive feature machine with AGOP-based splits
- **XGBoost** (Chen & Guestrin, 2016) — gradient boosted trees with early stopping
- **Random Forest** (Breiman, 2001) — bagged trees baseline
- **CatBoost** (Prokhorenkova et al., 2018) — ordered boosting with native categorical handling
- **TabPFN-v2** (Hollmann et al., *Nature* 2025) — prior-data-fitted transformer (run separately, see step 4)

## Structure

```
.
├── modal_app.py             # Modal GPU application
├── src/
│   ├── datasets.py          # UCI dataset loaders
│   ├── preprocessing.py     # Standardization + encoding
│   ├── models.py            # Unified model wrappers
│   ├── metrics.py           # RMSE / Accuracy / AUC-ROC
│   ├── experiments.py       # Optuna HP tuning loop
│   └── plots.py             # Figures + tables
├── scripts/                 # Smoke test, consolidation, PDF build
├── results/                 # Per-experiment JSONs
├── figures/                 # Generated figures
├── report/                  # Final report (markdown + PDF)
└── data_cache/              # Local dataset cache
```

## Setup

- Python 3.12 with the bundled `.venv`:
  ```bash
  source .venv/bin/activate
  ```
- Modal account + CLI:
  ```bash
  pip install modal
  modal token new
  ```
- For the PDF report: `pandoc` and `tectonic` on PATH.

## Running

The full pipeline is 25 main experiments (5 datasets × 5 models) plus interpretability and scaling sweeps. By default `run_all_main` runs 4 of the 5 models — TabPFN is excluded from CI because of its license and is launched separately in step 4.

```bash
# 1. (Optional) Local CPU smoke test — 5 min, no Modal credits.
#    Runs 3 Optuna trials per (Seoul Bike + HCC) x (xgboost, rf, xrfm).
python scripts/smoke_test.py

# 2. Cache all 5 UCI datasets into the Modal volume (one-time).
modal run modal_app.py::prepare_data

# 3. Main benchmark: xRFM, XGBoost, RF, CatBoost on all 5 datasets in parallel.
modal run modal_app.py::run_all_main
# optional: --seed 42 --crop-subsample 50000

# 4. TabPFN-v2 — license-gated, kicked off per-dataset:
for ds in seoul_bike appliances_energy hcc_survival ida2016 crop_mapping; do
    modal run --detach modal_app.py::run_single --dataset-name $ds --model-name tabpfn
done

# 5. Interpretability (AGOP vs PCA vs MI vs permutation on Appliances Energy).
modal run modal_app.py::interpretability

# 6. Scaling sweep on IDA2016 (sizes 1k, 2.5k, 5k, 10k, 20k, 36k).
modal run modal_app.py::run_scaling

# 7. Pull the per-experiment JSONs back into ./results/.
modal run modal_app.py::download_results

# 8. Aggregate, plot, and build the PDF report.
python scripts/final_consolidate.py
python scripts/plot_per_leaf_agop.py
bash   scripts/build_pdf.sh
```

`scripts/run_catboost_only.sh` is a convenience wrapper for re-running just the CatBoost sweep (e.g. after tweaking its HP space) without touching the other models.

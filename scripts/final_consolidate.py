"""Final consolidation — produces camera-ready tables and figures for the report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plots import (build_main_results_table, plot_interpretability,
                        plot_scaling, rank_random_features)


DATASETS_ORDER = ["seoul_bike", "appliances_energy", "hcc_survival", "ida2016", "crop_mapping"]
MODELS_ORDER = ["xrfm", "xgboost", "random_forest", "catboost"]
MODEL_LABELS = {"xrfm": "xRFM", "xgboost": "XGBoost", "random_forest": "RF", "catboost": "CatBoost"}
DATASET_LABELS = {
    "seoul_bike": "Seoul Bike",
    "appliances_energy": "Appliances Energy",
    "hcc_survival": "HCC Survival",
    "ida2016": "IDA2016",
    "crop_mapping": "Crop Mapping",
}


def build_consolidated_table(downloads_dir: Path) -> pd.DataFrame:
    """Build a wide table: rows=datasets, columns=(model, metric)."""
    rows = []
    for f in sorted(downloads_dir.glob("main_*.json")):
        if "tabpfn" in f.name:
            continue
        with f.open() as fh:
            d = json.load(fh)
        rows.append(d)
    df = build_main_results_table(rows)
    return df


def render_flat_table_markdown(df: pd.DataFrame) -> str:
    """Flat table: one row per (dataset, model), columns = metrics.
    One table for regression, one for classification."""
    out = []

    # Regression
    reg = df[df["task"] == "regression"].copy()
    if len(reg):
        out.append("### Regression results")
        reg["Dataset"] = reg["dataset"].map(lambda d: DATASET_LABELS.get(d, d))
        reg["Model"] = reg["model"].map(lambda m: MODEL_LABELS.get(m, m))
        reg["RMSE"] = reg.get("test_rmse", np.nan).map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        reg["R²"] = reg.get("test_r2", np.nan).map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        reg["Train (s)"] = reg["train_time_s"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        reg["Inf. (μs/sample)"] = reg["inf_s_per_sample"].map(lambda v: f"{v*1e6:.1f}" if pd.notna(v) else "—")
        reg = reg.sort_values(["dataset", "model"])
        out.append(reg[["Dataset", "Model", "RMSE", "R²", "Train (s)", "Inf. (μs/sample)"]].to_markdown(index=False))
        out.append("")

    # Classification
    cls = df[df["task"].isin(["binary", "multiclass"])].copy()
    if len(cls):
        out.append("### Classification results")
        cls["Dataset"] = cls["dataset"].map(lambda d: DATASET_LABELS.get(d, d))
        cls["Model"] = cls["model"].map(lambda m: MODEL_LABELS.get(m, m))
        cls["Accuracy"] = cls.get("test_accuracy", np.nan).map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        cls["AUC-ROC"] = cls.get("test_auc_roc", np.nan).map(lambda v: f"{v:.4f}" if pd.notna(v) and not (isinstance(v, float) and np.isnan(v)) else "—")
        cls["Train (s)"] = cls["train_time_s"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        cls["Inf. (μs/sample)"] = cls["inf_s_per_sample"].map(lambda v: f"{v*1e6:.1f}" if pd.notna(v) else "—")
        cls = cls.sort_values(["dataset", "model"])
        out.append(cls[["Dataset", "Model", "Accuracy", "AUC-ROC", "Train (s)", "Inf. (μs/sample)"]].to_markdown(index=False))
        out.append("")

    return "\n".join(out)


def bold_best(df: pd.DataFrame) -> pd.DataFrame:
    """Highlight best value per (dataset, metric)."""
    df = df.copy()
    for ds, sub in df.groupby("dataset"):
        if sub["task"].iloc[0] == "regression":
            metric = "test_rmse"
            best_idx = sub[metric].idxmin()
        else:
            metric = "test_auc_roc" if sub["task"].iloc[0] == "binary" else "test_accuracy"
            best_idx = sub[metric].idxmax()
        if pd.notna(best_idx):
            pass  # placeholder for actual formatting if needed
    return df


def main():
    downloads = Path("results/downloads")
    figures = Path("figures")
    figures.mkdir(exist_ok=True)

    # --- Main table ---
    df = build_consolidated_table(downloads)
    print(f"Loaded {len(df)} main results")
    md = render_flat_table_markdown(df)
    with open("results/final_table.md", "w") as f:
        f.write(md)
    df.to_csv("results/final_results.csv", index=False)
    print(md[:2000])
    print()

    # --- Interpretability ---
    interp = downloads / "interpretability_appliances_energy_seed42.json"
    if interp.exists():
        plot_interpretability(interp, figures / "interpretability_appliances.png", top_k=15)
        ranks = rank_random_features(interp)
        ranks.to_csv("results/random_feature_ranks.csv", index=False)
        print("Random feature ranks:")
        print(ranks.to_markdown(index=False))
        print()

    # --- Scaling ---
    scaling_files = sorted(downloads.glob("scaling_ida2016_*.json"))
    all_scaling = []
    for f in scaling_files:
        with f.open() as fh:
            all_scaling.append(json.load(fh))
    # Filter out known-stale: tabpfn (license-blocked) and n=25000 (old sizes list)
    cleaned = [r for r in all_scaling if r.get("model") != "tabpfn" and r.get("n_subsample") != 25000]
    import json as _json
    with open("results/scaling_results_clean.json", "w") as f:
        _json.dump(cleaned, f, indent=2, default=str)
    plot_scaling(Path("results/scaling_results_clean.json"), figures, dataset_name="ida2016")

    tuned_scaling_files = sorted(downloads.glob("scaling_tuned_ida2016_*.json"))
    if tuned_scaling_files:
        tuned_scaling = []
        for f in tuned_scaling_files:
            with f.open() as fh:
                tuned_scaling.append(json.load(fh))
        with open("results/scaling_tuned_results_clean.json", "w") as f:
            _json.dump(tuned_scaling, f, indent=2, default=str)
        plot_scaling(
            Path("results/scaling_tuned_results_clean.json"),
            figures,
            dataset_name="ida2016",
            output_stem="ida2016_tuned",
            display_name="IDA2016 (tuned frozen HP)",
        )

    print(f"Done. Files in results/ and figures/.")


if __name__ == "__main__":
    main()

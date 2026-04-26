"""Plotting and table-generation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def build_main_results_table(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-experiment JSONs into a combined table."""
    rows = []
    for r in results:
        row = {
            "dataset": r.get("dataset"),
            "model": r.get("model"),
            "task": r.get("task"),
            "n_train": r.get("n_train"),
            "d": r.get("d"),
            "primary_metric": r.get("primary_metric"),
            "val_primary": r.get("val_primary"),
            "train_time_s": r.get("train_time_s"),
            "inf_s_per_sample": r.get("inference_time_s_per_sample"),
            "error": r.get("error"),
        }
        tm = r.get("test_metrics") or {}
        for k, v in tm.items():
            row[f"test_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def render_main_table_markdown(df: pd.DataFrame) -> str:
    """Format main results for the report (markdown)."""
    lines = ["# Main Results\n"]
    for dataset, sub in df.groupby("dataset"):
        task = sub["task"].iloc[0]
        lines.append(f"## {dataset} ({task}, n_train={int(sub['n_train'].iloc[0])}, d={int(sub['d'].iloc[0])})")
        cols = ["model", "test_rmse", "test_r2", "test_accuracy", "test_auc_roc",
                "train_time_s", "inf_s_per_sample"]
        cols = [c for c in cols if c in sub.columns and sub[c].notna().any()]
        t = sub[cols].copy()
        for c in cols:
            if c in {"test_rmse", "test_r2", "test_accuracy", "test_auc_roc"}:
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
            elif c == "train_time_s":
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
            elif c == "inf_s_per_sample":
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v*1e6:.1f} μs")
        lines.append(t.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def render_main_table_latex(df: pd.DataFrame) -> str:
    """Format main results as a LaTeX table (rows = datasets x models)."""
    metric_cols = ["test_rmse", "test_accuracy", "test_auc_roc", "train_time_s", "inf_s_per_sample"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    cols = ["dataset", "model"] + metric_cols
    t = df[cols].copy()
    return t.to_latex(index=False, float_format="%.4f", na_rep="-")


def plot_interpretability(
    interpret_json_path: Path,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Side-by-side bar chart of AGOP vs PCA vs MI vs Permutation importance."""
    with open(interpret_json_path) as f:
        data = json.load(f)

    names = np.array(data["feature_names"])
    agop = np.array(data["agop_diagonal_agg"])
    pca = np.array(data["pca_loadings_abs_agg"])
    mi = np.array(data["mutual_info"])
    perm = np.array(data["permutation_importance"])
    random_features = set(data.get("random_features", []))

    def _norm(x):
        s = np.sum(np.abs(x))
        return np.abs(x) / s if s > 0 else np.abs(x)

    agop_n = _norm(agop)
    pca_n = _norm(pca)
    mi_n = _norm(mi)
    perm_n = _norm(perm)

    # rank by AGOP for stable x-axis order
    order = np.argsort(-agop_n)
    top = order[:top_k]

    x = np.arange(len(top))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, agop_n[top], width, label="AGOP diag", color="#1f77b4")
    ax.bar(x - 0.5 * width, pca_n[top], width, label="PCA loading", color="#ff7f0e")
    ax.bar(x + 0.5 * width, mi_n[top], width, label="Mutual Info", color="#2ca02c")
    ax.bar(x + 1.5 * width, perm_n[top], width, label="Perm. Imp. (XGB)", color="#d62728")

    labels = []
    for i in top:
        fname = names[i]
        if fname in random_features:
            labels.append(f"★ {fname}")  # mark ground-truth noise
        else:
            labels.append(fname)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Normalized importance")
    title = f"Feature importance comparison — {data.get('dataset', '')} (★ = ground-truth random noise)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def rank_random_features(interpret_json_path: Path) -> pd.DataFrame:
    """For each method, report the rank of the known random features (1 = most important).
    Lower rank = method incorrectly promotes noise. Higher rank (closer to d) = correct demotion.
    """
    with open(interpret_json_path) as f:
        data = json.load(f)

    names = data["feature_names"]
    random_features = data.get("random_features", [])
    d = len(names)

    # AGOP, PCA, MI are non-negative by construction; permutation_importance can
    # be negative (a feature whose permutation IMPROVES validation loss).
    # Ranking by abs() would promote a negative score to "high importance", which
    # is the opposite of what permutation importance means.
    methods = {
        "AGOP": (np.array(data["agop_diagonal_agg"]), True),       # non-negative
        "PCA": (np.array(data["pca_loadings_abs_agg"]), True),     # non-negative
        "MI": (np.array(data["mutual_info"]), True),               # non-negative
        "Permutation": (np.array(data["permutation_importance"]), False),  # can be negative
    }
    rows = []
    for mname, (vals, nonneg) in methods.items():
        # Use abs only for non-negative scores (numerical noise around 0 is fine);
        # for permutation, rank by signed value so negatives are demoted, not promoted.
        key = np.abs(vals) if nonneg else vals
        order = np.argsort(-key)
        for rf in random_features:
            if rf in names:
                idx = names.index(rf)
                rank = int(np.where(order == idx)[0][0]) + 1
                rows.append({"method": mname, "feature": rf, "rank": rank, "d": d})
    return pd.DataFrame(rows)


def plot_scaling(scaling_json_path: Path, output_dir: Path, dataset_name: str) -> None:
    """Two-panel plot: test metric vs n, train time vs n (log-log)."""
    with open(scaling_json_path) as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    df = df.dropna(subset=["train_time_s", "test_metrics"])

    # primary metric is the same across rows of one dataset
    primary = df["primary_metric"].iloc[0] if "primary_metric" in df.columns and len(df) else "accuracy"
    df["metric_val"] = df["test_metrics"].apply(lambda m: (m or {}).get(primary, None))
    df = df.dropna(subset=["metric_val"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    color_map = {
        "xrfm": "#1f77b4",
        "xgboost": "#ff7f0e",
        "random_forest": "#2ca02c",
        "catboost": "#d62728",
        "tabpfn": "#9467bd",
    }
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("n_train_actual")
        ax1.plot(sub["n_train_actual"], sub["metric_val"], "o-",
                 label=model, color=color_map.get(model))
        ax2.plot(sub["n_train_actual"], sub["train_time_s"], "o-",
                 label=model, color=color_map.get(model))

    ax1.set_xlabel("Training samples n")
    ax1.set_ylabel(f"Test {primary}")
    ax1.set_xscale("log")
    ax1.set_title(f"Test {primary} vs training size on {dataset_name}")
    ax1.legend(loc="best")

    ax2.set_xlabel("Training samples n")
    ax2.set_ylabel("Training time (s)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(f"Training time vs training size on {dataset_name}")
    ax2.legend(loc="best")

    fig.tight_layout()
    out = output_dir / f"scaling_{dataset_name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    csv_out = output_dir / f"scaling_{dataset_name}.csv"
    df[["model", "n_train_actual", "metric_val", "train_time_s", "inference_time_s_per_sample"]].to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

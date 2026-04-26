"""Aggregate Modal results into tables + figures for the report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.plots import (build_main_results_table, plot_interpretability,
                        plot_scaling, rank_random_features,
                        render_main_table_markdown)


def main():
    results_dir = Path("results")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True, parents=True)

    # --- Main results ---
    main_files = list((results_dir / "downloads").glob("main_*.json"))
    if not main_files:
        print("No main_*.json files found")
    else:
        results = []
        for f in main_files:
            with f.open() as fh:
                results.append(json.load(fh))
        df = build_main_results_table(results)
        # Sort rows sensibly: dataset then model
        model_order = ["xrfm", "xgboost", "random_forest", "tabpfn"]
        df["model_rank"] = df["model"].map(lambda m: model_order.index(m) if m in model_order else 99)
        ds_order = ["seoul_bike", "appliances_energy", "hcc_survival", "ida2016", "crop_mapping"]
        df["ds_rank"] = df["dataset"].map(lambda d: ds_order.index(d) if d in ds_order else 99)
        df = df.sort_values(["ds_rank", "model_rank"]).drop(columns=["model_rank", "ds_rank"])

        csv_path = Path("results/main_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        md = render_main_table_markdown(df)
        with open("results/main_table.md", "w") as f:
            f.write(md)
        print(f"Saved results/main_table.md")

    # --- Interpretability ---
    interp = results_dir / "downloads" / "interpretability_appliances_energy_seed42.json"
    if not interp.exists():
        interp = results_dir / "interpretability_appliances_energy.json"
    if interp.exists():
        out = figures_dir / "interpretability_appliances.png"
        plot_interpretability(interp, out, top_k=15)
        ranks = rank_random_features(interp)
        print("\nRandom feature ranks (lower=more prominent, higher=more demoted):")
        print(ranks)
        ranks.to_csv(results_dir / "random_feature_ranks.csv", index=False)
    else:
        print(f"(interpretability result not yet present: {interp})")

    # --- Scaling ---
    scaling = results_dir / "scaling_results.json"
    if scaling.exists():
        plot_scaling(scaling, figures_dir, dataset_name="ida2016")
    else:
        print(f"(scaling result not yet present: {scaling})")


if __name__ == "__main__":
    main()

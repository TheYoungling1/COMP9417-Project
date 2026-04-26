"""Locally rerun permutation importance with the corrected feature representation.

Loads the existing interpretability JSON, replaces the broken permutation column
with one computed on the same representation XGBoost trained on (X_val_ord),
and writes the corrected file alongside the original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset
from src.preprocessing import make_splits
from src.models import XGBoostWrapper

from sklearn.inspection import permutation_importance


SEED = 42
DATASET = "appliances_energy"
SRC_JSON = ROOT / "results" / "downloads" / f"interpretability_{DATASET}_seed{SEED}.json"
OUT_JSON = ROOT / "results" / "downloads" / f"interpretability_{DATASET}_seed{SEED}_perm_fixed.json"


def main() -> None:
    print(f"Loading {DATASET}...")
    ds = load_dataset(DATASET)
    print(f"  {ds.summary()}")

    split = make_splits(
        ds.X, ds.y,
        numerical_cols=ds.numerical_cols,
        categorical_cols=ds.categorical_cols,
        task=ds.task, n_classes=ds.n_classes,
        random_state=SEED,
    )

    # Same XGB config as modal_app.py:run_interpretability uses for the proxy.
    xgb_hp = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
    }
    print("Training XGBoost proxy...")
    xgb = XGBoostWrapper(xgb_hp, split.task, ds.n_classes)
    xgb.fit(split)

    # Permutation on the SAME representation the model trained on.
    print("Computing permutation importance on X_val_ord (matching training)...")
    perm = permutation_importance(
        xgb.model, split.X_val_ord, split.y_val,
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
    perm_mean = perm.importances_mean

    feature_names_ord = split.feature_names_ord

    # Quick sanity dump
    print("\nTop-5 by signed permutation importance:")
    order = np.argsort(-perm_mean)
    for i in order[:5]:
        print(f"  {feature_names_ord[i]:>20s}  perm={perm_mean[i]:+.6f}")
    print("\nBottom-5 by signed permutation importance:")
    for i in order[-5:]:
        print(f"  {feature_names_ord[i]:>20s}  perm={perm_mean[i]:+.6f}")

    print("\nRandom-feature ranks (signed, 1=most important, d=least):")
    d = len(perm_mean)
    for rf in ds.extra.get("random_features", []):
        if rf in feature_names_ord:
            idx = feature_names_ord.index(rf)
            rank = int(np.where(order == idx)[0][0]) + 1
            print(f"  {rf}: signed_perm={perm_mean[idx]:+.6f}, rank={rank}/{d}")

    # Merge with the existing interpretability artifact (preserve AGOP/MI/PCA).
    if SRC_JSON.exists():
        with SRC_JSON.open() as f:
            doc = json.load(f)
    else:
        doc = {"dataset": DATASET, "feature_names": feature_names_ord}

    doc["permutation_importance"] = perm_mean.tolist()
    doc["permutation_importance_features"] = feature_names_ord
    doc["xgb_hp_used"] = xgb_hp
    doc["permutation_fix_note"] = (
        "permutation_importance recomputed locally on X_val_ord (the same "
        "representation XGBoost trained on); previous values used X_val_enc "
        "which was inconsistent with X_train_ord training."
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

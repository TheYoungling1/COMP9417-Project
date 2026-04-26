"""Local smoke test: run 1 small experiment per model x dataset combo.
Tests the pipeline end-to-end on CPU before launching on Modal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import load_seoul_bike, load_hcc_survival
from src.experiments import run_experiment


def test_regression():
    ds = load_seoul_bike()
    print(f"\n=== {ds.name} ({ds.task}) ===")
    for m in ["xgboost", "random_forest", "xrfm"]:
        print(f"\n--- {m} ---")
        r = run_experiment(ds, m, device="cpu", n_trials=3, random_state=42)
        print(f"  val_primary: {r.val_primary:.4f}")
        print(f"  test_metrics: {r.test_metrics}")
        print(f"  train_time: {r.train_time_s:.2f}s")
        print(f"  best_hp: {r.best_hp}")
        if r.error:
            print(f"  ERROR: {r.error[:300]}")


def test_binary():
    ds = load_hcc_survival()
    print(f"\n=== {ds.name} ({ds.task}) ===")
    for m in ["xgboost", "random_forest", "xrfm"]:
        print(f"\n--- {m} ---")
        r = run_experiment(ds, m, device="cpu", n_trials=3, random_state=42)
        print(f"  val_primary: {r.val_primary:.4f}")
        print(f"  test_metrics: {r.test_metrics}")
        print(f"  train_time: {r.train_time_s:.2f}s")
        print(f"  best_hp: {r.best_hp}")
        if r.error:
            print(f"  ERROR: {r.error[:300]}")


if __name__ == "__main__":
    test_regression()
    test_binary()

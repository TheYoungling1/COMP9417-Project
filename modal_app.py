"""Modal app for running xRFM comparison experiments on cloud GPUs.

Usage:
    # Prepare datasets (download + cache on Volume). Run once.
    modal run modal_app.py::prepare_data

    # Run a single experiment
    modal run modal_app.py::run_single --dataset seoul_bike --model xrfm

    # Run all main experiments (5 datasets x 4 models)
    modal run modal_app.py::run_all_main

    # Run interpretability on Appliances Energy
    modal run modal_app.py::run_interpretability

    # Run scaling experiment on IDA2016
    modal run modal_app.py::run_scaling
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import modal


# =========== Modal image & volume ===========

# Use CUDA 12.4 base so xrfm[cu12] compiles cleanly
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "build-essential")
    .env({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .pip_install(
        "torch==2.5.1",  # compatible with CUDA 12
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2.3",
        "pandas",
        "scikit-learn>=1.5",
        "scipy",
        "matplotlib",
        "seaborn",
        "optuna>=3.5",
        "xgboost>=2.0",
        "catboost>=1.2",
        "ucimlrepo",
        "tqdm",
    )
    .pip_install("xrfm[cu12]")  # Mahalanobis-Laplace kernel RFM with tree partitioning
    .pip_install("tabpfn>=2.0")  # retained for future use; license-blocked in CI
    .add_local_python_source("src")
)

app = modal.App("xrfm-benchmark")
data_volume = modal.Volume.from_name("xrfm-data", create_if_missing=True)
results_volume = modal.Volume.from_name("xrfm-results", create_if_missing=True)

DATA_DIR = "/data"
RESULTS_DIR = "/results"


# =========== Data preparation ===========

@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def prepare_data():
    """Download all datasets to the shared Volume."""
    os.environ["XRFM_CACHE_DIR"] = DATA_DIR
    from src.datasets import load_dataset, DATASET_LOADERS
    results = {}
    for name in DATASET_LOADERS:
        print(f"\n=== {name} ===")
        try:
            kwargs = {}
            if name == "crop_mapping":
                # Cache full dataset; subsample at experiment time
                pass
            ds = load_dataset(name)
            print(f"  {ds.summary()}")
            results[name] = {
                "n": ds.n, "d": ds.d, "task": ds.task,
                "num_cols": len(ds.numerical_cols),
                "cat_cols": len(ds.categorical_cols),
                "n_classes": ds.n_classes,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}
    data_volume.commit()
    return results


# =========== Single experiment ===========

@app.function(
    image=image,
    volumes={DATA_DIR: data_volume, RESULTS_DIR: results_volume},
    gpu="A10G",
    timeout=7200,
    memory=32768,
)
def run_single(
    dataset_name: str,
    model_name: str,
    seed: int = 42,
    n_trials: int | None = None,
    crop_subsample: int = 50000,
) -> dict:
    """Run one (dataset, model) experiment with HP tuning.

    crop_subsample: subsample crop_mapping to this many rows to keep experiments tractable.
    """
    os.environ["XRFM_CACHE_DIR"] = DATA_DIR
    os.environ["HF_HOME"] = f"{DATA_DIR}/hf_cache"
    os.environ["TABPFN_MODEL_CACHE_DIR"] = f"{DATA_DIR}/tabpfn_cache"

    from src.datasets import load_dataset, load_crop_mapping
    from src.experiments import run_experiment

    if dataset_name == "crop_mapping":
        ds = load_crop_mapping(n_max=crop_subsample, random_state=seed)
    else:
        ds = load_dataset(dataset_name)

    print(f"Loaded {ds.summary()}")

    result = run_experiment(
        dataset=ds,
        model_name=model_name,
        device="cuda",
        random_state=seed,
    )
    r = asdict(result)
    # Save
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RESULTS_DIR) / f"main_{dataset_name}_{model_name}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump(r, f, indent=2, default=str)
    results_volume.commit()
    print(f"Saved -> {out_path}")
    print(f"  test_metrics: {r.get('test_metrics')}")
    print(f"  train_time_s: {r.get('train_time_s')}")
    print(f"  error: {r.get('error')}")
    return r


# =========== Run all main experiments in parallel ===========

@app.local_entrypoint()
def run_all_main(seed: int = 42, crop_subsample: int = 50000):
    """Launch all 5x4=20 main experiments in parallel."""
    datasets = ["seoul_bike", "appliances_energy", "hcc_survival", "ida2016", "crop_mapping"]
    models = ["xrfm", "xgboost", "random_forest", "catboost"]

    jobs = []
    for ds in datasets:
        for m in models:
            jobs.append((ds, m))

    print(f"Launching {len(jobs)} experiments in parallel...")
    futures = [
        run_single.spawn(dataset_name=d, model_name=m, seed=seed, crop_subsample=crop_subsample)
        for (d, m) in jobs
    ]
    results = []
    for i, fut in enumerate(futures):
        try:
            r = fut.get()
            print(f"[{i+1}/{len(futures)}] {jobs[i]} ✓ "
                  f"metric={r.get('test_metrics')} err={r.get('error') is not None}")
        except Exception as e:
            print(f"[{i+1}/{len(futures)}] {jobs[i]} ✗ {e}")
            r = {"dataset": jobs[i][0], "model": jobs[i][1], "error": str(e)}
        results.append(r)

    out_path = Path("results/main_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved combined results -> {out_path}")
    return results


# =========== Interpretability experiment ===========

@app.function(
    image=image,
    volumes={DATA_DIR: data_volume, RESULTS_DIR: results_volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def run_interpretability(
    dataset_name: str = "appliances_energy",
    seed: int = 42,
) -> dict:
    """On one dataset, fit xRFM; extract per-leaf AGOP diagonals and top eigenvectors.
    Compare to PCA loadings, mutual information, permutation importance.
    """
    os.environ["XRFM_CACHE_DIR"] = DATA_DIR
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from xrfm import xRFM

    from src.datasets import load_dataset
    from src.models import XGBoostWrapper
    from src.preprocessing import make_splits

    ds = load_dataset(dataset_name)
    print(f"Loaded {ds.summary()}")

    split = make_splits(
        ds.X, ds.y,
        numerical_cols=ds.numerical_cols,
        categorical_cols=ds.categorical_cols,
        task=ds.task, n_classes=ds.n_classes,
        random_state=seed,
    )

    feature_names = split.feature_names_enc
    d = split.X_train_enc.shape[1]
    print(f"Encoded dim d={d}, features: {feature_names[:5]}... (showing first 5)")

    # --- Fit xRFM (use reasonable defaults for Appliances) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cls = split.task in {"binary", "multiclass"}
    X_train = torch.tensor(split.X_train_enc, dtype=torch.float32, device=device)
    X_val = torch.tensor(split.X_val_enc, dtype=torch.float32, device=device)
    if is_cls:
        y_train = torch.tensor(np.eye(split.n_classes)[split.y_train.astype(int)], dtype=torch.float32, device=device)
        y_val = torch.tensor(np.eye(split.n_classes)[split.y_val.astype(int)], dtype=torch.float32, device=device)
    else:
        y_train = torch.tensor(split.y_train.reshape(-1, 1).astype(np.float32), device=device)
        y_val = torch.tensor(split.y_val.reshape(-1, 1).astype(np.float32), device=device)

    # Load the tuned best HPs from the main experiment so §3.2 analyses the
    # SAME model that §3.1 reports on. Falls back to defaults if the file is
    # missing (e.g., interpretability run before main run completed).
    best_hp_path = Path(RESULTS_DIR) / f"main_{dataset_name}_xrfm_seed{seed}.json"
    if best_hp_path.exists():
        with best_hp_path.open() as f:
            best_hp = json.load(f).get("best_hp", {})
        print(f"Loaded tuned HPs from {best_hp_path}: {best_hp}")
    else:
        print(f"WARNING: {best_hp_path} not found; falling back to defaults.")
        best_hp = {}
    rfm_params = {
        "model": {
            "kernel": best_hp.get("kernel", "l2"),
            "bandwidth": best_hp.get("bandwidth", 10.0),
            "exponent": best_hp.get("exponent", 1.0),
            "diag": best_hp.get("diag", False),
            "bandwidth_mode": best_hp.get("bandwidth_mode", "constant"),
        },
        "fit": {
            "reg": best_hp.get("reg", 1e-3),
            "iters": best_hp.get("iters", 5),
            "M_batch_size": best_hp.get("M_batch_size", 1000),
            "verbose": False,
            "early_stop_rfm": True,
        },
    }
    tuning_metric = "mse" if split.task == "regression" else "auc"
    model = xRFM(
        rfm_params=rfm_params,
        max_leaf_size=best_hp.get("max_leaf_size", min(60000, len(split.y_train))),
        device=device,
        tuning_metric=tuning_metric,
        n_trees=1,
        verbose=False,
    )
    print(f"Fitting xRFM with rfm_params={rfm_params}, max_leaf_size={best_hp.get('max_leaf_size', 'default')}...")
    model.fit(X_train, y_train, X_val, y_val)

    # --- Extract per-leaf AGOPs ---
    # xRFM's internal self.M may be None if early-stopping picked iteration 0.
    # Force-compute AGOP on each leaf's training data using the learned predictor.
    def _find_leaves(node):
        if node['type'] == 'leaf':
            return [node]
        return _find_leaves(node['left']) + _find_leaves(node['right'])

    leaves = _find_leaves(model.trees[0])
    print(f"#leaves: {len(leaves)}")
    leaf_diagonals = []
    leaf_top_eigvecs = []
    leaf_Ms_np = []
    X_train_t = X_train  # already on device

    for li, leaf in enumerate(leaves):
        rfm = leaf['model']
        # Training samples for this leaf
        train_idx = leaf.get('train_indices')
        if train_idx is None:
            leaf_samples = X_train_t
        else:
            if torch.is_tensor(train_idx):
                train_idx_t = train_idx.to(X_train_t.device)
            else:
                train_idx_t = torch.tensor(train_idx, device=X_train_t.device, dtype=torch.long)
            leaf_samples = X_train_t[train_idx_t]
        # Try getting existing M; if None, compute fresh AGOP
        M_t = rfm.M
        if M_t is None:
            try:
                num_classes_for_M = y_train.shape[1] if y_train.ndim > 1 else 1
                M_t = rfm.fit_M(
                    samples=leaf_samples,
                    num_classes=num_classes_for_M,
                    M_batch_size=min(1000, len(leaf_samples)),
                    inplace=False,
                )
            except Exception as e:
                print(f"[leaf {li}] fit_M failed: {e}")
                M_t = None
        if M_t is None:
            print(f"[leaf {li}] no M available, skipping")
            continue
        M_np = M_t.detach().cpu().numpy() if hasattr(M_t, "detach") else np.asarray(M_t)
        if M_np.ndim == 3:
            M_np = M_np.mean(axis=0)
        if M_np.ndim == 1:
            M_np = np.diag(M_np)
        if M_np.ndim != 2:
            print(f"[leaf {li}] unexpected M shape {M_np.shape}, skipping")
            continue
        leaf_Ms_np.append(M_np)
        leaf_diagonals.append(np.diag(M_np).copy())
        eigvals, eigvecs = np.linalg.eigh((M_np + M_np.T) / 2)
        leaf_top_eigvecs.append(eigvecs[:, -1])

    # --- Aggregate across leaves (mean of diagonals = overall AGOP diagonal) ---
    agop_diag_agg = np.mean(np.stack(leaf_diagonals), axis=0) if leaf_diagonals else np.zeros(d)

    # --- PCA on X_train_enc ---
    pca = PCA(n_components=min(10, d))
    pca.fit(split.X_train_enc)
    pca_loadings_abs_agg = (pca.components_ ** 2).T @ pca.explained_variance_ratio_  # weighted by variance

    # --- Mutual information ---
    if split.task == "regression":
        mi = mutual_info_regression(split.X_train_enc, split.y_train, random_state=seed)
    else:
        mi = mutual_info_classif(split.X_train_enc, split.y_train, random_state=seed)

    # --- Permutation importance (using XGBoost as proxy) ---
    # IMPORTANT: XGBoostWrapper.fit trains on split.X_train_ord (raw numericals
    # + ordinal-encoded categoricals). Permutation must be computed on the SAME
    # representation, otherwise we are permuting features the model never saw
    # (e.g. unstandardised vs standardised values, or one-hot vs ordinal codes).
    xgb_hp = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "early_stopping_rounds": 30}
    xgb = XGBoostWrapper(xgb_hp, split.task, ds.n_classes)
    xgb.fit(split)
    perm = permutation_importance(
        xgb.model, split.X_val_ord, split.y_val,
        n_repeats=10, random_state=seed, n_jobs=-1,
    )
    perm_importances = perm.importances_mean

    out = {
        "dataset": dataset_name,
        "feature_names": feature_names,
        "feature_names_ord": split.feature_names_ord,
        "agop_diagonals_per_leaf": [d.tolist() for d in leaf_diagonals],
        "agop_top_eigvec_per_leaf": [v.tolist() for v in leaf_top_eigvecs],
        "agop_diagonal_agg": agop_diag_agg.tolist(),
        "pca_loadings_abs_agg": pca_loadings_abs_agg.tolist(),
        "pca_top3_components": pca.components_[:3].tolist(),
        "pca_explained_var_ratio": pca.explained_variance_ratio_.tolist(),
        "mutual_info": mi.tolist(),
        "permutation_importance": perm_importances.tolist(),
        "permutation_importance_features": split.feature_names_ord,
        "random_features": ds.extra.get("random_features", []),
        "num_leaves": len(leaf_diagonals),
        "xrfm_hp_used": best_hp,
        "xgb_hp_used": xgb_hp,
    }

    out_path = Path(RESULTS_DIR) / f"interpretability_{dataset_name}_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(f"Saved -> {out_path}")
    return out


# =========== Scaling experiment ===========

@app.function(
    image=image,
    volumes={DATA_DIR: data_volume, RESULTS_DIR: results_volume},
    gpu="A10G",
    timeout=7200,
    memory=32768,
)
def run_scaling_point(
    dataset_name: str,
    n_subsample: int,
    model_name: str,
    seed: int = 42,
) -> dict:
    """Train one model on a specific subsample size. Returns metric + timing."""
    os.environ["XRFM_CACHE_DIR"] = DATA_DIR
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

    from src.datasets import load_dataset
    from src.experiments import run_experiment
    from src.metrics import compute_metrics, primary_metric_for_task
    from src.models import make_model
    from src.preprocessing import make_splits

    ds = load_dataset(dataset_name)
    # First split off a fixed test set (20% of full, stratified)
    # Then subsample training portion
    print(f"Full dataset: n={ds.n}, subsampling to n_train≈{n_subsample}")

    # Do the usual full split first
    full_split = make_splits(
        ds.X, ds.y,
        numerical_cols=ds.numerical_cols,
        categorical_cols=ds.categorical_cols,
        task=ds.task, n_classes=ds.n_classes,
        random_state=seed,
    )

    # Subsample training set
    n_available = len(full_split.y_train)
    if n_subsample > n_available:
        n_subsample = n_available
    if n_subsample == n_available:
        idx = np.arange(n_available)
    elif ds.task in {"binary", "multiclass"}:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_subsample, random_state=seed)
        idx, _ = next(sss.split(full_split.X_train_enc, full_split.y_train))
    else:
        ss = ShuffleSplit(n_splits=1, train_size=n_subsample, random_state=seed)
        idx, _ = next(ss.split(full_split.X_train_enc))
    idx = np.asarray(idx)

    # Build a reduced split (keep val and test as full)
    from src.preprocessing import SplitData
    reduced = SplitData(
        X_train=full_split.X_train.iloc[idx].reset_index(drop=True),
        y_train=full_split.y_train[idx],
        X_val=full_split.X_val, y_val=full_split.y_val,
        X_test=full_split.X_test, y_test=full_split.y_test,
        X_train_enc=full_split.X_train_enc[idx],
        X_val_enc=full_split.X_val_enc,
        X_test_enc=full_split.X_test_enc,
        X_train_ord=full_split.X_train_ord[idx],
        X_val_ord=full_split.X_val_ord,
        X_test_ord=full_split.X_test_ord,
        numerical_idx_enc=full_split.numerical_idx_enc,
        categorical_idx_enc=full_split.categorical_idx_enc,
        categorical_cols=full_split.categorical_cols,
        numerical_cols=full_split.numerical_cols,
        categorical_idx_ord=full_split.categorical_idx_ord,
        task=full_split.task,
        n_classes=full_split.n_classes,
        feature_names_enc=full_split.feature_names_enc,
        feature_names_ord=full_split.feature_names_ord,
    )

    # Fixed reasonable HPs (no HP tuning on subsamples to keep scaling fair)
    default_hp = {
        "xrfm": {"bandwidth": 10.0, "exponent": 1.0, "kernel": "l2", "reg": 1e-3,
                 "diag": False, "iters": 5, "max_leaf_size": 60000},
        "xgboost": {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1,
                    "early_stopping_rounds": 50},
        "random_forest": {"n_estimators": 300, "max_depth": 20,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "max_features": "sqrt"},
        "catboost": {"iterations": 1000, "depth": 6, "learning_rate": 0.1,
                     "l2_leaf_reg": 3.0, "bagging_temperature": 1.0,
                     "border_count": 128, "early_stopping_rounds": 50},
        "tabpfn": {},
    }[model_name]

    model = make_model(model_name, default_hp, ds.task, ds.n_classes, device="cuda")
    try:
        model.fit(reduced)
        res = model.predict(reduced)
        metrics = compute_metrics(reduced.y_test, res.y_pred, res.y_proba, ds.task)
        out = {
            "dataset": dataset_name,
            "model": model_name,
            "n_subsample": int(n_subsample),
            "n_train_actual": int(len(reduced.y_train)),
            "n_test": int(len(reduced.y_test)),
            "train_time_s": float(res.train_time_s),
            "inference_time_s_per_sample": float(res.inference_time_s_per_sample),
            "test_metrics": metrics,
            "primary_metric": primary_metric_for_task(ds.task),
        }
    except Exception as e:
        out = {
            "dataset": dataset_name,
            "model": model_name,
            "n_subsample": int(n_subsample),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    out_path = Path(RESULTS_DIR) / f"scaling_{dataset_name}_{model_name}_n{n_subsample}_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(f"Saved -> {out_path}")
    return out


@app.local_entrypoint()
def run_scaling(dataset: str = "ida2016", seed: int = 42):
    """Run the scaling experiment: fixed test set, varying train size.
    IDA2016 has ~36000 training samples (60% of 60000) after a 60/20/20 split.
    """
    sizes = [1000, 2500, 5000, 10000, 20000, 36000]
    models = ["xrfm", "xgboost", "random_forest", "catboost"]

    jobs = []
    for n in sizes:
        for m in models:
            jobs.append((n, m))

    print(f"Launching {len(jobs)} scaling experiments in parallel...")
    futures = [
        run_scaling_point.spawn(
            dataset_name=dataset, n_subsample=n, model_name=m, seed=seed
        )
        for (n, m) in jobs
    ]
    results = []
    for (n, m), fut in zip(jobs, futures):
        try:
            r = fut.get()
            tt = r.get("train_time_s", "ERR")
            met = r.get("test_metrics", {})
            print(f"  {m:15s} n={n:6d}  train={tt:.2f}s  metric={met}"
                  if isinstance(tt, float) else f"  {m:15s} n={n:6d}  ERR={r.get('error')}")
        except Exception as e:
            print(f"  {m:15s} n={n:6d}  FAILED: {e}")
            r = {"dataset": dataset, "model": m, "n_subsample": n, "error": str(e)}
        results.append(r)

    out = Path("results/scaling_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out}")
    return results


@app.local_entrypoint()
def interpretability(dataset: str = "appliances_energy", seed: int = 42):
    """Entry point: run interpretability analysis and download result."""
    print(f"Running interpretability on {dataset}...")
    result = run_interpretability.remote(dataset_name=dataset, seed=seed)
    out = Path(f"results/interpretability_{dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved -> {out}")
    return result


# =========== Download results to local ===========

@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=600)
def list_results() -> list:
    from pathlib import Path
    return sorted(str(p.relative_to(RESULTS_DIR)) for p in Path(RESULTS_DIR).rglob("*.json"))


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=600)
def read_result(path: str) -> dict:
    with (Path(RESULTS_DIR) / path).open() as f:
        return json.load(f)


@app.local_entrypoint()
def download_results(output_dir: str = "results"):
    """Download all JSON results from the Modal volume to local."""
    paths = list_results.remote()
    print(f"Found {len(paths)} result files.")
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for p in paths:
        data = read_result.remote(p)
        local = outp / p.replace("/", "_")
        with local.open("w") as f:
            json.dump(data, f, indent=2, default=str)
    print(f"Downloaded to {output_dir}/")

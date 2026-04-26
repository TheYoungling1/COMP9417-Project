"""Experiment runner with Optuna hyperparameter tuning."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import optuna

from .datasets import TabularDataset
from .metrics import compute_metrics, direction_for_metric, primary_metric_for_task
from .models import MODEL_REGISTRY, make_model
from .preprocessing import SplitData, make_splits


# quiet optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_xrfm_hp(trial: optuna.Trial, task: str, n: int) -> dict:
    """xRFM HP space from Table A.1 of the xRFM paper (TALENT ranges adapted)."""
    hp = {
        "bandwidth": trial.suggest_float("bandwidth", 1.0, 200.0, log=True),
        "exponent": trial.suggest_float("exponent", 0.7, 1.4),
        "kernel": trial.suggest_categorical("kernel", ["l2", "l1"]),
        "reg": trial.suggest_float("reg", 1e-6, 1.0, log=True),
        "diag": trial.suggest_categorical("diag", [True, False]),
        "bandwidth_mode": "constant",
        "iters": 5,
        "M_batch_size": 1000,
        "max_leaf_size": min(60000, n),
    }
    return hp


def suggest_xgb_hp(trial: optuna.Trial, task: str, n: int) -> dict:
    hp = {
        "n_estimators": 2000,  # large, with early stopping
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "early_stopping_rounds": 50,
    }
    return hp


def suggest_rf_hp(trial: optuna.Trial, task: str, n: int) -> dict:
    hp = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    return hp


def suggest_tabpfn_hp(trial: optuna.Trial, task: str, n: int) -> dict:
    # TabPFN has essentially no HPs; we still run 1 trial for consistency.
    return {}


def suggest_catboost_hp(trial: optuna.Trial, task: str, n: int) -> dict:
    hp = {
        "iterations": 2000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "early_stopping_rounds": 50,
    }
    return hp


HP_SUGGESTERS = {
    "xrfm": suggest_xrfm_hp,
    "xgboost": suggest_xgb_hp,
    "random_forest": suggest_rf_hp,
    "catboost": suggest_catboost_hp,
    "tabpfn": suggest_tabpfn_hp,
}


N_TRIALS_DEFAULT = {
    "xrfm": 25,
    "xgboost": 40,
    "random_forest": 20,
    "catboost": 30,
    "tabpfn": 1,
}


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    task: str
    n_train: int
    n_val: int
    n_test: int
    d: int
    primary_metric: str
    test_metrics: dict
    val_primary: float
    train_time_s: float
    inference_time_s_per_sample: float
    best_hp: dict
    n_trials: int
    extra: dict = field(default_factory=dict)
    error: Optional[str] = None


def _objective(
    model_name: str,
    split: SplitData,
    task: str,
    n_classes: Optional[int],
    n_train: int,
    device: str,
) -> Callable[[optuna.Trial], float]:
    suggest = HP_SUGGESTERS[model_name]
    primary = primary_metric_for_task(task)
    direction = direction_for_metric(primary)
    sign = 1.0 if direction == "maximize" else -1.0
    default_val = -np.inf if direction == "maximize" else np.inf

    def objective(trial: optuna.Trial) -> float:
        try:
            hp = suggest(trial, task, n_train)
            m = make_model(model_name, hp, task, n_classes, device=device)
            m.fit(split)
            val_split = _shim_val_as_test(split)
            res = m.predict(val_split)
            metrics = compute_metrics(split.y_val, res.y_pred, res.y_proba, task)
            trial.set_user_attr("hp", hp)
            val = metrics.get(primary)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                trial.set_user_attr("error", f"metric_{primary}_nan_or_missing")
                return default_val
            return float(val)
        except Exception as e:
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            return default_val
    return objective


def _shim_val_as_test(split: SplitData) -> SplitData:
    return SplitData(
        X_train=split.X_train, y_train=split.y_train,
        X_val=split.X_val, y_val=split.y_val,
        X_test=split.X_val, y_test=split.y_val,
        X_train_enc=split.X_train_enc,
        X_val_enc=split.X_val_enc,
        X_test_enc=split.X_val_enc,
        X_train_ord=split.X_train_ord,
        X_val_ord=split.X_val_ord,
        X_test_ord=split.X_val_ord,
        numerical_idx_enc=split.numerical_idx_enc,
        categorical_idx_enc=split.categorical_idx_enc,
        categorical_cols=split.categorical_cols,
        numerical_cols=split.numerical_cols,
        categorical_idx_ord=split.categorical_idx_ord,
        task=split.task,
        n_classes=split.n_classes,
        feature_names_enc=split.feature_names_enc,
        feature_names_ord=split.feature_names_ord,
    )


def run_experiment(
    dataset: TabularDataset,
    model_name: str,
    device: str = "cuda",
    n_trials: Optional[int] = None,
    random_state: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
    timeout_s: Optional[int] = None,
) -> ExperimentResult:
    """Run one (dataset, model) experiment:
    1. Train/val/test split with fixed seed.
    2. Optuna HP search on val.
    3. Retrain best HPs (on train only; val needed for xRFM inner loop and XGB early stop).
    4. Evaluate on held-out test set.
    """
    if n_trials is None:
        n_trials = N_TRIALS_DEFAULT.get(model_name, 20)

    task = dataset.task
    primary = primary_metric_for_task(task)

    split = make_splits(
        dataset.X, dataset.y,
        numerical_cols=dataset.numerical_cols,
        categorical_cols=dataset.categorical_cols,
        task=task, n_classes=dataset.n_classes,
        test_size=test_size, val_size=val_size,
        random_state=random_state,
    )

    direction = direction_for_metric(primary)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    objective_fn = _objective(model_name, split, task, dataset.n_classes, len(split.y_train), device)

    t0_tune = time.time()
    study.optimize(
        objective_fn, n_trials=n_trials, timeout=timeout_s,
        catch=(Exception,), show_progress_bar=False,
    )
    tune_time = time.time() - t0_tune
    best_hp = study.best_trial.user_attrs.get("hp", dict(study.best_trial.params))
    val_primary = study.best_trial.value

    final_model = make_model(model_name, best_hp, task, dataset.n_classes, device=device)
    try:
        final_model.fit(split)
        res = final_model.predict(split)
        test_metrics = compute_metrics(split.y_test, res.y_pred, res.y_proba, task)
        result = ExperimentResult(
            dataset=dataset.name,
            model=model_name,
            task=task,
            n_train=len(split.y_train),
            n_val=len(split.y_val),
            n_test=len(split.y_test),
            d=dataset.d,
            primary_metric=primary,
            test_metrics=test_metrics,
            val_primary=float(val_primary),
            train_time_s=float(res.train_time_s),
            inference_time_s_per_sample=float(res.inference_time_s_per_sample),
            best_hp=best_hp,
            n_trials=n_trials,
            extra={
                "tune_time_s": tune_time,
                "model_extra": res.extra,
            },
        )
        return result
    except Exception as e:
        return ExperimentResult(
            dataset=dataset.name, model=model_name, task=task,
            n_train=len(split.y_train), n_val=len(split.y_val), n_test=len(split.y_test),
            d=dataset.d, primary_metric=primary, test_metrics={},
            val_primary=float(val_primary) if val_primary is not None else float("nan"),
            train_time_s=float("nan"),
            inference_time_s_per_sample=float("nan"),
            best_hp=best_hp, n_trials=n_trials,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, mean_squared_error,
                              roc_auc_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None, task: str) -> dict:
    m = {}
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        m["rmse"] = rmse
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        m["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        if y_proba is not None:
            try:
                if task == "binary":
                    p1 = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    m["auc_roc"] = float(roc_auc_score(y_true, p1))
                else:
                    m["auc_roc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    ))
            except Exception as e:
                m["auc_roc"] = float("nan")
                m["auc_roc_error"] = str(e)
    return m


def primary_metric_for_task(task: str) -> str:
    return {
        "regression": "rmse",
        "binary": "auc_roc",
        "multiclass": "accuracy",
    }[task]


def direction_for_metric(metric: str) -> str:
    return {
        "rmse": "minimize",
        "r2": "maximize",
        "accuracy": "maximize",
        "auc_roc": "maximize",
    }[metric]

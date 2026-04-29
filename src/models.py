"""Unified model wrappers for the xRFM comparison benchmark.

All wrappers expose:
    .fit(split: SplitData) -> None
    .predict(X_ord, X_enc) -> np.ndarray  (regression) or class labels
    .predict_proba(X_ord, X_enc) -> np.ndarray (classification)
    .train_time_s, .inference_time_s
    .model  (underlying fitted model)
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .preprocessing import SplitData


@dataclass
class ModelResult:
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    train_time_s: float
    inference_time_s_per_sample: float
    n_test: int
    extra: dict = field(default_factory=dict)


class XRFMWrapper:
    """xRFM (Beaglehole et al. 2025). Tree-partitioned RFM with AGOP feature learning."""

    name = "xRFM"

    def __init__(self, hp: dict, task: str, n_classes: Optional[int], device: str = "cuda"):
        self.hp = dict(hp)
        self.task = task
        self.n_classes = n_classes
        self.device = device
        self.model = None
        self.train_time_s = 0.0
        self.n_leaves = None
        self.actual_max_leaf_size = None

    def _make_y(self, y: np.ndarray, is_classification: bool) -> np.ndarray:
        if is_classification:
            # xRFM expects one-hot (regression-style) targets for classification
            n_classes = int(self.n_classes)
            y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
            y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
            return y_onehot
        else:
            return y.astype(np.float32).reshape(-1, 1)

    def _count_leaves(self, node: dict) -> int:
        if node.get("type") == "leaf":
            return 1
        return self._count_leaves(node["left"]) + self._count_leaves(node["right"])

    def _sync_device(self, device: Any) -> None:
        if getattr(device, "type", None) == "cuda":
            import torch
            torch.cuda.synchronize(device)

    def _extra(self) -> dict:
        return {
            "n_leaves": self.n_leaves,
            "actual_max_leaf_size": self.actual_max_leaf_size,
        }

    def fit(self, split: SplitData) -> None:
        import torch
        from xrfm import xRFM

        is_classification = split.task in {"binary", "multiclass"}
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        X_train = torch.tensor(split.X_train_enc, dtype=torch.float32, device=device)
        X_val = torch.tensor(split.X_val_enc, dtype=torch.float32, device=device)
        y_train = torch.tensor(self._make_y(split.y_train, is_classification), dtype=torch.float32, device=device)
        y_val = torch.tensor(self._make_y(split.y_val, is_classification), dtype=torch.float32, device=device)

        rfm_params = {
            "model": {
                "kernel": self.hp.get("kernel", "l2"),
                "bandwidth": self.hp.get("bandwidth", 10.0),
                "exponent": self.hp.get("exponent", 1.0),
                "diag": self.hp.get("diag", False),
                "bandwidth_mode": self.hp.get("bandwidth_mode", "constant"),
            },
            "fit": {
                "reg": self.hp.get("reg", 1e-3),
                "iters": self.hp.get("iters", 5),
                "M_batch_size": self.hp.get("M_batch_size", 1000),
                "verbose": False,
                "early_stop_rfm": True,
            },
        }
        max_leaf_size = self.hp.get("max_leaf_size", 60000)
        tuning_metric = self.hp.get("tuning_metric")
        if tuning_metric is None:
            tuning_metric = "mse" if split.task == "regression" else ("auc" if split.task == "binary" else "accuracy")

        self.model = xRFM(
            rfm_params=rfm_params,
            max_leaf_size=max_leaf_size,
            device=device,
            tuning_metric=tuning_metric,
            n_trees=1,
            verbose=False,
            random_state=self.hp.get("random_state", 42),
        )
        self._sync_device(device)
        t0 = time.time()
        self.model.fit(X_train, y_train, X_val, y_val)
        self._sync_device(device)
        self.train_time_s = time.time() - t0
        trees = self.model.trees or []
        self.n_leaves = sum(self._count_leaves(tree) for tree in trees)
        self.actual_max_leaf_size = getattr(self.model, "max_leaf_size", None)

    def predict(self, split: SplitData) -> ModelResult:
        import torch
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        X_test = torch.tensor(split.X_test_enc, dtype=torch.float32, device=device)

        if split.task == "regression":
            self._sync_device(device)
            t0 = time.time()
            y_pred_raw = self.model.predict(X_test)
            if isinstance(y_pred_raw, torch.Tensor):
                y_pred_raw = y_pred_raw.detach().cpu().numpy()
            inf = time.time() - t0
            return ModelResult(
                y_pred=np.asarray(y_pred_raw).reshape(-1),
                y_proba=None,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
                extra=self._extra(),
            )
        else:
            self._sync_device(device)
            t0 = time.time()
            proba = self.model.predict_proba(X_test)
            if isinstance(proba, torch.Tensor):
                proba = proba.detach().cpu().numpy()
            proba = np.asarray(proba)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            # Some xRFM versions may still return 1-column for binary — normalise
            if proba.shape[1] == 1 and self.n_classes == 2:
                p1 = proba.reshape(-1)
                if p1.min() < 0 or p1.max() > 1:
                    p1 = 1.0 / (1.0 + np.exp(-p1))
                proba = np.stack([1.0 - p1, p1], axis=1)
            y_pred = proba.argmax(axis=1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred,
                y_proba=proba,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
                extra=self._extra(),
            )


class XGBoostWrapper:
    name = "XGBoost"

    def __init__(self, hp: dict, task: str, n_classes: Optional[int]):
        self.hp = dict(hp)
        self.task = task
        self.n_classes = n_classes
        self.model = None
        self.train_time_s = 0.0

    def fit(self, split: SplitData) -> None:
        import xgboost as xgb
        params = dict(
            n_estimators=self.hp.get("n_estimators", 1000),
            max_depth=self.hp.get("max_depth", 6),
            learning_rate=self.hp.get("learning_rate", 0.1),
            subsample=self.hp.get("subsample", 0.8),
            colsample_bytree=self.hp.get("colsample_bytree", 0.8),
            reg_alpha=self.hp.get("reg_alpha", 0.0),
            reg_lambda=self.hp.get("reg_lambda", 1.0),
            min_child_weight=self.hp.get("min_child_weight", 1),
            random_state=self.hp.get("random_state", 42),
            tree_method=self.hp.get("tree_method", "hist"),
            device=self.hp.get("device", "cpu"),
            verbosity=0,
            early_stopping_rounds=self.hp.get("early_stopping_rounds", 50),
            n_jobs=-1,
        )
        if split.task == "regression":
            model = xgb.XGBRegressor(**params, objective="reg:squarederror")
        elif split.task == "binary":
            model = xgb.XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss")
        else:
            model = xgb.XGBClassifier(**params, objective="multi:softprob", eval_metric="mlogloss")

        t0 = time.time()
        model.fit(
            split.X_train_ord, split.y_train,
            eval_set=[(split.X_val_ord, split.y_val)],
            verbose=False,
        )
        self.train_time_s = time.time() - t0
        self.model = model

    def predict(self, split: SplitData) -> ModelResult:
        t0 = time.time()
        if split.task == "regression":
            y_pred = self.model.predict(split.X_test_ord)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=None,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )
        else:
            proba = self.model.predict_proba(split.X_test_ord)
            y_pred = proba.argmax(axis=1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=proba,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )


class RandomForestWrapper:
    name = "RandomForest"

    def __init__(self, hp: dict, task: str, n_classes: Optional[int]):
        self.hp = dict(hp)
        self.task = task
        self.n_classes = n_classes
        self.model = None
        self.train_time_s = 0.0

    def fit(self, split: SplitData) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params = dict(
            n_estimators=self.hp.get("n_estimators", 500),
            max_depth=self.hp.get("max_depth", None),
            min_samples_split=self.hp.get("min_samples_split", 2),
            min_samples_leaf=self.hp.get("min_samples_leaf", 1),
            max_features=self.hp.get("max_features", "sqrt"),
            random_state=self.hp.get("random_state", 42),
            n_jobs=-1,
        )
        if split.task == "regression":
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)

        t0 = time.time()
        # RF doesn't use val
        model.fit(split.X_train_ord, split.y_train)
        self.train_time_s = time.time() - t0
        self.model = model

    def predict(self, split: SplitData) -> ModelResult:
        t0 = time.time()
        if split.task == "regression":
            y_pred = self.model.predict(split.X_test_ord)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=None,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )
        else:
            proba = self.model.predict_proba(split.X_test_ord)
            y_pred = proba.argmax(axis=1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=proba,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )


class CatBoostWrapper:
    name = "CatBoost"

    def __init__(self, hp: dict, task: str, n_classes: Optional[int]):
        self.hp = dict(hp)
        self.task = task
        self.n_classes = n_classes
        self.model = None
        self.train_time_s = 0.0

    def fit(self, split: SplitData) -> None:
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(
            iterations=self.hp.get("iterations", 2000),
            depth=self.hp.get("depth", 6),
            learning_rate=self.hp.get("learning_rate", 0.1),
            l2_leaf_reg=self.hp.get("l2_leaf_reg", 3.0),
            bagging_temperature=self.hp.get("bagging_temperature", 1.0),
            border_count=self.hp.get("border_count", 128),
            random_seed=self.hp.get("random_state", 42),
            verbose=False,
            early_stopping_rounds=self.hp.get("early_stopping_rounds", 50),
            allow_writing_files=False,
        )
        cat_features = split.categorical_idx_ord if split.categorical_cols else None
        # CatBoost needs object dtype for mixed num+str columns
        import numpy as np
        import pandas as pd
        if cat_features:
            X_train = pd.DataFrame(split.X_train_ord.copy())
            X_val = pd.DataFrame(split.X_val_ord.copy())
            for i in cat_features:
                X_train[i] = X_train[i].astype(int).astype(str)
                X_val[i] = X_val[i].astype(int).astype(str)
        else:
            X_train = split.X_train_ord
            X_val = split.X_val_ord
        if split.task == "regression":
            model = CatBoostRegressor(**params, loss_function="RMSE")
        elif split.task == "binary":
            model = CatBoostClassifier(**params, loss_function="Logloss", eval_metric="AUC")
        else:
            model = CatBoostClassifier(**params, loss_function="MultiClass",
                                       eval_metric="MultiClass")
        t0 = time.time()
        model.fit(X_train, split.y_train,
                  eval_set=(X_val, split.y_val),
                  cat_features=cat_features,
                  verbose=False)
        self.train_time_s = time.time() - t0
        self.model = model
        self.cat_features = cat_features

    def predict(self, split: SplitData) -> ModelResult:
        import numpy as np
        import pandas as pd
        if self.cat_features:
            X_test = pd.DataFrame(split.X_test_ord.copy())
            for i in self.cat_features:
                X_test[i] = X_test[i].astype(int).astype(str)
        else:
            X_test = split.X_test_ord
        t0 = time.time()
        if split.task == "regression":
            y_pred = self.model.predict(X_test)
            y_pred = np.asarray(y_pred).reshape(-1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=None,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )
        else:
            proba = self.model.predict_proba(X_test)
            y_pred = proba.argmax(axis=1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=proba,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
            )


class TabPFNWrapper:
    """TabPFN v2 (Hollmann et al., Nature 2025). Pretrained transformer.
    Scales to ~10k samples / 500 features natively; larger data must be subsampled.

    Requires TABPFN_TOKEN env var (https://ux.priorlabs.ai) — without it the
    .fit() call raises TabPFNLicenseError. Not exercised in our reported runs;
    every result file under results/downloads/main_*_tabpfn_*.json is an error
    record from a license-gated run.
    """

    name = "TabPFN-v2"
    MAX_N_FIT = 10000  # v2 training context cap

    def __init__(self, hp: dict, task: str, n_classes: Optional[int], device: str = "cuda"):
        self.hp = dict(hp)
        self.task = task
        self.n_classes = n_classes
        self.device = device
        self.model = None
        self.train_time_s = 0.0
        self.subsampled_n = None

    def fit(self, split: SplitData) -> None:
        import torch
        is_cls = split.task in {"binary", "multiclass"}
        device_str = self.device if torch.cuda.is_available() else "cpu"

        X_train = split.X_train_enc
        y_train = split.y_train
        if len(X_train) > self.MAX_N_FIT:
            rng = np.random.RandomState(self.hp.get("random_state", 42))
            if is_cls:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=self.MAX_N_FIT,
                                             random_state=self.hp.get("random_state", 42))
                idx, _ = next(sss.split(X_train, y_train))
            else:
                idx = rng.choice(len(X_train), self.MAX_N_FIT, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            self.subsampled_n = self.MAX_N_FIT

        if is_cls:
            from tabpfn import TabPFNClassifier
            model = TabPFNClassifier(device=device_str, ignore_pretraining_limits=True)
        else:
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor(device=device_str, ignore_pretraining_limits=True)

        t0 = time.time()
        model.fit(X_train, y_train)
        self.train_time_s = time.time() - t0
        self.model = model

    def predict(self, split: SplitData) -> ModelResult:
        t0 = time.time()
        if split.task == "regression":
            y_pred = self.model.predict(split.X_test_enc)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=None,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
                extra={"subsampled_n": self.subsampled_n},
            )
        else:
            proba = self.model.predict_proba(split.X_test_enc)
            y_pred = proba.argmax(axis=1)
            inf = time.time() - t0
            return ModelResult(
                y_pred=y_pred, y_proba=proba,
                train_time_s=self.train_time_s,
                inference_time_s_per_sample=inf / len(split.y_test),
                n_test=len(split.y_test),
                extra={"subsampled_n": self.subsampled_n},
            )


MODEL_REGISTRY = {
    "xrfm": XRFMWrapper,
    "xgboost": XGBoostWrapper,
    "random_forest": RandomForestWrapper,
    "catboost": CatBoostWrapper,
    "tabpfn": TabPFNWrapper,
}


def make_model(name: str, hp: dict, task: str, n_classes: Optional[int], device: str = "cuda"):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if name in {"xrfm", "tabpfn"}:
        return cls(hp=hp, task=task, n_classes=n_classes, device=device)
    else:
        return cls(hp=hp, task=task, n_classes=n_classes)

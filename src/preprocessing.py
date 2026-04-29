"""Preprocessing utilities: train/val/test splits, scaling, encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass
class SplitData:
    """Standardized train/val/test arrays, ready for any model.

    Three views of the same split: raw (pandas, keeps categorical dtype),
    `_enc` (scaled numerics + one-hot, for kernels/MLPs), and `_ord`
    (ordinal-encoded categoricals, for tree ensembles).
    """
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    X_train_enc: np.ndarray
    X_val_enc: np.ndarray
    X_test_enc: np.ndarray
    X_train_ord: np.ndarray
    X_val_ord: np.ndarray
    X_test_ord: np.ndarray
    numerical_idx_enc: list[int]
    categorical_idx_enc: list[int]
    categorical_cols: list[str]
    numerical_cols: list[str]
    categorical_idx_ord: list[int]
    task: str
    n_classes: Optional[int]
    feature_names_enc: list[str]
    feature_names_ord: list[str]


def make_splits(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    numerical_cols: list[str],
    categorical_cols: list[str],
    task: str,
    n_classes: Optional[int],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Split → impute → encode. Imputers and encoders are fit on train only;
    val/test go through transform."""
    y = np.asarray(y)
    stratify_y = y if task in {"binary", "multiclass"} else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    stratify_tv = y_trainval if task in {"binary", "multiclass"} else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=stratify_tv,
    )

    return preprocess_existing_splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        task=task,
        n_classes=n_classes,
    )


def preprocess_existing_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    numerical_cols: list[str],
    categorical_cols: list[str],
    task: str,
    n_classes: Optional[int],
) -> SplitData:
    """Impute and encode already chosen train/val/test partitions.

    Imputers, scalers, and encoders are fit on the provided train partition
    only. This is useful for scaling studies where each subsample size should
    have its own preprocessing fit rather than borrowing full-train statistics.
    """
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # impute: median for numerics, mode for categoricals
    num_imputer = SimpleImputer(strategy="median") if numerical_cols else None
    cat_imputer = SimpleImputer(strategy="most_frequent") if categorical_cols else None

    def _impute(X_df: pd.DataFrame) -> pd.DataFrame:
        out = X_df.copy()
        if numerical_cols and num_imputer is not None:
            out[numerical_cols] = num_imputer.transform(X_df[numerical_cols])
        if categorical_cols and cat_imputer is not None:
            out[categorical_cols] = cat_imputer.transform(X_df[categorical_cols].astype(str))
        return out

    if numerical_cols:
        num_imputer.fit(X_train[numerical_cols])
    if categorical_cols:
        cat_imputer.fit(X_train[categorical_cols].astype(str))

    X_train = _impute(X_train)
    X_val = _impute(X_val)
    X_test = _impute(X_test)

    # one-hot for kernel inputs, ordinal for tree inputs
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    num_train = scaler.fit_transform(X_train[numerical_cols].astype(float)) if numerical_cols else np.empty((len(X_train), 0))
    num_val = scaler.transform(X_val[numerical_cols].astype(float)) if numerical_cols else np.empty((len(X_val), 0))
    num_test = scaler.transform(X_test[numerical_cols].astype(float)) if numerical_cols else np.empty((len(X_test), 0))

    if categorical_cols:
        ohe.fit(X_train[categorical_cols].astype(str))
        cat_train = ohe.transform(X_train[categorical_cols].astype(str))
        cat_val = ohe.transform(X_val[categorical_cols].astype(str))
        cat_test = ohe.transform(X_test[categorical_cols].astype(str))
        ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    else:
        cat_train = np.empty((len(X_train), 0))
        cat_val = np.empty((len(X_val), 0))
        cat_test = np.empty((len(X_test), 0))
        ohe_names = []

    X_train_enc = np.concatenate([num_train, cat_train], axis=1).astype(np.float32)
    X_val_enc = np.concatenate([num_val, cat_val], axis=1).astype(np.float32)
    X_test_enc = np.concatenate([num_test, cat_test], axis=1).astype(np.float32)
    numerical_idx_enc = list(range(len(numerical_cols)))
    categorical_idx_enc = list(range(len(numerical_cols), len(numerical_cols) + cat_train.shape[1]))
    feature_names_enc = list(numerical_cols) + ohe_names

    # ordinal path for tree models
    if categorical_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        oe.fit(X_train[categorical_cols].astype(str))
        cat_train_ord = oe.transform(X_train[categorical_cols].astype(str))
        cat_val_ord = oe.transform(X_val[categorical_cols].astype(str))
        cat_test_ord = oe.transform(X_test[categorical_cols].astype(str))
    else:
        cat_train_ord = np.empty((len(X_train), 0))
        cat_val_ord = np.empty((len(X_val), 0))
        cat_test_ord = np.empty((len(X_test), 0))
    X_train_ord = np.concatenate(
        [X_train[numerical_cols].astype(float).to_numpy(), cat_train_ord], axis=1
    ).astype(np.float32)
    X_val_ord = np.concatenate(
        [X_val[numerical_cols].astype(float).to_numpy(), cat_val_ord], axis=1
    ).astype(np.float32)
    X_test_ord = np.concatenate(
        [X_test[numerical_cols].astype(float).to_numpy(), cat_test_ord], axis=1
    ).astype(np.float32)
    categorical_idx_ord = list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols)))
    feature_names_ord = list(numerical_cols) + list(categorical_cols)

    return SplitData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_train_enc=X_train_enc,
        X_val_enc=X_val_enc,
        X_test_enc=X_test_enc,
        X_train_ord=X_train_ord,
        X_val_ord=X_val_ord,
        X_test_ord=X_test_ord,
        numerical_idx_enc=numerical_idx_enc,
        categorical_idx_enc=categorical_idx_enc,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        categorical_idx_ord=categorical_idx_ord,
        task=task,
        n_classes=n_classes,
        feature_names_enc=feature_names_enc,
        feature_names_ord=feature_names_ord,
    )

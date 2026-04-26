"""
Dataset loaders for the xRFM comparison benchmark.

All 5 datasets are from UCI ML Repository and have been verified absent from
the TALENT (Ye et al. 2024) benchmark and the xRFM paper's meta-test benchmark.

Constraints met:
  - Regression: Seoul Bike (8760), Appliances Energy (19735)
  - Classification: HCC (165, binary), IDA2016 (60000, binary), Crop (325k, multiclass)
  - n > 10000: Appliances, IDA2016, Crop
  - d > 50: IDA2016 (171), Crop (174)
  - Mixed feature types: Seoul Bike, HCC Survival
  - Interpretability dataset: Appliances (has 2 ground-truth random features rv1, rv2)
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CACHE_DIR = Path(os.environ.get("XRFM_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "data_cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TabularDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task: str  # "regression", "binary", "multiclass"
    numerical_cols: list[str]
    categorical_cols: list[str]
    description: str
    n_classes: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        return (
            f"{self.name}: n={self.n}, d={self.d}, task={self.task}, "
            f"num={len(self.numerical_cols)}, cat={len(self.categorical_cols)}"
            + (f", classes={self.n_classes}" if self.n_classes else "")
        )


def _download(url: str, filename: str, *, quiet: bool = False) -> Path:
    path = CACHE_DIR / filename
    if not path.exists():
        if not quiet:
            print(f"[download] {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def _unzip(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)
    return target_dir


def load_seoul_bike() -> TabularDataset:
    """UCI 560: Seoul Bike Sharing Demand, n=8760, regression."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
    path = _download(url, "seoul_bike.csv")
    df = pd.read_csv(path, encoding="latin-1")
    df = df.drop(columns=["Date"])
    rename = {c: c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("°", "").replace("%", "pct") for c in df.columns}
    df = df.rename(columns=rename)
    target_col = [c for c in df.columns if "Rented_Bike_Count" in c][0]
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    categorical_cols = ["Seasons", "Holiday", "Functioning_Day"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    X[categorical_cols] = X[categorical_cols].astype(str).astype("category")
    return TabularDataset(
        name="seoul_bike",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task="regression",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        description="Hourly bike rental demand in Seoul with weather and temporal features.",
    )


def load_appliances_energy() -> TabularDataset:
    """UCI 374: Appliances Energy Prediction, n=19735, regression.
    Contains two ground-truth random features rv1, rv2 — perfect for feature importance tests.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
    path = _download(url, "appliances_energy.csv")
    df = pd.read_csv(path)
    df = df.drop(columns=["date"])
    y = df["Appliances"].astype(float)
    X = df.drop(columns=["Appliances"])
    numerical_cols = list(X.columns)
    categorical_cols: list[str] = []
    return TabularDataset(
        name="appliances_energy",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task="regression",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        description="Belgian house appliances energy consumption (Wh) from indoor/outdoor sensors. Includes two random noise features rv1 and rv2.",
        extra={"random_features": ["rv1", "rv2"]},
    )


def load_hcc_survival() -> TabularDataset:
    """UCI 423: HCC Survival (Hepatocellular Carcinoma), n=165, binary.
    Small-n mixed-type dataset with clinical features."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00423/hcc-survival.zip"
    zip_path = _download(url, "hcc_survival.zip")
    extract_dir = CACHE_DIR / "hcc_survival"
    if not extract_dir.exists():
        _unzip(zip_path, extract_dir)
    data_files = list(extract_dir.rglob("*.csv")) + list(extract_dir.rglob("*.txt"))
    data_file = None
    for f in data_files:
        if "hcc-data" in f.name.lower() or "hcc_data" in f.name.lower():
            data_file = f
            break
    if data_file is None:
        # fallback: first sizable data file
        for f in sorted(extract_dir.rglob("*")):
            if f.suffix in {".csv", ".data", ".txt"} and f.stat().st_size > 1000:
                data_file = f
                break
    assert data_file is not None, f"No data file in {extract_dir}"
    # 49 features + 1 target, missing values coded as "?"
    df = pd.read_csv(data_file, header=None, na_values=["?"])
    target_col = df.columns[-1]
    y = df[target_col].astype(int)  # 0 = dies, 1 = lives
    X = df.drop(columns=[target_col])
    # UCI docs say 23-48 are quantitative and 0-22 are categorical/binary, but we
    # detect from cardinality + value range to stay robust.
    categorical_cols, numerical_cols = [], []
    for c in X.columns:
        uniq = X[c].dropna().unique()
        if len(uniq) <= 10 and set(uniq).issubset(set(range(11))):
            categorical_cols.append(c)
        else:
            numerical_cols.append(c)
    X.columns = [f"x{c}" for c in X.columns]
    num_str = [f"x{c}" for c in numerical_cols]
    cat_str = [f"x{c}" for c in categorical_cols]
    X[cat_str] = X[cat_str].astype("category")
    return TabularDataset(
        name="hcc_survival",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task="binary",
        numerical_cols=num_str,
        categorical_cols=cat_str,
        description="Hepatocellular carcinoma 1-year survival prediction (Portuguese cohort). 49 mixed features, heavy missingness.",
        n_classes=2,
    )


def load_ida2016() -> TabularDataset:
    """UCI 414: IDA2016 Scania Truck APS Failures, n=60000, binary.
    Highly imbalanced (~1.6% positive). Asymmetric cost: 10*FN + 500*FP."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00414/to_uci.zip"
    zip_path = _download(url, "ida2016.zip")
    extract_dir = CACHE_DIR / "ida2016"
    if not extract_dir.exists():
        _unzip(zip_path, extract_dir)
    # There's a nested zip
    inner_zip = next((f for f in extract_dir.rglob("*.zip")), None)
    if inner_zip is not None:
        _unzip(inner_zip, extract_dir)
    train_file = None
    for f in extract_dir.rglob("*.csv"):
        if "train" in f.name.lower():
            train_file = f
            break
    assert train_file is not None, f"No training CSV in {extract_dir}"
    df = pd.read_csv(train_file, skiprows=20, na_values=["na", "NA", "?"])
    y = (df["class"] == "pos").astype(int)
    X = df.drop(columns=["class"])
    numerical_cols = list(X.columns)
    categorical_cols: list[str] = []
    return TabularDataset(
        name="ida2016",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task="binary",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        description="Scania heavy truck Air Pressure System failure prediction (2016 IDA Industrial Challenge). Severe class imbalance.",
        n_classes=2,
    )


def load_crop_mapping(n_max: Optional[int] = None, random_state: int = 42) -> TabularDataset:
    """UCI 525: Crop Mapping using Fused Optical-Radar Data, n=325834, multiclass.
    7 crop types. 174 numeric satellite features (SAR + optical).

    n_max: optionally subsample to this size (stratified) for feasibility.
    """
    url = "https://archive.ics.uci.edu/static/public/525/crop+mapping+using+fused+optical+radar+data+set.zip"
    zip_path = _download(url, "crop_mapping.zip")
    extract_dir = CACHE_DIR / "crop_mapping"
    data_file = None
    for f in extract_dir.rglob("*"):
        if f.is_file() and f.stat().st_size > 10_000_000 and f.suffix.lower() in {".csv", ".txt"}:
            data_file = f
            break
    if data_file is None:
        _unzip(zip_path, extract_dir)
        for f in extract_dir.rglob("*"):
            if f.is_file() and f.stat().st_size > 10_000_000 and f.suffix.lower() in {".csv", ".txt"}:
                data_file = f
                break
    assert data_file is not None, f"No large data file in {extract_dir}"
    # .txt or .csv, comma-separated either way
    df = pd.read_csv(data_file)
    label_col = None
    for c in df.columns:
        if c.lower() in {"label", "class", "target", "y"}:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[0]
    y_raw = df[label_col]
    classes = sorted(y_raw.unique())
    class_map = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_map).astype(int)
    X = df.drop(columns=[label_col])
    numerical_cols = list(X.columns)
    if n_max is not None and len(X) > n_max:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=random_state)
        idx, _ = next(sss.split(X, y))
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    return TabularDataset(
        name="crop_mapping",
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task="multiclass",
        numerical_cols=numerical_cols,
        categorical_cols=[],
        description="Crop classification from fused RADARSAT-2 SAR + Landsat-8 optical satellite features (Winnipeg 2015-2016).",
        n_classes=len(classes),
        extra={"class_map": class_map},
    )


DATASET_LOADERS = {
    "seoul_bike": load_seoul_bike,
    "appliances_energy": load_appliances_energy,
    "hcc_survival": load_hcc_survival,
    "ida2016": load_ida2016,
    "crop_mapping": load_crop_mapping,
}


def load_dataset(name: str) -> TabularDataset:
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name]()


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(DATASET_LOADERS)
    for name in names:
        ds = load_dataset(name)
        print(ds.summary())
        print(f"  X dtypes: {dict(ds.X.dtypes.astype(str))}")
        print(f"  y stats: min={ds.y.min()}, max={ds.y.max()}, unique={ds.y.nunique()}")
        print(f"  NaNs: {ds.X.isna().sum().sum()}")
        print()

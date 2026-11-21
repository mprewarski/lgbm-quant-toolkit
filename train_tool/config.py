"""Configuration for the P123-style training tool."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelSpec:
    """One LightGBM model: a name and its hyperparameters (P123 'Hyperparams' JSON)."""

    name: str
    params: Dict[str, Any]


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    models: List[ModelSpec]
    target_col: str = "TARGET"
    date_col: str = "Date"
    id_col: str = "P123 ID"
    exclude_columns: List[str] = field(default_factory=lambda: ["P123 ID", "Ticker"])
    categorical_columns: List[str] = field(default_factory=list)
    # Feature selection, for feature testing: `features` = explicit list (None = all
    # remaining columns), `drop_features` = removed after that.
    features: Optional[List[str]] = None
    drop_features: List[str] = field(default_factory=list)

    # Dataset period as set in P123 (Validation > Method > Dataset Period).
    # None = first / last date in the CSV.
    dataset_start: Optional[str] = None
    dataset_end: Optional[str] = None

    # Validation (P123 Rolling Time Series CV)
    validation_method: str = "rolling"  # "rolling" | "expanding" | "holdout"
    folds: int = 4
    training_years: float = 16.0
    gap_weeks: int = 20

    # Preprocessing that P123 applies on top of the downloaded (normalized) data
    # NAs go to the middle in P123: 0 for Z-score data, 0.5 for MinMax data. "median" = the
    # feature's median on that date, "none" = leave NAs to LightGBM.
    na_fill: Union[str, float] = "zero"
    max_na_pct: float = 30.0  # rows with more NA features than this are dropped
    # Clip features and TARGET at +-clip, like P123's Z-score outlier clipping (its "outlier
    # limit"), for datasets downloaded without clipping. None = use values as downloaded.
    clip: Optional[float] = None

    # Runs
    runs: int = 5
    seeds: Optional[List[int]] = None  # None = base_seed, base_seed+1, ...
    base_seed: int = 1
    n_jobs: int = -1

    # Weekly return per stock for P123-style portfolio stats: a dataset column holding the
    # next period's return (e.g. P123 feature Future%Chg(5)); never used as a feature.
    # If P123 normalized it, give the linear mapping back to percent:
    #   return % = (value - returns_zero) / returns_per_pct   (see python -m train_tool.calibrate)
    returns_col: Optional[str] = None
    # Optional separate CSV holding returns_col (joined on date and id), so one returns file
    # can serve every dataset; None = returns_col is a column of data_path.
    returns_path: Optional[str] = None
    returns_zero: float = 0.0
    returns_per_pct: float = 1.0
    slippage_pct: float = 0.1  # charged on every buy and sell, like P123's Slippage setting

    # Results
    quantiles: int = 10  # for report.txt and the CSVs
    report_quantiles: List[int] = field(default_factory=lambda: [10, 20, 50])  # report.html
    html_report: bool = True
    save_predictions: bool = True
    log_level: str = "INFO"


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}  # "_comment" keys are notes
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})

    validation = raw.pop("validation", {}) or {}
    raw.setdefault("validation_method", validation.get("method", "rolling"))
    for key in ("folds", "training_years", "gap_weeks"):
        if key in validation:
            raw.setdefault(key, validation[key])

    prep = raw.pop("preprocessing", {}) or {}
    for key in ("na_fill", "max_na_pct", "clip"):
        if key in prep:
            raw.setdefault(key, prep[key])

    models = [ModelSpec(name=m["name"], params=dict(m.get("params", {}))) for m in raw.pop("models", [])]
    if not models:
        raise ValueError("Config needs at least one entry in 'models'")

    known = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return TrainConfig(models=models, **raw)

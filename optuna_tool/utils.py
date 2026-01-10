import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def optional_import(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - runtime env dependent
        return None, str(exc)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str, level: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, (float, int, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_target(series: pd.Series) -> Tuple[pd.Series, str]:
    if pd.api.types.is_numeric_dtype(series):
        uniques = [v for v in pd.Series(series).dropna().unique()]
        if len(uniques) == 2:
            return series, "binary"
        return series, "numeric"
    encoded, uniques = pd.factorize(series)
    if len(uniques) == 2:
        return pd.Series(encoded, index=series.index), "binary"
    return pd.Series(encoded, index=series.index), "categorical"


def sanitize_columns(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    name_mapping = {}
    if any(re.search(r"[^0-9A-Za-z_]", col) for col in feature_df.columns):
        used = set()
        new_cols = []
        for col in feature_df.columns:
            safe = re.sub(r"[^0-9A-Za-z_]", "_", col)
            safe = re.sub(r"_+", "_", safe).strip("_")
            if not safe:
                safe = "feature"
            base = safe
            counter = 1
            while safe in used:
                safe = f"{base}_{counter}"
                counter += 1
            used.add(safe)
            name_mapping[safe] = col
            new_cols.append(safe)
        feature_df = feature_df.copy()
        feature_df.columns = new_cols
    return feature_df, name_mapping


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    exclude_columns: Optional[List[str]],
    categorical_columns: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    drop_cols = [target_col]
    if date_col and date_col in df.columns:
        drop_cols.append(date_col)
    if exclude_columns:
        drop_cols.extend([col for col in exclude_columns if col not in drop_cols])
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    feature_df, name_mapping = sanitize_columns(feature_df)

    requested_categoricals = set(categorical_columns or [])
    if name_mapping:
        reverse_mapping = {orig: safe for safe, orig in name_mapping.items()}
        requested_categoricals = {reverse_mapping.get(col, col) for col in requested_categoricals}
    requested_categoricals = {col for col in requested_categoricals if col in feature_df.columns}

    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    for col in requested_categoricals:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col not in categorical_cols:
            categorical_cols.append(col)

    for col in numeric_cols:
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)
        median = feature_df[col].median()
        if pd.isna(median):
            median = 0.0
        feature_df[col] = feature_df[col].fillna(median)

    for col in categorical_cols:
        feature_df[col] = feature_df[col].where(feature_df[col].notna(), "MISSING")
        feature_df[col] = feature_df[col].astype(str).astype("category")

    target_series, target_kind = coerce_target(df[target_col])

    metadata = {
        "target_kind": target_kind,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "name_mapping": name_mapping,
    }
    return feature_df, target_series, metadata

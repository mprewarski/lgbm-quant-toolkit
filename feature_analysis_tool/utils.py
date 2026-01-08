import json
import math
import os
from typing import Any, Dict, Iterable, Tuple
import logging


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


def is_binary(series) -> bool:
    uniques = [v for v in series.dropna().unique()]
    return len(uniques) == 2


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile_bins(values, bins: int = 10):
    import numpy as np

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) <= 2:
        return None
    return edges


def sample_df(df, max_rows: int):
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)

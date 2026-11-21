from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import StatsConfig

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
QUANTILE_KEYS = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]

# Flags: (key, kind, description). kind "warn" = worth reviewing; "info" = describes the feature.
FLAG_INFO = {
    "constant": ("warn", "Only one distinct value; carries no information."),
    "high_missing": ("warn", "Share of missing values is above missing_threshold."),
    "inf_values": ("warn", "Contains +/-inf values (treated as missing in all stats)."),
    "pile_up": ("warn", "A large share of values sit exactly at the min or max, usually from clipping/trimming."),
    "heavy_tails": ("warn", "Excess kurtosis above kurtosis_threshold; a few extreme values dominate."),
    "drift": ("warn", "The per-date mean moves a lot relative to the overall spread (not normalized by date)."),
    "date_constant": ("info", "Same value for every stock on a date (macro or market-level series)."),
    "binary": ("info", "Exactly two distinct values."),
    "discrete": ("info", "Few distinct values (below low_unique_threshold)."),
    "target": ("info", "Target column (shown for reference, not a feature)."),
}


def _round(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(round(value, digits))


def _round_list(values, digits: int = 4) -> List[Optional[float]]:
    return [_round(v, digits) for v in values]


class FeatureStatsAnalyzer:
    def __init__(self, config: StatsConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("feature_stats")

    # ------------------------------------------------------------------ loading
    def load(self) -> pd.DataFrame:
        cfg = self.config
        self.logger.info("Loading %s", cfg.data_path)
        df = pd.read_csv(cfg.data_path, low_memory=False)
        self.logger.info("Loaded data: rows=%d, cols=%d", len(df), df.shape[1])
        if cfg.sample_rows and len(df) > cfg.sample_rows:
            df = df.sample(n=cfg.sample_rows, random_state=cfg.random_state)
            self.logger.info("Sampled %d rows", len(df))
        return df

    def _split_columns(self, df: pd.DataFrame):
        cfg = self.config
        exclude = set(cfg.exclude_columns or [])
        if cfg.date_col:
            exclude.add(cfg.date_col)
        target = cfg.target_col if cfg.target_col and cfg.target_col in df.columns else None
        requested_cat = set(cfg.categorical_columns or [])
        features = [c for c in df.columns if c not in exclude and c != target]
        categorical = [
            c for c in features if c in requested_cat or not pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric = [c for c in features if c not in categorical]
        return numeric, categorical, target

    # ------------------------------------------------------------------ main
    def run(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        cfg = self.config
        if df is None:
            df = self.load()
        numeric, categorical, target = self._split_columns(df)
        stat_cols = list(numeric)
        if cfg.include_target and target:
            stat_cols.append(target)
        self.logger.info(
            "Columns: numeric=%d, categorical=%d, target=%s", len(numeric), len(categorical), target
        )

        num = df[stat_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        inf_counts = np.isinf(num).sum()
        num = num.replace([np.inf, -np.inf], np.nan)

        dates = None
        if cfg.date_col and cfg.date_col in df.columns:
            dates = pd.to_datetime(df[cfg.date_col], errors="coerce")
            if dates.isna().all():
                self.logger.warning("Date column %s could not be parsed; skipping date stats", cfg.date_col)
                dates = None

        overall = self._overall_stats(num, inf_counts)
        per_date = self._per_date_stats(num, dates) if dates is not None else {}
        timeseries = self._timeseries(num, dates) if dates is not None else {}
        histograms = {col: self._histogram(num[col]) for col in stat_cols}

        features = []
        for col in stat_cols:
            row = {"feature": col, "role": "target" if col == target else "feature"}
            row.update(overall[col])
            row.update(per_date.get(col, {}))
            row["flags"] = self._flags(row)
            features.append(row)

        categorical_stats = [self._categorical_stats(df[col]) for col in categorical]

        meta = {
            "data_path": cfg.data_path,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric_features": len(numeric),
            "categorical_features": len(categorical),
            "target": target,
            "config": asdict(cfg),
        }
        if dates is not None:
            valid_dates = dates.dropna()
            meta.update(
                {
                    "dates": int(valid_dates.nunique()),
                    "date_start": str(valid_dates.min().date()),
                    "date_end": str(valid_dates.max().date()),
                    "rows_per_date_mean": _round(valid_dates.value_counts().mean(), 1),
                }
            )

        self.logger.info("Computed stats for %d columns", len(features))
        return {
            "meta": meta,
            "features": features,
            "categorical": categorical_stats,
            "histograms": histograms,
            "timeseries": timeseries,
            "flag_info": {k: {"kind": v[0], "description": v[1]} for k, v in FLAG_INFO.items()},
        }

    # ------------------------------------------------------------------ pieces
    def _overall_stats(self, num: pd.DataFrame, inf_counts: pd.Series) -> Dict[str, Dict[str, Any]]:
        n = len(num)
        count = num.count()
        mean = num.mean()
        std = num.std()
        mn = num.min()
        mx = num.max()
        q = num.quantile(QUANTILES)
        skew = num.skew()
        kurt = num.kurt()
        nunique = num.nunique()
        safe_count = count.replace(0, np.nan)
        zero_pct = num.eq(0).sum() / safe_count
        at_min = num.eq(mn).sum() / safe_count
        at_max = num.eq(mx).sum() / safe_count
        z = (num - mean) / std.replace(0, np.nan)
        outlier_pct = z.abs().gt(3).sum() / safe_count

        out: Dict[str, Dict[str, Any]] = {}
        for col in num.columns:
            stats = {
                "count": int(count[col]),
                "missing": int(n - count[col]),
                "missing_pct": _round((n - count[col]) / n if n else None),
                "inf_count": int(inf_counts[col]),
                "unique": int(nunique[col]),
                "mean": _round(mean[col]),
                "std": _round(std[col]),
                "min": _round(mn[col]),
                "max": _round(mx[col]),
                "skew": _round(skew[col], 4),
                "kurtosis": _round(kurt[col], 4),
                "zero_pct": _round(zero_pct[col]),
                "pct_at_min": _round(at_min[col]),
                "pct_at_max": _round(at_max[col]),
                "outlier_pct": _round(outlier_pct[col]),
            }
            for key, qv in zip(QUANTILE_KEYS, QUANTILES):
                stats[key] = _round(q.loc[qv, col])
            out[col] = stats
        return out

    def _per_date_stats(self, num: pd.DataFrame, dates: pd.Series) -> Dict[str, Dict[str, Any]]:
        mask = dates.notna()
        num = num[mask]
        dates = dates[mask]
        grouped = num.groupby(dates)
        size = grouped.size()
        d_count = grouped.count()
        d_mean = grouped.mean()
        d_std = grouped.std()
        d_nunique = grouped.nunique()
        coverage = d_count.div(size, axis=0)
        overall_std = num.std()

        out: Dict[str, Dict[str, Any]] = {}
        for col in num.columns:
            has_data = d_count[col] > 0
            multi = d_count[col] > 1
            n_multi = int(multi.sum())
            date_constant_pct = (
                float(((d_nunique[col] <= 1) & multi).sum()) / n_multi if n_multi else None
            )
            sd = overall_std[col]
            drift = d_mean[col].std() / sd if sd and np.isfinite(sd) and sd > 0 else None
            out[col] = {
                "dates_with_data": int(has_data.sum()),
                "first_date": str(d_count.index[has_data.to_numpy()].min().date()) if has_data.any() else None,
                "last_date": str(d_count.index[has_data.to_numpy()].max().date()) if has_data.any() else None,
                "coverage_mean": _round(coverage[col].mean()),
                "coverage_min": _round(coverage[col][has_data].min()) if has_data.any() else None,
                "date_mean_std": _round(d_mean[col].std()),
                "within_date_std_mean": _round(d_std[col].mean()),
                "date_constant_pct": _round(date_constant_pct),
                "drift_ratio": _round(drift, 4),
            }
        return out

    def _timeseries(self, num: pd.DataFrame, dates: pd.Series) -> Dict[str, Any]:
        mask = dates.notna()
        num = num[mask]
        bucket = dates[mask].dt.to_period(self.config.time_bucket).dt.to_timestamp()
        grouped = num.groupby(bucket)
        ts_mean = grouped.mean()
        ts_q = grouped.quantile([0.05, 0.5, 0.95])
        ts_missing = 1 - grouped.count().div(grouped.size(), axis=0)
        p05 = ts_q.xs(0.05, level=1)
        p50 = ts_q.xs(0.5, level=1)
        p95 = ts_q.xs(0.95, level=1)
        series = {}
        for col in num.columns:
            series[col] = {
                "mean": _round_list(ts_mean[col]),
                "p05": _round_list(p05[col]),
                "p50": _round_list(p50[col]),
                "p95": _round_list(p95[col]),
                "missing_pct": _round_list(ts_missing[col]),
            }
        return {
            "bucket": self.config.time_bucket,
            "periods": [str(ts.date()) for ts in ts_mean.index],
            "series": series,
        }

    def _histogram(self, series: pd.Series) -> Dict[str, Any]:
        values = series.dropna().to_numpy()
        if values.size == 0:
            return {"edges": [], "counts": [], "clipped_low": 0, "clipped_high": 0}
        clip = self.config.hist_clip_quantile or 0.0
        if clip > 0:
            lo, hi = np.quantile(values, [clip, 1 - clip])
        else:
            lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = max(1, int(self.config.hist_bins))
        counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
        return {
            "edges": _round_list(edges, 6),
            "counts": [int(c) for c in counts],
            "clipped_low": int((values < lo).sum()),
            "clipped_high": int((values > hi).sum()),
        }

    def _flags(self, row: Dict[str, Any]) -> List[str]:
        cfg = self.config
        flags: List[str] = []
        if row.get("role") == "target":
            flags.append("target")
        unique = row.get("unique", 0)
        if unique <= 1:
            flags.append("constant")
        if (row.get("missing_pct") or 0) > cfg.missing_threshold:
            flags.append("high_missing")
        if row.get("inf_count", 0) > 0:
            flags.append("inf_values")
        pile = max(row.get("pct_at_min") or 0, row.get("pct_at_max") or 0)
        if unique > cfg.low_unique_threshold and pile > cfg.pileup_threshold:
            flags.append("pile_up")
        if (row.get("kurtosis") or 0) > cfg.kurtosis_threshold:
            flags.append("heavy_tails")
        date_constant = (row.get("date_constant_pct") or 0) >= cfg.date_constant_threshold
        if date_constant:
            flags.append("date_constant")
        elif (row.get("drift_ratio") or 0) > cfg.drift_threshold:
            flags.append("drift")
        if unique == 2:
            flags.append("binary")
        elif 2 < unique < cfg.low_unique_threshold:
            flags.append("discrete")
        return flags

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        n = len(series)
        counts = series.value_counts(dropna=True)
        top = counts.head(self.config.top_categories)
        return {
            "feature": series.name,
            "count": int(series.notna().sum()),
            "missing_pct": _round(series.isna().sum() / n if n else None),
            "categories": int(counts.size),
            "top": [{"value": str(k), "count": int(v), "pct": _round(v / n, 6)} for k, v in top.items()],
            "other_count": int(counts.iloc[len(top):].sum()),
        }


def stats_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Flat table of per-feature stats (for stats.csv)."""
    df = pd.DataFrame(results["features"])
    df["flags"] = df["flags"].apply(lambda f: ";".join(f))
    return df

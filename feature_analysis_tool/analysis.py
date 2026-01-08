from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig
import logging
import re

from .utils import (
    optional_import,
    percentile_bins,
    sample_df,
    safe_divide,
)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: Optional[str],
    exclude_columns: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Dict[str, Any]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    date_series = None
    if date_col and date_col in df.columns:
        date_series = pd.to_datetime(df[date_col], errors="coerce")

    target_series, target_kind = coerce_target(df[target_col])

    drop_cols = [target_col]
    if date_col and date_col in df.columns:
        drop_cols.append(date_col)
    if exclude_columns:
        drop_cols.extend([col for col in exclude_columns if col not in drop_cols])
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

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
        feature_df.columns = new_cols

    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    for col in numeric_cols:
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)
        median = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(median)

    for col in categorical_cols:
        feature_df[col] = feature_df[col].where(feature_df[col].notna(), "MISSING")
        feature_df[col] = feature_df[col].astype(str).astype("category")

    metadata = {
        "target_kind": target_kind,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    if name_mapping:
        metadata["name_mapping"] = name_mapping
    return feature_df, target_series, date_series, metadata


def train_or_load_model(
    feature_df: pd.DataFrame,
    target: pd.Series,
    config: AnalysisConfig,
    metadata: Dict[str, Any],
) -> Tuple[Optional[Any], List[str]]:
    notes = []
    logger = logging.getLogger("feature_analysis")
    lgb, err = optional_import("lightgbm")
    if lgb is None:
        notes.append(f"LightGBM not available: {err}")
        logger.warning("LightGBM not available: %s", err)
        return None, notes

    if config.model_path:
        try:
            booster = lgb.Booster(model_file=config.model_path)
            logger.info("Loaded LightGBM model from %s", config.model_path)
            return booster, notes
        except Exception as exc:
            notes.append(f"Failed to load model '{config.model_path}': {exc}")
            logger.warning("Failed to load model '%s': %s", config.model_path, exc)

    target_kind = metadata["target_kind"]
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": config.random_state,
    }
    if config.lgbm_params:
        params.update(config.lgbm_params)
    logger.info(
        "Training LightGBM model: target_kind=%s, rows=%d, features=%d, params=%s",
        target_kind,
        len(feature_df),
        feature_df.shape[1],
        params,
    )

    try:
        if target_kind in {"binary", "categorical"}:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        model.fit(feature_df, target)
        logger.info("Model training completed")
        return model, notes
    except Exception as exc:
        notes.append(f"Model training failed: {exc}")
        logger.exception("Model training failed")
        return None, notes


def compute_feature_importance(model: Any) -> Dict[str, Dict[str, float]]:
    if model is None:
        return {}

    if hasattr(model, "booster_"):
        booster = model.booster_
    else:
        booster = model

    feature_names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    importance = {}
    for name, gain_val, split_val in zip(feature_names, gain, split):
        importance[name] = {
            "gain": float(gain_val),
            "split": float(split_val),
        }
    return importance


def compute_correlation(feature_df: pd.DataFrame, target: pd.Series, numeric_cols: List[str]) -> Dict[str, float]:
    if len(numeric_cols) == 0:
        return {}
    corr = feature_df[numeric_cols].corrwith(target)
    return {name: float(val) for name, val in corr.items()}


def compute_mutual_info(feature_df: pd.DataFrame, target: pd.Series, numeric_cols: List[str], target_kind: str) -> Tuple[Dict[str, float], List[str]]:
    notes = []
    sklearn, err = optional_import("sklearn")
    if sklearn is None:
        notes.append(f"sklearn not available: {err}")
        return {}, notes

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if len(numeric_cols) == 0:
        return {}, notes

    X = feature_df[numeric_cols]
    if target_kind in {"binary", "categorical"}:
        mi = mutual_info_classif(X, target, random_state=42)
    else:
        mi = mutual_info_regression(X, target, random_state=42)
    return {name: float(val) for name, val in zip(numeric_cols, mi)}, notes


def compute_ic(feature_df: pd.DataFrame, target: pd.Series, numeric_cols: List[str]) -> Tuple[Dict[str, float], List[str]]:
    notes = []
    scipy, err = optional_import("scipy")
    if scipy is None:
        notes.append(f"scipy not available: {err}")
        corr = feature_df[numeric_cols].corrwith(target, method="spearman") if numeric_cols else {}
        return {name: float(val) for name, val in corr.items()}, notes

    from scipy.stats import spearmanr

    ic = {}
    for col in numeric_cols:
        val = spearmanr(feature_df[col], target).correlation
        ic[col] = float(val) if val is not None else 0.0
    return ic, notes


def compute_vif(feature_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[Dict[str, float], List[str]]:
    notes = []
    statsmodels, err = optional_import("statsmodels")
    if statsmodels is None:
        notes.append(f"statsmodels not available: {err}")
        return {}, notes

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if len(numeric_cols) == 0:
        return {}, notes

    subset_cols = numeric_cols[:200]
    X = feature_df[subset_cols].astype(float)
    vif = {}
    for idx, col in enumerate(subset_cols):
        try:
            vif[col] = float(variance_inflation_factor(X.values, idx))
        except Exception:
            vif[col] = float("nan")
    if len(numeric_cols) > len(subset_cols):
        notes.append("VIF computed on first 200 numeric features")
    return vif, notes



def compute_stability(
    feature_df: pd.DataFrame,
    date_series: Optional[pd.Series],
    numeric_cols: List[str],
    time_bucket: str,
) -> Dict[str, Dict[str, float]]:
    if date_series is None or len(numeric_cols) == 0:
        return {}

    df = feature_df.copy()
    df["__date__"] = date_series
    df = df.dropna(subset=["__date__"])
    if df.empty:
        return {}

    df["__bucket__"] = df["__date__"].dt.to_period(time_bucket).dt.to_timestamp()

    stability = {}
    for col in numeric_cols:
        series = df[col]
        edges = percentile_bins(series.values, bins=10)
        if edges is None:
            stability[col] = {"psi_max": 0.0, "psi_mean": 0.0, "mean_cv": 0.0}
            continue
        overall_hist, _ = np.histogram(series.values, bins=edges)
        overall_dist = overall_hist / max(overall_hist.sum(), 1)

        psi_scores = []
        means = []
        for _, group in df.groupby("__bucket__"):
            hist, _ = np.histogram(group[col].values, bins=edges)
            dist = hist / max(hist.sum(), 1)
            dist = np.where(dist == 0, 1e-6, dist)
            overall_adj = np.where(overall_dist == 0, 1e-6, overall_dist)
            psi = np.sum((dist - overall_adj) * np.log(dist / overall_adj))
            psi_scores.append(float(psi))
            means.append(float(group[col].mean()))
        mean_cv = float(np.std(means) / (np.mean(means) if np.mean(means) != 0 else 1))
        stability[col] = {
            "psi_max": float(np.max(psi_scores)) if psi_scores else 0.0,
            "psi_mean": float(np.mean(psi_scores)) if psi_scores else 0.0,
            "mean_cv": mean_cv,
        }
    return stability


def compute_woe_iv(
    feature_df: pd.DataFrame,
    target: pd.Series,
    numeric_cols: List[str],
    target_kind: str,
) -> Dict[str, Dict[str, float]]:
    if target_kind != "binary" or len(numeric_cols) == 0:
        return {}

    target_values = target.values
    total_good = (target_values == 0).sum()
    total_bad = (target_values == 1).sum()
    if total_good == 0 or total_bad == 0:
        return {}

    woe_iv = {}
    for col in numeric_cols:
        values = feature_df[col].values
        edges = percentile_bins(values, bins=10)
        if edges is None:
            continue
        bins = np.digitize(values, edges[1:-1], right=True)
        iv_total = 0.0
        for bin_idx in np.unique(bins):
            mask = bins == bin_idx
            good = (target_values[mask] == 0).sum()
            bad = (target_values[mask] == 1).sum()
            good_rate = safe_divide(good, total_good)
            bad_rate = safe_divide(bad, total_bad)
            if good_rate == 0 or bad_rate == 0:
                continue
            woe = math.log(good_rate / bad_rate)
            iv_total += (good_rate - bad_rate) * woe
        woe_iv[col] = {"iv": float(iv_total)}
    return woe_iv


def compute_shap(model: Any, feature_df: pd.DataFrame, config: AnalysisConfig) -> Tuple[Dict[str, float], List[str]]:
    notes = []
    shap_mod, err = optional_import("shap")
    if shap_mod is None:
        notes.append(f"shap not available: {err}")
        return {}, notes
    if model is None:
        return {}, notes

    sample = sample_df(feature_df, config.shap_sample_rows)

    try:
        explainer = shap_mod.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {name: float(val) for name, val in zip(sample.columns, mean_abs)}, notes
    except Exception as exc:
        notes.append(f"SHAP failed: {exc}")
        return {}, notes


def detect_leakage(
    feature_names: List[str],
    corr: Dict[str, float],
    mutual_info: Dict[str, float],
    config: AnalysisConfig,
) -> Dict[str, Dict[str, Any]]:
    leakage = {}
    if mutual_info:
        mi_values = np.array(list(mutual_info.values()))
        mi_threshold = float(np.quantile(mi_values, config.leakage_mi_quantile))
    else:
        mi_threshold = None

    for name in feature_names:
        corr_val = corr.get(name)
        mi_val = mutual_info.get(name)
        flag_corr = corr_val is not None and abs(corr_val) >= config.leakage_corr_threshold
        flag_mi = mi_threshold is not None and mi_val is not None and mi_val >= mi_threshold
        leakage[name] = {
            "flag": bool(flag_corr or flag_mi),
            "corr": float(corr_val) if corr_val is not None else None,
            "mutual_info": float(mi_val) if mi_val is not None else None,
        }
    return leakage


def build_feature_table(
    feature_names: List[str],
    importance: Dict[str, Dict[str, float]],
    corr: Dict[str, float],
    mutual_info: Dict[str, float],
    ic: Dict[str, float],
    vif: Dict[str, float],
    stability: Dict[str, Dict[str, float]],
    woe_iv: Dict[str, Dict[str, float]],
    shap_vals: Dict[str, float],
    leakage: Dict[str, Dict[str, Any]],
    name_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    table = []
    for name in feature_names:
        display_name = name_mapping.get(name, name) if name_mapping else name
        row = {
            "feature": display_name,
            "importance_gain": importance.get(name, {}).get("gain"),
            "importance_split": importance.get(name, {}).get("split"),
            "corr": corr.get(name),
            "mutual_info": mutual_info.get(name),
            "ic": ic.get(name),
            "vif": vif.get(name),
            "psi_max": stability.get(name, {}).get("psi_max"),
            "mean_cv": stability.get(name, {}).get("mean_cv"),
            "iv": woe_iv.get(name, {}).get("iv"),
            "shap_mean_abs": shap_vals.get(name),
            "leakage_flag": leakage.get(name, {}).get("flag"),
        }
        table.append(row)
    return table


class FeatureAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config

    def run(self) -> Tuple[Dict[str, Any], Optional[Any]]:
        notes: List[str] = []
        logger = logging.getLogger("feature_analysis")
        df = load_data(self.config.data_path)
        df = sample_df(df, self.config.sample_rows)
        logger.info("Loaded data: rows=%d, cols=%d", len(df), df.shape[1])

        feature_df, target, date_series, metadata = prepare_features(
            df,
            self.config.target_col,
            self.config.date_col,
            self.config.exclude_columns,
        )
        if metadata.get("name_mapping"):
            logger.info("Sanitized %d feature names for LightGBM", len(metadata["name_mapping"]))
            notes.append("Sanitized feature names to satisfy LightGBM constraints")
        if self.config.exclude_columns:
            notes.append(f"Excluded columns: {', '.join(self.config.exclude_columns)}")
            logger.info("Excluded columns: %s", ", ".join(self.config.exclude_columns))
        logger.info(
            "Prepared features: numeric=%d, categorical=%d",
            len(metadata["numeric_cols"]),
            len(metadata["categorical_cols"]),
        )

        model, model_notes = train_or_load_model(feature_df, target, self.config, metadata)
        notes.extend(model_notes)

        importance = compute_feature_importance(model)
        corr = compute_correlation(feature_df, target, metadata["numeric_cols"])
        mutual_info, mi_notes = compute_mutual_info(
            feature_df, target, metadata["numeric_cols"], metadata["target_kind"]
        )
        notes.extend(mi_notes)
        ic, ic_notes = compute_ic(feature_df, target, metadata["numeric_cols"])
        notes.extend(ic_notes)
        vif, vif_notes = compute_vif(feature_df, metadata["numeric_cols"])
        notes.extend(vif_notes)
        stability = compute_stability(
            feature_df,
            date_series,
            metadata["numeric_cols"],
            self.config.time_bucket,
        )
        woe_iv = compute_woe_iv(
            feature_df,
            target,
            metadata["numeric_cols"],
            metadata["target_kind"],
        )
        shap_vals, shap_notes = compute_shap(model, feature_df, self.config)
        notes.extend(shap_notes)

        leakage = detect_leakage(
            list(feature_df.columns),
            corr,
            mutual_info,
            self.config,
        )

        feature_table = build_feature_table(
            list(feature_df.columns),
            importance,
            corr,
            mutual_info,
            ic,
            vif,
            stability,
            woe_iv,
            shap_vals,
            leakage,
            metadata.get("name_mapping"),
        )

        results = {
            "config": asdict(self.config),
            "metadata": metadata,
            "importance": importance,
            "correlation": corr,
            "mutual_info": mutual_info,
            "ic": ic,
            "vif": vif,
            "stability": stability,
            "woe_iv": woe_iv,
            "shap": shap_vals,
            "leakage": leakage,
            "feature_table": feature_table,
            "notes": notes,
        }
        return results, model

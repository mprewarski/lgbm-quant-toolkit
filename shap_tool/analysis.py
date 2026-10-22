from __future__ import annotations

from dataclasses import asdict
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ShapConfig
from .report import generate_report
from .utils import dump_json, ensure_dir, optional_import, sample_df


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
    categorical_columns: Optional[List[str]],
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
    if any(not str(col).replace("_", "").isalnum() for col in feature_df.columns):
        used = set()
        new_cols = []
        for col in feature_df.columns:
            safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(col))
            safe = "_".join(filter(None, safe.split("_")))
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

    constant_cols = [col for col in feature_df.columns if feature_df[col].nunique(dropna=False) <= 1]

    metadata = {
        "target_kind": target_kind,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "constant_cols": constant_cols,
    }
    if name_mapping:
        metadata["name_mapping"] = name_mapping
    if requested_categoricals:
        metadata["requested_categoricals"] = sorted(requested_categoricals)
    return feature_df, target_series, date_series, metadata


def _parse_date_range(range_value: Optional[List[str]]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not range_value:
        return None
    if len(range_value) != 2:
        raise ValueError("Date ranges must be [start, end]")
    start = pd.to_datetime(range_value[0]) if range_value[0] else None
    end = pd.to_datetime(range_value[1]) if range_value[1] else None
    return start, end


def _filter_by_range(df: pd.DataFrame, date_series: pd.Series, date_range):
    if date_range is None:
        return df
    start, end = date_range
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= date_series >= start
    if end is not None:
        mask &= date_series <= end
    return df[mask]


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    date_series: Optional[pd.Series],
    config: ShapConfig,
) -> Dict[str, Any]:
    if date_series is None and (
        config.train_date_range or config.val_date_range or config.test_date_range
    ):
        raise ValueError("Date ranges require a valid date_col in the dataset")
    if date_series is not None and (
        config.train_date_range or config.val_date_range or config.test_date_range
    ):
        train_range = _parse_date_range(config.train_date_range)
        val_range = _parse_date_range(config.val_date_range)
        test_range = _parse_date_range(config.test_date_range)
        train_df = _filter_by_range(features, date_series, train_range)
        val_df = _filter_by_range(features, date_series, val_range)
        test_df = _filter_by_range(features, date_series, test_range)
        train_target = target.loc[train_df.index]
        val_target = target.loc[val_df.index]
        test_target = target.loc[test_df.index]
        return {
            "train_features": train_df,
            "val_features": val_df,
            "test_features": test_df,
            "train_target": train_target,
            "val_target": val_target,
            "test_target": test_target,
            "split_info": {
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "train_range": config.train_date_range,
                "val_range": config.val_date_range,
                "test_range": config.test_date_range,
            },
        }

    if date_series is not None:
        df = features.copy()
        df["__date__"] = date_series
        df = df.sort_values("__date__")
        target_sorted = target.loc[df.index]
        df = df.drop(columns=["__date__"])

        train_end = int(len(df) * config.train_ratio)
        train_df = df.iloc[:train_end]
        train_target = target_sorted.iloc[:train_end]

        remaining_df = df.iloc[train_end:]
        remaining_target = target_sorted.iloc[train_end:]

        if config.gap_days > 0 and len(train_df) > 0:
            train_last_date = date_series.loc[train_df.index].max()
            gap_cutoff = train_last_date + pd.Timedelta(days=config.gap_days)
            keep_mask = date_series.loc[remaining_df.index] > gap_cutoff
            remaining_df = remaining_df[keep_mask]
            remaining_target = remaining_target.loc[remaining_df.index]

        remaining_total = config.val_ratio + config.test_ratio
        if remaining_total > 0:
            val_ratio = config.val_ratio / remaining_total
        else:
            val_ratio = 0.0

        val_end = int(len(remaining_df) * val_ratio)
        val_df = remaining_df.iloc[:val_end]
        val_target = remaining_target.iloc[:val_end]
        test_df = remaining_df.iloc[val_end:]
        test_target = remaining_target.iloc[val_end:]

        if config.gap_days > 0 and len(val_df) > 0:
            val_last_date = date_series.loc[val_df.index].max()
            gap_cutoff = val_last_date + pd.Timedelta(days=config.gap_days)
            keep_mask = date_series.loc[test_df.index] > gap_cutoff
            test_df = test_df[keep_mask]
            test_target = test_target.loc[test_df.index]

        split_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_range": _date_range_string(date_series.loc[train_df.index]),
            "val_range": _date_range_string(date_series.loc[val_df.index]),
            "test_range": _date_range_string(date_series.loc[test_df.index]),
        }
        return {
            "train_features": train_df,
            "val_features": val_df,
            "test_features": test_df,
            "train_target": train_target,
            "val_target": val_target,
            "test_target": test_target,
            "split_info": split_info,
        }

    from sklearn.model_selection import train_test_split

    train_df, temp_df, train_target, temp_target = train_test_split(
        features,
        target,
        train_size=config.train_ratio,
        random_state=config.random_state,
        shuffle=True,
    )
    remaining_total = config.val_ratio + config.test_ratio
    if remaining_total > 0:
        val_ratio = config.val_ratio / remaining_total
    else:
        val_ratio = 0.0
    val_df, test_df, val_target, test_target = train_test_split(
        temp_df,
        temp_target,
        train_size=val_ratio,
        random_state=config.random_state,
        shuffle=True,
    )
    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_range": None,
        "val_range": None,
        "test_range": None,
    }
    return {
        "train_features": train_df,
        "val_features": val_df,
        "test_features": test_df,
        "train_target": train_target,
        "val_target": val_target,
        "test_target": test_target,
        "split_info": split_info,
    }


def _date_range_string(series: pd.Series) -> Optional[str]:
    if series is None or series.empty:
        return None
    return f"{series.min().date()} to {series.max().date()}"


def train_or_load_model(
    feature_df: pd.DataFrame,
    target: pd.Series,
    config: ShapConfig,
    metadata: Dict[str, Any],
) -> Tuple[Optional[Any], List[str]]:
    notes: List[str] = []
    logger = logging.getLogger("shap_tool")
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
        "n_estimators": 400,
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
        categorical_features = metadata.get("categorical_cols") or None
        model.fit(feature_df, target, categorical_feature=categorical_features)
        logger.info("Model training completed")
        return model, notes
    except Exception as exc:
        notes.append(f"Model training failed: {exc}")
        logger.exception("Model training failed")
        return None, notes


def _resolve_base_value(base_value):
    if isinstance(base_value, list):
        return base_value
    if hasattr(base_value, "tolist"):
        return base_value.tolist()
    return base_value


def _select_shap_output(shap_values, base_values, config: ShapConfig, target_kind: str):
    if not isinstance(shap_values, list):
        return shap_values, base_values

    class_index = config.shap_class_index
    if class_index is None:
        if target_kind == "binary" and len(shap_values) == 2:
            class_index = 1
        else:
            class_index = 0
    return shap_values[class_index], base_values[class_index]


def _safe_feature_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or "feature"


def compute_shap_outputs(
    model: Any,
    feature_df: pd.DataFrame,
    config: ShapConfig,
    metadata: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    shap_mod, err = optional_import("shap")
    if shap_mod is None:
        raise RuntimeError(f"shap is not available: {err}")

    logger = logging.getLogger("shap_tool")
    X_sample = sample_df(feature_df, config.shap_sample_rows, config.random_state)

    explainer = shap_mod.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    base_values = explainer.expected_value

    shap_values, base_values = _select_shap_output(
        shap_values, base_values, config, metadata["target_kind"]
    )

    if hasattr(base_values, "__len__") and not isinstance(base_values, str):
        base_value = np.array(base_values).flatten()[0]
    else:
        base_value = base_values

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    name_mapping = metadata.get("name_mapping", {})
    importance_df = pd.DataFrame(
        {
            "feature": feature_df.columns,
            "feature_original": [name_mapping.get(col, col) for col in feature_df.columns],
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    feature_importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(feature_importance_path, index=False)

    if config.low_importance_threshold is not None:
        threshold = config.low_importance_threshold
    else:
        threshold = float(np.quantile(mean_abs_shap, config.low_importance_quantile))

    low_importance_df = importance_df[importance_df["mean_abs_shap"] <= threshold]
    low_importance_path = os.path.join(output_dir, "low_importance.csv")
    low_importance_df.to_csv(low_importance_path, index=False)

    summary_plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.figure()
    shap_mod.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()

    dependence_paths: List[str] = []
    top_features = importance_df["feature"].head(config.dependence_plots).tolist()
    for feature in top_features:
        fig_path = os.path.join(output_dir, f"dependence_{_safe_feature_name(feature)}.png")
        plt.figure()
        shap_mod.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        dependence_paths.append(fig_path)

    waterfall_plot_path = os.path.join(output_dir, "waterfall_plot.png")
    waterfall_index = min(max(config.force_index, 0), len(X_sample) - 1)
    try:
        shap_exp = shap_mod.Explanation(
            values=shap_values[waterfall_index],
            base_values=base_value,
            data=X_sample.iloc[waterfall_index],
            feature_names=X_sample.columns,
        )
        plt.figure()
        shap_mod.plots.waterfall(shap_exp, show=False, max_display=config.top_features)
        plt.tight_layout()
        plt.savefig(waterfall_plot_path, bbox_inches="tight")
        plt.close()
    except Exception:
        waterfall_plot_path = None

    force_plot_path = os.path.join(output_dir, "force_plot.html")
    force_index = min(max(config.force_index, 0), len(X_sample) - 1)
    force_plot = shap_mod.force_plot(
        base_value,
        shap_values[force_index],
        X_sample.iloc[force_index],
        matplotlib=False,
    )
    shap_mod.save_html(force_plot_path, force_plot)

    interactions_path = None
    interaction_summary = []
    if config.interaction_sample_rows and config.interaction_sample_rows > 0:
        X_inter = sample_df(
            X_sample,
            min(config.interaction_sample_rows, len(X_sample)),
            config.random_state,
        )
        interaction_values = explainer.shap_interaction_values(X_inter)
        if isinstance(interaction_values, list):
            interaction_values, _ = _select_shap_output(
                interaction_values, base_values, config, metadata["target_kind"]
            )
        interaction_mean = np.abs(interaction_values).mean(axis=0)
        records = []
        for i, feature_i in enumerate(feature_df.columns):
            for j in range(i + 1, len(feature_df.columns)):
                records.append(
                    {
                        "feature_1": feature_i,
                        "feature_2": feature_df.columns[j],
                        "mean_abs_interaction": interaction_mean[i, j],
                    }
                )
        interactions_df = pd.DataFrame(records).sort_values(
            "mean_abs_interaction", ascending=False
        )
        interactions_path = os.path.join(output_dir, "shap_interactions.csv")
        interactions_df.to_csv(interactions_path, index=False)
        interaction_summary = (
            interactions_df.head(config.top_interactions).to_dict(orient="records")
        )
        logger.info("Computed SHAP interaction values")

    return {
        "feature_importance_path": feature_importance_path,
        "low_importance_path": low_importance_path,
        "summary_plot_path": summary_plot_path,
        "dependence_paths": dependence_paths,
        "waterfall_plot_path": waterfall_plot_path,
        "force_plot_path": force_plot_path,
        "interactions_path": interactions_path,
        "interaction_summary": interaction_summary,
        "top_features": importance_df.head(config.top_features).to_dict(orient="records"),
        "low_importance_features": low_importance_df.to_dict(orient="records"),
        "base_value": _resolve_base_value(base_values),
    }


def run_shap(config: ShapConfig) -> Dict[str, Any]:
    logger = logging.getLogger("shap_tool")
    df = load_data(config.data_path)
    if config.sample_rows:
        df = sample_df(df, config.sample_rows, config.random_state)

    feature_df, target, date_series, metadata = prepare_features(
        df,
        config.target_col,
        config.date_col,
        config.exclude_columns,
        config.categorical_columns,
    )

    split_data = split_dataset(feature_df, target, date_series, config)
    train_features = split_data["train_features"]
    train_target = split_data["train_target"]
    if train_features.empty:
        raise ValueError("Training split is empty; adjust split ratios or date ranges")

    model, notes = train_or_load_model(train_features, train_target, config, metadata)
    if model is None:
        raise RuntimeError("Model unavailable; see logs for details")

    outputs = compute_shap_outputs(model, train_features, config, metadata, config.output_dir)

    results = {
        "config": asdict(config),
        "dataset": {"rows": len(df), "columns": df.shape[1]},
        "metadata": metadata,
        "splits": split_data["split_info"],
        "outputs": {
            "summary_plot": outputs["summary_plot_path"],
            "waterfall_plot": outputs["waterfall_plot_path"],
            "force_plot": outputs["force_plot_path"],
            "feature_importance": outputs["feature_importance_path"],
            "low_importance": outputs["low_importance_path"],
            "dependence_plots": outputs["dependence_paths"],
            "shap_interactions": outputs["interactions_path"],
        },
        "top_features": outputs["top_features"],
        "low_importance_features": outputs["low_importance_features"],
        "interaction_summary": outputs["interaction_summary"],
        "notes": notes,
    }
    logger.info("SHAP analysis completed")
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> str:
    path = os.path.join(output_dir, "results.json")
    dump_json(path, results)
    return path


def run(config: ShapConfig) -> Dict[str, Any]:
    ensure_dir(config.output_dir)
    results = run_shap(config)
    save_results(results, config.output_dir)
    generate_report(results, config.output_dir)
    return results

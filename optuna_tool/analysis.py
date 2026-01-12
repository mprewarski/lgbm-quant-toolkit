from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import lightgbm as lgbm
import optuna

from .config import OptunaConfig
from .utils import dump_json, ensure_dir, prepare_features, setup_logger


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_metric(target_kind: str, metric: Optional[str]) -> str:
    if metric:
        return metric
    if target_kind == "binary":
        return "auc"
    if target_kind == "categorical":
        return "multi_logloss"
    return "rmse"


def _infer_direction(metric: str, direction: Optional[str]) -> str:
    if direction:
        return direction
    maximize_metrics = {"auc", "accuracy", "r2"}
    return "maximize" if metric in maximize_metrics else "minimize"


def _eval_metric_name(metric: str, target_kind: str) -> Optional[str]:
    metric_map = {
        "auc": "auc",
        "accuracy": "binary_error" if target_kind == "binary" else "multi_error",
        "logloss": "binary_logloss" if target_kind == "binary" else "multi_logloss",
        "multi_logloss": "multi_logloss",
        "rmse": "rmse",
        "mae": "mae",
        "mse": "l2",
        "r2": None,
    }
    return metric_map.get(metric)


def _evaluate_metric(
    metric: str,
    target_kind: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> float:
    if metric == "auc":
        if target_kind != "binary":
            raise ValueError("AUC is only supported for binary targets.")
        if y_proba is None:
            raise ValueError("AUC requires predicted probabilities.")
        return roc_auc_score(y_true, y_proba[:, 1])
    if metric in {"logloss", "multi_logloss"}:
        if y_proba is None:
            raise ValueError("Logloss requires predicted probabilities.")
        return log_loss(y_true, y_proba)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric == "mse":
        return mean_squared_error(y_true, y_pred)
    if metric == "rmse":
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "r2":
        return r2_score(y_true, y_pred)
    raise ValueError(f"Unsupported metric '{metric}'")


def _validate_metric(metric: str, target_kind: str) -> None:
    classification_metrics = {"auc", "logloss", "multi_logloss", "accuracy"}
    regression_metrics = {"rmse", "mae", "mse", "r2"}
    if target_kind == "numeric" and metric not in regression_metrics:
        raise ValueError(f"Metric '{metric}' is not supported for numeric targets.")
    if target_kind in {"binary", "categorical"} and metric not in classification_metrics:
        raise ValueError(f"Metric '{metric}' is not supported for classification targets.")


def _optuna_report_callback(
    trial: optuna.Trial,
    metric_name: Optional[str],
    metric: str,
    enable_pruning: bool,
):
    if metric_name is None:
        return None

    def callback(env) -> None:
        for _, eval_name, result, _ in env.evaluation_result_list:
            if eval_name != metric_name:
                continue
            report_value = result
            if metric == "accuracy":
                report_value = 1.0 - result
            trial.report(report_value, step=env.iteration)
            if enable_pruning and trial.should_prune():
                raise optuna.TrialPruned()
            break

    return callback


def _split_data(
    features: pd.DataFrame,
    target: pd.Series,
    target_kind: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_size is None:
        test_size = 0.2
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    stratify = target if target_kind in {"binary", "categorical"} else None
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


class OptunaTuner:
    def __init__(self, config: OptunaConfig) -> None:
        self.config = config
        ensure_dir(config.output_dir)
        log_path = config.log_path or os.path.join(config.output_dir, "run.log")
        self.logger = setup_logger("optuna_tool", config.log_level, log_path)

    def _param_bounds(self, name: str, default_min: float, default_max: float) -> Tuple[float, float]:
        ranges = self.config.param_ranges or {}
        entry = ranges.get(name) or {}
        min_val = entry.get("min", default_min)
        max_val = entry.get("max", default_max)
        if min_val > max_val:
            raise ValueError(f"Invalid bounds for '{name}': min {min_val} > max {max_val}")
        return min_val, max_val

    def _resolve_storage(self) -> Optional[str]:
        storage_path = self.config.storage_path
        if not storage_path:
            return None
        storage_path = os.path.expanduser(storage_path)
        if not os.path.isabs(storage_path):
            norm_output = os.path.normpath(self.config.output_dir)
            norm_storage = os.path.normpath(storage_path)
            if not norm_storage.startswith(norm_output):
                storage_path = os.path.join(self.config.output_dir, storage_path)
        ensure_dir(os.path.dirname(storage_path))
        return f"sqlite:///{storage_path}"

    def _build_model(
        self,
        trial: optuna.Trial,
        target_kind: str,
        num_classes: int,
        metric: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        learning_rate_min, learning_rate_max = self._param_bounds("learning_rate", 0.01, 0.3)
        num_leaves_min, num_leaves_max = self._param_bounds("num_leaves", 16, 256)
        max_depth_min, max_depth_max = self._param_bounds("max_depth", -1, 12)
        min_child_samples_min, min_child_samples_max = self._param_bounds(
            "min_child_samples", 5, 100
        )
        subsample_min, subsample_max = self._param_bounds("subsample", 0.6, 1.0)
        colsample_min, colsample_max = self._param_bounds("colsample_bytree", 0.6, 1.0)
        reg_alpha_min, reg_alpha_max = self._param_bounds("reg_alpha", 1e-3, 10.0)
        reg_lambda_min, reg_lambda_max = self._param_bounds("reg_lambda", 1e-3, 10.0)
        min_split_gain_min, min_split_gain_max = self._param_bounds("min_split_gain", 0.0, 1.0)
        n_estimators_min, n_estimators_max = self._param_bounds("n_estimators", 200, 1200)
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", learning_rate_min, learning_rate_max, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", int(num_leaves_min), int(num_leaves_max)
            ),
            "max_depth": trial.suggest_int(
                "max_depth", int(max_depth_min), int(max_depth_max)
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", int(min_child_samples_min), int(min_child_samples_max)
            ),
            "subsample": trial.suggest_float("subsample", subsample_min, subsample_max),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", colsample_min, colsample_max
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", reg_alpha_min, reg_alpha_max, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", reg_lambda_min, reg_lambda_max, log=True
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", min_split_gain_min, min_split_gain_max
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", int(n_estimators_min), int(n_estimators_max)
            ),
            "random_state": self.config.random_state,
            "verbosity": -1,
        }
        if params["subsample"] < 1.0:
            params["subsample_freq"] = 1

        if target_kind == "numeric":
            model = lgbm.LGBMRegressor(**params, objective="regression")
            return model, params
        if target_kind == "binary":
            model = lgbm.LGBMClassifier(**params, objective="binary")
            return model, params

        model = lgbm.LGBMClassifier(**params, objective="multiclass", num_class=num_classes)
        return model, params

    def _objective(
        self,
        trial: optuna.Trial,
        train_data: Tuple[pd.DataFrame, pd.Series],
        valid_data: Tuple[pd.DataFrame, pd.Series],
        target_kind: str,
        num_classes: int,
        metric: str,
        categorical_cols: list,
    ) -> float:
        x_train, y_train = train_data
        x_valid, y_valid = valid_data
        model, _ = self._build_model(trial, target_kind, num_classes, metric)
        eval_metric = _eval_metric_name(metric, target_kind)
        callbacks = [lgbm.early_stopping(50, verbose=False)]
        report_cb = _optuna_report_callback(
            trial, eval_metric, metric, self.config.enable_pruning
        )
        if report_cb is not None:
            callbacks.append(report_cb)

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=eval_metric,
            categorical_feature=categorical_cols or "auto",
            callbacks=callbacks,
        )

        if target_kind == "numeric":
            preds = model.predict(x_valid)
            return _evaluate_metric(metric, target_kind, y_valid, preds, None)

        proba = model.predict_proba(x_valid)
        if target_kind == "binary":
            preds = (proba[:, 1] >= 0.5).astype(int)
        else:
            preds = np.argmax(proba, axis=1)
        return _evaluate_metric(metric, target_kind, y_valid, preds, proba)

    def run(self) -> Tuple[Dict[str, Any], optuna.Study]:
        config = self.config
        df = load_data(config.data_path)
        features, target, metadata = prepare_features(
            df,
            config.target_col,
            config.date_col,
            config.exclude_columns,
            config.categorical_columns,
        )

        target_kind = metadata["target_kind"]
        metric = _infer_metric(target_kind, config.metric)
        direction = _infer_direction(metric, config.direction)
        _validate_metric(metric, target_kind)
        num_classes = int(target.nunique()) if target_kind == "categorical" else 1
        if metric == "auc" and target_kind != "binary":
            raise ValueError("Metric 'auc' only supports binary targets.")

        x_train, x_valid, y_train, y_valid = _split_data(
            features,
            target,
            target_kind,
            config.test_size,
            config.random_state,
        )

        storage_uri = self._resolve_storage()
        pruner = None
        if config.enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_warmup_steps=config.prune_warmup_steps,
                interval_steps=config.prune_interval_steps,
            )
        study = optuna.create_study(
            study_name=config.study_name,
            direction=direction,
            storage=storage_uri,
            load_if_exists=True,
            pruner=pruner,
        )
        self.logger.info("Starting Optuna study %s", study.study_name)
        study.optimize(
            lambda trial: self._objective(
                trial,
                (x_train, y_train),
                (x_valid, y_valid),
                target_kind,
                num_classes,
                metric,
                metadata.get("categorical_cols", []),
            ),
            n_trials=config.n_trials,
            timeout=config.timeout,
        )

        trials = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
            for trial in study.trials
        ]
        trials_sorted = sorted(
            [t for t in trials if t["value"] is not None],
            key=lambda t: t["value"],
            reverse=(direction == "maximize"),
        )

        results = {
            "config": asdict(config),
            "metadata": metadata,
            "study": {
                "study_name": study.study_name,
                "direction": direction,
                "metric": metric,
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            "dataset": {
                "rows": int(features.shape[0]),
                "columns": int(features.shape[1]),
            },
            "trials": trials_sorted,
        }

        results_path = os.path.join(config.output_dir, "results.json")
        dump_json(results_path, results)
        self.logger.info("Results saved to %s", results_path)
        return results, study

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from .analysis import run
from .config import ShapConfig
from .utils import ensure_dir, optional_import, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM SHAP Analysis Tool")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to JSON or YAML config file")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to CSV dataset")
    parser.add_argument("--target", dest="target_col", default=None, help="Target column name")
    parser.add_argument("--date", dest="date_col", default=None, help="Date column name")
    parser.add_argument("--model", dest="model_path", default=None, help="LightGBM model path")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--sample-rows", dest="sample_rows", type=int, default=None)
    parser.add_argument("--shap-sample-rows", dest="shap_sample_rows", type=int, default=None)
    parser.add_argument("--train-ratio", dest="train_ratio", type=float, default=None)
    parser.add_argument("--val-ratio", dest="val_ratio", type=float, default=None)
    parser.add_argument("--test-ratio", dest="test_ratio", type=float, default=None)
    parser.add_argument("--gap-days", dest="gap_days", type=int, default=None)
    parser.add_argument("--train-date-range", dest="train_date_range", default=None)
    parser.add_argument("--val-date-range", dest="val_date_range", default=None)
    parser.add_argument("--test-date-range", dest="test_date_range", default=None)
    parser.add_argument("--force-index", dest="force_index", type=int, default=None)
    parser.add_argument("--shap-class-index", dest="shap_class_index", type=int, default=None)
    parser.add_argument("--interaction-sample-rows", dest="interaction_sample_rows", type=int, default=None)
    parser.add_argument("--dependence-plots", dest="dependence_plots", type=int, default=None)
    parser.add_argument("--low-importance-quantile", dest="low_importance_quantile", type=float, default=None)
    parser.add_argument("--low-importance-threshold", dest="low_importance_threshold", type=float, default=None)
    return parser


def _load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        yaml_mod, err = optional_import("yaml")
        if yaml_mod is None:
            raise RuntimeError(f"PyYAML not installed: {err}")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml_mod.safe_load(handle) or {}
    raise ValueError("Config file must be .json or .yaml/.yml")


def _parse_range(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Date ranges must be 'start,end'")
    return parts


def _build_config(args: argparse.Namespace) -> ShapConfig:
    config_data: Dict[str, Any] = {}
    if args.config_path:
        config_data = _load_config(args.config_path)

    def pick(key: str, default: Any) -> Any:
        override = getattr(args, key, None)
        if override is not None:
            return override
        return config_data.get(key, default)

    return ShapConfig(
        data_path=pick("data_path", None),
        target_col=pick("target_col", None),
        date_col=pick("date_col", None),
        model_path=pick("model_path", None),
        output_dir=pick("output_dir", "shap_output"),
        sample_rows=pick("sample_rows", 200000),
        shap_sample_rows=pick("shap_sample_rows", 10000),
        random_state=pick("random_state", 42),
        exclude_columns=pick("exclude_columns", None),
        categorical_columns=pick("categorical_columns", None),
        lgbm_params=pick("lgbm_params", None),
        train_ratio=pick("train_ratio", 0.7),
        val_ratio=pick("val_ratio", 0.15),
        test_ratio=pick("test_ratio", 0.15),
        train_date_range=pick("train_date_range", _parse_range(args.train_date_range)),
        val_date_range=pick("val_date_range", _parse_range(args.val_date_range)),
        test_date_range=pick("test_date_range", _parse_range(args.test_date_range)),
        gap_days=pick("gap_days", 0),
        shap_class_index=pick("shap_class_index", None),
        interaction_sample_rows=pick("interaction_sample_rows", 2000),
        top_features=pick("top_features", 20),
        dependence_plots=pick("dependence_plots", 6),
        low_importance_quantile=pick("low_importance_quantile", 0.1),
        low_importance_threshold=pick("low_importance_threshold", None),
        force_index=pick("force_index", 0),
        top_interactions=pick("top_interactions", 30),
        log_level=pick("log_level", "INFO"),
        log_path=pick("log_path", None),
    )


def run_analysis(args: argparse.Namespace) -> None:
    config = _build_config(args)

    if not config.data_path or not config.target_col:
        raise ValueError("--data and --target are required")

    ensure_dir(config.output_dir)
    log_path = config.log_path or os.path.join(config.output_dir, "run.log")
    setup_logger("shap_tool", config.log_level, log_path)

    run(config)
    print(f"SHAP outputs written to {config.output_dir}")


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()

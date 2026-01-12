from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from .analysis import OptunaTuner
from .config import OptunaConfig
from .report import generate_report
from .utils import ensure_dir, optional_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM Optuna Hyperparameter Tool")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to JSON or YAML config file")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to CSV dataset")
    parser.add_argument("--target", dest="target_col", default=None, help="Target column name")
    parser.add_argument("--date", dest="date_col", default=None, help="Date column name")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--trials", dest="n_trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--timeout", dest="timeout", type=int, default=None, help="Optuna timeout in seconds")
    parser.add_argument("--test-size", dest="test_size", type=float, default=None)
    parser.add_argument("--random-state", dest="random_state", type=int, default=None)
    parser.add_argument("--metric", dest="metric", default=None)
    parser.add_argument("--direction", dest="direction", default=None)
    parser.add_argument("--study-name", dest="study_name", default=None)
    parser.add_argument("--storage", dest="storage_path", default=None, help="Optuna storage path")
    parser.add_argument("--exclude", dest="exclude_columns", default=None, help="Comma-separated columns to drop")
    parser.add_argument("--categorical", dest="categorical_columns", default=None, help="Comma-separated categorical cols")
    parser.add_argument("--log-level", dest="log_level", default=None)
    parser.add_argument("--log-path", dest="log_path", default=None)
    parser.add_argument("--enable-pruning", action="store_true", help="Enable Optuna pruning")
    parser.add_argument("--prune-warmup", dest="prune_warmup_steps", type=int, default=None)
    parser.add_argument("--prune-interval", dest="prune_interval_steps", type=int, default=None)
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


def _split_list(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _build_config(args: argparse.Namespace) -> OptunaConfig:
    config_data: Dict[str, Any] = {}
    if args.config_path:
        config_data = _load_config(args.config_path)

    def pick(key: str, default: Any) -> Any:
        override = getattr(args, key, None)
        if override is not None:
            return override
        return config_data.get(key, default)

    exclude_columns = pick("exclude_columns", None)
    if isinstance(exclude_columns, str):
        exclude_columns = _split_list(exclude_columns)
    categorical_columns = pick("categorical_columns", None)
    if isinstance(categorical_columns, str):
        categorical_columns = _split_list(categorical_columns)

    return OptunaConfig(
        data_path=pick("data_path", None),
        target_col=pick("target_col", None),
        output_dir=pick("output_dir", "optuna_output"),
        date_col=pick("date_col", None),
        exclude_columns=exclude_columns,
        categorical_columns=categorical_columns,
        n_trials=pick("n_trials", 50),
        timeout=pick("timeout", None),
        test_size=pick("test_size", 0.2),
        random_state=pick("random_state", 42),
        metric=pick("metric", None),
        direction=pick("direction", None),
        study_name=pick("study_name", "lgbm_optuna"),
        storage_path=pick("storage_path", "optuna_output/study.db"),
        log_level=pick("log_level", "INFO"),
        log_path=pick("log_path", None),
        enable_pruning=pick("enable_pruning", False),
        prune_warmup_steps=pick("prune_warmup_steps", 10),
        prune_interval_steps=pick("prune_interval_steps", 1),
        param_ranges=pick("param_ranges", None),
    )


def run_tuning(args: argparse.Namespace) -> None:
    config = _build_config(args)
    if not config.data_path or not config.target_col:
        raise ValueError("--data and --target are required")

    ensure_dir(config.output_dir)
    tuner = OptunaTuner(config)
    results, study = tuner.run()

    report_path = os.path.join(config.output_dir, "report.html")
    generate_report(results, study, report_path)
    print(f"Report generated at {report_path}")


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tuning(args)


if __name__ == "__main__":
    main()

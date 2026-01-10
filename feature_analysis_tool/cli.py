from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from .analysis import FeatureAnalyzer
from .config import AnalysisConfig
from .debug_report import generate_debug_report
from .project import load_project, save_project
from .report import generate_report
from .utils import ensure_dir, optional_import, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM Feature Analysis Tool")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to JSON or YAML config file")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to CSV dataset")
    parser.add_argument("--target", dest="target_col", default=None, help="Target column name")
    parser.add_argument("--date", dest="date_col", default=None, help="Date column name")
    parser.add_argument("--model", dest="model_path", default=None, help="LightGBM model path")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--sample-rows", dest="sample_rows", type=int, default=None)
    parser.add_argument("--time-bucket", dest="time_bucket", default=None)
    parser.add_argument("--save-project", action="store_true", help="Save analysis outputs")
    parser.add_argument("--load-project", action="store_true", help="Load existing project outputs")
    parser.add_argument("--debug", action="store_true", help="Generate debug report")
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


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    config_data: Dict[str, Any] = {}
    if args.config_path:
        config_data = _load_config(args.config_path)

    def pick(key: str, default: Any) -> Any:
        override = getattr(args, key, None)
        if override is not None:
            return override
        return config_data.get(key, default)

    return AnalysisConfig(
        data_path=pick("data_path", None),
        target_col=pick("target_col", None),
        date_col=pick("date_col", None),
        model_path=pick("model_path", None),
        output_dir=pick("output_dir", "feature_analysis_output"),
        sample_rows=pick("sample_rows", 50000),
        time_bucket=pick("time_bucket", "M"),
        random_state=pick("random_state", 42),
        leakage_corr_threshold=pick("leakage_corr_threshold", 0.9),
        leakage_mi_quantile=pick("leakage_mi_quantile", 0.99),
        vif_threshold=pick("vif_threshold", 10.0),
        stability_psi_threshold=pick("stability_psi_threshold", 0.2),
        shap_sample_rows=pick("shap_sample_rows", 2000),
        exclude_columns=pick("exclude_columns", None),
        log_level=pick("log_level", "INFO"),
        log_path=pick("log_path", None),
        lgbm_params=pick("lgbm_params", None),
        top_corr_pairs=pick("top_corr_pairs", 25),
        categorical_columns=pick("categorical_columns", None),
        debug_report=pick("debug_report", False),
        debug_feature=pick("debug_feature", None),
        debug_symbol_col=pick("debug_symbol_col", None),
        debug_symbol=pick("debug_symbol", None),
        debug_output=pick("debug_output", "debug_report.html"),
        debug_sample_rows=pick("debug_sample_rows", 200000),
    )


def run_analysis(args: argparse.Namespace) -> None:
    config = _build_config(args)

    if args.load_project:
        output_dir = config.output_dir
        if not output_dir:
            raise ValueError("--output or config output_dir is required when using --load-project")
        results = load_project(output_dir)
        report_path = os.path.join(output_dir, "report.html")
        generate_report(results, report_path)
        print(f"Report generated at {report_path}")
        return

    if not config.data_path or not config.target_col:
        raise ValueError("--data and --target are required unless --load-project is set")

    ensure_dir(config.output_dir)
    log_path = config.log_path or os.path.join(config.output_dir, "run.log")
    setup_logger("feature_analysis", config.log_level, log_path)

    analyzer = FeatureAnalyzer(config)
    results, model = analyzer.run()

    if args.save_project:
        save_project(config.output_dir, results, model)

    report_path = os.path.join(config.output_dir, "report.html")
    generate_report(results, report_path)
    if args.debug:
        config.debug_report = True
        debug_path = generate_debug_report(config)
        if debug_path:
            print(f"Debug report generated at {debug_path}")
    print(f"Report generated at {report_path}")


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()

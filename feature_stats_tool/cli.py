from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .analysis import FeatureStatsAnalyzer, stats_table
from .config import StatsConfig
from .report import generate_report
from .utils import dump_json, ensure_dir, load_json, setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature statistics: distribution stats and histograms per feature")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to JSON or YAML config file")
    parser.add_argument("--data", dest="data_path", default=None, help="Path to CSV dataset")
    parser.add_argument("--target", dest="target_col", default=None, help="Target column name")
    parser.add_argument("--date", dest="date_col", default=None, help="Date column name")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--exclude", dest="exclude_columns", default=None, help="Comma-separated columns to skip")
    parser.add_argument("--categorical", dest="categorical_columns", default=None, help="Comma-separated categorical cols")
    parser.add_argument("--sample-rows", dest="sample_rows", type=int, default=None, help="Random row sample (default: all rows)")
    parser.add_argument("--bins", dest="hist_bins", type=int, default=None, help="Histogram bins")
    parser.add_argument("--hist-clip", dest="hist_clip_quantile", type=float, default=None,
                        help="Clip histogram range to [q, 1-q] quantiles, e.g. 0.005")
    parser.add_argument("--time-bucket", dest="time_bucket", default=None, help="Period for over-time charts (W, M, Q)")
    parser.add_argument("--log-level", dest="log_level", default=None)
    parser.add_argument("--log-path", dest="log_path", default=None)
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report.html from results.json in the output directory")
    return parser


def _load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    raise ValueError("Config file must be .json or .yaml/.yml")


def _split_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def build_config(args: argparse.Namespace) -> StatsConfig:
    data: Dict[str, Any] = _load_config(args.config_path) if args.config_path else {}
    defaults = StatsConfig(data_path="")

    def pick(key: str) -> Any:
        override = getattr(args, key, None)
        if override is not None:
            return override
        return data.get(key, getattr(defaults, key))

    known = StatsConfig.__dataclass_fields__.keys()
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = {key: pick(key) for key in known}
    values["exclude_columns"] = _split_list(values["exclude_columns"])
    values["categorical_columns"] = _split_list(values["categorical_columns"])
    return StatsConfig(**values)


def run(args: argparse.Namespace) -> str:
    config = build_config(args)
    ensure_dir(config.output_dir)
    results_path = os.path.join(config.output_dir, "results.json")
    report_path = os.path.join(config.output_dir, "report.html")

    if args.report_only:
        generate_report(load_json(results_path), report_path)
        print(f"Report generated at {report_path}")
        return report_path

    if not config.data_path:
        raise ValueError("--data or config data_path is required")
    log_path = config.log_path or os.path.join(config.output_dir, "run.log")
    logger = setup_logger("feature_stats", config.log_level, log_path)

    results = FeatureStatsAnalyzer(config, logger).run()
    dump_json(results_path, results)
    stats_table(results).to_csv(os.path.join(config.output_dir, "stats.csv"), index=False)
    generate_report(results, report_path)
    logger.info("Results saved to %s", results_path)
    print(f"Report generated at {report_path}")
    return report_path


def main(argv: Optional[list] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

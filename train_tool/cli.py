from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from .config import load_config
from .trainer import rescore, run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train LightGBM with P123 AI Factor-style validation (rolling time series CV) "
        "and report RMSE, Spearman and quantile stats per run and averaged over runs."
    )
    p.add_argument("--config", required=True, help="Path to train_config.json")
    p.add_argument("--data", dest="data_path", help="Override data_path")
    p.add_argument("--output", dest="output_dir", help="Override output_dir")
    p.add_argument("--runs", type=int, help="Runs per model (different seeds)")
    p.add_argument("--folds", type=int)
    p.add_argument("--training-years", dest="training_years", type=float)
    p.add_argument("--gap-weeks", dest="gap_weeks", type=int)
    p.add_argument("--quantiles", type=int, help="Number of quantiles for High/Low stats")
    p.add_argument(
        "--report-only", action="store_true",
        help="Don't train: rescore the saved predictions (use with --quantiles to change buckets)",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {k: getattr(args, k) for k in
                 ("data_path", "output_dir", "runs", "folds", "training_years", "gap_weeks", "quantiles")}
    cfg = load_config(args.config, overrides)
    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(cfg.output_dir, "run.log"))],
    )
    if args.report_only:
        text = rescore(cfg, cfg.quantiles)
    else:
        text = run_training(cfg)
    print(text)
    print(f"Results in {cfg.output_dir}")


if __name__ == "__main__":
    main()

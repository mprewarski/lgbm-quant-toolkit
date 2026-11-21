"""Train LightGBM models the way P123 AI Factor validation does, and score the holdouts."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import html_report, metrics
from .config import ModelSpec, TrainConfig
from .folds import Fold, assign_rows, build_folds, holdout_months

log = logging.getLogger("train_tool")

SEED_KEYS = ("random_state", "seed", "random_seed")
RET_COL = "__ret"


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", name).strip("_")


# --------------------------------------------------------------------------- data
def load_dataset(cfg: TrainConfig) -> Tuple[pd.DataFrame, List[str]]:
    log.info("Loading %s", cfg.data_path)
    df = pd.read_csv(cfg.data_path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

    first, last = df[cfg.date_col].min(), df[cfg.date_col].max()
    start = pd.Timestamp(cfg.dataset_start) if cfg.dataset_start else first
    end = pd.Timestamp(cfg.dataset_end) if cfg.dataset_end else last
    if first - start > pd.Timedelta(days=14):
        log.warning(
            "CSV starts %s but dataset_start is %s: the first folds get less training data than in P123.",
            first.date(), start.date(),
        )
    if end - last > pd.Timedelta(days=14):
        log.warning("CSV ends %s but dataset_end is %s.", last.date(), end.date())
    df = df[(df[cfg.date_col] >= start) & (df[cfg.date_col] <= end)]

    reserved = {cfg.date_col, cfg.target_col, *cfg.exclude_columns}
    if cfg.returns_col and cfg.returns_path:
        log.info("Joining %s from %s", cfg.returns_col, cfg.returns_path)
        rets = pd.read_csv(cfg.returns_path, usecols=[cfg.date_col, cfg.id_col, cfg.returns_col])
        rets[cfg.date_col] = pd.to_datetime(rets[cfg.date_col])
        df = df.drop(columns=[cfg.returns_col], errors="ignore").merge(
            rets.drop_duplicates([cfg.date_col, cfg.id_col]), on=[cfg.date_col, cfg.id_col], how="left")
        missing_pct = df[cfg.returns_col].isna().mean() * 100
        if missing_pct > 5:
            log.warning("%.1f%% of rows have no return in %s", missing_pct, cfg.returns_path)
    if cfg.returns_col:
        if cfg.returns_col not in df.columns:
            raise ValueError(f"returns_col {cfg.returns_col!r} not in dataset")
        reserved.update({cfg.returns_col, RET_COL})
        # decimal period return, back from P123's normalization
        df[RET_COL] = (df[cfg.returns_col] - cfg.returns_zero) / cfg.returns_per_pct / 100
    features = cfg.features or [c for c in df.columns if c not in reserved]
    missing = [c for c in features + cfg.drop_features + cfg.categorical_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in dataset: {missing}")
    features = [c for c in features if c not in set(cfg.drop_features)]
    leaks = {cfg.target_col, cfg.returns_col, RET_COL} & set(features)
    if leaks:
        raise ValueError(f"Target/return columns can't be features: {sorted(c for c in leaks if c)}")

    n0 = len(df)
    df = df[df[cfg.target_col].notna()]
    na_pct = df[features].isna().mean(axis=1) * 100
    df = df[na_pct <= cfg.max_na_pct].copy()
    log.info(
        "%d rows, %d features, %s to %s (dropped %d rows: no target or > %.0f%% NA features)",
        len(df), len(features), start.date(), end.date(), n0 - len(df), cfg.max_na_pct,
    )

    for col in cfg.categorical_columns:
        df[col] = pd.Categorical(df[col]).codes.astype("int32")  # NA -> -1, LightGBM treats as missing
    num = [c for c in features if c not in cfg.categorical_columns]
    if cfg.clip is not None:
        df[num] = df[num].clip(-cfg.clip, cfg.clip)
        df[cfg.target_col] = df[cfg.target_col].clip(-cfg.clip, cfg.clip)
        log.info("Clipped features and %s at +-%g", cfg.target_col, cfg.clip)
    if cfg.na_fill == "zero":
        df[num] = df[num].fillna(0.0)
    elif isinstance(cfg.na_fill, (int, float)) and not isinstance(cfg.na_fill, bool):
        df[num] = df[num].fillna(float(cfg.na_fill))
    elif cfg.na_fill == "median":
        df[num] = df[num].fillna(df.groupby(cfg.date_col)[num].transform("median")).fillna(0.0)
    elif cfg.na_fill != "none":
        raise ValueError("na_fill must be 'zero', 'median', 'none' or a number")
    df[num] = df[num].astype("float32")
    df.attrs["period"] = (start, end)
    return df.reset_index(drop=True), features


# --------------------------------------------------------------------------- training
def _seeds(cfg: TrainConfig) -> List[int]:
    if cfg.seeds:  # an explicit seed list sets the number of runs
        return list(cfg.seeds)
    return [cfg.base_seed + i for i in range(cfg.runs)]


def train_run(
    df: pd.DataFrame, features: List[str], folds: List[Fold], spec: ModelSpec, seed: int, cfg: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One run = one model per fold. Returns concatenated holdout predictions and importances."""
    params = {k: v for k, v in spec.params.items() if k not in SEED_KEYS}
    params.update(random_state=seed, n_jobs=cfg.n_jobs, verbose=-1)
    dates = df[cfg.date_col]
    # LightGBM rejects JSON characters in feature names (P123 names have quotes, colons...),
    # so train on arrays with positional names and map importances back to the real names.
    safe_names = [f"f{i}" for i in range(len(features))]
    cat_idx = [features.index(c) for c in cfg.categorical_columns if c in features]
    preds, imps = [], []
    for i, fold in enumerate(folds):
        tr, ho = assign_rows(dates, fold, is_last=i == len(folds) - 1)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            df.loc[tr, features].set_axis(safe_names, axis=1), df.loc[tr, cfg.target_col].to_numpy(),
            categorical_feature=cat_idx or "auto",
        )
        part = pd.DataFrame({
            metrics.DATE: dates[ho].to_numpy(),
            metrics.ID: df.loc[ho, cfg.id_col].to_numpy(),
            "fold": fold.fold,
            metrics.Y: df.loc[ho, cfg.target_col].to_numpy(),
            metrics.P: model.predict(df.loc[ho, features].set_axis(safe_names, axis=1)),
        })
        if RET_COL in df.columns:
            part[metrics.R] = df.loc[ho, RET_COL].to_numpy()
        preds.append(part)
        imps.append(pd.DataFrame({
            "fold": fold.fold, "feature": features,
            "gain": model.booster_.feature_importance("gain"),
            "splits": model.booster_.feature_importance("split"),
        }))
    return pd.concat(preds, ignore_index=True), pd.concat(imps, ignore_index=True)


# --------------------------------------------------------------------------- results
def _bucket_rows(model, run, res) -> List[dict]:
    """Per-quantile rows: mean TARGET and, with returns, portfolio Avg%, Cmp%, Turn%."""
    rows = [{"model": model, "run": run, "stat": "target_mean",
             **{f"q{int(k)}": v for k, v in res.pop("target_buckets").items()}}]
    res.pop("weekly", None)
    table = res.pop("portfolio_buckets", None)
    if table is not None:
        for stat in ("avg", "cmp", "sd", "turn"):
            rows.append({"model": model, "run": run, "stat": f"portfolio_{stat}",
                         **{f"q{int(k)}": v for k, v in table[stat].items()}})
    return rows


def score_runs(pred_sets: Dict[Tuple[str, int, int], pd.DataFrame], n_quantiles: int, slippage_pct: float):
    """Stats per run, the mean/std over runs per model, an ensemble, and per-quantile tables."""
    run_rows, q_rows = [], []
    for (model, run, seed), pred in pred_sets.items():
        res = metrics.evaluate(pred, n_quantiles, slippage_pct)
        q_rows += _bucket_rows(model, run, res)
        run_rows.append({"model": model, "run": run, "seed": seed, **res})
    runs = pd.DataFrame(run_rows)
    quant = pd.DataFrame(q_rows)

    stat_cols = [c for c in runs.columns if c not in ("model", "run", "seed")]
    mean = runs.groupby("model", sort=False)[stat_cols].mean()
    std = runs.groupby("model", sort=False)[stat_cols].std()
    summary = mean.join(std, rsuffix="_std")
    summary.insert(0, "runs", runs.groupby("model", sort=False).size())

    # Ensemble: average the runs' predictions per model, then score once.
    ens_rows = []
    by_model: Dict[str, List[pd.DataFrame]] = {}
    for (model, _run, _seed), pred in pred_sets.items():
        by_model.setdefault(model, []).append(pred)
    for model, preds in by_model.items():
        if len(preds) < 2:
            continue
        avg = preds[0].copy()
        avg[metrics.P] = np.mean([p[metrics.P].to_numpy() for p in preds], axis=0)
        res = metrics.evaluate(avg, n_quantiles, slippage_pct)
        q_rows_e = _bucket_rows(model, "ensemble", res)
        quant = pd.concat([quant, pd.DataFrame(q_rows_e)], ignore_index=True)
        ens_rows.append({"model": model, "runs": len(preds), **res})
    ensemble = pd.DataFrame(ens_rows)

    per_run = quant[quant["run"] != "ensemble"]
    q_mean = per_run.groupby(["model", "stat"], sort=False).mean(numeric_only=True).drop(columns="run", errors="ignore")
    q_mean.insert(0, "run", "mean")
    quant = pd.concat([quant, q_mean.reset_index()], ignore_index=True)
    return runs, summary.reset_index(), ensemble, quant


def write_results(out_dir: str, runs, summary, ensemble, quant, n_quantiles: int) -> str:
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    if len(ensemble):
        ensemble.to_csv(os.path.join(out_dir, "ensemble.csv"), index=False)
    quant.to_csv(os.path.join(out_dir, f"quantiles_{n_quantiles}.csv"), index=False)
    text = format_report(runs, summary, ensemble, quant, n_quantiles)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


# (key, header, decimals). Portfolio columns follow P123's Compare All table.
ACC_COLS = [("rmse", "RMSE", 4), ("spearman", "Spearman", 4), ("pearson", "Pearson", 4), ("ic_mean", "IC/date", 4)]
PORT_COLS = [("hl_avg", "H-L Avg%", 2), ("high_avg", "Hi Avg%", 2), ("high_sr", "Hi SR", 2),
             ("high_turn", "Hi Turn%", 0), ("low_avg", "Lo Avg%", 2), ("low_sr", "Lo SR", 2),
             ("low_sd", "Lo SD", 2), ("low_turn", "Lo Turn%", 0)]
TGT_COLS = [("tgt_high", "TgtHigh", 3), ("tgt_low", "TgtLow", 3), ("tgt_hl", "TgtH-L", 3)]


def _table(df: pd.DataFrame, label_cols: List[str], cols) -> str:
    cols = [c for c in cols if c[0] in df.columns]
    head = [*label_cols, *[c[1] for c in cols]]
    rows = [[str(r[c]) for c in label_cols] + [f"{r[k]:,.{d}f}" for k, _, d in cols] for _, r in df.iterrows()]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(head)]
    fmt = lambda vals: "  ".join(v.ljust(w) if i < len(label_cols) else v.rjust(w)  # noqa: E731
                                 for i, (v, w) in enumerate(zip(vals, widths)))
    return "\n".join([fmt(head), fmt(["-" * w for w in widths]), *[fmt(r) for r in rows]])


def format_report(runs, summary, ensemble, quant, n_quantiles: int) -> str:
    has_port = "high_avg" in runs.columns
    cols = ACC_COLS + (PORT_COLS if has_port else []) + TGT_COLS
    note = (f"Quantiles: {n_quantiles} (High = bucket {n_quantiles}, Low = bucket 1). "
            + ("Avg% etc.: equal-weighted bucket portfolios rebalanced every date, annualized, after slippage. "
               if has_port else "No returns_col: only TARGET-based quantile stats. ")
            + "Tgt*: mean TARGET per bucket (TARGET units).")
    parts = [note, "", "Each run", _table(runs, ["model", "run", "seed"], cols)]
    if len(runs) > len(summary):
        parts += ["", "Average of runs (std in summary.csv)", _table(summary, ["model", "runs"], cols)]
    if len(ensemble):
        parts += ["", "Ensemble (runs' predictions averaged, then scored)", _table(ensemble, ["model", "runs"], cols)]
    if has_port:
        mean_rows = quant[(quant["run"].astype(str) == "mean") & (quant["stat"] == "portfolio_avg")]
        qcols = [f"q{i}" for i in range(1, n_quantiles + 1)]
        q = mean_rows[["model", *qcols]].copy()
        parts += ["", "Annualized Avg% by quantile (mean of runs)",
                  _table(q, ["model"], [(c, c, 2) for c in qcols])]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- entry points
def run_training(cfg: TrainConfig) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    pred_dir = os.path.join(cfg.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    df, features = load_dataset(cfg)
    start, end = df.attrs["period"]
    folds = build_folds(start, end, cfg.validation_method, cfg.folds, cfg.training_years, cfg.gap_weeks)
    fold_rows = []
    for i, f in enumerate(folds):
        tr, ho = assign_rows(df[cfg.date_col], f, is_last=i == len(folds) - 1)
        fold_rows.append({**{k: (v.date() if hasattr(v, "date") else v) for k, v in asdict(f).items()},
                          "train_rows": int(tr.sum()), "holdout_rows": int(ho.sum()),
                          "train_dates": int(df.loc[tr, cfg.date_col].nunique()),
                          "holdout_dates": int(df.loc[ho, cfg.date_col].nunique())})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(cfg.output_dir, "folds.csv"), index=False)
    log.info("Holdout period %.1f months per fold\n%s", holdout_months(folds), fold_df.to_string(index=False))

    pred_sets: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    imps = []
    seeds = _seeds(cfg)
    for spec in cfg.models:
        for run, seed in enumerate(seeds, start=1):
            t0 = time.time()
            pred, imp = train_run(df, features, folds, spec, seed, cfg)
            pred_sets[(spec.name, run, seed)] = pred
            imps.append(imp.assign(model=spec.name, run=run))
            if cfg.save_predictions:
                pred.to_csv(os.path.join(pred_dir, f"{slug(spec.name)}__run{run}__seed{seed}.csv.gz"), index=False)
            sc = metrics.accuracy(pred[metrics.Y], pred[metrics.P])  # quick progress line
            log.info("%s run %d (seed %d): RMSE %.4f Spearman %.4f [%.0fs]",
                     spec.name, run, seed, sc["rmse"], sc["spearman"], time.time() - t0)

    imp_all = pd.concat(imps, ignore_index=True)
    imp_all.to_csv(os.path.join(cfg.output_dir, "feature_importance.csv"), index=False)
    text = write_results(cfg.output_dir, *score_runs(pred_sets, cfg.quantiles, cfg.slippage_pct), cfg.quantiles)
    write_html_report(cfg, pred_sets, fold_df, imp_all)
    return text


def write_html_report(cfg: TrainConfig, pred_sets, folds=None, importance=None) -> None:
    if not cfg.html_report:
        return
    path = os.path.join(cfg.output_dir, "report.html")
    payload = html_report.build_payload(pred_sets, cfg, cfg.report_quantiles, folds, importance)
    title = "Training report: " + ", ".join(dict.fromkeys(m["name"] for m in payload["models"]))
    html_report.write_html(path, payload, title)
    log.info("HTML report: %s", path)


def rescore(cfg: TrainConfig, n_quantiles: int) -> str:
    """Recompute stats from saved predictions (e.g. a different number of quantiles)."""
    pred_dir = os.path.join(cfg.output_dir, "predictions")
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"No saved predictions in {pred_dir}; run training first")
    names = {slug(m.name): m.name for m in cfg.models}
    pred_sets = {}
    for fname in sorted(os.listdir(pred_dir)):
        m = re.match(r"(.+)__run(\d+)__seed(-?\d+)\.csv\.gz$", fname)
        if not m:
            continue
        pred = pd.read_csv(os.path.join(pred_dir, fname), parse_dates=[metrics.DATE])
        pred_sets[(names.get(m.group(1), m.group(1)), int(m.group(2)), int(m.group(3)))] = pred
    if not pred_sets:
        raise FileNotFoundError(f"No prediction files in {pred_dir}")
    order = {m.name: i for i, m in enumerate(cfg.models)}
    pred_sets = dict(sorted(pred_sets.items(), key=lambda kv: (order.get(kv[0][0], len(order)), kv[0][1])))
    text = write_results(cfg.output_dir, *score_runs(pred_sets, n_quantiles, cfg.slippage_pct), n_quantiles)
    extra = {}
    for name, fname in (("folds", "folds.csv"), ("importance", "feature_importance.csv")):
        path = os.path.join(cfg.output_dir, fname)
        extra[name] = pd.read_csv(path) if os.path.exists(path) else None
    write_html_report(cfg, pred_sets, extra["folds"], extra["importance"])
    return text

"""Recover the percent scale of a normalized weekly-return column (e.g. Future%Chg(5)).

With Dataset-scope normalization (MinMax or Z-score over the whole dataset), P123 maps the
raw return linearly: value = zero + per_pct * return%. Fitting value against the real
return of a couple of stocks gives `returns_zero` and `returns_per_pct` for the config.

P123 dataset dates are Saturdays. Fitted against real prices (R2 0.9999):
- Future%Chg(N): close of the next trading day to N trading days (bars) later.
- Future%Chg_D(N): counts weekdays, holidays included (FutureRel%Chg_D(190) leaves exactly
  38 weeks without a target). Close as of the Monday after the date to close as of N weekdays
  later; a holiday takes the previous trading day's close.
P123's weekly portfolios run from the first trading day after one Saturday to the first
trading day after the next. Both columns match that except in holiday weeks, where each is
one day off (in about 20% of weeks); the difference is random, not a bias.

Usage (fetches prices with the P123 API, 1 call per stock, cached in data/):
    python -m train_tool.calibrate --config models/sp500/train_config.json --uids 4737 5881
Add --write to store the result in the config.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import load_config


def fetch_prices(uid: int, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, f"calib_prices_{uid}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["dt"])
    from p123_client import get_client  # project-root module; uses the account's API credits

    with get_client() as client:
        prices = client.data_prices(str(uid), start, end, to_pandas=True)
    prices.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["dt"])


def fit(values: pd.DataFrame, prices: pd.DataFrame, horizon: int, date_col: str, col: str,
        weekdays: bool = False) -> Tuple[float, float, float, int]:
    """Least-squares fit value = zero + per_pct * return%. Returns zero, per_pct, R2, points."""
    prices = prices.sort_values("dt").reset_index(drop=True)
    close, days = prices["close"].to_numpy(), prices["dt"].to_numpy()
    dates = pd.to_datetime(values[date_col]).to_numpy()
    base = np.searchsorted(days, dates, side="right")  # next trading day
    if weekdays:
        # _D functions count weekdays and price each one at the last close on or before it
        # (a holiday takes the previous close): Monday after the date -> N weekdays later.
        d = dates.astype("datetime64[D]")
        tdays = days.astype("datetime64[D]")
        base = np.searchsorted(tdays, np.busday_offset(d, 0, roll="forward"), side="right") - 1
        end = np.searchsorted(tdays, np.busday_offset(d, horizon, roll="forward"), side="right") - 1
    else:
        end = base + horizon
    ok = end < len(close)
    ret = (close[end[ok]] / close[base[ok]] - 1) * 100
    val = values[col].to_numpy()[ok]
    per_pct, zero = np.polyfit(ret, val, 1)
    resid = val - (zero + per_pct * ret)
    r2 = 1 - np.sum(resid**2) / np.sum((val - val.mean()) ** 2)
    return float(zero), float(per_pct), float(r2), int(ok.sum())


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--uids", type=int, nargs="+", required=True, help="P123 IDs of 1-3 long-lived stocks")
    ap.add_argument("--write", action="store_true", help="Save returns_zero / returns_per_pct in the config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.returns_col:
        raise SystemExit("Set returns_col in the config first")
    horizon = int(re.search(r"\((\d+)", cfg.returns_col).group(1)) if "(" in cfg.returns_col else 5
    weekdays = "_D(" in cfg.returns_col

    source = cfg.returns_path or cfg.data_path  # the file that holds returns_col
    df = pd.read_csv(source, usecols=[cfg.date_col, cfg.id_col, cfg.returns_col])
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    col = df[cfg.returns_col].dropna()
    n_pinned = max((col == col.min()).sum(), (col == col.max()).sum())
    pinned = n_pinned / max(len(col), 1) * 100
    if pinned > 1 and n_pinned > 20:  # one stock at the extreme is normal; a pile-up is clipping
        print(f"WARNING: {pinned:.1f}% of {cfg.returns_col} values sit exactly at the column's min or "
              f"max ({col.min():g} / {col.max():g}): the download clipped outliers, so the biggest weekly "
              "moves are lost and portfolio returns will be wrong. Re-download the returns with outlier "
              "clipping off (MinMax, Dataset scope).\n")
    start = (df[cfg.date_col].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (df[cfg.date_col].max() + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    fits = []
    for uid in args.uids:
        vals = df[(df[cfg.id_col] == uid) & df[cfg.returns_col].notna()]
        if vals.empty:
            raise SystemExit(f"P123 ID {uid} not in {source}")
        prices = fetch_prices(uid, start, end, os.path.dirname(cfg.data_path) or ".")
        zero, per_pct, r2, n = fit(vals, prices, horizon, cfg.date_col, cfg.returns_col, weekdays)
        fits.append((zero, per_pct))
        print(f"P123 ID {uid}: zero={zero:.5f} per_pct={per_pct:.6f} R2={r2:.5f} ({n} weeks)")
        if r2 < 0.99:
            print("  warning: R2 below 0.99 - the column may not be Dataset-scope normalized")
    zero, per_pct = (float(np.mean(v)) for v in zip(*fits))
    print(f'\n"returns_zero": {zero:.5f}, "returns_per_pct": {per_pct:.6f}')
    print(f"i.e. return % = (value - {zero:.4f}) / {per_pct:.5f}")

    if args.write:
        with open(args.config, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["returns_zero"], raw["returns_per_pct"] = round(zero, 5), round(per_pct, 6)
        with open(args.config, "w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2)
            handle.write("\n")
        print(f"Saved to {args.config}")


if __name__ == "__main__":
    main()

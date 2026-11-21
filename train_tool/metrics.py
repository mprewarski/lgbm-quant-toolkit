"""Result statistics for holdout predictions (P123 Results tab equivalents).

Accuracy stats (RMSE, R2, Pearson, Spearman) are pooled over every holdout row of every
fold, as on P123's Lift tab.

Quantiles: on each date the predictions are split into N buckets (bucket N = highest
predictions = "High", bucket 1 = "Low").

* Portfolio stats (needs a weekly return per stock, see `returns_col` in the config): each
  bucket is an equal-weighted portfolio rebalanced every date and held one period, like
  P123's Portfolio/Return tabs. Avg% = mean weekly return compounded over a year (52.18
  weeks), Cmp% = compounded annual return, SD = annualized standard deviation, SR =
  (mean weekly x 52.18) / SD, Turn% = annualized one-way turnover. Slippage is charged on
  every buy and sell. These match P123's Compare All numbers computed from its own
  Portfolio download.
* TARGET stats (always available): the mean TARGET of each bucket, averaged over dates, in
  TARGET units. Useful when the dataset has no weekly return column.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DATE, ID, Y, P, R = "Date", "id", "target", "pred", "ret"


def accuracy(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    resid = y - p
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": float(1 - np.sum(resid**2) / sst) if sst > 0 else float("nan"),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "spearman": float(stats.spearmanr(y, p)[0]),
    }


def assign_quantiles(df: pd.DataFrame, n: int) -> pd.Series:
    """Bucket 1..n by prediction within each date (n = highest predictions)."""
    pct = df.groupby(DATE)[P].rank(method="first", pct=True)
    return np.ceil(pct * n).clip(1, n).astype(int)


def periods_per_year(dates) -> float:
    uniq = np.sort(pd.to_datetime(pd.Series(dates).unique()))
    if len(uniq) < 2:
        return float("nan")
    step = np.median(np.diff(uniq)).astype("timedelta64[D]").astype(float)
    return 365.25 / step  # 52.18 for weekly data, as P123 uses


def bucket_turnover(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """One-way turnover per date and bucket: share of the bucket's names that are new.
    The first date (initial purchase) is NaN."""
    out = {}
    for b in range(1, n + 1):
        members = df[df["q"] == b].groupby(DATE)[ID].apply(frozenset).sort_index()
        vals = [np.nan] + [len(cur - prev) / max(len(cur), 1)
                           for prev, cur in zip(members.iloc[:-1], members.iloc[1:])]
        out[b] = pd.Series(vals, index=members.index)
    return pd.DataFrame(out)


def _annualize(weekly: pd.Series, ppy: float) -> Tuple[float, float, float, float]:
    """P123 conventions (checked against a P123 Portfolio download): Avg = mean period return
    compounded over a year, Cmp = compounded actual growth, SD = std x sqrt(ppy),
    SR = arithmetic annual mean / SD."""
    weekly = weekly.dropna()
    if not len(weekly):
        return np.nan, np.nan, np.nan, np.nan
    mean = weekly.mean()
    avg = (1 + mean) ** ppy - 1
    cmp_ = np.prod(1 + weekly) ** (ppy / len(weekly)) - 1
    sd = weekly.std() * np.sqrt(ppy)
    return avg * 100, cmp_ * 100, sd * 100, (mean * ppy / sd if sd > 0 else np.nan)


def portfolio_stats(df: pd.DataFrame, n: int, slippage_pct: float) -> Dict[str, object]:
    """Weekly-rebalanced, equal-weighted bucket portfolios (needs df[R] = period return)."""
    ppy = periods_per_year(df[DATE])
    gross = df.groupby([DATE, "q"])[R].mean().unstack("q").sort_index()
    turn = bucket_turnover(df, n).reindex(gross.index)
    # each replaced name is one sell and one buy
    net = gross - 2 * turn.fillna(0) * slippage_pct / 100

    table = {}
    for b in range(1, n + 1):
        avg, cmp_, sd, sr = _annualize(net[b], ppy)
        table[b] = {"avg": avg, "cmp": cmp_, "sd": sd, "sr": sr,
                    "turn": float(turn[b].mean() * ppy * 100),
                    "stocks": float(df[df["q"] == b].groupby(DATE).size().mean())}
    table = pd.DataFrame(table).T
    univ = df.groupby(DATE)[R].mean()
    hl_avg, _, hl_sd, hl_sr = _annualize(net[n] - net[1], ppy)
    out: Dict[str, object] = {
        "hl_avg": hl_avg, "hl_sr": hl_sr,
        # weekly net returns, for charts
        "weekly": pd.DataFrame({"high": net[n], "low": net[1], "univ": univ}),
        "univ_avg": _annualize(univ, ppy)[0], "univ_cmp": _annualize(univ, ppy)[1],
        "portfolio_buckets": table,
    }
    for side, b in (("high", n), ("low", 1)):
        for k in ("avg", "cmp", "sd", "sr", "turn"):
            out[f"{side}_{k}"] = float(table.loc[b, k])
    return out


def target_quantile_stats(df: pd.DataFrame, n: int) -> Dict[str, object]:
    by_date = df.groupby([DATE, "q"])[Y].mean().unstack("q")
    hl = by_date[n] - by_date[1]
    return {
        "tgt_high": float(by_date[n].mean()),
        "tgt_low": float(by_date[1].mean()),
        "tgt_hl": float(hl.mean()),
        "tgt_hl_ir": float(hl.mean() / hl.std()) if hl.std() > 0 else float("nan"),
        "target_buckets": by_date.mean(),
    }


def per_date_ic(df: pd.DataFrame) -> Dict[str, float]:
    ic = df.groupby(DATE).apply(lambda g: g[Y].corr(g[P], method="spearman"), include_groups=False)
    return {"ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan")}


def evaluate(df: pd.DataFrame, n_quantiles: int, slippage_pct: float = 0.1) -> Dict[str, object]:
    """All stats for one run's concatenated holdout predictions."""
    df = df.assign(q=assign_quantiles(df, n_quantiles))
    out: Dict[str, object] = {}
    out.update(accuracy(df[Y].to_numpy(), df[P].to_numpy()))
    out.update(per_date_ic(df))
    if R in df.columns and df[R].notna().any():
        out.update(portfolio_stats(df, n_quantiles, slippage_pct))
    out.update(target_quantile_stats(df, n_quantiles))
    return out

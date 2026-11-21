"""Validation fold schedule, following P123 AI Factor's validation methods.

P123 Rolling Time Series CV (Validation > Method):
    holdout length L = (dataset period - training period - gap) / folds
    fold i (0-based): train  [start + i*L,               start + i*L + training)
                      gap    training end + gap weeks
                      holdout[start + training + gap + i*L, ... + L)
The holdouts are back to back and the last one ends at the dataset end, which is how P123
shows "Holdout Period 13.5 months" for 20.9 years, 16 training years, 20-week gap, 4 folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

DAYS_PER_YEAR = 365.25


@dataclass
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp  # exclusive, except the last fold which includes the dataset end


def build_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    method: str = "rolling",
    folds: int = 4,
    training_years: float = 16.0,
    gap_weeks: int = 20,
) -> List[Fold]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    training = pd.Timedelta(days=training_years * DAYS_PER_YEAR)
    gap = pd.Timedelta(weeks=gap_weeks)
    if method == "holdout":
        folds = 1
    holdout_total = end - start - training - gap
    if holdout_total <= pd.Timedelta(0):
        raise ValueError(
            f"Training period ({training_years} y) + gap ({gap_weeks} w) is longer than the dataset "
            f"period {start.date()} to {end.date()}"
        )
    length = holdout_total / folds

    out: List[Fold] = []
    for i in range(folds):
        h_start = start + training + gap + i * length
        h_end = h_start + length
        t_end = h_start - gap
        t_start = start if method in ("expanding", "holdout") else start + i * length
        out.append(Fold(i + 1, t_start, t_end, h_start, h_end))
    return out


def holdout_months(fold_list: List[Fold]) -> float:
    span = fold_list[0].holdout_end - fold_list[0].holdout_start
    return span.days / DAYS_PER_YEAR * 12


def assign_rows(dates: pd.Series, fold: Fold, is_last: bool):
    """Boolean masks (train, holdout) for one fold."""
    train = (dates >= fold.train_start) & (dates < fold.train_end)
    if is_last:
        holdout = (dates >= fold.holdout_start) & (dates <= fold.holdout_end)
    else:
        holdout = (dates >= fold.holdout_start) & (dates < fold.holdout_end)
    return train, holdout

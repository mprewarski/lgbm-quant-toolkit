"""Checks that the Optuna tool's time-based split doesn't leak future data."""
import numpy as np
import pandas as pd
import pytest

from optuna_tool.analysis import _split_data


def _panel(n_weeks=200, n_stocks=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-07", periods=n_weeks, freq="W-FRI")
    df = pd.DataFrame(
        {
            "Date": np.repeat(dates, n_stocks),
            "f1": rng.normal(size=n_weeks * n_stocks),
            "TARGET": rng.normal(size=n_weeks * n_stocks),
        }
    )
    return df[["f1"]], df["TARGET"], df["Date"]


def test_time_split_orders_and_gaps():
    x, y, dates = _panel()
    x_tr, x_va, y_tr, y_va = _split_data(x, y, "numeric", 0.2, 42, dates, gap_days=270, split_method="time")
    train_max = dates.loc[x_tr.index].max()
    valid_min = dates.loc[x_va.index].min()
    assert train_max < valid_min
    assert (valid_min - train_max).days >= 270
    # Whole dates go to one side; no date is split across train/validation.
    assert set(dates.loc[x_tr.index]).isdisjoint(set(dates.loc[x_va.index]))
    assert len(y_tr) == len(x_tr) and len(y_va) == len(x_va)


def test_time_split_requires_dates():
    x, y, _ = _panel()
    with pytest.raises(ValueError):
        _split_data(x, y, "numeric", 0.2, 42, None, split_method="time")


def test_gap_too_large_raises():
    x, y, dates = _panel(n_weeks=20)
    with pytest.raises(ValueError):
        _split_data(x, y, "numeric", 0.2, 42, dates, gap_days=5000, split_method="time")


def test_random_split_still_available():
    x, y, dates = _panel()
    x_tr, x_va, _, _ = _split_data(x, y, "numeric", 0.2, 42, dates, split_method="random")
    assert len(x_va) == pytest.approx(0.2 * len(x), rel=0.01)

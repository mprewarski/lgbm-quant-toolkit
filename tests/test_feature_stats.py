"""Tests for feature_stats_tool on a small synthetic stock panel."""
import json

import numpy as np
import pandas as pd
import pytest

from feature_stats_tool.analysis import FeatureStatsAnalyzer
from feature_stats_tool.cli import main
from feature_stats_tool.config import StatsConfig


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    n_dates, n_stocks = 60, 40
    dates = np.repeat(pd.date_range("2010-01-01", periods=n_dates, freq="W-SAT"), n_stocks)
    n = n_dates * n_stocks
    clipped = np.clip(rng.normal(size=n) * 3, -2.5, 2.5)
    missing = rng.normal(size=n)
    missing[rng.random(n) < 0.5] = np.nan
    macro = np.repeat(rng.normal(size=n_dates), n_stocks)
    trending = np.repeat(np.linspace(0, 10, n_dates), n_stocks) + rng.normal(size=n) * 0.1
    return pd.DataFrame({
        "Date": dates.astype(str),
        "P123 ID": np.tile(np.arange(n_stocks), n_dates),
        "Ticker": "X",
        "normal": rng.normal(size=n),
        "clipped": clipped,
        "mostly_missing": missing,
        "constant": 1.0,
        "binary": rng.integers(0, 2, size=n).astype(float),
        "macro": macro,
        "trending": trending,
        "Sub-Sector Code": rng.integers(1, 6, size=n),
        "TARGET": rng.normal(size=n),
    })


def _run(df, **kw):
    cfg = StatsConfig(data_path="x.csv", exclude_columns=["P123 ID", "Ticker"],
                      categorical_columns=["Sub-Sector Code"], **kw)
    res = FeatureStatsAnalyzer(cfg).run(df)
    return res, {r["feature"]: r for r in res["features"]}


def test_stats_match_pandas(panel):
    _, rows = _run(panel)
    s = panel["normal"]
    assert rows["normal"]["mean"] == pytest.approx(s.mean(), abs=1e-6)
    assert rows["normal"]["std"] == pytest.approx(s.std(), abs=1e-6)
    assert rows["normal"]["min"] == pytest.approx(s.min(), abs=1e-6)
    assert rows["normal"]["max"] == pytest.approx(s.max(), abs=1e-6)
    assert rows["normal"]["p50"] == pytest.approx(s.median(), abs=1e-6)
    assert rows["mostly_missing"]["missing_pct"] == pytest.approx(panel["mostly_missing"].isna().mean(), abs=1e-6)


def test_columns_and_flags(panel):
    res, rows = _run(panel)
    assert "P123 ID" not in rows and "Ticker" not in rows and "Date" not in rows
    assert "Sub-Sector Code" not in rows
    assert res["categorical"][0]["feature"] == "Sub-Sector Code"
    assert res["categorical"][0]["categories"] == 5
    assert rows["TARGET"]["role"] == "target" and "target" in rows["TARGET"]["flags"]
    assert "pile_up" in rows["clipped"]["flags"]
    assert "high_missing" in rows["mostly_missing"]["flags"]
    assert "constant" in rows["constant"]["flags"]
    assert "binary" in rows["binary"]["flags"]
    assert "date_constant" in rows["macro"]["flags"]
    assert "drift" not in rows["macro"]["flags"]
    assert "drift" in rows["trending"]["flags"]
    assert rows["normal"]["flags"] == []


def test_histogram_counts_all_values(panel):
    res, _ = _run(panel, hist_bins=20)
    h = res["histograms"]["normal"]
    assert len(h["counts"]) == 20 and len(h["edges"]) == 21
    assert sum(h["counts"]) == panel["normal"].notna().sum()
    clipped = _run(panel, hist_clip_quantile=0.01)[0]["histograms"]["normal"]
    assert clipped["clipped_low"] > 0 and clipped["clipped_high"] > 0


def test_cli_writes_outputs(panel, tmp_path):
    data = tmp_path / "d.csv"
    panel.to_csv(data, index=False)
    out = tmp_path / "out"
    main(["--data", str(data), "--output", str(out), "--exclude", "P123 ID,Ticker",
          "--categorical", "Sub-Sector Code", "--log-path", str(tmp_path / "run.log")])
    for name in ["report.html", "results.json", "stats.csv"]:
        assert (out / name).exists()
    results = json.loads((out / "results.json").read_text())
    assert results["meta"]["dates"] == 60
    assert pd.read_csv(out / "stats.csv").shape[0] == 8  # 7 numeric features + target
    main(["--output", str(out), "--report-only"])

import json

import numpy as np
import pandas as pd
import pytest

from train_tool import metrics
from train_tool.config import load_config
from train_tool.folds import assign_rows, build_folds, holdout_months
from train_tool.trainer import rescore, run_training


def test_rolling_folds_match_p123_screen():
    # P123: 2005-01-01 to 2025-11-22, 16 training years, 20-week gap, 4 folds
    # -> holdout 13.5 months, results 2021-05-20 to 2025-11-22
    folds = build_folds("2005-01-01", "2025-11-22", "rolling", 4, 16.0, 20)
    assert len(folds) == 4
    assert round(holdout_months(folds), 1) == 13.5
    assert folds[0].train_start == pd.Timestamp("2005-01-01")
    assert abs((folds[0].holdout_start - pd.Timestamp("2021-05-20")).days) <= 1
    assert folds[-1].holdout_end == pd.Timestamp("2025-11-22")
    for prev, cur in zip(folds, folds[1:]):
        assert cur.holdout_start == prev.holdout_end  # back to back holdouts
        assert cur.train_start - prev.train_start == prev.holdout_end - prev.holdout_start  # rolling
    for f in folds:
        assert f.holdout_start - f.train_end == pd.Timedelta(weeks=20)


def test_expanding_folds_keep_start():
    folds = build_folds("2005-01-01", "2025-11-22", "expanding", 4, 16.0, 20)
    assert all(f.train_start == pd.Timestamp("2005-01-01") for f in folds)


def test_rows_never_in_train_and_holdout():
    dates = pd.Series(pd.date_range("2005-01-01", "2025-11-22", freq="W-SAT"))
    folds = build_folds("2005-01-01", "2025-11-22", "rolling", 4, 16.0, 20)
    for i, f in enumerate(folds):
        tr, ho = assign_rows(dates, f, i == len(folds) - 1)
        assert not (tr & ho).any()
        assert dates[tr].max() < dates[ho].min() - pd.Timedelta(weeks=19)


def test_quantiles_perfect_prediction():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2020-01-04", periods=20, freq="W-SAT"):
        y = rng.normal(size=100)
        rows.append(pd.DataFrame({"Date": d, "id": range(100), "target": y, "pred": y}))
    df = pd.concat(rows, ignore_index=True)
    res = metrics.evaluate(df, 10)
    assert res["spearman"] == pytest.approx(1.0)
    assert res["rmse"] == pytest.approx(0.0)
    assert res["tgt_high"] > res["tgt_low"]
    assert res["tgt_hl"] == pytest.approx(res["tgt_high"] - res["tgt_low"])
    assert list(res["target_buckets"].index) == list(range(1, 11))
    assert res["target_buckets"].is_monotonic_increasing
    assert "high_avg" not in res  # no return column -> no portfolio stats


def test_portfolio_stats_known_returns():
    # 2 stocks, 2 quantiles, 52 weekly dates: top stock always earns 1%, bottom 0%.
    dates = pd.date_range("2020-01-04", periods=52, freq="W-SAT")
    df = pd.concat([pd.DataFrame({"Date": d, "id": [1, 2], "target": [1.0, 0.0],
                                  "pred": [1.0, 0.0], "ret": [0.01, 0.0]}) for d in dates])
    res = metrics.evaluate(df, 2, slippage_pct=0.1)
    ppy = 365.25 / 7  # P123 annualizes weekly data with 52.18 periods
    assert res["high_avg"] == pytest.approx((1.01**ppy - 1) * 100)  # mean weekly, compounded
    assert res["high_cmp"] == pytest.approx((1.01**ppy - 1) * 100)
    assert res["low_avg"] == pytest.approx(0.0)
    assert res["hl_avg"] == pytest.approx((1.01**ppy - 1) * 100)
    assert res["high_turn"] == pytest.approx(0.0)  # same stock every week, no slippage


def test_portfolio_turnover_and_slippage():
    # The top stock alternates every week: 100% one-way turnover, 2 x 0.1% cost per week.
    dates = pd.date_range("2020-01-04", periods=53, freq="W-SAT")
    rows = []
    for i, d in enumerate(dates):
        top = 1 if i % 2 == 0 else 2
        rows.append(pd.DataFrame({"Date": d, "id": [1, 2], "target": 0.0,
                                  "pred": [1.0 if top == 1 else 0.0, 0.0 if top == 1 else 1.0],
                                  "ret": 0.0}))
    res = metrics.evaluate(pd.concat(rows), 2, slippage_pct=0.1)
    ppy = 365.25 / 7
    assert res["high_turn"] == pytest.approx(100 * ppy)
    mean = -0.002 * 52 / 53  # 2 x 0.1% every week except the first (initial purchase)
    assert res["high_avg"] == pytest.approx(((1 + mean) ** ppy - 1) * 100, rel=1e-6)


def test_annualize_matches_p123_portfolio_download():
    # Weekly High returns (in %) from a P123 Portfolio download: Compare All shows
    # Avg% = (1 + mean)^52.18 - 1 and SR = mean x 52.18 / (std x sqrt(52.18)).
    weekly = pd.Series([0.5, -0.2, 1.0, 0.3, -0.7, 0.9]) / 100
    ppy = 365.25 / 7
    avg, cmp_, sd, sr = metrics._annualize(weekly, ppy)
    assert avg == pytest.approx(((1 + weekly.mean()) ** ppy - 1) * 100)
    assert sd == pytest.approx(weekly.std() * ppy**0.5 * 100)
    assert sr == pytest.approx(weekly.mean() * ppy / (weekly.std() * ppy**0.5))


def test_calibrate_weekdays_handles_holidays():
    from train_tool.calibrate import fit
    days = pd.bdate_range("2022-08-01", "2022-10-31")
    days = days[days != pd.Timestamp("2022-09-05")]  # Labor Day
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({"dt": days, "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))})
    sats = pd.date_range("2022-08-06", "2022-10-15", freq="W-SAT")
    idx = prices.set_index("dt")["close"]
    rows = []
    for d in sats:  # Future%Chg_D(5): close as of Monday -> close as of next Monday
        start = idx[idx.index <= d + pd.Timedelta(days=2)].iloc[-1]  # Labor Day -> Friday's close
        end = idx[idx.index <= d + pd.Timedelta(days=9)].iloc[-1]
        rows.append({"Date": d, "v": 0.5 + 0.07 * (end / start - 1) * 100})
    vals = pd.DataFrame(rows)
    zero, per_pct, r2, _ = fit(vals, prices, 5, "Date", "v", weekdays=True)
    assert zero == pytest.approx(0.5) and per_pct == pytest.approx(0.07) and r2 == pytest.approx(1.0)
    _, _, r2_bars, _ = fit(vals, prices, 5, "Date", "v", weekdays=False)
    assert r2_bars < r2  # 5 bars runs a day long in the Labor Day week


def test_end_to_end_and_rescore(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2010-01-02", "2020-12-26", freq="W-SAT")
    parts = []
    for d in dates:
        x = rng.normal(size=(60, 3))
        parts.append(pd.DataFrame({
            "Date": d.strftime("%Y-%m-%d"), "P123 ID": range(60), "Ticker": "T",
            "a: \"odd\" name": x[:, 0], "b": x[:, 1], "c": x[:, 2],
            "TARGET": x[:, 0] * 0.5 + rng.normal(size=60),
        }))
    data = tmp_path / "data.csv"
    pd.concat(parts).to_csv(data, index=False)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "_comment": "test",
        "data_path": str(data), "output_dir": str(tmp_path / "out"),
        "validation": {"method": "rolling", "folds": 2, "training_years": 6, "gap_weeks": 20},
        "runs": 2, "quantiles": 5, "n_jobs": 1,
        "models": [{"name": "m1", "params": {"n_estimators": 20, "num_leaves": 8,
                                              "subsample": 0.8, "subsample_freq": 1}}],
    }))
    cfg = load_config(str(cfg_path))
    run_training(cfg)
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert len(runs) == 2 and (runs["spearman"] > 0.2).all() and (runs["tgt_hl"] > 0).all()
    assert (tmp_path / "out" / "summary.csv").exists()
    rescore(cfg, 10)
    q = pd.read_csv(tmp_path / "out" / "quantiles_10.csv")
    assert "q10" in q.columns and "mean" in q["run"].astype(str).tolist()
    # 2 runs -> the report has "mean" and "ensemble" views before the runs; no returns -> TARGET only
    page = (tmp_path / "out" / "report.html").read_text()
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\nconst META", 1)[0])
    assert [v["key"] for v in payload["models"][0]["views"]] == ["mean", "ensemble", "r1", "r2"]
    assert not payload["meta"]["has_returns"]


def _write_dataset(tmp_path, with_ret: bool):
    rng = np.random.default_rng(2)
    parts = []
    for d in pd.date_range("2010-01-02", "2020-12-26", freq="W-SAT"):
        x = rng.normal(size=(60, 2))
        ret = 0.01 * x[:, 0] + rng.normal(scale=0.02, size=60)
        part = pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "P123 ID": range(60), "Ticker": "T",
                             "a": x[:, 0], "b": x[:, 1], "TARGET": x[:, 0] + rng.normal(size=60)})
        if with_ret:
            part["Future%Chg(5)"] = 0.5 + 0.07 * ret * 100  # MinMax-style scaling
        parts.append(part)
    path = tmp_path / ("ret.csv" if with_ret else "data.csv")
    pd.concat(parts).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("separate_file", [False, True])
def test_returns_column_is_excluded_and_scored(tmp_path, separate_file):
    data = _write_dataset(tmp_path, with_ret=not separate_file)
    cfg = {"data_path": str(data), "output_dir": str(tmp_path / "out"),
           "returns_col": "Future%Chg(5)", "returns_zero": 0.5, "returns_per_pct": 0.07,
           "validation": {"folds": 2, "training_years": 6, "gap_weeks": 20},
           "runs": 1, "n_jobs": 1, "quantiles": 5,
           "models": [{"name": "m", "params": {"n_estimators": 20, "num_leaves": 8}}]}
    if separate_file:
        cfg["returns_path"] = str(_write_dataset(tmp_path, with_ret=True))
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    run_training(load_config(str(tmp_path / "cfg.json")))
    imp = pd.read_csv(tmp_path / "out" / "feature_importance.csv")
    assert set(imp["feature"]) == {"a", "b"}  # the return column never becomes a feature
    page = (tmp_path / "out" / "report.html").read_text()
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\nconst META", 1)[0])
    assert payload["meta"]["has_returns"] and payload["meta"]["quantiles"] == [10, 20, 50]
    q50 = payload["models"][0]["views"][0]["q"]["50"]
    assert len(q50["buckets"]["avg"]) == 50 and len(q50["weekly"]["high"]) == len(payload["models"][0]["dates"])
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert runs.loc[0, "hl_avg"] > 0 and abs(runs.loc[0, "high_avg"]) < 500


def test_calibrate_reads_returns_path_and_flags_clipping(tmp_path, capsys):
    from train_tool import calibrate
    days = pd.bdate_range("2020-01-01", "2021-06-30")
    rng = np.random.default_rng(4)
    prices = pd.DataFrame({"dt": days, "close": 100 * np.cumprod(1 + rng.normal(0, 0.02, len(days)))})
    prices.to_csv(tmp_path / "calib_prices_7.csv", index=False)  # cached: no API call
    idx = prices.set_index("dt")["close"]
    rows = []
    for d in pd.date_range("2020-01-04", "2021-05-29", freq="W-SAT"):
        start = idx[idx.index <= d + pd.Timedelta(days=2)].iloc[-1]
        end = idx[idx.index <= d + pd.Timedelta(days=9)].iloc[-1]
        rows.append({"Date": d.strftime("%Y-%m-%d"), "P123 ID": 7, "Ticker": "X",
                     "Future%Chg_D(5)": round(0.5 + 0.07 * (end / start - 1) * 100, 4)})
    rets = pd.DataFrame(rows)
    (tmp_path / "data.csv").write_text("Date,P123 ID,Ticker,f,TARGET\n2020-01-04,7,X,0,0\n")
    cfg = {"data_path": str(tmp_path / "data.csv"), "output_dir": str(tmp_path / "out"),
           "returns_col": "Future%Chg_D(5)", "returns_path": str(tmp_path / "ret.csv"),
           "models": [{"name": "m", "params": {}}]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))

    rets.to_csv(tmp_path / "ret.csv", index=False)
    calibrate.main(["--config", str(tmp_path / "cfg.json"), "--uids", "7"])
    out = capsys.readouterr().out
    assert "R2=1.00000" in out and "WARNING" not in out

    clipped = rets.assign(**{"Future%Chg_D(5)": rets["Future%Chg_D(5)"].clip(0.4, 0.6)})
    clipped.to_csv(tmp_path / "ret.csv", index=False)
    calibrate.main(["--config", str(tmp_path / "cfg.json"), "--uids", "7"])
    assert "clipped outliers" in capsys.readouterr().out


def test_clip_option_clips_features_and_target(tmp_path):
    from train_tool.trainer import load_dataset
    df = pd.DataFrame({"Date": ["2020-01-04"] * 4, "P123 ID": [1, 2, 3, 4], "Ticker": "T",
                       "a": [-9.0, -1.0, 1.0, 40.0], "TARGET": [69.0, 0.5, -0.5, -10.6]})
    df.to_csv(tmp_path / "d.csv", index=False)
    (tmp_path / "c.json").write_text(json.dumps({
        "data_path": str(tmp_path / "d.csv"), "output_dir": str(tmp_path / "o"),
        "preprocessing": {"clip": 2.5}, "models": [{"name": "m", "params": {}}]}))
    out, feats = load_dataset(load_config(str(tmp_path / "c.json")))
    assert feats == ["a"]
    assert out["a"].tolist() == [-2.5, -1.0, 1.0, 2.5]
    assert out["TARGET"].tolist() == [2.5, 0.5, -0.5, -2.5]

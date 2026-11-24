import json
import os

import numpy as np
import pandas as pd
import pytest

from train_tool.feature_test import build_variants, load_test_config, run_feature_test, summarize


def test_variants_remove_and_add():
    feats = ["a", "b", "c", "d"]
    rem = build_variants("remove", feats, ["b", "c"], include_all=True)
    assert [v["name"] for v in rem] == ["baseline", "- b", "- c", "- all candidates"]
    assert rem[0]["features"] == feats and rem[1]["features"] == ["a", "c", "d"]
    assert rem[3]["features"] == ["a", "d"]
    add = build_variants("add", feats, ["c", "d"], include_all=False)
    assert [v["name"] for v in add] == ["baseline", "+ c", "+ d"]
    assert add[0]["features"] == ["a", "b"] and add[2]["features"] == ["a", "b", "d"]
    assert len(build_variants("remove", feats, "all", include_all=False)) == 5
    with pytest.raises(ValueError, match="not among"):
        build_variants("remove", feats, ["zz"], include_all=False)


def test_summary_is_paired_by_seed():
    # Variant is exactly 1.0 better than baseline in every seed, although seeds differ a lot:
    # the paired difference has zero spread, an unpaired comparison would not.
    base = [10.0, 20.0, 30.0]
    rows = [{"variant": "baseline", "feature": None, "n_features": 3, "seed": s, "hl_avg@10": v}
            for s, v in zip([1, 2, 3], base)]
    rows += [{"variant": "- x", "feature": "x", "n_features": 2, "seed": s, "hl_avg@10": v + 1}
             for s, v in zip([1, 2, 3], base)]
    s = summarize(pd.DataFrame(rows), [10]).set_index("variant")
    assert s.loc["- x", "hl_avg@10_delta"] == pytest.approx(1.0)
    assert s.loc["- x", "hl_avg@10_se"] == pytest.approx(0.0)
    assert s.loc["baseline", "hl_avg@10_delta"] == pytest.approx(0.0)


def _dataset(tmp_path):
    rng = np.random.default_rng(5)
    parts = []
    for d in pd.date_range("2010-01-02", "2020-12-26", freq="W-SAT"):
        x = rng.normal(size=(50, 3))
        ret = 0.01 * x[:, 0] + rng.normal(scale=0.02, size=50)
        parts.append(pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "P123 ID": range(50), "Ticker": "T",
                                   "good": x[:, 0], "noise1": x[:, 1], "noise2": x[:, 2],
                                   "TARGET": x[:, 0] + rng.normal(size=50), "ret": ret * 100}))
    path = tmp_path / "d.csv"
    pd.concat(parts).to_csv(path, index=False)
    base = {"data_path": str(path), "output_dir": str(tmp_path / "unused"), "returns_col": "ret",
            "validation": {"folds": 2, "training_years": 6, "gap_weeks": 20}, "n_jobs": 1,
            "models": [{"name": "m", "params": {"n_estimators": 30, "num_leaves": 8,
                                                 "subsample": 0.8, "subsample_freq": 1}}]}
    (tmp_path / "base.json").write_text(json.dumps(base))
    return tmp_path / "base.json"


def test_remove_test_end_to_end_and_resume(tmp_path):
    base = _dataset(tmp_path)
    cfg = {"base_config": str(base), "output_dir": str(tmp_path / "ft"), "mode": "remove",
           "candidates": ["good", "noise1"], "runs": 2, "quantiles": [5, 10]}
    (tmp_path / "ft.json").write_text(json.dumps(cfg))
    text = run_feature_test(load_test_config(str(tmp_path / "ft.json")))
    assert "- good" in text
    s = pd.read_csv(tmp_path / "ft" / "summary.csv").set_index("variant")
    assert list(s.index) == ["baseline", "- good", "- noise1", "- all candidates"]
    # removing the only informative feature must hurt clearly
    assert s.loc["- good", "spearman_delta"] < -0.1
    assert abs(s.loc["- noise1", "spearman_delta"]) < abs(s.loc["- good", "spearman_delta"])
    assert (tmp_path / "ft" / "report.html").exists()

    files = sorted(os.listdir(tmp_path / "ft" / "predictions"))
    assert len(files) == 4 * 2
    mtimes = {f: os.path.getmtime(tmp_path / "ft" / "predictions" / f) for f in files}
    run_feature_test(load_test_config(str(tmp_path / "ft.json")))  # resume: nothing retrained
    assert {f: os.path.getmtime(tmp_path / "ft" / "predictions" / f) for f in files} == mtimes


def test_add_test(tmp_path):
    base = _dataset(tmp_path)
    cfg = {"base_config": str(base), "output_dir": str(tmp_path / "ft"), "mode": "add",
           "candidates": ["good", "noise2"], "runs": 2, "quantiles": [5], "include_all": False}
    (tmp_path / "ft.json").write_text(json.dumps(cfg))
    run_feature_test(load_test_config(str(tmp_path / "ft.json")))
    s = pd.read_csv(tmp_path / "ft" / "summary.csv").set_index("variant")
    assert s.loc["baseline", "n_features"] == 1  # noise1 only
    assert s.loc["+ good", "spearman_delta"] > 0.1

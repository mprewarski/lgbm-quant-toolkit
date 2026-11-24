"""Feature-set tests: remove or add features one at a time and compare against a baseline.

    python -m train_tool.feature_test --config models/sp500/feature_test_remove.json

Every feature set (the baseline and one variant per candidate feature) is trained with the
same seeds, the same folds and the same data, and each variant is judged on its paired
difference from the baseline seed by seed. Pairing removes most of the run-to-run noise
that a plain comparison of averages would carry, so the standard error of a difference is
much smaller than the spread of single runs.

Finished runs are saved, so a test that stops can be resumed by running it again, and
adding candidates or runs only trains what's new. Results: report.html, report.txt,
runs.csv (every run) and summary.csv (per variant: mean, difference vs baseline, standard
error, t).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import html_report, metrics
from .config import load_config
from .folds import build_folds
from .trainer import load_dataset, slug, train_run

log = logging.getLogger("train_tool")

# metric key, label, decimals, better direction (+1 higher is better, -1 lower is better),
# depends on the quantile count
METRICS = [
    ("hl_avg", "H-L Avg%", 2, +1, True),
    ("high_avg", "High Avg%", 2, +1, True),
    ("low_avg", "Low Avg%", 2, -1, True),
    ("high_sr", "High SR", 3, +1, True),
    ("spearman", "Spearman", 4, +1, False),
    ("ic_mean", "IC/date", 4, +1, False),
    ("rmse", "RMSE", 4, -1, False),
    ("tgt_hl", "Tgt H-L", 3, +1, True),
]


@dataclass
class FeatureTestConfig:
    base_config: str
    output_dir: str
    candidates: object  # list of feature names, or "all" (remove mode)
    mode: str = "remove"  # "remove" | "add"
    name: Optional[str] = None
    data_path: Optional[str] = None  # overrides the base config's dataset (e.g. a new download)
    model: Optional[str] = None  # model name from the base config; default the first
    runs: int = 3
    base_seed: int = 1
    seeds: Optional[List[int]] = None
    quantiles: List[int] = field(default_factory=lambda: [10, 20, 50])
    include_all: bool = True  # also test removing / adding all candidates together
    rank_by: str = "hl_avg"


def load_test_config(path: str, overrides: Optional[dict] = None) -> FeatureTestConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = sorted(set(raw) - set(FeatureTestConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown feature-test config keys: {unknown}")
    cfg = FeatureTestConfig(**raw)
    if cfg.mode not in ("remove", "add"):
        raise ValueError("mode must be 'remove' or 'add'")
    if cfg.rank_by not in {m[0] for m in METRICS}:
        raise ValueError(f"rank_by must be one of {[m[0] for m in METRICS]}")
    return cfg


# --------------------------------------------------------------------------- variants
def build_variants(mode: str, features: List[str], candidates, include_all: bool) -> List[dict]:
    """Baseline first, then one variant per candidate (and optionally all candidates at once)."""
    if candidates == "all":
        if mode != "remove":
            raise ValueError('candidates "all" only works in remove mode')
        candidates = list(features)
    candidates = list(dict.fromkeys(candidates))
    missing = [c for c in candidates if c not in features]
    if missing:
        raise ValueError(f"Candidates not among the dataset's features: {missing}")
    if mode == "remove":
        base = list(features)
        variants = [{"name": "baseline", "label": "Baseline (all features)", "feature": None, "features": base}]
        for c in candidates:
            variants.append({"name": f"- {c}", "label": f"Remove {c}", "feature": c,
                             "features": [f for f in base if f != c]})
        if include_all and len(candidates) > 1:
            variants.append({"name": "- all candidates", "label": f"Remove all {len(candidates)} candidates",
                             "feature": "(all)", "features": [f for f in base if f not in set(candidates)]})
    else:
        base = [f for f in features if f not in set(candidates)]
        variants = [{"name": "baseline", "label": "Baseline (without candidates)", "feature": None, "features": base}]
        for c in candidates:
            variants.append({"name": f"+ {c}", "label": f"Add {c}", "feature": c, "features": base + [c]})
        if include_all and len(candidates) > 1:
            variants.append({"name": "+ all candidates", "label": f"Add all {len(candidates)} candidates",
                             "feature": "(all)", "features": base + candidates})
    return variants


def _run_key(variant_features: List[str], params: dict, train_cfg, seed: int) -> str:
    """Fingerprint of everything that determines a run's predictions (for resuming)."""
    blob = json.dumps({
        "features": sorted(variant_features), "params": params, "seed": seed,
        "data": os.path.normpath(train_cfg.data_path), "returns": train_cfg.returns_path,
        "validation": [train_cfg.validation_method, train_cfg.folds, train_cfg.training_years, train_cfg.gap_weeks],
        "period": [train_cfg.dataset_start, train_cfg.dataset_end],
        "prep": [train_cfg.na_fill, train_cfg.max_na_pct, train_cfg.clip, train_cfg.categorical_columns],
    }, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- scoring
def score_run(pred: pd.DataFrame, quantiles: List[int], slippage: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, n in enumerate(quantiles):
        res = metrics.evaluate(pred, n, slippage)
        for key, _label, _d, _dir, per_q in METRICS:
            if per_q:
                out[f"{key}@{n}"] = res.get(key)
            elif i == 0:
                out[key] = res.get(key)
    return out


def metric_columns(quantiles: List[int]) -> List[str]:
    cols = []
    for key, _l, _d, _dir, per_q in METRICS:
        cols += [f"{key}@{n}" for n in quantiles] if per_q else [key]
    return cols


def summarize(runs: pd.DataFrame, quantiles: List[int]) -> pd.DataFrame:
    """Per variant and metric: mean, difference vs baseline (paired by seed), its standard
    error and t. The baseline's own row has difference 0."""
    rows = []
    base = runs[runs["variant"] == "baseline"].set_index("seed")
    for variant, grp in runs.groupby("variant", sort=False):
        g = grp.set_index("seed")
        row = {"variant": variant, "feature": grp["feature"].iloc[0], "n_features": int(grp["n_features"].iloc[0]),
               "runs": len(g)}
        for col in metric_columns(quantiles):
            if col not in g.columns:
                continue
            row[col] = g[col].mean()
            common = g.index.intersection(base.index)
            d = (g.loc[common, col] - base.loc[common, col]).astype(float)
            row[f"{col}_delta"] = d.mean() if len(d) else np.nan
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
            row[f"{col}_se"] = se
            # a t from 2 runs is meaningless (the spread of 2 differences is mostly luck)
            row[f"{col}_t"] = d.mean() / se if len(d) >= 3 and se and se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def format_text(summary: pd.DataFrame, tcfg: FeatureTestConfig) -> str:
    q = tcfg.quantiles[0]
    key = f"{tcfg.rank_by}@{q}" if f"{tcfg.rank_by}@{q}" in summary.columns else tcfg.rank_by
    direction = next(m[3] for m in METRICS if m[0] == tcfg.rank_by)
    body = summary[summary["variant"] != "baseline"].copy()
    body = body.sort_values(f"{key}_delta", ascending=direction < 0)
    base = summary[summary["variant"] == "baseline"].iloc[0]
    cols = [(f"hl_avg@{q}", "H-L", 2), (f"high_avg@{q}", "High", 2), ("spearman", "Spearman", 4), ("rmse", "RMSE", 4)]
    head = ["variant", "feats"] + [h for c, h, _ in cols for h in (f"{h}", f"d{h}", "+-se", "t")]
    lines = []
    def fmt_row(r, is_base=False):
        cells = [str(r["variant"])[:60], str(r["n_features"])]
        for c, _h, d in cols:
            cells.append(f"{r[c]:.{d}f}")
            if is_base:
                cells += ["", "", ""]
            else:
                cells += [f"{r[c + '_delta']:+.{d}f}", f"{r[c + '_se']:.{d}f}" if pd.notna(r[c + '_se']) else "-",
                          f"{r[c + '_t']:+.1f}" if pd.notna(r[c + '_t']) else "-"]
        return cells
    rows = [fmt_row(base, True)] + [fmt_row(r) for _, r in body.iterrows()]
    widths = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(head)]
    fmt = lambda v: "  ".join(s.ljust(w) if i == 0 else s.rjust(w) for i, (s, w) in enumerate(zip(v, widths)))  # noqa: E731
    lines += [fmt(head), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]
    what = "removing" if tcfg.mode == "remove" else "adding"
    note = (f"Feature test ({tcfg.mode}), {int(summary['runs'].max())} runs per feature set, {q} quantiles. "
            f"d = change from {what} the feature vs the baseline, averaged over seeds; +-se = its standard "
            f"error; t = d / se (|t| > 2 is unlikely to be noise; shown from 3 runs). Sorted by {key}.")
    return note + "\n\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- html
def write_html(path: str, summary: pd.DataFrame, runs: pd.DataFrame, tcfg: FeatureTestConfig, train_cfg, model_name: str):
    variants = []
    cols = metric_columns(tcfg.quantiles)
    base_runs = runs[runs["variant"] == "baseline"].set_index("seed")
    for _, r in summary.iterrows():
        g = runs[runs["variant"] == r["variant"]].set_index("seed")
        per_run = {}
        for c in cols:
            if c in g.columns:
                common = g.index.intersection(base_runs.index)
                per_run[c] = {"values": g[c].tolist(), "deltas": (g.loc[common, c] - base_runs.loc[common, c]).tolist()}
        variants.append({
            "name": r["variant"], "feature": r["feature"], "n_features": r["n_features"], "runs": r["runs"],
            "stats": {c: {"mean": r.get(c), "delta": r.get(f"{c}_delta"), "se": r.get(f"{c}_se"), "t": r.get(f"{c}_t")}
                      for c in cols if c in summary.columns},
            "per_run": per_run,
        })
    payload = html_report._clean({
        "meta": {
            "mode": tcfg.mode, "name": tcfg.name or os.path.basename(tcfg.output_dir.rstrip("/")),
            "model": model_name, "data_path": train_cfg.data_path, "runs": int(runs.groupby("variant").size().max()),
            "seeds": sorted(runs["seed"].unique().tolist()), "quantiles": tcfg.quantiles,
            "slippage": train_cfg.slippage_pct, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "has_returns": "hl_avg@%d" % tcfg.quantiles[0] in runs.columns and runs["hl_avg@%d" % tcfg.quantiles[0]].notna().any(),
            "rank_by": tcfg.rank_by,
        },
        "metrics": [{"key": k, "label": l, "dec": d, "dir": s, "per_q": q} for k, l, d, s, q in METRICS],
        "variants": variants,
    })
    style = re.search(r"<style>(.*?)</style>", html_report.TEMPLATE, re.S).group(1)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = f"Feature test: {payload['meta']['name']}"
    page = FT_TEMPLATE.replace("__STYLE__", style).replace("__TITLE__", title).replace("__DATA__", data)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


# --------------------------------------------------------------------------- main
def run_feature_test(tcfg: FeatureTestConfig, rescore_only: bool = False) -> str:
    train_cfg = load_config(tcfg.base_config, {"data_path": tcfg.data_path, "output_dir": tcfg.output_dir})
    spec = next((m for m in train_cfg.models if tcfg.model in (None, m.name)), None)
    if spec is None:
        raise ValueError(f"Model {tcfg.model!r} not in {tcfg.base_config}")
    seeds = list(tcfg.seeds) if tcfg.seeds else [tcfg.base_seed + i for i in range(tcfg.runs)]
    os.makedirs(os.path.join(tcfg.output_dir, "predictions"), exist_ok=True)

    df, features = load_dataset(train_cfg)
    variants = build_variants(tcfg.mode, features, tcfg.candidates, tcfg.include_all)
    start, end = df.attrs["period"]
    folds = build_folds(start, end, train_cfg.validation_method, train_cfg.folds,
                        train_cfg.training_years, train_cfg.gap_weeks)
    with open(os.path.join(tcfg.output_dir, "variants.json"), "w", encoding="utf-8") as handle:
        json.dump({"mode": tcfg.mode, "model": spec.name, "seeds": seeds,
                   "variants": [{k: v[k] for k in ("name", "feature", "features")} for v in variants]}, handle, indent=2)

    total, done = len(variants) * len(seeds), 0
    log.info("%d feature sets x %d runs = %d runs (%s mode, model %s)", len(variants), len(seeds), total, tcfg.mode, spec.name)
    rows = []
    t_start = time.time()
    trained = 0
    for v in variants:
        for seed in seeds:
            key = _run_key(v["features"], spec.params, train_cfg, seed)
            path = os.path.join(tcfg.output_dir, "predictions", f"{slug(v['name'])[:80]}__seed{seed}__{key}.csv.gz")
            if os.path.exists(path):
                pred = pd.read_csv(path, parse_dates=[metrics.DATE])
            elif rescore_only:
                log.warning("No saved run for %s seed %d; skipped", v["name"], seed)
                continue
            else:
                t0 = time.time()
                pred, _imp = train_run(df, v["features"], folds, spec, seed, train_cfg)
                pred.to_csv(path, index=False)
                trained += 1
                eta = (time.time() - t_start) / trained * (total - done - 1)
                log.info("[%d/%d] %s seed %d: %.0fs (about %.0f min left)", done + 1, total, v["name"], seed,
                         time.time() - t0, eta / 60)
            done += 1
            rows.append({"variant": v["name"], "feature": v["feature"], "n_features": len(v["features"]),
                         "seed": seed, **score_run(pred, tcfg.quantiles, train_cfg.slippage_pct)})

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(tcfg.output_dir, "runs.csv"), index=False)
    summary = summarize(runs, tcfg.quantiles)
    summary.to_csv(os.path.join(tcfg.output_dir, "summary.csv"), index=False)
    text = format_text(summary, tcfg)
    with open(os.path.join(tcfg.output_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    write_html(os.path.join(tcfg.output_dir, "report.html"), summary, runs, tcfg, train_cfg, spec.name)
    return text


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Remove or add features one at a time and compare with a baseline.")
    ap.add_argument("--config", required=True, help="Feature-test config JSON")
    ap.add_argument("--runs", type=int, help="Runs (seeds) per feature set")
    ap.add_argument("--report-only", action="store_true", help="Rebuild the reports from saved runs, no training")
    args = ap.parse_args(argv)
    tcfg = load_test_config(args.config, {"runs": args.runs})
    os.makedirs(tcfg.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(tcfg.output_dir, "run.log"))])
    print(run_feature_test(tcfg, rescore_only=args.report_only))
    print(f"Results in {tcfg.output_dir}")


FT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>__STYLE__
.better { color: var(--pos); } .worse { color: var(--neg); }
td.feat { text-align: left; max-width: 520px; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
<main>
  <div class="card">
    <h1>__TITLE__</h1>
    <div class="sub" id="subtitle"></div>
    <div class="pills" id="meta"></div>
  </div>

  <div class="controls">
    <span><label for="metric">Metric</label><select id="metric"></select></span>
    <span id="qwrap"><label>Quantiles</label><span class="seg" id="qseg"></span></span>
  </div>

  <div class="card">
    <h2 id="chart-title"></h2>
    <div class="sub" id="chart-sub"></div>
    <div id="chart" class="chart"></div>
  </div>

  <div class="card">
    <h2>All feature sets</h2>
    <div class="sub">Sorted best to worst on the selected metric. Δ is paired by seed: each run is compared with the baseline run that used the same seed. ±se is the standard error of Δ; |t| above about 2 is unlikely to be noise.</div>
    <div class="table-wrap"><table id="table"></table></div>
  </div>

  <div class="card">
    <h2>How to read this</h2>
    <div class="glossary">
      <div><b>Remove test</b>: a positive Δ (better) means the model did better <i>without</i> that feature, so the feature may be hurting.</div>
      <div><b>Add test</b>: a positive Δ (better) means adding the feature helped.</div>
      <div><b>Better</b> is higher for returns, SR, Spearman and IC, and lower for RMSE and Low Avg%. Bars point right when the change is better.</div>
      <div><b>Dots</b>: the change in each individual run (one per seed). Dots on both sides of zero mean the effect isn't consistent.</div>
      <div><b>Noise</b>: with 3 runs a single test is rough. Re-run promising features with more runs (<code>--runs 5</code>) before acting; finished runs are reused.</div>
      <div><b>Avg%, SR, H-L</b>: as in the training report (weekly-rebalanced bucket portfolios, after slippage).</div>
    </div>
  </div>
</main>
<script>
const DATA = __DATA__, META = DATA.meta;
const state = { metric: META.has_returns ? META.rank_by : "spearman", q: String(META.quantiles[0]) };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (tag, attrs = {}, text) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
const fmt = (v, d) => (v === null || v === undefined) ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const sgn = (v, d) => (v === null || v === undefined) ? "–" : (v > 0 ? "+" : "") + fmt(v, d);
const metric = () => DATA.metrics.find(m => m.key === state.metric);
const col = () => metric().per_q ? `${state.metric}@${state.q}` : state.metric;
const better = (delta) => delta === null ? 0 : delta * metric().dir;

function setup() {
  const verb = META.mode === "remove" ? "Removing" : "Adding";
  document.getElementById("subtitle").textContent = `${verb} features one at a time · model ${META.model} · ${META.runs} runs per feature set (seeds ${META.seeds.join(", ")}) · generated ${META.generated}`;
  [`Data: ${META.data_path}`, `Mode: ${META.mode}`, `${DATA.variants.length - 1} feature sets vs baseline`, `Slippage ${META.slippage}%`]
    .forEach(t => document.getElementById("meta").appendChild(el("span", { class: "pill" }, t)));
  const ms = document.getElementById("metric");
  DATA.metrics.filter(m => META.has_returns || ["spearman", "ic_mean", "rmse", "tgt_hl"].includes(m.key))
    .forEach(m => ms.appendChild(el("option", { value: m.key }, m.label)));
  ms.value = state.metric;
  ms.onchange = () => { state.metric = ms.value; render(); };
  const seg = document.getElementById("qseg");
  META.quantiles.forEach(q => { const b = el("button", { type: "button", "data-q": q }, String(q)); b.onclick = () => { state.q = String(q); render(); }; seg.appendChild(b); });
}

function sorted() {
  const c = col();
  const rest = DATA.variants.filter(v => v.name !== "baseline");
  rest.sort((a, b) => better(b.stats[c] ? b.stats[c].delta : null) - better(a.stats[c] ? a.stats[c].delta : null));
  return rest;
}

function chart() {
  const m = metric(), c = col(), vs = sorted().slice().reverse(); // plotly draws bottom-up
  const verb = META.mode === "remove" ? "removing" : "adding";
  document.getElementById("chart-title").textContent = `Change in ${m.label}${m.per_q ? ` (${state.q} quantiles)` : ""} from ${verb} each feature`;
  const base = DATA.variants.find(v => v.name === "baseline").stats[c];
  document.getElementById("chart-sub").textContent = `Baseline ${m.label}: ${fmt(base.mean, m.dec)}. Bars: average change vs baseline, with ±1 standard error. Dots: each run. Right = better${m.dir < 0 ? " (lower " + m.label + ")" : ""}.`;
  const nCand = DATA.variants.filter(v => v.name !== "baseline" && v.feature !== "(all)").length;
  const y = vs.map(v => v.feature === "(all)" ? `All ${nCand} candidates` : (v.feature.length > 60 ? v.feature.slice(0, 58) + "…" : v.feature));
  const x = vs.map(v => v.stats[c].delta === null ? null : v.stats[c].delta * m.dir);
  const err = vs.map(v => v.stats[c].se);
  const colors = vs.map(v => better(v.stats[c].delta) >= 0 ? css("--pos") : css("--neg"));
  const custom = vs.map(v => [v.name, sgn(v.stats[c].delta, m.dec), fmt(v.stats[c].se, m.dec), v.stats[c].t === null ? "–" : sgn(v.stats[c].t, 1), fmt(v.stats[c].mean, m.dec)]);
  const traces = [{
    type: "bar", orientation: "h", x, y, customdata: custom, marker: { color: colors, cornerradius: 4 },
    error_x: { type: "data", array: err, color: css("--muted"), thickness: 1.5, width: 0 },
    hovertemplate: "<b>%{customdata[0]}</b><br>Δ %{customdata[1]} ± %{customdata[2]} (t %{customdata[3]})<br>Mean %{customdata[4]}<extra></extra>",
  }];
  const dx = [], dy = [];
  vs.forEach((v, i) => (v.per_run[c] ? v.per_run[c].deltas : []).forEach(d => { if (d !== null) { dx.push(d * m.dir); dy.push(y[i]); } }));
  traces.push({ type: "scatter", mode: "markers", x: dx, y: dy, marker: { color: css("--ink"), size: 9, opacity: 0.8, line: { color: css("--card"), width: 1.5 } }, hovertemplate: "run Δ %{x:.3f}<extra></extra>" });
  const ink = css("--muted"), grid = css("--grid");
  // the chart box grows with the number of feature sets (a fixed-height box let long
  // charts spill over the table below)
  const height = Math.max(320, 30 * vs.length + 90);
  document.getElementById("chart").style.height = height + "px";
  Plotly.react("chart", traces, {
    height, margin: { l: 20, r: 20, t: 10, b: 44 }, showlegend: false, bargap: 0.3,
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "Segoe UI, -apple-system, Arial, sans-serif", size: 12, color: ink },
    xaxis: { gridcolor: grid, zerolinecolor: css("--faint"), zerolinewidth: 1.5, title: { text: `Change in ${m.label} (right = better)` } },
    yaxis: { automargin: true, tickfont: { color: css("--ink") } },
    hoverlabel: { bgcolor: css("--card"), bordercolor: css("--line"), font: { color: css("--ink") } },
  }, { displaylogo: false, responsive: true });
}

function table() {
  const m = metric(), c = col(), t = document.getElementById("table"); t.innerHTML = "";
  const head = el("tr");
  ["Feature set", "Features", m.label, "Δ", "±se", "t", "Δ per run"].forEach((h, i) => head.appendChild(el("th", { class: i === 0 ? "l" : "" }, h)));
  t.appendChild(head);
  const base = DATA.variants.find(v => v.name === "baseline");
  [base, ...sorted()].forEach(v => {
    const s = v.stats[c], tr = el("tr", v.name === "baseline" ? { class: "mean" } : {});
    const cls = v.name === "baseline" ? "" : (better(s.delta) > 0 ? "better" : better(s.delta) < 0 ? "worse" : "");
    tr.appendChild(el("td", { class: "feat", title: v.name }, v.name === "baseline" ? "Baseline" : v.name));
    tr.appendChild(el("td", {}, String(v.n_features)));
    tr.appendChild(el("td", {}, fmt(s.mean, m.dec)));
    tr.appendChild(el("td", { class: cls }, v.name === "baseline" ? "" : sgn(s.delta, m.dec)));
    tr.appendChild(el("td", {}, v.name === "baseline" ? "" : fmt(s.se, m.dec)));
    tr.appendChild(el("td", { class: cls }, v.name === "baseline" || s.t === null ? "" : sgn(s.t, 1)));
    tr.appendChild(el("td", {}, v.name === "baseline" ? (v.per_run[c] ? v.per_run[c].values.map(x => fmt(x, m.dec)).join("  ") : "") : (v.per_run[c] ? v.per_run[c].deltas.map(x => sgn(x, m.dec)).join("  ") : "")));
    t.appendChild(tr);
  });
}

function render() {
  document.getElementById("qwrap").classList.toggle("hidden", !metric().per_q);
  document.querySelectorAll("#qseg button").forEach(b => b.classList.toggle("on", b.dataset.q === state.q));
  chart(); table();
}
setup(); render();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

"""Interactive HTML report for train_tool results (report.html).

Everything is computed here from the saved holdout predictions and embedded as JSON; the
page (Plotly from a CDN, like the other tools' reports) switches between models, runs and
quantile counts (default 10, 20 and 50) without recomputing.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import metrics
from .config import TrainConfig

SCALAR_KEYS = [
    "rmse", "spearman", "pearson", "r2", "ic_mean", "ic_ir",
    "hl_avg", "hl_sr", "high_avg", "high_cmp", "high_sd", "high_sr", "high_turn",
    "low_avg", "low_cmp", "low_sd", "low_sr", "low_turn", "univ_avg", "univ_cmp",
    "tgt_high", "tgt_low", "tgt_hl",
]
FOLD_KEYS = ["spearman", "ic_mean", "hl_avg", "high_avg", "low_avg", "univ_avg", "tgt_hl"]


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy -> python, rounded floats."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- per-run data
def _ic_series(pred: pd.DataFrame) -> pd.Series:
    return pred.groupby(metrics.DATE).apply(
        lambda g: g[metrics.Y].corr(g[metrics.P], method="spearman"), include_groups=False)


def _lift(pred: pd.DataFrame, bins: int = 100) -> Dict[str, list]:
    pct = pd.qcut(pred[metrics.P].rank(method="first"), bins, labels=False)
    g = pred.groupby(pct).agg(pred=(metrics.P, "mean"), actual=(metrics.Y, "mean"))
    return {"pred": g["pred"].tolist(), "actual": g["actual"].tolist()}


def _run_view(pred: pd.DataFrame, dates: pd.DatetimeIndex, quantiles: Sequence[int], slippage: float) -> dict:
    """All numbers the page needs for one set of predictions (a run or the ensemble)."""
    view = {"q": {}}
    ic = _ic_series(pred).reindex(dates)
    view["ic"] = ic.tolist()
    view["lift"] = _lift(pred)
    for n in quantiles:
        res = metrics.evaluate(pred, n, slippage)
        entry = {"stats": {k: res.get(k) for k in SCALAR_KEYS}}
        buckets = {"tgt": res["target_buckets"].reindex(range(1, n + 1)).tolist()}
        table = res.get("portfolio_buckets")
        if table is not None:
            for k in ("avg", "cmp", "turn", "sd", "stocks"):
                buckets[k] = table[k].reindex(range(1, n + 1)).tolist()
        entry["buckets"] = buckets
        weekly = res.get("weekly")
        if weekly is not None:
            w = weekly.reindex(dates)
            entry["weekly"] = {c: w[c].tolist() for c in ("high", "low", "univ")}
        folds = []
        for f, part in pred.groupby("fold"):
            r = metrics.evaluate(part, n, slippage)
            folds.append({"fold": int(f), **{k: r.get(k) for k in FOLD_KEYS},
                          "start": part[metrics.DATE].min().strftime("%Y-%m-%d"),
                          "end": part[metrics.DATE].max().strftime("%Y-%m-%d")})
        entry["folds"] = folds
        view["q"][str(n)] = entry
    return view


def _mean_view(views: List[dict]) -> dict:
    """Average of several runs' views, element by element (None-aware)."""
    def avg(values):
        if isinstance(values[0], dict):
            return {k: avg([v[k] for v in values]) for k in values[0]}
        if isinstance(values[0], list):
            return [avg([v[i] for v in values]) for i in range(len(values[0]))]
        nums = [v for v in values if isinstance(v, (int, float)) and v is not None and not
                (isinstance(v, float) and math.isnan(v))]
        if not nums:
            return values[0] if isinstance(values[0], str) else None
        if all(isinstance(v, int) for v in nums) and len(set(nums)) == 1:
            return nums[0]  # fold numbers etc.
        return float(np.mean(nums))
    return avg(views)


# --------------------------------------------------------------------------- payload
def build_payload(
    pred_sets: Dict[Tuple[str, int, int], pd.DataFrame],
    cfg: TrainConfig,
    quantiles: Sequence[int],
    folds: Optional[pd.DataFrame] = None,
    importance: Optional[pd.DataFrame] = None,
) -> dict:
    by_model: Dict[str, List[Tuple[int, int, pd.DataFrame]]] = {}
    for (model, run, seed), pred in pred_sets.items():
        by_model.setdefault(model, []).append((run, seed, pred))

    models = []
    for model, runs in by_model.items():
        runs.sort(key=lambda r: r[0])
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(runs[0][2][metrics.DATE]).unique()))
        run_views = []
        for run, seed, pred in runs:
            v = _run_view(pred, dates, quantiles, cfg.slippage_pct)
            v.update(key=f"r{run}", label=f"Run {run} (seed {seed})")
            run_views.append(v)
        views = []
        if len(run_views) > 1:
            mean = _mean_view([{k: v[k] for k in ("q", "ic", "lift")} for v in run_views])
            mean.update(key="mean", label=f"Mean of {len(run_views)} runs")
            views.append(mean)
            ens = runs[0][2].copy()
            ens[metrics.P] = np.mean([r[2][metrics.P].to_numpy() for r in runs], axis=0)
            e = _run_view(ens, dates, quantiles, cfg.slippage_pct)
            e.update(key="ensemble", label="Ensemble (averaged predictions)")
            views.append(e)
        views += run_views

        imp = None
        if importance is not None and len(importance):
            sub = importance[importance["model"] == model]
            if len(sub):
                gain = sub.groupby("feature")["gain"].mean()
                share = (gain / gain.sum() * 100).sort_values(ascending=False)
                imp = {"feature": share.index.tolist(), "share": share.tolist()}

        spec = next((m for m in cfg.models if m.name == model), None)
        models.append({
            "name": model,
            "params": spec.params if spec else {},
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "views": views,
            "importance": imp,
            "fold_starts": sorted(pd.to_datetime(runs[0][2].groupby("fold")[metrics.DATE].min())
                                  .dt.strftime("%Y-%m-%d").tolist()),
        })

    has_returns = any("ret" in p.columns and p["ret"].notna().any() for p in pred_sets.values())
    meta = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_path": cfg.data_path,
        "returns_path": cfg.returns_path or (cfg.data_path if cfg.returns_col else None),
        "returns_col": cfg.returns_col,
        "slippage": cfg.slippage_pct,
        "validation": f"{cfg.validation_method}, {cfg.folds} folds, {cfg.training_years:g} training years, "
                      f"{cfg.gap_weeks}-week gap",
        "period": f"{cfg.dataset_start or 'start'} to {cfg.dataset_end or 'end'}",
        "has_returns": has_returns,
        "quantiles": list(quantiles),
        "folds": folds.astype(str).to_dict("records") if folds is not None else [],
    }
    return _clean({"meta": meta, "models": models})


def write_html(path: str, payload: dict, title: str = "Training report") -> None:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


# --------------------------------------------------------------------------- page
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7fb; --card: #ffffff; --surface: #fcfcfb; --soft: #f1f3f9; --line: #e1e5ef;
  --ink: #1f2430; --muted: #4b5563; --faint: #8a8f9c; --grid: #eceef3;
  --high: #2a78d6; --low: #eb6834; --univ: #6b7280; --mid: #b7d3f6; --mid-ink: #5598e7;
  --pos: #2a78d6; --neg: #d03b3b; --shade: rgba(42,120,214,0.06); --sel: #e8f0fb;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #121211; --card: #1a1a19; --surface: #1a1a19; --soft: #242422; --line: #33332f;
    --ink: #f0efec; --muted: #c3c2b7; --faint: #8f8e86; --grid: #2a2a27;
    --high: #3987e5; --low: #d95926; --univ: #a3a29a; --mid: #2d4a6e; --mid-ink: #6da7ec;
    --pos: #3987e5; --neg: #e66767; --shade: rgba(57,135,229,0.10); --sel: #22303f;
  }
}
* { box-sizing: border-box; }
body { font-family: "Segoe UI", -apple-system, Arial, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
main { max-width: 1320px; margin: 0 auto; padding: 20px 24px 40px; }
h1 { margin: 0 0 4px; font-size: 22px; }
h2 { margin: 0 0 4px; font-size: 17px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
.card { background: var(--card); padding: 18px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18,23,40,0.08); margin-top: 16px; }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.pill { padding: 3px 9px; border-radius: 999px; font-size: 12px; background: var(--soft); color: var(--muted); }
.controls { position: sticky; top: 0; z-index: 10; background: var(--bg); padding: 10px 0; display: flex; gap: 18px; flex-wrap: wrap; align-items: center; border-bottom: 1px solid var(--line); }
.controls label { font-size: 12px; color: var(--muted); margin-right: 6px; }
select { padding: 6px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--ink); font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: var(--card); color: var(--muted); padding: 6px 14px; font-size: 13px; cursor: pointer; font-variant-numeric: tabular-nums; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button.on { background: var(--ink); color: var(--card); font-weight: 600; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 14px; }
.kpi { background: var(--soft); border-radius: 10px; padding: 10px 14px; }
.kpi .k { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.kpi .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
.kpi .d { font-size: 11.5px; color: var(--faint); margin-top: 2px; }
.key { display: inline-block; width: 12px; height: 3px; border-radius: 2px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }
.grid2 > .card { margin-top: 0; }
.chart { width: 100%; height: 360px; }
.chart.tall { height: 560px; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { background: var(--soft); text-align: right; font-weight: 600; color: var(--muted); }
th.l, td.l { text-align: left; }
th.grp { text-align: center; border-left: 1px solid var(--line); }
td { text-align: right; font-variant-numeric: tabular-nums; }
td.sep, th.sep { border-left: 1px solid var(--line); }
tr.hl td { background: var(--sel); font-weight: 600; }
tr.mean td { font-weight: 600; }
.neg { color: var(--neg); }
.note { color: var(--muted); font-size: 12.5px; margin-top: 8px; line-height: 1.45; }
.glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px 18px; font-size: 12.5px; color: var(--muted); }
.glossary b { color: var(--ink); }
.hidden { display: none !important; }
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
    <span><label for="model">Model</label><select id="model"></select></span>
    <span><label for="run">Run</label><select id="run"></select></span>
    <span><label>Quantiles</label><span class="seg" id="qseg"></span></span>
  </div>

  <div class="card">
    <h2 id="kpi-title">Headline</h2>
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="card">
    <h2>Quantile counts compared</h2>
    <div class="sub">Same predictions bucketed into 10, 20 and 50 quantiles. High = top bucket, Low = bottom bucket. The selected count is highlighted.</div>
    <div class="table-wrap"><table id="qcompare"></table></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2 id="bars-title">Annualized return by quantile</h2>
      <div class="sub" id="bars-sub"></div>
      <div id="bars" class="chart"></div>
    </div>
    <div class="card port">
      <h2>Growth of 100</h2>
      <div class="sub">High and Low bucket portfolios vs the equal-weighted universe, rebalanced weekly, after slippage. Log scale; shaded bands are alternate folds.</div>
      <div id="growth" class="chart"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card port">
      <h2>High minus Low, 26-week periods</h2>
      <div class="sub">Return of the High portfolio minus the Low portfolio in each 26-week block.</div>
      <div id="periods" class="chart"></div>
    </div>
    <div class="card">
      <h2>Lift chart</h2>
      <div class="sub">All holdout predictions pooled and split into 100 percentiles: mean prediction and mean actual TARGET per percentile (like P123's Lift tab). Does not depend on the quantile count.</div>
      <div id="lift" class="chart"></div>
    </div>
  </div>

  <div class="card">
    <h2>Rank correlation over time</h2>
    <div class="sub">Spearman correlation between prediction and TARGET on each date, with its 13-week moving average. Vertical lines mark the start of each fold's holdout.</div>
    <div id="ic" class="chart"></div>
  </div>

  <div class="card">
    <h2 id="runs-title">All runs</h2>
    <div class="sub">One row per run for every model, at the selected quantile count. Columns follow P123's Compare All table.</div>
    <div class="table-wrap"><table id="runs"></table></div>
  </div>

  <div class="card">
    <h2 id="folds-title">By fold</h2>
    <div class="sub">Each fold's holdout scored on its own.</div>
    <div class="table-wrap"><table id="folds"></table></div>
  </div>

  <div class="card" id="imp-card">
    <h2>Feature importance</h2>
    <div class="sub">Share of total split gain, averaged over runs and folds. Top 25 features.</div>
    <div id="imp" class="chart tall"></div>
  </div>

  <div class="card">
    <h2>Setup</h2>
    <div class="table-wrap"><table id="setup"></table></div>
    <div class="note" id="params"></div>
  </div>

  <div class="card">
    <h2>How to read this</h2>
    <div class="glossary">
      <div><b>RMSE, Spearman, Pearson</b>: prediction vs TARGET over all holdout rows pooled (P123 Lift tab).</div>
      <div><b>IC/date</b>: mean of the per-date Spearman correlations.</div>
      <div><b>Avg%</b>: mean weekly return compounded over 52.18 weeks, after slippage (P123 "Average").</div>
      <div><b>Cmp%</b>: compounded annual return (P123 "Compounded").</div>
      <div><b>SR</b>: mean weekly return x 52.18 / annualized standard deviation. <b>SD</b>: annualized standard deviation.</div>
      <div><b>Turn%</b>: annualized one-way turnover of the bucket portfolio.</div>
      <div><b>H-L</b>: mean weekly High minus Low return, annualized like Avg%.</div>
      <div><b>Tgt H-L</b>: mean TARGET of the High bucket minus the Low bucket (TARGET units).</div>
      <div><b>Univ</b>: equal-weighted average of all stocks, no slippage.</div>
      <div><b>Ensemble</b>: the runs' predictions averaged, then scored once.</div>
    </div>
  </div>
</main>

<script>
const DATA = __DATA__;
const META = DATA.meta;
const state = { model: 0, run: 0, q: String(META.quantiles[0]) };

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 2) => (v === null || v === undefined) ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const el = (tag, attrs = {}, text) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
const model = () => DATA.models[state.model];
const view = () => model().views[state.run];
const qd = (v = view()) => v.q[state.q];

function layout(extra = {}) {
  const ink = css("--muted"), grid = css("--grid");
  return Object.assign({
    margin: { l: 56, r: 16, t: 8, b: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Segoe UI, -apple-system, Arial, sans-serif", size: 12, color: ink },
    xaxis: { gridcolor: grid, zeroline: false, linecolor: grid, tickfont: { color: ink } },
    yaxis: { gridcolor: grid, zerolinecolor: css("--faint"), zerolinewidth: 1, linecolor: grid, tickfont: { color: ink } },
    legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { color: css("--ink") } },
    hoverlabel: { bgcolor: css("--card"), bordercolor: css("--line"), font: { color: css("--ink") } },
    bargap: 0.18,
  }, extra);
}
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

function setup() {
  document.getElementById("subtitle").textContent = `Holdout results, ${META.validation}. Generated ${META.generated}.`;
  const meta = document.getElementById("meta");
  const pills = [`Data: ${META.data_path}`, `Period: ${META.period}`];
  if (META.has_returns) pills.push(`Weekly returns: ${META.returns_col} from ${META.returns_path}`, `Slippage ${META.slippage}%`);
  else pills.push("No weekly returns: portfolio stats unavailable, quantile stats use TARGET");
  pills.forEach(t => meta.appendChild(el("span", { class: "pill" }, t)));

  const ms = document.getElementById("model");
  DATA.models.forEach((m, i) => ms.appendChild(el("option", { value: i }, m.name)));
  ms.onchange = () => { state.model = +ms.value; state.run = 0; fillRuns(); render(); };
  fillRuns();
  document.getElementById("run").onchange = e => { state.run = +e.target.value; render(); };

  const seg = document.getElementById("qseg");
  META.quantiles.forEach(q => {
    const b = el("button", { type: "button", "data-q": q }, String(q));
    b.onclick = () => { state.q = String(q); render(); };
    seg.appendChild(b);
  });
  if (!META.has_returns) document.querySelectorAll(".port").forEach(e => e.classList.add("hidden"));
}
function fillRuns() {
  const rs = document.getElementById("run"); rs.innerHTML = "";
  model().views.forEach((v, i) => rs.appendChild(el("option", { value: i }, v.label)));
  rs.value = state.run;
  document.getElementById("run").parentElement.classList.toggle("hidden", model().views.length < 2);
}

function kpis() {
  const s = qd().stats, box = document.getElementById("kpis"); box.innerHTML = "";
  document.getElementById("kpi-title").textContent = `${model().name} · ${view().label} · ${state.q} quantiles`;
  const items = META.has_returns ? [
    ["H-L Avg%", s.hl_avg, `SR ${fmt(s.hl_sr)}`, null],
    ["High Avg%", s.high_avg, `Cmp ${fmt(s.high_cmp)}% · SR ${fmt(s.high_sr)}`, "--high"],
    ["Low Avg%", s.low_avg, `Cmp ${fmt(s.low_cmp)}% · SR ${fmt(s.low_sr)}`, "--low"],
    ["Universe Avg%", s.univ_avg, `Cmp ${fmt(s.univ_cmp)}%`, "--univ"],
    ["Spearman", s.spearman, `IC/date ${fmt(s.ic_mean, 4)}`, null, 4],
    ["RMSE", s.rmse, `Pearson ${fmt(s.pearson, 4)}`, null, 4],
  ] : [
    ["Tgt H-L", s.tgt_hl, "TARGET units", null, 3],
    ["Tgt High", s.tgt_high, "", "--high", 3], ["Tgt Low", s.tgt_low, "", "--low", 3],
    ["Spearman", s.spearman, `IC/date ${fmt(s.ic_mean, 4)}`, null, 4],
    ["RMSE", s.rmse, `Pearson ${fmt(s.pearson, 4)}`, null, 4],
  ];
  items.forEach(([k, v, d, color, dec = 2]) => {
    const card = el("div", { class: "kpi" }), kk = el("div", { class: "k" });
    if (color) { const key = el("span", { class: "key" }); key.style.background = css(color); kk.appendChild(key); }
    kk.appendChild(document.createTextNode(k)); card.appendChild(kk);
    const vv = el("div", { class: "v" }, fmt(v, dec)); if (v !== null && v < 0) vv.classList.add("neg"); card.appendChild(vv);
    card.appendChild(el("div", { class: "d" }, d)); box.appendChild(card);
  });
}

function table(id, head, rows, opts = {}) {
  const t = document.getElementById(id); t.innerHTML = "";
  if (opts.groups) {
    const tr = el("tr"); opts.groups.forEach(([label, span, cls]) => { const th = el("th", { colspan: span, class: cls || "grp" }, label); tr.appendChild(th); }); t.appendChild(tr);
  }
  const tr = el("tr"); head.forEach(([label, cls]) => tr.appendChild(el("th", { class: cls || "" }, label))); t.appendChild(tr);
  rows.forEach(r => {
    const row = el("tr", r.cls ? { class: r.cls } : {});
    r.cells.forEach(([v, cls, dec]) => {
      const td = el("td", { class: cls || "" }, typeof v === "string" ? v : fmt(v, dec === undefined ? 2 : dec));
      if (typeof v === "number" && v < 0) td.classList.add("neg");
      row.appendChild(td);
    });
    t.appendChild(row);
  });
}

const PORT_HEAD = [["H-L Avg%", "sep"], ["Avg%", "sep"], ["Cmp%"], ["SR"], ["Turn%"], ["Avg%", "sep"], ["SR"], ["SD"], ["Turn%"]];
const portCells = s => [[s.hl_avg, "sep"], [s.high_avg, "sep"], [s.high_cmp], [s.high_sr], [s.high_turn, "", 0], [s.low_avg, "sep"], [s.low_sr], [s.low_sd], [s.low_turn, "", 0]];
const TGT_HEAD = [["Tgt High", "sep"], ["Tgt Low"], ["Tgt H-L"]];
const tgtCells = s => [[s.tgt_high, "sep", 3], [s.tgt_low, "", 3], [s.tgt_hl, "", 3]];

function qcompare() {
  const v = view();
  const head = [["Quantiles", "l"], ["Stocks/bucket"], ...(META.has_returns ? PORT_HEAD : []), ...TGT_HEAD];
  const groups = [["", 2, "l"], ...(META.has_returns ? [["", 1], ["High", 4], ["Low", 4]] : []), ["TARGET", 3]];
  const rows = META.quantiles.map(q => {
    const e = v.q[String(q)], s = e.stats;
    const stocks = e.buckets.stocks ? e.buckets.stocks[e.buckets.stocks.length - 1] : null;
    return { cls: String(q) === state.q ? "hl" : "", cells: [[String(q), "l"], [stocks, "", 0], ...(META.has_returns ? portCells(s) : []), ...tgtCells(s)] };
  });
  table("qcompare", head, rows, { groups });
}

function runsTable() {
  document.getElementById("runs-title").textContent = `All runs · ${state.q} quantiles`;
  const head = [["Model", "l"], ["Run", "l"], ["RMSE", "sep"], ["Spearman"], ["Pearson"], ["IC/date"], ...(META.has_returns ? PORT_HEAD : []), ...TGT_HEAD];
  const groups = [["", 2, "l"], ["Accuracy", 4], ...(META.has_returns ? [["H-L", 1], ["High", 4], ["Low", 4]] : []), ["TARGET", 3]];
  const rows = [];
  DATA.models.forEach((m, mi) => m.views.forEach((v, vi) => {
    const s = v.q[state.q].stats;
    const cls = (mi === state.model && vi === state.run) ? "hl" : (v.key === "mean" ? "mean" : "");
    rows.push({ cls, cells: [[m.name, "l"], [v.label, "l"], [s.rmse, "sep", 4], [s.spearman, "", 4], [s.pearson, "", 4], [s.ic_mean, "", 4], ...(META.has_returns ? portCells(s) : []), ...tgtCells(s)] });
  }));
  table("runs", head, rows, { groups });
}

function foldsTable() {
  document.getElementById("folds-title").textContent = `By fold · ${view().label} · ${state.q} quantiles`;
  const head = [["Fold", "l"], ["Holdout", "l"], ["Spearman", "sep"], ["IC/date"], ...(META.has_returns ? [["H-L Avg%", "sep"], ["High Avg%"], ["Low Avg%"], ["Univ Avg%"]] : []), ["Tgt H-L", "sep"]];
  const rows = qd().folds.map(f => ({ cells: [[String(f.fold), "l"], [`${f.start} to ${f.end}`, "l"], [f.spearman, "sep", 4], [f.ic_mean, "", 4],
    ...(META.has_returns ? [[f.hl_avg, "sep"], [f.high_avg], [f.low_avg], [f.univ_avg]] : []), [f.tgt_hl, "sep", 3]] }));
  table("folds", head, rows);
}

function bars() {
  const e = qd(), n = +state.q, port = META.has_returns;
  const y = port ? e.buckets.avg : e.buckets.tgt;
  const x = y.map((_, i) => i + 1);
  const colors = x.map(i => i === n ? css("--high") : i === 1 ? css("--low") : css("--mid"));
  document.getElementById("bars-title").textContent = port ? `Annualized return by quantile · ${n}` : `Mean TARGET by quantile · ${n}`;
  document.getElementById("bars-sub").textContent = port
    ? `Avg% of each bucket's equal-weighted portfolio, rebalanced weekly, after slippage. Dashed line: universe (${fmt(e.stats.univ_avg)}%).`
    : "Mean TARGET of each bucket, averaged over dates.";
  const custom = x.map((_, i) => [port ? e.buckets.cmp[i] : null, port ? e.buckets.turn[i] : null, e.buckets.stocks ? e.buckets.stocks[i] : null]);
  const traces = [{
    type: "bar", x, y, marker: { color: colors, cornerradius: 4 }, customdata: custom, name: "Bucket",
    hovertemplate: port ? "<b>%{y:.2f}%</b> Avg<br>Bucket %{x}<br>Cmp %{customdata[0]:.2f}% · Turn %{customdata[1]:,.0f}%<br>%{customdata[2]:.0f} stocks<extra></extra>"
                        : "<b>%{y:.3f}</b><br>Bucket %{x}<extra></extra>",
  }];
  const shapes = [];
  if (port && e.stats.univ_avg !== null) {
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: e.stats.univ_avg, y1: e.stats.univ_avg, line: { color: css("--univ"), width: 1.5, dash: "dash" } });
  }
  const ann = [
    { x: n, y: y[n - 1], text: "High", showarrow: false, yshift: y[n - 1] >= 0 ? 12 : -12, font: { color: css("--ink"), size: 11 } },
    { x: 1, y: y[0], text: "Low", showarrow: false, yshift: y[0] >= 0 ? 12 : -12, font: { color: css("--ink"), size: 11 } },
  ];
  Plotly.react("bars", traces, layout({
    shapes, annotations: ann, showlegend: false, bargap: n >= 50 ? 0.12 : 0.22,
    xaxis: Object.assign(layout().xaxis, { title: { text: "Quantile (1 = lowest predictions)" }, dtick: n <= 20 ? 1 : 5 }),
    yaxis: Object.assign(layout().yaxis, { title: { text: port ? "Avg% (annualized)" : "Mean TARGET" }, ticksuffix: port ? "%" : "" }),
  }), CFG);
}

function foldShapes(dates) {
  const starts = model().fold_starts, shapes = [];
  starts.forEach((s, i) => {
    const end = i + 1 < starts.length ? starts[i + 1] : dates[dates.length - 1];
    if (i % 2 === 0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: s, x1: end, y0: 0, y1: 1, fillcolor: css("--shade"), line: { width: 0 }, layer: "below" });
  });
  return shapes;
}

function growth() {
  if (!META.has_returns) return;
  const w = qd().weekly, dates = model().dates;
  const cum = arr => { let v = 100; return arr.map(r => (v *= 1 + (r || 0))); };
  const line = (key, name, color, dash) => ({ type: "scatter", mode: "lines", x: dates, y: cum(w[key]), name, line: { color, width: 2, dash }, hovertemplate: `%{y:.1f}<extra>${name}</extra>` });
  Plotly.react("growth", [line("high", "High", css("--high")), line("univ", "Universe", css("--univ"), "dash"), line("low", "Low", css("--low"))],
    layout({ hovermode: "x unified", shapes: foldShapes(dates), yaxis: Object.assign(layout().yaxis, { type: "log", title: { text: "Value (start = 100)" } }) }), CFG);
}

function periods() {
  if (!META.has_returns) return;
  const w = qd().weekly, dates = model().dates, x = [], y = [], text = [];
  for (let i = 0; i < dates.length; i += 26) {
    let h = 1, l = 1;
    for (let j = i; j < Math.min(i + 26, dates.length); j++) { h *= 1 + (w.high[j] || 0); l *= 1 + (w.low[j] || 0); }
    const end = dates[Math.min(i + 25, dates.length - 1)];
    x.push(dates[i]); y.push((h - l) * 100); text.push(`${dates[i]} to ${end}`);
  }
  Plotly.react("periods", [{ type: "bar", x, y, text, textposition: "none", marker: { color: y.map(v => v >= 0 ? css("--pos") : css("--neg")), cornerradius: 4 },
    hovertemplate: "<b>%{y:.2f}%</b><br>%{text}<extra>High − Low</extra>", xperiod: 0 }],
    layout({ showlegend: false, yaxis: Object.assign(layout().yaxis, { ticksuffix: "%", title: { text: "High − Low return" } }) }), CFG);
}

function lift() {
  const l = view().lift, x = l.pred.map((_, i) => i);
  Plotly.react("lift", [
    { type: "scatter", mode: "lines", x, y: l.pred, name: "Predicted", line: { color: css("--high"), width: 2 }, hovertemplate: "%{y:.3f}<extra>Predicted</extra>" },
    { type: "scatter", mode: "lines", x, y: l.actual, name: "Actual", line: { color: css("--low"), width: 2 }, hovertemplate: "%{y:.3f}<extra>Actual</extra>" },
  ], layout({ hovermode: "x unified", xaxis: Object.assign(layout().xaxis, { title: { text: "Prediction percentile" } }), yaxis: Object.assign(layout().yaxis, { title: { text: "Mean TARGET" } }) }), CFG);
}

function icChart() {
  const ic = view().ic, dates = model().dates, roll = [];
  for (let i = 0; i < ic.length; i++) {
    const win = ic.slice(Math.max(0, i - 12), i + 1).filter(v => v !== null);
    roll.push(i >= 12 && win.length ? win.reduce((a, b) => a + b, 0) / win.length : null);
  }
  const shapes = model().fold_starts.map(s => ({ type: "line", xref: "x", yref: "paper", x0: s, x1: s, y0: 0, y1: 1, line: { color: css("--faint"), width: 1, dash: "dot" } }));
  Plotly.react("ic", [
    { type: "bar", x: dates, y: ic, name: "Weekly", marker: { color: css("--mid") }, hovertemplate: "%{y:.3f}<extra>Weekly</extra>" },
    { type: "scatter", mode: "lines", x: dates, y: roll, name: "13-week average", line: { color: css("--high"), width: 2 }, hovertemplate: "%{y:.3f}<extra>13-week avg</extra>" },
  ], layout({ hovermode: "x unified", bargap: 0.1, shapes, yaxis: Object.assign(layout().yaxis, { title: { text: "Spearman" } }) }), CFG);
}

function importance() {
  const imp = model().importance, card = document.getElementById("imp-card");
  if (!imp) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  const k = Math.min(25, imp.feature.length);
  const f = imp.feature.slice(0, k).reverse(), s = imp.share.slice(0, k).reverse();
  const short = f.map(t => t.length > 48 ? t.slice(0, 46) + "…" : t);
  Plotly.react("imp", [{ type: "bar", orientation: "h", x: s, y: short, customdata: f, marker: { color: css("--high"), cornerradius: 4 },
    hovertemplate: "<b>%{x:.2f}%</b><br>%{customdata}<extra></extra>" }],
    layout({ showlegend: false, margin: { l: 300, r: 16, t: 8, b: 40 }, bargap: 0.25, xaxis: Object.assign(layout().xaxis, { ticksuffix: "%", title: { text: "Share of gain" } }), yaxis: Object.assign(layout().yaxis, { automargin: true }) }), CFG);
}

function setupTable() {
  const head = [["Fold", "l"], ["Train", "l"], ["Holdout", "l"], ["Train rows"], ["Holdout rows"]];
  const rows = META.folds.map(f => ({ cells: [[f.fold, "l"], [`${f.train_start} to ${f.train_end}`, "l"], [`${f.holdout_start} to ${f.holdout_end}`, "l"], [Number(f.train_rows).toLocaleString()], [Number(f.holdout_rows).toLocaleString()]] }));
  table("setup", head, rows);
  document.getElementById("params").textContent = `Hyperparameters (${model().name}): ${JSON.stringify(model().params)}`;
}

function render() {
  document.querySelectorAll("#qseg button").forEach(b => b.classList.toggle("on", b.dataset.q === state.q));
  kpis(); qcompare(); bars(); growth(); periods(); lift(); icChart(); runsTable(); foldsTable(); importance(); setupTable();
}
setup(); render();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""

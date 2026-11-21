from __future__ import annotations

import html
import json
from typing import Any, Dict, List

CSS = """
:root {
  --ink: #1f2430;
  --muted: #4b5563;
  --bg: #f6f7fb;
  --card: #ffffff;
  --accent: #0f4c5c;
  --viz: #2a78d6;
  --viz-soft: rgba(42, 120, 214, 0.18);
  --danger: #b91c1c;
  --line: #e1e5ef;
}
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; margin: 24px; background: var(--bg); color: var(--ink); }
h1 { margin: 0 0 6px; font-size: 22px; }
h2 { margin: 0 0 12px; font-size: 17px; }
h3 { margin: 0 0 8px; font-size: 15px; }
.header, .section { background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.section { margin-top: 20px; }
.metrics { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
.metric { background: #f1f3f9; padding: 10px 14px; border-radius: 10px; min-width: 130px; }
.metric .label { font-size: 12px; color: var(--muted); }
.metric .value { font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 13px; }
.controls { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
.controls input[type="text"] { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); min-width: 260px; }
.controls label, .controls select { font-size: 13px; color: var(--muted); }
.controls select { padding: 6px 8px; border-radius: 8px; border: 1px solid var(--line); }
.table-wrap { overflow: auto; max-height: 640px; border: 1px solid var(--line); border-radius: 8px; }
.table { border-collapse: collapse; width: 100%; }
.table th, .table td { padding: 6px 9px; border-bottom: 1px solid var(--line); font-size: 12.5px; white-space: nowrap; }
.table th { text-align: left; background: #f1f3f9; position: sticky; top: 0; cursor: pointer; z-index: 2; user-select: none; }
.table th.sorted-asc::after { content: " \\25B2"; font-size: 9px; }
.table th.sorted-desc::after { content: " \\25BC"; font-size: 9px; }
.table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.table td.name { max-width: 320px; overflow: hidden; text-overflow: ellipsis; }
.table tbody tr { cursor: pointer; }
.table tbody tr:hover { background: #fafbff; }
.table tbody tr.selected { background: #e8f0fb; }
.spark rect { fill: var(--viz); }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11.5px; background: #f1f3f9; color: var(--muted); margin: 1px 4px 1px 0; }
.pill.warn { background: #fee2e2; color: var(--danger); }
.pill.btn { cursor: pointer; border: 1px solid transparent; }
.pill.btn.active { border-color: var(--ink); }
.detail-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; margin: 12px 0 16px; }
.stat { background: #f7f8fc; border-radius: 8px; padding: 6px 10px; }
.stat .k { font-size: 11.5px; color: var(--muted); }
.stat .v { font-size: 14px; font-variant-numeric: tabular-nums; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.chart { width: 100%; height: 340px; }
.chart.small { height: 200px; }
.glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
.glossary-item { padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; background: #fafbff; }
.glossary-item h4 { margin: 0 0 4px; font-size: 13.5px; }
.glossary-item p { margin: 0; font-size: 12.5px; color: var(--muted); line-height: 1.4; }
"""

GLOSSARY = {
    "Distribution": "Mini histogram over the full value range (or the clipped range if hist_clip_quantile is set). Click a row for the full histogram.",
    "Missing %": "Share of rows where the value is empty or +/-inf.",
    "Mean / Std": "Mean and standard deviation over all rows with a value.",
    "P1 / Median / P99": "1st percentile, median and 99th percentile.",
    "Skew": "Asymmetry. Near 0 is symmetric; positive means a long right tail.",
    "Kurtosis": "Excess kurtosis (normal = 0). High values mean a few extreme values dominate.",
    "At min / At max": "Share of values exactly equal to the min / max. High values usually mean clipping or trimming (e.g. P123 outlier limits).",
    "Outliers": "Share of values more than 3 standard deviations from the mean.",
    "Coverage": "Average share of stocks with a value on each date.",
    "Drift": "Std of the per-date mean divided by the overall std. Near 0 for features normalized by date; high values mean the level moves over time.",
    "Date-constant": "Share of dates where every stock has the same value (macro / market-level series).",
}

TABLE_COLUMNS = [
    # key, label, kind
    ("feature", "Feature", "text"),
    ("spark", "Distribution", "spark"),
    ("missing_pct", "Missing %", "pct"),
    ("mean", "Mean", "num"),
    ("std", "Std", "num"),
    ("min", "Min", "num"),
    ("p01", "P1", "num"),
    ("p50", "Median", "num"),
    ("p99", "P99", "num"),
    ("max", "Max", "num"),
    ("skew", "Skew", "num"),
    ("kurtosis", "Kurtosis", "num"),
    ("unique", "Unique", "int"),
    ("pct_at_min", "At min", "pct"),
    ("pct_at_max", "At max", "pct"),
    ("coverage_mean", "Coverage", "pct"),
    ("drift_ratio", "Drift", "num"),
    ("flags", "Flags", "flags"),
]


def spark_svg(hist: Dict[str, Any], width: int = 120, height: int = 26) -> str:
    counts: List[int] = hist.get("counts") or []
    if not counts:
        return ""
    peak = max(counts) or 1
    n = len(counts)
    slot = width / n
    gap = 1 if slot >= 2.5 else 0
    bars = []
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        h = max(1.0, c / peak * (height - 2))
        bars.append(
            f'<rect x="{i * slot:.2f}" y="{height - h:.2f}" width="{max(slot - gap, 0.6):.2f}" height="{h:.2f}"/>'
        )
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'aria-hidden="true">{"".join(bars)}</svg>'
    )


def _json_for_script(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def generate_report(results: Dict[str, Any], path: str, stats_csv_name: str = "stats.csv") -> str:
    meta = results["meta"]
    features = results["features"]
    flag_info = results.get("flag_info", {})

    sparks = {f["feature"]: spark_svg(results["histograms"].get(f["feature"], {})) for f in features}
    flag_counts: Dict[str, int] = {}
    for f in features:
        for flag in f["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    n_warn = sum(
        1 for f in features if any(flag_info.get(fl, {}).get("kind") == "warn" for fl in f["flags"])
    )

    def metric(label: str, value: Any) -> str:
        return f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div></div>'

    metrics = [
        metric("Rows", f"{meta['rows']:,}"),
        metric("Numeric features", meta["numeric_features"]),
        metric("Categorical", meta["categorical_features"]),
    ]
    if meta.get("dates"):
        metrics += [
            metric("Dates", f"{meta['dates']:,}"),
            metric("Date range", f"{meta['date_start']} to {meta['date_end']}"),
            metric("Avg stocks / date", meta.get("rows_per_date_mean")),
        ]
    metrics.append(metric("Features with warnings", n_warn))

    flag_pills = "".join(
        f'<span class="pill btn {"warn" if flag_info.get(k, {}).get("kind") == "warn" else ""}" '
        f'data-flag="{html.escape(k)}" title="{html.escape(flag_info.get(k, {}).get("description", ""))}">'
        f"{html.escape(k)} ({v})</span>"
        for k, v in sorted(flag_counts.items(), key=lambda kv: (flag_info.get(kv[0], {}).get("kind") != "warn", kv[0]))
    ) or '<span class="note">No flags raised.</span>'

    header_cells = "".join(
        f'<th data-key="{key}" data-kind="{kind}">{html.escape(label)}</th>' for key, label, kind in TABLE_COLUMNS
    )
    glossary = "".join(
        f'<div class="glossary-item"><h4>{html.escape(k)}</h4><p>{html.escape(v)}</p></div>'
        for k, v in GLOSSARY.items()
    )
    flag_glossary = "".join(
        f'<div class="glossary-item"><h4><span class="pill {"warn" if v["kind"] == "warn" else ""}">{html.escape(k)}</span></h4>'
        f'<p>{html.escape(v["description"])}</p></div>'
        for k, v in flag_info.items()
    )
    cfg = meta.get("config", {})
    hist_note = (
        f"Histograms use {cfg.get('hist_bins')} bins"
        + (
            f", clipped to the {cfg.get('hist_clip_quantile'):.1%}-{1 - cfg.get('hist_clip_quantile'):.1%} quantile range"
            if cfg.get("hist_clip_quantile")
            else " over each feature's full min-max range"
        )
        + "."
    )
    ts_bucket = results.get("timeseries", {}).get("bucket")

    payload = {
        "features": features,
        "histograms": results["histograms"],
        "timeseries": results.get("timeseries", {}),
        "categorical": results.get("categorical", []),
        "flagInfo": flag_info,
        "sparks": sparks,
        "columns": [{"key": k, "label": l, "kind": kd} for k, l, kd in TABLE_COLUMNS],
    }

    categorical_section = ""
    if results.get("categorical"):
        categorical_section = """
  <div class="section">
    <h2>Categorical features</h2>
    <div id="cat-charts" class="chart-grid"></div>
  </div>"""

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature statistics - {html.escape(str(meta['data_path']))}</title>
  <style>{CSS}</style>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
  <div class="header">
    <h1>Feature statistics</h1>
    <div class="note">Dataset: <code>{html.escape(str(meta['data_path']))}</code> &middot; target: <code>{html.escape(str(meta.get('target')))}</code> &middot; <a href="{html.escape(stats_csv_name)}">{html.escape(stats_csv_name)}</a></div>
    <div class="metrics">{''.join(metrics)}</div>
  </div>

  <div class="section">
    <h2>Flags</h2>
    <div>{flag_pills}</div>
    <p class="note">Click a flag to filter the table. Red flags are worth reviewing; grey ones describe the feature.</p>
  </div>

  <div class="section">
    <h2>Numeric features</h2>
    <div class="controls">
      <input type="text" id="search" placeholder="Search features...">
      <label><input type="checkbox" id="warn-only"> Only features with warnings</label>
      <span class="note" id="row-count"></span>
    </div>
    <div class="table-wrap">
      <table class="table" id="stats-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <p class="note">{html.escape(hist_note)} Click a column header to sort, and a row to see its charts below.</p>
  </div>

  <div class="section" id="detail">
    <div class="detail-head">
      <h2 id="detail-title">Feature detail</h2>
      <label class="note"><input type="checkbox" id="log-y"> Log scale (histogram)</label>
    </div>
    <div id="detail-flags"></div>
    <div class="stat-grid" id="detail-stats"></div>
    <div class="chart-grid">
      <div><h3>Histogram</h3><div id="chart-hist" class="chart"></div><div class="note" id="hist-note"></div></div>
      <div><h3>Distribution over time ({html.escape(str(ts_bucket or 'n/a'))} buckets)</h3><div id="chart-time" class="chart"></div></div>
    </div>
    <h3>Missing values over time</h3>
    <div id="chart-missing" class="chart small"></div>
  </div>
{categorical_section}
  <div class="section">
    <h2>Glossary</h2>
    <div class="glossary">{glossary}</div>
    <h3 style="margin-top:16px">Flags</h3>
    <div class="glossary">{flag_glossary}</div>
  </div>

<script>
const DATA = {_json_for_script(payload)};
const VIZ = "#2a78d6", VIZ_SOFT = "rgba(42,120,214,0.18)", INK = "#1f2430", MUTED = "#6b7280", GRID = "#eceff5";
const BASE_LAYOUT = {{
  margin: {{l: 56, r: 16, t: 10, b: 44}}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: {{family: "Segoe UI, Arial, sans-serif", size: 12, color: INK}},
  xaxis: {{gridcolor: GRID, zeroline: false, linecolor: "#cfd5e2"}},
  yaxis: {{gridcolor: GRID, zeroline: false, linecolor: "#cfd5e2"}},
  hoverlabel: {{bgcolor: "#111827", font: {{color: "#f9fafb"}}}}, showlegend: false,
}};
const PLOT_CFG = {{displaylogo: false, responsive: true}};
// Plotly mutates layout objects, so every chart gets its own deep copy.
function layout(extra) {{
  const base = JSON.parse(JSON.stringify(BASE_LAYOUT));
  for (const [k, v] of Object.entries(extra || {{}})) {{
    base[k] = (v && typeof v === "object" && !Array.isArray(v) && base[k] && typeof base[k] === "object") ? Object.assign(base[k], v) : v;
  }}
  return base;
}}
const hasPlotly = () => typeof Plotly !== "undefined";

function fmt(v) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  if (a >= 1000) return v.toLocaleString(undefined, {{maximumFractionDigits: 1}});
  return Number(v.toPrecision(4)).toString();
}}
function pct(v) {{ return (v === null || v === undefined) ? "" : (v * 100).toFixed(v > 0 && v < 0.001 ? 3 : 1) + "%"; }}
function esc(s) {{ return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[c]); }}
function flagPill(f) {{
  const info = DATA.flagInfo[f] || {{}};
  return `<span class="pill ${{info.kind === "warn" ? "warn" : ""}}" title="${{esc(info.description || "")}}">${{esc(f)}}</span>`;
}}
function hasWarn(row) {{ return row.flags.some(f => (DATA.flagInfo[f] || {{}}).kind === "warn"); }}
function cell(row, col) {{
  const v = row[col.key];
  switch (col.kind) {{
    case "text": return `<td class="name" title="${{esc(v)}}">${{esc(v)}}</td>`;
    case "spark": return `<td>${{DATA.sparks[row.feature] || ""}}</td>`;
    case "pct": return `<td class="num">${{pct(v)}}</td>`;
    case "int": return `<td class="num">${{v === null || v === undefined ? "" : v.toLocaleString()}}</td>`;
    case "flags": return `<td>${{row.flags.map(flagPill).join("")}}</td>`;
    default: return `<td class="num">${{fmt(v)}}</td>`;
  }}
}}

const state = {{sortKey: null, sortDir: 1, search: "", warnOnly: false, flag: null, selected: null}};
const tbody = document.querySelector("#stats-table tbody");

function renderTable() {{
  let rows = DATA.features.slice();
  if (state.search) rows = rows.filter(r => r.feature.toLowerCase().includes(state.search));
  if (state.warnOnly) rows = rows.filter(hasWarn);
  if (state.flag) rows = rows.filter(r => r.flags.includes(state.flag));
  if (state.sortKey) {{
    const k = state.sortKey;
    rows.sort((a, b) => {{
      let x = k === "flags" ? a.flags.length : a[k], y = k === "flags" ? b.flags.length : b[k];
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      if (typeof x === "string") return state.sortDir * x.localeCompare(y);
      return state.sortDir * (x - y);
    }});
  }}
  tbody.innerHTML = rows.map(r =>
    `<tr data-feature="${{esc(r.feature)}}" class="${{r.feature === state.selected ? "selected" : ""}}">` +
    DATA.columns.map(c => cell(r, c)).join("") + "</tr>").join("");
  document.getElementById("row-count").textContent = `${{rows.length}} of ${{DATA.features.length}} shown`;
}}

document.querySelectorAll("#stats-table th").forEach(th => th.addEventListener("click", () => {{
  const key = th.dataset.key;
  if (key === "spark") return;
  state.sortDir = state.sortKey === key ? -state.sortDir : (th.dataset.kind === "text" ? 1 : -1);
  state.sortKey = key;
  document.querySelectorAll("#stats-table th").forEach(h => h.classList.remove("sorted-asc", "sorted-desc"));
  th.classList.add(state.sortDir === 1 ? "sorted-asc" : "sorted-desc");
  renderTable();
}}));
document.getElementById("search").addEventListener("input", e => {{ state.search = e.target.value.trim().toLowerCase(); renderTable(); }});
document.getElementById("warn-only").addEventListener("change", e => {{ state.warnOnly = e.target.checked; renderTable(); }});
document.querySelectorAll(".pill.btn").forEach(p => p.addEventListener("click", () => {{
  state.flag = state.flag === p.dataset.flag ? null : p.dataset.flag;
  document.querySelectorAll(".pill.btn").forEach(q => q.classList.toggle("active", q.dataset.flag === state.flag));
  renderTable();
}}));
tbody.addEventListener("click", e => {{
  const tr = e.target.closest("tr");
  if (!tr) return;
  selectFeature(tr.dataset.feature, true);
}});
document.getElementById("log-y").addEventListener("change", () => state.selected && drawHistogram(state.selected));

const STAT_FIELDS = [
  ["count", "Count", "int"], ["missing_pct", "Missing", "pct"], ["unique", "Unique values", "int"],
  ["mean", "Mean", "num"], ["std", "Std dev", "num"], ["min", "Min", "num"], ["p01", "P1", "num"],
  ["p05", "P5", "num"], ["p25", "P25", "num"], ["p50", "Median", "num"], ["p75", "P75", "num"],
  ["p95", "P95", "num"], ["p99", "P99", "num"], ["max", "Max", "num"], ["skew", "Skew", "num"],
  ["kurtosis", "Excess kurtosis", "num"], ["zero_pct", "Zeros", "pct"], ["pct_at_min", "At min", "pct"],
  ["pct_at_max", "At max", "pct"], ["outlier_pct", "Beyond 3 std", "pct"], ["inf_count", "Inf values", "int"],
  ["first_date", "First date", "text"], ["last_date", "Last date", "text"], ["coverage_mean", "Avg coverage", "pct"],
  ["coverage_min", "Min coverage", "pct"], ["within_date_std_mean", "Avg within-date std", "num"],
  ["drift_ratio", "Drift ratio", "num"], ["date_constant_pct", "Date-constant", "pct"],
];

function drawHistogram(name) {{
  const h = DATA.histograms[name];
  const note = document.getElementById("hist-note");
  if (!hasPlotly()) {{ note.textContent = "Plotly could not load (offline?), so charts are unavailable."; return; }}
  if (!h || !h.counts.length) {{ Plotly.purge("chart-hist"); note.textContent = "No values."; return; }}
  const mids = [], hover = [];
  for (let i = 0; i < h.counts.length; i++) {{
    mids.push((h.edges[i] + h.edges[i + 1]) / 2);
    hover.push(`${{fmt(h.edges[i])}} to ${{fmt(h.edges[i + 1])}}<br>${{h.counts[i].toLocaleString()}} rows`);
  }}
  const logY = document.getElementById("log-y").checked;
  Plotly.react("chart-hist", [{{
    type: "bar", x: mids, y: h.counts, marker: {{color: VIZ}},
    hovertext: hover, hoverinfo: "text",
  }}], layout({{
    bargap: 0.08, xaxis: {{title: {{text: "Value"}}, type: "linear"}},
    yaxis: {{title: {{text: "Rows"}}, type: logY ? "log" : "linear"}},
  }}), PLOT_CFG);
  const clipped = h.clipped_low + h.clipped_high;
  note.textContent = clipped ? `${{clipped.toLocaleString()}} values outside the plotted range (${{h.clipped_low.toLocaleString()}} below, ${{h.clipped_high.toLocaleString()}} above).` : "";
}}

function drawTime(name) {{
  const ts = DATA.timeseries;
  if (!hasPlotly() || !ts || !ts.series || !ts.series[name]) return;
  const s = ts.series[name], x = ts.periods;
  Plotly.react("chart-time", [
    {{x, y: s.p95, type: "scatter", mode: "lines", line: {{width: 0}}, hoverinfo: "skip", showlegend: false}},
    {{x, y: s.p05, type: "scatter", mode: "lines", name: "P5 to P95", line: {{width: 0}}, fill: "tonexty", fillcolor: VIZ_SOFT, hoverinfo: "skip"}},
    {{x, y: s.p50, type: "scatter", mode: "lines", name: "Median", line: {{color: VIZ, width: 2}},
      customdata: x.map((_, i) => [s.p05[i], s.p95[i], s.mean[i]]),
      hovertemplate: "%{{x|%b %Y}}<br>Median %{{y:.4g}}<br>P5-P95 %{{customdata[0]:.4g}} to %{{customdata[1]:.4g}}<br>Mean %{{customdata[2]:.4g}}<extra></extra>"}},
    {{x, y: s.mean, type: "scatter", mode: "lines", name: "Mean", line: {{color: MUTED, width: 1.5, dash: "dot"}}, hoverinfo: "skip"}},
  ], layout({{
    showlegend: true, legend: {{orientation: "h", y: 1.1, x: 0}},
    hovermode: "x unified", xaxis: {{type: "date"}},
    yaxis: {{title: {{text: "Value"}}}},
  }}), PLOT_CFG);
  Plotly.react("chart-missing", [{{
    x, y: s.missing_pct, type: "scatter", mode: "lines", line: {{color: VIZ, width: 2}},
    hovertemplate: "%{{x|%b %Y}}<br>Missing %{{y:.1%}}<extra></extra>",
  }}], layout({{
    margin: {{l: 56, r: 16, t: 6, b: 30}}, xaxis: {{type: "date"}},
    yaxis: {{tickformat: ".0%", rangemode: "tozero"}},
  }}), PLOT_CFG);
}}

function selectFeature(name, scroll) {{
  state.selected = name;
  const row = DATA.features.find(r => r.feature === name);
  if (!row) return;
  document.querySelectorAll("#stats-table tbody tr").forEach(tr => tr.classList.toggle("selected", tr.dataset.feature === name));
  document.getElementById("detail-title").textContent = name + (row.role === "target" ? " (target)" : "");
  document.getElementById("detail-flags").innerHTML = row.flags.map(flagPill).join("");
  document.getElementById("detail-stats").innerHTML = STAT_FIELDS.filter(([k]) => row[k] !== undefined).map(([k, label, kind]) => {{
    const v = row[k];
    const txt = kind === "pct" ? pct(v) : kind === "int" ? (v ?? "").toLocaleString() : kind === "text" ? esc(v ?? "") : fmt(v);
    return `<div class="stat"><div class="k">${{label}}</div><div class="v">${{txt}}</div></div>`;
  }}).join("");
  drawHistogram(name);
  drawTime(name);
  if (scroll) document.getElementById("detail").scrollIntoView({{behavior: "smooth", block: "start"}});
}}

function drawCategorical() {{
  const wrap = document.getElementById("cat-charts");
  if (!wrap || !hasPlotly()) return;
  DATA.categorical.forEach((c, i) => {{
    const div = document.createElement("div");
    const top = c.top.slice().reverse();
    div.innerHTML = `<h3>${{esc(c.feature)}}</h3><div class="note">${{c.categories}} categories &middot; missing ${{pct(c.missing_pct)}}${{c.other_count ? ` &middot; ${{c.other_count.toLocaleString()}} rows in categories not shown` : ""}}</div><div id="cat-${{i}}" style="height:${{Math.max(220, top.length * 18 + 60)}}px"></div>`;
    wrap.appendChild(div);
    Plotly.newPlot(`cat-${{i}}`, [{{
      type: "bar", orientation: "h", y: top.map(t => t.value), x: top.map(t => t.count), marker: {{color: VIZ}},
      customdata: top.map(t => t.pct), hovertemplate: "%{{y}}: %{{x:,}} rows (%{{customdata:.1%}})<extra></extra>",
    }}], layout({{
      margin: {{l: 70, r: 16, t: 6, b: 36}}, bargap: 0.3,
      yaxis: {{type: "category", automargin: true}},
      xaxis: {{title: {{text: "Rows"}}, type: "linear"}},
    }}), PLOT_CFG);
  }});
}}

renderTable();
const firstFeature = DATA.features.find(r => r.role !== "target") || DATA.features[0];
if (firstFeature) selectFeature(firstFeature.feature, false);
drawCategorical();
</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return path

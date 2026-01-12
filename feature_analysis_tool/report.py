from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd


CSS = """
:root {
  --ink: #1f2430;
  --muted: #4b5563;
  --bg: #f6f7fb;
  --card: #ffffff;
  --accent: #0f4c5c;
  --accent-2: #e36414;
  --danger: #b91c1c;
  --line: #e1e5ef;
}
body { font-family: "Segoe UI", Arial, sans-serif; margin: 24px; background: var(--bg); color: var(--ink); }
.header { background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.section { margin-top: 24px; background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.metrics { display: flex; gap: 16px; flex-wrap: wrap; }
.metric { background: #f1f3f9; padding: 12px 16px; border-radius: 10px; min-width: 160px; }
.table-wrap { overflow-x: auto; }
.table { border-collapse: collapse; width: 100%; }
.table th, .table td { padding: 8px 10px; border-bottom: 1px solid var(--line); font-size: 13px; }
.table th { text-align: left; background: #f1f3f9; position: sticky; top: 0; cursor: pointer; overflow: visible; }
.th-wrap { display: inline-flex; align-items: center; gap: 6px; }
.tooltip { position: relative; display: inline-flex; width: 16px; height: 16px; border-radius: 50%; background: #d8dde8; color: #243043; font-size: 11px; align-items: center; justify-content: center; cursor: help; }
.tooltip::after, .tooltip::before { opacity: 0; transform: translate(-50%, 4px); transition: opacity 60ms ease-out, transform 60ms ease-out; pointer-events: none; }
.tooltip:hover::after, .tooltip:focus::after { content: attr(data-tip); position: absolute; left: 50%; transform: translate(-50%, 0); bottom: 24px; background: #111827; color: #f9fafb; padding: 8px 10px; border-radius: 8px; font-size: 12px; line-height: 1.2; white-space: nowrap; box-shadow: 0 8px 16px rgba(0,0,0,0.2); z-index: 20; opacity: 1; }
.tooltip:hover::before, .tooltip:focus::before { content: ""; position: absolute; left: 50%; transform: translate(-50%, 0); bottom: 18px; border-width: 6px; border-style: solid; border-color: #111827 transparent transparent transparent; z-index: 19; opacity: 1; }
.glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.glossary-item { padding: 12px; border: 1px solid var(--line); border-radius: 10px; background: #fafbff; }
.glossary-item h4 { margin: 0 0 6px; font-size: 14px; }
.glossary-item p { margin: 0; font-size: 12.5px; color: var(--muted); line-height: 1.4; }
.table tr:hover { background: #fafbff; }
.flag { color: var(--danger); font-weight: 600; }
.note { color: var(--muted); font-size: 13px; }
.controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.controls input[type="text"] { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); min-width: 260px; }
.controls label { font-size: 13px; color: var(--muted); }
.bar-wrap { width: 120px; background: #eef1f8; border-radius: 999px; height: 8px; overflow: hidden; display: inline-block; margin-right: 8px; vertical-align: middle; }
.bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.num { font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #f1f3f9; color: var(--muted); margin-right: 6px; }
.pill.danger { background: #fee2e2; color: var(--danger); }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.chart { width: 100%; height: 360px; }
"""


def _format_float(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


METRIC_TOOLTIPS = {
    "feature": "Feature name used in training and diagnostics.",
    "importance_gain": "LightGBM total gain from splits using this feature.",
    "importance_split": "LightGBM split count using this feature.",
    "shap_mean_abs": "Mean absolute SHAP value for this feature.",
    "corr": "Correlation with target (Pearson or Spearman, per data type).",
    "mutual_info": "Mutual information with the target.",
    "ic": "Information coefficient (Spearman correlation with target).",
    "vif": "Variance inflation factor; higher implies multicollinearity.",
    "psi_max": "Max population stability index across time buckets.",
    "mean_cv": "Mean coefficient of variation across time buckets.",
    "leakage_flag": "Flag if correlation/MI exceed leakage thresholds.",
}

METRIC_GLOSSARY = {
    "Feature": "Column name used in model training and diagnostics, after any name mapping.",
    "Gain": "Total LightGBM split gain attributable to the feature. Higher implies stronger impact on loss reduction.",
    "Split": "Number of times the feature is used in LightGBM splits. High values can indicate broad utility or redundancy.",
    "SHAP": "Mean absolute SHAP value, reflecting average contribution magnitude per prediction.",
    "Corr": "Correlation of the feature with the target (Pearson for numeric, Spearman otherwise).",
    "MI": "Mutual information with the target; captures non-linear dependency beyond correlation.",
    "IC": "Information coefficient, the Spearman rank correlation with the target.",
    "VIF": "Variance inflation factor; values above typical thresholds suggest multicollinearity.",
    "PSI Max": "Maximum population stability index across time buckets; higher values indicate drift.",
    "Mean CV": "Mean coefficient of variation across time buckets; higher implies less stability.",
    "Leakage": "Flag for potential leakage when correlation or mutual information exceed thresholds.",
}


def _format_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No features to display.</p>"

    df = pd.DataFrame(rows)
    display_cols = [
        "feature",
        "importance_gain",
        "importance_split",
        "shap_mean_abs",
        "corr",
        "mutual_info",
        "ic",
        "vif",
        "psi_max",
        "mean_cv",
        "leakage_flag",
    ]
    df = df[display_cols]
    if "importance_gain" in df.columns:
        df["importance_gain"] = pd.to_numeric(df["importance_gain"], errors="coerce")
        df = df.sort_values(by="importance_gain", ascending=False, na_position="last")
    max_gain = float(df["importance_gain"].max()) if "importance_gain" in df.columns else 0.0

    headers = [
        ("feature", "Feature", "text"),
        ("importance_gain", "Gain", "number"),
        ("importance_split", "Split", "number"),
        ("shap_mean_abs", "SHAP", "number"),
        ("corr", "Corr", "number"),
        ("mutual_info", "MI", "number"),
        ("ic", "IC", "number"),
        ("vif", "VIF", "number"),
        ("psi_max", "PSI Max", "number"),
        ("mean_cv", "Mean CV", "number"),
        ("leakage_flag", "Leakage", "text"),
    ]

    header_html = "".join(
        (
            f"<th data-key=\"{key}\" data-type=\"{dtype}\">"
            f"<span class=\"th-wrap\">{label}"
            f"<span class=\"tooltip\" data-tip=\"{METRIC_TOOLTIPS.get(key, '')}\" "
            f"title=\"{METRIC_TOOLTIPS.get(key, '')}\" tabindex=\"0\">?</span>"
            f"</span></th>"
        )
        for key, label, dtype in headers
    )

    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "nan", ""}:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(num) else num

    body_rows = []
    for _, row in df.iterrows():
        leakage_flag = bool(row.get("leakage_flag"))
        cells = []
        for key, _, dtype in headers:
            val = row.get(key)
            if key == "feature":
                display = str(val)
                data_val = display.lower()
                cells.append(f"<td data-value=\"{data_val}\">{display}</td>")
                continue
            if key == "leakage_flag":
                display = "YES" if leakage_flag else ""
                data_val = "1" if leakage_flag else "0"
                css = "flag" if leakage_flag else ""
                cells.append(f"<td class=\"{css}\" data-value=\"{data_val}\">{display}</td>")
                continue
            num_val = _safe_float(val)
            data_val = f"{num_val:.6f}" if num_val is not None else ""
            if key == "importance_gain":
                pct = 0.0 if not max_gain or num_val is None else max(0.0, (num_val / max_gain) * 100)
                display = _format_float(val)
                cells.append(
                    "<td data-value=\"{data}\">"
                    "<span class=\"bar-wrap\"><span class=\"bar\" style=\"width:{pct:.1f}%\"></span></span>"
                    "<span class=\"num\">{disp}</span>"
                    "</td>".format(data=data_val, pct=pct, disp=display)
                )
            else:
                display = _format_float(val)
                cells.append(f"<td data-value=\"{data_val}\"><span class=\"num\">{display}</span></td>")
        row_html = (
            f"<tr data-leakage=\"{'1' if leakage_flag else '0'}\">"
            + "".join(cells)
            + "</tr>"
        )
        body_rows.append(row_html)

    table_html = (
        "<table id=\"feature-table\" class=\"table\">"
        "<thead><tr>"
        + header_html
        + "</tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )

    return table_html


def generate_report(results: Dict[str, Any], output_path: str) -> None:
    feature_table = results.get("feature_table", [])
    notes = results.get("notes", [])
    leakage = results.get("leakage", {})
    config = results.get("config", {})
    metadata = results.get("metadata", {})
    name_mapping = metadata.get("name_mapping", {})
    constant_cols = [name_mapping.get(col, col) for col in metadata.get("constant_cols", [])]
    top_corr_pairs = results.get("top_corr_pairs", [])

    leakage_flags = [name_mapping.get(name, name) for name, info in leakage.items() if info.get("flag")]

    summary = {
        "Total features": len(feature_table),
        "Leakage flags": len(leakage_flags),
    }

    df = pd.DataFrame(feature_table) if feature_table else pd.DataFrame()
    top_gain = []
    top_psi = []
    plot_payload = {}
    if not df.empty:
        df_plot = df.copy()
        for col in ["importance_gain", "shap_mean_abs", "psi_max", "corr"]:
            if col in df_plot.columns:
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
        df_plot = df_plot.sort_values(by="importance_gain", ascending=False, na_position="last")
        plot_payload = {
            "features_full": df_plot["feature"].tolist(),
            "importance_gain_full": df_plot.get("importance_gain", pd.Series()).fillna(0).tolist(),
            "shap_mean_abs_full": df_plot.get("shap_mean_abs", pd.Series()).fillna(0).tolist(),
            "psi_max_full": df_plot.get("psi_max", pd.Series()).fillna(0).tolist(),
            "corr_full": df_plot.get("corr", pd.Series()).fillna(0).tolist(),
        }
        top_rows = df_plot.head(25)
        plot_payload.update({
            "features": top_rows["feature"].tolist(),
            "importance_gain": top_rows.get("importance_gain", pd.Series()).fillna(0).tolist(),
            "shap_mean_abs": top_rows.get("shap_mean_abs", pd.Series()).fillna(0).tolist(),
            "psi_max": top_rows.get("psi_max", pd.Series()).fillna(0).tolist(),
            "corr": top_rows.get("corr", pd.Series()).fillna(0).tolist(),
        })
        if "importance_gain" in df.columns:
            top_gain = (
                df.sort_values(by="importance_gain", ascending=False, na_position="last")
                .head(10)["feature"]
                .tolist()
            )
        if "psi_max" in df_plot.columns:
            top_psi = df_plot.head(10)["feature"].tolist()

    metrics_html = "".join(
        f"<div class=\"metric\"><div>{key}</div><strong>{val}</strong></div>" for key, val in summary.items()
    )

    notes_blocks = []
    if notes:
        notes_blocks.append("<ul>" + "".join(f"<li class=\"note\">{note}</li>" for note in notes) + "</ul>")
    if constant_cols:
        notes_blocks.append(
            "<p class=\"note\">Constant features (likely dropped by LightGBM): "
            + ", ".join(constant_cols)
            + "</p>"
        )
    notes_html = "".join(notes_blocks)

    table_html = _format_table(feature_table)

    flagged_html = ""
    if leakage_flags:
        flagged_html = (
            "<p class=\"flag\">Potential leakage features:</p>"
            + "<p>" + ", ".join(leakage_flags[:50]) + ("..." if len(leakage_flags) > 50 else "") + "</p>"
        )

    config_html = ""
    if config:
        config_items = []
        for key in ["data_path", "target_col", "date_col", "time_bucket", "sample_rows"]:
            if config.get(key) is not None:
                config_items.append(f"<span class=\"pill\">{key}: {config.get(key)}</span>")
        if config.get("exclude_columns"):
            config_items.append(
                f"<span class=\"pill\">excluded: {len(config.get('exclude_columns', []))}</span>"
            )
        for key in ["leakage_corr_threshold", "leakage_mi_quantile", "vif_threshold", "stability_psi_threshold"]:
            if config.get(key) is not None:
                config_items.append(f"<span class=\"pill\">{key}: {config.get(key)}</span>")
        config_html = "<div>" + "".join(config_items) + "</div>"

    top_gain_html = ""
    if top_gain:
        top_gain_html = "<p><span class=\"pill\">Top gain</span> " + ", ".join(top_gain) + "</p>"

    top_psi_html = ""
    if top_psi:
        top_psi_html = "<p><span class=\"pill danger\">Most unstable</span> " + ", ".join(top_psi) + "</p>"

    plotly_data = json.dumps(plot_payload)
    thresholds = {
        "stability_psi_threshold": config.get("stability_psi_threshold"),
        "leakage_corr_threshold": config.get("leakage_corr_threshold"),
        "vif_threshold": config.get("vif_threshold"),
    }
    plotly_thresholds = json.dumps(thresholds)

    glossary_html = ""
    if METRIC_GLOSSARY:
        items = []
        for key, desc in METRIC_GLOSSARY.items():
            items.append(f"<div class=\"glossary-item\"><h4>{key}</h4><p>{desc}</p></div>")
        glossary_html = "<div class=\"glossary\">" + "".join(items) + "</div>"

    corr_pairs_html = ""
    if top_corr_pairs:
        rows = []
        for pair in top_corr_pairs:
            feature_a = name_mapping.get(pair["feature_a"], pair["feature_a"])
            feature_b = name_mapping.get(pair["feature_b"], pair["feature_b"])
            corr_val = _format_float(pair.get("corr"))
            rows.append(f"<tr><td>{feature_a}</td><td>{feature_b}</td><td>{corr_val}</td></tr>")
        corr_pairs_html = (
            "<div class=\"table-wrap\">"
            "<table class=\"table\">"
            "<thead><tr><th>Feature A</th><th>Feature B</th><th>|Corr|</th></tr></thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Feature Analysis Report</title>
  <style>{CSS}</style>
  <script src=\"https://cdn.plot.ly/plotly-2.26.0.min.js\"></script>
</head>
<body>
  <div class=\"header\">
    <h1>Feature Analysis Report</h1>
    <p class=\"note\">LightGBM-focused feature diagnostics: importance, stability, leakage risk.</p>
    {config_html}
  </div>

  <div class=\"section\">
    <h2>Summary</h2>
    <div class=\"metrics\">{metrics_html}</div>
    {flagged_html}
    {top_gain_html}
    {top_psi_html}
  </div>

  <div class=\"section\">
    <h2>Interactive Charts</h2>
    <div class=\"chart-grid\">
      <div id=\"chart-gain\" class=\"chart\"></div>
      <div id=\"chart-shap\" class=\"chart\"></div>
      <div id=\"chart-stability\" class=\"chart\"></div>
      <div id=\"chart-corr\" class=\"chart\"></div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Feature Table</h2>
    <div class=\"controls\">
      <input id=\"table-search\" type=\"text\" placeholder=\"Search feature names\" />
      <label><input id=\"leakage-only\" type=\"checkbox\" /> Show leakage only</label>
      <span class=\"note\">Click column headers to sort.</span>
    </div>
    <div class=\"table-wrap\">{table_html}</div>
  </div>

  <div class=\"section\">
    <h2>Notes</h2>
    {notes_html if notes_html else '<p class="note">No runtime warnings.</p>'}
  </div>

  <div class=\"section\">
    <h2>Top Correlated Feature Pairs</h2>
    {corr_pairs_html if corr_pairs_html else '<p class="note">No correlated pairs to display.</p>'}
  </div>

  <div class=\"section\">
    <h2>Glossary</h2>
    {glossary_html}
  </div>
  <script>
    const plotData = {plotly_data};
    const thresholds = {plotly_thresholds};

    function renderCharts() {{
      if (!window.Plotly || !plotData || !plotData.features) {{
        return;
      }}
      const features = plotData.features || [];
      const featuresFull = plotData.features_full || features;
      const gainFull = plotData.importance_gain_full || plotData.importance_gain || [];
      const psiFull = plotData.psi_max_full || plotData.psi_max || [];
      const corrFull = plotData.corr_full || plotData.corr || [];
      Plotly.newPlot("chart-gain", [{{
        type: "bar",
        x: plotData.importance_gain || [],
        y: features,
        orientation: "h",
        marker: {{ color: "#0f4c5c" }},
        name: "Gain"
      }}], {{
        title: "Top Importance (Gain)",
        margin: {{ l: 140, r: 20, t: 40, b: 40 }},
        height: 360
      }});

      Plotly.newPlot("chart-shap", [{{
        type: "bar",
        x: plotData.shap_mean_abs || [],
        y: features,
        orientation: "h",
        marker: {{ color: "#e36414" }},
        name: "SHAP"
      }}], {{
        title: "Top SHAP Mean |Value|",
        margin: {{ l: 140, r: 20, t: 40, b: 40 }},
        height: 360
      }});

      const psiThreshold = thresholds.stability_psi_threshold;
      const psiLine = psiThreshold ? {{
        type: "line",
        x0: 0,
        x1: 1,
        xref: "paper",
        y0: psiThreshold,
        y1: psiThreshold,
        line: {{ color: "#b91c1c", width: 2, dash: "dot" }}
      }} : null;
      Plotly.newPlot("chart-stability", [{{
        type: "scatter",
        mode: "markers",
        x: gainFull,
        y: psiFull,
        text: featuresFull,
        marker: {{ color: "#1f2430" }}
      }}], {{
        title: "Stability (PSI Max) vs Gain",
        xaxis: {{ title: "Gain" }},
        yaxis: {{ title: "PSI Max" }},
        shapes: psiLine ? [psiLine] : [],
        margin: {{ l: 60, r: 20, t: 40, b: 50 }},
        height: 360
      }});

      const corrThreshold = thresholds.leakage_corr_threshold;
      const corrLine = corrThreshold ? [
        {{
          type: "line",
          x0: corrThreshold,
          x1: corrThreshold,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: {{ color: "#b91c1c", width: 2, dash: "dot" }}
        }},
        {{
          type: "line",
          x0: -corrThreshold,
          x1: -corrThreshold,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: {{ color: "#b91c1c", width: 2, dash: "dot" }}
        }}
      ] : [];
      Plotly.newPlot("chart-corr", [{{
        type: "scatter",
        mode: "markers",
        x: corrFull,
        y: gainFull,
        text: featuresFull,
        marker: {{ color: "#0f4c5c" }}
      }}], {{
        title: "Correlation vs Gain",
        xaxis: {{ title: "Correlation with Target" }},
        yaxis: {{ title: "Gain" }},
        shapes: corrLine,
        margin: {{ l: 60, r: 20, t: 40, b: 50 }},
        height: 360
      }});
    }}

    renderCharts();

    const table = document.getElementById("feature-table");
    const searchInput = document.getElementById("table-search");
    const leakageToggle = document.getElementById("leakage-only");
    if (table) {{

    function filterRows() {{
      const query = (searchInput.value || "").toLowerCase();
      const leakOnly = leakageToggle.checked;
      const rows = table.querySelectorAll("tbody tr");
      rows.forEach(row => {{
        const featureCell = row.querySelector("td");
        const text = featureCell ? featureCell.textContent.toLowerCase() : "";
        const leakage = row.getAttribute("data-leakage") === "1";
        const matches = text.includes(query) && (!leakOnly || leakage);
        row.style.display = matches ? "" : "none";
      }});
    }}

    if (searchInput) {{
      searchInput.addEventListener("input", filterRows);
    }}
    if (leakageToggle) {{
      leakageToggle.addEventListener("change", filterRows);
    }}

    function getCellValue(row, index) {{
      const cell = row.cells[index];
      if (!cell) return "";
      return cell.getAttribute("data-value") || cell.textContent.trim();
    }}

    function sortTable(index, type, asc) {{
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => {{
        const aVal = getCellValue(a, index);
        const bVal = getCellValue(b, index);
        if (type === "number") {{
          const aNum = parseFloat(aVal || "0");
          const bNum = parseFloat(bVal || "0");
          return asc ? aNum - bNum : bNum - aNum;
        }}
        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }});
      rows.forEach(row => tbody.appendChild(row));
    }}

    table.querySelectorAll("th").forEach((th, index) => {{
      th.addEventListener("click", () => {{
        const asc = th.getAttribute("data-asc") !== "true";
        table.querySelectorAll("th").forEach(h => h.removeAttribute("data-asc"));
        th.setAttribute("data-asc", asc ? "true" : "false");
        sortTable(index, th.getAttribute("data-type"), asc);
      }});
    }});
    }}
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

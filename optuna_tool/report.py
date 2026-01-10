from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import optuna

from .utils import optional_import


CSS = """
:root {
  --ink: #1f2937;
  --muted: #6b7280;
  --bg: #f7f7fb;
  --card: #ffffff;
  --accent: #0f766e;
  --accent-2: #f97316;
  --line: #e5e7eb;
}
body { font-family: "Segoe UI", Arial, sans-serif; margin: 24px; background: var(--bg); color: var(--ink); }
.header, .section { background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.section { margin-top: 24px; }
.metrics { display: flex; gap: 16px; flex-wrap: wrap; }
.metric { background: #f1f5f9; padding: 12px 16px; border-radius: 10px; min-width: 160px; }
.table { border-collapse: collapse; width: 100%; }
.table th, .table td { padding: 8px 10px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
.table th { text-align: left; background: #f1f5f9; position: sticky; top: 0; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #eef2f7; color: var(--muted); margin-right: 6px; }
.note { color: var(--muted); font-size: 13px; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
"""


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _render_params(params: Dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{key}</th><td>{_format_value(value)}</td></tr>"
        for key, value in sorted(params.items())
    )
    return f"<table class=\"table\">{rows}</table>"


def _render_trials(trials: List[Dict[str, Any]], limit: int = 25) -> str:
    rows = []
    for trial in trials[:limit]:
        params_preview = ", ".join(
            f"{k}={_format_value(v)}" for k, v in sorted(trial["params"].items())
        )
        rows.append(
            "<tr>"
            f"<td>{trial['number']}</td>"
            f"<td>{_format_value(trial['value'])}</td>"
            f"<td>{trial['state']}</td>"
            f"<td>{params_preview}</td>"
            "</tr>"
        )
    body = "".join(rows) if rows else "<tr><td colspan=\"4\">No trials available.</td></tr>"
    return (
        "<table class=\"table\">"
        "<thead><tr><th>Trial</th><th>Value</th><th>State</th><th>Params</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _plot_sections(study: optuna.Study) -> Tuple[str, List[str]]:
    plotly_mod, plotly_err = optional_import("plotly")
    if plotly_mod is None:
        return "", [f"Plotly not available: {plotly_err}"]

    from plotly.io import to_html
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )

    charts = [
        ("Optimization History", plot_optimization_history(study)),
        ("Parameter Importances", plot_param_importances(study)),
        ("Parallel Coordinate", plot_parallel_coordinate(study)),
        ("Slice Plot", plot_slice(study)),
    ]

    chart_blocks = []
    include_plotly = "cdn"
    for title, fig in charts:
        html = to_html(fig, include_plotlyjs=include_plotly, full_html=False)
        include_plotly = False
        chart_blocks.append(f"<div><h3>{title}</h3>{html}</div>")

    return "<div class=\"chart-grid\">" + "".join(chart_blocks) + "</div>", []


def generate_report(results: Dict[str, Any], study: Optional[optuna.Study], output_path: str) -> None:
    study_meta = results.get("study", {})
    dataset = results.get("dataset", {})
    metadata = results.get("metadata", {})
    trials = results.get("trials", [])

    summary = {
        "Best value": _format_value(study_meta.get("best_value")),
        "Metric": study_meta.get("metric", ""),
        "Direction": study_meta.get("direction", ""),
        "Trials": study_meta.get("n_trials", 0),
        "Rows": dataset.get("rows", 0),
        "Columns": dataset.get("columns", 0),
        "Target kind": metadata.get("target_kind", ""),
    }

    metrics_html = "".join(
        f"<div class=\"metric\"><div>{key}</div><strong>{value}</strong></div>"
        for key, value in summary.items()
    )

    best_params_html = _render_params(study_meta.get("best_params", {}))
    trials_html = _render_trials(trials)

    chart_html = ""
    notes = []
    if study is not None:
        chart_html, notes = _plot_sections(study)
    if notes:
        notes_html = "<p class=\"note\">" + " ".join(notes) + "</p>"
    else:
        notes_html = ""

    name_mapping = metadata.get("name_mapping", {})
    mapping_note = ""
    if name_mapping:
        mapping_note = "<p class=\"note\">Column names were sanitized for LightGBM. See results.json for mapping.</p>"

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Optuna Hyperparameter Report</title>
      <style>{CSS}</style>
    </head>
    <body>
      <div class="header">
        <h1>Optuna Hyperparameter Optimization</h1>
        <p class="note">Study: <span class="pill">{study_meta.get('study_name', '')}</span></p>
        <div class="metrics">{metrics_html}</div>
        {mapping_note}
      </div>

      <div class="section">
        <h2>Best Parameters</h2>
        {best_params_html}
      </div>

      <div class="section">
        <h2>Top Trials</h2>
        {trials_html}
      </div>

      <div class="section">
        <h2>Optimization Charts</h2>
        {chart_html}
        {notes_html}
      </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

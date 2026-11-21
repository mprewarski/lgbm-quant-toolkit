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
.chart { margin-top: 16px; }
.slice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
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


def _fig_html(fig, height: int, include_plotlyjs) -> str:
    from plotly.io import to_html

    fig.update_layout(height=height, autosize=True, title=None, margin=dict(l=60, r=30, t=30, b=50))
    return to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height=f"{height}px",
        config={"responsive": True, "displaylogo": False},
    )


def _plot_sections(study: optuna.Study) -> Tuple[str, List[str]]:
    plotly_mod, plotly_err = optional_import("plotly")
    if plotly_mod is None:
        return "", [f"Plotly not available: {plotly_err}"]

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return "", ["No completed trials to chart."]
    param_names = sorted({name for t in completed for name in t.params})

    blocks = []
    notes: List[str] = []
    include_js = "cdn"

    def add(title: str, make_fig, height: int, note: str = "") -> None:
        nonlocal include_js
        try:
            fig = make_fig()
        except Exception as exc:  # e.g. importances need >1 completed trial
            notes.append(f"{title} unavailable: {exc}")
            return
        note_html = f"<p class=\"note\">{note}</p>" if note else ""
        blocks.append(f"<div class=\"chart\"><h3>{title}</h3>{note_html}{_fig_html(fig, height, include_js)}</div>")
        include_js = False

    add("Optimization History", lambda: plot_optimization_history(study), 420)
    add(
        "Parameter Importances",
        lambda: plot_param_importances(study),
        max(320, 40 * len(param_names) + 120),
        "fANOVA importance of each parameter for the objective value.",
    )
    add("Parallel Coordinate", lambda: plot_parallel_coordinate(study), 520)

    slice_blocks = []
    for name in param_names:
        try:
            fig = plot_slice(study, params=[name])
        except Exception as exc:
            notes.append(f"Slice plot for {name} unavailable: {exc}")
            continue
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text=name)
        slice_blocks.append(f"<div>{_fig_html(fig, 320, include_js)}</div>")
        include_js = False
    if slice_blocks:
        blocks.append(
            "<div class=\"chart\"><h3>Slice Plots</h3>"
            "<p class=\"note\">Objective value of each trial against one parameter; darker points are later trials.</p>"
            f"<div class=\"slice-grid\">{''.join(slice_blocks)}</div></div>"
        )

    return "".join(blocks), notes


def generate_report(results: Dict[str, Any], study: Optional[optuna.Study], output_path: str) -> None:
    study_meta = results.get("study", {})
    dataset = results.get("dataset", {})
    metadata = results.get("metadata", {})
    trials = results.get("trials", [])

    best_value = study_meta.get("best_value")
    baseline_value = study_meta.get("baseline_value")
    summary = {
        "Best value": _format_value(best_value),
        "Metric": study_meta.get("metric", ""),
        "Direction": study_meta.get("direction", ""),
        "Trials": study_meta.get("n_trials", 0),
        "Rows": dataset.get("rows", 0),
        "Columns": dataset.get("columns", 0),
        "Target kind": metadata.get("target_kind", ""),
    }
    if baseline_value is not None:
        summary["Baseline (train mean)"] = _format_value(baseline_value)
        if best_value is not None and baseline_value:
            improvement = (baseline_value - best_value) / abs(baseline_value) * 100
            if study_meta.get("direction") == "maximize":
                improvement = -improvement
            summary["Improvement vs baseline"] = f"{improvement:.2f}%"

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

    html = f"""<!DOCTYPE html>
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
        <p class="note">For interactive exploration: <code>optuna-dashboard sqlite:///{results.get('config', {}).get('storage_path') or 'path/to/study.db'}</code></p>
      </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

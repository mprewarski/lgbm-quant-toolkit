import html
import os
from typing import Any, Dict, List, Optional


def _html_table(rows: List[Dict[str, Any]], columns: List[str], limit: Optional[int] = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows) if body_rows else "<tr><td colspan=\"{}\">No data</td></tr>".format(len(columns))
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _relative(path: Optional[str], base_dir: str) -> Optional[str]:
    if not path:
        return None
    return os.path.relpath(path, base_dir)


def generate_report(results: Dict[str, Any], output_dir: str) -> str:
    report_path = os.path.join(output_dir, "report.html")
    outputs = results.get("outputs", {})

    summary_plot = _relative(outputs.get("summary_plot"), output_dir)
    waterfall_plot = _relative(outputs.get("waterfall_plot"), output_dir)
    force_plot = _relative(outputs.get("force_plot"), output_dir)
    dependence_plots = [_relative(p, output_dir) for p in outputs.get("dependence_plots", [])]

    top_features = results.get("top_features", [])
    low_features = results.get("low_importance_features", [])
    interactions = results.get("interaction_summary", [])

    html_parts = [
        "<html><head><meta charset=\"utf-8\">",
        "<title>SHAP Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }",
        "h1, h2 { color: #222; }",
        "table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }",
        "th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }",
        "th { background: #f4f4f4; text-align: left; }",
        "img { max-width: 100%; height: auto; border: 1px solid #eee; }",
        "section { margin-bottom: 32px; }",
        "</style></head><body>",
        "<h1>SHAP Analysis Report</h1>",
    ]

    html_parts.append("<section><h2>Overview</h2>")
    dataset = results.get("dataset", {})
    html_parts.append(
        f"<p>Rows: {html.escape(str(dataset.get('rows', 'N/A')))} | Columns: {html.escape(str(dataset.get('columns', 'N/A')))}</p>"
    )
    splits = results.get("splits", {})
    html_parts.append(
        "<p>Train rows: {} | Val rows: {} | Test rows: {}</p>".format(
            html.escape(str(splits.get("train_rows", "N/A"))),
            html.escape(str(splits.get("val_rows", "N/A"))),
            html.escape(str(splits.get("test_rows", "N/A"))),
        )
    )
    html_parts.append("</section>")

    if summary_plot:
        html_parts.append("<section><h2>SHAP Summary</h2>")
        html_parts.append(f"<img src=\"{html.escape(summary_plot)}\" alt=\"SHAP summary plot\">")
        html_parts.append("</section>")

    if waterfall_plot:
        html_parts.append("<section><h2>Waterfall Plot</h2>")
        html_parts.append(f"<img src=\"{html.escape(waterfall_plot)}\" alt=\"SHAP waterfall plot\">")
        html_parts.append("</section>")
    elif force_plot:
        html_parts.append("<section><h2>Force Plot</h2>")
        html_parts.append(
            f"<iframe src=\"{html.escape(force_plot)}\" width=\"100%\" height=\"420\" style=\"border:1px solid #eee;\"></iframe>"
        )
        html_parts.append("</section>")

    html_parts.append("<section><h2>Top Features</h2>")
    if top_features:
        html_parts.append(
            _html_table(top_features, ["rank", "feature", "feature_original", "mean_abs_shap", "mean_shap"])
        )
    else:
        html_parts.append("<p>No feature importance results.</p>")
    html_parts.append("</section>")

    html_parts.append("<section><h2>Low Importance Candidates</h2>")
    if low_features:
        html_parts.append(
            _html_table(low_features, ["rank", "feature", "feature_original", "mean_abs_shap", "mean_shap"])
        )
    else:
        html_parts.append("<p>No low-importance features identified.</p>")
    html_parts.append("</section>")

    if interactions:
        html_parts.append("<section><h2>Top Feature Interactions</h2>")
        html_parts.append(
            _html_table(interactions, ["feature_1", "feature_2", "mean_abs_interaction"], limit=30)
        )
        html_parts.append("</section>")

    if dependence_plots:
        html_parts.append("<section><h2>Dependence Plots</h2>")
        for plot in dependence_plots:
            if plot:
                html_parts.append(f"<img src=\"{html.escape(plot)}\" alt=\"Dependence plot\">")
        html_parts.append("</section>")

    html_parts.append("</body></html>")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_parts))

    return report_path

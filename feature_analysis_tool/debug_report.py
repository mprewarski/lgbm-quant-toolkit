from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from .config import AnalysisConfig
from .utils import ensure_dir


CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #1f2430; }
.header { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.section { margin-top: 24px; background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(18, 23, 40, 0.08); }
.note { color: #4b5563; font-size: 13px; }
.chart { width: 100%; height: 420px; }
"""


def generate_debug_report(config: AnalysisConfig) -> Optional[str]:
    if not config.debug_report:
        return None
    if not config.debug_feature or not config.date_col:
        raise ValueError("debug_feature and date_col are required for debug_report")

    ensure_dir(config.output_dir)

    df = pd.read_csv(config.data_path)
    if config.debug_sample_rows and len(df) > config.debug_sample_rows:
        df = df.sample(n=config.debug_sample_rows, random_state=config.random_state)

    if config.debug_feature not in df.columns:
        raise ValueError(f"debug_feature '{config.debug_feature}' not found in data")
    if config.date_col not in df.columns:
        raise ValueError(f"date_col '{config.date_col}' not found in data")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df = df.dropna(subset=[config.date_col])

    if config.debug_symbol_col and config.debug_symbol:
        if config.debug_symbol_col not in df.columns:
            raise ValueError(f"debug_symbol_col '{config.debug_symbol_col}' not found in data")
        df = df[df[config.debug_symbol_col] == config.debug_symbol]

    df = df.sort_values(by=config.date_col)

    x = df[config.date_col].astype(str).tolist()
    y = df[config.debug_feature].tolist()

    payload = {
        "x": x,
        "y": y,
        "feature": config.debug_feature,
        "symbol": config.debug_symbol,
        "symbol_col": config.debug_symbol_col,
    }

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Debug Feature Report</title>
  <style>{CSS}</style>
  <script src=\"https://cdn.plot.ly/plotly-2.26.0.min.js\"></script>
</head>
<body>
  <div class=\"header\">
    <h1>Debug Feature Report</h1>
    <p class=\"note\">Feature: {config.debug_feature}</p>
    <p class=\"note\">Symbol filter: {config.debug_symbol or 'None'} ({config.debug_symbol_col or 'n/a'})</p>
  </div>

  <div class=\"section\">
    <h2>Feature Over Time</h2>
    <div id=\"chart\" class=\"chart\"></div>
  </div>

  <script>
    const payload = {json.dumps(payload)};
    if (window.Plotly) {{
      Plotly.newPlot("chart", [{{ 
        x: payload.x,
        y: payload.y,
        type: "scatter",
        mode: "lines+markers",
        line: {{ color: "#0f4c5c" }}
      }}], {{
        title: `${{payload.feature}} over time`,
        xaxis: {{ title: "Date" }},
        yaxis: {{ title: payload.feature }},
        margin: {{ l: 60, r: 20, t: 50, b: 50 }}
      }});
    }}
  </script>
</body>
</html>
"""

    output_path = os.path.join(config.output_dir, config.debug_output)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return output_path

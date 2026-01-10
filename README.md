# LightGBM Feature Analysis Tool

A modular feature analysis pipeline for LightGBM models. It evaluates feature importance, stability, correlation, SHAP contributions, WOE/IV (for binary targets), and flags potential leakage risks. The output is an interactive HTML report plus optional saved project artifacts.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis using a config file:

```bash
python -m feature_analysis_tool --config feature_analysis_config.json --save-project
```

The report is generated at `feature_analysis_output/report.html`.

## Configuration

The tool reads a JSON or YAML config. CLI flags override config values.

Example `feature_analysis_config.json`:

```json
{
  "data_path": "sp500-y17-65.csv",
  "target_col": "TARGET",
  "date_col": "Date",
  "output_dir": "feature_analysis_output",
  "sample_rows": 50000,
  "time_bucket": "M",
  "random_state": 42,
  "exclude_columns": ["P123 ID", "Ticker"],
  "leakage_corr_threshold": 0.9,
  "leakage_mi_quantile": 0.99,
  "vif_threshold": 10.0,
  "stability_psi_threshold": 0.2,
  "shap_sample_rows": 2000,
  "log_level": "INFO",
  "log_path": "feature_analysis_output/run.log",
  "lgbm_params": {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_leaves": 31,
    "min_child_samples": 20
  },
  "top_corr_pairs": 25,
  "categorical_columns": ["Sub-Sector Code"],
  "debug_report": false,
  "debug_feature": "macro - VIX index (Series)",
  "debug_symbol_col": "Ticker",
  "debug_symbol": "AAPL",
  "debug_output": "debug_report.html",
  "debug_sample_rows": 200000
}
```

### Config options

- `data_path`: CSV dataset path.
- `target_col`: Target column name.
- `date_col`: Optional date column for time-based stability.
- `model_path`: Optional LightGBM model file path.
- `output_dir`: Output directory for reports and saved projects.
- `sample_rows`: Downsample to speed up analysis on large datasets.
- `time_bucket`: Pandas period alias for stability buckets (e.g. `M`, `Q`).
- `random_state`: Random seed for model training and sampling.
- `exclude_columns`: List of columns to exclude from analysis.
- `leakage_corr_threshold`: Correlation threshold for leakage flagging.
- `leakage_mi_quantile`: Mutual information quantile threshold for leakage flagging.
- `vif_threshold`: VIF threshold for multicollinearity review.
- `stability_psi_threshold`: PSI threshold for stability risk.
- `shap_sample_rows`: Rows sampled for SHAP calculations.
- `log_level`: Logging level (e.g. `INFO`, `DEBUG`).
- `log_path`: Path for the run log file.
- `lgbm_params`: Dictionary of LightGBM training parameters (overrides defaults).
- `top_corr_pairs`: Number of top absolute-correlation feature pairs to list.
- `categorical_columns`: Columns to force as categorical features for LightGBM.
- `debug_report`: Enable the debug report generation (only runs when `--debug` is passed).
- `debug_feature`: Feature/column to plot over time.
- `debug_symbol_col`: Column used to filter a single entity (e.g., `Ticker`).
- `debug_symbol`: Value to filter `debug_symbol_col`.
- `debug_output`: Debug report filename (written to `output_dir`).
- `debug_sample_rows`: Optional sampling cap for debug report data.

## CLI usage

```bash
python -m feature_analysis_tool --data sp500-y17-65.csv --target TARGET --date Date
```

You can also use a config file and override fields:

```bash
python -m feature_analysis_tool --config feature_analysis_config.json --sample-rows 20000
```

To reload a saved project and regenerate the report:

```bash
python -m feature_analysis_tool --config feature_analysis_config.json --load-project
```

To generate the debug report:

```bash
python -m feature_analysis_tool --config feature_analysis_config.json --save-project --debug
```

## Output

- `report.html`: Interactive report with sortable/filterable table and Plotly charts.
- `results.json`: JSON summary of all metrics.
- `model.txt`: Saved LightGBM model (when trained and `--save-project` is used).

## Report features

- Table search and leakage-only filtering.
- Click column headers to sort.
- Plotly charts for importance, SHAP, stability, and correlation with threshold overlays.
- Summary pills showing current config and threshold values.
- `run.log` captures data loading and model training status.
- Feature names are sanitized automatically for LightGBM (original names remain in the report).
- Top correlated feature pairs table is appended to the report for pruning review.
- A separate debug report can be generated to inspect a single feature over time.
- Constant features are flagged; LightGBM may drop them during training.

## Notes

- SHAP, statsmodels, scipy, and sklearn are optional but recommended for full metrics.
- If Plotly JS fails to load (offline environments), charts will be skipped but the report still renders.

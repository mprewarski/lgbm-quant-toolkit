# LightGBM Quant Toolkit — project notes for Claude

## What this project is
Tools for building LightGBM stock-return models on **Portfolio123 (P123)**. Datasets are
built and downloaded from P123's AI Factor, analyzed and tuned locally with these tools,
and the findings go back into P123 (feature set changes, hyperparameters). Use the
`portfolio123` skill for anything about P123 formulas, factors, ranking systems, AI Factor
or the API.

`README.md` is the user manual for the tools. Keep it in sync when tool behavior or
config options change.

## Layout
```
feature_analysis_tool/  importance, stability (PSI), correlation/VIF, leakage flags, SHAP
optuna_tool/            LightGBM hyperparameter search with Optuna
shap_tool/              SHAP values, dependence/interaction plots, low-importance features
feature_stats_tool/     per-feature distribution stats, histograms, stats over time, data-quality flags
p123_client.py          shared P123 API client (credentials from .env)
data/                   P123 dataset CSVs (gitignored, 100-250 MB each)
models/<model>/         one folder per model: *_config.json, train prompt XML, outputs/
docs/                   tool design specs (docs/design/), notes
scripts/                one-off helpers
tests/                  pytest tests
```
Run tools from the repo root, pointing at a model's config:
`python -m optuna_tool --config models/sp500/optuna_config.json`.
All paths in configs are relative to the repo root.

## Models
| Folder | Universe | Target | Dataset(s) |
|---|---|---|---|
| `models/sp500` (main model) | S&P 500 (IVV), benchmark SPY | 9MRel = `FutureRel%Chg_D(190, #Bench)`, weekly, 2005–2025 | `data/SP500-9MRel-ZsDs-Y-19-64-B.csv` (Optuna, SHAP); `data/sp500-y-31-62.csv` (feature analysis); `data/sp500_Y-34-61.csv` (feature stats, newest) |
| `models/us_smcap_3mtot` | US Small Cap | 3MTot = `Future%Chg_D(65)`, weekly | `data/US_SmCap_3MTot_RankData_S-56-70.csv` |
| `models/canada` | Canadian stocks | not set yet (planning) | none yet |

Each model's `train_prompt.xml` / `model_prompt.xml` has the full P123 AI Factor setup
(normalization, validation, feature formulas). To add a model: create
`models/<name>/`, copy configs from an existing model, change `data_path`, `output_dir`,
`log_path`, `storage_path`, `study_name` and `gap_days`.

Dataset CSV columns: `Date`, `P123 ID`, `Ticker`, features..., `Sub-Sector Code`
(categorical), `TARGET`. Always exclude `P123 ID` and `Ticker` from features.

## Rules that matter
- **No random splits on this data.** Rows are stock × week, and targets look months ahead,
  so a shuffled split leaks future returns into training. Split by date with a gap at least
  as long as the target horizon (`gap_days`: ~280 for 9-month targets, ~100 for 3-month).
  The Optuna tool's `split_method` defaults to `"time"`; `"random"` is only for
  non-time-series data.
- Use a new Optuna `study_name` whenever the split, data or target changes. Studies with
  `load_if_exists` otherwise mix old and new trials. (The old `lgbm_optuna` study in
  `models/sp500/outputs/optuna/study.db` was scored on a random split — don't reuse it.)
- Features in P123 datasets are already normalized (Z-score or rank by date). Without a
  P123 data license, downloads fail for features whose step 1 and step 2 normalization
  are both Skip (see `docs/p123_dataset_download_note.txt`).
- Never print, log or commit the API key. Credentials live in `.env` (gitignored);
  `.env.example` is the template.
- P123 API calls use the account's API credits; avoid unnecessary or repeated calls.
- `Sub-Sector Code` arrives Z-score normalized in the SP500 datasets (values like -0.74, 3.0), not as raw codes,
  because Skip/Skip features can't be downloaded without a data license. Treat it as categorical anyway.
- A new dataset: run `feature_stats_tool` on it first to check ranges, missing values and clipping.

## Environment
- Python 3.11 venv in `venv/` (macOS). `pip install -r requirements.txt`.
- Check API credentials: `python -m p123_client`.
- Tests: `python -m pytest tests`.

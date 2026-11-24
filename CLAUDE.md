# LightGBM Quant Toolkit — project notes for Claude

## What this project is
Tools for building LightGBM stock-return models on **Portfolio123 (P123)**. Datasets are
built and downloaded from P123's AI Factor, analyzed and tuned locally with these tools,
and the findings go back into P123 (feature set changes, hyperparameters). Use the
`portfolio123` skill for anything about P123 formulas, factors, ranking systems, AI Factor
or the API.

I would like to add capability to mimic the Portfolio123 model training, except greatly improve it
Portfolio123


`README.md` is the user manual for the tools. Keep it in sync when tool behavior or
config options change.

## Layout
```
feature_analysis_tool/  importance, stability (PSI), correlation/VIF, leakage flags, SHAP
optuna_tool/            LightGBM hyperparameter search with Optuna
shap_tool/              SHAP values, dependence/interaction plots, low-importance features
feature_stats_tool/     per-feature distribution stats, histograms, stats over time, data-quality flags
train_tool/             P123 AI Factor-style training: rolling CV folds, multi-seed runs, RMSE/Spearman/quantile stats;
                        train_tool.feature_test: remove/add features one at a time, paired-by-seed comparison
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
- `train_tool` mimics P123 validation (folds, NAs to the middle, rows >30% NA dropped). Portfolio
  stats (Avg%, SR, Turn%) need weekly returns: `data/sp500_Y-34-61_fut5.csv` holds P123's
  `Future%Chg(5)` (MinMax, Dataset scope, unclipped) and is joined via `returns_path`; map it back to
  percent with `python -m train_tool.calibrate`. Download returns separately from training data,
  with outlier clipping OFF: clipping (Z-score or MinMax) cuts big weekly moves; MinMax + clip pins
  ~7.5% of values at exactly 0 and 1 (`calibrate` warns). `data/sp500_fut_5.csv` (Future%Chg_D(5), MinMax,
  unclipped; zero 0.48568, per_pct 0.088849) is the current returns file.
- To compare with P123, the training dataset must use the same normalization settings as the P123
  run (Z-score, trim 7.5%, outlier clip, limit 2.5 for the SP500 Y-34-61 runs). Check: the holdout
  TARGET should have P123's mean/std from its Lift download (`data/lift-data.csv`: -0.190 / 1.392).
  `data/sp500_Y-34-61_zscore.csv` (trim 7.5%, downloaded unclipped) + `"clip": 2.5` in
  `train_config_zscore.json` gives holdout -0.188 / 1.390: matches. (`data/sp500_Y-34-61.csv` is capped at +-3.)
- Status of the P123 comparison (lvs=20, 3 runs each, Nov 20): RMSE, High/Low Avg%, SR, SD and turnover
  are close; Spearman is not (local ~0.00 vs P123 0.036-0.053). Not the cause: return calibration,
  trim/clip, per-fold target normalization (tested). Local predictions sit ~0.08 below P123's at every
  percentile of the lift chart. Results: `models/sp500/outputs/train_zscore_p123check/`.
- P123 Z-score = (x - mean) / std with mean/std from the data after trimming 7.5% from EACH tail,
  then clipped at the outlier limit. Rebuilding the TARGET this way from the MinMax file
  (`sp500_Y-34-61_fut5.csv`, linear in the raw return) gives holdout -0.184 / 1.388 vs P123 -0.190 / 1.392.
- P123's `_D` functions count weekdays (holidays included): `FutureRel%Chg_D(190)` leaves exactly
  38 weeks without a target.
- P123 portfolio conventions (verified exactly against its Portfolio download `data/perf.csv`):
  price returns (no dividends) over a calendar week, first trading day after Saturday to the first
  after the next Saturday. `Future%Chg_D(5)` = close as of Monday to close as of next Monday (holiday ->
  previous close); `Future%Chg(5)` = next trading day + 5 bars. Each is a day off in ~20% of weeks;
  slippage 0.1% per buy and sell; Avg% = (1 + mean weekly)^52.18 - 1; SR = mean x 52.18 / SD.
- Feature tests (`python -m train_tool.feature_test --config models/sp500/feature_test_remove.json`):
  every feature set uses the same seeds; judge a feature by its paired difference vs baseline and its t
  (shown from 3 runs). Seed noise is large (P123 H-L varies ~4 points between identical runs), so don't
  act on a single 2-3 run result. Finished runs are cached by a fingerprint and reused.
- A new dataset: run `feature_stats_tool` on it first to check ranges, missing values and clipping.

## Environment
- Python 3.11 venv in `venv/` (macOS). `pip install -r requirements.txt`.
- Check API credentials: `python -m p123_client`.
- Tests: `python -m pytest tests`.

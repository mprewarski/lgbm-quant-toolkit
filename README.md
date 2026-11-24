# LightGBM Quant Toolkit

This repo includes modular tools for LightGBM workflows:

- Feature Analysis Tool: evaluates feature importance, stability, correlation, SHAP contributions, WOE/IV (for binary targets), and flags potential leakage risks.
- Optuna Hyperparameter Tool: tunes LightGBM hyperparameters with Optuna, evaluates validation performance, and generates an HTML report with study visuals.
- SHAP Tool: computes SHAP values, ranks features, and produces SHAP visualizations and interaction summaries.
- Feature Stats Tool: per-feature distribution statistics (min, max, std, quantiles, skew, missing values, clipping, drift over time) with histograms and an over-time view for every feature.
- Train Tool: trains LightGBM the way P123 AI Factor validation does (rolling time series CV with a gap, several runs per model with different seeds) and reports RMSE, Spearman and High/Low/H-L quantile stats per run and averaged over runs.

Datasets come from Portfolio123 (P123) AI Factor downloads. Each model (stock universe +
target + feature set) has its own folder under `models/`.

## Project layout

```
data/                    P123 dataset CSVs (gitignored)
models/<model>/          configs, P123 training prompt XML, outputs/ (gitignored)
  sp500/                 main model: S&P 500, 9-month return relative to SPY
  us_smcap_3mtot/        US small cap, 3-month total return
  canada/                Canadian model (planning)
feature_analysis_tool/   optuna_tool/   shap_tool/   feature_stats_tool/   train_tool/ (incl. feature_test)
p123_client.py           shared Portfolio123 API client
docs/                    tool design specs and notes
scripts/                 one-off helpers (e.g. print_column_values.py)
tests/                   pytest tests
```

Run all commands from the repo root. Paths inside config files are relative to the repo root.

### Adding a new model

1. Put the P123 dataset CSV in `data/`.
2. Create `models/<name>/` and copy the config files from an existing model.
3. Change `data_path`, `output_dir`, `log_path`, and in the Optuna config `storage_path`,
   `study_name` and `gap_days` (at least the target horizon in calendar days).

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

   Optional, for the Portfolio123 API: copy `.env.example` to `.env`, fill in
   `P123_API_ID` and `P123_API_KEY`, then check them with `python -m p123_client`
   (see [Portfolio123 API](#portfolio123-api)).

3. Run the feature analysis using a config file:

```bash
python -m feature_analysis_tool --config models/sp500/feature_analysis_config.json --save-project
```

The report is generated at `models/sp500/outputs/feature_analysis/report.html`.

4. Run the Optuna hyperparameter search using a config file:

```bash
python -m optuna_tool --config models/sp500/optuna_config.json
```

The report is generated at `models/sp500/outputs/optuna/report.html`.

5. Run the SHAP tool using a config file:

```bash
python -m shap_tool --config models/sp500/shap_config.json
```

Outputs are generated in `models/sp500/outputs/shap/`, including `report.html`.

6. Run the feature statistics tool on a dataset:

```bash
python -m feature_stats_tool --config models/sp500/feature_stats_config.json
```

Or point it at any CSV without a config:

```bash
python -m feature_stats_tool --data data/sp500_Y-34-61.csv --output models/sp500/outputs/feature_stats \
  --exclude "P123 ID,Ticker" --categorical "Sub-Sector Code"
```

The report is generated at `models/sp500/outputs/feature_stats/report.html`, with
`stats.csv` (one row per feature) and `results.json` beside it. It uses every row by
default; a 500k-row, 60-feature dataset takes about 30 seconds and 1.6 GB of memory.

7. Train and validate models the P123 way:

```bash
python -m train_tool --config models/sp500/train_config.json
```

See [Train tool](#train-tool) below.

## Train tool

`train_tool` replicates P123 AI Factor validation locally on a downloaded dataset, so a
model's results can be checked against P123's Results tab and, later, feature sets can be
compared without spending P123 resources.

What it does, per model in the config:

1. Loads the dataset CSV and keeps the dates inside `dataset_start`..`dataset_end` (P123's
   Validation > Method > Dataset Period). Drops rows without a target and rows with more
   than `max_na_pct` % missing features, then fills the remaining NAs with 0 (P123: "NAs go
   to the middle"; the data is already normalized).
2. Builds the folds like P123's Rolling Time Series CV: holdout length = (dataset period -
   training period - gap) / folds; each fold trains on `training_years`, skips `gap_weeks`,
   then predicts its holdout. The holdouts are back to back and end at the dataset end.
   `folds.csv` lists the dates and row counts so you can compare them with P123's chart.
3. Trains `runs` LightGBM models per fold with different seeds (like repeated P123 models
   tagged `#random`) and saves each run's holdout predictions.
4. Scores every run, averages the runs, and (with more than one run) scores an ensemble of
   the runs' averaged predictions:
   - **RMSE, Spearman, Pearson, R2**: pooled over all holdout rows (P123 Lift tab).
   - **IC/date**: mean of the per-date Spearman correlations.
   - **Portfolio stats** (P123 Compare All / Return tabs; needs weekly returns, below): on
     each date the predictions are split into `quantiles` buckets (High = top, Low =
     bottom). Each bucket is an equal-weighted portfolio rebalanced every date and held one
     week. `Avg%` = mean weekly return compounded over a year (52.18 weeks), `Cmp%` =
     compounded annual return, `SD` = annualized standard deviation, `SR` = (mean weekly x
     52.18) / SD, `Turn%` = annualized one-way turnover. Slippage (`slippage_pct`, default
     0.1% like P123) is charged on every buy and sell. `H-L Avg%` = the mean weekly High
     minus Low return, annualized the same way. These conventions reproduce P123's Compare
     All numbers exactly from its Portfolio download (see below).
   - **TgtHigh, TgtLow, TgtH-L**: mean TARGET per bucket, in TARGET units. Available
     without weekly returns.

### Weekly returns for portfolio stats

P123 computes its portfolio returns from weekly price returns, which aren't in a dataset by
default. Add the feature `Future%Chg_D(5)` (the price change over the next 5 weekdays) to a
P123 dataset with Dataset-scope MinMax normalization, and set `returns_col` to it. The tool
never uses it as a feature.

P123's portfolio week runs from the close of the first trading day after one Saturday to
the close of the first trading day after the next Saturday (checked against the Benchmark
column of a P123 Portfolio download: identical every week). `Future%Chg_D(5)` runs from the
close as of Monday to the close as of the next Monday, a holiday taking the previous
trading day's close (`_D` functions count weekdays). `Future%Chg(5)` counts 5 trading days
from the first trading day after the date. Both match P123's week except in holiday weeks,
where each is one day off (about 20% of weeks). That difference is random, not a bias, and
both columns work equally well.
Download the returns in their own dataset rather than adding them to
a Z-score training dataset: Z-score outlier clipping would cut big weekly moves.

P123 normalizes it like any feature. With Dataset scope (MinMax or Z-score over the whole
dataset) that is a straight-line rescale, so it can be turned back into percent:
`return % = (value - returns_zero) / returns_per_pct`. Find the two numbers with:

```bash
python -m train_tool.calibrate --config models/sp500/train_config.json --uids 4737 5881 --write
```

It fits the column against real prices of a few long-lived stocks (P123 IDs from the
dataset; 4737 = IBM, 5881 = MSFT), using 1 API call per stock, cached as
`data/calib_prices_<id>.csv`. R2 should be about 0.9999. The numbers change with each
download, since the min/max depend on the dataset's period.

`returns_path` reads the returns from a different CSV (joined on Date and P123 ID). One
download with `Future%Chg(5)` can then supply returns for every other dataset of the same
universe, which saves adding the column to each feature-test dataset.

### Comparing with P123

To match P123, train on the same normalization P123 used. Features don't matter (LightGBM
only uses their order), but the TARGET does: a Z-score target is clipped at the outlier
limit, while a MinMax target keeps its outliers, and LightGBM's squared-error loss chases
them. `models/sp500/train_config_zscore.json` trains on the Z-score dataset and takes
returns from the MinMax download.

The training dataset must be downloaded with exactly the normalization the P123 run used
(scaling, trim %, outlier handling = clip, outlier limit). Different trim or clipping gives
a different TARGET, and so a different model. A check: P123's Lift tab download
(lift-data.csv) gives the holdout TARGET's mean and spread, which the local holdout
TARGET should reproduce.

Change the number of quantiles without retraining:

```bash
python -m train_tool --config models/sp500/train_config.json --report-only --quantiles 20
```

Other CLI overrides: `--data`, `--output`, `--runs`, `--folds`, `--training-years`,
`--gap-weeks`, `--quantiles`.

### HTML report

Every run also writes `report.html`, an interactive report (open it in a browser; Plotly
loads from a CDN, so it needs internet). It covers 10, 20 and 50 quantiles in the same file
(`report_quantiles`), and controls at the top switch the model, the run (each run, the
mean of runs, or the ensemble) and the quantile count:

- Headline numbers: H-L, High, Low and universe Avg%, Spearman, RMSE.
- Quantile counts compared: one table with 10, 20 and 50 side by side.
- Annualized return by quantile (P123 Return tab), with the universe as a dashed line.
- Growth of 100 for High, Low and the universe, with the folds shaded.
- High minus Low return in each 26-week period.
- Lift chart: mean prediction and mean actual TARGET per prediction percentile.
- Spearman per date over time, with a 13-week average and the fold starts.
- All runs (P123 Compare All columns), each fold scored on its own, feature importance
  (share of gain), the fold dates and the hyperparameters.

Without weekly returns the portfolio charts are hidden and the quantile charts show mean
TARGET instead. `--report-only` rebuilds the HTML report from the saved predictions too.
It follows the system light/dark setting.

Outputs in `output_dir`: `report.html`, `report.txt` (the tables printed at the end), `runs.csv` (one row
per model and run), `summary.csv` (mean and `_std` over runs per model), `ensemble.csv`,
`quantiles_<N>.csv` (per bucket: mean TARGET and portfolio Avg%, Cmp%, SD, Turn%, per run
and averaged), `folds.csv`, `feature_importance.csv` (gain and splits per model, run and
fold), `predictions/` (holdout predictions per run, used by `--report-only`),
`run_config.json`, `run.log`.

### Train config options

- `data_path`, `output_dir`, `target_col` (`TARGET`), `date_col` (`Date`), `id_col` (`P123 ID`).
- `exclude_columns`: non-feature columns (default `P123 ID`, `Ticker`).
- `features`: explicit feature list (default: every other column). `drop_features`: features
  to remove, for add/remove-one feature tests.
- `categorical_columns`: features passed to LightGBM as categorical. Default none, since the
  P123 datasets deliver `Sub-Sector Code` normalized.
- `dataset_start`, `dataset_end`: P123's Dataset Period. Default: the CSV's first/last date.
- `validation`: `method` (`rolling` = P123 Rolling Time Series CV, `expanding` = Time Series
  CV, `holdout` = Basic Holdout), `folds` (4), `training_years` (16), `gap_weeks` (20).
- `preprocessing`: `na_fill`: P123 puts NAs in the middle, so `"zero"` for Z-score data,
  `0.5` for MinMax data; also `"median"` (the feature's median on that date) or `"none"`
  (LightGBM handles NAs). `max_na_pct` (30): rows with more NA features are dropped.
  `clip`: clip features and TARGET at +-this value, like P123's Z-score outlier clipping
  (its outlier limit, e.g. 2.5), for a dataset downloaded without clipping. P123's Z-score
  uses the mean and std after trimming the trim % from each tail, then clips; downloading
  with P123's trim and no clipping, then setting `clip`, reproduces P123's values.
- `returns_col`, `returns_path`, `returns_zero`, `returns_per_pct`, `slippage_pct`: weekly
  returns for portfolio stats (see above). Without `returns_col` only TARGET-based quantile
  stats are reported.
- `runs` (5), `base_seed` (1; run k uses `base_seed + k - 1`), or `seeds` as an explicit list
  (sets the number of runs). `n_jobs` (-1 = all cores).
- `quantiles` (10): quantile count for `report.txt` and the CSVs.
- `report_quantiles` (`[10, 20, 50]`): quantile counts in `report.html`. `html_report`
  (true): set false to skip it.
- `models`: list of `{"name": ..., "params": {...}}`. `params` is P123's Hyperparams JSON
  (LightGBM scikit-learn names such as `n_estimators`, `num_leaves`, `learning_rate`,
  `min_child_samples`, `subsample`, `colsample_bytree`). A seed in `params` is replaced by
  the run's seed.
- Keys starting with `_` (e.g. `_comment`) are ignored.

## Feature tests

`train_tool.feature_test` compares feature sets with the train tool's validation: remove
current features one at a time, or add new ones one at a time, training each feature set
several times.

```bash
python -m train_tool.feature_test --config models/sp500/feature_test_remove.json
python -m train_tool.feature_test --config models/sp500/feature_test_add.json --runs 5
python -m train_tool.feature_test --config models/sp500/feature_test_remove.json --report-only
```

- **Remove test** (`"mode": "remove"`): the baseline is every feature of the base config;
  each variant drops one candidate. `"candidates": "all"` tests every feature.
- **Add test** (`"mode": "add"`): point `data_path` at a download with the current
  features plus the new ones. The baseline is that dataset without the candidates; each
  variant adds one.
- `include_all` (true) also tests removing / adding all candidates together.

How the comparison works: every feature set is trained with the same seeds, folds and
rows, and each variant is compared with the baseline run that used the same seed. The
report shows the average change (Δ), its standard error and t = Δ / se. Pairing by seed
removes most of the run-to-run noise (P123 runs of one model can differ by 4 points of
H-L), so smaller effects show up. |t| above about 2 is unlikely to be noise; with 3 runs a
test is still rough, so re-run promising features with more runs before acting.

For a remove test, a better result without a feature means the feature may be hurting.
"Better" is higher for returns, SR, Spearman and IC, and lower for RMSE and Low Avg%.

Finished runs are saved in `predictions/` with a fingerprint of the features, model,
seed, data and validation settings. Running a test again only trains what's missing, so
you can stop and resume, add candidates, or raise `--runs`; changing the model or data
retrains automatically.

Outputs in `output_dir`: `report.html` (a ranked chart of each feature's effect with error
bars and per-run dots, and a table; metric and quantile selectors), `report.txt`,
`summary.csv` (per feature set: mean, `_delta`, `_se`, `_t` for every metric),
`runs.csv` (every run), `variants.json`, `run.log`.

Each run trains one model per fold, so a test costs (candidates + 2) x runs trainings:
5 candidates with 3 runs is 21, all 61 features with 3 runs is 189.

### Feature test config options

- `base_config`: the train config with the model, data and validation settings.
- `output_dir`, `name` (title in the report; default the folder name).
- `mode`: `remove` or `add`. `candidates`: feature names (exact column names), or `"all"`
  in remove mode.
- `data_path`: use a different dataset than the base config (for add tests).
- `model`: model name from the base config (default the first).
- `runs` (3), `base_seed` (1) or `seeds` (explicit list).
- `quantiles` (`[10, 20, 50]`): quantile counts scored and shown in the report.
- `include_all` (true), `rank_by` (`hl_avg`; also `high_avg`, `low_avg`, `high_sr`,
  `spearman`, `ic_mean`, `rmse`, `tgt_hl`): the metric `report.txt` is sorted by and the
  report opens on.

## Configuration

The tool reads a JSON or YAML config. CLI flags override config values.

Example `models/sp500/feature_analysis_config.json`:

```json
{
  "data_path": "data/sp500-y-31-62.csv",
  "target_col": "TARGET",
  "date_col": "Date",
  "output_dir": "models/sp500/outputs/feature_analysis",
  "sample_rows": 380000,
  "time_bucket": "M",
  "random_state": 42,
  "exclude_columns": [
    "P123 ID",
    "Ticker",
    "TBill 10Y 1M change"
  ],
  "leakage_corr_threshold": 0.9,
  "leakage_mi_quantile": 0.99,
  "vif_threshold": 10.0,
  "stability_psi_threshold": 0.2,
  "shap_sample_rows": 10000,
  "log_level": "INFO",
  "log_path": "models/sp500/outputs/feature_analysis/run.log",
  "lgbm_params": {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "num_leaves": 36,
    "min_child_samples": 50
  },
  "top_corr_pairs": 25,
  "categorical_columns": [
    "Sub-Sector Code"
  ],
  "debug_report": false,
  "debug_feature": "macro - VIX index (Series)",
  "debug_symbol_col": "Ticker",
  "debug_symbol": "AAPL",
  "debug_output": "debug_report.html",
  "debug_sample_rows": 20000
}
```

Example `models/sp500/optuna_config.json`:

```json
{
  "data_path": "data/SP500-9MRel-ZsDs-Y-19-64-B.csv",
  "target_col": "TARGET",
  "date_col": "Date",
  "output_dir": "models/sp500/outputs/optuna",
  "exclude_columns": [
    "P123 ID",
    "Ticker"
  ],
  "categorical_columns": [
    "Sub-Sector Code"
  ],
  "n_trials": 50,
  "timeout": null,
  "test_size": 0.2,
  "split_method": "time",
  "gap_days": 280,
  "random_state": 42,
  "metric": "rmse",
  "direction": null,
  "study_name": "sp500_9mrel_timesplit",
  "storage_path": "models/sp500/outputs/optuna/study.db",
  "log_level": "INFO",
  "log_path": "models/sp500/outputs/optuna/run.log",
  "param_ranges": {
    "learning_rate": {
      "min": 0.005,
      "max": 0.1
    },
    "num_leaves": {
      "min": 16,
      "max": 64
    },
    "max_depth": {
      "min": -1,
      "max": 12
    },
    "min_child_samples": {
      "min": 5,
      "max": 100
    },
    "subsample": {
      "min": 0.6,
      "max": 1.0
    },
    "colsample_bytree": {
      "min": 0.6,
      "max": 1.0
    },
    "reg_alpha": {
      "min": 0.001,
      "max": 10.0
    },
    "reg_lambda": {
      "min": 0.001,
      "max": 10.0
    },
    "min_split_gain": {
      "min": 0.0,
      "max": 0.5
    },
    "n_estimators": {
      "min": 400,
      "max": 1200
    }
  }
}
```

Example `models/sp500/shap_config.json`:

```json
{
  "data_path": "data/SP500-9MRel-ZsDs-Y-19-64-B.csv",
  "target_col": "TARGET",
  "date_col": "Date",
  "output_dir": "models/sp500/outputs/shap",
  "model_path": null,
  "sample_rows": 200000,
  "shap_sample_rows": 10000,
  "random_state": 42,
  "exclude_columns": [
    "P123 ID",
    "Ticker"
  ],
  "categorical_columns": [
    "Sub-Sector Code"
  ],
  "lgbm_params": {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_leaves": 36,
    "min_child_samples": 20
  },
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "train_date_range": null,
  "val_date_range": null,
  "test_date_range": null,
  "gap_days": 280,
  "shap_class_index": null,
  "interaction_sample_rows": 2000,
  "top_features": 20,
  "dependence_plots": 6,
  "low_importance_quantile": 0.1,
  "low_importance_threshold": null,
  "force_index": 0,
  "top_interactions": 30,
  "log_level": "INFO",
  "log_path": "models/sp500/outputs/shap/run.log"
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
- Rows with missing or non-finite target values are excluded from feature analysis.
- `debug_report`: Enable the debug report generation (only runs when `--debug` is passed).
- `debug_feature`: Feature/column to plot over time.
- `debug_symbol_col`: Column used to filter a single entity (e.g., `Ticker`).
- `debug_symbol`: Value to filter `debug_symbol_col`.
- `debug_output`: Debug report filename (written to `output_dir`).
- `debug_sample_rows`: Optional sampling cap for debug report data.

### Optuna config options

Rows with a missing `TARGET` are dropped before the split. P123 leaves the target blank for the
last few months of a dataset (the forward return is not known yet) and for some delisted stocks.

- `data_path`: CSV dataset path.
- `target_col`: Target column name.
- `date_col`: Optional date column (removed from features).
- `output_dir`: Output directory for reports and Optuna storage.
- `exclude_columns`: List of columns to exclude from training.
- `categorical_columns`: Columns to treat as categorical features.
- `n_trials`: Number of Optuna trials to run.
- `timeout`: Optional max runtime in seconds.
- `test_size`: Fraction of the most recent dates used for validation (or of rows, with `split_method: "random"`).
- `split_method`: `"time"` (default) splits by date: the last `test_size` share of dates is validation, and training stops `gap_days` before it. Requires `date_col`. `"random"` is a shuffled split and leaks future returns on stock panel data; use it only for non-time-series datasets.
- `gap_days`: Calendar days between the end of training and the start of validation. Set it to at least the target horizon (about 280 for a 9-month target, 100 for a 3-month target).
- `random_state`: Random seed for LightGBM (and the split when `split_method` is `"random"`).
- `metric`: Override metric (`auc`, `logloss`, `multi_logloss`, `accuracy`, `rmse`, `mae`, `mse`, `r2`).
- `direction`: Override optimization direction (`maximize` or `minimize`).
- `study_name`: Optuna study name. Use a new name whenever the data, target or split changes; an existing study with the same name is resumed and its old trials are mixed in.
- `storage_path`: SQLite path for Optuna study storage.
- `log_level`: Logging level (e.g. `INFO`, `DEBUG`).
- `log_path`: Path for the run log file.
- `enable_pruning`: Enable Optuna pruning based on intermediate validation metrics.
- `prune_warmup_steps`: Iterations to wait before pruning starts.
- `prune_interval_steps`: Pruning check interval in boosting iterations.
- `param_ranges`: Optional min/max bounds per tuned parameter.

### SHAP config options

- `data_path`: CSV dataset path.
- `target_col`: Target column name.
- `date_col`: Optional date column; used for time-based splits and date ranges.
- `model_path`: Optional LightGBM model file path.
- `output_dir`: Output directory for SHAP outputs.
- `sample_rows`: Downsample to speed up on large datasets.
- `shap_sample_rows`: Rows sampled for SHAP calculations.
- `train_ratio`, `val_ratio`, `test_ratio`: Split ratios when date ranges are not supplied.
- `train_date_range`, `val_date_range`, `test_date_range`: Optional date ranges in `[start, end]` format.
- `gap_days`: Calendar days between train/validation/test splits when using date ordering. Set it to at least the target horizon.
- `categorical_columns`: Columns to treat as categorical features.
- `lgbm_params`: Dictionary of LightGBM training parameters.
- `interaction_sample_rows`: Rows sampled for SHAP interaction values.
- `dependence_plots`: Number of dependence plots for top features.
- `low_importance_quantile`: Quantile for low-importance candidates.
- `log_level`: Logging level (e.g. `INFO`, `DEBUG`).
- `log_path`: Path for the run log file.

### Feature stats config options

- `data_path`, `target_col`, `date_col`, `output_dir`, `log_level`, `log_path`: as for the other tools.
- `exclude_columns`: Columns to skip (e.g. `P123 ID`, `Ticker`). The date column is always skipped.
- `categorical_columns`: Columns summarized as category counts instead of numeric stats. Non-numeric columns are treated as categorical automatically.
- `include_target`: Also show stats for the target column (marked `target`). Default `true`.
- `sample_rows`: Random row sample; `null` (default) uses all rows.
- `time_bucket`: Period for the over-time charts (`W`, `M`, `Q`). Default `M`.
- `hist_bins`: Histogram bins. Default `50`.
- `hist_clip_quantile`: Clip histogram ranges to the `[q, 1-q]` quantiles (e.g. `0.005`) for raw features with extreme outliers. `0` (default) uses the full min-max range. Stats are never clipped.
- `top_categories`: Categories shown per categorical feature. Default `30`.
- Flag thresholds:
  - `missing_threshold` (default `0.2`): flag `high_missing` above this missing share.
  - `pileup_threshold` (default `0.10`): flag `pile_up` when more than this share of values sit exactly at the min or max. P123 Z-score clipping usually puts 2-9% at each bound, so the default only flags unusual cases.
  - `kurtosis_threshold` (default `10`): flag `heavy_tails` above this excess kurtosis.
  - `low_unique_threshold` (default `20`): tag `discrete` below this many distinct values.
  - `date_constant_threshold` (default `0.9`): tag `date_constant` (macro/market-level) when every stock has the same value on at least this share of dates.
  - `drift_threshold` (default `0.5`): flag `drift` when the std of the per-date mean exceeds this share of the overall std, i.e. the feature isn't normalized by date and its level moves over time.

## CLI usage

```bash
python -m feature_analysis_tool --data data/SP500-9MRel-ZsDs-Y-19-64-B.csv --target TARGET --date Date
```

You can also use a config file and override fields:

```bash
python -m feature_analysis_tool --config models/sp500/feature_analysis_config.json --sample-rows 20000
```

To reload a saved project and regenerate the report:

```bash
python -m feature_analysis_tool --config models/sp500/feature_analysis_config.json --load-project
```

To regenerate the report from saved results only:

```bash
python -m feature_analysis_tool --config models/sp500/feature_analysis_config.json --report-only
```

To generate the debug report:

```bash
python -m feature_analysis_tool --config models/sp500/feature_analysis_config.json --save-project --debug
```

Optuna CLI example:

```bash
python -m optuna_tool --data data/SP500-9MRel-ZsDs-Y-19-64-B.csv --target TARGET --date Date --gap-days 280 --trials 75 --metric rmse
```

SHAP CLI example:

```bash
python -m shap_tool --config models/sp500/shap_config.json
```

## Output

- `report.html`: Interactive report with sortable/filterable table and Plotly charts.
- `results.json`: JSON summary of all metrics.
- `model.txt`: Saved LightGBM model (when trained and `--save-project` is used).

Optuna outputs:

- `outputs/optuna/report.html`: HTML report with study summary, a baseline score (always predicting the
  training-set mean of `TARGET`) and the best trial's % improvement over it, and full-width Optuna charts
  (history, parameter importances, parallel coordinate, one slice plot per parameter). Raw RMSE depends on
  the scale of `TARGET` (a Z-score target has a much larger RMSE than a min-max one), so compare datasets
  by improvement vs baseline, not by RMSE.
- `outputs/optuna/results.json`: Best parameters, baseline score (`study.baseline_value`) and trial summary.
- `outputs/optuna/study.db`: SQLite Optuna study database.

Feature stats outputs:

- `outputs/feature_stats/report.html`: summary tiles, flag filters, a sortable table with a mini histogram per feature, and per-feature detail (all stats, full histogram with log-scale option, P5/median/P95 and mean over time, missing values over time), plus category counts for categorical features.
- `outputs/feature_stats/stats.csv`: one row per feature with every statistic and flag.
- `outputs/feature_stats/results.json`: everything above, including histogram bins and time series. Regenerate the report from it with `--report-only`.

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
- For interactive Optuna dashboards, run `optuna-dashboard sqlite:///models/sp500/outputs/optuna/study.db`.

## Portfolio123 API

`p123_client.py` gives every script the same authenticated
[`p123api`](https://pypi.org/project/p123api/) client:

```python
from p123_client import get_client

with get_client() as client:
    preds = client.aifactor_predict(predictor_id, params, to_pandas=True)
```

Credentials come from `P123_API_ID` and `P123_API_KEY`, either exported in your shell or
set in `.env` at the repo root (gitignored; copy `.env.example`). Check them with:

```bash
python -m p123_client
```

API calls use your P123 API credits. Without a P123 data license, dataset downloads fail
for features whose step 1 and step 2 normalization are both Skip
(see `docs/p123_dataset_download_note.txt`).

## Tests

```bash
python -m pytest tests
```

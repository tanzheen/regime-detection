# Regime Detection

Market-regime detection experiments for SPY using walk-forward validation. The
project currently supports:

- A full-covariance Gaussian Hidden Markov Model.
- Wasserstein K-means with exact multivariate Wasserstein distance.
- Wasserstein K-means with sliced Wasserstein distance.
- Configurable feature sets, walk-forward validation, regime plots, validation
  predictions, and serialized final models.

The latest experiment runner is designed to compare:

- `log_ret` only.
- `log_ret` plus `vol_zscore`.
- Full and sliced Wasserstein calculations for the two-feature case.

Note: Some source paths intentionally use the existing project spelling
`wasterstein` rather than `wasserstein`. Keep that spelling for file paths and
commands that reference existing modules.

## Repository Layout

```text
.
|-- config/
|   `-- regime_features.toml
|-- api/
|   `-- pipeline/
|       |-- app.py
|       `-- regime_pipeline.py
|-- data/
|   |-- inputs/
|   |   |-- spy500.parquet
|   |   `-- folds/
|   |       |-- fold_1_train.parquet
|   |       |-- fold_1_val.parquet
|   |       `-- ...
|   `-- outputs/
|-- scripts/
|   |-- run_hmm.sh
|   |-- run_wasterstein.sh
|   `-- run_regime_feature_experiment.py
|-- src/
|   |-- data/
|   |   |-- prepare.py
|   |   `-- tiingo.py
|   |-- models/
|   |   |-- hmm.py
|   |   `-- wasterstein.py
|   |-- train/
|   |   |-- hmm_train.py
|   |   `-- wasterstein_train.py
|   |-- features.py
|   `-- regime_config.py
|-- pyproject.toml
`-- uv.lock
```

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
```

If you need to download fresh data from Tiingo, create a `.env` file:

```bash
cp .env.example .env
```

Then set:

```text
TIINGO_API_KEY=your_tiingo_api_key
```

The repository already has prepared parquet inputs under `data/inputs/`. You
only need the Tiingo key when rebuilding those inputs from the API.

## Data Preparation

To download SPY daily history and rebuild the walk-forward folds:

```bash
uv run python src/data/prepare.py
```

This writes:

```text
data/inputs/spy500.parquet
data/inputs/folds/fold_1_train.parquet
data/inputs/folds/fold_1_val.parquet
...
data/inputs/folds/fold_5_train.parquet
data/inputs/folds/fold_5_val.parquet
```

The folds are generated with `sklearn.model_selection.TimeSeriesSplit`, so each
validation set is later in time than its corresponding training set.

## Feature Engineering

Feature creation lives in `src/features.py`.

Current available features include:

- `log_ret`: one-period log return from adjusted close.
- `abs_ret`: absolute value of `log_ret`.
- `rvol_<window>d`: rolling return volatility.
- `vol_change`: adjusted volume percentage change.
- `vol_zscore`: rolling z-score of adjusted volume.
- `vol_spike`: binary volume spike indicator.
- `momentum_<window>d`: adjusted-close momentum over a configured window.
- `hl_range`: high-low range normalized by adjusted close, when high and low
  columns are available.

The default config currently uses:

```toml
[hmm]
features = ["log_ret", "vol_zscore"]

[wasserstein]
features = ["log_ret", "vol_zscore"]
```

The experiment runner overrides these feature lists per run, so you do not need
to edit the TOML file for the standard comparisons.

## Configuration

The main config file is:

```text
config/regime_features.toml
```

Important sections:

- `[features]`: controls rolling feature windows and optional high-low range
  creation.
- `[hmm]`: controls HMM states, feature list, standardization, EM iterations,
  transition prior, and model saving.
- `[wasserstein]`: controls number of clusters, segment window parameters,
  feature list, distance methods, sliced projections, and model saving.

For Wasserstein training:

- `h1` is the segment length.
- `h2` is the overlap parameter.
- The step size is `h1 - h2`.
- `methods = ["full", "sliced"]` runs both variants in the direct trainer.
- `final_method` selects which method is used for the final full-data model in
  the direct trainer.

## Recommended Experiment Runner

Use `scripts/run_regime_feature_experiment.py` for the current comparison grid.
It runs walk-forward validation, saves fold plots and fold models, then trains
and saves a final model on the full dataset.

### HMM: log returns only

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret
```

### HMM: log returns plus volume z-score

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret_vol_zscore
```

### Wasserstein: log returns only, full calculation

```bash
uv run python scripts/run_regime_feature_experiment.py --model wasserstein --feature-set log_ret --method full
```

### Wasserstein: log returns plus volume z-score, full multivariate calculation

```bash
uv run python scripts/run_regime_feature_experiment.py --model wasserstein --feature-set log_ret_vol_zscore --method full
```

### Wasserstein: log returns plus volume z-score, sliced calculation

```bash
uv run python scripts/run_regime_feature_experiment.py --model wasserstein --feature-set log_ret_vol_zscore --method sliced
```

To run only specific folds:

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret --folds 1 2
```

To train only the final full-data model:

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret --final-only
```

To run validation without training the final model:

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret --skip-final
```

## Direct Trainers

The lower-level trainer scripts use `config/regime_features.toml` directly.

Run HMM training:

```bash
./scripts/run_hmm.sh
```

Equivalent command:

```bash
uv run python src/train/hmm_train.py
```

Run Wasserstein training:

```bash
./scripts/run_wasterstein.sh
```

Equivalent command:

```bash
uv run python src/train/wasterstein_train.py
```

Useful trainer flags:

```bash
--folds 1 2 3
--final-only
--skip-final
--config config/regime_features.toml
```

Example:

```bash
uv run python src/train/hmm_train.py --folds 1 2 --skip-final
```

## Outputs

The experiment runner writes HMM artifacts under:

```text
data/outputs/hmm/<feature_set>/
```

For example:

```text
data/outputs/hmm/log_ret/
|-- validation_predictions.parquet
|-- hmm_regimes_fold_1.png
|-- ...
|-- hmm_regimes_final.png
`-- models/
    |-- hmm_fold_1.pkl
    |-- ...
    `-- hmm_final.pkl
```

The experiment runner writes Wasserstein artifacts under:

```text
data/outputs/wasterstein/<feature_set>/<method>/
```

For example:

```text
data/outputs/wasterstein/log_ret_vol_zscore/full/
|-- validation_predictions.parquet
|-- regimes_fold_1.png
|-- ...
|-- regimes_final.png
`-- models/
    |-- full_wkmeans_fold_1.pkl
    |-- ...
    `-- wkmeans_final.pkl
```

Validation prediction files include the original validation rows plus:

- `fold`: walk-forward fold number.
- `model` or `method`: model identifier.
- `regime`: assigned regime label.
- `risk_on`: simple derived flag where the highest-volatility regime is treated
  as risk-off.

Regime labels are sorted by log-return variance:

- `0`: lowest volatility.
- `1`: medium volatility for three-state models.
- `2`: highest volatility for three-state models.

## Model Details

### Gaussian HMM

The HMM implementation is in `src/models/hmm.py`. It uses:

- Full covariance Gaussian emissions.
- K-means initialization.
- EM training.
- Optional transition prior favoring persistent regimes.
- State sorting by variance of the first feature, which is `log_ret` in the
  standard experiments.

Walk-forward validation trains on each fold's training data, filters states on
the training set, then initializes validation filtering from the final filtered
training-state distribution.

### Wasserstein K-means

The Wasserstein implementation is in `src/models/wasterstein.py`. It supports:

- One-dimensional Wasserstein distance.
- Exact empirical multivariate Wasserstein distance via optimal atom matching.
- Sliced Wasserstein distance using random projection directions.
- K-means-style centroid updates and cluster assignment.
- Cluster sorting by variance of the first feature, which is `log_ret` in the
  standard experiments.

Walk-forward validation converts feature rows into overlapping fixed-length
segments before fitting and prediction.

## Working With Saved Models

Models are serialized with `pickle`.

HMM final models are saved directly as `GaussianHMM` objects. The trainer adds:

- `feature_columns_`
- `feature_scaler_`
- `feature_config_`

Wasserstein models are saved as dictionaries containing:

- `model`
- `feature_columns`
- `feature_scaler`
- `feature_config`
- `method`
- metadata such as `fold` or `trained_on`

Example load:

```python
import pickle
from pathlib import Path

model_path = Path("data/outputs/hmm/log_ret/models/hmm_final.pkl")
with model_path.open("rb") as f:
    model = pickle.load(f)
```

For Wasserstein:

```python
import pickle
from pathlib import Path

model_path = Path(
    "data/outputs/wasterstein/log_ret_vol_zscore/full/models/wkmeans_final.pkl"
)
with model_path.open("rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_columns = artifact["feature_columns"]
```

## FastAPI Regime API

The API in `api/pipeline` serves the selected production-style model:

```text
log_ret + vol_zscore + sliced Wasserstein K-means
```

It loads:

```text
data/outputs/wasterstein/log_ret_vol_zscore/sliced/models/wkmeans_final.pkl
```

Run the service:

```bash
uv run uvicorn api.pipeline.app:app --host 127.0.0.1 --port 8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

```text
GET  /health
GET  /metadata
POST /predict/latest
POST /predict
```

The model requires daily bars with:

```text
date
adjClose
adjVolume
```

Send at least 25 recent bars so the 20-day `vol_zscore` and the 5-day
Wasserstein segment can be calculated. Extra OHLCV fields are accepted and
ignored.

Example request:

```json
{
  "include_features": true,
  "records": [
    {
      "date": "2026-05-18",
      "adjClose": 738.65,
      "adjVolume": 47843865
    },
    {
      "date": "2026-05-19",
      "adjClose": 733.73,
      "adjVolume": 54255913
    }
  ]
}
```

The example above is shortened for readability. A real prediction request
should include 25 or more bars.

Example latest-regime response:

```json
{
  "result": {
    "date": "2026-05-22",
    "adjClose": 745.64,
    "regime": 0.0,
    "regime_name": "low_volatility",
    "risk_on_low_medium": true,
    "risk_on_low_only": true,
    "high_volatility": false,
    "log_ret": 0.003923786914972638,
    "vol_zscore": -0.7103601858936398
  }
}
```

Response fields:

- `regime`: numeric volatility regime, where `0 = low`, `1 = medium`, and
  `2 = high`.
- `regime_name`: human-readable regime label.
- `risk_on_low_medium`: `true` when the regime is low or medium volatility.
- `risk_on_low_only`: `true` only when the regime is low volatility.
- `high_volatility`: `true` only when the regime is high volatility.
- `log_ret` and `vol_zscore`: unscaled features, returned only when
  `include_features` is `true`.

Recommended use of this API output:

- Use `switch_low_vs_medium_high` with the `log_ret_vol_zscore_sliced` model.
- In implementation terms, be long only when `regime == 0` or
  `risk_on_low_only == true`.
- Exit or stay flat when `regime` switches to `1` or `2`.
- This recommendation is based on the last two validation folds, where this
  rule had the highest Sharpe ratio among the tested rules.

You can also use the pipeline directly from Python:

```python
import pandas as pd
from api.pipeline import SlicedWassersteinRegimePipeline

prices = pd.read_parquet("data/inputs/spy500.parquet").tail(60)
pipeline = SlicedWassersteinRegimePipeline()
latest = pipeline.predict_latest(prices)
print(latest)
```

Or via CLI:

```bash
uv run python -m api.pipeline.regime_pipeline \
  --input data/inputs/spy500.parquet \
  --latest
```

## Backtest Results

The backtests use walk-forward validation predictions, not the final full-data
model. Signals are calculated at the close and applied to the next bar's
return. Annual average return is the mean daily strategy return multiplied by
252. Sharpe uses a zero risk-free rate.

Full report files:

```text
data/outputs/backtests/trend_following_wasterstein/report.md
data/outputs/backtests/trend_following_wasterstein_last2/report.md
```

Strategy definitions:

- `buy_hold`: buy and hold SPY over the same validation dates.
- `ma_only`: 10-day/30-day SMA crossover with no regime filter.
- `allow_low_medium`: 10-day/30-day SMA crossover; allow long exposure only in
  low or medium volatility.
- `allow_low_only`: 10-day/30-day SMA crossover; allow long exposure only in
  low volatility.
- `switch_low_vs_medium_high`: regime-switch strategy; long in low volatility
  and flat in medium or high volatility.
- `switch_low_medium_vs_high`: regime-switch strategy; long in low or medium
  volatility and flat in high volatility.

### Last Two Validation Folds

These are the most recent validation folds, `4` and `5`.

| Strategy | Version | Annual Avg Return | Sharpe | CAGR | Max Drawdown | Exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | SPY | 14.70% | 0.828 | 14.02% | -33.70% | 100.0% |
| ma_only | no regime filter | 8.63% | 0.789 | 8.36% | -17.67% | 68.6% |
| allow_low_medium | log_ret_vol_zscore_sliced | 8.33% | 0.833 | 8.14% | -17.01% | 62.8% |
| allow_low_only | log_ret_vol_zscore_sliced | 5.72% | 0.716 | 5.55% | -16.80% | 39.7% |
| switch_low_vs_medium_high | log_ret_vol_zscore_sliced | 10.67% | 1.018 | 10.65% | -21.89% | 53.5% |
| switch_low_medium_vs_high | log_ret_vol_zscore_sliced | 12.89% | 0.926 | 12.66% | -27.69% | 87.5% |

Recent-fold takeaway: buy-and-hold had the highest return, while the sliced
`log_ret + vol_zscore` `switch_low_vs_medium_high` rule had the highest Sharpe
ratio. The API model is the regime model behind the sliced rows.

Recommendation: use `switch_low_vs_medium_high` with
`log_ret_vol_zscore_sliced` as the default regime rule. It is the cleanest
risk-adjusted result in the most recent validation folds: long during
low-volatility regimes, flat during medium- or high-volatility regimes.

### All Validation Folds

This covers all five walk-forward validation folds.

| Strategy | Version | Annual Avg Return | Sharpe | CAGR | Max Drawdown | Exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| buy_hold | SPY | 10.67% | 0.549 | 9.17% | -55.20% | 100.0% |
| ma_only | no regime filter | 5.61% | 0.485 | 5.06% | -33.73% | 64.7% |
| allow_low_medium | log_ret_vol_zscore_sliced | 4.65% | 0.433 | 4.16% | -45.79% | 60.1% |
| allow_low_only | log_ret_vol_zscore_sliced | 3.04% | 0.355 | 2.71% | -32.46% | 37.4% |
| switch_low_vs_medium_high | log_ret_vol_zscore_sliced | 5.73% | 0.480 | 5.15% | -36.39% | 50.7% |
| switch_low_medium_vs_high | log_ret_vol_zscore_sliced | 7.41% | 0.473 | 6.37% | -61.56% | 86.4% |

All-fold takeaway: buy-and-hold won on return and Sharpe, but the regime model
can reduce exposure and, depending on the rule, reduce drawdown.

## Development Notes

Run a quick syntax check:

```bash
uv run python -m compileall src scripts
```

Run a one-fold smoke test without training a final model:

```bash
uv run python scripts/run_regime_feature_experiment.py --model hmm --feature-set log_ret --folds 1 --skip-final
```

The project currently does not have a formal test suite. The practical
verification path is to run one-fold smoke tests, then run the full experiment
grid.

## Caveats

- This is research code for regime-detection experiments, not investment
  advice.
- Results are sensitive to feature scaling, window sizes, number of states or
  clusters, and train/validation split design.
- Pickle artifacts are Python-specific and should only be loaded from trusted
  sources.
- The existing code and paths use `wasterstein` in several places. Renaming it
  would require coordinated updates across imports, scripts, and output paths.

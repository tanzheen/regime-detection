import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import build_feature_matrix, labels_to_price_rows
from models.hmm import GaussianHMM
from regime_config import DEFAULT_CONFIG_PATH, load_config

INPUT_FOLDS_DIR = PROJECT_ROOT / "data" / "inputs" / "folds"
FULL_INPUT_PATH = PROJECT_ROOT / "data" / "inputs" / "spy500.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "hmm"
MODEL_DIR = OUTPUT_DIR / "models"
REGIME_COLORS = ["green", "orange", "red", "purple", "blue", "brown"]
THREE_STATE_LABELS = ["Low volatility", "Medium volatility", "High volatility"]


def plot_regimes(
    df: pd.DataFrame,
    label_series: pd.Series,
    output_dir: Path,
    n_regimes: int,
    fold: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    run_label = f"Fold {fold}" if fold is not None else "Final full dataset"
    ax.set_title(f"SPY - Gaussian HMM Market Regimes (k={n_regimes}) | {run_label}")

    dates = df["date"].values
    prices = df["adjClose"].values
    labels = label_series.values

    for i in range(1, len(dates)):
        if np.isnan(labels[i]):
            continue
        regime = int(labels[i])
        ax.plot(
            dates[i - 1 : i + 1],
            prices[i - 1 : i + 1],
            color=_regime_color(regime),
            linewidth=0.8,
        )

    patches = [
        mpatches.Patch(color=_regime_color(state), label=_regime_label(state, n_regimes))
        for state in range(n_regimes)
    ]
    ax.legend(handles=patches)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"hmm_regimes_fold_{fold}.png" if fold is not None else "hmm_regimes_final.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def run_fold(fold: int, config: dict[str, Any]) -> pd.DataFrame:
    hmm_config = config["hmm"]
    feature_config = config["features"]
    feature_columns = list(hmm_config["features"])
    n_states = int(hmm_config.get("n_states", 3))
    standardize = bool(hmm_config.get("standardize", True))

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold}")
    print(f"{'=' * 60}")

    train = pd.read_parquet(INPUT_FOLDS_DIR / f"fold_{fold}_train.parquet")
    val = pd.read_parquet(INPUT_FOLDS_DIR / f"fold_{fold}_val.parquet")

    train_features = build_feature_matrix(
        train,
        feature_columns,
        feature_config,
        fit_scaler=True,
        standardize=standardize,
    )
    val_features = build_feature_matrix(
        val,
        feature_columns,
        feature_config,
        scaler=train_features.scaler,
        standardize=standardize,
    )

    print(f"Features: {feature_columns}")
    print(f"Train: {len(train)} rows -> {len(train_features.values)} feature rows")
    print(f"Val:   {len(val)} rows -> {len(val_features.values)} feature rows")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        max_iter=int(hmm_config.get("max_iter", 200)),
        tol=float(hmm_config.get("tol", 1e-6)),
        random_state=int(hmm_config.get("random_state", 42)),
        min_variance=float(hmm_config.get("min_variance", 1e-8)),
        verbose=bool(hmm_config.get("verbose", True)),
    )
    model.transmat_prior = _transition_prior(
        n_states,
        stay_probability=float(hmm_config.get("stay_probability", 0.9)),
        weight=float(hmm_config.get("transition_prior_weight", 0.0)),
    )
    model.fit(train_features.values)

    train_prob = model.filter_proba(train_features.values)
    train_states = np.argmax(train_prob, axis=1)
    train_labels = labels_to_price_rows(train, train_features.frame, train_states)

    val_initial_prob = train_prob[-1] @ model.transmat_
    val_prob = model.filter_proba(
        val_features.values,
        initial_state_prob=val_initial_prob,
    )
    val_states = np.argmax(val_prob, axis=1)
    val_labels = labels_to_price_rows(val, val_features.frame, val_states)

    print(
        f"\nTrain regime distribution:\n"
        f"{train_labels.value_counts(normalize=True, dropna=True).round(3)}"
    )
    print(
        f"Val regime distribution:\n"
        f"{val_labels.value_counts(normalize=True, dropna=True).round(3)}"
    )
    print(f"Log-likelihood: {model.score(train_features.values):.2f}")
    print(f"Transition matrix:\n{np.round(model.transmat_, 3)}")
    print("State parameters:")
    for state in range(n_states):
        means = {
            column: round(float(value), 4)
            for column, value in zip(feature_columns, model.means_[state])
        }
        print(
            f"  {_regime_label(state, n_states)} "
            f"log_ret_var={model.variances_[state]:.6f} means={means}"
        )

    model.feature_columns_ = feature_columns
    model.feature_scaler_ = train_features.scaler
    model.feature_config_ = feature_config

    if bool(hmm_config.get("save_fold_models", False)):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / f"hmm_fold_{fold}.pkl", "wb") as model_file:
            pickle.dump(model, model_file)

    plot_regimes(val, val_labels, output_dir=OUTPUT_DIR, n_regimes=n_states, fold=fold)
    val_result = val.copy()
    val_result["fold"] = fold
    val_result["model"] = "hmm"
    val_result["regime"] = val_labels
    val_result["risk_on"] = np.where(
        val_result["regime"].isna(),
        np.nan,
        val_result["regime"] != (n_states - 1),
    )
    return val_result


def run_final_model(config: dict[str, Any]) -> None:
    hmm_config = config["hmm"]
    feature_config = config["features"]
    feature_columns = list(hmm_config["features"])
    n_states = int(hmm_config.get("n_states", 3))
    standardize = bool(hmm_config.get("standardize", True))

    print(f"\n{'=' * 60}")
    print("FINAL HMM MODEL")
    print(f"{'=' * 60}")

    data = pd.read_parquet(FULL_INPUT_PATH)
    feature_matrix = build_feature_matrix(
        data,
        feature_columns,
        feature_config,
        fit_scaler=True,
        standardize=standardize,
    )

    print(f"Features: {feature_columns}")
    print(f"Full dataset: {len(data)} rows -> {len(feature_matrix.values)} feature rows")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        max_iter=int(hmm_config.get("max_iter", 200)),
        tol=float(hmm_config.get("tol", 1e-6)),
        random_state=int(hmm_config.get("random_state", 42)),
        min_variance=float(hmm_config.get("min_variance", 1e-8)),
        verbose=bool(hmm_config.get("verbose", True)),
    )
    model.transmat_prior = _transition_prior(
        n_states,
        stay_probability=float(hmm_config.get("stay_probability", 0.9)),
        weight=float(hmm_config.get("transition_prior_weight", 0.0)),
    )
    model.fit(feature_matrix.values)
    states = model.filter_states(feature_matrix.values)
    labels = labels_to_price_rows(data, feature_matrix.frame, states)

    print(
        f"Full-data regime distribution:\n"
        f"{labels.value_counts(normalize=True, dropna=True).round(3)}"
    )
    print(f"Log-likelihood: {model.score(feature_matrix.values):.2f}")
    print(f"Transition matrix:\n{np.round(model.transmat_, 3)}")

    model.feature_columns_ = feature_columns
    model.feature_scaler_ = feature_matrix.scaler
    model.feature_config_ = feature_config

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "hmm_final.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved final model: {MODEL_DIR / 'hmm_final.pkl'}")

    plot_regimes(
        data,
        labels,
        output_dir=OUTPUT_DIR,
        n_regimes=n_states,
        fold=None,
    )


def _transition_prior(
    n_states: int,
    *,
    stay_probability: float,
    weight: float,
) -> np.ndarray:
    if weight <= 0:
        return np.zeros((n_states, n_states))
    if n_states == 1:
        return np.ones((1, 1)) * weight

    stay_probability = float(np.clip(stay_probability, 0.0, 1.0))
    switch_probability = (1.0 - stay_probability) / (n_states - 1)
    prior = np.full((n_states, n_states), switch_probability)
    np.fill_diagonal(prior, stay_probability)
    return prior * weight


def _regime_label(state: int, n_regimes: int) -> str:
    if n_regimes == 3:
        return THREE_STATE_LABELS[state]
    return f"State {state}"


def _regime_color(state: int) -> str:
    return REGIME_COLORS[state % len(REGIME_COLORS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configured Gaussian HMM folds")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to regime feature TOML config",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, 6)),
        help="Fold numbers to run",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip walk-forward folds and train only the final full-data model",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip the final full-data model even if enabled in config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaded_config = load_config(args.config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    validation_results = []
    if not args.final_only:
        for fold_number in args.folds:
            validation_results.append(run_fold(fold_number, loaded_config))

    if validation_results:
        validation_path = OUTPUT_DIR / "validation_predictions.parquet"
        pd.concat(validation_results, ignore_index=True).to_parquet(
            validation_path,
            index=False,
        )
        print(f"\nSaved walk-forward validation predictions: {validation_path}")

    should_train_final = bool(loaded_config["hmm"].get("train_final_model", True))
    if should_train_final and not args.skip_final:
        run_final_model(loaded_config)

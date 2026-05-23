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

from features import (
    build_feature_matrix,
    build_segments,
    segment_labels_to_price_rows,
)
from models.wasterstein import MultivariateWKMeans, SlicedWKMeans
from regime_config import DEFAULT_CONFIG_PATH, load_config

INPUT_FOLDS_DIR = PROJECT_ROOT / "data" / "inputs" / "folds"
FULL_INPUT_PATH = PROJECT_ROOT / "data" / "inputs" / "spy500.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "wasterstein"
MODEL_DIR = OUTPUT_DIR / "models"
REGIME_COLORS = ["green", "orange", "red", "purple", "blue", "brown"]
THREE_STATE_LABELS = ["Low volatility", "Medium volatility", "High volatility"]


def plot_regimes(
    df: pd.DataFrame,
    label_series: pd.Series,
    output_dir: Path,
    n_regimes: int,
    fold: int | None,
    method: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    run_label = f"Fold {fold}" if fold is not None else "Final full dataset"
    ax.set_title(f"SPY - {method.title()} WK-means Market Regimes (k={n_regimes}) | {run_label}")

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
    filename = f"regimes_fold_{fold}.png" if fold is not None else "regimes_final.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def run_fold(fold: int, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    wasserstein_config = config["wasserstein"]
    feature_config = config["features"]
    feature_columns = list(wasserstein_config["features"])
    standardize = bool(wasserstein_config.get("standardize", True))
    h1 = int(wasserstein_config.get("h1", 5))
    h2 = int(wasserstein_config.get("h2", 4))
    k = int(wasserstein_config.get("k", 3))

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
    train_segments = build_segments(train_features.values, h1=h1, h2=h2)
    val_segments = build_segments(val_features.values, h1=h1, h2=h2)

    print(f"Features: {feature_columns}")
    print(f"Train segments: {train_segments.values.shape}")
    print(f"Val segments:   {val_segments.values.shape}")

    fold_results = {}
    for method in wasserstein_config.get("methods", ["full", "sliced"]):
        print(f"\n{method.upper()} multivariate WK-means")
        model = _make_model(method, wasserstein_config)
        model.fit(train_segments.values)

        train_labels = segment_labels_to_price_rows(
            train,
            train_features.frame,
            model.labels_,
            train_segments.end_feature_rows,
        )
        val_preds = model.predict(val_segments.values)
        val_labels = segment_labels_to_price_rows(
            val,
            val_features.frame,
            val_preds,
            val_segments.end_feature_rows,
        )

        print(
            f"Train regime distribution:\n"
            f"{train_labels.value_counts(normalize=True, dropna=True).round(3)}"
        )
        print(
            f"Val regime distribution:\n"
            f"{val_labels.value_counts(normalize=True, dropna=True).round(3)}"
        )
        centroid_variances = {
            _regime_label(state, k): round(float(np.var(model.centroids_[state][:, 0])), 6)
            for state in range(k)
        }
        print(f"Centroid log-return variances: {centroid_variances}")

        val_result = val.copy()
        val_result["fold"] = fold
        val_result["method"] = method
        val_result["regime"] = val_labels
        val_result["risk_on"] = np.where(
            val_result["regime"].isna(),
            np.nan,
            val_result["regime"] != (k - 1),
        )
        fold_results[method] = val_result

        method_output_dir = OUTPUT_DIR / method
        plot_regimes(
            val,
            val_labels,
            output_dir=method_output_dir,
            n_regimes=k,
            fold=fold,
            method=method,
        )

        if bool(wasserstein_config.get("save_fold_models", False)):
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / f"{method}_wkmeans_fold_{fold}.pkl"
            with open(model_path, "wb") as model_file:
                pickle.dump(
                    {
                        "model": model,
                        "feature_columns": feature_columns,
                        "feature_scaler": train_features.scaler,
                        "feature_config": feature_config,
                        "method": method,
                        "fold": fold,
                    },
                    model_file,
                )

    return fold_results


def run_final_model(config: dict[str, Any]) -> None:
    wasserstein_config = config["wasserstein"]
    feature_config = config["features"]
    feature_columns = list(wasserstein_config["features"])
    standardize = bool(wasserstein_config.get("standardize", True))
    h1 = int(wasserstein_config.get("h1", 5))
    h2 = int(wasserstein_config.get("h2", 4))
    k = int(wasserstein_config.get("k", 3))
    method = str(wasserstein_config.get("final_method", "sliced"))

    print(f"\n{'=' * 60}")
    print(f"FINAL {method.upper()} WK-MEANS MODEL")
    print(f"{'=' * 60}")

    data = pd.read_parquet(FULL_INPUT_PATH)
    feature_matrix = build_feature_matrix(
        data,
        feature_columns,
        feature_config,
        fit_scaler=True,
        standardize=standardize,
    )
    segments = build_segments(feature_matrix.values, h1=h1, h2=h2)

    print(f"Features: {feature_columns}")
    print(f"Full-data segments: {segments.values.shape}")

    model = _make_model(method, wasserstein_config)
    model.fit(segments.values)
    labels = segment_labels_to_price_rows(
        data,
        feature_matrix.frame,
        model.labels_,
        segments.end_feature_rows,
    )

    print(
        f"Full-data regime distribution:\n"
        f"{labels.value_counts(normalize=True, dropna=True).round(3)}"
    )
    centroid_variances = {
        _regime_label(state, k): round(float(np.var(model.centroids_[state][:, 0])), 6)
        for state in range(k)
    }
    print(f"Centroid log-return variances: {centroid_variances}")

    plot_regimes(
        data,
        labels,
        output_dir=OUTPUT_DIR / method,
        n_regimes=k,
        fold=None,
        method=method,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "wkmeans_final.pkl", "wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "feature_scaler": feature_matrix.scaler,
                "feature_config": feature_config,
                "method": method,
                "trained_on": str(FULL_INPUT_PATH),
            },
            model_file,
        )
    print(f"Saved final model: {MODEL_DIR / 'wkmeans_final.pkl'}")


def _make_model(method: str, config: dict[str, Any]) -> MultivariateWKMeans:
    model_kwargs = {
        "k": int(config.get("k", 3)),
        "p": int(config.get("p", 1)),
        "max_iter": int(config.get("max_iter", 50)),
        "tol": float(config.get("tol", 1e-7)),
        "random_state": int(config.get("random_state", 42)),
    }
    if method == "full":
        return MultivariateWKMeans(**model_kwargs)
    if method == "sliced":
        return SlicedWKMeans(
            **model_kwargs,
            n_projections=int(config.get("n_projections", 50)),
        )
    raise ValueError(f"Unsupported wasserstein method: {method}")


def _regime_label(state: int, n_regimes: int) -> str:
    if n_regimes == 3:
        return THREE_STATE_LABELS[state]
    return f"State {state}"


def _regime_color(state: int) -> str:
    return REGIME_COLORS[state % len(REGIME_COLORS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configured WK-means folds")
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

    validation_results: dict[str, list[pd.DataFrame]] = {}
    if not args.final_only:
        for fold_number in args.folds:
            fold_results = run_fold(fold_number, loaded_config)
            for method, result in fold_results.items():
                validation_results.setdefault(method, []).append(result)

    for method, results in validation_results.items():
        validation_path = OUTPUT_DIR / method / "validation_predictions.parquet"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(results, ignore_index=True).to_parquet(
            validation_path,
            index=False,
        )
        print(f"\nSaved {method} walk-forward validation predictions: {validation_path}")

    should_train_final = bool(
        loaded_config["wasserstein"].get("train_final_model", True)
    )
    if should_train_final and not args.skip_final:
        run_final_model(loaded_config)

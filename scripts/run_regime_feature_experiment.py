from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_config import DEFAULT_CONFIG_PATH, load_config
from train import hmm_train, wasterstein_train

FEATURE_SETS = {
    "log_ret": ["log_ret"],
    "log_ret_vol_zscore": ["log_ret", "vol_zscore"],
}

DEFAULT_FOLDS = list(range(1, 6))


def run_hmm(args: argparse.Namespace, config: dict[str, Any]) -> None:
    feature_columns = FEATURE_SETS[args.feature_set]
    run_config = copy.deepcopy(config)
    run_config["hmm"]["features"] = feature_columns
    run_config["hmm"]["save_fold_models"] = True
    run_config["hmm"]["train_final_model"] = True

    output_dir = PROJECT_ROOT / "data" / "outputs" / "hmm" / args.feature_set
    model_dir = output_dir / "models"
    hmm_train.OUTPUT_DIR = output_dir
    hmm_train.MODEL_DIR = model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running HMM experiment with features={feature_columns}")
    print(f"Output directory: {output_dir}")

    validation_results = []
    if not args.final_only:
        for fold in args.folds:
            validation_results.append(hmm_train.run_fold(fold, run_config))

    if validation_results:
        validation_path = output_dir / "validation_predictions.parquet"
        pd.concat(validation_results, ignore_index=True).to_parquet(
            validation_path,
            index=False,
        )
        print(f"\nSaved walk-forward validation predictions: {validation_path}")

    if not args.skip_final:
        hmm_train.run_final_model(run_config)


def run_wasserstein(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.method is None:
        raise ValueError("--method is required when --model wasserstein")

    feature_columns = FEATURE_SETS[args.feature_set]
    run_config = copy.deepcopy(config)
    run_config["wasserstein"]["features"] = feature_columns
    run_config["wasserstein"]["methods"] = [args.method]
    run_config["wasserstein"]["final_method"] = args.method
    run_config["wasserstein"]["save_fold_models"] = True
    run_config["wasserstein"]["train_final_model"] = True

    output_dir = PROJECT_ROOT / "data" / "outputs" / "wasterstein" / args.feature_set
    method_output_dir = output_dir / args.method
    model_dir = method_output_dir / "models"
    wasterstein_train.OUTPUT_DIR = output_dir
    wasterstein_train.MODEL_DIR = model_dir
    method_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Running Wasserstein experiment "
        f"with features={feature_columns}, method={args.method}"
    )
    print(f"Output directory: {method_output_dir}")

    validation_results = []
    if not args.final_only:
        for fold in args.folds:
            fold_results = wasterstein_train.run_fold(fold, run_config)
            validation_results.append(fold_results[args.method])

    if validation_results:
        validation_path = method_output_dir / "validation_predictions.parquet"
        pd.concat(validation_results, ignore_index=True).to_parquet(
            validation_path,
            index=False,
        )
        print(
            "\nSaved walk-forward validation predictions: "
            f"{validation_path}"
        )

    if not args.skip_final:
        wasterstein_train.run_final_model(run_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one regime-detection feature experiment with walk-forward "
            "validation, plots, fold models, and a final full-data model."
        )
    )
    parser.add_argument(
        "--model",
        choices=["hmm", "wasserstein", "wasterstein"],
        required=True,
        help="Model family to train.",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        required=True,
        help="Feature set to use for this experiment.",
    )
    parser.add_argument(
        "--method",
        choices=["full", "sliced"],
        help="Wasserstein distance variant. Required for --model wasserstein.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Base TOML config to copy hyperparameters from.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=DEFAULT_FOLDS,
        help="Walk-forward fold numbers to run.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip walk-forward folds and train only the final full-data model.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Run walk-forward folds but skip the final full-data model.",
    )

    args = parser.parse_args()
    if args.model in {"wasserstein", "wasterstein"} and args.method is None:
        parser.error("--method is required when --model wasserstein")
    if args.model == "hmm" and args.method is not None:
        parser.error("--method only applies to --model wasserstein")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model == "hmm":
        run_hmm(args, config)
    else:
        run_wasserstein(args, config)


if __name__ == "__main__":
    main()

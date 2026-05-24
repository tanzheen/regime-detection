from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import build_feature_matrix, build_segments, segment_labels_to_price_rows


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "wasterstein"
    / "log_ret_vol_zscore"
    / "sliced"
    / "models"
    / "wkmeans_final.pkl"
)

REGIME_NAMES = {
    0: "low_volatility",
    1: "medium_volatility",
    2: "high_volatility",
}

REQUIRED_PRICE_COLUMNS = {"date", "adjClose", "adjVolume"}


@dataclass(frozen=True)
class RegimePipelineMetadata:
    model_path: str
    method: str
    feature_columns: list[str]
    h1: int
    h2: int
    regime_names: dict[int, str]


class SlicedWassersteinRegimePipeline:
    """
    Prediction API for the log_ret + vol_zscore sliced Wasserstein model.

    The input DataFrame should contain enough price history to calculate the
    rolling volume z-score and the Wasserstein segment window. The minimum is
    usually 25 rows with the default config, but passing a longer recent history
    is preferable.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.artifact = self._load_artifact(self.model_path)
        self.model = self.artifact["model"]
        self.feature_columns = list(self.artifact["feature_columns"])
        self.feature_scaler = self.artifact.get("feature_scaler")
        self.feature_config = dict(self.artifact["feature_config"])
        self.method = str(self.artifact.get("method", "sliced"))
        self.h1 = self._infer_h1()
        self.h2 = self.h1 - 1

        if self.method != "sliced":
            raise ValueError(f"Expected sliced model artifact, got method={self.method!r}")
        if self.feature_columns != ["log_ret", "vol_zscore"]:
            raise ValueError(
                "Expected feature columns ['log_ret', 'vol_zscore'], "
                f"got {self.feature_columns}"
            )

    @property
    def metadata(self) -> RegimePipelineMetadata:
        return RegimePipelineMetadata(
            model_path=str(self.model_path),
            method=self.method,
            feature_columns=self.feature_columns,
            h1=self.h1,
            h2=self.h2,
            regime_names=REGIME_NAMES,
        )

    def predict(self, prices: pd.DataFrame, *, include_features: bool = False) -> pd.DataFrame:
        """
        Predict regimes for all rows where features and segments are available.

        Returns one row per input price row, with NaN regime values before the
        rolling features and segment window are available.
        """
        price_frame = self._prepare_prices(prices)
        feature_matrix = build_feature_matrix(
            price_frame,
            self.feature_columns,
            self.feature_config,
            scaler=self.feature_scaler,
            standardize=self.feature_scaler is not None,
        )
        segments = build_segments(feature_matrix.values, h1=self.h1, h2=self.h2)
        predictions = self.model.predict(segments.values)
        labels = segment_labels_to_price_rows(
            price_frame,
            feature_matrix.frame,
            predictions,
            segments.end_feature_rows,
        )

        result = price_frame[["date", "adjClose"]].copy()
        result["regime"] = labels
        result["regime_name"] = result["regime"].map(_regime_name)
        result["risk_on_low_medium"] = result["regime"].map(
            lambda value: _is_allowed(value, {0, 1})
        )
        result["risk_on_low_only"] = result["regime"].map(
            lambda value: _is_allowed(value, {0})
        )
        result["high_volatility"] = result["regime"].map(
            lambda value: _is_allowed(value, {2})
        )

        if include_features:
            result = self._attach_unscaled_features(result, price_frame)

        return result

    def predict_latest(
        self,
        prices: pd.DataFrame,
        *,
        include_features: bool = False,
    ) -> dict[str, Any]:
        """Return the latest available non-null regime prediction."""
        predictions = self.predict(prices, include_features=include_features)
        usable = predictions.dropna(subset=["regime"])
        if usable.empty:
            raise ValueError(
                "No regime prediction is available. Provide more rows of price history."
            )
        return _json_ready_row(usable.iloc[-1].to_dict())

    def predict_records(
        self,
        records: list[dict[str, Any]],
        *,
        include_features: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict regimes from JSON-style OHLCV records."""
        predictions = self.predict(
            pd.DataFrame.from_records(records),
            include_features=include_features,
        )
        return [_json_ready_row(row) for row in predictions.to_dict(orient="records")]

    def _load_artifact(self, model_path: Path) -> dict[str, Any]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        with model_path.open("rb") as model_file:
            artifact = pickle.load(model_file)

        if not isinstance(artifact, dict):
            raise ValueError(f"Expected dict model artifact, got {type(artifact)}")

        required = {"model", "feature_columns", "feature_config", "method"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
        return artifact

    def _infer_h1(self) -> int:
        centroids = getattr(self.model, "centroids_", None)
        if not centroids:
            raise ValueError("Model has no centroids. Was the model fitted?")
        first_centroid = np.asarray(centroids[0])
        if first_centroid.ndim != 2:
            raise ValueError(f"Expected 2D centroid, got shape {first_centroid.shape}")
        return int(first_centroid.shape[0])

    def _prepare_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
        if missing:
            raise ValueError(f"Missing required price columns: {sorted(missing)}")

        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame

    def _attach_unscaled_features(
        self,
        result: pd.DataFrame,
        price_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        unscaled = build_feature_matrix(
            price_frame,
            self.feature_columns,
            self.feature_config,
            standardize=False,
        )
        result = result.copy()
        for column in self.feature_columns:
            aligned = pd.Series(np.nan, index=result.index, dtype=float)
            rows = unscaled.frame["_price_row"].to_numpy(dtype=int)
            aligned.iloc[rows] = unscaled.frame[column].to_numpy(dtype=float)
            result[column] = aligned
        return result


def load_prices(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        records = payload["records"] if isinstance(payload, dict) else payload
        return pd.DataFrame.from_records(records)
    raise ValueError(f"Unsupported input format for {input_path}")


def write_predictions(predictions: pd.DataFrame | dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(predictions, pd.DataFrame):
        suffix = output_path.suffix.lower()
        if suffix == ".parquet":
            predictions.to_parquet(output_path, index=False)
            return
        if suffix == ".csv":
            predictions.to_csv(output_path, index=False)
            return
        records = [_json_ready_row(row) for row in predictions.to_dict(orient="records")]
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return

    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict regimes with the log_ret + vol_zscore sliced WK-means model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input price history file: parquet, csv, or JSON records.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path: parquet, csv, or JSON. Defaults to stdout JSON.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the sliced WK-means model artifact.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Return only the latest available regime prediction.",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include unscaled log_ret and vol_zscore columns in the output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SlicedWassersteinRegimePipeline(args.model_path)
    prices = load_prices(args.input)

    if args.latest:
        output = pipeline.predict_latest(prices, include_features=args.include_features)
    else:
        output = pipeline.predict(prices, include_features=args.include_features)

    if args.output:
        write_predictions(output, args.output)
        print(f"Saved predictions: {args.output}")
        return

    if isinstance(output, pd.DataFrame):
        records = [_json_ready_row(row) for row in output.to_dict(orient="records")]
        print(json.dumps(records, indent=2))
    else:
        print(json.dumps(output, indent=2))


def _regime_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return REGIME_NAMES.get(int(value), f"regime_{int(value)}")


def _is_allowed(value: Any, allowed: set[int]) -> bool | None:
    if pd.isna(value):
        return None
    return int(value) in allowed


def _json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, pd.Timestamp):
            cleaned[key] = value.date().isoformat()
        elif isinstance(value, np.integer):
            cleaned[key] = int(value)
        elif isinstance(value, np.floating):
            cleaned[key] = float(value)
        elif isinstance(value, np.bool_):
            cleaned[key] = bool(value)
        else:
            cleaned[key] = value
    return cleaned


if __name__ == "__main__":
    main()

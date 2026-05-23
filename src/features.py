from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureMatrix:
    values: np.ndarray
    frame: pd.DataFrame
    columns: list[str]
    scaler: StandardScaler | None = None


@dataclass
class SegmentMatrix:
    values: np.ndarray
    end_feature_rows: np.ndarray


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    *,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    standardize: bool = False,
) -> FeatureMatrix:
    feature_frame = build_feature_frame(df, feature_config)
    missing = [column for column in feature_columns if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Requested feature columns are unavailable: {missing}")

    feature_frame = feature_frame.dropna(subset=feature_columns).reset_index(drop=True)
    if feature_frame.empty:
        raise ValueError("No rows remain after building and dropping feature NaNs")

    values = feature_frame[feature_columns].to_numpy(dtype=float)
    fitted_scaler = scaler

    if standardize:
        if fit_scaler:
            fitted_scaler = StandardScaler()
            values = fitted_scaler.fit_transform(values)
        elif scaler is not None:
            values = scaler.transform(values)
        else:
            raise ValueError("standardize=True requires fit_scaler=True or a scaler")

    return FeatureMatrix(
        values=values,
        frame=feature_frame,
        columns=feature_columns,
        scaler=fitted_scaler,
    )


def build_feature_frame(df: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    frame = df.copy().sort_values("date").reset_index(drop=True)
    frame["_price_row"] = np.arange(len(frame))

    frame["log_ret"] = np.log(frame["adjClose"]).diff()
    frame["abs_ret"] = frame["log_ret"].abs()

    for window in _as_int_list(feature_config.get("vol_windows", [5])):
        frame[f"rvol_{window}d"] = frame["log_ret"].rolling(window).std()

    volume_window = int(feature_config.get("volume_zscore_window", 20))
    frame["vol_change"] = frame["adjVolume"].pct_change()
    volume_mean = frame["adjVolume"].rolling(volume_window).mean()
    volume_std = frame["adjVolume"].rolling(volume_window).std()
    frame["vol_zscore"] = (frame["adjVolume"] - volume_mean) / volume_std
    frame["vol_spike"] = (frame["vol_zscore"] > 2.0).astype(float)

    for window in _as_int_list(feature_config.get("momentum_windows", [5, 20])):
        frame[f"momentum_{window}d"] = frame["adjClose"].pct_change(window)

    if bool(feature_config.get("include_hl_range", True)):
        high_col, low_col = _range_columns(frame)
        if high_col and low_col:
            frame["hl_range"] = (frame[high_col] - frame[low_col]) / frame["adjClose"]

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_segments(features: np.ndarray, h1: int, h2: int) -> SegmentMatrix:
    if h1 <= h2:
        raise ValueError("h1 must be greater than h2")

    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    step = h1 - h2
    n_segments = (len(values) - h1) // step + 1
    if n_segments <= 0:
        raise ValueError("Not enough feature rows to build any segments")

    segments = np.array(
        [values[i * step : i * step + h1] for i in range(n_segments)]
    )
    end_feature_rows = np.array(
        [i * step + h1 - 1 for i in range(n_segments)],
        dtype=int,
    )
    return SegmentMatrix(values=segments, end_feature_rows=end_feature_rows)


def labels_to_price_rows(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    labels: np.ndarray,
) -> pd.Series:
    regime = np.full(len(df), np.nan)
    rows = feature_frame["_price_row"].to_numpy(dtype=int)[: len(labels)]
    regime[rows] = labels[: len(rows)]
    return pd.Series(regime, index=df.index, name="regime")


def segment_labels_to_price_rows(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    labels: np.ndarray,
    end_feature_rows: np.ndarray,
) -> pd.Series:
    regime = np.full(len(df), np.nan)
    usable_ends = end_feature_rows[: len(labels)]
    price_rows = feature_frame.iloc[usable_ends]["_price_row"].to_numpy(dtype=int)
    regime[price_rows] = labels[: len(price_rows)]
    return pd.Series(regime, index=df.index, name="regime")


def _range_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if {"adjHigh", "adjLow"}.issubset(frame.columns):
        return "adjHigh", "adjLow"
    if {"high", "low"}.issubset(frame.columns):
        return "high", "low"
    return None, None


def _as_int_list(values: Any) -> list[int]:
    if isinstance(values, int):
        return [values]
    return [int(value) for value in values]

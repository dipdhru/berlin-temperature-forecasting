"""Chronological splits, scaling, calendar features and sequence windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import METEO_FEATURES, TARGET


@dataclass
class SplitDates:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def chronological_split(
    index: pd.DatetimeIndex, train_ratio: float = 0.70, validation_ratio: float = 0.15
) -> SplitDates:
    n = len(index)
    train_end_idx = int(n * train_ratio) - 1
    val_end_idx = int(n * (train_ratio + validation_ratio)) - 1
    return SplitDates(
        train_end=index[train_end_idx],
        val_end=index[val_end_idx],
        test_start=index[val_end_idx + 1],
        test_end=index[-1],
    )


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    day_of_year = out.index.day_of_year
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return out


@dataclass
class ScaledHistory:
    available: pd.DataFrame
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    features: pd.DataFrame
    target: pd.Series

    def inverse_target(self, y_scaled) -> np.ndarray:
        arr = np.asarray(y_scaled)
        return self.target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)


def scale_history(
    frame: pd.DataFrame,
    fit_until: pd.Timestamp | None = None,
    transform_until: pd.Timestamp | None = None,
) -> ScaledHistory:
    """Scale features and target with scalers fitted on data up to ``fit_until``.

    ``transform_until`` controls which rows are returned (defaults to
    ``fit_until``). Passing a later date lets validation/test rows be
    transformed with scalers that never saw them.
    """
    fit_frame = frame.loc[:fit_until] if fit_until is not None else frame
    end = transform_until if transform_until is not None else fit_until
    available = frame.loc[:end] if end is not None else frame

    feature_scaler = MinMaxScaler().fit(fit_frame[METEO_FEATURES])
    target_scaler = MinMaxScaler().fit(fit_frame[[TARGET]])

    scaled = pd.DataFrame(
        feature_scaler.transform(available[METEO_FEATURES]),
        columns=METEO_FEATURES,
        index=available.index,
    )
    target = pd.Series(
        target_scaler.transform(available[[TARGET]]).ravel(), index=available.index, name=TARGET
    )
    return ScaledHistory(
        available=available,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        features=add_calendar_features(scaled),
        target=target,
    )


@dataclass
class Sequences:
    X: np.ndarray
    y: np.ndarray
    origins: pd.DatetimeIndex
    target_starts: pd.DatetimeIndex
    target_ends: pd.DatetimeIndex


def make_sequences(
    features: pd.DataFrame, target: pd.Series, lookback: int = 30, horizon: int = 7
) -> Sequences:
    """Sliding windows: ``lookback`` days of features -> next ``horizon`` targets.

    The origin of a window is its last input day; targets start the day after.
    """
    values = features.to_numpy(dtype=np.float32)
    target_values = target.to_numpy(dtype=np.float32)
    dates = features.index

    ends = np.arange(lookback, len(features) - horizon + 1)
    if len(ends) == 0:
        raise ValueError(
            f"Need at least lookback + horizon = {lookback + horizon} rows, got {len(features)}"
        )
    X = np.stack([values[e - lookback : e] for e in ends])
    y = np.stack([target_values[e : e + horizon] for e in ends])

    return Sequences(
        X=X,
        y=y,
        origins=dates[ends - 1],
        target_starts=dates[ends],
        target_ends=dates[ends + horizon - 1],
    )


def last_window(features: pd.DataFrame, lookback: int = 30) -> np.ndarray:
    """The most recent input window, shaped ``(1, lookback, n_features)``."""
    return features.iloc[-lookback:].to_numpy(dtype=np.float32)[None, :, :]

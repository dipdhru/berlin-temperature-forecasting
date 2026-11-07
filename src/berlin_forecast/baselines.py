"""Naive reference forecasts every model has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET


def persistence_forecast(frame: pd.DataFrame, origin: pd.Timestamp, horizon: int = 7) -> np.ndarray:
    """Every future day equals the last observed temperature."""
    return np.repeat(frame.loc[origin, TARGET], horizon)


def seasonal_naive_forecast(
    frame: pd.DataFrame, origin: pd.Timestamp, horizon: int = 7, period: int = 7
) -> np.ndarray:
    """Each future day equals the temperature ``period`` days before it."""
    dates = [origin + pd.Timedelta(days=h - period) for h in range(1, horizon + 1)]
    return frame.loc[dates, TARGET].to_numpy()

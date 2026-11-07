"""Rolling-origin evaluation.

At each forecast origin every model is refitted from scratch on data up to and
including that day (scalers too), then forecasts the next ``horizon`` days.
Origins are spaced ``test_origin_step_days`` apart so forecast windows never
overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from . import baselines, models
from .config import TARGET, Config
from .features import last_window, make_sequences, scale_history

ALL_MODELS = ("Persistence", "SeasonalNaive7", "Prophet", "BiLSTM", "Transformer")


def select_origins(
    index: pd.DatetimeIndex, after: pd.Timestamp, until: pd.Timestamp, horizon: int, step: int
) -> pd.DatetimeIndex:
    """Every ``step``-th day after ``after`` that still has a full horizon before ``until``."""
    candidates = index[(index > after) & (index + pd.Timedelta(days=horizon) <= until)]
    return pd.DatetimeIndex(candidates[::step])


def _neural_forecast(frame, origin, config: Config, kind: str) -> np.ndarray:
    history = scale_history(frame, fit_until=origin)
    seq = make_sequences(history.features, history.target, config.lookback, config.horizon)
    n_features = seq.X.shape[-1]

    if kind == "BiLSTM":
        model = models.build_bilstm(
            n_features, config.lookback, config.horizon, **config.best_lstm_config
        )
        epochs = config.lstm_train_epochs
    else:
        model = models.build_transformer(
            n_features, config.lookback, config.horizon, **config.best_transformer_config
        )
        epochs = config.transformer_train_epochs

    models.fit_neural(model, seq.X, seq.y, epochs=epochs)
    pred_scaled = model.predict(last_window(history.features, config.lookback), verbose=0)[0]
    return history.inverse_target(pred_scaled)


def forecast_at_origin(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    config: Config,
    model_names: Iterable[str] = ALL_MODELS,
) -> dict[str, np.ndarray]:
    """Forecast the ``horizon`` days after ``origin`` using only data up to ``origin``."""
    future_dates = [origin + pd.Timedelta(days=h) for h in range(1, config.horizon + 1)]
    out = {}
    for name in model_names:
        if name == "Persistence":
            out[name] = baselines.persistence_forecast(frame, origin, config.horizon)
        elif name == "SeasonalNaive7":
            out[name] = baselines.seasonal_naive_forecast(frame, origin, config.horizon)
        elif name == "Prophet":
            model = models.fit_prophet(frame.loc[:origin], config.best_prophet_config)
            out[name] = models.prophet_predict(model, future_dates)
        elif name in ("BiLSTM", "Transformer"):
            out[name] = _neural_forecast(frame, origin, config, name)
        else:
            raise ValueError(f"Unknown model: {name}")
    return out


def rolling_evaluation(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    config: Config,
    model_names: Iterable[str] = ALL_MODELS,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    """Long table with one row per (origin, horizon, model)."""
    model_names = tuple(model_names)
    records = []
    for i, origin in enumerate(origins, start=1):
        if progress:
            progress(i, len(origins), origin)
        target_dates = [origin + pd.Timedelta(days=h) for h in range(1, config.horizon + 1)]
        actual = frame.loc[target_dates, TARGET].to_numpy()

        for name, pred in forecast_at_origin(frame, origin, config, model_names).items():
            for h, (a, p) in enumerate(zip(actual, pred, strict=True), start=1):
                records.append(
                    {
                        "origin": origin,
                        "target_date": target_dates[h - 1],
                        "horizon": h,
                        "model": name,
                        "actual": a,
                        "prediction": p,
                        "error": p - a,
                        "absolute_error": abs(p - a),
                    }
                )
    return pd.DataFrame(records)

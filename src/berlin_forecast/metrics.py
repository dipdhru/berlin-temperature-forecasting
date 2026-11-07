"""Forecast metrics and the summary tables reported in the thesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0
    )


def mase(y_true, y_pred, reference_series) -> float:
    """MAE scaled by the in-sample MAE of a one-step naive forecast."""
    scale = np.mean(np.abs(np.diff(np.asarray(reference_series))))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def score(y_true, y_pred, mase_reference=None) -> dict:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    result = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
    if mase_reference is not None:
        result["MASE"] = mase(y_true, y_pred, mase_reference)
    return result


def overall_table(predictions: pd.DataFrame, mase_reference=None) -> pd.DataFrame:
    """One row per model, pooled over all origins and horizons, sorted by MAE."""
    rows = [
        {"Model": model, **score(g["actual"], g["prediction"], mase_reference)}
        for model, g in predictions.groupby("model")
    ]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def horizon_table(predictions: pd.DataFrame, mase_reference=None) -> pd.DataFrame:
    rows = [
        {"model": model, "horizon": h, **score(g["actual"], g["prediction"], mase_reference)}
        for (model, h), g in predictions.groupby(["model", "horizon"])
    ]
    return pd.DataFrame(rows)


def origin_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (origin, model), g in predictions.groupby(["origin", "model"]):
        s = score(g["actual"], g["prediction"])
        rows.append({"origin": origin, "model": model, "MAE": s["MAE"], "RMSE": s["RMSE"]})
    return pd.DataFrame(rows)


def bootstrap_ci(
    origin_metrics: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """95% bootstrap interval of the mean per-origin MAE for each model.

    Resampling whole origins (not individual days) respects the fact that the
    seven errors from one forecast are correlated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for model, g in origin_metrics.groupby("model"):
        values = g["MAE"].to_numpy()
        means = np.array(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
        )
        rows.append(
            {
                "Model": model,
                "Mean_origin_MAE": values.mean(),
                "MAE_CI_2.5%": np.percentile(means, 2.5),
                "MAE_CI_97.5%": np.percentile(means, 97.5),
                "N_origins": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean_origin_MAE").reset_index(drop=True)

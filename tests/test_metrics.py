import numpy as np
import pandas as pd
import pytest

from berlin_forecast import metrics
from berlin_forecast.config import RESULTS_DIR


def test_smape_known_value():
    # |2-1| * 2 / (1 + 2) = 2/3 -> 66.67 %
    assert metrics.smape([1.0], [2.0]) == pytest.approx(200 / 3)


def test_smape_is_finite_at_zero():
    assert np.isfinite(metrics.smape([0.0, 0.0], [0.0, 0.0]))


def test_mase_scales_by_naive_error():
    reference = [0, 1, 2, 3]  # one-step naive MAE = 1
    assert metrics.mase([10, 10], [12, 8], reference) == pytest.approx(2.0)


def test_mase_constant_reference_is_nan():
    assert np.isnan(metrics.mase([1], [2], [5, 5, 5]))


@pytest.fixture(scope="module")
def published_predictions() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "rolling_predictions.csv", parse_dates=["origin"])


def test_reproduces_published_overall_metrics(published_predictions):
    """The package recomputes the published table from the saved predictions."""
    published = pd.read_csv(RESULTS_DIR / "overall_metrics.csv").set_index("Model")
    ours = metrics.overall_table(published_predictions).set_index("Model")
    # MASE needs the training series, which is not stored; check everything else.
    cols = ["MAE", "RMSE", "R2", "sMAPE"]
    pd.testing.assert_frame_equal(ours.loc[published.index, cols], published[cols], rtol=1e-9)


def test_reproduces_published_horizon_metrics(published_predictions):
    published = pd.read_csv(RESULTS_DIR / "horizon_metrics.csv").set_index(["model", "horizon"])
    ours = metrics.horizon_table(published_predictions).set_index(["model", "horizon"])
    cols = ["MAE", "RMSE", "R2", "sMAPE"]
    pd.testing.assert_frame_equal(ours.loc[published.index, cols], published[cols], rtol=1e-9)


def test_reproduces_published_bootstrap_intervals(published_predictions):
    published = pd.read_csv(RESULTS_DIR / "origin_bootstrap_ci.csv")
    ours = metrics.bootstrap_ci(metrics.origin_table(published_predictions), seed=42)
    pd.testing.assert_frame_equal(ours, published, rtol=1e-9, check_dtype=False)

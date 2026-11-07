import numpy as np
import pandas as pd
import pytest

from berlin_forecast import baselines, data
from berlin_forecast.config import TARGET
from berlin_forecast.evaluate import select_origins
from berlin_forecast.features import chronological_split, make_sequences, scale_history


def test_split_is_chronological_and_complete(weather):
    split = chronological_split(weather.index)
    assert weather.index[0] < split.train_end < split.val_end < split.test_start <= split.test_end
    assert split.test_end == weather.index[-1]


def test_sequences_align_inputs_and_targets(weather):
    history = scale_history(weather)
    seq = make_sequences(history.features, history.target, lookback=30, horizon=7)

    assert seq.X.shape == (len(weather) - 30 - 7 + 1, 30, history.features.shape[1])
    assert seq.y.shape == (seq.X.shape[0], 7)
    # Targets start the day after the origin, which is the last input day.
    assert (seq.target_starts - seq.origins == pd.Timedelta(days=1)).all()
    np.testing.assert_allclose(seq.X[0, -1], history.features.iloc[29].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(seq.y[0], history.target.iloc[30:37].to_numpy(), rtol=1e-6)


def test_scaler_never_sees_future_data(weather):
    cutoff = weather.index[400]
    spiked = weather.copy()
    spiked.loc[spiked.index > cutoff, TARGET] = 1000.0  # absurd future values

    a = scale_history(weather, fit_until=cutoff)
    b = scale_history(spiked, fit_until=cutoff)
    pd.testing.assert_frame_equal(a.features, b.features)
    pd.testing.assert_series_equal(a.target, b.target)


def test_too_short_history_raises(weather):
    history = scale_history(weather.iloc[:20])
    with pytest.raises(ValueError):
        make_sequences(history.features, history.target, lookback=30, horizon=7)


def test_origins_are_weekly_and_have_full_horizon(weather):
    split = chronological_split(weather.index)
    origins = select_origins(weather.index, split.val_end, split.test_end, horizon=7, step=7)
    assert (np.diff(origins.values).astype("timedelta64[D]").astype(int) == 7).all()
    assert origins[0] > split.val_end
    assert (origins + pd.Timedelta(days=7) <= split.test_end).all()


def test_baselines(weather):
    origin = weather.index[100]
    assert (baselines.persistence_forecast(weather, origin, 7) == weather.loc[origin, TARGET]).all()
    seasonal = baselines.seasonal_naive_forecast(weather, origin, 7)
    np.testing.assert_array_equal(seasonal, weather[TARGET].iloc[94:101].to_numpy())


def test_clean_drops_trailing_gaps_and_never_backfills(weather):
    frame = weather.copy()
    frame.iloc[0, 0] = np.nan  # leading gap -> row dropped
    frame.iloc[50, 1] = np.nan  # interior gap -> forward-filled
    frame.iloc[-3:, frame.columns.get_loc(TARGET)] = np.nan  # not yet in the archive
    cleaned = data.clean(frame)

    assert cleaned.index[0] == weather.index[1]
    assert cleaned.index[-1] == weather.index[-4]
    assert cleaned.iloc[49, 1] == weather.iloc[49, 1]
    assert not cleaned.isna().any().any()


def test_parse_open_meteo_payload():
    payload = {
        "daily": {
            "time": ["2026-01-02", "2026-01-01"],
            **{var: [2.0, 1.0] for var in data.DAILY_VARIABLES},
        }
    }
    frame = data.parse_daily_payload(payload)
    assert list(frame.columns) == list(data.DAILY_VARIABLES.values())
    assert frame.index.is_monotonic_increasing
    assert frame.loc["2026-01-01", TARGET] == 1.0

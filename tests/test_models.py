import numpy as np
import pytest

from berlin_forecast import models
from berlin_forecast.config import Config
from berlin_forecast.evaluate import forecast_at_origin, rolling_evaluation

pytestmark = pytest.mark.slow


def test_networks_output_one_value_per_horizon_day():
    X = np.random.default_rng(0).random((4, 30, 12), dtype=np.float32)
    for build in (models.build_bilstm, models.build_transformer):
        assert build(12).predict(X, verbose=0).shape == (4, 7)


def test_transformer_rejects_bad_head_count():
    with pytest.raises(ValueError):
        models.build_transformer(12, model_dim=64, num_heads=3)


def test_forecast_at_origin_end_to_end(weather):
    config = Config(best_lstm_epochs=1, best_transformer_epochs=1, min_train_epochs=1)
    config.best_lstm_config.update(lstm1_units=8, lstm2_units=4)
    preds = forecast_at_origin(weather, weather.index[-10], config)
    assert set(preds) == {"Persistence", "SeasonalNaive7", "Prophet", "BiLSTM", "Transformer"}
    for name, p in preds.items():
        assert p.shape == (7,) and np.isfinite(p).all(), name


def test_rolling_evaluation_table(weather):
    origins = weather.index[[-30, -20]]
    table = rolling_evaluation(weather, origins, Config(), ["Persistence", "SeasonalNaive7"])
    assert len(table) == 2 * 7 * 2
    assert (table["absolute_error"] == (table["prediction"] - table["actual"]).abs()).all()

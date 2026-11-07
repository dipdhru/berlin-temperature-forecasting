"""Hyperparameter search on the validation period.

Neural models: Keras Tuner random search, 8 trials each, early stopping.
Prophet: 12 configurations sampled from a grid, scored on three validation
origins. Selected settings are written back into the :class:`Config`.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .config import DEFAULT_LSTM_CONFIG, DEFAULT_TRANSFORMER_CONFIG, TARGET, Config
from .features import SplitDates, make_sequences, scale_history
from .metrics import score

PROPHET_PARAM_SPACE = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 5.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
    "changepoint_range": [0.8, 0.9],
}


def train_val_sequences(frame: pd.DataFrame, split: SplitDates, config: Config):
    """Windows whose targets fall entirely in train / validation.

    Scalers are fitted on the training period only.
    """
    history = scale_history(frame, fit_until=split.train_end, transform_until=split.val_end)
    seq = make_sequences(history.features, history.target, config.lookback, config.horizon)
    train = seq.target_ends <= split.train_end
    val = (seq.target_starts > split.train_end) & (seq.target_ends <= split.val_end)
    return seq.X[train], seq.y[train], seq.X[val], seq.y[val], history


def _search(builder, X_train, y_train, X_val, y_val, max_trials, max_epochs, directory, name):
    import keras_tuner as kt
    from tensorflow.keras.callbacks import EarlyStopping

    tuner = kt.RandomSearch(
        builder,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=True,
        directory=str(directory),
        project_name=name,
    )
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0].values
    try:
        best_epochs = int(tuner.oracle.get_best_trials(1)[0].metrics.get_best_step("val_loss")) + 1
    except Exception:
        best_epochs = max_epochs
    return best_hp, best_epochs


def tune_neural(
    frame: pd.DataFrame,
    split: SplitDates,
    config: Config,
    max_trials: int = 8,
    max_epochs: int = 30,
    directory: str | Path = "tuner_results",
) -> pd.DataFrame:
    """Tune both networks and return the default-vs-tuned validation table."""
    X_tr, y_tr, X_val, y_val, history = train_val_sequences(frame, split, config)
    n_features = X_tr.shape[-1]
    lb, hz = config.lookback, config.horizon

    def lstm_builder(hp):
        return models.build_bilstm(
            n_features,
            lb,
            hz,
            lstm1_units=hp.Choice("lstm1_units", [32, 64, 128]),
            lstm2_units=hp.Choice("lstm2_units", [16, 32, 64]),
            dropout_rate=hp.Choice("dropout_rate", [0.1, 0.2, 0.3]),
            dense_units=hp.Choice("dense_units", [16, 32, 64]),
            learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4]),
        )

    def transformer_builder(hp):
        return models.build_transformer(
            n_features,
            lb,
            hz,
            model_dim=hp.Choice("model_dim", [32, 64, 96]),
            num_heads=hp.Choice("num_heads", [2, 4, 8]),
            ff_dim=hp.Choice("ff_dim", [64, 128, 192]),
            num_blocks=hp.Choice("num_blocks", [1, 2, 3]),
            dropout_rate=hp.Choice("dropout_rate", [0.05, 0.1, 0.2]),
            dense_units=hp.Choice("dense_units", [32, 64, 96]),
            learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4]),
        )

    config.best_lstm_config, config.best_lstm_epochs = _search(
        lstm_builder, X_tr, y_tr, X_val, y_val, max_trials, max_epochs, directory, "bilstm_7day"
    )
    config.best_transformer_config, config.best_transformer_epochs = _search(
        transformer_builder,
        X_tr,
        y_tr,
        X_val,
        y_val,
        max_trials,
        max_epochs,
        directory,
        "transformer_7day",
    )

    y_val_actual = history.inverse_target(y_val)
    rows = []
    for label, build, params, epochs in [
        ("Default BiLSTM", models.build_bilstm, DEFAULT_LSTM_CONFIG, max_epochs),
        ("Tuned BiLSTM", models.build_bilstm, config.best_lstm_config, config.lstm_train_epochs),
        ("Default Transformer", models.build_transformer, DEFAULT_TRANSFORMER_CONFIG, max_epochs),
        (
            "Tuned Transformer",
            models.build_transformer,
            config.best_transformer_config,
            config.transformer_train_epochs,
        ),
    ]:
        model = build(n_features, lb, hz, **params)
        models.fit_neural(model, X_tr, y_tr, epochs=epochs, X_val=X_val, y_val=y_val)
        pred = history.inverse_target(model.predict(X_val, verbose=0))
        rows.append({"Model": label, **score(y_val_actual, pred)})
    return pd.DataFrame(rows)


def sample_prophet_configs(n: int = 12, seed: int = 42) -> list[dict]:
    grid = [
        dict(zip(PROPHET_PARAM_SPACE, values, strict=True))
        for values in itertools.product(*PROPHET_PARAM_SPACE.values())
    ]
    if n >= len(grid):
        return grid
    chosen = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[i] for i in chosen]


def tune_prophet(frame: pd.DataFrame, split: SplitDates, config: Config, n_configs: int = 12):
    candidates = frame.index[
        (frame.index > split.train_end)
        & (frame.index + pd.Timedelta(days=config.horizon) <= split.val_end)
    ]
    idxs = np.unique(np.linspace(0, len(candidates) - 1, min(3, len(candidates)), dtype=int))
    origins = candidates[idxs]

    rows = []
    for params in sample_prophet_configs(n_configs, config.seed):
        errors = []
        for origin in origins:
            dates = [origin + pd.Timedelta(days=h) for h in range(1, config.horizon + 1)]
            model = models.fit_prophet(frame.loc[:origin], params)
            pred = models.prophet_predict(model, dates)
            errors.append(score(frame.loc[dates, TARGET], pred)["MAE"])
        rows.append({**params, "validation_MAE": float(np.mean(errors))})

    results = pd.DataFrame(rows).sort_values("validation_MAE").reset_index(drop=True)
    config.best_prophet_config = {k: results.loc[0, k] for k in PROPHET_PARAM_SPACE}
    config.best_prophet_config = {
        k: (v.item() if hasattr(v, "item") else v) for k, v in config.best_prophet_config.items()
    }
    return results

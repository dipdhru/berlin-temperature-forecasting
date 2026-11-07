"""Model definitions: Bidirectional LSTM, Transformer encoder and Prophet.

Both neural networks are *direct* multi-step forecasters: one forward pass
outputs all ``horizon`` future values, so errors are never fed back in.
TensorFlow and Prophet are imported lazily because they are slow to load.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pandas as pd

from .config import TARGET


def set_seed(seed: int = 42) -> None:
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # not available on every backend
        pass


def _compile(model, learning_rate: float):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_bilstm(
    n_features: int,
    lookback: int = 30,
    horizon: int = 7,
    lstm1_units: int = 64,
    lstm2_units: int = 32,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input

    inputs = Input(shape=(lookback, n_features))
    x = Bidirectional(LSTM(lstm1_units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm2_units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(horizon)(x)
    return _compile(Model(inputs, outputs, name="bilstm"), learning_rate)


def positional_encoding(length: int, depth: int) -> np.ndarray:
    """Sinusoidal position encoding, shaped ``(1, length, depth)``."""
    positions = np.arange(length)[:, None]
    dimensions = np.arange(depth)[None, :]
    angle_rads = positions / np.power(10000, (2 * (dimensions // 2)) / np.float32(depth))

    encoding = np.zeros((length, depth), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return encoding[None, :, :]


def _transformer_block(x, model_dim, num_heads, ff_dim, dropout_rate):
    from tensorflow.keras.layers import Add, Dense, Dropout, LayerNormalization, MultiHeadAttention

    # Pre-norm self-attention
    attn_input = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=model_dim // num_heads, dropout=dropout_rate
    )(attn_input, attn_input)
    x = Add()([x, Dropout(dropout_rate)(attn)])

    # Pre-norm feed-forward
    ff = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation="gelu")(ff)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(model_dim)(ff)
    return Add()([x, Dropout(dropout_rate)(ff)])


def build_transformer(
    n_features: int,
    lookback: int = 30,
    horizon: int = 7,
    model_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.1,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
):
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
    )

    if model_dim % num_heads:
        raise ValueError("model_dim must be divisible by num_heads")

    inputs = Input(shape=(lookback, n_features))
    x = Dense(model_dim)(inputs)
    x = x + tf.constant(positional_encoding(lookback, model_dim))
    x = Dropout(dropout_rate)(x)
    for _ in range(num_blocks):
        x = _transformer_block(x, model_dim, num_heads, ff_dim, dropout_rate)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(horizon)(x)
    return _compile(Model(inputs, outputs, name="transformer"), learning_rate)


def fit_neural(model, X, y, epochs: int, X_val=None, y_val=None, patience: int = 6):
    """Train in time order. With validation data, use early stopping."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    kwargs = {}
    if X_val is not None:
        kwargs["validation_data"] = (X_val, y_val)
        kwargs["callbacks"] = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
    model.fit(X, y, epochs=epochs, batch_size=32, shuffle=False, verbose=0, **kwargs)
    return model


def fit_prophet(train_frame: pd.DataFrame, params: dict):
    from prophet import Prophet

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, **params
    )
    model.fit(train_frame[[TARGET]].rename_axis("ds").reset_index().rename(columns={TARGET: "y"}))
    return model


def prophet_predict(model, dates) -> np.ndarray:
    return model.predict(pd.DataFrame({"ds": pd.to_datetime(list(dates))}))["yhat"].to_numpy()

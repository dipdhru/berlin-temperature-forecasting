"""Experiment configuration.

Defaults reproduce the final thesis run (V2.1). The tuned hyperparameters are
read from ``results/config.json`` so the pipeline and the app use exactly the
settings that produced the reported results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DATA_DIR = PROJECT_ROOT / "data"

METEO_FEATURES = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "precipitation",
    "wind_speed_max",
    "radiation",
    "evapotranspiration",
    "humidity_max",
    "humidity_min",
    "humidity_mean",
]
TARGET = "temp_mean"

DEFAULT_LSTM_CONFIG = {
    "lstm1_units": 64,
    "lstm2_units": 32,
    "dropout_rate": 0.2,
    "dense_units": 32,
    "learning_rate": 1e-3,
}

DEFAULT_TRANSFORMER_CONFIG = {
    "model_dim": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "num_blocks": 2,
    "dropout_rate": 0.1,
    "dense_units": 64,
    "learning_rate": 1e-3,
}

DEFAULT_PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "changepoint_range": 0.8,
}


@dataclass
class Config:
    seed: int = 42
    city: str = "Berlin"
    latitude: float = 52.52
    longitude: float = 13.405
    start_date: str = "2021-01-01"
    end_date: str = "2026-08-31"
    lookback: int = 30
    horizon: int = 7
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    test_origin_step_days: int = 7
    best_lstm_config: dict = field(default_factory=lambda: dict(DEFAULT_LSTM_CONFIG))
    best_lstm_epochs: int = 20
    best_transformer_config: dict = field(default_factory=lambda: dict(DEFAULT_TRANSFORMER_CONFIG))
    best_transformer_epochs: int = 20
    best_prophet_config: dict = field(default_factory=lambda: dict(DEFAULT_PROPHET_CONFIG))
    # The thesis run never trains the networks for fewer than 10 epochs.
    min_train_epochs: int = 10

    @property
    def lstm_train_epochs(self) -> int:
        return max(self.best_lstm_epochs, self.min_train_epochs)

    @property
    def transformer_train_epochs(self) -> int:
        return max(self.best_transformer_epochs, self.min_train_epochs)

    @classmethod
    def from_json(cls, path: str | Path = RESULTS_DIR / "config.json") -> Config:
        raw = json.loads(Path(path).read_text())
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

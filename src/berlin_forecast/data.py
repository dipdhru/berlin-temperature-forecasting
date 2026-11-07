"""Download and clean daily Berlin weather from the Open-Meteo archive API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import METEO_FEATURES, TARGET

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo variable name -> column name used throughout the project.
DAILY_VARIABLES = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precipitation",
    "windspeed_10m_max": "wind_speed_max",
    "shortwave_radiation_sum": "radiation",
    "et0_fao_evapotranspiration": "evapotranspiration",
    "relative_humidity_2m_max": "humidity_max",
    "relative_humidity_2m_min": "humidity_min",
    "relative_humidity_2m_mean": "humidity_mean",
}


def parse_daily_payload(payload: dict) -> pd.DataFrame:
    daily = payload["daily"]
    frame = pd.DataFrame(
        {column: daily[variable] for variable, column in DAILY_VARIABLES.items()},
        index=pd.to_datetime(daily["time"]),
    )
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated()].sort_index()
    return frame.astype(float)


def fetch_history(
    start_date: str,
    end_date: str,
    latitude: float = 52.52,
    longitude: float = 13.405,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Daily history for ``start_date``..``end_date`` (inclusive).

    With ``cache_dir`` set, the raw download is stored as CSV and reused, which
    keeps reruns reproducible even if the API later revises its reanalysis.
    """
    cache_file = None
    if cache_dir is not None:
        cache_file = (
            Path(cache_dir) / f"open_meteo_{latitude}_{longitude}_{start_date}_{end_date}.csv"
        )
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col="date", parse_dates=True)

    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": list(DAILY_VARIABLES),
            "timezone": "Europe/Berlin",
        },
        timeout=60,
    )
    response.raise_for_status()
    frame = parse_daily_payload(response.json())

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_file)
    return frame


def fetch_open_meteo_forecast(
    start_date: str, end_date: str, latitude: float = 52.52, longitude: float = 13.405
) -> pd.Series:
    """Open-Meteo's numerical weather prediction of daily mean temperature.

    A physics-based professional forecast, used in the app as a reference point.
    """
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_mean",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Berlin",
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    return pd.Series(
        daily["temperature_2m_mean"], index=pd.to_datetime(daily["time"]), name="Open-Meteo"
    )


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe cleaning.

    Trailing days without a target (recent days not yet in the archive) are
    dropped rather than filled. Remaining gaps are forward-filled only, so no
    value is ever filled from the future, and leading gaps are dropped.
    """
    assert frame.index.is_monotonic_increasing and frame.index.is_unique

    last_observed = frame[TARGET].last_valid_index()
    out = frame.loc[:last_observed, METEO_FEATURES].ffill().dropna()
    return out

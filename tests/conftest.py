import numpy as np
import pandas as pd
import pytest

from berlin_forecast.config import METEO_FEATURES


@pytest.fixture
def weather() -> pd.DataFrame:
    """Two years of synthetic daily weather with a seasonal temperature cycle."""
    index = pd.date_range("2022-01-01", periods=730, freq="D", name="date")
    rng = np.random.default_rng(0)
    seasonal = 10 + 9 * np.sin(2 * np.pi * (index.day_of_year - 110) / 365.25)
    frame = pd.DataFrame(
        {col: seasonal + rng.normal(0, 1, len(index)) for col in METEO_FEATURES}, index=index
    )
    frame["temp_mean"] = seasonal + rng.normal(0, 2, len(index))
    return frame

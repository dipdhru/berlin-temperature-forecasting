"""Draw the README figures from the saved published results (no training needed)."""

from __future__ import annotations

import pandas as pd

from berlin_forecast import plots
from berlin_forecast.config import FIGURES_DIR, RESULTS_DIR

EXAMPLE_ORIGIN = "2026-01-17"


def main() -> None:
    horizon = pd.read_csv(RESULTS_DIR / "horizon_metrics.csv")
    ci = pd.read_csv(RESULTS_DIR / "origin_bootstrap_ci.csv")
    predictions = pd.read_csv(RESULTS_DIR / "rolling_predictions.csv", parse_dates=["origin"])

    for path in (
        plots.plot_horizon_mae(horizon, FIGURES_DIR / "horizon_mae.png"),
        plots.plot_overall_ci(ci, FIGURES_DIR / "overall_mae_ci.png"),
        plots.plot_example_forecast(
            predictions, EXAMPLE_ORIGIN, FIGURES_DIR / "example_forecast.png"
        ),
    ):
        print("wrote", path)


if __name__ == "__main__":
    main()

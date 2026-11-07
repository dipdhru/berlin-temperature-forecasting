"""Run the full experiment: download -> (tune) -> rolling evaluation -> tables -> figures.

Examples
--------
Reproduce the thesis evaluation with the saved hyperparameters (slow on CPU):

    python scripts/run_pipeline.py

Quick smoke run on 3 forecast origins:

    python scripts/run_pipeline.py --max-origins 3

Re-tune everything from scratch:

    python scripts/run_pipeline.py --tune

Results go to ``runs/<timestamp>/`` so the published thesis results in
``results/`` are never overwritten by accident.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from berlin_forecast import data, evaluate, metrics, models, plots, tuning
from berlin_forecast.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, TARGET, Config
from berlin_forecast.evaluate import ALL_MODELS
from berlin_forecast.features import chronological_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        type=Path,
        default=RESULTS_DIR / "config.json",
        help="config with tuned hyperparameters (default: the thesis run)",
    )
    p.add_argument("--tune", action="store_true", help="re-run hyperparameter search first")
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="evaluate only the first N test origins (for a quick check)",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="output folder (default: runs/<timestamp>)"
    )
    p.add_argument("--no-cache", action="store_true", help="always re-download the weather data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_json(args.config) if args.config.exists() else Config()
    out = args.output or PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    models.set_seed(config.seed)

    print(f"Downloading {config.city} weather {config.start_date} .. {config.end_date}")
    raw = data.fetch_history(
        config.start_date,
        config.end_date,
        config.latitude,
        config.longitude,
        cache_dir=None if args.no_cache else DATA_DIR,
    )
    frame = data.clean(raw)
    split = chronological_split(frame.index, config.train_ratio, config.validation_ratio)
    print(
        f"{len(frame)} days | train to {split.train_end:%Y-%m-%d} | "
        f"validation to {split.val_end:%Y-%m-%d} | test {split.test_start:%Y-%m-%d} .. "
        f"{split.test_end:%Y-%m-%d}"
    )

    if args.tune:
        print("Tuning BiLSTM and Transformer (Keras Tuner) ...")
        tuning.tune_neural(frame, split, config, directory=out / "tuner").to_csv(
            out / "tuning_comparison.csv", index=False
        )
        print("Tuning Prophet ...")
        tuning.tune_prophet(frame, split, config).to_csv(out / "prophet_tuning.csv", index=False)

    origins = evaluate.select_origins(
        frame.index, split.val_end, split.test_end, config.horizon, config.test_origin_step_days
    )
    if args.max_origins:
        origins = origins[: args.max_origins]
    print(f"Rolling evaluation on {len(origins)} origins, models: {', '.join(args.models)}")

    started = time.time()
    predictions = evaluate.rolling_evaluation(
        frame,
        origins,
        config,
        args.models,
        progress=lambda i, n, o: print(
            f"  [{i}/{n}] origin {o:%Y-%m-%d}  ({time.time() - started:.0f}s)"
        ),
    )

    mase_reference = frame.loc[: split.val_end, TARGET].to_numpy()
    overall = metrics.overall_table(predictions, mase_reference)
    by_horizon = metrics.horizon_table(predictions, mase_reference)
    by_origin = metrics.origin_table(predictions)
    ci = metrics.bootstrap_ci(by_origin, seed=config.seed)

    predictions.to_csv(out / "rolling_predictions.csv", index=False)
    overall.to_csv(out / "overall_metrics.csv", index=False)
    by_horizon.to_csv(out / "horizon_metrics.csv", index=False)
    by_origin.to_csv(out / "origin_metrics.csv", index=False)
    ci.to_csv(out / "origin_bootstrap_ci.csv", index=False)

    saved = config.to_dict() | {
        "rolling_origins": [f"{o:%Y-%m-%d}" for o in origins],
        "evaluation_protocol": "weekly_non_overlapping_origins",
    }
    (out / "config.json").write_text(json.dumps(saved, indent=2, default=str))

    plots.plot_horizon_mae(by_horizon, out / "horizon_mae.png")
    plots.plot_overall_ci(ci, out / "overall_mae_ci.png")

    print("\n" + overall.round(3).to_string(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()

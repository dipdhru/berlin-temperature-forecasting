"""Static result figures for the README.

Learned models get the three colour slots (validated for colour-vision
deficiency); naive baselines are drawn as grey reference lines so the eye
compares the models against them rather than treating them as equals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXIS = "#c3c2b7"

STYLE = {
    "BiLSTM": {"color": "#2a78d6", "linestyle": "-", "label": "BiLSTM"},
    "Prophet": {"color": "#eb6834", "linestyle": "-", "label": "Prophet"},
    "Transformer": {"color": "#1baf7a", "linestyle": "-", "label": "Transformer"},
    "Persistence": {"color": "#8a8983", "linestyle": "--", "label": "Persistence"},
    "SeasonalNaive7": {"color": "#8a8983", "linestyle": ":", "label": "Seasonal naive"},
    "Open-Meteo": {"color": "#6250d6", "linestyle": "-", "label": "Open-Meteo NWP"},
    "Actual": {"color": TEXT, "linestyle": "-", "label": "Actual"},
}
MODEL_ORDER = ["BiLSTM", "Prophet", "Transformer", "Persistence", "SeasonalNaive7"]


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def _new_figure(size=(9, 5)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_horizon_mae(horizon_metrics: pd.DataFrame, path: str | Path) -> Path:
    """MAE by forecast day, with a direct label at the end of each line."""
    fig, ax = _new_figure()
    pivot = horizon_metrics.pivot(index="horizon", columns="model", values="MAE")

    label_y = {}
    for model in [m for m in MODEL_ORDER if m in pivot]:
        s = STYLE[model]
        ax.plot(
            pivot.index,
            pivot[model],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2,
            marker="o",
            markersize=5,
            label=s["label"],
        )
        label_y[model] = pivot[model].iloc[-1]

    # Push end labels apart (bottom-up) so close lines never overlap.
    min_gap = 0.14
    previous = None
    for model, y in sorted(label_y.items(), key=lambda kv: kv[1]):
        if previous is not None and y - previous < min_gap:
            y = previous + min_gap
        previous = y
        ax.annotate(
            STYLE[model]["label"],
            xy=(pivot.index[-1], y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
            annotation_clip=False,
        )

    _style_axes(ax, "Forecast error grows with lead time", "Forecast day", "MAE (°C)")
    ax.set_xticks(pivot.index)
    ax.set_xlim(pivot.index.min() - 0.2, pivot.index.max() + 0.2)
    ax.legend(frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_overall_ci(bootstrap_ci: pd.DataFrame, path: str | Path) -> Path:
    """Mean per-origin MAE with its 95% bootstrap interval, best at the top."""
    fig, ax = _new_figure((9, 4))
    ci = bootstrap_ci.sort_values("Mean_origin_MAE", ascending=False).reset_index(drop=True)

    for i, row in ci.iterrows():
        color = STYLE.get(row["Model"], {}).get("color", TEXT_SECONDARY)
        ax.plot(
            [row["MAE_CI_2.5%"], row["MAE_CI_97.5%"]],
            [i, i],
            color=color,
            linewidth=2,
            solid_capstyle="round",
        )
        ax.plot(
            row["Mean_origin_MAE"],
            i,
            "o",
            color=color,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"{row['Mean_origin_MAE']:.2f} °C",
            xy=(row["MAE_CI_97.5%"], i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    ax.set_yticks(range(len(ci)))
    ax.set_yticklabels([STYLE.get(m, {}).get("label", m) for m in ci["Model"]], color=TEXT)
    _style_axes(ax, "Overall error with 95% bootstrap interval", "Mean MAE per forecast (°C)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    xmax = ci["MAE_CI_97.5%"].max()
    ax.set_xlim(ci["MAE_CI_2.5%"].min() - 0.3, xmax + 0.6)
    fig.tight_layout()
    return _save(fig, path)


def plot_example_forecast(predictions: pd.DataFrame, origin, path: str | Path) -> Path:
    fig, ax = _new_figure()
    example = predictions[predictions["origin"] == pd.Timestamp(origin)]
    first = example[example["model"] == example["model"].iloc[0]]
    ax.plot(
        pd.to_datetime(first["target_date"]),
        first["actual"],
        color=TEXT,
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Actual",
        zorder=5,
    )
    for model in [m for m in MODEL_ORDER if m in set(example["model"])]:
        g = example[example["model"] == model]
        s = STYLE[model]
        ax.plot(
            pd.to_datetime(g["target_date"]),
            g["prediction"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2,
            marker="o",
            markersize=5,
            label=s["label"],
        )
    _style_axes(
        ax,
        f"Example 7-day forecast from {pd.Timestamp(origin):%d %b %Y}",
        "",
        "Daily mean temperature (°C)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path

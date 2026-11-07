"""Live 7-day Berlin temperature forecast from the thesis models.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets Streamlit Cloud run without installing the package

from berlin_forecast import data, evaluate, models  # noqa: E402
from berlin_forecast.config import RESULTS_DIR, TARGET, Config  # noqa: E402
from berlin_forecast.plots import STYLE  # noqa: E402

REPO_URL = "https://github.com/dipdhru/berlin-temperature-forecasting"
HISTORY_YEARS = 5

st.set_page_config(page_title="Berlin 7-day temperature forecast", page_icon="🌡️", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_history(end: date) -> pd.DataFrame:
    start = end - timedelta(days=365 * HISTORY_YEARS)
    return data.clean(data.fetch_history(start.isoformat(), end.isoformat()))


@st.cache_resource(ttl=24 * 3600, show_spinner=False, max_entries=4)
def run_forecast(origin: pd.Timestamp, model_names: tuple[str, ...]) -> pd.DataFrame:
    config = Config.from_json(RESULTS_DIR / "config.json")
    models.set_seed(config.seed)
    frame = load_history(origin.date())
    preds = evaluate.forecast_at_origin(frame, origin, config, model_names)
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=config.horizon, freq="D")
    return pd.DataFrame(preds, index=dates)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nwp(start: date, end: date) -> pd.Series | None:
    try:
        return data.fetch_open_meteo_forecast(start.isoformat(), end.isoformat())
    except Exception:
        return None


# --- Header -----------------------------------------------------------------
st.title("🌡️ Berlin 7-day temperature forecast")
st.markdown(
    "Daily mean temperature for the next week from the models compared in my MSc thesis: "
    "a **Bidirectional LSTM**, **Prophet**, and an optional **Transformer**, retrained on the "
    f"last {HISTORY_YEARS} years of Open-Meteo data. "
    f"[Code and thesis on GitHub]({REPO_URL})"
)

with st.sidebar:
    st.header("Settings")
    include_transformer = st.checkbox("Include Transformer (slower)", value=False)
    show_nwp = st.checkbox("Compare with Open-Meteo's weather model", value=True)
    st.caption(
        "Models are trained once per day and cached. The first load of the day takes "
        "about a minute while the networks train."
    )

origin = pd.Timestamp(date.today() - timedelta(days=1))
model_names = ("Persistence", "Prophet", "BiLSTM") + (
    ("Transformer",) if include_transformer else ()
)

try:
    with st.spinner("Downloading weather history ..."):
        history = load_history(origin.date())
except Exception as exc:  # network or API problem
    st.error(f"Could not download weather data from Open-Meteo: {exc}")
    st.stop()

origin = history.index[-1]  # last complete day actually available
with st.spinner("Training models on the latest data (first run of the day takes ~1 min) ..."):
    forecast = run_forecast(origin, model_names)

nwp = load_nwp(forecast.index[0].date(), forecast.index[-1].date()) if show_nwp else None
if nwp is not None:
    forecast["Open-Meteo"] = nwp.reindex(forecast.index)

# --- Headline numbers ---------------------------------------------------------
tomorrow = forecast.index[0]
c1, c2, c3 = st.columns(3)
c1.metric(f"Last observed ({origin:%a %d %b})", f"{history[TARGET].iloc[-1]:.1f} °C")
c2.metric(
    f"BiLSTM for {tomorrow:%a %d %b}",
    f"{forecast['BiLSTM'].iloc[0]:.1f} °C",
    f"{forecast['BiLSTM'].iloc[0] - history[TARGET].iloc[-1]:+.1f} °C",
    delta_color="off",
)
c3.metric("BiLSTM 7-day average", f"{forecast['BiLSTM'].mean():.1f} °C")

# --- Chart ------------------------------------------------------------------
recent = history[TARGET].iloc[-30:].rename("Observed").to_frame()
long = pd.concat(
    [
        recent.reset_index(names="date").melt("date", var_name="series", value_name="temp"),
        forecast.reset_index(names="date").melt("date", var_name="series", value_name="temp"),
    ]
).dropna()
long["series"] = long["series"].map(lambda s: STYLE.get(s, {}).get("label", s))

order = ["Observed"] + [STYLE[m]["label"] for m in forecast.columns]
colors = ["#0b0b0b"] + [STYLE[m]["color"] for m in forecast.columns]
dashes = [[1, 0]] + [[6, 4] if m == "Persistence" else [1, 0] for m in forecast.columns]

hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
base = alt.Chart(long).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
    y=alt.Y("temp:Q", title="Daily mean temperature (°C)", scale=alt.Scale(zero=False)),
    color=alt.Color(
        "series:N",
        scale=alt.Scale(domain=order, range=colors),
        legend=alt.Legend(title=None, orient="bottom"),
    ),
    strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=order, range=dashes), legend=None),
)
lines = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
# Past days have only observations and future days only forecasts, so show a dash
# for the missing values instead of "NaN".
rule = (
    alt.Chart(long)
    .mark_rule(color="#8a8983", strokeWidth=10)  # wide hit area so the tooltip is easy to reach
    .transform_pivot("series", value="temp", groupby=["date"])
    .transform_calculate(
        **{
            f"t{i}": f"isValid(datum['{s}']) ? format(datum['{s}'], '.1f') + ' °C' : '—'"
            for i, s in enumerate(order)
        }
    )
    .encode(
        x="date:T",
        opacity=alt.condition(hover, alt.value(0.25), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b %Y")]
        + [alt.Tooltip(f"t{i}:N", title=s) for i, s in enumerate(order)],
    )
    .add_params(hover)
)
today_line = (
    alt.Chart(pd.DataFrame({"date": [origin]}))
    .mark_rule(strokeDash=[2, 2], color="#8a8983")
    .encode(x="date:T")
)
st.altair_chart((lines + rule + today_line).properties(height=420), use_container_width=True)

# --- Table ------------------------------------------------------------------
table = forecast.rename(columns=lambda m: STYLE.get(m, {}).get("label", m))
table.index = table.index.strftime("%a %d %b")
st.dataframe(table.style.format("{:.1f} °C"), use_container_width=True)

# --- How much to trust it -----------------------------------------------------
st.subheader("How accurate are these models?")
left, right = st.columns([3, 2])
with left:
    st.image(str(ROOT / "docs" / "figures" / "horizon_mae.png"))
with right:
    st.markdown(
        """
Measured on 44 weekly forecasts (Oct 2025 – Aug 2026), the BiLSTM averaged an error of
**3.1 °C**, Prophet **3.2 °C** and the Transformer **3.4 °C**.

- For **tomorrow**, simply repeating today's temperature (*persistence*) is hard to beat.
  The learned models pay off from day 2 onwards.
- None of these models knows about incoming weather fronts. Professional forecasts such as
  Open-Meteo's use physics-based simulations of the whole atmosphere and are usually much more
  accurate. This app is a demonstration of time-series machine learning, not a weather service.
"""
    )

st.caption(
    f"Data: [Open-Meteo](https://open-meteo.com/) (ERA5 reanalysis), CC BY 4.0. "
    f"Forecast issued from {origin:%d %b %Y}. Built by Dipanshu Singh."
)

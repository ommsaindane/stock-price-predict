from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.feature_engineering import (
    FeatureEngineeringConfig,
    ReturnsConfig,
    compute_returns,
    engineer_features,
)
from src.inference import InferenceRuntime, load_inference_runtime, predict_next_day_price_from_runtime

INFERENCE_CONFIG_PATH = Path("models/inference_config.json")
MODEL_PATH = Path("models/lstm_return_model.keras")
SCALER_PATH = Path("models/feature_scaler_standard.joblib")
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,16}$")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@st.cache_resource
def get_runtime(config_path: str) -> InferenceRuntime:
    return load_inference_runtime(config_path)


@st.cache_data(ttl=1800, show_spinner=False)
def get_historical_enriched_data(
    ticker: str,
    period: str,
    interval: str,
    ma_short_window: int,
    ma_long_window: int,
    ema_span: int,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker} for period {period}.")

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(-1):
            raise KeyError(f"Returned data does not contain ticker {ticker}.")
        raw = raw.xs(ticker, axis=1, level=-1)

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise KeyError(f"Missing required columns from downloaded data: {missing}")

    clean = raw.loc[:, REQUIRED_COLUMNS].copy()
    clean.columns.name = None
    clean.index = pd.to_datetime(clean.index, errors="raise")
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean = clean.sort_index().dropna()

    feature_config = FeatureEngineeringConfig(
        ma_short_window=ma_short_window,
        ma_long_window=ma_long_window,
        ema_span=ema_span,
    )
    returns_config = ReturnsConfig()
    enriched = engineer_features(clean, feature_config)
    enriched = compute_returns(enriched, returns_config)
    return enriched


def validate_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker is required.")
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError("Ticker must match pattern: letters, numbers, dot, hyphen; max 16 chars.")
    return normalized


def artifacts_ready() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not INFERENCE_CONFIG_PATH.exists():
        missing.append(INFERENCE_CONFIG_PATH.as_posix())
    if not MODEL_PATH.exists():
        missing.append(MODEL_PATH.as_posix())
    if not SCALER_PATH.exists():
        missing.append(SCALER_PATH.as_posix())
    return len(missing) == 0, missing


def build_history_chart(df: pd.DataFrame, ticker: str, ma_short: int, ma_long: int, ema_span: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line={"width": 2})
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"MA{ma_short}"],
            mode="lines",
            name=f"MA{ma_short}",
            line={"width": 1.7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"MA{ma_long}"],
            mode="lines",
            name=f"MA{ma_long}",
            line={"width": 1.7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"EMA{ema_span}"],
            mode="lines",
            name=f"EMA{ema_span}",
            line={"width": 1.7, "dash": "dot"},
        )
    )
    figure.update_layout(
        title=f"{ticker} Price and Trend Features",
        margin={"l": 12, "r": 12, "t": 56, "b": 64},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0},
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=420,
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="Stock Prediction UI", layout="wide")
    st.title("Stock Price Prediction")
    st.caption("Predict next-day close from trained return model and visualize recent trend features.")

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "prediction_latency_ms" not in st.session_state:
        st.session_state.prediction_latency_ms = None
    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None
    if "last_prediction_at_utc" not in st.session_state:
        st.session_state.last_prediction_at_utc = None

    ready, missing = artifacts_ready()
    if not ready:
        st.error(
            "Required model artifacts are missing. Run `python main.py` to generate training artifacts first."
        )
        st.write("Missing artifacts:")
        for path in missing:
            st.write(f"- {path}")
        st.stop()

    runtime: InferenceRuntime | None = None
    runtime_error: str | None = None
    try:
        runtime = get_runtime(INFERENCE_CONFIG_PATH.as_posix())
    except Exception as exc:
        runtime_error = str(exc)

    status_col, info_col = st.columns([1, 2])
    with status_col:
        st.subheader("Runtime")
        if runtime is None:
            st.error("Not loaded")
        else:
            st.success("Loaded")
    with info_col:
        if runtime is None:
            st.write(f"Runtime error: {runtime_error}")
        else:
            st.write(
                f"Model window: {runtime.config.window_size} | Features: {len(runtime.config.model_input_features)}"
            )

    if runtime is None:
        st.stop()

    with st.container(border=True):
        st.subheader("Predict")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            ticker_input = st.text_input("Ticker", value=runtime.config.ticker, help="Example: AAPL")
        with c2:
            period = st.selectbox("Chart period", options=["6mo", "1y", "2y", "5y"], index=1)
        with c3:
            run_predict = st.button("Predict now", type="primary", use_container_width=True)

        if run_predict:
            st.session_state.prediction_error = None
            try:
                ticker = validate_ticker(ticker_input)
                start = time.perf_counter()
                prediction = predict_next_day_price_from_runtime(runtime, ticker=ticker)
                st.session_state.prediction = prediction
                st.session_state.prediction_latency_ms = (time.perf_counter() - start) * 1000.0
                st.session_state.last_prediction_at_utc = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                st.session_state.prediction = None
                st.session_state.prediction_latency_ms = None
                st.session_state.last_prediction_at_utc = None
                st.session_state.prediction_error = str(exc)

    if st.session_state.prediction_error is not None:
        st.error(st.session_state.prediction_error)

    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ticker", prediction["ticker"])
        m2.metric("Last Price", f"${prediction['last_price']:.2f}")
        m3.metric("Predicted Return", f"{prediction['predicted_return'] * 100:.3f}%")
        m4.metric("Predicted Next Price", f"${prediction['predicted_next_price']:.2f}")

        st.caption(
            " | ".join(
                [
                    f"Last timestamp: {prediction['last_timestamp']}",
                    f"Latency: {st.session_state.prediction_latency_ms:.1f} ms",
                    f"Predicted at: {st.session_state.last_prediction_at_utc}",
                ]
            )
        )

    chart_ticker = runtime.config.ticker
    try:
        chart_ticker = validate_ticker(ticker_input)
    except Exception:
        pass

    with st.container(border=True):
        st.subheader("Historical Trend Context")
        try:
            chart_df = get_historical_enriched_data(
                ticker=chart_ticker,
                period=period,
                interval=runtime.config.interval,
                ma_short_window=runtime.config.ma_short_window,
                ma_long_window=runtime.config.ma_long_window,
                ema_span=runtime.config.ema_span,
            )
            figure = build_history_chart(
                chart_df,
                chart_ticker,
                runtime.config.ma_short_window,
                runtime.config.ma_long_window,
                runtime.config.ema_span,
            )
            st.plotly_chart(figure, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to render chart: {exc}")

    with st.expander("Diagnostics"):
        st.write("Runtime configuration")
        st.write(
            {
                "window_size": runtime.config.window_size,
                "feature_count": len(runtime.config.model_input_features),
                "features": runtime.config.model_input_features,
                "interval": runtime.config.interval,
                "ma_short_window": runtime.config.ma_short_window,
                "ma_long_window": runtime.config.ma_long_window,
                "ema_span": runtime.config.ema_span,
                "max_abs_return_prediction": runtime.config.max_abs_return_prediction,
                "explosion_ratio_threshold": runtime.config.explosion_ratio_threshold,
            }
        )
        st.write("Latest prediction payload")
        st.write(st.session_state.prediction)


if __name__ == "__main__":
    main()

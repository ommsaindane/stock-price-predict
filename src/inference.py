from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import yfinance as yf

from src.feature_engineering import (
    FeatureEngineeringConfig,
    ReturnsConfig,
    compute_returns,
    engineer_features,
)
from src.inference_config import InferenceConfig, load_inference_config

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True, slots=True)
class InferenceRuntime:
    config: InferenceConfig
    model: Any
    scaler: Any


def run_reloaded_inference(
    X_seq: np.ndarray, config_path: Path | str
) -> np.ndarray:
    runtime = load_inference_runtime(config_path)
    _validate_inference_inputs(X_seq, runtime.config)

    predictions = runtime.model.predict(X_seq, verbose=0).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("Reloaded inference produced NaN or infinite predictions.")
    return predictions.astype(np.float64)


def load_inference_runtime(config_path: Path | str) -> InferenceRuntime:
    config = load_inference_config(config_path)
    model, scaler = _load_model_and_scaler(config)
    return InferenceRuntime(config=config, model=model, scaler=scaler)


def predict_next_day_price(config_path: Path | str, ticker: str | None = None) -> dict:
    runtime = load_inference_runtime(config_path)
    return predict_next_day_price_from_runtime(runtime, ticker=ticker)


def predict_next_day_price_from_runtime(
    runtime: InferenceRuntime, ticker: str | None = None
) -> dict:
    config = runtime.config
    selected_ticker = ticker.strip().upper() if ticker is not None else config.ticker
    if not selected_ticker:
        raise ValueError("ticker must be a non-empty value.")

    latest_data = _fetch_latest_data(config, selected_ticker)
    feature_config = FeatureEngineeringConfig(
        ma_short_window=config.ma_short_window,
        ma_long_window=config.ma_long_window,
        ema_span=config.ema_span,
    )
    returns_config = ReturnsConfig()

    engineered = engineer_features(latest_data, feature_config)
    enriched = compute_returns(engineered, returns_config)
    features = enriched.loc[:, config.model_input_features].to_numpy(dtype=np.float64)

    if features.shape[0] < config.window_size:
        raise ValueError(
            "Not enough inference rows for requested window_size after feature engineering. "
            f"rows={features.shape[0]}, window_size={config.window_size}."
        )

    scaled_features = runtime.scaler.transform(features)
    last_sequence = scaled_features[-config.window_size :]
    _validate_scaled_sequence(last_sequence)

    X_last = last_sequence.reshape(1, config.window_size, len(config.model_input_features))
    predicted_return = float(runtime.model.predict(X_last, verbose=0).reshape(-1)[0])
    if not np.isfinite(predicted_return):
        raise ValueError("Predicted return is NaN or infinite.")

    if abs(predicted_return) > config.max_abs_return_prediction:
        raise ValueError(
            "Predicted return is unrealistically large. "
            f"predicted_return={predicted_return:.6f}, "
            f"max_abs_return_prediction={config.max_abs_return_prediction:.6f}."
        )

    last_price = float(enriched["Close"].iloc[-1])
    next_price = float(last_price * (1.0 + predicted_return))
    ratio = next_price / last_price
    if ratio <= 0 or ratio > config.explosion_ratio_threshold:
        raise ValueError(
            "Predicted next-day price violates explosion checks. "
            f"ratio={ratio:.6f}, threshold={config.explosion_ratio_threshold:.6f}."
        )

    return {
        "ticker": selected_ticker,
        "last_timestamp": str(enriched.index[-1]),
        "last_price": last_price,
        "predicted_return": predicted_return,
        "predicted_next_price": next_price,
        "window_size": config.window_size,
        "feature_count": len(config.model_input_features),
    }


def _validate_inference_inputs(X_seq: np.ndarray, config: InferenceConfig) -> None:
    if X_seq.ndim != 3:
        raise ValueError("X_seq must be 3D with shape (samples, timesteps, features).")

    if X_seq.shape[0] == 0:
        raise ValueError("X_seq must not be empty.")

    if X_seq.shape[1] != config.window_size:
        raise ValueError(
            "X_seq timestep dimension does not match config window_size. "
            f"X_seq={X_seq.shape[1]}, config={config.window_size}."
        )

    expected_features = len(config.model_input_features)
    if X_seq.shape[2] != expected_features:
        raise ValueError(
            "X_seq feature dimension does not match config model_input_features. "
            f"X_seq={X_seq.shape[2]}, config={expected_features}."
        )

    if not np.isfinite(X_seq).all():
        raise ValueError("X_seq contains NaN or infinite values.")


def _load_model_and_scaler(config: InferenceConfig):
    model_file = Path(config.model_path)
    scaler_file = Path(config.scaler_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file.as_posix()}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_file.as_posix()}")

    scaler = joblib.load(scaler_file)
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError("Loaded scaler is not fitted: missing n_features_in_.")
    if int(scaler.n_features_in_) != len(config.model_input_features):
        raise ValueError(
            "Scaler feature count does not match inference config features. "
            f"scaler={int(scaler.n_features_in_)}, config={len(config.model_input_features)}."
        )

    model = load_model(model_file)
    return model, scaler


def _fetch_latest_data(config: InferenceConfig, ticker: str) -> pd.DataFrame:
    required_rows = config.window_size + max(
        config.ma_short_window,
        config.ma_long_window,
        config.ema_span,
    )
    fetch_days = max(required_rows * 3, 180)
    period = f"{fetch_days}d"

    data = yf.download(
        tickers=ticker,
        period=period,
        interval=config.interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise ValueError(
            f"No data returned for ticker {ticker} during inference fetch period {period}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(-1):
            raise KeyError(
                "Inference fetch returned multi-index columns without requested ticker. "
                f"ticker={ticker}."
            )
        data = data.xs(ticker, axis=1, level=-1)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Inference fetch missing required columns: {missing_columns}")

    clean = data.loc[:, REQUIRED_COLUMNS].copy()
    clean.columns.name = None
    clean.index = pd.to_datetime(clean.index, errors="raise")
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean = clean.sort_index().dropna()

    if clean.shape[0] < required_rows:
        raise ValueError(
            "Insufficient rows for inference pipeline after cleaning. "
            f"required={required_rows}, available={clean.shape[0]}."
        )

    return clean


def _validate_scaled_sequence(last_sequence: np.ndarray) -> None:
    if not np.isfinite(last_sequence).all():
        raise ValueError("Scaled inference sequence contains NaN or infinite values.")

    if last_sequence.shape[0] == 0:
        raise ValueError("Scaled inference sequence is empty.")

    spread = np.percentile(last_sequence, 99, axis=0) - np.percentile(last_sequence, 1, axis=0)
    if np.any(np.abs(spread) < 1e-4):
        raise ValueError("Scaled inference sequence appears saturated or collapsed.")

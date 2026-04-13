from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model_path: str
    scaler_path: str
    window_size: int
    model_input_features: list[str]
    target_column: str
    created_at_utc: str
    ticker: str = "AAPL"
    interval: str = "1d"
    ma_short_window: int = 10
    ma_long_window: int = 30
    ema_span: int = 10
    explosion_ratio_threshold: float = 2.0
    max_abs_return_prediction: float = 0.5


def build_inference_config(
    model_path: Path | str,
    scaler_path: Path | str,
    window_size: int,
    model_input_features: list[str],
    target_column: str,
    ticker: str = "AAPL",
    interval: str = "1d",
    ma_short_window: int = 10,
    ma_long_window: int = 30,
    ema_span: int = 10,
    explosion_ratio_threshold: float = 2.0,
    max_abs_return_prediction: float = 0.5,
) -> InferenceConfig:
    config = InferenceConfig(
        model_path=str(Path(model_path).as_posix()),
        scaler_path=str(Path(scaler_path).as_posix()),
        window_size=window_size,
        model_input_features=model_input_features,
        target_column=target_column,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        ticker=ticker,
        interval=interval,
        ma_short_window=ma_short_window,
        ma_long_window=ma_long_window,
        ema_span=ema_span,
        explosion_ratio_threshold=explosion_ratio_threshold,
        max_abs_return_prediction=max_abs_return_prediction,
    )
    _validate_inference_config(config)
    return config


def save_inference_config(config: InferenceConfig, output_path: Path | str) -> Path:
    _validate_inference_config(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def load_inference_config(config_path: Path | str) -> InferenceConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference config not found: {path.as_posix()}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "model_path",
        "scaler_path",
        "window_size",
        "model_input_features",
        "target_column",
        "created_at_utc",
    }
    missing_keys = sorted(required_keys.difference(raw.keys()))
    if missing_keys:
        raise KeyError(f"Inference config is missing required keys: {missing_keys}")

    config = InferenceConfig(
        model_path=str(raw["model_path"]),
        scaler_path=str(raw["scaler_path"]),
        window_size=int(raw["window_size"]),
        model_input_features=list(raw["model_input_features"]),
        target_column=str(raw["target_column"]),
        created_at_utc=str(raw["created_at_utc"]),
        ticker=str(raw.get("ticker", "AAPL")),
        interval=str(raw.get("interval", "1d")),
        ma_short_window=int(raw.get("ma_short_window", 10)),
        ma_long_window=int(raw.get("ma_long_window", 30)),
        ema_span=int(raw.get("ema_span", 10)),
        explosion_ratio_threshold=float(raw.get("explosion_ratio_threshold", 2.0)),
        max_abs_return_prediction=float(raw.get("max_abs_return_prediction", 0.5)),
    )
    _validate_inference_config(config)
    return config


def _validate_inference_config(config: InferenceConfig) -> None:
    if config.window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if not config.model_input_features:
        raise ValueError("model_input_features must not be empty.")

    if any(not feature for feature in config.model_input_features):
        raise ValueError("model_input_features must only contain non-empty values.")

    if not config.target_column:
        raise ValueError("target_column must be non-empty.")

    if not config.model_path:
        raise ValueError("model_path must be non-empty.")

    if not config.scaler_path:
        raise ValueError("scaler_path must be non-empty.")

    if not config.ticker:
        raise ValueError("ticker must be non-empty.")

    if not config.interval:
        raise ValueError("interval must be non-empty.")

    if config.ma_short_window <= 0 or config.ma_long_window <= 0 or config.ema_span <= 0:
        raise ValueError("Feature-engineering windows must be positive integers.")

    if config.ma_short_window >= config.ma_long_window:
        raise ValueError("ma_short_window must be smaller than ma_long_window.")

    if config.explosion_ratio_threshold <= 1.0:
        raise ValueError("explosion_ratio_threshold must be greater than 1.0.")

    if config.max_abs_return_prediction <= 0:
        raise ValueError("max_abs_return_prediction must be positive.")

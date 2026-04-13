from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    ma_short_window: int = 10
    ma_long_window: int = 30
    ema_span: int = 10


@dataclass(frozen=True, slots=True)
class ReturnsConfig:
    include_return_ma: bool = False
    include_volatility: bool = False
    return_ma_window: int = 10
    volatility_window: int = 10
    mean_center_tolerance: float = 0.05


def engineer_features(
    data: pd.DataFrame, config: FeatureEngineeringConfig | None = None
) -> pd.DataFrame:
    config = config or FeatureEngineeringConfig()
    _validate_input(data, config)

    enriched = data.copy()
    enriched[f"MA{config.ma_short_window}"] = (
        enriched["Close"].rolling(window=config.ma_short_window).mean()
    )
    enriched[f"MA{config.ma_long_window}"] = (
        enriched["Close"].rolling(window=config.ma_long_window).mean()
    )
    enriched[f"EMA{config.ema_span}"] = enriched["Close"].ewm(
        span=config.ema_span,
        adjust=False,
        min_periods=config.ema_span,
    ).mean()

    enriched = enriched.dropna()
    _validate_output(enriched, config)
    return enriched


def compute_returns(data: pd.DataFrame, config: ReturnsConfig | None = None) -> pd.DataFrame:
    config = config or ReturnsConfig()
    _validate_returns_input(data, config)

    enriched = data.copy()
    enriched["return"] = enriched["Close"].pct_change()

    if config.include_return_ma:
        enriched[f"ret_ma{config.return_ma_window}"] = (
            enriched["return"].rolling(window=config.return_ma_window).mean()
        )

    if config.include_volatility:
        enriched[f"ret_vol{config.volatility_window}"] = (
            enriched["return"].rolling(window=config.volatility_window).std()
        )

    enriched = enriched.dropna()
    _validate_returns_output(enriched, config)
    return enriched


def _validate_input(data: pd.DataFrame, config: FeatureEngineeringConfig) -> None:
    if data.empty:
        raise ValueError("Input dataframe is empty.")

    if config.ma_short_window <= 0 or config.ma_long_window <= 0 or config.ema_span <= 0:
        raise ValueError("Feature window sizes must be positive integers.")

    if config.ma_short_window >= config.ma_long_window:
        raise ValueError("ma_short_window must be smaller than ma_long_window.")

    missing_columns = [column for column in BASE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required base columns: {missing_columns}")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Input index must be sorted ascending.")

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Input index contains duplicate timestamps: {duplicate_count}")


def _validate_output(data: pd.DataFrame, config: FeatureEngineeringConfig) -> None:
    required_feature_columns = [
        f"MA{config.ma_short_window}",
        f"MA{config.ma_long_window}",
        f"EMA{config.ema_span}",
    ]
    missing_features = [column for column in required_feature_columns if column not in data.columns]
    if missing_features:
        raise KeyError(f"Missing engineered feature columns: {missing_features}")

    if data.empty:
        raise ValueError("Feature engineering removed all rows.")

    if data.isna().any().any():
        null_counts = data.isna().sum().to_dict()
        raise ValueError(f"NaN values remain after feature engineering: {null_counts}")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Output index must remain sorted ascending.")

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Output index contains duplicate timestamps: {duplicate_count}")


def _validate_returns_input(data: pd.DataFrame, config: ReturnsConfig) -> None:
    if data.empty:
        raise ValueError("Input dataframe is empty.")

    if "Close" not in data.columns:
        raise KeyError("Missing required column for returns computation: ['Close']")

    if config.mean_center_tolerance <= 0:
        raise ValueError("mean_center_tolerance must be a positive float.")

    if config.return_ma_window <= 0 or config.volatility_window <= 0:
        raise ValueError("Returns window sizes must be positive integers.")

    if config.include_return_ma and config.return_ma_window > len(data):
        raise ValueError(
            "return_ma_window cannot exceed the number of rows when include_return_ma is enabled."
        )

    if config.include_volatility and config.volatility_window > len(data):
        raise ValueError(
            "volatility_window cannot exceed the number of rows when include_volatility is enabled."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError("Input index must be sorted ascending.")

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Input index contains duplicate timestamps: {duplicate_count}")


def _validate_returns_output(data: pd.DataFrame, config: ReturnsConfig) -> None:
    required_columns = ["return"]
    if config.include_return_ma:
        required_columns.append(f"ret_ma{config.return_ma_window}")
    if config.include_volatility:
        required_columns.append(f"ret_vol{config.volatility_window}")

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing return-based columns: {missing_columns}")

    if data.empty:
        raise ValueError("Returns engineering removed all rows.")

    if data.isna().any().any():
        null_counts = data.isna().sum().to_dict()
        raise ValueError(f"NaN values remain after returns engineering: {null_counts}")

    return_mean = float(data["return"].mean())
    if abs(return_mean) > config.mean_center_tolerance:
        raise ValueError(
            "Return distribution is not centered near 0. "
            f"Observed mean={return_mean:.6f}, tolerance={config.mean_center_tolerance:.6f}."
        )

    if float(data["return"].std()) <= 0:
        raise ValueError("Return standard deviation must be positive.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Output index must remain sorted ascending.")

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Output index contains duplicate timestamps: {duplicate_count}")
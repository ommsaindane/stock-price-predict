from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

MissingValueStrategy = Literal["drop", "ffill"]

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True, slots=True)
class StockDataConfig:
    ticker: str = "AAPL"
    period: str = "10y"
    interval: str = "1d"
    missing_value_strategy: MissingValueStrategy = "drop"
    output_dir: Path = Path("data")


def load_stock_data(config: StockDataConfig | None = None) -> pd.DataFrame:
    config = config or StockDataConfig()
    raw_data = _download_stock_data(config)
    raw_csv_path = _save_raw_data(raw_data, config)
    clean_data = _clean_stock_data(raw_data, config)
    _validate_clean_data(clean_data)
    _print_validation_summary(clean_data, raw_csv_path)
    return clean_data


def _download_stock_data(config: StockDataConfig) -> pd.DataFrame:
    data = yf.download(
        tickers=config.ticker,
        period=config.period,
        interval=config.interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError(
            f"No stock data returned for ticker {config.ticker} with period {config.period}."
        )

    return data


def _save_raw_data(data: pd.DataFrame, config: StockDataConfig) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_dir / f"{config.ticker}_{config.period}_{config.interval}_raw_{timestamp}.csv"
    data.to_csv(file_path)
    return file_path


def _clean_stock_data(data: pd.DataFrame, config: StockDataConfig) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(config.ticker, axis=1, level=-1)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns for ticker {config.ticker}: {missing_columns}"
        )

    clean_data = data.loc[:, REQUIRED_COLUMNS].copy()
    clean_data.columns.name = None
    clean_data.index = pd.to_datetime(clean_data.index, errors="raise")
    if getattr(clean_data.index, "tz", None) is not None:
        clean_data.index = clean_data.index.tz_localize(None)
    clean_data = clean_data.sort_index()

    if config.missing_value_strategy == "drop":
        clean_data = clean_data.dropna()
    elif config.missing_value_strategy == "ffill":
        clean_data = clean_data.ffill()
    else:
        raise ValueError(
            f"Unsupported missing value strategy: {config.missing_value_strategy}"
        )

    if clean_data.isna().any().any():
        null_counts = clean_data.isna().sum().to_dict()
        raise ValueError(
            f"Missing values remain after applying {config.missing_value_strategy}: {null_counts}"
        )

    return clean_data


def _validate_clean_data(data: pd.DataFrame) -> None:
    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Duplicate timestamps found in cleaned data: {duplicate_count}")


def _print_validation_summary(data: pd.DataFrame, raw_csv_path: Path) -> None:
    print(f"Raw CSV saved to: {raw_csv_path}")
    print(f"Shape: {data.shape}")
    print("Head:")
    print(data.head())
    print("Tail:")
    print(data.tail())
    print("Null counts:")
    print(data.isna().sum())
    print(f"Duplicate timestamps: {data.index.has_duplicates}")
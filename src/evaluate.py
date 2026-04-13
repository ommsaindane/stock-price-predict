from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    outputs_dir: Path = Path("outputs")
    metrics_output_path: Path = Path("outputs/evaluation_metrics.json")
    reconstructed_prices_csv_path: Path = Path("outputs/reconstructed_prices.csv")
    returns_plot_path: Path = Path("outputs/returns_pred_vs_actual.png")
    prices_plot_path: Path = Path("outputs/prices_pred_vs_actual.png")
    error_hist_plot_path: Path = Path("outputs/price_error_distribution.png")
    explosion_ratio_threshold: float = 2.0


def evaluate_model(
    model: Model,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    enriched_data: pd.DataFrame,
    split_index: int,
    window_size: int,
    config: EvaluationConfig | None = None,
) -> tuple[dict, Path, Path, list[Path]]:
    config = config or EvaluationConfig()
    _validate_eval_inputs(
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        enriched_data=enriched_data,
        split_index=split_index,
        window_size=window_size,
        config=config,
    )

    y_pred_seq = predict_returns(model, X_test_seq)
    base_dates, base_prices, actual_prices = align_price_series(
        enriched_data=enriched_data,
        split_index=split_index,
        window_size=window_size,
        sequence_count=len(y_pred_seq),
    )

    predicted_prices, explosion_flags = reconstruct_prices(
        base_prices=base_prices,
        predicted_returns=y_pred_seq,
        explosion_ratio_threshold=config.explosion_ratio_threshold,
    )

    return_metrics = evaluate_return_metrics(y_true=y_test_seq, y_pred=y_pred_seq)
    price_metrics = evaluate_price_metrics(y_true=actual_prices, y_pred=predicted_prices)

    has_explosions = bool(explosion_flags.any())
    if has_explosions:
        raise ValueError(
            "Price reconstruction explosion detected. "
            f"threshold={config.explosion_ratio_threshold}, count={int(explosion_flags.sum())}."
        )

    metrics_payload = {
        "returns": return_metrics,
        "prices": price_metrics,
        "reconstruction": {
            "explosion_ratio_threshold": config.explosion_ratio_threshold,
            "explosion_count": int(explosion_flags.sum()),
            "has_explosions": has_explosions,
        },
    }

    metrics_path = save_metrics(metrics_payload, config.metrics_output_path)
    csv_path = save_reconstructed_prices_csv(
        dates=base_dates,
        actual_returns=y_test_seq,
        predicted_returns=y_pred_seq,
        base_prices=base_prices,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        explosion_flags=explosion_flags,
        output_path=config.reconstructed_prices_csv_path,
    )
    plot_paths = save_evaluation_plots(
        dates=base_dates,
        actual_returns=y_test_seq,
        predicted_returns=y_pred_seq,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        price_errors=(predicted_prices - actual_prices),
        config=config,
    )

    return metrics_payload, metrics_path, csv_path, plot_paths


def predict_returns(model: Model, X_test_seq: np.ndarray) -> np.ndarray:
    y_pred = model.predict(X_test_seq, verbose=0).reshape(-1)
    if y_pred.ndim != 1:
        raise ValueError("Predicted returns must be a 1D array.")
    if not np.isfinite(y_pred).all():
        raise ValueError("Predicted returns contain NaN or infinite values.")
    return y_pred.astype(np.float64)


def evaluate_return_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    _validate_metric_inputs(y_true, y_pred, label="return")
    errors = y_pred - y_true
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    directional_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "max_abs_error": float(np.max(np.abs(errors))),
    }


def evaluate_price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    _validate_metric_inputs(y_true, y_pred, label="price")
    errors = y_pred - y_true
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    direction_accuracy = float(
        np.mean(np.sign(np.diff(y_pred, prepend=y_pred[0])) == np.sign(np.diff(y_true, prepend=y_true[0])))
    )
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "median_abs_error": float(np.median(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "trend_direction_accuracy": direction_accuracy,
    }


def align_price_series(
    enriched_data: pd.DataFrame,
    split_index: int,
    window_size: int,
    sequence_count: int,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    if "Close" not in enriched_data.columns:
        raise KeyError("Missing required 'Close' column for price reconstruction.")

    local_indices = np.arange(window_size - 1, window_size - 1 + sequence_count, dtype=np.int64)
    aligned_positions = split_index + local_indices
    base_positions = aligned_positions
    actual_positions = aligned_positions + 1

    if int(actual_positions[-1]) >= len(enriched_data):
        raise ValueError("Price alignment out of bounds for reconstruction.")

    close_values = enriched_data["Close"].to_numpy(dtype=np.float64)
    base_prices = close_values[base_positions]
    actual_prices = close_values[actual_positions]
    dates = enriched_data.index[actual_positions]

    if not np.isfinite(base_prices).all() or not np.isfinite(actual_prices).all():
        raise ValueError("Aligned price series contain NaN or infinite values.")

    return dates, base_prices, actual_prices


def reconstruct_prices(
    base_prices: np.ndarray,
    predicted_returns: np.ndarray,
    explosion_ratio_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_metric_inputs(base_prices, predicted_returns, label="reconstruction")
    predicted_prices = base_prices * (1.0 + predicted_returns)
    ratio = predicted_prices / base_prices
    explosion_flags = (ratio > explosion_ratio_threshold) | (ratio <= 0)
    if not np.isfinite(predicted_prices).all():
        raise ValueError("Reconstructed prices contain NaN or infinite values.")
    return predicted_prices, explosion_flags


def save_metrics(metrics_payload: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return path


def save_reconstructed_prices_csv(
    dates: pd.Index,
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    base_prices: np.ndarray,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    explosion_flags: np.ndarray,
    output_path: Path | str,
) -> Path:
    df = pd.DataFrame(
        {
            "date": dates,
            "actual_return": actual_returns,
            "predicted_return": predicted_returns,
            "base_price": base_prices,
            "actual_price": actual_prices,
            "predicted_price": predicted_prices,
            "abs_price_error": np.abs(predicted_prices - actual_prices),
            "price_error_pct": np.abs(predicted_prices - actual_prices)
            / np.maximum(actual_prices, 1e-12),
            "explosion_flag": explosion_flags,
        }
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_evaluation_plots(
    dates: pd.Index,
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    price_errors: np.ndarray,
    config: EvaluationConfig,
) -> list[Path]:
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_returns, label="Actual return", linewidth=1.2)
    plt.plot(dates, predicted_returns, label="Predicted return", linewidth=1.2)
    plt.title("Returns: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.returns_plot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label="Actual price", linewidth=1.5)
    plt.plot(dates, predicted_prices, label="Predicted reconstructed price", linewidth=1.5)
    plt.title("Reconstructed Price vs Actual Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.prices_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(price_errors, bins=40)
    plt.title("Price Error Distribution")
    plt.xlabel("Predicted - Actual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(config.error_hist_plot_path)
    plt.close()

    return [config.returns_plot_path, config.prices_plot_path, config.error_hist_plot_path]


def _validate_eval_inputs(
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    enriched_data: pd.DataFrame,
    split_index: int,
    window_size: int,
    config: EvaluationConfig,
) -> None:
    if X_test_seq.ndim != 3:
        raise ValueError("X_test_seq must be 3D with shape (samples, timesteps, features).")

    if y_test_seq.ndim != 1:
        raise ValueError("y_test_seq must be 1D with shape (samples,).")

    if len(X_test_seq) != len(y_test_seq):
        raise ValueError("X_test_seq and y_test_seq length mismatch.")

    if len(X_test_seq) == 0:
        raise ValueError("Evaluation set is empty.")

    if enriched_data.empty:
        raise ValueError("enriched_data is empty.")

    if split_index <= 0 or split_index >= len(enriched_data):
        raise ValueError("split_index is outside valid range for enriched_data.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if config.explosion_ratio_threshold <= 1.0:
        raise ValueError("explosion_ratio_threshold must be greater than 1.0.")


def _validate_metric_inputs(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"{label} metric inputs must be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"{label} metric inputs length mismatch.")
    if len(y_true) == 0:
        raise ValueError(f"{label} metric inputs are empty.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(f"{label} metric inputs contain NaN or infinite values.")

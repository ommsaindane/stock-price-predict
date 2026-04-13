from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    model_input_features: list[str] = field(default_factory=lambda: ["return"])
    target_column: str = "return"
    train_ratio: float = 0.8
    scaler_path: Path = Path("models/feature_scaler_standard.joblib")
    metadata_path: Path = Path("models/preprocessing_config.json")
    mean_tolerance: float = 1e-2
    std_tolerance: float = 5e-2


def preprocess_data(
    data: pd.DataFrame, config: PreprocessConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Path, Path]:
    config = config or PreprocessConfig()
    _validate_preprocess_input(data, config)

    X_full = data.loc[:, config.model_input_features].to_numpy(dtype=np.float64)
    y_full = data[config.target_column].shift(-1).to_numpy(dtype=np.float64)
    aligned_index_full = data.index.to_numpy()[np.isfinite(y_full)]

    valid_mask = np.isfinite(y_full)
    X_aligned = X_full[valid_mask]
    y_aligned = y_full[valid_mask]

    X_train, X_test, y_train, y_test, split_index, split_boundary, test_start_boundary = (
        split_chronologically(X_aligned, y_aligned, aligned_index_full, config.train_ratio)
    )

    _validate_split(X_train, X_test, y_train, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    _validate_scaled_output(X_train_scaled, X_test_scaled, y_train, y_test, config)

    scaler_path = save_scaler(scaler, config.scaler_path)
    metadata_path = save_preprocessing_metadata(
        config=config,
        scaler_path=scaler_path,
        train_samples=len(X_train_scaled),
        test_samples=len(X_test_scaled),
        split_boundary=split_boundary,
        test_start_boundary=test_start_boundary,
        no_overlap_validated=True,
        train_feature_mean=X_train_scaled.mean(axis=0),
        train_feature_std=X_train_scaled.std(axis=0),
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, scaler_path, metadata_path


def split_chronologically(
    X: np.ndarray,
    y: np.ndarray,
    aligned_index: np.ndarray,
    train_ratio: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    object,
    object,
]:
    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError(
            "Train/test split created an empty partition. "
            f"Computed split_index={split_index}, samples={len(X)}."
        )

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    split_boundary = aligned_index[split_index - 1]
    test_start_boundary = aligned_index[split_index]
    if not split_boundary < test_start_boundary:
        raise ValueError(
            "Chronological split overlap detected: train boundary must be earlier than test start. "
            f"train_end={split_boundary}, test_start={test_start_boundary}."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        split_index,
        split_boundary,
        test_start_boundary,
    )


def save_scaler(scaler: StandardScaler, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def save_preprocessing_metadata(
    config: PreprocessConfig,
    scaler_path: Path,
    train_samples: int,
    test_samples: int,
    split_boundary: object,
    test_start_boundary: object,
    no_overlap_validated: bool,
    train_feature_mean: np.ndarray,
    train_feature_std: np.ndarray,
) -> Path:
    path = Path(config.metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    split_boundary_text = str(split_boundary)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "column": config.target_column,
            "alignment": "y[t] = return(t+1)",
        },
        "split": {
            "train_ratio": config.train_ratio,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "split_boundary": split_boundary_text,
            "test_start_boundary": str(test_start_boundary),
            "no_shuffle": True,
            "no_overlap_validated": no_overlap_validated,
        },
        "features": {
            "model_input_features": config.model_input_features,
            "feature_count": len(config.model_input_features),
        },
        "scaler": {
            "type": "StandardScaler",
            "path": str(scaler_path).replace('\\\\', '/'),
            "fit_scope": "train_only",
            "scaled_train_mean": train_feature_mean.tolist(),
            "scaled_train_std": train_feature_std.tolist(),
        },
    }

    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _validate_preprocess_input(data: pd.DataFrame, config: PreprocessConfig) -> None:
    if data.empty:
        raise ValueError("Input dataframe is empty.")

    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not config.model_input_features:
        raise ValueError("model_input_features must contain at least one column.")

    missing_feature_columns = [
        column for column in config.model_input_features if column not in data.columns
    ]
    if missing_feature_columns:
        raise KeyError(f"Missing feature columns: {missing_feature_columns}")

    if config.target_column not in data.columns:
        raise KeyError(f"Missing target column: {config.target_column}")

    if "Close" in config.model_input_features:
        raise ValueError("Close must not be included in model_input_features for Step 4.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Input index must be sorted ascending.")

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(f"Input index contains duplicate timestamps: {duplicate_count}")

    feature_values = data.loc[:, config.model_input_features].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature matrix contains NaN or infinite values before preprocessing.")

    target_values = data[config.target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError("Target column contains NaN or infinite values before preprocessing.")


def _validate_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if len(X_train) != len(y_train):
        raise ValueError("Training feature/target length mismatch.")

    if len(X_test) != len(y_test):
        raise ValueError("Test feature/target length mismatch.")

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Train/test split contains an empty feature matrix.")

    if y_train.size == 0 or y_test.size == 0:
        raise ValueError("Train/test split contains an empty target vector.")


def _validate_scaled_output(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: PreprocessConfig,
) -> None:
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():
        raise ValueError("Scaled features contain NaN or infinite values.")

    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Target vectors contain NaN or infinite values.")

    train_means = X_train_scaled.mean(axis=0)
    train_stds = X_train_scaled.std(axis=0)

    if np.any(np.abs(train_means) > config.mean_tolerance):
        raise ValueError(
            "Scaled training feature means are not centered near 0. "
            f"Observed means={train_means.tolist()}."
        )

    if np.any(np.abs(train_stds - 1.0) > config.std_tolerance):
        raise ValueError(
            "Scaled training feature std is not near 1. "
            f"Observed std={train_stds.tolist()}."
        )

    # StandardScaler does not clip values; this guards against collapsed/saturated feature ranges.
    p01 = np.percentile(X_train_scaled, 1, axis=0)
    p99 = np.percentile(X_train_scaled, 99, axis=0)
    if np.any(np.abs(p99 - p01) < 1e-3):
        raise ValueError(
            "Scaled training features appear saturated/collapsed (very narrow percentile spread)."
        )

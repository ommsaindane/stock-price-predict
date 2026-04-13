from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    window_size: int = 30
    metadata_path: Path = Path("models/sequence_config.json")


def create_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: SequenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, Path]:
    config = config or SequenceConfig()
    _validate_inputs(X_train, X_test, y_train, y_test, config)

    X_train_seq, y_train_idx = _sliding_window(X_train, config.window_size)
    X_test_seq, y_test_idx = _sliding_window(X_test, config.window_size)

    y_train_seq = y_train[y_train_idx]
    y_test_seq = y_test[y_test_idx]

    _validate_outputs(X_train_seq, X_test_seq, y_train_seq, y_test_seq, config)

    metadata = _build_sequence_metadata(
        X_train_seq=X_train_seq,
        X_test_seq=X_test_seq,
        y_train_seq=y_train_seq,
        y_test_seq=y_test_seq,
        y_train_idx=y_train_idx,
        y_test_idx=y_test_idx,
        config=config,
    )
    metadata_path = save_sequence_metadata(metadata, config.metadata_path)

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, metadata, metadata_path


def save_sequence_metadata(metadata: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _sliding_window(X: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count, feature_count = X.shape
    sequence_count = sample_count - window_size + 1

    X_seq = np.empty((sequence_count, window_size, feature_count), dtype=X.dtype)
    for start in range(sequence_count):
        X_seq[start] = X[start : start + window_size]

    # With y built as return(t+1), y[start + window_size - 1] is the next step for the window.
    y_indices = np.arange(window_size - 1, sample_count, dtype=np.int64)
    return X_seq, y_indices


def _validate_inputs(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: SequenceConfig,
) -> None:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Feature arrays must be 2D with shape (samples, features).")

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("Target arrays must be 1D with shape (samples,).")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training feature/target length mismatch.")

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test feature/target length mismatch.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature count mismatch.")

    if config.window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if X_train.shape[0] < config.window_size:
        raise ValueError(
            "Not enough training samples for the configured window_size. "
            f"window_size={config.window_size}, train_samples={X_train.shape[0]}."
        )

    if X_test.shape[0] < config.window_size:
        raise ValueError(
            "Not enough test samples for the configured window_size. "
            f"window_size={config.window_size}, test_samples={X_test.shape[0]}."
        )

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature arrays contain NaN or infinite values.")

    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Target arrays contain NaN or infinite values.")


def _validate_outputs(
    X_train_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_train_seq: np.ndarray,
    y_test_seq: np.ndarray,
    config: SequenceConfig,
) -> None:
    if X_train_seq.ndim != 3 or X_test_seq.ndim != 3:
        raise ValueError("Sequence feature arrays must be 3D.")

    if y_train_seq.ndim != 1 or y_test_seq.ndim != 1:
        raise ValueError("Sequence target arrays must be 1D.")

    if X_train_seq.shape[1] != config.window_size or X_test_seq.shape[1] != config.window_size:
        raise ValueError("Generated sequences do not match configured window_size.")

    if X_train_seq.shape[0] != y_train_seq.shape[0]:
        raise ValueError("Training sequence/target length mismatch.")

    if X_test_seq.shape[0] != y_test_seq.shape[0]:
        raise ValueError("Test sequence/target length mismatch.")

    if not np.isfinite(X_train_seq).all() or not np.isfinite(X_test_seq).all():
        raise ValueError("Generated sequence features contain NaN or infinite values.")

    if not np.isfinite(y_train_seq).all() or not np.isfinite(y_test_seq).all():
        raise ValueError("Generated sequence targets contain NaN or infinite values.")


def _build_sequence_metadata(
    X_train_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_train_seq: np.ndarray,
    y_test_seq: np.ndarray,
    y_train_idx: np.ndarray,
    y_test_idx: np.ndarray,
    config: SequenceConfig,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": config.window_size,
        "alignment_rule": "X[k:k+window_size] predicts y[k+window_size-1]",
        "shapes": {
            "X_train_seq": list(X_train_seq.shape),
            "X_test_seq": list(X_test_seq.shape),
            "y_train_seq": list(y_train_seq.shape),
            "y_test_seq": list(y_test_seq.shape),
        },
        "y_indices": {
            "train_first": int(y_train_idx[0]),
            "train_last": int(y_train_idx[-1]),
            "test_first": int(y_test_idx[0]),
            "test_last": int(y_test_idx[-1]),
        },
    }

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, History


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.0
    model_output_path: Path = Path("models/lstm_return_model.keras")
    history_output_path: Path = Path("models/training_history.json")


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig | None = None,
) -> tuple[Model, History, Path, Path]:
    config = config or TrainConfig()
    _validate_train_inputs(X_train, y_train, config)

    early_stopping = EarlyStopping(
        monitor=config.early_stopping_monitor,
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        restore_best_weights=True,
        mode="min",
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )

    _validate_training_history(history)
    model_path = save_trained_model(model, config.model_output_path)
    history_path = save_training_history(history, config.history_output_path)
    return model, history, model_path, history_path


def save_trained_model(model: Model, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def save_training_history(history: History, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    return path


def _validate_train_inputs(X_train: np.ndarray, y_train: np.ndarray, config: TrainConfig) -> None:
    if X_train.ndim != 3:
        raise ValueError("X_train must be 3D with shape (samples, timesteps, features).")

    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D with shape (samples,).")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train sample counts must match.")

    if X_train.shape[0] < 10:
        raise ValueError("Not enough samples for training.")

    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training arrays contain NaN or infinite values.")

    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not 0.1 <= config.validation_split <= 0.2:
        raise ValueError("validation_split must be in [0.1, 0.2] for this step.")

    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")


def _validate_training_history(history: History) -> None:
    metrics = history.history
    required_keys = ["loss", "val_loss"]
    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise ValueError(f"Training history is missing required metrics: {missing}")

    loss = np.asarray(metrics["loss"], dtype=np.float64)
    val_loss = np.asarray(metrics["val_loss"], dtype=np.float64)

    if loss.size == 0 or val_loss.size == 0:
        raise ValueError("Training history metrics are empty.")

    if not np.isfinite(loss).all() or not np.isfinite(val_loss).all():
        raise ValueError("Training produced NaN or infinite loss values.")

    if loss[-1] >= loss[0]:
        raise ValueError(
            "Training loss did not decrease overall. "
            f"initial={loss[0]:.6f}, final={loss[-1]:.6f}."
        )

    if val_loss[-1] >= val_loss[0]:
        raise ValueError(
            "Validation loss did not decrease overall. "
            f"initial={val_loss[0]:.6f}, final={val_loss[-1]:.6f}."
        )

    if np.max(loss) > loss[0] * 5 or np.max(val_loss) > val_loss[0] * 5:
        raise ValueError("Training appears unstable: detected strong loss explosion.")

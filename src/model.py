from __future__ import annotations

from dataclasses import dataclass

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


@dataclass(frozen=True, slots=True)
class ModelConfig:
    lstm_units: int = 64
    dropout_rate: float = 0.2
    output_units: int = 1
    learning_rate: float = 1e-3
    loss: str = "mse"


def build_lstm_model(
    input_shape: tuple[int, int], config: ModelConfig | None = None
) -> Model:
    config = config or ModelConfig()
    _validate_model_inputs(input_shape, config)

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(config.lstm_units),
            Dropout(config.dropout_rate),
            Dense(config.output_units),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss=config.loss)
    return model


def _validate_model_inputs(input_shape: tuple[int, int], config: ModelConfig) -> None:
    if len(input_shape) != 2:
        raise ValueError(
            "input_shape must be a 2-tuple: (timesteps, features)."
        )

    timesteps, features = input_shape
    if timesteps <= 0 or features <= 0:
        raise ValueError(
            "input_shape values must be positive integers. "
            f"Received timesteps={timesteps}, features={features}."
        )

    if config.lstm_units <= 0:
        raise ValueError("lstm_units must be a positive integer.")

    if not 0 <= config.dropout_rate < 1:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    if config.output_units != 1:
        raise ValueError("output_units must be 1 for next-step return regression.")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if config.loss != "mse":
        raise ValueError("loss must be 'mse' for this step.")

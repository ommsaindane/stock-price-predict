from pathlib import Path

import numpy as np

from src.data_loader import StockDataConfig, load_stock_data
from src.dataset import SequenceConfig, create_sequences
from src.evaluate import EvaluationConfig, evaluate_model
from src.feature_engineering import (
    FeatureEngineeringConfig,
    ReturnsConfig,
    compute_returns,
    engineer_features,
)
from src.model import ModelConfig, build_lstm_model
from src.preprocess import PreprocessConfig, preprocess_data
from src.train import TrainConfig, train_model
from src.inference import predict_next_day_price, run_reloaded_inference
from src.inference_config import build_inference_config, save_inference_config


def main() -> None:
    data_config = StockDataConfig()
    feature_config = FeatureEngineeringConfig()
    returns_config = ReturnsConfig()
    preprocess_config = PreprocessConfig()
    sequence_config = SequenceConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    evaluation_config = EvaluationConfig()

    clean_data = load_stock_data(data_config)
    engineered_data = engineer_features(clean_data, feature_config)
    enriched_data = compute_returns(engineered_data, returns_config)
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        _scaler,
        scaler_path,
        metadata_path,
    ) = preprocess_data(enriched_data, preprocess_config)
    (
        X_train_seq,
        X_test_seq,
        y_train_seq,
        y_test_seq,
        _sequence_metadata,
        sequence_metadata_path,
    ) = create_sequences(X_train_scaled, X_test_scaled, y_train, y_test, sequence_config)
    model_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(model_input_shape, model_config)
    trained_model, history, trained_model_path, history_path = train_model(
        model, X_train_seq, y_train_seq, train_config
    )
    evaluation_metrics, metrics_path, csv_path, plot_paths = evaluate_model(
        model=trained_model,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        enriched_data=enriched_data,
        split_index=len(y_train),
        window_size=sequence_config.window_size,
        config=evaluation_config,
    )
    inference_config = build_inference_config(
        model_path=trained_model_path,
        scaler_path=scaler_path,
        window_size=sequence_config.window_size,
        model_input_features=preprocess_config.model_input_features,
        target_column=preprocess_config.target_column,
        ticker=data_config.ticker,
        interval=data_config.interval,
        ma_short_window=feature_config.ma_short_window,
        ma_long_window=feature_config.ma_long_window,
        ema_span=feature_config.ema_span,
        explosion_ratio_threshold=2.0,
        max_abs_return_prediction=0.5,
    )
    inference_config_path = save_inference_config(
        config=inference_config,
        output_path=Path("models/inference_config.json"),
    )
    in_memory_predictions = trained_model.predict(X_test_seq, verbose=0).reshape(-1)
    reloaded_predictions = run_reloaded_inference(X_test_seq, inference_config_path)
    reload_parity_mse = float(
        np.mean(np.square(in_memory_predictions - reloaded_predictions))
    )
    reload_parity_tolerance = 1e-5
    if reload_parity_mse > reload_parity_tolerance:
        raise ValueError(
            "Reloaded inference parity check failed. "
            f"mse={reload_parity_mse:.10f}, tolerance={reload_parity_tolerance:.10f}."
        )
    next_day_inference = predict_next_day_price(inference_config_path)

    print(f"Enriched shape: {enriched_data.shape}")
    print(
        "Engineered columns: "
        f"MA{feature_config.ma_short_window}, "
        f"MA{feature_config.ma_long_window}, "
        f"EMA{feature_config.ema_span}, "
        "return"
    )
    print(f"NaNs remaining: {int(enriched_data.isna().sum().sum())}")
    print(f"Return mean: {float(enriched_data['return'].mean()):.6f}")
    print(f"Return std: {float(enriched_data['return'].std()):.6f}")
    print(f"Contains raw Close for reconstruction: {'Close' in enriched_data.columns}")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Chronological split: 80/20, no shuffle, no-overlap validation enabled")
    print(f"Scaled train mean: {float(X_train_scaled.mean()):.6f}")
    print(f"Scaled train std: {float(X_train_scaled.std()):.6f}")
    print(f"Scaler saved: {scaler_path.as_posix()}")
    print(f"Preprocess metadata saved: {metadata_path.as_posix()}")
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")
    print(f"y_train_seq shape: {y_train_seq.shape}")
    print(f"y_test_seq shape: {y_test_seq.shape}")
    print(f"Sequence metadata saved: {sequence_metadata_path.as_posix()}")
    print(f"Model input shape: {model_input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Model optimizer: {model.optimizer.__class__.__name__.lower()}")
    print(f"Model loss: {model.loss}")
    model.summary(print_fn=print)
    print(f"Training epochs run: {len(history.history['loss'])}")
    print(f"Initial loss: {float(history.history['loss'][0]):.6f}")
    print(f"Final loss: {float(history.history['loss'][-1]):.6f}")
    print(f"Initial val_loss: {float(history.history['val_loss'][0]):.6f}")
    print(f"Final val_loss: {float(history.history['val_loss'][-1]):.6f}")
    print(f"Trained model saved: {trained_model_path.as_posix()}")
    print(f"Training history saved: {history_path.as_posix()}")
    print(f"Trained model output shape: {trained_model.output_shape}")
    print(f"Return MSE: {evaluation_metrics['returns']['mse']:.8f}")
    print(f"Return MAE: {evaluation_metrics['returns']['mae']:.8f}")
    print(f"Price MSE: {evaluation_metrics['prices']['mse']:.8f}")
    print(f"Price MAE: {evaluation_metrics['prices']['mae']:.8f}")
    print(
        "Return directional accuracy: "
        f"{evaluation_metrics['returns']['directional_accuracy']:.4f}"
    )
    if evaluation_metrics["returns"]["directional_accuracy"] < 0.5:
        print("WARNING: Return directional accuracy is below 0.5.")
    print(
        "Price trend direction accuracy: "
        f"{evaluation_metrics['prices']['trend_direction_accuracy']:.4f}"
    )
    print(f"Evaluation metrics saved: {metrics_path.as_posix()}")
    print(f"Reconstructed price rows saved: {csv_path.as_posix()}")
    print("Evaluation plots saved:")
    for path in plot_paths:
        print(f"- {path.as_posix()}")
    print(f"Inference config saved: {inference_config_path.as_posix()}")
    print(
        "Reload inference validation: PASS "
        f"(mse={reload_parity_mse:.10f}, tolerance={reload_parity_tolerance:.10f})"
    )
    print(
        "Next-day inference: "
        f"ticker={next_day_inference['ticker']}, "
        f"last_price={next_day_inference['last_price']:.4f}, "
        f"predicted_return={next_day_inference['predicted_return']:.6f}, "
        f"predicted_next_price={next_day_inference['predicted_next_price']:.4f}, "
        f"last_timestamp={next_day_inference['last_timestamp']}"
    )


if __name__ == "__main__":
    main()

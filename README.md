# Stock Price Prediction

An end-to-end stock forecasting project that models next-day returns with TensorFlow LSTM, reconstructs next-day prices, and serves predictions through both API and UI.

The system is designed to be deterministic and fail-fast for invalid states: it validates data integrity, feature consistency, scaler/model compatibility, and inference sanity checks before returning predictions.

<img width="1769" height="906" alt="Image" src="https://github.com/user-attachments/assets/fd69039b-39be-4f93-b11d-fb92fb51e10b" />

## Table of Contents

- [What Has Been Implemented](#what-has-been-implemented)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Run Training Pipeline](#run-training-pipeline)
- [Run FastAPI](#run-fastapi)
- [Run Streamlit UI](#run-streamlit-ui)
- [Key Output Files](#key-output-files)
- [Notes](#notes)
- [Development Journey and Key Challenges](#development-journey-and-key-challenges)

## What Has Been Implemented

### 1. Data ingestion and validation

- Downloads OHLCV market data from yfinance.
- Supports configurable ticker, period, interval, and missing-value strategy.
- Handles yfinance MultiIndex columns safely.
- Enforces required columns: Open, High, Low, Close, Volume.
- Validates sorted, deduplicated timestamps.
- Saves raw dataset snapshots to data/ with UTC timestamped filenames.

### 2. Feature engineering (time-series safe)

- Computes trend features:
	- MA10 (default)
	- MA30 (default)
	- EMA10 (default)
- Computes return feature:
	- return = Close.pct_change()
- Drops warm-up NaNs from rolling/EMA/return computation.
- Validates centered return distribution and non-zero variability.

### 3. Preprocessing

- Uses return-only model input by default (no raw Close leakage in model features).
- Creates next-step target alignment:
	- y[t] = return(t+1)
- Chronological train/test split (default 80/20), no shuffling.
- Fits StandardScaler on train set only, transforms both train and test.
- Validates scaled train mean/std and guards against saturated feature ranges.
- Saves:
	- scaler artifact
	- preprocessing metadata JSON

### 4. Sequence generation for LSTM

- Sliding-window sequence builder (default window_size = 30).
- Shapes:
	- X: (samples, timesteps, features)
	- y: (samples,)
- Validates dimensions, finite values, and sample sufficiency.
- Saves sequence metadata JSON.

### 5. Modeling and training

- LSTM architecture:
	- Input(timesteps, features)
	- LSTM(64)
	- Dropout(0.2)
	- Dense(1)
- Compiled with Adam and MSE loss.
- Training:
	- validation_split = 0.1
	- EarlyStopping on val_loss (restore_best_weights=True)
	- shuffle=False (time-series safe)
- Saves trained model and training history.
- Validates training behavior (loss/val_loss quality checks).

### 6. Evaluation and artifacts

- Return-level metrics:
	- MSE, MAE, RMSE, directional accuracy, max absolute error
- Price-level metrics (from reconstructed price):
	- MSE, MAE, RMSE, median absolute error, max absolute error, trend direction accuracy
- Reconstructs price from predicted return:
	- predicted_price = base_price * (1 + predicted_return)
- Explosion checks for unrealistic reconstructed prices.
- Saves:
	- outputs/evaluation_metrics.json
	- outputs/reconstructed_prices.csv
	- outputs/returns_pred_vs_actual.png
	- outputs/prices_pred_vs_actual.png
	- outputs/price_error_distribution.png

### 7. Inference runtime and safety checks

- Persists inference contract in models/inference_config.json.
- Reloads model + scaler and validates compatibility with inference config.
- Fetches fresh market data at inference time.
- Re-applies identical feature engineering and scaling logic used in training.
- Predicts next-day return from last sequence window.
- Applies hard guards:
	- finite-value checks
	- max absolute return threshold
	- price explosion ratio threshold
- Includes parity validation between in-memory and reloaded inference predictions.

### 8. FastAPI service

- API app with startup lifespan loading inference runtime once.
- Endpoints:
	- GET /health
	- GET /predict?ticker=AAPL
- Input validation for ticker format and length.
- Returns structured response with:
	- ticker
	- last timestamp and last price
	- predicted return and next price
	- window size and feature count
	- inference timestamp and latency

### 9. Streamlit dashboard

- Interactive prediction interface using the same inference runtime.
- Fails clearly when required artifacts are missing.
- Features:
	- ticker input with validation
	- one-click prediction
	- key metric cards
	- prediction latency and timestamp display
	- historical trend visualization (Close, MA, EMA)
	- runtime diagnostics panel
- Plotly chart layout tuned for readability (including non-overlapping legend placement).
- Streamlit theming configured via .streamlit/config.toml.

## Dependencies

Core dependencies are managed in pyproject.toml and include:

- tensorflow
- pandas
- numpy
- scikit-learn
- yfinance
- fastapi
- uvicorn
- streamlit
- plotly
- matplotlib
- seaborn
- joblib

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

## Run Training Pipeline

Run full training + evaluation + inference-config generation:

```bash
python main.py
```

This produces the model/scaler/config artifacts required by both API and UI.

## Run FastAPI

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl "http://localhost:8000/health"
```

Prediction example:

```bash
curl "http://localhost:8000/predict?ticker=AAPL"
```

## Run Streamlit UI

```bash
streamlit run app.py
```

The UI automatically loads runtime artifacts and displays an actionable error if they are missing.

## Key Output Files

- models/lstm_return_model.keras: trained LSTM model
- models/feature_scaler_standard.joblib: fitted scaler
- models/inference_config.json: runtime inference contract
- outputs/evaluation_metrics.json: return + price metrics
- outputs/reconstructed_prices.csv: row-level reconstructed price comparison
- outputs/*.png: visual evaluation diagnostics

## Notes

- The model predicts returns, not raw prices.
- Close is preserved for price reconstruction and evaluation context.
- All train/test operations preserve chronological ordering.
- Inference uses the same feature/scaling contract as training.

## Development Journey and Key Challenges

### Initial Approach (Failed)

The initial implementation modeled **absolute stock prices** using MinMaxScaler.

It resulted in strong evaluation metrics on the test set but complete failure during live inference

**Root issue:**
- Training scaler max < real-world price range
- New prices exceeded scaler bounds as inputs saturated at 1.0 and clipped the output.
- Model received constant input. Hence, produced unrealistic predictions (price explosion)

---

### Attempts to Fix

Several approaches were tested:

- Re-fitting scaler on full dataset
- Trying different scalers (MinMaxScaler, StandardScaler)
- Adjusting train/test splits

These mitigated symptoms but did **not solve the root problem**.

---

### Final Solution (Correct Fix)

The pipeline was redesigned to predict **returns instead of prices**:

- return = Close.pct_change()
- Model predicts next-day return
- Price reconstructed using:
  predicted_price = last_price * (1 + predicted_return)

---

### Outcome

- Eliminated scaling failures completely
- Stable inference (no explosions)
- Slight drop in directional accuracy, but:
  - System became **correct and production-safe**

---

### Key Insight

> Stock prices are non-stationary.  
> Modeling absolute values with bounded scaling is fundamentally flawed.

Correct approach:
> Model relative changes (returns), not absolute levels.

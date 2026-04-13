## Stock Price Prediction

This project trains an LSTM model on return-based features and provides two serving options:

- Streamlit dashboard for interactive usage
- FastAPI service for endpoint-based integration

## Prerequisites

- Python 3.11+
- Project virtual environment created at `.venv`

Install dependencies:

```bash
pip install -e .
```

## Generate Model Artifacts

Run the training pipeline once before using inference UI/API:

```bash
python main.py
```

Expected artifacts:

- `models/inference_config.json`
- `models/lstm_return_model.keras`
- `models/feature_scaler_standard.joblib`

## Run Streamlit UI

```bash
streamlit run app.py
```

The UI supports:

- ticker-based next-day prediction
- latency display
- historical trend chart (Close, MA, EMA)
- runtime diagnostics

## Run FastAPI

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl "http://localhost:8000/predict?ticker=AAPL"
```

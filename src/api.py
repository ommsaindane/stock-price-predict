from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.inference import (
    InferenceRuntime,
    load_inference_runtime,
    predict_next_day_price_from_runtime,
)

INFERENCE_CONFIG_PATH = Path("models/inference_config.json")


class PredictionResponse(BaseModel):
    ticker: str
    last_timestamp: str
    last_price: float
    predicted_return: float
    predicted_next_price: float
    window_size: int
    feature_count: int
    predicted_at_utc: str
    inference_latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = load_inference_runtime(INFERENCE_CONFIG_PATH)
    app.state.runtime = runtime
    yield


app = FastAPI(
    title="Stock Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    runtime_loaded = isinstance(getattr(app.state, "runtime", None), InferenceRuntime)
    return {
        "status": "ok" if runtime_loaded else "error",
        "runtime_loaded": runtime_loaded,
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=16,
        pattern=r"^[A-Za-z0-9.\-]+$",
        description="Stock ticker symbol, for example AAPL.",
    )
) -> PredictionResponse:
    runtime = getattr(app.state, "runtime", None)
    if not isinstance(runtime, InferenceRuntime):
        raise HTTPException(
            status_code=503,
            detail="Inference runtime is not loaded.",
        )

    start = time.perf_counter()
    try:
        prediction = predict_next_day_price_from_runtime(runtime, ticker=ticker)
    except Exception as exc:  # Runtime errors become user-visible request errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    return PredictionResponse(
        ticker=prediction["ticker"],
        last_timestamp=prediction["last_timestamp"],
        last_price=float(prediction["last_price"]),
        predicted_return=float(prediction["predicted_return"]),
        predicted_next_price=float(prediction["predicted_next_price"]),
        window_size=int(prediction["window_size"]),
        feature_count=int(prediction["feature_count"]),
        predicted_at_utc=datetime.now(timezone.utc).isoformat(),
        inference_latency_ms=float(latency_ms),
    )
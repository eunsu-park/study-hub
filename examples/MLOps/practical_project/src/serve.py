"""
Model serving API with FastAPI.

Adapted from MLOps Lesson 12 §5.1.
Serves churn predictions via REST API with Prometheus metrics.

Usage:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

import os
import time

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

app = FastAPI(title="Churn Prediction API")

# Prometheus metrics
PREDICTIONS = Counter("predictions_total", "Total predictions", ["status"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

# Globals loaded at startup
model = None
feature_columns = None


class PredictionRequest(BaseModel):
    user_id: int
    age: int
    tenure_months: int
    total_purchases: int
    avg_purchase_amount: float
    days_since_last_purchase: int
    support_tickets: int


class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    prediction: str


@app.on_event("startup")
async def load_resources():
    """Load model from MLflow at startup."""
    global model
    model_uri = os.environ.get(
        "MODEL_URI", "models:/churn-prediction/latest"
    )
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from {model_uri}")
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Serve will use dummy predictions.")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict churn probability for a user."""
    start_time = time.time()

    try:
        feature_vector = np.array([[
            request.age,
            request.tenure_months,
            request.total_purchases,
            request.avg_purchase_amount,
            request.days_since_last_purchase,
            request.support_tickets,
        ]])

        if model is not None:
            probability = float(model.predict_proba(feature_vector)[0][1])
        else:
            probability = 0.5  # dummy fallback

        prediction = "High Risk" if probability > 0.5 else "Low Risk"

        PREDICTIONS.labels(status="success").inc()
        LATENCY.observe(time.time() - start_time)

        return PredictionResponse(
            user_id=request.user_id,
            churn_probability=probability,
            prediction=prediction,
        )

    except Exception as e:
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

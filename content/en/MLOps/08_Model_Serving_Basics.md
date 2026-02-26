# 08. Model Serving Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the three model serving paradigms — batch inference, online inference, and streaming inference — and select the appropriate pattern for a given use case
2. Build a REST API model server using FastAPI, implementing prediction endpoints with input validation and error handling
3. Containerize a model server with Docker and deploy it with proper resource allocation and health check configurations
4. Implement canary deployment and blue-green deployment strategies for safe model rollouts in production
5. Instrument a model serving endpoint with latency, throughput, and prediction distribution metrics for operational monitoring

---

## 1. Model Serving Concepts

Model serving is providing trained ML models as prediction services in production environments.

### 1.1 Serving Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Serving Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐                                                       │
│   │ Client  │                                                       │
│   │(App/Web)│                                                       │
│   └────┬────┘                                                       │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │    Load     │────▶│    API      │────▶│   Model     │          │
│   │  Balancer   │     │  Gateway    │     │   Server    │          │
│   └─────────────┘     └─────────────┘     └──────┬──────┘          │
│                                                   │                 │
│                             ┌─────────────────────┼────────┐       │
│                             │                     │        │       │
│                             ▼                     ▼        ▼       │
│                       ┌──────────┐          ┌──────────┐   ...     │
│                       │ Model A  │          │ Model B  │           │
│                       │ (v1.2.0) │          │ (v2.0.0) │           │
│                       └──────────┘          └──────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Serving Method Comparison

```python
"""
Model serving methods
"""

# Choose the serving pattern based on the latency budget, not convenience:
# batch trades freshness for cost efficiency (one GPU job for millions of rows),
# online pays per-request infrastructure cost in exchange for sub-second response,
# streaming sits between them for use cases where seconds of lag are acceptable but batch delay is not.
serving_methods = {
    "batch_inference": {
        "description": "Process large amounts of data in bulk",
        "latency": "High (minutes~hours)",
        # Best when results can be pre-computed ahead of need (e.g., overnight churn scores);
        # GPU/CPU utilization is far higher than online serving because there is no idle waiting.
        "use_cases": ["Recommendation system pre-computation", "Report generation", "Data pipelines"],
        "pros": ["High throughput", "Cost efficient"],
        "cons": ["Not real-time", "Data latency"]
    },
    "online_inference": {
        "description": "Real-time request-response",
        "latency": "Low (ms)",
        # Required when the prediction must react to an event happening right now (fraud, search);
        # keeping a model server always-on means paying for idle capacity to guarantee low p99 latency.
        "use_cases": ["Fraud detection", "Search ranking", "Chatbots"],
        "pros": ["Real-time response", "Latest data"],
        "cons": ["Infrastructure cost", "Complex operations"]
    },
    "streaming_inference": {
        "description": "Process continuous data streams",
        "latency": "Medium (seconds)",
        # Unlike batch, streaming processes each event as it arrives rather than accumulating a file;
        # unlike online, it is event-driven so there is no idle server — only triggered compute.
        "use_cases": ["IoT anomaly detection", "Real-time analytics"],
        "pros": ["Continuous processing", "Event-driven"],
        "cons": ["Complex architecture"]
    }
}
```

---

## 2. REST API Deployment

### 2.1 Model Serving with Flask

```python
"""
Flask-based model serving
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

app = Flask(__name__)

# Load model once at startup, not per-request
# Avoids 100-500ms model deserialization overhead on every prediction call;
# joblib.load reads and unpickles the file from disk each time if called inside the handler.
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Parse input data
        data = request.get_json()
        features = data.get("features")

        # Validate input before reaching the model: a missing key here would cause an
        # opaque TypeError deep inside sklearn; explicit 400 responses are debuggable.
        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Preprocessing
        df_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(df_scaled)[0]
        # .tolist() converts numpy float32/float64 to Python float;
        # json.dumps raises TypeError on numpy scalars, so this conversion must happen before serialization.
        probability = model.predict_proba(df_scaled)[0].tolist()

        return jsonify({
            "prediction": int(prediction),  # int() for the same reason: numpy int64 is not JSON-serializable
            "probability": probability,
            "model_version": "1.0.0"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        instances = data.get("instances", [])

        df = pd.DataFrame(instances)
        df_scaled = scaler.transform(df)

        predictions = model.predict(df_scaled).tolist()
        probabilities = model.predict_proba(df_scaled).tolist()

        return jsonify({
            "predictions": predictions,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### 2.2 Model Serving with FastAPI

```python
"""
FastAPI-based model serving (recommended)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI(
    title="ML Model API",
    description="Machine Learning Model Serving API",
    version="1.0.0"
)

# Define Pydantic models for request/response validation
# Pydantic validates types and required fields before the handler runs; this means
# malformed requests are rejected with a structured 422 error before touching the model,
# keeping model code free of defensive type-checking boilerplate.
class PredictionInput(BaseModel):
    """Prediction input schema"""
    age: float = Field(..., description="Customer age")
    tenure: int = Field(..., description="Months as customer")
    monthly_charges: float = Field(..., description="Monthly charges")
    total_charges: float = Field(..., description="Total charges")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35.0,
                "tenure": 24,
                "monthly_charges": 65.5,
                "total_charges": 1572.0
            }
        }

class PredictionOutput(BaseModel):
    """Prediction output schema"""
    prediction: int
    probability: List[float]
    model_version: str

class BatchInput(BaseModel):
    """Batch input schema"""
    instances: List[PredictionInput]

class BatchOutput(BaseModel):
    """Batch output schema"""
    predictions: List[int]
    probabilities: List[List[float]]

# Initialize as None so the /health endpoint can report model_loaded=False
# if startup fails — enables the orchestrator to withhold traffic until the model is ready.
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load model at application startup"""
    global model, scaler
    # Loading here (not per-request) amortizes deserialization cost across all requests;
    # the startup event also blocks the app from accepting traffic until this completes.
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model loaded successfully")

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Single prediction"""
    # 503 (not 500) signals a temporary unavailability; load balancers can retry on 503
    # whereas 500 implies a permanent bug — the distinction matters for upstream retry logic.
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Preprocess and predict
    df_scaled = scaler.transform(df)
    # int() and .tolist() convert numpy types to Python native types;
    # Pydantic's JSON serializer does not handle numpy.int64/float32 and will raise a ValueError.
    prediction = int(model.predict(df_scaled)[0])
    probability = model.predict_proba(df_scaled)[0].tolist()

    return PredictionOutput(
        prediction=prediction,
        probability=probability,
        model_version="1.0.0"
    )

@app.post("/batch_predict", response_model=BatchOutput)
async def batch_predict(input_data: BatchInput):
    """Batch prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    instances = [item.dict() for item in input_data.instances]
    df = pd.DataFrame(instances)

    # Preprocess and predict
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled).tolist()
    probabilities = model.predict_proba(df_scaled).tolist()

    return BatchOutput(
        predictions=predictions,
        probabilities=probabilities
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.3 Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```txt
# requirements.txt
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
scikit-learn==1.3.0
pandas==2.1.0
joblib==1.3.0
numpy==1.24.0
```

```bash
# Build and run
docker build -t ml-model-api:latest .
docker run -p 8000:8000 ml-model-api:latest
```

---

## 3. gRPC Serving

### 3.1 Proto Definition

```protobuf
// prediction.proto
syntax = "proto3";

package prediction;

service PredictionService {
    rpc Predict (PredictRequest) returns (PredictResponse);
    rpc BatchPredict (BatchPredictRequest) returns (BatchPredictResponse);
    rpc HealthCheck (HealthRequest) returns (HealthResponse);
}

message PredictRequest {
    repeated float features = 1;
}

message PredictResponse {
    int32 prediction = 1;
    repeated float probabilities = 2;
    string model_version = 3;
}

message BatchPredictRequest {
    repeated PredictRequest instances = 1;
}

message BatchPredictResponse {
    repeated int32 predictions = 1;
    repeated Probabilities probabilities = 2;
}

message Probabilities {
    repeated float values = 1;
}

message HealthRequest {}

message HealthResponse {
    string status = 1;
    bool model_loaded = 2;
}
```

### 3.2 gRPC Server Implementation

```python
"""
gRPC model server
"""

import grpc
from concurrent import futures
import joblib
import numpy as np

# Code generated from proto
import prediction_pb2
import prediction_pb2_grpc

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    """gRPC service implementation"""

    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.scaler = joblib.load("scaler.pkl")
        print("Model loaded")

    def Predict(self, request, context):
        """Single prediction"""
        features = np.array(request.features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        prediction = int(self.model.predict(features_scaled)[0])
        probabilities = self.model.predict_proba(features_scaled)[0].tolist()

        return prediction_pb2.PredictResponse(
            prediction=prediction,
            probabilities=probabilities,
            model_version="1.0.0"
        )

    def BatchPredict(self, request, context):
        """Batch prediction"""
        features = np.array([list(inst.features) for inst in request.instances])
        features_scaled = self.scaler.transform(features)

        predictions = self.model.predict(features_scaled).tolist()
        probabilities = self.model.predict_proba(features_scaled).tolist()

        prob_messages = [
            prediction_pb2.Probabilities(values=p)
            for p in probabilities
        ]

        return prediction_pb2.BatchPredictResponse(
            predictions=predictions,
            probabilities=prob_messages
        )

    def HealthCheck(self, request, context):
        """Health check"""
        return prediction_pb2.HealthResponse(
            status="healthy",
            model_loaded=True
        )

def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
```

### 3.3 gRPC Client

```python
"""
gRPC client
"""

import grpc
import prediction_pb2
import prediction_pb2_grpc

def predict_single(features: list):
    """Single prediction request"""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = prediction_pb2_grpc.PredictionServiceStub(channel)

        request = prediction_pb2.PredictRequest(features=features)
        response = stub.Predict(request)

        return {
            "prediction": response.prediction,
            "probabilities": list(response.probabilities),
            "model_version": response.model_version
        }

def predict_batch(instances: list):
    """Batch prediction request"""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = prediction_pb2_grpc.PredictionServiceStub(channel)

        request = prediction_pb2.BatchPredictRequest(
            instances=[
                prediction_pb2.PredictRequest(features=inst)
                for inst in instances
            ]
        )
        response = stub.BatchPredict(request)

        return {
            "predictions": list(response.predictions),
            "probabilities": [list(p.values) for p in response.probabilities]
        }

# Example usage
result = predict_single([35.0, 24, 65.5, 1572.0])
print(result)

batch_result = predict_batch([
    [35.0, 24, 65.5, 1572.0],
    [45.0, 36, 85.0, 3060.0]
])
print(batch_result)
```

### 3.4 REST vs gRPC Comparison

```python
"""
REST vs gRPC comparison
"""

comparison = {
    "REST/HTTP": {
        "protocol": "HTTP/1.1 or HTTP/2",
        "data_format": "JSON (text)",
        "schema": "Optional (OpenAPI)",
        "streaming": "Limited",
        "browser_support": "Native",
        "use_cases": "General web APIs, browser clients"
    },
    "gRPC": {
        "protocol": "HTTP/2",
        "data_format": "Protocol Buffers (binary)",
        "schema": "Required (.proto)",
        "streaming": "Bidirectional support",
        "browser_support": "Requires gRPC-Web",
        "use_cases": "Microservices, low latency required"
    }
}

# Performance comparison (general benchmarks)
performance = {
    "latency": {
        "REST": "~50-100ms",
        "gRPC": "~10-30ms"
    },
    "throughput": {
        "REST": "~1000 req/s",
        "gRPC": "~5000 req/s"
    },
    "payload_size": {
        "REST/JSON": "100 bytes",
        "gRPC/Protobuf": "~50 bytes"
    }
}
```

---

## 4. Batch Inference vs Online Inference

### 4.1 Batch Inference

```python
"""
Batch inference pipeline
"""

import pandas as pd
import joblib
from datetime import datetime
import pyarrow.parquet as pq

class BatchInference:
    """Batch inference class"""

    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def run_batch(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 10000
    ):
        """Run batch inference"""
        # Read large data in chunks
        reader = pq.ParquetFile(input_path)

        results = []

        for batch in reader.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            # Extract features
            feature_columns = ["age", "tenure", "monthly_charges", "total_charges"]
            features = df[feature_columns]

            # Preprocess and predict
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)[:, 1]

            # Add results
            df["prediction"] = predictions
            df["probability"] = probabilities
            df["predicted_at"] = datetime.now()

            results.append(df)

        # Save results
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_parquet(output_path, index=False)

        return len(final_df)

# Use with scheduler (e.g., Airflow)
# batch_inference = BatchInference("model.pkl", "scaler.pkl")
# count = batch_inference.run_batch(
#     "s3://bucket/daily_data.parquet",
#     "s3://bucket/predictions.parquet"
# )
```

### 4.2 Online Inference Optimization

```python
"""
Online inference optimization
"""

import numpy as np
from typing import List
from collections import deque
import asyncio
import time

class OptimizedInference:
    """Optimized online inference"""

    def __init__(self, model, batch_size: int = 32, max_wait_ms: int = 10):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue = deque()
        self.result_futures = {}

    async def predict(self, features: List[float]) -> dict:
        """Single prediction (with batching)"""
        request_id = id(features)
        future = asyncio.Future()
        self.result_futures[request_id] = future

        self.request_queue.append((request_id, features))

        # Trigger batch processing
        if len(self.request_queue) >= self.batch_size:
            await self._process_batch()
        else:
            # Process after timeout
            asyncio.create_task(self._delayed_process())

        return await future

    async def _delayed_process(self):
        """Delayed processing"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.request_queue:
            await self._process_batch()

    async def _process_batch(self):
        """Batch processing"""
        if not self.request_queue:
            return

        # Extract requests from queue
        batch = []
        request_ids = []

        while self.request_queue and len(batch) < self.batch_size:
            req_id, features = self.request_queue.popleft()
            batch.append(features)
            request_ids.append(req_id)

        # Batch prediction
        batch_array = np.array(batch)
        predictions = self.model.predict(batch_array)

        # Return results
        for req_id, pred in zip(request_ids, predictions):
            if req_id in self.result_futures:
                self.result_futures[req_id].set_result({"prediction": int(pred)})
                del self.result_futures[req_id]
```

### 4.3 Model Caching

```python
"""
Model and result caching
"""

from functools import lru_cache
import hashlib
import redis
import json
import pickle

class CachedInference:
    """Inference with caching"""

    def __init__(self, model, redis_client: redis.Redis):
        self.model = model
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour

    def _get_cache_key(self, features: tuple) -> str:
        """Generate cache key"""
        features_str = json.dumps(features, sort_keys=True)
        return f"pred:{hashlib.md5(features_str.encode()).hexdigest()}"

    def predict_with_cache(self, features: list) -> dict:
        """Cached prediction"""
        cache_key = self._get_cache_key(tuple(features))

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Perform prediction
        prediction = int(self.model.predict([features])[0])
        probability = self.model.predict_proba([features])[0].tolist()

        result = {
            "prediction": prediction,
            "probability": probability
        }

        # Save to cache
        self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(result)
        )

        return result

# In-memory LRU cache (for simple cases)
@lru_cache(maxsize=10000)
def predict_cached(features_tuple: tuple) -> dict:
    """LRU cached prediction"""
    features = list(features_tuple)
    prediction = model.predict([features])[0]
    return {"prediction": int(prediction)}
```

---

## 5. Serving Infrastructure

### 5.1 Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
        - name: api
          image: ml-model-api:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            - name: MODEL_PATH
              value: "/models/model.pkl"
          volumeMounts:
            - name: model-volume
              mountPath: /models
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 5.2 Load Balancing

```python
"""
Serving load balancing strategies
"""

load_balancing_strategies = {
    "round_robin": {
        "description": "Distribute requests sequentially",
        "use_case": "Uniform request processing time"
    },
    "least_connections": {
        "description": "Route to server with fewest connections",
        "use_case": "Varying request processing times"
    },
    "ip_hash": {
        "description": "Fixed server based on client IP",
        "use_case": "Session persistence required"
    },
    "weighted": {
        "description": "Weighted distribution by server capacity",
        "use_case": "Servers with different specs"
    }
}
```

---

## 6. Monitoring and Logging

```python
"""
Serving monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total predictions",
    ["model_version", "status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time to load model"
)

# Add metrics to prediction function
def predict_with_metrics(features):
    start_time = time.time()

    try:
        result = model.predict([features])
        PREDICTION_COUNT.labels(
            model_version="1.0.0",
            status="success"
        ).inc()
        return result

    except Exception as e:
        PREDICTION_COUNT.labels(
            model_version="1.0.0",
            status="error"
        ).inc()
        raise

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

# Start Prometheus server
start_http_server(9090)
```

---

## Practice Exercises

### Exercise 1: FastAPI Serving
Create a complete API serving a scikit-learn model with FastAPI.

### Exercise 2: Docker Containerization
Build and run the API as a Docker image.

### Exercise 3: Performance Testing
Measure API performance using locust or wrk.

---

## Summary

| Method | Advantages | Disadvantages | Use Cases |
|--------|-----------|---------------|-----------|
| REST API | Simple, universal | Relatively high latency | General web services |
| gRPC | Low latency, high throughput | Complexity | Microservices |
| Batch Inference | Efficient, cost-effective | Not real-time | Large-scale data processing |
| Online Inference | Immediate response | Infrastructure cost | Fraud detection, recommendations |

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [Kubernetes ML Serving](https://kubernetes.io/docs/concepts/workloads/)

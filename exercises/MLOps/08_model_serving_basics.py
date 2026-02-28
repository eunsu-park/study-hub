"""
Exercise Solutions: Model Serving Basics
===========================================
Lesson 08 from MLOps topic.

Exercises
---------
1. FastAPI Serving — Implement a model serving API with health checks,
   input validation, batch prediction, and response formatting.
2. Docker Containerization — Generate Dockerfile and docker-compose
   configurations for a model serving deployment.
3. Performance Testing — Simulate load testing with concurrent requests,
   measure latency distributions, and identify bottlenecks.
"""

import math
import random
import json
import time
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Shared: Simple Model for Serving
# ============================================================

class SimpleModel:
    """A trained logistic regression model ready for serving."""

    def __init__(self, weights, bias, feature_names):
        self.weights = weights
        self.bias = bias
        self.feature_names = feature_names
        self.version = "1.0.0"
        self.loaded_at = datetime.now()

    def predict(self, features):
        """Single prediction."""
        z = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        z = max(-500, min(500, z))
        probability = 1 / (1 + math.exp(-z))
        return {
            "prediction": 1 if probability >= 0.5 else 0,
            "probability": round(probability, 6),
            "label": "positive" if probability >= 0.5 else "negative",
        }

    def predict_batch(self, batch):
        """Batch prediction."""
        return [self.predict(features) for features in batch]


# Create a pre-trained model
def get_model():
    random.seed(42)
    return SimpleModel(
        weights=[0.5, -0.3, 0.8, 0.1, -0.6, 0.4, -0.2, 0.7],
        bias=-0.1,
        feature_names=["tenure", "monthly_charges", "total_charges", "contract_type",
                       "payment_method", "tech_support", "online_security", "paperless_billing"],
    )


# ============================================================
# Exercise 1: FastAPI Serving
# ============================================================

def exercise_1_fastapi_serving():
    """Implement a model serving API with FastAPI-style design.

    We simulate the FastAPI application structure without requiring FastAPI:
    - Health check endpoint
    - Single prediction endpoint with input validation
    - Batch prediction endpoint
    - Model metadata endpoint
    - Request/response formatting with proper error handling
    """

    class ModelServingApp:
        """Simulated FastAPI model serving application."""

        def __init__(self, model):
            self.model = model
            self.request_count = 0
            self.error_count = 0
            self.start_time = datetime.now()
            self.routes = {
                "GET /health": self.health_check,
                "GET /model/info": self.model_info,
                "POST /predict": self.predict,
                "POST /predict/batch": self.predict_batch,
            }

        def _validate_features(self, features):
            """Validate input features."""
            errors = []
            if not isinstance(features, list):
                errors.append("features must be a list")
                return errors

            if len(features) != len(self.model.feature_names):
                errors.append(
                    f"Expected {len(self.model.feature_names)} features, "
                    f"got {len(features)}"
                )
                return errors

            for i, (val, name) in enumerate(zip(features, self.model.feature_names)):
                if not isinstance(val, (int, float)):
                    errors.append(f"Feature '{name}' (index {i}) must be numeric, got {type(val).__name__}")
                elif math.isnan(val) or math.isinf(val):
                    errors.append(f"Feature '{name}' (index {i}) contains NaN or Inf")

            return errors

        def health_check(self, request=None):
            """GET /health — Liveness and readiness check."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_version": self.model.version,
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.request_count,
                "error_rate": round(self.error_count / max(1, self.request_count), 4),
            }

        def model_info(self, request=None):
            """GET /model/info — Model metadata."""
            return {
                "model_version": self.model.version,
                "algorithm": "logistic_regression",
                "feature_names": self.model.feature_names,
                "n_features": len(self.model.feature_names),
                "loaded_at": self.model.loaded_at.isoformat(),
                "output_format": {
                    "prediction": "int (0 or 1)",
                    "probability": "float [0, 1]",
                    "label": "str ('positive' or 'negative')",
                },
            }

        def predict(self, request):
            """POST /predict — Single prediction with validation."""
            self.request_count += 1
            start = time.time()

            # Validate request
            features = request.get("features")
            if features is None:
                self.error_count += 1
                return {"error": "Missing 'features' field", "status_code": 400}

            errors = self._validate_features(features)
            if errors:
                self.error_count += 1
                return {"error": errors, "status_code": 422}

            # Run prediction
            result = self.model.predict(features)
            latency_ms = (time.time() - start) * 1000

            return {
                "status_code": 200,
                "prediction": result["prediction"],
                "probability": result["probability"],
                "label": result["label"],
                "model_version": self.model.version,
                "latency_ms": round(latency_ms, 3),
            }

        def predict_batch(self, request):
            """POST /predict/batch — Batch prediction."""
            self.request_count += 1
            start = time.time()

            instances = request.get("instances")
            if not instances:
                self.error_count += 1
                return {"error": "Missing 'instances' field", "status_code": 400}

            if len(instances) > 100:
                self.error_count += 1
                return {"error": "Batch size exceeds maximum of 100", "status_code": 400}

            # Validate all instances
            for i, inst in enumerate(instances):
                errors = self._validate_features(inst)
                if errors:
                    self.error_count += 1
                    return {
                        "error": f"Validation failed for instance {i}: {errors}",
                        "status_code": 422,
                    }

            # Run batch prediction
            results = self.model.predict_batch(instances)
            latency_ms = (time.time() - start) * 1000

            return {
                "status_code": 200,
                "predictions": results,
                "batch_size": len(instances),
                "model_version": self.model.version,
                "latency_ms": round(latency_ms, 3),
            }

        def handle_request(self, method, path, body=None):
            """Route a request to the appropriate handler."""
            route_key = f"{method} {path}"
            handler = self.routes.get(route_key)
            if not handler:
                return {"error": f"Route not found: {route_key}", "status_code": 404}
            return handler(body)

    # --- Test the API ---
    model = get_model()
    app = ModelServingApp(model)

    print("FastAPI Model Serving Simulation")
    print("=" * 60)

    # Health check
    print("\n1. GET /health")
    print("-" * 40)
    response = app.handle_request("GET", "/health")
    print(f"  {json.dumps(response, indent=4)}")

    # Model info
    print("\n2. GET /model/info")
    print("-" * 40)
    response = app.handle_request("GET", "/model/info")
    print(f"  {json.dumps(response, indent=4)}")

    # Single prediction
    print("\n3. POST /predict (valid request)")
    print("-" * 40)
    response = app.handle_request("POST", "/predict", {
        "features": [24, 65.5, 1570.0, 1, 0, 1, 0, 1]
    })
    print(f"  {json.dumps(response, indent=4)}")

    # Invalid prediction
    print("\n4. POST /predict (invalid — wrong feature count)")
    print("-" * 40)
    response = app.handle_request("POST", "/predict", {
        "features": [24, 65.5, 1570.0]  # Only 3 features
    })
    print(f"  {json.dumps(response, indent=4)}")

    # Batch prediction
    print("\n5. POST /predict/batch")
    print("-" * 40)
    random.seed(42)
    batch = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
    response = app.handle_request("POST", "/predict/batch", {"instances": batch})
    print(f"  Batch size: {response.get('batch_size', 'N/A')}")
    print(f"  Latency: {response.get('latency_ms', 'N/A')}ms")
    for i, pred in enumerate(response.get("predictions", [])):
        print(f"    Instance {i}: {pred['label']} (prob={pred['probability']:.4f})")

    # Final health check shows updated stats
    print("\n6. GET /health (after requests)")
    print("-" * 40)
    response = app.handle_request("GET", "/health")
    print(f"  {json.dumps(response, indent=4)}")

    return app


# ============================================================
# Exercise 2: Docker Containerization
# ============================================================

def exercise_2_docker_containerization():
    """Generate Dockerfile and docker-compose for model serving.

    Demonstrates:
    - Multi-stage Docker build for smaller images
    - Health checks in Docker
    - Docker Compose with model server + monitoring
    - Environment variable configuration
    """

    dockerfile = """# ====================================
# Multi-stage Dockerfile for ML Model Serving
# ====================================

# Stage 1: Builder — install dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install dependencies first (leverages Docker cache)
# Why separate COPY for requirements? If only code changes (not dependencies),
# Docker reuses the cached layer, saving ~2-3 minutes on rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime — minimal image
FROM python:3.11-slim as runtime

# Security: run as non-root user
# Why? If the application is compromised, the attacker has limited privileges.
RUN groupadd -r mluser && useradd -r -g mluser mluser

WORKDIR /app

# Copy only installed packages from builder (not build tools)
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY config.py .

# Set environment variables
ENV MODEL_PATH=/app/models/model.pkl
ENV PORT=8080
ENV WORKERS=4
ENV LOG_LEVEL=info

# Health check — Docker will restart container if this fails 3 times
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose the serving port
EXPOSE ${PORT}

# Switch to non-root user
USER mluser

# Why uvicorn with multiple workers?
# - Each worker handles requests independently
# - 4 workers on a 4-core machine = ~4x throughput
# - --limit-concurrency prevents memory exhaustion under load
ENTRYPOINT ["uvicorn", "app.main:app", \\
    "--host", "0.0.0.0", \\
    "--port", "8080", \\
    "--workers", "4", \\
    "--limit-concurrency", "100", \\
    "--access-log"]
"""

    docker_compose = """# ====================================
# Docker Compose: Model Serving Stack
# ====================================
version: '3.8'

services:
  # ML Model Server
  model-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - LOG_LEVEL=info
      - WORKERS=4
    volumes:
      - model-storage:/app/models
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    depends_on:
      model-server:
        condition: service_healthy

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  # Nginx as reverse proxy + load balancer
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      model-server:
        condition: service_healthy

volumes:
  model-storage:
  prometheus-data:
  grafana-data:
"""

    requirements_txt = """fastapi==0.109.0
uvicorn[standard]==0.27.0
numpy==1.26.3
scikit-learn==1.4.0
joblib==1.3.2
prometheus-client==0.19.0
python-json-logger==2.0.7
pydantic==2.5.3
"""

    print("Docker Containerization")
    print("=" * 60)

    print("\n1. Dockerfile (multi-stage build)")
    print("-" * 40)
    print(dockerfile)

    print("\n2. docker-compose.yml")
    print("-" * 40)
    print(docker_compose)

    print("\n3. requirements.txt")
    print("-" * 40)
    print(requirements_txt)

    # Build analysis
    print("\n4. Build Analysis")
    print("-" * 40)
    print("  Multi-stage build benefits:")
    print("    - Builder stage: ~800MB (includes pip, setuptools, compilers)")
    print("    - Runtime stage: ~250MB (only Python runtime + packages)")
    print("    - Savings: ~550MB per image")
    print()
    print("  Security features:")
    print("    - Non-root user (mluser)")
    print("    - Health checks with auto-restart")
    print("    - Resource limits (CPU/memory)")
    print("    - Read-only code (model volume for updates)")
    print()
    print("  Deployment commands:")
    print("    docker compose up -d             # Start all services")
    print("    docker compose ps                 # Check status")
    print("    docker compose logs model-server  # View logs")
    print("    docker compose down               # Stop all services")

    return {"dockerfile": dockerfile, "compose": docker_compose}


# ============================================================
# Exercise 3: Performance Testing
# ============================================================

def exercise_3_performance_testing():
    """Simulate load testing with concurrent requests.

    Measures:
    - Throughput (requests/second)
    - Latency distribution (p50, p90, p95, p99)
    - Error rate under load
    - Identification of bottlenecks
    """

    model = get_model()

    # Simulated server with artificial latency and capacity limits
    class SimulatedServer:
        def __init__(self, model, max_concurrent=10, base_latency_ms=5):
            self.model = model
            self.max_concurrent = max_concurrent
            self.base_latency_ms = base_latency_ms
            self.active_requests = 0
            self.total_requests = 0
            self.errors = 0

        def handle_request(self, features):
            """Handle a single prediction request with simulated latency."""
            self.active_requests += 1
            self.total_requests += 1

            # Simulate latency that increases with load
            load_factor = self.active_requests / self.max_concurrent
            if load_factor > 1.0:
                # Over capacity — high chance of timeout/error
                if random.random() < 0.3:
                    self.active_requests -= 1
                    self.errors += 1
                    return {"error": "Server overloaded", "latency_ms": 5000}

            # Latency increases quadratically with load
            latency = self.base_latency_ms * (1 + load_factor ** 2)
            latency += random.gauss(0, latency * 0.1)  # Jitter
            latency = max(1, latency)

            # Simulate processing time
            time.sleep(latency / 1000)

            result = self.model.predict(features)
            self.active_requests -= 1

            return {**result, "latency_ms": round(latency, 2)}

    def run_load_test(server, num_requests, concurrency, label=""):
        """Run a load test with specified concurrency."""
        random.seed(42)
        test_inputs = [
            [random.gauss(0, 1) for _ in range(8)]
            for _ in range(num_requests)
        ]

        latencies = []
        errors = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(server.handle_request, inp)
                for inp in test_inputs
            ]
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors += 1
                latencies.append(result["latency_ms"])

        elapsed = time.time() - start_time
        throughput = num_requests / elapsed

        latencies.sort()
        stats = {
            "label": label,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "duration_s": round(elapsed, 2),
            "throughput_rps": round(throughput, 1),
            "errors": errors,
            "error_rate": round(errors / num_requests, 4),
            "latency_p50": round(latencies[int(len(latencies) * 0.50)], 2),
            "latency_p90": round(latencies[int(len(latencies) * 0.90)], 2),
            "latency_p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "latency_p99": round(latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)], 2),
            "latency_mean": round(statistics.mean(latencies), 2),
            "latency_std": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2),
        }
        return stats

    # --- Run tests at different concurrency levels ---
    print("Performance Load Testing")
    print("=" * 60)

    test_configs = [
        {"num_requests": 50, "concurrency": 1,  "label": "Single thread"},
        {"num_requests": 50, "concurrency": 5,  "label": "5 concurrent"},
        {"num_requests": 50, "concurrency": 10, "label": "10 concurrent"},
        {"num_requests": 50, "concurrency": 20, "label": "20 concurrent (overload)"},
    ]

    all_results = []
    for config in test_configs:
        server = SimulatedServer(model, max_concurrent=10, base_latency_ms=5)
        result = run_load_test(server, **config)
        all_results.append(result)

    # --- Display results ---
    print(f"\n{'Test':<25s} {'RPS':>6s} {'p50':>7s} {'p90':>7s} {'p95':>7s} "
          f"{'p99':>7s} {'Errors':>7s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<25s} {r['throughput_rps']:>6.1f} "
              f"{r['latency_p50']:>6.1f}ms {r['latency_p90']:>6.1f}ms "
              f"{r['latency_p95']:>6.1f}ms {r['latency_p99']:>6.1f}ms "
              f"{r['error_rate']:>6.1%}")

    # --- Analysis ---
    print(f"\nPerformance Analysis:")
    print("-" * 40)

    # Find saturation point
    for i in range(1, len(all_results)):
        prev = all_results[i - 1]
        curr = all_results[i]
        if curr["latency_p99"] > prev["latency_p99"] * 2:
            print(f"  Saturation point: between {prev['concurrency']} and "
                  f"{curr['concurrency']} concurrent requests")
            break

    # Identify bottlenecks
    overload = all_results[-1]
    if overload["error_rate"] > 0:
        print(f"  Bottleneck: Server capacity ({overload['error_rate']:.1%} error rate at "
              f"{overload['concurrency']} concurrent)")
    if overload["latency_p99"] > 100:
        print(f"  Bottleneck: Latency spikes ({overload['latency_p99']:.0f}ms p99)")

    print(f"\n  Recommendations:")
    print(f"    - Set max concurrency to ~{all_results[-2]['concurrency']} to avoid overload")
    print(f"    - Target SLO: p99 < {all_results[1]['latency_p99'] * 2:.0f}ms")
    print(f"    - Scale horizontally for > {all_results[1]['throughput_rps']:.0f} RPS")

    return all_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: FastAPI Serving")
    print("=" * 60)
    exercise_1_fastapi_serving()

    print("\n\n")
    print("Exercise 2: Docker Containerization")
    print("=" * 60)
    exercise_2_docker_containerization()

    print("\n\n")
    print("Exercise 3: Performance Testing")
    print("=" * 60)
    exercise_3_performance_testing()

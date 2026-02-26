# 모델 서빙 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 배치 추론(batch inference), 온라인 추론(online inference), 스트리밍 추론(streaming inference)의 세 가지 모델 서빙 패러다임을 설명하고 주어진 사용 사례에 적합한 패턴을 선택할 수 있다
2. FastAPI를 사용하여 입력 유효성 검사(input validation)와 오류 처리를 갖춘 예측 엔드포인트(prediction endpoint)를 구현한 REST API 모델 서버를 구축할 수 있다
3. 적절한 리소스 할당(resource allocation)과 헬스 체크(health check) 설정으로 Docker를 사용하여 모델 서버를 컨테이너화하고 배포할 수 있다
4. 프로덕션에서 안전한 모델 롤아웃을 위한 카나리 배포(canary deployment)와 블루-그린 배포(blue-green deployment) 전략을 구현할 수 있다
5. 운영 모니터링을 위해 모델 서빙 엔드포인트에 지연 시간(latency), 처리량(throughput), 예측 분포(prediction distribution) 메트릭을 추가할 수 있다

---

## 1. 모델 서빙 개념

모델 서빙은 학습된 ML 모델을 프로덕션 환경에서 예측 서비스로 제공하는 것입니다.

### 1.1 서빙 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                      모델 서빙 아키텍처                               │
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

### 1.2 서빙 방식 비교

```python
"""
모델 서빙 방식
"""

serving_methods = {
    "batch_inference": {
        "description": "대량 데이터를 일괄 처리",
        "latency": "높음 (분~시간)",
        "use_cases": ["추천 시스템 사전 계산", "리포트 생성", "데이터 파이프라인"],
        "pros": ["높은 처리량", "비용 효율적"],
        "cons": ["실시간 불가", "데이터 지연"]
    },
    "online_inference": {
        "description": "실시간 요청-응답",
        "latency": "낮음 (ms)",
        "use_cases": ["사기 탐지", "검색 랭킹", "챗봇"],
        "pros": ["실시간 응답", "최신 데이터"],
        "cons": ["인프라 비용", "복잡한 운영"]
    },
    "streaming_inference": {
        "description": "연속 데이터 스트림 처리",
        "latency": "중간 (초)",
        "use_cases": ["IoT 이상 탐지", "실시간 분석"],
        "pros": ["연속 처리", "이벤트 기반"],
        "cons": ["복잡한 아키텍처"]
    }
}
```

---

## 2. REST API 배포

### 2.1 Flask로 모델 서빙

```python
"""
Flask 기반 모델 서빙
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

app = Flask(__name__)

# 모델 로드 (애플리케이션 시작 시)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/health", methods=["GET"])
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """예측 엔드포인트"""
    try:
        # 입력 데이터 파싱
        data = request.get_json()
        features = data.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

        # DataFrame 변환
        df = pd.DataFrame([features])

        # 전처리
        df_scaled = scaler.transform(df)

        # 예측
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probability": probability,
            "model_version": "1.0.0"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """배치 예측 엔드포인트"""
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

### 2.2 FastAPI로 모델 서빙

```python
"""
FastAPI 기반 모델 서빙 (권장)
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

# Pydantic 모델 정의
class PredictionInput(BaseModel):
    """예측 입력 스키마"""
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
    """예측 출력 스키마"""
    prediction: int
    probability: List[float]
    model_version: str

class BatchInput(BaseModel):
    """배치 입력 스키마"""
    instances: List[PredictionInput]

class BatchOutput(BaseModel):
    """배치 출력 스키마"""
    predictions: List[int]
    probabilities: List[List[float]]

# 모델 로드
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """애플리케이션 시작 시 모델 로드"""
    global model, scaler
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model loaded successfully")

@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """단일 예측"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 입력을 DataFrame으로 변환
    df = pd.DataFrame([input_data.dict()])

    # 전처리 및 예측
    df_scaled = scaler.transform(df)
    prediction = int(model.predict(df_scaled)[0])
    probability = model.predict_proba(df_scaled)[0].tolist()

    return PredictionOutput(
        prediction=prediction,
        probability=probability,
        model_version="1.0.0"
    )

@app.post("/batch_predict", response_model=BatchOutput)
async def batch_predict(input_data: BatchInput):
    """배치 예측"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # DataFrame 변환
    instances = [item.dict() for item in input_data.instances]
    df = pd.DataFrame(instances)

    # 전처리 및 예측
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

### 2.3 Docker 컨테이너화

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .

# 포트 노출
EXPOSE 8000

# 실행
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
# 빌드 및 실행
docker build -t ml-model-api:latest .
docker run -p 8000:8000 ml-model-api:latest
```

---

## 3. gRPC 서빙

### 3.1 Proto 정의

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

### 3.2 gRPC 서버 구현

```python
"""
gRPC 모델 서버
"""

import grpc
from concurrent import futures
import joblib
import numpy as np

# Proto에서 생성된 코드
import prediction_pb2
import prediction_pb2_grpc

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    """gRPC 서비스 구현"""

    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.scaler = joblib.load("scaler.pkl")
        print("Model loaded")

    def Predict(self, request, context):
        """단일 예측"""
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
        """배치 예측"""
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
        """헬스 체크"""
        return prediction_pb2.HealthResponse(
            status="healthy",
            model_loaded=True
        )

def serve():
    """gRPC 서버 시작"""
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

### 3.3 gRPC 클라이언트

```python
"""
gRPC 클라이언트
"""

import grpc
import prediction_pb2
import prediction_pb2_grpc

def predict_single(features: list):
    """단일 예측 요청"""
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
    """배치 예측 요청"""
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

# 사용 예시
result = predict_single([35.0, 24, 65.5, 1572.0])
print(result)

batch_result = predict_batch([
    [35.0, 24, 65.5, 1572.0],
    [45.0, 36, 85.0, 3060.0]
])
print(batch_result)
```

### 3.4 REST vs gRPC 비교

```python
"""
REST vs gRPC 비교
"""

comparison = {
    "REST/HTTP": {
        "protocol": "HTTP/1.1 or HTTP/2",
        "data_format": "JSON (텍스트)",
        "schema": "선택적 (OpenAPI)",
        "streaming": "제한적",
        "browser_support": "네이티브",
        "use_cases": "일반 웹 API, 브라우저 클라이언트"
    },
    "gRPC": {
        "protocol": "HTTP/2",
        "data_format": "Protocol Buffers (바이너리)",
        "schema": "필수 (.proto)",
        "streaming": "양방향 지원",
        "browser_support": "gRPC-Web 필요",
        "use_cases": "마이크로서비스, 낮은 지연시간 필요"
    }
}

# 성능 비교 (일반적 기준)
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

## 4. 배치 추론 vs 실시간 추론

### 4.1 배치 추론

```python
"""
배치 추론 파이프라인
"""

import pandas as pd
import joblib
from datetime import datetime
import pyarrow.parquet as pq

class BatchInference:
    """배치 추론 클래스"""

    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def run_batch(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 10000
    ):
        """배치 추론 실행"""
        # 대용량 데이터 청크 단위 읽기
        reader = pq.ParquetFile(input_path)

        results = []

        for batch in reader.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            # 피처 추출
            feature_columns = ["age", "tenure", "monthly_charges", "total_charges"]
            features = df[feature_columns]

            # 전처리 및 예측
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)[:, 1]

            # 결과 추가
            df["prediction"] = predictions
            df["probability"] = probabilities
            df["predicted_at"] = datetime.now()

            results.append(df)

        # 결과 저장
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_parquet(output_path, index=False)

        return len(final_df)

# 스케줄러와 함께 사용 (예: Airflow)
# batch_inference = BatchInference("model.pkl", "scaler.pkl")
# count = batch_inference.run_batch(
#     "s3://bucket/daily_data.parquet",
#     "s3://bucket/predictions.parquet"
# )
```

### 4.2 실시간 추론 최적화

```python
"""
실시간 추론 최적화
"""

import numpy as np
from typing import List
from collections import deque
import asyncio
import time

class OptimizedInference:
    """최적화된 실시간 추론"""

    def __init__(self, model, batch_size: int = 32, max_wait_ms: int = 10):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue = deque()
        self.result_futures = {}

    async def predict(self, features: List[float]) -> dict:
        """단일 예측 (배칭 포함)"""
        request_id = id(features)
        future = asyncio.Future()
        self.result_futures[request_id] = future

        self.request_queue.append((request_id, features))

        # 배치 처리 트리거
        if len(self.request_queue) >= self.batch_size:
            await self._process_batch()
        else:
            # 타임아웃 후 처리
            asyncio.create_task(self._delayed_process())

        return await future

    async def _delayed_process(self):
        """지연 처리"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.request_queue:
            await self._process_batch()

    async def _process_batch(self):
        """배치 처리"""
        if not self.request_queue:
            return

        # 큐에서 요청 추출
        batch = []
        request_ids = []

        while self.request_queue and len(batch) < self.batch_size:
            req_id, features = self.request_queue.popleft()
            batch.append(features)
            request_ids.append(req_id)

        # 배치 예측
        batch_array = np.array(batch)
        predictions = self.model.predict(batch_array)

        # 결과 반환
        for req_id, pred in zip(request_ids, predictions):
            if req_id in self.result_futures:
                self.result_futures[req_id].set_result({"prediction": int(pred)})
                del self.result_futures[req_id]
```

### 4.3 모델 캐싱

```python
"""
모델 및 결과 캐싱
"""

from functools import lru_cache
import hashlib
import redis
import json
import pickle

class CachedInference:
    """캐시가 포함된 추론"""

    def __init__(self, model, redis_client: redis.Redis):
        self.model = model
        self.redis = redis_client
        self.cache_ttl = 3600  # 1시간

    def _get_cache_key(self, features: tuple) -> str:
        """캐시 키 생성"""
        features_str = json.dumps(features, sort_keys=True)
        return f"pred:{hashlib.md5(features_str.encode()).hexdigest()}"

    def predict_with_cache(self, features: list) -> dict:
        """캐시된 예측"""
        cache_key = self._get_cache_key(tuple(features))

        # 캐시 확인
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 예측 수행
        prediction = int(self.model.predict([features])[0])
        probability = self.model.predict_proba([features])[0].tolist()

        result = {
            "prediction": prediction,
            "probability": probability
        }

        # 캐시 저장
        self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(result)
        )

        return result

# 인메모리 LRU 캐시 (간단한 경우)
@lru_cache(maxsize=10000)
def predict_cached(features_tuple: tuple) -> dict:
    """LRU 캐시된 예측"""
    features = list(features_tuple)
    prediction = model.predict([features])[0]
    return {"prediction": int(prediction)}
```

---

## 5. 서빙 인프라

### 5.1 Kubernetes 배포

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

### 5.2 로드 밸런싱

```python
"""
서빙 로드 밸런싱 전략
"""

load_balancing_strategies = {
    "round_robin": {
        "description": "요청을 순차적으로 분배",
        "use_case": "균일한 요청 처리 시간"
    },
    "least_connections": {
        "description": "연결이 가장 적은 서버로 분배",
        "use_case": "요청 처리 시간이 다양한 경우"
    },
    "ip_hash": {
        "description": "클라이언트 IP 기반 고정 서버",
        "use_case": "세션 유지 필요"
    },
    "weighted": {
        "description": "서버 용량에 따른 가중치 분배",
        "use_case": "서버 스펙이 다른 경우"
    }
}
```

---

## 6. 모니터링 및 로깅

```python
"""
서빙 모니터링
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Prometheus 메트릭 정의
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

# 예측 함수에 메트릭 추가
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

# Prometheus 서버 시작
start_http_server(9090)
```

---

## 연습 문제

### 문제 1: FastAPI 서빙
scikit-learn 모델을 FastAPI로 서빙하는 완전한 API를 작성하세요.

### 문제 2: Docker 컨테이너화
작성한 API를 Docker 이미지로 빌드하고 실행하세요.

### 문제 3: 성능 테스트
locust 또는 wrk를 사용하여 API 성능을 측정하세요.

---

## 요약

| 방식 | 장점 | 단점 | 사용 사례 |
|------|------|------|----------|
| REST API | 간단, 범용 | 상대적 고지연 | 일반 웹 서비스 |
| gRPC | 저지연, 고처리량 | 복잡성 | 마이크로서비스 |
| 배치 추론 | 효율적, 비용 절감 | 실시간 불가 | 대량 데이터 처리 |
| 실시간 추론 | 즉시 응답 | 인프라 비용 | 사기 탐지, 추천 |

---

## 참고 자료

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [Kubernetes ML Serving](https://kubernetes.io/docs/concepts/workloads/)

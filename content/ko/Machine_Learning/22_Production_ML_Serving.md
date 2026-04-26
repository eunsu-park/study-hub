# 프로덕션 ML — 모델 서빙 패턴(Model Serving Patterns)

[← 이전: 21. 고급 앙상블 기법](21_Advanced_Ensemble.md) | [다음: 23. ML을 위한 A/B 테스트 →](23_AB_Testing_for_ML.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 학습 환경과 프로덕션 서빙 환경 간의 격차를 식별하고 일반적인 함정 나열
2. 모델 최적화 기법(양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), ONNX 내보내기)을 적용하여 추론 지연시간과 모델 크기 감소
3. 비즈니스 요구사항과 지연시간 제약 조건에 따라 적절한 서빙 패턴(배치, 실시간, 스트리밍) 선택
4. 학습-서빙 왜곡(Training-Serving Skew)을 방지하는 전처리 파이프라인 설계
5. joblib, ONNX, TorchScript를 사용하여 scikit-learn 및 PyTorch 모델을 프로덕션 배포용으로 패키징
6. 데이터 드리프트(Data Drift)와 개념 드리프트(Concept Drift)를 감지하기 위한 모니터링 전략 정의 (ML 관점)

---

주피터 노트북에서 95% 정확도를 달성하는 모델과, 10,000명의 동시 사용자에게 50ms 지연시간으로 95% 정확한 예측을 제공하는 모델은 같지 않습니다. "내 노트북에서는 돌아가는데"에서 "프로덕션에서 안정적으로 운영된다"까지의 여정은 대부분의 ML 프로젝트가 정체되는 구간입니다. 이 레슨은 그 여정의 **ML 측면**에 집중합니다 — 데이터 과학자로서 모델을 프로덕션에 맞게 준비하는 방법을 다루며, 인프라 측면(FastAPI, TorchServe, Kubernetes)은 MLOps 토픽에서 다룹니다. 이렇게 생각하면 됩니다: MLOps 엔지니어가 고속도로를 건설하고, 이 레슨은 실제로 그 도로 위를 달릴 수 있는 자동차를 만드는 방법을 알려줍니다.

---

> **비유**: ML 모델을 개발하는 것은 작업장에서 시제품 레이싱카를 만드는 것과 같습니다. 테스트 트랙(검증 세트)에서는 아름답게 달립니다. 하지만 실제 트랙에서 경주하려면, 연비를 위해 엔진을 최적화하고(양자화), 불필요한 무게를 줄이고(가지치기), 타이어가 다양한 노면에서 작동하는지 확인하고(분포 변화 처리), 모니터링 계기판을 설치해야 합니다(드리프트 감지). 그때서야 비로소 레이스에 나갈 준비가 된 것입니다.

---

## 이론과 원리

프로덕션 ML 서빙은 학습과는 다른 최적화 문제입니다. 학습 지표는 "검증 정확도"; 프로덕션 지표는 정확도, p99 지연시간, 처리량, 비용, 운영 위험의 가중 결합. 이 레슨의 각 기법 — 모델 버전 관리, 양자화, A/B 라우팅, 드리프트 탐지 — 은 이 다목적 공간의 손잡이이며, 모델이 노트북을 떠난 후에만 중요한 날카로운 트레이드오프를 가집니다.

### A. 지연시간 vs 처리량: 두 성능 차원

이 둘은 종종 혼동되지만 독립적:

- **지연시간(Latency)**: 단일 요청이 도착해서 응답이 떠날 때까지의 시간. p50, p95, p99 — 분포의 중앙값, 95번째, 99번째 백분위수로 보고. 요청당 컴퓨팅과 직렬화 비용에 의해 구동.
- **처리량(Throughput)**: 초당 처리되는 총 요청 수. 병렬성에 의해 구동 — 배치 처리, 다중 복제본, GPU 활용.

구별이 중요한 이유는 레버가 다르기 때문:

- **지연시간** 개선: 더 작은 모델(양자화, 증류), 더 빠른 전처리(핫 패스에서 Python 피하기), 네트워크 홉 피하기, 워밍 캐시.
- **처리량** 개선: 동적 배치(동시 요청을 그룹화하여 한 번의 더 큰 정방향 패스 실행), 로드 밸런서 뒤의 더 많은 복제본, 비동기 I/O.

일부 최적화는 한쪽을 돕고 다른 쪽을 해침. 20ms 대기 윈도우의 동적 배치는 요청당 지연시간에 ~20ms를 추가하는 비용으로 처리량을 개선. 올바른 선택은 어떤 제약이 구속적인지에 의존(대화형 추천기는 낮은 지연시간 필요; 야간 배치 점수 작업은 처리량만 신경).

### B. p99 함정

"평균 지연시간" 보고는 중요한 모든 것을 숨김. 평균 지연시간 50ms 모델이 p99가 2초일 수 있음 — 즉 사용자의 1%가 끔찍한 경험. SLA는 평균이 아니라 꼬리 백분위수로 작성.

꼬리 지연시간의 원천:
- 가비지 수집 일시 정지(Python, JVM).
- 새로 배포된 복제본의 콜드 모델 로딩.
- 공동 위치 워크로드와의 자원 경합.
- 가변 입력 크기(10배 더 긴 텍스트가 임베딩에 10배 더 걸림).

완화: 워밍 풀(복제본을 따뜻하게 유지), 패딩을 가진 고정 배치 크기, 신중한 자원 격리, 입력 길이 제한. 항상 평균이 아니라 p99 측정.

### C. 모델 버전 관리와 재현성 계약

프로덕션 모델은 단일 파일이 아닙니다 — 여러 아티팩트에 대한 *계약*입니다:
- 직렬화된 모델(가중치 / 트리 구조 / 계수).
- 정확한 전처리기(스케일러 평균, 인코더 어휘, 특성 이름).
- 학습 데이터 해시와 코드 버전.
- Python / 라이브러리 버전(sklearn 1.3 ≠ sklearn 1.4 미묘하게).
- 예상 입력 특성의 스키마(이름, 타입, 범위).

**MLflow**, **DVC**, **Weights & Biases** 같은 도구가 이를 단위로 추적. 규율은: 모델 버전 `v_n`이 추적된 메타데이터에서 재현 가능, 그리고 같은 입력에 대한 학습 시점과 서빙 시점 예측 사이의 어떤 차이도 버그로 다룸. 이것 없이는 불가피한 프로덕션 불일치를 디버그할 수 없음.

### D. A/B 테스트와 섀도우 배포

프로덕션 모델 교체는 위험. 표준 플레이북:

1. **섀도우 배포(Shadow deployment)**: 프로덕션 트래픽의 사본을 새 모델로 라우팅, 그 예측을 로그, 단 사용자에게는 옛 모델의 예측 제공. 두 예측 스트림을 오프라인으로 비교. 사용자에게 보이는 위험 없음.
2. **카나리 배포(Canary deployment)**: 작은 비율(1%, 그다음 5%, 그다음 25%)의 *실제* 트래픽을 새 모델로 라우팅, 비즈니스 지표 모니터. 저하 시 즉시 롤백.
3. **A/B 테스트**: 비즈니스 지표의 통계적으로 유의한 차이를 측정하기에 충분히 오래 안정 비율(보통 50/50)로 라우팅. Lesson 23이 통계 기계를 다룸.

각 단계가 위험을 점진적으로 이동: 섀도우는 위험 0, 카나리는 작은 위험, A/B는 측정된 위험. 위험을 무릅쓰고 단계 건너뛰기.

### E. 양자화, 가지치기, 증류: 더 저렴한 추론

모델이 너무 느리거나 너무 클 때, 처음부터 재학습 없이 줄이는 세 가지 기법:

- **양자화(Quantization)**: 32비트 부동소수를 8비트 정수(또는 16비트 부동)로 대체. 메모리와 컴퓨팅을 ~4× 줄이며 보통 <1% 정확도 손실. 딥넷의 표준; 그래디언트 부스팅에 덜 흔히 적용.
- **가지치기(Pruning)**: 예측에 거의 기여하지 않는 가중치나 필터 제거. 네트워크가 희소해짐; 올바른 하드웨어 지원으로 희소 곱셈이 더 빠름.
- **증류(Distillation)**: 큰 "교사" 모델의 예측을 모방하도록 작은 "학생" 모델 학습. 학생이 교사의 *소프트 확률*에서 학습, 단단한 레이블만이 아님 — 그 추가 신호가 작은 모델을 예상치 못하게 경쟁력 있게 만들기 종종.

각 기법이 용량을 속도와 맞바꿈. 실제 검증셋에서 정확도 비용 측정; 게시된 숫자가 데이터에 이전된다고 절대 가정 안 함.

### F. 드리프트: 데이터 드리프트 vs 개념 드리프트

모델의 정확도가 프로덕션에서 두 가지 구별되는 이유로 저하될 수 있음:

- **데이터 드리프트(공변량 이동)**: *입력 분포* `p(x)`가 변함. 예: 2020년 거래 패턴에서 학습된 사기 모델이 2025년에 다른 패턴(모바일 vs 데스크톱 비율, 새 가맹점 카테고리)을 봄. 관계 `p(y | x)`는 변하지 않을 수 있음 — 모델이 학습되지 않은 입력을 볼 뿐.
- **개념 드리프트**: *타깃 관계* `p(y | x)`가 변함. 예: 스팸 발신자 행동이 탐지를 회피하도록 진화 — 같은 입력 특성이 이제 스팸일 다른 확률을 가짐.

탐지 방법이 다름:
- 데이터 드리프트: 인구 통계(평균, 표준편차, 각 특성의 KS 검정, PSI)를 통해 `p(x)`를 직접 모니터. 정답 레이블 없이 가능.
- 개념 드리프트: 레이블(또는 대리)이 필요. 최근 윈도우의 모델 정확도 / 로그 손실을 베이스라인 윈도우와 비교.

완화: 일정한 재학습, 온라인 학습, 드리프트 트리거 재학습(알람 발생 ⟹ 재학습 파이프라인 시작). 올바른 케이던스는 세계가 얼마나 빨리 변하는지에 의존.

### G. 모니터링: 무엇을 로그할지

프로덕션 ML 모니터링은 네 층을 커버:

1. **인프라**: CPU, 메모리, 복제본 수, 요청 비율. 표준 SRE 텔레메트리.
2. **서비스**: 지연시간(p50/p95/p99), 오류율, 처리량. 표준 서비스 텔레메트리.
3. **모델 입력과 출력**: 특성 분포, 예측 분포, 신뢰도 히스토그램. ML 특화.
4. **비즈니스 지표**: 전환율, 예측당 수익, 클릭률, 사용자 불만. 궁극의 KPI.

층 3이 ML 서빙이 일반 서비스와 다른 곳. 입력 분포의 드리프트는 서비스가 기술적으로 건강하더라도 알람을 발생시켜야 함. **Evidently AI**, **WhyLabs**, **Arize** 같은 도구가 이 층에 특화.

### From Theory to the Code Below

- 섹션 1의 학습-vs-프로덕션 환경 표는 (A)–(G)의 모든 최적화를 구동하는 격차.
- 섹션 2의 `joblib.dump` / `joblib.load` 라운드트립과 버전 고정은 (C)의 계약; MLflow 예제가 같은 규율을 규모에서 구현.
- 섹션 3의 양자화 / 가지치기 예제(보통 `torch.quantization` 또는 `tensorflow_model_optimization`)는 (E)의 기법.
- 섹션 4의 배치 예제는 (A)의 처리량 최적화; 지연시간 측정 코드는 (B)의 규율.
- 섹션 5의 섀도우 배포 / 카나리 코드는 (D)의 롤아웃 패턴.
- 섹션 6의 드리프트 탐지 코드(KS 검정, PSI)는 (F.1)을 구현; 개념 드리프트 탐지는 (F.2)의 레이블된 피드백 루프 필요.
- 섹션 7의 모니터링 대시보드 코드는 (G)의 층 3을 겨냥.

---

## 1. 학습 환경 vs. 추론 환경: 격차에 주의하라

### 1.1 환경 차이

```python
"""
Training Environment:                Production Environment:
┌─────────────────────────┐         ┌─────────────────────────┐
│ ✓ GPU-rich machines     │         │ ✗ Often CPU-only        │
│ ✓ Large batch sizes     │         │ ✗ Single-sample latency │
│ ✓ Full dataset in memory│         │ ✗ Streaming data        │
│ ✓ Python + pandas       │         │ ✗ May need C++/Java     │
│ ✓ Library version pinned│         │ ✗ Version conflicts     │
│ ✓ Offline evaluation    │         │ ✗ Real-time SLAs        │
│ ✓ Errors → retry        │         │ ✗ Errors → customer pain│
└─────────────────────────┘         └─────────────────────────┘
"""
```

### 1.2 일반적인 프로덕션 실패 사례

```python
"""
Failure Category         Example                                 Frequency
─────────────────────────────────────────────────────────────────────────────
Training-serving skew    Different preprocessing in train vs prod   ~40%
Dependency mismatch      numpy 1.24 in train, 1.21 in production   ~20%
Feature unavailability   Feature needs DB join at inference time    ~15%
Memory/latency           Model too large for allocated resources    ~10%
Data schema change       Upstream data adds/removes columns        ~10%
Silent model decay       Accuracy drops gradually, no alert          ~5%
─────────────────────────────────────────────────────────────────────────────

The #1 cause of production ML failures is training-serving skew — when the
preprocessing logic during training differs from what runs in production.
"""
```

---

## 2. 추론을 위한 모델 최적화

### 2.1 양자화(Quantization)

양자화는 수치 정밀도를 줄여(예: float32 → int8) 모델 크기를 축소하고 추론 속도를 향상시킵니다:

```python
"""
Precision    Bits    Memory     Speed      Accuracy Loss
────────────────────────────────────────────────────────
float32      32      1x         1x         baseline
float16      16      0.5x       ~1.5-2x    negligible
int8          8      0.25x      ~2-4x      0.1-1%
int4          4      0.125x     ~3-6x      1-3%
────────────────────────────────────────────────────────

When to quantize:
  ✓ Latency-sensitive applications (real-time inference)
  ✓ Edge deployment (mobile, IoT, embedded)
  ✓ Cost reduction (fewer/cheaper servers)

When NOT to quantize:
  ✗ Model accuracy is already marginal
  ✗ Batch processing where latency doesn't matter
  ✗ Research/experimentation phase
"""
```

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import os

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Why: joblib with compression reduces disk size significantly.
# compress=3 is a good balance between size and load speed.
joblib.dump(model, 'model_uncompressed.joblib')
joblib.dump(model, 'model_compressed.joblib', compress=3)

size_raw = os.path.getsize('model_uncompressed.joblib') / 1024
size_compressed = os.path.getsize('model_compressed.joblib') / 1024
print(f"Uncompressed: {size_raw:.1f} KB")
print(f"Compressed:   {size_compressed:.1f} KB")
print(f"Reduction:    {(1 - size_compressed / size_raw) * 100:.1f}%")
```

### 2.2 가지치기(Pruning) — 트리 기반 모델

트리 기반 모델에서 가지치기는 정확도에 거의 기여하지 않는 가지를 제거합니다:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Why: Setting max_depth and min_samples_leaf removes low-value branches.
# This reduces both overfitting and inference time without separate pruning.
configs = [
    {"max_depth": None, "min_samples_leaf": 1},    # Full tree
    {"max_depth": 10,   "min_samples_leaf": 5},     # Moderate pruning
    {"max_depth": 5,    "min_samples_leaf": 10},    # Aggressive pruning
]

for cfg in configs:
    dt = DecisionTreeClassifier(**cfg, random_state=42)
    score = cross_val_score(dt, X, y, cv=5, scoring='accuracy').mean()
    dt.fit(X, y)
    n_leaves = dt.get_n_leaves()
    print(f"depth={str(cfg['max_depth']):>4s}, min_leaf={cfg['min_samples_leaf']:>2d} "
          f"→ leaves={n_leaves:>5d}, accuracy={score:.4f}")
```

### 2.3 지식 증류(Knowledge Distillation)

더 작은 "학생(Student)" 모델을 훈련하여 더 큰 "교사(Teacher)" 모델의 예측을 모방하게 합니다:

```python
"""
Knowledge Distillation Pipeline:

┌──────────────────┐     soft labels      ┌──────────────────┐
│  Teacher Model   │ ──────────────────→  │  Student Model   │
│  (Large, slow)   │                      │  (Small, fast)   │
│  RF(500 trees)   │                      │  LR or small RF  │
└──────────────────┘                      └──────────────────┘

Why it works:
  - Teacher's predicted probabilities carry richer information than hard labels.
  - [0.7, 0.2, 0.1] tells the student that class 2 is "somewhat similar" to
    class 1, while hard label [1, 0, 0] loses this inter-class relationship.
  - The student can learn the teacher's decision surface with fewer parameters.
"""
```

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Teacher: large, accurate model
teacher = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
teacher.fit(X_train, y_train)
teacher_acc = accuracy_score(y_test, teacher.predict(X_test))

# Why: Use teacher's soft probabilities (predict_proba) instead of hard labels.
# The student learns the teacher's confidence patterns, not just correct/incorrect.
soft_labels = teacher.predict_proba(X_train)[:, 1]

# Student: small, fast model trained on teacher's soft labels
student = LogisticRegression(max_iter=1000, random_state=42)
# Why: We threshold at 0.5 to create binary labels from soft probabilities.
# For multi-class, you would use argmax(teacher.predict_proba(X)).
student.fit(X_train, (soft_labels > 0.5).astype(int))
student_acc = accuracy_score(y_test, student.predict(X_test))

# Baseline: student trained on original labels
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

print(f"Teacher (GBM 200 trees): {teacher_acc:.4f}")
print(f"Student (distilled LR):  {student_acc:.4f}")
print(f"Baseline (standard LR):  {baseline_acc:.4f}")
```

### 2.4 ONNX 내보내기

ONNX(Open Neural Network Exchange)는 프레임워크에 독립적인 모델 형식을 제공합니다:

```python
"""
Why ONNX?
  1. Language-independent: Train in Python, serve in C++/Java/C#
  2. Runtime optimization: ONNX Runtime applies graph-level optimizations
  3. Hardware acceleration: Automatic GPU/NNAPI/CoreML backend selection
  4. Standardized format: Works across sklearn, PyTorch, TensorFlow, XGBoost

Conversion flow:
  sklearn model → skl2onnx → .onnx file → ONNX Runtime → predictions
  PyTorch model → torch.onnx.export() → .onnx file → ONNX Runtime → predictions
"""
```

```python
# ONNX conversion for sklearn (requires: pip install skl2onnx onnxruntime)
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# import onnxruntime as ort
#
# initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
# onnx_model = convert_sklearn(teacher, initial_types=initial_type)
#
# with open("model.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())
#
# # Why: ONNX Runtime is often 2-10x faster than native sklearn predict().
# session = ort.InferenceSession("model.onnx")
# input_name = session.get_inputs()[0].name
# pred = session.run(None, {input_name: X_test.astype(np.float32)})[0]

# Size comparison helper
def compare_model_sizes(sklearn_model, onnx_path=None):
    """Compare serialized model sizes across formats."""
    joblib.dump(sklearn_model, '/tmp/model.joblib')
    size_joblib = os.path.getsize('/tmp/model.joblib')

    joblib.dump(sklearn_model, '/tmp/model_compressed.joblib', compress=3)
    size_compressed = os.path.getsize('/tmp/model_compressed.joblib')

    print(f"joblib (raw):        {size_joblib / 1024:.1f} KB")
    print(f"joblib (compressed): {size_compressed / 1024:.1f} KB")
    if onnx_path and os.path.exists(onnx_path):
        size_onnx = os.path.getsize(onnx_path)
        print(f"ONNX:                {size_onnx / 1024:.1f} KB")
```

---

## 3. 서빙 패턴(Serving Patterns)

### 3.1 패턴 비교

```python
"""
Pattern         Latency        Throughput    Use Case
──────────────────────────────────────────────────────────────────
Batch           Minutes-hours  Very high     Nightly recommendations,
                                             risk scoring, reports

Real-time       10-100ms       Medium        Fraud detection, search
(synchronous)                                ranking, ad targeting

Near real-time  100ms-1s       Medium-high   Content moderation,
(async)                                      dynamic pricing

Streaming       50-200ms       High          IoT anomaly detection,
                                             real-time personalization
──────────────────────────────────────────────────────────────────
"""
```

### 3.2 배치 예측(Batch Prediction)

```python
"""
Batch Prediction Pipeline:

  ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
  │ Data      │ →   │ Preprocess│ →   │ Model     │ →   │ Store    │
  │ Warehouse │     │ (Spark)   │     │ Predict   │     │ Results  │
  └──────────┘     └──────────┘     └───────────┘     └──────────┘

When to use batch:
  ✓ Predictions needed periodically (hourly, daily)
  ✓ All input data is available ahead of time
  ✓ High throughput more important than low latency
  ✓ Resource-intensive models (large ensembles, deep learning)

Architecture:
  - Scheduler (cron, Airflow) triggers the pipeline
  - Results stored in DB/cache for fast lookup
  - Predictions served from pre-computed cache, not live model
"""
```

```python
import time

def batch_predict(model, data, batch_size=1000):
    """
    Why batch_size matters: Processing all data at once may exceed memory.
    Processing one sample at a time is slow due to Python overhead.
    Batching balances memory usage and throughput.
    """
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        predictions.extend(model.predict(batch))
    return np.array(predictions)

# Throughput comparison: single vs batch
large_data = np.random.randn(10000, 20)

start = time.time()
single_preds = [model.predict(large_data[i:i+1]) for i in range(len(large_data))]
single_time = time.time() - start

start = time.time()
batch_preds = batch_predict(model, large_data, batch_size=1000)
batch_time = time.time() - start

print(f"Single-sample loop: {single_time:.3f}s ({len(large_data)/single_time:.0f} samples/s)")
print(f"Batch (size=1000):  {batch_time:.3f}s ({len(large_data)/batch_time:.0f} samples/s)")
print(f"Speedup: {single_time / batch_time:.1f}x")
```

### 3.3 실시간 추론(Real-Time Inference)

```python
"""
Real-Time Inference Requirements:

  ┌─────────────┬──────────────────────────────────────────────────────┐
  │ Concern      │ Strategy                                            │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Latency      │ p50 < 20ms, p99 < 100ms                            │
  │ Model size   │ < 100MB in memory (or use model compression)       │
  │ Cold start   │ Pre-load model at server start                     │
  │ Concurrency  │ Thread-safe predict() or process-per-request       │
  │ Fallback     │ Return default prediction if model fails           │
  │ Input valid. │ Validate schema before predict (reject bad input)  │
  └─────────────┴──────────────────────────────────────────────────────┘

Latency budget breakdown (target: 100ms total):
  Network overhead:     ~10ms
  Input validation:      ~2ms
  Feature extraction:   ~20ms  ← often the bottleneck
  Preprocessing:        ~10ms
  Model inference:      ~30ms
  Post-processing:       ~5ms
  Response serialization:~3ms
  Buffer:               ~20ms
"""
```

### 3.4 의사 결정 프레임워크

```
Which serving pattern should I use?

  ├── Can all predictions be pre-computed? → Batch
  ├── Latency requirement < 200ms?
  │   ├── Yes → Real-time (synchronous API)
  │   └── No → Near real-time (async queue)
  ├── Continuous data stream? → Streaming
  └── Hybrid? → Batch (base) + Real-time (personalization layer)
```

---

## 4. 학습-서빙 왜곡(Training-Serving Skew) 방지

### 4.1 학습-서빙 왜곡이란?

학습-서빙 왜곡은 학습 시점의 데이터나 변환이 추론 시점과 다를 때 발생합니다:

```python
"""
Common sources of skew:

1. Preprocessing mismatch:
   Train: StandardScaler().fit_transform(X_train)
   Serve: X_raw / X_raw.max()  ← WRONG! Different normalization

2. Feature computation difference:
   Train: user_age = (current_date - birth_date).days / 365
   Serve: user_age = lookup_from_cache()  ← stale, computed yesterday

3. Missing value handling:
   Train: df['col'].fillna(df['col'].median())
   Serve: df['col'].fillna(0)  ← different imputation

4. Library version difference:
   Train: pandas 2.1 tokenizes "café" as ["café"]
   Serve: pandas 1.5 tokenizes "café" as ["caf", "é"]  ← different!
"""
```

### 4.2 해결책: sklearn 파이프라인(Pipeline)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Why: A Pipeline bundles preprocessing + model into one serializable object.
# When you save the pipeline, the fitted scaler/imputer parameters are preserved.
# At serving time, you call pipeline.predict(raw_input) — no separate preprocessing.
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
train_pred = pipeline.predict(X_test)

# Why: Save the entire pipeline, not just the model.
# This guarantees identical preprocessing at training and serving time.
joblib.dump(pipeline, 'production_pipeline.joblib')

# At serving time:
# loaded_pipeline = joblib.load('production_pipeline.joblib')
# prediction = loaded_pipeline.predict(raw_input)  ← includes all preprocessing
```

### 4.3 피처 스토어(Feature Store) 패턴

```python
"""
Without Feature Store:

  Training:   features = compute_features(raw_data)     # Python, offline
  Serving:    features = compute_features_v2(raw_data)   # Java, online ← SKEW!

With Feature Store:

  ┌────────────────┐        ┌────────────────┐
  │ Offline Store   │ ←───── │ Feature Store  │ ─────→ │ Online Store   │
  │ (Training)      │        │ (Single Source) │        │ (Serving)      │
  │ Batch features  │        │ One definition  │        │ Low-latency    │
  └────────────────┘        └────────────────┘        └────────────────┘

  Same computation logic is used for both training and serving.
  The feature store handles the offline/online split transparently.
"""
```

### 4.4 왜곡 감지 체크리스트

```python
"""
Before deploying a model, verify these items:

□ Preprocessing pipeline is serialized WITH the model (not separate scripts)
□ Feature computation logic is shared between training and serving code
□ Library versions are pinned (requirements.txt / poetry.lock)
□ Input schema is validated at serving time (column names, types, ranges)
□ Missing value strategy is identical (same imputer, same fill values)
□ Categorical encoding is identical (same mapping, same handling of unseen categories)
□ Numerical precision matches (float32 vs float64)
□ Time-dependent features use consistent time zones and windows
"""
```

---

## 5. 지연시간-정확도 트레이드오프(Latency-Accuracy Trade-offs)

### 5.1 트레이드오프 지형도

```python
"""
                    ▲ Accuracy
                    │
        ●           │          ● Deep Ensemble (5 models)
  Neural Net        │        ● Gradient Boosting (500 trees)
                    │      ● Random Forest (200 trees)
                    │    ● Random Forest (50 trees)
                    │  ● Logistic Regression
                    │● Decision Stump
                    └──────────────────────────→ Latency
                    1ms  5ms  20ms  50ms  200ms  1s

Each application has a latency budget. The optimal model is the most
accurate one that fits within that budget.
"""
```

### 5.2 실용적 의사 결정 프레임워크

```python
"""
Step 1: Define your latency budget
  - "Our API must respond in <100ms at p99"
  - Budget for model inference: total budget - network - preprocessing

Step 2: Benchmark candidate models
  For each model:
    - Measure predict() latency (p50, p95, p99)
    - Measure accuracy on validation set
    - Measure model size (memory footprint)

Step 3: Plot the Pareto frontier
  - Models on the frontier offer the best accuracy-latency trade-off
  - Discard models below the frontier (dominated by better alternatives)

Step 4: Apply optimization if needed
  - Can't fit budget? Try: ONNX conversion, quantization, distillation
  - Still too slow? Reduce model complexity (fewer trees, smaller depth)
  - Still too slow? Switch to a simpler model family
"""
```

```python
import time

# Benchmark different model complexities
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RF(10 trees)':   RandomForestClassifier(n_estimators=10,  random_state=42),
    'RF(100 trees)':  RandomForestClassifier(n_estimators=100, random_state=42),
    'RF(500 trees)':  RandomForestClassifier(n_estimators=500, random_state=42),
    'GBM(50 trees)':  GradientBoostingClassifier(n_estimators=50,  random_state=42),
    'GBM(200 trees)': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# Why: Measure single-sample latency because that's what production sees.
# Batch latency can be misleading — amortized overhead hides per-request cost.
single_sample = X_test[:1]
results = []

for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))

    # Warm-up run (JIT, cache warming)
    m.predict(single_sample)

    # Measure latency over 100 runs
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        m.predict(single_sample)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    results.append((name, acc, p50, p99))

print(f"{'Model':<22s} {'Accuracy':>8s} {'p50(ms)':>8s} {'p99(ms)':>8s}")
print("-" * 50)
for name, acc, p50, p99 in results:
    print(f"{name:<22s} {acc:>8.4f} {p50:>8.3f} {p99:>8.3f}")
```

---

## 6. 모델 패키징 모범 사례

### 6.1 직렬화 형식

```python
"""
Format       Library         Pros                          Cons
─────────────────────────────────────────────────────────────────────────
pickle       Python stdlib   Simple, universal             Security risk (arbitrary
                                                           code execution), fragile
                                                           across Python versions

joblib       scikit-learn    Efficient for numpy arrays,   Python-only
                             compression support

ONNX         Cross-platform  Language-agnostic, optimized  Limited operator support,
                             runtime, hardware accel.      conversion complexity

TorchScript  PyTorch         Python-free execution,        PyTorch-only
                             C++ integration

PMML/PFA     Legacy          Enterprise compatibility      Limited model support
─────────────────────────────────────────────────────────────────────────

Recommendation:
  sklearn models → joblib (with compression) or ONNX
  PyTorch models → TorchScript or ONNX
  XGBoost/LightGBM → native save + ONNX for cross-platform
"""
```

### 6.2 모델 아티팩트 구조

```python
"""
A production model artifact should include:

model_artifact/
├── model.joblib              # Serialized model (or .onnx, .pt)
├── metadata.json             # Model metadata
│   ├── model_version         # "1.2.0"
│   ├── training_date         # "2024-01-15T10:30:00Z"
│   ├── training_data_hash    # SHA256 of training data
│   ├── feature_names         # ["age", "income", ...]
│   ├── feature_types         # {"age": "float64", "income": "float64"}
│   ├── target_classes        # [0, 1] or ["spam", "ham"]
│   ├── performance_metrics   # {"accuracy": 0.95, "f1": 0.93}
│   ├── dependencies          # {"sklearn": "1.4.0", "numpy": "1.26.0"}
│   └── input_schema          # JSON Schema for validation
├── preprocessing.joblib      # Fitted transformers (if separate from model)
└── requirements.txt          # Exact dependency versions
"""
```

```python
import json
from datetime import datetime

def create_model_metadata(model, X_train, y_train, metrics, feature_names):
    """
    Why metadata matters: Without it, production teams can't verify
    which model version is running, what data it was trained on,
    or what inputs it expects.
    """
    metadata = {
        "model_version": "1.0.0",
        "model_class": type(model).__name__,
        "training_date": datetime.utcnow().isoformat() + "Z",
        "n_training_samples": len(X_train),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "target_classes": sorted(set(y_train.tolist())),
        "performance_metrics": metrics,
        "python_version": "3.10",
    }
    return metadata

metadata = create_model_metadata(
    model=pipeline,
    X_train=X_train, y_train=y_train,
    metrics={"accuracy": 0.95, "f1_macro": 0.94},
    feature_names=[f"feature_{i}" for i in range(X_train.shape[1])]
)
print(json.dumps(metadata, indent=2, default=str))
```

### 6.3 입력 검증(Input Validation)

```python
def validate_input(data, expected_features, expected_dtypes=None):
    """
    Why: Reject invalid inputs early rather than returning garbage predictions.
    A model that silently returns wrong predictions is worse than one that
    raises an error — at least the error is visible.
    """
    import numpy as np

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(data).__name__}")

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    if data.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {data.shape[1]}"
        )

    if np.any(np.isnan(data)):
        nan_cols = np.where(np.any(np.isnan(data), axis=0))[0]
        raise ValueError(f"NaN values found in columns: {nan_cols.tolist()}")

    if np.any(np.isinf(data)):
        raise ValueError("Infinite values found in input")

    return True
```

---

## 7. 모니터링: 무엇을 관찰할 것인가 (ML 관점)

### 7.1 데이터 드리프트(Data Drift)

데이터 드리프트는 입력 피처의 분포가 시간이 지남에 따라 변하는 현상입니다:

```python
"""
Types of Distribution Shift:

1. Covariate shift: P(X) changes, P(Y|X) stays
   Example: More older users sign up → age distribution shifts
   → Model predictions may be less calibrated for new population

2. Concept drift: P(Y|X) changes, P(X) may stay
   Example: "spam" patterns evolve — model's learned rules become stale
   → Model needs retraining

3. Label drift: P(Y) changes
   Example: Fraud rate increases from 1% to 5%
   → Threshold recalibration needed

Detection methods:
  ┌─────────────────────┬──────────────────────────────────┐
  │ Method              │ When to use                       │
  ├─────────────────────┼──────────────────────────────────┤
  │ PSI (Population     │ Categorical + binned numerical    │
  │ Stability Index)    │ features. PSI > 0.25 = major      │
  │                     │ shift.                            │
  ├─────────────────────┼──────────────────────────────────┤
  │ KS Test             │ Continuous features. p < 0.01 =   │
  │ (Kolmogorov-Smirnov)│ significant shift.                │
  ├─────────────────────┼──────────────────────────────────┤
  │ JS Divergence       │ Any distribution. Symmetric       │
  │ (Jensen-Shannon)    │ alternative to KL divergence.     │
  └─────────────────────┴──────────────────────────────────┘
"""
```

```python
from scipy import stats

def detect_drift_ks(reference_data, production_data, feature_names, threshold=0.05):
    """
    Why KS test: It's non-parametric — works regardless of the underlying
    distribution shape. It tests whether two samples come from the same distribution.
    """
    results = []
    for i, name in enumerate(feature_names):
        stat, p_value = stats.ks_2samp(reference_data[:, i], production_data[:, i])
        drifted = p_value < threshold
        results.append({
            "feature": name,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drifted": drifted
        })

    drifted_features = [r for r in results if r["drifted"]]
    print(f"Drift detected in {len(drifted_features)}/{len(feature_names)} features")
    for r in drifted_features:
        print(f"  {r['feature']}: KS={r['ks_statistic']}, p={r['p_value']}")
    return results

# Simulate drift: shift 3 features
reference = np.random.randn(1000, 5)
production = np.random.randn(1000, 5)
production[:, 0] += 0.5   # Moderate drift
production[:, 2] += 1.5   # Strong drift
production[:, 4] *= 2.0   # Variance change

feature_names = [f"feature_{i}" for i in range(5)]
detect_drift_ks(reference, production, feature_names)
```

### 7.2 성능 모니터링

```python
"""
What to monitor (from ML perspective):

1. Prediction distribution
   - Track mean, variance, percentiles of predicted probabilities
   - Alert if distribution shifts significantly from training baseline
   - Example: If model normally predicts 5% positive, alert at 15%

2. Confidence calibration
   - Track: "Of predictions with confidence 0.9, how many are correct?"
   - Degradation suggests data distribution has changed

3. Feature value ranges
   - Each feature should stay within [train_min, train_max] (with some margin)
   - Out-of-range features indicate data pipeline changes or new populations

4. Prediction latency
   - Track p50, p95, p99 latency
   - Sudden increases suggest input data issues or resource contention

Monitoring schedule:
  ┌────────────────┬──────────────────────┐
  │ Real-time       │ Latency, error rate  │
  │ Hourly          │ Prediction distrib.  │
  │ Daily           │ Feature drift (PSI)  │
  │ Weekly          │ Accuracy (if labels  │
  │                 │ available)           │
  └────────────────┴──────────────────────┘
"""
```

### 7.3 재학습 시점

```python
"""
Retraining Triggers (Decision Tree):

  Has accuracy dropped > 2% from baseline?
  ├── Yes → Retrain immediately
  └── No
      Has data drift been detected (PSI > 0.25)?
      ├── Yes → Retrain within 1 week
      └── No
          Has it been > 3 months since last training?
          ├── Yes → Scheduled retrain
          └── No → Continue monitoring

Retraining strategies:
  1. Full retrain: Train on all historical data (simple, but expensive)
  2. Incremental: Update model with new data only (faster, but may forget)
  3. Window-based: Train on last N months of data (adapts to recent patterns)
  4. Triggered: Retrain only when drift/degradation is detected (efficient)
"""
```

---

## 연습문제

### 연습문제 1: 모델 최적화 벤치마크

```python
"""
Using make_classification(n_samples=10000, n_features=30):

1. Train a GradientBoostingClassifier(n_estimators=300, max_depth=6)
2. Measure: accuracy, single-sample latency (p50, p99), model size on disk
3. Create optimized variants:
   a. Fewer trees (n_estimators=50)
   b. Shallower trees (max_depth=3)
   c. Knowledge distillation to LogisticRegression
   d. joblib compression (compress=5)
4. Compare all variants in a table: accuracy, latency, size
5. Which variant offers the best trade-off for a 50ms latency budget?
"""
```

### 연습문제 2: 학습-서빙 왜곡 방지

```python
"""
1. Create a dataset with mixed feature types:
   - 5 numerical features (some with missing values)
   - 3 categorical features (some with rare categories)
2. Build a Pipeline using ColumnTransformer:
   - SimpleImputer + StandardScaler for numerical features
   - OneHotEncoder(handle_unknown='ignore') for categorical features
   - RandomForestClassifier as the model
3. Save the entire pipeline with joblib
4. Load it in a fresh session and verify predictions match
5. Simulate a skew scenario: what happens if you preprocess
   differently at serving time? Compare predictions.
"""
```

### 연습문제 3: 드리프트 감지 시스템

```python
"""
1. Generate a reference dataset (normal distribution, 10 features)
2. Simulate 4 types of production data:
   a. No drift (same distribution)
   b. Mean shift in 2 features
   c. Variance change in 3 features
   d. New category/distribution in 1 feature
3. Implement both KS test and PSI-based drift detection
4. For each scenario, report which features are flagged
5. Compare: which method is more sensitive to each drift type?
"""
```

### 연습문제 4: 지연시간-정확도 파레토 프론티어(Pareto Frontier)

```python
"""
1. Train 8+ models of varying complexity:
   - LogisticRegression, Decision Tree (depth 3, 10, 20),
   - RF (10, 50, 200 trees), GBM (50, 200 trees), SVM
2. Measure accuracy and single-sample latency for each
3. Plot the Pareto frontier (latency on X, accuracy on Y)
4. Mark models that are Pareto-optimal (not dominated by any other)
5. Given a 30ms latency budget, which model should you choose?
"""
```

---

## 8. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **학습-서빙 왜곡(Training-Serving Skew)** | 프로덕션 ML 실패의 #1 원인 — 항상 전처리를 모델과 함께 직렬화할 것 |
| **양자화(Quantization)** | 정밀도를 줄여(float32→int8) 2-4배 속도 향상, ~1% 정확도 손실 |
| **지식 증류(Knowledge Distillation)** | 작은 학생 모델을 훈련하여 큰 교사 모델의 예측을 모방 |
| **ONNX** | 크로스 플랫폼 배포를 위한 프레임워크 독립적 모델 형식 |
| **서빙 패턴(Serving Patterns)** | 배치(사전 계산), 실시간(<100ms), 스트리밍(연속) |
| **데이터 드리프트(Data Drift)** | KS 검정이나 PSI로 피처 분포를 모니터링; 필요 시 재학습 |
| **모델 패키징(Model Packaging)** | 모델 + 메타데이터 + 전처리 + 버전 정보를 포함 |

### 다른 레슨과의 연결

- **L04 (모델 평가)**: 프로덕션 모니터링은 오프라인 평가를 지속적인 온라인 추적으로 확장
- **L05 (교차 검증)**: CV는 일반화 성능을 추정하고, 프로덕션 모니터링은 이를 *검증*
- **L13 (파이프라인)**: sklearn 파이프라인은 전처리를 번들링하여 학습-서빙 왜곡을 방지
- **L19 (AutoML)**: AutoML은 모델 선택을 자동화하고, 프로덕션 ML은 배포 준비를 자동화
- **L21 (고급 앙상블)**: 앙상블은 프로덕션에서 추가적인 지연시간 문제에 직면
- **MLOps L08-09**: 모델 서빙의 인프라 측면 (TorchServe, Triton, FastAPI)
- **MLOps L10**: 모니터링의 인프라 측면 (대시보드, 알림, 자동화)

# 머신러닝 개요

## 1. 머신러닝이란?

머신러닝(Machine Learning)은 명시적으로 프로그래밍하지 않아도 데이터로부터 학습하여 예측이나 결정을 수행하는 알고리즘입니다.

```python
# 전통적 프로그래밍 vs 머신러닝
# 전통적: 규칙 + 데이터 → 결과
# 머신러닝: 데이터 + 결과 → 규칙(모델)
```

---

## 2. 머신러닝의 유형

### 2.1 지도학습 (Supervised Learning)

입력(X)과 정답(y)이 있는 데이터로 학습합니다.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 예시: 집 크기로 가격 예측
X = np.array([[50], [60], [70], [80], [90], [100]])  # 크기 (평)
y = np.array([1.5, 1.8, 2.1, 2.5, 2.8, 3.2])  # 가격 (억)

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
new_house = [[75]]
predicted_price = model.predict(new_house)
print(f"75평 집 예상 가격: {predicted_price[0]:.2f}억")
```

**주요 알고리즘:**
- **회귀 (Regression)**: 연속적인 값 예측
  - 선형회귀, 다항회귀, 릿지, 라쏘
- **분류 (Classification)**: 범주 예측
  - 로지스틱 회귀, SVM, 결정트리, 랜덤포레스트

### 2.2 비지도학습 (Unsupervised Learning)

정답 없이 데이터의 구조나 패턴을 학습합니다.

```python
from sklearn.cluster import KMeans
import numpy as np

# 고객 데이터 (나이, 구매금액)
X = np.array([[25, 100], [30, 150], [35, 120],
              [50, 300], [55, 350], [60, 400]])

# K-Means 클러스터링
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

print(f"클러스터 레이블: {labels}")
print(f"클러스터 중심:\n{kmeans.cluster_centers_}")
```

**주요 알고리즘:**
- **클러스터링**: K-Means, DBSCAN, 계층적 군집화
- **차원축소**: PCA, t-SNE
- **이상치 탐지**: Isolation Forest

### 2.3 강화학습 (Reinforcement Learning)

환경과 상호작용하며 보상을 최대화하는 방향으로 학습합니다.

- 에이전트가 행동을 선택
- 환경에서 보상 또는 패널티 수신
- 누적 보상 최대화

**적용 분야:** 게임 AI, 로봇 제어, 자율주행

---

## 3. 머신러닝 워크플로우

```
1. 문제 정의 → 2. 데이터 수집 → 3. 데이터 탐색(EDA)
                                        ↓
        7. 배포/모니터링 ← 6. 모델 선택 ← 5. 모델 학습 ← 4. 데이터 전처리
```

### 3.1 기본 워크플로우 예시

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 데이터 전처리 (스케일링)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 5. 예측
y_pred = model.predict(X_test_scaled)

# 6. 평가
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 4. 핵심 개념

### 4.1 훈련/검증/테스트 분할

```python
from sklearn.model_selection import train_test_split

# 전체 데이터를 훈련(60%), 검증(20%), 테스트(20%)로 분할
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"훈련: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")
```

- **훈련 데이터**: 모델 학습에 사용
- **검증 데이터**: 하이퍼파라미터 튜닝에 사용
- **테스트 데이터**: 최종 성능 평가에 사용 (한 번만)

### 4.2 과적합과 과소적합

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 생성
np.random.seed(42)
X = np.sort(np.random.rand(20, 1) * 6, axis=0)
y = np.sin(X).ravel() + np.random.randn(20) * 0.1

# 다양한 복잡도의 모델
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ['과소적합 (Underfitting)', '적절한 적합', '과적합 (Overfitting)']

for ax, degree, title in zip(axes, degrees, titles):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    X_plot = np.linspace(0, 6, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    ax.scatter(X, y, color='blue', alpha=0.7)
    ax.plot(X_plot, y_plot, color='red', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
```

- **과소적합 (Underfitting)**: 모델이 너무 단순하여 훈련 데이터도 잘 학습하지 못함
- **과적합 (Overfitting)**: 모델이 훈련 데이터에 너무 맞춰져 새로운 데이터에 일반화 실패

### 4.3 편향-분산 트레이드오프

```
총 오차 = 편향² + 분산 + 노이즈

편향 (Bias): 모델의 단순함으로 인한 오차
분산 (Variance): 데이터 변화에 대한 모델의 민감도

높은 편향 → 과소적합
높은 분산 → 과적합
```

### 4.4 특성 스케일링

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 예시 데이터
X = np.array([[100, 0.001], [200, 0.002], [300, 0.003]])

# StandardScaler (Z-score 정규화)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print("StandardScaler 결과:")
print(X_std)

# MinMaxScaler (0-1 정규화)
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
print("\nMinMaxScaler 결과:")
print(X_minmax)
```

---

## 5. sklearn 기본 API

### 5.1 추정기 (Estimator) 인터페이스

```python
# 모든 sklearn 모델은 동일한 인터페이스를 따름
from sklearn.ensemble import RandomForestClassifier

# 1. 모델 생성
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. 학습 (fit)
model.fit(X_train, y_train)

# 3. 예측 (predict)
y_pred = model.predict(X_test)

# 4. 확률 예측 (predict_proba) - 분류 모델
y_proba = model.predict_proba(X_test)

# 5. 점수 (score)
accuracy = model.score(X_test, y_test)
```

### 5.2 변환기 (Transformer) 인터페이스

```python
from sklearn.preprocessing import StandardScaler

# 1. 변환기 생성
scaler = StandardScaler()

# 2. 학습 (fit)
scaler.fit(X_train)

# 3. 변환 (transform)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit + transform 동시에
X_train_scaled = scaler.fit_transform(X_train)
# 주의: test 데이터에는 transform만!
X_test_scaled = scaler.transform(X_test)
```

---

## 6. 데이터셋

### 6.1 sklearn 내장 데이터셋

```python
from sklearn.datasets import (
    load_iris,        # 분류 (3클래스)
    load_digits,      # 분류 (10클래스)
    load_breast_cancer,  # 이진 분류
    load_boston,      # 회귀 (deprecated)
    load_diabetes,    # 회귀
    make_classification,  # 합성 분류 데이터
    make_regression,      # 합성 회귀 데이터
)

# Iris 데이터셋
iris = load_iris()
print(f"특성: {iris.feature_names}")
print(f"타겟: {iris.target_names}")
print(f"데이터 형태: {iris.data.shape}")

# 합성 데이터 생성
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)
print(f"합성 데이터 형태: {X.shape}")
```

### 6.2 외부 데이터 로드

```python
import pandas as pd

# CSV
df = pd.read_csv('data.csv')

# 특성과 타겟 분리
X = df.drop('target', axis=1)
y = df['target']

# Kaggle 데이터 (예시)
# !pip install kaggle
# !kaggle datasets download -d username/dataset-name
```

---

## 7. 머신러닝 프로젝트 템플릿

```python
"""
머신러닝 프로젝트 기본 템플릿
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
# df = pd.read_csv('data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# 2. 탐색적 데이터 분석 (EDA)
# print(df.info())
# print(df.describe())
# print(df['target'].value_counts())

# 3. 데이터 전처리
# - 결측치 처리
# - 인코딩
# - 스케일링

# 4. 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# 5. 모델 선택 및 학습
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# 6. 교차 검증
# cv_scores = cross_val_score(model, X_train, y_train, cv=5)
# print(f"CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. 하이퍼파라미터 튜닝
# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_estimators': [50, 100, 200]}
# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# 8. 최종 평가
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# 9. 모델 저장
# import joblib
# joblib.dump(model, 'model.pkl')
```

---

## 연습 문제

### 문제 1: 데이터 분할
iris 데이터를 80:20으로 분할하고 클래스 비율을 유지하세요.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# 풀이
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터: {len(X_train)}")
print(f"테스트 데이터: {len(X_test)}")
print(f"테스트 클래스 분포: {np.bincount(y_test)}")
```

### 문제 2: 기본 모델 학습
로지스틱 회귀 모델을 학습하고 정확도를 구하세요.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 풀이
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 지도학습 | 정답이 있는 데이터로 학습 (회귀, 분류) |
| 비지도학습 | 정답 없이 패턴 학습 (클러스터링, 차원축소) |
| 강화학습 | 환경과 상호작용하며 보상 최대화 |
| 과적합 | 훈련 데이터에 과도하게 적합 |
| 과소적합 | 모델이 너무 단순함 |
| 편향-분산 | 모델 복잡도와 일반화 능력의 트레이드오프 |

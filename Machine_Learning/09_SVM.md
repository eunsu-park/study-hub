# 서포트 벡터 머신 (Support Vector Machine)

## 개요

서포트 벡터 머신(SVM)은 데이터를 분류하기 위한 최적의 결정 경계(초평면)를 찾는 알고리즘입니다. 마진 최대화와 커널 트릭을 통해 고차원 데이터와 비선형 패턴도 효과적으로 처리할 수 있습니다.

---

## 1. SVM의 기본 개념

### 1.1 초평면과 마진

```python
"""
SVM 핵심 개념:

1. 초평면 (Hyperplane):
   - 데이터를 분리하는 결정 경계
   - n차원에서 (n-1)차원 평면
   - 2D: 직선, 3D: 평면

2. 마진 (Margin):
   - 결정 경계와 가장 가까운 데이터 포인트 사이의 거리
   - SVM은 마진을 최대화하는 초평면을 찾음

3. 서포트 벡터 (Support Vectors):
   - 마진 경계에 위치한 데이터 포인트
   - 결정 경계를 정의하는 핵심 데이터
   - 다른 데이터 포인트는 결정 경계에 영향 없음
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
```

### 1.2 선형 SVM 시각화

```python
# 선형 분리 가능한 데이터 생성
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# 선형 SVM 학습
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# 시각화
plt.figure(figsize=(10, 8))

# 데이터 포인트
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100, edgecolors='black')

# 결정 경계와 마진
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 그리드 생성
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 결정 경계와 마진 그리기
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
           linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

# 서포트 벡터 표시
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=200, linewidth=2, facecolors='none', edgecolors='green',
           label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM: Maximum Margin Classifier')
plt.legend()
plt.show()

print(f"서포트 벡터 수: {len(clf.support_vectors_)}")
print(f"가중치 (w): {clf.coef_}")
print(f"절편 (b): {clf.intercept_}")
```

---

## 2. 하드 마진 vs 소프트 마진

### 2.1 하드 마진 SVM

```python
"""
하드 마진 (Hard Margin):
- 모든 데이터 포인트가 마진 바깥에 위치해야 함
- 오분류 허용하지 않음
- 선형 분리 가능한 데이터에만 적용 가능

최적화 문제:
minimize: (1/2)||w||²
subject to: y_i(w·x_i + b) >= 1, ∀i
"""
```

### 2.2 소프트 마진 SVM

```python
"""
소프트 마진 (Soft Margin):
- 일부 오분류 허용
- 슬랙 변수(ξ) 도입
- 실제 데이터에 적용 가능

최적화 문제:
minimize: (1/2)||w||² + C * Σξ_i
subject to: y_i(w·x_i + b) >= 1 - ξ_i, ξ_i >= 0

C: 규제 파라미터
- C 큼: 오분류에 큰 페널티 → 좁은 마진, 과적합 위험
- C 작음: 오분류 허용 → 넓은 마진, 일반화 향상
"""

# C 파라미터 효과
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 노이즈가 있는 데이터
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    flip_y=0.1,  # 10% 노이즈
    random_state=42
)

# 여러 C 값 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
C_values = [0.1, 1, 100]

for ax, C in zip(axes, C_values):
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y)

    # 결정 경계
    xlim = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
    ylim = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=150, facecolors='none', edgecolors='green', linewidths=2)
    ax.set_title(f'C = {C}\nSupport Vectors: {len(clf.support_vectors_)}')

plt.tight_layout()
plt.show()
```

---

## 3. 커널 트릭 (Kernel Trick)

### 3.1 커널 함수의 원리

```python
"""
커널 트릭:
- 비선형 데이터를 고차원 공간으로 매핑하여 선형 분리
- 실제로 고차원 변환 없이 내적만으로 계산 (효율적)
- K(x, y) = φ(x)·φ(y)

주요 커널 함수:

1. 선형 커널 (Linear):
   K(x, y) = x·y
   - 선형 분리 가능한 데이터

2. 다항식 커널 (Polynomial):
   K(x, y) = (γ * x·y + r)^d
   - d: 다항식 차수

3. RBF 커널 (Radial Basis Function, Gaussian):
   K(x, y) = exp(-γ||x - y||²)
   - 가장 널리 사용
   - 무한 차원으로 매핑

4. 시그모이드 커널:
   K(x, y) = tanh(γ * x·y + r)
   - 신경망과 유사
"""
```

### 3.2 비선형 데이터에 커널 적용

```python
from sklearn.datasets import make_moons, make_circles

# 비선형 데이터 생성
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

# 커널 비교
kernels = ['linear', 'poly', 'rbf']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row, (X, y, name) in enumerate([(X_moons, y_moons, 'Moons'),
                                     (X_circles, y_circles, 'Circles')]):
    for col, kernel in enumerate(kernels):
        ax = axes[row, col]

        # SVM 학습
        if kernel == 'poly':
            clf = svm.SVC(kernel=kernel, degree=3, gamma='scale')
        else:
            clf = svm.SVC(kernel=kernel, gamma='scale')
        clf.fit(X, y)

        # 결정 경계
        xlim = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
        ylim = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
        ax.set_title(f'{name} - {kernel}\nAccuracy: {clf.score(X, y):.3f}')

plt.tight_layout()
plt.show()
```

### 3.3 RBF 커널과 gamma 파라미터

```python
"""
RBF 커널: K(x, y) = exp(-γ||x - y||²)

gamma (γ):
- 데이터 포인트의 영향 범위 결정
- gamma 큼: 각 포인트 영향 좁음 → 복잡한 경계, 과적합 위험
- gamma 작음: 각 포인트 영향 넓음 → 단순한 경계, 과소적합 위험

gamma 설정:
- 'scale': 1 / (n_features * X.var()) - 기본값
- 'auto': 1 / n_features
"""

# gamma 효과
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
gamma_values = [0.1, 1, 10, 100]

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

for ax, gamma in zip(axes, gamma_values):
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=1)
    clf.fit(X, y)

    xlim = [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5]
    ylim = [X[:, 1].min() - 0.5, X[:, 1].max() + 0.5]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='green', linewidths=2)
    ax.set_title(f'gamma = {gamma}\nSVs: {len(clf.support_vectors_)}')

plt.tight_layout()
plt.show()
```

---

## 4. sklearn SVM 사용법

### 4.1 SVC (Support Vector Classification)

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링 (SVM은 스케일에 민감)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 학습
svm_clf = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,  # 확률 예측 활성화
    random_state=42
)
svm_clf.fit(X_train_scaled, y_train)

# 예측
y_pred = svm_clf.predict(X_test_scaled)

print("SVM 분류 결과:")
print(f"  정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 4.2 확률 예측

```python
# 확률 예측 (probability=True 필요)
y_proba = svm_clf.predict_proba(X_test_scaled[:5])

print("확률 예측 (처음 5개):")
print(y_proba)
print(f"\n예측 클래스: {y_pred[:5]}")
print(f"실제 클래스: {y_test[:5]}")
```

### 4.3 SVR (Support Vector Regression)

```python
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR 학습
svr = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,  # 튜브 폭: 이 안의 오차는 무시
    gamma='scale'
)
svr.fit(X_train_scaled, y_train)

# 예측
y_pred = svr.predict(X_test_scaled)

print("SVR 회귀 결과:")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R²: {r2_score(y_test, y_pred):.4f}")

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'SVR Regression (R² = {r2_score(y_test, y_pred):.4f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. 다중 클래스 분류

```python
"""
SVM은 본질적으로 이진 분류기
다중 클래스 처리 방법:

1. OvO (One-vs-One):
   - 모든 클래스 쌍에 대해 분류기 생성
   - k 클래스 → k(k-1)/2 분류기
   - sklearn의 SVC 기본 방식

2. OvR (One-vs-Rest):
   - 각 클래스 vs 나머지 모든 클래스
   - k 클래스 → k 분류기
   - LinearSVC 기본 방식
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# OvO (기본)
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svm_ovo.fit(X_train_scaled, y_train)
print(f"OvO 정확도: {svm_ovo.score(X_test_scaled, y_test):.4f}")

# OvR
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
svm_ovr.fit(X_train_scaled, y_train)
print(f"OvR 정확도: {svm_ovr.score(X_test_scaled, y_test):.4f}")

# LinearSVC (OvR 기본)
linear_svc = LinearSVC(dual=True, max_iter=10000)
linear_svc.fit(X_train_scaled, y_train)
print(f"LinearSVC 정확도: {linear_svc.score(X_test_scaled, y_test):.4f}")
```

---

## 6. 하이퍼파라미터 튜닝

### 6.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV

# 파라미터 그리드
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid Search
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\nGrid Search 결과:")
print(f"  최적 파라미터: {grid_search.best_params_}")
print(f"  최적 CV 점수: {grid_search.best_score_:.4f}")
print(f"  테스트 점수: {grid_search.score(X_test_scaled, y_test):.4f}")
```

### 6.2 C와 gamma 동시 튜닝 시각화

```python
from sklearn.datasets import load_breast_cancer

# 데이터
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# C와 gamma 그리드
C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-3, 1, 5)

# 점수 계산
scores = np.zeros((len(C_range), len(gamma_range)))

for i, C in enumerate(C_range):
    for j, gamma in enumerate(gamma_range):
        svm_clf = SVC(C=C, gamma=gamma, kernel='rbf')
        svm_clf.fit(X_train_scaled, y_train)
        scores[i, j] = svm_clf.score(X_test_scaled, y_test)

# 히트맵 시각화
plt.figure(figsize=(10, 8))
plt.imshow(scores, interpolation='nearest', cmap='viridis')
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar(label='Accuracy')
plt.xticks(np.arange(len(gamma_range)), [f'{g:.3f}' for g in gamma_range])
plt.yticks(np.arange(len(C_range)), [f'{c:.2f}' for c in C_range])
plt.title('SVM Hyperparameter Tuning (RBF Kernel)')

# 최적점 표시
best_i, best_j = np.unravel_index(scores.argmax(), scores.shape)
plt.scatter(best_j, best_i, marker='*', s=300, c='red', edgecolors='white')

plt.tight_layout()
plt.show()

print(f"최적 C: {C_range[best_i]:.2f}")
print(f"최적 gamma: {gamma_range[best_j]:.3f}")
print(f"최고 정확도: {scores.max():.4f}")
```

---

## 7. 스케일링의 중요성

```python
# 스케일링 효과 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 스케일링 없이
svm_no_scale = SVC(kernel='rbf', C=1, gamma='scale')
svm_no_scale.fit(X_train, y_train)
acc_no_scale = svm_no_scale.score(X_test, y_test)

# 스케일링 후
svm_scaled = SVC(kernel='rbf', C=1, gamma='scale')
svm_scaled.fit(X_train_scaled, y_train)
acc_scaled = svm_scaled.score(X_test_scaled, y_test)

# 시각화 (처음 두 특성만)
for ax, (X_tr, X_te, title, acc) in zip(
    axes,
    [(X_train[:, :2], X_test[:, :2], 'Without Scaling', acc_no_scale),
     (X_train_scaled[:, :2], X_test_scaled[:, :2], 'With Scaling', acc_scaled)]
):
    svm_temp = SVC(kernel='rbf', C=1, gamma='scale')
    svm_temp.fit(X_tr, y_train)

    xlim = [X_tr[:, 0].min() - 0.5, X_tr[:, 0].max() + 0.5]
    ylim = [X_tr[:, 1].min() - 0.5, X_tr[:, 1].max() + 0.5]
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))

    Z = svm_temp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_train, cmap='coolwarm', edgecolors='black')
    ax.set_title(f'{title}\nAccuracy: {acc:.4f}')

plt.tight_layout()
plt.show()

print(f"스케일링 없이 정확도: {acc_no_scale:.4f}")
print(f"스케일링 후 정확도: {acc_scaled:.4f}")
```

---

## 8. SVM의 장단점

```python
"""
장점:
1. 고차원 데이터에 효과적
   - 특성 수 > 샘플 수인 경우에도 잘 작동

2. 메모리 효율적
   - 서포트 벡터만 저장

3. 다양한 커널
   - 비선형 문제 해결 가능
   - 커스텀 커널 정의 가능

4. 일반화 성능
   - 마진 최대화로 과적합 방지

단점:
1. 대용량 데이터에 느림
   - O(n²) ~ O(n³) 시간 복잡도

2. 스케일링 필수
   - 특성 스케일에 민감

3. 파라미터 튜닝
   - C, gamma 등 튜닝 필요

4. 확률 예측
   - 기본적으로 확률 출력 안함
   - probability=True 시 추가 비용

5. 해석 어려움
   - 블랙박스 모델
"""
```

---

## 9. 대용량 데이터 처리

```python
from sklearn.linear_model import SGDClassifier

"""
대용량 데이터용 SVM 대안:

1. LinearSVC:
   - liblinear 라이브러리 사용
   - 선형 커널만 지원
   - 대용량 데이터에 효율적

2. SGDClassifier:
   - 확률적 경사 하강법
   - loss='hinge'로 SVM 근사
   - 온라인 학습 가능
"""

# SGDClassifier (SVM 근사)
sgd_svm = SGDClassifier(
    loss='hinge',          # 힌지 손실 (SVM)
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)
sgd_svm.fit(X_train_scaled, y_train)

print("SGDClassifier (SVM 근사) 결과:")
print(f"  정확도: {sgd_svm.score(X_test_scaled, y_test):.4f}")

# LinearSVC
from sklearn.svm import LinearSVC

linear_svm = LinearSVC(C=1.0, max_iter=10000, dual=True)
linear_svm.fit(X_train_scaled, y_train)

print(f"LinearSVC 정확도: {linear_svm.score(X_test_scaled, y_test):.4f}")
```

---

## 10. 커스텀 커널

```python
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

# 커스텀 커널 함수
def custom_kernel(X1, X2):
    """RBF + 선형 커널 조합"""
    return rbf_kernel(X1, X2, gamma=0.1) + 0.1 * np.dot(X1, X2.T)

# 커스텀 커널 사용
svm_custom = SVC(kernel=custom_kernel)
svm_custom.fit(X_train_scaled, y_train)

print(f"커스텀 커널 정확도: {svm_custom.score(X_test_scaled, y_test):.4f}")

# 사전 계산된 커널 사용
from sklearn.metrics.pairwise import pairwise_kernels

# 커널 행렬 사전 계산
K_train = pairwise_kernels(X_train_scaled, metric='rbf', gamma=0.1)
K_test = pairwise_kernels(X_test_scaled, X_train_scaled, metric='rbf', gamma=0.1)

# precomputed 커널 SVM
svm_precomputed = SVC(kernel='precomputed')
svm_precomputed.fit(K_train, y_train)
y_pred = svm_precomputed.predict(K_test)

print(f"Precomputed 커널 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 연습 문제

### 문제 1: 기본 SVM 분류
유방암 데이터로 SVM을 학습하고 평가하세요.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 풀이
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train_scaled, y_train)

print(f"정확도: {svm.score(X_test_scaled, y_test):.4f}")
print(f"서포트 벡터 수: {len(svm.support_vectors_)}")
```

### 문제 2: 커널 비교
여러 커널의 성능을 비교하세요.

```python
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# 풀이
print("커널별 성능:")
for kernel in kernels:
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, gamma='scale')
    else:
        svm = SVC(kernel=kernel, gamma='scale')

    svm.fit(X_train_scaled, y_train)
    acc = svm.score(X_test_scaled, y_test)
    print(f"  {kernel}: {acc:.4f}")
```

### 문제 3: 하이퍼파라미터 튜닝
Grid Search로 최적의 C와 gamma를 찾으세요.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1]
}

# 풀이
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 CV 점수: {grid.best_score_:.4f}")
print(f"테스트 점수: {grid.score(X_test_scaled, y_test):.4f}")
```

---

## 요약

| 개념 | 설명 | 파라미터 |
|------|------|----------|
| 서포트 벡터 | 마진 경계의 데이터 포인트 | - |
| 마진 | 결정 경계와 서포트 벡터 거리 | C로 조절 |
| C | 규제 파라미터 | 큼: 좁은 마진, 작음: 넓은 마진 |
| 커널 | 데이터 변환 함수 | linear, poly, rbf, sigmoid |
| gamma | RBF 커널 범위 | 큼: 좁은 영향, 작음: 넓은 영향 |

### SVM 사용 체크리스트

1. **스케일링 필수**: StandardScaler 또는 MinMaxScaler 적용
2. **커널 선택**: 선형 분리 가능 → linear, 그외 → rbf
3. **파라미터 튜닝**: C와 gamma Grid Search
4. **대용량 데이터**: LinearSVC 또는 SGDClassifier 사용
5. **확률 필요시**: probability=True 설정

### 커널 선택 가이드

| 상황 | 권장 커널 |
|------|-----------|
| 선형 분리 가능 | linear |
| 비선형 패턴 | rbf |
| 다항식 관계 | poly |
| 특성 수 >> 샘플 수 | linear |
| 잘 모르겠음 | rbf (기본) |

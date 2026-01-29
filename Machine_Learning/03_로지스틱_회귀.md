# 로지스틱 회귀 (Logistic Regression)

## 개요

로지스틱 회귀는 이름과 달리 분류 알고리즘입니다. 이진 분류와 다중 분류 문제에서 확률을 예측합니다.

---

## 1. 이진 분류

### 1.1 시그모이드 함수

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 시그모이드 함수 시각화
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 5))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('시그모이드 함수')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.show()

# 특성:
# - 출력 범위: (0, 1) → 확률로 해석 가능
# - z=0일 때 0.5
# - z → ∞ 일 때 1, z → -∞ 일 때 0
```

### 1.2 로지스틱 회귀 모델

```
P(y=1|X) = σ(θᵀX) = 1 / (1 + e^(-θᵀX))

결정 경계:
- P(y=1|X) >= 0.5 → 클래스 1 예측
- P(y=1|X) < 0.5 → 클래스 0 예측
```

### 1.3 sklearn 구현

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 유방암 데이터셋 (이진 분류)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"클래스: {cancer.target_names}")
print(f"특성 수: {X.shape[1]}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# 평가
print(f"\n정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 예측 확률 예시
print(f"\n첫 5개 샘플 예측 확률:")
for i in range(5):
    print(f"  샘플 {i}: {cancer.target_names[0]}={y_proba[i][0]:.3f}, "
          f"{cancer.target_names[1]}={y_proba[i][1]:.3f} → 예측: {cancer.target_names[y_pred[i]]}")
```

---

## 2. 비용 함수와 최적화

### 2.1 로그 손실 (Log Loss / Binary Cross-Entropy)

```python
# 비용 함수:
# J(θ) = -1/m * Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]

from sklearn.metrics import log_loss

# 예시
y_true = [0, 0, 1, 1]
y_proba = [0.1, 0.4, 0.35, 0.8]

loss = log_loss(y_true, y_proba)
print(f"Log Loss: {loss:.4f}")

# 완벽한 예측
y_proba_perfect = [0.0, 0.0, 1.0, 1.0]
loss_perfect = log_loss(y_true, y_proba_perfect)
print(f"완벽한 예측 Log Loss: {loss_perfect:.4f}")
```

### 2.2 경사하강법

```python
def logistic_regression_gd(X, y, learning_rate=0.1, n_iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # bias 추가
    theta = np.zeros(n + 1)

    for _ in range(n_iterations):
        z = X_b @ theta
        h = sigmoid(z)
        gradient = (1/m) * X_b.T @ (h - y)
        theta = theta - learning_rate * gradient

    return theta

# 테스트
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)

theta = logistic_regression_gd(X, y)
print(f"학습된 계수: {theta}")
```

---

## 3. 정규화

### 3.1 L2 정규화 (기본값)

```python
# penalty='l2' (기본값)
# C = 1/λ (작을수록 강한 정규화)

Cs = [0.001, 0.01, 0.1, 1, 10, 100]

for C in Cs:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"C={C:6}: Train={train_acc:.4f}, Test={test_acc:.4f}")
```

### 3.2 L1 정규화 (Lasso)

```python
# 특성 선택 효과
model_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)
model_l1.fit(X_train_scaled, y_train)

# 0이 아닌 계수 수
non_zero = np.sum(model_l1.coef_ != 0)
print(f"L1 정규화: 0이 아닌 계수 = {non_zero}/{X.shape[1]}")
print(f"정확도: {model_l1.score(X_test_scaled, y_test):.4f}")
```

### 3.3 Elastic Net

```python
model_en = LogisticRegression(penalty='elasticnet', solver='saga',
                              l1_ratio=0.5, C=1, max_iter=1000)
model_en.fit(X_train_scaled, y_train)
print(f"Elastic Net 정확도: {model_en.score(X_test_scaled, y_test):.4f}")
```

---

## 4. 다중 클래스 분류

### 4.1 One-vs-Rest (OvR)

```python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# OvR (기본값 for multi_class='ovr')
model_ovr = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ovr.fit(X_train, y_train)

print(f"OvR 정확도: {model_ovr.score(X_test, y_test):.4f}")
print(f"계수 형태: {model_ovr.coef_.shape}")  # (3, 4) = 클래스 수 x 특성 수
```

### 4.2 Softmax (Multinomial)

```python
# Softmax 함수: 각 클래스 확률 출력
# P(y=k|X) = exp(θₖᵀX) / Σexp(θⱼᵀX)

model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_softmax.fit(X_train, y_train)

print(f"Softmax 정확도: {model_softmax.score(X_test, y_test):.4f}")

# 예측 확률
y_proba = model_softmax.predict_proba(X_test[:3])
print("\n예측 확률 (첫 3개 샘플):")
for i, proba in enumerate(y_proba):
    print(f"  샘플 {i}: {proba} → 예측: {iris.target_names[np.argmax(proba)]}")
```

### 4.3 비교

```python
from sklearn.model_selection import cross_val_score

models = {
    'OvR': LogisticRegression(multi_class='ovr', max_iter=1000),
    'Multinomial': LogisticRegression(multi_class='multinomial', max_iter=1000)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 5. 결정 경계 시각화

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 2D 데이터 생성
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X, y)

# 결정 경계 시각화
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('로지스틱 회귀 결정 경계')
    plt.show()

plot_decision_boundary(model, X, y)

# 확률 경계 시각화
def plot_probability_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, levels=20, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='P(y=1)')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('예측 확률과 결정 경계 (0.5)')
    plt.show()

plot_probability_boundary(model, X, y)
```

---

## 6. 임계값 조정

```python
from sklearn.metrics import precision_recall_curve, roc_curve

# 데이터 준비
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

y_proba = model.predict_proba(X_test_s)[:, 1]

# 다양한 임계값으로 예측
thresholds = [0.3, 0.5, 0.7]

print("임계값에 따른 성능:")
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    print(f"  threshold={thresh}: Precision={prec:.3f}, Recall={rec:.3f}")

# Precision-Recall 곡선
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(thresholds_pr, precision[:-1], 'b-', label='Precision')
plt.plot(thresholds_pr, recall[:-1], 'r-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision/Recall vs Threshold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. 불균형 데이터 처리

```python
from sklearn.datasets import make_classification

# 불균형 데이터 생성
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_features=10, random_state=42)

print(f"클래스 분포: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기본 모델
model_default = LogisticRegression(max_iter=1000)
model_default.fit(X_train, y_train)

# class_weight='balanced'
model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
model_balanced.fit(X_train, y_train)

# 비교
from sklearn.metrics import classification_report

print("=== 기본 모델 ===")
print(classification_report(y_test, model_default.predict(X_test)))

print("=== class_weight='balanced' ===")
print(classification_report(y_test, model_balanced.predict(X_test)))
```

---

## 연습 문제

### 문제 1: 이진 분류
유방암 데이터로 로지스틱 회귀 모델을 학습하고 F1-score를 구하세요.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 풀이
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
```

### 문제 2: 다중 분류
Iris 데이터로 3-클래스 분류를 수행하세요.

```python
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 풀이
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)
print(f"정확도: {model.score(X_test, y_test):.4f}")
print(f"\n예측 확률 (첫 샘플): {model.predict_proba(X_test[:1])}")
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 시그모이드 | 확률 출력 (0~1) |
| Log Loss | 비용 함수 (Binary Cross-Entropy) |
| OvR | 다중 분류 (One-vs-Rest) |
| Softmax | 다중 분류 (Multinomial) |
| C | 정규화 강도 (1/λ) |
| class_weight | 불균형 데이터 처리 |

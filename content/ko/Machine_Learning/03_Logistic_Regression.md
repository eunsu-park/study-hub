# 로지스틱 회귀 (Logistic Regression)

**이전**: [선형회귀](./02_Linear_Regression.md) | **다음**: [모델 평가](./04_Model_Evaluation.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이름에도 불구하고 로지스틱 회귀가 분류 알고리즘인 이유를 설명할 수 있습니다
2. 시그모이드 함수(sigmoid function)가 선형 출력을 확률로 매핑하는 방식을 설명할 수 있습니다
3. scikit-learn과 경사 하강법(gradient descent)을 이용한 직접 구현으로 이진 로지스틱 회귀를 구현할 수 있습니다
4. 로지스틱 회귀 맥락에서 L1, L2, Elastic Net 정규화(regularization)를 비교할 수 있습니다
5. 다중 클래스 분류에서 일대다(One-vs-Rest)와 소프트맥스(Softmax) 방식을 구별할 수 있습니다
6. 주어진 문제에서 정밀도(precision)와 재현율(recall)의 균형을 맞추기 위해 임계값(threshold) 조정을 적용할 수 있습니다
7. 클래스 가중치(class weighting)가 불균형 데이터셋을 처리하는 방법을 설명할 수 있습니다

---

이름과 달리 로지스틱 회귀는 분류 알고리즘이며, 가장 중요하게 이해해야 할 알고리즘 중 하나입니다. 특성들의 선형 결합을 시그모이드 함수에 통과시켜 클래스 확률을 예측하며, 신경망과 더 발전된 분류기를 이해하는 토대가 되는 해석 가능한 모델을 제공합니다.

---

## 이론과 원리

로지스틱 회귀는 선형 회귀의 작은 수정처럼 보입니다 — 선형 출력에 시그모이드를 씌우는 것 — 하지만 완전히 다른 원리에 기반합니다: 베르누이(Bernoulli) 모델 하의 최대가능도 추정(maximum likelihood estimation, MLE). 그 한 가지 변경이 다른 손실 함수(교차 엔트로피), 다른 그래디언트, 다른 기하학적 해석(로그-오즈 공간에서의 선형 결정 경계)으로 연쇄됩니다.

### A. 시그모이드와 그 특별한 도함수가 가진 의미

`σ(z) = 1 / (1 + e^{-z})`라 정의합니다. 두 가지 성질이 중요합니다:

```
σ(z) ∈ (0, 1)               ← 확률로 해석 가능
σ(-z) = 1 - σ(z)            ← 대칭
σ'(z) = σ(z) · (1 - σ(z))   ← 도함수가 σ 자체로 표현됨
```

마지막 항등식이 시그모이드(다른 0–1 압축 함수가 아니라)가 표준이 된 이유입니다. 유도해보면:

```
σ(z)  = 1 / (1 + e^{-z})
σ'(z) = e^{-z} / (1 + e^{-z})²
      = [1 / (1 + e^{-z})] · [e^{-z} / (1 + e^{-z})]
      = σ(z) · (1 - σ(z))
```

도함수가 입력이 아니라 시그모이드의 *출력*에만 의존합니다. 경사 하강 도중 정방향 패스(forward pass)에서 이미 `σ(z)`를 계산해 두었으니 — 그래디언트를 얻는 데 추가 비용이 거의 없습니다. 이것이 신경망에서 시그모이드(그리고 tanh, 소프트맥스)를 수치적으로 효율적이게 만드는 같은 성질입니다.

### B. 최대가능도에서 교차 엔트로피로의 유도

각 레이블을 베르누이로 모델링합니다: `P(y = 1 | x) = σ(βᵀx) = p`, `P(y = 0 | x) = 1 - p`. 한 예제에 대해 다음과 같이 간결하게 표현됩니다:

```
P(y | x; β) = p^y · (1 - p)^{1-y}
```

데이터셋의 가능도(likelihood)는 `N`개 독립 예제에 대한 곱이고, 로그를 취하면 로그 가능도가 됩니다:

```
ℓ(β) = Σ_i [ y_i · log p_i + (1 - y_i) · log(1 - p_i) ]
```

MLE는 `ℓ(β)`를 최대화하며, 동등하게 음의 로그 가능도(negative log-likelihood)를 *최소화*합니다 — 이것이 정확히 **이진 교차 엔트로피 손실(binary cross-entropy loss)**입니다:

```
L_CE(β) = - (1/N) · Σ_i [ y_i · log p_i + (1 - y_i) · log(1 - p_i) ]
```

따라서 교차 엔트로피는 휴리스틱이 아니며 — 베르누이 모델의 음의 로그 가능도입니다. 그래디언트는 시그모이드 항등식 덕분에 아름답게 단순화됩니다:

```
∂L/∂β = (1/N) · Xᵀ (p - y)            ← OLS 그래디언트와 같은 형태
```

그래디언트는 설계 행렬 곱하기 잔차 `p - y`입니다. `p`를 `Xβ`로 바꾸면 OLS 그래디언트가 돌아옵니다 — 두 알고리즘은 그래디언트의 *형태*를 공유하며, 단지 "예측값"이 다를 뿐입니다.

OLS와 달리 이 손실은 닫힌 형태의 최소화기가 없습니다(시그모이드가 그래디언트 안의 선형성을 깹니다). 반복법으로 풉니다: 경사 하강, 뉴턴-랩슨(IRLS — 반복 가중 최소제곱을 줌), 또는 준-뉴턴(L-BFGS, scikit-learn의 기본).

### C. 다항 확장: 소프트맥스

`K`개 클래스에 대해 시그모이드를 **소프트맥스(softmax)**로 대체합니다:

```
p_k = exp(β_kᵀ x) / Σ_j exp(β_jᵀ x)         k = 1, ..., K
```

소프트맥스는 `K`개 클래스에 대한 유효한 확률 분포를 출력합니다(양수, 합이 1). 손실은 **범주형 교차 엔트로피(categorical cross-entropy)**로 일반화됩니다:

```
L_CE = - (1/N) · Σ_i Σ_k [ 1{y_i = k} · log p_{i,k} ]
```

그래디언트는 다시 우아한 형태 `∂L/∂β_k = (1/N) · Xᵀ (p_{·,k} - 1{y = k})`를 가집니다. 이진 교차 엔트로피는 가중치 벡터 둘 대신 하나로 작동하는 `K = 2` 특수 사례입니다(두 번째는 `Σ_k p_k = 1` 중복성에 의해 0으로 고정).

소프트맥스에는 비유일성이 있습니다: 모든 `β_kᵀ x`에 같은 상수를 더해도 모든 `p_k`가 변하지 않으므로, 매개변수는 하나의 가산 이동(additive shift)을 제외하고만 식별 가능합니다. scikit-learn은 `K-1`개 가중치 벡터를 적합(정칙화와 짝지을 때의 `multinomial` 형식)하거나 `K`개의 독립적 일대다(one-vs-rest) 이진 문제를 실행하여 이를 해소합니다.

### D. 로그-오즈 공간에서의 선형 결정 경계

로지스틱 회귀는 명확하지 않은 의미에서 *선형*입니다. 모델의 예측 `p`는 `x`의 비선형 함수이지만, **로그-오즈(logit)**는 선형입니다:

```
logit(p) = log(p / (1-p)) = βᵀx
```

즉, 로지스틱 회귀에 내장된 가정은: 양성 클래스의 로그-오즈가 특성의 선형 함수라는 것입니다. 결정 경계 `p = 0.5`는 `βᵀx = 0`에 해당합니다 — 특성 공간의 초평면, 정확히 선형 분류기처럼. 시그모이드는 단지 *신뢰도*가 경계로부터의 거리에 따라 어떻게 변하는지를 통제할 뿐, 경계 자체를 곡선으로 만들지 않습니다.

두 가지 실용적 결과:

1. **계수가 로그-오즈비(log-odds-ratio)로 해석 가능합니다.** 특성 `x_j`의 단위 증가는 오즈 `p / (1-p)`에 `exp(β_j)`를 곱합니다. 이 성질이 로지스틱 회귀를 의학, 신용 점수, 그리고 이해관계자가 계수 해석성을 요구하는 다른 영역에서 대체 불가능하게 만듭니다.
2. **비선형 경계에는 특성 공학이 필요합니다.** 진짜 경계가 굽어 있다면, 평범한 로지스틱 회귀는 그것을 적합할 수 없습니다. 다항 특성, 상호작용, 또는 커널 트릭(Lesson 9)이 같은 알고리즘을 비선형 결정면으로 확장합니다 — 하지만 *로그-오즈 선형성* 가정은 유지됩니다.

정칙화(L1/L2)는 선형 회귀와 같은 `λ ‖β‖_p` 페널티를 `L_CE`에 더하며, 같은 효과를 가집니다: L2는 상관 특성을 안정화하고 L1은 희소성을 만듭니다. scikit-learn의 `LogisticRegression(penalty=...)`이 둘 다 노출합니다.

### From Theory to the Code Below

- 섹션 1.1의 `sigmoid(z)`는 (A)의 함수이며, 플롯은 `σ' = σ(1-σ)`에서 발생하는 S 자형을 보여줍니다.
- 섹션 1.2의 `LogisticRegression().fit(X, y)`는 (B)의 이진 교차 엔트로피를 기본적으로 L-BFGS로 최소화합니다.
- 섹션 2에 그려진 결정 경계는 (D)의 `βᵀx = 0` 초평면입니다 — 예측 확률이 곡선으로 변하더라도 원시 특성 공간에서는 직선입니다.
- 섹션 3의 `multi_class='multinomial'`은 손실을 (C)의 범주형 교차 엔트로피로 전환합니다; 소프트맥스가 예측 단계 내부에서 계산됩니다.
- scikit-learn의 `C` 매개변수는 정칙화 강도 `λ`의 *역수*입니다: `C`가 작을수록 ⟹ 강한 정칙화. (이것이 가장 자주 발생하는 버그 중 하나입니다 — `Ridge(alpha=...)`와 반대 관례.)

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

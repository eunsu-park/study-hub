# 선형회귀 (Linear Regression)

**이전**: [머신러닝 개요](./01_ML_Overview.md) | **다음**: [로지스틱 회귀](./03_Logistic_Regression.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단순 선형회귀와 다중 선형회귀의 수학적 공식을 설명할 수 있습니다
2. 해석적 해(OLS) 방법과 경사 하강법(gradient descent) 모두를 사용하여 선형회귀를 구현할 수 있습니다
3. 배치(batch), 확률적(stochastic), 미니배치(mini-batch) 경사 하강법 방식을 비교할 수 있습니다
4. 릿지(Ridge, L2), 라쏘(Lasso, L1), 엘라스틱넷(Elastic Net) 정규화가 과적합(overfitting)을 방지하는 방법을 설명할 수 있습니다
5. 비선형 관계를 모델링하기 위해 다항 회귀(polynomial regression)를 적용할 수 있습니다
6. 회귀 평가 지표(MSE, RMSE, MAE, R-squared, MAPE)를 계산하고 해석할 수 있습니다
7. 특성 특성에 따라 릿지(Ridge)와 라쏘(Lasso) 중 언제 사용할지 구분할 수 있습니다

---

선형회귀는 연속적인 값을 예측하는 가장 기본적인 회귀 알고리즘으로, 실무에서 가장 널리 사용되는 모델 중 하나입니다. 입력 변수와 출력 변수 간의 선형 관계를 모델링함으로써, 더 복잡한 알고리즘에 도전하기 전에 모든 실무자가 마스터해야 할 해석 가능한 기준선(baseline)을 제공합니다.

---

## 이론과 원리

scikit-learn API는 하나의 동사(`.fit()`) 뒤에 매우 다른 네 가지 수학적 객체를 숨깁니다: 닫힌 형태의 최소제곱(closed-form least squares) 해, 경사 기반 반복 솔버, 그리고 L2 / L1 / Elastic Net 정칙화(regularizer). 이들 사이에서 선택하려면 각자가 실제로 무엇을 풀고 있고 어디서 깨지는지 알아야 합니다.

### A. 보통최소제곱(OLS): 닫힌 형태의 해

`N`개의 관측치를 설계 행렬(design matrix) `X ∈ ℝ^{N×p}`(절편을 위한 1로 채운 첫 열 포함)과 타깃 벡터 `y ∈ ℝ^N`으로 쌓습니다. OLS 목적함수는

```
L(β) = ‖y - Xβ‖² = (y - Xβ)ᵀ(y - Xβ)
```

`β`에 대한 그래디언트를 0으로 설정합니다:

```
∇_β L = -2 Xᵀ(y - Xβ) = 0
   ⟹  XᵀX β = Xᵀy
   ⟹  β̂ = (XᵀX)⁻¹ Xᵀy            ← 정규방정식(normal equations)
```

헤시안(Hessian) `2 XᵀX`는 양반정치(positive semi-definite)이므로 임계점이 전역 최소입니다. 이는 *정확한* 해를 갖는 몇 안 되는 ML 알고리즘 중 하나입니다 — 반복 없음, 학습률 없음, 수렴 걱정 없음.

함정은 역행렬 `(XᵀX)⁻¹`입니다. `XᵀX`가 완전 계수(full rank)를 가질 때 — 즉 `X`의 열들이 선형 독립일 때 — 만 존재합니다. 두 가지 경우 실패합니다: (1) 표본보다 특성이 많을 때(`p > N`), (2) 두 특성이 완전히 상관될 때. 두 경우 모두 시스템이 무한히 많은 최소제곱 해를 가지게 되어 OLS가 잘 정의되지 않습니다. 수치적으로는 *근사적으로* 특이행렬(singular)이기만 해도 `β̂`의 분산이 폭발합니다.

계산 비용은 `O(p² N + p³)` — 수천 특성까지는 괜찮지만 수백만 특성에는 비현실적입니다. 그 한계 아래에서는 OLS가 최적의 출발점입니다: 결정적, 재현 가능, 보정됨.

### B. 경사 하강법: 정확성과 확장성의 교환

`p`나 `N`이 정규방정식에 너무 크면, 같은 손실을 반복적으로 최소화합니다. 갱신 규칙은

```
β_{t+1} = β_t - η · ∇_β L(β_t) = β_t + (2η/N) · Xᵀ(y - X β_t)
```

세 가지 변형은 단지 각 단계에서 *어떤* 그래디언트를 쓰는지가 다릅니다:

- **Batch GD**: 전체 `N` 예제에 대한 그래디언트. 매끄러운 하강, 단계당 비싼 계산.
- **Stochastic GD (SGD)**: 무작위 한 예제에 대한 그래디언트. 단계당 저렴, 잡음 있는 궤적 — 잡음이 오히려 날카로운 곡률 영역을 *탈출*하는 데 도움이 될 수 있습니다.
- **Mini-batch GD**: `B`개 예제에 대한 그래디언트(통상 `B = 32-512`). 균형점: 안정성 충분한 평균화, 벡터화 하드웨어에서 돌릴 만큼 작은 크기.

볼록한 선형회귀 손실에서는 학습률이 충분히 작으면 세 가지 모두 *같은* OLS 최적점에 수렴합니다. 차이는 벽시계 시간 비용이지 답이 아닙니다. 스텝 크기 선택은 중요합니다: 너무 크면 발산, 너무 작으면 계산 낭비. 실용적인 기본값(Adam, 학습률 스케줄)이 이를 자동화해 줍니다.

### C. 정칙화: 해를 제약하기

`XᵀX`가 거의 특이이거나 `p`가 `N`과 비슷할 때 OLS는 분산이 큽니다: `y`의 작은 변화가 `β̂`에 큰 변동을 일으킵니다. 정칙화는 `β`를 0으로 끌어당기는 페널티를 추가해 약간의 편향을 받고 큰 분산 감소를 얻습니다.

#### C.1 Ridge (L2): 닫힌 형태가 살아남음

```
β̂_ridge = argmin_β  ‖y - Xβ‖² + λ ‖β‖²₂
        = (XᵀX + λI)⁻¹ Xᵀy
```

`λI`를 더하면 어떤 경우에도 행렬이 가역적이 됩니다 — 이 한 가지 사실이 Ridge가 존재하는 이유의 절반입니다. `λ → 0`이면 OLS로 환원되고, `λ → ∞`이면 모든 계수가 0으로 수축됩니다. 계수가 *비례적*으로 수축되지만 0에 정확히 도달하지는 않으므로 Ridge는 모든 특성을 모델에 유지합니다.

#### C.2 Lasso (L1): 희소성을 만들어내는 이유

```
β̂_lasso = argmin_β  ‖y - Xβ‖² + λ ‖β‖₁
```

L1 페널티는 `β_j = 0`에서 미분 불가능합니다. 부분미분(subdifferential)을 사용합니다:

```
∂|β_j| = { sign(β_j)         β_j ≠ 0이면
         { [-1, +1]           β_j = 0이면
```

좌표 `j`에 대한 최적성 조건은 `(Xᵀ(y - Xβ))_j ∈ λ · ∂|β_j|`입니다. 페널티 없는 잔차 상관 `|(Xᵀ(y - Xβ))_j|`이 `λ`보다 작으면, 포함을 만족시키는 유일한 방법은 `β_j = 0`을 정확히 설정하는 것뿐입니다. 기하학적으로 L1 공은 축에 모서리가 있고 — 제약된 최소제곱 해는 그 모서리에 떨어지는 경향이 있습니다. L2 공은 둥글고 모서리가 없으므로 Ridge는 정확한 0을 만들지 못합니다.

이것이 Lasso를 선택하는 *바로 그* 이유입니다: 자동 특성 선택. 대가는 닫힌 형태가 없다는 것(좌표 하강 또는 근사 경사법이 필요)과 특성이 강하게 상관될 때 불안정성 — Lasso는 상관된 그룹 중 하나를 임의로 고르고 나머지를 0으로 만듭니다.

#### C.3 Elastic Net: 볼록 조합

```
β̂_en = argmin_β  ‖y - Xβ‖² + λ [α ‖β‖₁ + (1-α) ‖β‖²₂]
```

L1과 L2를 섞는 가중치 `α ∈ [0, 1]`로 혼합합니다. Lasso의 희소성과 상관 특성에 대한 Ridge의 안정성을 둘 다 상속받습니다(상관된 특성을 하나만 고르기보다 그룹으로 유지하는 경향). `α = 1`이면 Lasso, `α = 0`이면 Ridge로 환원됩니다.

### D. 올바른 도구 선택

| 상황 | 최선의 선택 | 이유 |
|-----|-----------|------|
| `p`가 작고 `XᵀX`가 잘 조건화됨 | OLS | 정확, 무료 |
| 상관된 특성이 많고 모두 유지하고 싶음 | Ridge | 안정적 수축, 닫힌 형태 |
| `p > N` 또는 특성 선택이 필요 | Lasso | L1 모서리에서 오는 희소성 |
| 상관된 특성 + 희소성 필요 | Elastic Net | 그룹 선택 |
| `N > 10⁶` | SGD / mini-batch | 정규방정식 비현실적 |

선택은 스타일이 아니라 — `XᵀX`의 조건수와 계수 벡터에 무엇을 원하는지에서 따라옵니다.

### From Theory to the Code Below

- 섹션 1의 `LinearRegression().fit(X, y)`는 (A)의 정규방정식을 직접 풉니다.
- 섹션 2의 `SGDRegressor` 루프는 (B)의 경사 하강 점화식입니다.
- 섹션 3은 `Ridge`, `Lasso`, `ElasticNet`을 노출합니다 — (C)의 세 정칙화. scikit-learn의 `alpha` 매개변수는 우리 공식의 `λ`이고, `l1_ratio` 매개변수는 Elastic Net의 `α`입니다.
- 섹션 4의 다항 특성은 이 중 어떤 것도 바꾸지 않습니다 — 적합 전에 `X`를 확장할 뿐이므로 동일한 수학이 더 높은 차원의 설계 행렬에 적용됩니다.

---

## 1. 단순 선형회귀

### 1.1 개념

하나의 독립변수(X)로 종속변수(y)를 예측합니다.

```
y = β₀ + β₁x + ε

- β₀: 절편 (intercept)
- β₁: 기울기 (slope)
- ε: 오차항
```

### 1.2 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 계수 확인
print(f"절편 (β₀): {model.intercept_[0]:.4f}")
print(f"기울기 (β₁): {model.coef_[0][0]:.4f}")

# 예측
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
print(f"\n예측값: X=0 → y={y_pred[0][0]:.2f}, X=2 → y={y_pred[1][0]:.2f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='데이터')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label='회귀선')
plt.xlabel('X')
plt.ylabel('y')
plt.title('단순 선형회귀')
plt.legend()
plt.show()
```

### 1.3 최소자승법 (OLS)

```python
# 최소자승법: 잔차 제곱합(RSS)을 최소화
# RSS = Σ(yᵢ - ŷᵢ)²

# 수학적 해
X_b = np.c_[np.ones((100, 1)), X]  # bias 추가
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"수학적 해:")
print(f"θ₀ = {theta_best[0][0]:.4f}")
print(f"θ₁ = {theta_best[1][0]:.4f}")
```

---

## 2. 다중 선형회귀

### 2.1 개념

여러 개의 독립변수로 종속변수를 예측합니다.

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

### 2.2 구현

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 당뇨병 데이터셋
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"특성: {diabetes.feature_names}")
print(f"데이터 형태: {X.shape}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = model.predict(X_test_scaled)

print(f"\nMSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 계수 확인
coefficients = pd.DataFrame({
    'feature': diabetes.feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(f"\n회귀 계수:")
print(coefficients)
```

---

## 3. 경사하강법 (Gradient Descent)

### 3.1 배치 경사하강법

```python
# 비용 함수: J(θ) = (1/2m) Σ(h(xᵢ) - yᵢ)²
# 업데이트: θ = θ - α * ∇J(θ)

def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # bias 추가
    theta = np.random.randn(2, 1)  # 랜덤 초기화

    cost_history = []

    for iteration in range(n_iterations):
        gradients = (1/m) * X_b.T @ (X_b @ theta - y)
        theta = theta - learning_rate * gradients

        cost = (1/(2*m)) * np.sum((X_b @ theta - y)**2)
        cost_history.append(cost)

    return theta, cost_history

# 실행
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta, cost_history = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print(f"θ₀ = {theta[0][0]:.4f}")
print(f"θ₁ = {theta[1][0]:.4f}")

# 비용 함수 수렴 시각화
plt.figure(figsize=(10, 4))
plt.plot(cost_history[:100])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('경사하강법 수렴')
plt.show()
```

### 3.2 확률적 경사하강법 (SGD)

```python
from sklearn.linear_model import SGDRegressor

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.2)

# 스케일링 (SGD는 스케일링 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SGD 회귀
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None,
                       eta0=0.01, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)

print(f"SGD 절편: {sgd_reg.intercept_[0]:.4f}")
print(f"SGD 계수: {sgd_reg.coef_[0]:.4f}")
```

### 3.3 미니배치 경사하강법

```python
def mini_batch_gradient_descent(X, y, batch_size=20, learning_rate=0.01, n_epochs=50):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = (1/len(yi)) * xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradients

    return theta

theta = mini_batch_gradient_descent(X, y)
print(f"미니배치 GD 결과: θ₀={theta[0][0]:.4f}, θ₁={theta[1][0]:.4f}")
```

---

## 4. 정규화 (Regularization)

과적합을 방지하기 위해 모델의 복잡도에 패널티를 부여합니다.

### 4.1 Ridge 회귀 (L2 정규화)

```python
from sklearn.linear_model import Ridge

# 비용 함수: J(θ) = MSE + α * Σθᵢ²

# 다양한 alpha 값으로 실험
alphas = [0, 0.1, 1, 10, 100]

plt.figure(figsize=(12, 4))
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    print(f"Alpha={alpha}: R²={r2_score(y_test, y_pred):.4f}, 계수합={sum(abs(ridge.coef_)):.4f}")
```

### 4.2 Lasso 회귀 (L1 정규화)

```python
from sklearn.linear_model import Lasso

# 비용 함수: J(θ) = MSE + α * Σ|θᵢ|
# 특징: 일부 계수를 0으로 만듦 (특성 선택)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 0이 아닌 계수 확인
non_zero = np.sum(lasso.coef_ != 0)
print(f"0이 아닌 계수 수: {non_zero}/{len(lasso.coef_)}")

y_pred = lasso.predict(X_test_scaled)
print(f"Lasso R²: {r2_score(y_test, y_pred):.4f}")
```

### 4.3 Elastic Net

```python
from sklearn.linear_model import ElasticNet

# L1과 L2를 혼합
# 비용 함수: J(θ) = MSE + r*α*Σ|θᵢ| + (1-r)*α*Σθᵢ²/2

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = r
elastic.fit(X_train_scaled, y_train)

y_pred = elastic.predict(X_test_scaled)
print(f"Elastic Net R²: {r2_score(y_test, y_pred):.4f}")
```

### 4.4 정규화 비교

```python
from sklearn.datasets import make_regression

# 데이터 생성 (특성 > 샘플)
X, y = make_regression(n_samples=50, n_features=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 모델 비교
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

print("정규화 방법 비교:")
for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    non_zero = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else len(model.coef_)
    print(f"{name:12}: Train R²={train_score:.3f}, Test R²={test_score:.3f}, 비영 계수={non_zero}")
```

---

## 5. 다항 회귀

비선형 관계를 선형회귀로 모델링합니다.

```python
from sklearn.preprocessing import PolynomialFeatures

# 비선형 데이터 생성
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# 다항 특성 생성
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"원본 특성: {X.shape}")
print(f"다항 특성: {X_poly.shape}")
print(f"특성 이름: {poly.get_feature_names_out()}")

# 선형회귀 적용
model = LinearRegression()
model.fit(X_poly, y)

print(f"\n계수: {model.coef_}")
print(f"절편: {model.intercept_}")

# 시각화
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X_plot, y_plot, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('다항 회귀 (degree=2)')
plt.show()
```

---

## 6. 회귀 평가 지표

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# 예측
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

# MSE (Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# R² (결정계수)
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")
```

---

## 연습 문제

### 문제 1: 단순 선형회귀
다음 데이터로 선형회귀 모델을 학습하고 X=7일 때 예측값을 구하세요.

```python
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2, 4, 5, 4, 5, 7])

# 풀이
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[7]])
print(f"X=7일 때 예측값: {prediction[0]:.2f}")
print(f"R²: {model.score(X, y):.4f}")
```

### 문제 2: Ridge vs Lasso
당뇨병 데이터에서 Ridge와 Lasso의 성능을 비교하세요.

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 풀이
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for Model, name in [(Ridge, 'Ridge'), (Lasso, 'Lasso')]:
    model = Model(alpha=1)
    model.fit(X_train_s, y_train)
    print(f"{name} R²: {model.score(X_test_s, y_test):.4f}")
```

---

## 요약

| 방법 | 특징 | 사용 시점 |
|------|------|----------|
| 선형회귀 | 기본, 해석 용이 | 기준 모델 |
| Ridge (L2) | 계수 축소, 과적합 방지 | 다중공선성 |
| Lasso (L1) | 특성 선택, 희소 모델 | 많은 특성 |
| Elastic Net | L1+L2 혼합 | 상관된 특성 |
| 다항 회귀 | 비선형 관계 | 곡선 패턴 |

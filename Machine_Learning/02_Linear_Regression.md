# 선형회귀 (Linear Regression)

## 개요

선형회귀는 연속적인 값을 예측하는 가장 기본적인 회귀 알고리즘입니다. 입력 변수와 출력 변수 간의 선형 관계를 모델링합니다.

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

# 06. 베이지안 회귀(Bayesian Regression)

[이전: 계층 모델](./05_Hierarchical_Models.md) | [다음: Stan과 CmdStanPy](./07_Stan_and_CmdStanPy.md)

---

> **프레임워크 참고**: 이 레슨은 베이지안 회귀 모델링을 위해 PyMC 5.x를 사용합니다.
>
> 설치: `pip install pymc arviz numpy scipy matplotlib pandas`

## 학습 목표

- PyMC로 베이지안 선형 회귀 구현
- 일반화 선형 모델(GLM) 이해 및 사용
- 중두꺼운 꼬리 가능도를 사용한 로버스트 회귀 모델 구축
- 정보 기준을 사용한 모델 비교
- 베이지안 프레임워크에서 회귀 계수 해석

---

## 1. 베이지안 선형 회귀(Bayesian Linear Regression)

### 1.1 모델(The Model)

베이지안 선형 회귀에서는 모든 매개변수가 사전분포를 갖습니다:

$$y_i \sim \text{Normal}(\alpha + \mathbf{x}_i^T \boldsymbol{\beta}, \sigma)$$
$$\alpha \sim \text{Normal}(0, 10), \quad \beta_j \sim \text{Normal}(0, 5), \quad \sigma \sim \text{HalfNormal}(5)$$

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n = 100
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
true_alpha = 3.0
true_beta = np.array([2.0, -1.5])
true_sigma = 1.0
y = true_alpha + true_beta[0] * x1 + true_beta[1] * x2 + np.random.normal(0, true_sigma, n)

X = np.column_stack([x1, x2])
```

### 1.2 PyMC 구현(PyMC Implementation)

```python
with pm.Model() as linear_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=5, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Linear predictor
    mu = alpha + pm.math.dot(X, beta)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sample
    trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)

# Results
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print(summary)
print(f"\nTrue values: α={true_alpha}, β={true_beta}, σ={true_sigma}")
```

### 1.3 불확실성 밴드를 포함한 사후 예측(Posterior Predictive with Uncertainty Bands)

```python
# Prediction for new data
x_new = np.linspace(-3, 3, 100)
X_new = np.column_stack([x_new, np.zeros_like(x_new)])  # fix x2=0

alpha_samples = trace.posterior["alpha"].values.flatten()
beta_samples = trace.posterior["beta"].values.reshape(-1, 2)
sigma_samples = trace.posterior["sigma"].values.flatten()

# Posterior predictive
mu_pred = alpha_samples[:, None] + beta_samples[:, 0:1] * x_new[None, :]
y_pred = mu_pred + np.random.normal(0, 1, mu_pred.shape) * sigma_samples[:, None]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x1, y, alpha=0.5, color='gray', label='Data')
ax.plot(x_new, mu_pred.mean(axis=0), 'r-', linewidth=2, label='Posterior mean')
ax.fill_between(x_new,
                np.percentile(mu_pred, 2.5, axis=0),
                np.percentile(mu_pred, 97.5, axis=0),
                alpha=0.3, color='red', label='95% CI (mean)')
ax.fill_between(x_new,
                np.percentile(y_pred, 2.5, axis=0),
                np.percentile(y_pred, 97.5, axis=0),
                alpha=0.15, color='blue', label='95% PI (prediction)')
ax.set_xlabel("x₁")
ax.set_ylabel("y")
ax.set_title("Bayesian Linear Regression with Uncertainty")
ax.legend()
plt.tight_layout()
plt.savefig("bayesian_regression.png", dpi=100)
plt.show()
```

---

## 2. 일반화 선형 모델(Generalized Linear Models, GLMs)

### 2.1 로지스틱 회귀(Logistic Regression)

```python
# Binary classification data
np.random.seed(42)
n = 200
x_log = np.random.normal(0, 1, n)
p_true = 1 / (1 + np.exp(-(1.5 + 2.0 * x_log)))
y_binary = np.random.binomial(1, p_true)

with pm.Model() as logistic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta = pm.Normal("beta", mu=0, sigma=5)

    # Logit link function
    p = pm.math.sigmoid(alpha + beta * x_log)

    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_binary)

    trace_logistic = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_logistic, var_names=["alpha", "beta"])
print(summary)
print(f"\nTrue values: α=1.5, β=2.0")

# Plot decision boundary
x_grid = np.linspace(-3, 3, 200)
alpha_s = trace_logistic.posterior["alpha"].values.flatten()
beta_s = trace_logistic.posterior["beta"].values.flatten()
p_pred = 1 / (1 + np.exp(-(alpha_s[:, None] + beta_s[:, None] * x_grid[None, :])))

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_log, y_binary, alpha=0.3, c=y_binary, cmap='coolwarm')
ax.plot(x_grid, p_pred.mean(axis=0), 'k-', linewidth=2)
ax.fill_between(x_grid,
                np.percentile(p_pred, 2.5, axis=0),
                np.percentile(p_pred, 97.5, axis=0),
                alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("P(y=1)")
ax.set_title("Bayesian Logistic Regression")
plt.tight_layout()
plt.savefig("logistic_regression.png", dpi=100)
plt.show()
```

### 2.2 포아송 회귀(Poisson Regression)

```python
# Count data: number of daily bike rentals
np.random.seed(42)
n = 365
temperature = np.random.normal(15, 8, n)
weekend = np.random.binomial(1, 2/7, n)
log_rate = 3.5 + 0.05 * temperature + 0.3 * weekend
counts = np.random.poisson(np.exp(log_rate))

with pm.Model() as poisson_reg:
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta_temp = pm.Normal("beta_temp", mu=0, sigma=1)
    beta_weekend = pm.Normal("beta_weekend", mu=0, sigma=1)

    log_mu = alpha + beta_temp * temperature + beta_weekend * weekend
    y = pm.Poisson("y", mu=pm.math.exp(log_mu), observed=counts)

    trace_poisson = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_poisson, var_names=["alpha", "beta_temp", "beta_weekend"])
print(summary)
print(f"\nTrue values: α=3.5, β_temp=0.05, β_weekend=0.3")
```

### 2.3 음이항 회귀(Negative Binomial Regression) - 과분산 카운트 데이터

```python
# When Poisson is too restrictive (variance > mean)
with pm.Model() as negbin_reg:
    alpha = pm.Normal("alpha", mu=0, sigma=5)
    beta_temp = pm.Normal("beta_temp", mu=0, sigma=1)
    beta_weekend = pm.Normal("beta_weekend", mu=0, sigma=1)
    phi = pm.HalfNormal("phi", sigma=5)  # overdispersion

    mu = pm.math.exp(alpha + beta_temp * temperature + beta_weekend * weekend)
    y = pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=counts)

    trace_negbin = pm.sample(3000, tune=1000, chains=4, random_seed=42)
```

---

## 3. 로버스트 회귀(Robust Regression)

표준 선형 회귀는 가우시안 오차를 가정하므로 이상치에 민감합니다. 베이지안 로버스트 회귀는 중두꺼운 꼬리 가능도 분포를 사용합니다.

### 3.1 Student-t 가능도(Student-t Likelihood)

```python
# Data with outliers
np.random.seed(42)
n = 100
x_rob = np.random.uniform(0, 10, n)
y_rob = 2.0 + 1.5 * x_rob + np.random.normal(0, 1.0, n)

# Add outliers
outlier_idx = np.random.choice(n, 5, replace=False)
y_rob[outlier_idx] += np.random.normal(0, 10, 5)

# Normal model (sensitive to outliers)
with pm.Model() as normal_reg:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu = alpha + beta * x_rob
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_rob)
    trace_normal = pm.sample(3000, tune=1000, random_seed=42)

# Robust model (Student-t likelihood)
with pm.Model() as robust_reg:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=5)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)  # degrees of freedom
    mu = alpha + beta * x_rob
    y = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_rob)
    trace_robust = pm.sample(3000, tune=1000, random_seed=42)

# Compare
print("Normal regression:")
print(az.summary(trace_normal, var_names=["alpha", "beta"]))
print("\nRobust regression:")
print(az.summary(trace_robust, var_names=["alpha", "beta", "nu"]))
print(f"\nTrue values: α=2.0, β=1.5")
```

---

## 4. 변수 선택과 정규화(Variable Selection and Regularization)

### 4.1 베이지안 라쏘(Bayesian Lasso) - 라플라스 사전분포

```python
# Many predictors, some irrelevant
np.random.seed(42)
n, p = 100, 20
X_lasso = np.random.randn(n, p)
true_coefs = np.zeros(p)
true_coefs[:5] = [3, -2, 1.5, -1, 0.5]  # only 5 are nonzero
y_lasso = X_lasso @ true_coefs + np.random.normal(0, 1, n)

with pm.Model() as bayesian_lasso:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    lam = pm.HalfNormal("lambda", sigma=2)
    beta = pm.Laplace("beta", mu=0, b=1/lam, shape=p)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + pm.math.dot(X_lasso, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_lasso)

    trace_lasso = pm.sample(3000, tune=1000, chains=4, random_seed=42)

# Compare estimated vs true coefficients
beta_est = trace_lasso.posterior["beta"].values.mean(axis=(0, 1))
print("Variable selection (Bayesian Lasso):")
for j in range(p):
    marker = "***" if abs(true_coefs[j]) > 0 else ""
    print(f"  β_{j:2d}: true={true_coefs[j]:6.2f}, est={beta_est[j]:6.3f} {marker}")
```

### 4.2 말굽 사전분포(Horseshoe Prior) - 희소성 유도

```python
with pm.Model() as horseshoe_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Horseshoe prior
    tau = pm.HalfCauchy("tau", beta=1)  # global shrinkage
    lam = pm.HalfCauchy("lam", beta=1, shape=p)  # local shrinkage
    beta = pm.Normal("beta", mu=0, sigma=tau * lam, shape=p)

    mu = alpha + pm.math.dot(X_lasso, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_lasso)

    trace_horseshoe = pm.sample(3000, tune=2000, chains=4,
                                target_accept=0.95, random_seed=42)
```

---

## 5. 다항식 및 비선형 회귀(Polynomial and Nonlinear Regression)

```python
# Nonlinear data
np.random.seed(42)
x_poly = np.random.uniform(-3, 3, 80)
y_poly = 0.5 * x_poly**3 - 2 * x_poly**2 + x_poly + np.random.normal(0, 3, 80)

# Bayesian polynomial regression with model comparison
models_traces = {}
for degree in [1, 2, 3, 4]:
    X_poly = np.column_stack([x_poly**d for d in range(1, degree + 1)])

    with pm.Model() as poly_model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=5, shape=degree)
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + pm.math.dot(X_poly, beta)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_poly)

        trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)
        pm.compute_log_likelihood(trace)

    models_traces[f"degree_{degree}"] = trace

# Compare models
comparison = az.compare(models_traces, ic="loo")
print(comparison)
```

---

## 6. 상호작용 효과와 중심화(Interaction Effects and Centering)

```python
# Model with interaction
np.random.seed(42)
n = 200
x1_int = np.random.normal(0, 1, n)
x2_int = np.random.binomial(1, 0.5, n).astype(float)
y_int = 3 + 2 * x1_int + 1 * x2_int + 1.5 * x1_int * x2_int + np.random.normal(0, 1, n)

with pm.Model() as interaction_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=5)
    beta2 = pm.Normal("beta2", mu=0, sigma=5)
    beta_interact = pm.Normal("beta_interact", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta1 * x1_int + beta2 * x2_int + beta_interact * x1_int * x2_int
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_int)

    trace_interact = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_interact, var_names=["alpha", "beta1", "beta2", "beta_interact"])
print(summary)
print(f"\nTrue: α=3, β1=2, β2=1, β_interact=1.5")
```

---

## 7. 베이지안 R-제곱(Bayesian R-squared)

```python
def bayesian_r_squared(trace, y_obs, X, alpha_name="alpha", beta_name="beta"):
    """Compute posterior distribution of R-squared."""
    alpha = trace.posterior[alpha_name].values.flatten()
    beta = trace.posterior[beta_name].values.reshape(-1, X.shape[1])

    y_pred = alpha[:, None] + beta @ X.T
    var_pred = y_pred.var(axis=1)
    var_resid = np.array([(y_obs - y_pred[i])**2 for i in range(len(alpha))]).mean(axis=1)
    r_squared = var_pred / (var_pred + var_resid)

    print(f"Bayesian R²: {r_squared.mean():.3f} [{np.percentile(r_squared, 2.5):.3f}, {np.percentile(r_squared, 97.5):.3f}]")
    return r_squared

r2 = bayesian_r_squared(trace, y, X)
```

---

## 요약(Summary)

| 모델 | 연결 함수 | 가능도 | 사용 사례 |
|------|----------|--------|----------|
| 선형(Linear) | 항등(Identity) | 정규(Normal) | 연속 결과 |
| 로지스틱(Logistic) | 로짓(Logit) | 베르누이(Bernoulli) | 이항 결과 |
| 포아송(Poisson) | 로그(Log) | 포아송(Poisson) | 카운트 데이터 (평균 ≈ 분산) |
| 음이항(Negative Binomial) | 로그(Log) | 음이항(NegBinomial) | 과분산 카운트 |
| 로버스트(Robust) | 항등(Identity) | Student-t | 이상치가 있는 데이터 |

| 사전분포 | 효과 | 적합한 경우 |
|---------|------|-----------|
| 정규(Normal) | 릿지형 축소 | 모든 예측변수가 중요할 때 |
| 라플라스(Laplace) | 라쏘형 희소성 | 중간 수준의 희소성 |
| 말굽(Horseshoe) | 강한 희소성 | 무관한 예측변수가 많을 때 |

---

## 참고문헌(References)

1. Gelman, A., et al. (2020). "Regression and Other Stories." Cambridge.
2. McElreath, R. (2020). *Statistical Rethinking*, Ch. 4-5, 9.
3. Piironen, J. & Vehtari, A. (2017). "Sparsity information and regularization in the horseshoe." *Electronic Journal of Statistics*.
4. Gelman, A., et al. (2019). "R-squared for Bayesian regression models." *The American Statistician*.

---

[이전: 계층 모델](./05_Hierarchical_Models.md) | [다음: Stan과 CmdStanPy →](./07_Stan_and_CmdStanPy.md)

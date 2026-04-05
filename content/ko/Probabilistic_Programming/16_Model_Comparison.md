# 16. 모델 비교(Model Comparison)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 16번째

[이전: 인과 추론](./15_Causal_Inference.md) | [다음: 불확실성 정량화](./17_Uncertainty_Quantification.md)

---

> **프레임워크 참고**: 이 레슨에서는 베이지안 모델 비교(model comparison)를 위해 PyMC와 ArviZ를 사용합니다.
>
> 설치: `pip install pymc arviz numpy scipy matplotlib`

## 학습 목표(Learning Objectives)

- 모델 비교를 위한 정보 기준(WAIC, LOO-CV) 이해
- 베이즈 인자 계산 및 해석
- 모델 검증을 위한 사후 예측 검사 수행
- ArviZ의 모델 비교 도구 사용
- 베이지안 모델 선택의 일반적인 함정 회피

---

## 1. 모델 비교가 중요한 이유(Why Model Comparison Matters)

베이지안 모델링에서는 종종 여러 후보 모델을 가지게 됩니다. 과적합을 피하면서 데이터를 가장 잘 설명하는 모델을 평가하기 위한 원칙적인 방법이 필요합니다.

### 1.1 베이지안 모델 선택 프레임워크(The Bayesian Model Selection Framework)

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from a quadratic relationship
np.random.seed(42)
n = 50
x = np.random.uniform(-3, 3, n)
y = 1.0 + 0.5 * x + 0.3 * x**2 + np.random.normal(0, 0.5, n)

# Candidate models with increasing complexity
models = {}
traces = {}

for degree in [1, 2, 3, 5]:
    X = np.column_stack([x**d for d in range(1, degree + 1)])
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=2, shape=degree)
        sigma = pm.HalfNormal("sigma", sigma=3)
        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)
        pm.compute_log_likelihood(trace)

    models[f"poly_{degree}"] = model
    traces[f"poly_{degree}"] = trace
```

---

## 2. 정보 기준(Information Criteria)

### 2.1 WAIC (광범위 적용 정보 기준, Widely Applicable Information Criterion)

WAIC는 로그 점별 예측 밀도(lppd)와 유효 매개변수 수에 대한 벌칙을 사용하여 표본 외 예측 정확도를 추정합니다.

$$\text{WAIC} = -2(\text{lppd} - p_\text{WAIC})$$

```python
# Compute WAIC for all models
for name, trace in traces.items():
    waic = az.waic(trace)
    print(f"{name}: WAIC = {waic.waic:.2f}, p_waic = {waic.p_waic:.2f}")
```

### 2.2 LOO-CV (Leave-One-Out 교차 검증)

파레토 평활 중요도 샘플링(PSIS-LOO)을 통해 근사한 LOO-CV입니다.

```python
# LOO-CV comparison
comparison = az.compare(traces, ic="loo")
print(comparison)

# Visualize
az.plot_compare(comparison)
plt.title("Model Comparison: LOO-CV")
plt.savefig("model_comparison_loo.png", dpi=100)
plt.show()
```

### 2.3 결과 해석(Interpreting Results)

```python
# Key columns in the comparison table:
# - rank: model ranking (0 = best)
# - elpd_loo: expected log pointwise predictive density
# - p_loo: effective number of parameters
# - d_loo: difference from best model
# - se: standard error of the difference
# - dse: standard error of the difference relative to best
# - warning: True if Pareto k > 0.7 (unreliable estimate)

# Rule of thumb:
# - If d_loo / dse > 2: significant difference
# - If warning=True: use K-fold CV instead
```

---

## 3. 베이즈 인자(Bayes Factors)

베이즈 인자는 두 모델에 대한 증거를 직접 비교합니다.

$$BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int P(D|\theta_1, M_1) P(\theta_1 | M_1) d\theta_1}{\int P(D|\theta_2, M_2) P(\theta_2 | M_2) d\theta_2}$$

### 3.1 해석 척도(Interpretation Scale)

| 베이즈 인자 | 증거 강도 |
|-----------|---------|
| 1-3 | 거의 언급할 가치 없음 |
| 3-10 | 상당함 |
| 10-30 | 강함 |
| 30-100 | 매우 강함 |
| > 100 | 결정적 |

### 3.2 베이즈 인자 계산 (Savage-Dickey 비율)(Computing Bayes Factors)

```python
def savage_dickey_bf(trace, param_name, null_value, prior_density_at_null):
    """
    Compute Bayes factor for point null hypothesis using Savage-Dickey ratio.
    BF_01 = p(theta=null | data) / p(theta=null | prior)
    """
    from scipy.stats import gaussian_kde

    posterior_samples = trace.posterior[param_name].values.flatten()
    kde = gaussian_kde(posterior_samples)
    posterior_density_at_null = kde(null_value)[0]

    bf_01 = posterior_density_at_null / prior_density_at_null
    bf_10 = 1 / bf_01

    print(f"BF_01 (in favor of null): {bf_01:.4f}")
    print(f"BF_10 (against null):     {bf_10:.4f}")
    return bf_01, bf_10

# Test: is the quadratic coefficient zero?
prior_density = stats.norm.pdf(0, 0, 2)  # Normal(0, 2) prior at 0
bf_01, bf_10 = savage_dickey_bf(traces["poly_2"], "beta", 0, prior_density)
```

### 3.3 브릿지 샘플링(Bridge Sampling)

```python
def bridge_sampling_estimate(trace, model, n_bridge=10000):
    """Estimate marginal likelihood via bridge sampling (simplified)."""
    # This is a simplified version; use bridgesampling package for production
    log_lik = trace.log_likelihood
    # Approximate: harmonic mean estimator (known to be unstable)
    ll_values = sum(log_lik[v].values.flatten() for v in log_lik.data_vars)
    log_ml = -np.log(np.mean(np.exp(-ll_values)))
    return log_ml
```

---

## 4. 사후 예측 검사(Posterior Predictive Checks, PPC)

### 4.1 시각적 PPC(Visual PPC)

```python
# Generate posterior predictive samples for the quadratic model
with models["poly_2"]:
    ppc = pm.sample_posterior_predictive(traces["poly_2"], random_seed=42)

# Overlay plot
az.plot_ppc(az.from_pymc(trace=traces["poly_2"],
                          posterior_predictive=ppc,
                          model=models["poly_2"]))
plt.title("Posterior Predictive Check: Quadratic Model")
plt.savefig("ppc_quadratic.png", dpi=100)
plt.show()
```

### 4.2 검정 통계량(Test Statistics)

```python
def ppc_test_statistics(y_obs, y_rep, statistics=None):
    """Compute posterior predictive p-values for test statistics."""
    if statistics is None:
        statistics = {
            "mean": np.mean,
            "std": np.std,
            "skewness": lambda x: stats.skew(x),
            "min": np.min,
            "max": np.max,
        }

    print("Posterior Predictive p-values:")
    print("-" * 50)
    for name, stat_fn in statistics.items():
        observed_stat = stat_fn(y_obs)
        rep_stats = np.array([stat_fn(y_rep[i]) for i in range(len(y_rep))])
        p_value = np.mean(rep_stats >= observed_stat)
        print(f"  {name:12s}: observed={observed_stat:.3f}, p-value={p_value:.3f}")
        # p close to 0 or 1 indicates model misfit for this statistic

y_rep = ppc.posterior_predictive["y"].values.reshape(-1, n)
ppc_test_statistics(y, y_rep[:1000])
```

### 4.3 교정 검사(Calibration Check)

```python
def pit_calibration(y_obs, y_rep):
    """Probability Integral Transform calibration check."""
    n_obs = len(y_obs)
    pit_values = np.array([
        np.mean(y_rep[:, i] <= y_obs[i])
        for i in range(n_obs)
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(pit_values, bins=20, density=True, alpha=0.7)
    ax1.axhline(1.0, color='r', linestyle='--', label='Ideal (Uniform)')
    ax1.set_title("PIT Histogram (should be uniform)")
    ax1.legend()

    # ECDF vs uniform
    pit_sorted = np.sort(pit_values)
    ax2.plot(pit_sorted, np.linspace(0, 1, len(pit_sorted)), 'b-', label='PIT ECDF')
    ax2.plot([0, 1], [0, 1], 'r--', label='Ideal')
    ax2.set_title("PIT ECDF vs Uniform")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("pit_calibration.png", dpi=100)
    plt.show()
    return pit_values

pit = pit_calibration(y, y_rep[:2000])
```

---

## 5. 스태킹과 모델 평균화(Stacking and Model Averaging)

하나의 모델을 선택하는 대신, 여러 모델을 예측 성능에 따라 가중 결합합니다.

```python
# Bayesian stacking weights (ArviZ)
weights = az.compare(traces, ic="loo")["weight"].values
model_names = list(traces.keys())

print("Stacking weights:")
for name, w in zip(model_names, weights):
    print(f"  {name}: {w:.3f}")

# Model-averaged predictions
def model_averaged_prediction(traces, weights, X_new):
    """Combine predictions from multiple models."""
    predictions = []
    for name, trace in traces.items():
        alpha = trace.posterior["alpha"].values.flatten()
        beta = trace.posterior["beta"].values.reshape(-1, trace.posterior["beta"].shape[-1])
        degree = beta.shape[1]
        X_poly = np.column_stack([X_new**d for d in range(1, degree + 1)])
        pred = alpha[:, None] + beta @ X_poly.T
        predictions.append(pred)

    # Weighted average
    avg_pred = sum(w * pred for w, pred in zip(weights, predictions))
    return avg_pred.mean(axis=0), avg_pred.std(axis=0)
```

---

## 6. 교차 검증(Cross-Validation)

### 6.1 베이지안 환경에서의 K-Fold CV(K-Fold CV in Bayesian Setting)

```python
def bayesian_kfold_cv(X, y, model_fn, K=5, n_samples=2000):
    """K-fold cross-validation for Bayesian models."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    log_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        with model_fn(X_train, y_train) as model:
            trace = pm.sample(n_samples, tune=500, chains=2, random_seed=42)

        # Compute predictive log-likelihood on test set
        alpha = trace.posterior["alpha"].values.flatten()
        beta = trace.posterior["beta"].values.reshape(-1, X.shape[1])
        sigma = trace.posterior["sigma"].values.flatten()

        pred_mu = alpha[:, None] + beta @ X_test.T
        log_lik = stats.norm.logpdf(y_test[None, :], pred_mu, sigma[:, None])
        log_score = np.log(np.exp(log_lik).mean(axis=0)).sum()
        log_scores.append(log_score)

        print(f"Fold {fold}: log score = {log_score:.2f}")

    print(f"\nMean log score: {np.mean(log_scores):.2f} ± {np.std(log_scores):.2f}")
    return log_scores
```

---

## 7. 일반적인 함정(Common Pitfalls)

### 7.1 베이즈 인자의 사전분포 민감도(Prior Sensitivity of Bayes Factors)

```python
# Bayes factors are VERY sensitive to prior choice
# (unlike posterior inference, which is often robust)

# Example: same data, different priors → different Bayes factors
priors = [
    ("Narrow: N(0, 1)", 1.0),
    ("Medium: N(0, 5)", 5.0),
    ("Wide: N(0, 100)", 100.0),
]

for name, prior_sd in priors:
    prior_density = stats.norm.pdf(0, 0, prior_sd)
    bf_01, _ = savage_dickey_bf(traces["poly_2"], "beta", 0, prior_density)
    print(f"  {name}: BF_01 = {bf_01:.4f}")
```

### 7.2 과적합 경고 신호(Overfitting Warning Signs)

```python
# Warning signs in model comparison:
# 1. p_loo or p_waic >> actual number of parameters
# 2. Large Pareto k values (> 0.7)
# 3. Wide standard errors on elpd differences
# 4. Posterior predictive checks fail
```

---

## 요약(Summary)

| 기준 | 장점 | 단점 | 사용 시점 |
|------|------|------|---------|
| WAIC | 빠름, 자동 | LOO보다 덜 신뢰적 | 빠른 비교 |
| PSIS-LOO | 표준, 신뢰적 | 실패 가능(높은 Pareto k) | 기본 선택 |
| 베이즈 인자 | 직접적 증거 비교 | 사전분포에 매우 민감 | 명확한 가설 |
| K-fold CV | PSIS 문제 없음 | 비용 높음 (K번 재적합) | LOO 실패 시 |
| PPC | 직관적, 시각적 | 주관적 | 모델 확인 |
| 스태킹 | 모델 강점 결합 | 더 복잡 | 여러 좋은 모델 |

---

## 참고 문헌(References)

1. Vehtari, A., Gelman, A., & Gabry, J. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*, 27, 1413-1432.
2. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Ed., Ch. 7.
3. Yao, Y., et al. (2018). "Using Stacking to Average Bayesian Predictive Distributions." *Bayesian Analysis*.
4. Gronau, Q., et al. (2017). "A Tutorial on Bridge Sampling." *Journal of Mathematical Psychology*.

---

[이전: 인과 추론](./15_Causal_Inference.md) | [다음: 불확실성 정량화 →](./17_Uncertainty_Quantification.md)

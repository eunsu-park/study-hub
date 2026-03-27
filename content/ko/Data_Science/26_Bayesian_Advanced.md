# 베이지안 고급 방법론(Bayesian Advanced Methods)

[이전: 실전 프로젝트](./25_Practical_Projects.md) | [다음: 인과 추론](./27_Causal_Inference.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. HMC(Hamiltonian Monte Carlo)가 기울기(gradient) 정보를 활용해 메트로폴리스-헤이스팅스(Metropolis-Hastings)보다 효율적으로 샘플링하는 원리를 설명할 수 있다
2. NUTS(No-U-Turn Sampler)가 궤적 길이를 자동으로 조정하는 방식과 현대 확률적 프로그래밍 프레임워크에서 기본 샘플러로 채택된 이유를 설명할 수 있다
3. MCMC와 변분 추론(Variational Inference, ADVI)을 정확도, 속도, 적합한 사용 사례 측면에서 비교할 수 있다
4. PyMC에서 계층 모델(hierarchical model)을 구현하고, 부분 풀링(partial pooling)과 수축(shrinkage) 개념을 설명할 수 있다
5. LOO-CV와 WAIC를 이용한 베이지안 모델 비교를 적용하고, 파레토 k(Pareto k) 진단 결과를 해석할 수 있다
6. R-hat, 유효 표본 크기(ESS, Effective Sample Size), 발산(divergence) 검사, 사후 예측 검사(posterior predictive check)를 활용하여 MCMC 수렴을 평가할 수 있다

---

메트로폴리스-헤이스팅스 같은 기본적인 MCMC 방법은 단순한 모델에서는 충분하지만, 고차원 매개변수 공간에서는 어려움을 겪고 대규모 데이터셋에 잘 확장되지 않습니다. HMC/NUTS(효율적인 샘플링), 변분 추론(Variational Inference)(속도), 계층 모델(hierarchical models)(그룹화된 데이터) 등의 고급 계산 베이지안 방법론은 베이지안 프레임워크의 진정한 잠재력을 열어줍니다. 문제가 복잡한 경우에도 더 풍부한 모델을 구축하고 신뢰할 수 있는 결론을 도출할 수 있게 해줍니다.

---

## 1. 기본 MCMC를 넘어서

### 1.1 메트로폴리스-헤이스팅스(Metropolis-Hastings)의 한계

```python
"""
Metropolis-Hastings (from L16) works but has limitations:

1. Random walk proposal → slow exploration of high-dimensional spaces
2. Requires tuning step size manually
3. High autocorrelation → need many samples for good estimates
4. Scales poorly: O(d²) samples needed for d dimensions

Solution: Use gradient information to guide proposals
→ Hamiltonian Monte Carlo (HMC)
"""
```

### 1.2 해밀턴 몬테카를로(Hamiltonian Monte Carlo, HMC)

```python
"""
HMC Intuition:

Imagine a ball rolling on a surface shaped like the negative log-posterior.
The ball's position = parameter values.
The ball's momentum = auxiliary variable for exploration.

Algorithm:
1. Sample random momentum p ~ N(0, I)
2. Simulate Hamiltonian dynamics for L steps with step size ε:
   - θ_new = θ + ε * p
   - p_new = p - ε * ∇U(θ)    where U(θ) = -log π(θ|data)
3. Accept/reject with Metropolis correction

Key Parameters:
- Step size (ε): Too large → rejected, too small → slow
- Number of steps (L): Too few → random walk, too many → wasted computation

Advantages over Metropolis-Hastings:
- Uses gradient → moves efficiently through parameter space
- Much lower autocorrelation
- Scales better with dimensionality: O(d^{5/4}) vs O(d²)
"""
```

### 1.3 NUTS(No-U-Turn Sampler)

```python
"""
NUTS: Automatically tunes L (number of leapfrog steps).

Problem with HMC: Choosing L is difficult.
- Too few steps → behaves like random walk
- Too many → the trajectory loops back (U-turn), wasting computation

NUTS solution:
- Build a binary tree of leapfrog steps
- Stop when the trajectory starts to turn back (U-turn criterion)
- Select a point from the trajectory with correct stationary distribution

NUTS is the default sampler in PyMC, Stan, and NumPyro.
No need to tune L — only ε (step size), which is auto-tuned during warmup.
"""

import pymc as pm
import numpy as np
import arviz as az

# Generate data
np.random.seed(42)
N = 200
true_mu = [3.0, -1.0]
true_sigma = [1.5, 0.8]
data = np.concatenate([
    np.random.normal(true_mu[0], true_sigma[0], N),
    np.random.normal(true_mu[1], true_sigma[1], N),
])

# PyMC model with NUTS (default sampler)
with pm.Model() as mixture_model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=5, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=3, shape=2)
    weights = pm.Dirichlet("weights", a=np.ones(2))

    # Likelihood (mixture)
    likelihood = pm.Mixture(
        "obs",
        w=weights,
        comp_dists=[
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1]),
        ],
        observed=data,
    )

    # Sample with NUTS (default)
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# Diagnostics
print(az.summary(trace, var_names=["mu", "sigma", "weights"]))
az.plot_trace(trace, var_names=["mu", "sigma"])
```

---

## 2. 변분 추론(Variational Inference, VI)

### 2.1 VI 개념

```python
"""
Variational Inference: Approximate posterior with optimization instead of sampling.

MCMC: Sample from p(θ|data) directly → exact but slow
VI:   Find q(θ) ≈ p(θ|data) by minimizing KL divergence → fast but approximate

  Objective: minimize KL(q(θ) || p(θ|data))
  Equivalent to: maximize ELBO (Evidence Lower Bound)

  ELBO = E_q[log p(data, θ)] - E_q[log q(θ)]
       = E_q[log p(data|θ)] - KL(q(θ) || p(θ))
         ↑ data fit            ↑ stay close to prior

Mean-Field VI:
  Assume q(θ) = ∏ q_i(θ_i)  (fully factorized)
  Each factor is a simple distribution (e.g., Gaussian)
  Optimize the parameters of each q_i

ADVI (Automatic Differentiation VI):
  - Automatically transforms constrained parameters to unconstrained space
  - Uses stochastic gradient descent to optimize ELBO
  - Available in PyMC via pm.fit()

When to use VI vs MCMC:
  - VI: Large datasets, complex models, approximate answer OK, need speed
  - MCMC: Small-medium datasets, need exact posterior, model diagnostics important
"""
```

### 2.2 PyMC에서의 ADVI

```python
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Generate logistic regression data
np.random.seed(42)
N = 5000
d = 10  # features
X = np.random.randn(N, d)
true_beta = np.random.randn(d) * 0.5
p = 1 / (1 + np.exp(-X @ true_beta))
y = np.random.binomial(1, p)

# Model
with pm.Model() as logistic_model:
    beta = pm.Normal("beta", mu=0, sigma=1, shape=d)
    logit_p = pm.math.dot(X, beta)
    obs = pm.Bernoulli("obs", logit_p=logit_p, observed=y)

    # ADVI (fast approximate inference)
    approx = pm.fit(
        n=30000,
        method="advi",
        random_seed=42,
    )

# Plot ELBO convergence
plt.figure(figsize=(10, 4))
plt.plot(approx.hist)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ADVI Convergence")
plt.tight_layout()
plt.show()

# Sample from approximate posterior
vi_trace = approx.sample(2000)

# Compare VI vs true parameters
vi_means = vi_trace.posterior["beta"].mean(dim=["chain", "draw"]).values
print("\nParameter comparison:")
print(f"{'':>6} {'True':>8} {'VI Mean':>8}")
for i in range(d):
    print(f"  β_{i}: {true_beta[i]:>8.3f} {vi_means[i]:>8.3f}")
```

### 2.3 MCMC와 VI 비교

```python
"""
MCMC vs VI Comparison:

| Aspect         | MCMC (NUTS)                | VI (ADVI)                 |
|----------------|----------------------------|---------------------------|
| Output         | Samples from posterior     | Approximate distribution  |
| Accuracy       | Exact (given enough samples) | Approximate             |
| Speed          | Slow (sequential)          | Fast (optimization)       |
| Scalability    | O(N) per sample            | Mini-batch possible       |
| Diagnostics    | R-hat, ESS, trace plots    | ELBO convergence          |
| Multimodal     | Can explore multiple modes | May miss modes            |
| Uncertainty    | Full posterior uncertainty  | May underestimate         |
| Best for       | Small-medium N, accuracy   | Large N, speed, screening |
"""

# Side-by-side comparison
with pm.Model() as comparison_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma,
                    observed=np.random.normal(3, 1.5, 100))

    # MCMC
    mcmc_trace = pm.sample(2000, tune=1000, random_seed=42)

    # VI
    approx = pm.fit(20000, method="advi", random_seed=42)
    vi_trace = approx.sample(2000)

# Compare posteriors
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, var in zip(axes, ["mu", "sigma"]):
    mcmc_vals = mcmc_trace.posterior[var].values.flatten()
    vi_vals = vi_trace.posterior[var].values.flatten()
    ax.hist(mcmc_vals, bins=50, alpha=0.5, density=True, label="MCMC")
    ax.hist(vi_vals, bins=50, alpha=0.5, density=True, label="VI")
    ax.set_title(var)
    ax.legend()
plt.tight_layout()
plt.show()
```

---

## 3. 계층 모델(Hierarchical Models)

### 3.1 부분 풀링(Partial Pooling)

```python
"""
Hierarchical (multilevel) models: share information across groups.

Three approaches:
  1. Complete pooling: Ignore groups, one model for all
     → Ignores group differences
  2. No pooling: Separate model per group
     → Noisy estimates for small groups
  3. Partial pooling (hierarchical): Groups share a common prior
     → Borrows strength across groups (shrinkage)

Example: Estimating batting averages in baseball
  - Player with 4/10 hits (small sample) → shrunk toward league average
  - Player with 100/300 hits (large sample) → close to raw average
"""

import pymc as pm
import numpy as np
import arviz as az

# Simulated data: test scores across 8 schools
np.random.seed(42)
n_schools = 8
true_mu = 50
true_tau = 10
true_theta = np.random.normal(true_mu, true_tau, n_schools)  # school-level means
n_per_school = np.random.randint(10, 100, n_schools)

data = []
school_idx = []
for j in range(n_schools):
    scores = np.random.normal(true_theta[j], 15, n_per_school[j])
    data.extend(scores)
    school_idx.extend([j] * n_per_school[j])

data = np.array(data)
school_idx = np.array(school_idx)

# Hierarchical model
with pm.Model() as hierarchical_model:
    # Hyperpriors (population-level)
    mu = pm.Normal("mu", mu=50, sigma=20)           # population mean
    tau = pm.HalfNormal("tau", sigma=15)              # between-school SD

    # School-level parameters (partial pooling)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_schools)

    # Within-school variation
    sigma = pm.HalfNormal("sigma", sigma=20)

    # Likelihood
    obs = pm.Normal("obs", mu=theta[school_idx], sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(2000, tune=1000, random_seed=42)

# Compare raw means vs hierarchical estimates
raw_means = [data[school_idx == j].mean() for j in range(n_schools)]
hier_means = trace.posterior["theta"].mean(dim=["chain", "draw"]).values

print(f"\n{'School':>8} {'N':>5} {'Raw Mean':>10} {'Hier Mean':>10} {'True':>8}")
print("-" * 50)
for j in range(n_schools):
    print(f"  {j:>5} {n_per_school[j]:>5} {raw_means[j]:>10.2f} "
          f"{hier_means[j]:>10.2f} {true_theta[j]:>8.2f}")

print(f"\n  Population mean: {trace.posterior['mu'].mean().values:.2f} "
      f"(true: {true_mu})")
print(f"  Between-school SD: {trace.posterior['tau'].mean().values:.2f} "
      f"(true: {true_tau})")
```

---

## 4. 모델 비교(Model Comparison)

### 4.1 WAIC와 LOO-CV

```python
"""
Bayesian Model Comparison:

1. WAIC (Widely Applicable Information Criterion):
   WAIC = -2 * (lppd - p_waic)
   - lppd: log pointwise predictive density
   - p_waic: effective number of parameters
   - Lower is better

2. LOO-CV (Leave-One-Out Cross-Validation via PSIS):
   - Pareto-Smoothed Importance Sampling (PSIS)
   - Approximates LOO-CV without refitting
   - elpd_loo: expected log pointwise predictive density
   - Higher elpd_loo is better
   - k diagnostic: k > 0.7 indicates unreliable estimate

Both are available in ArviZ and preferred over DIC/BIC for Bayesian models.
"""

import pymc as pm
import numpy as np
import arviz as az

# Generate data with a quadratic relationship
np.random.seed(42)
N = 100
x = np.random.uniform(-3, 3, N)
y = 2 + 0.5 * x - 0.3 * x**2 + np.random.normal(0, 1, N)

# Model 1: Linear
with pm.Model() as linear_model:
    a = pm.Normal("a", mu=0, sigma=5)
    b = pm.Normal("b", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)
    mu = a + b * x
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    trace_linear = pm.sample(2000, tune=1000, random_seed=42,
                              idata_kwargs={"log_likelihood": True})

# Model 2: Quadratic
with pm.Model() as quad_model:
    a = pm.Normal("a", mu=0, sigma=5)
    b = pm.Normal("b", mu=0, sigma=5)
    c = pm.Normal("c", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)
    mu = a + b * x + c * x**2
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    trace_quad = pm.sample(2000, tune=1000, random_seed=42,
                            idata_kwargs={"log_likelihood": True})

# Compare with LOO
comparison = az.compare(
    {"linear": trace_linear, "quadratic": trace_quad},
    ic="loo",
)
print("Model Comparison (LOO-CV):")
print(comparison)

# Plot comparison
az.plot_compare(comparison, insample_dev=False)
```

---

## 5. 실용적 고려사항

### 5.1 진단 체크리스트(Diagnostics Checklist)

```python
"""
Bayesian Diagnostics Checklist:

1. Convergence:
   - R-hat < 1.01 for all parameters
   - Trace plots: chains should mix well (fuzzy caterpillars)
   - No divergences (NUTS specific)

2. Effective Sample Size (ESS):
   - ESS > 400 for reliable posterior estimates
   - ESS_bulk: overall posterior estimation
   - ESS_tail: tail probability estimation

3. NUTS Diagnostics:
   - Divergences = 0 (or very few)
   - Tree depth not hitting max (default 10)
   - Energy plot: marginal energy ≈ energy transition

4. Posterior Predictive Check:
   - Simulate data from posterior
   - Compare with observed data
   - If mismatched → model misspecification

5. Prior Predictive Check:
   - Sample from prior → simulate data
   - Check if prior generates plausible data
   - If absurd → revise priors
"""

# Comprehensive diagnostics
import pymc as pm
import numpy as np
import arviz as az

np.random.seed(42)
y_obs = np.random.normal(5, 2, 50)

with pm.Model() as diag_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_obs)

    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(500, random_seed=42)

    # Sample
    trace = pm.sample(2000, tune=1000, random_seed=42)

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

# R-hat and ESS
summary = az.summary(trace)
print("Parameter Summary:")
print(summary)

# Check for issues
rhat_ok = (summary["r_hat"] < 1.01).all()
ess_ok = (summary["ess_bulk"] > 400).all()
print(f"\nR-hat OK: {rhat_ok}")
print(f"ESS OK: {ess_ok}")

# Divergences
n_divergences = trace.sample_stats["diverging"].sum().values
print(f"Divergences: {n_divergences}")
```

---

## 6. 연습 문제

### 연습 1: 계층 회귀(Hierarchical Regression)

```python
"""
Build a hierarchical linear regression:
1. Data: Student test scores across 20 schools
   - Each school has 10-50 students
   - Covariates: study hours, prior GPA
2. Model: School-specific slopes and intercepts
   - Partial pooling: share strength across schools
3. Compare: no pooling vs partial pooling vs complete pooling
4. Visualize shrinkage: how do small-school estimates change?
5. Run diagnostics: R-hat, ESS, posterior predictive check
"""
```

### 연습 2: MCMC vs VI

```python
"""
Compare MCMC and VI on the same model:
1. Generate data from a mixture of 3 Gaussians (N=1000)
2. Fit with NUTS (pm.sample) and ADVI (pm.fit)
3. Compare: posterior means, uncertainties, computation time
4. When does VI's approximation break down?
5. Try FullRank ADVI (method='fullrank_advi') and compare
"""
```

---

## 7. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **HMC** | 효율적인 샘플링을 위해 기울기(gradient) 활용; 메트로폴리스보다 확장성 우수 |
| **NUTS** | HMC 궤적 길이 자동 조정; PyMC/Stan의 기본 샘플러 |
| **ADVI** | 최적화 기반의 빠른 근사 추론; 대규모 데이터에 유용 |
| **계층 모델** | 부분 풀링(partial pooling): 그룹 간 정보 공유 |
| **LOO-CV** | 파레토 평활 중요도 샘플링(Pareto-smoothed importance sampling)을 통한 베이지안 모델 비교 |
| **사전 예측 검사** | 건전성 확인: 사전 분포가 그럴듯한 데이터를 생성하는가? |

### 모범 사례

1. **NUTS로 시작** — 중간 차원 문제에서 황금 표준
2. **VI는 사전 탐색용** — 모델 공간을 빠르게 탐색한 뒤 MCMC로 검증
3. **진단 확인** — R-hat < 1.01, ESS > 400, 발산(divergence) 0
4. **사전 예측 검사** — 피팅 전 항상 사전 분포 검증
5. **그룹화 데이터엔 계층 모델** — 부분 풀링은 거의 항상 풀링 없음보다 우수
6. **LOO로 비교** — WAIC보다 LOO-CV 선호; 파레토 k 값 확인

### 다음 단계

- **L27**: 인과 추론(Causal Inference) — 상관관계에서 인과관계로
- **L15-L16** (베이지안 기초/추론)으로 돌아가 기반 개념 복습

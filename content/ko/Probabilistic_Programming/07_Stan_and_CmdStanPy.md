# 07. Stan과 CmdStanPy(Stan and CmdStanPy)

[이전: 베이지안 회귀](./06_Bayesian_Regression.md) | [다음: 변분 추론](./08_Variational_Inference.md)

---

> **프레임워크 참고**: 이 레슨은 권장 Python 인터페이스인 CmdStanPy를 통해 Stan을 사용합니다.
>
> 설치: `pip install cmdstanpy && install_cmdstan`

## 학습 목표

- Stan 확률적 프로그래밍 언어 이해
- data, parameters, model, generated quantities 블록으로 Stan 프로그램 작성
- CmdStanPy를 통한 Stan과 Python 인터페이스
- Stan의 HMC/NUTS 구현 세부사항 이해
- 일반적인 Stan 모델링 문제 디버깅

---

## 1. Stan을 사용하는 이유(Why Stan?)

Stan은 베이지안 추론의 표준입니다. 자동 미분과 컴파일 시점 최적화를 갖춘 최첨단 HMC/NUTS를 구현합니다.

### 1.1 Stan vs PyMC

| 특징 | Stan | PyMC |
|------|------|------|
| 언어 | 커스텀 DSL (C++로 컴파일) | Python API |
| 속도 | 매우 빠름 (컴파일됨) | 빠름 (PyTensor) |
| 샘플러 | NUTS (참조 구현) | NUTS (Stan 기반) |
| 이산 매개변수 | 네이티브 미지원 (주변화 필요) | Metropolis 단계 |
| 유연성 | 블록 구조 | Python의 자유도 |
| 진단 | 내장 + ShinyStan | ArviZ |
| 학습 곡선 | 가파름 | 완만함 |

### 1.2 CmdStanPy 설정(CmdStanPy Setup)

```python
import cmdstanpy
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Verify installation
print(f"CmdStan path: {cmdstanpy.cmdstan_path()}")
print(f"CmdStanPy version: {cmdstanpy.__version__}")
```

---

## 2. Stan 프로그램 구조(Stan Program Structure)

Stan 프로그램은 최대 7개의 블록으로 구성되며 순서대로 실행됩니다.

### 2.1 블록 구조(Block Anatomy)

```stan
// File: normal_model.stan

functions {
  // User-defined functions (optional)
}

data {
  // Declare observed data passed from Python
  int<lower=0> N;
  vector[N] y;
}

transformed data {
  // Compute derived data quantities (runs once)
  real y_mean = mean(y);
}

parameters {
  // Declare parameters to be estimated
  real mu;
  real<lower=0> sigma;
}

transformed parameters {
  // Compute derived parameters (runs every iteration)
}

model {
  // Priors + likelihood
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  y ~ normal(mu, sigma);    // vectorized likelihood
}

generated quantities {
  // Posterior predictions, log-likelihood, etc.
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
}
```

### 2.2 Python에서 실행(Running from Python)

```python
import os
import tempfile

# Write Stan model to file
stan_code = """
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
}
"""

# Save model file
model_path = os.path.join(tempfile.gettempdir(), "normal_model.stan")
with open(model_path, "w") as f:
    f.write(stan_code)

# Compile model
model = cmdstanpy.CmdStanModel(stan_file=model_path)

# Prepare data
np.random.seed(42)
data = np.random.normal(5.0, 2.0, 100)
stan_data = {"N": len(data), "y": data.tolist()}

# Sample
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_sampling=2000,
    iter_warmup=1000,
    seed=42,
    adapt_delta=0.8,
)

# Summary
print(fit.summary())

# Convert to ArviZ for visualization
idata = az.from_cmdstanpy(fit)
az.plot_trace(idata, var_names=["mu", "sigma"])
plt.tight_layout()
plt.savefig("stan_trace.png", dpi=100)
plt.show()
```

---

## 3. Stan 데이터 타입과 제약조건(Stan Data Types and Constraints)

### 3.1 타입 시스템(Type System)

```stan
data {
  // Scalars
  int N;                        // integer
  real x;                       // real number
  int<lower=0> N_pos;           // non-negative integer
  real<lower=0, upper=1> prob;  // bounded real

  // Vectors and matrices
  vector[N] y;                  // column vector
  row_vector[N] rv;             // row vector
  matrix[N, K] X;               // N×K matrix
  simplex[K] theta;             // sums to 1
  ordered[K] cutpoints;         // ordered vector
  positive_ordered[K] pos_ord;  // positive ordered
  unit_vector[K] uv;            // unit norm
  corr_matrix[K] Omega;         // correlation matrix
  cov_matrix[K] Sigma;          // covariance matrix
  cholesky_factor_corr[K] L;    // Cholesky factor
}
```

### 3.2 제약조건 변환(Constraint Transformations)

Stan은 제약된 매개변수에 대한 변환을 자동으로 처리합니다. 예를 들어, `real<lower=0> sigma`는 내부적으로 로그 스케일에서 매개변수화됩니다.

```python
# This means:
# - NUTS operates in unconstrained space
# - Stan adds the Jacobian adjustment automatically
# - You just declare the constraint and write natural code
```

---

## 4. Stan의 HMC/NUTS(HMC/NUTS in Stan)

### 4.1 샘플러 구성(Sampler Configuration)

```python
# Control HMC/NUTS parameters
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_sampling=2000,      # draws per chain
    iter_warmup=1000,        # adaptation period
    seed=42,
    adapt_delta=0.8,         # target acceptance rate (default 0.8)
    max_treedepth=10,        # NUTS max tree depth (default 10)
    # step_size=0.1,         # initial step size (usually auto-tuned)
)
```

### 4.2 진단 경고(Diagnostic Warnings)

```python
# Check for common issues
print(f"Divergent transitions: {fit.diagnose()}")

# Key diagnostics:
# 1. Divergent transitions → increase adapt_delta to 0.95 or 0.99
# 2. Max treedepth exceeded → increase max_treedepth to 12 or 15
# 3. Low E-BFMI → reparameterize the model
# 4. R-hat > 1.01 → run longer chains

# Access diagnostics programmatically
diagnostics = fit.method_variables()
n_divergent = sum(diagnostics["divergent__"].sum())
print(f"Total divergences: {n_divergent}")
```

### 4.3 질량 행렬 적응(Mass Matrix Adaptation)

워밍업 단계에서 Stan은 사후분포의 공분산에 맞추어 "질량 행렬"(메트릭)을 적응시킵니다. 이를 통해 효율적인 탐색이 가능합니다.

```python
# Three warmup phases:
# 1. Initial step size adaptation (first 75 iterations)
# 2. Mass matrix estimation (next ~850 iterations)
# 3. Final step size adaptation (last 75 iterations)
# Total default warmup: 1000 iterations
```

---

## 5. Stan 모델링 예제(Stan Modeling Examples)

### 5.1 계층 모델 - Eight Schools(Hierarchical Model)

```stan
// eight_schools.stan
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta;
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
```

```python
eight_schools_code = """
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] z;
}
transformed parameters {
  vector[J] theta = mu + tau * z;  // non-centered
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  z ~ std_normal();
  y ~ normal(theta, to_vector(sigma));
}
"""

model_path = os.path.join(tempfile.gettempdir(), "eight_schools.stan")
with open(model_path, "w") as f:
    f.write(eight_schools_code)

model_8s = cmdstanpy.CmdStanModel(stan_file=model_path)

data_8s = {
    "J": 8,
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
}

fit_8s = model_8s.sample(data=data_8s, chains=4, seed=42, adapt_delta=0.95)
print(fit_8s.summary())
```

### 5.2 로지스틱 회귀(Logistic Regression)

```stan
// logistic.stan
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2.5);
  y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = alpha + X[n] * beta;
    y_rep[n] = bernoulli_logit_rng(eta);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta);
  }
}
```

### 5.3 가우시안 프로세스(Gaussian Process)

```stan
// gp_regression.stan
data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
}
parameters {
  real<lower=0> alpha;      // signal variance
  real<lower=0> rho;        // length scale
  real<lower=0> sigma;      // noise
}
model {
  matrix[N, N] K;
  for (i in 1:N)
    for (j in i:N) {
      K[i, j] = square(alpha) * exp(-square(x[i] - x[j]) / (2 * square(rho)));
      K[j, i] = K[i, j];
    }
  for (n in 1:N)
    K[n, n] += square(sigma);

  alpha ~ normal(0, 2);
  rho ~ inv_gamma(5, 5);
  sigma ~ normal(0, 1);

  y ~ multi_normal(mu, K);
}
```

---

## 6. Stan 함수와 사용자 정의 코드(Stan Functions and User-Defined Code)

```stan
functions {
  // Custom log-likelihood for zero-inflated Poisson
  real zip_lpmf(int y, real lambda, real p_zero) {
    if (y == 0)
      return log_sum_exp(
        log(p_zero),
        log1m(p_zero) + poisson_lpmf(0 | lambda)
      );
    else
      return log1m(p_zero) + poisson_lpmf(y | lambda);
  }
}

data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real<lower=0> lambda;
  real<lower=0, upper=1> p_zero;
}
model {
  lambda ~ gamma(2, 0.5);
  p_zero ~ beta(1, 5);
  for (n in 1:N)
    target += zip_lpmf(y[n] | lambda, p_zero);
}
```

---

## 7. Stan의 최적화와 변분 추론(Optimization and Variational Inference in Stan)

### 7.1 MAP 추정(MAP Estimation) - 최적화

```python
# Maximum a Posteriori estimation (fast but no uncertainty)
opt_result = model.optimize(data=stan_data, seed=42)
print(f"MAP estimate: mu={opt_result.stan_variable('mu'):.3f}, "
      f"sigma={opt_result.stan_variable('sigma'):.3f}")
```

### 7.2 변분 추론(Variational Inference) - ADVI

```python
# Approximate posterior (fast but approximate)
vi_result = model.variational(data=stan_data, seed=42, output_samples=4000)
print(f"VI estimate: mu={vi_result.stan_variable('mu').mean():.3f}, "
      f"sigma={vi_result.stan_variable('sigma').mean():.3f}")
```

---

## 8. Stan 모델 디버깅(Debugging Stan Models)

### 8.1 일반적인 오류와 해결책(Common Errors and Fixes)

```python
# 1. "Initialization failed" → constrained parameters initialized outside support
#    Fix: provide init values or widen priors

# 2. "Divergent transitions" → posterior geometry too curved
#    Fix: reparameterize (non-centered) or increase adapt_delta

# 3. "Maximum treedepth exceeded" → chain exploring slowly
#    Fix: increase max_treedepth or reparameterize

# 4. "E-BFMI low" → energy distribution has problematic shape
#    Fix: reparameterize (often hierarchical models)

# 5. Compile errors → check Stan syntax carefully
#    Fix: arrays use array[] syntax in Stan 2.26+
```

### 8.2 Stan에서의 출력 디버깅(Print Debugging in Stan)

```stan
model {
  // Use print() for debugging (remove in production!)
  print("mu = ", mu, ", sigma = ", sigma);
  // Use reject() to halt with an error
  if (sigma < 0.001) reject("sigma too small: ", sigma);
}
```

---

## 9. Stan 고급 기능(Advanced Stan Features)

### 9.1 ODE 솔버(ODE Solvers)

```stan
functions {
  vector sir(real t, vector y, real beta, real gamma) {
    vector[3] dydt;
    real S = y[1], I = y[2], R = y[3];
    dydt[1] = -beta * S * I;
    dydt[2] = beta * S * I - gamma * I;
    dydt[3] = gamma * I;
    return dydt;
  }
}
// Use with: ode_rk45(sir, y0, t0, ts, beta, gamma)
```

### 9.2 병렬화를 위한 Map-Reduce(Map-Reduce for Parallelism)

```stan
// Within-chain parallelism using reduce_sum
functions {
  real partial_sum(array[] real y_slice,
                   int start, int end,
                   real mu, real sigma) {
    return normal_lpdf(to_vector(y_slice) | mu, sigma);
  }
}
model {
  target += reduce_sum(partial_sum, y, 1, mu, sigma);
}
```

---

## 10. Stan 모범 사례(Stan Best Practices)

### 10.1 벡터화(Vectorization)

```stan
// SLOW: loop-based
for (n in 1:N)
  y[n] ~ normal(mu, sigma);

// FAST: vectorized (preferred)
y ~ normal(mu, sigma);

// FAST: vectorized with varying parameters
y ~ normal(X * beta, sigma);
```

### 10.2 매개변수화 팁(Parameterization Tips)

```
1. Use non-centered parameterization for hierarchical models
2. Use Cholesky factors instead of covariance matrices
3. Work on the log scale for positive parameters
4. Use ordered vectors for mixture components (identifiability)
5. Standardize predictors before fitting
```

---

## 요약(Summary)

| Stan 블록 | 목적 | 실행 시점 |
|-----------|------|----------|
| `data` | 관측 데이터 선언 | 1회 |
| `transformed data` | 파생 데이터 계산 | 1회 |
| `parameters` | 매개변수 선언 | 매 반복 |
| `transformed parameters` | 파생 매개변수 | 매 반복 |
| `model` | 사전분포 + 가능도 | 매 반복 |
| `generated quantities` | 예측, 로그가능도 | 매 반복 |

| CmdStanPy 메서드 | 목적 |
|------------------|------|
| `model.sample()` | 전체 MCMC |
| `model.optimize()` | MAP 추정 |
| `model.variational()` | ADVI |
| `fit.summary()` | 사후분포 요약 |
| `fit.diagnose()` | 진단 검사 |

---

## 참고문헌(References)

1. Stan Development Team. *Stan User's Guide*: https://mc-stan.org/users/documentation/
2. Carpenter, B., et al. (2017). "Stan: A probabilistic programming language." *JOSS*, 76, 1-32.
3. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
4. Gabry, J., et al. (2019). "Visualization in Bayesian workflow." *JRSS-A*, 182(2), 389-402.

---

[이전: 베이지안 회귀](./06_Bayesian_Regression.md) | [다음: 변분 추론 →](./08_Variational_Inference.md)

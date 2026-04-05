# 18. 캡스톤: 응용 베이지안 프로젝트(Capstone: Applied Bayesian Project)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 18번째

[이전: 불확실성 정량화](./17_Uncertainty_Quantification.md)

---

> **프레임워크 참고**: 이 캡스톤 프로젝트는 PyMC, ArviZ 및 과정의 모든 기법을 통합합니다.
>
> 설치: `pip install pymc arviz numpy scipy matplotlib pandas scikit-learn`

## 학습 목표(Learning Objectives)

- 실전 문제에 전체 베이지안 워크플로우를 처음부터 끝까지 적용
- 여러 후보 모델을 구축, 진단, 비교
- 이해관계자에게 불확실성 소통
- 완전한 A/B 테스트, 임상시험, 또는 추천 시스템 파이프라인 구현

---

## 1. 프로젝트 개요(Project Overview)

이 캡스톤은 세 가지 완전한 베이지안 프로젝트를 제시합니다. 전체 파이프라인을 연습하기 위해 하나를 선택하거나 세 가지 모두 수행하세요.

### 1.1 베이지안 워크플로우 (요약)(The Bayesian Workflow Recap)

```
1. Define the problem and domain knowledge
2. Specify candidate models (priors + likelihood)
3. Prior predictive simulation (sanity check)
4. Fit models (MCMC or VI)
5. Diagnose convergence (R-hat, ESS, divergences)
6. Posterior predictive checks
7. Model comparison (LOO-CV, WAIC)
8. Interpret and communicate results
9. Make decisions under uncertainty
```

---

## 2. 프로젝트 A: 베이지안 A/B 테스트 플랫폼(Project A: Bayesian A/B Testing Platform)

조기 중단, 다중 비교, 비즈니스 지표 통합이 포함된 완전한 A/B 테스트 프레임워크입니다.

### 2.1 문제 설정(Problem Setup)

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

# Simulate an A/B test dataset
n_visitors = {"control": 5000, "variant_a": 4800, "variant_b": 5200}
true_rates = {"control": 0.120, "variant_a": 0.128, "variant_b": 0.115}
revenue_per_conversion = {"control": 45, "variant_a": 52, "variant_b": 40}

data = {}
for variant, n in n_visitors.items():
    conversions = np.random.binomial(n, true_rates[variant])
    revenue = conversions * revenue_per_conversion[variant]
    data[variant] = {"visitors": n, "conversions": conversions, "revenue": revenue}
    print(f"{variant}: {conversions}/{n} = {conversions/n:.3f} "
          f"(true: {true_rates[variant]}), revenue: ${revenue:,}")
```

### 2.2 다변량 베이지안 모델(Multi-Variant Bayesian Model)

```python
with pm.Model() as ab_model:
    variants = list(data.keys())
    n_variants = len(variants)

    # Hierarchical prior: variants share a common prior
    mu_logit = pm.Normal("mu_logit", mu=-2, sigma=1)
    sigma_logit = pm.HalfNormal("sigma_logit", sigma=0.5)

    # Variant-specific conversion rates (non-centered)
    z = pm.Normal("z", mu=0, sigma=1, shape=n_variants)
    logit_p = mu_logit + sigma_logit * z
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

    # Likelihoods
    for i, v in enumerate(variants):
        pm.Binomial(f"obs_{v}", n=data[v]["visitors"], p=p[i],
                    observed=data[v]["conversions"])

    # Derived: lift relative to control
    lift_a = pm.Deterministic("lift_a", (p[1] - p[0]) / p[0])
    lift_b = pm.Deterministic("lift_b", (p[2] - p[0]) / p[0])

    trace_ab = pm.sample(5000, tune=2000, chains=4, random_seed=42)

summary = az.summary(trace_ab, var_names=["p", "lift_a", "lift_b"])
print(summary)
```

### 2.3 의사결정 지표(Decision Metrics)

```python
p_samples = trace_ab.posterior["p"].values.reshape(-1, 3)
p_control, p_a, p_b = p_samples[:, 0], p_samples[:, 1], p_samples[:, 2]

# Probability each variant is best
best_idx = np.argmax(np.column_stack([p_control, p_a, p_b]), axis=1)
for i, name in enumerate(["control", "variant_a", "variant_b"]):
    print(f"P({name} is best): {(best_idx == i).mean():.3f}")

# Expected loss of choosing each variant
def expected_loss(p_chosen, p_all):
    """Expected loss: how much worse than the true best."""
    return np.mean(np.max(p_all, axis=1) - p_chosen)

for i, name in enumerate(["control", "variant_a", "variant_b"]):
    loss = expected_loss(p_samples[:, i], p_samples)
    print(f"Expected loss ({name}): {loss:.5f}")

# Revenue-aware comparison
rev = np.array([45, 52, 40])
expected_rev = p_samples * rev
best_rev_idx = np.argmax(expected_rev, axis=1)
for i, name in enumerate(["control", "variant_a", "variant_b"]):
    print(f"P({name} maximizes revenue): {(best_rev_idx == i).mean():.3f}")
```

### 2.4 순차적 모니터링(Sequential Monitoring)

```python
def sequential_ab_monitor(data_stream, stop_threshold=0.95):
    """Monitor A/B test and stop early if conclusive."""
    alpha, beta = 1, 1  # flat prior

    for day, (n_a, conv_a, n_b, conv_b) in enumerate(data_stream):
        alpha_a = alpha + conv_a
        beta_a = beta + n_a - conv_a
        alpha_b = alpha + conv_b
        beta_b = beta + n_b - conv_b

        # Monte Carlo comparison
        samples_a = np.random.beta(alpha_a, beta_a, 50000)
        samples_b = np.random.beta(alpha_b, beta_b, 50000)
        prob_b_better = (samples_b > samples_a).mean()

        if prob_b_better > stop_threshold or prob_b_better < (1 - stop_threshold):
            winner = "B" if prob_b_better > 0.5 else "A"
            print(f"Day {day}: STOP — {winner} wins with P={max(prob_b_better, 1-prob_b_better):.3f}")
            return day, winner

        alpha += conv_a + conv_b
        beta += (n_a - conv_a) + (n_b - conv_b)

    print("Test inconclusive after all days")
    return len(data_stream), None
```

---

## 3. 프로젝트 B: 베이지안 임상시험 분석(Project B: Bayesian Clinical Trial Analysis)

### 3.1 문제 설정(Problem Setup)

```python
# Phase III clinical trial: drug vs placebo for blood pressure reduction
np.random.seed(42)
n_drug, n_placebo = 150, 150

# True effect: drug reduces BP by 8 mmHg on average
bp_reduction_drug = np.random.normal(8, 12, n_drug)    # high variance
bp_reduction_placebo = np.random.normal(2, 12, n_placebo)

# Some patients have adverse events
adverse_drug = np.random.binomial(1, 0.15, n_drug)
adverse_placebo = np.random.binomial(1, 0.05, n_placebo)
```

### 3.2 하위 그룹 분석을 포함한 계층 모델(Hierarchical Model with Subgroup Analysis)

```python
# Add covariates: age group, baseline BP
age_group_drug = np.random.choice([0, 1, 2], n_drug, p=[0.3, 0.4, 0.3])
age_group_placebo = np.random.choice([0, 1, 2], n_placebo, p=[0.3, 0.4, 0.3])

with pm.Model() as trial_model:
    # Treatment effect (primary endpoint)
    mu_drug = pm.Normal("mu_drug", mu=0, sigma=20)
    mu_placebo = pm.Normal("mu_placebo", mu=0, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=15)

    # Age-group specific effects (hierarchical)
    age_effect = pm.Normal("age_effect", mu=0, sigma=5, shape=3)

    # Likelihoods
    mu_d = mu_drug + age_effect[age_group_drug]
    mu_p = mu_placebo + age_effect[age_group_placebo]

    y_drug = pm.Normal("y_drug", mu=mu_d, sigma=sigma, observed=bp_reduction_drug)
    y_placebo = pm.Normal("y_placebo", mu=mu_p, sigma=sigma, observed=bp_reduction_placebo)

    # Treatment effect
    ate = pm.Deterministic("ate", mu_drug - mu_placebo)

    # Probability of clinically meaningful effect (> 5 mmHg)
    prob_meaningful = pm.Deterministic("prob_meaningful",
                                        pm.math.gt(ate, 5).astype("float64"))

    trace_trial = pm.sample(5000, tune=2000, chains=4, random_seed=42)

# Results
print(az.summary(trace_trial, var_names=["mu_drug", "mu_placebo", "ate"]))

ate_samples = trace_trial.posterior["ate"].values.flatten()
print(f"\nP(ATE > 0): {(ate_samples > 0).mean():.3f}")
print(f"P(ATE > 5): {(ate_samples > 5).mean():.3f}")
print(f"P(ATE > 10): {(ate_samples > 10).mean():.3f}")
```

### 3.3 이상 반응 분석(Adverse Event Analysis)

```python
with pm.Model() as adverse_model:
    p_drug = pm.Beta("p_drug", alpha=1, beta=1)
    p_placebo = pm.Beta("p_placebo", alpha=1, beta=1)

    obs_drug = pm.Binomial("obs_drug", n=n_drug, p=p_drug,
                           observed=adverse_drug.sum())
    obs_placebo = pm.Binomial("obs_placebo", n=n_placebo, p=p_placebo,
                              observed=adverse_placebo.sum())

    risk_ratio = pm.Deterministic("risk_ratio", p_drug / p_placebo)
    risk_diff = pm.Deterministic("risk_diff", p_drug - p_placebo)
    nnt = pm.Deterministic("nnt", 1 / pm.math.abs(risk_diff))

    trace_adverse = pm.sample(5000, tune=1000, random_seed=42)

print(az.summary(trace_adverse, var_names=["p_drug", "p_placebo", "risk_ratio", "nnt"]))
```

---

## 4. 프로젝트 C: 베이지안 추천 시스템(Project C: Bayesian Recommender System)

### 4.1 베이지안 행렬 분해(Bayesian Matrix Factorization)

```python
# Simulate user-item rating data
np.random.seed(42)
n_users, n_items = 200, 50
latent_dim = 5
sparsity = 0.1  # 10% of ratings observed

# True latent factors
U_true = np.random.normal(0, 1, (n_users, latent_dim))
V_true = np.random.normal(0, 1, (n_items, latent_dim))
R_true = U_true @ V_true.T + 3.0  # ratings centered around 3

# Observe a subset with noise
mask = np.random.binomial(1, sparsity, (n_users, n_items)).astype(bool)
R_observed = R_true.copy()
R_observed += np.random.normal(0, 0.5, R_true.shape)
R_observed = np.clip(R_observed, 1, 5)

# Get observed indices
obs_users, obs_items = np.where(mask)
obs_ratings = R_observed[obs_users, obs_items]
n_obs = len(obs_ratings)
print(f"Observed ratings: {n_obs} ({n_obs / (n_users * n_items) * 100:.1f}%)")
```

### 4.2 PyMC 모델(PyMC Model)

```python
with pm.Model() as pmf_model:
    # Hyperpriors
    sigma_u = pm.HalfNormal("sigma_u", sigma=2)
    sigma_v = pm.HalfNormal("sigma_v", sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Latent factors
    U = pm.Normal("U", mu=0, sigma=sigma_u, shape=(n_users, latent_dim))
    V = pm.Normal("V", mu=0, sigma=sigma_v, shape=(n_items, latent_dim))

    # Global bias
    mu = pm.Normal("mu", mu=3, sigma=1)

    # Predicted ratings for observed entries
    pred = mu + pm.math.sum(U[obs_users] * V[obs_items], axis=1)

    # Likelihood
    y = pm.Normal("y", mu=pred, sigma=sigma, observed=obs_ratings)

    # Use ADVI for scalability
    approx = pm.fit(n=30000, method="advi", random_seed=42)

vi_trace = approx.sample(2000)
```

### 4.3 불확실성을 포함한 추천(Recommendation with Uncertainty)

```python
# For a specific user, recommend items they haven't rated
def recommend_with_uncertainty(user_id, vi_trace, n_top=10):
    """Recommend items with uncertainty quantification."""
    U_samples = vi_trace.posterior["U"].values[:, :, user_id, :]  # (chains, draws, dim)
    V_samples = vi_trace.posterior["V"].values  # (chains, draws, items, dim)
    mu_samples = vi_trace.posterior["mu"].values.flatten()

    # Predicted ratings
    U_flat = U_samples.reshape(-1, latent_dim)
    V_flat = V_samples.reshape(-1, n_items, latent_dim)
    pred = np.array([mu_samples[i] + U_flat[i] @ V_flat[i].T for i in range(len(mu_samples))])

    pred_mean = pred.mean(axis=0)
    pred_std = pred.std(axis=0)

    # Items not yet rated
    unrated = ~mask[user_id]
    unrated_items = np.where(unrated)[0]

    # Rank by expected rating
    scores = pred_mean[unrated_items]
    uncertainties = pred_std[unrated_items]

    top_idx = np.argsort(scores)[::-1][:n_top]
    print(f"\nTop {n_top} recommendations for user {user_id}:")
    print(f"{'Item':>6} {'Pred':>8} {'±Std':>8} {'P(>4)':>8}")
    for idx in top_idx:
        item = unrated_items[idx]
        p_good = (pred[:, item] > 4).mean()
        print(f"{item:6d} {scores[idx]:8.2f} {uncertainties[idx]:8.2f} {p_good:8.3f}")

recommend_with_uncertainty(0, vi_trace)
```

### 4.4 탐색 vs 활용(Exploration vs Exploitation)

```python
def thompson_sampling_recommend(user_id, vi_trace):
    """Use Thompson Sampling for exploration-aware recommendations."""
    U_samples = vi_trace.posterior["U"].values.reshape(-1, n_users, latent_dim)
    V_samples = vi_trace.posterior["V"].values.reshape(-1, n_items, latent_dim)
    mu_samples = vi_trace.posterior["mu"].values.flatten()

    # Draw one sample from the posterior
    idx = np.random.randint(len(mu_samples))
    pred = mu_samples[idx] + U_samples[idx, user_id] @ V_samples[idx].T

    # Recommend the item with highest sampled rating
    unrated = ~mask[user_id]
    pred[~unrated] = -np.inf
    best_item = np.argmax(pred)
    return best_item, pred[best_item]
```

---

## 5. 보고서 템플릿(Reporting Template)

### 5.1 경영진 요약 구조(Executive Summary Structure)

```python
def generate_report(trace, project_name):
    """Generate a Bayesian analysis report."""
    report = f"""
    =======================================
    Bayesian Analysis Report: {project_name}
    =======================================

    1. MODEL SPECIFICATION
    - Likelihood: [describe]
    - Priors: [describe and justify]
    - Parameters: [list]

    2. CONVERGENCE DIAGNOSTICS
    - R-hat: all < 1.01 ✓
    - ESS (bulk): all > 400 ✓
    - Divergences: 0 ✓

    3. KEY FINDINGS
    - Parameter estimates with 95% HDI
    - Posterior probability of hypotheses
    - Model comparison results

    4. DECISION RECOMMENDATION
    - Based on expected utility analysis
    - Risk quantification
    - Sensitivity to prior assumptions

    5. LIMITATIONS
    - Model assumptions
    - Data limitations
    - Robustness checks performed
    """
    print(report)

generate_report(trace_ab, "A/B Test Analysis")
```

---

## 6. 전체 파이프라인 체크리스트(Full Pipeline Checklist)

```
□ 문제 정의 및 도메인 조사
□ 데이터 탐색 및 정제
□ 모델 사양 (여러 후보)
□ 사전 예측 시뮬레이션
□ 사전분포 민감도 분석
□ 모델 적합 (4+ 체인의 MCMC)
□ 수렴 진단 (R-hat, ESS, 발산, 트레이스 플롯)
□ 사후 예측 검사
□ 모델 비교 (LOO-CV)
□ 사후분포 해석
□ 불확실성 포함 의사결정 분석
□ 시각화 포함 보고서
□ 코드 리뷰 및 재현성 확인
```

---

## 요약(Summary)

| 프로젝트 | 핵심 기법 | 의사결정 출력 |
|---------|---------|------------|
| A/B 테스트 | 계층적 베타-이항, 순차 분석 | 출시/미출시 + 기대 수익 |
| 임상시험 | 계층적 정규, 하위 그룹 분석 | 약물 승인 권고 + NNT |
| 추천 시스템 | 베이지안 행렬 분해, 톰슨 샘플링 | 신뢰도 포함 개인화 추천 |

---

## 참고 문헌(References)

1. Gelman, A., et al. (2020). "Bayesian Workflow." arXiv:2011.01808.
2. Stucchio, C. (2015). "Bayesian A/B Testing at VWO." Technical report.
3. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). "Probabilistic Programming in Python Using PyMC3." *PeerJ Computer Science*.
4. Mnih, A. & Salakhutdinov, R. (2008). "Probabilistic Matrix Factorization." *NeurIPS*.

---

[이전: 불확실성 정량화](./17_Uncertainty_Quantification.md)

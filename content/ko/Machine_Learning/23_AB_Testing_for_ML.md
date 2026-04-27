# ML 모델을 위한 A/B 테스팅

[← 이전: 22. 프로덕션 ML 서빙](22_Production_ML_Serving.md) | [다음: 24. 기호 회귀 →](24_Symbolic_Regression.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 오프라인 평가 지표만으로는 왜 불충분하며, ML 모델에 온라인 A/B 테스팅이 왜 필요한지 설명할 수 있다
2. 가설 정의, 지표 선택, 표본 크기 계산, 무작위 배정 단위 선택을 통해 ML 모델을 위한 A/B 테스트를 설계할 수 있다
3. 빈도주의 가설 검정(t-검정, 카이제곱 검정)을 적용하고 모델 비교를 위한 신뢰 구간(Confidence Interval)을 계산할 수 있다
4. 순차 검정 방법(SPRT)을 사용하여 위양성률(False Positive Rate)을 증가시키지 않고 유효한 조기 종료 결정을 내릴 수 있다
5. 다중 슬롯머신 알고리즘(Multi-Armed Bandit)(엡실론-탐욕, UCB, 톰슨 샘플링)을 고정 A/B 분할의 적응적 대안으로 구현할 수 있다
6. 모델 변경이 비즈니스 핵심 KPI를 해치지 않도록 가드레일 지표(Guardrail Metrics)를 정의할 수 있다

---

더 나은 모델을 만들었습니다. 교차 검증에서 정확도가 2% 더 높다고 합니다. SHAP 값도 합리적으로 보입니다. 지연 시간(Latency)도 예산 내에 있습니다. 이제 배포해야 할까요? 아직입니다. 오프라인 지표는 *예측 정확도*를 측정하지만, 비즈니스 가치는 *사용자 행동*에서 나옵니다. 정확도가 더 높은 추천 모델이 지나치게 틈새 상품을 추천한다면 오히려 매출을 줄일 수 있습니다. 정밀도(Precision)가 더 높은 사기 탐지 모델이 너무 많은 정상 거래를 차단할 수도 있습니다. A/B 테스팅은 "지표상 더 나은"과 "비즈니스에 더 나은" 사이의 격차를 연결합니다. 이 레슨에서는 온라인 실험의 통계적 방법론 — ML 모델 비교를 위한 A/B 테스트를 설계하고, 실행하고, 분석하는 방법 — 을 다룹니다.

---

> **비유**: 오프라인 평가는 주방에서 레시피를 맛보는 것과 같습니다 — 맛있다는 것을 알 수 있습니다. A/B 테스팅은 레스토랑에서 손님에게 내놓는 것입니다 — 손님이 실제로 주문하는지, 다 먹는지, 다시 오는지를 알 수 있습니다. 때로는 셰프에게 완벽한 레시피가 잘 팔리지 않고, 때로는 "더 못한" 레시피가 "더 나은" 레시피보다 잘 팔립니다. 식당 손님의 취향이 전문 시식가와 다르기 때문입니다.

---

## 1. ML 모델을 A/B 테스트하는 이유

### 이론: 가설 검정 프레임

A/B 테스트는 표준 통계 프레임워크에 맞음:

```
H_0 (귀무):       A와 B 사이에 차이 없음 (μ_A = μ_B)
H_1 (대립):        차이 있음 (μ_A ≠ μ_B)

결정 규칙:    p < α이면, H_0 기각  ⟹  "승자" 선언
```

`α`는 **유의 수준** — 차이가 없는데 있다고 잘못 선언할 확률(제1종 오류). 표준 `α = 0.05`는 "귀무가 실제 참일 때 5% 시간 동안 틀려도 좋다"고 함.

보완 오류: 진짜 차이를 탐지하지 못함(제2종 오류, 확률 `β`). **검정력(Power)**은 `1 - β`, 크기 `δ`의 진짜 차이를 탐지할 확률. 표준 목표는 검정력 = 0.80.

네 양 `(α, β, δ, n)`이 한 식으로 연결. 셋을 고르면 넷째가 결정. 이것이 표본 크기 계획에 필요한 *유일한* 식.

### 1.1 오프라인-온라인 격차(Offline-Online Gap)

```python
"""
Offline Metrics (what you measure in development):
  - Accuracy, F1, AUC-ROC, NDCG, BLEU, ...
  - Measured on historical, static test sets
  - Assumes past data represents future data

Online Metrics (what matters in production):
  - Click-through rate, conversion rate, revenue per user
  - User engagement, session duration, retention
  - Customer satisfaction, support ticket volume

Why they diverge:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Model A: AUC 0.92 → recommends popular items → CTR +3%          │
  │ Model B: AUC 0.95 → recommends niche items  → CTR -1%           │
  │                                                                  │
  │ Model B is "better" offline, but WORSE for the business.         │
  │ This happens because offline metrics don't capture:              │
  │   1. User preference diversity                                   │
  │   2. Feedback loops (popular items get more clicks → more data)  │
  │   3. Position bias (users click top results regardless)          │
  │   4. Novelty effects (new model = new experience → short-term   │
  │      engagement, may not last)                                   │
  └──────────────────────────────────────────────────────────────────┘
"""
```

### 1.2 A/B 테스트를 해야 할 때

```python
"""
✓ Always A/B test when:
  - Deploying a new ML model version to production
  - Changing the model architecture or feature set
  - Modifying business logic downstream of the model
  - The change could affect user experience or revenue

✗ Skip A/B testing when:
  - Bug fix (model was clearly broken, fix is verified offline)
  - Regulatory/compliance requirement (must deploy regardless)
  - No traffic (new product with zero users)
  - Cost of testing exceeds potential benefit

⚠ Consider alternatives when:
  - Very slow feedback loop (months to measure outcome)
    → Use leading indicators or proxy metrics
  - Ethical concerns (can't withhold treatment from control)
    → Use quasi-experimental methods (diff-in-diff, regression discontinuity)
"""
```

---

## 2. 실험 설계(Experimental Design)

### 이론: 표본 크기 공식과 MDE

두 비율(예: 전환율) 비교에 `α = 0.05`와 검정력 = 0.80인 경우, 그룹당 표본 크기는 대략:

```
n ≈ 16 · (p · (1-p)) / δ²        δ = 절대 차이
```

표준편차 `σ`를 가진 연속 지표:

```
n ≈ 16 · σ² / δ²
```

인수 16은 선택된 `α`와 검정력의 표준 정규 임계값에서 옴. 핵심 통찰: **표본 크기가 효과 크기에 역제곱**. 1% 향상을 탐지하려면 2% 향상의 4× 표본; 0.5%를 탐지하려면 16배 많은 사용자가 필요.

**MDE(minimum detectable effect)**가 공식을 역으로: 가진 트래픽에서 신뢰성 있게 탐지할 수 있는 가장 작은 효과는? MDE가 실제 기대하는 향상보다 크면, 테스트는 시작 전에 망함.

### 2.1 A/B 테스트의 구성 요소

```python
"""
Components of an A/B Test:

1. Hypothesis:
   H₀ (null):        New model performs same as current model
   H₁ (alternative): New model performs BETTER than current model

2. Primary Metric (OEC — Overall Evaluation Criterion):
   One metric that captures the business goal.
   Example: "Revenue per user per session"

3. Randomization Unit:
   What gets randomly assigned to treatment/control.
   Options: user_id, session_id, request_id, cookie_id

4. Treatment/Control Split:
   Typically 50/50, but can be asymmetric (e.g., 90/10 for risky changes)

5. Sample Size:
   How many observations needed for statistical significance.
   Determined by: effect size, significance level, statistical power.

6. Duration:
   How long to run the test.
   Must cover at least: 1 full business cycle (usually 1-2 weeks)
"""
```

### 2.2 무작위 배정 단위(Randomization Unit) 선택

```python
"""
Unit         Pros                            Cons
──────────────────────────────────────────────────────────────────
user_id      Consistent experience per       Need user authentication
             user; captures long-term        Slower to reach sample size
             effects

session_id   Faster data collection;         Same user may see both
             no auth needed                  treatments (inconsistent UX)

request_id   Fastest data collection;        No consistency within user
             simplest to implement           journey; high variance

cookie_id    Compromise between user         Cookie deletion/multiple
             and session                     devices cause spillover

Recommendation for ML models:
  - Personalization/recommendation → user_id (consistency matters)
  - Search ranking → session_id or query_id (faster iteration)
  - Fraud detection → request_id (each transaction is independent)
"""
```

### 2.3 표본 크기 계산(검정력 분석, Power Analysis)

```python
import numpy as np
from scipy import stats

def required_sample_size(baseline_rate, minimum_detectable_effect,
                         alpha=0.05, power=0.80):
    """
    Calculate the minimum sample size per group for a two-proportion z-test.

    Why: Running a test too short → underpowered → you may miss a real effect.
    Running a test too long → wasted time and resources.
    Power analysis tells you exactly how many observations you need.

    Parameters:
        baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
        minimum_detectable_effect: Smallest effect worth detecting
            (relative, e.g., 0.10 for a 10% relative lift)
        alpha: Significance level (false positive rate)
        power: Statistical power (1 - false negative rate)

    Returns:
        Required sample size per group
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled standard error under H₀ and H₁
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    p_pool = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_pool * (1 - p_pool))
    se_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    n = ((z_alpha * se_null + z_beta * se_alt) / (p2 - p1)) ** 2
    return int(np.ceil(n))

# Example: 5% baseline CTR, want to detect 10% relative lift (5% → 5.5%)
n = required_sample_size(
    baseline_rate=0.05,
    minimum_detectable_effect=0.10,
    alpha=0.05,
    power=0.80
)
print(f"Required sample size per group: {n:,}")
print(f"Total samples needed: {2 * n:,}")

# How long will it take?
daily_traffic = 10000
days_needed = np.ceil(2 * n / daily_traffic)
print(f"At {daily_traffic:,} users/day: {days_needed:.0f} days")
```

### 2.4 기간 고려사항(Duration Considerations)

```python
"""
Minimum test duration checklist:

□ Covers at least 1 full weekly cycle (weekday + weekend patterns)
□ Reaches required sample size
□ Avoids holidays and special events (Black Friday, etc.)
□ Accounts for novelty/primacy effects (users react differently to new things)
□ Allows time for long-term effects to manifest (1-2 week minimum)

Common mistake: "We got significant results after 2 hours!"
Problem: Early significance is often driven by:
  1. Time-of-day bias (morning users differ from evening users)
  2. Day-of-week effects (Monday traffic ≠ Saturday traffic)
  3. Multiple peeking (checking p-values repeatedly inflates Type I error)
"""
```

---

## 3. 통계적 기초(Statistical Foundations)

### 이론: 비율과 평균에 대한 Z-검정과 t-검정

이진 결과(클릭함 / 안 함):

```
z = (p̂_A - p̂_B) / sqrt(p̂(1-p̂)(1/n_A + 1/n_B))     p̂는 풀링됨
p-값 = 2 · (1 - Φ(|z|))                              양측
```

연속 결과(사용자당 수익, 페이지 시간):

```
t = (x̄_A - x̄_B) / sqrt(s²(1/n_A + 1/n_B))
p-값은 적절한 자유도의 t 분포에서
```

매우 큰 `n`에 대해 둘은 본질적으로 동일; t-검정이 일반적으로 안전한 선택.

A/B 테스트 플랫폼(Optimizely, Google Optimize, scikit-learn, scipy.stats)이 이 검정을 래핑하지만, 기저 수학이 정확히 위. 이를 알면 보여진 숫자를 정신 점검할 수 있음.

### 3.1 모델 비교를 위한 가설 검정(Hypothesis Testing)

```python
def ab_test_proportions(n_control, conversions_control,
                        n_treatment, conversions_treatment,
                        alpha=0.05):
    """
    Two-proportion z-test for comparing conversion rates.

    Why z-test: For large sample sizes (n > 30), the Central Limit Theorem
    guarantees the sample proportion is approximately normal, making the
    z-test both simple and reliable.
    """
    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed

    # Confidence interval for the difference
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control +
                      p_treatment * (1 - p_treatment) / n_treatment)
    ci_lower = (p_treatment - p_control) - stats.norm.ppf(1 - alpha/2) * se_diff
    ci_upper = (p_treatment - p_control) + stats.norm.ppf(1 - alpha/2) * se_diff

    relative_lift = (p_treatment - p_control) / p_control * 100

    print(f"Control:   {p_control:.4f} ({conversions_control}/{n_control})")
    print(f"Treatment: {p_treatment:.4f} ({conversions_treatment}/{n_treatment})")
    print(f"Lift:      {relative_lift:+.2f}%")
    print(f"95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value:   {p_value:.4f}")
    print(f"Result:    {'Significant' if p_value < alpha else 'Not significant'}")

    return {
        "p_control": p_control, "p_treatment": p_treatment,
        "lift": relative_lift, "p_value": p_value,
        "ci": (ci_lower, ci_upper),
        "significant": p_value < alpha
    }

# Example: New recommendation model vs. current
result = ab_test_proportions(
    n_control=50000, conversions_control=2500,       # 5.0% CTR
    n_treatment=50000, conversions_treatment=2650,    # 5.3% CTR
)
```

### 3.2 연속형 지표(매출, 참여 시간)

```python
def ab_test_means(control_values, treatment_values, alpha=0.05):
    """
    Welch's t-test for comparing means of continuous metrics.

    Why Welch's: Unlike Student's t-test, it doesn't assume equal variances.
    Revenue and engagement time often have very different variances between
    control and treatment (e.g., a new model might increase variance by
    creating more extreme outcomes).
    """
    n1, n2 = len(control_values), len(treatment_values)
    mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
    std1, std2 = np.std(control_values, ddof=1), np.std(treatment_values, ddof=1)

    t_stat, p_value = stats.ttest_ind(control_values, treatment_values,
                                       equal_var=False)

    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
    ci_lower = (mean2 - mean1) - stats.t.ppf(1 - alpha/2, df=min(n1, n2) - 1) * se_diff
    ci_upper = (mean2 - mean1) + stats.t.ppf(1 - alpha/2, df=min(n1, n2) - 1) * se_diff

    print(f"Control mean:   {mean1:.4f} (std={std1:.4f})")
    print(f"Treatment mean: {mean2:.4f} (std={std2:.4f})")
    print(f"Difference:     {mean2 - mean1:+.4f}")
    print(f"95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value:        {p_value:.4f}")

    return {"p_value": p_value, "ci": (ci_lower, ci_upper)}

# Example: Revenue per user (highly skewed)
np.random.seed(42)
control_revenue = np.random.exponential(scale=10, size=10000)
treatment_revenue = np.random.exponential(scale=10.5, size=10000)  # 5% lift
ab_test_means(control_revenue, treatment_revenue)
```

### 3.3 다중 비교 보정(Multiple Comparison Correction)

```python
"""
The Problem:
  Testing 1 metric at α=0.05 → 5% false positive rate
  Testing 10 metrics at α=0.05 → 1 - (1-0.05)^10 = 40% chance of ANY false positive!

Correction Methods:

  1. Bonferroni: α_adjusted = α / n_tests
     Simple but conservative. Use when tests are independent.
     10 metrics → α = 0.05/10 = 0.005

  2. Benjamini-Hochberg (FDR): Controls the false discovery rate
     Less conservative than Bonferroni. Use when you can tolerate some false positives.
     Sort p-values, compare each p(i) to (i/m) × α

  3. Primary + Secondary metric framework:
     Designate ONE primary metric (apply α=0.05)
     All others are "guardrails" (directional, no strict threshold)
     Most practical approach for ML A/B tests.
"""
```

```python
from scipy.stats import false_discovery_control

def apply_corrections(p_values, alpha=0.05):
    """Compare Bonferroni and BH corrections on multiple p-values."""
    n = len(p_values)
    metrics = [f"metric_{i+1}" for i in range(n)]

    # Bonferroni
    bonferroni_threshold = alpha / n

    # Benjamini-Hochberg
    sorted_indices = np.argsort(p_values)
    bh_thresholds = [(i + 1) / n * alpha for i in range(n)]
    bh_significant = [False] * n
    for rank, idx in enumerate(sorted_indices):
        if p_values[idx] <= bh_thresholds[rank]:
            bh_significant[idx] = True

    print(f"{'Metric':<12s} {'p-value':>8s} {'Raw':>5s} {'Bonf':>5s} {'BH':>5s}")
    print("-" * 42)
    for i in range(n):
        raw_sig = "✓" if p_values[i] < alpha else ""
        bonf_sig = "✓" if p_values[i] < bonferroni_threshold else ""
        bh_sig = "✓" if bh_significant[i] else ""
        print(f"{metrics[i]:<12s} {p_values[i]:>8.4f} {raw_sig:>5s} {bonf_sig:>5s} {bh_sig:>5s}")

# Example: 6 metrics tested, some are false positives
p_values = np.array([0.001, 0.013, 0.029, 0.041, 0.15, 0.88])
apply_corrections(p_values)
```

---

## 4. 순차 검정(Sequential Testing)

### 이론: 순차 검정과 베이지안 대안

고전적 고정 표본 크기 A/B 테스트는 두 가지 통점: 엿보기가 금지, 사전에 `n`에 약속해야 함. 두 대안이 이를 다룸:

- **순차 검정**(Wald의 SPRT, mSPRT, 그룹 순차): 제1종 오류를 부풀리지 않고 데이터를 주기적으로 확인할 수 있는 명시적 정지 규칙. 결과가 어느 방향으로든 결정적일 때 일찍 멈출 수 있음.
- **베이지안 A/B 테스트**: 데이터가 주어졌을 때 B가 A보다 나을 사후 확률 계산. 이 사후가 임계값을 넘을 때 멈춤. p-값보다 해석이 더 자연스러움("80% 확률로 B가 더 나음"), 엿보기 페널티 없이 연속 모니터링 지원.

현대 A/B 플랫폼(Eppo, Statsig, GrowthBook)이 보통 이 중 하나를 기본으로 구현.

### 4.1 엿보기 문제(The Peeking Problem)

```python
"""
The peeking problem:

  Day 1: p = 0.03 → "Significant! Ship it!"  ← WRONG!
  Day 2: p = 0.12 → "Hmm, not significant anymore..."
  Day 3: p = 0.04 → "It's back! Ship it!"     ← STILL WRONG!

Why it's wrong:
  Each peek is an independent test. If you check daily for 14 days,
  the probability of seeing at least one false positive is:
    1 - (1 - 0.05)^14 ≈ 51%

  The p-value is only valid for a SINGLE, pre-committed analysis.
  Repeated checking inflates the false positive rate dramatically.

Solutions:
  1. Pre-commit to a fixed sample size and analyze ONCE
  2. Use sequential testing methods (designed for continuous monitoring)
  3. Use Bayesian methods (naturally handle continuous updating)
"""
```

### 4.2 순차 확률비 검정(Sequential Probability Ratio Test, SPRT)

```python
def sequential_ab_test(control_outcomes, treatment_outcomes,
                       p0=0.05, p1=0.055, alpha=0.05, beta=0.20):
    """
    Sequential Probability Ratio Test for A/B testing.

    Why SPRT: It allows you to monitor results continuously and stop
    as soon as there's enough evidence — without inflating false positive rates.
    On average, SPRT needs 50-70% fewer samples than a fixed-sample test.

    Parameters:
        control_outcomes: Array of 0/1 outcomes for control
        treatment_outcomes: Array of 0/1 outcomes for treatment
        p0: Expected rate under H₀ (no difference)
        p1: Expected rate under H₁ (treatment is better)
        alpha: Type I error rate
        beta: Type II error rate
    """
    # Wald boundaries
    A = np.log((1 - beta) / alpha)     # Upper boundary (reject H₀)
    B = np.log(beta / (1 - alpha))     # Lower boundary (accept H₀)

    log_likelihood_ratio = 0
    decisions = []

    n = min(len(control_outcomes), len(treatment_outcomes))

    for i in range(n):
        x_c = control_outcomes[i]
        x_t = treatment_outcomes[i]

        # Log-likelihood ratio update
        if x_t == 1:
            log_likelihood_ratio += np.log(p1 / p0)
        else:
            log_likelihood_ratio += np.log((1 - p1) / (1 - p0))

        if log_likelihood_ratio >= A:
            decisions.append((i + 1, "REJECT H₀", "Treatment is better"))
            break
        elif log_likelihood_ratio <= B:
            decisions.append((i + 1, "ACCEPT H₀", "No significant difference"))
            break

    if not decisions:
        decisions.append((n, "INCONCLUSIVE", "Need more data"))

    sample_used, decision, interpretation = decisions[0]
    print(f"Decision at sample {sample_used}: {decision}")
    print(f"Interpretation: {interpretation}")
    print(f"Final log-LR: {log_likelihood_ratio:.4f}")
    print(f"Boundaries: reject H₀ > {A:.4f}, accept H₀ < {B:.4f}")
    return decisions

# Simulate: treatment has a real 10% relative improvement
np.random.seed(42)
control = np.random.binomial(1, 0.05, size=10000)
treatment = np.random.binomial(1, 0.055, size=10000)
sequential_ab_test(control, treatment)
```

---

## 5. 다중 슬롯머신(Multi-Armed Bandits)

### 이론: 인과 식별의 이야기

A/B 테스트의 궁극적 정당화는 **무작위 할당**. 무작위 할당이 처치 그룹을 *기댓값*에서 통제 그룹과 통계적으로 동일하게 만드므로, 관찰된 차이는 처치에 귀속 가능. 무작위 할당 없이는(예: 새 모델에 옵트인 선택한 사용자와 그렇지 않은 사용자 비교), 교란이 인과 추론을 불가능하게 함.

이래서 "점진적 램프업"(카나리)과 "N일의 새 모델을 N-30일의 옛 모델과 비교"가 적절한 A/B 테스트의 대체가 아닙니다. 첫 번째는 동시대 통제가 부족; 두 번째는 고정 시간 윈도우 안의 무작위 할당이 부족. *위험 감소*에는 유용하지만 *인과 귀속*에는 아님.

### 5.1 A/B 테스팅 vs. 밴딧(Bandits)

```python
"""
A/B Testing:                      Multi-Armed Bandits:
┌─────────────────────────┐      ┌─────────────────────────┐
│ Fixed 50/50 split       │      │ Adaptive allocation     │
│ Explore for N samples   │      │ Explore + exploit        │
│ Then exploit winner     │      │ simultaneously           │
│                         │      │                         │
│ Regret: high during     │      │ Regret: lower because   │
│ exploration phase       │      │ traffic shifts to winner │
│                         │      │ automatically            │
│ Statistical rigor: ✓✓✓  │      │ Statistical rigor: ✓✓   │
│ Speed to deploy: slow   │      │ Speed to deploy: fast   │
└─────────────────────────┘      └─────────────────────────┘

When to use A/B testing:
  ✓ Need rigorous statistical conclusions
  ✓ Measuring long-term effects
  ✓ Management requires p-value evidence
  ✓ Few variants (2-3)

When to use bandits:
  ✓ Many variants to test (5+)
  ✓ High cost of serving worse variant
  ✓ Need to minimize regret during experiment
  ✓ Short-lived content (news, promotions)
"""
```

### 5.2 엡실론-탐욕(Epsilon-Greedy)

```python
class EpsilonGreedy:
    """
    The simplest bandit algorithm.

    Why it works: With probability (1-ε), choose the best arm (exploit).
    With probability ε, choose a random arm (explore).
    Simple but effective — often a good baseline.
    """
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)       # How many times each arm was pulled
        self.rewards = np.zeros(n_arms)      # Total reward per arm
        self.estimates = np.zeros(n_arms)    # Estimated mean reward per arm

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        return np.argmax(self.estimates)            # Exploit

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        # Why: Incremental mean update — avoids storing all past rewards.
        # new_mean = old_mean + (reward - old_mean) / count
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

# Simulate: 3 model variants with different true CTRs
np.random.seed(42)
true_rates = [0.05, 0.052, 0.058]  # Model C is best
bandit = EpsilonGreedy(n_arms=3, epsilon=0.1)

total_reward = 0
for t in range(10000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    bandit.update(arm, reward)
    total_reward += reward

print("Epsilon-Greedy Results:")
for i in range(3):
    print(f"  Model {chr(65+i)}: pulled {int(bandit.counts[i]):>5d} times, "
          f"est. rate={bandit.estimates[i]:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward}")
```

### 5.3 신뢰 상한(Upper Confidence Bound, UCB)

```python
class UCB1:
    """
    UCB1 — balances exploration and exploitation using confidence intervals.

    Why UCB: Instead of exploring randomly (epsilon-greedy), UCB explores
    arms with high UNCERTAINTY. An arm that hasn't been tried much gets a
    wide confidence interval, making it more likely to be selected.
    This is provably optimal (up to constants) in the regret sense.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self):
        # Pull each arm once first
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        # UCB formula: estimate + sqrt(2 * ln(t) / count)
        # Why: The second term is the "exploration bonus" — it grows when
        # an arm hasn't been pulled recently, encouraging exploration.
        ucb_values = self.estimates + np.sqrt(
            2 * np.log(self.total_pulls) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_pulls += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

# Compare UCB vs Epsilon-Greedy
ucb = UCB1(n_arms=3)
total_reward_ucb = 0
for t in range(10000):
    arm = ucb.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    ucb.update(arm, reward)
    total_reward_ucb += reward

print("\nUCB1 Results:")
for i in range(3):
    print(f"  Model {chr(65+i)}: pulled {int(ucb.counts[i]):>5d} times, "
          f"est. rate={ucb.estimates[i]:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward_ucb} (vs ε-greedy: {total_reward})")
```

### 5.4 톰슨 샘플링(Thompson Sampling)

```python
class ThompsonSampling:
    """
    Thompson Sampling — Bayesian approach to multi-armed bandits.

    Why Thompson Sampling: It maintains a probability distribution over
    each arm's true reward rate. Arms with high uncertainty get explored
    naturally because their samples occasionally produce high values.
    As evidence accumulates, the distribution narrows and the algorithm
    exploits the best arm.

    Often achieves the best empirical performance among bandit algorithms.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta(1, 1) = Uniform prior (no prior knowledge)
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1

    def select_arm(self):
        # Why: Sample from each arm's posterior Beta distribution.
        # The arm with the highest sample gets pulled.
        # High-uncertainty arms produce high samples more often → explored more.
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        # Why: Beta-Bernoulli conjugacy gives exact posterior update.
        # Success → alpha += 1, Failure → beta += 1.
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Run Thompson Sampling
np.random.seed(42)
ts = ThompsonSampling(n_arms=3)
total_reward_ts = 0
for t in range(10000):
    arm = ts.select_arm()
    reward = np.random.binomial(1, true_rates[arm])
    ts.update(arm, reward)
    total_reward_ts += reward

print("\nThompson Sampling Results:")
for i in range(3):
    pulls = int(ts.alpha[i] + ts.beta[i] - 2)
    est = (ts.alpha[i] - 1) / pulls if pulls > 0 else 0
    print(f"  Model {chr(65+i)}: pulled {pulls:>5d} times, "
          f"est. rate={est:.4f} (true={true_rates[i]:.3f})")
print(f"Total reward: {total_reward_ts} (vs UCB: {total_reward_ucb}, "
      f"ε-greedy: {total_reward})")
```

---

## 6. 인터리빙 실험(Interleaving Experiments)

### 이론: p-값 함정

p-값은 *귀무가 참이라면* 관찰된 것보다 *적어도 극단적인* 데이터를 볼 확률. **아닌 것**:

- 귀무가 참일 확률.
- 결과가 우연 때문일 확률.
- 효과의 크기.

두 가지 흔한 실수:

1. **엿보기(Peeking)**: p-값을 계속 확인하고 0.05를 넘을 때 멈춤. 이는 보장된 거짓 양성 기계 — 귀무가 참일 때조차, 반복된 엿보기가 결국 우연으로 `p < 0.05`를 줄 것. "30일 동안 매일 엿보기"의 거짓 양성률은 5%가 아니라 약 30%. 시작 전 고정 표본 크기에 약속하거나 **순차 검정**(mSPRT, 그룹 순차 설계)을 사용.
2. **통계적 유의성과 실용적 유의성 혼동**: 충분한 표본 크기로 *어떤* 0이 아닌 차이도 통계적으로 유의해짐. "유의한" 0.01% 향상이 공학적 비용 가치가 없을 수 있음. 항상 p-값과 효과 크기 신뢰 구간 둘 다 보고.

### 6.1 인터리빙이 필요한 이유

```python
"""
For ranking/recommendation models, A/B testing requires MANY users because
each user sees either model A or model B results. Individual preference
is noisy, so you need large samples to detect small improvements.

Interleaving shows BOTH models' results to the SAME user in a merged list.
The user's clicks directly reveal which model's results they prefer.

Sensitivity: Interleaving needs ~100x fewer users than A/B testing
to detect the same effect (Chapelle et al., 2012).

Use cases: Search ranking, recommendation lists, ad ordering.

┌─────────────────────────────────────────────────────┐
│  A/B Test:                                          │
│    User 1 sees:  [A₁, A₂, A₃, A₄, A₅]             │
│    User 2 sees:  [B₁, B₂, B₃, B₄, B₅]             │
│    Compare: CTR(group A) vs CTR(group B)            │
│                                                     │
│  Interleaving:                                      │
│    User 1 sees:  [A₁, B₁, A₂, B₂, A₃]  (merged)   │
│    Count clicks on A-originated vs B-originated items│
│    Much more sensitive — same user, direct comparison│
└─────────────────────────────────────────────────────┘
"""
```

### 6.2 팀 드래프트 인터리빙(Team Draft Interleaving)

```python
def team_draft_interleaving(ranking_a, ranking_b, k=10):
    """
    Team Draft Interleaving algorithm.

    Why Team Draft: Like picking teams in sports — models take turns choosing
    their top-ranked item. This ensures balanced representation from both
    models while preserving relative ordering.

    Args:
        ranking_a: Ordered list of items from model A
        ranking_b: Ordered list of items from model B
        k: Number of items in the interleaved list
    """
    interleaved = []
    team_a = set()  # Items attributed to model A
    team_b = set()  # Items attributed to model B
    idx_a, idx_b = 0, 0

    for position in range(k):
        # Alternate which model picks first (with tie-breaking by team size)
        a_picks = len(team_a) <= len(team_b)

        if a_picks:
            # Model A picks its highest-ranked item not yet in the list
            while idx_a < len(ranking_a) and ranking_a[idx_a] in interleaved:
                idx_a += 1
            if idx_a < len(ranking_a):
                item = ranking_a[idx_a]
                interleaved.append(item)
                team_a.add(item)
                idx_a += 1
        else:
            while idx_b < len(ranking_b) and ranking_b[idx_b] in interleaved:
                idx_b += 1
            if idx_b < len(ranking_b):
                item = ranking_b[idx_b]
                interleaved.append(item)
                team_b.add(item)
                idx_b += 1

    return interleaved, team_a, team_b

# Example
ranking_a = ['item_1', 'item_3', 'item_5', 'item_7', 'item_9']
ranking_b = ['item_2', 'item_1', 'item_4', 'item_3', 'item_6']

interleaved, team_a, team_b = team_draft_interleaving(ranking_a, ranking_b, k=6)
print(f"Model A ranking: {ranking_a}")
print(f"Model B ranking: {ranking_b}")
print(f"Interleaved:     {interleaved}")
print(f"Team A items:    {team_a}")
print(f"Team B items:    {team_b}")
```

---

## 7. 가드레일 지표(Guardrail Metrics)

### 이론: 다중 비교 — 본페로니 함정

각각 `α = 0.05`로 20개 독립 A/B 테스트를 실행하면, 적어도 하나가 거짓 양성일 확률은 `1 - 0.95²⁰ ≈ 0.64`. 이것이 **다중 비교 문제** — 가족별 오류율(family-wise error rate)이 테스트 수와 함께 폭발.

세 가지 보정:

- **Bonferroni**: `α`를 테스트 수로 나눔. 보수적 — 가족별 오류 ≤ `α` 보장. 검정력 비용.
- **Benjamini-Hochberg (FDR)**: 가족별 오류 대신 거짓 발견율(거짓 양성의 선언된 양성 중 기대 비율)을 통제. 덜 보수적, 더 많은 검정력.
- **조정된 임계값을 가진 가족별 검정**: 예: Holm-Bonferroni — 같은 FWE 수준에서 평범한 Bonferroni보다 약간 더 강력.

ML A/B 테스트의 경우 여러 지표(CTR, 수익, 유지, 지연시간, ...)나 여러 변형(A vs B vs C vs D)에 걸쳐 비교할 때 중요. 결과 엿보기 *전에* 보정 결정.

### 7.1 가드레일 지표란?

```python
"""
Primary metric: The ONE metric you're optimizing for.
  Example: Revenue per user

Guardrail metrics: Metrics that must NOT degrade.
  They protect against unintended side effects.

┌────────────────────────┬────────────────────────────────────┐
│ Category               │ Example Guardrails                  │
├────────────────────────┼────────────────────────────────────┤
│ User experience        │ Page load time, error rate,         │
│                        │ crash rate, bounce rate              │
├────────────────────────┼────────────────────────────────────┤
│ Business health        │ Revenue, transactions per user,     │
│                        │ customer support contacts            │
├────────────────────────┼────────────────────────────────────┤
│ Engineering health     │ Server latency (p99), CPU usage,    │
│                        │ memory consumption                   │
├────────────────────────┼────────────────────────────────────┤
│ Fairness               │ Outcome equality across protected   │
│                        │ groups (gender, age, region)         │
└────────────────────────┴────────────────────────────────────┘

Decision rule:
  IF primary metric improves AND all guardrails pass → SHIP
  IF primary metric improves BUT guardrail fails → INVESTIGATE
  IF primary metric doesn't improve → DON'T SHIP
"""
```

### 7.2 가드레일 구현

```python
def evaluate_experiment(primary_result, guardrail_results,
                        primary_alpha=0.05, guardrail_threshold=0.01):
    """
    Evaluate A/B test with primary metric and guardrails.

    Why separate thresholds: The primary metric uses standard α=0.05.
    Guardrails use a stricter threshold for degradation (e.g., 0.01)
    because we want to be VERY sure there's no harm before ignoring
    a guardrail regression.
    """
    # Primary metric: is it significantly better?
    primary_pass = (primary_result['p_value'] < primary_alpha and
                    primary_result['lift'] > 0)

    # Guardrails: is any significantly WORSE?
    guardrail_failures = []
    for name, result in guardrail_results.items():
        if result['p_value'] < guardrail_threshold and result['lift'] < 0:
            guardrail_failures.append((name, result['lift']))

    print("=" * 50)
    print("EXPERIMENT DECISION")
    print("=" * 50)
    print(f"\nPrimary metric: {'PASS ✓' if primary_pass else 'FAIL ✗'}")
    print(f"  Lift: {primary_result['lift']:+.2f}%, p={primary_result['p_value']:.4f}")

    print(f"\nGuardrails: {len(guardrail_failures)} failures")
    for name, lift in guardrail_failures:
        print(f"  ✗ {name}: {lift:+.2f}%")

    if primary_pass and not guardrail_failures:
        print("\n→ DECISION: SHIP IT")
    elif primary_pass and guardrail_failures:
        print("\n→ DECISION: INVESTIGATE guardrail failures before shipping")
    else:
        print("\n→ DECISION: DO NOT SHIP")

# Example
primary = {"lift": 3.5, "p_value": 0.008}
guardrails = {
    "latency_p99":    {"lift": -0.5, "p_value": 0.45},    # Not significant
    "error_rate":     {"lift": 0.1,  "p_value": 0.82},    # Not significant
    "revenue":        {"lift": 1.2,  "p_value": 0.12},    # Positive but not sig
}
evaluate_experiment(primary, guardrails)
```

---

## 8. 실용적 A/B 테스팅 워크플로우

### 이론: ML 고유의 흔한 함정

ML 모델 A/B 테스트는 일반 A/B 테스트가 가지지 않는 실패 모드를 가짐:

- **네트워크 효과**: 사용자 A의 경험이 다른 사용자가 보는 것에 의존하면(추천, 소셜 피드), 무작위화의 단위가 개체가 아니라 클러스터여야 함. 그렇지 않으면 독립 처치 효과의 가정이 위반.
- **장기 효과**: 단기 클릭을 늘리는 모델이 장기 유지를 침식할 수 있음. 둘 다 포착할 만큼 충분히 오래 테스트 실행, 또는 **홀드백 실험** 실행(사용자의 일부가 통제로 몇 달 동안 항상 옛 모델을 봄).
- **이질적 처치 효과**: 평균 효과가 양수일 수 있지만 결정적인 하위 인구가 해를 받음. 항상 사용자 세그먼트, 지역, 기기 등으로 결과 슬라이스.
- **선택 편향**: A 또는 B에의 할당이 사용자 행동에 의존하면(예: "어제 클릭한 사용자가 모델 A를 봄"), 인과 효과 추론 불가. 무작위 할당은 결과에 무조건적이어야 함.

이는 통계적 정교함 문제가 아닙니다 — 어떤 테스트가 실행되기 전에 내려진 아키텍처적 결정.

### 8.1 전체 체크리스트

```python
"""
Phase 1: DESIGN
  □ Define primary metric (one metric, clearly measurable)
  □ Define guardrail metrics (2-5 metrics that must not degrade)
  □ State hypothesis (H₀ and H₁)
  □ Calculate required sample size (power analysis)
  □ Choose randomization unit (user_id, session_id, etc.)
  □ Determine test duration (≥ 1 weekly cycle)
  □ Document in experiment brief (share with stakeholders)

Phase 2: IMPLEMENT
  □ Implement feature flag / traffic routing
  □ Verify randomization is uniform (chi-square test on assignment)
  □ Log all necessary metrics (before running the test)
  □ Set up monitoring dashboard

Phase 3: RUN
  □ Start with small percentage (5-10%) → ramp to full
  □ Monitor guardrails daily (automated alerts)
  □ Do NOT peek at primary metric p-values (or use sequential testing)
  □ Run for pre-committed duration

Phase 4: ANALYZE
  □ Compute primary metric: effect size, CI, p-value
  □ Check guardrails: any significant degradation?
  □ Segment analysis: does the effect hold across segments?
  □ Check for novelty effects: first-week vs second-week trends
  □ Apply multiple comparison correction if testing multiple metrics

Phase 5: DECIDE
  □ Primary positive + guardrails pass → Ship
  □ Primary positive + guardrail fails → Investigate
  □ Primary neutral/negative → Don't ship
  □ Document decision and learnings
"""
```

### 8.2 흔한 실수들(Common Pitfalls)

```python
"""
Pitfall                          Fix
──────────────────────────────────────────────────────────────────────
1. Peeking at results early      Use sequential testing or pre-commit
                                 to a fixed analysis date

2. Stopping at first             Run for the full pre-committed
   significance                  duration

3. Wrong randomization unit      User-level for personalization,
                                 session-level for search

4. Network effects ignored       Use cluster-based randomization
                                 (geo, time buckets)

5. Survivorship bias             Include ALL users in analysis, not
                                 just those who completed actions

6. Seasonal confounds            Run across full business cycles;
                                 use year-over-year comparisons

7. Underpowered test             Always do power analysis first;
                                 better to run longer than to guess

8. Too many metrics              One primary metric. Everything
                                 else is a guardrail or secondary.
"""
```

---

## 연습문제

### 연습문제 1: 검정력 분석과 테스트 설계

```python
"""
A recommendation model has a 3.2% click-through rate (baseline).
You want to detect a 5% relative improvement (3.2% → 3.36%).

1. Calculate the required sample size per group (α=0.05, power=0.80)
2. If your site gets 5,000 users/day, how many days do you need?
3. Recalculate for: (a) 10% relative improvement, (b) power=0.90
4. Plot: required sample size vs. minimum detectable effect (MDE)
   for MDE from 1% to 20% relative improvement
5. Discussion: What if you can only run the test for 2 weeks?
   What is the smallest effect you can detect?
"""
```

### 연습문제 2: A/B 테스트 분석

```python
"""
You ran an A/B test for 14 days. Results:

  Control:   n=25,000, conversions=1,200, avg_revenue=$12.50 (std=$45.00)
  Treatment: n=25,000, conversions=1,350, avg_revenue=$13.10 (std=$48.00)

1. Perform a two-proportion z-test on conversion rate
2. Perform a Welch's t-test on revenue per user
3. Calculate 95% confidence intervals for both
4. The treatment also increased p99 latency by 15ms (from 85ms to 100ms).
   Is this a guardrail violation? How would you decide?
5. Should you ship the new model? Justify your decision.
"""
```

### 연습문제 3: 다중 슬롯머신 알고리즘 비교

```python
"""
You have 5 model variants with true CTRs: [0.04, 0.042, 0.045, 0.048, 0.05].
(You don't know these — the bandit must discover them.)

1. Implement and run for 50,000 rounds:
   a. Epsilon-greedy (ε=0.1)
   b. UCB1
   c. Thompson Sampling
   d. Random (baseline — choose uniformly at random)
2. For each algorithm, track:
   - Cumulative reward over time
   - Fraction of pulls allocated to each arm
   - Cumulative regret (vs. always choosing the best arm)
3. Plot cumulative regret curves for all 4 algorithms
4. Which algorithm converges fastest to the best arm?
"""
```

### 연습문제 4: 순차 검정

```python
"""
1. Implement an SPRT-based sequential A/B test
2. Simulate 1,000 experiments where:
   a. H₀ is true (both arms have 5% CTR)
   b. H₁ is true (control=5%, treatment=5.5%)
3. For each experiment, record:
   - Sample size at stopping
   - Decision (reject H₀ or accept H₀)
4. Calculate:
   - False positive rate (should be ≤ α=0.05)
   - True positive rate / power (should be ≥ 1-β=0.80)
   - Average sample size (compare to fixed-sample test)
5. Bonus: Compare SPRT to a naive approach that checks p-value daily
"""
```

---

## 9. 요약

### 핵심 정리

| 개념 | 설명 |
|---------|-------------|
| **오프라인-온라인 격차(Offline-Online Gap)** | 오프라인 지표(AUC, F1)가 항상 비즈니스 결과를 예측하지는 않는다 — A/B 테스팅이 이 격차를 연결한다 |
| **검정력 분석(Power Analysis)** | 테스트를 실행하기 전에 표본 크기를 계산해야 한다. 검정력이 부족한 테스트는 시간 낭비이다 |
| **다중 비교 보정(Multiple Comparisons)** | 여러 지표를 검정하면 위양성이 증가한다 — 본페로니(Bonferroni)나 BH 보정을 사용해야 한다 |
| **순차 검정(Sequential Testing)** | SPRT는 위양성률을 높이지 않으면서 지속적인 모니터링을 가능하게 한다 |
| **다중 슬롯머신(Multi-Armed Bandits)** | 고정 A/B 분할의 적응적 대안으로, 실험 중 후회(Regret)를 최소화한다 |
| **톰슨 샘플링(Thompson Sampling)** | 베이지안 밴딧 — 실증적으로 가장 좋은 성능을 보이며, 탐색과 활용을 자연스럽게 균형 잡는다 |
| **인터리빙(Interleaving)** | 랭킹/추천에서 A/B보다 100배 민감하다 — 같은 사용자에게 두 모델의 결과를 모두 보여준다 |
| **가드레일 지표(Guardrail Metrics)** | 비즈니스 핵심 KPI를 보호한다. 주요 지표가 개선되고 가드레일이 통과할 때만 배포한다 |

### 다른 레슨과의 연결

- **L04 (모델 평가)**: 오프라인 지표는 필요하지만 충분하지 않다 — A/B 테스팅은 온라인 보완 수단이다
- **L16 (모델 설명 가능성)**: 모델이 *왜* 더 나은지 설명하면 A/B 테스트 결과를 해석하는 데 도움이 된다
- **L22 (프로덕션 ML 서빙)**: A/B 테스팅은 프로덕션에 최적화된 모델이 여전히 비즈니스 가치를 제공하는지 검증한다
- **Data_Science L08 (가설 검정)**: A/B 테스트 분석에 사용되는 기초 통계학
- **MLOps L08-09 (모델 서빙)**: 모델 버전 간 트래픽을 라우팅하는 인프라
- **MLOps L13 (ML을 위한 CI/CD)**: A/B 테스팅 게이트를 통합하는 자동화된 배포 파이프라인

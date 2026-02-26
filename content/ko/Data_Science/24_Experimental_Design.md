# 24. 실험 설계 (Experimental Design)

[이전: 비모수 통계](./23_Nonparametric_Statistics.md) | [다음: 실전 프로젝트](./25_Practical_Projects.md)

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 실험 설계의 세 가지 핵심 원리인 무작위화(randomization), 반복(replication), 블로킹(blocking)을 설명한다
2. 참여자 배정을 위한 단순 무작위화(simple randomization)와 층화 무작위화(stratified randomization)를 구현한다
3. 가설 수립부터 의사결정까지 A/B 테스트 워크플로우를 기술한다
4. 두 비율 비교에서 검정력 분석(power analysis)을 사용하여 필요한 표본 크기를 계산한다
5. A/B 테스트 분석에서 빈도주의(frequentist) 접근법과 베이지안(Bayesian) 접근법을 비교한다
6. 피킹(peeking) 문제를 파악하고, 순차적 검정(sequential testing)을 위한 알파 소진 함수(alpha spending function)(O'Brien-Fleming, Pocock)를 적용한다
7. 다중 비교 보정(Bonferroni, Holm-Bonferroni)을 적용하고 그 필요성을 설명한다
8. 신규 효과(novelty effect), 네트워크 효과(network effect), 심슨의 역설(Simpson's paradox)을 포함한 A/B 테스트의 일반적인 함정을 평가한다

---

관찰 데이터는 상관관계를 밝힐 수 있지만, 잘 설계된 실험만이 인과관계를 확립할 수 있습니다. 웹사이트 A/B 테스트, 의학적 처치 평가, 또는 교수법 비교 등 어떤 경우든 실험 설계의 원리가 결론의 신뢰성을 결정합니다. 데이터 수집 전에 설계를 올바르게 하는 것 — 무작위화, 충분한 표본 크기, 적절한 대조군 — 이 사후 분석으로는 수정할 수 없는 잘못된 결론을 막아 줍니다.

---

## 1. 실험 설계의 기본 원리

### 1.1 세 가지 핵심 원리

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm, t

np.random.seed(42)

def experimental_design_principles():
    """실험 설계의 세 가지 핵심 원리"""
    print("""
    =================================================
    실험 설계의 세 가지 핵심 원리
    =================================================

    1. 무작위화 (Randomization)
    ─────────────────────────────
    - 피험자를 처리군에 무작위로 배정
    - 교란변수의 영향을 균등하게 분배
    - 인과관계 추론의 기초

    예시:
    - 동전 던지기로 A/B 그룹 배정
    - 컴퓨터 생성 난수 사용
    - 블록 무작위화 (층화 후 무작위)

    2. 반복 (Replication)
    ─────────────────────────────
    - 충분한 수의 독립적 관측
    - 통계적 검정력 확보
    - 변동성 추정 가능

    고려사항:
    - 표본 크기 계산 (검정력 분석)
    - 비용 대비 효과
    - 실용적 제약

    3. 블로킹 (Blocking)
    ─────────────────────────────
    - 알려진 변동 요인으로 피험자 그룹화
    - 그룹 내에서 무작위 배정
    - 오차 감소, 검정력 향상

    예시:
    - 성별로 블록 → 각 블록 내 무작위 배정
    - 연령대로 층화
    - 지역, 시간대 등

    =================================================
    추가 원리
    =================================================

    - 통제 (Control): 대조군 포함
    - 맹검 (Blinding): 단일/이중 맹검
    - 균형 (Balance): 그룹 간 균등 배정
    """)

experimental_design_principles()
```

### 1.2 무작위화 구현

```python
def randomize_participants(participants, n_groups=2, method='simple', block_var=None):
    """
    피험자 무작위화

    Parameters:
    -----------
    participants : DataFrame
        피험자 정보
    n_groups : int
        그룹 수
    method : str
        'simple' - 단순 무작위
        'stratified' - 층화 무작위
    block_var : str
        층화 변수 (method='stratified'일 때)
    """
    n = len(participants)
    result = participants.copy()

    if method == 'simple':
        # 단순 무작위 배정
        assignments = np.random.choice(range(n_groups), size=n)
        result['group'] = assignments

    elif method == 'stratified' and block_var is not None:
        # 층화 무작위 배정
        result['group'] = -1
        for block_value in participants[block_var].unique():
            mask = participants[block_var] == block_value
            block_n = mask.sum()
            assignments = np.random.choice(range(n_groups), size=block_n)
            result.loc[mask, 'group'] = assignments

    return result

# 예시: 100명의 피험자
np.random.seed(42)
participants = pd.DataFrame({
    'id': range(100),
    'age': np.random.choice(['young', 'middle', 'old'], 100),
    'gender': np.random.choice(['M', 'F'], 100)
})

# 단순 무작위
simple_rand = randomize_participants(participants, n_groups=2, method='simple')

# 층화 무작위 (성별 기준)
stratified_rand = randomize_participants(participants, n_groups=2,
                                          method='stratified', block_var='gender')

print("=== 단순 무작위화 결과 ===")
print(pd.crosstab(simple_rand['gender'], simple_rand['group']))

print("\n=== 층화 무작위화 결과 (성별 기준) ===")
print(pd.crosstab(stratified_rand['gender'], stratified_rand['group']))
```

### 1.3 실험 설계 유형

```python
def experimental_design_types():
    """주요 실험 설계 유형"""
    print("""
    =================================================
    실험 설계 유형
    =================================================

    1. 완전 무작위 설계 (Completely Randomized Design)
       - 가장 단순한 설계
       - 피험자를 처리군에 완전 무작위 배정
       - 분석: 독립표본 t-검정, 일원 ANOVA

    2. 무작위 블록 설계 (Randomized Block Design)
       - 블록 변수로 층화 후 무작위 배정
       - 각 블록 내 모든 처리 수준 포함
       - 분석: 이원 ANOVA (블록 효과 제거)

    3. 요인 설계 (Factorial Design)
       - 여러 요인의 조합 효과 연구
       - 상호작용 효과 검출 가능
       - 분석: 다원 ANOVA

    4. 교차 설계 (Crossover Design)
       - 피험자가 모든 처리를 순차적으로 받음
       - 개인 간 변동 통제
       - 이월 효과 주의

    5. 분할구 설계 (Split-Plot Design)
       - 한 요인은 전체에, 다른 요인은 부분에 적용
       - 농업, 공학에서 흔함
    """)

experimental_design_types()
```

---

## 2. A/B 테스트 이론

### 2.1 A/B 테스트 개요

```python
def ab_test_overview():
    """A/B 테스트 개요"""
    print("""
    =================================================
    A/B 테스트 (A/B Testing)
    =================================================

    정의:
    - 두 가지 버전(A, B)의 효과를 비교하는 무작위 대조 실험
    - 웹/앱에서 가장 널리 사용되는 실험 방법

    용어:
    - Control (A): 기존 버전 (대조군)
    - Treatment (B): 새 버전 (실험군)
    - 전환율 (Conversion Rate): 목표 행동 비율
    - 상승률 (Lift): (B - A) / A

    프로세스:
    1. 가설 수립
    2. 메트릭 정의
    3. 표본 크기 계산
    4. 실험 실행
    5. 통계 분석
    6. 의사결정

    주의사항:
    - 단위의 일관성 (사용자 vs 세션 vs 페이지뷰)
    - 실험 기간 (최소 1-2주, 요일 효과 고려)
    - 다중 비교 보정
    - 네트워크 효과 (spillover)
    """)

ab_test_overview()
```

### 2.2 A/B 테스트 분석

```python
class ABTest:
    """A/B 테스트 분석 클래스"""

    def __init__(self, control_visitors, control_conversions,
                 treatment_visitors, treatment_conversions):
        self.n_c = control_visitors
        self.x_c = control_conversions
        self.n_t = treatment_visitors
        self.x_t = treatment_conversions

        self.p_c = self.x_c / self.n_c
        self.p_t = self.x_t / self.n_t

    def z_test(self, alternative='two-sided'):
        """두 비율의 Z-검정"""
        # 통합 비율
        p_pooled = (self.x_c + self.x_t) / (self.n_c + self.n_t)

        # 표준오차
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/self.n_c + 1/self.n_t))

        # Z 통계량
        z = (self.p_t - self.p_c) / se

        # p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(z)))
        elif alternative == 'greater':  # treatment > control
            p_value = 1 - norm.cdf(z)
        else:  # treatment < control
            p_value = norm.cdf(z)

        return z, p_value

    def confidence_interval(self, alpha=0.05):
        """차이의 신뢰구간"""
        diff = self.p_t - self.p_c

        # 각 비율의 분산
        var_c = self.p_c * (1 - self.p_c) / self.n_c
        var_t = self.p_t * (1 - self.p_t) / self.n_t
        se = np.sqrt(var_c + var_t)

        z_crit = norm.ppf(1 - alpha/2)
        ci_lower = diff - z_crit * se
        ci_upper = diff + z_crit * se

        return diff, (ci_lower, ci_upper)

    def lift(self):
        """상승률 계산"""
        if self.p_c == 0:
            return np.inf
        return (self.p_t - self.p_c) / self.p_c

    def summary(self):
        """결과 요약"""
        print("=== A/B Test Summary ===")
        print(f"\nControl:   {self.x_c:,}/{self.n_c:,} = {self.p_c:.4f} ({self.p_c*100:.2f}%)")
        print(f"Treatment: {self.x_t:,}/{self.n_t:,} = {self.p_t:.4f} ({self.p_t*100:.2f}%)")

        z, p_value = self.z_test()
        diff, ci = self.confidence_interval()
        lift = self.lift()

        print(f"\n차이: {diff:.4f} ({diff*100:.2f}%p)")
        print(f"상승률: {lift*100:.2f}%")
        print(f"95% CI: ({ci[0]*100:.2f}%p, {ci[1]*100:.2f}%p)")
        print(f"\nZ 통계량: {z:.3f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("\n결론: 통계적으로 유의한 차이 있음 (p < 0.05)")
            if diff > 0:
                print("Treatment가 Control보다 유의하게 높음")
            else:
                print("Treatment가 Control보다 유의하게 낮음")
        else:
            print("\n결론: 통계적으로 유의한 차이 없음 (p >= 0.05)")


# 예시: 버튼 색상 A/B 테스트
ab_test = ABTest(
    control_visitors=10000,
    control_conversions=350,
    treatment_visitors=10000,
    treatment_conversions=420
)
ab_test.summary()

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 전환율 비교
ax = axes[0]
bars = ax.bar(['Control', 'Treatment'], [ab_test.p_c, ab_test.p_t], alpha=0.7)
ax.set_ylabel('전환율')
ax.set_title('A/B 테스트: 전환율 비교')

# 에러바 추가
se_c = np.sqrt(ab_test.p_c * (1 - ab_test.p_c) / ab_test.n_c)
se_t = np.sqrt(ab_test.p_t * (1 - ab_test.p_t) / ab_test.n_t)
ax.errorbar(['Control', 'Treatment'], [ab_test.p_c, ab_test.p_t],
            yerr=[1.96*se_c, 1.96*se_t], fmt='none', color='black', capsize=5)
ax.grid(True, alpha=0.3, axis='y')

# 차이의 신뢰구간
ax = axes[1]
diff, ci = ab_test.confidence_interval()
ax.errorbar([0], [diff], yerr=[[diff - ci[0]], [ci[1] - diff]],
            fmt='o', markersize=10, capsize=10, capthick=2)
ax.axhline(0, color='r', linestyle='--', label='차이 없음')
ax.set_xlim(-1, 1)
ax.set_ylabel('전환율 차이')
ax.set_title(f'차이의 95% 신뢰구간\n({ci[0]:.4f}, {ci[1]:.4f})')
ax.set_xticks([])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.3 베이지안 A/B 테스트

```python
def bayesian_ab_test(n_c, x_c, n_t, x_t, alpha_prior=1, beta_prior=1, n_samples=100000):
    """
    베이지안 A/B 테스트

    Beta 사전분포를 사용한 전환율 추정
    """
    # 사후분포 (Beta-Binomial conjugate)
    alpha_c = alpha_prior + x_c
    beta_c = beta_prior + n_c - x_c
    alpha_t = alpha_prior + x_t
    beta_t = beta_prior + n_t - x_t

    # 사후분포에서 샘플링
    samples_c = np.random.beta(alpha_c, beta_c, n_samples)
    samples_t = np.random.beta(alpha_t, beta_t, n_samples)

    # P(Treatment > Control)
    prob_t_better = np.mean(samples_t > samples_c)

    # 기대 상승률
    lift_samples = (samples_t - samples_c) / samples_c
    expected_lift = np.mean(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])

    print("=== 베이지안 A/B 테스트 ===")
    print(f"\nP(Treatment > Control): {prob_t_better:.4f} ({prob_t_better*100:.1f}%)")
    print(f"기대 상승률: {expected_lift*100:.2f}%")
    print(f"상승률 95% CI: ({lift_ci[0]*100:.2f}%, {lift_ci[1]*100:.2f}%)")

    # 의사결정 기준
    print("\n의사결정:")
    if prob_t_better > 0.95:
        print("  → Treatment 채택 권장 (P > 95%)")
    elif prob_t_better < 0.05:
        print("  → Control 유지 권장 (P < 5%)")
    else:
        print("  → 추가 데이터 수집 필요")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 사후분포 비교
    ax = axes[0]
    x_range = np.linspace(0, 0.1, 200)
    ax.plot(x_range, stats.beta(alpha_c, beta_c).pdf(x_range), label='Control')
    ax.plot(x_range, stats.beta(alpha_t, beta_t).pdf(x_range), label='Treatment')
    ax.fill_between(x_range, stats.beta(alpha_c, beta_c).pdf(x_range), alpha=0.3)
    ax.fill_between(x_range, stats.beta(alpha_t, beta_t).pdf(x_range), alpha=0.3)
    ax.set_xlabel('전환율')
    ax.set_ylabel('밀도')
    ax.set_title('전환율 사후분포')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 차이 분포
    ax = axes[1]
    diff_samples = samples_t - samples_c
    ax.hist(diff_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='차이 없음')
    ax.axvline(np.mean(diff_samples), color='g', linestyle='-',
               label=f'평균: {np.mean(diff_samples):.4f}')
    ax.set_xlabel('전환율 차이 (T - C)')
    ax.set_ylabel('밀도')
    ax.set_title(f'차이 사후분포\nP(T>C)={prob_t_better:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 상승률 분포
    ax = axes[2]
    lift_samples_clipped = np.clip(lift_samples, -1, 2)
    ax.hist(lift_samples_clipped, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='0%')
    ax.axvline(expected_lift, color='g', linestyle='-',
               label=f'기대값: {expected_lift*100:.1f}%')
    ax.set_xlabel('상승률')
    ax.set_ylabel('밀도')
    ax.set_title('상승률 사후분포')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return prob_t_better, expected_lift

# 베이지안 분석
prob_better, exp_lift = bayesian_ab_test(10000, 350, 10000, 420)
```

---

## 3. 표본 크기 결정 (검정력 분석)

### 3.1 검정력 분석 개념

```python
def power_analysis_concepts():
    """검정력 분석 핵심 개념"""
    print("""
    =================================================
    검정력 분석 (Power Analysis)
    =================================================

    네 가지 요소 (하나를 다른 셋으로부터 계산):
    ─────────────────────────────────────────────────
    1. 효과 크기 (Effect Size)
       - 탐지하고자 하는 최소 효과
       - 예: 전환율 차이 0.02 (2%p)

    2. 유의수준 α (Significance Level)
       - 제1종 오류 확률
       - 일반적으로 0.05

    3. 검정력 1-β (Power)
       - 효과가 있을 때 탐지할 확률
       - 일반적으로 0.80 (최소) ~ 0.90

    4. 표본 크기 n (Sample Size)
       - 필요한 관측 수

    계산 흐름:
    ─────────────────────────────────────────────────
    효과 크기 + α + (1-β) → n (사전 설계)
    n + α + (1-β) → 최소 탐지 가능 효과 (민감도 분석)
    n + α + 효과 크기 → 달성 검정력 (사후 분석)

    경험 법칙:
    ─────────────────────────────────────────────────
    - 검정력 80% 미만: 과소 검정력
    - 검정력 80-90%: 일반적 권장
    - 검정력 90% 이상: 고검정력 연구
    """)

power_analysis_concepts()
```

### 3.2 두 비율 비교의 표본 크기

```python
def sample_size_two_proportions(p1, p2, alpha=0.05, power=0.80, ratio=1):
    """
    두 비율 비교를 위한 표본 크기 계산

    Parameters:
    -----------
    p1 : float
        Control 전환율 (기준)
    p2 : float
        Treatment 전환율 (목표)
    alpha : float
        유의수준
    power : float
        검정력
    ratio : float
        n2/n1 비율 (기본값 1 = 동일 크기)

    Returns:
    --------
    n1, n2 : int
        각 그룹의 필요 표본 크기
    """
    # 효과 크기
    effect = abs(p2 - p1)
    p_pooled = (p1 + ratio * p2) / (1 + ratio)

    # Z 값
    z_alpha = norm.ppf(1 - alpha/2)  # 양측
    z_beta = norm.ppf(power)

    # 표본 크기 공식
    numerator = (z_alpha * np.sqrt((1 + ratio) * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + ratio * p2 * (1 - p2)))**2
    n1 = numerator / (effect**2 * ratio)
    n2 = n1 * ratio

    return int(np.ceil(n1)), int(np.ceil(n2))


def plot_sample_size_analysis(p1_base, effects, alpha=0.05, power=0.80):
    """효과 크기에 따른 필요 표본 크기"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 효과 크기 vs 표본 크기
    ax = axes[0]
    sample_sizes = []
    for effect in effects:
        p2 = p1_base + effect
        n1, _ = sample_size_two_proportions(p1_base, p2, alpha, power)
        sample_sizes.append(n1)

    ax.plot(np.array(effects)*100, sample_sizes, 'bo-', linewidth=2)
    ax.set_xlabel('효과 크기 (전환율 차이 %p)')
    ax.set_ylabel('그룹당 필요 표본 크기')
    ax.set_title(f'효과 크기 vs 표본 크기\n(기준 전환율={p1_base:.1%}, α={alpha}, power={power})')
    ax.grid(True, alpha=0.3)

    # 로그 스케일
    ax.set_yscale('log')
    for i, (eff, n) in enumerate(zip(effects, sample_sizes)):
        ax.annotate(f'{n:,}', (eff*100, n), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # 검정력 vs 표본 크기
    ax = axes[1]
    effect_fixed = 0.02  # 2%p 고정
    p2_fixed = p1_base + effect_fixed
    powers = np.linspace(0.5, 0.95, 10)
    sample_sizes_power = []

    for pwr in powers:
        n1, _ = sample_size_two_proportions(p1_base, p2_fixed, alpha, pwr)
        sample_sizes_power.append(n1)

    ax.plot(powers*100, sample_sizes_power, 'go-', linewidth=2)
    ax.set_xlabel('검정력 (%)')
    ax.set_ylabel('그룹당 필요 표본 크기')
    ax.set_title(f'검정력 vs 표본 크기\n(효과 크기={effect_fixed:.1%}p)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 예시
p1 = 0.05  # 기준 전환율 5%
effects = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5%p ~ 3%p

print("=== 표본 크기 계산 ===")
print(f"기준 전환율: {p1:.1%}")
print(f"α = 0.05, Power = 0.80")
print()
for effect in effects:
    p2 = p1 + effect
    n1, n2 = sample_size_two_proportions(p1, p2)
    print(f"효과 {effect*100:.1f}%p (상대 {effect/p1*100:.0f}%): n1={n1:,}, n2={n2:,}, 총={n1+n2:,}")

plot_sample_size_analysis(p1, effects)
```

### 3.3 statsmodels 검정력 분석

```python
from statsmodels.stats.power import TTestPower, NormalIndPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

def statsmodels_power_analysis():
    """statsmodels를 사용한 검정력 분석"""

    # 1. t-검정 검정력 분석
    print("=== t-검정 검정력 분석 ===")

    # 효과 크기 계산 (Cohen's d)
    # d = (μ1 - μ2) / σ
    mean_diff = 5
    std = 15
    d = mean_diff / std
    print(f"Cohen's d = {d:.3f}")

    # 필요 표본 크기
    power_analysis = TTestPower()
    n = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.80,
                                     alternative='two-sided')
    print(f"필요 표본 크기 (각 그룹): {int(np.ceil(n))}")

    # 달성 검정력
    achieved_power = power_analysis.power(effect_size=d, nobs=100, alpha=0.05,
                                           alternative='two-sided')
    print(f"n=100일 때 검정력: {achieved_power:.3f}")

    # 2. 비율 검정 검정력 분석
    print("\n=== 비율 검정 검정력 분석 ===")

    p1 = 0.05
    p2 = 0.07
    effect = proportion_effectsize(p1, p2)
    print(f"효과 크기 (h): {effect:.3f}")

    # 필요 표본 크기
    power_prop = NormalIndPower()
    n_prop = power_prop.solve_power(effect_size=effect, alpha=0.05, power=0.80,
                                      alternative='two-sided', ratio=1)
    print(f"필요 표본 크기 (각 그룹): {int(np.ceil(n_prop))}")

    return n, n_prop

n_t, n_prop = statsmodels_power_analysis()
```

### 3.4 검정력 곡선

```python
def plot_power_curve(effect_sizes, n_per_group, alpha=0.05):
    """검정력 곡선 시각화"""

    power_analysis = NormalIndPower()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 효과 크기 vs 검정력 (n 고정)
    ax = axes[0]
    for n in n_per_group:
        powers = [power_analysis.power(effect_size=es, nobs=n, alpha=alpha,
                                        alternative='two-sided', ratio=1)
                  for es in effect_sizes]
        ax.plot(effect_sizes, powers, '-o', label=f'n={n}')

    ax.axhline(0.80, color='r', linestyle='--', alpha=0.5, label='Power=0.80')
    ax.set_xlabel('효과 크기 (Cohen\'s h)')
    ax.set_ylabel('검정력')
    ax.set_title('검정력 곡선 (표본 크기별)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 표본 크기 vs 검정력 (효과 크기 고정)
    ax = axes[1]
    n_range = np.arange(50, 1001, 50)
    effect_fixed = [0.1, 0.2, 0.3, 0.5]

    for es in effect_fixed:
        powers = [power_analysis.power(effect_size=es, nobs=n, alpha=alpha,
                                        alternative='two-sided', ratio=1)
                  for n in n_range]
        ax.plot(n_range, powers, '-', label=f'h={es}')

    ax.axhline(0.80, color='r', linestyle='--', alpha=0.5, label='Power=0.80')
    ax.set_xlabel('그룹당 표본 크기')
    ax.set_ylabel('검정력')
    ax.set_title('검정력 곡선 (효과 크기별)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

effect_sizes = np.linspace(0.05, 0.5, 20)
n_per_group = [50, 100, 200, 500]
plot_power_curve(effect_sizes, n_per_group)
```

---

## 4. 순차적 검정 (Sequential Testing)

### 4.1 왜 순차적 검정인가?

```python
def sequential_testing_motivation():
    """순차적 검정의 필요성"""
    print("""
    =================================================
    순차적 검정 (Sequential Testing)
    =================================================

    문제: Peeking Problem
    ─────────────────────────────────────────────────
    - A/B 테스트 중간에 결과를 확인하면 제1종 오류율 증가
    - α=0.05로 설계해도 5번 중간 확인 시 실제 오류율 ~14%

    예시:
    - 1회 확인: α = 0.05
    - 5회 확인: α ≈ 0.14
    - 10회 확인: α ≈ 0.19

    해결책:
    ─────────────────────────────────────────────────
    1. 고정 표본 검정: 미리 정한 n까지 기다림
    2. 순차적 검정: 중간 확인을 허용하되 보정
       - O'Brien-Fleming
       - Pocock
       - Alpha spending functions

    장점:
    - 효과가 명확하면 조기 종료 → 비용/시간 절약
    - 효과가 없으면 빠른 종료
    - 통계적 타당성 유지
    """)

sequential_testing_motivation()
```

### 4.2 Peeking 문제 시뮬레이션

```python
def simulate_peeking_problem(n_simulations=10000, n_total=1000, n_looks=5):
    """
    Peeking 문제 시뮬레이션:
    귀무가설이 참일 때 (실제 차이 없음) 얼마나 자주 유의하게 나오는가
    """
    np.random.seed(42)
    alpha = 0.05

    # 중간 확인 시점
    look_points = np.linspace(n_total // n_looks, n_total, n_looks).astype(int)

    false_positives_fixed = 0  # 고정 표본 (마지막만 확인)
    false_positives_peeking = 0  # 모든 시점 확인

    for _ in range(n_simulations):
        # 귀무가설 하에서 데이터 생성 (두 그룹 동일)
        control = np.random.binomial(1, 0.1, n_total)
        treatment = np.random.binomial(1, 0.1, n_total)

        # Peeking: 각 시점에서 검정
        for look in look_points:
            x_c = control[:look].sum()
            x_t = treatment[:look].sum()
            n = look

            # 비율
            p_c = x_c / n
            p_t = x_t / n
            p_pooled = (x_c + x_t) / (2 * n)

            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * 2 / n)
                z = (p_t - p_c) / se if se > 0 else 0
                p_value = 2 * (1 - norm.cdf(abs(z)))

                if p_value < alpha:
                    false_positives_peeking += 1
                    break  # 한 번이라도 유의하면 종료

        # 고정 표본: 마지막만 확인
        x_c = control.sum()
        x_t = treatment.sum()
        p_c = x_c / n_total
        p_t = x_t / n_total
        p_pooled = (x_c + x_t) / (2 * n_total)

        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * 2 / n_total)
            z = (p_t - p_c) / se if se > 0 else 0
            p_value = 2 * (1 - norm.cdf(abs(z)))

            if p_value < alpha:
                false_positives_fixed += 1

    fpr_fixed = false_positives_fixed / n_simulations
    fpr_peeking = false_positives_peeking / n_simulations

    print("=== Peeking 문제 시뮬레이션 ===")
    print(f"시뮬레이션 횟수: {n_simulations:,}")
    print(f"총 표본 크기: {n_total}")
    print(f"중간 확인 횟수: {n_looks}")
    print(f"목표 α: {alpha}")
    print(f"\n고정 표본 검정 위양성률: {fpr_fixed:.4f} ({fpr_fixed*100:.2f}%)")
    print(f"Peeking 검정 위양성률: {fpr_peeking:.4f} ({fpr_peeking*100:.2f}%)")
    print(f"위양성률 증가: {(fpr_peeking/alpha - 1)*100:.1f}%")

    return fpr_fixed, fpr_peeking

fpr_fixed, fpr_peeking = simulate_peeking_problem()
```

### 4.3 Alpha Spending 함수

```python
def alpha_spending_pocock(t, alpha=0.05):
    """Pocock alpha spending function"""
    return alpha * np.log(1 + (np.e - 1) * t)

def alpha_spending_obrien_fleming(t, alpha=0.05):
    """O'Brien-Fleming alpha spending function"""
    return 2 * (1 - norm.cdf(norm.ppf(1 - alpha/2) / np.sqrt(t)))

def plot_alpha_spending():
    """Alpha spending 함수 시각화"""
    t = np.linspace(0.01, 1, 100)
    alpha = 0.05

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, alpha_spending_pocock(t, alpha), label='Pocock', linewidth=2)
    ax.plot(t, alpha_spending_obrien_fleming(t, alpha), label="O'Brien-Fleming", linewidth=2)
    ax.plot(t, t * alpha, '--', label='Linear (reference)', alpha=0.5)
    ax.axhline(alpha, color='r', linestyle=':', label=f'Total α={alpha}')

    ax.set_xlabel('정보 비율 (현재/최종)')
    ax.set_ylabel('누적 α spent')
    ax.set_title('Alpha Spending Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, alpha * 1.1)

    plt.show()

    print("=== Alpha Spending 함수 비교 ===")
    print("\nPocock:")
    print("  - 각 분석에서 동일한 임계값")
    print("  - 조기 종료에 더 관대")
    print("  - 최종 분석에서 더 보수적")

    print("\nO'Brien-Fleming:")
    print("  - 초기에 매우 보수적 (높은 임계값)")
    print("  - 후기에 고정 표본과 유사")
    print("  - 조기 종료는 극단적 효과에서만")

plot_alpha_spending()
```

### 4.4 순차적 검정 구현

```python
class SequentialTest:
    """순차적 A/B 테스트"""

    def __init__(self, max_n, n_looks, alpha=0.05, spending='obrien_fleming'):
        """
        Parameters:
        -----------
        max_n : int
            최대 표본 크기 (각 그룹)
        n_looks : int
            중간 분석 횟수
        alpha : float
            전체 유의수준
        spending : str
            'pocock' or 'obrien_fleming'
        """
        self.max_n = max_n
        self.n_looks = n_looks
        self.alpha = alpha
        self.spending = spending

        # 분석 시점
        self.look_times = np.linspace(1/n_looks, 1, n_looks)

        # 각 분석에서 사용할 alpha
        self.alphas = self._compute_alphas()

    def _compute_alphas(self):
        """각 분석 시점의 alpha 계산"""
        if self.spending == 'pocock':
            cumulative = [alpha_spending_pocock(t, self.alpha) for t in self.look_times]
        else:
            cumulative = [alpha_spending_obrien_fleming(t, self.alpha) for t in self.look_times]

        # 증분 alpha
        alphas = [cumulative[0]]
        for i in range(1, len(cumulative)):
            alphas.append(cumulative[i] - cumulative[i-1])

        return alphas

    def critical_values(self):
        """각 분석의 임계 Z 값"""
        return [norm.ppf(1 - a/2) for a in self.alphas]

    def summary(self):
        """분석 계획 요약"""
        print("=== 순차적 검정 계획 ===")
        print(f"최대 표본: {self.max_n} (각 그룹)")
        print(f"중간 분석: {self.n_looks}회")
        print(f"전체 α: {self.alpha}")
        print(f"Spending: {self.spending}")

        print("\n분석 시점별 계획:")
        print("-" * 50)
        print(f"{'분석':<6} {'n':<10} {'누적 α':<12} {'증분 α':<12} {'Z 임계값':<10}")
        print("-" * 50)

        cumulative_alpha = 0
        z_crits = self.critical_values()

        for i, (t, a) in enumerate(zip(self.look_times, self.alphas)):
            n = int(t * self.max_n)
            cumulative_alpha += a
            print(f"{i+1:<6} {n:<10} {cumulative_alpha:<12.4f} {a:<12.4f} {z_crits[i]:<10.3f}")


# 예시
seq_test = SequentialTest(max_n=5000, n_looks=5, alpha=0.05, spending='obrien_fleming')
seq_test.summary()

print("\n")

seq_test_pocock = SequentialTest(max_n=5000, n_looks=5, alpha=0.05, spending='pocock')
seq_test_pocock.summary()
```

---

## 5. 일반적인 함정과 주의사항

### 5.1 다중 비교 문제

```python
def multiple_comparisons_problem():
    """다중 비교 문제"""
    print("""
    =================================================
    다중 비교 문제 (Multiple Comparisons)
    =================================================

    문제:
    - 여러 검정을 동시에 수행하면 제1종 오류 증가
    - k개 검정 시 최소 하나 위양성 확률: 1 - (1-α)^k

    예시 (α=0.05):
    - 1개 검정: 5%
    - 5개 검정: 23%
    - 10개 검정: 40%
    - 20개 검정: 64%

    보정 방법:
    ─────────────────────────────────────────────────
    1. Bonferroni: α' = α/k (가장 보수적)
    2. Holm-Bonferroni: 순차적 Bonferroni
    3. Benjamini-Hochberg (FDR): 위발견률 통제
    4. 사전 등록: 주요 가설 1개 지정
    """)

    # 시각화
    k_values = range(1, 21)
    alpha = 0.05

    fwer = [1 - (1 - alpha)**k for k in k_values]
    bonferroni = [min(alpha * k, 1.0) for k in k_values]  # 보정 전 허용 범위

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, fwer, 'b-o', label='보정 없음 (FWER)')
    ax.axhline(alpha, color='r', linestyle='--', label=f'목표 α={alpha}')
    ax.set_xlabel('검정 횟수')
    ax.set_ylabel('최소 1개 위양성 확률')
    ax.set_title('다중 비교 문제: 검정 횟수와 위양성률')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.7)

    plt.show()

multiple_comparisons_problem()
```

### 5.2 기타 주의사항

```python
def common_pitfalls():
    """A/B 테스트의 일반적인 함정"""
    print("""
    =================================================
    A/B 테스트 주의사항
    =================================================

    1. Peeking (중간 확인)
       - 문제: 원하는 결과가 나올 때까지 확인
       - 해결: 순차적 검정 또는 고정 표본

    2. 다중 비교
       - 문제: 여러 메트릭/세그먼트 테스트
       - 해결: 사전 등록, 보정, 주요 메트릭 지정

    3. 부적절한 표본 크기
       - 문제: 너무 작으면 효과 탐지 실패, 너무 크면 낭비
       - 해결: 사전 검정력 분석

    4. 신규 효과 (Novelty Effect)
       - 문제: 새로움 자체가 일시적 효과 유발
       - 해결: 충분한 실험 기간

    5. 네트워크 효과 (Spillover)
       - 문제: 그룹 간 상호작용
       - 해결: 클러스터 무작위화

    6. Simpson's Paradox
       - 문제: 전체 vs 세그먼트별 결과 상충
       - 해결: 층화 분석, 인과 그래프

    7. 실제적 유의성 무시
       - 문제: 통계적 유의성 ≠ 실제적 중요성
       - 해결: 효과 크기, 신뢰구간, 비즈니스 영향 고려

    8. 검정력 부족
       - 문제: "효과 없음" ≠ "귀무가설 참"
       - 해결: 검정력 보고, 동등성 검정
    """)

common_pitfalls()
```

---

## 6. 실습 예제

### 6.1 종합 실험 설계

```python
def complete_ab_test_workflow():
    """A/B 테스트 전체 워크플로우"""

    print("="*60)
    print("A/B 테스트 워크플로우")
    print("="*60)

    # 1. 가설 수립
    print("\n[1단계] 가설 수립")
    print("  H0: 새 버튼 색상은 전환율에 영향 없음")
    print("  H1: 새 버튼 색상은 전환율을 변화시킴")

    # 2. 메트릭 정의
    print("\n[2단계] 메트릭 정의")
    baseline_rate = 0.05  # 5%
    mde = 0.01  # 최소 탐지 효과: 1%p
    print(f"  기준 전환율: {baseline_rate:.1%}")
    print(f"  MDE (Minimum Detectable Effect): {mde:.1%}p")

    # 3. 표본 크기 계산
    print("\n[3단계] 표본 크기 계산")
    target_rate = baseline_rate + mde
    n1, n2 = sample_size_two_proportions(baseline_rate, target_rate, alpha=0.05, power=0.80)
    print(f"  필요 표본 크기: {n1:,} (각 그룹)")
    print(f"  총 필요 트래픽: {n1 + n2:,}")

    # 4. 실험 실행 (시뮬레이션)
    print("\n[4단계] 실험 실행 (시뮬레이션)")
    np.random.seed(42)
    n_control = n1
    n_treatment = n2
    x_control = np.random.binomial(n_control, baseline_rate)
    x_treatment = np.random.binomial(n_treatment, baseline_rate + mde * 0.8)  # 실제 효과는 MDE의 80%

    print(f"  Control: {x_control:,}/{n_control:,} = {x_control/n_control:.2%}")
    print(f"  Treatment: {x_treatment:,}/{n_treatment:,} = {x_treatment/n_treatment:.2%}")

    # 5. 분석
    print("\n[5단계] 분석")
    ab = ABTest(n_control, x_control, n_treatment, x_treatment)
    z, p = ab.z_test()
    diff, ci = ab.confidence_interval()
    lift = ab.lift()

    print(f"  차이: {diff:.4f} ({diff*100:.2f}%p)")
    print(f"  상승률: {lift*100:.2f}%")
    print(f"  95% CI: ({ci[0]*100:.2f}%p, {ci[1]*100:.2f}%p)")
    print(f"  Z 통계량: {z:.3f}")
    print(f"  p-value: {p:.4f}")

    # 6. 의사결정
    print("\n[6단계] 의사결정")
    if p < 0.05:
        if diff > 0:
            print("  결론: Treatment 채택 (통계적으로 유의한 개선)")
        else:
            print("  결론: Control 유지 (Treatment가 더 나쁨)")
    else:
        print("  결론: 결정 보류 (유의한 차이 없음)")
        print("  고려사항: 표본 크기 증가 또는 다른 변형 테스트")

    return ab

ab_result = complete_ab_test_workflow()
```

---

## 7. 연습 문제

### 문제 1: 표본 크기 계산
기존 전환율이 3%이고, 최소 20%의 상대적 상승(3.6%로)을 탐지하고 싶다면:
1. α=0.05, Power=0.80에서 필요한 표본 크기
2. Power=0.90으로 높이면 표본 크기 변화
3. MDE를 10% 상승으로 낮추면 표본 크기 변화

### 문제 2: 실험 기간 추정
일일 트래픽이 10,000 방문이고, 50:50으로 분할한다면:
1. 문제 1의 표본 크기를 확보하는데 필요한 기간
2. 주말 효과를 고려하면 최소 몇 주 실험?

### 문제 3: 순차적 검정 설계
5회 중간 분석을 계획한다면:
1. O'Brien-Fleming 방법의 각 분석 임계값
2. Pocock 방법과 비교
3. 첫 번째 분석에서 조기 종료 조건

### 문제 4: 다중 비교 보정
5개 세그먼트(연령대)에서 A/B 테스트 결과를 분석한다면:
1. Bonferroni 보정된 유의수준
2. 하나의 세그먼트에서 p=0.02가 나왔을 때 결론
3. 사전 등록했다면 어떻게 다를까?

---

## 8. 핵심 요약

### 실험 설계 체크리스트

1. [ ] 명확한 가설과 메트릭 정의
2. [ ] 검정력 분석으로 표본 크기 결정
3. [ ] 무작위화 방법 선택
4. [ ] 실험 기간 설정 (주간 효과 고려)
5. [ ] 중간 분석 계획 (순차적 검정)
6. [ ] 다중 비교 고려
7. [ ] 사전 등록

### 표본 크기 공식 (비율 비교)

$$n = \frac{(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)})^2}{(p_1-p_2)^2}$$

### 검정력 관계

| 요인 | 증가 시 표본 크기 |
|------|------------------|
| 효과 크기 ↑ | 감소 |
| 검정력 ↑ | 증가 |
| α ↓ (더 엄격) | 증가 |
| 분산 ↑ | 증가 |

### 순차적 검정

| 방법 | 초기 | 후기 |
|------|------|------|
| O'Brien-Fleming | 매우 보수적 | 고정 표본 유사 |
| Pocock | 일정 | 더 보수적 |

### Python 라이브러리

```python
from statsmodels.stats.power import TTestPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import norm

# 표본 크기 계산
power_analysis = NormalIndPower()
n = power_analysis.solve_power(effect_size=h, alpha=0.05, power=0.80)
```

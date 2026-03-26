# 가설 검정

**이전**: [구간 추정](./13_Interval_Estimation.md) | **다음**: [베이즈 추론](./15_Bayesian_Inference.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 일반적인 검정 상황에서 귀무가설과 대립가설을 수립하기
2. 제1종 오류, 제2종 오류, 검정력을 구별하기
3. 검정 통계량을 구성하고 기각 영역을 결정하기
4. 최강력 검정에 대한 네이만-피어슨 보조정리를 진술하고 적용하기
5. p-값을 계산하고 올바르게 해석하기
6. z-검정, t-검정, 카이제곱 검정 수행하기
7. 가능도비 검정과 윌크스 정리 적용하기
8. 본페로니 보정과 위발견률 보정을 사용하여 다중 검정 문제 다루기
9. 통계적 유의성과 실질적 유의성을 구별하기

---

가설 검정 (Hypothesis Testing)은 표본 데이터를 기반으로 모집단 모수에 대한 결정을 내리기 위한 공식적인 프레임워크를 제공합니다. 이는 과학적 추론, 임상 시험, A/B 테스트, 품질 관리의 근간입니다.

---

## 1. 귀무가설과 대립가설

### 1.1 프레임워크

- **귀무가설 (Null Hypothesis)** $H_0$: "효과 없음" 또는 "차이 없음"의 진술. 현상 유지를 나타냅니다.
- **대립가설 (Alternative Hypothesis)** $H_1$ (또는 $H_a$): 증거를 찾고자 하는 주장.

**예시**:
- 약물 검정: $H_0: \mu_{\text{drug}} = \mu_{\text{placebo}}$ vs. $H_1: \mu_{\text{drug}} \neq \mu_{\text{placebo}}$
- 품질 관리: $H_0: p \leq 0.02$ vs. $H_1: p > 0.02$

### 1.2 검정의 종류

| 검정 유형 | $H_0$ | $H_1$ |
|-----------|-------|-------|
| 양측 검정 | $\theta = \theta_0$ | $\theta \neq \theta_0$ |
| 우측 검정 | $\theta \leq \theta_0$ | $\theta > \theta_0$ |
| 좌측 검정 | $\theta \geq \theta_0$ | $\theta < \theta_0$ |

---

## 2. 오류와 검정력

### 2.1 제1종 오류와 제2종 오류

|  | $H_0$ 참 | $H_0$ 거짓 |
|---|---|---|
| $H_0$ 기각 | **제1종 오류** ($\alpha$) | 올바름 (검정력) |
| $H_0$ 기각 못함 | 올바름 | **제2종 오류** ($\beta$) |

- **유의 수준** $\alpha = P(\text{reject } H_0 \mid H_0 \text{ true})$: 보통 0.05, 0.01, 또는 0.10으로 설정합니다.
- **제2종 오류** $\beta = P(\text{fail to reject } H_0 \mid H_1 \text{ true})$
- **검정력 (Power)** $= 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true})$

### 2.2 검정력 분석

검정력은 다음에 의존합니다:
1. **효과 크기 (Effect Size)**: 큰 참 효과는 탐지하기 쉽습니다.
2. **표본 크기** $n$: 더 많은 데이터는 더 큰 검정력을 줍니다.
3. **유의 수준** $\alpha$: 큰 $\alpha$는 더 큰 검정력을 주지만 제1종 오류가 더 많아집니다.
4. **변동성** $\sigma$: 잡음이 적을수록 검정력이 높아집니다.

```python
import math

def power_one_sample_z(mu_0, mu_1, sigma, n, alpha=0.05):
    """Compute power for a one-sample z-test (two-sided).

    H0: mu = mu_0 vs H1: mu = mu_1
    """
    z_alpha = 1.96 if alpha == 0.05 else 1.645  # two-sided for 0.05
    se = sigma / math.sqrt(n)
    # Non-centrality: how far mu_1 is from mu_0 in SE units
    delta = abs(mu_1 - mu_0) / se
    # Power ≈ P(Z > z_alpha - delta) + P(Z < -z_alpha - delta)
    # Using the approximation Phi(x) ≈ via the logistic function
    def phi(x):
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    power = 1 - phi(z_alpha - delta) + phi(-z_alpha - delta)
    return power

# Example: detect a shift from mu_0=100 to mu_1=105, sigma=15
for n in [10, 25, 50, 100, 200]:
    pwr = power_one_sample_z(100, 105, sigma=15, n=n)
    print(f"n={n:>3d}: power = {pwr:.4f}")
```

---

## 3. 검정 통계량과 기각 영역

### 3.1 일반 절차

1. $H_0$와 $H_1$을 진술합니다.
2. 유의 수준 $\alpha$를 선택합니다.
3. 표본에서 **검정 통계량** $T(X)$를 계산합니다.
4. **기각 영역 (Rejection Region)** $\mathcal{R}$을 결정합니다: $H_0$를 기각하게 되는 $T$ 값의 집합.
5. $T(X) \in \mathcal{R}$이면 $H_0$를 기각하고, 그렇지 않으면 기각하지 못합니다.

### 3.2 예시: 일표본 z-검정

$H_0: \mu = \mu_0$ vs. $H_1: \mu \neq \mu_0$ (양측), $\sigma$ 기지.

$$T = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}$$

기각 영역: $|T| > z_{\alpha/2}$.

---

## 4. 네이만-피어슨 보조정리

### 4.1 진술

단순가설 $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$을 검정할 때, 크기 $\alpha$의 **최강력 검정 (Most Powerful Test)** 의 기각 영역은:

$$\mathcal{R} = \left\{ x : \frac{L(\theta_1 \mid x)}{L(\theta_0 \mid x)} > k \right\}$$

여기서 $k$는 $P(X \in \mathcal{R} \mid H_0) = \alpha$가 되도록 선택됩니다.

### 4.2 해석

가능도비 검정 (Likelihood Ratio Test)은 같은 크기의 다른 검정보다 $\theta_1$에 대해 더 높은 검정력을 가진다는 의미에서 **최적**입니다. 이 기초적 결과가 가능도 기반 검정 통계량의 광범위한 사용을 정당화합니다.

### 4.3 예시

$X_1, \ldots, X_n \sim N(\mu, 1)$에 대해 $H_0: \mu = 0$ vs. $H_1: \mu = 1$ 검정:

$$\frac{L(1 \mid x)}{L(0 \mid x)} = \exp\left(n\bar{x} - \frac{n}{2}\right) > k$$

이는 어떤 상수 $c$에 대해 $\bar{x} > c$일 때 기각하는 것으로 귀결되며, 이는 정확히 단측 z-검정입니다.

---

## 5. p-값

### 5.1 정의

**p-값 (p-value)** 은 $H_0$ 하에서 실제 관측된 것만큼 극단적이거나 더 극단적인 검정 통계량을 관측할 확률입니다:

$$p = P(T \geq T_{\text{obs}} \mid H_0) \quad \text{(단측)}$$

$$p = P(|T| \geq |T_{\text{obs}}| \mid H_0) \quad \text{(양측)}$$

### 5.2 해석

- 작은 p-값은 관측된 데이터가 $H_0$ 하에서 가능성이 낮다는 것을 나타냅니다.
- $p \leq \alpha$이면 $H_0$를 기각합니다.
- p-값은 $H_0$가 참일 확률이 **아닙니다**.
- p-값은 오류를 범할 확률이 **아닙니다**.

### 5.3 피해야 할 흔한 오해

1. "$p = 0.03$은 $H_0$가 참일 확률이 3%라는 뜻이다." -- **틀림.**
2. "$1 - p$는 결과가 재현될 확률이다." -- **틀림.**
3. "p-값이 작을수록 효과가 크다." -- **틀림.** p-값은 효과 크기와 표본 크기 모두에 의존합니다.

---

## 6. 일반적인 모수적 검정

### 6.1 일표본 t-검정

$H_0: \mu = \mu_0$, $\sigma$ 미지:

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1} \quad \text{under } H_0$$

### 6.2 이표본 t-검정

**독립 표본** (등분산 가정):

$$T = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{1/n_1 + 1/n_2}}, \quad S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

**웰치 t-검정 (Welch's t-test)** (이분산): 개별 분산을 사용하며 새터스웨이트 자유도를 적용합니다.

### 6.3 대응 t-검정

대응 관측 $(X_i, Y_i)$에 대해, 차이 $D_i = X_i - Y_i$를 계산하고 $D$에 대해 일표본 t-검정을 적용합니다:

$$T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}$$

```python
import math
import statistics

def one_sample_t_test(data, mu_0):
    """One-sample t-test (two-sided)."""
    n = len(data)
    x_bar = statistics.mean(data)
    s = statistics.stdev(data)
    t_stat = (x_bar - mu_0) / (s / math.sqrt(n))
    df = n - 1
    # Approximate two-sided p-value using normal (good for df > 30)
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(t_stat)))
    return {"t_statistic": t_stat, "df": df, "p_value_approx": p_value}

def paired_t_test(x, y):
    """Paired t-test (two-sided)."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    return one_sample_t_test(diffs, mu_0=0)

# Example: test if mean differs from 50
data = [52, 48, 55, 51, 49, 53, 47, 56, 50, 54, 52, 48, 51, 53, 50]
result = one_sample_t_test(data, mu_0=50)
print(f"t = {result['t_statistic']:.4f}, df = {result['df']}, p ≈ {result['p_value_approx']:.4f}")

# Example: paired test (before vs after treatment)
before = [120, 135, 128, 140, 132, 125, 138, 130, 142, 136]
after  = [115, 130, 125, 136, 128, 120, 133, 126, 137, 131]
result_p = paired_t_test(before, after)
print(f"Paired t = {result_p['t_statistic']:.4f}, p ≈ {result_p['p_value_approx']:.4f}")
```

---

## 7. 카이제곱 검정

### 7.1 적합도 검정

관측된 빈도가 이론적 분포 하에서의 기대 빈도와 일치하는지 검정합니다.

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1-m}$$

여기서 $k$는 범주 수이고 $m$은 추정된 모수의 수입니다.

### 7.2 독립성 검정

$r \times c$ 분할표에 대해, 두 범주형 변수가 독립인지 검정합니다:

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}$$

자유도: $(r-1)(c-1)$.

```python
def chi_squared_gof(observed, expected):
    """Chi-squared goodness-of-fit test."""
    assert len(observed) == len(expected)
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return {"chi2": chi2, "df": df}

def chi_squared_independence(table):
    """Chi-squared test of independence for a 2D contingency table.

    table: list of lists (rows x cols)
    """
    rows = len(table)
    cols = len(table[0])
    N = sum(sum(row) for row in table)
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[r][c] for r in range(rows)) for c in range(cols)]

    chi2 = 0
    for i in range(rows):
        for j in range(cols):
            expected = row_totals[i] * col_totals[j] / N
            chi2 += (table[i][j] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    return {"chi2": chi2, "df": df}

# Goodness of fit: test if a die is fair
observed = [18, 15, 22, 20, 12, 13]  # 100 rolls
expected = [100/6] * 6
print("GoF:", chi_squared_gof(observed, expected))

# Independence: gender vs preference
table = [[30, 10, 20],   # Male:   A, B, C
         [25, 20, 15]]   # Female: A, B, C
print("Independence:", chi_squared_independence(table))
```

---

## 8. 가능도비 검정과 윌크스 정리

### 8.1 일반화된 가능도비

$H_0: \theta \in \Theta_0$ vs. $H_1: \theta \in \Theta \setminus \Theta_0$ 검정에 대해:

$$\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta \mid x)}{\sup_{\theta \in \Theta} L(\theta \mid x)}$$

$\Lambda$가 작을 때 (즉, 제한된 MLE가 비제한 MLE보다 훨씬 나쁠 때) $H_0$를 기각합니다.

### 8.2 윌크스 정리

정칙 조건 하에서, $n \to \infty$일 때:

$$-2 \ln \Lambda \xrightarrow{d} \chi^2_r$$

여기서 $r = \dim(\Theta) - \dim(\Theta_0)$은 자유 모수 수의 차이입니다.

이는 $\Lambda$의 정확한 분포를 유도할 필요 없이 점근적 귀무 분포를 제공하므로 매우 유용합니다.

---

## 9. 다중 검정 문제

### 9.1 문제

$m$개의 동시 검정을 수준 $\alpha$에서 수행할 때, 적어도 하나의 거짓 기각 확률 (가족별 오류율, Family-Wise Error Rate, FWER)은:

$$\text{FWER} = 1 - (1 - \alpha)^m$$

$\alpha = 0.05$에서 $m = 20$개 검정의 경우: FWER $\approx 0.64$. 거의 3분의 2의 확률로 적어도 하나의 위양성을 얻습니다.

### 9.2 본페로니 보정

$i$번째 가설을 $p_i \leq \alpha / m$이면 기각합니다. 이는 FWER를 수준 $\alpha$로 통제하지만 보수적입니다.

### 9.3 위발견률 (FDR)

**벤야미니-호흐베르크 (Benjamini-Hochberg, BH) 절차**는 모든 기각 중 거짓 발견의 기대 비율을 통제합니다:

1. p-값을 정렬합니다: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. $p_{(k)} \leq \frac{k}{m} \alpha$를 만족하는 가장 큰 $k$를 찾습니다.
3. $p_{(i)} \leq p_{(k)}$인 모든 가설을 기각합니다.

위발견률 (False Discovery Rate) 통제는 FWER보다 덜 보수적이며, 고차원 환경 (예: 유전체학)에서 선호됩니다.

```python
def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    m = len(p_values)
    threshold = alpha / m
    results = [(i, p, p <= threshold) for i, p in enumerate(p_values)]
    return threshold, results

def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR procedure."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    # Find the BH threshold
    bh_threshold = 0
    for rank, (idx, p) in enumerate(indexed, 1):
        if p <= rank / m * alpha:
            bh_threshold = p
    # Reject all with p <= bh_threshold
    rejected = [idx for idx, p in enumerate(p_values) if p <= bh_threshold]
    return bh_threshold, rejected

# Example: 10 tests, some with small p-values
p_values = [0.001, 0.008, 0.039, 0.041, 0.052, 0.10, 0.21, 0.45, 0.67, 0.91]
bonf_thresh, bonf_results = bonferroni_correction(p_values)
bh_thresh, bh_rejected = benjamini_hochberg(p_values)

print(f"Bonferroni threshold: {bonf_thresh:.4f}")
print(f"Bonferroni rejections: {[i for i, p, rej in bonf_results if rej]}")
print(f"BH threshold: {bh_thresh:.4f}")
print(f"BH rejections: {bh_rejected}")
```

---

## 10. 효과 크기와 실질적 유의성

### 10.1 구별

**통계적 유의성** ($p \leq \alpha$)은 관측된 효과가 $H_0$ 하에서 가능성이 낮다는 것을 의미합니다. 효과가 크거나 중요하다는 것을 의미하지는 **않습니다**.

**실질적 유의성 (Practical Significance)** 은 다음을 묻습니다: 효과가 현실 세계에서 중요할 만큼 충분히 큰가?

### 10.2 일반적인 효과 크기 측정

| 측정 | 공식 | 해석 |
|------|------|------|
| 코헨의 $d$ (Cohen's d) | $d = (\bar{X}_1 - \bar{X}_2)/S_p$ | 소: 0.2, 중: 0.5, 대: 0.8 |
| 상관계수 $r$ | 피어슨 또는 점이연 상관 | 소: 0.1, 중: 0.3, 대: 0.5 |
| 오즈비 (Odds Ratio) | $\text{OR} = \frac{p_1/(1-p_1)}{p_2/(1-p_2)}$ | 1 = 효과 없음 |
| $\eta^2$ (ANOVA) | $\text{SS}_B / \text{SS}_T$ | 설명된 분산의 비율 |

### 10.3 둘 다 중요한 이유

매우 큰 표본에서는 사소한 효과도 통계적으로 유의할 수 있습니다. 반대로, 의미 있는 효과도 표본이 너무 작으면 유의하지 않을 수 있습니다. 항상 p-값과 효과 크기를 함께 보고하세요.

```python
import math
import statistics

def cohens_d(group1, group2):
    """Compute Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)
    # Pooled standard deviation
    sp = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / sp

# Large sample, tiny effect
import random
random.seed(42)
group_a = [random.gauss(100.0, 15) for _ in range(10000)]
group_b = [random.gauss(100.5, 15) for _ in range(10000)]
d = cohens_d(group_a, group_b)
print(f"Cohen's d = {d:.4f} (very small effect)")
print("Even with n=10000, a tiny d can yield a significant p-value.")
```

---

## 11. 핵심 요약

| 개념 | 핵심 사항 |
|------|-----------|
| $H_0$ vs $H_1$ | $H_0$는 현상 유지; 입증 책임은 $H_1$에 있음 |
| 제1종 / 제2종 오류 | 트레이드오프: $\alpha$를 줄이면 $\beta$가 증가 |
| 검정력 | $n$, 효과 크기, $\alpha$가 증가하면 증가; $\sigma$가 증가하면 감소 |
| 네이만-피어슨 | 가능도비 검정이 단순가설에 대해 최강력 검정 |
| p-값 | $P(\text{이만큼 극단적이거나 더 극단적인 데이터} \mid H_0)$; $P(H_0)$가 아님 |
| 다중 검정 | 본페로니는 FWER을 통제 (보수적); BH는 FDR을 통제 |
| 효과 크기 | 통계적 유의성 $\neq$ 실질적 유의성 |
| 윌크스 정리 | $-2\ln\Lambda \to \chi^2_r$ 점근적으로 |

**보고 모범 사례**: 사용된 검정, 검정 통계량, 자유도, p-값, 신뢰구간, 효과 크기를 명시하세요.

---

**이전**: [구간 추정](./13_Interval_Estimation.md) | **다음**: [베이즈 추론](./15_Bayesian_Inference.md)

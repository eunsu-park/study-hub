# 비모수적 방법

**이전**: [베이즈 추론](./15_Bayesian_Inference.md) | **다음**: [회귀분석과 분산분석](./17_Regression_and_ANOVA.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 비모수적 방법이 모수적 방법보다 선호되는 경우와 이유를 설명하기
2. 모집단 중앙값에 대한 부호 검정 적용하기
3. 대응 데이터에 대한 윌콕슨 부호 순위 검정 수행하기
4. 두 독립 집단 비교를 위한 맨-위트니 U 검정 사용하기
5. 다중 집단에 대한 크루스칼-왈리스 검정 수행하기
6. 스피어만 순위 상관계수 계산하기
7. 커널 밀도 추정과 대역폭 선택 이해하기
8. 정확 및 근사 추론을 위한 순열 검정 구현하기
9. 일반적 추론을 위한 부트스트랩 방법 적용하기
10. 분포 비교를 위한 콜모고로프-스미르노프 검정 수행하기

---

비모수적 방법 (Nonparametric Methods)은 데이터의 기저 확률분포에 대해 최소한의 가정만을 합니다. 분포 가정 (예: 정규성)이 의심스러울 때, 데이터가 순서 척도일 때, 또는 표본 크기가 작아 분포 형태를 검증할 수 없을 때 특히 유용합니다.

---

## 1. 왜 비모수적 방법인가?

### 1.1 모수적 방법의 한계

모수적 검정 (t-검정, F-검정 등)은 특정 분포 형태를 가정합니다. 이러한 가정이 위반되면:

- 제1종 오류율이 팽창하거나 수축할 수 있습니다.
- 검정력이 크게 떨어질 수 있습니다.
- 신뢰구간의 포함률이 부정확할 수 있습니다.

### 1.2 비모수적 방법의 장점

- **분포 무관 (Distribution-Free)**: 더 약한 가정 하에서도 유효합니다.
- **강건 (Robust)**: 이상치와 두꺼운 꼬리에 덜 민감합니다.
- **유연**: 순서 데이터와 비표준 분포를 다룰 수 있습니다.
- **정확 검정 가능**: 소표본에서 정확한 p-값을 계산할 수 있습니다.

### 1.3 트레이드오프

- 모수적 가정이 실제로 성립할 때 **검정력이 낮습니다** (정규 데이터에서 윌콕슨 vs. t-검정의 점근 상대 효율은 $3/\pi \approx 0.955$).
- 일부 상황에서 **추정이 덜 정밀**합니다.
- 회귀 모형에 비해 **공변량 편입이 어렵습니다**.

---

## 2. 부호 검정

### 2.1 설정

모집단 중앙값 검정: $H_0: m = m_0$ vs. $H_1: m \neq m_0$.

### 2.2 절차

1. 각 관측에 대해 $D_i = X_i - m_0$를 계산합니다.
2. 영(0)을 버리고 남은 개수를 $n'$이라 합니다.
3. $S^+ = $ 양의 $D_i$ 수를 셉니다.
4. $H_0$ 하에서 $S^+ \sim \text{Binomial}(n', 0.5)$입니다.
5. $S^+$이 이 이항분포의 꼬리에 빠지면 기각합니다.

부호 검정 (Sign Test)은 가장 단순한 비모수적 검정입니다. 차이의 부호만 사용하고, 크기 정보는 버립니다.

```python
import math

def sign_test(data, m_0):
    """Two-sided sign test for median = m_0."""
    diffs = [x - m_0 for x in data if x != m_0]
    n = len(diffs)
    s_plus = sum(1 for d in diffs if d > 0)

    # P-value: 2 * P(X >= max(s_plus, n-s_plus)) under Binomial(n, 0.5)
    k = max(s_plus, n - s_plus)
    # Compute binomial tail probability
    p_tail = 0
    for i in range(k, n + 1):
        # Binomial coefficient * 0.5^n
        binom_coeff = math.comb(n, i)
        p_tail += binom_coeff * (0.5 ** n)
    p_value = 2 * p_tail
    p_value = min(p_value, 1.0)

    return {"s_plus": s_plus, "n": n, "p_value": p_value}

# Example: test if median weight is 70 kg
weights = [68, 72, 65, 74, 71, 69, 73, 67, 75, 70, 66, 77, 63, 72, 71]
result = sign_test(weights, m_0=70)
print(f"Sign test: S+ = {result['s_plus']}, n = {result['n']}, p = {result['p_value']:.4f}")
```

---

## 3. 윌콕슨 부호 순위 검정

### 3.1 설정

대응 데이터나 중앙값 검정에 대한 부호 검정의 더 강력한 대안입니다. 차이의 부호와 크기(순위) 모두를 사용합니다.

### 3.2 절차

1. $D_i = X_i - m_0$ (또는 대응 데이터의 경우 $D_i = X_i - Y_i$)를 계산합니다.
2. 영을 버립니다.
3. $|D_i|$를 작은 것부터 큰 것까지 순위를 매깁니다.
4. 검정 통계량은 $W^+ = \sum_{D_i > 0} R_i$ (양의 차이 순위의 합)입니다.
5. $H_0$ 하에서 $W^+$의 분포는 대칭입니다.

큰 $n$에서 정규 근사:

$$Z = \frac{W^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}$$

```python
def wilcoxon_signed_rank(data, m_0=0):
    """Wilcoxon signed-rank test (two-sided, normal approximation)."""
    diffs = [(x - m_0) for x in data if x != m_0]
    n = len(diffs)

    # Rank absolute differences
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort()
    ranks = [0] * n
    for rank, (_, idx) in enumerate(abs_diffs, 1):
        ranks[idx] = rank

    # W+: sum of ranks where diff is positive
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)

    # Normal approximation
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean_w) / std_w

    # Two-sided p-value (normal approximation)
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(z)))

    return {"W_plus": w_plus, "z": z, "p_value": p_value}

# Example: paired data (before vs after)
before = [125, 130, 118, 140, 135, 128, 132, 137, 122, 145, 138, 127]
after  = [118, 125, 112, 133, 130, 122, 128, 131, 117, 138, 131, 120]
diffs = [a - b for a, b in zip(before, after)]
result = wilcoxon_signed_rank(diffs, m_0=0)
print(f"Wilcoxon signed-rank: W+ = {result['W_plus']}, z = {result['z']:.3f}, p = {result['p_value']:.4f}")
```

---

## 4. 맨-위트니 U 검정

### 4.1 설정

정규성을 가정하지 않고 두 독립 집단을 비교합니다. 한 분포가 다른 것보다 확률적으로 큰지 검정합니다.

$H_0: P(X > Y) = 0.5$ (두 분포가 동일)

### 4.2 절차

1. 두 표본을 합치고 모든 관측을 1부터 $N = n_1 + n_2$까지 순위를 매깁니다.
2. $R_1 = $ 집단 1의 순위 합을 계산합니다.
3. $U_1 = R_1 - n_1(n_1+1)/2$.
4. $U_2 = n_1 n_2 - U_1$.
5. 검정 통계량은 $U = \min(U_1, U_2)$입니다.

대표본에서 정규 근사:

$$Z = \frac{U_1 - n_1 n_2 / 2}{\sqrt{n_1 n_2 (n_1 + n_2 + 1) / 12}}$$

```python
def mann_whitney_u(group1, group2):
    """Mann-Whitney U test (two-sided, normal approximation)."""
    n1, n2 = len(group1), len(group2)
    combined = [(val, 1) for val in group1] + [(val, 2) for val in group2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # average of ranks i+1 to j
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1

    # Normal approximation
    mu_u = n1 * n2 / 2
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu_u) / sigma_u

    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p_value = 2 * (1 - phi(abs(z)))

    return {"U1": u1, "U2": u2, "z": z, "p_value": p_value}

# Example: compare two teaching methods
method_a = [78, 82, 85, 71, 90, 76, 88, 83, 79, 86]
method_b = [65, 72, 80, 68, 75, 70, 74, 69, 77, 73]
result = mann_whitney_u(method_a, method_b)
print(f"Mann-Whitney U: U1={result['U1']:.0f}, z={result['z']:.3f}, p={result['p_value']:.4f}")
```

---

## 5. 크루스칼-왈리스 검정

### 5.1 설정

맨-위트니 U 검정을 $k \geq 3$개 독립 집단의 비교로 확장합니다. 일원 분산분석 (One-Way ANOVA)의 비모수적 대응입니다.

$H_0$: 모든 $k$ 집단이 동일한 분포를 가짐. $H_1$: 적어도 하나의 집단이 다름.

### 5.2 검정 통계량

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

여기서 $R_i$는 집단 $i$의 순위 합, $n_i$는 집단 $i$의 크기, $N = \sum n_i$입니다.

$H_0$ 하에서 대표본 근사로 $H \sim \chi^2_{k-1}$입니다.

```python
def kruskal_wallis(*groups):
    """Kruskal-Wallis H test (normal approximation)."""
    k = len(groups)
    all_data = []
    for i, group in enumerate(groups):
        for val in group:
            all_data.append((val, i))
    all_data.sort(key=lambda x: x[0])
    N = len(all_data)

    # Assign average ranks for ties
    ranks = [0.0] * N
    i = 0
    while i < N:
        j = i
        while j < N and all_data[j][0] == all_data[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for idx in range(i, j):
            ranks[idx] = avg_rank
        i = j

    # Sum of ranks per group
    rank_sums = [0.0] * k
    group_sizes = [len(g) for g in groups]
    for idx in range(N):
        group_id = all_data[idx][1]
        rank_sums[group_id] += ranks[idx]

    H = (12 / (N * (N + 1))) * sum(r**2 / n for r, n in zip(rank_sums, group_sizes)) - 3 * (N + 1)
    df = k - 1
    return {"H": H, "df": df}

# Example: compare 3 fertilizers on crop yield
fert_a = [45, 52, 48, 55, 50]
fert_b = [60, 58, 65, 62, 57]
fert_c = [40, 42, 38, 44, 41]
result = kruskal_wallis(fert_a, fert_b, fert_c)
print(f"Kruskal-Wallis: H={result['H']:.3f}, df={result['df']}")
```

---

## 6. 스피어만 순위 상관

### 6.1 정의

스피어만 순위 상관계수 (Spearman's Rank Correlation) $r_s$는 두 변수 간 단조 관계의 강도와 방향을 측정합니다.

$$r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

여기서 $d_i = \text{rank}(X_i) - \text{rank}(Y_i)$입니다.

동순위가 없을 때, $r_s$는 순위에 대해 계산한 피어슨 상관계수와 정확히 같습니다.

### 6.2 성질

- $-1 \leq r_s \leq 1$
- $r_s = 1$: 완벽한 단조 증가 관계
- $r_s = -1$: 완벽한 단조 감소 관계
- 피어슨의 $r$보다 이상치에 더 강건

```python
def spearman_rank_correlation(x, y):
    """Compute Spearman's rank correlation coefficient."""
    assert len(x) == len(y)
    n = len(x)

    def rank_data(data):
        indexed = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and data[indexed[j]] == data[indexed[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)
    d_sq = sum((a - b)**2 for a, b in zip(rx, ry))
    rs = 1 - 6 * d_sq / (n * (n**2 - 1))
    return rs

# Example
hours_studied = [2, 4, 6, 8, 10, 1, 3, 5, 7, 9]
test_scores   = [55, 70, 80, 88, 95, 50, 65, 75, 85, 92]
rs = spearman_rank_correlation(hours_studied, test_scores)
print(f"Spearman rs = {rs:.4f}")
```

---

## 7. 커널 밀도 추정 (KDE)

### 7.1 정의

커널 밀도 추정 (Kernel Density Estimation)은 표본에서 확률밀도함수 $f(x)$를 추정합니다:

$$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)$$

여기서 $K$는 커널 함수 (예: 가우스)이고 $h > 0$는 **대역폭 (Bandwidth)**입니다.

### 7.2 일반적인 커널

| 커널 | $K(u)$ |
|------|--------|
| 가우스 | $\frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ |
| 에파네치니코프 (Epanechnikov) | $\frac{3}{4}(1 - u^2)$ ($|u| \leq 1$) |
| 균등 | $\frac{1}{2}$ ($|u| \leq 1$) |

### 7.3 대역폭 선택

대역폭 $h$가 가장 중요한 매개변수입니다:

- **너무 작으면**: 과소 평활; 잡음이 많고 뾰족한 추정.
- **너무 크면**: 과대 평활; 특징이 사라짐.

**실버만 경험 규칙 (Silverman's Rule of Thumb)** (가우스 커널):

$$h = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5}$$

여기서 $\hat{\sigma}$는 표본 표준편차입니다.

```python
def gaussian_kde(data, x_grid, bandwidth=None):
    """Gaussian Kernel Density Estimation.

    Args:
        data: list of observations
        x_grid: list of points at which to evaluate the density
        bandwidth: bandwidth h; if None, uses Silverman's rule
    """
    n = len(data)
    if bandwidth is None:
        # Silverman's rule of thumb
        mean_val = sum(data) / n
        std_val = math.sqrt(sum((x - mean_val)**2 for x in data) / (n - 1))
        bandwidth = 1.06 * std_val * n**(-0.2)

    def gaussian_kernel(u):
        return math.exp(-0.5 * u**2) / math.sqrt(2 * math.pi)

    density = []
    for x in x_grid:
        val = sum(gaussian_kernel((x - xi) / bandwidth) for xi in data) / (n * bandwidth)
        density.append(val)
    return density, bandwidth

# Example: estimate density of bimodal data
import random
random.seed(42)
data = [random.gauss(2, 0.8) for _ in range(50)] + [random.gauss(5, 1.0) for _ in range(50)]
x_grid = [i * 0.1 for i in range(-20, 100)]

density, h = gaussian_kde(data, x_grid)
print(f"Bandwidth (Silverman): {h:.3f}")
peak_x = x_grid[density.index(max(density))]
print(f"Highest density at x = {peak_x:.1f}")
```

---

## 8. 순열 검정

### 8.1 아이디어

$H_0$ (예: 집단 간 차이 없음) 하에서 집단 레이블은 교환 가능합니다. 레이블의 모든 (또는 많은) 순열에 대해 검정 통계량을 계산하고, 관측된 통계량을 이 순열 분포와 비교합니다.

### 8.2 정확 순열 검정

$\binom{N}{n_1}$개의 가능한 모든 배정을 열거합니다. $N$이 작은 경우에만 실행 가능합니다.

### 8.3 근사 (몬테카를로) 순열 검정

순열 분포를 근사하기 위해 레이블을 $B$번 (예: $B = 10000$) 무작위로 섞습니다.

```python
import random

def permutation_test(group1, group2, n_permutations=10000, seed=42):
    """Two-sided permutation test for difference in means."""
    random.seed(seed)
    combined = group1 + group2
    n1 = len(group1)
    observed_diff = abs(sum(group1)/n1 - sum(group2)/len(group2))

    count_extreme = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_diff = abs(sum(perm_g1)/n1 - sum(perm_g2)/len(perm_g2))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return {"observed_diff": observed_diff, "p_value": p_value}

# Example
treatment = [5.2, 6.1, 4.8, 5.5, 6.3, 5.9, 6.0, 5.7]
control   = [4.1, 3.8, 4.5, 4.0, 3.9, 4.3, 4.2, 3.7]
result = permutation_test(treatment, control)
print(f"Observed diff: {result['observed_diff']:.3f}, p-value: {result['p_value']:.4f}")
```

---

## 9. 부트스트랩 방법

### 9.1 일반 프레임워크

부트스트랩 (Bootstrap)은 관측 데이터에서 **복원 추출**하여 임의의 통계량의 표본 분포를 근사합니다.

1. $B$개의 크기 $n$ 부트스트랩 표본을 추출합니다 (복원 추출).
2. 각각에 대해 통계량 $\hat{\theta}^*_b$를 계산합니다.
3. $\{\hat{\theta}^*_b\}$의 분포를 추론에 사용합니다 (CI, 표준 오차, 가설 검정).

### 9.2 부트스트랩 가설 검정

$H_0: \theta = \theta_0$를 검정하려면:
1. $H_0$가 참이 되도록 데이터를 이동합니다 (예: 각 관측에서 $\bar{x} - \theta_0$를 뺌).
2. 이동된 데이터에서 부트스트랩합니다.
3. 부트스트랩 통계량 중 관측된 것만큼 극단적인 비율을 p-값으로 계산합니다.

```python
import random
import statistics

def bootstrap_test_mean(data, mu_0, B=10000, seed=42):
    """Bootstrap hypothesis test for H0: mean = mu_0."""
    random.seed(seed)
    n = len(data)
    x_bar = statistics.mean(data)
    observed_stat = abs(x_bar - mu_0)

    # Shift data to enforce H0
    shifted = [x - x_bar + mu_0 for x in data]

    count = 0
    for _ in range(B):
        boot_sample = random.choices(shifted, k=n)
        boot_mean = statistics.mean(boot_sample)
        if abs(boot_mean - mu_0) >= observed_stat:
            count += 1

    return {"observed_mean": x_bar, "p_value": count / B}

data = [23.1, 25.4, 22.8, 24.0, 26.1, 23.5, 24.8, 25.0, 22.5, 24.3]
result = bootstrap_test_mean(data, mu_0=24.0)
print(f"Mean = {result['observed_mean']:.2f}, Bootstrap p = {result['p_value']:.4f}")
```

---

## 10. 콜모고로프-스미르노프 검정

### 10.1 일표본 KS 검정

표본이 지정된 분포 $F_0$에서 왔는지 검정합니다:

$$D_n = \sup_x |F_n(x) - F_0(x)|$$

여기서 $F_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(X_i \leq x)$는 경험적 CDF입니다.

### 10.2 이표본 KS 검정

두 경험적 CDF를 비교합니다:

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

$H_0$ 하에서 척도화된 통계량의 분포는 점근적으로 알려져 있습니다.

```python
def ks_two_sample(sample1, sample2):
    """Two-sample Kolmogorov-Smirnov test statistic."""
    s1 = sorted(sample1)
    s2 = sorted(sample2)
    n1, n2 = len(s1), len(s2)

    # Merge and compute ECDFs
    all_vals = sorted(set(s1 + s2))
    max_diff = 0.0
    for x in all_vals:
        ecdf1 = sum(1 for v in s1 if v <= x) / n1
        ecdf2 = sum(1 for v in s2 if v <= x) / n2
        diff = abs(ecdf1 - ecdf2)
        if diff > max_diff:
            max_diff = diff

    # Approximate critical value (alpha=0.05)
    c_alpha = 1.36  # for alpha=0.05
    critical = c_alpha * math.sqrt((n1 + n2) / (n1 * n2))

    return {"D": max_diff, "critical_value_05": critical,
            "reject_H0": max_diff > critical}

# Example: do two samples come from the same distribution?
random.seed(42)
sample_a = [random.gauss(0, 1) for _ in range(50)]
sample_b = [random.gauss(0.5, 1) for _ in range(50)]
result = ks_two_sample(sample_a, sample_b)
print(f"KS D = {result['D']:.4f}, critical (5%) = {result['critical_value_05']:.4f}")
print(f"Reject H0: {result['reject_H0']}")
```

---

## 11. 핵심 요약

| 방법 | 사용 상황 | 모수적 대응 |
|------|-----------|-------------|
| 부호 검정 | 중앙값 검정; 순서 데이터 | 일표본 t-검정 |
| 윌콕슨 부호 순위 | 대응 차이; 대칭 분포 | 대응 t-검정 |
| 맨-위트니 U | 두 독립 집단 비교 | 이표본 t-검정 |
| 크루스칼-왈리스 | $k \geq 3$개 집단 비교 | 일원 분산분석 |
| 스피어만 $r_s$ | 단조 관계 | 피어슨 $r$ |
| KDE | 모수 모형 없이 밀도 추정 | 모수적 밀도 적합 |
| 순열 검정 | 임의의 가설; 정확 또는 근사 | 다양 |
| 부트스트랩 | 일반적 추론 (CI, 검정) | 다양 |
| 콜모고로프-스미르노프 | 분포 비교 | 가능도비 |

**실용적 지침**: 모수적 가정이 성립하면 (최대 검정력을 위해) 모수적 검정을 사용하세요. 가정이 의심스럽거나 데이터가 순서 척도이면 비모수적 대안을 사용하세요. 순열 검정과 부트스트랩 방법은 거의 모든 상황에서 작동하는 다재다능한 도구입니다.

---

**이전**: [베이즈 추론](./15_Bayesian_Inference.md) | **다음**: [회귀분석과 분산분석](./17_Regression_and_ANOVA.md)

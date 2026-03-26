# 회귀분석과 분산분석

**이전**: [비모수적 방법](./16_Nonparametric_Methods.md) | **다음**: [확률과정 입문](./18_Stochastic_Processes_Introduction.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단순 선형 회귀 (simple linear regression)를 정식화하고 OLS 추정량을 유도하기
2. 회귀 계수, 표준 오차, $R^2$를 해석하기
3. 가우스-마르코프 조건 (Gauss-Markov conditions)을 검증하고 BLUE 성질을 이해하기
4. 모형 진단을 위한 잔차 분석 (residual analysis) 수행하기
5. 행렬 형식의 다중 선형 회귀 (multiple linear regression)로 확장하기
6. 전체 회귀의 유의성에 대한 F-검정 (F-test) 수행하기
7. 일원 및 이원 분산분석 (ANOVA)에서의 분산 분해하기
8. 쌍별 비교를 위한 사후 검정 (post-hoc test, Tukey HSD) 적용하기
9. Python으로 회귀분석과 분산분석을 처음부터 구현하기

---

회귀분석 (regression analysis)은 반응 변수와 하나 이상의 예측 변수 간의 관계를 모형화합니다. 분산분석(ANOVA, Analysis of Variance)은 집단 평균을 비교하기 위한 밀접하게 관련된 프레임워크입니다. 둘 다 응용 통계학의 핵심 기법입니다.

---

## 1. 단순 선형 회귀

### 1.1 모형

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad i = 1, \ldots, n$$

- $Y_i$: 반응 변수 (종속 변수)
- $X_i$: 예측 변수 (독립 변수)
- $\beta_0$: 절편 (intercept)
- $\beta_1$: 기울기 ($X$의 단위 변화당 $Y$의 변화량)
- $\varepsilon_i$: 확률 오차, 독립적으로 $\varepsilon_i \sim N(0, \sigma^2)$으로 가정

### 1.2 최소제곱법 (Ordinary Least Squares, OLS) 추정량

잔차 제곱합을 최소화합니다:

$$\text{RSS} = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2$$

편미분을 구하고 0으로 놓으면:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n}(X_i - \bar{X})^2} = \frac{S_{XY}}{S_{XX}}$$

$$\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$

### 1.3 성질

모형 가정 하에서:
- $\hat{\beta}_0$와 $\hat{\beta}_1$은 비편향 (unbiased)입니다: $E[\hat{\beta}_1] = \beta_1$, $E[\hat{\beta}_0] = \beta_0$.
- $\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{S_{XX}}$
- $\text{Var}(\hat{\beta}_0) = \sigma^2 \left(\frac{1}{n} + \frac{\bar{X}^2}{S_{XX}}\right)$
- $\sigma^2$의 비편향 추정량은 $\hat{\sigma}^2 = \frac{\text{RSS}}{n-2}$입니다.

```python
import math

def simple_linear_regression(x, y):
    """Fit simple linear regression Y = b0 + b1*X via OLS."""
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n

    s_xy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    s_xx = sum((xi - x_bar) ** 2 for xi in x)

    b1 = s_xy / s_xx
    b0 = y_bar - b1 * x_bar

    # Residuals and RSS
    residuals = [yi - (b0 + b1 * xi) for xi, yi in zip(x, y)]
    rss = sum(r ** 2 for r in residuals)
    sigma2 = rss / (n - 2)

    # Standard errors
    se_b1 = math.sqrt(sigma2 / s_xx)
    se_b0 = math.sqrt(sigma2 * (1/n + x_bar**2 / s_xx))

    # t-statistics
    t_b1 = b1 / se_b1
    t_b0 = b0 / se_b0

    return {
        "b0": b0, "b1": b1,
        "se_b0": se_b0, "se_b1": se_b1,
        "t_b0": t_b0, "t_b1": t_b1,
        "sigma2": sigma2, "residuals": residuals,
        "rss": rss
    }

# Example: study hours vs exam score
hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [52, 55, 60, 63, 68, 71, 75, 78, 82, 86]
result = simple_linear_regression(hours, scores)
print(f"Y = {result['b0']:.2f} + {result['b1']:.2f} * X")
print(f"SE(b1) = {result['se_b1']:.4f}, t(b1) = {result['t_b1']:.3f}")
print(f"Residual variance: {result['sigma2']:.4f}")
```

---

## 2. 결정계수 ($R^2$)

### 2.1 정의

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{SSR}}{\text{TSS}}$$

여기서:
- $\text{TSS} = \sum(Y_i - \bar{Y})^2$ (총 제곱합, Total Sum of Squares)
- $\text{RSS} = \sum(Y_i - \hat{Y}_i)^2$ (잔차 제곱합, Residual Sum of Squares)
- $\text{SSR} = \sum(\hat{Y}_i - \bar{Y})^2$ (회귀 제곱합, Regression Sum of Squares)
- $\text{TSS} = \text{SSR} + \text{RSS}$

### 2.2 해석

$R^2$는 회귀 모형에 의해 설명되는 $Y$ 분산의 비율을 나타냅니다. 절편이 포함된 모형에서 $R^2 \in [0, 1]$입니다.

### 2.3 수정 $R^2$

예측 변수를 추가하면 $R^2$는 항상 증가합니다. 수정 버전은 예측 변수의 수 $p$에 대해 벌칙을 부과합니다:

$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)} = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$

```python
def compute_r_squared(y, y_hat):
    """Compute R-squared and adjusted R-squared."""
    n = len(y)
    y_bar = sum(y) / n
    tss = sum((yi - y_bar)**2 for yi in y)
    rss = sum((yi - yhi)**2 for yi, yhi in zip(y, y_hat))
    r2 = 1 - rss / tss
    return r2

# Using previous regression
y_hat = [result['b0'] + result['b1'] * xi for xi in hours]
r2 = compute_r_squared(scores, y_hat)
n, p = len(hours), 1
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {r2_adj:.4f}")
```

---

## 3. 가우스-마르코프 조건과 BLUE

### 3.1 가우스-마르코프 가정

1. **선형성 (Linearity)**: $Y = X\beta + \varepsilon$ (모형이 모수에 대해 선형).
2. **엄격 외생성 (Strict exogeneity)**: $E[\varepsilon_i \mid X] = 0$.
3. **등분산성 (Homoscedasticity)**: $\text{Var}(\varepsilon_i \mid X) = \sigma^2$ (분산이 일정).
4. **자기상관 없음 (No autocorrelation)**: $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ ($i \neq j$인 경우).
5. **완전 계수 (Full rank)**: 설계 행렬 $X$가 완전 열 계수를 가짐 (완전 다중공선성 없음).

### 3.2 BLUE 성질

가정 1--5 하에서 OLS 추정량은 **BLUE** (Best Linear Unbiased Estimators, 최량 선형 비편향 추정량)입니다:
- **Best**: 모든 선형 비편향 추정량 중 최소 분산.
- **Linear**: $Y$의 선형 함수.
- **Unbiased**: $E[\hat{\beta}] = \beta$.

오차의 정규성 ($\varepsilon \sim N(0, \sigma^2 I)$)을 추가하면, OLS는 (선형뿐 아니라) 최소 분산 비편향 추정량이 됩니다.

---

## 4. 잔차 분석과 진단

### 4.1 점검 항목

잔차 $e_i = Y_i - \hat{Y}_i$는 모형이 올바를 경우 독립적인 $N(0, \sigma^2)$처럼 행동해야 합니다.

| 그래프 | 점검 내용 |
|---|---|
| 잔차 vs. 적합값 | 선형성, 등분산성 |
| 잔차의 Q-Q 도표 | 정규성 |
| 스케일-위치 도표 | 등분산성 |
| 잔차 vs. 순서 | 독립성 (시계열) |

### 4.2 흔한 문제와 해결 방법

| 문제 | 진단 신호 | 해결 방법 |
|---|---|---|
| 비선형성 | 잔차 vs. 적합값에서 곡선 패턴 | 다항식 항 추가, 변수 변환 |
| 이분산성 | 잔차에서 부채꼴 형태 | 가중 LS, 로버스트 표준 오차, 로그 변환 |
| 비정규성 | Q-Q 도표에서 이탈 | 반응 변수 변환, 로버스트 방법 사용 |
| 이상값 | 큰 표준화 잔차 ($|e_i/s| > 3$) | 조사; 로버스트 회귀 고려 |
| 영향력 있는 점 | 높은 지렛대 + 큰 잔차 | 쿡의 거리 (Cook's distance); 데이터 품질 조사 |

### 4.3 지렛대와 쿡의 거리

**지렛대 (leverage)** $h_{ii}$는 $X_i$가 얼마나 비정상적인지를 측정합니다. 높은 지렛대를 가진 점은 적합에 강한 영향을 미칠 수 있습니다.

**쿡의 거리 (Cook's distance)**는 지렛대와 잔차의 크기를 결합합니다:

$$D_i = \frac{e_i^2}{p \cdot \hat{\sigma}^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$$

$D_i > 4/n$ 또는 $D_i > 1$인 점은 면밀히 조사할 필요가 있습니다.

```python
def residual_diagnostics(x, y, b0, b1):
    """Basic residual diagnostics for simple linear regression."""
    n = len(x)
    x_bar = sum(x) / n
    s_xx = sum((xi - x_bar)**2 for xi in x)

    y_hat = [b0 + b1 * xi for xi, yi in zip(x, y)]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]
    rss = sum(r**2 for r in residuals)
    sigma2 = rss / (n - 2)
    sigma = math.sqrt(sigma2)

    # Leverage: h_ii = 1/n + (x_i - x_bar)^2 / S_xx
    leverages = [1/n + (xi - x_bar)**2 / s_xx for xi in x]

    # Standardized residuals
    std_residuals = [r / (sigma * math.sqrt(1 - h)) for r, h in zip(residuals, leverages)]

    # Cook's distance
    p = 2  # number of parameters (b0, b1)
    cooks_d = [(r**2 / (p * sigma2)) * (h / (1 - h)**2) for r, h in zip(residuals, leverages)]

    print("Residual Diagnostics:")
    print(f"{'i':>3} {'Y':>6} {'Yhat':>6} {'Resid':>7} {'StdRes':>7} {'Lever':>6} {'CooksD':>7}")
    for i in range(n):
        print(f"{i+1:>3} {y[i]:>6.1f} {y_hat[i]:>6.2f} {residuals[i]:>7.3f} "
              f"{std_residuals[i]:>7.3f} {leverages[i]:>6.3f} {cooks_d[i]:>7.4f}")

    # Flag influential points
    threshold = 4 / n
    influential = [i+1 for i, d in enumerate(cooks_d) if d > threshold]
    if influential:
        print(f"\nPotentially influential points (Cook's D > {threshold:.3f}): {influential}")

residual_diagnostics(hours, scores, result['b0'], result['b1'])
```

---

## 5. 다중 선형 회귀

### 5.1 행렬 정식화

$$Y = X\beta + \varepsilon$$

여기서 $Y$는 $n \times 1$, $X$는 $n \times (p+1)$ (절편 열 포함), $\beta$는 $(p+1) \times 1$, $\varepsilon$는 $n \times 1$입니다.

### 5.2 OLS 해

$$\hat{\beta} = (X^\top X)^{-1} X^\top Y$$

이것은 $\|Y - X\beta\|^2$를 최소화합니다. $X^\top X$가 역행렬이 존재할 때 (완전 계수 조건) 해가 존재합니다.

### 5.3 성질

- $E[\hat{\beta}] = \beta$
- $\text{Cov}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}$
- $\hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1}$

```python
def matrix_multiply(A, B):
    """Multiply two matrices represented as lists of lists."""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(A):
    """Transpose a matrix."""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def invert_2x2(M):
    """Invert a 2x2 matrix."""
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    det = a * d - b * c
    return [[d/det, -b/det], [-c/det, a/det]]

def simple_ols_matrix(x, y):
    """OLS using matrix formulation for simple regression."""
    n = len(x)
    # Design matrix X: column of 1s and x values
    X = [[1, xi] for xi in x]
    Y = [[yi] for yi in y]

    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtY = matrix_multiply(Xt, Y)
    XtX_inv = invert_2x2(XtX)
    beta_hat = matrix_multiply(XtX_inv, XtY)

    return [beta_hat[i][0] for i in range(len(beta_hat))]

beta = simple_ols_matrix(hours, scores)
print(f"Matrix OLS: b0 = {beta[0]:.4f}, b1 = {beta[1]:.4f}")
```

---

## 6. 전체 유의성에 대한 F-검정

### 6.1 가설

$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$ (선형 관계 없음)

$H_1:$ 적어도 하나의 $\beta_j \neq 0$

### 6.2 F-통계량

$$F = \frac{\text{SSR}/p}{\text{RSS}/(n-p-1)} = \frac{\text{MSR}}{\text{MSE}} \sim F_{p, n-p-1} \quad \text{under } H_0$$

$F > F_{\alpha, p, n-p-1}$일 때 $H_0$을 기각합니다.

### 6.3 $R^2$와의 관계

$$F = \frac{R^2 / p}{(1 - R^2)/(n - p - 1)}$$

```python
def f_test_regression(y, y_hat, p):
    """F-test for overall regression significance.

    Args:
        y: observed values
        y_hat: fitted values
        p: number of predictors (excluding intercept)
    """
    n = len(y)
    y_bar = sum(y) / n
    ssr = sum((yhi - y_bar)**2 for yhi in y_hat)
    rss = sum((yi - yhi)**2 for yi, yhi in zip(y, y_hat))
    msr = ssr / p
    mse = rss / (n - p - 1)
    f_stat = msr / mse
    return {"F": f_stat, "df1": p, "df2": n - p - 1, "MSR": msr, "MSE": mse}

y_hat = [result['b0'] + result['b1'] * xi for xi in hours]
f_result = f_test_regression(scores, y_hat, p=1)
print(f"F = {f_result['F']:.2f}, df = ({f_result['df1']}, {f_result['df2']})")
```

---

## 7. 일원 분산분석

### 7.1 설정

$k$개 집단 평균을 비교합니다: $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$ vs. $H_1$: 적어도 두 평균이 다름.

### 7.2 분산 분해

$$\text{SST} = \text{SSB} + \text{SSW}$$

- **SST** (총 제곱합, Total): $\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{\cdot\cdot})^2$
- **SSB** (집단 간 제곱합, Between groups): $\sum_{i=1}^{k} n_i(\bar{Y}_{i\cdot} - \bar{Y}_{\cdot\cdot})^2$
- **SSW** (집단 내 제곱합, Within groups): $\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{i\cdot})^2$

### 7.3 F-통계량

$$F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB}/(k-1)}{\text{SSW}/(N-k)} \sim F_{k-1, N-k} \quad \text{under } H_0$$

### 7.4 분산분석표

| 변동원 | SS | df | MS | F |
|---|---|---|---|---|
| 집단 간 | SSB | $k-1$ | SSB/$(k-1)$ | MSB/MSW |
| 집단 내 | SSW | $N-k$ | SSW/$(N-k)$ | |
| 합계 | SST | $N-1$ | | |

```python
def one_way_anova(*groups):
    """One-way ANOVA F-test."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / N

    # Between-group sum of squares
    ssb = sum(len(g) * (sum(g)/len(g) - grand_mean)**2 for g in groups)

    # Within-group sum of squares
    ssw = 0
    for g in groups:
        g_mean = sum(g) / len(g)
        ssw += sum((x - g_mean)**2 for x in g)

    sst = ssb + ssw
    df_between = k - 1
    df_within = N - k
    msb = ssb / df_between
    msw = ssw / df_within
    f_stat = msb / msw

    print("ANOVA Table:")
    print(f"{'Source':<10} {'SS':>10} {'df':>5} {'MS':>10} {'F':>10}")
    print(f"{'Between':<10} {ssb:>10.3f} {df_between:>5} {msb:>10.3f} {f_stat:>10.3f}")
    print(f"{'Within':<10} {ssw:>10.3f} {df_within:>5} {msw:>10.3f}")
    print(f"{'Total':<10} {sst:>10.3f} {N-1:>5}")

    return {"F": f_stat, "df_between": df_between, "df_within": df_within,
            "ssb": ssb, "ssw": ssw, "sst": sst}

# Example: three teaching methods
method_1 = [85, 90, 88, 92, 87, 91]
method_2 = [78, 82, 80, 75, 79, 81]
method_3 = [72, 68, 74, 70, 73, 71]
result = one_way_anova(method_1, method_2, method_3)
```

---

## 8. 이원 분산분석

### 8.1 모형

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}$$

여기서 $\alpha_i$는 요인 A의 효과 (수준 $i$), $\beta_j$는 요인 B의 효과 (수준 $j$), $(\alpha\beta)_{ij}$는 교호작용 (interaction) 효과입니다.

### 8.2 분해

$$\text{SST} = \text{SS}_A + \text{SS}_B + \text{SS}_{AB} + \text{SS}_E$$

세 가지 F-검정을 수행합니다:
- $F_A = \text{MS}_A / \text{MS}_E$: 요인 A의 주효과 (main effect)
- $F_B = \text{MS}_B / \text{MS}_E$: 요인 B의 주효과
- $F_{AB} = \text{MS}_{AB} / \text{MS}_E$: 교호작용

### 8.3 교호작용의 해석

교호작용 $(\alpha\beta)_{ij}$가 유의할 때, 한 요인의 효과는 다른 요인의 수준에 따라 달라집니다. 이 경우 주효과만으로는 데이터를 충분히 설명할 수 없으며, 교호작용 도표 (profile plots)를 검토해야 합니다.

---

## 9. 사후 검정: Tukey HSD

### 9.1 사후 검정이 필요한 이유

분산분석은 적어도 두 평균이 다르다는 것을 알려주지만, 어떤 쌍이 다른지는 알려주지 않습니다. 사후 검정 (post-hoc tests)은 족속 오류율 (family-wise error rate)을 통제하면서 쌍별 비교를 수행합니다.

### 9.2 Tukey의 정직 유의 차이 (Honestly Significant Difference)

평균 $\bar{Y}_i$와 $\bar{Y}_j$를 비교할 때:

$$|\bar{Y}_i - \bar{Y}_j| > q_{\alpha, k, N-k} \cdot \sqrt{\frac{\text{MSW}}{n}}$$

여기서 $q_{\alpha, k, N-k}$는 스튜던트화 범위 (studentized range) 임계값입니다 (등그룹 크기 $n$의 경우). 부등식이 성립하면 해당 쌍이 유의하게 다르다고 판정합니다.

```python
def tukey_hsd(*groups, alpha=0.05):
    """Simplified Tukey HSD pairwise comparisons.

    Uses approximate critical value for common scenarios.
    """
    k = len(groups)
    N = sum(len(g) for g in groups)
    means = [sum(g)/len(g) for g in groups]
    sizes = [len(g) for g in groups]

    # Within-group MSW
    ssw = 0
    for g in groups:
        g_mean = sum(g) / len(g)
        ssw += sum((x - g_mean)**2 for x in g)
    msw = ssw / (N - k)

    # Approximate q critical values (alpha=0.05)
    # q(0.05, k, df) -- rough approximations for common cases
    q_table = {(3, 15): 3.67, (3, 12): 3.77, (3, 20): 3.58,
               (4, 16): 3.99, (4, 20): 3.96}
    df_within = N - k
    q_val = q_table.get((k, df_within), 3.70)  # fallback

    print(f"\nTukey HSD (MSW = {msw:.3f}, q ≈ {q_val:.2f}):")
    print(f"{'Pair':<15} {'Diff':>8} {'HSD':>8} {'Significant?':>14}")
    for i in range(k):
        for j in range(i+1, k):
            diff = abs(means[i] - means[j])
            # For unequal sizes, use harmonic mean
            n_h = 2 / (1/sizes[i] + 1/sizes[j])
            hsd = q_val * math.sqrt(msw / n_h)
            sig = "Yes" if diff > hsd else "No"
            print(f"G{i+1} vs G{j+1}      {diff:>8.3f} {hsd:>8.3f} {sig:>14}")

tukey_hsd(method_1, method_2, method_3)
```

---

## 10. 종합 회귀 예제

```python
import random
import math

def multiple_regression_example():
    """Demonstrate a complete regression workflow."""
    random.seed(42)
    n = 30

    # Generate data: Y = 5 + 2*X1 - 1.5*X2 + noise
    x1 = [random.uniform(1, 10) for _ in range(n)]
    x2 = [random.uniform(0, 5) for _ in range(n)]
    y = [5 + 2*x1i - 1.5*x2i + random.gauss(0, 2) for x1i, x2i in zip(x1, x2)]

    # Fit simple regression on X1 only
    result = simple_linear_regression(x1, y)
    y_hat_simple = [result['b0'] + result['b1'] * xi for xi in x1]
    r2_simple = compute_r_squared(y, y_hat_simple)

    print("=== Simple Regression (Y ~ X1) ===")
    print(f"Y = {result['b0']:.3f} + {result['b1']:.3f} * X1")
    print(f"R-squared: {r2_simple:.4f}")

    # Compute correlations
    n_vals = len(x1)
    x1_bar = sum(x1) / n_vals
    x2_bar = sum(x2) / n_vals
    y_bar = sum(y) / n_vals

    def correlation(a, b):
        a_bar = sum(a) / len(a)
        b_bar = sum(b) / len(b)
        num = sum((ai - a_bar) * (bi - b_bar) for ai, bi in zip(a, b))
        den_a = math.sqrt(sum((ai - a_bar)**2 for ai in a))
        den_b = math.sqrt(sum((bi - b_bar)**2 for bi in b))
        return num / (den_a * den_b)

    print(f"\nCorrelations:")
    print(f"  r(X1, Y) = {correlation(x1, y):.4f}")
    print(f"  r(X2, Y) = {correlation(x2, y):.4f}")
    print(f"  r(X1, X2) = {correlation(x1, x2):.4f}")

multiple_regression_example()
```

---

## 11. 핵심 요약

| 개념 | 핵심 포인트 |
|---|---|
| OLS 추정량 | $\hat{\beta}_1 = S_{XY}/S_{XX}$; 잔차 제곱합을 최소화 |
| 가우스-마르코프 | 표준 조건 하에서 OLS는 BLUE |
| $R^2$ | 설명된 분산의 비율; 모형 비교에는 수정 $R^2$ 사용 |
| 잔차 분석 | 선형성, 등분산성, 정규성, 독립성 점검 |
| 다중 회귀 | $\hat{\beta} = (X^\top X)^{-1}X^\top Y$; 자연스럽게 확장 |
| F-검정 | 예측 변수가 $Y$와 유의한 선형 관계가 있는지 검정 |
| 일원 분산분석 | $F = \text{MSB}/\text{MSW}$; SST = SSB + SSW로 분해 |
| 이원 분산분석 | 두 요인의 주효과와 교호작용 검정 |
| Tukey HSD | 족속 오류율을 통제한 쌍별 비교 |

**핵심 연결**: $k=2$개 집단에 대한 일원 분산분석은 이표본 t-검정과 동치입니다. 지시 변수를 사용한 회귀는 분산분석과 동치입니다. 이들은 동일한 선형 모형 프레임워크의 서로 다른 관점입니다.

---

**이전**: [비모수적 방법](./16_Nonparametric_Methods.md) | **다음**: [확률과정 입문](./18_Stochastic_Processes_Introduction.md)

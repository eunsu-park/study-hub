# Regression and ANOVA

**Previous**: [Nonparametric Methods](./16_Nonparametric_Methods.md) | **Next**: [Stochastic Processes Introduction](./18_Stochastic_Processes_Introduction.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate simple linear regression and derive OLS estimators
2. Interpret regression coefficients, standard errors, and $R^2$
3. Verify Gauss-Markov conditions and understand the BLUE property
4. Perform residual analysis for model diagnostics
5. Extend to multiple linear regression in matrix form
6. Conduct F-tests for overall regression significance
7. Decompose variance in one-way and two-way ANOVA
8. Apply post-hoc tests (Tukey HSD) for pairwise comparisons
9. Implement regression and ANOVA from scratch in Python

---

Regression analysis models the relationship between a response variable and one or more predictor variables. ANOVA (Analysis of Variance) is a closely related framework for comparing group means. Both are cornerstones of applied statistics.

---

## 1. Simple Linear Regression

### 1.1 The Model

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad i = 1, \ldots, n$$

- $Y_i$: response (dependent variable)
- $X_i$: predictor (independent variable)
- $\beta_0$: intercept
- $\beta_1$: slope (change in $Y$ per unit change in $X$)
- $\varepsilon_i$: random error, assumed $\varepsilon_i \sim N(0, \sigma^2)$ independently

### 1.2 Ordinary Least Squares (OLS) Estimators

Minimize the sum of squared residuals:

$$\text{RSS} = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2$$

Taking partial derivatives and setting them to zero:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n}(X_i - \bar{X})^2} = \frac{S_{XY}}{S_{XX}}$$

$$\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$

### 1.3 Properties

Under the model assumptions:
- $\hat{\beta}_0$ and $\hat{\beta}_1$ are unbiased: $E[\hat{\beta}_1] = \beta_1$, $E[\hat{\beta}_0] = \beta_0$.
- $\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{S_{XX}}$
- $\text{Var}(\hat{\beta}_0) = \sigma^2 \left(\frac{1}{n} + \frac{\bar{X}^2}{S_{XX}}\right)$
- The unbiased estimator of $\sigma^2$ is $\hat{\sigma}^2 = \frac{\text{RSS}}{n-2}$.

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

## 2. Coefficient of Determination ($R^2$)

### 2.1 Definition

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{SSR}}{\text{TSS}}$$

where:
- $\text{TSS} = \sum(Y_i - \bar{Y})^2$ (Total Sum of Squares)
- $\text{RSS} = \sum(Y_i - \hat{Y}_i)^2$ (Residual Sum of Squares)
- $\text{SSR} = \sum(\hat{Y}_i - \bar{Y})^2$ (Regression Sum of Squares)
- $\text{TSS} = \text{SSR} + \text{RSS}$

### 2.2 Interpretation

$R^2$ represents the proportion of variance in $Y$ explained by the regression model. $R^2 \in [0, 1]$ for models with an intercept.

### 2.3 Adjusted $R^2$

$R^2$ always increases when predictors are added. The adjusted version penalizes for the number of predictors $p$:

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

## 3. Gauss-Markov Conditions and BLUE

### 3.1 Gauss-Markov Assumptions

1. **Linearity**: $Y = X\beta + \varepsilon$ (the model is linear in parameters).
2. **Strict exogeneity**: $E[\varepsilon_i \mid X] = 0$.
3. **Homoscedasticity**: $\text{Var}(\varepsilon_i \mid X) = \sigma^2$ (constant variance).
4. **No autocorrelation**: $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for $i \neq j$.
5. **Full rank**: The design matrix $X$ has full column rank (no perfect multicollinearity).

### 3.2 BLUE Property

Under assumptions 1--5, OLS estimators are **BLUE** (Best Linear Unbiased Estimators):
- **Best**: Minimum variance among all linear unbiased estimators.
- **Linear**: Linear functions of $Y$.
- **Unbiased**: $E[\hat{\beta}] = \beta$.

Adding normality of errors ($\varepsilon \sim N(0, \sigma^2 I)$) makes OLS the minimum variance unbiased estimator (not just among linear ones).

---

## 4. Residual Analysis and Diagnostics

### 4.1 What to Check

Residuals $e_i = Y_i - \hat{Y}_i$ should behave like independent $N(0, \sigma^2)$ if the model is correct.

| Plot | What It Checks |
|---|---|
| Residuals vs. Fitted | Linearity, homoscedasticity |
| Q-Q plot of residuals | Normality |
| Scale-Location plot | Homoscedasticity |
| Residuals vs. order | Independence (time series) |

### 4.2 Common Problems and Solutions

| Problem | Diagnostic Sign | Remedy |
|---|---|---|
| Nonlinearity | Curved pattern in residuals vs. fitted | Add polynomial terms, transform variables |
| Heteroscedasticity | Fan shape in residuals | Weighted LS, robust standard errors, log transform |
| Non-normality | Departure in Q-Q plot | Transform response, use robust methods |
| Outliers | Large standardized residuals ($|e_i/s| > 3$) | Investigate; consider robust regression |
| Influential points | High leverage + large residual | Cook's distance; investigate data quality |

### 4.3 Leverage and Cook's Distance

**Leverage** $h_{ii}$ measures how unusual $X_i$ is. High leverage points can strongly influence the fit.

**Cook's distance** combines leverage and residual size:

$$D_i = \frac{e_i^2}{p \cdot \hat{\sigma}^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$$

Points with $D_i > 4/n$ or $D_i > 1$ deserve scrutiny.

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

## 5. Multiple Linear Regression

### 5.1 Matrix Formulation

$$Y = X\beta + \varepsilon$$

where $Y$ is $n \times 1$, $X$ is $n \times (p+1)$ (including intercept column), $\beta$ is $(p+1) \times 1$, and $\varepsilon$ is $n \times 1$.

### 5.2 OLS Solution

$$\hat{\beta} = (X^\top X)^{-1} X^\top Y$$

This minimizes $\|Y - X\beta\|^2$. The solution exists when $X^\top X$ is invertible (full rank condition).

### 5.3 Properties

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

## 6. F-Test for Overall Significance

### 6.1 Hypotheses

$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$ (no linear relationship)

$H_1:$ At least one $\beta_j \neq 0$

### 6.2 F-Statistic

$$F = \frac{\text{SSR}/p}{\text{RSS}/(n-p-1)} = \frac{\text{MSR}}{\text{MSE}} \sim F_{p, n-p-1} \quad \text{under } H_0$$

Reject $H_0$ when $F > F_{\alpha, p, n-p-1}$.

### 6.3 Relationship to $R^2$

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

## 7. One-Way ANOVA

### 7.1 Setting

Compare $k$ group means: $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$ vs. $H_1$: at least two means differ.

### 7.2 Variance Decomposition

$$\text{SST} = \text{SSB} + \text{SSW}$$

- **SST** (Total): $\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{\cdot\cdot})^2$
- **SSB** (Between groups): $\sum_{i=1}^{k} n_i(\bar{Y}_{i\cdot} - \bar{Y}_{\cdot\cdot})^2$
- **SSW** (Within groups): $\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{i\cdot})^2$

### 7.3 F-Statistic

$$F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB}/(k-1)}{\text{SSW}/(N-k)} \sim F_{k-1, N-k} \quad \text{under } H_0$$

### 7.4 ANOVA Table

| Source | SS | df | MS | F |
|---|---|---|---|---|
| Between | SSB | $k-1$ | SSB/$(k-1)$ | MSB/MSW |
| Within | SSW | $N-k$ | SSW/$(N-k)$ | |
| Total | SST | $N-1$ | | |

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

## 8. Two-Way ANOVA

### 8.1 Model

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}$$

where $\alpha_i$ is the effect of factor A (level $i$), $\beta_j$ is the effect of factor B (level $j$), and $(\alpha\beta)_{ij}$ is the interaction effect.

### 8.2 Decomposition

$$\text{SST} = \text{SS}_A + \text{SS}_B + \text{SS}_{AB} + \text{SS}_E$$

Three F-tests are performed:
- $F_A = \text{MS}_A / \text{MS}_E$ for the main effect of A
- $F_B = \text{MS}_B / \text{MS}_E$ for the main effect of B
- $F_{AB} = \text{MS}_{AB} / \text{MS}_E$ for the interaction

### 8.3 Interpreting Interactions

When the interaction $(\alpha\beta)_{ij}$ is significant, the effect of one factor depends on the level of the other. In this case, main effects alone are not sufficient to describe the data; interaction plots (profile plots) should be examined.

---

## 9. Post-Hoc Tests: Tukey HSD

### 9.1 Why Post-Hoc?

ANOVA tells us that at least two means differ, but not which pairs differ. Post-hoc tests make pairwise comparisons while controlling the family-wise error rate.

### 9.2 Tukey's Honestly Significant Difference

For comparing means $\bar{Y}_i$ and $\bar{Y}_j$:

$$|\bar{Y}_i - \bar{Y}_j| > q_{\alpha, k, N-k} \cdot \sqrt{\frac{\text{MSW}}{n}}$$

where $q_{\alpha, k, N-k}$ is the studentized range critical value (for equal group sizes $n$). The pair is declared significantly different if the inequality holds.

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

## 10. Complete Regression Example

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

## 11. Key Takeaways

| Concept | Key Point |
|---|---|
| OLS estimators | $\hat{\beta}_1 = S_{XY}/S_{XX}$; minimizes sum of squared residuals |
| Gauss-Markov | Under standard conditions, OLS is BLUE |
| $R^2$ | Proportion of variance explained; use adjusted $R^2$ for model comparison |
| Residual analysis | Check linearity, homoscedasticity, normality, independence |
| Multiple regression | $\hat{\beta} = (X^\top X)^{-1}X^\top Y$; extends naturally |
| F-test | Tests whether any predictor has a significant linear relationship with $Y$ |
| One-way ANOVA | $F = \text{MSB}/\text{MSW}$; decomposes SST = SSB + SSW |
| Two-way ANOVA | Tests main effects and interaction between two factors |
| Tukey HSD | Pairwise comparisons with family-wise error rate control |

**Key connection**: One-way ANOVA with $k=2$ groups is equivalent to a two-sample t-test. Regression with indicator variables is equivalent to ANOVA. They are different perspectives on the same linear model framework.

---

**Previous**: [Nonparametric Methods](./16_Nonparametric_Methods.md) | **Next**: [Stochastic Processes Introduction](./18_Stochastic_Processes_Introduction.md)

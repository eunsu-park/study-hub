# 결합 분포

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 두 개 이상의 확률변수에 대한 결합 PMF와 결합 PDF를 정의하고 활용하기
2. 결합 분포로부터 주변 분포 유도하기
3. 조건부 분포를 계산하고 해석하기
4. 인수분해 기준을 사용하여 확률변수의 독립성 검정하기
5. 공분산과 피어슨 상관계수 계산하기
6. 반복 기댓값의 법칙(tower property) 적용하기
7. Python으로 결합 분포 계산 구현하기

---

## 개요

현실 세계의 문제는 여러 확률변수를 동시에 다룹니다. 결합 분포(joint distribution)는 두 개 이상의 확률변수 간의 전체 확률적 관계를 포함하여 종속 구조까지 파악합니다. 결합 분포로부터 주변 분포(marginal, 개별 변수의 분포), 조건부 분포(conditional, 다른 변수에 대한 정보가 주어졌을 때 한 변수의 행동), 그리고 공분산(covariance)과 상관(correlation)과 같은 요약 측도를 추출할 수 있습니다.

---

## 목차

1. [결합 PMF (이산 경우)](#1-결합-pmf-이산-경우)
2. [결합 PDF (연속 경우)](#2-결합-pdf-연속-경우)
3. [주변 분포](#3-주변-분포)
4. [조건부 분포](#4-조건부-분포)
5. [확률변수의 독립성](#5-확률변수의-독립성)
6. [공분산과 상관](#6-공분산과-상관)
7. [조건부 기댓값과 탑 성질](#7-조건부-기댓값과-탑-성질)
8. [Python 예제](#8-python-예제)
9. [핵심 요약](#9-핵심-요약)

---

## 1. 결합 PMF (이산 경우)

### 정의

이산 확률변수 $X$와 $Y$에 대해, **결합 PMF**(joint PMF)는:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

### 성질

1. **비음성**: 모든 $x, y$에 대해 $p_{X,Y}(x, y) \geq 0$
2. **정규화**: $\sum_{x}\sum_{y} p_{X,Y}(x, y) = 1$
3. **집합의 확률**: $P((X,Y) \in A) = \sum_{(x,y) \in A} p_{X,Y}(x, y)$

### 예제: 결합 PMF 표

동전 두 개를 고려합니다: $X$ = 동전 1(공정)의 앞면 수, $Y$ = 동전 2(편향, $P(H) = 0.7$)의 앞면 수.

|  | $Y=0$ | $Y=1$ | 주변 $p_X$ |
|---|---|---|---|
| $X=0$ | 0.15 | 0.35 | 0.50 |
| $X=1$ | 0.15 | 0.35 | 0.50 |
| 주변 $p_Y$ | 0.30 | 0.70 | 1.00 |

여기서 $X$와 $Y$는 독립입니다 (5절에서 공식적으로 검증합니다).

---

## 2. 결합 PDF (연속 경우)

### 정의

연속 확률변수 $X$와 $Y$에 대해, **결합 PDF** $f_{X,Y}(x, y)$는 다음을 만족합니다:

$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$

### 성질

1. **비음성**: 모든 $x, y$에 대해 $f_{X,Y}(x, y) \geq 0$
2. **정규화**: $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$

### 예제: 단위 정사각형 위의 균등분포

$$f_{X,Y}(x, y) = \begin{cases} 1 & \text{if } 0 \leq x \leq 1 \text{ and } 0 \leq y \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

$$P(X + Y \leq 1) = \int_0^1 \int_0^{1-x} 1 \, dy \, dx = \int_0^1 (1-x) \, dx = \frac{1}{2}$$

### 결합 CDF

$$F_{X,Y}(x, y) = P(X \leq x, Y \leq y)$$

연속 경우: $f_{X,Y}(x, y) = \frac{\partial^2}{\partial x \, \partial y} F_{X,Y}(x, y)$.

---

## 3. 주변 분포

### 이산 경우

$X$의 **주변 PMF**(marginal PMF)는 $Y$의 모든 값에 대해 합산하여 얻습니다:

$$p_X(x) = \sum_{y} p_{X,Y}(x, y)$$

마찬가지로: $p_Y(y) = \sum_{x} p_{X,Y}(x, y)$

### 연속 경우

$X$의 **주변 PDF**(marginal PDF)는 $Y$를 적분하여 얻습니다:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy$$

### 예제: 비균등 결합 분포

$f_{X,Y}(x,y) = 6(1-y)$이고 $0 \leq x \leq y \leq 1$이라 하겠습니다.

$X$의 주변분포:

$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[y - \frac{y^2}{2}\right]_x^1 = 6\left[\frac{1}{2} - x + \frac{x^2}{2}\right] = 3(1 - x)^2$$

$Y$의 주변분포:

$$f_Y(y) = \int_0^y 6(1-y) \, dx = 6y(1-y), \quad 0 \leq y \leq 1$$

---

## 4. 조건부 분포

### 이산 경우

$X = x$가 주어졌을 때 $Y$의 **조건부 PMF** ($p_X(x) > 0$인 경우):

$$p_{Y|X}(y \mid x) = \frac{p_{X,Y}(x, y)}{p_X(x)}$$

### 연속 경우

$X = x$가 주어졌을 때 $Y$의 **조건부 PDF** ($f_X(x) > 0$인 경우):

$$f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x, y)}{f_X(x)}$$

### 성질

조건부 분포는 올바른 분포입니다:

- $p_{Y|X}(y \mid x) \geq 0$이고 $\sum_y p_{Y|X}(y \mid x) = 1$ (이산)
- $f_{Y|X}(y \mid x) \geq 0$이고 $\int f_{Y|X}(y \mid x) \, dy = 1$ (연속)

### 예제

결합 PDF $f_{X,Y}(x,y) = 6(1-y)$ ($0 \leq x \leq y \leq 1$)를 사용하면:

$$f_{Y|X}(y \mid x) = \frac{6(1-y)}{3(1-x)^2} = \frac{2(1-y)}{(1-x)^2}, \quad x \leq y \leq 1$$

이것이 $X = x$가 주어졌을 때 $Y$의 조건부 밀도입니다. $x$에 따라 달라짐에 주목하세요.

---

## 5. 확률변수의 독립성

### 정의

확률변수 $X$와 $Y$가 **독립**(independent)인 것은, 결합 분포가 주변 분포의 곱으로 인수분해되는 것과 동치입니다:

**이산**: 모든 $x, y$에 대해 $p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y)$

**연속**: 모든 $x, y$에 대해 $f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)$

### 동치 조건

다음 중 하나가 성립하면 나머지도 성립합니다:

1. 모든 $x, y$에 대해 $F_{X,Y}(x, y) = F_X(x) \cdot F_Y(y)$
2. 위와 같이 결합 PMF/PDF가 인수분해
3. 모든 $x, y$에 대해 $f_{Y|X}(y \mid x) = f_Y(y)$ (조건부 = 주변)
4. 모든 함수 $g, h$에 대해 $E[g(X)h(Y)] = E[g(X)] \cdot E[h(Y)]$

### 독립성 확인: 인수분해 기준

연속 $(X, Y)$에 대해, $f_{X,Y}(x, y)$가 어떤 함수 $g(x) \cdot h(y)$로 쓸 수 있고 지지(support)가 직적(Cartesian product)이면 독립입니다.

**예제**: $f_{X,Y}(x,y) = 6(1-y)$이고 $\{0 \leq x \leq y \leq 1\}$. 지지가 직사각형이 **아니고** ($x \leq y$ 관계에 의존), $X$와 $Y$는 **독립이 아닙니다**.

---

## 6. 공분산과 상관

### 공분산

$X$와 $Y$의 **공분산**(covariance)은 선형 연관성을 측정합니다:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

### 공분산의 성질

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (대칭)
3. $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$ (쌍선형성)
5. $X \perp Y$이면 $\text{Cov}(X, Y) = 0$

**주의**: $\text{Cov}(X, Y) = 0$은 독립을 의미하지 **않습니다**. 비상관(uncorrelated) 변수라도 여전히 종속일 수 있습니다.

**고전적 반례**: $X \sim \text{Uniform}(-1, 1)$이고 $Y = X^2$이면, $\text{Cov}(X, Y) = E[X^3] - E[X]E[X^2] = 0 - 0 = 0$이지만, $Y$는 $X$에 의해 완전히 결정됩니다.

### 상관계수

**피어슨 상관계수**(Pearson correlation coefficient)는 공분산을 $[-1, 1]$로 정규화합니다:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- $\rho = 1$: 완전 양의 선형 관계 ($Y = aX + b$, $a > 0$)
- $\rho = -1$: 완전 음의 선형 관계 ($Y = aX + b$, $a < 0$)
- $\rho = 0$: 비상관 (선형 관계 없음, 그러나 비선형 종속은 가능)

### 합의 분산 (일반적 경우)

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

더 일반적으로:

$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)$$

---

## 7. 조건부 기댓값과 탑 성질

### 조건부 기댓값

$X = x$가 주어졌을 때 $Y$의 **조건부 기댓값**(conditional expectation)은:

**이산**: $E[Y \mid X = x] = \sum_y y \, p_{Y|X}(y \mid x)$

**연속**: $E[Y \mid X = x] = \int_{-\infty}^{\infty} y \, f_{Y|X}(y \mid x) \, dy$

$E[Y \mid X]$는 (확률변수 $X$의 함수로서) 그 자체가 확률변수입니다.

### 반복 기댓값의 법칙 (탑 성질)

$$E[Y] = E\big[E[Y \mid X]\big]$$

이산 경우:

$$E[Y] = \sum_x E[Y \mid X = x] \, p_X(x)$$

이것은 확률론에서 가장 강력한 도구 중 하나입니다. 다른 변수에 대해 먼저 조건을 걸어 복잡한 기댓값을 분해할 수 있습니다.

### 예제: 무작위 횟수의 동전 던지기

$N \sim \text{Poisson}(\lambda)$이 공정한 동전을 던지는 횟수이고, $Y$ = 앞면의 수라 하겠습니다.

$N = n$이 주어지면: $Y \mid N = n \sim \text{Binomial}(n, 0.5)$이므로, $E[Y \mid N = n] = 0.5n$.

탑 성질에 의해:

$$E[Y] = E[E[Y \mid N]] = E[0.5N] = 0.5 \, E[N] = 0.5\lambda$$

### 조건부 분산 공식

$$\text{Var}(Y) = E[\text{Var}(Y \mid X)] + \text{Var}(E[Y \mid X])$$

이것은 전체 분산을 다음으로 분해합니다:

- **그룹 내 분산**(within-group variance): $E[\text{Var}(Y \mid X)]$ (각 $X$ 수준 내의 평균 분산)
- **그룹 간 분산**(between-group variance): $\text{Var}(E[Y \mid X])$ (그룹 평균의 변동성)

---

## 8. Python 예제

### 결합 PMF 표

```python
def joint_pmf_example():
    """Work with a joint PMF stored as a 2D dictionary."""
    # Joint distribution of X (rows) and Y (columns)
    # X: number of defective items in batch (0, 1, 2)
    # Y: number returned by customer (0, 1)
    joint = {
        (0, 0): 0.40, (0, 1): 0.00,
        (1, 0): 0.15, (1, 1): 0.20,
        (2, 0): 0.05, (2, 1): 0.20,
    }

    # Verify normalization
    total = sum(joint.values())
    print(f"Sum of joint PMF: {total:.2f}")

    # Marginal of X
    x_values = sorted(set(x for x, y in joint))
    y_values = sorted(set(y for x, y in joint))

    print("\nMarginal of X:")
    p_x = {}
    for x in x_values:
        p_x[x] = sum(joint.get((x, y), 0) for y in y_values)
        print(f"  P(X={x}) = {p_x[x]:.2f}")

    # Marginal of Y
    print("\nMarginal of Y:")
    p_y = {}
    for y in y_values:
        p_y[y] = sum(joint.get((x, y), 0) for x in x_values)
        print(f"  P(Y={y}) = {p_y[y]:.2f}")

    # Conditional distribution: P(Y|X=1)
    print("\nConditional P(Y | X=1):")
    for y in y_values:
        cond = joint.get((1, y), 0) / p_x[1]
        print(f"  P(Y={y} | X=1) = {cond:.4f}")

    # Check independence: P(X=x, Y=y) == P(X=x)*P(Y=y)?
    print("\nIndependence check:")
    independent = True
    for x in x_values:
        for y in y_values:
            product = p_x[x] * p_y[y]
            actual = joint.get((x, y), 0)
            match = abs(actual - product) < 1e-10
            if not match:
                independent = False
            print(f"  P(X={x},Y={y})={actual:.2f}  vs  P(X={x})*P(Y={y})={product:.4f}  {'OK' if match else 'MISMATCH'}")
    print(f"  Independent? {independent}")

joint_pmf_example()
```

### 시뮬레이션으로 공분산과 상관 추정

```python
import random
import math

def covariance_simulation(n=500000):
    """Estimate covariance and correlation from simulated data."""
    random.seed(42)
    xs = []
    ys = []

    for _ in range(n):
        x = random.gauss(10, 3)      # X ~ Normal(10, 9)
        y = 2 * x + random.gauss(0, 2)  # Y = 2X + noise
        xs.append(x)
        ys.append(y)

    # Means
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    # Variances
    var_x = sum((x - mean_x)**2 for x in xs) / (n - 1)
    var_y = sum((y - mean_y)**2 for y in ys) / (n - 1)

    # Covariance
    cov_xy = sum((x - mean_x) * (y - mean_y)
                 for x, y in zip(xs, ys)) / (n - 1)

    # Correlation
    rho = cov_xy / (math.sqrt(var_x) * math.sqrt(var_y))

    print(f"E[X]     = {mean_x:.4f}  (theoretical: 10)")
    print(f"E[Y]     = {mean_y:.4f}  (theoretical: 20)")
    print(f"Var(X)   = {var_x:.4f}  (theoretical: 9)")
    print(f"Var(Y)   = {var_y:.4f}  (theoretical: 4*9+4 = 40)")
    print(f"Cov(X,Y) = {cov_xy:.4f}  (theoretical: 2*9 = 18)")
    print(f"rho(X,Y) = {rho:.4f}  (theoretical: 18/sqrt(9*40) ~ 0.9487)")

covariance_simulation()
```

### 비상관이지만 종속

```python
import random

def uncorrelated_dependent(n=500000):
    """Demonstrate Cov=0 does not imply independence."""
    random.seed(10)
    xs = []
    ys = []

    for _ in range(n):
        x = random.uniform(-1, 1)  # X ~ Uniform(-1, 1)
        y = x ** 2                 # Y = X^2 (fully dependent!)
        xs.append(x)
        ys.append(y)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov_xy = sum((x - mean_x) * (y - mean_y)
                 for x, y in zip(xs, ys)) / (n - 1)

    print(f"E[X]     = {mean_x:.6f}  (theoretical: 0)")
    print(f"E[Y]     = {mean_y:.6f}  (theoretical: 1/3)")
    print(f"Cov(X,Y) = {cov_xy:.6f}  (theoretical: 0)")
    print(f"\nY = X^2, so Y is completely determined by X,")
    print(f"yet Cov(X,Y) ~ 0. Uncorrelated != Independent!")

uncorrelated_dependent()
```

### 반복 기댓값의 법칙

```python
import random
import math

def tower_property_demo(lam=5.0, p=0.5, n=200000):
    """Demonstrate E[Y] = E[E[Y|N]] with N ~ Poisson, Y|N ~ Binomial."""
    random.seed(42)
    ys = []

    for _ in range(n):
        # Generate N ~ Poisson(lam) using inverse transform
        L = math.exp(-lam)
        k = 0
        prob = 1.0
        while prob > L:
            k += 1
            prob *= random.random()
        nn = k - 1  # Poisson sample

        # Given N=nn, generate Y ~ Binomial(nn, p)
        y = sum(1 for _ in range(nn) if random.random() < p)
        ys.append(y)

    empirical_mean = sum(ys) / n
    theoretical_mean = lam * p  # E[Y] = E[E[Y|N]] = E[pN] = p*lam

    print(f"Tower property: E[Y] = E[E[Y|N]] = p * lambda")
    print(f"  lambda = {lam}, p = {p}")
    print(f"  Theoretical E[Y] = {theoretical_mean:.4f}")
    print(f"  Empirical   E[Y] = {empirical_mean:.4f}")

tower_property_demo()
```

### 조건부 분산 공식

```python
import random
import math

def conditional_variance_demo(n=200000):
    """Verify Var(Y) = E[Var(Y|X)] + Var(E[Y|X])."""
    random.seed(55)
    xs = []
    ys = []

    for _ in range(n):
        # X takes values 1, 2, 3 equally likely
        x = random.choice([1, 2, 3])
        # Y | X=x ~ Normal(x, x)  (mean=x, variance=x)
        y = random.gauss(x, math.sqrt(x))
        xs.append(x)
        ys.append(y)

    # Total variance
    mean_y = sum(ys) / n
    var_y_total = sum((y - mean_y)**2 for y in ys) / (n - 1)

    # E[Var(Y|X)]: average of within-group variances
    # Var(E[Y|X]): variance of group means
    group_means = {}
    group_vars = {}
    for x_val in [1, 2, 3]:
        group = [y for x, y in zip(xs, ys) if x == x_val]
        gm = sum(group) / len(group)
        gv = sum((y - gm)**2 for y in group) / (len(group) - 1)
        group_means[x_val] = gm
        group_vars[x_val] = gv

    e_var_y_given_x = sum(group_vars[x] for x in [1, 2, 3]) / 3
    overall_mean_of_means = sum(group_means[x] for x in [1, 2, 3]) / 3
    var_e_y_given_x = sum((group_means[x] - overall_mean_of_means)**2
                          for x in [1, 2, 3]) / 2  # sample variance

    # Theoretical: E[Var(Y|X)] = E[X] = 2, Var(E[Y|X]) = Var(X) = 2/3
    print(f"Total Var(Y)     = {var_y_total:.4f}  (theoretical: E[X]+Var(X) = 2+2/3 ~ 2.6667)")
    print(f"E[Var(Y|X)]      = {e_var_y_given_x:.4f}  (theoretical: 2.0)")
    print(f"Var(E[Y|X])      = {var_e_y_given_x:.4f}  (theoretical: 2/3 ~ 0.6667)")
    print(f"Sum              = {e_var_y_given_x + var_e_y_given_x:.4f}")

conditional_variance_demo()
```

---

## 9. 핵심 요약

1. **결합 분포**는 두 개 이상의 확률변수의 동시적 행동을 기술합니다. 결합 PMF 또는 PDF는 개별 주변 분포보다 엄격하게 더 많은 정보를 담고 있습니다.

2. **주변 분포**는 다른 변수를 합산(이산) 또는 적분(연속)하여 얻습니다. 이 과정에서 종속성에 대한 정보가 소실됩니다.

3. **조건부 분포**는 다른 변수에 대한 지식이 주어졌을 때 한 변수의 행동을 기술합니다: $f_{Y|X}(y|x) = f_{X,Y}(x,y)/f_X(x)$.

4. **독립**은 결합이 주변의 곱으로 인수분해됨을 의미합니다: $f_{X,Y}(x,y) = f_X(x)f_Y(y)$. 이것은 "관계 없음"의 가장 강한 형태입니다.

5. **공분산**은 선형 연관성을 측정합니다: $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$. **상관** $\rho$는 이를 $[-1, 1]$로 정규화합니다. 공분산이 0이라고 독립을 의미하지는 **않습니다**.

6. **반복 기댓값의 법칙** $E[Y] = E[E[Y|X]]$은 강력한 분해 도구입니다. 분산 유사체인 $\text{Var}(Y) = E[\text{Var}(Y|X)] + \text{Var}(E[Y|X])$는 전체 변동성을 그룹 내 성분과 그룹 간 성분으로 분해합니다.

---

*이전: [04 - 기댓값과 적률](./04_Expectation_and_Moments.md) | 다음: [06 - 이산 분포족](./06_Discrete_Distribution_Families.md)*

# 연속 분포족

**이전**: [이산 분포족](./06_Discrete_Distribution_Families.md) | **다음**: [확률변수의 변환](./08_Transformations_of_Random_Variables.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 각 주요 연속 분포의 PDF, CDF, 평균, 분산, MGF 서술하기
2. 균등분포, 정규분포, 지수분포, 감마분포를 실제 현상 모델링에 적용하기
3. 지수분포의 무기억 성질 설명하기
4. 베타분포를 비율의 자연스러운 사전분포로 기술하기
5. 카이제곱, 스튜던트 t, F 분포가 정규분포로부터 어떻게 유도되는지 서술하기
6. 이들 분포족을 연결하는 관계를 매핑하기

---

연속 분포(continuous distribution)는 개별 점이 아닌 실수의 구간에 확률을 부여합니다. 이 레슨에서는 가장 중요한 연속 분포족을 조사하며, 각각은 $f(x) \ge 0$이고 $\int_{-\infty}^{\infty} f(x)\,dx = 1$을 만족하는 확률 밀도 함수(PDF)로 특성화됩니다.

---

## 1. 균등분포 -- $X \sim \text{Uniform}(a, b)$

### 1.1 정의와 PDF

가장 단순한 연속 분포로, 유한 구간 $[a, b]$ 위에 동일한 밀도를 부여합니다.

$$f(x) = \frac{1}{b - a}, \quad a \le x \le b$$

### 1.2 CDF

$$F(x) = \begin{cases} 0 & x < a \\ \frac{x - a}{b - a} & a \le x \le b \\ 1 & x > b \end{cases}$$

### 1.3 적률과 MGF

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = \frac{a + b}{2}$ |
| 분산 | $\text{Var}(X) = \frac{(b - a)^2}{12}$ |
| MGF | $M_X(t) = \frac{e^{tb} - e^{ta}}{t(b - a)}$ ($t \ne 0$) |

### 1.4 활용 사례

- 매개변수 위치에 대한 완전한 무지(ignorance) 모델링
- 난수 생성: 대부분의 의사 난수 생성기는 $\text{Uniform}(0,1)$을 생산
- 역변환 표집법은 $U \sim \text{Uniform}(0,1)$에서 시작

---

## 2. 정규(가우시안) 분포 -- $X \sim N(\mu, \sigma^2)$

### 2.1 PDF

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad -\infty < x < \infty$$

### 2.2 표준 정규

$\mu = 0$이고 $\sigma^2 = 1$이면, $Z \sim N(0,1)$로 씁니다. 임의의 정규 변수는 표준화할 수 있습니다:

$$Z = \frac{X - \mu}{\sigma}$$

표준 정규의 CDF는 $\Phi(z)$로 표기하며 폐쇄형(closed-form) 표현이 없습니다.

### 2.3 68-95-99.7 법칙

정규분포에서:

- $P(\mu - \sigma \le X \le \mu + \sigma) \approx 0.6827$
- $P(\mu - 2\sigma \le X \le \mu + 2\sigma) \approx 0.9545$
- $P(\mu - 3\sigma \le X \le \mu + 3\sigma) \approx 0.9973$

이 경험적 법칙은 표 없이 확률을 빠르게 평가하는 방법을 제공합니다.

### 2.4 CDF

$$F(x) = \Phi\!\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]$$

### 2.5 적률과 MGF

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = \mu$ |
| 분산 | $\text{Var}(X) = \sigma^2$ |
| MGF | $M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$ |

### 2.6 핵심 성질

- **선형 변환에 대한 폐쇄성**: $X \sim N(\mu, \sigma^2)$이면 $aX + b \sim N(a\mu + b,\, a^2\sigma^2)$.
- **독립 합의 합**: $X_i \sim N(\mu_i, \sigma_i^2)$가 독립이면, $\sum X_i \sim N\!\left(\sum \mu_i, \sum \sigma_i^2\right)$.
- 정규분포는 주어진 평균과 분산에 대해 **최대 엔트로피**(maximum entropy) 분포입니다.

---

## 3. 지수분포 -- $X \sim \text{Exp}(\lambda)$

### 3.1 PDF와 CDF

$$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0$$

$$F(x) = 1 - e^{-\lambda x}, \quad x \ge 0$$

여기서 $\lambda > 0$는 비율 매개변수(rate parameter)입니다. 일부 교재에서는 평균 $\beta = 1/\lambda$로 매개변수화합니다.

### 3.2 적률과 MGF

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = 1/\lambda$ |
| 분산 | $\text{Var}(X) = 1/\lambda^2$ |
| MGF | $M_X(t) = \frac{\lambda}{\lambda - t}$ ($t < \lambda$) |

### 3.3 무기억 성질

지수분포는 무기억 성질(memoryless property)을 갖는 **유일한** 연속 분포입니다:

$$P(X > s + t \mid X > s) = P(X > t) \quad \text{(모든 } s, t \ge 0\text{에 대해)}$$

이것은 잔여 수명이 이미 경과한 시간에 의존하지 않음을 의미합니다.

### 3.4 포아송 과정과의 연결

사건이 비율 $\lambda$의 포아송 과정(Poisson process)에 따라 발생하면, 연속 사건 간의 대기 시간은 $\text{Exp}(\lambda)$를 따릅니다. 동치로, 길이 $t$인 시간 구간에서의 사건 수는 $\text{Poisson}(\lambda t)$입니다.

---

## 4. 감마분포 -- $X \sim \text{Gamma}(\alpha, \beta)$

### 4.1 PDF

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0$$

여기서 $\alpha > 0$는 형상 매개변수(shape parameter), $\beta > 0$는 비율 매개변수(rate parameter), $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt$는 감마 함수(gamma function)입니다.

### 4.2 CDF

일반적으로 폐쇄형 표현이 없으며, 하부 불완전 감마 함수(lower incomplete gamma function)를 통해 계산합니다:

$$F(x) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)}$$

### 4.3 적률과 MGF

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = \alpha / \beta$ |
| 분산 | $\text{Var}(X) = \alpha / \beta^2$ |
| MGF | $M_X(t) = \left(\frac{\beta}{\beta - t}\right)^\alpha$ ($t < \beta$) |

### 4.4 특수한 경우

- **지수분포**: $\text{Gamma}(1, \lambda) = \text{Exp}(\lambda)$
- **얼랑 분포**(Erlang distribution): $\alpha = n$이 양의 정수일 때. 포아송 과정에서 $n$번째 사건까지의 대기 시간을 모델링합니다.
- **카이제곱**: $\text{Gamma}(k/2, 1/2) = \chi^2(k)$

### 4.5 가법 성질

$X_1 \sim \text{Gamma}(\alpha_1, \beta)$이고 $X_2 \sim \text{Gamma}(\alpha_2, \beta)$가 독립 (같은 비율)이면:

$$X_1 + X_2 \sim \text{Gamma}(\alpha_1 + \alpha_2, \beta)$$

---

## 5. 베타분포 -- $X \sim \text{Beta}(\alpha, \beta)$

### 5.1 PDF

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 < x < 1$$

여기서 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$는 베타 함수(beta function)입니다.

### 5.2 적률

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = \frac{\alpha}{\alpha + \beta}$ |
| 분산 | $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

### 5.3 특수한 경우와 형상

- $\text{Beta}(1, 1) = \text{Uniform}(0, 1)$
- $\alpha = \beta$: $1/2$에 대해 대칭
- $\alpha > \beta$: 왼쪽으로 치우침 (1 근처에 질량)
- $\alpha < \beta$: 오른쪽으로 치우침 (0 근처에 질량)
- $\alpha, \beta < 1$: U자형 (양 극단에 질량)

### 5.4 활용 사례

베타분포는 베르누이 및 이항 가능도(likelihood)에 대한 **켤레 사전분포**(conjugate prior)입니다. 사전분포가 $p \sim \text{Beta}(\alpha, \beta)$이고 $n$번의 시행에서 $k$번 성공을 관측하면, 사후분포는:

$$p \mid \text{data} \sim \text{Beta}(\alpha + k, \beta + n - k)$$

이것이 베이즈 통계에서 미지의 비율을 모델링하는 표준 선택이 되는 이유입니다.

---

## 6. 카이제곱 분포 -- $X \sim \chi^2(k)$

### 6.1 정의

$Z_1, Z_2, \ldots, Z_k$가 독립 표준 정규 변수이면:

$$X = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2(k)$$

매개변수 $k$를 **자유도**(degrees of freedom)라 합니다.

### 6.2 PDF

$$f(x) = \frac{1}{2^{k/2}\,\Gamma(k/2)} x^{k/2 - 1} e^{-x/2}, \quad x > 0$$

이것은 단순히 $\text{Gamma}(k/2, 1/2)$입니다.

### 6.3 적률과 MGF

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = k$ |
| 분산 | $\text{Var}(X) = 2k$ |
| MGF | $M_X(t) = (1 - 2t)^{-k/2}$ ($t < 1/2$) |

### 6.4 가법 성질

$X_1 \sim \chi^2(k_1)$이고 $X_2 \sim \chi^2(k_2)$가 독립이면, $X_1 + X_2 \sim \chi^2(k_1 + k_2)$.

### 6.5 응용

- 적합도 검정 (피어슨의 카이제곱 검정)
- 정규 모집단 분산에 대한 신뢰 구간 구성
- 표본 분산: $X_i \sim N(\mu, \sigma^2)$이면, $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$

---

## 7. 스튜던트 t-분포 -- $T \sim t(\nu)$

### 7.1 정의

$Z \sim N(0,1)$이고 $V \sim \chi^2(\nu)$가 독립이면:

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

매개변수 $\nu$는 자유도입니다.

### 7.2 PDF

$$f(t) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}, \quad -\infty < t < \infty$$

### 7.3 적률

| 성질 | 값 |
|------|-----|
| 평균 | $E[T] = 0$ ($\nu > 1$) |
| 분산 | $\text{Var}(T) = \frac{\nu}{\nu - 2}$ ($\nu > 2$) |

MGF는 $\nu > 1$에서만 존재하며 단순한 폐쇄형이 없습니다.

### 7.4 핵심 성질

- **두꺼운 꼬리**: $t$-분포는 표준 정규보다 꼬리가 두꺼워, 극단값이 더 가능합니다.
- **수렴**: $\nu \to \infty$이면, $t(\nu) \to N(0,1)$. 실제로 $\nu \ge 30$이면 $t$는 표준 정규로 잘 근사됩니다.
- **응용**: **분산을 모르고** 표본 크기가 작을 때, 정규 모집단의 평균을 추정하는 데 사용됩니다.

---

## 8. F-분포 -- $F \sim F(d_1, d_2)$

### 8.1 정의

$U \sim \chi^2(d_1)$이고 $V \sim \chi^2(d_2)$가 독립이면:

$$F = \frac{U/d_1}{V/d_2} \sim F(d_1, d_2)$$

### 8.2 PDF

$$f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x\,B(d_1/2, d_2/2)}, \quad x > 0$$

### 8.3 적률

| 성질 | 값 |
|------|-----|
| 평균 | $E[F] = \frac{d_2}{d_2 - 2}$ ($d_2 > 2$) |
| 분산 | $\text{Var}(F) = \frac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2-2)^2(d_2-4)}$ ($d_2 > 4$) |

### 8.4 t-분포와의 연결

$T \sim t(\nu)$이면, $T^2 \sim F(1, \nu)$.

### 8.5 응용

- **분산분석**(ANOVA, Analysis of Variance): 여러 그룹 간 평균 비교
- **회귀**: 예측 변수 집합이 전체적으로 유의한지 검정
- 두 모집단 분산 비교

---

## 9. 분포족 간의 관계

분포가 어떻게 연결되는지 이해하는 것은 이론과 실무 모두에 필수적입니다.

```
Bernoulli(p) --합--> Binomial(n,p) --CLT--> Normal(np, np(1-p))
                                                   |
Poisson(λ) --대기--> Exponential(λ) = Gamma(1,λ)   |
                                  |                 |
                             Gamma(α,β) <-- Exp의 합  |
                                  |                 |
                           Chi-squared(k) = Gamma(k/2, 1/2)
                                  |                 |
                                  +-----> t(ν) = Z / sqrt(χ²/ν)
                                  |
                                  +-----> F(d1,d2) = (χ²₁/d1) / (χ²₂/d2)

Uniform(0,1) --역CDF--> 임의의 분포
Beta(1,1) = Uniform(0,1)
```

핵심 관계 요약:

1. $\text{Exp}(\lambda)$는 $\text{Gamma}(1, \lambda)$
2. $\chi^2(k)$는 $\text{Gamma}(k/2, 1/2)$
3. 독립 $\chi^2$의 합은 $\chi^2$ (자유도 가법)
4. $t(\nu)$는 정규와 $\sqrt{\chi^2/\nu}$의 비
5. $F(d_1, d_2)$는 두 독립 $\chi^2$ 변수의 비 (각각 자유도로 나눔)
6. $\nu \to \infty$일 때, $t(\nu) \to N(0,1)$
7. $\text{Beta}(1,1) = \text{Uniform}(0,1)$

---

## 10. Python 예제

### 10.1 주요 분포의 PDF 계산

```python
import math

def normal_pdf(x, mu=0.0, sigma=1.0):
    """Standard or general normal PDF."""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    return coeff * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def exponential_pdf(x, lam=1.0):
    """Exponential PDF with rate parameter lambda."""
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)

def uniform_pdf(x, a=0.0, b=1.0):
    """Uniform PDF on [a, b]."""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

# Evaluate normal PDF at several points
xs = [mu * 0.5 for mu in range(-8, 9)]  # -4.0 to 4.0
for x in xs:
    print(f"  N(0,1) at x={x:5.1f}: {normal_pdf(x):.6f}")
```

### 10.2 68-95-99.7 법칙 시뮬레이션 검증

```python
import random
import math

random.seed(42)
n = 100_000
mu, sigma = 5.0, 2.0
samples = [random.gauss(mu, sigma) for _ in range(n)]

within_1 = sum(1 for x in samples if abs(x - mu) <= sigma) / n
within_2 = sum(1 for x in samples if abs(x - mu) <= 2 * sigma) / n
within_3 = sum(1 for x in samples if abs(x - mu) <= 3 * sigma) / n

print(f"Within 1 sigma: {within_1:.4f}  (expected ~0.6827)")
print(f"Within 2 sigma: {within_2:.4f}  (expected ~0.9545)")
print(f"Within 3 sigma: {within_3:.4f}  (expected ~0.9973)")
```

### 10.3 무기억 성질 시연

```python
import random

random.seed(123)
lam = 0.5
n = 200_000
samples = [random.expovariate(lam) for _ in range(n)]

s = 2.0
t = 3.0

# P(X > s + t | X > s) should equal P(X > t)
exceed_s = [x for x in samples if x > s]
conditional = sum(1 for x in exceed_s if x > s + t) / len(exceed_s)
marginal = sum(1 for x in samples if x > t) / n

print(f"P(X > {s+t} | X > {s}) = {conditional:.4f}")
print(f"P(X > {t})             = {marginal:.4f}")
print(f"Theoretical P(X > {t}) = {math.exp(-lam * t):.4f}")
```

### 10.4 감마분포: 지수분포의 합

```python
import random
import math

random.seed(99)
lam = 2.0
alpha = 5  # shape = number of exponentials to sum
n = 100_000

# Sum of alpha independent Exp(lam) is Gamma(alpha, lam)
gamma_samples = []
for _ in range(n):
    total = sum(random.expovariate(lam) for _ in range(alpha))
    gamma_samples.append(total)

sample_mean = sum(gamma_samples) / n
sample_var = sum((x - sample_mean) ** 2 for x in gamma_samples) / (n - 1)

print(f"Sample mean:     {sample_mean:.4f}  (theoretical: {alpha / lam:.4f})")
print(f"Sample variance: {sample_var:.4f}  (theoretical: {alpha / lam**2:.4f})")
```

### 10.5 정규의 제곱으로부터 카이제곱

```python
import random

random.seed(77)
k = 6  # degrees of freedom
n = 100_000

chi2_samples = []
for _ in range(n):
    val = sum(random.gauss(0, 1) ** 2 for _ in range(k))
    chi2_samples.append(val)

mean_est = sum(chi2_samples) / n
var_est = sum((x - mean_est) ** 2 for x in chi2_samples) / (n - 1)

print(f"Chi-squared({k}):")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: {k})")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {2 * k})")
```

### 10.6 정규와 카이제곱으로부터 스튜던트 t

```python
import random
import math

random.seed(55)
nu = 5  # degrees of freedom
n = 100_000

t_samples = []
for _ in range(n):
    z = random.gauss(0, 1)
    v = sum(random.gauss(0, 1) ** 2 for _ in range(nu))
    t_val = z / math.sqrt(v / nu)
    t_samples.append(t_val)

mean_est = sum(t_samples) / n
var_est = sum((x - mean_est) ** 2 for x in t_samples) / (n - 1)
theoretical_var = nu / (nu - 2) if nu > 2 else float('inf')

print(f"Student's t({nu}):")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: 0)")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {theoretical_var:.4f})")
```

---

## 핵심 요약

1. **정규분포**는 통계학의 초석이며, 그 종 모양 곡선은 중심 극한 정리(Central Limit Theorem)를 통해 자연스럽게 나타납니다.
2. **지수분포**는 기하분포의 연속 유사체이며 유일하게 무기억 성질을 갖습니다.
3. **감마분포**는 지수분포를 일반화합니다. 카이제곱은 특수한 감마분포입니다.
4. **베타분포**는 $[0, 1]$ 위에 정의되며, 이항 데이터에 대한 켤레 사전분포로서 베이즈 분석에 핵심적입니다.
5. **카이제곱, t, F 분포**는 정규분포로부터 유도되며, 고전적 가설 검정과 신뢰 구간의 기반입니다.
6. 많은 분포가 합, 비, 극한, 또는 특수한 매개변수 선택을 통해 연결됩니다. 이러한 관계를 이해하면 올바른 모델을 선택하고 표본 분포를 유도하는 데 도움이 됩니다.

---

*다음 레슨: [확률변수의 변환](./08_Transformations_of_Random_Variables.md)*

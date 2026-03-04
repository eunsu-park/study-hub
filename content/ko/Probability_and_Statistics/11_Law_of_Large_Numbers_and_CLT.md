# 큰 수의 법칙과 중심극한정리

**이전**: [수렴 개념](./10_Convergence_Concepts.md) | **다음**: [점 추정](./12_Point_Estimation.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 체비셰프 부등식을 사용하여 약한 큰 수의 법칙을 진술하고 증명하기
2. 강한 큰 수의 법칙을 진술하고 약한 큰 수의 법칙과의 차이를 설명하기
3. 큰 수의 법칙을 몬테카를로 추정 문제에 적용하기
4. 중심극한정리의 조건과 의미를 진술하고 해석하기
5. 베리-에센 한계 (Berry-Esseen Bound)와 수렴 속도에 대한 함의를 설명하기
6. 이산 분포를 정규분포로 근사할 때 연속성 보정을 적용하기
7. 중심극한정리가 신뢰구간과 여론조사를 뒷받침하는 방식을 미리보기
8. Python에서 큰 수의 법칙과 중심극한정리를 시뮬레이션하여 시각적 직관 쌓기

---

큰 수의 법칙 (Law of Large Numbers, LLN)과 중심극한정리 (Central Limit Theorem, CLT)는 확률론에서 가장 중요한 두 정리입니다. 이 둘은 평균이 왜 예측 가능하게 행동하는지, 그리고 정규분포가 어디에서나 나타나는 이유를 함께 설명합니다. 큰 수의 법칙은 표본 평균이 **어디로** 수렴하는지를, 중심극한정리는 그 극한 주위에서 **어떻게** 변동하는지를 알려줍니다.

---

## 1. 약한 큰 수의 법칙 (WLLN)

### 1.1 진술

$X_1, X_2, \ldots$가 $E[X_i] = \mu$이고 $\text{Var}(X_i) = \sigma^2 < \infty$인 i.i.d. 확률 변수라 합시다. 표본 평균을 다음과 같이 정의합니다:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$

그러면 모든 $\varepsilon > 0$에 대해:

$$\lim_{n \to \infty} P\!\left(|\bar{X}_n - \mu| > \varepsilon\right) = 0$$

즉, $\bar{X}_n \xrightarrow{P} \mu$입니다.

### 1.2 체비셰프 부등식을 이용한 증명

체비셰프 부등식 (Chebyshev's Inequality)은 다음과 같습니다: $P(|Y - E[Y]| \ge \varepsilon) \le \text{Var}(Y)/\varepsilon^2$.

이를 $Y = \bar{X}_n$에 적용합니다:

$$E[\bar{X}_n] = \mu, \qquad \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$$

따라서:

$$P\!\left(|\bar{X}_n - \mu| \ge \varepsilon\right) \le \frac{\sigma^2}{n\varepsilon^2} \to 0 \text{ as } n \to \infty$$

이 증명은 기초적이지만 유한 분산 가정이 필요합니다. 약한 큰 수의 법칙의 더 일반적인 버전은 $E[|X_i|] < \infty$만 요구합니다.

### 1.3 해석

약한 큰 수의 법칙은 모집단 평균의 추정치로서 표본 평균의 사용을 정당화합니다. 충분한 데이터가 있으면 표본 평균이 참 평균에 가까워질 가능성이 높다는 것을 보장합니다.

---

## 2. 강한 큰 수의 법칙 (SLLN)

### 2.1 진술

약한 큰 수의 법칙과 동일한 조건 하에서 (또는 더 일반적으로, 콜모고로프 버전에서는 $E[|X_i|] < \infty$만 요구):

$$P\!\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

즉, $\bar{X}_n \xrightarrow{a.s.} \mu$입니다.

### 2.2 WLLN vs. SLLN

| 속성 | WLLN | SLLN |
|------|------|------|
| 수렴 방식 | 확률 수렴 | 거의 확실한 수렴 |
| 조건 | 유한 분산 (단순 증명) | 유한 평균 (콜모고로프) |
| 강도 | 약함 | 강함 |
| 의미 | $P(\lvert\bar{X}_n - \mu\rvert > \varepsilon) \to 0$ | 표본 경로가 수렴 |

강한 큰 수의 법칙은 **하나의** 무한히 긴 관측 수열이 $\mu$로 수렴하는 표본 평균을 생성한다고 말합니다. 약한 큰 수의 법칙은 큰 편차의 확률이 사라진다고만 말합니다.

### 2.3 증명의 직관

강한 큰 수의 법칙의 증명 (예: 콜모고로프의 삼급수 정리나 절단 논증을 통한)은 약한 큰 수의 법칙보다 훨씬 복잡합니다. 핵심 아이디어는 4차 적률 한계 $E[(\bar{X}_n - \mu)^4] = O(1/n^2)$이 꼬리 확률을 합산 가능하게 만들어 보렐-칸텔리 보조정리의 적용을 허용한다는 것입니다.

---

## 3. 실용적 함의: 몬테카를로 추정

### 3.1 몬테카를로 방법

직접 계산이 어려운 $\theta = E[g(X)]$를 추정하려면:

1. $X$의 분포에서 $X_1, X_2, \ldots, X_n$을 i.i.d.로 추출합니다.
2. $\hat{\theta}_n = \frac{1}{n}\sum_{i=1}^n g(X_i)$를 계산합니다.
3. 강한 큰 수의 법칙에 의해 $\hat{\theta}_n \to \theta$ 거의 확실하게 수렴합니다.

### 3.2 예시: $\pi$ 추정하기

단위 정사각형 $[0,1]^2$과 사분원 $x^2 + y^2 \le 1$을 고려합니다. 사분원의 넓이는 $\pi/4$입니다. $(U_1, U_2) \sim \text{Uniform}([0,1]^2)$이면:

$$\pi = 4 \cdot E\!\left[\mathbf{1}(U_1^2 + U_2^2 \le 1)\right]$$

### 3.3 몬테카를로 오차

중심극한정리 (다음 절)에 의해, 몬테카를로 오차는 근사적으로:

$$\hat{\theta}_n - \theta \approx N\!\left(0,\, \frac{\text{Var}(g(X))}{n}\right)$$

오차는 $O(1/\sqrt{n})$으로 감소하며, 이는 정확도를 한 자리 더 얻기 위해 100배 더 많은 표본이 필요하다는 것을 의미합니다.

---

## 4. 중심극한정리 (CLT)

### 4.1 진술

$X_1, X_2, \ldots$가 $E[X_i] = \mu$이고 $0 < \text{Var}(X_i) = \sigma^2 < \infty$인 i.i.d.라 합시다. 그러면:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} N(0, 1)$$

동치 표현: $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$.

### 4.2 조건

고전적 (린데베르크-레비, Lindeberg-Levy) 중심극한정리는 다음을 요구합니다:

1. **독립성**: $X_i$들이 독립입니다.
2. **동일 분포**: 모든 $X_i$가 같은 분포를 따릅니다.
3. **유한 분산**: $0 < \sigma^2 < \infty$.

각 $X_i$의 분포는 이 조건들이 만족되는 한 **무엇이든** (이산, 연속, 비대칭, 다봉) 될 수 있습니다.

### 4.3 왜 중요한가

중심극한정리는 다음을 설명합니다:

- 정규분포가 자연에서 매우 빈번하게 나타나는 이유 (키, 측정 오차 등)
- 기저 데이터가 정규가 아니더라도 표본 평균이 근사적으로 정규인 이유
- 정규성에 기반한 많은 통계 절차가 비정규 데이터에서도 잘 작동하는 이유

### 4.4 증명 스케치 (MGF를 통한)

$Z_n = \sqrt{n}(\bar{X}_n - \mu)/\sigma$의 적률생성함수 (MGF)는:

$$M_{Z_n}(t) = \left[M_X\!\left(\frac{t}{\sigma\sqrt{n}}\right) e^{-\mu t/(\sigma\sqrt{n})}\right]^n$$

$X$의 적률생성함수를 영점 근처에서 전개하고 $n \to \infty$로 취하면, $M_{Z_n}(t) \to e^{t^2/2}$임을 보일 수 있으며, 이는 $N(0,1)$의 적률생성함수입니다.

---

## 5. 베리-에센 한계

### 5.1 진술

중심극한정리의 가정 하에서, 추가로 $E[|X_i - \mu|^3] = \rho < \infty$이면:

$$\sup_x \left|P\!\left(\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \le x\right) - \Phi(x)\right| \le \frac{C\rho}{\sigma^3 \sqrt{n}}$$

여기서 $C$는 보편 상수 (최대 $0.4748$로 알려져 있음)입니다.

### 5.2 해석

- 중심극한정리는 수렴이 일어난다고 알려주고, 베리-에센 한계 (Berry-Esseen Bound)는 **얼마나 빠른지**를 알려줍니다: 정규 근사의 오차는 $O(1/\sqrt{n})$입니다.
- 큰 3차 적률을 가진 분포 (강한 비대칭)는 더 느리게 수렴합니다.
- 한계는 모든 $x$에 대해 균등하므로, 최악의 CDF 불일치를 제어합니다.

---

## 6. 정규 근사와 연속성 보정

### 6.1 이산 분포의 근사

중심극한정리를 사용하여 이산 분포 (예: 이항분포)를 정규분포로 근사할 때:

$$P(X = k) \approx P(k - 0.5 < Y < k + 0.5)$$

여기서 $Y \sim N(\mu, \sigma^2)$입니다.

### 6.2 예시: 이항분포 근사

$X \sim \text{Binomial}(100, 0.3)$이라 합시다. 그러면 $\mu = 30$, $\sigma^2 = 21$, $\sigma \approx 4.583$입니다.

$P(X \le 25)$를 구하려면:

- **보정 없이**: $P\!\left(Z \le \frac{25 - 30}{4.583}\right) = P(Z \le -1.091) \approx 0.1377$
- **보정 적용**: $P\!\left(Z \le \frac{25.5 - 30}{4.583}\right) = P(Z \le -0.982) \approx 0.1631$

연속성 보정 (Continuity Correction)은 이산-연속 간격을 고려하기 위해 0.5를 더합니다. 중간 크기의 $n$에서 이 보정은 정확도를 크게 향상시킵니다.

---

## 7. 응용

### 7.1 신뢰구간 (미리보기)

중심극한정리에 의해, 큰 $n$에 대해:

$$P\!\left(-z_{\alpha/2} \le \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \le z_{\alpha/2}\right) \approx 1 - \alpha$$

정리하면:

$$P\!\left(\bar{X}_n - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \le \mu \le \bar{X}_n + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) \approx 1 - \alpha$$

95% 신뢰구간의 경우, $z_{0.025} = 1.96$이므로:

$$\bar{X}_n \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

### 7.2 여론조사와 오차 한계

선거 여론조사에서 각 응답자는 $p$ = 참 비율인 베르누이 시행입니다. $n$명의 응답자에 대해:

$$\hat{p} = \bar{X}_n, \qquad \text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

여론조사에서 보도하는 "오차 한계 (Margin of Error)"는 보통 $\pm 1.96 \cdot \text{SE}(\hat{p})$입니다. $n = 1000$이고 $\hat{p} = 0.5$이면, 이는 대략 $\pm 3.1\%$입니다.

### 7.3 표본 크기 결정

신뢰 수준 $1 - \alpha$에서 오차 한계 $\varepsilon$을 달성하려면:

$$n \ge \left(\frac{z_{\alpha/2}\, \sigma}{\varepsilon}\right)^2$$

최악의 경우 $p = 0.5$인 비율 추정과 95% 신뢰 수준의 경우: $n \ge (1.96)^2/(4\varepsilon^2) = 0.9604/\varepsilon^2$.

---

## 8. 다변량 CLT (간략)

### 8.1 진술

$\mathbf{X}_1, \mathbf{X}_2, \ldots$가 $\mathbb{R}^p$에서 $E[\mathbf{X}_i] = \boldsymbol{\mu}$이고 $\text{Cov}(\mathbf{X}_i) = \boldsymbol{\Sigma}$인 i.i.d. 확률 벡터라 합시다. 그러면:

$$\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} N_p(\mathbf{0}, \boldsymbol{\Sigma})$$

### 8.2 결과

$\bar{\mathbf{X}}_n$의 각 성분은 점근적으로 정규이며, 결합 분포는 점근적으로 다변량 정규입니다. 이는 다변량 신뢰 영역 (타원체)과 동시 추론을 정당화합니다.

---

## 9. Python 시뮬레이션 예제

### 9.1 큰 수의 법칙: 표본 평균의 수렴

```python
import random

random.seed(42)

# Simulate rolling a fair die: E[X] = 3.5
mu = 3.5
n = 10_000
running_sum = 0

checkpoints = [10, 50, 100, 500, 1000, 5000, 10000]
print("LLN: Sample mean of fair die rolls converging to 3.5")
print("-" * 50)

for i in range(1, n + 1):
    running_sum += random.randint(1, 6)
    if i in checkpoints:
        mean_i = running_sum / i
        print(f"  n = {i:5d}:  X_bar = {mean_i:.4f}  "
              f"(error = {abs(mean_i - mu):.4f})")
```

### 9.2 몬테카를로를 이용한 Pi 추정

```python
import random

random.seed(314)

print("\nMonte Carlo estimation of pi")
print("-" * 50)

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    pi_est = 4.0 * inside / n
    error = abs(pi_est - 3.141592653589793)
    print(f"  n = {n:>9,}:  pi ~ {pi_est:.6f}  (error = {error:.6f})")
```

### 9.3 CLT: 주사위 굴림의 평균

```python
import random
import math

random.seed(2024)
n_sim = 50_000

# For a fair die: mu=3.5, sigma^2=35/12, sigma~1.7078
mu = 3.5
sigma = math.sqrt(35.0 / 12.0)

print("\nCLT: Distribution of standardised mean of n dice rolls")
print("Fraction in [-1.96, 1.96] should approach 0.95")
print("-" * 55)

for n in [2, 5, 10, 30, 100]:
    count_in = 0
    for _ in range(n_sim):
        rolls = [random.randint(1, 6) for _ in range(n)]
        x_bar = sum(rolls) / n
        z = (x_bar - mu) / (sigma / math.sqrt(n))
        if -1.96 <= z <= 1.96:
            count_in += 1
    frac = count_in / n_sim
    print(f"  n = {n:3d}:  P(-1.96 < Z < 1.96) = {frac:.4f}")
```

### 9.4 CLT 히스토그램 (텍스트)

```python
import random
import math

random.seed(100)
n = 30          # samples per mean
n_sim = 20_000  # number of means to compute

# Exponential(1): mu=1, sigma=1, heavily right-skewed
mu, sigma = 1.0, 1.0

z_values = []
for _ in range(n_sim):
    vals = [random.expovariate(1.0) for _ in range(n)]
    x_bar = sum(vals) / n
    z = (x_bar - mu) / (sigma / math.sqrt(n))
    z_values.append(z)

# Build histogram from -4 to 4 with 20 bins
n_bins = 20
lo, hi = -4.0, 4.0
bin_width = (hi - lo) / n_bins
bins = [0] * n_bins

for z in z_values:
    idx = int((z - lo) / bin_width)
    if 0 <= idx < n_bins:
        bins[idx] += 1

print(f"\nCLT Histogram: standardised mean of {n} Exp(1) samples")
print(f"(n_sim = {n_sim})")
print("-" * 55)
max_count = max(bins)
for i in range(n_bins):
    left = lo + i * bin_width
    bar_len = int(bins[i] * 50 / max_count) if max_count > 0 else 0
    bar = '#' * bar_len
    print(f"  [{left:5.1f},{left+bin_width:5.1f}): {bar}")
```

### 9.5 연속성 보정 비교

```python
import random
import math

random.seed(55)

# Binomial(100, 0.3): exact P(X <= 25)
# We estimate by simulation, then compare normal approx with/without correction
n_binom = 100
p = 0.3
mu = n_binom * p          # 30
sigma = math.sqrt(n_binom * p * (1 - p))  # sqrt(21) ~ 4.583

n_sim = 500_000
count_le_25 = 0
for _ in range(n_sim):
    x = sum(1 for _ in range(n_binom) if random.random() < p)
    if x <= 25:
        count_le_25 += 1

sim_prob = count_le_25 / n_sim

# Normal CDF approximation using error function
def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

approx_no_corr = normal_cdf(25, mu, sigma)
approx_with_corr = normal_cdf(25.5, mu, sigma)

print(f"\nContinuity correction: P(Binomial(100, 0.3) <= 25)")
print(f"  Simulation:          {sim_prob:.4f}")
print(f"  Normal (no corr):    {approx_no_corr:.4f}")
print(f"  Normal (with corr):  {approx_with_corr:.4f}")
```

### 9.6 여론조사를 위한 표본 크기

```python
import math

print("\nSample sizes needed for margin of error (95% CI, worst case p=0.5)")
print("-" * 50)

z = 1.96
for margin in [0.05, 0.03, 0.02, 0.01, 0.005]:
    n_needed = math.ceil((z ** 2) * 0.25 / (margin ** 2))
    print(f"  Margin = {margin:.3f} ({margin*100:.1f}%):  n >= {n_needed:,}")
```

---

## 핵심 요약

1. **약한 큰 수의 법칙**은 $\bar{X}_n \xrightarrow{P} \mu$를 보장하며, 체비셰프를 이용한 증명은 직접적이고 유한 분산만 요구합니다.
2. **강한 큰 수의 법칙**은 이를 거의 확실한 수렴으로 강화하여, 개별 표본 경로가 수렴한다는 것을 의미합니다.
3. **몬테카를로 방법**은 큰 수의 법칙의 직접적인 응용입니다: 무작위 표본을 평균하여 기대값을 추정하며, 오차는 $O(1/\sqrt{n})$입니다.
4. **중심극한정리**는 기저 분포에 관계없이 (분산이 유한하면) 표준화된 표본 평균이 근사적으로 $N(0,1)$이라고 말합니다.
5. **베리-에센 한계**는 중심극한정리의 수렴 속도를 $O(1/\sqrt{n})$으로 정량화합니다.
6. 이산 분포에 대해 **연속성 보정**은 정규 근사를 향상시킵니다.
7. 중심극한정리는 **신뢰구간**, **가설 검정**, **여론조사 오차 한계**의 기초가 되어 응용 통계학에서 가장 중요한 단일 정리입니다.

---

*다음 단원: [점 추정](./12_Point_Estimation.md)*

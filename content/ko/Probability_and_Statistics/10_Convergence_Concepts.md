# 수렴 개념

**이전**: [다변량 정규분포](./09_Multivariate_Normal_Distribution.md) | **다음**: [큰 수의 법칙과 중심극한정리](./11_Law_of_Large_Numbers_and_CLT.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 네 가지 수렴 방식(분포 수렴, 확률 수렴, 거의 확실한 수렴, $L^p$ 수렴)을 정의하고 구별하기
2. 이들 수렴 방식 간의 함의 관계 위계를 서술하기
3. 특정 함의가 역으로 성립하지 않음을 보여주는 반례를 구성하기
4. 슬루츠키 정리 (Slutsky's theorem)를 적용하여 극한 논증을 단순화하기
5. 연속 사상 정리 (Continuous Mapping Theorem)를 사용하여 연속 함수를 통해 극한을 전달하기
6. 델타 방법 (Delta Method)을 서술하고 변환된 추정량의 근사 분포에 적용하기
7. 확률 변수 수열을 시뮬레이션하여 각 수렴 방식을 예시하기

---

표본 크기가 증가함에 따라, 확률 변수의 수열은 종종 예측 가능한 극한으로 수렴합니다. 그러나 "수렴한다"는 것은 수렴의 의미에 따라 다른 뜻을 가집니다. 이 단원에서는 네 가지 수렴 개념을 엄밀하게 정의하고, 그들의 논리적 관계를 정리하며, 수렴 논증을 실용적으로 만드는 핵심 정리들을 소개합니다.

---

## 1. 분포 수렴 (약 수렴)

### 1.1 정의

수열 $\{X_n\}$이 $X$로 **분포 수렴 (Convergence in Distribution)** 한다는 것은 $X_n \xrightarrow{d} X$로 표기하며, 다음이 성립할 때를 말합니다:

$$\lim_{n \to \infty} F_{X_n}(x) = F_X(x) \quad \text{at every continuity point } x \text{ of } F_X$$

동치 조건으로, 모든 유계 연속 함수 $g$에 대해 $E[g(X_n)] \to E[g(X)]$입니다.

### 1.2 직관

분포 수렴은 CDF들이 (연속점에서) 점별로 수렴한다는 것을 의미합니다. 이는 분포의 **형태**에 대한 진술이지, 같은 확률 공간에서 확률 변수들이 가까워진다는 것은 아닙니다.

### 1.3 주요 성질

- 네 가지 수렴 방식 중 **가장 약한** 것입니다: $X_n$과 $X$가 같은 확률 공간에서 정의될 필요가 없습니다.
- 극한 $X$가 상수 $c$일 수 있습니다; 이때 $X_n \xrightarrow{d} c$는 $F_{X_n}(x) \to \mathbf{1}(x \ge c)$를 의미합니다.
- **레비 연속성 정리 (Levy's Continuity Theorem)**: $X_n \xrightarrow{d} X$인 것과 특성함수가 점별 수렴하는 것($\varphi_{X_n}(t) \to \varphi_X(t)$ for all $t$)은 동치입니다.

### 1.4 예시

$X_n \sim N(0, 1/n)$이라 하면, $n$이 증가함에 따라 분포가 영점에 집중하므로 $X_n \xrightarrow{d} 0$입니다.

---

## 2. 확률 수렴

### 2.1 정의

$\{X_n\}$이 $X$로 **확률 수렴 (Convergence in Probability)** 한다는 것은 $X_n \xrightarrow{P} X$로 표기하며, 모든 $\varepsilon > 0$에 대해 다음이 성립할 때를 말합니다:

$$\lim_{n \to \infty} P(|X_n - X| > \varepsilon) = 0$$

### 2.2 직관

충분히 큰 $n$에 대해, $X_n$이 $X$에서 멀리 떨어져 있을 확률이 무시할 수 있을 정도로 작아집니다. 분포 수렴과 달리, $X_n$과 $X$가 **같은** 확률 공간에 있어야 하며 **차이** $X_n - X$가 관련됩니다.

### 2.3 예시

$X_n \sim \text{Uniform}(0, 1/n)$이라 하면, 임의의 $\varepsilon > 0$과 $n > 1/\varepsilon$에 대해:

$$P(|X_n| > \varepsilon) = P(X_n > \varepsilon) = 0$$

따라서 $X_n \xrightarrow{P} 0$입니다.

### 2.4 분포 수렴과의 관계

$$X_n \xrightarrow{P} X \implies X_n \xrightarrow{d} X$$

**특수한 경우**: $X_n \xrightarrow{d} c$ (상수)이면 $X_n \xrightarrow{P} c$입니다. 이 역방향은 극한이 상수일 때만 성립합니다.

---

## 3. 거의 확실한 수렴

### 3.1 정의

$\{X_n\}$이 $X$로 **거의 확실하게 (Almost Surely)** 수렴한다는 것은 $X_n \xrightarrow{a.s.} X$로 표기하며, 다음이 성립할 때를 말합니다:

$$P\!\left(\lim_{n \to \infty} X_n = X\right) = 1$$

동치 조건: $P\!\left(\omega : X_n(\omega) \to X(\omega)\right) = 1$.

### 3.2 직관

거의 모든 결과 $\omega$에 대해, 수열 $X_1(\omega), X_2(\omega), \ldots$이 $X(\omega)$로 통상적인 해석학적 의미에서 수렴합니다. 수렴이 실패하는 "나쁜" 결과의 집합은 확률이 0입니다.

### 3.3 확률 수렴과의 비교

거의 확실한 수렴은 확률 수렴보다 **강합니다**:

$$X_n \xrightarrow{a.s.} X \implies X_n \xrightarrow{P} X$$

역은 거짓입니다. 차이점: 확률 수렴은 $X_n$이 $X$에서 가끔 벗어나는 것을 허용하되 그 확률이 줄어들기만 하면 됩니다. 거의 확실한 수렴은 개별 표본 경로가 결국 가까이 머물 것을 요구합니다.

### 3.4 보렐-칸텔리 판정법

실용적인 도구: 모든 $\varepsilon > 0$에 대해 $\sum_{n=1}^{\infty} P(|X_n - X| > \varepsilon) < \infty$이면, $X_n \xrightarrow{a.s.} X$입니다.

---

## 4. $L^p$ 수렴

### 4.1 정의

$\{X_n\}$이 $X$로 **$L^p$ 수렴** ($p \ge 1$)한다는 것은 $X_n \xrightarrow{L^p} X$로 표기하며, 다음이 성립할 때를 말합니다:

$$\lim_{n \to \infty} E\!\left[|X_n - X|^p\right] = 0$$

가장 흔한 경우는 $p = 2$(**평균 제곱 수렴, Mean-Square Convergence**)입니다:

$$E\!\left[(X_n - X)^2\right] \to 0$$

### 4.2 직관

$L^p$ 수렴은 편차의 평균 크기를 제어합니다. 꼬리 행동에 민감합니다: $X_n$이 무거운 꼬리를 가져 가끔 큰 편차를 만들면, 확률 수렴이 성립하더라도 $L^p$ 수렴은 실패할 수 있습니다.

### 4.3 다른 수렴 방식과의 관계

$$X_n \xrightarrow{L^p} X \implies X_n \xrightarrow{P} X$$

이는 마르코프 부등식에서 따릅니다: $P(|X_n - X| > \varepsilon) \le E[|X_n - X|^p] / \varepsilon^p$.

또한, $p > q \ge 1$이면, $L^p$ 수렴은 $L^q$ 수렴을 함의합니다 (젠센 부등식에 의해).

---

## 5. 수렴 방식의 위계

### 5.1 함의 관계 요약

```
        a.s.
         |
         v
  Lp --> Prob --> Dist
```

상세:

| 출발 | 도착 | 성립? |
|------|------|-------|
| 거의 확실한 수렴 | 확률 수렴 | 예 |
| $L^p$ | 확률 수렴 | 예 |
| 확률 수렴 | 분포 수렴 | 예 |
| 분포 수렴 | 확률 수렴 | 극한이 상수일 때만 |
| 확률 수렴 | 거의 확실한 수렴 | 아니오 (일반적으로) |
| 확률 수렴 | $L^p$ | 아니오 (일반적으로) |
| 거의 확실한 수렴 | $L^p$ | 아니오 (일반적으로) |
| $L^p$ | 거의 확실한 수렴 | 아니오 (일반적으로) |

### 5.2 역방향 함의를 위한 추가 조건

- **확률 수렴에서 거의 확실한 수렴으로**: 확률 수렴이 충분히 빠르면 (예: 보렐-칸텔리를 통한 합산 가능한 꼬리 확률), 거의 확실한 수렴이 따릅니다.
- **확률 수렴에서 $L^p$으로**: 수열 $\{|X_n - X|^p\}$이 **균등 적분 가능 (Uniformly Integrable)** 하면, 확률 수렴은 $L^p$ 수렴을 함의합니다.

---

## 6. 반례

### 6.1 확률 수렴은 하지만 거의 확실한 수렴은 하지 않는 경우

**타자기 수열 (Typewriter Sequence)**: $\Omega = [0, 1]$에서 균등 측도로 확률 변수를 정의합니다. 각 $n$에 대해 $[0, 1]$을 $n$개의 동일한 부분 구간으로 나누고 순환합니다:

- $X_1 = \mathbf{1}_{[0, 1]}$
- $X_2 = \mathbf{1}_{[0, 1/2)}$, $X_3 = \mathbf{1}_{[1/2, 1]}$
- $X_4 = \mathbf{1}_{[0, 1/3)}$, $X_5 = \mathbf{1}_{[1/3, 2/3)}$, $X_6 = \mathbf{1}_{[2/3, 1]}$
- ...

임의의 $\varepsilon > 0$에 대해, $P(X_n > \varepsilon) \to 0$ (구간 폭이 줄어들므로)이므로 $X_n \xrightarrow{P} 0$입니다.

그러나 임의의 $\omega \in [0, 1]$에 대해, 무한히 많은 $X_n(\omega) = 1$이므로 $X_n(\omega) \not\to 0$입니다. 따라서 $X_n \not\xrightarrow{a.s.} 0$입니다.

### 6.2 확률 수렴은 하지만 $L^1$ 수렴은 하지 않는 경우

$X_n = n$ (확률 $1/n$), $X_n = 0$ (확률 $1 - 1/n$)으로 정의합니다.

- $P(|X_n| > \varepsilon) = 1/n \to 0$이므로 $X_n \xrightarrow{P} 0$입니다.
- $E[|X_n|] = n \cdot (1/n) = 1 \not\to 0$이므로 $X_n \not\xrightarrow{L^1} 0$입니다.

드물지만 큰 값이 $L^1$ 수렴을 방해합니다.

### 6.3 분포 수렴은 하지만 확률 수렴은 하지 않는 경우

$X \sim N(0,1)$이고 모든 $n$에 대해 $X_n = -X$로 정의합니다. 그러면 모든 $n$에 대해 $X_n \sim N(0,1)$이므로 $X_n \xrightarrow{d} X$입니다. 그러나 $P(|X_n - X| > \varepsilon) = P(|2X| > \varepsilon) > 0$이 모든 $n$에 대해 성립하므로 $X_n \not\xrightarrow{P} X$입니다.

---

## 7. 슬루츠키 정리

### 7.1 진술

$X_n \xrightarrow{d} X$이고 $Y_n \xrightarrow{P} c$ (상수)이면:

1. $X_n + Y_n \xrightarrow{d} X + c$
2. $X_n Y_n \xrightarrow{d} cX$
3. $X_n / Y_n \xrightarrow{d} X / c$ ($c \ne 0$인 경우)

### 7.2 중요성

슬루츠키 정리 (Slutsky's Theorem)는 점근적 논증의 핵심 도구입니다. 분포 수렴하는 항과 상수로 확률 수렴하는 항을 결합할 수 있게 해줍니다. 이는 검정 통계량의 점근 분포를 유도할 때 반복적으로 사용됩니다.

### 7.3 적용 예시

$\bar{X}_n \xrightarrow{d} N(\mu, \sigma^2/n)$ (중심극한정리에 의해)이고 $S_n^2 \xrightarrow{P} \sigma^2$ (약한 큰 수의 법칙에 의해)라 가정합니다. 그러면:

$$T_n = \frac{\bar{X}_n - \mu}{S_n / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

이는 $S_n/\sigma \xrightarrow{P} 1$이므로 슬루츠키 정리를 적용한 것입니다.

---

## 8. 연속 사상 정리

### 8.1 진술

$X_n \xrightarrow{d} X$이고 $g$가 연속 함수이면, $g(X_n) \xrightarrow{d} g(X)$입니다.

확률 수렴이나 거의 확실한 수렴에 대해서도 동일하게 성립합니다:

- $X_n \xrightarrow{P} X \implies g(X_n) \xrightarrow{P} g(X)$
- $X_n \xrightarrow{a.s.} X \implies g(X_n) \xrightarrow{a.s.} g(X)$

### 8.2 예시

$X_n \xrightarrow{d} N(0, 1)$이면, 연속 사상 정리 (Continuous Mapping Theorem)에 의해 $g(x) = x^2$을 적용하면 $X_n^2 \xrightarrow{d} \chi^2(1)$입니다.

---

## 9. 델타 방법

### 9.1 진술

$\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$이고 $g$가 $\theta$에서 미분 가능하며 $g'(\theta) \ne 0$이라 가정합니다. 그러면:

$$\sqrt{n}\!\left(g(X_n) - g(\theta)\right) \xrightarrow{d} N\!\left(0,\; [g'(\theta)]^2 \sigma^2\right)$$

### 9.2 직관

델타 방법 (Delta Method)은 1차 테일러 전개를 사용합니다: $g(X_n) \approx g(\theta) + g'(\theta)(X_n - \theta)$. 선형 근사는 $X_n$의 점근 정규성을 물려받으며, 분산은 $[g'(\theta)]^2$으로 스케일링됩니다.

### 9.3 예시: 분산 안정화 변환

$X_n \sim \text{Poisson}(\lambda)/n$ (포아송의 표본 평균)이면, $\sqrt{n}(X_n - \lambda) \xrightarrow{d} N(0, \lambda)$입니다.

$g(x) = \sqrt{x}$를 적용하면 $g'(\lambda) = 1/(2\sqrt{\lambda})$이므로:

$$\sqrt{n}\!\left(\sqrt{X_n} - \sqrt{\lambda}\right) \xrightarrow{d} N\!\left(0,\; \frac{1}{4}\right)$$

분산이 더 이상 $\lambda$에 의존하지 않으므로, 제곱근 변환은 포아송 데이터에 대한 "분산 안정화 변환 (Variance-Stabilising Transformation)"이라 불립니다.

### 9.4 다변량 델타 방법

$\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma})$이고 $g: \mathbb{R}^p \to \mathbb{R}$가 $\boldsymbol{\theta}$에서 미분 가능하면:

$$\sqrt{n}\!\left(g(\mathbf{X}_n) - g(\boldsymbol{\theta})\right) \xrightarrow{d} N\!\left(0,\; \nabla g(\boldsymbol{\theta})^T \boldsymbol{\Sigma}\, \nabla g(\boldsymbol{\theta})\right)$$

---

## 10. Python 시뮬레이션 예제

### 10.1 확률 수렴: 표본 평균

```python
import random

random.seed(42)

def sample_mean(dist_sampler, n):
    """Compute sample mean of n draws."""
    return sum(dist_sampler() for _ in range(n)) / n

# X_i ~ Uniform(0, 1), E[X] = 0.5
mu = 0.5
eps = 0.05

print("Convergence in probability: P(|X_bar_n - 0.5| > 0.05)")
print("-" * 50)

for n in [10, 50, 100, 500, 1000, 5000]:
    n_sim = 20_000
    violations = 0
    for _ in range(n_sim):
        x_bar = sample_mean(random.random, n)
        if abs(x_bar - mu) > eps:
            violations += 1
    prob = violations / n_sim
    print(f"  n = {n:5d}:  P(|X_bar - 0.5| > 0.05) = {prob:.4f}")
```

### 10.2 거의 확실한 수렴: 누적 평균의 표본 경로

```python
import random

random.seed(7)
n_paths = 5
N = 2000

print("\nAlmost sure convergence: sample paths of cumulative mean")
print("(Each path should converge to 0.5)")
print("-" * 60)

for path in range(n_paths):
    running_sum = 0.0
    deviations_at_checkpoints = []
    for i in range(1, N + 1):
        running_sum += random.random()
        if i in [10, 100, 500, 1000, 2000]:
            mean_i = running_sum / i
            deviations_at_checkpoints.append((i, mean_i))

    results = ", ".join(f"n={i}: {m:.4f}" for i, m in deviations_at_checkpoints)
    print(f"  Path {path + 1}: {results}")
```

### 10.3 분포 수렴: CLT 미리보기

```python
import random
import math

random.seed(123)
n_sim = 50_000

# Standardised sum of Exp(1) variables (mean=1, var=1)
# Should converge to N(0,1)

print("\nConvergence in distribution (CLT):")
print("Fraction of standardised means in [-1.96, 1.96] (should -> 0.95)")
print("-" * 60)

for n in [5, 20, 50, 200]:
    count_in = 0
    for _ in range(n_sim):
        vals = [random.expovariate(1.0) for _ in range(n)]
        x_bar = sum(vals) / n
        z = (x_bar - 1.0) / (1.0 / math.sqrt(n))
        if -1.96 <= z <= 1.96:
            count_in += 1
    fraction = count_in / n_sim
    print(f"  n = {n:4d}:  P(-1.96 < Z < 1.96) = {fraction:.4f}")
```

### 10.4 반례: 확률 수렴은 하지만 L1 수렴은 하지 않는 경우

```python
import random

random.seed(999)
n_sim = 100_000

print("\nCounterexample: Convergence in prob but not L1")
print("X_n = n with prob 1/n, else 0")
print("-" * 50)

for n in [10, 100, 1000, 10000]:
    deviations = 0
    total_abs = 0.0
    for _ in range(n_sim):
        if random.random() < 1.0 / n:
            x = n
        else:
            x = 0
        if abs(x) > 0.5:
            deviations += 1
        total_abs += abs(x)

    p_dev = deviations / n_sim
    e_abs = total_abs / n_sim
    print(f"  n = {n:5d}:  P(|Xn|>0.5) = {p_dev:.4f},  E[|Xn|] = {e_abs:.4f}")
```

### 10.5 슬루츠키 정리 예시

```python
import random
import math

random.seed(456)
n_sim = 50_000

print("\nSlutsky's theorem: X_n + Y_n where X_n ->d N(0,1), Y_n ->P 3")
print("-" * 60)

for n in [10, 50, 200, 1000]:
    sums = []
    for _ in range(n_sim):
        # X_n: standardised mean of n Uniform(0,1) -> N(0,1) by CLT
        vals = [random.random() for _ in range(n)]
        x_bar = sum(vals) / n
        x_n = (x_bar - 0.5) / (math.sqrt(1.0 / (12 * n)))

        # Y_n: converges to 3 (e.g., sample mean of Uniform(2,4))
        y_vals = [random.uniform(2, 4) for _ in range(n)]
        y_n = sum(y_vals) / n

        sums.append(x_n + y_n)

    mean_sum = sum(sums) / n_sim
    var_sum = sum((s - mean_sum) ** 2 for s in sums) / (n_sim - 1)
    print(f"  n = {n:4d}:  mean(Xn+Yn) = {mean_sum:.4f} (->3.0),  "
          f"var(Xn+Yn) = {var_sum:.4f} (->1.0)")
```

### 10.6 델타 방법: 포아송 평균의 제곱근

```python
import random
import math

random.seed(789)
n_sim = 50_000
lam = 4.0  # Poisson parameter

print(f"\nDelta method: sqrt(X_bar) for Poisson({lam}) data")
print(f"Asymptotic variance of sqrt(n)*(sqrt(X_bar)-sqrt(lam)) should be 1/4")
print("-" * 60)

for n in [30, 100, 500, 2000]:
    scaled_diffs = []
    for _ in range(n_sim):
        # Generate n Poisson(lam) using standard library
        # Poisson via inverse transform
        total = 0
        for _ in range(n):
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                p *= random.random()
                if p < L:
                    break
                k += 1
            total += k
        x_bar = total / n
        if x_bar > 0:
            scaled_diffs.append(math.sqrt(n) * (math.sqrt(x_bar) - math.sqrt(lam)))

    mean_d = sum(scaled_diffs) / len(scaled_diffs)
    var_d = sum((d - mean_d) ** 2 for d in scaled_diffs) / (len(scaled_diffs) - 1)
    print(f"  n = {n:4d}:  mean = {mean_d:.4f} (->0),  var = {var_d:.4f} (->0.25)")
```

---

## 핵심 요약

1. **네 가지 수렴 방식**이 존재하며, 가장 약한 것부터 강한 것 순으로 (대략적으로): 분포 수렴, 확률 수렴, $L^p$ 수렴, 거의 확실한 수렴입니다.
2. **거의 확실한 수렴**과 **$L^p$ 수렴** 모두 확률 수렴을 함의하고, 확률 수렴은 분포 수렴을 함의합니다. 추가 조건 없이는 다른 일반적인 함의 관계는 성립하지 않습니다.
3. **반례**는 이 수렴 방식들이 진정으로 구별된다는 것을 보여줍니다: 확률 수렴은 거의 확실한 수렴이나 $L^p$ 수렴을 보장하지 않습니다.
4. **슬루츠키 정리**는 분포 극한과 상수로의 확률 극한을 결합할 수 있게 해주며, 이는 검정 통계량을 유도하는 데 필수적입니다.
5. **연속 사상 정리**는 연속 함수를 통해 모든 수렴 방식을 보존합니다.
6. **델타 방법**은 미분 가능한 변환을 통해 점근 정규성을 확장하여, 추정량 함수의 근사 분포를 제공합니다.

---

*다음 단원: [큰 수의 법칙과 중심극한정리](./11_Law_of_Large_Numbers_and_CLT.md)*

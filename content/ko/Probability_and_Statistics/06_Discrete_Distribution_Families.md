# 이산 분포족

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 여섯 가지 주요 이산 분포 각각의 PMF, 평균, 분산, MGF 서술하기
2. 주어진 현실 시나리오에 어떤 분포가 적합한지 식별하기
3. 이항분포에 대한 포아송 근사 적용하기
4. 기하분포의 무기억 성질 증명 및 적용하기
5. 이들 분포족 간의 관계 설명하기
6. Python 표준 라이브러리를 사용하여 각 분포 시뮬레이션하기

---

## 개요

이산 확률 분포(discrete probability distribution)는 확률적 모델링의 기본 구성 요소입니다. 모든 문제에 대해 PMF를 처음부터 정의하는 대신, 시나리오를 성질이 잘 알려진 명명된 분포족에 대응시킵니다. 이 레슨에서는 가장 중요한 여섯 가지 분포족, 그 매개변수, 성질, 상호 연결성을 다룹니다.

---

## 목차

1. [베르누이 분포](#1-베르누이-분포)
2. [이항분포](#2-이항분포)
3. [포아송 분포](#3-포아송-분포)
4. [기하분포](#4-기하분포)
5. [음이항분포](#5-음이항분포)
6. [초기하분포](#6-초기하분포)
7. [분포 간의 관계](#7-분포-간의-관계)
8. [Python 예제](#8-python-예제)
9. [핵심 요약](#9-핵심-요약)

---

## 1. 베르누이 분포

### 설정

두 가지 결과를 갖는 단일 시행: 확률 $p$로 **성공**(1), 확률 $1-p$로 **실패**(0).

### 표기

$$X \sim \text{Bernoulli}(p), \quad 0 \leq p \leq 1$$

### PMF

$$p_X(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

| $x$ | 0 | 1 |
|-----|---|---|
| $P(X=x)$ | $1-p$ | $p$ |

### 성질

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = p$ |
| 분산 | $\text{Var}(X) = p(1-p)$ |
| MGF | $M_X(t) = (1-p) + pe^t$ |

### 활용 사례

- 동전 던지기 (공정: $p = 0.5$)
- 대출의 채무불이행/정상상환
- 불량품/정상품

---

## 2. 이항분포

### 설정

각각 성공 확률 $p$인 $n$번의 **독립** 베르누이 시행에서 성공 횟수를 셉니다.

### 표기

$$X \sim \text{Binomial}(n, p)$$

### PMF

$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n$$

### 성질

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = np$ |
| 분산 | $\text{Var}(X) = np(1-p)$ |
| MGF | $M_X(t) = [(1-p) + pe^t]^n$ |

### 평균의 유도 (선형성 이용)

$X = \sum_{i=1}^n X_i$ ($X_i \sim \text{Bernoulli}(p)$, 독립)로 쓰면:

$$E[X] = \sum_{i=1}^n E[X_i] = np$$

$$\text{Var}(X) = \sum_{i=1}^n \text{Var}(X_i) = np(1-p)$$

### 예제

공정한 동전을 10번 던집니다. $P(\text{정확히 3번 앞면})$은?

$$P(X = 3) = \binom{10}{3} (0.5)^3 (0.5)^7 = 120 \cdot \frac{1}{1024} = \frac{120}{1024} \approx 0.1172$$

### 활용 사례

- $n$번 동전 던지기에서 앞면의 수
- 표본에서 불량품의 수 (복원 추출)
- $n$명의 환자 중 치료에 반응한 환자 수

---

## 3. 포아송 분포

### 설정

사건이 일정한 평균 비율 $\lambda$로 독립적으로 발생할 때, 고정된 시간 또는 공간 구간에서 발생하는 사건의 수를 모델링합니다.

### 표기

$$X \sim \text{Poisson}(\lambda), \quad \lambda > 0$$

### PMF

$$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

### 성질

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = \lambda$ |
| 분산 | $\text{Var}(X) = \lambda$ |
| MGF | $M_X(t) = e^{\lambda(e^t - 1)}$ |

특징적 성질: 평균과 분산이 **같습니다** ($= \lambda$).

### 이항분포에 대한 포아송 근사

$n$이 크고, $p$가 작으며, $\lambda = np$가 적당한 크기일 때:

$$\binom{n}{k} p^k (1-p)^{n-k} \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

**경험적 기준**: $n \geq 20$이고 $p \leq 0.05$일 때 (또는 $n \geq 100$이고 $np \leq 10$) 포아송 근사를 사용합니다.

### 예제: 포아송 근사

1000개 품목의 배치에서 불량률이 0.2%입니다. $P(\text{정확히 3개 불량})$을 근사합니다:

$\lambda = np = 1000 \times 0.002 = 2$

$$P(X = 3) \approx \frac{2^3 e^{-2}}{3!} = \frac{8 \cdot 0.1353}{6} \approx 0.1804$$

### 가법 성질

$X \sim \text{Poisson}(\lambda_1)$이고 $Y \sim \text{Poisson}(\lambda_2)$가 독립이면:

$$X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$$

### 활용 사례

- 시간당 이메일 수
- 교차로에서 하루당 교통사고 수
- DNA 분절의 돌연변이 수
- 초당 검출기에 도달하는 광자 수

---

## 4. 기하분포

### 설정

독립 베르누이 시행 수열에서 **첫 번째 성공**까지의 시행 횟수를 셉니다.

### 표기

$$X \sim \text{Geometric}(p)$$

(관례: $X$ = 첫 번째 성공의 시행 번호, $X \in \{1, 2, 3, \ldots\}$.)

### PMF

$$p_X(k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

**대안적 관례**: 일부 교재에서는 $Y$ = 첫 성공 전 실패 횟수 ($Y = X - 1$, $Y \in \{0, 1, 2, \ldots\}$)로 정의하여, $p_Y(k) = (1-p)^k p$를 사용합니다.

### 성질 (첫 번째 성공 관례)

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = 1/p$ |
| 분산 | $\text{Var}(X) = (1-p)/p^2$ |
| MGF | $M_X(t) = \frac{pe^t}{1 - (1-p)e^t}$ ($t < -\ln(1-p)$) |

### 무기억 성질

기하분포는 무기억 성질(memoryless property)을 갖는 **유일한** 이산 분포입니다:

$$P(X > m + n \mid X > m) = P(X > n)$$

**해석**: 이미 $m$번 실패했더라도, 적어도 $n$번 더 시행이 필요할 확률은 처음부터 시작하는 것과 같습니다. 과거의 실패는 미래의 성공에 대한 정보를 제공하지 않습니다.

*증명*:

$$P(X > m + n \mid X > m) = \frac{P(X > m + n)}{P(X > m)} = \frac{(1-p)^{m+n}}{(1-p)^m} = (1-p)^n = P(X > n)$$

### 예제

주사위를 반복하여 던집니다. $X$ = 첫 번째 6이 나올 때까지의 던짐 횟수라 하면, $X \sim \text{Geometric}(1/6)$.

$$E[X] = 6, \quad P(X > 12 \mid X > 6) = P(X > 6) = (5/6)^6 \approx 0.335$$

### 활용 사례

- 첫 성공까지의 시도 횟수
- 이산 시간에서 첫 사건까지의 대기 시간
- 첫 불량품이 발견될 때까지 검사한 부품 수

---

## 5. 음이항분포

### 설정

독립 베르누이 시행 수열에서 $r$번째 성공까지의 시행 횟수를 셉니다.

### 표기

$$X \sim \text{NegBin}(r, p)$$

($X$ = $r$번째 성공의 시행 번호, $X \in \{r, r+1, r+2, \ldots\}$.)

### PMF

$$p_X(k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots$$

**직관**: 처음 $k-1$번의 시행 중 정확히 $r-1$번이 성공 ($\binom{k-1}{r-1}$), 그리고 $k$번째 시행이 $r$번째 성공 (마지막 $p$ 인수 기여).

### 성질

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = r/p$ |
| 분산 | $\text{Var}(X) = r(1-p)/p^2$ |
| MGF | $M_X(t) = \left(\frac{pe^t}{1-(1-p)e^t}\right)^r$ ($t < -\ln(1-p)$) |

### 기하분포와의 연결

$\text{NegBin}(1, p) = \text{Geometric}(p)$.

$X_1, X_2, \ldots, X_r$이 독립 $\text{Geometric}(p)$ 확률변수이면:

$$X_1 + X_2 + \cdots + X_r \sim \text{NegBin}(r, p)$$

### 활용 사례

- $r$명의 적격 참가자를 찾을 때까지 선별한 환자 수
- $r$건의 거래를 성사시킬 때까지의 영업 전화 횟수
- 과분산 계수 데이터 (포아송과 달리 분산이 평균을 초과할 때)

---

## 6. 초기하분포

### 설정

$K$개의 성공을 포함하는 $N$개 품목의 모집단에서 $n$개를 **비복원** 추출합니다. 추출된 성공의 수를 셉니다.

### 표기

$$X \sim \text{Hypergeometric}(N, K, n)$$

### PMF

$$p_X(k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \quad k = \max(0, n-N+K), \ldots, \min(n, K)$$

### 성질

| 성질 | 값 |
|------|-----|
| 평균 | $E[X] = n \cdot K/N$ |
| 분산 | $\text{Var}(X) = n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}$ |

인수 $\frac{N-n}{N-1}$을 **유한 모집단 보정 계수**(Finite Population Correction, FPC)라 합니다. $N \gg n$이면 FPC는 1에 접근하고, 초기하분포는 이항분포에 수렴합니다.

### 이항분포와의 비교

| 특성 | 이항분포 | 초기하분포 |
|------|----------|-----------|
| 표집 | 복원 추출 | 비복원 추출 |
| 시행 | 독립 | 종속 |
| 분산 | $np(1-p)$ | $np(1-p) \cdot \frac{N-n}{N-1}$ |
| 근사 | -- | $N \to \infty$에서 이항분포에 수렴 |

### 예제

52장의 카드 중 에이스 4장이 있습니다. 5장을 비복원 추출할 때, $P(\text{정확히 2장의 에이스})$는?

$$P(X = 2) = \frac{\binom{4}{2}\binom{48}{3}}{\binom{52}{5}} = \frac{6 \times 17296}{2598960} = \frac{103776}{2598960} \approx 0.0399$$

### 활용 사례

- 유한 로트(lot)에서의 품질 관리 표집
- 카드 추출 문제
- 생태학의 포획-재포획 (capture-recapture)
- $2 \times 2$ 분할표에 대한 피셔의 정확 검정

---

## 7. 분포 간의 관계

```
Bernoulli(p)
    |
    | n개의 독립 복사본의 합
    v
Binomial(n, p)
    |
    | n -> inf, p -> 0, np = lambda
    v
Poisson(lambda)

Geometric(p)   = NegBin(1, p)
    |
    | r개의 독립 복사본의 합
    v
NegBin(r, p)

Hypergeometric(N, K, n)
    |
    | N -> inf with K/N = p
    v
Binomial(n, p)
```

### 극한 관계 요약

| 시작 | 도달 | 조건 |
|------|------|------|
| Binomial$(n, p)$ | Poisson$(\lambda)$ | $n \to \infty$, $p \to 0$, $np = \lambda$ |
| Hypergeometric$(N, K, n)$ | Binomial$(n, p)$ | $N \to \infty$, $K/N \to p$ |
| Binomial$(n, p)$ | Normal$(np, np(1-p))$ | $n \to \infty$ (CLT, 레슨 11) |

---

## 8. Python 예제

### 베르누이와 이항분포 시뮬레이션

```python
import random
import math

def simulate_binomial(n, p, num_trials=100000):
    """Simulate Binomial(n, p) and compare to theoretical values."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        x = sum(1 for _ in range(n) if random.random() < p)
        samples.append(x)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"Binomial(n={n}, p={p})")
    print(f"  Theoretical: mean={n*p:.4f}, var={n*p*(1-p):.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

simulate_binomial(20, 0.3)
```

### 포아송 시뮬레이션과 근사

```python
import random
import math

def simulate_poisson(lam, num_trials=100000):
    """Simulate Poisson(lam) using the inverse transform method."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        L = math.exp(-lam)
        k = 0
        prob = 1.0
        while prob > L:
            k += 1
            prob *= random.random()
        samples.append(k - 1)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"Poisson(lambda={lam})")
    print(f"  Theoretical: mean={lam:.4f}, var={lam:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")
    return samples

samples = simulate_poisson(5.0)
```

### 포아송 근사와 이항분포 비교

```python
import math

def poisson_approximation_comparison(n, p, k_max=15):
    """Compare exact Binomial PMF with Poisson approximation."""
    lam = n * p
    print(f"Binomial(n={n}, p={p}) vs Poisson(lambda={lam})")
    print(f"{'k':>3}  {'Binomial':>12}  {'Poisson':>12}  {'Abs Error':>12}")
    print("-" * 45)

    for k in range(k_max + 1):
        # Exact Binomial
        binom_pmf = (math.comb(n, k) * p**k * (1-p)**(n-k))

        # Poisson approximation
        poisson_pmf = (lam**k * math.exp(-lam)) / math.factorial(k)

        error = abs(binom_pmf - poisson_pmf)
        print(f"{k:3d}  {binom_pmf:12.8f}  {poisson_pmf:12.8f}  {error:12.8f}")

# Good approximation: large n, small p
poisson_approximation_comparison(n=100, p=0.03)
print()
# Poor approximation: small n, large p
poisson_approximation_comparison(n=10, p=0.3)
```

### 기하분포와 무기억 성질

```python
import random

def simulate_geometric(p, num_trials=200000):
    """Simulate Geometric(p) and verify memoryless property."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        k = 1
        while random.random() >= p:  # Keep going until success
            k += 1
        samples.append(k)

    emp_mean = sum(samples) / num_trials
    print(f"Geometric(p={p})")
    print(f"  Theoretical mean = {1/p:.4f}")
    print(f"  Empirical mean   = {emp_mean:.4f}")

    # Verify memoryless property: P(X > m+n | X > m) = P(X > n)
    m, n = 5, 3
    count_gt_m = sum(1 for x in samples if x > m)
    count_gt_m_plus_n = sum(1 for x in samples if x > m + n)
    count_gt_n = sum(1 for x in samples if x > n)

    cond_prob = count_gt_m_plus_n / count_gt_m if count_gt_m > 0 else 0
    uncond_prob = count_gt_n / num_trials

    print(f"\n  Memoryless property (m={m}, n={n}):")
    print(f"  P(X>{m+n} | X>{m}) = {cond_prob:.4f}")
    print(f"  P(X>{n})           = {uncond_prob:.4f}")
    print(f"  Theoretical        = {(1-p)**n:.4f}")

simulate_geometric(p=1/6)
```

### 음이항분포

```python
import random

def simulate_negative_binomial(r, p, num_trials=100000):
    """Simulate NegBin(r, p) as sum of r Geometric(p) variables."""
    random.seed(42)
    samples = []
    for _ in range(num_trials):
        total = 0
        for _ in range(r):
            # One Geometric(p) sample
            k = 1
            while random.random() >= p:
                k += 1
            total += k
        samples.append(total)

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    print(f"NegBin(r={r}, p={p})")
    print(f"  Theoretical: mean={r/p:.4f}, var={r*(1-p)/p**2:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

simulate_negative_binomial(r=5, p=0.4)
```

### 초기하분포

```python
import random
import math

def hypergeometric_pmf(k, N, K, n):
    """Compute P(X=k) for Hypergeometric(N, K, n)."""
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

def simulate_hypergeometric(N, K, n, num_trials=100000):
    """Simulate Hypergeometric by drawing without replacement."""
    random.seed(42)
    population = [1] * K + [0] * (N - K)  # 1 = success, 0 = failure
    samples = []

    for _ in range(num_trials):
        draw = random.sample(population, n)
        samples.append(sum(draw))

    emp_mean = sum(samples) / num_trials
    emp_var = sum((x - emp_mean)**2 for x in samples) / (num_trials - 1)

    theo_mean = n * K / N
    theo_var = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))

    print(f"Hypergeometric(N={N}, K={K}, n={n})")
    print(f"  Theoretical: mean={theo_mean:.4f}, var={theo_var:.4f}")
    print(f"  Empirical:   mean={emp_mean:.4f}, var={emp_var:.4f}")

    # PMF comparison
    print(f"\n  {'k':>3}  {'Exact PMF':>12}  {'Empirical':>12}")
    print("  " + "-" * 30)
    from collections import Counter
    counts = Counter(samples)
    k_min = max(0, n - (N - K))
    k_max = min(n, K)
    for k in range(k_min, k_max + 1):
        exact = hypergeometric_pmf(k, N, K, n)
        emp = counts.get(k, 0) / num_trials
        print(f"  {k:3d}  {exact:12.6f}  {emp:12.6f}")

# Card example: 52 cards, 4 aces, draw 5
simulate_hypergeometric(N=52, K=4, n=5)
```

### 분포 요약 표

```python
import math

def distribution_summary():
    """Print a summary table of all discrete distributions."""
    distributions = [
        ("Bernoulli(p)", "p", "p(1-p)", "{0,1}"),
        ("Binomial(n,p)", "np", "np(1-p)", "{0,...,n}"),
        ("Poisson(lam)", "lam", "lam", "{0,1,2,...}"),
        ("Geometric(p)", "1/p", "(1-p)/p^2", "{1,2,3,...}"),
        ("NegBin(r,p)", "r/p", "r(1-p)/p^2", "{r,r+1,...}"),
        ("Hypergeo(N,K,n)", "nK/N", "nK(N-K)(N-n)/[N^2(N-1)]", "{0,...,min(n,K)}"),
    ]

    header = f"{'Distribution':<20} {'Mean':<12} {'Variance':<18} {'Support':<15}"
    print(header)
    print("-" * len(header))
    for name, mean, var, support in distributions:
        print(f"{name:<20} {mean:<12} {var:<18} {support:<15}")

distribution_summary()
```

---

## 9. 핵심 요약

1. **베르누이**는 기본 구성 단위입니다: 단일 이진 시행. 나머지 모든 것이 여기서 구축됩니다.

2. **이항분포**는 $n$번의 독립 시행에서 성공 횟수를 셉니다. 평균 $np$와 분산 $np(1-p)$는 기댓값의 선형성과 독립성으로부터 우아하게 유도됩니다.

3. **포아송**은 희귀 사건의 수를 모델링합니다. 정의적 특성은 $E[X] = \text{Var}(X) = \lambda$입니다. $n$이 크고 $p$가 작을 때 이항분포를 근사합니다.

4. **기하분포**는 첫 성공까지의 대기 시간을 모델링합니다. 무기억 성질 ($P(X > m+n \mid X > m) = P(X > n)$)은 지수분포의 이산 유사체입니다.

5. **음이항분포**는 기하분포를 $r$번째 성공 대기로 일반화합니다. $r$개의 독립 기하 확률변수의 합과 같습니다.

6. **초기하분포**는 유한 모집단에서의 비복원 추출을 다룹니다. 모집단 크기가 커지면 이항분포에 수렴하며, 유한 모집단 보정 계수가 그 차이를 연결합니다.

7. **올바른 분포 선택**은 문제의 구조에 달려 있습니다:
   - 고정된 $n$번 시행, 독립, 복원 추출 -> **이항분포**
   - 고정 구간에서의 희귀 사건 수 -> **포아송**
   - 첫 번째/$r$번째 성공 대기 -> **기하 / 음이항분포**
   - 비복원 추출 -> **초기하분포**

---

*이전: [05 - 결합 분포](./05_Joint_Distributions.md) | 다음: [07 - 연속 분포족](./07_Continuous_Distribution_Families.md)*

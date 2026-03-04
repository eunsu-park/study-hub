# 확률과정: 입문

**이전**: [회귀분석과 분산분석](./17_Regression_and_ANOVA.md) | **다음**: (현재 시리즈 종료)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 확률과정 (stochastic process)을 정의하고 시간 및 상태 공간에 따라 분류하기
2. 마르코프 체인 (Markov chain)을 정식화하고 전이 확률을 계산하기
3. 다단계 전이에 대해 채프먼-콜모고로프 방정식 (Chapman-Kolmogorov equations) 적용하기
4. 상태를 일시적 (transient), 재귀적 (recurrent), 흡수 (absorbing) 상태로 분류하기
5. 에르고딕 마르코프 체인의 정상 분포 (stationary distribution) 계산하기
6. 포아송 과정 (Poisson process)과 그 기본 성질을 기술하기
7. 랜덤 워크 (random walk, 단순 및 편향 있는 경우) 분석하기
8. 도박꾼의 파산 문제 (gambler's ruin problem) 풀기
9. 강정상성 (strict-sense stationarity)과 약정상성 (wide-sense stationarity) 구별하기
10. 자기상관 함수 (autocorrelation function)를 계산하고 해석하기

---

확률과정은 시간 (또는 공간)으로 인덱스된 확률 변수의 모음입니다. 이는 무작위로 진화하는 시스템을 모형화하기 위한 수학적 프레임워크를 제공하며, 물리학, 금융, 생물학, 컴퓨터 과학, 공학 등에서 응용됩니다.

---

## 1. 정의와 분류

### 1.1 형식적 정의

**확률과정 (stochastic process)**은 공통 확률 공간 위에 정의된 확률 변수의 모음 $\{X(t) : t \in T\}$이며, 여기서 $T$는 **인덱스 집합** (일반적으로 시간을 나타냄)입니다.

### 1.2 분류

| | 이산 상태 | 연속 상태 |
|---|---|---|
| **이산 시간** | 마르코프 체인, 랜덤 워크 | AR 과정, 가우시안 수열 |
| **연속 시간** | 출생-사멸 과정 | 브라운 운동, 포아송 과정 |

- **이산 시간**: $T = \{0, 1, 2, \ldots\}$, $X_0, X_1, X_2, \ldots$으로 표기
- **연속 시간**: $T = [0, \infty)$, $X(t)$로 표기
- **상태 공간 (state space)** $S$: $X(t)$가 취할 수 있는 값들의 집합

### 1.3 주요 성질

- **표본 경로 (sample path, realization)**: 고정된 결과 $\omega$에 대한 특정 궤적 $\{X(t, \omega) : t \in T\}$.
- **유한 차원 분포 (finite-dimensional distributions)**: 임의의 유한개 시점 모음에 대한 결합 분포 $(X(t_1), X(t_2), \ldots, X(t_n))$가 과정을 특성화합니다 (콜모고로프 일관성 정리, Kolmogorov consistency theorem).

---

## 2. 마르코프 체인

### 2.1 마르코프 성질

이산 시간 확률과정 $\{X_n\}$이 **마르코프 체인**이 되려면:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

미래는 현재에만 의존하고 과거에는 의존하지 않습니다. 이것이 **무기억 성질 (memoryless property)**입니다.

### 2.2 전이 행렬

상태 공간 $S = \{1, 2, \ldots, m\}$인 **시간 동질적 (time-homogeneous)** 마르코프 체인의 경우:

$$p_{ij} = P(X_{n+1} = j \mid X_n = i)$$

**전이 행렬 (transition matrix)** $P$의 원소는 $p_{ij}$입니다:

$$P = \begin{pmatrix} p_{11} & p_{12} & \cdots & p_{1m} \\ p_{21} & p_{22} & \cdots & p_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ p_{m1} & p_{m2} & \cdots & p_{mm} \end{pmatrix}$$

성질: $p_{ij} \geq 0$이고 각 행 $i$에 대해 $\sum_j p_{ij} = 1$ (각 행은 확률 분포).

### 2.3 채프먼-콜모고로프 방정식

$n$단계 전이 확률은 다음을 만족합니다:

$$p_{ij}^{(n)} = P(X_n = j \mid X_0 = i) = (P^n)_{ij}$$

$$p_{ij}^{(n+m)} = \sum_k p_{ik}^{(n)} \cdot p_{kj}^{(m)}$$

행렬 형식: $P^{(n+m)} = P^{(n)} \cdot P^{(m)}$.

```python
def matrix_power(P, n):
    """Compute P^n for a square matrix P (list of lists)."""
    size = len(P)
    # Start with identity matrix
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    base = [row[:] for row in P]  # copy
    while n > 0:
        if n % 2 == 1:
            result = mat_mult(result, base)
        base = mat_mult(base, base)
        n //= 2
    return result

def mat_mult(A, B):
    """Multiply two square matrices."""
    size = len(A)
    C = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Example: Weather Markov chain
# States: 0=Sunny, 1=Cloudy, 2=Rainy
P = [[0.7, 0.2, 0.1],
     [0.3, 0.4, 0.3],
     [0.2, 0.3, 0.5]]

print("Transition matrix P:")
for row in P:
    print([f"{x:.1f}" for x in row])

# 2-step transitions
P2 = matrix_power(P, 2)
print("\nP^2 (2-step transitions):")
for row in P2:
    print([f"{x:.4f}" for x in row])

# 10-step transitions
P10 = matrix_power(P, 10)
print("\nP^10 (10-step transitions):")
for row in P10:
    print([f"{x:.4f}" for x in row])
```

---

## 3. 상태의 분류

### 3.1 접근성과 소통

- 어떤 $n \geq 0$에 대해 $p_{ij}^{(n)} > 0$이면 상태 $j$는 상태 $i$로부터 **접근 가능 (accessible)**합니다.
- 상태 $i$와 $j$가 서로 접근 가능하면 **소통한다 (communicate)**고 합니다. $i \leftrightarrow j$로 표기합니다.
- 소통은 동치 관계 (equivalence relation)이며, 상태 공간을 **소통 클래스 (communicating classes)**로 분할합니다.
- 소통 클래스가 단 하나뿐이면 (모든 상태가 소통하면) 마르코프 체인은 **기약 (irreducible)**입니다.

### 3.2 일시적 상태와 재귀적 상태

$f_{ii}$를 상태 $i$에서 출발하여 $i$로 다시 돌아올 확률이라 하면:

- **재귀적 (recurrent)**: $f_{ii} = 1$ (체인이 확률 1로 $i$에 돌아옴).
- **일시적 (transient)**: $f_{ii} < 1$ (돌아오지 않을 양의 확률이 존재).

**판별 기준**: 상태 $i$가 재귀적일 필요충분조건은 $\sum_{n=0}^{\infty} p_{ii}^{(n)} = \infty$입니다.

### 3.3 흡수 상태

상태 $i$가 $p_{ii} = 1$이면 **흡수 (absorbing)** 상태입니다 (한번 진입하면 영원히 머뭅니다). 흡수 상태는 자명하게 재귀적입니다.

### 3.4 주기성

상태 $i$의 **주기 (period)**는 $d_i = \gcd\{n \geq 1 : p_{ii}^{(n)} > 0\}$입니다.
- $d_i = 1$: 상태 $i$는 **비주기적 (aperiodic)**.
- $d_i > 1$: 상태 $i$는 주기 $d_i$의 **주기적 (periodic)** 상태.

모든 상태가 비주기적이고 재귀적인 기약 마르코프 체인을 **에르고딕 (ergodic)**이라 합니다.

---

## 4. 정상 분포

### 4.1 정의

확률 벡터 $\pi = (\pi_1, \pi_2, \ldots, \pi_m)$가 **정상 분포 (stationary distribution)**이려면:

$$\pi P = \pi, \quad \sum_i \pi_i = 1, \quad \pi_i \geq 0$$

체인이 분포 $\pi$로 시작하면, 모든 미래 시점에서도 분포 $\pi$를 유지합니다.

### 4.2 존재성과 유일성

- 기약이고 양재귀적 (positive recurrent)인 마르코프 체인은 **유일한** 정상 분포를 가집니다.
- 유한 상태 기약 체인의 경우, 정상 분포는 항상 존재하고 유일합니다.
- 에르고딕 체인의 경우: 초기 상태 $i$에 무관하게 $\lim_{n \to \infty} P^n_{ij} = \pi_j$.

### 4.3 정상 분포의 계산

제약 조건 $\sum \pi_i = 1$과 함께 연립방정식 $\pi P = \pi$를 풉니다.

```python
def stationary_distribution(P, iterations=1000):
    """Find stationary distribution by repeated matrix multiplication.

    For an ergodic chain, any row of P^n converges to pi.
    """
    size = len(P)
    Pn = matrix_power(P, iterations)
    # Each row should converge to the same distribution
    pi = Pn[0]
    return pi

def stationary_by_solving(P):
    """Find stationary distribution by solving pi*P = pi.

    For a 3x3 matrix using simple Gaussian-like approach.
    Solves (P^T - I)pi = 0 with sum(pi) = 1.
    """
    size = len(P)
    # Set up equations: pi_j = sum_i pi_i * P_ij
    # Rearrange: sum_i pi_i * (P_ij - delta_ij) = 0
    # Replace last equation with sum(pi) = 1

    # Build augmented matrix [A | b]
    A = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            A[j][i] = P[i][j]  # Transpose
    for i in range(size):
        A[i][i] -= 1.0

    # Replace last row with sum constraint
    A[-1] = [1.0] * size
    b = [0.0] * size
    b[-1] = 1.0

    # Gaussian elimination (partial pivoting)
    for col in range(size):
        # Find pivot
        max_row = col
        for row in range(col + 1, size):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]
        b[col], b[max_row] = b[max_row], b[col]

        for row in range(col + 1, size):
            if A[col][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for j in range(size):
                A[row][j] -= factor * A[col][j]
            b[row] -= factor * b[col]

    # Back substitution
    pi = [0.0] * size
    for i in range(size - 1, -1, -1):
        pi[i] = b[i]
        for j in range(i + 1, size):
            pi[i] -= A[i][j] * pi[j]
        if A[i][i] != 0:
            pi[i] /= A[i][i]
    return pi

# Weather chain stationary distribution
pi_iter = stationary_distribution(P)
pi_exact = stationary_by_solving(P)
print("Stationary distribution (iterative):", [f"{x:.4f}" for x in pi_iter])
print("Stationary distribution (solving):  ", [f"{x:.4f}" for x in pi_exact])
print("Interpretation: Long-run fraction of Sunny/Cloudy/Rainy days")
```

---

## 5. 포아송 과정

### 5.1 정의

계수 과정 (counting process) $\{N(t), t \geq 0\}$가 비율 $\lambda > 0$인 **포아송 과정 (Poisson process)**이 되려면:

1. $N(0) = 0$
2. 독립 증분 (independent increments): 겹치지 않는 구간의 계수들은 독립.
3. 모든 $t, s \geq 0$에 대해 $N(t+s) - N(t) \sim \text{Poisson}(\lambda s)$.

### 5.2 성질

- **도착 간 시간 (inter-arrival times)**: $T_1, T_2, \ldots$는 i.i.d. $\text{Exponential}(\lambda)$.
- **도착 시각 (arrival times)**: $S_n = T_1 + \cdots + T_n \sim \text{Gamma}(n, \lambda)$.
- **무기억 성질 (memoryless property)**: $P(T > t + s \mid T > t) = P(T > s)$.
- **합병 (merging)**: 독립인 $N_1(t) \sim \text{Poisson}(\lambda_1 t)$와 $N_2(t) \sim \text{Poisson}(\lambda_2 t)$이면, $N_1(t) + N_2(t) \sim \text{Poisson}((\lambda_1 + \lambda_2)t)$.
- **분할 (splitting)**: 각 사건을 독립적으로 유형 1 (확률 $p$) 또는 유형 2에 배정하면, 비율 $\lambda p$와 $\lambda(1-p)$인 두 개의 독립 포아송 과정이 됩니다.

### 5.3 도착의 조건부 분포

$N(t) = n$이 주어지면, $n$개의 도착 시각은 $n$개의 i.i.d. $\text{Uniform}(0, t)$ 확률 변수의 순서 통계량 (order statistics)과 같은 분포를 따릅니다.

```python
import random
import math

def simulate_poisson_process(lam, T, seed=42):
    """Simulate a Poisson process on [0, T] with rate lambda."""
    random.seed(seed)
    arrivals = []
    t = 0
    while True:
        # Inter-arrival time ~ Exponential(lambda)
        inter_arrival = -math.log(1 - random.random()) / lam
        t += inter_arrival
        if t > T:
            break
        arrivals.append(t)
    return arrivals

# Simulate customer arrivals (rate = 3 per hour) over 8 hours
arrivals = simulate_poisson_process(lam=3, T=8)
print(f"Poisson process (lambda=3, T=8):")
print(f"  Total arrivals: {len(arrivals)} (expected: {3*8})")

# Count arrivals per hour
for hour in range(8):
    count = sum(1 for a in arrivals if hour <= a < hour + 1)
    bar = "#" * count
    print(f"  Hour {hour}-{hour+1}: {count:>2} arrivals  {bar}")

# Inter-arrival times
if len(arrivals) > 1:
    inter_arrivals = [arrivals[0]] + [arrivals[i] - arrivals[i-1] for i in range(1, len(arrivals))]
    mean_ia = sum(inter_arrivals) / len(inter_arrivals)
    print(f"  Mean inter-arrival time: {mean_ia:.3f} (expected: {1/3:.3f})")
```

---

## 6. 랜덤 워크

### 6.1 단순 랜덤 워크

매 단계마다 보행자가 확률 $p$로 $+1$, 확률 $q = 1 - p$로 $-1$만큼 이동합니다:

$$X_n = X_0 + \sum_{i=1}^{n} Z_i, \quad Z_i = \begin{cases} +1 & \text{prob } p \\ -1 & \text{prob } q \end{cases}$$

**성질**:
- $E[X_n] = X_0 + n(p - q) = X_0 + n(2p - 1)$
- $\text{Var}(X_n) = 4npq$
- **대칭인 경우** ($p = 1/2$): 1차원과 2차원에서 재귀적, 3차원 이상에서 일시적.

### 6.2 편향 있는 랜덤 워크

$p \neq 1/2$일 때, 워크에 편향 (drift)이 있습니다:

$$E[X_n] = X_0 + n\mu, \quad \mu = p - q$$

$p > 1/2$이면 상향 편향, $p < 1/2$이면 하향 편향입니다.

```python
import random

def simulate_random_walks(n_steps, n_walks, p=0.5, seed=42):
    """Simulate multiple random walks."""
    random.seed(seed)
    walks = []
    for _ in range(n_walks):
        position = 0
        path = [position]
        for _ in range(n_steps):
            step = 1 if random.random() < p else -1
            position += step
            path.append(position)
        walks.append(path)
    return walks

# Symmetric random walk
walks = simulate_random_walks(n_steps=100, n_walks=5, p=0.5)
print("Symmetric Random Walk (p=0.5), 5 walks of 100 steps:")
print(f"{'Walk':<6} {'Final':>6} {'Max':>6} {'Min':>6}")
for i, w in enumerate(walks):
    print(f"{i+1:<6} {w[-1]:>6} {max(w):>6} {min(w):>6}")

# Random walk with drift
walks_drift = simulate_random_walks(n_steps=100, n_walks=5, p=0.6)
print(f"\nRandom Walk with Drift (p=0.6):")
print(f"Expected final position: {100 * (0.6 - 0.4):.0f}")
print(f"{'Walk':<6} {'Final':>6}")
for i, w in enumerate(walks_drift):
    print(f"{i+1:<6} {w[-1]:>6}")
```

---

## 7. 도박꾼의 파산 문제

### 7.1 설정

도박꾼이 $i$ 달러를 가지고 시작하여 공정 (또는 편향된) 게임을 합니다. 매 라운드: 확률 $p$로 \$1을 얻고, 확률 $q = 1 - p$로 \$1을 잃습니다. 도박꾼이 $N$ 달러에 도달하면 (승리) 또는 $0$에 도달하면 (파산) 게임이 끝납니다.

### 7.2 파산 확률

$r_i = P(\text{파산} \mid X_0 = i)$라 하면:

**공정한 게임** ($p = q = 1/2$):

$$r_i = 1 - \frac{i}{N}$$

**편향된 게임** ($p \neq q$):

$$r_i = \frac{(q/p)^i - (q/p)^N}{1 - (q/p)^N}$$

### 7.3 기대 지속 시간

$i$에서 시작하는 공정한 게임의 경우:

$$E[\text{지속 시간}] = i(N - i)$$

```python
import random

def gamblers_ruin_analytical(i, N, p):
    """Compute ruin probability analytically."""
    q = 1 - p
    if abs(p - 0.5) < 1e-10:  # fair game
        return 1 - i / N
    else:
        ratio = q / p
        return (ratio**i - ratio**N) / (1 - ratio**N)

def gamblers_ruin_simulation(i, N, p, n_simulations=10000, seed=42):
    """Simulate the gambler's ruin problem."""
    random.seed(seed)
    ruins = 0
    total_steps = 0
    for _ in range(n_simulations):
        position = i
        steps = 0
        while 0 < position < N:
            position += 1 if random.random() < p else -1
            steps += 1
        if position == 0:
            ruins += 1
        total_steps += steps

    return {
        "ruin_prob": ruins / n_simulations,
        "avg_duration": total_steps / n_simulations
    }

# Example: start with $20, target $100
i, N = 20, 100
for p in [0.5, 0.49, 0.48, 0.45]:
    analytical = gamblers_ruin_analytical(i, N, p)
    sim = gamblers_ruin_simulation(i, N, p, n_simulations=5000)
    print(f"p={p:.2f}: P(ruin) analytical={analytical:.4f}, simulated={sim['ruin_prob']:.4f}, "
          f"avg duration={sim['avg_duration']:.0f}")
```

---

## 8. 정상성

### 8.1 강정상성 (Strict-Sense Stationarity, SSS)

과정 $\{X(t)\}$가 **강정상적 (strictly stationary)**이려면, 모든 $n$, 모든 시점 $t_1, \ldots, t_n$, 모든 이동 $\tau$에 대해:

$$(X(t_1), \ldots, X(t_n)) \overset{d}{=} (X(t_1 + \tau), \ldots, X(t_n + \tau))$$

모든 유한 차원 분포가 시간 이동에 불변합니다. 이것은 매우 강한 조건입니다.

### 8.2 약정상성 (Wide-Sense Stationarity, WSS)

과정이 **약정상적 (wide-sense stationary)**이려면:

1. $E[X(t)] = \mu$ (일정한 평균, $t$에 무관)
2. $\text{Cov}(X(t), X(t+\tau)) = C(\tau)$ (자기공분산이 시차 $\tau$에만 의존하고 $t$에는 의존하지 않음)
3. $E[|X(t)|^2] < \infty$

강정상성은 (이차 모멘트가 존재할 때) 약정상성을 함의하지만, 약정상성은 일반적으로 강정상성을 함의하지 않습니다. 예외: 가우시안 과정 (Gaussian process)에서는 약정상성과 강정상성이 동치입니다.

---

## 9. 자기상관 함수

### 9.1 정의

평균 $\mu$인 약정상 과정에 대해:

**자기공분산 함수 (autocovariance function)**:
$$C(\tau) = \text{Cov}(X(t), X(t+\tau)) = E[(X(t) - \mu)(X(t+\tau) - \mu)]$$

**자기상관 함수 (autocorrelation function, ACF)**:
$$R(\tau) = \frac{C(\tau)}{C(0)} = \frac{\text{Cov}(X(t), X(t+\tau))}{\text{Var}(X(t))}$$

### 9.2 성질

- $C(0) = \text{Var}(X(t)) \geq 0$
- $C(\tau) = C(-\tau)$ (대칭)
- $|C(\tau)| \leq C(0)$ (유계)
- $C(\tau)$는 양의 준정부호 (positive semi-definite)

### 9.3 표본 자기상관

시계열 $x_1, \ldots, x_n$에 대해:

$$\hat{R}(k) = \frac{\sum_{t=1}^{n-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{n}(x_t - \bar{x})^2}$$

```python
def sample_acf(data, max_lag=None):
    """Compute sample autocorrelation function."""
    n = len(data)
    if max_lag is None:
        max_lag = min(n // 4, 20)
    mean = sum(data) / n
    var = sum((x - mean)**2 for x in data)

    acf_values = []
    for k in range(max_lag + 1):
        cov_k = sum((data[t] - mean) * (data[t + k] - mean) for t in range(n - k))
        acf_values.append(cov_k / var if var > 0 else 0)
    return acf_values

# Example: ACF of a simulated AR(1) process
# X_t = 0.8 * X_{t-1} + epsilon_t
random.seed(42)
n = 200
phi = 0.8
x = [0.0]
for t in range(1, n):
    x.append(phi * x[-1] + random.gauss(0, 1))

acf = sample_acf(x, max_lag=10)
print("Sample ACF of AR(1) process (phi=0.8):")
print(f"Theoretical ACF: R(k) = 0.8^k")
print(f"{'Lag':>4} {'Sample':>8} {'Theory':>8}")
for k, r in enumerate(acf):
    print(f"{k:>4} {r:>8.4f} {phi**k:>8.4f}")
```

---

## 10. 마르코프 체인 시뮬레이션: 종합 예제

```python
import random

def simulate_markov_chain(P, initial_state, n_steps, seed=42):
    """Simulate a Markov chain trajectory.

    Args:
        P: transition matrix (list of lists)
        initial_state: starting state index
        n_steps: number of transitions
        seed: random seed
    """
    random.seed(seed)
    states = [initial_state]
    current = initial_state

    for _ in range(n_steps):
        r = random.random()
        cumsum = 0
        for j, prob in enumerate(P[current]):
            cumsum += prob
            if r < cumsum:
                current = j
                break
        states.append(current)
    return states

# Two-state Markov chain (healthy/sick)
P_health = [[0.95, 0.05],  # Healthy -> Healthy, Healthy -> Sick
            [0.30, 0.70]]  # Sick -> Healthy, Sick -> Sick

labels = ["Healthy", "Sick"]
chain = simulate_markov_chain(P_health, initial_state=0, n_steps=365)

# Count state frequencies
counts = [0, 0]
for s in chain:
    counts[s] += 1

print("=== Health Markov Chain (365 days) ===")
print(f"Days Healthy: {counts[0]}, Days Sick: {counts[1]}")
print(f"Empirical: Healthy={counts[0]/len(chain):.3f}, Sick={counts[1]/len(chain):.3f}")

# Analytical stationary distribution
pi = stationary_by_solving(P_health)
print(f"Stationary:  Healthy={pi[0]:.3f}, Sick={pi[1]:.3f}")

# Absorbing Markov chain example
print("\n=== Absorbing Markov Chain ===")
# States: 0=Start, 1=Middle, 2=Win(absorbing), 3=Lose(absorbing)
P_abs = [[0.0, 0.6, 0.3, 0.1],
         [0.2, 0.0, 0.5, 0.3],
         [0.0, 0.0, 1.0, 0.0],  # Absorbing (Win)
         [0.0, 0.0, 0.0, 1.0]]  # Absorbing (Lose)

n_sim = 10000
wins = 0
total_steps_to_absorb = 0
random.seed(42)
for _ in range(n_sim):
    state = 0
    steps = 0
    while state not in [2, 3]:
        r = random.random()
        cumsum = 0
        for j, prob in enumerate(P_abs[state]):
            cumsum += prob
            if r < cumsum:
                state = j
                break
        steps += 1
    if state == 2:
        wins += 1
    total_steps_to_absorb += steps

print(f"P(Win) from state 0: {wins/n_sim:.4f}")
print(f"P(Lose) from state 0: {(n_sim-wins)/n_sim:.4f}")
print(f"Avg steps to absorption: {total_steps_to_absorb/n_sim:.2f}")
```

---

## 11. 핵심 요약

| 개념 | 핵심 포인트 |
|---|---|
| 확률과정 | 확률 변수의 모음 $\{X(t), t \in T\}$; 시간과 상태 공간에 따라 분류 |
| 마르코프 성질 | 미래는 현재에만 의존하고 과거에는 의존하지 않음 |
| 전이 행렬 | $P^n$이 $n$단계 전이 확률을 제공 |
| 상태 분류 | 재귀적 ($f_{ii} = 1$), 일시적 ($f_{ii} < 1$), 흡수 ($p_{ii} = 1$) |
| 정상 분포 | $\pi P = \pi$; 각 상태에서 보내는 장기적 비율 |
| 포아송 과정 | 무기억 계수 과정; 지수 분포 도착 간 시간 |
| 랜덤 워크 | 1차원/2차원에서 재귀적, 3차원 이상에서 일시적; $E[X_n] = n(2p-1)$ |
| 도박꾼의 파산 | 공정한 게임에서 $P(\text{파산}) = 1 - i/N$; 공정한 무한 게임에서 항상 파산 |
| 약정상성 | 일정한 평균, 자기공분산이 시차에만 의존 |
| ACF | $R(\tau)$: 시차 $\tau$에서의 선형 의존성 측정 |

**앞으로의 전망**: 이 기초 개념들은 연속 시간 과정 (브라운 운동), 마팅게일 (martingale), 마르코프 체인 몬테 카를로 (MCMC), 은닉 마르코프 모형 (hidden Markov model), 확률 미적분학 (stochastic calculus) -- 금융, 신호 처리, 현대 기계 학습의 수학적 기반 -- 으로 이어집니다.

---

**이전**: [회귀분석과 분산분석](./17_Regression_and_ANOVA.md) | **다음**: (현재 시리즈 종료)

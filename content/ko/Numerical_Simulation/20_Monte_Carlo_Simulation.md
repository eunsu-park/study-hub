# 몬테카를로 시뮬레이션

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 몬테카를로(Monte Carlo) 방법의 원리를 설명하고 결정론적 방법보다 유리한 문제 유형을 파악할 수 있습니다.
2. 몬테카를로 적분(Monte Carlo Integration)을 구현하고 큰 수의 법칙(Law of Large Numbers)이 수렴과 오차 스케일링을 어떻게 지배하는지 분석할 수 있습니다.
3. 물리 및 금융 모델에서 확률과 기댓값 추정에 몬테카를로 시뮬레이션을 적용할 수 있습니다.
4. 마르코프 연쇄 몬테카를로(MCMC, Markov Chain Monte Carlo) 샘플링을 구현하고 메트로폴리스-헤이스팅스(Metropolis-Hastings) 알고리즘을 설명할 수 있습니다.
5. 분산 감소 기법(Variance Reduction Technique)(중요도 샘플링(Importance Sampling), 제어 변량(Control Variates))을 비교하고 효과를 평가할 수 있습니다.
6. 실제 문제에 대한 완전한 몬테카를로 시뮬레이션을 설계하고 실행하여 통계적 결과를 해석할 수 있습니다.

---

## 개요

몬테카를로(Monte Carlo) 방법은 난수를 이용하여 수치적 결과를 얻는 확률적 알고리즘입니다. 복잡한 적분, 최적화, 물리 시스템 시뮬레이션 등 다양한 분야에서 활용됩니다.

**이 레슨이 중요한 이유:** 지금까지 다룬 모든 방법은 결정론적(deterministic)입니다 -- 같은 입력이 주어지면 같은 출력을 생성합니다. 몬테카를로 방법은 대신 무작위성(randomness)을 받아들여, 결정론적으로 계산하기에 비용이 너무 높은 양을 랜덤 샘플링으로 추정합니다. 이 방법의 초능력: 수렴 속도 $O(1/\sqrt{N})$이 차원에 무관하므로, 격자 기반 방법이 차원의 저주(curse of dimensionality)로 인해 무력해지는 고차원 적분(10차원 이상)에서 몬테카를로가 최선의 선택이 됩니다.

---

## 1. 몬테카를로 방법 소개

### 1.1 역사와 개념

```python
"""
몬테카를로 방법의 역사:
- 1940년대 맨해튼 프로젝트에서 Stanislaw Ulam, John von Neumann이 개발
- 이름은 모나코의 몬테카를로 카지노에서 유래
- 핵심 아이디어: 무작위 샘플링으로 결정론적 문제를 해결

응용 분야:
- 수치 적분 (고차원 적분)
- 통계 물리학 (Ising 모델, 분자 동역학)
- 금융 공학 (옵션 가격 결정, 리스크 분석)
- 컴퓨터 그래픽스 (광선 추적)
- 기계 학습 (MCMC, 베이지안 추론)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
```

### 1.2 기본 원리

```python
def monte_carlo_principle():
    """
    몬테카를로 적분의 기본 원리

    ∫f(x)dx ≈ (b-a)/N * Σf(xᵢ)

    여기서 xᵢ는 [a, b]에서 균등하게 샘플링
    """
    # 예시: ∫₀¹ x² dx = 1/3

    def f(x):
        return x**2

    N_values = [100, 1000, 10000, 100000]

    print("∫₀¹ x² dx = 1/3 ≈ 0.3333...")
    print()

    for N in N_values:
        samples = np.random.uniform(0, 1, N)
        estimate = np.mean(f(samples))  # (b-a) = 1
        error = abs(estimate - 1/3)
        print(f"N = {N:6d}: 추정값 = {estimate:.6f}, 오차 = {error:.6f}")

monte_carlo_principle()
```

---

## 2. 난수 생성

### 2.1 의사 난수 (Pseudo-random Numbers)

```python
def random_number_basics():
    """NumPy 난수 생성기 기초"""

    # 시드 설정 (재현 가능성)
    rng = np.random.default_rng(seed=42)

    # 균등 분포 [0, 1)
    uniform = rng.random(5)
    print(f"균등 분포: {uniform}")

    # 정수 난수
    integers = rng.integers(1, 7, size=10)  # 주사위
    print(f"주사위 10회: {integers}")

    # 정규 분포
    normal = rng.normal(loc=0, scale=1, size=5)
    print(f"표준 정규: {normal}")

    # 기타 분포
    exponential = rng.exponential(scale=1.0, size=5)
    poisson = rng.poisson(lam=5, size=5)

    print(f"지수 분포: {exponential}")
    print(f"포아송: {poisson}")

random_number_basics()
```

### 2.2 분포에서 샘플링

```python
def distribution_sampling():
    """다양한 확률 분포에서 샘플링"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_samples = 10000

    # 1. 균등 분포
    samples = np.random.uniform(-1, 1, n_samples)
    axes[0, 0].hist(samples, bins=50, density=True, alpha=0.7)
    axes[0, 0].set_title('균등 분포 U(-1, 1)')

    # 2. 정규 분포
    samples = np.random.normal(0, 1, n_samples)
    axes[0, 1].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 100)
    axes[0, 1].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
    axes[0, 1].set_title('정규 분포 N(0, 1)')

    # 3. 지수 분포
    samples = np.random.exponential(1, n_samples)
    axes[0, 2].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 6, 100)
    axes[0, 2].plot(x, stats.expon.pdf(x), 'r-', linewidth=2)
    axes[0, 2].set_title('지수 분포 Exp(1)')

    # 4. 감마 분포
    samples = np.random.gamma(2, 1, n_samples)
    axes[1, 0].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 10, 100)
    axes[1, 0].plot(x, stats.gamma.pdf(x, 2), 'r-', linewidth=2)
    axes[1, 0].set_title('감마 분포 Γ(2, 1)')

    # 5. 베타 분포
    samples = np.random.beta(2, 5, n_samples)
    axes[1, 1].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 1, 100)
    axes[1, 1].plot(x, stats.beta.pdf(x, 2, 5), 'r-', linewidth=2)
    axes[1, 1].set_title('베타 분포 Beta(2, 5)')

    # 6. 카이제곱 분포
    samples = np.random.chisquare(3, n_samples)
    axes[1, 2].hist(samples, bins=50, density=True, alpha=0.7)
    x = np.linspace(0, 15, 100)
    axes[1, 2].plot(x, stats.chi2.pdf(x, 3), 'r-', linewidth=2)
    axes[1, 2].set_title('카이제곱 χ²(3)')

    plt.tight_layout()
    plt.show()

distribution_sampling()
```

### 2.3 역변환 샘플링

```python
def inverse_transform_sampling():
    """
    역변환 방법으로 임의의 분포에서 샘플링

    U ~ Uniform(0,1)이면,
    X = F⁻¹(U)는 CDF가 F인 분포를 따름
    """

    # 예시: 지수 분포
    # CDF: F(x) = 1 - e^(-λx)
    # 역함수: F⁻¹(u) = -ln(1-u)/λ

    def sample_exponential(lam, n):
        u = np.random.uniform(0, 1, n)
        return -np.log(1 - u) / lam

    lam = 2.0
    samples = sample_exponential(lam, 10000)

    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='역변환 샘플')

    x = np.linspace(0, 4, 100)
    plt.plot(x, lam * np.exp(-lam * x), 'r-', linewidth=2,
             label=f'이론: Exp({lam})')

    plt.xlabel('x')
    plt.ylabel('밀도')
    plt.title('역변환 샘플링: 지수 분포')
    plt.legend()
    plt.grid(True)
    plt.show()

inverse_transform_sampling()
```

---

## 3. 몬테카를로 적분

### 3.1 π 추정

```python
def estimate_pi():
    """
    원을 이용한 π 추정

    단위 정사각형 내 점 중 단위원 내부에 있는 비율:
    π/4 = (원 넓이) / (정사각형 넓이)
    """

    def estimate_pi_once(n_points):
        # [-1, 1] x [-1, 1] 정사각형에서 샘플링
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)

        # 원 내부 점의 비율
        inside = x**2 + y**2 <= 1
        return 4 * np.sum(inside) / n_points

    # 수렴 분석
    N_values = np.logspace(1, 6, 20).astype(int)
    estimates = [estimate_pi_once(n) for n in N_values]
    errors = [abs(e - np.pi) for e in estimates]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시각화 (N=1000)
    n = 1000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = x**2 + y**2 <= 1

    axes[0].scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5)
    axes[0].scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    axes[0].add_patch(circle)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'π 추정 (N={n}): {4*np.sum(inside)/n:.4f}')

    # 수렴
    axes[1].loglog(N_values, errors, 'bo-', label='실제 오차')
    axes[1].loglog(N_values, 1/np.sqrt(N_values), 'r--', label='O(1/√N)')
    axes[1].set_xlabel('샘플 수 N')
    axes[1].set_ylabel('|추정값 - π|')
    axes[1].set_title('수렴 속도')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"π = {np.pi:.10f}")
    print(f"추정값 (N=10⁶): {estimates[-1]:.10f}")

estimate_pi()
```

### 3.2 다차원 적분

```python
def multidimensional_integration():
    """
    고차원 적분에서 몬테카를로의 강점

    차원의 저주: 격자 기반 방법은 차원이 높아지면 지수적으로 느려짐
    몬테카를로: 수렴 속도 O(1/√N)이 차원에 무관
    """

    def integrand(x):
        """n차원 가우시안 적분: ∫...∫ exp(-||x||²) dx"""
        return np.exp(-np.sum(x**2, axis=1))

    def mc_integrate(dim, n_samples, limits=(-3, 3)):
        """d차원 입방체에서 적분"""
        # 균등 샘플링
        samples = np.random.uniform(limits[0], limits[1], (n_samples, dim))
        values = integrand(samples)

        # 적분 추정
        volume = (limits[1] - limits[0]) ** dim
        estimate = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)

        return estimate, std_error

    # 이론값: π^(d/2)
    print("다차원 가우시안 적분:")
    print(f"{'차원':<8}{'추정값':<15}{'이론값':<15}{'상대오차':<12}")
    print("-" * 50)

    for dim in [1, 2, 3, 5, 10]:
        estimate, std_err = mc_integrate(dim, 100000)
        theoretical = np.pi ** (dim/2)
        rel_error = abs(estimate - theoretical) / theoretical

        print(f"{dim:<8}{estimate:<15.6f}{theoretical:<15.6f}{rel_error:<12.4%}")

multidimensional_integration()
```

### 3.3 중요도 샘플링

```python
def importance_sampling():
    """
    중요도 샘플링 (Importance Sampling)

    ∫f(x)dx = ∫[f(x)/g(x)]g(x)dx ≈ (1/N)Σ f(xᵢ)/g(xᵢ)

    여기서 xᵢ ~ g(x)

    g(x)가 f(x)와 비슷할수록 분산 감소
    """

    # 예시: ∫₀^∞ x² * e^(-x) dx = 2 (감마 함수)

    def f(x):
        return x**2 * np.exp(-x)

    n_samples = 10000

    # 방법 1: 균등 샘플링 (잘림 적분)
    # ∫₀^10 x² * e^(-x) dx 로 근사
    x_uniform = np.random.uniform(0, 10, n_samples)
    estimate_uniform = 10 * np.mean(f(x_uniform))

    # 방법 2: 중요도 샘플링 (제안 분포: 지수분포)
    # g(x) = e^(-x), f(x)/g(x) = x²
    x_exp = np.random.exponential(1, n_samples)
    estimate_is = np.mean(x_exp**2)  # f(x)/g(x) = x²

    print("∫₀^∞ x² * e^(-x) dx = 2")
    print(f"균등 샘플링 (0~10): {estimate_uniform:.6f}")
    print(f"중요도 샘플링 (Exp): {estimate_is:.6f}")

    # 분산 비교
    var_uniform = 10**2 * np.var(f(x_uniform)) / n_samples
    var_is = np.var(x_exp**2) / n_samples

    print(f"\n분산 비교:")
    print(f"균등 샘플링 분산: {var_uniform:.6f}")
    print(f"중요도 샘플링 분산: {var_is:.6f}")
    print(f"분산 감소율: {var_uniform/var_is:.1f}배")

importance_sampling()
```

---

## 4. 확률적 시뮬레이션

### 4.1 랜덤 워크

```python
def random_walk():
    """1D 및 2D 랜덤 워크"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1D 랜덤 워크
    n_steps = 1000
    n_walks = 5

    for _ in range(n_walks):
        steps = np.random.choice([-1, 1], n_steps)
        position = np.cumsum(steps)
        axes[0].plot(position, alpha=0.7)

    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('시간')
    axes[0].set_ylabel('위치')
    axes[0].set_title('1D 랜덤 워크')
    axes[0].grid(True)

    # 2D 랜덤 워크
    n_steps = 5000
    directions = np.random.randint(0, 4, n_steps)
    dx = np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    dy = np.where(directions == 2, 1, np.where(directions == 3, -1, 0))

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    axes[1].plot(x, y, alpha=0.7, linewidth=0.5)
    axes[1].scatter([0], [0], color='green', s=100, zorder=5, label='시작')
    axes[1].scatter([x[-1]], [y[-1]], color='red', s=100, zorder=5, label='끝')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('2D 랜덤 워크')
    axes[1].legend()
    axes[1].set_aspect('equal')

    # 평균 제곱 변위 (확산)
    n_simulations = 1000
    n_steps = 500
    final_distances = []

    for _ in range(n_simulations):
        steps = np.random.choice([-1, 1], n_steps)
        final_pos = np.sum(steps)
        final_distances.append(final_pos**2)

    print(f"1D 랜덤 워크 ({n_steps} 스텝):")
    print(f"  평균 제곱 변위: {np.mean(final_distances):.2f}")
    print(f"  이론값 (N): {n_steps}")

    # MSD vs 시간
    msd = []
    for t in range(1, n_steps + 1):
        positions = [np.sum(np.random.choice([-1, 1], t)) for _ in range(500)]
        msd.append(np.mean(np.array(positions)**2))

    axes[2].plot(range(1, n_steps + 1), msd, 'b-', alpha=0.7, label='시뮬레이션')
    axes[2].plot(range(1, n_steps + 1), range(1, n_steps + 1), 'r--', label='⟨x²⟩ = t')
    axes[2].set_xlabel('시간 t')
    axes[2].set_ylabel('⟨x²⟩')
    axes[2].set_title('평균 제곱 변위')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

random_walk()
```

### 4.2 브라운 운동

```python
def brownian_motion():
    """기하 브라운 운동 (Geometric Brownian Motion)"""

    # dS = μSdt + σSdW
    # S(t) = S(0) * exp((μ - σ²/2)t + σW(t))

    S0 = 100      # 초기 가격
    mu = 0.1      # 기대 수익률 (연간)
    sigma = 0.2   # 변동성 (연간)
    T = 1.0       # 1년
    n_steps = 252 # 거래일 수
    n_paths = 100

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 경로 시뮬레이션
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, i+1] = paths[:, i] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

    for path in paths[:20]:
        axes[0].plot(t, path, alpha=0.5, linewidth=0.8)

    axes[0].set_xlabel('시간 (년)')
    axes[0].set_ylabel('가격')
    axes[0].set_title('기하 브라운 운동 경로')
    axes[0].grid(True)

    # 최종 가격 분포
    final_prices = paths[:, -1]

    axes[1].hist(final_prices, bins=30, density=True, alpha=0.7, label='시뮬레이션')

    # 이론적 분포: 로그정규
    log_mean = np.log(S0) + (mu - 0.5*sigma**2)*T
    log_std = sigma * np.sqrt(T)

    x = np.linspace(50, 200, 100)
    pdf = stats.lognorm.pdf(x, log_std, scale=np.exp(log_mean))
    axes[1].plot(x, pdf, 'r-', linewidth=2, label='이론 (로그정규)')

    axes[1].set_xlabel('최종 가격')
    axes[1].set_ylabel('밀도')
    axes[1].set_title('최종 가격 분포')
    axes[1].legend()
    axes[1].grid(True)

    print(f"기하 브라운 운동 시뮬레이션:")
    print(f"  초기 가격: {S0}")
    print(f"  평균 최종 가격: {np.mean(final_prices):.2f}")
    print(f"  이론적 기대값: {S0 * np.exp(mu * T):.2f}")

    plt.tight_layout()
    plt.show()

brownian_motion()
```

---

## 5. 물리 시스템

### 5.1 Ising 모델

```python
def ising_model():
    """
    2D Ising 모델: Metropolis 알고리즘

    H = -J Σ sᵢsⱼ

    스핀 sᵢ = ±1, J > 0 (강자성)
    """

    def calculate_energy(lattice, J=1):
        """전체 에너지 계산"""
        energy = 0
        N = lattice.shape[0]
        for i in range(N):
            for j in range(N):
                S = lattice[i, j]
                neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                            lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
                energy -= J * S * neighbors
        return energy / 2  # 중복 카운트 보정

    def metropolis_step(lattice, beta, J=1):
        """Metropolis 알고리즘 한 스텝"""
        N = lattice.shape[0]
        i, j = np.random.randint(0, N, 2)

        S = lattice[i, j]
        neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                    lattice[i, (j+1)%N] + lattice[i, (j-1)%N])

        dE = 2 * J * S * neighbors

        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            lattice[i, j] = -S

        return lattice

    def simulate_ising(N, T, n_steps, n_equilibrate):
        """Ising 모델 시뮬레이션"""
        beta = 1 / T
        lattice = np.random.choice([-1, 1], (N, N))

        magnetizations = []

        for step in range(n_steps + n_equilibrate):
            for _ in range(N * N):  # N² 번 시도 = 1 MC 스텝
                lattice = metropolis_step(lattice, beta)

            if step >= n_equilibrate:
                M = np.abs(np.mean(lattice))
                magnetizations.append(M)

        return lattice, np.mean(magnetizations)

    # 온도에 따른 상전이
    N = 20
    temperatures = np.linspace(1.0, 4.0, 20)
    magnetizations = []

    print("2D Ising 모델 시뮬레이션 중...")
    for T in temperatures:
        _, M = simulate_ising(N, T, n_steps=100, n_equilibrate=50)
        magnetizations.append(M)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 온도별 자화
    Tc = 2.269  # 임계 온도 (2D Ising)
    axes[0].plot(temperatures, magnetizations, 'bo-')
    axes[0].axvline(x=Tc, color='r', linestyle='--', label=f'Tc = {Tc:.3f}')
    axes[0].set_xlabel('온도 T')
    axes[0].set_ylabel('자화 |M|')
    axes[0].set_title('온도에 따른 자화')
    axes[0].legend()
    axes[0].grid(True)

    # 저온 상태 (정렬)
    lattice_low, _ = simulate_ising(30, T=1.5, n_steps=200, n_equilibrate=100)
    axes[1].imshow(lattice_low, cmap='coolwarm', interpolation='nearest')
    axes[1].set_title(f'T = 1.5 (저온, 정렬)')
    axes[1].axis('off')

    # 고온 상태 (무질서)
    lattice_high, _ = simulate_ising(30, T=4.0, n_steps=200, n_equilibrate=100)
    axes[2].imshow(lattice_high, cmap='coolwarm', interpolation='nearest')
    axes[2].set_title(f'T = 4.0 (고온, 무질서)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

ising_model()
```

### 5.2 분자 동역학 (간단한 예시)

```python
def lennard_jones_mc():
    """
    Lennard-Jones 기체의 몬테카를로 시뮬레이션

    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
    """

    def lj_potential(r, epsilon=1, sigma=1):
        """Lennard-Jones 포텐셜"""
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    def total_energy(positions, L, epsilon=1, sigma=1):
        """전체 에너지 (주기적 경계조건)"""
        N = len(positions)
        energy = 0
        for i in range(N):
            for j in range(i+1, N):
                dr = positions[j] - positions[i]
                # 최소 이미지 규약
                dr = dr - L * np.round(dr / L)
                r = np.linalg.norm(dr)
                if r < 3 * sigma:  # 컷오프
                    energy += lj_potential(r, epsilon, sigma)
        return energy

    def mc_step(positions, L, T, delta=0.1):
        """MC 이동 시도"""
        N = len(positions)
        beta = 1 / T

        old_E = total_energy(positions, L)

        # 랜덤 입자 선택 및 이동
        i = np.random.randint(N)
        old_pos = positions[i].copy()
        positions[i] += np.random.uniform(-delta, delta, 2)

        # 주기적 경계조건
        positions[i] = positions[i] % L

        new_E = total_energy(positions, L)
        dE = new_E - old_E

        if dE > 0 and np.random.random() > np.exp(-beta * dE):
            positions[i] = old_pos  # 거절
            return False
        return True

    # 시뮬레이션
    N = 20
    L = 5.0  # 박스 크기
    T = 1.0

    # 초기 배치 (랜덤)
    positions = np.random.uniform(0, L, (N, 2))

    # 평형화
    for _ in range(5000):
        mc_step(positions, L, T)

    # 샘플링
    n_samples = 100
    snapshots = []
    energies = []

    for i in range(n_samples):
        for _ in range(100):  # 100 스텝마다 샘플링
            mc_step(positions, L, T)
        snapshots.append(positions.copy())
        energies.append(total_energy(positions, L))

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(positions[:, 0], positions[:, 1], s=100)
    axes[0].set_xlim(0, L)
    axes[0].set_ylim(0, L)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'LJ 기체 (N={N}, T={T})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    axes[1].plot(energies)
    axes[1].set_xlabel('샘플')
    axes[1].set_ylabel('에너지')
    axes[1].set_title('에너지 시계열')
    axes[1].grid(True)

    print(f"평균 에너지: {np.mean(energies):.4f}")

    plt.tight_layout()
    plt.show()

lennard_jones_mc()
```

---

## 6. 금융 및 공학 응용

### 6.1 옵션 가격 결정

```python
def option_pricing():
    """
    블랙-숄즈 몬테카를로 시뮬레이션

    유럽식 콜 옵션: C = E[max(S(T) - K, 0)] * e^(-rT)
    """

    def black_scholes_call(S0, K, T, r, sigma):
        """블랙-숄즈 공식 (해석적)"""
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

    def monte_carlo_call(S0, K, T, r, sigma, n_paths=100000):
        """몬테카를로 시뮬레이션"""
        # 최종 가격 시뮬레이션
        Z = np.random.normal(0, 1, n_paths)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

        # 페이오프
        payoffs = np.maximum(ST - K, 0)

        # 할인된 기대값
        price = np.exp(-r*T) * np.mean(payoffs)
        std_error = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths)

        return price, std_error

    # 파라미터
    S0 = 100      # 현재 주가
    K = 100       # 행사가
    T = 1.0       # 만기 (년)
    r = 0.05      # 무위험 이자율
    sigma = 0.2   # 변동성

    # 가격 비교
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    mc_price, mc_error = monte_carlo_call(S0, K, T, r, sigma)

    print("유럽식 콜 옵션 가격:")
    print(f"  블랙-숄즈 공식: {bs_price:.4f}")
    print(f"  몬테카를로: {mc_price:.4f} ± {mc_error:.4f}")

    # 수렴 분석
    n_values = [1000, 5000, 10000, 50000, 100000, 500000]
    mc_prices = []
    mc_errors = []

    for n in n_values:
        price, error = monte_carlo_call(S0, K, T, r, sigma, n)
        mc_prices.append(price)
        mc_errors.append(error)

    plt.figure(figsize=(10, 5))
    plt.errorbar(n_values, mc_prices, yerr=mc_errors, fmt='bo-', capsize=3)
    plt.axhline(y=bs_price, color='r', linestyle='--', label='블랙-숄즈')
    plt.xscale('log')
    plt.xlabel('시뮬레이션 수')
    plt.ylabel('옵션 가격')
    plt.title('몬테카를로 옵션 가격 수렴')
    plt.legend()
    plt.grid(True)
    plt.show()

option_pricing()
```

### 6.2 신뢰성 분석

```python
def reliability_analysis():
    """
    시스템 신뢰성 몬테카를로 분석

    직렬/병렬 시스템의 고장 확률
    """

    def component_lifetime(mean_life, n_simulations):
        """지수 분포 수명"""
        return np.random.exponential(mean_life, n_simulations)

    def serial_system(mean_lives, n_simulations):
        """
        직렬 시스템: 하나라도 고장나면 시스템 고장
        시스템 수명 = min(각 부품 수명)
        """
        lifetimes = np.array([component_lifetime(m, n_simulations)
                             for m in mean_lives])
        return np.min(lifetimes, axis=0)

    def parallel_system(mean_lives, n_simulations):
        """
        병렬 시스템: 모두 고장나야 시스템 고장
        시스템 수명 = max(각 부품 수명)
        """
        lifetimes = np.array([component_lifetime(m, n_simulations)
                             for m in mean_lives])
        return np.max(lifetimes, axis=0)

    # 시뮬레이션
    n_sim = 100000
    mean_lives = [100, 150, 200]  # 각 부품의 평균 수명

    serial_life = serial_system(mean_lives, n_sim)
    parallel_life = parallel_system(mean_lives, n_sim)

    # 결과 분석
    t = 100  # 목표 시간

    serial_reliability = np.mean(serial_life > t)
    parallel_reliability = np.mean(parallel_life > t)

    print(f"t = {t}에서의 신뢰도:")
    print(f"  직렬 시스템: {serial_reliability:.4f}")
    print(f"  병렬 시스템: {parallel_reliability:.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 수명 분포
    axes[0].hist(serial_life, bins=50, density=True, alpha=0.7, label='직렬')
    axes[0].hist(parallel_life, bins=50, density=True, alpha=0.7, label='병렬')
    axes[0].axvline(x=t, color='r', linestyle='--', label=f't={t}')
    axes[0].set_xlabel('수명')
    axes[0].set_ylabel('밀도')
    axes[0].set_title('시스템 수명 분포')
    axes[0].legend()
    axes[0].grid(True)

    # 신뢰도 함수
    t_range = np.linspace(0, 500, 100)
    serial_R = [np.mean(serial_life > t) for t in t_range]
    parallel_R = [np.mean(parallel_life > t) for t in t_range]

    axes[1].plot(t_range, serial_R, 'b-', label='직렬')
    axes[1].plot(t_range, parallel_R, 'r-', label='병렬')
    axes[1].set_xlabel('시간 t')
    axes[1].set_ylabel('R(t)')
    axes[1].set_title('신뢰도 함수')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

reliability_analysis()
```

---

## 7. 분산 감소 기법

### 7.1 대조 변량 (Antithetic Variates)

```python
def antithetic_variates():
    """
    대조 변량: U와 1-U를 함께 사용하여 분산 감소

    U ~ Uniform(0,1)이면 1-U도 같은 분포
    f(U)와 f(1-U)가 음의 상관이면 분산 감소
    """

    # 예시: E[e^U] where U ~ Uniform(0,1)
    # 정확한 값: e - 1 ≈ 1.71828

    n = 10000
    true_value = np.e - 1

    # 표준 몬테카를로
    U = np.random.uniform(0, 1, n)
    standard_estimate = np.mean(np.exp(U))
    standard_var = np.var(np.exp(U)) / n

    # 대조 변량
    U = np.random.uniform(0, 1, n // 2)
    f_U = np.exp(U)
    f_1mU = np.exp(1 - U)
    av_estimate = np.mean((f_U + f_1mU) / 2)
    av_var = np.var((f_U + f_1mU) / 2) / (n // 2)

    print("E[e^U] 추정 (참값 = 1.71828):")
    print(f"\n표준 MC:")
    print(f"  추정값: {standard_estimate:.6f}")
    print(f"  분산: {standard_var:.2e}")

    print(f"\n대조 변량:")
    print(f"  추정값: {av_estimate:.6f}")
    print(f"  분산: {av_var:.2e}")

    print(f"\n분산 감소율: {standard_var / av_var:.2f}배")

antithetic_variates()
```

### 7.2 층화 샘플링

```python
def stratified_sampling():
    """
    층화 샘플링: 영역을 나누어 각 층에서 균등하게 샘플링

    분산 감소: 층 내 분산만 남음
    """

    # 예시: ∫₀¹ x² dx = 1/3

    n_total = 10000

    # 표준 MC
    X = np.random.uniform(0, 1, n_total)
    standard_estimate = np.mean(X**2)
    standard_var = np.var(X**2) / n_total

    # 층화 샘플링 (10개 층)
    n_strata = 10
    n_per_stratum = n_total // n_strata

    stratified_estimates = []
    for i in range(n_strata):
        low = i / n_strata
        high = (i + 1) / n_strata
        X_stratum = np.random.uniform(low, high, n_per_stratum)
        stratum_mean = np.mean(X_stratum**2)
        stratified_estimates.append(stratum_mean)

    stratified_estimate = np.mean(stratified_estimates)

    # 층화 샘플링 분산 (층 내 분산만)
    within_vars = []
    for i in range(n_strata):
        low = i / n_strata
        high = (i + 1) / n_strata
        X_stratum = np.random.uniform(low, high, 1000)
        within_vars.append(np.var(X_stratum**2))

    stratified_var = np.mean(within_vars) / n_total

    print("∫₀¹ x² dx = 1/3 ≈ 0.3333")
    print(f"\n표준 MC:")
    print(f"  추정값: {standard_estimate:.6f}")
    print(f"  분산: {standard_var:.2e}")

    print(f"\n층화 샘플링 (10개 층):")
    print(f"  추정값: {stratified_estimate:.6f}")
    print(f"  분산: {stratified_var:.2e}")

    print(f"\n분산 감소율: {standard_var / stratified_var:.2f}배")

stratified_sampling()
```

### 7.3 제어 변량 (Control Variates)

```python
def control_variates():
    """
    제어 변량: 기대값을 아는 변수를 이용하여 분산 감소

    θ̂_cv = θ̂ - c(Ŷ - E[Y])

    여기서 Y는 기대값 E[Y]를 아는 제어 변량
    c는 분산을 최소화하는 계수
    """

    # 예시: E[e^U]를 U를 제어 변량으로 사용
    # E[U] = 0.5 (알려진 값)

    n = 10000
    true_value = np.e - 1

    U = np.random.uniform(0, 1, n)
    f = np.exp(U)

    # 표준 MC
    standard_estimate = np.mean(f)
    standard_var = np.var(f) / n

    # 제어 변량
    Y = U
    EY = 0.5  # E[U]

    # 최적 c 추정
    cov_fY = np.cov(f, Y)[0, 1]
    var_Y = np.var(Y)
    c_opt = cov_fY / var_Y

    # 제어 변량 추정량
    cv_estimate = np.mean(f - c_opt * (Y - EY))
    cv_var = np.var(f - c_opt * (Y - EY)) / n

    print("E[e^U] 추정 (참값 = 1.71828):")
    print(f"\n표준 MC:")
    print(f"  추정값: {standard_estimate:.6f}")
    print(f"  분산: {standard_var:.2e}")

    print(f"\n제어 변량 (c = {c_opt:.4f}):")
    print(f"  추정값: {cv_estimate:.6f}")
    print(f"  분산: {cv_var:.2e}")

    print(f"\n분산 감소율: {standard_var / cv_var:.2f}배")

    # 상관계수로 분산 감소 예측
    corr = np.corrcoef(f, Y)[0, 1]
    theoretical_reduction = 1 / (1 - corr**2)
    print(f"이론적 분산 감소율 (ρ² = {corr**2:.4f}): {theoretical_reduction:.2f}배")

control_variates()
```

---

## 연습 문제

### 문제 1: 구의 부피
몬테카를로로 d차원 단위 구의 부피를 추정하세요. (d=3일 때 4π/3 ≈ 4.19)

```python
def exercise_1():
    def sphere_volume_mc(d, n_samples):
        points = np.random.uniform(-1, 1, (n_samples, d))
        inside = np.sum(points**2, axis=1) <= 1
        cube_volume = 2**d
        return cube_volume * np.mean(inside)

    for d in [2, 3, 4, 5]:
        volume = sphere_volume_mc(d, 100000)
        theoretical = np.pi**(d/2) / np.math.gamma(d/2 + 1)
        print(f"d={d}: MC={volume:.4f}, 이론={theoretical:.4f}")

exercise_1()
```

### 문제 2: 아시안 옵션
경로 의존 옵션(아시안 콜 옵션)의 가격을 시뮬레이션하세요.

```python
def exercise_2():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_steps, n_paths = 252, 100000

    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.normal(0, 1, n_paths)
        S[:, i+1] = S[:, i] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    # 산술 평균
    S_avg = np.mean(S[:, 1:], axis=1)
    payoffs = np.maximum(S_avg - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)

    print(f"아시안 콜 옵션 가격: {price:.4f}")

exercise_2()
```

---

## 요약

| 기법 | 설명 | 용도 |
|------|------|------|
| 기본 MC | 균등 샘플링으로 적분 | 일반 적분 |
| 중요도 샘플링 | 제안 분포 사용 | 희귀 사건, 분산 감소 |
| 대조 변량 | U와 1-U 사용 | 단조 함수 |
| 층화 샘플링 | 영역 분할 | 균등 커버리지 |
| 제어 변량 | 상관된 변수 활용 | 기대값 아는 변수 존재 |

| 응용 | 예시 |
|------|------|
| 물리학 | Ising 모델, 분자 동역학 |
| 금융 | 옵션 가격, VaR |
| 공학 | 신뢰성 분석, 불확실성 정량화 |

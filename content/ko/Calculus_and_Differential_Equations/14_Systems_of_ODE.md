# 14. 상미분방정식의 연립(Systems of Ordinary Differential Equations)

## 학습 목표

- 고계 ODE를 동치인 1계 연립으로 변환한다
- 고유값 방법(eigenvalue method)을 사용하여 제차 선형 시스템 $\mathbf{X}' = A\mathbf{X}$를 푼다
- 위상 평면(phase plane)에서 평형점을 분류한다 (마디, 안장, 나선, 중심)
- 평형점 근방에서 선형 및 비선형 시스템의 안정성을 분석한다
- Python을 사용하여 위상 초상화(phase portrait)를 구성하고 고유값 분석을 수행한다

---

## 1. 고계에서 1계 연립으로(From Higher-Order to First-Order Systems)

### 1.1 변환 기법

차수 $n$인 임의의 단일 ODE는 $n$개의 1계 ODE 연립으로 다시 쓸 수 있다. 이것은 단지 이론적 편의만이 아니다 -- **모든 수치 풀이기**가 작동하는 방식이다 (`solve_ivp` 포함).

**예제:** $y'' + 3y' + 2y = 0$을 1계 연립으로 변환하라.

새로운 변수를 정의한다:

$$x_1 = y, \quad x_2 = y'$$

그러면:

$$x_1' = x_2$$
$$x_2' = y'' = -3y' - 2y = -2x_1 - 3x_2$$

행렬 형태로:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

**일반 패턴:** $y^{(n)} + a_{n-1}y^{(n-1)} + \cdots + a_1 y' + a_0 y = 0$에 대해, $k = 1, \ldots, n$에 대해 $x_k = y^{(k-1)}$로 놓는다. 마지막 방정식은 $y^{(n)}$을 나머지 변수로 표현한다.

### 1.2 연성 시스템(Coupled Systems)

많은 실세계 문제는 자연스럽게 **연성** 방정식을 포함한다 -- 여러 양이 서로 영향을 미친다.

**예제: 포식자-피식자(로트카-볼테라, Lotka-Volterra)**

$$\frac{dx}{dt} = \alpha x - \beta xy \quad \text{(피식자)}$$
$$\frac{dy}{dt} = -\gamma y + \delta xy \quad \text{(포식자)}$$

- $x$: 피식자 개체수, $y$: 포식자 개체수
- $\alpha$: 피식자 출생률, $\beta$: 포식률
- $\gamma$: 포식자 사망률, $\delta$: 피식자 소비로 인한 포식자 번식

두 종은 연성되어 있다: 피식자 성장은 포식자 수에 의존하고, 그 역도 마찬가지이다. 어느 방정식도 독립적으로 풀 수 없다.

---

## 2. 선형 시스템: $\mathbf{X}' = A\mathbf{X}$

### 2.1 행렬 공식화

상수계수 **제차 선형 시스템**:

$$\mathbf{X}'(t) = A\mathbf{X}(t)$$

여기서 $\mathbf{X}(t) = \begin{pmatrix} x_1(t) \\ x_2(t) \end{pmatrix}$이고 $A$는 상수 행렬이다.

스칼라 방정식 $x' = ax$에서 해가 $x = Ce^{at}$인 것과 유사하게, $\mathbf{X} = \mathbf{v} e^{\lambda t}$를 추측한다 (여기서 $\mathbf{v}$는 상수 벡터).

### 2.2 고유값 방법(The Eigenvalue Method)

$\mathbf{X} = \mathbf{v} e^{\lambda t}$를 $\mathbf{X}' = A\mathbf{X}$에 대입하면:

$$\lambda \mathbf{v} e^{\lambda t} = A\mathbf{v} e^{\lambda t} \implies A\mathbf{v} = \lambda\mathbf{v}$$

이것은 **고유값 문제(eigenvalue problem)**이다: $A$의 고유값 $\lambda$와 고유벡터 $\mathbf{v}$를 찾는다.

**단계:**
1. 고유값 $\lambda_1, \lambda_2$에 대해 $\det(A - \lambda I) = 0$을 푼다
2. 각 $\lambda_i$에 대해, $(A - \lambda_i I)\mathbf{v}_i = \mathbf{0}$에서 고유벡터 $\mathbf{v}_i$를 찾는다
3. 일반해: $\mathbf{X}(t) = C_1 \mathbf{v}_1 e^{\lambda_1 t} + C_2 \mathbf{v}_2 e^{\lambda_2 t}$

**예제:** $A = \begin{pmatrix} 1 & 3 \\ 1 & -1 \end{pmatrix}$

특성방정식: $(1 - \lambda)(-1 - \lambda) - 3 = \lambda^2 - 4 = 0$

고유값: $\lambda_1 = 2$, $\lambda_2 = -2$

$\lambda_1 = 2$에 대해: $(A - 2I)\mathbf{v} = 0 \implies \mathbf{v}_1 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$

$\lambda_2 = -2$에 대해: $(A + 2I)\mathbf{v} = 0 \implies \mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

일반해:

$$\mathbf{X}(t) = C_1 \begin{pmatrix} 3 \\ 1 \end{pmatrix} e^{2t} + C_2 \begin{pmatrix} 1 \\ -1 \end{pmatrix} e^{-2t}$$

### 2.3 복소 고유값(Complex Eigenvalues)

$A$가 실수이고 고유값이 $\lambda = \alpha \pm i\beta$일 때, 해는 진동을 포함한다:

$$\mathbf{X}(t) = e^{\alpha t}\left[C_1(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + C_2(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)\right]$$

여기서 $\mathbf{v} = \mathbf{a} + i\mathbf{b}$는 $\lambda = \alpha + i\beta$에 대한 (복소) 고유벡터이다.

### 2.4 중복 고유값(Repeated Eigenvalues)

$\lambda$가 하나의 독립적인 고유벡터 $\mathbf{v}$만 가진 중복 고유값이면, $(A - \lambda I)\mathbf{w} = \mathbf{v}$를 만족하는 **일반화된 고유벡터(generalized eigenvector)** $\mathbf{w}$가 필요하다:

$$\mathbf{X}(t) = C_1 \mathbf{v} e^{\lambda t} + C_2 (\mathbf{v}t + \mathbf{w})e^{\lambda t}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Eigenvalue method: analytical vs numerical ---
A = np.array([[1, 3], [1, -1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Analytical solution with IC x1(0)=1, x2(0)=0
# X(0) = C1*v1 + C2*v2 = (1, 0)
# Solve: C1*(3,1) + C2*(1,-1) = (1,0)
# 3*C1 + C2 = 1, C1 - C2 = 0 => C1 = C2 = 1/4
C1, C2 = 0.25, 0.25
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
lam1, lam2 = eigenvalues

t = np.linspace(0, 2, 200)
x1_anal = C1 * v1[0] * np.exp(lam1 * t) + C2 * v2[0] * np.exp(lam2 * t)
x2_anal = C1 * v1[1] * np.exp(lam1 * t) + C2 * v2[1] * np.exp(lam2 * t)

# Numerical solution
def system(t, X):
    return A @ X

sol = solve_ivp(system, (0, 2), [1, 0], t_eval=t)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, x1_anal, 'b-', linewidth=2, label='$x_1$ (analytical)')
ax.plot(t, x2_anal, 'r-', linewidth=2, label='$x_2$ (analytical)')
ax.plot(sol.t, sol.y[0], 'b--', linewidth=2, label='$x_1$ (numerical)')
ax.plot(sol.t, sol.y[1], 'r--', linewidth=2, label='$x_2$ (numerical)')
ax.set_xlabel('Time t')
ax.set_ylabel('$x_i(t)$')
ax.set_title('Solution of $\\mathbf{X}\' = A\\mathbf{X}$ (Saddle Point)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 3. 위상 평면 분석(Phase Plane Analysis)

### 3.1 위상 초상화(The Phase Portrait)

2차원 시스템 $\mathbf{X}' = A\mathbf{X}$에서, **위상 평면** -- $(x_1, x_2)$ 평면에서 궤적을 그릴 수 있다. 원점 근방의 모든 궤적의 거동은 전적으로 고유값에 의해 결정된다.

### 3.2 평형점의 분류(Classification of Equilibrium Points)

| 고유값 | 유형 | 안정성 | 위상 초상화 |
|-------------|------|-----------|----------------|
| $\lambda_1, \lambda_2 > 0$ | 불안정 마디(node) | 불안정 | 궤적이 원점에서 멀어진다 |
| $\lambda_1, \lambda_2 < 0$ | 안정 마디 | 점근 안정 | 모든 궤적이 원점에 접근한다 |
| $\lambda_1 > 0 > \lambda_2$ | 안장점(saddle) | 불안정 | 궤적이 한 고유벡터를 따라 접근하고 다른 쪽을 따라 발산한다 |
| $\alpha \pm i\beta$, $\alpha < 0$ | 안정 나선(spiral) | 점근 안정 | 안쪽으로 나선을 그린다 |
| $\alpha \pm i\beta$, $\alpha > 0$ | 불안정 나선 | 불안정 | 바깥쪽으로 나선을 그린다 |
| $\pm i\beta$ (순허수) | 중심(center) | 안정 (점근이 아님) | 닫힌 타원형 궤도 |
| $\lambda_1 = \lambda_2 < 0$ (고유벡터 2개) | 안정 별점(star node) | 점근 안정 | 직선 궤적이 안쪽으로 |
| $\lambda_1 = \lambda_2 < 0$ (고유벡터 1개) | 안정 퇴화 마디 | 점근 안정 | 궤적이 고유벡터에 접선 |

**핵심 통찰:** 고유값의 **실수 부분**이 안정성을 결정하고(음수 = 안정), **허수 부분**이 진동을 결정한다(0이 아니면 = 나선/진동).

### 3.3 안정성 요약

- **점근적으로 안정(Asymptotically stable):** 모든 고유값이 **음의 실수 부분**을 가진다 -- 궤적이 원점으로 수렴한다
- **안정 (중립)(Stable):** 고유값이 순허수이다 -- 궤적이 성장하지도 감소하지도 않는다
- **불안정(Unstable):** 적어도 하나의 고유값이 **양의 실수 부분**을 가진다 -- 일부 궤적이 발산한다

**대각합-행렬식 평면과의 관계:** $2 \times 2$ 행렬에서, $\tau = \text{tr}(A)$, $\Delta = \det(A)$로 놓으면:

- $\Delta < 0$: 안장점
- $\Delta > 0$, $\tau^2 - 4\Delta > 0$, $\tau < 0$: 안정 마디
- $\Delta > 0$, $\tau^2 - 4\Delta < 0$, $\tau < 0$: 안정 나선
- $\Delta > 0$, $\tau = 0$: 중심

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_phase_portrait(A, title, ax, xlim=(-3, 3), ylim=(-3, 3)):
    """Plot the phase portrait for X' = AX."""
    # Direction field
    x = np.linspace(xlim[0], xlim[1], 20)
    y = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(x, y)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    ax.quiver(X, Y, U/magnitude, V/magnitude, magnitude,
              cmap='coolwarm', alpha=0.4, scale=25)

    # Trajectories from various initial conditions
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for r in [0.5, 1.5, 2.5]:
        for theta in angles:
            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)
            def system(t, state): return A @ state
            sol = solve_ivp(system, (0, 5), [x0, y0],
                           t_eval=np.linspace(0, 5, 300),
                           max_step=0.05)
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.7)

    # Plot eigenvectors for real eigenvalues
    evals, evecs = np.linalg.eig(A)
    if np.all(np.isreal(evals)):
        for i in range(2):
            v = np.real(evecs[:, i])
            v = v / np.linalg.norm(v) * 3
            ax.plot([-v[0], v[0]], [-v[1], v[1]], 'r-', linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Six representative cases
matrices = [
    (np.array([[-1, 0], [0, -2]]),   'Stable Node\n$\\lambda=-1,-2$'),
    (np.array([[1, 0], [0, 2]]),      'Unstable Node\n$\\lambda=1,2$'),
    (np.array([[2, 0], [0, -1]]),     'Saddle Point\n$\\lambda=2,-1$'),
    (np.array([[-0.5, 2], [-2, -0.5]]), 'Stable Spiral\n$\\lambda=-0.5\\pm 2i$'),
    (np.array([[0.3, 2], [-2, 0.3]]),   'Unstable Spiral\n$\\lambda=0.3\\pm 2i$'),
    (np.array([[0, 1], [-1, 0]]),       'Center\n$\\lambda=\\pm i$'),
]

for idx, (A_mat, title) in enumerate(matrices):
    row, col = divmod(idx, 3)
    evals = np.linalg.eigvals(A_mat)
    plot_phase_portrait(A_mat, title, axes[row, col])

plt.tight_layout()
plt.show()
```

---

## 4. 비선형 시스템과 선형화(Nonlinear Systems and Linearization)

### 4.1 비선형 시스템

대부분의 실제 시스템은 비선형이다:

$$\frac{dx}{dt} = f(x, y), \quad \frac{dy}{dt} = g(x, y)$$

정확한 해석적 해는 보통 불가능하지만, 평형점 근방의 **정성적 거동(qualitative behavior)**은 여전히 이해할 수 있다.

### 4.2 평형점(Equilibrium Points)

**평형점(equilibrium point)**(또는 고정점) $(x^*, y^*)$는 다음을 만족한다:

$$f(x^*, y^*) = 0 \quad \text{and} \quad g(x^*, y^*) = 0$$

평형에서 시스템은 정지해 있다 -- 모든 도함수가 0이다.

### 4.3 선형화(Linearization)

평형 $(x^*, y^*)$ 근방에서 테일러 급수로 전개하고 선형 항만 남긴다:

$$\begin{pmatrix} u' \\ v' \end{pmatrix} \approx J \begin{pmatrix} u \\ v \end{pmatrix}$$

여기서 $u = x - x^*$, $v = y - y^*$이고, $J$는 **야코비 행렬(Jacobian matrix)**이다:

$$J = \begin{pmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \\ \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} \end{pmatrix}_{(x^*, y^*)}$$

**하트만-그로브만 정리(Hartman-Grobman theorem)**는 $J$의 고유값이 0이 아닌 실수 부분을 가지면(평형이 **쌍곡적(hyperbolic)**이면), 평형 근방의 비선형 시스템 거동이 선형화된 시스템과 **정성적으로 동일**함을 보장한다.

**주의:** 중심 ($\lambda = \pm i\beta$)의 경우, 선형화는 결정적이지 않다 -- 비선형 항이 중심을 나선으로 바꿀 수 있다.

### 4.4 예제: 로트카-볼테라 포식자-피식자

$$\dot{x} = x(1 - y), \quad \dot{y} = y(x - 1)$$

**평형:** $(0, 0)$과 $(1, 1)$

**$(0, 0)$에서:** $J = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, 고유값 $\lambda = 1, -1$ -- **안장점** (불안정)

**$(1, 1)$에서:** $J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, 고유값 $\lambda = \pm i$ -- **중심** (선형화는 닫힌 궤도를 시사한다). 실제로, 완전한 비선형 시스템은 닫힌 궤도를 가진다 (보존량이 존재한다).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Lotka-Volterra Predator-Prey System ---
def lotka_volterra(t, state):
    """dx/dt = x(1-y), dy/dt = y(x-1)."""
    x, y = state
    return [x * (1 - y), y * (x - 1)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phase portrait
for x0 in np.arange(0.5, 3.5, 0.5):
    for y0 in np.arange(0.5, 3.5, 0.5):
        sol = solve_ivp(lotka_volterra, (0, 20), [x0, y0],
                       t_eval=np.linspace(0, 20, 2000),
                       max_step=0.01)
        axes[0].plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.5)

# Mark equilibrium point
axes[0].plot(1, 1, 'ro', markersize=10, label='Equilibrium (1,1)')
axes[0].set_xlabel('Prey x')
axes[0].set_ylabel('Predator y')
axes[0].set_title('Lotka-Volterra Phase Portrait')
axes[0].set_xlim(0, 4)
axes[0].set_ylim(0, 4)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Time series from one initial condition
sol = solve_ivp(lotka_volterra, (0, 30), [2.0, 1.0],
               t_eval=np.linspace(0, 30, 2000), max_step=0.01)

axes[1].plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Prey x(t)')
axes[1].plot(sol.t, sol.y[1], 'r-', linewidth=2, label='Predator y(t)')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Population')
axes[1].set_title('Lotka-Volterra Time Series')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# --- Linearization analysis ---
from sympy import symbols, Matrix

x, y = symbols('x y', positive=True)
f = x * (1 - y)
g = y * (x - 1)

J = Matrix([[f.diff(x), f.diff(y)],
            [g.diff(x), g.diff(y)]])

print("Jacobian:")
print(J)

# At equilibrium (1, 1)
J_eq = J.subs([(x, 1), (y, 1)])
print(f"\nJacobian at (1,1):\n{J_eq}")
print(f"Eigenvalues: {J_eq.eigenvals()}")
```

---

## 5. 감쇠 진자: 완전한 예제(The Damped Pendulum: A Complete Example)

비선형 진자 방정식 $\ddot{\theta} + \beta\dot{\theta} + \omega_0^2\sin\theta = 0$은 이 레슨의 모든 것을 결합한 풍부한 예제이다.

시스템으로 변환: $x_1 = \theta$, $x_2 = \dot{\theta}$:

$$x_1' = x_2$$
$$x_2' = -\omega_0^2\sin x_1 - \beta x_2$$

**평형:** 정수 $n$에 대해 $(n\pi, 0)$.
- $\theta = 0$ (아래로 매달린 상태): $J = \begin{pmatrix} 0 & 1 \\ -\omega_0^2 & -\beta \end{pmatrix}$ -- **안정** (감쇠에 따라 나선 또는 마디)
- $\theta = \pi$ (거꾸로 균형 잡힌 상태): $J = \begin{pmatrix} 0 & 1 \\ \omega_0^2 & -\beta \end{pmatrix}$ -- **불안정** (안장점)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Damped pendulum phase portrait ---
omega0 = 2.0
beta = 0.3

def pendulum(t, state):
    """theta' = omega, omega' = -omega0^2 * sin(theta) - beta * omega."""
    theta, omega = state
    return [omega, -omega0**2 * np.sin(theta) - beta * omega]

fig, ax = plt.subplots(figsize=(12, 8))

# Trajectories from many initial conditions
for theta0 in np.linspace(-3*np.pi, 3*np.pi, 20):
    for omega0_ic in np.linspace(-8, 8, 8):
        sol = solve_ivp(pendulum, (0, 20), [theta0, omega0_ic],
                       t_eval=np.linspace(0, 20, 1000),
                       max_step=0.05)
        ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.3, alpha=0.4)

# Mark equilibria
for n in range(-3, 4):
    if n % 2 == 0:
        ax.plot(n*np.pi, 0, 'go', markersize=8)  # stable
    else:
        ax.plot(n*np.pi, 0, 'rx', markersize=10, markeredgewidth=2)  # unstable

ax.set_xlabel('$\\theta$ (radians)')
ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
ax.set_title(f'Damped Pendulum Phase Portrait ($\\omega_0={omega0}$, $\\beta={beta}$)\n'
             f'Green circles = stable equilibria, Red crosses = saddle points')
ax.set_xlim(-3*np.pi, 3*np.pi)
ax.set_ylim(-10, 10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. 행렬 지수함수 (심화)(The Matrix Exponential (Advanced))

완전성을 위해, $\mathbf{X}' = A\mathbf{X}$의 일반해는 다음과 같이 간결하게 쓸 수 있다:

$$\mathbf{X}(t) = e^{At}\mathbf{X}(0)$$

여기서 **행렬 지수함수(matrix exponential)**는 멱급수로 정의된다:

$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots$$

대각화 가능한 $A = PDP^{-1}$ (여기서 $D$는 대각행렬)에 대해:

$$e^{At} = P\,e^{Dt}\,P^{-1} = P\begin{pmatrix} e^{\lambda_1 t} & 0 \\ 0 & e^{\lambda_2 t} \end{pmatrix}P^{-1}$$

이것은 고유값 방법을 행렬 지수함수 프레임워크와 연결한다.

```python
import numpy as np
from scipy.linalg import expm

# --- Matrix exponential solution ---
A = np.array([[1, 3], [1, -1]])
X0 = np.array([1, 0])

t_vals = np.linspace(0, 1, 5)
for t in t_vals:
    X_t = expm(A * t) @ X0
    print(f"t = {t:.2f}: X = [{X_t[0]:.4f}, {X_t[1]:.4f}]")

# Verify: eigendecomposition approach
evals, P = np.linalg.eig(A)
P_inv = np.linalg.inv(P)

t = 1.0
D_exp = np.diag(np.exp(evals * t))
X_eigen = P @ D_exp @ P_inv @ X0
X_expm = expm(A * t) @ X0

print(f"\nAt t=1.0:")
print(f"  Matrix exponential: {X_expm}")
print(f"  Eigendecomposition: {X_eigen}")
print(f"  Match: {np.allclose(X_expm, X_eigen)}")
```

---

## 7. 교차 참조

- **Mathematical Methods 레슨 10**은 고계 시스템, 경계값 문제, 스튀름-리우빌 이론(Sturm-Liouville theory)을 다룬다.
- **레슨 13 (2계 ODE)**에서 소개한 스프링-질량 시스템의 위상 평면 분석은 이 레슨의 특수한 경우이다.
- **제어이론(Control Theory) 레슨 06-09**는 상태공간 분석 ($\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$)을 사용하며, 이는 여기서 제시된 선형 시스템 이론의 직접적인 확장으로 피드백 제어를 포함한다.
- **딥러닝(Deep Learning) 레슨 34** (옵티마이저)는 기울기 흐름 시스템 $\dot{\mathbf{x}} = -\nabla f(\mathbf{x})$와 관련되며, 손실 경관이 퍼텐셜의 역할을 한다.

---

## 연습 문제

**1.** 3계 ODE $y''' - 6y'' + 11y' - 6y = 0$을 1계 연립으로 변환하라. 고유값과 일반해를 구하라.

**2.** $\mathbf{X}(0) = (1, 1)^T$로 시스템 $\mathbf{X}' = \begin{pmatrix} 3 & -2 \\ 4 & -1 \end{pmatrix}\mathbf{X}$를 풀어라. 평형점을 분류하고 위상 초상화를 스케치하라.

**3.** 경쟁 종 모델에 대해:
   $$\dot{x} = x(3 - x - 2y), \quad \dot{y} = y(2 - y - x)$$
   - (a) 모든 평형점을 구하라.
   - (b) 각 평형점에서 선형화하고 안정성을 분류하라.
   - (c) 수치적으로 위상 초상화를 그리고 해석하라: 어느 종이 생존하는가?

**4.** $\omega_0 = 3$, $\beta = 0.5$인 감쇠 진자가 $\theta(0) = \pi - 0.1$ (거의 거꾸로), $\dot{\theta}(0) = 0$에서 시작한다.
   - (a) 진자가 어떤 평형 근처로 가게 될 것인가?
   - (b) $\theta(t)$와 위상 초상화 궤적을 시뮬레이션하고 그려라.
   - (c) $\beta = 0$ (감쇠 없음)이면 거동이 어떻게 바뀌는가?

**5.** `scipy.linalg.expm`을 사용하여 $A = \begin{pmatrix} 0 & 1 \\ -4 & 0 \end{pmatrix}$에 대해 $t = 0, \pi/4, \pi/2, \pi$에서 $e^{At}$를 계산하라. 해가 닫힌 궤도(중심)를 이루는지 검증하라. 주기는 얼마인가?

---

## 참고 자료

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations*, 11th Edition, Chapters 7-9
- **Steven H. Strogatz**, *Nonlinear Dynamics and Chaos*, 2nd Edition (정성적 분석과 응용에 탁월)
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 4
- **Lawrence Perko**, *Differential Equations and Dynamical Systems*, 3rd Edition
- **SciPy expm**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

---

[이전: 2계 상미분방정식](./13_Second_Order_ODE.md) | [다음: ODE를 위한 라플라스 변환](./15_Laplace_Transform_for_ODE.md)

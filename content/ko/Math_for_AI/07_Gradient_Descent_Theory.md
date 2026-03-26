# 07. 경사 하강법 이론 (Gradient Descent Theory)

## 학습 목표

- 경사 하강법의 기본 원리와 업데이트 규칙을 이해하고 구현한다
- 볼록 함수와 강볼록 함수에서의 수렴 속도를 이론적으로 분석한다
- 확률적 경사 하강법(SGD)의 원리와 미니배치의 역할을 학습한다
- 모멘텀과 네스테로프 가속 경사법의 작동 원리를 물리적 직관으로 이해한다
- Adam, RMSProp 등 적응적 학습률 방법의 유도 과정을 학습한다
- 신경망 최적화에서의 실전 고려사항을 이해하고 적용한다

---

## 1. 경사 하강법 기본

### 1.1 기본 원리

경사 하강법(Gradient Descent)은 함수를 최소화하기 위해 그래디언트의 반대 방향으로 반복적으로 이동하는 1차 최적화 알고리즘입니다.

**업데이트 규칙:**

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

- $x_t$: $t$ 시점의 파라미터
- $\eta$: 학습률 (learning rate, step size)
- $\nabla f(x_t)$: $x_t$에서의 그래디언트

**직관:**
- 그래디언트 $\nabla f(x)$는 함수가 가장 빠르게 증가하는 방향
- 음의 그래디언트 $-\nabla f(x)$는 가장 빠르게 감소하는 방향 (최대 경사 하강)
- 학습률 $\eta$는 각 스텝의 크기를 조절

### 1.2 1차 테일러 근사

경사 하강법은 1차 테일러 근사에 기반합니다:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
$$

$\Delta x = -\eta \nabla f(x)$로 선택하면:

$$
f(x - \eta \nabla f(x)) \approx f(x) - \eta \|\nabla f(x)\|^2
$$

$\eta$가 충분히 작으면 함수 값이 감소합니다.

### 1.3 구현 및 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 목적 함수: f(x,y) = (x-1)^2 + 2(y-2)^2
def objective(x, y):
    return (x - 1)**2 + 2*(y - 2)**2

def gradient(x, y):
    df_dx = 2*(x - 1)
    df_dy = 4*(y - 2)
    return np.array([df_dx, df_dy])

# 경사 하강법
def gradient_descent(x0, learning_rate, n_iterations):
    """기본 경사 하강법"""
    trajectory = [x0]
    x = x0.copy()

    for _ in range(n_iterations):
        grad = gradient(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return np.array(trajectory)

# 초기점
x0 = np.array([-2.0, -1.0])

# 여러 학습률로 실험
learning_rates = [0.1, 0.3, 0.5, 0.9]
n_iterations = 50

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# 등고선 그리드
x_vals = np.linspace(-3, 3, 300)
y_vals = np.linspace(-2, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = objective(X, Y)

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]

    # 등고선
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # 경사 하강법 궤적
    trajectory = gradient_descent(x0, lr, n_iterations)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2,
            markersize=6, label='GD 궤적')
    ax.plot(x0[0], x0[1], 'g*', markersize=20, label='시작점')
    ax.plot(1, 2, 'r*', markersize=20, label='최솟값')

    # 그래디언트 벡터 표시 (처음 몇 개)
    for i in range(0, min(5, len(trajectory)-1), 1):
        x_curr = trajectory[i]
        grad = gradient(x_curr[0], x_curr[1])
        ax.quiver(x_curr[0], x_curr[1], -grad[0], -grad[1],
                  angles='xy', scale_units='xy', scale=5,
                  color='blue', width=0.005, alpha=0.7)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'학습률 η = {lr} ({len(trajectory)-1} iterations)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 4)

    # 수렴 정보
    final_x = trajectory[-1]
    final_loss = objective(final_x[0], final_x[1])
    distance_to_optimum = np.linalg.norm(final_x - np.array([1, 2]))
    ax.text(0.05, 0.95, f'최종 손실: {final_loss:.4f}\n최솟값까지 거리: {distance_to_optimum:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('gradient_descent_learning_rates.png', dpi=150)
plt.show()

print("학습률의 영향:")
print("  - η 너무 작음: 수렴 느림")
print("  - η 적절함: 빠른 수렴")
print("  - η 너무 큼: 발산 또는 진동")
```

### 1.4 학습률의 선택

**학습률이 너무 작으면:**
- 수렴이 매우 느림
- 많은 반복 필요

**학습률이 너무 크면:**
- 최솟값 주변에서 진동
- 발산 가능

**적절한 학습률:**
- 이론적 상한: $\eta \leq \frac{1}{L}$ (L: 립시츠 상수)
- 실전: 그리드 서치 또는 학습률 스케줄

```python
# 학습률에 따른 수렴 곡선
fig, ax = plt.subplots(figsize=(12, 6))

for lr in learning_rates:
    trajectory = gradient_descent(x0, lr, n_iterations)
    losses = [objective(x[0], x[1]) for x in trajectory]
    ax.plot(losses, linewidth=2, label=f'η = {lr}')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('학습률에 따른 손실 감소', fontsize=14)
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_rate_convergence.png', dpi=150)
plt.show()
```

## 2. 수렴 분석 (Convergence Analysis)

### 2.1 립시츠 연속 그래디언트

함수 $f$의 그래디언트가 립시츠 연속이라는 것은 다음을 만족하는 상수 $L > 0$이 존재한다는 의미입니다:

$$
\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|, \quad \forall x, y
$$

이는 $\nabla^2 f(x) \preceq LI$ (헤시안이 $LI$로 상한 bounded)와 동등합니다.

### 2.2 볼록 함수에서의 수렴

**정리 (볼록 함수):**
$f$가 볼록이고 그래디언트가 $L$-립시츠 연속이면, 학습률 $\eta = \frac{1}{L}$일 때:

$$
f(x_t) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2t}
$$

즉, **서브선형 (sublinear) 수렴**: $O(1/t)$

### 2.3 강볼록 함수에서의 수렴

함수 $f$가 $m$-강볼록이라는 것은:

$$
f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\|y-x\|^2
$$

또는 동등하게 $\nabla^2 f(x) \succeq mI$.

**정리 (강볼록 함수):**
$f$가 $m$-강볼록이고 그래디언트가 $L$-립시츠 연속이면, 학습률 $\eta = \frac{1}{L}$일 때:

$$
\|x_t - x^*\|^2 \leq \left(1 - \frac{m}{L}\right)^t \|x_0 - x^*\|^2
$$

즉, **선형 수렴 (linear convergence)**: $O(\rho^t)$, where $\rho = 1 - \frac{m}{L} < 1$

**조건수 (Condition Number):**
$$\kappa = \frac{L}{m}$$

조건수가 클수록 (ill-conditioned) 수렴이 느립니다.

### 2.4 수렴 속도 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

# 이차 형식: f(x) = 0.5 * x^T A x
# 강볼록: eigenvalues(A) > 0

def create_quadratic(m, L, dim=10):
    """강볼록 이차 함수 생성 (조건수 κ = L/m)"""
    # 고유값을 m과 L 사이에 균등 분포
    eigenvalues = np.linspace(m, L, dim)
    # 랜덤 직교 행렬
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    # A = Q Λ Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    return A

def quadratic_objective(x, A):
    return 0.5 * x @ A @ x

def quadratic_gradient(x, A):
    return A @ x

def gd_quadratic(A, x0, learning_rate, n_iterations):
    """이차 함수에 대한 경사 하강법"""
    x = x0.copy()
    trajectory = [np.linalg.norm(x)**2]  # ||x - x*||^2, x* = 0

    for _ in range(n_iterations):
        grad = quadratic_gradient(x, A)
        x = x - learning_rate * grad
        trajectory.append(np.linalg.norm(x)**2)

    return trajectory

# 실험 설정
dim = 10
n_iterations = 100

# 여러 조건수로 실험
condition_numbers = [1, 10, 100, 1000]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for kappa in condition_numbers:
    m = 1.0
    L = kappa * m
    A = create_quadratic(m, L, dim)

    # 초기점
    x0 = np.random.randn(dim)

    # 경사 하강법
    learning_rate = 1 / L
    trajectory = gd_quadratic(A, x0, learning_rate, n_iterations)

    # 선형 스케일
    axes[0].plot(trajectory, linewidth=2, label=f'κ = {kappa}')

    # 로그 스케일
    axes[1].semilogy(trajectory, linewidth=2, label=f'κ = {kappa}')

    # 이론적 수렴 속도
    rho = 1 - m/L
    theoretical = [trajectory[0] * (rho ** t) for t in range(n_iterations + 1)]
    axes[1].semilogy(theoretical, '--', linewidth=1, alpha=0.6,
                     label=f'이론 (κ={kappa})')

axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('$\|x_t - x^*\|^2$', fontsize=12)
axes[0].set_title('수렴 곡선 (선형 스케일)', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('$\|x_t - x^*\|^2$ (log scale)', fontsize=12)
axes[1].set_title('수렴 곡선 (로그 스케일): 선형 수렴', fontsize=14)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_analysis_condition_number.png', dpi=150)
plt.show()

print("조건수의 영향:")
print("  κ = 1: 완벽한 조건 (모든 방향 동일한 곡률)")
print("  κ >> 1: ill-conditioned (일부 방향 매우 평탄)")
print("  수렴 속도: O((1 - 1/κ)^t)")
```

## 3. 확률적 경사 하강법 (Stochastic Gradient Descent)

### 3.1 배치 vs 미니배치

**배치 경사 하강법 (Batch GD):**
전체 데이터셋으로 그래디언트 계산:
$$x_{t+1} = x_t - \eta \nabla f(x_t) = x_t - \eta \frac{1}{n}\sum_{i=1}^n \nabla f_i(x_t)$$

**확률적 경사 하강법 (SGD):**
랜덤하게 선택한 하나의 샘플로 그래디언트 추정:
$$x_{t+1} = x_t - \eta \nabla f_{i_t}(x_t)$$

**미니배치 SGD:**
$B$ 개 샘플의 미니배치로 그래디언트 추정:
$$x_{t+1} = x_t - \eta \frac{1}{|B|}\sum_{i \in B} \nabla f_i(x_t)$$

### 3.2 SGD의 장점과 단점

**장점:**
- **계산 효율**: 매 반복마다 전체 데이터 불필요
- **메모리 효율**: 큰 데이터셋에 적합
- **정규화 효과**: 노이즈가 날카로운 최솟값 탈출 도움
- **온라인 학습**: 데이터가 스트리밍으로 도착해도 가능

**단점:**
- **노이즈**: 그래디언트 추정이 불안정
- **학습률 조정**: 배치 GD보다 민감
- **수렴 속도**: 이론적으로 느림 (하지만 실전에서는 빠름)

### 3.3 미니배치 크기의 영향

```python
import numpy as np
import matplotlib.pyplot as plt

# 합성 데이터: 선형 회귀
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
y = X @ true_w + 0.1 * np.random.randn(n_samples)

def mse_loss(w, X, y):
    """MSE 손실"""
    return 0.5 * np.mean((X @ w - y) ** 2)

def mse_gradient(w, X, y):
    """MSE 그래디언트"""
    return X.T @ (X @ w - y) / len(y)

def sgd_minibatch(X, y, batch_size, learning_rate, n_epochs):
    """미니배치 SGD"""
    n_samples = len(X)
    w = np.zeros(X.shape[1])
    losses = []

    for epoch in range(n_epochs):
        # 데이터 셔플
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # 미니배치로 나누기
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 그래디언트 계산 및 업데이트
            grad = mse_gradient(w, X_batch, y_batch)
            w = w - learning_rate * grad

        # 에포크마다 전체 손실 기록
        loss = mse_loss(w, X, y)
        losses.append(loss)

    return w, losses

# 여러 배치 크기로 실험
batch_sizes = [1, 10, 50, 200, 1000]
n_epochs = 50
learning_rate = 0.01

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for batch_size in batch_sizes:
    w_final, losses = sgd_minibatch(X, y, batch_size, learning_rate, n_epochs)

    # 손실 곡선
    axes[0].plot(losses, linewidth=2, label=f'Batch size = {batch_size}')
    axes[1].semilogy(losses, linewidth=2, label=f'Batch size = {batch_size}')

axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('배치 크기에 따른 수렴', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MSE Loss (log scale)', fontsize=12)
axes[1].set_title('배치 크기에 따른 수렴 (로그 스케일)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sgd_batch_size_effect.png', dpi=150)
plt.show()

print("배치 크기의 영향:")
print("  작은 배치: 노이즈 크고 불안정, 정규화 효과, 메모리 효율")
print("  큰 배치: 안정적 그래디언트, 빠른 수렴, 계산 병렬화")
print("  실전: 32, 64, 128, 256 등 2의 거듭제곱 (GPU 최적화)")
```

### 3.4 SGD의 분산과 학습률

SGD 그래디언트의 기댓값은 참 그래디언트와 같지만, 분산이 존재합니다:

$$
\mathbb{E}[\nabla f_i(x)] = \nabla f(x), \quad \text{Var}[\nabla f_i(x)] = \sigma^2
$$

분산이 크면 수렴이 느려지고 불안정해집니다. 해결책:
- **학습률 감소 (learning rate decay)**: $\eta_t = \frac{\eta_0}{\sqrt{t}}$
- **미니배치 증가**: 분산 $\propto 1/|B|$
- **적응적 방법**: Adam, RMSProp

## 4. 모멘텀 기반 방법

### 4.1 모멘텀 SGD

기본 모멘텀 (Heavy Ball):

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla f(x_t) \\
x_{t+1} &= x_t - \eta v_t
\end{align}
$$

여기서 $\beta \in [0, 1)$은 모멘텀 계수 (일반적으로 0.9).

**물리적 직관:**
- 공이 언덕을 구르는 것처럼, 과거 방향의 관성 유지
- 일관된 방향으로 가속, 진동 감쇠
- 지역 최솟값과 안장점 탈출에 도움

### 4.2 네스테로프 가속 경사법 (NAG)

**Nesterov Accelerated Gradient:**

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla f(x_t - \beta v_{t-1}) \\
x_{t+1} &= x_t - \eta v_t
\end{align}
$$

**핵심 아이디어:**
- "미래"를 먼저 내다보고 ($x_t - \beta v_{t-1}$) 그곳에서 그래디언트 계산
- 모멘텀보다 더 똑똑한 look-ahead
- 이론적으로 더 나은 수렴 속도

```python
import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock 함수
def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_gradient(x, y):
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy])

# 기본 GD
def gd(x0, learning_rate, n_iterations):
    trajectory = [x0]
    x = x0.copy()

    for _ in range(n_iterations):
        grad = rosenbrock_gradient(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return np.array(trajectory)

# 모멘텀 GD
def momentum_gd(x0, learning_rate, beta, n_iterations):
    trajectory = [x0]
    x = x0.copy()
    v = np.zeros_like(x)

    for _ in range(n_iterations):
        grad = rosenbrock_gradient(x[0], x[1])
        v = beta * v + grad
        x = x - learning_rate * v
        trajectory.append(x.copy())

    return np.array(trajectory)

# Nesterov GD
def nesterov_gd(x0, learning_rate, beta, n_iterations):
    trajectory = [x0]
    x = x0.copy()
    v = np.zeros_like(x)

    for _ in range(n_iterations):
        # Look-ahead
        x_lookahead = x - beta * v
        grad = rosenbrock_gradient(x_lookahead[0], x_lookahead[1])
        v = beta * v + grad
        x = x - learning_rate * v
        trajectory.append(x.copy())

    return np.array(trajectory)

# 시각화
x0 = np.array([-1.0, 0.5])
learning_rate = 0.001
beta = 0.9
n_iterations = 200

trajectories = {
    'GD': gd(x0, learning_rate, n_iterations),
    'Momentum': momentum_gd(x0, learning_rate, beta, n_iterations),
    'Nesterov': nesterov_gd(x0, learning_rate, beta, n_iterations)
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 등고선
x_vals = np.linspace(-1.5, 1.5, 300)
y_vals = np.linspace(-0.5, 1.5, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

# 왼쪽: 궤적 비교
ax = axes[0]
levels = np.logspace(-1, 3, 20)
contour = ax.contour(X, Y, Z, levels=levels, alpha=0.4, cmap='gray')

colors = {'GD': 'blue', 'Momentum': 'red', 'Nesterov': 'green'}
for name, traj in trajectories.items():
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, color=colors[name],
            label=name, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=10, color=colors[name])

ax.plot(1, 1, 'k*', markersize=20, label='최솟값 (1, 1)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Rosenbrock 함수: 모멘텀 vs Nesterov', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 오른쪽: 손실 곡선
ax = axes[1]
for name, traj in trajectories.items():
    losses = [rosenbrock(x[0], x[1]) for x in traj]
    ax.semilogy(losses, linewidth=2, color=colors[name], label=name)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('손실 감소 비교', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_vs_nesterov.png', dpi=150)
plt.show()

print("모멘텀의 효과:")
print("  - 일관된 방향으로 가속")
print("  - 진동 감쇠 (특히 ill-conditioned 문제)")
print("  - 안장점 더 빠르게 통과")
print("\nNesterov의 장점:")
print("  - Look-ahead로 더 현명한 업데이트")
print("  - 이론적으로 더 나은 수렴 속도")
```

## 5. 적응적 학습률 방법

### 5.1 AdaGrad

**Adaptive Gradient Algorithm:**

$$
\begin{align}
G_t &= G_{t-1} + \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(누적 제곱)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(x_t)
\end{align}
$$

- $\odot$: 원소별 곱셈 (Hadamard product)
- $\epsilon$: 수치 안정성 ($10^{-8}$)

**특징:**
- 자주 업데이트되는 파라미터: 학습률 감소
- 드물게 업데이트되는 파라미터: 학습률 유지
- **문제점**: $G_t$가 계속 증가 → 학습률이 너무 작아짐

### 5.2 RMSProp

**Root Mean Square Propagation:**

$$
\begin{align}
G_t &= \beta G_{t-1} + (1-\beta) \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(지수 이동 평균)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(x_t)
\end{align}
$$

**AdaGrad 개선:**
- 누적 대신 지수 이동 평균 (EMA)
- 오래된 그래디언트의 영향 감쇠
- 학습률이 너무 작아지는 문제 해결

### 5.3 Adam

**Adaptive Moment Estimation:**

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla f(x_t) \quad \text{(1차 모멘트, 평균)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(2차 모멘트, 분산)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(편향 보정)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
\end{align}
$$

**하이퍼파라미터:**
- $\beta_1 = 0.9$ (1차 모멘트 감쇠)
- $\beta_2 = 0.999$ (2차 모멘트 감쇠)
- $\eta = 0.001$ (학습률)
- $\epsilon = 10^{-8}$

**특징:**
- 모멘텀 + 적응적 학습률
- 편향 보정: 초기 단계에서 모멘트 추정의 편향 제거
- 대부분의 딥러닝 문제에서 잘 작동

### 5.4 구현 및 비교

```python
import numpy as np
import matplotlib.pyplot as plt

# 최적화 알고리즘 구현
class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

class SGD(Optimizer):
    def update(self, x, grad):
        return x - self.lr * grad

class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v = None

    def update(self, x, grad):
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + grad
        return x - self.lr * self.v

class AdaGrad(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None

    def update(self, x, grad):
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad ** 2
        return x - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.G = None

    def update(self, x, grad):
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G = self.beta * self.G + (1 - self.beta) * grad ** 2
        return x - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, x, grad):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        # 편향 보정
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# 테스트 함수: Beale 함수
def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def beale_gradient(x, y):
    df_dx = 2*(1.5 - x + x*y)*(-1 + y) + 2*(2.25 - x + x*y**2)*(-1 + y**2) + \
            2*(2.625 - x + x*y**3)*(-1 + y**3)
    df_dy = 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + \
            2*(2.625 - x + x*y**3)*(3*x*y**2)
    return np.array([df_dx, df_dy])

# 최적화 실행
x0 = np.array([3.0, 3.0])
n_iterations = 500

optimizers = {
    'SGD': SGD(0.001),
    'Momentum': Momentum(0.001, beta=0.9),
    'AdaGrad': AdaGrad(0.5),
    'RMSProp': RMSProp(0.01, beta=0.9),
    'Adam': Adam(0.01, beta1=0.9, beta2=0.999)
}

trajectories = {}
for name, opt in optimizers.items():
    x = x0.copy()
    traj = [x.copy()]

    for _ in range(n_iterations):
        grad = beale_gradient(x[0], x[1])
        x = opt.update(x, grad)
        traj.append(x.copy())

    trajectories[name] = np.array(traj)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 등고선
x_vals = np.linspace(-1, 4, 300)
y_vals = np.linspace(-1, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = beale(X, Y)

# 왼쪽: 궤적
ax = axes[0]
levels = np.logspace(0, 4, 20)
contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, cmap='gray')

colors = {'SGD': 'blue', 'Momentum': 'red', 'AdaGrad': 'green',
          'RMSProp': 'purple', 'Adam': 'orange'}

for name, traj in trajectories.items():
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, color=colors[name],
            label=name, alpha=0.7)

ax.plot(3, 0.5, 'k*', markersize=20, label='최솟값')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Beale 함수: 옵티마이저 비교', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)

# 오른쪽: 손실 곡선
ax = axes[1]
for name, traj in trajectories.items():
    losses = [beale(x[0], x[1]) for x in traj]
    ax.semilogy(losses, linewidth=2, color=colors[name], label=name)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('손실 감소 비교', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
plt.show()

print("옵티마이저 특징:")
print("  SGD: 단순, 느림")
print("  Momentum: 가속, 진동 감쇠")
print("  AdaGrad: 희소 데이터 적합, 학습률 감소 문제")
print("  RMSProp: AdaGrad 개선, 안정적")
print("  Adam: 대부분 상황에서 우수, 사실상 표준")
```

### 5.5 Adam의 편향 보정 (Bias Correction)

초기 단계에서 $m_t$와 $v_t$는 0으로 초기화되므로 0으로 편향됩니다. 편향 보정은 이를 수정합니다.

$$
\mathbb{E}[m_t] = \mathbb{E}\left[\sum_{i=1}^t \beta_1^{t-i}(1-\beta_1)g_i\right] = \mathbb{E}[g_t](1 - \beta_1^t)
$$

따라서 $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$는 불편 추정량입니다.

```python
# 편향 보정 효과 시각화
fig, ax = plt.subplots(figsize=(12, 6))

beta1, beta2 = 0.9, 0.999
t_vals = np.arange(1, 101)

correction1 = 1 / (1 - beta1 ** t_vals)
correction2 = 1 / (1 - beta2 ** t_vals)

ax.plot(t_vals, correction1, linewidth=2, label=f'1차 모멘트 보정 (β₁={beta1})')
ax.plot(t_vals, correction2, linewidth=2, label=f'2차 모멘트 보정 (β₂={beta2})')
ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='보정 없음')

ax.set_xlabel('Iteration t', fontsize=12)
ax.set_ylabel('보정 계수', fontsize=12)
ax.set_title('Adam 편향 보정 계수', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('adam_bias_correction.png', dpi=150)
plt.show()

print("편향 보정의 중요성:")
print("  - 초기 스텝에서 모멘트가 0으로 편향")
print("  - 보정 없으면 초기 학습률이 과도하게 작음")
print("  - 수십 iteration 후 보정 효과 사라짐")
```

## 6. 학습률 스케줄

### 6.1 주요 스케줄링 전략

**Step Decay:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

**Exponential Decay:**
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

**Cosine Annealing:**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**1-Cycle Policy:**
- 초기: 학습률 증가 (warm-up)
- 중간: 최대 학습률 유지
- 마지막: 학습률 감소 (annealing)

```python
import numpy as np
import matplotlib.pyplot as plt

def step_decay(t, eta0=0.1, gamma=0.5, k=50):
    return eta0 * (gamma ** (t // k))

def exponential_decay(t, eta0=0.1, lam=0.01):
    return eta0 * np.exp(-lam * t)

def cosine_annealing(t, eta_min=0.0, eta_max=0.1, T=100):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

def one_cycle(t, eta_min=0.001, eta_max=0.1, T=100, warmup_frac=0.3):
    if t < warmup_frac * T:
        # Warm-up
        return eta_min + (eta_max - eta_min) * t / (warmup_frac * T)
    else:
        # Annealing
        progress = (t - warmup_frac * T) / ((1 - warmup_frac) * T)
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

# 시각화
t_vals = np.arange(0, 200)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

schedules = [
    ('Step Decay', step_decay, axes[0, 0]),
    ('Exponential Decay', exponential_decay, axes[0, 1]),
    ('Cosine Annealing', cosine_annealing, axes[1, 0]),
    ('1-Cycle Policy', one_cycle, axes[1, 1])
]

for name, func, ax in schedules:
    if name == 'Cosine Annealing':
        lr_vals = [func(t, T=200) for t in t_vals]
    elif name == '1-Cycle Policy':
        lr_vals = [func(t, T=200) for t in t_vals]
    else:
        lr_vals = [func(t) for t in t_vals]

    ax.plot(t_vals, lr_vals, linewidth=2, color='blue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_schedules.png', dpi=150)
plt.show()

print("학습률 스케줄의 역할:")
print("  - 초기: 큰 학습률로 빠르게 좋은 영역 탐색")
print("  - 후반: 작은 학습률로 미세 조정")
print("  - Warm-up: 초기 불안정성 방지")
print("  - Cosine/1-Cycle: 부드러운 감소, Transformer 등에서 인기")
```

## 7. 신경망 최적화의 실전 고려사항

### 7.1 손실 지형의 기하학

신경망의 손실 함수는:
- **비볼록**: 여러 지역 최솟값
- **고차원**: 파라미터가 수백만~수십억 개
- **안장점 (Saddle Points)**: 어떤 방향은 최솟값, 다른 방향은 최댓값
- **평탄 영역 (Plateaus)**: 그래디언트가 거의 0

### 7.2 Sharp Minima vs Flat Minima

**Sharp Minima:**
- 좁고 날카로운 최솟값
- 테스트 데이터에서 일반화 성능 낮음
- 큰 배치 크기에서 자주 발생

**Flat Minima:**
- 넓고 평탄한 최솟값
- 테스트 데이터에서 일반화 성능 높음
- 작은 배치 크기 + SGD 노이즈로 선호

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 손실 지형 시뮬레이션
def sharp_minimum(x, y):
    """날카로운 최솟값"""
    return x**2 + y**2 + 0.01 * np.random.randn()

def flat_minimum(x, y):
    """평탄한 최솟값"""
    return 0.1 * x**2 + 0.1 * y**2 + 0.01 * np.random.randn()

fig = plt.figure(figsize=(16, 6))

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Sharp minimum
ax1 = fig.add_subplot(121, projection='3d')
Z_sharp = X**2 + Y**2
ax1.plot_surface(X, Y, Z_sharp, cmap='Reds', alpha=0.8, edgecolor='none')
ax1.set_title('Sharp Minimum (일반화 낮음)', fontsize=14)
ax1.set_xlabel('w₁')
ax1.set_ylabel('w₂')
ax1.set_zlabel('Loss')

# Flat minimum
ax2 = fig.add_subplot(122, projection='3d')
Z_flat = 0.1 * X**2 + 0.1 * Y**2
ax2.plot_surface(X, Y, Z_flat, cmap='Blues', alpha=0.8, edgecolor='none')
ax2.set_title('Flat Minimum (일반화 높음)', fontsize=14)
ax2.set_xlabel('w₁')
ax2.set_ylabel('w₂')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.savefig('sharp_vs_flat_minima.png', dpi=150)
plt.show()

print("Sharp vs Flat Minima:")
print("  Sharp: 파라미터 작은 변화에 손실 크게 변함 → 과적합")
print("  Flat: 파라미터 변화에 손실 둔감 → 일반화")
print("  SGD의 노이즈가 Flat Minima 선호하도록 암묵적 정규화")
```

### 7.3 그래디언트 클리핑 (Gradient Clipping)

RNN/Transformer 등에서 그래디언트가 폭발하는 문제 방지:

**Norm-based Clipping:**
$$
\tilde{g} = \begin{cases}
g & \text{if } \|g\| \leq \theta \\
\theta \frac{g}{\|g\|} & \text{otherwise}
\end{cases}
$$

**Value-based Clipping:**
$$
\tilde{g}_i = \max(-\theta, \min(\theta, g_i))
$$

```python
import torch

def gradient_clipping_demo():
    # 간단한 RNN 시뮬레이션
    torch.manual_seed(42)
    hidden_size = 10
    W = torch.randn(hidden_size, hidden_size, requires_grad=True) * 2  # 큰 가중치

    # 순방향 (여러 시간 스텝)
    h = torch.randn(hidden_size)
    for _ in range(20):
        h = torch.tanh(W @ h)

    loss = h.sum()
    loss.backward()

    grad_norm = W.grad.norm().item()
    print(f"클리핑 전 그래디언트 노름: {grad_norm:.4f}")

    # 그래디언트 클리핑
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_([W], max_norm)

    clipped_grad_norm = W.grad.norm().item()
    print(f"클리핑 후 그래디언트 노름: {clipped_grad_norm:.4f}")

gradient_clipping_demo()

print("\n그래디언트 클리핑:")
print("  - RNN/Transformer에서 필수")
print("  - 일반적으로 max_norm = 1.0 또는 5.0")
print("  - 학습 안정성 크게 향상")
```

### 7.4 실전 최적화 레시피

```python
# PyTorch 스타일 최적화 설정 예제
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNN()

# 1. 옵티마이저 선택: Adam (대부분의 경우)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# 2. 학습률 스케줄러: Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. 학습 루프 (가상)
for epoch in range(10):
    # Forward & Backward
    # loss.backward()

    # 그래디언트 클리핑 (RNN/Transformer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 옵티마이저 스텝
    optimizer.step()
    optimizer.zero_grad()

    # 학습률 업데이트
    scheduler.step()

    print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")

print("\n실전 최적화 팁:")
print("  1. Adam을 기본으로 시작 (lr=1e-3)")
print("  2. Cosine Annealing 또는 Step Decay 적용")
print("  3. Warm-up 사용 (Transformer 등)")
print("  4. 그래디언트 클리핑 (RNN/Transformer)")
print("  5. 배치 크기: 32-256 (GPU 메모리에 따라)")
print("  6. Weight Decay (L2 정규화): 1e-4 ~ 1e-5")
```

## 연습 문제

1. **수렴 속도 분석**: $f(x) = \frac{1}{2}x^T A x$ (이차 형식)에 대해, 조건수 $\kappa = 10$일 때와 $\kappa = 100$일 때 경사 하강법의 수렴 속도를 시뮬레이션하고 비교하시오. 이론적 수렴 속도 $O((1 - 1/\kappa)^t)$와 일치하는지 검증하시오.

2. **SGD vs 배치 GD**: MNIST 데이터셋에서 간단한 2층 신경망을 학습시키되, 배치 GD, 미니배치 SGD (배치 크기 32, 128, 512), 그리고 full SGD (배치 크기 1)를 비교하시오. 수렴 속도, 최종 테스트 정확도, 그리고 계산 시간을 비교 분석하시오.

3. **옵티마이저 구현**: Adam 옵티마이저를 NumPy로 처음부터 구현하고, 편향 보정의 유무에 따른 차이를 시각화하시오. Rosenbrock 함수나 Beale 함수를 사용하여 테스트하시오.

4. **모멘텀 vs Nesterov**: ill-conditioned 이차 함수 ($\kappa = 100$)에서 모멘텀 SGD와 Nesterov 가속 경사법의 수렴 속도를 비교하시오. 어떤 상황에서 Nesterov가 더 우수한지 분석하시오.

5. **학습률 스케줄**: 간단한 CNN을 CIFAR-10에서 학습시키되, 다음 스케줄을 비교하시오: (1) 고정 학습률, (2) Step Decay, (3) Cosine Annealing, (4) 1-Cycle Policy. 각 방법의 학습 곡선과 최종 테스트 정확도를 보고하시오.

## 참고 자료

- Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms". arXiv:1609.04747
  - 모든 주요 옵티마이저의 포괄적 리뷰
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning". *SIAM Review*, 60(2), 223-311
  - 대규모 ML 최적화의 이론과 실전
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization". ICLR
  - Adam 논문 원문
- Sutskever, I., et al. (2013). "On the Importance of Initialization and Momentum in Deep Learning". ICML
  - 모멘텀의 중요성
- Keskar, N. S., et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima". ICLR
  - Sharp vs Flat Minima
- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks". WACV
  - 1-Cycle Policy
- PyTorch Optimization Documentation: https://pytorch.org/docs/stable/optim.html
- Stanford CS231n Lecture Notes: http://cs231n.github.io/neural-networks-3/

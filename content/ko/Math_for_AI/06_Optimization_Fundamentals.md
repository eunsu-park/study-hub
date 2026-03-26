# 06. 최적화 기초 (Optimization Fundamentals)

## 학습 목표

- 최적화 문제의 수학적 형식화와 제약 조건의 종류를 이해한다
- 볼록 집합과 볼록 함수의 정의와 성질을 학습하고, 볼록성이 최적화에 미치는 영향을 파악한다
- 극값의 필요조건과 충분조건을 그래디언트와 헤시안으로 표현하고 적용한다
- 라그랑주 승수법을 사용하여 등식 제약 최적화 문제를 해결한다
- KKT 조건을 이해하고 부등식 제약이 있는 최적화 문제에 적용한다
- 머신러닝에서 최적화 이론이 어떻게 활용되는지 실제 예제를 통해 학습한다

---

## 1. 최적화 문제의 형식화

### 1.1 최적화 문제의 일반 형태

최적화는 주어진 제약 조건 하에서 목적 함수를 최소화(또는 최대화)하는 변수 값을 찾는 과정입니다.

**표준 형태:**

$$
\begin{align}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & h_i(x) = 0, \quad i = 1, \ldots, m \\
& g_j(x) \leq 0, \quad j = 1, \ldots, p
\end{align}
$$

- $f(x)$: 목적 함수 (objective function)
- $h_i(x) = 0$: 등식 제약 조건 (equality constraints)
- $g_j(x) \leq 0$: 부등식 제약 조건 (inequality constraints)

### 1.2 실현 가능 영역 (Feasible Region)

실현 가능 영역 $\mathcal{F}$는 모든 제약 조건을 만족하는 점들의 집합입니다:

$$
\mathcal{F} = \{x \in \mathbb{R}^n : h_i(x) = 0, \; g_j(x) \leq 0, \; \forall i, j\}
$$

**예제: 2차원 제약 최적화**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize

# 목적 함수: f(x,y) = (x-1)^2 + (y-2)^2
def objective(X):
    x, y = X
    return (x - 1)**2 + (y - 2)**2

# 등식 제약: h(x,y) = x + y - 1 = 0
def equality_constraint(X):
    x, y = X
    return x + y - 1

# 부등식 제약: g(x,y) = x^2 + y^2 - 4 <= 0 (원 안쪽)
def inequality_constraint(X):
    x, y = X
    return x**2 + y**2 - 4

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 목적 함수 등고선
x = np.linspace(-2.5, 3, 300)
y = np.linspace(-2.5, 3, 300)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2)**2
contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 등식 제약 (직선)
x_line = np.linspace(-2.5, 3, 100)
y_line = 1 - x_line
ax.plot(x_line, y_line, 'r-', linewidth=2, label='$x + y = 1$ (등식 제약)')

# 부등식 제약 (원)
circle = Circle((0, 0), 2, fill=False, edgecolor='blue', linewidth=2,
                label='$x^2 + y^2 \leq 4$ (부등식 제약)')
ax.add_patch(circle)

# 실현 가능 영역 강조
theta = np.linspace(0, 2*np.pi, 100)
x_circle = 2 * np.cos(theta)
y_circle = 2 * np.sin(theta)
mask = (x_circle + y_circle >= 0.9) & (x_circle + y_circle <= 1.1)
ax.fill_between(x_circle[mask], -2.5, y_circle[mask], alpha=0.2, color='green')

ax.set_xlim(-2.5, 3)
ax.set_ylim(-2.5, 3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('제약 최적화 문제: 실현 가능 영역과 목적 함수', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('optimization_feasible_region.png', dpi=150)
plt.show()

print("실현 가능 영역: 직선과 원의 교선 중 제약을 만족하는 부분")
```

### 1.3 최적화 문제의 분류

- **제약 없는 최적화 (Unconstrained)**: $\min f(x)$
- **제약 있는 최적화 (Constrained)**: 등식/부등식 제약 포함
- **선형 계획법 (Linear Programming)**: $f$, $h$, $g$ 모두 선형
- **볼록 최적화 (Convex Optimization)**: $f$, $g$ 볼록, $h$ 아핀(affine)
- **비선형 최적화 (Nonlinear)**: 일반적인 비선형 함수

## 2. 볼록 집합과 볼록 함수

### 2.1 볼록 집합 (Convex Set)

집합 $C \subseteq \mathbb{R}^n$이 볼록이라는 것은 집합 내 임의의 두 점을 잇는 선분이 완전히 집합 내부에 포함되는 것을 의미합니다:

$$
x, y \in C, \; \lambda \in [0, 1] \implies \lambda x + (1-\lambda)y \in C
$$

**예제: 볼록 집합과 비볼록 집합**

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1. 원 (볼록)
theta = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta)
y1 = np.sin(theta)
axes[0].fill(x1, y1, alpha=0.3, color='green')
axes[0].plot([0.5, -0.5], [0.3, -0.7], 'ro-', linewidth=2, markersize=8)
axes[0].set_title('원 (볼록)', fontsize=12)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 2. 반평면 (볼록)
x2 = np.array([-2, 2, 2, -2])
y2 = np.array([0, 0, 2, 2])
axes[1].fill(x2, y2, alpha=0.3, color='green')
axes[1].plot([0.5, -0.5], [0.5, 1.5], 'ro-', linewidth=2, markersize=8)
axes[1].set_title('반평면 $y \geq 0$ (볼록)', fontsize=12)
axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].grid(True, alpha=0.3)

# 3. 다각형 (볼록)
x3 = np.array([0, 1, 0.5, -0.5, -1])
y3 = np.array([1, 0, -1, -1, 0])
axes[2].fill(x3, y3, alpha=0.3, color='green')
axes[2].plot([0.5, -0.5], [0.3, -0.3], 'ro-', linewidth=2, markersize=8)
axes[2].set_title('볼록 다각형 (볼록)', fontsize=12)
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

# 4. 초승달 (비볼록)
theta = np.linspace(0, 2*np.pi, 100)
x4_outer = 1.5 * np.cos(theta)
y4_outer = 1.5 * np.sin(theta)
x4_inner = 1.0 * np.cos(theta) + 0.5
y4_inner = 1.0 * np.sin(theta)
axes[3].fill(x4_outer, y4_outer, alpha=0.3, color='red')
axes[3].fill(x4_inner, y4_inner, alpha=1.0, color='white')
axes[3].plot([-0.5, 1.0], [0.5, -0.5], 'bo--', linewidth=2, markersize=8)
axes[3].set_title('초승달 (비볼록)', fontsize=12)
axes[3].set_aspect('equal')
axes[3].grid(True, alpha=0.3)

# 5. 별 모양 (비볼록)
theta = np.linspace(0, 2*np.pi, 11)
r = np.where(np.arange(11) % 2 == 0, 1.5, 0.7)
x5 = r * np.cos(theta)
y5 = r * np.sin(theta)
axes[4].fill(x5, y5, alpha=0.3, color='red')
axes[4].plot([1.0, -1.0], [0.0, 0.0], 'bo--', linewidth=2, markersize=8)
axes[4].set_title('별 (비볼록)', fontsize=12)
axes[4].set_aspect('equal')
axes[4].grid(True, alpha=0.3)

# 6. 고리 (비볼록)
theta = np.linspace(0, 2*np.pi, 100)
x6_outer = 1.5 * np.cos(theta)
y6_outer = 1.5 * np.sin(theta)
x6_inner = 0.7 * np.cos(theta)
y6_inner = 0.7 * np.sin(theta)
axes[5].fill(x6_outer, y6_outer, alpha=0.3, color='red')
axes[5].fill(x6_inner, y6_inner, alpha=1.0, color='white')
axes[5].plot([-1.0, 1.0], [0.0, 0.0], 'bo--', linewidth=2, markersize=8)
axes[5].set_title('고리 (비볼록)', fontsize=12)
axes[5].set_aspect('equal')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex_sets.png', dpi=150)
plt.show()

print("볼록 집합(초록): 두 점을 잇는 선분이 집합 내부에 있음")
print("비볼록 집합(빨강): 선분이 집합 밖으로 나감")
```

### 2.2 볼록 함수 (Convex Function)

함수 $f: \mathbb{R}^n \to \mathbb{R}$이 볼록이라는 것은 정의역이 볼록 집합이고 다음을 만족하는 것입니다:

$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall x, y \in \text{dom}(f), \; \lambda \in [0,1]
$$

**기하학적 의미**: 함수 그래프 위 두 점을 잇는 선분이 항상 함수 그래프 위쪽에 위치합니다.

**엄격한 볼록 (Strictly Convex)**: 부등호가 $<$로 바뀐 경우

**강볼록 (Strongly Convex)**: $\exists m > 0$ such that

$$
f(x) - \frac{m}{2}\|x\|^2 \text{ is convex}
$$

### 2.3 볼록 함수의 판별

**1차 조건 (미분 가능한 경우):**

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x), \quad \forall x, y
$$

**2차 조건 (두 번 미분 가능한 경우):**

$$
\nabla^2 f(x) \succeq 0 \quad \text{(헤시안이 양반정치)}
$$

강볼록: $\nabla^2 f(x) \succ 0$ (헤시안이 양정치)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 볼록 함수 예제
def convex_func(x, y):
    """볼록: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def nonconvex_func(x, y):
    """비볼록: f(x,y) = x^2 - y^2 (안장점)"""
    return x**2 - y**2

def visualize_function(func, title, ax):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title(title)

fig = plt.figure(figsize=(15, 6))

# 볼록 함수
ax1 = fig.add_subplot(121, projection='3d')
visualize_function(convex_func, '볼록 함수: $f(x,y) = x^2 + y^2$', ax1)

# 비볼록 함수
ax2 = fig.add_subplot(122, projection='3d')
visualize_function(nonconvex_func, '비볼록 함수: $f(x,y) = x^2 - y^2$', ax2)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex_functions.png', dpi=150)
plt.show()

# 헤시안 확인
def check_convexity(x, y, name):
    if name == "convex":
        # f(x,y) = x^2 + y^2
        # ∇²f = [[2, 0], [0, 2]]
        hessian = np.array([[2, 0], [0, 2]])
    else:
        # f(x,y) = x^2 - y^2
        # ∇²f = [[2, 0], [0, -2]]
        hessian = np.array([[2, 0], [0, -2]])

    eigenvalues = np.linalg.eigvals(hessian)
    print(f"\n{name} 함수의 헤시안 고유값: {eigenvalues}")

    if np.all(eigenvalues >= 0):
        print("  → 모든 고유값 ≥ 0: 볼록 함수")
    elif np.all(eigenvalues > 0):
        print("  → 모든 고유값 > 0: 강볼록 함수")
    else:
        print("  → 음의 고유값 존재: 비볼록 함수")

check_convexity(0, 0, "convex")
check_convexity(0, 0, "nonconvex")
```

### 2.4 볼록 함수의 중요한 성질

**정리 1**: 볼록 함수의 지역 최솟값은 전역 최솟값이다.

**정리 2**: 강볼록 함수는 유일한 전역 최솟값을 가진다.

**정리 3**: 볼록 함수의 합, 양의 스칼라배, 최댓값은 볼록이다.

**예제: 머신러닝에서 볼록 함수들**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 이차 함수 (Quadratic)
axes[0, 0].plot(x, x**2, linewidth=2)
axes[0, 0].set_title('$f(x) = x^2$ (Ridge 정규화)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# 2. 절댓값 (L1 norm)
axes[0, 1].plot(x, np.abs(x), linewidth=2, color='orange')
axes[0, 1].set_title('$f(x) = |x|$ (Lasso 정규화)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 3. 지수 함수
axes[0, 2].plot(x, np.exp(x), linewidth=2, color='green')
axes[0, 2].set_title('$f(x) = e^x$', fontsize=12)
axes[0, 2].set_ylim(0, 10)
axes[0, 2].grid(True, alpha=0.3)

# 4. 로지스틱 손실
axes[1, 0].plot(x, np.log(1 + np.exp(x)), linewidth=2, color='red')
axes[1, 0].set_title('$f(x) = \log(1 + e^x)$ (로지스틱 손실)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 5. Huber 손실
delta = 1.0
huber = np.where(np.abs(x) <= delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
axes[1, 1].plot(x, huber, linewidth=2, color='purple')
axes[1, 1].set_title('Huber 손실 ($\delta=1$)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# 6. -log (음의 로그, 볼록)
x_pos = np.linspace(0.01, 3, 300)
axes[1, 2].plot(x_pos, -np.log(x_pos), linewidth=2, color='brown')
axes[1, 2].set_title('$f(x) = -\log(x)$ (x > 0)', fontsize=12)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_convex_functions.png', dpi=150)
plt.show()

print("머신러닝에서 자주 사용하는 볼록 함수들:")
print("- 이차 함수: Ridge 정규화, MSE 손실")
print("- L1 노름: Lasso 정규화")
print("- 로지스틱 손실: 로지스틱 회귀")
print("- Huber 손실: 로버스트 회귀")
```

## 3. 극값의 필요조건과 충분조건

### 3.1 1차 필요조건 (First-Order Necessary Condition)

제약 없는 최적화에서 $x^*$가 지역 최솟값이면:

$$
\nabla f(x^*) = 0
$$

이러한 점을 **정류점 (stationary point)** 또는 **임계점 (critical point)**이라고 합니다.

### 3.2 2차 충분조건 (Second-Order Sufficient Condition)

$x^*$가 정류점일 때:
- **지역 최솟값**: $\nabla^2 f(x^*) \succ 0$ (헤시안이 양정치)
- **지역 최댓값**: $\nabla^2 f(x^*) \prec 0$ (헤시안이 음정치)
- **안장점 (Saddle Point)**: 헤시안이 부정부호 (양의 고유값과 음의 고유값이 섞임)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(X):
    """Rosenbrock 함수: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    x, y = X
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(X):
    x, y = X
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy])

def rosenbrock_hessian(X):
    x, y = X
    d2f_dx2 = 2 - 400*y + 1200*x**2
    d2f_dxdy = -400*x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# 최솟값 찾기
x0 = np.array([0.0, 0.0])
result = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_gradient)
x_min = result.x

print("Rosenbrock 함수 최적화")
print(f"최솟값 위치: x* = {x_min}")
print(f"최솟값: f(x*) = {rosenbrock(x_min):.6f}")
print(f"\n그래디언트: ∇f(x*) = {rosenbrock_gradient(x_min)}")

hessian = rosenbrock_hessian(x_min)
eigenvalues = np.linalg.eigvals(hessian)
print(f"\n헤시안:\n{hessian}")
print(f"고유값: {eigenvalues}")
print(f"  → 모든 고유값 > 0: 지역 최솟값 확인")

# 시각화
fig = plt.figure(figsize=(16, 6))

# 3D 표면
ax1 = fig.add_subplot(131, projection='3d')
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax1.scatter([x_min[0]], [x_min[1]], [rosenbrock(x_min)],
            color='red', s=100, label='최솟값')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Rosenbrock 함수', fontsize=12)
ax1.legend()

# 등고선
ax2 = fig.add_subplot(132)
levels = np.logspace(-1, 3, 20)
contour = ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(x_min[0], x_min[1], 'r*', markersize=20, label='최솟값 (1, 1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('등고선 플롯', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 안장점 예제: f(x,y) = x^2 - y^2
ax3 = fig.add_subplot(133, projection='3d')
Z_saddle = X**2 - Y**2
ax3.plot_surface(X, Y, Z_saddle, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax3.scatter([0], [0], [0], color='red', s=100, label='안장점 (0, 0)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('f(x,y)')
ax3.set_title('안장점: $f(x,y) = x^2 - y^2$', fontsize=12)
ax3.legend()

plt.tight_layout()
plt.savefig('critical_points_analysis.png', dpi=150)
plt.show()

# 안장점에서의 헤시안
print("\n\n안장점 예제: f(x,y) = x^2 - y^2")
saddle_hessian = np.array([[2, 0], [0, -2]])
saddle_eigenvalues = np.linalg.eigvals(saddle_hessian)
print(f"원점 (0, 0)에서의 헤시안:\n{saddle_hessian}")
print(f"고유값: {saddle_eigenvalues}")
print(f"  → 양의 고유값과 음의 고유값 혼재: 안장점")
```

## 4. 라그랑주 승수법 (Lagrange Multipliers)

### 4.1 등식 제약 최적화

다음 문제를 고려합니다:

$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & h(x) = 0
\end{align}
$$

**라그랑지안 (Lagrangian):**

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda^T h(x)
$$

여기서 $\lambda$는 **라그랑주 승수 (Lagrange multiplier)**입니다.

**최적성의 필요조건:**

$$
\begin{align}
\nabla_x \mathcal{L} &= \nabla f(x^*) + \lambda^* \nabla h(x^*) = 0 \\
\nabla_\lambda \mathcal{L} &= h(x^*) = 0
\end{align}
$$

**기하학적 해석**: 최적점에서 $\nabla f$와 $\nabla h$가 평행합니다 (같은 방향 또는 반대 방향).

### 4.2 예제: 등식 제약 최적화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 목적 함수: f(x,y) = x^2 + y^2
def objective(X):
    return X[0]**2 + X[1]**2

# 등식 제약: h(x,y) = x + 2y - 1 = 0
def constraint_eq(X):
    return X[0] + 2*X[1] - 1

# 라그랑지안: L(x,y,λ) = x^2 + y^2 + λ(x + 2y - 1)
def lagrangian(X, lam):
    return objective(X) + lam * constraint_eq(X)

# scipy로 최적화
constraints = {'type': 'eq', 'fun': constraint_eq}
x0 = np.array([0.0, 0.0])
result = minimize(objective, x0, method='SLSQP', constraints=constraints)

x_opt = result.x
print("라그랑주 승수법 예제")
print(f"최적 해: x* = {x_opt}")
print(f"목적 함수 값: f(x*) = {objective(x_opt):.6f}")
print(f"제약 조건 확인: h(x*) = {constraint_eq(x_opt):.6e}")

# 해석적 해 (손으로 계산)
# ∇L = 0: 2x + λ = 0, 2y + 2λ = 0, x + 2y - 1 = 0
# → x = -λ/2, y = -λ, x + 2y = 1 → -λ/2 - 2λ = 1 → λ = -2/5
# → x = 1/5, y = 2/5
x_analytical = np.array([1/5, 2/5])
lambda_analytical = -2/5

print(f"\n해석적 해: x* = {x_analytical}")
print(f"라그랑주 승수: λ* = {lambda_analytical}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 목적 함수 등고선
x = np.linspace(-1, 2, 300)
y = np.linspace(-1, 2, 300)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
contour = ax.contour(X, Y, Z, levels=15, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 제약 조건 (직선)
x_line = np.linspace(-1, 2, 100)
y_line = (1 - x_line) / 2
ax.plot(x_line, y_line, 'r-', linewidth=3, label='제약: $x + 2y = 1$')

# 최적점
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20, label=f'최적점 ({x_opt[0]:.2f}, {x_opt[1]:.2f})')

# 그래디언트 벡터
grad_f = 2 * x_opt  # ∇f = [2x, 2y]
grad_h = np.array([1, 2])  # ∇h = [1, 2]
ax.quiver(x_opt[0], x_opt[1], grad_f[0], grad_f[1],
          angles='xy', scale_units='xy', scale=5, color='blue', width=0.006,
          label='$\\nabla f$ (목적함수 그래디언트)')
ax.quiver(x_opt[0], x_opt[1], grad_h[0], grad_h[1],
          angles='xy', scale_units='xy', scale=5, color='green', width=0.006,
          label='$\\nabla h$ (제약 그래디언트)')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('라그랑주 승수법: $\\nabla f$와 $\\nabla h$가 평행', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('lagrange_multipliers.png', dpi=150)
plt.show()

print(f"\n최적점에서 ∇f = {grad_f}, ∇h = {grad_h}")
print(f"∇f = λ∇h 확인: λ = {grad_f[0] / grad_h[0]:.3f} (x 성분)")
```

## 5. KKT 조건 (Karush-Kuhn-Tucker Conditions)

### 5.1 부등식 제약 최적화

다음과 같은 일반적인 최적화 문제를 고려합니다:

$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & h_i(x) = 0, \quad i = 1, \ldots, m \\
& g_j(x) \leq 0, \quad j = 1, \ldots, p
\end{align}
$$

**일반화된 라그랑지안:**

$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{j=1}^p \mu_j g_j(x)
$$

### 5.2 KKT 조건

$x^*$가 최적해이고 어떤 정규성 조건이 만족되면, 다음 KKT 조건을 만족하는 $\lambda^*, \mu^*$가 존재합니다:

1. **정류성 (Stationarity):**
   $$\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = 0$$

2. **원시 실현가능성 (Primal Feasibility):**
   $$h_i(x^*) = 0, \quad g_j(x^*) \leq 0$$

3. **쌍대 실현가능성 (Dual Feasibility):**
   $$\mu_j^* \geq 0$$

4. **상보 이완 (Complementary Slackness):**
   $$\mu_j^* g_j(x^*) = 0, \quad \forall j$$

**상보 이완의 의미**: 각 부등식 제약에 대해
- $g_j(x^*) < 0$ (제약이 활성화되지 않음) → $\mu_j^* = 0$
- $\mu_j^* > 0$ (라그랑주 승수가 양) → $g_j(x^*) = 0$ (제약이 활성화됨)

### 5.3 예제: KKT 조건 적용

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 목적 함수: f(x,y) = (x-2)^2 + (y-1)^2
def objective(X):
    x, y = X
    return (x - 2)**2 + (y - 1)**2

def objective_grad(X):
    x, y = X
    return np.array([2*(x - 2), 2*(y - 1)])

# 부등식 제약: g1(x,y) = x + y - 1 <= 0
def constraint_ineq1(X):
    return X[0] + X[1] - 1

# 부등식 제약: g2(x,y) = -x <= 0 (즉, x >= 0)
def constraint_ineq2(X):
    return -X[0]

# 부등식 제약: g3(x,y) = -y <= 0 (즉, y >= 0)
def constraint_ineq3(X):
    return -X[1]

# scipy로 최적화
constraints = [
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq1(X)},  # scipy는 g(x) >= 0 형식
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq2(X)},
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq3(X)}
]

x0 = np.array([0.5, 0.5])
result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                  options={'disp': True})

x_opt = result.x
print("KKT 조건 예제")
print(f"최적 해: x* = {x_opt}")
print(f"목적 함수 값: f(x*) = {objective(x_opt):.6f}")

# 제약 조건 확인
g1 = constraint_ineq1(x_opt)
g2 = constraint_ineq2(x_opt)
g3 = constraint_ineq3(x_opt)
print(f"\n제약 조건 값:")
print(f"  g1(x*) = x + y - 1 = {g1:.6f} {'(활성)' if abs(g1) < 1e-6 else '(비활성)'}")
print(f"  g2(x*) = -x = {g2:.6f} {'(활성)' if abs(g2) < 1e-6 else '(비활성)'}")
print(f"  g3(x*) = -y = {g3:.6f} {'(활성)' if abs(g3) < 1e-6 else '(비활성)'}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 목적 함수 등고선
x = np.linspace(-0.5, 2.5, 300)
y = np.linspace(-0.5, 2.5, 300)
X, Y = np.meshgrid(x, y)
Z = (X - 2)**2 + (Y - 1)**2
contour = ax.contour(X, Y, Z, levels=15, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 실현 가능 영역 (삼각형)
feasible_x = [0, 1, 0, 0]
feasible_y = [0, 0, 1, 0]
ax.fill(feasible_x, feasible_y, alpha=0.2, color='green', label='실현 가능 영역')

# 제약 조건 경계
ax.plot([0, 1], [1, 0], 'r-', linewidth=2, label='$x + y = 1$')
ax.axvline(x=0, color='blue', linewidth=2, linestyle='--', label='$x = 0$')
ax.axhline(y=0, color='purple', linewidth=2, linestyle='--', label='$y = 0$')

# 제약 없는 최솟값
ax.plot(2, 1, 'go', markersize=12, label='제약 없는 최솟값 (2, 1)')

# KKT 최적점
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20,
        label=f'KKT 최적점 ({x_opt[0]:.2f}, {x_opt[1]:.2f})')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('KKT 조건: 부등식 제약 최적화', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('kkt_conditions.png', dpi=150)
plt.show()

# KKT 조건 수동 검증 (활성 제약이 g1인 경우)
print("\n\nKKT 조건 수동 검증:")
print("활성 제약: g1(x,y) = x + y - 1 = 0")
print("∇f(x*) + μ1∇g1(x*) = 0")
print(f"∇f(x*) = {objective_grad(x_opt)}")
print("∇g1 = [1, 1]")
mu1 = -objective_grad(x_opt)[0]  # 또는 -objective_grad(x_opt)[1]
print(f"→ μ1 = {mu1:.4f}")
print(f"상보 이완: μ1 * g1(x*) = {mu1 * g1:.6e} ≈ 0 ✓")
```

## 6. 머신러닝에서의 최적화 응용

### 6.1 Support Vector Machine (SVM)의 쌍대 문제

**원시 문제 (Primal):**

$$
\begin{align}
\min_{w, b, \xi} \quad & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^T x_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0
\end{align}
$$

라그랑지안을 구성하고 KKT 조건을 적용하면 **쌍대 문제 (Dual)**를 얻습니다:

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{subject to} \quad & 0 \leq \alpha_i \leq C \\
& \sum_{i=1}^n \alpha_i y_i = 0
\end{align}
$$

쌍대 문제의 장점:
- 커널 트릭 적용 가능
- 차원이 데이터 개수에 의존 (고차원 데이터에 유리)

```python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2,
                                     n_redundant=0, n_clusters_per_class=1,
                                     random_state=42)
y = 2*y - 1  # {0,1} → {-1,1}

# 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# SVM 학습
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# 결정 경계 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 데이터 포인트
ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', s=100,
           edgecolors='k', label='Class +1')
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='red', marker='s', s=100,
           edgecolors='k', label='Class -1')

# 결정 경계
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
XX, YY = np.meshgrid(xx, yy)
Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

# 결정 경계와 마진
ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
           linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

# 서포트 벡터 강조
support_vectors = svm.support_vectors_
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=300,
           linewidths=2, facecolors='none', edgecolors='green',
           label='Support Vectors')

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('SVM: KKT 조건으로 유도된 쌍대 문제', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svm_kkt_dual.png', dpi=150)
plt.show()

print(f"서포트 벡터 개수: {len(support_vectors)}")
print(f"쌍대 계수 (α): {svm.dual_coef_}")
print("\nKKT 상보 이완:")
print("  α = 0: 마진 바깥 (비활성 제약)")
print("  0 < α < C: 마진 위 (서포트 벡터)")
print("  α = C: 마진 안쪽 (오분류 가능)")
```

### 6.2 정규화 = 제약 최적화

**Ridge 회귀의 두 가지 표현:**

**정규화 형태:**
$$\min_w \|Xw - y\|^2 + \lambda\|w\|^2$$

**제약 형태:**
$$
\begin{align}
\min_w \quad & \|Xw - y\|^2 \\
\text{subject to} \quad & \|w\|^2 \leq t
\end{align}
$$

이 두 문제는 적절한 $\lambda$와 $t$의 대응으로 동등합니다 (라그랑주 쌍대성).

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
true_w[20:] = 0  # 희소성
y = X @ true_w + 0.1 * np.random.randn(n_samples)

# Ridge 회귀 (여러 λ 값)
lambdas = np.logspace(-3, 3, 100)
coefs = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

# 시각화: 정규화 경로 (Regularization Path)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: 계수 경로
for i in range(min(10, n_features)):
    axes[0].plot(np.log10(lambdas), coefs[:, i], linewidth=2)
axes[0].set_xlabel('$\log_{10}(\lambda)$', fontsize=12)
axes[0].set_ylabel('계수 값', fontsize=12)
axes[0].set_title('Ridge 정규화 경로', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0, color='r', linestyle='--', label='$\lambda=1$')
axes[0].legend()

# 오른쪽: L2 노름
l2_norms = np.linalg.norm(coefs, axis=1)
axes[1].plot(np.log10(lambdas), l2_norms, linewidth=3, color='purple')
axes[1].set_xlabel('$\log_{10}(\lambda)$', fontsize=12)
axes[1].set_ylabel('$\|w\|_2$', fontsize=12)
axes[1].set_title('계수의 L2 노름 vs 정규화 강도', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='r', linestyle='--', label='$\lambda=1$')
axes[1].legend()

plt.tight_layout()
plt.savefig('ridge_regularization_path.png', dpi=150)
plt.show()

print("정규화와 제약 최적화의 관계:")
print(f"  λ ↑ → 제약 t ↓ → ||w|| ↓")
print(f"  λ=0: 제약 없음 (OLS)")
print(f"  λ→∞: w → 0")
```

### 6.3 볼록 최적화 vs 비볼록 최적화

**볼록 최적화 (Convex):**
- 선형/로지스틱 회귀, SVM, Lasso
- 전역 최솟값 보장
- 효율적인 알고리즘 존재

**비볼록 최적화 (Non-convex):**
- 신경망 (다층 퍼셉트론, CNN, Transformer)
- 지역 최솟값, 안장점 존재
- 휴리스틱: SGD + 모멘텀, 적응적 학습률
- 오버파라미터화 → 좋은 지역 최솟값이 많음

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 간단한 신경망: 비볼록 손실 지형
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]))
        self.w2 = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w1 * x + self.w2 * x**2

# 손실 함수 (비볼록 지형 시뮬레이션)
def loss_landscape(w1, w2):
    """인위적인 비볼록 손실 지형"""
    return (w1**2 + w2**2) * (1 + 0.3*np.sin(5*w1) * np.sin(5*w2))

# 시각화
w1_vals = np.linspace(-2, 2, 200)
w2_vals = np.linspace(-2, 2, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = loss_landscape(W1, W2)

fig = plt.figure(figsize=(16, 6))

# 3D 표면
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('$w_1$', fontsize=12)
ax1.set_ylabel('$w_2$', fontsize=12)
ax1.set_zlabel('Loss', fontsize=12)
ax1.set_title('신경망 손실 지형 (비볼록)', fontsize=14)

# 등고선
ax2 = fig.add_subplot(122)
contour = ax2.contourf(W1, W2, Loss, levels=30, cmap='viridis')
plt.colorbar(contour, ax=ax2)
ax2.set_xlabel('$w_1$', fontsize=12)
ax2.set_ylabel('$w_2$', fontsize=12)
ax2.set_title('손실 등고선: 여러 지역 최솟값', fontsize=14)

# 지역 최솟값 위치 (근사)
local_minima = [(-1.2, -1.2), (0, 0), (1.2, 1.2)]
for x, y in local_minima:
    ax2.plot(x, y, 'r*', markersize=15)

plt.tight_layout()
plt.savefig('nonconvex_loss_landscape.png', dpi=150)
plt.show()

print("신경망 최적화의 특징:")
print("  - 비볼록 손실 지형")
print("  - 여러 지역 최솟값과 안장점")
print("  - SGD + 모멘텀으로 안장점 탈출")
print("  - 오버파라미터화: 좋은 최솟값이 많음")
```

## 7. 최적화 알고리즘 구현

### 7.1 scipy.optimize 활용

```python
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np

# 예제: Portfolio Optimization
# 목적: 위험 최소화 (분산)
# 제약: 기대 수익률 달성, 투자 비율 합 = 1

np.random.seed(42)
n_assets = 5
returns = np.random.randn(n_assets) * 0.1 + 0.05  # 기대 수익률
cov_matrix = np.random.randn(n_assets, n_assets)
cov_matrix = cov_matrix @ cov_matrix.T  # 양정치 공분산

def portfolio_variance(weights):
    """포트폴리오 분산 (목적 함수)"""
    return weights @ cov_matrix @ weights

def portfolio_return(weights):
    """포트폴리오 기대 수익률"""
    return weights @ returns

# 제약 조건
target_return = 0.06
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 합 = 1
    {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}  # 수익률
]

# 경계: 0 <= w_i <= 1 (공매도 금지)
bounds = [(0, 1) for _ in range(n_assets)]

# 초기값
w0 = np.ones(n_assets) / n_assets

# 최적화
result = minimize(portfolio_variance, w0, method='SLSQP',
                  bounds=bounds, constraints=constraints)

optimal_weights = result.x

print("Portfolio Optimization (KKT 조건)")
print(f"최적 투자 비율: {optimal_weights}")
print(f"포트폴리오 분산: {portfolio_variance(optimal_weights):.6f}")
print(f"포트폴리오 수익률: {portfolio_return(optimal_weights):.6f}")
print(f"목표 수익률: {target_return}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
assets = [f'Asset {i+1}' for i in range(n_assets)]
bars = ax.bar(assets, optimal_weights, color='skyblue', edgecolor='black', linewidth=1.5)

# 값 표시
for bar, weight in zip(bars, optimal_weights):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{weight:.3f}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('투자 비율', fontsize=12)
ax.set_title(f'최적 포트폴리오 (목표 수익률 {target_return:.1%})', fontsize=14)
ax.set_ylim(0, max(optimal_weights) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=150)
plt.show()
```

## 연습 문제

1. **볼록성 판별**: 다음 함수들이 볼록인지 판별하고, 헤시안을 계산하여 증명하시오.
   - (a) $f(x, y) = e^x + e^y$
   - (b) $f(x, y) = x^2 - xy + y^2$
   - (c) $f(x, y) = \log(e^x + e^y)$

2. **라그랑주 승수법**: 다음 최적화 문제를 라그랑주 승수법으로 해결하시오.
   $$
   \begin{align}
   \min_{x, y} \quad & x^2 + 2y^2 \\
   \text{subject to} \quad & x + y = 1
   \end{align}
   $$
   해석적 해를 구하고, Python으로 검증하시오.

3. **KKT 조건 적용**: 다음 문제의 KKT 조건을 작성하고, 최적해를 찾으시오.
   $$
   \begin{align}
   \min_{x, y} \quad & (x-3)^2 + (y-2)^2 \\
   \text{subject to} \quad & x + 2y \leq 4 \\
   & x \geq 0, \; y \geq 0
   \end{align}
   $$
   어떤 제약이 활성화되는지 분석하시오.

4. **SVM 쌍대 문제**: 선형 분리 가능한 2차원 데이터셋을 생성하고, SVM의 쌍대 문제를 직접 구현하여 해결하시오. `cvxopt` 라이브러리를 사용하여 이차 계획법(QP) 문제를 풀고, 서포트 벡터를 찾으시오.

5. **비볼록 최적화**: Rosenbrock 함수에 대해 여러 초기점에서 경사 하강법을 실행하고, 수렴하는 지역 최솟값을 비교하시오. 모멘텀을 추가했을 때 수렴 속도가 어떻게 변하는지 분석하시오.

## 참고 자료

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
  - 볼록 최적화의 바이블, 온라인 무료 제공
- Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer.
  - 수치 최적화 알고리즘의 이론과 구현
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
  - Chapter 4: Numerical Computation (최적화 기초)
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning". *SIAM Review*.
  - 머신러닝 최적화의 현대적 관점
- SciPy Optimize Documentation: https://docs.scipy.org/doc/scipy/reference/optimize.html
- CVX Forum: https://ask.cvxr.com/
  - 볼록 최적화 질문과 답변

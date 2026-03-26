# 14. 볼록성과 쌍대성

## 학습 목표

- 볼록 집합과 볼록 함수의 심화된 성질을 이해한다
- 라그랑주 쌍대성의 원리와 약/강 쌍대성을 파악한다
- SVM의 쌍대 문제 유도 과정을 완전히 이해하고 커널 트릭과의 연결을 학습한다
- 펜첼 켤레(Fenchel Conjugate)의 개념과 응용을 습득한다
- 근위 연산자와 근위 경사법을 이해하고 LASSO에 적용한다
- 머신러닝에서 볼록 최적화와 쌍대성의 실제 활용 사례를 학습한다

---

## 1. 볼록 최적화 복습 및 심화

### 1.1 볼록 집합

집합 $C$가 **볼록(convex)**하다:
$$
\forall x, y \in C, \forall \theta \in [0, 1]: \quad \theta x + (1-\theta) y \in C
$$

**예시**:
- 초평면: $\{x : a^T x = b\}$
- 반공간: $\{x : a^T x \leq b\}$
- 노름 볼: $\{x : \|x\| \leq r\}$
- 양의 준정부호 행렬: $\mathbb{S}_+^n$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 볼록 집합 vs 비볼록 집합 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 볼록 집합 (다각형)
convex_points = np.array([[0, 0], [2, 0], [2.5, 1], [1.5, 2], [0.5, 1.5]])
axes[0].add_patch(Polygon(convex_points, fill=True, alpha=0.3, color='blue'))
axes[0].plot(convex_points[:, 0], convex_points[:, 1], 'bo-')

# 두 점 연결
p1, p2 = convex_points[0], convex_points[3]
axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='선분')
axes[0].scatter(*p1, color='red', s=100, zorder=5)
axes[0].scatter(*p2, color='red', s=100, zorder=5)
axes[0].set_title('볼록 집합: 두 점 사이 선분이 집합 내부')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# 비볼록 집합 (초승달 모양)
theta = np.linspace(0, 2*np.pi, 100)
outer = np.column_stack([np.cos(theta), np.sin(theta)])
inner = np.column_stack([0.5*np.cos(theta) + 0.3, 0.5*np.sin(theta)])
crescent = np.vstack([outer[:50], inner[50:0:-1]])
axes[1].add_patch(Polygon(crescent, fill=True, alpha=0.3, color='orange'))

# 볼록성 위반 보이기
p1, p2 = crescent[10], crescent[-10]
axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='선분 (외부 지나감)')
axes[1].scatter(*p1, color='red', s=100, zorder=5)
axes[1].scatter(*p2, color='red', s=100, zorder=5)
axes[1].set_title('비볼록 집합: 선분이 집합 밖으로')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()
```

### 1.2 볼록 함수

함수 $f: \mathbb{R}^n \to \mathbb{R}$가 **볼록**하다:
$$
f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y), \quad \forall x, y, \forall \theta \in [0, 1]
$$

**1차 조건** (미분 가능할 때):
$$
f(y) \geq f(x) + \nabla f(x)^T (y - x)
$$
(함수가 항상 접선 위에)

**2차 조건** (2차 미분 가능할 때):
$$
\nabla^2 f(x) \succeq 0 \quad \text{(헤시안이 양의 준정부호)}
$$

```python
# 볼록 함수 vs 비볼록 함수
x = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 볼록 함수: f(x) = x^2
axes[0, 0].plot(x, x**2, 'b-', linewidth=2, label='$f(x) = x^2$')
x0 = 1.0
axes[0, 0].plot(x, x0**2 + 2*x0*(x - x0), 'r--', label=f'접선 (x={x0})')
axes[0, 0].scatter([x0], [x0**2], color='red', s=100, zorder=5)
axes[0, 0].set_title('볼록 함수: 함수가 접선 위에')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 비볼록 함수: f(x) = x^3
axes[0, 1].plot(x, x**3, 'b-', linewidth=2, label='$f(x) = x^3$')
x0 = -1.0
axes[0, 1].plot(x, x0**3 + 3*x0**2*(x - x0), 'r--', label=f'접선 (x={x0})')
axes[0, 1].scatter([x0], [x0**3], color='red', s=100, zorder=5)
axes[0, 1].set_title('비볼록 함수: 함수가 접선 아래로')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 강볼록 함수: f(x) = x^2 + 2x + 1
axes[1, 0].plot(x, x**2 + 2*x + 1, 'b-', linewidth=2, label='$f(x) = x^2 + 2x + 1$')
minimum = -1  # f'(x) = 2x + 2 = 0
axes[1, 0].scatter([minimum], [minimum**2 + 2*minimum + 1], color='green', s=100,
                   zorder=5, label='전역 최솟값')
axes[1, 0].set_title('강볼록 함수: 유일한 전역 최솟값')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 헤시안 확인
def hessian_demo():
    """2변량 함수의 헤시안"""
    from mpl_toolkits.mplot3d import Axes3D

    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)

    # 볼록: f(x1, x2) = x1^2 + x2^2
    Z = X1**2 + X2**2

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax.set_title('2변량 볼록 함수: $f(x_1, x_2) = x_1^2 + x_2^2$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')

hessian_demo()
plt.tight_layout()
plt.show()

# 헤시안 양의 정부호 확인
def check_convexity(f, grad, hessian, x):
    """2차 조건으로 볼록성 확인"""
    H = hessian(x)
    eigenvalues = np.linalg.eigvals(H)
    print(f"점 {x}에서:")
    print(f"  헤시안 고유값: {eigenvalues}")
    print(f"  모두 >= 0? {np.all(eigenvalues >= -1e-10)}")
    return np.all(eigenvalues >= -1e-10)

# 예: f(x) = x^T A x (A가 양의 정부호)
A = np.array([[2, 0], [0, 3]])
hessian = lambda x: 2 * A
x_test = np.array([1.0, 1.0])
is_convex = check_convexity(None, None, hessian, x_test)
print(f"\n함수 x^T A x는 볼록: {is_convex}")
```

### 1.3 에피그래프 (Epigraph) 관점

함수 $f$의 **에피그래프**:
$$
\text{epi}(f) = \{(x, t) : f(x) \leq t\}
$$

**정리**: $f$가 볼록 $\Leftrightarrow$ $\text{epi}(f)$가 볼록 집합

이 관점은 볼록 함수를 기하학적으로 이해하는 데 유용합니다.

### 1.4 볼록 최적화의 중요성

**핵심 성질**: 지역 최솟값 = 전역 최솟값

```python
from scipy.optimize import minimize

# 볼록 함수: 어디서 시작해도 같은 최솟값
def convex_func(x):
    return x[0]**2 + x[1]**2 + 2*x[0] + 3*x[1]

starting_points = [
    np.array([10.0, 10.0]),
    np.array([-5.0, 5.0]),
    np.array([0.0, -8.0])
]

print("볼록 함수 최적화:")
for i, x0 in enumerate(starting_points):
    result = minimize(convex_func, x0, method='BFGS')
    print(f"시작점 {i+1} {x0}: 최솟값 위치 {result.x}, 값 {result.fun:.4f}")

# 비볼록 함수: 시작점에 따라 다른 결과
def nonconvex_func(x):
    return np.sin(x[0]) + np.cos(x[1]) + 0.1*(x[0]**2 + x[1]**2)

print("\n비볼록 함수 최적화:")
for i, x0 in enumerate(starting_points):
    result = minimize(nonconvex_func, x0, method='BFGS')
    print(f"시작점 {i+1} {x0}: 최솟값 위치 {result.x}, 값 {result.fun:.4f}")
```

---

## 2. 라그랑주 쌍대성 (Lagrangian Duality)

### 2.1 원초 문제 (Primal Problem)

**일반적인 최적화 문제**:
$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}
$$

- $f$: 목적 함수
- $g_i$: 부등식 제약
- $h_j$: 등식 제약

### 2.2 라그랑지안 (Lagrangian)

**라그랑지안 함수**:
$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)
$$

- $\lambda_i \geq 0$: 부등식 제약의 라그랑주 승수
- $\nu_j$: 등식 제약의 라그랑주 승수

```python
# 간단한 예제: min x^2 + y^2 s.t. x + y = 1
def primal_objective(x):
    return x[0]**2 + x[1]**2

def equality_constraint(x):
    return x[0] + x[1] - 1

def lagrangian(x, nu):
    """라그랑지안: L(x, ν) = x^2 + y^2 + ν(x + y - 1)"""
    return primal_objective(x) + nu * equality_constraint(x)

# 시각화
x_range = np.linspace(-1, 2, 100)
y_range = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + Y**2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, alpha=0.6)
plt.plot(x_range, 1 - x_range, 'r-', linewidth=3, label='제약: x + y = 1')

# 최적해 (라그랑주 조건으로부터)
# ∇_x L = 2x + ν = 0, ∇_y L = 2y + ν = 0, x + y = 1
# => x = y = 1/2
optimal_x = np.array([0.5, 0.5])
plt.scatter(*optimal_x, color='green', s=200, zorder=5, label=f'최적해 {optimal_x}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('제약 최적화: 라그랑주 승수법')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"최적해: x = {optimal_x}")
print(f"최적값: f(x*) = {primal_objective(optimal_x):.4f}")
```

### 2.3 쌍대 함수 (Dual Function)

**쌍대 함수**:
$$
g(\lambda, \nu) = \inf_{x} \mathcal{L}(x, \lambda, \nu)
$$

**핵심 성질**: $g(\lambda, \nu)$는 항상 **볼록 함수**입니다 (원초 문제가 볼록이 아니어도!)

### 2.4 약 쌍대성 (Weak Duality)

**정리**: 임의의 $\lambda \geq 0, \nu$에 대해
$$
g(\lambda, \nu) \leq p^*
$$
여기서 $p^*$는 원초 문제의 최적값입니다.

**증명**:
$$
g(\lambda, \nu) = \inf_{x} \mathcal{L}(x, \lambda, \nu) \leq \mathcal{L}(x^*, \lambda, \nu) = f(x^*) + \sum \lambda_i \underbrace{g_i(x^*)}_{\leq 0} + \sum \nu_j \underbrace{h_j(x^*)}_{=0} \leq f(x^*) = p^*
$$

### 2.5 강 쌍대성 (Strong Duality)

**쌍대 문제**:
$$
\max_{\lambda \geq 0, \nu} \quad g(\lambda, \nu)
$$

**강 쌍대성**: $p^* = d^*$ (최적값이 같음)

**Slater 조건** (충분 조건):
- 원초 문제가 볼록
- 실행 가능한 $x$가 존재하여 모든 $g_i(x) < 0$ (엄격한 부등식)

```python
# 강 쌍대성 예제: 2차 계획법 (QP)
from scipy.optimize import minimize
import cvxpy as cp

# 원초 문제: min 1/2 x^T Q x + c^T x, s.t. Ax <= b
Q = np.array([[2, 0], [0, 2]])
c = np.array([1, 1])
A = np.array([[1, 1], [-1, 0], [0, -1]])
b = np.array([1, 0, 0])

# CVXPY로 원초 문제 풀기
x_primal = cp.Variable(2)
objective_primal = cp.Minimize(0.5 * cp.quad_form(x_primal, Q) + c @ x_primal)
constraints_primal = [A @ x_primal <= b]
prob_primal = cp.Problem(objective_primal, constraints_primal)
p_star = prob_primal.solve()

print(f"원초 문제:")
print(f"  최적값: p* = {p_star:.4f}")
print(f"  최적해: x* = {x_primal.value}")

# 쌍대 문제 (유도 생략, 이론적으로)
# max -1/2 λ^T (A Q^{-1} A^T) λ - b^T λ - 1/2 c^T Q^{-1} c, s.t. λ >= 0, A^T λ = -c
lambda_dual = cp.Variable(3)
Q_inv = np.linalg.inv(Q)
objective_dual = cp.Maximize(
    -0.5 * cp.quad_form(lambda_dual, A @ Q_inv @ A.T) - b @ lambda_dual - 0.5 * c @ Q_inv @ c
)
constraints_dual = [lambda_dual >= 0, A.T @ lambda_dual == -c]
prob_dual = cp.Problem(objective_dual, constraints_dual)
d_star = prob_dual.solve()

print(f"\n쌍대 문제:")
print(f"  최적값: d* = {d_star:.4f}")
print(f"  최적 승수: λ* = {lambda_dual.value}")

print(f"\n강 쌍대성 확인: p* - d* = {p_star - d_star:.6f} ≈ 0")
```

---

## 3. SVM의 쌍대 문제

### 3.1 원초 문제

**하드 마진 SVM**:
$$
\begin{align}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n
\end{align}
$$

목표: 마진 $\frac{2}{\|w\|}$을 최대화 $\Leftrightarrow$ $\|w\|^2$를 최소화

### 3.2 라그랑지안 구성

$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1]
$$

여기서 $\alpha_i \geq 0$은 라그랑주 승수입니다.

### 3.3 KKT 조건

**1. 정상성(Stationarity)**:
$$
\nabla_w \mathcal{L} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^{n} \alpha_i y_i x_i
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{n} \alpha_i y_i = 0
$$

**2. 상보 슬랙성(Complementary Slackness)**:
$$
\alpha_i [y_i(w^T x_i + b) - 1] = 0
$$

**3. 원초/쌍대 실행 가능성**: $y_i(w^T x_i + b) \geq 1$, $\alpha_i \geq 0$

### 3.4 쌍대 문제 유도

$w = \sum \alpha_i y_i x_i$를 라그랑지안에 대입:

$$
\begin{align}
\mathcal{L}(w, b, \alpha) &= \frac{1}{2} \left\|\sum_{i} \alpha_i y_i x_i\right\|^2 - \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j) - b\sum_i \alpha_i y_i + \sum_i \alpha_i \\
&= -\frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j) + \sum_i \alpha_i
\end{align}
$$

**쌍대 문제**:
$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & \sum_{i=1}^{n} \alpha_i y_i = 0 \\
& \alpha_i \geq 0, \quad i = 1, \ldots, n
\end{align}
$$

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 간단한 2D 데이터
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, flip_y=0.1,
                           class_sep=1.5, random_state=42)
y = 2*y - 1  # {0, 1} -> {-1, 1}

# SVM 훈련 (선형 커널)
svm = SVC(kernel='linear', C=1000)  # C가 크면 하드 마진에 근접
svm.fit(X, y)

# 결과
w = svm.coef_[0]
b = svm.intercept_[0]
support_vectors = svm.support_vectors_
alphas = svm.dual_coef_[0]  # α_i * y_i

print("SVM 결과:")
print(f"  가중치 w: {w}")
print(f"  절편 b: {b}")
print(f"  서포트 벡터 수: {len(support_vectors)}")
print(f"  쌍대 계수 (α_i * y_i): {alphas[:5]}...")  # 처음 5개만

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', label='클래스 +1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='red', marker='s', label='클래스 -1')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200, linewidth=2,
            facecolors='none', edgecolors='green', label='서포트 벡터')

# 결정 경계
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_plot = np.linspace(x_min, x_max, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='결정 경계')

# 마진
margin = 1 / np.linalg.norm(w)
y_plot_margin_up = y_plot + margin * np.sqrt(1 + (w[0]/w[1])**2)
y_plot_margin_down = y_plot - margin * np.sqrt(1 + (w[0]/w[1])**2)
plt.plot(x_plot, y_plot_margin_up, 'k--', linewidth=1, alpha=0.5)
plt.plot(x_plot, y_plot_margin_down, 'k--', linewidth=1, alpha=0.5)
plt.fill_between(x_plot, y_plot_margin_down, y_plot_margin_up, alpha=0.1, color='gray')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'SVM: 마진 = {2*margin:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.5 커널 트릭의 자연스러운 등장

쌍대 문제에서 데이터는 **내적 $x_i^T x_j$로만 나타남**:

$$
\max_{\alpha} \quad \sum_{i} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \underbrace{(x_i^T x_j)}_{k(x_i, x_j)}
$$

**커널 트릭**: $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ (고차원 특성 공간)

**예**: 다항식 커널 $k(x, x') = (x^T x' + 1)^2$

```python
# 비선형 SVM (RBF 커널)
from sklearn.datasets import make_moons

X_nonlinear, y_nonlinear = make_moons(n_samples=200, noise=0.15, random_state=42)
y_nonlinear = 2*y_nonlinear - 1

svm_rbf = SVC(kernel='rbf', gamma=2, C=1.0)
svm_rbf.fit(X_nonlinear, y_nonlinear)

# 결정 경계 시각화
h = 0.02
x_min, x_max = X_nonlinear[:, 0].min() - 0.5, X_nonlinear[:, 0].max() + 0.5
y_min, y_max = X_nonlinear[:, 1].min() - 0.5, X_nonlinear[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1],
            c='blue', marker='o', edgecolors='k', label='클래스 +1')
plt.scatter(X_nonlinear[y_nonlinear==-1, 0], X_nonlinear[y_nonlinear==-1, 1],
            c='red', marker='s', edgecolors='k', label='클래스 -1')
plt.scatter(svm_rbf.support_vectors_[:, 0], svm_rbf.support_vectors_[:, 1],
            s=200, linewidth=2, facecolors='none', edgecolors='green', label='서포트 벡터')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('비선형 SVM (RBF 커널)')
plt.legend()
plt.show()

print(f"서포트 벡터 수: {len(svm_rbf.support_vectors_)}")
```

---

## 4. 펜첼 켤레 (Fenchel Conjugate)

### 4.1 정의

함수 $f: \mathbb{R}^n \to \mathbb{R}$의 **펜첼 켤레**:
$$
f^*(y) = \sup_{x} \left( y^T x - f(x) \right)
$$

**직관**: $y$는 기울기, $f^*(y)$는 기울기 $y$인 접선의 $x=0$에서의 절편

### 4.2 기하학적 해석

```python
# 펜첼 켤레 시각화
def fenchel_conjugate_demo():
    x = np.linspace(-3, 3, 200)
    f_x = x**2  # f(x) = x^2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 원 함수
    axes[0].plot(x, f_x, 'b-', linewidth=2, label='$f(x) = x^2$')

    # 여러 기울기의 접선
    slopes = [-2, 0, 2, 4]
    colors = ['red', 'green', 'orange', 'purple']

    for y, color in zip(slopes, colors):
        # f*(y) = sup_x (yx - x^2)
        # 최대화: d/dx (yx - x^2) = y - 2x = 0 => x* = y/2
        x_star = y / 2
        f_star_y = y * x_star - x_star**2  # = y^2/4

        # 접선: yx - f*(y)
        tangent = y * x - f_star_y

        axes[0].plot(x, tangent, color=color, linestyle='--', alpha=0.7,
                     label=f'y={y}: $f^*({y})={f_star_y:.2f}$')
        axes[0].scatter([x_star], [f_x[np.abs(x - x_star).argmin()]],
                        color=color, s=100, zorder=5)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('펜첼 켤레: 접선 관점')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-2, 10)

    # 켤레 함수
    y_range = np.linspace(-4, 4, 100)
    f_star = y_range**2 / 4  # f*(y) = y^2/4 for f(x) = x^2

    axes[1].plot(y_range, f_star, 'b-', linewidth=2, label='$f^*(y) = y^2/4$')
    axes[1].scatter(slopes, [s**2/4 for s in slopes], color=colors, s=100, zorder=5)
    axes[1].set_xlabel('y (기울기)')
    axes[1].set_ylabel('$f^*(y)$')
    axes[1].set_title('펜첼 켤레 함수')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

fenchel_conjugate_demo()
```

### 4.3 중요한 예시

**1. 노름의 켤레**: $f(x) = \frac{1}{2}\|x\|_2^2$
$$
f^*(y) = \frac{1}{2}\|y\|_2^2
$$

**2. 지시 함수의 켤레**: $f(x) = I_C(x)$ (집합 $C$의 지시 함수)
$$
f^*(y) = \sup_{x \in C} y^T x \quad \text{(support function)}
$$

**3. 로그 합의 켤레**: $f(x) = \log(\sum_i e^{x_i})$
$$
f^*(y) = \sum_i y_i \log y_i \quad \text{(negative entropy)}
$$

```python
# 펜첼 켤레 계산 예제
def conjugate_squared_norm(y):
    """f(x) = 1/2 ||x||^2의 켤레"""
    return 0.5 * np.linalg.norm(y)**2

def conjugate_indicator_ball(y, radius=1.0):
    """f(x) = I_{||x|| <= r}의 켤레 (support function)"""
    return radius * np.linalg.norm(y)

# 검증: f**(x) = f(x) (이중 켤레)
x_test = np.array([1.0, 2.0])
f_x = 0.5 * np.linalg.norm(x_test)**2

# f**를 수치적으로 계산
from scipy.optimize import minimize_scalar

def double_conjugate(x):
    """이중 켤레 f**(x) = sup_y (x^T y - f*(y))"""
    # 1차원으로 단순화 (예시)
    def negative_objective(y_norm):
        y = y_norm * x / np.linalg.norm(x)  # 방향 고정
        return -(np.dot(x, y) - conjugate_squared_norm(y))

    result = minimize_scalar(negative_objective, bounds=(0, 10), method='bounded')
    return -result.fun

f_double_star_x = double_conjugate(x_test)
print(f"이중 켤레 정리 검증:")
print(f"  f(x) = {f_x:.4f}")
print(f"  f**(x) = {f_double_star_x:.4f}")
print(f"  일치: {np.isclose(f_x, f_double_star_x)}")
```

---

## 5. 근위 연산자 (Proximal Operator)

### 5.1 정의

함수 $f$의 **근위 연산자**:
$$
\text{prox}_f(v) = \arg\min_{x} \left( f(x) + \frac{1}{2}\|x - v\|^2 \right)
$$

**직관**: $v$에 가까우면서 $f$를 작게 만드는 $x$ 찾기

### 5.2 L1 노름의 근위 연산자 (소프트 임계값)

$f(x) = \lambda \|x\|_1$일 때:
$$
[\text{prox}_{\lambda \|\cdot\|_1}(v)]_i = \text{sign}(v_i) \max(|v_i| - \lambda, 0)
$$

이를 **소프트 임계값(soft thresholding)**이라 합니다.

```python
def soft_threshold(v, lambda_param):
    """L1 근위 연산자 (소프트 임계값)"""
    return np.sign(v) * np.maximum(np.abs(v) - lambda_param, 0)

# 시각화
v = np.linspace(-3, 3, 200)
lambda_values = [0.5, 1.0, 1.5]

plt.figure(figsize=(10, 6))
plt.plot(v, v, 'k--', label='항등 함수 (λ=0)', alpha=0.5)

for lam in lambda_values:
    prox_v = soft_threshold(v, lam)
    plt.plot(v, prox_v, linewidth=2, label=f'λ={lam}')

plt.xlabel('v')
plt.ylabel('prox(v)')
plt.title('L1 근위 연산자 (소프트 임계값)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
```

### 5.3 근위 경사 하강법 (Proximal Gradient Descent)

**문제**: $\min f(x) + g(x)$, 여기서 $f$는 매끄럽고 $g$는 비매끄러움 (예: L1)

**알고리즘**:
$$
x_{k+1} = \text{prox}_{\alpha g}(x_k - \alpha \nabla f(x_k))
$$

```python
# LASSO 문제: min 1/2 ||Ax - b||^2 + λ||x||_1
def lasso_proximal_gradient(A, b, lambda_param, max_iter=1000, alpha=0.01):
    """근위 경사 하강법으로 LASSO 풀기"""
    n = A.shape[1]
    x = np.zeros(n)
    losses = []

    for k in range(max_iter):
        # 매끄러운 부분의 그래디언트: ∇(1/2 ||Ax - b||^2) = A^T(Ax - b)
        residual = A @ x - b
        grad_f = A.T @ residual

        # 경사 하강
        x_temp = x - alpha * grad_f

        # 근위 연산 (L1)
        x = soft_threshold(x_temp, alpha * lambda_param)

        # 손실 계산
        loss = 0.5 * np.sum(residual**2) + lambda_param * np.sum(np.abs(x))
        losses.append(loss)

        if k > 0 and abs(losses[-1] - losses[-2]) < 1e-6:
            break

    return x, losses

# 테스트: 희소 회귀
np.random.seed(42)
n, p = 100, 50
A = np.random.randn(n, p)
x_true = np.zeros(p)
x_true[:10] = np.random.randn(10)  # 10개만 0이 아님
b = A @ x_true + 0.1 * np.random.randn(n)

lambda_param = 0.1
x_lasso, losses = lasso_proximal_gradient(A, b, lambda_param, max_iter=1000, alpha=0.001)

# 결과
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 손실 함수
axes[0].plot(losses)
axes[0].set_xlabel('반복')
axes[0].set_ylabel('손실')
axes[0].set_title('LASSO 근위 경사 하강법: 수렴')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# 계수 비교
axes[1].stem(x_true, linefmt='b-', markerfmt='bo', basefmt=' ', label='실제 계수')
axes[1].stem(x_lasso, linefmt='r-', markerfmt='rs', basefmt=' ', label='추정 계수 (LASSO)')
axes[1].set_xlabel('계수 인덱스')
axes[1].set_ylabel('값')
axes[1].set_title(f'LASSO 결과 (λ={lambda_param})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"0이 아닌 계수 수:")
print(f"  실제: {np.sum(x_true != 0)}")
print(f"  추정: {np.sum(np.abs(x_lasso) > 1e-3)}")
```

---

## 6. 머신러닝 응용

### 6.1 로지스틱 회귀의 쌍대 해석

로지스틱 회귀:
$$
\min_{w} \sum_{i=1}^{n} \log(1 + e^{-y_i w^T x_i}) + \frac{\lambda}{2} \|w\|^2
$$

쌍대 문제는 **엔트로피 최대화**와 연결됩니다.

### 6.2 ADMM (Alternating Direction Method of Multipliers)

**문제**: $\min f(x) + g(z)$ s.t. $Ax + Bz = c$

**ADMM 알고리즘**:
1. $x_{k+1} = \arg\min_x \mathcal{L}_\rho(x, z_k, \lambda_k)$
2. $z_{k+1} = \arg\min_z \mathcal{L}_\rho(x_{k+1}, z, \lambda_k)$
3. $\lambda_{k+1} = \lambda_k + \rho(Ax_{k+1} + Bz_{k+1} - c)$

여기서 $\mathcal{L}_\rho$는 증강 라그랑지안입니다.

```python
# ADMM으로 LASSO 풀기
def lasso_admm(A, b, lambda_param, rho=1.0, max_iter=100):
    """ADMM으로 LASSO: min 1/2||Ax-b||^2 + λ||z||_1, s.t. x = z"""
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)  # 스케일된 쌍대 변수

    # 사전 계산
    AtA = A.T @ A
    Atb = A.T @ b
    I = np.eye(p)

    losses = []

    for k in range(max_iter):
        # x 업데이트: (A^T A + ρI)x = A^T b + ρ(z - u)
        x = np.linalg.solve(AtA + rho * I, Atb + rho * (z - u))

        # z 업데이트: soft thresholding
        z_old = z.copy()
        z = soft_threshold(x + u, lambda_param / rho)

        # u 업데이트 (쌍대 변수)
        u = u + x - z

        # 손실
        loss = 0.5 * np.linalg.norm(A @ x - b)**2 + lambda_param * np.linalg.norm(z, 1)
        losses.append(loss)

        # 수렴 확인
        if np.linalg.norm(z - z_old) < 1e-4:
            break

    return x, z, losses

# 비교: Proximal Gradient vs ADMM
x_admm, z_admm, losses_admm = lasso_admm(A, b, lambda_param, rho=1.0, max_iter=100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='근위 경사 하강법')
plt.plot(losses_admm, label='ADMM')
plt.xlabel('반복')
plt.ylabel('손실')
plt.title('수렴 속도 비교')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.stem(x_lasso, linefmt='b-', markerfmt='bo', basefmt=' ', label='Proximal GD')
plt.stem(z_admm, linefmt='r-', markerfmt='rs', basefmt=' ', label='ADMM')
plt.xlabel('계수 인덱스')
plt.ylabel('값')
plt.title('최종 계수 비교')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"계수 차이 (L2): {np.linalg.norm(x_lasso - z_admm):.6f}")
```

### 6.3 실제 응용 사례

- **SVM**: 대규모 분류 문제에서 쌍대 문제로 효율적 풀이
- **L1 정규화**: LASSO, Elastic Net에서 근위 연산자
- **분산 최적화**: ADMM으로 대규모 데이터 병렬 처리
- **딥러닝**: Fenchel 쌍대성이 GAN, f-divergence에 활용

---

## 연습 문제

### 문제 1: 볼록성 증명
다음 함수가 볼록한지 확인하세요:
(a) $f(x) = e^x$
(b) $f(x) = -\log(x)$ (x > 0)
(c) $f(x, y) = x^2 + xy + y^2$

각각 2차 조건(헤시안)을 사용하여 증명하세요.

### 문제 2: 라그랑주 쌍대성
다음 문제의 라그랑지안과 쌍대 함수를 유도하세요:
$$
\min x^2 + y^2 \quad \text{s.t.} \quad x + y \geq 1, \quad x, y \geq 0
$$

KKT 조건을 적용하여 최적해를 구하세요.

### 문제 3: SVM 구현
처음부터 선형 SVM을 구현하세요 (cvxpy 사용 가능).
(a) 원초 문제와 쌍대 문제를 모두 풀기
(b) 두 해가 강 쌍대성을 만족하는지 확인
(c) 서포트 벡터 식별 및 시각화

### 문제 4: 펜첼 켤레
다음 함수의 펜첼 켤레를 유도하세요:
(a) $f(x) = \frac{1}{2}x^2 + x$ (1차원)
(b) $f(x) = |x|$ (L1 노름)
(c) $f(x) = \max(0, x)$ (ReLU)

### 문제 5: ADMM 응용
분산 선형 회귀 문제를 ADMM으로 풀어보세요:
$$
\min \sum_{i=1}^{N} \frac{1}{2}\|A_i x - b_i\|^2
$$

여기서 데이터가 $N$개 노드에 분산되어 있다고 가정합니다.

---

## 참고 자료

### 서적
- **Convex Optimization** (Boyd & Vandenberghe, 2004) - 표준 교재
- **Proximal Algorithms** (Parikh & Boyd, 2014) - 근위 방법의 집대성
- **Online Convex Optimization** (Hazan, 2016) - 온라인 학습과의 연결

### 논문
- Vapnik (1995), "The Nature of Statistical Learning Theory" - SVM 기초
- Boyd et al. (2011), "Distributed Optimization and Statistical Learning via ADMM"
- Rockafellar (1970), "Convex Analysis" - 펜첼 켤레 등 이론

### 온라인 자료
- [Stanford EE364a: Convex Optimization I](https://web.stanford.edu/class/ee364a/)
- [Boyd's ADMM page](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
- [CVXPY Tutorial](https://www.cvxpy.org/tutorial/index.html)

### 라이브러리
- **CVXPY**: Python 볼록 최적화
- **scikit-learn**: SVM, LASSO 등 구현
- **ProximalOperators.jl**: Julia 근위 연산자 라이브러리

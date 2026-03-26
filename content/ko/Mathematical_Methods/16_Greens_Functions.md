# 16. 그린 함수 (Green's Functions)

## 학습 목표

- **디랙 델타 함수** $\delta(x)$의 정의, 성질, 다양한 표현을 이해하고 활용할 수 있다
- 비제차 미분방정식 $L[u] = f(x)$의 해를 **그린 함수**를 이용하여 적분 형태로 표현할 수 있다
- **경계값 문제**에서 그린 함수를 직접 구성하고, 접합 조건(matching conditions)을 적용할 수 있다
- **고유함수 전개법**으로 그린 함수를 급수 형태로 나타내고, 스투름-리우빌 이론과의 연결을 이해한다
- **편미분방정식**의 그린 함수(푸아송 방정식, 열방정식, 파동방정식)를 구하고 물리적으로 해석할 수 있다
- 정전기학, 양자역학, 음향학 등 **물리적 응용**에서 그린 함수를 활용하여 실제 문제를 풀 수 있다

> **물리학에서의 중요성**: 그린 함수는 "점 소스가 만드는 응답"을 기술하는 보편적 도구이다. 일단 점 소스에 대한 응답(그린 함수)을 알면, 중첩 원리에 의해 **임의의 소스 분포**에 대한 해를 적분 하나로 구할 수 있다. 정전기학에서 점전하의 전위, 양자역학에서 전파함수(propagator), 음향학에서 점음원의 방사 등 현대 물리학의 핵심 문제들이 모두 그린 함수로 귀결된다.

---

## 1. 디랙 델타 함수

### 1.1 정의와 기본 성질

**디랙 델타 함수** $\delta(x)$는 엄밀한 의미에서 함수가 아니라 **일반화된 함수(generalized function)** 또는 **분포(distribution)**이다. 다음 두 성질로 정의한다:

$$\delta(x) = 0 \quad (x \neq 0), \qquad \int_{-\infty}^{\infty} \delta(x) \, dx = 1$$

**체(sifting) 성질**: 델타 함수의 가장 중요한 성질이다.

$$\int_{-\infty}^{\infty} f(x) \delta(x - a) \, dx = f(a)$$

이것은 $\delta(x-a)$가 $x = a$에서 함수값을 "추출"한다는 뜻이다.

**추가 성질들**:

$$\delta(-x) = \delta(x) \quad \text{(짝함수)}$$

$$\delta(ax) = \frac{1}{|a|}\delta(x) \quad (a \neq 0)$$

$$x\delta(x) = 0$$

$$\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{|g'(x_i)|} \quad (g(x_i) = 0, \; g'(x_i) \neq 0)$$

### 1.2 델타 함수의 표현

$\delta(x)$는 정칙 함수의 극한으로 표현할 수 있다:

**가우시안 표현**:
$$\delta(x) = \lim_{\epsilon \to 0} \frac{1}{\epsilon\sqrt{\pi}} e^{-x^2/\epsilon^2}$$

**로렌츠 표현**:
$$\delta(x) = \lim_{\epsilon \to 0} \frac{1}{\pi} \frac{\epsilon}{x^2 + \epsilon^2}$$

**sinc 표현 (푸리에 표현)**:
$$\delta(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{ikx} dk = \lim_{N \to \infty} \frac{\sin(Nx)}{\pi x}$$

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 가우시안 표현
for eps in [1.0, 0.5, 0.2, 0.05]:
    delta_gauss = np.exp(-x**2 / eps**2) / (eps * np.sqrt(np.pi))
    axes[0].plot(x, delta_gauss, label=f'$\\epsilon={eps}$')
axes[0].set_title('가우시안 표현')
axes[0].legend(); axes[0].set_ylim(0, 10); axes[0].grid(True, alpha=0.3)

# 로렌츠 표현
for eps in [1.0, 0.5, 0.2, 0.05]:
    delta_lorentz = (1/np.pi) * eps / (x**2 + eps**2)
    axes[1].plot(x, delta_lorentz, label=f'$\\epsilon={eps}$')
axes[1].set_title('로렌츠 표현')
axes[1].legend(); axes[1].set_ylim(0, 10); axes[1].grid(True, alpha=0.3)

# sinc 표현
for N in [5, 20, 50, 200]:
    delta_sinc = np.sin(N * x) / (np.pi * x + 1e-30)
    axes[2].plot(x, delta_sinc, label=f'$N={N}$', alpha=0.8)
axes[2].set_title('sinc 표현 (푸리에)')
axes[2].legend(); axes[2].set_ylim(-5, 70); axes[2].grid(True, alpha=0.3)

plt.suptitle('디랙 델타 함수의 다양한 표현', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

### 1.3 델타 함수의 도함수

$\delta'(x)$는 다음 적분 성질로 정의된다:

$$\int_{-\infty}^{\infty} f(x) \delta'(x-a) \, dx = -f'(a)$$

일반적으로 $n$차 도함수:

$$\int_{-\infty}^{\infty} f(x) \delta^{(n)}(x-a) \, dx = (-1)^n f^{(n)}(a)$$

### 1.4 다차원 델타 함수

3차원 디랙 델타:

$$\delta^3(\mathbf{r} - \mathbf{r}') = \delta(x - x')\delta(y - y')\delta(z - z')$$

**체 성질**: $\int f(\mathbf{r}) \delta^3(\mathbf{r} - \mathbf{r}') \, d^3r = f(\mathbf{r}')$

**구면좌표에서**: $\delta^3(\mathbf{r} - \mathbf{r}') = \frac{\delta(r-r')}{r^2} \frac{\delta(\theta-\theta')}{\sin\theta} \delta(\phi-\phi')$

중요한 관계식 (정전기학의 핵심):

$$\nabla^2 \left(\frac{1}{|\mathbf{r} - \mathbf{r}'|}\right) = -4\pi \delta^3(\mathbf{r} - \mathbf{r}')$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 체(sifting) 성질의 수치 검증
# f(x) = cos(x), a = 1.0에 대해 integral f(x) delta_eps(x-a) dx ~ f(a)
a = 1.0
f = lambda t: np.cos(t)
x = np.linspace(-10, 10, 100000)

print("=== 체(sifting) 성질 수치 검증 ===")
print(f"f(a) = cos({a}) = {f(a):.8f}")
print()

for eps in [1.0, 0.1, 0.01, 0.001]:
    # 가우시안 근사 delta 사용
    delta_approx = np.exp(-(x - a)**2 / eps**2) / (eps * np.sqrt(np.pi))
    integral = np.trapz(f(x) * delta_approx, x)
    error = abs(integral - f(a))
    print(f"  eps = {eps:.3f}: integral = {integral:.8f}, 오차 = {error:.2e}")
```

---

## 2. 그린 함수의 개념

### 2.1 비제차 미분방정식과 중첩 원리

**선형 미분 연산자** $L$에 대해 비제차 방정식:

$$L[u(x)] = f(x)$$

을 풀고자 한다. 만약 $L$이 선형이면 **중첩 원리(superposition principle)**가 성립한다: $L[u_1] = f_1$이고 $L[u_2] = f_2$이면 $L[\alpha u_1 + \beta u_2] = \alpha f_1 + \beta f_2$.

### 2.2 점 소스 응답으로서의 그린 함수

소스 $f(x)$를 델타 함수의 중첩으로 표현하면:

$$f(x) = \int f(x') \delta(x - x') \, dx'$$

**그린 함수** $G(x, x')$를 "점 소스 $\delta(x - x')$에 대한 응답"으로 정의한다:

$$L[G(x, x')] = \delta(x - x')$$

그러면 중첩 원리에 의해 원래 방정식의 해는:

$$\boxed{u(x) = \int G(x, x') f(x') \, dx'}$$

이것이 그린 함수 방법의 핵심이다. 점 소스의 응답 $G$를 한 번만 구하면, 임의의 소스 $f$에 대한 해를 적분으로 바로 얻을 수 있다.

### 2.3 물리적 직관

| 물리 시스템 | 연산자 $L$ | 소스 $f$ | 그린 함수 $G$ |
|---|---|---|---|
| 정전기학 | $\nabla^2$ | $-\rho/\epsilon_0$ | 점전하의 전위 |
| 열전도 | $\partial_t - \alpha^2\nabla^2$ | 열원 | 점열원의 온도 응답 |
| 파동 | $\partial_t^2 - c^2\nabla^2$ | 외력 | 점충격에 대한 파동 |
| 양자역학 | $i\hbar\partial_t - H$ | — | 전파함수(propagator) |

```python
import numpy as np
import matplotlib.pyplot as plt

# 개념 시연: 1D 현에 가해진 점하중의 응답
# L[u] = u'' = f(x), u(0) = u(1) = 0
# 그린 함수: G(x, x') = x'(1-x) for x > x', x(1-x') for x < x'

def greens_function_string(x, xp):
    """양 끝 고정 현의 그린 함수"""
    return np.where(x < xp, x * (1 - xp), xp * (1 - x))

x = np.linspace(0, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 다양한 점 소스 위치에서의 그린 함수
for xp in [0.2, 0.4, 0.5, 0.6, 0.8]:
    G = greens_function_string(x, xp)
    ax1.plot(x, G, linewidth=2, label=f"$x' = {xp}$")
    ax1.plot(xp, greens_function_string(xp, xp), 'ko', markersize=5)

ax1.set_xlabel('$x$'); ax1.set_ylabel("$G(x, x')$")
ax1.set_title("점 소스 위치별 그린 함수 $G(x, x')$")
ax1.legend(); ax1.grid(True, alpha=0.3)

# 오른쪽: 중첩 원리 - 임의의 소스에 대한 해
f_source = lambda t: np.sin(2 * np.pi * t)  # 임의의 소스 f(x)
u_solution = np.array([np.trapz(greens_function_string(xi, x) * f_source(x), x)
                        for xi in x])

ax2.plot(x, f_source(x), 'r--', linewidth=1.5, label='소스 $f(x) = \\sin(2\\pi x)$')
ax2.plot(x, u_solution, 'b-', linewidth=2.5, label='해 $u(x) = \\int G f \\, dx\'$')
ax2.set_xlabel('$x$'); ax2.set_title('중첩 원리에 의한 해 구성')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle('그린 함수의 개념: 점 소스 응답의 중첩', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 3. 경계값 문제의 그린 함수

### 3.1 구성 방법

2차 ODE $L[y] = y'' + p(x)y' + q(x)y = f(x)$에 대해 제차 경계 조건 $y(a) = 0$, $y(b) = 0$에서의 그린 함수를 구성하자.

**단계 1**: 제차 방정식 $L[y] = 0$의 두 독립해를 구한다:
- $y_1(x)$: $y_1(a) = 0$을 만족
- $y_2(x)$: $y_2(b) = 0$을 만족

**단계 2**: 그린 함수는 구간별로 정의된다:

$$G(x, x') = \begin{cases} A \, y_1(x) y_2(x') & x < x' \\ A \, y_1(x') y_2(x) & x > x' \end{cases}$$

**단계 3**: 접합 조건(matching conditions)을 적용한다:

(1) **연속성**: $G$는 $x = x'$에서 연속

$$G(x'^-, x') = G(x'^+, x')$$

(2) **도함수의 불연속**: $G'$는 $x = x'$에서 점프 불연속

$$\left.\frac{\partial G}{\partial x}\right|_{x'^+} - \left.\frac{\partial G}{\partial x}\right|_{x'^-} = \frac{1}{p(x')}$$

여기서 $p(x')$는 $L$의 최고차 항의 계수이다 (표준형 $y''$이면 $p=1$).

상수 $A$는 이 조건들로부터 결정된다:

$$A = \frac{1}{p(x') W(y_1, y_2)(x')}$$

$W$는 **론스키안(Wronskian)**: $W = y_1 y_2' - y_1' y_2$.

### 3.2 대칭성

**정리**: 자기수반 연산자의 그린 함수는 대칭이다:

$$G(x, x') = G(x', x)$$

이것은 **상반 정리(reciprocity theorem)**를 수학적으로 표현한 것이다: 점 $x'$의 소스가 점 $x$에 만드는 응답은, 점 $x$의 소스가 점 $x'$에 만드는 응답과 같다.

### 3.3 예제: $y'' = f(x)$, $y(0) = y(1) = 0$

제차해: $y_1 = x$, $y_2 = 1 - x$. $W = y_1 y_2' - y_1'y_2 = -x - (1-x) = -1$.

$$G(x, x') = \begin{cases} x(1-x') & x < x' \\ x'(1-x) & x > x' \end{cases}$$

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# SymPy로 그린 함수 구성 및 검증
x_sym, xp_sym = sp.symbols('x xp')

# y'' = f(x), y(0) = y(1) = 0
# 제차해: y1 = x (y1(0)=0), y2 = 1-x (y2(1)=0)
y1 = x_sym
y2 = 1 - x_sym
W = y1 * sp.diff(y2, x_sym) - sp.diff(y1, x_sym) * y2
print(f"론스키안 W = {W}")  # -1

# 그린 함수 (x < x' 영역)
G_left = -y1 * y2.subs(x_sym, xp_sym) / W  # x < x'
G_right = -y1.subs(x_sym, xp_sym) * y2 / W  # x > x'
print(f"G(x, x') = {G_left} (x < x')")
print(f"G(x, x') = {G_right} (x > x')")

# 접합 조건 검증
print("\n=== 접합 조건 검증 (x = x') ===")
# 연속성
G_left_at_xp = G_left.subs(x_sym, xp_sym)
G_right_at_xp = G_right.subs(x_sym, xp_sym)
print(f"연속성: G(x'^-, x') = {G_left_at_xp}, G(x'^+, x') = {G_right_at_xp}")
print(f"  차이 = {sp.simplify(G_left_at_xp - G_right_at_xp)}")

# 도함수 점프
dG_left = sp.diff(G_left, x_sym).subs(x_sym, xp_sym)
dG_right = sp.diff(G_right, x_sym).subs(x_sym, xp_sym)
jump = sp.simplify(dG_right - dG_left)
print(f"도함수 점프: G'(x'^+) - G'(x'^-) = {jump}")  # 1 (= 1/p(x'))

# 수치 시각화
x_num = np.linspace(0, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# G(x, x') 등고선 시각화 (대칭성 확인)
xp_vals = np.linspace(0, 1, 500)
X, XP = np.meshgrid(x_num, xp_vals)
G_vals = np.where(X < XP, X * (1 - XP), XP * (1 - X))

c = ax1.contourf(X, XP, G_vals, levels=30, cmap='viridis')
plt.colorbar(c, ax=ax1)
ax1.set_xlabel("$x$"); ax1.set_ylabel("$x'$")
ax1.set_title("$G(x, x')$ 등고선 — 대칭성 $G(x,x') = G(x',x)$")

# 특정 x'에서의 G(x, x') 단면
for xp in [0.25, 0.5, 0.75]:
    G_section = np.where(x_num < xp, x_num * (1 - xp), xp * (1 - x_num))
    ax2.plot(x_num, G_section, linewidth=2, label=f"$x' = {xp}$")
    # 꺾이는 점(도함수 불연속) 표시
    ax2.plot(xp, xp * (1 - xp), 'ko', markersize=6)

ax2.set_xlabel('$x$'); ax2.set_ylabel("$G(x, x')$")
ax2.set_title("그린 함수 단면: 도함수 불연속 확인")
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. 스투름-리우빌 문제와 고유함수 전개

### 4.1 고유함수 전개법

[스투름-리우빌 이론 (Lesson 12)](12_Sturm_Liouville_Theory.md)에서 배운 고유함수의 완비성을 이용하면, 그린 함수를 고유함수의 급수로 전개할 수 있다.

자기수반 연산자 $L$의 고유값 문제:

$$L[\phi_n] = \lambda_n w(x) \phi_n, \quad n = 1, 2, 3, \ldots$$

고유함수 $\{\phi_n\}$이 $L^2_w[a,b]$에서 완비 직교계를 이루면, 그린 함수는:

$$\boxed{G(x, x') = \sum_{n=1}^{\infty} \frac{\phi_n(x) \phi_n(x')}{\lambda_n \|\phi_n\|_w^2}}$$

여기서 $\|\phi_n\|_w^2 = \int_a^b w(x) |\phi_n(x)|^2 dx$이다.

### 4.2 유도

$G(x, x')$를 고유함수로 전개한다:

$$G(x, x') = \sum_n c_n(x') \phi_n(x)$$

$L[G] = \delta(x - x')$에 대입하고, 양변에 $\phi_m(x)$를 곱하고 적분하면:

$$\lambda_m c_m(x') \|\phi_m\|_w^2 = \phi_m(x')$$

따라서 $c_m(x') = \phi_m(x') / (\lambda_m \|\phi_m\|_w^2)$.

### 4.3 예제: $y'' = f(x)$, $y(0) = y(\pi) = 0$

고유값 문제: $\phi_n'' = -\lambda_n \phi_n$, $\phi_n(0) = \phi_n(\pi) = 0$

$\phi_n = \sin(nx)$, $\lambda_n = n^2$, $\|\phi_n\|^2 = \pi/2$

$$G(x, x') = \frac{2}{\pi} \sum_{n=1}^{\infty} \frac{\sin(nx)\sin(nx')}{n^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi, 500)

def G_eigenfunction(x_val, xp, N_terms):
    """고유함수 전개로 구한 그린 함수"""
    G = np.zeros_like(x_val)
    for n in range(1, N_terms + 1):
        G += (2 / np.pi) * np.sin(n * x_val) * np.sin(n * xp) / n**2
    return G

def G_exact(x_val, xp):
    """정확한 그린 함수 (닫힌 형태)"""
    return np.where(x_val < xp,
                    x_val * (np.pi - xp) / np.pi,
                    xp * (np.pi - x_val) / np.pi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
xp = np.pi / 2  # x' = pi/2

# 왼쪽: 급수 수렴 시각화
ax1.plot(x, G_exact(x, xp), 'k-', linewidth=3, label='정확한 해')
for N in [1, 3, 10, 50]:
    ax1.plot(x, G_eigenfunction(x, xp, N), '--', linewidth=1.5, label=f'$N = {N}$')
ax1.set_title(f"고유함수 전개의 수렴 ($x' = \\pi/2$)")
ax1.set_xlabel('$x$'); ax1.legend(); ax1.grid(True, alpha=0.3)

# 오른쪽: 항 수에 따른 오차
N_values = np.arange(1, 101)
errors = []
for N in N_values:
    G_approx = G_eigenfunction(x, xp, N)
    G_true = G_exact(x, xp)
    errors.append(np.max(np.abs(G_approx - G_true)))

ax2.semilogy(N_values, errors, 'b-', linewidth=2)
ax2.set_xlabel('항 수 $N$'); ax2.set_ylabel('최대 오차')
ax2.set_title('고유함수 전개의 수렴 속도')
ax2.grid(True, alpha=0.3)

plt.suptitle('고유함수 전개법에 의한 그린 함수', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 5. 상미분방정식의 그린 함수

### 5.1 일반적인 2차 ODE

$$y'' + p(x)y' + q(x)y = f(x), \quad y(a) = y(b) = 0$$

이 문제의 해는 **매개변수 변환법(variation of parameters)**과 밀접하게 연결된다. 사실 매개변수 변환법으로 구한 특수해가 그린 함수를 이용한 적분 표현과 동치임을 보일 수 있다.

### 5.2 상수계수 방정식

$y'' + k^2 y = f(x)$, $y(0) = y(L) = 0$인 경우:

제차해: $y_1 = \sin(kx)$, $y_2 = \sin(k(L-x))$

$$G(x, x') = \frac{1}{k\sin(kL)} \begin{cases} \sin(kx)\sin(k(L-x')) & x < x' \\ \sin(kx')\sin(k(L-x)) & x > x' \end{cases}$$

### 5.3 예제: 조화 진동자의 그린 함수

**감쇠 조화 진동자**: $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = f(t)$

초기 조건 $x(0) = \dot{x}(0) = 0$인 **인과적(causal) 그린 함수** 또는 **지연 그린 함수(retarded Green's function)**:

$$G_R(t, t') = \begin{cases} \frac{1}{\omega_d} e^{-\gamma(t-t')} \sin(\omega_d(t-t')) & t > t' \\ 0 & t < t' \end{cases}$$

여기서 $\omega_d = \sqrt{\omega_0^2 - \gamma^2}$ (미소 감쇠, $\gamma < \omega_0$).

해: $x(t) = \int_0^t G_R(t, t') f(t') \, dt'$

```python
import numpy as np
import matplotlib.pyplot as plt

# 감쇠 조화 진동자의 그린 함수
omega0, gamma = 5.0, 0.5
omega_d = np.sqrt(omega0**2 - gamma**2)

def G_retarded(t, tp):
    """지연 그린 함수"""
    dt = t - tp
    return np.where(dt > 0,
                    np.exp(-gamma * dt) * np.sin(omega_d * dt) / omega_d,
                    0.0)

t = np.linspace(0, 10, 2000)

# 다양한 소스 함수에 대한 응답 계산
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sources = {
    '임펄스': lambda tp: np.where(np.abs(tp - 1.0) < 0.05, 1.0/0.1, 0.0),
    '계단 함수': lambda tp: np.where(tp > 1.0, 1.0, 0.0),
    '정현파': lambda tp: np.sin(3.0 * tp),
    '이중 펄스': lambda tp: (np.where(np.abs(tp-1)<0.05, 1/0.1, 0.0) +
                             np.where(np.abs(tp-3)<0.05, -1/0.1, 0.0))
}

for ax, (name, f_source) in zip(axes.flat, sources.items()):
    # 그린 함수를 이용한 컨볼루션 적분
    f_vals = f_source(t)
    x_response = np.array([np.trapz(G_retarded(ti, t[:i+1]) * f_vals[:i+1], t[:i+1])
                           if i > 0 else 0.0 for i, ti in enumerate(t)])

    ax.plot(t, f_vals * 0.1, 'r--', alpha=0.5, label='소스 $f(t)$ (축소)')
    ax.plot(t, x_response, 'b-', linewidth=2, label='응답 $x(t)$')
    ax.set_title(f'소스: {name}'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('$t$')

plt.suptitle(f'감쇠 조화 진동자 ($\\omega_0={omega0}, \\gamma={gamma}$)',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 6. 편미분방정식의 그린 함수

### 6.1 푸아송 방정식의 자유 공간 그린 함수

**푸아송 방정식**: $\nabla^2 \phi = -\rho/\epsilon_0$

그린 함수 정의: $\nabla^2 G(\mathbf{r}, \mathbf{r}') = \delta^3(\mathbf{r} - \mathbf{r}')$

**3차원 자유 공간**:

$$G(\mathbf{r}, \mathbf{r}') = -\frac{1}{4\pi|\mathbf{r} - \mathbf{r}'|}$$

**2차원 자유 공간**:

$$G(\mathbf{r}, \mathbf{r}') = \frac{1}{2\pi}\ln|\mathbf{r} - \mathbf{r}'|$$

해: $\phi(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d^3r'$ — 바로 **쿨롱 전위**!

### 6.2 영상법 (Method of Images)

유한 경계가 있으면 자유 공간 그린 함수를 직접 쓸 수 없다. **영상법(method of images)**은 경계 조건을 만족하는 그린 함수를 "허상 소스"를 추가하여 구성하는 기법이다.

**예**: 접지된 무한 평면 ($z = 0$) 위의 점전하

실제 전하 $q$가 $(0, 0, d)$에 있으면, 허상 전하 $-q$를 $(0, 0, -d)$에 놓는다:

$$G(\mathbf{r}, \mathbf{r}') = -\frac{1}{4\pi}\left(\frac{1}{|\mathbf{r} - \mathbf{r}'|} - \frac{1}{|\mathbf{r} - \mathbf{r}''|}\right)$$

여기서 $\mathbf{r}'' = (x', y', -z')$는 영상점이다.

### 6.3 열방정식의 그린 함수

**열방정식**: $\frac{\partial u}{\partial t} = \alpha^2 \nabla^2 u$

자유 공간 그린 함수 (1D):

$$G(x, t; x', t') = \frac{1}{\sqrt{4\pi\alpha^2(t-t')}} \exp\left(-\frac{(x-x')^2}{4\alpha^2(t-t')}\right), \quad t > t'$$

이것은 $t = t'$에서 $\delta(x - x')$로 시작하여 시간에 따라 가우시안으로 퍼져 나가는 모양이다.

### 6.4 파동방정식의 지연 그린 함수

**파동방정식**: $\nabla^2 G - \frac{1}{c^2}\frac{\partial^2 G}{\partial t^2} = \delta^3(\mathbf{r} - \mathbf{r}')\delta(t - t')$

3D **지연 그린 함수(retarded Green's function)**:

$$G_R(\mathbf{r}, t; \mathbf{r}', t') = -\frac{\delta(t - t' - |\mathbf{r}-\mathbf{r}'|/c)}{4\pi|\mathbf{r}-\mathbf{r}'|}$$

이것은 인과율(causality)을 반영한다: 신호가 빛의 속도 $c$로 전파되어 시간 $|\mathbf{r}-\mathbf{r}'|/c$ 후에 도달한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 열방정식 그린 함수 시각화 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

alpha2 = 0.01
x = np.linspace(-2, 2, 1000)
xp = 0.0  # 소스 위치

for t_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    G_heat = np.exp(-(x - xp)**2 / (4 * alpha2 * t_val)) / np.sqrt(4 * np.pi * alpha2 * t_val)
    ax1.plot(x, G_heat, linewidth=2, label=f'$t = {t_val}$')

ax1.set_xlabel('$x$'); ax1.set_ylabel("$G(x, t; 0, 0)$")
ax1.set_title('열방정식 그린 함수 (1D)')
ax1.legend(); ax1.grid(True, alpha=0.3)

# --- 2D 푸아송 방정식 그린 함수 시각화 ---
x2d = np.linspace(-2, 2, 300)
y2d = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(x2d, y2d)

# 점 소스 위치
xp2, yp2 = 0.5, 0.3
R = np.sqrt((X - xp2)**2 + (Y - yp2)**2)
R = np.maximum(R, 0.01)  # 특이점 방지
G_2d = np.log(R) / (2 * np.pi)

c = ax2.contourf(X, Y, G_2d, levels=30, cmap='RdBu_r')
ax2.plot(xp2, yp2, 'k*', markersize=15, label="소스 위치 $(x', y')$")
plt.colorbar(c, ax=ax2)
ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$')
ax2.set_title("2D 푸아송 그린 함수 $G = \\frac{1}{2\\pi}\\ln r$")
ax2.legend(); ax2.set_aspect('equal')

plt.suptitle('편미분방정식의 그린 함수', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

---

## 7. 물리적 응용

### 7.1 정전기학: 전하 분포의 전위

전하 밀도 $\rho(\mathbf{r})$가 주어지면 전위는:

$$\phi(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d^3r'$$

이것은 정확히 $\nabla^2\phi = -\rho/\epsilon_0$의 그린 함수 해이다.

### 7.2 양자역학: 전파함수

시간 의존 슈뢰딩거 방정식의 그린 함수가 바로 **전파함수(propagator)** $K(\mathbf{r}, t; \mathbf{r}', t')$이다:

$$\psi(\mathbf{r}, t) = \int K(\mathbf{r}, t; \mathbf{r}', t_0) \psi(\mathbf{r}', t_0) \, d^3r'$$

자유 입자의 전파함수: $K = \left(\frac{m}{2\pi i\hbar(t-t')}\right)^{3/2} \exp\left(\frac{im|\mathbf{r}-\mathbf{r}'|^2}{2\hbar(t-t')}\right)$

### 7.3 음향학: 점음원의 방사

음파 방정식 $\nabla^2 p - \frac{1}{c^2}\ddot{p} = -S(\mathbf{r}, t)$에서 단색파 점음원 $S = \delta^3(\mathbf{r})e^{-i\omega t}$의 음압:

$$p(\mathbf{r}) = -\frac{e^{ikr}}{4\pi r} \quad (k = \omega/c)$$

이것이 **구면파(spherical wave)**이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# === 정전기학 응용: 2D 전하 분포의 전위 ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# (a) 점전하
q1, x1, y1 = 1.0, 0.0, 0.0
R1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
phi_point = q1 / (2 * np.pi * np.maximum(R1, 0.05))

axes[0].contourf(X, Y, phi_point, levels=30, cmap='hot_r')
axes[0].plot(x1, y1, 'k+', markersize=15, markeredgewidth=3)
axes[0].set_title('(a) 점전하 $+q$'); axes[0].set_aspect('equal')

# (b) 쌍극자 (dipole)
q2, d = 1.0, 0.5
R_plus = np.sqrt((X - d)**2 + Y**2)
R_minus = np.sqrt((X + d)**2 + Y**2)
phi_dipole = q2 / (2*np.pi*np.maximum(R_plus, 0.05)) - q2 / (2*np.pi*np.maximum(R_minus, 0.05))

axes[1].contourf(X, Y, phi_dipole, levels=np.linspace(-3, 3, 31), cmap='RdBu_r')
axes[1].plot(d, 0, 'r+', markersize=12, markeredgewidth=3)
axes[1].plot(-d, 0, 'b_', markersize=12, markeredgewidth=3)
axes[1].set_title('(b) 전기 쌍극자 $+q, -q$'); axes[1].set_aspect('equal')

# (c) 영상법: 접지면 근처 점전하
q3, d3 = 1.0, 1.0
R_real = np.sqrt(X**2 + (Y - d3)**2)
R_image = np.sqrt(X**2 + (Y + d3)**2)  # 허상 전하
phi_image = q3/(2*np.pi*np.maximum(R_real, 0.05)) - q3/(2*np.pi*np.maximum(R_image, 0.05))
phi_image[Y < 0] = 0  # 접지면 아래는 0

axes[2].contourf(X, Y, phi_image, levels=30, cmap='hot_r')
axes[2].axhline(0, color='green', linewidth=3, label='접지면')
axes[2].plot(0, d3, 'k+', markersize=12, markeredgewidth=3)
axes[2].plot(0, -d3, 'kx', markersize=12, markeredgewidth=3, alpha=0.4)
axes[2].set_title('(c) 영상법: 접지면 위 점전하')
axes[2].legend(); axes[2].set_aspect('equal')

plt.suptitle('정전기학에서의 그린 함수 응용 (2D)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# 전기장 벡터 (쌍극자)
fig, ax = plt.subplots(figsize=(8, 6))
Ex = q2*(X-d)/(2*np.pi*np.maximum(R_plus,0.05)**2) - q2*(X+d)/(2*np.pi*np.maximum(R_minus,0.05)**2)
Ey = q2*Y/(2*np.pi*np.maximum(R_plus,0.05)**2) - q2*Y/(2*np.pi*np.maximum(R_minus,0.05)**2)
E_mag = np.sqrt(Ex**2 + Ey**2)

ax.streamplot(X, Y, Ex, Ey, color=np.log10(E_mag+1e-3), cmap='inferno',
              density=2, linewidth=1)
ax.plot(d, 0, 'ro', markersize=10, label='$+q$')
ax.plot(-d, 0, 'bo', markersize=10, label='$-q$')
ax.set_title('전기 쌍극자의 전기장선 (그린 함수 중첩)')
ax.legend(); ax.set_aspect('equal'); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
plt.tight_layout(); plt.show()
```

---

## 8. 그린 정리와 적분 표현

### 8.1 그린 항등식

**그린 제1 항등식**: $u$, $v$가 영역 $\Omega$에서 충분히 매끄러우면:

$$\int_\Omega (u \nabla^2 v + \nabla u \cdot \nabla v) \, dV = \oint_{\partial\Omega} u \frac{\partial v}{\partial n} \, dS$$

**그린 제2 항등식** (대칭 형태):

$$\int_\Omega (u \nabla^2 v - v \nabla^2 u) \, dV = \oint_{\partial\Omega} \left(u \frac{\partial v}{\partial n} - v \frac{\partial u}{\partial n}\right) dS$$

### 8.2 적분 표현

그린 제2 항등식에서 $v = G$로 놓으면 ($\nabla^2 G = \delta^3(\mathbf{r} - \mathbf{r}')$):

$$u(\mathbf{r}) = \int_\Omega G(\mathbf{r}, \mathbf{r}') f(\mathbf{r}') \, dV' + \oint_{\partial\Omega} \left(G \frac{\partial u}{\partial n'} - u \frac{\partial G}{\partial n'}\right) dS'$$

**디리클레 경계 조건** ($u = h$ on $\partial\Omega$): $G = 0$ on $\partial\Omega$으로 선택하면:

$$u(\mathbf{r}) = \int_\Omega G f \, dV' - \oint_{\partial\Omega} h \frac{\partial G}{\partial n'} dS'$$

**노이만 경계 조건** ($\partial u/\partial n = g$ on $\partial\Omega$): $\partial G/\partial n' = -1/|\partial\Omega|$ (상수)로 선택.

```python
import numpy as np
import matplotlib.pyplot as plt

# 그린 항등식의 수치 검증 (1D 버전)
# integral_0^1 (u v'' + u'v') dx = [u v']_0^1
# u(x) = sin(pi*x), v(x) = x^2

x = np.linspace(0, 1, 10000)

u = np.sin(np.pi * x)
u_prime = np.pi * np.cos(np.pi * x)
v = x**2
v_prime = 2 * x
v_double_prime = 2 * np.ones_like(x)

# 좌변
lhs = np.trapz(u * v_double_prime + u_prime * v_prime, x)

# 우변: [u v']_0^1 = u(1)v'(1) - u(0)v'(0)
rhs = u[-1] * v_prime[-1] - u[0] * v_prime[0]

print("=== 그린 제1 항등식 수치 검증 (1D) ===")
print(f"u(x) = sin(pi*x), v(x) = x^2")
print(f"좌변: integral(u v'' + u'v') dx = {lhs:.8f}")
print(f"우변: [u v']_0^1               = {rhs:.8f}")
print(f"차이: {abs(lhs - rhs):.2e}")

# 그린 제2 항등식
u_dbl_prime = -np.pi**2 * np.sin(np.pi * x)

lhs2 = np.trapz(u * v_double_prime - v * u_dbl_prime, x)
rhs2 = (u[-1]*v_prime[-1] - v[-1]*u_prime[-1]) - (u[0]*v_prime[0] - v[0]*u_prime[0])

print("\n=== 그린 제2 항등식 수치 검증 (1D) ===")
print(f"좌변: integral(u v'' - v u'') dx = {lhs2:.8f}")
print(f"우변: [uv' - vu']_0^1           = {rhs2:.8f}")
print(f"차이: {abs(lhs2 - rhs2):.2e}")

# 그린 함수를 이용한 적분 표현 검증
# 문제: u'' = f(x), u(0) = u(1) = 0, f(x) = -pi^2 sin(pi*x)
# 정확한 해: u(x) = sin(pi*x)
print("\n=== 적분 표현 검증 ===")
f_rhs = -np.pi**2 * np.sin(np.pi * x)
u_green = np.array([np.trapz(np.where(x < xi, x*(1-xi), xi*(1-x)) * f_rhs, x)
                     for xi in x])
u_exact = np.sin(np.pi * x)

print(f"최대 오차 |u_Green - u_exact| = {np.max(np.abs(u_green - u_exact)):.6e}")
```

---

## 연습 문제

### 기본 문제

**문제 1.** 다음 디랙 델타 함수 적분을 계산하라.

(a) $\int_{-\infty}^{\infty} (x^3 + 2x + 1)\delta(x - 2) \, dx$

(b) $\int_0^5 e^{-x}\delta(x - 3) \, dx$

(c) $\int_{-\infty}^{\infty} \cos(x)\delta'(x) \, dx$

**문제 2.** $\delta(x^2 - a^2) = \frac{1}{2|a|}[\delta(x-a) + \delta(x+a)]$ ($a > 0$)임을 보이고, $\int_{-\infty}^{\infty} e^{x}\delta(x^2 - 4)\,dx$를 계산하라.

**문제 3.** $y'' = f(x)$, $y(0) = y(L) = 0$의 그린 함수를 직접 구성하고, $G(x, x') = G(x', x)$를 확인하라.

**문제 4.** $y'' + y = f(x)$, $y(0) = y(\pi/2) = 0$의 그린 함수를 구하라. (힌트: 제차해 $\sin x$, $\cos x$ 이용)

### 심화 문제

**문제 5.** $y'' = f(x)$, $y(0) = y(\pi) = 0$의 그린 함수를 고유함수 전개로 구하고, 닫힌 형태 $G(x,x') = \frac{1}{\pi}[x(\pi-x') \text{ or } x'(\pi-x)]$과 비교하여 다음 급수를 유도하라:

$$\sum_{n=1}^{\infty} \frac{\sin(nx)\sin(nx')}{n^2} = \frac{\pi}{2} \begin{cases} x(1-x'/\pi) & x < x' \\ x'(1-x/\pi) & x > x' \end{cases}$$

**문제 6.** 영상법을 이용하여 $y > 0$ 반평면에서 $\nabla^2 G = \delta^2(\mathbf{r} - \mathbf{r}')$, $G(x, 0) = 0$ (디리클레)인 그린 함수를 구하라.

**문제 7.** 1차원 열방정식 그린 함수 $G(x, t; 0, 0) = \frac{1}{\sqrt{4\pi\alpha^2 t}}e^{-x^2/(4\alpha^2 t)}$가 다음을 만족함을 보여라:

(a) $\partial_t G = \alpha^2 \partial_{xx} G$ ($t > 0$)

(b) $\int_{-\infty}^{\infty} G \, dx = 1$ (모든 $t > 0$)

(c) $\lim_{t \to 0^+} G(x, t; 0, 0) = \delta(x)$

**문제 8.** 감쇠 조화 진동자 $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = \delta(t)$에 대해 $\gamma > \omega_0$ (과감쇠), $\gamma = \omega_0$ (임계 감쇠), $\gamma < \omega_0$ (미소 감쇠) 세 경우의 그린 함수를 각각 구하라.

**문제 9.** 다음 2차원 푸아송 방정식을 풀어라:
$$\nabla^2 \phi = -\delta(\mathbf{r} - \mathbf{r}_1) + \delta(\mathbf{r} - \mathbf{r}_2)$$
$\mathbf{r}_1 = (1, 0)$, $\mathbf{r}_2 = (-1, 0)$. 전위 $\phi$와 전기장 $\mathbf{E} = -\nabla\phi$를 시각화하라.

**문제 10.** 그린 제2 항등식을 이용하여 $G(x, x') = G(x', x)$ (자기수반 연산자의 그린 함수 대칭성)를 증명하라.

---

## 심화 학습

### 다이아딕 그린 함수 (Dyadic Green's Functions)

스칼라가 아닌 **벡터장**의 비제차 문제(예: 맥스웰 방정식)에서는 그린 함수가 텐서(다이아딕) 형태가 된다:

$$\mathbf{E}(\mathbf{r}) = \int \overleftrightarrow{G}(\mathbf{r}, \mathbf{r}') \cdot \mathbf{J}(\mathbf{r}') \, d^3r'$$

### 주파수 영역의 그린 함수

시간 의존 문제에서 푸리에 변환을 취하면:

$$G(\mathbf{r}, \mathbf{r}'; \omega) = \int_{-\infty}^{\infty} G(\mathbf{r}, t; \mathbf{r}', t') e^{i\omega(t-t')} d(t-t')$$

**헬름홀츠 방정식**의 그린 함수: $(\nabla^2 + k^2)G = \delta^3(\mathbf{r} - \mathbf{r}')$ $\rightarrow$ $G = -\frac{e^{ik|\mathbf{r}-\mathbf{r}'|}}{4\pi|\mathbf{r}-\mathbf{r}'|}$

### 수치적 그린 함수와 경계요소법 (BEM)

해석적으로 그린 함수를 구하기 어려운 복잡한 기하학에서는 **경계요소법(Boundary Element Method)**을 사용한다. 자유 공간 그린 함수를 알면, 체적 적분 대신 **경계면 적분**만으로 해를 구할 수 있어 차원이 하나 줄어드는 장점이 있다.

### 참고 자료

- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd Ed., Ch. 13
- **Arfken, Weber, Harris.** *Mathematical Methods for Physicists*, 7th Ed., Ch. 10
- **Jackson, J. D.** *Classical Electrodynamics*, 3rd Ed., Ch. 1-2 (정전기학 그린 함수)
- **Stakgold, I., Holst, M.** *Green's Functions and Boundary Value Problems*, 3rd Ed. (2011)
- **Duffy, D. G.** *Green's Functions with Applications*, 2nd Ed. (2015)

---

**이전**: [15. 라플라스 변환](15_Laplace_Transform.md)
**다음**: [17. 변분법](17_Calculus_of_Variations.md)
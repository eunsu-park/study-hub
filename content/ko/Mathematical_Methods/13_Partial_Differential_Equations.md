# 13. 편미분방정식 (Partial Differential Equations)

## 학습 목표

- 편미분방정식(PDE)을 **타원형, 포물형, 쌍곡형**으로 분류하고, 각 유형에 적합한 **경계 조건 및 초기 조건**을 설정할 수 있다
- **변수분리법(separation of variables)**을 직교좌표, 원통좌표, 구면좌표에서 적용하여 PDE를 ODE로 분해할 수 있다
- **열방정식**, **파동방정식**, **라플라스 방정식**의 해석적 해를 구하고 물리적 의미를 해석할 수 있다
- **달랑베르 해**를 이용하여 파동의 전파를 기술할 수 있다
- **전자기학의 전위 문제**와 **양자역학의 무한 퍼텐셜 우물**에 PDE 해법을 적용할 수 있다

---

## 1. PDE의 분류

### 1.1 타원형, 포물형, 쌍곡형

2차 선형 PDE의 일반적 형태:

$$A \frac{\partial^2 u}{\partial x^2} + 2B \frac{\partial^2 u}{\partial x \partial y} + C \frac{\partial^2 u}{\partial y^2} + \text{(저차 항)} = 0$$

**판별식** $\Delta = B^2 - AC$에 따라 세 가지 유형으로 분류한다:

| 판별식 | 유형 | 대표 방정식 | 물리적 의미 |
|--------|------|-------------|-------------|
| $\Delta < 0$ | **타원형(elliptic)** | 라플라스 방정식 $\nabla^2 u = 0$ | 정상 상태(평형) |
| $\Delta = 0$ | **포물형(parabolic)** | 열방정식 $u_t = \alpha^2 u_{xx}$ | 확산 과정 |
| $\Delta > 0$ | **쌍곡형(hyperbolic)** | 파동방정식 $u_{tt} = c^2 u_{xx}$ | 파동 전파 |

이 분류는 원뿔 곡선(conic section)의 분류와 동일한 수학적 구조를 갖는다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- PDE 유형별 해의 특성 비교 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 타원형: 라플라스 방정식의 조화함수 u = x² - y²
x, y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
axes[0].contourf(X, Y, X**2 - Y**2, levels=20, cmap='RdBu_r')
axes[0].set_title('타원형: $\\nabla^2 u = 0$ (정상 상태)')

# 포물형: 열방정식 해의 시간 스냅샷
x_h = np.linspace(0, np.pi, 100)
for t in [0.0, 0.05, 0.2, 0.5, 1.0]:
    axes[1].plot(x_h, np.sin(x_h) * np.exp(-t), label=f't={t}')
axes[1].set_title('포물형: $u_t = \\alpha^2 u_{xx}$ (열 확산)')
axes[1].legend(fontsize=8)

# 쌍곡형: 파동방정식 해의 시간 스냅샷
x_w = np.linspace(0, 2*np.pi, 200)
for t in [0.0, 0.5, 1.0, 1.5]:
    axes[2].plot(x_w, np.sin(x_w - t) + np.sin(x_w + t), label=f't={t:.1f}')
axes[2].set_title('쌍곡형: $u_{tt} = c^2 u_{xx}$ (파동 전파)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

### 1.2 경계 조건과 초기 조건

PDE의 유일한 해를 결정하려면 적절한 **경계 조건(BC)**과 **초기 조건(IC)**이 필요하다.

| 유형 | 이름 | 조건 | 물리적 의미 |
|------|------|------|-------------|
| 제1종 | **디리클레(Dirichlet)** | $u = f$ on $\partial\Omega$ | 경계의 온도/전위 지정 |
| 제2종 | **노이만(Neumann)** | $\partial u / \partial n = g$ on $\partial\Omega$ | 경계의 열유속/전기장 지정 |
| 제3종 | **로빈(Robin)** | $\alpha u + \beta \partial u / \partial n = h$ | 대류 열전달 (뉴턴 냉각) |

**Hadamard의 잘 놓인 문제(well-posed problem)** 조건: 해가 (1) 존재하고, (2) 유일하며, (3) 데이터에 연속적으로 의존해야 한다.

- **타원형**: 경계 조건만 필요 (BVP)
- **포물형**: 초기 조건 + 경계 조건 (IBVP)
- **쌍곡형**: 초기 조건 ($u$와 $u_t$ 모두) + 경계 조건

---

## 2. 변수분리법 (Separation of Variables)

### 2.1 직교좌표에서의 변수분리

핵심 아이디어: 해를 각 변수의 함수의 곱 $u(x, t) = X(x) T(t)$로 가정한다.

**예시 - 1차원 열방정식** ($u(0,t) = u(L,t) = 0$, $u(x,0) = f(x)$):

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}$$

$u = X(x)T(t)$를 대입하고 양변을 $\alpha^2 XT$로 나누면:

$$\frac{T'}{\alpha^2 T} = \frac{X''}{X} = -\lambda \quad (\text{분리 상수})$$

두 개의 ODE로 분리된다: $X'' + \lambda X = 0$ (경계 조건 포함)과 $T' + \alpha^2 \lambda T = 0$.

고유값: $\lambda_n = (n\pi/L)^2$, 고유함수: $X_n = \sin(n\pi x/L)$. 일반해:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha^2 (n\pi/L)^2 t}, \quad b_n = \frac{2}{L}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right)dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 변수분리법: 1차원 열방정식 풀이 ---
L, alpha, N_terms = np.pi, 1.0, 50

def initial_condition(x):
    return x * (np.pi - x)

# 푸리에 사인 계수 (수치 적분)
def compute_bn(n):
    x = np.linspace(0, L, 1000)
    return (2/L) * np.trapz(initial_condition(x) * np.sin(n*np.pi*x/L), x)

coeffs = [compute_bn(n) for n in range(1, N_terms + 1)]

def heat_solution(x, t):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N_terms + 1):
        u += coeffs[n-1] * np.sin(n*np.pi*x/L) * np.exp(-alpha**2*(n*np.pi/L)**2*t)
    return u

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    plt.plot(x, heat_solution(x, t), label=f't = {t}')
plt.xlabel('x'); plt.ylabel('u(x, t)')
plt.title('1차원 열방정식의 해 (변수분리법)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

print("=== 푸리에 사인 계수 (처음 5개) ===")
for n in range(1, 6):
    print(f"  b_{n} = {coeffs[n-1]:.6f}")
```

### 2.2 원통좌표에서의 변수분리

원통좌표 $(r, \phi, z)$에서 라플라스 방정식에 $u = R(r)\Phi(\phi)Z(z)$를 대입하면 세 ODE로 분리된다:

$$Z'' - k^2 Z = 0, \qquad \Phi'' + m^2 \Phi = 0, \qquad r^2 R'' + rR' + (k^2r^2 - m^2)R = 0$$

마지막 방정식이 **베셀 방정식**이며, 해는 $J_m(kr)$과 $Y_m(kr)$이다. 원점을 포함하는 영역에서는 $Y_m$이 발산하므로 $R(r) = J_m(kr)$만 취한다.

### 2.3 구면좌표에서의 변수분리

구면좌표 $(r, \theta, \phi)$에서 $u = R(r)\Theta(\theta)\Phi(\phi)$로 놓으면:

- $\Phi'' + m^2\Phi = 0$ $\Rightarrow$ $\Phi = e^{im\phi}$
- $\Theta$: **연관 르장드르 방정식** $\Rightarrow$ $P_l^m(\cos\theta)$
- $R$: $r^2R'' + 2rR' - l(l+1)R = 0$ $\Rightarrow$ $R = Ar^l + Br^{-(l+1)}$

일반해는 **구면 조화 함수** $Y_l^m(\theta,\phi)$로 표현된다. 축대칭 경우($m=0$):

$$u(r, \theta) = \sum_{l=0}^{\infty}\left(A_l r^l + B_l r^{-(l+1)}\right)P_l(\cos\theta)$$

---

## 3. 열방정식 (Heat/Diffusion Equation)

### 3.1 1차원 열방정식의 풀이

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}, \qquad \alpha^2 = \frac{k}{\rho c_p}$$

**물리적 의미**: 온도의 시간 변화율은 공간 곡률에 비례한다. 볼록한 부분($u_{xx} < 0$)은 냉각, 오목한 부분($u_{xx} > 0$)은 가열된다.

### 3.2 비제차 경계 조건

$u(0,t) = T_1$, $u(L,t) = T_2$인 경우, **정상 상태 해**를 분리한다:

$$u(x, t) = \underbrace{T_1 + \frac{T_2 - T_1}{L}x}_{v(x) \text{ (정상 상태)}} + w(x, t)$$

$w$는 제차 BC($w(0,t) = w(L,t) = 0$)를 만족하며, 초기 조건 $w(x,0) = f(x) - v(x)$로 결정된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 비제차 경계 조건의 열방정식 ---
L, alpha, T1, T2, N_terms = 1.0, 0.1, 100.0, 50.0, 30

steady = lambda x: T1 + (T2 - T1) * x / L
w_init = lambda x: -steady(x)  # 초기: 균일 0°C

# 과도 해의 푸리에 계수
cn = [(2/L) * np.trapz(w_init(np.linspace(0,L,1000)) * np.sin(n*np.pi*np.linspace(0,L,1000)/L),
      np.linspace(0,L,1000)) for n in range(1, N_terms+1)]

def full_solution(x, t):
    u = steady(x)
    for n in range(1, N_terms+1):
        u += cn[n-1] * np.sin(n*np.pi*x/L) * np.exp(-alpha**2*(n*np.pi/L)**2*t)
    return u

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]:
    plt.plot(x, full_solution(x, t), label=f't = {t}')
plt.plot(x, steady(x), 'k--', lw=2, label='정상 상태')
plt.xlabel('x'); plt.ylabel('u(x,t) [°C]')
plt.title('비제차 경계 조건의 열방정식'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 3.3 시간 변화와 정상 상태

$n$번째 모드의 **시간 상수**: $\tau_n = 1 / [\alpha^2(n\pi/L)^2]$. 고주파 모드일수록 빠르게 감쇄하므로, 충분한 시간 후 $n=1$ 모드만 남고, $t \to \infty$에서 정상 상태 $u \to v(x)$에 도달한다.

---

## 4. 파동방정식 (Wave Equation)

### 4.1 1차원 파동방정식과 달랑베르 해

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

**달랑베르 해**: 무한 영역에서 $u(x,0)=f(x)$, $u_t(x,0)=g(x)$일 때:

$$u(x,t) = \frac{1}{2}[f(x-ct) + f(x+ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct} g(s)\,ds$$

**유한 구간** ($u(0,t) = u(L,t) = 0$)에서의 변수분리법:

$$u(x,t) = \sum_{n=1}^{\infty}\sin\left(\frac{n\pi x}{L}\right)\left[A_n\cos(\omega_n t) + B_n\sin(\omega_n t)\right], \quad \omega_n = \frac{n\pi c}{L}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 달랑베르 해: 가우시안 펄스의 좌우 전파 ---
c = 1.0
f = lambda x: np.exp(-10 * x**2)
dalembert = lambda x, t: 0.5 * (f(x - c*t) + f(x + c*t))

x = np.linspace(-5, 5, 500)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]):
    ax = axes[idx//3, idx%3]
    ax.plot(x, dalembert(x, t), 'b-', lw=2)
    ax.fill_between(x, dalembert(x, t), alpha=0.2)
    ax.set_xlim(-5, 5); ax.set_ylim(-0.3, 1.1)
    ax.set_title(f't = {t:.1f}'); ax.grid(True, alpha=0.3)
fig.suptitle("달랑베르 해: 가우시안 펄스의 좌우 전파", fontsize=14)
plt.tight_layout(); plt.show()
```

**기타 현의 진동** (중앙을 뽑아올린 삼각형 초기 변위):

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 기타 현의 진동 (변수분리법) ---
L, c, N = 1.0, 1.0, 20
An = lambda n: 8/(n**2 * np.pi**2) * np.sin(n*np.pi/2)

def string_sol(x, t):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N+1):
        u += An(n) * np.sin(n*np.pi*x/L) * np.cos(n*np.pi*c*t/L)
    return u

x = np.linspace(0, L, 200)
T_period = 2*L/c
plt.figure(figsize=(10, 6))
for t in [0, T_period/8, T_period/4, 3*T_period/8, T_period/2]:
    plt.plot(x, string_sol(x, t), label=f't = {t:.3f} ({t/T_period:.0%} T)')
plt.xlabel('x'); plt.ylabel('u(x,t)')
plt.title(f'기타 현의 진동 (기본 주기 T = {T_period:.2f})')
plt.legend(); plt.grid(True, alpha=0.3); plt.axhline(0, color='k', lw=0.5)
plt.show()

print("=== 고유진동수 ===")
for n in range(1, 6):
    print(f"  n={n}: f_{n} = {n*c/(2*L):.2f} Hz")
```

### 4.2 2차원 원형 막 진동 (드럼)

원형 막의 축대칭 모드는 원통좌표 파동방정식에서 $u(r,t) = J_0(\lambda r)[A\cos(\lambda ct) + B\sin(\lambda ct)]$로 주어진다. 경계 조건 $u(a,t) = 0$으로부터 $\lambda_{0n} = j_{0n}/a$ ($j_{0n}$: $J_0$의 영점).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0

# --- 원형 막 진동의 축대칭 모드 ---
a = 1.0
zeros_J0 = jn_zeros(0, 3)  # [2.4048, 5.5201, 8.6537]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
r = np.linspace(0, a, 100)
for idx, j0n in enumerate(zeros_J0):
    R = j0(j0n * r / a)
    axes[idx].plot(r, R, 'b-', lw=2)
    axes[idx].fill_between(r, R, alpha=0.2)
    axes[idx].axhline(0, color='k', lw=0.5)
    axes[idx].set_title(f'모드 (0,{idx+1}): $j_{{0,{idx+1}}}$ = {j0n:.4f}')
    axes[idx].set_xlabel('r'); axes[idx].grid(True, alpha=0.3)
plt.suptitle('원형 막의 축대칭 진동 모드', fontsize=14)
plt.tight_layout(); plt.show()
```

---

## 5. 라플라스 방정식 (Laplace's Equation)

### 5.1 직사각형 영역

$\nabla^2 u = 0$, 세 변 접지, 윗변 $u(x,b) = f(x)$:

$$u(x,y) = \sum_{n=1}^{\infty} c_n \sin\left(\frac{n\pi x}{a}\right)\sinh\left(\frac{n\pi y}{a}\right), \quad c_n = \frac{2}{a\sinh(n\pi b/a)}\int_0^a f(x)\sin\left(\frac{n\pi x}{a}\right)dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 직사각형 영역에서의 라플라스 방정식 ---
a, b, N = 2.0, 1.0, 30
f_top = lambda x: np.sin(np.pi * x / a)

xq = np.linspace(0, a, 1000)
cn = [(2/(a*np.sinh(n*np.pi*b/a))) * np.trapz(f_top(xq)*np.sin(n*np.pi*xq/a), xq)
      for n in range(1, N+1)]

def laplace_rect(x, y):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, N+1):
        u += cn[n-1] * np.sin(n*np.pi*x/a) * np.sinh(n*np.pi*y/a)
    return u

x, y = np.linspace(0, a, 100), np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(10, 5))
cs = ax.contourf(X, Y, laplace_rect(X, Y), levels=30, cmap='hot')
plt.colorbar(cs, label='u(x,y)')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
ax.set_title('직사각형 영역에서의 라플라스 방정식 해')
plt.tight_layout(); plt.show()
```

### 5.2 원형/구면 영역

**원형 영역** ($r < a$): $u(r,\theta) = a_0/2 + \sum_{n=1}^{\infty}(r/a)^n(a_n\cos n\theta + b_n\sin n\theta)$

**구면 영역** (축대칭): $u(r,\theta) = \sum_{l=0}^{\infty} A_l (r/a)^l P_l(\cos\theta)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- 구면 영역에서의 라플라스 방정식 ---
a, L_max = 1.0, 10
f_bdy = lambda th: np.cos(th)**2  # u(a,θ) = cos²θ

theta_q = np.linspace(0, np.pi, 1000)
Al = [(2*l+1)/2 * np.trapz(f_bdy(theta_q)*legendre(l)(np.cos(theta_q))*np.sin(theta_q), theta_q)
      for l in range(L_max+1)]

print("=== 르장드르 계수 ===")
for l in range(5): print(f"  A_{l} = {Al[l]:.6f}")  # A_0≈1/3, A_2≈2/3

# 시각화 (단면)
r_v, th_v = np.linspace(0.01, a, 50), np.linspace(0, np.pi, 100)
R, Th = np.meshgrid(r_v, th_v)
U = sum(Al[l]*(R/a)**l * legendre(l)(np.cos(Th)) for l in range(L_max+1))

fig, ax = plt.subplots(figsize=(6, 8))
cs = ax.contourf(R*np.sin(Th), R*np.cos(Th), U, levels=30, cmap='coolwarm')
plt.colorbar(cs, label='u(r,θ)')
th_c = np.linspace(0, np.pi, 100)
ax.plot(a*np.sin(th_c), a*np.cos(th_c), 'k-', lw=2)
ax.set_xlabel('r sin(θ)'); ax.set_ylabel('r cos(θ)'); ax.set_aspect('equal')
ax.set_title('구 내부의 라플라스 방정식 해: $u(a,\\theta)=\\cos^2\\theta$')
plt.tight_layout(); plt.show()
```

### 5.3 포아송 방정식

**포아송 방정식** $\nabla^2 u = f(\mathbf{r})$은 라플라스 방정식의 비제차 형태이다. 전자기학에서는 $\nabla^2\phi = -\rho/\epsilon_0$으로 나타난다.

풀이: $u = u_h + u_p$ ($\nabla^2 u_h = 0$, $\nabla^2 u_p = f$). 특수 해는 **그린 함수**로 구한다 (레슨 14 참조):

$$u_p(\mathbf{r}) = \int G(\mathbf{r}, \mathbf{r}') f(\mathbf{r}') \, d^3r'$$

---

## 6. 헬름홀츠 방정식 (Helmholtz Equation)

### 6.1 정의와 물리적 배경

파동방정식에서 시간 의존부를 분리하면 **헬름홀츠 방정식**이 나타난다:

$$\nabla^2 u + k^2 u = 0$$

여기서 $k = \omega/c$는 파수(wavenumber)이다. 라플라스 방정식($k = 0$)의 일반화이다.

**물리적 등장 상황**:

| 분야 | 방정식 | $k$의 의미 |
|------|--------|-----------|
| 음향학 | $\nabla^2 p + k^2 p = 0$ | $k = \omega/c_s$ (음파 파수) |
| 전자기학 | $\nabla^2 \mathbf{E} + k^2 \mathbf{E} = 0$ | $k = \omega\sqrt{\mu\epsilon}$ |
| 양자역학 | $\nabla^2 \psi + \frac{2mE}{\hbar^2}\psi = 0$ | $k^2 = 2mE/\hbar^2$ |
| 확산 (정상) | $\nabla^2 c - \alpha^2 c = 0$ | $k^2 = -\alpha^2 < 0$ (변형 헬름홀츠) |

### 6.2 직교좌표에서의 풀이

직사각형 영역 $[0, a] \times [0, b]$에서 디리클레 경계 조건:

$$u_{n_x, n_y}(x, y) = \sin\frac{n_x \pi x}{a} \sin\frac{n_y \pi y}{b}$$

고유값: $k^2_{n_x, n_y} = \left(\frac{n_x \pi}{a}\right)^2 + \left(\frac{n_y \pi}{b}\right)^2$

### 6.3 원통좌표에서의 헬름홀츠 방정식

$\nabla^2 u + k^2 u = 0$을 원통좌표 $(r, \phi, z)$에서 변수분리 $u = R(r)\Phi(\phi)Z(z)$:

$$R'' + \frac{1}{r}R' + \left(\kappa^2 - \frac{m^2}{r^2}\right)R = 0$$

여기서 $\kappa^2 = k^2 - k_z^2$. 이것은 **베셀 방정식**이며 해는 $J_m(\kappa r)$이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros

# --- 원형 도파관의 헬름홀츠 방정식 모드 ---
a = 1.0  # 도파관 반지름
modes = [(0, 1), (1, 1), (2, 1), (0, 2)]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for idx, (m, n) in enumerate(modes):
    ax = axes[idx // 2, idx % 2]
    alpha_mn = jn_zeros(m, n)[-1]

    r = np.linspace(0, a, 100)
    theta = np.linspace(0, 2 * np.pi, 200)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    Z = jv(m, alpha_mn * R / a) * np.cos(m * Theta)
    ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    ax.set_title(f'TM$_{{{m}{n}}}$ 모드, $k_c a$ = {alpha_mn:.3f}')
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

plt.suptitle('원형 도파관 헬름홀츠 모드', fontsize=14)
plt.tight_layout()
plt.show()
```

### 6.4 구면좌표에서의 헬름홀츠 방정식

구면좌표에서 변수분리하면 방사 부분이 **구면 베셀 방정식**이 된다:

$$r^2 R'' + 2rR' + [k^2 r^2 - l(l+1)]R = 0$$

해: 구면 베셀 함수 $j_l(kr) = \sqrt{\frac{\pi}{2kr}} J_{l+1/2}(kr)$

$$u_{lm}(r, \theta, \phi) = j_l(kr) Y_l^m(\theta, \phi)$$

---

## 7. 비제차 PDE와 고유함수 전개법

### 7.1 비제차 열방정식

$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2} + F(x, t)$$

여기서 $F(x, t)$는 열원/열흡수를 나타내는 소스 항이다. 제차 경계 조건 $u(0,t) = u(L,t) = 0$ 하에서, 해를 고유함수로 전개한다:

$$u(x, t) = \sum_{n=1}^{\infty} T_n(t) \sin\frac{n\pi x}{L}$$

소스 항도 전개한다:

$$F(x, t) = \sum_{n=1}^{\infty} F_n(t) \sin\frac{n\pi x}{L}$$

대입하면 각 모드에 대한 1차 ODE를 얻는다:

$$T_n'(t) + \alpha^2\left(\frac{n\pi}{L}\right)^2 T_n(t) = F_n(t)$$

이것은 적분 인자 방법으로 풀 수 있다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 비제차 열방정식: 정현파 열원 ---
L, alpha2, N = np.pi, 1.0, 30

# 열원: F(x,t) = sin(x) (공간적으로 불균일한 정상 열원)
# F_n = (2/L) * ∫₀ᴸ sin(x)sin(nx) dx = δ_{n,1}
# T_1' + T_1 = 1, T_1(0) = 0 → T_1(t) = 1 - e^{-t}
# 나머지 T_n(0) = 0, F_n = 0 → T_n(t) = 0

x = np.linspace(0, L, 200)
plt.figure(figsize=(10, 6))
for t in [0.0, 0.1, 0.3, 0.5, 1.0, 3.0]:
    T1 = 1 - np.exp(-t)
    u = T1 * np.sin(x)
    plt.plot(x, u, label=f't = {t}')

plt.plot(x, np.sin(x), 'k--', lw=2, label='정상 상태')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('비제차 열방정식: 정현파 열원')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 7.2 비제차 파동방정식과 공명

비제차 파동방정식에서 외력의 진동수가 고유진동수와 일치하면 **공명(resonance)**이 발생한다. 이때 진폭이 시간에 비례하여 증가한다:

$$u_{tt} = c^2 u_{xx} + F_0 \sin\left(\frac{n\pi x}{L}\right)\cos(\omega_n t)$$

정확한 공명($\omega = \omega_n$)에서 특수해가 $t\sin(\omega_n t)$에 비례하여 선형 성장한다.

---

## 8. 해의 유일성 정리 (Uniqueness Theorems)

### 8.1 에너지 방법

**정리**: 유계 영역 $\Omega$에서 열방정식 $u_t = \alpha^2 \nabla^2 u$의 디리클레 문제의 해는 유일하다.

**증명**: 두 해 $u_1, u_2$가 있다면 차이 $w = u_1 - u_2$는 제차 방정식과 영 경계/초기 조건을 만족한다.

$$E(t) = \int_\Omega w^2 \, dV \geq 0$$

$$\frac{dE}{dt} = 2\int_\Omega w \, w_t \, dV = 2\alpha^2 \int_\Omega w \nabla^2 w \, dV = -2\alpha^2 \int_\Omega |\nabla w|^2 \, dV \leq 0$$

따라서 $E(t) \leq E(0) = 0$이므로 $w \equiv 0$, 즉 $u_1 = u_2$. $\blacksquare$

### 8.2 라플라스 방정식의 유일성

**디리클레 문제**: $\nabla^2 u = 0$ in $\Omega$, $u = f$ on $\partial\Omega$ — 해가 유일하다.

**노이만 문제**: $\nabla^2 u = 0$ in $\Omega$, $\partial u/\partial n = g$ on $\partial\Omega$ — 해가 상수 차이까지 유일하다.

### 8.3 최대값 원리 (Maximum Principle)

**정리**: 영역 $\Omega$에서 조화함수($\nabla^2 u = 0$)의 최댓값과 최솟값은 **경계** $\partial\Omega$에서 달성된다.

**물리적 의미**: 열원/열흡수가 없는 정상 상태에서 내부의 온도는 항상 경계 온도 사이의 값을 가진다. 이것이 경계 조건만으로 내부 전체의 온도를 결정할 수 있는 물리적 이유이다.

---

## 9. 물리학 응용

### 9.1 전자기학: 전위 문제

정전기학의 기본 방정식 $\nabla^2\phi = -\rho/\epsilon_0$. 전하가 없으면 라플라스 방정식이 된다.

**예제**: 한 면이 $V_0$이고 나머지 면이 접지된 직사각형 도체 상자의 내부 전위:

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 직사각형 도체 상자 내부 전위 ---
a, b, c_box, V0, N = 1.0, 1.0, 1.0, 100.0, 15

def potential_box(x, y, z_val):
    u = np.zeros_like(x, dtype=float)
    for m in range(1, N+1, 2):     # 홀수만
        for n in range(1, N+1, 2):
            gamma = np.pi * np.sqrt((m/a)**2 + (n/b)**2)
            Amn = 16*V0 / (m*n*np.pi**2 * np.sinh(gamma*c_box))
            u += Amn * np.sin(m*np.pi*x/a) * np.sin(n*np.pi*y/b) * np.sinh(gamma*z_val)
    return u

x, y = np.linspace(0, a, 100), np.linspace(0, b, 100)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, potential_box(X, Y, 0.5), levels=30, cmap='RdYlBu_r')
plt.colorbar(cs, label='$\\phi$ [V]')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
ax.set_title('직사각형 도체 상자 내부 전위 (z = 0.5)')
plt.tight_layout(); plt.show()
```

### 9.2 양자역학: 무한 퍼텐셜 우물

**시간 독립 슈뢰딩거 방정식**: $-(\hbar^2/2m)\nabla^2\psi + V\psi = E\psi$

**1D 무한 우물** ($0 < x < L$):

$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right), \quad E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$

**2D 정사각형 우물** ($a = b$): $E_{n_x,n_y} = \frac{\pi^2\hbar^2}{2m}(n_x^2 + n_y^2)/a^2$. $(1,2)$와 $(2,1)$이 같은 에너지를 가지는 **축퇴(degeneracy)**가 발생한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 2D 무한 퍼텐셜 우물 ---
a = 1.0  # 자연 단위계: ℏ = m = 1
psi = lambda x, y, nx, ny: 2/a * np.sin(nx*np.pi*x/a) * np.sin(ny*np.pi*y/a)
E = lambda nx, ny: np.pi**2/2 * (nx**2 + ny**2) / a**2

x, y = np.linspace(0, a, 100), np.linspace(0, a, 100)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, (nx, ny) in enumerate([(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]):
    ax = axes[idx//3, idx%3]
    prob = np.abs(psi(X, Y, nx, ny))**2
    cs = ax.contourf(X, Y, prob, levels=30, cmap='inferno')
    plt.colorbar(cs, ax=ax, label='$|\\psi|^2$')
    ax.set_title(f'$(n_x,n_y)=({nx},{ny})$, $E={E(nx,ny)/E(1,1):.1f}\\,E_{{11}}$')
    ax.set_aspect('equal')
fig.suptitle('2D 무한 퍼텐셜 우물의 확률밀도 분포', fontsize=14)
plt.tight_layout(); plt.show()

# 축퇴 확인
print(f"E(1,2) = {E(1,2):.4f}, E(2,1) = {E(2,1):.4f}")
print(f"축퇴: E(1,2) = E(2,1) → {'Yes' if abs(E(1,2)-E(2,1))<1e-10 else 'No'}")
```

**시간 의존 슈뢰딩거 방정식**은 포물형 PDE이다. 변수분리로 $\Psi = \psi(\mathbf{r})e^{-iEt/\hbar}$가 나오고, 일반해는 정상 상태의 중첩: $\Psi(\mathbf{r},t) = \sum_n c_n \psi_n(\mathbf{r})e^{-iE_n t/\hbar}$.

---

## 연습 문제

### 문제 1: PDE 분류

다음 PDE를 분류하라: (a) $u_{xx}+4u_{xy}+u_{yy}=0$ (b) $u_{xx}+2u_{xy}+u_{yy}=0$ (c) $3u_{xx}-6u_{yy}=0$

**힌트**: $\Delta = B^2 - AC$. 교차 항 계수가 $2B$임에 유의.

### 문제 2: 열방정식

$L=\pi$, $\alpha=1$, 양 끝 0°C, 초기 $f(x)=\sin x + 3\sin 2x$일 때:
(a) $u(x,t)$를 구하라. (b) $t=0.5$의 온도를 그려라. (c) $n=2$가 $n=1$보다 몇 배 빠르게 감쇄하는가?

### 문제 3: 파동방정식

$L=1$, $c=2$, $u(x,0)=0$, $u_t(x,0)=\sin(\pi x)$일 때: (a) $u(x,t)$. (b) 기본 진동수.

### 문제 4: 라플라스 방정식

단위 원판에서 $\nabla^2 u = 0$, $u(1,\theta) = \cos^2\theta$일 때:
(a) $\cos^2\theta$의 푸리에 전개. (b) $u(r,\theta)$. (c) 원판 중심의 온도와 물리적 의미.

### 문제 5: 양자역학

2D 정사각형 우물에서: (a) 처음 5개 에너지 준위와 축퇴도. (b) $(1,2)$와 $(2,1)$의 $|\psi|^2$ 비교.

### 문제 6: 헬름홀츠 방정식

반지름 $a$의 원형 영역에서 $\nabla^2 u + k^2 u = 0$, $u(a, \theta) = 0$. 축대칭($m=0$) 고유값 $k_{0n}$과 고유함수를 구하시오.

### 문제 7: 비제차 PDE

$u_t = u_{xx} + 2\sin(3\pi x)$, $u(0,t) = u(1,t) = 0$, $u(x,0) = \sin(\pi x)$의 해를 고유함수 전개법으로 구하시오.

### 문제 8: 유일성

에너지 방법을 사용하여, 파동방정식 $u_{tt} = c^2 u_{xx}$ ($0 < x < L$)에서 디리클레 경계 조건과 초기 조건이 주어졌을 때 해가 유일함을 증명하시오. (힌트: $E(t) = \int_0^L (w_t^2 + c^2 w_x^2) dx$를 사용)

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 13. Wiley.
2. **Haberman, R.** (2013). *Applied Partial Differential Equations*, 5th ed. Pearson.
3. **Strauss, W. A.** (2008). *Partial Differential Equations: An Introduction*, 2nd ed. Wiley.
4. **Jackson, J. D.** (1999). *Classical Electrodynamics*, 3rd ed. Wiley. — 헬름홀츠 방정식의 전자기학 응용

### 연관 레슨
- [07. 푸리에 급수](07_Fourier_Series.md): 변수분리법의 핵심 도구
- [11. 급수해와 특수함수](11_Series_Solutions_Special_Functions.md): 베셀/르장드르 함수
- [12. 스투름-리우빌 이론](12_Sturm_Liouville_Theory.md): 고유값 문제의 일반적 틀
- [16. 그린 함수](16_Greens_Functions.md): 비제차 PDE의 체계적 풀이법

---

## 다음 레슨

[14. 복소해석 (Complex Analysis)](14_Complex_Analysis.md)에서는 복소 변수 함수의 미분과 적분, 코시-리만 조건, 등각사상, 유수 정리를 다루며, PDE의 경계값 문제를 등각사상으로 변환하는 기법도 살펴봅니다.

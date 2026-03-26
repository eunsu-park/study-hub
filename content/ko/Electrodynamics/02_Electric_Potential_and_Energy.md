# 전기 퍼텐셜과 에너지

[← 이전: 01. 정전기학 복습](01_Electrostatics_Review.md) | [다음: 03. 도체와 유전체 →](03_Conductors_and_Dielectrics.md)

---

## 학습 목표

1. 스칼라 퍼텐셜(scalar potential) $V$를 정의하고 전기장 $\mathbf{E}$와의 관계를 유도한다
2. 점전하와 연속 전하 분포로부터 퍼텐셜을 계산한다
3. 푸아송 방정식(Poisson's equation)과 라플라스 방정식(Laplace's equation)을 유도하고 해석한다
4. 정전기 경계값 문제에 대한 유일성 정리(uniqueness theorems)를 서술하고 적용한다
5. 정전기 배치에 저장된 에너지를 계산한다
6. 정전기 에너지를 장의 에너지 밀도로 표현한다
7. 완화법(relaxation method)으로 라플라스 방정식의 수치해를 구현한다

---

전기장은 벡터다 — 공간의 모든 점에서 세 성분을 갖는다. 그러나 정전기학에서 $\nabla \times \mathbf{E} = 0$이기 때문에, 장의 모든 정보는 단 하나의 스칼라 함수인 전기 퍼텐셜(electric potential) $V$에 담겨 있다. 이 단순화는 단순히 계산상의 편의가 아니다 — 깊은 물리적 의미를 담고 있다. 퍼텐셜은 에너지와 직결되며, 에너지 논증은 흔히 전하가 왜 그 배치를 취하는지 이해하는 가장 빠른 경로가 된다. 이 레슨에서는 퍼텐셜과 에너지의 도구를 개발하며, 그 정점에 있는 푸아송 방정식 — 일단 풀면 모든 것을 알려주는 근본 방정식 — 으로 마무리한다.

---

## 스칼라 퍼텐셜

$\nabla \times \mathbf{E} = 0$이므로, 전기장은 어떤 스칼라의 기울기(gradient)로 쓸 수 있다:

$$\mathbf{E} = -\nabla V$$

음수 부호는 $\mathbf{E}$가 높은 퍼텐셜에서 낮은 퍼텐셜로 향하도록 (공이 언덕 아래로 굴러가듯이) 하는 약속이다.

두 점 $\mathbf{a}$와 $\mathbf{b}$ 사이의 퍼텐셜 차이는:

$$V(\mathbf{b}) - V(\mathbf{a}) = -\int_{\mathbf{a}}^{\mathbf{b}} \mathbf{E} \cdot d\mathbf{l}$$

이 적분은 **경로에 무관하다** ($\nabla \times \mathbf{E} = 0$이기 때문에). 이것이 $V$가 잘 정의되는 이유이다.

> **비유**: 전기 퍼텐셜은 지형도의 고도와 같다. 전기장은 '경사' — 위에서 아래로 (높은 $V$에서 낮은 $V$로) 향하며, 그 크기가 얼마나 가파른지를 나타낸다. 물이 낮은 곳으로 흐르듯이, 양전하는 높은 퍼텐셜에서 낮은 퍼텐셜로 '흐른다'.

### 점전하의 퍼텐셜

무한대에서 $V = 0$으로 설정하면 (표준 기준점):

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \frac{q}{|\mathbf{r} - \mathbf{r}'|}$$

여러 점전하의 경우:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_{i=1}^{N} \frac{q_i}{|\mathbf{r} - \mathbf{r}_i'|}$$

연속 분포의 경우:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\tau'$$

주목할 점: 퍼텐셜은 $1/r$ (스칼라)이지 $1/r^2$ (벡터)가 아니다. 스칼라 적분은 벡터 적분보다 훨씬 계산하기 쉽다. 이것이 퍼텐셜로 작업하는 주된 계산상 이점이다: 먼저 $V$를 구하고 (스칼라 적분 하나), 그 다음 미분하여 $\mathbf{E} = -\nabla V$를 얻는다 (편미분 세 번). 대안인 $\mathbf{E}$를 벡터 적분으로 직접 구하는 방법은 $\hat{\boldsymbol{\mathscr{r}}}$의 방향을 포함하는 세 개의 별도 적분을 필요로 한다.

### 단위와 차원

퍼텐셜 $V$의 단위는 볼트(V = J/C)이다. 단위 전하당 퍼텐셜 에너지를 나타낸다:

$$V = \frac{U}{q} \qquad [\text{V} = \text{J/C} = \text{kg}\cdot\text{m}^2/(\text{A}\cdot\text{s}^3)]$$

전기장의 단위는 V/m으로, $\mathbf{E} = -\nabla V$ (기울기 연산이 $1/\text{m}$를 도입)와 일치한다.

### 등퍼텐셜 면

$V$가 일정한 면을 **등퍼텐셜 면(equipotential surface)**이라 한다. 주요 성질:
- $\mathbf{E}$는 어디서나 등퍼텐셜 면에 수직이다
- 등퍼텐셜 면을 따라 전하를 이동시킬 때 한 일은 0이다
- 평형 상태의 도체는 등퍼텐셜체이다 (레슨 3에서 증명)

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize equipotential surfaces for a dipole
# Why contour plot: equipotentials are curves of constant V in 2D

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

# Dipole: +q at (-d/2, 0), -q at (+d/2, 0)
q = 1e-9
d = 0.2
charges = [(-d/2, 0, q), (d/2, 0, -q)]

x = np.linspace(-0.6, 0.6, 400)
y = np.linspace(-0.6, 0.6, 400)
X, Y = np.meshgrid(x, y)

V = np.zeros_like(X)
for (cx, cy, qi) in charges:
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r = np.maximum(r, 1e-4)  # avoid singularity at charge locations
    V += k_e * qi / r

# Why clip: potential diverges near charges; clipping makes contours visible
V_clipped = np.clip(V, -500, 500)

fig, ax = plt.subplots(figsize=(8, 8))
levels = np.linspace(-400, 400, 41)
cs = ax.contour(X, Y, V_clipped, levels=levels, cmap='RdBu_r')
ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Equipotential Lines of an Electric Dipole')
ax.set_aspect('equal')

for (cx, cy, qi) in charges:
    color = 'red' if qi > 0 else 'blue'
    ax.plot(cx, cy, 'o', color=color, markersize=10)

plt.tight_layout()
plt.savefig('dipole_equipotentials.png', dpi=150)
plt.show()
```

---

## 푸아송 방정식과 라플라스 방정식

$\mathbf{E} = -\nabla V$와 가우스 법칙 $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$를 결합하면:

$$\nabla \cdot (-\nabla V) = \frac{\rho}{\epsilon_0}$$

$$\boxed{\nabla^2 V = -\frac{\rho}{\epsilon_0}} \qquad \text{(푸아송 방정식)}$$

전하가 없는 영역 ($\rho = 0$)에서:

$$\boxed{\nabla^2 V = 0} \qquad \text{(라플라스 방정식)}$$

이것들은 **2계 편미분 방정식(second-order partial differential equations)**이다. 푸아송 방정식은 정전기학의 핵심 방정식으로, 적절한 경계 조건과 함께 풀면 모든 곳에서의 퍼텐셜(과 전기장)을 알 수 있다.

### 다양한 좌표계에서의 라플라시안

| 좌표계 | $\nabla^2 V$ |
|---|---|
| 직교 좌표계(Cartesian) | $\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} + \frac{\partial^2 V}{\partial z^2}$ |
| 구면 좌표계(Spherical) | $\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial V}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial V}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 V}{\partial \phi^2}$ |
| 원통 좌표계(Cylindrical) | $\frac{1}{s}\frac{\partial}{\partial s}\left(s\frac{\partial V}{\partial s}\right) + \frac{1}{s^2}\frac{\partial^2 V}{\partial \phi^2} + \frac{\partial^2 V}{\partial z^2}$ |

### 라플라스 방정식의 주요 성질

1. **극값 없음**: $\nabla^2 V = 0$을 만족하는 함수는 정의 영역 내부에 극대나 극소를 가질 수 없다. 이것이 **최대 원리(maximum principle)**이다 — 최댓값과 최솟값은 경계에서만 나타난다.

2. **평균값 성질**: 임의의 점에서 $V$의 값은 그 점을 중심으로 하는 임의의 구 위의 $V$ 평균값과 같다 (구 내부에 전하가 없는 경우):

$$V(\mathbf{r}_0) = \frac{1}{4\pi R^2} \oint_{\text{sphere}} V \, da$$

이 성질들은 수학적으로 우아할 뿐 아니라, 수치적 완화법의 이론적 기반이 된다.

---

## 유일성 정리

어떻게 우리가 찾은 해가 '올바른' 해임을 알 수 있을까? 유일성 정리(uniqueness theorems)가 이를 보장한다.

### 제1 유일성 정리

다음 조건이 주어지면 부피 $\mathcal{V}$에서의 퍼텐셜 $V$는 유일하게 결정된다:
1. 전하 밀도 $\rho$가 $\mathcal{V}$ 전체에서 지정되고,
2. $V$의 값이 경계면 $\mathcal{S}$ 위에서 지정된다 (디리클레 경계 조건(Dirichlet boundary condition))

**증명 개요**: $V_1$과 $V_2$가 모두 해라고 가정하자. 그 차이 $V_3 = V_1 - V_2$는 $\mathcal{V}$ 내에서 $\nabla^2 V_3 = 0$을 만족하고 $\mathcal{S}$ 위에서 $V_3 = 0$이다. 최대 원리에 의해 $V_3 = 0$이 어디서나 성립하므로, $V_1 = V_2$이다.

### 제2 유일성 정리

도체들로 둘러싸인 부피에서, 각 도체의 총 전하가 지정되면 전기장은 유일하게 결정된다. (이는 노이만 경계 조건(Neumann boundary condition) — $V$ 자체가 아닌 $\partial V/\partial n$을 지정하는 것 — 을 허용한다.)

이 정리들은 대단히 강력하다: 해를 찾는 **어떤** 방법을 사용하든 — 추측, 대칭성 논증, 수치 계산 — 그 결과가 **유일한** 해임을 보장한다.

---

## 수치해: 완화법

평균값 성질로부터 수치 알고리즘을 도출할 수 있다. 공간을 격자로 이산화하고, 각 격자점을 이웃한 점들의 평균으로 반복적으로 업데이트한다:

```python
import numpy as np
import matplotlib.pyplot as plt

# Solve Laplace's equation in 2D using the Jacobi relaxation method
# Why relaxation: it directly exploits the mean-value property of harmonic functions

# Problem: square region, V=100 on top edge, V=0 on other edges
N = 100                          # grid points per side
V = np.zeros((N, N))

# Boundary conditions — these drive the entire solution
V[0, :] = 100.0     # top edge at 100 V
V[-1, :] = 0.0      # bottom edge at 0 V
V[:, 0] = 0.0       # left edge at 0 V
V[:, -1] = 0.0      # right edge at 0 V

# Relaxation iterations
# Why 5000 iterations: convergence is slow for Jacobi; SOR would be faster
n_iter = 5000
for iteration in range(n_iter):
    V_old = V.copy()
    # Update interior points: each point becomes average of 4 neighbors
    # Why average of neighbors: this is the discrete form of ∇²V = 0
    V[1:-1, 1:-1] = 0.25 * (
        V_old[0:-2, 1:-1] +    # top neighbor
        V_old[2:, 1:-1] +      # bottom neighbor
        V_old[1:-1, 0:-2] +    # left neighbor
        V_old[1:-1, 2:]        # right neighbor
    )
    # Re-enforce boundary conditions (they must not change)
    V[0, :] = 100.0
    V[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0

    # Check convergence
    if iteration % 1000 == 0:
        diff = np.max(np.abs(V - V_old))
        print(f"Iteration {iteration}: max change = {diff:.2e}")

# Plot the solution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot of potential
cs = axes[0].contourf(V, levels=50, cmap='hot')
plt.colorbar(cs, ax=axes[0], label='V (volts)')
axes[0].set_title("Potential V (Laplace's equation)")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Electric field (negative gradient of V)
# Why np.gradient: numerical differentiation to get E from V
Ey, Ex = np.gradient(-V)   # note: gradient returns (row, col) = (y, x)
E_mag = np.sqrt(Ex**2 + Ey**2)

# Subsample for cleaner arrows
step = 5
xx = np.arange(0, N, step)
yy = np.arange(0, N, step)
XX, YY = np.meshgrid(xx, yy)

axes[1].quiver(XX, YY, Ex[::step, ::step], Ey[::step, ::step],
               E_mag[::step, ::step], cmap='viridis', scale=500)
axes[1].set_title('Electric Field E = -∇V')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.suptitle("Solving Laplace's Equation by Relaxation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplace_relaxation.png', dpi=150)
plt.show()
```

---

## 정전기 에너지

### 점전하 배치의 에너지

점전하들을 무한대에서 현재 위치로 가져오는 데 필요한 에너지:

$$W = \frac{1}{2} \sum_{i=1}^{N} \sum_{\substack{j=1 \\ j \neq i}}^{N} \frac{q_i q_j}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{r}_j|}$$

$1/2$는 각 쌍을 두 번 세는 것을 보정하기 위한 인수이다. 이를 동등하게 쓰면:

$$W = \frac{1}{2} \sum_{i=1}^{N} q_i V(\mathbf{r}_i)$$

여기서 $V(\mathbf{r}_i)$는 다른 모든 전하들에 의해 $q_i$ 위치에서의 퍼텐셜이다.

### 연속 분포의 에너지

부피 전하 밀도의 경우:

$$W = \frac{1}{2} \int \rho \, V \, d\tau$$

### 장으로 표현한 에너지

가우스 법칙을 이용하면 에너지를 전적으로 전기장으로 나타낼 수 있다:

$$\boxed{W = \frac{\epsilon_0}{2} \int_{\text{all space}} |\mathbf{E}|^2 \, d\tau}$$

이것은 놀라운 결과이다. 에너지는 전하들 '사이'가 아니라 **장 자체에** 저장된다. **에너지 밀도(energy density)**는:

$$u = \frac{\epsilon_0}{2} E^2 \quad [\text{J/m}^3]$$

> **비유**: 팽팽하게 늘어난 고무 시트를 생각해 보자. 에너지는 그것을 고정하는 기둥들이 아니라 시트의 탄성 변형 속에 저장된다. 마찬가지로, 정전기 에너지는 공간 전체에 걸쳐 전기장의 '변형' 속에 저장된다.

### 유도 개요

$W = \frac{1}{2}\int \rho V \, d\tau$와 $\rho = \epsilon_0 \nabla \cdot \mathbf{E}$를 이용하면:

$$W = \frac{\epsilon_0}{2}\int V (\nabla \cdot \mathbf{E}) \, d\tau$$

곱 규칙 $\nabla \cdot (V\mathbf{E}) = V(\nabla \cdot \mathbf{E}) + \mathbf{E} \cdot (\nabla V)$와 $\nabla V = -\mathbf{E}$를 적용하면:

$$W = \frac{\epsilon_0}{2}\left[\int \nabla \cdot (V\mathbf{E}) \, d\tau + \int E^2 \, d\tau \right]$$

첫 번째 적분은 (발산 정리에 의해) 면 적분이 되며, 곡면을 무한대로 밀어내면 소멸한다 ($V \sim 1/r$이고 $E \sim 1/r^2$이므로 $VE \sim 1/r^3$이지만 $da \sim r^2$). 남는 것은:

$$W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$$

```python
import numpy as np

# Compute electrostatic energy of a uniformly charged sphere two ways
# Why two methods: comparing charge-based and field-based gives confidence

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

Q = 1e-9      # total charge (1 nC)
R = 0.1       # sphere radius (10 cm)

# Method 1: Assembly energy (bringing shells of charge from infinity)
# W = (3/5) * kQ²/R — classic result for uniform sphere
W_assembly = (3 / 5) * k_e * Q**2 / R
print(f"Assembly energy:  W = {W_assembly:.6e} J")

# Method 2: Field energy — integrate (ε₀/2)E² over all space
# Inside: E = kQr/R³, Outside: E = kQ/r²
# Why split integral: E has different functional forms inside and outside

N_r = 100000
r_inner = np.linspace(1e-6, R, N_r)
r_outer = np.linspace(R, 100 * R, N_r)  # integrate far enough

dr_in = r_inner[1] - r_inner[0]
dr_out = r_outer[1] - r_outer[0]

# Inside the sphere: E(r) = kQr/R³
E_in = k_e * Q * r_inner / R**3
u_in = 0.5 * epsilon_0 * E_in**2
# Why 4πr²: spherical shell volume element in radial integration
W_in = np.sum(u_in * 4 * np.pi * r_inner**2 * dr_in)

# Outside the sphere: E(r) = kQ/r²
E_out = k_e * Q / r_outer**2
u_out = 0.5 * epsilon_0 * E_out**2
W_out = np.sum(u_out * 4 * np.pi * r_outer**2 * dr_out)

W_field = W_in + W_out
print(f"Field energy:     W = {W_field:.6e} J")
print(f"Relative error:   {abs(W_field - W_assembly)/W_assembly:.4f}")
print(f"\nEnergy density at surface: {0.5*epsilon_0*(k_e*Q/R**2)**2:.4e} J/m³")
```

---

## 자기 에너지 문제

미묘한 문제가 하나 있다: 점전하의 에너지는 **무한대**이다. $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$에 $E = kq/r^2$를 사용하면:

$$W = \frac{\epsilon_0}{2} \int_0^\infty \left(\frac{q}{4\pi\epsilon_0 r^2}\right)^2 4\pi r^2 \, dr = \frac{q^2}{8\pi\epsilon_0} \int_0^\infty \frac{dr}{r^2} = \infty$$

이 '자기 에너지(self-energy)' 발산은 고전 전자기학의 진지한 문제이다. 해결책으로는:
- 전하에 유한한 크기를 부여하는 것 (고전적 전자 반지름(classical electron radius) $r_e = e^2/(4\pi\epsilon_0 m_e c^2) \approx 2.8$ fm)
- 양자 전기역학(quantum electrodynamics)의 재규격화(renormalization)

실용적인 목적에서는 자기 에너지를 단순히 제외하고 전하들 사이의 **상호 에너지(interaction energy)**만 계산한다.

고전적 전자 반지름 $r_e = e^2/(4\pi\epsilon_0 m_e c^2) \approx 2.82$ fm은 고전 전자기학이 무너지는 스케일을 정해준다. $r_e$보다 짧은 거리에서는 전자 장의 자기 에너지가 전자의 정지 질량 에너지를 초과하게 된다 — 양자역학이 반드시 개입해야 한다는 명백한 신호이다.

> **비유**: 자기 에너지 문제는 "동전 하나를 모으는 데 얼마가 드는가?"라고 묻는 것과 같다 — 동전 한 무더기를 모으는 비용을 묻는 것과 같은 방식으로 의미가 잘 통하지 않는다. 에너지 개념은 전하들 사이의 상호 작용에는 완벽하게 작동하지만, 단독 점전하에는 문제가 생긴다.

---

## 표준 배치에서의 퍼텐셜

### 대전된 원판

균일한 면전하 밀도 $\sigma$를 가진 반지름 $R$인 원판. 높이 $z$인 축 위의 점에서:

$$V(z) = \frac{\sigma}{2\epsilon_0}\left(\sqrt{z^2 + R^2} - |z|\right)$$

$z \gg R$이면: $V \approx \frac{Q}{4\pi\epsilon_0 z}$ (점전하처럼 보임, $Q = \sigma\pi R^2$)

$z = 0$이면: $V = \frac{\sigma R}{2\epsilon_0}$ (유한한 값!)

### 대전된 선분

$z$축을 따라 원점을 중심으로 길이 $2L$, 선전하 밀도 $\lambda$인 균일한 선전하. 중간 평면 위에서 수직 거리 $s$에서의 퍼텐셜:

$$V(s) = \frac{\lambda}{4\pi\epsilon_0}\ln\left(\frac{L + \sqrt{L^2 + s^2}}{s}\right) \cdot 2$$

$s \gg L$이면: $V \approx \frac{2L\lambda}{4\pi\epsilon_0 s} = \frac{Q}{4\pi\epsilon_0 s}$ (먼 거리에서는 점전하)

$s \ll L$이면: $V \approx \frac{\lambda}{2\pi\epsilon_0}\ln(2L/s)$ (무한 선전하처럼 로그함수)

```python
import numpy as np
import matplotlib.pyplot as plt

# Potential of a finite line charge — transition from logarithmic to 1/r
# Why study this: it shows how finite geometry interpolates between ideal limits

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

lam = 1e-9        # linear charge density (1 nC/m)
L = 0.2           # half-length of line charge (20 cm)
Q = 2 * L * lam   # total charge

s = np.linspace(0.01, 2.0, 500)

# Exact potential at perpendicular distance s on the midplane
V_exact = 2 * k_e * lam * np.log((L + np.sqrt(L**2 + s**2)) / s)

# Approximation 1: point charge (valid for s >> L)
V_point = k_e * Q / s

# Approximation 2: infinite line (valid for s << L)
# Why reference at s=L: infinite line has arbitrary reference, we match at s=L
V_inf_line = k_e * 2 * lam * np.log(L / s)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s * 100, V_exact, 'b-', linewidth=2, label='Exact')
ax.plot(s * 100, V_point, 'r--', linewidth=1.5, label='Point charge approx')
ax.plot(s * 100, V_inf_line, 'g:', linewidth=1.5, label='Infinite line approx')
ax.axvline(x=L * 100, color='gray', linestyle='--', alpha=0.5, label=f'L = {L*100:.0f} cm')
ax.set_xlabel('s (cm)')
ax.set_ylabel('V (V)')
ax.set_title('Potential of Finite Line Charge')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_charge_potential.png', dpi=150)
plt.show()
```

### 구형 껍질

총 전하 $Q$를 가진 반지름 $R$인 껍질:

$$V(r) = \begin{cases} \frac{Q}{4\pi\epsilon_0 R} & r < R \text{ (내부에서 일정)} \\ \frac{Q}{4\pi\epsilon_0 r} & r > R \end{cases}$$

내부의 일정한 퍼텐셜은 $\mathbf{E} = 0$임과 일치한다 ($\mathbf{E} = -\nabla V$이고 $\nabla(\text{상수}) = 0$이므로).

---

## V의 경계 조건

면전하 $\sigma$를 가진 계면에서:

$$V_{\text{above}} = V_{\text{below}} \quad \text{($V$는 연속이다)}$$

$$\frac{\partial V_{\text{above}}}{\partial n} - \frac{\partial V_{\text{below}}}{\partial n} = -\frac{\sigma}{\epsilon_0} \quad \text{(법선 도함수는 불연속이다)}$$

이 조건들은 푸아송 방정식 또는 라플라스 방정식과 함께 $V$를 유일하게 결정한다.

---

## 일과 퍼텐셜 차이

점 $\mathbf{a}$에서 점 $\mathbf{b}$로 전하 $q$를 이동시킬 때 전기장이 하는 일:

$$W_{a \to b} = q \int_{\mathbf{a}}^{\mathbf{b}} \mathbf{E} \cdot d\mathbf{l} = q[V(\mathbf{a}) - V(\mathbf{b})]$$

핵심 사항:
- **양의 일**은 전기장이 전하에 일을 한다는 의미이다 (양전하는 높은 퍼텐셜에서 낮은 퍼텐셜로 이동)
- 일은 **경로에 무관하다** — 처음과 끝 위치만이 중요하다
- **전자볼트(eV)** = $1.6 \times 10^{-19}$ J는 전자 하나를 1볼트의 전위차로 이동시킬 때 하는 일이다

### 회로와의 연결

기전력(electromotive force, EMF) $\mathcal{E}$를 가진 전지가 있는 회로에서:
- 전지는 단자 양단에 전위차 $\Delta V = \mathcal{E}$를 유지한다
- 전류는 외부 회로에서 높은 퍼텐셜에서 낮은 퍼텐셜로 흐른다
- 전지는 통과하는 전하에 일 $W = q\mathcal{E}$를 한다
- 전달 전력: $P = IV = I^2 R = V^2/R$ (줄의 법칙(Joule's law))

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 퍼텐셜의 정의 | $\mathbf{E} = -\nabla V$ |
| 점전하의 퍼텐셜 | $V = q/(4\pi\epsilon_0 r)$ |
| 푸아송 방정식 | $\nabla^2 V = -\rho/\epsilon_0$ |
| 라플라스 방정식 | $\nabla^2 V = 0$ |
| 에너지 (이산) | $W = \frac{1}{2}\sum_i q_i V(\mathbf{r}_i)$ |
| 에너지 (장) | $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$ |
| 에너지 밀도 | $u = \frac{\epsilon_0}{2}E^2$ |
| 평균값 성질 | $V(\mathbf{r}_0)$ = 주변 구 위의 평균 |
| 유일성 | $\rho$ + 경계 조건이 주어지면 해는 유일 |

---

## 연습 문제

### 연습 문제 1: 퍼텐셜과 전기장 계산
반지름 $R = 0.15$ m인 얇은 링에 총 전하 $Q = 2$ nC이 분포해 있다. 축 위의 $V(z)$와 $E_z(z)$를 계산하고 그래프로 나타내라. $E_z = -dV/dz$가 수치적으로 성립함을 검증하라.

### 연습 문제 2: 복잡한 경계 조건에서의 라플라스 방정식 풀기
완화법 코드를 수정하여 다음 경계 조건을 가진 정사각형 영역에서 라플라스 방정식을 풀어라: 위쪽 경계에서 $V = 100\sin(\pi x/L)$, 나머지 경계에서 $V = 0$. 수치해를 해석적 변수분리법 결과 $V(x, y) = 100\sin(\pi x/L)\sinh(\pi y/L)/\sinh(\pi)$와 비교하라.

### 연습 문제 3: 동심 구형 껍질의 에너지
반지름 $a$와 $b$ ($a < b$)인 두 동심 구형 껍질이 각각 전하 $Q_a$와 $Q_b$를 띠고 있다. 장 방법으로 전체 정전기 에너지를 계산하라. $Q_a = -Q_b$ (축전기)인 특수한 경우를 검증하라.

### 연습 문제 4: 수치적 푸아송 방정식 풀기
국소 전하 분포 $\rho(x,y) = \rho_0 \exp(-(x^2+y^2)/w^2)$를 가진 푸아송 방정식을 완화법으로 풀 수 있도록 코드를 확장하라. 결과로 얻은 퍼텐셜과 전기장을 그래프로 나타내라.

### 연습 문제 5: 다중극 전개
$(0, 0, d)$에 $+q$, 원점에 $-2q$, $(0, 0, -d)$에 $+q$로 이루어진 전하 분포에 대한 다중극 전개의 단극(monopole), 쌍극자(dipole), 사중극자(quadrupole) 항을 계산하라. 이것은 **선형 사중극자(linear quadrupole)**이다. 원거리 장에서의 선두 항은 무엇인가?

---

[← 이전: 01. 정전기학 복습](01_Electrostatics_Review.md) | [다음: 03. 도체와 유전체 →](03_Conductors_and_Dielectrics.md)

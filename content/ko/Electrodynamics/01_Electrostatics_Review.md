# 정전기학 복습

[다음: 02. 전기 퍼텐셜과 에너지 →](02_Electric_Potential_and_Energy.md)

---

## 학습 목표

1. 쿨롱 법칙(Coulomb's law)을 서술하고, 점전하 사이의 힘을 벡터 형태로 계산한다
2. 전기장(electric field)을 정의하고, 여러 전하에 중첩 원리(superposition principle)를 적용한다
3. 연속 전하 분포(선, 면, 부피)에서 전기장을 계산한다
4. 가우스 법칙(Gauss's law)을 적분 형태와 미분 형태로 유도하고 적용한다
5. 높은 대칭성(구형, 원통형, 평면형)을 갖는 경우에 가우스 법칙으로 전기장을 구한다
6. 정전기장의 발산(divergence)과 회전(curl)을 이해한다
7. Python으로 전하 분포에 의한 전기장을 수치적으로 계산한다

---

정전기학(electrostatics)은 전자기학 전체를 떠받치는 기반이다. 전자기파, 복사, 맥스웰 방정식의 완전한 의미를 이해하려면 먼저 정지해 있는 전하의 물리학을 완전히 익혀야 한다. 이 레슨에서는 정적 전기장을 지배하는 기본 법칙들 — 쿨롱 법칙, 중첩 원리, 가우스 법칙 — 을 복습하고, 벡터 미적분의 정밀한 수학적 언어로 발전시킨다. 여기서 등장하는 모든 방정식은 이후 레슨에서 더 일반화되고 심화된 형태로 다시 나타날 것이다.

---

## 쿨롱 법칙

정전기학의 출발점은 두 점전하가 서로 힘을 주고받는다는 실험적 관찰이다. 쿨롱 법칙은 전하 $q_1$이 전하 $q_2$에 작용하는 힘을 다음과 같이 나타낸다:

$$\mathbf{F}_{12} = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{|\mathbf{r}_2 - \mathbf{r}_1|^2} \hat{\mathbf{r}}_{12}$$

각 기호의 의미:
- $\epsilon_0 \approx 8.854 \times 10^{-12}$ C$^2$/(N$\cdot$m$^2$)는 진공의 유전율(permittivity of free space)
- $\mathbf{r}_1, \mathbf{r}_2$는 두 전하의 위치 벡터
- $\hat{\mathbf{r}}_{12} = \frac{\mathbf{r}_2 - \mathbf{r}_1}{|\mathbf{r}_2 - \mathbf{r}_1|}$는 $q_1$에서 $q_2$를 향하는 단위 벡터

상수 $k_e = \frac{1}{4\pi\epsilon_0} \approx 8.988 \times 10^9$ N$\cdot$m$^2$/C$^2$는 쿨롱 상수로, 표기를 간략하게 할 때 자주 쓰인다.

### 역사적 배경

샤를-오귀스탱 드 쿨롱(Charles-Augustin de Coulomb)은 1785년 비틀림 저울을 이용해 이 법칙을 실험적으로 확립했다. 그가 이룬 측정의 정밀도는 당시로서는 놀라운 수준으로, 역제곱 의존성을 수 퍼센트 이내의 오차로 확인했다. 현대의 실험들은 지수를 $2 \pm 10^{-16}$ 수준으로 제한하고 있으며, 쿨롱 법칙은 물리학 전체에서 가장 정밀하게 검증된 법칙 중 하나이다.

### 중력과의 비교

| 성질 | 쿨롱(전기력) | 뉴턴(중력) |
|---|---|---|
| 힘의 법칙 | $F \propto q_1 q_2/r^2$ | $F \propto m_1 m_2/r^2$ |
| 원천의 부호 | $+$와 $-$ 모두 존재 | $+$만 존재 (질량 > 0) |
| 인력/척력 | 둘 다 | 인력만 |
| 상대적 세기 | 중력보다 $\sim 10^{36}$배 강함 | 1 (기준) |
| 매개 장 | 전기장 $\mathbf{E}$ | 중력장 $\mathbf{g}$ |

양성자와 전자 사이의 전기력과 중력의 비는 $F_e/F_g \approx 2.3 \times 10^{39}$에 달한다 — 원자 스케일에서는 전자기력이 압도적으로 지배적이다.

**핵심 성질:**
- 이 힘은 **중심력(central force)** 이다 — 두 전하를 잇는 직선 방향으로 작용한다
- **뉴턴의 제3법칙**을 따른다: $\mathbf{F}_{12} = -\mathbf{F}_{21}$
- 같은 부호의 전하는 척력($F > 0$), 다른 부호의 전하는 인력($F < 0$)
- 중력과 마찬가지로 **역제곱 법칙(inverse-square law)**을 따른다

> **비유**: 쿨롱 법칙은 전기력 버전의 중력이라고 생각하면 된다. 둘 다 '원천'(중력은 질량, 전기력은 전하) 사이에 작용하는 역제곱 힘이다. 결정적 차이는 전하가 양과 음 두 종류가 있어 인력과 척력이 모두 가능하다는 점이다 — 중력은 오직 인력만 작용한다.

---

## 전기장

전하 쌍 사이의 힘을 직접 생각하는 대신, 단위 양전하에 작용하는 힘으로 **전기장(electric field)** $\mathbf{E}$를 정의한다:

$$\mathbf{E}(\mathbf{r}) = \frac{\mathbf{F}}{q_{\text{test}}} = \frac{1}{4\pi\epsilon_0} \frac{q}{|\mathbf{r} - \mathbf{r}'|^2} \hat{\mathbf{r}}$$

여기서 $\mathbf{r}'$는 원천 전하의 위치, $\mathbf{r}$는 장을 계산하는 관측점이다.

전기장은 **벡터장(vector field)** 이다 — 공간의 모든 점에 벡터를 대응시킨다. 이것은 깊은 개념적 전환을 의미한다: 장은 시험 전하가 없어도 공간에 독립적으로 존재한다.

### 중첩 원리

위치 $\mathbf{r}_1', \mathbf{r}_2', \ldots, \mathbf{r}_N'$에 있는 $N$개의 점전하 $q_1, q_2, \ldots, q_N$에 의해 점 $\mathbf{r}$에서의 전체 전기장은 벡터 합으로 주어진다:

$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_{i=1}^{N} \frac{q_i}{|\mathbf{r} - \mathbf{r}_i'|^2} \hat{\boldsymbol{\mathscr{r}}}_i$$

여기서 $\hat{\boldsymbol{\mathscr{r}}}_i = \frac{\mathbf{r} - \mathbf{r}_i'}{|\mathbf{r} - \mathbf{r}_i'|}$이다.

중첩은 **정확히** 성립한다 — 보정항이나 고차항이 없다. 각 전하가 만드는 장은 다른 전하들과 완전히 독립적이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute electric field from multiple point charges using superposition
# Why NumPy: vectorized operations let us evaluate the field on a grid efficiently

k_e = 8.988e9  # Coulomb constant (N*m^2/C^2)

# Define charges: (x, y, charge) — a dipole-like configuration
charges = [
    (−0.5, 0.0, 1e-9),   # +1 nC at (-0.5, 0)
    ( 0.5, 0.0, -1e-9),   # -1 nC at (+0.5, 0)
]

# Create a 2D grid of field points
# Why meshgrid: we need E evaluated at every (x,y) point for visualization
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

# Superposition: sum contributions from each charge
for (qx, qy, q) in charges:
    dx = X - qx                       # displacement vectors (x-component)
    dy = Y - qy                       # displacement vectors (y-component)
    r_sq = dx**2 + dy**2              # squared distance
    r_sq = np.maximum(r_sq, 1e-6)     # avoid division by zero near charges
    r = np.sqrt(r_sq)
    # Why we divide by r^3: E ~ q*r_hat/r^2 = q*(r_vec/r)/r^2 = q*r_vec/r^3
    Ex += k_e * q * dx / r_sq**(3/2)
    Ey += k_e * q * dy / r_sq**(3/2)

# Visualize with streamlines — they follow the direction of E
E_mag = np.sqrt(Ex**2 + Ey**2)
fig, ax = plt.subplots(figsize=(8, 8))
ax.streamplot(X, Y, Ex, Ey, color=np.log10(E_mag), cmap='inferno', density=2)
for (qx, qy, q) in charges:
    color = 'red' if q > 0 else 'blue'
    ax.plot(qx, qy, 'o', color=color, markersize=12)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Electric Field Lines of a Dipole')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('dipole_field.png', dpi=150)
plt.show()
```

---

## 연속 전하 분포

실제 전하 분포는 흔히 연속적이다. 이 경우 이산 합을 적분으로 바꾼다. 차원에 따라 세 가지 경우로 나뉜다:

| 분포 종류 | 전하 요소 | 전기장 |
|---|---|---|
| **선** (단위 길이당 전하 $\lambda$) | $dq = \lambda \, dl'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\lambda(\mathbf{r}') \, dl'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |
| **면** (단위 면적당 전하 $\sigma$) | $dq = \sigma \, da'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\sigma(\mathbf{r}') \, da'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |
| **부피** (단위 부피당 전하 $\rho$) | $dq = \rho \, d\tau'$ | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}') \, d\tau'}{|\mathbf{r}-\mathbf{r}'|^2}\hat{\boldsymbol{\mathscr{r}}}$ |

### 예제: 균일하게 대전된 링의 전기장

반지름 $R$인 링에 전체 전하 $Q$가 균일하게 분포해 있다. 중심에서 거리 $z$인 축 위의 점에서:

$$E_z = \frac{1}{4\pi\epsilon_0} \frac{Qz}{(z^2 + R^2)^{3/2}}$$

이 결과는 대칭성에서 비롯된다 — 가로 성분들이 쌍으로 상쇄되고, 축 방향 성분만 살아남는다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Electric field on axis of a uniformly charged ring
# Why analytic + numerical: comparing them verifies our integration approach

epsilon_0 = 8.854e-12
Q = 1e-9       # total charge (1 nC)
R = 0.1        # ring radius (10 cm)

z = np.linspace(-0.5, 0.5, 500)

# Analytic formula — derived from symmetry and direct integration
# Why (z^2 + R^2)^(3/2): this comes from the geometry of the separation vector
E_analytic = Q * z / (4 * np.pi * epsilon_0 * (z**2 + R**2)**1.5)

# Numerical integration — discretize ring into N small charges
N = 1000
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
dq = Q / N  # each small segment carries charge dq

E_numerical = np.zeros_like(z)
for i, zi in enumerate(z):
    # Sum contributions from each segment; only z-component survives
    for th in theta:
        rx = -R * np.cos(th)       # vector from segment to axis point (x)
        ry = -R * np.sin(th)       # vector from segment to axis point (y)
        rz = zi                     # z-component of separation
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        # Why only z: by symmetry, x and y components cancel over the full ring
        E_numerical[i] += dq * rz / (4 * np.pi * epsilon_0 * r_mag**3)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z * 100, E_analytic, 'b-', linewidth=2, label='Analytic')
ax.plot(z * 100, E_numerical, 'r--', linewidth=2, label=f'Numerical (N={N})')
ax.set_xlabel('z (cm)')
ax.set_ylabel('$E_z$ (V/m)')
ax.set_title('Electric Field on Axis of Charged Ring')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ring_field.png', dpi=150)
plt.show()
```

---

## E의 발산과 회전

정적 전하 분포의 전기장은 두 가지 기본적인 벡터 미적분 성질을 갖는다.

### E의 발산

(아래에서 유도할) 가우스 법칙에 발산 정리를 적용하면:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

이것이 **가우스 법칙의 미분 형태**이다. 전기장선은 양전하에서 **발생**하고 음전하에서 **소멸**한다는 것을 의미한다. 전하가 없는 영역에서는 $\nabla \cdot \mathbf{E} = 0$ — 장은 그곳에서 솔레노이드형(solenoidal)이다.

### E의 회전

정전기학(정지 전하, 시간에 따라 변하는 자기장 없음)에서:

$$\nabla \times \mathbf{E} = 0$$

회전이 0이라는 것은 전기장이 **보존적(conservative)** 임을 의미한다 — 임의의 닫힌 경로를 따라 전하를 이동시킬 때 한 일은 0이다:

$$\oint \mathbf{E} \cdot d\mathbf{l} = 0$$

이 성질 덕분에 $\mathbf{E} = -\nabla V$를 만족하는 스칼라 퍼텐셜(scalar potential) $V$를 정의할 수 있다 (다음 레슨에서 다룬다).

> **비유**: 발산은 어떤 점이 장선의 '원천'인지 '소멸점'인지를 측정한다 — 욕조의 물에서 수도꼭지(+)나 배수구(-)에 해당한다. 회전이 0이라는 것은 장이 결코 '소용돌이치지' 않는다는 뜻이다 — 소용돌이와 달리 정전기장은 오직 전하를 향하거나 전하에서 멀어지는 방향으로만 향한다.

---

## 가우스 법칙

가우스 법칙은 맥스웰 방정식 네 개 중 하나로, 높은 대칭성을 가진 정전기 문제에서 가장 강력한 도구이다.

### 적분 형태

$$\oint_S \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\epsilon_0}$$

임의의 닫힌 곡면 $S$를 통과하는 **전기 선속(electric flux)**은 내부에 있는 전하를 $\epsilon_0$로 나눈 값과 같다.

**물리적 의미**: 닫힌 곡면 내부의 전하 분포가 아무리 복잡해도, 그 곡면을 통과하는 전체 선속은 내부 총 전하에만 의존한다. 외부의 전하는 알짜 선속에 기여하지 않는다.

### 미분 형태

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

두 형태는 **발산 정리(divergence theorem)**로 연결되어 동등하다:

$$\oint_S \mathbf{E} \cdot d\mathbf{a} = \int_V (\nabla \cdot \mathbf{E}) \, d\tau = \int_V \frac{\rho}{\epsilon_0} \, d\tau = \frac{Q_{\text{enc}}}{\epsilon_0}$$

### 점전하에 대한 증명 개요

원점에 놓인 점전하 $q$와 반지름 $r$인 구형 가우스 곡면을 고려하면:

$$\oint \mathbf{E} \cdot d\mathbf{a} = \oint \frac{q}{4\pi\epsilon_0 r^2} \hat{r} \cdot r^2 \sin\theta \, d\theta \, d\phi \, \hat{r} = \frac{q}{4\pi\epsilon_0} \cdot 4\pi = \frac{q}{\epsilon_0}$$

임의의 전하 분포는 점전하들의 중첩이고 적분은 선형이므로, 가우스 법칙은 임의의 전하 분포로 확장된다.

---

## 가우스 법칙의 응용

가우스 법칙은 전하 분포의 대칭성이 충분하여 $\mathbf{E}$를 선속 적분 밖으로 꺼낼 수 있을 때 가장 유용하다. 세 가지 대표적인 대칭성은 다음과 같다.

### 1. 구형 대칭 — 균일하게 대전된 구

반지름 $R$, 전체 전하 $Q$가 부피 전체에 균일하게 분포한 구 ($\rho = \frac{3Q}{4\pi R^3}$):

**외부** ($r > R$): 반지름 $r$인 구형 가우스 곡면을 선택하면:

$$E(4\pi r^2) = \frac{Q}{\epsilon_0} \implies E = \frac{Q}{4\pi\epsilon_0 r^2}$$

점전하의 장과 동일하다 — 아름다운 결과이다.

**내부** ($r < R$): 내부 전하는 $Q_{\text{enc}} = Q\left(\frac{r}{R}\right)^3$이므로:

$$E(4\pi r^2) = \frac{Q}{\epsilon_0}\left(\frac{r}{R}\right)^3 \implies E = \frac{Q r}{4\pi\epsilon_0 R^3}$$

구 내부에서 전기장은 $r$에 비례하여 선형으로 증가한다.

### 2. 원통형 대칭 — 무한 선전하

단위 길이당 전하가 균일하게 $\lambda$인 무한히 긴 직선. 반지름 $s$, 길이 $L$인 원통형 가우스 곡면을 선택하면:

$$E(2\pi s L) = \frac{\lambda L}{\epsilon_0} \implies E = \frac{\lambda}{2\pi\epsilon_0 s}$$

전기장은 $1/s$로 감소한다 ($1/s^2$가 아님!) — 원통형 기하의 특징이다.

### 3. 평면 대칭 — 무한 평면 전하

균일한 면전하 밀도 $\sigma$를 가진 무한 평면. 납작한 원통(pillbox) 모양의 가우스 곡면을 적용하면:

$$2EA = \frac{\sigma A}{\epsilon_0} \implies E = \frac{\sigma}{2\epsilon_0}$$

전기장은 **균일하다** — 평면으로부터의 거리에 무관하다. 이것이 평행판 축전기가 두 판 사이에 거의 균일한 전기장을 만드는 이유이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Gauss's law applications: E vs distance for three geometries
# Why plot all three together: comparing their distance dependence is instructive

epsilon_0 = 8.854e-12

# --- Spherical: uniformly charged solid sphere ---
Q = 1e-9      # total charge (1 nC)
R_sphere = 0.1  # radius (10 cm)

r = np.linspace(0.001, 0.4, 500)
E_sphere = np.where(
    r < R_sphere,
    Q * r / (4 * np.pi * epsilon_0 * R_sphere**3),        # inside: linear
    Q / (4 * np.pi * epsilon_0 * r**2)                     # outside: 1/r^2
)

# --- Cylindrical: infinite line charge ---
lam = 1e-9     # linear charge density (1 nC/m)
s = r          # use same radial array
E_line = lam / (2 * np.pi * epsilon_0 * s)                 # 1/s dependence

# --- Planar: infinite sheet ---
sigma = 1e-9   # surface charge density (1 nC/m^2)
E_plane = sigma / (2 * epsilon_0) * np.ones_like(r)        # constant

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(r * 100, E_sphere, 'b-', linewidth=2)
axes[0].axvline(x=R_sphere*100, color='gray', linestyle='--', label=f'R={R_sphere*100} cm')
axes[0].set_xlabel('r (cm)')
axes[0].set_ylabel('E (V/m)')
axes[0].set_title('Sphere (Q = 1 nC, R = 10 cm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(s * 100, E_line, 'r-', linewidth=2)
axes[1].set_xlabel('s (cm)')
axes[1].set_ylabel('E (V/m)')
axes[1].set_title(r'Line Charge ($\lambda$ = 1 nC/m)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(r * 100, E_plane, 'g-', linewidth=2)
axes[2].set_xlabel('distance (cm)')
axes[2].set_ylabel('E (V/m)')
axes[2].set_title(r'Plane ($\sigma$ = 1 nC/m$^2$)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, E_plane[0] * 1.5)

plt.suptitle("Gauss's Law: Three Classic Symmetries", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gauss_three_symmetries.png', dpi=150)
plt.show()
```

---

## 쌍극자의 전기장

전기 **쌍극자(electric dipole)**는 거리 $d$만큼 떨어진 같은 크기의 반대 전하 $\pm q$로 구성된 고전적이고 중요한 배치이다. **쌍극자 모멘트(dipole moment)**는 다음과 같다:

$$\mathbf{p} = q\mathbf{d}$$

여기서 $\mathbf{d}$는 음전하에서 양전하를 향하는 벡터이다.

쌍극자로부터 멀리 떨어진 곳 ($r \gg d$)에서 구면 좌표로 나타낸 전기장은:

$$\mathbf{E}_{\text{dip}}(r, \theta) = \frac{p}{4\pi\epsilon_0 r^3}\left(2\cos\theta \, \hat{r} + \sin\theta \, \hat{\theta}\right)$$

주요 특성:
- 전기장은 $1/r^3$으로 감소한다 (점전하의 $1/r^2$보다 빠름)
- 축 위 ($\theta = 0$): $\mathbf{E} = \frac{2p}{4\pi\epsilon_0 r^3}\hat{r}$
- 적도면 ($\theta = \pi/2$): $\mathbf{E} = \frac{p}{4\pi\epsilon_0 r^3}\hat{\theta}$

쌍극자는 총 전하가 0인 임의의 전하 분포의 **다중극 전개(multipole expansion)**에서 선두 항이다.

### 외부 전기장 속의 쌍극자: 토크와 에너지

외부 전기장 $\mathbf{E}$ 속에 놓인 쌍극자 $\mathbf{p}$에는 다음이 작용한다:

**토크(torque)**: $\boldsymbol{\tau} = \mathbf{p} \times \mathbf{E}$ — $\mathbf{p}$를 $\mathbf{E}$ 방향으로 정렬시키려 한다

**퍼텐셜 에너지**: $U = -\mathbf{p} \cdot \mathbf{E}$

- $\mathbf{p} \parallel \mathbf{E}$ (정렬 상태)일 때 에너지 최솟값
- $\mathbf{p}$와 $\mathbf{E}$가 반평행일 때 에너지 최댓값

**힘** (불균일 전기장 속에서): $\mathbf{F} = (\mathbf{p} \cdot \nabla)\mathbf{E}$

균일한 전기장 속의 쌍극자는 토크를 받지만 알짜 힘은 받지 않는다. 비균일한 전기장만이 쌍극자에 병진 방향의 알짜 힘을 가할 수 있다 — 이것이 미세유체역학에서 생물학적 세포를 조작하는 유전영동(dielectrophoresis)의 원리이다.

---

## E의 경계 조건

면전하 밀도 $\sigma$를 갖는 계면에서 전기장은 다음 경계 조건을 만족한다:

**법선 성분은 불연속이다**:
$$E_{\text{above}}^{\perp} - E_{\text{below}}^{\perp} = \frac{\sigma}{\epsilon_0}$$

**접선 성분은 연속이다**:
$$E_{\text{above}}^{\parallel} = E_{\text{below}}^{\parallel}$$

이 경계 조건들은 가우스 법칙(법선 성분)과 $\mathbf{E}$의 회전이 0임(접선 성분)을 무한히 얇은 면과 루프에 적용하여 유도된다.

이 조건들을 이해하는 것은 계면에서의 문제 — 도체 표면, 유전체 경계, 전하 시트 — 를 풀기 위해 필수적이며, 레슨 3에서 광범위하게 다룰 것이다.

---

## 가우스 법칙의 수치적 검증

닫힌 곡면을 통과하는 선속을 수치적으로 계산하여 가우스 법칙을 검증할 수 있다:

```python
import numpy as np

# Numerically verify Gauss's law: compute flux through a spherical surface
# Why numerical verification: builds confidence that the math works in code

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

# Point charge at origin
q = 1e-9  # 1 nC

# Create a spherical Gaussian surface (radius = 0.2 m)
R = 0.2
N_theta = 200       # polar angle resolution
N_phi = 400         # azimuthal angle resolution

theta = np.linspace(0, np.pi, N_theta)
phi = np.linspace(0, 2 * np.pi, N_phi)
THETA, PHI = np.meshgrid(theta, phi)

# Surface element dA = R^2 sin(theta) dtheta dphi * r_hat
# Why R^2 sin(theta): this is the Jacobian for spherical coordinates
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
dA = R**2 * np.sin(THETA) * dtheta * dphi

# E on the surface: for a point charge at origin, E_r = kq/R^2
E_r = k_e * q / R**2

# Total flux = sum of E_r * dA over the surface
flux_numerical = np.sum(E_r * dA)
flux_exact = q / epsilon_0

print(f"Numerical flux:  {flux_numerical:.6f} V·m")
print(f"Exact (q/ε₀):   {flux_exact:.6f} V·m")
print(f"Relative error:  {abs(flux_numerical - flux_exact)/flux_exact:.2e}")

# Verify with an off-center charge — charge OUTSIDE the surface
# Gauss's law predicts zero enclosed charge => zero net flux
q2_pos = np.array([0.5, 0.0, 0.0])  # charge at (0.5, 0, 0), outside R=0.2

# Points on the Gaussian surface
X_s = R * np.sin(THETA) * np.cos(PHI)
Y_s = R * np.sin(THETA) * np.sin(PHI)
Z_s = R * np.cos(THETA)

# Separation vectors from external charge to surface points
dx = X_s - q2_pos[0]
dy = Y_s - q2_pos[1]
dz = Z_s - q2_pos[2]
r_mag = np.sqrt(dx**2 + dy**2 + dz**2)

# E field at each surface point due to external charge
Ex = k_e * q * dx / r_mag**3
Ey = k_e * q * dy / r_mag**3
Ez = k_e * q * dz / r_mag**3

# Outward normal = (X_s, Y_s, Z_s)/R on a sphere centered at origin
# Why dot product with normal: flux = E · n dA
E_dot_n = (Ex * X_s + Ey * Y_s + Ez * Z_s) / R
flux_external = np.sum(E_dot_n * dA)

print(f"\nFlux from external charge: {flux_external:.6e} V·m  (should be ~0)")
```

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 쿨롱 법칙 | $\mathbf{F} = \frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{r^2}\hat{r}$ |
| 전기장 (점전하) | $\mathbf{E} = \frac{1}{4\pi\epsilon_0}\frac{q}{r^2}\hat{r}$ |
| 중첩 원리 | $\mathbf{E}_{\text{total}} = \sum_i \mathbf{E}_i$ |
| 가우스 법칙 (적분) | $\oint \mathbf{E}\cdot d\mathbf{a} = Q_{\text{enc}}/\epsilon_0$ |
| 가우스 법칙 (미분) | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ |
| E의 회전 (정전기) | $\nabla \times \mathbf{E} = 0$ |
| 구 ($r > R$) | $E = Q/(4\pi\epsilon_0 r^2)$ |
| 선전하 | $E = \lambda/(2\pi\epsilon_0 s)$ |
| 무한 평면 | $E = \sigma/(2\epsilon_0)$ |
| 쌍극자 (원거리장) | $E \sim p/(4\pi\epsilon_0 r^3)$ |

---

## 연습 문제

### 연습 문제 1: 중첩 원리 계산
$(0, 0, 0)$에 $q_1 = +3\,\mu\text{C}$, $(1, 0, 0)$ m에 $q_2 = -5\,\mu\text{C}$가 있다. $(0.5, 0.5, 0)$ m에서의 전기장(크기와 방향)을 구하라. 해석적으로 계산하고, Python으로도 수치적으로 검증하라.

### 연습 문제 2: 가우스 법칙 — 구형 껍질
반지름 $R$인 얇은 구형 껍질에 전체 전하 $Q$가 분포해 있다. 가우스 법칙을 이용해 다음을 증명하라:
- 껍질 내부 어디서나 $\mathbf{E} = 0$
- 껍질 외부에서 $\mathbf{E} = \frac{Q}{4\pi\epsilon_0 r^2}\hat{r}$

그런 다음 Python 코드를 작성하여 $r = R$에서의 불연속성을 수치적으로 검증하라.

### 연습 문제 3: 비균일 전하 분포
반지름 $R$인 단단한 구에서 전하 밀도가 $\rho(r) = \rho_0(1 - r/R)$ ($r \leq R$)으로 변한다. 가우스 법칙을 이용해 $r < R$과 $r > R$ 모두에서 $\mathbf{E}(r)$을 구하라. 결과를 그래프로 나타내라.

### 연습 문제 4: 수치적 선속 계산
정육면체 내부의 임의 위치에 점전하 세 개를 배치하라. 정육면체 각 면을 통과하는 전기 선속을 수치적으로 계산하고, 전체 선속이 $Q_{\text{enc}}/\epsilon_0$와 같음을 검증하라.

### 연습 문제 5: 사중극자 전기장 시각화
정사각형의 네 꼭짓점에 $+q, -q, -q, +q$를 배치한 사중극자(quadrupole)의 전기장을 계산하고 시각화하는 Python 프로그램을 작성하라. 원거리 장의 거동이 쌍극자와 어떻게 다른가?

---

[다음: 02. 전기 퍼텐셜과 에너지 →](02_Electric_Potential_and_Energy.md)

# 07. 편광

[← 이전: 06. 회절](06_Diffraction.md) | [다음: 08. 레이저 기초 →](08_Laser_Fundamentals.md)

---

## 학습 목표

1. 선형, 원형, 타원형의 세 가지 편광(Polarization) 종류와 그 수학적 표현을 설명한다
2. 말뤼스의 법칙(Malus's Law)을 적용하여 여러 편광자를 통과한 빛의 투과 세기를 계산한다
3. 반사에 의한 편광(브루스터 각, Brewster's Angle), 산란, 복굴절(Birefringence), 이색성(Dichroism)을 설명한다
4. 1/4 파장판과 1/2 파장판이 편광 상태에 미치는 효과를 분석한다
5. 존스 벡터(Jones Vector)와 존스 행렬(Jones Matrix)을 사용하여 편광 광학 소자 문제를 풀어낸다
6. 광학적 활성(Optical Activity), 패러데이 효과(Faraday Effect), 그리고 이들의 물리적 기원을 설명한다
7. LCD 디스플레이, 편광 선글라스, 3D 영화관이 편광을 어떻게 활용하는지 설명한다

---

## 왜 배우는가

편광은 육안으로는 보이지 않지만 기술적으로 매우 유용한, 빛의 "숨겨진 차원"이다. 액정 디스플레이(LCD), 3D 영화관 안경, 광섬유 통신, 레이저 레이더(라이다, Lidar), 투명 재료의 응력 분석, 우주 자기장을 측정하는 천문 기기 모두 편광의 제어 또는 측정에 의존한다. 편광을 이해하면 빛을 횡파(Transverse Wave) 전자기파로 이해하는 시각도 깊어진다 — 전기장(Electric Field) 진동 방향은 단순한 이론적 세부 사항이 아니라 조작·측정·활용할 수 있는 물리적 성질이다.

> **비유**: 말뚝 울타리 사이로 꿴 밧줄을 떠올려 보자. 밧줄을 위아래로 흔들면 수직 진동이 울타리의 수직 틈을 통과한다. 그러나 좌우로 흔들면 수평 진동은 막힌다. 편광자(Polarizer)는 빛에 대해 같은 방식으로 작동한다: 투과축(Transmission Axis)에 정렬된 전기장 성분은 통과시키고, 수직 성분은 차단한다. 빛을 위한 "말뚝 울타리"는 정렬된 분자 또는 금속 나노선으로 만들어진다.

---

## 1. 편광이란 무엇인가?

### 1.1 빛의 횡파 성질

빛은 횡파 전자기파이다: 전기장 $\mathbf{E}$는 진행 방향에 수직으로 진동한다. $z$ 방향으로 진행하는 파동에 대해:

$$\mathbf{E}(z, t) = E_x(z, t)\,\hat{\mathbf{x}} + E_y(z, t)\,\hat{\mathbf{y}}$$

빛의 **편광(Polarization)**은 파동이 고정된 점을 지나갈 때 $\mathbf{E}$ 벡터 끝이 그리는 궤적을 나타낸다. 이 궤적은 $E_x$와 $E_y$의 진폭과 위상에 의해 결정된다.

### 1.2 일반적인 경우: 타원 편광

일반적으로:

$$E_x = E_{0x}\cos(\omega t - kz)$$
$$E_y = E_{0y}\cos(\omega t - kz + \delta)$$

여기서 $E_{0x}$와 $E_{0y}$는 진폭이고, $\delta$는 $x$와 $y$ 성분 사이의 **위상차(Phase Difference)**이다. $\mathbf{E}$의 끝은 타원을 그린다 — 따라서 **타원 편광(Elliptical Polarization)**이 가장 일반적인 편광 상태이다.

위상차 $\delta$의 특수한 경우:

| $\delta$ | 편광 |
|----------|------|
| $0$ 또는 $\pi$ | **선형(Linear)** (고정된 방향으로 진동) |
| $\pm\pi/2$이고 $E_{0x} = E_{0y}$ | **원형(Circular)** |
| 기타 $\delta$ | **타원형(Elliptical)** (일반적인 경우) |

### 1.3 편광의 종류

**선형 편광(Linear Polarization)**: $\mathbf{E}$가 $xy$ 평면의 고정된 직선을 따라 진동한다. 방향각 $\alpha$:

$$\tan\alpha = \frac{E_{0y}}{E_{0x}} \quad (\delta = 0 \text{ 일 때})$$

**원형 편광(Circular Polarization)**: $\mathbf{E}$ 벡터가 일정한 크기를 유지하며 회전하여 원을 그린다. 두 가지 회전 방향:
- **우원형 편광(RCP, Right Circular Polarization)**: $\delta = -\pi/2$ — 다가오는 파동을 마주 보며 $\mathbf{E}$가 시계 방향으로 회전 (IEEE 규약)
- **좌원형 편광(LCP, Left Circular Polarization)**: $\delta = +\pi/2$ — 반시계 방향으로 회전

**비편광(Unpolarized Light)**: 측정 시간보다 훨씬 짧은 시간 척도에서 편광 방향이 무작위로 변한다. 자연광(태양광, 백열등, LED)은 일반적으로 비편광 상태이다.

**부분 편광(Partially Polarized Light)**: 편광 성분과 비편광 성분의 혼합. **편광도(Degree of Polarization)** $P$ ($0 \leq P \leq 1$)로 특성화된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize different polarization states by tracing the E-field vector tip

omega_t = np.linspace(0, 2*np.pi, 500)  # one full cycle

# Define polarization states: (E_0x, E_0y, delta, label)
states = [
    (1.0, 0.0, 0, 'Linear (horizontal)'),
    (1.0, 1.0, 0, 'Linear (45°)'),
    (1.0, 1.0, -np.pi/2, 'Right circular'),
    (1.0, 1.0, np.pi/2, 'Left circular'),
    (1.0, 0.5, np.pi/4, 'Elliptical'),
    (1.0, 0.7, np.pi/3, 'Elliptical (general)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, (E0x, E0y, delta, label) in zip(axes.flat, states):
    # Electric field components
    Ex = E0x * np.cos(omega_t)
    Ey = E0y * np.cos(omega_t + delta)

    # Plot the polarization ellipse (trajectory of E-vector tip)
    ax.plot(Ex, Ey, 'b-', linewidth=2)

    # Mark the starting point and direction
    ax.plot(Ex[0], Ey[0], 'ro', markersize=8, zorder=5, label='Start')
    # Arrow showing direction of rotation
    mid = len(omega_t) // 8
    ax.annotate('', xy=(Ex[mid], Ey[mid]), xytext=(Ex[mid-5], Ey[mid-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('$E_x$', fontsize=11)
    ax.set_ylabel('$E_y$', fontsize=11)
    ax.set_title(f'{label}\n$E_{{0x}}$={E0x}, $E_{{0y}}$={E0y}, $\\delta$={delta/np.pi:.2f}π',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

plt.suptitle('Polarization States of Light', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('polarization_states.png', dpi=150)
plt.show()
```

---

## 2. 편광자와 말뤼스의 법칙

### 2.1 이상적인 선형 편광자

이상적인 선형 편광자는 **투과축(Transmission Axis)** 방향의 $\mathbf{E}$ 성분만 통과시키고 수직 성분은 완전히 흡수한다.

세기 $I_0$이고 편광각 $\theta_0$인 선편광이 투과축이 각도 $\theta$인 편광자에 입사하면, 투과 세기는:

$$I = I_0 \cos^2(\theta - \theta_0)$$

이것이 **말뤼스의 법칙(Malus's Law)** (에티엔-루이 말뤼스, 1809)이다.

**특수한 경우**:
- 평행 ($\theta = \theta_0$): $I = I_0$ (완전 투과)
- 수직 ($\theta - \theta_0 = 90°$): $I = 0$ (완전 소멸)
- 45°: $I = I_0/2$

### 2.2 비편광이 편광자를 통과할 때

비편광의 경우 편광각이 무작위이다. 말뤼스의 법칙을 모든 각도에 대해 평균하면:

$$I = I_0 \langle\cos^2\theta\rangle = \frac{I_0}{2}$$

이상적인 편광자는 비편광 세기의 정확히 절반을 투과시킨다.

### 2.3 여러 편광자

"교차 편광자의 역설": 두 편광자의 투과축을 90°로 교차시키면 빛이 완전히 차단된다. 그러나 그 사이에 45° 방향의 세 번째 편광자를 삽입하면 일부 빛이 통과한다:

1. 비편광 $I_0$ → 첫 번째 편광자 (0°): $I_1 = I_0/2$ (0°로 편광)
2. 45° 편광자 통과: $I_2 = I_1\cos^2 45° = I_0/4$ (45°로 편광)
3. 90° 편광자 통과: $I_3 = I_2\cos^2 45° = I_0/8$ (90°로 편광)

45° 편광자는 각 단계에서 편광을 45°씩 "회전"시켜, 원래는 불투명했던 쌍을 통과할 수 있게 한다. 이는 측정(사영, Projection)이 편광 상태를 변화시키기 때문이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Malus's Law: transmitted intensity through two polarizers
# as a function of the angle between their transmission axes

theta_deg = np.linspace(0, 360, 500)
theta_rad = np.deg2rad(theta_deg)

# Malus's law for initially polarized light
I_malus = np.cos(theta_rad)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Malus's law
ax1.plot(theta_deg, I_malus, 'b-', linewidth=2, label="$I = I_0 \\cos^2\\theta$")
ax1.set_xlabel('Angle between polarizers $\\theta$ (degrees)', fontsize=12)
ax1.set_ylabel('Transmitted Intensity $I / I_0$', fontsize=12)
ax1.set_title("Malus's Law", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(np.arange(0, 361, 45))

# Right: N polarizers at equal angular increments from 0 to 90 degrees
# As N increases, more light gets through!
N_values = range(1, 11)
transmission = []

for N in N_values:
    # N polarizers equally spaced from 0 to 90 degrees
    # Each step rotates by 90/N degrees
    step_angle = np.deg2rad(90 / N)
    # Starting with polarized light (after first polarizer from unpolarized: I_0/2)
    I = 0.5  # after first polarizer
    for _ in range(N):
        I *= np.cos(step_angle)**2
    transmission.append(I)

ax2.bar(list(N_values), transmission, color='steelblue', edgecolor='navy', alpha=0.8)
ax2.set_xlabel('Number of intermediate polarizers N', fontsize=12)
ax2.set_ylabel('Final transmitted intensity $I / I_0$', fontsize=12)
ax2.set_title('Transmission through N+1 polarizers\n(spanning 0° to 90° in equal steps)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Annotate key values
for N, T in zip(N_values, transmission):
    ax2.text(N, T + 0.01, f'{T:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('malus_law.png', dpi=150)
plt.show()
```

---

## 3. 편광 생성 메커니즘

### 3.1 반사에 의한 편광: 브루스터 각

비편광이 유전체 표면에 입사하면 반사광은 부분적으로 편광된다. 특정 각도인 **브루스터 각(Brewster's Angle)** $\theta_B$에서 반사광은 완전히 편광된다(입사면에 수직인 $s$ 편광):

$$\tan\theta_B = \frac{n_2}{n_1}$$

브루스터 각에서 반사광과 굴절광은 서로 수직이다 ($\theta_B + \theta_t = 90°$).

| 경계면 | 브루스터 각 |
|--------|------------|
| 공기 → 유리 ($n=1.5$) | 56.3° |
| 공기 → 물 ($n=1.33$) | 53.1° |
| 공기 → 다이아몬드 ($n=2.42$) | 67.5° |

**원리**: 브루스터 각에서 표면의 진동 쌍극자는 반사광 방향으로 정렬된다. 쌍극자는 진동축 방향으로 복사하지 않으므로, $p$ 편광(평행 성분)의 반사율이 0이 된다.

이것이 바로 편광 선글라스가 도로와 수면의 눈부심을 효과적으로 줄이는 이유다 — 이들 표면에서 반사된 빛은 강하게 $s$ 편광(수평)되어 있으며, 편광 선글라스는 $s$ 편광을 차단하도록 배향되어 있다.

### 3.2 산란에 의한 편광

빛이 파장보다 훨씬 작은 입자로부터 산란될 때(레일리 산란, Rayleigh Scattering), 산란광은 편광된다. 편광도는 산란 각도에 따라 달라진다:
- 0°와 180° (전방/후방 산란): 비편광
- 90° (수직 방향): 완전 편광

하늘은 부분적으로 편광되어 있으며, 태양으로부터 90° 방향에서 최대 편광이 나타난다. 꿀벌을 비롯한 많은 곤충이 이 편광을 감지하여 항법에 활용한다.

### 3.3 복굴절에 의한 편광

**복굴절(Birefringent, 이중 굴절)** 결정은 편광 방향에 따라 굴절률이 다르다. 비편광이 이런 결정에 입사하면 두 빔으로 분리된다:

- **정상광선(Ordinary Ray, $o$-ray)**: 스넬의 법칙을 정상적으로 따르며, 굴절률 $n_o$
- **이상광선(Extraordinary Ray, $e$-ray)**: 일반적으로 스넬의 법칙을 따르지 않으며, 방향에 따라 변하는 굴절률 $n_e$

두 광선은 서로 직교 편광된다.

| 결정 | $n_o$ | $n_e$ | $\Delta n = n_e - n_o$ | 종류 |
|------|-------|-------|----------------------|------|
| 방해석 (CaCO$_3$) | 1.658 | 1.486 | -0.172 | 음성 |
| 석영 (SiO$_2$) | 1.544 | 1.553 | +0.009 | 양성 |
| 운모 (Mica) | 1.599 | 1.594 | -0.005 | 음성 |
| 루타일 (TiO$_2$) | 2.616 | 2.903 | +0.287 | 양성 |

$n_o = n_e$인 방향을 **광학축(Optic Axis)**이라 한다. 광학축을 따라 진행하는 빛은 복굴절을 경험하지 않는다.

### 3.4 이색성에 의한 편광

**이색성(Dichroic)** 재료는 한 편광 방향을 다른 방향보다 훨씬 강하게 흡수한다. 폴라로이드 필름(Edwin Land, 1928)은 분자 사슬 방향으로 편광된 빛은 흡수하고 수직 편광은 투과시키는 정렬된 폴리비닐 알코올 분자로 구성된다.

현대의 와이어 격자 편광자(Wire-Grid Polarizer)는 기판 위에 나노 스케일 금속 선을 배치한다 — 선에 평행한 $\mathbf{E}$ 성분은 전류를 유발하여 흡수되고, 수직 성분은 통과한다.

---

## 4. 파장판 (위상 지연판, Retarder)

### 4.1 원리

파장판(Wave Plate)은 광학축이 판 표면에 놓이도록 절단된 복굴절 결정이다. 빛이 통과하면 두 편광 성분이 서로 다른 위상 속도를 경험하여 제어된 위상차가 도입된다:

$$\delta = \frac{2\pi}{\lambda}(n_e - n_o) \cdot t$$

여기서 $t$는 판의 두께이다. 판 내의 두 편광 방향을 **빠른 축(Fast Axis)** (낮은 $n$)과 **느린 축(Slow Axis)** (높은 $n$)이라 한다.

### 4.2 1/4 파장판 ($\lambda/4$)

1/4 파장판은 위상차 $\delta = \pi/2$ (90°)를 도입한다:

$$t = \frac{\lambda}{4|n_e - n_o|}$$

**주요 변환**:
- 선형 (빠른 축에 45°) → 원형: 45° 입사 시 두 축에 동일한 진폭이 생기며 $\pi/2$ 위상차 = 원형 편광
- 원형 → 선형: 반대로, 원형 편광은 $\lambda/4$ 판을 통과하면 선형 편광이 된다

### 4.3 1/2 파장판 ($\lambda/2$)

1/2 파장판은 $\delta = \pi$ (180°)를 도입한다:

$$t = \frac{\lambda}{2|n_e - n_o|}$$

**주요 변환**:
- 선형 편광 회전: 입력 편광이 빠른 축에 대해 각도 $\alpha$를 이루면, 출력 편광은 $-\alpha$ — 편광이 빠른 축에 대해 "반사"되어 $2\alpha$만큼 회전
- 원형 편광의 회전 방향 반전: 우원형 → 좌원형, 또는 그 반대

```python
import numpy as np
import matplotlib.pyplot as plt

# Effect of wave plates on polarization state
# Visualize the transformation of the E-field tip trajectory

omega_t = np.linspace(0, 2*np.pi, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define input and output states for different wave plate configurations
configs = [
    # (title, E0x_in, E0y_in, delta_in, retardation, description)
    ('Input: Linear 45°', 1, 1, 0, 0, 'No plate'),
    ('λ/4 plate → RCP', 1, 1, 0, -np.pi/2, 'Quarter-wave'),
    ('λ/2 plate → Linear -45°', 1, 1, 0, -np.pi, 'Half-wave'),
    ('Input: RCP', 1, 1, -np.pi/2, 0, 'No plate'),
    ('λ/4 plate → Linear', 1, 1, -np.pi/2, -np.pi/2, 'Quarter-wave'),
    ('λ/2 plate → LCP', 1, 1, -np.pi/2, -np.pi, 'Half-wave'),
]

for ax, (title, E0x, E0y, delta_in, retard, desc) in zip(axes.flat, configs):
    # Input state
    Ex_in = E0x * np.cos(omega_t)
    Ey_in = E0y * np.cos(omega_t + delta_in)

    # Output state: add retardation to the y-component (assuming fast axis = x)
    delta_out = delta_in + retard
    Ex_out = E0x * np.cos(omega_t)
    Ey_out = E0y * np.cos(omega_t + delta_out)

    # Plot input (gray dashed) and output (blue solid)
    ax.plot(Ex_in, Ey_in, 'gray', linewidth=1, linestyle='--', alpha=0.5, label='Input')
    ax.plot(Ex_out, Ey_out, 'b-', linewidth=2, label='Output')

    # Direction arrow on output
    mid = len(omega_t) // 8
    ax.annotate('', xy=(Ex_out[mid], Ey_out[mid]),
                xytext=(Ex_out[mid-5], Ey_out[mid-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$E_x$', fontsize=10)
    ax.set_ylabel('$E_y$', fontsize=10)
    ax.set_title(f'{title}\n({desc})', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Wave Plate Effects on Polarization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_plate_effects.png', dpi=150)
plt.show()
```

---

## 5. 존스 벡터와 존스 행렬 형식론

### 5.1 존스 벡터

**존스 벡터(Jones Vector)**는 완전 편광 단색광의 편광 상태를 2차원 복소 벡터로 표현한다:

$$\mathbf{J} = \begin{pmatrix} E_{0x} \\ E_{0y} e^{i\delta} \end{pmatrix}$$

단위 세기로 정규화된 주요 존스 벡터:

| 편광 상태 | 존스 벡터 |
|----------|----------|
| 수평 ($x$) | $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ |
| 수직 ($y$) | $\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ |
| $+45°$ 선형 | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ |
| $-45°$ 선형 | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ |
| 우원형 (RCP) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$ |
| 좌원형 (LCP) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$ |

세기는 $I = |\mathbf{J}|^2 = |E_{0x}|^2 + |E_{0y}|^2$이다.

### 5.2 존스 행렬

각 편광 광학 소자는 $2 \times 2$ **존스 행렬(Jones Matrix)** $\mathbf{M}$으로 표현된다. 출력 존스 벡터는:

$$\mathbf{J}_{\text{out}} = \mathbf{M} \cdot \mathbf{J}_{\text{in}}$$

여러 소자가 연속될 경우, 오른쪽부터 왼쪽으로 행렬을 곱한다:

$$\mathbf{J}_{\text{out}} = \mathbf{M}_N \cdots \mathbf{M}_2 \cdot \mathbf{M}_1 \cdot \mathbf{J}_{\text{in}}$$

**주요 존스 행렬** ($x$, $y$에 주축이 정렬된 소자의 경우):

| 소자 | 존스 행렬 |
|------|----------|
| 수평 편광자 | $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ |
| 수직 편광자 | $\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ |
| 1/4 파장판 (빠른 축이 수평) | $e^{-i\pi/4}\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ |
| 1/2 파장판 (빠른 축이 수평) | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |
| 일반 파장판 (위상 지연 $\Gamma$) | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\Gamma} \end{pmatrix}$ |

### 5.3 소자의 회전

$x$축으로부터 각도 $\alpha$만큼 회전된 소자를 표현하려면 회전 변환을 적용한다:

$$\mathbf{M}(\alpha) = \mathbf{R}(-\alpha) \cdot \mathbf{M}(0) \cdot \mathbf{R}(\alpha)$$

회전 행렬:

$$\mathbf{R}(\alpha) = \begin{pmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{pmatrix}$$

**각도 $\alpha$의 선형 편광자**:

$$\mathbf{M}_{\text{pol}}(\alpha) = \begin{pmatrix} \cos^2\alpha & \sin\alpha\cos\alpha \\ \sin\alpha\cos\alpha & \sin^2\alpha \end{pmatrix}$$

```python
import numpy as np

# Jones calculus: solve a multi-element polarization problem
# Example: Light through polarizer → quarter-wave plate → analyzer

def jones_polarizer(angle_deg):
    """Jones matrix for a linear polarizer at angle alpha from horizontal."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c**2, s*c],
                     [s*c, s**2]])

def jones_waveplate(retardation, angle_deg):
    """
    Jones matrix for a wave plate with given retardation (radians),
    fast axis at angle_deg from horizontal.
    """
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)],
                  [-np.sin(a), np.cos(a)]])
    R_inv = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
    # Wave plate with fast axis along x
    W = np.array([[1, 0],
                  [0, np.exp(1j * retardation)]])
    return R_inv @ W @ R

def jones_intensity(J):
    """Calculate intensity from a Jones vector."""
    return float(np.abs(J[0])**2 + np.abs(J[1])**2)

# Problem: Unpolarized light approximated as horizontal
# → Vertical polarizer → Quarter-wave plate at 45° → Horizontal analyzer

print("Jones Calculus Example")
print("=" * 50)

# Step 1: Start with horizontally polarized light
J_in = np.array([1.0, 0.0], dtype=complex)
print(f"Input: {J_in}  (horizontal, I = {jones_intensity(J_in):.3f})")

# Step 2: Pass through vertical polarizer
M_vpol = jones_polarizer(90)
J_after_vpol = M_vpol @ J_in
print(f"After V-polarizer: {J_after_vpol}  (I = {jones_intensity(J_after_vpol):.3f})")
print("  → Blocked! (crossed polarizers)")

print()
print("Now insert a quarter-wave plate at 45° between crossed polarizers:")

# Step 1: Horizontal polarizer
M_hpol = jones_polarizer(0)
J1 = M_hpol @ np.array([1.0, 0.0], dtype=complex)
print(f"After H-polarizer: {J1}  (I = {jones_intensity(J1):.3f})")

# Step 2: Quarter-wave plate at 45°
M_qwp_45 = jones_waveplate(np.pi/2, 45)
J2 = M_qwp_45 @ J1
print(f"After QWP at 45°: [{J2[0]:.4f}, {J2[1]:.4f}]  (I = {jones_intensity(J2):.3f})")
# This should be circularly polarized

# Step 3: Vertical polarizer (analyzer)
M_vpol = jones_polarizer(90)
J3 = M_vpol @ J2
print(f"After V-polarizer: [{J3[0]:.4f}, {J3[1]:.4f}]  (I = {jones_intensity(J3):.3f})")
print(f"  → Transmission: {jones_intensity(J3)*100:.1f}% of input")

print()
print("Verify Malus's law with rotation:")
print("-" * 40)
# Rotate the analyzer and check intensity
for angle in range(0, 181, 15):
    M_analyzer = jones_polarizer(angle)
    J_out = M_analyzer @ M_qwp_45 @ J1
    I = jones_intensity(J_out)
    # After QWP, light is circularly polarized → intensity should be constant!
    print(f"  Analyzer at {angle:3d}°: I = {I:.4f}")
```

---

## 6. 광학적 활성과 패러데이 효과

### 6.1 광학적 활성

특정 재료는 선편광의 편광면을 빛이 진행함에 따라 회전시킨다. 이 현상을 **광학적 활성(Optical Activity)** 또는 **광학적 회전(Optical Rotation)**이라 한다.

회전각:

$$\phi = [\alpha] \cdot \ell \cdot c$$

여기서 $[\alpha]$는 **비선광도(Specific Rotation)** (파장과 온도에 의존), $\ell$은 경로 길이, $c$는 농도 (용액의 경우)이다.

| 물질 | 비선광도 $[\alpha]_D$ (°/dm per g/mL) |
|------|--------------------------------------|
| 설탕 (수크로스) | +66.5° |
| 포도당 (글루코스) | +52.7° |
| 과당 (프럭토스) | -92.0° |
| 석영 (결정, per mm) | +21.7° |

**물리적 기원**: 광학적 활성은 분자 구조의 키랄성(Chirality, 손대칭성)에서 비롯된다. 키랄 분자와 그 거울상은 빛을 서로 반대 방향으로 회전시킨다. 이것이 화학에서 **편광측정법(Polarimetry)**의 기초가 된다 — 식품 가공에서의 설탕 농도 측정, 의약품의 거울상 이성질체(Enantiomer) 순도 결정 등에 활용된다.

### 6.2 패러데이 효과

마이클 패러데이(Michael Faraday)는 1845년, 특정 재료에서 빛의 진행 방향을 따라 자기장을 인가하면 편광면이 회전함을 발견했다:

$$\phi = V \cdot B \cdot \ell$$

여기서 $V$는 **베르데 상수(Verdet Constant)** (재료에 따라 다름), $B$는 자기장 세기, $\ell$은 경로 길이이다.

| 재료 | 베르데 상수 (rad/T/m), 589 nm |
|------|-------------------------------|
| 물 | 1.309 |
| 유리 (플린트) | 3.17 |
| 테르븀 갈륨 가넷 (TGG) | 40 |

**광학적 활성과의 핵심 차이**: 패러데이 회전은 **비가역적(Non-Reciprocal)**이다 — 빛의 진행 방향에 상관없이 같은 방향으로 누적된다. 이 성질은 **광학 아이솔레이터(Optical Isolator)** (한 방향으로만 빛을 통과시키는 소자) 제작에 매우 중요하며, 레이저를 역반사(Back-Reflection)로부터 보호하는 데 필수적이다.

### 6.3 광학 아이솔레이터

광학 아이솔레이터는 패러데이 회전자(Faraday Rotator)와 편광자를 결합한 장치이다:

1. 입력 편광자 (0°): 수평 편광을 투과
2. 패러데이 회전자 (45°): 편광을 45° 회전
3. 출력 편광자 (45°): 45°로 회전된 빛을 투과

정방향: 0° → (+45° 회전) → 45° → **투과됨** ✓

역방향 (반사): 45° → (+45° 회전, -45°가 아님!) → 90° → **입력 편광자에 의해 차단됨** ✗

패러데이 효과의 비가역적 성질이 핵심이다 — 1/2 파장판은 귀환 경로에서 회전을 되돌리겠지만, 패러데이 회전자는 추가로 회전시킨다.

---

## 7. 프레넬 방정식과 경계면에서의 편광

### 7.1 프레넬 방정식

유전체 경계면에서의 반사 및 투과 계수는 편광에 따라 달라진다:

**$s$ 편광** (입사면에 수직):

$$r_s = \frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}$$

**$p$ 편광** (입사면에 평행):

$$r_p = \frac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}$$

반사율은 $R_s = |r_s|^2$, $R_p = |r_p|^2$이다.

브루스터 각에서 $r_p = 0$ (반사된 $p$ 성분 없음), 따라서 반사광은 순수하게 $s$ 편광된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fresnel equations: reflectance vs angle for s and p polarization
# Shows Brewster's angle where R_p = 0

def fresnel_reflectance(theta_i_deg, n1, n2):
    """
    Calculate s and p reflectances using Fresnel equations.
    theta_i_deg: angle of incidence in degrees
    n1, n2: refractive indices of the two media
    Returns: (R_s, R_p)
    """
    theta_i = np.deg2rad(theta_i_deg)

    # Snell's law: sin(theta_t) = n1/n2 * sin(theta_i)
    sin_theta_t = (n1 / n2) * np.sin(theta_i)

    # Handle total internal reflection
    mask = sin_theta_t <= 1.0
    cos_theta_t = np.where(mask, np.sqrt(1 - sin_theta_t**2), 0)

    # Fresnel coefficients
    r_s = np.where(mask,
                   (n1 * np.cos(theta_i) - n2 * cos_theta_t) /
                   (n1 * np.cos(theta_i) + n2 * cos_theta_t + 1e-30),
                   1.0)

    r_p = np.where(mask,
                   (n2 * np.cos(theta_i) - n1 * cos_theta_t) /
                   (n2 * np.cos(theta_i) + n1 * cos_theta_t + 1e-30),
                   1.0)

    R_s = r_s**2
    R_p = r_p**2

    return R_s, R_p

theta = np.linspace(0, 89.9, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: external reflection (air → glass)
n1, n2 = 1.0, 1.5
R_s, R_p = fresnel_reflectance(theta, n1, n2)
theta_B = np.rad2deg(np.arctan(n2/n1))

ax1.plot(theta, R_s * 100, 'b-', linewidth=2, label='$R_s$ (s-polarization)')
ax1.plot(theta, R_p * 100, 'r-', linewidth=2, label='$R_p$ (p-polarization)')
ax1.plot(theta, (R_s + R_p) / 2 * 100, 'k--', linewidth=1, label='Unpolarized (average)')
ax1.axvline(theta_B, color='green', linestyle=':', linewidth=1.5,
            label=f'Brewster angle = {theta_B:.1f}°')
ax1.set_xlabel('Angle of Incidence (degrees)', fontsize=12)
ax1.set_ylabel('Reflectance (%)', fontsize=12)
ax1.set_title(f'External Reflection: Air (n={n1}) → Glass (n={n2})', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 100)

# Right: internal reflection (glass → air) — shows TIR and Brewster's angle
n1, n2 = 1.5, 1.0
R_s_int, R_p_int = fresnel_reflectance(theta, n1, n2)
theta_B_int = np.rad2deg(np.arctan(n2/n1))
theta_c = np.rad2deg(np.arcsin(n2/n1))

ax2.plot(theta, R_s_int * 100, 'b-', linewidth=2, label='$R_s$')
ax2.plot(theta, R_p_int * 100, 'r-', linewidth=2, label='$R_p$')
ax2.axvline(theta_B_int, color='green', linestyle=':', linewidth=1.5,
            label=f'Brewster = {theta_B_int:.1f}°')
ax2.axvline(theta_c, color='orange', linestyle='--', linewidth=1.5,
            label=f'Critical angle = {theta_c:.1f}°')
ax2.set_xlabel('Angle of Incidence (degrees)', fontsize=12)
ax2.set_ylabel('Reflectance (%)', fontsize=12)
ax2.set_title(f'Internal Reflection: Glass (n={n1}) → Air (n={n2})', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('fresnel_equations.png', dpi=150)
plt.show()
```

---

## 8. 편광의 응용

### 8.1 액정 디스플레이 (LCD)

LCD 픽셀은 빛의 편광을 제어하여 작동한다:

1. **백라이트**가 비편광을 생성
2. **첫 번째 편광자**가 하나의 선형 편광을 투과
3. **액정(Liquid Crystal) 층**: 보통 LC 분자가 90° 뒤틀려(비틀린 네마틱 모드, Twisted Nematic Mode) 편광을 90° 회전시킨다. 전압을 인가하면 분자의 뒤틀림이 풀려 편광이 변하지 않는다.
4. **두 번째 편광자** (첫 번째와 교차): 90° 회전된 빛을 투과 (밝은 픽셀). 전압 인가 시 회전되지 않은 빛이 차단됨 (어두운 픽셀).

면내 전환(IPS, In-Plane Switching), 수직 정렬(VA, Vertical Alignment) 등 다른 LC 방식은 더 넓은 시야각과 높은 명암비를 제공한다.

### 8.2 편광 선글라스

수평 표면(도로, 물, 눈)에서의 눈부심(글레어)은 주로 $s$ 편광(수평)이다. 편광 선글라스는 투과축이 수직으로 배향되어 있어, 글레어를 차단하면서 환경으로부터의 수직 편광 및 비편광은 투과시킨다.

### 8.3 3D 영화관

**수동형 3D (RealD)**: 두 대의 프로젝터(또는 빠른 전환이 가능한 한 대)가 좌안과 우안 영상을 서로 반대 원형 편광(RCP와 LCP)으로 투영한다. 관객은 각 눈에 LCP와 RCP 필터가 있는 안경을 착용한다. 선형 편광 대신 원형 편광을 사용하므로 머리를 기울여도 3D 효과가 유지된다.

### 8.4 광탄성(Photoelasticity)

기계적 응력을 받은 투명 재료는 복굴절이 나타난다(응력 유발 복굴절). 교차 편광자 사이에서 응력을 받은 시료를 관찰하면 응력 분포를 나타내는 색상 줄무늬 패턴이 나타난다. 이 기법은 구조 부품 시험을 위한 기계공학에서 널리 사용된다.

### 8.5 타원편광측정법(Ellipsometry)

타원편광측정법은 표면 반사 시 편광 상태의 변화를 측정하여 나노미터 이하 정밀도로 박막 두께와 광학 상수를 결정한다. 반도체 제조에서 박막 증착 모니터링을 위한 표준 기법이다.

---

## 연습 문제

### 연습 문제 1: 말뤼스의 법칙 연쇄

세 개의 이상적인 선형 편광자가 투과축 방향 0°, 30°, 75° (수평 기준)으로 순서대로 배치되어 있다.

(a) 세기 $I_0$의 비편광이 첫 번째 편광자에 입사할 때, 각 편광자를 통과한 후의 세기는?

(b) 최종 편광 방향은?

(c) 중간 편광자(30°)를 제거하면 투과 세기는 얼마인가? 얼핏 역설적인 결과를 설명하라.

### 연습 문제 2: 브루스터 각

빛이 공기에서 유리판($n = 1.52$)에 입사한다.

(a) 브루스터 각을 계산하라.

(b) 브루스터 각에서 입사 비편광의 몇 분의 몇이 반사되는가? (힌트: 브루스터 각에서 $R_s$에 대한 프레넬 방정식을 사용하라.)

(c) 10장의 유리판이 브루스터 각으로 쌓여 있을 때(브루스터 스택), 투과빔의 편광도는?

### 연습 문제 3: 파장판 분석

1/2 파장판의 빠른 축이 수평으로부터 22.5° 방향으로 배향되어 있다.

(a) 수평 편광이 입사하면 출력 편광 방향은?

(b) 입력과 출력을 존스 벡터로 나타내라.

(c) 이 1/2 파장판에 대한 존스 행렬을 작성하고, 행렬 곱을 통해 답을 검증하라.

### 연습 문제 4: 원형 편광자 설계

비편광을 우원형 편광으로 변환하는 광학 시스템을 설계하라.

(a) 필요한 소자와 배향을 명시하라.

(b) 입력 세기의 몇 분의 몇이 투과되는가?

(c) 존스 계산법으로 설계를 검증하라.

### 연습 문제 5: 광학 아이솔레이터

광학 아이솔레이터가 1064 nm에서 베르데 상수 $V = 40$ rad/(T$\cdot$m)인 패러데이 회전자를 사용한다.

(a) 2 cm 결정에서 45° 회전을 위해 필요한 자기장 세기는?

(b) 역반사광은 차단되고 정방향 빛은 투과되는 이유를 단계별로 설명하라.

(c) 패러데이 회전자의 회전각이 이상적인 45°에서 $\pm 1°$ 벗어날 경우, 아이솔레이션 비율(정방향 대 역방향 투과 세기의 비)은?

---

## 요약

| 개념 | 핵심 공식 / 사실 |
|------|----------------|
| 편광 종류 | 선형 ($\delta = 0, \pi$), 원형 ($\delta = \pm\pi/2$, 동일 진폭), 타원형 (일반) |
| 말뤼스의 법칙 | $I = I_0\cos^2\theta$ (편광과 투과축 사이의 각도) |
| 비편광 → 편광자 | $I = I_0/2$ |
| 브루스터 각 | $\tan\theta_B = n_2/n_1$; 반사광은 $s$ 편광 |
| 복굴절 | $n_o \neq n_e$; 빛을 정상광선과 이상광선으로 분리 |
| 1/4 파장판 | $\delta = \pi/2$; 선형 ↔ 원형 변환 |
| 1/2 파장판 | $\delta = \pi$; 선형 편광을 $2\alpha$ 회전; 원형 편광의 방향 반전 |
| 존스 벡터 | $\mathbf{J} = \begin{pmatrix} E_{0x} \\ E_{0y}e^{i\delta} \end{pmatrix}$ |
| 프레넬 방정식 | $r_s, r_p$는 각도에 따라 다름; 브루스터 각에서 $R_p = 0$ |
| 광학적 활성 | $\phi = [\alpha] \ell c$ (편광면의 회전) |
| 패러데이 효과 | $\phi = VB\ell$ (비가역적 회전; 광학 아이솔레이터의 기초) |
| LCD 동작 | 교차 편광자 + 비틀린 네마틱 LC; 전압이 뒤틀림을 제어 → 밝기 조절 |

---

[← 이전: 06. 회절](06_Diffraction.md) | [다음: 08. 레이저 기초 →](08_Laser_Fundamentals.md)

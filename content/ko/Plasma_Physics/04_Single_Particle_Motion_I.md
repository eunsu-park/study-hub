# 4. 단일 입자 운동 I

## 학습 목표

- 전자기장에서 하전 입자의 운동 방정식 유도
- 균일한 자기장에서의 cyclotron 운동 분석 및 Larmor 반지름과 gyrofrequency 계산
- E×B drift와 비대칭 gyration으로부터의 물리적 기원 이해
- 빠른 gyration과 느린 drift를 분리하는 guiding center 근사 소개
- 정확한 입자 궤도 적분을 위한 Boris 알고리즘 구현
- Python을 사용하여 다양한 장 구성에서 입자 궤적 시각화

## 1. 운동 방정식

### 1.1 Lorentz 힘

전하 $q$와 질량 $m$을 가진 하전 입자가 속도 $\mathbf{v}$로 전기장 $\mathbf{E}$와 자기장 $\mathbf{B}$에서 움직일 때 **Lorentz 힘**을 경험합니다:

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

Newton의 제2법칙은 **운동 방정식**을 제공합니다:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

$$\frac{d\mathbf{x}}{dt} = \mathbf{v}$$

이것은 $(\mathbf{x}, \mathbf{v})$에 대한 6개의 1차 상미분방정식 시스템입니다.

### 1.2 주요 특성

**1. 자기력은 일을 하지 않음:**

$$\frac{dE_{kin}}{dt} = \mathbf{F} \cdot \mathbf{v} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \mathbf{v} = q\mathbf{E} \cdot \mathbf{v}$$

$(\mathbf{v} \times \mathbf{B}) \cdot \mathbf{v} = 0$이므로, 자기력은 항상 속도에 수직이며 일을 하지 않습니다.

**에너지 보존:** $\mathbf{E} = -\nabla \phi$인 정적 장에서:

$$E_{total} = \frac{1}{2}m v^2 + q\phi = \text{const}$$

**2. 자기력은 편향을 일으키지만 가속은 하지 않음:**

자기력은 $\mathbf{v}$의 **방향**을 변경하지만 그 **크기**는 변경하지 않습니다.

**3. 평행 및 수직 분해:**

$\hat{\mathbf{b}} = \mathbf{B}/B$를 $\mathbf{B}$를 따르는 단위 벡터로 정의합니다. 속도를 분해:

$$\mathbf{v} = v_\parallel \hat{\mathbf{b}} + \mathbf{v}_\perp$$

여기서 $v_\parallel = \mathbf{v} \cdot \hat{\mathbf{b}}$ 이고 $\mathbf{v}_\perp = \mathbf{v} - v_\parallel \hat{\mathbf{b}}$.

운동 방정식은 다음이 됩니다:

$$\frac{dv_\parallel}{dt} = \frac{q}{m} E_\parallel$$

$$\frac{d\mathbf{v}_\perp}{dt} = \frac{q}{m}(\mathbf{E}_\perp + \mathbf{v}_\perp \times \mathbf{B})$$

평행 운동은 수직 운동으로부터 **분리됩니다**(균일한 $\mathbf{B}$ 가정).

## 2. 균일 자기장: Cyclotron 운동

### 2.1 B-장만 있는 경우의 해

전기장 없이 균일한 자기장 $\mathbf{B} = B\hat{\mathbf{z}}$에서 입자를 고려하세요.

**평행 운동:**

$$\frac{dv_z}{dt} = 0 \quad \Rightarrow \quad v_z = v_{z,0} = \text{const}$$

입자는 $\mathbf{B}$를 따라 자유롭게 움직입니다.

**수직 운동:**

$(x, y)$ 평면에서:

$$m\frac{dv_x}{dt} = qv_y B$$

$$m\frac{dv_y}{dt} = -qv_x B$$

**gyrofrequency** (cyclotron frequency)를 정의:

$$\omega_c = \frac{|q|B}{m}$$

**부호 규약:** $\omega_c > 0$이 되도록 $|q|$를 사용합니다. 실제 부호 있는 주파수는:

$$\Omega_c = \frac{qB}{m} = \begin{cases}
\omega_c & \text{for } q > 0 \\
-\omega_c & \text{for } q < 0
\end{cases}$$

방정식은 다음이 됩니다:

$$\frac{dv_x}{dt} = \pm \omega_c v_y, \quad \frac{dv_y}{dt} = \mp \omega_c v_x$$

여기서 위의 부호는 양전하용, 아래는 음전하용입니다.

### 2.2 원운동

결합하면:

$$\frac{d^2 v_x}{dt^2} = \pm \omega_c \frac{dv_y}{dt} = -\omega_c^2 v_x$$

이것은 각주파수 $\omega_c$를 가진 단순 조화 운동입니다:

$$v_x(t) = v_\perp \cos(\omega_c t + \phi_0)$$

$$v_y(t) = \mp v_\perp \sin(\omega_c t + \phi_0)$$

여기서 $v_\perp = \sqrt{v_x^2 + v_y^2}$는 일정한 수직 속력입니다.

위치에 대해 적분(입자가 원점에서 시작한다고 가정):

$$x(t) = \frac{v_\perp}{\omega_c} \sin(\omega_c t + \phi_0) = r_L \sin(\omega_c t + \phi_0)$$

$$y(t) = \pm \frac{v_\perp}{\omega_c} \cos(\omega_c t + \phi_0) = \pm r_L \cos(\omega_c t + \phi_0)$$

이것은 반지름을 가진 **원운동**을 기술합니다:

$$r_L = \frac{v_\perp}{\omega_c} = \frac{m v_\perp}{|q|B}$$

이것이 **Larmor 반지름** (또는 gyroradius)입니다.

### 2.3 회전 방향

```
Gyration Direction:

B field into page (⊗):

Positive charge (e.g., proton):          Negative charge (e.g., electron):
        ↑                                         ↑
        │                                         │
    ←───●───→  Counterclockwise               ───●───→  Clockwise
        │      (right-hand rule)                  │      (opposite)
        ↓                                         ↓

For B = B ẑ (out of page):
- Ions (q > 0): rotate counterclockwise (viewed from +z)
- Electrons (q < 0): rotate clockwise
```

**오른손 법칙:** 엄지손가락을 $\mathbf{B}$를 따라 가리키면; 손가락들이 **양** 전하 gyration의 방향으로 말립니다.

### 2.4 나선형 궤적

평행 및 수직 운동을 결합:

$$\mathbf{r}(t) = \begin{pmatrix} r_L \sin(\omega_c t) \\ \pm r_L \cos(\omega_c t) \\ v_z t \end{pmatrix}$$

이것은 다음을 가진 **나선(helix)**입니다:
- 반지름 $r_L$
- 피치 $p = 2\pi v_z / \omega_c$
- $q$의 부호에 의해 결정되는 chirality

```
Helical Trajectory:

        z (along B)
        ↑
        │     ╱╲
        │    ╱  ╲
        │   ╱    ╲
        │  ╱      ╲
        │ ╱        ╲
        │╱__________╲___→ y
       ╱│            ╲
      ╱ │             ╲
     ╱  │              ╲
    ╱   │               ╲
   ╱    │                ╲
  x     │
       (gyration in xy-plane)

Particle spirals around field line with:
- Gyroradius r_L
- Parallel velocity v_z
```

### 2.5 특성 스케일

전자의 경우:

$$\omega_{ce} = \frac{eB}{m_e} \approx 1.76 \times 10^{11} B[\text{T}] \quad [\text{rad/s}]$$

$$r_{Le} = \frac{v_\perp}{\omega_{ce}} \approx 5.69 \times 10^{-6} \frac{v_\perp[\text{m/s}]}{B[\text{T}]} \quad [\text{m}]$$

$T_e = 100$ eV에서 열 전자의 경우:

$$v_{te} \approx 4.2 \times 10^6 \text{ m/s}$$

$B = 1$ T에서:

$$\omega_{ce} \approx 1.76 \times 10^{11} \text{ rad/s}, \quad r_{Le} \approx 24 \text{ μm}$$

양성자의 경우 ($m_p = 1836 m_e$):

$$\omega_{ci} = \frac{\omega_{ce}}{1836} \approx 9.6 \times 10^7 \text{ rad/s}$$

$$r_{Li} = \sqrt{1836} \, r_{Le} \approx 43 r_{Le} \approx 1 \text{ mm}$$

**순서:**
$$r_{Le} \ll r_{Li}, \quad \omega_{ci} \ll \omega_{ce}$$

전자는 이온보다 훨씬 빠르게 그리고 작은 반지름으로 gyration합니다.

## 3. 균일 E와 B 장: E×B Drift

### 3.1 수직 전기장

이제 $\mathbf{B}$에 수직인 균일한 전기장 $\mathbf{E}$를 추가합니다. 운동 방정식은:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

**핵심 관찰:** $\mathbf{E}$ 항은 원 궤도의 대칭성을 깨뜨립니다.

### 3.2 Drift 속도

다음으로 구성된 해를 보일 수 있습니다(아래 유도):
1. 주파수 $\omega_c$와 반지름 $r_L$로 **Gyration**
2. $\mathbf{E}$와 $\mathbf{B}$ 둘 다에 수직인 **Drift**

**E×B drift 속도**는:

$$\mathbf{v}_E = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

**주요 특성:**
- 전하 $q$와 질량 $m$에 독립적!
- $\mathbf{E}$와 $\mathbf{B}$ 둘 다에 수직
- 크기: $v_E = E/B$ ($\mathbf{E} \perp \mathbf{B}$인 경우)

### 3.3 유도

$\mathbf{v} = \mathbf{v}_g + \mathbf{v}'$로 쓰세요. 여기서 $\mathbf{v}_g$는 (결정될) 일정한 drift 속도이고 $\mathbf{v}'$는 drifting frame에서의 gyration 속도입니다.

운동 방정식에 대입:

$$m\frac{d\mathbf{v}'}{dt} = q[\mathbf{E} + (\mathbf{v}_g + \mathbf{v}') \times \mathbf{B}]$$

$$= q[\mathbf{E} + \mathbf{v}_g \times \mathbf{B}] + q\mathbf{v}' \times \mathbf{B}$$

**일정 항이 사라지도록 $\mathbf{v}_g$를 선택:**

$$\mathbf{E} + \mathbf{v}_g \times \mathbf{B} = 0$$

양변에 $\mathbf{B}$를 외적:

$$\mathbf{E} \times \mathbf{B} + (\mathbf{v}_g \times \mathbf{B}) \times \mathbf{B} = 0$$

벡터 항등식 $(\mathbf{A} \times \mathbf{B}) \times \mathbf{C} = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{A}(\mathbf{B} \cdot \mathbf{C})$ 사용:

$$\mathbf{E} \times \mathbf{B} + \mathbf{B}(\mathbf{v}_g \cdot \mathbf{B}) - \mathbf{v}_g B^2 = 0$$

$\mathbf{v}_g \perp \mathbf{B}$이면, $\mathbf{v}_g \cdot \mathbf{B} = 0$:

$$\mathbf{v}_g = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

이것이 E×B drift입니다.

drifting frame에서, 방정식은 다음이 됩니다:

$$m\frac{d\mathbf{v}'}{dt} = q\mathbf{v}' \times \mathbf{B}$$

이것은 단지 gyration입니다(Section 2에서 해결).

### 3.4 물리적 그림

왜 E×B drift가 발생합니까?

```
Physical Mechanism of E×B Drift:

Consider E pointing in +y direction, B in +z (into page).

Electron orbit:

    Point A: v pointing +x        Point C: v pointing -x
             E accelerates up               E accelerates down
             ↑                              ↓
    ╭───────●───────╮            ╭─────────●─────────╮
   ╱        │        ╲          ╱          │          ╲
  │    D    │    B    │        │      D    │    B      │
  │         ●         │        │           ●           │
  │    ←─── ● ───→   │        │      ←─── ● ───→     │
  │         E         │        │           E           │
   ╲                 ╱          ╲                     ╱
    ╰───────────────╯            ╰───────────────────╯

At A: E accelerates particle upward → larger v_y → larger r_L
At C: E decelerates particle → smaller v_y → smaller r_L

Asymmetry: Top of orbit has larger radius than bottom
Result: Net drift to the right (−x direction)

Direction: E×B/B² = (E_y ŷ) × (B_z ẑ) / B² = -E_y/B x̂ (−x direction)

Both electrons and ions drift in the same direction!
(ions gyrate opposite direction but E affects them oppositely)
```

### 3.5 예제: Magnetron

magnetron에서, 전자는 교차하는 $\mathbf{E}$와 $\mathbf{B}$ 장에서 drift합니다:

- 반경 방향 전기장: $E_r$ (바깥쪽)
- 축 방향 자기장: $B_z$
- E×B drift: 방위각, $v_\theta = E_r/B_z$

전자는 anode 주위를 "spoke" 패턴으로 궤도를 돌며, RF cavity와 상호작용하여 마이크로파를 생성합니다.

## 4. 평행 전기장

### 4.1 B를 따라 가속

$\mathbf{E}$가 $\mathbf{B}$에 평행한 성분을 가지면:

$$\frac{dv_\parallel}{dt} = \frac{q}{m} E_\parallel$$

입자는 field line을 따라 가속(또는 감속)됩니다.

**에너지 증가:**

$$\frac{d}{dt}\left(\frac{1}{2}mv^2\right) = q\mathbf{E} \cdot \mathbf{v} = q E_\parallel v_\parallel$$

### 4.2 결합 운동

$E_\parallel$와 $E_\perp$ 둘 다 있을 때:

- **평행:** $\mathbf{B}$를 따라 가속 운동
- **수직:** Gyration + E×B drift
- 결과: 변화하는 피치와 drifting guiding center를 가진 나선형 경로

```
Combined E∥ and E⊥:

        z (B direction)
        ↑
        │   ___
        │  /   \___
        │ /        \___
        │/  increasing v_z
       /│   (E∥ accelerates)
      / │
     /  │            ←─ E⊥ causes E×B drift
    /   │               (guiding center moves)
   /    │
  x     │

Particle spirals with:
- Increasing v_z (if qE∥ > 0)
- Drifting guiding center (E×B)
- Approximately constant r_L (v_⊥ constant if E⊥ constant)
```

## 5. Guiding Center 근사

### 5.1 동기

많은 플라스마에서, gyroradius $r_L$은 장 변화의 스케일 길이 $L$보다 훨씬 작습니다:

$$r_L \ll L$$

느린, 더 큰 스케일의 역학에 관심이 있을 때 빠른 gyration을 해결하는 것은 비효율적입니다.

**Guiding center 근사:** 빠른 gyration을 느린 drift로부터 분리합니다.

### 5.2 Guiding Center 위치

gyro-궤도의 중심으로 **guiding center 위치** $\mathbf{R}$을 정의:

$$\mathbf{R} = \mathbf{r} - \frac{\mathbf{v}_\perp \times \mathbf{B}}{\omega_c B}$$

위치 $\mathbf{r}$과 수직 속도 $\mathbf{v}_\perp$를 가진 입자에 대해:

$$\mathbf{R} = \mathbf{r} - r_L \hat{\mathbf{e}}_\perp$$

여기서 $\hat{\mathbf{e}}_\perp$는 guiding center에서 입자 위치를 향해 가리킵니다.

```
Guiding Center:

         B (out of page)
              ⊙

        Particle orbit
          ╭───╮
         ╱     ╲
        │   ●   │  ← Guiding center R
        │  ╱│╲  │
         ╲╱ │ ╲╱
          ╰─●─╯
            ↑
       Particle position r

r_L = |r - R|
```

### 5.3 Guiding Center 속도

guiding center의 속도는 gyro-period에 대해 평균한 **drift 속도**입니다:

$$\mathbf{V}_g = \langle \mathbf{v} \rangle_\text{gyro} = \mathbf{v}_E + \mathbf{v}_\text{other drifts}$$

균일한 $\mathbf{E}$와 $\mathbf{B}$에 대해:

$$\mathbf{V}_g = \frac{\mathbf{E} \times \mathbf{B}}{B^2} + v_\parallel \hat{\mathbf{b}}$$

gyration 운동은 이 기술에서 "평균화됩니다".

### 5.4 Adiabatic Invariant

장이 공간과 시간에서 천천히 변할 때(gyration에 비해), 특정 양들은 **adiabatic invariant**입니다 - 근사적으로 보존됩니다.

가장 중요한 것은 **magnetic moment**입니다:

$$\mu = \frac{m v_\perp^2}{2B} = \frac{E_\perp}{B}$$

여기서 $E_\perp = \frac{1}{2}m v_\perp^2$는 수직 운동 에너지입니다.

**보존:** $B$가 입자 궤적을 따라 천천히 변하면, $\mu$는 거의 일정하게 유지됩니다.

다음 레슨(Single Particle Motion II)에서 이것을 탐구할 것입니다.

## 6. 계산 구현: Boris 알고리즘

### 6.1 입자 Pushing의 도전 과제

Lorentz 힘 방정식을 적분하는 것은 다음 때문에 도전적입니다:

1. **Gyration 시간스케일:** $\tau_c = 2\pi/\omega_c$는 매우 짧을 수 있습니다(1 T에서 전자의 경우 $\sim 10^{-11}$ s)
2. **Stiffness:** 명시적 방법(Euler, RK4)은 안정성을 위해 $\Delta t \ll \tau_c$를 요구합니다
3. **에너지 보존:** 순진한 방법은 오차를 누적하여, 비물리적 에너지 drift를 야기합니다

### 6.2 Boris 알고리즘

**Boris 알고리즘** (Boris, 1970)은 플라스마 시뮬레이션에서 particle pushing을 위한 gold standard입니다. 이것은:

- **2차 정확**합니다
- **에너지를 보존**합니다(일정한 장의 경우)
- **효율적**이고 **안정적**입니다

**알고리즘:**

시간 $t^n$에서 $\mathbf{v}^n$과 $\mathbf{x}^n$이 주어졌을 때, $t^{n+1} = t^n + \Delta t$에서 $\mathbf{v}^{n+1}$과 $\mathbf{x}^{n+1}$을 계산:

1. **절반 전기 push:**
   $$\mathbf{v}^- = \mathbf{v}^n + \frac{q\mathbf{E}}{m} \frac{\Delta t}{2}$$

2. **자기 회전:**
   $\mathbf{t} = \frac{q\mathbf{B}}{m} \frac{\Delta t}{2}$와 $s = \frac{2\mathbf{t}}{1 + t^2}$를 정의.

   $$\mathbf{v}' = \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t}$$

   $$\mathbf{v}^+ = \mathbf{v}^- + \mathbf{v}' \times \mathbf{s}$$

3. **절반 전기 push:**
   $$\mathbf{v}^{n+1} = \mathbf{v}^+ + \frac{q\mathbf{E}}{m} \frac{\Delta t}{2}$$

4. **위치 업데이트:**
   $$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

**왜 작동하는가:** 회전 단계는 정확히 $|\mathbf{v}|$를 보존하고(자기력은 일을 하지 않음), half-step 전기 가속은 에너지 오차를 최소화합니다.

### 6.3 Python 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
e = 1.602176634e-19      # Elementary charge [C]
m_e = 9.1093837015e-31   # Electron mass [kg]
m_p = 1.672621898e-27    # Proton mass [kg]

class ParticleTracer:
    """
    Trace charged particle trajectories using the Boris algorithm.
    """

    def __init__(self, q, m, x0, v0):
        """
        Parameters:
        -----------
        q : float
            Charge [C]
        m : float
            Mass [kg]
        x0 : array_like
            Initial position [m] (3D)
        v0 : array_like
            Initial velocity [m/s] (3D)
        """
        self.q = q
        self.m = m
        self.x = np.array(x0, dtype=float)
        self.v = np.array(v0, dtype=float)

        # History
        self.x_history = [self.x.copy()]
        self.v_history = [self.v.copy()]
        self.t_history = [0.0]

    def boris_step(self, E, B, dt):
        """
        Advance one timestep using Boris algorithm.

        Parameters:
        -----------
        E : array_like
            Electric field [V/m] (3D)
        B : array_like
            Magnetic field [T] (3D)
        dt : float
            Timestep [s]
        """
        q_over_m = self.q / self.m

        # 절반 전기 push: 회전 전에 E 가속을 dt/2 동안 적용한다.
        # E와 B를 이렇게 분리하면 순수 회전 자기 스텝을 독립적으로 처리할 수 있어,
        # 나이브 Euler 스텝에서와 달리 강성(stiff) B-장 항을 dt에 대한
        # 안정성 제약 없이 정확하게 다룰 수 있다.
        v_minus = self.v + 0.5 * q_over_m * E * dt

        # 자기 반 스텝을 위한 회전 벡터 t와 s를 구성한다.
        # t = (q/m) * B * (dt/2)는 B에 수직인 평면에서의 반-각(half-angle)이다.
        # Cayley-Klein(탄 반각(tan half-angle)) 매개변수화를 선택하는 이유는
        # 단위 크기 회전에 정확히 대응하기 때문이다:
        # |v_plus| == |v_minus|는 |t|의 크기에 관계없이 대수적으로 보장된다.
        # 나이브 외적 회전(v += v × ω * dt)은 |v|를 (1 ± |t|^2)만큼 수축/팽창시킨다.
        t = 0.5 * q_over_m * B * dt
        t_mag_sq = np.dot(t, t)
        # s = 2t / (1 + |t|^2)는 Cayley-Klein 사상의 두 번째 절반이다.
        # v_prime = v + v × t 와 v_plus = v_minus + v_prime × s를 합치면
        # B 주위로 2*arctan(|t|)만큼 정확한 회전을 구현한다.
        s = 2 * t / (1 + t_mag_sq)

        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)

        # 두 번째 절반 전기 push: 대칭 반 스텝 구조(E/2 → 회전 → E/2)는
        # 알고리즘을 시간 가역적으로 만든다(dt → -dt로 치환하고 v를 반전하면
        # 초기 상태로 정확히 복원된다). 시간 가역성이 바로 단방향 방법(예: 단순 Euler)이
        # 많은 gyroperiod에 걸쳐 겪는 장기 에너지 드리프트를 막는 이유다.
        self.v = v_plus + 0.5 * q_over_m * E * dt

        # Position update
        self.x += self.v * dt

    def trace(self, E_func, B_func, t_max, dt):
        """
        Trace particle trajectory.

        Parameters:
        -----------
        E_func : callable
            Function E(x, t) returning electric field
        B_func : callable
            Function B(x, t) returning magnetic field
        t_max : float
            Maximum simulation time [s]
        dt : float
            Timestep [s]
        """
        t = 0.0
        while t < t_max:
            # 매 스텝마다 현재 위치에서 장을 다시 평가해 공간적·시간적으로
            # 변화하는 장(비균일 B, 진동 E)을 지원한다.
            # 균일 정적 장에서는 중복이지만, 인터페이스를 범용적으로 유지하면서
            # 성능 비용은 크지 않다.
            E = E_func(self.x, t)
            B = B_func(self.x, t)

            self.boris_step(E, B, dt)

            t += dt
            self.t_history.append(t)
            # 참조가 아닌 복사본을 저장해서, 이후 self.x와 self.v에 대한
            # in-place 업데이트가 기록된 이력을 덮어쓰지 않도록 한다.
            self.x_history.append(self.x.copy())
            self.v_history.append(self.v.copy())

        # 루프 내부에서 numpy 배열에 행을 추가하면 O(N^2) 메모리 재할당이
        # 발생하므로, 마지막에 한 번만 배열로 변환한다.
        self.x_history = np.array(self.x_history)
        self.v_history = np.array(self.v_history)
        self.t_history = np.array(self.t_history)


# Example 1: Gyration in uniform B field
def example_gyration():
    """Pure gyration in uniform magnetic field."""

    # E = 0이고 B가 z 방향인 가장 단순한 설정으로, drift 없이 순수한
    # 원형(gyro) 운동을 만든다. B를 z 방향으로 선택하면 궤적 플롯에서
    # xy 평면의 gyration을 직접 읽을 수 있다.
    B0 = 0.1  # Tesla
    B_func = lambda x, t: np.array([0, 0, B0])
    E_func = lambda x, t: np.array([0, 0, 0])

    # 초기 속도를 순수 x 방향으로 설정하면 Larmor 반지름이 r_L = v_x/omega_c가 되어,
    # 평행·수직 성분 벡터 분해 없이 해석 공식과 바로 대조 확인할 수 있다.
    v_perp = 1e6  # m/s
    x0 = [0, 0, 0]
    v0 = [v_perp, 0, 0]

    # Create tracer
    electron = ParticleTracer(q=-e, m=m_e, x0=x0, v0=v0)

    # Gyrofrequency and period
    omega_ce = e * B0 / m_e
    T_gyro = 2 * np.pi / omega_ce
    r_L = v_perp / omega_ce

    print(f"Electron gyration:")
    print(f"  Gyrofrequency: ω_ce = {omega_ce:.3e} rad/s")
    print(f"  Gyroperiod: T = {T_gyro:.3e} s")
    print(f"  Larmor radius: r_L = {r_L:.3e} m = {r_L*1e6:.2f} μm")

    # Trace for 3 periods
    dt = T_gyro / 100
    electron.trace(E_func, B_func, t_max=3*T_gyro, dt=dt)

    # Plot
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(electron.x_history[:, 0] * 1e6, electron.x_history[:, 1] * 1e6,
             'b-', linewidth=1.5)
    ax1.scatter([0], [0], c='r', s=100, marker='x', label='Guiding center')
    ax1.set_xlabel('x [μm]', fontsize=11)
    ax1.set_ylabel('y [μm]', fontsize=11)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Electron Gyration (top view)', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(122)
    ax2.plot(electron.t_history / T_gyro,
             np.sqrt(electron.v_history[:, 0]**2 + electron.v_history[:, 1]**2),
             'r-', linewidth=1.5, label='$v_\\perp$')
    ax2.axhline(v_perp, color='k', linestyle='--', alpha=0.5, label='Expected')
    ax2.set_xlabel('Time [gyroperiods]', fontsize=11)
    ax2.set_ylabel('Perpendicular velocity [m/s]', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Velocity Conservation', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('gyration_uniform_B.png', dpi=150)
    plt.show()

example_gyration()


# Example 2: Helical trajectory
def example_helix():
    """Helical trajectory with parallel velocity."""

    B0 = 0.5  # Tesla
    B_func = lambda x, t: np.array([0, 0, B0])
    E_func = lambda x, t: np.array([0, 0, 0])

    # Proton with both perpendicular and parallel velocity
    v_perp = 1e5  # m/s
    v_para = 2e5  # m/s
    x0 = [0, 0, 0]
    v0 = [v_perp, 0, v_para]

    proton = ParticleTracer(q=e, m=m_p, x0=x0, v0=v0)

    # Parameters
    omega_ci = e * B0 / m_p
    T_gyro = 2 * np.pi / omega_ci
    r_L = v_perp / omega_ci
    pitch = 2 * np.pi * v_para / omega_ci

    print(f"\nProton helix:")
    print(f"  Gyroperiod: T = {T_gyro:.3e} s")
    print(f"  Larmor radius: r_L = {r_L:.3e} m = {r_L*1e3:.2f} mm")
    print(f"  Pitch: {pitch:.3e} m = {pitch*100:.2f} cm")

    dt = T_gyro / 100
    proton.trace(E_func, B_func, t_max=5*T_gyro, dt=dt)

    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = proton.x_history[:, 0] * 1e3
    y = proton.x_history[:, 1] * 1e3
    z = proton.x_history[:, 2]

    ax.plot(x, y, z, 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='s', label='End')

    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('y [mm]', fontsize=11)
    ax.set_zlabel('z [m]', fontsize=11)
    ax.set_title('Proton Helical Trajectory', fontsize=13, fontweight='bold')
    ax.legend()

    plt.savefig('helical_trajectory.png', dpi=150)
    plt.show()

example_helix()


# Example 3: E×B drift
def example_ExB_drift():
    """E×B drift in crossed electric and magnetic fields."""

    # Fields
    E0 = 1e3  # V/m (in x-direction)
    B0 = 0.1  # T (in z-direction)
    E_func = lambda x, t: np.array([E0, 0, 0])
    B_func = lambda x, t: np.array([0, 0, B0])

    # E×B drift velocity (should be in -y direction)
    v_ExB = E0 / B0  # m/s

    print(f"\nE×B drift:")
    print(f"  E = {E0} V/m (x-direction)")
    print(f"  B = {B0} T (z-direction)")
    print(f"  v_E = E/B = {v_ExB:.3e} m/s (-y direction)")

    # Electron and proton with small initial perpendicular velocity
    v_perp = 1e5  # m/s
    x0 = [0, 0, 0]
    v0_e = [v_perp, 0, 0]
    v0_p = [v_perp, 0, 0]

    electron = ParticleTracer(q=-e, m=m_e, x0=x0, v0=v0_e)
    proton = ParticleTracer(q=e, m=m_p, x0=x0, v0=v0_p)

    # Gyroperiods
    T_e = 2 * np.pi * m_e / (e * B0)
    T_p = 2 * np.pi * m_p / (e * B0)

    dt_e = T_e / 100
    dt_p = T_p / 100
    t_max = 20 * T_p  # Simulate long enough to see drift

    electron.trace(E_func, B_func, t_max, dt_e)
    proton.trace(E_func, B_func, t_max, dt_p)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(electron.x_history[:, 0] * 1e3, electron.x_history[:, 1] * 1e6,
            'b-', linewidth=1, alpha=0.7, label='Electron')
    ax.plot(proton.x_history[:, 0] * 1e3, proton.x_history[:, 1] * 1e6,
            'r-', linewidth=1, alpha=0.7, label='Proton')

    # Expected drift
    y_drift = -v_ExB * t_max
    ax.axhline(y_drift * 1e6, color='k', linestyle='--', linewidth=2,
               label=f'Expected drift: {y_drift*1e6:.1f} μm')

    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('y [μm]', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('E×B Drift (both species drift together)', fontsize=12, fontweight='bold')

    # Guiding center positions
    ax = axes[1]
    # Average over gyration
    window = 20
    y_e_gc = np.convolve(electron.x_history[:, 1], np.ones(window)/window, mode='valid')
    y_p_gc = np.convolve(proton.x_history[:, 1], np.ones(window)/window, mode='valid')
    t_gc = electron.t_history[:len(y_e_gc)]

    ax.plot(t_gc * 1e6, y_e_gc * 1e6, 'b-', linewidth=2, label='Electron GC')
    ax.plot(proton.t_history[:len(y_p_gc)] * 1e6, y_p_gc * 1e6,
            'r-', linewidth=2, label='Proton GC')

    # Expected
    ax.plot(t_gc * 1e6, -v_ExB * t_gc * 1e6, 'k--', linewidth=2,
            label=f'Expected: $-v_E t$')

    ax.set_xlabel('Time [μs]', fontsize=11)
    ax.set_ylabel('Guiding Center y [μm]', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Guiding Center Drift', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('ExB_drift.png', dpi=150)
    plt.show()

example_ExB_drift()
```

### 6.4 검증: 에너지 보존

```python
def validate_energy_conservation():
    """Check energy conservation in various field configurations."""

    B0 = 1.0
    v0 = 1e6

    # 세 가지 설정은 서로 다른 에너지 보존 시나리오를 검증한다:
    # "B only" — 총 운동 에너지(KE)가 엄격히 상수여야 함(B는 일을 하지 않음);
    # "E⊥ + B" — E×B drift는 일을 하지 않으므로 KE가 다시 보존됨;
    # "E∥ + B" — E가 B 방향으로 가속하므로 KE는 증가하고 PE는 감소함;
    # 세 경우를 모두 추적하면 Boris 알고리즘의 어느 부분이 어떤 보존 법칙을
    # 지키는지 드러나고, 나이브 적분기가 실패하는 지점을 보여준다.
    configs = [
        ("B only", lambda x, t: np.array([0, 0, 0]),
                    lambda x, t: np.array([0, 0, B0])),
        ("E⊥ + B", lambda x, t: np.array([1e3, 0, 0]),
                   lambda x, t: np.array([0, 0, B0])),
        ("E∥ + B", lambda x, t: np.array([0, 0, 1e3]),
                   lambda x, t: np.array([0, 0, B0])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, E_func, B_func) in zip(axes, configs):
        electron = ParticleTracer(q=-e, m=m_e, x0=[0, 0, 0], v0=[v0, 0, 0])

        T_gyro = 2 * np.pi * m_e / (e * B0)
        # dt = T_gyro / 100은 일반적으로 사용되는 경험 법칙이다: gyroperiod당 100 스텝이면
        # Boris 회전이 충분히 작아져 이산 궤도가 해석 원에 가깝게 추적되면서도,
        # 명시적 Euler 적분기에서 필요한 것보다 10배 더 거친 스텝을 사용할 수 있다.
        dt = T_gyro / 100
        electron.trace(E_func, B_func, t_max=10*T_gyro, dt=dt)

        # Compute kinetic energy
        KE = 0.5 * m_e * np.sum(electron.v_history**2, axis=1)

        # E∥ 경우에는 정전 퍼텐셜 에너지를 더해 보존량이 KE + PE(전체 역학 에너지)가
        # 되도록 한다. PE를 생략하면 증가하는 KE가 보이고 에너지 비보존으로 오해될 수 있다.
        if name == "E∥ + B":
            PE = -(-e) * 1e3 * electron.x_history[:, 2]
            total_E = KE + PE
        else:
            total_E = KE

        # Normalize
        total_E /= total_E[0]

        ax.plot(electron.t_history / T_gyro, total_E, 'b-', linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time [gyroperiods]', fontsize=11)
        ax.set_ylabel('Normalized Total Energy', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # y 범위 ±0.001은 Boris 알고리즘의 거의 완벽한 보존과 장기 오차를
        # 누적하는 방법들 사이의 0.1% 미만 에너지 드리프트를 드러낸다.
        ax.set_ylim(0.999, 1.001)

    plt.tight_layout()
    plt.savefig('energy_conservation_boris.png', dpi=150)
    plt.show()

validate_energy_conservation()
```

## 요약

전자기장에서의 단일 입자 운동은 플라스마 kinetic theory의 기초를 형성합니다:

1. **Lorentz 힘**은 하전 입자 궤적을 지배합니다: $m d\mathbf{v}/dt = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$, 자기력은 일을 하지 않습니다.

2. 균일한 $\mathbf{B}$에서의 **Cyclotron 운동**은 gyrofrequency $\omega_c = |q|B/m$과 Larmor 반지름 $r_L = mv_\perp/(|q|B)$를 가진 나선형 궤적을 생성합니다.

3. **E×B drift**는 수직 전기장에서 발생하며, 전하와 질량에 독립적인 drift 속도 $\mathbf{v}_E = (\mathbf{E} \times \mathbf{B})/B^2$를 제공합니다.

4. **Guiding center 근사**는 빠른 gyration을 느린 drift로부터 분리하여, $r_L \ll L$일 때 입자 운동의 효율적인 추적을 가능하게 합니다.

5. **Boris 알고리즘**은 임의의 전자기장에서 입자 궤적을 적분하기 위한 강건하고 에너지 보존적인 수치 방법을 제공합니다.

다음 레슨에서, 이 개념들을 비균일 장으로 확장하여, grad-B drift, curvature drift, magnetic mirror 효과를 도입할 것입니다.

## 연습 문제

### 문제 1: Cyclotron 주파수와 반지름

운동 에너지 $E_k = 100$ eV를 가진 전자가 균일한 자기장 $B = 0.5$ T에 수직으로 들어갑니다.

(a) cyclotron 주파수 $\omega_{ce}$와 gyroperiod $T$를 계산하세요.

(b) Larmor 반지름 $r_L$을 구하세요.

(c) 전자가 평행 속도 $v_\parallel = 10^7$ m/s도 가지면, 나선의 피치를 계산하세요.

(d) 같은 에너지를 가진 양성자에 대해 (a)-(c)를 반복하세요.

### 문제 2: E×B Drift 계산

플라스마가 자기장 $\mathbf{B} = 2\hat{\mathbf{z}}$ T에 있습니다. 전기장 $\mathbf{E} = 500\hat{\mathbf{x}}$ V/m이 인가됩니다.

(a) E×B drift 속도 벡터를 계산하세요.

(b) 이 drift가 전자와 양성자에 대해 동일함을 확인하세요.

(c) 전기장이 $\omega = 2\pi \times 10^6$ rad/s로 진동하면, drift 속도가 즉시 따릅니까? ($\omega$를 $\omega_{ce}$와 $\omega_{ci}$와 비교하세요.)

(d) $\mathbf{E} = E_0 \cos(\omega t) \hat{\mathbf{x}}$이면 시간 평균 drift를 추정하세요.

### 문제 3: Magnetron 구성

원통형 magnetron에서, 반지름 $r$의 전자는 다음을 경험합니다:
- 반경 방향 전기장: $E_r = V_0/r$ (여기서 $V_0$는 상수)
- 축 방향 자기장: $B_z = B_0$

(a) 방위각 E×B drift 속도 $v_\theta(r)$를 계산하세요.

(b) $v_\theta$가 최대인 반지름 $r_0$를 구하세요.

(c) $V_0 = 10$ kV와 $B_0 = 0.1$ T에 대해, $r \in [1, 10]$ cm에서 $v_\theta(r)$를 플롯하세요.

(d) $r = 5$ cm에서, 전자 Larmor 반지름을 계산하세요. $r_L \ll r$입니까?

### 문제 4: Boris 알고리즘 분석

초기 속도 $\mathbf{v}_0 = v_0 \hat{\mathbf{x}}$로 $\mathbf{B} = B_0 \hat{\mathbf{z}}$에 있는 입자에 대해 Boris 알고리즘을 구현하세요.

(a) $N = 10, 20, 50, 100$에 대해 timestep $\Delta t = T/N$으로 10 gyroperiod 동안 시뮬레이션하세요.

(b) 각 $N$에 대해, 최종 운동 에너지를 계산하고 초기 값과 비교하세요. 상대 오차 vs. $N$을 플롯하세요.

(c) 10 gyroperiod 후 위치 오차를 측정하세요(입자는 시작점으로 돌아와야 함). $N$에 따라 어떻게 스케일됩니까?

(d) 간단한 Euler 방법으로 반복하고 정확도와 에너지 보존을 비교하세요.

### 문제 5: 결합 장

전하 $q = -e$, 질량 $m = m_e$를 가진 입자가 장에서 원점에서 정지 상태로 시작합니다:
- $\mathbf{E} = E_0 \hat{\mathbf{z}}$ (상수)
- $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (상수)

(a) $v_x(t)$, $v_y(t)$, $v_z(t)$에 대한 운동 방정식을 쓰세요.

(b) $v_z(t)$를 풀으세요. 평행 운동을 설명하세요.

(c) 수직 운동에 대해, 이 경우 $\mathbf{E} \times \mathbf{B} = 0$임에도 guiding center가 E×B 속도로 drift함을 보이세요. (힌트: 이것은 trick 질문입니다 - E와 B가 평행할 때 E×B는 무엇입니까?)

(d) $E_0 = 1000$ V/m, $B_0 = 0.1$ T에 대해 100 gyroperiod 동안 궤적을 수치적으로 적분하세요. $z(t)$를 플롯하고 분석 해를 확인하세요.

---

**이전:** [Plasma Description Hierarchy](./03_Plasma_Description_Hierarchy.md) | **다음:** [Single Particle Motion II](./05_Single_Particle_Motion_II.md)

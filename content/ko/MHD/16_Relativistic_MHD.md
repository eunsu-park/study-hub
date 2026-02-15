# 16. 상대론적 MHD

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

- 공변 표기법을 사용하여 특수 상대론적 MHD(SRMHD) 방정식을 정식화하기
- SRMHD에서 응력-에너지 텐서와 전자기장 텐서 이해하기
- 수치 구현을 위한 SRMHD 방정식의 3+1 분해 유도하기
- 상대론적 원시 변수 복원 문제 해결하기
- 상대론적 MHD의 파동 구조(fast, slow, Alfvén) 분석하기
- 상대론적 제트, 펄서 자기권, 블랙홀 강착에 SRMHD 적용하기
- Python으로 1D SRMHD 충격파 튜브 솔버 구현하기
- 일반 상대론적 MHD(GRMHD)의 기초와 응용 이해하기

---

## 1. 상대론적 MHD 소개

### 1.1 상대론적 MHD가 필요한 경우

비상대론적 MHD는 $v \ll c$를 가정하고 다음을 무시합니다:
- 전자기장의 Lorentz 수축
- Maxwell 방정식의 변위 전류
- 상대론적 질량-에너지 등가성

상대론적 MHD(RMHD)는 다음의 경우 필수적입니다:

```
• 유속이 c에 근접: v/c ~ 0.1-1
  - 상대론적 제트: AGN, GRB (Γ ~ 10-100)
  - 펄서 바람 (Γ ~ 10⁴-10⁶)
  - 강착 디스크 내부 영역 (ISCO에서 v ~ 0.3c)

• 자기 압력이 지배: σ = B²/(4πρc²) ≫ 1
  - Magnetar 자기권: B ~ 10¹⁵ G
  - 펄서 극관

• 강한 중력장:
  - 블랙홀 강착 (r ~ 2-10 GM/c²)
  - 중성자별 병합
```

### 1.2 비상대론적 MHD와의 주요 차이점

| 측면 | 비상대론적 | 상대론적 |
|--------|------------------|--------------|
| 전기장 | $\mathbf{E} = -\mathbf{v} \times \mathbf{B}/c$ | $\mathbf{E}$는 독립적인 동적 변수 |
| 변위 전류 | 무시됨 | $\partial \mathbf{E}/\partial t$ 포함 |
| 보존 법칙 | 질량, 운동량, 에너지 분리 | $\partial_\mu T^{\mu\nu} = 0$에서 통합 |
| 파동 속도 | $v$와 무관 | Lorentz 인자 $W$에 의해 수정 |
| 원시변수 복원 | 대수적 | 암묵적 근찾기 필요 |

---

## 2. 특수 상대론적 MHD (SRMHD)

### 2.1 공변 정식화

**계량과 4-벡터:**

Minkowski 계량 (부호 $-,+,+,+$):
$$
\eta_{\mu\nu} = \text{diag}(-1, 1, 1, 1)
$$

4-속도:
$$
u^\mu = W(c, \mathbf{v}), \quad u^\mu u_\mu = -c^2
$$
여기서 $W = 1/\sqrt{1 - v^2/c^2}$는 Lorentz 인자입니다.

**전자기장 텐서:**
$$
F^{\mu\nu} = \begin{pmatrix}
0 & -E_x/c & -E_y/c & -E_z/c \\
E_x/c & 0 & -B_z & B_y \\
E_y/c & B_z & 0 & -B_x \\
E_z/c & -B_y & B_x & 0
\end{pmatrix}
$$

쌍대 텐서:
$$
F^{*\mu\nu} = \frac{1}{2} \epsilon^{\mu\nu\alpha\beta} F_{\alpha\beta}
$$
여기서 $\epsilon^{\mu\nu\alpha\beta}$는 Levi-Civita 텐서입니다.

### 2.2 공변 형태의 Maxwell 방정식

**소스 없는 Maxwell 방정식:**
$$
\partial_\mu F^{*\mu\nu} = 0 \quad \Rightarrow \quad \nabla \cdot \mathbf{B} = 0, \quad \nabla \times \mathbf{E} + \frac{\partial \mathbf{B}}{\partial t} = 0
$$

**전류 포함 (저항성 RMHD):**
$$
\partial_\mu F^{\mu\nu} = \frac{4\pi}{c} J^\nu
$$
여기서 $J^\mu = (c\rho_e, \mathbf{J})$는 4-전류입니다.

### 2.3 응력-에너지 텐서

**총 응력-에너지 텐서:**
$$
T^{\mu\nu} = T^{\mu\nu}_{\text{fluid}} + T^{\mu\nu}_{\text{EM}}
$$

**유체 기여:**
$$
T^{\mu\nu}_{\text{fluid}} = (\rho h + u_m) u^\mu u^\nu + (p + p_m) \eta^{\mu\nu}
$$
여기서:
- $\rho$ = 정지 질량 밀도
- $h = 1 + \epsilon + p/(\rho c^2)$ = 비엔탈피
- $\epsilon$ = 비내부 에너지
- $u_m = b^2/(8\pi)$ = 공동 좌표계의 자기 에너지 밀도
- $p_m = b^2/(8\pi)$ = 자기 압력 (등방성 부분)

**전자기 기여:**
$$
T^{\mu\nu}_{\text{EM}} = \frac{1}{4\pi} \left( F^{\mu\alpha} F^\nu_{\ \alpha} - \frac{1}{4} \eta^{\mu\nu} F^{\alpha\beta} F_{\alpha\beta} \right)
$$

**4-자기장 (공동 좌표계):**
$$
b^\mu = \frac{1}{c} F^{*\mu\nu} u_\nu = W(\mathbf{v} \cdot \mathbf{B}/c, \mathbf{B}/W + W(\mathbf{v} \cdot \mathbf{B})\mathbf{v}/c^2)
$$

$b^\mu u_\mu = 0$ (직교성)을 만족하며, $b^2 = b^\mu b_\mu = (B^2 + (v \times B)^2/c^2)/W^2$입니다.

### 2.4 보존 법칙

**에너지-운동량 보존:**
$$
\partial_\mu T^{\mu\nu} = 0
$$

다음으로 확장됩니다:
- $\nu = 0$: 에너지 보존
- $\nu = i$: 운동량 보존

**질량 보존:**
$$
\partial_\mu (\rho u^\mu) = 0
$$

---

## 3. 이상 SRMHD

### 3.1 이상 조건

이상 SRMHD에서 전기장은 공동 좌표계에서 사라집니다:
$$
F^{\mu\nu} u_\nu = 0
$$

이는 다음을 제공합니다:
$$
\mathbf{E} = -\frac{\mathbf{v} \times \mathbf{B}}{c} \frac{1}{1 - v^2/c^2}
$$

자기장은 플라즈마에 **동결**됩니다 (상대론적 버전).

### 3.2 3+1 분해

수치 구현을 위해 실험실 좌표계 (3+1) 변수로 분해합니다.

**보존 변수:**
$$
\mathbf{U} = \begin{pmatrix} D \\ \mathbf{S} \\ \tau \\ \mathbf{B} \end{pmatrix}
$$
여기서:
- $D = \rho W$ (보존 밀도)
- $\mathbf{S} = (\rho h + b^2) W^2 \mathbf{v} - (\mathbf{v} \cdot \mathbf{B})\mathbf{B}/(4\pi)$ (운동량 밀도)
- $\tau = (\rho h + b^2) W^2 - p - b^2/2 - D c^2$ (에너지 밀도)
- $\mathbf{B}$ (자기장)

**플럭스 함수:**
$$
\mathbf{F}(\mathbf{U}) = \begin{pmatrix}
D v_x \\
S_x v_x + p_{\text{tot}} - B_x^2/(4\pi) \\
S_y v_x - B_x B_y/(4\pi) \\
S_z v_x - B_x B_z/(4\pi) \\
\tau v_x + p_{\text{tot}} v_x - (\mathbf{v} \cdot \mathbf{B}) B_x/(4\pi) \\
0 \\
B_y v_x - B_x v_y \\
B_z v_x - B_x v_z
\end{pmatrix}
$$
여기서 $p_{\text{tot}} = p + B^2/(8\pi)$입니다.

**보존 형태:**
$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0
$$

### 3.3 원시 변수 복원

**문제:** 보존 변수 $(\mathbf{U})$가 주어졌을 때, 원시 변수 $(\rho, \mathbf{v}, p, \mathbf{B})$를 복원합니다.

**대수적 제약:**
$$
\begin{aligned}
D &= \rho W \\
\mathbf{B} &= \text{(알려짐)} \\
\mathbf{S} &= (\rho h + b^2) W^2 \mathbf{v} - (\mathbf{v} \cdot \mathbf{B})\mathbf{B}/(4\pi) \\
\tau &= (\rho h + b^2) W^2 - p - b^2/2 - D c^2
\end{aligned}
$$

**문제:** $W$, $p$, $\rho$, $\mathbf{v}$를 결합하는 비선형 시스템.

**표준 접근법 (2D Newton-Raphson):**

1. 미지수 선택: $z = W$, $w = p$
2. 역변환:
   $$
   v^2 = 1 - \frac{1}{z^2}
   $$
3. $\mathbf{S}$와 운동량 방정식으로부터 $\rho$, $h$ 해결
4. EOS 사용: $p = p(\rho, \epsilon)$
5. 수렴할 때까지 반복

**대안 (1D 근찾기):**

압력 $p$를 독립 변수로 사용하여 해결:
$$
f(p) = \tau + p - \frac{(\mathbf{S} \cdot \mathbf{B})^2}{(\tau + p + D c^2 + B^2/(4\pi))^2 - (\mathbf{S}^2 + (\mathbf{S} \cdot \mathbf{B})^2/(B^2))} - D c^2 = 0
$$

**도전 과제:**
- 여러 근이 가능
- $W \gg 1$에 대한 수치적 stiffness
- 진공 근처에서 붕괴 ($\rho \to 0$)

**모범 사례:**
- 견고한 근찾기 사용 (Brent 방법)
- 좋은 초기 추측 (이전 타임스텝으로부터)
- $\rho$, $p$에 대한 플로어 값

---

## 4. SRMHD의 파동 구조

### 4.1 고유구조

SRMHD 시스템은 **7개의 파동**을 갖습니다 (1D):
$$
\lambda_{1,7} = \alpha_{\pm}^{\text{fast}}, \quad \lambda_{2,6} = \alpha_{\pm}^{\text{slow}}, \quad \lambda_{3,5} = \alpha_{\pm}^{\text{Alf}}, \quad \lambda_4 = v_x
$$

**상대론적 파동 속도:**

정의:
$$
c_s^2 = \frac{\partial p}{\partial \rho h} \quad \text{(상대론적 음속)}
$$
$$
v_A^2 = \frac{B^2/(4\pi)}{\rho h + B^2/(4\pi)} \quad \text{(상대론적 Alfvén 속도)}
$$

Fast magnetosonic 속도:
$$
\alpha_{\pm}^{\text{fast}} = \frac{v_x \pm c_{\text{fast}}}{1 \pm v_x c_{\text{fast}}/c^2}
$$
여기서:
$$
c_{\text{fast}}^2 = \frac{c_s^2 + v_A^2 - c_s^2 v_A^2}{1 - c_s^2 v_A^2}
$$

Slow magnetosonic 및 Alfvén 속도도 수정된 분산 관계로 유사하게 정의됩니다.

**비상대론적과의 주요 차이점:**
- 모든 파동 속도 < $c$ (인과성)
- Lorentz 인자 $W$가 분산 관계에 포함
- $v \to c$일 때, 파동이 유동을 따라잡을 수 없음 (bunching)

### 4.2 상대론적 Riemann 문제

**SRMHD용 HLLC 솔버:**

파동 속도 추정:
$$
\lambda_L = \min(\lambda_L^{\text{fast}}, \lambda_R^{\text{fast}}), \quad \lambda_R = \max(\lambda_L^{\text{fast}}, \lambda_R^{\text{fast}})
$$

접촉파 속도 $\lambda_*$는 운동량 점프 조건으로부터.

**HLLD 솔버:**

5개의 모든 파동 포함 (fast, Alfvén, contact). 더 정확하지만 복잡함.

**상대론적 Brio-Wu 충격파 튜브:**

초기 조건:
$$
(\rho, v_x, v_y, v_z, p, B_y) = \begin{cases}
(1, 0, 0, 0, 1, 1) & x < 0.5 \\
(0.125, 0, 0, 0, 0.1, -1) & x > 0.5
\end{cases}
$$
$B_x = 0.5$ 상수.

예상 구조: left fast → compound → contact → slow → right fast.

---

## 5. SRMHD의 응용

### 5.1 상대론적 제트

**천체물리학적 맥락:**
- Active Galactic Nuclei (AGN): 초대질량 블랙홀의 제트 (Lorentz 인자 $\Gamma \sim 10-30$)
- Gamma-Ray Bursts (GRB): 항성 질량 블랙홀의 제트 ($\Gamma \sim 100-1000$)
- Microquasar: X선 쌍성계의 제트 ($\Gamma \sim 2-10$)

**물리학:**
- **가속:** 자기 압력이 운동 에너지로 변환 ($\sigma \to 0$)
- **준직:** 자기 hoop stress가 제트 제한
- **불안정성:** Kelvin-Helmholtz (제트-주변 경계면), 전류 구동 kink

**Light cylinder:**

각속도 $\Omega$로 회전하는 자기권의 경우:
$$
R_L = \frac{c}{\Omega}
$$

$R_L$ 내부: corotation 가능. 외부: 자기장이 열리고, 바람 발사.

**수치적 도전 과제:**
- 큰 Lorentz 인자: $W \sim 100$ → stiff 원시변수 복원
- 자화 매개변수 $\sigma = B^2/(4\pi \rho h W^2)$가 수십 년에 걸쳐 변화
- $\sigma \gg 1$일 때 그리드 스케일 불안정성

### 5.2 펄서 자기권

**경사 회전자:**
- 회전 자기 쌍극자: $\mathbf{m}$이 $\boldsymbol{\Omega}$에 각도 $\alpha$로 기울어짐
- 열린 장 선: $\theta < \theta_{\text{pc}}$ (극관 각도)
- 닫힌 장 선: corotating 플라즈마

**Force-free electrodynamics (FFE):**

$\sigma \gg 1$일 때, 플라즈마 관성 무시 가능:
$$
\rho_e \mathbf{E} + \frac{\mathbf{J} \times \mathbf{B}}{c} = 0, \quad \mathbf{E} \cdot \mathbf{B} = 0
$$

경계 조건이 주어진 $\mathbf{E}$, $\mathbf{B}$ 해결 (중성자별 표면, 무한대).

**펄서 바람:**
- 열린 장 선을 따라 입자 가속
- light cylinder에서 자화 $\sigma \sim 10^4$
- 종결 충격파에서 소산 (펄서 바람 성운)

**SRMHD vs FFE:**
- FFE: $\sigma \gg 1$에 대해 유효, 소산 없음
- SRMHD: 관성, 재결합, 입자 가열 포함

### 5.3 블랙홀 강착

**Innermost Stable Circular Orbit (ISCO):**
- Schwarzschild: $r_{\text{ISCO}} = 6 GM/c^2$
- Kerr (극단적): $r_{\text{ISCO}} = GM/c^2$ (순행)

ISCO에서의 궤도 속도:
$$
v_{\text{ISCO}} \sim 0.5c \quad \text{(Schwarzschild)} \quad \text{to} \quad 0.7c \quad \text{(Kerr, 순행)}
$$

**제트 발사:**

Blandford-Znajek 메커니즘 (회전 블랙홀):
- 자기장이 지평선을 관통
- Frame dragging: $\Omega_H$ (지평선 각속도)
- Poynting 플럭스 추출: $L_{\text{BZ}} \sim \Omega_H^2 B_H^2 r_H^4 / c$

**일반 상대론적 MHD** (GRMHD) 필요.

**Magnetically Arrested Disk (MAD):**

자기 플럭스가 축적될 때:
$$
\phi \sim \sqrt{\dot{M} r_g c} \quad \Rightarrow \quad \text{자기적으로 지배됨}
$$

결과:
- 억제된 강착 (플럭스 장벽)
- 향상된 제트 파워
- 시간 변동 유동

---

## 6. 일반 상대론적 MHD (GRMHD)

### 6.1 곡선 시공간 정식화

**Kerr 계량 (Boyer-Lindquist 좌표):**
$$
ds^2 = -\alpha^2 dt^2 + \gamma_{ij} (dx^i + \beta^i dt)(dx^j + \beta^j dt)
$$
여기서:
- $\alpha$ = lapse 함수
- $\beta^i$ = shift 벡터
- $\gamma_{ij}$ = 공간 계량

**응력-에너지 텐서:**
$$
\nabla_\mu T^{\mu\nu} = 0
$$
여기서 $\nabla_\mu$는 공변 미분 (Christoffel 기호 포함).

**3+1 ADM 형태:**

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}^i}{\partial x^i} = \mathbf{S}
$$

여기서 $\mathbf{S}$는 계량 소스 항 (시공간 곡률) 포함.

### 6.2 HARM 코드 정식화

**HARM** (High Accuracy Relativistic Magnetohydrodynamics) 코드는 사용합니다:

- 보존 변수: $\sqrt{-g} (\rho u^t, T^t_{\ i}, \sqrt{-g} B^i)$
- Flux-conservative 방식
- 발산 없는 $\mathbf{B}$를 위한 Flux-CT
- 곡선 시공간에서 원시변수 복원

**계량:**
수정된 Kerr-Schild 좌표 (지평선 관통).

**응용:**
- 블랙홀 강착 디스크 (Sgr A*, M87)
- 중성자별 병합
- 제트 발사 시뮬레이션

### 6.3 지평선 경계 조건

**Excision:**
- 지평선 내부 점 제거 (인과적으로 단절됨)
- 유출 경계 조건

**유입 평형:**
- 물질이 유입 속도로 안쪽으로 떨어지도록 강제
- 정상 디스크 프로파일 유지

---

## 7. SRMHD의 수치 방법

### 7.1 Godunov형 방식

**HLLC 플럭스:**

$$
\mathbf{F}_{\text{HLLC}} = \begin{cases}
\mathbf{F}_L & \lambda_L \geq 0 \\
\mathbf{F}_L^* & \lambda_L < 0 \leq \lambda_* \\
\mathbf{F}_R^* & \lambda_* < 0 \leq \lambda_R \\
\mathbf{F}_R & \lambda_R < 0
\end{cases}
$$

Star 상태 $\mathbf{U}_L^*$, $\mathbf{U}_R^*$는 Rankine-Hugoniot 점프 조건으로부터.

**시간 단계:**

- RK2 또는 RK3 (TVD)
- CFL 조건:
$$
\Delta t \leq C \min_i \frac{\Delta x_i}{c_{\text{fast}, i} + |v_i|}
$$
여기서 $C \sim 0.4-0.5$ (비상대론적보다 더 제한적).

### 7.2 적응형 타임스텝

높은 Lorentz 인자 ($W \gg 1$)에 대해 로컬 타임스텝이 아주 작을 수 있음. 사용:

- 계층적 타임스텝핑 (AMR 코드)
- Implicit-explicit (IMEX) 방식 (stiff 항 암묵적)

### 7.3 플로어와 상한

**밀도 플로어:**
$$
\rho \geq \rho_{\min} = 10^{-6} \rho_{\text{max}}
$$

**온도 상한:**
$$
T \leq T_{\max} = 10^{13} \, \text{K} \quad \text{(쌍생성 방지)}
$$

**자화 상한:**
$$
\sigma \leq \sigma_{\max} \sim 100 \quad \text{(수치적 불안정성 방지)}
$$

---

## 8. Python 구현: 1D SRMHD 충격파 튜브

### 8.1 문제 설정

상대론적 Brio-Wu 테스트:
$$
(\rho, v_x, p, B_y) = \begin{cases}
(1.0, 0.0, 1.0, 1.0) & x < 0.5 \\
(0.125, 0.0, 0.1, -1.0) & x \geq 0.5
\end{cases}
$$
$B_x = 0.5$ (상수), $\Gamma = 5/3$ (이상 기체).

영역: $x \in [0, 1]$, $t \in [0, 0.4]$.

### 8.2 코드 구조

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.0  # Speed of light
GAMMA = 5.0/3.0  # Adiabatic index

# Grid
NX = 400
XL, XR = 0.0, 1.0
dx = (XR - XL) / NX
x = np.linspace(XL + dx/2, XR - dx/2, NX)

# Primitive variables: [rho, vx, vy, vz, p, Bx, By, Bz]
def initial_conditions():
    prim = np.zeros((NX, 8))
    Bx_const = 0.5

    for i in range(NX):
        if x[i] < 0.5:
            prim[i] = [1.0, 0.0, 0.0, 0.0, 1.0, Bx_const, 1.0, 0.0]
        else:
            prim[i] = [0.125, 0.0, 0.0, 0.0, 0.1, Bx_const, -1.0, 0.0]

    return prim

# Lorentz factor
def lorentz_factor(vx, vy, vz):
    v2 = vx**2 + vy**2 + vz**2
    return 1.0 / np.sqrt(1.0 - v2 / C**2)

# Primitive to conserved
def prim2cons(prim):
    rho, vx, vy, vz, p = prim[:5]
    Bx, By, Bz = prim[5:]

    W = lorentz_factor(vx, vy, vz)
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    vdotB = vx*Bx + vy*By + vz*Bz

    # Specific enthalpy
    h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

    # Comoving magnetic field
    b0 = W * vdotB / C
    bx = Bx/W + W*vx*vdotB/C**2
    by = By/W + W*vy*vdotB/C**2
    bz = Bz/W + W*vz*vdotB/C**2
    b2 = (B2 + (vdotB)**2/C**2) / W**2

    # Conserved variables
    D = rho * W
    Sx = (rho*h + b2)*W**2*vx - (vdotB)*Bx/(4.0*np.pi)
    Sy = (rho*h + b2)*W**2*vy - (vdotB)*By/(4.0*np.pi)
    Sz = (rho*h + b2)*W**2*vz - (vdotB)*Bz/(4.0*np.pi)
    tau = (rho*h + b2)*W**2 - p - b2/2.0 - D*C**2

    return np.array([D, Sx, Sy, Sz, tau, Bx, By, Bz])

# Conserved to primitive (simplified 1D Newton-Raphson)
def cons2prim(cons, prim_guess):
    D, Sx, Sy, Sz, tau = cons[:5]
    Bx, By, Bz = cons[5:]

    # Initial guess
    rho, vx, vy, vz, p = prim_guess[:5]

    # Newton-Raphson iteration (simplified)
    max_iter = 50
    tol = 1e-10

    for iteration in range(max_iter):
        W = lorentz_factor(vx, vy, vz)
        h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

        vdotB = vx*Bx + vy*By + vz*Bz
        b2 = (Bx**2 + By**2 + Bz**2 + (vdotB)**2/C**2) / W**2

        # Residuals
        f1 = D - rho*W
        f2 = Sx - ((rho*h + b2)*W**2*vx - (vdotB)*Bx/(4.0*np.pi))
        f3 = tau - ((rho*h + b2)*W**2 - p - b2/2.0 - D*C**2)

        if abs(f1) + abs(f2) + abs(f3) < tol:
            break

        # Simple update (damped)
        rho = D / W
        p_new = (tau + D*C**2 + p + b2/2.0) / (W**2) - rho*h - b2
        p = 0.5 * p + 0.5 * max(p_new, 1e-10)

        # Update velocity (simplified)
        S2 = Sx**2 + Sy**2 + Sz**2
        S = np.sqrt(S2)
        if S > 1e-12:
            vx = Sx / (rho*h*W**2)
            vy = Sy / (rho*h*W**2)
            vz = Sz / (rho*h*W**2)

        # Limit velocity
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v >= C:
            scale = 0.99 * C / v
            vx *= scale
            vy *= scale
            vz *= scale

    # Apply floors
    rho = max(rho, 1e-10)
    p = max(p, 1e-10)

    return np.array([rho, vx, vy, vz, p, Bx, By, Bz])

# Flux function
def flux(prim):
    rho, vx, vy, vz, p = prim[:5]
    Bx, By, Bz = prim[5:]

    W = lorentz_factor(vx, vy, vz)
    vdotB = vx*Bx + vy*By + vz*Bz
    B2 = Bx**2 + By**2 + Bz**2
    b2 = (B2 + (vdotB)**2/C**2) / W**2
    h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho

    ptot = p + B2/(8.0*np.pi)

    F = np.zeros(8)
    F[0] = rho * W * vx
    F[1] = (rho*h + b2)*W**2*vx*vx + ptot - Bx**2/(4.0*np.pi) - (vdotB)*Bx*vx/(4.0*np.pi)
    F[2] = (rho*h + b2)*W**2*vx*vy - Bx*By/(4.0*np.pi) - (vdotB)*Bx*vy/(4.0*np.pi)
    F[3] = (rho*h + b2)*W**2*vx*vz - Bx*Bz/(4.0*np.pi) - (vdotB)*Bx*vz/(4.0*np.pi)
    F[4] = ((rho*h + b2)*W**2 - p - b2/2.0)*vx + ptot*vx - (vdotB)*Bx/(4.0*np.pi)
    F[5] = 0.0  # Bx constant
    F[6] = By*vx - Bx*vy
    F[7] = Bz*vx - Bx*vz

    return F

# HLLC Riemann solver (simplified HLL for SRMHD)
def hll_flux(pL, pR):
    # Estimate wave speeds (very simplified)
    rhoL, vxL, pL_val = pL[0], pL[1], pL[4]
    rhoR, vxR, pR_val = pR[0], pR[1], pR[4]

    # Sound speed
    csL = np.sqrt(GAMMA * pL_val / (rhoL * (1.0 + GAMMA/(GAMMA-1.0)*pL_val/rhoL)))
    csR = np.sqrt(GAMMA * pR_val / (rhoR * (1.0 + GAMMA/(GAMMA-1.0)*pR_val/rhoR)))

    WL = lorentz_factor(pL[1], pL[2], pL[3])
    WR = lorentz_factor(pR[1], pR[2], pR[3])

    # Fast magnetosonic speed estimate (crude)
    BxL, ByL, BzL = pL[5:]
    BxR, ByR, BzR = pR[5:]
    B2L = BxL**2 + ByL**2 + BzL**2
    B2R = BxR**2 + ByR**2 + BzR**2

    hL = 1.0 + GAMMA/(GAMMA-1.0)*pL_val/rhoL
    hR = 1.0 + GAMMA/(GAMMA-1.0)*pR_val/rhoR

    vAL = np.sqrt(B2L/(4.0*np.pi*(rhoL*hL + B2L/(4.0*np.pi))))
    vAR = np.sqrt(B2R/(4.0*np.pi*(rhoR*hR + B2R/(4.0*np.pi))))

    cfL = np.sqrt((csL**2 + vAL**2 - csL**2*vAL**2)/(1.0 - csL**2*vAL**2))
    cfR = np.sqrt((csR**2 + vAR**2 - csR**2*vAR**2)/(1.0 - csR**2*vAR**2))

    # Wave speeds (relativistic addition)
    lamL = (vxL - cfL) / (1.0 - vxL*cfL/C**2)
    lamR = (vxR + cfR) / (1.0 + vxR*cfR/C**2)

    # HLL flux
    consL = prim2cons(pL)
    consR = prim2cons(pR)
    FL = flux(pL)
    FR = flux(pR)

    if lamL >= 0:
        return FL
    elif lamR <= 0:
        return FR
    else:
        F_hll = (lamR*FL - lamL*FR + lamL*lamR*(consR - consL)) / (lamR - lamL)
        return F_hll

# Main evolution
def evolve_srmhd():
    prim = initial_conditions()
    cons = np.array([prim2cons(p) for p in prim])

    t = 0.0
    t_end = 0.4
    CFL = 0.4

    snapshots = []

    while t < t_end:
        # Compute dt
        v_max = 0.0
        for i in range(NX):
            rho, vx, vy, vz, p = prim[i, :5]
            Bx, By, Bz = prim[i, 5:]

            cs = np.sqrt(GAMMA * p / (rho * (1.0 + GAMMA/(GAMMA-1.0)*p/rho)))
            B2 = Bx**2 + By**2 + Bz**2
            h = 1.0 + GAMMA/(GAMMA-1.0)*p/rho
            vA = np.sqrt(B2/(4.0*np.pi*(rho*h + B2/(4.0*np.pi))))
            cf = np.sqrt((cs**2 + vA**2)/(1.0 + cs**2*vA**2))

            W = lorentz_factor(vx, vy, vz)
            v_signal = max(abs((vx + cf)/(1.0 + vx*cf/C**2)),
                          abs((vx - cf)/(1.0 - vx*cf/C**2)))
            v_max = max(v_max, v_signal)

        dt = CFL * dx / v_max
        if t + dt > t_end:
            dt = t_end - t

        # RK2 time integration
        # Stage 1
        flux_arr = np.zeros((NX+1, 8))
        for i in range(NX+1):
            if i == 0:
                flux_arr[i] = flux(prim[0])
            elif i == NX:
                flux_arr[i] = flux(prim[-1])
            else:
                flux_arr[i] = hll_flux(prim[i-1], prim[i])

        cons_1 = cons.copy()
        for i in range(NX):
            cons_1[i] -= dt/dx * (flux_arr[i+1] - flux_arr[i])

        # Recover primitives
        prim_1 = np.array([cons2prim(cons_1[i], prim[i]) for i in range(NX)])

        # Stage 2
        for i in range(NX+1):
            if i == 0:
                flux_arr[i] = flux(prim_1[0])
            elif i == NX:
                flux_arr[i] = flux(prim_1[-1])
            else:
                flux_arr[i] = hll_flux(prim_1[i-1], prim_1[i])

        cons_2 = cons_1.copy()
        for i in range(NX):
            cons_2[i] -= dt/dx * (flux_arr[i+1] - flux_arr[i])

        cons = 0.5 * (cons + cons_2)
        prim = np.array([cons2prim(cons[i], prim_1[i]) for i in range(NX)])

        t += dt

        if len(snapshots) < 5:
            if t >= len(snapshots) * t_end / 4:
                snapshots.append((t, prim.copy()))

    return prim, snapshots

# Run simulation
print("Running SRMHD shock tube...")
prim_final, snapshots = evolve_srmhd()

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

vars_to_plot = [
    ('Density', 0),
    ('Velocity vx', 1),
    ('Pressure', 4),
    ('By', 6),
    ('Lorentz Factor', None),
    ('Energy Density', None)
]

for idx, (var_name, var_idx) in enumerate(vars_to_plot):
    ax = axes[idx // 2, idx % 2]

    if var_idx is not None:
        ax.plot(x, prim_final[:, var_idx], 'b-', linewidth=1.5, label='t=0.4')
    else:
        if 'Lorentz' in var_name:
            W = np.array([lorentz_factor(p[1], p[2], p[3]) for p in prim_final])
            ax.plot(x, W, 'b-', linewidth=1.5)
        elif 'Energy' in var_name:
            energy = prim_final[:, 4] / (GAMMA - 1.0)
            ax.plot(x, energy, 'b-', linewidth=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel(var_name)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{var_name} at t=0.4')

plt.tight_layout()
plt.savefig('srmhd_shock_tube.png', dpi=150, bbox_inches='tight')
print("Plot saved: srmhd_shock_tube.png")
plt.close()

# Plot wave structure
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red', 'orange', 'green', 'blue']
for i, (t, prim_snap) in enumerate(snapshots):
    ax.plot(x, prim_snap[:, 0], color=colors[i], label=f't={t:.2f}', alpha=0.7)

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('SRMHD Shock Tube: Density Evolution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('srmhd_evolution.png', dpi=150, bbox_inches='tight')
print("Plot saved: srmhd_evolution.png")
plt.close()
```

### 8.3 예상 출력

코드는 다음을 생성합니다:
- 충격파에서의 밀도 점프
- 희박화 팬
- 접촉 불연속
- 자기장 반전
- 제트형 특징에서 Lorentz 인자 피크

**도전 과제:**
- 초기 추측이 좋지 않으면 원시변수 복원 실패 가능
- Newton 반복에서 감쇠 필요
- 플로어가 비물리적 상태 방지

---

## 9. 상대론적 Alfvén 속도

### 9.1 비교: 비상대론적 vs 상대론적

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 1.0
p = 1.0
B_range = np.logspace(-2, 2, 100)
GAMMA = 5.0/3.0

# Non-relativistic Alfven speed
vA_nr = B_range / np.sqrt(4.0 * np.pi * rho)

# Relativistic Alfven speed
h = 1.0 + GAMMA/(GAMMA-1.0) * p/rho
vA_r = B_range / np.sqrt(4.0*np.pi*(rho*h + B_range**2/(4.0*np.pi)))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(B_range, vA_nr, 'b-', linewidth=2, label='Non-relativistic')
ax.loglog(B_range, vA_r, 'r--', linewidth=2, label='Relativistic')
ax.axhline(1.0, color='k', linestyle=':', label='Speed of light c=1')
ax.set_xlabel('Magnetic Field B')
ax.set_ylabel('Alfvén Speed $v_A$')
ax.set_title('Alfvén Speed: Non-Relativistic vs Relativistic')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('alfven_speed_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved: alfven_speed_comparison.png")
plt.close()
```

**핵심 관찰:**
- 비상대론적 $v_A$는 큰 $B$에 대해 $c$를 초과 가능 (비물리적)
- 상대론적 $v_A < c$ 항상 ($B \to \infty$일 때 포화)

---

## 10. Light Cylinder 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# Pulsar parameters
R_NS = 1.0  # Neutron star radius
Omega = 1.0  # Angular velocity
c = 1.0
R_L = c / Omega  # Light cylinder radius

# Grid
theta = np.linspace(0, 2*np.pi, 100)
r = np.linspace(0.5*R_NS, 3*R_L, 100)
R, Theta = np.meshgrid(r, theta)

# Corotation velocity
v_corot = Omega * R

# Plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Velocity field (normalized)
v_norm = np.clip(v_corot / c, 0, 1.5)
cf = ax.contourf(Theta, R, v_norm, levels=20, cmap='RdYlBu_r')

# Light cylinder
ax.plot(theta, R_L * np.ones_like(theta), 'k--', linewidth=2, label='Light Cylinder')

# Neutron star
ax.fill_between(theta, 0, R_NS, color='gray', alpha=0.5, label='Neutron Star')

ax.set_ylim(0, 3*R_L)
ax.set_title('Pulsar Magnetosphere: Corotation Velocity\n(Dashed = Light Cylinder)', pad=20)
plt.colorbar(cf, ax=ax, label='$v_{corot}/c$')
ax.legend(loc='upper right')
plt.savefig('light_cylinder.png', dpi=150, bbox_inches='tight')
print("Plot saved: light_cylinder.png")
plt.close()
```

$R_L$ 내부: corotation 가능 ($v < c$).
$R_L$ 외부: 장 선이 열리고, 입자 탈출 (펄서 바람).

---

## 요약

상대론적 MHD는 고전 MHD를 $v \sim c$ 영역으로 확장합니다:

1. **SRMHD 정식화:** 공변 4-텐서 접근법, 응력-에너지 텐서, 동결 조건
2. **3+1 분해:** 수치 구현을 위한 실험실 좌표계 보존 변수
3. **원시변수 복원:** 비선형 암묵적 해결 (주요 수치적 도전)
4. **파동 구조:** 7개의 파동, 상대론적 분산 (모든 속도 < $c$)
5. **응용:** 상대론적 제트, 펄서 자기권, 블랙홀 강착
6. **GRMHD:** 곡선 시공간, Kerr 계량, 지평선 관통 좌표
7. **수치 방법:** HLLC/HLLD Riemann 솔버, 적응형 타임스텝핑, 플로어

**주요 도전 과제:**
- 원시변수 복원 견고성
- 높은 Lorentz 인자 처리 ($W \gg 1$)
- 수십 년에 걸쳐 변화하는 자화 매개변수 $\sigma$
- 복사, 쌍생성과의 결합 (이상 MHD를 넘어서)

상대론적 MHD는 우주에서 가장 에너지가 높은 현상 이해에 필수적입니다: 제트, 펄서, 블랙홀, 중성자별 병합.

---

## 연습 문제

1. **Lorentz 변환:**
   실험실 좌표계에서 $\mathbf{E} = (1, 0, 0)$, $\mathbf{B} = (0, 1, 0)$가 주어졌을 때, $\mathbf{v} = (0.5c, 0, 0)$로 움직이는 좌표계에서 $\mathbf{E}'$와 $\mathbf{B}'$를 계산하세요. $\mathbf{E}' \cdot \mathbf{B}' = \mathbf{E} \cdot \mathbf{B}$ (Lorentz 불변량) 확인하세요.

2. **원시변수 복원:**
   1D Newton-Raphson 원시변수 복원 루틴을 구현하세요. 테스트: $D = 2.0$, $S_x = 1.0$, $\tau = 3.0$, $B_x = 0.5$, $B_y = 1.0$. 초기 추측: $\rho = 1.0$, $v_x = 0.3$, $p = 1.0$. 수렴합니까? 몇 번 반복합니까?

3. **상대론적 Brio-Wu:**
   충격파 튜브 코드를 $B_x = 0$ (순수 횡방향 장)으로 수정하세요. 파동 속도와 구조를 $B_x = 0.5$ 경우와 비교하세요. 왜 해가 다릅니까?

4. **자화 매개변수:**
   $B = 10^{12}$ G, $\rho = 10^7$ g/cm³, $\Gamma = 10$인 펄서에 대해 $\sigma = B^2/(4\pi \rho h W^2 c^2)$를 계산하세요. $h \approx 1$로 가정하세요. 이것은 force-free ($\sigma \gg 1$)입니까, 유체 지배입니까?

5. **Light Cylinder:**
   Crab 펄서 ($P = 33$ ms)에 대해 $R_L$을 계산하세요. 어느 반경에서 corotation 속도가 $c$와 같습니까? 펄서 반경 $R_{NS} \sim 10$ km와 비교하세요.

6. **ISCO 속도:**
   Schwarzschild 블랙홀에 대해 ISCO ($r = 6GM/c^2$)에서 궤도 속도를 계산하세요. Lorentz 인자 $W$는 얼마입니까? $a = 0.998$ (거의 극단적)인 Kerr 블랙홀의 순행 ISCO에서 반복하세요.

7. **Alfvén 속도 포화:**
   고정된 $\rho = 1$, $p = 0.1$, $\Gamma = 4/3$에 대해 상대론적 Alfvén 속도 $v_A = B/\sqrt{4\pi(\rho h + B^2/(4\pi))}$ vs $B$를 그리세요. $B \to \infty$일 때 $v_A \to c$이지만 $c$를 절대 초과하지 않음을 보이세요.

8. **HARM 타임스텝:**
   블랙홀 지평선 근처의 GRMHD에서 lapse 함수 $\alpha \to 0$입니다. 왜 더 작은 타임스텝이 필요합니까? $\Delta r = 0.01 GM/c^2$인 방사형 그리드에 대해 $r = 2.01 GM/c^2$ 근처의 CFL 타임스텝을 추정하세요.

---

**이전:** [2D MHD 솔버](./15_2D_MHD_Solver.md) | **다음:** [스펙트럴 및 고급 방법](./17_Spectral_Methods.md)

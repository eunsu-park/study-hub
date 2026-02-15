# 1. MHD 평형

## 학습 목표

- 이상 MHD 운동량 방정식으로부터 MHD 힘 균형 방정식 유도
- 힘 균형의 결과 이해: 플럭스 표면 상에서 압력 일정
- 1차원 평형 분석: θ-pinch, Z-pinch, screw pinch 구성
- 축대칭 평형에 대한 Grad-Shafranov 방정식 수식화 및 풀이
- 안전 인자 q 계산 및 안정성에 대한 의미 이해
- 플라즈마 베타 계산 및 운영 한계 이해
- 간단한 평형 구성에 대한 수치 해 구현

## 1. MHD 평형 소개

자기유체역학 평형은 모든 힘이 균형을 이루고 알짜 가속도가 없는 자화 플라즈마의 정상 상태 구성을 설명합니다. 평형을 이해하는 것은 핵융합 에너지 연구, 천체물리학 플라즈마 물리학, 그리고 갇힌 플라즈마와 관련된 모든 응용에 기본적입니다.

MHD 평형은 다음을 만족합니다:

```
∂/∂t = 0  (시간 독립)
v = 0      (정지 플라즈마)
```

평형 상태는 다음 사이의 균형에 의해 지배됩니다:
- **플라즈마 압력 구배 힘**: ∇p (외부로 밀어냄)
- **자기 장력**: (B·∇)B/μ₀ (자기장선을 따라 당김)
- **자기 압력 구배**: -∇(B²/2μ₀) (높은 장에서 낮은 장으로 밀어냄)

힘 균형 방정식은 모든 평형 계산의 기초입니다.

## 2. 힘 균형 방정식 유도

### 2.1 이상 MHD 운동량 방정식에서 시작

이상 MHD 운동량 방정식은:

$$
\rho\frac{D\mathbf{v}}{Dt} = -\nabla p + \mathbf{J}\times\mathbf{B}
$$

평형에서 좌변은 사라집니다 (가속도 없음, 정지 플라즈마):

$$
\nabla p = \mathbf{J}\times\mathbf{B}
$$

이것이 **기본 MHD 평형 방정식**입니다.

### 2.2 자기장으로 표현

앙페르 법칙 사용 (변위 전류 무시):

$$
\mathbf{J} = \frac{1}{\mu_0}\nabla\times\mathbf{B}
$$

힘 균형은 다음과 같이 됩니다:

$$
\nabla p = \frac{1}{\mu_0}(\nabla\times\mathbf{B})\times\mathbf{B}
$$

벡터 항등식 사용:

$$
(\nabla\times\mathbf{B})\times\mathbf{B} = (\mathbf{B}\cdot\nabla)\mathbf{B} - \nabla\left(\frac{B^2}{2}\right)
$$

다음을 얻습니다:

$$
\nabla p = \frac{1}{\mu_0}\left[(\mathbf{B}\cdot\nabla)\mathbf{B} - \nabla\left(\frac{B^2}{2}\right)\right]
$$

재배열:

$$
\nabla\left(p + \frac{B^2}{2\mu_0}\right) = \frac{1}{\mu_0}(\mathbf{B}\cdot\nabla)\mathbf{B}
$$

좌변은 **전체 압력** (운동 + 자기)의 구배입니다:

$$
p_{total} = p + \frac{B^2}{2\mu_0}
$$

우변은 자기장선을 따른 **자기 장력**을 나타냅니다.

### 2.3 물리적 해석

```
힘 균형 성분:
=======================

1. 압력 구배: ∇p
   - 높은 압력에서 낮은 압력으로 밀어냄
   - 등방성 힘

2. 자기 압력: -∇(B²/2μ₀)
   - 높은 장에서 낮은 장으로 밀어냄
   - B에 수직으로 작용

3. 자기 장력: (B·∇)B/μ₀
   - 굽은 자기장선을 따라 당김
   - 늘어난 고무줄처럼

알짜 힘: ∇p + ∇(B²/2μ₀) - (B·∇)B/μ₀ = 0
```

## 3. 힘 균형의 결과

### 3.1 플럭스 표면 상에서 압력 일정

힘 균형 방정식을 **B**와 내적하면:

$$
\mathbf{B}\cdot\nabla p = \mathbf{B}\cdot(\mathbf{J}\times\mathbf{B}) = 0
$$

이는 다음을 의미합니다:

$$
\mathbf{B}\cdot\nabla p = 0
$$

**결과**: 압력은 자기장선을 따라 일정합니다. 토로이달 구성에서 자기장선은 중첩된 플럭스 표면 위에 놓여있으므로 **압력은 각 플럭스 표면에서 일정**합니다.

### 3.2 압력 구배에 수직인 전류

힘 균형 방정식을 **J**와 내적하면:

$$
\mathbf{J}\cdot\nabla p = \mathbf{J}\cdot(\mathbf{J}\times\mathbf{B}) = 0
$$

이는 다음을 의미합니다:

$$
\mathbf{J}\cdot\nabla p = 0
$$

**결과**: 전류는 압력 구배에 수직으로 흐르므로, 전류도 플럭스 표면 위에 놓입니다.

### 3.3 플럭스 표면 좌표계

위의 두 결과는 **B**와 **J** 모두 일정한 압력의 표면 위에 놓인다는 것을 의미합니다. 이 표면들을 **자기 플럭스 표면**이라고 합니다. 이것이 토카막 평형 계산에 사용되는 플럭스 표면 좌표계의 기초입니다.

```
플럭스 표면 구조:
======================

        ┌─────────────┐
        │             │   외부 플럭스 표면 (낮은 p)
        │  ┌───────┐  │
        │  │       │  │   중간 플럭스 표면
        │  │   ┌─┐ │  │
        │  │   │·│ │  │   자기 축 (최고 p)
        │  │   └─┘ │  │
        │  │       │  │
        │  └───────┘  │
        │             │
        └─────────────┘

특성:
- p = p(ψ) 여기서 ψ는 플럭스 표면 레이블
- B는 플럭스 표면 내에 놓임
- J는 플럭스 표면 내에 놓임
```

## 4. 1차원 평형

### 4.1 θ-Pinch (종방향 자기장)

θ-pinch는 순수 종방향 (축) 자기장을 사용하여 플라즈마를 방사상으로 가둡니다.

**구성**:
- 원통 기하학 (r, θ, z)
- $B_z(r)$, $p(r)$
- $B_r = B_θ = 0$
- 플라즈마에 전류 없음: $J_z = 0$

**힘 균형**:

원통 좌표계에서:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_z^2}{2}\right)
$$

$r$에서 $r=a$의 플라즈마 경계까지 적분 (여기서 $p(a)=0$):

$$
p(r) = \frac{B_z^2(a) - B_z^2(r)}{2\mu_0}
$$

**물리적 그림**:
- 플라즈마 압력은 자기 압력에 의해 균형
- 자기 장력 없음 (직선 자기장선)
- 순수 방사상 가둠

**Bennett 관계** (전체 압력 균형):

단면에 대해 적분:

$$
\int_0^a 2\pi r\, p(r)\, dr = \frac{\pi a^2 B_{ext}^2}{2\mu_0}
$$

여기서 $B_{ext}$는 외부 장입니다.

### 4.2 Z-Pinch (방위각 자기장)

Z-pinch는 축방향 전류로부터 자체 생성된 방위각 자기장을 사용합니다.

**구성**:
- $B_θ(r)$, $p(r)$
- $J_z(r)$ (축방향 전류)
- $B_r = B_z = 0$

**앙페르 법칙**:

$$
B_θ(r) = \frac{\mu_0}{2\pi r}\int_0^r J_z(r')2\pi r'\, dr' = \frac{\mu_0 I(r)}{2\pi r}
$$

여기서 $I(r)$은 반지름 $r$ 내에 둘러싸인 전류입니다.

**힘 균형**:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_θ^2}{2}\right) = -\frac{B_θ}{\mu_0 r}
$$

$B_θ = \mu_0 I/(2\pi r)$ 사용:

$$
\frac{dp}{dr} = -\frac{J_z B_θ}{\mu_0}
$$

**Bennett 관계** (차원 분석):

균일한 전류 밀도 $J_z = I/(\pi a^2)$에 대해:

$$
I^2 = \frac{8\pi}{\mu_0}NkT
$$

여기서 $N$은 총 입자 수이고 $T$는 온도입니다.

**물리적 그림**:
- 전류가 방위각 장 생성
- 자기 압력이 플라즈마를 안쪽으로 조임
- 플라즈마 압력이 바깥쪽으로 밀어냄
- 매우 불안정 (kink, sausage 불안정성)

### 4.3 Screw Pinch (결합 장)

Screw pinch는 개선된 안정성을 위해 축방향과 방위각 장을 결합합니다.

**구성**:
- $B_z(r)$, $B_θ(r)$, $p(r)$
- $J_z(r)$, $J_θ(r)$

**힘 균형**:

$$
\frac{dp}{dr} = -\frac{1}{\mu_0}\frac{d}{dr}\left(\frac{B_z^2 + B_θ^2}{2}\right)
$$

**안전 인자** (자기장선의 피치):

$$
q(r) = \frac{rB_z}{RB_θ}
$$

여기서 $R$은 주요 반지름 (토로이달 시스템) 또는 특성 길이입니다.

**물리적 그림**:
- $B_z$는 kink 모드에 대한 안정성 제공
- $B_θ$는 가둠 제공
- $q(r)$의 전단이 짧은 파장 모드 안정화
- 토카막 가둠의 기초

```
Screw Pinch 자기장선:
======================

      z ^
        |    /
        |   / ← 자기장선 나선
        |  /
        | /
        |/________> θ

q = Δz / (2πR) 폴로이달 회전당
```

## 5. Grad-Shafranov 방정식

### 5.1 축대칭 평형

축대칭 토로이달 시스템 (토카막, 단순화된 형태의 stellarator)에 대해, 평형은 단일 스칼라 함수로 설명될 수 있습니다: **폴로이달 플럭스 함수** $\psi(R,Z)$.

**원통 좌표계**: $(R, \phi, Z)$ 여기서 $\phi$는 토로이달 각입니다.

**자기장 표현**:

$$
\mathbf{B} = F(R,Z)\nabla\phi + \nabla\phi\times\nabla\psi
$$

여기서:
- $F(R,Z) = RB_\phi$ (토로이달 장 함수)
- $\psi$는 폴로이달 플럭스 함수

성분 형태로:

$$
B_R = \frac{1}{R}\frac{\partial\psi}{\partial Z}, \quad B_Z = -\frac{1}{R}\frac{\partial\psi}{\partial R}, \quad B_\phi = \frac{F}{R}
$$

### 5.2 Grad-Shafranov 방정식 유도

힘 균형에서 시작:

$$
\nabla p = \mathbf{J}\times\mathbf{B}
$$

그리고 $\mathbf{J} = \nabla\times\mathbf{B}/\mu_0$ 사용, 토로이달 성분은:

$$
\frac{dp}{d\psi} = -\frac{1}{\mu_0 R^2}\frac{dF}{d\psi}F
$$

폴로이달 성분은 **Grad-Shafranov 방정식**을 제공합니다:

$$
\Delta^*\psi \equiv R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right) + \frac{\partial^2\psi}{\partial Z^2} = -\mu_0 R^2\frac{dp}{d\psi} - F\frac{dF}{d\psi}
$$

이것은 $\psi(R,Z)$에 대한 **타원형 편미분 방정식**입니다.

### 5.3 자유 함수

Grad-Shafranov 방정식은 두 개의 **자유 함수** 지정을 필요로 합니다:

1. **압력 프로파일**: $p(\psi)$
2. **토로이달 장 함수**: $F(\psi)$ (또는 동등하게 $F^2(\psi)$)

이 함수들은 다음에 의해 결정됩니다:
- 플라즈마 가열 및 전류 구동
- 경계 조건 (전도 벽, 외부 코일)
- 수송 과정 (MHD를 넘어서)

해석적 해에 대한 일반적인 선택:
- $p(\psi) = p_0(1 - \psi/\psi_0)^\alpha$
- $F^2(\psi) = F_0^2 + \beta(\psi - \psi_0)$

### 5.4 Solovev 평형 (해석적 해)

자유 함수에 대해:

$$
p(\psi) = 0, \quad F^2(\psi) = F_0^2 + c\psi
$$

Grad-Shafranov 방정식은 선형이 됩니다:

$$
\Delta^*\psi = -\mu_0 c R^2
$$

**Solovev 해** (원형 단면에 대해):

$$
\psi(R,Z) = \frac{1}{8}c\mu_0\left[(R^2 - R_0^2)^2 + Z^2\right] + \psi_0
$$

이것은 Shafranov 이동만큼 바깥쪽으로 이동한 원형 플럭스 표면을 나타냅니다.

### 5.5 수치 해법

일반적인 $p(\psi)$와 $F(\psi)$에 대해, Grad-Shafranov 방정식은 수치적으로 풀어야 합니다:

**반복 스킴**:
1. 초기 $\psi^{(0)}(R,Z)$ 추정
2. $p(\psi^{(n)})$과 $F(\psi^{(n)})$ 계산
3. 선형 타원 방정식 풀이:
   $$
   \Delta^*\psi^{(n+1)} = -\mu_0 R^2\frac{dp}{d\psi}\Big|_{\psi^{(n)}} - F\frac{dF}{d\psi}\Big|_{\psi^{(n)}}
   $$
4. 수렴할 때까지 반복: $|\psi^{(n+1)} - \psi^{(n)}| < \epsilon$

**이산화**: $(R,Z)$ 그리드에서 유한 차분 또는 유한 요소 방법.

## 6. 안전 인자

### 6.1 정의

**안전 인자** $q(\psi)$는 플럭스 표면에서 자기장선의 피치를 측정합니다:

$$
q = \frac{1}{2\pi}\oint\frac{d\ell_\parallel}{Rd\phi}
$$

여기서 적분은 한 폴로이달 회전에 대한 것입니다.

**물리적 해석**:
- $q$는 자기장선이 폴로이달 회전당 만드는 토로이달 회전 수
- 유리수 값 ($q = m/n$)은 섭동이 공명할 수 있는 **공명 표면**을 정의

### 6.2 원통 근사

큰 종횡비 토카막 ($R \approx R_0 + r\cos\theta$)에 대해:

$$
q(r) \approx \frac{rB_z}{R_0 B_θ}
$$

$B_θ = \mu_0 I(r)/(2\pi r)$ 사용:

$$
q(r) = \frac{2\pi r^2 B_z}{\mu_0 R_0 I(r)}
$$

### 6.3 전류 밀도와의 관계

미분:

$$
\frac{dq}{dr} = \frac{2\pi r B_z}{\mu_0 R_0}\left(\frac{2}{I} - \frac{r J_z}{I}\right)
$$

**자기 전단**:

$$
s = \frac{r}{q}\frac{dq}{dr}
$$

양의 전단 ($dq/dr > 0$)은 일반적으로 안정화입니다.

### 6.4 안정성에 대한 의미

- **Kruskal-Shafranov 기준**: 외부 kink 모드는 $q(a) > m/n$ 필요
- $m=1, n=1$에 대해: $q(a) > 1$이 필요 (충분하지 않음)
- **내부 kink** (sawtooth 진동): $q(0) < 1$
- **유리수 표면** $q = m/n$: tearing 모드의 위치

```
토카막의 일반적인 q-프로파일:
============================

q |     ___________________  q(a) > 2
  |    /
  |   /                     ← 단조 (양의 전단)
  |  /
  | /                       q(0) ~ 1
  |/________________________ r
  0                        a

특징:
- q(0) ~ 1 (축 상)
- q(a) > 2-3 (가장자리, 안정성 위해)
- 역전단: 중심부에서 dq/dr < 0 (고급 시나리오)
```

## 7. 플럭스 표면과 Shafranov 이동

### 7.1 중첩 플럭스 표면

잘 갇힌 플라즈마에서 플럭스 표면은 중첩된 위상학적 토러스입니다. 자기 축 (가장 안쪽 표면)은 다음을 가진 표면입니다:
- 최고 압력
- $q(\psi_{axis})$ 최소
- $|\nabla\psi| = 0$

### 7.2 Shafranov 이동

토로이달 효과로 인해, 자기 축은 기하학적 중심에 대해 **바깥쪽** (더 큰 $R$)으로 이동합니다.

**물리적 기원**:
- 안쪽에서 더 높은 장 → 더 높은 자기 압력
- 플라즈마 압력 구배가 균형을 위한 바깥쪽 이동 생성

**근사 공식**:

$$
\Delta_{Shafranov} \approx \frac{a^2\beta_p}{2R_0}
$$

여기서 $\beta_p = 2\mu_0\langle p\rangle/B_p^2$는 폴로이달 베타입니다.

### 7.3 플럭스 표면 성형

현대 토카막은 비원형 단면을 사용합니다:
- **늘림** $\kappa = b/a$ (수직 늘림): 안정성과 가둠 개선
- **삼각형성** $\delta$: 안쪽 들여쓰기, ballooning 모드 안정화

## 8. 플라즈마 베타

### 8.1 정의

**플라즈마 베타**는 플라즈마 압력 대 자기 압력의 비율입니다:

$$
\beta = \frac{2\mu_0 p}{B^2}
$$

**체적 평균 베타**:

$$
\langle\beta\rangle = \frac{2\mu_0\langle p\rangle}{\langle B^2\rangle}
$$

**폴로이달 베타**:

$$
\beta_p = \frac{2\mu_0\langle p\rangle}{B_p^2}
$$

**토로이달 베타**:

$$
\beta_t = \frac{2\mu_0\langle p\rangle}{B_t^2}
$$

### 8.2 베타들 사이의 관계

큰 종횡비 토카막에 대해:

$$
\beta_t \approx \beta_p\left(\frac{B_p}{B_t}\right)^2 \approx \beta_p\left(\frac{a}{R_0}\right)^2\frac{1}{q^2}
$$

### 8.3 베타 한계

높은 베타는 핵융합 로에 바람직하지만 (더 많은 핵융합 출력), MHD 안정성이 $\beta$를 제한합니다:

**Troyon 한계** (경험적 스케일링):

$$
\beta_N \equiv \frac{\beta_t(\%)}{I_p/(aB_t)} \lesssim 2.8 - 3.5
$$

여기서 $I_p$는 플라즈마 전류 (MA), $a$는 미터, $B_t$는 Tesla입니다.

**물리적 기원**:
- 높은 $\beta$ → 강한 압력 구배 → 압력 구동 불안정성
- 외부 kink 모드가 고정된 $q(a)$에서 $\beta$ 제한

### 8.4 베타 최적화

베타 한계를 높이는 전략:
- 높은 늘림 $\kappa$
- 높은 삼각형성 $\delta$
- 전도 벽 안정화
- 플라즈마 회전
- 고급 토카막 시나리오 (역전단, 수송 장벽)

## 9. Python 구현: Grad-Shafranov 솔버

### 9.1 간단한 Solovev 평형

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Solovev equilibrium solver
class SolovevEquilibrium:
    def __init__(self, R0, a, kappa, delta, Bt0, Ip):
        """
        R0: major radius [m]
        a: minor radius [m]
        kappa: elongation
        delta: triangularity
        Bt0: toroidal field on axis [T]
        Ip: plasma current [A]
        """
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.Bt0 = Bt0
        self.Ip = Ip

        # Physical constants
        self.mu0 = 4*np.pi*1e-7

    def compute_psi(self, R, Z):
        """Compute poloidal flux function (Solovev solution)"""
        # Simplified Solovev: circular, low beta
        c = -2 * self.mu0 * self.Ip / (np.pi * self.a**2)

        # Normalized coordinates
        r_norm = np.sqrt((R - self.R0)**2 + (Z/self.kappa)**2) / self.a

        # Flux function (normalized)
        psi = -c * self.R0**2 * self.a**2 * r_norm**2 / 8

        return psi

    def compute_B(self, R, Z):
        """Compute magnetic field components"""
        # Numerical derivatives
        dR = 0.001
        dZ = 0.001

        dpsi_dR = (self.compute_psi(R+dR, Z) - self.compute_psi(R-dR, Z)) / (2*dR)
        dpsi_dZ = (self.compute_psi(R, Z+dZ) - self.compute_psi(R, Z-dZ)) / (2*dZ)

        BR = -1/R * dpsi_dZ
        BZ = 1/R * dpsi_dR
        Bphi = self.Bt0 * self.R0 / R

        return BR, BZ, Bphi

    def compute_q(self, psi_vals):
        """Compute safety factor profile"""
        # Simplified q-profile for circular equilibrium
        psi_edge = self.compute_psi(self.R0 + self.a, 0)
        psi_norm = psi_vals / psi_edge

        # Parabolic q-profile
        q0 = 1.0  # On-axis q
        qa = 3.0  # Edge q

        q = q0 + (qa - q0) * psi_norm**2

        return q

    def plot_flux_surfaces(self, nr=50, nz=50):
        """Plot flux surfaces"""
        R_grid = np.linspace(self.R0 - 1.2*self.a, self.R0 + 1.2*self.a, nr)
        Z_grid = np.linspace(-1.2*self.kappa*self.a, 1.2*self.kappa*self.a, nz)

        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)
        psi_mesh = self.compute_psi(R_mesh, Z_mesh)

        fig, ax = plt.subplots(figsize=(8, 10))

        # Contour plot of flux surfaces
        levels = 20
        CS = ax.contour(R_mesh, Z_mesh, psi_mesh, levels=levels, colors='blue')
        ax.clabel(CS, inline=True, fontsize=8)

        # Mark magnetic axis
        ax.plot(self.R0, 0, 'r*', markersize=15, label='Magnetic axis')

        # Mark last closed flux surface
        theta = np.linspace(0, 2*np.pi, 100)
        R_lcfs = self.R0 + self.a*np.cos(theta + self.delta*np.sin(theta))
        Z_lcfs = self.kappa*self.a*np.sin(theta)
        ax.plot(R_lcfs, Z_lcfs, 'r--', linewidth=2, label='LCFS')

        ax.set_xlabel('R [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.set_title('Flux Surfaces (Solovev Equilibrium)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_q_profile(self):
        """Plot safety factor profile"""
        # Radial coordinate (minor radius)
        r = np.linspace(0, self.a, 100)
        R_vals = self.R0 + r
        Z_vals = np.zeros_like(r)

        # Compute psi along midplane
        psi_vals = np.array([self.compute_psi(R, 0) for R in R_vals])

        # Compute q
        q_vals = self.compute_q(psi_vals)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(r/self.a, q_vals, 'b-', linewidth=2)
        ax.axhline(y=1, color='r', linestyle='--', label='q = 1 (sawtooth)')
        ax.axhline(y=2, color='g', linestyle='--', label='q = 2 (m=2 resonance)')

        ax.set_xlabel('r/a (normalized radius)', fontsize=12)
        ax.set_ylabel('q (safety factor)', fontsize=12)
        ax.set_title('Safety Factor Profile', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_pressure_profile(self):
        """Plot pressure and current density profiles"""
        r = np.linspace(0, self.a, 100)

        # Parabolic pressure profile
        p0 = 1e5  # Central pressure [Pa]
        p = p0 * (1 - (r/self.a)**2)**2

        # Current density (from force balance)
        # J_phi ~ dp/dr (simplified)
        dp_dr = np.gradient(p, r)
        j_phi = -dp_dr / (self.Bt0 * self.R0) * 1e-6  # Normalized

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Pressure
        ax1.plot(r/self.a, p/1e3, 'b-', linewidth=2)
        ax1.set_xlabel('r/a', fontsize=12)
        ax1.set_ylabel('Pressure [kPa]', fontsize=12)
        ax1.set_title('Pressure Profile', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Current density
        ax2.plot(r/self.a, j_phi, 'r-', linewidth=2)
        ax2.set_xlabel('r/a', fontsize=12)
        ax2.set_ylabel('Current Density (normalized)', fontsize=12)
        ax2.set_title('Toroidal Current Density Profile', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
def example_tokamak_equilibrium():
    """ITER-like parameters"""
    R0 = 6.2    # Major radius [m]
    a = 2.0     # Minor radius [m]
    kappa = 1.7 # Elongation
    delta = 0.33# Triangularity
    Bt0 = 5.3   # Toroidal field [T]
    Ip = 15e6   # Plasma current [A]

    eq = SolovevEquilibrium(R0, a, kappa, delta, Bt0, Ip)

    print("=== ITER-like Tokamak Equilibrium ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Aspect ratio A = {R0/a:.2f}")
    print(f"Elongation κ = {kappa}")
    print(f"Triangularity δ = {delta}")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {Ip/1e6:.1f} MA")

    # Compute some equilibrium quantities
    psi_axis = eq.compute_psi(R0, 0)
    psi_edge = eq.compute_psi(R0 + a, 0)

    print(f"\nFlux at axis: {psi_axis:.3e} Wb")
    print(f"Flux at edge: {psi_edge:.3e} Wb")

    # Safety factor
    q_axis = eq.compute_q(np.array([psi_axis]))[0]
    q_edge = eq.compute_q(np.array([psi_edge]))[0]

    print(f"\nSafety factor q(0) = {q_axis:.2f}")
    print(f"Safety factor q(a) = {q_edge:.2f}")

    # Plot results
    fig1 = eq.plot_flux_surfaces()
    plt.savefig('/tmp/flux_surfaces.png', dpi=150)
    print("\nFlux surfaces plot saved to /tmp/flux_surfaces.png")

    fig2 = eq.plot_q_profile()
    plt.savefig('/tmp/q_profile.png', dpi=150)
    print("q-profile plot saved to /tmp/q_profile.png")

    fig3 = eq.plot_pressure_profile()
    plt.savefig('/tmp/pressure_profile.png', dpi=150)
    print("Pressure profile plot saved to /tmp/pressure_profile.png")

    plt.close('all')

if __name__ == "__main__":
    example_tokamak_equilibrium()
```

### 9.2 수치적 Grad-Shafranov 솔버

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

class GradShafranovSolver:
    """Numerical solver for Grad-Shafranov equation"""

    def __init__(self, R_min, R_max, Z_min, Z_max, nR, nZ):
        """
        Set up computational domain
        """
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.nR = nR
        self.nZ = nZ

        # Grid spacing
        self.dR = (R_max - R_min) / (nR - 1)
        self.dZ = (Z_max - Z_min) / (nZ - 1)

        # Create grid
        self.R = np.linspace(R_min, R_max, nR)
        self.Z = np.linspace(Z_min, Z_max, nZ)
        self.R_grid, self.Z_grid = np.meshgrid(self.R, self.Z)

        # Initialize flux
        self.psi = np.zeros((nZ, nR))

        # Constants
        self.mu0 = 4*np.pi*1e-7

    def set_free_functions(self, p_func, F_func):
        """
        Set pressure and toroidal field functions
        p_func: p(psi) callable
        F_func: F(psi) callable where F = R*B_phi
        """
        self.p_func = p_func
        self.F_func = F_func

    def build_operator_matrix(self):
        """
        Build discrete Grad-Shafranov operator matrix
        Δ* ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²
        """
        nR = self.nR
        nZ = self.nZ
        N = nR * nZ

        # Flatten index: (i,j) -> i*nR + j
        def idx(i, j):
            return i * nR + j

        # Build sparse matrix
        row = []
        col = []
        data = []

        for i in range(nZ):
            for j in range(nR):
                n = idx(i, j)
                R = self.R[j]

                # Interior points
                if 0 < i < nZ-1 and 0 < j < nR-1:
                    # R derivatives: R ∂/∂R(1/R ∂ψ/∂R)
                    # = ∂²ψ/∂R² + (1/R)∂ψ/∂R - (ψ/R²)
                    # Discretize: centered differences

                    # ∂²ψ/∂R²
                    coef_R_pp = 1 / self.dR**2
                    coef_R_0 = -2 / self.dR**2
                    coef_R_mm = 1 / self.dR**2

                    # (1/R)∂ψ/∂R
                    coef_R_p_1st = 1 / (2*R*self.dR)
                    coef_R_m_1st = -1 / (2*R*self.dR)

                    # Z derivatives: ∂²ψ/∂Z²
                    coef_Z_pp = 1 / self.dZ**2
                    coef_Z_0 = -2 / self.dZ**2
                    coef_Z_mm = 1 / self.dZ**2

                    # Combine
                    # Center
                    row.append(n)
                    col.append(n)
                    data.append(coef_R_0 + coef_Z_0)

                    # R+1
                    row.append(n)
                    col.append(idx(i, j+1))
                    data.append(coef_R_pp + coef_R_p_1st)

                    # R-1
                    row.append(n)
                    col.append(idx(i, j-1))
                    data.append(coef_R_mm + coef_R_m_1st)

                    # Z+1
                    row.append(n)
                    col.append(idx(i+1, j))
                    data.append(coef_Z_pp)

                    # Z-1
                    row.append(n)
                    col.append(idx(i-1, j))
                    data.append(coef_Z_mm)

                else:
                    # Boundary: ψ = 0
                    row.append(n)
                    col.append(n)
                    data.append(1.0)

        matrix = csr_matrix((data, (row, col)), shape=(N, N))
        return matrix

    def solve_fixed_boundary(self, psi_boundary=0, max_iter=100, tol=1e-6):
        """
        Solve Grad-Shafranov with fixed boundary using Picard iteration
        """
        nR = self.nR
        nZ = self.nZ
        N = nR * nZ

        # Build operator matrix (constant for linear problem)
        A = self.build_operator_matrix()

        # Picard iteration
        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            # Compute RHS: -μ₀ R² dp/dψ - F dF/dψ
            rhs = np.zeros((nZ, nR))

            for i in range(nZ):
                for j in range(nR):
                    R = self.R[j]
                    psi_val = self.psi[i, j]

                    # Numerical derivatives of p and F
                    dpsi = 1e-6
                    dpdpsi = (self.p_func(psi_val + dpsi) - self.p_func(psi_val - dpsi)) / (2*dpsi)

                    F_val = self.F_func(psi_val)
                    dFdpsi = (self.F_func(psi_val + dpsi) - self.F_func(psi_val - dpsi)) / (2*dpsi)

                    rhs[i, j] = -self.mu0 * R**2 * dpdpsi - F_val * dFdpsi

            # Apply boundary conditions
            rhs[0, :] = psi_boundary
            rhs[-1, :] = psi_boundary
            rhs[:, 0] = psi_boundary
            rhs[:, -1] = psi_boundary

            # Solve linear system
            rhs_flat = rhs.flatten()
            psi_flat = spsolve(A, rhs_flat)
            self.psi = psi_flat.reshape((nZ, nR))

            # Check convergence
            error = np.max(np.abs(self.psi - psi_old))

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: max error = {error:.3e}")

            if error < tol:
                print(f"Converged in {iteration+1} iterations")
                break

        return self.psi

    def plot_solution(self):
        """Plot the computed equilibrium"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Flux surfaces
        ax = axes[0]
        levels = 30
        CS = ax.contour(self.R_grid, self.Z_grid, self.psi, levels=levels, colors='blue')
        ax.clabel(CS, inline=True, fontsize=8)
        ax.set_xlabel('R [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.set_title('Poloidal Flux Surfaces', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Pressure profile
        ax = axes[1]
        psi_flat = self.psi.flatten()
        psi_min = np.min(psi_flat)
        psi_max = np.max(psi_flat)
        psi_range = np.linspace(psi_min, psi_max, 100)

        p_range = np.array([self.p_func(psi) for psi in psi_range])

        ax.plot(psi_range, p_range/1e3, 'r-', linewidth=2)
        ax.set_xlabel('ψ [Wb]', fontsize=12)
        ax.set_ylabel('Pressure [kPa]', fontsize=12)
        ax.set_title('Pressure Profile p(ψ)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example: solve simple equilibrium
def example_gs_solver():
    """Example Grad-Shafranov solution"""

    # Domain: tokamak-like geometry
    R0 = 1.0  # Major radius
    a = 0.3   # Minor radius

    R_min = R0 - 1.5*a
    R_max = R0 + 1.5*a
    Z_min = -1.5*a
    Z_max = 1.5*a

    nR = 80
    nZ = 80

    solver = GradShafranovSolver(R_min, R_max, Z_min, Z_max, nR, nZ)

    # Define free functions
    # Simple parabolic pressure
    p0 = 1e5  # 100 kPa
    psi0 = -0.1

    def p_func(psi):
        if psi > psi0:
            return 0.0
        else:
            return p0 * (1 - psi/psi0)**2

    # Constant F (uniform toroidal field)
    Bt0 = 2.0  # Tesla
    F0 = Bt0 * R0

    def F_func(psi):
        return F0

    solver.set_free_functions(p_func, F_func)

    print("=== Grad-Shafranov Solver ===")
    print(f"Grid: {nR} x {nZ}")
    print(f"Domain: R ∈ [{R_min}, {R_max}], Z ∈ [{Z_min}, {Z_max}]")
    print(f"Central pressure: {p0/1e3} kPa")
    print(f"Toroidal field: {Bt0} T")

    # Solve
    psi = solver.solve_fixed_boundary(max_iter=100, tol=1e-6)

    # Plot
    fig = solver.plot_solution()
    plt.savefig('/tmp/gs_solution.png', dpi=150)
    print("\nSolution plot saved to /tmp/gs_solution.png")
    plt.close()

if __name__ == "__main__":
    example_gs_solver()
```

## 10. 베타 계산

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_beta(R0, a, p_profile, B_profiles, nr=100):
    """
    Compute various beta values for a tokamak equilibrium

    Parameters:
    -----------
    R0: major radius [m]
    a: minor radius [m]
    p_profile: function p(r) giving pressure profile
    B_profiles: dict with 'Bt', 'Bp' functions of r
    nr: number of radial points

    Returns:
    --------
    dict with beta_p, beta_t, beta_N
    """
    r = np.linspace(0, a, nr)
    dr = r[1] - r[0]

    # Volume element in torus: dV = 2π R₀ · 2πr dr
    def volume_element(r_val):
        return 4 * np.pi**2 * R0 * r_val * dr

    # Compute volume-averaged quantities
    p_vals = np.array([p_profile(r_val) for r_val in r])
    Bt_vals = np.array([B_profiles['Bt'](r_val) for r_val in r])
    Bp_vals = np.array([B_profiles['Bp'](r_val) for r_val in r])

    # Volume integrals
    V_total = np.sum([volume_element(r[i]) for i in range(nr)])

    p_avg = np.sum([p_vals[i] * volume_element(r[i]) for i in range(nr)]) / V_total

    Bt2_avg = np.sum([Bt_vals[i]**2 * volume_element(r[i]) for i in range(nr)]) / V_total
    Bp2_avg = np.sum([Bp_vals[i]**2 * volume_element(r[i]) for i in range(nr)]) / V_total

    mu0 = 4*np.pi*1e-7

    # Poloidal beta
    beta_p = 2 * mu0 * p_avg / Bp2_avg

    # Toroidal beta
    beta_t = 2 * mu0 * p_avg / Bt2_avg

    # For beta_N, need plasma current
    # I_p = ∮ J·dl ~ B_p * circumference / μ₀
    Bp_edge = Bp_vals[-1]
    Ip = 2 * np.pi * a * Bp_edge / mu0

    # Troyon normalized beta
    Bt_axis = Bt_vals[0]
    beta_N = beta_t * 100 / (Ip / (a * Bt_axis))  # percentage

    results = {
        'beta_p': beta_p,
        'beta_t': beta_t,
        'beta_N': beta_N,
        'p_avg': p_avg,
        'Ip': Ip
    }

    return results

def example_beta_calculation():
    """Example beta calculation for tokamak"""

    # ITER-like parameters
    R0 = 6.2
    a = 2.0

    # Parabolic pressure profile
    p0 = 5e5  # 500 kPa
    def p_profile(r):
        return p0 * (1 - (r/a)**2)**2

    # Magnetic field profiles
    Bt0 = 5.3  # On-axis toroidal field
    Ip = 15e6  # Plasma current

    def Bt_profile(r):
        return Bt0 * R0 / (R0 + r)  # 1/R dependence

    def Bp_profile(r):
        mu0 = 4*np.pi*1e-7
        # From Ampere's law, current ~ r² profile
        I_enclosed = Ip * (r/a)**2
        return mu0 * I_enclosed / (2 * np.pi * r) if r > 0 else 0

    B_profiles = {
        'Bt': Bt_profile,
        'Bp': Bp_profile
    }

    # Compute betas
    results = compute_beta(R0, a, p_profile, B_profiles)

    print("=== Beta Calculation ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Central pressure p0 = {p0/1e3} kPa")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {results['Ip']/1e6:.1f} MA")
    print(f"\nAverage pressure <p> = {results['p_avg']/1e3:.1f} kPa")
    print(f"Poloidal beta β_p = {results['beta_p']:.3f}")
    print(f"Toroidal beta β_t = {results['beta_t']*100:.2f} %")
    print(f"Normalized beta β_N = {results['beta_N']:.2f}")

    # Troyon limit check
    beta_N_limit = 3.5
    print(f"\nTroyon limit β_N < {beta_N_limit}")
    if results['beta_N'] < beta_N_limit:
        print("✓ Within stability limit")
    else:
        print("✗ Exceeds stability limit!")

    # Plot profiles
    r = np.linspace(0.01, a, 100)
    p = np.array([p_profile(ri) for ri in r])
    Bt = np.array([Bt_profile(ri) for ri in r])
    Bp = np.array([Bp_profile(ri) for ri in r])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pressure
    axes[0,0].plot(r/a, p/1e3, 'b-', linewidth=2)
    axes[0,0].set_xlabel('r/a')
    axes[0,0].set_ylabel('Pressure [kPa]')
    axes[0,0].set_title('Pressure Profile')
    axes[0,0].grid(True, alpha=0.3)

    # Toroidal field
    axes[0,1].plot(r/a, Bt, 'g-', linewidth=2)
    axes[0,1].set_xlabel('r/a')
    axes[0,1].set_ylabel('B_t [T]')
    axes[0,1].set_title('Toroidal Field')
    axes[0,1].grid(True, alpha=0.3)

    # Poloidal field
    axes[1,0].plot(r/a, Bp, 'r-', linewidth=2)
    axes[1,0].set_xlabel('r/a')
    axes[1,0].set_ylabel('B_p [T]')
    axes[1,0].set_title('Poloidal Field')
    axes[1,0].grid(True, alpha=0.3)

    # Local beta
    beta_local = 2 * (4*np.pi*1e-7) * p / (Bt**2 + Bp**2)
    axes[1,1].plot(r/a, beta_local*100, 'm-', linewidth=2)
    axes[1,1].set_xlabel('r/a')
    axes[1,1].set_ylabel('β [%]')
    axes[1,1].set_title('Local Beta Profile')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/beta_profiles.png', dpi=150)
    print("\nProfiles plot saved to /tmp/beta_profiles.png")
    plt.close()

if __name__ == "__main__":
    example_beta_calculation()
```

## 요약

이 강의에서 MHD 평형의 기초를 다루었습니다:

1. **힘 균형**: 기본 방정식 $\nabla p = \mathbf{J}\times\mathbf{B}$가 플라즈마 압력 구배와 자기력 (압력 + 장력)의 균형을 맞춥니다.

2. **결과**: 압력과 전류는 자기 플럭스 표면 위에 놓이며, 플럭스 표면 좌표계를 가능하게 합니다.

3. **1차원 평형**: θ-pinch (순수 축방향 장), Z-pinch (자체 생성 방위각 장, Bennett 관계), screw pinch (전단을 가진 결합 장).

4. **Grad-Shafranov 방정식**: 두 개의 자유 함수 $p(\psi)$와 $F(\psi)$ 지정이 필요한 축대칭 토로이달 평형을 지배하는 타원형 PDE.

5. **안전 인자**: 자기장선 피치를 측정하는 매개변수 $q$, 안정성 분석에 중요 (Kruskal-Shafranov 한계, 유리수 표면).

6. **플럭스 표면**: 압력이 일정한 중첩된 토로이달 표면, 토로이달 효과로 인한 Shafranov 이동.

7. **플라즈마 베타**: 플라즈마 대 자기 압력의 비율, MHD 안정성이 설정한 운영 한계 (Troyon 한계).

8. **수치적 방법**: 유한 차분과 반복 기법을 사용한 평형 솔버 구현.

이러한 평형 개념은 플라즈마 안정성 (다음 강의), 수송, 그리고 핵융합 장치의 가둠을 이해하는 기초를 형성합니다.

## 연습 문제

### 문제 1: 원통형 플라즈마에서의 힘 균형

원통형 플라즈마 기둥이 다음 프로파일을 가집니다:
- 압력: $p(r) = p_0(1 - r^2/a^2)$ for $r < a$, $p=0$ for $r \geq a$
- 축방향 장: $B_z = B_0 = \text{const}$
- 방위각 장: $B_θ(r)$ 결정 필요

**(a)** 방사상 힘 균형 방정식을 사용하여 $B_θ(r)$을 유도하세요.

**(b)** 총 플라즈마 전류 $I_p$를 계산하세요.

**(c)** 주요 반지름 $R_0 = 5a$를 가정하고 안전 인자 $q(r)$을 계산하세요.

**(d)** 자기 축 ($r=0$)에서 $q$는 무엇입니까?

**힌트**: 원통 좌표계에서 $\nabla p = \mathbf{J}\times\mathbf{B}$를 사용하세요.

### 문제 2: Z-Pinch에 대한 Bennett 관계

Z-pinch가 균일한 밀도 $n = 10^{20}$ m$^{-3}$, 온도 $T = 10$ keV, 길이 $L = 1$ m, 반지름 $a = 1$ cm를 가집니다.

**(a)** 총 입자 수 $N = n \pi a^2 L$을 계산하세요.

**(b)** Bennett 관계 $I^2 = (8\pi/\mu_0)NkT$를 사용하여 필요한 전류 $I_p$를 계산하세요.

**(c)** 표면에서 자기장 $B_θ(a) = \mu_0 I_p/(2\pi a)$를 추정하세요.

**(d)** 자기 압력 $B_θ^2/(2\mu_0)$를 계산하고 플라즈마 압력 $p = nkT$와 비교하세요.

**(e)** 이 구성의 안정성 함의를 논의하세요.

### 문제 3: 일정 압력을 가진 Grad-Shafranov

다음을 가진 Grad-Shafranov 방정식을 고려하세요:
- $p(\psi) = p_0 = \text{const}$
- $F(\psi) = F_0 = \text{const}$

**(a)** 방정식이 다음으로 축약됨을 보이세요:
$$
\Delta^*\psi = -\mu_0 p_0 R^2
$$

**(b)** 큰 종횡비 원형 토카막 ($R \approx R_0$)에 대해 다음과 같이 근사하세요:
$$
\frac{1}{r}\frac{d}{dr}\left(r\frac{d\psi}{dr}\right) + \frac{d^2\psi}{dz^2} \approx -\mu_0 p_0 R_0^2
$$

**(c)** 분리 가능한 해 $\psi(r,z) = R_r(r)Z_z(z)$를 제안하고 $R_r$과 $Z_z$에 대한 ODE를 유도하세요.

**(d)** 원형 플럭스 표면 $\psi \propto r^2 + \kappa^2 z^2$에 대해 풀고 $\kappa$ (늘림)를 결정하세요.

### 문제 4: 안전 인자와 전류 프로파일

토카막이 주요 반지름 $R_0 = 3$ m, 소반지름 $a = 1$ m, 토로이달 장 $B_t = 5$ T (거의 일정)를 가집니다. 전류 밀도 프로파일은:

$$
J_z(r) = J_0\left(1 - \frac{r^2}{a^2}\right)
$$

**(a)** 둘러싸인 전류 $I(r) = \int_0^r J_z(r') 2\pi r' dr'$를 계산하세요.

**(b)** 앙페르 법칙을 사용하여 $B_θ(r) = \mu_0 I(r)/(2\pi r)$를 구하세요.

**(c)** 안전 인자 프로파일 $q(r) = rB_t/(R_0 B_θ(r))$를 계산하세요.

**(d)** $q(0)$ (축 상)과 $q(a)$ (가장자리)를 결정하세요.

**(e)** $q(r_s) = 2$ ($m=2$ 유리수 표면)인 반지름 $r_s$를 구하세요.

**(f)** $q(r)$을 플롯하고 안정성 관련 특징을 식별하세요.

### 문제 5: 베타 한계

실험 토카막이 다음으로 작동합니다:
- 소반지름 $a = 0.5$ m
- 토로이달 장 $B_t = 3$ T
- 플라즈마 전류 $I_p = 1$ MA
- 중심 압력 $p_0 = 10^5$ Pa
- 압력 프로파일 $p(r) = p_0(1 - r^2/a^2)^2$

**(a)** 체적 평균 압력 $\langle p\rangle$을 계산하세요.

**(b)** 토로이달 베타 $\beta_t = 2\mu_0\langle p\rangle/B_t^2$를 계산하세요.

**(c)** 정규화된 베타 $\beta_N = \beta_t(\%)/(I_p[MA]/(a[m]B_t[T]))$를 계산하세요.

**(d)** $\beta_N$을 Troyon 한계 $\beta_N < 3.5$와 비교하세요.

**(e)** 실험이 압력을 두 배로 하려면 안정성을 유지하기 위해 $I_p$ 또는 $B_t$에 어떤 조정이 필요합니까?

**힌트**: 원통에서 체적 평균: $\langle p\rangle = \frac{\int_0^a p(r) 2\pi r dr}{\pi a^2}$.

---

**이전**: [Overview](./00_Overview.md) | **다음**: [Linear Stability](./02_Linear_Stability.md)

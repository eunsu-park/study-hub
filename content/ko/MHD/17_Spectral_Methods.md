# 17. 스펙트럴 및 고급 방법

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

- MHD를 위한 pseudo-spectral 방법의 원리 이해하기
- 주기적 MHD 문제를 위한 Fourier 기반 스펙트럴 솔버 구현하기
- 비선형 항을 올바르게 처리하기 위한 dealiasing 기법 적용하기
- 비주기 경계값 문제를 위한 Chebyshev 스펙트럴 방법 사용하기
- MHD를 위한 Adaptive Mesh Refinement (AMR) 이해하기
- 운동론적 효과를 위한 hybrid MHD-PIC 방법 설명하기
- SPH-MHD를 그리드 기반 방법과 비교하기
- 생산 MHD 코드 조사하기: Athena++, PLUTO, FLASH, Pencil Code, Dedalus
- Python으로 간단한 2D pseudo-spectral MHD 솔버 구현하기

---

## 1. 스펙트럴 방법 소개

### 1.1 왜 스펙트럴 방법인가?

유한 차분 및 유한 체적 방법은 제공합니다:
- **정확도:** $O(\Delta x^2)$에서 $O(\Delta x^5)$ (WENO)
- **유연성:** 복잡한 기하학, AMR, 충격파

스펙트럴 방법은 제공합니다:
- **정확도:** 매끄러운 해에 대해 지수적 수렴
- **효율성:** FFT를 통한 정확한 도함수 ($O(N \log N)$)
- **단순성:** 비압축성 난류, dynamo에 자연스러움

**절충:**
| 방법 | 수렴 | 충격파 | BC | 기하학 |
|--------|-------------|--------|-----|------------|
| Finite Volume | 대수적 | 우수 | 유연 | 임의 |
| Spectral (Fourier) | 지수적 | 불량 (Gibbs) | 주기만 | 단순 |
| Spectral (Chebyshev) | 지수적 | 불량 | 유연 | 1D/2D slab |

**최적 사용 사례:**
- MHD 난류 (주기 박스)
- Dynamo 시뮬레이션 (높은 Reynolds 수)
- 선형 안정성 분석 (고유 모드)

### 1.2 Fourier vs Chebyshev 기저

**Fourier:**
$$
f(x) = \sum_{k=-N/2}^{N/2-1} \hat{f}_k e^{i k x}
$$
- 주기 BC: $f(0) = f(2\pi)$
- 도함수: $\partial_x f \leftrightarrow i k \hat{f}_k$
- 직교성: $\int_0^{2\pi} e^{i k x} e^{-i k' x} dx = 2\pi \delta_{kk'}$

**Chebyshev:**
$$
f(x) = \sum_{n=0}^{N} a_n T_n(x), \quad T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]
$$
- 비주기 BC: Dirichlet, Neumann, 혼합
- 도함수: 재귀 관계 (Fourier보다 복잡)
- 클러스터링: Chebyshev 점이 경계 근처에 클러스터 (경계층 해결)

---

## 2. MHD를 위한 Pseudo-Spectral 방법

### 2.1 기본 알고리즘

주기 박스에서 비압축성 MHD 고려:
$$
\frac{\partial \mathbf{v}}{\partial t} = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B} - \nabla p + \nu \nabla^2 \mathbf{v}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
$$
$$
\nabla \cdot \mathbf{v} = 0, \quad \nabla \cdot \mathbf{B} = 0
$$

**Pseudo-spectral 방식:**

1. **초기화:** $\mathbf{v}(\mathbf{x}, 0)$과 $\mathbf{B}(\mathbf{x}, 0)$의 FFT → $\hat{\mathbf{v}}(\mathbf{k}, 0)$, $\hat{\mathbf{B}}(\mathbf{k}, 0)$

2. **시간 루프:**
   - **비선형 항:** **물리 공간**에서 $(\mathbf{v} \cdot \nabla)\mathbf{v}$와 $\mathbf{v} \times \mathbf{B}$ 계산
     - IFFT: $\hat{\mathbf{v}}(\mathbf{k}) \to \mathbf{v}(\mathbf{x})$, $\hat{\mathbf{B}}(\mathbf{k}) \to \mathbf{B}(\mathbf{x})$
     - 계산: $\mathbf{NL}_v = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B}$, $\mathbf{NL}_B = \nabla \times (\mathbf{v} \times \mathbf{B})$
     - FFT: $\mathbf{NL}_v(\mathbf{x}) \to \widehat{\mathbf{NL}}_v(\mathbf{k})$, $\mathbf{NL}_B(\mathbf{x}) \to \widehat{\mathbf{NL}}_B(\mathbf{k})$

   - **선형 항:** **스펙트럴 공간**에서 계산
     - 확산: $\widehat{\nabla^2 \mathbf{v}} = -k^2 \hat{\mathbf{v}}$
     - 압력: $\widehat{\mathbf{NL}}_v$를 발산 없는 부분공간에 투영
       $$
       \hat{\mathbf{v}}_{\perp}(\mathbf{k}) = \hat{\mathbf{v}}(\mathbf{k}) - \frac{\mathbf{k} \cdot \hat{\mathbf{v}}(\mathbf{k})}{k^2} \mathbf{k}
       $$

   - **시간 적분:** RK4 또는 지수 적분기
     $$
     \frac{d \hat{\mathbf{v}}}{dt} = \widehat{\mathbf{NL}}_v - \nu k^2 \hat{\mathbf{v}}
     $$

3. **발산 청소:** 각 단계에서 $\nabla \cdot \mathbf{B} = 0$ 강제:
   $$
   \hat{\mathbf{B}}_{\perp}(\mathbf{k}) = \hat{\mathbf{B}}(\mathbf{k}) - \frac{\mathbf{k} \cdot \hat{\mathbf{B}}(\mathbf{k})}{k^2} \mathbf{k}
   $$

### 2.2 Dealiasing (2/3 규칙)

**문제:** 비선형 항으로부터의 aliasing 오류.

두 함수의 곱:
$$
f(x) g(x) \quad \text{최대 파수 } k_{\max}
$$
은 $2 k_{\max}$까지 Fourier 모드를 가집니다 (convolution 정리).

그리드가 $N$ 점을 가지면, Nyquist 파수 $k_N = N/2$. 모드 $k > k_N$은 더 낮은 $k$로 **alias**됩니다.

**해결책:** 2/3 dealiasing (Orszag 1971)
- 모드 $|k| \leq k_{\max} = 2N/3$만 유지
- IFFT 전에 모드 $|k| > 2N/3$ 0으로 만들기
- 비선형 항 계산 후, 다시 높은 $k$ 모드 0으로 만들기

**비용:** 모드의 1/3을 잃지만, 이차 비선형성의 **정확한** 표현.

### 2.3 시간 적분

**명시적 RK4:**
$$
\frac{d \hat{u}}{dt} = L \hat{u} + N(\hat{u})
$$
여기서 $L = -\nu k^2$ (선형), $N$ (비선형).

표준 RK4:
$$
\begin{aligned}
k_1 &= L \hat{u}^n + N(\hat{u}^n) \\
k_2 &= L(\hat{u}^n + \frac{\Delta t}{2} k_1) + N(\hat{u}^n + \frac{\Delta t}{2} k_1) \\
k_3 &= L(\hat{u}^n + \frac{\Delta t}{2} k_2) + N(\hat{u}^n + \frac{\Delta t}{2} k_2) \\
k_4 &= L(\hat{u}^n + \Delta t k_3) + N(\hat{u}^n + \Delta t k_3) \\
\hat{u}^{n+1} &= \hat{u}^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

**지수 적분기 (ETDRK4):**

stiff 선형 항에 대해 (높은 $k$에서 $\nu k^2$ 큼):
$$
\hat{u}^{n+1} = e^{L \Delta t} \hat{u}^n + \Delta t \phi_1(L \Delta t) N(\hat{u}^n) + \ldots
$$
여기서 $\phi_1(z) = (e^z - 1)/z$.

**장점:** 확산에 대해 무조건 안정 (더 큰 $\Delta t$ 허용).

---

## 3. 비압축성 MHD 난류

### 3.1 지배 방정식 (무차원)

$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\nabla p + (\mathbf{B} \cdot \nabla)\mathbf{B} + \frac{1}{Re} \nabla^2 \mathbf{v} + \mathbf{f}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \frac{1}{Rm} \nabla^2 \mathbf{B}
$$
$$
\nabla \cdot \mathbf{v} = 0, \quad \nabla \cdot \mathbf{B} = 0
$$

매개변수:
- $Re = U L / \nu$ (Reynolds 수)
- $Rm = U L / \eta$ (자기 Reynolds 수)
- $\mathbf{f}$ = 대규모 강제 (난류 유지)

### 3.2 에너지 스펙트럼

**운동 에너지 스펙트럼:**
$$
E_K(k) = \frac{1}{2} \sum_{|\mathbf{k}'| \in [k, k+dk]} |\hat{\mathbf{v}}(\mathbf{k}')|^2
$$

**자기 에너지 스펙트럼:**
$$
E_M(k) = \frac{1}{2} \sum_{|\mathbf{k}'| \in [k, k+dk]} |\hat{\mathbf{B}}(\mathbf{k}')|^2
$$

**총 에너지:**
$$
E_{\text{total}} = \int E_K(k) dk + \int E_M(k) dk
$$

**Kolmogorov 스펙트럼 (유체역학 난류):**
$$
E(k) \propto k^{-5/3}
$$

**Iroshnikov-Kraichnan 스펙트럼 (MHD 난류):**
$$
E(k) \propto k^{-3/2}
$$
(논쟁 중; 현대 DNS는 강한 $B_0$에 대해 $k^{-5/3}$, 약한 $B_0$에 대해 $k^{-3/2}$ 보임)

### 3.3 불변량

**이상 MHD는 보존합니다:**
- 총 에너지: $E = \int (\mathbf{v}^2 + \mathbf{B}^2)/2 \, d^3x$
- 자기 헬리시티: $H_M = \int \mathbf{A} \cdot \mathbf{B} \, d^3x$ ($\eta = 0$일 때)
- 교차 헬리시티: $H_C = \int \mathbf{v} \cdot \mathbf{B} \, d^3x$

**스펙트럴 방법:** 에너지 보존 검증을 위해 이들을 모니터 (수치 소산 확인).

---

## 4. Chebyshev 스펙트럴 방법

### 4.1 Chebyshev 다항식

**정의:**
$$
T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]
$$

**재귀:**
$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
$$

**직교성:**
$$
\int_{-1}^{1} \frac{T_m(x) T_n(x)}{\sqrt{1-x^2}} dx = \begin{cases} 0 & m \neq n \\ \pi/2 & m = n \neq 0 \\ \pi & n = 0 \end{cases}
$$

**Collocation 점 (Gauss-Lobatto):**
$$
x_j = \cos\left(\frac{\pi j}{N}\right), \quad j = 0, 1, \ldots, N
$$

### 4.2 미분 행렬

근사:
$$
f(x) \approx \sum_{n=0}^N a_n T_n(x)
$$

**도함수:**
$$
\frac{df}{dx}(x_i) = \sum_{j=0}^N D_{ij} f(x_j)
$$

여기서 $D_{ij}$는 Chebyshev 미분 행렬 (한 번 계산, 밀집 행렬).

**2차 도함수:**
$$
\frac{d^2 f}{dx^2}(x_i) = \sum_{j=0}^N D^{(2)}_{ij} f(x_j)
$$

### 4.3 응용: Grad-Shafranov 솔버

**축대칭 MHD 평형:**
$$
\nabla^2 \psi = -\mu_0 R^2 \frac{dp}{d\psi} - \frac{1}{2} \frac{dF^2}{d\psi}
$$
여기서 $\psi(R, Z)$는 poloidal flux 함수.

**Chebyshev 이산화:**
- $(R, Z) \in [R_{\min}, R_{\max}] \times [Z_{\min}, Z_{\max}]$를 $[-1, 1]^2$로 매핑
- Chebyshev 그리드에서 $\psi$ 확장
- 미분 행렬을 통해 $\nabla^2 \psi$ 계산
- 비선형 BVP 해결 (Newton 반복)

**경계 조건:**
- Dirichlet: 벽에서 $\psi = 0$
- $D^{(2)}$의 첫/마지막 행 수정

---

## 5. Adaptive Mesh Refinement (AMR)

### 5.1 블록 구조 AMR

**아이디어:** 기울기가 큰 곳 (충격파, 전류 시트) 그리드 정제, 다른 곳 조대화.

**계층:**
```
Level 0: 조대 그리드 (기반)
Level 1: 정제 패치 (2x 해상도)
Level 2: 이중 정제 (4x 해상도)
...
```

**데이터 구조:**
- 중첩 그리드의 트리
- 각 레벨은 영역의 부분집합 커버
- 레벨 간 통신을 위한 ghost 셀

### 5.2 정제 기준

**기울기 기반:**
$$
\text{정제 if } \frac{|\nabla \rho|}{\rho} > \epsilon_{\text{refine}}
$$

**전류 밀도:**
$$
\text{정제 if } |\mathbf{J}| > J_{\text{thresh}}
$$

**Löhner 추정기:**
$$
\mathcal{E} = \frac{\sum_i |\Delta u_i|}{\sum_i |u_i| + \delta}
$$
여기서 $\Delta u_i = u_{i+1} - 2u_i + u_{i-1}$ (2차 차분).

### 5.3 Prolongation과 Restriction

**Prolongation (조대 → 미세):**
- 조대 셀 값을 미세 셀로 보간
- 구간 선형 또는 3차 보간

**Restriction (미세 → 조대):**
- 미세 셀 평균:
$$
U_{\text{coarse}} = \frac{1}{4} \sum_{\text{fine cells}} U_{\text{fine}}
$$
(2D; 3D에서 인수 8)

**보존 restriction:**
$$
U_{\text{coarse}} A_{\text{coarse}} = \sum_{\text{fine}} U_{\text{fine}} A_{\text{fine}}
$$
질량/에너지 보존 보장.

### 5.4 플럭스 보정

**문제:** 미세/조대 경계면에서 플럭스 불일치.

**플럭스 보정 알고리즘:**
1. 미세 레벨에서 플럭스 계산
2. 조대 경계에서 미세 플럭스 합산
3. 조대 플럭스를 합산된 미세 플럭스로 대체
4. 보존 양 보정

**소프트웨어:** Paramesh, CHOMBO, AMReX.

---

## 6. Hybrid MHD-PIC 방법

### 6.1 동기

순수 MHD: 유체 설명, 운동론적 효과 놓침 (Landau 감쇠, 파동-입자 상호작용).

순수 PIC: 완전 운동론적, 대규모 천체물리학 시스템에 대해 계산적으로 매우 비쌈.

**Hybrid 접근법:**
- **벌크 플라즈마:** MHD (이온 + 전자를 단일 유체로)
- **에너지적 입자:** PIC (우주선, SEP, 초열 집단)

### 6.2 결합

**MHD → 입자:**
- 입자가 Lorentz 힘 경험:
$$
\frac{d\mathbf{p}_i}{dt} = q_i (\mathbf{E} + \mathbf{v}_i \times \mathbf{B})
$$
여기서 $\mathbf{E}$, $\mathbf{B}$는 MHD 해로부터.

**입자 → MHD:**
- 입자가 압력/운동량 기여:
$$
p_{\text{CR}} = \sum_i \frac{p_i^2}{3m_i}, \quad \mathbf{f}_{\text{CR}} = -\nabla p_{\text{CR}}
$$
- 운동량 방정식에 $\mathbf{f}_{\text{CR}}$ 추가:
$$
\frac{\partial (\rho \mathbf{v})}{\partial t} + \ldots = \ldots + \mathbf{f}_{\text{CR}}
$$

### 6.3 응용

- **우주선 수송:** 확산, 스트리밍 불안정성
- **태양 에너지 입자:** 충격파에서 가속
- **행성 자기권:** 링 전류 입자

---

## 7. SPH-MHD

### 7.1 Smoothed Particle Hydrodynamics (SPH)

**Lagrangian 방법:**
- 입자로 표현된 유체 (질량 요소)
- 커널에 대해 평활화된 속성:
$$
f(\mathbf{x}) = \sum_j m_j \frac{f_j}{\rho_j} W(|\mathbf{x} - \mathbf{x}_j|, h)
$$
여기서 $W$는 평활화 커널 (예: cubic spline), $h$ = 평활화 길이.

**운동 방정식:**
$$
\frac{d\mathbf{v}_i}{dt} = -\sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla_i W_{ij}
$$

### 7.2 SPH에서 MHD

**도전:** 자기장은 Eulerian (그리드 기반)이지만, SPH는 Lagrangian.

**접근법:**
1. **보존 정식화:** 입자에서 $\mathbf{B}$ 진화, $\nabla \cdot \mathbf{B} = 0$ 청소 사용
2. **벡터 퍼텐셜:** $\mathbf{A}$ 진화, $\mathbf{B} = \nabla \times \mathbf{A}$ 계산 ($\nabla \cdot \mathbf{B} = 0$ 보장)

**인장 불안정성:**
- 자기 압력 기울기 영역에서 입자 클러스터링
- 완화: 인공 점성, 개선된 커널

### 7.3 응용

- **별 형성:** 자화된 분자 구름의 붕괴
- **원시행성 디스크:** MRI, 디스크-행성 상호작용
- **은하 진화:** B-장이 있는 우주론적 시뮬레이션

**장점:**
- 자연스럽게 Lagrangian (유체 추적)
- 이류 오류 없음
- 큰 밀도 대비에 좋음

**단점:**
- 낮은 차수 정확도 (SPH 커널 $\sim O(h^2)$)
- $\nabla \cdot \mathbf{B}$ 오류 축적
- 비싼 이웃 검색

---

## 8. 생산 MHD 코드

### 8.1 Athena++

**타입:** 그리드 기반, finite volume

**기능:**
- MHD, SRMHD, radiation MHD
- Static mesh refinement (SMR) 및 AMR
- 곡선 좌표 (구형, 원통형)
- $\nabla \cdot \mathbf{B} = 0$를 위한 Constrained transport (CT)
- 다중 물리: 먼지, 우주선, 화학

**Riemann 솔버:** HLLC, HLLD, Roe

**사용 사례:**
- 강착 디스크 (MRI 난류)
- 원시별 제트
- 은하단 시뮬레이션

**웹사이트:** https://www.athena-astro.app/

### 8.2 PLUTO

**타입:** 그리드 기반, finite volume

**기능:**
- MHD, RMHD, RHD, HD
- 모듈형 물리: 냉각, 열전도, 입자
- 다중 Riemann 솔버
- 적응형 그리드 (CHOMBO AMR)
- 사용자 친화적 설정 (pluto.ini 구성 파일)

**좌표:** Cartesian, 원통형, 구형, 곡선

**사용 사례:**
- 천체물리학 제트
- 항성풍
- 행성-디스크 상호작용

**웹사이트:** http://plutocode.ph.unito.it/

### 8.3 FLASH

**타입:** 그리드 기반, AMR (Paramesh/CHOMBO)

**기능:**
- 압축성 MHD, radiation hydro
- 핵연소 (항성 진화)
- 중력 (Poisson 솔버, 트리 코드)
- 레이저 에너지 증착 (ICF)

**최적:** 다중 물리 시뮬레이션 (초신성, ICF, 항성 내부)

**웹사이트:** https://flash.rochester.edu/

### 8.4 Pencil Code

**타입:** Finite difference, 고차 (6차 공간)

**기능:**
- Cartesian 그리드에서 **스펙트럴 수준 정확도**
- MHD 난류, dynamo
- 비이상 효과: 양극성 확산, Hall 효과
- 먼지, 화학, 복사

**최적:**
- 난류의 Direct numerical simulation (DNS)
- 큰 $Re$, $Rm$ 연구
- Dynamo 이론

**웹사이트:** https://github.com/pencil-code/

### 8.5 Dedalus

**타입:** Spectral (Fourier + Chebyshev)

**기능:**
- 유연한 PDE 솔버 (사용자가 방정식을 상징적으로 지정)
- 병렬화 (MPI + 효율적인 고유값 솔버)
- 시간 단계: SBDF, RK443
- 선형 안정성 분석 (고유값 문제)

**최적:**
- 구형 쉘에서 대류, dynamo
- 안정성 분석 (MRI, MHD 불안정성)
- 맞춤 PDE

**웹사이트:** https://dedalus-project.org/

---

## 9. Python 구현: 2D Pseudo-Spectral MHD

### 9.1 문제 설정

강제가 있는 2D 비압축성 MHD 난류 시뮬레이션:
$$
\frac{\partial \mathbf{v}}{\partial t} = -(\mathbf{v} \cdot \nabla)\mathbf{v} + (\mathbf{B} \cdot \nabla)\mathbf{B} - \nabla p + \nu \nabla^2 \mathbf{v} + \mathbf{f}
$$
$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
$$

영역: $[0, 2\pi]^2$, 주기 BC, $N_x = N_y = 128$.

### 9.2 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# Parameters
N = 128          # Grid points
L = 2.0 * np.pi  # Domain size
nu = 1e-3        # Viscosity
eta = 1e-3       # Resistivity
dt = 0.001       # Timestep
T_end = 1.0      # End time
forcing_amp = 0.1

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wavenumbers
kx = fftfreq(N, L/(2*np.pi*N)) * 2*np.pi
ky = fftfreq(N, L/(2*np.pi*N)) * 2*np.pi
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
# K2[0,0] = 1은 압력 투영 P⊥ = k/k²에서 0으로 나누는 것을 방지;
# k=0 (평균) 모드는 정의상 발산이 없으므로 K2=1로 설정하면
# P⊥ = 0이 되어 평균 장을 변경하지 않고 올바르게 남김
K2[0, 0] = 1.0  # Avoid division by zero

# 비앨리아싱 마스크(Dealiasing Mask) (2/3 규칙): |k| ≤ N/3만 유지 (N/2가 아님),
# 최대 파수 N/3을 가진 두 함수의 곱이 최대 2N/3 ≤ N/2까지 모드를 생성하기 때문 —
# 나이퀴스트 한계 내에서 앨리아싱이 없음; N/3을 넘는 모드는 역 FFT 전에 0으로 설정하여
# 앨리아싱된 고-k 내용이 낮은 모드를 오염시키는 것을 방지
kmax = N // 3
dealias = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)

# Initial conditions (random vorticity + magnetic field)
np.random.seed(42)
vx = np.random.randn(N, N) * 0.1
vy = np.random.randn(N, N) * 0.1
Bx = np.sin(2*X) * np.cos(Y)
By = -np.cos(X) * np.sin(2*Y)

# 첫 번째 타임스텝 전에 초기 v와 B를 발산 없는 부분공간으로 투영:
# 임의의 초기 장은 일반적으로 0이 아닌 발산을 가지며, t=0에서 작은 ∇·v ≠ 0도
# 각 스텝에서 성장하는 가짜 압력을 생성할 것임
vx_hat = fftn(vx)
vy_hat = fftn(vy)
# 스펙트럴 공간에서의 div_v = ik_x * vx_hat + ik_y * vy_hat: 스펙트럴 공간에서
# 미분이 정확(절단 오차 없음)하므로, 이 발산은 연속 발산의 기계 정밀도 대표임
div_v = 1j*KX*vx_hat + 1j*KY*vy_hat
# 포텐셜 부분 빼기 (k * (k·v)/k²): 이것이 Helmholtz 분해(Helmholtz Decomposition) —
# 임의의 벡터 장은 고유하게 컬 없는(포텐셜) + 발산 없는 부분으로 분리됨;
# 포텐셜 부분을 빼면 솔레노이달(Solenoidal, 발산 없는) 나머지만 남음
vx_hat -= 1j*KX*div_v/K2
vy_hat -= 1j*KY*div_v/K2
vx = np.real(ifftn(vx_hat))
vy = np.real(ifftn(vy_hat))

# B에 대한 동일한 발산 청소: ∇·B = 0은 Maxwell 방정식에서 요구되는 제약;
# 스펙트럴 방법에서는 project_divergence_free()의 투영으로 매 스텝마다 유지되지만,
# t=0에서 이것이 성립하도록 보장하면 많은 타임스텝에 걸쳐 기계 정밀도 오차가
# 증폭되는 것을 방지함
Bx_hat = fftn(Bx)
By_hat = fftn(By)
div_B = 1j*KX*Bx_hat + 1j*KY*By_hat
Bx_hat -= 1j*KX*div_B/K2
By_hat -= 1j*KY*div_B/K2
Bx = np.real(ifftn(Bx_hat))
By = np.real(ifftn(By_hat))

def compute_nonlinear(vx, vy, Bx, By):
    """Compute nonlinear terms in physical space."""
    # Velocity advection
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    # 스펙트럴 공간에서 도함수를 계산(정확)한 후 물리 공간으로 변환:
    # 이것이 유사 스펙트럴(Pseudo-Spectral) 전략 — ik*f_hat이 정확한 스펙트럴 도함수를 제공하고,
    # 그런 다음 물리 공간에서 장과 곱해짐; 물리 공간에서 곱셈하면 스펙트럴 공간에서
    # 완전히 수행했을 때 발생할 비싼 합성곱(Convolution)을 피함
    dvx_dx = np.real(ifftn(1j*KX*vx_hat))
    dvx_dy = np.real(ifftn(1j*KY*vx_hat))
    dvy_dx = np.real(ifftn(1j*KX*vy_hat))
    dvy_dy = np.real(ifftn(1j*KY*vy_hat))

    NL_vx = -(vx*dvx_dx + vy*dvx_dy)
    NL_vy = -(vx*dvy_dx + vy*dvy_dy)

    # 자기 장력(Magnetic Tension) (B·∇)B: 이 항이 MHD를 순수 유체역학과 다르게 만듦 —
    # 장선을 따른 장력이 굽힘에 저항하며,
    # 이것이 Alfvén 파 전파와 자기 제동(Magnetic Braking) 뒤의 물리적 메커니즘
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)
    dBx_dx = np.real(ifftn(1j*KX*Bx_hat))
    dBx_dy = np.real(ifftn(1j*KY*Bx_hat))
    dBy_dx = np.real(ifftn(1j*KX*By_hat))
    dBy_dy = np.real(ifftn(1j*KY*By_hat))

    NL_vx += Bx*dBx_dx + By*dBx_dy
    NL_vy += Bx*dBy_dx + By*dBy_dy

    # 유도 방정식(Induction Equation) 우변: ∇×(v×B)을 성분별로 vx*∂B - B*∂v로 전개;
    # 반대칭 구조가 이상 진화 하에서 자기 헬리시티(Magnetic Helicity)를 보존 —
    # 이 형식에서 벗어나면 가짜 헬리시티 주입이 도입됨
    NL_Bx = vx*dBx_dx + vy*dBx_dy - Bx*dvx_dx - By*dvx_dy
    NL_By = vx*dBy_dx + vy*dBy_dy - Bx*dvy_dx - By*dvy_dy

    return NL_vx, NL_vy, NL_Bx, NL_By

def project_divergence_free(fx_hat, fy_hat):
    """Project to divergence-free subspace."""
    div_f = 1j*KX*fx_hat + 1j*KY*fy_hat
    fx_hat -= 1j*KX*div_f/K2
    fy_hat -= 1j*KY*div_f/K2
    return fx_hat, fy_hat

def rhs(vx, vy, Bx, By, t):
    """Compute RHS of equations."""
    NL_vx, NL_vy, NL_Bx, NL_By = compute_nonlinear(vx, vy, Bx, By)

    # Add forcing (low wavenumber)
    forcing_vx = forcing_amp * np.sin(2*X + t) * np.cos(Y)
    forcing_vy = forcing_amp * np.cos(X) * np.sin(2*Y + t)
    NL_vx += forcing_vx
    NL_vy += forcing_vy

    # 비선형 곱 FFT 후 비앨리아싱 마스크 적용: 물리 공간 곱셈이 앨리아싱 없이
    # 2*kmax까지 높은-k 모드를 생성 — dealias 배열이 이 마스크 이상의 모드를 0으로 설정하여
    # 에너지 계단(Energy Cascade)에서 낮은 모드를 오염시키지 않도록 함
    NL_vx_hat = fftn(NL_vx) * dealias
    NL_vy_hat = fftn(NL_vy) * dealias
    NL_Bx_hat = fftn(NL_Bx) * dealias
    NL_By_hat = fftn(NL_By) * dealias

    # 비선형 속도 강제를 발산 없는 부분공간으로 투영: v가 이미 발산 없더라도,
    # 비선형 항 (v·∇)v가 압축 가능한 성분을 생성할 수 있음; 투영이 이를 제거하고
    # Poisson 방정식을 명시적으로 풀 필요 없이 압력 방정식을 암묵적으로 강제함
    NL_vx_hat, NL_vy_hat = project_divergence_free(NL_vx_hat, NL_vy_hat)
    NL_Bx_hat, NL_By_hat = project_divergence_free(NL_Bx_hat, NL_By_hat)

    # Add diffusion
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)

    # -ν*k²*v_hat은 정확한 스펙트럴 확산: ∇²v ≈ (v_{i+1}-2v_i+v_{i-1})/Δx²가
    # O(Δx²) 오차를 가지는 유한 차분과 달리 공간 절단 오차가 없음;
    # Fourier 모드가 ∇²의 고유함수(Eigenfunctions)이기 때문에 스펙트럴 확산이 정확하며,
    # 이것이 스펙트럴 방법이 매끄러운 흐름에서 지수 수렴을 달성하는 이유
    dvx_dt_hat = NL_vx_hat - nu*K2*vx_hat
    dvy_dt_hat = NL_vy_hat - nu*K2*vy_hat
    dBx_dt_hat = NL_Bx_hat - eta*K2*Bx_hat
    dBy_dt_hat = NL_By_hat - eta*K2*By_hat

    return (np.real(ifftn(dvx_dt_hat)), np.real(ifftn(dvy_dt_hat)),
            np.real(ifftn(dBx_dt_hat)), np.real(ifftn(dBy_dt_hat)))

# Time integration (RK4)
def rk4_step(vx, vy, Bx, By, dt, t):
    k1 = rhs(vx, vy, Bx, By, t)
    k2 = rhs(vx + 0.5*dt*k1[0], vy + 0.5*dt*k1[1],
             Bx + 0.5*dt*k1[2], By + 0.5*dt*k1[3], t + 0.5*dt)
    k3 = rhs(vx + 0.5*dt*k2[0], vy + 0.5*dt*k2[1],
             Bx + 0.5*dt*k2[2], By + 0.5*dt*k2[3], t + 0.5*dt)
    k4 = rhs(vx + dt*k3[0], vy + dt*k3[1],
             Bx + dt*k3[2], By + dt*k3[3], t + dt)

    vx_new = vx + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    vy_new = vy + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    Bx_new = Bx + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    By_new = By + (dt/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    return vx_new, vy_new, Bx_new, By_new

# Energy spectrum
def compute_spectrum(vx, vy, Bx, By):
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    Bx_hat = fftn(Bx)
    By_hat = fftn(By)

    E_kin = 0.5*(np.abs(vx_hat)**2 + np.abs(vy_hat)**2)
    E_mag = 0.5*(np.abs(Bx_hat)**2 + np.abs(By_hat)**2)

    k_bins = np.arange(1, N//2)
    E_kin_spec = np.zeros(len(k_bins))
    E_mag_spec = np.zeros(len(k_bins))

    K_mag = np.sqrt(KX**2 + KY**2)
    for i, k in enumerate(k_bins):
        mask = (K_mag >= k) & (K_mag < k+1)
        E_kin_spec[i] = np.sum(E_kin[mask])
        E_mag_spec[i] = np.sum(E_mag[mask])

    return k_bins, E_kin_spec, E_mag_spec

# Main loop
t = 0.0
step = 0
snapshots = []

print("Running 2D spectral MHD simulation...")
while t < T_end:
    if step % 100 == 0:
        E_kin_total = 0.5*np.sum(vx**2 + vy**2)
        E_mag_total = 0.5*np.sum(Bx**2 + By**2)
        print(f"Step {step:4d}, t={t:.3f}, E_kin={E_kin_total:.4f}, E_mag={E_mag_total:.4f}")

        if len(snapshots) < 4:
            snapshots.append((t, vx.copy(), vy.copy(), Bx.copy(), By.copy()))

    vx, vy, Bx, By = rk4_step(vx, vy, Bx, By, dt, t)
    t += dt
    step += 1

# Final spectrum
k_bins, E_kin_spec, E_mag_spec = compute_spectrum(vx, vy, Bx, By)

# Plot spectra
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(k_bins, E_kin_spec, 'b-', label='Kinetic', linewidth=2)
ax.loglog(k_bins, E_mag_spec, 'r--', label='Magnetic', linewidth=2)

# Reference slopes
k_ref = np.array([2, 20])
ax.loglog(k_ref, 1e2*k_ref**(-5.0/3.0), 'k:', label=r'$k^{-5/3}$', alpha=0.5)
ax.loglog(k_ref, 1e2*k_ref**(-3.0/2.0), 'g:', label=r'$k^{-3/2}$', alpha=0.5)

ax.set_xlabel('Wavenumber k')
ax.set_ylabel('Energy E(k)')
ax.set_title('MHD Turbulence Energy Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('mhd_spectrum.png', dpi=150, bbox_inches='tight')
print("Spectrum saved: mhd_spectrum.png")
plt.close()

# Plot snapshots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, (t_snap, vx_snap, vy_snap, Bx_snap, By_snap) in enumerate(snapshots):
    ax = axes[i // 2, i % 2]
    vorticity = np.real(ifftn(-K2 * fftn(vx_snap)))  # Simplified vorticity
    im = ax.contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
    ax.streamplot(X, Y, Bx_snap, By_snap, color='k', density=0.8, linewidth=0.5)
    ax.set_title(f't = {t_snap:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Vorticity')

plt.tight_layout()
plt.savefig('mhd_snapshots.png', dpi=150, bbox_inches='tight')
print("Snapshots saved: mhd_snapshots.png")
plt.close()
```

### 9.3 예상 출력

- **에너지 스펙트럼:** 멱법칙 감소 ($k^{-5/3}$ 또는 $k^{-3/2}$ 확인)
- **스냅샷:** 와도 윤곽 + 자기장 선
- **에너지 보존:** 총 에너지는 대략 일정하게 유지 ($\nu$, $\eta$로부터 느린 소산과 함께)

---

## 10. Chebyshev 평형 솔버

### 10.1 1D 힘 균형

해결:
$$
\frac{d}{dx}\left(p + \frac{B^2}{2}\right) = 0
$$
$x \in [-1, 1]$에서 $p(x)$, $B(x)$에 대해 BC: $p(-1) = 1$, $B(-1) = 0.5$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Chebyshev differentiation matrix
def cheb_diff_matrix(N):
    """Compute Chebyshev differentiation matrix."""
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1)
    c[0] = c[-1] = 2.0
    c[1:-1] = 1.0
    c *= (-1)**np.arange(N+1)

    X = np.tile(x, (N+1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0/c) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D, axis=1))

    return D, x

N = 32
D, x = cheb_diff_matrix(N)

# Initial guess
p = 1.0 - 0.5*(x + 1.0)
B = 0.5 + 0.3*(x + 1.0)

# Solve force balance: d/dx(p + B^2/2) = 0
# Integrate: p + B^2/2 = const
# BC: p(-1) = 1, B(-1) = 0.5 => const = 1 + 0.5^2/2 = 1.125

const = 1.0 + 0.5**2 / 2.0
p_sol = const - B**2 / 2.0

# Verify
ptot = p_sol + B**2 / 2.0
dptot_dx = D @ ptot

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, p_sol, 'b-', linewidth=2, label='Pressure p')
axes[0].set_xlabel('x')
axes[0].set_ylabel('p')
axes[0].set_title('Pressure Profile')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(x, B, 'r-', linewidth=2, label='Magnetic Field B')
axes[1].set_xlabel('x')
axes[1].set_ylabel('B')
axes[1].set_title('Magnetic Field Profile')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(x, dptot_dx, 'g-', linewidth=2, label=r'$d(p+B^2/2)/dx$')
axes[2].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[2].set_xlabel('x')
axes[2].set_ylabel('Residual')
axes[2].set_title('Force Balance Residual')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig('chebyshev_equilibrium.png', dpi=150, bbox_inches='tight')
print("Plot saved: chebyshev_equilibrium.png")
plt.close()
```

---

## 11. AMR 개념 시연

### 11.1 1D 적응형 그리드

```python
import numpy as np
import matplotlib.pyplot as plt

# Function with sharp gradient
def f(x):
    return np.tanh(20*(x - 0.5))

# Initial coarse grid
x_coarse = np.linspace(0, 1, 11)
f_coarse = f(x_coarse)

# Refinement criterion: large gradient
df_dx = np.gradient(f_coarse, x_coarse)
refine_indices = np.where(np.abs(df_dx) > 5.0)[0]

# Refined grid (insert midpoints)
x_fine = []
f_fine = []
for i in range(len(x_coarse)):
    x_fine.append(x_coarse[i])
    f_fine.append(f_coarse[i])
    if i in refine_indices and i < len(x_coarse) - 1:
        x_mid = 0.5*(x_coarse[i] + x_coarse[i+1])
        x_fine.append(x_mid)
        f_fine.append(f(x_mid))

x_fine = np.array(x_fine)
f_fine = np.array(f_fine)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_exact = np.linspace(0, 1, 500)
f_exact = f(x_exact)

axes[0].plot(x_exact, f_exact, 'k-', linewidth=1, label='Exact', alpha=0.5)
axes[0].plot(x_coarse, f_coarse, 'bo-', linewidth=2, label='Coarse Grid', markersize=8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Coarse Grid (11 points)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_exact, f_exact, 'k-', linewidth=1, label='Exact', alpha=0.5)
axes[1].plot(x_fine, f_fine, 'ro-', linewidth=2, label='AMR Grid', markersize=6)
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title(f'AMR Grid ({len(x_fine)} points)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('amr_concept.png', dpi=150, bbox_inches='tight')
print("Plot saved: amr_concept.png")
print(f"Coarse grid: {len(x_coarse)} points")
print(f"AMR grid: {len(x_fine)} points")
plt.close()
```

**출력:** AMR은 높은 기울기 영역 ($x = 0.5$ 주변)에 점을 집중시킵니다.

---

## 요약

고급 MHD 방법은 기본 finite-volume 접근법을 확장합니다:

1. **Pseudo-spectral:** 매끄럽고 주기적인 문제에 대해 지수적 정확도 (MHD 난류, dynamo)
2. **Dealiasing:** 올바른 비선형 항 평가에 필수적 (2/3 규칙)
3. **Chebyshev:** 비주기 BC, 경계값 문제에 이상적 (평형, 안정성)
4. **AMR:** 다중 스케일 문제를 위한 적응형 해상도 (충격파, 전류 시트)
5. **Hybrid MHD-PIC:** 유체 벌크 + 운동론적 입자 결합 (우주선)
6. **SPH-MHD:** 별 형성, 천체물리학 유동을 위한 Lagrangian 방법
7. **생산 코드:** Athena++, PLUTO, FLASH (그리드), Pencil Code (고차 FD), Dedalus (스펙트럴)

**사용 시기:**
- **Spectral:** 난류 DNS, dynamo (높은 $Re$, $Rm$)
- **AMR:** 충격파, 제트, 강착 (다중 스케일)
- **Chebyshev:** 평형, 선형 안정성
- **Hybrid:** 우주선 가속, 에너지적 입자

각 방법은 절충이 있습니다; 물리학과 계산 자원에 따라 선택하세요.

---

## 연습 문제

1. **Fourier 도함수:**
   $[0, 2\pi]$에서 $N = 16$ 점으로 $f(x) = \sin(3x)$에 대해 FFT를 사용하여 $f'(x)$를 계산하세요. 정확한 $f'(x) = 3\cos(3x)$와 비교하세요. 최대 오차는 얼마입니까?

2. **2/3 Dealiasing:**
   $f(x) = \sin(5x)$와 $g(x) = \sin(7x)$를 고려하세요. 그들의 곱은 최대 파수 12를 갖습니다. $N = 16$ (Nyquist $k_N = 8$)일 때, dealiasing 없이 aliasing이 발생합니까? alias된 파수는 무엇입니까?

3. **Chebyshev 점:**
   $N = 8$ Gauss-Lobatto Chebyshev 점 $x_j = \cos(\pi j / N)$을 계산하세요. 분포를 그리세요. 왜 끝점에 클러스터링됩니까?

4. **에너지 스펙트럼 기울기:**
   2D 스펙트럴 MHD 코드에서 $\nu$와 $\eta$를 2배씩 변화시키세요. 관성 범위 기울기가 변합니까? 어느 $Re$에서 명확한 $k^{-5/3}$ 범위를 관찰합니까?

5. **AMR 효율:**
   1D 충격파 (Sod 테스트)에 대해, 충격파를 1% 정확도로 해결하기 위해 필요한 균일 그리드 vs 2-레벨 AMR (정제 인수 4) 그리드 점 수를 추정하세요. 충격파 폭 $\sim 10$ 셀 가정.

6. **Hybrid MHD-PIC:**
   hybrid 코드에서 우주선 입자가 $D = 10^{28}$ cm²/s로 확산합니다. MHD 그리드가 $\Delta x = 10^{10}$ cm를 가지면, 확산 CFL 조건 $\Delta t < (\Delta x)^2 / D$로부터 타임스텝을 추정하세요.

7. **SPH 커널:**
   $r < h$에 대해 cubic spline 커널 $W(r, h) \propto (1 - r/h)^3$에 대해 $\int W(\mathbf{r}, h) d^3r = 1$을 확인하세요. 2D에서 정규화 상수를 계산하세요.

8. **Athena++ vs PLUTO:**
   조사: Athena++가 MHD에 대해 기본적으로 어떤 Riemann 솔버를 사용합니까? PLUTO의 기본과 비교하세요. 어느 것이 더 확산적입니까?

---

**이전:** [상대론적 MHD](./16_Relativistic_MHD.md) | **다음:** [프로젝트](./18_Projects.md)

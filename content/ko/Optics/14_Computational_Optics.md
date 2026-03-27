# 14. 전산 광학

[← 이전: 13. 양자광학 입문](13_Quantum_Optics_Primer.md) | [다음: 15. 체르니케 다항식 →](15_Zernike_Polynomials.md)

---

이 강좌 전반에 걸쳐 우리는 빛이 어떻게 전파하고, 간섭하고, 회절하고, 물질과 상호작용하는지를 공부해 왔으며, 이러한 현상들을 기술하기 위해 주로 해석적 공식에 의존해 왔습니다. 그러나 실제 광학 시스템은 닫힌 형태의 해(closed-form solution)를 구하기에는 너무 복잡합니다. 카메라 렌즈에는 비구면 프로파일을 가진 10개 이상의 면이 있고, 광자 결정(Photonic Crystal) 도파로에는 복잡한 패턴으로 빛을 산란시키는 파장 이하(subwavelength) 특징들이 있으며, 적응 광학(adaptive optics) 시스템은 실시간으로 대기 난류를 보정해야 합니다. **전산 광학(Computational Optics)**은 수치적 방법을 활용하여 광학 시스템을 설계, 시뮬레이션하고, 심지어 전통적인 광학 소자를 대체함으로써 광학 이론과 실용 공학 사이의 간극을 메우는 분야입니다.

이 분야는 두 가지 상호 보완적인 면을 가집니다. 한편으로는 컴퓨터가 광학 시스템을 통한 빛의 전파를 시뮬레이션합니다 — 복잡한 렌즈를 통한 단순한 광선 추적부터 나노광자 소자를 위한 완전파(full-wave) 맥스웰 솔버까지. 다른 한편으로는 계산이 물리적 광학을 대체합니다. 위상 복원(phase retrieval) 알고리즘이 검출기가 직접 측정할 수 없는 정보를 복원하고, 전산 사진(computational photography) 기법이 어떤 단일 노출로도 포착할 수 없는 이미지를 합성하며, 머신러닝 모델이 인간의 직관으로는 상상조차 하지 못할 광학 소자를 설계합니다. 이러한 접근법들이 합쳐져 광학을 유리를 연마하는 기술에서 전산 과학으로 변환시켰습니다.

이 레슨은 고전적인 광선 추적에서 현대 머신러닝 기반 역설계까지, 광학의 주요 전산 방법들을 개괄하며 이론적 기초와 실용적 구현 모두를 제공합니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 스넬의 법칙과 ABCD 전달 행렬법을 사용하여 다면 광학 시스템을 통한 순차 광선 추적을 구현한다
2. 근축 파동 방정식으로부터 빔 전파법(BPM)을 유도하고 분리 단계 푸리에 알고리즘을 구현한다
3. 이(Yee) 격자에서 맥스웰 방정식을 풀기 위한 FDTD 방법을, 안정성 조건 및 흡수 경계를 포함하여 설명한다
4. 반복 위상 복원을 위한 게르히베르크-색스턴(Gerchberg-Saxton) 알고리즘을 구현하고 강도의 전파 방정식(TIE)을 설명한다
5. 샥-하트만(Shack-Hartmann) 센서를 이용한 파면 감지를 기술하고 제르니케(Zernike) 다항식으로 수차를 특성화한다
6. 라이트 필드 이미징, HDR, 코딩된 조리개(coded aperture)를 포함한 전산 사진 기법을 설명한다
7. 각 스펙트럼법과 위상 되풀이(phase unwrapping)를 사용한 디지털 홀로그래피 복원을 기술한다
8. 학습된 위상 복원, 초해상도, 역설계를 포함한 광학에서의 머신러닝 응용을 파악한다

---

## 목차

1. [기하 광선 추적](#1-기하-광선-추적)
2. [빔 전파법 (BPM)](#2-빔-전파법-bpm)
3. [광학을 위한 FDTD](#3-광학을-위한-fdtd)
4. [위상 복원](#4-위상-복원)
5. [파면 감지](#5-파면-감지)
6. [전산 사진](#6-전산-사진)
7. [디지털 홀로그래피](#7-디지털-홀로그래피)
8. [광학에서의 머신러닝](#8-광학에서의-머신러닝)
9. [Python 예제](#9-python-예제)
10. [요약](#10-요약)
11. [연습문제](#11-연습문제)
12. [참고문헌](#12-참고문헌)

---

## 1. 기하 광선 추적

### 1.1 광선 추적 패러다임

광선 추적(Ray Tracing)은 광학 설계에서 가장 오래되고 가장 널리 사용되는 전산 방법입니다. 빛을 균질 매질에서 직선으로 이동하고 스넬의 법칙에 따라 계면에서 굴절하는 광선들의 집합으로 취급합니다. 회절과 간섭을 무시함에도 불구하고, 기하 광선 추적은 렌즈 설계의 핵심 도구로 남아 있습니다 — 현대 카메라 렌즈는 수백만 개의 광선을 각 면을 통해 추적하여 거의 전적으로 설계됩니다.

> **비유**: 광선 추적은 서로 다른 속도 제한을 가진 여러 나라를 통과하는 자동차 여행 계획과 같습니다. 각 국경(광학 면)에서 지역 "빛의 속도"(굴절률)에 따라 방향을 재계산합니다. ABCD 행렬법은 여행의 각 구간마다 하나의 전달 행렬을 곱하는 GPS를 갖는 것과 같습니다 — 마지막에는 경로를 단계별로 되짚지 않고도 하나의 행렬 곱셈으로 최종 위치와 각도를 알 수 있습니다.

### 1.2 순차 광선 추적

순차 광학 시스템(빛이 고정된 순서로 면들을 통과하는)에서, 각 광선은 각 면에서의 광축 위 높이 $y$와 각도 $\theta$ (또는 근축 한계에서 경사 $u = \tan\theta \approx \theta$)로 정의됩니다.

굴절률 $n_1$과 $n_2$를 가진 매질을 분리하는, 곡률 반경 $R$의 굴절면에서 벡터 형태의 스넬의 법칙은 다음과 같습니다:

$$n_2 \hat{\mathbf{s}}_2 = n_1 \hat{\mathbf{s}}_1 + (n_2\cos\theta_2 - n_1\cos\theta_1)\hat{\mathbf{n}}$$

여기서 $\hat{\mathbf{s}}_1, \hat{\mathbf{s}}_2$는 입사 및 굴절 광선의 단위 방향 벡터, $\hat{\mathbf{n}}$은 면의 법선이며, 각도는 법선으로부터 측정됩니다. 정밀(비근축) 추적의 경우, 광선-면 교차점을 계산하고, 국소 법선을 찾고, 소각도 근사 없이 스넬의 법칙을 적용합니다.

### 1.3 ABCD 행렬법

**근축 근사(Paraxial Approximation)** ($\sin\theta \approx \theta$)에서, 입력과 출력 평면에서의 광선 매개변수 $(y, u)$ 사이의 관계는 선형입니다:

$$\begin{pmatrix} y_{\text{out}} \\ u_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} y_{\text{in}} \\ u_{\text{in}} \end{pmatrix}$$

주요 행렬들은 (이미 레슨 8에서 가우스 빔에 도입된) 다음과 같습니다:

**자유 공간 전파(Free-Space Propagation)** (굴절률 $n$의 매질에서 거리 $d$):

$$M_{\text{prop}} = \begin{pmatrix} 1 & d/n \\ 0 & 1 \end{pmatrix}$$

**구면에서의 굴절(Refraction at a Spherical Surface)** (반경 $R$, $n_1$에서 $n_2$로):

$$M_{\text{refr}} = \begin{pmatrix} 1 & 0 \\ -(n_2 - n_1)/R & 1 \end{pmatrix}$$

참고: 이 표기는 $R > 0$이 곡률 중심이 오른쪽에 있음을 의미하는 부호 규약을 사용합니다.

**박막 렌즈(Thin Lens)** (초점 거리 $f$):

$$M_{\text{lens}} = \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$$

완전한 시스템의 경우, 행렬을 역순으로 (오른쪽에서 왼쪽으로) 곱합니다:

$$M_{\text{sys}} = M_N \cdot M_{N-1} \cdots M_2 \cdot M_1$$

시스템 행렬은 근축 결상에 대한 모든 것을 담고 있습니다: 유효 초점 거리는 $f = -1/C$이며, 상의 위치는 $B = 0$ 조건으로 구할 수 있습니다.

### 1.4 스팟 다이어그램과 광선 팬

**스팟 다이어그램(Spot Diagram)**은 단일 물체점에서 출발한 광선 다발이 상 평면에 어디에 맺히는지를 보여줍니다. 완벽한 렌즈의 경우 모든 광선이 하나의 점으로 수렴하지만, 실제로는 수차(aberration)가 이를 퍼뜨립니다. 스팟의 형태가 지배적인 수차를 드러냅니다:
- 원형 퍼짐 → 구면 수차(Spherical Aberration)
- 혜성 모양 → 코마(Coma)
- 십자 모양 → 비점 수차(Astigmatism)

**광선 팬(Ray Fan)**(또는 광선 수차 도)은 입사 동공 높이 $h$의 함수로서 횡방향 광선 오차 $\Delta y$를 나타냅니다. 서로 다른 곡선이 서로 다른 자이델(Seidel) 수차를 드러냅니다:
- $\Delta y \propto h^3$ → 구면 수차
- $\Delta y \propto h^2$ → 코마
- $\Delta y \propto h$ → 디포커스(defocus) 또는 상면 만곡(field curvature)

### 1.5 근축을 넘어서: 실제 광선 추적

실제 렌즈 설계에서는 근축 근사가 불충분합니다. Zemax, Code V, OpticStudio 같은 소프트웨어는 스넬의 법칙의 완전한 형태를 사용하여 정밀 광선을 추적합니다. 각 광선-면 교차점은 반복적으로 구해지고(비구면의 경우 뉴턴법), 굴절은 소각도 근사 없이 계산됩니다. 수백만 개의 광선을 추적하여 상질(image quality)의 통계적 그림을 구성합니다.

---

## 2. 빔 전파법 (BPM)

### 2.1 광선으로 충분하지 않을 때

광선 추적은 회절 효과가 중요한 경우 — 특징 크기가 파장에 가까울 때, 또는 빛이 도파 구조(광섬유, 집적 도파로) 내에서 장거리를 전파할 때 — 에는 실패합니다. **빔 전파법(Beam Propagation Method, BPM)**은 근축 파동 방정식을 수치적으로 풀어 회절과 공간적으로 변하는 굴절률의 효과를 모두 포착합니다.

### 2.2 근축 파동 방정식

단색 장(monochromatic field) $U(\mathbf{r})e^{-i\omega t}$에 대한 헬름홀츠(Helmholtz) 방정식으로부터 시작합니다:

$$\nabla^2 U + k_0^2 n^2(x, y, z) U = 0$$

$U = \psi(x, y, z) e^{ikz}$로 쓰며, 여기서 $\psi$는 $z$를 따라 전파하는 천천히 변하는 포락선(slowly varying envelope)이고 $k = k_0 n_0$는 기준 파수($n_0$는 배경 굴절률)입니다. 대입하고 $\partial^2\psi/\partial z^2$를 무시하면(**근축** 또는 **천천히 변하는 포락선** 근사):

$$\frac{\partial \psi}{\partial z} = \frac{i}{2k}\nabla_\perp^2 \psi + \frac{ik_0}{2n_0}\Delta n^2(x,y,z)\,\psi$$

여기서 $\nabla_\perp^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$는 횡방향 라플라시안이고 $\Delta n^2 = n^2 - n_0^2 \approx 2n_0\Delta n$은 작은 굴절률 섭동에 대한 근사입니다.

이 방정식은 다음의 형태를 가집니다:

$$\frac{\partial\psi}{\partial z} = (i\hat{D} + i\hat{N})\psi$$

여기서 $\hat{D} = \nabla_\perp^2/(2k)$는 **회절 연산자(Diffraction Operator)**이고 $\hat{N} = k_0\Delta n^2/(2n_0)$는 **굴절률 연산자(Refractive Index Operator)**입니다.

> **비유**: BPM은 기울어지고 울퉁불퉁한 표면 위를 굴러가는 공을 시뮬레이션하는 것과 같습니다. 회절 연산자 $\hat{D}$는 공이 자연스럽게 퍼지는 경향을 처리하고(파동이 회절하듯), 굴절률 연산자 $\hat{N}$는 공을 조종하는 요철을 처리합니다(도파로가 빛을 가두듯). 분리 단계법은 두 효과를 각각 별도로 처리하면서 번갈아 가며 진행합니다 — 먼저 공이 짧은 단계 동안 자유롭게 퍼지도록 하고, 그 다음 요철에 따라 조정하고, 반복합니다.

### 2.3 분리 단계 푸리에법

분리 단계법은 두 연산자를 번갈아 가며 $\psi$를 작은 단계 $\Delta z$만큼 전진시킵니다:

$$\psi(z + \Delta z) \approx e^{i\hat{N}\Delta z}\,\mathcal{F}^{-1}\!\left\{e^{i\hat{D}_k \Delta z}\,\mathcal{F}\{\psi(z)\}\right\}$$

여기서 $\hat{D}_k = -(k_x^2 + k_y^2)/(2k)$는 공간 주파수 영역에서의 회절 연산자입니다. 알고리즘은 다음과 같습니다:

1. **순방향 FFT**: $\psi(x,y)$를 공간 주파수 영역 $\tilde{\psi}(k_x, k_y)$로 변환
2. **회절 단계**: $\exp\!\left[-i(k_x^2 + k_y^2)\Delta z/(2k)\right]$를 곱함
3. **역 FFT**: 다시 공간 영역으로 변환
4. **굴절 단계**: $\exp\!\left[ik_0\Delta n(x,y,z)\Delta z / n_0\right]$를 곱함 ($\Delta n^2 \approx 2n_0\Delta n$ 사용)
5. 각 $z$ 단계마다 반복

핵심 통찰은 회절 연산자는 푸리에 영역에서 대각(각 공간 주파수가 자유 공간에서 독립적으로 전파)인 반면, 굴절률 연산자는 실공간에서 대각(각 점에서 국소적으로 작용)이라는 점입니다.

### 2.4 각 스펙트럼 전파법

밀접하게 관련된 방법인 **각 스펙트럼법(Angular Spectrum Method)**은 임의의 장 $U(x,y,z_0)$를 근축 근사 없이 새로운 평면 $z_0 + d$로 전파시킵니다:

$$U(x, y, z_0 + d) = \mathcal{F}^{-1}\!\left\{\tilde{U}(k_x, k_y, z_0) \cdot e^{ik_z d}\right\}$$

여기서 $k_z = \sqrt{k^2 - k_x^2 - k_y^2}$ (전파파의 경우; $k_x^2 + k_y^2 > k^2$이면 에바네센트파(evanescent)). 이는 스칼라 회절 이론 내에서 정확하며, 디지털 홀로그래피의 수치적 전파 기반입니다(7절).

### 2.5 응용

BPM은 다음을 시뮬레이션하는 표준 도구입니다:
- **광섬유 모드와 전파**: 굴절률 점진 변화형(graded-index) 및 광자 결정 광섬유
- **집적 광자 도파로**: 방향성 결합기(directional coupler), Y-분기, 링 공진기
- **레이저 빔 전파**: 난류 대기를 통한 전파
- **굴절률 점진 변화형(GRIN) 렌즈**: 연속적으로 변하는 굴절률

---

## 3. 광학을 위한 FDTD

### 3.1 완전파 시뮬레이션

근축 근사가 실패하는 경우 — 파장 이하의 특징, 급격한 굴곡, 또는 금속 나노구조에서 — 맥스웰 방정식을 직접 풀어야 합니다. 1966년 케인 이(Kane Yee)가 도입한 **유한 차분 시간 영역(Finite-Difference Time-Domain, FDTD)** 방법이 정확히 이를 수행합니다: 격자에서 맥스웰의 회전 방정식(curl equations)을 이산화하고 시간을 앞으로 단계적으로 진행시킵니다.

### 3.2 맥스웰의 회전 방정식

소스가 없는 선형 매질에서:

$$\frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu}\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{E}}{\partial t} = \frac{1}{\epsilon}\nabla \times \mathbf{H}$$

FDTD는 공간과 시간 모두에서 중심 차분을 사용하여 이 방정식들을 이산화합니다.

### 3.3 이(Yee) 격자

이(Yee)의 핵심 통찰은 $\mathbf{E}$와 $\mathbf{H}$ 장 성분들을 공간과 시간 모두에서 어긋나게(stagger) 배치하는 것이었습니다:

- $E_x, E_y, E_z$와 $H_x, H_y, H_z$는 단위 셀의 서로 다른 점들에 배치됨
- $\mathbf{E}$는 정수 시간 단계에서 갱신되고, $\mathbf{H}$는 반정수 시간 단계에서 갱신됨

이 어긋남은 가우스 법칙($\nabla \cdot \mathbf{E} = 0$, $\nabla \cdot \mathbf{B} = 0$)을 자연스럽게 만족시키며 공간과 시간 모두에서 2차 정확도를 제공합니다. 1D에서 $E_x$와 $H_y$의 갱신 방정식은:

$$H_y^{n+1/2}(i+\tfrac{1}{2}) = H_y^{n-1/2}(i+\tfrac{1}{2}) - \frac{\Delta t}{\mu\Delta x}\left[E_x^n(i+1) - E_x^n(i)\right]$$

$$E_x^{n+1}(i) = E_x^n(i) + \frac{\Delta t}{\epsilon(i)\Delta x}\left[H_y^{n+1/2}(i+\tfrac{1}{2}) - H_y^{n+1/2}(i-\tfrac{1}{2})\right]$$

여기서 위첨자는 시간 단계를, 인자는 공간 격자점을 나타냅니다.

### 3.4 안정성: 쿠랑 조건

시간 단계 $\Delta t$는 수치적 안정성을 위한 **쿠랑-프리드리히스-레비(Courant-Friedrichs-Lewy, CFL)** 조건을 만족해야 합니다:

$$c\Delta t \leq \frac{1}{\sqrt{1/\Delta x^2 + 1/\Delta y^2 + 1/\Delta z^2}}$$

1D에서 이는 $c\Delta t \leq \Delta x$로 단순화됩니다. 균일한 격자 간격 $\Delta$의 3D에서: $c\Delta t \leq \Delta/\sqrt{3}$.

공간 격자는 또한 시뮬레이션의 가장 짧은 파장을 해상해야 합니다. 유전체 구조에서 정확한 결과를 위한 일반적인 경험 규칙은 $\Delta \leq \lambda_{\min}/20$이며, 장이 급격히 변하는 금속 표면 근처에서는 $\Delta \leq \lambda_{\min}/40$입니다.

### 3.5 흡수 경계 조건: PML

FDTD 시뮬레이션은 유한한 계산 영역이 필요하지만, 나가는 파동이 경계에서 반사되지 않아야 합니다. 베렝거(Berenger, 1994)가 도입한 **완벽 정합층(Perfectly Matched Layers, PML)**은 모든 입사각에서 반사 없이 파동을 흡수하는 인공 흡수 매질로 계산 영역을 둘러쌉니다. PML은 공간 좌표를 복소 평면으로 해석학적으로 연속시켜 작동합니다:

$$\tilde{x} = \int_0^x s_x(x')\,dx', \quad s_x(x') = 1 + \frac{\sigma_x(x')}{i\omega\epsilon_0}$$

여기서 $\sigma_x$는 PML 경계에서 0에서 PML 내부 깊숙한 곳의 최대값까지 증가하는 전도도 프로파일입니다. 지수 프로파일 $\sigma_x(x) = \sigma_{\max}(x/d)^m$, $m = 3\text{-}4$가 실제로 잘 작동합니다.

### 3.6 광자공학에서의 응용

FDTD는 다음 분야의 선호 도구입니다:
- **광자 결정(Photonic Crystals)**: 밴드 구조 계산, 결함 모드, 느린 빛
- **플라즈모닉 나노구조(Plasmonic Nanostructures)**: 금속 나노입자 근처의 장 향상, 나노안테나
- **메타물질(Metamaterials)**: 음굴절률, 투명망토(cloaking), 완벽 흡수체
- **나노광자 소자**: 도파로 굴곡, 격자 결합기(grating coupler), 공진기
- **태양 전지**: 나노구조화된 박막에서의 빛 포획

인기 있는 FDTD 소프트웨어로는 Lumerical FDTD, Meep (오픈 소스, MIT 개발), COMSOL (주파수 영역 FEM, 하지만 FDTD와 자주 비교됨) 등이 있습니다.

> **비유**: BPM이 강이 하류로 흐르는 것을 보는 것(순방향 전파만)이라면, FDTD는 전체 바다를 보는 것과 같습니다 — 파동은 모든 방향으로 이동하고, 반사하고, 산란하고, 간섭하며, 시간 진화의 모든 세부 사항을 포착합니다. 대가는 계산 비용입니다: FDTD는 세 공간 차원 모두와 시간을 이산화해야 하는 반면, BPM은 하나의 축을 따라 단계적으로 진행합니다.

---

## 4. 위상 복원

### 4.1 위상 문제

광학 검출기 — 카메라, CCD, 광다이오드 — 는 **강도(Intensity)** $I = |U|^2$를 측정합니다. 복소 장 $U = |U|e^{i\phi}$의 **위상(Phase)** $\phi$는 손실됩니다. 그러나 위상은 중요한 정보를 담고 있습니다: 파면 형상, 깊이, 굴절률 변화를 인코딩합니다. 강도 측정으로부터 위상을 복원하는 것이 **위상 문제(Phase Problem)**이며, 광학에서 가장 중요한 역문제 중 하나입니다.

위상 문제는 다음에서 발생합니다:
- **천문학**: 망원경 광학의 수차 측정
- **전자 현미경**: 원자 규모 구조 이미징
- **X선 결정학**: 분자 구조 결정 (여러 노벨상을 수상한 분야)
- **적응 광학**: 대기 난류 보정
- **결맞음 회절 이미징**: 렌즈 없이 나노미터 해상도로 이미징

### 4.2 게르히베르크-색스턴 알고리즘

**게르히베르크-색스턴(Gerchberg-Saxton, GS) 알고리즘**(1972)은 기본적인 반복 위상 복원 방법입니다. 두 개의 강도 측정 — 물체 평면($I_{\text{obj}} = |U_{\text{obj}}|^2$)과 원시야 또는 푸리에 평면($I_{\text{far}} = |\tilde{U}|^2$)에서의 — 이 주어졌을 때 파면의 위상을 복원합니다.

**알고리즘**:

1. 무작위 위상 추정으로 시작: $U_{\text{obj}}^{(0)} = \sqrt{I_{\text{obj}}}\,e^{i\phi_{\text{random}}}$
2. 푸리에 평면으로 전파: $\tilde{U}^{(k)} = \mathcal{F}\{U_{\text{obj}}^{(k)}\}$
3. **푸리에 구속 조건 적용**: 진폭을 측정된 $\sqrt{I_{\text{far}}}$로 교체하고 위상 유지:
   $$\tilde{U}_{\text{corrected}}^{(k)} = \sqrt{I_{\text{far}}}\,\frac{\tilde{U}^{(k)}}{|\tilde{U}^{(k)}|}$$
4. 역방향 전파: $U_{\text{obj}}^{(k)} = \mathcal{F}^{-1}\{\tilde{U}_{\text{corrected}}^{(k)}\}$
5. **물체 구속 조건 적용**: 진폭을 측정된 $\sqrt{I_{\text{obj}}}$로 교체하고 위상 유지:
   $$U_{\text{obj}}^{(k+1)} = \sqrt{I_{\text{obj}}}\,\frac{U_{\text{obj}}^{(k)}}{|U_{\text{obj}}^{(k)}|}$$
6. 수렴할 때까지 반복 (위상이 안정화될 때)

GS 알고리즘은 **교대 투영법(Alternating Projections)**의 예입니다: 각 단계는 현재 추정치를 하나의 측정치와 일치하는 장의 집합으로 투영합니다. 단조 수렴하지만(오차가 증가하지 않음), 국소 최솟값에 정체될 수 있습니다.

> **비유**: 두 다른 각도에서 물체의 그림자를 알고 있지만 3D 형태를 재구성해야 한다고 상상해봅시다. 추측으로 시작하고, 그림자 1과 일치하도록 조정하고, 그림자 2와 일치하도록 조정하고, 교대로 계속합니다. 각 조정이 진짜 형태에 가까워집니다. 게르히베르크-색스턴 알고리즘은 두 "그림자"가 두 다른 평면(물체와 푸리에)에서의 강도 패턴인 바로 이것을 합니다.

### 4.3 강도의 전파 방정식 (TIE)

비반복적 대안은 **강도의 전파 방정식(Transport of Intensity Equation, TIE)**입니다. 이는 강도의 축방향 도함수를 위상에 연결합니다:

$$-k\frac{\partial I}{\partial z} = \nabla_\perp \cdot (I \nabla_\perp \phi)$$

여기서 $I(x,y)$는 강도이고 $\phi(x,y)$는 주어진 평면에서의 위상입니다. 세 개의 근접한 평면($z - \delta z$, $z$, $z + \delta z$)에서 강도를 측정하고, 유한 차분으로 $\partial I/\partial z$를 추정하고, $\phi$에 대한 포아송형 방정식을 풉니다.

**장점**: 결정론적(반복 불필요), 단일 해(단순 연결 영역), 부분 결맞음 빛에 작동.
**한계**: 매우 정밀한 강도 측정 필요, 잡음에 민감, 작은 전파 거리 가정.

### 4.4 현대적 확장

- **혼합 입-출력(Hybrid Input-Output, HIO)**: 피엔업(Fienup)의 GS 수정으로, 물체 구속 조건에 피드백 매개변수 $\beta$를 적용하여 더 빠른 수렴을 제공
- **프티코그래피(Ptychography)**: 겹치는 스캔 회절 패턴이 중복 정보를 제공하여 위상 복원을 훨씬 더 강건하고 유일하게 결정
- **단일 촬영 위상 복원(Single-Shot Phase Retrieval)**: 구조화된 조명이나 코딩된 조리개를 사용하여 단일 강도 측정에서 위상 복원

---

## 5. 파면 감지

### 5.1 파면을 측정하는 이유

완벽한 광학 시스템은 평평한(또는 완벽하게 구면인) 파면을 생성합니다. 대기 난류, 제조 오차, 또는 열적 효과로 인한 수차가 파면을 왜곡하여 상질을 저하시킵니다. 파면 감지(wavefront sensing)는 이러한 왜곡을 측정하여 보정(적응 광학)하거나 특성화(광학 검사)합니다.

### 5.2 샥-하트만 파면 센서

**샥-하트만(Shack-Hartmann) 센서**는 가장 널리 사용되는 파면 센서입니다. 카메라 앞에 배치된 마이크로렌즈 배열로 구성됩니다. 각 마이크로렌즈는 들어오는 파면의 작은 부분을 샘플링하여 검출기의 스팟으로 집속합니다.

```
들어오는 파면 (수차 있음)
  ↓   ↓   ↓   ↓   ↓   ↓
┌───┬───┬───┬───┬───┬───┐  ← 마이크로렌즈 배열
│ · │ · │ · │ · │ · │ · │
│  ·│·  │ · │·  │  ·│ · │  ← 검출기의 초점 스팟
└───┴───┴───┴───┴───┴───┘
     스팟 변위 ∝ 국소 파면 기울기
```

평평한 파면의 경우 각 스팟이 부분 조리개(subaperture)의 중앙에 맺힙니다. 수차가 있는 파면의 경우 스팟이 국소 파면 기울기에 비례하여 변위됩니다:

$$\frac{\partial W}{\partial x} \approx \frac{\Delta x_{\text{spot}}}{f_{\text{lenslet}}}, \quad \frac{\partial W}{\partial y} \approx \frac{\Delta y_{\text{spot}}}{f_{\text{lenslet}}}$$

여기서 $W(x,y)$는 파면 오차이고 $f_{\text{lenslet}}$은 마이크로렌즈 초점 거리입니다. 파면 $W$는 측정된 기울기를 적분하여 재구성됩니다 — 포아송 방정식 $\nabla^2 W = \nabla \cdot (\text{측정된 기울기})$를 푸는 것과 동치인 문제입니다.

### 5.3 제르니케 다항식

원형 조리개에서 파면 수차를 기술하는 표준 기저는 **제르니케 다항식(Zernike Polynomials)** $Z_n^m(\rho, \theta)$입니다. 이들은 단위 원판에서 직교합니다:

$$\int_0^1\int_0^{2\pi} Z_n^m(\rho, \theta)\,Z_{n'}^{m'}(\rho, \theta)\,\rho\,d\rho\,d\theta = \frac{\pi}{2(n+1)}\delta_{nn'}\delta_{mm'}$$

각 제르니케 모드는 익숙한 수차에 해당합니다:

| 놀(Noll) 지수 $j$ | $n$ | $m$ | 이름 | 수식 |
|:---:|:---:|:---:|------|---------|
| 1 | 0 | 0 | 피스톤(Piston) | $1$ |
| 2 | 1 | 1 | 기울기(Tilt) (x) | $2\rho\cos\theta$ |
| 3 | 1 | -1 | 기울기(Tilt) (y) | $2\rho\sin\theta$ |
| 4 | 2 | 0 | 디포커스(Defocus) | $\sqrt{3}(2\rho^2 - 1)$ |
| 5 | 2 | -2 | 비점 수차 (사선) | $\sqrt{6}\,\rho^2\sin 2\theta$ |
| 6 | 2 | 2 | 비점 수차 (수직) | $\sqrt{6}\,\rho^2\cos 2\theta$ |
| 7 | 3 | -1 | 코마(Coma) (y) | $\sqrt{8}(3\rho^3 - 2\rho)\sin\theta$ |
| 8 | 3 | 1 | 코마(Coma) (x) | $\sqrt{8}(3\rho^3 - 2\rho)\cos\theta$ |
| 11 | 4 | 0 | 구면 수차(Spherical) | $\sqrt{5}(6\rho^4 - 6\rho^2 + 1)$ |

파면은 $W(\rho, \theta) = \sum_j a_j Z_j(\rho, \theta)$로 표현되며, 계수 $a_j$는 측정된 기울기 데이터에 맞춰 결정됩니다. 제곱 평균 제곱근(RMS) 파면 오차는 단순히 $\sigma_W = \sqrt{\sum_j a_j^2}$입니다 (피스톤 제외).

### 5.4 적응 광학

천문학적 적응 광학(AO)에서 파면 센서는 안내 별(자연적이거나 레이저로 생성된)에서 나오는 빛을 사용하여 대기 왜곡을 측정합니다. 변형 가능 거울(deformable mirror) — 표면을 밀고 당기는 작동기가 있는 — 이 실시간으로(일반적으로 500-1000 Hz에서) 파면을 보정합니다. 스트렐 비(Strehl Ratio, 최고 PSF 강도와 회절 한계 최고치의 비)가 AO 없는 ~0.01(시상 한계(seeing-limited))에서 >0.5(근회절 한계(near diffraction-limited))로 향상됩니다.

8-10 m급 망원경의 현대 AO 시스템은 근적외선 파장에서 일상적으로 회절 한계 이미징을 달성하며, 다중 켤레 AO(Multi-Conjugate AO, MCAO)는 여러 변형 가능 거울을 서로 다른 대기층에 켤레시켜 보정 시야를 확장합니다.

---

## 6. 전산 사진

### 6.1 단일 사진을 넘어서

전통적인 사진은 3D 장면의 단일 2D 투영을 포착합니다. **전산 사진(Computational Photography)**은 광학, 센서, 알고리즘의 조합을 사용하여 단일 통상적인 노출의 한계를 극복합니다 — 장면에 대한 더 많은 정보(깊이, 동적 범위, 스펙트럼 내용)를 포착하고, 어떤 단일 렌즈-검출기 시스템도 물리적으로 불가능한 이미지를 합성합니다.

### 6.2 라이트 필드 카메라

**라이트 필드 카메라(Light Field Camera)**(플레놉틱(plenoptic) 카메라)는 각 픽셀의 강도뿐만 아니라 빛이 도착하는 방향도 포착합니다. 이는 샥-하트만 센서와 유사하게 이미지 센서 앞에 마이크로렌즈 배열을 배치하여 달성됩니다.

라이트 필드 $L(x, y, u, v)$는 4D 함수입니다: $(x, y)$는 센서의 공간 위치이고, $(u, v)$는 각도 좌표(광선이 주 렌즈의 어느 부분을 통과했는지)입니다. 완전한 4D 라이트 필드로 다음이 가능합니다:

- **촬영 후 계산적 초점 조절**: 서로 다른 오프셋으로 부분 조리개 이미지를 이동-합산하여 서로 다른 깊이에 초점을 맞춤
- **시점 변경**: 약간 다른 원근에서 이미지 합성
- **깊이 추정**: 부분 조리개 이미지 사이의 시차로부터
- **피사계 심도 조절**: 조리개 크기를 디지털로 변경

라이트로(Lytro) 카메라(2012)는 최초의 상업적 라이트 필드 카메라였습니다. 계산적 초점 재조절 공식은:

$$I_\alpha(x, y) = \iint L\!\left(x + (1 - \alpha)u,\; y + (1 - \alpha)v,\; u, v\right)du\,dv$$

$\alpha = 1$: 마이크로렌즈 평면에 초점. $\alpha \neq 1$: 다른 깊이에 초점.

### 6.3 고동적범위 (HDR) 이미징

실세계 장면은 $10^6$:1을 초과하는 동적 범위를 가지지만, 카메라 센서는 일반적으로 $10^3$:1(12비트)만 포착합니다. **HDR 이미징(High Dynamic Range Imaging)**은 여러 노출을 합성하여 전체 동적 범위를 복원합니다:

$$\ln E(x,y) = \sum_j w(Z_j)\left[\ln Z_j - \ln \Delta t_j\right] \bigg/ \sum_j w(Z_j)$$

여기서 $Z_j$는 노출 시간 $\Delta t_j$에서의 픽셀 값이고, $w$는 잘 노출된 픽셀을 강조하는 가중 함수입니다. 복원된 복사 조도(radiance) 맵 $E$는 표준 모니터에 표시하기 위해 **톤 매핑(tone-mapping)**됩니다.

### 6.4 코딩된 조리개

**코딩된 조리개(Coded Aperture)**는 원형 렌즈 개구를 특별히 설계된 패턴(예: 무작위 이진 마스크 또는 광대역 최적화 패턴)으로 교체합니다. 이는 점 퍼짐 함수(Point Spread Function, PSF)를 제어된 방식으로 수정하여 다음을 가능하게 합니다:

- **확장된 피사계 심도(Extended Depth of Field)**: 3차 위상 마스크는 디포커스에 거의 불변인 PSF를 생성; 그 후 디컨벌루션으로 넓은 깊이 범위에서 선명한 이미지를 얻음
- **깊이 추정**: 디포커스 흐림은 깊이와 조리개 패턴 모두에 의존; 잘 설계된 패턴이 이 관계를 역변환 가능하게 만듦
- **모션 디블러(Motion Deblur)**: 플러터 셔터(노출 동안 무작위 개폐)가 광대역 시간적 코드를 생성하여 모션 블러를 역변환 가능하게 함

### 6.5 단일 픽셀 카메라

**단일 픽셀 카메라(Single-Pixel Camera)**(압축 감지(compressive sensing) 카메라)는 공간 광 변조기(Spatial Light Modulator, SLM)를 사용하여 장면에 무작위 패턴 시퀀스를 투영하고 각 패턴에 대해 단일 검출기로 총 강도를 측정합니다:

$$y_i = \sum_{j} \Phi_{ij} x_j$$

여기서 $x$는 장면(벡터화됨), $\Phi_i$는 $i$번째 측정 패턴, $y_i$는 측정된 강도입니다. $M \ll N$번의 측정($N$은 픽셀 수)으로 희소성 기반 최적화(압축 감지)를 사용하여 미결정(underdetermined) 시스템을 풀어 장면을 재구성할 수 있습니다:

$$\hat{x} = \arg\min_x \|\Psi x\|_1 \quad \text{subject to} \quad y = \Phi x$$

여기서 $\Psi$는 희소화 변환(sparsifying transform)(예: 웨이블릿)입니다. 이것은 검출기 배열이 비싼 파장(적외선, 테라헤르츠)에서 특히 강력합니다.

---

## 7. 디지털 홀로그래피

### 7.1 광학에서 디지털로

고전적 홀로그래피(레슨 11)에서는 홀로그램이 사진 필름에 기록되고 참조 빔으로 조명하여 광학적으로 재구성합니다. **디지털 홀로그래피(Digital Holography)**는 필름을 CCD/CMOS 센서로 교체하고 수치적으로 재구성합니다. 이는 정량적 위상 측정, 화학적 처리 없음, 디지털 신호 처리의 모든 강점을 가져옵니다.

### 7.2 기록

디지털 홀로그램은 디지털 센서에서 물체파 $U_o$와 참조파 $U_r$ 사이의 간섭 패턴으로 기록됩니다:

$$I(x, y) = |U_o + U_r|^2 = |U_r|^2 + |U_o|^2 + U_r^*U_o + U_rU_o^*$$

마지막 두 항이 홀로그래픽 정보를 담고 있습니다. **오프축(Off-Axis)** 기하에서 참조파가 기울어져 있어 이 항들이 푸리에 영역에서 분리되어 공간 필터링으로 격리할 수 있습니다.

### 7.3 수치 재구성: 각 스펙트럼법

홀로그램 평면으로부터 거리 $d$에서 물체파를 재구성하려면:

1. 푸리에 필터링(오프축) 또는 위상 스텝(온축)으로 $U_r^*U_o$ 항 분리
2. 디지털 참조파를 곱함: $U_h(x,y) = U_r^*U_o \cdot U_r = |U_r|^2 U_o \propto U_o$
3. 각 스펙트럼법으로 물체 평면으로 전파:

$$U_{\text{obj}}(x,y) = \mathcal{F}^{-1}\!\left\{\mathcal{F}\{U_h\} \cdot e^{ik_z d}\right\}$$

여기서 $k_z = \sqrt{k^2 - k_x^2 - k_y^2}$.

재구성된 복소 장은 **진폭** $|U_{\text{obj}}|$(강도 이미지)와 **위상** $\arg(U_{\text{obj}})$(정량적 위상 맵) 모두를 제공합니다.

### 7.4 위상 되풀이

측정된 위상 $\phi_{\text{wrapped}} = \arg(U)$는 구간 $(-\pi, \pi]$에 감겨 있습니다. 광경로 차이가 하나의 파장을 초과하는 샘플의 경우, 연속적인 위상 맵을 얻기 위해 $2\pi$ 도약을 해결하는 **위상 되풀이(Phase Unwrapping)**가 필요합니다.

일반적인 알고리즘:
- **경로 따르기(Path-Following)**: 이토(Itoh)의 방법 — 경로를 따라 위상 차이를 적분; 잔류(residue)(불일치 픽셀) 근처에서 실패
- **품질 유도(Quality-Guided)**: 고품질(저잡음) 영역을 먼저 되풀이
- **최소 자승(Least-Squares)**: $\sum|\nabla\phi_{\text{unwrapped}} - \text{감긴 그래디언트}|^2$ 최소화; 강건하지만 불연속성을 평활화
- **골드스타인(Goldstein) 알고리즘**: 잔류(되풀이 오차의 원천)를 식별하고 오차 전파를 방지하는 분기 절단(branch cut)을 배치

### 7.5 디지털 홀로그래피 응용

- **정량적 위상 이미징(Quantitative Phase Imaging, QPI)**: 염색 없이 세포 두께, 굴절률, 건조 질량 측정
- **디지털 홀로그래픽 현미경(Digital Holographic Microscopy, DHM)**: 수치적 초점 재조절로 단일 홀로그램에서 3D 재구성
- **진동 분석**: 시간 분해 홀로그램이 나노미터 정밀도로 표면 변형 드러냄
- **입자 추적**: 부피 내 미소 입자의 3D 위치

---

## 8. 광학에서의 머신러닝

### 8.1 딥러닝 혁명

머신러닝, 특히 심층 신경망은 ~2017년 이후 전산 광학에서 변혁적 도구로 부상했습니다. 신경망은 데이터로부터 복잡한 매핑을 학습할 수 있으며 — 해석적으로 표현하기 어렵거나 불가능한 매핑들을 학습합니다.

### 8.2 학습된 위상 복원

전통적인 위상 복원 알고리즘(GS, HIO)은 반복적이며 느리게 수렴하거나 국소 최솟값에 정체될 수 있습니다. 신경망은 단일 회절 패턴에서 직접 위상으로 매핑하는 것을 학습할 수 있습니다:

$$\phi_{\text{predicted}} = f_\theta(I_{\text{measured}})$$

여기서 $f_\theta$는 훈련된 합성곱 신경망(convolutional neural network)입니다. U-Net, ResNet, 물리 정보 신경망(Physics-Informed Networks, 파동 전파 연산자를 손실 함수에 통합하는)과 같은 아키텍처는 반복 방법보다 특히 잡음이 있는 데이터에서 더 빠르고 정확한 위상 복원을 달성합니다.

### 8.3 전산 이미징: 잡음 제거와 초해상도

**잡음 제거(Denoising)**: 잡음 있는 이미지와 깨끗한 이미지 쌍으로 훈련된 신경망이 특징을 보존하면서 잡음을 제거하는 것을 학습합니다. 형광 현미경에서 이는 10-100배 낮은 광량으로 이미징을 가능하게 하여 광독성(phototoxicity)을 줄입니다.

**초해상도(Super-Resolution)**: 신경망이 저해상도 측정에서 고해상도 이미지를 재구성하는 것을 학습하여, 특정 맥락에서 회절 한계를 초월합니다:
- 심층 CNN을 사용한 단일 이미지 초해상도(Single-Image Super-Resolution, SISR)
- 신경망을 사용한 구조화 조명 현미경(Structured Illumination Microscopy, SIM) 재구성
- 학습된 입자 검출을 이용한 국소화 현미경(PALM/STORM)

### 8.4 광학 소자의 역설계

아마도 가장 흥미로운 응용: 경사 기반 최적화(수반법(adjoint methods)) 또는 생성 모델을 사용하여 원하는 성능을 갖는 광학 구조를 **설계**하는 것:

- **메타표면 설계(Metasurface Design)**: 신경망이 나노구조 표면의 원시야 응답을 예측하여, 메타렌즈, 빔 분리기, 홀로그램의 신속한 역설계를 가능하게 함
- **위상 최적화(Topology Optimization)**: 광자 소자의 굴절률 분포에 대한 경사 하강법(FDTD 시뮬레이터를 수반법을 통해 미분) — 원하는 투과/반사 달성
- **회절 광학 신경망(Diffractive Optical Neural Networks, DONNs)**: 광속으로 신경망 추론을 물리적으로 수행하는 광학 소자 — 각 회절층의 위상 프로파일을 훈련하여 설계

### 8.5 물리 정보 신경망 (PINNs)

PINNs(Physics-Informed Neural Networks)는 물리 법칙(맥스웰 방정식, 파동 방정식)을 손실 함수에 통합합니다:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{physics}}$$

여기서 $\mathcal{L}_{\text{physics}}$는 지배 방정식의 위반에 페널티를 부여합니다. 이 정규화(regularization)는 필요한 훈련 데이터의 양을 크게 줄이고 물리적으로 그럴듯한 해를 보장합니다.

> **비유**: 전통적인 광학 설계는 경험과 경험 규칙(rule of thumb)에 의존하여 손으로 도면을 그리는 건축가와 같습니다. ML 기반 역설계는 수백만 개의 설계를 동시에 탐색하여 인간이 상상조차 하지 못할 구조들을 찾아내는 마법의 스케치패드를 건축가에게 주는 것과 같습니다 — 나노스케일 기둥으로 만들어진 평판 렌즈, 프랙탈 모양의 단면을 가진 도파로, 거의 완벽한 효율로 파장별로 빛을 분류하는 회절 소자. "마법"은 자동 미분에서 옵니다: 컴퓨터는 설계의 아주 작은 변화가 성능에 어떤 영향을 미치는지 계산하고 최적 방향으로 따라갈 수 있습니다.

---

## 9. Python 예제

### 9.1 순차 광선 추적기

```python
import numpy as np
import matplotlib.pyplot as plt

class Surface:
    """A spherical refracting surface in a sequential optical system."""
    def __init__(self, z_pos, radius, n_before, n_after):
        self.z = z_pos          # position along optical axis
        self.R = radius         # radius of curvature (inf for flat)
        self.n1 = n_before      # refractive index before surface
        self.n2 = n_after       # refractive index after surface

def trace_paraxial(y_in, u_in, surfaces):
    """
    Trace a paraxial ray through a sequence of surfaces using ABCD matrices.

    Why ABCD matrices? For paraxial rays, refraction and propagation are
    linear operations on (y, u). This lets us chain arbitrary sequences
    of surfaces as simple 2x2 matrix multiplications — far faster than
    solving Snell's law iteratively for each ray-surface intersection.

    Parameters
    ----------
    y_in : float — initial ray height above axis
    u_in : float — initial ray angle (radians, paraxial)
    surfaces : list of Surface objects in order of propagation

    Returns
    -------
    positions : list of (z, y) tuples for plotting the ray path
    """
    y, u = y_in, u_in
    positions = [(surfaces[0].z, y)]

    for i, surf in enumerate(surfaces):
        # Propagate from previous surface (or input) to this surface
        if i > 0:
            d = surf.z - surfaces[i-1].z
            # Free-space propagation: y' = y + d*u (paraxial, n=1 between lenses)
            y = y + d * u
        positions.append((surf.z, y))

        # Refract at this surface using the lensmaker's equation form
        # The ABCD refraction matrix has C = -(n2 - n1) / R
        if np.isinf(surf.R):
            power = 0.0
        else:
            power = (surf.n2 - surf.n1) / surf.R
        u = u - y * power  # paraxial Snell's law

    # Propagate to a final observation plane beyond the last surface
    d_final = 50.0
    y_final = y + d_final * u
    positions.append((surfaces[-1].z + d_final, y_final))

    return positions

# Define a simple two-lens system (doublet + singlet)
# This demonstrates how compound systems are built from elementary surfaces
surfaces = [
    Surface(z_pos=0,   radius=100.0,    n_before=1.0,  n_after=1.52),   # Front of lens 1
    Surface(z_pos=5,   radius=-100.0,   n_before=1.52, n_after=1.0),    # Back of lens 1
    Surface(z_pos=50,  radius=80.0,     n_before=1.0,  n_after=1.67),   # Front of lens 2
    Surface(z_pos=53,  radius=-200.0,   n_before=1.67, n_after=1.0),    # Back of lens 2
]

# Trace a fan of rays at different heights
fig, ax = plt.subplots(figsize=(12, 5))

heights = np.linspace(-15, 15, 11)
for y0 in heights:
    positions = trace_paraxial(y0, 0.0, surfaces)
    zs, ys = zip(*positions)
    ax.plot(zs, ys, 'b-', alpha=0.6, linewidth=0.8)

# Draw the lens surfaces as vertical lines
for surf in surfaces:
    ax.axvline(x=surf.z, color='gray', linestyle='--', alpha=0.5)
    ax.text(surf.z, 17, f'R={surf.R:.0f}', fontsize=7, ha='center')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Ray height y (mm)')
ax.set_title('Paraxial Ray Trace Through a Two-Lens System')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ray_trace.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 빔 전파법 (분리 단계)

```python
import numpy as np
import matplotlib.pyplot as plt

def bpm_split_step(psi0, x, z_array, n_profile, k0, n0):
    """
    Beam Propagation Method using the split-step Fourier algorithm.

    Why split-step? The paraxial wave equation has two terms: diffraction
    (diagonal in k-space) and refraction (diagonal in real space). By
    alternating between Fourier domain and real-space operations, we handle
    each term where it is simplest — this is both accurate and efficient (FFT
    gives O(N log N) cost per step instead of O(N^2) for direct convolution).

    Parameters
    ----------
    psi0 : 1D array — initial field envelope
    x : 1D array — transverse coordinate (m)
    z_array : 1D array — propagation positions (m)
    n_profile : callable — n_profile(x, z) returns refractive index array
    k0 : float — free-space wavenumber 2*pi/lambda
    n0 : float — background refractive index

    Returns
    -------
    field : 2D array (len(z_array), len(x)) — field at each z
    """
    dx = x[1] - x[0]
    N = len(x)
    k = k0 * n0  # reference wavenumber

    # Spatial frequency axis — must match FFT conventions
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Store field at each z position for visualization
    field = np.zeros((len(z_array), N), dtype=complex)
    field[0] = psi0
    psi = psi0.copy()

    for i in range(1, len(z_array)):
        dz = z_array[i] - z_array[i-1]
        z_mid = 0.5 * (z_array[i] + z_array[i-1])

        # Step 1: Diffraction in Fourier domain
        # The free-space propagator exp(-i*kx^2*dz/(2k)) spreads the beam;
        # each spatial frequency acquires a phase shift proportional to kx^2
        psi_k = np.fft.fft(psi)
        diffraction_phase = np.exp(-1j * kx**2 * dz / (2 * k))
        psi_k *= diffraction_phase
        psi = np.fft.ifft(psi_k)

        # Step 2: Refraction in real space
        # The local index variation dn bends the wavefront — this is where
        # the waveguide confinement enters the simulation
        n = n_profile(x, z_mid)
        dn = n - n0
        refraction_phase = np.exp(1j * k0 * dn * dz)
        psi *= refraction_phase

        field[i] = psi

    return field

# Example: Gaussian beam in a GRIN (graded-index) waveguide
# The parabolic index profile n(x) = n0 * sqrt(1 - alpha^2 * x^2)
# guides the beam by continuously bending rays toward the axis
lambda0 = 1.0e-6           # wavelength: 1 um
k0 = 2 * np.pi / lambda0
n0 = 1.5                   # core index
alpha = 200.0              # GRIN parameter (1/m)

# Transverse and longitudinal grids
x = np.linspace(-50e-6, 50e-6, 512)
z = np.linspace(0, 2e-3, 500)

# GRIN index profile: parabolic approximation
def grin_profile(x, z):
    return n0 * np.sqrt(np.maximum(1 - (alpha * x)**2, 0.5))

# Launch a Gaussian beam offset from the axis to excite oscillation
w0 = 10e-6  # beam waist
x_offset = 15e-6
psi0 = np.exp(-(x - x_offset)**2 / w0**2)

# Run the BPM simulation
field = bpm_split_step(psi0, x, z, grin_profile, k0, n0)

# Visualize the propagation — intensity shows the beam oscillating
# in the parabolic potential, analogous to a mass on a spring
fig, ax = plt.subplots(figsize=(12, 5))
extent = [z[0]*1e3, z[-1]*1e3, x[0]*1e6, x[-1]*1e6]
im = ax.imshow(np.abs(field.T)**2, aspect='auto', extent=extent,
               origin='lower', cmap='hot')
ax.set_xlabel('Propagation z (mm)')
ax.set_ylabel('Transverse x (um)')
ax.set_title('BPM: Gaussian Beam in a GRIN Waveguide')
plt.colorbar(im, label='Intensity |psi|^2')
plt.tight_layout()
plt.savefig('bpm_grin.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 게르히베르크-색스턴 위상 복원

```python
import numpy as np
import matplotlib.pyplot as plt

def gerchberg_saxton(target_amplitude, source_amplitude, n_iter=100):
    """
    Gerchberg-Saxton algorithm for phase retrieval.

    Why iterative? We know the amplitude in two planes (source and target)
    but not the phase in either. A single Fourier transform cannot recover
    phase from amplitude alone — the problem is underdetermined. By
    alternating between planes and enforcing the known amplitude at each,
    we progressively narrow down the consistent phase distribution.

    Parameters
    ----------
    target_amplitude : 2D array — desired amplitude in the Fourier plane
    source_amplitude : 2D array — known amplitude in the source plane
    n_iter : int — number of iterations

    Returns
    -------
    phase : 2D array — recovered phase in the source plane
    errors : list — RMS error at each iteration for convergence monitoring
    """
    # Initialize with random phase — the algorithm will iteratively
    # refine this guess using the two amplitude constraints
    phase = 2 * np.pi * np.random.random(source_amplitude.shape)
    errors = []

    for k in range(n_iter):
        # Source plane: combine known amplitude with current phase estimate
        u_source = source_amplitude * np.exp(1j * phase)

        # Propagate to Fourier plane (forward FFT with proper centering)
        u_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_source)))

        # Record error: how well does the Fourier amplitude match the target?
        error = np.sqrt(np.mean((np.abs(u_fourier) - target_amplitude)**2))
        errors.append(error)

        # Fourier constraint: replace amplitude, keep phase
        # This is the "projection onto the Fourier amplitude constraint set"
        fourier_phase = np.angle(u_fourier)
        u_fourier = target_amplitude * np.exp(1j * fourier_phase)

        # Propagate back to source plane (inverse FFT)
        u_source = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fourier)))

        # Source constraint: replace amplitude, keep phase
        phase = np.angle(u_source)

    return phase, errors

# Example: Design a phase mask that produces a ring pattern in the far field.
# This is a practical beam shaping problem — we want to redistribute a
# uniform laser beam into a ring, using only a phase-only spatial light
# modulator (SLM). We can control phase but NOT amplitude.
N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

# Target: ring pattern in Fourier plane
target_amp = np.exp(-((R - 0.3) / 0.05)**2)  # ring at radius 0.3
target_amp /= target_amp.max()

# Source: uniform amplitude (representing a flat-top laser beam)
source_amp = np.ones((N, N))
source_amp[R > 0.8] = 0  # circular aperture

# Run the GS algorithm
np.random.seed(42)
recovered_phase, errors = gerchberg_saxton(target_amp, source_amp, n_iter=200)

# Verify: propagate the recovered phase and check the far-field pattern
u_result = source_amp * np.exp(1j * recovered_phase)
far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_result)))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(source_amp, cmap='gray')
axes[0].set_title('Source Amplitude')

axes[1].imshow(recovered_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[1].set_title('Recovered Phase')

axes[2].imshow(np.abs(far_field)**2, cmap='hot')
axes[2].set_title('Reconstructed Far Field')

axes[3].semilogy(errors)
axes[3].set_xlabel('Iteration')
axes[3].set_ylabel('RMS Error')
axes[3].set_title('Convergence')
axes[3].grid(True, alpha=0.3)

for ax in axes[:3]:
    ax.axis('off')

plt.suptitle('Gerchberg-Saxton Phase Retrieval', fontsize=13)
plt.tight_layout()
plt.savefig('gs_phase_retrieval.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.4 제르니케 다항식

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def zernike_radial(n, m, rho):
    """
    Compute the radial component R_n^m(rho) of the Zernike polynomial.

    Why not just use a lookup table? The radial polynomial formula works
    for arbitrary (n, m) pairs, making this function general enough to
    compute any Zernike mode up to arbitrary order — essential for
    high-order adaptive optics with hundreds of modes.
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)

    R = np.zeros_like(rho, dtype=float)
    for s in range(int((n - m_abs) / 2) + 1):
        # Each term in the sum contributes a power of rho;
        # the alternating signs produce the oscillatory radial pattern
        coef = ((-1)**s * factorial(n - s)
                / (factorial(s)
                   * factorial(int((n + m_abs) / 2) - s)
                   * factorial(int((n - m_abs) / 2) - s)))
        R += coef * rho**(n - 2*s)
    return R

def zernike(n, m, rho, theta):
    """
    Compute Zernike polynomial Z_n^m(rho, theta) over the unit disk.

    The normalization follows the standard Noll convention so that
    each mode has unit RMS over the pupil — this means the Zernike
    coefficient a_j directly gives the RMS contribution of that mode
    to the total wavefront error.
    """
    # Normalization factor (Noll convention)
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))

    R = zernike_radial(n, m, rho)

    if m >= 0:
        Z = norm * R * np.cos(m * theta)
    else:
        Z = norm * R * np.sin(abs(m) * theta)

    # Mask outside the unit disk — Zernike polynomials are undefined there
    Z[rho > 1.0] = np.nan
    return Z

# Generate a grid over the unit disk
N = 300
x = np.linspace(-1.1, 1.1, N)
X, Y = np.meshgrid(x, x)
rho = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Plot the first 15 Zernike modes (Noll ordering)
# These are the modes most relevant to optical testing:
# low-order = alignment errors, mid-order = manufacturing, high-order = turbulence
noll_modes = [
    (0, 0, 'Piston'),
    (1, 1, 'Tilt X'), (1, -1, 'Tilt Y'),
    (2, 0, 'Defocus'), (2, -2, 'Astig 45'), (2, 2, 'Astig 0'),
    (3, -1, 'Coma Y'), (3, 1, 'Coma X'),
    (3, -3, 'Trefoil Y'), (3, 3, 'Trefoil X'),
    (4, 0, 'Spherical'), (4, 2, 'Sec Astig 0'),
    (4, -2, 'Sec Astig 45'), (4, 4, 'Tetrafoil 0'),
    (4, -4, 'Tetrafoil 45'),
]

fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.ravel()

for idx, (n, m, name) in enumerate(noll_modes):
    Z = zernike(n, m, rho, theta)
    im = axes[idx].imshow(Z, cmap='RdBu_r', extent=[-1.1, 1.1, -1.1, 1.1])
    axes[idx].set_title(f'Z({n},{m:+d})\n{name}', fontsize=9)
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle('Zernike Polynomials (First 15 Modes)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('zernike_modes.png', dpi=150, bbox_inches='tight')
plt.show()

# Example: Synthesize an aberrated wavefront and decompose it
# This simulates what a Shack-Hartmann sensor + Zernike fitting does:
# measure the wavefront, express it as a sum of Zernike modes, and
# identify which aberrations dominate
coefficients = {
    (2, 0): 0.5,    # defocus — the largest contributor
    (2, 2): 0.3,    # astigmatism
    (3, 1): -0.2,   # coma X
    (4, 0): 0.1,    # spherical aberration
}

W = np.zeros_like(rho)
for (n, m), a in coefficients.items():
    W += a * zernike(n, m, rho, theta)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(W, cmap='RdBu_r', extent=[-1.1, 1.1, -1.1, 1.1])
ax.set_title('Synthesized Aberrated Wavefront\n'
             r'($0.5\lambda$ defocus + $0.3\lambda$ astig + '
             r'$0.2\lambda$ coma + $0.1\lambda$ spherical)')
ax.axis('off')
plt.colorbar(im, label='Wavefront error (waves)')
plt.tight_layout()
plt.savefig('aberrated_wavefront.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. 요약

| 방법 | 영역 | 핵심 방정식 / 아이디어 | 전형적 응용 |
|--------|--------|---------------------|---------------------|
| 광선 추적 | 기하 광학 | 스넬의 법칙 + ABCD 행렬 | 렌즈 설계, 카메라 |
| BPM | 근축 파동 광학 | 분리 단계 푸리에: $\psi(z+\Delta z) = e^{i\hat{N}\Delta z}\mathcal{F}^{-1}\{e^{i\hat{D}\Delta z}\mathcal{F}\{\psi\}\}$ | 도파로, 광섬유 |
| FDTD | 완전파 맥스웰 | 이 격자 + 도약 시간 단계; CFL 안정성 | 광자 결정, 플라즈모닉스 |
| 위상 복원 | 역문제 | GS: 두 평면에서 교대 투영 | 파면 보정, CDI |
| TIE | 결정론적 위상 | $-k\,\partial I/\partial z = \nabla \cdot (I\nabla\phi)$ | 현미경, 전자 이미징 |
| 파면 감지 | 수차 측정 | 샥-하트만 스팟 → 제르니케 분해 | 적응 광학, 광학 검사 |
| 전산 사진 | 향상된 이미징 | 라이트 필드, HDR, 코딩된 조리개, 압축 감지 | 소비자 카메라, 현미경 |
| 디지털 홀로그래피 | 정량적 위상 | 각 스펙트럼 재구성 + 위상 되풀이 | DHM, 진동 분석 |
| 광학에서의 ML | 데이터 기반 | 신경망: $\phi = f_\theta(I)$ | 위상 복원, 역설계 |

**핵심 교훈**: 전산 광학은 단순한 광선 추적에서 알고리즘이 렌즈만큼 중요한 분야로 발전했습니다. 추세는 명확합니다 — 광학은 소프트웨어와 하드웨어가 어느 하나만으로는 달성할 수 없는 이미징 성능을 위해 공동 설계되는 전산 과학이 되고 있습니다.

---

## 11. 연습문제

### 연습문제 1: ABCD 광선 추적

$f_1 = 50\,\text{mm}$, $f_2 = 100\,\text{mm}$인 두 렌즈 광학 릴레이(4f 시스템)를 설계하라. (a) 시스템 ABCD 행렬을 작성하고 배율이 $M = -f_2/f_1 = -2$임을 검증하라. (b) 각도 $u = 0$에서 $-10$에서 $+10\,\text{mm}$까지의 높이를 가진 11개 광선의 팬을 추적하라. (c) 두 번째 렌즈 뒤 20 mm에 $f_3 = -30\,\text{mm}$인 세 번째 렌즈를 추가하고 새로운 광선 팬을 그려라. 초점에는 어떤 변화가 생기는가?

### 연습문제 2: BPM 도파로 모드

BPM 코드를 사용하여 코어 굴절률 $n_{\text{core}} = 1.50$, 클래딩 굴절률 $n_{\text{clad}} = 1.48$, 코어 반폭 $a = 5\,\mu\text{m}$, $\lambda = 1.55\,\mu\text{m}$의 계단 굴절률(step-index) 슬래브 도파로를 시뮬레이션하라. (a) 도파로 중심에 집중된 가우스 빔을 입사하고 5 mm 전파하라. 몇 개의 모드가 여기(excite)되는가? (b) V-수를 계산하고 유도 모드의 수를 해석적으로 예측하라. (c) 기울어진 가우스 빔을 입사하고 유도되지 않는 빛의 복사(radiation)를 관찰하라.

### 연습문제 3: 위상 복원 도전

복소 테스트 물체를 생성하라: $U = A(x,y)\exp[i\phi(x,y)]$, 여기서 $A$는 문자 "F"의 실루엣이고 $\phi$는 무작위 매끄러운 위상(저차 제르니케 다항식의 합)이다. (a) $I_{\text{obj}} = |U|^2$와 $I_{\text{far}} = |\mathcal{F}\{U\}|^2$를 계산하라. (b) 500번 반복으로 GS 알고리즘을 실행하고 수렴 곡선을 그려라. (c) 서로 다른 무작위 위상으로 시작해 보라 — 알고리즘이 항상 같은 해로 수렴하는가? (d) HIO 알고리즘(피엔업, $\beta = 0.9$)을 구현하고 수렴 속도를 비교하라.

### 연습문제 4: 제르니케 파면 분석

샥-하트만 센서가 망원경 주 거울에 대해 다음과 같은 제르니케 계수(단위: 파장)를 측정했다: $a_4 = 0.35$ (디포커스), $a_5 = -0.12$ (비점 수차), $a_7 = 0.08$ (코마), $a_{11} = 0.15$ (구면 수차). (a) 파면 오차 맵을 합성하고 그려라. (b) 총 RMS 파면 오차를 계산하라. (c) 마레샬(Marechal) 근사 $S \approx e^{-(2\pi\sigma)^2}$를 사용하여 스트렐 비를 추정하라. (d) 하나의 모드만 보정할 수 있다면, 어느 것이 스트렐 비를 가장 크게 향상시키는가?

### 연습문제 5: 디지털 홀로그래피 시뮬레이션

센서로부터 10 mm 떨어진 점 물체의 디지털 홀로그램을 생성하라 ($\lambda = 633\,\text{nm}$, 센서 크기 $5\,\text{mm} \times 5\,\text{mm}$, $1024 \times 1024$ 픽셀, $3^\circ$ 오프축 평면파 참조). (a) 홀로그램 강도 패턴을 계산하라. (b) 5, 10, 15 mm의 거리에서 각 스펙트럼법으로 물체를 재구성하라. (c) 어떤 거리에서 점 물체가 가장 잘 초점이 맞는가? (d) 홀로그램에 가우스 잡음(SNR = 20 dB)을 추가하고 재구성을 반복하라 — 잡음이 결과에 어떤 영향을 미치는가?

### 연습문제 6: 전산 방법 비교

$d = 2\lambda$의 두께와 $n = 2.0$의 굴절률을 가진 유전체 슬래브가 수직 입사 평면파로 조명될 때: (a) 파브리-페로(Fabry-Perot) 공식을 사용하여 투과율을 해석적으로 계산하라. (b) BPM을 사용하여 같은 구성을 시뮬레이션하라. (c) BPM이 근사적인 결과를 주는 이유를 논의하라 (어떤 물리를 놓치는가?). (d) 이 문제에서 BPM 대신 FDTD가 필요한 조건은 무엇인가?

---

## 12. 참고문헌

1. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W.H. Freeman. — 푸리에 광학과 전파 방법의 결정적 참고문헌 (3-4장: 각 스펙트럼, 5장: 결맞음 이미징).
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — 4장 (광선 광학 행렬), 7장 (빔 전파).
3. Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method* (3rd ed.). Artech House. — 표준 FDTD 참고문헌.
4. Fienup, J. R. (1982). "Phase retrieval algorithms: a comparison." *Applied Optics*, 21(15), 2758-2769. — GS, HIO, 오차 감소 알고리즘에 관한 고전적 논문.
5. Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." *JOSA*, 66(3), 207-211. — 파면 분석을 위한 표준 제르니케 인덱싱.
6. Kim, M. K. (2010). "Principles and techniques of digital holographic microscopy." *SPIE Reviews*, 1(1), 018005. — 디지털 홀로그래피의 포괄적 리뷰.
7. Rivenson, Y., Zhang, Y., Gunaydin, H., Teng, D., & Ozcan, A. (2018). "Phase recovery and holographic image reconstruction using deep learning in neural networks." *Light: Science & Applications*, 7, 17141. — 홀로그래픽 재구성을 위한 딥러닝에 관한 선구적 연구.
8. Molesky, S., et al. (2018). "Inverse design in nanophotonics." *Nature Photonics*, 12, 659-670. — 광자 구조의 전산 역설계 리뷰.

---

[← 이전: 13. 양자광학 입문](13_Quantum_Optics_Primer.md) | [다음: 15. 체르니케 다항식 →](15_Zernike_Polynomials.md)

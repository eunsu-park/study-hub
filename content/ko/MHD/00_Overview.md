# 자기유체역학 (MHD)

## 개요

이 토픽은 전기 전도성 유체와 자기장의 상호작용을 다루는 심화 자기유체역학을 다룹니다. Numerical Simulation(L17-L18)의 MHD 기초와 Plasma_Physics의 플라즈마 물리 기반 위에, 평형 이론, 안정성, 자기 재결합, 난류, 다이나모 작용, 천체물리/핵융합 응용을 포괄적인 계산 예제와 함께 탐구합니다.

## 선수 지식

- **Plasma_Physics** L04-L06 (하전 입자 운동, 드리프트, 단열 불변량)
- **Plasma_Physics** L13-L14 (이유체 모형, 운동론→MHD 유도)
- **Numerical_Simulation** L17-L18 (이상 MHD 방정식, 1D MHD 수치해법)
- **Mathematical_Methods** L05 (벡터 해석), L13 (PDE 방법)
- Python 중급 (NumPy, SciPy, Matplotlib)

## 레슨 계획

### 평형과 안정성 (L01-L04)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [01_MHD_Equilibria.md](./01_MHD_Equilibria.md) | ⭐⭐ | 힘 균형, Z-핀치, θ-핀치, Grad-Shafranov, 안전인자, 자속면 | 평형 이론 |
| [02_Linear_Stability.md](./02_Linear_Stability.md) | ⭐⭐⭐ | 선형화 MHD, 에너지 원리(δW), Kruskal-Shafranov, Suydam 기준 | 안정성 틀 |
| [03_Pressure_Driven_Instabilities.md](./03_Pressure_Driven_Instabilities.md) | ⭐⭐⭐ | Rayleigh-Taylor, Parker 불안정, interchange, ballooning, Mercier | 압력 구동 모드 |
| [04_Current_Driven_Instabilities.md](./04_Current_Driven_Instabilities.md) | ⭐⭐⭐ | kink(m=1), sausage(m=0), tearing 모드, NTM, 저항벽 모드 | 전류 구동 모드 |

### 자기 재결합 (L05-L07)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [05_Reconnection_Theory.md](./05_Reconnection_Theory.md) | ⭐⭐⭐⭐ | Sweet-Parker, Petschek, Hall MHD 재결합, X-point 기하학 | 이론 기초 |
| [06_Reconnection_Applications.md](./06_Reconnection_Applications.md) | ⭐⭐⭐⭐ | 태양 플레어, CME, 서브스톰, 톱니파 붕괴, 자기섬 합체 | 천체/핵융합 응용 |
| [07_Advanced_Reconnection.md](./07_Advanced_Reconnection.md) | ⭐⭐⭐⭐ | plasmoid 불안정, 난류 재결합, guide field, 상대론적 | 최신 주제 |

### MHD 난류와 다이나모 (L08-L10)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [08_MHD_Turbulence.md](./08_MHD_Turbulence.md) | ⭐⭐⭐⭐ | IK vs GS95 스펙트럼, Elsässer 변수, 임계 균형, 비등방성 | 난류 이론 |
| [09_Dynamo_Theory.md](./09_Dynamo_Theory.md) | ⭐⭐⭐⭐ | Cowling 정리, 평균장 이론, α-Ω 다이나모, 지구/태양 다이나모 | 자기장 생성 |
| [10_Turbulent_Dynamo.md](./10_Turbulent_Dynamo.md) | ⭐⭐⭐⭐ | 소규모(Kazantsev), 대규모 다이나모, 헬리시티, DNS/LES | 심화 다이나모 |

### 천체물리·핵융합 응용 (L11-L14)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [11_Solar_MHD.md](./11_Solar_MHD.md) | ⭐⭐⭐ | 자속관, 흑점, 태양 다이나모, 코로나 가열, Parker 풍 | 태양 물리 |
| [12_Accretion_Disk_MHD.md](./12_Accretion_Disk_MHD.md) | ⭐⭐⭐⭐ | MRI, 각운동량 수송, α-원반, 원반풍/제트 | 강착 물리 |
| [13_Fusion_MHD.md](./13_Fusion_MHD.md) | ⭐⭐⭐ | 토카막/스텔러레이터, 디스럽션, ELM, 톱니파, 베타 한계 | 핵융합 플라즈마 |
| [14_Space_Weather.md](./14_Space_Weather.md) | ⭐⭐⭐ | 자기권, Dungey 순환, 자기폭풍, CME 전파, GIC | 우주날씨 |

### 고급 계산 기법과 프로젝트 (L15-L18)

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [15_2D_MHD_Solver.md](./15_2D_MHD_Solver.md) | ⭐⭐⭐⭐ | 2D 유한체적, Constrained Transport, WENO, Orszag-Tang | 2D 솔버 |
| [16_Relativistic_MHD.md](./16_Relativistic_MHD.md) | ⭐⭐⭐⭐ | SRMHD, GRMHD 기초, 상대론적 제트, 블랙홀 강착 | 상대론적 영역 |
| [17_Spectral_Methods.md](./17_Spectral_Methods.md) | ⭐⭐⭐⭐ | 유사-스펙트럴, Chebyshev, MHD-PIC 하이브리드, AMR, SPH-MHD | 고급 기법 |
| [18_Projects.md](./18_Projects.md) | ⭐⭐⭐⭐ | 태양 플레어 시뮬, 디스럽션 예측, 구각 다이나모 | 종합 프로젝트 3개 |

## 추천 학습 경로

```
평형 및 안정성 (L01-L04)
         │
         ├──→ 재결합 (L05-L07)
         │           │
         │           ▼
         ├──→ 난류 및 다이나모 (L08-L10)
         │           │
         │           ▼
         ├──→ 응용 (L11-L14)
         │    태양, 강착, 핵융합, 우주날씨
         │           │
         └───────────┘
                     │
                     ▼
         고급 기법 및 프로젝트 (L15-L18)
         2D 솔버, 상대론, 스펙트럴, 프로젝트
```

### 집중 경로

| 경로 | 레슨 | 기간 |
|------|------|------|
| **핵융합 집중** | L01-L04 → L13 → L15 | 4주 |
| **천체물리 집중** | L01-L04 → L05-L07 → L08-L10 → L11-L12 → L15-L16 | 8주 |
| **계산 집중** | L01-L02 → L15 → L17 → L18 | 4주 |
| **전체 이수** | L01-L18 순서대로 | 12주 |

## 예제 코드

이 토픽의 예제 코드는 `examples/MHD/`에서 확인할 수 있습니다.

## 합계

- **18개 레슨** (평형/안정성 4 + 재결합 3 + 난류/다이나모 3 + 응용 4 + 고급/프로젝트 4)
- **난이도 범위**: ⭐⭐ ~ ⭐⭐⭐⭐
- **언어**: Python (주)
- **주요 라이브러리**: NumPy, SciPy, Matplotlib, Numba (2D 솔버)

## 참고 문헌

### 교과서
- J.P. Freidberg, *Ideal MHD* (Cambridge, 2014)
- D. Biskamp, *Nonlinear Magnetohydrodynamics* (Cambridge, 1993)
- E. Priest, *Magnetohydrodynamics of the Sun* (Cambridge, 2014)
- J.P. Goedbloed, R. Keppens, S. Poedts, *Magnetohydrodynamics of Laboratory and Astrophysical Plasmas* (Cambridge, 2019)

### 온라인
- NCAR HAO MHD 튜토리얼
- Athena++ 문서: https://www.athena-astro.app/

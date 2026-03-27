# 플라즈마 물리

## 개요

이 토픽은 물질의 네 번째 상태인 플라즈마의 기초 물리를 단일 입자 역학부터 운동론, 유체 기술까지 다룹니다. 기초 전자기학과 자기유체역학(MHD) 등 고급 주제 사이의 간극을 메우며, 핵융합, 우주, 천체물리 플라즈마 연구에 필요한 물리적 기반을 제공합니다.

## 선수 지식

- 벡터 해석 (Mathematical_Methods L05)
- 편미분방정식 (Mathematical_Methods L13)
- 기초 전자기학 (Maxwell 방정식, Lorentz 힘)
- Python 중급 (NumPy, SciPy, Matplotlib)

## 레슨 계획

### 기초

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [01_Introduction_to_Plasma.md](./01_Introduction_to_Plasma.md) | ⭐ | Debye 차폐, 플라즈마 주파수, 자이로주파수, 플라즈마 β, 준중성 | 개념적 기초 |
| [02_Coulomb_Collisions.md](./02_Coulomb_Collisions.md) | ⭐⭐ | 쿨롱 산란, 충돌 주파수, Spitzer 저항, 평균자유경로 | 충돌 레짐 |
| [03_Plasma_Description_Hierarchy.md](./03_Plasma_Description_Hierarchy.md) | ⭐⭐ | Klimontovich → Vlasov → 유체 계층, 모형 선택 기준 | 체계 개관 |

### 하전 입자 운동

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [04_Single_Particle_Motion_I.md](./04_Single_Particle_Motion_I.md) | ⭐⭐ | 자이로운동, Larmor 반경, E×B 드리프트, 안내중심 | 균일 자기장 |
| [05_Single_Particle_Motion_II.md](./05_Single_Particle_Motion_II.md) | ⭐⭐⭐ | ∇B 드리프트, 곡률 드리프트, 편극 드리프트, 일반 힘 드리프트 | 비균일 자기장 |
| [06_Magnetic_Mirrors_Adiabatic_Invariants.md](./06_Magnetic_Mirrors_Adiabatic_Invariants.md) | ⭐⭐⭐ | 자기거울, μ/J/Φ 불변량, 손실 원뿔, 바나나 궤도 | 포획 입자 |

### 운동론

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [07_Vlasov_Equation.md](./07_Vlasov_Equation.md) | ⭐⭐⭐ | 위상공간, 분포함수, Vlasov 방정식, BGK 모드 | 비충돌 운동론 |
| [08_Landau_Damping.md](./08_Landau_Damping.md) | ⭐⭐⭐⭐ | 란다우 컨투어, 파동-입자 공명, 역란다우 감쇠, 입자 포획 | 핵심 운동론 효과 |
| [09_Collisional_Kinetics.md](./09_Collisional_Kinetics.md) | ⭐⭐⭐⭐ | Fokker-Planck, Rosenbluth 포텐셜, Braginskii 수송, 신고전 | 충돌 효과 |

### 플라즈마 파동

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [10_Electrostatic_Waves.md](./10_Electrostatic_Waves.md) | ⭐⭐⭐ | Langmuir 파, 이온 음향파, Bernstein 모드 | 정전파 분산관계 |
| [11_Electromagnetic_Waves.md](./11_Electromagnetic_Waves.md) | ⭐⭐⭐ | R/L/O/X 모드, 휘슬러파, CMA 다이어그램, Faraday 회전 | 전자기파 전파 |
| [12_Wave_Heating_and_Instabilities.md](./12_Wave_Heating_and_Instabilities.md) | ⭐⭐⭐⭐ | ECRH, ICRH, 빔-플라즈마, Weibel, firehose, mirror 불안정 | 가열 및 안정성 |

### 유체 기술

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [13_Two_Fluid_Model.md](./13_Two_Fluid_Model.md) | ⭐⭐⭐ | 모멘트 방정식, 일반화 Ohm 법칙, Hall 효과, 반자성 드리프트 | MHD로의 다리 |
| [14_From_Kinetic_to_MHD.md](./14_From_Kinetic_to_MHD.md) | ⭐⭐⭐⭐ | CGL 모형, MHD 유효 조건, 드리프트/자이로운동론 개요 | 체계적 환원 |

### 응용 및 프로젝트

| 파일명 | 난이도 | 주요 내용 | 비고 |
|--------|--------|----------|------|
| [15_Plasma_Diagnostics.md](./15_Plasma_Diagnostics.md) | ⭐⭐⭐ | Langmuir 탐침, Thomson 산란, 간섭계, 분광 | 실험 방법 |
| [16_Projects.md](./16_Projects.md) | ⭐⭐⭐⭐ | 궤도 시뮬레이터, 분산관계 솔버, 1D Vlasov-Poisson 솔버 | 종합 프로젝트 3개 |

## 추천 학습 경로

```
기초 (L01-L03)                  하전 입자 운동 (L04-L06)
       │                                │
       ▼                                ▼
  플라즈마 매개변수            자이로운동, 드리프트, 거울
  충돌, 기술 모형             단열 불변량
       │                                │
       └────────────┬───────────────────┘
                    │
                    ▼
          운동론 (L07-L09)
          Vlasov, 란다우 감쇠
          Fokker-Planck, 수송
                    │
                    ▼
          플라즈마 파동 (L10-L12)
          정전파/전자기파, CMA
          가열, 불안정성
                    │
                    ▼
          유체 기술 (L13-L14)
          이유체, Ohm 법칙
          MHD 유도, 자이로운동론
                    │
            ┌───────┴───────┐
            ▼               ▼
    진단 (L15)          프로젝트 (L16)
    탐침, 산란          궤도 시뮬, Vlasov
            │
            ▼
    → MHD 토픽 (심화)
```

## 관련 토픽

- **Numerical_Simulation L17-L18**: MHD 기초 및 수치 해법 (MHD 토픽의 선수 과목, L13-L14의 보완)
- **Numerical_Simulation L19**: PIC 시뮬레이션 방법 (L04-L06의 계산적 보완)
- **Mathematical_Methods L05**: 벡터 해석 (전체에서 사용)
- **Mathematical_Methods L13**: PDE 방법 (파동 이론에서 사용)
- **MHD 토픽**: 심화 자기유체역학 (L04-L06, L13-L14 위에 구축)

## 예제 코드

이 토픽의 예제 코드는 `examples/Plasma_Physics/`에서 확인할 수 있습니다.

## 합계

- **16개 레슨** (기초 3 + 입자운동 3 + 운동론 3 + 파동 3 + 유체 2 + 응용/프로젝트 2)
- **난이도 범위**: ⭐ ~ ⭐⭐⭐⭐
- **언어**: Python (주)
- **주요 라이브러리**: NumPy, SciPy, Matplotlib, Numba (Vlasov 솔버 선택)

## 참고 문헌

### 교과서
- F.F. Chen, *Introduction to Plasma Physics and Controlled Fusion* (Vol. 1, 3rd ed.)
- R.J. Goldston & P.H. Rutherford, *Introduction to Plasma Physics*
- D.R. Nicholson, *Introduction to Plasma Theory*
- T.J.M. Boyd & J.J. Sanderson, *The Physics of Plasmas*
- J.A. Bittencourt, *Fundamentals of Plasma Physics*

### 온라인
- MIT OCW 22.611J: Introduction to Plasma Physics I
- Princeton Plasma Physics Laboratory 교육 자료

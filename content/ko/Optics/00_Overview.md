# 광학 (Optics)

## 개요

광학은 빛의 생성, 전파, 물질과의 상호작용을 연구하는 학문입니다. 빛이 직진한다는 고대의 관찰부터 현대의 양자광학과 포토닉스에 이르기까지, 광학은 안경과 카메라에서 광섬유 통신, 레이저, 홀로그래픽 디스플레이에 이르는 기술의 기반을 제공합니다. 이 토픽은 전자기학을 이미징, 통신, 포토닉스의 실용적 응용과 연결합니다.

## 선수 과목

- **전자기학(Electrodynamics)**: 맥스웰 방정식, 전자기파 (Electrodynamics L07-L11)
- **물리수학(Mathematical Methods)**: 푸리에 변환, 복소해석학 (Mathematical_Methods L06-L08)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **신호 처리(Signal Processing)** (선택): 푸리에 해석 배경 (Signal_Processing L03-L05)

## 학습 경로

```
기초 (L01-L04)
├── L01: 빛의 본질
├── L02: 기하광학 기초
├── L03: 거울과 렌즈
└── L04: 광학 기기

파동광학 (L05-L07)
├── L05: 파동광학 — 간섭
├── L06: 회절
└── L07: 편광

현대 및 고급 광학 (L08-L14)
├── L08: 레이저 기초
├── L09: 광섬유
├── L10: 푸리에 광학
├── L11: 홀로그래피
├── L12: 비선형 광학
├── L13: 양자광학 입문
└── L14: 전산 광학

응용 및 측정 광학 (L15-L17)
├── L15: 체르니케 다항식
├── L16: 적응광학
└── L17: 분광학
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [빛의 본질](01_Nature_of_Light.md) | 파동-입자 이중성, 전자기 스펙트럼, 굴절률, 분산 |
| 02 | [기하광학 기초](02_Geometric_Optics_Fundamentals.md) | 페르마의 원리, 스넬의 법칙, 전반사, 프리즘 |
| 03 | [거울과 렌즈](03_Mirrors_and_Lenses.md) | 거울/렌즈 방정식, 배율, 수차 |
| 04 | [광학 기기](04_Optical_Instruments.md) | 현미경, 망원경, 카메라, 인간의 눈, 레일리 기준 |
| 05 | [파동광학 — 간섭](05_Wave_Optics_Interference.md) | 영의 이중 슬릿, 박막 간섭, 마이컬슨 간섭계, 결맞음 |
| 06 | [회절](06_Diffraction.md) | 단일 슬릿, 에어리 패턴, 회절 격자, 프레넬 회절 |
| 07 | [편광](07_Polarization.md) | 말뤼스의 법칙, 파장판, 존스 행렬, 복굴절, 광학 활성 |
| 08 | [레이저 기초](08_Laser_Fundamentals.md) | 유도 방출, 레이저 유형, 가우시안 빔, ABCD 행렬 |
| 09 | [광섬유](09_Fiber_Optics.md) | 계단/구배 굴절률, NA, 분산, EDFA, WDM, 광섬유 브래그 격자 |
| 10 | [푸리에 광학](10_Fourier_Optics.md) | 각 스펙트럼, 렌즈의 FT, 4f 시스템, OTF/MTF/PSF, 위상 대비 |
| 11 | [홀로그래피](11_Holography.md) | 기록/재생, 체적 홀로그램, 디지털 홀로그래피 |
| 12 | [비선형 광학](12_Nonlinear_Optics.md) | SHG, 위상 정합, 커 효과, 4파 혼합, OPO |
| 13 | [양자광학 입문](13_Quantum_Optics_Primer.md) | 광자 상태, 압축광, 얽힘, QKD (BB84) |
| 14 | [전산 광학](14_Computational_Optics.md) | 광선 추적, BPM, 위상 복원, 전산 사진학 |
| 15 | [체르니케 다항식](15_Zernike_Polynomials.md) | 파면 분석, 놀 인덱싱, 직교성, 콜모고로프 난류 |
| 16 | [적응광학](16_Adaptive_Optics.md) | 샤크-하트만, 변형 거울, 폐루프 제어, 레이저 가이드 별 |
| 17 | [분광학](17_Spectroscopy.md) | 선 확대, 회절 격자, 파브리-페로, 비어-람베르트, 라만 |

## 관련 토픽

| 토픽 | 연결 |
|------|------|
| Electrodynamics | 빛 전파의 기초인 맥스웰 방정식 |
| Signal_Processing | 푸리에 해석, 공간 필터링, 전달 함수 |
| Computer_Vision | 이미지 형성, 렌즈 모델, 카메라 캘리브레이션 |
| Quantum_Computing | 광학 양자 컴퓨팅, 얽힌 광자 소스 |
| Mathematical_Methods | 푸리에 변환, 베셀 함수, 복소해석학 |
| Numerical_Simulation | 포토닉스 소자 시뮬레이션을 위한 FDTD 및 BPM |

## 예제 파일

`examples/Optics/`에 위치:

| 파일 | 설명 |
|------|------|
| `01_snells_law.py` | 스넬의 법칙 시각화, 전반사, 프리즘 분산 |
| `02_thin_lens.py` | 얇은 렌즈 광선 추적, 상 형성, 수차 |
| `03_interference.py` | 영의 이중 슬릿, 박막 코팅, 마이컬슨 간섭계 |
| `04_diffraction.py` | 단일 슬릿, 에어리 패턴, 회절 격자 |
| `05_polarization.py` | 존스 행렬, 말뤼스의 법칙, 파장판, 브루스터 각 |
| `06_gaussian_beam.py` | 가우시안 빔 전파, ABCD 행렬, 빔 품질 |
| `07_fiber_optics.py` | 광섬유 모드, 분산, 감쇠 버짓 |
| `08_fourier_optics.py` | 2D 푸리에 변환, 4f 필터링, PSF/OTF |
| `09_ray_tracing.py` | 순차 광선 추적, 스팟 다이어그램, 수차 해석 |
| `10_holography_sim.py` | 홀로그램 기록/재생, 위상 복원 |
| `11_zernike_polynomials.py` | 체르니케 모드 갤러리, 파면 피팅, 콜모고로프 위상 스크린 |
| `12_adaptive_optics.py` | 샤크-하트만 WFS, 변형 거울, 폐루프 적응광학 시뮬레이션 |
| `13_spectroscopy.py` | 선 프로파일(보이트), 격자 분광기, 파브리-페로, 비어-람베르트 |

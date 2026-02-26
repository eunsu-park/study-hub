# 전자기학 (Electrodynamics)

## 개요

전자기학(Electrodynamics)은 전자기장과 하전 물질의 상호작용을 연구하는 분야입니다. 쿨롱 법칙과 정자기학에서 출발하여, 정전하에서 빛의 전파까지 모든 고전 전자기 현상을 기술하는 통합 프레임워크인 맥스웰 방정식(Maxwell's equations)으로 귀결됩니다. 이 토픽은 기초 물리학과 안테나, 도파관, 전산 전자기학의 실용적 응용을 연결합니다.

## 선수 과목

- **물리수학(Mathematical Methods)**: 벡터 미적분, 복소 해석, 푸리에 변환 (Mathematical_Methods L05-L08)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **선형대수(Linear Algebra)**: 고유값, 행렬 연산 (Math_for_AI L01-L03)
- **신호 처리(Signal Processing)** (선택): 푸리에 해석 배경 (Signal_Processing L03-L05)

## 학습 경로

```
기초 (L01-L06)
├── L01: 정전기학 복습
├── L02: 전기 퍼텐셜과 에너지
├── L03: 도체와 유전체
├── L04: 정자기학
├── L05: 자기 벡터 퍼텐셜
└── L06: 전자기 유도

맥스웰 방정식 (L07-L11)
├── L07: 맥스웰 방정식 — 미분 형태
├── L08: 맥스웰 방정식 — 적분 형태
├── L09: 진공에서의 전자기파
├── L10: 물질에서의 전자기파
└── L11: 반사와 굴절

심화 주제 (L12-L18)
├── L12: 도파관과 공동
├── L13: 복사와 안테나
├── L14: 상대론적 전자기학
├── L15: 다중극 전개
├── L16: 전산 전자기학 (FDTD)
├── L17: 전자기 산란
└── L18: 응용 — 플라즈모닉스와 메타물질
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [정전기학 복습](01_Electrostatics_Review.md) | 쿨롱 법칙, 가우스 법칙, 전기장, 중첩 원리 |
| 02 | [전기 퍼텐셜과 에너지](02_Electric_Potential_and_Energy.md) | 스칼라 퍼텐셜, 푸아송/라플라스 방정식, 에너지 밀도 |
| 03 | [도체와 유전체](03_Conductors_and_Dielectrics.md) | 경계 조건, 분극, 전기용량, 유전 상수 |
| 04 | [정자기학](04_Magnetostatics.md) | 비오-사바르 법칙, 앙페르 법칙, 자기 쌍극자, 벡터 퍼텐셜 |
| 05 | [자기 벡터 퍼텐셜](05_Magnetic_Vector_Potential.md) | 게이지 자유도, 쿨롱 게이지, 다중극 전개 |
| 06 | [전자기 유도](06_Electromagnetic_Induction.md) | 패러데이 법칙, 렌츠 법칙, 인덕턴스, 상호 인덕턴스 |
| 07 | [맥스웰 방정식 — 미분 형태](07_Maxwells_Equations_Differential.md) | 변위 전류, 완전한 맥스웰 방정식, 파동 방정식 유도 |
| 08 | [맥스웰 방정식 — 적분 형태](08_Maxwells_Equations_Integral.md) | 스토크스/발산 정리, 보존 법칙, 포인팅 벡터 |
| 09 | [진공에서의 전자기파](09_EM_Waves_Vacuum.md) | 평면파, 편광, 에너지 수송, 자유 공간 임피던스 |
| 10 | [물질에서의 전자기파](10_EM_Waves_Matter.md) | 분산, 흡수, 복소 굴절률, 표피 깊이 |
| 11 | [반사와 굴절](11_Reflection_and_Refraction.md) | 프레넬 방정식, 브루스터 각, 전반사, 코팅 |
| 12 | [도파관과 공동](12_Waveguides_and_Cavities.md) | TE/TM 모드, 차단 주파수, 직사각형 도파관, 공진 공동 |
| 13 | [복사와 안테나](13_Radiation_and_Antennas.md) | 지연 퍼텐셜, 라모어 공식, 쌍극자 복사, 안테나 배열 |
| 14 | [상대론적 전자기학](14_Relativistic_Electrodynamics.md) | 장의 로렌츠 변환, 전자기 텐서, 공변 형식 |
| 15 | [다중극 전개](15_Multipole_Expansion.md) | 단극, 쌍극, 사중극, 구면 조화함수, 복사 패턴 |
| 16 | [전산 전자기학](16_Computational_Electrodynamics.md) | FDTD 방법, 이 격자, 흡수 경계(PML), 시뮬레이션 예제 |
| 17 | [전자기 산란](17_Electromagnetic_Scattering.md) | 레일리 산란, 미 이론, 보른 근사, 단면적 |
| 18 | [응용 — 플라즈모닉스와 메타물질](18_Plasmonics_and_Metamaterials.md) | 표면 플라즈몬, 음굴절, 클로킹, 광자 결정 |

## 관련 토픽

| 토픽 | 연관성 |
|------|--------|
| Mathematical_Methods | 벡터 미적분, 푸리에 변환, 그린 함수가 전반에 걸쳐 사용됨 |
| Plasma_Physics | 맥스웰 방정식이 플라즈마의 유체 방정식과 결합 |
| MHD | 도체 유체에서 전자기학의 저주파 극한 |
| Signal_Processing | 파동 전파, 푸리에 해석, 필터 설계와의 연결 |
| Numerical_Simulation | 장 계산을 위한 FDTD와 FEM 방법 |
| Computer_Vision | 영상 형성을 위한 광학 기초 |

## 예제 파일

`examples/Electrodynamics/`에 위치:

| 파일 | 설명 |
|------|------|
| `01_electrostatics.py` | 전기장 시각화, 가우스 법칙 검증 |
| `02_potential_laplace.py` | 라플라스 방정식 풀이, 등전위선 |
| `03_capacitor_sim.py` | 평행판 축전기, 유전체 효과 |
| `04_magnetostatics.py` | 비오-사바르 장 계산, 자기 쌍극자 |
| `05_faraday_induction.py` | 패러데이 법칙 시뮬레이션, 기전력 계산 |
| `06_maxwell_waves.py` | 평면파 전파, 포인팅 벡터 |
| `07_fresnel.py` | 프레넬 계수, 브루스터 각 그래프 |
| `08_waveguide_modes.py` | TE/TM 모드 패턴, 차단 주파수 |
| `09_dipole_radiation.py` | 진동 쌍극자 복사 패턴 |
| `10_fdtd_1d.py` | 1D FDTD 시뮬레이션, 흡수 경계 |
| `11_fdtd_2d.py` | 2D FDTD 시뮬레이션, 도파관, 산란 |
| `12_mie_scattering.py` | 미 이론 계산, 산란 단면적 |

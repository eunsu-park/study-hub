# 태양 물리학(Solar Physics)

## 개요

태양 물리학(Solar Physics)은 가장 가까운 항성인 태양을 핵 연소가 이루어지는 중심핵부터 태양권(heliosphere)의 외곽까지 연구하는 학문입니다. 공간적으로 분해된 세부 구조를 관측할 수 있는 유일한 항성으로서, 태양은 항성 천체물리학(stellar astrophysics), 플라즈마 물리학(plasma physics), 자기유체역학(magnetohydrodynamics)의 근본적인 실험실 역할을 합니다. 태양을 이해하는 것은 단순한 학문적 탐구가 아닙니다. 태양 활동은 인공위성 운용, 전력망, 통신, 유인 우주 비행에 영향을 미치는 우주 기상(space weather)을 일으키기 때문입니다.

이 주제는 태양의 내부 구조와 에너지 생성, 태양 내부를 밝혀내는 태양진동학(helioseismology) 기법, 광구(photosphere)에서 코로나(corona)까지 이어지는 태양 대기의 층상 구조, 활동 영역(active region)·플레어(flare)·코로나 질량 방출(coronal mass ejection)과 같은 자기 현상, 그리고 행성 간 공간을 채우는 태양풍(solar wind)을 다룹니다. 현대 관측 기법, 우주 기상 응용, 최신 탐사 임무도 살펴봅니다. 전 내용에 걸쳐 유체 역학(fluid dynamics), 복사 전달(radiative transfer), MHD 등의 물리 원리를 강조하여 가장 가까운 항성에 대한 일관된 그림을 제시합니다.

## 선수 지식

- **Plasma_Physics** (L04-L06, L13-L14): 단입자 운동, 유체 기술, 플라즈마파
- **MHD** (L05-L06, L09, L11): 자기 재연결(magnetic reconnection), 다이나모 이론, 태양 MHD 개요
- **Electrodynamics** (L04-L06, L10, L13): 정자기학, 전자기파, 복사
- **Mathematical_Methods** (L04-L06, L10): 벡터 해석, 푸리에 방법, ODE/PDE
- **Python**: NumPy, Matplotlib, SciPy (중급 수준)

## 학습 경로

```
태양 내부 (L01-L04)
├── L01: 태양 내부 — 구조, 정수압 평형, 에너지 전달
├── L02: 핵 에너지 생성 — pp 연쇄, CNO 순환, 중성미자
├── L03: 태양진동학 — 진동 모드, 역산, 내부 자전
└── L04: 광구 — 복사 전달, 주연 감광, 과립 구조, 분광선

태양 대기 (L05-L07)
├── L05: 채층과 전이 영역 — 스피큘, 네트워크, UV 방출
├── L06: 코로나 — 가열 문제, 코로나 루프, X선 방출
└── L07: 태양 자기장 — 자속관, 자력도, 힘-자유장

태양 활동 (L08-L12)
├── L08: 활동 영역과 흑점 — 구조, 진화, 자기 분류
├── L09: 태양 다이나모와 활동 주기 — 알파-오메가 다이나모, 나비 그림
├── L10: 태양 플레어 — 자기 재연결, 입자 가속, 방출
├── L11: 코로나 질량 방출 — 개시, 전파, 행성 간 CME
└── L12: 태양풍 — 파커 모델, 빠른/느린 태양풍, 태양권 구조

관측 및 응용 (L13-L16)
├── L13: 태양 분광학과 기기 — 분광 진단, 코로나그래프, EUV 영상
├── L14: 태양 고에너지 입자 — 가속, 수송, SEP 이벤트
├── L15: 우주 기상과 최신 임무 — 예측 모델, SDO, 파커 태양 탐사선, Solar Orbiter
└── L16: 종합 프로젝트 — 통합 모델링 및 분석 실습
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [태양 내부](01_Solar_Interior.md) | 정수압 평형, 복사/대류 에너지 전달, 타코클라인, 표준 태양 모델 |
| 02 | [핵 에너지 생성](02_Nuclear_Energy_Generation.md) | 열핵반응, 가모프 피크, pp 연쇄, CNO 순환, 태양 중성미자 문제 |
| 03 | [태양진동학](03_Helioseismology.md) | 태양 진동, p/g/f 모드, l-ν 다이어그램, 역산 기법, 내부 자전 |
| 04 | [광구](04_Photosphere.md) | 복사 전달, 주연 감광, 과립 구조, 초과립 구조, 분광선 형성 |
| 05 | [채층과 전이 영역](05_Chromosphere_and_TR.md) | 채층 구조, 스피큘, 전이 영역, UV/EUV 방출 |
| 06 | [코로나](06_Corona.md) | 코로나 가열 문제, 코로나 루프, X선/EUV 관측, 태양풍 기원 |
| 07 | [태양 자기장](07_Solar_Magnetic_Fields.md) | 자속관, 자력도, 포텐셜 및 힘-자유장 모델, 자기 나선도 |
| 08 | [활동 영역과 흑점](08_Active_Regions_and_Sunspots.md) | 흑점 구조, 반암부 미세 구조, 자기 분류, 활동 영역 진화 |
| 09 | [태양 다이나모와 활동 주기](09_Solar_Dynamo_and_Cycle.md) | 알파-오메가 다이나모, Babcock-Leighton 메커니즘, 나비 그림, 대극소기 |
| 10 | [태양 플레어](10_Solar_Flares.md) | 표준 플레어 모델, 자기 재연결, 입자 가속, 다파장 방출 |
| 11 | [코로나 질량 방출](11_CMEs.md) | CME 개시 메커니즘, 전파 모델, 행성 간 CME, 지자기 폭풍 |
| 12 | [태양풍](12_Solar_Wind.md) | 파커 나선, 빠른/느린 태양풍 원천, 태양권 전류 시트, 종단 충격 |
| 13 | [태양 분광학과 기기](13_Spectroscopy_Instruments.md) | 방출/흡수 진단, 코로나그래프, EUV/X선 영상기, 전파 관측 |
| 14 | [태양 고에너지 입자](14_SEPs.md) | 충격파 이벤트 vs 충동적 이벤트, 확산 충격 가속, 입자 수송 |
| 15 | [우주 기상과 최신 임무](15_Modern_Solar_Missions.md) | 예측 모델, SDO, 파커 태양 탐사선, Solar Orbiter, DKIST |
| 16 | [종합 프로젝트](16_Projects.md) | 통합 모델링: 태양 내부 모델, 플레어 분석, CME 전파, 태양진동학 파이프라인 |

## 연관 주제

| 주제 | 연관성 |
|------|--------|
| Plasma_Physics | 단입자 운동, Vlasov 방정식, 플라즈마파 — 코로나 및 태양권 물리의 기반 |
| MHD | 자기 재연결, 다이나모 이론, MHD 불안정성 — 플레어, CME, 태양풍에 필수적 |
| Electrodynamics | 전자기파 전파, 복사 이론 — 태양 전파 및 분광 관측의 기초 |
| Signal_Processing | 푸리에 해석, 스펙트럼 방법 — 태양진동학 및 시계열 분석의 핵심 도구 |
| Optics | 분광학, 회절, 편광 측정 — 기기 설계 및 관측 기법 |
| Numerical_Simulation | ODE/PDE 솔버, MHD 코드 — 전산 태양 물리학 |

## 예제 파일

`examples/Solar_Physics/` 에 위치:

| 파일 | 설명 |
|------|------|
| `01_solar_interior_model.py` | 정수압 평형 및 온도 분포의 수치 적분 |
| `02_nuclear_reactions.py` | 가모프 피크 계산, pp 연쇄 에너지 생성률, 중성미자 플럭스 |
| `03_helioseismology.py` | 구면 조화 분해, l-ν 다이어그램, 음파 광선 추적 |
| `04_radiative_transfer.py` | Eddington-Barbier 해, 주연 감광 곡선, 분광선 프로파일 |
| `05_chromosphere_tr.py` | 온도-고도 모델, DEM 분석, 전이 영역 방출 |
| `06_coronal_loop.py` | 정수압 코로나 루프 모델, 스케일링 법칙, 에너지 균형 |
| `07_magnetic_field.py` | PFSS(Potential Field Source Surface) 외삽, 자기 위상 구조 |
| `08_sunspot_model.py` | 흑점 냉각 모델, 윌슨 강하, 반암부 흐름 시뮬레이션 |
| `09_dynamo_model.py` | Babcock-Leighton 자속 수송 다이나모, 나비 그림 생성 |
| `10_flare_reconnection.py` | Sweet-Parker 및 Petschek 재연결률, 플레어 에너지 방출 |
| `11_cme_propagation.py` | 항력 기반 CME 전파 모델, 도달 시간 예측 |
| `12_parker_wind.py` | 파커 태양풍 방정식 솔버, 나선형 자기장, 태양풍 속도 프로파일 |

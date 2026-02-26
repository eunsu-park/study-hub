# 우주 기상(Space Weather)

## 개요

우주 기상(Space Weather)은 태양에서 행성 간 공간을 거쳐 지구의 자기권(magnetosphere), 전리권(ionosphere), 열권(thermosphere)에 이르는 우주 환경의 동적 상태를 다루며, 기술 시스템과 인간 활동에 영향을 미칩니다. 대기의 차등 태양 가열에 의해 구동되는 지상 날씨와 달리, 우주 기상은 태양의 자기 활동, 즉 태양 플레어(solar flare), 코로나 질량 방출(CME, coronal mass ejection), 끊임없이 흐르는 태양풍(solar wind)이 태양권(heliosphere)을 통해 전파되는 연쇄 교란을 일으키고 행성 자기장과 상호 작용함으로써 구동됩니다.

우주 기상을 이해하려면 태양 물리학(solar physics), 플라즈마 물리학(plasma physics), 자기유체역학(MHD, magnetohydrodynamics), 전기역학(electrodynamics) 분야의 지식을 통합해야 합니다. 현대 사회가 지자기 교란(geomagnetic disturbance)에 취약한 우주 및 지상 기술 인프라에 점점 더 의존하게 되면서, 이 분야는 학문적 호기심을 넘어 운영상의 필수 과제로 성장했습니다. 이 토픽은 지구에 태양풍이 도달하는 순간부터 자기권 역학(magnetospheric dynamics), 전리권 효과(ionospheric effect), 실용적 영향, 머신러닝 기법을 포함한 현대적 예보 방식에 이르는 전체 연쇄 과정을 추적합니다.

## 선수 과목

| 토픽 | 레슨 | 개념 |
|-------|---------|----------|
| Plasma_Physics | L04-L06, L10-L12 | 입자 운동, 단열 불변량, 플라즈마 파동, 운동 이론 |
| MHD | L05-L06, L14 | 자기 재연결, MHD 평형, 우주 기상 MHD 개요 |
| Electrodynamics | L04-L06, L10 | 자기정역학, 맥스웰 방정식, 물질 내 EM 파 |
| Mathematical_Methods | L04-L06, L10 | 벡터 해석, 푸리에 방법, 상미분방정식 |
| Python | — | NumPy, Matplotlib, SciPy (중급 수준) |

## 학습 경로

```
                        Space Weather Learning Path
                        ===========================

Block 1: Magnetosphere          Block 2: Geomagnetic Activity
(L01-L04)                       (L05-L07)
┌─────────────────────┐         ┌─────────────────────┐
│ L01 Introduction    │         │ L05 Geomagnetic     │
│ L02 Magnetosphere   │────────▶│     Storms          │
│     Structure       │         │ L06 Substorms       │
│ L03 Current Systems │         │ L07 Radiation Belts │
│ L04 SW-M Coupling   │         │                     │
└─────────────────────┘         └────────┬────────────┘
                                         │
                                         ▼
Block 3: Near-Earth Environment  Block 4: Impacts & Forecasting
(L08-L10)                       (L11-L16)
┌─────────────────────┐         ┌─────────────────────────┐
│ L08 Ionosphere      │         │ L11 GICs & Power Grids  │
│ L09 Thermosphere &  │────────▶│ L12 Satellite & Tech    │
│     Drag            │         │ L13 Geomagnetic Indices │
│ L10 SEP Events      │         │ L14 Forecasting Methods │
└─────────────────────┘         │ L15 AI/ML for SW       │
                                │ L16 Capstone Projects   │
                                └─────────────────────────┘
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|--------|-------------|
| 01 | [우주 기상 개론](./01_Introduction_to_Space_Weather.md) | 정의, 태양-지구 연결 연쇄, 역사적 사건, 사회경제적 영향 |
| 02 | [자기권 구조](./02_Magnetosphere_Structure.md) | 쌍극자장, 자기권계면, 활 충격파, 자기초, 플라즈마권, L-쉘 |
| 03 | [자기권 전류계](./03_Magnetospheric_Current_Systems.md) | Chapman-Ferraro 전류, 링 전류, 꼬리 전류, 버클랜드 전류, 전리권 전류 |
| 04 | [태양풍-자기권 결합](./04_Solar_Wind_Magnetosphere_Coupling.md) | 낮쪽 재연결, 결합 함수, 극모자 전위, 점성 상호 작용 |
| 05 | [지자기 폭풍](./05_Geomagnetic_Storms.md) | 폭풍 단계, Dst 발달, CME 유발 vs CIR 유발 폭풍, 링 전류 주입 |
| 06 | [부폭풍](./06_Magnetospheric_Substorms.md) | 성장/팽창/회복 단계, 전류 교란, 플라스모이드 형성 |
| 07 | [방사선대](./07_Radiation_Belts.md) | 내/외부 방사선대, 포획 입자 역학, 파동-입자 상호 작용, 슬롯 영역 |
| 08 | [전리권 우주 기상](./08_Ionosphere.md) | 전리권 층, 폭풍, 섬광, TEC 변화, GNSS 영향 |
| 09 | [열권과 위성 항력](./09_Thermosphere_and_Satellite_Drag.md) | 열권 가열, 밀도 증가, 항력 모델링, 궤도 예측 |
| 10 | [태양 에너지 입자 이벤트](./10_Solar_Energetic_Particle_Events.md) | SEP 가속, 수송, GLE 이벤트, 방사선 위험 |
| 11 | [지자기 유도 전류](./11_Geomagnetically_Induced_Currents.md) | GIC 물리학, 전력망 영향, 파이프라인 영향, 완화 전략 |
| 12 | [위성 및 기술 영향](./12_Technological_Impacts.md) | 표면/심층 대전, 단일 사건 효과, HF 차단, 항공 영향 |
| 13 | [지자기 지수](./13_Space_Weather_Indices.md) | Dst, Kp, AE, SYM-H, Ap, F10.7, NOAA 척도, 도출 및 해석 |
| 14 | [우주 기상 예보](./14_Forecasting_Models.md) | 경험적·물리 기반·앙상블 모델; ENLIL, WSA, 실시간 운영 |
| 15 | [우주 기상을 위한 AI/머신러닝](./15_AI_ML_for_Space_Weather.md) | Dst용 신경망, 태양 이미지용 CNN, 시계열용 LSTM, 전이 학습 |
| 16 | [종합 프로젝트](./16_Projects.md) | 종단 간 폭풍 분석, Dst 예측 파이프라인, 방사선대 모델링 |

## 관련 토픽

| 토픽 | 연관성 |
|-------|------------|
| Plasma_Physics | 기본 플라즈마 과정: 입자 운동, 파동, 운동 이론 |
| MHD | 자기 재연결, MHD 평형, 대규모 플라즈마 역학 |
| Electrodynamics | 맥스웰 방정식, EM 파 전파, 자기정역학 |
| Deep_Learning | 우주 기상 예측을 위한 신경망 아키텍처 |
| Machine_Learning | 지자기 지수 예보를 위한 고전적 ML 접근법 |
| Signal_Processing | 지자기 데이터의 시계열 분석, 스펙트럼 방법 |

## 예제 파일

| 파일 | 설명 |
|------|-------------|
| `dipole_field.py` | 지구 자기 쌍극자장 시각화 및 L-쉘 매핑 |
| `magnetopause_standoff.py` | 자기권계면 정지 거리 계산 (Shue 모델) |
| `ring_current_dst.py` | 링 전류 에너지 및 Dst 감소 (Dessler-Parker-Sckopke) |
| `coupling_functions.py` | Akasofu epsilon, Newell, Borovsky 결합 함수 |
| `burton_equation.py` | Burton 방정식 Dst 예측 모델 |
| `substorm_current_wedge.py` | 부폭풍 전류 웨지 자기 섭동 모델 |
| `radiation_belt_diffusion.py` | 방사선대 전자의 반경 방향 확산 방정식 |
| `ionospheric_conductivity.py` | 높이에 따른 Pedersen 및 Hall 전도율 |
| `gic_calculation.py` | 지전기장 및 네트워크 모델로부터 GIC 계산 |
| `geomagnetic_indices.py` | 자력계 데이터로부터 Kp, Dst, AE 지수 계산 |
| `dst_neural_network.py` | LSTM 기반 Dst 지수 예측 |
| `storm_analysis_pipeline.py` | 종단 간 지자기 폭풍 이벤트 분석 |

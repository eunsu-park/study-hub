# 양자 컴퓨팅(Quantum Computing)

## 개요

양자 컴퓨팅(Quantum Computing)은 중첩(Superposition), 얽힘(Entanglement), 간섭(Interference) 등 양자역학적 현상을 활용하여 고전 컴퓨터와 근본적으로 다른 방식으로 정보를 처리합니다. 고전 비트가 0 또는 1인 반면, 양자 비트(큐비트, Qubit)는 두 상태의 중첩으로 존재할 수 있어 특정 계산에서 지수적 속도 향상을 달성할 수 있습니다.

이 토픽은 이론적 기초(양자역학 입문, 큐비트 수학, 게이트 모델), 주요 알고리즘(Deutsch-Jozsa, Grover, Shor), 오류 정정, 현대 응용(변분 알고리즘, 양자 머신러닝)을 다룹니다. 예제는 Qiskit 스타일 의사코드와 NumPy 시뮬레이션을 사용합니다.

## 선수 지식

- **선형대수**: 복소 벡터 공간, 행렬 연산, 고유값 (Math_for_AI L01-L03)
- **확률론**: 기초 확률 이론 (Data_Science L11-L13)
- **Python**: NumPy 활용 능력 (Python L01-L08)
- **암호학** (선택): Shor 알고리즘 이해를 위한 RSA (Cryptography_Theory L05)

## 학습 경로

```
기초 (L01-L06)
├── L01: 양자역학 입문
├── L02: 큐비트와 블로흐 구
├── L03: 양자 게이트
├── L04: 양자 회로
├── L05: 얽힘과 벨 상태
└── L06: 양자 측정

알고리즘 (L07-L10)
├── L07: Deutsch-Jozsa 알고리즘
├── L08: Grover 탐색 알고리즘
├── L09: 양자 푸리에 변환
└── L10: Shor 소인수분해 알고리즘

심화 주제 (L11-L16)
├── L11: 양자 오류 정정
├── L12: 양자 텔레포테이션과 통신
├── L13: 변분 양자 고유값 풀이(VQE)
├── L14: QAOA와 조합 최적화
├── L15: 양자 머신러닝
└── L16: 양자 컴퓨팅 현황과 미래
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [양자역학 입문](01_Quantum_Mechanics_Primer.md) | 파동-입자 이중성, 중첩, 측정 공준, 디랙 표기법 |
| 02 | [큐비트와 블로흐 구](02_Qubits_and_Bloch_Sphere.md) | 단일 큐비트 상태, 블로흐 구 시각화, 다중 큐비트 시스템 |
| 03 | [양자 게이트](03_Quantum_Gates.md) | 파울리 게이트, 아다마르, 위상 게이트, CNOT, 범용 게이트 집합 |
| 04 | [양자 회로](04_Quantum_Circuits.md) | 회로 모델, 회로 깊이/너비, 행렬 시뮬레이션 |
| 05 | [얽힘과 벨 상태](05_Entanglement_and_Bell_States.md) | EPR 쌍, 벨 상태, CHSH 부등식, 비국소성 |
| 06 | [양자 측정](06_Quantum_Measurement.md) | 사영 측정, POVM, 측정 기저, 부분 측정 |
| 07 | [Deutsch-Jozsa 알고리즘](07_Deutsch_Jozsa_Algorithm.md) | 오라클 모델, 양자 병렬성, 최초의 지수적 속도향상 |
| 08 | [Grover 탐색 알고리즘](08_Grovers_Search.md) | 진폭 증폭, 오라클 구성, 이차 속도향상 |
| 09 | [양자 푸리에 변환](09_Quantum_Fourier_Transform.md) | QFT 회로, 위상 추정, 고전 FFT와의 연결 |
| 10 | [Shor 소인수분해 알고리즘](10_Shors_Algorithm.md) | 주기 찾기, 모듈러 거듭제곱, RSA에 대한 영향 |
| 11 | [양자 오류 정정](11_Quantum_Error_Correction.md) | 비트 반전/위상 반전 코드, Shor 코드, 안정자 형식, 표면 코드 |
| 12 | [양자 텔레포테이션과 통신](12_Quantum_Teleportation.md) | 텔레포테이션 프로토콜, 초밀집 부호화, 복제 불가 정리 |
| 13 | [변분 양자 고유값 풀이(VQE)](13_VQE.md) | 변분 원리, 안자츠 설계, 매개변수 최적화, 분자 시뮬레이션 |
| 14 | [QAOA와 조합 최적화](14_QAOA.md) | MaxCut, 혼합/비용 해밀토니안, 매개변수 지형 |
| 15 | [양자 머신러닝](15_Quantum_Machine_Learning.md) | 양자 특성 맵, 변분 분류기, 양자 커널, 불모 고원 |
| 16 | [양자 컴퓨팅 현황과 미래](16_Landscape_and_Future.md) | 하드웨어 플랫폼, 양자 우위, NISQ 시대, 내결함성 로드맵 |

## 관련 토픽

| 토픽 | 연결 |
|------|------|
| Math_for_AI | 선형대수 기초 (복소 벡터 공간, 유니터리 행렬) |
| Cryptography_Theory | Shor 알고리즘이 동기가 된 포스트 양자 암호 |
| Signal_Processing | QFT는 이산 푸리에 변환의 양자 버전 |
| Machine_Learning | 양자 ML은 양자 특성 공간으로 고전 ML을 확장 |
| Mathematical_Methods | 복소 해석, 선형대수가 전반에 사용됨 |

## 예제 파일

`examples/Quantum_Computing/`에 위치:

| 파일 | 설명 |
|------|------|
| `01_qubit_simulation.py` | 큐비트 상태 벡터, 블로흐 구, 측정 시뮬레이션 |
| `02_quantum_gates.py` | 게이트 행렬, 게이트 적용, 범용 게이트 분해 |
| `03_quantum_circuits.py` | 회로 빌더, 다중 큐비트 시뮬레이션, 얽힘 |
| `04_bell_states.py` | 벨 상태 준비, CHSH 게임 시뮬레이션 |
| `05_deutsch_jozsa.py` | Deutsch-Jozsa 오라클 및 알고리즘 구현 |
| `06_grovers_search.py` | Grover 알고리즘, 진폭 증폭 |
| `07_quantum_fourier.py` | QFT 회로, 위상 추정 |
| `08_shors_algorithm.py` | 주기 찾기, 소수 인수분해 |
| `09_error_correction.py` | 비트 반전 코드, Shor 9-큐비트 코드, 신드롬 측정 |
| `10_teleportation.py` | 양자 텔레포테이션 프로토콜 시뮬레이션 |
| `11_vqe.py` | H₂ 분자 바닥 상태 VQE |
| `12_qaoa.py` | MaxCut 문제 QAOA |
| `13_quantum_ml.py` | 양자 특성 맵, 변분 분류기 |

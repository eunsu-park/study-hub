# 제어 이론(Control Theory)

## 개요

이 토픽은 고전적인 전달 함수(transfer function) 기법부터 현대적인 상태 공간(state-space) 기법까지, 피드백 제어 시스템(feedback control system)의 해석과 설계를 다룹니다. 제어 이론은 시스템이 원하는 대로 동작하도록 만드는 수학적 프레임워크를 제공합니다. 즉, 기준 신호 추종, 외란(disturbance) 거부, 그리고 안정성(stability) 유지를 다룹니다.

## 선수 지식

- 미적분학 및 선형대수학 (행렬 연산, 고유값)
- 상미분 방정식(Ordinary Differential Equations) (Mathematical_Methods L09-L10)
- 라플라스 변환(Laplace Transforms) (Mathematical_Methods L15)
- 권장: LTI 시스템 및 주파수 개념 (Signal_Processing L02, L07)

## 학습 경로

### 파트 I: 기초 (L01-L03)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 01 | 제어 시스템 입문 | 개루프(open-loop) vs. 폐루프(closed-loop), 피드백(feedback), 블록 선도(block diagrams), 역사적 배경 |
| 02 | 물리 시스템의 수학적 모델링 | 스프링-질량-댐퍼(spring-mass-damper), DC 모터, 열 시스템, 선형화(linearization) |
| 03 | 전달 함수와 블록 선도 | 라플라스 영역 표현, 극점/영점(poles/zeros), 블록 선도 대수 |

### 파트 II: 고전 해석 (L04-L06)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 04 | 시간 영역 해석 | 계단/임펄스 응답, 2차 시스템 규격, 정상 상태 오차(steady-state error) |
| 05 | 안정도 해석 | 루스-후르비츠 판별법(Routh-Hurwitz criterion), BIBO 안정도, 상대 안정도 |
| 06 | 근궤적법(Root Locus Method) | 작도 규칙, 이득 선택, 근궤적 기반 설계 |

### 파트 III: 주파수 영역 기법 (L07-L08)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 07 | 주파수 응답 — 보드 선도(Bode Plots) | 크기/위상 선도, 점근선 근사, 시스템 식별 |
| 08 | 나이퀴스트 안정도와 견고성(Robustness) | 나이퀴스트 판별법(Nyquist criterion), 이득 여유/위상 여유, 감도 함수 |

### 파트 IV: 제어기 설계 (L09-L10)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 09 | PID 제어 | P/PI/PD/PID 동작, 지글러-니콜스 동조(Ziegler-Nichols tuning), 안티-와인드업(anti-windup), 실용 지침 |
| 10 | 보상기 설계 | 진상(lead), 지상(lag), 진상-지상(lead-lag) 보상기, 주파수 영역 설계 |

### 파트 V: 현대 제어 (L11-L14)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 11 | 상태 공간 표현(State-Space Representation) | 상태 변수, 상태 방정식, 전달 함수와의 상호 변환 |
| 12 | 상태 공간 해석 | 가제어성(controllability), 가관측성(observability), 정준형(canonical forms), 최소 실현(minimal realizations) |
| 13 | 상태 피드백과 관측기 설계 | 극점 배치(pole placement), 루엔버거 관측기(Luenberger observer), 분리 원리(separation principle) |
| 14 | 최적 제어(Optimal Control) | 선형 이차 조정기(LQR), 칼만 필터(Kalman filter), LQG |

### 파트 VI: 심화 주제 (L15-L16)

| 레슨 | 제목 | 핵심 개념 |
|------|------|-----------|
| 15 | 디지털 제어 시스템 | 샘플링, 영차 홀드(zero-order hold), z 영역 해석, 이산 PID |
| 16 | 비선형 제어와 고급 주제 | 선형화, 리아푸노프 안정도(Lyapunov stability), 위상 궤적(phase portraits), 모델 예측 제어(MPC) |

## 타 토픽과의 연계

- **Mathematical_Methods**: 상미분 방정식 풀이, 라플라스 변환, 위상 평면 해석 (L09-L10, L15)
- **Signal_Processing**: LTI 시스템, 주파수 응답, Z 변환 (L02, L07-L10)
- **Numerical_Simulation**: 제어 시스템 시뮬레이션을 위한 ODE 솔버 (L04-L06)
- **Reinforcement_Learning**: MDP/최적 제어 이중성(duality), 모델 기반 RL (L03, L13)
- **Math_for_AI**: 최적화 이론과 최적 제어의 연계 (L05-L07)

## 예제 코드

실행 가능한 Python 예제는 [`examples/Control_Theory/`](../../../examples/Control_Theory/)에 있습니다. 전달 함수 조작, 시뮬레이션, 제어기 설계를 위해 `numpy`, `scipy`, `matplotlib`, `control` 라이브러리를 사용합니다.

## 참고 문헌

- *Modern Control Engineering* by Katsuhiko Ogata (주요 참고서)
- *Modern Control Systems* by Richard C. Dorf and Robert H. Bishop
- *Feedback Control of Dynamic Systems* by Gene F. Franklin, J. David Powell, Abbas Emami-Naeini
- *Linear Systems Theory* by João P. Hespanha

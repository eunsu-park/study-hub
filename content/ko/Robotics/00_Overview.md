# 로보틱스 (Robotics)

## 개요

로보틱스는 물리 세계에서 인지하고, 추론하고, 행동할 수 있는 기계를 설계, 제작, 프로그래밍하기 위해 기계공학, 전기공학, 컴퓨터 과학을 통합하는 학제간 분야입니다. 이 토픽은 수학적 기초(강체 변환, 기구학, 동역학), 운동 계획과 제어, 인지와 상태 추정(센서, SLAM), 현대 소프트웨어 프레임워크(ROS2)를 다룹니다. 산업용 매니퓰레이터, 자율 이동 로봇, 협동 인간-로봇 시스템을 구축하든, 로보틱스 원리를 이해하면 계산과 물리적 행동 사이의 간극을 메울 수 있는 도구를 갖추게 됩니다.

## 선수 과목

- **물리수학(Mathematical Methods)**: 선형대수, 행렬 연산, 좌표 프레임 (Mathematical_Methods L03-L05)
- **제어이론(Control Theory)**: PID 제어, 상태공간 방법, 안정성 (Control_Theory L01-L10)
- **프로그래밍(Programming)**: 객체지향 프로그래밍, 자료구조 (Programming L05-L06)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **컴퓨터 비전(Computer Vision)** (선택): 영상 처리, 특징 검출 (Computer_Vision L01-L08)

## 학습 경로

```
기초 (L01-L04)
├── L01: 로보틱스 개요와 분류
├── L02: 강체 변환
├── L03: 순운동학 (DH 매개변수)
└── L04: 역운동학

동역학과 제어 (L05-L08)
├── L05: 속도 기구학과 야코비안
├── L06: 로봇 동역학 (오일러-라그랑주)
├── L07: 운동 계획 (RRT, PRM)
└── L08: 궤적 계획과 실행

인지와 추정 (L09-L12)
├── L09: 로봇 제어 (PID, 계산 토크)
├── L10: 센서와 인지
├── L11: 상태 추정과 필터링
└── L12: SLAM (동시 위치 추정 및 지도 작성)

현대 로보틱스 (L13-L16)
├── L13: ROS2 기초
├── L14: ROS2 내비게이션 스택
├── L15: 로보틱스를 위한 강화학습
└── L16: 다중 로봇 시스템과 군집
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [로보틱스 개요와 분류](01_Robotics_Overview.md) | 역사, 분류 (매니퓰레이터, 이동, 비행), 응용, 자유도, 작업 공간 |
| 02 | [강체 변환](02_Rigid_Body_Transformations.md) | 회전 행렬, 동차 변환, 오일러 각, 쿼터니언, 축-각도 |
| 03 | [순운동학](03_Forward_Kinematics.md) | DH 매개변수, 기구학 체인, 작업 공간 해석, 직렬 vs 병렬 |
| 04 | [역운동학](04_Inverse_Kinematics.md) | 해석적/수치적 IK, 야코비안 기반 방법, 특이점, 여유 자유도 |
| 05 | [속도 기구학과 야코비안](05_Velocity_Kinematics.md) | 매니퓰레이터 야코비안, 특이점 해석, 힘/토크 매핑, 조작성 |
| 06 | [로봇 동역학](06_Robot_Dynamics.md) | 오일러-라그랑주 공식, 관성 행렬, 코리올리/중력 항, 뉴턴-오일러 |
| 07 | [운동 계획](07_Motion_Planning.md) | 형상 공간, RRT/RRT*, PRM, 포텐셜 필드, 샘플링 기반 계획 |
| 08 | [궤적 계획과 실행](08_Trajectory_Planning.md) | 다항식 궤적, 스플라인, 최소 저크, 시간 최적, 작업 공간 계획 |
| 09 | [로봇 제어](09_Robot_Control.md) | 관절 공간 PID, 계산 토크, 임피던스 제어, 힘 제어, 하이브리드 |
| 10 | [센서와 인지](10_Sensors_and_Perception.md) | 인코더, IMU, LiDAR, 카메라, 깊이 센서, 센서 융합 기초 |
| 11 | [상태 추정과 필터링](11_State_Estimation.md) | 칼만 필터, EKF, 파티클 필터, 센서 융합, 위치 추정 |
| 12 | [SLAM](12_SLAM.md) | 그래프 기반 SLAM, EKF-SLAM, 파티클 필터 SLAM, 비주얼 SLAM, 루프 클로저 |
| 13 | [ROS2 기초](13_ROS2_Fundamentals.md) | 노드, 토픽, 서비스, 액션, 런치 파일, 매개변수 서버, 라이프사이클 |
| 14 | [ROS2 내비게이션 스택](14_ROS2_Navigation.md) | Nav2, 코스트맵, 경로 계획 플러그인, 복구 행동, 행동 트리 |
| 15 | [로보틱스를 위한 강화학습](15_RL_for_Robotics.md) | 시뮬레이션-실세계 전이, 보상 형성, 안전한 RL, 조작/보행 정책 |
| 16 | [다중 로봇 시스템과 군집](16_Multi_Robot_Systems.md) | 작업 할당, 대형 제어, 합의 알고리즘, 군집 지능 |

## 관련 토픽

| 토픽 | 연결 |
|------|------|
| Control_Theory | PID, 상태공간 제어, 로봇 제어기를 위한 안정성 해석 |
| Computer_Vision | 인지 파이프라인, 비주얼 SLAM, 파지를 위한 물체 검출 |
| Reinforcement_Learning | 조작과 보행을 위한 정책 학습 |
| Mathematical_Methods | 선형대수, 좌표 변환, 미분방정식 |
| Signal_Processing | 센서 신호 처리, 필터링 |
| Deep_Learning | 신경망 기반 인지 및 제어 정책 |

## 예제 파일

`examples/Robotics/`에 위치:

| 파일 | 설명 |
|------|------|
| `01_rigid_transforms.py` | 회전 행렬, 동차 변환, 쿼터니언 연산 |
| `02_forward_kinematics.py` | DH 매개변수 테이블, 직렬 매니퓰레이터 FK 계산 |
| `03_inverse_kinematics.py` | 해석적/수치적 IK 풀이, 야코비안 의사역행렬 |
| `04_jacobian.py` | 매니퓰레이터 야코비안 계산, 특이점 해석, 조작성 타원체 |
| `05_dynamics.py` | 2링크 평면 팔의 오일러-라그랑주 동역학 |
| `06_motion_planning.py` | 장애물 회피를 포함한 RRT 및 RRT* 경로 계획 |
| `07_trajectory.py` | 다항식 및 최소 저크 궤적 생성 |
| `08_pid_control.py` | 로봇 팔의 관절 공간 PID 제어 시뮬레이션 |
| `09_kalman_filter.py` | 로봇 위치 추정을 위한 확장 칼만 필터 |
| `10_slam.py` | 랜드마크 관측 기반 간단한 EKF-SLAM 구현 |
| `11_particle_filter.py` | 파티클 필터를 이용한 몬테카를로 위치 추정 |
| `12_swarm.py` | 다중 로봇 군집 시뮬레이션을 위한 레이놀즈 플로킹 모델 |

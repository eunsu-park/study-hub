# Study Hub

이 레포지터리는 제가 스스로 공부하기 위해 만든 자료를 정리한 것입니다.
일부 내용은 제가 직접 작성했으며, 일부는 AI가 검수하거나 AI의 도움을 받아 작성되었습니다.

This repository is a personal study project. Some content was written by me, some was reviewed by AI, and some was created with the help of AI.

55개 토픽, 4단계 난이도 체계, 한/영 이중언어를 지원하며, ~1,085개의 실행 가능한 예제와 ~858개의 연습문제 풀이를 포함합니다.

55 topics across 4 difficulty tiers, bilingual (EN/KR), with ~1,085 runnable examples and ~858 exercise solutions.

---

## 프로젝트 구조 / Project Structure

```
├── content/          # 학습 자료 Markdown / Study materials
│   ├── ko/           # 한국어 (55개 토픽) / Korean (55 topics)
│   └── en/           # 영어 (55개 토픽) / English (55 topics)
│
├── examples/         # 예제 코드 (55개 토픽, ~1,085개 파일) / Example code
│
├── exercises/        # 연습문제 풀이 (53개 토픽, ~858개 파일) / Exercise solutions
│
└── viewer/           # Flask 웹 뷰어 / Flask web viewer
```

## 빠른 시작 / Quick Start

```bash
# 웹 뷰어 실행 / Run web viewer
cd viewer
pip install -r requirements.txt
flask run --port 5050

# 브라우저에서 접속 / Access in browser
# http://127.0.0.1:5050
```

## 목차 / Table of Contents

### Tier 1 — 입문 (Beginner)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [Claude_Ecosystem](./content/ko/Claude_Ecosystem/00_Overview.md) | Claude Code, CLAUDE.md, Hooks, MCP, Agent SDK | 22 |
| [Cloud_Computing](./content/ko/Cloud_Computing/00_Overview.md) | AWS/GCP, EC2, Lambda, S3, VPC, Terraform | 17 |
| [Docker](./content/ko/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD, 컨테이너 네트워킹 | 12 |
| [Git](./content/ko/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 | 10 |
| [LaTeX](./content/ko/LaTeX/00_Overview.md) | 문서 조판, 수학 수식, TikZ, Beamer, 빌드 자동화 | 16 |
| [Linux](./content/ko/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 | 26 |
| [Programming](./content/ko/Programming/00_Overview.md) | 프로그래밍 개념, 패러다임, 디자인 패턴, 클린 코드, 테스팅 | 16 |
| [Python](./content/ko/Python/00_Overview.md) | 파이썬 고급 문법, 테스트, 패키징 | 16 |
| [Shell_Script](./content/ko/Shell_Script/00_Overview.md) | Bash 심화, 매개변수 확장, 프로세스 관리, 배포 자동화 | 16 |
| [Spanish](./content/ko/Spanish/00_Overview.md) | 스페인어 문법/어휘, 접속법, DELE/SIELE | 25 |
| [VIM](./content/ko/VIM/00_Overview.md) | 모달 편집, 모션, 매크로, 플러그인, Neovim/LSP | 14 |
| [Web_Development](./content/ko/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, SEO, Flask | 16 |

### Tier 2 — 중급 (Intermediate)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [C_Programming](./content/ko/C_Programming/00_Overview.md) | C 기초 ~ 임베디드, 디버깅, 포인터 심화, 네트워크, IPC, 크로스 플랫폼 | 24 |
| [Calculus_and_Differential_Equations](./content/ko/Calculus_and_Differential_Equations/00_Overview.md) | 미적분학, 다변수 미적분, ODE, PDE, 모델링 | 20 |
| [Compiler_Design](./content/ko/Compiler_Design/00_Overview.md) | 어휘/구문/의미 분석, IR, 코드 생성, 최적화, GC, LLVM | 16 |
| [Computer_Architecture](./content/ko/Computer_Architecture/00_Overview.md) | CPU, 캐시, 가상메모리, 병렬처리, RISC-V, 전력 관리 | 20 |
| [CPP](./content/ko/CPP/00_Overview.md) | Modern C++, 멀티스레딩, C++20, C++23, 디자인패턴, CMake | 22 |
| [Cryptography_Theory](./content/ko/Cryptography_Theory/00_Overview.md) | 정수론, 대칭/비대칭 암호, ECC, 전자서명, 양자내성, ZKP | 14 |
| [Data_Science](./content/ko/Data_Science/00_Overview.md) | NumPy, Pandas, 시각화, EDA, 확률, 추론, 베이지안, 시계열, 인과추론 | 29 |
| [Database_Theory](./content/ko/Database_Theory/00_Overview.md) | 관계형 모델, 정규화, 트랜잭션, 인덱싱, NoSQL, 분산 DB | 16 |
| [Formal_Languages](./content/ko/Formal_Languages/00_Overview.md) | DFA, NFA, 정규식, 문맥자유문법, PDA, 튜링머신, P/NP | 14 |
| [IoT_Embedded](./content/ko/IoT_Embedded/00_Overview.md) | Raspberry Pi, MQTT, Edge AI, Zigbee/Z-Wave, 센서 퓨전 | 14 |
| [Machine_Learning](./content/ko/Machine_Learning/00_Overview.md) | 회귀, 앙상블, SVM, 클러스터링, SHAP/LIME, AutoML, Production ML, A/B Testing | 23 |
| [Networking](./content/ko/Networking/00_Overview.md) | OSI/TCP-IP, 라우팅, 보안, IPv6, SDN, QoS, 멀티캐스트 | 22 |
| [OS_Theory](./content/ko/OS_Theory/00_Overview.md) | 프로세스, 스케줄링, 메모리, 파일시스템 | 18 |
| [PostgreSQL](./content/ko/PostgreSQL/00_Overview.md) | SQL, JSON, 복제, 파티셔닝, FTS, 보안/RLS | 20 |
| [Rust](./content/ko/Rust/00_Overview.md) | 소유권, 차용, 라이프타임, 트레이트, 동시성, async/await | 18 |
| [Security](./content/ko/Security/00_Overview.md) | CIA Triad, 암호학, TLS, OWASP, 인증/인가, 컨테이너 보안 | 16 |
| [Software_Engineering](./content/ko/Software_Engineering/00_Overview.md) | SDLC, 애자일, UML, 품질보증, DevOps, 유지보수 | 16 |
| [System_Design](./content/ko/System_Design/00_Overview.md) | 확장성, 분산시스템, 관측성, 검색 시스템 | 20 |

### Tier 3 — 고급 (Advanced)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [Algorithm](./content/ko/Algorithm/00_Overview.md) | 알고리즘/자료구조, 정렬, 그래프, DP, HLD, LCT, PST | 32 |
| [Backend_Frameworks](./content/ko/Backend_Frameworks/00_Overview.md) | FastAPI, Express, Django, API 설계, 인증, 배포, 관측성 | 18 |
| [Computer_Graphics](./content/ko/Computer_Graphics/00_Overview.md) | 그래픽스 파이프라인, 래스터화, 셰이딩, WebGL, 레이 트레이싱 | 16 |
| [Computer_Vision](./content/ko/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 3D비전, SLAM | 23 |
| [Control_Theory](./content/ko/Control_Theory/00_Overview.md) | 전달함수, PID, 상태공간, LQR, 칼만 필터, MPC | 16 |
| [Data_Engineering](./content/ko/Data_Engineering/00_Overview.md) | Airflow, Spark, Kafka, dbt, CDC, Lakehouse, Dagster | 21 |
| [Deep_Learning](./content/ko/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, Diffusion, Few-Shot, TTA | 44 |
| [Electrodynamics](./content/ko/Electrodynamics/00_Overview.md) | 정전기학, 자기정역학, 맥스웰 방정식, 전자기파, FDTD | 18 |
| [Foundation_Models](./content/ko/Foundation_Models/00_Overview.md) | Scaling Laws, LLaMA, DINOv2, PEFT, RAG | 25 |
| [Frontend_Frameworks](./content/ko/Frontend_Frameworks/00_Overview.md) | React, Vue, Svelte, TypeScript, SSR/SSG, 컴포넌트 패턴, 테스팅 | 18 |
| [GraphQL](./content/ko/GraphQL/00_Overview.md) | 스키마, 리졸버, DataLoader, Federation, Apollo, Strawberry | 16 |
| [LLM_and_NLP](./content/ko/LLM_and_NLP/00_Overview.md) | BERT, GPT, HuggingFace, RAG, LangChain, RLHF | 16 |
| [Math_for_AI](./content/ko/Math_for_AI/00_Overview.md) | 선형대수, 최적화, 확률, 정보이론, Transformer 수학 | 18 |
| [Mathematical_Methods](./content/ko/Mathematical_Methods/00_Overview.md) | 푸리에, ODE/PDE, 특수함수, 텐서, 그린함수, 변분법 | 18 |
| [MLOps](./content/ko/MLOps/00_Overview.md) | MLflow, Kubeflow, 모델 서빙, CI/CD, DVC, LLMOps | 17 |
| [Numerical_Simulation](./content/ko/Numerical_Simulation/00_Overview.md) | ODE/PDE, CFD, FDTD, MHD, FEM, GPU 가속, PINN | 24 |
| [Optics](./content/ko/Optics/00_Overview.md) | 기하광학, 간섭, 회절, 편광, 레이저, 홀로그래피, 적응광학 | 17 |
| [Quantum_Computing](./content/ko/Quantum_Computing/00_Overview.md) | 큐비트, 양자 게이트, Grover, Shor, VQE, QAOA | 16 |
| [Reinforcement_Learning](./content/ko/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, SAC, 커리큘럼, 계층적 RL | 16 |
| [Robotics](./content/ko/Robotics/00_Overview.md) | 운동학, 동역학, 경로 계획, SLAM, ROS2 | 16 |
| [Signal_Processing](./content/ko/Signal_Processing/00_Overview.md) | 푸리에 해석, 샘플링, FFT, Z변환, 디지털 필터, 적응 필터 | 16 |

### Tier 4 — 전문 (Expert)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [MHD](./content/ko/MHD/00_Overview.md) | MHD 평형, 안정성, 자기재결합, 난류, 다이나모, 핵융합 | 18 |
| [Plasma_Physics](./content/ko/Plasma_Physics/00_Overview.md) | 디바이 차폐, 블라소프 방정식, 란다우 감쇠, 플라즈마 파동 | 16 |
| [Solar_Physics](./content/ko/Solar_Physics/00_Overview.md) | 태양 내부, 핵에너지, 코로나, 자기장, 플레어, CME | 16 |
| [Space_Weather](./content/ko/Space_Weather/00_Overview.md) | 자기권, 지자기 폭풍, 방사선대, 전리층, GIC, 예보 모델 | 16 |

---

## 시작하기 / Getting Started

각 폴더의 `00_Overview.md` 파일에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

Check the `00_Overview.md` file in each folder for learning roadmaps and detailed table of contents.

### 추천 학습 순서 / Recommended Learning Path

#### 0단계: 기초 (공통 필수) / Stage 0: Foundation (Required for All)

```
Programming → Python → Git → Linux (01~05) → Shell_Script (01~05)
```

1. **프로그래밍 입문 / Programming Intro**: Programming (01~08) — 개념, 패러다임, OOP, 함수형
2. **Python 기초 / Python Basics**: Python (01~08) — 문법, 타입 힌트, 데코레이터
3. **버전 관리 / Version Control**: Git (01~05) — 브랜치, 머지, GitHub
4. **리눅스 기초 / Linux Basics**: Linux (01~05) — 명령어, 파일시스템, 권한
5. **셸 스크립팅 / Shell Scripting**: Shell_Script (01~05) — 변수, 조건문, 반복문, 함수

#### 1트랙: 시스템 프로그래밍 / Track 1: Systems Programming

```
C_Programming → CPP → Rust
```

6. **C 언어**: C_Programming (01~12) — 포인터, 메모리, 시스템 호출
7. **C++ 학습**: CPP (01~12) — 클래스, STL, 스마트 포인터, 템플릿
8. **Rust**: Rust (01~12) — 소유권, 차용, 라이프타임, 트레이트, 동시성

#### 2트랙: 웹 풀스택 / Track 2: Full-Stack Web

```
Web_Development → Frontend_Frameworks → Backend_Frameworks → GraphQL → Docker → Cloud_Computing
```

9. **웹 기초 / Web Basics**: Web_Development (01~09) — HTML, CSS, JavaScript, TypeScript
10. **프론트엔드 / Frontend**: Frontend_Frameworks (01~10) — React, Vue, Svelte
11. **백엔드 / Backend**: Backend_Frameworks (01~12) — FastAPI, Express, Django
12. **GraphQL**: GraphQL (01~10) — 스키마, 리졸버, Apollo, Strawberry
13. **컨테이너 / Containers**: Docker (01~08) — Docker, Compose, Kubernetes
14. **클라우드 / Cloud**: Cloud_Computing (01~10) — AWS/GCP, Terraform

#### 3트랙: 데이터 & 인프라 / Track 3: Data & Infrastructure

```
PostgreSQL → Data_Science → Data_Engineering
Linux (심화) → Networking → System_Design → Security
```

15. **데이터베이스 / Database**: PostgreSQL (01~10) — SQL, 인덱싱, 트랜잭션
16. **데이터 과학 / Data Science**: Data_Science (01~15) — NumPy, Pandas, 통계, 시각화
17. **데이터 엔지니어링 / Data Engineering**: Data_Engineering (01~12) — Airflow, Spark, Kafka
18. **리눅스 심화 / Linux Advanced**: Linux (06~20) — 시스템 관리, 네트워크, 보안
19. **네트워킹 / Networking**: Networking (01~12) — TCP/IP, 라우팅, DNS, 방화벽
20. **시스템 설계 / System Design**: System_Design (01~12) — 확장성, 캐싱, 샤딩
21. **보안 / Security**: Security (01~10) — TLS, OWASP, 인증/인가

#### 4트랙: CS 이론 / Track 4: CS Theory

```
Algorithm → Formal_Languages
OS_Theory → Computer_Architecture → Compiler_Design
Software_Engineering → Database_Theory → Cryptography_Theory
```

22. **알고리즘 / Algorithms**: Algorithm (01~15) — 정렬, 그래프, DP, 세그먼트 트리
23. **형식 언어 / Formal Languages**: Formal_Languages (01~10) — DFA, NFA, CFG, 튜링 머신
24. **운영체제 / OS Theory**: OS_Theory (01~10) — 프로세스, 스케줄링, 메모리, 파일시스템
25. **컴퓨터 구조 / Computer Architecture**: Computer_Architecture (01~12) — CPU, 캐시, 파이프라인
26. **컴파일러 / Compiler Design**: Compiler_Design (01~10) — 렉서, 파서, IR, 코드 생성
27. **소프트웨어 공학 / Software Engineering**: Software_Engineering (01~10) — SDLC, 애자일, UML
28. **데이터베이스 이론 / Database Theory**: Database_Theory (01~10) — 정규화, MVCC, 분산 DB
29. **암호학 / Cryptography**: Cryptography_Theory (01~10) — AES, RSA, ECC, ZKP

#### 5트랙: AI & 머신러닝 / Track 5: AI & Machine Learning

```
Math_for_AI → Machine_Learning → Deep_Learning → LLM_and_NLP → Foundation_Models
├─ 응용: Computer_Vision, Reinforcement_Learning, MLOps
└─ 수학 심화: Calculus, Mathematical_Methods, Numerical_Simulation
```

30. **AI 수학 / Math for AI**: Math_for_AI (01~10) — 선형대수, 최적화, 확률
31. **머신러닝 / Machine Learning**: Machine_Learning (01~15) — 회귀, 앙상블, SVM, 클러스터링
32. **딥러닝 / Deep Learning**: Deep_Learning (01~20) — PyTorch, CNN, RNN, Transformer
33. **LLM & NLP**: LLM_and_NLP (01~10) — BERT, GPT, RAG, LangChain
34. **파운데이션 모델 / Foundation Models**: Foundation_Models (01~12) — Scaling Laws, PEFT, RAG
35. **컴퓨터 비전 / Computer Vision**: Computer_Vision (01~12) — OpenCV, 객체 검출, SLAM
36. **강화학습 / Reinforcement Learning**: Reinforcement_Learning (01~10) — DQN, PPO, SAC
37. **MLOps**: MLOps (01~10) — MLflow, 모델 서빙, CI/CD for ML
38. **미적분학 / Calculus**: Calculus_and_Differential_Equations (01~12) — 미분, 적분, ODE, PDE
39. **물리수학 / Mathematical Methods**: Mathematical_Methods (01~10) — 푸리에, 특수함수, 텐서
40. **수치 시뮬레이션 / Numerical Simulation**: Numerical_Simulation (01~12) — CFD, FEM, PINN

#### 6트랙: 물리 & 공학 / Track 6: Physics & Engineering

```
Electrodynamics → Optics → Signal_Processing → Control_Theory
├─ 우주물리: Plasma_Physics → MHD → Solar_Physics → Space_Weather
└─ 응용공학: Robotics, IoT_Embedded, Quantum_Computing, Computer_Graphics
```

41. **전자기학 / Electrodynamics**: Electrodynamics (01~10) — 맥스웰 방정식, 전자기파
42. **광학 / Optics**: Optics (01~10) — 기하광학, 간섭, 회절, 레이저
43. **신호처리 / Signal Processing**: Signal_Processing (01~10) — 푸리에, FFT, 디지털 필터
44. **제어이론 / Control Theory**: Control_Theory (01~10) — PID, 상태공간, LQR, 칼만 필터
45. **로보틱스 / Robotics**: Robotics (01~10) — 운동학, 경로 계획, SLAM, ROS2
46. **IoT / IoT_Embedded**: IoT_Embedded (01~08) — Raspberry Pi, MQTT, Edge AI
47. **양자 컴퓨팅 / Quantum Computing**: Quantum_Computing (01~10) — 큐비트, Grover, Shor, VQE
48. **컴퓨터 그래픽스 / Computer Graphics**: Computer_Graphics (01~10) — 래스터화, 셰이딩, 레이 트레이싱
49. **플라즈마 물리 / Plasma Physics**: Plasma_Physics (01~10) — 디바이 차폐, 란다우 감쇠
50. **MHD**: MHD (01~10) — MHD 평형, 자기재결합, 다이나모
51. **태양 물리 / Solar Physics**: Solar_Physics (01~10) — 태양 내부, 코로나, 플레어, CME
52. **우주 기상 / Space Weather**: Space_Weather (01~10) — 자기권, 방사선대, GIC, 예보

---

## 다국어 지원 / Multilingual Support

- 모든 학습 자료는 **한국어(ko)**와 **영어(en)** 두 버전으로 관리
  All study materials are maintained in both **Korean (ko)** and **English (en)**
- 55개 토픽 모두 한/영 번역 완료
  All 55 topics fully translated to Korean/English

## 라이센스 / License

이 프로젝트는 이중 라이센스를 적용합니다:
This project uses a dual license:

| 대상 / Target | 라이센스 / License |
|---|---|
| 학습 자료 (`content/`) / Study materials | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| 코드 (`examples/`, `exercises/`, `viewer/`) / Code | [MIT License](./LICENSE) |

### AI 도움 명시 / AI Assistance Disclosure

이 저장소의 콘텐츠는 부분적으로 AI(Claude)의 도움을 받아 작성되었습니다.
Some content in this repository was created with the assistance of AI (Claude).

- 일부 학습 자료는 AI가 생성하고 저자가 검토함
  Some study materials were generated by AI and reviewed by the author
- 일부는 저자가 작성하고 AI가 검토함
  Some were written by the author and reviewed by AI

## Author

**Eunsu Park**
- [ORCID: 0000-0003-0969-286X](https://orcid.org/0000-0003-0969-286X)

# Study Hub

이 레포지터리는 제가 스스로 공부하기 위해 만든 자료를 정리한 것입니다.
일부 내용은 제가 직접 작성했으며, 일부는 AI가 검수하거나 AI의 도움을 받아 작성되었습니다.
This repository is a personal study project. Some content was written by me, some was reviewed by AI, and some was created with the help of AI.

51개 토픽, 4단계 난이도 체계, 한/영 이중언어를 지원하며, ~1,080개의 실행 가능한 예제와 ~820개의 연습문제 풀이를 포함합니다.
51 topics across 4 difficulty tiers, bilingual (EN/KR), with ~1,080 runnable examples and ~820 exercise solutions.

---

## 프로젝트 구조 / Project Structure

```
├── content/          # 학습 자료 Markdown / Study materials
│   ├── ko/           # 한국어 (51개 토픽) / Korean (51 topics)
│   └── en/           # 영어 (51개 토픽) / English (51 topics)
│
├── examples/         # 예제 코드 (51개 토픽, ~1,080개 파일) / Example code
│
├── exercises/        # 연습문제 풀이 (49개 토픽, ~820개 파일) / Exercise solutions
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
| [Programming](./content/ko/Programming/00_Overview.md) | 프로그래밍 개념, 패러다임, 디자인 패턴, 클린 코드, 테스팅 | 16 |
| [Python](./content/ko/Python/00_Overview.md) | 파이썬 고급 문법, 테스트, 패키징 | 16 |
| [Git](./content/ko/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 | 10 |
| [Shell_Script](./content/ko/Shell_Script/00_Overview.md) | Bash 심화, 매개변수 확장, 프로세스 관리, 배포 자동화 | 16 |
| [Linux](./content/ko/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 | 26 |
| [Web_Development](./content/ko/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, SEO, Flask | 16 |
| [Docker](./content/ko/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD, 컨테이너 네트워킹 | 12 |
| [LaTeX](./content/ko/LaTeX/00_Overview.md) | 문서 조판, 수학 수식, TikZ, Beamer, 빌드 자동화 | 16 |
| [Cloud_Computing](./content/ko/Cloud_Computing/00_Overview.md) | AWS/GCP, EC2, Lambda, S3, VPC, Terraform | 17 |
| [Claude_Ecosystem](./content/ko/Claude_Ecosystem/00_Overview.md) | Claude Code, CLAUDE.md, Hooks, MCP, Agent SDK | 22 |
| [VIM](./content/ko/VIM/00_Overview.md) | 모달 편집, 모션, 매크로, 플러그인, Neovim/LSP | 14 |
| [Spanish](./content/ko/Spanish/00_Overview.md) | 스페인어 문법/어휘, 접속법, DELE/SIELE | 25 |

### Tier 2 — 중급 (Intermediate)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [C_Programming](./content/ko/C_Programming/00_Overview.md) | C 기초 ~ 임베디드, 디버깅, 포인터 심화, 네트워크, IPC, 크로스 플랫폼 | 24 |
| [CPP](./content/ko/CPP/00_Overview.md) | Modern C++, 멀티스레딩, C++20, C++23, 디자인패턴, CMake | 22 |
| [Cryptography_Theory](./content/ko/Cryptography_Theory/00_Overview.md) | 정수론, 대칭/비대칭 암호, ECC, 전자서명, 양자내성, ZKP | 14 |
| [Data_Science](./content/ko/Data_Science/00_Overview.md) | NumPy, Pandas, 시각화, EDA, 확률, 추론, 베이지안, 시계열, 인과추론 | 29 |
| [Machine_Learning](./content/ko/Machine_Learning/00_Overview.md) | 회귀, 앙상블, SVM, 클러스터링, SHAP/LIME, AutoML | 21 |
| [PostgreSQL](./content/ko/PostgreSQL/00_Overview.md) | SQL, JSON, 복제, 파티셔닝, FTS, 보안/RLS | 20 |
| [Networking](./content/ko/Networking/00_Overview.md) | OSI/TCP-IP, 라우팅, 보안, IPv6, SDN, QoS, 멀티캐스트 | 22 |
| [OS_Theory](./content/ko/OS_Theory/00_Overview.md) | 프로세스, 스케줄링, 메모리, 파일시스템 | 18 |
| [Software_Engineering](./content/ko/Software_Engineering/00_Overview.md) | SDLC, 애자일, UML, 품질보증, DevOps, 유지보수 | 16 |
| [System_Design](./content/ko/System_Design/00_Overview.md) | 확장성, 분산시스템, 관측성, 검색 시스템 | 20 |
| [Computer_Architecture](./content/ko/Computer_Architecture/00_Overview.md) | CPU, 캐시, 가상메모리, 병렬처리, RISC-V, 전력 관리 | 20 |
| [IoT_Embedded](./content/ko/IoT_Embedded/00_Overview.md) | Raspberry Pi, MQTT, Edge AI, Zigbee/Z-Wave, 센서 퓨전 | 14 |
| [Compiler_Design](./content/ko/Compiler_Design/00_Overview.md) | 어휘/구문/의미 분석, IR, 코드 생성, 최적화, GC, LLVM | 16 |
| [Database_Theory](./content/ko/Database_Theory/00_Overview.md) | 관계형 모델, 정규화, 트랜잭션, 인덱싱, NoSQL, 분산 DB | 16 |
| [Security](./content/ko/Security/00_Overview.md) | CIA Triad, 암호학, TLS, OWASP, 인증/인가, 컨테이너 보안 | 16 |
| [Formal_Languages](./content/ko/Formal_Languages/00_Overview.md) | DFA, NFA, 정규식, 문맥자유문법, PDA, 튜링머신, P/NP | 14 |
| [Calculus_and_Differential_Equations](./content/ko/Calculus_and_Differential_Equations/00_Overview.md) | 미적분학, 다변수 미적분, ODE, PDE, 모델링 | 20 |

### Tier 3 — 고급 (Advanced)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [Algorithm](./content/ko/Algorithm/00_Overview.md) | 알고리즘/자료구조, 정렬, 그래프, DP, HLD, LCT, PST | 32 |
| [Computer_Graphics](./content/ko/Computer_Graphics/00_Overview.md) | 그래픽스 파이프라인, 래스터화, 셰이딩, WebGL, 레이 트레이싱 | 16 |
| [Control_Theory](./content/ko/Control_Theory/00_Overview.md) | 전달함수, PID, 상태공간, LQR, 칼만 필터, MPC | 16 |
| [Deep_Learning](./content/ko/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, Diffusion, Few-Shot, TTA | 44 |
| [Electrodynamics](./content/ko/Electrodynamics/00_Overview.md) | 정전기학, 자기정역학, 맥스웰 방정식, 전자기파, FDTD | 18 |
| [LLM_and_NLP](./content/ko/LLM_and_NLP/00_Overview.md) | BERT, GPT, HuggingFace, RAG, LangChain, RLHF | 16 |
| [Computer_Vision](./content/ko/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 3D비전, SLAM | 23 |
| [Optics](./content/ko/Optics/00_Overview.md) | 기하광학, 간섭, 회절, 편광, 레이저, 홀로그래피, 적응광학 | 17 |
| [Signal_Processing](./content/ko/Signal_Processing/00_Overview.md) | 푸리에 해석, 샘플링, FFT, Z변환, 디지털 필터, 적응 필터 | 16 |
| [Math_for_AI](./content/ko/Math_for_AI/00_Overview.md) | 선형대수, 최적화, 확률, 정보이론, Transformer 수학 | 18 |
| [Mathematical_Methods](./content/ko/Mathematical_Methods/00_Overview.md) | 푸리에, ODE/PDE, 특수함수, 텐서, 그린함수, 변분법 | 18 |
| [Numerical_Simulation](./content/ko/Numerical_Simulation/00_Overview.md) | ODE/PDE, CFD, FDTD, MHD, FEM, GPU 가속, PINN | 24 |
| [Data_Engineering](./content/ko/Data_Engineering/00_Overview.md) | Airflow, Spark, Kafka, dbt, CDC, Lakehouse, Dagster | 21 |
| [Quantum_Computing](./content/ko/Quantum_Computing/00_Overview.md) | 큐비트, 양자 게이트, Grover, Shor, VQE, QAOA | 16 |
| [Reinforcement_Learning](./content/ko/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, SAC, 커리큘럼, 계층적 RL | 16 |
| [Foundation_Models](./content/ko/Foundation_Models/00_Overview.md) | Scaling Laws, LLaMA, DINOv2, PEFT, RAG | 25 |
| [MLOps](./content/ko/MLOps/00_Overview.md) | MLflow, Kubeflow, 모델 서빙, CI/CD, DVC, LLMOps | 17 |
| [Robotics](./content/ko/Robotics/00_Overview.md) | 운동학, 동역학, 경로 계획, SLAM, ROS2 | 16 |

### Tier 4 — 전문 (Expert)

| 토픽 / Topic | 설명 / Description | 레슨 / Lessons |
|---|---|---|
| [Plasma_Physics](./content/ko/Plasma_Physics/00_Overview.md) | 디바이 차폐, 블라소프 방정식, 란다우 감쇠, 플라즈마 파동 | 16 |
| [MHD](./content/ko/MHD/00_Overview.md) | MHD 평형, 안정성, 자기재결합, 난류, 다이나모, 핵융합 | 18 |
| [Solar_Physics](./content/ko/Solar_Physics/00_Overview.md) | 태양 내부, 핵에너지, 코로나, 자기장, 플레어, CME | 16 |
| [Space_Weather](./content/ko/Space_Weather/00_Overview.md) | 자기권, 지자기 폭풍, 방사선대, 전리층, GIC, 예보 모델 | 16 |

---

## 시작하기 / Getting Started

각 폴더의 `00_Overview.md` 파일에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

Check the `00_Overview.md` file in each folder for learning roadmaps and detailed table of contents.

### 추천 학습 순서 / Recommended Learning Path

1. **프로그래밍 입문 / Programming Intro**: C_Programming (01~05)
2. **C++ 학습 / C++ Learning**: CPP (01~09) - C 기초 후 진행 권장 / Recommended after C basics
3. **알고리즘 기초 / Algorithm Basics**: Algorithm (01~05) - 복잡도, 배열/문자열, 정렬/탐색 / Complexity, Arrays, Sorting
4. **웹 개발 / Web Development**: Web_Development (01~09) - HTML/CSS/JS 입문 / HTML/CSS/JS Intro
5. **버전 관리 / Version Control**: Git (01~05)
6. **리눅스 기초 / Linux Basics**: Linux (01~03)
7. **개발 환경 / Dev Environment**: Docker (01~04)
8. **데이터베이스 / Database**: PostgreSQL (01~07)
9. **서버 관리 / Server Admin**: Linux (06~12)
10. **알고리즘 심화 / Advanced Algorithms**: Algorithm (06~15) - 그래프, DP, 실전 문제풀이 / Graphs, DP, Practice

## 다국어 지원 / Multilingual Support

- 모든 학습 자료는 **한국어(ko)**와 **영어(en)** 두 버전으로 관리
  All study materials are maintained in both **Korean (ko)** and **English (en)**
- 51개 토픽 모두 한/영 번역 완료
  All 51 topics fully translated to Korean/English
- 신규 자료는 영문 원본 작성 후 한국어 번역
  New materials are authored in English, then translated to Korean

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

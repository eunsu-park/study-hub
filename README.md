# Study Hub

프로그래밍 기초부터 플라즈마 물리학까지, 영한 이중언어 기술 학습 자료를 공개하는 레포.

A public repository of bilingual (EN/KO) technical study materials, from programming fundamentals to plasma physics.

---

## Project Structure / 프로젝트 구조

```
├── content/          # Study materials / 학습 자료 Markdown
│   ├── ko/           # Korean / 한국어
│   ├── en/           # English / 영어
│   ├── topic_metadata.yaml  # 4-Tier difficulty classification / 난이도 분류
│   └── learning_paths.yaml  # Cross-topic curricula / 학습 경로 정의
│
├── examples/         # Example code / 예제 코드
│
└── exercises/        # Exercise solutions / 연습문제 풀이
```

## Table of Contents / 목차

### Tier 1 — Beginner (입문)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| [C_Basics](./content/en/C_Basics/00_Overview.md) | C 언어 기초: 변수, 포인터, 구조체, 동적 메모리, 파일 I/O, 전처리기 | 15 |
| [CPP_Basics](./content/en/CPP_Basics/00_Overview.md) | C++ 기초: OOP, STL 컨테이너/알고리즘, 예외 처리, 파일 I/O, CMake | 15 |
| [CSharp_Basics](./content/en/CSharp_Basics/00_Overview.md) | C# 기초: 문법, 타입, OOP, 컬렉션, 제네릭, 예외 처리, 파일 I/O | 15 |
| [Docker](./content/en/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD, 컨테이너 네트워킹 | 16 |
| [Go_Basics](./content/en/Go_Basics/00_Overview.md) | Go 언어 기초: 타입, 함수, 인터페이스, 동시성, 테스팅 | 11 |
| [IDL_Basics](./content/en/IDL_Basics/00_Overview.md) | IDL 기초: 배열, 플로팅, FITS 파일, 구조체, 태양 데이터 처리 | 15 |
| [Rust_Basics](./content/en/Rust_Basics/00_Overview.md) | Rust 기초: 소유권, 빌림, 트레이트, 동시성, 비동기, Cargo | 16 |
| [Claude_Ecosystem](./content/en/Claude_Ecosystem/00_Overview.md) | Claude Code, MCP, Agent SDK, API, 비전, RAG | 25 |
| [Cloud_Computing](./content/en/Cloud_Computing/00_Overview.md) | 클라우드 서비스, AWS, GCP, 인프라 | 17 |
| [LaTeX](./content/en/LaTeX/00_Overview.md) | LaTeX 문서 조판, 수식, 그래픽스, 참고문헌 | 16 |
| [Web_Development](./content/en/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, 접근성, SEO, PWA, 웹 컴포넌트 | 19 |
| [Git](./content/en/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 | 14 |
| [Linux](./content/en/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 | 26 |
| [Programming](./content/en/Programming/00_Overview.md) | 프로그래밍 개념, 패러다임, 디자인 패턴, 클린 코드, 테스팅 | 16 |
| [Python_Basics](./content/en/Python_Basics/00_Overview.md) | Python 언어 기초: 변수, 제어문, 함수, 자료구조, OOP, 모듈, 표준 라이브러리 | 14 |
| [Shell_Script](./content/en/Shell_Script/00_Overview.md) | Bash 심화, 매개변수 확장, 프로세스 관리, 배포 자동화 | 16 |
| [VIM](./content/en/VIM/00_Overview.md) | 모달 편집, 모션, 매크로, 플러그인, Neovim/LSP | 14 |

### Tier 2 — Intermediate (중급)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| [C_Advanced](./content/en/C_Advanced/00_Overview.md) | 고급 C: 시스템 프로그래밍, 자료구조, 네트워크, 동시성, 임베디드, 크로스 플랫폼 | 17 |
| [Calculus_and_Differential_Equations](./content/en/Calculus_and_Differential_Equations/00_Overview.md) | 미적분학, 다변수 미적분, ODE, PDE, 모델링 | 20 |
| [CPP_Advanced](./content/en/CPP_Advanced/00_Overview.md) | 고급 C++: 템플릿, 모던 C++11~23, 동시성, 디자인 패턴 | 17 |
| [CSharp_Advanced](./content/en/CSharp_Advanced/00_Overview.md) | 고급 C#: LINQ, async/await, 패턴 매칭, 레코드, Span, DI, .NET 생태계 | 17 |
| [Data_Science](./content/en/Data_Science/00_Overview.md) | NumPy, Pandas, 시각화, EDA, 확률, 추론, 베이지안, 시계열 | 29 |
| [Go_Advanced](./content/en/Go_Advanced/00_Overview.md) | Go 고급: HTTP 서버, REST API, DB, 제네릭, 리플렉션, 마이크로서비스 | 11 |
| [IDL_Advanced](./content/en/IDL_Advanced/00_Overview.md) | IDL 고급: SolarSoft, SDO/AIA/HMI 분석, 영상 처리, IDL-Python 브리지 | 15 |
| [Database_Theory](./content/en/Database_Theory/00_Overview.md) | 관계형 모델, 정규화, 트랜잭션, 인덱싱, NoSQL, 분산 DB | 16 |
| [Linear_Algebra](./content/en/Linear_Algebra/00_Overview.md) | 벡터 공간, 행렬 분해, SVD, PCA, 수치 해법, ML/DL/CG 응용 | 20 |
| [Machine_Learning](./content/en/Machine_Learning/00_Overview.md) | 회귀, 앙상블, SVM, 클러스터링, SHAP/LIME, AutoML, Symbolic Regression | 24 |
| [Networking](./content/en/Networking/00_Overview.md) | OSI/TCP-IP, 라우팅, 보안, IPv6, SDN, QoS, 멀티캐스트 | 22 |
| [OS_Theory](./content/en/OS_Theory/00_Overview.md) | 프로세스, 스케줄링, 메모리, 파일시스템, 컨테이너 내부, eBPF | 27 |
| [PostgreSQL](./content/en/PostgreSQL/00_Overview.md) | SQL, JSON, 복제, 파티셔닝, FTS, 보안/RLS | 20 |
| [Probability_and_Statistics](./content/en/Probability_and_Statistics/00_Overview.md) | 확률론, 통계적 추론, 이산/연속 분포, 베이지안, 확률 과정 | 18 |
| [Cryptography_Theory](./content/en/Cryptography_Theory/00_Overview.md) | 암호화 알고리즘: 대칭/비��칭, RSA, ECC, 격자, 포스트양자, ZKP | 14 |
| [Formal_Languages](./content/en/Formal_Languages/00_Overview.md) | 오토마타 이론, 형식 언어, 튜링 머신, 계산 가능성 | 14 |
| [Python_Advanced](./content/en/Python_Advanced/00_Overview.md) | Python 고급: 데코레이터, 메타클래스, async, 디스크립터, 함수형, 성능 최적화 | 14 |
| [Rust_Advanced](./content/en/Rust_Advanced/00_Overview.md) | Rust 고급: unsafe, 매크로, FFI, WebAssembly, 임베디드, 네트워킹, 성능 | 14 |
| [Security](./content/en/Security/00_Overview.md) | 사이버보안: CIA, TLS, 인증/인가, OWASP, 컨테이너 보안, 취약점 스캐너 | 16 |
| [Software_Engineering](./content/en/Software_Engineering/00_Overview.md) | 소프트웨어 공학: SDLC, 애자일, UML, QA, CI/CD, 기술 문서 | 16 |
| [API_Design](./content/en/API_Design/00_Overview.md) | REST, GraphQL, gRPC — API 설계, 버저닝, 인증, 게이트웨이 | 25 |
| [Backend_Frameworks](./content/en/Backend_Frameworks/00_Overview.md) | FastAPI, Express, Django — 백엔드 API, 인증, 배포 | 21 |
| [Frontend_Frameworks](./content/en/Frontend_Frameworks/00_Overview.md) | React, Vue, Svelte — 컴포넌트, 상태 관리, SSR, 테스팅 | 18 |
| [Testing_and_QA](./content/en/Testing_and_QA/00_Overview.md) | pytest, TDD, 통합/E2E/속성 기반 테스팅, CI/CD | 18 |
| [Computer_Architecture](./content/en/Computer_Architecture/00_Overview.md) | ��리 ������트, CPU, 파이프라인, 캐시, 가상 메모리, RISC-V | 20 |
| [DevOps](./content/en/DevOps/00_Overview.md) | IaC, CI/CD, Terraform, 모니터링, SRE, GitOps, 플랫폼 엔지니어링 | 28 |
| [Kubernetes](./content/en/Kubernetes/00_Overview.md) | 아키텍처, 워크로드, 네트워킹, CRD, 오퍼레이터, 프로덕션 운영 | 19 |
| [System_Design](./content/en/System_Design/00_Overview.md) | 확장성, 캐싱, DB 스케일링, 메시지 큐, 마이크로서비스, 합의 알고리즘 | 20 |
| [IoT_Embedded](./content/en/IoT_Embedded/00_Overview.md) | IoT, 라즈베리파이, MQTT, BLE, 엣지 AI, 센서 퓨전 | 14 |

### Tier 3 — Advanced (고급)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| [Algorithm](./content/en/Algorithm/00_Overview.md) | 알고리즘/자료구조, 정렬, 그래프, DP, HLD, LCT, PST | 32 |
| [Computer_Graphics](./content/en/Computer_Graphics/00_Overview.md) | 렌더링 파이프라인, 셰이딩, 레이 트레이싱, WebGL, GPU 컴퓨팅 | 16 |
| [Control_Theory](./content/en/Control_Theory/00_Overview.md) | 피드백 제어, PID, 근궤적, 보드/나이퀴스트, 상태공간, 디지털 제어 | 16 |
| [Signal_Processing](./content/en/Signal_Processing/00_Overview.md) | 신호/시스템, 푸리에, DFT/FFT, 디지털 필터, 적응 필터, 스펙트로그램 | 16 |
| [Interpretable_AI](./content/en/Interpretable_AI/00_Overview.md) | 그래디언트 어트리뷰션, SHAP, 공정성, 인과 추론, AI 거버넌스 | 16 |
| [MLOps](./content/en/MLOps/00_Overview.md) | MLflow, W&B, 모델 서빙, 드리프트 감지, LLMOps, DVC | 17 |
| [Probabilistic_Programming](./content/en/Probabilistic_Programming/00_Overview.md) | 베이지안, MCMC, PyMC, Stan, Pyro, GP, 변분 추론 | 18 |
| [Prompt_Engineering](./content/en/Prompt_Engineering/00_Overview.md) | 프롬프트 설계, CoT, 구조화 출력, 멀티모달, 에이전트 패턴 | 17 |
| [Compiler_Design](./content/en/Compiler_Design/00_Overview.md) | 렉서, 파서, AST, IR, 최적화, GC, SSA, JIT, LLVM | 28 |
| [CUDA](./content/en/CUDA/00_Overview.md) | GPU 프로그래밍, 스레드 모델, 메모리 계층, 병렬 알고리즘, 과학 시뮬레이션 | 38 |
| [Distributed_Systems](./content/en/Distributed_Systems/00_Overview.md) | 합의 프로토콜, Raft/Paxos, CRDT, 분산 트랜잭션, 형식 검증 | 27 |
| [Quantum_Computing](./content/en/Quantum_Computing/00_Overview.md) | 큐비트, 양자 게이트, Shor/Grover, VQE, QAOA, 양자 네트워킹 | 24 |
| [Computer_Vision](./content/en/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 세그멘테이션, 3D비전, NeRF, SLAM | 31 |
| [Data_Engineering](./content/en/Data_Engineering/00_Overview.md) | Airflow, Spark, Kafka, dbt, CDC, Lakehouse, 벡터 검색 | 23 |
| [Deep_Learning](./content/en/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, Diffusion, Few-Shot, TTA | 47 |
| [Electrodynamics](./content/en/Electrodynamics/00_Overview.md) | 정전기학, 자기정역학, 맥스웰 방정식, 전자기파, FDTD | 18 |
| [Foundation_Models](./content/en/Foundation_Models/00_Overview.md) | FM 패러다임, Scaling Laws, LLaMA, DINOv2, SAM, 멀티모달 | 22 |
| [Math_for_AI](./content/en/Math_for_AI/00_Overview.md) | 선형대수, 최적화, 확률, 정보이론, Transformer 수학 | 18 |
| [Mathematical_Methods](./content/en/Mathematical_Methods/00_Overview.md) | 푸리에, ODE/PDE, 특수함수, 텐서, 그린함수, 변분법 | 18 |
| [NLP_and_LLM](./content/en/NLP_and_LLM/00_Overview.md) | NLP, BERT, GPT, HuggingFace, PEFT, RAG, LangChain, 에이전트 | 27 |
| [Numerical_Simulation](./content/en/Numerical_Simulation/00_Overview.md) | ODE/PDE, CFD, FDTD, MHD, FEM, GPU 가속, PINN | 24 |
| [Optics](./content/en/Optics/00_Overview.md) | 기하광학, 간섭, 회절, 편광, 레이저, 홀로그래피, 적응광학 | 17 |
| [Reinforcement_Learning](./content/en/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, SAC, Offline RL, RLHF, World Models | 27 |

### Tier 4 — Expert (전문)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| [DL_Scratch_C](./content/en/DL_Scratch_C/00_Overview.md) | C/C++로 딥러닝 밑바닥 구현: 텐서 엔진, 자동미분, CNN, Transformer, 학습, 양자화, 추론 엔진 | 46 |
| [MHD](./content/en/MHD/00_Overview.md) | MHD 평형, 안정성, 자기재결합, 난류, 다이나모, 핵융합 | 18 |
| [Plasma_Physics](./content/en/Plasma_Physics/00_Overview.md) | 디바이 차폐, 블라소프 방정식, 란다우 감쇠, 플라즈마 파동 | 16 |
| [Solar_Physics](./content/en/Solar_Physics/00_Overview.md) | 태양 내부, 핵에너지, 코로나, 자기장, 플레어, CME | 16 |
| [Space_Weather](./content/en/Space_Weather/00_Overview.md) | 자기권, 지자기 폭풍, 방사선대, 전리층, GIC, 예보 모델 | 16 |

## Learning Paths / 학습 경로

| Path / 경로 | Topics / 토픽 |
|---|---|
| Python Developer / Python 개발자 | Programming → Python_Basics → Python_Advanced |
| Systems Programmer / 시스템 프로그래머 | Programming → C_Basics → C_Advanced → CPP_Basics → CPP_Advanced → Rust_Basics → Rust_Advanced |
| .NET Developer / .NET 개발자 | Programming → CSharp_Basics → CSharp_Advanced |
| CV Engineer / 컴퓨터 비전 엔지니어 | Programming → Python_Basics → Python_Advanced → Machine_Learning → Deep_Learning → Computer_Vision |
| ML Engineer / 머신러닝 엔지니어 | Programming → Python_Basics → Python_Advanced → Machine_Learning → Deep_Learning |
| Linux & DevOps | Linux → Shell_Script → Git → Docker |
| Scientific Computing / 과학 계산 | Calculus → Linear_Algebra → Probability_and_Statistics → Mathematical_Methods → Math_for_AI → Electrodynamics |
| Space Physics / 우주물리학 | Calculus → Mathematical_Methods → Electrodynamics → IDL_Basics → IDL_Advanced → Numerical_Simulation → Plasma_Physics → MHD → Solar_Physics → Space_Weather |
| Data & AI / 데이터 & AI | Data_Science → Machine_Learning → Deep_Learning → NLP_and_LLM → Foundation_Models → Reinforcement_Learning |

---

## Getting Started / 시작하기

Check the `00_Overview.md` file in each folder for learning roadmaps and detailed table of contents.

각 폴더의 `00_Overview.md` 파일에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

---

## Companion Viewer / 뷰어

This content is rendered by the [study-hub-viewer](https://github.com/eunsu-park/study-hub-viewer) — a Flask-based web viewer with bilingual support and progress tracking.

이 콘텐츠는 [study-hub-viewer](https://github.com/eunsu-park/study-hub-viewer)로 렌더링됩니다 — Flask 기반 웹 뷰어로 다국어 지원과 진도 추적을 지원합니다.

---

## License / 라이센스

This project uses a dual license:
이 프로젝트는 이중 라이센스를 적용합니다:

| Target / 대상 | License / 라이센스 |
|---|---|
| Study materials (`content/`) / 학습 자료 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Code (`examples/`, `exercises/`) / 코드 | [MIT License](./LICENSE) |

## Author

**Eunsu Park**
- [ORCID: 0000-0003-0969-286X](https://orcid.org/0000-0003-0969-286X)

# Personal Study Repository

이 레포지터리는 제가 스스로 공부하기 위해 만든 자료를 정리한 것입니다. 일부 내용은 제가 직접 작성했으며, 일부는 AI가 검수하거나 AI의 도움을 받아 작성되었습니다.

This repository is a personal study project. Some content was written by me, some was reviewed by AI, and some was created with the help of AI.

---

## 프로젝트 구조 / Project Structure

```
03_Study/
├── content/          # 학습 자료 (한국어/영어 Markdown)
│   ├── ko/           # 한국어 자료 (39개 토픽)
│   └── en/           # 영어 자료 (39개 토픽)
│
├── examples/         # 예제 코드 (32개 토픽, 766개 파일)
│   ├── Algorithm/    # Python, C, C++ 예제
│   ├── C_Programming/
│   └── ...
│
├── site/             # 정적 사이트 생성기 (GitHub Pages)
│
└── viewer/           # Flask 웹 뷰어 애플리케이션
```

```
03_Study/
├── content/          # Study materials (Korean/English Markdown)
│   ├── ko/           # Korean materials (39 topics)
│   └── en/           # English materials (39 topics)
│
├── examples/         # Example code (32 topics, 766 files)
│   ├── Algorithm/    # Python, C, C++ examples
│   ├── C_Programming/
│   └── ...
│
├── site/             # Static site generator (GitHub Pages)
│
└── viewer/           # Flask web viewer application
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

| 폴더 / Folder | 내용 / Description | 레슨 수 / Lessons |
|---------------|-------------------|-------------------|
| [Algorithm/](./content/ko/Algorithm/00_Overview.md) | 알고리즘/자료구조, 코딩테스트 준비 / Algorithms, Data Structures, Coding Test Prep | 29 |
| [C_Programming/](./content/ko/C_Programming/00_Overview.md) | C 언어 기초 ~ 임베디드, 디버깅, 포인터 심화, 네트워크, IPC / C Basics to Embedded, Debugging, Advanced Pointers, Network, IPC | 22 |
| [Compiler_Design/](./content/ko/Compiler_Design/00_Overview.md) | 컴파일러 설계, 어휘/구문/의미 분석, AST, IR, 코드 생성, 최적화, GC, VM, LLVM / Compiler Design, Lexical/Syntax/Semantic Analysis, Code Generation, Optimization | 16 |
| [Claude_Ecosystem/](./content/ko/Claude_Ecosystem/00_Overview.md) | Claude Code, CLAUDE.md, Hooks, MCP, Agent SDK / Claude Code, MCP, Agent SDK | 22 |
| [Cloud_Computing/](./content/ko/Cloud_Computing/00_Overview.md) | AWS/GCP, EC2, Lambda, S3, VPC, IAM, Terraform | 17 |
| [Computer_Architecture/](./content/ko/Computer_Architecture/00_Overview.md) | 컴퓨터 구조, CPU, 메모리, 병렬처리 / Computer Architecture, CPU, Memory, Parallelism | 18 |
| [Computer_Vision/](./content/ko/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 3D비전, SLAM / OpenCV, Image Processing, Object Detection, 3D Vision | 23 |
| [CPP/](./content/ko/CPP/00_Overview.md) | C++ 입문 ~ 모던 C++, 디자인패턴, 프로젝트 / C++ Intro to Modern C++, Design Patterns, Projects | 19 |
| [Data_Engineering/](./content/ko/Data_Engineering/00_Overview.md) | 데이터 파이프라인, Airflow, Spark, Kafka, CDC, Lakehouse / Data Pipelines | 19 |
| [Data_Science/](./content/ko/Data_Science/00_Overview.md) | NumPy, Pandas, 시각화, EDA, 확률, 추론, 베이지안, 시계열, 인과추론, 생존분석 / NumPy, Pandas, Visualization, EDA, Probability, Inference, Bayesian, Time Series, Causal, Survival | 29 |
| [Database_Theory/](./content/ko/Database_Theory/00_Overview.md) | 관계형 모델, 정규화, 트랜잭션, 인덱싱, NoSQL, 분산 DB, NewSQL / Relational Model, Normalization, Transactions, Indexing, NoSQL, Distributed DB, NewSQL | 16 |
| [Deep_Learning/](./content/ko/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, ViT, from-scratch 구현, Modern Architectures | 42 |
| [Docker/](./content/ko/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD, Container Networking, Security | 12 |
| [Foundation_Models/](./content/ko/Foundation_Models/00_Overview.md) | Scaling Laws, LLaMA, Mistral, DINOv2, SAM, PEFT, RAG | 25 |
| [Git/](./content/ko/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 / Git, GitHub, Workflows, Monorepo | 10 |
| [IoT_Embedded/](./content/ko/IoT_Embedded/00_Overview.md) | Raspberry Pi, MQTT, Edge AI, 클라우드 IoT / Cloud IoT | 12 |
| [LaTeX/](./content/ko/LaTeX/00_Overview.md) | LaTeX 문서 조판, 수학 수식, TikZ, Beamer, 빌드 자동화 / LaTeX Typesetting, Math, TikZ, Beamer, Build Automation | 16 |
| [Linux/](./content/ko/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 / Linux Basics to HA Cluster, Troubleshooting | 26 |
| [LLM_and_NLP/](./content/ko/LLM_and_NLP/00_Overview.md) | NLP, BERT, GPT, HuggingFace, RAG, LangChain, RLHF, Agent | 16 |
| [Machine_Learning/](./content/ko/Machine_Learning/00_Overview.md) | 회귀, 분류, 앙상블, 클러스터링, Feature Engineering, SHAP/LIME, AutoML / Regression, Classification, Ensemble, Clustering, Explainability, AutoML | 20 |
| [Math_for_AI/](./content/ko/Math_for_AI/00_Overview.md) | AI/ML/DL 수학, 선형대수, 최적화, 확률, 정보이론, Transformer 수학 / Math for AI/ML/DL | 18 |
| [Mathematical_Methods/](./content/ko/Mathematical_Methods/00_Overview.md) | 물리과학 수학, 선형대수, 편미분, 푸리에, PDE, 구면조화함수, 텐서 / Math Methods for Physical Sciences | 18 |
| [MHD/](./content/ko/MHD/00_Overview.md) | 자기유체역학, 평형, 안정성, 자기재결합, 난류, 다이나모, 핵융합 / MHD Equilibria, Stability, Reconnection, Turbulence, Dynamo, Fusion | 18 |
| [MLOps/](./content/ko/MLOps/00_Overview.md) | MLflow, Kubeflow, 모델 서빙, 드리프트 감지, CI/CD, DVC, LLMOps / Model Serving, Drift Detection, CI/CD, DVC, LLMOps | 15 |
| [Networking/](./content/ko/Networking/00_Overview.md) | 네트워크, OSI/TCP-IP, 라우팅, 보안, IPv6, 컨테이너 / Networking, OSI/TCP-IP, Routing, Security, IPv6, Container | 19 |
| [Numerical_Simulation/](./content/ko/Numerical_Simulation/00_Overview.md) | 수치해석, ODE/PDE, CFD, FDTD, MHD, 몬테카를로, Spectral, FEM / Numerical Analysis, Monte Carlo, Spectral, FEM | 22 |
| [OS_Theory/](./content/ko/OS_Theory/00_Overview.md) | 운영체제, 프로세스, 메모리, 파일시스템 / OS, Processes, Memory, Filesystem | 18 |
| [Plasma_Physics/](./content/ko/Plasma_Physics/00_Overview.md) | 플라즈마 물리, 디바이 차폐, 입자 운동, 블라소프 방정식, 란다우 감쇠, 플라즈마 파동 / Plasma Physics, Debye Shielding, Vlasov, Landau Damping | 16 |
| [PostgreSQL/](./content/ko/PostgreSQL/00_Overview.md) | SQL 기초 ~ 복제, 파티셔닝, FTS, 보안/RLS / SQL Basics to Replication, Partitioning, FTS, Security/RLS | 20 |
| [Programming/](./content/ko/Programming/00_Overview.md) | 프로그래밍 개념, 패러다임, 디자인 패턴, 클린 코드, 테스팅, 아키텍처 / Programming Concepts, Paradigms, Design Patterns, Clean Code, Testing, Architecture | 16 |
| [Python/](./content/ko/Python/00_Overview.md) | 파이썬 고급 문법, 테스트, 패키징 / Advanced Python, Testing, Packaging | 16 |
| [Reinforcement_Learning/](./content/ko/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, Model-Based RL, SAC | 14 |
| [Security/](./content/ko/Security/00_Overview.md) | 보안, 암호학, OWASP, 인증/인가, 취약점 스캐너 / Security, Cryptography, OWASP, Auth, Vulnerability Scanner | 16 |
| [Shell_Script/](./content/ko/Shell_Script/00_Overview.md) | Bash 심화, 매개변수 확장, 프로세스 관리, 테스팅, 배포 자동화 / Advanced Bash, Parameter Expansion, Process Mgmt, Testing, Deployment | 16 |
| [Signal_Processing/](./content/ko/Signal_Processing/00_Overview.md) | 신호와 시스템, 푸리에 해석, 샘플링, FFT, Z변환, 디지털 필터, FIR/IIR 설계, 다중률, 적응 필터, 시간-주파수 분석 / Signals, Fourier, Sampling, FFT, Z-Transform, Digital Filters, Multirate, Adaptive, Time-Frequency | 16 |
| [Software_Engineering/](./content/ko/Software_Engineering/00_Overview.md) | SDLC, 애자일, 요구공학, UML, 추정/계획, 품질보증, V&V, 형상관리, 프로젝트관리, 유지보수, DevOps / SDLC, Agile, Requirements, UML, Estimation, QA, V&V, Config Mgmt, Project Mgmt, Maintenance, DevOps | 16 |
| [Spanish/](./content/ko/Spanish/00_Overview.md) | 스페인어 문법/어휘, 접속법, 지역 변형 / Spanish Grammar, Vocabulary, Subjunctive | 11 |
| [System_Design/](./content/ko/System_Design/00_Overview.md) | 시스템 설계, 확장성, 분산시스템, 관측성, 검색 / System Design, Scalability, Distributed Systems, Observability, Search | 20 |
| [Web_Development/](./content/ko/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, SEO, Flask | 16 |

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
- 39개 토픽 모두 한/영 번역 완료
  All 39 topics fully translated to Korean/English
- 신규 자료는 영문 원본 작성 후 한국어 번역
  New materials are authored in English, then translated to Korean

## 라이센스 / License

이 프로젝트는 이중 라이센스를 적용합니다:
This project uses a dual license:

| 대상 / Target | 라이센스 / License |
|---------------|-------------------|
| 학습 자료 (`content/`) / Study materials | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| 코드 (`examples/`, `viewer/`) / Code | [MIT License](./LICENSE) |

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

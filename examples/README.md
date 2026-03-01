# Examples (예제 코드)

학습 자료에 대응하는 실행 가능한 예제 코드 모음입니다.

A collection of executable example code corresponding to the study materials.

## 디렉토리 구조 / Directory Structure

```
examples/
├── Algorithm/              # Python, C, C++ 예제 / examples (89 files)
│   ├── python/             # Python 구현 / implementation (29)
│   ├── c/                  # C 구현 / implementation (29 + Makefile)
│   └── cpp/                # C++ 구현 / implementation (29 + Makefile)
│
├── C_Programming/          # C 프로젝트 예제 / C project examples (88 files)
│   ├── 02_calculator/
│   ├── 03_number_guess/
│   ├── 04_address_book/    # 주소록 관리 / Address book
│   ├── 05_dynamic_array/
│   ├── 06_linked_list/
│   ├── 07_file_crypto/     # 파일 암호화 / File encryption
│   ├── 08_stack_queue/     # 스택/큐 구현 / Stack/Queue
│   ├── 09_hash_table/      # 해시 테이블 / Hash table
│   ├── 10_snake_game/      # 뱀 게임 / Snake game
│   ├── 11_minishell/       # 미니 쉘 / Mini shell
│   ├── 12_multithread/     # 멀티스레딩 / Multithreading
│   ├── 13_embedded_basic/
│   ├── 14_network/         # 네트워크 프로그래밍 / Network programming
│   ├── 15_ipc/             # IPC, 시그널 / IPC, Signals
│   └── 17_testing/         # 단위 테스트, 프로파일링 / Unit testing, Profiling
│
├── Compiler_Design/        # 컴파일러 설계 Python / Compiler design examples (13 files)
│   ├── 01_lexer.py ~ 10_bytecode_vm.py  # 렉서→바이트코드 VM / Lexer to Bytecode VM
│   ├── 11_register_allocator.py         # 레지스터 할당 / Register allocation
│   └── 12_mini_compiler.py             # 미니 컴파일러 / End-to-end mini compiler
│
├── Computer_Architecture/  # 컴퓨터 구조 시뮬레이터 / Architecture simulators (12 files)
│   ├── 02_number_systems.py ~ 07_cpu_datapath.py  # 수 체계, 논리 회로, CPU
│   ├── 10_assembly_sim.py ~ 12_branch_predictor.py # ISA, 파이프라인, 분기 예측
│   └── 15_cache_sim.py, 16_tlb_sim.py             # 캐시, TLB
│
├── CPP/                    # C++ 고급 예제 / C++ advanced examples (37 files)
│   ├── 01_modern_cpp.cpp       # Modern C++ (C++17/20)
│   ├── 02_stl_containers.cpp   # STL containers & algorithms
│   ├── 03_smart_pointers.cpp   # Smart pointers
│   ├── 04_threading.cpp        # Multithreading
│   ├── 05_design_patterns.cpp  # Design patterns
│   ├── 06_templates.cpp        # Template metaprogramming
│   ├── 07_move_semantics.cpp   # Move semantics
│   ├── 08_cmake_demo/          # CMake 프로젝트 / CMake demo project
│   ├── student_management/     # 학생 관리 프로젝트 / Student management project
│   └── Makefile                # Build system
│
├── Claude_Ecosystem/      # Claude 생태계 예제 / Claude ecosystem examples (16 files)
│   ├── 03_claude_md/          # CLAUDE.md, settings 예제 / examples
│   ├── 05_hooks/              # Hook 설정 예제 / Hook config examples
│   ├── 06_skills/             # 커스텀 스킬 / Custom skills
│   ├── 07_subagents/          # 서브에이전트 정의 / Subagent definitions
│   ├── 13_mcp_server/         # MCP 서버 구현 / MCP server implementations
│   ├── 16_tool_use/           # API 도구 사용 / API tool use
│   └── 17_agent_sdk/          # Agent SDK 예제 / Agent SDK examples
│
├── Control_Theory/        # 제어 이론 Python 예제 / Control theory examples (12 files)
│   ├── 01_modeling.py ~ 11_digital_control.py  # 모델링→디지털제어 / Modeling to Digital Control
│   └── README.md
│
├── Computer_Vision/        # OpenCV/Python 예제 / examples (23 files)
├── Database_Theory/        # 데이터베이스 이론 Python / Database theory examples (15 files)
│   ├── 01-10_*.py                 # 관계형 모델, B+트리, MVCC 등 / Relational, B+Tree, MVCC etc.
│   ├── 11_two_phase_locking.py    # 2PL 동시성 / 2PL concurrency
│   ├── 12_aries_recovery.py       # ARIES 복구 / ARIES recovery
│   ├── 14_distributed_2pc.py      # 분산 2PC / Distributed 2PC
│   └── 16_design_case_study.py    # 설계 사례 / Design case study
│
├── Data_Science/           # 데이터 과학 예제 / Data science examples (27 files)
│   ├── data_analysis/      # NumPy, Pandas, 시각화, Polars, DuckDB / Visualization (10)
│   └── statistics/         # 통계학 예제 / Statistics (17)
├── Data_Engineering/       # Airflow/Spark/Kafka/CDC/Lakehouse 예제 / examples (33 files)
│   ├── airflow/            # TaskFlow API
│   ├── cdc/                # Debezium CDC
│   ├── kafka/              # Kafka Streams, ksqlDB
│   ├── lakehouse/          # Delta Lake patterns
│   ├── practical_pipeline/ # 실습 파이프라인 프로젝트 / Practical pipeline project (L14)
│   └── spark/              # Structured Streaming
├── Deep_Learning/          # PyTorch 예제 / examples (47 files)
│   ├── numpy/              # NumPy 기초 구현 / basic implementation (5)
│   ├── pytorch/            # PyTorch 구현 / implementation (22)
│   └── implementations/   # 모델 구현 코드 / Model implementation code (15)
│       ├── 01_Linear_Logistic/  # 선형/로지스틱 회귀 / Linear/Logistic regression
│       ├── 03_CNN_LeNet/        # LeNet 구현 / LeNet implementation
│       ├── 06_LSTM_GRU/         # LSTM/GRU 구현 / LSTM/GRU implementation
│       └── ...                  # 12 model directories
│
├── Docker/                 # Docker/Kubernetes 예제 / examples (15 files)
│   ├── 01_multi_stage/     # Multi-stage Docker build
│   ├── 02_compose/         # Docker Compose 3-tier stack
│   ├── 03_k8s/             # Kubernetes manifests
│   └── 04_ci_cd/           # GitHub Actions CI/CD pipeline
│
├── LaTeX/                  # LaTeX 예제 / examples (19 files)
│   ├── 01_hello_world/        # 첫 문서 / First document
│   ├── 02_document_structure/ # 문서 구조 / Document structure
│   ├── 04_math_basics/        # 수학 기초 / Math basics
│   ├── 05_math_advanced/      # 고급 수학 / Advanced math
│   ├── 06_figures/            # 그림 / Figures
│   ├── 07_tables/             # 표 / Tables
│   ├── 08_bibliography/       # 참고문헌 / Bibliography
│   ├── 10_tikz_basics/        # TikZ 기초 / TikZ basics
│   ├── 11_tikz_advanced/      # 고급 TikZ / Advanced TikZ
│   ├── 12_beamer/             # Beamer 프레젠테이션 / Beamer presentations
│   ├── 13_custom_commands/    # 사용자 정의 명령 / Custom commands
│   └── 16_projects/           # 프로젝트 / Projects
│
├── IoT_Embedded/           # Raspberry Pi/MQTT 예제 / examples (17 files)
│   ├── edge_ai/            # TFLite, ONNX 추론 / inference
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # 스마트홈, 이미지분석, 클라우드IoT / Smart home, Image analysis, Cloud IoT
│   └── raspberry_pi/       # GPIO, 센서 / sensors
│
├── LLM_and_NLP/            # NLP/HuggingFace 예제 / examples (15 files)
├── Machine_Learning/       # sklearn/Jupyter 예제 / examples (25 files)
├── Math_for_AI/            # AI 수학 Python 예제 / AI math examples (13 files)
├── MLOps/                  # MLflow/CI/CD/DVC/LLMOps 예제 / examples (32 files)
│   ├── cicd/               # ML CI/CD 파이프라인 / ML CI/CD pipeline
│   ├── dvc/                # DVC 데이터 버전 관리 / DVC data version control
│   ├── feature_store/      # Feast 피처 스토어 예제 / Feature store examples (L11)
│   ├── llmops/             # LLMOps 모니터링 / LLMOps monitoring
│   └── practical_project/  # E2E MLOps 실습 프로젝트 / E2E MLOps project (L12)
├── Numerical_Simulation/   # 수치해석 Python 예제 / Numerical analysis examples (8 files)
├── Programming/           # 프로그래밍 개념 예제 / Programming concepts examples (13 files)
│   ├── 02_paradigms/          # 패러다임 비교 / Paradigm comparison
│   ├── 05_oop/                # OOP, SOLID / OOP principles
│   ├── 06_functional/         # 함수형 프로그래밍 / Functional programming
│   ├── 07_design_patterns/    # 디자인 패턴 / Design patterns
│   ├── 08_clean_code/         # 클린 코드 / Clean code refactoring
│   ├── 09_error_handling/     # 에러 처리 / Error handling
│   ├── 10_testing/            # 테스팅 / Testing (pytest)
│   └── 12_concurrency/       # 동시성 / Concurrency (threading, asyncio)
│
├── Networking/             # 네트워킹 시뮬레이터 / Networking simulators (12 files)
│   ├── 02_osi_packet_builder.py ~ 09_routing_protocol_sim.py  # OSI, 서브넷, 라우팅
│   ├── 10_tcp_state_machine.py ~ 13_http_client.py            # TCP, DNS, HTTP
│   └── 15_firewall_sim.py ~ 18_ipv6_demo.py                  # 방화벽, IPv6
│
├── OS_Theory/              # 운영체제 시뮬레이터 / OS theory simulators (12 files)
│   ├── 02_process_demo.py ~ 09_deadlock_detection.py  # 프로세스, 스케줄링, 동기화
│   ├── 12_paging_sim.py, 15_page_replacement.py       # 메모리 관리
│   └── 17_filesystem_sim.py, 18_ipc_demo.py           # 파일시스템, IPC
│
├── PostgreSQL/             # SQL 예제 / examples (18 files)
│   ├── 01-07_*.sql                     # SQL queries (CRUD, joins, window, FTS, RLS)
│   ├── 08_primary_standby_compose.yml  # Primary-Standby replication setup
│   ├── 09_primary_standby_setup.sh     # Automated replication setup script
│   ├── 10-18_*.sql                     # Functions, transactions, triggers, monitoring, JSON, optimization, windows, partitioning
│   └── README.md                       # PostgreSQL examples guide
├── Reinforcement_Learning/ # RL Python 예제 / examples (14 files)
├── Formal_Languages/      # 형식 언어 예제 / Formal languages examples (9 files)
│   ├── 01_dfa_simulator.py            # DFA 시뮬레이터, 최소화 / DFA simulator, minimization
│   ├── 02_nfa_subset_construction.py  # NFA, 부분집합 구성 / NFA, subset construction
│   ├── 03_regular_expressions.py      # 정규 표현식 엔진 (Thompson) / Regex engine
│   ├── 04_pumping_lemma.py            # 펌핑 보조정리 / Pumping lemma
│   ├── 05_cfg_cyk_parser.py           # CFG, CNF, CYK 파싱 / CFG, CNF, CYK parsing
│   ├── 06_pushdown_automaton.py       # 푸시다운 오토마타 / PDA
│   ├── 07_turing_machine.py           # 튜링 기계 / Turing machine
│   └── 08_chomsky_hierarchy.py        # 촘스키 계층 / Chomsky hierarchy
│
├── Foundation_Models/      # 파운데이션 모델 예제 / Foundation model examples (8 files)
├── Mathematical_Methods/  # 물리수학 Python 예제 / Math methods examples (13 files)
├── MHD/                   # MHD Python 예제 / MHD examples (32 files)
│   ├── 01_equilibria/         # 평형 / Equilibria
│   ├── 02_stability/          # 안정성 / Stability analysis
│   ├── 03_instabilities/      # 불안정성 / Instabilities
│   ├── 04_reconnection/       # 자기재결합 / Reconnection
│   ├── 05_turbulence/         # 난류 / Turbulence
│   ├── 06_dynamo/             # 다이나모 / Dynamo
│   ├── 07_astrophysics/       # 천체물리 / Astrophysics
│   ├── 08_fusion/             # 핵융합 / Fusion
│   ├── 09_solvers/            # 수치 솔버 / Solvers
│   └── 10_projects/           # 종합 프로젝트 / Projects
│
├── Plasma_Physics/        # 플라즈마 물리 Python 예제 / Plasma physics examples (26 files)
│   ├── 01_fundamentals/       # 기초 매개변수 / Fundamental parameters
│   ├── 02_particle_motion/    # 입자 운동 / Particle motion
│   ├── 03_kinetic/            # 운동론 / Kinetic theory
│   ├── 04_waves/              # 플라즈마 파동 / Plasma waves
│   ├── 05_fluid/              # 유체 모델 / Fluid models
│   ├── 06_diagnostics/        # 진단 / Diagnostics
│   └── 07_projects/           # 프로젝트 / Projects
│
├── Python/                # Python 고급 예제 / Advanced Python examples (16 files)
├── Security/              # 보안 Python 예제 / Security examples (16 files)
│   ├── 02_cryptography/       # AES, RSA, ECDSA
│   ├── 03_hashing/            # SHA, bcrypt, HMAC
│   ├── 04_tls/                # TLS 클라이언트, 인증서 / TLS client, certificates
│   ├── 05_authentication/     # OAuth2, JWT, TOTP
│   ├── 06_authorization/      # RBAC 미들웨어 / RBAC middleware
│   ├── 07_owasp/              # 취약 코드 + 수정 / Vulnerable + fixed code
│   ├── 08_injection/          # SQL injection, XSS 방어 / defense
│   ├── 10_api_security/       # Rate limiter, CORS
│   ├── 11_secrets/            # Vault, .env 관리 / management
│   ├── 13_testing/            # Bandit, 보안 테스트 / security testing
│   ├── 15_secure_api/         # Flask 보안 API 프로젝트 / Secure API project
│   └── 16_scanner/            # 취약점 스캐너 / Vulnerability scanner
│
├── Signal_Processing/      # 신호 처리 Python 예제 / Signal processing examples (19 files)
│   ├── 01_signals_classification.py  # 신호 분류 / Signal classification
│   ├── 02_convolution.py             # 컨볼루션 / Convolution
│   ├── ...                           # 03-15: 푸리에, 샘플링, FFT, Z변환, 필터, 적응, 영상
│   └── README.md
│
├── Software_Engineering/   # 소프트웨어 공학 예제 / SE examples (11 files)
│   ├── 04_user_story_template.md   # 사용자 스토리 / User stories
│   ├── 05_uml_class_diagram.py     # UML 클래스 다이어그램 / UML class diagram
│   ├── 06_estimation_calculator.py # 추정 계산기 / Estimation calculator
│   ├── 07_code_metrics.py          # 코드 메트릭 / Code metrics
│   ├── 10_gantt_chart.py           # 간트 차트 / Gantt chart + CPM
│   ├── 11_tech_debt_tracker.py     # 기술 부채 / Tech debt tracker
│   ├── 13_ci_cd_pipeline.yml       # CI/CD 파이프라인 / GitHub Actions
│   └── 14_adr_template.md          # ADR 템플릿 / ADR template
│
├── Shell_Script/           # Bash 스크립팅 예제 / scripting examples (30 files)
│   ├── 02_parameter_expansion/  # 매개변수 확장 / Parameter expansion
│   ├── 03_arrays/               # 배열 / Arrays
│   ├── 05_function_library/     # 함수 라이브러리 / Function libraries
│   ├── 06_io_redirection/       # I/O 리다이렉션 / I/O redirection
│   ├── 08_regex/                # 정규표현식 / Regex
│   ├── 09_process_management/   # 프로세스 관리 / Process management
│   ├── 10_error_handling/       # 에러 처리 / Error handling
│   ├── 11_argument_parsing/     # 인자 파싱 / Argument parsing
│   ├── 13_testing/              # 테스팅 / Testing (Bats)
│   ├── 14_task_runner/          # 태스크 러너 / Task runner
│   ├── 15_deployment/           # 배포 자동화 / Deployment
│   └── 16_monitoring/           # 모니터링 / Monitoring
│
├── System_Design/          # 시스템 설계 시뮬레이터 / System design simulators (14 files)
│   ├── 04_load_balancer.py ~ 08_sharding_sim.py       # 로드밸런서, 캐시, 해싱, 샤딩
│   ├── 10_eventual_consistency.py ~ 11_message_queue.py # 일관성, 메시지 큐
│   ├── 14_circuit_breaker.py ~ 16_raft_sim.py          # 서킷브레이커, 사가, Raft
│   └── 17_url_shortener.py ~ 20_inverted_index.py      # URL 단축, 메트릭, 역인덱스
│
└── Web_Development/        # HTML/CSS/JS 프로젝트 / projects (50 files)
    ├── 15_project_spa/         # Single Page Application demo
    │   ├── index.html          # Main HTML
    │   ├── style.css           # Responsive styles with animations
    │   ├── router.js           # Hash-based SPA router
    │   └── app.js              # Application logic and components
```

**총 예제 파일 / Total example files: ~872** (40개 토픽 / 40 topics)

## 빌드 방법 / How to Build

### C/C++ 예제 / C/C++ Examples (Algorithm)

```bash
cd examples/Algorithm/c
make          # 전체 빌드 / Build all
make clean    # 정리 / Clean

cd examples/Algorithm/cpp
make          # 전체 빌드 / Build all
```

### C 프로그래밍 예제 / C Programming Examples

```bash
cd examples/C_Programming/<project>
make          # 프로젝트별 빌드 / Build per project
make clean    # 정리 / Clean
```

### C++ 예제 / C++ Examples

```bash
cd examples/CPP
make          # 전체 빌드 / Build all
make modern   # Modern C++ 빌드 / Build modern C++
make run-01_modern_cpp  # 실행 / Run example
make clean    # 정리 / Clean
```

### Python 예제 / Python Examples

```bash
python examples/Algorithm/python/01_complexity.py
python examples/Reinforcement_Learning/06_q_learning.py
```

### Jupyter 노트북 / Jupyter Notebooks (Machine_Learning)

```bash
cd examples/Machine_Learning
jupyter notebook
```

## 토픽별 예제 목록 / Examples by Topic

| 토픽 / Topic | 파일 수 / Files | 언어 / Language | 설명 / Description |
|--------------|-----------------|-----------------|-------------------|
| Algorithm | 92 | Python, C, C++ | 자료구조, 알고리즘 / Data structures, Algorithms |
| C_Programming | 88 | C | 시스템 프로그래밍 프로젝트, 네트워크, IPC, 테스팅 / System programming projects, Network, IPC, Testing |
| Claude_Ecosystem | 16 | Python/JSON | Claude Code, MCP 서버, Agent SDK / Claude Code, MCP Servers, Agent SDK |
| Compiler_Design | 13 | Python | 렉서, 파서, AST, 타입 체커, 바이트코드 VM, 레지스터 할당, 미니 컴파일러 / Lexer, Parser, AST, Type Checker, Bytecode VM, Register Allocator, Mini Compiler |
| Computer_Architecture | 12 | Python | 수 체계, IEEE754, 논리 게이트, ALU, CPU, 파이프라인, 분기 예측, 캐시, TLB / Number systems, Logic gates, ALU, CPU, Pipeline, Branch predictor, Cache, TLB |
| Control_Theory | 12 | Python | 전달함수, 근궤적, 보드/나이퀴스트, PID, 상태공간, LQR, 칼만, 디지털제어 / Transfer functions, Root locus, Bode/Nyquist, PID, State-space, LQR, Kalman, Digital control |
| Computer_Vision | 23 | Python | OpenCV, 이미지 처리 / Image processing |
| CPP | 37 | C++ | Modern C++, STL, 스마트 포인터, 스레딩, 디자인 패턴, CMake, 학생관리 프로젝트 / Modern C++, STL, Smart Pointers, Threading, Design Patterns, CMake, Student Management Project |
| Data_Engineering | 33 | Python/SQL/YAML/JSON | Airflow, Spark, Kafka, CDC, Lakehouse, 실습 파이프라인 / Practical Pipeline |
| Data_Science | 27 | Python | NumPy, Pandas, 시각화, 통계학, 베이지안, 인과추론, 생존분석, GARCH, Polars, DuckDB / Visualization, Statistics, Bayesian, Causal, Survival, GARCH, Polars, DuckDB |
| Database_Theory | 15 | Python | 관계형 모델, 정규화, B+트리, MVCC, 쿼리 옵티마이저, 2PL, ARIES, 2PC / Relational, Normalization, B+Tree, MVCC, Query Optimizer, 2PL, ARIES, 2PC |
| Deep_Learning | 47 | Python | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, 모델 구현 / Model Implementations |
| Docker | 15 | Docker/YAML | Multi-stage build, Compose, Kubernetes, CI/CD |
| Formal_Languages | 9 | Python | DFA, NFA, 정규식, 펌핑 보조정리, CFG/CYK, PDA, 튜링 기계, 촘스키 계층 / DFA, NFA, Regex, Pumping Lemma, CFG/CYK, PDA, Turing Machine, Chomsky Hierarchy |
| Foundation_Models | 8 | Python | Scaling Laws, 토크나이저, LoRA, RAG, 양자화, 증류 / Tokenizer, LoRA, RAG, Quantization, Distillation |
| IoT_Embedded | 17 | Python | Raspberry Pi, MQTT, Edge AI |
| LaTeX | 19 | LaTeX | 문서 조판, 수학, TikZ, Beamer, 참고문헌, 빌드 / Document typesetting, Math, TikZ, Beamer, Bibliography, Build |
| Linux | 3 | Bash | 재해복구, 성능 진단 / Disaster recovery, Performance diagnostics |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 25 | Python/Jupyter | sklearn, 분류, 회귀, 앙상블, Feature Engineering, SHAP/LIME, AutoML, 이상탐지, Production ML, A/B Testing / Classification, Regression, Ensemble, Explainability, AutoML, Anomaly Detection, Production ML, A/B Testing |
| Math_for_AI | 13 | Python | 선형대수, SVD/PCA, 최적화, 확률, 정보이론, 텐서, 그래프, 어텐션 / Linear Algebra, Optimization, Probability, Attention |
| Mathematical_Methods | 13 | Python | 급수, 복소수, 선형대수, 푸리에, ODE/PDE, 특수함수, 텐서 / Series, Complex, Linear Algebra, Fourier, ODE/PDE, Special Functions, Tensors |
| MHD | 32 | Python | 평형, 안정성, 불안정성, 재결합, 난류, 다이나모, 천체물리, 핵융합 / Equilibria, Stability, Reconnection, Turbulence, Dynamo, Fusion |
| MLOps | 32 | Python/YAML/JSON | MLflow, CI/CD, DVC, LLMOps, Feast Feature Store, E2E 실습 프로젝트 / Practical Project |
| Numerical_Simulation | 8 | Python | 수치해석, ODE, Monte Carlo / Numerical analysis |
| Plasma_Physics | 26 | Python | 디바이 차폐, 입자 운동, 란다우 감쇠, 플라즈마 파동, 유체 모델, 진단 / Debye shielding, Particle motion, Landau damping, Waves, Fluid, Diagnostics |
| Programming | 13 | Python | 패러다임, OOP, 함수형, 디자인 패턴, 클린 코드, 테스팅, 동시성 / Paradigms, OOP, Functional, Design Patterns, Clean Code, Testing, Concurrency |
| Networking | 12 | Python | OSI 패킷, 서브넷, 라우팅, TCP 상태머신, DNS, HTTP, 방화벽, IPv6 / OSI packets, Subnets, Routing, TCP state machine, DNS, HTTP, Firewall, IPv6 |
| OS_Theory | 12 | Python | 프로세스, 스레딩, 스케줄링, 동기화, 교착, 페이징, 페이지교체, 파일시스템, IPC / Process, Threading, Scheduling, Sync, Deadlock, Paging, Page Replacement, Filesystem, IPC |
| PostgreSQL | 18 | SQL/Docker/Bash | CRUD, JOIN, 윈도우 함수, FTS, RLS, 함수, 트랜잭션, 트리거, 모니터링, JSON, 최적화, 파티셔닝, 복제 / Window, FTS, RLS, Functions, Transactions, Triggers, Monitoring, JSON, Optimization, Partitioning, Replication |
| Python | 16 | Python | 타입 힌트, 데코레이터, 제너레이터, 비동기, 메타클래스, 테스팅 / Type Hints, Decorators, Generators, Async, Metaclasses, Testing |
| Reinforcement_Learning | 14 | Python | Q-Learning, DQN, PPO, A2C, Model-Based RL, SAC |
| Security | 16 | Python | 암호학, 해싱, TLS, 인증, OWASP, 인젝션, API 보안, CIA, 보안 헤더, 컨테이너, 인시던트 / Cryptography, Hashing, TLS, Auth, OWASP, Injection, API Security, CIA, Headers, Container, Incident |
| Shell_Script | 30 | Bash | 매개변수 확장, 배열, I/O, 정규식, 프로세스, 에러처리, 테스팅, 배포, 모니터링 / Parameter expansion, Arrays, I/O, Regex, Process, Error handling, Testing, Deployment, Monitoring |
| Signal_Processing | 19 | Python | 신호 분류, 컨볼루션, 푸리에, 샘플링, FFT, Z변환, 필터 설계, 다중률, 적응 필터, 스펙트로그램, 영상 필터링 / Signals, Convolution, Fourier, Sampling, FFT, Z-Transform, Filter Design, Multirate, Adaptive, Spectrogram, Image Filtering |
| Software_Engineering | 11 | Python/MD/YAML | 사용자 스토리, UML, 추정, 코드 메트릭, 간트 차트, 기술 부채, CI/CD, ADR, 테스트, 브랜칭 / User Stories, UML, Estimation, Code Metrics, Gantt Chart, Tech Debt, CI/CD, ADR, Testing, Branching |
| System_Design | 14 | Python | 로드밸런서, 레이트리미터, 캐시, 해싱, 샤딩, 일관성, 메시지큐, 서킷브레이커, Raft, URL단축, 역인덱스 / Load Balancer, Rate Limiter, Cache, Hashing, Sharding, Consistency, Message Queue, Circuit Breaker, Raft, URL Shortener, Inverted Index |
| Web_Development | 50 | HTML/CSS/JS/TS | 웹 프로젝트, TypeScript, SPA 라우터 / Web projects, SPA router |

## 예제와 학습 자료 매핑 / Mapping Examples to Study Materials

예제 파일은 `content/` 의 학습 자료와 대응됩니다.

Example files correspond to study materials in `content/`.

예시 / Examples:
- `content/ko/Algorithm/01_Complexity_Analysis.md` → `examples/Algorithm/python/01_complexity.py`
- `content/ko/C_Programming/05_Project_Address_Book.md` → `examples/C_Programming/04_address_book/`
- `content/ko/Machine_Learning/04_Model_Evaluation.md` → `examples/Machine_Learning/03_model_evaluation.ipynb`

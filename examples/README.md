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
├── C_Programming/          # C 프로젝트 예제 / C project examples (56 files)
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
│   ├── 14_bit_operations/
│   ├── 15_gpio_control/    # GPIO 제어 / GPIO control
│   └── 16_serial_comm/     # 시리얼 통신 / Serial communication
│
├── CPP/                    # C++ 고급 예제 / C++ advanced examples (9 files)
│   ├── 01_modern_cpp.cpp       # Modern C++ (C++17/20)
│   ├── 02_stl_containers.cpp   # STL containers & algorithms
│   ├── 03_smart_pointers.cpp   # Smart pointers
│   ├── 04_threading.cpp        # Multithreading
│   ├── 05_design_patterns.cpp  # Design patterns
│   ├── 06_templates.cpp        # Template metaprogramming
│   ├── 07_move_semantics.cpp   # Move semantics
│   └── Makefile                # Build system
│
├── Computer_Vision/        # OpenCV/Python 예제 / examples (21 files)
├── Data_Analysis/          # Pandas/NumPy 예제 / examples (5 files)
├── Data_Engineering/       # Airflow/Spark/Kafka 예제 / examples (6 files)
├── Deep_Learning/          # PyTorch 예제 / examples (28 files)
│   ├── numpy/              # NumPy 기초 구현 / basic implementation (5)
│   └── pytorch/            # PyTorch 구현 / implementation (22)
│
├── Docker/                 # Docker/Kubernetes 예제 / examples (15 files)
│   ├── 01_multi_stage/     # Multi-stage Docker build
│   ├── 02_compose/         # Docker Compose 3-tier stack
│   ├── 03_k8s/             # Kubernetes manifests
│   └── 04_ci_cd/           # GitHub Actions CI/CD pipeline
│
├── IoT_Embedded/           # Raspberry Pi/MQTT 예제 / examples (12 files)
│   ├── edge_ai/            # TFLite, ONNX 추론 / inference
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # 스마트홈, 이미지분석, 클라우드IoT / Smart home, Image analysis, Cloud IoT
│   └── raspberry_pi/       # GPIO, 센서 / sensors
│
├── LLM_and_NLP/            # NLP/HuggingFace 예제 / examples (15 files)
├── Machine_Learning/       # sklearn/Jupyter 예제 / examples (15 files)
├── Math_for_AI/            # AI 수학 Python 예제 / AI math examples (13 files)
├── MLOps/                  # MLflow/서빙 예제 / serving examples (5 files)
├── Numerical_Simulation/   # 수치해석 Python 예제 / Numerical analysis examples (6 files)
├── PostgreSQL/             # SQL 예제 / examples (5 files)
├── Reinforcement_Learning/ # RL Python 예제 / examples (10 files)
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
├── Security/              # 보안 Python 예제 / Security examples (12 files)
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
├── Shell_Script/           # Bash 스크립팅 예제 / scripting examples (27 files)
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
├── Statistics/             # 통계학 Python 예제 / Statistics examples (11 files)
└── Web_Development/        # HTML/CSS/JS 프로젝트 / projects (46 files)
```

**총 예제 파일 / Total example files: 501**

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
| Algorithm | 89 | Python, C, C++ | 자료구조, 알고리즘 / Data structures, Algorithms |
| C_Programming | 56 | C | 시스템 프로그래밍 프로젝트 / System programming projects |
| CPP | 9 | C++ | Modern C++, STL, 스마트 포인터, 스레딩, 디자인 패턴, 템플릿, 이동 시맨틱 / Modern C++, STL, Smart Pointers, Threading, Design Patterns, Templates, Move Semantics |
| Computer_Vision | 21 | Python | OpenCV, 이미지 처리 / Image processing |
| Data_Analysis | 5 | Python | NumPy, Pandas, 시각화 / Visualization |
| Data_Engineering | 6 | Python | Airflow, Spark, Kafka |
| Deep_Learning | 28 | Python | PyTorch, CNN, RNN, Transformer |
| Docker | 15 | Docker/YAML | Multi-stage build, Compose, Kubernetes, CI/CD |
| Foundation_Models | 8 | Python | Scaling Laws, 토크나이저, LoRA, RAG, 양자화, 증류 / Tokenizer, LoRA, RAG, Quantization, Distillation |
| IoT_Embedded | 12 | Python | Raspberry Pi, MQTT, Edge AI |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 15 | Python/Jupyter | sklearn, 분류, 회귀, 클러스터링 / Classification, Regression, Clustering |
| Math_for_AI | 13 | Python | 선형대수, SVD/PCA, 최적화, 확률, 정보이론, 텐서, 그래프, 어텐션 / Linear Algebra, Optimization, Probability, Attention |
| Mathematical_Methods | 13 | Python | 급수, 복소수, 선형대수, 푸리에, ODE/PDE, 특수함수, 텐서 / Series, Complex, Linear Algebra, Fourier, ODE/PDE, Special Functions, Tensors |
| MHD | 32 | Python | 평형, 안정성, 불안정성, 재결합, 난류, 다이나모, 천체물리, 핵융합 / Equilibria, Stability, Reconnection, Turbulence, Dynamo, Fusion |
| MLOps | 5 | Python | MLflow, 모델 서빙, 드리프트 감지 / Model serving, Drift detection |
| Numerical_Simulation | 6 | Python | 수치해석, ODE, Monte Carlo / Numerical analysis |
| Plasma_Physics | 26 | Python | 디바이 차폐, 입자 운동, 란다우 감쇠, 플라즈마 파동, 유체 모델, 진단 / Debye shielding, Particle motion, Landau damping, Waves, Fluid, Diagnostics |
| PostgreSQL | 5 | SQL | CRUD, JOIN, 윈도우 함수 / Window functions |
| Python | 16 | Python | 타입 힌트, 데코레이터, 제너레이터, 비동기, 메타클래스, 테스팅 / Type Hints, Decorators, Generators, Async, Metaclasses, Testing |
| Reinforcement_Learning | 10 | Python | Q-Learning, DQN, PPO, A2C |
| Security | 12 | Python | 암호학, 해싱, TLS, 인증, OWASP, 인젝션 방어, API 보안, 취약점 스캐너 / Cryptography, Hashing, TLS, Auth, OWASP, Injection defense, API Security, Vulnerability Scanner |
| Shell_Script | 27 | Bash | 매개변수 확장, 배열, I/O, 정규식, 프로세스, 에러처리, 테스팅, 배포, 모니터링 / Parameter expansion, Arrays, I/O, Regex, Process, Error handling, Testing, Deployment, Monitoring |
| Statistics | 11 | Python | 확률, 추론, 회귀, 베이지안, 시계열, 다변량 / Probability, Inference, Regression, Bayesian, Time Series, Multivariate |
| Web_Development | 46 | HTML/CSS/JS/TS | 웹 프로젝트, TypeScript / Web projects |

## 예제와 학습 자료 매핑 / Mapping Examples to Study Materials

예제 파일은 `content/` 의 학습 자료와 대응됩니다.

Example files correspond to study materials in `content/`.

예시 / Examples:
- `content/ko/Algorithm/01_Complexity_Analysis.md` → `examples/Algorithm/python/01_complexity.py`
- `content/ko/C_Programming/05_Project_Address_Book.md` → `examples/C_Programming/04_address_book/`
- `content/ko/Machine_Learning/04_Model_Evaluation.md` → `examples/Machine_Learning/03_model_evaluation.ipynb`

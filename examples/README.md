# Examples (예제 코드)

학습 자료에 대응하는 실행 가능한 예제 코드 모음입니다.

A collection of executable example code corresponding to the study materials.

## 디렉토리 구조 / Directory Structure

```
examples/
├── Algorithm/              # Python, C, C++ 예제 / examples (87 files)
│   ├── python/             # Python 구현 / implementation (29)
│   ├── c/                  # C 구현 / implementation (29 + Makefile)
│   └── cpp/                # C++ 구현 / implementation (29 + Makefile)
│
├── C_Programming/          # C 프로젝트 예제 / C project examples (41 files)
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
│   └── 14_bit_operations/
│
├── Computer_Vision/        # OpenCV/Python 예제 / examples (20 files)
├── Data_Analysis/          # Pandas/NumPy 예제 / examples (5 files)
├── Data_Engineering/       # Airflow/Spark/Kafka 예제 / examples (6 files)
├── Deep_Learning/          # PyTorch 예제 / examples (24 files)
│   ├── numpy/              # NumPy 기초 구현 / basic implementation (5)
│   └── pytorch/            # PyTorch 구현 / implementation (19)
│
├── IoT_Embedded/           # Raspberry Pi/MQTT 예제 / examples (12 files)
│   ├── edge_ai/            # TFLite, ONNX 추론 / inference
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # 스마트홈, 이미지분석, 클라우드IoT / Smart home, Image analysis, Cloud IoT
│   └── raspberry_pi/       # GPIO, 센서 / sensors
│
├── LLM_and_NLP/            # NLP/HuggingFace 예제 / examples (15 files)
├── Machine_Learning/       # sklearn/Jupyter 예제 / examples (14 files)
├── Math_for_AI/            # AI 수학 Python 예제 / AI math examples (12 files)
├── MLOps/                  # MLflow/서빙 예제 / serving examples (5 files)
├── Numerical_Simulation/   # 수치해석 Python 예제 / Numerical analysis examples (6 files)
├── PostgreSQL/             # SQL 예제 / examples (5 files)
├── Reinforcement_Learning/ # RL Python 예제 / examples (10 files)
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
└── Web_Development/        # HTML/CSS/JS 프로젝트 / projects (17 files)
```

**총 예제 파일 / Total example files: 306**

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
| Algorithm | 87 | Python, C, C++ | 자료구조, 알고리즘 / Data structures, Algorithms |
| C_Programming | 41 | C | 시스템 프로그래밍 프로젝트 / System programming projects |
| Computer_Vision | 20 | Python | OpenCV, 이미지 처리 / Image processing |
| Data_Analysis | 5 | Python | NumPy, Pandas, 시각화 / Visualization |
| Data_Engineering | 6 | Python | Airflow, Spark, Kafka |
| Deep_Learning | 24 | Python | PyTorch, CNN, RNN, Transformer |
| IoT_Embedded | 12 | Python | Raspberry Pi, MQTT, Edge AI |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 14 | Python/Jupyter | sklearn, 분류, 회귀, 클러스터링 / Classification, Regression, Clustering |
| Math_for_AI | 12 | Python | 선형대수, SVD/PCA, 최적화, 확률, 정보이론, 텐서, 그래프, 어텐션 / Linear Algebra, Optimization, Probability, Attention |
| MLOps | 5 | Python | MLflow, 모델 서빙, 드리프트 감지 / Model serving, Drift detection |
| Numerical_Simulation | 6 | Python | 수치해석, ODE, Monte Carlo / Numerical analysis |
| PostgreSQL | 5 | SQL | CRUD, JOIN, 윈도우 함수 / Window functions |
| Reinforcement_Learning | 10 | Python | Q-Learning, DQN, PPO, A2C |
| Shell_Script | 27 | Bash | 매개변수 확장, 배열, I/O, 정규식, 프로세스, 에러처리, 테스팅, 배포, 모니터링 / Parameter expansion, Arrays, I/O, Regex, Process, Error handling, Testing, Deployment, Monitoring |
| Web_Development | 17 | HTML/CSS/JS/TS | 웹 프로젝트, TypeScript / Web projects |

## 예제와 학습 자료 매핑 / Mapping Examples to Study Materials

예제 파일은 `content/` 의 학습 자료와 대응됩니다.

Example files correspond to study materials in `content/`.

예시 / Examples:
- `content/ko/Algorithm/01_Complexity_Analysis.md` → `examples/Algorithm/python/01_complexity.py`
- `content/ko/C_Programming/05_Project_Address_Book.md` → `examples/C_Programming/04_address_book/`
- `content/ko/Machine_Learning/04_Model_Evaluation.md` → `examples/Machine_Learning/03_model_evaluation.ipynb`

# Examples (예제 코드)

학습 자료에 대응하는 실행 가능한 예제 코드 모음입니다.

## 디렉토리 구조

```
examples/
├── Algorithm/              # Python, C, C++ 예제 (90개 파일)
│   ├── python/             # Python 구현 (29개)
│   ├── c/                  # C 구현 (29개 + Makefile)
│   └── cpp/                # C++ 구현 (29개 + Makefile)
│
├── C_Programming/          # C 프로젝트 예제 (19개 파일)
│   ├── 02_calculator/
│   ├── 03_number_guess/
│   ├── 05_dynamic_array/
│   ├── 06_linked_list/
│   ├── 12_multithread/
│   ├── 13_embedded_basic/
│   ├── 14_bit_operations/
│   └── ...
│
├── Computer_Vision/        # OpenCV/Python 예제 (21개 파일)
├── Data_Analysis/          # Pandas/NumPy 예제 (5개 파일)
├── Data_Engineering/       # Airflow/Spark/Kafka 예제 (6개 파일)
├── Deep_Learning/          # PyTorch 예제 (25개 파일)
├── IoT_Embedded/           # Raspberry Pi/MQTT 예제 (5개 파일)
├── LLM_and_NLP/            # NLP/HuggingFace 예제 (15개 파일)
├── Machine_Learning/       # sklearn/Jupyter 예제 (7개 파일)
├── MLOps/                  # MLflow/서빙 예제 (5개 파일)
├── Numerical_Simulation/   # 수치해석 Python 예제 (6개 파일)
├── PostgreSQL/             # SQL 예제 (5개 파일)
├── Reinforcement_Learning/ # RL Python 예제 (4개 파일)
└── Web_Development/        # HTML/CSS/JS 프로젝트 (46개 파일)
```

## 빌드 방법

### C/C++ 예제 (Algorithm)

```bash
cd examples/Algorithm/c
make          # 전체 빌드
make clean    # 정리

cd examples/Algorithm/cpp
make          # 전체 빌드
```

### C 프로그래밍 예제

```bash
cd examples/C_Programming
make          # 전체 빌드
make clean    # 정리
```

### Python 예제

```bash
python examples/Algorithm/python/01_complexity.py
```

## 예제와 학습 자료 매핑

예제 파일은 `content/` 의 학습 자료와 1:1 대응됩니다.

예시:
- `content/ko/Algorithm/01_복잡도_분석.md` → `examples/Algorithm/python/01_complexity.py`
- `content/ko/C_Programming/02_프로젝트_계산기.md` → `examples/C_Programming/02_calculator/`

# Study Content (학습 자료)

학습 자료 Markdown 파일 모음입니다. 한국어(ko)와 영어(en) 두 언어로 관리됩니다.

A collection of study material Markdown files. Maintained in both Korean (ko) and English (en).

## 디렉토리 구조 / Directory Structure

```
content/
├── ko/                    # 한국어 자료 (원본) / Korean (original)
│   ├── Algorithm/         # 29개 레슨 / 29 lessons
│   ├── C_Programming/     # 20개 레슨 / 20 lessons
│   └── ...                # 총 28개 토픽 / 28 topics total
│
└── en/                    # 영어 자료 (번역본) / English (translated)
    ├── Algorithm/
    ├── C_Programming/
    └── ...                # 총 28개 토픽 (번역 완료) / 28 topics (fully translated)
```

## 토픽 목록 / Topic List

각 토픽의 `00_Overview.md`에서 학습 로드맵과 상세 목차를 확인할 수 있습니다.

Check the `00_Overview.md` in each topic for learning roadmaps and detailed table of contents.

| 폴더 / Folder | 내용 / Description | 레슨 수 / Lessons |
|---------------|-------------------|-------------------|
| [Algorithm/](./ko/Algorithm/00_Overview.md) | 알고리즘/자료구조, 코딩테스트 준비 / Algorithms, Data Structures | 29 |
| [C_Programming/](./ko/C_Programming/00_Overview.md) | C 언어 기초 ~ 임베디드, 디버깅, 포인터 심화 / C Basics to Embedded | 20 |
| [Cloud_Computing/](./ko/Cloud_Computing/00_Overview.md) | AWS/GCP, EC2, Lambda, S3, VPC, IAM, Terraform | 17 |
| [Computer_Architecture/](./ko/Computer_Architecture/00_Overview.md) | 컴퓨터 구조, CPU, 메모리, 병렬처리 / Computer Architecture | 18 |
| [Computer_Vision/](./ko/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 3D비전, SLAM / Image Processing | 23 |
| [CPP/](./ko/CPP/00_Overview.md) | C++ 입문 ~ 모던 C++, 디자인패턴 / C++ to Modern C++ | 18 |
| [Data_Analysis/](./ko/Data_Analysis/00_Overview.md) | NumPy, Pandas, 시각화, 통계분석 / Visualization, Statistics | 11 |
| [Data_Engineering/](./ko/Data_Engineering/00_Overview.md) | 데이터 파이프라인, Airflow, Spark, Kafka, dbt / Data Pipelines | 14 |
| [Deep_Learning/](./ko/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, ViT | 24 |
| [Docker/](./ko/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD | 10 |
| [Foundation_Models/](./ko/Foundation_Models/00_Overview.md) | Scaling Laws, LLaMA, Mistral, DINOv2, SAM, PEFT, RAG | 25 |
| [Git/](./ko/Git/00_Overview.md) | Git, GitHub, 워크플로우, 모노레포 / Workflows, Monorepo | 10 |
| [IoT_Embedded/](./ko/IoT_Embedded/00_Overview.md) | Raspberry Pi, MQTT, Edge AI, 클라우드 IoT / Cloud IoT | 12 |
| [Linux/](./ko/Linux/00_Overview.md) | Linux 기초 ~ HA 클러스터, 트러블슈팅 / HA Cluster, Troubleshooting | 26 |
| [LLM_and_NLP/](./ko/LLM_and_NLP/00_Overview.md) | NLP, BERT, GPT, HuggingFace, RAG, LangChain, RLHF, Agent | 16 |
| [Machine_Learning/](./ko/Machine_Learning/00_Overview.md) | 회귀, 분류, 앙상블, 클러스터링, sklearn / Regression, Classification | 14 |
| [MLOps/](./ko/MLOps/00_Overview.md) | MLflow, Kubeflow, 모델 서빙, 드리프트 감지 / Model Serving, Drift | 12 |
| [Model_Implementations/](./ko/Model_Implementations/00_Overview.md) | 딥러닝 모델 from-scratch 구현 / DL Models from Scratch | 12 |
| [Networking/](./ko/Networking/00_Overview.md) | 네트워크, OSI/TCP-IP, 라우팅, 보안 / Routing, Security | 17 |
| [Numerical_Simulation/](./ko/Numerical_Simulation/00_Overview.md) | 수치해석, ODE/PDE, CFD, FDTD, MHD, 몬테카를로 / Monte Carlo | 20 |
| [OS_Theory/](./ko/OS_Theory/00_Overview.md) | 운영체제, 프로세스, 메모리, 파일시스템 / OS, Processes, Memory | 18 |
| [PostgreSQL/](./ko/PostgreSQL/00_Overview.md) | SQL 기초 ~ 복제, 파티셔닝 / SQL to Replication, Partitioning | 18 |
| [Python/](./ko/Python/00_Overview.md) | 파이썬 고급 문법, 테스트, 패키징 / Advanced Python, Testing | 16 |
| [Reinforcement_Learning/](./ko/Reinforcement_Learning/00_Overview.md) | MDP, Q-Learning, DQN, PPO, Multi-Agent RL | 12 |
| [Spanish/](./ko/Spanish/00_Overview.md) | 스페인어 문법/어휘, 접속법, 지역 변형 / Grammar, Vocabulary | 11 |
| [Statistics/](./ko/Statistics/00_Overview.md) | 심화통계, 베이지안, 시계열, 실험설계 / Bayesian, Time Series | 14 |
| [System_Design/](./ko/System_Design/00_Overview.md) | 시스템 설계, 확장성, 분산시스템 / Scalability, Distributed | 18 |
| [Web_Development/](./ko/Web_Development/00_Overview.md) | HTML, CSS, JS, TypeScript, SEO, Flask | 16 |

## 보기 방법 / How to View

1. **웹 뷰어 / Web Viewer**: `viewer/` 폴더의 Flask 앱 실행 / Run Flask app in `viewer/`
2. **직접 열기 / Direct Open**: VS Code 등에서 Markdown 파일 열기 / Open Markdown files in VS Code
3. **GitHub**: GitHub에서 직접 렌더링 / Render directly on GitHub

## 파일 명명 규칙 / File Naming Convention

- `00_Overview.md` - 토픽 개요 및 목차 / Topic overview and table of contents
- `01_*.md`, `02_*.md`, ... - 순차적 레슨 (숫자 접두사로 순서 지정) / Sequential lessons (numbered prefix)
- 토픽 폴더명 / Topic folders: `Upper_Snake_Case` (예 / e.g., `C_Programming`, `Deep_Learning`)
- 레슨 파일명 / Lesson files: 영문, 공백은 언더스코어로 대체 / English, spaces replaced with underscores (예 / e.g., `01_Getting_Started.md`)

## 예제 코드 / Example Code

예제 코드 파일은 프로젝트 루트의 `examples/` 폴더에 위치합니다.

Example code files are located in the `examples/` folder at the project root.

## 다국어 정책 / Multilingual Policy

- **한국어(ko)**: 원본 자료. 모든 토픽에 존재
  **Korean (ko)**: Original materials. Present in all topics
- **영어(en)**: 번역본. 28개 토픽 모두 번역 완료
  **English (en)**: Translated. All 28 topics fully translated
- 신규 자료 작성 시 한/영 두 버전 동시 작성 원칙
  New materials are written in both Korean and English simultaneously

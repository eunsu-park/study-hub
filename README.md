# Study Hub

직접 만들어보지 않으면 안다고 할 수 없다는 믿음으로 시작한 프로젝트입니다.
프로그래밍 기초부터 플라즈마 물리학까지, 품질 검사를 통과한 학습 자료를 공개합니다.

This project started from a simple belief: you don't truly understand something until you can build it from scratch.
From programming fundamentals to plasma physics — curated study materials that have passed quality review.

> This repository contains curated content selected from a private working repository after passing quality checks.
> It starts empty and grows as content passes quality review.

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
| [Docker](./content/en/Docker/00_Overview.md) | Docker, Kubernetes, Helm, CI/CD, 컨테이너 네트워킹 | 16 |
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
| [CPP_Advanced](./content/en/CPP_Advanced/00_Overview.md) | 고급 C++: 템플릿, 모던 C++11~23, 동시성, 디자인 패턴 | 17 |
| [Machine_Learning](./content/en/Machine_Learning/00_Overview.md) | 회귀, 앙상블, SVM, 클러스터링, SHAP/LIME, AutoML, Symbolic Regression | 24 |
| [Python_Advanced](./content/en/Python_Advanced/00_Overview.md) | Python 고급: 데코레이터, 메타클래스, async, 디스크립터, 함수형, 성능 최적화 | 14 |

### Tier 3 — Advanced (고급)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| [Algorithm](./content/en/Algorithm/00_Overview.md) | 알고리즘/자료구조, 정렬, 그래프, DP, HLD, LCT, PST | 32 |
| [Computer_Vision](./content/en/Computer_Vision/00_Overview.md) | OpenCV, 이미지처리, 객체검출, 세그멘테이션, 3D비전, NeRF, SLAM | 31 |
| [Deep_Learning](./content/en/Deep_Learning/00_Overview.md) | PyTorch, CNN, RNN, Transformer, GAN, Diffusion, Few-Shot, TTA | 47 |

### Tier 4 — Expert (전문)

| Topic / 토픽 | Description / 설명 | Lessons / 레슨 |
|---|---|---|
| *(no content published yet / 아직 게시된 콘텐츠 없음)* | | |

## Learning Paths / 학습 경로

| Path / 경로 | Topics / 토픽 |
|---|---|
| Python Developer / Python 개발자 | Programming → Python_Basics → Python_Advanced |
| Systems Programmer / 시스템 프로그래머 | Programming → C_Basics → C_Advanced → CPP_Basics → CPP_Advanced |
| CV Engineer / 컴퓨터 비전 엔지니어 | Programming → Python_Basics → Python_Advanced → Machine_Learning → Deep_Learning → Computer_Vision |
| ML Engineer / 머신러닝 엔지니어 | Programming → Python_Basics → Python_Advanced → Machine_Learning → Deep_Learning |
| Linux & DevOps | Linux → Shell_Script → Git → Docker |

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

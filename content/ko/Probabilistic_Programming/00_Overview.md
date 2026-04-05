# 확률적 프로그래밍 학습 가이드

## 소개

이 폴더는 **확률적 프로그래밍**에 대한 종합 가이드를 제공하며, 베이지안 추론 이론과 현대 PPL 프레임워크(PyMC, Stan, Pyro/NumPyro)를 활용한 실습을 결합합니다. 기초적인 베이지안 사고방식부터 정규화 플로우, 베이지안 딥러닝, 인과 추론 등 고급 주제까지 체계적으로 다룹니다.

## 대상 독자

- **Probability_and_Statistics**와 **Machine_Learning** 폴더를 완료한 학습자
- Python, NumPy, 기본 확률론(베이즈 정리, 분포)에 익숙한 독자
- 엄격하고 구현 중심의 확률적 모델링 교육을 원하는 모든 분

## 학습 로드맵

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   기초 과정      │────▶│   핵심 PPL       │────▶│   고급 추론      │
│    L01-L03       │     │    L04-L08       │     │    L09-L11       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   캡스톤         │◀────│   응용           │◀────│  딥베이즈 &      │
│    L18           │     │    L15-L17       │     │  최신 PPL        │
└──────────────────┘     └──────────────────┘     │    L12-L14       │
                                                   └──────────────────┘
```

**권장 학습 경로**:
1. 기초 과정(L01-L03)으로 베이지안 사고, 그래프 모델, MCMC 마스터
2. 핵심 PPL(L04-L08)로 PyMC, 계층 모델, 회귀, Stan, VI 학습
3. 고급 추론(L09-L11)으로 가우시안 프로세스, 시계열, 최적화 학습
4. 딥베이즈 & 최신 PPL(L12-L14)로 Pyro, 정규화 플로우, BNN 탐구
5. 응용(L15-L17)으로 인과 추론, 모델 비교, 불확실성 정량화 학습
6. 캡스톤 프로젝트(L18)로 지식 적용

## 선수 과목

- **Probability_and_Statistics**: 분포, 베이즈 정리, MLE/MAP, 가설검정
- **Machine_Learning**: 회귀, 분류, 교차검증, 경사 하강법
- **Python**: NumPy, SciPy, matplotlib 활용 능력

## 사용 프레임워크

| 프레임워크 | 백엔드 | 핵심 기능 | 레슨 |
|-----------|--------|----------|------|
| PyMC 5.x | PyTensor | Python 친화적 API, ArviZ 통합 | L04, L05, L06, L10 |
| Stan / CmdStanPy | C++ | HMC/NUTS 표준 구현 | L07 |
| Pyro / NumPyro | PyTorch / JAX | 딥 확률 모델, SVI | L12 |
| ArviZ | - | 진단 및 시각화 | L04, L16 |

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|------|--------|--------|------|
| **블록 1: 기초 과정** |
| L01 | `01_Bayesian_Thinking.md` | ⭐⭐ | 베이즈 정리, 사전/사후/가능도, 켤레 사전분포 |
| L02 | `02_Probabilistic_Graphical_Models.md` | ⭐⭐⭐ | 베이지안 네트워크, 마르코프 랜덤 필드, d-분리 |
| L03 | `03_MCMC_Fundamentals.md` | ⭐⭐⭐ | 메트로폴리스-해스팅스, 깁스 샘플링, 수렴 진단 |
| **블록 2: 핵심 PPL** |
| L04 | `04_PyMC_Introduction.md` | ⭐⭐ | PyMC 모델 구축, 샘플링, 추적 분석, ArviZ |
| L05 | `05_Hierarchical_Models.md` | ⭐⭐⭐ | 다수준 모델, 부분 풀링, 축소 추정 |
| L06 | `06_Bayesian_Regression.md` | ⭐⭐ | 선형 회귀, GLM, 로버스트 회귀, 모델 비교 |
| L07 | `07_Stan_and_CmdStanPy.md` | ⭐⭐⭐ | Stan 언어, CmdStanPy 인터페이스, HMC/NUTS 상세 |
| L08 | `08_Variational_Inference.md` | ⭐⭐⭐ | ELBO, 평균장 VI, ADVI, MCMC 비교 |
| **블록 3: 고급 추론** |
| L09 | `09_Gaussian_Processes.md` | ⭐⭐⭐ | GP 회귀, 커널, 하이퍼파라미터 최적화, 희소 GP |
| L10 | `10_Bayesian_Time_Series.md` | ⭐⭐⭐ | 구조적 시계열, Prophet, 상태공간 모델 |
| L11 | `11_Bayesian_Optimization.md` | ⭐⭐⭐ | 대리 모델, 획득 함수, 하이퍼파라미터 튜닝 |
| **블록 4: 딥베이즈 & 최신 PPL** |
| L12 | `12_Pyro_and_NumPyro.md` | ⭐⭐⭐ | Pyro 모델 프리미티브, 이펙트 핸들러, NumPyro JAX 백엔드 |
| L13 | `13_Normalizing_Flows.md` | ⭐⭐⭐⭐ | 플로우 기반 모델, RealNVP, 신경 스플라인 플로우 |
| L14 | `14_Bayesian_Deep_Learning.md` | ⭐⭐⭐⭐ | BNN, MC 드롭아웃, Bayes by Backprop, 불확실성 분해 |
| **블록 5: 응용** |
| L15 | `15_Causal_Inference.md` | ⭐⭐⭐ | 구조적 인과 모델, do-미적분, 백도어/프론트도어 기준 |
| L16 | `16_Model_Comparison.md` | ⭐⭐⭐ | WAIC, LOO-CV, 베이즈 인자, 사후 예측 검사 |
| L17 | `17_Uncertainty_Quantification.md` | ⭐⭐⭐ | 교정, 등각 예측, 불확실성 하의 의사결정 |
| **블록 6: 캡스톤** |
| L18 | `18_Capstone_Applied_Bayesian.md` | ⭐⭐⭐⭐ | 종합 프로젝트: A/B 테스트, 임상시험, 추천 시스템 |

## 사용 방법

1. `content/ko/Probabilistic_Programming/`에서 레슨을 읽습니다 (영어는 `en/`)
2. `examples/Probabilistic_Programming/`에서 예제를 실행합니다
3. `exercises/Probabilistic_Programming/`에서 연습문제를 풀어봅니다
4. 사전분포, 가능도, 모델 구조를 변경하며 실험합니다

## 환경 설정

```bash
pip install pymc arviz numpy scipy matplotlib pandas
pip install cmdstanpy  # 이후: install_cmdstan
pip install numpyro jax jaxlib
# 선택: pip install pyro-ppl torch
```

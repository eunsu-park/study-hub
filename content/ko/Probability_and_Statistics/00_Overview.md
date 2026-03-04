# 확률과 통계 학습 가이드

## 소개

**확률과 통계 학습 가이드**에 오신 것을 환영합니다! 이 토픽은 확률 이론 (probability theory), 통계적 추론 (statistical inference), 확률 과정 (stochastic processes)을 엄밀하게 다루며, Data Science의 응용 통계학과 Math for AI의 ML 중심 확률론 사이의 간격을 메워줍니다.

선수 과목: Python (계산 예제용)

---

## 학습 로드맵

18개 레슨이 체계적인 순서로 구성되어 있습니다:

### 1단계: 확률 기초 (L01--L05)
셈 원리부터 다변량 분포까지의 핵심 확률 이론.

### 2단계: 분포 이론 (L06--L09)
이산 및 연속 분포족, 변환, 다변량 정규분포.

### 3단계: 수렴과 극한 정리 (L10--L11)
수렴 개념, 큰 수의 법칙, 중심극한정리.

### 4단계: 통계적 추론 (L12--L15)
점 추정, 신뢰 구간, 가설 검정, 베이즈 방법.

### 5단계: 응용 방법론 (L16--L18)
비모수적 방법, 회귀분석/분산분석, 확률 과정 입문.

---

## 파일 목록

| 파일 | 주제 | 주요 내용 |
|------|------|----------|
| [01_Combinatorics_and_Counting.md](./01_Combinatorics_and_Counting.md) | 조합론과 셈 | 순열, 조합, 다항 계수, 포함-배제 원리 |
| [02_Probability_Axioms_and_Rules.md](./02_Probability_Axioms_and_Rules.md) | 확률 공리와 법칙 | 표본공간, 사건, 공리, 조건부 확률, 베이즈 정리 |
| [03_Random_Variables_and_Distributions.md](./03_Random_Variables_and_Distributions.md) | 확률변수와 분포 | 확률변수, CDF, PMF/PDF, 이산 vs 연속 |
| [04_Expectation_and_Moments.md](./04_Expectation_and_Moments.md) | 기댓값과 적률 | 기댓값, 분산, MGF, 마르코프/체비셰프/옌센 부등식 |
| [05_Joint_Distributions.md](./05_Joint_Distributions.md) | 결합 분포 | 결합 분포, 주변 분포, 독립성, 공분산, 상관계수 |
| [06_Discrete_Distribution_Families.md](./06_Discrete_Distribution_Families.md) | 이산 분포족 | 베르누이, 이항, 포아송, 기하, 음이항, 초기하 분포 |
| [07_Continuous_Distribution_Families.md](./07_Continuous_Distribution_Families.md) | 연속 분포족 | 균등, 정규, 지수, 감마, 베타, 카이제곱, t, F 분포 |
| [08_Transformations_of_Random_Variables.md](./08_Transformations_of_Random_Variables.md) | 확률변수의 변환 | 변수 변환, 야코비안, MGF 기법, 순서통계량 |
| [09_Multivariate_Normal_Distribution.md](./09_Multivariate_Normal_Distribution.md) | 다변량 정규분포 | MVN 정의, 조건부 분포, 마할라노비스 거리 |
| [10_Convergence_Concepts.md](./10_Convergence_Concepts.md) | 수렴 개념 | 확률 수렴, 분포 수렴, 거의 확실한 수렴, Lp 수렴 |
| [11_Law_of_Large_Numbers_and_CLT.md](./11_Law_of_Large_Numbers_and_CLT.md) | 큰 수의 법칙과 CLT | 약한/강한 큰 수의 법칙, 중심극한정리, 정규 근사 |
| [12_Point_Estimation.md](./12_Point_Estimation.md) | 점 추정 | MLE, 적률법, 충분성, 완비성, UMVUE, 크래머-라오 하한 |
| [13_Interval_Estimation.md](./13_Interval_Estimation.md) | 구간 추정 | 신뢰 구간, 피봇, 부트스트랩 CI |
| [14_Hypothesis_Testing.md](./14_Hypothesis_Testing.md) | 가설 검정 | 네이만-피어슨, 유의성, 검정력, p-값, 우도비 검정 |
| [15_Bayesian_Inference.md](./15_Bayesian_Inference.md) | 베이즈 추론 | 사전/사후 분포, 켤레 사전분포, MAP, 신용 구간 |
| [16_Nonparametric_Methods.md](./16_Nonparametric_Methods.md) | 비모수적 방법 | 부호 검정, 윌콕슨, KDE, 순위 검정, 순열 검정 |
| [17_Regression_and_ANOVA.md](./17_Regression_and_ANOVA.md) | 회귀분석과 분산분석 | 단순/다중 회귀, OLS, F-검정, 일원/이원 분산분석 |
| [18_Stochastic_Processes_Introduction.md](./18_Stochastic_Processes_Introduction.md) | 확률 과정 입문 | 마르코프 체인, 포아송 과정, 랜덤 워크, 정상 과정 |

---

## 이 가이드 활용 방법

1. **순서대로 학습**: 레슨은 서로 연결되므로 번호 순서를 따라 진행하세요
2. **코드로 연습**: 각 레슨에는 표준 라이브러리만 사용하는 Python 예제가 포함되어 있습니다
3. **상호 참조**: 응용 방법은 Data_Science를, ML 응용은 Math_for_AI를 참고하세요
4. **연습문제**: `exercises/Probability_and_Statistics/`의 연습문제를 풀어보세요

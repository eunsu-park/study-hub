# 데이터 과학 학습 가이드

## 소개

**데이터 과학 학습 가이드**에 오신 것을 환영합니다! 이 포괄적인 주제는 현대 데이터 분석을 위한 필수 도구와 통계 방법을 다룹니다. 업계 표준 Python 라이브러리를 사용하여 데이터를 조작하고, 시각화하고, 의미 있는 결론을 도출하는 방법을 배우게 됩니다.

데이터 과학은 다음을 결합합니다:
- **데이터 조작 도구** (NumPy, Pandas) - 구조화된 데이터 처리
- **시각화 기법** (Matplotlib, Seaborn) - 패턴 탐색
- **통계적 추론** (scipy, statsmodels) - 타당한 결론 도출
- **실전 응용** - 실습 프로젝트를 통한 적용

이 주제는 기본 데이터 조작에서 탐색적 자료 분석(EDA)을 거쳐 엄격한 통계적 추론과 고급 모델링 기법까지 안내하도록 설계되었습니다.

---

## 학습 로드맵

29개의 강의는 구조화된 진행을 따릅니다:

### 1단계: 데이터 도구 (L01-L06)
데이터 조작과 전처리를 위한 기본 라이브러리를 숙달합니다.

### 2단계: 탐색적 자료 분석 (L07-L09)
데이터 패턴을 시각화하고, 요약하고, 탐색하는 방법을 배웁니다.

### 3단계: 추론으로의 다리 (L10) 🌉
**중요한 전환**: 언제 그리고 왜 기술통계를 넘어 정식 통계 검정으로 나아가야 하는지 이해합니다.

### 4단계: 통계 기초 (L11-L14)
확률 기초를 구축하고 가설 검정 프레임워크를 배웁니다.

### 5단계: 고급 추론 (L15-L24)
전문화된 기법을 숙달합니다: ANOVA, 회귀, 베이지안 방법, 시계열, 실험 설계.

### 6단계: 실전 통합 (L25)
종합적인 실제 프로젝트에서 모든 것을 적용합니다.

### 7단계: 심화 방법론 (L26-L29)
베이지안 심화, 인과 추론, 생존 분석, 모던 데이터 도구.

---

## 강의 목록

| 강의 | 제목 | 난이도 | 주제 |
|--------|-------|------------|--------|
| 01 | [NumPy 기초](./01_NumPy_Basics.md) | ⭐ | 배열, 인덱싱, 브로드캐스팅, 기본 연산 |
| 02 | [NumPy 고급](./02_NumPy_Advanced.md) | ⭐⭐ | 벡터화, 선형대수, 난수 생성 |
| 03 | [Pandas 기초](./03_Pandas_Basics.md) | ⭐ | Series, DataFrames, 데이터 읽기/쓰기 |
| 04 | [Pandas 데이터 조작](./04_Pandas_Data_Manipulation.md) | ⭐⭐ | 필터링, groupby, 병합, 재구조화 |
| 05 | [Pandas 고급](./05_Pandas_Advanced.md) | ⭐⭐⭐ | MultiIndex, 시계열, 범주형 데이터 |
| 06 | [데이터 전처리](./06_Data_Preprocessing.md) | ⭐⭐ | 결측 데이터, 이상치, 스케일링, 인코딩 |
| 07 | [기술통계 및 EDA](./07_Descriptive_Stats_EDA.md) | ⭐⭐ | 요약 통계, 분포, 상관관계 |
| 08 | [데이터 시각화 기초](./08_Data_Visualization_Basics.md) | ⭐⭐ | Matplotlib 기본, 플롯 유형 |
| 09 | [데이터 시각화 고급](./09_Data_Visualization_Advanced.md) | ⭐⭐⭐ | Seaborn, 복잡한 플롯, 대화형 시각화 |
| **10** | **[EDA에서 추론으로](./10_From_EDA_to_Inference.md)** | **⭐⭐** | **다리 강의**: 모집단 대 표본, 통계적 사고, 검정 선택 |
| 11 | [확률 복습](./11_Probability_Review.md) | ⭐⭐ | 확률 변수, 분포, 기댓값 |
| 12 | [표본추출과 추정](./12_Sampling_and_Estimation.md) | ⭐⭐ | 표본추출 방법, 점추정, 편향/분산 |
| 13 | [신뢰구간](./13_Confidence_Intervals.md) | ⭐⭐⭐ | CI 구성, 해석, 오차 한계 |
| 14 | [가설 검정 고급](./14_Hypothesis_Testing_Advanced.md) | ⭐⭐⭐ | p-값, 제1종/제2종 오류, 검정력 분석 |
| 15 | [분산분석(ANOVA)](./15_ANOVA.md) | ⭐⭐⭐ | 일원, 이원, 사후 검정 |
| 16 | [회귀 분석 고급](./16_Regression_Analysis_Advanced.md) | ⭐⭐⭐ | 다중 회귀, 진단, 정규화 |
| 17 | [일반화 선형 모델](./17_Generalized_Linear_Models.md) | ⭐⭐⭐⭐ | 로지스틱 회귀, 포아송 회귀, GLM 이론 |
| 18 | [베이지안 통계 기초](./18_Bayesian_Statistics_Basics.md) | ⭐⭐⭐ | 베이즈 정리, 사전/사후 확률, 켤레성 |
| 19 | [베이지안 추론](./19_Bayesian_Inference.md) | ⭐⭐⭐⭐ | MCMC, PyMC, 신용구간 |
| 20 | [시계열 기초](./20_Time_Series_Basics.md) | ⭐⭐⭐ | 추세, 계절성, 분해 |
| 21 | [시계열 모델](./21_Time_Series_Models.md) | ⭐⭐⭐⭐ | ARIMA, SARIMA, 예측, 진단 |
| 22 | [다변량 분석](./22_Multivariate_Analysis.md) | ⭐⭐⭐ | PCA, 요인 분석, 군집화 |
| 23 | [비모수 통계](./23_Nonparametric_Statistics.md) | ⭐⭐⭐ | 순위 검정, 부트스트랩, 순열 검정 |
| 24 | [실험 설계](./24_Experimental_Design.md) | ⭐⭐⭐ | A/B 테스팅, 무작위화, DOE 원리 |
| 25 | [실전 프로젝트](./25_Practical_Projects.md) | ⭐⭐⭐⭐ | 종단간 데이터 과학 프로젝트 |
| 26 | [베이지안 심화](./26_Bayesian_Advanced.md) | ⭐⭐⭐⭐ | HMC/NUTS, 변분 추론, 계층 모델, 모델 비교 |
| 27 | [인과 추론](./27_Causal_Inference.md) | ⭐⭐⭐⭐ | DAG, 성향 점수, DID, RDD, 도구 변수 |
| 28 | [생존 분석](./28_Survival_Analysis.md) | ⭐⭐⭐⭐ | Kaplan-Meier, Cox PH, 모수적 모델, 경쟁 위험 |
| 29 | [모던 데이터 도구](./29_Modern_Data_Tools.md) | ⭐⭐⭐ | Polars, DuckDB, Arrow 상호운용, 성능 벤치마크 |

---

## 선수 지식

### 필수 지식
- **Python 기초**: 변수, 함수, 반복문, 조건문
- **기본 수학**: 대수학, 기초 미적분학 (도움이 되지만 필수는 아님)
- **호기심**: "왜?"와 "어떻게 검증할 수 있을까?"를 묻는 의지

### 권장 사항 (필수는 아님)
- Jupyter 노트북에 대한 익숙함
- 과학적 표기법에 대한 기본 이해
- 어떤 프로그래밍 언어든 경험

---

## 환경 설정

### 설치

pip를 사용하여 필요한 라이브러리를 설치합니다:

```bash
# 핵심 데이터 과학 스택
pip install numpy pandas matplotlib seaborn

# 통계 라이브러리
pip install scipy statsmodels

# 선택 사항: 베이지안 추론
pip install pymc arviz

# 선택 사항: 기계 학습 통합
pip install scikit-learn

# 선택 사항: 대화형 시각화
pip install plotly
```

### 설치 확인

모든 라이브러리가 설치되었는지 확인하려면 다음 Python 코드를 실행하세요:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.__version__)
print("Seaborn version:", sns.__version__)
print("SciPy version:", stats.__version__)
print("Statsmodels version:", sm.__version__)
```

### 권장 IDE
- **Jupyter Notebook** 또는 **JupyterLab**: 탐색적 분석에 최적
- **VS Code** with Python 확장: 스크립트 개발에 좋음
- **Google Colab**: 무료 클라우드 환경 (설치 불필요)

---

## 관련 주제

이 주제는 학습 가이드의 다른 영역과 밀접하게 연결됩니다:

### 선수 과목 (권장)
- **[Python](../Python/)**: 먼저 Python 기초를 배우세요
- **[Programming](../Programming/)**: 핵심 프로그래밍 개념

### 다음 단계
- **[Machine Learning](../Machine_Learning/)**: scikit-learn을 사용한 예측 모델링
- **[Deep Learning](../Deep_Learning/)**: PyTorch를 사용한 신경망

### 관련 응용
- **[Data Engineering](../Data_Engineering/)**: 대규모 데이터 파이프라인
- **[MLOps](../MLOps/)**: 프로덕션에 모델 배포

---

## 이 가이드 사용 방법

### 초보자를 위한 방법
1. **L01-L06**으로 시작하여 데이터 조작 기술 구축
2. 제공된 연습 문제와 데이터셋으로 연습
3. **L07-L09**로 시각화 학습
4. **L10을 건너뛰지 마세요!** 추론으로의 중요한 다리입니다
5. 자신의 속도로 추론 주제(L11-L24)를 진행

### 중급 학습자를 위한 방법
1. NumPy/Pandas를 알고 있다면 L01-L09를 훑어보세요
2. **L10을 신중하게 학습**하여 통계적 사고를 확고히 하세요
3. 관심사에 따라 추론 주제(L11-L24)에 집중
4. L25 프로젝트를 완료하여 지식 통합

### 고급 사용자를 위한 방법
1. 특정 기법에 대한 참조로 사용
2. 검정 선택에 대한 결정 프레임워크를 위해 L10 검토
3. 고급 주제(베이지안, GLM, 시계열)에 집중
4. L25 프로젝트를 자신의 도메인에 맞게 조정

---

## 학습 팁

### 능동적 학습
- **함께 코딩**: 단순히 읽지 말고 모든 코드 예제를 실행하세요
- **예제 수정**: 매개변수를 변경하고 다른 데이터셋을 시도하세요
- **"만약에?"를 물어보세요**: 극단적 경우와 가정을 테스트하세요

### 연습용 데이터셋
연습을 위해 다음 내장 데이터셋을 사용하세요:
```python
import seaborn as sns

# 샘플 데이터셋 불러오기
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
diamonds = sns.load_dataset('diamonds')
```

### 핵심 습관
1. **항상 시각화**를 통계 검정 실행 전에 수행
2. **가정 확인** (정규성, 독립성 등)
3. p-값뿐만 아니라 **효과 크기 보고**
4. 주석/마크다운에 **추론 근거 문서화**

---

## 평가 및 프로젝트

### 자기 평가
각 강의에는 다음이 포함됩니다:
- **연습 문제**: 솔루션이 있는 연습 문제
- **개념 질문**: 이해도 테스트
- **코드 챌린지**: 새로운 시나리오에 기법 적용

### 캡스톤 프로젝트 (L25)
마지막 강의에는 완전한 프로젝트가 포함됩니다:
1. **소매 판매 분석**: 시계열 예측
2. **A/B 테스트 평가**: 가설 검정 워크플로
3. **설문 조사 데이터 분석**: 다변량 기법
4. **예측 모델링**: 회귀 및 분류

---

## 추가 자료

### 책
- **"Python for Data Analysis"** by Wes McKinney (Pandas 창시자)
- **"The Art of Statistics"** by David Spiegelhalter
- **"Statistical Rethinking"** by Richard McElreath

### 온라인 코스
- [Kaggle Learn](https://www.kaggle.com/learn): 무료 대화형 튜토리얼
- [StatQuest](https://statquest.org/): 통계에 대한 비디오 설명
- [Seeing Theory](https://seeing-theory.brown.edu/): 시각적 확률/통계

### 문서
- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Statsmodels Docs](https://www.statsmodels.org/)

---

## 도움 받기

### 학습 중
- 먼저 공식 문서를 확인하세요
- Jupyter에서 `help()` 함수 또는 `?`를 사용하세요
- pandas/numpy 질문은 [Stack Overflow](https://stackoverflow.com/questions/tagged/pandas)에서 검색
- 통계 질문은 [Cross Validated](https://stats.stackexchange.com/)에 물어보세요

### 흔한 문제
- **ImportError**: `pip install --upgrade <라이브러리>`로 라이브러리 재설치
- **DeprecationWarning**: 호환성을 위해 라이브러리 버전 확인
- **MemoryError**: 큰 데이터셋에 대해 더 작은 샘플 또는 청킹 사용

---

## 이 가이드의 철학

### 엄격함과 직관의 균형
우리의 목표:
- **직관 먼저 구축**: 공식 전에 시각적이고 개념적인 이해
- **이론과 실전 연결**: 모든 개념에 코드 예제
- **비판적 사고 강조**: 기법을 *어떻게* 사용할지뿐만 아니라 *언제* 사용할지 알기

### EDA-추론 연결
**10번 강의**는 이 가이드의 핵심입니다. 대부분의 강의는 EDA와 추론을 별개의 주제로 다룹니다. 우리는 **전환**을 강조합니다:
- EDA는 질문을 생성 → 추론은 이를 엄격하게 답변
- 시각화는 패턴을 제안 → 검정은 통제된 오류로 이를 확인
- 기술통계는 표본을 설명 → 추론은 모집단으로 일반화

---

## 내비게이션
- **여기서 시작**: [01_NumPy_Basics](./01_NumPy_Basics.md)
- **중요한 다리**: [10_From_EDA_to_Inference](./10_From_EDA_to_Inference.md)
- **최종 프로젝트**: [25_Practical_Projects](./25_Practical_Projects.md)

---

**데이터 과학 여정을 시작할 준비가 되셨나요? NumPy 기초부터 시작해 봅시다!**

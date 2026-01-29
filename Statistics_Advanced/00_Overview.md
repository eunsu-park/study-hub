# 통계학 심화 (Advanced Statistics) - Overview

## 소개

이 폴더는 **고급 통계학**을 체계적으로 학습하기 위한 자료 모음입니다. 기초 통계학(기술통계, 기본적인 가설검정)을 이수한 학습자를 대상으로, 확률론의 이론적 토대부터 일반화 선형모형까지 단계별로 심화 학습할 수 있도록 구성되어 있습니다.

모든 개념은 Python 코드 예제와 함께 제공되며, 실무에서 바로 활용할 수 있는 실용적인 내용을 담고 있습니다.

---

## 파일 목록

| 번호 | 파일명 | 주제 | 주요 내용 |
|------|--------|------|-----------|
| 00 | 00_Overview.md | 학습 로드맵 | 전체 구성, 필수 라이브러리, 학습 순서 |
| 01 | 01_확률론_복습.md | 확률론 복습 | 확률 공리, 확률변수, 분포, 기대값, 중심극한정리 |
| 02 | 02_표본과_추정.md | 표본과 추정 | 표본분포, 표준오차, 점추정, MLE, 적률추정 |
| 03 | 03_신뢰구간.md | 신뢰구간 | 구간추정, t-분포, 비율/분산 신뢰구간, 부트스트랩 |
| 04 | 04_가설검정_심화.md | 가설검정 심화 | 검정력, 효과크기, 표본크기 결정, 다중검정 보정 |
| 05 | 05_분산분석_ANOVA.md | 분산분석 | 일원/이원 ANOVA, F-분포, 사후검정 |
| 06 | 06_회귀분석_심화.md | 회귀분석 심화 | 다중회귀, 진단, 다중공선성, 변수선택 |
| 07 | 07_일반화_선형모형.md | GLM | 로지스틱 회귀, 포아송 회귀, 링크함수 |

---

## 필수 라이브러리

### 핵심 라이브러리

```bash
# 기본 설치
pip install numpy pandas scipy statsmodels

# 시각화
pip install matplotlib seaborn

# 고급 통계 분석
pip install pingouin  # ANOVA, 효과크기, 다양한 검정

# 베이지안 통계 (선택사항)
pip install pymc arviz  # 베이지안 추론 (Python 3.9+)
```

### 라이브러리별 용도

| 라이브러리 | 주요 용도 | 사용 예시 |
|------------|-----------|-----------|
| `scipy.stats` | 확률분포, 기본 검정 | 정규분포, t-검정, 카이제곱 검정 |
| `statsmodels` | 회귀분석, GLM, 시계열 | OLS, 로지스틱 회귀, ANOVA |
| `pingouin` | 고급 검정, 효과크기 | ANOVA, 사후검정, Cohen's d |
| `pymc` | 베이지안 추론 | MCMC, 사후분포 추정 |

### 환경 설정 예시

```python
# 기본 import 설정
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 (Windows)
# plt.rcParams['font.family'] = 'Malgun Gothic'

# 시각화 스타일
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 경고 무시 (필요시)
import warnings
warnings.filterwarnings('ignore')
```

---

## 권장 학습 순서

### Phase 1: 이론적 기초 (1-2주)

```
01_확률론_복습 → 02_표본과_추정 → 03_신뢰구간
```

- 확률론의 핵심 개념 복습
- 추정 이론의 수학적 기반 이해
- 구간추정의 원리와 해석

### Phase 2: 추론 통계 심화 (1-2주)

```
04_가설검정_심화 → 05_분산분석_ANOVA
```

- 검정력 분석과 표본크기 설계
- 다중검정 문제와 해결책
- 그룹 비교를 위한 ANOVA

### Phase 3: 회귀 모형 (2-3주)

```
06_회귀분석_심화 → 07_일반화_선형모형
```

- 회귀모형의 가정과 진단
- 범주형/카운트 데이터를 위한 GLM

### 학습 팁

1. **순서대로 학습**: 앞 장의 개념이 뒤 장의 기초가 됩니다
2. **코드 직접 실행**: 예제 코드를 복사하지 말고 직접 타이핑하세요
3. **데이터 바꿔보기**: 예제 데이터를 변형하여 결과 변화를 관찰하세요
4. **시각화 활용**: 그래프를 통해 통계적 개념을 직관적으로 이해하세요

---

## 선수 지식

### 필수

- **기초 통계학**
  - 기술통계 (평균, 분산, 표준편차)
  - 기본적인 가설검정 (t-검정, 카이제곱 검정)
  - 상관분석, 단순회귀

- **Python 프로그래밍**
  - 기본 문법 (변수, 함수, 조건문, 반복문)
  - NumPy 배열 연산
  - Pandas DataFrame 조작
  - Matplotlib 기본 시각화

### 권장

- **선형대수 기초**
  - 행렬 연산, 역행렬
  - 벡터 내적

- **미적분 기초**
  - 미분의 개념 (최적화 이해에 필요)
  - 적분의 개념 (확률분포 이해에 필요)

---

## 학습 목표

이 과정을 완료하면 다음을 수행할 수 있습니다:

1. **확률론 기반 이해**: 통계적 방법의 수학적 기초를 설명할 수 있다
2. **추정 이론 적용**: MLE와 구간추정을 실제 데이터에 적용할 수 있다
3. **검정 설계**: 검정력 분석을 통해 적절한 표본크기를 결정할 수 있다
4. **ANOVA 분석**: 다중 그룹 비교를 위한 분산분석을 수행할 수 있다
5. **회귀 진단**: 회귀모형의 가정을 검토하고 문제를 해결할 수 있다
6. **GLM 적용**: 다양한 유형의 종속변수에 맞는 모형을 선택할 수 있다

---

## 참고 자료

### 교재
- Casella & Berger, *Statistical Inference* (이론적 깊이)
- Agresti, *Foundations of Linear and Generalized Linear Models*
- James et al., *An Introduction to Statistical Learning* (실용적)

### 온라인 자료
- [scipy.stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [pingouin Documentation](https://pingouin-stats.org/)

---

## 버전 정보

- **최초 작성**: 2026-01-29
- **Python 버전**: 3.9+
- **주요 라이브러리 버전**: scipy >= 1.9, statsmodels >= 0.14, pingouin >= 0.5

# Data Analysis (데이터 분석) 학습 가이드

## 개요

데이터 분석은 데이터로부터 유의미한 정보를 추출하고 인사이트를 도출하는 과정입니다. 이 학습 자료는 Python 기반의 데이터 분석 도구와 기법을 체계적으로 다룹니다.

---

## 학습 로드맵

```
NumPy 기초 → NumPy 고급 → Pandas 기초 → Pandas 데이터 조작 → Pandas 고급
                                              ↓
실전 프로젝트 ← 통계 분석 ← 시각화 고급 ← 시각화 기초 ← 기술통계/EDA ← 데이터 전처리
```

---

## 파일 목록

| 파일 | 주제 | 핵심 내용 |
|------|------|----------|
| 01 | NumPy 기초 | 배열 생성, 인덱싱/슬라이싱, 브로드캐스팅, 기본 연산 |
| 02 | NumPy 고급 | 선형대수, 통계함수, 난수 생성, 성능 최적화 |
| 03 | Pandas 기초 | Series, DataFrame, 데이터 로딩(CSV/Excel/JSON) |
| 04 | Pandas 데이터 조작 | 필터링, 정렬, 그룹화, 병합(merge/join/concat) |
| 05 | Pandas 고급 | 피벗테이블, 멀티인덱스, 시계열 데이터 처리 |
| 06 | 데이터 전처리 | 결측치 처리, 이상치 탐지, 정규화, 인코딩 |
| 07 | 기술통계와 EDA | 기술통계량, 분포 분석, 탐색적 데이터 분석 |
| 08 | 데이터 시각화 기초 | Matplotlib 기초, 다양한 차트 유형 |
| 09 | 데이터 시각화 고급 | Seaborn, 고급 시각화 기법, 대시보드 |
| 10 | 통계 분석 기초 | 확률분포, 가설검정, 신뢰구간, 상관분석 |
| 11 | 실전 프로젝트 | Kaggle 데이터셋 EDA, 종합 실습 |

---

## 환경 설정

### 필수 라이브러리 설치

```bash
# pip 사용
pip install numpy pandas matplotlib seaborn scipy

# conda 사용
conda install numpy pandas matplotlib seaborn scipy

# Jupyter Notebook (권장)
pip install jupyter
jupyter notebook
```

### 버전 확인

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

### 권장 버전
- Python: 3.9+
- NumPy: 1.21+
- Pandas: 1.5+
- Matplotlib: 3.5+
- Seaborn: 0.12+

---

## 학습 순서 권장

### 1단계: NumPy 기초 (01-02)
- 배열 연산의 기초
- 벡터화 연산 이해

### 2단계: Pandas 기초 (03-05)
- 데이터 구조 이해
- 데이터 조작 능력

### 3단계: 데이터 전처리와 EDA (06-07)
- 실무에서 가장 많이 사용
- 데이터 품질 관리

### 4단계: 시각화 (08-09)
- 데이터 시각적 표현
- 인사이트 전달

### 5단계: 통계와 실전 (10-11)
- 통계적 분석
- 종합 프로젝트

---

## 참고 자료

### 공식 문서
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### 추천 데이터셋
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [공공데이터포털](https://www.data.go.kr/)

# 선형대수학 (Linear Algebra)

## 소개

선형대수학은 벡터 공간, 벡터 공간 사이의 선형 사상, 그리고 이러한 구조에서 발생하는 연립일차방정식을 다루는 수학의 한 분야입니다. 선형대수학은 연립방정식의 풀이와 기하학적 변환 분석부터 신경망 학습과 이미지 압축에 이르기까지 현대 과학 및 공학의 거의 모든 분야에 언어적, 계산적 기반을 제공합니다.

이 과정은 벡터와 행렬의 기초부터 시작하여 고유값 이론, 행렬 분해, 차원 축소까지 선형대수학을 체계적으로 구축합니다. 모든 개념에는 엄밀한 정의, 풀이 예제, 실용적인 Python/NumPy 구현이 함께 제공되어 배운 내용을 즉시 적용할 수 있습니다.

머신러닝, 컴퓨터 그래픽스, 신호 처리, 수치 시뮬레이션 등의 고급 과정을 준비하고 있다면, 선형대수학에 대한 탄탄한 이해는 필수입니다. 이 과정의 목표는 단순히 계산 방법을 가르치는 것이 아니라, 새로운 문제에서 선형대수학적 구조를 인식하고 효과적으로 활용할 수 있는 기하학적 직관과 대수적 유창성을 기르는 것입니다.

## 선행 지식

### 필수
- **Programming** -- 기본 Python 문법, 함수, 반복문, 자료구조
- **Python** -- NumPy 배열 생성, 인덱싱, 산술 연산에 대한 기본 지식

### 권장
- 고등학교 대수 (방정식, 부등식, 함수 표기법)
- 기본 좌표 기하학 (점 표시, 기울기, 직선)

## 파일 목록

| No. | 파일명 | 주제 | 주요 내용 |
|-----|--------|------|----------|
| 00 | 00_Overview.md | 개요 | 과정 소개 및 학습 안내 |
| 01 | 01_Vectors_and_Vector_Spaces.md | 벡터와 벡터 공간 | 벡터 연산, 선형 독립, 기저, 생성, 부분공간 |
| 02 | 02_Matrices_and_Operations.md | 행렬과 연산 | 행렬 곱셈, 전치, 역행렬, 행렬식, 대각합, 특수 행렬 |
| 03 | 03_Systems_of_Linear_Equations.md | 연립일차방정식 | 가우스 소거법, REF, RREF, LU 분해, 해의 존재성과 유일성 |
| 04 | 04_Vector_Norms_and_Inner_Products.md | 벡터 노름과 내적 | L1/L2/Lp/Frobenius 노름, 내적, Cauchy-Schwarz, 직교성 |
| 05 | 05_Linear_Transformations.md | 선형 변환 | 변환 행렬, 핵, 상, 차원 정리, 합성 |
| 06 | 06_Eigenvalues_and_Eigenvectors.md | 고유값과 고유벡터 | 특성 다항식, 대각화, 스펙트럼 정리, 거듭제곱법 |
| 07 | 07_Singular_Value_Decomposition.md | 특이값 분해 | SVD 유도, 기하학적 해석, 저랭크 근사, 이미지 압축 |
| 08 | 08_Principal_Component_Analysis.md | 주성분 분석 | SVD/고유 분해를 통한 PCA, 설명된 분산, 스크리 도표, 차원 축소 |
| 09 | 09_Orthogonality_and_Projections.md | 직교성과 투영 | QR 분해, Gram-Schmidt, 직교 투영, 최소자승법 |
| 10 | 10_Matrix_Decompositions.md | 행렬 분해 | Cholesky, LDL^T, Schur, 극분해, 분해법 비교 |
| 11 | 11_Quadratic_Forms_and_Definiteness.md | 이차 형식과 정부호성 | 양/음의 정부호 행렬, Sylvester 판정법, 최적화와의 연결 |
| 12 | 12_Vector_Spaces_Advanced.md | 고급 벡터 공간 | 쌍대 공간, 몫 공간, 직합, 함수 공간 |
| 13 | 13_Numerical_Linear_Algebra.md | 수치 선형대수학 | 부동소수점 문제, 조건수, 반복 해법, 희소 행렬 |
| 14 | 14_Tensors_and_Multilinear_Algebra.md | 텐서와 다중선형대수 | 텐서 곱, 아인슈타인 표기법, einsum, 브로드캐스팅 |
| 15 | 15_Linear_Algebra_in_Machine_Learning.md | ML에서의 선형대수 | 피처 행렬, 커널 방법, 워드 임베딩, 신경망 층 |
| 16 | 16_Linear_Algebra_in_Graphics.md | 그래픽스에서의 선형대수 | 동차 좌표, 모델-뷰-투영, 쿼터니언, 레이 트레이싱 |
| 17 | 17_Linear_Algebra_in_Signal_Processing.md | 신호 처리에서의 선형대수 | 행렬로서의 DFT, 합성곱, 필터링, 웨이블릿 |
| 18 | 18_Matrix_Functions_and_Exponentials.md | 행렬 함수와 행렬 지수 | 행렬 지수, 멱급수, Cayley-Hamilton, ODE 응용 |
| 19 | 19_Iterative_Methods.md | 반복법 | Jacobi, Gauss-Seidel, 켤레 기울기법, Krylov 부분공간 |
| 20 | 20_Advanced_Decompositions_and_Applications.md | 고급 분해와 응용 | Jordan 표준형, 일반화 고유벡터, 행렬 로그, Kronecker 곱 |

## 필수 라이브러리

```bash
pip install numpy scipy matplotlib
```

- **NumPy** -- 벡터 및 행렬 연산, 선형대수 루틴
- **SciPy** -- 고급 분해, 희소 행렬, 반복 해법
- **Matplotlib** -- 기하학적 개념 시각화

## 권장 학습 경로

### 1단계: 기초 (레슨 01-05) -- 2-3주
- 벡터, 행렬 및 연산
- 연립방정식 풀이
- 노름, 내적, 직교성
- 선형 변환과 그 성질

**목표**: 선형대수학의 핵심 대상과 연산에 대한 유창성을 기릅니다.

### 2단계: 스펙트럼 이론과 분해 (레슨 06-10) -- 2-3주
- 고유값과 고유벡터
- SVD와 PCA
- QR, Cholesky 등 행렬 분해

**목표**: 행렬이 어떻게 분해되는지, 분해가 왜 중요한지 이해합니다.

### 3단계: 고급 이론 (레슨 11-14) -- 2주
- 이차 형식과 정부호성
- 고급 벡터 공간 개념
- 수치적 안정성
- 텐서와 다중선형대수

**목표**: 이론적 이해를 심화하고 실용적인 계산 문제를 다룹니다.

### 4단계: 응용 (레슨 15-20) -- 2-3주
- 머신러닝, 컴퓨터 그래픽스, 신호 처리
- 행렬 지수와 반복법
- 고급 분해

**목표**: 선형대수학을 실제 분야에 적용합니다.

## 관련 주제

- [Math_for_AI](../Math_for_AI/00_Overview.md) -- AI/ML/DL을 위한 수학적 기초
- [Deep_Learning](../Deep_Learning/00_Overview.md) -- 신경망 아키텍처와 학습
- [Machine_Learning](../Machine_Learning/00_Overview.md) -- 고전 및 현대 ML 알고리즘
- [Computer_Graphics](../Computer_Graphics/00_Overview.md) -- 렌더링, 변환, 셰이딩
- [Signal_Processing](../Signal_Processing/00_Overview.md) -- 푸리에 분석, 필터링, 웨이블릿
- [Numerical_Simulation](../Numerical_Simulation/00_Overview.md) -- PDE 풀이 및 수치 방법

## 참고 자료

### 교재
1. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
2. **Axler, S.** (2015). *Linear Algebra Done Right* (3rd ed.). Springer.
3. **Boyd, S., & Vandenberghe, L.** (2018). *Introduction to Applied Linear Algebra*. Cambridge University Press.
4. **Horn, R. A., & Johnson, C. R.** (2012). *Matrix Analysis* (2nd ed.). Cambridge University Press.

### 온라인 자료
1. **3Blue1Brown -- Essence of Linear Algebra**: 핵심 개념에 대한 시각적 직관
2. **MIT 18.06 (Gilbert Strang)**: 고전적인 대학 강의 시리즈
3. **Khan Academy -- Linear Algebra**: 단계별 입문 과정

## 버전 정보

- **최초 작성일**: 2026-03-07
- **저자**: Claude (Anthropic)
- **Python 버전**: 3.8+
- **주요 라이브러리 버전**:
  - NumPy >= 1.20
  - SciPy >= 1.7
  - Matplotlib >= 3.4

## License

This material is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

---

**다음 단계**: [01. 벡터와 벡터 공간](01_Vectors_and_Vector_Spaces.md)에서 시작하세요.

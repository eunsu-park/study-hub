# 물리과학을 위한 수학적 방법론 (Mathematical Methods in the Physical Sciences) - Overview

## 소개

물리학과 공학의 핵심 문제들을 해결하기 위해서는 체계적인 수학적 도구가 필수적입니다. 이 과정은 Mary L. Boas의 *Mathematical Methods in the Physical Sciences*를 기반으로, 물리과학에서 가장 빈번하게 사용되는 수학적 방법론들을 체계적으로 다룹니다.

무한급수와 복소수로 시작하여, 선형대수, 편미분, 벡터 해석, 푸리에 해석, 미분방정식, 특수함수, 복소해석, 적분변환, 변분법, 텐서 해석까지 — 현대 물리학과 공학의 이론적 기반을 이루는 수학적 도구들을 빠짐없이 다룹니다.

각 레슨은 엄밀한 수학적 이론과 함께 Python(NumPy, SciPy, SymPy, Matplotlib) 구현을 제공하여, 추상적인 수식을 직접 계산하고 시각화할 수 있도록 구성되어 있습니다.

## 파일 목록

| 번호 | 파일명 | 주제 | 주요 내용 |
|------|--------|------|-----------|
| 00 | 00_Overview.md | 개요 | 과정 소개 및 학습 가이드 |
| 01 | 01_Infinite_Series.md | 무한급수와 수렴 | 수렴 판정법, 멱급수, 테일러 급수, 점근 급수 |
| 02 | 02_Complex_Numbers.md | 복소수 | 복소 대수, 극좌표/지수 표현, 드모아브르 정리, 오일러 공식 |
| 03 | 03_Linear_Algebra.md | 선형대수 | 행렬, 행렬식, 연립방정식, 고유값/고유벡터, 대각화, 이차형식 |
| 04 | 04_Partial_Differentiation.md | 편미분 | 편미분, 연쇄법칙, 라그랑주 승수법, 완전미분, 테일러 급수 |
| 05 | 05_Vector_Analysis.md | 벡터 해석 | 기울기·발산·회전, 선적분·면적분, 스토크스·가우스·그린 정리 |
| 06 | 06_Curvilinear_Coordinates.md | 곡선좌표계와 다중적분 | 원통·구면 좌표, 야코비안, 좌표 변환, 체적·면적 요소 |
| 07 | 07_Fourier_Series.md | 푸리에 급수 | 푸리에 계수, 수렴 조건, 깁스 현상, 파르세발 정리 |
| 08 | 08_Fourier_Transforms.md | 푸리에 변환 | 연속 푸리에 변환, DFT, FFT, 컨볼루션 정리, 응용 |
| 09 | 09_ODE_First_Second_Order.md | 상미분방정식 (1·2차) | 분리형·완전형·선형 ODE, 적분인자, 특성방정식 |
| 10 | 10_Higher_Order_ODE_Systems.md | 고차 ODE와 연립계 | 매개변수 변환법, 연립 ODE, 위상 평면, 안정성 |
| 11 | 11_Series_Solutions_Special_Functions.md | 급수해와 특수함수 | 프로베니우스 방법, 베셀·르장드르·에르미트·라게르, 구면조화함수 |
| 12 | 12_Sturm_Liouville_Theory.md | 스투름-리우빌 이론 | 고유값 문제, 직교함수, 완비성, 레일리 몫, 비교 정리 |
| 13 | 13_Partial_Differential_Equations.md | 편미분방정식 | PDE 분류, 변수분리법, 헬름홀츠 방정식, 유일성 정리 |
| 14 | 14_Complex_Analysis.md | 복소해석 | 해석함수, 유수 정리, 실수 적분 4유형, 해석적 연속 |
| 15 | 15_Laplace_Transform.md | 라플라스 변환 | 정의와 성질, 역변환, ODE/회로 문제 풀이, 전달함수 |
| 16 | 16_Greens_Functions.md | 그린 함수 | 델타 함수, 그린 함수 구성, 경계값 문제, 물리 응용 |
| 17 | 17_Calculus_of_Variations.md | 변분법 | 오일러-라그랑주 방정식, 구속 조건, 라그랑주 역학 |
| 18 | 18_Tensor_Analysis.md | 텐서 해석 | 인덱스 표기법, 계량 텐서, 공변 미분, 물리 응용 |

## 필수 라이브러리

```bash
pip install numpy scipy matplotlib sympy
```

- **NumPy**: 수치 계산, 배열 연산, 선형대수
- **SciPy**: 특수함수, 적분, ODE/PDE 솔버, FFT
- **Matplotlib**: 함수 그래프, 벡터장, 등고선 시각화
- **SymPy**: 심볼릭 미적분, 급수 전개, 라플라스 변환

## 권장 학습 순서

### Phase 1: 기초 도구 (01-06) — 3-4주

```
01 무한급수 ──→ 02 복소수 ──→ 03 선형대수 ──→ 04 편미분
                                                    │
                              05 벡터 해석 ──→ 06 곡선좌표계
```

- 급수의 수렴과 발산을 판별하는 방법
- 복소수의 대수적·기하학적 성질
- 행렬, 고유값, 이차형식 (이후 ODE/S-L/텐서의 기초)
- 편미분, 라그랑주 승수법, 열역학 관계식
- 벡터장의 미분과 적분 (grad, div, curl)
- 다양한 좌표계에서의 연산

**목표**: 이후 모든 주제의 기반이 되는 수학적 도구 확보

### Phase 2: 푸리에 해석 (07-08) — 1-2주

```
07 푸리에 급수 ──→ 08 푸리에 변환
```

- 주기 함수의 주파수 분해
- 연속/이산 푸리에 변환과 FFT
- 신호 처리 및 PDE 풀이의 핵심 도구

**목표**: 주파수 영역 분석 능력 확보

### Phase 3: 미분방정식 (09-13) — 3-4주

```
09 ODE (1·2차) ──→ 10 고차 ODE/연립계
                          │
11 급수해/특수함수 ──→ 12 S-L 이론 ──→ 13 PDE
```

- 상미분방정식의 해석적 해법
- 특수함수와 직교함수계 (베셀, 르장드르, 구면조화함수)
- 편미분방정식의 변수분리법, 헬름홀츠 방정식

**목표**: 물리학의 핵심 방정식들을 해석적으로 풀 수 있는 능력

### Phase 4: 고급 주제 (14-18) — 2-3주

```
14 복소해석 ──→ 15 라플라스 변환
                      │
16 그린 함수 ──→ 17 변분법 ──→ 18 텐서 해석
```

- 복소 적분과 유수 정리, 실수 적분의 4가지 유형
- 라플라스 변환을 이용한 초기값 문제
- 그린 함수와 경계값 문제
- 오일러-라그랑주 방정식과 라그랑주 역학
- 텐서와 일반 상대론 기초

**목표**: 고급 물리학과 공학 문제를 다루는 정교한 수학적 도구 습득

## 선수 지식

### 필수
- **미적분학**: 미분, 적분, 편미분, 연쇄 법칙
- **선형대수**: 벡터, 행렬, 고유값, 행렬식
- **Python 기본**: 함수, 루프, 리스트

### 권장
- **NumPy 기본**: 배열 생성과 연산
- **대학 물리**: 역학, 전자기학 기초 (응용 예제 이해에 도움)

### 연계 과정
- [Calculus_and_Differential_Equations](../Calculus_and_Differential_Equations/00_Overview.md): 선행 과목 — 미적분학 기초, ODE/PDE 입문
- [Math_for_AI](../Math_for_AI/00_Overview.md): ML/DL 관점의 수학 (보완적)
- [Numerical_Simulation](../Numerical_Simulation/00_Overview.md): 수치적 방법 (해석적 방법의 보완)
- [Data_Science](../Data_Science/00_Overview.md): 확률과 통계

## 학습 목표

이 과정을 완료하면 다음을 할 수 있습니다:

1. **급수 수렴 판별**: 다양한 판정법을 적용하여 급수의 수렴/발산 판별
2. **복소수 활용**: 복소 지수를 이용한 삼각함수 항등식 유도, 다항식의 근 탐색
3. **벡터장 분석**: 물리적 장(field)의 발산과 회전 계산, 적분 정리 적용
4. **좌표 변환**: 문제의 대칭성에 맞는 좌표계 선택과 변환
5. **푸리에 분석**: 신호의 주파수 성분 분석, 필터링, PDE 풀이
6. **ODE 해석해**: 다양한 유형의 상미분방정식의 일반해와 특수해 구하기
7. **특수함수 이해**: 베셀, 르장드르 등 특수함수의 성질과 물리적 응용
8. **PDE 풀이**: 변수분리법을 이용한 열방정식, 파동방정식, 라플라스 방정식 풀이
9. **복소 적분**: 유수 정리를 이용한 실수 적분 계산
10. **변분 문제**: 오일러-라그랑주 방정식을 이용한 최적화 문제 풀이
11. **텐서 연산**: 인덱스 표기법과 텐서 변환 규칙 적용
12. **물리 문제 해결**: 위 도구들을 종합하여 실제 물리/공학 문제를 수학적으로 정식화하고 풀기

## 기존 과정과의 관계

```
Mathematical_Methods          Math_for_AI              Numerical_Simulation
──────────────────────────────────────────────────────────────────────────────
해석적·일반적 수학 도구     ML/DL 특화 수학           수치적 계산 방법
물리/공학 응용 중심         최적화, 확률, 정보이론      ODE/PDE 수치 솔버
Boas 교재 기반             딥러닝 아키텍처 수학        시뮬레이션 응용
```

- **Mathematical_Methods**: *어떻게 풀어야 하는가* (해석적 방법)
- **Numerical_Simulation**: *어떻게 계산할 것인가* (수치적 방법)
- **Math_for_AI**: *AI에 어떻게 적용하는가* (ML/DL 관점)

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed. Wiley.
   - 본 과정의 주요 참고서
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed. Academic Press.
   - 대학원 수준 참고서
3. **Kreyszig, E.** (2011). *Advanced Engineering Mathematics*, 10th ed. Wiley.
   - 공학 수학 종합 참고서
4. **Riley, K. F., Hobson, M. P., & Bence, S. J.** (2006). *Mathematical Methods for Physics and Engineering*, 3rd ed. Cambridge University Press.
   - 물리/공학 수학의 또 다른 표준 교재

### 온라인 자료
1. **MIT OCW 18.04**: Complex Variables with Applications
2. **MIT OCW 18.03**: Differential Equations
3. **3Blue1Brown**: Fourier Transform 시각화
4. **Paul's Online Math Notes**: ODE/PDE 참고

### 도구
1. **Wolfram Alpha**: 수식 검증
2. **Desmos**: 함수 시각화
3. **SymPy Live**: 온라인 심볼릭 계산

## 버전 정보

- **최초 작성**: 2026-02-08
- **작성자**: Claude (Anthropic)
- **기반 교재**: Boas, *Mathematical Methods in the Physical Sciences*, 3rd ed.
- **Python 버전**: 3.8+
- **주요 라이브러리 버전**:
  - NumPy >= 1.20
  - SciPy >= 1.7
  - Matplotlib >= 3.4
  - SymPy >= 1.9

## 라이선스

이 자료는 교육 목적으로 자유롭게 사용할 수 있습니다. 상업적 사용 시 출처를 명시해주세요.

---

**다음 단계**: [01. 무한급수와 수렴](01_Infinite_Series.md)으로 시작하세요.

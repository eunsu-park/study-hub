# 미적분학과 미분방정식

## 소개

미적분학(Calculus)은 변화의 수학적 언어이다. 대수학(Algebra)이 정적인 관계를 설명하는 반면, 미적분학은 연속적으로 변하는 양을 분석하는 도구를 제공한다 -- 우주선의 궤도, 인구의 성장, 재료를 통한 열의 흐름, 또는 주가의 변동 등. 미분방정식(Differential Equations)은 함수와 그 변화율 사이의 관계를 표현함으로써 이 힘을 확장하며, 과학과 공학의 거의 모든 분야에서 수학적 모델링의 근간을 형성한다.

이 과정은 극한(limits)과 도함수(derivatives)의 기초 개념에서 시작하여 적분(integration) 기법을 거쳐 상미분방정식(ODE)과 편미분방정식(PDE)으로 나아간다. 물리학, 데이터 과학, 기계 학습(Machine Learning), 또는 공학을 공부하든, 이 개념들은 표면 아래의 수학적 기반으로서 반복적으로 등장할 것이다.

## 학습 목표

이 과정을 마치면 다음을 할 수 있게 된다:

1. 엡실론-델타(epsilon-delta) 체계를 사용하여 극한(limit)과 연속(continuity)의 형식적 정의를 **설명**할 수 있다
2. 미분 규칙을 사용하여 도함수(derivative)를 **계산**하고 최적화(optimization) 문제에 적용할 수 있다
3. 치환(substitution), 부분적분(by-parts), 부분분수(partial fractions)를 사용하여 정적분과 부정적분을 **구할** 수 있다
4. 적분을 **적용**하여 넓이, 부피, 호의 길이(arc length), 물리량을 계산할 수 있다
5. 표준 판정법을 사용하여 무한 수열(sequence)과 급수(series)의 수렴(convergence)을 **분석**할 수 있다
6. 일반적인 함수에 대한 테일러 급수(Taylor series)와 매클로린 급수(Maclaurin series)를 **유도**하고 근사 오차를 한정할 수 있다
7. 변수분리법(separation of variables), 적분인자(integrating factors), 완전방정식(exact equations)을 사용하여 1계 상미분방정식(ODE)을 **풀** 수 있다
8. 상수계수를 가진 2계 선형 상미분방정식(제차 및 비제차)을 **풀** 수 있다
9. 미분방정식을 사용하여 실세계 현상(인구 동태, 회로, 역학)을 **모델링**할 수 있다
10. 라플라스 변환(Laplace transform)을 **적용**하여 초기값 문제를 풀 수 있다
11. 기본적인 편미분방정식(열 방정식, 파동 방정식, 라플라스 방정식)을 **분류**하고 풀 수 있다
12. Python(NumPy, SciPy, SymPy)을 사용하여 적분과 ODE 풀이를 위한 수치적 방법을 **구현**할 수 있다

## 선수 과목

- **고등학교 대수학과 삼각함수**: 대수적 조작, 함수, 기본 삼각 항등식에 대한 숙련도
- **기본 Python 프로그래밍**: 변수, 반복문, 함수, 기본 그래프 그리기 (참고: [프로그래밍](../Programming/00_Overview.md) 토픽)
- **권장**: NumPy 배열에 대한 친숙함 (참고: [데이터 과학](../Data_Science/00_Overview.md) L01-L03)

## 필수 라이브러리

```bash
pip install numpy scipy sympy matplotlib
```

| 라이브러리 | 용도 |
|---------|---------|
| **NumPy** | 수치 배열, 기본 수학 연산 |
| **SciPy** | 수치 적분 (`scipy.integrate`), ODE 풀이기 |
| **SymPy** | 기호 미분, 적분, 방정식 풀이 |
| **Matplotlib** | 함수, 수렴, 위상 초상(phase portrait) 시각화 |

## 강의 개요

| # | 파일명 | 제목 | 설명 |
|---|----------|-------|-------------|
| 00 | `00_Overview.md` | 과정 개요 | 소개, 선수 과목, 학습 경로 |
| 01 | `01_Limits_and_Continuity.md` | 극한과 연속 | 엡실론-델타 정의, 극한 법칙, 연속, 중간값 정리 |
| 02 | `02_Derivatives_Fundamentals.md` | 도함수의 기초 | 차분 몫, 미분 규칙, 연쇄 법칙 |
| 03 | `03_Applications_of_Derivatives.md` | 도함수의 응용 | 최적화, 관련 변화율, 로피탈 법칙, 테일러 다항식 |
| 04 | `04_Integration_Fundamentals.md` | 적분의 기초 | 리만 합, 미적분학의 기본 정리, 역도함수 |
| 05 | `05_Integration_Techniques.md` | 적분 기법 | 치환, 부분적분, 부분분수, 이상 적분 |
| 06 | `06_Applications_of_Integration.md` | 적분의 응용 | 부피, 호의 길이, 겉넓이, 물리적 응용 |
| 07 | `07_Sequences_and_Series.md` | 수열과 급수 | 수렴 판정법, 멱급수, 테일러 급수 |
| 08 | `08_Parametric_and_Polar.md` | 매개변수 곡선과 극좌표 | 매개변수 방정식, 극곡선, 넓이와 호의 길이 |
| 09 | `09_Multivariable_Functions.md` | 다변수 함수 | 편도함수, 기울기, 방향도함수 |
| 10 | `10_Multiple_Integrals.md` | 중적분 | 이중 및 삼중 적분, 변수 변환 |
| 11 | `11_Vector_Calculus.md` | 벡터 미적분학 | 선적분, 면적분, 그린/스토크스/발산 정리 |
| 12 | `12_First_Order_ODE.md` | 1계 상미분방정식 | 분리형, 선형, 완전, 베르누이 방정식 |
| 13 | `13_Second_Order_ODE.md` | 2계 상미분방정식 | 제차, 비제차, 미정계수법, 매개변수 변환법 |
| 14 | `14_Systems_of_ODE.md` | 연립 상미분방정식 | 행렬 방법, 위상 초상, 안정성 분석 |
| 15 | `15_Laplace_Transform_for_ODE.md` | ODE를 위한 라플라스 변환 | 변환 쌍, 역변환, 초기값 문제 풀이 |
| 16 | `16_Power_Series_Solutions.md` | ODE의 멱급수 해법 | 멱급수 방법, 프로베니우스 방법, 베셀/르장드르 |
| 17 | `17_Introduction_to_PDE.md` | 편미분방정식 입문 | 분류, 열 방정식, 파동 방정식, 라플라스 방정식 |
| 18 | `18_Fourier_Series_and_PDE.md` | 푸리에 급수와 PDE | 푸리에 계수, 스튀름-리우빌, 변수분리법 |
| 19 | `19_Numerical_Methods_for_DE.md` | 미분방정식의 수치적 방법 | 오일러, 룽게-쿠타, 적응 단계, 강성 시스템 |

## 학습 경로

```
1단계: 미적분학의 기초 (레슨 01-04)
  극한 --> 도함수 --> 적분 기초
       |
2단계: 기법과 응용 (레슨 05-08)
  적분 기법 --> 응용 --> 급수
  --> 매개변수 & 극좌표
       |
3단계: 다변수 및 벡터 미적분학 (레슨 09-11)
  다변수 함수 --> 중적분 --> 벡터 미적분학
       |
4단계: 상미분방정식 (레슨 12-16)
  1계 --> 2계 --> 연립
  --> 라플라스 변환 --> 멱급수 해법
       |
5단계: PDE와 수치적 방법 (레슨 17-19)
  편미분방정식 입문 --> 푸리에 급수 --> 수치적 방법
```

**권장 학습 속도**: 주당 1-2개 레슨, 다음으로 넘어가기 전에 연습 문제를 완료할 것. 이후의 모든 내용이 도함수와 적분에 기반하므로 1단계를 완전히 숙달한 후에 진행해야 한다.

## 다른 토픽과의 연결

이 과정은 학습 자료의 여러 다른 토픽과 깊은 연관이 있다:

| 관련 토픽 | 연결 |
|---------------|------------|
| [물리수학](../Mathematical_Methods/00_Overview.md) | 푸리에 해석(L06), ODE/PDE(L07-L08), 특수 함수(L09), 복소해석(L11), 그린 함수(L13), 변분법(L14)으로 확장 |
| [수치 시뮬레이션](../Numerical_Simulation/00_Overview.md) | 이 과정의 수치 ODE/PDE 풀이기를 물리적 시뮬레이션에 적용 |
| [AI를 위한 수학](../Math_for_AI/00_Overview.md) | 도함수(역전파), 기울기(최적화), 행렬 미적분을 전반적으로 활용 |
| [딥러닝](../Deep_Learning/00_Overview.md) | 경사 기반 최적화, 손실 함수 지형, 자동 미분 |
| [신호 처리](../Signal_Processing/00_Overview.md) | 푸리에 급수/변환, 합성곱, LTI 시스템을 위한 미분방정식 모델 |
| [플라즈마 물리학](../Plasma_Physics/00_Overview.md) | MHD 방정식, 수송 이론, 파동 방정식은 모두 PDE |
| [제어 이론](../Control_Theory/00_Overview.md) | 전달 함수, 라플라스 변환, 상태공간 표현 |

## 참고 자료

### 교과서
- **Stewart, J.** *Calculus: Early Transcendentals*, 9th Ed. (Cengage, 2020) -- 포괄적인 학부 참고서
- **Zill, D.G.** *Advanced Engineering Mathematics*, 7th Ed. (Jones & Bartlett, 2022) -- 응용이 포함된 ODE/PDE
- **Strang, G.** *Calculus* (Wellesley-Cambridge Press) -- [ocw.mit.edu](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf)에서 무료 제공
- **Tenenbaum, M. & Pollard, H.** *Ordinary Differential Equations* (Dover) -- 고전적이고 저렴한 참고서

### 온라인 자료
- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) -- 뛰어난 시각적 직관
- [MIT OCW 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)
- [MIT OCW 18.03 Differential Equations](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/)
- [Paul's Online Math Notes](https://tutorial.math.lamar.edu/) -- 훌륭한 풀이 예제
- [Khan Academy Calculus](https://www.khanacademy.org/math/calculus-1)

---

[다음: 극한과 연속](./01_Limits_and_Continuity.md)

# 딥러닝을 위한 수학 (Mathematics for Deep Learning)

## 소개

딥러닝은 컴퓨터 비전과 자연어 처리에서 과학 컴퓨팅과 신약 개발에 이르기까지 인공지능의 거의 모든 영역을 혁신했습니다. 그러나 모든 신경망 뒤에는 놀랍도록 우아한 수학적 아이디어들이 존재합니다 -- 미분 가능한 계산 그래프 위의 경사 기반 최적화, 데이터 분포의 확률적 모델링, 그리고 대규모 학습을 안정적으로 만드는 수치적 기법들입니다.

이 과정은 딥러닝 실무자에게 필요한 핵심 수학을 집중적이고 자기 완결적인 패키지로 정리합니다. 최적화 이론, 측도론, 추상대수를 아우르는 보다 포괄적인 "AI를 위한 수학" 과정과 달리, **딥러닝을 위한 수학**은 딥러닝 연구와 엔지니어링에서 매일 등장하는 특정 수학 도구에 집중합니다: 역전파를 위한 행렬 미적분, 손실 함수를 위한 확률 분포, 모델 평가를 위한 정보 이론, 안정적인 학습을 위한 수치 안정성 기법입니다.

모든 개념은 먼저 구체적인 딥러닝 시나리오로 동기를 부여하고, 수학적 엄밀성을 갖추어 전개한 뒤, NumPy 코드로 검증합니다. 목표는 논문의 수학 섹션을 유창하게 읽고, 그래디언트 문제를 자신 있게 디버깅하며, 수학적 인식을 바탕으로 아키텍처를 설계할 수 있는 직관을 구축하는 것입니다.

## 선행 지식

### 필수
- **선형대수** -- 벡터, 행렬, 행렬 곱셈, 고유값, 기본 분해
- **미적분학과 미분방정식** -- 단일 변수 도함수, 적분, 테일러 급수, 기본 다변수 미적분

### 권장
- **Python 기초** -- NumPy 배열 연산, Matplotlib 기본 플로팅
- **확률과 통계** -- 확률 변수, 기댓값, 분산 (도움이 되지만 레슨 06에서 복습)

## 학습 목표

이 과정을 마치면 다음을 할 수 있습니다:

1. 행렬 미적분 수행 -- 야코비안, 헤시안, 벡터 대 벡터 도함수 계산
2. 계산 그래프 위의 연쇄 법칙으로 역전파 알고리즘 유도
3. 신경망의 최적화 지형 분석 (볼록성, 안장점, 수렴 조건)
4. 최대 우도를 통해 확률 분포와 표준 손실 함수 연결
5. 정보 이론 측도 (엔트로피, KL 발산, 교차 엔트로피)로 모델 평가
6. 딥러닝 맥락에서 행렬 분해 (고유 분해, SVD) 적용
7. 학습 파이프라인에서 수치 안정성 문제 진단 및 해결
8. 어텐션 메커니즘과 소프트맥스의 수학적 기초 이해

## 학습 로드맵

```
1단계: 미적분 엔진              2단계: 확률적 렌즈
┌─────────────────────┐           ┌─────────────────────┐
│ 01 DL을 위한         │           │ 06 확률 분포         │
│    벡터/행렬         │           │                     │
│         │           │           │         │            │
│         ▼           │           │         ▼            │
│ 02 편미분과          │           │ 07 최대 우도          │
│    그래디언트        │           │    추정              │
│         │           │           │         │            │
│         ▼           │           │         ▼            │
│ 03 연쇄 법칙과       │           │ 08 정보 이론          │
│    계산 그래프       │           │                     │
│         │           │           └─────────────────────┘
│         ▼           │
│ 04 야코비안과        │           3단계: 도구와 종합
│    헤시안           │           ┌─────────────────────┐
│         │           │           │ 09 행렬 분해          │
│         ▼           │           │         │            │
│ 05 최적화 이론       │           │         ▼            │
│                     │           │ 10 수치 안정성         │
└─────────────────────┘           │         │            │
                                  │         ▼            │
                                  │ 11 어텐션과           │
                                  │    소프트맥스 수학     │
                                  │         │            │
                                  │         ▼            │
                                  │ 12 종합 정리          │
                                  └─────────────────────┘
```

## 파일 목록

| No. | 파일명 | 주제 | 주요 내용 |
|-----|--------|------|----------|
| 00 | 00_Overview.md | 개요 | 과정 소개 및 학습 안내 |
| 01 | 01_Vectors_and_Matrices_for_DL.md | DL을 위한 벡터와 행렬 | 텐서 표기법, 배치 연산, 행렬 미분 규칙 |
| 02 | 02_Partial_Derivatives_and_Gradients.md | 편미분과 그래디언트 | 다변수 함수, 그래디언트 벡터, 방향 도함수 |
| 03 | 03_Chain_Rule_and_Computation_Graphs.md | 연쇄 법칙과 계산 그래프 | 다변수 연쇄 법칙, 순방향/역방향 자동 미분, 역전파 |
| 04 | 04_Jacobian_and_Hessian.md | 야코비안과 헤시안 | 벡터 함수 도함수, 2차 최적화, 피셔 정보 |
| 05 | 05_Optimization_Theory.md | 최적화 이론 | 볼록 최적화, 안장점, 수렴 조건, SGD 분석 |
| 06 | 06_Probability_Distributions_for_DL.md | DL을 위한 확률 분포 | 가우시안, 베르누이, 카테고리컬, 재매개변수화 기법 |
| 07 | 07_Maximum_Likelihood_Estimation.md | 최대 우도 추정 | MLE 유도, 로그 우도, 손실 함수와의 연결 |
| 08 | 08_Information_Theory.md | 정보 이론 | 엔트로피, 교차 엔트로피, KL 발산, 상호 정보량 |
| 09 | 09_Matrix_Decompositions.md | 행렬 분해 | 고유값 분해, SVD, DL에서의 활용 |
| 10 | 10_Numerical_Stability.md | 수치 안정성 | 오버플로우, 언더플로우, log-sum-exp, 부동소수점 |
| 11 | 11_Attention_and_Softmax_Math.md | 어텐션과 소프트맥스 수학 | 스케일링, 온도, 소프트맥스의 수학적 속성 |
| 12 | 12_Putting_It_All_Together.md | 종합 정리 | 수학이 DL에서 만나는 지점, 추가 학습 가이드 |

## 필수 라이브러리

```bash
pip install numpy matplotlib
```

- **NumPy** -- 행렬 연산, 선형대수, 수치 계산
- **Matplotlib** -- 수학 개념과 함수의 시각화

## 권장 학습 경로

### 1단계: 미적분 엔진 (레슨 01-05) -- 2-3주
- 텐서 표기법과 행렬 미적분 규칙
- 편미분, 그래디언트, 연쇄 법칙
- 야코비안, 헤시안, 최적화 이론

**목표**: 역전파와 경사 기반 최적화를 구동하는 미적분 기계를 마스터합니다.

### 2단계: 확률적 렌즈 (레슨 06-08) -- 1-2주
- DL에서 사용되는 확률 분포
- 최대 우도 추정과 손실 함수와의 연결
- 모델 평가를 위한 정보 이론

**목표**: 교차 엔트로피 손실, KL 발산을 사용하는 이유와 확률이 DL의 기반이 되는 방식을 이해합니다.

### 3단계: 도구와 종합 (레슨 09-12) -- 2주
- DL 맥락에서의 행렬 분해
- 수치 안정성과 부동소수점 함정
- 어텐션 메커니즘의 수학
- 모든 개념의 종합적 통합

**목표**: 실용적인 수학 도구를 습득하고 현대 아키텍처에서 모든 것이 어떻게 연결되는지 봅니다.

## 관련 토픽

| 토픽 | 관계 |
|------|------|
| [Linear_Algebra](../Linear_Algebra/00_Overview.md) | 선행 지식 -- 행렬/벡터 기초 제공 |
| [Calculus_and_Differential_Equations](../Calculus_and_Differential_Equations/00_Overview.md) | 선행 지식 -- 단일 변수 미적분 기초 제공 |
| [Math_for_AI](../Math_for_AI/00_Overview.md) | 더 광범위하고 고급 -- 측도론, 함수해석학 포함 |
| [Deep_Learning](../Deep_Learning/00_Overview.md) | 소비자 -- 여기서 개발한 모든 수학을 사용 |
| [Probability_and_Statistics](../Probability_and_Statistics/00_Overview.md) | 보완적 -- 더 깊은 확률론 |
| [Machine_Learning](../Machine_Learning/00_Overview.md) | 보완적 -- 유사한 수학적 기초를 사용하는 고전 ML |

## 참고 자료

### 교과서
1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*, Part I: Applied Math and ML Basics. MIT Press.
2. **Petersen, K. B., & Pedersen, M. S.** (2012). *The Matrix Cookbook*. Technical University of Denmark.
3. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
4. **Cover, T. M., & Thomas, J. A.** (2006). *Elements of Information Theory* (2nd ed.). Wiley.

### 온라인 자료
1. **3Blue1Brown -- Essence of Calculus**: 도함수와 적분에 대한 시각적 직관
2. **Terence Parr & Jeremy Howard -- The Matrix Calculus You Need For Deep Learning**: 실용적 행렬 미적분 가이드
3. **Distill.pub**: DL 수학 개념에 대한 인터랙티브 기사

## 버전 정보

- **최초 작성**: 2026-04-14
- **저자**: Claude (Anthropic)
- **Python 버전**: 3.8+
- **주요 라이브러리 버전**:
  - NumPy >= 1.20
  - Matplotlib >= 3.4

## 라이선스

이 자료는 **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International) 라이선스로 제공됩니다.

---

**다음 단계**: [01. DL을 위한 벡터와 행렬](01_Vectors_and_Matrices_for_DL.md)로 시작하세요.

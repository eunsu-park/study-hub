# 레슨 15: 양자 머신러닝

[← 이전: QAOA와 조합 최적화](14_QAOA.md) | [다음: 양자 컴퓨팅 현황과 미래 →](16_Landscape_and_Future.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 다양한 인코딩 전략을 사용하여 고전 데이터를 양자 상태로 인코딩하는 방법 설명
2. 양자 특성 맵(quantum feature map)과 고차원 특성 공간 생성에서의 역할 설명
3. 변분 양자 분류기(variational quantum classifier) 구성 및 훈련
4. 양자 커널 방법(quantum kernel method)과 고전 커널 SVM과의 연관성 설명
5. 불모 고원(barren plateau) 문제와 양자 ML 훈련 가능성에 미치는 영향 설명
6. 머신러닝에서의 양자 우위(quantum advantage) 주장 비판적 평가
7. Python으로 양자 특성 맵, 분류기, 커널 계산 구현

---

양자 머신러닝(Quantum Machine Learning, QML)은 우리 시대의 가장 혁신적인 두 기술인 양자 컴퓨팅과 머신러닝의 교차점에 위치합니다. 핵심 질문은 양자 컴퓨터가 고전 컴퓨터보다 더 빠르게 혹은 더 잘 데이터에서 패턴을 학습할 수 있는가입니다. 약속은 매혹적입니다 — 양자 컴퓨터의 지수적으로 큰 힐베르트 공간(Hilbert space)이 방대한 특성 공간으로 작용하여, 고전적으로는 시뮬레이션 불가능한 분류기를 만들 수 있을지도 모릅니다.

하지만 현실은 더 복잡합니다. 여러 이론적 결과가 특정하고 종종 인위적인 문제들에 대해 양자 우위를 보여주지만, 실세계 데이터셋에서 실용적 우위를 입증하는 것은 여전히 열린 과제입니다. 이 분야는 또한 불모 고원(barren plateau) 문제로 어려움을 겪고 있는데, 이는 무작위로 초기화된 양자 회로가 큐비트 수가 늘어남에 따라 훈련이 지수적으로 어려워짐을 시사합니다. 이 레슨은 진정한 통찰과 미해결 질문 모두를 강조하며 핵심 아이디어를 솔직하게 제시합니다.

> **비유:** 양자 ML은 힐베르트 공간을 지수적으로 큰 특성 공간으로 활용합니다 — 2D 지도를 초구(hypersphere) 표면에 투영하는 것과 같아서, 이전에 선형 분리 불가능했던 데이터 점들이 선형 분리 가능해질 수 있습니다. 고전 ML에서 커널 트릭이 좌표를 명시적으로 계산하지 않고 데이터를 고차원에 매핑하듯이, 양자 특성 맵은 양자 간섭(quantum interference)을 통해 지수적으로 큰 공간에 암묵적으로 접근합니다.

## 목차

1. [고전 ML 복습](#1-고전-ml-복습)
2. [데이터 인코딩 전략](#2-데이터-인코딩-전략)
3. [양자 특성 맵](#3-양자-특성-맵)
4. [변분 양자 분류기](#4-변분-양자-분류기)
5. [양자 커널 방법](#5-양자-커널-방법)
6. [불모 고원 문제](#6-불모-고원-문제)
7. [양자 우위: 주장과 현실](#7-양자-우위-주장과-현실)
8. [Python 구현](#8-python-구현)
9. [연습 문제](#9-연습-문제)

---

## 1. 고전 ML 복습

### 1.1 지도 학습 프레임워크

지도 학습(supervised learning)(자세한 내용은 Machine_Learning L01-L05 참조)에서는 다음을 다룹니다:

- **훈련 데이터**: $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ 여기서 $\mathbf{x}_i \in \mathbb{R}^d$는 특성(feature)이고 $y_i \in \{0, 1\}$은 레이블
- **목표**: 미지의 데이터에 일반화되는 함수 $f: \mathbb{R}^d \to \{0, 1\}$ 학습
- **모델**: $f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \phi(\mathbf{x}) + b)$ 여기서 $\phi$는 특성 맵

### 1.2 특성 맵과 커널

**특성 맵(feature map)** $\phi: \mathbb{R}^d \to \mathcal{H}$은 데이터를 선형 분류기로 충분한 (가능하면 고차원) 특성 공간으로 매핑합니다.

**커널 트릭(kernel trick)**은 다음을 정의함으로써 $\phi(\mathbf{x})$를 명시적으로 계산하는 것을 회피합니다:

$$K(\mathbf{x}, \mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$$

커널 함수 $K$는 특성 공간에서 데이터 점 사이의 유사도를 포착합니다. 일반적인 커널:

| 커널 | 수식 | 특성 공간 차원 |
|------|------|--------------|
| 선형(Linear) | $\mathbf{x} \cdot \mathbf{x}'$ | $d$ |
| 다항식(Polynomial) | $(\mathbf{x} \cdot \mathbf{x}' + c)^p$ | $\binom{d+p}{p}$ |
| RBF (가우시안) | $e^{-\gamma\|\mathbf{x} - \mathbf{x}'\|^2}$ | $\infty$ |
| **양자(Quantum)** | $|\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$ | $2^n$ |

### 1.3 왜 양자인가?

양자 특성 맵은 $n$개의 큐비트만으로 $2^n$ 차원의 특성 공간에 접근할 수 있습니다. 이 특성 공간이 학습 과제에 "유용"하다면 — 즉 데이터를 선형 분리 가능하게 만든다면 — 양자 분류기가 우위를 가질 수 있습니다.

핵심 질문들은:
1. 양자 특성 맵이 고전 방법으로는 효율적으로 시뮬레이션할 수 없는 유용한 특성 공간을 만들 수 있는가?
2. 양자 모델을 효율적으로 훈련할 수 있는가 (불모 고원 회피)?
3. 이러한 우위가 실용적으로 관련된 데이터셋에서도 지속되는가?

---

## 2. 데이터 인코딩 전략

### 2.1 인코딩 문제

고전 데이터 $\mathbf{x} \in \mathbb{R}^d$는 양자 처리 전에 양자 상태로 인코딩되어야 합니다. 인코딩 전략은 양자 모델의 표현력(expressiveness)과 훈련 가능성(trainability)을 근본적으로 결정합니다.

### 2.2 기저 인코딩(Basis Encoding)

$d$-비트 이진 문자열을 계산 기저 상태로 매핑:

$$\mathbf{x} = (x_1, x_2, \ldots, x_d) \to |x_1 x_2 \cdots x_d\rangle$$

**예시**: $\mathbf{x} = (1, 0, 1) \to |101\rangle$

**장점**: 단순하고 일대일 매핑
**단점**: 이진 데이터에만 작동, $d$개의 큐비트 필요 (압축 없음), 간섭 효과 없음

### 2.3 진폭 인코딩(Amplitude Encoding)

$d$-차원 벡터를 $\lceil\log_2 d\rceil$개 큐비트의 진폭으로 인코딩:

$$\mathbf{x} = (x_1, \ldots, x_d) \to |\psi_{\mathbf{x}}\rangle = \frac{1}{\|\mathbf{x}\|} \sum_{i=1}^{d} x_i |i\rangle$$

**예시**: $\mathbf{x} = (1, 2, 3, 4)/\sqrt{30} \to \frac{1}{\sqrt{30}}(|00\rangle + 2|01\rangle + 3|10\rangle + 4|11\rangle)$

**장점**: 지수적 압축 ($d$개 특성에 $\log_2 d$개 큐비트), 양자 선형 대수 활성화
**단점**: 상태 준비에 $O(d)$ 게이트가 필요하여 압축 이점이 상쇄될 수 있음

### 2.4 각도 인코딩(Angle Encoding)

각 특성을 별도의 큐비트에 회전 각도로 인코딩:

$$\mathbf{x} = (x_1, \ldots, x_d) \to \bigotimes_{i=1}^{d} R_y(x_i)|0\rangle = \bigotimes_{i=1}^{d} [\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle]$$

**예시**: $\mathbf{x} = (\pi/4, \pi/2) \to [\cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle] \otimes [|0\rangle + |1\rangle]/\sqrt{2}$

**장점**: 단순한 회로 ($d$개의 단일 큐비트 게이트), 변분 방법에 자연스러움
**단점**: $d$개의 큐비트 필요 (압축 없음), 곱 상태 (얽힘 없음)

### 2.5 비교

| 인코딩 | 큐비트 수 | 회로 깊이 | 데이터 유형 | 얽힘 |
|--------|----------|----------|------------|------|
| 기저 | $d$ | $O(d)$ | 이진 | 없음 |
| 진폭 | $\lceil\log_2 d\rceil$ | $O(d)$ | 연속 | 있음 |
| 각도 | $d$ | $O(1)$ | 연속 | 없음 (추가하지 않으면) |
| **특성 맵** | $n$ | $O(n \cdot L)$ | 연속 | 있음 (설계에 의해) |

특성 맵 인코딩(3절)은 현대 QML에서 사용되는 가장 표현력이 풍부한 방법입니다.

---

## 3. 양자 특성 맵

### 3.1 정의

**양자 특성 맵(quantum feature map)**은 고전 데이터 $\mathbf{x}$를 양자 상태로 매핑하는 매개변수화된 양자 회로 $S(\mathbf{x})$입니다:

$$|\phi(\mathbf{x})\rangle = S(\mathbf{x})|0\rangle^{\otimes n}$$

회로 $S(\mathbf{x})$는 일반적으로 데이터 인코딩 게이트와 얽힘 게이트를 교대로 배치합니다:

$$S(\mathbf{x}) = \prod_{l=1}^{L} U_{\text{ent}} \cdot U_{\text{enc}}(\mathbf{x})$$

### 3.2 ZZ 특성 맵

2D 데이터 $\mathbf{x} = (x_1, x_2)$에 대한 일반적인 특성 맵:

**레이어 1 (인코딩)**:
$$U_{\text{enc}}(\mathbf{x}) = \bigotimes_{i=1}^{n} R_z(x_i) H$$

**레이어 2 (얽힘)**:
$$U_{\text{ent}}(\mathbf{x}) = \prod_{(i,j)} e^{-i x_i x_j Z_i Z_j / 2}$$

얽힘 레이어는 특성의 *곱* $x_i x_j$를 상호작용 강도로 사용하여 데이터에 의존하는 큐비트 간 상관관계를 만듭니다.

### 3.3 특성 맵이 고차원 공간을 만드는 이유

상태 $|\phi(\mathbf{x})\rangle$은 $2^n$-차원 힐베르트 공간에 존재합니다. 밀도 행렬(density matrix):

$$\rho(\mathbf{x}) = |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})|$$

은 $2^n \times 2^n$ 행렬 — 효과적으로 데이터를 $2^{2n}$-차원 특성 공간(밀도 행렬 공간)에 임베딩합니다.

양자 커널은:

$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = \text{Tr}[\rho(\mathbf{x})\rho(\mathbf{x}')]$$

이 커널은 특성 벡터를 명시적으로 구성하지 않고 지수적으로 큰 특성 공간에서 암묵적으로 작동합니다.

### 3.4 특성 맵 설계 원칙

1. **표현력(Expressibility)**: 특성 맵이 다양한 상태를 만들 수 있어야 함 (힐베르트 공간의 높은 커버리지)
2. **데이터 의존적 얽힘**: 얽힘 게이트가 고정된 얽힘만이 아닌 데이터를 포함해야 함
3. **고전적으로 시뮬레이션하기 어려워야 함**: 특성 맵을 고전적으로 효율적으로 시뮬레이션할 수 있다면 양자 우위가 없음
4. **훈련 가능성**: 특성 맵이 불모 고원을 생성하지 않아야 함

---

## 4. 변분 양자 분류기

### 4.1 아키텍처

변분 양자 분류기(Variational Quantum Classifier, VQC)는 데이터 인코딩 회로와 훈련 가능한 회로를 결합합니다:

```
|0⟩⊗n → S(x) → W(θ) → Measure → Classical postprocessing → label
         ↑         ↑
    data encoding  trainable parameters
```

**구성 요소**:
1. **인코딩 회로** $S(\mathbf{x})$: 데이터를 양자 상태로 매핑
2. **변분 회로** $W(\boldsymbol{\theta})$: 훈련 가능한 매개변수화된 유니터리(unitary)
3. **측정**: 하나 이상의 큐비트 측정
4. **고전 처리**: 측정 결과를 클래스 예측으로 변환

### 4.2 훈련

VQC는 손실 함수를 최소화하는 방식으로 훈련됩니다:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \ell\left(y_i, \hat{y}(\mathbf{x}_i; \boldsymbol{\theta})\right)$$

여기서 $\hat{y}(\mathbf{x}_i; \boldsymbol{\theta}) = \langle\phi(\mathbf{x}_i)|W^\dagger(\boldsymbol{\theta}) M W(\boldsymbol{\theta})|\phi(\mathbf{x}_i)\rangle$는 모델 예측값입니다.

**훈련 루프**:
1. 데이터 포인트 배치 인코딩
2. 변분 회로 적용
3. 기댓값(expectation value) 측정
4. 손실 및 기울기 계산 (파라미터 이동 규칙(parameter shift rule))
5. 파라미터 업데이트

### 4.3 분류 규칙

이진 분류를 위해 Z 기저에서 큐비트 0을 측정:

$$P(\text{class 0}) = \langle Z_0 \rangle = \text{Tr}[Z_0 \rho_{\text{out}}]$$

$\langle Z_0 \rangle > 0$이면 클래스 0, 그렇지 않으면 클래스 1로 예측.

### 4.4 표현력과 일반화

$L$개 레이어의 $n$-큐비트 게이트를 가진 VQC는 $O(nL)$개의 매개변수를 가집니다. **유효 차원(effective dimension)**(모델 복잡도 척도)이 일반화 능력을 결정합니다:

- 매개변수가 너무 적음: 과소적합(underfitting) (경계를 학습할 수 없음)
- 매개변수가 너무 많음: 과적합(overfitting) (훈련 데이터 암기) 또는 불모 고원
- 최적 지점: 좋은 일반화와 충분한 표현력

---

## 5. 양자 커널 방법

### 5.1 양자 커널 정의

두 데이터 포인트 사이의 **양자 커널(quantum kernel)**(또는 충실도 커널(fidelity kernel))은:

$$K(\mathbf{x}, \mathbf{x}') = |\langle 0^n | S^\dagger(\mathbf{x}') S(\mathbf{x}) | 0^n \rangle|^2$$

이는 $\mathbf{x}$와 $\mathbf{x}'$을 인코딩하는 양자 상태 사이의 겹침(overlap)을 측정합니다.

### 5.2 양자 하드웨어에서의 커널 추정

$K(\mathbf{x}, \mathbf{x}')$를 추정하려면:

1. $|0\rangle^{\otimes n}$ 준비
2. $S(\mathbf{x})$ 적용 (첫 번째 데이터 포인트 인코딩)
3. $S^\dagger(\mathbf{x}')$ 적용 (두 번째 데이터 포인트 역방향 인코딩)
4. 모든 큐비트 측정
5. $K(\mathbf{x}, \mathbf{x}') = P(\text{모두 0})$

확률을 추정하기 위해 여러 번 반복합니다.

### 5.3 양자 커널 SVM

모든 훈련 쌍에 대한 커널 행렬 $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$를 구하면 고전 SVM 솔버를 사용할 수 있습니다:

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{ij} \alpha_i \alpha_j y_i y_j K_{ij}$$

제약 조건: $0 \leq \alpha_i \leq C$이고 $\sum_i \alpha_i y_i = 0$.

**VQC 대비 장점**:
- 불모 고원 문제 없음 (훈련할 매개변수화된 회로 없음)
- 볼록 최적화 (전역 최적해 보장)
- SVM 이론의 이론적 보장

**단점**:
- $O(N^2)$회의 커널 평가 필요 ($K$의 모든 쌍 계산)
- 각 평가에 많은 회로 반복 필요
- 대부분의 실용적 커널에서 명확한 양자 우위 없음

### 5.4 양자 커널이 도움이 되는 경우

양자 커널이 우위를 제공하는 경우:

1. 커널이 **고전적으로 계산하기 어려울** 때: 특성 맵이 효율적으로 시뮬레이션할 수 없는 높은 얽힘 회로를 포함
2. 커널이 **유용할** 때: 데이터가 양자 특성 공간에서 분리 가능
3. 데이터 분포가 양자 특성 공간 구조와 **일치할** 때

이론적 결과는 특정 구조화된 문제 (예: 이산 로그 기반 데이터)에 대한 양자 우위를 보여주지만, 일반적인 실세계 데이터셋에서의 우위는 불분명합니다.

---

## 6. 불모 고원 문제

### 6.1 정의

**불모 고원(barren plateau)**은 비용 함수의 기울기가 큐비트 수에 대해 지수적으로 작은 파라미터 공간의 영역입니다:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_i}\right] \leq O\left(\frac{1}{2^n}\right)$$

이는 큰 $n$에 대해 기울기가 모든 곳에서 본질적으로 0임을 의미하며, 기울기 기반 최적화를 불가능하게 만듭니다.

### 6.2 원인

불모 고원은 여러 원인에서 발생합니다:

**무작위 회로**: 매개변수화된 회로가 근사 2-설계(2-design)(하르 랜덤 유니터리(Haar-random unitary)에 접근)를 형성하면 기울기가 지수적으로 소멸합니다. 이는 깊은 하드웨어 효율적 앤잗제(ansatz)에서 발생합니다.

**전역 비용 함수**: 많은 큐비트를 측정하는 비용 함수 (예: 목표 상태와의 충실도)는 지역 비용 함수보다 불모 고원을 더 쉽게 생성합니다.

**얽힘**: 높은 얽힘 회로는 정보를 모든 큐비트에 분산시켜 지역 측정을 무의미하게 만듭니다.

**잡음**: NISQ 장치의 물리적 잡음도 비용 경관을 평탄하게 하여 효과적인 불모 고원을 생성할 수 있습니다.

### 6.3 수학적 분석

$n$개 큐비트를 가진 무작위 매개변수화된 양자 회로에 대해, 임의의 파라미터 $\theta_k$에 대한 편미분의 분산은:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_k}\right] \leq \frac{c}{2^n}$$

여기서 $c$는 비용 함수와 회로 아키텍처에 의존하는 상수입니다.

이는 일정한 확률로 0이 아닌 기울기를 감지하려면 $O(2^n)$번의 측정 샷이 필요함을 의미합니다 — 큐비트 수에 지수적입니다. 이는 잠재적인 양자 우위를 완전히 무력화합니다.

### 6.4 완화 전략

| 전략 | 아이디어 | 한계 |
|------|---------|------|
| 얕은 회로 | 깊이를 줄여 불모 고원 방지 | 표현력 감소 가능성 |
| 지역 비용 함수 | 일부 큐비트만 측정 | 전역 속성 포착 못할 수 있음 |
| 파라미터 초기화 | 알려진 좋은 해 근처에서 시작 (예: 항등원) | 문제 특정 |
| 레이어별 훈련 | 한 번에 한 레이어씩 훈련 | 휴리스틱, 보장 없음 |
| 대칭 보존 앤잗제 | 관련 대칭 섹터로 제한 | 문제 특정 |
| 고전 사전 훈련 | 초기 파라미터에 고전 방법 사용 | 양자 우위 감소 |

### 6.5 함의

불모 고원 문제는 실용적 양자 ML에서 가장 큰 장애물로 볼 수 있습니다. 이는 다음을 시사합니다:

- 무작위 양자 회로는 대규모에서 훈련 불가능
- 지수적 스케일링을 피하기 위해 문제 구조를 활용해야 함
- 양자 ML은 고전 딥러닝과 근본적으로 다른 훈련 전략이 필요할 수 있음

---

## 7. 양자 우위: 주장과 현실

### 7.1 이론적 우위

여러 결과가 특정 학습 과제에 대해 증명 가능한 양자 우위를 보여줍니다:

- **이산 로그 문제**: 양자 커널이 모든 고전 커널에 어려운 분류 문제를 풀 수 있음 (Liu et al., 2021)
- **양자 데이터**: 양자 시스템의 속성 학습은 고전 모델보다 양자 모델로 지수적으로 빠름
- **특정 분포**: 양자 모델이 특정 분포를 지수적으로 빠르게 학습 가능

### 7.2 실용적 한계

- **데이터 로딩 병목**: $N$개의 고전 데이터 포인트를 양자 상태로 인코딩하려면 처리의 양자 속도 향상과 무관하게 $O(N)$번의 회로 준비 필요
- **불모 고원**: 대규모 시스템에서 훈련 가능성 제한
- **샷 잡음**: 각 기댓값에 많은 측정 필요, 곱셈적 오버헤드 추가
- **제한된 큐비트**: 현재 장치는 $\leq 1000$개의 잡음 있는 큐비트로, 의미 있는 우위에 불충분
- **고전 기준선**: 고전 신경망과 커널 방법이 매우 강력하고 잘 최적화되어 있음

### 7.3 현재 평가 (2025년 기준)

| 주장 | 상태 |
|------|------|
| 양자 데이터에 대한 지수적 속도 향상 | **증명됨** |
| 고전 데이터에 대한 지수적 속도 향상 | **미해결/일반 데이터에는 불가능해 보임** |
| 실제 데이터셋에서의 실용적 우위 | **미입증** |
| 특정 구조화된 문제에서의 우위 | **이론적으로 가능** |
| ML을 위한 NISQ 우위 | **오류 수정 없이는 불가능해 보임** |

### 7.4 양자 ML이 도움이 될 수 있는 곳

가장 유망한 방향:

1. **양자 시스템 학습**: 양자 물질, 분자, 또는 다체 시스템의 속성 예측
2. **양자 시뮬레이션 + ML**: 양자 회로가 고전 ML을 위한 훈련 데이터를 생성하는 하이브리드 접근법
3. **생성 모델링**: 구조화된 분포를 위한 표현력 있는 생성 모델로서의 양자 회로
4. **조합 최적화**: 워밍 스타트(warm-starting)를 위한 ML과 결합된 QAOA 같은 접근법

---

## 8. Python 구현

### 8.1 양자 특성 맵

```python
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor_product(ops):
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def rz_gate(angle):
    """Single-qubit Z rotation gate."""
    return np.array([[np.exp(-1j*angle/2), 0],
                     [0, np.exp(1j*angle/2)]], dtype=complex)

def rx_gate(angle):
    """Single-qubit X rotation gate."""
    return np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                     [-1j*np.sin(angle/2), np.cos(angle/2)]], dtype=complex)

def ry_gate(angle):
    """Single-qubit Y rotation gate."""
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                     [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)

def zz_interaction(n_qubits, q1, q2, angle):
    """Implement e^{-i*angle*Z_q1*Z_q2/2} interaction gate.

    Why ZZ interaction? This creates data-dependent entanglement between
    qubits, which is essential for making the feature map expressive.
    The interaction strength depends on the product of features, creating
    nonlinear feature combinations in the quantum state.
    """
    N = 2**n_qubits
    gate = np.eye(N, dtype=complex)
    for state in range(N):
        z1 = 1 - 2 * ((state >> (n_qubits - 1 - q1)) & 1)
        z2 = 1 - 2 * ((state >> (n_qubits - 1 - q2)) & 1)
        gate[state, state] = np.exp(-1j * angle * z1 * z2 / 2)
    return gate

def quantum_feature_map(x, n_qubits, n_layers=2):
    """Apply the ZZ feature map to encode classical data x into a quantum state.

    Why the ZZ feature map? It creates entangled states where the entanglement
    structure depends on the input data. This means different data points
    produce states in different parts of Hilbert space, enabling quantum
    kernel methods to exploit the exponential dimensionality.

    Args:
        x: data point, array of length n_qubits (features padded/truncated as needed)
        n_qubits: number of qubits
        n_layers: number of repetitions of the encoding layer

    Returns:
        Quantum state vector (2^n_qubits dimensional)
    """
    N = 2**n_qubits
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0  # |00...0⟩

    # Pad or truncate x to match n_qubits
    x_padded = np.zeros(n_qubits)
    x_padded[:min(len(x), n_qubits)] = x[:n_qubits]

    for layer in range(n_layers):
        # Hadamard on all qubits
        H_all = tensor_product([H_gate] * n_qubits)
        state = H_all @ state

        # Rz encoding: Rz(x_i) on each qubit
        for i in range(n_qubits):
            ops = [I] * n_qubits
            ops[i] = rz_gate(x_padded[i])
            state = tensor_product(ops) @ state

        # ZZ entangling: e^{-i x_i x_j Z_i Z_j / 2} for all pairs
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                angle = x_padded[i] * x_padded[j]
                state = zz_interaction(n_qubits, i, j, angle) @ state

    return state

# Demonstrate feature map
print("=" * 55)
print("Quantum Feature Map Demonstration")
print("=" * 55)

n_qubits = 2
x1 = np.array([0.5, 1.0])
x2 = np.array([0.5, 1.1])  # Slightly different
x3 = np.array([2.0, -1.0])  # Very different

state1 = quantum_feature_map(x1, n_qubits)
state2 = quantum_feature_map(x2, n_qubits)
state3 = quantum_feature_map(x3, n_qubits)

# Compute quantum kernels (overlaps)
k12 = abs(np.dot(state1.conj(), state2))**2
k13 = abs(np.dot(state1.conj(), state3))**2
k23 = abs(np.dot(state2.conj(), state3))**2

print(f"\nData points: x1={x1}, x2={x2}, x3={x3}")
print(f"Kernel(x1, x2) = {k12:.4f}  (similar → high overlap)")
print(f"Kernel(x1, x3) = {k13:.4f}  (different → low overlap)")
print(f"Kernel(x2, x3) = {k23:.4f}")
```

### 8.2 변분 양자 분류기

```python
import numpy as np
from scipy.optimize import minimize

def variational_layer(state, params, n_qubits):
    """Apply one layer of the variational circuit.

    Each layer consists of:
    1. Ry rotations on each qubit (parameterized)
    2. CNOT entangling gates (ring topology)
    This creates a trainable transformation that can learn decision boundaries.
    """
    N = 2**n_qubits

    # Ry rotations
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = ry_gate(params[i])
        state = tensor_product(ops) @ state

    # CNOT ring: (0,1), (1,2), ..., (n-2,n-1)
    for i in range(n_qubits - 1):
        # CNOT: control=i, target=i+1
        cnot = np.eye(N, dtype=complex)
        for s in range(N):
            ctrl = (s >> (n_qubits - 1 - i)) & 1
            if ctrl == 1:
                tgt = (s >> (n_qubits - 1 - (i+1))) & 1
                new_s = s ^ (1 << (n_qubits - 1 - (i+1)))
                cnot[s, :] = 0
                cnot[new_s, :] = 0
                cnot[new_s, s] = 1
                # Also set the original state back if it was swapped
        # Simpler approach: build CNOT properly
        cnot = np.eye(N, dtype=complex)
        for s in range(N):
            ctrl_bit = (s >> (n_qubits - 1 - i)) & 1
            if ctrl_bit == 1:
                new_s = s ^ (1 << (n_qubits - 1 - (i + 1)))
                cnot[s, s] = 0
                cnot[new_s, s] = 1
        state = cnot @ state

    return state

def quantum_classifier(x, params, n_qubits, n_layers):
    """Apply the full quantum classifier circuit.

    Pipeline: |0⟩ → Feature Map S(x) → Variational W(θ) → Measure Z_0

    The feature map encodes the data, and the variational circuit learns
    the decision boundary. The measurement outcome gives the classification.
    """
    # Encode data using feature map
    state = quantum_feature_map(x, n_qubits, n_layers=1)

    # Apply variational layers
    params_per_layer = n_qubits
    for l in range(n_layers):
        layer_params = params[l*params_per_layer:(l+1)*params_per_layer]
        state = variational_layer(state, layer_params, n_qubits)

    # Measure Z on qubit 0: expectation value
    N = 2**n_qubits
    Z0 = np.zeros((N, N), dtype=complex)
    for s in range(N):
        bit0 = (s >> (n_qubits - 1)) & 1
        Z0[s, s] = 1 - 2 * bit0  # +1 for |0⟩, -1 for |1⟩

    expectation = np.real(state.conj() @ Z0 @ state)
    return expectation

def train_quantum_classifier(X_train, y_train, n_qubits=2, n_layers=2,
                              n_epochs=100, lr=0.1):
    """Train a variational quantum classifier using gradient descent.

    Why use a simple training loop? On real quantum hardware, we would
    estimate gradients using the parameter shift rule (2 circuit evaluations
    per parameter). Here we use finite differences for simplicity.
    """
    n_params = n_qubits * n_layers
    params = np.random.uniform(-np.pi, np.pi, n_params)

    # Convert labels from {0,1} to {+1,-1}
    y_signed = 2 * y_train - 1

    losses = []

    for epoch in range(n_epochs):
        # Compute predictions and loss
        predictions = np.array([quantum_classifier(x, params, n_qubits, n_layers)
                               for x in X_train])

        # Hinge loss: max(0, 1 - y*f(x))
        loss = np.mean(np.maximum(0, 1 - y_signed * predictions))
        losses.append(loss)

        # Compute gradients via finite differences
        grad = np.zeros_like(params)
        eps = 0.01
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            loss_plus = np.mean(np.maximum(0, 1 - y_signed *
                np.array([quantum_classifier(x, params_plus, n_qubits, n_layers)
                          for x in X_train])))
            loss_minus = np.mean(np.maximum(0, 1 - y_signed *
                np.array([quantum_classifier(x, params_minus, n_qubits, n_layers)
                          for x in X_train])))

            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Update parameters
        params -= lr * grad

        if epoch % 20 == 0:
            accuracy = np.mean((predictions > 0) == (y_signed > 0))
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, accuracy={accuracy:.2%}")

    return params, losses

# Generate a simple dataset (XOR-like, 2D)
np.random.seed(42)
n_samples = 40
X_data = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
# Labels: 1 if x1*x2 > 0, else 0 (XOR-like pattern)
y_data = (X_data[:, 0] * X_data[:, 1] > 0).astype(int)

print("=" * 55)
print("Variational Quantum Classifier Training")
print("=" * 55)
print(f"Dataset: {n_samples} points, 2 features, 2 classes")
print(f"Class balance: {np.mean(y_data):.2%} class 1\n")

params, losses = train_quantum_classifier(X_data, y_data, n_qubits=2,
                                           n_layers=2, n_epochs=80, lr=0.3)

# Final evaluation
predictions = np.array([quantum_classifier(x, params, 2, 2) for x in X_data])
final_acc = np.mean((predictions > 0) == (2*y_data - 1 > 0))
print(f"\nFinal accuracy: {final_acc:.2%}")
```

### 8.3 양자 커널 계산

```python
import numpy as np

def quantum_kernel_matrix(X, n_qubits, n_layers=2):
    """Compute the quantum kernel matrix for a dataset.

    Why use a kernel matrix? The kernel matrix K_{ij} = |⟨φ(x_i)|φ(x_j)⟩|²
    captures pairwise similarities in the quantum feature space. This can
    be used with classical SVM for classification, avoiding the barren
    plateau problem entirely.
    """
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))

    # Precompute all quantum states
    states = [quantum_feature_map(x, n_qubits, n_layers) for x in X]

    for i in range(n_samples):
        for j in range(i, n_samples):
            overlap = abs(np.dot(states[i].conj(), states[j]))**2
            K[i, j] = overlap
            K[j, i] = overlap  # Kernel matrix is symmetric

    return K

def quantum_kernel_svm(X_train, y_train, X_test, y_test, n_qubits=2):
    """Implement a quantum kernel SVM classifier.

    This uses the quantum kernel for similarity computation but a
    classical SVM for the actual optimization — avoiding barren plateaus
    while still leveraging the quantum feature space.
    """
    # Compute kernel matrices
    print("Computing quantum kernel matrix (training)...")
    K_train = quantum_kernel_matrix(X_train, n_qubits)

    print("Computing quantum kernel matrix (test)...")
    # For test, we need K(x_test, x_train) for all pairs
    n_train = len(X_train)
    n_test = len(X_test)
    K_test = np.zeros((n_test, n_train))

    train_states = [quantum_feature_map(x, n_qubits) for x in X_train]
    test_states = [quantum_feature_map(x, n_qubits) for x in X_test]

    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = abs(np.dot(test_states[i].conj(), train_states[j]))**2

    # Simple kernel classification (nearest centroid in kernel space)
    # For a proper SVM, you would use sklearn.svm.SVC(kernel='precomputed')
    y_signed = 2 * y_train - 1

    # Classify by weighted kernel sum
    predictions = []
    for i in range(n_test):
        # Score = Σ y_j * K(x_test, x_j_train)
        score = np.sum(y_signed * K_test[i])
        predictions.append(1 if score > 0 else 0)

    accuracy = np.mean(np.array(predictions) == y_test)
    return accuracy, K_train

# Generate dataset
np.random.seed(42)
n_train, n_test = 30, 10
X_all = np.random.uniform(-2, 2, (n_train + n_test, 2))
y_all = ((X_all[:, 0]**2 + X_all[:, 1]**2) < 2).astype(int)  # Circle boundary

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test, y_test = X_all[n_train:], y_all[n_train:]

print("=" * 55)
print("Quantum Kernel SVM")
print("=" * 55)
print(f"Training: {n_train} samples, Test: {n_test} samples\n")

accuracy, K = quantum_kernel_svm(X_train, y_train, X_test, y_test, n_qubits=2)
print(f"\nQuantum Kernel SVM accuracy: {accuracy:.2%}")

# Analyze kernel matrix structure
print(f"\nKernel matrix statistics:")
print(f"  Mean diagonal: {np.mean(np.diag(K)):.4f} (should be 1.0)")
print(f"  Mean off-diagonal: {np.mean(K[~np.eye(n_train, dtype=bool)]):.4f}")
print(f"  Min off-diagonal: {np.min(K[~np.eye(n_train, dtype=bool)]):.4f}")
print(f"  Max off-diagonal: {np.max(K[~np.eye(n_train, dtype=bool)]):.4f}")
```

### 8.4 불모 고원 시연

```python
import numpy as np

def random_parameterized_circuit(n_qubits, n_layers, params):
    """Apply a random hardware-efficient ansatz."""
    N = 2**n_qubits
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0

    # Initial Hadamard layer
    H_all = tensor_product([H_gate] * n_qubits)
    state = H_all @ state

    idx = 0
    for l in range(n_layers):
        # Ry rotations
        for q in range(n_qubits):
            ops = [I] * n_qubits
            ops[q] = ry_gate(params[idx])
            state = tensor_product(ops) @ state
            idx += 1

        # Rz rotations
        for q in range(n_qubits):
            ops = [I] * n_qubits
            ops[q] = rz_gate(params[idx])
            state = tensor_product(ops) @ state
            idx += 1

        # CNOT chain
        for q in range(n_qubits - 1):
            cnot = np.eye(N, dtype=complex)
            for s in range(N):
                if (s >> (n_qubits - 1 - q)) & 1:
                    new_s = s ^ (1 << (n_qubits - 1 - (q + 1)))
                    cnot[s, s] = 0
                    cnot[new_s, s] = 1
            state = cnot @ state

    return state

def estimate_gradient_variance(n_qubits, n_layers, n_samples=200):
    """Estimate the variance of the cost function gradient.

    Why measure gradient variance? If the variance decreases exponentially
    with n_qubits, we have a barren plateau: the gradient is essentially
    zero everywhere, making optimization impossible. This is the central
    challenge for scalable quantum ML.
    """
    n_params = 2 * n_qubits * n_layers
    N = 2**n_qubits

    # Cost function: ⟨Z_0⟩ (local cost)
    Z0 = np.zeros((N, N), dtype=complex)
    for s in range(N):
        Z0[s, s] = 1 - 2 * ((s >> (n_qubits - 1)) & 1)

    gradients = []
    for _ in range(n_samples):
        params = np.random.uniform(0, 2*np.pi, n_params)

        # Gradient of first parameter via parameter shift rule
        params_plus = params.copy()
        params_plus[0] += np.pi/2
        params_minus = params.copy()
        params_minus[0] -= np.pi/2

        state_plus = random_parameterized_circuit(n_qubits, n_layers, params_plus)
        state_minus = random_parameterized_circuit(n_qubits, n_layers, params_minus)

        e_plus = np.real(state_plus.conj() @ Z0 @ state_plus)
        e_minus = np.real(state_minus.conj() @ Z0 @ state_minus)

        grad = (e_plus - e_minus) / 2
        gradients.append(grad)

    return np.var(gradients), np.mean(np.abs(gradients))

print("=" * 55)
print("Barren Plateau Demonstration")
print("=" * 55)
print(f"\n{'n_qubits':>10} {'n_layers':>10} {'Var[∂L/∂θ]':>14} {'Mean|∂L/∂θ|':>14}")
print("-" * 52)

for n_layers in [2, 4]:
    for n_qubits in [2, 3, 4, 5, 6]:
        var_grad, mean_grad = estimate_gradient_variance(n_qubits, n_layers, n_samples=100)
        print(f"{n_qubits:10d} {n_layers:10d} {var_grad:14.6f} {mean_grad:14.6f}")

print("\nNote: If gradient variance decreases exponentially with n_qubits,")
print("this indicates a barren plateau — gradient-based training becomes")
print("exponentially harder as the system grows.")
```

---

## 9. 연습 문제

### 연습 문제 1: 인코딩 전략 비교

50개의 샘플을 가진 4차원 데이터셋에 대해:
(a) 기저, 진폭, 각도 인코딩을 구현하세요.
(b) 각 인코딩에 대한 양자 커널 행렬을 계산하세요.
(c) 어떤 인코딩이 가장 "구조화된" (가장 균일하지 않은) 커널 행렬을 생성하나요?
(d) 각 인코딩으로 커널 SVM을 훈련하고 분류 정확도를 비교하세요.

### 연습 문제 2: 특성 맵 표현력

특성 맵 레이어 수가 표현력에 미치는 영향을 조사하세요:
(a) $n = 2$개 큐비트를 가진 ZZ 특성 맵에 대해 1000개의 무작위 데이터 포인트를 생성하고 $L = 1, 2, 3, 4$ 레이어에 대한 쌍방향 커널 값 $K(\mathbf{x}, \mathbf{x}')$ 분포를 계산하세요.
(b) 분포는 어떻게 변하나요? (더 균일한 분포는 더 높은 표현력을 나타냅니다.)
(c) 각 경우에 대한 유효 차원(커널 행렬의 대각합(trace))을 계산하세요.

### 연습 문제 3: VQC vs 고전 SVM

같은 데이터셋에서 변분 양자 분류기와 고전 RBF 커널 SVM을 비교하세요:
(a) 비선형 경계(예: 동심원)를 가진 2D 데이터셋을 생성하세요.
(b) $n = 2$개 큐비트와 $L = 2$ 레이어를 가진 VQC를 훈련하세요.
(c) RBF 커널을 사용한 고전 SVM을 훈련하세요.
(d) 정확도, 훈련 시간, 조정 가능한 매개변수 수를 비교하세요.
(e) VQC가 SVM을 능가하는 데이터셋 복잡도가 있나요?

### 연습 문제 4: 불모 고원 스케일링

불모 고원 실험을 확장하세요:
(a) $n = 2, 3, 4, 5, 6, 7$개 큐비트와 $L = n$ 레이어에 대해 기울기 분산을 계산하세요.
(b) $n$에 대한 $\log(\text{Var}[\partial \mathcal{L}/\partial\theta])$를 도식화하세요. 관계가 선형인가요 (지수적 감소)?
(c) 지역 비용 함수 ($\langle Z_0 \rangle$) vs 전역 비용 함수 ($\langle Z_0 Z_1 \cdots Z_n \rangle$)로 반복하세요. 어느 쪽이 더 가파른 불모 고원을 가지나요?
(d) 파라미터를 무작위 대신 0 근처로 초기화해 보세요. 불모 고원이 완화되나요?

### 연습 문제 5: 양자 커널 설계

특정 데이터셋을 위한 사용자 정의 양자 커널을 설계하세요:
(a) 고전 RBF 커널이 70% 정확도만 달성하는 데이터셋을 생성하세요.
(b) 더 높은 정확도를 달성하는 양자 특성 맵을 설계하세요 (인코딩 회전과 얽힘 구조를 선택하여).
(c) 특성 맵 구조(레이어 수, 얽힘 토폴로지, 회전 축)를 체계적으로 변화시키고 정확도를 기록하세요.
(d) 효과적인 양자 특성 맵을 설계하기 위한 어떤 원칙이 도출되나요?

---

[← 이전: QAOA와 조합 최적화](14_QAOA.md) | [다음: 양자 컴퓨팅 현황과 미래 →](16_Landscape_and_Future.md)

# 레슨 13: 변분 양자 고유값 풀이(Variational Quantum Eigensolver, VQE)

[← 이전: 양자 텔레포테이션과 통신](12_Quantum_Teleportation.md) | [다음: QAOA와 조합 최적화 →](14_QAOA.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 분자 전자 구조 문제와 고전 컴퓨터로 다루기 어려운 이유를 설명할 수 있다
2. 변분 원리(variational principle)와 바닥 상태 에너지의 상한값을 보장하는 방법을 서술할 수 있다
3. 하이브리드 양자-고전 VQE 루프를 설명할 수 있다
4. 하드웨어 효율적 앤사츠(ansatz)와 화학적으로 영감을 받은 앤사츠(UCCSD)를 비교할 수 있다
5. 해밀토니안 인코딩 방법인 조르당-위그너(Jordan-Wigner) 변환과 브라비-키타에프(Bravyi-Kitaev) 변환을 설명할 수 있다
6. H₂ 분자에 대한 VQE 계산을 단계별로 수행할 수 있다
7. 실용적인 과제인 황무지 고원(barren plateau), 잡음, 최적화 수렴 문제를 논의할 수 있다

---

양자 화학은 양자 컴퓨터의 가장 유망한 근기(near-term) 응용 분야 중 하나로 널리 인정받고 있습니다. 전자 구조 문제 — 분자의 바닥 상태 에너지를 찾는 것 — 는 전자의 수가 늘어날수록 양자 상태의 수가 지수적으로 증가하기 때문에 고전 컴퓨터로는 지수적으로 어렵습니다. 수십 년의 알고리즘 개발에도 불구하고, 완전 배치 상호작용(Full Configuration Interaction, FCI) 같은 고전적 방법은 약 20개의 전자로 제한되며, DFT나 CCSD(T) 같은 근사 방법은 정확도를 희생하여 처리 가능성을 얻습니다.

2014년 Peruzzo 등이 제안한 변분 양자 고유값 풀이(VQE)는 잡음 있는 중간 규모 양자(NISQ) 장치를 위해 특별히 설계되었습니다. 깊은 결함 허용 회로가 필요한 쇼어(Shor) 알고리즘과 달리, VQE는 얕은 양자 회로와 고전적 최적화를 결합합니다. 이 하이브리드 접근법은 회로 깊이를 다수의 회로 반복으로 교환하여 잡음에 더 강인하게 만들며, 이는 현재 하드웨어의 주요 과제입니다.

> **비유:** VQE는 라디오를 조율하는 것과 같습니다. 양자 회로는 후보 해(정적)를 생성하고, 고전 최적화기는 신호(에너지)가 가장 선명해질 때(최소)까지 다이얼(매개변수)을 조정합니다. 라디오 내부 작동 원리(전체 양자 상태)를 이해할 필요는 없습니다 — 출력 품질(에너지 기댓값)을 평가하고 조정하기만 하면 됩니다.

## 목차

1. [전자 구조 문제](#1-전자-구조-문제)
2. [변분 원리](#2-변분-원리)
3. [VQE 알고리즘](#3-vqe-알고리즘)
4. [앤사츠 설계](#4-앤사츠-설계)
5. [해밀토니안 인코딩](#5-해밀토니안-인코딩)
6. [H₂에 대한 VQE: 완전한 단계별 안내](#6-h₂에-대한-vqe-완전한-단계별-안내)
7. [고전 최적화기 전략](#7-고전-최적화기-전략)
8. [과제와 한계](#8-과제와-한계)
9. [파이썬 구현](#9-파이썬-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 전자 구조 문제

### 1.1 문제 설명

고정된 위치에 $M$개의 원자핵이 있는 분자(보른-오펜하이머(Born-Oppenheimer) 근사)와 $N$개의 전자가 있을 때, 전자 해밀토니안의 바닥 상태 에너지 $E_0$를 구합니다:

$$\hat{H} = -\sum_{i=1}^{N} \frac{\nabla_i^2}{2} - \sum_{i=1}^{N}\sum_{A=1}^{M} \frac{Z_A}{|\mathbf{r}_i - \mathbf{R}_A|} + \sum_{i<j}^{N} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$$

세 항은 각각 다음을 나타냅니다:
1. **전자의 운동 에너지(Kinetic energy)**
2. **전자-핵 인력(Electron-nucleus attraction)** (쿨롱(Coulomb))
3. **전자-전자 반발(Electron-electron repulsion)** (쿨롱(Coulomb))

### 1.2 어려운 이유

두 번째 양자화(second quantization)에서, 해밀토니안은 $K$개의 분자 궤도함수(molecular orbital)에 작용하는 생성 연산자($a_p^\dagger$)와 소멸 연산자($a_q$)로 표현됩니다:

$$\hat{H} = \sum_{pq} h_{pq} \, a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} \, a_p^\dagger a_q^\dagger a_r a_s$$

힐베르트 공간 차원은 $\binom{2K}{N}$으로 지수적으로 증가합니다. 카페인($C_8H_{10}N_4O_2$)과 같은 적당한 분자도 약 100개의 전자와 200개의 궤도함수를 가지며, 힐베르트 공간은 $\sim 10^{75}$차원에 달해 — 어떤 고전 컴퓨터도 감당할 수 없습니다.

### 1.3 고전적 접근법과 한계

| 방법 | 정확도 | 확장성 | 최대 시스템 크기 |
|--------|----------|---------|-----------------|
| 하트리-포크(Hartree-Fock, HF) | ~99% 상관 에너지 누락 | $O(K^4)$ | 수천 원자 |
| 밀도 범함수 이론(DFT) | ~1 kcal/mol (범함수에 따라 다름) | $O(K^3)$ | 수천 원자 |
| MP2 | 약한 상관에 좋음 | $O(K^5)$ | 수백 원자 |
| CCSD(T) | "황금 표준" ~1 kcal/mol | $O(K^7)$ | ~30 원자 |
| FCI (정확) | 정확 (기저 내에서) | $O(K!)$ | ~20 전자 |

양자 우위 가설: $K$개의 큐비트를 가진 양자 컴퓨터는 전체 $\binom{2K}{N}$ 힐베르트 공간을 직접 나타낼 수 있어, 다항 시간 내에 전자 구조 문제를 풀 수 있을 것입니다.

---

## 2. 변분 원리

### 2.1 서술

임의의 정규화된 시험 상태(trial state) $|\psi(\boldsymbol{\theta})\rangle$에 대해:

$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle \geq E_0$$

여기서 $E_0$는 참된 바닥 상태 에너지입니다. 등호는 $|\psi(\boldsymbol{\theta})\rangle$이 바닥 상태일 때만 성립합니다.

### 2.2 증명

시험 상태를 $\hat{H}$의 고유 기저로 전개합니다: $|\psi\rangle = \sum_i c_i |E_i\rangle$ 여기서 $\hat{H}|E_i\rangle = E_i|E_i\rangle$이고 $E_0 \leq E_1 \leq \cdots$입니다.

$$\langle\psi|\hat{H}|\psi\rangle = \sum_i |c_i|^2 E_i \geq E_0 \sum_i |c_i|^2 = E_0$$

### 2.3 VQE에 대한 의미

변분 원리는 다음을 보장합니다:
1. 어떤 시험 상태도 바닥 상태 에너지의 **상한값**을 제공한다
2. 매개변수 공간 $\boldsymbol{\theta}$에서 에너지를 **최소화**하면 앤사츠 계열 내에서 가장 좋은 근사를 얻는다
3. 전체 고유 분해를 계산할 필요가 없다 — $\hat{H}$의 기댓값만 필요하다

이것이 VQE의 기반입니다: 양자 컴퓨터를 사용하여 $|\psi(\boldsymbol{\theta})\rangle$를 준비하고 $\langle\hat{H}\rangle$를 추정한 다음, 고전 최적화기를 사용하여 에너지를 최소화하는 매개변수 $\boldsymbol{\theta}^*$를 찾습니다.

---

## 3. VQE 알고리즘

### 3.1 알고리즘 개요

```
┌──────────────────────────────────────────────────┐
│              Classical Computer                   │
│                                                   │
│  1. Initialize parameters θ                       │
│  2. Receive E(θ) from quantum computer            │
│  3. Update θ using classical optimizer             │
│  4. Check convergence → if not, go to step 5      │
│  5. Send updated θ to quantum computer             │
│                                                   │
└───────────────┬──────────────────┬───────────────┘
                │                  ↑
                ↓                  │
┌───────────────┴──────────────────┴───────────────┐
│              Quantum Computer                     │
│                                                   │
│  1. Prepare |ψ(θ)⟩ using parameterized circuit    │
│  2. Measure ⟨ψ(θ)|H|ψ(θ)⟩ term by term          │
│  3. Return E(θ) to classical computer              │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 3.2 상세 단계

**단계 1: 해밀토니안 분해**

$\hat{H}$를 파울리(Pauli) 문자열의 합으로 표현합니다 (큐비트 매핑 후):

$$\hat{H} = \sum_i c_i P_i, \quad P_i \in \{I, X, Y, Z\}^{\otimes n}$$

예를 들어, $\hat{H} = 0.5 \, Z_0 Z_1 - 0.3 \, X_0 + 0.2 \, I$.

**단계 2: 매개변수 초기화**

초기 매개변수 $\boldsymbol{\theta}_0$을 선택합니다 (보통 무작위로 또는 하트리-포크 결과로부터).

**단계 3: 시험 상태 준비**

매개변수화된 양자 회로(앤사츠) $U(\boldsymbol{\theta})$를 초기 상태 $|0\rangle^{\otimes n}$에 적용합니다:

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta})|0\rangle^{\otimes n}$$

**단계 4: 에너지 추정**

각 파울리 문자열 $P_i$에 대해 반복적인 준비-측정을 통해 $\langle P_i \rangle$를 측정합니다. 합산:

$$E(\boldsymbol{\theta}) = \sum_i c_i \langle P_i \rangle$$

각 $\langle P_i \rangle$는 통계적 정확도를 위해 많은 "샷(shot)"(반복)이 필요합니다.

**단계 5: 고전 최적화**

$E(\boldsymbol{\theta})$를 고전 최적화기(COBYLA, L-BFGS-B, SPSA 등)에 제공하여 업데이트된 매개변수를 얻습니다.

**단계 6: 수렴할 때까지 반복.**

### 3.3 측정 횟수

$\hat{H}$에 $M$개의 파울리 항이 있고 에너지 정밀도 $\epsilon$을 원한다면, 각 항은 $O(c_i^2 / \epsilon^2)$번의 측정이 필요합니다. 총 측정 횟수는 다음과 같이 확장됩니다:

$$N_{\text{shots}} \sim O\left(\frac{(\sum_i |c_i|)^2}{\epsilon^2}\right)$$

전형적인 분자의 경우 $M$은 수백에서 수백만 개의 파울리 항에 이르러, 측정 오버헤드가 중요한 병목이 됩니다.

---

## 4. 앤사츠 설계

### 4.1 앤사츠란 무엇인가?

**앤사츠(ansatz)**(독일어로 "접근" 또는 "가정")는 시험 상태의 계열을 정의하는 매개변수화된 양자 회로 $U(\boldsymbol{\theta})$입니다. 앤사츠의 선택은 다음을 결정적으로 좌우합니다:

- **표현력(Expressibility)**: 앤사츠가 참된 바닥 상태를 나타낼 수 있는가?
- **훈련 가능성(Trainability)**: 최적화기가 좋은 매개변수를 찾을 수 있는가?
- **하드웨어 효율성(Hardware efficiency)**: 회로가 얼마나 깊은가?

### 4.2 하드웨어 효율적 앤사츠(Hardware-Efficient Ansatz, HEA)

HEA는 하드웨어가 기본적으로 지원하는 게이트를 레이어로 배열하여 사용합니다:

```
Layer 1:        Layer 2:        ...
|0⟩ ─ Ry(θ₁) ─ CNOT ─ Ry(θ₅) ─ CNOT ─ ...
|0⟩ ─ Ry(θ₂) ─ ──●── Ry(θ₆) ─ ──●── ...
|0⟩ ─ Ry(θ₃) ─ CNOT ─ Ry(θ₇) ─ CNOT ─ ...
|0⟩ ─ Ry(θ₄) ─ ──●── Ry(θ₈) ─ ──●── ...
```

**장점**: 짧은 회로 깊이, 하드웨어 기본 게이트
**단점**: 화학적 직관 없음, 황무지 고원(barren plateau, 8절 참조)에 취약, 많은 매개변수가 필요할 수 있음

### 4.3 유니터리 결합 클러스터(Unitary Coupled Cluster, UCCSD)

UCCSD 앤사츠는 양자 화학의 결합 클러스터(coupled-cluster) 방법에서 영감을 받았습니다:

$$U_{\text{UCCSD}}(\boldsymbol{\theta}) = e^{T(\boldsymbol{\theta}) - T^\dagger(\boldsymbol{\theta})}$$

여기서 $T = T_1 + T_2$는 단일(single) 및 이중(double) 들뜸(excitation)을 포함합니다:

$$T_1 = \sum_{i,a} \theta_i^a \, a_a^\dagger a_i, \quad T_2 = \sum_{ij,ab} \theta_{ij}^{ab} \, a_a^\dagger a_b^\dagger a_j a_i$$

여기서 $i, j$는 점유(occupied) 궤도함수 인덱스이고 $a, b$는 가상(virtual, 비점유) 궤도함수 인덱스입니다.

**장점**: 화학적으로 동기 부여됨, 체계적으로 개선 가능
**단점**: 깊은 회로(많은 트로터(Trotter) 단계), 대형 시스템에서 많은 매개변수

### 4.4 기타 앤사츠

| 앤사츠 | 설명 | 깊이 | 매개변수 |
|--------|-------------|-------|-----------|
| HEA | 하드웨어 레이어 | $O(L)$ 레이어 | $O(nL)$ |
| UCCSD | 화학적 들뜸 | $O(n^4)$ | $O(n^2 K^2)$ |
| ADAPT-VQE | 반복적으로 성장 | 가변 | 적응형 |
| 대칭 보존 | 대칭을 존중함 | $O(n^2)$ | 감소됨 |
| 큐비트-ADAPT | 큐비트 수준 ADAPT | 가변 | 적응형 |

---

## 5. 해밀토니안 인코딩

### 5.1 인코딩 문제

전자 해밀토니안은 페르미온 연산자($a_p^\dagger$, $a_q$)로 표현되지만, 양자 컴퓨터는 큐비트(스핀-1/2 시스템)에서 작동합니다. 페르미온에서 큐비트로의 매핑이 필요합니다.

### 5.2 조르당-위그너(Jordan-Wigner) 변환

조르당-위그너(JW) 매핑은 각 분자 궤도함수를 하나의 큐비트에 할당합니다:

$$a_p^\dagger \to \frac{1}{2}(X_p - iY_p) \otimes Z_{p-1} \otimes Z_{p-2} \otimes \cdots \otimes Z_0$$

$Z$ 연산자 문자열은 페르미온 반교환 관계 $\{a_p, a_q^\dagger\} = \delta_{pq}$를 강제합니다.

**예시**: 4개의 궤도함수에 대해:
- $a_0^\dagger = \frac{1}{2}(X_0 - iY_0)$
- $a_1^\dagger = \frac{1}{2}(X_1 - iY_1) \otimes Z_0$
- $a_2^\dagger = \frac{1}{2}(X_2 - iY_2) \otimes Z_1 \otimes Z_0$

**장점**: 단순하고 직관적 (큐비트 $p$ = 궤도함수 $p$)
**단점**: 시스템 크기에 따라 $Z$-문자열이 증가하여 일부 연산자가 매우 비국소적(non-local)이 됨

### 5.3 브라비-키타에프(Bravyi-Kitaev) 변환

BK 매핑은 이진 트리 구조를 사용하여 $O(n)$ 대신 $O(\log n)$ 비국소성을 달성합니다:

$$a_p^\dagger \to O(\log n) \text{-weight Pauli operators}$$

이는 큐비트 해밀토니안의 항 수를 줄이고 더 효율적인 회로로 이어질 수 있습니다.

### 5.4 예시: H₂ 해밀토니안

최소 STO-3G 기저와 JW 변환을 사용하면, H₂ 해밀토니안(4 큐비트, 대칭성으로 2 큐비트로 감소)은 다음이 됩니다:

$$\hat{H}_{H_2} = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0 Z_1 + g_4 X_0 X_1 + g_5 Y_0 Y_1$$

여기서 $g_i$ 계수는 결합 길이 $R$에 따라 다릅니다.

$R = 0.735$ \AA (평형 상태)에서:

$$\hat{H} \approx -0.4804 I + 0.3435 Z_0 - 0.4347 Z_1 + 0.5716 Z_0 Z_1 + 0.0910 X_0 X_1 + 0.0910 Y_0 Y_1$$

---

## 6. H₂에 대한 VQE: 완전한 단계별 안내

### 6.1 설정

- **분자**: 결합 길이 $R$의 H₂
- **기저**: STO-3G (최소 기저, 2개 공간 궤도함수, 4개 스핀-궤도함수)
- **대칭성 감소**: 4 큐비트 → 2 큐비트 ($\mathbb{Z}_2$ 대칭성 사용)
- **앤사츠**: 단일 매개변수 $R_y(\theta)$ 회로

### 6.2 앤사츠 회로

2-큐비트 축소 H₂ 문제에는 간단한 앤사츠로 충분합니다:

```
|0⟩ ─── X ─── Ry(θ) ─── CNOT ─── M
|0⟩ ─────────────────── ──●──── M
```

$X$ 게이트는 큐비트 0을 $|1\rangle$로 초기화합니다 (하트리-포크 상태 $|01\rangle$ 표현). $R_y(\theta)$ 게이트와 CNOT은 $\theta$로 매개변수화된 얽힘(entanglement)을 만듭니다.

$\theta = 0$일 때: 상태는 $|10\rangle$ (하트리-포크)
$\theta = \pi$일 때: 상태는 $|01\rangle$
중간 $\theta$: 배치(configuration)의 중첩(superposition)

### 6.3 θ의 함수로서의 에너지

에너지 $E(\theta)$는 각 파울리 항을 측정하여 계산됩니다:

$$E(\theta) = g_0 + g_1 \langle Z_0\rangle + g_2 \langle Z_1\rangle + g_3 \langle Z_0 Z_1\rangle + g_4 \langle X_0 X_1\rangle + g_5 \langle Y_0 Y_1\rangle$$

단순한 앤사츠의 경우, 이것은 단일 매개변수 $\theta$의 함수로 그래프로 그리고 최소화할 수 있습니다.

### 6.4 결합 해리 곡선(Bond Dissociation Curve)

서로 다른 결합 길이 $R$에서 VQE를 실행하면, 퍼텐셜 에너지 면(potential energy surface) $E(R)$을 얻습니다. 최솟값은 평형 결합 길이와 해리 에너지를 제공합니다.

- **하트리-포크**는 평형 근처에서 좋은 결과를 제공하지만 해리 시 실패함
- **VQE**는 큰 $R$에서의 강한 상관관계를 포착할 수 있음 (두 전자가 별도 원자에 국소화되는 경우)
- **FCI** (정확한)는 기준점(benchmark)

---

## 7. 고전 최적화기 전략

### 7.1 기울기 없는(Gradient-Free) 방법

| 최적화기 | 설명 | 장점 | 단점 |
|-----------|-------------|------|------|
| COBYLA | 선형 근사에 의한 제약 최적화(Constrained Optimization By Linear Approximation) | 강건함, 기울기 불필요 | 느린 수렴 |
| 넬더-미드(Nelder-Mead) | 단체(simplex) 방법 | 간단함, 기울기 불필요 | 고차원에서 나쁨 |
| 파웰(Powell) | 방향 집합 방법 | 부드러운 지형에 좋음 | 잡음에 민감 |

### 7.2 기울기 기반(Gradient-Based) 방법

| 최적화기 | 설명 | 장점 | 단점 |
|-----------|-------------|------|------|
| L-BFGS-B | 준-뉴턴(Quasi-Newton) | 빠른 수렴 | 정확한 기울기 필요 |
| Adam | 적응형 학습률 | 잡음 있는 기울기에 좋음 | 하이퍼파라미터 조정 |
| SPSA | 동시 교란(Simultaneous Perturbation) | 단계당 2번 평가 | 잡음 있는 수렴 |

### 7.3 매개변수 이동 규칙(Parameter Shift Rule)

양자 기울기는 **매개변수 이동 규칙**을 사용하여 정확하게 계산할 수 있습니다. 게이트 $R_y(\theta) = e^{-i\theta Y/2}$에 대해:

$$\frac{\partial E}{\partial \theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$$

이는 매개변수당 단 두 번의 추가 회로 평가만 필요하므로, 양자 하드웨어에서 기울기 기반 최적화가 실현 가능합니다.

---

## 8. 과제와 한계

### 8.1 황무지 고원(Barren Plateau)

무작위 매개변수화된 회로에서, 기울기 분산은 큐비트 수에 따라 **지수적으로** 감소합니다:

$$\text{Var}\left[\frac{\partial E}{\partial \theta_i}\right] \leq O(2^{-n})$$

이는 대형 시스템에서 에너지 지형이 지수적으로 평탄해져(황무지 고원) 기울기 기반 최적화가 비효율적이 됨을 의미합니다 — 기울기가 너무 작아 유용한 방향을 제공하지 못합니다.

**완화 전략**:
- 문제별 앤사츠 사용 (UCCSD, 대칭 보존)
- 레이어별 훈련
- 고전적으로 계산된 상태(예: 하트리-포크) 근처에서 초기화

### 8.2 잡음(Noise)

NISQ 장치에서는 게이트 오류($\sim 10^{-3}$~$10^{-2}$)와 결맞음 소실(decoherence)이 양자 상태를 손상시킵니다. $d$개의 게이트를 가진 회로에서 기대 충실도(fidelity)는 다음과 같이 감소합니다:

$$F \approx (1 - p)^d \approx e^{-pd}$$

이는 유용한 회로 깊이를 $d \lesssim 1/p \sim 100\text{-}1000$개 게이트로 제한합니다.

**완화 전략**:
- 오류 완화 (영-잡음 외삽(zero-noise extrapolation), 확률적 오류 취소(probabilistic error cancellation))
- 얕은 앤사츠 (하드웨어 효율적)
- 잡음 인식 최적화

### 8.3 측정 오버헤드(Measurement Overhead)

$\langle H \rangle$를 정밀도 $\epsilon$으로 추정하려면 $O(M / \epsilon^2)$번의 측정이 필요하며, 여기서 $M$은 파울리 항의 수입니다. 대형 분자의 경우 수십억 번의 회로 실행이 필요할 수 있습니다.

**완화 전략**:
- 교환 가능한 파울리 항 묶기 (동시 측정)
- 고전 그림자(classical shadows)
- 중요도 샘플링(importance sampling)

### 8.4 양자 우위 문제

NISQ 장치의 VQE가 실용적으로 관련된 분자에 대해 최선의 고전적 방법을 능가할 수 있는지는 여전히 열린 문제입니다. 현재의 시연은 고전적으로 쉽게 풀 수 있는 소형 분자(H₂, LiH, BeH₂)로 제한됩니다.

---

## 9. 파이썬 구현

### 9.1 2-큐비트 해밀토니안에 대한 간단한 VQE

```python
import numpy as np
from scipy.optimize import minimize

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(A, B):
    """Tensor (Kronecker) product of two matrices."""
    return np.kron(A, B)

def build_h2_hamiltonian(R=0.735):
    """Build the 2-qubit H₂ Hamiltonian for a given bond length.

    Why these specific coefficients? They come from computing the molecular
    integrals in the STO-3G basis and applying the Jordan-Wigner transformation.
    The coefficients vary with bond length R, defining the potential energy surface.
    """
    # Coefficients for H₂ in STO-3G basis (2-qubit reduced Hamiltonian)
    # These are approximate values at equilibrium bond length R=0.735 Å
    # In a real application, these would be computed from molecular integrals
    if abs(R - 0.735) < 0.01:
        g = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
    else:
        # Simplified parametric model for demonstration
        # Real computation would use PySCF or similar
        g = [-0.4804 + 0.1*(R-0.735),
             0.3435 * np.exp(-0.5*(R-0.735)**2),
             -0.4347 * np.exp(-0.3*(R-0.735)**2),
             0.5716 / (1 + 0.5*abs(R-0.735)),
             0.0910 * np.exp(-0.2*(R-0.735)**2),
             0.0910 * np.exp(-0.2*(R-0.735)**2)]

    H = (g[0] * tensor(I, I) +
         g[1] * tensor(Z, I) +
         g[2] * tensor(I, Z) +
         g[3] * tensor(Z, Z) +
         g[4] * tensor(X, X) +
         g[5] * tensor(Y, Y))

    return H, g

def ansatz_state(theta):
    """Prepare the VQE trial state for the 2-qubit H₂ problem.

    Why this specific circuit? Starting from the Hartree-Fock state |10⟩,
    the Ry rotation and CNOT create a parameterized superposition of
    |10⟩ and |01⟩ — the two dominant configurations for H₂. The parameter
    θ controls the mixing, and the optimizer finds the optimal value.
    """
    # Start with |00⟩
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Apply X to qubit 0: |00⟩ → |10⟩ (Hartree-Fock state)
    X_gate = tensor(X, I)
    state = X_gate @ state

    # Apply Ry(θ) to qubit 0
    Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                   [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
    Ry_full = tensor(Ry, I)
    state = Ry_full @ state

    # Apply CNOT (qubit 0 controls qubit 1)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    state = CNOT @ state

    return state

def vqe_energy(theta, H):
    """Compute the energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

    On a real quantum computer, this would be estimated from many measurement
    shots. Here we compute it exactly from the state vector.
    """
    state = ansatz_state(theta[0])
    return np.real(state.conj() @ H @ state)

def run_vqe(R=0.735, verbose=True):
    """Run the complete VQE algorithm for H₂.

    Returns the optimized energy and parameters.
    """
    H, g = build_h2_hamiltonian(R)

    # Exact ground state energy (for comparison)
    eigenvalues = np.linalg.eigvalsh(H)
    exact_energy = eigenvalues[0]

    # Hartree-Fock energy (θ=0)
    hf_energy = vqe_energy([0], H)

    if verbose:
        print(f"Bond length R = {R:.3f} Å")
        print(f"Exact ground state energy: {exact_energy:.6f} Ha")
        print(f"Hartree-Fock energy (θ=0): {hf_energy:.6f} Ha")

    # Run optimization
    result = minimize(vqe_energy, x0=[0.1], args=(H,), method='COBYLA',
                     options={'maxiter': 200, 'rhobeg': 0.5})

    vqe_e = result.fun
    opt_theta = result.x[0]

    if verbose:
        print(f"VQE optimized energy: {vqe_e:.6f} Ha")
        print(f"Optimal θ: {opt_theta:.4f} rad")
        print(f"Error: {abs(vqe_e - exact_energy):.2e} Ha")
        print(f"Chemical accuracy (1.6 mHa): "
              f"{'ACHIEVED' if abs(vqe_e - exact_energy) < 0.0016 else 'NOT achieved'}")

    return vqe_e, exact_energy, opt_theta

# Run VQE at equilibrium
print("=" * 55)
print("VQE for H₂ Molecule")
print("=" * 55)
run_vqe(0.735)
```

### 9.2 에너지 지형 시각화

```python
import numpy as np

def visualize_energy_landscape(R=0.735):
    """Plot the energy as a function of the ansatz parameter θ.

    Why visualize? The energy landscape reveals the optimization difficulty.
    A smooth, single-minimum landscape is easy to optimize; multiple local
    minima or flat regions indicate potential convergence problems.
    """
    H, g = build_h2_hamiltonian(R)

    thetas = np.linspace(-np.pi, np.pi, 200)
    energies = [vqe_energy([t], H) for t in thetas]

    # Exact energies
    eigenvalues = sorted(np.linalg.eigvalsh(H))

    print(f"Energy landscape for H₂ at R = {R:.3f} Å")
    print(f"{'θ':>8} {'E(θ)':>12} {'Distance from E_0':>18}")
    print("-" * 42)

    # Sample a few points
    for t in np.linspace(-np.pi, np.pi, 13):
        e = vqe_energy([t], H)
        dist = e - eigenvalues[0]
        bar = "#" * int(min(dist * 200, 40))
        print(f"{t:8.3f} {e:12.6f} {dist:18.6f} {bar}")

    # Find the minimum
    min_idx = np.argmin(energies)
    print(f"\nMinimum at θ = {thetas[min_idx]:.4f}, E = {energies[min_idx]:.6f}")
    print(f"Exact E₀ = {eigenvalues[0]:.6f}")
    print(f"Exact E₁ = {eigenvalues[1]:.6f}")

visualize_energy_landscape()
```

### 9.3 결합 해리 곡선

```python
import numpy as np
from scipy.optimize import minimize

def bond_dissociation_curve():
    """Compute the H₂ potential energy surface using VQE.

    Why scan bond lengths? The PES reveals the equilibrium geometry
    (minimum energy) and dissociation behavior. VQE should capture
    the correct dissociation limit where Hartree-Fock fails.
    """
    print("=" * 55)
    print("H₂ Bond Dissociation Curve: VQE vs HF vs Exact")
    print("=" * 55)

    bond_lengths = np.linspace(0.3, 3.0, 28)
    vqe_energies = []
    hf_energies = []
    exact_energies = []

    for R in bond_lengths:
        H, g = build_h2_hamiltonian(R)
        eigenvalues = sorted(np.linalg.eigvalsh(H))

        # Exact
        exact_energies.append(eigenvalues[0])

        # Hartree-Fock (θ = 0)
        hf_e = vqe_energy([0], H)
        hf_energies.append(hf_e)

        # VQE
        result = minimize(vqe_energy, x0=[0.1], args=(H,), method='COBYLA')
        vqe_energies.append(result.fun)

    # Print comparison table
    print(f"\n{'R (Å)':>8} {'Exact':>10} {'VQE':>10} {'HF':>10} {'VQE err':>10}")
    print("-" * 52)
    for i, R in enumerate(bond_lengths):
        if i % 3 == 0:  # Print every 3rd point
            err = abs(vqe_energies[i] - exact_energies[i])
            print(f"{R:8.3f} {exact_energies[i]:10.6f} {vqe_energies[i]:10.6f} "
                  f"{hf_energies[i]:10.6f} {err:10.2e}")

    # Find equilibrium
    min_idx = np.argmin(exact_energies)
    print(f"\nEquilibrium bond length: R = {bond_lengths[min_idx]:.3f} Å")
    print(f"Equilibrium energy (exact): {exact_energies[min_idx]:.6f} Ha")

bond_dissociation_curve()
```

### 9.4 샷 잡음을 포함한 측정 시뮬레이션

```python
import numpy as np

def measure_pauli_expectation(state, pauli_ops, n_shots=1000):
    """Simulate measuring a Pauli operator with finite shots.

    Why simulate shots? On a real quantum computer, we estimate ⟨P⟩ from
    a finite number of measurements. Each measurement gives +1 or -1
    with probabilities determined by the quantum state. The statistical
    error scales as 1/√(n_shots).
    """
    # Build the full Pauli operator
    op = pauli_ops[0]
    for p in pauli_ops[1:]:
        op = np.kron(op, p)

    # Compute exact expectation value for comparison
    exact = np.real(state.conj() @ op @ state)

    # Simulate measurement: diagonalize the Pauli operator
    eigenvalues, eigenvectors = np.linalg.eigh(op)
    # Probability of each eigenvalue
    probs = np.abs(eigenvectors.conj().T @ state)**2

    # Sample from the distribution
    outcomes = np.random.choice(eigenvalues, size=n_shots, p=probs)
    estimated = np.mean(outcomes)
    std_error = np.std(outcomes) / np.sqrt(n_shots)

    return estimated, std_error, exact

def vqe_with_shot_noise(theta, H_terms, n_shots=1000):
    """Run VQE energy estimation with simulated shot noise.

    This demonstrates the realistic scenario where each Pauli term
    must be estimated from a finite number of measurements.
    """
    state = ansatz_state(theta)

    # H₂ Hamiltonian terms: (coefficient, [pauli_ops])
    total_energy = 0
    total_variance = 0

    for coeff, ops in H_terms:
        if all(np.allclose(op, I) for op in ops):
            # Identity term: exact, no measurement needed
            total_energy += coeff
            continue

        est, err, exact = measure_pauli_expectation(state, ops, n_shots)
        total_energy += coeff * est
        total_variance += (coeff * err)**2

    return total_energy, np.sqrt(total_variance)

# Demonstrate shot noise effect
print("=" * 55)
print("VQE with Finite Measurement Shots")
print("=" * 55)

H, g = build_h2_hamiltonian(0.735)
H_terms = [
    (g[0], [I, I]),
    (g[1], [Z, I]),
    (g[2], [I, Z]),
    (g[3], [Z, Z]),
    (g[4], [X, X]),
    (g[5], [Y, Y]),
]

theta_opt = 0.2267  # Approximately optimal

exact_energy = vqe_energy([theta_opt], H)
print(f"Exact energy at θ={theta_opt:.4f}: {exact_energy:.6f} Ha\n")

for n_shots in [10, 100, 1000, 10000, 100000]:
    energies = [vqe_with_shot_noise(theta_opt, H_terms, n_shots)[0]
                for _ in range(20)]
    mean_e = np.mean(energies)
    std_e = np.std(energies)
    print(f"  {n_shots:>7} shots: E = {mean_e:.6f} ± {std_e:.6f} Ha "
          f"(error: {abs(mean_e - exact_energy):.6f})")
```

---

## 10. 연습 문제

### 연습 문제 1: 다른 앤사츠를 이용한 VQE

2-큐비트 H₂ 문제에 대한 두 가지 추가 앤사츠를 구현합니다:
(a) **두 매개변수 앤사츠**: 큐비트 0에 $R_y(\theta_1)$, CNOT, 큐비트 1에 $R_y(\theta_2)$.
(b) **세 매개변수 앤사츠**: 큐비트 0에 $R_y(\theta_1) R_z(\theta_2)$, CNOT, 큐비트 1에 $R_y(\theta_3)$.
각 앤사츠에서 달성 가능한 최소 에너지를 비교합니다. 매개변수를 늘리는 것이 항상 도움이 되는가?

### 연습 문제 2: 매개변수 지형 분석

$R = 0.735$ \AA에서 2-큐비트 H₂ VQE에 대해:
(a) $\theta \in [-\pi, \pi]$에 대해 200개의 점을 사용하여 $E(\theta)$를 그립니다.
(b) 지역 최솟값(local minima)이 몇 개 있는가?
(c) 매개변수 이동 규칙을 사용하여 $dE/d\theta$를 계산합니다. 유한 차분법(finite differences)과 일치하는지 확인합니다.
(d) 20개의 무작위 초기점에서 최적화기를 실행합니다. 전역 최솟값(global minimum)을 얼마나 자주 찾는가?

### 연습 문제 3: 샷 잡음과 수렴

측정 잡음이 VQE 수렴에 미치는 영향을 조사합니다:
(a) 에너지 평가당 100, 1000, 10000 샷으로 VQE 최적화를 실행합니다.
(b) 각 경우에 대해 수렴 곡선(에너지 vs 반복 횟수)을 그립니다.
(c) 화학적 정확도($1.6 \times 10^{-3}$ Ha)를 달성하는 데 총 몇 샷이 필요한가?
(d) 샷 잡음 하에서 COBYLA(기울기 없는)와 SPSA(확률적 기울기)를 비교합니다.

### 연습 문제 4: 일반 2-큐비트 해밀토니안

일반 2-큐비트 해밀토니안 $H = J_x X_0 X_1 + J_y Y_0 Y_1 + J_z Z_0 Z_1 + h (Z_0 + Z_1)$ (장(field) 내의 하이젠베르크(Heisenberg) 모델)을 고려합니다.
(a) 2-레이어 하드웨어 효율적 앤사츠로 이 해밀토니안에 대한 VQE 풀이기를 작성합니다.
(b) 상도(phase diagram)를 계산합니다: $J_x = J_y = J_z = J$ 및 $h/J$의 함수로 바닥 상태 에너지.
(c) VQE 결과와 정확한 대각화(exact diagonalization)를 비교합니다.

### 연습 문제 5: 황무지 고원 조사

(a) $n$개의 큐비트와 $L$개의 레이어로 무작위 하드웨어 효율적 앤사츠를 구현합니다.
(b) $n = 2, 4, 6, 8$ 큐비트에 대해 1000개의 무작위 매개변수 초기화에서 $\partial E / \partial \theta_1$의 분산을 계산합니다.
(c) 로그 스케일에서 분산 vs $n$을 그립니다. 지수적으로 감소하는가?
(d) 문제별 초기화(예: 하트리-포크 근처)로 반복합니다. 황무지 고원이 완화되는가?

---

[← 이전: 양자 텔레포테이션과 통신](12_Quantum_Teleportation.md) | [다음: QAOA와 조합 최적화 →](14_QAOA.md)

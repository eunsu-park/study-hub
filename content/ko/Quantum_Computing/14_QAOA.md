# 레슨 14: QAOA와 조합 최적화(Combinatorial Optimization)

[← 이전: 변분 양자 고유값 풀이](13_VQE.md) | [다음: 양자 기계 학습 →](15_Quantum_Machine_Learning.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 조합 최적화 문제를 이징(Ising) 해밀토니안으로 공식화할 수 있다
2. 최대 컷(MaxCut) 문제와 그 그래프 이론적 공식화를 설명할 수 있다
3. 교대 비용 및 혼합 유니터리로 QAOA 회로를 구성할 수 있다
4. 비용 해밀토니안(cost Hamiltonian)과 혼합 해밀토니안(mixing Hamiltonian)의 역할을 설명할 수 있다
5. $p$-레이어 QAOA의 깊이-성능 트레이드오프를 분석할 수 있다
6. QAOA와 단열 양자 컴퓨팅(adiabatic quantum computing)의 연관성을 이해할 수 있다
7. MaxCut에 대한 QAOA를 구현하고 전수조사(brute-force) 해와 비교할 수 있다

---

조합 최적화(Combinatorial Optimization)는 컴퓨터 과학과 운영 연구(operations research)에서 가장 중요한 문제 부류 중 하나입니다. 항공 노선 스케줄링부터 회로 설계, 포트폴리오 최적화부터 신약 개발까지, 수많은 현실 문제가 지수적으로 많은 가능성 중에서 최선의 구성을 찾는 것으로 귀결됩니다. 이러한 문제의 대부분은 NP-난해(NP-hard)하여, 최악의 경우를 효율적으로 풀 수 있는 고전적 알고리즘이 알려져 있지 않습니다.

2014년 Farhi, Goldstone, Gutmann이 도입한 양자 근사 최적화 알고리즘(Quantum Approximate Optimization Algorithm, QAOA)은 양자 컴퓨터에서 조합 최적화를 공략하는 체계적인 프레임워크를 제공합니다. VQE(레슨 13)와 마찬가지로, QAOA는 NISQ 장치를 위해 설계된 하이브리드 양자-고전 알고리즘입니다. 최적화 문제를 양자 해밀토니안으로 인코딩하고, 구조화된 매개변수화 회로를 사용하여 근사해를 탐색합니다. 회로 깊이가 증가함에 따라 QAOA는 무작위 추측과 최적해 사이를 보간(interpolate)하며, 수렴의 이론적 보장을 제공합니다.

> **비유:** QAOA는 조각과 같습니다. 각 레이어가 교대로 양자 상태를 깎아내고(비용 유니터리) 다듬어(혼합 유니터리), 돌 속에 숨어 있는 최적해를 점차 드러냅니다. 비용 유니터리는 점수가 좋은 해를 향해 깎아내는 반면, 혼합 유니터리는 상태가 나쁜 구성에 갇히는 것을 방지합니다. 충분한 레이어를 거치면 조각이 최적의 형태로 수렴합니다.

## 목차

1. [조합 최적화 문제](#1-조합-최적화-문제)
2. [최대 컷(MaxCut) 문제](#2-최대-컷maxcut-문제)
3. [QAOA 프레임워크](#3-qaoa-프레임워크)
4. [QAOA 회로 구성](#4-qaoa-회로-구성)
5. [비용 및 혼합 해밀토니안](#5-비용-및-혼합-해밀토니안)
6. [매개변수 최적화](#6-매개변수-최적화)
7. [단열 양자 컴퓨팅과의 연관성](#7-단열-양자-컴퓨팅과의-연관성)
8. [성능 분석](#8-성능-분석)
9. [파이썬 구현](#9-파이썬-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 조합 최적화 문제

### 1.1 일반 프레임워크

조합 최적화 문제는 다음을 묻습니다: 유한한 후보해 집합 $\{z\}$와 목적 함수 $C(z)$가 주어졌을 때, $C$를 최대화(또는 최소화)하는 해 $z^*$를 찾아라:

$$z^* = \arg\max_{z \in \{0,1\}^n} C(z)$$

여기서 $z = (z_1, z_2, \ldots, z_n)$은 길이 $n$의 이진 문자열입니다.

### 1.2 NP-난해 문제

많은 조합 최적화 문제는 NP-난해(NP-hard)합니다:

| 문제 | 설명 | 응용 |
|---------|-------------|-------------|
| 최대 컷(MaxCut) | 컷 간선을 최대화하도록 그래프 정점을 분할 | 회로 배치, 소셜 네트워크 분석 |
| 순회 판매원 문제(Traveling Salesman) | 모든 도시를 방문하는 최단 경로 | 물류, 라우팅 |
| 그래프 색칠(Graph Coloring) | 최소 색으로 인접 정점이 같은 색이 되지 않도록 색칠 | 스케줄링, 레지스터 할당 |
| Max-SAT | 부울 공식에서 만족되는 절(clause)을 최대화 | 검증, AI 계획 |
| 포트폴리오 최적화(Portfolio Optimization) | 위험 제약 하에서 수익 최대화 | 금융 |

### 1.3 이징(Ising) 공식화

많은 조합 문제는 **이징 해밀토니안(Ising Hamiltonian)**의 바닥 상태를 찾는 것으로 인코딩할 수 있습니다:

$$H_C = \sum_{(i,j) \in E} J_{ij} Z_i Z_j + \sum_i h_i Z_i$$

여기서 $Z_i \in \{+1, -1\}$은 이징 스핀 변수입니다. 이진 변수로의 매핑은 $z_i = (1 - Z_i)/2$이므로, $Z_i = +1 \leftrightarrow z_i = 0$이고 $Z_i = -1 \leftrightarrow z_i = 1$입니다.

이 이징 공식화는 고전적 최적화와 양자 알고리즘을 잇는 다리입니다: 이징 해밀토니안은 QAOA에서 비용 해밀토니안으로 직접 사용됩니다.

---

## 2. 최대 컷(MaxCut) 문제

### 2.1 정의

$n$개의 정점과 $m$개의 간선을 가진 비방향 그래프(undirected graph) $G = (V, E)$가 주어졌을 때:

- 정점을 두 개의 서로소 집합(disjoint set) $S$와 $\bar{S}$로 **분할**
- **컷(Cut)**: 간선 $(i, j)$는 $i \in S$이고 $j \in \bar{S}$인 경우(또는 그 반대) "컷됨"
- **목적**: 컷된 간선의 수를 최대화

공식적으로, 정점 $i$가 어느 집합에 속하는지를 나타내는 $z_i \in \{0, 1\}$로:

$$C(z) = \sum_{(i,j) \in E} z_i (1 - z_j) + (1 - z_i) z_j = \sum_{(i,j) \in E} z_i \oplus z_j$$

### 2.2 이징 공식화

$Z_i = 1 - 2z_i$를 사용하면:

$$C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2} = \frac{|E|}{2} - \frac{1}{2}\sum_{(i,j) \in E} Z_i Z_j$$

$C$를 최대화하는 것은 $H_C = \sum_{(i,j) \in E} Z_i Z_j$를 최소화하는 것과 동치입니다.

양자 연산자로 (고전 스핀을 파울리-Z로 대체):

$$\hat{H}_C = \sum_{(i,j) \in E} \hat{Z}_i \hat{Z}_j$$

$\hat{H}_C$의 바닥 상태가 MaxCut 해를 인코딩합니다.

### 2.3 예시: 삼각형 그래프

삼각형(3개의 정점, 3개의 간선: (0,1), (1,2), (0,2))에 대해:

| 배정 $z$ | 컷 값 $C(z)$ |
|-----------------|-------------------|
| 000 | 0 |
| 001 | 2 |
| 010 | 2 |
| 011 | 2 |
| 100 | 2 |
| 101 | 2 |
| 110 | 2 |
| 111 | 0 |

최대 컷 = 2이며, $S$에 정확히 1개 또는 2개의 정점이 있는 모든 배정으로 달성됩니다. 삼각형의 경우 세 간선을 모두 컷할 수 없습니다 (이는 이분 그래프(bipartite graph)를 요구합니다).

### 2.4 복잡도

MaxCut은 일반 그래프에서 NP-난해입니다. 최선의 고전적 근사 알고리즘(Goemans-Williamson, 1995)은 $\alpha_{GW} \approx 0.878$의 비율을 달성합니다 (즉, 해가 최적의 적어도 87.8%입니다). QAOA가 이 비율을 능가할 수 있는지는 열린 문제입니다.

---

## 3. QAOA 프레임워크

### 3.1 개요

QAOA는 정수 $p \geq 1$ (레이어 수 또는 "깊이")로 매개변수화됩니다. $p$-레이어 QAOA 회로는 교대하는 비용 및 혼합 유니터리를 적용합니다:

$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = U_M(\beta_p) U_C(\gamma_p) \cdots U_M(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}$$

여기서:
- $\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_p)$와 $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)$는 $2p$개의 실수 매개변수
- $U_C(\gamma) = e^{-i\gamma \hat{H}_C}$는 **비용 유니터리(cost unitary)**
- $U_M(\beta) = e^{-i\beta \hat{H}_M}$은 **혼합 유니터리(mixer unitary)**
- $|+\rangle^{\otimes n}$은 균일 중첩(initial state)

### 3.2 최적화 루프

1. $p$ 선택 (회로 깊이)
2. 매개변수 $\boldsymbol{\gamma}_0, \boldsymbol{\beta}_0$ 초기화
3. **양자**: $|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$ 준비, $\langle \hat{H}_C \rangle$ 측정
4. **고전**: $\langle \hat{H}_C \rangle$를 최소화하도록 매개변수 업데이트
5. 수렴할 때까지 반복
6. 최종 측정: $|\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*\rangle$에서 샘플링하여 후보해 얻기

### 3.3 VQE와 QAOA의 차이점

| 특징 | VQE | QAOA |
|---------|-----|------|
| 앤사츠 | 문제에 무관하거나 화학적으로 영감받음 | 문제 구조 기반 (비용 + 혼합) |
| 매개변수 | 임의의 회전 각도 | 교대하는 $\gamma_i, \beta_i$ |
| 구조 | 일반 레이어 | 교대하는 비용/혼합 유니터리 |
| 수렴 | 변분적 (상한값) | $p \to \infty$이면 정확한 해 |
| 목표 | 바닥 상태 에너지 | 최적화 해 (비트열) |

---

## 4. QAOA 회로 구성

### 4.1 초기 상태

혼합 해밀토니안 $\hat{H}_M = \sum_i X_i$의 바닥 상태인 균일 중첩에서 시작합니다:

$$|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} |z\rangle$$

### 4.2 비용 유니터리

MaxCut에서 $\hat{H}_C = \sum_{(i,j) \in E} Z_i Z_j$이므로:

$$U_C(\gamma) = e^{-i\gamma \hat{H}_C} = \prod_{(i,j) \in E} e^{-i\gamma Z_i Z_j}$$

각 인수(factor) $e^{-i\gamma Z_i Z_j}$는 다음과 같이 구현될 수 있습니다:

```
q_i ─── ●──── Rz(2γ) ──── ●────
        │                   │
q_j ─── ⊕─────────────── ⊕────
```

즉: CNOT(i→j), 그 다음 큐비트 $j$에 $R_z(2\gamma)$, 그 다음 다시 CNOT(i→j). 이는 $z_i = z_j$일 때 위상 $e^{-i\gamma}$를, $z_i \neq z_j$일 때 $e^{+i\gamma}$를 적용합니다.

또는, $ZZ$ 상호작용을 다음과 같이 분해할 수 있습니다:

$$e^{-i\gamma Z_i Z_j} = \text{CNOT}_{ij} \cdot (I \otimes R_z(2\gamma)) \cdot \text{CNOT}_{ij}$$

### 4.3 혼합 유니터리

표준 혼합기는 횡자기장(transverse field)입니다:

$$\hat{H}_M = \sum_{i=1}^{n} X_i$$

$$U_M(\beta) = e^{-i\beta \hat{H}_M} = \prod_{i=1}^{n} e^{-i\beta X_i} = \prod_{i=1}^{n} R_x(2\beta)$$

각 인수는 단일 큐비트에 대한 $R_x(2\beta)$ 회전으로, 얽힘(entangling) 게이트가 필요 없습니다.

### 4.4 완전한 회로 (p=1)

```
|0⟩ ── H ── ●── Rz ──●── ●── Rz ──│── Rx(2β) ── M
             │        │   │        │
|0⟩ ── H ── ⊕────────⊕── │── Rz ──│── Rx(2β) ── M
                           │        │
|0⟩ ── H ──────────────── ⊕────────⊕── Rx(2β) ── M

     H⊗n        U_C(γ)                U_M(β)
```

### 4.5 게이트 수

$n$개의 정점과 $m$개의 간선을 가진 그래프에 대해:
- **초기 레이어**: $n$개의 아다마르(Hadamard) 게이트
- **QAOA 레이어당**: $2m$개의 CNOT + $m$개의 $R_z$ + $n$개의 $R_x$ 게이트
- **$p$ 레이어 합계**: $n + p(2m + m + n) = n + p(3m + n)$ 게이트

---

## 5. 비용 및 혼합 해밀토니안

### 5.1 비용 해밀토니안의 속성

비용 해밀토니안 $\hat{H}_C$는 계산 기저(computational basis)에서 대각(diagonal)입니다:

$$\hat{H}_C|z\rangle = C(z)|z\rangle$$

여기서 $C(z)$는 고전적 목적 함수 값입니다. 이는 $U_C(\gamma)$가 목적 값에 비례하는 위상을 각 계산 기저 상태에 적용함을 의미합니다:

$$U_C(\gamma)|z\rangle = e^{-i\gamma C(z)}|z\rangle$$

$C(z)$가 낮은 상태(최소화에서 더 나은 해)는 $C(z)$가 높은 상태와 다른 위상을 얻습니다. 이후 연산에서의 QFT 유사 간섭(interference)이 좋은 해를 증폭합니다.

### 5.2 혼합 해밀토니안의 속성

혼합 해밀토니안 $\hat{H}_M = \sum_i X_i$는 계산 기저 상태 간의 전이(transition)를 생성합니다. 각 $X_i$는 큐비트 $i$를 뒤집습니다:

$$e^{-i\beta X_i}|z_1 \cdots z_i \cdots z_n\rangle = \cos\beta |z_1 \cdots z_i \cdots z_n\rangle - i\sin\beta |z_1 \cdots \bar{z}_i \cdots z_n\rangle$$

혼합기는 $\sin\beta$ 진폭으로 개별 비트를 뒤집어 인접 해를 "탐색"합니다.

### 5.3 상호작용: 탐색 대 활용(Exploration vs Exploitation)

- **비용 유니터리** ($U_C$): 활용(Exploitation) — 좋은 목적 값을 가진 상태를 증폭
- **혼합 유니터리** ($U_M$): 탐색(Exploration) — 인접 상태로의 전이 허용

두 가지를 교대로 적용함으로써, QAOA는 모의 담금질(simulated annealing)이나 진화 알고리즘과 유사하게 탐색과 활용의 균형을 맞춥니다.

### 5.4 맞춤형 혼합기(Custom Mixers)

제약 최적화 문제에서 표준 $X$ 혼합기는 실현 불가능한(infeasible) 해를 생성할 수 있습니다. 맞춤형 혼합기는 제약 조건을 강제할 수 있습니다:

- **그로버 혼합기(Grover mixer)**: 실현 가능한 해에 대한 균일 중첩을 보존
- **XY 혼합기(XY mixer)**: 1-비트의 수(해밍 가중치(Hamming weight) 제약)를 보존
- **패리티 혼합기(Parity mixer)**: 패리티를 보존

---

## 6. 매개변수 최적화

### 6.1 최적화 지형

QAOA 에너지 $E(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle\boldsymbol{\gamma}, \boldsymbol{\beta}|\hat{H}_C|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$는 $2p$개의 실수 매개변수의 부드러운 함수입니다. 최적화 지형은 다음을 가질 수 있습니다:

- 다수의 지역 최솟값(local minima)
- 안장점(saddle points)
- 대칭성 (매개변수의 주기성)

### 6.2 매개변수 주기성

QAOA의 구조로 인해 매개변수는 주기성을 가집니다:

$$\gamma_i \in [0, 2\pi), \quad \beta_i \in [0, \pi)$$

비가중(unweighted) 그래프의 MaxCut에 대해, 탐색 공간을 더욱 줄이는 추가적인 대칭성이 있습니다.

### 6.3 최적화 전략

**격자 탐색(Grid search)** (작은 $p$에 대해): $(\gamma, \beta)$ 값의 격자에서 $E$를 평가합니다. $p = 1$ 또는 $p = 2$에서만 실현 가능합니다.

**기울기 기반(Gradient-based)**: 매개변수 이동 규칙(parameter shift rule)을 사용하여(VQE와 같이) 기울기를 계산하고 Adam이나 L-BFGS-B와 같은 방법을 적용합니다.

**보간법(Interpolation, INTERP)**: $p$-레이어 QAOA를 최적화한 다음, 최적 매개변수를 $(p+1)$-레이어 QAOA의 시작점으로 사용합니다. 구체적으로, $p$개의 매개변수를 $p+1$개의 매개변수로 보간합니다.

**푸리에 휴리스틱(Fourier heuristic)**: $\gamma_i$와 $\beta_i$를 $i$에 대한 푸리에 급수로 매개변수화하여 자유 매개변수의 수를 줄입니다.

### 6.4 매개변수 이전(Parameter Transfer)

실용적인 전략: 같은 그래프 패밀리의 작은 인스턴스에 대해 QAOA를 최적화한 다음, 해당 매개변수(적절히 스케일링)를 더 큰 인스턴스의 시작점으로 사용합니다. 이 "매개변수 이전"은 종종 놀랍도록 잘 작동합니다.

---

## 7. 단열 양자 컴퓨팅과의 연관성

### 7.1 단열 양자 컴퓨팅(Adiabatic Quantum Computing, AQC)

AQC는 최적화 문제를 해밀토니안 $\hat{H}_C$에 인코딩하고, 간단한 초기 해밀토니안 $\hat{H}_M$의 바닥 상태에서 $\hat{H}_C$의 바닥 상태로 천천히 진화합니다:

$$\hat{H}(t) = \left(1 - \frac{t}{T}\right)\hat{H}_M + \frac{t}{T}\hat{H}_C, \quad t \in [0, T]$$

**단열 정리(adiabatic theorem)**는 $T$가 충분히 크다면(최소 에너지 간격의 제곱에 반비례), 시스템이 진화 내내 바닥 상태에 머문다는 것을 보장합니다.

### 7.2 트로터화된 단열 진화로서의 QAOA

QAOA는 단열 진화의 트로터(Trotter) 근사로 볼 수 있습니다. $p$개의 레이어로, $\hat{H}_M$에서 $\hat{H}_C$로의 진화는 $p$단계로 이산화됩니다:

$$U_{\text{QAOA}} = \prod_{k=1}^{p} e^{-i\beta_k \hat{H}_M} e^{-i\gamma_k \hat{H}_C}$$

$p \to \infty$인 단열 극한에서, $\beta_k \to 0$이고 $\gamma_k \to 0$ (적절한 스케일링으로), 이는 정확한 단열 진화로 수렴합니다.

### 7.3 QAOA 대 AQC: 주요 차이점

| 측면 | QAOA | AQC |
|--------|------|-----|
| 매개변수 | 자유 ($2p$개 최적화) | 스케줄 $s(t)$로 고정 |
| 하드웨어 | 게이트 기반 | 어닐링(annealing) 기반 |
| 깊이 | $p$ 레이어 (유한) | 연속 (긴 시간 $T$) |
| 최적화 | 고전적 외부 루프 | 없음 (물리가 작동) |
| 유연성 | 작은 $p$에서 단열을 능가할 수 있음 | 큰 $T$에 대해 보장됨 |

### 7.4 수렴 보장

**정리** (Farhi 등): 어떤 조합 최적화 문제에 대해서도, $p \to \infty$이고 최적화된 매개변수를 가진 QAOA는 $\hat{H}_C$의 정확한 바닥 상태로 수렴합니다.

이는 QAOA가 원칙적으로 보편적인 최적화 알고리즘임을 의미합니다. 실용적인 질문은: 좋은 해를 찾으려면 $p$가 얼마나 커야 하는가입니다.

---

## 8. 성능 분석

### 8.1 MaxCut에 대한 QAOA $p=1$

3-정칙 그래프(모든 정점의 차수가 3)에서 $p = 1$에 대해, Farhi 등이 증명했습니다:

$$\frac{\langle C \rangle_{\text{QAOA}}}{C_{\max}} \geq 0.6924$$

이는 $p = 1$에서 QAOA가 최적 컷 값의 적어도 69.24%를 보장함을 의미합니다. 비교하자면, 무작위 배정은 50%를 줍니다.

### 8.2 $p$에 따른 확장성

$p$가 증가함에 따라:
- 근사 비율이 향상됨
- 매개변수 수($2p$)가 증가하여 최적화가 더 어려워짐
- 회로 깊이가 증가하여 잡음에 더 취약해짐

경험적으로, 많은 그래프 인스턴스에서 $p = 3\text{-}5$로 최적에 가까운 해를 얻습니다.

### 8.3 고전 알고리즘과의 비교

| 알고리즘 | 근사 비율 | 유형 |
|-----------|-------------------|------|
| 무작위 배정 | 0.500 | 간단 |
| QAOA $p=1$ (3-정칙) | 0.6924 | 양자 |
| QAOA $p=2$ (3-정칙) | ~0.756 | 양자 |
| 탐욕 알고리즘(Greedy) | ~0.5 (최악의 경우) | 고전 |
| Goemans-Williamson SDP | 0.878 | 고전 |
| QAOA $p \to \infty$ | 1.000 | 양자 |

유한-$p$ QAOA가 Goemans-Williamson을 능가할 수 있는지는 여전히 열린 문제입니다.

---

## 9. 파이썬 구현

### 9.1 MaxCut QAOA 핵심 코드

```python
import numpy as np
from scipy.optimize import minimize
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def build_cost_hamiltonian(n_qubits, edges):
    """Build the MaxCut cost Hamiltonian as a matrix.

    Why matrix form? For small instances (n ≤ 12), the full matrix allows
    exact simulation and verification. For larger instances, one would use
    state-vector simulation or actual quantum hardware.
    """
    N = 2**n_qubits
    H_C = np.zeros((N, N), dtype=complex)

    for (i, j) in edges:
        # Z_i Z_j term
        op = [I] * n_qubits
        op[i] = Z
        op[j] = Z
        term = op[0]
        for k in range(1, n_qubits):
            term = np.kron(term, op[k])
        H_C += term

    return H_C

def build_mixer_hamiltonian(n_qubits):
    """Build the mixer Hamiltonian H_M = Σ X_i."""
    N = 2**n_qubits
    H_M = np.zeros((N, N), dtype=complex)

    for i in range(n_qubits):
        op = [I] * n_qubits
        op[i] = X
        term = op[0]
        for k in range(1, n_qubits):
            term = np.kron(term, op[k])
        H_M += term

    return H_M

def maxcut_value(bitstring, edges):
    """Compute the MaxCut value for a given bitstring.

    A cut edge is one where the two endpoints have different bit values.
    """
    return sum(1 for (i, j) in edges if bitstring[i] != bitstring[j])

def brute_force_maxcut(n_qubits, edges):
    """Find the optimal MaxCut by exhaustive enumeration.

    Why brute force? For small instances, this gives the exact answer to
    compare against QAOA. For n > 20, brute force becomes infeasible,
    which is precisely where quantum algorithms might help.
    """
    best_cut = 0
    best_assignment = None

    for bits in product([0, 1], repeat=n_qubits):
        cut = maxcut_value(bits, edges)
        if cut > best_cut:
            best_cut = cut
            best_assignment = bits

    return best_cut, best_assignment

def qaoa_circuit(params, n_qubits, H_C, H_M, p):
    """Simulate the QAOA circuit and return the final state.

    The circuit alternates p layers of:
    1. Cost unitary: e^{-iγ H_C} (phases based on objective value)
    2. Mixer unitary: e^{-iβ H_M} (transitions between solutions)
    """
    gammas = params[:p]
    betas = params[p:]
    N = 2**n_qubits

    # Initial state: uniform superposition |+⟩^n
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    for layer in range(p):
        # Cost unitary: e^{-iγ H_C}
        # Since H_C is diagonal in computational basis, this is efficient
        # But for full matrix simulation, we use matrix exponential
        U_C = np.diag(np.exp(-1j * gammas[layer] * np.diag(H_C).real))
        state = U_C @ state

        # Mixer unitary: e^{-iβ H_M}
        # H_M = Σ X_i, and X_i commute when acting on different qubits
        # So e^{-iβ Σ X_i} = ⊗ e^{-iβ X_i} = ⊗ Rx(2β)
        Rx = np.array([[np.cos(betas[layer]), -1j*np.sin(betas[layer])],
                       [-1j*np.sin(betas[layer]), np.cos(betas[layer])]])
        U_M = Rx
        for _ in range(n_qubits - 1):
            U_M = np.kron(U_M, Rx)
        state = U_M @ state

    return state

def qaoa_expectation(params, n_qubits, edges, H_C, H_M, p):
    """Compute the QAOA expected cost value.

    This is the objective function for the classical optimizer:
    minimize ⟨γ,β|H_C|γ,β⟩ (for MaxCut, we want the minimum of ZZ terms,
    which corresponds to the maximum cut).
    """
    state = qaoa_circuit(params, n_qubits, H_C, H_M, p)
    energy = np.real(state.conj() @ H_C @ state)
    return energy

def run_qaoa_maxcut(n_qubits, edges, p=1, n_restarts=10):
    """Run QAOA for MaxCut with multiple random restarts.

    Why multiple restarts? The QAOA landscape can have multiple local minima.
    Running from different starting points increases the chance of finding
    the global optimum.
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    # Brute force for comparison
    optimal_cut, optimal_assignment = brute_force_maxcut(n_qubits, edges)

    print(f"Graph: {n_qubits} vertices, {len(edges)} edges")
    print(f"Optimal MaxCut: {optimal_cut} (assignment: {optimal_assignment})")
    print(f"QAOA depth: p = {p}")

    best_energy = float('inf')
    best_params = None

    for restart in range(n_restarts):
        # Random initial parameters
        gamma0 = np.random.uniform(0, 2*np.pi, p)
        beta0 = np.random.uniform(0, np.pi, p)
        params0 = np.concatenate([gamma0, beta0])

        result = minimize(qaoa_expectation, params0,
                         args=(n_qubits, edges, H_C, H_M, p),
                         method='COBYLA',
                         options={'maxiter': 500})

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

    # Analyze the best solution
    state = qaoa_circuit(best_params, n_qubits, H_C, H_M, p)
    probs = np.abs(state)**2

    # Convert energy to cut value
    # H_C = Σ Z_i Z_j, and cut value C = (|E| - ⟨H_C⟩) / 2
    expected_cut = (len(edges) - best_energy) / 2

    print(f"\nQAOA results:")
    print(f"  Expected cut value: {expected_cut:.4f}")
    print(f"  Approximation ratio: {expected_cut/optimal_cut:.4f}")
    print(f"  Optimal γ: {best_params[:p].round(4)}")
    print(f"  Optimal β: {best_params[p:].round(4)}")

    # Top solutions by probability
    print(f"\n  Top 5 measurement outcomes:")
    top_indices = np.argsort(probs)[::-1][:5]
    for idx in top_indices:
        bits = tuple(int(b) for b in f"{idx:0{n_qubits}b}")
        cut = maxcut_value(bits, edges)
        print(f"    |{''.join(map(str, bits))}⟩: prob={probs[idx]:.4f}, cut={cut}")

    return expected_cut, optimal_cut, best_params

# === Example 1: Triangle graph ===
print("=" * 55)
print("QAOA for MaxCut: Triangle Graph")
print("=" * 55)
run_qaoa_maxcut(3, [(0,1), (1,2), (0,2)], p=1)

# === Example 2: 4-vertex graph ===
print("\n" + "=" * 55)
print("QAOA for MaxCut: 4-Vertex Graph")
print("=" * 55)
edges_4 = [(0,1), (1,2), (2,3), (0,3), (0,2)]
run_qaoa_maxcut(4, edges_4, p=2)
```

### 9.2 매개변수 지형 시각화

```python
import numpy as np

def qaoa_landscape(n_qubits, edges, gamma_range, beta_range, resolution=50):
    """Compute the QAOA energy landscape for p=1.

    Why visualize? The landscape reveals the optimization difficulty and
    the structure of the parameter space. Smooth landscapes with few
    local minima are easy to optimize; rugged landscapes with many
    local minima require more sophisticated strategies.
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    gammas = np.linspace(gamma_range[0], gamma_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)

    landscape = np.zeros((resolution, resolution))

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            landscape[i, j] = qaoa_expectation(
                [gamma, beta], n_qubits, edges, H_C, H_M, p=1)

    # Convert to cut values
    cut_landscape = (len(edges) - landscape) / 2

    # Find the optimal point
    max_idx = np.unravel_index(np.argmax(cut_landscape), cut_landscape.shape)
    opt_gamma = gammas[max_idx[0]]
    opt_beta = betas[max_idx[1]]
    opt_cut = cut_landscape[max_idx]

    print(f"Landscape for {n_qubits}-vertex graph, {len(edges)} edges")
    print(f"Optimal parameters: γ={opt_gamma:.4f}, β={opt_beta:.4f}")
    print(f"Maximum expected cut: {opt_cut:.4f}")
    print(f"Brute force optimal: {brute_force_maxcut(n_qubits, edges)[0]}")

    # Print a text-based visualization
    print(f"\nLandscape (cut value, rows=γ, cols=β):")
    print(f"  β: {beta_range[0]:.2f} {'→':>20} {beta_range[1]:.2f}")
    step = max(1, resolution // 10)
    for i in range(0, resolution, step):
        row = ""
        for j in range(0, resolution, step):
            val = cut_landscape[i, j]
            if val > 0.9 * opt_cut:
                row += "█"
            elif val > 0.7 * opt_cut:
                row += "▓"
            elif val > 0.5 * opt_cut:
                row += "▒"
            elif val > 0.3 * opt_cut:
                row += "░"
            else:
                row += " "
        print(f"  γ={gammas[i]:5.2f} |{row}|")

    return cut_landscape, gammas, betas

# Visualize for a 4-vertex graph
print("=" * 55)
print("QAOA Parameter Landscape (p=1)")
print("=" * 55)
edges_4 = [(0,1), (1,2), (2,3), (0,3)]
qaoa_landscape(4, edges_4, (0, 2*np.pi), (0, np.pi))
```

### 9.3 깊이 확장성 실험

```python
import numpy as np
from scipy.optimize import minimize

def qaoa_depth_experiment(n_qubits, edges, max_p=5, n_restarts=5):
    """Study how QAOA performance scales with circuit depth p.

    Key insight: increasing p should improve the approximation ratio,
    but at the cost of more parameters to optimize and deeper circuits
    (more susceptible to noise on real hardware).
    """
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)
    optimal_cut, _ = brute_force_maxcut(n_qubits, edges)

    print(f"Graph: {n_qubits} vertices, {len(edges)} edges, MaxCut = {optimal_cut}")
    print(f"\n{'p':>4} {'Expected Cut':>14} {'Approx Ratio':>14} {'Best Sampled':>14}")
    print("-" * 50)

    for p in range(1, max_p + 1):
        best_energy = float('inf')
        best_params = None

        for _ in range(n_restarts):
            gamma0 = np.random.uniform(0, np.pi, p)
            beta0 = np.random.uniform(0, np.pi/2, p)
            params0 = np.concatenate([gamma0, beta0])

            result = minimize(qaoa_expectation, params0,
                            args=(n_qubits, edges, H_C, H_M, p),
                            method='COBYLA',
                            options={'maxiter': 1000})

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x

        expected_cut = (len(edges) - best_energy) / 2
        ratio = expected_cut / optimal_cut

        # Sample from the optimal state
        state = qaoa_circuit(best_params, n_qubits, H_C, H_M, p)
        probs = np.abs(state)**2
        best_sampled = max(maxcut_value(
            tuple(int(b) for b in f"{idx:0{n_qubits}b}"), edges)
            for idx in np.argsort(probs)[::-1][:3])

        print(f"{p:4d} {expected_cut:14.4f} {ratio:14.4f} {best_sampled:14d}")

    return

# Experiment on a 5-vertex graph
print("=" * 55)
print("QAOA Depth Scaling Experiment")
print("=" * 55)
edges_5 = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)]
qaoa_depth_experiment(5, edges_5, max_p=5)
```

### 9.4 비교: QAOA 대 무작위 대 탐욕

```python
import numpy as np

def compare_algorithms(n_qubits, edges, p_qaoa=2, n_trials=1000):
    """Compare QAOA against classical heuristics for MaxCut.

    This puts QAOA in context: how does it compare to simple classical
    baselines? Understanding this is crucial for assessing whether quantum
    algorithms provide a real advantage.
    """
    optimal_cut, _ = brute_force_maxcut(n_qubits, edges)

    # 1. Random assignment (baseline)
    random_cuts = []
    for _ in range(n_trials):
        bits = tuple(np.random.randint(0, 2, n_qubits))
        random_cuts.append(maxcut_value(bits, edges))
    avg_random = np.mean(random_cuts)

    # 2. Greedy algorithm
    def greedy_maxcut():
        assignment = [0] * n_qubits
        for v in range(n_qubits):
            # Try v=0 and v=1, pick the one that maximizes cut so far
            cut_0, cut_1 = 0, 0
            for (i, j) in edges:
                if i == v or j == v:
                    other = j if i == v else i
                    if other < v:  # Only count edges to already-assigned vertices
                        cut_0 += (0 != assignment[other])
                        cut_1 += (1 != assignment[other])
            assignment[v] = 1 if cut_1 > cut_0 else 0
        return maxcut_value(tuple(assignment), edges)

    greedy_cut = greedy_maxcut()

    # 3. QAOA
    H_C = build_cost_hamiltonian(n_qubits, edges)
    H_M = build_mixer_hamiltonian(n_qubits)

    best_energy = float('inf')
    for _ in range(10):
        params0 = np.random.uniform(0, np.pi, 2 * p_qaoa)
        result = minimize(qaoa_expectation, params0,
                        args=(n_qubits, edges, H_C, H_M, p_qaoa),
                        method='COBYLA', options={'maxiter': 500})
        if result.fun < best_energy:
            best_energy = result.fun
    qaoa_cut = (len(edges) - best_energy) / 2

    print(f"\nMaxCut Comparison ({n_qubits} vertices, {len(edges)} edges)")
    print(f"{'Algorithm':>20} {'Cut Value':>12} {'Ratio':>8}")
    print("-" * 44)
    print(f"{'Random (avg)':>20} {avg_random:12.2f} {avg_random/optimal_cut:8.4f}")
    print(f"{'Greedy':>20} {greedy_cut:12d} {greedy_cut/optimal_cut:8.4f}")
    print(f"{'QAOA (p={p_qaoa})':>20} {qaoa_cut:12.4f} {qaoa_cut/optimal_cut:8.4f}")
    print(f"{'Optimal':>20} {optimal_cut:12d} {1.0:8.4f}")

# Compare on different graphs
print("=" * 55)
print("Algorithm Comparison for MaxCut")
print("=" * 55)

# Pentagon with diagonals
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)]
compare_algorithms(5, edges, p_qaoa=2)

# 6-vertex random graph
edges_6 = [(0,1), (0,2), (0,5), (1,2), (1,3), (2,4), (3,4), (3,5), (4,5)]
compare_algorithms(6, edges_6, p_qaoa=2)
```

---

## 10. 연습 문제

### 연습 문제 1: 정사각형 그래프에 대한 QAOA

간선 $\{(0,1), (1,2), (2,3), (3,0)\}$을 가진 4-정점 정사각형 그래프에 QAOA를 적용합니다:
(a) 최적 MaxCut은 무엇인가? 모든 최적 배정을 나열합니다.
(b) $p = 1$로 QAOA를 실행합니다. 어떤 근사 비율을 달성하는가?
(c) 에너지 지형 $E(\gamma, \beta)$를 그리고 모든 지역 최솟값을 식별합니다.
(d) $p = 2, 3$으로 QAOA를 실행합니다. 근사 비율이 어떻게 향상되는가?

### 연습 문제 2: 가중 MaxCut

각 간선 $(i,j)$에 가중치 $w_{ij}$가 있는 가중 그래프를 처리하도록 QAOA 구현을 확장합니다:

$$C(z) = \sum_{(i,j) \in E} w_{ij} (z_i \oplus z_j)$$

(a) 가중치를 포함하도록 비용 해밀토니안을 수정합니다.
(b) 간선 $\{(0,1,3), (1,2,1), (2,3,2), (0,3,4)\}$를 가진 4-정점 그래프에서 테스트합니다 (형식: (i, j, weight)).
(c) QAOA 결과와 전수조사(brute force)를 비교합니다.

### 연습 문제 3: 최대 독립 집합(Max Independent Set)에 대한 QAOA

그래프 $G = (V, E)$의 독립 집합(independent set) $S$는 그 사이에 간선이 없는 정점의 부분 집합입니다. 최대 독립 집합 문제는 그러한 집합 중 가장 큰 것을 찾습니다.

(a) 인접하여 선택된 정점에 대한 패널티 항과 함께 최대 독립 집합을 이징 해밀토니안으로 공식화합니다.
(b) 이 문제에 대한 QAOA를 구현합니다.
(c) 페터슨 그래프(Petersen graph, 10개의 정점)에서 테스트합니다. 알려진 최대 독립 집합 크기 4와 비교합니다.

### 연습 문제 4: QAOA에 대한 잡음 영향

QAOA에 대한 탈분극 잡음(depolarizing noise)의 영향을 시뮬레이션합니다:
(a) QAOA 회로의 각 게이트 후에 확률 $p_{\text{noise}}$로 탈분극 잡음을 적용합니다.
(b) $p_{\text{noise}} = 0, 0.001, 0.01, 0.05$로 삼각형 그래프에서 MaxCut QAOA를 실행합니다.
(c) 잡음 수준 대 근사 비율을 그립니다.
(d) 각 잡음 수준에서 최적 QAOA 깊이 $p$는 무엇인가? (더 깊은 회로는 더 많은 잡음을 누적합니다.)

### 연습 문제 5: 매개변수 집중(Parameter Concentration)

(a) 각각 8개의 정점을 가진 20개의 무작위 3-정칙 그래프를 생성합니다.
(b) 각 그래프에서 $p=1$ QAOA를 실행하고 최적 $(\gamma^*, \beta^*)$를 기록합니다.
(c) 서로 다른 그래프에서 최적 매개변수가 얼마나 집중되어 있는가? (분포를 그립니다.)
(d) 한 그래프 집합에서 구한 평균 최적 매개변수를 새로운 그래프 집합에 사용합니다. 인스턴스별 최적화와 비교하여 얼마나 성능이 저하되는가?

---

[← 이전: 변분 양자 고유값 풀이](13_VQE.md) | [다음: 양자 기계 학습 →](15_Quantum_Machine_Learning.md)

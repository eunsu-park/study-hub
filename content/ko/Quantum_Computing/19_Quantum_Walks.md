# 레슨 19: 양자 걷기

[← 이전: 양자 시뮬레이션](18_Quantum_Simulation.md) | [다음: 잡음과 양자 채널 →](20_Noise_and_Quantum_Channels.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 이산 시간 및 연속 시간 양자 걷기를 정의하고 고전 랜덤 워크와 대비할 수 있다
2. 시프트 및 코인 연산자를 사용하여 직선과 그래프에서 코인 양자 걷기를 구성할 수 있다
3. 공간 탐색 문제에 대한 양자 걷기의 이차 속도향상을 설명할 수 있다
4. 그래프 동형 및 원소 구별성에 양자 걷기 프레임워크를 적용할 수 있다
5. 양자 걷기의 확산 거동과 탄도 수송을 분석할 수 있다
6. Python으로 이산 및 연속 양자 걷기를 구현할 수 있다

---

양자 걷기(Quantum Walk)는 고전 랜덤 워크의 양자역학적 일반화입니다. 고전 랜덤 워커가 동일한 확률로 왼쪽 또는 오른쪽으로 이동하는 반면, 양자 워커는 모든 방향으로 동시에 이동하는 중첩 상태에 존재합니다. 이는 근본적으로 다른 거동을 초래합니다: 고전 랜덤 워크가 $\sigma \sim \sqrt{t}$(확산적)로 퍼지는 반면, 양자 걷기는 $\sigma \sim t$(탄도적)로 퍼집니다 — 확산에서 이차 속도향상입니다.

이 속도향상은 단순한 호기심이 아닙니다. 양자 걷기는 Grover 탐색(양자 걷기로 재구성 가능), 원소 구별성 알고리즘, 그래프 동형 접근법 등 여러 중요한 양자 알고리즘의 기반을 이룹니다.

> **비유:** 갈림길에서 동전을 던져 왼쪽 또는 오른쪽을 선택하는 고전 랜덤 워커를 상상하세요. 양자 워커는 대신 파동으로 두 경로를 동시에 이동합니다. 경로가 다시 만나면 파동이 간섭합니다 — 일부 위치에서는 보강 간섭(높은 확률), 다른 위치에서는 상쇄 간섭(낮은 확률)이 발생합니다.

## 목차

1. [고전 랜덤 워크 복습](#1-고전-랜덤-워크-복습)
2. [이산 시간 양자 걷기](#2-이산-시간-양자-걷기)
3. [연속 시간 양자 걷기](#3-연속-시간-양자-걷기)
4. [코인 양자 걷기](#4-코인-양자-걷기)
5. [양자 걷기 탐색](#5-양자-걷기-탐색)
6. [그래프 문제 응용](#6-그래프-문제-응용)
7. [양자 걷기로부터의 양자 속도향상](#7-양자-걷기로부터의-양자-속도향상)
8. [양자 걷기의 보편성](#8-양자-걷기의-보편성)
9. [Python 구현](#9-python-구현)
10. [연습 문제](#10-연습-문제)

---

## 1. 고전 랜덤 워크 복습

### 1.1 직선 위의 랜덤 워크

정수 위의 고전 랜덤 워크: 각 단계에서 확률 $1/2$로 왼쪽 또는 오른쪽으로 이동합니다.

**$t$ 단계 후 위치**: $X_t = \sum_{i=1}^{t} s_i$, 여기서 $s_i \in \{-1, +1\}$ 균일 분포.

**성질**:
- 평균: $\langle X_t \rangle = 0$
- 분산: $\text{Var}(X_t) = t$
- 표준편차: $\sigma = \sqrt{t}$ (확산적 퍼짐)
- 분포: 큰 $t$에서 대략 가우시안 (중심극한정리)

### 1.2 그래프 위의 랜덤 워크

인접 행렬 $A$를 가진 그래프 $G = (V, E)$ 위의 랜덤 워크:

$$P_{ij} = \frac{A_{ij}}{d_j}$$

여기서 $d_j$는 꼭짓점 $j$의 차수입니다. 워커는 각 단계에서 무작위 이웃으로 이동합니다.

**혼합 시간(Mixing time)**: 분포가 정상 분포 $\pi_i = d_i / (2|E|)$에 가까워지기까지의 단계 수. $n$-꼭짓점 그래프에서 혼합 시간은 전형적으로 $O(n \log n)$에서 $O(n^3)$입니다.

**도달 시간(Hitting time)**: 소스에서 대상 꼭짓점에 도달하기까지의 기대 단계 수. 길이 $n$인 직선에서: 도달 시간은 $O(n^2)$입니다.

### 1.3 고전 랜덤 워크의 응용

| 응용 | 복잡도 |
|-------------|-----------|
| 그래프 연결성 | $O(n^3)$ |
| 2-SAT | $O(n^2)$ 무작위 |
| 비방향 s-t 연결성 | $O(n^2)$ |
| 볼록체 부피 추정 | 다항식 (마르코프 체인 사용) |
| PageRank | 웹 그래프의 혼합 시간 |

---

## 2. 이산 시간 양자 걷기

### 2.1 정의

직선 위의 이산 시간 양자 걷기(DTQW)는 두 레지스터가 필요합니다:
- **위치 레지스터**: $|x\rangle$, $x \in \mathbb{Z}$
- **코인 레지스터**: $|c\rangle$, $c \in \{0, 1\}$

### 2.2 걷기 연산자

**코인 연산자** $C$: 코인 레지스터에 작용하여 중첩을 생성합니다. 가장 일반적인 선택은 아다마르 코인입니다.

**시프트 연산자** $S$: 코인 상태에 따라 워커를 이동합니다:
$$S|0\rangle|x\rangle = |0\rangle|x-1\rangle, \quad S|1\rangle|x\rangle = |1\rangle|x+1\rangle$$

### 2.3 탄도적 퍼짐

$|0\rangle|0\rangle$에서 시작하여 $t$ 단계 후, 확률 분포는:
- 가우시안이 **아닙니다**
- $x = \pm t/\sqrt{2}$ 근처에 봉우리가 있는 이중 모달 분포
- **탄도적 퍼짐**: $\sigma \sim t$ (시간에 선형, $\sqrt{t}$이 아님)

---

## 3. 연속 시간 양자 걷기

### 3.1 정의

연속 시간 양자 걷기(CTQW)는 그래프 위의 슈뢰딩거 방정식으로 정의됩니다:
$$i\frac{d}{dt}|\psi(t)\rangle = H_{\text{walk}}|\psi(t)\rangle$$

시간 진화: $|\psi(t)\rangle = e^{-iH_{\text{walk}}t}|\psi(0)\rangle$

### 3.2 코인 불필요

이산 시간 양자 걷기와 달리, CTQW는 코인 레지스터가 필요 없습니다. 그래프 구조 자체가 동역학을 결정합니다.

---

## 4. 코인 양자 걷기

### 4.1 그래프로의 일반화

$d$-정규 그래프 위의 양자 걷기에서 Grover 확산 연산자는 자연스러운 코인입니다:
$$G_d = \frac{2}{d}\mathbf{1}\mathbf{1}^T - I_d$$

### 4.2 Szegedy 양자 걷기

Szegedy의 공식화는 고전 마르코프 체인 $P$를 직접 양자화합니다. 핵심 성질: 걷기 연산자의 고유값 갭은 $P$의 스펙트럼 갭 $\delta$와 $\Delta_W = \Theta(\sqrt{\delta})$로 관련됩니다 — 스펙트럼 갭의 이차 개선.

---

## 5. 양자 걷기 탐색

### 5.1 Grover 알고리즘의 양자 걷기 형태

Grover 탐색은 완전 그래프 $K_N$ 위의 양자 걷기로 재구성될 수 있습니다.

### 5.2 공간 탐색

| 그래프 | 고전 도달 시간 | 양자 도달 시간 |
|--------|-------------|-------------|
| 완전 $K_N$ | $O(N)$ | $O(\sqrt{N})$ |
| 2D 격자 | $O(N \log N)$ | $O(\sqrt{N} \log N)$ |
| 초입방체 | $O(2^n)$ | $O(\sqrt{2^n})$ |

---

## 6. 그래프 문제 응용

### 6.1 원소 구별성

**문제**: $N$개 원소가 주어졌을 때, 두 개가 같은지 판별.
**고전**: $O(N)$, **양자**: $O(N^{2/3})$

### 6.2 삼각형 찾기

**고전**: $O(n^2)$, **양자**: $O(n^{5/4})$

### 6.3 그래프 동형

양자 걷기가 생성하는 확률 분포가 비동형 그래프에서 다르다는 점을 활용한 휴리스틱 접근법.

---

## 7. 양자 걷기로부터의 양자 속도향상

### 7.1 양자 걷기가 빠른 이유

**간섭**: 양자 워커가 여러 경로를 동시에 탐색. 대상으로 향하는 경로가 보강 간섭하고, 멀어지는 경로가 상쇄 간섭합니다.

**탄도 수송**: 고전 랜덤 워크가 확산적($\sigma \sim \sqrt{t}$)으로 퍼지는 반면, 양자 걷기는 탄도적($\sigma \sim t$)으로 퍼집니다.

### 7.2 계산 모델로서의 양자 걷기

양자 걷기는 **양자 계산에 대해 보편적**입니다: 적절하게 구성된 그래프 위의 양자 걷기로 모든 양자 회로를 시뮬레이션할 수 있습니다.

---

## 8. 양자 걷기의 보편성

### 8.1 연속 시간 걷기

Childs는 희소 그래프 위의 CTQW가 모든 양자 계산을 시뮬레이션할 수 있음을 증명했습니다.

### 8.2 이산 시간 걷기

이산 시간 코인 걷기의 보편성은 충분히 복잡한 그래프 구조, 다른 꼭짓점에서 다른 코인을 선택할 수 있는 능력, 적절한 초기 상태 준비를 필요로 합니다.

---

## 9. Python 구현

### 9.1 직선 위의 이산 시간 양자 걷기

```python
import numpy as np

def discrete_quantum_walk_line(n_steps, n_positions=None, coin='hadamard',
                                initial_coin=None):
    """정수 직선 위의 이산 시간 양자 걷기를 시뮬레이션합니다.

    워커는 코인(내부) 자유도와 위치(외부) 자유도를 가집니다.
    각 단계에서:
    1. 코인 레지스터에 코인 연산자 적용
    2. 코인 상태에 따라 위치 시프트

    이는 고전 랜덤 워크의 확산적 퍼짐(sigma ~ sqrt(t)) 대신
    탄도적 퍼짐(sigma ~ t)을 생성합니다.
    """
    if n_positions is None:
        n_positions = 2 * n_steps + 1

    center = n_positions // 2

    coins = {
        'hadamard': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        'balanced': np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2),
    }
    C = coins.get(coin, coins['hadamard'])

    state = np.zeros(2 * n_positions, dtype=complex)
    if initial_coin is None:
        initial_coin = np.array([1, 0], dtype=complex)

    state[0 * n_positions + center] = initial_coin[0]
    state[1 * n_positions + center] = initial_coin[1]

    def shift(state):
        new_state = np.zeros_like(state)
        for x in range(1, n_positions):
            new_state[0 * n_positions + x - 1] = state[0 * n_positions + x]
        for x in range(n_positions - 1):
            new_state[1 * n_positions + x + 1] = state[1 * n_positions + x]
        return new_state

    def apply_coin(state, C):
        new_state = np.zeros_like(state)
        for x in range(n_positions):
            c0 = state[0 * n_positions + x]
            c1 = state[1 * n_positions + x]
            new_state[0 * n_positions + x] = C[0, 0] * c0 + C[0, 1] * c1
            new_state[1 * n_positions + x] = C[1, 0] * c0 + C[1, 1] * c1
        return new_state

    for step in range(n_steps):
        state = apply_coin(state, C)
        state = shift(state)

    prob_dist = np.zeros(n_positions)
    for x in range(n_positions):
        prob_dist[x] = (abs(state[0 * n_positions + x]) ** 2 +
                        abs(state[1 * n_positions + x]) ** 2)

    positions = np.arange(n_positions) - center
    return positions, prob_dist, state


# 양자 대 고전 걷기 시연
print("=" * 65)
print("이산 시간 양자 걷기 대 고전 랜덤 워크")
print("=" * 65)

n_steps = 50
positions_q, prob_q, _ = discrete_quantum_walk_line(n_steps, coin='hadamard')

mean_q = np.sum(positions_q * prob_q)
std_q = np.sqrt(np.sum(positions_q ** 2 * prob_q) - mean_q ** 2)

print(f"\n{n_steps} 단계 후:")
print(f"  양자: 평균 = {mean_q:.2f}, 표준편차 = {std_q:.2f}")
print(f"  고전 표준편차 (이론): {np.sqrt(n_steps):.2f}")
print(f"  퍼짐 속도향상: {std_q / np.sqrt(n_steps):.2f}x")
```

### 9.2 연속 시간 양자 걷기

```python
import numpy as np
from scipy.linalg import expm

def continuous_quantum_walk(adjacency, gamma, t, initial_vertex):
    """그래프 위의 연속 시간 양자 걷기를 시뮬레이션합니다."""
    N = adjacency.shape[0]
    H = gamma * adjacency
    psi0 = np.zeros(N, dtype=complex)
    psi0[initial_vertex] = 1.0
    psi_t = expm(-1j * H * t) @ psi0
    prob_dist = np.abs(psi_t) ** 2
    return prob_dist, psi_t


def build_graph(graph_type, n):
    """일반적인 그래프 유형에 대한 인접 행렬을 구축합니다."""
    if graph_type == 'line':
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        return A
    elif graph_type == 'cycle':
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[(i + 1) % n, i] = 1
        return A
    elif graph_type == 'complete':
        return np.ones((n, n)) - np.eye(n)

print("=" * 65)
print("다양한 그래프에서의 연속 시간 양자 걷기")
print("=" * 65)

for graph_type, n in [('line', 10), ('cycle', 8), ('complete', 6)]:
    A = build_graph(graph_type, n)
    print(f"\n--- {graph_type} 그래프 (n={n}), 꼭짓점 0에서 시작 ---")
    for t in [0.0, 1.0, 3.0]:
        prob, _ = continuous_quantum_walk(A, 1.0, t, 0)
        prob_str = ', '.join(f'{p:.3f}' for p in prob[:6])
        print(f"  t={t:.1f}: [{prob_str}, ...]")
```

---

## 10. 연습 문제

### 연습 1: 양자 걷기 퍼짐

아다마르 코인을 사용한 직선 위 이산 시간 양자 걷기에 대해:
(a) $t = 10, 20, 50, 100, 200$ 단계의 걷기를 시뮬레이션하고 표준편차 $\sigma(t)$를 계산하세요.
(b) $\sigma(t) = a \cdot t^b$를 피팅하여 $b \approx 1$임을 확인하세요.
(c) 매 단계 후 코인 레지스터를 측정하면 어떤 일이 발생합니까?

### 연습 2: CTQW 완벽 상태 전달

$n$개 꼭짓점의 경로에서 CTQW:
(a) 꼭짓점 0에서 시작하여 꼭짓점 $n-1$에 있을 확률이 최대인 시간을 찾으세요.
(b) $n = 2, 3, 4, 5, 6$에 대해 완벽 상태 전달이 발생합니까?

### 연습 3: 양자 걷기 탐색 최적화

2D 격자에서 양자 걷기 탐색:
(a) CTQW 탐색 해밀토니안을 구현하세요.
(b) 최적 $\gamma$를 찾으세요.
(c) 격자 크기 $N = 16, 36, 64$에 대해 성공 확률 대 시간을 그리세요.

### 연습 4: 양자 걷기 그래프 분류

양자 걷기 지문을 사용하여 소형 그래프를 분류하세요.

### 연습 5: Szegedy 걷기 스펙트럼

순환 $C_n$ 위의 고전 랜덤 워크에 대한 Szegedy 양자 걷기를 구현하고 $\Delta_W = \Theta(\sqrt{\delta})$ 관계를 확인하세요.

---

[← 이전: 양자 시뮬레이션](18_Quantum_Simulation.md) | [다음: 잡음과 양자 채널 →](20_Noise_and_Quantum_Channels.md)

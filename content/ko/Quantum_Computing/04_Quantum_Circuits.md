# 레슨 4: 양자 회로(Quantum Circuits)

[<- 이전: 양자 게이트(Quantum Gates)](03_Quantum_Gates.md) | [다음: 얽힘과 벨 상태(Entanglement and Bell States) ->](05_Entanglement_and_Bell_States.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 양자 회로 다이어그램을 읽고 해석하며, 와이어(큐비트), 게이트, 측정 연산을 식별한다
2. 양자 계산의 회로 모델을 설명하고 고전 회로와의 차이를 설명한다
3. 회로 깊이(depth)와 너비(width)를 정의하고 알고리즘 복잡도에 대한 의의를 설명한다
4. 양자 회로 다이어그램을 행렬 표현식으로 변환하고 출력 상태를 계산한다
5. 측정이 양자 상태를 고전 비트로 변환하는 고전-양자 경계를 다룬다
6. 행렬 곱셈을 사용하여 파이썬으로 간단한 양자 회로 시뮬레이터를 구축한다
7. 양자 회로를 고전적으로 시뮬레이션하는 계산 비용을 분석한다

---

양자 회로 모델은 양자 계산을 기술하는 표준 프레임워크입니다. 고전 알고리즘이 논리 게이트를 연결한 시퀀스로 표현되듯, 양자 알고리즘은 큐비트에 적용된 양자 게이트의 시퀀스로 표현됩니다. 이 모델은 양자 연산의 수학적 기술([레슨 3](03_Quantum_Gates.md)에서 학습한 유니터리 행렬)과 하드웨어 상의 물리적 구현 사이에 깔끔한 추상화 계층을 제공합니다.

양자 회로를 읽고, 쓰고, 시뮬레이션하는 방법을 이해하는 것은 앞으로 다룰 모든 양자 알고리즘, 즉 도이치-조자 알고리즘([레슨 7](07_Deutsch_Jozsa_Algorithm.md))과 그로버 탐색([레슨 8](08_Grovers_Search.md))을 학습하는 데 필수적입니다. 이 레슨에서는 개념적 프레임워크와 함께 이후 레슨들에서 활용할 실용적인 파이썬 시뮬레이터를 개발합니다.

> **비유:** 양자 회로는 악보(musical score)와 같습니다 -- 각 선(와이어)은 큐비트를 나타내고, 게이트는 왼쪽에서 오른쪽 순서로 연주해야 할 음표입니다. 음악가가 악보를 보고 어떤 음표를 언제 연주할지 알듯이, 양자 컴퓨터는 회로를 읽어 어떤 연산을 어떤 순서로 수행할지 파악합니다. 여러 성부의 상호작용에서 음악의 아름다움이 나오듯, 양자 회로의 강력함은 여러 큐비트 간의 상호작용에서 비롯됩니다.

## 목차

1. [회로 모델](#1-회로-모델)
2. [회로 표기법](#2-회로-표기법)
3. [회로 깊이와 너비](#3-회로-깊이와-너비)
4. [회로의 행렬 표현](#4-회로의-행렬-표현)
5. [고전-양자 경계](#5-고전-양자-경계)
6. [회로 시뮬레이터 구축](#6-회로-시뮬레이터-구축)
7. [시뮬레이션 복잡도](#7-시뮬레이션-복잡도)
8. [연습 문제](#8-연습-문제)

---

## 1. 회로 모델

### 1.1 구성 요소

양자 회로는 세 가지 유형의 구성 요소를 가집니다:

1. **와이어(수평선)**: 각 와이어는 하나의 큐비트를 나타냅니다. 시간은 와이어를 따라 왼쪽에서 오른쪽으로 흐릅니다. 큐비트는 회로 전체에 걸쳐 유지됩니다; 일부 고전 게이트 모델의 신호처럼 소비되지 않습니다.

2. **게이트(와이어 위의 상자)**: 각 게이트는 하나 이상의 큐비트에 적용되는 유니터리 연산입니다. 단일 큐비트 게이트는 하나의 와이어에, 다중 큐비트 게이트는 여러 와이어에 걸쳐 위치합니다.

3. **측정(미터 기호)**: 측정은 양자 정보를 고전 정보로 변환합니다. 일반적으로 회로의 끝에 위치합니다.

### 1.2 실행 모델

양자 회로는 다음과 같이 실행됩니다:

1. **초기화**: 모든 큐비트는 알려진 상태, 일반적으로 $|0\rangle$에서 시작합니다.
2. **게이트 적용**: 게이트는 왼쪽에서 오른쪽으로 적용됩니다. 서로 다른 큐비트에 작용하고 같은 시간 단계에 나타나는 게이트는 병렬로 실행될 수 있습니다.
3. **측정**: 끝(또는 중간 지점)에서 큐비트를 측정하여 고전적인 출력 비트를 생성합니다.

### 1.3 회로 vs 알고리즘

양자 회로는 고정된 입력 크기에 대한 *고정된* 계산을 기술합니다. 양자 *알고리즘*은 회로들의 묶음(family)입니다 -- 각 입력 크기마다 하나의 회로 -- 그리고 주어진 입력에 대한 회로를 구성하는 방법에 대한 고전적인 기술을 포함합니다.

예를 들어, $N = 2^n$개 원소에 대한 그로버 탐색은 회로 묶음으로 기술됩니다, 각 $n$에 대해 하나씩. $n = 10$에 대한 회로는 10개 큐비트와 특정 게이트 시퀀스를 가지며, $n = 20$에 대한 회로는 20개 큐비트와 다른 (더 큰) 시퀀스를 가집니다.

---

## 2. 회로 표기법

### 2.1 텍스트 기반 회로 다이어그램

텍스트 환경에서 작업하므로 ASCII 표기법을 사용하여 회로를 표현합니다. 다음은 벨 상태(Bell state) 준비 회로의 예시입니다:

```
q0: ─[H]─●─
          │
q1: ──────X─
```

- `q0`, `q1`: 큐비트 레이블
- `[H]`: q0에 대한 아다마르 게이트
- `●`: CNOT의 제어 큐비트
- `X`: CNOT의 대상 큐비트 (조건부로 적용되는 NOT 게이트)
- `│`: 제어와 대상을 연결하는 수직선

### 2.2 일반적인 게이트 기호

| 기호 | 게이트 | 설명 |
|--------|------|-------------|
| `[H]` | 아다마르 | 중첩 생성/소멸 |
| `[X]` | 파울리-X | 비트 뒤집기(bit flip) |
| `[Z]` | 파울리-Z | 위상 뒤집기(phase flip) |
| `[S]` | S 게이트 | $\pi/2$ 위상 |
| `[T]` | T 게이트 | $\pi/4$ 위상 |
| `●─X` | CNOT | 제어-NOT |
| `●─●` | CZ | 제어-Z |
| `[M]` | 측정 | 큐비트를 고전 비트로 붕괴 |
| `[Rz(θ)]` | 회전 | 매개변수화된 게이트 |

### 2.3 순서 규약

**중요**: 표기법과 코드에서 큐비트 0은 *최하위 비트(least significant bit)*입니다. 2큐비트 시스템의 경우:

$$|q_1 q_0\rangle \quad \text{여기서 } q_1 \text{이 최상위 비트}$$

상태 벡터는 인덱스 0, 1, 2, 3에 해당하는 $|00\rangle, |01\rangle, |10\rangle, |11\rangle$ 순서로 정렬됩니다.

```python
import numpy as np

# Circuit notation: translating diagrams to operations

# Bell state preparation circuit:
# q0: -[H]-*-
#           |
# q1: ------X-

# Step 1: Initialize |00>
state = np.array([1, 0, 0, 0], dtype=complex)

# Step 2: Apply H to q0 (H tensor I on the full state space)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)

# Why kron(H, I) and not kron(I, H)?
# Because q0 is the FIRST qubit in the tensor product.
# Our convention: |q1 q0> means q0 is rightmost (least significant).
# But in the Kronecker product, the FIRST matrix acts on the FIRST qubit.
# So H on q0 = kron(I_q1, H_q0) if we label q1 as "first/left" and q0 as "second/right".
# HOWEVER, we follow the common CS convention where q0 is the LEAST significant bit,
# and the Kronecker product has q0 as the rightmost factor.
# So H on q0 = kron(I, H) in |q1, q0> ordering.

# Actually, let's be precise:
# State vector order: |00>, |01>, |10>, |11> where rightmost bit = q0
# To apply H to q0: kron(I, H)  (I acts on q1, H acts on q0)
# To apply H to q1: kron(H, I)  (H acts on q1, I acts on q0)

H_on_q0 = np.kron(I, H)  # H on qubit 0
state = H_on_q0 @ state
print("After H on q0:")
print(f"  State: {state}")
print(f"  = ({state[0]:.4f})|00> + ({state[1]:.4f})|01> + "
      f"({state[2]:.4f})|10> + ({state[3]:.4f})|11>")

# Step 3: Apply CNOT (control=q0, target=q1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

state = CNOT @ state
print("\nAfter CNOT (control=q0, target=q1):")
print(f"  State: {state}")
print(f"  = ({state[0]:.4f})|00> + ({state[1]:.4f})|01> + "
      f"({state[2]:.4f})|10> + ({state[3]:.4f})|11>")
print(f"\nThis is the Bell state |Phi+> = (|00> + |11>)/sqrt(2)")
```

---

## 3. 회로 깊이와 너비

### 3.1 정의

- **너비(Width)**: 회로의 큐비트 수. 상태 공간의 차원($2^{\text{width}}$)을 결정합니다.

- **깊이(Depth)**: 회로의 시간 단계(레이어) 수로, 같은 레이어의 게이트들은 서로 겹치지 않는 큐비트에 작용하며 병렬로 실행될 수 있습니다. 알고리즘 속도에 대한 핵심 지표입니다.

### 3.2 예제: 깊이 계산

다음 회로를 생각해 봅시다:
```
q0: ─[H]─●─────[T]─
          │
q1: ─[H]─X──●──────
             │
q2: ─────────X──[H]─
```

- **너비**: 3 큐비트
- **레이어 1**: q0에 H, q1에 H (병렬 -- 서로 다른 큐비트)
- **레이어 2**: CNOT(q0, q1)
- **레이어 3**: CNOT(q1, q2)
- **레이어 4**: q0에 T, q2에 H (병렬)
- **깊이**: 4

### 3.3 깊이가 중요한 이유

양자 상태는 취약합니다 -- 시간이 지남에 따라 결맞음(coherence)을 잃습니다(결어긋남, decoherence). 깊이 $d$의 회로는 큐비트가 결어긋나기 전에 $d$개 레이어를 모두 완료해야 합니다. 따라서 회로 깊이는 근거리 양자 컴퓨팅에서 가장 중요한 복잡도 지표입니다.

**게이트 수 vs 깊이**: 회로에 게이트가 많더라도, 대부분의 게이트를 병렬화할 수 있다면 깊이는 낮을 수 있습니다. 반대로, 게이트 수가 적어도 순차적 의존성이 있으면 깊이가 높을 수 있습니다.

```python
import numpy as np

# Analyzing circuit structure

class CircuitAnalyzer:
    """
    Simple circuit structure analyzer.

    Why track layers instead of just gates? On real quantum hardware,
    the total execution time is determined by the DEPTH (number of
    sequential layers), not the total gate count. Gates on different
    qubits in the same layer run in parallel.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []  # List of (gate_name, qubit_indices)

    def add_gate(self, name, qubits):
        """Add a gate acting on specified qubits."""
        self.gates.append((name, tuple(qubits)))

    def compute_depth(self):
        """
        Compute circuit depth using a greedy layering algorithm.

        Why greedy? We assign each gate to the earliest possible layer
        (the first layer where none of its qubits are already occupied).
        This gives the minimum depth, assuming all gates take 1 time unit.
        """
        # Track the latest layer each qubit is involved in
        qubit_latest_layer = [-1] * self.n_qubits
        layers = []

        for name, qubits in self.gates:
            # This gate must go after all previous gates on any of its qubits
            min_layer = max(qubit_latest_layer[q] for q in qubits) + 1

            # Ensure layers list is long enough
            while len(layers) <= min_layer:
                layers.append([])

            layers[min_layer].append((name, qubits))

            # Update qubit usage
            for q in qubits:
                qubit_latest_layer[q] = min_layer

        return len(layers), layers

# Example: Bell state circuit
print("=== Bell State Circuit ===\n")
bell = CircuitAnalyzer(2)
bell.add_gate("H", [0])
bell.add_gate("CNOT", [0, 1])

depth, layers = bell.compute_depth()
print(f"Width: {bell.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(bell.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")

# Example: GHZ state circuit (3-qubit entangled state)
print("\n=== GHZ State Circuit ===\n")
ghz = CircuitAnalyzer(3)
ghz.add_gate("H", [0])
ghz.add_gate("CNOT", [0, 1])
ghz.add_gate("CNOT", [1, 2])

depth, layers = ghz.compute_depth()
print(f"Width: {ghz.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(ghz.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")

# Example: Circuit with parallelism
print("\n=== Circuit with Parallelism ===\n")
par = CircuitAnalyzer(4)
par.add_gate("H", [0])
par.add_gate("H", [1])
par.add_gate("H", [2])
par.add_gate("H", [3])
par.add_gate("CNOT", [0, 1])
par.add_gate("CNOT", [2, 3])
par.add_gate("CNOT", [1, 2])

depth, layers = par.compute_depth()
print(f"Width: {par.n_qubits} qubits")
print(f"Depth: {depth} layers")
print(f"Gate count: {len(par.gates)}")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")
print("\nNote: 4 H gates execute in parallel (depth 1, not 4)")
print("and 2 CNOTs on disjoint qubits also parallelize!")
```

---

## 4. 회로의 행렬 표현

### 4.1 순차 게이트 = 행렬 곱셈

회로가 게이트 $U_1$, 그 다음 $U_2$, 그 다음 $U_3$을 적용한다면, 전체 유니터리는:

$$U_{\text{circuit}} = U_3 \cdot U_2 \cdot U_1$$

순서가 역전됨에 주의하세요: 행렬 곱셈은 오른쪽에서 왼쪽으로 적용되기 때문입니다 (가장 오른쪽 행렬이 상태 벡터에 먼저 작용).

최종 상태는:

$$|\psi_{\text{out}}\rangle = U_{\text{circuit}} |\psi_{\text{in}}\rangle = U_3 U_2 U_1 |\psi_{\text{in}}\rangle$$

### 4.2 큐비트 부분 집합에 대한 게이트

게이트 $G$가 $n$큐비트 시스템에서 일부 큐비트에만 작용할 때, 텐서곱을 사용하여 나머지 큐비트에 항등 연산으로 "패딩"해야 합니다:

- 큐비트 0에만 작용하는 게이트 (3큐비트 시스템에서): $I \otimes I \otimes G$
- 큐비트 1에만 작용: $I \otimes G \otimes I$
- 큐비트 0, 1에 작용하는 2큐비트 게이트: $I \otimes G_{01}$

큐비트 순서 $|q_{n-1} \cdots q_1 q_0\rangle$에 대한 일반 규칙: 텐서곱은 가장 오른쪽 큐비트($q_0$)에 작용하는 게이트를 가장 오른쪽 인수로 가집니다.

### 4.3 예제: 3큐비트 회로의 전체 행렬

다음 회로를 생각해 봅시다:
```
q0: ─[H]─●─
          │
q1: ──────X─
q2: ──────── (유휴)
```

1단계: $U_1 = I_{q2} \otimes I_{q1} \otimes H_{q0}$

2단계: $U_2 = I_{q2} \otimes \text{CNOT}_{q0,q1}$

전체: $U_{\text{circuit}} = U_2 \cdot U_1$

```python
import numpy as np

# Matrix representation of multi-qubit circuits

def gate_on_qubit(gate, target_qubit, n_qubits):
    """
    Create the full-system matrix for a single-qubit gate acting
    on a specific qubit in an n-qubit system.

    Why tensor products? Each qubit has its own 2D space. The full
    system lives in the tensor product of all these spaces. A gate
    on one qubit acts as identity on all others.
    """
    # Build up the full matrix using Kronecker products
    # Qubit ordering: |q_{n-1} ... q_1 q_0>
    # The tensor product factors are ordered from q_{n-1} (left) to q_0 (right)
    matrices = []
    for q in range(n_qubits - 1, -1, -1):  # From q_{n-1} down to q_0
        if q == target_qubit:
            matrices.append(gate)
        else:
            matrices.append(np.eye(2, dtype=complex))

    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)

    return result

# Standard gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

# CNOT for 3 qubits (control=q0, target=q1, q2 idle)
# This is I_q2 tensor CNOT_{q1,q0}
CNOT_q0q1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)
CNOT_3q = np.kron(I2, CNOT_q0q1)  # I on q2, CNOT on q1,q0

# Build the circuit: H on q0, then CNOT(q0, q1)
print("=== 3-Qubit Circuit: H(q0) then CNOT(q0,q1) ===\n")

# Step 1: H on q0
U1 = gate_on_qubit(H, 0, 3)
print(f"U1 (H on q0) shape: {U1.shape}")

# Step 2: CNOT on q0, q1
U2 = CNOT_3q
print(f"U2 (CNOT q0,q1) shape: {U2.shape}")

# Full circuit
U_circuit = U2 @ U1
print(f"U_circuit shape: {U_circuit.shape}")

# Apply to |000>
state_in = np.zeros(8, dtype=complex)
state_in[0] = 1  # |000>
state_out = U_circuit @ state_in

print(f"\nInput:  |000>")
print(f"Output: ")
for i in range(8):
    if abs(state_out[i]) > 1e-10:
        label = format(i, '03b')
        print(f"  |{label}>: {state_out[i]:.4f} (P = {abs(state_out[i])**2:.4f})")

print("\nResult: (|000> + |110>)/sqrt(2)")
print("q2 remains |0>, while q0 and q1 form a Bell pair.")
```

---

## 5. 고전-양자 경계

### 5.1 회로에서의 측정

측정은 큐비트를 양자에서 고전으로 변환합니다. 회로에서 측정은 일반적으로 미터 기호로 표현되며 고전 비트를 생성합니다.

**핵심 규칙**:
1. 측정은 비가역적입니다 (중첩을 붕괴시킴)
2. 측정 후 큐비트는 확정적인 상태($|0\rangle$ 또는 $|1\rangle$)가 됩니다
3. 측정된 큐비트에 추가 양자 게이트를 적용하는 것이 가능하지만, 붕괴된 상태에 작용합니다
4. 측정 결과는 확률적입니다

### 5.2 지연 측정 원리

중요한 정리: **측정은 항상 회로의 끝으로 미룰 수 있으며**, 출력 확률 분포는 변하지 않습니다. 이는 양자 회로를 다음과 같이 처리할 수 있음을 의미합니다:

1. 모든 큐비트에 대한 유니터리 변환 $U$
2. 이어서 모든 큐비트 측정

원래 회로에 중간 측정이 있더라도, 다음으로 시뮬레이션할 수 있습니다:
- 각 중간 측정을 새로운 "보조(ancilla)" 큐비트에 대한 CNOT으로 대체
- 끝에서 모든 보조 큐비트 측정

이는 이론적 간소화이자 회로 분석을 위한 실용적인 도구입니다.

### 5.3 고전적 제어

일부 회로는 측정 결과를 사용하여 이후 게이트를 제어합니다(고전적 피드백). 예를 들어:

```
q0: ─[H]─[M]═══╗
                ║
q1: ────────[X if M=1]─
```

여기서 q1의 X 게이트는 q0가 1로 측정된 경우에만 적용됩니다. 이를 **고전적으로 제어된(classically controlled)** 연산이라 하며, 양자 원격 전송(quantum teleportation) 같은 프로토콜에 필수적입니다.

```python
import numpy as np

# Simulating measurement in quantum circuits

def simulate_measurement(state, qubit, n_qubits):
    """
    Simulate measuring one qubit in the computational basis.

    Why is this non-trivial? Measuring one qubit in a multi-qubit system
    requires:
    1. Computing probabilities of 0 and 1 for that qubit
    2. Randomly choosing an outcome
    3. Collapsing the full state to the post-measurement state
    4. Re-normalizing

    Returns: (outcome, post_measurement_state)
    """
    dim = 2**n_qubits

    # Compute probability of measuring qubit as |0>
    # Sum |amplitude|^2 over all basis states where the target qubit is 0
    prob_0 = 0.0
    for i in range(dim):
        # Check if bit 'qubit' of index i is 0
        if (i >> qubit) & 1 == 0:
            prob_0 += abs(state[i])**2

    prob_1 = 1.0 - prob_0

    # Random measurement outcome
    outcome = 0 if np.random.random() < prob_0 else 1

    # Collapse: zero out amplitudes inconsistent with outcome
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bit_value = (i >> qubit) & 1
        if bit_value == outcome:
            new_state[i] = state[i]

    # Re-normalize
    norm = np.linalg.norm(new_state)
    if norm > 1e-15:
        new_state = new_state / norm

    return outcome, new_state

# Example: Measure one qubit of a Bell state
np.random.seed(42)

print("=== Measuring a Bell State ===\n")

# Bell state: (|00> + |11>)/sqrt(2)
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
print(f"Initial state: (|00> + |11>)/sqrt(2)")

# Measure qubit 0
print("\nMeasuring qubit 0 ten times:")
for trial in range(10):
    outcome, post_state = simulate_measurement(bell.copy(), 0, 2)
    # Format post-measurement state
    components = []
    for i in range(4):
        if abs(post_state[i]) > 1e-10:
            components.append(f"|{format(i, '02b')}>")
    print(f"  Trial {trial}: q0={outcome}, collapsed to {', '.join(components)}")

# Show correlation: if we measure q0=0, then q1 must be 0
# if we measure q0=1, then q1 must be 1
print("\n--- Demonstrating Entanglement Correlation ---")
print("In a Bell state, measuring one qubit determines the other:\n")

outcomes = {0: 0, 1: 0}
for _ in range(10000):
    outcome0, post = simulate_measurement(bell.copy(), 0, 2)
    outcome1, _ = simulate_measurement(post, 1, 2)
    assert outcome0 == outcome1, "Bell state qubits must agree!"
    outcomes[outcome0] += 1

print(f"  10000 trials: q0=q1=0 occurred {outcomes[0]} times, "
      f"q0=q1=1 occurred {outcomes[1]} times")
print(f"  Outcomes ALWAYS agree (entanglement!)")
```

---

## 6. 회로 시뮬레이터 구축

단일 큐비트 게이트, CNOT, 측정을 지원하는 간단하지만 완전한 양자 회로 시뮬레이터를 구축해 봅시다.

```python
import numpy as np

class QuantumCircuit:
    """
    A simple quantum circuit simulator using state vector simulation.

    Why state vector simulation? It tracks the full quantum state
    (all 2^n amplitudes) and applies gates via matrix multiplication.
    This is the most straightforward simulation method and gives
    exact results, but scales exponentially with qubit count.

    For small circuits (< ~25 qubits), this is perfectly practical.
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        # Initialize to |00...0>
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.operations = []  # Log of operations for display

    def _full_gate_matrix(self, gate_matrix, target_qubits):
        """
        Build the full 2^n x 2^n matrix for a gate acting on specific qubits.

        Why build the full matrix? For small systems, explicit matrix construction
        is clear and correct. Production simulators use more efficient approaches
        (e.g., applying gates directly to the state vector without building the
        full matrix), but this is ideal for learning.
        """
        if len(target_qubits) == 1:
            # Single-qubit gate: tensor product with identities
            q = target_qubits[0]
            matrices = []
            for i in range(self.n_qubits - 1, -1, -1):
                if i == q:
                    matrices.append(gate_matrix)
                else:
                    matrices.append(np.eye(2, dtype=complex))
            result = matrices[0]
            for m in matrices[1:]:
                result = np.kron(result, m)
            return result
        else:
            # Multi-qubit gate: more complex embedding
            # For now, handle specific cases
            return gate_matrix  # Assumes gate_matrix is already full-size

    def h(self, qubit):
        """Apply Hadamard gate to a qubit."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        full = self._full_gate_matrix(H, [qubit])
        self.state = full @ self.state
        self.operations.append(f"H(q{qubit})")

    def x(self, qubit):
        """Apply Pauli-X gate to a qubit."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        full = self._full_gate_matrix(X, [qubit])
        self.state = full @ self.state
        self.operations.append(f"X(q{qubit})")

    def z(self, qubit):
        """Apply Pauli-Z gate to a qubit."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        full = self._full_gate_matrix(Z, [qubit])
        self.state = full @ self.state
        self.operations.append(f"Z(q{qubit})")

    def rz(self, qubit, theta):
        """Apply Rz(theta) gate to a qubit."""
        Rz = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
        full = self._full_gate_matrix(Rz, [qubit])
        self.state = full @ self.state
        self.operations.append(f"Rz({theta:.3f})(q{qubit})")

    def cnot(self, control, target):
        """
        Apply CNOT gate with specified control and target qubits.

        Why build CNOT this way? The CNOT matrix depends on which qubits
        are control and target. We construct it by iterating over all basis
        states: for each state, if the control bit is 1, flip the target bit.
        """
        full = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim):
            control_bit = (i >> control) & 1
            if control_bit == 1:
                # Flip the target bit
                j = i ^ (1 << target)
                full[j, i] = 1
            else:
                full[i, i] = 1
        self.state = full @ self.state
        self.operations.append(f"CNOT(q{control}->q{target})")

    def measure_all(self, shots=1024):
        """
        Simulate measuring all qubits, repeated 'shots' times.

        Why multiple shots? A single measurement gives one random outcome.
        To estimate the probability distribution, we need many repetitions.
        """
        probabilities = np.abs(self.state)**2
        outcomes = np.random.choice(self.dim, size=shots, p=probabilities)
        counts = {}
        for outcome in outcomes:
            label = format(outcome, f'0{self.n_qubits}b')
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def get_statevector(self):
        """Return the current state vector."""
        return self.state.copy()

    def get_probabilities(self):
        """Return measurement probabilities for each basis state."""
        probs = np.abs(self.state)**2
        result = {}
        for i in range(self.dim):
            if probs[i] > 1e-10:
                label = format(i, f'0{self.n_qubits}b')
                result[label] = probs[i]
        return result

    def __repr__(self):
        ops = " -> ".join(self.operations) if self.operations else "(empty)"
        return f"QuantumCircuit({self.n_qubits}q): {ops}"


# === Demonstration ===

# Bell state preparation
print("=== Bell State Circuit ===\n")
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
print(f"Circuit: {qc}")
print(f"State vector: {qc.get_statevector()}")
print(f"Probabilities: {qc.get_probabilities()}")
np.random.seed(42)
print(f"Measurement (1000 shots): {qc.measure_all(1000)}")

# GHZ state: (|000> + |111>)/sqrt(2)
print("\n=== GHZ State Circuit ===\n")
qc = QuantumCircuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)
print(f"Circuit: {qc}")
print(f"Probabilities: {qc.get_probabilities()}")
print(f"Measurement (1000 shots): {qc.measure_all(1000)}")

# Superposition of all states
print("\n=== Uniform Superposition (3 qubits) ===\n")
qc = QuantumCircuit(3)
for i in range(3):
    qc.h(i)
print(f"Circuit: {qc}")
probs = qc.get_probabilities()
print(f"Probabilities (should all be 1/8 = 0.125):")
for label, prob in probs.items():
    print(f"  |{label}>: {prob:.4f}")
```

시뮬레이터를 사용한 더 복잡한 예제:

```python
import numpy as np

# Using the QuantumCircuit class defined above
# (In practice, you would import or define it in the same file)

# Demonstrate: creating a specific superposition
# Target: |psi> = (|00> + |01> + |10>)/sqrt(3)
# This requires more careful gate selection

print("=== Custom State Preparation ===\n")
print("Goal: prepare (|00> + |01> + |10>)/sqrt(3)")
print("Strategy: Use Ry rotation to get amplitude 1/sqrt(3) on |0> and")
print("sqrt(2/3) on |1>, then conditionally create superposition.\n")

# This demonstrates that not every state has a simple circuit!
# For now, let's verify the simulator on known circuits.

# Verify: X gate followed by CNOT creates |11> from |00>
qc = QuantumCircuit(2)
qc.x(0)           # |00> -> |01>
qc.cnot(0, 1)     # |01> -> |11>
print(f"X(q0) then CNOT(q0->q1):")
print(f"  Expected: |11>")
print(f"  Got: {qc.get_probabilities()}")

# Verify: H on both qubits creates uniform superposition
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
print(f"\nH(q0) then H(q1):")
print(f"  Expected: uniform over |00>, |01>, |10>, |11>")
print(f"  Got: {qc.get_probabilities()}")
```

---

## 7. 시뮬레이션 복잡도

### 7.1 고전적 시뮬레이션 비용

고전 컴퓨터에서 양자 회로를 시뮬레이션하려면 $n$ 큐비트에 대한 $2^n$개의 복소 진폭을 추적해야 합니다. 각 게이트 적용은 행렬-벡터 곱셈을 포함합니다:

| 연산 | 시간 복잡도 | 공간 복잡도 |
|-----------|----------------|-----------------|
| 상태 저장 | -- | $O(2^n)$ |
| 1큐비트 게이트 적용 | $O(2^n)$ | $O(2^n)$ |
| 2큐비트 게이트 적용 | 단순하게 $O(4^n)$, 최적화 시 $O(2^n)$ | $O(2^n)$ |
| 측정 | $O(2^n)$ | $O(2^n)$ |

### 7.2 지수적 장벽

지수적 공간 요구사항이 근본적인 장벽입니다:

| 큐비트 | 상태 벡터 크기 | 필요 RAM |
|:---:|:---:|:---:|
| 20 | ~$10^6$ | 16 MB |
| 30 | ~$10^9$ | 16 GB |
| 40 | ~$10^{12}$ | 16 TB |
| 50 | ~$10^{15}$ | 16 PB |

~45-50 큐비트를 넘어서면 전체 상태 벡터 시뮬레이션은 어떤 고전 컴퓨터에서도 비실용적이 됩니다. 이것이 양자 컴퓨터가 잠재적 이점을 가질 수 있는 영역입니다.

### 7.3 특수 경우의 효율적인 시뮬레이션

모든 양자 회로가 고전적으로 시뮬레이션하기 어려운 것은 아닙니다:

- **클리퍼드 회로(Clifford circuits)** (H, S, CNOT, 파울리 게이트만 사용): 고트스만-크닐 정리(Gottesman-Knill theorem)를 사용하여 다항 시간에 시뮬레이션 가능합니다. 이것이 T 게이트가 보편성에 필수적인 이유입니다.
- **낮은 얽힘 회로**: 텐서 네트워크 방법으로 제한된 얽힘을 생성하는 회로를 효율적으로 시뮬레이션할 수 있습니다.
- **얕은 회로**: 2D 격자의 상수 깊이 회로는 때때로 효율적으로 시뮬레이션될 수 있습니다.

```python
import numpy as np
import time

# Benchmarking simulation cost

def benchmark_simulation(n_qubits, n_gates=10):
    """
    Measure the time to simulate a circuit of given size.

    Why benchmark? Understanding the exponential cost of simulation
    is crucial for appreciating why quantum computers could be useful.
    A circuit that takes seconds to simulate at 20 qubits would take
    longer than the age of the universe at 300 qubits.
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # Create a random single-qubit gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    start = time.time()
    for _ in range(n_gates):
        # Apply H to qubit 0 (build full matrix)
        # In a smarter simulator, we would avoid building the full matrix
        matrices = [H] + [np.eye(2, dtype=complex)] * (n_qubits - 1)
        full_gate = matrices[0]
        for m in matrices[1:]:
            full_gate = np.kron(full_gate, m)
        state = full_gate @ state
    elapsed = time.time() - start

    return elapsed, dim

print("=== Simulation Cost Scaling ===\n")
print(f"{'Qubits':>8} {'Dim':>12} {'Time (s)':>12} {'Time/gate (ms)':>16}")
print("-" * 52)

for n in range(4, 15):
    try:
        elapsed, dim = benchmark_simulation(n, n_gates=5)
        print(f"{n:>8} {dim:>12,} {elapsed:>12.4f} {elapsed/5*1000:>16.2f}")
    except MemoryError:
        print(f"{n:>8} {'(memory limit)':>12}")
        break

print("\nObserve: time roughly quadruples with each additional qubit (exponential growth).")
print("This is why quantum computers are needed for large quantum circuits!")
```

---

## 8. 연습 문제

### 연습 1: 회로 추적

다음 회로를 각 게이트 후의 상태를 쓰면서 단계별로 추적하세요:

```
q0: ─[H]─●─[H]─
          │
q1: ──────X─────
```

$|00\rangle$에서 시작하여 최종 상태는 무엇인가요? 이것이 2절의 벨 상태와 같은가요? 왜 그런가요, 또는 왜 다른가요?

### 연습 2: 회로 행렬

다음 회로에 대한 전체 $4 \times 4$ 유니터리 행렬을 계산하세요:

```
q0: ─[X]─●─
          │
q1: ─[H]─X─
```

$|00\rangle$에 적용하고 결과를 단계별 추적과 비교하여 답을 검증하세요.

### 연습 3: 깊이 최적화

다음 회로의 깊이는 4입니다:

```
q0: ─[H]─[T]─●─────
              │
q1: ─[H]─────X─[T]─
```

같은 출력을 내면서 깊이 3을 달성하도록 게이트를 재배치할 수 있나요? 최소 가능한 깊이는 무엇인가요? (힌트: 어떤 게이트를 병렬화할 수 있는지 생각해 보세요.)

### 연습 4: 시뮬레이터 확장

`QuantumCircuit` 시뮬레이터를 다음을 지원하도록 확장하세요:
a) S 게이트와 T 게이트
b) CZ 게이트 (제어-Z)
c) 아무것도 하지 않지만 회로 로그에 시각적 구분을 표시하는 `barrier()` 메서드

$\frac{1}{2}(|00\rangle + i|01\rangle + |10\rangle + i|11\rangle)$ 상태를 생성하는 회로를 구성하여 확장을 테스트하세요.

### 연습 5: 시뮬레이션 한계

a) `QuantumCircuit` 시뮬레이터를 사용하여 여러분의 컴퓨터에서 시뮬레이션할 수 있는 가장 큰 회로(큐비트 수)는 얼마인가요? 증가하는 큐비트 수를 시도하여 경험적으로 결정하세요.
b) 고정된 큐비트 수에서 게이트 수에 따라 시뮬레이션 시간이 어떻게 확장되나요? 실험을 실행하고 결과를 그래프로 그려보세요.
c) 고트스만-크닐 정리(Clifford 회로의 효율적인 시뮬레이션)가 양자 컴퓨팅이 쓸모없다는 의미가 아닌 이유를 설명하세요. 빠진 요소는 무엇인가요?

---

[<- 이전: 양자 게이트(Quantum Gates)](03_Quantum_Gates.md) | [다음: 얽힘과 벨 상태(Entanglement and Bell States) ->](05_Entanglement_and_Bell_States.md)

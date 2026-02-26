# 레슨 9: 양자 푸리에 변환(Quantum Fourier Transform)

[← 이전: 그로버 검색 알고리즘](08_Grovers_Search.md) | [다음: 쇼어의 소인수분해 알고리즘 →](10_Shors_Algorithm.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 고전적 이산 푸리에 변환(Discrete Fourier Transform)과 양자 푸리에 변환(Quantum Fourier Transform) 사이의 연결을 설명할 수 있다
2. QFT 변환을 디랙 표기법(Dirac notation)과 행렬 형식으로 표현할 수 있다
3. 아다마르 게이트(Hadamard gate)와 제어 회전 게이트(controlled rotation gate)를 사용해 QFT 회로를 구성할 수 있다
4. QFT의 게이트 복잡도를 분석할 수 있다: 양자 $O(n^2)$ vs 고전 $O(n \cdot 2^n)$
5. 양자 위상 추정(Quantum Phase Estimation, QPE) 알고리즘과 서브루틴으로서의 역할을 설명할 수 있다
6. 역 QFT(inverse QFT)를 구현하고 언제 필요한지 설명할 수 있다
7. NumPy를 사용해 QFT와 위상 추정을 시뮬레이션할 수 있다

---

양자 푸리에 변환(Quantum Fourier Transform)은 양자 컴퓨팅에서 가장 중요한 서브루틴이라고 해도 과언이 아닙니다. QFT는 쇼어의 소인수분해 알고리즘, 양자 위상 추정, 그리고 지수적 속도 향상을 달성하는 수많은 양자 알고리즘의 핵심에 자리하고 있습니다. QFT를 이해하는 것은 양자 컴퓨터가 현대 암호학을 위협하는 이유, 그리고 화학, 최적화 등의 분야에서 문제를 해결하는 방법을 이해하는 관문입니다.

QFT가 놀라운 이유는 고전적인 DFT와 다른 변환을 계산하기 때문이 아닙니다 — QFT는 수학적으로 완전히 동일한 변환을 계산합니다. 마법은 *어떻게* 계산하느냐에 있습니다: $O(n^2)$개의 게이트만으로 이루어진 양자 회로가 $2^n$개의 진폭을 변환할 수 있는 반면, 최고의 고전 알고리즘인 FFT는 $O(n \cdot 2^n)$번의 연산이 필요합니다. 이 지수적인 게이트 수 감소가 주기 찾기(period-finding) 문제에서 양자 속도 향상을 이끄는 원동력입니다.

> **비유:** QFT는 양자 상태를 위한 프리즘(prism)과 같습니다 — 상태를 주파수 성분으로 분해하여 숨겨진 주기 구조를 드러냅니다. 흰 빛이 프리즘을 통과하면 다양한 주파수의 무지개로 분리되듯, QFT는 양자 상태를 위상이 어긋난 기저 상태들의 중첩으로 분리하여 숨겨진 주기성을 측정으로 가시화합니다.

## 목차

1. [고전적 DFT 복습](#1-고전적-dft-복습)
2. [고전적 DFT에서 양자 FT로](#2-고전적-dft에서-양자-ft로)
3. [QFT 정의와 행렬 구조](#3-qft-정의와-행렬-구조)
4. [QFT 회로](#4-qft-회로)
5. [게이트 복잡도 분석](#5-게이트-복잡도-분석)
6. [역 QFT](#6-역-qft)
7. [양자 위상 추정](#7-양자-위상-추정)
8. [Python 구현](#8-python-구현)
9. [연습 문제](#9-연습-문제)

---

## 1. 고전적 DFT 복습

양자 버전에 들어가기 전에, 고전적 이산 푸리에 변환(Discrete Fourier Transform)을 복습해 봅시다 (자세한 내용은 Signal_Processing L05도 참고하세요).

### 1.1 DFT 정의

$N$개의 복소수 수열 $(x_0, x_1, \ldots, x_{N-1})$이 주어졌을 때, DFT는 다음으로 정의되는 또 다른 수열 $(y_0, y_1, \ldots, y_{N-1})$을 생성합니다:

$$y_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j \, \omega^{jk}$$

여기서 $\omega = e^{2\pi i / N}$은 원시(primitive) $N$번째 단위근(root of unity)입니다.

$\frac{1}{\sqrt{N}}$ 인자는 *유니터리(unitary)* 정규화 관례입니다 (일부 참고문헌은 $\frac{1}{N}$ 또는 인자 없이 사용합니다). 양자 변환은 반드시 유니터리해야 하므로 우리는 유니터리 관례를 사용합니다.

### 1.2 DFT 행렬

DFT는 행렬-벡터 곱셈 $\mathbf{y} = F_N \mathbf{x}$으로 표현할 수 있습니다. 여기서:

$$F_N = \frac{1}{\sqrt{N}} \begin{pmatrix} 1 & 1 & 1 & \cdots & 1 \\ 1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\ 1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2} \end{pmatrix}$$

$k$행, $j$열의 원소는:

$$(F_N)_{kj} = \frac{1}{\sqrt{N}} \omega^{jk} = \frac{1}{\sqrt{N}} e^{2\pi i \, jk / N}$$

### 1.3 주요 성질

- **유니터리성(Unitarity)**: $F_N^{\dagger} F_N = I$, 즉 $F_N^{-1} = F_N^{\dagger}$
- **주기성 검출(Periodicity detection)**: $x_j$의 주기가 $r$이면 (즉 $x_j = x_{j+r}$), $y_k$는 $N/r$의 배수에서 집중됩니다
- **복잡도(Complexity)**: 직접 계산은 $O(N^2)$이고, 고속 푸리에 변환(Fast Fourier Transform, FFT)은 $O(N \log N)$을 달성합니다

주기성 검출 성질이 QFT가 양자 알고리즘에서 강력한 이유입니다 — 고전적으로 지수 시간이 걸릴 숨겨진 주기를 드러냅니다.

### 1.4 Python: 고전적 DFT

```python
import numpy as np

def classical_dft(x):
    """Compute the DFT using the unitary normalization convention.

    Why unitary normalization? Quantum mechanics requires all transformations
    to preserve the norm of state vectors. The 1/sqrt(N) factor ensures
    that the DFT matrix is unitary (F†F = I), making it a valid quantum gate.
    """
    N = len(x)
    # Build the DFT matrix element by element
    # (F_N)_{kj} = (1/sqrt(N)) * exp(2πi * j * k / N)
    j = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    omega_matrix = np.exp(2j * np.pi * j * k / N)
    return omega_matrix @ x / np.sqrt(N)

# Example: DFT of a periodic signal
N = 8
# Signal with period 2: [1, 0, 1, 0, 1, 0, 1, 0]
x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=complex)
y = classical_dft(x)

print("Input signal:", x.real)
print("DFT magnitudes:", np.abs(y).round(4))
# The peak at index 4 = N/2 reveals period r=2 (since N/r = 8/2 = 4)
```

---

## 2. 고전적 DFT에서 양자 FT로

### 2.1 개념적 연결고리

고전적 DFT와 양자 버전을 연결하는 핵심 통찰은 단순합니다: **QFT는 정확히 동일한 수학적 변환을 수행하지만, 양자 상태의 진폭(amplitude)에 적용합니다**.

고전적 경우에는 $N$개의 숫자 벡터를 변환합니다. 양자적 경우에는 $n$큐비트 상태의 $N = 2^n$개의 진폭을 변환합니다. 다음 상태가 있을 때:

$$|\psi\rangle = \sum_{j=0}^{N-1} x_j |j\rangle$$

QFT는 다음을 생성합니다:

$$\text{QFT}|\psi\rangle = \sum_{k=0}^{N-1} y_k |k\rangle$$

여기서 $y_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j \, e^{2\pi i \, jk/N}$는 정확히 DFT 공식입니다.

### 2.2 양자 버전이 특별한 이유

$N = 2^n$개의 숫자에 대한 고전적 DFT는 FFT를 사용해도 $O(N \log N) = O(n \cdot 2^n)$번의 연산이 필요합니다. 양자 버전은 $O(n^2)$개의 게이트만 필요합니다. 이것은 *지수적* 감소입니다.

그러나 중요한 미묘한 점이 있습니다: **$N$개의 진폭 모두를 직접 읽어낼 수 없습니다**. 측정은 상태를 확률 $|y_k|^2$로 하나의 기저 상태 $|k\rangle$으로 붕괴시킵니다. 따라서 QFT는 고전 FFT의 범용 대체제가 아닙니다. QFT의 위력은 푸리에 변환된 분포에서 *샘플링*만 하면 되고, 모든 계수를 읽을 필요가 없는 알고리즘에서 발휘됩니다.

이 샘플링은 주기 찾기(쇼어의 알고리즘), 위상 추정, 숨겨진 부분군 문제(hidden subgroup problem)에 충분합니다 — 이 모두는 푸리에 스펙트럼이 날카로운 구조를 가지는 경우입니다.

---

## 3. QFT 정의와 행렬 구조

### 3.1 형식적 정의

$N = 2^n$개의 기저 상태에 대한 양자 푸리에 변환(Quantum Fourier Transform)은 계산 기저 상태에 대한 작용으로 정의됩니다:

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i \, jk / N} |k\rangle$$

이것은 행렬 원소가 다음과 같은 유니터리 변환 $U_{\text{QFT}}$입니다:

$$(U_{\text{QFT}})_{kj} = \frac{1}{\sqrt{N}} e^{2\pi i \, jk / N}$$

### 3.2 곱 표현

QFT는 회로 구현을 직접 도출할 수 있는 아름다운 곱 표현을 가집니다. 이를 유도하려면 $j$와 $k$를 이진수로 씁니다:

$$j = j_1 j_2 \cdots j_n = j_1 \cdot 2^{n-1} + j_2 \cdot 2^{n-2} + \cdots + j_n \cdot 2^0$$

이진 분수 표기법 $0.j_l j_{l+1} \cdots j_m = j_l/2 + j_{l+1}/4 + \cdots + j_m/2^{m-l+1}$을 사용하면:

$$\text{QFT}|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left( |0\rangle + e^{2\pi i \, 0.j_{n-l+1} \cdots j_n} |1\rangle \right)$$

이 인수분해된 형태는 출력의 각 큐비트가 입력 비트의 일부에만 의존하며, 출력이 *텐서 곱(tensor product)*임을 보여줍니다 — 얽힘(entanglement)이 없습니다! 이 곱 구조가 QFT 회로를 효율적으로 만드는 이유입니다.

### 3.3 명시적 예제: 2큐비트 QFT

$n = 2$, $N = 4$이고 $\omega = e^{2\pi i/4} = i$일 때:

$$U_{\text{QFT}}^{(2)} = \frac{1}{2} \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & i & -1 & -i \\ 1 & -1 & 1 & -1 \\ 1 & -i & -1 & i \end{pmatrix}$$

상태 $|2\rangle = |10\rangle$에 대해 검증해 봅시다:

$$\text{QFT}|2\rangle = \frac{1}{2}(|0\rangle + e^{2\pi i \cdot 2/4}|1\rangle + e^{2\pi i \cdot 4/4}|2\rangle + e^{2\pi i \cdot 6/4}|3\rangle) = \frac{1}{2}(|0\rangle - |1\rangle + |2\rangle - |3\rangle)$$

### 3.4 명시적 예제: 3큐비트 QFT

$n = 3$, $N = 8$이고 $\omega = e^{2\pi i/8} = e^{i\pi/4}$일 때:

$$U_{\text{QFT}}^{(3)} = \frac{1}{\sqrt{8}} \begin{pmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 1 & \omega & \omega^2 & \omega^3 & \omega^4 & \omega^5 & \omega^6 & \omega^7 \\ 1 & \omega^2 & \omega^4 & \omega^6 & 1 & \omega^2 & \omega^4 & \omega^6 \\ \vdots & & & & & & & \vdots \\ 1 & \omega^7 & \omega^6 & \omega^5 & \omega^4 & \omega^3 & \omega^2 & \omega \end{pmatrix}$$

행렬은 매우 구조적인 패턴을 가집니다: $k$행, $j$열에는 $\omega^{jk}$가 들어갑니다. 이 구조가 $O(n^2)$개의 게이트로 인수분해할 수 있게 해줍니다.

### 3.5 Python: QFT 행렬 구성

```python
import numpy as np

def qft_matrix(n):
    """Construct the QFT matrix for n qubits.

    Why build the full matrix? For small n, the full matrix lets us verify
    our circuit implementation against the mathematical definition. For
    large n, only the circuit approach is feasible (the matrix is 2^n × 2^n).
    """
    N = 2**n
    omega = np.exp(2j * np.pi / N)

    # Build using outer product of indices: (F_N)_{kj} = omega^{jk} / sqrt(N)
    indices = np.arange(N)
    # indices[:, None] * indices[None, :] creates the jk product matrix
    return np.array([[omega**(j*k) for j in range(N)] for k in range(N)]) / np.sqrt(N)

# Verify unitarity: F†F should equal I
n = 3
F = qft_matrix(n)
print("Unitarity check (should be ~identity):")
print(np.round(F.conj().T @ F, 10).real)

# Apply QFT to |5⟩ = |101⟩ for 3 qubits
state = np.zeros(8, dtype=complex)
state[5] = 1.0  # |101⟩
result = F @ state
print("\nQFT|5⟩ amplitudes:")
for k in range(8):
    print(f"  |{k:03b}⟩: {result[k]:.4f}  (magnitude: {abs(result[k]):.4f})")
```

---

## 4. QFT 회로

### 4.1 구성 요소

QFT 회로는 두 종류의 게이트를 사용합니다:

**아다마르 게이트(Hadamard gate)** $H$:

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$|0\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$이고 $|1\rangle \to \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$로 매핑합니다.

**제어 회전 게이트(Controlled rotation gate)** $CR_k$ (controlled-$R_k$):

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i / 2^k} \end{pmatrix}$$

제어 버전 $CR_k$는 제어 큐비트가 $|1\rangle$일 때만 타겟 큐비트에 $R_k$를 적용합니다. $R_1 = Z$ (파울리-Z 게이트)이고 $R_2 = S$ (위상 게이트)입니다.

### 4.2 회로 구성

$n$큐비트 QFT 회로는 다음과 같이 진행됩니다:

**1단계**: 큐비트 1에 $H$를 적용한 후, $CR_2$ (큐비트 2가 제어, 큐비트 1이 타겟), $CR_3$ (큐비트 3이 제어, 큐비트 1이 타겟), ..., $CR_n$ (큐비트 $n$이 제어, 큐비트 1이 타겟)을 순서대로 적용합니다.

이 단계 후 큐비트 1은 다음 상태가 됩니다:

$$\frac{1}{\sqrt{2}} \left( |0\rangle + e^{2\pi i \, 0.j_1 j_2 \cdots j_n} |1\rangle \right)$$

**2단계**: 큐비트 2에 $H$를 적용한 후, $CR_2$ (큐비트 3이 제어), ..., $CR_{n-1}$ (큐비트 $n$이 제어)를 적용합니다.

**3단계부터 n단계**: 패턴을 계속하되, 각 단계마다 $H$를 적용하고 제어 회전의 수가 점점 줄어듭니다.

**마지막 단계**: SWAP 게이트를 적용하여 큐비트 순서를 역전시킵니다 (곱 표현은 표준 관례와 반대 순서로 큐비트가 배열되어 있습니다).

### 4.3 회로 다이어그램 (3큐비트 QFT)

```
q₁ ─── H ─── CR₂ ─── CR₃ ─── × ───────────────────
              │        │       │
q₂ ─────────●──── H ─ CR₂ ── │ ── × ───────────────
                       │       │    │
q₃ ──────────────●────●────── × ── × ── H ──────────
```

여기서 `●`는 제어 큐비트를, `×`는 SWAP 연산을 나타냅니다.

### 4.4 이 회로가 작동하는 이유

$|j\rangle = |j_1 j_2 j_3\rangle$에 대해 회로를 단계별로 추적해 봅시다.

**큐비트 1에 H 적용 후**: 큐비트 1은 $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1}|1\rangle)$이 됩니다.
- $j_1 = 0$이면: 위상이 $e^0 = 1$이므로 $(|0\rangle + |1\rangle)/\sqrt{2}$
- $j_1 = 1$이면: 위상이 $e^{i\pi} = -1$이므로 $(|0\rangle - |1\rangle)/\sqrt{2}$

**$CR_2$ (큐비트 2가 제어) 적용 후**: $j_2 = 1$이면, 큐비트 1은 추가 위상 $e^{2\pi i / 4}$를 얻어 $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1 j_2}|1\rangle)$이 됩니다.

**$CR_3$ (큐비트 3이 제어) 적용 후**: $j_3 = 1$이면, 또 다른 위상이 추가되어 $\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_1 j_2 j_3}|1\rangle)$이 됩니다.

이는 곱 표현의 첫 번째 인자와 일치합니다. 동일한 논리가 이후 단계에서 큐비트 2와 3에도 적용됩니다. 마지막의 SWAP은 표준 관례에 맞게 큐비트 순서를 역전시킵니다.

### 4.5 Python: QFT 회로 시뮬레이션

```python
import numpy as np

def controlled_rotation(n_qubits, control, target, k):
    """Build a controlled-R_k gate for an n-qubit system.

    Why build the full matrix? For small systems (n ≤ 6), the full matrix
    approach is conceptually clear and easy to verify. For larger systems,
    one would use sparse representations or gate-by-gate state updates.
    """
    N = 2**n_qubits
    gate = np.eye(N, dtype=complex)
    phase = np.exp(2j * np.pi / 2**k)

    # Apply phase only when both control and target are |1⟩
    for state in range(N):
        # Check if control bit is 1 AND target bit is 1
        if (state >> (n_qubits - 1 - control)) & 1 and \
           (state >> (n_qubits - 1 - target)) & 1:
            gate[state, state] = phase
    return gate

def hadamard_on_qubit(n_qubits, qubit):
    """Build Hadamard gate acting on specified qubit in n-qubit system."""
    # Construct using tensor products: I ⊗ ... ⊗ H ⊗ ... ⊗ I
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I = np.eye(2)

    gate = np.array([[1]])
    for q in range(n_qubits):
        gate = np.kron(gate, H if q == qubit else I)
    return gate

def swap_gate(n_qubits, q1, q2):
    """Build SWAP gate between two qubits."""
    N = 2**n_qubits
    gate = np.zeros((N, N), dtype=complex)
    for state in range(N):
        bit1 = (state >> (n_qubits - 1 - q1)) & 1
        bit2 = (state >> (n_qubits - 1 - q2)) & 1
        if bit1 != bit2:
            # Swap the two bits
            new_state = state ^ (1 << (n_qubits - 1 - q1)) ^ (1 << (n_qubits - 1 - q2))
            gate[new_state, state] = 1
        else:
            gate[state, state] = 1
    return gate

def qft_circuit(n):
    """Build the QFT as a product of gates (circuit simulation).

    This constructs the QFT matrix by multiplying individual gates in
    the order they appear in the circuit. The result should match
    qft_matrix(n) from the direct definition.
    """
    N = 2**n
    circuit = np.eye(N, dtype=complex)

    # Main QFT gates
    for target in range(n):
        # Hadamard on target qubit
        circuit = hadamard_on_qubit(n, target) @ circuit

        # Controlled rotations: CR_2, CR_3, ..., CR_{n-target}
        for control in range(target + 1, n):
            k = control - target + 1  # R_k rotation angle
            circuit = controlled_rotation(n, control, target, k) @ circuit

    # SWAP gates to reverse qubit order
    for i in range(n // 2):
        circuit = swap_gate(n, i, n - 1 - i) @ circuit

    return circuit

# Verify: circuit QFT should match the matrix definition
n = 3
F_matrix = qft_matrix(n)  # From the earlier function
F_circuit = qft_circuit(n)

print("Max difference between matrix and circuit QFT:",
      np.max(np.abs(F_matrix - F_circuit)))

# Test on a specific state: |5⟩ = |101⟩
state = np.zeros(8, dtype=complex)
state[5] = 1.0
result = F_circuit @ state
print("\nQFT|5⟩ via circuit:")
for k in range(8):
    mag = abs(result[k])
    phase = np.angle(result[k]) / np.pi
    if mag > 1e-10:
        print(f"  |{k:03b}⟩: magnitude={mag:.4f}, phase={phase:.4f}π")
```

---

## 5. 게이트 복잡도 분석

### 5.1 게이트 수 계산

$n$큐비트 QFT 회로는 다음을 포함합니다:

- **아다마르 게이트(Hadamard gate)**: 총 $n$개 (큐비트당 하나)
- **제어 회전 게이트(Controlled rotation gate)**: 총 $(n-1) + (n-2) + \cdots + 1 + 0 = \frac{n(n-1)}{2}$개
- **SWAP 게이트**: 총 $\lfloor n/2 \rfloor$개 (각각 3개의 CNOT으로 분해 가능)

**총 게이트 수**: $n + \frac{n(n-1)}{2} + \lfloor n/2 \rfloor = O(n^2)$

### 5.2 고전과의 비교

| 방법 | 입력 크기 | 연산 수 |
|--------|-----------|-----------|
| 고전적 DFT (직접) | $N = 2^n$개의 숫자 | $O(N^2) = O(4^n)$ |
| 고전적 FFT | $N = 2^n$개의 숫자 | $O(N \log N) = O(n \cdot 2^n)$ |
| 양자 QFT | $n$큐비트 ($2^n$개의 진폭) | $O(n^2)$개의 게이트 |

QFT는 고전적 FFT에 비해 *지수적* 속도 향상을 달성합니다. 그러나 이 비교에는 미묘한 점이 있습니다:

- 고전적 FFT는 명시적으로 주어진 $N$개의 숫자를 변환합니다
- QFT는 $n$큐비트에 *암묵적으로* 인코딩된 $N$개의 진폭을 변환합니다
- 임의의 고전 데이터를 양자 상태로 효율적으로 로드할 수 없습니다 (이것이 "상태 준비(state preparation)" 병목입니다)
- $N$개의 진폭 모두를 읽어낼 수 없습니다 (측정은 하나의 샘플을 줍니다)

QFT의 위력은 입력 상태가 양자 연산에 의해 준비될 때 (고전 데이터 로딩이 아닌) 더 큰 양자 알고리즘 내의 *서브루틴*으로 사용될 때 실현됩니다.

### 5.3 근사 QFT(Approximate QFT)

실용적인 구현에서는, $k \gg 1$인 먼 거리의 제어 회전 $CR_k$가 아주 작은 위상 $e^{2\pi i / 2^k} \approx 1$을 적용합니다. 어떤 컷오프 $m$에 대해 $k > m$인 모든 $CR_k$를 생략하여 회로를 *절단*할 수 있으며, 이로 인해 **근사 QFT(approximate QFT)**가 생성됩니다:

- 게이트 수: $O(n^2)$ 대신 $O(nm)$
- 오류: 연산자 노름(operator norm)에서 $O(n \cdot 2^{-m})$

$m = O(\log n)$으로 설정하면 $O(n \log n)$개의 게이트만으로 $O(n^{-c})$의 오류를 달성합니다 — FFT 스케일링과 일치합니다!

---

## 6. 역 QFT

### 6.1 정의

QFT는 유니터리이므로, 그 역변환은 단순히 켤레 전치(conjugate transpose)입니다:

$$\text{QFT}^{-1} = \text{QFT}^{\dagger}$$

상태에 적용하면:

$$\text{QFT}^{-1}|k\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{-2\pi i \, jk / N} |j\rangle$$

순방향 QFT와의 유일한 차이는 지수의 부호입니다: $e^{+2\pi i jk/N}$ 대신 $e^{-2\pi i jk/N}$.

### 6.2 역 QFT 회로

역 QFT 회로를 얻기 위해 다음을 수행합니다:

1. QFT 회로의 모든 게이트 순서를 역전합니다
2. 각 $R_k$를 $R_k^{\dagger}$로 교체합니다 (즉, 위상을 음수로: $e^{2\pi i/2^k} \to e^{-2\pi i/2^k}$)

이것은 양자 회로의 일반적인 성질입니다: 게이트 $U_1 U_2 \cdots U_m$으로 구성된 유니터리를 역전하려면 $U_m^{\dagger} \cdots U_2^{\dagger} U_1^{\dagger}$를 적용합니다.

### 6.3 역 QFT가 필요한 때

역 QFT는 다음에서 등장합니다:

- **양자 위상 추정(Quantum Phase Estimation)**: 역 QFT가 제어-$U$ 연산 후 레지스터에서 위상을 추출합니다 (7절 참고)
- **쇼어의 알고리즘(Shor's algorithm)**: 역 QFT를 사용해 주기를 읽어냅니다 (레슨 10 참고)
- **양자 시뮬레이션(Quantum simulation)**: 운동량 기저에서 위치 기저로 변환할 때

```python
import numpy as np

def inverse_qft_matrix(n):
    """Construct the inverse QFT matrix.

    Why conjugate transpose? The QFT is unitary, so its inverse equals
    its conjugate transpose. This is the quantum analog of the inverse DFT.
    """
    return qft_matrix(n).conj().T

# Verify: QFT followed by inverse QFT should be identity
n = 3
F = qft_matrix(n)
F_inv = inverse_qft_matrix(n)
print("QFT · QFT⁻¹ ≈ I:", np.allclose(F @ F_inv, np.eye(2**n)))

# Round-trip test: start with |3⟩, apply QFT, then inverse QFT
state = np.zeros(8, dtype=complex)
state[3] = 1.0
recovered = F_inv @ (F @ state)
print("Round-trip recovery of |3⟩:", np.argmax(np.abs(recovered)))
```

---

## 7. 양자 위상 추정

### 7.1 문제

양자 위상 추정(Quantum Phase Estimation, QPE)은 QFT의 가장 중요한 응용 중 하나입니다. 다음이 주어졌을 때:

- 유니터리 연산자 $U$
- $U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$를 만족하는 고유 상태(eigenstate) $|\psi\rangle$

QPE는 $n$개의 보조(ancilla) 큐비트를 사용해 $n$비트 정밀도로 위상 $\theta \in [0, 1)$을 추정합니다.

### 7.2 QPE 회로

QPE 알고리즘은 두 개의 레지스터를 사용합니다:

- **카운팅 레지스터(Counting register)**: $|0\rangle^{\otimes n}$으로 초기화된 $n$큐비트
- **고유 상태 레지스터(Eigenstate register)**: $|\psi\rangle$으로 초기화됨

**1단계**: 카운팅 레지스터에 $H^{\otimes n}$을 적용하여 균등한 중첩 상태 생성.

**2단계**: 제어-$U^{2^j}$ 연산 적용. $j$번째 카운팅 큐비트가 $|\psi\rangle$에 작용하는 $U^{2^j}$를 제어합니다.

$U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$이므로, $U^{2^j}|\psi\rangle = e^{2\pi i \cdot 2^j \theta}|\psi\rangle$입니다. 모든 제어-$U^{2^j}$ 연산 후 상태는 다음이 됩니다:

$$\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{2\pi i k \theta} |k\rangle \otimes |\psi\rangle$$

**3단계**: 카운팅 레지스터에 **역 QFT(inverse QFT)**를 적용합니다.

$\theta = m/2^n$인 정수 $m$이 존재할 때 (위상이 $n$비트로 정확히 표현 가능), 역 QFT는 카운팅 레지스터를 정확히 $|m\rangle$으로 매핑합니다. 카운팅 레지스터를 측정하면 $m$을 얻고, 여기서 $\theta = m/2^n$을 계산합니다.

$\theta$가 정확히 표현 불가능한 경우, 측정은 높은 확률로 가장 가까운 $n$비트 근사값을 냅니다. 보조 큐비트를 더 추가할수록 정밀도가 높아집니다.

### 7.3 회로 다이어그램

```
|0⟩ ─── H ─── ────────── ────── ────── ┌──────────┐
|0⟩ ─── H ─── ────────── ────── U²    │          │
|0⟩ ─── H ─── ────────── U⁴   ────── │ QFT⁻¹   │ ──── Measure
|0⟩ ─── H ─── U⁸        ────── ────── │          │
                                        └──────────┘
|ψ⟩ ─── ───── target ── target  target ───────────────
```

### 7.4 QPE 정확도

$n$큐비트 카운팅 레지스터에 대해:

- $\theta$가 $n$비트로 정확히 표현 가능하면: QPE가 확률 1로 성공
- 그렇지 않으면: $|\theta - \tilde{\theta}| \leq 2^{-n}$을 만족하는 최선의 $n$비트 근사 $\tilde{\theta}$를 얻을 확률이 최소 $4/\pi^2 \approx 0.405$

$O(\log(1/\epsilon))$개의 추가 큐비트를 더하면 성공 확률이 $1 - \epsilon$으로 높아집니다.

### 7.5 Python: 위상 추정 시뮬레이션

```python
import numpy as np

def phase_estimation(unitary, eigenstate, n_counting):
    """Simulate Quantum Phase Estimation.

    Why simulate the full algorithm? This demonstrates the mathematical
    structure: controlled-U operations create phase-encoded superpositions,
    and the inverse QFT extracts the phase as a binary fraction.

    Args:
        unitary: d×d unitary matrix
        eigenstate: d-dimensional eigenvector of unitary
        n_counting: number of counting qubits (precision bits)

    Returns:
        probabilities: probability distribution over counting register states
        estimated_phase: most likely phase estimate
    """
    d = len(eigenstate)
    N = 2**n_counting

    # Step 1: Create equal superposition in counting register
    # After H⊗n, counting register is (1/√N) Σ|k⟩
    # After controlled-U operations, state is:
    # (1/√N) Σ_k e^{2πi·k·θ} |k⟩ ⊗ |ψ⟩

    # Compute the eigenvalue phase
    # U|ψ⟩ = e^{2πiθ}|ψ⟩, so θ = angle(⟨ψ|U|ψ⟩) / (2π)
    eigenvalue = np.dot(eigenstate.conj(), unitary @ eigenstate)
    theta = np.angle(eigenvalue) / (2 * np.pi)
    if theta < 0:
        theta += 1  # Ensure θ ∈ [0, 1)

    # Step 2: The counting register state before inverse QFT
    counting_state = np.array([np.exp(2j * np.pi * k * theta) for k in range(N)]) / np.sqrt(N)

    # Step 3: Apply inverse QFT to counting register
    F_inv = np.conj(qft_matrix(n_counting)).T
    result = F_inv @ counting_state

    # Measurement probabilities
    probs = np.abs(result)**2

    best_k = np.argmax(probs)
    estimated_phase = best_k / N

    return probs, estimated_phase, theta

# Example: Estimate the phase of a rotation gate R_z(2π·0.375)
# The eigenvalue should be e^{2πi·0.375}, so θ = 0.375 = 3/8
theta_true = 0.375
Rz = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]])
eigenstate = np.array([0, 1], dtype=complex)  # |1⟩ is an eigenstate of Rz

n_counting = 4
probs, estimated, true_theta = phase_estimation(Rz, eigenstate, n_counting)

print(f"True phase: θ = {true_theta:.6f}")
print(f"Estimated phase: θ = {estimated:.6f}")
print(f"Counting register probabilities:")
for k in range(2**n_counting):
    if probs[k] > 0.01:
        print(f"  |{k:0{n_counting}b}⟩ = |{k}⟩: {probs[k]:.4f}  → θ = {k/2**n_counting:.4f}")
```

---

## 8. Python 구현

### 8.1 시각화가 포함된 완전한 QFT

```python
import numpy as np

def qft_matrix(n):
    """Build the full QFT matrix for n qubits."""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for j in range(N)] for k in range(N)]) / np.sqrt(N)

def demonstrate_periodicity_detection():
    """Show how QFT detects hidden periodicity.

    This is the core reason QFT matters: given a quantum state with
    amplitudes that repeat with some period r, the QFT concentrates
    the probability at multiples of N/r. This is the foundation of
    Shor's algorithm.
    """
    n = 4  # 4 qubits → 16-dimensional space
    N = 2**n

    print("=" * 60)
    print("QFT Periodicity Detection Demo")
    print("=" * 60)

    for period in [2, 4, 8]:
        # Create a state with the given period: nonzero at 0, r, 2r, ...
        state = np.zeros(N, dtype=complex)
        num_peaks = N // period
        for i in range(num_peaks):
            state[i * period] = 1.0
        state /= np.linalg.norm(state)  # Normalize

        # Apply QFT
        F = qft_matrix(n)
        result = F @ state
        probs = np.abs(result)**2

        print(f"\nPeriod r = {period}:")
        print(f"  Input state nonzero at: {[i*period for i in range(num_peaks)]}")
        print(f"  QFT peaks at (prob > 0.01):")
        for k in range(N):
            if probs[k] > 0.01:
                print(f"    k = {k} (N/r = {N}/{period} = {N//period}, "
                      f"k is {'a' if k % (N//period) == 0 else 'NOT a'} multiple)")

def demonstrate_phase_estimation_accuracy():
    """Show how QPE accuracy improves with more counting qubits.

    Key insight: adding one counting qubit doubles the precision of
    our phase estimate. This is why QPE can achieve exponential precision
    with only linear resources.
    """
    theta_true = 1/3  # θ = 1/3, not exactly representable in binary
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta_true)]])
    eigenstate = np.array([0, 1], dtype=complex)

    print("\n" + "=" * 60)
    print(f"QPE Accuracy vs Counting Qubits (θ = 1/3 ≈ 0.3333...)")
    print("=" * 60)

    for n_count in range(3, 9):
        N = 2**n_count
        # Simulate QPE
        counting = np.array([np.exp(2j * np.pi * k * theta_true)
                            for k in range(N)]) / np.sqrt(N)
        F_inv = np.conj(qft_matrix(n_count)).T
        result = F_inv @ counting
        probs = np.abs(result)**2

        best_k = np.argmax(probs)
        estimated = best_k / N
        error = abs(estimated - theta_true)

        print(f"  n = {n_count}: best estimate = {estimated:.6f}, "
              f"error = {error:.6f}, prob = {probs[best_k]:.4f}")

# Run demonstrations
demonstrate_periodicity_detection()
demonstrate_phase_estimation_accuracy()
```

### 8.2 게이트별 QFT 회로

```python
import numpy as np

def qft_step_by_step(state_vector, n):
    """Apply QFT gate by gate, printing intermediate states.

    Why trace through step by step? Understanding the QFT circuit at the
    gate level builds intuition for how the product representation arises.
    Each gate adds one more bit to the binary fraction in the phase.
    """
    state = state_vector.copy()
    N = 2**n

    def apply_single_qubit_gate(state, gate, target, n):
        """Apply a single-qubit gate to the target qubit."""
        I = np.eye(2)
        full_gate = np.array([[1]])
        for q in range(n):
            full_gate = np.kron(full_gate, gate if q == target else I)
        return full_gate @ state

    def apply_controlled_phase(state, control, target, phase, n):
        """Apply controlled phase gate."""
        N = 2**n
        new_state = state.copy()
        for s in range(N):
            c_bit = (s >> (n - 1 - control)) & 1
            t_bit = (s >> (n - 1 - target)) & 1
            if c_bit == 1 and t_bit == 1:
                new_state[s] *= phase
        return new_state

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    print(f"Initial state: {np.array2string(state, precision=3)}")

    for target in range(n):
        # Hadamard
        state = apply_single_qubit_gate(state, H, target, n)
        print(f"After H on q{target}: ", end="")
        # Show nonzero amplitudes
        nonzero = [(k, state[k]) for k in range(N) if abs(state[k]) > 1e-10]
        print(", ".join(f"|{k:0{n}b}⟩:{v:.3f}" for k, v in nonzero[:4]),
              "..." if len(nonzero) > 4 else "")

        # Controlled rotations
        for control in range(target + 1, n):
            k = control - target + 1
            phase = np.exp(2j * np.pi / 2**k)
            state = apply_controlled_phase(state, control, target, phase, n)
            print(f"After CR_{k}(q{control}→q{target}): applied phase 2π/2^{k}")

    # Swap qubits
    for i in range(n // 2):
        j = n - 1 - i
        # Swap qubits i and j
        new_state = np.zeros_like(state)
        for s in range(N):
            bi = (s >> (n-1-i)) & 1
            bj = (s >> (n-1-j)) & 1
            if bi != bj:
                ns = s ^ (1 << (n-1-i)) ^ (1 << (n-1-j))
            else:
                ns = s
            new_state[ns] = state[s]
        state = new_state
        print(f"After SWAP(q{i},q{j})")

    return state

# Trace QFT|6⟩ = QFT|110⟩ for 3 qubits
n = 3
state = np.zeros(2**n, dtype=complex)
state[6] = 1.0  # |110⟩
print("=" * 60)
print("Step-by-step QFT on |110⟩ (3 qubits)")
print("=" * 60)
result = qft_step_by_step(state, n)
```

---

## 9. 연습 문제

### 연습 1: 손으로 계산하는 QFT (2큐비트)

다음 방법으로 2큐비트 QFT에 대해 $\text{QFT}|3\rangle$을 계산하세요:
(a) QFT 행렬 $F_4$를 벡터 $(0, 0, 0, 1)^T$에 직접 적용하는 방법
(b) 곱 표현 사용: $|3\rangle = |11\rangle$을 쓰고 $\frac{1}{2}(|0\rangle + e^{2\pi i \cdot 0.j_2}|1\rangle) \otimes (|0\rangle + e^{2\pi i \cdot 0.j_1 j_2}|1\rangle)$을 전개하는 방법
(c) 두 방법의 결과가 일치하는지 확인하세요.

### 연습 2: QFT를 이용한 주기 찾기

양자 오라클(quantum oracle)이 4큐비트 레지스터에 상태 $|\psi\rangle = \frac{1}{2}(|0\rangle + |3\rangle + |6\rangle + |9\rangle)$을 준비한다고 가정합니다 (주기 $r = 3$, $N = 16$).

(a) $\text{QFT}|\psi\rangle$를 명시적으로 계산하세요 (또는 Python 시뮬레이션 사용).
(b) 측정 확률은 어떻게 됩니까? 피크(peak)는 어디에 있습니까?
(c) 측정 결과에서 주기 $r = 3$을 어떻게 추출하겠습니까?
(d) $r = 2, 4, 8$인 경우보다 왜 더 미묘합니까? (힌트: $N/r$이 정수가 아닙니다.)

### 연습 3: QPE 정밀도

$\phi = 2\pi \cdot 0.1101_2 = 2\pi \cdot (13/16)$ 이고 고유 상태가 $|1\rangle$인 게이트 $R_z(\phi)$에 QPE를 적용하는 경우를 고려하세요.

(a) $\theta = 13/16$을 정확히 추정하는 데 필요한 최소 카운팅 큐비트 수는 얼마입니까?
(b) 4, 5, 6개의 카운팅 큐비트로 QPE를 시뮬레이션하세요. 확률 분포에 어떤 변화가 생깁니까?
(c) 이제 $\theta = 0.3$ (이진수로 정확히 표현 불가)으로 변경하세요. $n = 4, 6, 8, 10$개의 카운팅 큐비트로 시뮬레이션하고 오류 대 $n$ 그래프를 그리세요.

### 연습 4: 근사 QFT(Approximate QFT)

컷오프 파라미터 $m$에 대해 $k > m$인 모든 $CR_k$ 게이트를 생략하도록 회로를 수정하여 근사 QFT를 구현하세요.

(a) $n = 6$큐비트에서 $m = 2, 3, 4, 5$에 대한 연산자 노름 오류 $\|U_{\text{QFT}} - U_{\text{approx}}\|$를 계산하세요.
(b) 게이트 수가 $m$에 따라 어떻게 스케일됩니까?
(c) 오류가 무시 가능해지는 (예: $< 10^{-4}$) $m$은 얼마입니까?

### 연습 5: 역 QFT 응용

상태 $|\phi\rangle = \frac{1}{\sqrt{8}} \sum_{k=0}^{7} e^{2\pi i \cdot 5k/8} |k\rangle$ (주파수 5의 푸리에 모드)가 주어졌을 때:

(a) 역 QFT를 적용하세요. 어떤 상태를 얻습니까?
(b) 행렬 방법과 회로 방법 모두로 검증하세요.
(c) 일반화: 임의의 정수 $m$에 대해 $\text{QFT}^{-1} \left(\frac{1}{\sqrt{N}} \sum_k e^{2\pi i mk/N} |k\rangle\right)$는 무엇입니까?

---

[← 이전: 그로버 검색 알고리즘](08_Grovers_Search.md) | [다음: 쇼어의 소인수분해 알고리즘 →](10_Shors_Algorithm.md)

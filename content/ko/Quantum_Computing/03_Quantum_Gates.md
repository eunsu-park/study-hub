# 레슨 3: 양자 게이트(Quantum Gates)

[<- 이전: 큐비트와 블로흐 구(Qubits and the Bloch Sphere)](02_Qubits_and_Bloch_Sphere.md) | [다음: 양자 회로(Quantum Circuits) ->](04_Quantum_Circuits.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 단일 큐비트 게이트(X, Y, Z, H, S, T, $R_z(\theta)$)를 유니터리 행렬로 기술하고 블로흐 구 위에서의 동작을 시각화한다
2. 행렬-벡터 곱셈을 사용하여 상태 벡터에 양자 게이트를 적용한다
3. 제어 게이트(CNOT, CZ, controlled-U)와 그것이 얽힘(entanglement)을 생성하는 역할을 설명한다
4. 기본 게이트로부터 SWAP 같은 다중 큐비트 게이트를 구성한다
5. 어떤 게이트 집합이 "보편적(universal)"인지를 정의하고, $\{H, T, \text{CNOT}\}$으로 충분한 이유를 설명한다
6. 임의의 단일 큐비트 게이트를 기본 게이트 시퀀스로 분해한다
7. 주요 게이트를 모두 numpy 행렬로 구현하고 양자 상태 벡터에 적용한다

---

양자 게이트는 양자 컴퓨팅의 기본 연산으로, 고전적인 논리 게이트(AND, OR, NOT)와 유사하지만 결정적인 차이가 있습니다. 고전 게이트는 비가역적(AND 게이트의 출력만으로는 입력을 고유하게 결정할 수 없음)인 반면, 양자 게이트는 항상 *가역적(reversible)*입니다 -- 모든 게이트는 잘 정의된 역원을 가진 유니터리 변환(unitary transformation)입니다. 이 가역성은 양자역학이 부과하는 근본적인 제약이며, 양자 회로 설계의 전체 패러다임을 형성합니다.

이 레슨에서는 단일 큐비트 회전부터 2큐비트 얽힘 연산까지, 표준 양자 게이트의 완전한 도구 모음을 구축합니다. 레슨을 마칠 때쯤에는 왜 소수의 게이트 집합으로 임의의 양자 계산을 근사할 수 있는지 이해하게 될 것입니다 -- NAND 게이트 하나만으로 모든 고전 회로를 구성할 수 있는 것의 양자적 유사물입니다.

> **비유:** 양자 게이트는 빛의 편광 필터(polarizing filter)와 같습니다 -- 각각은 특정 방식으로 양자 상태를 변환하며, 이들을 조합하면 복잡한 변환이 만들어집니다. 편광 필터의 시퀀스가 빛의 편광을 원하는 각도로 회전시킬 수 있듯, 양자 게이트의 시퀀스는 임의의 양자 상태를 다른 임의의 상태로 변환할 수 있습니다.

## 목차

1. [유니터리성: 근본적인 제약](#1-유니터리성-근본적인-제약)
2. [파울리 게이트: X, Y, Z](#2-파울리-게이트-x-y-z)
3. [아다마르 게이트](#3-아다마르-게이트)
4. [위상 게이트: S와 T](#4-위상-게이트-s와-t)
5. [회전 게이트](#5-회전-게이트)
6. [2큐비트 게이트](#6-2큐비트-게이트)
7. [보편 게이트 집합](#7-보편-게이트-집합)
8. [게이트 분해](#8-게이트-분해)
9. [연습 문제](#9-연습-문제)

---

## 1. 유니터리성: 근본적인 제약

### 1.1 왜 유니터리인가?

모든 양자 게이트는 다음을 만족하는 **유니터리 행렬(unitary matrix)** $U$로 표현됩니다:

$$U^\dagger U = UU^\dagger = I$$

여기서 $U^\dagger = (U^*)^T$는 $U$의 켤레 전치(conjugate transpose, adjoint)입니다.

이 제약은 두 가지 물리적 요구사항에서 비롯됩니다:
1. **정규화 보존(Normalization preservation)**: 게이트 적용 전 $\langle\psi|\psi\rangle = 1$이면, 적용 후에도 $\langle\psi|U^\dagger U|\psi\rangle = \langle\psi|\psi\rangle = 1$입니다.
2. **가역성(Reversibility)**: $U$의 역원은 $U^\dagger$이며, 이것도 유니터리입니다. 모든 양자 연산은 되돌릴 수 있습니다.

### 1.2 유니터리 행렬의 성질

$n \times n$ 유니터리 행렬의 성질:

| 성질 | 의미 |
|----------|---------|
| $U^\dagger U = I$ | 열벡터들이 정규직교 집합을 이룸 |
| $\|U\|_\text{op} = 1$ | 임의의 벡터의 길이를 변화시키지 않음 |
| $\|\det(U)\| = 1$ | 행렬식의 크기가 1 |
| 고유값 $= e^{i\lambda}$ | 모든 고유값이 단위원 위에 위치 |

### 1.3 매개변수 개수는?

$n \times n$ 유니터리 행렬은 $n^2$개의 실수 매개변수를 가집니다. 양자 컴퓨팅에서:
- 단일 큐비트 게이트 ($n=2$): 4개 매개변수 (단, 하나는 전역 위상이므로 실질적으로 3개)
- 2큐비트 게이트 ($n=4$): 16개 매개변수
- $k$큐비트 게이트 ($n=2^k$): $4^k$개 매개변수

```python
import numpy as np

def is_unitary(M, tol=1e-10):
    """
    Check if a matrix is unitary: U^dagger @ U = I.

    Why check this? In quantum computing, every gate MUST be unitary.
    This is the fundamental constraint from quantum mechanics. A non-unitary
    operation would either create or destroy probability, which is unphysical.
    """
    n = M.shape[0]
    product = M.conj().T @ M
    return np.allclose(product, np.eye(n), atol=tol)

def apply_gate(gate, state):
    """
    Apply a quantum gate (matrix) to a state vector.

    Why matrix-vector multiplication? Quantum mechanics says time evolution
    is linear, so the map from input state to output state is a linear
    transformation -- exactly what matrices do.
    """
    new_state = gate @ state
    return new_state

# Example: verify that a random unitary is indeed unitary
# Why generate random unitaries this way? The QR decomposition of a random
# complex matrix gives a uniformly distributed (Haar random) unitary matrix.
rng = np.random.default_rng(42)
random_matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
Q, R = np.linalg.qr(random_matrix)

print("=== Unitarity Check ===\n")
print(f"Random Q from QR decomposition:\n{Q}")
print(f"Is unitary? {is_unitary(Q)}")
print(f"Q^dagger @ Q =\n{Q.conj().T @ Q}")
print(f"\nDeterminant: {np.linalg.det(Q):.4f}")
print(f"|det| = {abs(np.linalg.det(Q)):.6f}")

# Non-unitary matrix for comparison
non_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
print(f"\nNon-unitary matrix [[1,1],[0,1]]:")
print(f"Is unitary? {is_unitary(non_unitary)}")
```

---

## 2. 파울리 게이트: X, Y, Z

세 파울리 행렬은 기본적인 단일 큐비트 게이트입니다. 이들은 블로흐 구의 $x$, $y$, $z$ 축을 중심으로 한 180도 회전에 해당합니다.

### 2.1 파울리-X (NOT 게이트)

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**동작**: 고전적인 NOT 게이트처럼 $|0\rangle \leftrightarrow |1\rangle$을 뒤집습니다.

$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

**블로흐 구**: $x$축을 중심으로 한 180도 회전.

### 2.2 파울리-Y

$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

**동작**: 위상과 함께 뒤집습니다.

$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

**블로흐 구**: $y$축을 중심으로 한 180도 회전.

### 2.3 파울리-Z (위상 뒤집기)

$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**동작**: $|0\rangle$은 변화 없이 두고, $|1\rangle$의 부호를 뒤집습니다.

$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$$

**블로흐 구**: $z$축을 중심으로 한 180도 회전.

### 2.4 파울리 대수

파울리 행렬은 중요한 대수적 관계를 만족합니다:

$$X^2 = Y^2 = Z^2 = I$$

$$XY = iZ, \quad YZ = iX, \quad ZX = iY$$

$$\{X, Y\} = XY + YX = 0 \quad \text{(반교환 관계)}$$

이 관계들은 양자 컴퓨팅의 증명과 간소화에서 매우 자주 사용됩니다.

```python
import numpy as np

# Pauli gates

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

print("=== Pauli Gate Actions ===\n")

for name, gate in [("X", X), ("Y", Y), ("Z", Z)]:
    print(f"{name} gate:")
    print(f"  {name}|0> = {gate @ ket_0}")
    print(f"  {name}|1> = {gate @ ket_1}")
    print(f"  {name}|+> = {gate @ ket_plus}")
    print(f"  Is unitary? {is_unitary(gate)}")
    print()

# Verify Pauli algebra
print("=== Pauli Algebra ===\n")
print(f"X^2 = I? {np.allclose(X @ X, I)}")
print(f"Y^2 = I? {np.allclose(Y @ Y, I)}")
print(f"Z^2 = I? {np.allclose(Z @ Z, I)}")
print(f"XY = iZ? {np.allclose(X @ Y, 1j * Z)}")
print(f"YZ = iX? {np.allclose(Y @ Z, 1j * X)}")
print(f"ZX = iY? {np.allclose(Z @ X, 1j * Y)}")

# Anticommutation: {X,Y} = XY + YX = 0
print(f"\n{{X,Y}} = 0? {np.allclose(X @ Y + Y @ X, np.zeros((2,2)))}")
print(f"{{Y,Z}} = 0? {np.allclose(Y @ Z + Z @ Y, np.zeros((2,2)))}")
print(f"{{Z,X}} = 0? {np.allclose(Z @ X + X @ Z, np.zeros((2,2)))}")

# Eigenvalues (should all be +1 and -1 for Paulis)
print("\n=== Eigenvalues ===\n")
for name, gate in [("X", X), ("Y", Y), ("Z", Z)]:
    eigenvalues = np.linalg.eigvals(gate)
    print(f"Eigenvalues of {name}: {eigenvalues}")
```

---

## 3. 아다마르 게이트

아다마르 게이트(Hadamard gate) $H$는 양자 컴퓨팅에서 가장 중요한 단일 큐비트 게이트라 할 수 있습니다. 기저 상태로부터 중첩(superposition)을 만들고 거의 모든 양자 알고리즘에서 사용됩니다.

### 3.1 행렬 표현

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

### 3.2 기저 상태에 대한 동작

$$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |+\rangle$$

$$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |-\rangle$$

역방향:

$$H|+\rangle = |0\rangle, \quad H|-\rangle = |1\rangle$$

아다마르 게이트는 $Z$-기저와 $X$-기저 사이를 변환합니다.

### 3.3 주요 성질

- **자기 역원(Self-inverse)**: $H^2 = I$ (두 번 적용하면 원래 상태로 돌아옴)
- **파울리와의 관계**: $H = \frac{X + Z}{\sqrt{2}}$, 그리고 $HXH = Z$, $HZH = X$
- **블로흐 구**: $x$와 $z$ 축의 중간인 축(대각선 방향)을 중심으로 한 180도 회전($xz$-평면)

### 3.4 n 큐비트에 대한 아다마르

$|0\rangle$으로 초기화된 $n$개 큐비트 각각에 $H$를 적용하면 $2^n$개 기저 상태에 대한 균일한 중첩이 생성됩니다:

$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle$$

이는 도이치-조자(Deutsch-Jozsa) 알고리즘([레슨 7](07_Deutsch_Jozsa_Algorithm.md))과 그로버 탐색(Grover's search)([레슨 8](08_Grovers_Search.md))을 포함한 많은 양자 알고리즘의 시작점입니다.

더 일반적으로, $x \in \{0, 1\}$인 단일 기저 상태 $|x\rangle$에 아다마르를 적용하면:

$$H|x\rangle = \frac{1}{\sqrt{2}}\sum_{y=0}^{1}(-1)^{xy}|y\rangle$$

이는 $n$ 큐비트로 일반화됩니다:

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$

여기서 $x \cdot y$는 비트 단위 내적(bitwise inner product)을 mod 2로 취한 것입니다.

```python
import numpy as np

# The Hadamard gate

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("=== Hadamard Gate ===\n")
print(f"H|0> = {H @ ket_0}  (= |+>)")
print(f"H|1> = {H @ ket_1}  (= |->)")
print(f"H|+> = {H @ (ket_0 + ket_1)/np.sqrt(2)}  (= |0>)")
print(f"\nH^2 = I? {np.allclose(H @ H, np.eye(2))}")

# Hadamard on n qubits: creates uniform superposition
def hadamard_n(n):
    """
    Create the n-qubit Hadamard gate H^{tensor n}.

    Why tensor product? When gates act independently on separate qubits,
    the combined operation is the tensor product of individual gates.
    H^n creates a uniform superposition from |00...0>, which is the
    starting step of most quantum algorithms.
    """
    H_n = np.array([[1]], dtype=complex)
    for _ in range(n):
        H_n = np.kron(H_n, H)
    return H_n

# Demonstrate uniform superposition for n = 1, 2, 3
for n in range(1, 4):
    H_n = hadamard_n(n)
    initial = np.zeros(2**n, dtype=complex)
    initial[0] = 1  # |00...0>
    result = H_n @ initial

    print(f"\nH^{n} |{'0'*n}> = equal superposition of {2**n} states")
    print(f"  Amplitudes: {result}")
    print(f"  Each amplitude = 1/sqrt({2**n}) = {1/np.sqrt(2**n):.4f}")
    print(f"  Each probability = 1/{2**n} = {1/2**n:.4f}")

# Verify the (-1)^{x.y} formula for H|x>
print("\n=== Hadamard Action Formula: H|x> = (1/sqrt(2)) sum_y (-1)^{x*y} |y> ===\n")

for x in [0, 1]:
    ket_x = np.array([1-x, x], dtype=complex)
    result = H @ ket_x
    print(f"H|{x}> = {result}")
    for y in [0, 1]:
        expected_amp = (-1)**(x*y) / np.sqrt(2)
        print(f"  Coefficient of |{y}>: computed={result[y]:.4f}, "
              f"formula=(-1)^({x}*{y})/sqrt(2)={expected_amp:.4f}")
```

---

## 4. 위상 게이트: S와 T

위상 게이트(Phase gates)는 계산 기저에서의 측정 확률을 변화시키지 않으면서 $|0\rangle$과 $|1\rangle$ 사이의 상대 위상을 수정합니다.

### 4.1 일반 위상 게이트

$$R_\phi = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$$

이는 $|0\rangle$을 변화 없이 두고 $|1\rangle$에 $e^{i\phi}$를 곱합니다.

### 4.2 S 게이트 (위상 게이트, $\sqrt{Z}$)

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = R_{\pi/2}$$

성질:
- $S^2 = Z$ (따라서 "$\sqrt{Z}$"라 불림)
- $S|+\rangle = |i\rangle$: 블로흐 적도에서 $x$축에서 $y$축으로 회전
- **블로흐 구**: $z$축을 중심으로 한 90도 회전

### 4.3 T 게이트 ($\pi/8$ 게이트, $\sqrt{S}$)

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = R_{\pi/4}$$

성질:
- $T^2 = S$, $T^4 = Z$ (따라서 "$\sqrt[4]{Z}$")
- T 게이트는 역사적 이유로 "$\pi/8$ 게이트"라 불립니다 ($e^{i\pi/8} R_z(\pi/4)$와 같으며, $e^{i\pi/8}$는 전역 위상)
- **결정적 역할**: T 게이트는 H 및 CNOT와 결합될 때 양자 컴퓨팅을 보편적으로 만드는 요소입니다 (7절)
- **블로흐 구**: $z$축을 중심으로 한 45도 회전

### 4.4 게이트 계층 구조

$$T \xrightarrow{T^2} S \xrightarrow{S^2} Z \xrightarrow{Z^2} I$$

각 게이트는 다음 게이트의 "제곱근"입니다.

```python
import numpy as np

# Phase gates

S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

print("=== Phase Gate Hierarchy ===\n")
print(f"T^2 = S? {np.allclose(T @ T, S)}")
print(f"S^2 = Z? {np.allclose(S @ S, Z)}")
print(f"Z^2 = I? {np.allclose(Z @ Z, I)}")
print(f"T^4 = Z? {np.allclose(T @ T @ T @ T, Z)}")
print(f"T^8 = I? {np.allclose(np.linalg.matrix_power(T, 8), I)}")

# Action on |+> state
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

print("\n=== Phase Gates Acting on |+> ===\n")
print(f"|+> = {ket_plus}")

for name, gate in [("T", T), ("S", S), ("Z", Z)]:
    result = gate @ ket_plus
    # Compute the phase angle of the |1> component relative to |0>
    rel_phase = np.angle(result[1]) - np.angle(result[0])
    print(f"{name}|+> = {result}")
    print(f"  Relative phase = {rel_phase:.4f} rad = {rel_phase/np.pi:.4f}*pi")
    print(f"  P(0) = {abs(result[0])**2:.4f}, P(1) = {abs(result[1])**2:.4f}")
    print()

print("Key insight: Phase gates don't change P(0) or P(1) in the Z-basis,")
print("but they rotate the state around the equator of the Bloch sphere.")
print("The phase change becomes visible when measuring in the X or Y basis.")
```

---

## 5. 회전 게이트

### 5.1 일반 회전

모든 단일 큐비트 유니터리는 블로흐 구 위의 어떤 축을 중심으로 한 회전으로 분해될 수 있습니다. 세 주축에 대한 회전 게이트는 다음과 같습니다:

$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,X = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,Y = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### 5.2 명명된 게이트와 회전의 연결

명명된 게이트들은 회전의 특수 사례입니다:

| 게이트 | 회전 | 각도 |
|------|----------|-------|
| $X$ | $R_x(\pi)$ | $x$ 축 180° 회전 (전역 위상 제외) |
| $Y$ | $R_y(\pi)$ | $y$ 축 180° 회전 (전역 위상 제외) |
| $Z$ | $R_z(\pi)$ | $z$ 축 180° 회전 (전역 위상 제외) |
| $H$ | $R_y(\pi/2) \cdot R_z(\pi)$ | 결합 회전 |
| $S$ | $R_z(\pi/2)$ | $z$ 축 90° 회전 (전역 위상 제외) |
| $T$ | $R_z(\pi/4)$ | $z$ 축 45° 회전 (전역 위상 제외) |

참고: "전역 위상 제외(up to global phase)"는 회전 행렬이 명명된 게이트와 $e^{i\alpha}$ 인수만큼 다를 수 있음을 의미하며, 이는 물리적으로 관련이 없습니다.

### 5.3 오일러 분해

모든 단일 큐비트 게이트 $U$는 다음과 같이 쓸 수 있습니다:

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

여기서 $\alpha, \beta, \gamma, \delta$는 실수 각도입니다. 이는 고전 역학의 오일러 각도(Euler angle) 분해의 양자적 유사물입니다. 즉, 임의의 단일 큐비트 연산은 $R_z$와 $R_y$ 회전만으로 구성할 수 있습니다.

```python
import numpy as np

# Rotation gates

def Rx(theta):
    """Rotation around x-axis by angle theta."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Ry(theta):
    """Rotation around y-axis by angle theta."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Rz(theta):
    """
    Rotation around z-axis by angle theta.

    Why does Rz have this form? The Z-axis on the Bloch sphere corresponds
    to the computational basis. Rz simply applies a relative phase between
    |0> and |1>, which is a rotation around the Z-axis.
    """
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

# Verify named gates are special cases of rotations
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

print("=== Rotation Gate Verification ===\n")
print("Checking named gates as special rotations (up to global phase):\n")

# Why check "up to global phase"? Two unitary matrices that differ by a
# global phase e^{i*alpha} represent the same physical operation.
def same_up_to_phase(A, B):
    """Check if A = e^{i*alpha} * B for some alpha."""
    # If A = e^{i*alpha} * B, then A @ B.conj().T should be proportional to I
    ratio = A @ np.linalg.inv(B)
    phase = ratio[0, 0]
    return np.allclose(ratio, phase * np.eye(2))

print(f"X = Rx(pi) up to phase? {same_up_to_phase(X, Rx(np.pi))}")
print(f"Z = Rz(pi) up to phase? {same_up_to_phase(Z, Rz(np.pi))}")
print(f"S = Rz(pi/2) up to phase? {same_up_to_phase(S, Rz(np.pi/2))}")
print(f"T = Rz(pi/4) up to phase? {same_up_to_phase(T, Rz(np.pi/4))}")

# Demonstrate rotation with varying angles
print("\n=== Rz(theta) acting on |+> for various theta ===\n")
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

for angle_frac, label in [(0, "0"), (1/4, "pi/4"), (1/2, "pi/2"),
                           (1, "pi"), (3/2, "3pi/2")]:
    theta = angle_frac * np.pi
    result = Rz(theta) @ ket_plus
    rel_phase = np.angle(result[1]) - np.angle(result[0])
    print(f"Rz({label})|+> = [{result[0]:.4f}, {result[1]:.4f}], "
          f"relative phase = {rel_phase/np.pi:.3f}*pi")
```

---

## 6. 2큐비트 게이트

단일 큐비트 게이트만으로는 얽힘을 생성하거나 보편적인 양자 계산을 수행할 수 없습니다. 둘 이상의 큐비트에 동시에 작용하는 게이트가 필요합니다.

### 6.1 CNOT (제어-NOT)

CNOT 게이트는 가장 중요한 2큐비트 게이트입니다. **제어(control)** 큐비트와 **대상(target)** 큐비트를 가집니다:

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

**동작**: 제어 큐비트가 $|0\rangle$이면 아무것도 하지 않습니다. 제어 큐비트가 $|1\rangle$이면 대상 큐비트에 $X$를 적용합니다.

$$\text{CNOT}|00\rangle = |00\rangle, \quad \text{CNOT}|01\rangle = |01\rangle$$
$$\text{CNOT}|10\rangle = |11\rangle, \quad \text{CNOT}|11\rangle = |10\rangle$$

요약: $\text{CNOT}|a, b\rangle = |a, a \oplus b\rangle$ (여기서 $\oplus$는 XOR).

### 6.2 CZ (제어-Z)

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**동작**: 제어 큐비트가 $|1\rangle$이면 대상 큐비트에 $Z$를 적용합니다. 동등하게, $|11\rangle$ 성분에만 $-1$ 위상을 적용합니다.

**대칭성**: CNOT와 달리 CZ는 대칭입니다 -- 제어와 대상을 바꿔도 같은 게이트가 됩니다. $Z$가 대각행렬이기 때문입니다.

### 6.3 SWAP 게이트

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**동작**: 두 큐비트의 상태를 교환합니다: $\text{SWAP}|a, b\rangle = |b, a\rangle$.

**분해**: SWAP은 세 개의 CNOT으로 구성할 수 있습니다:

$$\text{SWAP} = \text{CNOT}_{12} \cdot \text{CNOT}_{21} \cdot \text{CNOT}_{12}$$

### 6.4 Controlled-U

임의의 단일 큐비트 게이트 $U$를 제어 게이트로 만들 수 있습니다:

$$C\text{-}U = \begin{pmatrix} I & 0 \\ 0 & U \end{pmatrix} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U$$

제어 큐비트가 $|0\rangle$이면 대상은 변화 없이 유지됩니다. 제어가 $|1\rangle$이면 대상에 $U$가 적용됩니다.

```python
import numpy as np

# Two-qubit gates

# CNOT gate (control=qubit 0, target=qubit 1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# CZ gate
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# SWAP gate
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# Basis states for 2 qubits
basis_2q = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

print("=== CNOT Action ===\n")
for label, state in basis_2q.items():
    result = CNOT @ state
    # Find the basis state label for the result
    for rlabel, rstate in basis_2q.items():
        if np.allclose(result, rstate):
            print(f"CNOT{label} = {rlabel}")
            break

print("\n=== CZ Action ===\n")
for label, state in basis_2q.items():
    result = CZ @ state
    for rlabel, rstate in basis_2q.items():
        if np.allclose(result, rstate):
            print(f"CZ{label} = {rlabel}")
            break
        if np.allclose(result, -rstate):
            print(f"CZ{label} = -{rlabel}")
            break

# Verify SWAP = CNOT_12 * CNOT_21 * CNOT_12
# Why does this work? Each CNOT transfers information one direction.
# Three alternating CNOTs effectively swap the two qubits, similar
# to the classical trick: a ^= b; b ^= a; a ^= b;
print("\n=== SWAP Decomposition ===\n")

# CNOT with control=0, target=1 (standard)
CNOT_01 = CNOT.copy()

# CNOT with control=1, target=0 (reversed)
# This is SWAP * CNOT * SWAP, but we can write it directly
CNOT_10 = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

swap_from_cnots = CNOT_01 @ CNOT_10 @ CNOT_01
print(f"SWAP = CNOT_01 * CNOT_10 * CNOT_01? {np.allclose(swap_from_cnots, SWAP)}")

# Creating entanglement with CNOT!
print("\n=== Creating Entanglement ===\n")
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Apply H to qubit 0, then CNOT
# H tensor I
H_I = np.kron(H, np.eye(2, dtype=complex))

# Start from |00>
state = np.array([1, 0, 0, 0], dtype=complex)
state = H_I @ state       # Apply H to qubit 0 -> (|00> + |10>)/sqrt(2)
state = CNOT @ state      # CNOT -> (|00> + |11>)/sqrt(2) = Bell state!

print(f"H(x)I |00>  then CNOT:")
for label, bstate in basis_2q.items():
    amp = np.vdot(bstate, state)
    if abs(amp) > 1e-10:
        print(f"  {label}: amplitude = {amp:.4f}, P = {abs(amp)**2:.4f}")

print("\nThis is the Bell state |Phi+> = (|00> + |11>)/sqrt(2)!")
print("We will study Bell states in Lesson 5.")
```

---

## 7. 보편 게이트 집합

### 7.1 보편성이란?

양자 게이트의 집합이 **보편적(universal)**이라는 것은 임의 개수의 큐비트에 대한 임의의 유니터리 연산을 해당 집합의 게이트들의 유한한 시퀀스로 임의의 정밀도까지 근사할 수 있다는 의미입니다.

이는 NAND 게이트 하나만으로 고전 계산에 대해 보편적이라는 고전적 결과와 유사합니다.

### 7.2 표준 보편 집합: {H, T, CNOT}

집합 $\{H, T, \text{CNOT}\}$은 양자 컴퓨팅에 대해 보편적입니다. 각 게이트가 필요한 이유:

| 게이트 | 역할 | 제공하는 것 |
|------|------|-----------------|
| $H$ | 중첩 생성 | $X$-기저에의 접근; $T$와 결합하면 임의의 $z$-회전 가능 |
| $T$ | 세밀한 위상 조정 | $H$와 결합하면 임의의 단일 큐비트 게이트 근사 |
| CNOT | 2큐비트 얽힘 | 단일 큐비트 게이트만으로는 얽힘 생성 불가 |

### 7.3 S 대신 T를 사용하는 이유?

S 게이트는 $z$축에 대한 $\pi/2$ 회전을 제공합니다. H와 결합하면 유한한 상태 집합(클리퍼드 군, Clifford group)에만 도달할 수 있습니다. T ($\pi/4$ 회전)를 추가하면 이 제한에서 벗어나 *임의의* 단일 큐비트 회전을 근사할 수 있습니다.

**솔로베이-키타예프 정리(Solovay-Kitaev theorem)**는 임의의 단일 큐비트 게이트를 $\{H, T\}$에서 $O(\log^c(1/\epsilon))$개 게이트로 정밀도 $\epsilon$까지 근사할 수 있음을 보장합니다 (여기서 $c \approx 3.97$). 이는 놀랍도록 효율적입니다.

### 7.4 다른 보편 집합들

여러 다른 게이트 집합도 보편적입니다:

- $\{H, T, \text{CNOT}\}$ -- 표준 집합
- $\{H, \text{Toffoli}\}$ -- 아다마르 + 3큐비트 게이트
- $\{R_y(\theta), R_z(\phi), \text{CNOT}\}$ (비유리수 $\theta/\pi$에 대해) -- 연속 회전 집합
- $\{\text{CNOT}, \text{임의의 비-클리퍼드 단일 큐비트 게이트}\}$

```python
import numpy as np

# Demonstrating universality: approximating arbitrary gates with H, T

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
T_dag = T.conj().T  # T-dagger (inverse of T)

# Key identity: HT^k H generates rotations that, when composed,
# can approximate any single-qubit gate

print("=== Building Gates from H and T ===\n")

# Some useful compositions
S = T @ T  # S = T^2
Z = S @ S  # Z = S^2 = T^4

print(f"T^2 = S? {np.allclose(T @ T, np.array([[1,0],[0,1j]]))}")
print(f"T^4 = Z? {np.allclose(np.linalg.matrix_power(T, 4), np.array([[1,0],[0,-1]]))}")

# HTH gives a rotation in a different direction
HTH = H @ T @ H
print(f"\nHTH (a rotation mixing X and Z):\n{HTH}")
print(f"Is unitary? {np.allclose(HTH @ HTH.conj().T, np.eye(2))}")

# Approximating Rx(pi/8) using H and T sequences
# The Solovay-Kitaev theorem guarantees this is possible
# Here we show that composing HT sequences creates diverse rotations

gates_from_HT = {
    "H": H,
    "T": T,
    "HT": H @ T,
    "TH": T @ H,
    "HTH": H @ T @ H,
    "THT": T @ H @ T,
    "HTHT": H @ T @ H @ T,
    "THTH": T @ H @ T @ H,
}

print("\n=== Diverse Gates from H, T Compositions ===\n")
print(f"{'Sequence':<10} {'Eigenvalues':>30} {'Det':>15}")
print("-" * 60)

for name, gate in gates_from_HT.items():
    eigs = np.linalg.eigvals(gate)
    det = np.linalg.det(gate)
    print(f"{name:<10} {str(np.round(eigs, 4)):>30} {det:>15.4f}")

print("\nThe diverse eigenvalues show that H and T compositions")
print("reach many different rotations -- the basis of universality.")
```

---

## 8. 게이트 분해

### 8.1 ZYZ 분해

임의의 단일 큐비트 유니터리 $U$는 다음과 같이 분해될 수 있습니다:

$$U = e^{i\alpha} R_z(\beta) R_y(\gamma) R_z(\delta)$$

이는 4개의 실수 매개변수를 사용하며, 2x2 유니터리 행렬의 4 자유도($SU(2)$ 부분 3개 + 전역 위상 1개)와 일치합니다.

### 8.2 제어 게이트를 위한 ABC 분해

임의의 제어-$U$ 게이트는 단일 큐비트 게이트와 최대 2개의 CNOT으로 분해될 수 있습니다:

$$C\text{-}U = (I \otimes A) \cdot \text{CNOT} \cdot (I \otimes B) \cdot \text{CNOT} \cdot (I \otimes C)$$

여기서 $ABC = I$이고 적절한 행렬 $A, B, C$와 위상 $\alpha$에 대해 $e^{i\alpha} AXBXC = U$입니다.

### 8.3 실용적인 예제

$S = T^2$이고 $T$가 $z$ 축에 대한 $\pi/4$ 회전이라는 사실을 이용하여 제어-$S$ 게이트를 분해해 봅시다.

```python
import numpy as np

# Gate decomposition examples

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def zyz_decompose(U):
    """
    Decompose a single-qubit unitary into Rz-Ry-Rz form.

    Why ZYZ? Any rotation in 3D can be decomposed into three rotations
    about two fixed axes (Euler angles). In quantum computing, we use
    the Z-Y-Z convention because Rz gates are often 'free' (implemented
    by adjusting control software timing) on many hardware platforms.
    """
    # Extract global phase
    det = np.linalg.det(U)
    alpha = np.angle(det) / 2
    # Remove global phase to get SU(2) matrix
    V = U * np.exp(-1j * alpha)

    # For SU(2): V = [[a, -b*], [b, a*]] with |a|^2 + |b|^2 = 1
    a = V[0, 0]
    b = V[1, 0]

    # gamma = 2*arccos(|a|)
    gamma = 2 * np.arccos(np.clip(abs(a), -1, 1))

    if np.isclose(gamma, 0):
        beta = np.angle(a) * 2
        delta = 0
    elif np.isclose(gamma, np.pi):
        beta = np.angle(b) * 2
        delta = 0
    else:
        beta = np.angle(a) - np.angle(b)  # Phase of |0>-like component
        delta = np.angle(a) + np.angle(b)  # Phase of |1>-like component

    return alpha, beta, gamma, delta

# Test with Hadamard gate
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
alpha, beta, gamma, delta = zyz_decompose(H)

print("=== ZYZ Decomposition of Hadamard ===\n")
print(f"H = exp(i*{alpha/np.pi:.4f}*pi) * Rz({beta/np.pi:.4f}*pi) "
      f"* Ry({gamma/np.pi:.4f}*pi) * Rz({delta/np.pi:.4f}*pi)")

# Verify
H_reconstructed = np.exp(1j * alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)
print(f"\nOriginal H:\n{H}")
print(f"\nReconstructed:\n{np.round(H_reconstructed, 4)}")
print(f"Match? {np.allclose(H, H_reconstructed)}")

# Decompose a random single-qubit gate
print("\n=== ZYZ Decomposition of Random Gate ===\n")
rng = np.random.default_rng(123)
random_matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
Q, R = np.linalg.qr(random_matrix)
U_random = Q

alpha, beta, gamma, delta = zyz_decompose(U_random)
U_reconstructed = np.exp(1j * alpha) * Rz(beta) @ Ry(gamma) @ Rz(delta)

print(f"Parameters: alpha={alpha:.4f}, beta={beta:.4f}, "
      f"gamma={gamma:.4f}, delta={delta:.4f}")
print(f"Match? {np.allclose(U_random, U_reconstructed)}")
```

---

## 9. 연습 문제

### 연습 1: 게이트 검증

명시적인 행렬 곱셈으로 다음을 확인하세요:
a) $HZH = X$ (아다마르가 $Z$를 $X$로 켤레 변환)
b) $HXH = Z$
c) $SXS^\dagger = Y$ (위상 제외)

각 항등식을 확인하는 파이썬 코드를 작성하세요.

### 연습 2: 블로흐 구 회전

a) $|0\rangle$에서 시작하여 $R_y(\pi/4)$를 적용하세요. 결과 상태는 무엇인가요? 블로흐 구 위의 어디에 있나요?
b) $|+\rangle$에서 시작하여 $R_z(\pi/2)$를 적용하세요. 결과 상태는 무엇인가요?
c) $\theta = \pi/4$에 대해 $R_x(\theta)R_y(\theta)$와 $R_y(\theta)R_x(\theta)$가 같지 않음을 확인하세요. 양자 게이트는 일반적으로 교환하지 않습니다!

### 연습 3: 제어 게이트 구성

제어-$H$ 게이트를 4x4 행렬로 구성하세요. 다음을 확인하세요:
a) $CH|00\rangle = |00\rangle$
b) $CH|10\rangle = |1\rangle \otimes H|0\rangle = |1,+\rangle$

### 연습 4: 게이트 수 세기

$\{H, T, \text{CNOT}\}$에서 다음 각각을 정확히 구현하는 데 몇 개의 게이트가 필요한가요?
a) $X$ 게이트
b) $S$ 게이트
c) SWAP 게이트
d) 토폴리(Toffoli) 게이트 (힌트: 토폴리는 6개의 CNOT과 여러 단일 큐비트 게이트가 필요합니다)

### 연습 5: 사용자 정의 게이트

$|0\rangle \to \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$을 매핑하는 단일 큐비트 게이트를 설계하고 ZYZ 분해를 구하세요. 이 매핑과 일관된 전체 $2 \times 2$ 유니터리 행렬을 확인하세요. (기억하세요: 게이트를 완전히 정의하려면 $|1\rangle$에 일어나는 일도 명시해야 합니다.)

---

[<- 이전: 큐비트와 블로흐 구(Qubits and the Bloch Sphere)](02_Qubits_and_Bloch_Sphere.md) | [다음: 양자 회로(Quantum Circuits) ->](04_Quantum_Circuits.md)

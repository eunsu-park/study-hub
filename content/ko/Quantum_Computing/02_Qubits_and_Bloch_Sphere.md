# 레슨 2: 큐비트와 블로흐 구

[<- 이전: 양자역학 입문](01_Quantum_Mechanics_Primer.md) | [다음: 양자 게이트 ->](03_Quantum_Gates.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 큐비트 상태를 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$로 표현하고 정규화 조건(normalization constraint)을 설명할 수 있다
2. 전역 위상(global phase)과 상대 위상(relative phase)을 구별하고, 상대 위상만이 관측 가능한 이유를 설명할 수 있다
3. $(\theta, \phi)$ 매개변수화(parametrization)를 사용하여 임의의 단일 큐비트 상태를 블로흐 구(Bloch sphere) 위의 한 점으로 대응시킬 수 있다
4. $\{|+\rangle, |-\rangle\}$ 및 $\{|i\rangle, |-i\rangle\}$를 포함한 여러 측정 기저(measurement basis)를 다룰 수 있다
5. 텐서곱(tensor product)을 사용하여 다중 큐비트 상태(multi-qubit state)를 구성할 수 있다
6. $n$큐비트 상태 공간의 차원을 계산할 수 있다
7. Python으로 큐비트 상태와 블로흐 구 좌표를 구현할 수 있다

---

큐비트(qubit)는 고전적인 비트에 유사한 양자 정보의 기본 단위입니다. 고전 비트가 0 또는 1만 될 수 있는 반면, 큐비트는 두 복소수로 매개변수화된 $|0\rangle$과 $|1\rangle$의 임의의 중첩으로 존재할 수 있습니다. 큐비트 상태를 어떻게 표현하고, 시각화하고, 조작하는지 이해하는 것이 양자 컴퓨팅 전체의 필수적인 출발점입니다.

이 레슨에서는 단일 큐비트 및 다중 큐비트 상태에 대한 완전한 수학적 틀을 개발합니다. 블로흐 구는 단일 큐비트 상태를 아름답게 기하학적으로 시각화하여, 추상적인 양자 상태를 구체적으로 만들고 양자 게이트(다음 레슨에서 공부할)에 대한 직관을 형성합니다.

> **비유:** 블로흐 구는 양자 상태를 위한 지구본과 같습니다 — 위도는 $|0\rangle$ 대비 $|1\rangle$의 비율을 결정하고("북극"이 $|0\rangle$, "남극"이 $|1\rangle$), 경도는 그 사이의 위상 관계를 결정합니다. 지구상의 어느 점이든 위도와 경도로 특정할 수 있듯이, 모든 단일 큐비트 상태는 블로흐 구 위의 두 각도로 특정할 수 있습니다.

## 목차

1. [큐비트 상태](#1-큐비트-상태)
2. [전역 위상 vs 상대 위상](#2-전역-위상-vs-상대-위상)
3. [블로흐 구](#3-블로흐-구)
4. [중요한 기저들](#4-중요한-기저들)
5. [다중 큐비트 시스템](#5-다중-큐비트-시스템)
6. [N큐비트 상태 공간](#6-n큐비트-상태-공간)
7. [연습문제](#7-연습문제)

---

## 1. 큐비트 상태

### 1.1 수학적 정의

**큐비트(qubit, quantum bit)**는 2준위 양자 시스템(two-level quantum system)입니다. 그 상태는 2차원 복소 힐베르트 공간(Hilbert space) $\mathbb{C}^2$의 벡터입니다:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

여기서 $\alpha, \beta \in \mathbb{C}$는 정규화 조건(normalization constraint)을 만족하는 복소 **진폭(amplitude)**입니다:

$$|\alpha|^2 + |\beta|^2 = 1$$

이 정규화는 모든 측정 결과의 총 확률이 1이 됨을 보장합니다([레슨 1](01_Quantum_Mechanics_Primer.md)의 보른 규칙).

### 1.2 "2준위"가 의미하는 것

물리적으로, 큐비트는 두 개의 구별 가능한 상태를 가진 양자 시스템으로 구현될 수 있습니다:

| 물리적 시스템 | $\|0\rangle$ | $\|1\rangle$ |
|--------------|-------------|-------------|
| 광자 편광(Photon polarization) | 수평(Horizontal) | 수직(Vertical) |
| 전자 스핀(Electron spin) | 스핀 업(Spin up) | 스핀 다운(Spin down) |
| 초전도 회로(Superconducting circuit) | 바닥 상태(Ground state) | 들뜬 상태(Excited state) |
| 포획된 이온(Trapped ion) | 기저 에너지 준위 | 들뜬 에너지 준위 |
| 양자점(Quantum dot) | 전자 없음 | 전자 하나 |

수학은 물리적 구현에 관계없이 동일합니다. 이 추상화가 양자 컴퓨팅 이론의 큰 강점 중 하나입니다.

### 1.3 자유도

상태 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$에는 두 복소수 = 4개의 실수 매개변수가 있습니다. 그러나:
- 정규화(normalization)가 자유도를 1개 제거합니다: $|\alpha|^2 + |\beta|^2 = 1$
- 전역 위상(global phase)이 1개를 더 제거합니다 (다음 섹션에서 볼 것처럼)

이로써 큐비트 상태를 지정하는 데 **2개의 실수 매개변수**가 남습니다 — 블로흐 구 위의 두 각도 $(\theta, \phi)$가 정확히 이것입니다.

```python
import numpy as np

# Creating qubit states

# Basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

def make_qubit(alpha, beta):
    """
    Create a qubit state |psi> = alpha|0> + beta|1>.

    Why check normalization? A non-normalized state would give probabilities
    that don't sum to 1, which is physically meaningless. This is the
    fundamental constraint on all quantum states.
    """
    state = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        print(f"  Warning: State not normalized (norm = {norm:.6f}). Normalizing...")
        state = state / norm
    return state

# Example states
print("=== Various Qubit States ===\n")

states = {
    "|0>": make_qubit(1, 0),
    "|1>": make_qubit(0, 1),
    "|+>": make_qubit(1/np.sqrt(2), 1/np.sqrt(2)),
    "|->": make_qubit(1/np.sqrt(2), -1/np.sqrt(2)),
    "|i>": make_qubit(1/np.sqrt(2), 1j/np.sqrt(2)),
    "custom": make_qubit(1/np.sqrt(3), np.sqrt(2/3) * np.exp(1j * np.pi/4)),
}

for name, state in states.items():
    p0 = abs(state[0])**2
    p1 = abs(state[1])**2
    print(f"{name:>10}: [{state[0]:.4f}, {state[1]:.4f}]  "
          f"P(0)={p0:.4f}, P(1)={p1:.4f}")

# A deliberately non-normalized state (will be corrected)
print("\n--- Non-normalized state ---")
bad_state = make_qubit(1, 1)  # Sum of squares = 2, not 1
print(f"Normalized: [{bad_state[0]:.4f}, {bad_state[1]:.4f}]")
```

---

## 2. 전역 위상 vs 상대 위상

전역 위상과 상대 위상의 구분을 이해하는 것은 양자 컴퓨팅에서 매우 중요합니다. 이것이 큐비트가 3개가 아닌 2개의 실수 자유도만 가지는 이유를 설명합니다.

### 2.1 전역 위상은 관측 불가능

전체적인 (전역) 위상 인수 $e^{i\gamma}$만큼 다른 두 상태는 **물리적으로 동일**합니다:

$$|\psi\rangle \equiv e^{i\gamma}|\psi\rangle \quad \text{모든 } \gamma \in \mathbb{R}$$

왜냐하면? 모든 관측 가능한 양은 $|\langle\phi|\psi\rangle|^2$에 의존하는데:

$$|\langle\phi|e^{i\gamma}\psi\rangle|^2 = |e^{i\gamma}|^2 \cdot |\langle\phi|\psi\rangle|^2 = |\langle\phi|\psi\rangle|^2$$

항상 $|e^{i\gamma}| = 1$이기 때문입니다.

**실용적 결과**: 일반성을 잃지 않고 $\alpha$를 실수이자 비음수로 선택할 수 있습니다. $\alpha = |\alpha|e^{i\gamma}$이면, 전체 상태에 $e^{-i\gamma}$를 곱합니다:

$$|\psi\rangle = |\alpha|e^{i\gamma}|0\rangle + \beta|1\rangle \equiv |\alpha||0\rangle + \beta e^{-i\gamma}|1\rangle$$

### 2.2 상대 위상은 관측 가능

중첩의 성분들 *사이*의 위상은 물리적으로 의미가 있습니다. 다음을 고려합니다:

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

이 상태들은 $\{|0\rangle, |1\rangle\}$ 기저에서 같은 측정 확률을 가집니다 (각각 50/50). 하지만 이들은 물리적으로 다른 상태입니다! $|0\rangle$과 $|1\rangle$ 성분 사이의 상대 위상(0 vs $\pi$)이 다릅니다.

어떻게 구별할 수 있을까요? *다른* 기저에서 측정함으로써. $\{|+\rangle, |-\rangle\}$ 기저에서:
- $|+\rangle$은 확률 1로 $|+\rangle$을 줍니다
- $|-\rangle$은 확률 1로 $|-\rangle$을 줍니다

상대 위상은 실제 물리적 결과를 가집니다 — 이를 드러내기 위한 올바른 측정이 필요할 뿐입니다.

```python
import numpy as np

# Demonstrating global vs relative phase

print("=== Global Phase: Unobservable ===\n")

# State 1: |psi> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

# State 2: e^(i*pi/4) * |psi>  (global phase of pi/4)
gamma = np.pi / 4
psi2 = np.exp(1j * gamma) * psi1

print(f"State 1: {psi1}")
print(f"State 2: {psi2}  (State 1 * e^(i*pi/4))")
print(f"\nMeasurement probabilities in {{|0>, |1>}} basis:")
print(f"  State 1: P(0) = {abs(psi1[0])**2:.4f}, P(1) = {abs(psi1[1])**2:.4f}")
print(f"  State 2: P(0) = {abs(psi2[0])**2:.4f}, P(1) = {abs(psi2[1])**2:.4f}")
print("  => Identical! Global phase has no effect on measurement.")

# Also check in the {|+>, |->} basis
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

# Why use inner product for basis change? The probability of outcome |phi>
# given state |psi> is always |<phi|psi>|^2, regardless of basis
p_plus_1 = abs(np.vdot(ket_plus, psi1))**2
p_plus_2 = abs(np.vdot(ket_plus, psi2))**2
print(f"\nIn {{|+>, |->}} basis:")
print(f"  State 1: P(+) = {p_plus_1:.4f}")
print(f"  State 2: P(+) = {p_plus_2:.4f}")
print("  => Still identical in any basis!")

print("\n\n=== Relative Phase: Observable ===\n")

# |+> = (|0> + |1>)/sqrt(2)         relative phase = 0
# |-> = (|0> - |1>)/sqrt(2)         relative phase = pi
# |i> = (|0> + i|1>)/sqrt(2)        relative phase = pi/2

states = {
    "|+> (phase=0)": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|-> (phase=pi)": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|i> (phase=pi/2)": np.array([1, 1j], dtype=complex) / np.sqrt(2),
}

print("Measurement in {|0>, |1>} basis (relative phase invisible):")
for name, state in states.items():
    print(f"  {name}: P(0) = {abs(state[0])**2:.4f}, P(1) = {abs(state[1])**2:.4f}")

print("\nMeasurement in {|+>, |->} basis (relative phase revealed!):")
for name, state in states.items():
    p_p = abs(np.vdot(ket_plus, state))**2
    p_m = abs(np.vdot(ket_minus, state))**2
    print(f"  {name}: P(+) = {p_p:.4f}, P(-) = {p_m:.4f}")
```

---

## 3. 블로흐 구

### 3.1 매개변수화

단일 큐비트 순수 상태(pure state)가 2개의 실수 자유도를 가지므로, 단위 구(unit sphere)의 표면 위의 점으로 대응시킬 수 있습니다 — 이것이 **블로흐 구(Bloch sphere)**입니다.

$\alpha$를 실수이자 비음수로 하는 관례를 사용하면, 임의의 큐비트 상태를 다음과 같이 쓸 수 있습니다:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

여기서:
- $\theta \in [0, \pi]$는 **극각(polar angle)** (북극에서의 여위도)입니다
- $\phi \in [0, 2\pi)$는 **방위각(azimuthal angle)** (경도)입니다

### 3.2 왜 $\theta$ 대신 $\theta/2$인가?

$1/2$ 인수는 임의적이지 않습니다. 다음 이유로 발생합니다:

1. **직교 상태가 대척점으로 대응**: $|0\rangle$과 $|1\rangle$은 힐베르트 공간에서 직교하지만, 지름 방향으로 반대편인 북극과 남극($\theta = 0$과 $\theta = \pi$)에 대응합니다. $1/2$ 없이는 이 대응을 달성할 수 없습니다.

2. **완전한 커버리지**: $\theta$는 $0$에서 $\pi$까지이므로 $\theta/2$는 $0$에서 $\pi/2$까지 범위입니다. 이는 $\cos(\theta/2)$가 1에서 0으로, $\sin(\theta/2)$가 0에서 1로 변함을 의미합니다. 이것이 가능한 모든 진폭 분포를 커버합니다.

### 3.3 블로흐 구 위의 특수 점들

| 상태 | $\theta$ | $\phi$ | 블로흐 좌표 $(x, y, z)$ |
|------|:---:|:---:|:---:|
| $\|0\rangle$ | $0$ | 임의 | $(0, 0, 1)$ — 북극 |
| $\|1\rangle$ | $\pi$ | 임의 | $(0, 0, -1)$ — 남극 |
| $\|+\rangle = \frac{\|0\rangle + \|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $0$ | $(1, 0, 0)$ — 양의 $x$ |
| $\|-\rangle = \frac{\|0\rangle - \|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $\pi$ | $(-1, 0, 0)$ — 음의 $x$ |
| $\|i\rangle = \frac{\|0\rangle + i\|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $\pi/2$ | $(0, 1, 0)$ — 양의 $y$ |
| $\|-i\rangle = \frac{\|0\rangle - i\|1\rangle}{\sqrt{2}}$ | $\pi/2$ | $3\pi/2$ | $(0, -1, 0)$ — 음의 $y$ |

### 3.4 직교 좌표

블로흐 구 좌표 $(x, y, z)$는 각도와 다음과 같이 관련됩니다:

$$x = \sin\theta\cos\phi, \quad y = \sin\theta\sin\phi, \quad z = \cos\theta$$

이 좌표들은 또한 **파울리 행렬(Pauli matrices)**의 기댓값에도 대응합니다([레슨 3](03_Quantum_Gates.md)에서 만날 것입니다):

$$x = \langle\psi|X|\psi\rangle, \quad y = \langle\psi|Y|\psi\rangle, \quad z = \langle\psi|Z|\psi\rangle$$

### 3.5 측정의 기하학적 해석

계산 기저($\{|0\rangle, |1\rangle\}$)에서의 측정은 블로흐 벡터를 $z$축에 사영(projection)하는 것에 해당합니다:

- $P(0) = \cos^2(\theta/2) = (1 + z)/2$
- $P(1) = \sin^2(\theta/2) = (1 - z)/2$

북극 근처($z \approx 1$)의 상태는 "대부분 $|0\rangle$"이고, 남극 근처($z \approx -1$)의 상태는 "대부분 $|1\rangle$"입니다. 적도($z = 0$)의 상태는 완벽하게 균형잡힌 중첩입니다.

```python
import numpy as np

def qubit_to_bloch(state):
    """
    Convert a qubit state vector to Bloch sphere coordinates.

    Why extract theta and phi this way? We use the standard parametrization
    |psi> = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>, first removing
    the global phase to make alpha real and non-negative.

    Returns (theta, phi, x, y, z).
    """
    alpha, beta = state[0], state[1]

    # Remove global phase: make alpha real and non-negative
    if abs(alpha) > 1e-10:
        phase = np.exp(-1j * np.angle(alpha))
        alpha = alpha * phase
        beta = beta * phase

    # Extract theta from |alpha| = cos(theta/2)
    alpha_real = np.real(alpha)  # Now alpha is real
    alpha_real = np.clip(alpha_real, -1, 1)  # Numerical safety
    theta = 2 * np.arccos(alpha_real)

    # Extract phi from beta = e^(i*phi) * sin(theta/2)
    if abs(beta) > 1e-10:
        phi = np.angle(beta)
    else:
        phi = 0.0

    # Cartesian coordinates on the Bloch sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return theta, phi, x, y, z

def bloch_to_qubit(theta, phi):
    """
    Convert Bloch sphere angles to a qubit state vector.

    Why this is useful: it lets us think geometrically about quantum states.
    Any point on the Bloch sphere corresponds to a unique qubit state
    (up to global phase, which is unobservable).
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=complex)

# Map special states to the Bloch sphere
print("=== Special States on the Bloch Sphere ===\n")
print(f"{'State':<20} {'theta/pi':>10} {'phi/pi':>10} {'x':>8} {'y':>8} {'z':>8}")
print("-" * 70)

special_states = {
    "|0>": np.array([1, 0], dtype=complex),
    "|1>": np.array([0, 1], dtype=complex),
    "|+>": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|->": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|i>": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "|-i>": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}

for name, state in special_states.items():
    theta, phi, x, y, z = qubit_to_bloch(state)
    print(f"{name:<20} {theta/np.pi:>10.4f} {phi/np.pi:>10.4f} "
          f"{x:>8.4f} {y:>8.4f} {z:>8.4f}")

# Round trip: angles -> state -> angles
print("\n=== Round-Trip Verification ===\n")
test_angles = [(np.pi/3, np.pi/4), (np.pi/2, np.pi), (2*np.pi/3, 3*np.pi/2)]

for theta_in, phi_in in test_angles:
    state = bloch_to_qubit(theta_in, phi_in)
    theta_out, phi_out, x, y, z = qubit_to_bloch(state)
    print(f"Input: theta={theta_in/np.pi:.3f}*pi, phi={phi_in/np.pi:.3f}*pi")
    print(f"  State: [{state[0]:.4f}, {state[1]:.4f}]")
    print(f"  Recovered: theta={theta_out/np.pi:.3f}*pi, phi={phi_out/np.pi:.3f}*pi")
    print(f"  Bloch: ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"  P(0) = {abs(state[0])**2:.4f}, P(1) = {abs(state[1])**2:.4f}")
    print()
```

---

## 4. 중요한 기저들

계산 기저 $\{|0\rangle, |1\rangle\}$이 가장 흔히 사용되지만, 다른 기저들도 양자 컴퓨팅에 필수적입니다.

### 4.1 X-기저 (아다마르 기저)

$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

이 기저는 블로흐 구의 적도에서 $x$축 방향에 놓입니다. 아다마르 게이트(Hadamard gate, [레슨 3](03_Quantum_Gates.md) 참조)가 $Z$-기저와 $X$-기저 사이를 변환하기 때문에 "아다마르 기저(Hadamard basis)"라고도 합니다.

**역관계** (동등하게 유용):

$$|0\rangle = \frac{|+\rangle + |-\rangle}{\sqrt{2}}, \quad |1\rangle = \frac{|+\rangle - |-\rangle}{\sqrt{2}}$$

### 4.2 Y-기저 (원형 기저)

$$|i\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt{2}}, \quad |-i\rangle = \frac{|0\rangle - i|1\rangle}{\sqrt{2}}$$

이 기저는 $y$축 방향의 적도에 놓입니다. 광자 편광(photon polarization) 구현에서 이들이 우원편광과 좌원편광에 해당하기 때문에 때로 "원형(circular)" 기저라 불립니다.

### 4.3 다른 기저에서의 측정

기저 $\{|b_0\rangle, |b_1\rangle\}$에서 측정하려면 다음을 계산합니다:

$$P(b_0) = |\langle b_0|\psi\rangle|^2, \quad P(b_1) = |\langle b_1|\psi\rangle|^2$$

측정 기저의 선택은 양자 컴퓨팅에서 핵심적인 자유도입니다. 다른 기저들은 상태에 대해 다른 정보를 드러냅니다.

```python
import numpy as np

# Measurement in different bases

def measure_in_basis(state, basis_states, basis_names):
    """
    Compute measurement probabilities in an arbitrary orthonormal basis.

    Why allow arbitrary bases? In quantum computing, the choice of
    measurement basis is as important as the state preparation. Some
    algorithms require measuring in the X-basis or Y-basis, not just
    the computational (Z) basis.
    """
    print(f"  State: [{state[0]:.4f}, {state[1]:.4f}]")
    for bstate, bname in zip(basis_states, basis_names):
        # Inner product <basis|state>
        amplitude = np.vdot(bstate, state)
        prob = abs(amplitude)**2
        print(f"  P({bname}) = |<{bname}|psi>|^2 = |{amplitude:.4f}|^2 = {prob:.4f}")
    print()

# Define three measurement bases
bases = {
    "Z-basis {|0>, |1>}": (
        [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)],
        ["|0>", "|1>"]
    ),
    "X-basis {|+>, |->}": (
        [np.array([1, 1], dtype=complex)/np.sqrt(2),
         np.array([1, -1], dtype=complex)/np.sqrt(2)],
        ["|+>", "|->"]
    ),
    "Y-basis {|i>, |-i>}": (
        [np.array([1, 1j], dtype=complex)/np.sqrt(2),
         np.array([1, -1j], dtype=complex)/np.sqrt(2)],
        ["|i>", "|-i>"]
    ),
}

# Test with |+> state
test_state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+>
print("=== Measuring |+> in Different Bases ===\n")
for basis_name, (bstates, bnames) in bases.items():
    print(f"{basis_name}:")
    measure_in_basis(test_state, bstates, bnames)

# Test with |i> state
test_state = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # |i>
print("=== Measuring |i> in Different Bases ===\n")
for basis_name, (bstates, bnames) in bases.items():
    print(f"{basis_name}:")
    measure_in_basis(test_state, bstates, bnames)

# Key insight: each basis reveals different information!
print("Key Insight:")
print("  |+> is definite in X-basis but random in Y and Z bases")
print("  |i> is definite in Y-basis but random in X and Z bases")
print("  This is the quantum uncertainty principle in action!")
```

---

## 5. 다중 큐비트 시스템

### 5.1 텐서곱

두 개의 큐비트가 있을 때, 결합 상태 공간(combined state space)은 개별 공간들의 **텐서곱(tensor product)**입니다:

$$\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$$

큐비트 A가 상태 $|\psi_A\rangle$에, 큐비트 B가 상태 $|\psi_B\rangle$에 있고 (그들이 얽히지 않았다면), 결합 상태는:

$$|\psi_{AB}\rangle = |\psi_A\rangle \otimes |\psi_B\rangle$$

계산 기저에 대해:

$$|0\rangle \otimes |0\rangle = |00\rangle, \quad |0\rangle \otimes |1\rangle = |01\rangle, \quad |1\rangle \otimes |0\rangle = |10\rangle, \quad |1\rangle \otimes |1\rangle = |11\rangle$$

### 5.2 크로네커 곱으로서의 텐서곱

행렬 표현에서 텐서곱은 **크로네커 곱(Kronecker product)**으로 계산됩니다:

$$|0\rangle \otimes |1\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \cdot 0 \\ 1 \cdot 1 \\ 0 \cdot 0 \\ 0 \cdot 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} = |01\rangle$$

규칙: 첫 번째 벡터의 각 원소가 두 번째 벡터 전체를 곱합니다.

### 5.3 2큐비트 계산 기저

$$|00\rangle = \begin{pmatrix} 1\\0\\0\\0 \end{pmatrix}, \quad |01\rangle = \begin{pmatrix} 0\\1\\0\\0 \end{pmatrix}, \quad |10\rangle = \begin{pmatrix} 0\\0\\1\\0 \end{pmatrix}, \quad |11\rangle = \begin{pmatrix} 0\\0\\0\\1 \end{pmatrix}$$

이진 문자열이 1의 위치를 결정합니다: $|b_1 b_0\rangle$은 위치 $2b_1 + b_0$에 1이 있습니다 (이진 문자열을 정수로 해석).

### 5.4 일반 2큐비트 상태

일반 2큐비트 상태는:

$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

정규화 조건: $|\alpha_{00}|^2 + |\alpha_{01}|^2 + |\alpha_{10}|^2 + |\alpha_{11}|^2 = 1$.

**중요**: 모든 2큐비트 상태가 단일 큐비트 상태들의 텐서곱으로 쓰여질 수 있는 것은 아닙니다. 그럴 수 없는 상태를 **얽힘 상태(entangled state)**라 합니다 — [레슨 5](05_Entanglement_and_Bell_States.md)에서 깊이 탐구할 주제입니다.

```python
import numpy as np

# Multi-qubit states via tensor products

def tensor_product(state_a, state_b):
    """
    Compute the tensor product of two quantum states using Kronecker product.

    Why Kronecker product? It implements the tensor product of vector spaces
    in matrix representation. For n-qubit systems, we chain multiple
    Kronecker products together.
    """
    return np.kron(state_a, state_b)

# Single-qubit basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Two-qubit computational basis
print("=== Two-Qubit Computational Basis ===\n")
basis_2q = {}
for b1 in [0, 1]:
    for b0 in [0, 1]:
        label = f"|{b1}{b0}>"
        state = tensor_product(
            ket_0 if b1 == 0 else ket_1,
            ket_0 if b0 == 0 else ket_1
        )
        basis_2q[label] = state
        print(f"{label} = {state}")

# Product state: |+> tensor |0>
print("\n=== Product State: |+> (x) |0> ===\n")
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
product = tensor_product(ket_plus, ket_0)
print(f"|+> (x) |0> = {product}")
print(f"= ({product[0]:.4f})|00> + ({product[1]:.4f})|01> + "
      f"({product[2]:.4f})|10> + ({product[3]:.4f})|11>")

# Verify normalization
print(f"Norm: {np.linalg.norm(product):.4f}")

# Measurement probabilities
print(f"\nMeasurement probabilities:")
for i, label in enumerate(["00", "01", "10", "11"]):
    print(f"  P({label}) = {abs(product[i])**2:.4f}")

# Three-qubit state
print("\n=== Three-Qubit State: |+> (x) |0> (x) |1> ===\n")
three_qubit = tensor_product(tensor_product(ket_plus, ket_0), ket_1)
print(f"State vector (8 components): {three_qubit}")
print(f"Non-zero components:")
for i in range(8):
    if abs(three_qubit[i]) > 1e-10:
        label = format(i, '03b')
        print(f"  |{label}> : amplitude = {three_qubit[i]:.4f}, "
              f"P = {abs(three_qubit[i])**2:.4f}")
```

---

## 6. N큐비트 상태 공간

### 6.1 지수적 증가

$n$개의 큐비트에 대해 상태 공간은 $\mathbb{C}^{2^n}$입니다. 일반 상태는 $2^n$개의 복소 진폭을 필요로 합니다:

$$|\psi\rangle = \sum_{x=0}^{2^n - 1} \alpha_x |x\rangle$$

여기서 $|x\rangle$은 정수 $x$의 이진 표현에 해당하는 계산 기저 상태입니다.

정규화 조건:

$$\sum_{x=0}^{2^n - 1} |\alpha_x|^2 = 1$$

### 6.2 능력과 도전

이 지수적 스케일링은 양날의 검입니다:

**능력**: $n$큐비트 양자 컴퓨터는 원칙적으로 $2^n$개의 진폭을 동시에 조작할 수 있습니다. 50큐비트 시스템은 $\sim 10^{15}$개의 진폭을 포함합니다 — 어떤 고전 컴퓨터도 효율적으로 추적할 수 없는 수입니다.

**도전**: $2^n$개의 진폭 모두를 단순히 읽어낼 수는 없습니다. 측정은 상태를 단일 기저 상태로 붕괴시켜 오직 $n$개의 고전 비트만 산출합니다. 양자 알고리즘 설계의 기술은 측정할 때 *유용한* 답이 높은 확률을 가지도록 간섭을 배열하는 것입니다.

### 6.3 분리 가능 상태 vs 얽힘 상태

$n$큐비트 상태가 **분리 가능(separable)** (곱 상태)하다는 것은 다음과 같이 쓸 수 있다는 의미입니다:

$$|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle$$

$n$큐비트의 분리 가능 상태는 $2^n$이 아닌 오직 $2n$개의 복소수(큐비트당 2개)로 기술됩니다. 나머지 상태들 — 힐베르트 공간의 대다수 — 은 **얽힘 상태(entangled)**입니다.

얽힘은 양자 컴퓨팅을 고전 컴퓨팅보다 강력하게 만드는 자원입니다. [레슨 5](05_Entanglement_and_Bell_States.md)에서 자세히 공부할 것입니다.

```python
import numpy as np

# Exploring the exponential growth of quantum state spaces

print("=== Quantum State Space Scaling ===\n")
print(f"{'Qubits':>8} {'Dimensions':>15} {'Classical bits':>16} {'Memory (complex128)':>22}")
print("-" * 65)

for n in range(1, 21):
    dim = 2**n
    # Each complex128 is 16 bytes (8 bytes real + 8 bytes imaginary)
    memory_bytes = dim * 16
    if memory_bytes < 1024:
        mem_str = f"{memory_bytes} B"
    elif memory_bytes < 1024**2:
        mem_str = f"{memory_bytes/1024:.1f} KB"
    elif memory_bytes < 1024**3:
        mem_str = f"{memory_bytes/1024**2:.1f} MB"
    elif memory_bytes < 1024**4:
        mem_str = f"{memory_bytes/1024**3:.1f} GB"
    else:
        mem_str = f"{memory_bytes/1024**4:.1f} TB"
    print(f"{n:>8} {dim:>15,} {n:>16} {mem_str:>22}")

# Practical demonstration: create and manipulate small multi-qubit states
print("\n=== Separable vs Entangled States (2 qubits) ===\n")

# Separable state: |+>|0> = (|00> + |10>)/sqrt(2)
separable = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

# Entangled state (Bell state): (|00> + |11>)/sqrt(2)
# Cannot be written as |a> (x) |b> for any single-qubit states |a>, |b>
entangled = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

def check_separability(state_2q):
    """
    Check if a 2-qubit state is separable by attempting to factorize it.

    Why this approach? A 2-qubit state |psi> = a|00> + b|01> + c|10> + d|11>
    is separable if and only if ad = bc (the matrix [[a,b],[c,d]] has rank 1).
    This is a necessary and sufficient condition.
    """
    # Reshape into 2x2 matrix
    M = state_2q.reshape(2, 2)
    # Check rank via determinant (rank 1 means det = 0)
    det = np.linalg.det(M)
    is_separable = np.isclose(abs(det), 0)
    return is_separable, abs(det)

for name, state in [("Separable |+>|0>", separable), ("Entangled (Bell)", entangled)]:
    is_sep, det = check_separability(state)
    print(f"{name}:")
    print(f"  State: {state}")
    print(f"  |det| = {det:.6f}")
    print(f"  Separable: {is_sep}")
    print()
```

---

## 7. 연습문제

### 연습문제 1: 블로흐 구 대응

다음 각 상태에 대해 블로흐 구 좌표 $(\theta, \phi)$와 직교 좌표 $(x, y, z)$를 계산하세요:

a) $|\psi\rangle = \cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$
b) $|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{e^{i\pi/3}}{\sqrt{2}}|1\rangle$
c) $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$

이 레슨에서 제공된 Python 코드를 사용하여 답을 검증하세요.

### 연습문제 2: 위상 구별

상태 $|\psi_1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$과 $|\psi_2\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$를 고려합니다.

a) 두 상태에 대해 $Z$-기저($\{|0\rangle, |1\rangle\}$)에서의 측정 확률을 계산하세요. 이 기저에서 구별 가능한가요?
b) 두 상태에 대해 $X$-기저($\{|+\rangle, |-\rangle\}$)에서의 측정 확률을 계산하세요. 구별 가능한가요?
c) 두 상태가 최대로 다른 결과를 주는 측정 기저를 찾으세요. (힌트: 블로흐 구를 생각해보세요.)

### 연습문제 3: 텐서곱

a) $|+\rangle \otimes |-\rangle$을 4성분 벡터로 계산하세요.
b) $|-\rangle \otimes |+\rangle$을 4성분 벡터로 계산하세요. (a)와 같은가요?
c) 상태 $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$을 두 단일 큐비트 상태의 텐서곱으로 쓰세요.
d) 상태 $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$이 두 단일 큐비트 상태의 텐서곱으로 쓰여질 수 없음을 증명하세요. (힌트: 위 코드의 행렬식 기준 사용.)

### 연습문제 4: 상태 공간 계산

a) 일반 3큐비트 상태를 지정하는 데 몇 개의 실수 매개변수가 필요한가요? (정규화와 전역 위상을 고려하세요.)
b) 분리 가능한 3큐비트 상태에는 몇 개의 실수 매개변수가 필요한가요?
c) 분리 가능 상태가 "사용하는" 매개변수 공간의 비율은 얼마인가요? 얽힘에 대해 무엇을 말해주나요?

### 연습문제 5: 블로흐 구 궤적

다음을 수행하는 Python 코드를 작성하세요:
a) 블로흐 구 위에 균일하게 분포된 100개의 무작위 큐비트 상태를 생성합니다 (힌트: $\theta$는 균일하게 분포되지 않아야 합니다 — $\cos\theta$를 $[-1, 1]$에서 균일하게, $\phi$를 $[0, 2\pi)$에서 균일하게 사용하세요).
b) 각 상태에 대해 세 기저($X$, $Y$, $Z$) 모두에서의 측정 확률을 계산합니다.
c) 모든 상태에 대해 $P(0)_Z + P(0)_X + P(0)_Y$의 합이 일정하지 않지만 특정 범위 내에 있음을 검증합니다. 그 범위는 무엇인가요?

---

[<- 이전: 양자역학 입문](01_Quantum_Mechanics_Primer.md) | [다음: 양자 게이트 ->](03_Quantum_Gates.md)

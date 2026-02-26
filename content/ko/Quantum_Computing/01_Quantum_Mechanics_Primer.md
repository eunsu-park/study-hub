# 레슨 1: 양자역학 입문

| [다음: 큐비트와 블로흐 구 ->](02_Qubits_and_Bloch_Sphere.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 이중 슬릿 실험을 통해 파동-입자 이중성(wave-particle duality)과 그 함의를 설명할 수 있다
2. 진폭(amplitude) 기반 언어를 사용하여 중첩 원리(superposition principle)를 올바르게 기술할 수 있다
3. 보른 규칙(Born rule)을 적용하여 양자 상태로부터 측정 확률을 계산할 수 있다
4. 디랙(브라-켓) 표기법(Dirac bra-ket notation)을 사용하여 양자 상태를 읽고 쓸 수 있다
5. 복소 진폭(complex amplitude)을 다루고 그로부터 확률을 계산할 수 있다
6. 힐베르트 공간(Hilbert space)이 양자 컴퓨팅의 수학적 무대임을 이해할 수 있다
7. 고전 확률적 혼합(classical probabilistic mixture)과 진정한 양자 중첩(quantum superposition)을 구별할 수 있다

---

양자 컴퓨팅은 원자, 전자, 광자를 지배하는 물리 법칙인 양자역학(quantum mechanics)을 활용하여 근본적으로 새로운 방식으로 정보를 처리합니다. 양자 알고리즘을 이해하거나 양자 회로를 구성하기 전에, 먼저 양자 현상이 우리의 일상적인 고전적 경험과 어떻게 다른지를 만드는 핵심 원리들을 내면화해야 합니다. 이 원리들은 단순한 수학적 추상이 아니라, 가장 작은 규모에서 우주가 어떻게 작동하는지에 대해 실험적으로 검증된 사실입니다.

이 레슨은 앞으로 이어질 모든 내용의 개념적·수학적 토대를 마련합니다. 고전적 직관에 도전하는 아이디어들을 만나게 됩니다: 파동처럼 행동하는 입자, 가능성들의 조합으로 존재하는 상태, 그리고 그 가능성들을 돌이킬 수 없이 확정적 결과로 붕괴시키는 측정. 이러한 아이디어들, 특히 이를 기술하는 수학적 틀에 익숙해지는 것이 이를 계산에 활용하기 위한 필수 전제입니다.

> **비유:** 공중에서 회전하는 동전은 중첩 상태(superposition)의 큐비트(qubit)와 같습니다 — 앞면이면서 동시에 뒷면인 것이 아니라, 어느 쪽이든 될 수 있는 잠재성을 가지고 있는 것입니다. 동전이 착지할 때(측정)만 확정됩니다. 핵심적인 양자적 반전은 이 "회전"이 진폭(amplitude)이라 불리는 복소수로 기술되며, 이 진폭들이 서로 간섭(interference)하여 달리라면 가능했을 결과를 때로는 상쇄시킨다는 점입니다.

## 목차

1. [파동-입자 이중성](#1-파동-입자-이중성)
2. [중첩 원리](#2-중첩-원리)
3. [측정 공준과 보른 규칙](#3-측정-공준과-보른-규칙)
4. [디랙 표기법](#4-디랙-표기법)
5. [복소 진폭과 확률](#5-복소-진폭과-확률)
6. [양자 컴퓨팅을 위한 힐베르트 공간](#6-양자-컴퓨팅을-위한-힐베르트-공간)
7. [고전 확률 vs 양자 확률](#7-고전-확률-vs-양자-확률)
8. [연습문제](#8-연습문제)

---

## 1. 파동-입자 이중성

20세기 물리학의 가장 놀라운 발견 중 하나는, 우리가 "입자"라고 생각하는 것들 — 전자, 광자, 심지어 분자 전체 — 이 파동과 같은 행동을 보일 수 있다는 것입니다.

### 1.1 이중 슬릿 실험

이중 슬릿 실험(double-slit experiment)은 양자역학의 가장 전형적인 증명입니다. 실험 구성은 다음과 같습니다:

1. **소스(Source)**: 장치가 입자(예: 전자)를 하나씩 장벽을 향해 방출합니다.
2. **장벽(Barrier)**: 장벽에는 A와 B라는 두 개의 좁은 슬릿이 있습니다.
3. **스크린(Screen)**: 장벽 뒤에 검출 스크린이 각 입자가 도달하는 위치를 기록합니다.

**고전적 예상**: 입자들이 작은 당구공이라면, 스크린에 각 슬릿 뒤에 두 개의 군집 충돌 지점을 예상할 것입니다. 전체 패턴은 개별 슬릿 패턴의 합이 됩니다.

**양자적 현실**: 대신, 우리는 *간섭 패턴(interference pattern)* — 스크린 전체에 걸쳐 밝고 어두운 줄무늬가 교대로 나타나는 패턴 — 을 관찰합니다. 이 패턴은 두 슬릿을 통과하며 보강 간섭(constructive, 밝음)과 상쇄 간섭(destructive, 어두움)을 일으키는 *파동*의 특징입니다.

놀라운 점: 이 패턴은 입자를 *하나씩* 보낼 때도 나타납니다. 각 개별 입자는 스크린의 특정 지점에 도달하지만(입자적 성질), 많은 시도를 통해 통계적 분포가 간섭 패턴(파동적 성질)을 형성합니다.

### 1.2 관측 시 붕괴

슬릿에 검출기를 설치하여 각 입자가 어느 슬릿을 통과하는지 확인하면, 간섭 패턴이 *사라집니다*. 대신 고전적인 두 군집 패턴을 얻게 됩니다. "어느 경로"에 대한 정보를 획득하는 행위가 양자 간섭을 파괴합니다.

이것은 측정의 물리적 교란에 관한 것이 아닙니다(흔한 오해). 이는 양자역학의 근본적인 특성입니다: 경로에 대한 정보와 간섭은 상보적(complementary)이며 — 둘 다 동시에 가질 수 없습니다.

### 1.3 수학적 기술

입자의 상태를 *파동 함수(wave function)* $\psi(x)$로 기술합니다. 두 슬릿이 모두 열려 있을 때:

$$\psi(x) = \psi_A(x) + \psi_B(x)$$

위치 $x$에서 입자를 검출할 확률은:

$$P(x) = |\psi(x)|^2 = |\psi_A(x) + \psi_B(x)|^2$$

이를 전개하면:

$$P(x) = |\psi_A(x)|^2 + |\psi_B(x)|^2 + 2\,\text{Re}[\psi_A^*(x)\,\psi_B(x)]$$

세 번째 항이 *간섭항(interference term)*입니다. 이 항은 양수(보강)이거나 음수(상쇄)일 수 있어 특징적인 패턴을 만듭니다. 이 항에는 고전적 유사물이 없습니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating double-slit interference pattern
# We model each slit as a point source of waves

x = np.linspace(-5, 5, 1000)   # Screen positions
d = 1.0    # Slit separation
lam = 0.5  # Wavelength (de Broglie wavelength of the particle)
L = 10.0   # Distance from slits to screen

# Wave amplitudes from each slit (simplified far-field approximation)
# Phase difference depends on path length difference
# Why: The path difference creates a position-dependent phase shift,
# which is the physical origin of the interference pattern
theta = np.arctan(x / L)
delta_phi = (2 * np.pi / lam) * d * np.sin(theta)

# Amplitudes from slit A and B (equal magnitude, phase-shifted)
psi_A = np.ones_like(x) * np.exp(1j * delta_phi / 2)
psi_B = np.ones_like(x) * np.exp(-1j * delta_phi / 2)

# Quantum case: add amplitudes THEN square
psi_total = psi_A + psi_B
P_quantum = np.abs(psi_total)**2

# Classical case: add probabilities (no interference)
P_classical = np.abs(psi_A)**2 + np.abs(psi_B)**2

print("Quantum interference pattern (first 5 positions):")
print(f"  Positions: {x[:5]}")
print(f"  Probabilities: {P_quantum[:5]}")
print(f"\nMax quantum probability: {P_quantum.max():.4f}")
print(f"Min quantum probability: {P_quantum.min():.4f}")
print(f"Classical probability (constant): {P_classical[0]:.4f}")
print(f"\nKey insight: Quantum probabilities vary from {P_quantum.min():.4f} to "
      f"{P_quantum.max():.4f}")
print(f"while classical probability is uniformly {P_classical[0]:.4f}")
print("The interference term creates the oscillation!")
```

### 1.4 핵심 요점

파동-입자 이중성은 양자 객체가 "파동"이나 "입자" 어느 한 범주에 깔끔하게 맞지 않음을 말해줍니다. 이들은 파동처럼 전파되고 입자처럼 검출되는 진폭으로 기술됩니다. 양자 컴퓨팅에 있어 핵심 통찰은: **양자 정보는 진폭에 의해 전달되며, 이 진폭들은 서로 간섭할 수 있다**는 것입니다.

---

## 2. 중첩 원리

중첩(superposition)은 양자역학의 가장 근본적인 원리이며 양자 컴퓨팅 능력의 초석입니다.

### 2.1 중첩이 실제로 의미하는 것

흔하지만 오해를 불러일으키는 설명은 양자 시스템이 "여러 상태에 동시에 있다"고 말합니다. 이는 정확하지 않습니다. 더 정확한 설명은:

> 중첩 상태에 있는 양자 시스템은 각각의 가능한 결과에 *진폭(amplitude)*이 할당됩니다. 이 진폭들은 결과의 확률을 결정하지만, 시스템이 동시에 그 모든 결과 "안에" 있는 것은 아닙니다. 시스템은 잠재성(potentiality)을 인코딩한 *하나의* 양자 상태에 있습니다.

$|0\rangle$ 또는 $|1\rangle$로 측정될 수 있는 큐비트를 생각해봅시다. 일반 상태는:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

여기서 $\alpha$와 $\beta$는 복소수입니다. 이 표현이 의미하는 것:
- 큐비트는 *하나의* 확정적인 양자 상태 $|\psi\rangle$에 있습니다
- 측정하면, 결과 $|0\rangle$의 확률은 $|\alpha|^2$이고 $|1\rangle$의 확률은 $|\beta|^2$입니다
- 측정 전에 큐비트는 "0이면서 1"이 아닙니다 — 그 어느 것과도 근본적으로 다른 상태에 있습니다

### 2.2 중첩 vs 고전적 확률

고전적 동전은 확률적으로 기술할 수 있습니다: "앞면 50%, 뒷면 50%." 이것이 $|\alpha|^2 = |\beta|^2 = 1/2$인 큐비트와 어떻게 다를까요?

결정적 차이는 **간섭(interference)**입니다. 고전적 확률은 항상 더해집니다:

$$P_{\text{classical}}(\text{outcome}) = P_A(\text{outcome}) + P_B(\text{outcome}) \geq 0$$

양자 진폭은 상쇄될 수 있습니다:

$$\alpha_{\text{total}} = \alpha_A + \alpha_B \quad \Rightarrow \quad P = |\alpha_A + \alpha_B|^2$$

이는 양자 확률이 개별 확률을 더한 것보다 *작을* 수 있음을 의미합니다 — 상쇄 간섭은 각 개별 경로가 허용하더라도 특정 결과를 불가능하게 만들 수 있습니다.

### 2.3 양자역학의 선형성

$|\psi_1\rangle$과 $|\psi_2\rangle$이 유효한 양자 상태라면, 임의의 선형 결합:

$$|\psi\rangle = c_1|\psi_1\rangle + c_2|\psi_2\rangle$$

도 (정규화 후) 유효한 양자 상태입니다. 이것이 수학적 형태의 중첩 원리입니다. 이는 양자 진화를 지배하는 슈뢰딩거 방정식(Schrodinger equation)이 *선형*이기 때문에 발생합니다.

```python
import numpy as np

# Demonstrating superposition and interference
# Two quantum states represented as column vectors

# |0> and |1> basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# A superposition state: |+> = (|0> + |1>) / sqrt(2)
# Why divide by sqrt(2)? Normalization ensures probabilities sum to 1
ket_plus = (ket_0 + ket_1) / np.sqrt(2)

# Another superposition: |-> = (|0> - |1>) / sqrt(2)
# The MINUS sign is the key difference -- it changes the relative phase
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("State |+>:", ket_plus)
print("State |->:", ket_minus)

# Both have the same measurement probabilities in the {|0>, |1>} basis!
print(f"\n|+> probabilities: P(0) = {abs(ket_plus[0])**2:.2f}, "
      f"P(1) = {abs(ket_plus[1])**2:.2f}")
print(f"|-> probabilities: P(0) = {abs(ket_minus[0])**2:.2f}, "
      f"P(1) = {abs(ket_minus[1])**2:.2f}")

# But they are DIFFERENT states! Adding them:
# |+> + |-> = sqrt(2)|0> -- interference eliminated |1>!
sum_state = ket_plus + ket_minus
sum_normalized = sum_state / np.linalg.norm(sum_state)
print(f"\nAfter interference (|+> + |->): {sum_state}")
print(f"Normalized: {sum_normalized}")
print(f"Probabilities: P(0) = {abs(sum_normalized[0])**2:.2f}, "
      f"P(1) = {abs(sum_normalized[1])**2:.2f}")
print("\nThe |1> component completely canceled out -- destructive interference!")
```

---

## 3. 측정 공준과 보른 규칙

### 3.1 측정 문제

양자역학에서 측정은 미리 존재하는 값을 수동적으로 읽는 행위가 아닙니다. 시스템의 상태를 *변화*시키는 능동적인 과정입니다.

**측정 전**: 시스템은 $|\psi\rangle = \sum_i c_i |i\rangle$ 상태, 즉 가능한 결과 $|i\rangle$의 중첩 상태에 있습니다.

**측정 중**: 시스템이 가능한 결과 중 하나인 $|i\rangle$로 "붕괴(collapse)"됩니다.

**측정 후**: 시스템은 확정적으로 상태 $|i\rangle$에 있습니다. 중첩은 파괴됩니다.

### 3.2 보른 규칙

1926년 막스 보른(Max Born)이 제안한 보른 규칙(Born rule)은 각 결과의 확률을 알려줍니다:

$$P(i) = |c_i|^2 = |\langle i|\psi\rangle|^2$$

여기서:
- $c_i$는 상태 $|\psi\rangle$에서 결과 $|i\rangle$의 진폭(amplitude)입니다
- $|c_i|^2$는 복소수 $c_i$의 모듈루스 제곱(modulus squared)을 의미합니다
- $\langle i|\psi\rangle$는 내적(inner product)입니다 (다음 섹션에서 정확히 정의합니다)

**정규화(Normalization)**: 확률의 합이 1이어야 하므로:

$$\sum_i |c_i|^2 = 1$$

### 3.3 예시: 큐비트 측정

상태 $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$을 고려합니다.

- $P(0) = \left|\frac{1}{\sqrt{3}}\right|^2 = \frac{1}{3} \approx 33.3\%$
- $P(1) = \left|\sqrt{\frac{2}{3}}\right|^2 = \frac{2}{3} \approx 66.7\%$
- 확인: $\frac{1}{3} + \frac{2}{3} = 1$ (정규화됨)

측정 후 결과 $|0\rangle$을 얻으면, 상태는 이제 확정적으로 $|0\rangle$입니다. $|1\rangle$에 대한 진폭 $\sqrt{2/3}$은 돌이킬 수 없이 사라집니다.

### 3.4 반복 측정

같은 큐비트를 즉시 (같은 기저에서) 다시 측정하면, 항상 같은 결과를 얻습니다. 상태가 붕괴되었고 다른 결과를 만들 중첩이 더 이상 없기 때문입니다. 이를 측정의 *멱등성(idempotency)*이라 합니다.

```python
import numpy as np

# Simulating quantum measurement with the Born rule

def measure_qubit(state, num_shots=10000):
    """
    Simulate measuring a qubit state many times.

    Why simulate many shots? A single measurement gives one outcome.
    To verify the Born rule probabilities, we need statistics from
    many identical preparations and measurements.
    """
    # Extract probabilities using the Born rule: P(i) = |c_i|^2
    probabilities = np.abs(state)**2

    # Simulate measurements by sampling from the probability distribution
    # Why use np.random.choice? It samples from discrete distribution,
    # which is exactly what quantum measurement does
    outcomes = np.random.choice(len(state), size=num_shots, p=probabilities)

    # Count occurrences of each outcome
    counts = np.bincount(outcomes, minlength=len(state))
    frequencies = counts / num_shots

    return counts, frequencies

# State: |psi> = (1/sqrt(3))|0> + sqrt(2/3)|1>
psi = np.array([1/np.sqrt(3), np.sqrt(2/3)], dtype=complex)

# Verify normalization
norm = np.sum(np.abs(psi)**2)
print(f"State: ({psi[0]:.4f})|0> + ({psi[1]:.4f})|1>")
print(f"Normalization check: |alpha|^2 + |beta|^2 = {norm:.6f}")

# Theoretical probabilities
print(f"\nTheoretical: P(0) = {abs(psi[0])**2:.4f}, P(1) = {abs(psi[1])**2:.4f}")

# Simulated measurements
np.random.seed(42)  # For reproducibility
counts, freqs = measure_qubit(psi, num_shots=100000)
print(f"Simulated (100000 shots): P(0) = {freqs[0]:.4f}, P(1) = {freqs[1]:.4f}")
print(f"Counts: |0> = {counts[0]}, |1> = {counts[1]}")

# Demonstrating collapse: after measuring |0>, re-measuring always gives |0>
print("\n--- Post-measurement collapse ---")
collapsed_state = np.array([1, 0], dtype=complex)  # Collapsed to |0>
_, collapsed_freqs = measure_qubit(collapsed_state, num_shots=1000)
print(f"After collapse to |0>, re-measurement: P(0) = {collapsed_freqs[0]:.4f}, "
      f"P(1) = {collapsed_freqs[1]:.4f}")
```

---

## 4. 디랙 표기법

폴 디랙(Paul Dirac)은 양자역학을 위한 우아한 표기법을 도입했으며, 이는 양자 컴퓨팅에서 보편적으로 사용됩니다. 처음에는 낯설어 보일 수 있지만, 일단 내면화하면 매우 실용적입니다.

### 4.1 켓, 브라, 브래킷

| 표기법 | 이름 | 수학적 객체 | 열/행 |
|--------|------|-------------|-------|
| $\|v\rangle$ | "켓 v" | 열 벡터(Column vector) | $\begin{pmatrix} v_1 \\ v_2 \\ \vdots \end{pmatrix}$ |
| $\langle v\|$ | "브라 v" | 행 벡터(Row vector, 켤레 전치) | $(v_1^*, v_2^*, \ldots)$ |
| $\langle u\|v\rangle$ | "브래킷" 또는 "내적(inner product)" | 복소수 | $\sum_i u_i^* v_i$ |
| $\|u\rangle\langle v\|$ | "외적(outer product)" | 행렬 | $u_i v_j^*$ |

이름은 "bracket(괄호)"이라는 단어를 "bra-ket"으로 나눈 것에서 유래합니다.

### 4.2 계산 기저

단일 큐비트의 경우, 두 계산 기저 상태(computational basis state)는:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

이들은 정규직교(orthonormal)합니다:
- $\langle 0|0\rangle = 1$, $\langle 1|1\rangle = 1$ (정규화됨)
- $\langle 0|1\rangle = 0$, $\langle 1|0\rangle = 0$ (직교함)

일반 큐비트 상태는:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$

### 4.3 내적과 진폭

상태 $|\psi\rangle$에서 결과 $|i\rangle$의 진폭은 내적 $\langle i|\psi\rangle$입니다:

$$\langle 0|\psi\rangle = \alpha, \quad \langle 1|\psi\rangle = \beta$$

그러면 보른 규칙은 다음과 같이 줍니다:

$$P(0) = |\langle 0|\psi\rangle|^2 = |\alpha|^2, \quad P(1) = |\langle 1|\psi\rangle|^2 = |\beta|^2$$

### 4.4 외적과 사영 연산자

외적 $|i\rangle\langle i|$는 *사영 연산자(projection operator)*를 만듭니다:

$$|0\rangle\langle 0| = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

이를 상태에 적용하면 $|0\rangle$ 방향의 성분을 추출합니다:

$$(|0\rangle\langle 0|)|\psi\rangle = |0\rangle(\langle 0|\psi\rangle) = \alpha|0\rangle$$

**완전성 관계(completeness relation)** (항등원의 분해)는 다음과 같습니다:

$$|0\rangle\langle 0| + |1\rangle\langle 1| = I$$

이는 양자 컴퓨팅 증명에서 항상 사용되는 근본적인 항등식입니다.

```python
import numpy as np

# Dirac notation in practice using numpy

# Basis kets as column vectors
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Bras are conjugate transposes of kets
# Why conjugate transpose (not just transpose)? Because quantum amplitudes
# are complex, and the inner product must be positive-definite
bra_0 = ket_0.conj().T
bra_1 = ket_1.conj().T

# Inner products (brackets)
print("Inner products:")
print(f"  <0|0> = {(bra_0 @ ket_0)[0,0]}")
print(f"  <1|1> = {(bra_1 @ ket_1)[0,0]}")
print(f"  <0|1> = {(bra_0 @ ket_1)[0,0]}")
print(f"  <1|0> = {(bra_1 @ ket_0)[0,0]}")

# A general state
alpha = 1/np.sqrt(3)
beta = np.sqrt(2/3) * np.exp(1j * np.pi/4)  # Complex amplitude with phase!
psi = alpha * ket_0 + beta * ket_1
print(f"\n|psi> = ({alpha:.4f})|0> + ({beta:.4f})|1>")

# Extract amplitudes via inner products
amp_0 = (bra_0 @ psi)[0,0]
amp_1 = (bra_1 @ psi)[0,0]
print(f"<0|psi> = {amp_0:.4f}")
print(f"<1|psi> = {amp_1:.4f}")

# Born rule probabilities
print(f"\nP(0) = |<0|psi>|^2 = {abs(amp_0)**2:.4f}")
print(f"P(1) = |<1|psi>|^2 = {abs(amp_1)**2:.4f}")
print(f"Sum = {abs(amp_0)**2 + abs(amp_1)**2:.4f}")

# Outer products (projectors)
proj_0 = ket_0 @ bra_0  # |0><0|
proj_1 = ket_1 @ bra_1  # |1><1|
print(f"\n|0><0| = \n{proj_0}")
print(f"\n|1><1| = \n{proj_1}")

# Completeness relation: |0><0| + |1><1| = I
identity = proj_0 + proj_1
print(f"\n|0><0| + |1><1| = \n{identity}")
print(f"Is identity? {np.allclose(identity, np.eye(2))}")
```

---

## 5. 복소 진폭과 확률

### 5.1 왜 복소수인가?

양자 진폭은 $i = \sqrt{-1}$인 복소수 $\alpha = a + bi$입니다. 이것은 수학적 편의가 아닙니다 — 물리적 필연입니다. 복소 진폭은 다음을 위해 필요합니다:

1. **간섭 기술**: 실수는 보강적으로만 더해질 수 있습니다. 복소수는 서로를 상쇄시킬 수 있습니다.
2. **위상 정보 인코딩**: 진폭 사이의 상대 위상(relative phase)은 간섭에 영향을 미치지만 개별 확률에는 영향을 미치지 않습니다.
3. **유니터리 진화 표현**: 양자 게이트는 유니터리 행렬(unitary matrix)이며, 이는 자연스럽게 복소수 체(complex number field)에 속합니다.

### 5.2 모듈루스와 위상

임의의 복소수는 극형식으로 쓸 수 있습니다:

$$\alpha = |\alpha| \cdot e^{i\phi} = |\alpha|(\cos\phi + i\sin\phi)$$

여기서:
- $|\alpha| = \sqrt{a^2 + b^2}$는 **모듈루스(modulus)** (크기)입니다
- $\phi = \text{atan2}(b, a)$는 **위상(phase)** (각도)입니다

확률은 $|\alpha|^2$이며, 이는 위상이 아닌 모듈루스에만 의존합니다. 그렇다면 위상은 왜 중요할까요? 진폭이 (게이트나 간섭을 통해) *결합*될 때, 그들의 위상이 보강적으로 더해질지 상쇄적으로 더해질지를 결정하기 때문입니다.

### 5.3 복소 진폭의 간섭

두 진폭이 결합되는 것을 고려합니다:

$$\alpha_{\text{total}} = \alpha_1 + \alpha_2 = |\alpha_1|e^{i\phi_1} + |\alpha_2|e^{i\phi_2}$$

결과 확률:

$$P = |\alpha_{\text{total}}|^2 = |\alpha_1|^2 + |\alpha_2|^2 + 2|\alpha_1||\alpha_2|\cos(\phi_1 - \phi_2)$$

간섭항 $2|\alpha_1||\alpha_2|\cos(\phi_1 - \phi_2)$은 *위상 차이* $\phi_1 - \phi_2$에 의존합니다:
- **보강 간섭** ($\phi_1 - \phi_2 = 0$): 진폭이 강화되어 $P$가 최대화됩니다
- **상쇄 간섭** ($\phi_1 - \phi_2 = \pi$): 진폭이 상쇄되어 $P$가 최소화됩니다
- **부분 간섭**: 다른 위상 차이는 중간 결과를 줍니다

```python
import numpy as np

# Demonstrating how phase affects interference

def interference_probability(mag1, phase1, mag2, phase2):
    """
    Compute the probability when two complex amplitudes combine.

    Why separate magnitude and phase? This makes the physics transparent:
    magnitudes set the 'strength' of each path, while phases determine
    whether the paths reinforce or cancel each other.
    """
    alpha1 = mag1 * np.exp(1j * phase1)
    alpha2 = mag2 * np.exp(1j * phase2)
    total = alpha1 + alpha2
    return np.abs(total)**2

# Two equal-magnitude amplitudes with varying phase difference
mag = 1 / np.sqrt(2)  # Each amplitude contributes 50% individually
phases = np.linspace(0, 2*np.pi, 8)

print("Phase difference vs Combined probability")
print("=" * 50)
print(f"Individual probability of each path: {mag**2:.4f}")
print(f"Classical sum (no interference): {2*mag**2:.4f}")
print()

for phi in phases:
    P = interference_probability(mag, 0, mag, phi)
    interference_type = ""
    if np.isclose(phi % (2*np.pi), 0):
        interference_type = "(constructive)"
    elif np.isclose(phi % (2*np.pi), np.pi):
        interference_type = "(destructive)"
    print(f"  Phase diff = {phi/np.pi:.2f}*pi  =>  P = {P:.4f}  {interference_type}")

print(f"\nKey insight: Same individual probabilities, but combined probability")
print(f"ranges from 0.00 to 2.00 depending on PHASE!")
print(f"This is the power quantum computing exploits.")
```

### 5.4 전역 위상 vs 상대 위상

중요한 구분:

- **전역 위상(Global phase)**: 전체 상태에 $e^{i\theta}$를 곱해도 관측 가능한 효과가 없습니다. 상태 $|\psi\rangle$와 $e^{i\theta}|\psi\rangle$는 $|e^{i\theta}\alpha|^2 = |\alpha|^2$이기 때문에 물리적으로 동일합니다.

- **상대 위상(Relative phase)**: 중첩의 성분들 *사이*의 위상은 물리적으로 의미가 있습니다. 상태 $|0\rangle + |1\rangle$과 $|0\rangle + e^{i\phi}|1\rangle$은 $\phi \neq 0$일 때 물리적으로 다릅니다.

이 구분은 [레슨 2: 큐비트와 블로흐 구](02_Qubits_and_Bloch_Sphere.md)에서 깊이 탐구할 것입니다.

---

## 6. 양자 컴퓨팅을 위한 힐베르트 공간

### 6.1 힐베르트 공간이란?

**힐베르트 공간(Hilbert space)**은 내적(inner product)이 갖추어진 완전 벡터 공간(complete vector space)입니다. 양자 컴퓨팅에서는 *유한 차원* 복소 힐베르트 공간을 다루는데, 이는 일반 양자역학의 무한 차원 공간보다 훨씬 단순합니다.

단일 큐비트의 경우, 힐베르트 공간은 $\mathbb{C}^2$ — 2성분 복소 벡터 전체의 공간 — 입니다. $n$개의 큐비트의 경우 $\mathbb{C}^{2^n}$입니다.

### 6.2 필요한 성질

우리의 힐베르트 공간의 핵심 성질:

| 성질 | 정의 | 양자 컴퓨팅 관련성 |
|------|------|-------------------|
| **벡터 공간(Vector space)** | 상태를 더하고 스칼라를 곱할 수 있음 | 중첩 원리 |
| **복소 체(Complex field)** | 스칼라가 복소수 | 위상과 간섭 |
| **내적(Inner product)** | $\langle u\|v\rangle \in \mathbb{C}$ | 보른 규칙을 통한 확률 |
| **완전성(Completeness)** | 모든 코시 수열이 수렴 | 유한 차원에서 자동 |
| **유한 차원(Finite dimension)** | $n$개 큐비트에 대해 $\dim = 2^n$ | 소규모 시스템에서 다루기 가능 |

### 6.3 정규직교 기저

정규직교 기저(orthonormal basis) $\{|e_i\rangle\}$는 다음을 만족합니다:

$$\langle e_i|e_j\rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

$n$개의 큐비트에 대해 계산 기저는 $2^n$개의 상태로 구성됩니다. 2개의 큐비트의 경우:

$$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$

이것들이 우리가 측정하는 상태이며, 임의의 양자 상태는 이들의 중첩으로 쓸 수 있습니다.

### 6.4 지수적 증가가 중요한 이유

힐베르트 공간의 차원은 큐비트 수에 따라 *지수적으로* 증가합니다. 이 지수적 증가가 양자 컴퓨팅의 잠재적 능력의 원천입니다:

| 큐비트 수 ($n$) | 차원 ($2^n$) | 필요한 복소수 수 |
|:---:|:---:|:---:|
| 1 | 2 | 2 |
| 10 | 1,024 | 1,024 |
| 20 | ~100만 | ~100만 |
| 50 | ~$10^{15}$ | ~$10^{15}$ |
| 300 | $2^{300} > 10^{90}$ | 우주의 원자 수보다 많음 |

300개의 큐비트 양자 시스템은 원칙적으로도 어떤 고전 컴퓨터로도 완전히 시뮬레이션할 수 없습니다. 이것이 양자 컴퓨팅이 잠재적으로 고전적 능력을 넘어서는 문제를 해결할 수 있는 근본적인 이유입니다.

```python
import numpy as np

# Exploring Hilbert space dimensions

def describe_hilbert_space(n_qubits):
    """
    Describe the Hilbert space for n qubits.

    Why is this important? The exponential growth of state space is THE
    fundamental resource that quantum algorithms exploit. Understanding
    this scaling is essential for appreciating quantum advantage.
    """
    dim = 2**n_qubits

    # A general state needs 'dim' complex amplitudes (2*dim real numbers)
    # minus 1 for normalization, minus 1 for global phase
    real_params = 2 * dim - 2

    return dim, real_params

print("Hilbert Space Dimensions for n Qubits")
print("=" * 60)
print(f"{'Qubits':>8} {'Dimension':>15} {'Real Parameters':>18}")
print("-" * 60)

for n in [1, 2, 3, 5, 10, 20, 30]:
    dim, params = describe_hilbert_space(n)
    print(f"{n:>8} {dim:>15,} {params:>18,}")

# Demonstrate: creating a random state in n-qubit Hilbert space
n = 3
dim = 2**n
print(f"\n--- Random {n}-qubit state (dimension {dim}) ---")

# Why generate random states this way? A truly random quantum state (Haar random)
# is created by generating random complex numbers and normalizing.
rng = np.random.default_rng(42)
raw = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
state = raw / np.linalg.norm(raw)  # Normalize

print(f"State vector ({dim} complex amplitudes):")
for i in range(dim):
    # Convert index to binary string for qubit label
    label = format(i, f'0{n}b')
    prob = abs(state[i])**2
    print(f"  |{label}> : amplitude = {state[i]:.4f}, P = {prob:.4f}")

print(f"\nTotal probability: {sum(abs(state[i])**2 for i in range(dim)):.6f}")
```

---

## 7. 고전 확률 vs 양자 확률

고전 확률과 양자 확률의 구분을 명확히 하는 것은 중요합니다. 이 부분의 혼동이 양자 컴퓨팅에 관한 많은 오해의 원천이기 때문입니다.

### 7.1 고전 확률: 무지(Ignorance)

"동전이 앞면이 나올 확률이 50%"라고 말할 때, 우리는 어느 면이 나올지 *모른다*는 의미입니다. 원칙적으로, 정확한 초기 조건(힘, 각도, 공기 저항)을 알면 결과를 확실하게 예측할 수 있습니다. 고전적 확률은 우리의 *무지*를 반영합니다.

고전적 상태는 **확률 분포(probability distribution)**를 사용하여 혼합(확률적)될 수 있습니다:

$$\rho_{\text{classical}} = \{(p_1, \text{state}_1), (p_2, \text{state}_2), \ldots\}$$

### 7.2 양자 확률: 근본적(Fundamental)

양자 확률은 무지에 관한 것이 아닙니다 — *현실의 근본적인 속성*에 관한 것입니다. 양자 상태 $|\psi\rangle$에 대한 완전한 지식(이것이 가능한 최대 지식)이 있더라도, 측정 결과는 본질적으로 확률적입니다. 이것은 우리 기술이나 지식의 한계가 아니라 우주의 특성입니다.

### 7.3 복제 불가 정리 (미리보기)

양자 확률이 근본적인 것의 결과 중 하나: 알 수 없는 양자 상태를 복사할 수 없습니다. 복사할 수 있다면 많은 복사본을 만들어 모두 측정하여 상태를 완벽히 추론할 수 있어 측정 공준을 위반하게 됩니다. 이 **복제 불가 정리(no-cloning theorem)**는 양자 컴퓨팅과 양자 암호(quantum cryptography)에 심오한 함의를 가집니다 (이후 레슨에서 더 자세히 다룹니다).

### 7.4 비교 요약

| 측면 | 고전 확률 | 양자 확률 |
|------|-----------|-----------|
| **본질** | 주관적 (무지) | 객관적 (근본적) |
| **값** | 실수, $0 \leq p \leq 1$ | 복소 진폭(Complex amplitudes) |
| **결합** | 확률이 더해짐 | 진폭이 더해진 후 제곱 |
| **간섭** | 없음 | 있음 (보강 및 상쇄) |
| **복사** | 가능 | 금지됨 (복제 불가) |
| **상태** | 항상 완전히 알 수 있음 | 완전한 지식으로도 무작위 결과 발생 |

```python
import numpy as np

# Classical vs Quantum: a concrete comparison

np.random.seed(42)

print("=== Classical Coin vs Quantum Qubit ===\n")

# Scenario: Two paths (A and B) lead to a detector.
# Each path has a 50% probability of triggering the detector.

# Classical case: Probabilities add
p_A = 0.5
p_B = 0.5
p_classical = p_A + p_B  # Capped at 1 in practice, but the principle is additive
print(f"Classical: P(A) + P(B) = {p_A} + {p_B} = {min(p_classical, 1.0)}")
print(f"  (Probabilities always add -- no cancellation possible)")

# Quantum case: Amplitudes add, THEN we square
# Case 1: Same phase (constructive interference)
alpha_A = 1/np.sqrt(2)
alpha_B = 1/np.sqrt(2)
p_constructive = abs(alpha_A + alpha_B)**2
print(f"\nQuantum (constructive): |alpha_A + alpha_B|^2 = |{alpha_A:.4f} + {alpha_B:.4f}|^2 "
      f"= {p_constructive:.4f}")

# Case 2: Opposite phase (destructive interference)
alpha_A = 1/np.sqrt(2)
alpha_B = -1/np.sqrt(2)  # Note the minus sign!
p_destructive = abs(alpha_A + alpha_B)**2
print(f"Quantum (destructive): |alpha_A + alpha_B|^2 = |{alpha_A:.4f} + ({alpha_B:.4f})|^2 "
      f"= {p_destructive:.4f}")

print(f"\nSame individual probabilities ({abs(alpha_A)**2:.2f} each), but:")
print(f"  Classical result: always {min(p_A + p_B, 1.0):.2f}")
print(f"  Quantum (constructive): {p_constructive:.2f}  <- probability INCREASED")
print(f"  Quantum (destructive):  {p_destructive:.2f}  <- probability VANISHED!")
print(f"\nThis is the essence of quantum computing:")
print(f"  Arrange amplitudes so wrong answers cancel (destructive)")
print(f"  and right answers reinforce (constructive).")
```

---

## 8. 연습문제

### 연습문제 1: 정규화

큐비트가 상태 $|\psi\rangle = \frac{3}{5}|0\rangle + \frac{4}{5}e^{i\pi/3}|1\rangle$에 있습니다.

a) 이 상태가 정규화되어 있음을 검증하세요.
b) $|0\rangle$과 $|1\rangle$을 측정할 확률은 무엇인가요?
c) 위상 인수 $e^{i\pi/3}$이 계산 기저에서의 측정 확률에 영향을 미치나요?
d) 다른 기저에서 측정하면 이 위상 인수가 중요할까요? (개념적 답변으로 충분합니다.)

### 연습문제 2: 간섭 계산

두 진폭 $\alpha_1 = \frac{1}{2}$과 $\alpha_2 = \frac{1}{2}e^{i\phi}$가 결합하여 $\alpha_{\text{total}} = \alpha_1 + \alpha_2$를 줍니다.

a) $|\alpha_{\text{total}}|^2$를 $\phi$의 함수로 계산하세요.
b) 확률이 최대화되는 $\phi$ 값은 무엇인가요? 최댓값은 얼마인가요?
c) 확률이 최소화되는 $\phi$ 값은 무엇인가요? 최솟값은 얼마인가요?
d) $\phi \in [0, 2\pi]$에 대해 $P(\phi)$를 그리는 Python 스크립트를 작성하세요.

### 연습문제 3: 디랙 표기법 연습

상태 $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$이 주어졌을 때:

a) $|\psi\rangle$를 열 벡터로 쓰세요.
b) $\langle\psi|$를 행 벡터로 쓰세요.
c) $\langle\psi|\psi\rangle$를 계산하세요.
d) 사영자(projector) $|\psi\rangle\langle\psi|$를 2×2 행렬로 계산하세요.
e) $(|\psi\rangle\langle\psi|)^2 = |\psi\rangle\langle\psi|$임을 검증하세요 (이것이 사영자의 정의적 성질입니다).

### 연습문제 4: 힐베르트 공간 탐구

a) 4개의 큐비트 시스템에서 일반 상태를 기술하는 데 몇 개의 복소 진폭이 필요한가요?
b) 정규화와 전역 위상을 고려할 때 몇 개의 *실수* 매개변수가 필요한가요?
c) 무작위 4큐비트 상태를 만들고, 10,000번 측정하여 결과의 히스토그램을 그리는 Python 코드를 작성하세요. 분포가 진폭으로부터 기대하는 것과 어떻게 비교되나요?

### 연습문제 5: 개념적 이해

a) "큐비트가 동시에 0과 1이다"가 중첩에 대한 오해를 불러일으키는 설명인 이유를 자신의 말로 설명하세요. 더 정확한 설명은 무엇일까요?
b) 간섭은 왜 *복소* 진폭에서만 발생할 수 있나요? 실수 값 진폭이 상쇄 간섭을 만들 수 있을까요? (힌트: 실수가 상쇄될 수 있는지 생각해보세요.)
c) 친구가 양자 무작위성이 "동전 던지기와 같다"고 주장합니다. 양자 무작위성이 고전적 동전 던지기 무작위성과 근본적으로 다른 구체적인 두 가지 방식을 제시하세요.

---

| [다음: 큐비트와 블로흐 구 ->](02_Qubits_and_Bloch_Sphere.md)

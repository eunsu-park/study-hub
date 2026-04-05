# Lesson 12: Quantum Teleportation and Communication

[← Previous: Quantum Error Correction](11_Quantum_Error_Correction.md) | [Next: Variational Quantum Eigensolver →](13_VQE.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. State and prove the no-cloning theorem and explain its physical significance
2. Describe the quantum teleportation protocol step by step, including the role of classical communication
3. Explain why teleportation does not violate the no-signaling principle or special relativity
4. Implement superdense coding — sending 2 classical bits using 1 qubit and shared entanglement
5. Outline the BB84 quantum key distribution protocol and its security basis
6. Describe the concept of quantum repeaters for long-distance quantum communication
7. Simulate teleportation, superdense coding, and BB84 in Python

---

Quantum teleportation, first proposed by Bennett et al. in 1993, is one of the most striking and counterintuitive phenomena in quantum information theory. It allows the transfer of an unknown quantum state from one location to another, using only a shared entangled pair and two bits of classical communication. The original quantum state is destroyed in the process (satisfying no-cloning), and no physical matter or energy carrying the quantum information travels between the two locations.

Despite the science-fiction name, quantum teleportation does not allow faster-than-light communication. The classical bits required for the protocol travel at the speed of light (or slower), and without them, the receiver has only random noise. This subtle interplay between quantum entanglement and classical communication reveals deep truths about the nature of information in quantum mechanics.

> **Analogy:** Quantum teleportation is like faxing a document. The original is destroyed (no-cloning), the information travels via classical and quantum channels, and an identical copy appears at the destination. Just as a fax requires both a phone line (classical channel) and a fax machine (shared resource), teleportation requires both classical bits and a shared entangled pair.

## Table of Contents

1. [No-Cloning Theorem Revisited](#1-no-cloning-theorem-revisited)
2. [Quantum Teleportation Protocol](#2-quantum-teleportation-protocol)
3. [Why Teleportation Does Not Violate No-Signaling](#3-why-teleportation-does-not-violate-no-signaling)
4. [Superdense Coding](#4-superdense-coding)
5. [Quantum Key Distribution: BB84](#5-quantum-key-distribution-bb84)
6. [Quantum Repeaters](#6-quantum-repeaters)
7. [Python Implementation](#7-python-implementation)
8. [Exercises](#8-exercises)

---

## 1. No-Cloning Theorem Revisited

### 1.1 Formal Statement

**No-Cloning Theorem**: There is no unitary operator $U$ and no fixed ancilla state $|s\rangle$ such that for all quantum states $|\psi\rangle$:

$$U(|\psi\rangle \otimes |s\rangle) = |\psi\rangle \otimes |\psi\rangle$$

### 1.2 Proof

Suppose such a $U$ exists. For two arbitrary states $|\psi\rangle$ and $|\phi\rangle$:

$$U(|\psi\rangle|s\rangle) = |\psi\rangle|\psi\rangle$$
$$U(|\phi\rangle|s\rangle) = |\phi\rangle|\phi\rangle$$

Since $U$ is unitary, it preserves inner products:

$$\langle\psi|\phi\rangle \cdot \langle s|s\rangle = (\langle\psi|\phi\rangle)^2$$

Since $\langle s|s\rangle = 1$:

$$\langle\psi|\phi\rangle = (\langle\psi|\phi\rangle)^2$$

Let $c = \langle\psi|\phi\rangle$. Then $c = c^2$, which implies $c(c-1) = 0$, so $c = 0$ or $c = 1$.

This means $U$ can only clone states that are either identical ($c = 1$) or orthogonal ($c = 0$). It cannot clone a general, unknown quantum state. $\square$

### 1.3 Physical Significance

The no-cloning theorem has profound implications:

- **Quantum teleportation**: The original state must be destroyed during teleportation (no copies remain)
- **Quantum cryptography**: An eavesdropper cannot copy quantum key bits without disturbing them
- **Quantum error correction**: We cannot protect quantum information by simple duplication — we must use entanglement-based encoding (Lesson 11)
- **Quantum computing**: We cannot "fan out" quantum information the way we fan out classical bits

### 1.4 What *Can* Be Cloned

- **Known states**: If you know $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ (i.e., you know $\alpha$ and $\beta$), you can prepare as many copies as you wish
- **Orthogonal states**: A CNOT gate clones computational basis states: $\text{CNOT}|b\rangle|0\rangle = |b\rangle|b\rangle$ for $b \in \{0, 1\}$
- **Classical information**: Classical bits can be freely copied (classical information is represented by orthogonal states)

---

## 2. Quantum Teleportation Protocol

### 2.1 Setup

Three parties and three qubits:

- **Alice** has qubit $A$ in an unknown state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ that she wants to teleport to Bob
- **Alice** also has qubit $E_A$ (her half of an entangled pair)
- **Bob** has qubit $E_B$ (his half of the entangled pair)

The entangled pair shared by Alice and Bob is a Bell state:

$$|\Phi^+\rangle_{E_A E_B} = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

### 2.2 Initial State

The complete 3-qubit state is:

$$|\Psi_0\rangle = |\psi\rangle_A \otimes |\Phi^+\rangle_{E_A E_B}$$

$$= (\alpha|0\rangle + \beta|1\rangle)_A \otimes \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)_{E_A E_B}$$

$$= \frac{1}{\sqrt{2}} [\alpha|0\rangle(|00\rangle + |11\rangle) + \beta|1\rangle(|00\rangle + |11\rangle)]$$

$$= \frac{1}{\sqrt{2}} [\alpha|000\rangle + \alpha|011\rangle + \beta|100\rangle + \beta|111\rangle]$$

### 2.3 Step 1: Alice Applies CNOT

Alice applies a CNOT gate with qubit $A$ as control and $E_A$ as target:

$$|\Psi_1\rangle = \frac{1}{\sqrt{2}} [\alpha|000\rangle + \alpha|011\rangle + \beta|110\rangle + \beta|101\rangle]$$

### 2.4 Step 2: Alice Applies Hadamard

Alice applies a Hadamard gate to qubit $A$:

Using $H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$ and $H|1\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$:

$$|\Psi_2\rangle = \frac{1}{2} [\alpha(|0\rangle + |1\rangle)|00\rangle + \alpha(|0\rangle + |1\rangle)|11\rangle + \beta(|0\rangle - |1\rangle)|10\rangle + \beta(|0\rangle - |1\rangle)|01\rangle]$$

Regrouping by Alice's two qubits ($A$ and $E_A$):

$$|\Psi_2\rangle = \frac{1}{2} [|00\rangle(\alpha|0\rangle + \beta|1\rangle) + |01\rangle(\alpha|1\rangle + \beta|0\rangle) + |10\rangle(\alpha|0\rangle - \beta|1\rangle) + |11\rangle(\alpha|1\rangle - \beta|0\rangle)]$$

### 2.5 Step 3: Alice Measures

Alice measures her two qubits (A and $E_A$) in the computational basis. Each outcome occurs with probability $1/4$:

| Alice's result | Bob's qubit state | Correction needed |
|----------------|-------------------|-------------------|
| $\|00\rangle$ | $\alpha\|0\rangle + \beta\|1\rangle$ | None ($I$) |
| $\|01\rangle$ | $\alpha\|1\rangle + \beta\|0\rangle$ | $X$ |
| $\|10\rangle$ | $\alpha\|0\rangle - \beta\|1\rangle$ | $Z$ |
| $\|11\rangle$ | $\alpha\|1\rangle - \beta\|0\rangle$ | $ZX$ |

### 2.6 Step 4: Classical Communication and Correction

Alice sends her 2-bit measurement result to Bob over a **classical channel**. Based on these bits, Bob applies the corresponding Pauli correction:

| Classical bits | Correction |
|---------------|-----------|
| 00 | $I$ (do nothing) |
| 01 | $X$ (bit flip) |
| 10 | $Z$ (phase flip) |
| 11 | $ZX = iY$ (both) |

After correction, Bob's qubit is exactly $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$. The teleportation is complete.

### 2.7 What Happened to Alice's Original?

Alice's measurement has collapsed her qubit $A$ into either $|0\rangle$ or $|1\rangle$ — the original state $|\psi\rangle$ is destroyed. The information has been transferred, not copied. No-cloning is satisfied.

### 2.8 Resources Used

- **1 entangled pair** (shared beforehand, consumed during protocol)
- **2 classical bits** (sent from Alice to Bob)
- **1 qubit** teleported

This resource accounting is tight: quantum teleportation requires exactly 1 ebit (entangled bit) and 2 cbits (classical bits) to teleport 1 qubit.

---

## 3. Why Teleportation Does Not Violate No-Signaling

### 3.1 The Concern

Entanglement is shared between Alice and Bob, who could be light-years apart. When Alice measures her qubits, Bob's qubit instantly (as far as quantum mechanics is concerned) changes. Does this allow faster-than-light communication?

### 3.2 The Resolution

**No.** Before Alice sends the classical bits, Bob's qubit is in a maximally mixed state — completely random, carrying no information:

$$\rho_B = \text{Tr}_{AE_A}(|\Psi_2\rangle\langle\Psi_2|) = \frac{1}{2}I$$

Regardless of what $|\psi\rangle$ is, Bob sees a completely random qubit. He cannot extract any information about $\alpha$ or $\beta$ without the classical bits from Alice.

The classical bits are essential and cannot travel faster than light. The quantum state is only recovered after receiving the classical correction, which is limited by the speed of light.

### 3.3 Information Accounting

- **Before classical bits**: Bob has 0 bits of information about $|\psi\rangle$
- **After classical bits**: Bob has complete information about $|\psi\rangle$
- **Classical bits alone**: 2 classical bits cannot describe an arbitrary qubit state (which requires continuous parameters $\alpha, \beta$)

The teleportation protocol exploits the pre-shared entanglement as a "quantum channel" that, when activated by the 2 classical bits, transfers the full continuous quantum information. Neither the entanglement nor the classical bits alone suffice.

---

## 4. Superdense Coding

### 4.1 The Dual of Teleportation

Superdense coding is the "dual" of teleportation: while teleportation sends 1 qubit using 2 classical bits + 1 ebit, superdense coding sends 2 classical bits using 1 qubit + 1 ebit.

### 4.2 Protocol

**Setup**: Alice and Bob share a Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$. Alice has the first qubit, Bob has the second.

**Step 1**: Alice wants to send 2 classical bits to Bob. She applies one of four operations to her qubit:

| Message | Alice's operation | Resulting state |
|---------|-------------------|-----------------|
| 00 | $I$ | $\|\Phi^+\rangle = \frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ |
| 01 | $X$ | $\|\Psi^+\rangle = \frac{1}{\sqrt{2}}(\|10\rangle + \|01\rangle)$ |
| 10 | $Z$ | $\|\Phi^-\rangle = \frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ |
| 11 | $ZX$ | $\|\Psi^-\rangle = \frac{1}{\sqrt{2}}(\|10\rangle - \|01\rangle)$ |

**Step 2**: Alice sends her qubit to Bob (1 qubit transmitted).

**Step 3**: Bob now has both qubits. He performs a Bell measurement (CNOT followed by H on the first qubit, then measures both). The four Bell states are orthogonal, so Bob perfectly distinguishes them and recovers the 2-bit message.

### 4.3 Resource Comparison

| Protocol | Quantum sent | Classical sent | Entanglement | Information transferred |
|----------|-------------|---------------|-------------|----------------------|
| Teleportation | 0 qubits | 2 bits | 1 ebit | 1 qubit |
| Superdense coding | 1 qubit | 0 bits | 1 ebit | 2 classical bits |

These two protocols are dual: they trade quantum and classical resources in complementary ways.

---

## 5. Quantum Key Distribution: BB84

### 5.1 The Key Distribution Problem

Alice and Bob want to establish a shared secret key over an insecure channel. Classical key distribution either requires a pre-shared secret (symmetric crypto) or relies on computational hardness assumptions (RSA, Diffie-Hellman). Quantum key distribution (QKD) provides **information-theoretic security** — security guaranteed by the laws of physics, not computational assumptions.

### 5.2 BB84 Protocol

Proposed by Bennett and Brassard in 1984:

**Step 1: Alice prepares and sends qubits**

For each bit of the key, Alice randomly chooses:
- A bit value $b \in \{0, 1\}$
- A basis: $Z$ (computational) or $X$ (Hadamard)

She prepares the qubit:

| Bit | Basis | State |
|-----|-------|-------|
| 0 | Z | $\|0\rangle$ |
| 1 | Z | $\|1\rangle$ |
| 0 | X | $\|+\rangle$ |
| 1 | X | $\|-\rangle$ |

**Step 2: Bob measures**

For each received qubit, Bob randomly chooses a measurement basis ($Z$ or $X$) and measures.

- If Bob chooses the **same basis** as Alice: he gets Alice's bit value with certainty
- If Bob chooses a **different basis**: he gets a random result (50/50)

**Step 3: Basis reconciliation (public channel)**

Alice and Bob publicly announce their basis choices (but NOT their bit values). They discard all bits where they used different bases. The remaining bits (about 50% of the total) form the **sifted key**.

**Step 4: Eavesdropper detection**

Alice and Bob sacrifice a random subset of the sifted key and compare values publicly. If an eavesdropper (Eve) intercepted and measured the qubits, she would have introduced errors (due to the no-cloning theorem and measurement disturbance).

- **No eavesdropper**: 0% error rate in the comparison
- **Eavesdropper**: approximately 25% error rate (Eve guesses the wrong basis 50% of the time, and wrong-basis measurement gives the wrong value 50% of the time)

If the error rate is below a threshold, they proceed to use the remaining sifted key (with privacy amplification). If the error rate is too high, they abort — the channel is compromised.

### 5.3 Security Basis

BB84 security rests on two quantum mechanical principles:

1. **No-cloning theorem**: Eve cannot copy the qubits to measure in both bases
2. **Measurement disturbance**: Measuring a qubit in the wrong basis irreversibly changes its state, introducing detectable errors

Even a quantum computer cannot break BB84 (unlike RSA). QKD provides information-theoretic security independent of computational assumptions.

### 5.4 Practical Limitations

- **Distance**: Photon loss in fiber limits range to ~100 km without repeaters
- **Key rate**: Typically kilobits/second, far too slow for bulk encryption (used only for key exchange)
- **Side channels**: Real implementations may have vulnerabilities in the hardware (detector blinding, Trojan horse attacks)
- **Authentication**: BB84 requires an authenticated classical channel (which itself needs some initial shared secret)

---

## 6. Quantum Repeaters

### 6.1 The Distance Problem

Quantum communication over long distances faces photon loss: in optical fiber, about 0.2 dB/km loss, meaning after 100 km, only ~1% of photons arrive. Classical communication solves this with amplifiers (repeaters), but the no-cloning theorem prevents us from simply amplifying quantum signals.

### 6.2 Entanglement Swapping

The key idea behind quantum repeaters is **entanglement swapping**:

1. Create an entangled pair between Alice and a midpoint node (Charlie)
2. Create another entangled pair between Charlie and Bob
3. Charlie performs a Bell measurement on his two qubits
4. This "swaps" the entanglement: Alice and Bob are now entangled, even though they never interacted directly

```
Alice ~~~~ Charlie ~~~~ Bob
  A-C entangled    C-B entangled

Charlie performs Bell measurement:

Alice ~~~~~~~~~~~~~ Bob
    Now directly entangled!
```

### 6.3 Entanglement Distillation

Imperfect entangled pairs can be "distilled" into fewer but higher-quality pairs using local operations and classical communication (LOCC). Multiple noisy Bell pairs are converted into fewer near-perfect Bell pairs.

### 6.4 Quantum Repeater Architecture

A practical quantum repeater chain works as follows:

1. Divide the total distance into segments
2. Create entanglement within each segment (short-distance, manageable loss)
3. Use entanglement swapping to extend entanglement across segments
4. Use distillation to maintain fidelity
5. Repeat until end-to-end entanglement is established

This architecture enables quantum communication over thousands of kilometers, in principle. Research prototypes have demonstrated entanglement over ~1,200 km via the Micius satellite (2017).

---

## 7. Python Implementation

### 7.1 Quantum Teleportation

```python
import numpy as np

def teleportation_simulation(alpha, beta, verbose=True):
    """Simulate the quantum teleportation protocol.

    Why simulate the full 3-qubit system? This demonstrates how the
    entanglement between Alice's and Bob's qubits enables the transfer
    of an unknown state through measurement and classical communication.
    """
    # Validate input: |α|² + |β|² should equal 1
    assert abs(abs(alpha)**2 + abs(beta)**2 - 1) < 1e-10, "State not normalized"

    if verbose:
        print("=" * 55)
        print("Quantum Teleportation Protocol")
        print("=" * 55)
        print(f"\nState to teleport: |ψ⟩ = ({alpha:.4f})|0⟩ + ({beta:.4f})|1⟩")

    # === Step 0: Initial state ===
    # |ψ⟩_A ⊗ |Φ+⟩_{E_A E_B}
    # 3 qubits: A, E_A, E_B in that order
    # Basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 on qubits E_A, E_B
    bell = np.zeros(4, dtype=complex)
    bell[0b00] = 1/np.sqrt(2)  # |00⟩
    bell[0b11] = 1/np.sqrt(2)  # |11⟩

    # Alice's qubit
    psi = np.array([alpha, beta], dtype=complex)

    # Full 3-qubit state: |ψ⟩ ⊗ |Φ+⟩
    state = np.kron(psi, bell)
    if verbose:
        print(f"\nInitial 3-qubit state:")
        _print_state(state, 3)

    # === Step 1: CNOT (A→E_A) ===
    # CNOT flips E_A when A=1
    CNOT = np.eye(8, dtype=complex)
    # Swap |100⟩↔|110⟩ and |101⟩↔|111⟩
    for i in range(8):
        if (i >> 2) & 1:  # If qubit A is |1⟩
            j = i ^ (1 << 1)  # Flip qubit E_A
            CNOT[i, i] = 0
            CNOT[j, j] = 0
            CNOT[i, j] = 1
            CNOT[j, i] = 1

    state = CNOT @ state
    if verbose:
        print("\nAfter CNOT (A controls E_A):")
        _print_state(state, 3)

    # === Step 2: Hadamard on A ===
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I2 = np.eye(2)
    H_full = np.kron(np.kron(H, I2), I2)  # H on qubit A only

    state = H_full @ state
    if verbose:
        print("\nAfter Hadamard on A:")
        _print_state(state, 3)

    # === Step 3: Alice measures qubits A and E_A ===
    # Compute probabilities for each measurement outcome
    probs = {}
    bob_states = {}
    for m in range(4):  # Alice's 2 qubits: 00, 01, 10, 11
        # Project onto |m⟩ on qubits A and E_A
        bob_state = np.zeros(2, dtype=complex)
        for b in range(2):  # Bob's qubit
            idx = (m << 1) | b  # Combine Alice's bits with Bob's bit
            bob_state[b] = state[idx]
        prob = np.sum(np.abs(bob_state)**2)
        if prob > 1e-10:
            probs[m] = prob
            bob_states[m] = bob_state / np.sqrt(prob)

    # Simulate measurement (random outcome)
    outcomes = list(probs.keys())
    probabilities = [probs[m] for m in outcomes]
    measurement = outcomes[np.random.choice(len(outcomes), p=probabilities)]

    if verbose:
        print(f"\nAlice measures: |{measurement:02b}⟩ (prob = {probs[measurement]:.4f})")
        print(f"Bob's qubit before correction: {bob_states[measurement].round(4)}")

    # === Step 4: Bob applies correction ===
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    corrections = {
        0b00: np.eye(2, dtype=complex),
        0b01: X,
        0b10: Z,
        0b11: Z @ X,
    }
    correction_names = {0b00: "I", 0b01: "X", 0b10: "Z", 0b11: "ZX"}

    bob_final = corrections[measurement] @ bob_states[measurement]

    if verbose:
        print(f"\nClassical bits sent: {measurement >> 1}, {measurement & 1}")
        print(f"Bob applies correction: {correction_names[measurement]}")
        print(f"Bob's final state: ({bob_final[0]:.4f})|0⟩ + ({bob_final[1]:.4f})|1⟩")

    # Verify
    fidelity = abs(np.dot(psi.conj(), bob_final))**2
    if verbose:
        print(f"\nFidelity with original: {fidelity:.10f}")
        print("Teleportation " + ("SUCCESS" if fidelity > 0.999 else "FAILED"))

    return fidelity

def _print_state(state, n_qubits):
    """Print a quantum state with labeled basis states."""
    for i in range(len(state)):
        if abs(state[i]) > 1e-10:
            label = f"|{i:0{n_qubits}b}⟩"
            print(f"  {state[i]:+.4f} {label}")

# Test with several states
print("Test 1: |ψ⟩ = |0⟩")
teleportation_simulation(1, 0)

print("\n\nTest 2: |ψ⟩ = |+⟩")
teleportation_simulation(1/np.sqrt(2), 1/np.sqrt(2))

print("\n\nTest 3: |ψ⟩ = (1+2i)|0⟩/√5 + (0)|1⟩... arbitrary state")
alpha = (1 + 2j) / np.sqrt(5 + 4)
beta = 2 / np.sqrt(5 + 4)
# Normalize
norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
teleportation_simulation(alpha/norm, beta/norm)
```

### 7.2 Superdense Coding

```python
import numpy as np

def superdense_coding(message_bits):
    """Simulate the superdense coding protocol.

    Why superdense coding? It demonstrates the "dual" of teleportation:
    1 qubit + 1 ebit can carry 2 classical bits. This exceeds the Holevo
    bound for a single qubit (which can carry at most 1 classical bit
    without entanglement).

    Args:
        message_bits: tuple (b1, b2) where each is 0 or 1
    """
    b1, b2 = message_bits
    print(f"Message to send: ({b1}, {b2})")

    # Shared Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    state = np.zeros(4, dtype=complex)
    state[0b00] = 1/np.sqrt(2)
    state[0b11] = 1/np.sqrt(2)
    print(f"Shared Bell state: (|00⟩ + |11⟩)/√2")

    # Alice's encoding: apply operation to her qubit (qubit 0)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Alice's operation depends on the message
    # 00 → I, 01 → X, 10 → Z, 11 → ZX
    if (b1, b2) == (0, 0):
        alice_op = I
        op_name = "I"
    elif (b1, b2) == (0, 1):
        alice_op = X
        op_name = "X"
    elif (b1, b2) == (1, 0):
        alice_op = Z
        op_name = "Z"
    else:  # (1, 1)
        alice_op = Z @ X
        op_name = "ZX"

    # Apply Alice's operation to qubit 0
    full_op = np.kron(alice_op, I)
    state = full_op @ state
    print(f"Alice applies {op_name} to her qubit")

    # Alice sends her qubit to Bob (Bob now has both qubits)

    # Bob's decoding: CNOT then Hadamard, then measure
    # CNOT (qubit 0 controls qubit 1)
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    state = CNOT @ state

    # Hadamard on qubit 0
    H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    HI = np.kron(H, I)
    state = HI @ state

    # Measure
    probs = np.abs(state)**2
    result = np.argmax(probs)
    decoded_b1 = (result >> 1) & 1
    decoded_b2 = result & 1

    print(f"Bob measures: |{result:02b}⟩ with probability {probs[result]:.4f}")
    print(f"Decoded message: ({decoded_b1}, {decoded_b2})")
    print(f"Correct: {'YES' if (decoded_b1, decoded_b2) == (b1, b2) else 'NO'}")
    return (decoded_b1, decoded_b2)

print("=" * 50)
print("Superdense Coding Protocol")
print("=" * 50)

for msg in [(0,0), (0,1), (1,0), (1,1)]:
    print()
    superdense_coding(msg)
```

### 7.3 BB84 Quantum Key Distribution

```python
import numpy as np

def bb84_simulation(n_bits=100, eve_present=False, verbose=True):
    """Simulate the BB84 quantum key distribution protocol.

    Why simulate BB84? This demonstrates how quantum mechanics enables
    information-theoretically secure key exchange. The no-cloning theorem
    prevents an eavesdropper from copying qubits, and measurement in the
    wrong basis introduces detectable errors.
    """
    np.random.seed(42)

    # === Alice's preparation ===
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z basis, 1=X basis

    # Prepare quantum states
    # Z basis: 0→|0⟩, 1→|1⟩
    # X basis: 0→|+⟩, 1→|−⟩
    states = []
    for i in range(n_bits):
        if alice_bases[i] == 0:  # Z basis
            if alice_bits[i] == 0:
                states.append(np.array([1, 0], dtype=complex))
            else:
                states.append(np.array([0, 1], dtype=complex))
        else:  # X basis
            if alice_bits[i] == 0:
                states.append(np.array([1, 1], dtype=complex) / np.sqrt(2))
            else:
                states.append(np.array([1, -1], dtype=complex) / np.sqrt(2))

    # === Eve's interception (if present) ===
    eve_bases = None
    if eve_present:
        eve_bases = np.random.randint(0, 2, n_bits)
        for i in range(n_bits):
            # Eve measures in her randomly chosen basis
            if eve_bases[i] == 0:  # Z basis measurement
                prob_0 = abs(states[i][0])**2
                result = 0 if np.random.random() < prob_0 else 1
                # After measurement, state collapses
                if result == 0:
                    states[i] = np.array([1, 0], dtype=complex)
                else:
                    states[i] = np.array([0, 1], dtype=complex)
            else:  # X basis measurement
                # Project onto |+⟩ and |−⟩
                plus = np.array([1, 1]) / np.sqrt(2)
                prob_plus = abs(np.dot(plus, states[i]))**2
                result = 0 if np.random.random() < prob_plus else 1
                if result == 0:
                    states[i] = np.array([1, 1], dtype=complex) / np.sqrt(2)
                else:
                    states[i] = np.array([1, -1], dtype=complex) / np.sqrt(2)

    # === Bob's measurement ===
    bob_bases = np.random.randint(0, 2, n_bits)
    bob_bits = np.zeros(n_bits, dtype=int)

    for i in range(n_bits):
        if bob_bases[i] == 0:  # Z basis measurement
            prob_0 = abs(states[i][0])**2
            bob_bits[i] = 0 if np.random.random() < prob_0 else 1
        else:  # X basis measurement
            plus = np.array([1, 1]) / np.sqrt(2)
            prob_plus = abs(np.dot(plus, states[i]))**2
            bob_bits[i] = 0 if np.random.random() < prob_plus else 1

    # === Basis reconciliation ===
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]
    n_sifted = len(sifted_alice)

    # === Error estimation ===
    # Use first half for error check, second half as key
    n_check = n_sifted // 2
    check_alice = sifted_alice[:n_check]
    check_bob = sifted_bob[:n_check]
    errors = np.sum(check_alice != check_bob)
    error_rate = errors / n_check if n_check > 0 else 0

    key_alice = sifted_alice[n_check:]
    key_bob = sifted_bob[n_check:]
    key_match = np.all(key_alice == key_bob)

    if verbose:
        print("=" * 55)
        print(f"BB84 Protocol (Eve {'present' if eve_present else 'absent'})")
        print("=" * 55)
        print(f"  Qubits sent: {n_bits}")
        print(f"  Matching bases: {n_sifted} ({100*n_sifted/n_bits:.0f}%)")
        print(f"  Check bits used: {n_check}")
        print(f"  Errors in check: {errors}/{n_check} = {100*error_rate:.1f}%")
        print(f"  Key length: {len(key_alice)} bits")
        print(f"  Keys match: {key_match}")

        if error_rate > 0.11:
            print("  ALERT: Error rate too high — eavesdropper likely detected!")
            print("  Protocol ABORTED.")
        else:
            print("  Key accepted — channel appears secure.")

    return error_rate, key_match

# Run BB84 without eavesdropper
print("\n--- Without Eavesdropper ---")
bb84_simulation(n_bits=200, eve_present=False)

# Run BB84 with eavesdropper
print("\n--- With Eavesdropper ---")
bb84_simulation(n_bits=200, eve_present=True)
```

### 7.4 Entanglement Swapping

```python
import numpy as np

def entanglement_swapping():
    """Simulate entanglement swapping for quantum repeaters.

    Why entanglement swapping? It's the quantum analog of a relay: it extends
    entanglement across distances that are too far for direct transmission.
    Two short-distance entangled pairs are "joined" into one long-distance pair
    via a Bell measurement at the intermediate node.
    """
    print("=" * 55)
    print("Entanglement Swapping (Quantum Repeater Building Block)")
    print("=" * 55)

    # 4 qubits: A, C1, C2, B
    # A-C1 entangled (Bell pair 1): |Φ+⟩_{A,C1}
    # C2-B entangled (Bell pair 2): |Φ+⟩_{C2,B}
    # Charlie holds C1 and C2

    # Build initial state: |Φ+⟩_{A,C1} ⊗ |Φ+⟩_{C2,B}
    bell = np.zeros(4, dtype=complex)
    bell[0b00] = 1/np.sqrt(2)
    bell[0b11] = 1/np.sqrt(2)

    state = np.kron(bell, bell)  # 16-dimensional (4 qubits)
    print("\nInitial state: |Φ+⟩_{A,C1} ⊗ |Φ+⟩_{C2,B}")

    # Charlie performs Bell measurement on C1, C2 (qubits 1 and 2)
    # First: CNOT with C1 as control, C2 as target
    CNOT_C1C2 = np.eye(16, dtype=complex)
    for i in range(16):
        c1 = (i >> 2) & 1  # qubit 1
        c2 = (i >> 1) & 1  # qubit 2
        if c1 == 1:
            j = i ^ (1 << 1)  # flip qubit 2
            CNOT_C1C2[i, :] = 0
            CNOT_C1C2[i, j] = 1
        # else: identity (already set)

    state = CNOT_C1C2 @ state

    # Hadamard on C1 (qubit 1)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    I2 = np.eye(2)
    H_C1 = np.kron(np.kron(I2, H), np.kron(I2, I2))
    state = H_C1 @ state

    # Measure C1 and C2
    # For each measurement outcome (c1, c2), find Alice-Bob state
    print("\nCharlie measures C1 and C2:")
    for c1 in range(2):
        for c2 in range(2):
            # Extract Alice-Bob state for this measurement
            ab_state = np.zeros(4, dtype=complex)
            for a in range(2):
                for b in range(2):
                    idx = (a << 3) | (c1 << 2) | (c2 << 1) | b
                    ab_idx = (a << 1) | b
                    ab_state[ab_idx] = state[idx]

            prob = np.sum(np.abs(ab_state)**2)
            if prob > 1e-10:
                ab_state /= np.sqrt(prob)
                print(f"\n  Charlie gets |{c1}{c2}⟩ (prob={prob:.4f}):")
                print(f"  Alice-Bob state:")
                for i in range(4):
                    if abs(ab_state[i]) > 1e-10:
                        print(f"    {ab_state[i]:+.4f} |{i:02b}⟩")

                # Check if it's a Bell state
                bell_states = {
                    'Φ+': np.array([1,0,0,1])/np.sqrt(2),
                    'Φ-': np.array([1,0,0,-1])/np.sqrt(2),
                    'Ψ+': np.array([0,1,1,0])/np.sqrt(2),
                    'Ψ-': np.array([0,1,-1,0])/np.sqrt(2),
                }
                for name, bs in bell_states.items():
                    if abs(abs(np.dot(bs.conj(), ab_state)) - 1) < 1e-10:
                        print(f"  → This is |{name}⟩! Alice and Bob are now entangled!")

    print("\nResult: Regardless of Charlie's measurement outcome,")
    print("Alice and Bob end up sharing a Bell state (up to known Pauli correction).")
    print("Entanglement has been 'swapped' from A-C1 and C2-B to A-B!")

entanglement_swapping()
```

---

## 8. Exercises

### Exercise 1: Teleportation with Different Bell States

Repeat the teleportation protocol using $|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$ as the shared entangled pair instead of $|\Phi^+\rangle$.
(a) Work through the algebra step by step.
(b) How do the corrections change?
(c) Verify using the Python simulation.

### Exercise 2: Teleportation Fidelity with Noise

Suppose the shared Bell pair is imperfect — it is a Werner state:

$$\rho = p|\Phi^+\rangle\langle\Phi^+| + (1-p)\frac{I}{4}$$

(a) For what values of $p$ can the teleportation fidelity exceed $2/3$ (the classical limit)?
(b) Simulate teleportation with a noisy Bell pair for $p = 0.5, 0.7, 0.9, 1.0$. Plot fidelity vs $p$.
(c) Show that for $p < 1/3$, the protocol is useless (fidelity $\leq 1/2$).

### Exercise 3: Superdense Coding Capacity

(a) Prove that without shared entanglement, a single qubit can carry at most 1 classical bit (Holevo bound).
(b) Show that with 1 ebit of shared entanglement, 1 qubit can carry exactly 2 classical bits (superdense coding).
(c) What if Alice and Bob share a non-maximally entangled state $\cos\theta|00\rangle + \sin\theta|11\rangle$? How does the capacity change with $\theta$?

### Exercise 4: BB84 Security Analysis

Run the BB84 simulation 1000 times each with and without Eve:
(a) Plot a histogram of the error rates for both cases.
(b) What threshold error rate best separates the two distributions?
(c) Compute the probability that Eve is present but undetected (false negative) for your chosen threshold.
(d) How does the detection probability change with the number of check bits?

### Exercise 5: Quantum Repeater Chain

Extend the entanglement swapping simulation to a 3-node chain (Alice-Charlie1-Charlie2-Bob):
(a) Start with 3 Bell pairs: A-C1, C1'-C2, C2'-B.
(b) Perform entanglement swapping at both intermediate nodes.
(c) Verify that Alice and Bob end up with a Bell state.
(d) If each link has fidelity $F$, what is the end-to-end fidelity after 2 swaps? After $n$ swaps?

---

[← Previous: Quantum Error Correction](11_Quantum_Error_Correction.md) | [Next: Variational Quantum Eigensolver →](13_VQE.md)

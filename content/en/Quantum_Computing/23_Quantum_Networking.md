# Lesson 23: Quantum Networking

[← Previous: Topological Quantum Computing](22_Topological_Quantum_Computing.md) | [Next: Qiskit Deep Dive →](24_Qiskit_Deep_Dive.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain quantum key distribution protocols (BB84 and E91) and prove their security
2. Describe quantum repeaters and how they overcome the exponential photon loss problem
3. Analyze entanglement swapping and distillation as building blocks for quantum networks
4. Design a quantum internet architecture with different network layers
5. Compare quantum networking with classical networking in terms of capabilities and limitations
6. Implement QKD protocols, entanglement swapping, and quantum repeater simulations in Python

---

Quantum networking extends quantum computation beyond individual quantum processors to distributed systems connected by quantum communication channels. The most mature application is **quantum key distribution** (QKD), which enables two parties to establish a shared secret key with security guaranteed by the laws of physics — not by computational assumptions that could be broken by future algorithms or hardware.

But quantum networking is far more than secure communication. A future **quantum internet** would connect quantum computers, enabling distributed quantum computation, blind quantum computing (where a client delegates computation to a server without revealing the data), and quantum sensor networks with Heisenberg-limited precision. The fundamental building blocks are entanglement distribution, quantum repeaters, and quantum error correction across network links.

> **Analogy:** Classical internet transmits bits over fiber optic cables. A quantum internet transmits qubits and, more importantly, distributes entanglement. Entanglement is like a shared, unbreakable secret between two nodes — once established, it enables secure communication, teleportation of quantum states, and coordination of distributed quantum computations, all without the possibility of eavesdropping.

## Table of Contents

1. [Quantum Communication Fundamentals](#1-quantum-communication-fundamentals)
2. [BB84 Protocol](#2-bb84-protocol)
3. [E91 Protocol](#3-e91-protocol)
4. [Quantum Repeaters](#4-quantum-repeaters)
5. [Entanglement Swapping](#5-entanglement-swapping)
6. [Entanglement Distillation](#6-entanglement-distillation)
7. [Quantum Internet Architecture](#7-quantum-internet-architecture)
8. [Applications Beyond QKD](#8-applications-beyond-qkd)
9. [Python Implementation](#9-python-implementation)
10. [Exercises](#10-exercises)

---

## 1. Quantum Communication Fundamentals

### 1.1 No-Cloning Theorem

The no-cloning theorem (proven by Wootters and Zurek, 1982) states that there is no quantum operation that can copy an arbitrary quantum state:

$$\nexists U: U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle \quad \forall |\psi\rangle$$

**Consequences for networking**:
- Classical amplifiers (which copy and re-transmit signals) cannot work for quantum signals
- Eavesdropping on quantum communication inevitably disturbs the quantum state
- Quantum repeaters must use entanglement swapping instead of amplification

### 1.2 Quantum Channels

Quantum information is transmitted through physical channels (typically optical fiber or free space):

| Channel | Medium | Loss rate | Distance limit |
|---------|--------|-----------|---------------|
| Optical fiber | Glass fiber | $\sim 0.2$ dB/km at 1550 nm | $\sim 100$ km (direct) |
| Free space | Atmosphere/vacuum | Variable | $> 1000$ km (satellite) |
| Microwave | Cryogenic cable | High | $\sim 1$ m |

The key challenge: photon loss in fiber is exponential — after 100 km, only $\sim 1\%$ of photons survive.

### 1.3 Quantum vs. Classical Communication

| Feature | Classical | Quantum |
|---------|-----------|---------|
| Signal amplification | Yes (copy and boost) | No (no-cloning) |
| Eavesdropping detection | Impossible | Guaranteed by physics |
| Error correction | Standard (repetition) | Quantum error correction |
| Bandwidth | Gbps-Tbps | kbps-Mbps (current QKD) |
| Distance | Global (with repeaters) | $\sim 100$ km (without quantum repeaters) |

---

## 2. BB84 Protocol

### 2.1 Protocol Description

BB84 (Bennett and Brassard, 1984) is the first and most widely implemented QKD protocol.

**Setup**: Alice wants to send a secret key to Bob over a quantum channel (e.g., optical fiber) and an authenticated classical channel.

**Protocol steps**:

1. **Preparation**: For each bit $b_i \in \{0, 1\}$, Alice randomly chooses a basis $\theta_i \in \{+, \times\}$:

   | Bit | + basis | $\times$ basis |
   |-----|---------|----------------|
   | 0 | $\|0\rangle$ | $\|+\rangle = (\|0\rangle + \|1\rangle)/\sqrt{2}$ |
   | 1 | $\|1\rangle$ | $\|-\rangle = (\|0\rangle - \|1\rangle)/\sqrt{2}$ |

2. **Transmission**: Alice sends each qubit to Bob through the quantum channel.

3. **Measurement**: Bob randomly chooses a basis $\theta_i' \in \{+, \times\}$ for each qubit and measures.

4. **Sifting**: Over the classical channel, Alice and Bob announce their basis choices (but NOT their bits). They keep only the bits where they used the same basis. On average, they agree $50\%$ of the time.

5. **Error estimation**: They publicly compare a random subset of their sifted key. If the error rate exceeds a threshold ($\sim 11\%$ for BB84), they abort (eavesdropper detected).

6. **Privacy amplification**: They use hashing to distill a shorter, perfectly secure key from the raw key.

### 2.2 Security Proof Sketch

**Why eavesdropping fails**: If Eve intercepts a qubit, she must measure it (collapsing the state) and re-send it. If she measures in the wrong basis, she introduces errors detectable by Alice and Bob.

**Quantitative**: For each qubit Eve intercepts, she introduces an error with probability $1/4$ (she guesses the wrong basis $50\%$ of the time, and then gets the wrong bit $50\%$ of the time). With $n$ intercepted qubits, the probability of going undetected decreases exponentially.

**Information-theoretic security**: Even with unlimited computational power, Eve cannot learn the key without being detected. This is fundamentally stronger than computational security (e.g., RSA).

### 2.3 Practical Considerations

- **Dark counts**: Detector noise creates false positives, limiting distance
- **Multi-photon attacks**: If Alice's source emits multiple photons, Eve can split one off (photon-number splitting attack). Countermeasure: decoy states
- **Key rate**: Scales as $\eta \cdot R$ where $\eta$ is the channel transmittance and $R$ is the source rate. For 100 km fiber: $\eta \approx 10^{-2}$
- **Side channels**: Implementation imperfections can leak information (e.g., timing, detector efficiency mismatch)

---

## 3. E91 Protocol

### 3.1 Entanglement-Based QKD

E91 (Ekert, 1991) uses entangled photon pairs instead of single photons:

1. **Source**: A source produces Bell pairs $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ and sends one photon to Alice and one to Bob

2. **Measurement**: Each party randomly measures in one of three bases (rotated by $0^\circ, 22.5^\circ, 45^\circ$ for Alice; $22.5^\circ, 45^\circ, 67.5^\circ$ for Bob)

3. **Sifting**: When both measure in the same basis ($22.5^\circ$ or $45^\circ$), their results are perfectly correlated → key bits

4. **Security check**: When they measure in different bases, they test the CHSH inequality. A violation of $S > 2$ certifies entanglement and rules out eavesdropping

### 3.2 CHSH Inequality Test

The CHSH parameter:

$$S = |E(a_1, b_1) - E(a_1, b_2) + E(a_2, b_1) + E(a_2, b_2)|$$

where $E(a_i, b_j)$ is the correlation between Alice's measurement in direction $a_i$ and Bob's in $b_j$.

- **Classical bound**: $S \leq 2$
- **Quantum maximum**: $S = 2\sqrt{2} \approx 2.828$
- **Tsirelson bound**: No quantum state can exceed $2\sqrt{2}$

If $S > 2$, the correlations are genuinely quantum, and no eavesdropper could have measured the qubits without reducing $S$.

### 3.3 Device-Independent QKD

E91 enables **device-independent QKD**: security holds even if Alice and Bob do not trust their own devices. As long as the CHSH violation is observed, the key is secure — regardless of the internal workings of the measurement apparatus.

---

## 4. Quantum Repeaters

### 4.1 The Distance Problem

Direct quantum communication over optical fiber is limited to $\sim 100$ km due to exponential photon loss. Classical repeaters amplify signals by copying them, but the no-cloning theorem forbids copying quantum states.

**Loss scaling**: Transmission probability $\eta = 10^{-\alpha L/10}$ where $\alpha \approx 0.2$ dB/km. For $L = 100$ km: $\eta \approx 0.01$. For $L = 1000$ km: $\eta \approx 10^{-20}$.

### 4.2 Quantum Repeater Concept

Quantum repeaters divide the total distance into segments and use **entanglement swapping** to extend entanglement across segments:

```
Alice --- R1 --- R2 --- R3 --- Bob
  |-------|-------|-------|------|
  entangle  swap    swap   entangle
  A-R1    R1-R2  R2-R3   R3-B
```

### 4.3 Three Generations of Repeaters

| Generation | Mechanism | Error handling | Rate scaling |
|-----------|-----------|---------------|-------------|
| **1st gen** | Entanglement swap + distillation | Heralded entanglement | Polynomial in $L$ |
| **2nd gen** | Quantum error correction on links | QEC codes | Polynomial, faster |
| **3rd gen** | Full quantum error correction | Encoded qubits throughout | Constant (like classical) |

### 4.4 Repeater Rate Analysis

For $n$ segments of length $L_0 = L/n$:

- **Direct transmission**: Rate $\sim R_0 \eta^L \sim R_0 e^{-\alpha L}$ (exponential decay)
- **With repeaters**: Rate $\sim R_0 \eta^{L/n} / \text{poly}(n)$ (polynomial decay)

The crossover distance where repeaters help depends on the repeater quality (memory coherence, gate fidelity, swap success probability).

---

## 5. Entanglement Swapping

### 5.1 Protocol

Entanglement swapping creates entanglement between two parties that have never interacted:

1. **Start**: Alice shares a Bell pair with Charlie: $(|\Phi^+\rangle)_{AC}$. Charlie shares a Bell pair with Bob: $(|\Phi^+\rangle)_{CB}$.

2. **Bell measurement**: Charlie performs a Bell measurement on his two qubits (one from each pair).

3. **Result**: Alice and Bob now share a Bell pair, conditioned on Charlie's measurement outcome.

### 5.2 Mathematical Description

Initial state (4 qubits: $A, C_1, C_2, B$):

$$|\Psi\rangle = |\Phi^+\rangle_{AC_1} \otimes |\Phi^+\rangle_{C_2B}$$

Charlie measures $C_1 C_2$ in the Bell basis. The result is one of four outcomes, each projecting $A, B$ into a Bell state:

| Charlie's result | Alice-Bob state | Correction needed |
|-----------------|-----------------|-------------------|
| $\|\Phi^+\rangle$ | $\|\Phi^+\rangle_{AB}$ | None |
| $\|\Phi^-\rangle$ | $\|\Phi^-\rangle_{AB}$ | $Z$ on Bob |
| $\|\Psi^+\rangle$ | $\|\Psi^+\rangle_{AB}$ | $X$ on Bob |
| $\|\Psi^-\rangle$ | $\|\Psi^-\rangle_{AB}$ | $XZ$ on Bob |

### 5.3 Nested Swapping

For $n$ intermediate nodes, entanglement swapping can be applied hierarchically:

```
Level 0: A-R1  R1-R2  R2-R3  R3-B
Level 1: A---R2       R2---B
Level 2: A-----------B
```

This requires $n - 1$ swap operations and has success probability $(p_{\text{swap}})^{n-1}$ where $p_{\text{swap}}$ is the swap success probability.

---

## 6. Entanglement Distillation

### 6.1 The Problem

Real quantum channels produce **noisy entanglement**: instead of perfect Bell pairs, we get mixed states $\rho_{AB}$ with fidelity $F = \langle\Phi^+|\rho_{AB}|\Phi^+\rangle < 1$.

### 6.2 Bennett et al. Distillation Protocol

Given $n$ copies of a noisy Bell pair with fidelity $F > 1/2$, produce $m < n$ pairs with fidelity $F' > F$:

1. Alice and Bob each apply a random bilateral CNOT between two pairs
2. They measure the target pair and compare results classically
3. If results agree, keep the control pair (higher fidelity); otherwise discard

**Fidelity improvement**: $F' = \frac{F^2 + (1-F)^2/9}{F^2 + 2F(1-F)/3 + 5(1-F)^2/9}$

### 6.3 DEJMPS Protocol

A more efficient distillation protocol:

1. Apply bilateral $Y$ rotations
2. Bilateral CNOT
3. Measure and post-select

This achieves higher yield (fraction of input pairs that become distilled pairs) for the same output fidelity.

### 6.4 One-Way Distillation

Using quantum error correction: encode logical qubits across multiple noisy Bell pairs and decode to get fewer, higher-fidelity pairs. This avoids the two-way classical communication overhead.

---

## 7. Quantum Internet Architecture

### 7.1 Network Stack

The quantum internet is organized in layers analogous to the classical internet:

| Layer | Function | Quantum analog |
|-------|----------|---------------|
| Physical | Signal transmission | Photon transmission, qubit-photon interfaces |
| Link | Point-to-point connection | Entanglement generation, error correction |
| Network | Routing | Entanglement routing, path selection |
| Transport | End-to-end reliability | Entanglement distillation, verification |
| Application | User services | QKD, distributed computing, sensing |

### 7.2 Stages of Quantum Internet Development

| Stage | Capability | Key feature |
|-------|-----------|-------------|
| **1. Trusted-node QKD** | QKD with trusted intermediate nodes | Current (deployed) |
| **2. Prepare-and-measure** | Direct qubit transmission | Near-term |
| **3. Entanglement distribution** | Short-range Bell pairs | Medium-term |
| **4. Quantum memory network** | Store and forward entanglement | Requires quantum memories |
| **5. Full quantum internet** | Arbitrary quantum communication | Long-term goal |

### 7.3 Key Technologies

| Technology | Current state | Needed for |
|-----------|--------------|-----------|
| Single-photon sources | $> 90\%$ efficiency | All stages |
| Quantum memories | $\sim 1$ s coherence (ions) | Stage 4+ |
| Photon-matter interfaces | $50$-$80\%$ efficiency | Stage 3+ |
| Bell-state measurement | $50\%$ linear optics limit | Stage 3+ |
| Quantum error correction | Surface code demos | Stage 5 |

### 7.4 Current Deployments

- **China**: Beijing-Shanghai QKD backbone (2,000 km, trusted-node), Micius satellite QKD
- **Europe**: OPENQKD testbed, EuroQCI initiative
- **USA**: DOE quantum network testbeds (Chicago, New York)
- **Netherlands**: QuTech network connecting Delft-The Hague

---

## 8. Applications Beyond QKD

### 8.1 Distributed Quantum Computing

Connect multiple small quantum processors to create a larger, distributed quantum computer:

- **Entanglement-assisted computation**: Use shared Bell pairs to implement non-local gates
- **Remote CNOT**: Use teleportation to apply a CNOT gate between qubits on different processors
- **Distributed VQE**: Split the ansatz across processors, communicate via entanglement

### 8.2 Blind Quantum Computing

A client with minimal quantum capability delegates computation to a powerful quantum server without revealing the computation:

1. Client prepares random single-qubit states
2. Server entangles them according to a graph state
3. Client instructs server to measure qubits one at a time, with measurement angles that encode the computation
4. Server cannot learn the computation because each instruction looks random

### 8.3 Quantum Sensor Networks

Entanglement-enhanced sensor networks achieve Heisenberg-limited precision ($1/N$ scaling with $N$ sensors instead of the classical $1/\sqrt{N}$):

- **Quantum clock synchronization**: Distribute entanglement between atomic clocks for improved synchronization
- **Distributed quantum sensing**: Detect weak signals by correlating measurements across a sensor network
- **Gravitational wave detection**: Entanglement between interferometer arms could improve LIGO sensitivity

### 8.4 Quantum Money and Tokens

Quantum states that cannot be counterfeited (by the no-cloning theorem):
- **Quantum money**: Each banknote contains a quantum state that can be verified but not copied
- **Quantum tokens**: Single-use authentication tokens with guaranteed unforgeability

---

## 9. Python Implementation

### 9.1 BB84 Protocol Simulation

```python
import numpy as np

def bb84_protocol(n_bits, eve_present=False, eve_fraction=1.0, seed=42):
    """Simulate the BB84 quantum key distribution protocol.

    BB84 uses four quantum states in two conjugate bases:
    - Z basis: |0> (bit 0), |1> (bit 1)
    - X basis: |+> (bit 0), |-> (bit 1)

    Security comes from the fact that measuring in the wrong basis
    gives a random result, so an eavesdropper inevitably introduces errors.

    Args:
        n_bits: Number of qubits Alice sends
        eve_present: Whether Eve intercepts and measures
        eve_fraction: Fraction of qubits Eve intercepts
        seed: Random seed

    Returns:
        Dict with protocol results
    """
    rng = np.random.default_rng(seed)

    # Step 1: Alice prepares random bits and bases
    alice_bits = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)  # 0 = Z, 1 = X

    # Step 2: Eve intercepts (if present)
    eve_bits = np.full(n_bits, -1)
    if eve_present:
        eve_bases = rng.integers(0, 2, n_bits)
        eve_intercepts = rng.random(n_bits) < eve_fraction

        for i in range(n_bits):
            if eve_intercepts[i]:
                if eve_bases[i] == alice_bases[i]:
                    # Eve measures in correct basis: gets correct bit
                    eve_bits[i] = alice_bits[i]
                else:
                    # Eve measures in wrong basis: random result
                    eve_bits[i] = rng.integers(0, 2)
                # Eve resends in her basis (may introduce errors)

    # Step 3: Bob measures
    bob_bases = rng.integers(0, 2, n_bits)
    bob_bits = np.zeros(n_bits, dtype=int)

    for i in range(n_bits):
        if eve_present and eve_bits[i] >= 0:
            # Bob receives Eve's re-sent qubit
            if bob_bases[i] == alice_bases[i]:
                # If Eve measured in wrong basis, ~50% error
                if eve_bits[i] == alice_bits[i]:
                    bob_bits[i] = alice_bits[i]
                else:
                    # Eve corrupted it, Bob might get wrong result
                    if bob_bases[i] == alice_bases[i]:
                        bob_bits[i] = rng.integers(0, 2)
                    else:
                        bob_bits[i] = rng.integers(0, 2)
            else:
                bob_bits[i] = rng.integers(0, 2)
        else:
            # No Eve: Bob gets correct result only if bases match
            if bob_bases[i] == alice_bases[i]:
                bob_bits[i] = alice_bits[i]
            else:
                bob_bits[i] = rng.integers(0, 2)

    # Step 4: Sifting — keep only matching bases
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]

    # Step 5: Error estimation (use 50% of sifted bits)
    n_sifted = len(sifted_alice)
    n_check = n_sifted // 2
    check_indices = rng.choice(n_sifted, n_check, replace=False)

    errors = sifted_alice[check_indices] != sifted_bob[check_indices]
    qber = np.mean(errors) if n_check > 0 else 0.0

    # Step 6: Key is the remaining (unchecked) bits
    key_mask = np.ones(n_sifted, dtype=bool)
    key_mask[check_indices] = False
    key_alice = sifted_alice[key_mask]
    key_bob = sifted_bob[key_mask]

    key_errors = np.sum(key_alice != key_bob)

    return {
        'n_sent': n_bits,
        'n_sifted': n_sifted,
        'n_key': len(key_alice),
        'qber': qber,
        'key_errors': key_errors,
        'key_match': np.all(key_alice == key_bob),
        'eve_detected': qber > 0.11,  # BB84 security threshold ~11%
    }


# Demonstrate BB84
print("=" * 65)
print("BB84 Quantum Key Distribution")
print("=" * 65)

# Without eavesdropper
print("\n--- No eavesdropper ---")
result = bb84_protocol(10000, eve_present=False)
print(f"  Bits sent: {result['n_sent']}")
print(f"  Sifted key length: {result['n_sifted']}")
print(f"  Final key length: {result['n_key']}")
print(f"  QBER: {result['qber']:.4f}")
print(f"  Key errors: {result['key_errors']}")
print(f"  Keys match: {result['key_match']}")

# With eavesdropper
print("\n--- Eavesdropper (intercepts 100%) ---")
result = bb84_protocol(10000, eve_present=True, eve_fraction=1.0)
print(f"  Bits sent: {result['n_sent']}")
print(f"  Sifted key length: {result['n_sifted']}")
print(f"  QBER: {result['qber']:.4f}")
print(f"  Eve detected: {result['eve_detected']} (threshold: QBER > 11%)")

# Varying interception rates
print("\n--- QBER vs Eve interception rate ---")
print(f"  {'Intercept %':>12} {'QBER':>10} {'Detected':>10}")
print(f"  {'-' * 35}")
for frac in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]:
    result = bb84_protocol(10000, eve_present=frac > 0, eve_fraction=frac)
    print(f"  {frac * 100:12.0f}% {result['qber']:10.4f} {'Yes' if result['eve_detected'] else 'No':>10}")
```

### 9.2 E91 Protocol Simulation

```python
import numpy as np

def e91_protocol(n_pairs, eve_present=False, seed=42):
    """Simulate the E91 entanglement-based QKD protocol.

    E91 uses entangled Bell pairs and the CHSH inequality for security.
    Alice and Bob each measure in one of three bases, and the subset
    where they use the same basis gives the key bits.

    The CHSH violation certifies genuine quantum entanglement,
    which guarantees no eavesdropper has measured the qubits.

    Args:
        n_pairs: Number of Bell pairs distributed
        eve_present: Whether Eve performs entanglement-based attack
        seed: Random seed

    Returns:
        Dict with protocol results including CHSH value
    """
    rng = np.random.default_rng(seed)

    # Alice's bases: 0°, 22.5°, 45° (indices 0, 1, 2)
    # Bob's bases: 22.5°, 45°, 67.5° (indices 0, 1, 2)
    alice_angles = np.array([0, np.pi / 8, np.pi / 4])
    bob_angles = np.array([np.pi / 8, np.pi / 4, 3 * np.pi / 8])

    alice_basis_choices = rng.integers(0, 3, n_pairs)
    bob_basis_choices = rng.integers(0, 3, n_pairs)

    alice_results = np.zeros(n_pairs, dtype=int)
    bob_results = np.zeros(n_pairs, dtype=int)

    for i in range(n_pairs):
        a = alice_angles[alice_basis_choices[i]]
        b = bob_angles[bob_basis_choices[i]]

        if eve_present:
            # Eve's attack reduces the entanglement quality
            # Model: Werner state with visibility v < 1
            v = 0.5  # Reduced visibility due to Eve
        else:
            v = 1.0

        # For a Bell pair |Phi+>, the correlation is:
        # P(same) = cos^2(a-b)/2,  P(different) = sin^2(a-b)/2
        # With visibility v: P(same) = (1+v*cos(2(a-b)))/2

        p_same = (1 + v * np.cos(2 * (a - b))) / 2

        # Alice gets +1 or -1 with equal probability
        alice_results[i] = rng.integers(0, 2)  # 0 or 1

        # Bob's result is correlated with Alice's
        if rng.random() < p_same:
            bob_results[i] = alice_results[i]  # Same
        else:
            bob_results[i] = 1 - alice_results[i]  # Different

    # Sifting: keep pairs where they used the same effective basis
    # Alice basis 1 (22.5°) = Bob basis 0 (22.5°)
    # Alice basis 2 (45°) = Bob basis 1 (45°)
    key_mask = ((alice_basis_choices == 1) & (bob_basis_choices == 0) |
                (alice_basis_choices == 2) & (bob_basis_choices == 1))

    key_alice = alice_results[key_mask]
    key_bob = bob_results[key_mask]

    # CHSH test: use pairs where bases differ
    # Need correlations E(a_i, b_j) for specific combinations
    def compute_correlation(a_idx, b_idx):
        mask = (alice_basis_choices == a_idx) & (bob_basis_choices == b_idx)
        if np.sum(mask) == 0:
            return 0
        a = 2 * alice_results[mask] - 1  # Convert to +1/-1
        b = 2 * bob_results[mask] - 1
        return np.mean(a * b)

    # CHSH: S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
    # Alice: a1=0° (idx 0), a2=45° (idx 2)
    # Bob: b1=22.5° (idx 0), b2=67.5° (idx 2)
    E_00 = compute_correlation(0, 0)
    E_02 = compute_correlation(0, 2)
    E_20 = compute_correlation(2, 0)
    E_22 = compute_correlation(2, 2)

    S = abs(E_00 - E_02 + E_20 + E_22)

    qber = np.mean(key_alice != key_bob) if len(key_alice) > 0 else 0

    return {
        'n_pairs': n_pairs,
        'n_key': len(key_alice),
        'qber': qber,
        'CHSH_S': S,
        'CHSH_violation': S > 2,
        'quantum_certified': S > 2,
        'eve_detected': S <= 2 if eve_present else False,
    }


# Demonstrate E91
print("=" * 65)
print("E91 Entanglement-Based QKD")
print("=" * 65)

print("\n--- No eavesdropper ---")
result = e91_protocol(50000, eve_present=False)
print(f"  Bell pairs used: {result['n_pairs']}")
print(f"  Key length: {result['n_key']}")
print(f"  QBER: {result['qber']:.4f}")
print(f"  CHSH S = {result['CHSH_S']:.4f} (quantum bound: 2*sqrt(2) = {2*np.sqrt(2):.4f})")
print(f"  CHSH violation (S > 2): {result['CHSH_violation']}")

print("\n--- Eavesdropper present ---")
result = e91_protocol(50000, eve_present=True)
print(f"  Bell pairs used: {result['n_pairs']}")
print(f"  QBER: {result['qber']:.4f}")
print(f"  CHSH S = {result['CHSH_S']:.4f}")
print(f"  CHSH violation: {result['CHSH_violation']}")
print(f"  Eve detected (S <= 2): {not result['CHSH_violation']}")
```

### 9.3 Entanglement Swapping

```python
import numpy as np

def bell_state(idx=0):
    """Create a Bell state density matrix.

    Bell states:
    0: |Phi+> = (|00> + |11>) / sqrt(2)
    1: |Phi-> = (|00> - |11>) / sqrt(2)
    2: |Psi+> = (|01> + |10>) / sqrt(2)
    3: |Psi-> = (|01> - |10>) / sqrt(2)
    """
    states = {
        0: np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        1: np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        2: np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        3: np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }
    psi = states[idx]
    return np.outer(psi, psi.conj())


def entanglement_swap(rho_AC, rho_CB):
    """Perform entanglement swapping on systems A-C and C-B.

    Charlie performs a Bell measurement on his two qubits (C1, C2),
    projecting Alice and Bob into an entangled state.

    The 4-qubit state A-C1-C2-B is projected onto Bell states of C1C2,
    leaving AB in a Bell state (up to local corrections).

    Args:
        rho_AC: Density matrix of Alice-Charlie1 pair (4x4)
        rho_CB: Density matrix of Charlie2-Bob pair (4x4)

    Returns:
        rho_AB: Density matrix of Alice-Bob after swap (4x4)
        outcome: Bell measurement outcome (0-3)
        fidelity: Fidelity with target Bell state
    """
    # Full 4-qubit state: A (x) C1 (x) C2 (x) B
    rho_full = np.kron(rho_AC, rho_CB)  # 16x16

    # Bell measurement projectors on C1-C2 (qubits 1 and 2 in 4-qubit system)
    bell_projectors = []
    for i in range(4):
        psi = bell_state(i)
        # Need to embed in 4-qubit space: I_A (x) |Bell><Bell|_{C1C2} (x) I_B
        proj = np.kron(np.eye(2), np.kron(psi, np.eye(2)))
        bell_projectors.append(proj)

    # Probabilities of each outcome
    probs = [np.real(np.trace(proj @ rho_full)) for proj in bell_projectors]

    # Choose outcome (randomly weighted by probability)
    rng = np.random.default_rng()
    outcome = rng.choice(4, p=np.array(probs) / sum(probs))

    # Post-measurement state of AB (trace out C)
    proj = bell_projectors[outcome]
    rho_post = proj @ rho_full @ proj

    # Normalize
    if np.trace(rho_post) > 1e-10:
        rho_post = rho_post / np.trace(rho_post)

    # Partial trace over C1 C2 to get AB state
    # rho_post is 16x16 (ACAB). Reshape and trace
    rho_post_reshaped = rho_post.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    rho_AB = np.einsum('aibjaibj->abab', rho_post_reshaped.reshape(2, 2, 2, 2, 2, 2, 2, 2))
    # Simplified: just trace out middle qubits
    rho_AB_simple = np.zeros((4, 4), dtype=complex)
    for c1 in range(2):
        for c2 in range(2):
            for a1 in range(2):
                for b1 in range(2):
                    for a2 in range(2):
                        for b2 in range(2):
                            i = a1 * 8 + c1 * 4 + c2 * 2 + b1
                            j = a2 * 8 + c1 * 4 + c2 * 2 + b2
                            if i < 16 and j < 16:
                                rho_AB_simple[a1 * 2 + b1, a2 * 2 + b2] += rho_post[i, j]

    if np.abs(np.trace(rho_AB_simple)) > 1e-10:
        rho_AB_simple /= np.trace(rho_AB_simple)

    # Fidelity with |Phi+>
    phi_plus = bell_state(0)
    fidelity = np.real(np.trace(phi_plus @ rho_AB_simple))

    return rho_AB_simple, outcome, max(fidelity, 1 - fidelity)


# Demonstrate entanglement swapping
print("=" * 65)
print("Entanglement Swapping")
print("=" * 65)

# Perfect Bell pairs
rho_AC = bell_state(0)  # |Phi+> for Alice-Charlie
rho_CB = bell_state(0)  # |Phi+> for Charlie-Bob

print("\nPerfect Bell pairs:")
rho_AB, outcome, fidelity = entanglement_swap(rho_AC, rho_CB)
print(f"  Bell measurement outcome: {outcome}")
print(f"  Fidelity of Alice-Bob state: {fidelity:.4f}")

# Noisy Bell pairs (Werner state: F * |Phi+><Phi+| + (1-F)/4 * I)
print("\nNoisy Bell pairs (Werner states):")
for F in [1.0, 0.95, 0.9, 0.8, 0.7]:
    rho_AC_noisy = F * bell_state(0) + (1 - F) / 4 * np.eye(4)
    rho_CB_noisy = F * bell_state(0) + (1 - F) / 4 * np.eye(4)

    fidelities = []
    for trial in range(100):
        _, _, f = entanglement_swap(rho_AC_noisy, rho_CB_noisy)
        fidelities.append(f)

    print(f"  Input fidelity: {F:.2f}, Output fidelity: {np.mean(fidelities):.4f} +/- {np.std(fidelities):.4f}")
```

### 9.4 Quantum Repeater Chain

```python
import numpy as np

def quantum_repeater_chain(total_distance, n_segments, fiber_loss_db_per_km=0.2,
                            link_fidelity=0.98, swap_fidelity=0.99, n_trials=1000,
                            seed=42):
    """Simulate a quantum repeater chain.

    The chain divides the total distance into n_segments, generates
    entanglement in each segment, then performs entanglement swapping
    to extend entanglement across the full distance.

    Args:
        total_distance: Total distance in km
        n_segments: Number of repeater segments
        fiber_loss_db_per_km: Fiber loss in dB/km
        link_fidelity: Fidelity of each elementary link
        swap_fidelity: Fidelity of each entanglement swap
        n_trials: Number of simulation trials
        seed: Random seed

    Returns:
        Dict with success rate, average fidelity, key rate estimate
    """
    rng = np.random.default_rng(seed)

    segment_length = total_distance / n_segments
    segment_transmission = 10 ** (-fiber_loss_db_per_km * segment_length / 10)

    successes = 0
    fidelities = []

    for _ in range(n_trials):
        # Step 1: Generate elementary links
        all_links_up = True
        for seg in range(n_segments):
            if rng.random() > segment_transmission:
                all_links_up = False
                break

        if not all_links_up:
            continue

        # Step 2: Entanglement swapping
        current_fidelity = link_fidelity
        for swap in range(n_segments - 1):
            # Each swap reduces fidelity
            # For Werner states: F_out = F1 * F2 * swap_fidelity (approximate)
            current_fidelity = current_fidelity * link_fidelity * swap_fidelity
            # Better model: F_out = F1*F2 + (1-F1)(1-F2)/3
            F1, F2 = current_fidelity, swap_fidelity
            current_fidelity = F1 * F2 + (1 - F1) * (1 - F2) / 3

        successes += 1
        fidelities.append(current_fidelity)

    success_rate = successes / n_trials
    avg_fidelity = np.mean(fidelities) if fidelities else 0
    key_rate = success_rate * max(0, 1 - 2 * binary_entropy(1 - avg_fidelity)) if avg_fidelity > 0.5 else 0

    return {
        'total_distance': total_distance,
        'n_segments': n_segments,
        'segment_length': segment_length,
        'segment_transmission': segment_transmission,
        'success_rate': success_rate,
        'avg_fidelity': avg_fidelity,
        'key_rate': key_rate,
    }


def binary_entropy(p):
    """Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Demonstrate quantum repeater chain
print("=" * 65)
print("Quantum Repeater Chain Simulation")
print("=" * 65)

# Direct transmission vs repeater
print("\nDirect transmission (no repeaters):")
for dist in [10, 50, 100, 200, 500]:
    transmission = 10 ** (-0.2 * dist / 10)
    print(f"  {dist:>5d} km: transmission = {transmission:.2e}")

print("\nRepeater chain (varying segments):")
print(f"  {'Distance':>10} {'Segments':>10} {'Seg len':>10} {'Success':>10} {'Fidelity':>10}")
print(f"  {'-' * 55}")

for dist in [100, 200, 500]:
    for n_seg in [1, 2, 4, 8]:
        result = quantum_repeater_chain(dist, n_seg, n_trials=5000)
        print(f"  {dist:10d} {n_seg:10d} {result['segment_length']:10.1f}"
              f" {result['success_rate']:10.4f} {result['avg_fidelity']:10.4f}")
```

---

## 10. Exercises

### Exercise 1: BB84 Security Analysis

(a) Implement BB84 with a photon-number-splitting attack: Eve splits multi-photon pulses and measures one copy.
(b) Compare the QBER introduced by intercept-resend vs. PNS attacks.
(c) Implement the decoy-state protocol and show it detects PNS attacks.
(d) Calculate the secret key rate as a function of distance for standard BB84 and decoy-state BB84.

### Exercise 2: CHSH Test

(a) Simulate the CHSH test with perfect Bell pairs and verify $S = 2\sqrt{2}$.
(b) Add noise: Werner state $\rho = v|\Phi^+\rangle\langle\Phi^+| + (1-v)I/4$. At what visibility $v^*$ does $S$ drop below 2?
(c) Implement the CHSH test with detector efficiency $\eta < 1$ (detection loophole). At what $\eta$ does the loophole-free test fail?

### Exercise 3: Repeater Optimization

For a 500 km quantum link:
(a) Compute the optimal number of repeater segments as a function of link loss and swap fidelity.
(b) Include quantum memory decoherence: memories lose fidelity at rate $e^{-t/T_2}$. How does memory coherence time $T_2$ affect the optimal architecture?
(c) Compare 1st generation (swap + distillation) with 2nd generation (QEC-based) repeaters.

### Exercise 4: Entanglement Distillation

Implement the BBPSSW distillation protocol:
(a) Start with 100 Werner states of fidelity $F = 0.7$.
(b) Apply one round of distillation. How many pairs survive? What is the output fidelity?
(c) Apply multiple rounds until $F > 0.99$. How many initial pairs are consumed?
(d) Plot the distillation yield (output/input pairs) vs. target fidelity.

### Exercise 5: Quantum Network Routing

Design a routing algorithm for a quantum network:
(a) Create a graph with 10 nodes and random link qualities (fidelities).
(b) Find the path between two nodes that maximizes the end-to-end fidelity.
(c) Implement entanglement swapping along the path and compute the actual fidelity.
(d) Compare greedy routing (best next hop) with global optimal routing.

---

[← Previous: Topological Quantum Computing](22_Topological_Quantum_Computing.md) | [Next: Qiskit Deep Dive →](24_Qiskit_Deep_Dive.md)

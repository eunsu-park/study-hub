# Lesson 12: Zero-Knowledge Proofs

**Previous**: [Post-Quantum Cryptography](./11_Post_Quantum_Cryptography.md) | **Next**: [Homomorphic Encryption](./13_Homomorphic_Encryption.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of zero-knowledge proofs using concrete analogies and formal definitions
2. Define the three properties of ZKPs: completeness, soundness, and zero-knowledge
3. Implement the Schnorr identification protocol and understand its security reduction
4. Apply the Fiat-Shamir heuristic to transform interactive proofs into non-interactive ones
5. Distinguish between zk-SNARKs and zk-STARKs and their respective trade-offs
6. Implement a simple zero-knowledge proof for graph 3-coloring
7. Identify real-world applications of ZKPs in blockchain, identity, and voting systems

---

Can you prove that you know something without revealing what you know? This seemingly paradoxical question has a precise mathematical answer: **zero-knowledge proofs (ZKPs)**. Introduced by Goldwasser, Micali, and Rackoff in 1985, ZKPs are one of the most surprising and powerful ideas in theoretical computer science. They have transformed from a purely theoretical concept into practical tools powering privacy-preserving cryptocurrencies, decentralized identity systems, and verifiable computation. In this lesson, we build intuition from simple examples and work our way up to the cutting-edge proof systems that are reshaping digital privacy.

## Table of Contents

1. [The Intuition: Proving Without Revealing](#1-the-intuition-proving-without-revealing)
2. [Formal Definitions](#2-formal-definitions)
3. [The Schnorr Identification Protocol](#3-the-schnorr-identification-protocol)
4. [The Fiat-Shamir Heuristic](#4-the-fiat-shamir-heuristic)
5. [Zero-Knowledge Proof for Graph 3-Coloring](#5-zero-knowledge-proof-for-graph-3-coloring)
6. [zk-SNARKs](#6-zk-snarks)
7. [zk-STARKs](#7-zk-starks)
8. [Applications](#8-applications)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)

---

## 1. The Intuition: Proving Without Revealing

### 1.1 The Ali Baba Cave

The classic analogy (by Quisquater et al., 1990):

> **Analogy:** A zero-knowledge proof is like proving you know the secret word to open a magic door in a cave — without revealing the word itself. Imagine a cave shaped like a ring with a locked door at the back. Alice (prover) enters from either the left or right path. Bob (verifier) shouts which side he wants Alice to exit from. If Alice knows the secret word, she can always comply (opening the door if needed). If she doesn't, she has only a 50% chance of guessing correctly. After 20 rounds, Bob is $1 - 2^{-20}$ confident that Alice knows the word, yet he learned nothing about what the word is.

```
         Entrance
         /      \
        /        \
       A          B
       |          |
       |  [DOOR]  |
       |          |
        \        /
         \      /
```

### 1.2 More Everyday Examples

| Scenario | What Is Proved | What Is NOT Revealed |
|----------|---------------|---------------------|
| Age verification | "I am over 18" | Exact age, birth date |
| Solvency proof | "I have > $1M in assets" | Exact balance, account details |
| Password authentication | "I know the password" | The password itself |
| Credential verification | "I have a valid driver's license" | License number, address |

### 1.3 Why ZKPs Matter

- **Privacy**: Prove statements without exposing underlying data
- **Scalability**: Verify computations without re-executing them (succinct proofs)
- **Regulation compliance**: Prove compliance without revealing business secrets

---

## 2. Formal Definitions

### 2.1 Interactive Proof System

An **interactive proof system** for a language $L$ consists of two probabilistic polynomial-time machines:
- **Prover** $P$: Has unlimited computational power (or knows a witness $w$)
- **Verifier** $V$: Runs in polynomial time

They exchange messages, and at the end, $V$ outputs accept or reject.

### 2.2 The Three Properties

For a proof system $(P, V)$ for language $L$:

**1. Completeness**: If the statement is true, the honest prover can convince the honest verifier.

$$
x \in L \implies \Pr[V \text{ accepts after interacting with } P] \geq 1 - \text{negl}(\lambda)
$$

**2. Soundness**: If the statement is false, no cheating prover can convince the verifier (except with negligible probability).

$$
x \notin L \implies \forall P^*, \Pr[V \text{ accepts after interacting with } P^*] \leq \text{negl}(\lambda)
$$

**3. Zero-Knowledge**: The verifier learns nothing beyond the truth of the statement. Formally, there exists a simulator $S$ that, given only the statement (not the witness), can produce a transcript indistinguishable from a real interaction.

$$
\text{View}_V(P(w) \leftrightarrow V)(x) \approx_c S(x)
$$

### 2.3 Flavors of Zero-Knowledge

| Type | Indistinguishability | Strength |
|------|---------------------|----------|
| **Perfect ZK** | Identical distributions | Strongest (information-theoretic) |
| **Statistical ZK** | Negligible statistical distance | Strong |
| **Computational ZK** | Indistinguishable by poly-time algorithms | Standard (most practical schemes) |

---

## 3. The Schnorr Identification Protocol

### 3.1 Setup

- **Public parameters**: Prime $p$, generator $g$ of a subgroup of order $q$
- **Prover's secret**: $x \in \mathbb{Z}_q$
- **Prover's public key**: $y = g^x \bmod p$
- **Goal**: Prover convinces verifier that they know $x$ (the discrete log of $y$)

### 3.2 Protocol Steps

| Step | Prover | | Verifier |
|------|--------|-|----------|
| 1 | Choose random $r \in \mathbb{Z}_q$ | | |
| 2 | Compute $R = g^r \bmod p$ | $\xrightarrow{R}$ | |
| 3 | | $\xleftarrow{c}$ | Choose random challenge $c \in \mathbb{Z}_q$ |
| 4 | Compute $s = r + cx \bmod q$ | $\xrightarrow{s}$ | |
| 5 | | | Verify: $g^s \equiv R \cdot y^c \pmod{p}$ |

### 3.3 Why It Works

**Completeness**: $g^s = g^{r+cx} = g^r \cdot g^{cx} = R \cdot (g^x)^c = R \cdot y^c$

**Soundness**: If the prover can answer two different challenges $c_1, c_2$ for the same $R$, then $s_1 - s_2 = (c_1 - c_2)x$, revealing $x = (s_1 - s_2)(c_1 - c_2)^{-1} \bmod q$. This proves the prover "knows" $x$.

**Zero-Knowledge**: A simulator can produce valid-looking transcripts without knowing $x$:
1. Choose random $s$ and $c$
2. Compute $R = g^s \cdot y^{-c}$
3. The triple $(R, c, s)$ is indistinguishable from a real transcript

### 3.4 Implementation

```python
"""
Schnorr Zero-Knowledge Identification Protocol.

Why Schnorr and not other ZKP protocols? Schnorr is the simplest
honest-verifier ZKP for discrete log knowledge, and it's the building
block for Schnorr signatures (used in Bitcoin's Taproot, EdDSA).
"""

import secrets
import hashlib


class SchnorrZKP:
    """
    Schnorr Zero-Knowledge Proof of knowledge of discrete log.

    Security relies on the hardness of the discrete logarithm problem
    in the subgroup of order q in Z_p*.
    """

    def __init__(self, p: int, q: int, g: int):
        """
        Initialize with group parameters.

        Why separate p and q? We work in a subgroup of order q (prime)
        inside Z_p*. This ensures all operations are in a prime-order group,
        which prevents small-subgroup attacks.
        """
        self.p = p  # Large prime
        self.q = q  # Prime order of subgroup
        self.g = g  # Generator of subgroup

    def keygen(self) -> tuple[int, int]:
        """Generate (public_key, secret_key) pair."""
        x = secrets.randbelow(self.q - 1) + 1  # Secret key
        y = pow(self.g, x, self.p)               # Public key
        return y, x

    def prover_commit(self, secret_key: int) -> tuple[int, int]:
        """
        Prover's first message: commitment R = g^r mod p.

        Why random r? The randomness r is essential for zero-knowledge.
        Without it (or if r is reused), the secret key leaks.
        This is exactly what happened with the PlayStation 3 ECDSA hack
        — Sony reused the nonce, and the private key was extracted.
        """
        r = secrets.randbelow(self.q - 1) + 1
        R = pow(self.g, r, self.p)
        return R, r  # r is kept secret by prover

    def verifier_challenge(self) -> int:
        """Verifier's random challenge."""
        return secrets.randbelow(self.q - 1) + 1

    def prover_respond(self, r: int, c: int, secret_key: int) -> int:
        """
        Prover's response: s = r + c*x mod q.

        Why mod q (not mod p)? We're doing arithmetic in the exponent,
        which is modulo the group order q (by Fermat's little theorem).
        """
        s = (r + c * secret_key) % self.q
        return s

    def verify(self, public_key: int, R: int, c: int, s: int) -> bool:
        """
        Verify: g^s == R * y^c (mod p).

        Why this equation? It checks that s = r + cx without
        knowing r or x individually. The verifier can compute
        both sides using only public information.
        """
        lhs = pow(self.g, s, self.p)
        rhs = (R * pow(public_key, c, self.p)) % self.p
        return lhs == rhs

    def simulate(self, public_key: int) -> tuple[int, int, int]:
        """
        Simulator: produce a valid-looking transcript WITHOUT the secret.

        This proves zero-knowledge: if a simulator can produce
        indistinguishable transcripts, the real interaction reveals
        nothing beyond what the verifier could compute alone.
        """
        s = secrets.randbelow(self.q - 1) + 1
        c = secrets.randbelow(self.q - 1) + 1

        # Compute R = g^s * y^(-c) mod p
        # Why this formula? We need R such that g^s = R * y^c,
        # i.e., R = g^s * y^(-c). This "reverse engineers" a valid R.
        y_inv_c = pow(public_key, self.q - c, self.p)  # y^(-c) mod p
        R = (pow(self.g, s, self.p) * y_inv_c) % self.p

        return R, c, s


def demo_schnorr():
    """Demonstrate the Schnorr ZKP protocol."""

    # Small parameters for demonstration
    # In practice: p ~ 2048 bits, q ~ 256 bits
    # Here: p = 23, q = 11, g = 4 (g has order 11 in Z_23*)
    p, q, g = 23, 11, 4

    zkp = SchnorrZKP(p, q, g)

    # Key generation
    public_key, secret_key = zkp.keygen()
    print(f"Public key y = {public_key}, Secret key x = {secret_key}")
    print(f"Verification: g^x mod p = {pow(g, secret_key, p)} == {public_key}")

    # Run protocol multiple rounds
    print("\n--- Interactive Protocol (5 rounds) ---")
    for i in range(5):
        # Prover commits
        R, r = zkp.prover_commit(secret_key)

        # Verifier challenges
        c = zkp.verifier_challenge()

        # Prover responds
        s = zkp.prover_respond(r, c, secret_key)

        # Verifier checks
        valid = zkp.verify(public_key, R, c, s)
        print(f"  Round {i+1}: R={R:3d}, c={c:3d}, s={s:3d} → {'ACCEPT' if valid else 'REJECT'}")

    # Demonstrate simulation (no secret key needed!)
    print("\n--- Simulated Transcripts (no secret key!) ---")
    for i in range(5):
        R, c, s = zkp.simulate(public_key)
        valid = zkp.verify(public_key, R, c, s)
        print(f"  Simulated {i+1}: R={R:3d}, c={c:3d}, s={s:3d} → {'VALID' if valid else 'INVALID'}")

    print("\n  Note: Simulated transcripts are indistinguishable from real ones!")


if __name__ == "__main__":
    demo_schnorr()
```

---

## 4. The Fiat-Shamir Heuristic

### 4.1 The Problem with Interactive Proofs

Interactive proofs require real-time communication between prover and verifier. This is impractical for many applications:
- Blockchain: proofs must be verified by anyone, at any time
- Digital signatures: the "verifier" is whoever checks the signature later
- Batch verification: verifying thousands of proofs one at a time is slow

### 4.2 The Transformation

The **Fiat-Shamir heuristic** (1986) replaces the verifier's random challenge with the output of a hash function:

$$
c = H(R \| \text{message} \| \text{context})
$$

Since the hash function is deterministic and unpredictable, it acts as a "virtual verifier" — the prover cannot predict the challenge when choosing $R$, so soundness is preserved.

### 4.3 From Schnorr ZKP to Schnorr Signature

Applying Fiat-Shamir to the Schnorr protocol yields **Schnorr signatures** (as discussed in Lesson 7):

1. Choose random $r$, compute $R = g^r$
2. Compute challenge: $c = H(R \| m)$
3. Compute response: $s = r + cx$
4. Signature: $(R, s)$ or $(c, s)$

Verification: check $g^s = R \cdot y^c$ where $c = H(R \| m)$.

```python
"""
Fiat-Shamir transform: from interactive Schnorr ZKP to non-interactive proof.

Why hash-based challenge? The key insight is that the hash function
is a "random oracle" — its output is unpredictable until you commit
to the input. Since R is committed before c is computed, the prover
cannot cheat by choosing R to match a predetermined c.
"""

import secrets
import hashlib


def fiat_shamir_schnorr_prove(p: int, q: int, g: int,
                                x: int, message: bytes) -> tuple[int, int, int]:
    """
    Non-interactive Schnorr proof via Fiat-Shamir.

    Returns (R, c, s) that anyone can verify without interaction.
    """
    # Step 1: Commit
    r = secrets.randbelow(q - 1) + 1
    R = pow(g, r, p)

    # Step 2: Challenge via hash (replaces verifier)
    # Why include the message? It binds the proof to a specific context,
    # preventing the proof from being replayed in a different context.
    hash_input = f"{R}{message.hex()}".encode()
    c_bytes = hashlib.sha256(hash_input).digest()
    c = int.from_bytes(c_bytes, 'big') % q

    # Step 3: Respond
    s = (r + c * x) % q

    return R, c, s


def fiat_shamir_schnorr_verify(p: int, q: int, g: int,
                                  y: int, message: bytes,
                                  R: int, c: int, s: int) -> bool:
    """
    Verify a non-interactive Schnorr proof.

    Anyone can run this with just the public key and proof —
    no interaction with the prover needed.
    """
    # Recompute challenge
    hash_input = f"{R}{message.hex()}".encode()
    c_check_bytes = hashlib.sha256(hash_input).digest()
    c_check = int.from_bytes(c_check_bytes, 'big') % q

    if c != c_check:
        return False  # Challenge doesn't match

    # Verify g^s == R * y^c (mod p)
    lhs = pow(g, s, p)
    rhs = (R * pow(y, c, p)) % p
    return lhs == rhs


# Demo
p, q, g = 23, 11, 4
x = 7  # Secret
y = pow(g, x, p)  # Public key

message = b"I know the secret key"

# Prove (non-interactively)
R, c, s = fiat_shamir_schnorr_prove(p, q, g, x, message)
print(f"Proof: R={R}, c={c}, s={s}")

# Verify (anyone can do this)
valid = fiat_shamir_schnorr_verify(p, q, g, y, message, R, c, s)
print(f"Verification: {'VALID' if valid else 'INVALID'}")

# Tampered message fails
valid2 = fiat_shamir_schnorr_verify(p, q, g, y, b"Different message", R, c, s)
print(f"Tampered verification: {'VALID' if valid2 else 'INVALID'}")
```

### 4.4 Security of Fiat-Shamir

In the **random oracle model** (where the hash function is modeled as a truly random function), Fiat-Shamir preserves soundness. In the standard model (without random oracles), the situation is more nuanced — there exist contrived examples where Fiat-Shamir is insecure, but for practical schemes like Schnorr, it is considered safe.

---

## 5. Zero-Knowledge Proof for Graph 3-Coloring

### 5.1 Why Graph Coloring?

The graph 3-coloring problem is NP-complete. Goldreich, Micali, and Wigderson (1986) showed that any NP statement has a zero-knowledge proof by reducing it to graph 3-coloring. This is a foundational result: if graph 3-coloring has a ZKP, then **every** NP statement has a ZKP.

### 5.2 The Protocol

Given a graph $G = (V, E)$ with a valid 3-coloring $\chi: V \to \{1, 2, 3\}$:

**Each round:**
1. **Prover**: Randomly permute the 3 colors (so the actual coloring is hidden). Commit to each vertex's (permuted) color using a cryptographic commitment.
2. **Verifier**: Choose a random edge $(u, v)$.
3. **Prover**: Open the commitments for $u$ and $v$.
4. **Verifier**: Check that $u$ and $v$ have different colors.

**Soundness**: If the coloring is invalid, at least one edge has same-colored endpoints. The verifier catches this with probability $\geq 1/|E|$ per round. After $k \cdot |E|$ rounds, the cheating probability drops to $(1 - 1/|E|)^{k \cdot |E|} \approx e^{-k}$.

**Zero-knowledge**: The color permutation ensures the verifier sees random colors each round — learning two colors of one edge reveals nothing about the actual coloring.

### 5.3 Implementation

```python
"""
Zero-Knowledge Proof for Graph 3-Coloring.

Why graph coloring? It's NP-complete, so a ZKP for it implies ZKPs
exist for ALL NP problems (via polynomial-time reductions). This is
the most general ZKP construction, though not the most efficient.
"""

import secrets
import hashlib
import json
from typing import Optional


class Commitment:
    """
    Simple hash-based commitment scheme.

    Commit(x) = H(x || r) where r is random.

    Why commitment? It provides two properties:
    - Hiding: The commitment reveals nothing about x
    - Binding: The committer cannot change x after committing
    """

    @staticmethod
    def commit(value: int) -> tuple[bytes, bytes]:
        """Return (commitment, opening) for a value."""
        randomness = secrets.token_bytes(32)
        data = json.dumps({"value": value, "r": randomness.hex()}).encode()
        commitment = hashlib.sha256(data).digest()
        return commitment, data  # data is the opening

    @staticmethod
    def verify(commitment: bytes, opening: bytes, expected_value: int) -> bool:
        """Verify a commitment opens to the expected value."""
        # Check the hash matches
        if hashlib.sha256(opening).digest() != commitment:
            return False
        # Check the value matches
        data = json.loads(opening)
        return data["value"] == expected_value


class GraphColoringZKP:
    """
    Interactive ZKP for graph 3-coloring.

    The prover convinces the verifier that they know a valid
    3-coloring of a graph without revealing the coloring.
    """

    def __init__(self, edges: list[tuple[int, int]], num_vertices: int):
        self.edges = edges
        self.num_vertices = num_vertices

    def prover_commit(self, coloring: dict[int, int]) -> tuple[dict, dict]:
        """
        Prover's first step: permute colors and commit.

        Why permute? Without permutation, the verifier would learn
        the actual colors over many rounds. The random permutation
        ensures each round reveals only random color pairs.
        """
        # Random permutation of {1, 2, 3}
        perm = [1, 2, 3]
        # Fisher-Yates shuffle
        for i in range(2, 0, -1):
            j = secrets.randbelow(i + 1)
            perm[i], perm[j] = perm[j], perm[i]

        # Apply permutation and commit
        commitments = {}
        openings = {}
        for vertex, color in coloring.items():
            permuted_color = perm[color - 1]  # Colors are 1-indexed
            comm, opening = Commitment.commit(permuted_color)
            commitments[vertex] = comm
            openings[vertex] = (opening, permuted_color)

        return commitments, openings

    def verifier_challenge(self) -> tuple[int, int]:
        """Verifier chooses a random edge."""
        idx = secrets.randbelow(len(self.edges))
        return self.edges[idx]

    def prover_respond(self, openings: dict,
                        edge: tuple[int, int]) -> tuple:
        """Prover reveals colors for the challenged edge."""
        u, v = edge
        opening_u, color_u = openings[u]
        opening_v, color_v = openings[v]
        return (opening_u, color_u), (opening_v, color_v)

    def verify_round(self, commitments: dict,
                      edge: tuple[int, int],
                      response: tuple) -> bool:
        """
        Verify one round of the protocol.

        Checks:
        1. Commitments open correctly
        2. The two vertices have different colors
        3. Colors are valid (1, 2, or 3)
        """
        u, v = edge
        (opening_u, color_u), (opening_v, color_v) = response

        # Check commitments
        if not Commitment.verify(commitments[u], opening_u, color_u):
            return False
        if not Commitment.verify(commitments[v], opening_v, color_v):
            return False

        # Check colors are valid
        if color_u not in {1, 2, 3} or color_v not in {1, 2, 3}:
            return False

        # Check colors are different (the actual 3-coloring property)
        return color_u != color_v


def demo_graph_coloring_zkp():
    """Demonstrate ZKP for graph 3-coloring."""

    # Example: A simple graph (triangle + extra vertex)
    #   0 --- 1
    #   |   / |
    #   |  /  |
    #   | /   |
    #   2 --- 3
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    num_vertices = 4

    # Valid 3-coloring (the prover's secret)
    coloring = {0: 1, 1: 2, 2: 3, 3: 1}  # R, G, B, R

    # Verify coloring is actually valid
    for u, v in edges:
        assert coloring[u] != coloring[v], f"Invalid coloring at edge ({u},{v})"

    zkp = GraphColoringZKP(edges, num_vertices)

    # Run multiple rounds
    num_rounds = 20
    all_passed = True

    print(f"Graph: {num_vertices} vertices, {len(edges)} edges")
    print(f"Running {num_rounds} rounds of ZKP...\n")

    for round_num in range(num_rounds):
        # Step 1: Prover commits
        commitments, openings = zkp.prover_commit(coloring)

        # Step 2: Verifier challenges
        edge = zkp.verifier_challenge()

        # Step 3: Prover responds
        response = zkp.prover_respond(openings, edge)

        # Step 4: Verifier checks
        valid = zkp.verify_round(commitments, edge, response)

        if not valid:
            all_passed = False
            print(f"  Round {round_num + 1}: FAILED (edge {edge})")
        else:
            u, v = edge
            (_, cu), (_, cv) = response
            print(f"  Round {round_num + 1}: PASS (edge {edge}, "
                  f"colors: {cu}, {cv})")

    # Soundness: probability of cheating prover passing all rounds
    cheat_prob = (1 - 1/len(edges)) ** num_rounds
    print(f"\nAll rounds passed: {all_passed}")
    print(f"Soundness: cheating prover succeeds with prob <= "
          f"{cheat_prob:.6f}")
    print(f"Verifier confidence: {1 - cheat_prob:.6f}")


if __name__ == "__main__":
    demo_graph_coloring_zkp()
```

---

## 6. zk-SNARKs

### 6.1 What Are zk-SNARKs?

**zk-SNARK**: Zero-Knowledge **Succinct** Non-interactive **Argument** of Knowledge

- **Succinct**: Proof size is $O(\log n)$ or even $O(1)$ — constant regardless of computation size
- **Non-interactive**: Single message from prover to verifier
- **Argument**: Computationally sound (secure against poly-time adversaries, not unbounded)
- **Knowledge**: Prover must "know" a witness, not just convince of a statement

### 6.2 The Pipeline (Simplified)

```
Computation → Arithmetic Circuit → R1CS → QAP → SNARK Proof
```

1. **Computation**: Express the statement as an arithmetic circuit over a finite field
2. **R1CS** (Rank-1 Constraint System): Convert the circuit into a system of constraints of the form $\mathbf{a} \cdot \mathbf{w} \times \mathbf{b} \cdot \mathbf{w} = \mathbf{c} \cdot \mathbf{w}$
3. **QAP** (Quadratic Arithmetic Program): Interpolate constraints into polynomials
4. **SNARK**: Use elliptic curve pairings to create a succinct proof that the polynomial identity holds

### 6.3 Trusted Setup

Most zk-SNARK constructions (notably Groth16) require a **trusted setup ceremony**:

- A **structured reference string (SRS)** is generated using secret randomness
- The secret randomness ("toxic waste") must be destroyed
- If anyone knows the toxic waste, they can create false proofs

This is the main criticism of zk-SNARKs. Mitigations include:
- **MPC ceremonies**: Multiple parties contribute randomness; the setup is secure if at least one participant is honest (e.g., Zcash's ceremony involved thousands of participants)
- **Universal setups**: Schemes like PLONK use a universal SRS that works for any circuit (one ceremony for all applications)
- **Transparent setups** (zk-STARKs): No trusted setup needed

### 6.4 Performance Characteristics

| Property | Groth16 | PLONK | Typical Value |
|----------|---------|-------|---------------|
| Proof size | 3 group elements | ~10 group elements | 128-512 bytes |
| Verification time | 3 pairings | ~10 pairings | 2-10 ms |
| Prover time | $O(n \log n)$ | $O(n \log n)$ | Seconds to minutes |
| Trusted setup | Per-circuit | Universal | One-time |

---

## 7. zk-STARKs

### 7.1 What Are zk-STARKs?

**zk-STARK**: Zero-Knowledge **Scalable** **Transparent** Argument of Knowledge

- **Scalable**: Prover time is quasi-linear $O(n \log^2 n)$; verifier time is $O(\log^2 n)$
- **Transparent**: No trusted setup — uses hash functions only
- **Post-quantum**: Security based on collision-resistant hashing (survives quantum computers)

### 7.2 Key Differences from SNARKs

| Property | zk-SNARKs | zk-STARKs |
|----------|-----------|-----------|
| Trusted setup | Required (most) | **Not required** |
| Proof size | ~128-512 bytes | ~40-200 KB |
| Verification time | ~2-5 ms | ~10-50 ms |
| Post-quantum | No (relies on pairings/DLP) | **Yes** (hash-based) |
| Cryptographic assumptions | Elliptic curve pairings, knowledge assumptions | Collision-resistant hashing |
| Prover complexity | $O(n \log n)$ | $O(n \log^2 n)$ |

### 7.3 The Trade-off

- **SNARKs win** when proof size matters (blockchain on-chain verification, bandwidth-constrained)
- **STARKs win** when trust matters (no ceremony), quantum resistance is needed, or proofs are verified off-chain

### 7.4 Hybrid Approaches

Modern systems often combine techniques:
- **Recursive SNARKs**: A SNARK that verifies another SNARK (compressing STARKs)
- **SNARK-of-STARK**: Use a STARK for the bulk of computation, then compress the proof with a SNARK for on-chain verification

---

## 8. Applications

### 8.1 Blockchain Privacy

**Zcash** (2016): Uses zk-SNARKs to prove that a transaction is valid (inputs = outputs, sender has funds) without revealing sender, receiver, or amount. This is "private money" — like digital cash.

**zkRollups** (Ethereum L2): Batch thousands of transactions off-chain, generate a single SNARK/STARK proof of correctness, and post only the proof on-chain. This provides scalability (1000x throughput increase) with the same security as the base chain.

### 8.2 Decentralized Identity

**Verifiable Credentials with ZKPs:**
- Prove "I am over 18" without revealing your birth date
- Prove "I am a licensed doctor" without revealing your license number
- Prove "I live in the EU" without revealing your address

Standards: W3C Verifiable Credentials, BBS+ signatures (which allow selective disclosure)

### 8.3 Voting

ZKPs enable **verifiable yet secret** voting:
- Each voter proves they are eligible and voted exactly once
- The tally can be verified without revealing individual votes
- Systems: Vocdoni, MACI (Minimum Anti-Collusion Infrastructure)

### 8.4 Verifiable Computation

Outsource computation to an untrusted server and verify the result efficiently:
- Cloud computing: Verify that AWS actually ran your computation correctly
- Machine learning: Prove a model was trained on specific data without revealing the data
- Supply chain: Prove a product meets specifications without revealing the manufacturing process

---

## 9. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Zero-knowledge | Prove a statement without revealing the underlying witness |
| Completeness/Soundness/ZK | Three properties every ZKP must satisfy |
| Schnorr protocol | Simple ZKP for discrete log knowledge; foundation for Schnorr signatures |
| Fiat-Shamir | Transform interactive proofs to non-interactive using hash functions |
| Graph coloring ZKP | Proves any NP statement can have a ZKP |
| zk-SNARKs | Tiny proofs (~128 B), fast verification, but need trusted setup |
| zk-STARKs | Larger proofs (~100 KB), no trusted setup, post-quantum |
| Applications | Privacy (Zcash), scalability (zkRollups), identity, voting |

---

## 10. Exercises

### Exercise 1: Schnorr Protocol Security (Conceptual)

1. Explain what happens if the prover reuses the same random $r$ for two different challenges. Show how the verifier can extract the secret key $x$.
2. The PlayStation 3 hack in 2010 exploited this exact vulnerability in ECDSA. Research and summarize how it worked.
3. Why is it critical that $r$ is generated using a cryptographically secure random number generator and not a PRNG seeded with the current time?

### Exercise 2: Fiat-Shamir Implementation (Coding)

Extend the Fiat-Shamir Schnorr example to create a complete non-interactive ZKP system:
1. Allow proving knowledge of a secret key for arbitrary (safe) prime group parameters
2. Include a `prove(secret_key, message)` function and `verify(public_key, message, proof)` function
3. Show that proofs for different messages are unlinkable (a verifier cannot tell if two proofs came from the same prover)

### Exercise 3: Graph Coloring Soundness (Coding + Conceptual)

1. Modify the graph coloring ZKP to use an **invalid** coloring (where at least one edge has the same color on both endpoints)
2. Run the protocol for increasing numbers of rounds (10, 50, 100, 500)
3. Plot the probability that the cheating prover is caught vs. the number of rounds
4. How many rounds are needed for 99.99% confidence?

### Exercise 4: Commitment Scheme (Coding)

Implement a more complete commitment scheme:
1. **Pedersen commitment**: $C = g^v h^r$ (computationally hiding, perfectly binding)
2. Show that Pedersen commitments are homomorphic: $C_1 \cdot C_2 = \text{Commit}(v_1 + v_2, r_1 + r_2)$
3. Use Pedersen commitments in the graph coloring ZKP instead of hash-based commitments
4. Explain why homomorphic commitments are important for building efficient ZKPs

### Exercise 5: ZKP Application Design (Challenging)

Design a zero-knowledge proof system for a simplified "proof of solvency" for a cryptocurrency exchange:
1. The exchange has a Merkle tree of account balances (leaves are $(account\_id, balance)$)
2. The exchange needs to prove that the sum of all balances equals a publicly known total
3. No individual account balance should be revealed
4. Sketch the protocol (what does the prover compute? what does the verifier check?)
5. What ZKP system would you use and why? (Schnorr, graph coloring, SNARK, STARK?)

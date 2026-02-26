"""
Zero-Knowledge Proofs
======================
Schnorr identification protocol (interactive ZKP for discrete log),
Fiat-Shamir heuristic (non-interactive), and graph 3-coloring ZKP.
"""

from __future__ import annotations
import hashlib
import random


# ---------------------------------------------------------------------------
# Discrete Log Setup (small group for demonstration)
# ---------------------------------------------------------------------------

# Why: We work in Z_p* with a safe prime p = 2q+1 and generator g.
# The discrete log problem (DLP) asks: given g and y = g^x mod p, find x.
# ZKPs let us prove "I know x such that y = g^x" WITHOUT revealing x.
def find_safe_prime_and_generator(
    min_val: int = 500,
) -> tuple[int, int, int]:
    """Find safe prime p, Sophie Germain prime q, and generator g."""
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True

    for q in range(min_val, min_val * 10):
        if is_prime(q):
            p = 2 * q + 1
            if is_prime(p):
                # Find generator of order q (subgroup)
                for g in range(2, p):
                    if pow(g, 2, p) != 1 and pow(g, q, p) == 1:
                        return p, q, g
    raise ValueError("No suitable parameters found")


# ---------------------------------------------------------------------------
# Schnorr Identification Protocol (Interactive ZKP)
# ---------------------------------------------------------------------------

# Why: The Schnorr protocol is the canonical interactive ZKP for discrete
# log. It has three properties:
# 1) Completeness: honest prover always convinces honest verifier
# 2) Soundness: cheating prover fails with probability 1-1/q per round
# 3) Zero-knowledge: verifier learns nothing about x (can simulate
#    transcripts without knowing x)

class SchnorrProver:
    """Prover in the Schnorr identification protocol.
    Knows the secret x such that y = g^x mod p.
    """

    def __init__(self, p: int, q: int, g: int, x: int) -> None:
        self.p = p
        self.q = q
        self.g = g
        self.x = x  # secret
        self.y = pow(g, x, p)  # public key
        self._k: int = 0  # ephemeral secret

    def commit(self) -> int:
        """Step 1: Prover sends commitment r = g^k mod p."""
        # Why: k is a random nonce (like in Schnorr signatures). The
        # commitment r hides k due to the DLP. The verifier cannot
        # extract k from r, just as they can't extract x from y.
        self._k = random.randrange(1, self.q)
        r = pow(self.g, self._k, self.p)
        return r

    def respond(self, challenge: int) -> int:
        """Step 3: Prover sends response s = k + c*x mod q."""
        # Why: s = k + c*x mod q. This is the heart of the ZKP:
        # s reveals nothing about x because k is random and unknown
        # to the verifier. But s is "bound" to x through the challenge c.
        return (self._k + challenge * self.x) % self.q


class SchnorrVerifier:
    """Verifier in the Schnorr identification protocol."""

    def __init__(self, p: int, q: int, g: int, y: int) -> None:
        self.p = p
        self.q = q
        self.g = g
        self.y = y  # prover's public key

    def challenge(self) -> int:
        """Step 2: Verifier sends random challenge c."""
        # Why: The challenge must be random and unpredictable to the prover.
        # If the prover could predict c, they could forge a proof without
        # knowing x (breaking soundness).
        return random.randrange(0, self.q)

    def verify(self, r: int, c: int, s: int) -> bool:
        """Check: g^s == r * y^c (mod p).

        This works because g^s = g^{k+cx} = g^k * g^{cx} = r * y^c.
        """
        lhs = pow(self.g, s, self.p)
        rhs = (r * pow(self.y, c, self.p)) % self.p
        return lhs == rhs


def run_schnorr_protocol(
    prover: SchnorrProver,
    verifier: SchnorrVerifier,
    rounds: int = 5,
) -> list[bool]:
    """Run the interactive Schnorr protocol for multiple rounds."""
    results = []
    for _ in range(rounds):
        # 1. Prover commits
        r = prover.commit()
        # 2. Verifier challenges
        c = verifier.challenge()
        # 3. Prover responds
        s = prover.respond(c)
        # 4. Verifier checks
        ok = verifier.verify(r, c, s)
        results.append(ok)
    return results


# ---------------------------------------------------------------------------
# Fiat-Shamir Transform (Non-Interactive ZKP)
# ---------------------------------------------------------------------------

# Why: The Fiat-Shamir heuristic replaces the verifier's random challenge
# with a hash of the commitment. This makes the proof non-interactive
# (no back-and-forth needed), at the cost of relying on the random oracle
# model. This is how Schnorr signatures work — they're non-interactive
# ZKPs of knowledge of a discrete log.
def fiat_shamir_prove(
    p: int, q: int, g: int, x: int
) -> tuple[int, int, int]:
    """Non-interactive proof of knowledge of discrete log.

    Returns (r, c, s) — the proof that can be verified by anyone.
    """
    y = pow(g, x, p)
    k = random.randrange(1, q)
    r = pow(g, k, p)

    # Why: The challenge is H(g || y || r) — a hash of all public values
    # and the commitment. This "fixes" the challenge deterministically,
    # preventing the prover from choosing k after seeing c.
    c_data = f"{g}:{y}:{r}".encode()
    c = int.from_bytes(hashlib.sha256(c_data).digest(), "big") % q

    s = (k + c * x) % q
    return r, c, s


def fiat_shamir_verify(
    p: int, q: int, g: int, y: int,
    proof: tuple[int, int, int],
) -> bool:
    """Verify a Fiat-Shamir non-interactive proof."""
    r, c, s = proof

    # Recompute challenge
    c_data = f"{g}:{y}:{r}".encode()
    c_expected = int.from_bytes(hashlib.sha256(c_data).digest(), "big") % q

    if c != c_expected:
        return False

    lhs = pow(g, s, p)
    rhs = (r * pow(y, c, p)) % p
    return lhs == rhs


# ---------------------------------------------------------------------------
# Graph 3-Coloring ZKP
# ---------------------------------------------------------------------------

# Why: Graph 3-coloring is NP-complete. A ZKP for it demonstrates that
# ZKPs exist for ALL NP problems (since any NP problem can be reduced
# to 3-coloring). The prover shows they know a valid coloring without
# revealing which colors go where.

def generate_sample_graph() -> tuple[int, list[tuple[int, int]]]:
    """Return a small graph with a known 3-coloring."""
    # A simple 5-node graph
    n_vertices = 5
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    return n_vertices, edges


def find_3_coloring(
    n: int, edges: list[tuple[int, int]]
) -> list[int] | None:
    """Find a valid 3-coloring using backtracking (brute force)."""
    coloring = [0] * n

    def is_valid(v: int, c: int) -> bool:
        for u, w in edges:
            if u == v and coloring[w] == c and w < v:
                return False
            if w == v and coloring[u] == c and u < v:
                return False
        return True

    def solve(v: int) -> bool:
        if v == n:
            return True
        for c in range(1, 4):
            if is_valid(v, c):
                coloring[v] = c
                if solve(v + 1):
                    return True
        coloring[v] = 0
        return False

    return coloring if solve(0) else None


# Why: Each round of the ZKP works as follows:
# 1. Prover randomly permutes the 3 colors (so the verifier can't track
#    which "real" color is which across rounds)
# 2. Prover commits to each vertex's (permuted) color using a hash
# 3. Verifier picks a random edge
# 4. Prover reveals the colors of the two endpoints of that edge
# 5. Verifier checks: (a) revealed colors match commitments,
#    (b) the two colors are different
# After many rounds, the verifier is convinced the coloring is valid.

def zkp_3coloring_round(
    coloring: list[int],
    edges: list[tuple[int, int]],
) -> tuple[bool, str]:
    """One round of the graph 3-coloring ZKP.

    Returns (accepted, explanation).
    """
    n = len(coloring)

    # Step 1: Random color permutation
    # Why: Permuting colors each round prevents the verifier from
    # correlating revelations across rounds. After round 1 reveals
    # vertex A is "red", round 2 might show A as "blue" due to
    # re-permutation. The verifier learns nothing about actual colors.
    perm = [0, 1, 2, 3]
    random.shuffle(perm[1:])  # permute colors 1,2,3
    permuted = [perm[c] for c in coloring]

    # Step 2: Commit to each vertex's color
    nonces = [random.getrandbits(128).to_bytes(16, "big") for _ in range(n)]
    commitments = [
        hashlib.sha256(nonces[i] + bytes([permuted[i]])).digest()
        for i in range(n)
    ]

    # Step 3: Verifier picks a random edge
    edge_idx = random.randrange(len(edges))
    u, v = edges[edge_idx]

    # Step 4: Prover reveals colors + nonces for u, v
    revealed_u = (permuted[u], nonces[u])
    revealed_v = (permuted[v], nonces[v])

    # Step 5: Verifier checks
    # Check commitments
    check_u = hashlib.sha256(
        revealed_u[1] + bytes([revealed_u[0]])
    ).digest()
    check_v = hashlib.sha256(
        revealed_v[1] + bytes([revealed_v[0]])
    ).digest()

    if check_u != commitments[u]:
        return False, f"Commitment mismatch for vertex {u}"
    if check_v != commitments[v]:
        return False, f"Commitment mismatch for vertex {v}"

    # Check different colors
    if revealed_u[0] == revealed_v[0]:
        return False, f"Edge ({u},{v}): same color {revealed_u[0]}"

    return True, f"Edge ({u},{v}): colors {revealed_u[0]} != {revealed_v[0]}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Zero-Knowledge Proofs")
    print("=" * 65)

    # Setup
    p, q, g = find_safe_prime_and_generator()
    print(f"\n[Group Parameters]")
    print(f"  Safe prime p = {p}")
    print(f"  Subgroup order q = {q}")
    print(f"  Generator g = {g}")

    # Prover's secret
    x = random.randrange(1, q)
    y = pow(g, x, p)
    print(f"\n  Secret x = {x}")
    print(f"  Public y = g^x = {y}")

    # --- Interactive Schnorr Protocol ---
    print(f"\n[Interactive Schnorr ZKP]")
    prover = SchnorrProver(p, q, g, x)
    verifier = SchnorrVerifier(p, q, g, y)

    rounds = 10
    results = run_schnorr_protocol(prover, verifier, rounds)
    print(f"  Ran {rounds} rounds:")
    for i, ok in enumerate(results):
        print(f"    Round {i+1}: {'ACCEPT' if ok else 'REJECT'}")
    print(f"  All accepted: {all(results)}")

    # Cheating prover (doesn't know x)
    print(f"\n  Cheating prover (random x):")
    fake_x = random.randrange(1, q)
    cheater = SchnorrProver(p, q, g, fake_x)
    fake_results = run_schnorr_protocol(cheater, verifier, rounds)
    # Note: cheater has wrong x, so verification fails
    accepted = sum(fake_results)
    print(f"  Accepted {accepted}/{rounds} rounds")
    if fake_x == x:
        print(f"  (Cheater accidentally guessed the right x!)")
    else:
        print(f"  (Cheater's x = {fake_x}, real x = {x})")

    # --- Fiat-Shamir (Non-Interactive) ---
    print(f"\n[Fiat-Shamir Non-Interactive ZKP]")
    proof = fiat_shamir_prove(p, q, g, x)
    r, c, s = proof
    print(f"  Proof: (r={r}, c={c}, s={s})")

    valid = fiat_shamir_verify(p, q, g, y, proof)
    print(f"  Verification: {'VALID' if valid else 'INVALID'}")

    # Wrong public key
    y_wrong = pow(g, x + 1, p)
    valid_wrong = fiat_shamir_verify(p, q, g, y_wrong, proof)
    print(f"  Wrong public key: {'VALID' if valid_wrong else 'INVALID'}")

    # --- Graph 3-Coloring ZKP ---
    print(f"\n[Graph 3-Coloring ZKP]")
    n_vertices, edges = generate_sample_graph()
    print(f"  Graph: {n_vertices} vertices, {len(edges)} edges")
    print(f"  Edges: {edges}")

    coloring = find_3_coloring(n_vertices, edges)
    if coloring is None:
        print(f"  No 3-coloring found!")
    else:
        color_names = {0: "-", 1: "R", 2: "G", 3: "B"}
        print(f"  Coloring: {[color_names[c] for c in coloring]}")

        # Run many rounds
        rounds = 20
        results = []
        for _ in range(rounds):
            ok, explanation = zkp_3coloring_round(coloring, edges)
            results.append((ok, explanation))

        print(f"\n  Running {rounds} ZKP rounds:")
        for i, (ok, exp) in enumerate(results[:5]):
            print(f"    Round {i+1}: {'ACCEPT' if ok else 'REJECT'} — {exp}")
        print(f"    ... ({rounds - 5} more rounds)")
        print(f"  All accepted: {all(ok for ok, _ in results)}")

        # Probability analysis
        print(f"\n  [Soundness Analysis]")
        print(f"    Edges = {len(edges)}")
        print(f"    Cheating probability per round: "
              f"1 - 1/{len(edges)} = {1 - 1/len(edges):.4f}")
        prob_cheat_all = (1 - 1 / len(edges)) ** rounds
        print(f"    Prob(cheat {rounds} rounds): "
              f"(1 - 1/{len(edges)})^{rounds} = {prob_cheat_all:.6f}")
        print(f"    Confidence after {rounds} rounds: "
              f"{(1 - prob_cheat_all) * 100:.4f}%")

    print(f"\n{'=' * 65}")

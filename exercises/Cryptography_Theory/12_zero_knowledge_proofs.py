"""
Exercises for Lesson 12: Zero-Knowledge Proofs
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import hashlib
import json
import math
import secrets


def exercise_1():
    """Exercise 1: Schnorr Protocol Security (Conceptual + Coding)

    1. Show what happens when the prover reuses the same random r
    2. Demonstrate secret key extraction from two transcripts with same R
    3. Explain why time-seeded PRNG is dangerous for nonce generation
    """
    print(f"  Schnorr Protocol: Nonce Reuse Attack")

    # Setup: small parameters for demonstration
    p, q, g = 23, 11, 4

    # Secret key
    x = 7
    y = pow(g, x, p)
    print(f"    Public key y = {y}, Secret x = {x} (attacker does NOT know this)")
    print(f"    Group: p={p}, q={q}, g={g}")

    # Part 1: Normal operation (fresh r each round)
    print(f"\n  Part 1: Normal Schnorr (fresh nonce each round)")
    for round_num in range(3):
        r = secrets.randbelow(q - 1) + 1
        R = pow(g, r, p)
        c = secrets.randbelow(q - 1) + 1
        s = (r + c * x) % q
        lhs = pow(g, s, p)
        rhs = (R * pow(y, c, p)) % p
        print(f"    Round {round_num + 1}: r={r}, R={R}, c={c}, s={s}, "
              f"valid={lhs == rhs}")

    # Part 2: Nonce reuse -- two rounds with SAME r
    print(f"\n  Part 2: Nonce Reuse Attack (SAME r in two rounds)")
    r_reused = 5
    R_reused = pow(g, r_reused, p)

    c1 = 3
    s1 = (r_reused + c1 * x) % q

    c2 = 8
    s2 = (r_reused + c2 * x) % q

    print(f"    Both rounds use r={r_reused}, R={R_reused}")
    print(f"    Round A: c1={c1}, s1={s1}")
    print(f"    Round B: c2={c2}, s2={s2}")

    # Attacker observes (R, c1, s1) and (R, c2, s2) with SAME R
    # s1 = r + c1*x  and  s2 = r + c2*x
    # s1 - s2 = (c1 - c2)*x
    # x = (s1 - s2) * (c1 - c2)^(-1) mod q
    s_diff = (s1 - s2) % q
    c_diff = (c1 - c2) % q
    c_diff_inv = pow(c_diff, -1, q)
    x_recovered = (s_diff * c_diff_inv) % q

    print(f"\n    Attacker computes:")
    print(f"    s1 - s2 = {s_diff}")
    print(f"    c1 - c2 = {c_diff}")
    print(f"    (c1 - c2)^(-1) mod {q} = {c_diff_inv}")
    print(f"    x = (s1-s2) * (c1-c2)^(-1) = {x_recovered}")
    print(f"    Secret key recovered: {x_recovered == x}")

    # Part 3: PS3 ECDSA hack analogy
    print(f"\n  Part 3: PlayStation 3 ECDSA Nonce Reuse (2010)")
    print(f"    Sony used a FIXED nonce k for all ECDSA signatures.")
    print(f"    ECDSA signature: s = k^(-1) * (hash(m) + r*x) mod n")
    print(f"    Two signatures with same k:")
    print(f"      s1 = k^(-1) * (h1 + r*x)")
    print(f"      s2 = k^(-1) * (h2 + r*x)")
    print(f"      s1 - s2 = k^(-1) * (h1 - h2)")
    print(f"      k = (h1 - h2) / (s1 - s2)")
    print(f"      x = (s1*k - h1) / r")
    print(f"    Result: Sony's PS3 private signing key was fully recovered.")

    # Part 4: Why time-seeded PRNG is dangerous
    print(f"\n  Part 4: Why time-seeded PRNG is dangerous")
    print(f"    If r = PRNG(time), attacker can:")
    print(f"    1. Observe the signature timestamp (often in the message)")
    print(f"    2. Try all PRNG seeds near that timestamp")
    print(f"    3. For each candidate r, check if g^r mod p == R")
    print(f"    4. Once r is found: x = (s - r) * c^(-1) mod q")

    # Demonstrate: with 1-second resolution, only ~1000 seeds to try
    timestamp = 1700000000  # example timestamp
    test_prng = random.Random(timestamp)
    r_from_time = test_prng.randrange(1, q)
    R_from_time = pow(g, r_from_time, p)

    # Attacker tries timestamps near the known time
    found = False
    for t_guess in range(timestamp - 5, timestamp + 5):
        guess_prng = random.Random(t_guess)
        r_guess = guess_prng.randrange(1, q)
        if pow(g, r_guess, p) == R_from_time:
            found = True
            print(f"    Brute-force: found r={r_guess} at timestamp offset "
                  f"{t_guess - timestamp}")
            break
    print(f"    Nonce recovered by brute-force: {found}")
    print(f"    Always use secrets.randbelow() or os.urandom() for nonces!")


def exercise_2():
    """Exercise 2: Fiat-Shamir Non-Interactive ZKP (Coding)

    1. Implement complete Fiat-Shamir Schnorr NIZK
    2. Prove and verify for arbitrary group parameters
    3. Show proof unlinkability (different messages, same prover)
    """
    print(f"  Fiat-Shamir Non-Interactive Schnorr ZKP")

    # Parameters: safe prime group
    # p = 23, q = 11, g = 4 (g has order 11 in Z_23*)
    p, q, g = 23, 11, 4

    def prove(p, q, g, secret_key, message):
        """Generate a non-interactive ZKP of discrete log knowledge."""
        y = pow(g, secret_key, p)

        # Commit
        r = secrets.randbelow(q - 1) + 1
        R = pow(g, r, p)

        # Challenge = H(g || y || R || message)
        hash_input = "{}:{}:{}:{}".format(g, y, R, message.hex())
        c_bytes = hashlib.sha256(hash_input.encode()).digest()
        c = int.from_bytes(c_bytes, "big") % q
        if c == 0:
            c = 1  # avoid degenerate case

        # Respond
        s = (r + c * secret_key) % q

        return {"R": R, "c": c, "s": s, "y": y}

    def verify(p, q, g, public_key, message, proof):
        """Verify a non-interactive ZKP."""
        R, c, s, y = proof["R"], proof["c"], proof["s"], proof["y"]

        if y != public_key:
            return False

        # Recompute challenge
        hash_input = "{}:{}:{}:{}".format(g, y, R, message.hex())
        c_bytes = hashlib.sha256(hash_input.encode()).digest()
        c_check = int.from_bytes(c_bytes, "big") % q
        if c_check == 0:
            c_check = 1

        if c != c_check:
            return False

        # Verify g^s == R * y^c mod p
        lhs = pow(g, s, p)
        rhs = (R * pow(y, c, p)) % p
        return lhs == rhs

    # Part 1: Basic prove/verify
    x = 7
    y = pow(g, x, p)
    print(f"\n  Part 1: Prove and Verify")
    print(f"    Secret key x = {x}, Public key y = {y}")

    msg = b"I know the discrete log of y"
    proof = prove(p, q, g, x, msg)
    valid = verify(p, q, g, y, msg, proof)
    print(f"    Message: {msg.decode()}")
    print(f"    Proof: R={proof['R']}, c={proof['c']}, s={proof['s']}")
    print(f"    Valid: {valid}")

    # Tampered message
    bad_msg = b"Tampered message"
    valid_bad = verify(p, q, g, y, bad_msg, proof)
    print(f"    Tampered message valid: {valid_bad}")

    # Part 2: Multiple proofs for different messages
    print(f"\n  Part 2: Proofs for Different Messages")
    messages = [
        b"Transfer $100 to Alice",
        b"Transfer $200 to Bob",
        b"Authorize login at 10:00",
    ]
    for msg in messages:
        proof = prove(p, q, g, x, msg)
        valid = verify(p, q, g, y, msg, proof)
        print(f"    Msg: {msg.decode()[:30]:30s} -> R={proof['R']:3d}, "
              f"c={proof['c']:3d}, s={proof['s']:3d}, valid={valid}")

    # Part 3: Unlinkability demonstration
    print(f"\n  Part 3: Proof Unlinkability")
    print(f"    Can a verifier tell if two proofs came from the same prover?")

    # Two provers, same group
    x_alice = 3
    y_alice = pow(g, x_alice, p)
    x_bob = 9
    y_bob = pow(g, x_bob, p)

    msg1 = b"Message from prover A"
    msg2 = b"Message from prover B"

    # Generate many proofs from each prover
    alice_R_values = []
    bob_R_values = []
    for _ in range(100):
        pa = prove(p, q, g, x_alice, msg1)
        pb = prove(p, q, g, x_bob, msg2)
        alice_R_values.append(pa["R"])
        bob_R_values.append(pb["R"])

    # Statistical comparison of R values
    alice_avg = sum(alice_R_values) / len(alice_R_values)
    bob_avg = sum(bob_R_values) / len(bob_R_values)
    print(f"    Alice's avg R: {alice_avg:.1f}")
    print(f"    Bob's avg R:   {bob_avg:.1f}")
    print(f"    R values are random in both cases -- no statistical")
    print(f"    difference reveals the prover's identity.")
    print(f"    The proofs are unlinkable (given different messages).")
    print(f"    Note: if the SAME public key y appears in both proofs,")
    print(f"    they are trivially linkable. Unlinkability requires")
    print(f"    different public keys or anonymous credential schemes.")


def exercise_3():
    """Exercise 3: Graph Coloring Soundness (Coding + Conceptual)

    1. Run the graph coloring ZKP with an INVALID coloring
    2. Measure detection rate vs number of rounds
    3. Compute required rounds for 99.99% confidence
    """
    print(f"  Graph Coloring ZKP: Soundness Analysis")

    # Commitment helpers
    def commit(value):
        randomness = secrets.token_bytes(32)
        data = json.dumps({"v": value, "r": randomness.hex()}).encode()
        commitment = hashlib.sha256(data).digest()
        return commitment, data, value

    def verify_commitment(commitment, opening, expected_value):
        if hashlib.sha256(opening).digest() != commitment:
            return False
        data = json.loads(opening)
        return data["v"] == expected_value

    # Graph: pentagon
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    num_vertices = 5

    # INVALID coloring: vertex 0 and vertex 1 share color 1
    invalid_coloring = {0: 1, 1: 1, 2: 2, 3: 3, 4: 2}
    bad_edges = [(u, v) for u, v in edges if invalid_coloring[u] == invalid_coloring[v]]
    total_edges = len(edges)
    print(f"    Graph: {num_vertices} vertices, {total_edges} edges (pentagon)")
    print(f"    Invalid coloring: {invalid_coloring}")
    print(f"    Bad edges (same color): {bad_edges}")
    print(f"    P(catch per round) = {len(bad_edges)}/{total_edges} = "
          f"{len(bad_edges)/total_edges:.3f}")

    # Part 1: Run cheating prover for various round counts
    print(f"\n  Part 1: Detection Rate vs Number of Rounds")
    print(f"    {'Rounds':>8} {'Caught':>8} {'Escaped':>8} {'P(escape)':>12} {'Theory':>12}")
    print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")

    p_escape_per_round = 1 - len(bad_edges) / total_edges
    trials = 2000

    for num_rounds in [1, 5, 10, 20, 50, 100]:
        caught_count = 0
        for _ in range(trials):
            caught = False
            for _ in range(num_rounds):
                # Prover commits with random permutation
                perm = [1, 2, 3]
                for i in range(2, 0, -1):
                    j = random.randint(0, i)
                    perm[i], perm[j] = perm[j], perm[i]

                commitments = {}
                openings = {}
                for v, color in invalid_coloring.items():
                    permuted = perm[color - 1]
                    comm, opening, val = commit(permuted)
                    commitments[v] = comm
                    openings[v] = (opening, permuted)

                # Verifier picks random edge
                edge_idx = random.randint(0, total_edges - 1)
                u, v = edges[edge_idx]
                opening_u, color_u = openings[u]
                opening_v, color_v = openings[v]

                # Verify commitments
                valid_u = verify_commitment(commitments[u], opening_u, color_u)
                valid_v = verify_commitment(commitments[v], opening_v, color_v)

                if not valid_u or not valid_v or color_u == color_v:
                    caught = True
                    break

            if caught:
                caught_count += 1

        escape_count = trials - caught_count
        p_escape = escape_count / trials
        p_theory = p_escape_per_round ** num_rounds
        print(f"    {num_rounds:>8} {caught_count:>8} {escape_count:>8} "
              f"{p_escape:>12.4f} {p_theory:>12.4f}")

    # Part 2: Required rounds for various confidence levels
    print(f"\n  Part 2: Rounds Needed for Target Confidence")
    print(f"    {'Confidence':>12} {'P(escape)':>12} {'Rounds':>8}")
    print(f"    {'-'*12} {'-'*12} {'-'*8}")

    for target in [0.99, 0.999, 0.9999, 0.99999]:
        # (1 - bad/total)^rounds <= 1 - target
        # rounds >= log(1 - target) / log(1 - bad/total)
        p_esc = 1 - target
        rounds_needed = math.ceil(
            math.log(p_esc) / math.log(p_escape_per_round)
        )
        print(f"    {target:>12.5f} {p_esc:>12.6f} {rounds_needed:>8}")

    print(f"\n    For {total_edges} edges with {len(bad_edges)} bad edge(s):")
    print(f"    99.99% confidence requires "
          f"{math.ceil(math.log(0.0001) / math.log(p_escape_per_round))} rounds")


def exercise_4():
    """Exercise 4: Pedersen Commitment Scheme (Coding)

    1. Implement Pedersen commitments
    2. Demonstrate homomorphic property
    3. Use Pedersen commitments in graph coloring ZKP
    """
    print(f"  Pedersen Commitment Scheme")

    # Group parameters: p = 23, g = 4 (order 11), h = 9 (order 11)
    p, q = 23, 11
    g, h = 4, 9

    # Part 1: Basic Pedersen commitment
    print(f"\n  Part 1: Basic Pedersen Commitment")
    print(f"    C = g^v * h^r mod p, where g={g}, h={h}, p={p}")

    def pedersen_commit(value, p, g, h):
        r = secrets.randbelow(p - 1) + 1
        c = (pow(g, value, p) * pow(h, r, p)) % p
        return c, r

    def pedersen_verify(commitment, value, r, p, g, h):
        expected = (pow(g, value, p) * pow(h, r, p)) % p
        return commitment == expected

    v1, v2 = 3, 7
    c1, r1 = pedersen_commit(v1, p, g, h)
    c2, r2 = pedersen_commit(v2, p, g, h)

    print(f"    Commit({v1}) = {c1} (r={r1})")
    print(f"    Commit({v2}) = {c2} (r={r2})")
    print(f"    Verify Commit({v1}): {pedersen_verify(c1, v1, r1, p, g, h)}")
    print(f"    Verify Commit({v2}): {pedersen_verify(c2, v2, r2, p, g, h)}")

    # Part 2: Homomorphic property
    print(f"\n  Part 2: Homomorphic Property")
    print(f"    Commit(a) * Commit(b) = Commit(a+b, r1+r2)")

    c_sum = (c1 * c2) % p
    r_sum = (r1 + r2) % (p - 1)
    v_sum = (v1 + v2) % q

    print(f"    Commit({v1}) * Commit({v2}) mod {p} = {c_sum}")
    expected_sum = (pow(g, v_sum, p) * pow(h, r_sum, p)) % p
    print(f"    Commit({v_sum}, r1+r2) = {expected_sum}")
    print(f"    Homomorphic: {c_sum == expected_sum}")

    # Demonstrate with multiple additions
    print(f"\n    Chained additions:")
    values = [2, 5, 8, 1]
    commitments = []
    randomnesses = []
    running_commitment = 1
    running_r = 0

    for val in values:
        c, r = pedersen_commit(val, p, g, h)
        commitments.append(c)
        randomnesses.append(r)
        running_commitment = (running_commitment * c) % p
        running_r = (running_r + r) % (p - 1)

    total_value = sum(values) % q
    expected = (pow(g, total_value, p) * pow(h, running_r, p)) % p
    print(f"    Values: {values}, Sum mod {q} = {total_value}")
    print(f"    Product of commitments = {running_commitment}")
    print(f"    Commit(sum, sum_r) = {expected}")
    print(f"    Homomorphic chain valid: {running_commitment == expected}")

    # Part 3: Graph coloring ZKP with Pedersen commitments
    print(f"\n  Part 3: Graph Coloring ZKP with Pedersen Commitments")

    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    num_vertices = 4
    coloring = {0: 1, 1: 2, 2: 3, 3: 1}

    print(f"    Graph: {num_vertices} vertices, {len(edges)} edges")
    print(f"    Coloring: {coloring}")

    num_rounds = 15
    all_passed = True

    for round_num in range(num_rounds):
        # Random permutation of colors
        perm = [1, 2, 3]
        for i in range(2, 0, -1):
            j = secrets.randbelow(i + 1)
            perm[i], perm[j] = perm[j], perm[i]

        # Commit to permuted colors using Pedersen
        vertex_commitments = {}
        vertex_openings = {}
        for v, color in coloring.items():
            permuted_color = perm[color - 1]
            c, r = pedersen_commit(permuted_color, p, g, h)
            vertex_commitments[v] = c
            vertex_openings[v] = (permuted_color, r)

        # Verifier picks random edge
        edge = edges[secrets.randbelow(len(edges))]
        u, v = edge
        color_u, r_u = vertex_openings[u]
        color_v, r_v = vertex_openings[v]

        # Verify
        valid_u = pedersen_verify(vertex_commitments[u], color_u, r_u, p, g, h)
        valid_v = pedersen_verify(vertex_commitments[v], color_v, r_v, p, g, h)
        colors_differ = color_u != color_v

        passed = valid_u and valid_v and colors_differ
        if not passed:
            all_passed = False

    print(f"    Ran {num_rounds} rounds: all passed = {all_passed}")
    print(f"\n    Why Pedersen over hash commitments?")
    print(f"    1. Homomorphic: can prove sums of committed values")
    print(f"    2. Perfectly hiding: info-theoretically secure privacy")
    print(f"    3. Enables efficient ZKP constructions (Bulletproofs, etc.)")


def exercise_5():
    """Exercise 5: ZKP for Proof of Solvency (Challenging)

    Design a simplified proof-of-solvency for a cryptocurrency exchange:
    1. Merkle tree of account balances
    2. Prove total matches claimed amount
    3. No individual balance revealed
    """
    print(f"  Zero-Knowledge Proof of Solvency")

    # Part 1: Build a Merkle tree of account balances
    print(f"\n  Part 1: Merkle Tree of Encrypted Balances")

    accounts = [
        ("alice", 1500),
        ("bob", 3200),
        ("carol", 750),
        ("dave", 4100),
        ("eve", 2800),
        ("frank", 900),
        ("grace", 5500),
        ("heidi", 1250),
    ]

    total_balance = sum(bal for _, bal in accounts)
    print(f"    {len(accounts)} accounts, total balance: {total_balance}")

    # Build Merkle tree
    def hash_leaf(account_id, balance):
        salt = hashlib.sha256(account_id.encode()).digest()[:8]
        data = account_id.encode() + balance.to_bytes(8, "big") + salt
        return hashlib.sha256(data).digest()

    def hash_pair(left, right):
        return hashlib.sha256(left + right).digest()

    leaves = [hash_leaf(acct, bal) for acct, bal in accounts]

    # Build tree bottom-up
    tree_levels = [leaves]
    current = leaves
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                next_level.append(hash_pair(current[i], current[i + 1]))
            else:
                next_level.append(current[i])
        tree_levels.append(next_level)
        current = next_level

    merkle_root = current[0]
    print(f"    Merkle root: {merkle_root.hex()[:16]}...")
    print(f"    Tree depth: {len(tree_levels)}")

    # Part 2: Proof of inclusion for a specific account
    print(f"\n  Part 2: Merkle Proof of Inclusion")

    def get_merkle_proof(tree_levels, leaf_index):
        """Get the Merkle proof (sibling hashes) for a leaf."""
        proof = []
        idx = leaf_index
        for level in tree_levels[:-1]:
            if idx % 2 == 0:
                if idx + 1 < len(level):
                    proof.append(("right", level[idx + 1]))
                else:
                    proof.append(("none", b""))
            else:
                proof.append(("left", level[idx - 1]))
            idx //= 2
        return proof

    def verify_merkle_proof(leaf_hash, proof, root):
        """Verify a Merkle proof."""
        current = leaf_hash
        for direction, sibling in proof:
            if direction == "left":
                current = hash_pair(sibling, current)
            elif direction == "right":
                current = hash_pair(current, sibling)
        return current == root

    # Prove carol (index 2) is in the tree
    carol_idx = 2
    carol_leaf = hash_leaf("carol", 750)
    carol_proof = get_merkle_proof(tree_levels, carol_idx)
    carol_valid = verify_merkle_proof(carol_leaf, carol_proof, merkle_root)
    print(f"    Carol's inclusion proof valid: {carol_valid}")
    print(f"    Proof size: {len(carol_proof)} hashes "
          f"(log2({len(accounts)}) = {math.log2(len(accounts)):.0f})")

    # Part 3: Simplified proof of solvency protocol
    print(f"\n  Part 3: Proof of Solvency Protocol")
    print(f"    Claimed total: {total_balance}")

    # The exchange commits to each balance using hash commitments
    # and proves the sum equals the claimed total
    committed_balances = []
    for acct, bal in accounts:
        salt = secrets.token_bytes(16)
        commitment = hashlib.sha256(
            bal.to_bytes(8, "big") + salt
        ).digest()
        committed_balances.append((commitment, bal, salt))

    # Verifiable sum: exchange reveals partial information
    # In a real system, this would use Pedersen commitments + range proofs
    running_sum = 0
    for i, (comm, bal, salt) in enumerate(committed_balances):
        running_sum += bal
        # Verify commitment
        expected = hashlib.sha256(
            bal.to_bytes(8, "big") + salt
        ).digest()
        assert comm == expected

    print(f"    Verified sum of committed balances: {running_sum}")
    print(f"    Matches claimed total: {running_sum == total_balance}")

    # Part 4: Protocol design analysis
    print(f"\n  Part 4: Protocol Design Analysis")
    print(f"    Full ZKP Solvency Protocol:")
    print(f"    1. Exchange publishes Merkle root of (account, balance) pairs")
    print(f"    2. Each user verifies their account is in the tree (Merkle proof)")
    print(f"    3. Exchange proves sum = claimed_total using:")
    print(f"       - Pedersen commitments for each balance (homomorphic)")
    print(f"       - Range proofs (e.g., Bulletproofs) to prove each balance >= 0")
    print(f"       - Product of all commitments = Commit(total)")
    print(f"    4. Exchange proves on-chain reserves >= total (simple verification)")
    print(f"")
    print(f"    Which ZKP system to use?")
    print(f"    - NOT Schnorr: only proves discrete log knowledge, not sums")
    print(f"    - NOT graph coloring: too inefficient for arithmetic statements")
    print(f"    - zk-SNARK (Groth16/PLONK): tiny proof (~128 bytes),")
    print(f"      ideal for on-chain verification, but needs trusted setup")
    print(f"    - zk-STARK: no trusted setup, post-quantum, but larger proofs")
    print(f"    - Best choice: Bulletproofs (range proofs) + Pedersen commitments")
    print(f"      No trusted setup, efficient for range proofs, and")
    print(f"      Pedersen homomorphism gives sum verification for free")


def exercise_6():
    """Exercise 6: zk-SNARK vs zk-STARK Tradeoff Analysis (Conceptual)

    Compare proof systems across multiple dimensions for different use cases.
    """
    print(f"  zk-SNARK vs zk-STARK Tradeoff Analysis")

    # Part 1: Property comparison
    print(f"\n  Part 1: Proof System Comparison")
    print(f"    {'Property':<28} {'Groth16':>12} {'PLONK':>12} {'STARK':>12}")
    print(f"    {'-'*28} {'-'*12} {'-'*12} {'-'*12}")

    properties = [
        ("Proof size", "128 B", "~1 KB", "~100 KB"),
        ("Verification time", "~3 ms", "~10 ms", "~50 ms"),
        ("Prover time", "~10 s", "~30 s", "~60 s"),
        ("Trusted setup", "Per-circuit", "Universal", "None"),
        ("Post-quantum", "No", "No", "Yes"),
        ("Crypto assumption", "Pairing+KEA", "Pairing+ROM", "CRHF only"),
    ]

    for name, groth, plonk, stark in properties:
        print(f"    {name:<28} {groth:>12} {plonk:>12} {stark:>12}")

    # Part 2: Use case recommendations
    print(f"\n  Part 2: Recommended System by Use Case")

    use_cases = [
        ("Blockchain L2 rollup", "Groth16/PLONK",
         "Small proofs minimize on-chain gas cost"),
        ("Private cryptocurrency", "Groth16 or Halo2",
         "Every transaction carries a proof; size matters"),
        ("Verifiable ML inference", "STARK",
         "Large circuits; quasi-linear prover; no trusted setup"),
        ("Confidential transactions", "Bulletproofs",
         "Optimized for range proofs; no trusted setup"),
        ("Government/military", "STARK",
         "Post-quantum; no trusted setup attack surface"),
    ]

    for name, best, reason in use_cases:
        print(f"\n    {name}: {best}")
        print(f"      {reason}")

    # Part 3: On-chain cost comparison
    print(f"\n  Part 3: On-Chain Cost (1M txns/day, 30 gwei, ETH=$3000)")
    gas_per_byte = 16
    print(f"    {'System':<14} {'Proof (B)':>10} {'Daily cost':>14}")
    print(f"    {'-'*14} {'-'*10} {'-'*14}")

    for name, proof_bytes in [("Groth16", 128), ("PLONK", 1024), ("STARK", 102400)]:
        gas = (proof_bytes * gas_per_byte + 200000) * 1_000_000
        usd = gas * 30 / 1e9 * 3000
        print(f"    {name:<14} {proof_bytes:>10,} ${usd:>13,.0f}")

    print(f"\n    Modern approach: STARK proof compressed with SNARK (best of both).")


if __name__ == "__main__":
    print("=== Exercise 1: Schnorr Protocol Security ===")
    exercise_1()

    print("\n=== Exercise 2: Fiat-Shamir Non-Interactive ZKP ===")
    exercise_2()

    print("\n=== Exercise 3: Graph Coloring Soundness ===")
    exercise_3()

    print("\n=== Exercise 4: Pedersen Commitment Scheme ===")
    exercise_4()

    print("\n=== Exercise 5: ZKP for Proof of Solvency ===")
    exercise_5()

    print("\n=== Exercise 6: zk-SNARK vs zk-STARK Tradeoffs ===")
    exercise_6()

    print("\nAll exercises completed!")

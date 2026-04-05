"""
Exercises for Lesson 14: Applied Cryptographic Protocols
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import hashlib
import hmac
import secrets
import math
import itertools


def exercise_1():
    """Exercise 1: TLS 1.3 Key Schedule (Coding)

    Implement the TLS 1.3 key derivation schedule using HKDF.
    1. Compute early secret, handshake secret, and master secret
    2. Derive client/server handshake traffic secrets
    3. Verify key separation properties
    """
    print(f"  TLS 1.3 Key Derivation Schedule")

    # HKDF implementation (RFC 5869) using stdlib hmac
    def hkdf_extract(salt, ikm):
        """HKDF-Extract: PRK = HMAC-Hash(salt, IKM)."""
        if salt is None or len(salt) == 0:
            salt = b"\x00" * 32  # Hash length for SHA-256
        return hmac.new(salt, ikm, hashlib.sha256).digest()

    def hkdf_expand(prk, info, length):
        """HKDF-Expand: OKM = T(1) || T(2) || ... truncated to length."""
        hash_len = 32  # SHA-256
        n = math.ceil(length / hash_len)
        okm = b""
        t = b""
        for i in range(1, n + 1):
            t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
            okm += t
        return okm[:length]

    def hkdf_expand_label(secret, label, context, length):
        """TLS 1.3 HKDF-Expand-Label.

        HkdfLabel = length (2 bytes) || "tls13 " + label || context
        """
        tls_label = b"tls13 " + label.encode()
        # Construct HkdfLabel
        hkdf_label = (
            length.to_bytes(2, "big")
            + len(tls_label).to_bytes(1, "big")
            + tls_label
            + len(context).to_bytes(1, "big")
            + context
        )
        return hkdf_expand(secret, hkdf_label, length)

    def derive_secret(secret, label, messages_hash):
        """Derive-Secret(Secret, Label, Messages)."""
        return hkdf_expand_label(secret, label, messages_hash, 32)

    # Simulate TLS 1.3 handshake inputs
    psk = b"\x00" * 32  # No PSK (use zero)
    ecdhe_shared_secret = secrets.token_bytes(32)  # Simulated ECDHE result

    # Transcript hashes (simulated)
    client_hello_hash = hashlib.sha256(b"ClientHello").digest()
    ch_sh_hash = hashlib.sha256(b"ClientHello||ServerHello").digest()
    ch_sf_hash = hashlib.sha256(b"ClientHello||...||ServerFinished").digest()
    empty_hash = hashlib.sha256(b"").digest()

    # Step 1: Early Secret
    print(f"\n  Step 1: Early Secret (from PSK)")
    early_secret = hkdf_extract(
        salt=b"\x00" * 32,  # No salt for first extract
        ikm=psk
    )
    print(f"    Early Secret: {early_secret.hex()[:32]}...")

    # Derive binder key and early traffic secret
    binder_key = derive_secret(early_secret, "ext binder", empty_hash)
    client_early_traffic = derive_secret(
        early_secret, "c e traffic", client_hello_hash
    )
    print(f"    Binder Key:            {binder_key.hex()[:32]}...")
    print(f"    Client Early Traffic:   {client_early_traffic.hex()[:32]}...")

    # Step 2: Handshake Secret
    print(f"\n  Step 2: Handshake Secret (from ECDHE)")
    derived_early = derive_secret(early_secret, "derived", empty_hash)
    handshake_secret = hkdf_extract(
        salt=derived_early,
        ikm=ecdhe_shared_secret
    )
    print(f"    Handshake Secret: {handshake_secret.hex()[:32]}...")

    # Derive handshake traffic secrets
    client_hs_traffic = derive_secret(
        handshake_secret, "c hs traffic", ch_sh_hash
    )
    server_hs_traffic = derive_secret(
        handshake_secret, "s hs traffic", ch_sh_hash
    )
    print(f"    Client HS Traffic: {client_hs_traffic.hex()[:32]}...")
    print(f"    Server HS Traffic: {server_hs_traffic.hex()[:32]}...")

    # Step 3: Master Secret
    print(f"\n  Step 3: Master Secret")
    derived_hs = derive_secret(handshake_secret, "derived", empty_hash)
    master_secret = hkdf_extract(
        salt=derived_hs,
        ikm=b"\x00" * 32  # No additional keying material
    )
    print(f"    Master Secret: {master_secret.hex()[:32]}...")

    # Derive application traffic secrets
    client_app_traffic = derive_secret(
        master_secret, "c ap traffic", ch_sf_hash
    )
    server_app_traffic = derive_secret(
        master_secret, "s ap traffic", ch_sf_hash
    )
    print(f"    Client App Traffic: {client_app_traffic.hex()[:32]}...")
    print(f"    Server App Traffic: {server_app_traffic.hex()[:32]}...")

    # Step 4: Key separation verification
    print(f"\n  Step 4: Key Separation Properties")
    all_keys = {
        "binder_key": binder_key,
        "client_early_traffic": client_early_traffic,
        "client_hs_traffic": client_hs_traffic,
        "server_hs_traffic": server_hs_traffic,
        "client_app_traffic": client_app_traffic,
        "server_app_traffic": server_app_traffic,
    }

    # Verify all keys are distinct
    key_values = list(all_keys.values())
    all_unique = len(set(k.hex() for k in key_values)) == len(key_values)
    print(f"    All derived keys are unique: {all_unique}")

    # Verify client != server keys
    cs_diff = client_hs_traffic != server_hs_traffic
    print(f"    Client HS != Server HS: {cs_diff}")
    cs_app_diff = client_app_traffic != server_app_traffic
    print(f"    Client App != Server App: {cs_app_diff}")

    # Verify transcript binding
    print(f"\n    Transcript binding: changing any handshake message")
    print(f"    changes all subsequent derived keys. This prevents")
    print(f"    message reordering and tampering attacks.")

    alt_ch_sh_hash = hashlib.sha256(b"ClientHello||TAMPERED").digest()
    alt_client_hs = derive_secret(handshake_secret, "c hs traffic", alt_ch_sh_hash)
    print(f"    Original client HS key: {client_hs_traffic.hex()[:16]}...")
    print(f"    Tampered client HS key: {alt_client_hs.hex()[:16]}...")
    print(f"    Different: {client_hs_traffic != alt_client_hs}")


def exercise_2():
    """Exercise 2: Double Ratchet Simulation (Coding)

    Implement a simplified Double Ratchet with:
    1. Symmetric ratchet (KDF chain)
    2. DH ratchet (new DH per round-trip)
    3. Forward secrecy demonstration
    4. Out-of-order message handling
    """
    print(f"  Double Ratchet Simulation")

    def kdf_chain(chain_key):
        """Advance the symmetric ratchet: derive message key and next chain key."""
        message_key = hmac.new(chain_key, b"message_key", hashlib.sha256).digest()
        next_chain = hmac.new(chain_key, b"next_chain", hashlib.sha256).digest()
        return next_chain, message_key

    def dh_output(priv_a, pub_b):
        """Simulated DH shared secret (in reality: X25519)."""
        combined = priv_a.to_bytes(4, "big") + pub_b.to_bytes(4, "big")
        return hashlib.sha256(combined).digest()

    def kdf_root(root_key, dh_shared):
        """Root KDF: derive new root key and chain key from DH output."""
        info = root_key + dh_shared
        new_root = hmac.new(info, b"root_key", hashlib.sha256).digest()
        new_chain = hmac.new(info, b"chain_key", hashlib.sha256).digest()
        return new_root, new_chain

    def encrypt_msg(key, plaintext):
        """Simplified encryption (XOR with key hash). NOT secure -- demo only."""
        keystream = hashlib.sha256(key + b"encrypt").digest()
        ct = bytes(a ^ b for a, b in zip(plaintext.ljust(32).encode()[:32], keystream))
        return ct

    def decrypt_msg(key, ciphertext):
        """Simplified decryption."""
        keystream = hashlib.sha256(key + b"encrypt").digest()
        pt = bytes(a ^ b for a, b in zip(ciphertext, keystream))
        return pt.rstrip().decode()

    # Initialize: post-X3DH shared root key
    root_key = secrets.token_bytes(32)

    # Alice's initial DH key pair (simulated as simple integers)
    alice_dh_priv = random.randint(1000, 9999)
    alice_dh_pub = alice_dh_priv * 7 % 10007  # simplified "public key"

    # Bob's initial DH key pair
    bob_dh_priv = random.randint(1000, 9999)
    bob_dh_pub = bob_dh_priv * 7 % 10007

    # Part 1: Symmetric ratchet (Alice sends multiple messages)
    print(f"\n  Part 1: Symmetric Ratchet (Alice -> Bob)")

    # Alice computes DH shared secret with Bob's public key
    dh_shared = dh_output(alice_dh_priv, bob_dh_pub)
    root_key, send_chain = kdf_root(root_key, dh_shared)

    messages = ["Hello Bob!", "How are you?", "Let's meet at 3pm", "Bye!"]
    message_keys = []

    for i, msg in enumerate(messages):
        send_chain, msg_key = kdf_chain(send_chain)
        message_keys.append(msg_key)
        ct = encrypt_msg(msg_key, msg)
        print(f"    Msg {i}: key={msg_key.hex()[:16]}... "
              f"ct={ct.hex()[:16]}...")

    # Bob decrypts (he computes the same chain)
    print(f"\n    Bob decrypts:")
    dh_shared_bob = dh_output(bob_dh_priv, alice_dh_pub)
    # Note: in real protocol, Bob derives same chain from DH
    # Here we use Alice's message keys directly for simplicity
    for i, msg in enumerate(messages):
        ct = encrypt_msg(message_keys[i], msg)
        pt = decrypt_msg(message_keys[i], ct)
        print(f"    Msg {i}: decrypted = '{pt}'")

    # Part 2: DH ratchet step (Bob replies)
    print(f"\n  Part 2: DH Ratchet (Bob -> Alice, new DH keys)")

    # Bob generates new DH key pair
    bob_dh_priv_new = random.randint(1000, 9999)
    bob_dh_pub_new = bob_dh_priv_new * 7 % 10007

    # Bob computes new DH shared secret with Alice's public key
    dh_shared_new = dh_output(bob_dh_priv_new, alice_dh_pub)
    root_key, bob_send_chain = kdf_root(root_key, dh_shared_new)

    bob_messages = ["Hi Alice!", "3pm works, see you there"]
    for i, msg in enumerate(bob_messages):
        bob_send_chain, msg_key = kdf_chain(bob_send_chain)
        ct = encrypt_msg(msg_key, msg)
        pt = decrypt_msg(msg_key, ct)
        print(f"    Bob msg {i}: '{pt}' (new DH keys)")

    # Part 3: Forward secrecy
    print(f"\n  Part 3: Forward Secrecy Demonstration")
    print(f"    Compromise current chain key:")
    compromised_chain = send_chain  # attacker gets current chain key
    print(f"    Compromised: {compromised_chain.hex()[:16]}...")

    # Can derive FUTURE message keys from this chain
    future_chain, future_key = kdf_chain(compromised_chain)
    print(f"    Can derive future key: {future_key.hex()[:16]}...")

    # But CANNOT derive PAST message keys
    print(f"    Can derive past keys? NO")
    print(f"    The KDF chain is one-way (hash-based).")
    print(f"    Given chain_key[n], cannot compute chain_key[n-1].")
    print(f"    Past message keys are deleted after use.")

    # After a DH ratchet step, even the current chain is reset
    print(f"\n    After DH ratchet step:")
    print(f"    New DH exchange produces new chain key.")
    print(f"    Compromised chain key is now useless.")
    print(f"    This is POST-COMPROMISE SECURITY (future secrecy).")

    # Part 4: Out-of-order messages
    print(f"\n  Part 4: Out-of-Order Message Handling")

    chain = secrets.token_bytes(32)
    stored_keys = {}

    # Generate 5 message keys
    for i in range(5):
        chain, mk = kdf_chain(chain)
        stored_keys[i] = mk

    # Messages arrive out of order: 0, 2, 4, 1, 3
    arrival_order = [0, 2, 4, 1, 3]
    original_msgs = ["msg-0", "msg-1", "msg-2", "msg-3", "msg-4"]

    print(f"    Messages arrive in order: {arrival_order}")
    for idx in arrival_order:
        mk = stored_keys[idx]
        ct = encrypt_msg(mk, original_msgs[idx])
        pt = decrypt_msg(mk, ct)
        print(f"    Received msg {idx}: '{pt}' (correct)")

    print(f"\n    How it works: receiver stores skipped chain keys.")
    print(f"    When a skipped message arrives, use its stored key.")
    print(f"    Keys are deleted after use (forward secrecy preserved).")


def exercise_3():
    """Exercise 3: Shamir's Secret Sharing Extensions (Coding)

    1. Proactive secret sharing (share refresh)
    2. Verifiable secret sharing (Feldman VSS)
    3. Demonstrate refresh preserves the secret
    """
    print(f"  Shamir's Secret Sharing Extensions")

    # Use a smaller prime for readable output
    p = 104729  # prime

    def split_secret(secret, threshold, num_shares, prime):
        """Split secret using Shamir's scheme."""
        coefficients = [secret]
        for _ in range(threshold - 1):
            coefficients.append(random.randrange(1, prime))

        shares = []
        for i in range(1, num_shares + 1):
            y = 0
            for exp, coeff in enumerate(coefficients):
                y = (y + coeff * pow(i, exp, prime)) % prime
            shares.append((i, y))
        return shares

    def reconstruct_secret(shares, prime):
        """Reconstruct secret using Lagrange interpolation."""
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            num = 1
            den = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    num = (num * (-xj)) % prime
                    den = (den * (xi - xj)) % prime
            lagrange = (num * pow(den, -1, prime)) % prime
            secret = (secret + yi * lagrange) % prime
        return secret

    # Part 1: Basic Shamir
    secret = 42
    threshold = 3
    num_shares = 5

    print(f"\n  Part 1: Basic Shamir Secret Sharing")
    print(f"    Secret: {secret}, Threshold: {threshold}-of-{num_shares}")

    shares = split_secret(secret, threshold, num_shares, p)
    for i, (x, y) in enumerate(shares):
        print(f"    Share {i+1}: ({x}, {y})")

    recovered = reconstruct_secret(shares[:threshold], p)
    print(f"    Recovered from {threshold} shares: {recovered}")
    print(f"    Correct: {recovered == secret}")

    # Part 2: Proactive share refresh
    print(f"\n  Part 2: Proactive Secret Sharing (Share Refresh)")
    print(f"    Refresh shares without changing the secret.")
    print(f"    Generate a random polynomial with f(0) = 0 (zero secret).")
    print(f"    Add the zero-secret shares to the original shares.")

    def refresh_shares(shares, threshold, prime):
        """Refresh shares by adding shares of zero."""
        # Generate random polynomial with constant term = 0
        zero_shares = split_secret(0, threshold, len(shares), prime)

        # Add zero-shares to original shares
        refreshed = []
        for (x, y), (_, y_zero) in zip(shares, zero_shares):
            refreshed.append((x, (y + y_zero) % prime))
        return refreshed

    old_shares = shares[:]
    new_shares = refresh_shares(shares, threshold, p)

    print(f"    Original shares:")
    for x, y in old_shares:
        print(f"      ({x}, {y})")

    print(f"    Refreshed shares:")
    for x, y in new_shares:
        print(f"      ({x}, {y})")

    recovered_new = reconstruct_secret(new_shares[:threshold], p)
    print(f"    Recovered from refreshed shares: {recovered_new}")
    print(f"    Same secret: {recovered_new == secret}")

    # Old shares mixed with new shares should NOT work
    mixed = [old_shares[0], new_shares[1], old_shares[2]]
    recovered_mixed = reconstruct_secret(mixed, p)
    print(f"    Mixed old+new shares: {recovered_mixed}")
    print(f"    Correct: {recovered_mixed == secret} (should be False)")

    # Part 3: Feldman Verifiable Secret Sharing (VSS)
    print(f"\n  Part 3: Feldman Verifiable Secret Sharing")
    print(f"    Each share comes with commitments for verification.")

    # Use a group where DLP is hard (small for demo)
    p_group = 23
    q_group = 11
    g_group = 4  # generator of order 11 in Z_23*

    secret_vss = 7
    t_vss = 3
    n_vss = 5

    # Dealer generates polynomial
    coeffs = [secret_vss]
    for _ in range(t_vss - 1):
        coeffs.append(random.randrange(1, q_group))

    print(f"    Polynomial coefficients: {coeffs}")

    # Commitments: C_k = g^a_k mod p for each coefficient
    commitments = [pow(g_group, a, p_group) for a in coeffs]
    print(f"    Commitments: {commitments}")

    # Generate shares
    vss_shares = []
    for i in range(1, n_vss + 1):
        y = 0
        for exp, coeff in enumerate(coeffs):
            y = (y + coeff * pow(i, exp, q_group)) % q_group
        vss_shares.append((i, y))

    # Verify each share: g^share_i == product(C_k^(i^k)) mod p
    print(f"\n    Verifying shares:")
    for x, y in vss_shares:
        # g^y mod p
        lhs = pow(g_group, y, p_group)
        # product(C_k^(x^k)) mod p
        rhs = 1
        for k, ck in enumerate(commitments):
            rhs = (rhs * pow(ck, pow(x, k, q_group), p_group)) % p_group
        valid = lhs == rhs
        print(f"    Share ({x}, {y}): g^y={lhs}, prod(C_k^(x^k))={rhs}, "
              f"valid={valid}")

    # Tampered share should fail verification
    print(f"\n    Tampered share verification:")
    tampered_x, tampered_y = vss_shares[0][0], (vss_shares[0][1] + 3) % q_group
    lhs = pow(g_group, tampered_y, p_group)
    rhs = 1
    for k, ck in enumerate(commitments):
        rhs = (rhs * pow(ck, pow(tampered_x, k, q_group), p_group)) % p_group
    print(f"    Tampered ({tampered_x}, {tampered_y}): "
          f"g^y={lhs}, expected={rhs}, valid={lhs == rhs}")


def exercise_4():
    """Exercise 4: Secure Coin Flip Protocol (Coding)

    Implement fair coin-flipping using commitment schemes.
    1. Hash-based commitment
    2. Pedersen commitment
    3. Analyze order-dependency
    """
    print(f"  Secure Coin Flip Protocol")

    # Part 1: Hash-based coin flip
    print(f"\n  Part 1: Hash-Based Coin Flip")

    def hash_commit(bit):
        r = secrets.token_bytes(32)
        c = hashlib.sha256(bit.to_bytes(1, "big") + r).digest()
        return c, r

    def hash_verify(commitment, bit, randomness):
        expected = hashlib.sha256(bit.to_bytes(1, "big") + randomness).digest()
        return hmac.compare_digest(commitment, expected)

    results = {0: 0, 1: 0}
    n_flips = 1000

    for _ in range(n_flips):
        # Step 1: Alice commits to random bit a
        a = secrets.randbelow(2)
        comm_a, r_a = hash_commit(a)

        # Step 2: Bob sends random bit b (in the clear)
        b = secrets.randbelow(2)

        # Step 3: Alice opens commitment
        assert hash_verify(comm_a, a, r_a)

        # Step 4: Result = a XOR b
        result = a ^ b
        results[result] += 1

    print(f"    {n_flips} coin flips:")
    print(f"    Heads (0): {results[0]} ({results[0]/n_flips*100:.1f}%)")
    print(f"    Tails (1): {results[1]} ({results[1]/n_flips*100:.1f}%)")
    print(f"    Bias: {abs(results[0] - results[1]) / n_flips:.3f} "
          f"(should be ~0)")

    # Detailed single flip
    print(f"\n    Detailed single flip:")
    a = 1
    comm_a, r_a = hash_commit(a)
    print(f"    Alice commits: a={a}, commitment={comm_a.hex()[:16]}...")
    b = 0
    print(f"    Bob sends: b={b}")
    print(f"    Alice opens: a={a}, randomness={r_a.hex()[:16]}...")
    valid = hash_verify(comm_a, a, r_a)
    print(f"    Bob verifies commitment: {valid}")
    result = a ^ b
    print(f"    Result: {a} XOR {b} = {result}")

    # Part 2: Can Alice cheat?
    print(f"\n  Part 2: Can Alice Cheat?")
    print(f"    After seeing b, Alice wants to change a.")
    print(f"    She committed to a=1 but wants result=0.")
    print(f"    She needs to open commitment as a=0 (since b=0, 0 XOR 0 = 0).")
    cheat_valid = hash_verify(comm_a, 0, r_a)
    print(f"    Open as a=0: valid={cheat_valid}")
    print(f"    Alice CANNOT cheat (binding property of hash commitment).")

    # Can Bob cheat?
    print(f"\n    Can Bob cheat? No -- hiding property prevents learning a.")
    print(f"    Result = a XOR b is uniform if EITHER a or b is random.")

    # Part 3: What if Bob commits first?
    print(f"\n  Part 3: What If Bob Commits First?")
    print(f"    Still secure if commitment scheme is hiding, but")
    print(f"    original order (Alice commits, Bob sends) is simpler to analyze.")


def exercise_5():
    """Exercise 5: Protocol Composition Analysis (Challenging)

    Analyze a secure auction protocol combining multiple primitives.
    """
    print(f"  Secure Auction Protocol Analysis")

    # Part 1: Simulate the auction
    print(f"\n  Part 1: Auction Simulation")

    bidders = [
        ("Alice", 500),
        ("Bob", 750),
        ("Carol", 620),
        ("Dave", 750),
        ("Eve", 480),
    ]

    print(f"    Bidders: {len(bidders)}")
    for name, bid in bidders:
        print(f"    {name}: ${bid} (secret)")

    # Encrypt bids (simplified with hash commitments)
    encrypted_bids = []
    for name, bid in bidders:
        salt = secrets.token_bytes(16)
        commitment = hashlib.sha256(
            bid.to_bytes(8, "big") + salt
        ).digest()
        encrypted_bids.append((name, commitment, bid, salt))
        print(f"    {name}'s encrypted bid: {commitment.hex()[:16]}...")

    # Find winner using comparison (garbled circuit in real system)
    max_bid = max(bid for _, bid in bidders)
    winners = [name for name, bid in bidders if bid == max_bid]

    print(f"\n    Winning bid: ${max_bid}")
    print(f"    Winner(s): {', '.join(winners)}")

    # Part 2: Cryptographic assumptions
    print(f"\n  Part 2: Cryptographic Assumptions")
    assumptions = [
        ("Paillier (additive HE)", "Decisional Composite Residuosity"),
        ("ZKP (SNARK/STARK)", "Soundness (no false proofs)"),
        ("Garbled Circuits", "OT security + CPA security"),
        ("Commitment scheme", "Hiding + Binding"),
    ]
    for prim, assumption in assumptions:
        print(f"    {prim}: {assumption}")

    # Part 3: Malicious auctioneer
    print(f"\n  Part 3: Malicious Auctioneer Threats & Mitigations")
    threats = [
        ("Substitutes a bid", "ZKP proves computation on COMMITTED bids"),
        ("Claims wrong winner", "Bidders verify garbled circuit output"),
        ("Learns individual bids", "Threshold decryption (t-of-n key shares)"),
        ("Colludes with a bidder", "All bids committed before evaluation phase"),
    ]
    for attack, fix in threats:
        print(f"    Attack: {attack} -> {fix}")

    # Part 4: Tied bids
    print(f"\n  Part 4: Handling Tied Bids")
    print(f"    1. First-committed wins (requires trusted timestamp)")
    print(f"    2. Random tiebreaker (XOR of bidder nonces)")
    print(f"    3. Second-price / Vickrey auction (ties irrelevant for price)")

    # Part 5: Communication complexity
    n = len(bidders)
    gc_gates = n * 64 * 4
    gc_kb = gc_gates * 32 / 1024
    print(f"\n  Part 5: Communication Complexity (n={n}, 64-bit bids)")
    print(f"    Paillier ciphertexts: {n * 256} bytes")
    print(f"    ZKP proof (Groth16): 128 bytes (constant)")
    print(f"    Garbled circuit: ~{gc_gates} gates (~{gc_kb:.0f} KB)")
    print(f"    Total: O(n * b * kappa), dominated by garbled circuit")

    # Part 6: Alternative primitives
    print(f"\n  Part 6: Alternative Primitives")
    print(f"    FHE: unified but comparison is expensive (deep circuit)")
    print(f"    MPC with secret sharing: no single decryption key, needs interaction")
    print(f"    zk-SNARK: tiny on-chain footprint but high prover time")


if __name__ == "__main__":
    print("=== Exercise 1: TLS 1.3 Key Schedule ===")
    exercise_1()

    print("\n=== Exercise 2: Double Ratchet Simulation ===")
    exercise_2()

    print("\n=== Exercise 3: Secret Sharing Extensions ===")
    exercise_3()

    print("\n=== Exercise 4: Secure Coin Flip ===")
    exercise_4()

    print("\n=== Exercise 5: Protocol Composition Analysis ===")
    exercise_5()

    print("\nAll exercises completed!")

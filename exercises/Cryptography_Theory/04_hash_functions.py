"""
Exercises for Lesson 04: Hash Functions
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import hashlib
import hmac as hmac_lib
import os
import random
import time


def exercise_1():
    """Exercise 1: Hash Properties (Basic)

    1. Compute SHA-256 of "Hello" and "Hello " (trailing space). Count bit differences.
    2. Find a message whose SHA-256 starts with "0000" (16 zero bits).
    3. Relate to Bitcoin mining difficulty.
    """
    # Part 1: Avalanche effect
    h1 = hashlib.sha256(b"Hello").hexdigest()
    h2 = hashlib.sha256(b"Hello ").hexdigest()

    h1_int = int(h1, 16)
    h2_int = int(h2, 16)
    diff_bits = bin(h1_int ^ h2_int).count('1')

    print(f"  Part 1: Avalanche effect")
    print(f"    SHA-256('Hello')  = {h1}")
    print(f"    SHA-256('Hello ') = {h2}")
    print(f"    Bits different: {diff_bits}/256 ({diff_bits/256*100:.1f}%)")
    print(f"    Expected ~128 bits (50%) due to avalanche property")

    # Part 2: Find hash starting with "0000" (16 zero bits)
    print(f"\n  Part 2: Find message with SHA-256 starting with '0000'")
    attempts = 0
    for nonce in range(10_000_000):
        msg = f"nonce={nonce}".encode()
        h = hashlib.sha256(msg).hexdigest()
        attempts += 1
        if h.startswith("0000"):
            print(f"    Found! Message: {msg.decode()}")
            print(f"    Hash: {h}")
            print(f"    Attempts: {attempts}")
            break

    # Part 3: Bitcoin mining connection
    print(f"\n  Part 3: Bitcoin mining analogy")
    print(f"    Bitcoin requires finding a nonce where SHA-256(block_header) < target.")
    print(f"    Finding 16 leading zero bits took ~{attempts} attempts.")
    print(f"    Expected: 2^16 = {2**16} attempts (birthday math doesn't apply here).")
    print(f"    Bitcoin difficulty as of 2024 requires ~80+ leading zero bits,")
    print(f"    meaning ~2^80 hashes per block -- requiring enormous computation.")


def exercise_2():
    """Exercise 2: Birthday Attack (Intermediate)

    1. Implement birthday attack on truncated hash (first 4 bytes = 32 bits).
    2. Find a collision and count required hashes.
    3. Compare with theoretical sqrt(2^32) ~ 65536.
    """
    import math

    print(f"  Birthday Attack on 32-bit truncated SHA-256")

    hash_dict = {}
    attempts = 0

    while True:
        # Generate random message
        msg = os.urandom(16)
        # Truncate SHA-256 to 4 bytes (32 bits)
        h = hashlib.sha256(msg).digest()[:4]
        attempts += 1

        if h in hash_dict:
            if hash_dict[h] != msg:
                # Found a collision!
                msg1 = hash_dict[h]
                msg2 = msg
                print(f"    Collision found after {attempts} hashes!")
                print(f"    Message 1: {msg1.hex()}")
                print(f"    Message 2: {msg2.hex()}")
                print(f"    Truncated hash: {h.hex()}")
                print(f"    Full SHA-256 msg1: {hashlib.sha256(msg1).hexdigest()[:8]}...")
                print(f"    Full SHA-256 msg2: {hashlib.sha256(msg2).hexdigest()[:8]}...")
                break
        else:
            hash_dict[h] = msg

        if attempts > 500000:
            print(f"    Stopped after {attempts} attempts (should not happen for 32-bit)")
            break

    theoretical = int(math.sqrt(2 ** 32))
    print(f"\n    Theoretical (sqrt(2^32)): ~{theoretical}")
    print(f"    Actual attempts: {attempts}")
    print(f"    Ratio: {attempts / theoretical:.2f}x")
    print(f"    Memory used: {len(hash_dict)} stored hashes ({len(hash_dict) * 20 // 1024} KB)")


def exercise_3():
    """Exercise 3: HMAC Verification (Intermediate)

    1. Implement HMAC-SHA256 from scratch.
    2. Test against hmac library for 10 key/message pairs.
    3. Show H(key || msg) differs from HMAC(key, msg).
    """
    def hmac_sha256(key, message):
        """HMAC-SHA256 from scratch using only hashlib.sha256."""
        block_size = 64  # SHA-256 block size in bytes

        # If key longer than block size, hash it
        if len(key) > block_size:
            key = hashlib.sha256(key).digest()

        # Pad key to block_size
        key = key + b'\x00' * (block_size - len(key))

        # Compute inner and outer padded keys
        o_key_pad = bytes(k ^ 0x5C for k in key)
        i_key_pad = bytes(k ^ 0x36 for k in key)

        # HMAC = H(o_key_pad || H(i_key_pad || message))
        inner_hash = hashlib.sha256(i_key_pad + message).digest()
        return hashlib.sha256(o_key_pad + inner_hash).hexdigest()

    # Part 1 & 2: Test against library
    print(f"  Part 1-2: HMAC-SHA256 from scratch vs library")
    all_match = True
    for i in range(10):
        key = os.urandom(random.randint(8, 64))
        msg = os.urandom(random.randint(0, 200))

        our_hmac = hmac_sha256(key, msg)
        lib_hmac = hmac_lib.new(key, msg, hashlib.sha256).hexdigest()

        match = our_hmac == lib_hmac
        if not match:
            all_match = False
        if i < 3:  # Show first 3 for brevity
            print(f"    Test {i+1}: key={key[:8].hex()}... msg={msg[:8].hex()}...")
            print(f"      Ours:    {our_hmac[:32]}...")
            print(f"      Library: {lib_hmac[:32]}...")
            print(f"      Match: {match}")

    print(f"    All 10 tests match: {all_match}")

    # Part 3: H(key || msg) vs HMAC(key, msg)
    print(f"\n  Part 3: H(key || msg) vs HMAC(key, msg)")
    key = b"my_secret_key"
    msg = b"important data"

    naive = hashlib.sha256(key + msg).hexdigest()
    hmac_result = hmac_sha256(key, msg)

    print(f"    Key: {key.decode()}")
    print(f"    Message: {msg.decode()}")
    print(f"    H(key || msg):  {naive}")
    print(f"    HMAC(key, msg): {hmac_result}")
    print(f"    Different: {naive != hmac_result}")
    print(f"\n    H(key || msg) is vulnerable to length extension attacks.")
    print(f"    HMAC's nested hash construction prevents this.")


def exercise_4():
    """Exercise 4: Length Extension (Challenging)

    Explain the length extension attack and demonstrate HMAC is immune.
    """
    print(f"  Length Extension Attack Explanation")
    print(f"  {'=' * 45}")

    # Demonstrate the concept
    secret = b"secret"
    message = b"data"
    mac = hashlib.sha256(secret + message).hexdigest()

    print(f"    Server computes: MAC = SHA-256(secret || 'data') = {mac[:32]}...")
    print(f"    Attacker knows: MAC value, len(secret) = 6, message = 'data'")
    print()
    print(f"    Attack mechanism:")
    print(f"    1. SHA-256 uses Merkle-Damgard: state after processing 'secret' + 'data'")
    print(f"       is exactly the MAC value (before finalization).")
    print(f"    2. Attacker can compute the padding that SHA-256 would add:")
    print(f"       'secret' + 'data' + 0x80 + 0x00... + length_field")
    print(f"    3. Using MAC as the initial state, attacker continues hashing:")
    print(f"       SHA-256(secret || data || padding || 'evil')")
    print(f"       without knowing 'secret'!")
    print()
    print(f"    The extended message includes internal padding bytes,")
    print(f"    but the MAC is valid for the full extended message.")

    # Show HMAC is immune
    print(f"\n    HMAC immunity:")
    key = b"secret"
    msg = b"data"
    hmac_mac = hmac_lib.new(key, msg, hashlib.sha256).hexdigest()
    print(f"    HMAC(secret, 'data') = {hmac_mac[:32]}...")
    print(f"    HMAC = H(K' XOR opad || H(K' XOR ipad || msg))")
    print(f"    The outer hash H(... || inner_hash) uses a different key derivation.")
    print(f"    Even if the attacker extends the inner hash, the outer hash")
    print(f"    recomputation would require knowing the key, which the")
    print(f"    attacker cannot derive from the HMAC output.")
    print(f"    Result: length extension attack does NOT apply to HMAC.")


def exercise_5():
    """Exercise 5: Password Cracking (Challenging)

    1. Hash 1000 common passwords with SHA-256 (no salt). Time matching.
    2. Repeat with PBKDF2 (100,000 iterations). Compare time.
    3. Demonstrate salt preventing rainbow tables.
    4. Calculate Argon2 memory requirements for parallel attack.
    """
    # Simulate common passwords
    common_passwords = [f"password{i}" for i in range(1000)]
    common_passwords[42] = "correcthorsebatterystaple"
    target_password = "correcthorsebatterystaple"

    # Part 1: SHA-256 (no salt) -- fast
    print(f"  Part 1: SHA-256 password cracking (no salt)")
    sha_hashes = {}
    for pw in common_passwords:
        h = hashlib.sha256(pw.encode()).hexdigest()
        sha_hashes[h] = pw

    target_hash = hashlib.sha256(target_password.encode()).hexdigest()

    start = time.time()
    found = sha_hashes.get(target_hash)
    sha_time = time.time() - start
    print(f"    Target hash: {target_hash[:32]}...")
    print(f"    Found: '{found}' in {sha_time*1000:.3f} ms")
    print(f"    (Lookup in precomputed table is O(1))")

    # Part 2: PBKDF2 (100k iterations) -- slow
    print(f"\n  Part 2: PBKDF2 password cracking")
    salt = os.urandom(16)

    # Hash target
    start = time.time()
    target_pbkdf2 = hashlib.pbkdf2_hmac('sha256', target_password.encode(), salt, 100_000)
    one_hash_time = time.time() - start

    # Try cracking (first 50 passwords for time)
    start = time.time()
    found_pb = None
    checked = 0
    for pw in common_passwords[:100]:
        h = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, 100_000)
        checked += 1
        if h == target_pbkdf2:
            found_pb = pw
            break
    pbkdf2_time = time.time() - start

    print(f"    One PBKDF2 hash: {one_hash_time*1000:.1f} ms")
    print(f"    Checked {checked} passwords in {pbkdf2_time:.2f}s")
    estimated_full = one_hash_time * 1000
    print(f"    Estimated time for 1000 passwords: {estimated_full:.1f}s")
    print(f"    Slowdown vs SHA-256: ~{estimated_full / 0.001:.0f}x")

    # Part 3: Salt prevents rainbow tables
    print(f"\n  Part 3: Salt prevents rainbow tables")
    salt1 = os.urandom(16)
    salt2 = os.urandom(16)
    pw = "password123"
    h1 = hashlib.sha256(salt1 + pw.encode()).hexdigest()
    h2 = hashlib.sha256(salt2 + pw.encode()).hexdigest()
    print(f"    Same password, different salts:")
    print(f"    Salt 1: {salt1.hex()}")
    print(f"    Hash 1: {h1}")
    print(f"    Salt 2: {salt2.hex()}")
    print(f"    Hash 2: {h2}")
    print(f"    Hashes equal: {h1 == h2}")
    print(f"    A precomputed rainbow table for one salt is useless for another.")
    print(f"    With 16-byte random salts, attacker must compute a separate table")
    print(f"    for each user -- effectively requiring online brute force.")

    # Part 4: Argon2 memory calculation
    print(f"\n  Part 4: Argon2 memory requirements for parallel attack")
    memory_per_hash_mb = 64  # Argon2id recommended: 64 MB
    parallel_attempts = 1_000_000
    total_memory_tb = memory_per_hash_mb * parallel_attempts / 1_000_000
    print(f"    Memory per hash (Argon2id): {memory_per_hash_mb} MB")
    print(f"    Parallel attempts: {parallel_attempts:,}")
    print(f"    Total RAM needed: {memory_per_hash_mb} MB * {parallel_attempts:,} = {total_memory_tb:.0f} TB")
    print(f"    This is {total_memory_tb / 1:.0f} TB of RAM -- cost prohibitive!")
    print(f"    Even top-end GPU clusters have ~80 GB per GPU.")
    print(f"    Would need {total_memory_tb * 1000 / 80:.0f} GPUs (${total_memory_tb * 1000 / 80 * 10000:,.0f}+).")
    print(f"    Memory-hardness is Argon2's key advantage over bcrypt/PBKDF2.")


if __name__ == "__main__":
    print("=== Exercise 1: Hash Properties ===")
    exercise_1()

    print("\n=== Exercise 2: Birthday Attack ===")
    exercise_2()

    print("\n=== Exercise 3: HMAC Verification ===")
    exercise_3()

    print("\n=== Exercise 4: Length Extension ===")
    exercise_4()

    print("\n=== Exercise 5: Password Cracking ===")
    exercise_5()

    print("\nAll exercises completed!")

"""
Exercises for Lesson 03: Block Cipher Modes of Operation
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import os
import struct


def toy_encrypt(block, key):
    """Toy block cipher (XOR-based) for demonstration. Replace with AES in production."""
    return bytes(b ^ k for b, k in zip(block, key))


toy_decrypt = toy_encrypt  # XOR is its own inverse


def exercise_1():
    """Exercise 1: ECB Pattern Detection (Basic)

    Create 10 plaintext blocks where 4 are identical.
    Encrypt with ECB and CBC. Count unique ciphertext blocks.
    """
    key = os.urandom(16)
    iv = os.urandom(16)

    # Create 10 blocks: 4 identical (block A), 6 unique
    block_a = bytes([0xAA] * 16)
    blocks = [block_a] * 4 + [os.urandom(16) for _ in range(6)]

    # ECB mode: encrypt each block independently
    ecb_ct = [toy_encrypt(block, key) for block in blocks]
    ecb_unique = len(set(b.hex() for b in ecb_ct))

    # CBC mode: chain blocks together
    cbc_ct = []
    prev = iv
    for block in blocks:
        xored = bytes(p ^ c for p, c in zip(block, prev))
        encrypted = toy_encrypt(xored, key)
        cbc_ct.append(encrypted)
        prev = encrypted
    cbc_unique = len(set(b.hex() for b in cbc_ct))

    print(f"  Input: 10 blocks, 4 identical + 6 unique")
    print(f"  ECB unique ciphertext blocks: {ecb_unique}/10")
    print(f"  CBC unique ciphertext blocks: {cbc_unique}/10")
    print(f"\n  Explanation:")
    print(f"    ECB: identical plaintext blocks -> identical ciphertext blocks")
    print(f"    The 4 identical blocks produce the same ciphertext, so only 7 unique blocks.")
    print(f"    CBC: each block is XORed with the previous ciphertext before encryption.")
    print(f"    Even identical plaintext blocks produce different ciphertext because")
    print(f"    their position (via chaining) makes each input to the cipher unique.")


def exercise_2():
    """Exercise 2: CTR Mode Implementation (Intermediate)

    Implement CTR mode, encrypt a message, then decrypt by re-encrypting.
    """
    key = os.urandom(16)
    nonce = os.urandom(8)

    def ctr_mode(data, key, nonce):
        """CTR mode: encryption and decryption are identical."""
        output = bytearray()
        block_size = 16
        for i in range(0, len(data), block_size):
            # Counter block = nonce (8 bytes) || counter (8 bytes, big-endian)
            counter_block = nonce + struct.pack('>Q', i // block_size)
            keystream = toy_encrypt(counter_block, key)
            block = data[i:i + block_size]
            for j in range(len(block)):
                output.append(block[j] ^ keystream[j])
        return bytes(output)

    message = b"CTR mode is a stream cipher built from a block cipher!"
    ciphertext = ctr_mode(message, key, nonce)
    decrypted = ctr_mode(ciphertext, key, nonce)  # Same operation decrypts!

    print(f"  Plaintext:  {message.decode()}")
    print(f"  Ciphertext: {ciphertext.hex()[:64]}...")
    print(f"  Decrypted:  {decrypted.decode()}")
    print(f"  Round-trip:  {message == decrypted}")
    print(f"\n  Key property: CTR decryption == CTR encryption (XOR is self-inverse)")
    print(f"  No padding needed: plaintext length = {len(message)}, ciphertext length = {len(ciphertext)}")


def exercise_3():
    """Exercise 3: Nonce Reuse Attack (Intermediate)

    1. Encrypt two known messages with same nonce.
    2. Recover XOR of plaintexts from ciphertexts alone.
    3. Given one plaintext, recover the other.
    """
    key = os.urandom(16)
    nonce = os.urandom(8)  # SAME nonce for both -- the vulnerability!

    def ctr_encrypt(data, key, nonce):
        output = bytearray()
        for i in range(0, len(data), 16):
            counter_block = nonce + struct.pack('>Q', i // 16)
            keystream = toy_encrypt(counter_block, key)
            block = data[i:i + 16]
            for j in range(len(block)):
                output.append(block[j] ^ keystream[j])
        return bytes(output)

    msg1 = b"Attack at dawn!!"  # 16 bytes
    msg2 = b"Meet me at noon!"  # 16 bytes

    ct1 = ctr_encrypt(msg1, key, nonce)
    ct2 = ctr_encrypt(msg2, key, nonce)

    # Step 2: XOR of ciphertexts = XOR of plaintexts (keystream cancels!)
    ct_xor = bytes(a ^ b for a, b in zip(ct1, ct2))
    pt_xor = bytes(a ^ b for a, b in zip(msg1, msg2))

    print(f"  Message 1: {msg1}")
    print(f"  Message 2: {msg2}")
    print(f"\n  C1 XOR C2 = {ct_xor.hex()}")
    print(f"  P1 XOR P2 = {pt_xor.hex()}")
    print(f"  C1 XOR C2 == P1 XOR P2: {ct_xor == pt_xor}")

    # Step 3: If attacker knows P1, recover P2
    recovered_msg2 = bytes(a ^ b for a, b in zip(ct_xor, msg1))
    print(f"\n  If attacker knows P1, P2 = (C1 XOR C2) XOR P1")
    print(f"  Recovered P2: {recovered_msg2}")
    print(f"  Correct: {recovered_msg2 == msg2}")

    # Relation to OTP
    print(f"\n  Analysis:")
    print(f"    CTR mode with fixed nonce = one-time pad with REUSED key.")
    print(f"    The keystream acts as the OTP key. Reusing the nonce")
    print(f"    reuses the keystream, which is equivalent to key reuse.")
    print(f"    Shannon proved OTP is perfectly secure ONLY with unique keys.")
    print(f"    Nonce reuse violates this, completely breaking confidentiality.")


def exercise_4():
    """Exercise 4: Padding Oracle Simulator (Challenging)

    Implement a complete padding oracle attack on CBC mode.
    """
    block_size = 16

    def pkcs7_pad(data):
        pad_len = block_size - (len(data) % block_size)
        return data + bytes([pad_len] * pad_len)

    def pkcs7_unpad(data):
        if len(data) == 0 or len(data) % block_size != 0:
            raise ValueError("Invalid length")
        pad_len = data[-1]
        if pad_len == 0 or pad_len > block_size:
            raise ValueError("Invalid padding value")
        for i in range(pad_len):
            if data[-(i + 1)] != pad_len:
                raise ValueError("Invalid padding")
        return data[:-pad_len]

    # Setup: secret message encrypted with CBC
    key = os.urandom(16)
    iv = os.urandom(16)
    secret = b"Top secret msg!!"  # 16 bytes
    padded = pkcs7_pad(secret)

    # CBC encrypt (2 blocks: 16 bytes message + 16 bytes padding)
    ciphertext_blocks = []
    prev = iv
    for i in range(0, len(padded), block_size):
        block = padded[i:i + block_size]
        xored = bytes(p ^ c for p, c in zip(block, prev))
        encrypted = toy_encrypt(xored, key)
        ciphertext_blocks.append(encrypted)
        prev = encrypted

    # The oracle: returns True if padding is valid after decryption
    oracle_queries = 0

    def padding_oracle(modified_prev, target_block):
        nonlocal oracle_queries
        oracle_queries += 1
        decrypted = toy_decrypt(target_block, key)
        plaintext = bytes(d ^ p for d, p in zip(decrypted, modified_prev))
        try:
            pkcs7_unpad(plaintext)
            return True
        except ValueError:
            return False

    # Attack: recover plaintext of the first ciphertext block
    print("  Padding Oracle Attack:")
    print(f"  Target: {len(ciphertext_blocks)} ciphertext blocks")

    recovered_plaintext = bytearray()

    # Attack each block
    for block_idx in range(len(ciphertext_blocks)):
        # The "previous" block (IV for first block, C_{i-1} for others)
        prev_block = iv if block_idx == 0 else ciphertext_blocks[block_idx - 1]
        target = ciphertext_blocks[block_idx]

        # Recover the intermediate value D_K(C_i)
        intermediate = bytearray(16)

        for byte_pos in range(15, -1, -1):
            pad_value = 16 - byte_pos
            modified = bytearray(prev_block)

            # Set already-recovered bytes to produce correct padding
            for j in range(byte_pos + 1, 16):
                modified[j] = intermediate[j] ^ pad_value

            # Try all 256 values for the target byte
            found = False
            for guess in range(256):
                modified[byte_pos] = guess
                if padding_oracle(bytes(modified), target):
                    # Avoid false positive when pad_value == 1
                    if byte_pos < 15 or pad_value > 1:
                        intermediate[byte_pos] = guess ^ pad_value
                        found = True
                        break
                    else:
                        # Verify by changing another byte
                        test = bytearray(modified)
                        test[byte_pos - 1] ^= 1
                        if padding_oracle(bytes(test), target):
                            intermediate[byte_pos] = guess ^ pad_value
                            found = True
                            break

            if not found:
                # Fallback: try all again
                for guess in range(256):
                    modified[byte_pos] = guess
                    if padding_oracle(bytes(modified), target):
                        intermediate[byte_pos] = guess ^ pad_value
                        break

        # Recover plaintext: P_i = D_K(C_i) XOR prev_block
        block_pt = bytes(i ^ p for i, p in zip(intermediate, prev_block))
        recovered_plaintext.extend(block_pt)

    # Remove padding from recovered plaintext
    try:
        recovered = pkcs7_unpad(bytes(recovered_plaintext))
    except ValueError:
        recovered = bytes(recovered_plaintext)

    print(f"  Recovered plaintext: {recovered}")
    print(f"  Original plaintext:  {secret}")
    print(f"  Match: {recovered == secret}")
    print(f"  Total oracle queries: {oracle_queries}")
    print(f"  Expected (approx): {len(ciphertext_blocks)} * 16 * 128 = {len(ciphertext_blocks) * 16 * 128}")
    print(f"  Note: actual queries vary; worst case is 256 per byte = {len(ciphertext_blocks) * 16 * 256}")


def exercise_5():
    """Exercise 5: GCM Tag Forgery Under Nonce Reuse (Challenging)

    Explain how nonce reuse in GCM enables tag forgery.
    """
    print("  GCM Nonce Reuse Attack -- Theoretical Explanation")
    print("  " + "=" * 55)

    print("""
  GCM computes the authentication tag as:
    T = GHASH(H, A, C) XOR E_K(Nonce || 0^32)

  where H = E_K(0^128) is the hash key, and GHASH is polynomial evaluation
  over GF(2^128):
    GHASH = (... ((A_1 * H) XOR A_2) * H XOR ... XOR C_1) * H XOR ... XOR len_block) * H

  Step 1: How nonce reuse reveals H
  ---------------------------------
  If the same nonce is used for two different (A, C) pairs:
    T1 = GHASH(H, A1, C1) XOR E_K(Nonce || 0^32)
    T2 = GHASH(H, A2, C2) XOR E_K(Nonce || 0^32)

  XOR them: T1 XOR T2 = GHASH(H, A1, C1) XOR GHASH(H, A2, C2)

  This gives a polynomial equation in H over GF(2^128).
  Since GHASH is a polynomial of degree (len_A + len_C + 1), the attacker
  can find the roots of this polynomial to recover H.
  For degree-d polynomials over GF(2^128), there are at most d roots.

  Step 2: How knowledge of H enables forgery
  -------------------------------------------
  Once the attacker knows H, they can compute GHASH for any (A*, C*) pair.
  They also know E_K(Nonce || 0^32) = T1 XOR GHASH(H, A1, C1).
  Therefore: T* = GHASH(H, A*, C*) XOR E_K(Nonce || 0^32)
  This is a valid tag for the forged message!

  Step 3: Why AES-GCM-SIV degrades gracefully
  ---------------------------------------------
  AES-GCM-SIV derives the nonce from both the key and the plaintext:
    SIV = PRF(key, nonce, plaintext)

  Under nonce reuse:
  - If the same plaintext is encrypted twice with the same nonce,
    the SIV (and thus the ciphertext) will be identical -- no new
    information is leaked, only the fact that the same message was
    sent twice (loss of IND-CPA, not authenticity).
  - If different plaintexts are encrypted with the same nonce,
    the SIVs differ, so different keystreams are used.
    The attacker cannot XOR away the keystream.

  Result: nonce misuse in SIV only leaks whether identical messages
  were encrypted, rather than enabling complete forgery.
  """)


if __name__ == "__main__":
    print("=== Exercise 1: ECB Pattern Detection ===")
    exercise_1()

    print("\n=== Exercise 2: CTR Mode Implementation ===")
    exercise_2()

    print("\n=== Exercise 3: Nonce Reuse Attack ===")
    exercise_3()

    print("\n=== Exercise 4: Padding Oracle Simulator ===")
    exercise_4()

    print("\n=== Exercise 5: GCM Tag Forgery Under Nonce Reuse ===")
    exercise_5()

    print("\nAll exercises completed!")

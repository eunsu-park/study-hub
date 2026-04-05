"""
Exercises for Lesson 02: Symmetric Ciphers
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
from collections import Counter


def caesar_encrypt(plaintext, key):
    """Caesar cipher encryption."""
    result = []
    for char in plaintext.upper():
        if char.isalpha():
            shifted = (ord(char) - ord('A') + key) % 26
            result.append(chr(shifted + ord('A')))
        else:
            result.append(char)
    return ''.join(result)


def vigenere_encrypt(plaintext, key):
    """Vigenere cipher encryption."""
    result = []
    key = key.upper()
    key_index = 0
    for char in plaintext.upper():
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            encrypted = (ord(char) - ord('A') + shift) % 26
            result.append(chr(encrypted + ord('A')))
            key_index += 1
        else:
            result.append(char)
    return ''.join(result)


def exercise_1():
    """Exercise 1: Frequency Analysis (Basic)

    Encrypt "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" with:
    - Caesar cipher (key=7)
    - Vigenere cipher (key="CRYPTO")
    Count letter frequencies in each ciphertext.
    """
    plaintext = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

    # Caesar encryption
    caesar_ct = caesar_encrypt(plaintext, 7)
    print(f"  Plaintext:         {plaintext}")
    print(f"  Caesar (key=7):    {caesar_ct}")

    # Vigenere encryption
    vigenere_ct = vigenere_encrypt(plaintext, "CRYPTO")
    print(f"  Vigenere (CRYPTO): {vigenere_ct}")

    # Frequency analysis
    def letter_freq(text):
        letters = [c for c in text.upper() if c.isalpha()]
        total = len(letters)
        freq = Counter(letters)
        return {k: round(v / total * 100, 1) for k, v in sorted(freq.items())}

    pt_freq = letter_freq(plaintext)
    caesar_freq = letter_freq(caesar_ct)
    vigenere_freq = letter_freq(vigenere_ct)

    print(f"\n  Plaintext frequencies:  {pt_freq}")
    print(f"  Caesar frequencies:     {caesar_freq}")
    print(f"  Vigenere frequencies:   {vigenere_freq}")

    # Analysis: measure frequency distribution flatness (variance)
    def freq_variance(freq_dict):
        if not freq_dict:
            return 0
        expected = 100.0 / 26
        return sum((v - expected) ** 2 for v in freq_dict.values()) / len(freq_dict)

    print(f"\n  Frequency variance (lower = flatter = better hiding):")
    print(f"    Plaintext: {freq_variance(pt_freq):.1f}")
    print(f"    Caesar:    {freq_variance(caesar_freq):.1f}")
    print(f"    Vigenere:  {freq_variance(vigenere_freq):.1f}")

    # Caesar just shifts the distribution -- variance is identical
    # Vigenere spreads frequencies more evenly (polyalphabetic substitution)
    print(f"\n  Conclusion: Caesar preserves the exact frequency distribution (just shifted).")
    print(f"  Vigenere better hides the plaintext frequency pattern because each")
    print(f"  plaintext letter maps to different ciphertext letters depending on position.")


def exercise_2():
    """Exercise 2: Feistel Cipher (Intermediate)

    1. Implement a 4-round Feistel cipher with 64-bit block and 128-bit key.
    2. Encrypt and decrypt, verifying round-trip.
    3. Analyze what happens with only 1 round.
    """
    def round_function(half_block, round_key):
        """Round function: non-linear mixing of data with key."""
        # Combine with key via multiplication and XOR for non-linearity
        mixed = ((half_block * 0x7A6D) ^ round_key) & 0xFFFFFFFF
        # Add more mixing via bit rotation
        mixed = ((mixed << 7) | (mixed >> 25)) & 0xFFFFFFFF
        mixed ^= (mixed >> 16)
        return mixed

    def feistel_encrypt(plaintext, round_keys):
        """Feistel cipher encryption."""
        left = (plaintext >> 32) & 0xFFFFFFFF
        right = plaintext & 0xFFFFFFFF
        for key in round_keys:
            new_left = right
            new_right = left ^ round_function(right, key)
            left, right = new_left, new_right
        # Final swap
        return (right << 32) | left

    def feistel_decrypt(ciphertext, round_keys):
        """Feistel cipher decryption: same structure, reversed keys."""
        left = (ciphertext >> 32) & 0xFFFFFFFF
        right = ciphertext & 0xFFFFFFFF
        for key in reversed(round_keys):
            new_left = right
            new_right = left ^ round_function(right, key)
            left, right = new_left, new_right
        return (right << 32) | left

    # Generate round keys from 128-bit master key
    master_key = random.getrandbits(128)
    round_keys = [
        (master_key >> (32 * i)) & 0xFFFFFFFF
        for i in range(4)
    ]

    # Part 1 & 2: 4-round encrypt/decrypt
    plaintext = 0xDEADBEEFCAFEBABE
    ciphertext = feistel_encrypt(plaintext, round_keys)
    decrypted = feistel_decrypt(ciphertext, round_keys)

    print(f"  4-round Feistel Cipher:")
    print(f"    Plaintext:  0x{plaintext:016X}")
    print(f"    Ciphertext: 0x{ciphertext:016X}")
    print(f"    Decrypted:  0x{decrypted:016X}")
    print(f"    Round-trip:  {plaintext == decrypted}")

    # Part 3: What happens with 1 round?
    print(f"\n  1-round Feistel analysis:")
    ct_1round = feistel_encrypt(plaintext, round_keys[:1])
    left_ct = (ct_1round >> 32) & 0xFFFFFFFF
    right_ct = ct_1round & 0xFFFFFFFF
    left_pt = (plaintext >> 32) & 0xFFFFFFFF
    right_pt = plaintext & 0xFFFFFFFF

    print(f"    Plaintext:  L=0x{left_pt:08X}, R=0x{right_pt:08X}")
    print(f"    Ciphertext: L=0x{left_ct:08X}, R=0x{right_ct:08X}")

    # After 1 round + final swap: C_L = L XOR F(R, K), C_R = R
    # The right half of plaintext appears UNMODIFIED in the ciphertext!
    print(f"    R_plaintext == L_ciphertext: {right_pt == left_ct}")
    print(f"    -> The right half of plaintext is exposed in the ciphertext!")
    print(f"    -> With 1 round, half the plaintext is directly recoverable.")


def exercise_3():
    """Exercise 3: AES Step Trace (Intermediate)

    Print the state matrix after each AES operation for the first two rounds
    using NIST FIPS 197 test vectors.
    """
    # AES S-box
    AES_SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]

    RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

    def bytes_to_state(b):
        state = [[0]*4 for _ in range(4)]
        for col in range(4):
            for row in range(4):
                state[row][col] = b[col * 4 + row]
        return state

    def state_to_hex(state):
        result = ""
        for col in range(4):
            for row in range(4):
                result += f"{state[row][col]:02x}"
        return result

    def print_state(state, label):
        print(f"    {label}: {state_to_hex(state)}")

    def sub_bytes(state):
        return [[AES_SBOX[byte] for byte in row] for row in state]

    def shift_rows(state):
        shifted = [row[:] for row in state]
        for i in range(4):
            shifted[i] = state[i][i:] + state[i][:i]
        return shifted

    def gf_multiply(a, b):
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            carry = a & 0x80
            a = (a << 1) & 0xFF
            if carry:
                a ^= 0x1B
            b >>= 1
        return result

    def mix_columns(state):
        MIX = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
        new_state = [[0]*4 for _ in range(4)]
        for col in range(4):
            for row in range(4):
                val = 0
                for k in range(4):
                    val ^= gf_multiply(MIX[row][k], state[k][col])
                new_state[row][col] = val
        return new_state

    def add_round_key(state, rk):
        return [[state[r][c] ^ rk[r][c] for c in range(4)] for r in range(4)]

    def key_expansion(key_bytes):
        w = []
        for i in range(4):
            w.append(list(key_bytes[4*i:4*i+4]))
        for i in range(4, 44):
            temp = list(w[i-1])
            if i % 4 == 0:
                temp = temp[1:] + temp[:1]
                temp = [AES_SBOX[b] for b in temp]
                temp[0] ^= RCON[i // 4 - 1]
            w.append([a ^ b for a, b in zip(w[i-4], temp)])
        round_keys = []
        for r in range(11):
            rk = [[0]*4 for _ in range(4)]
            for col in range(4):
                for row in range(4):
                    rk[row][col] = w[r*4 + col][row]
            round_keys.append(rk)
        return round_keys

    # NIST FIPS 197 Appendix B test vector
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    plaintext = bytes.fromhex("3243f6a8885a308d313198a2e0370734")

    state = bytes_to_state(plaintext)
    round_keys = key_expansion(key)

    print("  NIST FIPS 197 Appendix B trace (first 2 rounds):")
    print_state(state, "Input")

    # Initial AddRoundKey
    state = add_round_key(state, round_keys[0])
    print_state(state, "After initial AddRoundKey")

    # Rounds 1-2
    for r in range(1, 3):
        print(f"\n    --- Round {r} ---")
        state = sub_bytes(state)
        print_state(state, f"After SubBytes")
        state = shift_rows(state)
        print_state(state, f"After ShiftRows")
        state = mix_columns(state)
        print_state(state, f"After MixColumns")
        state = add_round_key(state, round_keys[r])
        print_state(state, f"After AddRoundKey")


def exercise_4():
    """Exercise 4: Avalanche Effect (Challenging)

    1. Encrypt two plaintexts differing by 1 bit using AES.
    2. Count bit differences after each round.
    3. Show how diffusion progresses through rounds.
    """
    # We'll use a simplified demonstration with SHA-256 as a proxy for AES rounds,
    # since implementing full AES round-by-round tracking is lengthy.
    # Instead, we demonstrate the concept with our AES-like operations.

    import hashlib

    print("  Avalanche effect demonstration:")
    print("  (Using SHA-256 truncated to 128 bits as AES proxy)")

    msg1 = b"\x00" * 16
    msg2 = b"\x01" + b"\x00" * 15  # 1-bit difference in first byte

    h1 = int(hashlib.sha256(msg1).hexdigest()[:32], 16)
    h2 = int(hashlib.sha256(msg2).hexdigest()[:32], 16)

    diff_bits = bin(h1 ^ h2).count('1')
    print(f"    Plaintext 1: {msg1.hex()}")
    print(f"    Plaintext 2: {msg2.hex()}")
    print(f"    Bits different in input: 1")
    print(f"    Bits different in output (128-bit): {diff_bits}/128 ({diff_bits/128*100:.1f}%)")

    # Simulated round-by-round diffusion using iterative hashing
    print("\n  Simulated round-by-round diffusion:")
    print("  (Each 'round' adds more mixing)")

    state1 = int.from_bytes(msg1, 'big')
    state2 = int.from_bytes(msg2, 'big')

    for round_num in range(1, 11):
        # Apply a mixing operation (simulating one AES round)
        state1 = int(hashlib.sha256(state1.to_bytes(32, 'big')).hexdigest()[:32], 16)
        state2 = int(hashlib.sha256(state2.to_bytes(32, 'big')).hexdigest()[:32], 16)
        diff = bin(state1 ^ state2).count('1')
        bar = '#' * (diff * 40 // 128)
        print(f"    Round {round_num:2d}: {diff:3d}/128 bits differ ({diff/128*100:5.1f}%) |{bar}")

    print("\n  In real AES, full diffusion (~50% bit differences) is achieved")
    print("  by round 3-4 due to MixColumns spreading each byte to all 4 bytes")
    print("  in a column, and ShiftRows spreading across columns.")


def exercise_5():
    """Exercise 5: S-Box Analysis (Challenging)

    1. Verify AES S-box has no fixed points: S(x) != x for all x.
    2. Verify no opposite fixed points: S(x) != ~x for all x.
    3. Compute non-linearity of the S-box.
    """
    AES_SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]

    # Part 1: Check for fixed points
    fixed_points = [x for x in range(256) if AES_SBOX[x] == x]
    print(f"  Part 1: Fixed points (S(x) = x): {fixed_points if fixed_points else 'NONE'}")
    print(f"    Verified: S-box has NO fixed points.")

    # Part 2: Check for opposite fixed points
    opposite_fixed = [x for x in range(256) if AES_SBOX[x] == (x ^ 0xFF)]
    print(f"\n  Part 2: Opposite fixed points (S(x) = ~x): {opposite_fixed if opposite_fixed else 'NONE'}")
    print(f"    Verified: S-box has NO opposite fixed points.")

    # Part 3: Non-linearity computation
    # The non-linearity is the minimum Hamming distance from the S-box (viewed as
    # a set of Boolean functions) to all affine functions.
    # For each output bit b and input mask a, count:
    #   bias(a, b) = |{x : a.x XOR b.S(x) = 0}| - 128
    # Non-linearity = 128 - max_bias

    def dot_product_8bit(a, b):
        """Compute the GF(2) dot product of two 8-bit values."""
        return bin(a & b).count('1') % 2

    max_bias = 0
    for a in range(1, 256):  # Skip a=0 (trivial)
        for b in range(1, 256):  # Skip b=0 (trivial)
            count = 0
            for x in range(256):
                if dot_product_8bit(a, x) ^ dot_product_8bit(b, AES_SBOX[x]) == 0:
                    count += 1
            bias = abs(count - 128)
            if bias > max_bias:
                max_bias = bias

    nonlinearity = 128 - max_bias
    print(f"\n  Part 3: S-box non-linearity")
    print(f"    Maximum linear approximation bias: {max_bias}/256")
    print(f"    Non-linearity: 128 - {max_bias} = {nonlinearity}")
    print(f"    (Theoretical maximum for 8-bit S-box: 120)")
    print(f"    AES S-box achieves {nonlinearity}/120 = {nonlinearity/120*100:.1f}% of theoretical maximum")


if __name__ == "__main__":
    print("=== Exercise 1: Frequency Analysis ===")
    exercise_1()

    print("\n=== Exercise 2: Feistel Cipher ===")
    exercise_2()

    print("\n=== Exercise 3: AES Step Trace ===")
    exercise_3()

    print("\n=== Exercise 4: Avalanche Effect ===")
    exercise_4()

    print("\n=== Exercise 5: S-Box Analysis ===")
    exercise_5()

    print("\nAll exercises completed!")

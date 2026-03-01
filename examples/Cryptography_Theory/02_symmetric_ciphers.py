"""
Symmetric Ciphers: Feistel Networks and AES Internals
=====================================================
Demonstrates the Feistel cipher structure (used in DES, Blowfish, Camellia)
and the core transformations inside AES (SubBytes, ShiftRows, MixColumns).
"""

from __future__ import annotations
import struct
import hashlib


# ---------------------------------------------------------------------------
# Feistel Cipher
# ---------------------------------------------------------------------------

# Why: The Feistel structure is elegant because only the round function F
# needs to be designed — decryption uses the SAME code with reversed key
# order. This halves the implementation effort and proof burden.


def _feistel_round_function(half: bytes, round_key: bytes) -> bytes:
    """Simple round function: hash(half || round_key) truncated to block half."""
    # Why: In a real Feistel cipher (e.g., DES), F involves expansion,
    # S-box substitution, and permutation. We use SHA-256 as a stand-in
    # to demonstrate the structure without implementing DES internals.
    h = hashlib.sha256(half + round_key).digest()
    return h[: len(half)]


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    return bytes(x ^ y for x, y in zip(a, b))


# Why: We derive round keys from a master key using a hash. Real ciphers
# use key schedules (e.g., DES has a permuted-choice schedule, AES uses
# Rijndael's key expansion). The principle is the same: one master key
# produces independent-looking subkeys for each round.
def _derive_round_keys(key: bytes, rounds: int) -> list[bytes]:
    """Derive `rounds` subkeys from the master key."""
    keys = []
    for i in range(rounds):
        h = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        keys.append(h[:4])  # 4 bytes per round key (matches half-block)
    return keys


class FeistelCipher:
    """A simple 4-round Feistel cipher with 8-byte blocks.

    Block layout: [Left (4 bytes) | Right (4 bytes)]
    """

    BLOCK_SIZE = 8
    ROUNDS = 4

    def __init__(self, key: bytes) -> None:
        self.round_keys = _derive_round_keys(key, self.ROUNDS)

    def encrypt_block(self, block: bytes) -> bytes:
        """Encrypt a single 8-byte block."""
        assert len(block) == self.BLOCK_SIZE
        left, right = block[:4], block[4:]

        # Why: Each Feistel round swaps halves and XORs with F(right, key).
        # After all rounds, the final swap is undone so that decryption
        # works by simply reversing the key order.
        for i in range(self.ROUNDS):
            f_out = _feistel_round_function(right, self.round_keys[i])
            left, right = right, _xor_bytes(left, f_out)

        # Why: Final swap so that decryption mirrors encryption structure.
        return right + left

    def decrypt_block(self, block: bytes) -> bytes:
        """Decrypt a single 8-byte block (reverse key order)."""
        assert len(block) == self.BLOCK_SIZE
        left, right = block[:4], block[4:]

        for i in range(self.ROUNDS - 1, -1, -1):
            f_out = _feistel_round_function(right, self.round_keys[i])
            left, right = right, _xor_bytes(left, f_out)

        return right + left


# ---------------------------------------------------------------------------
# AES Internals Visualization
# ---------------------------------------------------------------------------

# Why: AES uses a substitution-permutation network (SPN), not Feistel.
# SubBytes provides non-linearity (confusion), ShiftRows and MixColumns
# provide diffusion. Together they resist both linear and differential
# cryptanalysis — the two most powerful attacks on block ciphers.

# The full AES S-box (256 entries). Each byte is replaced by its entry.
# Why: The S-box is constructed as the multiplicative inverse in GF(2^8)
# followed by an affine transformation. This mathematical structure
# provides maximum non-linearity and resistance to algebraic attacks.
AES_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
    0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
    0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
    0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
    0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
    0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
    0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
    0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
    0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
    0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
    0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
    0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
    0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
    0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
    0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
    0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
    0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]


def sub_bytes(state: list[list[int]]) -> list[list[int]]:
    """AES SubBytes: replace each byte using the S-box."""
    return [[AES_SBOX[state[r][c]] for c in range(4)] for r in range(4)]


# Why: ShiftRows creates inter-column diffusion. Without it, each column
# would be processed independently and an attacker could break AES one
# column at a time (reducing a 128-bit problem to four 32-bit problems).
def shift_rows(state: list[list[int]]) -> list[list[int]]:
    """AES ShiftRows: cyclically shift row i left by i positions."""
    result = [row[:] for row in state]
    for i in range(4):
        result[i] = state[i][i:] + state[i][:i]
    return result


def _gf_mult(a: int, b: int) -> int:
    """Multiply two bytes in GF(2^8) with irreducible polynomial x^8+x^4+x^3+x+1."""
    # Why: AES arithmetic lives in GF(2^8). The irreducible polynomial
    # 0x11B was chosen by Rijndael's designers because it yields an S-box
    # with optimal non-linearity properties.
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi_bit = a & 0x80
        a = (a << 1) & 0xFF
        if hi_bit:
            a ^= 0x1B  # reduce by x^8+x^4+x^3+x+1
        b >>= 1
    return p


# Why: MixColumns provides diffusion within each column. Combined with
# ShiftRows, it ensures that after 2 rounds, every output byte depends
# on ALL 16 input bytes — this is called "full diffusion."
def mix_columns(state: list[list[int]]) -> list[list[int]]:
    """AES MixColumns: multiply each column by a fixed polynomial in GF(2^8)."""
    result = [[0] * 4 for _ in range(4)]
    # Fixed matrix for MixColumns
    matrix = [
        [2, 3, 1, 1],
        [1, 2, 3, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2],
    ]
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        for r in range(4):
            val = 0
            for k in range(4):
                val ^= _gf_mult(matrix[r][k], col[k])
            result[r][c] = val
    return result


def _format_state(state: list[list[int]], label: str) -> str:
    """Pretty-print a 4x4 AES state matrix."""
    lines = [f"  {label}:"]
    for row in state:
        lines.append("    [" + " ".join(f"{b:02X}" for b in row) + "]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Avalanche Effect Demonstration
# ---------------------------------------------------------------------------

# Why: A good cipher exhibits the avalanche effect — flipping one input bit
# changes ~50% of output bits. This makes the cipher resistant to
# differential cryptanalysis, where attackers study how input differences
# propagate through the cipher.
def avalanche_demo(cipher: FeistelCipher, block: bytes) -> None:
    """Show how a 1-bit change in plaintext changes the ciphertext."""
    ct1 = cipher.encrypt_block(block)

    # Flip one bit in the first byte
    flipped = bytes([block[0] ^ 0x01]) + block[1:]
    ct2 = cipher.encrypt_block(flipped)

    diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(ct1, ct2))
    total_bits = len(ct1) * 8

    print(f"  Original plaintext:  {block.hex()}")
    print(f"  Flipped plaintext:   {flipped.hex()}  (1 bit changed)")
    print(f"  Ciphertext 1:        {ct1.hex()}")
    print(f"  Ciphertext 2:        {ct2.hex()}")
    print(f"  Bits changed: {diff_bits}/{total_bits} ({diff_bits/total_bits*100:.1f}%)")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Symmetric Ciphers: Feistel & AES Internals")
    print("=" * 65)

    # --- Feistel Cipher ---
    print("\n[Feistel Cipher - 4 rounds, 8-byte blocks]")
    key = b"my_secret_key_16"
    cipher = FeistelCipher(key)

    plaintext = b"ABCDEFGH"  # 8 bytes
    ciphertext = cipher.encrypt_block(plaintext)
    decrypted = cipher.decrypt_block(ciphertext)

    print(f"  Key:        {key}")
    print(f"  Plaintext:  {plaintext}  ({plaintext.hex()})")
    print(f"  Ciphertext: {ciphertext.hex()}")
    print(f"  Decrypted:  {decrypted}  ({decrypted.hex()})")
    print(f"  Match: {plaintext == decrypted}")

    # --- Avalanche Effect ---
    print("\n[Avalanche Effect]")
    avalanche_demo(cipher, plaintext)

    # --- AES SubBytes ---
    print("\n[AES SubBytes]")
    state = [
        [0x32, 0x88, 0x31, 0xE0],
        [0x43, 0x5A, 0x31, 0x37],
        [0xF6, 0x30, 0x98, 0x07],
        [0xA8, 0x8D, 0xA2, 0x34],
    ]
    print(_format_state(state, "Before SubBytes"))
    after_sub = sub_bytes(state)
    print(_format_state(after_sub, "After SubBytes"))

    # --- AES ShiftRows ---
    print("\n[AES ShiftRows]")
    state2 = [
        [0x01, 0x02, 0x03, 0x04],
        [0x05, 0x06, 0x07, 0x08],
        [0x09, 0x0A, 0x0B, 0x0C],
        [0x0D, 0x0E, 0x0F, 0x10],
    ]
    print(_format_state(state2, "Before ShiftRows"))
    after_shift = shift_rows(state2)
    print(_format_state(after_shift, "After ShiftRows"))
    print("  Row 0: no shift, Row 1: <<1, Row 2: <<2, Row 3: <<3")

    # --- AES MixColumns ---
    print("\n[AES MixColumns]")
    state3 = [
        [0xDB, 0x01, 0x01, 0x01],
        [0x13, 0x01, 0x01, 0x01],
        [0x53, 0x01, 0x01, 0x01],
        [0x45, 0x01, 0x01, 0x01],
    ]
    print(_format_state(state3, "Before MixColumns"))
    after_mix = mix_columns(state3)
    print(_format_state(after_mix, "After MixColumns"))

    print(f"\n{'=' * 65}")

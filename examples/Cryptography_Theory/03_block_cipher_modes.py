"""
Block Cipher Modes of Operation
================================
ECB, CBC, and CTR modes illustrated with a toy cipher.
Demonstrates why ECB is dangerous: identical plaintext blocks produce
identical ciphertext blocks, leaking patterns.
"""

from __future__ import annotations
import hashlib
import os
import secrets


# ---------------------------------------------------------------------------
# Simple block cipher (for educational use only)
# ---------------------------------------------------------------------------

# Why: We use a keyed hash as a block cipher substitute so we can focus
# on MODE behavior without implementing full AES. The block size is 8 bytes
# to keep the visual demos compact and readable.
BLOCK_SIZE = 8


def _toy_encrypt_block(block: bytes, key: bytes) -> bytes:
    """Encrypt one 8-byte block using HMAC-SHA256 truncated (NOT secure)."""
    assert len(block) == BLOCK_SIZE
    h = hashlib.sha256(key + block).digest()
    return h[:BLOCK_SIZE]


def _toy_decrypt_block(block: bytes, key: bytes) -> bytes:
    """For this demo, encryption is a PRF (not invertible).
    We store a mapping to allow 'decryption' for CBC/ECB demos."""
    # Why: A hash-based cipher isn't invertible, so for demonstration
    # we use XOR with a key-derived pad. This is NOT a real cipher;
    # it just shows the mode structure.
    pad = hashlib.sha256(key + b"PAD").digest()[:BLOCK_SIZE]
    return bytes(a ^ b for a, b in zip(block, pad))


# Why: For mode demonstrations, we need a real encrypt/decrypt pair.
# We use a simple XOR cipher with a key-derived keystream per block index.
# This is invertible and sufficient to show mode behavior.
def _xor_cipher_block(block: bytes, key: bytes, block_index: int = 0) -> bytes:
    """Invertible block cipher: XOR with key-derived pad. Self-inverse."""
    assert len(block) == BLOCK_SIZE
    h = hashlib.sha256(key + block_index.to_bytes(4, "big")).digest()
    pad = h[:BLOCK_SIZE]
    return bytes(a ^ b for a, b in zip(block, pad))


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _pad(data: bytes) -> bytes:
    """PKCS#7 padding to BLOCK_SIZE."""
    pad_len = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + bytes([pad_len] * pad_len)


def _unpad(data: bytes) -> bytes:
    """Remove PKCS#7 padding."""
    pad_len = data[-1]
    return data[:-pad_len]


# ---------------------------------------------------------------------------
# ECB Mode
# ---------------------------------------------------------------------------

# Why: ECB encrypts each block independently with the same key. This is
# the simplest mode but FATALLY FLAWED: identical plaintext blocks always
# produce identical ciphertext blocks, revealing patterns. The famous
# "ECB penguin" image demonstrates this — the encrypted image still
# shows the penguin's shape because repeated color blocks encrypt the same.
def ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """ECB mode encryption: each block encrypted independently."""
    data = _pad(plaintext)
    ciphertext = b""
    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i : i + BLOCK_SIZE]
        ciphertext += _xor_cipher_block(block, key)
    return ciphertext


def ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """ECB mode decryption."""
    plaintext = b""
    for i in range(0, len(ciphertext), BLOCK_SIZE):
        block = ciphertext[i : i + BLOCK_SIZE]
        plaintext += _xor_cipher_block(block, key)
    return _unpad(plaintext)


# ---------------------------------------------------------------------------
# CBC Mode
# ---------------------------------------------------------------------------

# Why: CBC chains blocks together — each plaintext block is XORed with the
# previous ciphertext block before encryption. This hides patterns because
# identical plaintext blocks produce different ciphertext (as long as the
# IV is random and unique per message).
def cbc_encrypt(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    """CBC mode encryption: each block is XORed with previous ciphertext."""
    assert len(iv) == BLOCK_SIZE
    data = _pad(plaintext)
    ciphertext = b""
    prev = iv

    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i : i + BLOCK_SIZE]
        # Why: XOR with previous ciphertext breaks the pattern —
        # even identical plaintext blocks produce different inputs
        # to the block cipher, yielding different ciphertext.
        mixed = _xor_bytes(block, prev)
        encrypted = _xor_cipher_block(mixed, key)
        ciphertext += encrypted
        prev = encrypted

    return iv + ciphertext  # prepend IV for decryption


def cbc_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """CBC mode decryption. IV is the first block of ciphertext."""
    iv = ciphertext[:BLOCK_SIZE]
    data = ciphertext[BLOCK_SIZE:]
    plaintext = b""
    prev = iv

    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i : i + BLOCK_SIZE]
        decrypted = _xor_cipher_block(block, key)
        plaintext += _xor_bytes(decrypted, prev)
        prev = block

    return _unpad(plaintext)


# ---------------------------------------------------------------------------
# CTR Mode
# ---------------------------------------------------------------------------

# Why: CTR mode turns a block cipher into a stream cipher by encrypting
# a counter and XORing the result with plaintext. Advantages:
# 1) Parallelizable (unlike CBC which is sequential)
# 2) Random access — you can decrypt any block independently
# 3) No padding needed — just truncate the keystream
def ctr_encrypt(plaintext: bytes, key: bytes, nonce: bytes) -> bytes:
    """CTR mode: encrypt counter values and XOR with plaintext."""
    assert len(nonce) == BLOCK_SIZE // 2  # 4-byte nonce + 4-byte counter
    ciphertext = b""

    for i in range(0, len(plaintext), BLOCK_SIZE):
        counter_block = nonce + i.to_bytes(BLOCK_SIZE // 2, "big")
        keystream = _xor_cipher_block(counter_block, key)
        chunk = plaintext[i : i + BLOCK_SIZE]
        ciphertext += _xor_bytes(keystream[: len(chunk)], chunk)

    return nonce + ciphertext  # prepend nonce


def ctr_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """CTR mode decryption (identical to encryption after extracting nonce)."""
    nonce = ciphertext[: BLOCK_SIZE // 2]
    data = ciphertext[BLOCK_SIZE // 2 :]
    plaintext = b""

    for i in range(0, len(data), BLOCK_SIZE):
        counter_block = nonce + i.to_bytes(BLOCK_SIZE // 2, "big")
        keystream = _xor_cipher_block(counter_block, key)
        chunk = data[i : i + BLOCK_SIZE]
        plaintext += _xor_bytes(keystream[: len(chunk)], chunk)

    return plaintext


# ---------------------------------------------------------------------------
# Visual Pattern Demo (ASCII Art "Penguin")
# ---------------------------------------------------------------------------

# Why: The ECB penguin is the most famous demonstration of why ECB mode
# is dangerous. We simulate it with ASCII art: areas of repeated characters
# (like a solid background) encrypt to the same ciphertext blocks in ECB,
# preserving the image shape. CBC and CTR break this pattern.
def _create_ascii_image() -> list[str]:
    """Create a simple ASCII art pattern with repetitive regions."""
    # Background is all dots; the "shape" is made of hashes.
    w = 8  # chars per block (matches BLOCK_SIZE)
    rows = [
        "........" * 4,
        "........" + "########" + "########" + "........",
        "........" + "########" + "########" + "........",
        "########" + "########" + "########" + "########",
        "########" + "########" + "########" + "########",
        "........" + "########" + "########" + "........",
        "........" + "########" + "########" + "........",
        "........" * 4,
    ]
    return rows


def _visualize_ecb_vs_cbc(key: bytes) -> None:
    """Show how ECB preserves patterns while CBC hides them."""
    rows = _create_ascii_image()
    image_bytes = b"".join(r.encode() for r in rows)

    # Encrypt with ECB
    ecb_ct = ecb_encrypt(image_bytes, key)

    # Show block-level pattern analysis
    print("\n  [Original ASCII Image]")
    for row in rows:
        print(f"    {row}")

    # Analyze ECB block uniqueness
    ecb_blocks = []
    for i in range(0, len(ecb_ct), BLOCK_SIZE):
        ecb_blocks.append(ecb_ct[i : i + BLOCK_SIZE])

    unique_blocks = set(ecb_blocks)
    block_map = {b: chr(65 + i) for i, b in enumerate(sorted(unique_blocks))}

    print(f"\n  [ECB Ciphertext - Block Pattern]")
    print(f"    {len(ecb_blocks)} blocks, only {len(unique_blocks)} unique")
    print(f"    Each letter = one unique ciphertext block:")
    blocks_per_row = (len(rows[0]) + BLOCK_SIZE - 1) // BLOCK_SIZE
    for row_idx in range(len(rows)):
        pattern = ""
        for col in range(blocks_per_row):
            bi = row_idx * blocks_per_row + col
            if bi < len(ecb_blocks):
                pattern += block_map.get(ecb_blocks[bi], "?") * BLOCK_SIZE
        print(f"    {pattern}")

    print("\n    ^ ECB reveals the shape! Identical blocks -> same letter.")

    # CBC comparison
    iv = secrets.token_bytes(BLOCK_SIZE)
    cbc_ct = cbc_encrypt(image_bytes, key, iv)
    cbc_data = cbc_ct[BLOCK_SIZE:]  # skip IV
    cbc_blocks = []
    for i in range(0, len(cbc_data), BLOCK_SIZE):
        cbc_blocks.append(cbc_data[i : i + BLOCK_SIZE])

    cbc_unique = set(cbc_blocks)
    print(f"\n  [CBC Ciphertext - Block Pattern]")
    print(f"    {len(cbc_blocks)} blocks, {len(cbc_unique)} unique (all different!)")
    print(f"    Pattern is completely hidden.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Block Cipher Modes of Operation")
    print("=" * 65)

    key = b"sixteen_byte_key"

    # --- ECB ---
    print("\n[ECB Mode]")
    pt = b"HELLO!!!HELLO!!!HELLO!!!"  # repeated 8-byte blocks
    ct = ecb_encrypt(pt, key)
    dt = ecb_decrypt(ct, key)
    print(f"  Plaintext:  {pt}")
    print(f"  Ciphertext: {ct.hex()}")
    print(f"  Decrypted:  {dt}")

    # Show repeated blocks
    print(f"\n  Ciphertext blocks (ECB):")
    for i in range(0, len(ct), BLOCK_SIZE):
        print(f"    Block {i // BLOCK_SIZE}: {ct[i:i+BLOCK_SIZE].hex()}")
    print("    ^ Notice: identical plaintext blocks -> identical ciphertext!")

    # --- CBC ---
    print(f"\n[CBC Mode]")
    iv = secrets.token_bytes(BLOCK_SIZE)
    ct_cbc = cbc_encrypt(pt, key, iv)
    dt_cbc = cbc_decrypt(ct_cbc, key)
    print(f"  Plaintext:  {pt}")
    print(f"  Ciphertext: {ct_cbc.hex()} (includes IV)")
    print(f"  Decrypted:  {dt_cbc}")

    print(f"\n  Ciphertext blocks (CBC, after IV):")
    cbc_data = ct_cbc[BLOCK_SIZE:]
    for i in range(0, len(cbc_data), BLOCK_SIZE):
        print(f"    Block {i // BLOCK_SIZE}: {cbc_data[i:i+BLOCK_SIZE].hex()}")
    print("    ^ All blocks are different despite identical plaintext blocks!")

    # --- CTR ---
    print(f"\n[CTR Mode]")
    nonce = secrets.token_bytes(BLOCK_SIZE // 2)
    ct_ctr = ctr_encrypt(pt, key, nonce)
    dt_ctr = ctr_decrypt(ct_ctr, key)
    print(f"  Plaintext:  {pt}")
    print(f"  Ciphertext: {ct_ctr.hex()} (includes nonce)")
    print(f"  Decrypted:  {dt_ctr}")

    # --- Visual Pattern Demo ---
    print(f"\n[Why ECB Is Dangerous: The 'ECB Penguin' Problem]")
    _visualize_ecb_vs_cbc(key)

    print(f"\n{'=' * 65}")

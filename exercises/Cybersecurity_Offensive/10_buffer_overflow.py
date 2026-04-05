"""
Exercises for Lesson 10: Buffer Overflow
Topic: Cybersecurity_Offensive

Practice problems covering stack-based overflows, return address
overwrite, NOP sleds, and DEP/ASLR bypass concepts.
"""


# === Exercise 1: Overflow Offset Calculator ===
# Problem: Given a crash analysis where EIP/RIP is overwritten with a
# pattern, calculate the exact offset to the return address.
# Use a cyclic pattern approach.

def exercise_1():
    """
    Implement cyclic_pattern(length: int) -> str that generates a
    De Bruijn-like pattern (e.g., "Aa0Aa1Aa2...").

    Then implement find_offset(pattern: str, value: str) -> int
    that finds where 'value' appears in the pattern.

    Test: Generate 200-byte pattern. If crash shows EIP = "6Aa7",
    find the offset.
    """
    # TODO: Implement cyclic pattern generator and offset finder
    pass


# === Exercise 2: Shellcode Encoder ===
# Problem: Encode shellcode to avoid null bytes (common bad character).
# Implement XOR encoding with a single-byte key.

def exercise_2():
    """
    raw_shellcode = b"\\x48\\x31\\xc0\\x50\\x48\\x89\\xe7\\x48\\x31\\xf6\\x48\\x31\\xd2\\x48\\xc7\\xc0\\x3b\\x00\\x00\\x00"
    Note the null bytes at positions 17, 18, 19.

    Implement:
      encode_xor(shellcode: bytes, key: int) -> bytes
      find_safe_key(shellcode: bytes) -> int  # key that produces no nulls

    Return the encoded shellcode and the decoder stub logic.
    """
    # TODO: Implement XOR encoder that eliminates null bytes
    pass


# === Exercise 3: NOP Sled Builder ===
# Problem: Build a buffer overflow payload with proper structure:
# [NOP sled] + [shellcode] + [return address]

def exercise_3():
    """
    Parameters:
      buffer_size = 64
      shellcode = b"\\xCC" * 20  # placeholder (INT3 breakpoints)
      ret_addr = 0x7FFFFFFFE100   # target return address
      nop = b"\\x90"

    Build payload where:
      1. NOP sled fills from start up to shellcode
      2. Shellcode is placed after NOP sled
      3. Padding fills remaining space
      4. Return address overwrites saved RIP (at offset 72 from buffer start)

    Return the complete payload as bytes and its total length.
    """
    # TODO: Build buffer overflow payload
    pass


# === Exercise 4: Protection Detection ===
# Problem: Given binary properties, determine which exploit mitigations
# are active and what bypass techniques are needed.

def exercise_4():
    """
    binary_info = {
        "NX": True,       # Non-executable stack
        "ASLR": True,     # Address space layout randomization
        "Stack Canary": False,  # No stack canary
        "PIE": False,     # No position-independent executable
        "RELRO": "Partial",
    }
    For each protection that is enabled, suggest a bypass technique.
    For each disabled protection, explain the exploitation advantage.
    Return dict mapping each protection to {"status": str, "implication": str}
    """
    # TODO: Analyze protections and suggest bypasses
    pass


if __name__ == "__main__":
    print("=== Exercise 1: Overflow Offset Calculator ===")
    print(exercise_1())
    print("\n=== Exercise 2: Shellcode Encoder ===")
    print(exercise_2())
    print("\n=== Exercise 3: NOP Sled Builder ===")
    print(exercise_3())
    print("\n=== Exercise 4: Protection Detection ===")
    print(exercise_4())

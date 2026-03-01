"""
Number Systems and Conversions

Demonstrates:
- Binary, octal, hexadecimal conversions
- Two's complement representation
- Signed integer encoding/decoding
- BCD (Binary-Coded Decimal)

Theory:
- Computers represent all data in binary (base 2).
- Hex (base 16) is a compact notation for binary (4 bits per digit).
- Two's complement: standard encoding for signed integers.
  For N bits, range is [-2^(N-1), 2^(N-1)-1].
  Negation: invert all bits + add 1.

Adapted from Computer Architecture Lesson 02.
"""


def to_binary(n: int, bits: int = 8) -> str:
    """Convert integer to binary string (two's complement for negatives)."""
    if n < 0:
        # Two's complement: adding 2^bits maps a negative value into its
        # unsigned bit pattern.  This works because -k and 2^bits - k
        # share the same low-order `bits` binary digits.
        n = (1 << bits) + n
    return format(n, f"0{bits}b")


def from_binary_signed(s: str) -> int:
    """Interpret binary string as signed two's complement."""
    bits = len(s)
    value = int(s, 2)
    if s[0] == "1":  # negative
        # MSB=1 means the number is negative in two's complement.
        # Subtracting 2^bits converts the unsigned interpretation
        # back to the correct negative signed value.
        value -= (1 << bits)
    return value


def to_hex(n: int, bits: int = 8) -> str:
    """Convert integer to hexadecimal."""
    if n < 0:
        n = (1 << bits) + n
    hex_digits = bits // 4
    return format(n, f"0{hex_digits}X")


def to_octal(n: int, bits: int = 12) -> str:
    """Convert integer to octal."""
    if n < 0:
        n = (1 << bits) + n
    return format(n, "o")


# ── Number System Conversions ───────────────────────────────────────────

def demo_conversions():
    """Demonstrate base conversions."""
    print("=" * 60)
    print("NUMBER SYSTEM CONVERSIONS")
    print("=" * 60)

    values = [0, 1, 42, 127, 128, 255]
    print(f"\n  {'Decimal':>8}  {'Binary':>10}  {'Hex':>5}  {'Octal':>6}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*5}  {'-'*6}")
    for v in values:
        print(f"  {v:>8}  {to_binary(v):>10}  {to_hex(v):>5}  {to_octal(v, 12):>6}")

    # Show binary ↔ hex relationship
    print("\n  Binary ↔ Hex (4-bit groups):")
    hex_val = 0xDEAD
    binary = format(hex_val, "016b")
    groups = [binary[i:i+4] for i in range(0, 16, 4)]
    hex_digits = [format(int(g, 2), "X") for g in groups]
    print(f"    0x{hex_val:04X} = {' '.join(groups)}")
    print(f"           = {'    '.join(hex_digits)}")


# ── Two's Complement ────────────────────────────────────────────────────

def demo_twos_complement():
    """Demonstrate two's complement representation."""
    print("\n" + "=" * 60)
    print("TWO'S COMPLEMENT (8-bit)")
    print("=" * 60)

    print(f"\n  Range: [{-(1<<7)}, {(1<<7)-1}] = [-128, 127]")
    print(f"\n  {'Decimal':>8}  {'Binary':>10}  {'Hex':>5}  {'Unsigned':>9}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*5}  {'-'*9}")

    for v in [0, 1, 42, 127, -1, -42, -127, -128]:
        binary = to_binary(v, 8)
        unsigned = int(binary, 2)
        print(f"  {v:>8}  {binary:>10}  {to_hex(v, 8):>5}  {unsigned:>9}")

    # Show negation process
    print("\n  Negation of 42:")
    pos = to_binary(42, 8)
    inverted = "".join("1" if b == "0" else "0" for b in pos)
    result = to_binary(-42, 8)
    print(f"     42 = {pos}")
    print(f"    ~42 = {inverted}  (invert bits)")
    print(f"     +1 = {to_binary(1, 8)}")
    print(f"    -42 = {result}  (two's complement)")

    # Verify arithmetic
    print("\n  Arithmetic verification:")
    a, b = 42, -42
    sum_bits = to_binary(a + b, 8)
    print(f"    42 + (-42) = {sum_bits} = {from_binary_signed(sum_bits)}")

    a, b = 100, 50
    sum_bits = to_binary(a - b, 8)
    print(f"    100 - 50   = {sum_bits} = {from_binary_signed(sum_bits)}")


# ── Overflow Detection ──────────────────────────────────────────────────

def demo_overflow():
    """Demonstrate signed overflow."""
    print("\n" + "=" * 60)
    print("OVERFLOW DETECTION (8-bit)")
    print("=" * 60)

    test_cases = [
        (100, 27, "+"),   # no overflow
        (100, 50, "+"),   # overflow! (150 > 127)
        (-100, -30, "+"), # overflow! (-130 < -128)
        (100, -50, "+"),  # no overflow
    ]

    print(f"\n  {'A':>5}  {'B':>5}  {'Op':>3}  {'Result':>7}  {'8-bit':>10}  {'Decoded':>8}  {'Overflow':>9}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*9}")

    for a, b, op in test_cases:
        result = a + b if op == "+" else a - b
        # 8-bit truncation
        truncated = result & 0xFF
        decoded = from_binary_signed(to_binary(result, 8))
        # Overflow occurs when the true mathematical result exceeds the
        # representable range.  Comparing the decoded (truncated) value
        # to the full-precision result detects this without inspecting
        # carry/sign bits explicitly.
        overflow = decoded != result
        print(f"  {a:>5}  {b:>5}  {op:>3}  {result:>7}  {to_binary(result, 8):>10}  "
              f"{decoded:>8}  {'YES' if overflow else 'no':>9}")


# ── BCD (Binary-Coded Decimal) ──────────────────────────────────────────

def to_bcd(n: int) -> str:
    """Convert non-negative integer to BCD."""
    if n < 0:
        return "N/A"
    # Each decimal digit is encoded independently into 4 bits.
    # This wastes storage compared to pure binary but makes
    # decimal display trivial — hardware just reads 4-bit groups.
    return " ".join(format(int(d), "04b") for d in str(n))


def demo_bcd():
    """Demonstrate BCD encoding."""
    print("\n" + "=" * 60)
    print("BCD (Binary-Coded Decimal)")
    print("=" * 60)

    values = [0, 5, 42, 127, 999]
    print(f"\n  {'Decimal':>8}  {'BCD':>20}  {'Pure Binary':>12}")
    print(f"  {'-'*8}  {'-'*20}  {'-'*12}")
    for v in values:
        bcd = to_bcd(v)
        binary = format(v, "b")
        print(f"  {v:>8}  {bcd:>20}  {binary:>12}")

    print(f"\n  BCD uses 4 bits per decimal digit (0-9).")
    print(f"  Less space-efficient than binary, but easy decimal display.")
    print(f"  Example: 42 in BCD = 0100 0010 (8 bits)")
    print(f"           42 in binary = 101010 (6 bits)")


if __name__ == "__main__":
    demo_conversions()
    demo_twos_complement()
    demo_overflow()
    demo_bcd()

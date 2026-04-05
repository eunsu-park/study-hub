"""
IEEE 754 Floating-Point Representation

Demonstrates:
- IEEE 754 single-precision (32-bit) encoding/decoding
- Special values (±∞, NaN, denormalized numbers)
- Precision limitations and rounding errors
- Machine epsilon

Theory:
- IEEE 754 single: 1 sign + 8 exponent + 23 mantissa bits
- Value = (-1)^s × 1.mantissa × 2^(exponent - 127)
- Bias = 127 for single precision
- Special cases: exponent=0 (denorm/zero), exponent=255 (inf/NaN)

Adapted from Computer Architecture Lesson 03.
"""

import struct
import math


def float_to_ieee754(value: float) -> dict:
    """Decompose a float into IEEE 754 single-precision components."""
    # Pack as float, then reinterpret the same 4 bytes as an unsigned
    # int — this gives us the raw bit pattern without manual encoding.
    # Big-endian (">") ensures consistent bit ordering across platforms.
    packed = struct.pack(">f", value)
    bits = struct.unpack(">I", packed)[0]

    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    # Exponent=0 and exponent=255 are reserved for special values.
    # This avoids ambiguity: every normal float has exponent in [1,254],
    # so the implicit leading 1 is always present for normalized numbers.
    if exponent == 0:
        if mantissa == 0:
            category = "zero"
        else:
            category = "denormalized"
    elif exponent == 255:
        if mantissa == 0:
            category = "infinity"
        else:
            category = "NaN"
    else:
        category = "normalized"

    return {
        "value": value,
        "bits": format(bits, "032b"),
        "hex": format(bits, "08X"),
        "sign": sign,
        "exponent": exponent,
        "mantissa": mantissa,
        "category": category,
    }


def ieee754_to_float(sign: int, exponent: int, mantissa: int) -> float:
    """Reconstruct float from IEEE 754 components."""
    bits = (sign << 31) | (exponent << 23) | mantissa
    packed = struct.pack(">I", bits)
    return struct.unpack(">f", packed)[0]


def format_bits(bits_str: str) -> str:
    """Format 32-bit string with field separators."""
    return f"{bits_str[0]} {bits_str[1:9]} {bits_str[9:]}"


# ── IEEE 754 Encoding Demo ──────────────────────────────────────────────

def demo_encoding():
    """Show IEEE 754 encoding for various values."""
    print("=" * 60)
    print("IEEE 754 SINGLE-PRECISION ENCODING")
    print("=" * 60)

    values = [0.0, 1.0, -1.0, 0.5, 0.1, 3.14, 42.0, 1000.0]

    print(f"\n  {'Value':>10}  {'Sign':>4}  {'Exp':>5}  {'Mantissa':>25}  {'Hex':>10}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*5}  {'-'*25}  {'-'*10}")

    for v in values:
        info = float_to_ieee754(v)
        mantissa_bits = format(info["mantissa"], "023b")
        print(f"  {v:>10.4f}  {info['sign']:>4}  {info['exponent']:>5}  "
              f"{mantissa_bits:>25}  0x{info['hex']}")

    # Detailed breakdown for 3.14
    print("\n  Detailed breakdown for 3.14:")
    info = float_to_ieee754(3.14)
    print(f"    Bits: {format_bits(info['bits'])}")
    print(f"    Sign: {info['sign']} ({'−' if info['sign'] else '+'})")
    print(f"    Exponent: {info['exponent']} (biased), "
          f"{info['exponent'] - 127} (actual)")
    print(f"    Mantissa: 1.{format(info['mantissa'], '023b')} "
          f"(implicit leading 1)")
    actual_exp = info['exponent'] - 127
    mantissa_val = 1.0 + info['mantissa'] / (2**23)
    reconstructed = (-1)**info['sign'] * mantissa_val * (2**actual_exp)
    print(f"    Value: (-1)^{info['sign']} × {mantissa_val:.10f} × 2^{actual_exp}")
    print(f"         = {reconstructed:.10f}")


# ── Special Values ──────────────────────────────────────────────────────

def demo_special_values():
    """Show special IEEE 754 values."""
    print("\n" + "=" * 60)
    print("SPECIAL VALUES")
    print("=" * 60)

    specials = [
        (0.0, "+0"),
        (-0.0, "−0"),
        (float("inf"), "+∞"),
        (float("-inf"), "−∞"),
        (float("nan"), "NaN"),
    ]

    print(f"\n  {'Name':>6}  {'Bits (S E M)':>36}  {'Category':>14}")
    print(f"  {'-'*6}  {'-'*36}  {'-'*14}")

    for val, name in specials:
        info = float_to_ieee754(val)
        formatted = format_bits(info["bits"])
        print(f"  {name:>6}  {formatted:>36}  {info['category']:>14}")

    # Denormalized numbers
    print("\n  Denormalized numbers (subnormal):")
    print("  Exponent=0, mantissa≠0: value = (-1)^s × 0.mantissa × 2^(-126)")

    # Smallest denorm: mantissa=1 with exponent=0.  Denormalized numbers
    # fill the gap between zero and the smallest normalized float,
    # providing "gradual underflow" instead of an abrupt jump to zero.
    smallest_denorm = ieee754_to_float(0, 0, 1)
    info = float_to_ieee754(smallest_denorm)
    print(f"    Smallest denorm: {smallest_denorm:.2e}")
    print(f"    Bits: {format_bits(info['bits'])}")

    # Largest denorm
    largest_denorm = ieee754_to_float(0, 0, 0x7FFFFF)
    print(f"    Largest denorm:  {largest_denorm:.2e}")

    # Smallest normalized
    smallest_norm = ieee754_to_float(0, 1, 0)
    print(f"    Smallest norm:   {smallest_norm:.2e}")


# ── Precision Limitations ───────────────────────────────────────────────

def demo_precision():
    """Demonstrate floating-point precision issues."""
    print("\n" + "=" * 60)
    print("PRECISION LIMITATIONS")
    print("=" * 60)

    # Classic: 0.1 + 0.2 ≠ 0.3
    print(f"\n  Classic precision issue:")
    a, b = 0.1, 0.2
    result = a + b
    print(f"    0.1 + 0.2 = {result:.20f}")
    print(f"    0.3       = {0.3:.20f}")
    print(f"    Equal?      {result == 0.3}")
    print(f"    Close?      {math.isclose(result, 0.3)}")

    # Why 0.1 can't be exactly represented
    print(f"\n  0.1 in binary is repeating:")
    print(f"    0.1₁₀ = 0.0001100110011...₂ (repeating)")
    info = float_to_ieee754(0.1)
    print(f"    Stored: {format_bits(info['bits'])}")
    print(f"    Actual: {0.1:.20f}")

    # Catastrophic cancellation
    print(f"\n  Catastrophic cancellation:")
    x = 1e10
    print(f"    (1e10 + 1) - 1e10 = {(x + 1) - x}")
    print(f"    Expected: 1")
    print(f"    1e10 + 1e-10 - 1e10 = {(x + 1e-10) - x}")
    print(f"    Expected: 1e-10")

    # Machine epsilon = smallest e where 1.0 + e != 1.0.
    # It equals 2^(-mantissa_bits) because the mantissa has that many
    # significant fraction bits — any smaller increment is rounded away.
    eps32 = 2**-23  # single precision (23-bit mantissa)
    eps64 = 2**-52  # double precision (52-bit mantissa, Python default)
    print(f"\n  Machine epsilon:")
    print(f"    Single (32-bit): {eps32:.2e}")
    print(f"    Double (64-bit): {eps64:.2e}")
    print(f"    Python float eps: {float.__class__.__name__} uses 64-bit double")

    # Representation density
    print(f"\n  Representation density varies with magnitude:")
    for exp in [0, 10, 20, 30]:
        base = 2.0 ** exp
        info1 = float_to_ieee754(base)
        next_val = ieee754_to_float(0, info1["exponent"], info1["mantissa"] + 1)
        gap = next_val - base
        print(f"    Near 2^{exp:>2} ({base:.0e}): gap = {gap:.2e}")


if __name__ == "__main__":
    demo_encoding()
    demo_special_values()
    demo_precision()

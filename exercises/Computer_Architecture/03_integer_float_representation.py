"""
Exercises for Lesson 03: Integer and Float Representation
Topic: Computer_Architecture

Solutions to practice problems covering two's complement representation,
IEEE 754 single-precision floating-point, overflow detection, and
floating-point precision issues.
"""
import struct


def exercise_1():
    """
    Represent numbers in 8-bit two's complement:
    (a) +50, (b) -50, (c) -1, (d) -128
    """
    def to_twos_complement_8bit(value):
        """Convert signed integer to 8-bit two's complement binary string."""
        if value >= 0:
            return format(value, '08b')
        else:
            return format(value & 0xFF, '08b')

    values = [(50, "+50"), (-50, "-50"), (-1, "-1"), (-128, "-128")]

    print("8-bit two's complement representation:")
    for val, label in values:
        binary = to_twos_complement_8bit(val)
        if val < 0:
            pos = format(abs(val), '08b')
            inv = ''.join('1' if b == '0' else '0' for b in pos)
            print(f"  {label:>5s} : {binary}")
            print(f"         (|{abs(val)}| = {pos}, invert = {inv}, +1 = {binary})")
        else:
            print(f"  {label:>5s} : {binary}")


def exercise_2():
    """
    Convert 8-bit two's complement to decimal:
    (a) 01100100, (b) 11001110, (c) 10000000, (d) 11111111
    """
    def from_twos_complement_8bit(binary_str):
        """Convert 8-bit two's complement binary string to signed integer."""
        val = int(binary_str, 2)
        if binary_str[0] == '1':
            val -= 256
        return val

    values = ["01100100", "11001110", "10000000", "11111111"]

    print("8-bit two's complement to decimal:")
    for binary in values:
        decimal = from_twos_complement_8bit(binary)
        msb = binary[0]
        sign = "positive" if msb == '0' else "negative"
        print(f"  {binary} → MSB={msb} ({sign}) → {decimal}")
        if msb == '1':
            inv = ''.join('1' if b == '0' else '0' for b in binary)
            mag = int(inv, 2) + 1
            print(f"    Verification: invert={inv}, +1={format(mag, '08b')}={mag}, negate → {-mag}")


def exercise_3():
    """
    8-bit two's complement arithmetic:
    (a) 45+30, (b) 45-30, (c) -45-30, (d) -1+1
    """
    def twos_comp_8bit(val):
        return format(val & 0xFF, '08b')

    def interpret_8bit(binary_str):
        val = int(binary_str, 2)
        if val >= 128:
            val -= 256
        return val

    operations = [
        (45, 30, "+"),
        (45, -30, "+"),   # 45 - 30 = 45 + (-30)
        (-45, -30, "+"),  # -45 - 30 = -45 + (-30)
        (-1, 1, "+"),
    ]
    labels = ["45 + 30", "45 - 30", "-45 - 30", "-1 + 1"]

    print("8-bit two's complement arithmetic:")
    for (a, b, op), label in zip(operations, labels):
        bin_a = twos_comp_8bit(a)
        bin_b = twos_comp_8bit(b)
        raw = (a + b) if op == "+" else (a - b)
        result_bin = twos_comp_8bit(raw)
        result_dec = interpret_8bit(result_bin)

        print(f"\n  {label}:")
        print(f"    {bin_a}  ({a:>4d})")
        print(f"  + {bin_b}  ({b:>4d})")
        print(f"  ----------")
        print(f"    {result_bin}  ({result_dec:>4d})")


def exercise_4():
    """
    Convert decimal to IEEE 754 single precision:
    (a) 5.75, (b) -0.375, (c) 1.0
    """
    def float_to_ieee754(value):
        """Convert a float to IEEE 754 single-precision bit representation."""
        # Use struct to get the actual hardware representation
        packed = struct.pack('>f', value)
        bits = ''.join(format(byte, '08b') for byte in packed)
        sign = bits[0]
        exponent = bits[1:9]
        mantissa = bits[9:32]
        return sign, exponent, mantissa

    def explain_conversion(value):
        """Step-by-step IEEE 754 conversion explanation."""
        sign_bit = '0' if value >= 0 else '1'
        abs_val = abs(value)

        # Integer and fractional parts
        int_part = int(abs_val)
        frac_part = abs_val - int_part

        # Convert integer part to binary
        int_binary = bin(int_part)[2:] if int_part > 0 else '0'

        # Convert fractional part to binary
        frac_binary = ''
        f = frac_part
        for _ in range(23):
            f *= 2
            if f >= 1:
                frac_binary += '1'
                f -= 1
            else:
                frac_binary += '0'
            if f == 0:
                break

        full_binary = f"{int_binary}.{frac_binary}"

        # Normalize: find position of first 1
        if int_part > 0:
            exp = len(int_binary) - 1
        else:
            exp = 0
            for i, bit in enumerate(frac_binary):
                if bit == '1':
                    exp = -(i + 1)
                    break

        biased_exp = exp + 127
        exp_binary = format(biased_exp, '08b')

        # Mantissa (hidden 1 removed)
        combined = int_binary + frac_binary
        # Find the position of the leading 1 and take everything after
        first_one = combined.index('1')
        mantissa_str = combined[first_one + 1:]
        mantissa_str = mantissa_str.ljust(23, '0')[:23]

        return sign_bit, exp_binary, mantissa_str, full_binary, exp, biased_exp

    values = [5.75, -0.375, 1.0]
    print("IEEE 754 Single Precision (32-bit) conversion:")

    for val in values:
        s, e, m, full_bin, exp, biased_exp = explain_conversion(val)
        s_hw, e_hw, m_hw = float_to_ieee754(val)

        print(f"\n  {val}:")
        print(f"    Binary: {full_bin}")
        print(f"    Normalized: 1.{m} x 2^{exp}")
        print(f"    Sign: {s} ({'−' if s == '1' else '+'})")
        print(f"    Exponent: {exp} + 127 (bias) = {biased_exp} = {e}")
        print(f"    Mantissa: {m} (implicit leading 1 omitted)")
        print(f"    Full: {s} {e} {m}")
        # Verify against hardware
        assert s == s_hw and e == e_hw and m == m_hw, \
            f"Mismatch! Expected: {s_hw} {e_hw} {m_hw}"


def exercise_5():
    """
    Convert IEEE 754 single precision bit patterns to decimal:
    (a) 0 10000001 01000000000000000000000 -> +5.0
    (b) 1 01111111 00000000000000000000000 -> -1.0
    (c) 0 00000000 00000000000000000000000 -> +0.0
    """
    def ieee754_to_float(sign_str, exp_str, mant_str):
        """Convert IEEE 754 bit strings to decimal value."""
        sign = (-1) ** int(sign_str)
        exp = int(exp_str, 2)
        mantissa_val = 0
        for i, bit in enumerate(mant_str):
            if bit == '1':
                mantissa_val += 2 ** (-(i + 1))

        if exp == 0 and mantissa_val == 0:
            return sign * 0.0  # Zero
        elif exp == 0:
            # Denormalized: (−1)^s × 0.mantissa × 2^(1−127)
            return sign * mantissa_val * (2 ** (1 - 127))
        elif exp == 255:
            if mantissa_val == 0:
                return sign * float('inf')
            else:
                return float('nan')
        else:
            # Normalized: (−1)^s × 1.mantissa × 2^(exp−127)
            return sign * (1 + mantissa_val) * (2 ** (exp - 127))

    patterns = [
        ("0", "10000001", "01000000000000000000000"),
        ("1", "01111111", "00000000000000000000000"),
        ("0", "00000000", "00000000000000000000000"),
    ]

    print("IEEE 754 bit pattern to decimal conversion:")
    for s, e, m in patterns:
        result = ieee754_to_float(s, e, m)
        exp_val = int(e, 2)
        sign = "+" if s == "0" else "-"

        print(f"\n  {s} {e} {m}")
        print(f"    Sign: {s} → {sign}")
        print(f"    Exponent: {e} = {exp_val}, unbiased = {exp_val} - 127 = {exp_val - 127}")

        if exp_val == 0 and all(b == '0' for b in m):
            print(f"    Special case: Zero")
        else:
            # Show mantissa value
            mant_val = 0
            for i, bit in enumerate(m):
                if bit == '1':
                    mant_val += 2 ** (-(i + 1))
            print(f"    Mantissa: 1 + {mant_val} = {1 + mant_val}")
            print(f"    Value: {sign}1 × {1 + mant_val} × 2^{exp_val - 127} = {result}")

        print(f"    Result: {result}")


def exercise_6():
    """
    Detect overflow in 8-bit two's complement operations:
    (a) 01111111 + 00000001
    (b) 10000000 + 11111111
    (c) 01000000 + 01000000
    """
    def add_with_overflow_check(bin_a, bin_b):
        """Add two 8-bit two's complement numbers and check for overflow."""
        val_a = int(bin_a, 2)
        val_b = int(bin_b, 2)
        if val_a >= 128: val_a -= 256
        if val_b >= 128: val_b -= 256

        raw_sum = val_a + val_b
        result_bin = format(raw_sum & 0xFF, '08b')
        result_signed = raw_sum & 0xFF
        if result_signed >= 128:
            result_signed -= 256

        # Overflow occurs when:
        # - Two positives produce a negative, or
        # - Two negatives produce a positive
        sign_a = bin_a[0]
        sign_b = bin_b[0]
        sign_r = result_bin[0]

        overflow = (sign_a == '0' and sign_b == '0' and sign_r == '1') or \
                   (sign_a == '1' and sign_b == '1' and sign_r == '0')

        return result_bin, val_a, val_b, result_signed, overflow

    operations = [
        ("01111111", "00000001"),  # 127 + 1
        ("10000000", "11111111"),  # -128 + (-1)
        ("01000000", "01000000"),  # 64 + 64
    ]

    print("8-bit two's complement overflow detection:")
    print("Rule: Overflow = same-sign operands produce different-sign result")

    for bin_a, bin_b in operations:
        result, a, b, r, overflow = add_with_overflow_check(bin_a, bin_b)
        print(f"\n  {bin_a} ({a:>4d})")
        print(f"+ {bin_b} ({b:>4d})")
        print(f"= {result} ({r:>4d})")
        if overflow:
            print(f"  ** OVERFLOW! ** {a}+{b}={a+b} is outside [-128, 127]")
            reason = "positive+positive=negative" if a >= 0 else "negative+negative=positive"
            print(f"     Reason: {reason}")
        else:
            print(f"  No overflow.")


def exercise_7():
    """Explain why 0.1 + 0.2 != 0.3 in IEEE 754."""
    print("Why 0.1 + 0.2 ≠ 0.3 in floating-point:")
    print()

    # Show actual stored values
    values = [0.1, 0.2, 0.3, 0.1 + 0.2]
    labels = ["0.1", "0.2", "0.3", "0.1+0.2"]

    for val, label in zip(values, labels):
        # Get IEEE 754 bits
        packed = struct.pack('>d', val)
        bits = ''.join(format(b, '08b') for b in packed)
        print(f"  {label:>8s} stored as: {val:.20f}")
        print(f"          {'bits: ' + bits[:1] + ' ' + bits[1:12] + ' ' + bits[12:]}")

    print(f"\n  0.1 + 0.2 == 0.3 ? {0.1 + 0.2 == 0.3}")
    print(f"  Difference: {(0.1 + 0.2) - 0.3:.25e}")

    print(f"\n  Root cause: 0.1 and 0.2 are infinite repeating fractions in binary.")
    print(f"  0.1 (decimal) = 0.0001100110011... (binary, repeating)")
    print(f"  0.2 (decimal) = 0.0011001100110... (binary, repeating)")
    print(f"  Both are truncated to fit in 52-bit mantissa (double precision),")
    print(f"  and the truncation errors accumulate when added.")


def exercise_8():
    """Explain the Kahan Summation algorithm."""
    print("Kahan Summation Algorithm:")
    print("  Compensated summation that tracks and corrects rounding error.\n")

    # Demonstrate the problem with naive summation
    n = 1_000_000
    naive_sum = 0.0
    for _ in range(n):
        naive_sum += 0.0001

    # Kahan summation
    kahan_sum = 0.0
    compensation = 0.0  # Running compensation for lost low-order bits
    for _ in range(n):
        y = 0.0001 - compensation       # Compensated value to add
        t = kahan_sum + y                # Partial sum (may lose low-order bits of y)
        compensation = (t - kahan_sum) - y  # Recover what was lost
        kahan_sum = t

    expected = 0.0001 * n

    print("  Algorithm pseudocode:")
    print("    sum = 0.0")
    print("    c = 0.0  (compensation)")
    print("    for each value x:")
    print("        y = x - c            # Apply compensation")
    print("        t = sum + y           # Update sum (may lose low bits)")
    print("        c = (t - sum) - y     # Compute lost bits")
    print("        sum = t")
    print()
    print(f"  Adding 0.0001 one million times:")
    print(f"    Expected:        {expected:.15f}")
    print(f"    Naive sum:       {naive_sum:.15f}  (error: {abs(naive_sum - expected):.2e})")
    print(f"    Kahan sum:       {kahan_sum:.15f}  (error: {abs(kahan_sum - expected):.2e})")
    print(f"\n  The Kahan algorithm reduces error from O(n*eps) to O(eps),")
    print(f"  where eps is machine epsilon (~2.2e-16 for double).")


def exercise_9():
    """Explain IEEE 754 positive zero (+0) and negative zero (-0)."""
    pos_zero = 0.0
    neg_zero = -0.0

    print("IEEE 754: +0 and -0")
    print()

    # Bit representations
    for val, label in [(pos_zero, "+0.0"), (neg_zero, "-0.0")]:
        packed = struct.pack('>f', val)
        bits = ''.join(format(b, '08b') for b in packed)
        print(f"  {label}: {bits[:1]} {bits[1:9]} {bits[9:]}")

    print()
    print(f"  +0 == -0   ? {pos_zero == neg_zero}")    # True
    print(f"  +0 is -0   ? {pos_zero is neg_zero}")     # True in CPython (cached)

    # Key differences
    print(f"\n  Key differences:")
    print(f"    1/+0 = {'+inf' if True else 'err'}  (positive infinity)")
    print(f"    1/-0 = {'-inf' if True else 'err'}  (negative infinity)")
    # Actually demonstrate:
    import math
    print(f"    math.copysign(1.0, +0.0) = {math.copysign(1.0, pos_zero)}")
    print(f"    math.copysign(1.0, -0.0) = {math.copysign(1.0, neg_zero)}")

    print(f"\n  Why -0 exists:")
    print(f"    - Sign-magnitude representation: sign bit is independent of value")
    print(f"    - Preserves sign info during underflow: very small negative → -0")
    print(f"    - Useful in complex analysis and limit computations")
    print(f"    - IEEE 754 mandates: +0 == -0 for comparison purposes")


def exercise_10():
    """
    Fix the accumulation error in:
    float sum = 0.0f; for (i=0; i<1000000; i++) sum += 0.0001f;
    """
    print("Fixing floating-point accumulation error:")
    print()

    # Simulate the problem (Python float = double, so we simulate float32)
    import array
    # Use Python's float (double) to show the concept
    n = 1_000_000

    # Method 1: Naive single-precision simulation
    # (Python doesn't have float32, so we use double to demonstrate concept)
    naive_sum = 0.0
    for _ in range(n):
        naive_sum += 0.0001
    print(f"  Naive sum (double):   {naive_sum:.10f}  (expected 100.0)")

    # Method 2: Kahan summation
    kahan_sum = 0.0
    c = 0.0
    for _ in range(n):
        y = 0.0001 - c
        t = kahan_sum + y
        c = (t - kahan_sum) - y
        kahan_sum = t
    print(f"  Kahan sum (double):   {kahan_sum:.10f}")

    # Method 3: Integer arithmetic then divide
    int_sum = n * 1  # 0.0001 * 10000 = 1
    result = int_sum / 10000
    print(f"  Integer arithmetic:   {result:.10f}")

    # Method 4: math.fsum (Python's exact summation)
    import math
    fsum_result = math.fsum([0.0001] * n)
    print(f"  math.fsum:            {fsum_result:.10f}")

    print(f"\n  Solutions for C code:")
    print(f"    1. Use double instead of float (more mantissa bits)")
    print(f"    2. Use Kahan Summation algorithm")
    print(f"    3. Use integer arithmetic: sum_int += 1; result = sum_int / 10000.0")
    print(f"    4. Sum in blocks and add subtotals")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Decimal to Two's Complement", exercise_1),
        ("Exercise 2: Two's Complement to Decimal", exercise_2),
        ("Exercise 3: Two's Complement Arithmetic", exercise_3),
        ("Exercise 4: Decimal to IEEE 754", exercise_4),
        ("Exercise 5: IEEE 754 to Decimal", exercise_5),
        ("Exercise 6: Overflow Detection", exercise_6),
        ("Exercise 7: Why 0.1 + 0.2 ≠ 0.3", exercise_7),
        ("Exercise 8: Kahan Summation Algorithm", exercise_8),
        ("Exercise 9: IEEE 754 +0 and -0", exercise_9),
        ("Exercise 10: Fixing Accumulation Error", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

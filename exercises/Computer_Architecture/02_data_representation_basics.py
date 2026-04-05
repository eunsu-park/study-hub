"""
Exercises for Lesson 02: Data Representation Basics
Topic: Computer_Architecture

Solutions to practice problems covering base conversion (binary, decimal,
hexadecimal), binary arithmetic, two's complement, and data units.
"""


def exercise_1():
    """Convert decimal numbers to binary: (a) 42, (b) 100, (c) 255."""
    # Repeated division by 2: track remainders from bottom to top
    def decimal_to_binary(n):
        """Convert decimal to binary string using repeated division."""
        if n == 0:
            return "0"
        bits = []
        original = n
        while n > 0:
            bits.append(str(n % 2))
            n //= 2
        result = ''.join(reversed(bits))
        return result

    test_values = [42, 100, 255]
    print("Decimal to Binary conversion (repeated division by 2):")
    for val in test_values:
        binary = decimal_to_binary(val)
        # Show step-by-step for first value
        if val == 42:
            print(f"\n  Step-by-step for {val}:")
            n = val
            while n > 0:
                print(f"    {n} / 2 = {n // 2} remainder {n % 2}")
                n //= 2
            print(f"    Read remainders bottom-to-top: {binary}")
        print(f"  {val:>5d} = {binary}")
        # Verify with Python built-in
        assert binary == bin(val)[2:], f"Mismatch for {val}"


def exercise_2():
    """Convert binary numbers to decimal: (a) 10110, (b) 11111111, (c) 10000000."""
    def binary_to_decimal(binary_str):
        """Convert binary string to decimal using positional notation."""
        result = 0
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '1':
                result += 2 ** i
        return result

    test_values = ["10110", "11111111", "10000000"]
    print("Binary to Decimal conversion (positional weight sum):")
    for binary in test_values:
        decimal = binary_to_decimal(binary)
        # Show expansion
        terms = []
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                terms.append(f"2^{i}")
        expansion = " + ".join(reversed(terms))
        print(f"  {binary} = {expansion} = {decimal}")
        assert decimal == int(binary, 2), f"Mismatch for {binary}"


def exercise_3():
    """Convert binary to hexadecimal: (a) 11011010, (b) 101111110000, (c) 32 1-bits."""
    def binary_to_hex(binary_str):
        """Convert binary string to hex by grouping 4 bits from the right."""
        # Pad to multiple of 4
        padded = binary_str.zfill(len(binary_str) + (4 - len(binary_str) % 4) % 4)
        hex_digits = []
        for i in range(0, len(padded), 4):
            nibble = padded[i:i+4]
            hex_digits.append(format(int(nibble, 2), 'X'))
        return ''.join(hex_digits)

    test_values = [
        ("11011010", "DA"),
        ("101111110000", "BF0"),
        ("1" * 32, "FFFFFFFF"),
    ]
    print("Binary to Hexadecimal conversion (group 4 bits = 1 hex digit):")
    for binary, expected in test_values:
        result = binary_to_hex(binary)
        # Show grouping
        padded = binary.zfill(len(binary) + (4 - len(binary) % 4) % 4)
        groups = [padded[i:i+4] for i in range(0, len(padded), 4)]
        group_str = " ".join(groups)
        hex_digits = [format(int(g, 2), 'X') for g in groups]
        print(f"  {binary}")
        print(f"    Grouped: {group_str}")
        print(f"    Hex:     {'    '.join(hex_digits)}")
        print(f"    Result:  0x{result}")
        assert result == expected, f"Expected {expected}, got {result}"


def exercise_4():
    """Convert hexadecimal to binary: (a) 0xAB, (b) 0x1234, (c) 0xDEADBEEF."""
    def hex_to_binary(hex_str):
        """Convert hex string to binary (each hex digit = 4 bits)."""
        hex_str = hex_str.upper().replace("0X", "")
        binary = ''.join(format(int(d, 16), '04b') for d in hex_str)
        return binary

    test_values = [
        ("0xAB", "10101011"),
        ("0x1234", "0001001000110100"),
        ("0xDEADBEEF", "11011110101011011011111011101111"),
    ]
    print("Hexadecimal to Binary conversion (each hex digit -> 4 bits):")
    for hex_val, expected in test_values:
        result = hex_to_binary(hex_val)
        clean = hex_val.upper().replace("0X", "")
        parts = [f"{d}->{format(int(d, 16), '04b')}" for d in clean]
        print(f"  {hex_val}: {', '.join(parts)}")
        print(f"    Result: {result}")
        assert result == expected, f"Expected {expected}, got {result}"


def exercise_5():
    """
    Perform binary additions:
    (a) 1011 + 1101
    (b) 11111111 + 00000001
    (c) 10101010 + 01010101
    """
    def binary_add(a_str, b_str):
        """Simulate binary addition with carry propagation."""
        # Pad to same length
        max_len = max(len(a_str), len(b_str))
        a = a_str.zfill(max_len)
        b = b_str.zfill(max_len)

        result = []
        carry = 0
        carries = []

        for i in range(max_len - 1, -1, -1):
            bit_a = int(a[i])
            bit_b = int(b[i])
            total = bit_a + bit_b + carry
            result.append(str(total % 2))
            carry = total // 2
            carries.append(carry)

        if carry:
            result.append('1')

        return ''.join(reversed(result))

    additions = [
        ("1011", "1101", "11 + 13 = 24"),
        ("11111111", "00000001", "255 + 1 = 256 (overflow in 8-bit!)"),
        ("10101010", "01010101", "170 + 85 = 255"),
    ]

    print("Binary addition with carry propagation:")
    for a, b, note in additions:
        result = binary_add(a, b)
        max_len = max(len(a), len(b), len(result))
        print(f"\n    {a.rjust(max_len)}")
        print(f"  + {b.rjust(max_len)}")
        print(f"  {'-' * (max_len + 1)}")
        print(f"    {result.rjust(max_len)}")
        print(f"    ({note})")
        # Verify
        assert int(result, 2) == int(a, 2) + int(b, 2)


def exercise_6():
    """
    8-bit two's complement problems:
    (a) Binary representation of -45
    (b) Decimal value of 11101100
    (c) Result of 7 - 12
    """
    def to_twos_complement(value, bits=8):
        """Convert a signed integer to two's complement binary string."""
        if value >= 0:
            return format(value, f'0{bits}b')
        else:
            # Two's complement: invert bits of |value| then add 1
            positive = format(abs(value), f'0{bits}b')
            inverted = ''.join('1' if b == '0' else '0' for b in positive)
            # Add 1
            result = format(int(inverted, 2) + 1, f'0{bits}b')
            return result[-bits:]  # Keep only 'bits' bits

    def from_twos_complement(binary_str, bits=8):
        """Convert two's complement binary string to signed integer."""
        value = int(binary_str, 2)
        if binary_str[0] == '1':  # Negative number (MSB = 1)
            value -= (1 << bits)
        return value

    print("8-bit Two's Complement exercises:")

    # (a) -45 in two's complement
    print("\n(a) Represent -45 in 8-bit two's complement:")
    pos_45 = format(45, '08b')
    inverted = ''.join('1' if b == '0' else '0' for b in pos_45)
    result_a = to_twos_complement(-45)
    print(f"    Step 1: +45 in binary   = {pos_45}")
    print(f"    Step 2: Invert all bits = {inverted}")
    print(f"    Step 3: Add 1           = {result_a}")
    print(f"    Verification: {from_twos_complement(result_a)} = -45 ✓")

    # (b) Decimal value of 11101100
    binary_b = "11101100"
    decimal_b = from_twos_complement(binary_b)
    print(f"\n(b) Decimal value of {binary_b}:")
    print(f"    MSB is 1 → negative number")
    inverted_b = ''.join('1' if b == '0' else '0' for b in binary_b)
    magnitude = int(inverted_b, 2) + 1
    print(f"    Step 1: Invert bits = {inverted_b}")
    print(f"    Step 2: Add 1       = {format(magnitude, '08b')} = {magnitude}")
    print(f"    Result: -{magnitude} = {decimal_b}")

    # (c) 7 - 12 in 8-bit two's complement
    print(f"\n(c) Compute 7 - 12 in 8-bit two's complement:")
    bin_7 = to_twos_complement(7)
    bin_neg12 = to_twos_complement(-12)
    print(f"    7  = {bin_7}")
    print(f"   -12 = {bin_neg12}")
    # Add them
    raw_sum = int(bin_7, 2) + int(bin_neg12, 2)
    result_c = format(raw_sum & 0xFF, '08b')  # Mask to 8 bits
    print(f"    Sum = {result_c}")
    print(f"    Decimal = {from_twos_complement(result_c)}")


def exercise_7():
    """
    Convert decimal fractions to binary (up to 4 decimal places):
    (a) 0.5, (b) 0.25, (c) 0.1
    """
    def decimal_fraction_to_binary(fraction, max_places=8):
        """Convert a decimal fraction to binary by repeated multiplication by 2."""
        bits = []
        seen = {}
        for i in range(max_places):
            if fraction in seen:
                bits.append(f"... (repeating from position {seen[fraction]})")
                break
            seen[fraction] = i
            fraction *= 2
            if fraction >= 1:
                bits.append('1')
                fraction -= 1
            else:
                bits.append('0')
            if fraction == 0:
                break
        return '0.' + ''.join(bits)

    values = [0.5, 0.25, 0.1]
    print("Decimal fraction to binary (repeated multiplication by 2):")
    for val in values:
        result = decimal_fraction_to_binary(val, max_places=16)
        print(f"\n  {val}:")
        # Show steps
        frac = val
        for step in range(min(6, 16)):
            frac *= 2
            integer_part = int(frac >= 1)
            print(f"    Step {step+1}: {val if step == 0 else frac/2:.10f} * 2 = {frac:.10f} → integer part = {integer_part}")
            if frac >= 1:
                frac -= 1
            if frac == 0:
                break
        print(f"    Result: {result}")

    # Key insight about 0.1
    print("\nKey insight: 0.1 cannot be represented exactly in binary!")
    print("  This is why 0.1 + 0.2 ≠ 0.3 in floating-point arithmetic.")
    print(f"  Python: 0.1 + 0.2 = {0.1 + 0.2}")
    print(f"  Python: 0.1 + 0.2 == 0.3 → {0.1 + 0.2 == 0.3}")


def exercise_8():
    """Calculate the difference between 1 GiB and 1 GB."""
    gib = 2**30  # 1 GiB = 1,073,741,824 bytes (binary: 1024^3)
    gb = 10**9   # 1 GB  = 1,000,000,000 bytes (decimal: 1000^3)

    diff = gib - gb
    diff_mib = diff / (2**20)

    print("GiB (Gibibyte, binary) vs GB (Gigabyte, decimal):")
    print(f"  1 GiB = 2^30 = {gib:>15,d} bytes")
    print(f"  1 GB  = 10^9 = {gb:>15,d} bytes")
    print(f"  Difference    = {diff:>15,d} bytes")
    print(f"             ≈ {diff_mib:.1f} MiB")
    print(f"             ≈ {diff / 1e6:.1f} MB")
    print(f"\n  This is a {diff/gb*100:.2f}% difference.")
    print("  When you buy a '1 TB' hard drive, you get ~931 GiB usable space.")

    # Show the hierarchy
    print("\n  Binary prefixes (IEC):    Decimal prefixes (SI):")
    print("  1 KiB = 1,024 B          1 KB = 1,000 B")
    print("  1 MiB = 1,048,576 B      1 MB = 1,000,000 B")
    print("  1 GiB = 1,073,741,824 B  1 GB = 1,000,000,000 B")
    print("  1 TiB = 1,099,511,627,776 B  1 TB = 1,000,000,000,000 B")


def exercise_9():
    """Explain why a 32-bit system can only address 4GB of memory."""
    bits = 32
    max_addresses = 2 ** bits
    max_bytes = max_addresses  # Each address points to 1 byte (byte-addressable)

    print(f"32-bit address space analysis:")
    print(f"  Address width: {bits} bits")
    print(f"  Total addressable locations: 2^{bits} = {max_addresses:,} addresses")
    print(f"  Since each address points to 1 byte (byte-addressable memory):")
    print(f"  Maximum memory = {max_bytes:,} bytes")
    print(f"                 = {max_bytes / 2**30:.0f} GiB")
    print(f"                 ≈ 4 GB")

    print(f"\n  This is why 32-bit operating systems cannot use more than ~4 GiB RAM.")
    print(f"  In practice, usable memory is less because some addresses are reserved")
    print(f"  for memory-mapped I/O, BIOS, and video memory.")
    print(f"\n  64-bit systems: 2^64 = {2**64:,} addresses = 16 EiB (exbibytes)")
    print(f"  This is effectively unlimited for current hardware.")

    # PAE (Physical Address Extension) as a workaround
    print(f"\n  Historical workaround: PAE (Physical Address Extension)")
    print(f"  - Extended physical address to 36 bits → 64 GiB addressable")
    print(f"  - But each process still limited to 4 GiB virtual address space")


def exercise_10():
    """
    Analyze signed char overflow: signed char a = 127; a = a + 1;
    What is the value of a?
    """
    import ctypes

    print("Signed char overflow analysis:")
    print("  signed char range: -128 to 127 (8-bit two's complement)")
    print()

    # Simulate 8-bit two's complement
    a = 127
    print(f"  a = {a}")
    print(f"  Binary: {format(a & 0xFF, '08b')}")

    # Adding 1 to 127
    a_plus_1 = (a + 1) & 0xFF  # Keep only 8 bits
    # Interpret as signed
    if a_plus_1 >= 128:
        signed_result = a_plus_1 - 256
    else:
        signed_result = a_plus_1

    print(f"\n  a + 1 operation:")
    print(f"    01111111 (127)")
    print(f"  + 00000001 (  1)")
    print(f"  ----------")
    print(f"    10000000")
    print(f"\n  MSB is now 1 → interpreted as negative in two's complement")
    print(f"  10000000 in two's complement = {signed_result}")
    print(f"\n  Answer: a = {signed_result}")
    print(f"\n  This is integer overflow (undefined behavior in C standard,")
    print(f"  but wraps around on most architectures using two's complement).")

    # Verify with ctypes
    c_val = ctypes.c_int8(128)
    print(f"  Verification (ctypes.c_int8(128)): {c_val.value}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Decimal to Binary Conversion", exercise_1),
        ("Exercise 2: Binary to Decimal Conversion", exercise_2),
        ("Exercise 3: Binary to Hexadecimal Conversion", exercise_3),
        ("Exercise 4: Hexadecimal to Binary Conversion", exercise_4),
        ("Exercise 5: Binary Addition", exercise_5),
        ("Exercise 6: Two's Complement", exercise_6),
        ("Exercise 7: Decimal Fractions to Binary", exercise_7),
        ("Exercise 8: GiB vs GB Difference", exercise_8),
        ("Exercise 9: 32-bit Address Space Limitation", exercise_9),
        ("Exercise 10: Signed Char Overflow", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

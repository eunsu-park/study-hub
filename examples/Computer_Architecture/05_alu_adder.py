"""
ALU and Adder Circuits

Demonstrates:
- Half adder, full adder (gate-level)
- Ripple-carry adder (N-bit)
- ALU operations (add, sub, AND, OR, SLT)
- Carry-lookahead concept

Theory:
- Ripple-carry adder: chain N full adders. Simple but slow
  (O(N) carry propagation delay).
- Carry-lookahead: precompute carries using generate/propagate
  signals. O(log N) delay but more hardware.
- ALU: combinational circuit performing arithmetic and logic
  operations, selected by ALU control lines.

Adapted from Computer Architecture Lesson 05.
"""


# ── Gate-Level Building Blocks ──────────────────────────────────────────

def half_adder(a: int, b: int) -> tuple[int, int]:
    """Returns (sum, carry)."""
    return a ^ b, a & b


def full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    """Returns (sum, carry_out)."""
    s1 = a ^ b
    c1 = a & b
    s = s1 ^ cin
    # Carry-out is 1 if either (a AND b) generated a carry, or if the
    # partial sum (a XOR b) propagated the incoming carry.  This
    # generate/propagate decomposition is the foundation of carry-
    # lookahead optimization.
    cout = c1 | (s1 & cin)
    return s, cout


# ── Ripple-Carry Adder ──────────────────────────────────────────────────

def ripple_carry_add(a: list[int], b: list[int], cin: int = 0) -> tuple[list[int], int]:
    """N-bit ripple-carry adder. Bits are LSB-first.

    Returns (sum_bits, carry_out).
    """
    n = len(a)
    result = []
    carry = cin

    for i in range(n):
        s, carry = full_adder(a[i], b[i], carry)
        result.append(s)

    return result, carry


def int_to_bits(n: int, width: int) -> list[int]:
    """Convert integer to LSB-first bit list (two's complement)."""
    if n < 0:
        n = (1 << width) + n
    return [(n >> i) & 1 for i in range(width)]


def bits_to_int(bits: list[int], signed: bool = True) -> int:
    """Convert LSB-first bit list to integer."""
    n = sum(b << i for i, b in enumerate(bits))
    if signed and bits[-1] == 1:
        n -= (1 << len(bits))
    return n


def bits_to_str(bits: list[int]) -> str:
    """Convert LSB-first bits to MSB-first string."""
    return "".join(str(b) for b in reversed(bits))


def demo_ripple_carry():
    """Demonstrate ripple-carry adder."""
    print("=" * 60)
    print("RIPPLE-CARRY ADDER (8-bit)")
    print("=" * 60)

    test_cases = [
        (5, 3),
        (42, 85),
        (127, 1),    # overflow
        (-1, 1),     # -1 + 1 = 0
        (-50, 30),
    ]

    width = 8
    print(f"\n  {'A':>5}  {'B':>5}  {'A bits':>10}  {'B bits':>10}  "
          f"{'Sum bits':>10}  {'Result':>7}  {'Cout':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*7}  {'-'*5}")

    for a, b in test_cases:
        a_bits = int_to_bits(a, width)
        b_bits = int_to_bits(b, width)
        sum_bits, cout = ripple_carry_add(a_bits, b_bits)
        result = bits_to_int(sum_bits)
        print(f"  {a:>5}  {b:>5}  {bits_to_str(a_bits):>10}  "
              f"{bits_to_str(b_bits):>10}  {bits_to_str(sum_bits):>10}  "
              f"{result:>7}  {cout:>5}")

    # Show carry propagation
    print("\n  Carry propagation for 127 + 1:")
    a_bits = int_to_bits(127, width)
    b_bits = int_to_bits(1, width)
    carry = 0
    print(f"    Bit:   ", end="")
    for i in range(width):
        print(f"  {i}", end="")
    print()
    print(f"    A:     ", end="")
    for b in a_bits:
        print(f"  {b}", end="")
    print()
    print(f"    B:     ", end="")
    for b in b_bits:
        print(f"  {b}", end="")
    print()
    print(f"    Carry: ", end="")
    for i in range(width):
        s, carry = full_adder(a_bits[i], b_bits[i], carry)
        print(f"  {carry}", end="")
    print(f"  (propagates through all bits)")


# ── ALU ─────────────────────────────────────────────────────────────────

class ALU:
    """Simple N-bit ALU supporting basic operations.

    Operations (2-bit control):
    - 00: AND
    - 01: OR
    - 10: ADD
    - 11: SUB (also sets SLT flag)
    """

    def __init__(self, width: int = 8):
        self.width = width

    def execute(self, a: int, b: int, op: int) -> dict:
        """Execute ALU operation. Returns result dict."""
        a_bits = int_to_bits(a, self.width)
        b_bits = int_to_bits(b, self.width)

        if op == 0b00:  # AND
            result_bits = [a_bits[i] & b_bits[i] for i in range(self.width)]
            cout = 0
            op_name = "AND"
        elif op == 0b01:  # OR
            result_bits = [a_bits[i] | b_bits[i] for i in range(self.width)]
            cout = 0
            op_name = "OR"
        elif op == 0b10:  # ADD
            result_bits, cout = ripple_carry_add(a_bits, b_bits)
            op_name = "ADD"
        elif op == 0b11:  # SUB (A - B = A + ~B + 1)
            # Subtraction reuses the adder by exploiting two's complement:
            # A - B = A + (~B + 1).  The bitwise inversion plus cin=1
            # computes the negation, so no separate subtractor circuit is
            # needed — just an inverter and a mux on the B input.
            b_inv = [1 - bit for bit in b_bits]
            result_bits, cout = ripple_carry_add(a_bits, b_inv, cin=1)
            op_name = "SUB"
        else:
            raise ValueError(f"Unknown ALU op: {op}")

        result = bits_to_int(result_bits)
        zero = all(b == 0 for b in result_bits)
        negative = result_bits[-1] == 1
        overflow = False
        if op in (0b10, 0b11):
            # Signed overflow can only occur when both operands have the
            # same sign (after accounting for SUB negation), yet the
            # result has the opposite sign.  Two opposite-sign operands
            # can never overflow because their sum is between them.
            a_sign = a_bits[-1]
            b_sign = b_bits[-1] if op == 0b10 else (1 - b_bits[-1])
            if a_sign == b_sign and result_bits[-1] != a_sign:
                overflow = True

        return {
            "op": op_name,
            "a": a,
            "b": b,
            "result": result,
            "result_bits": bits_to_str(result_bits),
            "zero": zero,
            "negative": negative,
            "overflow": overflow,
            "carry_out": cout,
        }


def demo_alu():
    """Demonstrate ALU operations."""
    print("\n" + "=" * 60)
    print("ALU OPERATIONS (8-bit)")
    print("=" * 60)

    alu = ALU(width=8)

    operations = [
        (42, 15, 0b00, "AND"),
        (42, 15, 0b01, "OR"),
        (42, 15, 0b10, "ADD"),
        (42, 15, 0b11, "SUB"),
        (100, 50, 0b10, "ADD"),
        (100, 50, 0b11, "SUB"),
        (0, 0, 0b11, "SUB (zero test)"),
        (-50, 30, 0b10, "ADD (neg)"),
    ]

    print(f"\n  {'Op':>5}  {'A':>5}  {'B':>5}  {'Result':>7}  {'Bits':>10}  "
          f"{'Z':>2}  {'N':>2}  {'V':>2}  {'C':>2}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*10}  "
          f"{'-'*2}  {'-'*2}  {'-'*2}  {'-'*2}")

    for a, b, op, label in operations:
        r = alu.execute(a, b, op)
        print(f"  {r['op']:>5}  {a:>5}  {b:>5}  {r['result']:>7}  "
              f"{r['result_bits']:>10}  "
              f"{int(r['zero']):>2}  {int(r['negative']):>2}  "
              f"{int(r['overflow']):>2}  {r['carry_out']:>2}")

    # SLT (Set Less Than) using SUB
    print("\n  SLT (Set Less Than) via SUB:")
    for a, b in [(5, 10), (10, 5), (5, 5), (-3, 2)]:
        r = alu.execute(a, b, 0b11)
        # SLT checks (negative XOR overflow) rather than just negative,
        # because overflow inverts the sign bit's meaning.  Without this
        # correction, comparing -128 < 1 would wrongly return 0.
        slt = 1 if r["negative"] != r["overflow"] else 0
        print(f"    {a} < {b}? SLT = {slt} "
              f"(negative={int(r['negative'])}, overflow={int(r['overflow'])})")


# ── Carry-Lookahead Concept ─────────────────────────────────────────────

def demo_carry_lookahead():
    """Explain carry-lookahead concept."""
    print("\n" + "=" * 60)
    print("CARRY-LOOKAHEAD CONCEPT")
    print("=" * 60)

    print("""
  Ripple-carry delay: O(N) — carry must propagate through all bits.

  Carry-lookahead: precompute using Generate (G) and Propagate (P):
    Gi = Ai · Bi     (bit i generates carry)
    Pi = Ai ⊕ Bi     (bit i propagates carry)

    C1 = G0 + P0·C0
    C2 = G1 + P1·G0 + P1·P0·C0
    C3 = G2 + P2·G1 + P2·P1·G0 + P2·P1·P0·C0

  All carries computed in O(1) from G, P values (2 gate levels).
  Total delay: O(log N) using hierarchical CLA groups.
""")

    # Compute G and P for an example
    a, b = 0b10110, 0b01101
    width = 5
    a_bits = int_to_bits(a, width)
    b_bits = int_to_bits(b, width)

    print(f"  Example: A={a:05b}, B={b:05b}")
    print(f"  {'Bit':>5}  {'A':>3}  {'B':>3}  {'G=A·B':>6}  {'P=A⊕B':>6}")
    print(f"  {'-'*5}  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*6}")

    for i in range(width):
        g = a_bits[i] & b_bits[i]
        p = a_bits[i] ^ b_bits[i]
        print(f"  {i:>5}  {a_bits[i]:>3}  {b_bits[i]:>3}  {g:>6}  {p:>6}")


if __name__ == "__main__":
    demo_ripple_carry()
    demo_alu()
    demo_carry_lookahead()

"""
Logic Gate Simulator

Demonstrates:
- Basic gates: AND, OR, NOT, XOR, NAND, NOR, XNOR
- Truth table generation
- Combinational circuit building (multiplexer, decoder, adder)
- Boolean expression evaluation

Theory:
- Logic gates are the fundamental building blocks of digital circuits.
- Any Boolean function can be implemented using only NAND (or only NOR)
  gates — they are "universal" gates.
- Combinational circuits: output depends only on current inputs.

Adapted from Computer Architecture Lesson 04.
"""

from typing import Callable


# ── Basic Gates ─────────────────────────────────────────────────────────

def AND(a: int, b: int) -> int:
    return a & b

def OR(a: int, b: int) -> int:
    return a | b

def NOT(a: int) -> int:
    # Using 1-a instead of bitwise ~a because our signals are single-bit
    # (0 or 1).  Python's ~ would produce -2 or -1 due to arbitrary-
    # precision integers, so arithmetic inversion is the correct choice.
    return 1 - a

def XOR(a: int, b: int) -> int:
    return a ^ b

def NAND(a: int, b: int) -> int:
    return NOT(AND(a, b))

def NOR(a: int, b: int) -> int:
    return NOT(OR(a, b))

def XNOR(a: int, b: int) -> int:
    return NOT(XOR(a, b))


# ── Truth Table Generator ──────────────────────────────────────────────

def truth_table_2input(name: str, gate: Callable[[int, int], int]) -> None:
    """Print truth table for a 2-input gate."""
    print(f"  {name}:")
    print(f"    A  B | Out")
    print(f"    ---------")
    for a in [0, 1]:
        for b in [0, 1]:
            print(f"    {a}  {b} |  {gate(a, b)}")
    print()


def truth_table_1input(name: str, gate: Callable[[int], int]) -> None:
    """Print truth table for a 1-input gate."""
    print(f"  {name}:")
    print(f"    A | Out")
    print(f"    ------")
    for a in [0, 1]:
        print(f"    {a} |  {gate(a)}")
    print()


def demo_basic_gates():
    """Show truth tables for all basic gates."""
    print("=" * 60)
    print("BASIC LOGIC GATES")
    print("=" * 60)
    print()

    truth_table_1input("NOT", NOT)
    truth_table_2input("AND", AND)
    truth_table_2input("OR", OR)
    truth_table_2input("XOR", XOR)
    truth_table_2input("NAND", NAND)
    truth_table_2input("NOR", NOR)
    truth_table_2input("XNOR", XNOR)


# ── NAND Universality ───────────────────────────────────────────────────

def demo_nand_universal():
    """Show that NAND can implement all other gates."""
    print("=" * 60)
    print("NAND UNIVERSALITY")
    print("=" * 60)

    # NOT from NAND: feeding the same signal to both inputs of NAND
    # gives NAND(a,a) = ~(a AND a) = ~a.  This is the basis for
    # NAND universality — all other gates are built on top of this.
    def NOT_nand(a):
        return NAND(a, a)

    # AND from NAND
    def AND_nand(a, b):
        return NOT_nand(NAND(a, b))

    # OR from NAND
    def OR_nand(a, b):
        return NAND(NOT_nand(a), NOT_nand(b))

    # XOR from NAND
    def XOR_nand(a, b):
        t1 = NAND(a, b)
        return NAND(NAND(a, t1), NAND(b, t1))

    gates = [
        ("NOT (from NAND)", lambda a, b: NOT_nand(a), NOT),
        ("AND (from NAND)", AND_nand, AND),
        ("OR (from NAND)", OR_nand, OR),
        ("XOR (from NAND)", XOR_nand, XOR),
    ]

    for name, nand_impl, original in gates:
        print(f"\n  {name}:")
        all_match = True
        for a in [0, 1]:
            for b in [0, 1]:
                result = nand_impl(a, b)
                expected = original(a, b) if original != NOT else NOT(a)
                match = result == expected
                all_match &= match
        print(f"    Verified: {'PASS' if all_match else 'FAIL'}")


# ── Combinational Circuits ──────────────────────────────────────────────

def mux_2to1(a: int, b: int, sel: int) -> int:
    """2-to-1 multiplexer: output = a if sel=0, b if sel=1."""
    # MUX is the hardware analogue of an if-else.  The AND gates act as
    # enable masks — only one input passes through depending on sel,
    # and OR combines the two mutually exclusive paths.
    return OR(AND(a, NOT(sel)), AND(b, sel))


def decoder_2to4(a: int, b: int) -> list[int]:
    """2-to-4 decoder: activates one of 4 outputs."""
    return [
        AND(NOT(a), NOT(b)),  # 00
        AND(NOT(a), b),       # 01
        AND(a, NOT(b)),       # 10
        AND(a, b),            # 11
    ]


def half_adder(a: int, b: int) -> tuple[int, int]:
    """Half adder: returns (sum, carry)."""
    return XOR(a, b), AND(a, b)


def full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    """Full adder: returns (sum, carry_out)."""
    # A full adder is two half-adders chained: the first adds a+b,
    # the second folds in carry-in.  OR on the two carries works
    # because at most one half-adder can generate a carry at a time.
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return s2, OR(c1, c2)


def demo_combinational():
    """Demonstrate combinational circuits."""
    print("\n" + "=" * 60)
    print("COMBINATIONAL CIRCUITS")
    print("=" * 60)

    # Multiplexer
    print("\n  2-to-1 Multiplexer:")
    print("    A  B  Sel | Out")
    print("    -------------")
    for a in [0, 1]:
        for b in [0, 1]:
            for sel in [0, 1]:
                out = mux_2to1(a, b, sel)
                print(f"    {a}  {b}   {sel}  |  {out}")

    # Decoder
    print("\n  2-to-4 Decoder:")
    print("    A  B | D0 D1 D2 D3")
    print("    --------------------")
    for a in [0, 1]:
        for b in [0, 1]:
            outputs = decoder_2to4(a, b)
            out_str = "  ".join(str(o) for o in outputs)
            print(f"    {a}  {b} |  {out_str}")

    # Half adder
    print("\n  Half Adder:")
    print("    A  B | Sum Carry")
    print("    -----------------")
    for a in [0, 1]:
        for b in [0, 1]:
            s, c = half_adder(a, b)
            print(f"    {a}  {b} |  {s}    {c}")

    # Full adder
    print("\n  Full Adder:")
    print("    A  B  Cin | Sum Cout")
    print("    ---------------------")
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                s, cout = full_adder(a, b, cin)
                print(f"    {a}  {b}   {cin}  |  {s}    {cout}")


# ── Boolean Expression Evaluator ────────────────────────────────────────

def demo_boolean_expressions():
    """Evaluate and verify Boolean algebra identities."""
    print("\n" + "=" * 60)
    print("BOOLEAN ALGEBRA IDENTITIES")
    print("=" * 60)

    identities = [
        ("De Morgan: ~(A·B) = ~A+~B",
         lambda a, b: NAND(a, b),
         lambda a, b: OR(NOT(a), NOT(b))),
        ("De Morgan: ~(A+B) = ~A·~B",
         lambda a, b: NOR(a, b),
         lambda a, b: AND(NOT(a), NOT(b))),
        ("Distributive: A·(B+C) = A·B+A·C",
         lambda a, b, c: AND(a, OR(b, c)),
         lambda a, b, c: OR(AND(a, b), AND(a, c))),
        ("Absorption: A+(A·B) = A",
         lambda a, b: OR(a, AND(a, b)),
         lambda a, b: a),
    ]

    for name, lhs, rhs in identities:
        # Determine arity
        import inspect
        n_args = len(inspect.signature(lhs).parameters)
        all_match = True
        for vals in range(1 << n_args):
            args = [(vals >> i) & 1 for i in range(n_args)]
            if lhs(*args) != rhs(*args):
                all_match = False
                break
        status = "VERIFIED" if all_match else "FAILED"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    demo_basic_gates()
    demo_nand_universal()
    demo_combinational()
    demo_boolean_expressions()

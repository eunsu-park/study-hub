"""
Exercises for Lesson 05: Combinational Logic
Topic: Computer_Architecture

Solutions to practice problems covering half/full adders, ripple carry adders,
multiplexers, decoders, priority encoders, and adder/subtractor design.
"""
from itertools import product


def exercise_1():
    """Explain the difference between a half adder and a full adder."""
    print("Half Adder vs Full Adder:")
    print()

    # Half Adder: 2 inputs, 2 outputs
    print("  Half Adder:")
    print("    Inputs:  A, B (two single bits)")
    print("    Outputs: Sum = A XOR B,  Carry = A AND B")
    print("    No carry-in from a previous stage.")
    print()
    print("    Truth table:")
    print("    A  B | Sum Carry")
    for a, b in product([0, 1], repeat=2):
        s = a ^ b
        c = a & b
        print(f"    {a}  {b} |  {s}    {c}")

    print()

    # Full Adder: 3 inputs, 2 outputs
    print("  Full Adder:")
    print("    Inputs:  A, B, Cin (carry from lower bit)")
    print("    Outputs: Sum = A XOR B XOR Cin")
    print("             Cout = (A AND B) OR (Cin AND (A XOR B))")
    print("    Accepts carry-in — can be chained for multi-bit addition.")
    print()
    print("    Truth table:")
    print("    A  B  Cin | Sum Cout")
    for a, b, cin in product([0, 1], repeat=3):
        s = a ^ b ^ cin
        cout = (a & b) | (cin & (a ^ b))
        print(f"    {a}  {b}   {cin}  |  {s}    {cout}")


def exercise_2():
    """
    Calculate 0111 + 0011 in a 4-bit ripple carry adder.
    Show S and Cout for each Full Adder stage.
    """
    def full_adder(a, b, cin):
        """Single-bit full adder: returns (sum, carry_out)."""
        s = a ^ b ^ cin
        cout = (a & b) | (cin & (a ^ b))
        return s, cout

    A = [1, 1, 1, 0]  # 0111 (LSB first: bit 0..3)
    B = [1, 1, 0, 0]  # 0011 (LSB first)

    print("4-bit Ripple Carry Adder: 0111 + 0011")
    print()
    print("  Carry propagates from FA0 (LSB) to FA3 (MSB):")
    print()

    carry = 0
    result = []
    for i in range(4):
        s, cout = full_adder(A[i], B[i], carry)
        print(f"  FA{i}: A={A[i]}, B={B[i]}, Cin={carry} → Sum={s}, Cout={cout}")
        result.append(s)
        carry = cout

    # Format result (MSB first)
    result_str = ''.join(str(b) for b in reversed(result))
    carry_str = str(carry)
    print(f"\n  Final: Cout={carry}, Result={carry_str}{result_str}")
    print(f"  Decimal: 7 + 3 = {int(carry_str + result_str, 2)} = 10")


def exercise_3():
    """
    8:1 MUX with select lines S2S1S0 = 101.
    Which data input goes to output?
    """
    print("8:1 Multiplexer with S2S1S0 = 101:")
    print()

    s2, s1, s0 = 1, 0, 1
    selected = s2 * 4 + s1 * 2 + s0
    print(f"  Select value: S2={s2}, S1={s1}, S0={s0}")
    print(f"  Binary: 101 = {selected} (decimal)")
    print(f"  Data input D{selected} is routed to the output Y")
    print()

    # Simulate MUX operation
    data_inputs = [0, 0, 0, 0, 0, 1, 0, 0]  # D5 = 1, rest = 0
    print("  Simulation (D5 = 1, all others = 0):")
    print(f"  Data inputs: D0..D7 = {data_inputs}")

    # MUX function
    output = data_inputs[selected]
    print(f"  Output Y = D{selected} = {output}")
    print()
    print("  MUX equation: Y = S2'S1'S0'·D0 + S2'S1'S0·D1 + ... + S2·S1·S0·D7")
    print(f"  With S=101: Y = D{selected}")


def exercise_4():
    """Implement a full adder using only NAND gates (minimum 9 gates)."""
    NAND = lambda a, b: 1 - (a & b)

    def full_adder_nand(a, b, cin):
        """Full adder using 9 NAND gates."""
        # XOR(A,B) using 4 NANDs
        g1 = NAND(a, b)
        g2 = NAND(a, g1)
        g3 = NAND(g1, b)
        g4 = NAND(g2, g3)  # g4 = A XOR B

        # XOR(g4, Cin) using 4 NANDs for Sum
        g5 = NAND(g4, cin)
        g6 = NAND(g4, g5)
        g7 = NAND(g5, cin)
        g8 = NAND(g6, g7)  # g8 = Sum = A XOR B XOR Cin

        # Carry = (A AND B) OR (Cin AND (A XOR B))
        # = NAND(NAND(A,B), NAND(Cin, A XOR B))
        g9 = NAND(g1, g5)  # g1 = NAND(A,B), g5 = NAND(A^B, Cin)
        # Cout = NAND(NAND(A,B), NAND(A^B, Cin))
        return g8, g9

    print("Full Adder using 9 NAND gates:")
    print("  Gate connections:")
    print("    g1 = NAND(A, B)")
    print("    g2 = NAND(A, g1)         ─┐")
    print("    g3 = NAND(g1, B)          ├─ g4 = NAND(g2,g3) = A XOR B")
    print("    g5 = NAND(g4, Cin)")
    print("    g6 = NAND(g4, g5)         ─┐")
    print("    g7 = NAND(g5, Cin)         ├─ g8 = NAND(g6,g7) = Sum")
    print("    g9 = NAND(g1, g5)         = Cout")
    print()

    # Verify against standard full adder
    print("  Verification:")
    print("  A  B  Cin | Sum Cout | NAND_Sum NAND_Cout")
    all_correct = True
    for a, b, cin in product([0, 1], repeat=3):
        expected_sum = a ^ b ^ cin
        expected_cout = (a & b) | (cin & (a ^ b))
        nand_sum, nand_cout = full_adder_nand(a, b, cin)
        match = "ok" if (nand_sum == expected_sum and nand_cout == expected_cout) else "FAIL"
        print(f"  {a}  {b}   {cin}  |  {expected_sum}    {expected_cout}   |    {nand_sum}        {nand_cout}     {match}")
        if match != "ok":
            all_correct = False
    print(f"  All correct: {all_correct}")


def exercise_5():
    """
    Implement Y = A'B + AB'C + ABC' using a 4:1 MUX.
    Use A, B as select lines; express data inputs in terms of C.
    """
    NOT = lambda x: 1 - x

    print("Implement Y = A'B + AB'C + ABC' using 4:1 MUX:")
    print()
    print("  Strategy: Use A,B as select lines (S1=A, S0=B)")
    print("  For each combination of A,B, express Y in terms of C:")
    print()

    # Enumerate: for each (A,B), what is Y as function of C?
    for a, b in product([0, 1], repeat=2):
        y0 = (NOT(a) & b & 1) | (a & NOT(b) & 0) | (a & b & NOT(0))  # C=0
        y1 = (NOT(a) & b & 1) | (a & NOT(b) & 1) | (a & b & NOT(1))  # C=1

        # Original function
        y_c0 = (NOT(a) * b) | (a * NOT(b) * 0) | (a * b * NOT(0))
        y_c1 = (NOT(a) * b) | (a * NOT(b) * 1) | (a * b * NOT(1))

        if y_c0 == 0 and y_c1 == 0:
            data = "0"
        elif y_c0 == 0 and y_c1 == 1:
            data = "C"
        elif y_c0 == 1 and y_c1 == 0:
            data = "C'"
        elif y_c0 == 1 and y_c1 == 1:
            data = "1"

        idx = a * 2 + b
        print(f"  D{idx} (A={a}, B={b}): Y(C=0)={y_c0}, Y(C=1)={y_c1} → D{idx} = {data}")

    print()
    print("  MUX connections:")
    print("    S1 = A, S0 = B")
    print("    D0 = 0   (A=0, B=0)")
    print("    D1 = 1   (A=0, B=1) → always A'B")
    print("    D2 = C   (A=1, B=0) → AB'C")
    print("    D3 = C'  (A=1, B=1) → ABC'")

    # Verify
    print("\n  Verification:")
    for a, b, c in product([0, 1], repeat=3):
        original = (NOT(a) & b) | (a & NOT(b) & c) | (a & b & NOT(c))
        # MUX output
        data = [0, 1, c, NOT(c)]
        mux_out = data[a * 2 + b]
        status = "ok" if original == mux_out else "FAIL"
        print(f"    A={a} B={b} C={c}: original={original}, MUX={mux_out} {status}")


def exercise_6():
    """
    Implement Y = Σm(0,2,5,7) using a 3:8 decoder and OR gate.
    """
    print("Implement Y = Σm(0,2,5,7) using 3:8 Decoder + OR gate:")
    print()

    minterms = {0, 2, 5, 7}

    print("  A 3:8 decoder has 3 inputs (A,B,C) and 8 outputs (m0..m7).")
    print("  Each output mi = 1 only when the input equals i.")
    print(f"  Connect outputs {sorted(minterms)} to an OR gate:")
    print(f"  Y = m0 + m2 + m5 + m7")
    print()

    # Simulate decoder
    print("  Verification:")
    print("  A  B  C | m0 m1 m2 m3 m4 m5 m6 m7 | Y")
    for a, b, c in product([0, 1], repeat=3):
        idx = a * 4 + b * 2 + c
        decoder_outputs = [1 if i == idx else 0 for i in range(8)]
        y = 1 if idx in minterms else 0
        m_str = '  '.join(str(d) for d in decoder_outputs)
        print(f"  {a}  {b}  {c} |  {m_str} | {y}")


def exercise_7():
    """
    Analyze MUX circuit with A as both data and select.
    S=A, D0=A, D1=B → Y = ?
    """
    print("MUX circuit analysis (A is both data input and select):")
    print()
    print("  Circuit: D0=A, D1=B, Select=A")
    print("  MUX equation: Y = S'·D0 + S·D1")
    print("               Y = A'·A + A·B")
    print("               Y = 0 + AB  (since A'·A = 0)")
    print("               Y = AB")
    print()

    print("  Truth table:")
    print("  A  B | S  D0 D1 | Y=AB")
    for a, b in product([0, 1], repeat=2):
        s = a
        d0 = a
        d1 = b
        y_mux = (1 - s) * d0 + s * d1  # MUX formula
        y_and = a & b                    # Simplified
        print(f"  {a}  {b} | {s}   {d0}  {d1} |  {y_mux}")
        assert y_mux == y_and, f"Mismatch at A={a}, B={b}"

    print("\n  Result: The circuit implements an AND gate (Y = AB)")
    print("  When A=0: select D0=A=0, so Y=0")
    print("  When A=1: select D1=B, so Y=B → Y=A·B")


def exercise_8():
    """
    Calculate worst-case delay of an 8-bit ripple carry adder.
    Each FA has 10ns delay.
    """
    num_bits = 8
    fa_delay_ns = 10

    print(f"Worst-case delay of {num_bits}-bit Ripple Carry Adder:")
    print(f"  Each Full Adder gate delay: {fa_delay_ns} ns")
    print()

    # In a ripple carry adder, the carry must propagate through ALL stages
    # The worst case is when the carry propagates from bit 0 to bit 7
    total_delay = num_bits * fa_delay_ns

    print(f"  Worst case: carry propagates from FA0 through FA{num_bits-1}")
    print(f"  Total delay = {num_bits} stages × {fa_delay_ns} ns/stage = {total_delay} ns")
    print(f"  Maximum clock frequency = 1 / {total_delay}ns = {1e9/total_delay/1e6:.1f} MHz")
    print()

    # Compare with Carry Lookahead Adder
    cla_delay = 4 * fa_delay_ns  # Approximately O(log n) depth
    print(f"  Comparison: Carry Lookahead Adder (CLA)")
    print(f"  CLA generates carry in O(log n) time: ~{cla_delay} ns")
    print(f"  Speedup: {total_delay/cla_delay:.1f}x faster")
    print()
    print(f"  This is why ripple carry adders are not used in high-performance CPUs.")
    print(f"  Modern CPUs use CLA or prefix adders (Kogge-Stone, Brent-Kung).")


def exercise_9():
    """
    Design a 4-bit adder/subtractor.
    Sub=0: addition, Sub=1: subtraction (using two's complement).
    """
    def full_adder(a, b, cin):
        s = a ^ b ^ cin
        cout = (a & b) | (cin & (a ^ b))
        return s, cout

    print("4-bit Adder/Subtractor design:")
    print()
    print("  Key idea: A - B = A + (~B + 1) = A + (B XOR 1...1) + 1")
    print("  Use XOR gates to conditionally invert B:")
    print("    - Sub=0: B XOR 0 = B (addition), Cin=0")
    print("    - Sub=1: B XOR 1 = B' (inversion), Cin=1 (adds 1 for two's complement)")
    print()
    print("  Circuit:")
    print("    For each bit i: B_eff[i] = B[i] XOR Sub")
    print("    Carry-in to FA0 = Sub")
    print()

    # Simulate
    test_cases = [
        (7, 3, 0, "Addition:    7 + 3"),
        (5, 3, 1, "Subtraction: 5 - 3"),
        (3, 5, 1, "Subtraction: 3 - 5"),
        (0, 0, 0, "Addition:    0 + 0"),
    ]

    for a_val, b_val, sub, label in test_cases:
        # Convert to 4-bit arrays (LSB first)
        a_bits = [(a_val >> i) & 1 for i in range(4)]
        b_bits = [(b_val >> i) & 1 for i in range(4)]

        # Apply XOR with Sub
        b_eff = [b ^ sub for b in b_bits]

        # Ripple carry addition
        carry = sub  # Cin = Sub
        result = []
        for i in range(4):
            s, carry = full_adder(a_bits[i], b_eff[i], carry)
            result.append(s)

        result_val = sum(bit << i for i, bit in enumerate(result))
        # Interpret as signed 4-bit
        if result_val >= 8:
            signed_result = result_val - 16
        else:
            signed_result = result_val

        a_str = ''.join(str(b) for b in reversed(a_bits))
        b_str = ''.join(str(b) for b in reversed(b_bits))
        r_str = ''.join(str(b) for b in reversed(result))
        print(f"  {label}: {a_str} op {b_str} = {r_str} ({signed_result})")


def exercise_10():
    """
    Design a priority encoder for 8 interrupt requests.
    D7 has highest priority.
    """
    print("8:3 Priority Encoder for interrupt handling:")
    print()
    print("  8 interrupt lines (D0..D7), D7 = highest priority")
    print("  Outputs: A2,A1,A0 (3-bit interrupt number), V (valid flag)")
    print()

    def priority_encoder(d):
        """8:3 priority encoder. d is a list of 8 bits (D0..D7)."""
        # Find highest-priority active input
        for i in range(7, -1, -1):
            if d[i] == 1:
                return (i >> 2) & 1, (i >> 1) & 1, i & 1, 1  # A2, A1, A0, V
        return 0, 0, 0, 0  # No active interrupt

    # Example scenarios
    scenarios = [
        ([0, 0, 0, 0, 0, 0, 0, 0], "No interrupts"),
        ([0, 0, 1, 0, 0, 0, 0, 0], "Only D2 active"),
        ([0, 0, 1, 0, 0, 1, 0, 0], "D2 and D5 active"),
        ([1, 1, 1, 1, 1, 1, 1, 1], "All active"),
        ([1, 0, 0, 0, 0, 0, 0, 1], "D0 and D7 active"),
    ]

    print("  D7 D6 D5 D4 D3 D2 D1 D0 | A2 A1 A0 V | Selected")
    print("  " + "-" * 55)
    for d, desc in scenarios:
        a2, a1, a0, v = priority_encoder(d)
        d_str = '  '.join(str(b) for b in reversed(d))
        selected = a2 * 4 + a1 * 2 + a0 if v else "-"
        print(f"   {d_str} |  {a2}  {a1}  {a0} {v} | D{selected} ({desc})")

    print()
    print("  Key property: Higher-numbered interrupts always take priority.")
    print("  If D5 and D2 are both active, the encoder outputs 101 (D5).")
    print("  The CPU services D5 first; D2 remains pending.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Half Adder vs Full Adder", exercise_1),
        ("Exercise 2: 4-bit Ripple Carry Addition", exercise_2),
        ("Exercise 3: 8:1 MUX Select Lines", exercise_3),
        ("Exercise 4: Full Adder with NAND Gates", exercise_4),
        ("Exercise 5: Function with 4:1 MUX", exercise_5),
        ("Exercise 6: Function with 3:8 Decoder", exercise_6),
        ("Exercise 7: MUX Circuit Analysis", exercise_7),
        ("Exercise 8: Ripple Carry Adder Delay", exercise_8),
        ("Exercise 9: 4-bit Adder/Subtractor", exercise_9),
        ("Exercise 10: Priority Encoder", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

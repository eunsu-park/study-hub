"""
Exercises for Lesson 06: Sequential Logic
Topic: Computer_Architecture

Solutions to practice problems covering latches vs flip-flops,
D/JK/T flip-flops, registers, counters, state machines, and timing analysis.
"""


def exercise_1():
    """Explain the difference between a latch and a flip-flop."""
    print("Latch vs Flip-Flop:")
    print()
    comparison = [
        ("Trigger type",    "Level-triggered",              "Edge-triggered"),
        ("Transparency",    "Transparent when Enable=1",    "Samples only at clock edge"),
        ("Timing",          "Output follows input while EN","Output changes once per clock"),
        ("Sensitivity",     "Sensitive to glitches on D",   "Immune to glitches away from edge"),
        ("Building block",  "Basic storage element",        "Built from 2 latches (master-slave)"),
        ("Usage",           "Simple storage, async designs","Synchronous sequential circuits"),
        ("Example",         "SR latch, D latch",            "D flip-flop, JK flip-flop"),
    ]

    print(f"  {'Feature':<22s} {'Latch':<32s} {'Flip-Flop':<35s}")
    print(f"  {'-'*22} {'-'*32} {'-'*35}")
    for feature, latch, ff in comparison:
        print(f"  {feature:<22s} {latch:<32s} {ff:<35s}")

    print()
    print("  Key insight: Flip-flops are preferred in synchronous designs because")
    print("  they only change state at the clock edge, making timing predictable.")
    print("  Latches are used in specific cases (e.g., transparent latch pipelines,")
    print("  low-power designs).")


def exercise_2():
    """D flip-flop characteristic equation and operation."""
    print("D Flip-Flop Characteristic Equation:")
    print()
    print("  Q(t+1) = D")
    print()
    print("  At each active clock edge (rising or falling):")
    print("    - The value on input D is captured and stored in Q")
    print("    - Q holds this value until the next active clock edge")
    print("    - Q' is always the complement of Q")
    print()

    # Simulate D flip-flop
    print("  Simulation (rising-edge triggered):")
    print()
    d_sequence =   [0, 1, 1, 0, 1, 0, 0, 1]
    q = 0  # Initial state

    print(f"  {'Clock':>6s} {'D':>3s} {'Q(before)':>10s} {'Q(after)':>10s}")
    print(f"  {'-'*6} {'-'*3} {'-'*10} {'-'*10}")
    for i, d in enumerate(d_sequence):
        q_before = q
        q = d  # D flip-flop: Q(t+1) = D
        print(f"  {i+1:>6d} {d:>3d} {q_before:>10d} {q:>10d}")

    print()
    print("  Applications: Data storage, pipeline registers, shift registers")
    print("  The D flip-flop is the most commonly used flip-flop in digital design.")


def exercise_3():
    """JK flip-flop with J=K=1 (toggle mode)."""
    print("JK Flip-Flop Toggle Mode (J=K=1):")
    print()

    # JK flip-flop characteristic equation: Q(t+1) = J·Q' + K'·Q
    print("  Characteristic equation: Q(t+1) = J·Q'(t) + K'·Q(t)")
    print()
    print("  Full truth table:")
    print(f"  {'J':>3s} {'K':>3s} {'Q(t)':>5s} | {'Q(t+1)':>7s} {'Mode':>10s}")
    print(f"  {'-'*3} {'-'*3} {'-'*5} | {'-'*7} {'-'*10}")

    modes = {(0, 0): "Hold", (0, 1): "Reset", (1, 0): "Set", (1, 1): "Toggle"}
    for j, k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for q in [0, 1]:
            q_next = (j & (1 - q)) | ((1 - k) & q)
            mode = modes[(j, k)]
            print(f"  {j:>3d} {k:>3d} {q:>5d} | {q_next:>7d} {mode:>10s}")

    print()
    print("  When J=K=1 (Toggle mode):")
    print("    Q(t+1) = 1·Q' + 0·Q = Q'")
    print("    The output toggles at every clock edge: 0→1→0→1→...")
    print()

    # Simulate toggle mode
    q = 0
    print("  Toggle simulation (J=K=1, starting Q=0):")
    states = [q]
    for i in range(8):
        q = 1 - q  # Toggle
        states.append(q)
    print(f"  Clock edges: {list(range(9))}")
    print(f"  Q values:    {states}")
    print("  This creates a frequency divider (output frequency = clock/2)")


def exercise_4():
    """
    Analyze D flip-flop with Q' fed back to D input.
    D = Q' → toggles every clock, equivalent to T flip-flop.
    """
    print("Circuit Analysis: D flip-flop with D connected to Q'")
    print()
    print("  ┌───────────────┐")
    print("  │  D ──┤ D   Q ├──┬── Q")
    print("  │      │       │  │")
    print("  │ CLK ─┤ >  Q' ├──┼── Q'")
    print("  │      └───────┘  │")
    print("  │         ↑       │")
    print("  └─────────┘  (D = Q')")
    print()

    print("  Analysis:")
    print("    Since D = Q', at every clock edge: Q(t+1) = D = Q'(t)")
    print("    This is exactly a T flip-flop with T=1 (always toggle)")
    print()

    # State transition table
    print("  State Transition Table:")
    print(f"  {'Q(t)':>5s} | {'D = Q\\'(t)':>10s} | {'Q(t+1)':>7s}")
    print(f"  {'-'*5} | {'-'*10} | {'-'*7}")
    for q in [0, 1]:
        d = 1 - q
        q_next = d
        print(f"  {q:>5d} | {d:>10d} | {q_next:>7d}")

    print()
    # Simulate
    q = 0
    print(f"  Timing simulation (starting Q=0):")
    for clk in range(8):
        q_next = 1 - q
        print(f"    Clock {clk}: Q={q} → Q'={1-q} → next Q={q_next}")
        q = q_next
    print()
    print("  Result: Output toggles every clock edge → frequency divider by 2")


def exercise_5():
    """
    Timing diagram for a 4-bit ripple counter. Initial state 0000.
    """
    print("4-bit Ripple Counter Timing Diagram:")
    print("  (Asynchronous counter: each stage triggers the next)")
    print()

    # Simulate 16 clock cycles
    count = 0
    states = []
    for clk in range(17):  # 0 to 16
        q3 = (count >> 3) & 1
        q2 = (count >> 2) & 1
        q1 = (count >> 1) & 1
        q0 = count & 1
        states.append((q3, q2, q1, q0))
        count = (count + 1) % 16

    # Print as table
    print(f"  {'CLK':>4s}  Q3 Q2 Q1 Q0  Decimal")
    print(f"  {'-'*4}  {'-'*2} {'-'*2} {'-'*2} {'-'*2}  {'-'*7}")
    for clk, (q3, q2, q1, q0) in enumerate(states):
        val = q3 * 8 + q2 * 4 + q1 * 2 + q0
        print(f"  {clk:>4d}   {q3}  {q2}  {q1}  {q0}    {val:>2d}")

    print()
    print("  Key observations:")
    print("    - Q0 toggles every clock edge (CLK/2)")
    print("    - Q1 toggles every time Q0 falls (CLK/4)")
    print("    - Q2 toggles every time Q1 falls (CLK/8)")
    print("    - Q3 toggles every time Q2 falls (CLK/16)")
    print("    - Counter wraps: 15 (1111) → 0 (0000)")
    print()
    print("  Disadvantage: Ripple delay accumulates through stages.")
    print("  Total delay = n × t_pd (vs synchronous counter: 1 × t_pd)")


def exercise_6():
    """
    4-bit right shift register operation.
    Initial: 0000, serial input sequence: 1, 0, 1, 1
    """
    print("4-bit Right Shift Register (serial input):")
    print()
    print("  Serial In → [Q3][Q2][Q1][Q0] → Serial Out")
    print("  Each clock: bits shift right, new bit enters at Q3")
    print()

    register = [0, 0, 0, 0]  # Q3, Q2, Q1, Q0
    inputs = [1, 0, 1, 1]

    print(f"  {'Clock':>6s}  {'Input':>6s}  Q3 Q2 Q1 Q0  {'(Hex)':>6s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*2} {'-'*2} {'-'*2} {'-'*2}  {'-'*6}")

    # Initial state
    val = register[0] * 8 + register[1] * 4 + register[2] * 2 + register[3]
    print(f"  {'Init':>6s}  {'--':>6s}   {register[0]}  {register[1]}  {register[2]}  {register[3]}   0x{val:01X}")

    for i, serial_in in enumerate(inputs):
        # Shift right: Q0 lost, Q1→Q0, Q2→Q1, Q3→Q2, serial_in→Q3
        serial_out = register[3]  # Q0 is shifted out
        register = [serial_in, register[0], register[1], register[2]]
        val = register[0] * 8 + register[1] * 4 + register[2] * 2 + register[3]
        print(f"  {i+1:>6d}  {serial_in:>6d}   {register[0]}  {register[1]}  {register[2]}  {register[3]}   0x{val:01X}")

    print()
    print("  After 4 shifts with input [1,0,1,1]:")
    print(f"  Register = {register[0]}{register[1]}{register[2]}{register[3]} = 1101")
    print()
    print("  Applications: Serial-to-parallel conversion, data buffering,")
    print("  serial communication (UART), LFSR (pseudo-random number generation)")


def exercise_7():
    """
    Implement T flip-flop using D flip-flop.
    D = T XOR Q: T=0 → hold, T=1 → toggle.
    """
    print("T Flip-Flop from D Flip-Flop:")
    print()
    print("  T flip-flop: Q(t+1) = T XOR Q(t)")
    print("    T=0: Q(t+1) = Q(t)  [Hold]")
    print("    T=1: Q(t+1) = Q'(t) [Toggle]")
    print()
    print("  Implementation: Connect D = T XOR Q")
    print("    D = T ⊕ Q")
    print("    At clock edge: Q(t+1) = D = T ⊕ Q(t)")
    print()

    # Verify equivalence
    print("  Verification:")
    print(f"  {'T':>3s} {'Q(t)':>5s} | {'D=T⊕Q':>7s} {'Q(t+1)':>7s} {'Mode':>8s}")
    print(f"  {'-'*3} {'-'*5} | {'-'*7} {'-'*7} {'-'*8}")
    for t in [0, 1]:
        for q in [0, 1]:
            d = t ^ q
            q_next = d  # D flip-flop: Q(t+1) = D
            mode = "Hold" if t == 0 else "Toggle"
            print(f"  {t:>3d} {q:>5d} | {d:>7d} {q_next:>7d} {mode:>8s}")

    print()
    # Simulate
    t_sequence = [1, 1, 0, 1, 0, 0, 1, 1]
    q = 0
    print("  Simulation:")
    print(f"  T: {t_sequence}")
    q_vals = [q]
    for t in t_sequence:
        q = t ^ q
        q_vals.append(q)
    print(f"  Q: {q_vals}")


def exercise_8():
    """
    Design a MOD-5 synchronous counter (states: 0,1,2,3,4,0,...).
    """
    print("MOD-5 Synchronous Counter Design:")
    print("  States: 000 → 001 → 010 → 011 → 100 → 000 (repeat)")
    print("  3 flip-flops needed (Q2, Q1, Q0)")
    print()

    # State transition table
    states = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)]
    next_states = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 0)]
    # Don't care states: 101, 110, 111

    print("  State Transition Table:")
    print(f"  {'Q2':>3s} {'Q1':>3s} {'Q0':>3s} | {'Q2+':>4s} {'Q1+':>4s} {'Q0+':>4s} | State")
    print(f"  {'-'*3} {'-'*3} {'-'*3} | {'-'*4} {'-'*4} {'-'*4} | {'-'*5}")
    for i, ((q2, q1, q0), (n2, n1, n0)) in enumerate(zip(states, next_states)):
        print(f"  {q2:>3d} {q1:>3d} {q0:>3d} | {n2:>4d} {n1:>4d} {n0:>4d} |   {i}")

    print()
    print("  Using JK flip-flops:")
    print("    J0 = 1, K0 = 1             (Q0 always toggles)")
    print("    J1 = Q0 · Q2', K1 = Q0     (Q1 sets when Q0=1 and Q2=0)")
    print("    J2 = Q0 · Q1,  K2 = Q0     (Q2 sets when Q0=Q1=1)")
    print()

    # Simulate
    q2, q1, q0 = 0, 0, 0
    print("  Simulation (10 clock cycles):")
    for clk in range(10):
        count = q2 * 4 + q1 * 2 + q0
        print(f"    CLK {clk}: Q2Q1Q0 = {q2}{q1}{q0} (count = {count})")
        # Next state logic
        count_next = (count + 1) % 5
        q2 = (count_next >> 2) & 1
        q1 = (count_next >> 1) & 1
        q0 = count_next & 1


def exercise_9():
    """
    Design a 4-bit bidirectional shift register (Left, Right, Hold modes).
    """
    print("4-bit Bidirectional Shift Register:")
    print()
    print("  Control: Mode (2 bits)")
    print("    00 = Hold (no change)")
    print("    01 = Shift Right (serial input at MSB)")
    print("    10 = Shift Left (serial input at LSB)")
    print("    11 = Parallel Load")
    print()
    print("  Implementation: 4:1 MUX before each D flip-flop")
    print("    For bit i:")
    print("      Mode 00 (Hold):  MUX selects Q[i] (current value)")
    print("      Mode 01 (Right): MUX selects Q[i+1] (left neighbor)")
    print("      Mode 10 (Left):  MUX selects Q[i-1] (right neighbor)")
    print("      Mode 11 (Load):  MUX selects D[i] (parallel input)")
    print()

    class BiDirShiftRegister:
        def __init__(self, bits=4):
            self.bits = bits
            self.q = [0] * bits  # Q[0]=MSB, Q[3]=LSB convention

        def clock(self, mode, serial_in=0, parallel_data=None):
            """Apply one clock cycle."""
            if mode == 0b00:  # Hold
                pass
            elif mode == 0b01:  # Shift Right
                self.q = [serial_in] + self.q[:-1]
            elif mode == 0b10:  # Shift Left
                self.q = self.q[1:] + [serial_in]
            elif mode == 0b11:  # Parallel Load
                if parallel_data:
                    self.q = list(parallel_data)

        def __str__(self):
            return ''.join(str(b) for b in self.q)

    reg = BiDirShiftRegister()
    print("  Simulation:")

    operations = [
        (0b11, 0, [1, 0, 1, 1], "Parallel Load 1011"),
        (0b01, 0, None, "Shift Right (serial=0)"),
        (0b01, 1, None, "Shift Right (serial=1)"),
        (0b10, 0, None, "Shift Left (serial=0)"),
        (0b10, 1, None, "Shift Left (serial=1)"),
        (0b00, 0, None, "Hold"),
    ]

    print(f"  {'CLK':>4s}  {'Mode':>12s}  {'Q3Q2Q1Q0':>8s}  Operation")
    print(f"  {'-'*4}  {'-'*12}  {'-'*8}  {'-'*25}")
    print(f"  {'Init':>4s}  {'--':>12s}  {str(reg):>8s}  Initial state")

    for i, (mode, sin, pdata, desc) in enumerate(operations):
        reg.clock(mode, sin, pdata)
        mode_str = ["Hold", "Right", "Left", "Load"][mode]
        print(f"  {i+1:>4d}  {mode_str:>12s}  {str(reg):>8s}  {desc}")


def exercise_10():
    """
    Calculate maximum clock frequency.
    t_pd = 5ns, t_comb = 15ns, t_setup = 3ns, t_hold = 2ns
    """
    t_pd = 5      # Flip-flop propagation delay (ns)
    t_comb = 15   # Combinational logic delay (ns)
    t_setup = 3   # Setup time (ns)
    t_hold = 2    # Hold time (ns)

    print("Maximum Clock Frequency Calculation:")
    print()
    print("  Timing parameters:")
    print(f"    Flip-flop propagation delay (t_pd):  {t_pd} ns")
    print(f"    Combinational circuit delay (t_comb): {t_comb} ns")
    print(f"    Setup time (t_su):                    {t_setup} ns")
    print(f"    Hold time (t_hold):                   {t_hold} ns")
    print()

    # Minimum clock period: the signal must propagate through the entire path
    # and arrive at the next flip-flop before its setup time deadline
    # T_min = t_pd + t_comb + t_setup
    t_min = t_pd + t_comb + t_setup
    f_max = 1 / (t_min * 1e-9)  # Convert to Hz

    print("  Critical path: FF output → Combinational logic → FF input")
    print(f"  T_min = t_pd + t_comb + t_su = {t_pd} + {t_comb} + {t_setup} = {t_min} ns")
    print(f"  f_max = 1 / T_min = 1 / {t_min}ns = {f_max/1e6:.1f} MHz")
    print()

    # Hold time check
    # Data must be stable for t_hold after clock edge
    # Hold margin = t_pd - t_hold (must be >= 0)
    hold_margin = t_pd - t_hold
    print(f"  Hold time check:")
    print(f"    Hold margin = t_pd - t_hold = {t_pd} - {t_hold} = {hold_margin} ns")
    if hold_margin >= 0:
        print(f"    Hold time is satisfied (margin = {hold_margin} ns)")
    else:
        print(f"    WARNING: Hold time violation! Need buffer delay.")

    print()
    print("  Clock skew effect:")
    print("    If clock arrives at different FFs at different times (skew),")
    print("    T_min = t_pd + t_comb + t_su + t_skew")
    t_skew = 2  # Example
    t_min_skew = t_pd + t_comb + t_setup + t_skew
    f_max_skew = 1 / (t_min_skew * 1e-9)
    print(f"    With t_skew = {t_skew}ns: T_min = {t_min_skew}ns, f_max = {f_max_skew/1e6:.1f} MHz")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Latch vs Flip-Flop", exercise_1),
        ("Exercise 2: D Flip-Flop Equation", exercise_2),
        ("Exercise 3: JK Flip-Flop Toggle", exercise_3),
        ("Exercise 4: D FF with Q' Feedback", exercise_4),
        ("Exercise 5: 4-bit Ripple Counter", exercise_5),
        ("Exercise 6: Shift Register Operation", exercise_6),
        ("Exercise 7: T FF from D FF", exercise_7),
        ("Exercise 8: MOD-5 Counter Design", exercise_8),
        ("Exercise 9: Bidirectional Shift Register", exercise_9),
        ("Exercise 10: Maximum Clock Frequency", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

"""
Flip-Flops and Registers

Demonstrates:
- SR (Set-Reset) latch
- D (Data) flip-flop
- JK flip-flop
- Register file (collection of D flip-flops)
- Shift register

Theory:
- Latches/flip-flops are bistable elements that store 1 bit.
- SR latch: Set (Q=1) / Reset (Q=0), invalid when S=R=1.
- D flip-flop: captures input D on clock edge. Most common in practice.
- JK flip-flop: like SR but J=K=1 toggles instead of being invalid.
- Register: group of flip-flops storing a multi-bit word.

Adapted from Computer Architecture Lesson 06.
"""


# ── SR Latch ────────────────────────────────────────────────────────────

class SRLatch:
    """SR (Set-Reset) latch."""

    def __init__(self):
        self.q = 0

    def update(self, s: int, r: int) -> int:
        # S=R=1 is forbidden because both NOR gates in a real SR latch
        # would output 0, violating the Q/~Q complementary invariant.
        # When both inputs deassert simultaneously, the output is
        # indeterminate (race condition), so hardware specs ban this.
        if s == 1 and r == 1:
            raise ValueError("Invalid: S=R=1")
        elif s == 1:
            self.q = 1
        elif r == 1:
            self.q = 0
        # else: hold
        return self.q


# ── D Flip-Flop ─────────────────────────────────────────────────────────

class DFlipFlop:
    """Positive-edge-triggered D flip-flop."""

    def __init__(self):
        self.q = 0
        self._prev_clk = 0

    def update(self, d: int, clk: int) -> int:
        # Edge-triggered (not level-sensitive): the flip-flop samples D
        # only on the 0-to-1 clock transition.  This prevents D changes
        # during the high phase from propagating through, which is
        # essential for synchronous circuit timing guarantees.
        if clk == 1 and self._prev_clk == 0:
            self.q = d
        self._prev_clk = clk
        return self.q


# ── JK Flip-Flop ────────────────────────────────────────────────────────

class JKFlipFlop:
    """Positive-edge-triggered JK flip-flop."""

    def __init__(self):
        self.q = 0
        self._prev_clk = 0

    def update(self, j: int, k: int, clk: int) -> int:
        if clk == 1 and self._prev_clk == 0:
            if j == 0 and k == 0:
                pass  # hold
            elif j == 0 and k == 1:
                self.q = 0  # reset
            elif j == 1 and k == 0:
                self.q = 1  # set
            else:  # j == 1, k == 1
                # Toggle mode is what makes JK superior to SR: the
                # forbidden S=R=1 state becomes a useful operation,
                # enabling frequency dividers and counters.
                self.q = 1 - self.q
        self._prev_clk = clk
        return self.q


def demo_flip_flops():
    """Demonstrate flip-flop behavior."""
    print("=" * 60)
    print("FLIP-FLOP DEMONSTRATIONS")
    print("=" * 60)

    # SR Latch
    print("\n  SR Latch:")
    print("    S  R | Q  Action")
    print("    ---------------")
    sr = SRLatch()
    operations = [
        (0, 0, "Hold"),
        (1, 0, "Set"),
        (0, 0, "Hold"),
        (0, 1, "Reset"),
        (0, 0, "Hold"),
    ]
    for s, r, action in operations:
        q = sr.update(s, r)
        print(f"    {s}  {r} | {q}  {action}")

    # D Flip-Flop
    print("\n  D Flip-Flop (edge-triggered):")
    print("    CLK  D | Q  Notes")
    print("    -----------------")
    dff = DFlipFlop()
    # Simulate clock cycles
    signals = [
        (0, 1, "D=1, clock low — no change"),
        (1, 1, "Rising edge — capture D=1"),
        (0, 1, "Clock low — hold"),
        (0, 0, "D changes, clock low — no change"),
        (1, 0, "Rising edge — capture D=0"),
        (1, 1, "D changes, clock high — no change (not edge)"),
        (0, 1, "Clock low — hold"),
        (1, 1, "Rising edge — capture D=1"),
    ]
    for clk, d, note in signals:
        q = dff.update(d, clk)
        print(f"     {clk}   {d} | {q}  {note}")

    # JK Flip-Flop
    print("\n  JK Flip-Flop:")
    print("    CLK  J  K | Q  Action")
    print("    ----------------------")
    jk = JKFlipFlop()
    jk_ops = [
        (1, 0, 0, "Hold (Q=0)"),
        (0, 0, 0, ""),
        (1, 1, 0, "Set"),
        (0, 0, 0, ""),
        (1, 0, 0, "Hold (Q=1)"),
        (0, 0, 0, ""),
        (1, 0, 1, "Reset"),
        (0, 0, 0, ""),
        (1, 1, 1, "Toggle (0→1)"),
        (0, 0, 0, ""),
        (1, 1, 1, "Toggle (1→0)"),
    ]
    for clk, j, k, action in jk_ops:
        q = jk.update(j, k, clk)
        if action:
            print(f"     {clk}   {j}  {k} | {q}  {action}")


# ── Register File ───────────────────────────────────────────────────────

class RegisterFile:
    """Simple register file with N registers of W bits each."""

    def __init__(self, n_regs: int = 8, width: int = 8):
        self.n_regs = n_regs
        self.width = width
        self.regs = [0] * n_regs

    def read(self, reg_num: int) -> int:
        # R0 is hardwired to zero (MIPS/RISC-V convention).  This gives
        # the ISA a free constant source and simplifies instruction
        # encoding — no need for a separate "load immediate 0" form.
        if reg_num == 0:
            return 0
        return self.regs[reg_num]

    def write(self, reg_num: int, value: int) -> None:
        if reg_num == 0:
            return  # Writes to R0 are silently discarded to preserve the zero invariant
        mask = (1 << self.width) - 1
        self.regs[reg_num] = value & mask

    def display(self) -> None:
        print("  Register File:")
        for i in range(self.n_regs):
            val = self.regs[i]
            note = " (hardwired 0)" if i == 0 else ""
            print(f"    R{i}: {val:>5} ({val:08b}){note}")


def demo_register_file():
    """Demonstrate register file operations."""
    print("\n" + "=" * 60)
    print("REGISTER FILE")
    print("=" * 60)

    rf = RegisterFile(n_regs=8, width=8)

    operations = [
        ("write", 1, 42),
        ("write", 2, 100),
        ("write", 3, 255),
        ("write", 0, 99),   # should be ignored
        ("read", 1, None),
        ("read", 0, None),
    ]

    print("\n  Operations:")
    for op, reg, val in operations:
        if op == "write":
            rf.write(reg, val)
            actual = rf.read(reg)
            print(f"    Write R{reg} ← {val:>5} (actual: {actual})")
        else:
            result = rf.read(reg)
            print(f"    Read  R{reg} → {result:>5}")

    print()
    rf.display()


# ── Shift Register ──────────────────────────────────────────────────────

class ShiftRegister:
    """N-bit shift register (SISO - Serial In, Serial Out)."""

    def __init__(self, width: int = 8):
        self.width = width
        self.bits = [0] * width

    def shift_left(self, serial_in: int = 0) -> int:
        """Shift left, insert serial_in at LSB. Returns shifted-out MSB."""
        out = self.bits[-1]
        self.bits = [serial_in] + self.bits[:-1]
        return out

    def shift_right(self, serial_in: int = 0) -> int:
        """Shift right, insert serial_in at MSB. Returns shifted-out LSB."""
        out = self.bits[0]
        self.bits = self.bits[1:] + [serial_in]
        return out

    def load(self, value: int) -> None:
        self.bits = [(value >> i) & 1 for i in range(self.width)]

    def value(self) -> int:
        return sum(b << i for i, b in enumerate(self.bits))

    def __repr__(self) -> str:
        return "".join(str(b) for b in reversed(self.bits))


def demo_shift_register():
    """Demonstrate shift register."""
    print("\n" + "=" * 60)
    print("SHIFT REGISTER (8-bit)")
    print("=" * 60)

    sr = ShiftRegister(8)
    sr.load(0b10110100)
    print(f"\n  Initial: {sr} ({sr.value()})")

    print("\n  Left shifts (serial_in=0):")
    for i in range(4):
        out = sr.shift_left(0)
        print(f"    Step {i+1}: {sr} ({sr.value():>3}) out={out}")

    sr.load(0b10110100)
    print(f"\n  Reset: {sr} ({sr.value()})")
    print("\n  Right shifts (serial_in=0):")
    for i in range(4):
        out = sr.shift_right(0)
        print(f"    Step {i+1}: {sr} ({sr.value():>3}) out={out}")

    # Serial data loading
    print("\n  Serial load of 0xA5 (10100101):")
    sr2 = ShiftRegister(8)
    data = 0b10100101
    for i in range(7, -1, -1):
        bit = (data >> i) & 1
        sr2.shift_left(bit)
        print(f"    In={bit}: {sr2}")
    print(f"  Result: {sr2.value()} (0x{sr2.value():02X})")


if __name__ == "__main__":
    demo_flip_flops()
    demo_register_file()
    demo_shift_register()

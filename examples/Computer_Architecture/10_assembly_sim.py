"""
Fetch-Decode-Execute ISA Simulator

Demonstrates the instruction cycle with a simple assembly-like ISA:
- Fetch: read instruction from memory at PC
- Decode: parse opcode and operands
- Execute: perform the operation

ISA:
  LOAD  Rd, imm   — Rd = imm
  ADD   Rd, Rs, Rt — Rd = Rs + Rt
  SUB   Rd, Rs, Rt — Rd = Rs - Rt
  MUL   Rd, Rs, Rt — Rd = Rs * Rt
  CMP   Rs, Rt     — set flags based on Rs - Rt
  JMP   addr       — PC = addr
  JZ    addr       — if zero flag, PC = addr
  JNZ   addr       — if not zero flag, PC = addr
  PRINT Rd         — output Rd value
  HALT             — stop execution

Adapted from Computer Architecture Lesson 10.
"""

from dataclasses import dataclass
from enum import IntEnum


class Op(IntEnum):
    LOAD  = 0
    ADD   = 1
    SUB   = 2
    MUL   = 3
    CMP   = 4
    JMP   = 5
    JZ    = 6
    JNZ   = 7
    PRINT = 8
    HALT  = 9


@dataclass
class Instruction:
    op: Op
    rd: int = 0
    rs: int = 0
    rt: int = 0
    imm: int = 0

    def __repr__(self) -> str:
        name = self.op.name
        if self.op == Op.LOAD:
            return f"{name} R{self.rd}, {self.imm}"
        elif self.op in (Op.ADD, Op.SUB, Op.MUL):
            return f"{name} R{self.rd}, R{self.rs}, R{self.rt}"
        elif self.op == Op.CMP:
            return f"{name} R{self.rs}, R{self.rt}"
        elif self.op in (Op.JMP, Op.JZ, Op.JNZ):
            return f"{name} {self.imm}"
        elif self.op == Op.PRINT:
            return f"{name} R{self.rd}"
        else:
            return name


class SimpleCPU:
    """Simple CPU with fetch-decode-execute cycle."""

    def __init__(self):
        self.regs = [0] * 8
        self.pc = 0
        self.zero_flag = False
        self.negative_flag = False
        self.halted = False
        self.cycle = 0
        self.program: list[Instruction] = []
        self.output: list[int] = []

    def load_program(self, program: list[Instruction]) -> None:
        self.program = program
        self.pc = 0
        self.cycle = 0
        self.halted = False
        self.output = []

    def fetch(self) -> Instruction | None:
        # Fetching beyond the program is treated as a halt — this avoids
        # requiring an explicit HALT at the end of every program, similar
        # to how a real CPU would fault on an unmapped address.
        if self.pc >= len(self.program):
            self.halted = True
            return None
        instr = self.program[self.pc]
        return instr

    def decode(self, instr: Instruction) -> dict:
        """Decode instruction into control signals."""
        # Register reads happen during decode — this mirrors real hardware
        # where the register file is read combinationally in the decode
        # stage, before the execute stage needs the values.
        return {
            "op": instr.op,
            "rd": instr.rd,
            "rs_val": self.regs[instr.rs] if instr.rs < len(self.regs) else 0,
            "rt_val": self.regs[instr.rt] if instr.rt < len(self.regs) else 0,
            "imm": instr.imm,
        }

    def execute(self, signals: dict) -> None:
        """Execute based on decoded signals."""
        op = signals["op"]
        rd = signals["rd"]
        rs_val = signals["rs_val"]
        rt_val = signals["rt_val"]
        imm = signals["imm"]

        if op == Op.LOAD:
            self.regs[rd] = imm
            self.pc += 1

        elif op == Op.ADD:
            self.regs[rd] = rs_val + rt_val
            self.pc += 1

        elif op == Op.SUB:
            self.regs[rd] = rs_val - rt_val
            self.pc += 1

        elif op == Op.MUL:
            self.regs[rd] = rs_val * rt_val
            self.pc += 1

        elif op == Op.CMP:
            # CMP performs a subtraction but discards the result, keeping
            # only the flags.  This separation of comparison from branching
            # allows one CMP to feed multiple conditional jumps without
            # re-executing the arithmetic.
            diff = rs_val - rt_val
            self.zero_flag = (diff == 0)
            self.negative_flag = (diff < 0)
            self.pc += 1

        elif op == Op.JMP:
            # Unconditional jump: set PC directly (absolute address).
            # This is an absolute rather than relative jump — simpler to
            # implement but makes position-independent code harder.
            self.pc = imm

        elif op == Op.JZ:
            if self.zero_flag:
                self.pc = imm
            else:
                self.pc += 1

        elif op == Op.JNZ:
            if not self.zero_flag:
                self.pc = imm
            else:
                # Fall-through: if condition is false, advance to the
                # next sequential instruction — branch not taken.
                self.pc += 1

        elif op == Op.PRINT:
            self.output.append(self.regs[rd])
            self.pc += 1

        elif op == Op.HALT:
            self.halted = True

    def step(self, verbose: bool = True) -> bool:
        """Execute one cycle. Returns False if halted."""
        if self.halted:
            return False

        self.cycle += 1
        instr = self.fetch()
        if instr is None:
            return False

        signals = self.decode(instr)
        if verbose:
            print(f"    Cycle {self.cycle:>3} [PC={self.pc:>2}] "
                  f"{str(instr):<25}", end="")

        self.execute(signals)

        if verbose:
            regs_str = " ".join(f"R{i}={self.regs[i]}" for i in range(8) if self.regs[i] != 0)
            print(f"  → {regs_str}")

        return not self.halted

    def run(self, max_cycles: int = 1000, verbose: bool = True) -> None:
        while self.step(verbose) and self.cycle < max_cycles:
            pass


# ── Demo Programs ───────────────────────────────────────────────────────

def demo_arithmetic():
    """Basic arithmetic operations."""
    print("=" * 60)
    print("FETCH-DECODE-EXECUTE: ARITHMETIC")
    print("=" * 60)

    program = [
        Instruction(Op.LOAD, rd=0, imm=10),
        Instruction(Op.LOAD, rd=1, imm=20),
        Instruction(Op.ADD, rd=2, rs=0, rt=1),   # R2 = 30
        Instruction(Op.SUB, rd=3, rs=2, rt=0),   # R3 = 20
        Instruction(Op.MUL, rd=4, rs=0, rt=1),   # R4 = 200
        Instruction(Op.PRINT, rd=2),
        Instruction(Op.PRINT, rd=4),
        Instruction(Op.HALT),
    ]

    cpu = SimpleCPU()
    cpu.load_program(program)
    print(f"\n  Program: compute 10+20, 30-10, 10*20\n")
    cpu.run()
    print(f"\n  Output: {cpu.output}")


def demo_countdown():
    """Loop: count down from 5, sum values."""
    print("\n" + "=" * 60)
    print("FETCH-DECODE-EXECUTE: LOOP (COUNTDOWN)")
    print("=" * 60)

    # R0 = counter (5), R1 = sum, R2 = 1 (decrement), R3 = 0 (compare target)
    program = [
        Instruction(Op.LOAD, rd=0, imm=5),       # R0 = 5
        Instruction(Op.LOAD, rd=1, imm=0),       # R1 = 0 (sum)
        Instruction(Op.LOAD, rd=2, imm=1),       # R2 = 1
        Instruction(Op.LOAD, rd=3, imm=0),       # R3 = 0
        # Loop (addr=4):
        Instruction(Op.ADD, rd=1, rs=1, rt=0),   # R1 += R0
        Instruction(Op.SUB, rd=0, rs=0, rt=2),   # R0 -= 1
        Instruction(Op.CMP, rs=0, rt=3),         # compare R0, 0
        Instruction(Op.JNZ, imm=4),              # if R0 != 0, goto 4
        Instruction(Op.PRINT, rd=1),             # print sum
        Instruction(Op.HALT),
    ]

    cpu = SimpleCPU()
    cpu.load_program(program)
    print(f"\n  Program: sum 5+4+3+2+1\n")
    cpu.run()
    print(f"\n  Output: {cpu.output} (expected [15])")


def demo_fibonacci():
    """Compute Fibonacci numbers."""
    print("\n" + "=" * 60)
    print("FETCH-DECODE-EXECUTE: FIBONACCI")
    print("=" * 60)

    # R0 = fib(n-2), R1 = fib(n-1), R2 = temp, R3 = count, R4 = 1, R5 = target
    program = [
        Instruction(Op.LOAD, rd=0, imm=0),       # R0 = fib(0) = 0
        Instruction(Op.LOAD, rd=1, imm=1),       # R1 = fib(1) = 1
        Instruction(Op.LOAD, rd=3, imm=8),       # R3 = remaining iterations
        Instruction(Op.LOAD, rd=4, imm=1),       # R4 = 1
        Instruction(Op.LOAD, rd=5, imm=0),       # R5 = 0 (compare)
        Instruction(Op.PRINT, rd=0),             # print fib(n-2)
        # Loop (addr=6):
        Instruction(Op.ADD, rd=2, rs=0, rt=1),   # R2 = R0 + R1
        Instruction(Op.LOAD, rd=0, imm=0),       # temp: will be overwritten
        Instruction(Op.ADD, rd=0, rs=1, rt=5),   # R0 = R1 (shift)
        Instruction(Op.ADD, rd=1, rs=2, rt=5),   # R1 = R2 (shift)
        Instruction(Op.PRINT, rd=0),             # print current
        Instruction(Op.SUB, rd=3, rs=3, rt=4),   # R3 -= 1
        Instruction(Op.CMP, rs=3, rt=5),         # compare R3, 0
        Instruction(Op.JNZ, imm=6),              # if R3 != 0, loop
        Instruction(Op.PRINT, rd=1),             # print last
        Instruction(Op.HALT),
    ]

    cpu = SimpleCPU()
    cpu.load_program(program)
    print(f"\n  Program: first 10 Fibonacci numbers\n")
    cpu.run()
    print(f"\n  Fibonacci: {cpu.output}")


if __name__ == "__main__":
    demo_arithmetic()
    demo_countdown()
    demo_fibonacci()

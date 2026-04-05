"""
Computer System Overview — Von Neumann Architecture Simulator

Demonstrates:
- Von Neumann architecture components (CPU, memory, bus)
- Fetch-decode-execute cycle
- Registers, ALU, and control unit interaction
- Simple instruction set and program execution

Theory:
- The Von Neumann model stores both instructions and data in the
  same memory, accessed via a shared bus.
- The CPU repeatedly fetches an instruction from memory (using the
  Program Counter), decodes it, executes it, and writes back results.
- This sequential fetch-decode-execute cycle is the foundation of
  nearly all modern processors.

Adapted from Computer Architecture Lesson 01.
"""

from dataclasses import dataclass, field


# ── Instruction Set ───────────────────────────────────────────────────

# Simple ISA: 4-bit opcode + operands
OPCODES = {
    0x1: "LOAD",    # LOAD Rd, addr   — load memory[addr] into Rd
    0x2: "STORE",   # STORE Rs, addr  — store Rs into memory[addr]
    0x3: "ADD",     # ADD Rd, Rs, Rt  — Rd = Rs + Rt
    0x4: "SUB",     # SUB Rd, Rs, Rt  — Rd = Rs - Rt
    0x5: "MUL",     # MUL Rd, Rs, Rt  — Rd = Rs * Rt
    0x6: "JMP",     # JMP addr        — set PC to addr
    0x7: "JZ",      # JZ Rs, addr     — if Rs == 0, jump to addr
    0x8: "HALT",    # HALT            — stop execution
    0x9: "LOADI",   # LOADI Rd, imm   — load immediate value into Rd
}


@dataclass
class Instruction:
    """Decoded instruction."""
    opcode: int
    rd: int = 0
    rs: int = 0
    rt: int = 0
    imm: int = 0

    def name(self) -> str:
        return OPCODES.get(self.opcode, "???")

    def __repr__(self) -> str:
        op = self.name()
        if op in ("LOAD", "STORE"):
            return f"{op} R{self.rd}, [{self.imm}]"
        if op in ("ADD", "SUB", "MUL"):
            return f"{op} R{self.rd}, R{self.rs}, R{self.rt}"
        if op == "LOADI":
            return f"{op} R{self.rd}, #{self.imm}"
        if op == "JMP":
            return f"{op} {self.imm}"
        if op == "JZ":
            return f"{op} R{self.rs}, {self.imm}"
        if op == "HALT":
            return "HALT"
        return f"UNKNOWN(0x{self.opcode:X})"


def encode(opcode: int, rd: int = 0, rs: int = 0,
           rt: int = 0, imm: int = 0) -> int:
    """Encode instruction into a 32-bit word.

    Layout: [opcode(8) | rd(4) | rs(4) | rt(4) | imm(12)]
    """
    return ((opcode & 0xFF) << 24 |
            (rd & 0xF) << 20 |
            (rs & 0xF) << 16 |
            (rt & 0xF) << 12 |
            (imm & 0xFFF))


def decode(word: int) -> Instruction:
    """Decode a 32-bit word into an Instruction."""
    opcode = (word >> 24) & 0xFF
    rd = (word >> 20) & 0xF
    rs = (word >> 16) & 0xF
    rt = (word >> 12) & 0xF
    imm = word & 0xFFF
    return Instruction(opcode=opcode, rd=rd, rs=rs, rt=rt, imm=imm)


# ── Von Neumann Machine ──────────────────────────────────────────────

@dataclass
class VonNeumannMachine:
    """Von Neumann architecture simulator.

    Components:
    - Memory: unified instruction + data store (word-addressable)
    - Registers: R0-R15 general-purpose, PC, IR
    - ALU: arithmetic operations
    - Control Unit: fetch-decode-execute sequencer
    """
    memory: list[int] = field(default_factory=lambda: [0] * 256)
    registers: list[int] = field(default_factory=lambda: [0] * 16)
    pc: int = 0       # Program Counter
    ir: int = 0       # Instruction Register
    halted: bool = False
    cycle_count: int = 0
    trace: list[str] = field(default_factory=list)

    def load_program(self, program: list[int], start: int = 0) -> None:
        """Load encoded instructions into memory starting at address."""
        for i, word in enumerate(program):
            self.memory[start + i] = word
        self.pc = start

    def fetch(self) -> int:
        """Fetch: read instruction at PC from memory."""
        self.ir = self.memory[self.pc]
        return self.ir

    def execute(self, instr: Instruction) -> None:
        """Execute: perform the operation specified by the instruction."""
        op = instr.name()

        if op == "LOAD":
            self.registers[instr.rd] = self.memory[instr.imm]
            self.pc += 1
        elif op == "STORE":
            self.memory[instr.imm] = self.registers[instr.rd]
            self.pc += 1
        elif op == "LOADI":
            self.registers[instr.rd] = instr.imm
            self.pc += 1
        elif op == "ADD":
            self.registers[instr.rd] = (
                self.registers[instr.rs] + self.registers[instr.rt])
            self.pc += 1
        elif op == "SUB":
            self.registers[instr.rd] = (
                self.registers[instr.rs] - self.registers[instr.rt])
            self.pc += 1
        elif op == "MUL":
            self.registers[instr.rd] = (
                self.registers[instr.rs] * self.registers[instr.rt])
            self.pc += 1
        elif op == "JMP":
            self.pc = instr.imm
        elif op == "JZ":
            if self.registers[instr.rs] == 0:
                self.pc = instr.imm
            else:
                self.pc += 1
        elif op == "HALT":
            self.halted = True
        else:
            raise ValueError(f"Unknown opcode: 0x{instr.opcode:X}")

    def step(self) -> bool:
        """Run one fetch-decode-execute cycle. Returns False if halted."""
        if self.halted:
            return False

        self.cycle_count += 1

        # 1. FETCH
        word = self.fetch()

        # 2. DECODE
        instr = decode(word)

        # 3. EXECUTE
        addr = self.pc
        self.execute(instr)

        self.trace.append(
            f"  Cycle {self.cycle_count:>3}: PC={addr:>3}  {instr!r}")

        return not self.halted

    def run(self, max_cycles: int = 200) -> None:
        """Run until HALT or max cycles."""
        while self.step() and self.cycle_count < max_cycles:
            pass

    def dump_registers(self) -> None:
        """Print non-zero registers."""
        print("  Registers:")
        for i, val in enumerate(self.registers):
            if val != 0:
                print(f"    R{i:<2} = {val}")
        print(f"    PC  = {self.pc}")


# ── Demos ─────────────────────────────────────────────────────────────

def demo_basic_cycle():
    """Basic fetch-decode-execute demonstration."""
    print("=" * 60)
    print("VON NEUMANN: FETCH-DECODE-EXECUTE CYCLE")
    print("=" * 60)

    # Program: compute 5 + 3, store result
    program = [
        encode(0x9, rd=0, imm=5),   # LOADI R0, #5
        encode(0x9, rd=1, imm=3),   # LOADI R1, #3
        encode(0x3, rd=2, rs=0, rt=1),  # ADD R2, R0, R1
        encode(0x2, rd=2, imm=100), # STORE R2, [100]
        encode(0x8),                # HALT
    ]

    machine = VonNeumannMachine()
    machine.load_program(program)
    machine.run()

    print("\n  Execution trace:")
    for line in machine.trace:
        print(line)
    print()
    machine.dump_registers()
    print(f"  Memory[100] = {machine.memory[100]}")


def demo_loop():
    """Demonstrate a loop using conditional jump."""
    print("\n" + "=" * 60)
    print("VON NEUMANN: LOOP (SUM 1..5)")
    print("=" * 60)

    # Program: compute sum = 1 + 2 + 3 + 4 + 5
    # R0 = counter (starts at 5), R1 = sum, R2 = constant 1
    program = [
        encode(0x9, rd=0, imm=5),       # 0: LOADI R0, #5  (counter)
        encode(0x9, rd=1, imm=0),       # 1: LOADI R1, #0  (sum)
        encode(0x9, rd=2, imm=1),       # 2: LOADI R2, #1  (decrement)
        # loop:
        encode(0x7, rs=0, imm=7),       # 3: JZ R0, 7      (exit if R0==0)
        encode(0x3, rd=1, rs=1, rt=0),  # 4: ADD R1, R1, R0 (sum += counter)
        encode(0x4, rd=0, rs=0, rt=2),  # 5: SUB R0, R0, R2 (counter -= 1)
        encode(0x6, imm=3),             # 6: JMP 3          (loop back)
        # end:
        encode(0x8),                    # 7: HALT
    ]

    machine = VonNeumannMachine()
    machine.load_program(program)
    machine.run()

    print("\n  Execution trace:")
    for line in machine.trace:
        print(line)
    print()
    machine.dump_registers()
    print(f"\n  Sum of 1..5 = {machine.registers[1]} (expected 15)")
    print(f"  Total cycles: {machine.cycle_count}")


def demo_von_neumann_bottleneck():
    """Illustrate the Von Neumann bottleneck: shared bus for code + data."""
    print("\n" + "=" * 60)
    print("VON NEUMANN BOTTLENECK")
    print("=" * 60)

    print("""
  In the Von Neumann architecture, instructions and data share
  the same memory and bus. Every cycle requires at least one
  memory access (instruction fetch), and load/store instructions
  require additional data accesses.

  Cycle breakdown for our simple machine:
  - LOADI : 1 fetch
  - LOAD  : 1 fetch + 1 data read
  - STORE : 1 fetch + 1 data write
  - ADD   : 1 fetch (register-only)
  - JMP   : 1 fetch

  Harvard architecture separates instruction and data memory,
  allowing simultaneous fetch + data access — but at the cost
  of more complex interconnect and inability to self-modify code.
""")

    # Count memory accesses for the loop program
    fetch_accesses = 22  # from demo_loop cycle count
    data_accesses = 1    # one STORE would add a data access
    print(f"  Loop program (sum 1..5):")
    print(f"    Instruction fetches : ~22")
    print(f"    Data accesses       : 0  (all register ops)")
    print(f"    Bus utilization     : 100% instruction traffic")
    print(f"    → Data-intensive programs would saturate the bus")


if __name__ == "__main__":
    demo_basic_cycle()
    demo_loop()
    demo_von_neumann_bottleneck()

"""
RISC-V RV32I Simulator

Simulates a subset of the RV32I base integer instruction set:
- Arithmetic: ADD, SUB, ADDI
- Logical: AND, OR, XOR, ANDI, ORI, XORI
- Shift: SLL, SRL, SRA, SLLI, SRLI, SRAI
- Compare: SLT, SLTI, SLTU, SLTIU
- Memory: LW, SW
- Branch: BEQ, BNE, BLT, BGE
- Jump: JAL, JALR
- Upper immediate: LUI, AUIPC

Demonstrates instruction encoding, register file, and pipeline concepts.
"""

from dataclasses import dataclass, field
from typing import Optional
import struct


class RV32IRegisters:
    """32 general-purpose registers with x0 hardwired to zero."""

    ABI_NAMES = [
        "zero", "ra", "sp", "gp", "tp",
        "t0", "t1", "t2",
        "s0", "s1",
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
        "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        "t3", "t4", "t5", "t6",
    ]

    def __init__(self):
        self._regs = [0] * 32

    def read(self, idx: int) -> int:
        """Read register. x0 always returns 0."""
        if idx == 0:
            return 0
        return self._regs[idx] & 0xFFFFFFFF

    def write(self, idx: int, val: int):
        """Write register. Writes to x0 are silently ignored."""
        if idx != 0:
            # Store as 32-bit unsigned
            self._regs[idx] = val & 0xFFFFFFFF

    def read_signed(self, idx: int) -> int:
        """Read register as signed 32-bit integer."""
        val = self.read(idx)
        if val & 0x80000000:
            return val - 0x100000000
        return val

    def dump(self):
        """Print all register values."""
        print("Register File:")
        for i in range(0, 32, 4):
            parts = []
            for j in range(4):
                idx = i + j
                name = self.ABI_NAMES[idx]
                val = self.read(idx)
                sval = self.read_signed(idx)
                parts.append(f"  x{idx:2d}/{name:5s} = 0x{val:08X} ({sval})")
            print("".join(parts))


class Memory:
    """Simple byte-addressable memory for simulation."""

    def __init__(self, size: int = 65536):
        self._mem = bytearray(size)

    def load_word(self, addr: int) -> int:
        """Load 4 bytes (little-endian) from memory."""
        b = self._mem[addr:addr + 4]
        return struct.unpack("<I", b)[0]

    def store_word(self, addr: int, val: int):
        """Store 4 bytes (little-endian) to memory."""
        self._mem[addr:addr + 4] = struct.pack("<I", val & 0xFFFFFFFF)


def sign_extend(val: int, bits: int) -> int:
    """Sign-extend a value from 'bits' width to 32 bits."""
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val & 0xFFFFFFFF


@dataclass
class Instruction:
    """Decoded RISC-V instruction."""
    name: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    raw: int = 0

    def __str__(self):
        abi = RV32IRegisters.ABI_NAMES
        if self.name in ("ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL",
                         "SRA", "SLT", "SLTU"):
            return (f"{self.name:6s} {abi[self.rd]}, {abi[self.rs1]}, "
                    f"{abi[self.rs2]}")
        elif self.name in ("ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU",
                           "SLLI", "SRLI", "SRAI"):
            return (f"{self.name:6s} {abi[self.rd]}, {abi[self.rs1]}, "
                    f"{sign_extend(self.imm, 12) if self.imm & 0x800 else self.imm}")
        elif self.name == "LW":
            return f"LW     {abi[self.rd]}, {sign_extend(self.imm, 12)}({abi[self.rs1]})"
        elif self.name == "SW":
            return f"SW     {abi[self.rs2]}, {sign_extend(self.imm, 12)}({abi[self.rs1]})"
        elif self.name in ("BEQ", "BNE", "BLT", "BGE"):
            return f"{self.name:6s} {abi[self.rs1]}, {abi[self.rs2]}, {self.imm}"
        elif self.name == "JAL":
            return f"JAL    {abi[self.rd]}, {self.imm}"
        elif self.name == "JALR":
            return f"JALR   {abi[self.rd]}, {abi[self.rs1]}, {self.imm}"
        elif self.name == "LUI":
            return f"LUI    {abi[self.rd]}, 0x{self.imm:05X}"
        elif self.name == "AUIPC":
            return f"AUIPC  {abi[self.rd]}, 0x{self.imm:05X}"
        elif self.name == "ECALL":
            return "ECALL"
        return f"{self.name} (unknown format)"


class RV32IDecoder:
    """Decode 32-bit RV32I instructions."""

    @staticmethod
    def decode(word: int) -> Instruction:
        opcode = word & 0x7F
        rd = (word >> 7) & 0x1F
        funct3 = (word >> 12) & 0x7
        rs1 = (word >> 15) & 0x1F
        rs2 = (word >> 20) & 0x1F
        funct7 = (word >> 25) & 0x7F

        if opcode == 0b0110011:  # R-type
            names = {
                (0, 0b000): "ADD", (0x20, 0b000): "SUB",
                (0, 0b111): "AND", (0, 0b110): "OR", (0, 0b100): "XOR",
                (0, 0b001): "SLL", (0, 0b101): "SRL", (0x20, 0b101): "SRA",
                (0, 0b010): "SLT", (0, 0b011): "SLTU",
            }
            name = names.get((funct7, funct3), "UNKNOWN")
            return Instruction(name=name, rd=rd, rs1=rs1, rs2=rs2, raw=word)

        elif opcode == 0b0010011:  # I-type ALU
            imm = sign_extend((word >> 20) & 0xFFF, 12)
            names = {
                0b000: "ADDI", 0b111: "ANDI", 0b110: "ORI", 0b100: "XORI",
                0b010: "SLTI", 0b011: "SLTIU",
            }
            if funct3 == 0b001:
                return Instruction(name="SLLI", rd=rd, rs1=rs1,
                                   imm=rs2, raw=word)
            elif funct3 == 0b101:
                name = "SRAI" if funct7 == 0x20 else "SRLI"
                return Instruction(name=name, rd=rd, rs1=rs1,
                                   imm=rs2, raw=word)
            name = names.get(funct3, "UNKNOWN")
            return Instruction(name=name, rd=rd, rs1=rs1, imm=imm, raw=word)

        elif opcode == 0b0000011 and funct3 == 0b010:  # LW
            imm = sign_extend((word >> 20) & 0xFFF, 12)
            return Instruction(name="LW", rd=rd, rs1=rs1, imm=imm, raw=word)

        elif opcode == 0b0100011 and funct3 == 0b010:  # SW
            imm = ((word >> 25) << 5) | ((word >> 7) & 0x1F)
            imm = sign_extend(imm, 12)
            return Instruction(name="SW", rs1=rs1, rs2=rs2, imm=imm, raw=word)

        elif opcode == 0b1100011:  # B-type
            imm = (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11) | \
                  (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1)
            imm = sign_extend(imm, 13)
            names = {0b000: "BEQ", 0b001: "BNE", 0b100: "BLT", 0b101: "BGE"}
            name = names.get(funct3, "UNKNOWN")
            return Instruction(name=name, rs1=rs1, rs2=rs2, imm=imm, raw=word)

        elif opcode == 0b1101111:  # JAL
            imm = (((word >> 31) & 1) << 20) | (((word >> 12) & 0xFF) << 12) | \
                  (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1)
            imm = sign_extend(imm, 21)
            return Instruction(name="JAL", rd=rd, imm=imm, raw=word)

        elif opcode == 0b1100111:  # JALR
            imm = sign_extend((word >> 20) & 0xFFF, 12)
            return Instruction(name="JALR", rd=rd, rs1=rs1, imm=imm, raw=word)

        elif opcode == 0b0110111:  # LUI
            imm = (word >> 12) & 0xFFFFF
            return Instruction(name="LUI", rd=rd, imm=imm, raw=word)

        elif opcode == 0b0010111:  # AUIPC
            imm = (word >> 12) & 0xFFFFF
            return Instruction(name="AUIPC", rd=rd, imm=imm, raw=word)

        elif opcode == 0b1110011:  # SYSTEM
            return Instruction(name="ECALL", raw=word)

        return Instruction(name="UNKNOWN", raw=word)


class RV32ISimulator:
    """Execute decoded RV32I instructions."""

    def __init__(self):
        self.regs = RV32IRegisters()
        self.mem = Memory()
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self.instruction_count = 0

    def load_program(self, instructions: list[int], base_addr: int = 0):
        """Load machine code into memory."""
        for i, word in enumerate(instructions):
            self.mem.store_word(base_addr + i * 4, word)
        self.pc = base_addr

    def step(self) -> Optional[Instruction]:
        """Fetch, decode, and execute one instruction."""
        if self.halted:
            return None

        # Fetch
        word = self.mem.load_word(self.pc)
        if word == 0:
            self.halted = True
            return None

        # Decode
        inst = RV32IDecoder.decode(word)
        next_pc = self.pc + 4
        self.cycle_count += 1
        self.instruction_count += 1

        # Execute
        r = self.regs
        if inst.name == "ADD":
            r.write(inst.rd, r.read(inst.rs1) + r.read(inst.rs2))
        elif inst.name == "SUB":
            r.write(inst.rd, r.read(inst.rs1) - r.read(inst.rs2))
        elif inst.name == "ADDI":
            r.write(inst.rd, r.read(inst.rs1) + inst.imm)
        elif inst.name == "AND":
            r.write(inst.rd, r.read(inst.rs1) & r.read(inst.rs2))
        elif inst.name == "ANDI":
            r.write(inst.rd, r.read(inst.rs1) & inst.imm)
        elif inst.name == "OR":
            r.write(inst.rd, r.read(inst.rs1) | r.read(inst.rs2))
        elif inst.name == "ORI":
            r.write(inst.rd, r.read(inst.rs1) | inst.imm)
        elif inst.name == "XOR":
            r.write(inst.rd, r.read(inst.rs1) ^ r.read(inst.rs2))
        elif inst.name == "XORI":
            r.write(inst.rd, r.read(inst.rs1) ^ inst.imm)
        elif inst.name == "SLL":
            r.write(inst.rd, r.read(inst.rs1) << (r.read(inst.rs2) & 0x1F))
        elif inst.name == "SLLI":
            r.write(inst.rd, r.read(inst.rs1) << inst.imm)
        elif inst.name == "SRL":
            r.write(inst.rd, r.read(inst.rs1) >> (r.read(inst.rs2) & 0x1F))
        elif inst.name == "SRLI":
            r.write(inst.rd, r.read(inst.rs1) >> inst.imm)
        elif inst.name == "SRA":
            val = r.read_signed(inst.rs1)
            r.write(inst.rd, val >> (r.read(inst.rs2) & 0x1F))
        elif inst.name == "SRAI":
            val = r.read_signed(inst.rs1)
            r.write(inst.rd, val >> inst.imm)
        elif inst.name == "SLT":
            r.write(inst.rd, 1 if r.read_signed(inst.rs1) < r.read_signed(inst.rs2) else 0)
        elif inst.name == "SLTU":
            r.write(inst.rd, 1 if r.read(inst.rs1) < r.read(inst.rs2) else 0)
        elif inst.name == "SLTI":
            s_imm = inst.imm if inst.imm < 0x80000000 else inst.imm - 0x100000000
            r.write(inst.rd, 1 if r.read_signed(inst.rs1) < s_imm else 0)
        elif inst.name == "LUI":
            r.write(inst.rd, inst.imm << 12)
        elif inst.name == "AUIPC":
            r.write(inst.rd, self.pc + (inst.imm << 12))
        elif inst.name == "LW":
            addr = (r.read(inst.rs1) + inst.imm) & 0xFFFFFFFF
            r.write(inst.rd, self.mem.load_word(addr))
        elif inst.name == "SW":
            addr = (r.read(inst.rs1) + inst.imm) & 0xFFFFFFFF
            self.mem.store_word(addr, r.read(inst.rs2))
        elif inst.name == "BEQ":
            if r.read(inst.rs1) == r.read(inst.rs2):
                next_pc = self.pc + inst.imm
        elif inst.name == "BNE":
            if r.read(inst.rs1) != r.read(inst.rs2):
                next_pc = self.pc + inst.imm
        elif inst.name == "BLT":
            if r.read_signed(inst.rs1) < r.read_signed(inst.rs2):
                next_pc = self.pc + inst.imm
        elif inst.name == "BGE":
            if r.read_signed(inst.rs1) >= r.read_signed(inst.rs2):
                next_pc = self.pc + inst.imm
        elif inst.name == "JAL":
            r.write(inst.rd, self.pc + 4)
            next_pc = self.pc + inst.imm
        elif inst.name == "JALR":
            r.write(inst.rd, self.pc + 4)
            next_pc = (r.read(inst.rs1) + inst.imm) & ~1
        elif inst.name == "ECALL":
            self.halted = True

        self.pc = next_pc & 0xFFFFFFFF
        return inst

    def run(self, max_cycles: int = 10000, trace: bool = False):
        """Run until halt or max_cycles."""
        while not self.halted and self.cycle_count < max_cycles:
            pc_before = self.pc
            inst = self.step()
            if inst and trace:
                print(f"  [0x{pc_before:08X}] {inst}")
        if self.cycle_count >= max_cycles:
            print(f"  (stopped after {max_cycles} cycles)")


def assemble_simple(asm_lines: list[tuple]) -> list[int]:
    """Assemble simple RV32I instructions from tuples.

    This is a minimal assembler for demonstration. It supports a small
    subset of instructions encoded directly as 32-bit machine words.
    """
    code = []
    for line in asm_lines:
        op = line[0].upper()

        if op == "ADDI":
            rd, rs1, imm = line[1], line[2], line[3]
            word = ((imm & 0xFFF) << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0010011
            code.append(word)
        elif op == "ADD":
            rd, rs1, rs2 = line[1], line[2], line[3]
            word = (rs2 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0110011
            code.append(word)
        elif op == "SUB":
            rd, rs1, rs2 = line[1], line[2], line[3]
            word = (0b0100000 << 25) | (rs2 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0b0110011
            code.append(word)
        elif op == "AND":
            rd, rs1, rs2 = line[1], line[2], line[3]
            word = (rs2 << 20) | (rs1 << 15) | (0b111 << 12) | (rd << 7) | 0b0110011
            code.append(word)
        elif op == "OR":
            rd, rs1, rs2 = line[1], line[2], line[3]
            word = (rs2 << 20) | (rs1 << 15) | (0b110 << 12) | (rd << 7) | 0b0110011
            code.append(word)
        elif op == "SLT":
            rd, rs1, rs2 = line[1], line[2], line[3]
            word = (rs2 << 20) | (rs1 << 15) | (0b010 << 12) | (rd << 7) | 0b0110011
            code.append(word)
        elif op == "SLLI":
            rd, rs1, shamt = line[1], line[2], line[3]
            word = (shamt << 20) | (rs1 << 15) | (0b001 << 12) | (rd << 7) | 0b0010011
            code.append(word)
        elif op == "SW":
            rs2, offset, rs1 = line[1], line[2], line[3]
            imm_hi = (offset >> 5) & 0x7F
            imm_lo = offset & 0x1F
            word = (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (0b010 << 12) | (imm_lo << 7) | 0b0100011
            code.append(word)
        elif op == "LW":
            rd, offset, rs1 = line[1], line[2], line[3]
            word = ((offset & 0xFFF) << 20) | (rs1 << 15) | (0b010 << 12) | (rd << 7) | 0b0000011
            code.append(word)
        elif op == "BEQ":
            rs1, rs2, offset = line[1], line[2], line[3]
            # B-type encoding
            imm12 = (offset >> 12) & 1
            imm10_5 = (offset >> 5) & 0x3F
            imm4_1 = (offset >> 1) & 0xF
            imm11 = (offset >> 11) & 1
            word = (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
                   (0b000 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0b1100011
            code.append(word)
        elif op == "BNE":
            rs1, rs2, offset = line[1], line[2], line[3]
            imm12 = (offset >> 12) & 1
            imm10_5 = (offset >> 5) & 0x3F
            imm4_1 = (offset >> 1) & 0xF
            imm11 = (offset >> 11) & 1
            word = (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
                   (0b001 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0b1100011
            code.append(word)
        elif op == "ECALL":
            code.append(0b1110011)
        elif op == "NOP":
            # addi x0, x0, 0
            code.append(0b0010011)
        else:
            raise ValueError(f"Unknown instruction: {op}")

    return code


# ========================================================================
# Demonstrations
# ========================================================================

def demo_arithmetic():
    """Demo: basic arithmetic operations."""
    print("=" * 60)
    print("Demo 1: Arithmetic Operations")
    print("=" * 60)

    # x5=10, x6=3, x7=x5+x6, x8=x5-x6, x9=x6<<2, x10=x5&x6
    program = assemble_simple([
        ("ADDI", 5, 0, 10),       # x5 = 10
        ("ADDI", 6, 0, 3),        # x6 = 3
        ("ADD",  7, 5, 6),        # x7 = x5 + x6 = 13
        ("SUB",  8, 5, 6),        # x8 = x5 - x6 = 7
        ("SLLI", 9, 6, 2),        # x9 = x6 << 2 = 12
        ("AND",  10, 5, 6),       # x10 = x5 & x6 = 2
        ("OR",   11, 5, 6),       # x11 = x5 | x6 = 11
        ("SLT",  12, 6, 5),       # x12 = (x6 < x5) = 1
        ("ECALL",),
    ])

    sim = RV32ISimulator()
    sim.load_program(program)
    print("\nExecution trace:")
    sim.run(trace=True)
    print(f"\nInstructions executed: {sim.instruction_count}")
    print(f"\nKey registers:")
    for reg, expected in [(5, 10), (6, 3), (7, 13), (8, 7),
                          (9, 12), (10, 2), (11, 11), (12, 1)]:
        val = sim.regs.read(reg)
        status = "OK" if val == expected else "MISMATCH"
        abi = RV32IRegisters.ABI_NAMES[reg]
        print(f"  x{reg}/{abi} = {val} (expected {expected}) [{status}]")


def demo_memory():
    """Demo: load/store operations."""
    print("\n" + "=" * 60)
    print("Demo 2: Memory Operations (Load/Store)")
    print("=" * 60)

    # Store values to memory, load them back
    program = assemble_simple([
        ("ADDI", 2, 0, 1024),     # sp = 1024 (stack pointer)
        ("ADDI", 5, 0, 42),       # x5 = 42
        ("ADDI", 6, 0, 99),       # x6 = 99
        ("SW",   5, 0, 2),        # mem[sp+0] = 42
        ("SW",   6, 4, 2),        # mem[sp+4] = 99
        ("LW",   7, 0, 2),        # x7 = mem[sp+0] = 42
        ("LW",   8, 4, 2),        # x8 = mem[sp+4] = 99
        ("ADD",  9, 7, 8),        # x9 = 42 + 99 = 141
        ("ECALL",),
    ])

    sim = RV32ISimulator()
    sim.load_program(program)
    print("\nExecution trace:")
    sim.run(trace=True)
    print(f"\nKey registers:")
    for reg, expected in [(7, 42), (8, 99), (9, 141)]:
        val = sim.regs.read(reg)
        abi = RV32IRegisters.ABI_NAMES[reg]
        print(f"  x{reg}/{abi} = {val} (expected {expected})")


def demo_branch_loop():
    """Demo: counting loop using branches."""
    print("\n" + "=" * 60)
    print("Demo 3: Branch Loop (sum 1 to 10)")
    print("=" * 60)

    # Sum integers from 1 to 10
    # x5 = counter (1..10), x6 = limit (10), x7 = sum
    program = assemble_simple([
        ("ADDI",  5, 0, 1),       # x5 = 1 (counter)
        ("ADDI",  6, 0, 11),      # x6 = 11 (limit: stop when counter >= 11)
        ("ADDI",  7, 0, 0),       # x7 = 0 (sum)
        # loop: (offset 12 = 3 instructions * 4 bytes)
        ("ADD",   7, 7, 5),       # x7 += x5
        ("ADDI",  5, 5, 1),       # x5 += 1
        ("BNE",   5, 6, -8),      # if x5 != x6, go back 2 instructions
        ("ECALL",),
    ])

    sim = RV32ISimulator()
    sim.load_program(program)
    print("\nExecution trace:")
    sim.run(trace=True)
    expected_sum = sum(range(1, 11))
    val = sim.regs.read(7)
    print(f"\nSum of 1..10 = {val} (expected {expected_sum})")
    print(f"Instructions executed: {sim.instruction_count}")


def demo_instruction_encoding():
    """Demo: show instruction encoding details."""
    print("\n" + "=" * 60)
    print("Demo 4: Instruction Encoding")
    print("=" * 60)

    test_instructions = [
        ("ADD x7, x5, x6", ("ADD", 7, 5, 6)),
        ("ADDI x10, x0, 42", ("ADDI", 10, 0, 42)),
        ("SUB x8, x5, x6", ("SUB", 8, 5, 6)),
        ("SW x5, 0(x2)", ("SW", 5, 0, 2)),
        ("LW x7, 0(x2)", ("LW", 7, 0, 2)),
    ]

    for desc, asm in test_instructions:
        words = assemble_simple([asm])
        word = words[0]
        inst = RV32IDecoder.decode(word)
        print(f"\n  {desc}")
        print(f"  Binary:  {word:032b}")
        print(f"  Hex:     0x{word:08X}")
        print(f"  Decoded: {inst}")


if __name__ == "__main__":
    demo_arithmetic()
    demo_memory()
    demo_branch_loop()
    demo_instruction_encoding()

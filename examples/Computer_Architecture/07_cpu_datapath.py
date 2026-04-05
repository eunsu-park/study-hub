"""
Single-Cycle CPU Datapath Simulator

Demonstrates a simplified MIPS-like single-cycle CPU with:
- Fetch → Decode → Execute → Memory → Writeback
- R-type instructions (add, sub, and, or, slt)
- I-type instructions (addi, lw, sw, beq)
- Register file, ALU, data memory

Theory:
- Single-cycle: each instruction completes in one clock cycle.
- Clock period must accommodate the slowest instruction (lw).
- Datapath components: PC, instruction memory, register file,
  ALU, data memory, sign extender, muxes.

Adapted from Computer Architecture Lesson 07.
"""

from dataclasses import dataclass, field


# ── Instruction Encoding ────────────────────────────────────────────────

# Opcodes
OP_RTYPE = 0x00
OP_ADDI  = 0x08
OP_LW    = 0x23
OP_SW    = 0x2B
OP_BEQ   = 0x04

# Funct codes (R-type)
FUNCT_ADD = 0x20
FUNCT_SUB = 0x22
FUNCT_AND = 0x24
FUNCT_OR  = 0x25
FUNCT_SLT = 0x2A


def encode_rtype(rs: int, rt: int, rd: int, funct: int) -> int:
    return (OP_RTYPE << 26) | (rs << 21) | (rt << 16) | (rd << 11) | funct


def encode_itype(op: int, rs: int, rt: int, imm: int) -> int:
    if imm < 0:
        imm = imm & 0xFFFF  # 16-bit two's complement
    # MIPS I-type: 6-bit opcode | 5-bit rs | 5-bit rt | 16-bit immediate.
    # The fixed field positions allow the register-file read addresses
    # to be decoded in parallel with opcode interpretation — a key
    # single-cycle and pipeline timing optimization.
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


# ── CPU Components ──────────────────────────────────────────────────────

class CPU:
    """Single-cycle MIPS-like CPU simulator."""

    def __init__(self):
        self.pc = 0
        self.regs = [0] * 32        # 32 registers, R0 = 0
        self.imem: list[int] = []    # instruction memory
        self.dmem = [0] * 256        # data memory (256 words)
        self.cycle = 0
        self.log: list[str] = []

    def load_program(self, instructions: list[int]) -> None:
        self.imem = instructions
        self.pc = 0
        self.cycle = 0

    def _sign_extend(self, value: int, bits: int = 16) -> int:
        if value & (1 << (bits - 1)):
            return value - (1 << bits)
        return value

    def _reg_write(self, reg: int, value: int) -> None:
        # R0 is hardwired to zero — silently discard writes.
        # The 32-bit mask simulates fixed-width register overflow
        # behavior (real hardware simply ignores upper bits).
        if reg != 0:
            self.regs[reg] = value & 0xFFFFFFFF

    def step(self) -> bool:
        """Execute one instruction. Returns False if done."""
        if self.pc // 4 >= len(self.imem):
            return False

        # ── FETCH ──
        # PC is byte-addressed but instructions are word-aligned (4 bytes),
        # so dividing by 4 converts the byte address to an instruction index.
        instr = self.imem[self.pc // 4]
        self.cycle += 1

        # ── DECODE ──
        opcode = (instr >> 26) & 0x3F
        rs = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        rd = (instr >> 11) & 0x1F
        funct = instr & 0x3F
        imm = self._sign_extend(instr & 0xFFFF)

        rs_val = self.regs[rs]
        rt_val = self.regs[rt]

        msg = f"  Cycle {self.cycle:>2} [PC={self.pc:>3}]: "

        # ── EXECUTE ──
        if opcode == OP_RTYPE:
            if funct == FUNCT_ADD:
                result = rs_val + rt_val
                msg += f"add  R{rd}, R{rs}, R{rt}  → R{rd}={result}"
                self._reg_write(rd, result)
            elif funct == FUNCT_SUB:
                result = rs_val - rt_val
                msg += f"sub  R{rd}, R{rs}, R{rt}  → R{rd}={result}"
                self._reg_write(rd, result)
            elif funct == FUNCT_AND:
                result = rs_val & rt_val
                msg += f"and  R{rd}, R{rs}, R{rt}  → R{rd}={result}"
                self._reg_write(rd, result)
            elif funct == FUNCT_OR:
                result = rs_val | rt_val
                msg += f"or   R{rd}, R{rs}, R{rt}  → R{rd}={result}"
                self._reg_write(rd, result)
            elif funct == FUNCT_SLT:
                result = 1 if rs_val < rt_val else 0
                msg += f"slt  R{rd}, R{rs}, R{rt}  → R{rd}={result}"
                self._reg_write(rd, result)
            self.pc += 4

        elif opcode == OP_ADDI:
            result = rs_val + imm
            msg += f"addi R{rt}, R{rs}, {imm}  → R{rt}={result}"
            self._reg_write(rt, result)
            self.pc += 4

        elif opcode == OP_LW:
            addr = rs_val + imm
            word_addr = addr // 4
            value = self.dmem[word_addr] if 0 <= word_addr < len(self.dmem) else 0
            msg += f"lw   R{rt}, {imm}(R{rs})  → R{rt}=mem[{addr}]={value}"
            self._reg_write(rt, value)
            self.pc += 4

        elif opcode == OP_SW:
            addr = rs_val + imm
            word_addr = addr // 4
            if 0 <= word_addr < len(self.dmem):
                self.dmem[word_addr] = rt_val
            msg += f"sw   R{rt}, {imm}(R{rs})  → mem[{addr}]={rt_val}"
            self.pc += 4

        elif opcode == OP_BEQ:
            if rs_val == rt_val:
                # Branch offset is relative to PC+4 (the *next* instruction),
                # not the current PC.  This is because the PC has already
                # been conceptually incremented during fetch in real hardware.
                self.pc += 4 + (imm * 4)
                msg += f"beq  R{rs}, R{rt}, {imm}  → TAKEN (PC={self.pc})"
            else:
                self.pc += 4
                msg += f"beq  R{rs}, R{rt}, {imm}  → NOT taken"

        else:
            msg += f"UNKNOWN opcode={opcode}"
            self.pc += 4

        self.log.append(msg)
        print(msg)
        return True

    def run(self, max_cycles: int = 100) -> None:
        """Run until program ends or max cycles."""
        while self.step() and self.cycle < max_cycles:
            pass

    def dump_regs(self) -> None:
        """Print non-zero registers."""
        print("  Registers (non-zero):")
        for i in range(32):
            if self.regs[i] != 0:
                print(f"    R{i:>2} = {self.regs[i]}")


# ── Demo Programs ───────────────────────────────────────────────────────

def demo_basic_arithmetic():
    """Simple arithmetic program."""
    print("=" * 60)
    print("SINGLE-CYCLE CPU: BASIC ARITHMETIC")
    print("=" * 60)

    cpu = CPU()

    # Program: compute (5 + 3) * 2 - 1
    # R1 = 5, R2 = 3, R3 = R1 + R2, R4 = R3 + R3, R5 = R4 - 1
    program = [
        encode_itype(OP_ADDI, 0, 1, 5),          # addi R1, R0, 5
        encode_itype(OP_ADDI, 0, 2, 3),          # addi R2, R0, 3
        encode_rtype(1, 2, 3, FUNCT_ADD),         # add  R3, R1, R2
        encode_rtype(3, 3, 4, FUNCT_ADD),         # add  R4, R3, R3
        encode_itype(OP_ADDI, 4, 5, -1),         # addi R5, R4, -1
    ]

    print(f"\n  Program: (5 + 3) * 2 - 1 = {(5+3)*2-1}\n")
    cpu.load_program(program)
    cpu.run()
    print()
    cpu.dump_regs()


def demo_memory_access():
    """Program with load/store."""
    print("\n" + "=" * 60)
    print("SINGLE-CYCLE CPU: MEMORY ACCESS")
    print("=" * 60)

    cpu = CPU()

    # Store values, load back, compute
    program = [
        encode_itype(OP_ADDI, 0, 1, 42),          # R1 = 42
        encode_itype(OP_ADDI, 0, 2, 100),         # R2 = 100 (base addr)
        encode_itype(OP_SW, 2, 1, 0),             # mem[100] = R1 (42)
        encode_itype(OP_ADDI, 0, 3, 58),          # R3 = 58
        encode_itype(OP_SW, 2, 3, 4),             # mem[104] = R3 (58)
        encode_itype(OP_LW, 2, 4, 0),             # R4 = mem[100] (42)
        encode_itype(OP_LW, 2, 5, 4),             # R5 = mem[104] (58)
        encode_rtype(4, 5, 6, FUNCT_ADD),          # R6 = R4 + R5 (100)
    ]

    print(f"\n  Program: store 42 and 58, load back, add\n")
    cpu.load_program(program)
    cpu.run()
    print()
    cpu.dump_regs()


def demo_branch():
    """Program with conditional branch."""
    print("\n" + "=" * 60)
    print("SINGLE-CYCLE CPU: BRANCH")
    print("=" * 60)

    cpu = CPU()

    # Count down from 5 to 0
    # R1 = 5, R2 = 0 (target), R3 = accumulator
    # Loop: R3 = R3 + R1, R1 = R1 - 1, if R1 != R2 goto Loop
    program = [
        encode_itype(OP_ADDI, 0, 1, 5),           # R1 = 5
        encode_itype(OP_ADDI, 0, 3, 0),           # R3 = 0 (sum)
        # Loop (PC=8):
        encode_rtype(3, 1, 3, FUNCT_ADD),          # R3 = R3 + R1
        encode_itype(OP_ADDI, 1, 1, -1),          # R1 = R1 - 1
        encode_itype(OP_BEQ, 1, 0, 1),            # if R1==0, skip next
        encode_itype(OP_BEQ, 0, 0, -4),           # unconditional: goto Loop
        # Done: R3 should be 5+4+3+2+1 = 15
    ]

    print(f"\n  Program: sum 5+4+3+2+1 = 15\n")
    cpu.load_program(program)
    cpu.run()
    print()
    cpu.dump_regs()
    print(f"\n  R3 = {cpu.regs[3]} (expected 15)")


if __name__ == "__main__":
    demo_basic_arithmetic()
    demo_memory_access()
    demo_branch()

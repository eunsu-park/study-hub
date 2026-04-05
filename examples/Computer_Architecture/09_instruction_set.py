"""
Instruction Set Architecture (ISA) Design

Demonstrates:
- Instruction encoding and decoding (fixed-width 32-bit)
- R-type, I-type, and J-type instruction formats
- Addressing modes: register, immediate, base+offset, PC-relative
- Assembly-to-machine-code translation

Theory:
- The ISA is the contract between software and hardware — it defines
  instruction formats, opcodes, registers, and addressing modes.
- Fixed-width encoding (e.g., 32-bit RISC) simplifies fetch and
  decode at the cost of code density.
- Common formats:
  - R-type: opcode | rs | rt | rd | funct  (register-register)
  - I-type: opcode | rs | rt | immediate   (register-immediate)
  - J-type: opcode | address               (jump)
- Addressing modes determine how operands are located:
  - Register direct: operand is in a register
  - Immediate: operand is embedded in the instruction
  - Base + offset: effective address = register + displacement
  - PC-relative: target = PC + offset (for branches)

Adapted from Computer Architecture Lesson 09.
"""

from dataclasses import dataclass


# ── Instruction Formats ───────────────────────────────────────────────

# 32-bit fixed-width instruction layout (MIPS-like)
# R-type: [opcode(6) | rs(5) | rt(5) | rd(5) | shamt(5) | funct(6)]
# I-type: [opcode(6) | rs(5) | rt(5) | immediate(16)]
# J-type: [opcode(6) | address(26)]

FUNCT_CODES = {
    "ADD":  0x20,
    "SUB":  0x22,
    "AND":  0x24,
    "OR":   0x25,
    "SLT":  0x2A,
    "SLL":  0x00,
    "SRL":  0x02,
    "JR":   0x08,
}

OPCODE_MAP = {
    "R-TYPE": 0x00,
    "ADDI":   0x08,
    "ANDI":   0x0C,
    "ORI":    0x0D,
    "LW":     0x23,
    "SW":     0x2B,
    "BEQ":    0x04,
    "BNE":    0x05,
    "J":      0x02,
    "JAL":    0x03,
}

# Reverse maps for decoding
FUNCT_TO_NAME = {v: k for k, v in FUNCT_CODES.items()}
OPCODE_TO_NAME = {v: k for k, v in OPCODE_MAP.items() if k != "R-TYPE"}


@dataclass
class DecodedInstr:
    """Fully decoded instruction."""
    fmt: str        # "R", "I", or "J"
    name: str
    opcode: int
    rs: int = 0
    rt: int = 0
    rd: int = 0
    shamt: int = 0
    funct: int = 0
    imm: int = 0
    addr: int = 0


def encode_r(funct_name: str, rd: int, rs: int, rt: int,
             shamt: int = 0) -> int:
    """Encode an R-type instruction."""
    opcode = 0x00
    funct = FUNCT_CODES[funct_name]
    return ((opcode << 26) | (rs << 21) | (rt << 16) |
            (rd << 11) | (shamt << 6) | funct)


def encode_i(op_name: str, rt: int, rs: int, imm: int) -> int:
    """Encode an I-type instruction."""
    opcode = OPCODE_MAP[op_name]
    # Sign-extend 16-bit immediate (handle negative values)
    imm_bits = imm & 0xFFFF
    return (opcode << 26) | (rs << 21) | (rt << 16) | imm_bits


def encode_j(op_name: str, target: int) -> int:
    """Encode a J-type instruction."""
    opcode = OPCODE_MAP[op_name]
    return (opcode << 26) | (target & 0x3FFFFFF)


def decode(word: int) -> DecodedInstr:
    """Decode a 32-bit instruction word."""
    opcode = (word >> 26) & 0x3F

    if opcode == 0x00:
        # R-type
        rs = (word >> 21) & 0x1F
        rt = (word >> 16) & 0x1F
        rd = (word >> 11) & 0x1F
        shamt = (word >> 6) & 0x1F
        funct = word & 0x3F
        name = FUNCT_TO_NAME.get(funct, f"FUNCT_{funct:#x}")
        return DecodedInstr("R", name, opcode, rs=rs, rt=rt, rd=rd,
                            shamt=shamt, funct=funct)

    elif opcode in (0x02, 0x03):
        # J-type
        addr = word & 0x3FFFFFF
        name = OPCODE_TO_NAME.get(opcode, f"OP_{opcode:#x}")
        return DecodedInstr("J", name, opcode, addr=addr)

    else:
        # I-type
        rs = (word >> 21) & 0x1F
        rt = (word >> 16) & 0x1F
        imm = word & 0xFFFF
        # Sign-extend
        if imm & 0x8000:
            imm -= 0x10000
        name = OPCODE_TO_NAME.get(opcode, f"OP_{opcode:#x}")
        return DecodedInstr("I", name, opcode, rs=rs, rt=rt, imm=imm)


def format_binary(word: int) -> str:
    """Format 32-bit word as binary with field separators."""
    b = f"{word:032b}"
    opcode_bits = b[:6]
    rest = b[6:]
    return f"{opcode_bits}|{rest}"


def disassemble(instr: DecodedInstr) -> str:
    """Convert decoded instruction to assembly string."""
    if instr.fmt == "R":
        if instr.name in ("SLL", "SRL"):
            return f"{instr.name} $r{instr.rd}, $r{instr.rt}, {instr.shamt}"
        if instr.name == "JR":
            return f"{instr.name} $r{instr.rs}"
        return f"{instr.name} $r{instr.rd}, $r{instr.rs}, $r{instr.rt}"
    elif instr.fmt == "I":
        if instr.name in ("LW", "SW"):
            return f"{instr.name} $r{instr.rt}, {instr.imm}($r{instr.rs})"
        if instr.name in ("BEQ", "BNE"):
            return f"{instr.name} $r{instr.rs}, $r{instr.rt}, {instr.imm}"
        return f"{instr.name} $r{instr.rt}, $r{instr.rs}, {instr.imm}"
    else:
        return f"{instr.name} {instr.addr}"


# ── Addressing Mode Demos ────────────────────────────────────────────

def compute_effective_address(mode: str, base: int = 0,
                              offset: int = 0, pc: int = 0) -> int:
    """Compute effective address for various addressing modes."""
    if mode == "register":
        return base  # address is in a register
    elif mode == "immediate":
        return offset  # address is the immediate itself
    elif mode == "base_offset":
        return base + offset  # base register + displacement
    elif mode == "pc_relative":
        return pc + 4 + (offset * 4)  # PC + 4 + sign-extended offset * 4
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Demos ─────────────────────────────────────────────────────────────

def demo_encoding():
    """Demonstrate instruction encoding."""
    print("=" * 60)
    print("INSTRUCTION ENCODING")
    print("=" * 60)

    examples = [
        ("R-type", "ADD $r3, $r1, $r2",
         encode_r("ADD", rd=3, rs=1, rt=2)),
        ("R-type", "SLL $r4, $r5, 2",
         encode_r("SLL", rd=4, rs=0, rt=5, shamt=2)),
        ("I-type", "ADDI $r2, $r1, 100",
         encode_i("ADDI", rt=2, rs=1, imm=100)),
        ("I-type", "LW $r3, -4($r8)",
         encode_i("LW", rt=3, rs=8, imm=-4)),
        ("I-type", "BEQ $r1, $r2, 5",
         encode_i("BEQ", rt=2, rs=1, imm=5)),
        ("J-type", "J 1024",
         encode_j("J", target=1024)),
    ]

    for fmt, asm, word in examples:
        print(f"\n  {fmt}: {asm}")
        print(f"    Hex:    0x{word:08X}")
        print(f"    Binary: {format_binary(word)}")

        # Verify round-trip
        d = decode(word)
        print(f"    Decode: {disassemble(d)}")


def demo_decoding():
    """Demonstrate instruction decoding."""
    print("\n" + "=" * 60)
    print("INSTRUCTION DECODING")
    print("=" * 60)

    # Raw machine words to decode
    words = [
        0x00221820,  # ADD $r3, $r1, $r2
        0x2129FFFF,  # ADDI $r9, $r9, -1
        0x8C620004,  # LW $r2, 4($r3)
        0xAC050008,  # SW $r5, 8($r0)
        0x10220003,  # BEQ $r1, $r2, 3
        0x08000100,  # J 256
    ]

    print(f"\n  {'Hex':<12} {'Format':<4} {'Instruction':<30}")
    print(f"  {'-'*12} {'-'*4} {'-'*30}")
    for word in words:
        d = decode(word)
        asm = disassemble(d)
        print(f"  0x{word:08X}  {d.fmt:<4} {asm}")


def demo_addressing_modes():
    """Demonstrate addressing modes and effective address calculation."""
    print("\n" + "=" * 60)
    print("ADDRESSING MODES")
    print("=" * 60)

    modes = [
        ("register",    {"base": 0x1000},
         "JR $r4          — EA = Reg[$r4]"),
        ("immediate",   {"offset": 42},
         "ADDI $r1, $r0, 42  — operand is 42"),
        ("base_offset", {"base": 0x1000, "offset": 8},
         "LW $r2, 8($r4)  — EA = 0x1000 + 8"),
        ("base_offset", {"base": 0x1000, "offset": -4},
         "LW $r2, -4($r4) — EA = 0x1000 + (-4)"),
        ("pc_relative", {"pc": 0x0400, "offset": 3},
         "BEQ ..., 3       — target = PC+4 + 3*4"),
        ("pc_relative", {"pc": 0x0400, "offset": -2},
         "BEQ ..., -2      — target = PC+4 + (-2)*4"),
    ]

    print(f"\n  {'Mode':<14} {'Example':<35} {'EA/Value'}")
    print(f"  {'-'*14} {'-'*35} {'-'*12}")
    for mode, kwargs, desc in modes:
        ea = compute_effective_address(mode, **kwargs)
        print(f"  {mode:<14} {desc:<35} 0x{ea:04X}")


def demo_instruction_formats():
    """Show bit-field layout of each format."""
    print("\n" + "=" * 60)
    print("INSTRUCTION FORMAT LAYOUTS (32-bit)")
    print("=" * 60)

    print("""
  R-type: Register-register operations
  ┌────────┬───────┬───────┬───────┬───────┬────────┐
  │ opcode │  rs   │  rt   │  rd   │ shamt │ funct  │
  │  6 bit │ 5 bit │ 5 bit │ 5 bit │ 5 bit │ 6 bit  │
  └────────┴───────┴───────┴───────┴───────┴────────┘

  I-type: Immediate / load-store / branch
  ┌────────┬───────┬───────┬──────────────────────────┐
  │ opcode │  rs   │  rt   │       immediate           │
  │  6 bit │ 5 bit │ 5 bit │        16 bit             │
  └────────┴───────┴───────┴──────────────────────────┘

  J-type: Jump
  ┌────────┬────────────────────────────────────────────┐
  │ opcode │                 address                    │
  │  6 bit │                  26 bit                    │
  └────────┴────────────────────────────────────────────┘

  Trade-offs:
  - More format types → more flexible encoding, but harder to decode
  - Wider immediate → larger constants, fewer instructions needed
  - RISC favors few formats; CISC may use variable-length encoding
""")


if __name__ == "__main__":
    demo_encoding()
    demo_decoding()
    demo_addressing_modes()
    demo_instruction_formats()

"""
Exercises for Lesson 19: RISC-V Architecture
Topic: Computer_Architecture

Solutions to practice problems covering RISC-V instruction encoding,
assembly tracing, ISA comparison, extension analysis, and pipeline hazards.
"""


def exercise_1():
    """
    Encode the instruction: addi x10, x5, -3
    Identify format type and fill in each field.
    """
    print("=== Exercise 1: Instruction Encoding ===\n")
    print("Instruction: addi x10, x5, -3\n")

    # ADDI is I-type format:
    #   [imm[11:0] | rs1 | funct3 | rd | opcode]
    #   [  12 bits | 5b  |  3b    | 5b |  7b   ]

    opcode = 0b0010011   # I-type ALU (OP-IMM)
    rd     = 10          # x10 = a0
    funct3 = 0b000       # ADDI
    rs1    = 5           # x5 = t0
    imm    = -3 & 0xFFF  # -3 in 12-bit two's complement = 0xFFD = 111111111101

    encoded = (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    print("Format: I-type")
    print(f"  opcode = 0010011 (OP-IMM)")
    print(f"  rd     = {rd:05b} (x10/a0)")
    print(f"  funct3 = {funct3:03b} (ADDI)")
    print(f"  rs1    = {rs1:05b} (x5/t0)")
    print(f"  imm    = {imm:012b} (-3 in 12-bit two's complement)")
    print()
    print(f"  Complete encoding: {encoded:032b}")
    print(f"  Hex: 0x{encoded:08X}")

    # Verify field positions
    print(f"\n  Bit layout:")
    print(f"  [imm=111111111101 | rs1=00101 | f3=000 | rd=01010 | op=0010011]")
    print(f"  [    31:20        | 19:15     | 14:12  | 11:7    | 6:0       ]")
    print()


def exercise_2():
    """
    Trace register values after each instruction.
    """
    print("=== Exercise 2: Assembly Tracing ===\n")

    regs = {"x0": 0}

    trace_steps = [
        ("addi  x5, x0, 10",  "x5",  10,  "x5 = 0 + 10 = 10"),
        ("addi  x6, x0, 3",   "x6",   3,  "x6 = 0 + 3 = 3"),
        ("add   x7, x5, x6",  "x7",  13,  "x7 = 10 + 3 = 13"),
        ("sub   x8, x5, x6",  "x8",   7,  "x8 = 10 - 3 = 7"),
        ("slli  x9, x6, 2",   "x9",  12,  "x9 = 3 << 2 = 12 (multiply by 4)"),
        ("and   x10, x5, x6", "x10",  2,  "x10 = 0b1010 & 0b0011 = 0b0010 = 2"),
        ("or    x11, x5, x6", "x11", 11,  "x11 = 0b1010 | 0b0011 = 0b1011 = 11"),
        ("slt   x12, x6, x5", "x12",  1,  "x12 = (3 < 10) = 1 (true)"),
    ]

    for asm, reg, val, explanation in trace_steps:
        regs[reg] = val
        print(f"  {asm:<25} → {reg} = {val:<5} // {explanation}")

    print("\nFinal register state:")
    for reg in ["x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]:
        print(f"  {reg} = {regs[reg]}")
    print()


def exercise_3():
    """
    Write max(a, b) in RISC-V, x86-64, and AArch64 assembly.
    Compare instruction counts and features used.
    """
    print("=== Exercise 3: ISA Comparison — max(a, b) ===\n")

    print("--- RISC-V RV32I (a0=a, a1=b, result in a0) ---")
    print("  bge   a0, a1, done    # if a >= b, a is already max")
    print("  mv    a0, a1          # else: a0 = b")
    print("  done:")
    print("  ret")
    print("  → 3 instructions (branch + move + return)")
    print("  → Uses compare-and-branch (no condition flags)")
    print()

    print("--- x86-64 (edi=a, esi=b, result in eax) ---")
    print("  mov   eax, edi        # eax = a")
    print("  cmp   edi, esi        # compare a, b (sets FLAGS)")
    print("  cmovl eax, esi        # if a < b: eax = b (conditional move)")
    print("  ret")
    print("  → 4 instructions (no branch — uses conditional move)")
    print("  → Uses FLAGS register + cmov (branchless)")
    print()

    print("--- AArch64 (w0=a, w1=b, result in w0) ---")
    print("  cmp   w0, w1          # compare a, b (sets NZCV)")
    print("  csel  w0, w0, w1, ge  # w0 = (a >= b) ? a : b")
    print("  ret")
    print("  → 3 instructions (branchless with conditional select)")
    print("  → Uses NZCV flags + csel")
    print()

    print("Comparison:")
    print(f"  {'ISA':<12} {'Instructions':<14} {'Branch?':<10} {'Key Feature'}")
    print(f"  {'RISC-V':<12} {'3':<14} {'Yes':<10} {'Fused compare-and-branch'}")
    print(f"  {'x86-64':<12} {'4':<14} {'No (cmov)':<10} {'Conditional move'}")
    print(f"  {'AArch64':<12} {'3':<14} {'No (csel)':<10} {'Conditional select'}")
    print()
    print("  RISC-V lacks conditional moves in the base ISA, requiring a branch.")
    print("  x86 and ARM can do branchless max, avoiding branch prediction overhead.")
    print("  The Zicond extension adds conditional operations to RISC-V.")
    print()


def exercise_4():
    """
    Analyze RV64IMAFDC: list each component, its meaning, and shorthand.
    """
    print("=== Exercise 4: Extension Analysis (RV64IMAFDC) ===\n")

    components = [
        ("RV64", "Base", "64-bit address space, 64-bit integer registers (x0-x31)"),
        ("I", "Integer", "Base integer instruction set: 47 instructions "
         "(arithmetic, logical, shift, load/store, branch, jump, system)"),
        ("M", "Multiply/Divide", "Integer multiply: MUL, MULH, MULHSU, MULHU; "
         "Integer divide: DIV, DIVU, REM, REMU (32-bit and 64-bit versions)"),
        ("A", "Atomic", "Load-Reserved/Store-Conditional (LR.W/D, SC.W/D); "
         "Atomic Memory Operations (AMOSWAP, AMOADD, AMOAND, AMOOR, etc.)"),
        ("F", "Single-Precision Float", "32 floating-point registers (f0-f31); "
         "IEEE 754 single-precision: FADD.S, FSUB.S, FMUL.S, FDIV.S, FSQRT.S, etc."),
        ("D", "Double-Precision Float", "Extends f registers to 64-bit; "
         "IEEE 754 double-precision: FADD.D, FSUB.D, FMUL.D, etc. Requires F extension."),
        ("C", "Compressed", "16-bit encodings for common instructions; "
         "reduces code size by 25-30%; mixes freely with 32-bit instructions"),
    ]

    for letter, name, desc in components:
        print(f"  {letter:<5} — {name}")
        print(f"         {desc}")
        print()

    print("Shorthand equivalence:")
    print("  RV64IMAFDC = RV64GC")
    print("  where G = I + M + A + F + D (General-purpose)")
    print("  and C = Compressed instructions")
    print()
    print("  RV64GC is the standard combination for application-class processors")
    print("  that run Linux and general-purpose software.")
    print()


def exercise_5():
    """
    Pipeline hazard analysis for load-use sequence.
    """
    print("=== Exercise 5: Pipeline Hazard Analysis ===\n")

    print("Code:")
    print("  I1: lw    x5, 0(x10)     # Load word into x5")
    print("  I2: add   x6, x5, x7    # Uses x5 (loaded value)")
    print("  I3: sub   x8, x5, x9    # Also uses x5")
    print()

    print("1) Hazard Type: RAW (Read After Write) — True Dependence")
    print("   I1 writes x5, I2 reads x5 → RAW between I1 and I2")
    print("   I1 writes x5, I3 reads x5 → RAW between I1 and I3")
    print()

    print("2) Stalls WITHOUT forwarding:")
    print()
    print("   Cycle:  1    2    3    4    5    6    7    8    9")
    print("   I1:     IF   ID   EX   MEM  WB")
    print("   I2:          IF   ID   --- --- EX  MEM   WB")
    print("   I3:               IF   --- --- ID   EX   MEM  WB")
    print()
    print("   I2 needs x5 at ID (cycle 3), but x5 is written at WB (cycle 5)")
    print("   → 2 stall cycles for I2")
    print("   I3 is delayed by I2's stalls → effectively 2 stalls total")
    print()

    print("3) Stalls WITH forwarding (EX→EX and MEM→EX):")
    print()
    print("   Cycle:  1    2    3    4    5    6    7")
    print("   I1:     IF   ID   EX   MEM  WB")
    print("   I2:          IF   ID   ---  EX   MEM  WB")
    print("   I3:               IF   ---  ID   EX   MEM  WB")
    print()
    print("   I1's LW result is available at end of MEM stage (cycle 4)")
    print("   I2 needs x5 at EX stage → earliest is cycle 5")
    print("   → 1 stall cycle (the 'load-use hazard')")
    print("   MEM→EX forwarding cannot eliminate this stall because the")
    print("   data isn't ready until MEM finishes.")
    print()

    print("4) Compiler optimization — insert independent instruction:")
    print()
    print("   Original:           Optimized:")
    print("   lw  x5, 0(x10)     lw  x5, 0(x10)")
    print("   add x6, x5, x7     addi x11, x0, 1   ← independent work")
    print("   sub x8, x5, x9     add  x6, x5, x7   ← no stall needed")
    print("                      sub  x8, x5, x9")
    print()
    print("   By placing an independent instruction between the load and use,")
    print("   the compiler fills the 'load delay slot' and eliminates the stall.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

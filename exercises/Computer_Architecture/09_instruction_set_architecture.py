"""
Exercises for Lesson 09: Instruction Set Architecture
Topic: Computer_Architecture

Solutions to practice problems covering ISA definitions, CISC vs RISC,
MIPS instruction formats and encoding, addressing modes, x86 micro-ops,
RISC-V advantages, and C-to-assembly translation.
"""


def exercise_1():
    """List 5 things that an ISA defines."""
    isa_definitions = [
        ("Instructions",    "The set of operations the CPU can execute (ADD, MOV, LOAD, etc.)"),
        ("Registers",       "Number, size, and purpose of programmer-visible registers"),
        ("Data Types",      "Supported data sizes (byte, half, word, double) and formats (int, float)"),
        ("Addressing Modes","How operands are specified (register, immediate, base+offset, etc.)"),
        ("Memory Model",    "Byte/word addressable, endianness (big/little), alignment rules"),
    ]

    print("Five things an ISA (Instruction Set Architecture) defines:")
    print("  (The ISA is the contract between hardware and software)")
    print()
    for i, (item, desc) in enumerate(isa_definitions, 1):
        print(f"  {i}. {item:<20s}: {desc}")

    print("\n  Additional ISA concerns:")
    print("    - Interrupt/exception handling model")
    print("    - Privilege levels (user/kernel mode)")
    print("    - Virtual memory support (page tables, TLB management)")
    print("    - SIMD/vector extensions (SSE, AVX, NEON, SVE)")


def exercise_2():
    """Explain 3 major differences between CISC and RISC."""
    differences = [
        {
            "dimension": "Instruction Complexity",
            "cisc": "Complex, multi-step instructions (e.g., MOVSB moves a string byte)",
            "risc": "Simple, single-operation instructions (e.g., ADD, LW, SW)",
            "why": "RISC simplicity enables pipelining; CISC complexity helps code density",
        },
        {
            "dimension": "Instruction Length",
            "cisc": "Variable length (x86: 1-15 bytes)",
            "risc": "Fixed length (MIPS/ARM: 4 bytes)",
            "why": "Fixed length simplifies fetch and decode hardware dramatically",
        },
        {
            "dimension": "Memory Access",
            "cisc": "Any instruction can access memory (ADD can use memory operand)",
            "risc": "Only Load/Store instructions access memory (load-store architecture)",
            "why": "RISC separates memory access from computation for simpler pipelining",
        },
    ]

    print("Three major CISC vs RISC differences:")
    for i, d in enumerate(differences, 1):
        print(f"\n  {i}. {d['dimension']}")
        print(f"     CISC: {d['cisc']}")
        print(f"     RISC: {d['risc']}")
        print(f"     Rationale: {d['why']}")


def exercise_3():
    """Explain three MIPS instruction formats (R, I, J)."""
    print("MIPS Instruction Formats (all 32 bits):")
    print()

    formats = {
        "R-type (Register)": {
            "fields": "opcode(6) | rs(5) | rt(5) | rd(5) | shamt(5) | funct(6)",
            "usage": "Register-to-register operations",
            "examples": "ADD, SUB, AND, OR, SLT, SLL, SRL",
            "notes": "opcode=000000 for all R-type; funct field distinguishes operations",
        },
        "I-type (Immediate)": {
            "fields": "opcode(6) | rs(5) | rt(5) | immediate(16)",
            "usage": "Operations with an immediate value or memory access",
            "examples": "ADDI, LW, SW, BEQ, BNE, SLTI",
            "notes": "Immediate is sign-extended to 32 bits for arithmetic, address calc",
        },
        "J-type (Jump)": {
            "fields": "opcode(6) | address(26)",
            "usage": "Unconditional jumps",
            "examples": "J, JAL",
            "notes": "Target = {PC[31:28], address, 00} (word-aligned, 256MB range)",
        },
    }

    for fmt_name, info in formats.items():
        print(f"  {fmt_name}:")
        print(f"    Fields:   {info['fields']}")
        print(f"    Usage:    {info['usage']}")
        print(f"    Examples: {info['examples']}")
        print(f"    Notes:    {info['notes']}")
        print()


def exercise_4():
    """
    Encode ADD $t2, $s0, $s1 as a 32-bit binary number.
    $t2=10, $s0=16, $s1=17, ADD funct=0x20
    """
    # R-type format: opcode | rs | rt | rd | shamt | funct
    opcode = 0b000000  # R-type
    rs = 16            # $s0 = register 16
    rt = 17            # $s1 = register 17
    rd = 10            # $t2 = register 10
    shamt = 0          # No shift
    funct = 0x20       # ADD function code = 32

    print("Encoding ADD $t2, $s0, $s1:")
    print(f"  Format: R-type")
    print(f"  ADD $t2, $s0, $s1 → rd = $s0 + $s1")
    print()

    fields = [
        ("opcode", opcode, 6),
        ("rs ($s0)", rs, 5),
        ("rt ($s1)", rt, 5),
        ("rd ($t2)", rd, 5),
        ("shamt", shamt, 5),
        ("funct (ADD)", funct, 6),
    ]

    binary_parts = []
    print(f"  {'Field':<14s} {'Decimal':>8s} {'Binary':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8}")
    for name, value, bits in fields:
        bin_str = format(value, f'0{bits}b')
        binary_parts.append(bin_str)
        print(f"  {name:<14s} {value:>8d} {bin_str:>8s}")

    full_binary = ''.join(binary_parts)
    hex_val = int(full_binary, 2)
    print(f"\n  Full binary: {full_binary}")
    print(f"  Hex:         0x{hex_val:08X}")

    # Verify
    expected = 0x02115020
    assert hex_val == expected, f"Expected 0x{expected:08X}, got 0x{hex_val:08X}"
    print(f"  Verified:    0x{expected:08X}")


def exercise_5():
    """
    Encode LW $t0, 200($s2) as a 32-bit binary number.
    $t0=8, $s2=18, LW opcode=0x23
    """
    # I-type format: opcode | rs | rt | immediate
    opcode = 0x23  # LW opcode
    rs = 18        # $s2 = register 18 (base register)
    rt = 8         # $t0 = register 8 (destination)
    imm = 200      # Offset

    print("Encoding LW $t0, 200($s2):")
    print(f"  Format: I-type")
    print(f"  LW $t0, 200($s2) → $t0 = Memory[$s2 + 200]")
    print()

    fields = [
        ("opcode (LW)", opcode, 6),
        ("rs ($s2)", rs, 5),
        ("rt ($t0)", rt, 5),
        ("immediate", imm, 16),
    ]

    binary_parts = []
    print(f"  {'Field':<16s} {'Decimal':>8s} {'Binary':>18s}")
    print(f"  {'-'*16} {'-'*8} {'-'*18}")
    for name, value, bits in fields:
        bin_str = format(value, f'0{bits}b')
        binary_parts.append(bin_str)
        print(f"  {name:<16s} {value:>8d} {bin_str:>18s}")

    full_binary = ''.join(binary_parts)
    hex_val = int(full_binary, 2)
    print(f"\n  Full binary: {full_binary}")
    print(f"  Hex:         0x{hex_val:08X}")

    expected = 0x8E4800C8
    assert hex_val == expected, f"Expected 0x{expected:08X}, got 0x{hex_val:08X}"
    print(f"  Verified:    0x{expected:08X}")


def exercise_6():
    """
    Identify addressing modes:
    (a) ADDI $t0,$t1,100  (b) LW $t0,0($s0)  (c) ADD $t0,$t1,$t2
    (d) J 0x00400000      (e) BEQ $t0,$t1,label
    """
    modes = [
        ("ADDI $t0, $t1, 100",  "Immediate Addressing",
         "The constant 100 is encoded directly in the instruction"),
        ("LW $t0, 0($s0)",      "Base/Displacement Addressing",
         "Effective address = $s0 + 0 (base register + offset)"),
        ("ADD $t0, $t1, $t2",   "Register Addressing",
         "Both operands come from registers; no memory or immediate"),
        ("J 0x00400000",         "Pseudo-Direct Addressing",
         "26-bit address field, combined with PC upper bits: {PC[31:28], addr, 00}"),
        ("BEQ $t0, $t1, label", "PC-Relative Addressing",
         "Target = PC + 4 + (offset << 2); offset relative to current PC"),
    ]

    print("Addressing mode identification:")
    for instr, mode, explanation in modes:
        print(f"\n  {instr}")
        print(f"    Mode: {mode}")
        print(f"    Why:  {explanation}")


def exercise_7():
    """
    Explain x86 effective address calculation for MOV EAX, [EBX + ECX*4 + 100].
    """
    print("x86 Effective Address: MOV EAX, [EBX + ECX*4 + 100]")
    print()
    print("  General form: [Base + Index * Scale + Displacement]")
    print()

    components = [
        ("EBX",  "Base register",   "Starting address (e.g., array base)"),
        ("ECX",  "Index register",  "Element index within array"),
        ("4",    "Scale factor",    "Element size in bytes (1, 2, 4, or 8)"),
        ("100",  "Displacement",    "Constant offset (e.g., struct field offset)"),
    ]

    print(f"  {'Component':>6s}  {'Role':<18s}  Description")
    print(f"  {'-'*6}  {'-'*18}  {'-'*45}")
    for comp, role, desc in components:
        print(f"  {comp:>6s}  {role:<18s}  {desc}")

    # Example calculation
    ebx = 0x1000  # Array base
    ecx = 5       # Index
    scale = 4     # sizeof(int)
    disp = 100    # Offset

    eff_addr = ebx + ecx * scale + disp
    print(f"\n  Example calculation:")
    print(f"    EBX = 0x{ebx:04X} (base), ECX = {ecx} (index)")
    print(f"    EA = 0x{ebx:04X} + {ecx} * {scale} + {disp}")
    print(f"       = 0x{ebx:04X} + {ecx * scale} + {disp}")
    print(f"       = {ebx} + {ecx * scale} + {disp} = {eff_addr} = 0x{eff_addr:04X}")
    print()
    print("  Use case: arr[i].field where")
    print("    EBX = &arr, ECX = i, Scale = sizeof(element), Displacement = field offset")


def exercise_8():
    """Explain why modern x86 processors use RISC-style micro-ops internally."""
    print("Why x86 CPUs Decode to RISC-Style Micro-ops:")
    print()

    reasons = [
        {
            "reason": "Pipeline Optimization",
            "detail": (
                "RISC-like micro-ops have uniform format and execution time, "
                "making them ideal for deep, high-frequency pipelines. Variable-length "
                "x86 instructions would create pipeline bubbles and stalls."
            ),
        },
        {
            "reason": "Superscalar Execution",
            "detail": (
                "Simple micro-ops can be issued to multiple execution units simultaneously. "
                "A complex x86 instruction like REP MOVSB becomes many simple micro-ops "
                "that can execute in parallel across multiple ALUs."
            ),
        },
        {
            "reason": "Out-of-Order Execution",
            "detail": (
                "Uniform micro-ops are easier to reorder, rename, and dispatch "
                "in the out-of-order engine. Tomasulo's algorithm and the ROB "
                "work best with simple, fixed-latency operations."
            ),
        },
        {
            "reason": "Backward Compatibility",
            "detail": (
                "The x86 ISA has decades of legacy instructions. By decoding to "
                "micro-ops, Intel/AMD maintain full ISA compatibility externally "
                "while optimizing the internal execution engine independently."
            ),
        },
    ]

    for i, r in enumerate(reasons, 1):
        print(f"  {i}. {r['reason']}")
        print(f"     {r['detail']}")
        print()

    print("  Example: x86 ADD [mem], reg →")
    print("    Micro-op 1: LOAD tmp, [mem]    (memory read)")
    print("    Micro-op 2: ADD tmp, tmp, reg   (arithmetic)")
    print("    Micro-op 3: STORE [mem], tmp    (memory write)")
    print("  These three micro-ops can be pipelined and reordered.")


def exercise_9():
    """Describe 3 advantages of RISC-V as an open-source ISA."""
    advantages = [
        {
            "advantage": "Royalty-Free / No License Costs",
            "detail": (
                "Unlike ARM (requires licensing from ARM Holdings) or x86 (proprietary to "
                "Intel/AMD), RISC-V is completely free to implement. Companies like SiFive, "
                "Alibaba (Xuantie), and NVIDIA use RISC-V without licensing fees."
            ),
        },
        {
            "advantage": "Customizable / Extensible",
            "detail": (
                "RISC-V has a modular design: base integer ISA (RV32I/RV64I) plus optional "
                "extensions (M=multiply, A=atomic, F=float, V=vector, etc.). Designers can "
                "add custom extensions for domain-specific acceleration (AI, crypto, DSP) "
                "without violating the standard."
            ),
        },
        {
            "advantage": "Academic and Research Value",
            "detail": (
                "Open specification enables universities to teach real ISA design. "
                "Students can study, modify, and implement actual CPUs. Tools like "
                "CHISEL and open-source cores (BOOM, Rocket) make research accessible. "
                "No vendor lock-in encourages innovation and experimentation."
            ),
        },
    ]

    print("Three advantages of RISC-V as an open-source ISA:")
    for i, a in enumerate(advantages, 1):
        print(f"\n  {i}. {a['advantage']}")
        print(f"     {a['detail']}")


def exercise_10():
    """
    Convert C code to MIPS assembly:
    int a = 10; int b = 20; int c = a + b;
    """
    print("C to MIPS Assembly Translation:")
    print()
    print("  C code:")
    print("    int a = 10;")
    print("    int b = 20;")
    print("    int c = a + b;")
    print()

    print("  MIPS Assembly (using saved registers):")
    assembly = [
        ("li   $s0, 10",       "a = 10  (load immediate 10 into $s0)"),
        ("li   $s1, 20",       "b = 20  (load immediate 20 into $s1)"),
        ("add  $s2, $s0, $s1", "c = a + b  ($s2 = $s0 + $s1 = 30)"),
    ]
    for instr, comment in assembly:
        print(f"    {instr:<24s} # {comment}")

    print()
    print("  Note: 'li' (load immediate) is a pseudo-instruction.")
    print("  The assembler expands it to real instructions:")
    print("    li $s0, 10  →  ori $s0, $zero, 10  (for small values)")
    print("    li $s0, 100000  →  lui $s0, upper16; ori $s0, $s0, lower16")
    print()

    # Simulate execution
    regs = {"$s0": 0, "$s1": 0, "$s2": 0}
    print("  Simulated execution:")
    regs["$s0"] = 10
    print(f"    li $s0, 10   → $s0 = {regs['$s0']}")
    regs["$s1"] = 20
    print(f"    li $s1, 20   → $s1 = {regs['$s1']}")
    regs["$s2"] = regs["$s0"] + regs["$s1"]
    print(f"    add $s2, $s0, $s1 → $s2 = {regs['$s0']} + {regs['$s1']} = {regs['$s2']}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: ISA Definitions", exercise_1),
        ("Exercise 2: CISC vs RISC", exercise_2),
        ("Exercise 3: MIPS Instruction Formats", exercise_3),
        ("Exercise 4: Encode ADD Instruction", exercise_4),
        ("Exercise 5: Encode LW Instruction", exercise_5),
        ("Exercise 6: Addressing Modes", exercise_6),
        ("Exercise 7: x86 Effective Address", exercise_7),
        ("Exercise 8: x86 Micro-ops", exercise_8),
        ("Exercise 9: RISC-V Advantages", exercise_9),
        ("Exercise 10: C to MIPS Assembly", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

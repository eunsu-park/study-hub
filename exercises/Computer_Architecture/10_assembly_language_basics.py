"""
Exercises for Lesson 10: Assembly Language Basics
Topic: Computer_Architecture

Solutions to practice problems covering assembly-machine code relationship,
x86/ARM registers, Intel vs AT&T syntax, instruction tracing,
C-to-assembly conversion, and addressing modes.
"""


def exercise_1():
    """Explain the relationship between assembly language and machine code."""
    print("Assembly Language vs Machine Code:")
    print()
    print("  Machine Code: Raw binary (0s and 1s) that the CPU directly executes.")
    print("    Example: 00000010000100010101000000100000")
    print()
    print("  Assembly Language: Human-readable mnemonics that represent machine code.")
    print("    Example: ADD $t2, $s0, $s1")
    print()
    print("  Relationship:")
    print("    - Assembly has a near 1:1 mapping to machine code")
    print("    - The assembler translates assembly → machine code")
    print("    - Each mnemonic (ADD, MOV, LW) maps to a specific opcode")
    print("    - Pseudo-instructions (LI, LA) may expand to multiple machine instructions")
    print()
    print("  Translation chain:")
    print("    C/C++ → [Compiler] → Assembly → [Assembler] → Object Code → [Linker] → Executable")
    print()
    print("  Key differences from high-level languages:")
    print("    - No variables: only registers and memory addresses")
    print("    - No loops/if: only branches and jumps")
    print("    - No types: just raw bit patterns")
    print("    - Architecture-specific: x86 assembly != ARM assembly")


def exercise_2():
    """
    Explain x86 registers: (a) EAX, (b) ESP, (c) EIP/RIP.
    """
    registers = {
        "(a) EAX (Accumulator)": {
            "64-bit": "RAX",
            "32-bit": "EAX",
            "16-bit": "AX",
            "8-bit": "AH (high), AL (low)",
            "purpose": "Primary accumulator for arithmetic results",
            "conventions": "Function return value (EAX in 32-bit, RAX in 64-bit)",
            "history": "Originally the only register for multiply/divide in 8086",
        },
        "(b) ESP (Stack Pointer)": {
            "64-bit": "RSP",
            "32-bit": "ESP",
            "purpose": "Points to the current top of the stack",
            "conventions": "Modified by PUSH, POP, CALL, RET instructions",
            "note": "Stack grows downward (toward lower addresses) on x86",
        },
        "(c) EIP/RIP (Instruction Pointer)": {
            "64-bit": "RIP",
            "32-bit": "EIP",
            "purpose": "Address of the NEXT instruction to execute",
            "conventions": "Modified by jumps, calls, returns, and interrupts",
            "note": "Cannot be directly modified by MOV; only via JMP/CALL/RET",
        },
    }

    print("x86 Register Purposes:")
    for name, info in registers.items():
        print(f"\n  {name}:")
        for key, val in info.items():
            print(f"    {key:<14s}: {val}")


def exercise_3():
    """List 2 major differences between Intel syntax and AT&T syntax."""
    print("Intel vs AT&T Syntax (2 major differences):")
    print()

    differences = [
        {
            "feature": "Operand Order",
            "intel": "destination, source (MOV EAX, 5 → EAX = 5)",
            "att": "source, destination (movl $5, %eax → EAX = 5)",
            "mnemonic": "Intel: 'MOV dst, src'; AT&T: 'mov src, dst'",
        },
        {
            "feature": "Prefixes/Decorators",
            "intel": "No prefixes: MOV EAX, [EBX+4]",
            "att": "% for registers, $ for immediates: movl 4(%ebx), %eax",
            "mnemonic": "AT&T also uses size suffixes: b(byte), w(word), l(long), q(quad)",
        },
    ]

    for d in differences:
        print(f"  {d['feature']}:")
        print(f"    Intel: {d['intel']}")
        print(f"    AT&T:  {d['att']}")
        print(f"    Note:  {d['mnemonic']}")
        print()

    print("  Same instruction in both:")
    print("    Intel: MOV DWORD PTR [EBX+ECX*4+100], EAX")
    print("    AT&T:  movl %eax, 100(%ebx,%ecx,4)")
    print()
    print("  Usage: GCC/GAS defaults to AT&T; NASM/MASM uses Intel.")
    print("  GCC option: -masm=intel to use Intel syntax.")


def exercise_4():
    """
    Trace x86 code execution:
    mov eax, 10; mov ebx, 5; add eax, ebx; shl eax, 1
    """
    print("x86 Code Execution Trace:")
    print()

    # Simulate registers
    regs = {"eax": 0, "ebx": 0}

    instructions = [
        ("mov eax, 10", lambda: (10, None), "eax", "Load immediate 10"),
        ("mov ebx, 5",  lambda: (5, None),  "ebx", "Load immediate 5"),
        ("add eax, ebx", lambda: (regs["eax"] + regs["ebx"], None), "eax", "EAX = EAX + EBX"),
        ("shl eax, 1",  lambda: (regs["eax"] << 1, None), "eax", "Shift left by 1 (multiply by 2)"),
    ]

    print(f"  {'Instruction':<20s} {'Operation':<30s} {'EAX':>6s} {'EBX':>6s}")
    print(f"  {'-'*20} {'-'*30} {'-'*6} {'-'*6}")
    print(f"  {'(initial)':>20s} {'':>30s} {regs['eax']:>6d} {regs['ebx']:>6d}")

    for instr, compute, dest, desc in instructions:
        result, _ = compute()
        regs[dest] = result
        print(f"  {instr:<20s} {desc:<30s} {regs['eax']:>6d} {regs['ebx']:>6d}")

    print(f"\n  Final EAX = {regs['eax']}")
    print(f"  Verification: (10 + 5) << 1 = 15 << 1 = 30")


def exercise_5():
    """
    Convert C abs() function to x86-64 assembly.
    int abs(int x) { if (x < 0) return -x; return x; }
    """
    print("C to x86-64 Assembly: abs(int x)")
    print()
    print("  C code:")
    print("    int abs(int x) {")
    print("        if (x < 0) return -x;")
    print("        return x;")
    print("    }")
    print()

    print("  x86-64 Assembly (System V ABI: arg in EDI, return in EAX):")
    assembly = [
        ("abs:",          "",                         "Function label"),
        ("    mov eax, edi", "eax = x",               "Copy argument to return register"),
        ("    test eax, eax", "set flags based on eax","AND with self (check sign bit)"),
        ("    jns .positive", "if (SF==0) goto",      "Jump if Not Sign (x >= 0)"),
        ("    neg eax",     "eax = -eax",             "Two's complement negation"),
        (".positive:",    "",                          ""),
        ("    ret",         "return eax",              "Return value in EAX"),
    ]

    for instr, operation, comment in assembly:
        if comment:
            print(f"    {instr:<22s} ; {operation:<20s} # {comment}")
        else:
            print(f"    {instr}")

    print()
    print("  Alternative (branchless version using CDQ trick):")
    print("    abs_branchless:")
    print("        mov  eax, edi      ; eax = x")
    print("        cdq                 ; edx = sign-extend of eax (all 0s or all 1s)")
    print("        xor  eax, edx      ; if negative: flip all bits; if positive: no change")
    print("        sub  eax, edx      ; if negative: add 1 (complete two's complement)")
    print("        ret")

    # Simulate
    print("\n  Simulation:")
    for x in [5, -5, 0, -128, 127]:
        result = x if x >= 0 else -x
        print(f"    abs({x:>4d}) = {result}")


def exercise_6():
    """Explain advantages of conditional execution in ARM."""
    print("ARM Conditional Execution Advantages:")
    print()
    print("  ARM (AArch32) allows almost any instruction to be conditionally executed")
    print("  based on condition flags (N, Z, C, V), without requiring a branch.")
    print()

    advantages = [
        ("Avoids Pipeline Flush",
         "Branch misprediction flushes the pipeline (3-15 cycle penalty). "
         "Conditional execution keeps the pipeline full by executing NOPs "
         "for untaken conditions instead of flushing."),
        ("Reduces Code Size",
         "Short if-else blocks can be implemented without branch/label overhead. "
         "Example: abs in 3 instructions vs 5+ with branches."),
        ("No Branch Prediction Needed",
         "For short conditional sequences, the CPU does not need to predict "
         "a branch direction, eliminating misprediction penalty entirely."),
    ]

    for name, detail in advantages:
        print(f"  - {name}")
        print(f"    {detail}")
        print()

    print("  Example: abs(x) in ARM:")
    print("    CMP  R0, #0       ; Compare x with 0")
    print("    RSBLT R0, R0, #0  ; If less than: R0 = 0 - R0 (negate)")
    print("    ; Only 2 instructions, no branch needed!")
    print()
    print("  Note: AArch64 (ARMv8) reduced conditional execution scope,")
    print("  using CSEL/CINC/CNEG instead of full predication.")


def exercise_7():
    """
    Trace ARM code: MOV X0,#10; MOV X1,#3; MUL X0,X0,X1; SUB X0,X0,#5
    """
    print("ARM AArch64 Code Execution Trace:")
    print()

    x0, x1 = 0, 0
    steps = []

    # MOV X0, #10
    x0 = 10
    steps.append(("MOV X0, #10", f"X0 = 10", x0, x1))

    # MOV X1, #3
    x1 = 3
    steps.append(("MOV X1, #3", f"X1 = 3", x0, x1))

    # MUL X0, X0, X1
    x0 = x0 * x1
    steps.append(("MUL X0, X0, X1", f"X0 = 10 * 3 = {x0}", x0, x1))

    # SUB X0, X0, #5
    x0 = x0 - 5
    steps.append(("SUB X0, X0, #5", f"X0 = 30 - 5 = {x0}", x0, x1))

    print(f"  {'Instruction':<22s} {'Operation':<20s} {'X0':>4s} {'X1':>4s}")
    print(f"  {'-'*22} {'-'*20} {'-'*4} {'-'*4}")
    print(f"  {'(initial)':<22s} {'':>20s} {0:>4d} {0:>4d}")
    for instr, op, xr0, xr1 in steps:
        print(f"  {instr:<22s} {op:<20s} {xr0:>4d} {xr1:>4d}")

    print(f"\n  Final X0 = {x0}")


def exercise_8():
    """
    Explain x86 addressing mode [EBX + ECX*4 + 100].
    """
    print("x86 Complex Addressing Mode: [EBX + ECX*4 + 100]")
    print()
    print("  Format: [Base + Index * Scale + Displacement]")
    print()

    components = {
        "EBX (Base)":       "Starting address — typically points to the beginning of a data structure",
        "ECX (Index)":      "Offset index — typically an array element counter",
        "4 (Scale)":        "Multiplier for index — 1, 2, 4, or 8 (element size in bytes)",
        "100 (Displacement)":"Fixed offset — typically a struct field offset",
    }

    for comp, meaning in components.items():
        print(f"  {comp:<24s}: {meaning}")

    # Practical example
    print()
    print("  Practical example:")
    print("    struct Employee { char name[100]; int salary; ... };")
    print("    Employee employees[N];")
    print("    // Accessing employees[i].salary:")
    print("    // EBX = &employees[0], ECX = i, Scale = sizeof(Employee),")
    print("    // Displacement = offsetof(Employee, salary) = 100")
    print()
    print("    MOV EAX, [EBX + ECX*4 + 100]")
    print("    ; If sizeof(Employee)=4 (simplified): EAX = employees[ECX].salary")


def exercise_9():
    """
    Explain ARM pre-indexed vs post-indexed addressing.
    """
    print("ARM Pre-indexed vs Post-indexed Addressing:")
    print()

    # Simulate with Python
    class Memory:
        def __init__(self):
            self.data = {i: i * 10 for i in range(0, 200, 4)}

        def load(self, addr):
            return self.data.get(addr, 0)

    mem = Memory()

    # Pre-indexed: LDR X0, [X1, #8]!
    print("  Pre-indexed: LDR X0, [X1, #8]!")
    print("    Step 1: X1 = X1 + 8     (update base FIRST)")
    print("    Step 2: X0 = Memory[X1]  (then load from updated address)")
    print("    The '!' means write-back (X1 is permanently updated)")
    print()
    x1 = 100
    x0 = 0
    print(f"    Before: X1 = {x1}, X0 = {x0}")
    x1 = x1 + 8   # Update first
    x0 = mem.load(x1)
    print(f"    After:  X1 = {x1}, X0 = Memory[{x1}] = {x0}")

    print()

    # Post-indexed: LDR X0, [X1], #8
    print("  Post-indexed: LDR X0, [X1], #8")
    print("    Step 1: X0 = Memory[X1]  (load from CURRENT address)")
    print("    Step 2: X1 = X1 + 8      (then update base)")
    print("    (Write-back is implicit in post-indexed)")
    print()
    x1 = 100
    x0 = 0
    print(f"    Before: X1 = {x1}, X0 = {x0}")
    x0 = mem.load(x1)  # Load from current
    x1 = x1 + 8        # Then update
    print(f"    After:  X0 = Memory[100] = {x0}, X1 = {x1}")

    print()
    print("  Use case: Iterating through arrays")
    print("    Pre:  Address-then-access (useful for skipping headers)")
    print("    Post: Access-then-advance (natural for sequential traversal)")
    print("    Both save an instruction vs separate LDR + ADD")


def exercise_10():
    """
    Convert C factorial to x86-64 assembly.
    int factorial(int n) { int result = 1; for (i=2; i<=n; i++) result *= i; }
    """
    print("C to x86-64 Assembly: factorial(int n)")
    print()
    print("  C code:")
    print("    int factorial(int n) {")
    print("        int result = 1;")
    print("        for (int i = 2; i <= n; i++) {")
    print("            result *= i;")
    print("        }")
    print("        return result;")
    print("    }")
    print()

    print("  x86-64 Assembly (arg n in EDI, return in EAX):")
    assembly = [
        "factorial:",
        "    mov  eax, 1         ; result = 1",
        "    cmp  edi, 1         ; if n <= 1",
        "    jle  .done          ;   return 1",
        "    mov  ecx, 2         ; i = 2",
        ".loop:",
        "    imul eax, ecx       ; result *= i",
        "    inc  ecx            ; i++",
        "    cmp  ecx, edi       ; compare i with n",
        "    jle  .loop          ; if i <= n, continue loop",
        ".done:",
        "    ret                  ; return result in EAX",
    ]

    for line in assembly:
        print(f"    {line}")

    # Simulate
    print("\n  Simulation:")
    for n in [0, 1, 5, 10]:
        result = 1
        for i in range(2, n + 1):
            result *= i
        print(f"    factorial({n:>2d}) = {result}")

    print()
    print("  ARM AArch64 equivalent:")
    arm_assembly = [
        "factorial:",
        "    MOV  W0, #1          ; result = 1",
        "    CMP  W0_arg, #1      ; if n <= 1",
        "    BLE  .done            ;   return 1",
        "    MOV  W1, #2          ; i = 2",
        ".loop:",
        "    MUL  W0, W0, W1     ; result *= i",
        "    ADD  W1, W1, #1     ; i++",
        "    CMP  W1, W0_arg     ; compare i with n",
        "    BLE  .loop           ; if i <= n, continue",
        ".done:",
        "    RET",
    ]
    for line in arm_assembly:
        print(f"    {line}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Assembly vs Machine Code", exercise_1),
        ("Exercise 2: x86 Register Purposes", exercise_2),
        ("Exercise 3: Intel vs AT&T Syntax", exercise_3),
        ("Exercise 4: x86 Code Tracing", exercise_4),
        ("Exercise 5: C abs() to x86-64", exercise_5),
        ("Exercise 6: ARM Conditional Execution", exercise_6),
        ("Exercise 7: ARM Code Tracing", exercise_7),
        ("Exercise 8: x86 Addressing Mode", exercise_8),
        ("Exercise 9: ARM Pre/Post-Indexed Addressing", exercise_9),
        ("Exercise 10: C factorial to Assembly", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

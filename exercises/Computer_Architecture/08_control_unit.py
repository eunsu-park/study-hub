"""
Exercises for Lesson 08: Control Unit
Topic: Computer_Architecture

Solutions to practice problems covering hardwired vs microprogrammed control,
control signal generation, microinstructions, FSM-based multi-cycle control,
and microprogram sequencing.
"""


def exercise_1():
    """Explain three main roles of the control unit."""
    roles = [
        {
            "role": "Instruction Decoding",
            "description": (
                "Analyzes the opcode field of the instruction register (IR) "
                "to identify the instruction type (R-type, I-type, J-type, etc.). "
                "The decoder maps opcode bits to internal control states."
            ),
        },
        {
            "role": "Timing/Sequencing Generation",
            "description": (
                "Determines the correct sequence and timing of operations for "
                "each instruction. In single-cycle: all in one clock. In multi-cycle: "
                "generates state transitions (FSM) to step through IF, ID, EX, MEM, WB."
            ),
        },
        {
            "role": "Control Signal Generation",
            "description": (
                "Produces the specific control signals (RegWrite, ALUSrc, MemRead, etc.) "
                "that drive the datapath components. Each signal enables/disables a "
                "particular datapath element or selects a MUX input."
            ),
        },
    ]

    print("Three main roles of the Control Unit:")
    for i, r in enumerate(roles, 1):
        print(f"\n  {i}. {r['role']}")
        print(f"     {r['description']}")


def exercise_2():
    """Compare hardwired control and microprogrammed control."""
    comparison = [
        ("Implementation",  "Combinational logic circuits",    "Microprogram stored in control ROM"),
        ("Speed",           "Fast (direct logic)",              "Slower (ROM lookup + decode)"),
        ("Flexibility",     "Hard to modify (rewire circuit)",  "Easy to modify (update ROM)"),
        ("Complexity",      "Grows with ISA complexity",        "Handles complex ISAs well"),
        ("Design time",     "Long, error-prone",                "Systematic, easier to debug"),
        ("Cost",            "Lower for simple ISAs",            "Higher (ROM + sequencer)"),
        ("Use cases",       "RISC (ARM, MIPS, RISC-V)",        "CISC (x86 microcode)"),
        ("Modification",    "Requires hardware change",         "ROM/microcode update"),
    ]

    print("Hardwired Control vs Microprogrammed Control:")
    print()
    print(f"  {'Feature':<18s} {'Hardwired':<35s} {'Microprogrammed':<35s}")
    print(f"  {'-'*18} {'-'*35} {'-'*35}")
    for feature, hw, mp in comparison:
        print(f"  {feature:<18s} {hw:<35s} {mp:<35s}")

    print("\n  Modern CPUs often combine both:")
    print("    - x86: Microprogrammed for complex CISC instructions,")
    print("      hardwired fast path for simple (common) instructions")


def exercise_3():
    """
    Explain control signals: RegWrite, ALUSrc, MemtoReg.
    """
    signals = {
        "RegWrite": {
            "purpose": "Controls write-enable of the register file",
            "0": "Register file is read-only (no write occurs)",
            "1": "ALU result or memory data is written to destination register",
            "active_for": "R-type (ADD, SUB), Load (LW), but NOT Store (SW) or Branch (BEQ)",
        },
        "ALUSrc": {
            "purpose": "Selects the second input to the ALU",
            "0": "ALU input B comes from register file (Read Data 2)",
            "1": "ALU input B comes from sign-extended immediate field",
            "active_for": "0 for R-type, 1 for I-type (LW, SW, ADDI)",
        },
        "MemtoReg": {
            "purpose": "Selects what data is written back to the register file",
            "0": "Write ALU result to register",
            "1": "Write memory data (from load) to register",
            "active_for": "0 for R-type (ALU result), 1 for LW (memory data)",
        },
    }

    print("Control Signal Details:")
    for name, info in signals.items():
        print(f"\n  {name}:")
        print(f"    Purpose: {info['purpose']}")
        print(f"    Value 0: {info['0']}")
        print(f"    Value 1: {info['1']}")
        print(f"    Active:  {info['active_for']}")


def exercise_4():
    """
    Determine control signals for ADD $t0, $t1, $t2.
    """
    print("Control signals for ADD $t0, $t1, $t2 (R-type):")
    print()

    # Simulate control unit logic
    opcode = 0b000000  # R-type opcode

    # Control signal truth table for R-type
    signals = {
        "RegDst":   1,    # rd field (not rt)
        "ALUSrc":   0,    # Register (not immediate)
        "MemtoReg": 0,    # ALU result (not memory)
        "RegWrite": 1,    # Write to register
        "MemRead":  0,    # No memory read
        "MemWrite": 0,    # No memory write
        "Branch":   0,    # Not a branch
        "ALUOp":    "10", # R-type (ALU control uses funct field)
    }

    print(f"  Opcode: {format(opcode, '06b')} (R-type)")
    print()
    print(f"  {'Signal':<12s} {'Value':>6s}  Explanation")
    print(f"  {'-'*12} {'-'*6}  {'-'*45}")
    explanations = {
        "RegDst":   "Destination = rd ($t0), not rt",
        "ALUSrc":   "Second ALU input from register ($t2)",
        "MemtoReg": "Write ALU result, not memory data",
        "RegWrite": "Write result to register file",
        "MemRead":  "No memory access needed",
        "MemWrite": "No memory write needed",
        "Branch":   "Not a branch instruction",
        "ALUOp":    "R-type: funct field determines ALU op",
    }
    for signal, value in signals.items():
        print(f"  {signal:<12s} {str(value):>6s}  {explanations[signal]}")


def exercise_5():
    """
    Determine control signals for SW $t0, 100($s0).
    """
    print("Control signals for SW $t0, 100($s0) (Store Word):")
    print()

    signals = {
        "RegDst":   "X",  # Don't care (no register write)
        "ALUSrc":   1,     # Immediate (offset 100)
        "MemtoReg": "X",  # Don't care (no register write)
        "RegWrite": 0,     # No register write
        "MemRead":  0,     # Not reading memory
        "MemWrite": 1,     # Writing to memory
        "Branch":   0,     # Not a branch
        "ALUOp":    "00", # Add (address calculation)
    }

    explanations = {
        "RegDst":   "Don't care: no register is being written",
        "ALUSrc":   "Immediate: sign-extended offset 100",
        "MemtoReg": "Don't care: no register write-back",
        "RegWrite": "Disabled: SW does not write to register file",
        "MemRead":  "Not reading: this is a store, not load",
        "MemWrite": "ENABLED: write $t0's value to Memory[$s0+100]",
        "Branch":   "Not a branch instruction",
        "ALUOp":    "Add: compute effective address $s0 + 100",
    }

    print(f"  {'Signal':<12s} {'Value':>6s}  Explanation")
    print(f"  {'-'*12} {'-'*6}  {'-'*50}")
    for signal, value in signals.items():
        print(f"  {signal:<12s} {str(value):>6s}  {explanations[signal]}")

    print("\n  Datapath flow:")
    print("    1. Fetch SW instruction from Memory[PC]")
    print("    2. Read $s0 (base addr) and $t0 (data to store)")
    print("    3. ALU computes: $s0 + SignExt(100) = effective address")
    print("    4. Write $t0's value to Memory[effective address]")


def exercise_6():
    """Explain the role of the sequencer in microprogrammed control."""
    print("Sequencer in Microprogrammed Control:")
    print()
    print("  The sequencer determines the NEXT microinstruction address.")
    print("  It is the 'program counter' of the microprogram.")
    print()

    print("  Sequencer functions:")
    functions = [
        ("Next sequential", "MicroPC += 1 (continue to next microinstruction)"),
        ("Branch/Jump",     "MicroPC = target address (conditional or unconditional)"),
        ("Dispatch",        "MicroPC = dispatch_table[opcode] (decode opcode to handler)"),
        ("Return to fetch", "MicroPC = 0 (restart instruction cycle for next instruction)"),
    ]
    for func, desc in functions:
        print(f"    {func:<18s}: {desc}")

    print()
    print("  Sequence control field in microinstruction:")
    print("    Encodes which sequencing action to take:")
    print("    00 = next sequential")
    print("    01 = branch (to address in addr field)")
    print("    10 = dispatch (opcode-based lookup)")
    print("    11 = return to fetch (microPC = 0)")


def exercise_7():
    """Compare horizontal and vertical microinstructions."""
    print("Horizontal vs Vertical Microinstructions:")
    print()

    comparison = [
        ("Width",           "Wide (50-200+ bits)",     "Narrow (15-40 bits)"),
        ("Control fields",  "One bit per control line", "Encoded fields (need decoder)"),
        ("Parallelism",     "Multiple ops per cycle",   "One op at a time"),
        ("ROM size",        "Large (wide words)",       "Small (narrow words)"),
        ("Execution speed", "Fast (no decode delay)",   "Slower (decoding overhead)"),
        ("Design",          "More complex to write",    "Easier, like assembly coding"),
    ]

    print(f"  {'Feature':<18s} {'Horizontal':<28s} {'Vertical':<28s}")
    print(f"  {'-'*18} {'-'*28} {'-'*28}")
    for feat, h, v in comparison:
        print(f"  {feat:<18s} {h:<28s} {v:<28s}")

    print()
    print("  Example microinstruction formats:")
    print()
    print("  Horizontal (one bit = one control signal):")
    print("    |RegWrite|ALUSrc|MemRead|MemWrite|Branch|ALUOp1|ALUOp0|...|")
    print("    |   1    |  0   |   0   |   0    |  0   |  1   |  0  |...|")
    print()
    print("  Vertical (encoded fields):")
    print("    |ALU_op(3)|Dest(2)|Src(2)|Seq(2)|Addr(8)|")
    print("    | 010     | 01    | 00   | 00   |00000000|")
    print("    (Needs decoder to expand fields into control signals)")


def exercise_8():
    """
    Write microinstruction sequence for BEQ $t0, $t1, label.
    """
    print("Microinstruction sequence for BEQ $t0, $t1, label:")
    print()

    micro_sequence = [
        {
            "state": "IF (Instruction Fetch)",
            "micro_ops": [
                "MAR <- PC",
                "IR <- Memory[MAR]",
                "PC <- PC + 4",
            ],
            "signals": "IorD=0, MemRead=1, IRWrite=1, PCWrite=1, PCSource=00",
        },
        {
            "state": "ID (Instruction Decode / Register Fetch)",
            "micro_ops": [
                "A <- Regs[$t0]  (rs register)",
                "B <- Regs[$t1]  (rt register)",
                "ALUOut <- PC + (SignExt(offset) << 2)  (branch target)",
            ],
            "signals": "ALUSrcA=0, ALUSrcB=11, ALUOp=00 (add for target calc)",
        },
        {
            "state": "EX (Branch Comparison)",
            "micro_ops": [
                "Compare: A - B (check Zero flag)",
                "If Zero=1: PC <- ALUOut (branch taken)",
                "Else: continue to next instruction (PC already updated)",
            ],
            "signals": "ALUSrcA=1, ALUSrcB=00, ALUOp=01 (subtract), PCWriteCond=1, PCSource=01",
        },
    ]

    for step in micro_sequence:
        print(f"  {step['state']}:")
        for op in step['micro_ops']:
            print(f"    - {op}")
        print(f"    Signals: {step['signals']}")
        print()

    print("  Note: BEQ only takes 3 cycles in multi-cycle CPU (no MEM or WB needed)")
    print("  This is an advantage over single-cycle where BEQ takes the full clock period")


def exercise_9():
    """
    Recommend control method for: (a) simple RISC, (b) complex CISC, (c) updateable.
    """
    scenarios = [
        {
            "scenario": "(a) Simple RISC processor",
            "recommendation": "Hardwired Control",
            "reasons": [
                "RISC has a small, regular instruction set (few opcodes)",
                "Fixed instruction length simplifies decoding logic",
                "Speed is paramount in RISC philosophy",
                "Combinational logic is faster than ROM lookup",
                "Examples: ARM Cortex-M, RISC-V simple cores",
            ],
        },
        {
            "scenario": "(b) Complex CISC processor",
            "recommendation": "Microprogrammed Control",
            "reasons": [
                "CISC has hundreds of instructions with variable encoding",
                "Complex instructions need multi-step microcode sequences",
                "Microcode makes complex instruction implementation manageable",
                "Easier to debug and verify correct behavior",
                "Examples: x86 microcode engine, IBM System/360",
            ],
        },
        {
            "scenario": "(c) When microcode updates are needed",
            "recommendation": "Microprogrammed Control",
            "reasons": [
                "Microcode stored in updateable ROM/SRAM",
                "Can patch bugs without hardware changes",
                "Real example: Intel CPU microcode updates (Spectre/Meltdown)",
                "x86 CPUs load microcode patches from BIOS/OS at boot",
                "Hardwired control would require physical chip replacement",
            ],
        },
    ]

    print("Control method recommendations:")
    for s in scenarios:
        print(f"\n  {s['scenario']}")
        print(f"  Recommendation: {s['recommendation']}")
        for reason in s['reasons']:
            print(f"    - {reason}")


def exercise_10():
    """
    List all FSM states for LW instruction in multi-cycle CPU.
    """
    print("LW instruction FSM states in multi-cycle CPU:")
    print()

    states = [
        {
            "state": "State 0: Instruction Fetch (IF)",
            "operations": "MAR <- PC; IR <- Memory[MAR]; PC <- PC+4",
            "signals": {
                "MemRead": 1, "IorD": 0, "IRWrite": 1,
                "ALUSrcA": 0, "ALUSrcB": "01", "ALUOp": "00",
                "PCWrite": 1, "PCSource": "00",
            },
        },
        {
            "state": "State 1: Instruction Decode (ID)",
            "operations": "A <- Regs[rs]; B <- Regs[rt]; ALUOut <- PC + SignExt(imm)<<2",
            "signals": {
                "ALUSrcA": 0, "ALUSrcB": "11", "ALUOp": "00",
            },
        },
        {
            "state": "State 2: Memory Address Computation",
            "operations": "ALUOut <- A + SignExt(offset)",
            "signals": {
                "ALUSrcA": 1, "ALUSrcB": "10", "ALUOp": "00",
            },
        },
        {
            "state": "State 3: Memory Read",
            "operations": "MDR <- Memory[ALUOut]",
            "signals": {
                "MemRead": 1, "IorD": 1,
            },
        },
        {
            "state": "State 4: Write Back (LW-WB)",
            "operations": "Regs[rt] <- MDR",
            "signals": {
                "RegDst": 0, "MemtoReg": 1, "RegWrite": 1,
            },
        },
    ]

    for s in states:
        print(f"  {s['state']}")
        print(f"    Operations: {s['operations']}")
        print(f"    Active signals:", end="")
        sig_strs = [f"{k}={v}" for k, v in s['signals'].items()]
        print(f" {', '.join(sig_strs)}")
        print()

    print("  Total: 5 clock cycles for LW instruction")
    print("  FSM transition: 0 -> 1 -> 2 -> 3 -> 4 -> 0 (back to fetch)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Control Unit Roles", exercise_1),
        ("Exercise 2: Hardwired vs Microprogrammed", exercise_2),
        ("Exercise 3: Control Signal Details", exercise_3),
        ("Exercise 4: ADD Control Signals", exercise_4),
        ("Exercise 5: SW Control Signals", exercise_5),
        ("Exercise 6: Sequencer Role", exercise_6),
        ("Exercise 7: Horizontal vs Vertical Microinstructions", exercise_7),
        ("Exercise 8: BEQ Microinstruction Sequence", exercise_8),
        ("Exercise 9: Control Method Selection", exercise_9),
        ("Exercise 10: LW FSM States", exercise_10),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

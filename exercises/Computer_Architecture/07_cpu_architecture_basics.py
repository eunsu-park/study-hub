"""
Exercises for Lesson 07: CPU Architecture Basics
Topic: Computer_Architecture

Solutions to practice problems covering CPU components, registers,
control signals, instruction execution stages, and performance analysis
(single-cycle vs multi-cycle CPU).
"""


def exercise_1():
    """List three main components inside a CPU."""
    components = {
        "ALU (Arithmetic Logic Unit)": (
            "Performs all arithmetic (add, subtract, multiply, divide) "
            "and logic (AND, OR, XOR, NOT) operations. Also handles "
            "comparisons and shift operations."
        ),
        "Control Unit (CU)": (
            "Fetches instructions from memory, decodes them, and generates "
            "control signals that orchestrate the datapath. Determines the "
            "sequence of operations for each instruction."
        ),
        "Register File": (
            "High-speed storage within the CPU. Includes general-purpose "
            "registers (data, address), special-purpose registers (PC, IR, "
            "MAR, MBR/MDR), and status registers (flags)."
        ),
    }

    print("Three main CPU components:")
    for i, (name, desc) in enumerate(components.items(), 1):
        print(f"\n  {i}. {name}")
        print(f"     {desc}")


def exercise_2():
    """
    Explain the purpose of special-purpose registers:
    (a) PC, (b) IR, (c) MAR
    """
    registers = {
        "PC (Program Counter)": {
            "purpose": "Stores the memory address of the NEXT instruction to execute",
            "behavior": "Automatically incremented after each fetch (PC += 4 for 32-bit ISA)",
            "modified_by": "Branch/jump instructions, interrupts, exceptions",
        },
        "IR (Instruction Register)": {
            "purpose": "Holds the CURRENT instruction being decoded/executed",
            "behavior": "Loaded during the Fetch stage from memory at address [PC]",
            "modified_by": "Instruction fetch at beginning of each cycle",
        },
        "MAR (Memory Address Register)": {
            "purpose": "Holds the memory address for the current read/write operation",
            "behavior": "Set before any memory access; connected to address bus",
            "modified_by": "Fetch (MAR=PC), Load/Store (MAR=effective address)",
        },
    }

    print("Special-purpose CPU registers:")
    for name, info in registers.items():
        print(f"\n  {name}")
        print(f"    Purpose:     {info['purpose']}")
        print(f"    Behavior:    {info['behavior']}")
        print(f"    Modified by: {info['modified_by']}")


def exercise_3():
    """List three types of operations the ALU can perform."""
    alu_operations = {
        "Arithmetic": ["Addition (ADD)", "Subtraction (SUB)", "Multiplication (MUL)", "Division (DIV)"],
        "Logical": ["AND", "OR", "XOR", "NOT", "Comparison (SLT)"],
        "Shift": ["Logical shift left (SLL)", "Logical shift right (SRL)",
                   "Arithmetic shift right (SRA)", "Rotate"],
    }

    print("ALU Operation Categories:")
    for category, ops in alu_operations.items():
        print(f"\n  {category} Operations:")
        for op in ops:
            print(f"    - {op}")

    # Simulate ALU
    print("\n  ALU Simulation:")

    def alu(a, b, op):
        """Simulate ALU operations."""
        operations = {
            "ADD": a + b,
            "SUB": a - b,
            "AND": a & b,
            "OR":  a | b,
            "XOR": a ^ b,
            "SLT": 1 if a < b else 0,
            "SLL": (a << b) & 0xFFFFFFFF,
            "SRL": (a & 0xFFFFFFFF) >> b,
        }
        return operations.get(op, 0)

    test_a, test_b = 42, 15
    for op in ["ADD", "SUB", "AND", "OR", "XOR", "SLT", "SLL", "SRL"]:
        result = alu(test_a, test_b, op)
        print(f"    ALU({test_a}, {test_b}, {op:>3s}) = {result}")


def exercise_4():
    """
    Determine control signals for ADD $t0, $t1, $t2 (R-type instruction).
    """
    print("Control signals for ADD $t0, $t1, $t2:")
    print("  (R-type instruction: operates on registers only)")
    print()

    signals = {
        "RegDst":   (1, "Use rd field (bits 15:11) as destination register"),
        "ALUSrc":   (0, "Second ALU input from register file (not immediate)"),
        "MemtoReg": (0, "Write ALU result to register (not memory data)"),
        "RegWrite": (1, "ENABLED: write result back to register file"),
        "MemRead":  (0, "No memory read needed"),
        "MemWrite": (0, "No memory write needed"),
        "Branch":   (0, "Not a branch instruction"),
        "ALUOp":    ("10", "R-type: ALU control determined by funct field"),
    }

    print(f"  {'Signal':<12s} {'Value':>6s}  Reason")
    print(f"  {'-'*12} {'-'*6}  {'-'*50}")
    for signal, (value, reason) in signals.items():
        print(f"  {signal:<12s} {str(value):>6s}  {reason}")

    print("\n  Datapath flow:")
    print("    1. PC → Instruction Memory → IR (fetch ADD instruction)")
    print("    2. IR[25:21]=$t1 → Read Register 1 → ALU input A")
    print("    3. IR[20:16]=$t2 → Read Register 2 → ALU input B (ALUSrc=0)")
    print("    4. ALU performs ADD (ALUOp=10 + funct=100000)")
    print("    5. ALU result → Write Data → Register $t0 (RegWrite=1, RegDst=1)")


def exercise_5():
    """
    Explain the 5-stage execution of LW $t0, 100($t1).
    """
    print("5-stage execution of LW $t0, 100($t1):")
    print("  (Load Word: read 4 bytes from Memory[$t1 + 100] into $t0)")
    print()

    # Control signals for LW
    signals = {
        "RegDst": ("0", "Use rt field as destination"),
        "ALUSrc": ("1", "Use sign-extended immediate (100)"),
        "MemtoReg": ("1", "Write memory data to register"),
        "RegWrite": ("1", "Enable register write"),
        "MemRead": ("1", "Enable memory read"),
        "MemWrite": ("0", "No memory write"),
        "Branch": ("0", "Not a branch"),
        "ALUOp": ("00", "Add (for address calculation)"),
    }

    stages = [
        ("IF (Instruction Fetch)",
         "Fetch LW instruction from Memory[PC]. PC = PC + 4.",
         "MAR = PC; IR = Memory[MAR]; PC += 4"),
        ("ID (Instruction Decode)",
         "Read $t1 from register file. Sign-extend immediate 100.",
         "A = Regs[$t1]; Imm = SignExt(100)"),
        ("EX (Execute)",
         "Calculate effective address: $t1 + 100.",
         "ALUOut = A + Imm = $t1 + 100"),
        ("MEM (Memory Access)",
         "Read data from calculated memory address.",
         "MDR = Memory[ALUOut]"),
        ("WB (Write Back)",
         "Write loaded data to destination register $t0.",
         "Regs[$t0] = MDR"),
    ]

    for stage, description, operation in stages:
        print(f"  Stage: {stage}")
        print(f"    Description: {description}")
        print(f"    Operation:   {operation}")
        print()

    print("  Control signals:")
    for signal, (value, reason) in signals.items():
        print(f"    {signal:<12s} = {value:<3s} ({reason})")


def exercise_6():
    """
    Single-cycle CPU performance analysis.
    Stage times: IF=250ps, ID=150ps, EX=200ps, MEM=300ps, WB=100ps
    """
    stages = {"IF": 250, "ID": 150, "EX": 200, "MEM": 300, "WB": 100}

    print("Single-Cycle CPU Performance Analysis:")
    print()
    print("  Stage delays:")
    for stage, delay in stages.items():
        print(f"    {stage}: {delay} ps")

    # (a) Clock period = sum of all stages (worst-case path)
    clock_period = sum(stages.values())
    print(f"\n  (a) Clock period = sum of all stages (every instruction uses all stages)")
    print(f"      = {' + '.join(f'{d}' for d in stages.values())} = {clock_period} ps = {clock_period/1000} ns")

    # (b) Instructions per second
    ips = 1e12 / clock_period  # ps to seconds: 1s = 10^12 ps
    print(f"\n  (b) Instructions per second = 1 / {clock_period} ps")
    print(f"      = {ips:.2e} instructions/second")
    print(f"      = {ips/1e9:.1f} GIPS (Giga Instructions Per Second)")

    print(f"\n  Key limitation: Clock period is determined by the SLOWEST instruction.")
    print(f"  Even simple ALU operations take {clock_period}ps because the clock")
    print(f"  must accommodate the longest path (load instruction).")


def exercise_7():
    """
    Multi-cycle CPU average CPI calculation.
    Load: 30% (5 cycles), Store: 15% (4 cycles),
    R-type: 40% (4 cycles), Branch: 15% (3 cycles)
    """
    instructions = [
        ("Load",   0.30, 5),
        ("Store",  0.15, 4),
        ("R-type", 0.40, 4),
        ("Branch", 0.15, 3),
    ]

    print("Multi-Cycle CPU Average CPI Calculation:")
    print()
    print(f"  {'Type':<10s} {'Freq':>6s} {'Cycles':>8s} {'Contribution':>14s}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*14}")

    avg_cpi = 0
    for instr_type, freq, cycles in instructions:
        contribution = freq * cycles
        avg_cpi += contribution
        print(f"  {instr_type:<10s} {freq:>6.0%} {cycles:>8d} {contribution:>14.2f}")

    print(f"  {'-'*40}")
    print(f"  {'Average CPI':>26s} = {avg_cpi:.2f}")

    print(f"\n  Interpretation: On average, each instruction takes {avg_cpi:.2f} clock cycles.")
    print(f"  This is better than single-cycle for short instructions (Branch: 3 cycles)")
    print(f"  but worse for long ones that still need 5 cycles (Load).")


def exercise_8():
    """Explain three CPU design techniques to solve the Von Neumann bottleneck."""
    techniques = [
        {
            "name": "Cache Memory",
            "description": (
                "Small, fast SRAM placed between CPU and main memory. "
                "Exploits temporal and spatial locality to serve most memory "
                "requests from cache rather than slow DRAM."
            ),
            "impact": "Reduces average memory access time by 10-100x",
        },
        {
            "name": "Pipelining",
            "description": (
                "Overlap multiple instruction phases (IF, ID, EX, MEM, WB). "
                "While one instruction is in Execute, the next is in Decode, "
                "and another is being Fetched. Increases throughput to ~1 IPC."
            ),
            "impact": "Increases throughput by pipeline depth (ideally)",
        },
        {
            "name": "Prefetching / Speculative Execution",
            "description": (
                "CPU predicts what data or instructions will be needed next "
                "and fetches them before they are actually requested. "
                "Hardware prefetchers detect access patterns; branch predictors "
                "speculatively fetch instructions down predicted paths."
            ),
            "impact": "Hides memory latency by overlapping computation and memory access",
        },
    ]

    print("Three techniques to address the Von Neumann bottleneck:")
    for i, tech in enumerate(techniques, 1):
        print(f"\n  {i}. {tech['name']}")
        print(f"     {tech['description']}")
        print(f"     Impact: {tech['impact']}")


def exercise_9():
    """
    Compare single-cycle vs multi-cycle execution time for:
    LW $t0, 0($s0); ADD $t1, $t0, $t2; SW $t1, 4($s0)
    """
    stages = {"IF": 250, "ID": 150, "EX": 200, "MEM": 300, "WB": 100}

    print("Single-cycle vs Multi-cycle comparison:")
    print("  Code: LW $t0, 0($s0)")
    print("        ADD $t1, $t0, $t2")
    print("        SW  $t1, 4($s0)")
    print()

    # Single-cycle: clock period = max path = all stages
    sc_period = sum(stages.values())
    sc_instructions = 3
    sc_total = sc_period * sc_instructions

    print(f"  Single-Cycle CPU:")
    print(f"    Clock period = {sc_period} ps")
    print(f"    Instructions = {sc_instructions}")
    print(f"    Total time = {sc_instructions} × {sc_period} ps = {sc_total} ps = {sc_total/1000:.1f} ns")
    print()

    # Multi-cycle: clock period = max single stage
    mc_period = max(stages.values())
    # LW: 5 cycles, ADD: 4 cycles, SW: 4 cycles
    mc_cycles = {"LW": 5, "ADD": 4, "SW": 4}
    mc_total_cycles = sum(mc_cycles.values())
    mc_total = mc_total_cycles * mc_period

    print(f"  Multi-Cycle CPU:")
    print(f"    Clock period = max(stage delays) = {mc_period} ps")
    for instr, cycles in mc_cycles.items():
        print(f"    {instr}: {cycles} cycles × {mc_period} ps = {cycles * mc_period} ps")
    print(f"    Total cycles = {mc_total_cycles}")
    print(f"    Total time = {mc_total_cycles} × {mc_period} ps = {mc_total} ps = {mc_total/1000:.1f} ns")
    print()

    faster = "Single-cycle" if sc_total < mc_total else "Multi-cycle"
    print(f"  Winner: {faster} is faster for this sequence")
    print(f"    Single-cycle: {sc_total} ps")
    print(f"    Multi-cycle:  {mc_total} ps")
    print(f"    Difference:   {abs(sc_total - mc_total)} ps")
    print()
    print(f"  Note: Multi-cycle advantage appears with many short instructions")
    print(f"  (branches, R-type), while single-cycle wastes time on simple ops.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: CPU Components", exercise_1),
        ("Exercise 2: Special-Purpose Registers", exercise_2),
        ("Exercise 3: ALU Operations", exercise_3),
        ("Exercise 4: R-type Control Signals", exercise_4),
        ("Exercise 5: LW Instruction Execution", exercise_5),
        ("Exercise 6: Single-Cycle Performance", exercise_6),
        ("Exercise 7: Multi-Cycle Average CPI", exercise_7),
        ("Exercise 8: Von Neumann Bottleneck Solutions", exercise_8),
        ("Exercise 9: Single-Cycle vs Multi-Cycle", exercise_9),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

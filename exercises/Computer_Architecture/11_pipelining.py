"""
Exercises for Lesson 11: Pipelining
Topic: Computer_Architecture

Solutions to practice problems covering 5-stage pipeline operation,
data/control/structural hazards, forwarding (bypassing), stalls,
and pipeline performance calculations.
"""


def exercise_1():
    """
    Identify pipeline stages and explain what happens in each stage
    for the instruction: ADD $t0, $t1, $t2
    """
    print("5-Stage Pipeline for ADD $t0, $t1, $t2:")
    print()

    stages = [
        ("IF (Instruction Fetch)",
         "Fetch the ADD instruction from Instruction Memory at address PC. "
         "Increment PC = PC + 4. Store instruction in IF/ID pipeline register."),
        ("ID (Instruction Decode / Register Fetch)",
         "Decode opcode (R-type). Read $t1 and $t2 from register file. "
         "Sign-extend immediate field (not used for R-type). "
         "Store values in ID/EX pipeline register."),
        ("EX (Execute)",
         "ALU computes $t1 + $t2. ALU control derives ADD operation from "
         "ALUOp and funct field. Store result in EX/MEM pipeline register."),
        ("MEM (Memory Access)",
         "No memory operation for ADD (MemRead=0, MemWrite=0). "
         "ALU result passes through to MEM/WB pipeline register."),
        ("WB (Write Back)",
         "Write ALU result back to register file at $t0 (RegWrite=1, RegDst=1). "
         "This is where the result becomes architecturally visible."),
    ]

    for stage, description in stages:
        print(f"  {stage}")
        print(f"    {description}")
        print()

    # Pipeline timing diagram
    print("  Pipeline timing (one instruction):")
    print("  Cycle:  1    2    3    4    5")
    print("  ADD:   [IF] [ID] [EX] [MEM][WB]")


def exercise_2():
    """
    Identify all hazards in the following instruction sequence.
    LW  $t0, 0($s0)
    ADD $t1, $t0, $t2
    SW  $t1, 4($s0)
    """
    print("Hazard Identification:")
    print()
    print("  Instruction sequence:")
    print("    I1: LW  $t0, 0($s0)")
    print("    I2: ADD $t1, $t0, $t2")
    print("    I3: SW  $t1, 4($s0)")
    print()

    hazards = [
        {
            "type": "Data Hazard (RAW - Read After Write)",
            "between": "I1 and I2",
            "detail": ("I2 reads $t0 in ID, but I1 writes $t0 in WB (2 cycles later). "
                        "This is a LOAD-USE hazard — data is not available until after MEM stage."),
            "solution": "Forwarding from MEM/WB to EX + 1 stall cycle (load-use stall)",
        },
        {
            "type": "Data Hazard (RAW - Read After Write)",
            "between": "I2 and I3",
            "detail": ("I3 reads $t1 (to store), but I2 writes $t1 in WB. "
                        "However, the value needed by SW is the data to store, "
                        "read in ID stage."),
            "solution": "Forwarding from EX/MEM or MEM/WB to the appropriate stage",
        },
    ]

    for h in hazards:
        print(f"  Hazard: {h['type']}")
        print(f"  Between: {h['between']}")
        print(f"  Detail: {h['detail']}")
        print(f"  Solution: {h['solution']}")
        print()

    # Pipeline diagram with stall
    print("  Pipeline diagram (with forwarding + 1 stall):")
    print("  Cycle: 1    2    3    4    5    6    7")
    print("  LW:   [IF] [ID] [EX] [MEM][WB]")
    print("  ADD:        [IF] [ID] [**] [EX] [MEM][WB]    (** = stall/bubble)")
    print("  SW:              [IF] [**] [ID] [EX] [MEM][WB]")
    print()
    print("  The 1-cycle stall (bubble) is unavoidable for load-use hazards")
    print("  because the data is only available after the MEM stage of LW.")


def exercise_3():
    """
    Calculate pipeline speedup and throughput.
    Non-pipelined: 1000ps per instruction.
    5-stage pipeline with 200ps per stage.
    """
    non_pipelined_time = 1000  # ps per instruction
    stages = 5
    stage_time = 200  # ps per stage
    pipeline_cycle = stage_time  # Pipeline clock = slowest stage

    print("Pipeline Performance Analysis:")
    print(f"  Non-pipelined: {non_pipelined_time} ps per instruction")
    print(f"  Pipeline: {stages} stages, {stage_time} ps per stage")
    print()

    # Throughput: instructions per time unit
    non_pipe_throughput = 1e12 / non_pipelined_time  # instructions per second
    pipe_throughput = 1e12 / pipeline_cycle

    print(f"  Throughput:")
    print(f"    Non-pipelined: 1 instr / {non_pipelined_time}ps = {non_pipe_throughput:.2e} IPS")
    print(f"    Pipelined:     1 instr / {pipeline_cycle}ps = {pipe_throughput:.2e} IPS")
    print(f"    Speedup = {pipe_throughput / non_pipe_throughput:.1f}x")
    print()

    # Latency vs Throughput
    print(f"  Key distinction:")
    print(f"    Latency (single instruction): {stages * stage_time}ps (pipelined, WORSE)")
    print(f"    Latency (single instruction): {non_pipelined_time}ps (non-pipelined)")
    print(f"    Throughput: {stages}x better with pipelining")
    print()

    # For N instructions
    for n in [1, 10, 100, 1000]:
        non_pipe = n * non_pipelined_time
        pipe = (stages + n - 1) * pipeline_cycle  # Fill + N-1 subsequent
        speedup = non_pipe / pipe
        print(f"    N={n:>4d}: non-pipe={non_pipe:>10.0f}ps, pipe={pipe:>10.0f}ps, speedup={speedup:.2f}x")

    print(f"\n  As N → infinity, speedup approaches {stages}x (ideal)")


def exercise_4():
    """
    Explain forwarding (bypassing) and when it cannot resolve hazards.
    """
    print("Forwarding (Bypassing) Mechanism:")
    print()
    print("  Without forwarding:")
    print("    ADD $t0, $t1, $t2  → writes $t0 in WB (cycle 5)")
    print("    SUB $t3, $t0, $t4  → reads $t0 in ID (cycle 3)")
    print("    Result: $t0 not ready for 2 cycles → 2 stall bubbles")
    print()
    print("  With forwarding:")
    print("    ADD computes result at end of EX stage (cycle 3)")
    print("    Forward EX/MEM result directly to SUB's EX stage input")
    print("    Result: Zero stalls for EX-to-EX forwarding")
    print()

    print("  Forwarding paths (most common):")
    paths = [
        ("EX/MEM → EX", "Forward ALU result to next instruction's ALU input", "0 stalls"),
        ("MEM/WB → EX", "Forward result from 2 cycles ago to ALU input", "0 stalls"),
        ("MEM/WB → ID", "Forward to register read (half-cycle optimization)", "0 stalls"),
    ]
    for path, desc, stalls in paths:
        print(f"    {path:<16s}: {desc} ({stalls})")

    print()
    print("  When forwarding CANNOT help (requires stall):")
    print("    Load-Use Hazard:")
    print("      LW  $t0, 0($s0)  → data available after MEM (cycle 4)")
    print("      ADD $t1, $t0, $t2 → needs $t0 at EX (cycle 3)")
    print("      Data not ready yet! Must insert 1 bubble (stall).")
    print()
    print("    Pipeline with stall:")
    print("    Cycle: 1    2    3    4    5    6")
    print("    LW:   [IF] [ID] [EX] [MEM][WB]")
    print("    ADD:        [IF] [ID] [**] [EX] [MEM][WB]")
    print("           (** = bubble; forward MEM→EX after stall)")


def exercise_5():
    """
    Calculate CPI with branch misprediction penalty.
    Base CPI=1, Branch frequency=20%, Misprediction rate=30%, Penalty=3 cycles.
    """
    base_cpi = 1.0
    branch_freq = 0.20
    mispredict_rate = 0.30
    penalty = 3  # cycles

    print("CPI with Branch Misprediction:")
    print(f"  Base CPI:           {base_cpi}")
    print(f"  Branch frequency:   {branch_freq:.0%}")
    print(f"  Misprediction rate: {mispredict_rate:.0%}")
    print(f"  Penalty:            {penalty} cycles")
    print()

    # CPI = base_CPI + branch_freq * mispredict_rate * penalty
    branch_penalty = branch_freq * mispredict_rate * penalty
    effective_cpi = base_cpi + branch_penalty

    print(f"  CPI = base + freq × mispredict × penalty")
    print(f"      = {base_cpi} + {branch_freq} × {mispredict_rate} × {penalty}")
    print(f"      = {base_cpi} + {branch_penalty}")
    print(f"      = {effective_cpi}")
    print()

    # Impact analysis
    print(f"  Performance loss: {branch_penalty / base_cpi:.0%}")
    print()

    # What if we improve prediction?
    print("  Sensitivity analysis (varying misprediction rate):")
    for mr in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        cpi = base_cpi + branch_freq * mr * penalty
        print(f"    Mispredict={mr:.0%}: CPI={cpi:.2f} (loss={branch_freq * mr * penalty:.0%})")

    print()
    print("  This shows why branch prediction is critical for pipeline performance.")
    print("  Modern CPUs achieve <5% misprediction with sophisticated predictors.")


def exercise_6():
    """
    Pipeline scheduling: reorder instructions to minimize stalls.
    Original: LW $t0, 0($s0); ADD $t1, $t0, $t2; LW $t3, 4($s0); ADD $t4, $t3, $t5
    """
    print("Pipeline Instruction Scheduling:")
    print()
    print("  Original order (2 load-use stalls):")
    original = [
        "LW  $t0, 0($s0)",
        "ADD $t1, $t0, $t2   ← load-use stall (1 bubble)",
        "LW  $t3, 4($s0)",
        "ADD $t4, $t3, $t5   ← load-use stall (1 bubble)",
    ]
    for i, instr in enumerate(original):
        print(f"    {i+1}. {instr}")

    print()
    print("  Pipeline diagram (original, 2 stalls):")
    print("  Cycle: 1    2    3    4    5    6    7    8    9")
    print("  LW t0: [IF] [ID] [EX] [MEM][WB]")
    print("  ADD t1:      [IF] [ID] [**] [EX] [MEM][WB]")
    print("  LW t3:            [IF] [**] [ID] [EX] [MEM][WB]")
    print("  ADD t4:                 [**] [IF] [ID] [**] [EX] [MEM][WB]")
    print("  Total: 10 cycles for 4 instructions (CPI = 2.5)")

    print()
    print("  Reordered (0 stalls — interleave independent LW and ADD):")
    reordered = [
        "LW  $t0, 0($s0)",
        "LW  $t3, 4($s0)    ← independent of $t0, fills the load delay",
        "ADD $t1, $t0, $t2   ← $t0 ready (2 cycles after LW), no stall",
        "ADD $t4, $t3, $t5   ← $t3 ready (2 cycles after LW), no stall",
    ]
    for i, instr in enumerate(reordered):
        print(f"    {i+1}. {instr}")

    print()
    print("  Pipeline diagram (reordered, 0 stalls):")
    print("  Cycle: 1    2    3    4    5    6    7    8")
    print("  LW t0: [IF] [ID] [EX] [MEM][WB]")
    print("  LW t3:      [IF] [ID] [EX] [MEM][WB]")
    print("  ADD t1:           [IF] [ID] [EX] [MEM][WB]   (forward MEM→EX)")
    print("  ADD t4:                [IF] [ID] [EX] [MEM][WB]")
    print("  Total: 8 cycles for 4 instructions (CPI = 2.0)")
    print()
    print("  Improvement: 20% faster (10 → 8 cycles)")
    print("  This is what compilers do with instruction scheduling optimization.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Pipeline Stages for ADD", exercise_1),
        ("Exercise 2: Hazard Identification", exercise_2),
        ("Exercise 3: Pipeline Speedup Calculation", exercise_3),
        ("Exercise 4: Forwarding Mechanism", exercise_4),
        ("Exercise 5: CPI with Branch Misprediction", exercise_5),
        ("Exercise 6: Instruction Scheduling", exercise_6),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

"""
Exercises for Lesson 13: Superscalar and Out-of-Order Execution
Topic: Computer_Architecture

Solutions to practice problems covering ILP, data dependence classification,
register renaming, Tomasulo's algorithm, Reorder Buffer (ROB),
and CDB (Common Data Bus) operation.
"""


def exercise_1():
    """
    Classify data dependences in an instruction sequence:
    I1: ADD R1, R2, R3
    I2: SUB R4, R1, R5
    I3: MUL R1, R6, R7
    I4: DIV R8, R1, R4
    """
    print("Data Dependence Classification:")
    print()
    print("  I1: ADD R1, R2, R3    (R1 = R2 + R3)")
    print("  I2: SUB R4, R1, R5    (R4 = R1 - R5)")
    print("  I3: MUL R1, R6, R7    (R1 = R6 * R7)")
    print("  I4: DIV R8, R1, R4    (R8 = R1 / R4)")
    print()

    deps = [
        {
            "type": "RAW (Read After Write) — True Dependence",
            "pair": "I1 → I2",
            "register": "R1",
            "detail": "I2 reads R1, which I1 writes. I2 must wait for I1's result.",
            "eliminable": False,
        },
        {
            "type": "WAW (Write After Write) — Output Dependence",
            "pair": "I1 → I3",
            "register": "R1",
            "detail": "Both I1 and I3 write R1. I3 must write after I1 to preserve program order.",
            "eliminable": True,
        },
        {
            "type": "WAR (Write After Read) — Anti-Dependence",
            "pair": "I2 → I3",
            "register": "R1",
            "detail": "I2 reads R1, then I3 writes R1. I3 must not overwrite before I2 reads.",
            "eliminable": True,
        },
        {
            "type": "RAW (Read After Write) — True Dependence",
            "pair": "I3 → I4",
            "register": "R1",
            "detail": "I4 reads R1 (from I3's MUL result). Must wait for I3.",
            "eliminable": False,
        },
        {
            "type": "RAW (Read After Write) — True Dependence",
            "pair": "I2 → I4",
            "register": "R4",
            "detail": "I4 reads R4, which I2 writes. I4 must wait for I2.",
            "eliminable": False,
        },
    ]

    for d in deps:
        elim = "Yes (via register renaming)" if d["eliminable"] else "No (true data flow)"
        print(f"  {d['type']}")
        print(f"    {d['pair']} on {d['register']}: {d['detail']}")
        print(f"    Eliminable: {elim}")
        print()


def exercise_2():
    """
    Calculate theoretical IPC (Instructions Per Cycle) for a superscalar processor.
    """
    print("Superscalar IPC Analysis:")
    print()

    # Factors limiting IPC
    issue_width = 4  # Can issue 4 instructions per cycle
    print(f"  Issue width: {issue_width} instructions/cycle")
    print()

    # IPC limiters
    print("  Factors that reduce IPC below issue width:")
    factors = [
        ("True data dependencies",  "RAW hazards force serialization", 0.85),
        ("Branch mispredictions",   "Pipeline flush wastes cycles", 0.90),
        ("Cache misses",            "Stalls waiting for memory", 0.80),
        ("Resource conflicts",      "Not enough functional units", 0.95),
        ("Instruction fetch limits","Fetch bandwidth < issue width", 0.90),
    ]

    effective_utilization = 1.0
    for name, desc, factor in factors:
        effective_utilization *= factor
        print(f"    {name:<28s}: {factor:.0%} utilization ({desc})")

    effective_ipc = issue_width * effective_utilization
    print(f"\n  Effective IPC = {issue_width} x {effective_utilization:.2%} = {effective_ipc:.2f}")
    print(f"  (Typical real-world: 1.5-3.0 IPC on 4-wide superscalar)")

    # Diminishing returns of wider issue
    print(f"\n  Diminishing returns of issue width:")
    for width in [1, 2, 4, 6, 8]:
        # Simplified model: utilization drops with wider issue
        util = effective_utilization ** (width / 4)
        ipc = width * util
        print(f"    {width}-wide: IPC ≈ {ipc:.2f} ({util:.0%} utilization)")


def exercise_3():
    """
    Apply register renaming to eliminate WAW and WAR dependences.
    I1: ADD R1, R2, R3
    I2: SUB R4, R1, R5
    I3: MUL R1, R6, R7
    I4: DIV R8, R1, R4
    """
    print("Register Renaming:")
    print()
    print("  Original code (with false dependences):")
    print("    I1: ADD R1, R2, R3")
    print("    I2: SUB R4, R1, R5")
    print("    I3: MUL R1, R6, R7")
    print("    I4: DIV R8, R1, R4")
    print()
    print("  Dependences before renaming:")
    print("    I1→I2: RAW on R1 (true)")
    print("    I1→I3: WAW on R1 (false — eliminated by renaming)")
    print("    I2→I3: WAR on R1 (false — eliminated by renaming)")
    print("    I3→I4: RAW on R1 (true)")
    print("    I2→I4: RAW on R4 (true)")
    print()

    print("  After register renaming (using physical registers P1-P8):")
    print("    Rename table: R1→P1 (initially)")
    print()
    renamed = [
        ("I1: ADD P1, R2, R3",  "R1 mapped to P1"),
        ("I2: SUB P2, P1, R5",  "R4 mapped to P2; reads P1 from I1"),
        ("I3: MUL P3, R6, R7",  "R1 now mapped to P3 (new physical reg!)"),
        ("I4: DIV P4, P3, P2",  "R8→P4; reads P3 from I3, P2 from I2"),
    ]

    for instr, note in renamed:
        print(f"    {instr:<28s}  ({note})")

    print()
    print("  After renaming:")
    print("    I1→I2: RAW on P1 (preserved — true dependence)")
    print("    I1→I3: NO dependence (I1 writes P1, I3 writes P3)")
    print("    I2→I3: NO dependence (I2 reads P1, I3 writes P3)")
    print("    I3→I4: RAW on P3 (preserved — true dependence)")
    print("    I2→I4: RAW on P2 (preserved — true dependence)")
    print()
    print("  Now I1 and I3 can execute in PARALLEL (no false dependences)")
    print("  Maximum parallelism: I1||I3, then I2, then I4")


def exercise_4():
    """
    Trace Tomasulo's algorithm for a simple instruction sequence.
    """
    print("Tomasulo's Algorithm Trace:")
    print()
    print("  Instructions:")
    print("    I1: MUL F0, F2, F4    (latency: 10 cycles)")
    print("    I2: ADD F6, F0, F8    (latency: 2 cycles, depends on I1)")
    print("    I3: SUB F8, F10, F14  (latency: 2 cycles, independent)")
    print()

    print("  Reservation Stations (simplified):")
    print()
    print("  Cycle 1: Issue I1 to MUL RS")
    print("    MUL RS: [MUL, F2, F4, busy]")
    print("    Register Status: F0 → MUL RS")
    print()
    print("  Cycle 2: Issue I2 to ADD RS (F0 not ready, tag = MUL RS)")
    print("    ADD RS: [ADD, <MUL RS>, F8, waiting]  ← waiting for F0 from MUL")
    print("    Issue I3 to SUB RS (independent)")
    print("    SUB RS: [SUB, F10, F14, busy]")
    print()
    print("  Cycle 4: I3 completes (SUB), broadcasts F8 on CDB")
    print("    Note: ADD RS still waiting for F0 from MUL")
    print()
    print("  Cycle 11: I1 completes (MUL), broadcasts F0 on CDB")
    print("    ADD RS captures F0 value, begins execution")
    print()
    print("  Cycle 13: I2 completes (ADD), broadcasts F6 on CDB")
    print()

    print("  Execution timeline:")
    print(f"  {'Instr':<12s} {'Issue':>6s} {'ExecStart':>10s} {'ExecEnd':>8s} {'WB':>4s}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*4}")
    print(f"  {'MUL F0,F2,F4':<12s} {'1':>6s} {'2':>10s} {'11':>8s} {'12':>4s}")
    print(f"  {'ADD F6,F0,F8':<12s} {'2':>6s} {'12':>10s} {'13':>8s} {'14':>4s}")
    print(f"  {'SUB F8,F10,14':<12s} {'3':>6s} {'4':>10s} {'5':>8s} {'6':>4s}")
    print()
    print("  Key Tomasulo features:")
    print("    1. Distributed reservation stations (no central scoreboard)")
    print("    2. Register renaming via RS tags (eliminates WAR/WAW)")
    print("    3. CDB broadcasts results to ALL waiting RSes simultaneously")
    print("    4. I3 executes out-of-order (before I2) — OoO execution!")


def exercise_5():
    """
    Explain the Reorder Buffer (ROB) and why it is needed.
    """
    print("Reorder Buffer (ROB):")
    print()
    print("  Purpose: Enable speculative out-of-order execution while maintaining")
    print("  the illusion of sequential (in-order) completion.")
    print()

    print("  ROB Structure (circular buffer):")
    print("  ┌─────┬──────────┬────────┬───────┬───────┬──────────┐")
    print("  │ ROB# │ Instr    │ State  │ Dest  │ Value │ Spec?    │")
    print("  ├─────┼──────────┼────────┼───────┼───────┼──────────┤")
    print("  │  1   │ MUL F0,..│ Commit │ F0    │ 42.0  │ No       │")
    print("  │  2   │ ADD F6,..│ Done   │ F6    │ 50.0  │ No       │")
    print("  │  3   │ BEQ ...  │ Done   │ -     │ -     │ Predicted│")
    print("  │  4   │ SUB F8,..│ Exec   │ F8    │ -     │ Yes      │")
    print("  │  5   │ LW F2,.. │ Issue  │ F2    │ -     │ Yes      │")
    print("  └─────┴──────────┴────────┴───────┴───────┴──────────┘")
    print()

    print("  ROB lifecycle:")
    lifecycle = [
        ("Issue",    "Instruction enters ROB tail with unique ROB#"),
        ("Execute",  "Instruction executes out-of-order; result stored in ROB"),
        ("Complete", "Execution finished; result in ROB but NOT in register file"),
        ("Commit",   "Head of ROB retires IN ORDER; writes to register file/memory"),
    ]
    for stage, desc in lifecycle:
        print(f"    {stage:<10s}: {desc}")

    print()
    print("  Why ROB is essential:")
    print("    1. Precise exceptions: Only committed state is architecturally visible")
    print("    2. Branch misprediction recovery: Flush speculative entries from ROB")
    print("    3. In-order completion: Results appear in program order despite OoO execution")
    print("    4. Register reclamation: Old physical regs freed when ROB entry commits")


def exercise_6():
    """
    Explain Common Data Bus (CDB) operation.
    """
    print("Common Data Bus (CDB) in Tomasulo's Algorithm:")
    print()
    print("  The CDB is a broadcast bus that distributes execution results")
    print("  to all reservation stations and the ROB simultaneously.")
    print()

    print("  CDB Operation:")
    print("    1. Functional unit completes execution")
    print("    2. Result + tag (RS#/ROB#) placed on CDB")
    print("    3. ALL reservation stations compare their source tags with CDB tag")
    print("    4. Matching entries capture the value (operand is now ready)")
    print("    5. ROB entry updated with the computed value")
    print("    6. Register file updated (if committing)")
    print()

    # Simulate CDB broadcast
    print("  CDB Broadcast Simulation:")
    print("  Suppose MUL completes: Result=42.0, Tag=ROB#3")
    print()

    waiting_rs = [
        {"name": "ADD RS1", "src1": ("ROB#3", None), "src2": ("ready", 10.0), "waiting": True},
        {"name": "SUB RS2", "src1": ("ready", 5.0),  "src2": ("ROB#3", None), "waiting": True},
        {"name": "DIV RS3", "src1": ("ROB#5", None), "src2": ("ready", 2.0),  "waiting": True},
    ]

    broadcast_tag = "ROB#3"
    broadcast_value = 42.0

    print(f"  Before broadcast (Tag={broadcast_tag}, Value={broadcast_value}):")
    for rs in waiting_rs:
        s1_str = f"{rs['src1'][0]}={rs['src1'][1]}" if rs['src1'][1] else rs['src1'][0]
        s2_str = f"{rs['src2'][0]}={rs['src2'][1]}" if rs['src2'][1] else rs['src2'][0]
        print(f"    {rs['name']}: src1={s1_str}, src2={s2_str}")

    print(f"\n  CDB broadcasts: Tag={broadcast_tag}, Value={broadcast_value}")

    for rs in waiting_rs:
        if rs['src1'][0] == broadcast_tag:
            rs['src1'] = ("ready", broadcast_value)
            print(f"    {rs['name']}: src1 CAPTURED value {broadcast_value}")
        if rs['src2'][0] == broadcast_tag:
            rs['src2'] = ("ready", broadcast_value)
            print(f"    {rs['name']}: src2 CAPTURED value {broadcast_value}")

    print(f"\n  After broadcast:")
    for rs in waiting_rs:
        s1_ready = rs['src1'][1] is not None
        s2_ready = rs['src2'][1] is not None
        can_execute = s1_ready and s2_ready
        status = "READY to execute" if can_execute else "still waiting"
        print(f"    {rs['name']}: {status}")


def exercise_7():
    """
    Compare in-order vs out-of-order execution performance.
    """
    print("In-Order vs Out-of-Order Execution Performance:")
    print()

    instructions = [
        ("I1: LW  R1, 0(R10)",   3, [],     "Cache hit: 3 cycles"),
        ("I2: ADD R2, R1, R3",    1, ["I1"], "Depends on I1 (R1)"),
        ("I3: LW  R4, 0(R11)",  20, [],     "Cache MISS: 20 cycles"),
        ("I4: MUL R5, R6, R7",   4, [],     "Independent of I3"),
        ("I5: ADD R8, R4, R5",    1, ["I3", "I4"], "Depends on I3, I4"),
    ]

    print("  Instructions:")
    for instr, lat, deps, note in instructions:
        dep_str = f" (depends on {','.join(deps)})" if deps else " (independent)"
        print(f"    {instr:<24s} latency={lat:>2d} {dep_str} [{note}]")

    # In-order execution
    print("\n  In-order execution timeline:")
    in_order_time = {}
    current_cycle = 1
    for instr, lat, deps, _ in instructions:
        name = instr.split(":")[0].strip()
        # Must wait for all dependencies
        start = current_cycle
        for dep in deps:
            if dep in in_order_time:
                start = max(start, in_order_time[dep][1] + 1)
        end = start + lat - 1
        in_order_time[name] = (start, end)
        current_cycle = end + 1  # In-order: can't start next until this finishes

    for name, (start, end) in in_order_time.items():
        bar = "." * (start - 1) + "#" * (end - start + 1)
        print(f"    {name}: cycle {start:>2d}-{end:>2d}  {bar}")
    in_order_total = max(end for _, end in in_order_time.values())
    print(f"    Total: {in_order_total} cycles")

    # Out-of-order execution
    print("\n  Out-of-order execution timeline:")
    ooo_time = {}
    for instr, lat, deps, _ in instructions:
        name = instr.split(":")[0].strip()
        start = 1
        for dep in deps:
            if dep in ooo_time:
                start = max(start, ooo_time[dep][1] + 1)
        end = start + lat - 1
        ooo_time[name] = (start, end)

    for name, (start, end) in ooo_time.items():
        bar = "." * (start - 1) + "#" * (end - start + 1)
        print(f"    {name}: cycle {start:>2d}-{end:>2d}  {bar}")
    ooo_total = max(end for _, end in ooo_time.values())
    print(f"    Total: {ooo_total} cycles")

    speedup = in_order_total / ooo_total
    print(f"\n  Speedup: {in_order_total}/{ooo_total} = {speedup:.2f}x")
    print(f"  OoO executed I4 (MUL) while waiting for I3 (cache miss)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Data Dependence Classification", exercise_1),
        ("Exercise 2: Superscalar IPC Analysis", exercise_2),
        ("Exercise 3: Register Renaming", exercise_3),
        ("Exercise 4: Tomasulo's Algorithm Trace", exercise_4),
        ("Exercise 5: Reorder Buffer (ROB)", exercise_5),
        ("Exercise 6: Common Data Bus (CDB)", exercise_6),
        ("Exercise 7: In-Order vs Out-of-Order Performance", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

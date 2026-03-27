"""
Exercises for Lesson 11: Contiguous Memory Allocation
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers First-Fit/Best-Fit/Worst-Fit placement, fragmentation analysis,
compaction, variable partitioning traces, and strategy evaluation.
"""


# === Exercise 1: Placement Strategy Simulation ===
# Problem: Simulate First-Fit, Best-Fit, and Worst-Fit for 3 processes.

def exercise_1():
    """Simulate memory allocation with three placement strategies."""
    # Memory layout: 512KB total, OS=64KB
    # Free holes (address, size): after several alloc/dealloc
    initial_holes = [
        (64, 80),    # Hole at 64KB, size 80KB
        (256, 120),  # Hole at 256KB, size 120KB
        (440, 72),   # Hole at 440KB, size 72KB
    ]
    processes = [("P1", 90), ("P2", 70), ("P3", 115)]

    for strategy in ["First-Fit", "Best-Fit", "Worst-Fit"]:
        print(f"\n--- {strategy} ---")
        holes = [list(h) for h in initial_holes]  # mutable copy
        allocations = {}

        for pname, psize in processes:
            if strategy == "First-Fit":
                # Pick first hole that fits
                candidates = [(i, h) for i, h in enumerate(holes) if h[1] >= psize]
                chosen = candidates[0] if candidates else None
            elif strategy == "Best-Fit":
                # Pick smallest hole that fits
                candidates = [(i, h) for i, h in enumerate(holes) if h[1] >= psize]
                if candidates:
                    chosen = min(candidates, key=lambda x: x[1][1])
                else:
                    chosen = None
            else:  # Worst-Fit
                # Pick largest hole
                candidates = [(i, h) for i, h in enumerate(holes) if h[1] >= psize]
                if candidates:
                    chosen = max(candidates, key=lambda x: x[1][1])
                else:
                    chosen = None

            if chosen:
                idx, hole = chosen
                alloc_addr = hole[0]
                allocations[pname] = (alloc_addr, psize)
                remaining = hole[1] - psize
                if remaining > 0:
                    holes[idx] = [alloc_addr + psize, remaining]
                else:
                    holes.pop(idx)
                print(f"  {pname} ({psize}KB) -> Hole at {alloc_addr}KB "
                      f"(size {hole[1]}KB), remaining: {remaining}KB")
            else:
                print(f"  {pname} ({psize}KB) -> ALLOCATION FAILED (no hole large enough)")
                allocations[pname] = None

        # Calculate remaining fragmentation
        total_free = sum(h[1] for h in holes)
        print(f"  Remaining holes: {holes}")
        print(f"  Total external fragmentation: {total_free}KB in {len(holes)} fragment(s)")


# === Exercise 2: Fragmentation Analysis ===
# Problem: Analyze internal fragmentation in fixed partitioning.

def exercise_2():
    """Calculate fragmentation in fixed vs variable partitioning."""
    partition_size = 256  # KB
    num_partitions = 4

    processes = [
        ("P1", 200, 1),
        ("P2", 50, 2),
        ("P3", 255, 3),
        ("P4", 100, 4),
    ]

    print("Fixed Partitioning: 4 partitions of 256KB each\n")
    total_internal_frag = 0
    print(f"  {'Process':<10} {'Size':<10} {'Partition':<12} {'Internal Frag'}")
    print("  " + "-" * 42)
    for pname, psize, part in processes:
        frag = partition_size - psize
        total_internal_frag += frag
        print(f"  {pname:<10} {psize:<10} {part:<12} {frag}KB")

    print(f"\nQ1: Internal fragmentation per partition: see table above")
    print(f"Q2: Total internal fragmentation: {total_internal_frag}KB\n")

    p5_size = 270
    print(f"Q3: P5 ({p5_size}KB) arrives. Can it be loaded?")
    print(f"  NO. Each partition is 256KB, and P5 needs 270KB.")
    print(f"  Even though total free space is {total_internal_frag}KB > {p5_size}KB,")
    print(f"  no single partition can hold P5.")
    print(f"  This is INTERNAL fragmentation: space is wasted inside partitions.\n")

    print(f"Q4: Same 4 processes with variable partitioning:")
    total_used = sum(psize for _, psize, _ in processes)
    total_mem = partition_size * num_partitions
    total_free_var = total_mem - total_used
    print(f"  Total memory: {total_mem}KB")
    print(f"  Total used: {total_used}KB")
    print(f"  Internal fragmentation: 0KB (variable partitions match process size)")
    print(f"  Free space: {total_free_var}KB (one contiguous block)")
    print(f"  P5 ({p5_size}KB) CAN be loaded ({total_free_var}KB >= {p5_size}KB)")


# === Exercise 3: Compaction Cost-Benefit ===
# Problem: Analyze compaction for a fragmented memory layout.

def exercise_3():
    """Analyze compaction cost and benefit."""
    layout = [
        ("OS", 100),
        ("P1", 150),
        ("Hole", 50),
        ("P2", 200),
        ("Hole", 80),
        ("P3", 100),
        ("Hole", 320),
    ]

    print("Current memory layout:")
    addr = 0
    for name, size in layout:
        print(f"  [{name}: {size}KB] @ {addr}KB")
        addr += size

    total_free = sum(size for name, size in layout if name == "Hole")
    p4_size = 350

    print(f"\nQ1: Total free memory: {total_free}KB")
    print(f"  Largest hole: 320KB")
    print(f"  P4 needs {p4_size}KB but largest hole is 320KB < {p4_size}KB")
    print(f"  Despite {total_free}KB total free > {p4_size}KB, P4 CANNOT load.")
    print(f"  This is EXTERNAL fragmentation.\n")

    data_to_move = sum(size for name, size in layout if name.startswith("P"))
    print(f"Q2: Data to move for compaction: {data_to_move}KB")
    print(f"  Move P1 (150KB), P2 (200KB), P3 (100KB) to close all holes.\n")

    bandwidth_gbps = 10  # GB/s
    bandwidth_kbps = bandwidth_gbps * 1024 * 1024  # KB/s
    time_s = data_to_move / bandwidth_kbps
    time_us = time_s * 1e6
    print(f"Q3: Compaction time at {bandwidth_gbps} GB/s:")
    print(f"  {data_to_move}KB / {bandwidth_gbps}GB/s = {time_us:.1f} microseconds")
    print(f"  This is extremely fast (~{time_us:.0f}us). For interactive systems,")
    print(f"  this is acceptable (sub-millisecond). However, processes must be")
    print(f"  paused during compaction, causing a brief stall.\n")

    print("Q4: Memory layout after compaction:")
    addr = 0
    new_layout = [("OS", 100), ("P1", 150), ("P2", 200), ("P3", 100)]
    for name, size in new_layout:
        print(f"  [{name}: {size}KB] @ {addr}KB")
        addr += size
    free_start = addr
    print(f"  [Free: {total_free}KB] @ {free_start}KB")
    print(f"  P4 ({p4_size}KB) can now load in the {total_free}KB contiguous block.\n")

    print("Q5: Hardware requirement for safe compaction:")
    print("  RELOCATION REGISTERS (or base registers) are essential.")
    print("  When processes are moved, their physical addresses change.")
    print("  The OS updates each process's relocation register to the new")
    print("  base address. All logical-to-physical translations use this")
    print("  register, so the process doesn't need to know it was moved.")
    print("  Without relocation hardware, all hard-coded addresses would break.")


# === Exercise 4: Variable Partitioning Trace ===
# Problem: Trace First-Fit allocation through a series of events.

def exercise_4():
    """Trace variable partitioning with First-Fit allocation."""
    total_mem = 1024  # KB
    os_size = 128

    # Track allocated regions as (name, start, size)
    allocated = [("OS", 0, os_size)]

    def find_holes(allocated, total):
        """Find free holes between allocated regions."""
        allocated_sorted = sorted(allocated, key=lambda x: x[1])
        holes = []
        pos = 0
        for name, start, size in allocated_sorted:
            if start > pos:
                holes.append((pos, start - pos))
            pos = start + size
        if pos < total:
            holes.append((pos, total - pos))
        return holes

    def first_fit(holes, size):
        """Find first hole that fits."""
        for addr, hsize in holes:
            if hsize >= size:
                return addr
        return None

    def print_memory(allocated, total):
        allocated_sorted = sorted(allocated, key=lambda x: x[1])
        pos = 0
        parts = []
        for name, start, size in allocated_sorted:
            if start > pos:
                parts.append(f"[Hole: {start - pos}KB @ {pos}]")
            parts.append(f"[{name}: {size}KB @ {start}]")
            pos = start + size
        if pos < total:
            parts.append(f"[Hole: {total - pos}KB @ {pos}]")
        for p in parts:
            print(f"    {p}")

    events = [
        ("t=1", "arrive", "P1", 200),
        ("t=2", "arrive", "P2", 300),
        ("t=3", "arrive", "P3", 150),
        ("t=4", "terminate", "P1", 0),
        ("t=5", "arrive", "P4", 100),
        ("t=6", "terminate", "P3", 0),
        ("t=7", "arrive", "P5", 400),
    ]

    for time, action, pname, psize in events:
        print(f"\n{time}: {pname} {'arrives (' + str(psize) + 'KB)' if action == 'arrive' else 'terminates'}")

        if action == "arrive":
            holes = find_holes(allocated, total_mem)
            addr = first_fit(holes, psize)
            if addr is not None:
                allocated.append((pname, addr, psize))
                print(f"  Allocated at address {addr}KB")
            else:
                print(f"  ALLOCATION FAILED -- no hole of size {psize}KB")
                # Check if compaction would help
                total_free = sum(h[1] for h in holes)
                if total_free >= psize:
                    print(f"  (Total free: {total_free}KB >= {psize}KB -- compaction would help)")
                else:
                    print(f"  (Total free: {total_free}KB < {psize}KB -- even compaction won't help)")
        else:  # terminate
            allocated = [a for a in allocated if a[0] != pname]
            print(f"  {pname} terminated, memory freed")

        print(f"  Memory state:")
        print_memory(allocated, total_mem)


# === Exercise 5: Strategy Evaluation ===
# Problem: Design memory management for a real-time embedded system.

def exercise_5():
    """Evaluate memory management strategies for an embedded system."""
    print("Real-time Embedded System: 256KB RAM")
    print("  8 fixed-size tasks (16KB each), load/unload frequently")
    print("  2 variable-size tasks (20-60KB), run occasionally")
    print("  Allocation must complete in < 10 microseconds")
    print("  No compaction allowed\n")

    fixed_tasks = 8
    fixed_size = 16
    total_fixed = fixed_tasks * fixed_size

    print(f"Q1: Fixed partitioning for the 8 fixed-size tasks")
    print(f"  8 partitions of {fixed_size}KB = {total_fixed}KB")
    print(f"  Internal fragmentation: 0KB (tasks exactly fit partitions)")
    print(f"  Allocation time: O(1) -- just check if the target partition is free")
    print(f"  This is optimal: zero waste, constant-time allocation.\n")

    remaining = 256 - total_fixed
    print(f"Q2: For 2 variable-size tasks (remaining {remaining}KB):")
    print(f"  Best-Fit minimizes fragmentation by choosing the smallest")
    print(f"  sufficient hole, but has O(n) search time.")
    print(f"  For only 2 tasks and limited holes, the search space is tiny")
    print(f"  (at most 2-3 holes), so O(n) is effectively O(1).")
    print(f"  Best-Fit is recommended: minimal fragmentation, predictable time.\n")

    print(f"Q3: Will external fragmentation become a problem?")
    print(f"  For fixed-size tasks: NO. Same-size allocations never fragment.")
    print(f"  For variable-size tasks: POTENTIALLY, but unlikely.")
    print(f"  With only 2 variable tasks in {remaining}KB, there are at most")
    print(f"  2-3 holes. After both tasks terminate, all {remaining}KB is")
    print(f"  contiguous again. Fragmentation is temporary and self-healing")
    print(f"  because the variable tasks are 'occasional' (not continuous).\n")

    print(f"Q4: Data structure for the free list:")
    print(f"  For fixed-size partitions: BITMAP (8 bits, one per partition)")
    print(f"    Search time: O(1) -- check single bit")
    print(f"    Update time: O(1) -- flip single bit")
    print(f"    Memory: 1 byte")
    print(f"  For variable-size area: SORTED LINKED LIST (by address)")
    print(f"    Search time: O(n) where n=number of holes (at most 3)")
    print(f"    Update time: O(n) for coalescing adjacent free blocks")
    print(f"    Merging: Adjacent free blocks are merged on deallocation")
    print(f"    (Buddy system is overkill for 2 tasks; a simple sorted list")
    print(f"    meets the <10us deadline with such a small n)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Placement Strategy Simulation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Fragmentation Analysis ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Compaction Cost-Benefit ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Variable Partitioning Trace ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Strategy Evaluation ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")

"""
Exercises for Lesson 01: Computer System Overview
Topic: Computer_Architecture

Solutions to practice problems covering Von Neumann architecture,
CPU performance metrics (CPI, clock speed), and Amdahl's Law.
"""


def exercise_1():
    """List the five components of Von Neumann architecture."""
    # The Von Neumann architecture (1945) defines a stored-program computer
    # with five fundamental components connected via a shared bus.
    components = [
        "Input devices    - Accept data from the outside world (keyboard, mouse, sensor)",
        "Output devices   - Present results to the outside world (monitor, printer, speaker)",
        "Memory (RAM)     - Stores both instructions and data (unified memory model)",
        "ALU              - Arithmetic Logic Unit: performs computations and logic operations",
        "Control Unit     - Fetches, decodes, and orchestrates instruction execution",
    ]

    print("Five components of Von Neumann architecture:")
    for i, comp in enumerate(components, 1):
        print(f"  {i}. {comp}")

    # Key insight: The Von Neumann bottleneck arises because instructions
    # and data share the same memory bus, limiting throughput.
    print("\nNote: The shared bus between memory and CPU is called the")
    print("'Von Neumann bottleneck' — it limits throughput because instructions")
    print("and data compete for the same bus bandwidth.")


def exercise_2():
    """Identify which memory type is volatile: (a) ROM, (b) RAM, (c) SSD, (d) HDD."""
    memory_types = {
        "ROM":  {"volatile": False, "reason": "Read-Only Memory retains data without power (firmware, BIOS)"},
        "RAM":  {"volatile": True,  "reason": "Random Access Memory loses all data when power is removed"},
        "SSD":  {"volatile": False, "reason": "Solid State Drive uses NAND flash — persistent storage"},
        "HDD":  {"volatile": False, "reason": "Hard Disk Drive uses magnetic platters — persistent storage"},
    }

    print("Memory volatility analysis:")
    for name, info in memory_types.items():
        status = "VOLATILE" if info["volatile"] else "Non-volatile"
        print(f"  ({name:>3s}) [{status:>12s}] — {info['reason']}")

    print("\nAnswer: (b) RAM is volatile memory.")
    print("  - DRAM (Dynamic RAM) needs periodic refresh; loses data without power.")
    print("  - SRAM (Static RAM, used in cache) is also volatile but faster.")


def exercise_3():
    """Identify which part of the CPU decodes instructions."""
    # The instruction cycle: Fetch -> Decode -> Execute -> Writeback
    cpu_parts = {
        "Control Unit": "Fetches instructions from memory, DECODES them, generates control signals",
        "ALU": "Executes arithmetic and logic operations as directed by the Control Unit",
        "Registers": "High-speed storage for operands, results, and CPU state (PC, IR, MAR, MBR)",
    }

    print("CPU components and their roles in instruction processing:")
    for part, role in cpu_parts.items():
        print(f"  {part}: {role}")

    print("\nAnswer: The Control Unit decodes instructions.")
    print("  Specifically, the Instruction Register (IR) holds the fetched instruction,")
    print("  and the Control Unit's decoder circuit interprets the opcode to generate")
    print("  the appropriate control signals for the datapath.")


def exercise_4():
    """
    Calculate execution time for 900 million instructions.
    Given: Clock speed = 3 GHz, CPI = 1.5

    Formula: Execution Time = (Instruction Count * CPI) / Clock Rate
    """
    clock_speed_ghz = 3.0
    clock_speed_hz = clock_speed_ghz * 1e9  # 3 * 10^9 Hz
    cpi = 1.5
    instruction_count = 900e6  # 900 million

    # Total clock cycles needed
    total_cycles = instruction_count * cpi
    print(f"Total clock cycles = {instruction_count:.0e} instructions x {cpi} CPI = {total_cycles:.2e} cycles")

    # Execution time = total cycles / clock rate
    exec_time = total_cycles / clock_speed_hz
    print(f"Execution time = {total_cycles:.2e} cycles / {clock_speed_hz:.2e} Hz = {exec_time} seconds")

    # Verification
    # (9 * 10^8 * 1.5) / (3 * 10^9) = 13.5 * 10^8 / 3 * 10^9 = 0.45 seconds
    print(f"\nAnswer: {exec_time} seconds")

    # Additional metric: MIPS (Million Instructions Per Second)
    mips = (clock_speed_hz / cpi) / 1e6
    print(f"This CPU runs at {mips:.0f} MIPS (Million Instructions Per Second)")


def exercise_5():
    """
    Apply Amdahl's Law for parallel speedup.
    Given: 70% parallelizable, 4 cores.

    Amdahl's Law: Speedup = 1 / ((1 - P) + P/N)
    where P = parallelizable fraction, N = number of processors.
    """
    p = 0.70  # 70% parallelizable
    n = 4     # 4 cores

    serial_fraction = 1 - p
    parallel_fraction_speedup = p / n
    speedup = 1 / (serial_fraction + parallel_fraction_speedup)

    print("Amdahl's Law: Speedup = 1 / ((1 - P) + P/N)")
    print(f"  P = {p} (parallelizable fraction)")
    print(f"  N = {n} (number of processors)")
    print(f"  Serial portion = 1 - {p} = {serial_fraction}")
    print(f"  Parallel portion after speedup = {p}/{n} = {parallel_fraction_speedup:.4f}")
    print(f"  Total = {serial_fraction} + {parallel_fraction_speedup:.4f} = {serial_fraction + parallel_fraction_speedup:.4f}")
    print(f"  Speedup = 1 / {serial_fraction + parallel_fraction_speedup:.4f} = {speedup:.4f}x")

    # Key insight: Even with infinite cores, the maximum speedup is limited by serial fraction
    max_speedup = 1 / serial_fraction
    print(f"\nWith infinite cores: Max speedup = 1 / {serial_fraction} = {max_speedup:.2f}x")
    print("This shows the fundamental limit of Amdahl's Law — the serial fraction")
    print("becomes the bottleneck regardless of how many cores you add.")

    # Show speedup for various core counts
    print("\nSpeedup vs. core count (P = 70%):")
    for cores in [1, 2, 4, 8, 16, 64, 256]:
        s = 1 / ((1 - p) + p / cores)
        print(f"  {cores:>3d} cores: {s:.4f}x speedup ({s/max_speedup*100:.1f}% of theoretical max)")


def exercise_6():
    """Describe three methods to solve the Von Neumann bottleneck."""
    solutions = [
        {
            "method": "Cache Memory",
            "description": (
                "Place small, fast SRAM between CPU and main memory. "
                "Exploits temporal and spatial locality to reduce memory access latency. "
                "Modern CPUs use multi-level caches (L1, L2, L3) with increasing "
                "size and latency at each level."
            ),
            "impact": "Reduces effective memory access time by 10-100x for cached data"
        },
        {
            "method": "Pipelining",
            "description": (
                "Overlap instruction execution stages (Fetch, Decode, Execute, Memory, Writeback). "
                "While one instruction is being executed, the next is being decoded, "
                "and the one after that is being fetched. Ideally achieves 1 instruction "
                "per cycle throughput (IPC) even though each instruction takes multiple cycles."
            ),
            "impact": "Increases throughput by the number of pipeline stages (ideally)"
        },
        {
            "method": "Multiple/Wide Buses",
            "description": (
                "Use separate buses for instructions and data (Harvard architecture concept). "
                "Modern CPUs use split L1 caches (I-cache and D-cache) even though "
                "main memory is unified. Also, wider data buses (64-bit, 128-bit) "
                "transfer more data per cycle."
            ),
            "impact": "Doubles memory bandwidth by allowing simultaneous instruction and data fetch"
        },
    ]

    print("Three methods to address the Von Neumann bottleneck:")
    for i, sol in enumerate(solutions, 1):
        print(f"\n  {i}. {sol['method']}")
        print(f"     Description: {sol['description']}")
        print(f"     Impact: {sol['impact']}")


def exercise_7():
    """Compare Von Neumann vs Harvard architecture."""
    print("Von Neumann Architecture vs Harvard Architecture")
    print("=" * 65)

    comparison = [
        ("Memory model",       "Unified (code + data)",     "Separate instruction & data memory"),
        ("Bus structure",      "Single shared bus",         "Separate instruction & data buses"),
        ("Bandwidth",          "Limited by shared bus",     "Higher (simultaneous I & D access)"),
        ("Flexibility",        "Can treat data as code",    "Strict separation (more secure)"),
        ("Complexity",         "Simpler design",            "More complex (extra buses/memory)"),
        ("Self-modifying code","Possible",                  "Not possible (or restricted)"),
        ("Use cases",          "General-purpose CPUs",      "DSPs, microcontrollers, L1 cache"),
    ]

    print(f"{'Feature':<22s} {'Von Neumann':<28s} {'Harvard':<35s}")
    print("-" * 85)
    for feature, vn, harvard in comparison:
        print(f"{feature:<22s} {vn:<28s} {harvard:<35s}")

    print("\nKey insight: Modern CPUs use a 'Modified Harvard' architecture:")
    print("  - Split L1 cache (separate I-cache and D-cache) = Harvard at cache level")
    print("  - Unified main memory = Von Neumann at memory level")
    print("  This combines Harvard's bandwidth advantage with Von Neumann's flexibility.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Von Neumann Architecture Components", exercise_1),
        ("Exercise 2: Volatile Memory Identification", exercise_2),
        ("Exercise 3: Instruction Decoding", exercise_3),
        ("Exercise 4: CPU Execution Time Calculation", exercise_4),
        ("Exercise 5: Amdahl's Law Parallel Speedup", exercise_5),
        ("Exercise 6: Von Neumann Bottleneck Solutions", exercise_6),
        ("Exercise 7: Von Neumann vs Harvard Architecture", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

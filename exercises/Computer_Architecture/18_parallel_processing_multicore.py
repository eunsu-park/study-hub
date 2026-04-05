"""
Exercises for Lesson 18: Parallel Processing and Multicore
Topic: Computer_Architecture

Solutions to practice problems covering Flynn's taxonomy,
Amdahl's law, Gustafson's law, cache coherence (MESI protocol),
multicore performance, synchronization primitives, and GPU/SIMT concepts.
"""


def exercise_1():
    """
    Classify computing architectures using Flynn's taxonomy:
    SISD, SIMD, MISD, MIMD.
    """
    print("Flynn's Taxonomy:")
    print()

    categories = [
        {
            "name": "SISD (Single Instruction, Single Data)",
            "description": "Traditional sequential processor",
            "examples": "Classic single-core CPU (e.g., Intel 486)",
            "parallelism": "None (scalar execution)",
            "diagram": "  I -> [PU] -> D",
        },
        {
            "name": "SIMD (Single Instruction, Multiple Data)",
            "description": "Same instruction applied to multiple data elements simultaneously",
            "examples": "SSE/AVX (x86), NEON (ARM), GPU shader cores",
            "parallelism": "Data-level parallelism",
            "diagram": "        -> [PU0] -> D0\n  I ----|-> [PU1] -> D1\n        -> [PU2] -> D2",
        },
        {
            "name": "MISD (Multiple Instruction, Single Data)",
            "description": "Multiple instructions operate on same data stream",
            "examples": "Rare; systolic arrays, some fault-tolerant systems",
            "parallelism": "Redundant computation (fault tolerance)",
            "diagram": "  I0 -> [PU0] -|\n  I1 -> [PU1] -|-> D\n  I2 -> [PU2] -|",
        },
        {
            "name": "MIMD (Multiple Instruction, Multiple Data)",
            "description": "Multiple processors execute different instructions on different data",
            "examples": "Multicore CPUs, clusters, distributed systems",
            "parallelism": "Task-level + data-level parallelism",
            "diagram": "  I0 -> [PU0] -> D0\n  I1 -> [PU1] -> D1\n  I2 -> [PU2] -> D2",
        },
    ]

    for cat in categories:
        print(f"  {cat['name']}:")
        print(f"    Description:  {cat['description']}")
        print(f"    Examples:     {cat['examples']}")
        print(f"    Parallelism:  {cat['parallelism']}")
        print()

    # SIMD example: vector addition
    print("  SIMD Example: Vector Addition (4-wide)")
    print()
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]

    # Scalar (SISD)
    print("    Scalar (SISD): 4 ADD instructions, 4 cycles")
    for i in range(4):
        print(f"      cycle {i+1}: c[{i}] = a[{i}] + b[{i}] = {a[i] + b[i]}")

    # Vector (SIMD)
    print()
    print("    Vector (SIMD): 1 VADD instruction, 1 cycle")
    c = [a[i] + b[i] for i in range(4)]
    print(f"      cycle 1: c[0:3] = a[0:3] + b[0:3] = {c}")
    print(f"      Speedup: 4x (ideal)")


def exercise_2():
    """
    Apply Amdahl's Law to calculate speedup with parallelism.
    Speedup = 1 / ((1-P) + P/N)
    where P = parallel fraction, N = number of processors.
    """
    print("Amdahl's Law:")
    print()
    print("  Speedup(N) = 1 / ((1 - P) + P / N)")
    print("  P = parallelizable fraction, N = number of processors")
    print()

    parallel_fractions = [0.50, 0.75, 0.90, 0.95, 0.99]
    core_counts = [1, 2, 4, 8, 16, 64, 256, 1024]

    # Header
    header = f"  {'P':>5s}"
    for n in core_counts:
        header += f" {'N=' + str(n):>7s}"
    header += f" {'N=inf':>7s}"
    print(header)
    print(f"  {'-'*5}", end="")
    for _ in core_counts:
        print(f" {'-'*7}", end="")
    print(f" {'-'*7}")

    for p in parallel_fractions:
        row = f"  {p:>5.0%}"
        for n in core_counts:
            speedup = 1.0 / ((1 - p) + p / n)
            row += f" {speedup:>7.2f}"
        # Infinite processors
        max_speedup = 1.0 / (1 - p)
        row += f" {max_speedup:>7.1f}"
        print(row)

    print()
    print("  Key insights:")
    print("    - With 90% parallel code, max speedup is only 10x (even with 1000 cores)")
    print("    - The serial fraction (1-P) dominates at high core counts")
    print("    - Going from 95% to 99% parallel is more impactful than 2x more cores")
    print()

    # Visualize diminishing returns
    print("  Diminishing returns for P=0.95:")
    p = 0.95
    print(f"    {'Cores':>6s} {'Speedup':>8s} {'Efficiency':>11s} {'Marginal Gain':>14s}")
    print(f"    {'-'*6} {'-'*8} {'-'*11} {'-'*14}")
    prev_speedup = 1.0
    for n in [1, 2, 4, 8, 16, 32, 64]:
        speedup = 1.0 / ((1 - p) + p / n)
        efficiency = speedup / n
        marginal = speedup - prev_speedup
        print(f"    {n:>6d} {speedup:>8.2f} {efficiency:>10.0%} {marginal:>14.2f}")
        prev_speedup = speedup


def exercise_3():
    """
    Apply Gustafson's Law (scaled speedup) and compare with Amdahl's Law.
    Gustafson: Speedup = N - (1-P) * (N-1) = N * P + (1-P)
    """
    print("Gustafson's Law vs Amdahl's Law:")
    print()
    print("  Amdahl:    Fixed problem size, add processors")
    print("  Gustafson: Fixed time, scale problem size with processors")
    print()
    print("  Gustafson's Speedup = N - s*(N-1)")
    print("  where s = serial fraction, N = processors")
    print()

    serial_fraction = 0.05  # 5% serial
    p = 1 - serial_fraction
    print(f"  Serial fraction: {serial_fraction:.0%}, Parallel fraction: {p:.0%}")
    print()

    core_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"  {'Cores':>6s} {'Amdahl':>8s} {'Gustafson':>10s} {'Difference':>11s}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*11}")

    for n in core_counts:
        amdahl = 1.0 / ((1 - p) + p / n)
        gustafson = n - serial_fraction * (n - 1)
        diff = gustafson - amdahl
        print(f"  {n:>6d} {amdahl:>8.2f} {gustafson:>10.2f} {diff:>+11.2f}")

    print()
    print("  Gustafson's Law is more optimistic because it assumes the problem")
    print("  grows with the number of processors (which is realistic for many")
    print("  workloads like scientific simulation and data processing).")
    print("  Amdahl's Law applies when problem size is fixed (e.g., latency-critical tasks).")


def exercise_4():
    """
    Simulate the MESI cache coherence protocol for a dual-core system.
    States: Modified, Exclusive, Shared, Invalid
    """
    print("MESI Cache Coherence Protocol Simulation:")
    print()
    print("  States: M=Modified, E=Exclusive, S=Shared, I=Invalid")
    print()

    class CacheLine:
        def __init__(self):
            self.state = "I"  # Invalid initially

        def __repr__(self):
            return self.state

    # Two cores, each with one cache line for address X
    core0 = CacheLine()
    core1 = CacheLine()

    operations = [
        (0, "READ",  "Core 0 reads X (cache miss -> bus read)"),
        (1, "READ",  "Core 1 reads X (snoop: Core 0 shares)"),
        (0, "WRITE", "Core 0 writes X (invalidate Core 1's copy)"),
        (1, "READ",  "Core 1 reads X (Core 0 flushes modified data)"),
        (1, "WRITE", "Core 1 writes X (invalidate Core 0's copy)"),
        (0, "READ",  "Core 0 reads X (Core 1 flushes modified data)"),
    ]

    print(f"  {'Step':>5s} {'Core':>5s} {'Op':>6s} {'Core0':>6s} {'Core1':>6s} "
          f"{'Bus Action':>20s} {'Description'}")
    print(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*20} {'-'*45}")

    for step, (core_id, op, desc) in enumerate(operations):
        bus_action = ""

        if op == "READ":
            if core_id == 0:
                if core0.state == "I":
                    if core1.state == "M":
                        bus_action = "Flush + Share"
                        core1.state = "S"
                        core0.state = "S"
                    elif core1.state in ("E", "S"):
                        bus_action = "Share"
                        core1.state = "S"
                        core0.state = "S"
                    else:
                        bus_action = "Bus Read"
                        core0.state = "E"
            else:
                if core1.state == "I":
                    if core0.state == "M":
                        bus_action = "Flush + Share"
                        core0.state = "S"
                        core1.state = "S"
                    elif core0.state in ("E", "S"):
                        bus_action = "Share"
                        core0.state = "S"
                        core1.state = "S"
                    else:
                        bus_action = "Bus Read"
                        core1.state = "E"

        elif op == "WRITE":
            if core_id == 0:
                if core0.state in ("E", "M"):
                    bus_action = "Silent upgrade"
                else:
                    bus_action = "Bus Invalidate"
                core0.state = "M"
                if core1.state != "I":
                    core1.state = "I"
                    if "Silent" in bus_action:
                        bus_action = "Bus Invalidate"
            else:
                if core1.state in ("E", "M"):
                    bus_action = "Silent upgrade"
                else:
                    bus_action = "Bus Invalidate"
                core1.state = "M"
                if core0.state != "I":
                    core0.state = "I"
                    if "Silent" in bus_action:
                        bus_action = "Bus Invalidate"

        print(
            f"  {step+1:>5d} {core_id:>5d} {op:>6s} {core0.state:>6s} {core1.state:>6s} "
            f"{bus_action:>20s}  {desc}"
        )

    print()
    print("  MESI ensures all cores see a consistent view of memory.")
    print("  Writes invalidate other copies (write-invalidate protocol).")
    print("  Modified state means only this core has the latest data.")


def exercise_5():
    """
    Demonstrate false sharing: two cores writing to different variables
    that happen to share the same cache line.
    """
    print("False Sharing Problem:")
    print()

    print("  Scenario: Core 0 writes var_A, Core 1 writes var_B")
    print("  Both variables are on the same 64-byte cache line!")
    print()

    cache_line_size = 64
    print(f"  Cache line size: {cache_line_size} bytes")
    print()

    # Simulate false sharing overhead
    print("  Without false sharing (variables on separate cache lines):")
    iterations = 1000
    cycles_per_write = 4  # L1 hit latency
    no_sharing_cycles = iterations * cycles_per_write * 2  # both cores in parallel
    # With parallelism, effective time is just one core's work
    no_sharing_time = iterations * cycles_per_write
    print(f"    {iterations} writes per core, {cycles_per_write} cycles/write")
    print(f"    Time: {no_sharing_time} cycles (cores work in parallel)")
    print()

    print("  With false sharing (both variables on same cache line):")
    invalidation_cycles = 50  # cost of cache line bouncing
    sharing_cycles_per_write = cycles_per_write + invalidation_cycles
    sharing_time = iterations * sharing_cycles_per_write
    print(f"    Each write triggers invalidation: +{invalidation_cycles} cycles")
    print(f"    Time: {sharing_time} cycles")
    print()

    slowdown = sharing_time / no_sharing_time
    print(f"  Slowdown due to false sharing: {slowdown:.1f}x")
    print()

    print("  Solution: Pad structures to cache line boundaries")
    print()
    print("    // Bad: false sharing")
    print("    struct counters {")
    print("        int core0_count;  // same cache line!")
    print("        int core1_count;  // same cache line!")
    print("    };")
    print()
    print("    // Good: no false sharing")
    print("    struct counters {")
    print("        int core0_count;")
    print("        char pad[60];     // padding to separate cache lines")
    print("        int core1_count;")
    print("    };")


def exercise_6():
    """
    Simulate a simple spinlock and show the scalability problem.
    """
    print("Spinlock and Synchronization:")
    print()

    print("  Test-and-Set Spinlock (pseudocode):")
    print()
    print("    lock(L):")
    print("      while test_and_set(L) == 1:  # atomically set L=1, return old value")
    print("          spin                      # busy wait")
    print()
    print("    unlock(L):")
    print("      L = 0                         # release lock")
    print()

    # Simulate lock contention with increasing cores
    print("  Lock Contention Scalability:")
    print()

    critical_section_cycles = 100
    lock_acquire_cycles = 20  # uncontended

    core_counts = [1, 2, 4, 8, 16, 32]
    work_per_core = 1000  # iterations

    print(f"  Critical section: {critical_section_cycles} cycles")
    print(f"  Work per core: {work_per_core} iterations")
    print()
    print(f"  {'Cores':>6s} {'Ideal Time':>11s} {'Serialized':>12s} "
          f"{'Speedup':>8s} {'Efficiency':>11s}")
    print(f"  {'-'*6} {'-'*11} {'-'*12} {'-'*8} {'-'*11}")

    for n in core_counts:
        # Ideal: all cores work in parallel
        ideal_time = work_per_core * critical_section_cycles

        # With lock: critical sections are serialized
        # Each core does work_per_core/n iterations (split work)
        work_each = work_per_core // n if n > 0 else work_per_core

        # Serialized time = sum of all critical sections
        # Plus contention overhead (simplified)
        contention_overhead = (n - 1) * lock_acquire_cycles * work_each
        serial_time = n * work_each * critical_section_cycles + contention_overhead
        parallel_time = work_each * critical_section_cycles + contention_overhead

        # Effective parallel time (dominated by serialization)
        effective_time = max(parallel_time, serial_time // n)

        speedup = ideal_time / effective_time if effective_time > 0 else n
        speedup = min(speedup, n)  # cap at linear speedup
        efficiency = speedup / n

        print(
            f"  {n:>6d} {ideal_time:>11,d} {effective_time:>12,d} "
            f"{speedup:>8.2f} {efficiency:>10.0%}"
        )

    print()
    print("  Lock contention limits scalability. Solutions:")
    print("    1. Fine-grained locking (lock per data element)")
    print("    2. Lock-free data structures (CAS-based)")
    print("    3. Reduce critical section size")
    print("    4. Reader-writer locks (many readers, few writers)")


def exercise_7():
    """
    Compare GPU vs CPU architecture and explain SIMT execution.
    """
    print("GPU Architecture and SIMT Execution:")
    print()

    print("  CPU vs GPU Architecture:")
    print()

    comparison = [
        ("Core count", "4-64 cores", "1000-10000+ cores"),
        ("Clock speed", "3-5 GHz", "1-2 GHz"),
        ("Cache per core", "32-64 KB L1", "16-64 KB L1 (shared)"),
        ("Control logic", "Complex (OoO, speculation)", "Simple (in-order)"),
        ("Thread count", "2 threads/core (SMT)", "32+ threads/core (warp)"),
        ("Optimized for", "Latency (single-thread)", "Throughput (many threads)"),
        ("Memory BW", "50-100 GB/s", "500-3000 GB/s (HBM)"),
        ("Power", "65-250W", "150-700W"),
    ]

    print(f"    {'Feature':<20s} {'CPU':<30s} {'GPU':<30s}")
    print(f"    {'-'*20} {'-'*30} {'-'*30}")
    for feature, cpu, gpu in comparison:
        print(f"    {feature:<20s} {cpu:<30s} {gpu:<30s}")
    print()

    # SIMT execution model
    print("  SIMT (Single Instruction, Multiple Threads):")
    print()
    print("  A warp (NVIDIA) or wavefront (AMD) = group of threads executing")
    print("  the SAME instruction on DIFFERENT data, simultaneously.")
    print()

    warp_size = 32
    print(f"  Warp size: {warp_size} threads")
    print()

    # Demonstrate warp divergence
    print("  Warp Divergence Example:")
    print("    if (threadIdx < 16):")
    print("        path_A()   // threads 0-15")
    print("    else:")
    print("        path_B()   // threads 16-31")
    print()
    print("  Without divergence: 1 pass (all threads active)")
    print("  With divergence:    2 passes (threads masked)")
    print()

    active_per_pass = [16, 16]  # path_A then path_B
    total_active_slots = sum(active_per_pass)
    total_slots = warp_size * len(active_per_pass)
    utilization = total_active_slots / total_slots

    print(f"    Pass 1 (path_A): {active_per_pass[0]}/{warp_size} threads active")
    print(f"    Pass 2 (path_B): {active_per_pass[1]}/{warp_size} threads active")
    print(f"    SIMT utilization: {utilization:.0%} (wastes {1-utilization:.0%} of compute)")
    print()
    print("  Rule: Minimize warp divergence for GPU performance.")
    print("  GPU code should avoid thread-dependent branching within a warp.")


def exercise_8():
    """
    Calculate theoretical FLOPS for CPU vs GPU.
    """
    print("CPU vs GPU FLOPS Comparison:")
    print()

    cpus = [
        {
            "name": "Intel i9-13900K",
            "cores": 8,  # P-cores
            "clock_ghz": 5.8,
            "simd_width": 16,  # AVX-512: 16 single-precision floats
            "fma": 2,  # 2 FMA units
        },
        {
            "name": "Apple M3 Max",
            "cores": 12,
            "clock_ghz": 4.0,
            "simd_width": 4,  # NEON: 4 single-precision floats
            "fma": 2,
        },
    ]

    gpus = [
        {
            "name": "NVIDIA RTX 4090",
            "cuda_cores": 16384,
            "clock_ghz": 2.52,
            "ops_per_core": 2,  # FMA = 2 FLOP
        },
        {
            "name": "NVIDIA H100 (SXM)",
            "cuda_cores": 16896,
            "clock_ghz": 1.83,
            "ops_per_core": 2,
        },
    ]

    print("  CPU Theoretical Peak FLOPS (FP32):")
    print(f"    {'Processor':<22s} {'Cores':>6s} {'GHz':>5s} {'SIMD':>5s} "
          f"{'TFLOPS':>7s}")
    print(f"    {'-'*22} {'-'*6} {'-'*5} {'-'*5} {'-'*7}")

    for cpu in cpus:
        # FLOPS = cores x clock x SIMD_width x 2 (FMA) x FMA_units
        flops = (cpu["cores"] * cpu["clock_ghz"] * 1e9 *
                 cpu["simd_width"] * 2 * cpu["fma"])
        tflops = flops / 1e12
        print(
            f"    {cpu['name']:<22s} {cpu['cores']:>6d} {cpu['clock_ghz']:>5.1f} "
            f"{cpu['simd_width']:>5d} {tflops:>7.2f}"
        )

    print()
    print("  GPU Theoretical Peak FLOPS (FP32):")
    print(f"    {'Processor':<22s} {'Cores':>6s} {'GHz':>5s} "
          f"{'TFLOPS':>7s}")
    print(f"    {'-'*22} {'-'*6} {'-'*5} {'-'*7}")

    for gpu in gpus:
        flops = gpu["cuda_cores"] * gpu["clock_ghz"] * 1e9 * gpu["ops_per_core"]
        tflops = flops / 1e12
        print(
            f"    {gpu['name']:<22s} {gpu['cuda_cores']:>6d} {gpu['clock_ghz']:>5.2f} "
            f"{tflops:>7.1f}"
        )

    print()
    print("  GPUs achieve 10-100x more FLOPS than CPUs by having thousands")
    print("  of simple cores instead of a few complex ones.")
    print("  But GPUs only achieve peak on highly parallel, regular workloads.")
    print("  For serial or branchy code, CPUs are still much faster per-thread.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Flynn's Taxonomy", exercise_1),
        ("Exercise 2: Amdahl's Law", exercise_2),
        ("Exercise 3: Gustafson's Law", exercise_3),
        ("Exercise 4: MESI Cache Coherence Protocol", exercise_4),
        ("Exercise 5: False Sharing", exercise_5),
        ("Exercise 6: Spinlock and Scalability", exercise_6),
        ("Exercise 7: GPU Architecture and SIMT", exercise_7),
        ("Exercise 8: CPU vs GPU FLOPS", exercise_8),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

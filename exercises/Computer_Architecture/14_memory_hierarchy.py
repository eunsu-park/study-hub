"""
Exercises for Lesson 14: Memory Hierarchy
Topic: Computer_Architecture

Solutions to practice problems covering the CPU-memory speed gap,
principle of locality, memory technology comparison, memory levels,
AMAT calculation, and memory performance optimization.
"""


def exercise_1():
    """
    Compare memory technology characteristics across the hierarchy.
    Show speed, capacity, and cost per GB for each level.
    """
    print("Memory Technology Comparison:")
    print()

    levels = [
        {
            "name": "Registers",
            "access_time_ns": 0.25,
            "typical_size": "~1 KB",
            "cost_per_gb": 10_000_000,
            "technology": "Flip-flops (CMOS)",
            "volatile": True,
        },
        {
            "name": "L1 Cache",
            "access_time_ns": 1,
            "typical_size": "32-64 KB",
            "cost_per_gb": 1_000_000,
            "technology": "SRAM",
            "volatile": True,
        },
        {
            "name": "L2 Cache",
            "access_time_ns": 4,
            "typical_size": "256 KB - 1 MB",
            "cost_per_gb": 200_000,
            "technology": "SRAM",
            "volatile": True,
        },
        {
            "name": "L3 Cache",
            "access_time_ns": 12,
            "typical_size": "4-64 MB",
            "cost_per_gb": 50_000,
            "technology": "SRAM",
            "volatile": True,
        },
        {
            "name": "Main Memory",
            "access_time_ns": 60,
            "typical_size": "8-128 GB",
            "cost_per_gb": 5,
            "technology": "DRAM",
            "volatile": True,
        },
        {
            "name": "SSD Storage",
            "access_time_ns": 25_000,
            "typical_size": "256 GB - 4 TB",
            "cost_per_gb": 0.10,
            "technology": "NAND Flash",
            "volatile": False,
        },
        {
            "name": "HDD Storage",
            "access_time_ns": 5_000_000,
            "typical_size": "1-20 TB",
            "cost_per_gb": 0.02,
            "technology": "Magnetic disk",
            "volatile": False,
        },
    ]

    header = (
        f"  {'Level':<14s} {'Access Time':>12s} {'Size':<16s} "
        f"{'$/GB':>12s} {'Tech':<16s}"
    )
    print(header)
    print(f"  {'-'*14} {'-'*12} {'-'*16} {'-'*12} {'-'*16}")

    for lv in levels:
        t = lv["access_time_ns"]
        if t < 1000:
            time_str = f"{t:.2f} ns"
        elif t < 1_000_000:
            time_str = f"{t/1000:.1f} us"
        else:
            time_str = f"{t/1_000_000:.1f} ms"

        cost = lv["cost_per_gb"]
        if cost >= 1000:
            cost_str = f"${cost:,.0f}"
        else:
            cost_str = f"${cost:.2f}"

        print(
            f"  {lv['name']:<14s} {time_str:>12s} {lv['typical_size']:<16s} "
            f"{cost_str:>12s} {lv['technology']:<16s}"
        )

    print()
    print("  Key tradeoff: As speed increases, capacity decreases and cost increases.")
    print("  Registers are ~20,000,000x faster than HDD but ~500,000,000x more expensive.")


def exercise_2():
    """
    Demonstrate temporal and spatial locality using array access patterns.
    """
    print("Principle of Locality:")
    print()

    # Temporal locality: recently accessed data will be accessed again soon
    print("  1. Temporal Locality (recently used data reused soon):")
    print("     Example: Loop variable 'sum' accessed every iteration")
    print()

    data = list(range(100))
    total = 0
    accesses = []
    for i in range(10):
        total += data[i]
        accesses.append(("sum", "temporal"))
        accesses.append((f"data[{i}]", "spatial"))

    print(f"     Code: sum = 0; for i in range(10): sum += data[i]")
    print(f"     'sum' accessed {10} times (temporal locality)")
    print(f"     'data[0..9]' accessed sequentially (spatial locality)")
    print()

    # Spatial locality: nearby data will be accessed soon
    print("  2. Spatial Locality (nearby data accessed together):")
    print("     Example: Sequential array traversal")
    print()

    # Demonstrate with stride access patterns
    print("     Cache behavior with different access strides:")
    cache_line_size = 64  # bytes
    element_size = 8      # bytes (double/int64)
    elements_per_line = cache_line_size // element_size
    array_size = 1024     # elements

    strides = [1, 2, 4, 8, 16, 64]
    print(f"     Cache line: {cache_line_size} bytes = {elements_per_line} elements")
    print(f"     Array: {array_size} elements")
    print()
    print(f"     {'Stride':>7s} {'Elements Accessed':>18s} {'Cache Lines Touched':>20s} {'Utilization':>12s}")
    print(f"     {'-'*7} {'-'*18} {'-'*20} {'-'*12}")

    for stride in strides:
        elements_accessed = array_size // stride
        # Each access touches a cache line; with stride, we may skip lines
        lines_touched = set()
        for i in range(0, array_size, stride):
            line_index = i // elements_per_line
            lines_touched.add(line_index)
        utilization = elements_accessed / (len(lines_touched) * elements_per_line)
        print(
            f"     {stride:>7d} {elements_accessed:>18d} "
            f"{len(lines_touched):>20d} {utilization:>11.0%}"
        )

    print()
    print("  Lesson: Stride-1 access maximizes spatial locality (100% cache utilization).")
    print("  Larger strides waste fetched cache lines, reducing effective bandwidth.")


def exercise_3():
    """
    Calculate Average Memory Access Time (AMAT) for a single-level cache.
    AMAT = Hit Time + Miss Rate x Miss Penalty
    """
    print("Average Memory Access Time (AMAT) Calculation:")
    print()
    print("  Formula: AMAT = Hit Time + Miss Rate x Miss Penalty")
    print()

    # Single-level cache
    print("  Scenario 1: Single-level cache")
    hit_time = 1       # ns
    miss_rate = 0.05    # 5%
    miss_penalty = 100  # ns (go to main memory)

    amat = hit_time + miss_rate * miss_penalty
    print(f"    Hit time:      {hit_time} ns")
    print(f"    Miss rate:     {miss_rate:.0%}")
    print(f"    Miss penalty:  {miss_penalty} ns")
    print(f"    AMAT = {hit_time} + {miss_rate} x {miss_penalty} = {amat:.1f} ns")
    print()

    # Impact of miss rate on AMAT
    print("  Impact of miss rate on AMAT:")
    print(f"    {'Miss Rate':>10s} {'AMAT (ns)':>10s} {'Slowdown vs Ideal':>18s}")
    print(f"    {'-'*10} {'-'*10} {'-'*18}")

    for mr in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        a = hit_time + mr * miss_penalty
        slowdown = a / hit_time
        print(f"    {mr:>10.0%} {a:>10.1f} {slowdown:>17.1f}x")

    print()
    print("  Key insight: Even a 5% miss rate causes a 6x slowdown!")
    print("  Cache design focuses on minimizing miss rate while keeping hit time low.")


def exercise_4():
    """
    Calculate AMAT for a multi-level cache hierarchy (L1 + L2 + L3 + Memory).
    """
    print("Multi-Level Cache AMAT:")
    print()
    print("  For a multi-level hierarchy:")
    print("  AMAT = L1_hit_time + L1_miss_rate x")
    print("         (L2_hit_time + L2_miss_rate x")
    print("          (L3_hit_time + L3_miss_rate x Memory_access_time))")
    print()

    l1_hit_time = 1     # ns
    l1_miss_rate = 0.05  # 5% local miss rate
    l2_hit_time = 4     # ns
    l2_miss_rate = 0.20  # 20% local miss rate (of L1 misses)
    l3_hit_time = 12    # ns
    l3_miss_rate = 0.30  # 30% local miss rate (of L2 misses)
    mem_time = 60       # ns

    # Calculate from innermost level outward
    l3_amat = l3_hit_time + l3_miss_rate * mem_time
    l2_amat = l2_hit_time + l2_miss_rate * l3_amat
    l1_amat = l1_hit_time + l1_miss_rate * l2_amat

    print(f"    L1: hit_time={l1_hit_time} ns, miss_rate={l1_miss_rate:.0%}")
    print(f"    L2: hit_time={l2_hit_time} ns, miss_rate={l2_miss_rate:.0%}")
    print(f"    L3: hit_time={l3_hit_time} ns, miss_rate={l3_miss_rate:.0%}")
    print(f"    Memory: access_time={mem_time} ns")
    print()
    print(f"    L3 AMAT = {l3_hit_time} + {l3_miss_rate} x {mem_time} = {l3_amat:.1f} ns")
    print(f"    L2 AMAT = {l2_hit_time} + {l2_miss_rate} x {l3_amat:.1f} = {l2_amat:.2f} ns")
    print(f"    L1 AMAT = {l1_hit_time} + {l1_miss_rate} x {l2_amat:.2f} = {l1_amat:.3f} ns")
    print()

    # Global miss rates
    global_l2_miss = l1_miss_rate * l2_miss_rate
    global_l3_miss = global_l2_miss * l3_miss_rate
    print(f"    Global miss rates (fraction of all accesses):")
    print(f"      L1 miss: {l1_miss_rate:.2%}")
    print(f"      L2 miss: {global_l2_miss:.2%}")
    print(f"      L3 miss: {global_l3_miss:.3%}")
    print(f"      Only {global_l3_miss:.3%} of accesses go to main memory!")


def exercise_5():
    """
    Demonstrate the memory wall: CPU-memory performance gap over time.
    Calculate effective CPU utilization limited by memory latency.
    """
    print("The Memory Wall Problem:")
    print()

    # Historical CPU vs Memory performance trends
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
    cpu_mhz = [50, 200, 1000, 3000, 3500, 4000, 5000]
    mem_latency_ns = [100, 80, 60, 50, 45, 40, 35]

    print("  CPU Clock vs Memory Latency over Time:")
    print(f"    {'Year':>5s} {'CPU MHz':>10s} {'Mem Lat (ns)':>13s} "
          f"{'Cycles/Miss':>12s} {'Ratio':>8s}")
    print(f"    {'-'*5} {'-'*10} {'-'*13} {'-'*12} {'-'*8}")

    for i in range(len(years)):
        cycle_ns = 1000 / cpu_mhz[i]  # ns per clock cycle
        cycles_per_miss = mem_latency_ns[i] / cycle_ns
        ratio = cpu_mhz[i] / (1000 / mem_latency_ns[i])  # CPU speed / mem speed ratio
        print(
            f"    {years[i]:>5d} {cpu_mhz[i]:>10d} {mem_latency_ns[i]:>13d} "
            f"{cycles_per_miss:>12.0f} {ratio:>7.0f}x"
        )

    print()
    print("  In 1990: A cache miss cost ~5 CPU cycles.")
    print("  In 2020: A cache miss costs ~175 CPU cycles!")
    print("  The memory wall means each miss is increasingly catastrophic.")
    print()

    # CPU utilization model
    print("  CPU utilization limited by memory stalls:")
    miss_rate = 0.02  # 2% miss rate
    print(f"    Assumption: {miss_rate:.0%} L1 cache miss rate")
    print(f"    {'Cycles/Miss':>12s} {'Stall Fraction':>16s} {'CPU Utilization':>16s}")
    print(f"    {'-'*12} {'-'*16} {'-'*16}")

    for cycles in [5, 20, 50, 100, 175]:
        stall_fraction = miss_rate * cycles / (1 + miss_rate * cycles)
        utilization = 1 - stall_fraction
        print(f"    {cycles:>12d} {stall_fraction:>15.0%} {utilization:>15.0%}")


def exercise_6():
    """
    Simulate memory access patterns and measure hit rates for
    different working set sizes relative to cache capacity.
    """
    print("Working Set Size vs Cache Hit Rate:")
    print()

    import random
    random.seed(42)

    cache_size = 64  # cache can hold 64 unique items
    print(f"  Cache capacity: {cache_size} items (fully associative, LRU)")
    print()

    working_set_sizes = [16, 32, 48, 64, 96, 128, 256, 512]
    num_accesses = 10000

    print(f"  Simulation: {num_accesses} random accesses per working set size")
    print(f"  {'Working Set':>12s} {'Hit Rate':>10s} {'Misses':>8s} {'Fits in Cache?':>16s}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*16}")

    for ws_size in working_set_sizes:
        # Simulate LRU cache
        cache = []  # ordered list (MRU at end)
        hits = 0

        for _ in range(num_accesses):
            addr = random.randint(0, ws_size - 1)
            if addr in cache:
                hits += 1
                cache.remove(addr)
                cache.append(addr)  # Move to MRU
            else:
                cache.append(addr)
                if len(cache) > cache_size:
                    cache.pop(0)  # Evict LRU

        hit_rate = hits / num_accesses
        misses = num_accesses - hits
        fits = "Yes" if ws_size <= cache_size else "No"
        print(f"  {ws_size:>12d} {hit_rate:>10.1%} {misses:>8d} {fits:>16s}")

    print()
    print("  Key insight: When working set fits in cache, hit rate is high.")
    print("  As working set exceeds cache capacity, hit rate drops sharply.")
    print("  This is the 'capacity cliff' -- the most important performance cliff.")


def exercise_7():
    """
    Compare row-major vs column-major array traversal
    to illustrate spatial locality impact on cache performance.
    """
    print("Row-Major vs Column-Major Traversal:")
    print()

    rows = 8
    cols = 8
    cache_line_elements = 4  # 4 elements per cache line
    cache_lines = 4          # tiny cache to show effect

    print(f"  Matrix: {rows}x{cols}, cache line = {cache_line_elements} elements, "
          f"cache = {cache_lines} lines")
    print()

    # Row-major storage: element (r,c) at address r*cols + c
    def simulate_traversal(order_name, access_sequence):
        cache = []  # list of cache line tags (FIFO replacement)
        hits = 0
        misses = 0

        for (r, c) in access_sequence:
            # Cache line tag = element address // elements_per_line
            addr = r * cols + c
            tag = addr // cache_line_elements
            if tag in cache:
                hits += 1
            else:
                misses += 1
                if len(cache) >= cache_lines:
                    cache.pop(0)  # FIFO eviction
                cache.append(tag)

        total = hits + misses
        hit_rate = hits / total
        print(f"  {order_name}:")
        print(f"    Hits: {hits}, Misses: {misses}, Hit Rate: {hit_rate:.1%}")
        return hit_rate

    # Row-major traversal: for r in rows: for c in cols
    row_major = [(r, c) for r in range(rows) for c in range(cols)]
    # Column-major traversal: for c in cols: for r in rows
    col_major = [(r, c) for c in range(cols) for r in range(rows)]

    print("  Row-major layout in memory: elements in same row are contiguous")
    print()

    rm_rate = simulate_traversal("Row-major traversal (good locality)", row_major)
    print()
    cm_rate = simulate_traversal("Column-major traversal (poor locality)", col_major)

    print()
    if rm_rate > 0:
        ratio = (1 - cm_rate) / (1 - rm_rate) if rm_rate < 1.0 else float("inf")
        print(f"  Column-major has {ratio:.1f}x more cache misses!")
    print("  This is why matrix traversal order matters enormously in practice.")
    print("  C/C++ use row-major; Fortran uses column-major layout.")


def exercise_8():
    """
    Calculate memory bandwidth requirements for a modern CPU
    and show why bandwidth is as important as latency.
    """
    print("Memory Bandwidth Analysis:")
    print()

    # Modern CPU parameters
    clock_ghz = 4.0
    issue_width = 4
    l1_miss_rate = 0.02
    cache_line_bytes = 64

    instructions_per_sec = clock_ghz * 1e9 * issue_width  # assuming ideal IPC
    memory_accesses_per_sec = instructions_per_sec * 0.30  # ~30% are memory ops
    misses_per_sec = memory_accesses_per_sec * l1_miss_rate
    bandwidth_needed = misses_per_sec * cache_line_bytes  # bytes per second

    bandwidth_gb = bandwidth_needed / 1e9

    print(f"  CPU: {clock_ghz} GHz, {issue_width}-wide issue")
    print(f"  L1 miss rate: {l1_miss_rate:.0%}")
    print(f"  Cache line: {cache_line_bytes} bytes")
    print()
    print(f"  Peak instructions/sec: {instructions_per_sec/1e9:.1f} billion")
    print(f"  Memory accesses/sec:   {memory_accesses_per_sec/1e9:.2f} billion (~30% are memory ops)")
    print(f"  L1 misses/sec:         {misses_per_sec/1e6:.0f} million")
    print(f"  Bandwidth needed:      {bandwidth_gb:.1f} GB/s")
    print()

    # Compare with available memory bandwidth
    print("  Available memory bandwidth (typical):")
    mem_configs = [
        ("DDR4-2400 (single channel)", 19.2),
        ("DDR4-3200 (dual channel)", 51.2),
        ("DDR5-4800 (dual channel)", 76.8),
        ("DDR5-6400 (quad channel)", 204.8),
    ]

    print(f"    {'Configuration':<35s} {'BW (GB/s)':>10s} {'Sufficient?':>12s}")
    print(f"    {'-'*35} {'-'*10} {'-'*12}")

    for config, bw in mem_configs:
        sufficient = "Yes" if bw >= bandwidth_gb else "No"
        print(f"    {config:<35s} {bw:>10.1f} {sufficient:>12s}")

    print()
    print("  Bandwidth is a shared resource: CPU + GPU + DMA all compete.")
    print("  Prefetching can hide latency but increases bandwidth demand.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Memory Technology Comparison", exercise_1),
        ("Exercise 2: Principle of Locality", exercise_2),
        ("Exercise 3: AMAT Calculation (Single Level)", exercise_3),
        ("Exercise 4: Multi-Level Cache AMAT", exercise_4),
        ("Exercise 5: The Memory Wall Problem", exercise_5),
        ("Exercise 6: Working Set vs Cache Hit Rate", exercise_6),
        ("Exercise 7: Row-Major vs Column-Major Traversal", exercise_7),
        ("Exercise 8: Memory Bandwidth Analysis", exercise_8),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

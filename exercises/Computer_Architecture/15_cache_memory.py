"""
Exercises for Lesson 15: Cache Memory
Topic: Computer_Architecture

Solutions to practice problems covering cache mapping schemes
(direct-mapped, set-associative, fully-associative), replacement policies
(LRU, FIFO), write policies, the 3 Cs of cache misses,
and multi-level cache design.
"""


def exercise_1():
    """
    Simulate a direct-mapped cache.
    Given a sequence of memory block addresses, trace hits and misses.
    """
    print("Direct-Mapped Cache Simulation:")
    print()

    cache_lines = 4  # 4 cache lines, indexed 0-3
    print(f"  Cache: {cache_lines} lines (direct-mapped)")
    print(f"  Mapping: block_address mod {cache_lines}")
    print()

    # Memory block access sequence
    accesses = [0, 4, 8, 0, 12, 0, 4, 8, 0, 12]
    cache = [None] * cache_lines  # tag stored in each line
    hits = 0
    misses = 0

    print(f"  {'Access':>7s} {'Index':>6s} {'Tag':>5s} {'Result':>7s} {'Cache State':>30s}")
    print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*7} {'-'*30}")

    for addr in accesses:
        index = addr % cache_lines
        tag = addr // cache_lines

        if cache[index] == tag:
            hits += 1
            result = "HIT"
        else:
            misses += 1
            result = "MISS"
            cache[index] = tag

        state_str = str([f"B{c*cache_lines+i}" if c is not None else "---"
                        for i, c in enumerate(cache)])
        print(f"  {addr:>7d} {index:>6d} {tag:>5d} {result:>7s} {state_str:>30s}")

    total = hits + misses
    hit_rate = hits / total
    print(f"\n  Hit rate: {hits}/{total} = {hit_rate:.1%}")
    print(f"\n  Problem: Blocks 0, 4, 8, 12 all map to index 0.")
    print(f"  This causes conflict misses even though cache has empty lines.")


def exercise_2():
    """
    Simulate a 2-way set-associative cache with LRU replacement.
    Same access sequence as exercise 1 to compare.
    """
    print("2-Way Set-Associative Cache Simulation:")
    print()

    num_sets = 2      # 2 sets
    ways = 2          # 2 ways per set = 4 lines total
    total_lines = num_sets * ways
    print(f"  Cache: {total_lines} lines, {num_sets} sets, {ways}-way associative (LRU)")
    print(f"  Set mapping: block_address mod {num_sets}")
    print()

    accesses = [0, 4, 8, 0, 12, 0, 4, 8, 0, 12]

    # Each set is a list of (tag, lru_counter) pairs
    # Higher lru_counter = more recently used
    sets = [[None] * ways for _ in range(num_sets)]
    lru_counters = [[0] * ways for _ in range(num_sets)]
    global_counter = 0
    hits = 0
    misses = 0

    print(f"  {'Access':>7s} {'Set':>4s} {'Tag':>5s} {'Result':>7s} {'Set Contents':>25s}")
    print(f"  {'-'*7} {'-'*4} {'-'*5} {'-'*7} {'-'*25}")

    for addr in accesses:
        set_idx = addr % num_sets
        tag = addr // num_sets
        global_counter += 1

        # Check for hit
        hit = False
        for w in range(ways):
            if sets[set_idx][w] == tag:
                hits += 1
                hit = True
                lru_counters[set_idx][w] = global_counter
                result = "HIT"
                break

        if not hit:
            misses += 1
            result = "MISS"
            # Find LRU way to replace
            lru_way = lru_counters[set_idx].index(min(lru_counters[set_idx]))
            sets[set_idx][lru_way] = tag
            lru_counters[set_idx][lru_way] = global_counter

        set_contents = []
        for s in range(num_sets):
            entries = []
            for w in range(ways):
                if sets[s][w] is not None:
                    block = sets[s][w] * num_sets + s
                    entries.append(f"B{block}")
                else:
                    entries.append("---")
            set_contents.append(f"S{s}:{entries}")

        print(f"  {addr:>7d} {set_idx:>4d} {tag:>5d} {result:>7s}  {set_contents}")

    total = hits + misses
    hit_rate = hits / total
    print(f"\n  Hit rate: {hits}/{total} = {hit_rate:.1%}")
    print(f"\n  Improvement over direct-mapped: Higher associativity reduces conflict misses.")


def exercise_3():
    """
    Implement and compare LRU and FIFO cache replacement policies.
    """
    print("Cache Replacement Policies: LRU vs FIFO:")
    print()

    cache_size = 4
    accesses = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

    print(f"  Cache size: {cache_size} blocks (fully associative)")
    print(f"  Access sequence: {accesses}")
    print()

    def simulate_lru(accesses, size):
        """LRU: Evict least recently used."""
        cache = []  # ordered: LRU at front, MRU at back
        results = []
        for addr in accesses:
            if addr in cache:
                cache.remove(addr)
                cache.append(addr)
                results.append(("HIT", list(cache)))
            else:
                if len(cache) >= size:
                    cache.pop(0)
                cache.append(addr)
                results.append(("MISS", list(cache)))
        return results

    def simulate_fifo(accesses, size):
        """FIFO: Evict oldest entry regardless of recent use."""
        cache = []
        results = []
        for addr in accesses:
            if addr in cache:
                results.append(("HIT", list(cache)))
            else:
                if len(cache) >= size:
                    cache.pop(0)
                cache.append(addr)
                results.append(("MISS", list(cache)))
        return results

    for policy_name, simulate_fn in [("LRU", simulate_lru), ("FIFO", simulate_fifo)]:
        results = simulate_fn(accesses, cache_size)
        hits = sum(1 for r, _ in results if r == "HIT")
        total = len(results)
        print(f"  {policy_name} Policy:")
        print(f"    {'Access':>7s} {'Result':>7s} {'Cache State':>20s}")
        print(f"    {'-'*7} {'-'*7} {'-'*20}")
        for i, (result, state) in enumerate(results):
            print(f"    {accesses[i]:>7d} {result:>7s} {str(state):>20s}")
        print(f"    Hit rate: {hits}/{total} = {hits/total:.1%}")
        print()

    print("  LRU is generally better because it considers recency of access.")
    print("  FIFO can suffer from Belady's anomaly (more cache = more misses).")


def exercise_4():
    """
    Explain and compare write-through vs write-back cache policies.
    Calculate memory traffic for each policy.
    """
    print("Cache Write Policies: Write-Through vs Write-Back:")
    print()

    # Simulate a sequence of reads and writes
    operations = [
        ("R", 0x100, "Read A"),
        ("W", 0x100, "Write A"),
        ("W", 0x100, "Write A again"),
        ("R", 0x200, "Read B"),
        ("W", 0x200, "Write B"),
        ("W", 0x100, "Write A again"),
        ("R", 0x300, "Read C (evicts A)"),
    ]

    cache_size = 2  # only 2 blocks

    print(f"  Operations: {len(operations)} memory ops, cache = {cache_size} blocks")
    print()

    # Write-through analysis
    print("  Write-Through Policy:")
    print("    Every write goes to BOTH cache and memory immediately.")
    print()
    wt_mem_writes = 0
    wt_mem_reads = 0
    wt_cache = {}

    for op, addr, desc in operations:
        if op == "R":
            if addr not in wt_cache:
                wt_mem_reads += 1
                if len(wt_cache) >= cache_size:
                    oldest = next(iter(wt_cache))
                    del wt_cache[oldest]
                wt_cache[addr] = True
        else:  # Write
            wt_mem_writes += 1  # Always write to memory
            if addr not in wt_cache:
                wt_mem_reads += 1  # Write-allocate: fetch block first
                if len(wt_cache) >= cache_size:
                    oldest = next(iter(wt_cache))
                    del wt_cache[oldest]
            wt_cache[addr] = True

    print(f"    Memory reads:  {wt_mem_reads}")
    print(f"    Memory writes: {wt_mem_writes}")
    print(f"    Total memory traffic: {wt_mem_reads + wt_mem_writes} transactions")
    print()

    # Write-back analysis
    print("  Write-Back Policy:")
    print("    Writes go only to cache; dirty blocks written to memory on eviction.")
    print()
    wb_mem_writes = 0
    wb_mem_reads = 0
    wb_cache = {}  # addr -> dirty flag

    for op, addr, desc in operations:
        if op == "R":
            if addr not in wb_cache:
                wb_mem_reads += 1
                if len(wb_cache) >= cache_size:
                    evict_addr = next(iter(wb_cache))
                    if wb_cache[evict_addr]:
                        wb_mem_writes += 1  # Write-back dirty block
                    del wb_cache[evict_addr]
                wb_cache[addr] = False
        else:  # Write
            if addr not in wb_cache:
                wb_mem_reads += 1  # Write-allocate
                if len(wb_cache) >= cache_size:
                    evict_addr = next(iter(wb_cache))
                    if wb_cache[evict_addr]:
                        wb_mem_writes += 1
                    del wb_cache[evict_addr]
            wb_cache[addr] = True  # Mark dirty

    print(f"    Memory reads:  {wb_mem_reads}")
    print(f"    Memory writes: {wb_mem_writes}")
    print(f"    Total memory traffic: {wb_mem_reads + wb_mem_writes} transactions")
    print()

    print("  Write-back reduces memory traffic when same block is written multiple times.")
    print("  Write-through is simpler and ensures memory is always up-to-date.")
    print("  Modern CPUs use write-back for L1/L2 (performance), and sometimes")
    print("  write-through between L2-L3 (coherence simplicity).")


def exercise_5():
    """
    Classify cache misses into the 3 Cs: Compulsory, Capacity, Conflict.
    """
    print("The 3 Cs of Cache Misses:")
    print()

    print("  Definitions:")
    print("    Compulsory (Cold): First access to a block. Unavoidable.")
    print("    Capacity: Working set exceeds cache size. Would miss even fully assoc.")
    print("    Conflict: Multiple blocks map to same set. Would NOT miss if fully assoc.")
    print()

    # Simulate with direct-mapped (to identify conflict misses)
    # and fully-associative (to separate capacity from conflict)
    cache_size = 4  # 4 blocks
    accesses = [0, 1, 2, 3, 4, 0, 1, 5, 6, 7, 8, 0]

    print(f"  Cache: {cache_size} blocks")
    print(f"  Access sequence: {accesses}")
    print()

    # Fully-associative simulation (LRU) -- no conflict misses
    fa_cache = []
    fa_ever_seen = set()
    fa_results = []
    for addr in accesses:
        if addr in fa_cache:
            fa_cache.remove(addr)
            fa_cache.append(addr)
            fa_results.append("HIT")
        else:
            miss_type = "COMPULSORY" if addr not in fa_ever_seen else "CAPACITY"
            fa_results.append(miss_type)
            fa_ever_seen.add(addr)
            if len(fa_cache) >= cache_size:
                fa_cache.pop(0)
            fa_cache.append(addr)

    # Direct-mapped simulation
    dm_cache = [None] * cache_size
    dm_ever_seen = set()
    dm_results = []
    for addr in accesses:
        index = addr % cache_size
        if dm_cache[index] == addr:
            dm_results.append("HIT")
        else:
            if addr not in dm_ever_seen:
                dm_results.append("COMPULSORY")
            else:
                # Miss in DM but hit in FA? Conflict miss.
                # Miss in both? Capacity (or compulsory already handled)
                fa_idx = accesses.index(addr) if addr in fa_ever_seen else -1
                dm_results.append("CONFLICT")  # simplified classification
            dm_ever_seen.add(addr)
            dm_cache[index] = addr

    print(f"  {'Access':>7s} {'Fully Assoc':>14s} {'Direct Mapped':>14s}")
    print(f"  {'-'*7} {'-'*14} {'-'*14}")
    for i, addr in enumerate(accesses):
        print(f"  {addr:>7d} {fa_results[i]:>14s} {dm_results[i]:>14s}")

    compulsory = sum(1 for r in fa_results if r == "COMPULSORY")
    capacity = sum(1 for r in fa_results if r == "CAPACITY")
    conflict = sum(1 for r in dm_results if r == "CONFLICT")
    print()
    print(f"  Compulsory misses: {compulsory}")
    print(f"  Capacity misses:   {capacity} (with fully-associative cache)")
    print(f"  Conflict misses:   {conflict} (additional misses in direct-mapped)")
    print()
    print("  Reducing each type:")
    print("    Compulsory → Prefetching")
    print("    Capacity   → Larger cache")
    print("    Conflict   → Higher associativity")


def exercise_6():
    """
    Parse a memory address into tag, index, and offset fields
    for different cache configurations.
    """
    print("Cache Address Decomposition:")
    print()

    addr_bits = 32
    block_size = 64  # bytes

    configs = [
        ("Direct-mapped, 256 lines", 256, 1),
        ("2-way, 128 sets", 128, 2),
        ("4-way, 64 sets", 64, 4),
        ("8-way, 32 sets", 32, 8),
        ("Fully associative, 256 lines", 1, 256),
    ]

    import math

    offset_bits = int(math.log2(block_size))

    print(f"  Address: {addr_bits} bits, Block size: {block_size} bytes "
          f"(offset = {offset_bits} bits)")
    print()
    print(f"  {'Configuration':<30s} {'Sets':>5s} {'Offset':>7s} {'Index':>6s} "
          f"{'Tag':>5s} {'Total Lines':>12s}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*12}")

    for name, num_sets, associativity in configs:
        index_bits = int(math.log2(num_sets))
        tag_bits = addr_bits - index_bits - offset_bits
        total_lines = num_sets * associativity
        print(
            f"  {name:<30s} {num_sets:>5d} {offset_bits:>7d} {index_bits:>6d} "
            f"{tag_bits:>5d} {total_lines:>12d}"
        )

    print()
    # Example address decomposition
    example_addr = 0x0001_ABCD
    print(f"  Example: Address 0x{example_addr:08X}")
    print(f"  For direct-mapped, 256 lines, 64B blocks:")
    offset = example_addr & ((1 << offset_bits) - 1)
    index_bits_dm = int(math.log2(256))
    index = (example_addr >> offset_bits) & ((1 << index_bits_dm) - 1)
    tag = example_addr >> (offset_bits + index_bits_dm)
    print(f"    Tag: 0x{tag:04X} ({tag})")
    print(f"    Index: 0x{index:02X} ({index})")
    print(f"    Offset: 0x{offset:02X} ({offset})")


def exercise_7():
    """
    Calculate the impact of cache line size on miss rate and bandwidth.
    Larger lines improve spatial locality but increase miss penalty.
    """
    print("Cache Line Size Tradeoff:")
    print()

    cache_capacity = 32 * 1024  # 32 KB
    mem_latency_ns = 50         # fixed latency component
    mem_bandwidth_gbps = 25.6   # GB/s

    line_sizes = [16, 32, 64, 128, 256]

    print(f"  Cache: {cache_capacity//1024} KB, Memory latency: {mem_latency_ns} ns, "
          f"Bandwidth: {mem_bandwidth_gbps} GB/s")
    print()

    # Simulated miss rates (spatial locality benefit of larger lines)
    # Larger lines capture more spatial locality but reduce number of lines
    base_miss_rate = 0.08
    miss_rates = {}
    for ls in line_sizes:
        # Spatial locality benefit (diminishing returns)
        spatial_benefit = 1.0 - 0.3 * (ls / 64)
        spatial_benefit = max(spatial_benefit, 0.3)
        # Capacity penalty (fewer lines = more capacity misses)
        num_lines = cache_capacity // ls
        capacity_penalty = 1.0 + max(0, (256 - num_lines) / 256)
        miss_rates[ls] = base_miss_rate * spatial_benefit * capacity_penalty

    print(f"  {'Line Size':>10s} {'#Lines':>7s} {'Miss Rate':>10s} "
          f"{'Transfer (ns)':>14s} {'Miss Penalty':>13s} {'AMAT (ns)':>10s}")
    print(f"  {'-'*10} {'-'*7} {'-'*10} {'-'*14} {'-'*13} {'-'*10}")

    for ls in line_sizes:
        num_lines = cache_capacity // ls
        mr = miss_rates[ls]
        transfer_ns = ls / (mem_bandwidth_gbps * 1e9 / 1e9)  # ns
        miss_penalty = mem_latency_ns + transfer_ns
        hit_time = 1  # ns
        amat = hit_time + mr * miss_penalty
        print(
            f"  {ls:>10d} {num_lines:>7d} {mr:>10.2%} "
            f"{transfer_ns:>14.1f} {miss_penalty:>13.1f} {amat:>10.2f}"
        )

    print()
    print("  64-byte lines are the sweet spot for most workloads.")
    print("  Too small: poor spatial locality. Too large: high miss penalty + capacity waste.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Direct-Mapped Cache", exercise_1),
        ("Exercise 2: Set-Associative Cache", exercise_2),
        ("Exercise 3: LRU vs FIFO Replacement", exercise_3),
        ("Exercise 4: Write-Through vs Write-Back", exercise_4),
        ("Exercise 5: The 3 Cs of Cache Misses", exercise_5),
        ("Exercise 6: Address Decomposition", exercise_6),
        ("Exercise 7: Cache Line Size Tradeoff", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

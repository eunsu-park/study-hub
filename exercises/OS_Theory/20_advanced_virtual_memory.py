"""
Exercises for Lesson 20: Advanced Virtual Memory
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers TLB performance analysis, huge page benefit calculation,
and NUMA memory access latency modeling.
"""


# === Exercise 1: TLB Performance Analysis ===
# Problem: Calculate effective memory access time with multi-level TLBs
# for different working set sizes.

def exercise_1():
    """Analyze TLB hit rates and effective memory access time."""
    # TLB hierarchy parameters
    l1_tlb_entries = 64
    l1_tlb_latency = 1      # cycles
    l2_tlb_entries = 1024
    l2_tlb_latency = 7      # cycles
    page_walk_latency = 200  # cycles
    memory_access = 100      # cycles for the actual data access

    page_size = 4096  # 4 KB

    # Working set sizes to test (in pages)
    working_sets = [32, 64, 128, 512, 1024, 2048, 4096, 10000]

    print("TLB Performance Analysis")
    print(f"L1 TLB: {l1_tlb_entries} entries, {l1_tlb_latency} cycle")
    print(f"L2 TLB: {l2_tlb_entries} entries, {l2_tlb_latency} cycles")
    print(f"Page walk: {page_walk_latency} cycles")
    print(f"Memory access: {memory_access} cycles")
    print(f"Page size: {page_size} bytes\n")

    print(f"{'Pages':<8} {'WS Size':<12} {'L1 Hit%':<10} {'L2 Hit%':<10} "
          f"{'Walk%':<10} {'EAT (cyc)':<12} {'Slowdown'}")
    print("-" * 75)

    baseline_eat = None

    for num_pages in working_sets:
        ws_bytes = num_pages * page_size

        # Model TLB hit rates based on working set size
        # L1 TLB: covers l1_tlb_entries pages fully
        if num_pages <= l1_tlb_entries:
            l1_hit_rate = 1.0
        else:
            l1_hit_rate = l1_tlb_entries / num_pages

        # L2 TLB: covers remaining pages up to l2_tlb_entries
        l1_miss_rate = 1.0 - l1_hit_rate
        if num_pages <= l2_tlb_entries:
            # All misses from L1 are caught by L2
            l2_hit_rate = l1_miss_rate
            walk_rate = 0.0
        else:
            # L2 can cover l2_tlb_entries - l1_tlb_entries additional pages
            l2_coverage = min(l2_tlb_entries, num_pages) / num_pages
            l2_hit_rate = l2_coverage - l1_hit_rate
            if l2_hit_rate < 0:
                l2_hit_rate = 0.0
            walk_rate = 1.0 - l1_hit_rate - l2_hit_rate

        # Effective access time
        eat = (l1_hit_rate * (l1_tlb_latency + memory_access) +
               l2_hit_rate * (l1_tlb_latency + l2_tlb_latency + memory_access) +
               walk_rate * (l1_tlb_latency + l2_tlb_latency +
                            page_walk_latency + memory_access))

        if baseline_eat is None:
            baseline_eat = eat

        slowdown = eat / baseline_eat

        ws_label = f"{ws_bytes // 1024} KB" if ws_bytes < 1024 * 1024 else \
                   f"{ws_bytes // (1024 * 1024)} MB"

        print(f"{num_pages:<8} {ws_label:<12} {l1_hit_rate * 100:>6.1f}%   "
              f"{l2_hit_rate * 100:>6.1f}%   {walk_rate * 100:>6.1f}%   "
              f"{eat:>8.1f}     {slowdown:.2f}x")

    print(f"\nKey observations:")
    print(f"  1. Working sets fitting in L1 TLB ({l1_tlb_entries} pages = "
          f"{l1_tlb_entries * page_size // 1024} KB) have optimal performance.")
    print(f"  2. Working sets fitting in L2 TLB ({l2_tlb_entries} pages = "
          f"{l2_tlb_entries * page_size // (1024 * 1024)} MB) have minor slowdown.")
    print(f"  3. Beyond L2 TLB capacity, page walks cause significant slowdown.")
    print(f"  4. This is why huge pages (2 MB) help: they reduce TLB pressure.")


# === Exercise 2: Huge Page Benefit Calculator ===
# Problem: Calculate TLB coverage and miss reduction when using huge pages.

def exercise_2():
    """Calculate performance benefits of huge pages vs regular pages."""
    memory_sizes_mb = [256, 1024, 4096, 16384]  # MB
    regular_page_kb = 4        # 4 KB
    huge_page_kb = 2 * 1024    # 2 MB

    tlb_entries = 1024  # Total TLB entries (L1 + L2 combined for simplicity)
    page_walk_cost_ns = 50  # nanoseconds per page walk

    print("Huge Page Benefit Analysis")
    print(f"TLB entries: {tlb_entries}")
    print(f"Regular page: {regular_page_kb} KB, Huge page: {huge_page_kb} KB")
    print(f"Page walk cost: {page_walk_cost_ns} ns\n")

    print(f"{'Memory':<10} {'Reg Pages':<12} {'Huge Pages':<12} "
          f"{'Reg TLB Cov':<14} {'Huge TLB Cov':<14} {'Reg Miss%':<12} "
          f"{'Huge Miss%'}")
    print("-" * 90)

    for size_mb in memory_sizes_mb:
        size_kb = size_mb * 1024
        regular_pages = size_kb // regular_page_kb
        huge_pages = size_kb // huge_page_kb

        # TLB coverage: what fraction of the working set fits in the TLB
        reg_coverage = min(1.0, tlb_entries / regular_pages)
        huge_coverage = min(1.0, tlb_entries / huge_pages)

        # Approximate miss rate (assuming uniform random access)
        reg_miss_rate = max(0.0, 1.0 - reg_coverage)
        huge_miss_rate = max(0.0, 1.0 - huge_coverage)

        reg_cov_label = f"{reg_coverage * 100:.2f}%"
        huge_cov_label = f"{huge_coverage * 100:.2f}%"

        size_label = f"{size_mb} MB" if size_mb < 1024 else f"{size_mb // 1024} GB"

        print(f"{size_label:<10} {regular_pages:<12,} {huge_pages:<12,} "
              f"{reg_cov_label:<14} {huge_cov_label:<14} "
              f"{reg_miss_rate * 100:>7.2f}%     {huge_miss_rate * 100:>7.2f}%")

    # Detailed analysis for 1 GB working set
    print(f"\n--- Detailed Analysis: 1 GB Working Set ---\n")
    size_kb = 1024 * 1024
    reg_pages = size_kb // regular_page_kb
    huge_pages_count = size_kb // huge_page_kb
    accesses = 1_000_000  # 1M memory accesses

    reg_misses = int(accesses * max(0.0, 1.0 - tlb_entries / reg_pages))
    huge_misses = int(accesses * max(0.0, 1.0 - tlb_entries / huge_pages_count))

    reg_walk_time_ms = reg_misses * page_walk_cost_ns / 1e6
    huge_walk_time_ms = huge_misses * page_walk_cost_ns / 1e6

    print(f"  Regular pages: {reg_pages:,} pages needed, "
          f"{reg_misses:,} TLB misses, {reg_walk_time_ms:.1f} ms in page walks")
    print(f"  Huge pages:    {huge_pages_count:,} pages needed, "
          f"{huge_misses:,} TLB misses, {huge_walk_time_ms:.1f} ms in page walks")
    print(f"  Time saved: {reg_walk_time_ms - huge_walk_time_ms:.1f} ms "
          f"({(1 - huge_walk_time_ms / reg_walk_time_ms) * 100:.1f}% reduction)")

    print(f"\nWhen to use huge pages:")
    print(f"  - Databases (large buffer pools)")
    print(f"  - Scientific computing (large arrays)")
    print(f"  - Virtual machines (guest memory)")
    print(f"  - Any workload with working set >> TLB coverage")
    print(f"\nWhen NOT to use huge pages:")
    print(f"  - Small processes (waste from internal fragmentation)")
    print(f"  - Many small processes (huge page supply is limited)")
    print(f"  - Workloads with sparse memory access patterns")


# === Exercise 3: NUMA Topology Simulator ===
# Problem: Model NUMA memory access costs and demonstrate the impact
# of local vs remote memory allocation.

def exercise_3():
    """Simulate NUMA memory access latency for different allocation strategies."""
    # NUMA topology: 2 nodes, 4 cores each
    num_nodes = 2
    cores_per_node = 4
    total_cores = num_nodes * cores_per_node

    local_latency_ns = 80
    remote_latency_ns = 140
    remote_penalty = remote_latency_ns / local_latency_ns

    memory_per_node_gb = 32

    print("NUMA Topology Simulator")
    print(f"Nodes: {num_nodes}, Cores per node: {cores_per_node}")
    print(f"Memory per node: {memory_per_node_gb} GB")
    print(f"Local access: {local_latency_ns} ns, Remote access: {remote_latency_ns} ns")
    print(f"Remote penalty: {remote_penalty:.2f}x\n")

    # Show NUMA topology
    print("Topology:")
    for node in range(num_nodes):
        cores = [node * cores_per_node + c for c in range(cores_per_node)]
        print(f"  Node {node}: Cores {cores}, {memory_per_node_gb} GB RAM")
    print()

    # Allocation strategies
    num_accesses = 1_000_000

    strategies = [
        ("Local Only", "All memory allocated on the thread's local node"),
        ("Remote Only", "All memory allocated on a remote node"),
        ("Interleaved", "Memory interleaved across all nodes (round-robin)"),
        ("First-Touch", "Memory placed on the node that first accesses it"),
    ]

    print(f"Memory Access Simulation ({num_accesses:,} accesses per thread)\n")
    print(f"{'Strategy':<16} {'Local%':<10} {'Remote%':<10} "
          f"{'Avg Lat (ns)':<14} {'Total (ms)':<12} {'vs Local'}")
    print("-" * 70)

    local_only_time = None

    for name, description in strategies:
        if name == "Local Only":
            local_pct = 1.0
            remote_pct = 0.0
        elif name == "Remote Only":
            local_pct = 0.0
            remote_pct = 1.0
        elif name == "Interleaved":
            # Round-robin: 1/num_nodes chance of being local
            local_pct = 1.0 / num_nodes
            remote_pct = 1.0 - local_pct
        elif name == "First-Touch":
            # Assuming good first-touch: 90% local
            local_pct = 0.90
            remote_pct = 0.10

        avg_latency = local_pct * local_latency_ns + remote_pct * remote_latency_ns
        total_time_ms = (num_accesses * avg_latency) / 1e6

        if local_only_time is None:
            local_only_time = total_time_ms

        ratio = total_time_ms / local_only_time

        print(f"{name:<16} {local_pct * 100:>6.1f}%   {remote_pct * 100:>6.1f}%   "
              f"{avg_latency:>10.1f}     {total_time_ms:>8.1f}     {ratio:.2f}x")

    # Thread migration impact
    print(f"\n--- Thread Migration Impact ---\n")
    print("  When the OS migrates a thread from Node 0 to Node 1:")
    print(f"  - Memory stays on Node 0 (physical location does not change)")
    print(f"  - Thread on Node 1 now accesses remote memory: {remote_latency_ns} ns")
    print(f"  - Performance degrades by {remote_penalty:.2f}x")
    print(f"\n  Mitigation strategies:")
    print(f"  1. CPU affinity: Pin threads to specific cores (sched_setaffinity)")
    print(f"  2. Memory policy: Use mbind() or set_mempolicy() for NUMA-aware allocation")
    print(f"  3. numactl: Run processes with --membind or --cpunodebind")
    print(f"  4. Interleaved for shared data: Spread across nodes for balanced access")

    # Working example: matrix multiply with NUMA
    print(f"\n--- Example: Matrix Multiply (4096x4096) ---\n")
    n = 4096
    total_elements = n * n
    bytes_per_element = 8  # double
    total_bytes = total_elements * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)

    # Each element accessed ~n times in naive multiply
    total_mem_accesses = total_elements * n
    local_time_s = (total_mem_accesses * local_latency_ns) / 1e9
    remote_time_s = (total_mem_accesses * remote_latency_ns) / 1e9

    print(f"  Matrix size: {n}x{n}, {total_mb:.0f} MB per matrix")
    print(f"  Memory accesses: ~{total_mem_accesses / 1e9:.1f} billion")
    print(f"  All-local estimate:  {local_time_s:.1f} s")
    print(f"  All-remote estimate: {remote_time_s:.1f} s")
    print(f"  Difference: {remote_time_s - local_time_s:.1f} s "
          f"({(remote_time_s / local_time_s - 1) * 100:.0f}% slower)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: TLB Performance Analysis ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Huge Page Benefit Calculator ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: NUMA Topology Simulator ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")

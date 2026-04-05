"""
Exercises for Lesson 16: Virtual Memory
Topic: Computer_Architecture

Solutions to practice problems covering address translation,
page tables, TLB operation, page fault handling,
page replacement algorithms (FIFO, LRU, Optimal),
and multi-level page tables.
"""

import math


def exercise_1():
    """
    Perform virtual-to-physical address translation with a single-level page table.
    """
    print("Virtual-to-Physical Address Translation:")
    print()

    page_size = 4096  # 4 KB
    virtual_addr_bits = 32
    physical_addr_bits = 28

    offset_bits = int(math.log2(page_size))
    vpn_bits = virtual_addr_bits - offset_bits
    ppn_bits = physical_addr_bits - offset_bits
    num_virtual_pages = 2 ** vpn_bits
    num_physical_frames = 2 ** ppn_bits

    print(f"  Page size: {page_size} bytes ({offset_bits} offset bits)")
    print(f"  Virtual address: {virtual_addr_bits} bits -> VPN: {vpn_bits} bits")
    print(f"  Physical address: {physical_addr_bits} bits -> PPN: {ppn_bits} bits")
    print(f"  Virtual pages: {num_virtual_pages:,}")
    print(f"  Physical frames: {num_physical_frames:,}")
    print()

    # Simplified page table (VPN -> PPN mapping)
    page_table = {
        0x00000: 0x1234,
        0x00001: 0x5678,
        0x0ABCD: 0x0042,
        0x12345: None,  # Not in memory (page fault)
        0xFFFFF: 0x0001,
    }

    example_addrs = [0x00000A5C, 0x00001F00, 0x0ABCD100, 0x12345678, 0xFFFFF000]

    print(f"  {'Virtual Addr':>14s} {'VPN':>8s} {'Offset':>8s} "
          f"{'PPN':>8s} {'Physical Addr':>14s} {'Status':>12s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*14} {'-'*12}")

    for va in example_addrs:
        vpn = va >> offset_bits
        offset = va & ((1 << offset_bits) - 1)

        if vpn in page_table and page_table[vpn] is not None:
            ppn = page_table[vpn]
            pa = (ppn << offset_bits) | offset
            status = "OK"
            pa_str = f"0x{pa:07X}"
        else:
            ppn_str = "---"
            pa_str = "---"
            status = "PAGE FAULT"
            ppn = None

        ppn_display = f"0x{ppn:04X}" if ppn is not None else "---"
        print(
            f"  0x{va:08X} 0x{vpn:05X} 0x{offset:03X} "
            f"{ppn_display:>8s} {pa_str:>14s} {status:>12s}"
        )

    print()
    print("  Key: VPN (Virtual Page Number) indexes the page table.")
    print("  PPN (Physical Page Number) replaces VPN; offset unchanged.")


def exercise_2():
    """
    Simulate TLB (Translation Lookaside Buffer) operation.
    Show TLB hits, misses, and page table walks.
    """
    print("TLB Simulation:")
    print()

    tlb_entries = 4
    print(f"  TLB: {tlb_entries} entries, fully associative, LRU replacement")
    print()

    # Page table (complete mapping)
    page_table = {0: 10, 1: 20, 2: 30, 3: 40, 4: 50, 5: 60, 6: 70, 7: 80}

    # Access sequence (page numbers)
    accesses = [0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4]

    tlb = []  # list of (vpn, ppn) pairs, MRU at end
    tlb_hits = 0
    tlb_misses = 0

    print(f"  {'Access':>7s} {'VPN':>5s} {'TLB':>5s} {'PPN':>5s} {'TLB State':>30s}")
    print(f"  {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*30}")

    for i, vpn in enumerate(accesses):
        # TLB lookup
        tlb_vpns = [entry[0] for entry in tlb]
        if vpn in tlb_vpns:
            tlb_hits += 1
            result = "HIT"
            idx = tlb_vpns.index(vpn)
            ppn = tlb[idx][1]
            # Move to MRU position
            entry = tlb.pop(idx)
            tlb.append(entry)
        else:
            tlb_misses += 1
            result = "MISS"
            ppn = page_table[vpn]
            # Page table walk, then insert into TLB
            if len(tlb) >= tlb_entries:
                tlb.pop(0)  # Evict LRU
            tlb.append((vpn, ppn))

        tlb_state = str([(v, p) for v, p in tlb])
        print(f"  {i+1:>7d} {vpn:>5d} {result:>5s} {ppn:>5d} {tlb_state:>30s}")

    total = tlb_hits + tlb_misses
    print(f"\n  TLB hit rate: {tlb_hits}/{total} = {tlb_hits/total:.1%}")
    print(f"\n  TLB miss cost: ~10-100 cycles (page table walk)")
    print(f"  Page fault cost: ~1,000,000 cycles (disk access)")
    print(f"  TLB hit cost: ~1 cycle (parallel with cache access)")


def exercise_3():
    """
    Calculate the size of a single-level vs multi-level page table.
    Show why multi-level page tables save memory.
    """
    print("Page Table Size: Single-Level vs Multi-Level:")
    print()

    va_bits = 48  # 64-bit system typically uses 48-bit VA
    page_size = 4096  # 4 KB
    pte_size = 8  # 8 bytes per page table entry

    offset_bits = int(math.log2(page_size))
    vpn_bits = va_bits - offset_bits

    # Single-level page table
    single_entries = 2 ** vpn_bits
    single_size = single_entries * pte_size

    print(f"  Virtual address: {va_bits} bits, Page size: {page_size} bytes")
    print(f"  VPN bits: {vpn_bits}, PTE size: {pte_size} bytes")
    print()
    print(f"  Single-level page table:")
    print(f"    Entries: 2^{vpn_bits} = {single_entries:,}")
    print(f"    Size: {single_size:,} bytes = {single_size / (1024**3):.0f} GB")
    print(f"    Per process! Clearly impractical.")
    print()

    # Multi-level page table (x86-64 style: 4 levels)
    # Each level indexes with 9 bits, page offset = 12 bits
    # 9 + 9 + 9 + 9 + 12 = 48 bits
    levels = 4
    bits_per_level = 9
    entries_per_table = 2 ** bits_per_level

    print(f"  Multi-level page table ({levels} levels, {bits_per_level} bits each):")
    print(f"    Entries per table: {entries_per_table}")
    one_table_size = entries_per_table * pte_size
    print(f"    Size of one table: {one_table_size} bytes = {one_table_size / 1024:.0f} KB")
    print()

    # For a process using only 16 MB of memory
    used_memory_mb = 16
    used_pages = (used_memory_mb * 1024 * 1024) // page_size

    # Worst case: all pages in different L3 tables
    # Best case: contiguous pages
    l4_tables = 1  # always 1 root
    l3_tables = 1  # 16MB fits in one L3 entry range (512 GB)
    l2_tables = 1  # 16MB fits in one L2 entry range (1 GB)
    l1_tables = (used_pages + entries_per_table - 1) // entries_per_table

    total_tables = l4_tables + l3_tables + l2_tables + l1_tables
    total_size = total_tables * one_table_size

    print(f"    Process using {used_memory_mb} MB ({used_pages} pages):")
    print(f"      L4 (PML4):  {l4_tables} table")
    print(f"      L3 (PDPT):  {l3_tables} table")
    print(f"      L2 (PD):    {l2_tables} table")
    print(f"      L1 (PT):    {l1_tables} tables")
    print(f"      Total: {total_tables} tables = {total_size:,} bytes = {total_size / 1024:.0f} KB")
    print()
    print(f"    Savings: {single_size / total_size:,.0f}x smaller than single-level!")
    print(f"    Multi-level only allocates tables for address ranges actually used.")


def exercise_4():
    """
    Simulate page replacement algorithms: FIFO, LRU, and Optimal.
    """
    print("Page Replacement Algorithms:")
    print()

    num_frames = 3
    # Reference string from classic OS examples
    ref_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

    print(f"  Frames: {num_frames}")
    print(f"  Reference string: {ref_string}")
    print()

    def simulate_fifo(refs, frames):
        memory = []
        faults = 0
        for page in refs:
            if page not in memory:
                faults += 1
                if len(memory) >= frames:
                    memory.pop(0)
                memory.append(page)
        return faults

    def simulate_lru(refs, frames):
        memory = []
        faults = 0
        for page in refs:
            if page in memory:
                memory.remove(page)
                memory.append(page)
            else:
                faults += 1
                if len(memory) >= frames:
                    memory.pop(0)
                memory.append(page)
        return faults

    def simulate_optimal(refs, frames):
        memory = []
        faults = 0
        for i, page in enumerate(refs):
            if page not in memory:
                faults += 1
                if len(memory) >= frames:
                    # Find page used furthest in future (or never)
                    farthest = -1
                    victim = memory[0]
                    for m in memory:
                        try:
                            next_use = refs[i+1:].index(m)
                        except ValueError:
                            victim = m
                            break
                        if next_use > farthest:
                            farthest = next_use
                            victim = m
                    memory.remove(victim)
                memory.append(page)
        return faults

    results = [
        ("FIFO", simulate_fifo(ref_string, num_frames)),
        ("LRU", simulate_lru(ref_string, num_frames)),
        ("Optimal", simulate_optimal(ref_string, num_frames)),
    ]

    print(f"  {'Algorithm':>10s} {'Page Faults':>12s} {'Fault Rate':>12s}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")

    for name, faults in results:
        rate = faults / len(ref_string)
        print(f"  {name:>10s} {faults:>12d} {rate:>12.1%}")

    print()
    print("  Optimal is unrealizable (requires future knowledge) but serves as")
    print("  a lower bound. LRU approximates Optimal well in practice.")
    print("  FIFO is simplest but can suffer from Belady's anomaly.")


def exercise_5():
    """
    Calculate the cost of a TLB miss vs a page fault.
    Show why TLB coverage matters.
    """
    print("TLB Miss vs Page Fault Cost Analysis:")
    print()

    # System parameters
    clock_ghz = 4.0
    cycle_ns = 1.0 / clock_ghz

    costs = {
        "TLB hit": 1,             # cycles
        "TLB miss (page walk)": 50,  # cycles (hardware page walk)
        "TLB miss (SW walk)": 200,   # cycles (software-managed TLB)
        "Page fault (SSD)": 40_000,  # ~10 us
        "Page fault (HDD)": 40_000_000,  # ~10 ms
    }

    print(f"  CPU clock: {clock_ghz} GHz ({cycle_ns:.2f} ns/cycle)")
    print()
    print(f"  {'Event':<25s} {'Cycles':>12s} {'Time':>12s} {'Relative':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    base_cycles = costs["TLB hit"]
    for event, cycles in costs.items():
        time_ns = cycles * cycle_ns
        if time_ns < 1000:
            time_str = f"{time_ns:.1f} ns"
        elif time_ns < 1_000_000:
            time_str = f"{time_ns/1000:.1f} us"
        else:
            time_str = f"{time_ns/1_000_000:.1f} ms"
        relative = cycles / base_cycles
        print(f"  {event:<25s} {cycles:>12,d} {time_str:>12s} {relative:>9.0f}x")

    print()

    # TLB coverage calculation
    print("  TLB Coverage:")
    tlb_sizes = [64, 256, 1024, 2048]
    page_sizes = [4096, 2 * 1024 * 1024, 1024 * 1024 * 1024]  # 4KB, 2MB, 1GB

    print(f"  {'TLB Entries':>12s}", end="")
    for ps in page_sizes:
        if ps < 1024 * 1024:
            label = f"{ps // 1024} KB pages"
        elif ps < 1024 * 1024 * 1024:
            label = f"{ps // (1024*1024)} MB pages"
        else:
            label = f"{ps // (1024*1024*1024)} GB pages"
        print(f" {label:>14s}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in page_sizes:
        print(f" {'-'*14}", end="")
    print()

    for entries in tlb_sizes:
        print(f"  {entries:>12d}", end="")
        for ps in page_sizes:
            coverage = entries * ps
            if coverage < 1024 * 1024:
                cov_str = f"{coverage // 1024} KB"
            elif coverage < 1024 * 1024 * 1024:
                cov_str = f"{coverage // (1024*1024)} MB"
            else:
                cov_str = f"{coverage // (1024*1024*1024)} GB"
            print(f" {cov_str:>14s}", end="")
        print()

    print()
    print("  Huge pages (2MB/1GB) dramatically increase TLB coverage.")
    print("  A 1024-entry TLB with 4KB pages covers only 4MB.")
    print("  With 2MB pages, same TLB covers 2GB -- enough for most workloads.")


def exercise_6():
    """
    Trace address translation through a 2-level page table.
    """
    print("Two-Level Page Table Walk:")
    print()

    va_bits = 32
    page_size = 4096
    offset_bits = 12
    # Split VPN into two levels: 10 bits each
    l1_bits = 10
    l2_bits = 10

    print(f"  32-bit VA: [{l1_bits} L1 index | {l2_bits} L2 index | {offset_bits} offset]")
    print(f"  L1 table: {2**l1_bits} entries (one per process)")
    print(f"  L2 tables: {2**l2_bits} entries each (allocated on demand)")
    print()

    # Build a small page table for demonstration
    # L1 table: index -> L2 table base address (or None if not allocated)
    l2_table_0 = {0: 0x00A, 1: 0x00B, 2: 0x00C}  # L2 table for L1[0]
    l2_table_3 = {5: 0x0FF, 6: 0x100}              # L2 table for L1[3]

    l1_table = {0: l2_table_0, 3: l2_table_3}

    example_addrs = [0x00000A5C, 0x00001100, 0x00C05F00, 0x00200000]

    print(f"  {'Virtual Addr':>14s} {'L1 Idx':>7s} {'L2 Idx':>7s} {'Offset':>8s} "
          f"{'PPN':>6s} {'Result':>12s}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*12}")

    for va in example_addrs:
        offset = va & 0xFFF
        l2_idx = (va >> 12) & 0x3FF
        l1_idx = (va >> 22) & 0x3FF

        if l1_idx in l1_table:
            l2_table = l1_table[l1_idx]
            if l2_idx in l2_table:
                ppn = l2_table[l2_idx]
                result = "OK"
                ppn_str = f"0x{ppn:03X}"
            else:
                result = "PAGE FAULT"
                ppn_str = "---"
        else:
            result = "L1 NOT ALLOC"
            ppn_str = "---"

        print(
            f"  0x{va:08X} {l1_idx:>7d} {l2_idx:>7d} 0x{offset:03X}  "
            f"{ppn_str:>6s} {result:>12s}"
        )

    print()
    print("  Two memory accesses required per translation (L1 + L2 table lookups).")
    print("  This is why the TLB is critical -- it caches completed translations.")
    print("  x86-64 uses 4-level page tables (4 memory accesses without TLB!).")


def exercise_7():
    """
    Compare VIVT, VIPT, and PIPT cache-virtual memory interactions.
    """
    print("Cache and Virtual Memory Interaction:")
    print()

    designs = [
        {
            "name": "VIVT (Virtually Indexed, Virtually Tagged)",
            "index": "Virtual",
            "tag": "Virtual",
            "tlb_on_hit": False,
            "pros": "Fastest: no TLB lookup needed on cache hit",
            "cons": "Aliasing problems; must flush on context switch",
            "used_in": "Older CPUs, some embedded processors",
        },
        {
            "name": "PIPT (Physically Indexed, Physically Tagged)",
            "index": "Physical",
            "tag": "Physical",
            "tlb_on_hit": True,
            "pros": "No aliasing; no flush on context switch",
            "cons": "TLB lookup on every access (serial with cache)",
            "used_in": "L2/L3 caches (larger, latency-tolerant)",
        },
        {
            "name": "VIPT (Virtually Indexed, Physically Tagged)",
            "index": "Virtual",
            "tag": "Physical",
            "tlb_on_hit": True,
            "pros": "Index cache in parallel with TLB; no aliasing if index < page offset",
            "cons": "Constraint: cache set size <= page size",
            "used_in": "Most modern L1 caches (best compromise)",
        },
    ]

    for d in designs:
        print(f"  {d['name']}:")
        print(f"    Index source: {d['index']}")
        print(f"    Tag source:   {d['tag']}")
        print(f"    TLB needed for hit: {'Yes' if d['tlb_on_hit'] else 'No'}")
        print(f"    Pros: {d['pros']}")
        print(f"    Cons: {d['cons']}")
        print(f"    Used in: {d['used_in']}")
        print()

    print("  Why VIPT is the sweet spot for L1 caches:")
    print("    Index bits come from virtual address (same as physical for bits < page offset)")
    print("    Tag comparison uses physical address from TLB")
    print("    Cache indexing and TLB lookup happen IN PARALLEL")
    print("    Constraint: cache_size / associativity <= page_size")
    print()
    print("    Example: 32KB L1, 8-way = 32KB/8 = 4KB per set = 4KB page size. OK!")
    print("    If 64KB L1, need 16-way to keep sets at 4KB (expensive in hardware).")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Address Translation", exercise_1),
        ("Exercise 2: TLB Simulation", exercise_2),
        ("Exercise 3: Page Table Size Comparison", exercise_3),
        ("Exercise 4: Page Replacement Algorithms", exercise_4),
        ("Exercise 5: TLB Miss vs Page Fault Cost", exercise_5),
        ("Exercise 6: Two-Level Page Table Walk", exercise_6),
        ("Exercise 7: VIVT/VIPT/PIPT Cache Interaction", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

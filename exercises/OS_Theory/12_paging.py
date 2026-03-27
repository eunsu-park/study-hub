"""
Exercises for Lesson 12: Paging
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers address translation, TLB EAT calculation, page table sizing,
multi-level page table memory, and inverted page table design.
"""

import math


# === Exercise 1: Address Translation ===
# Problem: Calculate physical address given page size 4KB,
# logical address 25000, and page table: Page 6 -> Frame 4.

def exercise_1():
    """Perform page-to-frame address translation."""
    page_size = 4096  # 4KB
    logical_addr = 25000

    # Page table mapping (only relevant entry shown)
    page_table = {6: 4}

    # Step 1: Extract page number and offset
    page_number = logical_addr // page_size
    offset = logical_addr % page_size

    print(f"Given: Page size = {page_size} bytes ({page_size // 1024}KB)")
    print(f"       Logical address = {logical_addr}")
    print(f"       Page table: Page 6 -> Frame 4\n")

    print(f"Step 1: Separate logical address into page number and offset")
    print(f"  Page number = {logical_addr} / {page_size} = {page_number}")
    print(f"  Offset      = {logical_addr} % {page_size} = {offset}")
    print(f"  Verification: {page_number} * {page_size} + {offset} = {page_number * page_size + offset}\n")

    # Step 2: Look up page table
    frame_number = page_table[page_number]
    print(f"Step 2: Look up page table")
    print(f"  Page {page_number} -> Frame {frame_number}\n")

    # Step 3: Calculate physical address
    physical_addr = frame_number * page_size + offset
    print(f"Step 3: Calculate physical address")
    print(f"  Physical address = Frame {frame_number} * {page_size} + {offset}")
    print(f"                   = {frame_number * page_size} + {offset}")
    print(f"                   = {physical_addr}\n")

    # Binary verification
    offset_bits = int(math.log2(page_size))
    print(f"Binary verification (offset = {offset_bits} bits):")
    print(f"  Logical  {logical_addr:>6d} = {logical_addr:020b}")
    print(f"    Page number bits:   {logical_addr >> offset_bits:>8b} (= {page_number})")
    print(f"    Offset bits:        {offset:>12b} (= {offset})")
    print(f"  Physical {physical_addr:>6d} = {physical_addr:020b}")
    print(f"    Frame number bits:  {frame_number:>8b} (= {frame_number})")
    print(f"    Offset bits:        {offset:>12b} (= {offset}, same!)")

    # Additional examples
    print("\n--- Additional translations ---\n")
    full_page_table = {0: 2, 1: 7, 2: 1, 3: 5, 4: 8, 5: 3, 6: 4}
    test_addrs = [0, 4095, 4096, 8192, 25000, 20480]
    print(f"  {'Logical':<12} {'Page':<6} {'Offset':<8} {'Frame':<7} {'Physical':<12}")
    print("  " + "-" * 45)
    for addr in test_addrs:
        pn = addr // page_size
        off = addr % page_size
        if pn in full_page_table:
            fn = full_page_table[pn]
            pa = fn * page_size + off
            print(f"  {addr:<12} {pn:<6} {off:<8} {fn:<7} {pa:<12}")
        else:
            print(f"  {addr:<12} {pn:<6} {off:<8} {'N/A':<7} {'Page Fault'}")


# === Exercise 2: TLB EAT Calculation ===
# Problem: Calculate Effective Access Time with TLB.

def exercise_2():
    """Calculate Effective Access Time with TLB."""
    tlb_time = 20      # ns
    mem_time = 100      # ns
    hit_ratio = 0.95    # 95%

    print(f"Given:")
    print(f"  TLB access time (epsilon): {tlb_time}ns")
    print(f"  Memory access time (m):    {mem_time}ns")
    print(f"  TLB hit ratio (alpha):     {hit_ratio * 100}%\n")

    # EAT formula:
    # TLB hit:  epsilon + m  (TLB lookup + one memory access for data)
    # TLB miss: epsilon + 2m (TLB lookup + page table access + data access)
    hit_time = tlb_time + mem_time
    miss_time = tlb_time + 2 * mem_time

    eat = hit_ratio * hit_time + (1 - hit_ratio) * miss_time

    print(f"EAT = alpha * (epsilon + m) + (1 - alpha) * (epsilon + 2m)")
    print(f"    = {hit_ratio} * ({tlb_time} + {mem_time}) + {1 - hit_ratio} * ({tlb_time} + {2 * mem_time})")
    print(f"    = {hit_ratio} * {hit_time} + {1 - hit_ratio} * {miss_time}")
    print(f"    = {hit_ratio * hit_time} + {(1 - hit_ratio) * miss_time}")
    print(f"    = {eat}ns\n")

    no_tlb = 2 * mem_time
    improvement = (no_tlb - eat) / no_tlb * 100
    print(f"Without TLB: {no_tlb}ns (two memory accesses)")
    print(f"With TLB:    {eat}ns")
    print(f"Improvement: {improvement:.1f}%\n")

    # Sensitivity analysis: how EAT changes with hit ratio
    print("--- EAT Sensitivity to Hit Ratio ---\n")
    print(f"  {'Hit Ratio':<12} {'EAT (ns)':<12} {'Slowdown vs ideal'}")
    print("  " + "-" * 40)
    for ratio in [0.80, 0.90, 0.95, 0.98, 0.99, 0.999]:
        e = ratio * hit_time + (1 - ratio) * miss_time
        slowdown = (e - mem_time) / mem_time * 100
        print(f"  {ratio * 100:>5.1f}%      {e:>7.1f}      {slowdown:>+.1f}%")


# === Exercise 3: Page Table Size ===
# Problem: Calculate page table size for 32-bit address space, 4KB pages, 4-byte PTEs.
# Also determine address bit allocation for a two-level page table.

def exercise_3():
    """Calculate page table sizes for single and multi-level tables."""
    addr_bits = 32
    page_size = 4096  # 4KB
    pte_size = 4      # bytes

    offset_bits = int(math.log2(page_size))
    page_bits = addr_bits - offset_bits
    num_pages = 2 ** page_bits

    print(f"System: {addr_bits}-bit virtual address, {page_size // 1024}KB pages, {pte_size}-byte PTEs\n")

    print(f"Q1: Single page table size:")
    print(f"  Offset bits = log2({page_size}) = {offset_bits} bits")
    print(f"  Page number bits = {addr_bits} - {offset_bits} = {page_bits} bits")
    print(f"  Number of pages = 2^{page_bits} = {num_pages:,} (~{num_pages // (1024 * 1024)}M entries)")
    table_size = num_pages * pte_size
    print(f"  Page table size = {num_pages:,} * {pte_size} = {table_size:,} bytes = {table_size // (1024 * 1024)}MB\n")

    print(f"Q2: Two-level page table address split:")
    # Split page_bits (20) into two levels: 10 + 10
    level1_bits = page_bits // 2
    level2_bits = page_bits - level1_bits

    print(f"  Total {addr_bits} bits = offset({offset_bits}) + page table({page_bits})")
    print(f"  Split {page_bits} page bits into two levels: {level1_bits} + {level2_bits}")
    print(f"  Address structure: [p1: {level1_bits} bits][p2: {level2_bits} bits][offset: {offset_bits} bits]\n")

    entries_per_level = 2 ** level1_bits
    level_table_size = entries_per_level * pte_size
    print(f"  Each level table:")
    print(f"    Entries = 2^{level1_bits} = {entries_per_level}")
    print(f"    Size = {entries_per_level} * {pte_size} = {level_table_size} bytes = {level_table_size // 1024}KB")
    print(f"    Exactly fits in 1 page ({page_size // 1024}KB) -- this is intentional!\n")

    print(f"  Advantage: Only allocate level-2 tables for used regions.")
    print(f"  A process using only a small part of address space needs")
    print(f"  far less than {table_size // (1024 * 1024)}MB for its page table.")


# === Exercise 4: Multi-level Table Memory ===
# Problem: Calculate total two-level page table size for a process using only 3 regions.

def exercise_4():
    """Calculate memory usage of a two-level page table for sparse address space."""
    page_size = 4096  # 4KB
    pte_size = 4      # bytes
    offset_bits = 12
    level1_bits = 10
    level2_bits = 10

    regions = [
        ("Code",  0x00000000, 0x00400000),  # 4MB
        ("Data",  0x10000000, 0x10100000),  # 1MB
        ("Stack", 0xBFF00000, 0xC0000000),  # 1MB
    ]

    print(f"Two-level page table: [{level1_bits} bits | {level2_bits} bits | {offset_bits} bits]")
    print(f"Page size: {page_size // 1024}KB, PTE size: {pte_size} bytes\n")

    print(f"Process memory regions:")
    for name, start, end in regions:
        size = end - start
        print(f"  {name}: 0x{start:08X} ~ 0x{end:08X} ({size // (1024 * 1024)}MB)")
    print()

    # Determine which level-1 indices are used
    print(f"Level-1 (p1) index analysis:")
    print(f"  p1 = bits [31:22] of address (top 10 bits)\n")
    used_l1_indices = set()
    for name, start, end in regions:
        p1_start = start >> (level2_bits + offset_bits)
        p1_end = (end - 1) >> (level2_bits + offset_bits)
        indices = list(range(p1_start, p1_end + 1))
        used_l1_indices.update(indices)
        print(f"  {name} (0x{start:08X}~0x{end:08X}):")
        print(f"    p1 range: {p1_start} to {p1_end} ({len(indices)} level-2 table(s))")

    num_l2_tables = len(used_l1_indices)
    l1_table_size = page_size  # 1024 entries * 4 bytes = 4KB
    l2_table_size = page_size  # same

    total_size = l1_table_size + num_l2_tables * l2_table_size

    print(f"\nRequired tables:")
    print(f"  Level-1 table: 1 * {l1_table_size // 1024}KB = {l1_table_size // 1024}KB")
    print(f"  Level-2 tables: {num_l2_tables} * {l2_table_size // 1024}KB = {num_l2_tables * l2_table_size // 1024}KB")
    print(f"  (for p1 indices: {sorted(used_l1_indices)})")

    print(f"\nTotal page table size = {l1_table_size // 1024}KB + {num_l2_tables * l2_table_size // 1024}KB = {total_size // 1024}KB")

    single_table_size = 4 * 1024 * 1024  # 4MB
    savings = (single_table_size - total_size) / single_table_size * 100
    print(f"\nComparison:")
    print(f"  Single-level table: {single_table_size // (1024 * 1024)}MB")
    print(f"  Two-level table:    {total_size // 1024}KB")
    print(f"  Savings: ({single_table_size // (1024 * 1024)}MB - {total_size // 1024}KB) / {single_table_size // (1024 * 1024)}MB = {savings:.1f}%")


# === Exercise 5: Inverted Page Table Design ===
# Problem: Design an inverted page table for 1GB physical memory, 8KB page size.

def exercise_5():
    """Design an inverted page table and calculate its size."""
    phys_mem = 1 * 1024 * 1024 * 1024   # 1GB
    page_size = 8 * 1024                  # 8KB
    virt_addr_bits = 64                   # assume 64-bit system

    num_frames = phys_mem // page_size
    offset_bits = int(math.log2(page_size))
    vpn_bits = virt_addr_bits - offset_bits

    print(f"System specifications:")
    print(f"  Physical memory: {phys_mem // (1024 * 1024 * 1024)}GB")
    print(f"  Page size: {page_size // 1024}KB")
    print(f"  Virtual address: {virt_addr_bits}-bit\n")

    print(f"Q1: Number of frames (= number of entries in inverted page table):")
    print(f"  Frames = {phys_mem // (1024 * 1024)}MB / {page_size // 1024}KB")
    print(f"         = 2^{int(math.log2(phys_mem))} / 2^{offset_bits}")
    print(f"         = 2^{int(math.log2(num_frames))} = {num_frames:,} frames\n")

    # Entry structure design
    pid_bits = 16
    protection_bits = 4   # R/W/X/Valid
    chain_bits = 17       # for hash chaining (log2(num_frames))
    spare_bits = 80 - pid_bits - vpn_bits - protection_bits  # pad to nice boundary

    print(f"Q2: Entry structure:")
    print(f"  {'Field':<25} {'Bits':<8} {'Purpose'}")
    print("  " + "-" * 60)
    print(f"  {'PID':<25} {pid_bits:<8} {'Identifies owning process (max 65536)'}")
    print(f"  {'Virtual Page Number':<25} {vpn_bits:<8} {f'{virt_addr_bits}-bit addr - {offset_bits}-bit offset'}")
    print(f"  {'Protection':<25} {protection_bits:<8} {'R/W/X/Valid flags'}")
    print(f"  {'Hash chain pointer':<25} {chain_bits:<8} {'For collision resolution'}")

    # Round entry size to a power-of-2 byte boundary
    total_bits = pid_bits + vpn_bits + protection_bits + chain_bits
    entry_bytes = 16  # pad to 16 bytes for alignment
    print(f"\n  Total meaningful bits: {total_bits}")
    print(f"  Entry size (padded): {entry_bytes} bytes\n")

    table_size = num_frames * entry_bytes
    table_size_mb = table_size / (1024 * 1024)
    overhead_pct = table_size / phys_mem * 100

    print(f"Q3: Total inverted page table size:")
    print(f"  Entries: {num_frames:,}")
    print(f"  Entry size: {entry_bytes} bytes")
    print(f"  Total: {num_frames:,} * {entry_bytes} = {table_size:,} bytes = {table_size_mb:.1f}MB")
    print(f"  Overhead: {table_size_mb:.1f}MB / {phys_mem // (1024 * 1024 * 1024)}GB = {overhead_pct:.1f}% of physical memory\n")

    print(f"Key advantage: Table size is proportional to PHYSICAL memory,")
    print(f"not virtual address space. A traditional page table for a")
    print(f"{virt_addr_bits}-bit address space would be astronomically large.")

    # Comparison with traditional page table
    traditional_entries = 2 ** vpn_bits
    traditional_size = traditional_entries * 4  # 4-byte PTE
    print(f"\n  Traditional table: 2^{vpn_bits} entries * 4 bytes = impractically large")
    print(f"  Inverted table:   {num_frames:,} entries * {entry_bytes} bytes = {table_size_mb:.1f}MB")
    print(f"\n  Trade-off: Inverted tables make lookup slower (hash + chain traversal)")
    print(f"  but use dramatically less memory. TLBs mitigate the lookup cost.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Address Translation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: TLB EAT Calculation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Page Table Size ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Multi-level Table Memory ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Inverted Page Table Design ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")

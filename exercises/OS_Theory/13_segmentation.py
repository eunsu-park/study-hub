"""
Exercises for Lesson 13: Segmentation
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers segment address translation, segment protection, segment sharing,
external fragmentation, and x86 segment selectors.
"""


# === Exercise 1: Segment Address Translation ===
# Problem: Translate logical address <1, 500> using a segment table.

def exercise_1():
    """Perform segment-based address translation."""
    # Segment table: (base, limit)
    segment_table = {
        0: (1000, 600),
        1: (2000, 400),
        2: (3000, 800),
    }

    logical_addr = (1, 500)  # (segment_number, offset)
    seg_num, offset = logical_addr

    print(f"Segment table:")
    print(f"  {'Segment':<10} {'Base':<10} {'Limit':<10}")
    print("  " + "-" * 30)
    for seg, (base, limit) in segment_table.items():
        print(f"  {seg:<10} {base:<10} {limit:<10}")

    print(f"\nTranslate logical address <{seg_num}, {offset}>:\n")

    base, limit = segment_table[seg_num]
    print(f"  Step 1: Look up segment {seg_num}")
    print(f"    Base = {base}, Limit = {limit}\n")

    print(f"  Step 2: Check offset against limit")
    print(f"    Offset ({offset}) < Limit ({limit})? {offset < limit}")

    if offset < limit:
        physical = base + offset
        print(f"\n  Step 3: Physical address = Base + Offset")
        print(f"    = {base} + {offset} = {physical}")
    else:
        print(f"\n  SEGMENTATION FAULT!")
        print(f"  Offset {offset} >= Limit {limit}: segment boundary exceeded")
        print(f"  The OS would trap and typically terminate the process.")

    # Additional test cases
    print("\n--- Additional translations ---\n")
    test_cases = [
        (0, 100),   # Valid
        (0, 599),   # Edge case: exactly at limit-1
        (0, 600),   # At limit: fault
        (1, 500),   # Exceeds limit: fault (original problem)
        (1, 399),   # Edge case: exactly at limit-1
        (2, 0),     # Start of segment
        (2, 800),   # At limit: fault
    ]

    print(f"  {'Address':<15} {'Base':<8} {'Limit':<8} {'Check':<20} {'Result'}")
    print("  " + "-" * 65)
    for seg, off in test_cases:
        b, l = segment_table[seg]
        if off < l:
            phys = b + off
            result = f"Physical: {phys}"
            check = f"{off} < {l}: OK"
        else:
            result = "SEGFAULT"
            check = f"{off} >= {l}: FAIL"
        print(f"  <{seg}, {off:<5}>{'':<5} {b:<8} {l:<8} {check:<20} {result}")


# === Exercise 2: Segment Protection ===
# Problem: Check protection violations in various access scenarios.

def exercise_2():
    """Determine protection violations for segment accesses."""
    # Segment table: (base, limit, protection)
    # Protection: R=read, W=write, X=execute
    segment_table = {
        0: {"name": "code", "base": 0x1000, "limit": 2000, "prot": "R-X"},
        1: {"name": "data", "base": 0x5000, "limit": 3000, "prot": "RW-"},
    }

    print(f"Segment table:")
    print(f"  {'Seg':<5} {'Name':<8} {'Base':<10} {'Limit':<8} {'Protection'}")
    print("  " + "-" * 40)
    for seg, info in segment_table.items():
        print(f"  {seg:<5} {info['name']:<8} 0x{info['base']:04X}{'':>2} {info['limit']:<8} {info['prot']}")
    print()

    # Access scenarios
    scenarios = [
        (0, 500, "X", "Fetch instruction from segment 0 at offset 500"),
        (0, 100, "W", "Write data to segment 0 at offset 100"),
        (1, 2500, "R", "Read from segment 1 at offset 2500"),
    ]

    for i, (seg, offset, access_type, description) in enumerate(scenarios, 1):
        info = segment_table[seg]
        access_names = {"R": "read", "W": "write", "X": "execute"}

        print(f"Scenario {i}: {description}")
        print(f"  Segment {seg} ({info['name']}), offset {offset}, access: {access_names[access_type]}")

        # Check range
        range_ok = offset < info["limit"]
        print(f"  Range check: {offset} < {info['limit']}? {range_ok}", end="")
        if not range_ok:
            print(f" -> SEGMENTATION FAULT (out of bounds)")
            print()
            continue
        else:
            print(f" -> OK")

        # Check protection
        prot_ok = access_type in info["prot"]
        print(f"  Protection check: '{access_type}' in '{info['prot']}'? {prot_ok}", end="")
        if not prot_ok:
            print(f" -> PROTECTION FAULT")
            print(f"  The OS traps: segment {seg} ({info['name']}) does not allow {access_names[access_type]}.")
        else:
            physical = info["base"] + offset
            print(f" -> OK")
            print(f"  Access granted. Physical address = 0x{info['base']:04X} + {offset} = 0x{physical:04X}")
        print()


# === Exercise 3: Segment Sharing ===
# Problem: Create segment tables for two processes sharing a library.

def exercise_3():
    """Demonstrate segment sharing between processes."""
    shared_lib_phys = 0x10000
    shared_lib_size = 1000

    print(f"Shared library loaded at physical address 0x{shared_lib_phys:05X}, size {shared_lib_size} bytes")
    print(f"Process A maps it to segment 2, Process B maps it to segment 3\n")

    proc_a = {
        0: {"desc": "A's code",       "base": 0x05000, "limit": 2000, "prot": "R-X"},
        1: {"desc": "A's data",       "base": 0x08000, "limit": 1500, "prot": "RW-"},
        2: {"desc": "Shared library", "base": 0x10000, "limit": 1000, "prot": "R-X"},
    }

    proc_b = {
        0: {"desc": "B's code",       "base": 0x20000, "limit": 3000, "prot": "R-X"},
        1: {"desc": "B's data",       "base": 0x25000, "limit": 2000, "prot": "RW-"},
        2: {"desc": "B's other",      "base": 0x28000, "limit": 500,  "prot": "RW-"},
        3: {"desc": "Shared library", "base": 0x10000, "limit": 1000, "prot": "R-X"},
    }

    for name, table in [("Process A", proc_a), ("Process B", proc_b)]:
        print(f"{name} Segment Table:")
        print(f"  {'Seg':<5} {'Base':<10} {'Limit':<8} {'Prot':<6} {'Description'}")
        print("  " + "-" * 45)
        for seg, info in table.items():
            print(f"  {seg:<5} 0x{info['base']:05X}{'':<2} {info['limit']:<8} {info['prot']:<6} {info['desc']}")
        print()

    print(f"Key observations:")
    print(f"  - A's segment 2 and B's segment 3 both point to 0x{shared_lib_phys:05X}")
    print(f"  - The library code exists ONCE in physical memory")
    print(f"  - Both processes can execute it independently")
    print(f"  - Protection is R-X (read + execute, no write) to prevent")
    print(f"    one process from corrupting the shared code")
    print(f"  - Each process maps the library to a DIFFERENT segment number --")
    print(f"    the sharing is at the physical level, not the logical level")


# === Exercise 4: External Fragmentation ===
# Problem: Explain why external fragmentation occurs with segmentation.

def exercise_4():
    """Demonstrate external fragmentation with segmentation."""
    total_mem = 2000  # bytes (simplified)

    # Simulate a sequence of allocations and deallocations
    print(f"Memory: {total_mem} bytes total\n")

    # Initial state after several operations
    layout = [
        ("Seg A", 400),
        ("Hole",  500),
        ("Seg B", 300),
        ("Hole",  300),
        ("Seg C", 200),
        ("Hole",  200),
        ("Seg D", 100),
    ]

    print("Current memory layout:")
    addr = 0
    total_free = 0
    max_hole = 0
    num_holes = 0
    for name, size in layout:
        status = "FREE" if name == "Hole" else "USED"
        print(f"  [{name}: {size} bytes] @ {addr} ({status})")
        addr += size
        if name == "Hole":
            total_free += size
            max_hole = max(max_hole, size)
            num_holes += 1

    print(f"\nFragmentation analysis:")
    print(f"  Total free space: {total_free} bytes in {num_holes} fragments")
    print(f"  Largest hole: {max_hole} bytes")
    print(f"  Smallest hole: {min(s for n, s in layout if n == 'Hole')} bytes\n")

    new_seg_size = 600
    print(f"New segment E needs {new_seg_size} bytes:")
    print(f"  Total free ({total_free}) >= needed ({new_seg_size})? {total_free >= new_seg_size}")
    print(f"  Largest hole ({max_hole}) >= needed ({new_seg_size})? {max_hole >= new_seg_size}")
    print(f"  Result: ALLOCATION FAILS -- external fragmentation!\n")

    print("Solutions:")
    print("  1. Compaction:")
    print("     Move all segments to one end, creating one large hole.")
    addr = 0
    for name, size in layout:
        if name != "Hole":
            print(f"       [{name}: {size}] @ {addr}")
            addr += size
    print(f"       [Free: {total_free}] @ {addr}")
    print(f"     Now segment E ({new_seg_size}) fits in the {total_free}-byte hole.\n")

    print("  2. Combine with paging (segmented paging):")
    print("     Divide each segment into fixed-size pages.")
    print("     Pages can be scattered across frames.")
    print("     Eliminates external fragmentation entirely.")
    print("     This is the approach used by modern x86 systems.\n")

    print("  3. Buddy system:")
    print("     Divide memory into power-of-2 sized blocks.")
    print("     Fast merging of adjacent free blocks.")
    print("     Reduces but does not eliminate fragmentation.")


# === Exercise 5: x86 Segment Selector ===
# Problem: Interpret CS register value 0x23 in Linux.

def exercise_5():
    """Interpret x86 segment selector values."""
    cs_value = 0x23

    print(f"x86 Segment Selector: CS = 0x{cs_value:02X}\n")

    # Segment selector structure: [Index(13 bits) | TI(1 bit) | RPL(2 bits)]
    rpl = cs_value & 0x3            # bits 1:0
    ti = (cs_value >> 2) & 0x1      # bit 2
    index = (cs_value >> 3) & 0x1FFF  # bits 15:3

    print(f"Binary: 0x{cs_value:04X} = {cs_value:016b}\n")
    print(f"Field breakdown:")
    print(f"  Bits [15:3] Index = {index} (GDT/LDT entry number)")
    print(f"  Bit  [2]    TI    = {ti} ({'LDT' if ti else 'GDT'})")
    print(f"  Bits [1:0]  RPL   = {rpl} (Ring {rpl})\n")

    ti_name = "LDT (Local Descriptor Table)" if ti else "GDT (Global Descriptor Table)"
    ring_desc = {0: "kernel mode", 1: "ring 1 (unused)", 2: "ring 2 (unused)", 3: "user mode"}

    print(f"Interpretation:")
    print(f"  Index {index} in {ti_name}")
    print(f"  Ring {rpl} = {ring_desc[rpl]}")
    print(f"  This is GDT entry {index}: GDT_ENTRY_DEFAULT_USER_CS (user code segment)\n")

    # Verify with Linux kernel constant
    gdt_entry = 4  # GDT_ENTRY_DEFAULT_USER_CS
    expected = gdt_entry * 8 + 3  # *8 shifts index into bits[15:3], +3 sets RPL=3
    print(f"Verification with Linux kernel:")
    print(f"  #define __USER_CS (GDT_ENTRY_DEFAULT_USER_CS * 8 + 3)")
    print(f"                   = ({gdt_entry} * 8 + 3)")
    print(f"                   = {gdt_entry * 8} + 3")
    print(f"                   = {expected}")
    print(f"                   = 0x{expected:02X}")
    print(f"  Match: {expected == cs_value}\n")

    # Decode other common Linux segment selectors
    print("--- Common Linux segment selectors ---\n")
    selectors = [
        (0x10, "__KERNEL_CS", "Kernel code segment"),
        (0x18, "__KERNEL_DS", "Kernel data segment"),
        (0x23, "__USER_CS",   "User code segment"),
        (0x2B, "__USER_DS",   "User data segment"),
        (0x33, "__USER_CS (64-bit)", "User code segment (long mode)"),
    ]

    print(f"  {'Value':<8} {'Constant':<22} {'Index':<7} {'TI':<5} {'RPL':<5} {'Description'}")
    print("  " + "-" * 65)
    for val, name, desc in selectors:
        r = val & 0x3
        t = (val >> 2) & 0x1
        idx = (val >> 3) & 0x1FFF
        print(f"  0x{val:02X}    {name:<22} {idx:<7} {'LDT' if t else 'GDT':<5} {r:<5} {desc}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Segment Address Translation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Segment Protection ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Segment Sharing ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: External Fragmentation ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: x86 Segment Selector ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")

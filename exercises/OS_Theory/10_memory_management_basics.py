"""
Exercises for Lesson 10: Memory Management Basics
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers address binding modes, MMU relocation with base/limit registers,
swap time calculation, and dynamic loading vs dynamic linking.
"""


# === Exercise 1: Address Translation ===
# Problem: Translate logical addresses using relocation and limit registers.

def exercise_1():
    """Address translation with relocation and limit registers."""
    relocation = 0x4000
    limit = 0x3000

    print(f"Given: Relocation register = 0x{relocation:04X}")
    print(f"       Limit register = 0x{limit:04X}\n")

    # Test case 1
    logical_addr_1 = 0x1500
    physical_addr_1 = relocation + logical_addr_1
    within_limit_1 = logical_addr_1 < limit

    print(f"Q1: Logical address 0x{logical_addr_1:04X}")
    print(f"  Check: 0x{logical_addr_1:04X} < 0x{limit:04X}? {within_limit_1}")
    if within_limit_1:
        print(f"  Physical address = 0x{relocation:04X} + 0x{logical_addr_1:04X} = 0x{physical_addr_1:04X}")
        print(f"  Result: Valid access, physical address is 0x{physical_addr_1:04X}")
    else:
        print(f"  Result: Segmentation Fault! Address exceeds limit.")
    print()

    # Test case 2
    logical_addr_2 = 0x3500
    within_limit_2 = logical_addr_2 < limit

    print(f"Q2: Logical address 0x{logical_addr_2:04X}")
    print(f"  Check: 0x{logical_addr_2:04X} < 0x{limit:04X}? {within_limit_2}")
    if within_limit_2:
        physical_addr_2 = relocation + logical_addr_2
        print(f"  Physical address = 0x{relocation:04X} + 0x{logical_addr_2:04X} = 0x{physical_addr_2:04X}")
    else:
        print(f"  0x{logical_addr_2:04X} >= 0x{limit:04X} -- EXCEEDS LIMIT!")
        print(f"  Result: Segmentation Fault (trap to OS)")
        print(f"  The OS will typically terminate the process or send SIGSEGV.")

    print("\n  Address translation diagram:")
    print(f"  Logical addr -> [< Limit?] --YES--> [+ Relocation] -> Physical addr")
    print(f"                      |")
    print(f"                      NO -> Segmentation Fault (trap)")

    # Additional examples for deeper understanding
    print("\n\n--- Additional address translation examples ---\n")

    test_addrs = [0x0000, 0x1000, 0x2FFF, 0x3000, 0x3001, 0xFFFF]
    print(f"  {'Logical':<12} {'< Limit?':<12} {'Physical':<14} {'Result'}")
    print("  " + "-" * 50)
    for addr in test_addrs:
        valid = addr < limit
        if valid:
            phys = relocation + addr
            print(f"  0x{addr:04X}      {'Yes':<12} 0x{phys:04X}        Valid")
        else:
            print(f"  0x{addr:04X}      {'No':<12} {'N/A':<14} Seg Fault")


# === Exercise 2: Address Binding Modes ===
# Problem: Identify and compare compile-time, load-time, and execution-time binding.
# For each binding mode, determine when the logical-to-physical mapping is fixed
# and describe the hardware/OS support required.

def exercise_2():
    """Compare address binding modes and their characteristics."""
    print("Q1: When is the address binding fixed for each mode?\n")

    modes = [
        {
            "name": "Compile-time binding",
            "fixed_at": "Compilation",
            "address_type": "Absolute (physical) addresses embedded in binary",
            "relocation": False,
            "example": "Early MS-DOS .COM files; single fixed address space",
            "pros": "Simplest; no translation overhead at runtime",
            "cons": "Process must always load at same physical address; no multiprogramming",
        },
        {
            "name": "Load-time binding",
            "fixed_at": "Program loading (before first instruction executes)",
            "address_type": "Relocatable addresses in object file; absolute after loader fixes them",
            "relocation": True,
            "example": "Early Unix a.out with static relocation table",
            "pros": "Flexible load address; no hardware MMU required",
            "cons": "Address fixed after load; process cannot be moved while running (no compaction)",
        },
        {
            "name": "Execution-time (dynamic) binding",
            "fixed_at": "Each memory access at runtime",
            "address_type": "Logical addresses throughout; MMU translates on every access",
            "relocation": True,
            "example": "Modern OS with paging/segmentation (Linux, Windows)",
            "pros": "Process can be relocated, swapped, or compacted while running",
            "cons": "Requires hardware MMU; small translation latency per access (mitigated by TLB)",
        },
    ]

    for i, m in enumerate(modes, 1):
        print(f"  Mode {i}: {m['name']}")
        print(f"    Binding fixed at : {m['fixed_at']}")
        print(f"    Address type     : {m['address_type']}")
        print(f"    Requires reloc.  : {'Yes' if m['relocation'] else 'No'}")
        print(f"    Typical example  : {m['example']}")
        print(f"    Pros             : {m['pros']}")
        print(f"    Cons             : {m['cons']}")
        print()

    print("Q2: Which binding mode does a modern Linux process use?")
    print("  Answer: Execution-time binding.")
    print("  The kernel assigns virtual (logical) addresses at compile/link time,")
    print("  but the MMU maps them to physical frames dynamically on each access.")
    print("  This allows demand paging, swapping, and ASLR.\n")

    print("Q3: Why can't a load-time-bound process be compacted?")
    print("  Because addresses were patched to absolute physical values at load time.")
    print("  Moving the process in memory would require re-patching every address")
    print("  embedded in the code and data -- impractical at runtime without")
    print("  a complete relocation table and support from the running process itself.")


# === Exercise 3: MMU Relocation with Base/Limit Registers ===
# Problem: A system uses base/limit registers for memory protection.
# Given three processes with different base/limit values, calculate the physical
# address for each logical access and flag any illegal accesses.

def exercise_3():
    """MMU base/limit register calculations for multiple processes."""
    # Each process: (name, base, limit)
    processes = [
        ("P1", 0x08000, 0x04000),   # base=32KB, limit=16KB
        ("P2", 0x20000, 0x08000),   # base=128KB, limit=32KB
        ("P3", 0x40000, 0x02000),   # base=256KB, limit=8KB
    ]

    # Access requests: (process_name, logical_address)
    accesses = [
        ("P1", 0x0000),    # valid: first byte
        ("P1", 0x03FFF),   # valid: last legal byte
        ("P1", 0x04000),   # invalid: equals limit (limit is exclusive bound)
        ("P2", 0x07FFF),   # valid: last byte
        ("P2", 0x08001),   # invalid: exceeds limit
        ("P3", 0x0100),    # valid
        ("P3", 0x2000),    # invalid: exceeds limit
    ]

    proc_map = {name: (base, limit) for name, base, limit in processes}

    print("Process memory layout:")
    print(f"  {'Process':<10} {'Base':<12} {'Limit':<12} {'End (base+limit)'}")
    print("  " + "-" * 50)
    for name, base, limit in processes:
        end = base + limit
        print(f"  {name:<10} 0x{base:05X}      0x{limit:05X}      0x{end:05X}")

    print()
    print("Logical address access results:")
    print(f"  {'Process':<8} {'Logical':<12} {'< Limit?':<10} {'Physical':<12} {'Result'}")
    print("  " + "-" * 56)

    for proc_name, logical in accesses:
        base, limit = proc_map[proc_name]
        valid = logical < limit
        if valid:
            physical = base + logical
            result = "OK"
            phys_str = f"0x{physical:05X}"
        else:
            result = "TRAP (Seg Fault)"
            phys_str = "N/A"
        print(f"  {proc_name:<8} 0x{logical:05X}      {'Yes' if valid else 'No':<10} {phys_str:<12} {result}")

    print()
    print("Key insight: The limit register enforces that a process cannot")
    print("access memory outside its own partition. The OS saves and restores")
    print("base/limit registers on every context switch.")

    print()
    print("Q: If P1 (base=0x08000, limit=0x04000) is relocated to base=0x50000,")
    print("   what physical address does logical 0x1000 translate to?")
    new_base = 0x50000
    logical = 0x1000
    new_phys = new_base + logical
    print(f"  Old physical: 0x08000 + 0x1000 = 0x{0x08000 + logical:05X}")
    print(f"  New physical: 0x{new_base:05X} + 0x{logical:04X} = 0x{new_phys:05X}")
    print("  Only the base register changes -- the process binary is unchanged.")


# === Exercise 4: Swap Time Calculation ===
# Problem: Calculate total swap-out and swap-in times for processes of different sizes.
# Given disk bandwidth and seek/latency overhead, find total swap time.

def exercise_4():
    """Compute swap time for processes given disk bandwidth and transfer size."""
    # System parameters
    disk_bandwidth_mb_s = 200     # MB/s sustained transfer rate
    avg_seek_latency_ms = 8       # ms per seek (seek + rotational avg)
    context_switch_overhead_ms = 1  # ms OS overhead per swap

    # Processes to swap
    processes = [
        ("P_small",  5),    # 5 MB
        ("P_medium", 50),   # 50 MB
        ("P_large",  200),  # 200 MB
    ]

    print(f"Disk parameters:")
    print(f"  Bandwidth          : {disk_bandwidth_mb_s} MB/s")
    print(f"  Avg seek+latency   : {avg_seek_latency_ms} ms")
    print(f"  Context switch OH  : {context_switch_overhead_ms} ms")
    print()

    print("Swap time = seek_latency + transfer_time")
    print("  transfer_time = process_size / bandwidth")
    print("  Full swap cycle = swap-out + swap-in (two disk transfers)\n")

    print(f"  {'Process':<12} {'Size':<8} {'Transfer (ms)':<16} {'Swap-out (ms)':<16} {'Swap-in (ms)':<15} {'Total cycle (ms)'}")
    print("  " + "-" * 82)

    for name, size_mb in processes:
        transfer_ms = (size_mb / disk_bandwidth_mb_s) * 1000   # convert s -> ms
        swap_out_ms = avg_seek_latency_ms + transfer_ms
        swap_in_ms  = avg_seek_latency_ms + transfer_ms
        total_ms    = swap_out_ms + swap_in_ms + context_switch_overhead_ms

        print(f"  {name:<12} {size_mb:>4} MB  {transfer_ms:>10.1f}     "
              f"{swap_out_ms:>11.1f}     {swap_in_ms:>11.1f}     {total_ms:>10.1f}")

    print()
    print("Q: Why does process size dominate swap time for large processes?")
    print("  The seek/latency component (8 ms) is fixed and small compared to")
    print("  the transfer of a 200 MB process (1000 ms). For large processes,")
    print("  swap overhead is primarily determined by disk bandwidth, not latency.")

    print()
    print("Q: What techniques reduce effective swap overhead?")
    swaps = [
        "Only swap modified (dirty) pages rather than the entire process",
        "Use faster storage (SSD ~500 MB/s vs HDD ~200 MB/s) to cut transfer time",
        "Demand paging: bring in only the needed pages, not the whole process",
        "Prefetching: load pages the process is likely to need before it faults",
    ]
    for i, s in enumerate(swaps, 1):
        print(f"  {i}. {s}")


# === Exercise 5: Dynamic Loading vs Dynamic Linking ===
# Problem: Compare dynamic loading and dynamic linking across key dimensions.
# Determine memory savings for a system with many processes sharing a library.

def exercise_5():
    """Compare dynamic loading and dynamic linking; compute sharing savings."""
    print("Q1: Key differences between dynamic loading and dynamic linking\n")

    comparison = [
        ("Aspect",             "Dynamic Loading",                        "Dynamic Linking (Shared Libs)"),
        ("-" * 20,             "-" * 38,                                 "-" * 38),
        ("When resolved",      "At explicit load call (dlopen/LoadLib)", "At first call or program start"),
        ("Who triggers it",    "Application code (programmer decides)",  "OS loader / dynamic linker (ld.so)"),
        ("Typical use",        "Plugins, optional modules, drivers",     "System libs: libc, libpthread, Qt"),
        ("Code location",      "Loaded into process address space",      "Mapped as shared anonymous region"),
        ("Memory sharing",     "Each process has own copy (usually)",    "One physical copy, many virtual maps"),
        ("Relocation",         "At load time (fixed after load)",        "Position-independent code (PIC) + GOT/PLT"),
        ("Unloading",          "App calls dlclose() explicitly",         "Reference-counted; unloaded when last user exits"),
        ("Example API",        "dlopen/dlsym/dlclose (POSIX)",           "ELF shared objects (.so); PE DLL (.dll)"),
    ]

    for row in comparison:
        print(f"  {row[0]:<22} {row[1]:<40} {row[2]}")

    print()
    print("Q2: Memory savings from dynamic linking (shared library)\n")

    num_processes   = 40        # processes all using the same library
    lib_size_kb     = 2048      # 2 MB shared library (e.g., libc)
    stub_size_kb    = 8         # per-process stub/PLT overhead in KB

    # Without shared library: each process gets a static copy
    static_total_kb = num_processes * lib_size_kb

    # With shared library: one physical copy + per-process stub
    shared_physical_kb = lib_size_kb
    shared_total_kb    = shared_physical_kb + num_processes * stub_size_kb

    savings_kb  = static_total_kb - shared_total_kb
    savings_pct = savings_kb / static_total_kb * 100

    print(f"  Scenario: {num_processes} processes each using a {lib_size_kb // 1024} MB library")
    print()
    print(f"  Static linking (each process gets a full copy):")
    print(f"    Physical memory used = {num_processes} x {lib_size_kb} KB = {static_total_kb:,} KB "
          f"({static_total_kb // 1024} MB)")
    print()
    print(f"  Dynamic linking (one shared physical copy):")
    print(f"    Library physical copy  = {shared_physical_kb} KB ({shared_physical_kb // 1024} MB)")
    print(f"    Per-process stub overhead = {num_processes} x {stub_size_kb} KB = {num_processes * stub_size_kb} KB")
    print(f"    Total physical memory  = {shared_total_kb:,} KB ({shared_total_kb // 1024} MB)")
    print()
    print(f"  Memory savings = {static_total_kb:,} KB - {shared_total_kb:,} KB = {savings_kb:,} KB ({savings_kb // 1024} MB)")
    print(f"  Savings percentage = {savings_pct:.1f}%")

    print()
    print("Q3: What is Position-Independent Code (PIC) and why does dynamic linking need it?")
    print("  PIC uses the Global Offset Table (GOT) and Procedure Linkage Table (PLT)")
    print("  to reference external symbols through an indirection table rather than")
    print("  hardcoded absolute addresses. This allows a single .so to be mapped at")
    print("  different virtual addresses in different processes without relocation patches.")
    print("  Without PIC, the OS would need a separate physical copy per process.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Address Translation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Address Binding Modes ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: MMU Relocation with Base/Limit Registers ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Swap Time Calculation ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Dynamic Loading vs Dynamic Linking ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")

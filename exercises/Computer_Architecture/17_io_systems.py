"""
Exercises for Lesson 17: I/O Systems
Topic: Computer_Architecture

Solutions to practice problems covering programmed I/O (polling),
interrupt-driven I/O, DMA (Direct Memory Access), bus architecture,
I/O performance analysis, and memory-mapped vs port-mapped I/O.
"""


def exercise_1():
    """
    Compare programmed I/O (polling), interrupt-driven I/O, and DMA.
    Calculate CPU utilization for each method.
    """
    print("I/O Methods Comparison:")
    print()

    methods = [
        {
            "name": "Programmed I/O (Polling)",
            "mechanism": "CPU repeatedly checks device status register in a loop",
            "cpu_involved": "100% during transfer",
            "best_for": "Simple, low-speed devices (LED, button)",
            "overhead": "Wastes CPU cycles polling",
        },
        {
            "name": "Interrupt-Driven I/O",
            "mechanism": "Device interrupts CPU when data is ready",
            "cpu_involved": "Only during interrupt handling",
            "best_for": "Medium-speed, asynchronous devices (keyboard, mouse)",
            "overhead": "Context switch overhead per interrupt",
        },
        {
            "name": "DMA (Direct Memory Access)",
            "mechanism": "DMA controller transfers data directly to/from memory",
            "cpu_involved": "Only to set up and finalize transfer",
            "best_for": "High-speed, bulk transfers (disk, network, GPU)",
            "overhead": "DMA controller hardware cost; bus contention",
        },
    ]

    for m in methods:
        print(f"  {m['name']}:")
        print(f"    Mechanism:    {m['mechanism']}")
        print(f"    CPU involved: {m['cpu_involved']}")
        print(f"    Best for:     {m['best_for']}")
        print(f"    Overhead:     {m['overhead']}")
        print()

    # CPU utilization comparison
    print("  CPU Utilization Example:")
    print("  Transfer 1 MB from disk at 100 MB/s, CPU clock = 4 GHz")
    print()

    transfer_size = 1 * 1024 * 1024  # 1 MB
    disk_speed = 100 * 1024 * 1024    # 100 MB/s
    cpu_clock = 4e9                   # 4 GHz
    transfer_time = transfer_size / disk_speed  # seconds

    # Polling: CPU checks status every byte (worst case)
    poll_cycles_per_check = 10
    checks_polling = transfer_size  # one check per byte
    poll_cpu_cycles = checks_polling * poll_cycles_per_check
    poll_utilization = poll_cpu_cycles / (cpu_clock * transfer_time)

    # Interrupt: one interrupt per 512 bytes (sector)
    sector_size = 512
    interrupts = transfer_size // sector_size
    isr_cycles = 500  # cycles per interrupt service routine
    int_cpu_cycles = interrupts * isr_cycles
    int_utilization = int_cpu_cycles / (cpu_clock * transfer_time)

    # DMA: CPU sets up transfer + one interrupt at end
    setup_cycles = 1000
    dma_cpu_cycles = setup_cycles + isr_cycles
    dma_utilization = dma_cpu_cycles / (cpu_clock * transfer_time)

    results = [
        ("Polling", poll_cpu_cycles, poll_utilization),
        ("Interrupt-driven", int_cpu_cycles, int_utilization),
        ("DMA", dma_cpu_cycles, dma_utilization),
    ]

    print(f"  {'Method':<20s} {'CPU Cycles':>14s} {'CPU Utilization':>16s}")
    print(f"  {'-'*20} {'-'*14} {'-'*16}")

    for name, cycles, util in results:
        util_display = min(util, 1.0)
        print(f"  {name:<20s} {cycles:>14,d} {util_display:>15.4%}")

    print()
    print("  DMA frees the CPU almost entirely during bulk data transfers.")


def exercise_2():
    """
    Simulate a DMA transfer with cycle-stealing and burst modes.
    """
    print("DMA Transfer Modes:")
    print()

    total_words = 16  # words to transfer
    mem_cycle = 1     # 1 cycle per memory access

    print(f"  Transfer: {total_words} words, memory cycle = {mem_cycle} cycle(s)")
    print()

    # Cycle-stealing mode: DMA steals one bus cycle at a time
    print("  1. Cycle-Stealing Mode:")
    print("     DMA takes one memory cycle, then releases bus for CPU.")
    print()
    print("     Timeline (C=CPU, D=DMA):")

    timeline_cs = []
    dma_remaining = total_words
    cycle = 0
    while dma_remaining > 0:
        # DMA steals one cycle
        timeline_cs.append("D")
        dma_remaining -= 1
        cycle += 1
        # CPU gets one cycle
        if dma_remaining > 0:
            timeline_cs.append("C")
            cycle += 1

    cs_display = " ".join(timeline_cs[:32])
    print(f"     {cs_display}")
    cs_total_cycles = len(timeline_cs)
    cs_cpu_cycles = sum(1 for x in timeline_cs if x == "C")
    print(f"     Total cycles: {cs_total_cycles}, CPU cycles available: {cs_cpu_cycles}")
    print()

    # Burst mode: DMA holds bus for entire transfer
    print("  2. Burst Mode:")
    print("     DMA holds bus for entire block transfer.")
    print()
    burst_dma_cycles = total_words * mem_cycle
    timeline_burst = ["D"] * burst_dma_cycles + ["C"] * burst_dma_cycles
    burst_display = " ".join(timeline_burst[:32])
    print(f"     {burst_display} ...")
    print(f"     DMA phase: {burst_dma_cycles} cycles, then CPU resumes")
    print()

    # Transparent mode: DMA uses bus only when CPU doesn't need it
    print("  3. Transparent Mode:")
    print("     DMA transfers only when CPU is not using the bus.")
    print("     No CPU slowdown, but transfer takes longer.")
    print()
    # Assume CPU uses bus 60% of cycles
    cpu_bus_usage = 0.60
    cycles_for_transparent = int(total_words / (1 - cpu_bus_usage))
    print(f"     CPU bus usage: {cpu_bus_usage:.0%}")
    print(f"     Estimated cycles to complete: {cycles_for_transparent}")
    print(f"     CPU impact: None (zero slowdown)")
    print()

    print("  Tradeoff: Burst is fastest for DMA but stalls CPU.")
    print("  Cycle-stealing balances CPU and DMA. Transparent has no CPU impact.")


def exercise_3():
    """
    Compare memory-mapped I/O vs port-mapped (isolated) I/O.
    """
    print("Memory-Mapped I/O vs Port-Mapped I/O:")
    print()

    print("  Memory-Mapped I/O:")
    print("    Device registers mapped into the CPU's address space.")
    print("    Access using normal LOAD/STORE instructions.")
    print()
    print("    Example (ARM, RISC-V):")
    print("      GPIO_BASE  = 0x4002_0000")
    print("      GPIO_DATA  = GPIO_BASE + 0x00  # data register")
    print("      GPIO_DIR   = GPIO_BASE + 0x04  # direction register")
    print()
    print("      # Turn on LED (Python-like pseudocode):")
    print("      mem[GPIO_DIR]  = 0x01  # set pin 0 as output")
    print("      mem[GPIO_DATA] = 0x01  # set pin 0 high")
    print()

    print("  Port-Mapped I/O (Isolated I/O):")
    print("    Separate I/O address space with special IN/OUT instructions.")
    print()
    print("    Example (x86):")
    print("      # Read keyboard scan code:")
    print("      status = IN(0x64)     # read status port")
    print("      data = IN(0x60)       # read data port")
    print("      OUT(0x20, 0x20)       # send EOI to interrupt controller")
    print()

    comparison = [
        ("Address space", "Shared with memory", "Separate I/O space"),
        ("Instructions", "Regular LOAD/STORE", "Special IN/OUT"),
        ("Protection", "Via page table (MMU)", "Via I/O privilege level"),
        ("Addressing", "Full address range", "Limited (64K ports on x86)"),
        ("Caching", "Must mark uncacheable", "Inherently uncacheable"),
        ("Used by", "ARM, RISC-V, MIPS", "x86 (legacy devices)"),
    ]

    print("  Comparison:")
    print(f"    {'Feature':<16s} {'Memory-Mapped':<25s} {'Port-Mapped':<25s}")
    print(f"    {'-'*16} {'-'*25} {'-'*25}")

    for feature, mmio, pmio in comparison:
        print(f"    {feature:<16s} {mmio:<25s} {pmio:<25s}")

    print()
    print("  Modern trend: Memory-mapped I/O is dominant (simpler, uniform).")
    print("  x86 still supports port I/O for legacy compatibility (keyboard, PIC).")


def exercise_4():
    """
    Calculate interrupt overhead and determine when DMA becomes worthwhile.
    """
    print("Interrupt Overhead vs DMA Crossover Point:")
    print()

    cpu_clock_ghz = 4.0
    isr_overhead_cycles = 500   # cycles to enter/exit ISR
    bytes_per_interrupt = 1     # 1 byte per interrupt (character device)
    dma_setup_cycles = 2000     # cycles to program DMA controller
    dma_completion_cycles = 500 # cycles for DMA completion interrupt

    print(f"  CPU: {cpu_clock_ghz} GHz")
    print(f"  ISR overhead: {isr_overhead_cycles} cycles per interrupt")
    print(f"  DMA setup: {dma_setup_cycles} cycles + {dma_completion_cycles} cycles completion")
    print()

    print("  When is DMA cheaper than interrupt-driven I/O?")
    print()

    transfer_sizes = [1, 4, 8, 16, 64, 256, 512, 1024, 4096]

    print(f"  {'Transfer Size':>14s} {'Interrupts':>11s} {'Int Cycles':>12s} "
          f"{'DMA Cycles':>11s} {'Winner':>8s}")
    print(f"  {'-'*14} {'-'*11} {'-'*12} {'-'*11} {'-'*8}")

    crossover = None
    for size in transfer_sizes:
        num_interrupts = size // bytes_per_interrupt
        int_cycles = num_interrupts * isr_overhead_cycles
        dma_cycles = dma_setup_cycles + dma_completion_cycles

        winner = "DMA" if dma_cycles < int_cycles else "INT"
        if winner == "DMA" and crossover is None:
            crossover = size

        print(
            f"  {size:>14d} {num_interrupts:>11d} {int_cycles:>12,d} "
            f"{dma_cycles:>11,d} {winner:>8s}"
        )

    print()
    if crossover:
        print(f"  Crossover point: DMA becomes cheaper at {crossover}+ bytes.")
    print("  For small transfers, interrupt overhead < DMA setup overhead.")
    print("  For large transfers, DMA is dramatically more efficient.")


def exercise_5():
    """
    Explain and simulate a simple interrupt priority scheme.
    """
    print("Interrupt Priority and Handling:")
    print()

    print("  Interrupt Priority Levels (higher number = higher priority):")
    print()

    devices = [
        {"name": "Keyboard",     "priority": 1, "isr_cycles": 200,  "freq_hz": 100},
        {"name": "Mouse",        "priority": 1, "isr_cycles": 150,  "freq_hz": 125},
        {"name": "Disk (SATA)",  "priority": 3, "isr_cycles": 500,  "freq_hz": 1000},
        {"name": "Network (NIC)","priority": 4, "isr_cycles": 800,  "freq_hz": 50000},
        {"name": "Timer",        "priority": 5, "isr_cycles": 100,  "freq_hz": 1000},
    ]

    cpu_clock = 4e9  # 4 GHz

    print(f"  {'Device':<16s} {'Priority':>9s} {'ISR Cycles':>11s} "
          f"{'Freq (Hz)':>10s} {'CPU Load':>9s}")
    print(f"  {'-'*16} {'-'*9} {'-'*11} {'-'*10} {'-'*9}")

    total_load = 0
    for dev in devices:
        cycles_per_sec = dev["isr_cycles"] * dev["freq_hz"]
        load = cycles_per_sec / cpu_clock
        total_load += load
        print(
            f"  {dev['name']:<16s} {dev['priority']:>9d} {dev['isr_cycles']:>11d} "
            f"{dev['freq_hz']:>10,d} {load:>8.3%}"
        )

    print(f"\n  Total interrupt CPU load: {total_load:.3%}")
    print()

    # Simulate nested interrupt handling
    print("  Nested Interrupt Example:")
    print("  Time 0: Keyboard ISR starts (priority 1)")
    print("  Time 50: Disk interrupt arrives (priority 3)")
    print("    -> Keyboard ISR preempted, Disk ISR starts")
    print("  Time 100: Timer interrupt arrives (priority 5)")
    print("    -> Disk ISR preempted, Timer ISR starts")
    print("  Time 200: Timer ISR completes")
    print("    -> Disk ISR resumes")
    print("  Time 550: Disk ISR completes")
    print("    -> Keyboard ISR resumes")
    print("  Time 700: Keyboard ISR completes")
    print()
    print("  Higher-priority interrupts can preempt lower-priority ones.")
    print("  This is called nested interrupt handling.")


def exercise_6():
    """
    Analyze I/O bandwidth requirements for common workloads.
    """
    print("I/O Bandwidth Analysis:")
    print()

    workloads = [
        {
            "name": "4K Video Playback",
            "resolution": "3840x2160",
            "fps": 60,
            "bpp": 24,  # bits per pixel
            "compressed_ratio": 200,  # H.265 compression ratio
        },
        {
            "name": "Database Server",
            "iops": 100_000,
            "io_size_bytes": 8192,
        },
        {
            "name": "Web Server",
            "requests_per_sec": 50_000,
            "avg_response_bytes": 10_000,
        },
        {
            "name": "ML Training (GPU)",
            "batch_size_mb": 256,
            "batches_per_sec": 10,
        },
    ]

    print(f"  {'Workload':<25s} {'Raw BW':>12s} {'Effective BW':>14s}")
    print(f"  {'-'*25} {'-'*12} {'-'*14}")

    for w in workloads:
        if w["name"] == "4K Video Playback":
            res = w["resolution"].split("x")
            pixels = int(res[0]) * int(res[1])
            raw_bps = pixels * w["fps"] * w["bpp"]
            eff_bps = raw_bps / w["compressed_ratio"]
        elif w["name"] == "Database Server":
            raw_bps = w["iops"] * w["io_size_bytes"] * 8
            eff_bps = raw_bps
        elif w["name"] == "Web Server":
            raw_bps = w["requests_per_sec"] * w["avg_response_bytes"] * 8
            eff_bps = raw_bps
        else:
            raw_bps = w["batch_size_mb"] * 1024 * 1024 * 8 * w["batches_per_sec"]
            eff_bps = raw_bps

        raw_gbps = raw_bps / 1e9
        eff_gbps = eff_bps / 1e9

        print(f"  {w['name']:<25s} {raw_gbps:>10.1f} Gb {eff_gbps:>12.1f} Gb")

    print()

    # Compare with I/O interface bandwidths
    print("  I/O Interface Bandwidths:")
    interfaces = [
        ("USB 2.0", 0.48),
        ("USB 3.2 Gen 2", 10),
        ("SATA III", 6),
        ("PCIe 4.0 x4 (NVMe)", 64),
        ("PCIe 5.0 x16 (GPU)", 512),
        ("Thunderbolt 4", 40),
        ("100GbE", 100),
    ]

    print(f"    {'Interface':<25s} {'Bandwidth (Gbps)':>17s}")
    print(f"    {'-'*25} {'-'*17}")
    for name, bw in interfaces:
        print(f"    {name:<25s} {bw:>17.1f}")


def exercise_7():
    """
    Explain PCIe lane structure and calculate bandwidth.
    """
    print("PCIe Bandwidth Calculation:")
    print()

    pcie_gens = [
        {"gen": "1.0", "rate_gt": 2.5,  "encoding": "8b/10b",   "overhead": 0.20},
        {"gen": "2.0", "rate_gt": 5.0,  "encoding": "8b/10b",   "overhead": 0.20},
        {"gen": "3.0", "rate_gt": 8.0,  "encoding": "128b/130b","overhead": 2/130},
        {"gen": "4.0", "rate_gt": 16.0, "encoding": "128b/130b","overhead": 2/130},
        {"gen": "5.0", "rate_gt": 32.0, "encoding": "128b/130b","overhead": 2/130},
        {"gen": "6.0", "rate_gt": 64.0, "encoding": "FLIT",     "overhead": 0.02},
    ]

    lane_configs = [1, 4, 8, 16]

    print(f"  {'Gen':>5s} {'GT/s':>6s} {'Encoding':>10s} {'x1 (GB/s)':>10s} "
          f"{'x4 (GB/s)':>10s} {'x8 (GB/s)':>10s} {'x16 (GB/s)':>11s}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")

    for gen in pcie_gens:
        # Effective rate per lane (each direction)
        effective_per_lane = gen["rate_gt"] * (1 - gen["overhead"]) / 8  # GB/s per lane

        bw_values = []
        for lanes in lane_configs:
            bw = effective_per_lane * lanes
            bw_values.append(bw)

        print(
            f"  {gen['gen']:>5s} {gen['rate_gt']:>6.1f} {gen['encoding']:>10s} "
            f"{bw_values[0]:>10.2f} {bw_values[1]:>10.1f} "
            f"{bw_values[2]:>10.1f} {bw_values[3]:>11.1f}"
        )

    print()
    print("  Note: These are unidirectional bandwidths. PCIe is full-duplex.")
    print("  A PCIe 4.0 x4 NVMe SSD: ~8 GB/s read (matches NVMe SSD limits).")
    print("  A PCIe 5.0 x16 GPU: ~64 GB/s to/from system memory.")
    print()
    print("  PCIe is point-to-point (unlike older shared buses like PCI).")
    print("  Each device gets dedicated lanes with guaranteed bandwidth.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: I/O Methods Comparison", exercise_1),
        ("Exercise 2: DMA Transfer Modes", exercise_2),
        ("Exercise 3: Memory-Mapped vs Port-Mapped I/O", exercise_3),
        ("Exercise 4: Interrupt vs DMA Crossover", exercise_4),
        ("Exercise 5: Interrupt Priority and Handling", exercise_5),
        ("Exercise 6: I/O Bandwidth Analysis", exercise_6),
        ("Exercise 7: PCIe Bandwidth Calculation", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'='*70}")
        print(f"=== {title} ===")
        print(f"{'='*70}")
        func()

    print(f"\n{'='*70}")
    print("All exercises completed!")

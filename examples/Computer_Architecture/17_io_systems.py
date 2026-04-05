"""
I/O Systems — Polling, Interrupt, and DMA Simulation

Demonstrates:
- Programmed I/O (polling)
- Interrupt-driven I/O
- Direct Memory Access (DMA)
- CPU utilization comparison across I/O techniques
- I/O bus bandwidth calculation

Theory:
- Programmed I/O (polling): CPU repeatedly checks device status
  register in a busy-wait loop.  Simple but wastes CPU cycles.
- Interrupt-driven I/O: device signals the CPU when data is ready.
  CPU can do useful work between interrupts, but each byte still
  requires CPU intervention.
- DMA: a dedicated controller transfers blocks between device and
  memory without CPU involvement.  CPU only sets up the transfer
  and is interrupted on completion.  Best for bulk transfers.

Adapted from Computer Architecture Lesson 17.
"""

from dataclasses import dataclass, field
import random


# ── Device Model ──────────────────────────────────────────────────────

@dataclass
class IODevice:
    """Simulated I/O device with variable latency."""
    name: str
    latency_cycles: int    # cycles per byte of transfer
    transfer_size: int     # total bytes to transfer
    ready_interval: int    # cycles between data-ready events (for polling)

    def __repr__(self) -> str:
        return f"{self.name} ({self.transfer_size}B, {self.latency_cycles}cyc/B)"


# ── Polling Simulation ───────────────────────────────────────────────

def simulate_polling(device: IODevice) -> dict:
    """Simulate programmed I/O with busy-wait polling.

    The CPU spins in a loop checking the device status register.
    Every ready_interval cycles, one byte becomes available and
    the CPU reads it.  All poll cycles are wasted.
    """
    total_cycles = 0
    cpu_busy_cycles = 0
    bytes_transferred = 0

    while bytes_transferred < device.transfer_size:
        # Poll loop: check status register every cycle
        for _ in range(device.ready_interval):
            total_cycles += 1
            cpu_busy_cycles += 1  # CPU is stuck polling

        # Data is ready — transfer one byte
        total_cycles += device.latency_cycles
        cpu_busy_cycles += device.latency_cycles
        bytes_transferred += 1

    return {
        "method": "Polling",
        "total_cycles": total_cycles,
        "cpu_busy_cycles": cpu_busy_cycles,
        "cpu_utilization": cpu_busy_cycles / total_cycles,
        "cpu_useful_work": 0,  # no useful work during polling
        "bytes": device.transfer_size,
    }


# ── Interrupt-Driven Simulation ──────────────────────────────────────

@dataclass
class InterruptController:
    """Simple interrupt controller."""
    pending: list[str] = field(default_factory=list)
    handler_overhead: int = 20  # cycles to save/restore context

    def raise_irq(self, source: str) -> None:
        self.pending.append(source)

    def has_pending(self) -> bool:
        return len(self.pending) > 0

    def acknowledge(self) -> str:
        return self.pending.pop(0) if self.pending else ""


def simulate_interrupt(device: IODevice,
                       handler_overhead: int = 20) -> dict:
    """Simulate interrupt-driven I/O.

    Between interrupts the CPU executes useful instructions.  When
    the device raises an interrupt, the CPU saves context, runs the
    ISR (reads one byte), and restores context.
    """
    total_cycles = 0
    cpu_busy_cycles = 0
    cpu_useful_cycles = 0
    bytes_transferred = 0
    interrupts = 0

    ic = InterruptController(handler_overhead=handler_overhead)

    while bytes_transferred < device.transfer_size:
        # CPU does useful work until next interrupt
        useful = device.ready_interval - handler_overhead
        if useful < 0:
            useful = 0
        cpu_useful_cycles += useful
        total_cycles += device.ready_interval

        # Device raises interrupt
        ic.raise_irq(device.name)
        interrupts += 1

        # Handle interrupt: context save + transfer + context restore
        isr_cycles = handler_overhead + device.latency_cycles
        total_cycles += isr_cycles
        cpu_busy_cycles += isr_cycles
        bytes_transferred += 1
        ic.acknowledge()

    cpu_busy_cycles += cpu_useful_cycles  # useful work is also CPU busy

    return {
        "method": "Interrupt",
        "total_cycles": total_cycles,
        "cpu_busy_cycles": cpu_busy_cycles,
        "cpu_utilization": cpu_busy_cycles / total_cycles,
        "cpu_useful_work": cpu_useful_cycles / total_cycles,
        "bytes": device.transfer_size,
        "interrupts": interrupts,
    }


# ── DMA Simulation ───────────────────────────────────────────────────

def simulate_dma(device: IODevice, block_size: int = 64,
                 dma_setup_cycles: int = 100) -> dict:
    """Simulate DMA transfer.

    CPU programs the DMA controller (source, destination, count),
    then is free to do useful work.  The DMA controller transfers
    data block-by-block.  When the entire transfer completes, the
    DMA controller interrupts the CPU.
    """
    total_cycles = 0
    cpu_busy_cycles = 0
    cpu_useful_cycles = 0
    dma_interrupts = 0

    # Number of DMA blocks
    n_blocks = (device.transfer_size + block_size - 1) // block_size

    for block_idx in range(n_blocks):
        # CPU sets up DMA transfer for this block
        total_cycles += dma_setup_cycles
        cpu_busy_cycles += dma_setup_cycles

        # DMA transfers block while CPU does useful work
        bytes_in_block = min(block_size,
                             device.transfer_size - block_idx * block_size)
        transfer_cycles = bytes_in_block * device.latency_cycles
        # CPU is free during transfer (minus occasional bus contention)
        bus_steal_cycles = transfer_cycles // 10  # ~10% bus contention
        useful = transfer_cycles - bus_steal_cycles
        cpu_useful_cycles += useful
        cpu_busy_cycles += useful
        total_cycles += transfer_cycles

        # DMA completion interrupt
        handler_cycles = 20
        total_cycles += handler_cycles
        cpu_busy_cycles += handler_cycles
        dma_interrupts += 1

    return {
        "method": "DMA",
        "total_cycles": total_cycles,
        "cpu_busy_cycles": cpu_busy_cycles,
        "cpu_utilization": cpu_busy_cycles / total_cycles if total_cycles else 0,
        "cpu_useful_work": cpu_useful_cycles / total_cycles if total_cycles else 0,
        "bytes": device.transfer_size,
        "dma_interrupts": dma_interrupts,
        "blocks": n_blocks,
    }


# ── Demos ─────────────────────────────────────────────────────────────

def demo_comparison():
    """Compare all three I/O techniques."""
    print("=" * 60)
    print("I/O TECHNIQUE COMPARISON")
    print("=" * 60)

    device = IODevice(
        name="Disk Controller",
        latency_cycles=2,
        transfer_size=256,
        ready_interval=50,
    )
    print(f"\n  Device: {device}")

    results = [
        simulate_polling(device),
        simulate_interrupt(device),
        simulate_dma(device),
    ]

    print(f"\n  {'Method':<12} {'Total Cyc':>12} {'CPU Useful':>12} "
          f"{'CPU Util':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for r in results:
        print(f"  {r['method']:<12} {r['total_cycles']:>12,} "
              f"{r['cpu_useful_work']:>11.0%} {r['cpu_utilization']:>9.0%}")

    print("""
  Key observations:
  - Polling: 0% useful CPU work (all cycles wasted on status checks)
  - Interrupt: CPU does useful work between interrupts
  - DMA: CPU free during bulk transfer (highest useful work %)
""")


def demo_scaling():
    """Show how techniques scale with transfer size."""
    print("=" * 60)
    print("SCALING WITH TRANSFER SIZE")
    print("=" * 60)

    sizes = [64, 256, 1024, 4096, 16384]

    print(f"\n  {'Size':>8}  {'Polling':>12}  {'Interrupt':>12}  "
          f"{'DMA':>12}  {'DMA Useful%':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for size in sizes:
        device = IODevice("Dev", latency_cycles=2,
                          transfer_size=size, ready_interval=50)
        p = simulate_polling(device)
        i = simulate_interrupt(device)
        d = simulate_dma(device)

        print(f"  {size:>7}B  {p['total_cycles']:>12,}  "
              f"{i['total_cycles']:>12,}  {d['total_cycles']:>12,}  "
              f"{d['cpu_useful_work']:>11.0%}")

    print(f"\n  DMA advantage grows with transfer size — setup cost")
    print(f"  is amortized over more bytes.")


def demo_bus_bandwidth():
    """Calculate I/O bus bandwidth requirements."""
    print("\n" + "=" * 60)
    print("I/O BUS BANDWIDTH")
    print("=" * 60)

    @dataclass
    class BusDevice:
        name: str
        data_rate_mbps: float  # MB/s
        burst_size_bytes: int

    devices = [
        BusDevice("Keyboard",       0.001,    1),
        BusDevice("Mouse",          0.01,     4),
        BusDevice("Ethernet 1G",    125.0,    1500),
        BusDevice("SSD (NVMe)",     3500.0,   4096),
        BusDevice("GPU (PCIe 4)",   25000.0,  256),
    ]

    print(f"\n  {'Device':<18} {'Data Rate':>12} {'Burst':>8} "
          f"{'Recommended I/O':>18}")
    print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*18}")

    for dev in devices:
        if dev.data_rate_mbps < 1:
            rec = "Polling / IRQ"
        elif dev.data_rate_mbps < 500:
            rec = "Interrupt"
        else:
            rec = "DMA"
        rate = (f"{dev.data_rate_mbps:.3f} MB/s" if dev.data_rate_mbps < 1
                else f"{dev.data_rate_mbps:,.0f} MB/s")
        print(f"  {dev.name:<18} {rate:>12} {dev.burst_size_bytes:>6}B "
              f"{rec:>18}")

    print("""
  Guidelines:
  - Low-rate devices (keyboard, mouse): polling or interrupt is fine
  - Medium-rate (network): interrupt-driven with coalescing
  - High-rate (SSD, GPU): DMA is essential to avoid CPU saturation
""")


def demo_interrupt_coalescing():
    """Show interrupt coalescing to reduce overhead."""
    print("=" * 60)
    print("INTERRUPT COALESCING")
    print("=" * 60)

    device = IODevice("NIC", latency_cycles=1,
                      transfer_size=1000, ready_interval=10)

    # Without coalescing: 1 interrupt per byte
    no_coal = simulate_interrupt(device, handler_overhead=20)

    # With coalescing: 1 interrupt per N bytes (simulated by
    # increasing transfer size per interrupt)
    coal_sizes = [1, 4, 16, 64]
    print(f"\n  {'Coalesce':>10} {'Interrupts':>12} {'Total Cyc':>12} "
          f"{'Useful%':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for coal in coal_sizes:
        # Simulate by reducing interrupt count
        total = device.transfer_size
        n_irqs = total // coal
        handler = 20  # per interrupt
        transfer_per_irq = coal * device.latency_cycles
        irq_overhead = n_irqs * handler
        transfer_time = total * device.latency_cycles
        cpu_free = total * device.ready_interval - irq_overhead
        if cpu_free < 0:
            cpu_free = 0
        total_cyc = total * device.ready_interval + irq_overhead
        useful = cpu_free / total_cyc if total_cyc else 0

        print(f"  {coal:>8}B  {n_irqs:>12,}  {total_cyc:>12,}  "
              f"{useful:>9.0%}")

    print(f"\n  Coalescing reduces interrupt overhead at the cost of")
    print(f"  increased latency for individual bytes.")


if __name__ == "__main__":
    demo_comparison()
    demo_scaling()
    demo_bus_bandwidth()
    demo_interrupt_coalescing()

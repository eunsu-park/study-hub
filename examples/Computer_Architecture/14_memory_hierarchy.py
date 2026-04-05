"""
Memory Hierarchy Simulation

Demonstrates:
- Memory hierarchy levels (registers, L1, L2, L3, DRAM, disk)
- Access time comparison and cost-capacity trade-off
- Spatial and temporal locality patterns
- Working set analysis
- AMAT calculation for multi-level hierarchies

Theory:
- Memory hierarchy exploits the locality principle: programs tend
  to access a small portion of address space at any given time.
- Temporal locality: recently accessed data will be accessed again.
- Spatial locality: nearby data will be accessed soon.
- Each level trades capacity for speed: registers are fastest but
  smallest; disk is largest but slowest.
- AMAT = Hit Time + Miss Rate × Miss Penalty, applied recursively
  for multi-level caches.

Adapted from Computer Architecture Lesson 14.
"""

from dataclasses import dataclass


# ── Memory Level Specifications ───────────────────────────────────────

@dataclass
class MemoryLevel:
    """Specification of one memory hierarchy level."""
    name: str
    size_bytes: int
    access_time_ns: float
    cost_per_gb: float  # USD per GB

    @property
    def size_str(self) -> str:
        if self.size_bytes >= 1 << 40:
            return f"{self.size_bytes / (1 << 40):.0f} TB"
        if self.size_bytes >= 1 << 30:
            return f"{self.size_bytes / (1 << 30):.0f} GB"
        if self.size_bytes >= 1 << 20:
            return f"{self.size_bytes / (1 << 20):.0f} MB"
        if self.size_bytes >= 1 << 10:
            return f"{self.size_bytes / (1 << 10):.0f} KB"
        return f"{self.size_bytes} B"


# Typical modern processor hierarchy (approximate 2024 values)
HIERARCHY = [
    MemoryLevel("Registers",   256,        0.25,    0),
    MemoryLevel("L1 Cache",    64 * 1024,  1.0,     7_000_000),
    MemoryLevel("L2 Cache",    512 * 1024, 4.0,     1_000_000),
    MemoryLevel("L3 Cache",    32 * (1 << 20), 12.0, 200_000),
    MemoryLevel("DRAM",        32 * (1 << 30), 80.0, 5),
    MemoryLevel("SSD",         1 * (1 << 40), 50_000, 0.08),
    MemoryLevel("HDD",         4 * (1 << 40), 5_000_000, 0.02),
]


# ── Access Pattern Simulator ─────────────────────────────────────────

class HierarchySimulator:
    """Simulate memory accesses through a multi-level hierarchy.

    Each level has a fixed size.  An access hits if the address
    falls within the working set that fits in that level (simplified
    model based on locality).
    """

    def __init__(self, levels: list[MemoryLevel]):
        self.levels = levels
        self.access_count = [0] * len(levels)
        self.hit_count = [0] * len(levels)
        self.total_accesses = 0

    def access(self, addr: int, working_set_size: int) -> tuple[str, float]:
        """Simulate access, returning (level_name, latency_ns).

        The working_set_size determines which cache level can hold
        the active data.  If the working set fits in L1, most accesses
        hit there; if it exceeds L1 but fits L2, L2 serves the miss.
        """
        self.total_accesses += 1
        total_latency = 0.0

        for i, level in enumerate(self.levels):
            self.access_count[i] += 1
            total_latency += level.access_time_ns

            # Simplified hit model: if address < level capacity and
            # working set fits in this level, it's a hit
            if working_set_size <= level.size_bytes:
                self.hit_count[i] += 1
                return level.name, total_latency

        # Fallback to last level
        self.hit_count[-1] += 1
        return self.levels[-1].name, total_latency

    def reset(self) -> None:
        self.access_count = [0] * len(self.levels)
        self.hit_count = [0] * len(self.levels)
        self.total_accesses = 0


# ── AMAT Calculator ──────────────────────────────────────────────────

def compute_amat(hit_times: list[float],
                 miss_rates: list[float]) -> tuple[float, list[str]]:
    """Compute Average Memory Access Time for multi-level hierarchy.

    AMAT = HitTime_L1 + MissRate_L1 × (HitTime_L2 + MissRate_L2 × (...))

    Returns (amat, step-by-step explanation).
    """
    steps = []
    n = len(hit_times)

    # Build from bottom up
    penalty = 0.0
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            penalty = hit_times[i]
            steps.append(
                f"  Level {i} (base): {hit_times[i]:.1f} ns")
        else:
            amat_this = hit_times[i] + miss_rates[i] * penalty
            steps.append(
                f"  Level {i}: {hit_times[i]:.1f} + "
                f"{miss_rates[i]:.2%} × {penalty:.1f} = {amat_this:.2f} ns")
            penalty = amat_this

    steps.reverse()
    return penalty, steps


# ── Demos ─────────────────────────────────────────────────────────────

def demo_hierarchy_overview():
    """Display memory hierarchy specifications."""
    print("=" * 60)
    print("MEMORY HIERARCHY OVERVIEW")
    print("=" * 60)

    print(f"\n  {'Level':<12} {'Size':>10} {'Access Time':>14} "
          f"{'$/GB':>12} {'Speedup vs DRAM':>16}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*12} {'-'*16}")

    dram_time = next(l.access_time_ns for l in HIERARCHY if l.name == "DRAM")
    for level in HIERARCHY:
        speedup = dram_time / level.access_time_ns
        cost = f"${level.cost_per_gb:,.0f}" if level.cost_per_gb else "N/A"
        if level.access_time_ns >= 1000:
            time_str = f"{level.access_time_ns / 1000:,.0f} μs"
        else:
            time_str = f"{level.access_time_ns:.1f} ns"
        print(f"  {level.name:<12} {level.size_str:>10} {time_str:>14} "
              f"{cost:>12} {speedup:>15.0f}x")

    print(f"\n  Speed ratio (registers vs HDD): "
          f"{HIERARCHY[-1].access_time_ns / HIERARCHY[0].access_time_ns:,.0f}x")
    print(f"  Capacity ratio (HDD vs registers): "
          f"{HIERARCHY[-1].size_bytes / HIERARCHY[0].size_bytes:,.0f}x")


def demo_locality_impact():
    """Show how locality affects which hierarchy level serves data."""
    print("\n" + "=" * 60)
    print("LOCALITY AND WORKING SET SIZE")
    print("=" * 60)

    sim = HierarchySimulator(HIERARCHY[:5])  # Registers through DRAM

    working_sets = [
        (128,       "Tight loop (few variables)"),
        (32 * 1024, "Array scan (fits L1)"),
        (256 * 1024, "Medium dataset (fits L2)"),
        (16 * (1 << 20), "Large dataset (fits L3)"),
        (1 * (1 << 30), "Very large dataset (DRAM)"),
    ]

    print(f"\n  {'Working Set':<15} {'Served By':<14} {'Latency':>10}  "
          f"{'Description'}")
    print(f"  {'-'*15} {'-'*14} {'-'*10}  {'-'*30}")

    for ws_size, desc in working_sets:
        sim.reset()
        level_name, latency = sim.access(0, ws_size)
        ws_str = MemoryLevel("", ws_size, 0, 0).size_str
        print(f"  {ws_str:<15} {level_name:<14} {latency:>8.1f} ns  {desc}")


def demo_amat_calculation():
    """Multi-level AMAT calculation."""
    print("\n" + "=" * 60)
    print("AMAT CALCULATION (3-LEVEL CACHE)")
    print("=" * 60)

    # L1, L2, L3, DRAM
    hit_times = [1.0, 4.0, 12.0, 80.0]
    miss_rates = [0.05, 0.10, 0.20]  # local miss rates

    print(f"\n  Configuration:")
    labels = ["L1", "L2", "L3", "DRAM"]
    for i, label in enumerate(labels):
        mr_str = f", miss rate = {miss_rates[i]:.0%}" if i < len(miss_rates) else ""
        print(f"    {label}: hit time = {hit_times[i]:.0f} ns{mr_str}")

    amat, steps = compute_amat(hit_times, miss_rates)
    print(f"\n  AMAT computation (inside out):")
    for step in steps:
        print(f"  {step}")
    print(f"\n  Effective AMAT = {amat:.2f} ns")
    print(f"  Speedup vs always-DRAM: {80.0 / amat:.1f}x")

    # Show sensitivity to L1 miss rate
    print(f"\n  Sensitivity to L1 miss rate:")
    print(f"  {'L1 Miss Rate':>13}  {'AMAT':>8}  {'Speedup':>8}")
    print(f"  {'-'*13}  {'-'*8}  {'-'*8}")
    for mr in [0.01, 0.02, 0.05, 0.10, 0.20]:
        rates = [mr, 0.10, 0.20]
        a, _ = compute_amat(hit_times, rates)
        print(f"  {mr:>12.0%}  {a:>7.2f}  {80.0 / a:>7.1f}x")


def demo_locality_patterns():
    """Compare access patterns and their locality characteristics."""
    print("\n" + "=" * 60)
    print("ACCESS PATTERN LOCALITY ANALYSIS")
    print("=" * 60)

    array_size = 1024  # elements

    # Pattern 1: Sequential scan (excellent spatial locality)
    sequential = list(range(array_size))

    # Pattern 2: Stride-N (decreasing spatial locality)
    stride_4 = list(range(0, array_size * 4, 4))
    stride_16 = list(range(0, array_size * 16, 16))

    # Pattern 3: Repeated small working set (temporal locality)
    small_ws = list(range(16)) * (array_size // 16)

    # Pattern 4: Random access (poor locality)
    import random
    random.seed(42)
    random_access = [random.randint(0, array_size - 1)
                     for _ in range(array_size)]

    patterns = [
        ("Sequential",     sequential),
        ("Stride-4",       stride_4),
        ("Stride-16",      stride_16),
        ("Small WS (16)",  small_ws),
        ("Random",         random_access),
    ]

    print(f"\n  {'Pattern':<18} {'Unique Addrs':>13} {'Temporal':>10} "
          f"{'Spatial':>10}")
    print(f"  {'-'*18} {'-'*13} {'-'*10} {'-'*10}")

    for name, addrs in patterns:
        unique = len(set(addrs))
        # Temporal locality: ratio of reuses to total accesses
        temporal = 1.0 - (unique / len(addrs))
        # Spatial locality: fraction of consecutive accesses within
        # a cache line (64 bytes ≈ 16 int elements)
        spatial_hits = sum(
            1 for i in range(1, len(addrs))
            if abs(addrs[i] - addrs[i-1]) <= 16
        )
        spatial = spatial_hits / (len(addrs) - 1)

        print(f"  {name:<18} {unique:>13} {temporal:>9.0%} {spatial:>9.0%}")

    print("""
  Key insights:
  - Sequential access maximizes spatial locality (prefetcher friendly)
  - Small working sets maximize temporal locality (cache friendly)
  - Random access has neither — worst case for cache performance
  - Stride patterns lose spatial locality as stride grows
""")


if __name__ == "__main__":
    demo_hierarchy_overview()
    demo_locality_impact()
    demo_amat_calculation()
    demo_locality_patterns()

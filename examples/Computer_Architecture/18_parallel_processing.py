"""
Parallel Processing and Multicore

Demonstrates:
- Amdahl's law and Gustafson's law calculators
- SIMD vector operation simulation
- Multithreading models (coarse-grain, fine-grain, SMT)
- Parallel speedup analysis
- Cache coherence (MESI protocol states)

Theory:
- Amdahl's law: speedup is limited by the serial fraction of
  a program.  S = 1 / ((1 - P) + P/N) where P is the parallel
  fraction and N is the number of processors.
- Gustafson's law: scaled speedup assumes problem size grows with
  processors.  S = N - α(N - 1) where α is the serial fraction.
- SIMD: Single Instruction, Multiple Data — one instruction operates
  on a vector of data elements simultaneously (e.g., SSE, AVX).
- Multithreading: multiple threads share a single core to hide
  latency (memory stalls, etc.).
- Cache coherence: protocols (e.g., MESI) ensure all cores see a
  consistent view of shared memory.

Adapted from Computer Architecture Lesson 18.
"""

from dataclasses import dataclass


# ── Amdahl's Law ──────────────────────────────────────────────────────

def amdahl_speedup(parallel_fraction: float, n_processors: int) -> float:
    """Compute speedup under Amdahl's law.

    S(N) = 1 / ((1 - P) + P / N)

    The serial portion (1-P) forms an absolute lower bound on
    execution time regardless of processor count.
    """
    serial = 1.0 - parallel_fraction
    return 1.0 / (serial + parallel_fraction / n_processors)


def amdahl_max_speedup(parallel_fraction: float) -> float:
    """Maximum speedup (N → ∞) = 1 / (1 - P)."""
    return 1.0 / (1.0 - parallel_fraction)


def gustafson_speedup(serial_fraction: float, n_processors: int) -> float:
    """Compute scaled speedup under Gustafson's law.

    S(N) = N - α(N - 1)

    Assumes the problem size scales with N so that the parallel
    portion grows proportionally.
    """
    return n_processors - serial_fraction * (n_processors - 1)


# ── SIMD Simulation ──────────────────────────────────────────────────

@dataclass
class SIMDRegister:
    """Simulated SIMD register holding multiple data elements."""
    width: int       # number of elements (e.g., 4 for 128-bit/32-bit)
    data: list[float]

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.1f}" for v in self.data)
        return f"[{vals}]"


def simd_add(a: SIMDRegister, b: SIMDRegister) -> SIMDRegister:
    """SIMD vector addition: element-wise a + b in one instruction."""
    assert a.width == b.width
    return SIMDRegister(
        width=a.width,
        data=[x + y for x, y in zip(a.data, b.data)]
    )


def simd_mul(a: SIMDRegister, b: SIMDRegister) -> SIMDRegister:
    """SIMD vector multiplication: element-wise a * b."""
    assert a.width == b.width
    return SIMDRegister(
        width=a.width,
        data=[x * y for x, y in zip(a.data, b.data)]
    )


def scalar_add(a: list[float], b: list[float]) -> list[float]:
    """Scalar addition: one element per instruction."""
    return [x + y for x, y in zip(a, b)]


# ── MESI Cache Coherence ─────────────────────────────────────────────

MESI_STATES = {
    "M": "Modified  — dirty, exclusive, must write back before sharing",
    "E": "Exclusive — clean, only copy, can write without bus traffic",
    "S": "Shared    — clean, other caches may also hold this line",
    "I": "Invalid   — not valid, must fetch on access",
}

MESI_TRANSITIONS = [
    # (current_state, event) → new_state
    ("I", "PrRd (miss, no other)",   "E"),
    ("I", "PrRd (miss, shared)",     "S"),
    ("I", "PrWr (miss)",            "M"),
    ("E", "PrRd",                    "E"),
    ("E", "PrWr",                    "M"),
    ("S", "PrRd",                    "S"),
    ("S", "PrWr (invalidate others)","M"),
    ("M", "PrRd",                    "M"),
    ("M", "PrWr",                    "M"),
    ("M", "BusRd (snoop)",          "S"),
    ("E", "BusRd (snoop)",          "S"),
    ("S", "BusRdX (snoop)",         "I"),
    ("M", "BusRdX (snoop)",         "I"),
]


# ── Demos ─────────────────────────────────────────────────────────────

def demo_amdahl():
    """Amdahl's law analysis."""
    print("=" * 60)
    print("AMDAHL'S LAW")
    print("=" * 60)

    parallel_fractions = [0.50, 0.75, 0.90, 0.95, 0.99]
    processors = [1, 2, 4, 8, 16, 64, 256]

    print(f"\n  Speedup table (rows = parallel fraction, cols = processors):")
    header = f"  {'P':>5}" + "".join(f" {n:>6}" for n in processors) + f" {'Max':>7}"
    print(header)
    print(f"  {'-'*5}" + f" {'-'*6}" * len(processors) + f" {'-'*7}")

    for p in parallel_fractions:
        row = f"  {p:>4.0%}"
        for n in processors:
            s = amdahl_speedup(p, n)
            row += f" {s:>6.1f}"
        row += f" {amdahl_max_speedup(p):>6.1f}x"
        print(row)

    print(f"\n  Key insight: even with 99% parallelizable code,")
    print(f"  max speedup is only {amdahl_max_speedup(0.99):.0f}x "
          f"(the 1% serial portion dominates)")


def demo_gustafson():
    """Gustafson's law — scaled speedup."""
    print("\n" + "=" * 60)
    print("GUSTAFSON'S LAW (SCALED SPEEDUP)")
    print("=" * 60)

    serial_fractions = [0.01, 0.05, 0.10, 0.20]
    processors = [1, 4, 16, 64, 256]

    print(f"\n  {'Serial α':>9}", end="")
    for n in processors:
        print(f" {n:>6}P", end="")
    print()
    print(f"  {'-'*9}" + f" {'-'*7}" * len(processors))

    for alpha in serial_fractions:
        print(f"  {alpha:>8.0%}", end="")
        for n in processors:
            s = gustafson_speedup(alpha, n)
            print(f" {s:>6.1f}x", end="")
        print()

    print(f"\n  Gustafson assumes problem size grows with N.")
    print(f"  More optimistic than Amdahl for scalable workloads.")


def demo_simd():
    """SIMD vector operations vs scalar."""
    print("\n" + "=" * 60)
    print("SIMD VECTOR OPERATIONS")
    print("=" * 60)

    # 128-bit SIMD register = 4 × 32-bit floats
    width = 4
    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [5.0, 6.0, 7.0, 8.0]

    a = SIMDRegister(width=width, data=a_data)
    b = SIMDRegister(width=width, data=b_data)

    print(f"\n  SIMD width: {width} elements (128-bit / 32-bit floats)")
    print(f"\n  A = {a}")
    print(f"  B = {b}")

    result_add = simd_add(a, b)
    result_mul = simd_mul(a, b)

    print(f"\n  SIMD ADD: A + B = {result_add}  (1 instruction)")
    print(f"  SIMD MUL: A * B = {result_mul}  (1 instruction)")

    print(f"\n  Scalar equivalent: {width} ADD + {width} MUL = "
          f"{width * 2} instructions")
    print(f"  SIMD equivalent:   1 ADD + 1 MUL = 2 instructions")
    print(f"  Speedup: {width}x for data-parallel operations")

    # Larger example
    sizes = [4, 8, 16, 32]
    print(f"\n  SIMD width scaling:")
    print(f"  {'Width':>7} {'Example':>12} {'Scalar Ops':>12} {'SIMD Ops':>10} "
          f"{'Speedup':>8}")
    print(f"  {'-'*7} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    n_elements = 1024
    for w in sizes:
        bits = w * 32
        scalar = n_elements
        simd = n_elements // w
        print(f"  {w:>7} {bits:>8}-bit {scalar:>12} {simd:>10} {w:>7}x")


def demo_threading_models():
    """Compare multithreading models."""
    print("\n" + "=" * 60)
    print("MULTITHREADING MODELS")
    print("=" * 60)

    @dataclass
    class ThreadModel:
        name: str
        description: str
        switch_cost: str
        latency_hiding: str
        hw_complexity: str

    models = [
        ThreadModel(
            "Coarse-Grain MT",
            "Switch threads on long-latency events (cache miss)",
            "Moderate (pipeline flush)",
            "Partial (only on stalls)",
            "Low"
        ),
        ThreadModel(
            "Fine-Grain MT",
            "Switch threads every cycle (round-robin)",
            "Zero (interleaved pipeline)",
            "Good (hides single-cycle gaps)",
            "Moderate"
        ),
        ThreadModel(
            "SMT (Hyper-threading)",
            "Multiple threads issue simultaneously each cycle",
            "None (truly concurrent)",
            "Best (fills all issue slots)",
            "High"
        ),
    ]

    for model in models:
        print(f"\n  {model.name}")
        print(f"    Description:     {model.description}")
        print(f"    Switch cost:     {model.switch_cost}")
        print(f"    Latency hiding:  {model.latency_hiding}")
        print(f"    HW complexity:   {model.hw_complexity}")

    # Throughput comparison simulation
    print(f"\n  Throughput comparison (relative to single-thread baseline):")
    print(f"  {'Model':<22} {'1 Thread':>10} {'2 Threads':>10} "
          f"{'4 Threads':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    # Simplified IPC model with memory stalls
    base_ipc = 1.0
    stall_fraction = 0.3  # 30% of cycles are memory stalls
    for name, t1, t2, t4 in [
        ("No MT",            1.0, 1.0, 1.0),
        ("Coarse-Grain",     1.0, 1.2, 1.3),
        ("Fine-Grain",       0.9, 1.5, 1.8),   # slight single-thread loss
        ("SMT",              1.0, 1.6, 2.2),
    ]:
        print(f"  {name:<22} {t1 * base_ipc:>9.1f}x "
              f"{t2 * base_ipc:>9.1f}x {t4 * base_ipc:>9.1f}x")


def demo_mesi():
    """MESI cache coherence protocol states and transitions."""
    print("\n" + "=" * 60)
    print("MESI CACHE COHERENCE PROTOCOL")
    print("=" * 60)

    print("\n  States:")
    for state, desc in MESI_STATES.items():
        print(f"    {state}: {desc}")

    print(f"\n  Transitions:")
    print(f"  {'From':>4}  {'Event':<32} {'To':>4}")
    print(f"  {'-'*4}  {'-'*32} {'-'*4}")
    for cur, event, nxt in MESI_TRANSITIONS:
        print(f"  {cur:>4}  {event:<32} {nxt:>4}")

    # Scenario walkthrough
    print(f"\n  Scenario: Core 0 reads X, Core 1 reads X, Core 0 writes X")
    scenario = [
        ("Core 0 reads X",  {"C0": "E", "C1": "I"}, "C0 gets exclusive (no sharers)"),
        ("Core 1 reads X",  {"C0": "S", "C1": "S"}, "C0 snoops → E→S, C1 gets S"),
        ("Core 0 writes X", {"C0": "M", "C1": "I"}, "C0 invalidates C1 → S→I, C0 → M"),
        ("Core 1 reads X",  {"C0": "S", "C1": "S"}, "C0 writes back → M→S, C1 gets S"),
    ]
    print(f"\n  {'Step':<4} {'Action':<22} {'C0':>4} {'C1':>4}  {'Explanation'}")
    print(f"  {'-'*4} {'-'*22} {'-'*4} {'-'*4}  {'-'*36}")
    for i, (action, states, explanation) in enumerate(scenario):
        print(f"  {i+1:<4} {action:<22} {states['C0']:>4} "
              f"{states['C1']:>4}  {explanation}")


if __name__ == "__main__":
    demo_amdahl()
    demo_gustafson()
    demo_simd()
    demo_threading_models()
    demo_mesi()

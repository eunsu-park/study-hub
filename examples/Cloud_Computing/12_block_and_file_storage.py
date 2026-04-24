"""
Block vs. File Storage — Performance and Cost Simulation

Simulates the trade-offs between the three storage categories every cloud
offers:

- Block     — raw disk, single-attach (EBS, Persistent Disk, Azure Disk)
- File      — shared filesystem, multi-attach (EFS, Filestore, Azure Files)
- Object    — HTTP-accessible blobs (covered in example 05)

The simulation models latency, throughput, cost, and the "can N pods attach"
question that drives the block-vs-file decision for most workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# =============================================================================
# Storage option definitions (approximate, teaching values)
# =============================================================================

@dataclass
class Storage:
    name: str
    kind: str                    # "block" | "file"
    gb_month_usd: float
    iops_included: int           # IOPS baseline
    throughput_mbps: int
    read_latency_us: int         # microseconds, small-block random read
    multi_attach: bool           # can more than one VM mount at the same time?
    scale_ceiling_tb: int        # max practical single-volume size

    def monthly_cost(self, size_gb: int) -> float:
        return size_gb * self.gb_month_usd


# =============================================================================
# Workload and fit-scoring
# =============================================================================

@dataclass
class Workload:
    name: str
    size_gb: int
    concurrent_readers: int      # how many VMs need to see the data
    rw_pattern: str              # "random" | "sequential" | "append"
    ops_per_sec: int


def fit_score(storage: Storage, workload: Workload) -> int:
    """Return an integer 0..10 estimating how well storage matches workload.

    Rationale embedded in scoring so the student can read the deductions:
     - multi-reader workloads on non-multi-attach block storage are strongly
       penalized (they would require app-level sharding or a sync layer)
     - tiny volumes on shared filesystems waste the per-GB premium
     - IOPS shortage caps score regardless of other factors
    """
    score = 10

    # Multi-attach requirement
    if workload.concurrent_readers > 1 and not storage.multi_attach:
        score -= 6

    # Tiny data on expensive shared storage
    if workload.size_gb < 50 and storage.kind == "file":
        score -= 1

    # IOPS budget
    if workload.ops_per_sec > storage.iops_included:
        score -= 4

    # Over-sized requirement
    if workload.size_gb > storage.scale_ceiling_tb * 1024:
        score -= 5

    # Random read patterns prefer lower latency (block storage on local NVMe)
    if workload.rw_pattern == "random" and storage.read_latency_us > 1000:
        score -= 2

    return max(0, score)


def build_options() -> List[Storage]:
    return [
        Storage(
            name="EBS gp3 (block)", kind="block",
            gb_month_usd=0.080,
            iops_included=3000, throughput_mbps=125,
            read_latency_us=500,
            multi_attach=False,
            scale_ceiling_tb=16,
        ),
        Storage(
            name="EBS io2 (block, high-IOPS)", kind="block",
            gb_month_usd=0.125,
            iops_included=64000, throughput_mbps=1000,
            read_latency_us=300,
            multi_attach=False,
            scale_ceiling_tb=64,
        ),
        Storage(
            name="EFS (file, shared)", kind="file",
            gb_month_usd=0.30,
            iops_included=7000, throughput_mbps=150,
            read_latency_us=2500,
            multi_attach=True,
            scale_ceiling_tb=1024,
        ),
        Storage(
            name="FSx for Lustre (file, HPC)", kind="file",
            gb_month_usd=0.145,
            iops_included=1_000_000, throughput_mbps=125_000,
            read_latency_us=200,
            multi_attach=True,
            scale_ceiling_tb=1024,
        ),
    ]


# =============================================================================
# Report
# =============================================================================

def report(workload: Workload) -> None:
    options = build_options()

    print("=" * 80)
    print(f"Workload: {workload.name}")
    print(f"  size={workload.size_gb} GB, concurrent readers={workload.concurrent_readers},")
    print(f"  pattern={workload.rw_pattern}, IOPS target={workload.ops_per_sec:,}")
    print("=" * 80)

    print(f"{'Storage':<32} {'Monthly':<12} {'IOPS':<10} {'Latency':<10} {'Multi':<6} {'Fit':<5}")
    print(f"{'-'*31:<32} {'-'*11:<12} {'-'*9:<10} {'-'*9:<10} {'-'*5:<6} {'-'*4:<5}")
    for s in options:
        print(
            f"{s.name:<32} "
            f"${s.monthly_cost(workload.size_gb):>8.2f}   "
            f"{s.iops_included:>7,}   "
            f"{s.read_latency_us:>6} us   "
            f"{'yes' if s.multi_attach else 'no ':<6}"
            f"{fit_score(s, workload)}/10"
        )


def discuss() -> None:
    print()
    print("Heuristics:")
    print("  * Single-writer DB data dir      → block (gp3 first, io2 if IOPS-bound)")
    print("  * Many VMs reading shared data   → file (EFS) — ONLY use case where block loses")
    print("  * HPC or ML training with shared → FSx Lustre / parallel filesystem")
    print("  * Analytics / data lake          → object storage (see example 05), not block/file")
    print()
    print("The temptation is to pick file storage 'because it is shared' even when one writer")
    print("is enough; the per-GB premium (~3×) adds up fast on large volumes.")


# =============================================================================

def main() -> None:
    workloads = [
        Workload("Postgres data dir (single writer)",
                 size_gb=500, concurrent_readers=1, rw_pattern="random", ops_per_sec=5_000),
        Workload("Shared web assets (16 web nodes)",
                 size_gb=100, concurrent_readers=16, rw_pattern="random", ops_per_sec=800),
        Workload("ML training scratch (8 GPU nodes)",
                 size_gb=50_000, concurrent_readers=8, rw_pattern="sequential", ops_per_sec=200_000),
    ]
    for w in workloads:
        report(w)
        print()
    discuss()


if __name__ == "__main__":
    main()

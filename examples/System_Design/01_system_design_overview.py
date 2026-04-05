"""
System Design Overview — Back-of-Envelope Calculations

Demonstrates:
- Capacity estimation (storage, bandwidth, QPS)
- Latency number intuition
- Power-of-two reference table
- Request estimation from DAU

Theory:
- System design interviews and real architecture work begin with rough
  capacity estimates to determine whether a design is feasible.
- Back-of-envelope calculations use order-of-magnitude math to quickly
  approximate storage, bandwidth, and compute requirements.
- Key latency numbers (L1 cache ~1ns, SSD read ~100μs, network round-trip
  ~500μs, disk seek ~10ms) drive architectural decisions.

Adapted from System Design Lesson 01.
"""

from dataclasses import dataclass


# ── Power-of-Two Reference ────────────────────────────────────────────

POWERS_OF_TWO = {
    "1 KB": 2**10,
    "1 MB": 2**20,
    "1 GB": 2**30,
    "1 TB": 2**40,
    "1 PB": 2**50,
}

# Why: These latency numbers (from Jeff Dean's famous list) are the foundation
# of every back-of-envelope calculation. Knowing that a disk seek is 10,000x
# slower than a memory read immediately tells you whether caching is worthwhile.
LATENCY_NUMBERS = {
    "L1 cache reference":           0.5,       # ns
    "Branch mispredict":            5,          # ns
    "L2 cache reference":           7,          # ns
    "Mutex lock/unlock":            25,         # ns
    "Main memory reference":        100,        # ns
    "Compress 1KB (Snappy)":        3_000,      # ns
    "Send 1KB over 1Gbps network":  10_000,     # ns
    "Read 4KB randomly from SSD":   150_000,    # ns
    "Read 1MB sequentially (mem)":  250_000,    # ns
    "Round trip in same datacenter": 500_000,   # ns
    "Read 1MB sequentially (SSD)":  1_000_000,  # ns
    "Disk seek":                    10_000_000, # ns
    "Read 1MB sequentially (disk)": 20_000_000, # ns
    "Send packet CA→NL→CA":        150_000_000, # ns
}


# ── Capacity Estimator ────────────────────────────────────────────────

@dataclass
class ServiceEstimate:
    """Back-of-envelope estimate for a service."""
    name: str
    dau: int                    # daily active users
    requests_per_user_day: int  # avg requests per user per day
    avg_request_kb: float       # average request size in KB
    avg_storage_kb: float       # average storage per write in KB
    write_ratio: float          # fraction of requests that are writes (0-1)
    replication_factor: int = 3

    @property
    def daily_requests(self) -> int:
        return self.dau * self.requests_per_user_day

    @property
    def peak_qps(self) -> float:
        # Why: Peak QPS is typically 2-3x the average. Using 2x is a common
        # conservative estimate. We divide daily requests by 86400 seconds
        # to get average QPS, then multiply by peak factor.
        avg_qps = self.daily_requests / 86_400
        return avg_qps * 2  # 2x peak factor

    @property
    def daily_bandwidth_gb(self) -> float:
        total_kb = self.daily_requests * self.avg_request_kb
        return total_kb / (1024 * 1024)

    @property
    def daily_storage_gb(self) -> float:
        writes = self.daily_requests * self.write_ratio
        raw_gb = (writes * self.avg_storage_kb) / (1024 * 1024)
        return raw_gb * self.replication_factor

    @property
    def yearly_storage_tb(self) -> float:
        return self.daily_storage_gb * 365 / 1024

    def summary(self) -> str:
        lines = [
            f"  Service: {self.name}",
            f"  DAU: {self.dau:,}",
            f"  Daily requests: {self.daily_requests:,}",
            f"  Average QPS: {self.daily_requests / 86_400:,.0f}",
            f"  Peak QPS (2x): {self.peak_qps:,.0f}",
            f"  Daily bandwidth: {self.daily_bandwidth_gb:,.1f} GB",
            f"  Daily storage (x{self.replication_factor} repl): "
            f"{self.daily_storage_gb:,.1f} GB",
            f"  Yearly storage: {self.yearly_storage_tb:,.1f} TB",
        ]
        return "\n".join(lines)


# Why: Converting between units quickly is essential during design interviews.
# A wrong unit conversion can lead to an estimate off by 1000x, causing you
# to over-provision (wasting money) or under-provision (causing outages).
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between storage units."""
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    bytes_val = value * units[from_unit]
    return bytes_val / units[to_unit]


# ── SLA Calculator ────────────────────────────────────────────────────

def availability_downtime(nines: int) -> dict[str, float]:
    """Calculate allowed downtime for a given number of nines."""
    fraction = 1 - (1 - 10**(-nines))
    seconds_year = 365.25 * 24 * 3600
    down_seconds = fraction * seconds_year
    return {
        "availability": f"{(1 - fraction) * 100:.{nines}f}%",
        "downtime_per_year_hours": down_seconds / 3600,
        "downtime_per_month_minutes": down_seconds / 12 / 60,
        "downtime_per_day_seconds": down_seconds / 365.25,
    }


# ── Demos ─────────────────────────────────────────────────────────────

def demo_latency_reference():
    print("=" * 60)
    print("LATENCY NUMBERS EVERY ENGINEER SHOULD KNOW")
    print("=" * 60)

    print(f"\n  {'Operation':<40} {'Latency':>12} {'Relative':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*10}")

    base = LATENCY_NUMBERS["Main memory reference"]
    for name, ns in LATENCY_NUMBERS.items():
        if ns < 1_000:
            fmt = f"{ns:.0f} ns"
        elif ns < 1_000_000:
            fmt = f"{ns/1_000:.0f} μs"
        else:
            fmt = f"{ns/1_000_000:.0f} ms"
        relative = f"{ns/base:.0f}x mem"
        print(f"  {name:<40} {fmt:>12} {relative:>10}")


def demo_capacity_estimation():
    print("\n" + "=" * 60)
    print("CAPACITY ESTIMATION EXAMPLES")
    print("=" * 60)

    services = [
        ServiceEstimate("Twitter-like", dau=200_000_000,
                        requests_per_user_day=20, avg_request_kb=5,
                        avg_storage_kb=2, write_ratio=0.05),
        ServiceEstimate("URL Shortener", dau=100_000_000,
                        requests_per_user_day=5, avg_request_kb=0.5,
                        avg_storage_kb=0.1, write_ratio=0.01),
        ServiceEstimate("Chat Service", dau=50_000_000,
                        requests_per_user_day=100, avg_request_kb=1,
                        avg_storage_kb=0.5, write_ratio=0.5),
    ]

    for svc in services:
        print(f"\n{svc.summary()}")


def demo_sla_availability():
    print("\n" + "=" * 60)
    print("SLA AVAILABILITY TABLE")
    print("=" * 60)

    print(f"\n  {'Nines':>6} {'Availability':>14} {'Year (hrs)':>12} "
          f"{'Month (min)':>12} {'Day (sec)':>10}")
    print(f"  {'-'*6} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")

    for nines in range(1, 6):
        dt = availability_downtime(nines)
        print(f"  {nines:>6} {dt['availability']:>14} "
              f"{dt['downtime_per_year_hours']:>12.1f} "
              f"{dt['downtime_per_month_minutes']:>12.1f} "
              f"{dt['downtime_per_day_seconds']:>10.1f}")


def demo_quick_math():
    print("\n" + "=" * 60)
    print("QUICK MATH: POWER-OF-TWO REFERENCE")
    print("=" * 60)

    print(f"\n  {'Name':<8} {'Exact Bytes':>20} {'Approx':>12}")
    print(f"  {'-'*8} {'-'*20} {'-'*12}")
    for name, val in POWERS_OF_TWO.items():
        print(f"  {name:<8} {val:>20,} {val:.2e}")

    print(f"\n  Quick conversions:")
    examples = [
        (500, "MB", "GB"), (2.5, "TB", "GB"),
        (100, "GB", "TB"), (8, "KB", "B"),
    ]
    for val, frm, to in examples:
        result = convert_units(val, frm, to)
        print(f"    {val} {frm} = {result:,.1f} {to}")


if __name__ == "__main__":
    demo_latency_reference()
    demo_capacity_estimation()
    demo_sla_availability()
    demo_quick_math()

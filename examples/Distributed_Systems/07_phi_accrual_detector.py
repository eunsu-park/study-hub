"""
Phi Accrual Failure Detector

Unlike binary failure detectors (alive/dead), the phi accrual failure detector
outputs a continuous suspicion level (phi). This value represents the
confidence that the monitored node has failed, based on the statistical
distribution of heartbeat inter-arrival times.

Key concepts:
- Track heartbeat inter-arrival times with a sliding window
- Model inter-arrival distribution as normal (Gaussian)
- Compute phi = -log10(1 - CDF(t_now - t_last))
- Higher phi = higher confidence the node has failed
- Configurable threshold (e.g., phi > 8 => declare failure)
- Used in Apache Cassandra, Akka, and other distributed systems

Reference: Hayashibara et al., "The Phi Accrual Failure Detector" (2004)

Usage:
    python 07_phi_accrual_detector.py
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


class PhiAccrualDetector:
    """
    Phi accrual failure detector.

    Tracks heartbeat inter-arrival times and computes a suspicion level
    (phi) based on the probability that the monitored node has failed.
    """

    def __init__(self, threshold: float = 8.0, max_sample_size: int = 200,
                 min_std_dev_ms: float = 100.0):
        """
        Args:
            threshold: Phi value above which a node is considered failed.
                       phi=1 => 10% chance of false positive
                       phi=2 => 1%
                       phi=8 => 0.00000001% (very conservative)
            max_sample_size: Maximum number of inter-arrival samples to keep.
            min_std_dev_ms: Minimum standard deviation (ms) to avoid division
                            by zero when heartbeats are very regular.
        """
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.min_std_dev_ms = min_std_dev_ms

        # Sliding window of inter-arrival times (in milliseconds)
        self._intervals: deque[float] = deque(maxlen=max_sample_size)
        self._last_heartbeat_ms: float | None = None

    def heartbeat(self, now_ms: float | None = None) -> None:
        """
        Record a heartbeat arrival.

        Args:
            now_ms: Current time in milliseconds (uses wall clock if None).
        """
        if now_ms is None:
            now_ms = time.time() * 1000.0

        if self._last_heartbeat_ms is not None:
            interval = now_ms - self._last_heartbeat_ms
            self._intervals.append(interval)

        self._last_heartbeat_ms = now_ms

    def phi(self, now_ms: float | None = None) -> float:
        """
        Compute the current phi (suspicion level).

        Returns:
            The phi value. Higher = more suspicious.
            Returns 0.0 if insufficient data (< 2 samples).
        """
        if now_ms is None:
            now_ms = time.time() * 1000.0

        if len(self._intervals) < 2 or self._last_heartbeat_ms is None:
            return 0.0

        # Time elapsed since last heartbeat
        t_diff = now_ms - self._last_heartbeat_ms

        # Compute mean and std dev of inter-arrival times
        mean = sum(self._intervals) / len(self._intervals)
        variance = sum((x - mean) ** 2 for x in self._intervals) / len(self._intervals)
        std_dev = max(math.sqrt(variance), self.min_std_dev_ms)

        # Phi = -log10(1 - CDF(t_diff))
        # CDF of normal distribution
        # Using the complementary CDF for numerical stability:
        # P(X > t_diff) = 1 - CDF(t_diff)
        # phi = -log10(P(X > t_diff))
        p_later = self._complementary_cdf(t_diff, mean, std_dev)

        if p_later <= 0:
            return float("inf")  # Certainly failed
        if p_later >= 1:
            return 0.0  # No suspicion

        return -math.log10(p_later)

    @staticmethod
    def _complementary_cdf(x: float, mean: float, std_dev: float) -> float:
        """
        Compute P(X > x) for normal distribution N(mean, std_dev^2).
        Uses the complementary error function for numerical stability.
        """
        z = (x - mean) / std_dev
        # P(X > x) = 0.5 * erfc(z / sqrt(2))
        return 0.5 * math.erfc(z / math.sqrt(2))

    def is_available(self, now_ms: float | None = None) -> bool:
        """Check if the monitored node is considered available."""
        return self.phi(now_ms) < self.threshold

    @property
    def mean_interval(self) -> float:
        """Mean inter-arrival time in ms."""
        if not self._intervals:
            return 0.0
        return sum(self._intervals) / len(self._intervals)

    @property
    def std_dev_interval(self) -> float:
        """Standard deviation of inter-arrival times in ms."""
        if len(self._intervals) < 2:
            return 0.0
        mean = self.mean_interval
        var = sum((x - mean) ** 2 for x in self._intervals) / len(self._intervals)
        return math.sqrt(var)


@dataclass
class SimulatedNode:
    """A simulated node that sends heartbeats with configurable behavior."""
    name: str
    heartbeat_interval_ms: float = 1000.0   # Normal interval
    jitter_ms: float = 50.0                  # Random jitter (+/-)
    is_alive: bool = True
    slow_factor: float = 1.0                 # >1 means slower heartbeats


def simulate_nodes() -> None:
    """
    Simulate three nodes: healthy, slow, and failed.
    Track phi values over time for each.
    """
    print("=" * 70)
    print("Phi Accrual Failure Detector Simulation")
    print("=" * 70)

    # Use deterministic simulation time (not wall clock)
    import random
    random.seed(42)

    threshold = 8.0
    heartbeat_interval = 1000.0  # 1 second in ms

    # Create detectors for 3 nodes
    detector_healthy = PhiAccrualDetector(threshold=threshold)
    detector_slow = PhiAccrualDetector(threshold=threshold)
    detector_failed = PhiAccrualDetector(threshold=threshold)

    detectors = {
        "healthy": detector_healthy,
        "slow":    detector_slow,
        "failed":  detector_failed,
    }

    # --- Phase 1: Warmup (all nodes send regular heartbeats) ---
    print("\nPhase 1: Warmup (all nodes healthy, 10 heartbeats each)")
    t = 0.0
    for i in range(10):
        for name, det in detectors.items():
            jitter = random.uniform(-50, 50)
            det.heartbeat(t + jitter)
        t += heartbeat_interval

    print(f"  Mean interval (healthy): {detector_healthy.mean_interval:.0f} ms")
    print(f"  Std dev       (healthy): {detector_healthy.std_dev_interval:.0f} ms")

    # --- Phase 2: Divergent behavior ---
    print("\nPhase 2: Divergent behavior over 15 seconds")
    print(f"  - healthy: continues normal heartbeats (interval ~{heartbeat_interval:.0f}ms)")
    print(f"  - slow:    heartbeats slow down to 3x normal interval")
    print(f"  - failed:  stops sending heartbeats at t=10s\n")

    fail_time = t  # Node "failed" stops here

    print(f"  {'Time(s)':>8}  {'healthy_phi':>12}  {'slow_phi':>10}  {'failed_phi':>12}  "
          f"{'healthy':>8}  {'slow':>6}  {'failed':>8}")
    print("  " + "-" * 76)

    # Store data for "plot-ready" output
    plot_data: list[tuple[float, float, float, float]] = []

    for step in range(15):
        current_t = t + step * heartbeat_interval

        # Healthy node: normal heartbeat
        jitter = random.uniform(-50, 50)
        detector_healthy.heartbeat(current_t + jitter)

        # Slow node: heartbeat every 3 intervals (only every 3rd step)
        if step % 3 == 0:
            jitter = random.uniform(-100, 100)
            detector_slow.heartbeat(current_t + jitter)

        # Failed node: no heartbeats at all

        # Measure phi at a point slightly after expected heartbeat
        measure_t = current_t + heartbeat_interval * 0.5

        phi_h = detector_healthy.phi(measure_t)
        phi_s = detector_slow.phi(measure_t)
        phi_f = detector_failed.phi(measure_t)

        avail_h = "UP" if detector_healthy.is_available(measure_t) else "DOWN"
        avail_s = "UP" if detector_slow.is_available(measure_t) else "DOWN"
        avail_f = "UP" if detector_failed.is_available(measure_t) else "DOWN"

        time_s = (current_t - fail_time) / 1000.0

        print(f"  {time_s:>7.1f}s  {phi_h:>12.2f}  {phi_s:>10.2f}  {phi_f:>12.2f}  "
              f"{avail_h:>8}  {avail_s:>6}  {avail_f:>8}")

        plot_data.append((time_s, phi_h, phi_s, phi_f))

    # --- Summary ---
    print(f"\n  Threshold: phi > {threshold} => declare FAILED")
    print(f"\n  healthy node: phi stays low — heartbeats arrive on time")
    print(f"  slow node:    phi spikes between heartbeats, drops when one arrives")
    print(f"  failed node:  phi increases monotonically — no heartbeats")


def demonstrate_threshold_sensitivity() -> None:
    """Show how different thresholds affect detection time."""
    print("\n" + "=" * 70)
    print("Threshold Sensitivity Analysis")
    print("=" * 70)

    import random
    random.seed(99)

    heartbeat_interval = 1000.0
    thresholds = [1, 2, 4, 8, 12, 16]

    print(f"\n  How quickly each threshold detects a failed node")
    print(f"  (node sends 20 heartbeats at 1s interval, then stops)\n")

    print(f"  {'Threshold':>10}  {'Detection Time':>15}  {'False Positive Rate':>20}")
    print("  " + "-" * 50)

    for thresh in thresholds:
        det = PhiAccrualDetector(threshold=thresh)

        # Warmup: 20 regular heartbeats
        t = 0.0
        for i in range(20):
            jitter = random.uniform(-50, 50)
            det.heartbeat(t + jitter)
            t += heartbeat_interval

        fail_start = t

        # Check every 100ms after failure
        detection_time = None
        for step in range(200):
            check_t = fail_start + step * 100.0
            if not det.is_available(check_t):
                detection_time = (check_t - fail_start) / 1000.0
                break

        fp_rate = 10.0 ** (-thresh) * 100  # Theoretical false positive %

        det_str = f"{detection_time:.1f}s" if detection_time else ">20s"
        print(f"  {thresh:>10}  {det_str:>15}  {fp_rate:>19.8f}%")

    print("""
  Tradeoff:
  - Low threshold (1-2):  Fast detection, but higher false positive rate
  - High threshold (8+):  Slower detection, but extremely low false positives
  - Cassandra default:    phi=8 (good balance for production systems)
""")


def explain_phi_calculation() -> None:
    """Walk through the phi calculation step by step."""
    print("=" * 70)
    print("Phi Calculation Walkthrough")
    print("=" * 70)

    det = PhiAccrualDetector(threshold=8.0)

    # Simulate known heartbeat pattern
    intervals = [1000, 1050, 980, 1020, 990, 1010, 1030, 970, 1000, 1040]
    t = 0.0
    for interval in intervals:
        det.heartbeat(t)
        t += interval

    mean = det.mean_interval
    std = det.std_dev_interval

    print(f"\n  Heartbeat intervals (ms): {intervals}")
    print(f"  Mean:    {mean:.1f} ms")
    print(f"  Std dev: {std:.1f} ms\n")

    # Calculate phi at various delays after last heartbeat
    delays = [500, 1000, 1500, 2000, 3000, 5000, 10000]

    print(f"  {'Delay (ms)':>12}  {'P(X > delay)':>14}  {'phi':>8}  {'Interpretation':>20}")
    print("  " + "-" * 60)

    for delay in delays:
        measure_t = t + delay
        p = det._complementary_cdf(delay, mean, max(std, det.min_std_dev_ms))
        phi = det.phi(measure_t)

        if phi < 1:
            interp = "likely alive"
        elif phi < 4:
            interp = "uncertain"
        elif phi < 8:
            interp = "suspicious"
        else:
            interp = "probably dead"

        print(f"  {delay:>12}  {p:>14.8f}  {phi:>8.2f}  {interp:>20}")

    print("""
  Formula: phi = -log10(P(X > t_since_last))
  where X ~ N(mean, std_dev^2) is the inter-arrival time distribution.

  Interpretation:
    phi = 1  =>  10%       chance this is a false alarm
    phi = 2  =>  1%        chance this is a false alarm
    phi = 8  =>  0.000001% chance this is a false alarm
""")


if __name__ == "__main__":
    explain_phi_calculation()
    simulate_nodes()
    demonstrate_threshold_sensitivity()
    print("Done.")

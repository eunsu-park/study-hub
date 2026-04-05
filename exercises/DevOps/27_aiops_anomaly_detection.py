#!/usr/bin/env python3
"""Exercises for Lesson 27: AIOps and Anomaly Detection
Topic: DevOps
"""

import math


def exercise_1():
    """EWMA-based anomaly detection on latency data."""
    print("=== Exercise 1: EWMA Anomaly Detection ===\n")

    latency_data = [
        100, 102, 98, 105, 99, 101, 97, 103, 100, 98,
        102, 99, 101, 100, 103, 97, 105, 98, 100, 101,
        250, 280, 260,
        102, 100, 98, 101, 99,
        100, 101, 99, 102, 100, 98, 103, 101, 100, 99,
        115, 118, 120, 122, 125, 128, 130, 133, 135, 138,
    ]

    alpha = 0.1
    threshold_sigma = 3.0
    ewma = float(latency_data[0])
    ewma_var = 0.0
    anomalies = []

    for i, value in enumerate(latency_data):
        diff = value - ewma
        ewma = alpha * value + (1 - alpha) * ewma
        ewma_var = (1 - alpha) * (ewma_var + alpha * diff * diff)
        ewma_std = math.sqrt(ewma_var) if ewma_var > 0 else 0

        is_anomaly = abs(value - ewma) > threshold_sigma * ewma_std if ewma_std > 0 else False

        if is_anomaly:
            anomalies.append(i + 1)
            print(f"  Minute {i+1}: value={value}, ewma={ewma:.1f}, "
                  f"std={ewma_std:.1f}, ANOMALY")

    print(f"\nTotal anomalies detected: {len(anomalies)}")
    print(f"Anomaly minutes: {anomalies}")
    print("\nNote: Minutes 21-23 (250, 280, 260) are clear anomalies.")
    print("Minutes 39-48 (gradual increase) are NOT detected because EWMA adapts.")
    print("For gradual trends, use trend detection (slope analysis) or longer baselines.")


def exercise_2():
    """Alert correlation strategy design."""
    print("\n=== Exercise 2: Alert Correlation Design ===\n")

    print("Strategy to reduce 47 alerts to ~3 actionable groups:\n")
    print("1. Grouping: group_by=[cluster, namespace, service], group_wait=60s")
    print("   → 47 alerts collapse to ~5-10 groups\n")
    print("2. Inhibition rules:")
    print("   - Infrastructure down → suppress app alerts (same cluster)")
    print("   - Database down → suppress dependent service alerts")
    print("   - Critical → suppress warning (same service)\n")
    print("3. Topology-aware root cause identification:")
    print("   - Use service graph to find first-to-alert service")
    print("   - Group downstream cascade under root cause\n")
    print("4. Presentation:")
    print("   Group 1: CRITICAL -- payment-service (root cause)")
    print("     3 alerts + 25 suppressed downstream")
    print("   Group 2: WARNING -- Cascading from Group 1")
    print("     18 alerts, will resolve when Group 1 fixed")
    print("   Group 3: INFO -- Unrelated (disk space on monitoring node)")
    print("     1 alert, separate issue")
    print("\n   Result: 47 alerts → 3 actionable groups")


if __name__ == "__main__":
    exercise_1()
    exercise_2()

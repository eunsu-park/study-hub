#!/usr/bin/env python3
"""Exercises for Lesson 23: OpenTelemetry Pipelines
Topic: DevOps
"""


def exercise_1():
    """Pipeline design with cost analysis."""
    print("=== Exercise 1: OTel Pipeline Design ===\n")
    initial_spans_sec = 10000
    after_health_filter = int(initial_spans_sec * 0.8)
    after_sampling = int(after_health_filter * 0.15)  # errors + slow + 10% sample
    print(f"Initial: {initial_spans_sec:,} spans/sec")
    print(f"After health check filter (~20% removed): {after_health_filter:,}")
    print(f"After tail sampling (errors + slow + 10%): ~{after_sampling:,}")
    print(f"Total reduction: {(1 - after_sampling/initial_spans_sec)*100:.0f}%")
    print(f"\nKey: spanmetrics generates metrics from ALL traces (before sampling)")
    print("so exemplars reference sampled traces.")


def exercise_2():
    """Troubleshooting cascade diagnosis."""
    print("\n=== Exercise 2: Collector Troubleshooting ===\n")
    symptoms = [
        ("Export latency 50ms → 5s", "ROOT CAUSE: Tempo backend overloaded"),
        ("Queue 4800/5000", "EFFECT: Slow exports fill the queue"),
        ("Memory 1.8/2.0 GB", "EFFECT: Full queue + buffered data consume memory"),
        ("Refused spans 500/sec", "EFFECT: memory_limiter rejects to prevent OOM"),
    ]
    print("Cascade analysis (bottom-up):")
    for symptom, diagnosis in symptoms:
        print(f"  {symptom}")
        print(f"    → {diagnosis}")

    print("\nFixes:")
    fixes = [
        ("Export latency", "Check Tempo health, scale Tempo ingesters"),
        ("Queue full", "Increase queue_size to 10000, add num_consumers"),
        ("High memory", "Increase limit_mib, reduce tail_sampling num_traces"),
        ("Refused spans", "Add more Collector replicas, load balance"),
    ]
    for symptom, fix in fixes:
        print(f"  {symptom}: {fix}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()

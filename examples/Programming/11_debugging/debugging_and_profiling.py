"""
Debugging & Profiling

Demonstrates systematic debugging and performance profiling:
1. A bug hunt — reproduce, hypothesize, test, fix
2. Interactive debugging with breakpoint() / pdb
3. CPU profiling with cProfile to find hotspots
4. Wall-clock timing with timeit for micro-benchmarks
5. Memory tracking with tracemalloc to spot allocation-heavy code paths

Running this script prints the full workflow. The breakpoint() call in
`demonstrate_pdb` is guarded behind a flag so the script runs non-interactively
by default; set `INTERACTIVE=1` in the environment to drop into pdb.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import timeit
import tracemalloc
from typing import List


# =============================================================================
# 1. A BUG HUNT — from observation to verified fix
# =============================================================================
#
# Scenario: `average_positives` should return the mean of positive numbers,
# or 0 when the list has none. The buggy version has a classic off-by-one /
# empty-sequence oversight. The fixed version handles it correctly.


def average_positives_buggy(numbers: List[float]) -> float:
    """Buggy: divides by len(numbers), not the count of positives."""
    total = sum(n for n in numbers if n > 0)
    return total / len(numbers)  # wrong denominator


def average_positives_fixed(numbers: List[float]) -> float:
    """Fixed: divides by the count of positives, handles empty case."""
    positives = [n for n in numbers if n > 0]
    if not positives:
        return 0.0
    return sum(positives) / len(positives)


def demonstrate_bug_hunt() -> None:
    """
    Systematic debugging workflow:
      (1) Observe: buggy and fixed disagree on simple input.
      (2) Hypothesize: denominator is wrong when negatives are present.
      (3) Test the hypothesis with targeted inputs.
      (4) Fix, then verify with the previously failing case.
    """
    data = [10, -5, 20, -3]

    buggy = average_positives_buggy(data)
    fixed = average_positives_fixed(data)

    print(f"  input:          {data}")
    print(f"  expected mean of positives: {(10 + 20) / 2} = 15.0")
    print(f"  buggy output:   {buggy}   (divided by {len(data)} instead of 2)")
    print(f"  fixed output:   {fixed}")

    # Regression check: the empty-positives case used to raise/diverge
    edge = [-1, -2, -3]
    print(f"  edge case {edge} -> fixed returns {average_positives_fixed(edge)}")


# =============================================================================
# 2. INTERACTIVE DEBUGGING — pdb via breakpoint()
# =============================================================================

def _find_max_inefficient(numbers: List[int]) -> int:
    """O(n^2) max finder — deliberately slow for the profiler demo below."""
    largest = numbers[0]
    for i in range(len(numbers)):
        # Intentional inefficiency: scanning from 0 each time
        for j in range(len(numbers)):
            if numbers[j] > largest:
                largest = numbers[j]
    return largest


def demonstrate_pdb() -> None:
    """
    In a real session, `breakpoint()` drops you into pdb. Inside pdb:
      n  — next line
      s  — step into
      p <expr> — print expression
      l  — list surrounding source
      c  — continue
    Set `INTERACTIVE=1` in the environment to exercise it here.
    """
    data = [3, 1, 4, 1, 5, 9, 2, 6]

    if os.environ.get("INTERACTIVE") == "1":
        breakpoint()  # noqa: T100 — intentional for educational demo

    result = _find_max_inefficient(data)
    print(f"  max of {data} = {result}")
    print(f"  (set INTERACTIVE=1 to drop into pdb before computing)")


# =============================================================================
# 3. CPU PROFILING — cProfile for hotspot discovery
# =============================================================================

def demonstrate_cprofile() -> None:
    """
    cProfile counts calls and cumulative time per function. The output
    shows `_find_max_inefficient` dominating — proving the hotspot without
    guessing. Fix the hotspot, re-profile, and verify the improvement.
    """
    data = list(range(1, 501))  # 500 elements → ~250k inner comparisons

    profiler = cProfile.Profile()
    profiler.enable()
    _find_max_inefficient(data)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(5)  # top 5 by cumulative time

    print("  top 5 functions by cumulative time:")
    for line in buffer.getvalue().splitlines()[-10:]:  # trim header
        if line.strip():
            print(f"    {line}")


# =============================================================================
# 4. MICRO-BENCHMARKS — timeit for repeatable measurements
# =============================================================================

def demonstrate_timeit() -> None:
    """
    timeit runs the snippet many times and averages. A single run is noisy;
    many runs level out caches, GC, and scheduler jitter.

    We compare the O(n^2) loop against the built-in max(), which is O(n).
    """
    setup = "data = list(range(1, 501))"
    stmts = {
        "built-in max()": "max(data)",
        "O(n^2) inefficient": (
            "largest = data[0]\n"
            "for i in range(len(data)):\n"
            "    for j in range(len(data)):\n"
            "        if data[j] > largest:\n"
            "            largest = data[j]"
        ),
    }

    for label, stmt in stmts.items():
        seconds = timeit.timeit(stmt, setup=setup, number=200)
        print(f"  {label:<22} avg: {seconds / 200 * 1e6:8.2f} µs/call")


# =============================================================================
# 5. MEMORY TRACKING — tracemalloc for allocation hotspots
# =============================================================================

def _build_list_naive(n: int) -> List[int]:
    """Appends repeatedly — lots of small allocations."""
    result: List[int] = []
    for i in range(n):
        result = result + [i]  # creates a NEW list each iteration
    return result


def _build_list_efficient(n: int) -> List[int]:
    """Single list comprehension — one allocation for the final list."""
    return [i for i in range(n)]


def demonstrate_tracemalloc() -> None:
    """
    tracemalloc snapshots memory at two points; the diff reveals the
    call sites responsible for the most allocated bytes.
    """
    tracemalloc.start()

    snapshot_before = tracemalloc.take_snapshot()
    _build_list_naive(1000)
    snapshot_after = tracemalloc.take_snapshot()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

    print("  top 3 allocation sources (naive build):")
    for stat in top_stats[:3]:
        # filename:line and total size
        frame = stat.traceback[0]
        filename = os.path.basename(frame.filename)
        print(f"    {filename}:{frame.lineno}  +{stat.size_diff / 1024:.1f} KiB  ({stat.count_diff:+d} blocks)")

    tracemalloc.stop()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sections = [
        ("1. BUG HUNT (observe, hypothesize, test, fix)", demonstrate_bug_hunt),
        ("2. INTERACTIVE DEBUGGING (breakpoint / pdb)", demonstrate_pdb),
        ("3. CPU PROFILING (cProfile)", demonstrate_cprofile),
        ("4. MICRO-BENCHMARKS (timeit)", demonstrate_timeit),
        ("5. MEMORY TRACKING (tracemalloc)", demonstrate_tracemalloc),
    ]
    for title, fn in sections:
        print("=" * 70)
        print(title)
        print("=" * 70)
        fn()
        print()


if __name__ == "__main__":
    main()

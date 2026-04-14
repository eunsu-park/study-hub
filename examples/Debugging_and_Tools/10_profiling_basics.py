"""
10 Profiling Basics
===================
Demonstrates timing, cProfile, and memory profiling techniques
to find and fix performance bottlenecks.
"""
import cProfile
import pstats
import time
import timeit
import tracemalloc
from contextlib import contextmanager
from io import StringIO


@contextmanager
def timer(label="Block"):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {elapsed:.4f}s")


def timing_demo():
    """Demonstrate different timing approaches."""
    print("=== Timing Demo ===")

    with timer("sum(range(100000))"):
        total = sum(range(100000))

    with timer("list comprehension"):
        squares = [x**2 for x in range(100000)]

    with timer("generator expression"):
        total = sum(x**2 for x in range(100000))
    print()


def timeit_comparison():
    """Compare two approaches using timeit."""
    print("=== timeit Comparison ===")

    # String concatenation vs join
    n = 5000

    def concat_approach():
        s = ""
        for i in range(n):
            s += str(i)
        return s

    def join_approach():
        return "".join(str(i) for i in range(n))

    t1 = timeit.timeit(concat_approach, number=100)
    t2 = timeit.timeit(join_approach, number=100)

    print(f"  Concatenation: {t1:.4f}s")
    print(f"  Join:          {t2:.4f}s")
    print(f"  Join is {t1/t2:.1f}x faster")
    print()

    # List vs set lookup
    data_list = list(range(10000))
    data_set = set(range(10000))

    def list_lookup():
        return 9999 in data_list

    def set_lookup():
        return 9999 in data_set

    t1 = timeit.timeit(list_lookup, number=10000)
    t2 = timeit.timeit(set_lookup, number=10000)
    print(f"  List lookup: {t1:.4f}s")
    print(f"  Set lookup:  {t2:.4f}s")
    print(f"  Set is {t1/t2:.0f}x faster")
    print()


def cprofile_demo():
    """Demonstrate cProfile to find bottleneck functions."""
    print("=== cProfile Demo ===")
    import random

    def generate_data(n):
        return [random.random() for _ in range(n)]

    def bubble_sort(data):
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def find_median(sorted_data):
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        return sorted_data[n // 2]

    def pipeline():
        data = generate_data(2000)
        sorted_data = bubble_sort(data)
        median = find_median(sorted_data)
        return median

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    pipeline()
    profiler.disable()

    # Print results
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime")
    stats.print_stats(5)
    print(stream.getvalue())

    print("  Observation: bubble_sort takes >99% of time")
    print("  Fix: Replace with sorted() (O(n log n) vs O(n^2))")

    with timer("bubble_sort(2000 items)"):
        data = generate_data(2000)
        bubble_sort(data)

    with timer("sorted(2000 items)"):
        data = generate_data(2000)
        sorted(data)
    print()


def memory_profiling_demo():
    """Demonstrate memory profiling with tracemalloc."""
    print("=== Memory Profiling (tracemalloc) ===")

    tracemalloc.start()

    # Allocate some memory
    data = [i ** 2 for i in range(50000)]
    big_dict = {str(i): i ** 2 for i in range(50000)}

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")

    print("  Top 3 memory consumers:")
    for stat in stats[:3]:
        print(f"    {stat}")

    tracemalloc.stop()
    print()


def optimization_workflow():
    """Show the complete profile-optimize-verify workflow."""
    print("=== Optimization Workflow ===")
    import random

    data = [random.random() for _ in range(100000)]

    # Profile: identify bottleneck
    print("  1. Profile the slow version:")
    with timer("slow: O(n) search in list"):
        lookup_list = data.copy()
        count = sum(1 for _ in range(1000) if random.random() in lookup_list)

    # Optimize: use set
    print("  2. Optimize: use set for O(1) lookup:")
    with timer("fast: O(1) search in set"):
        lookup_set = set(data)
        count = sum(1 for _ in range(1000) if random.random() in lookup_set)

    # Verify improvement
    print("  3. Verify: set version is dramatically faster")
    print()


if __name__ == "__main__":
    timing_demo()
    timeit_comparison()
    cprofile_demo()
    memory_profiling_demo()
    optimization_workflow()

"""
Exercises for Lesson 10: Performance Optimization
Topic: Python

Solutions to practice problems from the lesson.
"""

import cProfile
import io
import pstats
import csv
import os
import tempfile
import time
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor


# === Exercise 1: Profiling ===
# Problem: Profile slow code and find and improve bottlenecks.
#
# Original slow code:
#   def slow_function(n):
#       result = ""
#       for i in range(n):
#           if i in [j for j in range(i)]:  # O(n) list creation + O(n) membership
#               result += str(i)             # O(n) string concatenation
#       return result

def slow_function(n: int) -> str:
    """Original slow implementation for comparison.

    Two performance issues:
    1. `[j for j in range(i)]` creates a new list each iteration -- O(i) work
       and `i in [...]` is O(i) linear scan, making the whole loop O(n^2).
    2. `result += str(i)` creates a new string each time -- O(n) amortized
       due to immutable string concatenation.
    """
    result = ""
    for i in range(n):
        if i in [j for j in range(i)]:
            result += str(i)
    return result


def fast_function(n: int) -> str:
    """Optimized version that fixes both bottlenecks.

    Analysis: `i in [j for j in range(i)]` checks if i is in [0..i-1].
    Since range(i) produces [0, 1, ..., i-1], the value i itself is
    NEVER in that range. So the condition is always False, and the
    original function always returns an empty string.

    The real lesson: profiling reveals that the slow code does nothing
    useful. The optimized version recognizes this and returns immediately.
    Both bottlenecks (list creation and string concatenation) are
    completely eliminated by understanding what the code actually does.
    """
    # The condition `i in range(i)` is always False, so no strings are ever appended
    return ""


def exercise_1():
    """Profile slow vs fast and show the improvement."""
    n = 3000

    # Profile the slow version
    print("Profiling slow_function:")
    profiler = cProfile.Profile()
    profiler.enable()
    slow_result = slow_function(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(5)
    print(stream.getvalue()[:500])

    # Time comparison
    start = time.time()
    slow_function(n)
    slow_time = time.time() - start

    start = time.time()
    fast_function(n)
    fast_time = time.time() - start

    print(f"slow_function({n}): {slow_time:.4f}s")
    print(f"fast_function({n}): {fast_time:.6f}s")
    print(f"Speedup: {slow_time / max(fast_time, 1e-9):.0f}x")

    # Verify correctness
    assert slow_function(100) == fast_function(100), "Results differ!"
    print("Correctness verified: both produce identical output.")


# === Exercise 2: Memory Optimization ===
# Problem: Write a function to efficiently process large CSV files.

def generate_test_csv(filepath: str, num_rows: int = 100_000):
    """Create a test CSV file for processing."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "department", "salary"])
        for i in range(num_rows):
            dept = ["Engineering", "Marketing", "Sales", "HR"][i % 4]
            writer.writerow([i, f"Employee_{i}", dept, 50000 + (i % 50) * 1000])


def process_csv_naive(filepath: str) -> dict[str, float]:
    """Naive approach: load entire file into memory.

    This works for small files but fails for files larger than available RAM.
    """
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)  # Load ALL rows into memory

    # Calculate average salary by department
    dept_totals: dict[str, list[float]] = {}
    for row in rows:
        dept = row["department"]
        salary = float(row["salary"])
        dept_totals.setdefault(dept, []).append(salary)

    return {dept: sum(salaries) / len(salaries) for dept, salaries in dept_totals.items()}


def process_csv_efficient(filepath: str) -> dict[str, float]:
    """Memory-efficient approach: stream rows one at a time.

    Uses a generator-based reader so only one row is in memory at a time.
    Accumulates running totals instead of storing all salaries in lists.
    Memory usage is O(number of departments) instead of O(number of rows).
    """
    dept_sum: dict[str, float] = {}
    dept_count: dict[str, int] = {}

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dept = row["department"]
            salary = float(row["salary"])
            dept_sum[dept] = dept_sum.get(dept, 0.0) + salary
            dept_count[dept] = dept_count.get(dept, 0) + 1

    return {dept: dept_sum[dept] / dept_count[dept] for dept in dept_sum}


def exercise_2():
    """Compare naive vs efficient CSV processing."""
    tmpfile = os.path.join(tempfile.gettempdir(), "test_data.csv")
    num_rows = 50_000

    generate_test_csv(tmpfile, num_rows)
    file_size = os.path.getsize(tmpfile)
    print(f"Generated CSV: {num_rows} rows, {file_size / 1024:.0f} KB")

    # Naive approach
    start = time.time()
    result1 = process_csv_naive(tmpfile)
    naive_time = time.time() - start
    print(f"\nNaive (load all): {naive_time:.4f}s")
    for dept, avg in sorted(result1.items()):
        print(f"  {dept}: ${avg:,.0f}")

    # Efficient approach
    start = time.time()
    result2 = process_csv_efficient(tmpfile)
    efficient_time = time.time() - start
    print(f"\nEfficient (streaming): {efficient_time:.4f}s")
    for dept, avg in sorted(result2.items()):
        print(f"  {dept}: ${avg:,.0f}")

    # Verify same results
    for dept in result1:
        assert abs(result1[dept] - result2[dept]) < 0.01
    print("\nResults match!")

    os.remove(tmpfile)


# === Exercise 3: Parallel Processing ===
# Problem: Parallelize CPU-bound tasks to improve performance.

def compute_heavy(n: int) -> int:
    """CPU-bound computation: sum of squares up to n.

    Deliberately uses a loop instead of a formula to simulate
    meaningful CPU work.
    """
    total = 0
    for i in range(n):
        total += i * i
    return total


def sequential_processing(tasks: list[int]) -> list[int]:
    """Process tasks sequentially -- baseline for comparison."""
    return [compute_heavy(n) for n in tasks]


def parallel_processing(tasks: list[int], num_workers: int = 4) -> list[int]:
    """Process tasks in parallel using ProcessPoolExecutor.

    Each worker gets a separate Python process, bypassing the GIL.
    This is effective for CPU-bound work. For I/O-bound work,
    use ThreadPoolExecutor or asyncio instead.
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_heavy, tasks))
    return results


def exercise_3():
    """Compare sequential vs parallel processing."""
    # Create CPU-bound tasks of varying sizes
    tasks = [500_000 + i * 100_000 for i in range(8)]
    print(f"Tasks: {len(tasks)} computations")

    # Sequential
    start = time.time()
    seq_results = sequential_processing(tasks)
    seq_time = time.time() - start
    print(f"Sequential: {seq_time:.3f}s")

    # Parallel
    start = time.time()
    par_results = parallel_processing(tasks, num_workers=4)
    par_time = time.time() - start
    print(f"Parallel (4 workers): {par_time:.3f}s")

    # Verify correctness
    assert seq_results == par_results, "Results differ!"
    print(f"Speedup: {seq_time / max(par_time, 1e-9):.2f}x")
    print("Results verified: sequential and parallel produce identical output.")


if __name__ == "__main__":
    print("=== Exercise 1: Profiling ===")
    exercise_1()

    print("\n=== Exercise 2: Memory Optimization ===")
    exercise_2()

    print("\n=== Exercise 3: Parallel Processing ===")
    exercise_3()

    print("\nAll exercises completed!")

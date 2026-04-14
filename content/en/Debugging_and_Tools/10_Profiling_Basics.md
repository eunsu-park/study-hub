# Profiling Basics

**Previous**: [Type Checking](./09_Type_Checking.md) | **Next**: [Version Control for Debugging](./11_Version_Control_for_Debugging.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between profiling and benchmarking
2. Use `timeit` to accurately measure execution time of code snippets
3. Use `cProfile` to find the slowest functions in a program
4. Read and interpret profiling output (ncalls, tottime, cumtime)
5. Use `time.perf_counter()` for manual timing of code sections
6. Profile memory usage with `tracemalloc` and `memory_profiler`
7. Avoid premature optimization by profiling before optimizing
8. Apply the 80/20 rule: find the 20% of code causing 80% of slowness

---

"Make it work, make it right, make it fast -- in that order." Before you optimize code, you need to **measure** it. Profiling tells you exactly where your program spends its time and memory, replacing guesses with data. Without profiling, you risk spending hours optimizing code that runs for 0.001 seconds while the real bottleneck hides elsewhere.

> **Premature Optimization:** Donald Knuth famously said, "Premature optimization is the root of all evil." Always profile first to identify the actual bottleneck before optimizing.

---

## 1. Timing Code

### 1.1 `time.perf_counter()` -- Manual Timing

```python
import time

start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"Took {elapsed:.4f} seconds")
```

### 1.2 A Reusable Timer

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(label="Block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{label}] {elapsed:.4f}s")

# Usage
with timer("Sort"):
    sorted_data = sorted(large_list)

with timer("Search"):
    result = binary_search(sorted_data, target)
```

### 1.3 `timeit` -- Accurate Microbenchmarks

`timeit` runs code many times to get a reliable measurement:

```python
import timeit

# Time a single expression
t = timeit.timeit('sum(range(1000))', number=10000)
print(f"Total: {t:.4f}s, Per call: {t/10000:.6f}s")

# Compare two approaches
t1 = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
t2 = timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000)
print(f"Generator: {t1:.4f}s")
print(f"List comp: {t2:.4f}s")
```

From the command line:

```bash
python -m timeit "sum(range(1000))"
# 100000 loops, best of 5: 12.3 usec per loop
```

### 1.4 Comparing Approaches

```python
import timeit

def approach_a():
    """String concatenation with +"""
    s = ""
    for i in range(1000):
        s += str(i)
    return s

def approach_b():
    """String building with join"""
    return "".join(str(i) for i in range(1000))

ta = timeit.timeit(approach_a, number=1000)
tb = timeit.timeit(approach_b, number=1000)
print(f"Concatenation: {ta:.4f}s")
print(f"Join:          {tb:.4f}s")
print(f"Join is {ta/tb:.1f}x faster")
```

---

## 2. cProfile: Finding Slow Functions

### 2.1 Basic Usage

```bash
python -m cProfile my_script.py
```

### 2.2 Programmatic Usage

```python
import cProfile

def main():
    data = generate_data(10000)
    processed = process_data(data)
    result = analyze(processed)
    return result

cProfile.run('main()')
```

### 2.3 Reading cProfile Output

```
         1000003 function calls in 2.543 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    2.543    2.543 script.py:1(main)
        1    0.012    0.012    2.543    2.543 script.py:5(process_data)
  1000000    2.100    0.000    2.100    0.000 script.py:12(transform)
        1    0.431    0.431    0.431    0.431 script.py:8(generate_data)
        1    0.000    0.000    0.000    0.000 script.py:15(analyze)
```

### Column Meanings

```
┌──────────────────────────────────────────────────────────────┐
│  Column     Meaning                                         │
├──────────────────────────────────────────────────────────────┤
│  ncalls     Number of times the function was called         │
│  tottime    Total time IN the function (excluding subcalls) │
│  percall    tottime / ncalls                                │
│  cumtime    Total time IN the function (including subcalls) │
│  percall    cumtime / ncalls                                │
│  filename   Source file, line number, and function name     │
└──────────────────────────────────────────────────────────────┘
```

**Key insight**: Look at `tottime` to find where time is actually spent. Look at `cumtime` to find which high-level functions are slow overall.

### 2.4 Sorting and Filtering

```python
import cProfile
import pstats

# Profile and save results
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions

# Filter to your code only (exclude library code)
stats.print_stats('my_module')
```

### 2.5 Saving Profile Results

```bash
python -m cProfile -o profile.dat my_script.py
```

```python
import pstats
stats = pstats.Stats('profile.dat')
stats.sort_stats('tottime')
stats.print_stats(10)
```

---

## 3. Practical Profiling Example

### Finding the Bottleneck

```python
import cProfile
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

def main():
    data = generate_data(5000)
    sorted_data = bubble_sort(data)    # This is the bottleneck!
    median = find_median(sorted_data)
    return median

cProfile.run('main()')
```

Output shows `bubble_sort` takes 99% of the time:
```
   ncalls  tottime  percall  cumtime  percall
        1    0.000    0.000    3.456    3.456 main()
        1    3.421    3.421    3.421    3.421 bubble_sort()  ← BOTTLENECK
        1    0.035    0.035    0.035    0.035 generate_data()
        1    0.000    0.000    0.000    0.000 find_median()
```

Fix: Replace `bubble_sort` with `sorted()`:
```python
def main():
    data = generate_data(5000)
    sorted_data = sorted(data)  # O(n log n) instead of O(n^2)
    median = find_median(sorted_data)
    return median
```

---

## 4. Memory Profiling

### 4.1 `tracemalloc` (Built-in)

```python
import tracemalloc

tracemalloc.start()

# Your code here
data = [i ** 2 for i in range(100000)]
big_dict = {str(i): i ** 2 for i in range(100000)}

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')

print("Top 10 memory consumers:")
for stat in stats[:10]:
    print(stat)
```

Output:
```
script.py:6: size=4816 KiB, count=100000, average=49 B
script.py:5: size=824 KiB, count=1, average=824 KiB
```

### 4.2 Tracking Memory Growth

```python
import tracemalloc

tracemalloc.start()

# Snapshot before
snapshot1 = tracemalloc.take_snapshot()

# Do some work
process_data()

# Snapshot after
snapshot2 = tracemalloc.take_snapshot()

# Compare
stats = snapshot2.compare_to(snapshot1, 'lineno')
print("Memory changes:")
for stat in stats[:10]:
    print(stat)
```

### 4.3 `memory_profiler` (Third-party, Line-by-Line)

```bash
pip install memory-profiler
```

```python
from memory_profiler import profile

@profile
def memory_hungry():
    a = [i for i in range(100000)]         # 3.5 MiB
    b = {i: i**2 for i in range(100000)}   # 6.0 MiB
    del a                                   # -3.5 MiB
    return b
```

```bash
$ python -m memory_profiler script.py
Line #  Mem usage    Increment  Line Contents
     4   45.2 MiB    45.2 MiB   @profile
     5                          def memory_hungry():
     6   48.7 MiB     3.5 MiB       a = [i for i in range(100000)]
     7   54.7 MiB     6.0 MiB       b = {i: i**2 for i in range(100000)}
     8   51.2 MiB    -3.5 MiB       del a
     9   51.2 MiB     0.0 MiB       return b
```

---

## 5. Common Performance Pitfalls

### 5.1 List Append vs Concatenation

```python
# SLOW: O(n^2) due to string copying
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# FAST: O(n)
result = "".join(str(item) for item in items)
```

### 5.2 Searching in Lists vs Sets

```python
# SLOW: O(n) lookup per check
items = list(range(100000))
if target in items:  # Linear scan
    ...

# FAST: O(1) lookup per check
items = set(range(100000))
if target in items:  # Hash lookup
    ...
```

### 5.3 Repeated Function Calls

```python
# SLOW: len() called every iteration
for i in range(len(data)):
    for j in range(len(data)):
        ...

# FASTER: Cache the length
n = len(data)
for i in range(n):
    for j in range(n):
        ...
```

### 5.4 Unnecessary Copies

```python
# SLOW: Creates a copy of the entire list
def process(data):
    sorted_data = sorted(data)  # Creates new list
    return sorted_data[0]

# FASTER: Use min() directly
def process(data):
    return min(data)  # O(n), no copy needed
```

---

## 6. The 80/20 Rule of Optimization

```
┌──────────────────────────────────────────────────┐
│  Typical Program Time Distribution               │
│                                                  │
│  ████████████████████████████████████  90%       │
│  ↑ One function (the bottleneck)                 │
│                                                  │
│  ██  5%                                          │
│  ↑ A few supporting functions                    │
│                                                  │
│  █  3%                                           │
│  ↑ I/O and system calls                          │
│                                                  │
│  ░  2%                                           │
│  ↑ Everything else                               │
│                                                  │
│  Optimizing "everything else" = wasted effort    │
│  Optimizing the bottleneck = massive speedup     │
└──────────────────────────────────────────────────┘
```

**Workflow:**
1. Profile first -- don't guess where the bottleneck is
2. Identify the top 1-3 functions by `tottime`
3. Optimize only those functions
4. Re-profile to verify improvement
5. Stop when performance is "good enough"

---

## 7. Profiling Tips

### 7.1 Profile with Realistic Data

```python
# BAD: Profiling with 10 items tells you nothing
profile(process, small_data)

# GOOD: Profile with production-sized data
profile(process, production_data)
```

### 7.2 Profile Multiple Times

Single measurements are noisy. Run profiling multiple times and look at the trend.

### 7.3 Use snakeviz for Visualization

```bash
pip install snakeviz
python -m cProfile -o profile.dat my_script.py
snakeviz profile.dat  # Opens interactive visualization in browser
```

### 7.4 Profile I/O Separately from CPU

```python
import time

# CPU profiling
start = time.process_time()
compute_heavy_function()
cpu_time = time.process_time() - start

# Wall clock profiling (includes I/O waits)
start = time.perf_counter()
io_heavy_function()
wall_time = time.perf_counter() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"Wall time: {wall_time:.4f}s")
print(f"I/O wait: {wall_time - cpu_time:.4f}s")
```

---

## 8. Quick Reference

| Tool | What It Measures | When to Use |
|------|-----------------|-------------|
| `time.perf_counter()` | Wall clock time | Quick manual timing |
| `timeit` | Execution time (averaged) | Microbenchmarks, comparing approaches |
| `cProfile` | Function call counts and times | Finding slow functions |
| `tracemalloc` | Memory allocation | Finding memory-heavy code |
| `memory_profiler` | Line-by-line memory usage | Detailed memory analysis |
| `snakeviz` | Visual profile viewer | Understanding complex profiles |

---

## Summary

- Always profile before optimizing -- don't guess where the bottleneck is
- `timeit` is the right tool for comparing two approaches
- `cProfile` identifies which functions consume the most time
- `tracemalloc` and `memory_profiler` identify memory bottlenecks
- Focus optimization on the top 1-3 bottleneck functions
- Re-profile after optimizing to verify improvement
- Premature optimization wastes time; data-driven optimization delivers results

---

## Exercises

1. Use `timeit` to compare string concatenation vs `join()`
2. Profile a slow function with `cProfile` and identify the bottleneck
3. Use `tracemalloc` to find the most memory-consuming line in a script
4. Optimize a function based on profiling results and verify the speedup

**Previous**: [Type Checking](./09_Type_Checking.md) | **Next**: [Version Control for Debugging](./11_Version_Control_for_Debugging.md)

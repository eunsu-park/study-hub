# Performance Optimization

**Previous**: [Functional Programming](./09_Functional_Programming.md) | **Next**: [Testing & Quality Assurance](./11_Testing_and_Quality.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply `timeit`, `cProfile`, and `line_profiler` to measure and identify performance bottlenecks
2. Use `memory_profiler` and `tracemalloc` to diagnose memory consumption issues
3. Select appropriate data structures (list, set, dict, tuple) based on time-complexity trade-offs
4. Implement string optimization techniques including join-based concatenation and f-strings
5. Apply loop and function optimization strategies such as local variable caching and memoization
6. Describe the GIL and choose between `multiprocessing`, `threading`, and `asyncio` for different workloads
7. Leverage NumPy vectorized operations to replace slow Python loops in numerical code
8. Write a systematic optimization checklist that balances readability with performance

---

Performance is rarely about clever tricks; it is about measuring first, then making targeted improvements where they matter most. A single poorly chosen data structure or an unnecessary allocation inside a tight loop can dominate your application's runtime, while premature micro-optimizations elsewhere waste development time with no visible benefit. This lesson equips you with the profiling tools and optimization strategies to make evidence-based decisions.

## 1. Performance Measurement Basics

Always measure before optimizing. "Don't guess, measure."

### timeit Module

```python
import timeit

# Measure code as string
time = timeit.timeit(
    'sum(range(1000))',
    number=10000
)
print(f"Execution time: {time:.4f}s")

# Measure function
def my_sum():
    return sum(range(1000))

time = timeit.timeit(my_sum, number=10000)
print(f"Execution time: {time:.4f}s")
```

### In IPython/Jupyter

```python
# Measure single line
%timeit sum(range(1000))

# Measure entire cell
%%timeit
total = 0
for i in range(1000):
    total += i
```

### Timing Decorator

```python
import time
from functools import wraps

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

@timing
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function: 1.0012s
```

---

## 2. Profiling

### cProfile

```python
import cProfile
import pstats

def expensive_function():
    total = 0
    for i in range(10000):
        total += sum(range(100))
    return total

# Profiling
profiler = cProfile.Profile()
profiler.enable()

expensive_function()

profiler.disable()

# Print results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10
```

### Command Line

```bash
# Basic profiling
python -m cProfile my_script.py

# Sort results
python -m cProfile -s cumulative my_script.py

# Save to file
python -m cProfile -o output.prof my_script.py
```

### Analyzing Profile Results

```python
import pstats

# Load saved profile
stats = pstats.Stats('output.prof')

# Sort criteria:
# 'calls': call count
# 'time': internal time
# 'cumulative': cumulative time
stats.sort_stats('cumulative')
stats.print_stats(20)

# Show specific function
stats.print_stats('my_function')
```

### line_profiler (Line-by-Line Analysis)

```bash
pip install line_profiler
```

```python
# my_script.py
@profile  # Add decorator
def slow_function():
    result = []
    for i in range(10000):
        result.append(i ** 2)
    return result

slow_function()
```

```bash
kernprof -l -v my_script.py
```

---

## 3. Memory Profiling

### memory_profiler

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def memory_hungry():
    big_list = [i for i in range(1000000)]
    del big_list
    small_list = [i for i in range(1000)]
    return small_list

memory_hungry()
```

Output:
```
Line #    Mem usage    Increment  Occurrences   Line Contents
     3     38.5 MiB     38.5 MiB           1   @profile
     4                                         def memory_hungry():
     5     76.8 MiB     38.3 MiB           1       big_list = [i for i in range(1000000)]
     6     38.5 MiB    -38.3 MiB           1       del big_list
     7     38.5 MiB      0.0 MiB           1       small_list = [i for i in range(1000)]
     8     38.5 MiB      0.0 MiB           1       return small_list
```

### sys.getsizeof()

```python
import sys

# Check object size
print(sys.getsizeof([]))         # 56 (empty list)
print(sys.getsizeof([1, 2, 3]))  # 88
print(sys.getsizeof({}))         # 64 (empty dict)
print(sys.getsizeof("hello"))    # 54

# Doesn't include nested objects
nested = [[1, 2, 3] for _ in range(10)]
print(sys.getsizeof(nested))  # Only outer list
```

### tracemalloc (Memory Tracking)

```python
import tracemalloc

tracemalloc.start()

# Memory-consuming code
big_list = [i ** 2 for i in range(100000)]

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 Memory Usage ]")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

---

## 4. Data Structure Selection

### List vs Tuple

```python
import sys

# Tuples are smaller and faster
lst = [1, 2, 3, 4, 5]
tup = (1, 2, 3, 4, 5)

print(sys.getsizeof(lst))  # 104
print(sys.getsizeof(tup))  # 80

# Creation speed
%timeit [1, 2, 3, 4, 5]  # Slower
%timeit (1, 2, 3, 4, 5)  # Faster (constant)
```

### List vs Set (Membership Testing)

```python
import timeit

data_list = list(range(10000))
data_set = set(range(10000))

# List: O(n)
print(timeit.timeit('9999 in data_list', globals=globals(), number=10000))
# ~0.8s

# Set: O(1)
print(timeit.timeit('9999 in data_set', globals=globals(), number=10000))
# ~0.001s
```

### Dictionary vs List (Search)

```python
# Search by name
users_list = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    # ... 10000 items
]

users_dict = {
    "Alice": {"age": 30},
    "Bob": {"age": 25},
    # ... 10000 items
}

# List: O(n)
def find_in_list(name):
    for user in users_list:
        if user["name"] == name:
            return user

# Dictionary: O(1)
def find_in_dict(name):
    return users_dict.get(name)
```

### Time Complexity Summary

| Operation | list | dict | set |
|-----------|------|------|-----|
| Index access | O(1) | - | - |
| Search (in) | O(n) | O(1) | O(1) |
| Insert (end) | O(1) | O(1) | O(1) |
| Insert (middle) | O(n) | - | - |
| Delete | O(n) | O(1) | O(1) |

---

## 5. String Optimization

### String Concatenation

```python
# Bad: O(n²)
result = ""
for i in range(10000):
    result += str(i)

# Good: O(n)
result = "".join(str(i) for i in range(10000))

# Better (using list)
parts = []
for i in range(10000):
    parts.append(str(i))
result = "".join(parts)
```

### f-string vs format vs %

```python
name = "Alice"
age = 30

# f-string (fastest, Python 3.6+)
s = f"Name: {name}, Age: {age}"

# format (medium)
s = "Name: {}, Age: {}".format(name, age)

# % formatting (slowest)
s = "Name: %s, Age: %d" % (name, age)
```

### String Interning

```python
# Python automatically interns short strings
a = "hello"
b = "hello"
print(a is b)  # True

# Long strings
a = "hello world" * 100
b = "hello world" * 100
print(a is b)  # False

# Manual interning
import sys
a = sys.intern("hello world" * 100)
b = sys.intern("hello world" * 100)
print(a is b)  # True
```

---

## 6. Loop Optimization

### Use Local Variables

```python
import math

# Slow: global lookup every time
def slow():
    result = []
    for i in range(10000):
        result.append(math.sqrt(i))
    return result

# Fast: cache as local variable
def fast():
    result = []
    sqrt = math.sqrt  # Local variable
    append = result.append  # Local variable
    for i in range(10000):
        append(sqrt(i))
    return result
```

### List Comprehension

```python
# for loop
result = []
for i in range(10000):
    result.append(i ** 2)

# List comprehension (faster)
result = [i ** 2 for i in range(10000)]

# map (similar speed)
result = list(map(lambda x: x ** 2, range(10000)))
```

### Remove Unnecessary Work

```python
# Bad: len() called every iteration
for i in range(len(my_list)):
    if i < len(my_list) - 1:
        pass

# Good: precompute
length = len(my_list)
for i in range(length):
    if i < length - 1:
        pass

# Better: use enumerate
for i, item in enumerate(my_list):
    pass
```

---

## 7. Function Optimization

### Memoization

```python
from functools import lru_cache

# Without memoization (slow)
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

# With memoization (fast)
@lru_cache(maxsize=None)
def fib_fast(n):
    if n < 2:
        return n
    return fib_fast(n - 1) + fib_fast(n - 2)

# fib_slow(35) → several seconds
# fib_fast(35) → instant
```

### Use Generators

```python
# Memory intensive
def get_squares_list(n):
    return [i ** 2 for i in range(n)]

# Memory efficient
def get_squares_gen(n):
    for i in range(n):
        yield i ** 2

# Usage
for square in get_squares_gen(1000000):
    if square > 100:
        break
```

### Use Built-in Functions

```python
# Custom implementation (slow)
def my_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

# Built-in (fast, implemented in C)
total = sum(numbers)

# Other examples
max_val = max(numbers)      # Maximum
min_val = min(numbers)      # Minimum
sorted_nums = sorted(numbers)  # Sorting
any_positive = any(n > 0 for n in numbers)  # Condition check
```

---

## 8. __slots__ Optimization

Reduces memory usage for class instances.

```python
import sys

# Regular class
class PointRegular:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Using __slots__
class PointSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# Memory comparison
p1 = PointRegular(1, 2)
p2 = PointSlots(1, 2)

print(sys.getsizeof(p1.__dict__))  # 104 (dict size)
# PointSlots has no __dict__

# Large difference with many instances
regular_points = [PointRegular(i, i) for i in range(100000)]
slots_points = [PointSlots(i, i) for i in range(100000)]
```

### __slots__ Cautions

```python
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)

# Cannot add dynamic attributes
# p.z = 3  # AttributeError

# Careful with inheritance
class Point3D(Point):
    __slots__ = ['z']  # Only define additional slots

    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z
```

---

## 9. Parallel Processing

### multiprocessing (CPU-bound)

```python
from multiprocessing import Pool
import time

def cpu_bound_task(n):
    """CPU-intensive task"""
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    numbers = [10000000] * 4

    # Sequential execution
    start = time.time()
    results = [cpu_bound_task(n) for n in numbers]
    print(f"Sequential: {time.time() - start:.2f}s")

    # Parallel execution
    start = time.time()
    with Pool(4) as pool:
        results = pool.map(cpu_bound_task, numbers)
    print(f"Parallel: {time.time() - start:.2f}s")
```

### concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def task(n):
    return sum(range(n))

# ProcessPoolExecutor (CPU-bound)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, [10000000] * 4))

# ThreadPoolExecutor (I/O-bound)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, n) for n in [10000000] * 4]
    results = [f.result() for f in futures]
```

### asyncio (I/O-bound)

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ["http://example.com"] * 10

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

# asyncio.run(main())
```

---

## 10. Using NumPy

Provides performance improvements for scientific computing.

### Basic Comparison

```python
import numpy as np

# Pure Python
def python_sum_squares(n):
    return sum(i ** 2 for i in range(n))

# NumPy
def numpy_sum_squares(n):
    arr = np.arange(n)
    return np.sum(arr ** 2)

# NumPy is much faster
%timeit python_sum_squares(1000000)  # ~100ms
%timeit numpy_sum_squares(1000000)   # ~2ms
```

### Vectorized Operations

```python
import numpy as np

# Python loop
def normalize_python(data):
    result = []
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    for x in data:
        result.append((x - mean) / std)
    return result

# NumPy vectorization
def normalize_numpy(data):
    arr = np.array(data)
    return (arr - arr.mean()) / arr.std()
```

---

## 11. Cython Introduction

Compile Python to C for performance.

### Simple Example

```python
# fib.pyx
def fib_py(int n):
    cdef int a = 0
    cdef int b = 1
    cdef int i
    for i in range(n):
        a, b = b, a + b
    return a
```

### Build

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib.pyx")
)
```

```bash
python setup.py build_ext --inplace
```

---

## 12. Optimization Checklist

### General Principles

1. **Measure First**: Don't guess, profile
2. **Find Bottlenecks**: 20% causes 80% of time
3. **Improve Algorithms**: Data structures and algorithms matter most
4. **Maintain Readability**: Don't over-optimize

### Quick Wins

| Item | Solution |
|------|----------|
| Membership test | list → set |
| String concat | + → join() |
| Loop | for → comprehension |
| Function calls | Cache local variables |
| Repeated calc | @lru_cache |
| Many objects | __slots__ |

### Situation-based Choices

| Situation | Solution |
|-----------|----------|
| CPU-bound | multiprocessing, NumPy, Cython |
| I/O-bound | asyncio, threading |
| Memory shortage | Generators, __slots__ |
| Repeated calc | Memoization |

---

## 13. Summary

| Tool | Purpose |
|------|---------|
| timeit | Execution time measurement |
| cProfile | Function-level profiling |
| line_profiler | Line-level profiling |
| memory_profiler | Memory profiling |
| tracemalloc | Memory tracking |

| Optimization Technique | Effect |
|----------------------|---------|
| Appropriate data structure | O(n) → O(1) |
| List comprehension | Faster than loop |
| Local variable caching | Reduce lookup cost |
| Memoization | Eliminate duplicate calculations |
| __slots__ | Save memory |
| Parallel processing | Maximize CPU utilization |

---

## 14. Practice Problems

### Exercise 1: Profiling

Profile slow code and find and improve bottlenecks.

```python
def slow_function(n):
    result = ""
    for i in range(n):
        if i in [j for j in range(i)]:
            result += str(i)
    return result
```

### Exercise 2: Memory Optimization

Write a function to efficiently process large CSV files.

### Exercise 3: Parallel Processing

Parallelize CPU-bound tasks to improve performance.

---

**Previous**: [Functional Programming](./09_Functional_Programming.md) | **Next**: [Testing & Quality Assurance](./11_Testing_and_Quality.md)

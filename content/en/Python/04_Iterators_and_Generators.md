# Iterators & Generators

**Previous**: [Context Managers](./03_Context_Managers.md) | **Next**: [Closures & Scope](./05_Closures_and_Scope.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between iterables and iterators and explain the `__iter__` / `__next__` protocol
2. Trace how a `for` loop internally uses `iter()` and `next()` with `StopIteration`
3. Implement custom iterator classes with separate iterable and iterator objects for reusability
4. Write generator functions using `yield` and explain their lazy evaluation behavior
5. Compare generator expressions with list comprehensions in terms of memory efficiency
6. Apply `yield from` for delegation and recursive generators (e.g., flattening nested structures)
7. Use advanced generator features: `send()`, `throw()`, and `close()`
8. Apply `itertools` functions (`chain`, `groupby`, `islice`, `product`, `combinations`) to build efficient data pipelines

---

Processing a million-row CSV file should not require loading a million rows into memory. Iterators and generators are Python's answer to this problem: they produce values one at a time, on demand, keeping memory consumption constant regardless of data size. From reading log files line-by-line to chaining transformation stages in a data pipeline, the iterator protocol underpins much of Python's elegance and efficiency in handling sequences.

## 1. Iterables and Iterators

### Concept Distinction

| Term | Description | Examples |
|------|-------------|----------|
| Iterable | Object with `__iter__` method | list, str, dict, set |
| Iterator | Object with `__iter__` and `__next__` methods | iter(list), file objects |

```
┌──────────────────────────────────────────┐
│              Iterable                     │
│  ┌────────────────────────────────────┐  │
│  │    Iterator                         │  │
│  │    __iter__() → self               │  │
│  │    __next__() → next value or StopIteration │
│  └────────────────────────────────────┘  │
│  __iter__() → Returns Iterator           │
└──────────────────────────────────────────┘
```

### How for Loops Work

```python
# for item in iterable:
#     ...

# The above code is equivalent to:
iterator = iter(iterable)  # Calls __iter__()
while True:
    try:
        item = next(iterator)  # Calls __next__()
        # Execute loop body
    except StopIteration:
        break
```

### Example

```python
numbers = [1, 2, 3]

# Create iterator with iter()
iterator = iter(numbers)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration exception!
```

---

## 2. Custom Iterators

### Implementing __iter__ and __next__

```python
class Counter:
    """Iterator that counts from 1 to max"""

    def __init__(self, max_count):
        self.max_count = max_count
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current > self.max_count:
            raise StopIteration
        return self.current

# Usage
for num in Counter(5):
    print(num, end=" ")  # 1 2 3 4 5
```

### Separating Iterable and Iterator

Separate them to create reusable iterables.

```python
class Range:
    """Reusable range"""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return RangeIterator(self.start, self.end)

class RangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can iterate multiple times
r = Range(1, 4)
print(list(r))  # [1, 2, 3]
print(list(r))  # [1, 2, 3] (can be used again)
```

---

## 3. Generator Functions

Using the `yield` keyword makes it easy to create iterators.

### Basic Syntax

```python
def count_up_to(max_count):
    count = 1
    while count <= max_count:
        yield count  # Return value and pause
        count += 1

# Usage
for num in count_up_to(5):
    print(num, end=" ")  # 1 2 3 4 5

# Or
gen = count_up_to(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
```

### How It Works

```
count_up_to(3) called
    │
    ▼
┌─────────────────┐
│  count = 1      │
│  while True:    │
│    yield 1 ─────┼──▶ Return, pause
│                 │
│  (next called)  │
│    count = 2    │
│    yield 2 ─────┼──▶ Return, pause
│                 │
│  (next called)  │
│    count = 3    │
│    yield 3 ─────┼──▶ Return, pause
│                 │
│  (next called)  │
│    while exits  │
│  StopIteration  │
└─────────────────┘
```

### Multiple yield Values

```python
def multi_yield():
    yield "First"
    yield "Second"
    yield "Third"

for value in multi_yield():
    print(value)
```

---

## 4. Generator Expressions

Similar to list comprehensions but use parentheses.

```python
# List comprehension - stores everything in memory
squares_list = [x**2 for x in range(10)]

# Generator expression - generates on demand
squares_gen = (x**2 for x in range(10))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# Generators can only be iterated once
print(list(squares_gen))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(list(squares_gen))  # [] (already exhausted)
```

### Memory Efficiency

```python
import sys

# List: uses full memory
list_comp = [x for x in range(1000000)]
print(sys.getsizeof(list_comp))  # ~8MB

# Generator: minimal memory
gen_exp = (x for x in range(1000000))
print(sys.getsizeof(gen_exp))    # ~200 bytes
```

---

## 5. yield from

Delegates values from another iterable.

```python
def chain(*iterables):
    for it in iterables:
        yield from it  # Same as: for item in it: yield item

result = list(chain([1, 2], [3, 4], [5, 6]))
print(result)  # [1, 2, 3, 4, 5, 6]
```

### Recursive Generator

```python
def flatten(nested):
    """Flatten nested list"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6, [7]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7]
```

---

## 6. Advanced Generator Features

### send() - Send Values

You can send values to generators.

```python
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

gen = accumulator()
print(next(gen))      # 0 (initialize)
print(gen.send(10))   # 10
print(gen.send(20))   # 30
print(gen.send(5))    # 35
```

### throw() - Send Exceptions

```python
def generator():
    try:
        yield 1
        yield 2
        yield 3
    except ValueError as e:
        yield f"Exception handled: {e}"

gen = generator()
print(next(gen))              # 1
print(gen.throw(ValueError, "Test"))  # Exception handled: Test
```

### close() - Terminate Generator

```python
def generator():
    try:
        yield 1
        yield 2
        yield 3
    finally:
        print("Cleanup")

gen = generator()
print(next(gen))  # 1
gen.close()       # Outputs "Cleanup"
```

---

## 7. itertools Module

Provides efficient iterator tools.

### Infinite Iterators

```python
from itertools import count, cycle, repeat

# count: infinite counter
for i in count(10, 2):  # Start at 10, increment by 2
    if i > 20:
        break
    print(i, end=" ")  # 10 12 14 16 18 20

# cycle: infinite repetition
colors = cycle(["red", "blue", "green"])
for _ in range(5):
    print(next(colors), end=" ")  # red blue green red blue

# repeat: repetition
for item in repeat("Hello", 3):
    print(item)  # Hello Hello Hello
```

### Combinatoric Iterators

```python
from itertools import chain, zip_longest, product, permutations, combinations

# chain: connect multiple iterables
print(list(chain([1, 2], [3, 4])))  # [1, 2, 3, 4]

# zip_longest: zip iterables of different lengths
a = [1, 2, 3]
b = ["a", "b"]
print(list(zip_longest(a, b, fillvalue="-")))
# [(1, 'a'), (2, 'b'), (3, '-')]

# product: Cartesian product
print(list(product("AB", [1, 2])))
# [('A', 1), ('A', 2), ('B', 1), ('B', 2)]

# permutations: permutations
print(list(permutations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# combinations: combinations
print(list(combinations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]
```

### Filtering Iterators

```python
from itertools import takewhile, dropwhile, filterfalse, compress

numbers = [1, 3, 5, 2, 4, 6]

# takewhile: while condition is true
print(list(takewhile(lambda x: x < 5, numbers)))  # [1, 3]

# dropwhile: skip while condition is true
print(list(dropwhile(lambda x: x < 5, numbers)))  # [5, 2, 4, 6]

# filterfalse: only items where condition is false
print(list(filterfalse(lambda x: x % 2, numbers)))  # [2, 4, 6]

# compress: filter by selector
data = ["A", "B", "C", "D"]
selectors = [1, 0, 1, 0]
print(list(compress(data, selectors)))  # ['A', 'C']
```

### Grouping

```python
from itertools import groupby

data = [
    {"name": "Alice", "dept": "HR"},
    {"name": "Bob", "dept": "IT"},
    {"name": "Charlie", "dept": "HR"},
    {"name": "David", "dept": "IT"},
]

# Must sort first!
data.sort(key=lambda x: x["dept"])

for dept, group in groupby(data, key=lambda x: x["dept"]):
    print(f"{dept}: {[p['name'] for p in group]}")
# HR: ['Alice', 'Charlie']
# IT: ['Bob', 'David']
```

### Slicing

```python
from itertools import islice

# Extract portion from infinite iterator
from itertools import count

first_10 = list(islice(count(1), 10))
print(first_10)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Specify start, stop, step
result = list(islice(range(100), 10, 20, 2))
print(result)  # [10, 12, 14, 16, 18]
```

---

## 8. Lazy Evaluation

Generators don't pre-compute values but generate them when needed.

### Large File Processing

```python
def read_large_file(filepath):
    """Generator that reads line by line"""
    with open(filepath, "r") as f:
        for line in f:
            yield line.strip()

# Memory-efficient processing
for line in read_large_file("huge_file.txt"):
    if "ERROR" in line:
        print(line)
```

### Pipeline Processing

```python
def numbers():
    for i in range(1, 1000001):
        yield i

def even_only(nums):
    for n in nums:
        if n % 2 == 0:
            yield n

def squared(nums):
    for n in nums:
        yield n ** 2

def less_than(nums, limit):
    for n in nums:
        if n >= limit:
            break
        yield n

# Pipeline: memory efficient
pipeline = less_than(squared(even_only(numbers())), 100)
print(list(pipeline))  # [4, 16, 36, 64]
```

---

## 9. Infinite Sequences

### Fibonacci Sequence

```python
def fibonacci():
    """Infinite Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# First 10
from itertools import islice
print(list(islice(fibonacci(), 10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Prime Generator

```python
def primes():
    """Infinite prime generation"""
    yield 2
    candidate = 3
    found = [2]
    while True:
        if all(candidate % p != 0 for p in found):
            found.append(candidate)
            yield candidate
        candidate += 2

# First 10 primes
from itertools import islice
print(list(islice(primes(), 10)))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

---

## 10. Summary

| Concept | Description |
|---------|-------------|
| Iterable | Implements `__iter__`, usable in for loops |
| Iterator | Implements `__iter__` + `__next__` |
| Generator | Function using `yield` |
| Generator Expression | `(x for x in iterable)` |
| `yield from` | Delegate to another iterable |
| `send()` | Send value to generator |
| Lazy Evaluation | Generate values when needed |

---

## 11. Practice Problems

### Exercise 1: Chunk Division

Create a generator that divides a list into chunks of specified size.

```python
# chunk([1,2,3,4,5], 2) → [1,2], [3,4], [5]
```

### Exercise 2: Sliding Window

Create a generator that produces sliding windows.

```python
# sliding_window([1,2,3,4,5], 3) → (1,2,3), (2,3,4), (3,4,5)
```

### Exercise 3: Tree Traversal

Create a generator that traverses a binary tree.

---

**Previous**: [Context Managers](./03_Context_Managers.md) | **Next**: [Closures & Scope](./05_Closures_and_Scope.md)

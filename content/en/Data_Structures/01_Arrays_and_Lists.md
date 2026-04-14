# Arrays and Lists

**Next**: [Linked Lists](./02_Linked_Lists.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between static arrays and dynamic arrays
2. Describe how Python's `list` is implemented as a dynamic array internally
3. Perform indexing, slicing, and common list operations efficiently
4. Analyze the time complexity of list operations (append, insert, delete, access)
5. Understand amortized O(1) analysis for dynamic array resizing
6. Use list comprehensions to create and transform lists concisely
7. Identify when arrays/lists are the right choice versus other data structures

---

An **array** is the simplest and most fundamental data structure in computer science. It stores elements in contiguous memory locations, allowing constant-time access by index. Understanding arrays deeply is essential because nearly every other data structure is built on top of or compared against them.

## What Is an Array?

An array is a fixed-size, ordered collection of elements stored in a contiguous block of memory. Each element occupies the same amount of space, and you can access any element directly using its index.

```
Index:    0     1     2     3     4
        +-----+-----+-----+-----+-----+
Array:  | 10  | 20  | 30  | 40  | 50  |
        +-----+-----+-----+-----+-----+
Address: 1000  1008  1016  1024  1032
         base  +8    +16   +24   +32

Address of element i = base_address + i * element_size
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Contiguous memory** | Elements are stored next to each other in RAM |
| **Fixed size** (static) | Size is determined at creation and cannot change |
| **Homogeneous** (traditional) | All elements share the same type |
| **Random access** | Any element accessible in O(1) via index arithmetic |

### Why Contiguous Memory Matters

Modern CPUs use a **cache hierarchy** (L1, L2, L3) to speed up memory access. When you access one element of an array, the CPU loads a whole **cache line** (typically 64 bytes) into the cache. Since array elements are contiguous, the next elements are likely already in the cache. This is called **spatial locality**, and it makes array traversal extremely fast in practice.

```
CPU Cache Line (64 bytes):
+------+------+------+------+------+------+------+------+
| a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7] |
+------+------+------+------+------+------+------+------+
   All loaded in one memory fetch -- subsequent accesses are cache hits!
```

## Static Arrays in Python: `array` Module

Python does not have C-style static arrays, but the `array` module provides typed arrays that store elements more compactly than lists:

```python
import array

# Create an array of signed integers
nums = array.array('i', [10, 20, 30, 40, 50])

# Access by index -- O(1)
print(nums[2])  # 30

# Arrays are typed -- this would raise TypeError:
# nums.append("hello")

# Supported type codes:
# 'b' signed char    'B' unsigned char
# 'h' signed short   'H' unsigned short
# 'i' signed int     'I' unsigned int
# 'l' signed long    'L' unsigned long
# 'f' float          'd' double
```

## Dynamic Arrays: Python's `list`

Python's `list` is a **dynamic array** -- it automatically resizes when more space is needed. Under the hood, a list maintains:

1. A pointer to a contiguous block of memory holding **references** to objects
2. The current **length** (number of elements)
3. The current **capacity** (allocated slots)

```
Python list internals:
+------------------+
| length: 5        |
| capacity: 8      |
| data_ptr: ------>+--+-----+-----+-----+-----+-----+-----+-----+-----+
+------------------+  | ptr | ptr | ptr | ptr | ptr |     |     |     |
                      +--+--+--+--+--+--+--+--+--+--+-----+-----+-----+
                         |     |     |     |     |
                         v     v     v     v     v
                        10    20    30    40    50
                                          (3 empty slots for future growth)
```

### How Dynamic Resizing Works

When you `append()` to a list that is full (length == capacity), Python:

1. Allocates a **new, larger** block of memory (roughly 1.125x the old capacity)
2. Copies all existing references to the new block
3. Frees the old block
4. Inserts the new element

```
Before append (full):     After resize + append:
capacity = 4              capacity = 8
length = 4                length = 5

+---+---+---+---+         +---+---+---+---+---+---+---+---+
| A | B | C | D |         | A | B | C | D | E |   |   |   |
+---+---+---+---+         +---+---+---+---+---+---+---+---+
```

### Amortized Analysis

Individual resize operations are O(n) because they copy all elements. But since resizes happen infrequently (the capacity grows geometrically), the **amortized cost** of `append()` is **O(1)**.

The intuition: if we start with capacity 1 and double each time, after n appends we perform copies at sizes 1, 2, 4, 8, ..., n. Total copies = 1 + 2 + 4 + ... + n = 2n - 1 = O(n). Spread across n operations, each append costs O(n)/n = O(1) amortized.

## Time Complexity of List Operations

| Operation | Average Case | Worst Case | Notes |
|-----------|-------------|------------|-------|
| `lst[i]` (access) | O(1) | O(1) | Direct index arithmetic |
| `lst[i] = x` (assign) | O(1) | O(1) | Direct index arithmetic |
| `lst.append(x)` | O(1)* | O(n) | Amortized O(1) |
| `lst.insert(i, x)` | O(n) | O(n) | Shifts elements right |
| `lst.pop()` | O(1) | O(1) | Remove from end |
| `lst.pop(i)` | O(n) | O(n) | Shifts elements left |
| `del lst[i]` | O(n) | O(n) | Shifts elements left |
| `x in lst` | O(n) | O(n) | Linear scan |
| `lst.index(x)` | O(n) | O(n) | Linear scan |
| `len(lst)` | O(1) | O(1) | Stored as attribute |
| `lst.sort()` | O(n log n) | O(n log n) | Timsort |
| `lst + lst2` | O(n+m) | O(n+m) | Creates new list |
| `lst[a:b]` (slice) | O(b-a) | O(b-a) | Creates new list |

*Amortized

## Indexing and Slicing

Python supports powerful indexing with negative indices and slicing:

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# Positive indexing (0-based)
fruits[0]    # "apple"
fruits[2]    # "cherry"

# Negative indexing (from the end)
fruits[-1]   # "elderberry"
fruits[-2]   # "date"

# Slicing: lst[start:stop:step]
fruits[1:3]    # ["banana", "cherry"]       -- start inclusive, stop exclusive
fruits[:3]     # ["apple", "banana", "cherry"]  -- from beginning
fruits[2:]     # ["cherry", "date", "elderberry"]  -- to end
fruits[::2]    # ["apple", "cherry", "elderberry"] -- every 2nd element
fruits[::-1]   # reversed list

# Slice assignment (modifies in-place)
fruits[1:3] = ["blueberry"]  # replaces 2 elements with 1
# ["apple", "blueberry", "date", "elderberry"]
```

### How Negative Indexing Works

```
Positive:  0       1        2       3       4
         +-------+--------+-------+------+-----------+
         | apple | banana | cherry| date | elderberry|
         +-------+--------+-------+------+-----------+
Negative: -5      -4       -3      -2     -1

lst[-k] is equivalent to lst[len(lst) - k]
```

## Common List Patterns

### Building Lists Efficiently

```python
# BAD: String concatenation in a loop -- O(n^2)
result = ""
for i in range(10000):
    result += str(i)  # Creates a new string each time!

# GOOD: Join a list -- O(n)
parts = []
for i in range(10000):
    parts.append(str(i))
result = "".join(parts)

# BEST: List comprehension + join -- O(n)
result = "".join(str(i) for i in range(10000))
```

### List Comprehensions

List comprehensions provide a concise way to create lists:

```python
# Basic comprehension
squares = [x ** 2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition (filter)
evens = [x for x in range(20) if x % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Nested comprehension (matrix)
matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
# [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Flattening a matrix
flat = [x for row in matrix for x in row]
# [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

### Two-Pointer Technique

A common pattern for solving array problems efficiently:

```python
def two_sum_sorted(nums, target):
    """Find two numbers in a sorted list that sum to target.
    
    Uses two pointers -- O(n) time, O(1) space.
    """
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None
```

### Sliding Window

Another powerful array technique:

```python
def max_sum_subarray(nums, k):
    """Find the maximum sum of any subarray of length k.
    
    Sliding window -- O(n) time, O(1) space.
    """
    if len(nums) < k:
        return None
    
    # Compute sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # Add new, remove old
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## Multi-Dimensional Arrays

### 2D Lists (Matrices)

```python
# Creating a 3x4 matrix
rows, cols = 3, 4
matrix = [[0] * cols for _ in range(rows)]

# WARNING: This creates shared references!
# bad_matrix = [[0] * cols] * rows  # All rows point to same list!

# Accessing elements
matrix[1][2] = 42  # Row 1, Column 2

# Traversing a matrix
for i in range(rows):
    for j in range(cols):
        print(f"matrix[{i}][{j}] = {matrix[i][j]}")

# Matrix as ASCII art
# matrix[0]: [0, 0, 0, 0]
# matrix[1]: [0, 0, 42, 0]
# matrix[2]: [0, 0, 0, 0]
```

## Python's `list` vs `array` vs `numpy.ndarray`

| Feature | `list` | `array.array` | `numpy.ndarray` |
|---------|--------|---------------|-----------------|
| Element types | Mixed | Homogeneous | Homogeneous |
| Memory | Higher (object refs) | Lower (raw values) | Lowest |
| Math operations | Manual loops | Manual loops | Vectorized |
| Resizable | Yes | Yes | Fixed (view-based) |
| Use case | General purpose | Simple typed arrays | Numerical computing |

## When to Use Arrays/Lists

**Use lists when:**
- You need random access by index
- Most operations are at the end (append/pop)
- The data size is moderate
- You need mixed types

**Avoid lists when:**
- You frequently insert/delete at the beginning or middle (use `deque`)
- You need fast membership testing (use `set`)
- You need key-value associations (use `dict`)
- You need sorted order with fast insert (use a balanced BST or `sortedcontainers`)

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Static array | Fixed size, contiguous, O(1) access |
| Dynamic array | Resizable, amortized O(1) append |
| Python `list` | Dynamic array of object references |
| Indexing | O(1) access, supports negative indices |
| Slicing | Creates new list, O(k) for k elements |
| Insert/delete middle | O(n) due to shifting |
| Cache friendliness | Contiguous memory = fast traversal |

---

**Next**: [Linked Lists](./02_Linked_Lists.md) -- Learn about non-contiguous data structures where insertions and deletions are O(1).

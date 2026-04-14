# Sets and Maps

**Previous**: [Graphs Basics](./09_Graphs_Basics.md) | **Next**: [Strings as Data Structures](./11_Strings_as_Data_Structures.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the mathematical foundations of sets and their operations
2. Implement a set using a hash table
3. Perform union, intersection, difference, and symmetric difference
4. Understand the map (dictionary) abstraction and its implementations
5. Use Python's `set`, `frozenset`, `dict`, `defaultdict`, and `Counter`
6. Apply sets and maps to solve deduplication, counting, and grouping problems
7. Analyze time complexity of set and map operations

---

**Sets** and **maps** are two of the most heavily used data structures in practical programming. A **set** is an unordered collection of unique elements. A **map** (also called a dictionary or associative array) stores key-value pairs with unique keys.

## Sets

### Mathematical Definition

A set is a collection of distinct elements with no defined order:
- `S = {1, 2, 3}` -- a set of three integers
- `1 in S` is True, `4 in S` is False
- `{1, 2, 3} == {3, 1, 2}` -- order does not matter
- `{1, 1, 2}` is the same as `{1, 2}` -- no duplicates

### Set Operations

```
A = {1, 2, 3, 4}        B = {3, 4, 5, 6}

Union (A | B):           {1, 2, 3, 4, 5, 6}
  +-------+-------+
  | 1 2   | 3 4   |  5 6  -- everything in A or B
  +-------+-------+

Intersection (A & B):    {3, 4}
            +-----+
  1 2  |    | 3 4 |    5 6  -- only in both A and B
            +-----+

Difference (A - B):      {1, 2}
  +-------+
  | 1 2   |  3 4   5 6     -- in A but not in B
  +-------+

Symmetric Diff (A ^ B):  {1, 2, 5, 6}
  +-----+       +-----+
  | 1 2 |  3 4  | 5 6 |    -- in A or B but not both
  +-----+       +-----+

Subset (A <= B):         False
Superset (A >= B):       False
```

### Python `set` Usage

```python
# Creation
s = {1, 2, 3}
s = set([1, 2, 2, 3])  # {1, 2, 3} -- duplicates removed
empty = set()           # {} creates a dict, not a set!

# Membership testing -- O(1) average
3 in s      # True
4 not in s  # True

# Adding and removing
s.add(4)         # {1, 2, 3, 4}
s.discard(2)     # {1, 3, 4}  -- no error if absent
s.remove(3)      # {1, 4}     -- raises KeyError if absent
s.pop()          # removes arbitrary element

# Set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b   # Union:        {1, 2, 3, 4, 5, 6}
a & b   # Intersection: {3, 4}
a - b   # Difference:   {1, 2}
a ^ b   # Sym. diff:    {1, 2, 5, 6}

a <= b  # Subset:   False
a >= b  # Superset: False
a.isdisjoint(b)  # False (they share 3, 4)

# Set comprehension
squares = {x ** 2 for x in range(10)}
# {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
```

### Implementing a Set

A set is essentially a hash table that stores only keys (no values):

```python
class HashSet:
    """Set implementation using a hash table."""
    
    def __init__(self, capacity=8):
        self._buckets = [[] for _ in range(capacity)]
        self._size = 0
        self._capacity = capacity
    
    def add(self, item):
        """Add an item -- O(1) average."""
        if item in self:
            return
        idx = hash(item) % self._capacity
        self._buckets[idx].append(item)
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._resize(self._capacity * 2)
    
    def remove(self, item):
        """Remove an item -- O(1) average."""
        idx = hash(item) % self._capacity
        bucket = self._buckets[idx]
        for i, val in enumerate(bucket):
            if val == item:
                bucket.pop(i)
                self._size -= 1
                return
        raise KeyError(item)
    
    def __contains__(self, item):
        idx = hash(item) % self._capacity
        return any(val == item for val in self._buckets[idx])
    
    def __len__(self):
        return self._size
    
    def union(self, other):
        result = HashSet()
        for item in self:
            result.add(item)
        for item in other:
            result.add(item)
        return result
    
    def intersection(self, other):
        result = HashSet()
        for item in self:
            if item in other:
                result.add(item)
        return result
    
    def __iter__(self):
        for bucket in self._buckets:
            for item in bucket:
                yield item
    
    def _resize(self, new_capacity):
        old_buckets = self._buckets
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0
        for bucket in old_buckets:
            for item in bucket:
                self.add(item)
```

### `frozenset` -- Immutable Set

```python
# frozenset is hashable, can be used as dict key or set element
fs = frozenset([1, 2, 3])

# Sets of sets
power_set = {frozenset(), frozenset([1]), frozenset([2]), 
             frozenset([1, 2])}
```

## Maps (Dictionaries)

### The Map ADT

A map stores key-value pairs, where each key is unique:

| Operation | Description | Average Time |
|-----------|-------------|-------------|
| `put(key, val)` | Insert or update | O(1) |
| `get(key)` | Retrieve value by key | O(1) |
| `delete(key)` | Remove key-value pair | O(1) |
| `contains(key)` | Check if key exists | O(1) |
| `keys()` | Iterate all keys | O(n) |
| `values()` | Iterate all values | O(n) |
| `items()` | Iterate key-value pairs | O(n) |

### Python `dict` -- A Map

```python
# Creation
d = {"name": "Alice", "age": 30}
d = dict(name="Alice", age=30)

# Access
d["name"]             # "Alice"
d.get("name")         # "Alice"
d.get("height", 0)    # 0 (default if missing)

# Modification
d["age"] = 31
d.setdefault("city", "NYC")  # Set only if key absent
d.update({"age": 32, "job": "engineer"})

# Deletion
del d["job"]
d.pop("city")         # Returns value and removes key

# Iteration
for key in d:            pass  # Keys
for val in d.values():   pass  # Values
for k, v in d.items():   pass  # Key-value pairs

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### `defaultdict`

Automatically creates missing keys:

```python
from collections import defaultdict

# Group words by first letter
words = ["apple", "banana", "avocado", "blueberry", "cherry"]
groups = defaultdict(list)
for word in words:
    groups[word[0]].append(word)
# {'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}

# Count occurrences
counts = defaultdict(int)
for word in words:
    counts[word] += 1
```

### `Counter`

Specialized dictionary for counting:

```python
from collections import Counter

text = "abracadabra"
counter = Counter(text)
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

counter.most_common(2)   # [('a', 5), ('b', 2)]
counter['a']             # 5
counter['z']             # 0 (no KeyError!)

# Arithmetic on Counters
c1 = Counter("aabbcc")
c2 = Counter("abcdef")
c1 + c2  # Counter({'a': 3, 'b': 3, 'c': 3, 'd': 1, 'e': 1, 'f': 1})
c1 - c2  # Counter({'a': 1, 'b': 1, 'c': 1})
```

### `OrderedDict`

Remembers insertion order (regular `dict` does too since Python 3.7, but `OrderedDict` provides `move_to_end()`):

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache using OrderedDict."""
    
    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = capacity
    
    def get(self, key):
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)  # Remove oldest
```

## Practical Applications

### Two Sum (Hash Map)

```python
def two_sum(nums, target):
    """Find two indices whose values sum to target -- O(n).
    
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### Finding Duplicates (Set)

```python
def find_duplicates(nums):
    """Find all duplicate values -- O(n).
    
    >>> find_duplicates([1, 2, 3, 2, 4, 3])
    {2, 3}
    """
    seen = set()
    duplicates = set()
    for num in nums:
        if num in seen:
            duplicates.add(num)
        seen.add(num)
    return duplicates
```

### Group Anagrams (Hash Map)

```python
def group_anagrams(words):
    """Group words that are anagrams of each other.
    
    >>> group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
    """
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())
```

## Time Complexity Summary

| Operation | `set` | `dict` | Sorted set (BST) |
|-----------|-------|--------|-------------------|
| Add/Insert | O(1) | O(1) | O(log n) |
| Remove/Delete | O(1) | O(1) | O(log n) |
| Lookup/Contains | O(1) | O(1) | O(log n) |
| Min/Max | O(n) | O(n) | O(log n) |
| Iteration | O(n) | O(n) | O(n) |
| Union | O(n+m) | -- | O(n+m) |
| Intersection | O(min(n,m)) | -- | O(n log m) |

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Set | Unordered, unique elements, O(1) lookup |
| Set operations | Union, intersection, difference, symmetric difference |
| frozenset | Immutable, hashable set |
| Map/dict | Key-value pairs, O(1) operations |
| defaultdict | Auto-creates missing keys with default factory |
| Counter | Purpose-built counting dict |
| OrderedDict | move_to_end() for LRU caches |
| Common patterns | Two-sum, dedup, grouping, counting |

---

**Next**: [Strings as Data Structures](./11_Strings_as_Data_Structures.md) -- Explore specialized structures for string processing.

# Hash Tables

**Previous**: [Queues](./04_Queues.md) | **Next**: [Trees Basics](./06_Trees_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of hashing and hash functions
2. Implement a hash table from scratch with collision resolution
3. Compare chaining and open addressing strategies
4. Analyze average and worst-case time complexity of hash table operations
5. Understand how Python's `dict` works internally
6. Design effective hash functions for custom objects
7. Identify when hash tables are the optimal choice
8. Understand load factor and its role in performance

---

A **hash table** (also called a hash map) is a data structure that maps keys to values using a **hash function**. It provides average-case O(1) lookups, insertions, and deletions, making it one of the most practical data structures in computing.

## The Big Idea

```
Key: "alice" ──> hash("alice") = 42 ──> 42 % 8 = 2 ──> table[2] = "alice's data"

         Hash Function        Modulo         Store
  Key ──────────────> Number ──────> Index ──────> Bucket
```

```
Hash Table (size 8):
Index:  0     1     2          3     4     5     6     7
      +-----+-----+----------+-----+-----+-----+-----+-----+
      |     |     | "alice"  |     |     | "bob"|     |     |
      |     |     | -> data  |     |     | ->dat|     |     |
      +-----+-----+----------+-----+-----+-----+-----+-----+
```

## Hash Functions

A hash function converts a key of any type into an integer. Good hash functions have these properties:

1. **Deterministic**: Same input always produces the same output
2. **Uniform distribution**: Spreads keys evenly across buckets
3. **Fast to compute**: O(1) or O(len(key))
4. **Minimizes collisions**: Different keys rarely map to the same index

### Python's `hash()` Function

```python
# Immutable types are hashable
hash(42)          # 42
hash("hello")     # Varies by session (randomized for security)
hash((1, 2, 3))   # Tuple of hashables is hashable
hash(frozenset({1, 2}))  # frozenset is hashable

# Mutable types are NOT hashable
# hash([1, 2, 3])    # TypeError: unhashable type: 'list'
# hash({1: 2})       # TypeError: unhashable type: 'dict'
# hash({1, 2, 3})    # TypeError: unhashable type: 'set'
```

### Building a Simple Hash Function

```python
def simple_hash(key, table_size):
    """A basic hash function for strings."""
    hash_value = 0
    for char in str(key):
        hash_value = (hash_value * 31 + ord(char)) % table_size
    return hash_value

# Why 31? It's a small prime that produces good distribution.
# Many languages (Java, etc.) use 31 for string hashing.
```

## Collision Resolution

A **collision** occurs when two different keys hash to the same index. There are two main strategies:

### Strategy 1: Separate Chaining

Each bucket contains a linked list (or other collection) of key-value pairs:

```
Index 0: -> None
Index 1: -> ("bob", 25) -> None
Index 2: -> ("alice", 30) -> ("charlie", 35) -> None  (collision!)
Index 3: -> None
Index 4: -> ("dave", 28) -> None
```

```python
class HashTableChaining:
    """Hash table with separate chaining."""
    
    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._buckets = [[] for _ in range(capacity)]
    
    def _hash(self, key):
        return hash(key) % self._capacity
    
    def put(self, key, value):
        """Insert or update a key-value pair -- O(1) average."""
        idx = self._hash(key)
        bucket = self._buckets[idx]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update
                return
        
        bucket.append((key, value))  # Insert
        self._size += 1
        
        # Resize if load factor exceeds threshold
        if self._size / self._capacity > 0.75:
            self._resize(self._capacity * 2)
    
    def get(self, key):
        """Retrieve a value by key -- O(1) average."""
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)
    
    def delete(self, key):
        """Remove a key-value pair -- O(1) average."""
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return
        raise KeyError(key)
    
    def _resize(self, new_capacity):
        """Rehash all entries into a larger table."""
        old_buckets = self._buckets
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def __len__(self):
        return self._size
```

### Strategy 2: Open Addressing (Linear Probing)

When a collision occurs, search for the next available slot:

```
Insert "alice" -> hash = 2, table[2] is empty, place here
Insert "charlie" -> hash = 2, collision! Try 3... empty, place here

Index:  0     1     2          3            4     5
      +-----+-----+----------+------------+-----+-----+
      |     |     | "alice"  | "charlie"  |     |     |
      +-----+-----+----------+------------+-----+-----+
                     ^           ^
                   hash=2     probe to 3
```

```python
class HashTableOpenAddr:
    """Hash table with linear probing."""
    
    _DELETED = object()  # Sentinel for deleted slots
    
    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._keys = [None] * capacity
        self._values = [None] * capacity
    
    def _hash(self, key):
        return hash(key) % self._capacity
    
    def _probe(self, key):
        """Find the index for a key using linear probing."""
        idx = self._hash(key)
        first_deleted = None
        
        for _ in range(self._capacity):
            if self._keys[idx] is None:
                return first_deleted if first_deleted is not None else idx
            if self._keys[idx] is self._DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif self._keys[idx] == key:
                return idx
            idx = (idx + 1) % self._capacity
        
        return first_deleted
    
    def put(self, key, value):
        """Insert or update -- O(1) average."""
        if self._size / self._capacity > 0.5:
            self._resize(self._capacity * 2)
        
        idx = self._probe(key)
        if self._keys[idx] != key:
            self._size += 1
        self._keys[idx] = key
        self._values[idx] = value
    
    def get(self, key):
        """Retrieve -- O(1) average."""
        idx = self._probe(key)
        if self._keys[idx] == key:
            return self._values[idx]
        raise KeyError(key)
    
    def delete(self, key):
        """Delete using tombstone -- O(1) average."""
        idx = self._probe(key)
        if self._keys[idx] == key:
            self._keys[idx] = self._DELETED
            self._values[idx] = None
            self._size -= 1
        else:
            raise KeyError(key)
    
    def _resize(self, new_capacity):
        old_keys = self._keys
        old_values = self._values
        self._capacity = new_capacity
        self._keys = [None] * new_capacity
        self._values = [None] * new_capacity
        self._size = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self._DELETED:
                self.put(k, v)
```

## Load Factor

The **load factor** (alpha) = number of entries / table capacity.

```
Load factor = n / m

  alpha = 0.0    Empty table, no collisions, wasted memory
  alpha = 0.5    Good balance (open addressing sweet spot)
  alpha = 0.75   Good balance (chaining sweet spot, Python's threshold)
  alpha = 1.0    Full table, many collisions
  alpha > 1.0    Possible with chaining, guaranteed collisions
```

When the load factor exceeds a threshold, the table is **resized** (typically doubled) and all entries are rehashed.

## How Python's `dict` Works

Python's `dict` uses **open addressing with a perturbation-based probing** strategy:

1. **Hash computation**: `hash(key)` returns a 64-bit integer
2. **Index mapping**: `idx = hash(key) & (table_size - 1)` (bitwise AND, table size is power of 2)
3. **Probing**: If slot is occupied, perturb the index: `idx = (5 * idx + perturb + 1) % size`
4. **Load factor**: Resizes when 2/3 full
5. **Compact dict** (Python 3.7+): Maintains insertion order using a separate indices array

```
Python dict internal layout (simplified):

Indices array:    [None, 0, None, 1, 2, None, None, None]
                           ^           ^    ^
                         "alice"     "bob"  "charlie"

Entries array:    [("alice", 30), ("bob", 25), ("charlie", 35)]
                     entry 0       entry 1       entry 2
```

This compact layout saves memory and preserves insertion order.

## Time Complexity

| Operation | Average | Worst Case | Notes |
|-----------|---------|------------|-------|
| `put(key, val)` | **O(1)** | O(n) | Worst case: all keys collide |
| `get(key)` | **O(1)** | O(n) | Worst case: long probe chain |
| `delete(key)` | **O(1)** | O(n) | Worst case: long probe chain |
| `key in table` | **O(1)** | O(n) | Same as get |
| Resize | O(n) | O(n) | Amortized into put operations |

**Why O(n) worst case?** If all keys hash to the same index (pathological input), the hash table degenerates to a linked list. This is extremely rare with good hash functions.

## Hashing Custom Objects

To use custom objects as dict keys, implement `__hash__` and `__eq__`:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

# Now you can use Point as a dict key
distances = {Point(0, 0): 0.0, Point(3, 4): 5.0}
```

**Rules:**
- If `a == b`, then `hash(a) == hash(b)` (required)
- If `hash(a) == hash(b)`, `a == b` is NOT guaranteed (collisions are fine)
- Mutable objects should not be used as keys (hash could change)

## Chaining vs Open Addressing

| Aspect | Chaining | Open Addressing |
|--------|----------|-----------------|
| Collision handling | Linked list per bucket | Probe next slot |
| Load factor limit | Can exceed 1.0 | Must stay < 1.0 |
| Memory | Extra pointers | No extra pointers |
| Cache performance | Poor (pointer chasing) | Good (contiguous) |
| Deletion | Simple | Needs tombstones |
| Clustering | No clustering | Primary/secondary clustering |
| Python's choice | -- | Yes (with perturbation) |

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Hash function | Maps keys to array indices |
| Collision | Two keys map to the same index |
| Chaining | Linked list at each bucket |
| Open addressing | Probe for next open slot |
| Load factor | n/m; resize when too high |
| Python dict | Open addressing, insertion-ordered, resizes at 2/3 |
| Average case | O(1) for get, put, delete |
| Custom hashing | Implement `__hash__` and `__eq__` |

---

**Next**: [Trees Basics](./06_Trees_Basics.md) -- Move from linear to hierarchical data structures.

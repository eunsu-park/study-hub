# Lesson 12: Magic Methods

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what magic methods (dunder methods) are and how Python uses them
2. Implement `__str__` and `__repr__` for human-readable and debug representations
3. Define equality and hashing with `__eq__` and `__hash__`
4. Make objects iterable with `__iter__` and `__next__`
5. Implement container protocols with `__getitem__`, `__len__`, `__contains__`
6. Create callable objects with `__call__`
7. Use context managers with `__enter__` and `__exit__`

## What Are Magic Methods?

Magic methods (also called **dunder methods** — "double underscore") are special methods surrounded by double underscores that Python calls automatically in response to operators, built-in functions, and language constructs.

```
┌────────────────────────────────────────────┐
│  Python syntax     →  Magic method called  │
├────────────────────────────────────────────┤
│  str(obj)          →  obj.__str__()        │
│  repr(obj)         →  obj.__repr__()       │
│  len(obj)          →  obj.__len__()        │
│  obj[key]          →  obj.__getitem__(key) │
│  obj == other      →  obj.__eq__(other)    │
│  for x in obj:     →  obj.__iter__()       │
│  obj(args)         →  obj.__call__(args)   │
│  with obj as x:    →  obj.__enter__()      │
│  hash(obj)         →  obj.__hash__()       │
│  bool(obj)         →  obj.__bool__()       │
└────────────────────────────────────────────┘
```

## `__str__` and `__repr__`

The two most important representation methods:

```
┌──────────────────────────────────────────────────┐
│  __repr__: Unambiguous, for DEVELOPERS           │
│  - Should look like valid Python if possible     │
│  - Used by repr(), debuggers, and containers     │
│  - Goal: reproduce the object                    │
│                                                  │
│  __str__: Readable, for END USERS                │
│  - Friendly, human-readable output               │
│  - Used by str(), print(), f-strings             │
│  - Falls back to __repr__ if not defined         │
└──────────────────────────────────────────────────┘
```

```python
class Product:
    def __init__(self, name, price, sku):
        self.name = name
        self.price = price
        self.sku = sku

    def __repr__(self):
        """For developers: unambiguous, reconstructable."""
        return f"Product({self.name!r}, {self.price}, {self.sku!r})"

    def __str__(self):
        """For users: readable, friendly."""
        return f"{self.name} - ${self.price:.2f}"


p = Product("Laptop", 999.99, "SKU-001")
print(repr(p))   # Product('Laptop', 999.99, 'SKU-001')
print(str(p))    # Laptop - $999.99
print(p)         # Laptop - $999.99 (print uses __str__)
print([p])       # [Product('Laptop', 999.99, 'SKU-001')] (containers use __repr__)
```

### Rule of Thumb

```python
# Always implement __repr__. Implement __str__ only if you need
# a different user-facing representation.

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    # No __str__ needed — __repr__ is sufficient
```

## `__eq__` and `__hash__`

### Equality

```python
class Card:
    """A playing card with custom equality."""

    SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
    RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10",
             "J", "Q", "K", "A")

    def __init__(self, rank, suit):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        """Two cards are equal if they have the same rank and suit."""
        if not isinstance(other, Card):
            return NotImplemented  # Let Python try other.__eq__(self)
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        """Must implement __hash__ if __eq__ is defined.
        Objects that are equal MUST have the same hash.
        """
        return hash((self.rank, self.suit))

    def __repr__(self):
        return f"Card({self.rank!r}, {self.suit!r})"

    def __str__(self):
        return f"{self.rank} of {self.suit}"


c1 = Card("A", "Spades")
c2 = Card("A", "Spades")
c3 = Card("K", "Hearts")

print(c1 == c2)   # True (same rank and suit)
print(c1 == c3)   # False
print(c1 == 42)   # False (NotImplemented -> False)

# Because __hash__ is defined, cards can be in sets and dicts
hand = {c1, c2, c3}
print(len(hand))  # 2 (c1 and c2 are equal, so deduplicated)
```

### The `__eq__` / `__hash__` Contract

```
┌─────────────────────────────────────────────────────┐
│  Rules:                                             │
│  1. If a == b, then hash(a) == hash(b)  (REQUIRED) │
│  2. If hash(a) == hash(b), a may or may not == b    │
│  3. If you define __eq__, Python sets __hash__=None │
│     unless you explicitly define __hash__            │
│  4. Unhashable objects can't be in sets or dict keys│
└─────────────────────────────────────────────────────┘
```

## Ordering: `__lt__`, `__le__`, `__gt__`, `__ge__`

```python
from functools import total_ordering


@total_ordering  # Generates __le__, __gt__, __ge__ from __eq__ and __lt__
class Temperature:
    """A comparable temperature."""

    def __init__(self, celsius):
        self.celsius = celsius

    def __eq__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    def __hash__(self):
        return hash(self.celsius)

    def __repr__(self):
        return f"Temperature({self.celsius}C)"


temps = [Temperature(100), Temperature(0), Temperature(37), Temperature(-40)]
print(sorted(temps))  # [Temperature(-40C), Temperature(0C), Temperature(37C), Temperature(100C)]
print(Temperature(100) > Temperature(0))  # True (generated by @total_ordering)
```

## `__iter__` and `__next__`

Make your objects work with `for` loops:

```python
class Fibonacci:
    """An iterable Fibonacci sequence."""

    def __init__(self, max_count):
        self.max_count = max_count

    def __iter__(self):
        """Return an iterator (fresh state each time)."""
        return FibonacciIterator(self.max_count)

    def __repr__(self):
        return f"Fibonacci(max_count={self.max_count})"


class FibonacciIterator:
    """Iterator that generates Fibonacci numbers."""

    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        value = self.a
        self.a, self.b = self.b, self.a + self.b
        self.count += 1
        return value


fib = Fibonacci(10)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Can iterate multiple times (because __iter__ returns fresh iterator)
for n in fib:
    if n > 10:
        break
    print(n, end=" ")  # 0 1 1 2 3 5 8
```

## Container Protocol: `__getitem__`, `__len__`, `__contains__`

```python
class Matrix:
    """A 2D matrix with container protocol support."""

    def __init__(self, rows):
        self._data = [list(row) for row in rows]
        self._rows = len(self._data)
        self._cols = len(self._data[0]) if self._data else 0

    def __getitem__(self, key):
        """Access elements: matrix[row, col] or matrix[row]."""
        if isinstance(key, tuple):
            row, col = key
            return self._data[row][col]
        return self._data[key]  # Return entire row

    def __setitem__(self, key, value):
        """Set elements: matrix[row, col] = value."""
        if isinstance(key, tuple):
            row, col = key
            self._data[row][col] = value
        else:
            self._data[key] = list(value)

    def __len__(self):
        """Number of rows."""
        return self._rows

    def __contains__(self, value):
        """Check if a value exists in the matrix."""
        return any(value in row for row in self._data)

    def __iter__(self):
        """Iterate over rows."""
        return iter(self._data)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self._data)
        return f"Matrix(\n  {rows_str}\n)"


m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(m[0, 1])      # 2 (row 0, col 1)
print(m[1])          # [4, 5, 6] (entire row 1)
print(len(m))        # 3 (number of rows)
print(5 in m)        # True
print(10 in m)       # False

m[0, 0] = 99
print(m[0])          # [99, 2, 3]
```

## `__call__`: Callable Objects

```python
class Validator:
    """A callable validator."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value):
        """Make the object callable like a function."""
        if not self.min_val <= value <= self.max_val:
            raise ValueError(
                f"{value} is not in range [{self.min_val}, {self.max_val}]"
            )
        return True


# Use like a function
validate_age = Validator(0, 150)
validate_score = Validator(0, 100)

print(validate_age(25))     # True
print(validate_score(95))   # True
# validate_age(200)         # ValueError
# validate_score(-5)        # ValueError
```

## Context Managers: `__enter__` and `__exit__`

```python
class Timer:
    """A context manager that measures execution time."""

    def __init__(self, label="Block"):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        """Called when entering the `with` block."""
        import time
        self._start = time.perf_counter()
        return self  # The `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the `with` block."""
        import time
        self.elapsed = time.perf_counter() - self._start
        print(f"{self.label}: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions


class FileManager:
    """Custom file manager context manager."""

    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False


# Usage
with Timer("Computation"):
    total = sum(range(1_000_000))
# Output: Computation: 0.0234 seconds
```

## Complete Example: Custom Collection

```python
class SortedList:
    """A list that maintains sorted order with full protocol support."""

    def __init__(self, initial=None):
        self._data = sorted(initial) if initial else []

    def add(self, value):
        """Insert value in sorted position."""
        import bisect
        bisect.insort(self._data, value)

    # Representation
    def __repr__(self):
        return f"SortedList({self._data})"

    def __str__(self):
        return str(self._data)

    # Container protocol
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __contains__(self, value):
        import bisect
        i = bisect.bisect_left(self._data, value)
        return i < len(self._data) and self._data[i] == value

    # Iteration
    def __iter__(self):
        return iter(self._data)

    # Comparison
    def __eq__(self, other):
        if isinstance(other, SortedList):
            return self._data == other._data
        return NotImplemented

    # Boolean
    def __bool__(self):
        return len(self._data) > 0

    # Concatenation
    def __add__(self, other):
        if isinstance(other, SortedList):
            return SortedList(self._data + other._data)
        return NotImplemented


sl = SortedList([5, 2, 8, 1, 9])
print(sl)           # [1, 2, 5, 8, 9]
sl.add(3)
print(sl)           # [1, 2, 3, 5, 8, 9]
print(len(sl))      # 6
print(sl[2])        # 3
print(5 in sl)      # True
print(bool(sl))     # True
```

## Summary

- Magic methods let you customize how Python's syntax and built-ins interact with your objects
- `__repr__` for developers (unambiguous), `__str__` for users (readable)
- `__eq__` and `__hash__` must follow the contract: equal objects must have equal hashes
- `__iter__` and `__next__` make objects work with `for` loops
- `__getitem__`, `__len__`, `__contains__` implement the container protocol
- `__call__` makes objects callable like functions
- `__enter__` and `__exit__` implement context managers (`with` blocks)
- Use `@total_ordering` to reduce boilerplate for comparison operators

## Next Steps

In [Lesson 13: Dataclasses and Modern OOP](13_Dataclasses_and_Modern_OOP.md), we will explore Python's modern tools for reducing OOP boilerplate.

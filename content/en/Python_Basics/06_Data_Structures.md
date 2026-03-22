# Data Structures

**Previous**: [Functions](./05_Functions.md) | **Next**: [Strings and Text Processing](./07_Strings_and_Text_Processing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create, index, slice, and manipulate lists using built-in methods and list comprehensions
2. Use tuples for immutable sequences, including packing, unpacking, and named tuples
3. Build and query dictionaries with common methods and dictionary comprehensions
4. Perform set operations (union, intersection, difference) and apply set comprehensions
5. Choose the appropriate data structure for different problems based on characteristics like mutability, ordering, and lookup speed
6. Work with nested data structures and understand their access patterns
7. Distinguish between shallow and deep copies and avoid common aliasing bugs

---

Data structures are containers that organize and store data so it can be accessed and modified efficiently. Python provides four fundamental built-in collection types -- lists, tuples, dictionaries, and sets -- each with different properties that make them suited to different tasks. Choosing the right data structure is one of the most important decisions in writing clean, performant code.

## 1. Lists

Lists are ordered, mutable sequences. They can hold items of any type and are the most commonly used data structure in Python.

### Creating Lists

```python
# Empty list
empty = []
also_empty = list()

# List with values
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, None]

# From other iterables
from_range = list(range(5))        # [0, 1, 2, 3, 4]
from_string = list("hello")       # ['h', 'e', 'l', 'l', 'o']
from_tuple = list((10, 20, 30))   # [10, 20, 30]

# Repetition
zeros = [0] * 5                    # [0, 0, 0, 0, 0]
pattern = [1, 2] * 3               # [1, 2, 1, 2, 1, 2]
```

### Indexing

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# Positive indexing (from start, 0-based)
print(fruits[0])    # Output: apple
print(fruits[2])    # Output: cherry

# Negative indexing (from end)
print(fruits[-1])   # Output: elderberry
print(fruits[-2])   # Output: date

# Modifying by index
fruits[1] = "blueberry"
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry']
```

### Slicing

Slicing uses the syntax `list[start:stop:step]`. The `stop` index is exclusive.

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])     # [2, 3, 4]
print(numbers[:4])      # [0, 1, 2, 3]       (from beginning)
print(numbers[6:])      # [6, 7, 8, 9]       (to end)
print(numbers[::2])     # [0, 2, 4, 6, 8]    (every 2nd element)
print(numbers[1::2])    # [1, 3, 5, 7, 9]    (odd-indexed elements)
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  (reversed)
print(numbers[7:2:-1])  # [7, 6, 5, 4, 3]    (reverse slice)

# Slice assignment
numbers[2:5] = [20, 30, 40]
print(numbers)  # [0, 1, 20, 30, 40, 5, 6, 7, 8, 9]

# Delete via slice
numbers[2:5] = []
print(numbers)  # [0, 1, 5, 6, 7, 8, 9]
```

### List Methods

```python
fruits = ["apple", "banana", "cherry"]

# append: add one item to the end
fruits.append("date")
print(fruits)  # ['apple', 'banana', 'cherry', 'date']

# extend: add all items from another iterable
fruits.extend(["elderberry", "fig"])
print(fruits)  # ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']

# insert: add item at a specific position
fruits.insert(1, "blueberry")
print(fruits)  # ['apple', 'blueberry', 'banana', 'cherry', 'date', 'elderberry', 'fig']

# remove: remove the first occurrence of a value
fruits.remove("banana")
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry', 'fig']

# pop: remove and return item at index (default: last)
last = fruits.pop()
print(last)    # fig
print(fruits)  # ['apple', 'blueberry', 'cherry', 'date', 'elderberry']

second = fruits.pop(1)
print(second)  # blueberry
print(fruits)  # ['apple', 'cherry', 'date', 'elderberry']

# index: find position of first occurrence
print(fruits.index("cherry"))  # 1

# count: count occurrences
numbers = [1, 2, 3, 2, 4, 2, 5]
print(numbers.count(2))  # 3

# sort: sort in place
numbers.sort()
print(numbers)  # [1, 2, 2, 2, 3, 4, 5]

numbers.sort(reverse=True)
print(numbers)  # [5, 4, 3, 2, 2, 2, 1]

# reverse: reverse in place
numbers.reverse()
print(numbers)  # [1, 2, 2, 2, 3, 4, 5]

# clear: remove all items
numbers.clear()
print(numbers)  # []
```

### List Membership and Length

```python
colors = ["red", "green", "blue"]

print("red" in colors)      # True
print("yellow" in colors)   # False
print("yellow" not in colors)  # True
print(len(colors))          # 3
```

### List Comprehensions

List comprehensions provide a concise way to create lists based on existing iterables.

```python
# Basic syntax: [expression for item in iterable]
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition: [expression for item in iterable if condition]
evens = [x for x in range(20) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# With transformation and condition
words = ["Hello", "WORLD", "Python", "CODE"]
lower_short = [w.lower() for w in words if len(w) <= 5]
print(lower_short)  # ['hello', 'world', 'code']

# Nested loops in comprehension
pairs = [(x, y) for x in range(3) for y in range(3)]
print(pairs)
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

# Flattening a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# If-else in comprehension (expression part, not filter)
labels = ["even" if x % 2 == 0 else "odd" for x in range(6)]
print(labels)  # ['even', 'odd', 'even', 'odd', 'even', 'odd']
```

---

## 2. Tuples

Tuples are ordered, **immutable** sequences. Once created, their elements cannot be changed.

### Creating Tuples

```python
# With parentheses
point = (3, 4)
rgb = (255, 128, 0)

# Without parentheses (tuple packing)
coordinates = 10, 20, 30
print(type(coordinates))  # <class 'tuple'>

# Single-element tuple (comma is required!)
single = (42,)
print(type(single))   # <class 'tuple'>

not_tuple = (42)
print(type(not_tuple))  # <class 'int'>

# Empty tuple
empty = ()
also_empty = tuple()

# From iterable
from_list = tuple([1, 2, 3])
from_string = tuple("abc")  # ('a', 'b', 'c')
```

### Tuple Operations

```python
point = (3, 4, 5)

# Indexing and slicing (same as lists)
print(point[0])     # 3
print(point[-1])    # 5
print(point[1:])    # (4, 5)

# Immutability: cannot modify
# point[0] = 10  # TypeError: 'tuple' object does not support item assignment

# Concatenation and repetition
a = (1, 2)
b = (3, 4)
print(a + b)       # (1, 2, 3, 4)
print(a * 3)       # (1, 2, 1, 2, 1, 2)

# Membership
print(3 in point)  # True
print(len(point))  # 3

# Methods
numbers = (1, 2, 3, 2, 4, 2)
print(numbers.count(2))  # 3
print(numbers.index(3))  # 2
```

### Tuple Packing and Unpacking

```python
# Packing
person = "Alice", 30, "Seoul"

# Unpacking
name, age, city = person
print(f"{name}, {age}, {city}")  # Alice, 30, Seoul

# Swap variables (Python idiom)
a, b = 10, 20
a, b = b, a
print(a, b)  # 20 10

# Extended unpacking with *
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# Ignore values with _
_, age, _ = person
print(age)  # 30

# Unpacking in loops
pairs = [(1, "a"), (2, "b"), (3, "c")]
for number, letter in pairs:
    print(f"{number} -> {letter}")
# 1 -> a
# 2 -> b
# 3 -> c
```

### When to Use Tuples vs Lists

| Tuples | Lists |
|--------|-------|
| Immutable (safe, hashable) | Mutable (flexible) |
| Fixed collection of items | Collection that grows/shrinks |
| Can be dictionary keys | Cannot be dictionary keys |
| Slightly faster | Slightly slower |
| Represent records (name, age) | Represent collections of same type |

### Named Tuples

Named tuples give each position a meaningful name, making code more readable.

```python
from collections import namedtuple

# Define a named tuple type
Point = namedtuple("Point", ["x", "y"])

p = Point(3, 4)
print(p.x)       # 3
print(p.y)       # 4
print(p[0])      # 3 (still works like a tuple)
print(p)         # Point(x=3, y=4)

# Real-world example
Student = namedtuple("Student", ["name", "grade", "age"])

students = [
    Student("Alice", 95, 20),
    Student("Bob", 88, 22),
    Student("Charlie", 92, 21),
]

# Access by name is much clearer
for s in students:
    print(f"{s.name}: grade={s.grade}, age={s.age}")

# Named tuples are still immutable
# p.x = 10  # AttributeError

# Create modified copy with _replace
p2 = p._replace(x=10)
print(p2)  # Point(x=10, y=4)

# Convert to dictionary
print(p._asdict())  # {'x': 3, 'y': 4}
```

---

## 3. Dictionaries

Dictionaries are **unordered** (insertion-ordered since Python 3.7), mutable mappings of key-value pairs. Keys must be hashable (strings, numbers, tuples of hashables).

### Creating Dictionaries

```python
# Curly braces
student = {"name": "Alice", "age": 20, "grade": 95}

# dict() constructor
config = dict(host="localhost", port=8080, debug=True)

# From list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
d = dict(pairs)
print(d)  # {'a': 1, 'b': 2, 'c': 3}

# dict.fromkeys
keys = ["name", "age", "city"]
defaults = dict.fromkeys(keys, "unknown")
print(defaults)  # {'name': 'unknown', 'age': 'unknown', 'city': 'unknown'}

# Empty dictionary
empty = {}
also_empty = dict()
```

### Accessing and Modifying

```python
student = {"name": "Alice", "age": 20, "grade": 95}

# Access by key
print(student["name"])   # Alice

# KeyError if key does not exist
# print(student["email"])  # KeyError: 'email'

# get() returns None (or default) instead of raising error
print(student.get("email"))           # None
print(student.get("email", "N/A"))    # N/A

# Add or update
student["email"] = "alice@example.com"   # Add new key
student["age"] = 21                       # Update existing key
print(student)
# {'name': 'Alice', 'age': 21, 'grade': 95, 'email': 'alice@example.com'}

# Delete
del student["email"]
print(student)  # {'name': 'Alice', 'age': 21, 'grade': 95}
```

### Dictionary Methods

```python
student = {"name": "Alice", "age": 20, "grade": 95}

# keys(), values(), items()
print(list(student.keys()))    # ['name', 'age', 'grade']
print(list(student.values()))  # ['Alice', 20, 95]
print(list(student.items()))   # [('name', 'Alice'), ('age', 20), ('grade', 95)]

# update: merge another dictionary
student.update({"age": 21, "city": "Seoul"})
print(student)  # {'name': 'Alice', 'age': 21, 'grade': 95, 'city': 'Seoul'}

# setdefault: get value or set it if missing
email = student.setdefault("email", "unknown@example.com")
print(email)     # unknown@example.com
print(student["email"])  # unknown@example.com

# setdefault does not overwrite existing keys
name = student.setdefault("name", "Bob")
print(name)  # Alice (not overwritten)

# pop: remove and return value
grade = student.pop("grade")
print(grade)    # 95

# pop with default (no KeyError)
missing = student.pop("phone", "not found")
print(missing)  # not found

# popitem: remove and return last inserted pair
last = student.popitem()
print(last)  # ('email', 'unknown@example.com')

# copy: shallow copy
copy_student = student.copy()

# clear: remove all items
student.clear()
print(student)  # {}
```

### Iterating Over Dictionaries

```python
scores = {"Alice": 95, "Bob": 88, "Charlie": 92}

# Iterate over keys (default)
for name in scores:
    print(name)

# Iterate over values
for score in scores.values():
    print(score)

# Iterate over key-value pairs
for name, score in scores.items():
    print(f"{name}: {score}")

# Check membership (checks keys by default)
print("Alice" in scores)     # True
print(95 in scores)           # False (checks keys, not values)
print(95 in scores.values())  # True
```

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
squares = {x: x ** 2 for x in range(6)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With condition
even_squares = {x: x ** 2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Invert a dictionary (swap keys and values)
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}

# From two lists
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
name_age = {name: age for name, age in zip(names, ages)}
print(name_age)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# Filter and transform
words = ["Hello", "World", "Python", "Go", "JS"]
long_words = {w: len(w) for w in words if len(w) > 3}
print(long_words)  # {'Hello': 5, 'World': 5, 'Python': 6}
```

### Merge Operators (Python 3.9+)

```python
defaults = {"color": "blue", "size": "medium", "theme": "light"}
user_prefs = {"color": "red", "font": "Arial"}

# Merge with | (creates new dict, right side wins on conflicts)
merged = defaults | user_prefs
print(merged)
# {'color': 'red', 'size': 'medium', 'theme': 'light', 'font': 'Arial'}

# In-place merge with |=
defaults |= user_prefs
print(defaults)
# {'color': 'red', 'size': 'medium', 'theme': 'light', 'font': 'Arial'}
```

---

## 4. Sets

Sets are **unordered** collections of **unique** elements. They are ideal for membership testing, deduplication, and mathematical set operations.

### Creating Sets

```python
# Curly braces (but {} creates a dict, not a set!)
colors = {"red", "green", "blue"}
print(type(colors))  # <class 'set'>

# Empty set must use set()
empty = set()
print(type(empty))   # <class 'set'>

# From iterable (automatically removes duplicates)
numbers = set([1, 2, 2, 3, 3, 3])
print(numbers)  # {1, 2, 3}

from_string = set("mississippi")
print(from_string)  # {'m', 'i', 's', 'p'} (order may vary)
```

### Set Operations

```python
# Adding and removing
colors = {"red", "green", "blue"}

colors.add("yellow")
print(colors)  # {'red', 'green', 'blue', 'yellow'}

colors.discard("green")   # Remove if present (no error if missing)
colors.remove("blue")     # Remove (raises KeyError if missing)
# colors.remove("purple")  # KeyError

popped = colors.pop()     # Remove and return arbitrary element
print(f"Popped: {popped}")
```

### Mathematical Set Operations

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Union: elements in either set
print(a | b)                # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))           # {1, 2, 3, 4, 5, 6, 7, 8}

# Intersection: elements in both sets
print(a & b)                # {4, 5}
print(a.intersection(b))    # {4, 5}

# Difference: elements in a but not in b
print(a - b)                # {1, 2, 3}
print(a.difference(b))      # {1, 2, 3}

# Symmetric difference: elements in either but not both
print(a ^ b)                        # {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))    # {1, 2, 3, 6, 7, 8}
```

### Set Relationships

```python
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}
c = {6, 7}

# Subset and superset
print(a.issubset(b))      # True (all of a is in b)
print(b.issuperset(a))    # True (b contains all of a)
print(a <= b)             # True (operator form of issubset)

# Disjoint: no common elements
print(a.isdisjoint(c))    # True
print(a.isdisjoint(b))    # False
```

### Practical Set Examples

```python
# Remove duplicates from a list (order not preserved)
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique = list(set(data))
print(sorted(unique))  # [1, 2, 3, 4, 5, 6, 9]

# Remove duplicates preserving order (Python 3.7+)
unique_ordered = list(dict.fromkeys(data))
print(unique_ordered)  # [3, 1, 4, 5, 9, 2, 6]

# Find common elements
list_a = [1, 2, 3, 4, 5]
list_b = [4, 5, 6, 7, 8]
common = set(list_a) & set(list_b)
print(common)  # {4, 5}

# Find items in one list but not another
only_a = set(list_a) - set(list_b)
print(only_a)  # {1, 2, 3}

# Efficient membership testing
valid_usernames = {"alice", "bob", "charlie", "diana"}
username = "bob"
if username in valid_usernames:  # O(1) average time
    print(f"{username} is valid")
```

### Set Comprehensions

```python
# Basic set comprehension
squares = {x ** 2 for x in range(-5, 6)}
print(squares)  # {0, 1, 4, 9, 16, 25}

# With condition
even_squares = {x ** 2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0, 4, 16, 36, 64}

# Extract unique first characters
words = ["apple", "avocado", "banana", "blueberry", "cherry"]
first_chars = {w[0] for w in words}
print(first_chars)  # {'a', 'b', 'c'}
```

### Frozen Sets

Frozen sets are immutable sets. They can be used as dictionary keys or elements of other sets.

```python
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError: 'frozenset' has no method 'add'

# Can be used as a dictionary key
permissions = {
    frozenset({"read"}): "viewer",
    frozenset({"read", "write"}): "editor",
    frozenset({"read", "write", "admin"}): "admin",
}
user_perms = frozenset({"read", "write"})
print(permissions[user_perms])  # editor
```

---

## 5. Choosing the Right Data Structure

| Feature | List | Tuple | Dict | Set |
|---------|------|-------|------|-----|
| Ordered | Yes | Yes | Yes (3.7+) | No |
| Mutable | Yes | No | Yes | Yes |
| Duplicates | Yes | Yes | Keys: No | No |
| Indexable | Yes | Yes | By key | No |
| Hashable | No | Yes* | No | No |
| Use case | General sequence | Immutable record | Key-value mapping | Unique elements |

\* Tuples are hashable only if all their elements are hashable.

### Decision Guide

```python
# Need ordered, changeable collection? -> list
shopping = ["milk", "eggs", "bread"]

# Need immutable record or dictionary key? -> tuple
coordinate = (40.7128, -74.0060)

# Need to look up values by key? -> dict
phonebook = {"Alice": "555-0100", "Bob": "555-0200"}

# Need unique elements or set math? -> set
tags = {"python", "tutorial", "beginner"}

# Need to count occurrences? -> dict or Counter
from collections import Counter
words = ["the", "cat", "sat", "on", "the", "mat"]
word_count = Counter(words)
print(word_count)  # Counter({'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1})
```

---

## 6. Nested Data Structures

Real-world data is often complex, requiring nesting of data structures.

### List of Dictionaries

```python
students = [
    {"name": "Alice", "grades": [95, 88, 92]},
    {"name": "Bob", "grades": [78, 85, 90]},
    {"name": "Charlie", "grades": [92, 95, 88]},
]

# Access nested data
print(students[0]["name"])        # Alice
print(students[0]["grades"][1])   # 88

# Calculate average for each student
for student in students:
    avg = sum(student["grades"]) / len(student["grades"])
    print(f"{student['name']}: {avg:.1f}")
# Alice: 91.7
# Bob: 84.3
# Charlie: 91.7
```

### Dictionary of Lists

```python
class_roster = {
    "math": ["Alice", "Bob", "Charlie"],
    "science": ["Bob", "Diana", "Eve"],
    "history": ["Alice", "Charlie", "Eve"],
}

# Find students taking all subjects
all_students = set()
for subject_students in class_roster.values():
    all_students.update(subject_students)
print(all_students)  # {'Alice', 'Bob', 'Charlie', 'Diana', 'Eve'}

# Find students in both math and science
math_students = set(class_roster["math"])
science_students = set(class_roster["science"])
print(math_students & science_students)  # {'Bob'}
```

### Nested Dictionary

```python
company = {
    "engineering": {
        "backend": {"lead": "Alice", "members": 5},
        "frontend": {"lead": "Bob", "members": 3},
    },
    "marketing": {
        "digital": {"lead": "Charlie", "members": 4},
    },
}

# Deep access
print(company["engineering"]["backend"]["lead"])  # Alice

# Safe deep access
def deep_get(data, *keys, default=None):
    """Safely get a value from a nested dictionary."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

print(deep_get(company, "engineering", "backend", "lead"))      # Alice
print(deep_get(company, "sales", "team", "lead", default="N/A")) # N/A
```

### Building Nested Structures with defaultdict

```python
from collections import defaultdict

# Group items by category
items = [
    ("fruit", "apple"),
    ("vegetable", "carrot"),
    ("fruit", "banana"),
    ("vegetable", "broccoli"),
    ("fruit", "cherry"),
]

grouped = defaultdict(list)
for category, item in items:
    grouped[category].append(item)

print(dict(grouped))
# {'fruit': ['apple', 'banana', 'cherry'], 'vegetable': ['carrot', 'broccoli']}

# Nested defaultdict for two-level grouping
sales = [
    ("2024", "Q1", 100),
    ("2024", "Q2", 150),
    ("2024", "Q1", 200),
    ("2025", "Q1", 300),
]

yearly = defaultdict(lambda: defaultdict(list))
for year, quarter, amount in sales:
    yearly[year][quarter].append(amount)

print(yearly["2024"]["Q1"])  # [100, 200]
```

---

## 7. Copying: Shallow vs Deep

### The Aliasing Problem

```python
# Assignment creates an alias, NOT a copy
original = [1, 2, 3]
alias = original

alias.append(4)
print(original)  # [1, 2, 3, 4] -- both names point to the same list!
print(alias is original)  # True
```

### Shallow Copy

A shallow copy creates a new outer container but shares references to the inner objects.

```python
import copy

# Ways to create a shallow copy of a list
original = [1, 2, 3, [4, 5]]

copy1 = original.copy()        # list.copy() method
copy2 = original[:]            # slice
copy3 = list(original)         # constructor
copy4 = copy.copy(original)    # copy module

# The outer list is independent
copy1.append(6)
print(original)  # [1, 2, 3, [4, 5]] -- not affected
print(copy1)     # [1, 2, 3, [4, 5], 6]

# But nested objects are still shared!
copy1[3].append(99)
print(original)  # [1, 2, 3, [4, 5, 99]] -- nested list IS affected!
print(copy1)     # [1, 2, 3, [4, 5, 99], 6]
```

### Deep Copy

A deep copy creates completely independent copies of all nested objects.

```python
import copy

original = [1, 2, [3, 4, [5, 6]]]

# Deep copy
deep = copy.deepcopy(original)

deep[2][2].append(7)
print(original)  # [1, 2, [3, 4, [5, 6]]]  -- not affected
print(deep)      # [1, 2, [3, 4, [5, 6, 7]]]
```

### Dictionary Copying

```python
import copy

original = {
    "name": "Alice",
    "scores": [95, 88, 92],
    "address": {"city": "Seoul", "zip": "12345"},
}

# Shallow copy
shallow = original.copy()
shallow["scores"].append(100)
print(original["scores"])  # [95, 88, 92, 100] -- shared!

# Deep copy
original["scores"].pop()  # Remove the 100 we just added
deep = copy.deepcopy(original)
deep["scores"].append(100)
deep["address"]["city"] = "Busan"
print(original["scores"])         # [95, 88, 92] -- independent
print(original["address"]["city"]) # Seoul -- independent
```

### When to Use Each

| Scenario | Method |
|----------|--------|
| Simple flat list/dict | Shallow copy is fine |
| Nested mutable objects | Deep copy needed |
| Performance-critical, read-only | Share reference (alias) |
| Immutable data (tuples, strings) | No copy needed |

---

## 8. Practical Examples

### Example: Inventory Management

```python
inventory = {}

def add_item(name, quantity, price):
    """Add or update an item in inventory."""
    if name in inventory:
        inventory[name]["quantity"] += quantity
    else:
        inventory[name] = {"quantity": quantity, "price": price}

def remove_item(name, quantity):
    """Remove quantity of an item. Remove entry if quantity reaches 0."""
    if name not in inventory:
        print(f"{name} not in inventory")
        return
    inventory[name]["quantity"] -= quantity
    if inventory[name]["quantity"] <= 0:
        del inventory[name]

def get_total_value():
    """Calculate total inventory value."""
    return sum(
        item["quantity"] * item["price"]
        for item in inventory.values()
    )

def get_report():
    """Generate inventory report sorted by value."""
    items = []
    for name, info in inventory.items():
        value = info["quantity"] * info["price"]
        items.append((name, info["quantity"], info["price"], value))

    items.sort(key=lambda x: -x[3])  # Sort by value descending
    return items

add_item("apple", 50, 1.20)
add_item("banana", 30, 0.50)
add_item("cherry", 100, 3.00)
add_item("apple", 20, 1.20)  # Add more apples

print(f"Total value: ${get_total_value():.2f}")
# Total value: $399.00

for name, qty, price, value in get_report():
    print(f"  {name}: {qty} x ${price:.2f} = ${value:.2f}")
# cherry: 100 x $3.00 = $300.00
# apple: 70 x $1.20 = $84.00
# banana: 30 x $0.50 = $15.00
```

### Example: Matrix Operations with Nested Lists

```python
def create_matrix(rows, cols, fill=0):
    """Create a matrix (list of lists)."""
    return [[fill] * cols for _ in range(rows)]

def print_matrix(matrix):
    """Pretty-print a matrix."""
    for row in matrix:
        print("  ".join(f"{val:4}" for val in row))

def matrix_add(a, b):
    """Add two matrices."""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] + b[r][c] for c in range(cols)] for r in range(rows)]

def matrix_transpose(matrix):
    """Transpose a matrix."""
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]

m1 = [[1, 2, 3],
      [4, 5, 6]]

m2 = [[7, 8, 9],
      [10, 11, 12]]

print("Matrix 1:")
print_matrix(m1)

print("\nMatrix 2:")
print_matrix(m2)

print("\nSum:")
print_matrix(matrix_add(m1, m2))
# 8  10  12
# 14  16  18

print("\nTranspose of Matrix 1:")
print_matrix(matrix_transpose(m1))
# 1   4
# 2   5
# 3   6
```

---

## 9. Summary

| Data Structure | Ordered | Mutable | Duplicates | Key Feature |
|----------------|---------|---------|------------|-------------|
| **List** | Yes | Yes | Yes | General-purpose sequence |
| **Tuple** | Yes | No | Yes | Immutable, hashable |
| **Dict** | Yes (3.7+) | Yes | Keys: No | Key-value mapping, O(1) lookup |
| **Set** | No | Yes | No | Unique elements, set math |

Key takeaways:
- **Lists** are your default go-to for ordered collections
- **Tuples** protect data from accidental modification and work as dictionary keys
- **Dictionaries** excel at fast lookups and representing structured data
- **Sets** handle uniqueness and set operations efficiently
- **Comprehensions** provide a Pythonic way to build all four types
- Always be aware of **shallow vs deep copy** when working with nested structures

---

## Exercises

1. Write a function that takes a list of numbers and returns a dictionary with keys `"min"`, `"max"`, `"mean"`, and `"median"`.
2. Given two dictionaries, write a function that returns a dictionary of keys that exist in both, with values as a tuple `(value_from_dict1, value_from_dict2)`.
3. Implement a simple phone book using a dictionary: add, delete, search, and list all contacts.
4. Write a function that finds all duplicate elements in a list using sets.
5. Create a nested data structure representing a school (departments -> classes -> students) and write functions to query it (e.g., find all students in a department, count students per class).

---

**Previous**: [Functions](./05_Functions.md) | **Next**: [Strings and Text Processing](./07_Strings_and_Text_Processing.md)

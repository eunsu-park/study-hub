# Common Bug Patterns

**Previous**: [Using a Debugger](./03_Using_a_Debugger.md) | **Next**: [Debugging Strategy](./05_Debugging_Strategy.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify off-by-one errors in loops, slicing, and range calculations
2. Recognize mutable default argument bugs and apply the `None` sentinel pattern
3. Detect shared mutable state bugs (aliasing) in lists and dictionaries
4. Understand variable scope issues including the `UnboundLocalError` trap
5. Handle `None` values safely and distinguish between `None`, `0`, `""`, and `False`
6. Avoid common equality vs identity mistakes (`==` vs `is`)
7. Recognize integer division and floating-point precision bugs
8. Identify common string and encoding pitfalls

---

Experienced developers don't just fix bugs one at a time -- they learn to **recognize patterns**. Certain categories of bugs appear over and over across all Python projects. Once you learn to recognize these patterns, you can spot them during code review or even prevent them as you write code. This lesson catalogs the most common bug patterns in Python, each with a buggy example, explanation, and fix.

> **80/20 Rule of Bugs:** About 80% of beginner bugs fall into fewer than 10 patterns. Learning these patterns is the fastest way to level up your debugging skills.

---

## 1. Off-by-One Errors

The most classic bug in programming: your loop or index is off by exactly one.

### 1.1 Range Boundary

```python
# BUG: Prints 1 to 9, not 1 to 10
for i in range(1, 10):
    print(i)
```

```python
# FIX: range() end is exclusive
for i in range(1, 11):
    print(i)
```

### 1.2 List Indexing

```python
# BUG: Misses the last element
items = ["a", "b", "c", "d"]
for i in range(len(items) - 1):  # range(3) → 0, 1, 2
    print(items[i])              # "d" is never printed
```

```python
# FIX: Use range(len(items)) or better yet, iterate directly
for item in items:
    print(item)
```

### 1.3 Fence Post Error

```python
# BUG: How many fence posts for 10 sections of fence?
sections = 10
posts = sections  # Wrong! You need sections + 1

# FIX
posts = sections + 1  # 11 posts for 10 sections
```

### 1.4 Slicing

```python
# BUG: Getting the "middle" element
data = [10, 20, 30, 40, 50]
middle_index = len(data) / 2       # 2.5 (float, not int!)
# TypeError: list indices must be integers

# FIX: Use integer division
middle_index = len(data) // 2      # 2
print(data[middle_index])          # 30
```

---

## 2. Mutable Default Arguments

One of Python's most infamous gotchas.

### 2.1 The Bug

```python
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']       -- looks fine
print(add_item("b"))  # ['a', 'b']  -- BUG! Where did 'a' come from?
print(add_item("c"))  # ['a', 'b', 'c']  -- It keeps accumulating!
```

**Why**: Default arguments are evaluated **once** when the function is defined, not each time it's called. The same list object is reused across all calls.

### 2.2 The Fix: None Sentinel Pattern

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['b']  -- Fresh list each time
```

This applies to all mutable defaults: `list`, `dict`, `set`, and custom objects.

```python
# BAD
def create_user(name, preferences={}):  # Shared dict!
    ...

# GOOD
def create_user(name, preferences=None):
    if preferences is None:
        preferences = {}
    ...
```

---

## 3. Shared Mutable State (Aliasing)

### 3.1 List Aliasing

```python
# BUG: Two names, one list
original = [1, 2, 3]
copy = original           # Not a copy! Both point to the same list
copy.append(4)
print(original)           # [1, 2, 3, 4]  -- original was modified!
```

```python
# FIX: Make an actual copy
copy = original.copy()         # Shallow copy
copy = list(original)          # Also shallow copy
copy = original[:]             # Also shallow copy

import copy
deep = copy.deepcopy(original) # Deep copy (for nested structures)
```

### 3.2 Nested List Trap

```python
# BUG: Creating a 2D grid
grid = [[0] * 3] * 3
grid[0][0] = 1
print(grid)  # [[1, 0, 0], [1, 0, 0], [1, 0, 0]]  -- All rows changed!
```

**Why**: `[[0] * 3] * 3` creates three references to the **same** inner list.

```python
# FIX: Use list comprehension
grid = [[0] * 3 for _ in range(3)]
grid[0][0] = 1
print(grid)  # [[1, 0, 0], [0, 0, 0], [0, 0, 0]]  -- Only first row changed
```

### 3.3 Dictionary Aliasing in Loops

```python
# BUG: Reusing the same dict
users = []
user = {}
for name in ["Alice", "Bob", "Charlie"]:
    user["name"] = name
    users.append(user)

print(users)
# [{'name': 'Charlie'}, {'name': 'Charlie'}, {'name': 'Charlie'}]
```

```python
# FIX: Create a new dict each iteration
users = []
for name in ["Alice", "Bob", "Charlie"]:
    user = {"name": name}  # New dict each time
    users.append(user)
```

---

## 4. Scope Issues

### 4.1 UnboundLocalError

```python
count = 0

def increment():
    count += 1   # UnboundLocalError: local variable 'count' referenced
                 #   before assignment
    return count
```

**Why**: The assignment `count += 1` makes Python treat `count` as a local variable. But the local variable hasn't been assigned yet when `+= 1` tries to read it.

```python
# FIX 1: Use global (discouraged)
def increment():
    global count
    count += 1
    return count

# FIX 2: Pass as argument and return (preferred)
def increment(count):
    return count + 1

count = 0
count = increment(count)
```

### 4.2 Late Binding Closures

```python
# BUG: Lambda captures variable by reference, not value
functions = []
for i in range(5):
    functions.append(lambda: i)

print([f() for f in functions])  # [4, 4, 4, 4, 4]  -- All return 4!
```

**Why**: The lambda captures the variable `i` itself, not its current value. By the time the lambdas execute, `i` is 4.

```python
# FIX: Use default argument to capture current value
functions = []
for i in range(5):
    functions.append(lambda i=i: i)

print([f() for f in functions])  # [0, 1, 2, 3, 4]
```

### 4.3 Variable Shadowing

```python
items = [1, 2, 3]

def process():
    items = [4, 5, 6]  # Creates a LOCAL variable, doesn't modify the global
    items.append(7)
    print(f"Inside: {items}")   # [4, 5, 6, 7]

process()
print(f"Outside: {items}")     # [1, 2, 3]  -- unmodified
```

---

## 5. None Handling

### 5.1 Forgetting to Return

```python
def find_user(name, users):
    for user in users:
        if user["name"] == name:
            return user
    # No return statement if not found → returns None implicitly

user = find_user("Dave", users)
print(user["email"])  # TypeError: 'NoneType' object is not subscriptable
```

```python
# FIX: Always handle the None case
user = find_user("Dave", users)
if user is not None:
    print(user["email"])
else:
    print("User not found")
```

### 5.2 Truthy/Falsy Confusion

```python
# BUG: Treating 0 and "" as "missing"
def display(value):
    if not value:           # This catches 0, "", [], {}, False, AND None!
        print("No value")
    else:
        print(f"Value: {value}")

display(0)     # "No value"  -- BUG! 0 is a valid value
display("")    # "No value"  -- BUG! Empty string might be valid
```

```python
# FIX: Check specifically for None
def display(value):
    if value is None:
        print("No value")
    else:
        print(f"Value: {value}")

display(0)     # "Value: 0"
display("")    # "Value: "
display(None)  # "No value"
```

### 5.3 Chained None Access

```python
# BUG: Any step can return None
result = get_user().get_profile().get_avatar()

# FIX: Check at each step or use try/except
user = get_user()
if user is not None:
    profile = user.get_profile()
    if profile is not None:
        avatar = profile.get_avatar()
```

---

## 6. Equality vs Identity

### 6.1 `==` vs `is`

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)   # True  -- same value
print(a is b)   # False -- different objects

# Use == for value comparison
# Use is ONLY for None, True, False
if x is None:     # CORRECT
    ...
if x == None:     # WRONG (works but bad practice)
    ...
```

### 6.2 Integer Caching Surprise

```python
a = 256
b = 256
print(a is b)   # True  -- Python caches small integers (-5 to 256)

a = 257
b = 257
print(a is b)   # False (might vary by implementation)
```

**Rule**: Never use `is` to compare numbers or strings. Always use `==`.

---

## 7. Numeric Pitfalls

### 7.1 Integer Division

```python
# Python 3: / always returns float
result = 7 / 2    # 3.5
result = 7 // 2   # 3 (integer division)

# BUG: Mixing up / and // for indexing
mid = len(data) / 2    # TypeError: float index
mid = len(data) // 2   # Correct: integer index
```

### 7.2 Floating-Point Precision

```python
# BUG: Floating-point comparison
print(0.1 + 0.2 == 0.3)  # False!
print(0.1 + 0.2)          # 0.30000000000000004
```

```python
# FIX: Use math.isclose() or a tolerance
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# Or use decimal for exact arithmetic
from decimal import Decimal
print(Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))  # True
```

---

## 8. String and Encoding Pitfalls

### 8.1 String Immutability

```python
# BUG: Strings are immutable
s = "hello"
s[0] = "H"  # TypeError: 'str' object does not support item assignment

# FIX:
s = "H" + s[1:]  # "Hello"
```

### 8.2 Accidental String Iteration

```python
# BUG: Iterating over a string instead of a list
def process_items(items):
    for item in items:
        print(f"Processing: {item}")

process_items("hello")
# Processing: h
# Processing: e
# Processing: l  ... not what you wanted!

process_items(["hello"])  # FIX: Pass a list
```

### 8.3 Encoding Mismatch

```python
# BUG: Default encoding may not match file encoding
with open("data.csv") as f:      # Uses system default encoding
    data = f.read()               # UnicodeDecodeError on non-UTF-8 files

# FIX: Specify encoding explicitly
with open("data.csv", encoding="utf-8") as f:
    data = f.read()
```

---

## 9. Iteration Pitfalls

### 9.1 Modifying a List During Iteration

```python
# BUG: Removing items while iterating
numbers = [1, 2, 3, 4, 5, 6]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)

print(numbers)  # [1, 3, 5, 6]  -- 6 wasn't removed!
```

**Why**: Removing items shifts indices, causing elements to be skipped.

```python
# FIX 1: Build a new list
numbers = [n for n in numbers if n % 2 != 0]

# FIX 2: Iterate over a copy
for n in numbers[:]:  # [:] creates a copy
    if n % 2 == 0:
        numbers.remove(n)
```

### 9.2 Exhausted Iterators

```python
# BUG: Generator is consumed after first use
squares = (x**2 for x in range(5))
print(list(squares))  # [0, 1, 4, 9, 16]
print(list(squares))  # []  -- Empty! Generator is exhausted

# FIX: Use a list if you need to iterate multiple times
squares = [x**2 for x in range(5)]
```

---

## 10. Quick Reference: Bug Pattern Checklist

| Pattern | Symptom | Fix |
|---------|---------|-----|
| Off-by-one | Loop misses first/last element | Check `range()` bounds |
| Mutable default | Function accumulates state | Use `None` sentinel |
| Aliasing | Changing a "copy" changes the original | Use `.copy()` or `copy.deepcopy()` |
| UnboundLocalError | Error on `x += 1` inside function | Pass as argument, return result |
| Late binding | All lambdas return the same value | Use default argument `lambda i=i: i` |
| None handling | `NoneType has no attribute` | Check `is None` before accessing |
| Truthy trap | `0` or `""` treated as missing | Use `is None` instead of `not value` |
| Float comparison | `0.1 + 0.2 != 0.3` | Use `math.isclose()` |
| Modify during iteration | Items skipped during removal | Build new list with comprehension |
| Exhausted iterator | Second iteration yields nothing | Use list instead of generator |

---

## Summary

- Off-by-one errors are the most common: always double-check `range()` boundaries
- Never use mutable objects as default arguments -- use the `None` sentinel pattern
- Assignment creates aliases, not copies -- use `.copy()` for independent copies
- Check for `None` explicitly with `is None`, not truthiness tests
- Use `==` for values, `is` only for `None`/`True`/`False`
- Never compare floats with `==` -- use `math.isclose()`
- Never modify a collection while iterating over it

---

## Exercises

1. Identify and fix off-by-one errors in given code
2. Fix mutable default argument bugs
3. Fix aliasing bugs in list and dictionary code
4. Handle None values correctly in a data processing function
5. Fix a floating-point comparison bug

**Previous**: [Using a Debugger](./03_Using_a_Debugger.md) | **Next**: [Debugging Strategy](./05_Debugging_Strategy.md)

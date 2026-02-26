# Functional Programming

**Previous**: [Async Programming](./08_Async_Programming.md) | **Next**: [Performance Optimization](./10_Performance_Optimization.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the core principles of functional programming and how they differ from imperative programming
2. Write and use first-class functions, including passing and returning functions as values
3. Apply higher-order functions (`map`, `filter`, `reduce`) to transform collections
4. Use lambda expressions appropriately and identify when regular functions are preferred
5. Implement partial application, memoization, and function overloading with the `functools` module
6. Distinguish between pure and impure functions and apply immutability patterns in Python
7. Build data processing pipelines using function composition and chaining
8. Apply `itertools` combinatoric, grouping, and infinite iterator utilities effectively

---

Functional programming is one of the most influential paradigms in modern software development. Whether you are writing data pipelines, building concurrent systems, or simply trying to produce cleaner and more testable code, understanding how to think in terms of pure functions and immutable data will sharpen every part of your Python practice.

## 1. What is Functional Programming?

Functional programming is a paradigm that structures programs around pure functions and immutable data.

### Core Principles

| Principle | Description |
|-----------|-------------|
| Pure Functions | Same input → Same output, no side effects |
| Immutability | Create new data instead of modifying existing |
| First-Class Functions | Pass/return functions as values |
| Declarative | Describe "what" rather than "how" |

### Imperative vs Declarative

```python
numbers = [1, 2, 3, 4, 5]

# Imperative (how)
result = []
for n in numbers:
    if n % 2 == 0:
        result.append(n ** 2)
print(result)  # [4, 16]

# Declarative/Functional (what)
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))
print(result)  # [4, 16]

# More Pythonic (list comprehension)
result = [n ** 2 for n in numbers if n % 2 == 0]
print(result)  # [4, 16]
```

---

## 2. First-Class Functions

In Python, functions are first-class objects.

### Assign to Variables

```python
def greet(name):
    return f"Hello, {name}!"

# Assign function to variable
say_hello = greet
print(say_hello("Python"))  # Hello, Python!

# Functions are objects
print(type(greet))  # <class 'function'>
print(greet.__name__)  # greet
```

### Pass as Arguments

```python
def apply_twice(func, value):
    return func(func(value))

def add_ten(x):
    return x + 10

result = apply_twice(add_ten, 5)
print(result)  # 25 (5 + 10 + 10)
```

### Return Functions

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

---

## 3. Higher-Order Functions

Functions that take functions as arguments or return functions.

### map()

Apply a function to all elements.

```python
numbers = [1, 2, 3, 4, 5]

# Square
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# String conversion
words = ["hello", "world"]
upper_words = list(map(str.upper, words))
print(upper_words)  # ['HELLO', 'WORLD']

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # [11, 22, 33]
```

### filter()

Select elements that satisfy a condition.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Even numbers only
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Remove empty strings
words = ["hello", "", "world", "", "python"]
non_empty = list(filter(None, words))  # Remove falsy values
print(non_empty)  # ['hello', 'world', 'python']

# Custom condition
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 17},
    {"name": "Charlie", "age": 25},
]
adults = list(filter(lambda u: u["age"] >= 18, users))
print(adults)  # [{'name': 'Alice', ...}, {'name': 'Charlie', ...}]
```

### reduce()

Accumulate elements to a single value.

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Sum
total = reduce(lambda acc, x: acc + x, numbers)
print(total)  # 15

# Product
product = reduce(lambda acc, x: acc * x, numbers)
print(product)  # 120

# Initial value
total_with_init = reduce(lambda acc, x: acc + x, numbers, 100)
print(total_with_init)  # 115

# Find maximum
max_val = reduce(lambda a, b: a if a > b else b, numbers)
print(max_val)  # 5
```

### Combining map + filter + reduce

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Sum of squares of even numbers
result = reduce(
    lambda acc, x: acc + x,
    map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers))
)
print(result)  # 220 (4 + 16 + 36 + 64 + 100)
```

---

## 4. lambda Expressions

Define anonymous functions concisely.

### Basic Syntax

```python
# lambda arguments: expression
add = lambda x, y: x + y
print(add(3, 4))  # 7

# Default values
greet = lambda name="World": f"Hello, {name}!"
print(greet())        # Hello, World!
print(greet("Python"))  # Hello, Python!

# Variable arguments
sum_all = lambda *args: sum(args)
print(sum_all(1, 2, 3, 4))  # 10
```

### Using in Sorting

```python
# Sort tuples
points = [(1, 2), (3, 1), (5, 4), (2, 2)]

# Sort by y coordinate
sorted_by_y = sorted(points, key=lambda p: p[1])
print(sorted_by_y)  # [(3, 1), (1, 2), (2, 2), (5, 4)]

# Sort dictionaries
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]
sorted_by_age = sorted(users, key=lambda u: u["age"])
print([u["name"] for u in sorted_by_age])  # ['Bob', 'Alice', 'Charlie']
```

### lambda Cautions

```python
# For complex logic, use regular functions
# Bad
complex_op = lambda x: (x ** 2 + 3 * x - 5) if x > 0 else (x ** 2 - 3 * x + 5)

# Good
def complex_operation(x):
    if x > 0:
        return x ** 2 + 3 * x - 5
    return x ** 2 - 3 * x + 5
```

---

## 5. functools Module

Tools for functional programming.

### partial (Partial Application)

Fix some arguments of a function.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Square function
square = partial(power, exponent=2)
print(square(5))  # 25

# Cube function
cube = partial(power, exponent=3)
print(cube(5))  # 125

# Power of 2
power_of_two = partial(power, 2)
print(power_of_two(10))  # 1024
```

### lru_cache (Memoization)

Cache function results.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Fast thanks to cache
print(fibonacci(100))  # 354224848179261915075

# Cache info
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# Clear cache
fibonacci.cache_clear()
```

### cache (Python 3.9+)

Unlimited cache (equivalent to lru_cache(maxsize=None)).

```python
from functools import cache

@cache
def factorial(n):
    return n * factorial(n - 1) if n else 1

print(factorial(10))  # 3628800
```

### wraps (Preserve Decorator Metadata)

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper docstring"""
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    """Say hello function"""
    return "Hello!"

# Preserve original function info
print(say_hello.__name__)  # say_hello
print(say_hello.__doc__)   # Say hello function
```

### singledispatch (Function Overloading)

Call different functions based on type.

```python
from functools import singledispatch

@singledispatch
def process(value):
    raise NotImplementedError(f"Cannot process {type(value)}")

@process.register(int)
def _(value):
    return f"Processing integer: {value * 2}"

@process.register(str)
def _(value):
    return f"Processing string: {value.upper()}"

@process.register(list)
def _(value):
    return f"Processing list of {len(value)} items"

print(process(10))       # Processing integer: 20
print(process("hello"))  # Processing string: HELLO
print(process([1, 2]))   # Processing list of 2 items
```

---

## 6. operator Module

Use operators as functions.

### Arithmetic Operators

```python
from operator import add, sub, mul, truediv, mod, pow
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Use operator instead of lambda
total = reduce(add, numbers)
print(total)  # 15

product = reduce(mul, numbers)
print(product)  # 120
```

### Comparison Operators

```python
from operator import lt, le, eq, ne, ge, gt

print(lt(3, 5))  # True (3 < 5)
print(eq(3, 3))  # True (3 == 3)
print(ge(5, 3))  # True (5 >= 3)
```

### itemgetter, attrgetter

```python
from operator import itemgetter, attrgetter

# itemgetter - access by index/key
data = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_data = sorted(data, key=itemgetter(1))  # Sort by age
print(sorted_data)  # [('Bob', 25), ('Alice', 30), ('Charlie', 35)]

# Use with dictionaries
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
]
get_name = itemgetter("name")
names = list(map(get_name, users))
print(names)  # ['Alice', 'Bob']

# Multiple fields
get_info = itemgetter("name", "age")
print(get_info(users[0]))  # ('Alice', 30)

# attrgetter - access attributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 30), Person("Bob", 25)]
sorted_people = sorted(people, key=attrgetter("age"))
print([p.name for p in sorted_people])  # ['Bob', 'Alice']
```

### methodcaller

```python
from operator import methodcaller

words = ["hello", "WORLD", "Python"]

# Call method
upper = methodcaller("upper")
print(list(map(upper, words)))  # ['HELLO', 'WORLD', 'PYTHON']

# Method with arguments
replace_o = methodcaller("replace", "o", "0")
print(replace_o("hello world"))  # hell0 w0rld
```

---

## 7. Pure Functions and Immutability

### Pure Functions

```python
# Pure function: no side effects
def pure_add(a, b):
    return a + b

# Impure function: modifies external state
total = 0
def impure_add(value):
    global total
    total += value
    return total

# Impure function: modifies input
def impure_append(lst, value):
    lst.append(value)  # Modifies original
    return lst

# Pure version
def pure_append(lst, value):
    return lst + [value]  # Returns new list
```

### Maintaining Immutability

```python
# Lists: create new list
original = [1, 2, 3]
modified = original + [4]  # Keep original
modified = [*original, 4]  # Unpacking

# Dictionaries: create new dictionary
config = {"debug": True, "port": 8080}
new_config = {**config, "port": 9090}  # Keep original

# Immutable data structures
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p1 = Point(1, 2)
p2 = p1._replace(x=10)  # Create new object
print(p1)  # Point(x=1, y=2)
print(p2)  # Point(x=10, y=2)
```

### frozenset and tuple

```python
# Immutable set
mutable_set = {1, 2, 3}
immutable_set = frozenset([1, 2, 3])

# Can be used as dictionary key
cache = {frozenset([1, 2]): "result"}

# Immutable list (tuple)
mutable_list = [1, 2, 3]
immutable_list = (1, 2, 3)
```

---

## 8. Comprehensions

Pythonic functional patterns.

### List Comprehension

```python
# Replace map
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Replace filter
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# map + filter
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Nested
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Dict/Set Comprehensions

```python
# Dictionary comprehension
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
print(name_lengths)  # {'Alice': 5, 'Bob': 3, 'Charlie': 7}

# Set comprehension
unique_lengths = {len(name) for name in names}
print(unique_lengths)  # {3, 5, 7}

# Reverse dictionary keys/values
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}
```

### Generator Expressions

```python
# Memory-efficient lazy evaluation
large_squares = (x ** 2 for x in range(1000000))
print(next(large_squares))  # 0
print(next(large_squares))  # 1

# Use with sum, any, all
numbers = [1, 2, 3, 4, 5]
total = sum(x ** 2 for x in numbers)  # Parentheses optional
print(total)  # 55

# Conditional checks
all_positive = all(x > 0 for x in numbers)
any_even = any(x % 2 == 0 for x in numbers)
print(all_positive, any_even)  # True True
```

---

## 9. Pipeline Pattern

Connect functions to create data processing pipelines.

### Function Composition

```python
def compose(*functions):
    """Compose functions right to left"""
    def composed(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return composed

def pipe(*functions):
    """Compose functions left to right"""
    def piped(x):
        for func in functions:
            x = func(x)
        return x
    return piped

# Example usage
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x ** 2

# compose: square(double(add_one(5))) = square(double(6)) = square(12) = 144
composed = compose(square, double, add_one)
print(composed(5))  # 144

# pipe: same result but easier to read
piped = pipe(add_one, double, square)
print(piped(5))  # 144
```

### Data Processing Pipeline

```python
from functools import reduce

# Simulate pipe operator
class Pipe:
    def __init__(self, value):
        self.value = value

    def __or__(self, func):
        return Pipe(func(self.value))

    def __repr__(self):
        return repr(self.value)

# Usage
result = (
    Pipe([1, 2, 3, 4, 5])
    | (lambda x: [n * 2 for n in x])      # Double
    | (lambda x: [n for n in x if n > 4]) # Greater than 4
    | sum                                  # Sum
)
print(result)  # 24 (6 + 8 + 10)
```

### Practical Pipeline

```python
from functools import partial

def process_data(data):
    """Data processing pipeline"""
    pipeline = [
        # 1. Clean strings
        lambda items: [s.strip().lower() for s in items],
        # 2. Remove empty strings
        lambda items: [s for s in items if s],
        # 3. Remove duplicates (preserve order)
        lambda items: list(dict.fromkeys(items)),
        # 4. Sort
        sorted,
    ]

    result = data
    for transform in pipeline:
        result = transform(result)
    return result

raw_data = ["  Hello ", "world", "  ", "HELLO", "Python", "world"]
print(process_data(raw_data))  # ['hello', 'python', 'world']
```

---

## 10. Using itertools

### Combinations/Permutations

```python
from itertools import permutations, combinations, product

# Permutations
print(list(permutations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# Combinations
print(list(combinations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 3)]

# Cartesian product
print(list(product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
```

### Grouping

```python
from itertools import groupby

data = [
    ("fruit", "apple"),
    ("fruit", "banana"),
    ("vegetable", "carrot"),
    ("vegetable", "daikon"),
    ("fruit", "elderberry"),
]

# Sort then group (only groups consecutive equal keys)
sorted_data = sorted(data, key=lambda x: x[0])
for key, group in groupby(sorted_data, key=lambda x: x[0]):
    print(f"{key}: {[item[1] for item in group]}")
# fruit: ['apple', 'banana', 'elderberry']
# vegetable: ['carrot', 'daikon']
```

### Infinite Iterators

```python
from itertools import count, cycle, repeat, islice

# count: infinite counter
for i in islice(count(10, 2), 5):
    print(i, end=" ")  # 10 12 14 16 18

# cycle: infinite repetition
colors = cycle(["red", "green", "blue"])
for _ in range(5):
    print(next(colors), end=" ")  # red green blue red green

# repeat: repeat value
threes = list(repeat(3, 4))
print(threes)  # [3, 3, 3, 3]
```

### Chaining and Slicing

```python
from itertools import chain, islice, takewhile, dropwhile

# chain: connect multiple iterables
combined = list(chain([1, 2], [3, 4], [5, 6]))
print(combined)  # [1, 2, 3, 4, 5, 6]

# islice: slicing
first_three = list(islice(range(10), 3))
print(first_three)  # [0, 1, 2]

# takewhile: while condition is true
nums = [1, 3, 5, 8, 2, 4]
result = list(takewhile(lambda x: x < 6, nums))
print(result)  # [1, 3, 5]

# dropwhile: skip while condition is true
result = list(dropwhile(lambda x: x < 6, nums))
print(result)  # [8, 2, 4]
```

---

## 11. Practical Examples

### Data Transformation

```python
from functools import reduce
from operator import itemgetter

# Original data
orders = [
    {"product": "apple", "quantity": 10, "price": 1.5},
    {"product": "banana", "quantity": 5, "price": 0.8},
    {"product": "apple", "quantity": 3, "price": 1.5},
    {"product": "orange", "quantity": 8, "price": 2.0},
]

# Calculate total sales by product
from itertools import groupby

def calculate_sales(orders):
    # Sort and group by product
    sorted_orders = sorted(orders, key=itemgetter("product"))

    result = {}
    for product, group in groupby(sorted_orders, key=itemgetter("product")):
        total = sum(o["quantity"] * o["price"] for o in group)
        result[product] = total

    return result

print(calculate_sales(orders))
# {'apple': 19.5, 'banana': 4.0, 'orange': 16.0}
```

### Functional Validation

```python
from functools import reduce
from typing import Callable, List, Any

def validate(value: Any, *validators: Callable) -> tuple[bool, list[str]]:
    """Apply multiple validators sequentially"""
    errors = []
    for validator in validators:
        result = validator(value)
        if result is not None:
            errors.append(result)
    return len(errors) == 0, errors

# Validator functions
def not_empty(s): return "Cannot be empty" if not s else None
def min_length(n): return lambda s: f"Min length is {n}" if len(s) < n else None
def max_length(n): return lambda s: f"Max length is {n}" if len(s) > n else None

# Usage
is_valid, errors = validate(
    "hi",
    not_empty,
    min_length(3),
    max_length(10)
)
print(is_valid, errors)  # False ['Min length is 3']
```

### Currying

```python
def curry(func):
    """Curry a function"""
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more: curried(*args, *more)
    return curried

@curry
def add_three(a, b, c):
    return a + b + c

# Various call styles
print(add_three(1, 2, 3))     # 6
print(add_three(1)(2)(3))     # 6
print(add_three(1, 2)(3))     # 6
print(add_three(1)(2, 3))     # 6
```

---

## 12. Summary

| Concept | Description |
|---------|-------------|
| First-Class Functions | Pass/return functions as values |
| map/filter/reduce | Higher-order functions for collection transformation |
| lambda | Anonymous functions |
| partial | Partial application |
| lru_cache | Memoization |
| operator | Operators as functions |
| Pure Functions | Functions without side effects |
| Immutability | Create new data instead of modifying |
| Pipeline | Function chaining |

---

## 13. Practice Problems

### Exercise 1: Function Composition

Write a `compose2` function that takes two functions and returns their composition.

### Exercise 2: Transducer

Write a function that combines `map` and `filter` in a single iteration.

### Exercise 3: Memoization

Implement your own memoization decorator.

---

**Previous**: [Async Programming](./08_Async_Programming.md) | **Next**: [Performance Optimization](./10_Performance_Optimization.md)

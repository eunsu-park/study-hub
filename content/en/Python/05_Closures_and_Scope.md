# Closures & Scope

**Previous**: [Iterators & Generators](./04_Iterators_and_Generators.md) | **Next**: [Metaclasses](./06_Metaclasses.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the LEGB rule (Local, Enclosing, Global, Built-in) for variable name resolution
2. Apply the `global` keyword to modify module-level variables from within a function
3. Apply the `nonlocal` keyword to modify enclosing function variables from a nested function
4. Describe the three conditions required to create a closure
5. Implement factory functions, accumulators, and configuration closures
6. Compare closures with callable classes and identify when each approach is preferable
7. Identify and fix the classic "loop variable capture" pitfall with closures
8. Write practical closure-based patterns: memoization, debounce, and retry logic

---

Understanding how Python resolves variable names -- and how inner functions can capture and carry state from their enclosing scope -- is essential for writing decorators, callbacks, and factory functions. Closures turn functions into lightweight, stateful objects without the ceremony of defining a full class, making them the backbone of functional patterns in Python and the mechanism that powers every decorator you write.

## 1. Variable Scope

In Python, variable access scope is determined by where it's defined.

### LEGB Rule

The order in which variables are searched.

```
┌─────────────────────────────────────────────┐
│ B - Built-in                                 │
│   print, len, range, ...                    │
│ ┌─────────────────────────────────────────┐ │
│ │ G - Global                               │ │
│ │   Variables defined at module level      │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │ E - Enclosing (enclosing function)  │ │ │
│ │ │   Local variables of outer function │ │ │
│ │ │ ┌─────────────────────────────────┐ │ │ │
│ │ │ │ L - Local                        │ │ │ │
│ │ │ │   Inside current function        │ │ │ │
│ │ │ └─────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Example

```python
# Built-in (B)
# print, len, str, ...

# Global (G)
x = "global"

def outer():
    # Enclosing (E)
    x = "enclosing"

    def inner():
        # Local (L)
        x = "local"
        print(x)  # local

    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

---

## 2. global Keyword

Used to modify global variables inside functions.

```python
count = 0

def increment():
    global count  # Declare global variable use
    count += 1

increment()
increment()
print(count)  # 2
```

### Without global?

```python
count = 0

def increment():
    count += 1  # UnboundLocalError!
    # count recognized as local variable in count = count + 1

increment()
```

### Read-Only Doesn't Need global

```python
name = "Python"

def greet():
    print(f"Hello, {name}")  # Can read without global

greet()  # Hello, Python
```

---

## 3. nonlocal Keyword

Used to modify variables in enclosing functions.

```python
def outer():
    count = 0

    def inner():
        nonlocal count  # Declare use of outer function's variable
        count += 1

    inner()
    inner()
    print(count)  # 2

outer()
```

### global vs nonlocal

```python
x = "global"

def outer():
    x = "outer"

    def inner():
        nonlocal x  # Modify outer's x
        x = "inner"

    inner()
    print(f"In outer: {x}")  # inner

outer()
print(f"In global: {x}")    # global (unchanged)
```

---

## 4. Closures

> **Analogy:** when a nested function leaves its enclosing scope, it carries captured variables in a backpack (the closure). The backpack travels with the function wherever it goes, so the variables remain accessible even after the outer function has returned.

A closure is a function that remembers the environment (scope) where it was defined.

### Closure Conditions

1. Must have nested functions
2. Inner function must reference outer function's variables
3. Outer function must return inner function

### Basic Example

```python
def make_multiplier(n):
    """Return function that multiplies by n"""
    def multiplier(x):
        return x * n  # Remembers n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
print(double(10))  # 20
```

### Check Closure Variables

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c = make_counter()
print(c.__closure__)  # Check closure cells
print(c.__closure__[0].cell_contents)  # 0

c()
print(c.__closure__[0].cell_contents)  # 1
```

---

## 5. Closure Usage Patterns

### Factory Functions

```python
def make_power(exp):
    """Generate power function"""
    def power(base):
        return base ** exp
    return power

square = make_power(2)
cube = make_power(3)

print(square(4))  # 16
print(cube(4))    # 64
```

### State Maintenance

```python
def make_accumulator(initial=0):
    """Create accumulator"""
    total = initial

    def add(value):
        nonlocal total
        total += value
        return total

    return add

acc = make_accumulator(100)
print(acc(10))   # 110
print(acc(20))   # 130
print(acc(30))   # 160
```

### Configuration Storage

```python
def make_logger(prefix):
    """Create logger with prefix"""
    def log(message):
        print(f"[{prefix}] {message}")
    return log

error_log = make_logger("ERROR")
info_log = make_logger("INFO")

error_log("Problem occurred!")   # [ERROR] Problem occurred!
info_log("Starting")    # [INFO] Starting
```

### Function Customization

```python
def make_formatter(template):
    """Create template-based formatter"""
    def format_data(**kwargs):
        return template.format(**kwargs)
    return format_data

user_format = make_formatter("Name: {name}, Age: {age}")
product_format = make_formatter("{name} - {price} USD")

print(user_format(name="Alice", age=30))
# Name: Alice, Age: 30

print(product_format(name="Apple", price=1000))
# Apple - 1000 USD
```

---

## 6. Closures vs Classes

The same functionality can be implemented with closures or classes.

### Closure Version

```python
def make_counter():
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter

c = make_counter()
print(c())  # 1
print(c())  # 2
```

### Class Version

```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

c = Counter()
print(c())  # 1
print(c())  # 2
```

### When to Use What?

| Situation | Recommendation |
|-----------|----------------|
| Simple state maintenance | Closure |
| Multiple methods needed | Class |
| Inheritance/extension needed | Class |
| Functional style | Closure |
| Complex state management | Class |

---

## 7. Closure Caveats

### Loops and Closures

```python
# Wrong example
functions = []
for i in range(3):
    def f():
        return i
    functions.append(f)

# All return 2! (last i value)
print([f() for f in functions])  # [2, 2, 2]
```

### Solution 1: Use Default Argument

```python
functions = []
for i in range(3):
    def f(x=i):  # Capture value with default argument
        return x
    functions.append(f)

print([f() for f in functions])  # [0, 1, 2]
```

### Solution 2: Wrap with Closure

```python
def make_func(i):
    def f():
        return i
    return f

functions = [make_func(i) for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]
```

### Solution 3: Use lambda

```python
functions = [lambda x=i: x for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]
```

---

## 8. Practical Examples

### Memoization (Caching)

```python
def memoize(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Calculated quickly
```

### Debounce (Limit Consecutive Calls)

```python
import time

def debounce(wait):
    """Ignore consecutive calls within specified time"""
    def decorator(func):
        last_call = [0]

        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_call[0] >= wait:
                last_call[0] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

@debounce(1.0)  # Ignore re-calls within 1 second
def save_data():
    print("Data saved")
```

### Retry Logic

```python
import time

def retry(max_attempts=3, delay=1):
    """Retry on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Retry {attempts}/{max_attempts}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Connection failed")
    return "Success"
```

---

## 9. Scope-Related Functions

### locals() and globals()

```python
x = 10

def func():
    y = 20
    print(f"Local variables: {locals()}")   # {'y': 20}
    print(f"Global variable x: {globals()['x']}")  # 10

func()
```

### vars()

```python
class MyClass:
    def __init__(self):
        self.a = 1
        self.b = 2

obj = MyClass()
print(vars(obj))  # {'a': 1, 'b': 2}
```

---

## 10. Summary

| Keyword/Concept | Description |
|-----------------|-------------|
| LEGB | Variable search order: Local → Enclosing → Global → Built-in |
| global | Use when modifying global variables |
| nonlocal | Use when modifying enclosing function's variables |
| Closure | Inner function that remembers outer function's environment |
| `__closure__` | Check variables referenced by closure |

---

## 11. Practice Problems

### Exercise 1: Counter Factory

Create a counter factory that allows setting start value and increment.

```python
# counter = make_counter(start=10, step=5)
# counter() → 10
# counter() → 15
# counter() → 20
```

### Exercise 2: Function Call History

Create a closure that records function call history.

```python
# tracked_add, get_history = track_calls(add)
# tracked_add(1, 2)
# tracked_add(3, 4)
# get_history() → [(1, 2, 3), (3, 4, 7)]
```

### Exercise 3: Rate Limiter

Create a closure that limits calls per second.

---

**Previous**: [Iterators & Generators](./04_Iterators_and_Generators.md) | **Next**: [Metaclasses](./06_Metaclasses.md)

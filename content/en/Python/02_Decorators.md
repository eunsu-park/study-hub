# Decorators

**Previous**: [Type Hints](./01_Type_Hints.md) | **Next**: [Context Managers](./03_Context_Managers.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how the `@decorator` syntax is syntactic sugar for function wrapping
2. Implement basic decorators that accept `*args` and `**kwargs`
3. Apply `functools.wraps` to preserve the decorated function's metadata
4. Write decorators that accept their own configuration arguments (three-level nesting)
5. Implement class-based decorators using the `__call__` method
6. Distinguish between `@property`, `@staticmethod`, `@classmethod`, and `@lru_cache`
7. Apply decorator chaining and explain the bottom-to-top evaluation order
8. Write class decorators that modify or enhance entire classes (e.g., singleton pattern)

---

Decorators are one of Python's most powerful abstraction mechanisms. Every time you add logging, enforce authentication, cache expensive results, or validate inputs, you face the same problem: cross-cutting logic that clutters the core function. Decorators let you factor out that boilerplate into reusable wrappers, keeping business logic clean while adding capabilities declaratively with a single `@` line. Frameworks like Flask, Django, and pytest rely heavily on decorators for routing, middleware, and test fixtures.

> **Analogy:** a decorator wraps a function the way gift paper wraps a present: the gift inside is unchanged, but the wrapping adds a bow (logging), a card (timing), or security tape (authentication) on the outside.

## 1. What are Decorators?

Decorators are a pattern that adds functionality without modifying the function or class. They use the `@` syntax.

```python
@decorator
def function():
    pass

# The above code is equivalent to:
def function():
    pass
function = decorator(function)
```

### Decorator Structure

```
┌─────────────────────────────────────────┐
│              Decorator                   │
│  ┌─────────────────────────────────┐    │
│  │         wrapper function         │    │
│  │  ┌─────────────────────────┐    │    │
│  │  │     Call original func   │    │    │
│  │  └─────────────────────────┘    │    │
│  │  + Additional features (before/after) │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 2. Basic Decorators

### Simplest Form

```python
def my_decorator(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

Output:
```
Before function execution
Hello!
After function execution
```

### Applied to Functions with Arguments

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(3, 5)
```

Output:
```
Arguments: (3, 5), {}
Result: 8
```

---

## 3. Preserving Metadata with @wraps

When decorators are applied, the original function's metadata (name, docstring, etc.) is lost.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greeting function"""
    return f"Hello, {name}"

print(greet.__name__)  # wrapper (not original name!)
print(greet.__doc__)   # None (docstring lost!)
```

### Using functools.wraps

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Without @wraps, greet.__name__ would be 'wrapper' — breaking help(), logging,
                  # and any introspection-based tools like debuggers or documentation generators
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greeting function"""
    return f"Hello, {name}"

print(greet.__name__)  # greet (preserved!)
print(greet.__doc__)   # Greeting function (preserved!)
```

---

## 4. Decorators with Arguments

To pass arguments to the decorator itself, you need one more level of wrapping.

```python
from functools import wraps

def repeat(times):
    """Decorator that repeats function execution n times"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hi():
    print("Hi!")

say_hi()
```

Output:
```
Hi!
Hi!
Hi!
```

### Practical Example: Permission Check

```python
from functools import wraps

def require_role(role):
    """Apply to functions that require specific role"""
    def decorator(func):
        @wraps(func)
        def wrapper(user, *args, **kwargs):
            if user.get("role") != role:
                raise PermissionError(f"'{role}' permission required")
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_role("admin")
def delete_user(user, target_id):
    print(f"User {target_id} deleted")

admin = {"name": "Alice", "role": "admin"}
guest = {"name": "Bob", "role": "guest"}

delete_user(admin, 123)  # OK
# delete_user(guest, 123)  # PermissionError!
```

---

## 5. Class-Based Decorators

Classes can be used as decorators by implementing the `__call__` method.

```python
class CountCalls:
    """Decorator that tracks function call count"""

    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} call count: {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()  # Call count: 1
say_hello()  # Call count: 2
say_hello()  # Call count: 3
```

### Class Decorator with Arguments

```python
class Retry:
    """Decorator that retries on failure.

    Why a decorator instead of inline try/except?  Retry logic is cross-cutting:
    every fragile call site would need the same boilerplate loop.  A decorator
    keeps the call sites clean and lets you change the retry policy in one place.
    """

    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt == self.max_attempts:
                        raise  # Re-raise the original exception on final attempt
        return wrapper

@Retry(max_attempts=3)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure!")
    return "Success!"

result = unstable_function()
```

---

## 6. Built-in Decorators

### @property

Define getter/setter/deleter.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """Radius (read)"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Radius (write)"""
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value

    @property
    def area(self):
        """Area (computed property)"""
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.radius)  # 5
print(circle.area)    # 78.53975

circle.radius = 10
print(circle.area)    # 314.159
```

### @staticmethod

Method that doesn't access instance or class.

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def multiply(a, b):
        return a * b

# Call without instance
print(Math.add(3, 5))       # 8
print(Math.multiply(3, 5))  # 15
```

### @classmethod

Method that receives class as first argument.

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_string):
        """Create Date from string"""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """Create Date with today's date"""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

date1 = Date.from_string("2024-01-23")
date2 = Date.today()
print(date1)  # Date(2024, 1, 23)
```

---

## 7. Practical Decorator Patterns

### Timing Measurement

```python
import time
from functools import wraps

def timer(func):
    """Measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function: 1.0012s
```

### Logging

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_calls(func):
    """Log function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        logging.info(f"Returned: {result}")
        return result
    return wrapper

@log_calls
def add(a, b):
    return a + b

add(3, 5)
```

### Caching (Memoization)

```python
from functools import wraps

def memoize(func):
    """Decorator that caches results"""
    cache = {}  # Closure-captured dict persists across calls; keys are arg tuples
                # (must be hashable — lists/dicts as args would fail)

    @wraps(func)
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

print(fibonacci(100))  # Very slow without caching
```

**Note**: Use Python's built-in `functools.lru_cache` for more convenience.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### Input Validation

```python
from functools import wraps

def validate_types(**expected_types):
    """Decorator that validates argument types"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate keyword arguments
            for name, expected in expected_types.items():
                if name in kwargs:
                    if not isinstance(kwargs[name], expected):
                        raise TypeError(
                            f"{name} must be of type {expected.__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

create_user(name="Alice", age=30)  # OK
# create_user(name="Alice", age="30")  # TypeError!
```

---

## 8. Decorator Chaining

Multiple decorators can be applied simultaneously. Application order is **bottom to top**.

```python
@decorator1
@decorator2
@decorator3
def func():
    pass

# The above code is equivalent to:
func = decorator1(decorator2(decorator3(func)))
```

### Example

```python
from functools import wraps

def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}"

print(greet("World"))  # <b><i>Hello, World</i></b>
```

---

## 9. Class Decorators

Decorators can be applied to entire classes.

```python
def singleton(cls):
    """Singleton pattern decorator"""
    instances = {}  # Keyed by class object (not string name) to support subclasses independently

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # Creating database connection
db2 = Database()  # (no output - same instance)
print(db1 is db2)  # True
```

### dataclass (Built-in Class Decorator)

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
print(p)              # Point(x=3, y=4)
print(p.distance())   # 5.0
print(p == Point(3, 4))  # True
```

---

## 10. Summary

| Pattern | Description | Example |
|---------|-------------|---------|
| Basic decorator | Wrap function to add functionality | `@timer` |
| Decorator with args | Pass configuration to decorator | `@repeat(3)` |
| Class-based decorator | When state needs to be maintained | `@CountCalls` |
| @wraps | Preserve metadata | `@wraps(func)` |
| @property | Define getter/setter | `@property` |
| @staticmethod | Static method | `@staticmethod` |
| @classmethod | Class method | `@classmethod` |
| @lru_cache | Cache results | `@lru_cache(128)` |

---

## 11. Practice Problems

### Exercise 1: Execution Time Limit

Create a decorator that raises TimeoutError if the function doesn't complete within a specified time.

### Exercise 2: Result Logging

Create a decorator that logs both input and output of a function to a file.

### Exercise 3: Debug Mode

Create a decorator that outputs debug information only when a DEBUG flag is True.

---

**Previous**: [Type Hints](./01_Type_Hints.md) | **Next**: [Context Managers](./03_Context_Managers.md)

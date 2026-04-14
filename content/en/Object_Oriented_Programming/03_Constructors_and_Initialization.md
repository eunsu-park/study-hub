# Lesson 03: Constructors and Initialization

## Learning Objectives

By the end of this lesson, you will be able to:
1. Write `__init__` methods with various parameter patterns
2. Validate constructor arguments and raise appropriate exceptions
3. Use default values and optional parameters effectively
4. Implement the builder pattern for complex object construction
5. Understand the difference between `__new__` and `__init__`
6. Create immutable-style objects through careful initialization
7. Apply common initialization patterns and anti-patterns

## The `__init__` Method

The `__init__` method is Python's **initializer** (often called the constructor, though technically `__new__` is the true constructor). It is called automatically after the object is created, allowing you to set up initial state.

```
    Car("Toyota", 2024)
          │
          ▼
   ┌──────────────┐
   │ Python calls  │
   │ Car.__new__() │  ── allocates memory, returns empty object
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │ Python calls  │
   │ Car.__init__()│  ── sets up attributes on the object
   └──────┬───────┘
          │
          ▼
   Object is ready to use
```

### Basic `__init__`

```python
class Point:
    """A 2D point."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


p = Point(3, 4)
print(p)  # Point(3, 4)
```

### The `self` Parameter

`self` is a reference to the **current instance** being initialized. It is always the first parameter of instance methods, but you never pass it explicitly — Python handles that.

```python
class Greeter:
    def __init__(self, name):
        # self.name creates an instance attribute
        # name (without self) is just the parameter
        self.name = name

    def greet(self):
        # self.name accesses the instance attribute
        return f"Hello, {self.name}!"


g = Greeter("World")
# Python translates g.greet() to Greeter.greet(g)
print(g.greet())  # Hello, World!
```

## Default Values

Default values let callers omit arguments that have sensible defaults:

```python
class Connection:
    """A database connection with sensible defaults."""

    def __init__(self, host="localhost", port=5432, database="mydb",
                 timeout=30, max_retries=3):
        self.host = host
        self.port = port
        self.database = database
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_connected = False

    def __repr__(self):
        return f"Connection({self.host}:{self.port}/{self.database})"


# All defaults
c1 = Connection()                           # localhost:5432/mydb

# Override some
c2 = Connection(host="prod-db", port=5433)  # prod-db:5433/mydb

# Override all
c3 = Connection("remote", 3306, "sales", 60, 5)
```

### The Mutable Default Argument Trap

One of the most common Python gotchas:

```python
# BAD: Mutable default argument
class BadCollector:
    def __init__(self, items=[]):  # DON'T DO THIS
        self.items = items

a = BadCollector()
a.items.append("x")

b = BadCollector()
print(b.items)  # ['x']  -- Surprise! The list is SHARED


# GOOD: Use None as default
class GoodCollector:
    def __init__(self, items=None):
        self.items = items if items is not None else []

a = GoodCollector()
a.items.append("x")

b = GoodCollector()
print(b.items)  # []  -- Each instance gets its own list
```

```
┌─────────────────────────────────────────────┐
│  Rule: NEVER use mutable defaults           │
│  (list, dict, set) in function signatures.  │
│  Use None and create inside __init__.       │
└─────────────────────────────────────────────┘
```

## Validation in `__init__`

Always validate inputs early — fail fast with clear error messages:

```python
class Temperature:
    """A temperature value with validation."""

    ABSOLUTE_ZERO_C = -273.15
    VALID_SCALES = ("C", "F", "K")

    def __init__(self, value, scale="C"):
        # Validate scale
        if scale not in self.VALID_SCALES:
            raise ValueError(
                f"Invalid scale '{scale}'. Must be one of {self.VALID_SCALES}"
            )

        # Validate value type
        if not isinstance(value, (int, float)):
            raise TypeError(f"Temperature value must be numeric, got {type(value).__name__}")

        # Validate physical constraints
        celsius = self._to_celsius(value, scale)
        if celsius < self.ABSOLUTE_ZERO_C:
            raise ValueError(
                f"Temperature {value}{scale} is below absolute zero "
                f"({self.ABSOLUTE_ZERO_C}C)"
            )

        self._value = value
        self._scale = scale

    @staticmethod
    def _to_celsius(value, scale):
        if scale == "C":
            return value
        elif scale == "F":
            return (value - 32) * 5 / 9
        elif scale == "K":
            return value - 273.15

    @property
    def celsius(self):
        return self._to_celsius(self._value, self._scale)

    def __repr__(self):
        return f"Temperature({self._value}{self._scale})"


# Valid
t1 = Temperature(100, "C")
t2 = Temperature(32, "F")
t3 = Temperature(0, "K")

# Invalid — clear errors
# Temperature(-300, "C")  -> ValueError: below absolute zero
# Temperature(100, "X")   -> ValueError: Invalid scale 'X'
# Temperature("hot")      -> TypeError: must be numeric
```

## Initialization Patterns

### Pattern 1: Keyword-Only Arguments

Force callers to use keyword arguments for clarity:

```python
class Config:
    """Configuration with keyword-only arguments."""

    def __init__(self, *, debug=False, verbose=False, log_file=None,
                 max_workers=4):
        # The * means ALL arguments must be passed as keywords
        self.debug = debug
        self.verbose = verbose
        self.log_file = log_file
        self.max_workers = max_workers


# This is clear and self-documenting
config = Config(debug=True, max_workers=8)

# This raises TypeError — no positional args allowed
# config = Config(True, False, None, 8)
```

### Pattern 2: Alternative Constructors with `@classmethod`

```python
class Vector:
    """A 2D vector with multiple construction methods."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_polar(cls, r, theta):
        """Create a vector from polar coordinates."""
        import math
        return cls(r * math.cos(theta), r * math.sin(theta))

    @classmethod
    def from_tuple(cls, coords):
        """Create a vector from a (x, y) tuple."""
        if len(coords) != 2:
            raise ValueError("Tuple must have exactly 2 elements")
        return cls(*coords)

    @classmethod
    def zero(cls):
        """Create a zero vector."""
        return cls(0, 0)

    def __repr__(self):
        return f"Vector({self.x:.2f}, {self.y:.2f})"


import math
v1 = Vector(3, 4)
v2 = Vector.from_polar(5, math.pi / 4)
v3 = Vector.from_tuple((1, 2))
v4 = Vector.zero()
```

### Pattern 3: Builder Pattern

For objects with many optional configuration parameters:

```python
class Pizza:
    """A pizza built with the builder pattern."""

    def __init__(self, size):
        self.size = size
        self.cheese = False
        self.pepperoni = False
        self.mushrooms = False
        self.onions = False
        self.extra_sauce = False

    def add_cheese(self):
        self.cheese = True
        return self  # Return self for chaining

    def add_pepperoni(self):
        self.pepperoni = True
        return self

    def add_mushrooms(self):
        self.mushrooms = True
        return self

    def add_onions(self):
        self.onions = True
        return self

    def add_extra_sauce(self):
        self.extra_sauce = True
        return self

    def describe(self):
        toppings = []
        if self.cheese: toppings.append("cheese")
        if self.pepperoni: toppings.append("pepperoni")
        if self.mushrooms: toppings.append("mushrooms")
        if self.onions: toppings.append("onions")
        extras = " + extra sauce" if self.extra_sauce else ""
        return f"{self.size} pizza with {', '.join(toppings or ['plain'])}{extras}"


# Fluent builder interface (method chaining)
pizza = (Pizza("large")
         .add_cheese()
         .add_pepperoni()
         .add_mushrooms()
         .add_extra_sauce())

print(pizza.describe())
# large pizza with cheese, pepperoni, mushrooms + extra sauce
```

### Pattern 4: Post-Init Processing

Sometimes you need to compute derived attributes after the basic initialization:

```python
class Rectangle:
    """A rectangle with computed properties."""

    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        self.width = width
        self.height = height
        # Derived attributes computed during init
        self._area = width * height
        self._perimeter = 2 * (width + height)
        self._diagonal = (width ** 2 + height ** 2) ** 0.5

    @property
    def area(self):
        return self._area

    @property
    def perimeter(self):
        return self._perimeter

    @property
    def diagonal(self):
        return self._diagonal

    def __repr__(self):
        return f"Rectangle({self.width}x{self.height})"
```

## `__new__` vs `__init__`

Most of the time you only need `__init__`, but understanding `__new__` is important:

```
┌─────────────────────────────────────────────┐
│  __new__(cls, ...)                          │
│  - Called FIRST                              │
│  - Allocates memory                          │
│  - Returns a new instance                    │
│  - Rarely overridden (except singletons,     │
│    immutable types)                          │
├─────────────────────────────────────────────┤
│  __init__(self, ...)                        │
│  - Called SECOND                             │
│  - Receives the already-created instance     │
│  - Sets up attributes                        │
│  - Returns None (always)                     │
│  - Overridden in almost every class          │
└─────────────────────────────────────────────┘
```

```python
class Singleton:
    """A class that only allows one instance.

    Uses __new__ to control instance creation.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        # __init__ is called every time, but on the SAME instance
        if value is not None:
            self.value = value


s1 = Singleton(42)
s2 = Singleton(99)
print(s1 is s2)    # True — same instance
print(s1.value)    # 99 — __init__ was called again with 99
```

## Common Mistakes

### Mistake 1: Forgetting `self`

```python
class Broken:
    def __init__(self, name):
        name = name  # This does NOTHING — local variable assigned to itself

class Fixed:
    def __init__(self, name):
        self.name = name  # This creates an instance attribute
```

### Mistake 2: Returning from `__init__`

```python
class Bad:
    def __init__(self, x):
        self.x = x
        return self  # TypeError! __init__ must return None

class Good:
    def __init__(self, x):
        self.x = x
        # Implicit return None
```

### Mistake 3: Heavy Work in `__init__`

```python
# BAD: Doing too much in __init__
class BadReport:
    def __init__(self, data_path):
        self.data = self._load_data(data_path)        # I/O
        self.analysis = self._run_analysis(self.data)   # CPU-heavy
        self._send_notification()                       # Side effect

# GOOD: Lazy initialization
class GoodReport:
    def __init__(self, data_path):
        self.data_path = data_path
        self._data = None
        self._analysis = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._load_data(self.data_path)
        return self._data

    def analyze(self):
        """Explicitly trigger analysis."""
        self._analysis = self._run_analysis(self.data)
        return self._analysis
```

## Summary

- `__init__` is the initializer called after object creation; it sets up instance attributes
- `self` refers to the current instance — always use `self.attr` to create instance attributes
- Use `None` as default for mutable arguments (lists, dicts, sets)
- Validate inputs in `__init__` and fail fast with clear error messages
- Use `@classmethod` for alternative constructors
- Use keyword-only arguments (`*`) for clarity with many parameters
- Understand `__new__` vs `__init__`: creation vs initialization
- Avoid heavy I/O or side effects in `__init__` — prefer lazy initialization

## Next Steps

In [Lesson 04: Encapsulation](04_Encapsulation.md), we will learn how to protect an object's internal state and expose controlled interfaces.

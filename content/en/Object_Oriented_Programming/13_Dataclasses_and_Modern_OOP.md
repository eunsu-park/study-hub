# Lesson 13: Dataclasses and Modern OOP

## Learning Objectives

By the end of this lesson, you will be able to:
1. Use `@dataclass` to eliminate boilerplate in data-holding classes
2. Configure dataclass options: frozen, ordering, slots
3. Use `field()` for default factories and metadata
4. Create immutable value objects with frozen dataclasses
5. Use `NamedTuple` for lightweight immutable records
6. Apply `Protocol` for structural typing without inheritance
7. Choose between dataclasses, NamedTuples, and regular classes

## The Boilerplate Problem

Traditional Python classes require repetitive code:

```python
# Traditional class — lots of boilerplate
class PointOld:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"PointOld({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        if not isinstance(other, PointOld):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))
```

## Dataclasses: The Modern Solution

```python
from dataclasses import dataclass


@dataclass
class Point:
    """Same functionality, minimal code."""
    x: float
    y: float
    z: float = 0.0


# Auto-generated: __init__, __repr__, __eq__
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)

print(p1)          # Point(x=1.0, y=2.0, z=0.0)
print(p1 == p2)    # True (auto __eq__)
print(p1.x)        # 1.0
```

### What `@dataclass` Generates

```
┌──────────────────────────────────────────────────┐
│  @dataclass generates:                           │
│                                                  │
│  __init__()    - from field definitions           │
│  __repr__()    - "ClassName(field1=val1, ...)"   │
│  __eq__()      - compares all fields             │
│                                                  │
│  Optional (with parameters):                     │
│  __hash__()    - if frozen=True or unsafe_hash   │
│  __lt__(), __le__(), __gt__(), __ge__()          │
│                - if order=True                   │
│  __slots__     - if slots=True (Python 3.10+)    │
└──────────────────────────────────────────────────┘
```

## Dataclass Options

```python
from dataclasses import dataclass, field


# Frozen: immutable (generates __hash__ too)
@dataclass(frozen=True)
class Color:
    red: int
    green: int
    blue: int

    def hex(self):
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"


c = Color(255, 128, 0)
print(c.hex())      # #ff8000
# c.red = 0         # FrozenInstanceError!

# Can be used as dict key or in sets (hashable)
colors = {Color(255, 0, 0): "red", Color(0, 255, 0): "green"}


# Ordering: generates comparison operators
@dataclass(order=True)
class Student:
    gpa: float
    name: str = ""  # Included in comparisons but gpa comes first

students = [Student(3.5, "Alice"), Student(3.8, "Bob"), Student(3.5, "Charlie")]
print(sorted(students))
# [Student(gpa=3.5, name='Alice'), Student(gpa=3.5, name='Charlie'), Student(gpa=3.8, name='Bob')]


# Slots: more memory-efficient (Python 3.10+)
@dataclass(slots=True)
class Coordinate:
    x: float
    y: float
```

## The `field()` Function

For advanced field configuration:

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class ShoppingCart:
    owner: str
    items: List[str] = field(default_factory=list)  # Mutable default!
    _total: float = field(default=0.0, repr=False)  # Hidden from repr
    created: str = field(default="", init=False)     # Not in __init__

    def __post_init__(self):
        """Called after __init__ — for derived/computed values."""
        from datetime import datetime
        self.created = datetime.now().isoformat()

    def add_item(self, item, price):
        self.items.append(item)
        self._total += price

    @property
    def total(self):
        return self._total


cart = ShoppingCart("Alice")
cart.add_item("Book", 29.99)
cart.add_item("Pen", 4.99)

print(cart)         # ShoppingCart(owner='Alice', items=['Book', 'Pen'], created='...')
print(cart.total)   # 34.98
```

### Field Options

```
┌──────────────────┬───────────────────────────────────┐
│  Option          │  Description                      │
├──────────────────┼───────────────────────────────────┤
│ default          │ Default value for the field       │
│ default_factory  │ Callable for mutable defaults     │
│ init             │ Include in __init__? (default True)│
│ repr             │ Include in __repr__? (default True)│
│ compare          │ Include in __eq__? (default True) │
│ hash             │ Include in __hash__?              │
│ kw_only          │ Keyword-only arg (Python 3.10+)  │
└──────────────────┴───────────────────────────────────┘
```

## `__post_init__`: Validation and Computed Fields

```python
from dataclasses import dataclass, field


@dataclass
class Temperature:
    celsius: float

    def __post_init__(self):
        """Validate after automatic __init__."""
        if self.celsius < -273.15:
            raise ValueError(f"Temperature {self.celsius}C is below absolute zero")

    @property
    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

    @property
    def kelvin(self):
        return self.celsius + 273.15


t = Temperature(100)
print(t.fahrenheit)  # 212.0
print(t.kelvin)      # 373.15
# Temperature(-300)  # ValueError!
```

## Dataclass Inheritance

```python
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int


@dataclass
class Employee(Person):
    company: str
    salary: float = 50000.0


emp = Employee("Alice", 30, "Acme Corp", 75000)
print(emp)  # Employee(name='Alice', age=30, company='Acme Corp', salary=75000.0)
```

## NamedTuple

`NamedTuple` creates lightweight, immutable records — useful when you want tuple-like behavior with named fields:

```python
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0


p = Point(1.0, 2.0)
print(p)         # Point(x=1.0, y=2.0, z=0.0)
print(p.x)       # 1.0 (named access)
print(p[0])      # 1.0 (index access — it's a tuple!)

# Immutable
# p.x = 5.0     # AttributeError!

# Tuple operations work
x, y, z = p     # Unpacking
print(len(p))    # 3

# Can be used as dict keys (hashable)
points = {Point(0, 0): "origin", Point(1, 0): "unit x"}
```

### NamedTuple with Methods

```python
from typing import NamedTuple
from math import sqrt


class Vector2D(NamedTuple):
    x: float
    y: float

    @property
    def magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self):
        mag = self.magnitude
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other):
        return self.x * other.x + self.y * other.y


v = Vector2D(3.0, 4.0)
print(v.magnitude)     # 5.0
print(v.normalized())  # Vector2D(x=0.6, y=0.8)
```

## Protocol: Structural Typing

Protocols define interfaces through structure rather than inheritance:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Renderable(Protocol):
    """Anything that can be rendered to a string."""

    def render(self) -> str:
        ...


@runtime_checkable
class Sized(Protocol):
    """Anything with a size."""

    def __len__(self) -> int:
        ...


# These classes don't inherit from Renderable — they just match the structure
class HTMLElement:
    def __init__(self, tag, content):
        self.tag = tag
        self.content = content

    def render(self) -> str:
        return f"<{self.tag}>{self.content}</{self.tag}>"


class MarkdownHeader:
    def __init__(self, text, level=1):
        self.text = text
        self.level = level

    def render(self) -> str:
        return f"{'#' * self.level} {self.text}"


def render_all(items: list[Renderable]) -> str:
    """Works with ANY object that has a render() method."""
    return "\n".join(item.render() for item in items)


elements = [
    HTMLElement("h1", "Hello"),
    MarkdownHeader("World", 2),
]

print(render_all(elements))
# <h1>Hello</h1>
# ## World

# Runtime check
print(isinstance(HTMLElement("p", "hi"), Renderable))  # True
```

## Choosing the Right Tool

```
┌─────────────────┬──────────────────────────────────────┐
│  Tool           │  Best For                            │
├─────────────────┼──────────────────────────────────────┤
│ Regular class   │ Complex behavior, mutable state,     │
│                 │ custom __init__, non-data classes     │
├─────────────────┼──────────────────────────────────────┤
│ @dataclass      │ Data-holding classes with behavior,  │
│                 │ mutable by default, rich features    │
├─────────────────┼──────────────────────────────────────┤
│ @dataclass      │ Value objects, dict keys, config     │
│ (frozen=True)   │ objects, immutable records            │
├─────────────────┼──────────────────────────────────────┤
│ NamedTuple      │ Lightweight immutable records,       │
│                 │ tuple compatibility, simple data     │
├─────────────────┼──────────────────────────────────────┤
│ Protocol        │ Structural interfaces, duck typing   │
│                 │ with type checker support             │
├─────────────────┼──────────────────────────────────────┤
│ dict            │ Dynamic/unknown keys, JSON data,     │
│                 │ throwaway structures                  │
└─────────────────┴──────────────────────────────────────┘
```

## Practical Example: Configuration System

```python
from dataclasses import dataclass, field, asdict
from typing import NamedTuple, Protocol
import json


@dataclass(frozen=True)
class DatabaseConfig:
    """Immutable database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "mydb"
    pool_size: int = 5


@dataclass(frozen=True)
class CacheConfig:
    host: str = "localhost"
    port: int = 6379
    ttl: int = 300


@dataclass
class AppConfig:
    """Application configuration combining sub-configs."""
    name: str
    debug: bool = False
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


config = AppConfig(
    name="MyApp",
    debug=True,
    db=DatabaseConfig(host="prod-db", port=5433),
)
print(config.to_json())
```

## Summary

- `@dataclass` eliminates boilerplate by auto-generating `__init__`, `__repr__`, `__eq__`
- Use `frozen=True` for immutable dataclasses (value objects, dict keys)
- Use `field()` for mutable defaults, hidden fields, and computed values
- `__post_init__` runs after `__init__` for validation and derived attributes
- `NamedTuple` creates immutable, tuple-compatible records
- `Protocol` enables structural typing — matching interfaces without inheritance
- Choose the right tool: regular class for complex behavior, dataclass for data, NamedTuple for immutable records, Protocol for interfaces

## Next Steps

In [Lesson 14: OOP Best Practices](14_OOP_Best_Practices.md), we will cover anti-patterns to avoid and practical guidelines for writing clean OOP code.

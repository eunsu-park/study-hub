# 13. Dataclasses

**Previous**: [Packaging & Distribution](./12_Packaging_and_Distribution.md) | **Next**: [Pattern Matching](./14_Pattern_Matching.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of the `dataclasses` module and the boilerplate it eliminates compared to regular classes
2. Use the `@dataclass` decorator with options such as `order`, `frozen`, `slots`, and `kw_only`
3. Apply the `field()` function to configure default factories, exclude fields from repr/compare, and attach metadata
4. Implement `__post_init__` and `InitVar` for computed fields and initialization-only parameters
5. Use utility functions (`asdict`, `astuple`, `replace`, `fields`) to convert and inspect dataclass instances
6. Apply dataclass inheritance and composition patterns while avoiding default-value ordering pitfalls
7. Compare `dataclasses` with Pydantic and identify the appropriate choice for different use cases

---

Python classes that primarily hold data often require a surprising amount of repetitive code: `__init__`, `__repr__`, `__eq__`, and more. The `dataclasses` module, introduced in Python 3.7, generates these methods automatically from type-annotated field declarations, letting you focus on the data itself rather than the plumbing. Understanding dataclasses is essential for writing concise, correct, and maintainable data-oriented code.

## Table of Contents
1. [Dataclass Basics](#1-dataclass-basics)
2. [@dataclass Options](#2-dataclass-options)
3. [field() Function](#3-field-function)
4. [Advanced Features](#4-advanced-features)
5. [Inheritance and Composition](#5-inheritance-and-composition)
6. [Pydantic Comparison](#6-pydantic-comparison)
7. [Practice Problems](#7-practice-problems)

---

## 1. Dataclass Basics

### 1.1 What are Dataclasses?

```python
# Regular class for data structure
class PersonManual:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"PersonManual(name={self.name!r}, age={self.age}, email={self.email!r})"

    def __eq__(self, other):
        if not isinstance(other, PersonManual):
            return NotImplemented
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)


# Same functionality with dataclass (concise!)
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int
    email: str
```

### 1.2 Auto-Generated Methods

```python
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


# Auto-generated methods:

# 1. __init__
p = Point(3.0, 4.0)

# 2. __repr__
print(p)  # Point(x=3.0, y=4.0)

# 3. __eq__
p1 = Point(3.0, 4.0)
p2 = Point(3.0, 4.0)
print(p1 == p2)  # True

# All fields used for comparison
p3 = Point(3.0, 5.0)
print(p1 == p3)  # False
```

### 1.3 Setting Default Values

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    name: str
    email: str
    age: int = 0                    # Default value
    active: bool = True             # Default value
    nickname: Optional[str] = None  # Optional with default


# Fields with defaults must come after those without
user = User("Alice", "alice@example.com")
print(user)  # User(name='Alice', email='alice@example.com', age=0, active=True, nickname=None)

user2 = User("Bob", "bob@example.com", age=30, nickname="bobby")
```

---

## 2. @dataclass Options

### 2.1 Main Options

```python
@dataclass(
    init=True,          # Generate __init__ (default: True)
    repr=True,          # Generate __repr__ (default: True)
    eq=True,            # Generate __eq__ (default: True)
    order=False,        # Generate comparison methods (default: False)
    unsafe_hash=False,  # Generate __hash__ (default: False)
    frozen=False,       # Immutable object (default: False)
    match_args=True,    # Pattern matching support (Python 3.10+)
    kw_only=False,      # All fields keyword-only (Python 3.10+)
    slots=False,        # Use __slots__ (Python 3.10+)
)
class MyClass:
    pass
```

### 2.2 order - Comparison Operators

```python
@dataclass(order=True)
class Student:
    name: str
    grade: float
    age: int


students = [
    Student("Alice", 3.8, 20),
    Student("Bob", 3.5, 22),
    Student("Charlie", 3.8, 21),
]

# Sorting (compares all fields in order)
sorted_students = sorted(students)
# Compares by name first, then grade, then age

# Sort by specific field
sorted_by_grade = sorted(students, key=lambda s: s.grade, reverse=True)
```

### 2.3 frozen - Immutable Objects

```python
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float


point = ImmutablePoint(1.0, 2.0)

# Error on modification attempt
try:
    point.x = 3.0
except AttributeError as e:
    print(f"Error: {e}")  # cannot assign to field 'x'

# Immutable means hashable (can use as dict key, set element)
points = {point: "origin"}
point_set = {point, ImmutablePoint(3.0, 4.0)}
```

### 2.4 slots - Memory Optimization

```python
@dataclass(slots=True)  # Python 3.10+
class OptimizedPoint:
    x: float
    y: float


# slots=True advantages:
# 1. Reduced memory usage
# 2. Faster attribute access
# 3. No __dict__ (cannot add dynamic attributes)

point = OptimizedPoint(1.0, 2.0)
# point.z = 3.0  # AttributeError - not in slots


# Memory comparison
import sys

@dataclass
class RegularPoint:
    x: float
    y: float

regular = RegularPoint(1.0, 2.0)
optimized = OptimizedPoint(1.0, 2.0)

print(sys.getsizeof(regular.__dict__))  # dict size
# OptimizedPoint has no __dict__
```

### 2.5 kw_only - Keyword-Only Arguments

```python
@dataclass(kw_only=True)  # Python 3.10+
class Config:
    host: str
    port: int
    debug: bool = False


# Must pass as keyword arguments
config = Config(host="localhost", port=8080)
# config = Config("localhost", 8080)  # TypeError
```

---

## 3. field() Function

### 3.1 Basic field() Usage

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class Team:
    name: str
    members: List[str] = field(default_factory=list)  # Mutable default
    score: int = field(default=0)


# Wrong way (direct mutable assignment)
# members: List[str] = []  # All instances share same list!

team1 = Team("Alpha")
team2 = Team("Beta")
team1.members.append("Alice")
print(team2.members)  # [] - correctly separated
```

### 3.2 field() Options

```python
from dataclasses import dataclass, field


@dataclass
class Product:
    # Basic field
    name: str

    # Default factory
    tags: list = field(default_factory=list)

    # Exclude from repr
    internal_id: str = field(repr=False, default="")

    # Exclude from comparison
    cache: dict = field(compare=False, default_factory=dict)

    # Exclude from __init__ (set in post-processing)
    computed: str = field(init=False)

    # Exclude from hash (with frozen=True)
    mutable_data: list = field(hash=False, default_factory=list)

    # Add metadata
    price: float = field(metadata={"unit": "USD", "min": 0})

    def __post_init__(self):
        self.computed = f"{self.name}_computed"


product = Product("Widget", price=9.99)
print(product)  # internal_id hidden
print(product.computed)  # Widget_computed

# Access metadata
from dataclasses import fields
for f in fields(Product):
    if f.metadata:
        print(f"{f.name}: {f.metadata}")
```

### 3.3 Using __post_init__

```python
from dataclasses import dataclass, field


@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)


rect = Rectangle(5, 3)
print(f"Area: {rect.area}")  # 15
print(f"Perimeter: {rect.perimeter}")  # 16
```

### 3.4 InitVar - Initialization-Only Variables

```python
from dataclasses import dataclass, field, InitVar


@dataclass
class User:
    name: str
    email: str
    password: InitVar[str]  # Only for init, not stored as field
    password_hash: str = field(init=False)

    def __post_init__(self, password: str):
        import hashlib
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()


user = User("Alice", "alice@example.com", "secret123")
print(user)  # No password, has password_hash
# User(name='Alice', email='alice@example.com', password_hash='...')

# user.password  # AttributeError - InitVar not stored
```

---

## 4. Advanced Features

### 4.1 Dataclass Utility Functions

```python
from dataclasses import dataclass, field, fields, asdict, astuple, replace


@dataclass
class Person:
    name: str
    age: int
    email: str = ""


person = Person("Alice", 30, "alice@example.com")

# fields(): Query field information
for f in fields(person):
    print(f"Name: {f.name}, Type: {f.type}, Default: {f.default}")

# asdict(): Convert to dictionary
person_dict = asdict(person)
print(person_dict)  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# astuple(): Convert to tuple
person_tuple = astuple(person)
print(person_tuple)  # ('Alice', 30, 'alice@example.com')

# replace(): Create new instance with some fields changed
person2 = replace(person, name="Bob", age=25)
print(person2)  # Person(name='Bob', age=25, email='alice@example.com')
```

### 4.2 Custom __hash__

```python
@dataclass(eq=True, frozen=True)
class HashableItem:
    id: int
    name: str
    # frozen=True automatically generates __hash__


# Or manual hash
@dataclass(eq=True)
class CustomHashItem:
    id: int
    name: str
    data: list  # Mutable field

    def __hash__(self):
        return hash(self.id)  # Hash by id only


item = CustomHashItem(1, "test", [1, 2, 3])
items_set = {item}  # Hashable
```

### 4.3 JSON Serialization

```python
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from typing import List


@dataclass
class Event:
    name: str
    date: datetime
    attendees: List[str]


def serialize_event(event: Event) -> str:
    def encoder(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(asdict(event), default=encoder)


def deserialize_event(json_str: str) -> Event:
    data = json.loads(json_str)
    data["date"] = datetime.fromisoformat(data["date"])
    return Event(**data)


event = Event("Meeting", datetime.now(), ["Alice", "Bob"])
json_str = serialize_event(event)
print(json_str)

restored = deserialize_event(json_str)
print(restored)
```

---

## 5. Inheritance and Composition

### 5.1 Dataclass Inheritance

```python
from dataclasses import dataclass


@dataclass
class Animal:
    name: str
    age: int


@dataclass
class Dog(Animal):
    breed: str
    is_trained: bool = False


dog = Dog("Buddy", 3, "Labrador")
print(dog)  # Dog(name='Buddy', age=3, breed='Labrador', is_trained=False)
```

### 5.2 Default Values and Inheritance Gotchas

```python
from dataclasses import dataclass, field


@dataclass
class Base:
    x: int
    y: int = 0  # Has default


# Error! Field without default cannot come after field with default
# @dataclass
# class Derived(Base):
#     z: int  # Error!

# Solution 1: Provide default
@dataclass
class Derived1(Base):
    z: int = 0


# Solution 2: Use kw_only (Python 3.10+)
@dataclass
class Derived2(Base):
    z: int = field(kw_only=True, default=0)
```

### 5.3 Composition

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str


@dataclass
class ContactInfo:
    email: str
    phone: str


@dataclass
class Employee:
    name: str
    employee_id: str
    address: Address          # Composition
    contact: ContactInfo      # Composition
    skills: List[str] = field(default_factory=list)


employee = Employee(
    name="Alice",
    employee_id="E001",
    address=Address("123 Main St", "Seoul", "Korea", "12345"),
    contact=ContactInfo("alice@company.com", "010-1234-5678"),
    skills=["Python", "SQL"]
)
```

### 5.4 Mixin Pattern

```python
from dataclasses import dataclass, field
from datetime import datetime


class TimestampMixin:
    """Timestamp functionality mixin"""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def touch(self):
        self.updated_at = datetime.now()


@dataclass
class Article(TimestampMixin):
    title: str
    content: str
    author: str


article = Article("Hello", "World", "Alice")
print(article.created_at)
article.touch()
print(article.updated_at)
```

---

## 6. Pydantic Comparison

### 6.1 dataclass vs Pydantic

```python
# dataclass - simple data structures
from dataclasses import dataclass


@dataclass
class UserDataclass:
    name: str
    age: int
    email: str


# No type validation!
user = UserDataclass("Alice", "not an int", "invalid-email")
print(user)  # Created without error


# Pydantic - runtime validation
from pydantic import BaseModel, EmailStr, Field


class UserPydantic(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: EmailStr


try:
    user = UserPydantic(name="Alice", age="25", email="alice@example.com")
    print(user)  # age automatically converted to int
    print(user.age, type(user.age))  # 25 <class 'int'>
except Exception as e:
    print(f"Validation error: {e}")
```

### 6.2 Selection Criteria

```
┌─────────────────────────────────────────────────────────────────┐
│                 dataclass vs Pydantic Selection                 │
│                                                                 │
│   Choose dataclass:                                             │
│   - Internal data structures                                    │
│   - Performance is critical                                     │
│   - Simple data containers                                      │
│   - Minimize external dependencies                              │
│   - No type validation needed                                   │
│                                                                 │
│   Choose Pydantic:                                              │
│   - API input validation                                        │
│   - Configuration file parsing                                  │
│   - External data processing                                    │
│   - JSON serialization/deserialization                          │
│   - Complex validation logic                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Pydantic dataclass

```python
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field


@pydantic_dataclass
class Person:
    """Pydantic validation + dataclass interface"""
    name: str
    age: int = Field(ge=0)


# Validation performed
try:
    person = Person(name="Alice", age=-1)  # ValidationError
except Exception as e:
    print(e)

# Valid data
person = Person(name="Bob", age=30)
print(person)
```

---

## 7. Practice Problems

### Exercise 1: Immutable Config Class
Write an immutable (frozen) configuration class.

```python
# Sample solution
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    version: str
    debug: bool = False
    settings: tuple = field(default_factory=tuple)  # Immutable collection

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            app_name=data["app_name"],
            version=data["version"],
            debug=data.get("debug", False),
            settings=tuple(data.get("settings", [])),
        )


config = AppConfig("MyApp", "1.0.0")
config_dict = {"app_name": "MyApp", "version": "1.0.0", "debug": True}
config2 = AppConfig.from_dict(config_dict)
```

### Exercise 2: Auto-Calculated Fields
Write a Circle class that auto-calculates area and circumference.

```python
# Sample solution
from dataclasses import dataclass, field
import math


@dataclass
class Circle:
    radius: float
    area: float = field(init=False)
    circumference: float = field(init=False)

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("Radius cannot be negative")
        self.area = math.pi * self.radius ** 2
        self.circumference = 2 * math.pi * self.radius


circle = Circle(5)
print(f"Area: {circle.area:.2f}")  # ~78.54
print(f"Circumference: {circle.circumference:.2f}")  # ~31.42
```

### Exercise 3: Nested Dataclasses and JSON
Serialize/deserialize nested dataclasses to/from JSON.

```python
# Sample solution
from dataclasses import dataclass, asdict, field
from typing import List
import json


@dataclass
class OrderItem:
    product_name: str
    quantity: int
    price: float


@dataclass
class Order:
    order_id: str
    customer: str
    items: List[OrderItem] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Order":
        data = json.loads(json_str)
        items = [OrderItem(**item) for item in data.pop("items", [])]
        return cls(**data, items=items)


order = Order(
    "ORD001",
    "Alice",
    [OrderItem("Book", 2, 15.99), OrderItem("Pen", 5, 1.99)]
)
json_str = order.to_json()
print(json_str)

restored = Order.from_json(json_str)
print(restored)
```

---

## References
- [Python dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
- [PEP 557 - Data Classes](https://peps.python.org/pep-0557/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [attrs Library](https://www.attrs.org/) (alternative)

---

**Previous**: [Packaging & Distribution](./12_Packaging_and_Distribution.md) | **Next**: [Pattern Matching](./14_Pattern_Matching.md)

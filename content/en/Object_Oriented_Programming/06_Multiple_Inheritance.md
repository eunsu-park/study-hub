# Lesson 06: Multiple Inheritance

## Learning Objectives

By the end of this lesson, you will be able to:
1. Define classes that inherit from multiple parents
2. Explain the Method Resolution Order (MRO) and how Python uses C3 linearization
3. Identify and resolve the diamond problem
4. Design mixins to add reusable behavior without deep hierarchies
5. Use `super()` correctly in multiple inheritance scenarios
6. Apply best practices for multiple inheritance

## What Is Multiple Inheritance?

Multiple inheritance allows a class to inherit from **more than one parent class**, combining attributes and methods from all parents.

```
┌──────────┐     ┌──────────┐
│  Flyer   │     │ Swimmer  │
│ fly()    │     │ swim()   │
└────┬─────┘     └────┬─────┘
     │                │
     └────┬───────────┘
          │
     ┌────▼─────┐
     │  Duck    │
     │ fly()   │  inherited from Flyer
     │ swim()  │  inherited from Swimmer
     │ quack() │  own method
     └─────────┘
```

```python
class Flyer:
    def fly(self):
        return f"{self.__class__.__name__} is flying!"

class Swimmer:
    def swim(self):
        return f"{self.__class__.__name__} is swimming!"

class Duck(Flyer, Swimmer):
    def quack(self):
        return "Quack! Quack!"


donald = Duck()
print(donald.fly())    # Duck is flying!
print(donald.swim())   # Duck is swimming!
print(donald.quack())  # Quack! Quack!
```

## The Diamond Problem

The diamond problem occurs when a class inherits from two classes that share a common ancestor:

```
        ┌───────────┐
        │  Animal   │
        │  __init__ │
        │  eat()    │
        └─────┬─────┘
         ┌────┴────┐
         │         │
    ┌────▼───┐ ┌───▼────┐
    │ Flyer  │ │Swimmer │
    │ fly()  │ │ swim() │
    └────┬───┘ └───┬────┘
         │         │
         └────┬────┘
         ┌────▼────┐
         │  Duck   │   <-- Should Animal.__init__ be called once or twice?
         └─────────┘
```

```python
class Animal:
    def __init__(self, name):
        print(f"Animal.__init__({name})")
        self.name = name

    def eat(self):
        return f"{self.name} is eating"


class Flyer(Animal):
    def __init__(self, name, wingspan):
        print(f"Flyer.__init__({name})")
        super().__init__(name)
        self.wingspan = wingspan

    def fly(self):
        return f"{self.name} flies with {self.wingspan}cm wingspan"


class Swimmer(Animal):
    def __init__(self, name, swim_speed):
        print(f"Swimmer.__init__({name})")
        super().__init__(name)
        self.swim_speed = swim_speed

    def swim(self):
        return f"{self.name} swims at {self.swim_speed} km/h"


class Duck(Flyer, Swimmer):
    def __init__(self, name, wingspan, swim_speed):
        print(f"Duck.__init__({name})")
        super().__init__(name, wingspan)
        # How does swim_speed get set? We need cooperative __init__!

    def quack(self):
        return f"{self.name} says Quack!"
```

### Solving It with Cooperative `super()`

The proper solution uses `**kwargs` to pass arguments through the MRO chain:

```python
class Animal:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)  # Pass remaining kwargs up
        self.name = name

class Flyer(Animal):
    def __init__(self, wingspan=0, **kwargs):
        super().__init__(**kwargs)
        self.wingspan = wingspan

class Swimmer(Animal):
    def __init__(self, swim_speed=0, **kwargs):
        super().__init__(**kwargs)
        self.swim_speed = swim_speed

class Duck(Flyer, Swimmer):
    def __init__(self, name, wingspan, swim_speed):
        super().__init__(name=name, wingspan=wingspan, swim_speed=swim_speed)


donald = Duck("Donald", wingspan=60, swim_speed=5)
print(donald.name)        # Donald
print(donald.wingspan)    # 60
print(donald.swim_speed)  # 5
```

## Method Resolution Order (MRO)

When you call a method on an object, Python needs to determine which class's method to use. It follows the **Method Resolution Order** — a linearization of the inheritance graph using the **C3 algorithm**.

```python
class A:
    def who(self):
        return "A"

class B(A):
    def who(self):
        return "B"

class C(A):
    def who(self):
        return "C"

class D(B, C):
    pass

# What does D().who() return?
print(D().who())  # "B"

# MRO explains why:
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

### MRO Visualization

```
D.__mro__:  D -> B -> C -> A -> object

Search order for D().who():
  D: not defined -> skip
  B: found! -> return "B"

Search order for D().method_only_in_A():
  D: skip -> B: skip -> C: skip -> A: found!
```

### MRO Rules (C3 Linearization)

1. A class always comes before its parents
2. If a class inherits from multiple parents, they maintain their order
3. A common parent comes after ALL its children in the MRO

```python
# Invalid MRO — Python raises TypeError
class X:
    pass

class Y(X):
    pass

# This would violate C3 rules:
# class Z(X, Y):  # TypeError: Cannot create a consistent MRO
#     pass

# Because X should come before Y (as Y's parent),
# but Z(X, Y) says X should come first in the list
```

## Mixins

A **mixin** is a class designed to provide specific behavior to other classes through multiple inheritance. Mixins are not meant to stand alone — they add a focused capability.

```
┌─────────────────────────────────────────────────┐
│  Mixin Rules:                                   │
│  1. Never instantiate a mixin directly          │
│  2. A mixin should provide a single capability  │
│  3. A mixin should not have __init__            │
│     (or use cooperative **kwargs)               │
│  4. Name it with a "Mixin" suffix              │
└─────────────────────────────────────────────────┘
```

```python
import json
from datetime import datetime


class SerializableMixin:
    """Adds JSON serialization capability."""

    def to_json(self):
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


class TimestampMixin:
    """Adds creation and modification timestamps."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kw):
            original_init(self, *args, **kw)
            self.created_at = datetime.now().isoformat()
            self.modified_at = self.created_at

        cls.__init__ = new_init

    def touch(self):
        """Update the modification timestamp."""
        self.modified_at = datetime.now().isoformat()


class LoggableMixin:
    """Adds logging capability."""

    def log(self, message, level="INFO"):
        class_name = self.__class__.__name__
        print(f"[{level}] {class_name}: {message}")


class ComparableMixin:
    """Adds comparison operators based on a `_compare_key` method."""

    def _compare_key(self):
        raise NotImplementedError("Subclass must define _compare_key()")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._compare_key() == other._compare_key()

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._compare_key() < other._compare_key()

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


# Combine mixins to build a feature-rich class
class Product(SerializableMixin, LoggableMixin, ComparableMixin):
    """A product with serialization, logging, and comparison."""

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def _compare_key(self):
        return self.price

    def __repr__(self):
        return f"Product({self.name!r}, ${self.price})"


# Usage
laptop = Product("Laptop", 999)
phone = Product("Phone", 699)

laptop.log("Created new product")   # [INFO] Product: Created new product
print(laptop.to_json())             # {"name": "Laptop", "price": 999}
print(laptop > phone)               # True (999 > 699)
print(sorted([laptop, phone]))      # [Product('Phone', $699), Product('Laptop', $999)]
```

## Real-World Example: Django-Style Mixins

```python
class LoginRequiredMixin:
    """Ensures the user is authenticated before accessing a view."""

    def dispatch(self, request, *args, **kwargs):
        if not request.get("user"):
            return {"status": 403, "error": "Login required"}
        return super().dispatch(request, *args, **kwargs)


class PermissionRequiredMixin:
    """Ensures the user has the required permission."""

    required_permission = None

    def dispatch(self, request, *args, **kwargs):
        user = request.get("user", {})
        if self.required_permission not in user.get("permissions", []):
            return {"status": 403, "error": "Permission denied"}
        return super().dispatch(request, *args, **kwargs)


class View:
    """Base view class."""

    def dispatch(self, request, *args, **kwargs):
        method = request.get("method", "GET").lower()
        handler = getattr(self, method, self.method_not_allowed)
        return handler(request, *args, **kwargs)

    def method_not_allowed(self, request, *args, **kwargs):
        return {"status": 405, "error": "Method not allowed"}


class AdminDashboard(LoginRequiredMixin, PermissionRequiredMixin, View):
    """A protected admin view combining multiple mixins."""

    required_permission = "admin"

    def get(self, request, *args, **kwargs):
        return {"status": 200, "data": "Admin Dashboard"}
```

## Best Practices

1. **Prefer composition over deep multiple inheritance** — mixins are fine, complex diamond hierarchies are not
2. **Use `super()` cooperatively** — always pass `**kwargs` in mixin `__init__` methods
3. **Keep mixin classes focused** — one capability per mixin
4. **Name mixins clearly** — use the `Mixin` suffix
5. **Avoid state in mixins** — mixins should add behavior, not data
6. **Check MRO** — use `ClassName.__mro__` or `help(ClassName)` to verify resolution order

## Summary

- Multiple inheritance lets a class combine behavior from multiple parents
- The **diamond problem** occurs when two parents share a common ancestor
- Python resolves method lookups using the **MRO** (C3 linearization)
- **Mixins** are the recommended pattern for multiple inheritance: focused, stateless behavior units
- Use cooperative `super()` with `**kwargs` for proper initialization chains
- Always verify the MRO with `ClassName.__mro__`

## Next Steps

In [Lesson 07: Polymorphism](07_Polymorphism.md), we will explore how the same interface can produce different behavior depending on the object type.

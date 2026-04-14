# Lesson 07: Polymorphism

## Learning Objectives

By the end of this lesson, you will be able to:
1. Define polymorphism and explain why it is essential to OOP
2. Implement subtype polymorphism through method overriding
3. Apply duck typing — Python's preferred polymorphism approach
4. Overload operators using magic methods
5. Use protocols for structural typing (Python 3.8+)
6. Distinguish between parametric, ad-hoc, and subtype polymorphism
7. Design polymorphic interfaces for extensible systems

## What Is Polymorphism?

Polymorphism (Greek: "many forms") means that the **same interface** can produce **different behavior** depending on the type of object. A single function call like `shape.area()` works correctly whether `shape` is a Circle, Rectangle, or Triangle — each computes its area differently.

```
           shape.area()
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼────┐ ┌──▼─────┐ ┌──▼──────┐
│ Circle │ │ Rect-  │ │Triangle │
│        │ │ angle  │ │         │
│ pi*r^2 │ │ w * h  │ │ 0.5*b*h │
└────────┘ └────────┘ └─────────┘

Same method name, different implementations
```

## Subtype Polymorphism (Inheritance-Based)

The most traditional form: subclasses override a parent method:

```python
from math import pi


class Shape:
    """Base class for geometric shapes."""

    def area(self):
        raise NotImplementedError("Subclasses must implement area()")

    def perimeter(self):
        raise NotImplementedError("Subclasses must implement perimeter()")

    def describe(self):
        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2

    def perimeter(self):
        return 2 * pi * self.radius


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeter(self):
        return self.a + self.b + self.c


# Polymorphism: same interface, different behavior
shapes = [Circle(5), Rectangle(4, 6), Triangle(3, 4, 5)]

for shape in shapes:
    print(shape.describe())
# Circle: area=78.54, perimeter=31.42
# Rectangle: area=24.00, perimeter=20.00
# Triangle: area=6.00, perimeter=12.00

# Functions that work with ANY shape
def total_area(shapes):
    """Works with any objects that have an area() method."""
    return sum(s.area() for s in shapes)

print(f"Total area: {total_area(shapes):.2f}")  # Total area: 108.54
```

## Duck Typing

Python's preferred approach to polymorphism: "If it walks like a duck and quacks like a duck, it's a duck." No inheritance required — just implement the right methods.

```python
# No shared base class needed!
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Robot:
    def speak(self):
        return "Beep boop!"

class Duck:
    def speak(self):
        return "Quack!"


# This function works with ANY object that has speak()
def make_them_speak(things):
    for thing in things:
        print(f"{thing.__class__.__name__}: {thing.speak()}")


# No common parent, but polymorphism works!
make_them_speak([Dog(), Cat(), Robot(), Duck()])
# Dog: Woof!
# Cat: Meow!
# Robot: Beep boop!
# Duck: Quack!
```

### Duck Typing in the Standard Library

Python's built-in functions use duck typing extensively:

```python
# len() works with anything that has __len__
class Playlist:
    def __init__(self, songs):
        self.songs = songs

    def __len__(self):
        return len(self.songs)

playlist = Playlist(["Song A", "Song B", "Song C"])
print(len(playlist))  # 3

# iter() works with anything that has __iter__
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        current = self.start
        while current > 0:
            yield current
            current -= 1

for n in Countdown(5):
    print(n, end=" ")  # 5 4 3 2 1
```

## Operator Overloading

Python lets you define how operators (`+`, `-`, `*`, `==`, `<`, etc.) work with your objects:

```python
class Vector:
    """A 2D vector with operator overloading."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Addition: v1 + v2
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    # Subtraction: v1 - v2
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    # Scalar multiplication: v * 3
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    # Reverse multiplication: 3 * v
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    # Magnitude: abs(v)
    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    # Equality: v1 == v2
    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    # Negation: -v
    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)       # Vector(4, 6)
print(v1 - v2)       # Vector(2, 2)
print(v1 * 3)        # Vector(9, 12)
print(3 * v1)        # Vector(9, 12)
print(abs(v1))        # 5.0
print(-v1)            # Vector(-3, -4)
print(v1 == Vector(3, 4))  # True
```

### Common Operators to Overload

```
┌────────────────┬────────────────┬──────────────────┐
│  Operator      │  Method        │  Example         │
├────────────────┼────────────────┼──────────────────┤
│  +             │  __add__       │  a + b           │
│  -             │  __sub__       │  a - b           │
│  *             │  __mul__       │  a * b           │
│  /             │  __truediv__   │  a / b           │
│  //            │  __floordiv__  │  a // b          │
│  **            │  __pow__       │  a ** b          │
│  ==            │  __eq__        │  a == b          │
│  !=            │  __ne__        │  a != b          │
│  <             │  __lt__        │  a < b           │
│  <=            │  __le__        │  a <= b          │
│  >             │  __gt__        │  a > b           │
│  >=            │  __ge__        │  a >= b          │
│  len()         │  __len__       │  len(a)          │
│  str()         │  __str__       │  str(a)          │
│  bool()        │  __bool__      │  bool(a)         │
│  []            │  __getitem__   │  a[key]          │
│  in            │  __contains__  │  x in a          │
└────────────────┴────────────────┴──────────────────┘
```

## Protocols (Structural Typing)

Python 3.8+ introduced `Protocol` for formal duck typing — defining expected interfaces without inheritance:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Drawable(Protocol):
    """Any object with a draw() method."""

    def draw(self) -> str:
        ...


@runtime_checkable
class Resizable(Protocol):
    """Any object with a resize() method."""

    def resize(self, factor: float) -> None:
        ...


# These classes don't inherit from Drawable, but satisfy the protocol
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"

    def resize(self, factor: float) -> None:
        self.radius *= factor


class TextBox:
    def __init__(self, text):
        self.text = text

    def draw(self) -> str:
        return f"Drawing text: {self.text}"

    # No resize() — only satisfies Drawable, not Resizable


def render(item: Drawable) -> None:
    """Works with anything that has draw()."""
    print(item.draw())


# Runtime checking
print(isinstance(Circle(5), Drawable))   # True
print(isinstance(TextBox("hi"), Drawable))  # True
print(isinstance(Circle(5), Resizable))  # True
print(isinstance(TextBox("hi"), Resizable))  # False
```

## Types of Polymorphism

```
┌─────────────────────────────────────────────────────────┐
│                    POLYMORPHISM                         │
├──────────────────┬──────────────────┬───────────────────┤
│ Subtype          │ Ad-hoc           │ Parametric        │
│ (Inheritance)    │ (Overloading)    │ (Generics)        │
├──────────────────┼──────────────────┼───────────────────┤
│ class Dog(Animal)│ def add(a, b)    │ def first(items:  │
│   def speak():   │   # works with  │   list[T]) -> T   │
│     "Woof"       │   # int, float, │                   │
│                  │   # str, Vector  │                   │
├──────────────────┼──────────────────┼───────────────────┤
│ Dog IS Animal    │ Same name,       │ Works with any    │
│ Cat IS Animal    │ different types  │ type parameter    │
└──────────────────┴──────────────────┴───────────────────┘
```

## Practical Example: Payment Processing

```python
class PaymentProcessor:
    """Process payments polymorphically."""

    def charge(self, amount):
        raise NotImplementedError

    def refund(self, amount):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class CreditCard(PaymentProcessor):
    def __init__(self, card_number):
        self.card_number = card_number[-4:]  # Store last 4 only

    def charge(self, amount):
        return f"Charged ${amount:.2f} to card ending in {self.card_number}"

    def refund(self, amount):
        return f"Refunded ${amount:.2f} to card ending in {self.card_number}"


class PayPal(PaymentProcessor):
    def __init__(self, email):
        self.email = email

    def charge(self, amount):
        return f"Charged ${amount:.2f} via PayPal ({self.email})"

    def refund(self, amount):
        return f"Refunded ${amount:.2f} via PayPal ({self.email})"


class Cryptocurrency(PaymentProcessor):
    def __init__(self, wallet_address):
        self.wallet = wallet_address[:8] + "..."

    def charge(self, amount):
        return f"Sent ${amount:.2f} in crypto to {self.wallet}"

    def refund(self, amount):
        return f"Returned ${amount:.2f} in crypto from {self.wallet}"


def process_order(payment: PaymentProcessor, amount: float):
    """This function doesn't know or care about the payment type."""
    print(f"Processing order for ${amount:.2f}")
    print(payment.charge(amount))
    print("Order complete!\n")


# Same function, different payment types
process_order(CreditCard("4111111111111234"), 99.99)
process_order(PayPal("alice@example.com"), 49.99)
process_order(Cryptocurrency("0xABCDEF1234567890"), 29.99)
```

## Summary

- Polymorphism means "same interface, different behavior" depending on the object type
- **Subtype polymorphism**: Subclasses override parent methods (classic OOP)
- **Duck typing**: Python's preferred style — no inheritance needed, just matching method names
- **Operator overloading**: Define `__add__`, `__eq__`, etc. for custom operator behavior
- **Protocols**: Formal duck typing (Python 3.8+) for type-checked structural interfaces
- Polymorphism enables extensible code — add new types without changing existing functions

## Next Steps

In [Lesson 08: Abstraction](08_Abstraction.md), we will learn how to define abstract interfaces that enforce a contract on subclasses.

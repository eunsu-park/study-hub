"""
Example 07: Polymorphism
Topic: Object-Oriented Programming

Demonstrates subtype polymorphism, duck typing, operator overloading,
and Protocol-based structural typing.
"""

from math import pi, sqrt
from typing import Protocol, runtime_checkable


# =============================================================================
# SUBTYPE POLYMORPHISM
# =============================================================================

class Shape:
    """Base class for shapes."""

    def area(self):
        raise NotImplementedError

    def perimeter(self):
        raise NotImplementedError

    def describe(self):
        return (f"{self.__class__.__name__}: "
                f"area={self.area():.2f}, perimeter={self.perimeter():.2f}")


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
        return sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def perimeter(self):
        return self.a + self.b + self.c


# =============================================================================
# OPERATOR OVERLOADING
# =============================================================================

class Vector:
    """2D vector with full operator overloading."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __abs__(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __hash__(self):
        return hash((self.x, self.y))

    def dot(self, other):
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def normalized(self):
        """Return unit vector."""
        mag = abs(self)
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


# =============================================================================
# DUCK TYPING
# =============================================================================

class CSVExporter:
    def export(self, data):
        return ",".join(str(v) for v in data)

class JSONExporter:
    def export(self, data):
        import json
        return json.dumps(data)

class XMLExporter:
    def export(self, data):
        items = "".join(f"<item>{v}</item>" for v in data)
        return f"<data>{items}</data>"


def export_data(exporter, data):
    """Works with any object that has an export() method — duck typing."""
    return exporter.export(data)


# =============================================================================
# PROTOCOL (STRUCTURAL TYPING)
# =============================================================================

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str: ...


class TerminalCircle:
    def __init__(self, radius):
        self.radius = radius

    def draw(self) -> str:
        return f"( o ) radius={self.radius}"


class TerminalBox:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def draw(self) -> str:
        top = "+" + "-" * self.width + "+"
        mid = ("|" + " " * self.width + "|\n") * self.height
        return f"{top}\n{mid}{top}"


if __name__ == "__main__":
    # Subtype polymorphism
    print("=== Subtype Polymorphism ===")
    shapes = [Circle(5), Rectangle(4, 6), Triangle(3, 4, 5)]
    for s in shapes:
        print(s.describe())
    print(f"Total area: {sum(s.area() for s in shapes):.2f}")

    # Operator overloading
    print("\n=== Operator Overloading ===")
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 3 = {v1 * 3}")
    print(f"|v1| = {abs(v1):.2f}")
    print(f"v1 . v2 = {v1.dot(v2)}")
    print(f"v1 normalized = {v1.normalized()}")

    # Duck typing
    print("\n=== Duck Typing ===")
    data = [1, 2, 3, 4, 5]
    for exp in [CSVExporter(), JSONExporter(), XMLExporter()]:
        print(f"{exp.__class__.__name__}: {export_data(exp, data)}")

    # Protocol
    print("\n=== Protocol (Structural Typing) ===")
    drawables = [TerminalCircle(3), TerminalBox(5, 2)]
    for d in drawables:
        print(f"Is Drawable? {isinstance(d, Drawable)}")
        print(d.draw())
        print()

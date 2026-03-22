"""
Exercise 09: OOP Advanced

Implement abstract base classes, operator overloading, and a linked list.
"""

from abc import ABC, abstractmethod
import math


class Shape(ABC):
    """Abstract base class for shapes.

    Subclasses must implement:
        area() -> float
        perimeter() -> float

    Provides:
        __lt__(other): Compare by area for sorting.
        __repr__(): Return "{ClassName}(...)"
    """

    @abstractmethod
    def area(self):
        """Return the area of the shape."""
        pass

    @abstractmethod
    def perimeter(self):
        """Return the perimeter of the shape."""
        pass

    def __lt__(self, other):
        return self.area() < other.area()


class Circle(Shape):
    """A circle shape.

    Args:
        radius: Radius of the circle.

    __repr__: "Circle(radius={radius})"
    """

    # TODO: Implement __init__, area, perimeter, __repr__
    pass


class Triangle(Shape):
    """A triangle shape defined by three side lengths.

    Args:
        a, b, c: Side lengths.

    area: Use Heron's formula.
    __repr__: "Triangle(a={a}, b={b}, c={c})"
    """

    # TODO: Implement __init__, area (Heron's formula), perimeter, __repr__
    pass


class Vector:
    """A 2D vector with operator overloading.

    Supports:
        __add__(other): Vector addition.
        __sub__(other): Vector subtraction.
        __mul__(scalar): Scalar multiplication (Vector * scalar).
        __eq__(other): Component-wise equality.
        __abs__(): Return magnitude (Euclidean length).
        __repr__(): "Vector({x}, {y})"
        dot(other): Dot product of two vectors.
    """

    # TODO: Implement this class
    pass


class Node:
    """A node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next_node = next_node


class LinkedList:
    """A singly linked list.

    Methods:
        append(data): Add element to end.
        prepend(data): Add element to front.
        __len__(): Return number of elements.
        __contains__(data): Check if data exists in list.
        to_list(): Return a Python list of all elements.
        __repr__(): "LinkedList([elem1, elem2, ...])"
    """

    # TODO: Implement using Node class above
    pass


# === Tests ===

# Shape hierarchy
c = Circle(5)
assert round(c.area(), 2) == 78.54, "Circle area"
assert round(c.perimeter(), 2) == 31.42, "Circle perimeter"
assert repr(c) == "Circle(radius=5)", "Circle repr"

t = Triangle(3, 4, 5)
assert t.area() == 6.0, "Triangle area (3-4-5)"
assert t.perimeter() == 12, "Triangle perimeter"

# Shapes are sortable by area
shapes = [Circle(5), Triangle(3, 4, 5), Circle(1)]
shapes.sort()
assert shapes[0].area() < shapes[1].area() < shapes[2].area(), "Sort by area"

# Cannot instantiate Shape directly
try:
    Shape()
    assert False, "Should not instantiate ABC"
except TypeError:
    pass

# Vector operations
v1 = Vector(3, 4)
v2 = Vector(1, 2)
assert v1 + v2 == Vector(4, 6), "Vector add"
assert v1 - v2 == Vector(2, 2), "Vector sub"
assert v1 * 2 == Vector(6, 8), "Vector scalar mul"
assert abs(v1) == 5.0, "Vector magnitude"
assert v1.dot(v2) == 11, "Dot product"
assert repr(v1) == "Vector(3, 4)", "Vector repr"

# LinkedList
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
assert len(ll) == 3, "LinkedList length"
assert 2 in ll, "LinkedList contains"
assert 99 not in ll, "LinkedList not contains"
assert ll.to_list() == [1, 2, 3], "LinkedList to_list"
ll.prepend(0)
assert ll.to_list() == [0, 1, 2, 3], "LinkedList prepend"

print("All tests passed!")

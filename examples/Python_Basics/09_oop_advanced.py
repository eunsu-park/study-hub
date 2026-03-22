"""
09 OOP Advanced
===============
Demonstrates inheritance, abstract base classes, operator overloading,
dataclasses, composition, and the MRO (Method Resolution Order).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import total_ordering
import math


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class Shape(ABC):
    """Abstract base class for geometric shapes."""

    @abstractmethod
    def area(self) -> float:
        """Calculate the area."""

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter."""

    def describe(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"area={self.area():.2f}, perimeter={self.perimeter():.2f}"
        )


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Square(Rectangle):
    """Square is a specialized Rectangle."""

    def __init__(self, side):
        super().__init__(side, side)


def inheritance_demo():
    """Show inheritance hierarchy and abstract classes."""
    shapes = [Circle(5), Rectangle(4, 6), Square(3)]

    for shape in shapes:
        print(f"  {shape.describe()}")

    # isinstance and issubclass
    sq = Square(4)
    print(f"\nSquare is Rectangle? {isinstance(sq, Rectangle)}")
    print(f"Square is Shape?     {isinstance(sq, Shape)}")
    print(f"Square subclass of Rectangle? {issubclass(Square, Rectangle)}")

    # Cannot instantiate abstract class
    try:
        Shape()
    except TypeError as e:
        print(f"\nCannot instantiate ABC: {e}")


# ---------------------------------------------------------------------------
# MRO and Multiple Inheritance
# ---------------------------------------------------------------------------

class Loggable:
    """Mixin class providing logging capability."""
    def log(self, message):
        print(f"  [{self.__class__.__name__}] {message}")


class Serializable:
    """Mixin class providing serialization."""
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class User(Loggable, Serializable):
    """Demonstrates multiple inheritance with mixins."""
    def __init__(self, name, email):
        self.name = name
        self.email = email


def mro_demo():
    """Method Resolution Order with multiple inheritance."""
    user = User("Alice", "alice@example.com")
    user.log("User created")
    print(f"  Serialized: {user.to_dict()}")

    # MRO shows lookup order
    print(f"\n  MRO: {[cls.__name__ for cls in User.__mro__]}")

    # Diamond problem example
    class A:
        def method(self):
            return "A"

    class B(A):
        def method(self):
            return "B"

    class C(A):
        def method(self):
            return "C"

    class D(B, C):
        pass

    d = D()
    print(f"\n  Diamond: D.method() = {d.method()!r}")
    print(f"  D MRO: {[cls.__name__ for cls in D.__mro__]}")


# ---------------------------------------------------------------------------
# Operator Overloading
# ---------------------------------------------------------------------------

@total_ordering
class Vector:
    """2D vector with operator overloading."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # Arithmetic operators
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    # Comparison (total_ordering fills in the rest)
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.magnitude < other.magnitude

    # Container protocol
    def __getitem__(self, index):
        return (self.x, self.y)[index]

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


def operator_overloading_demo():
    """Demonstrate operator overloading on Vector."""
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)

    print(f"  v1 = {v1},  v2 = {v2}")
    print(f"  v1 + v2  = {v1 + v2}")
    print(f"  v1 - v2  = {v1 - v2}")
    print(f"  v1 * 3   = {v1 * 3}")
    print(f"  2 * v2   = {2 * v2}")
    print(f"  -v1      = {-v1}")
    print(f"  |v1|     = {v1.magnitude:.2f}")

    # Comparison (by magnitude via @total_ordering)
    print(f"\n  v1 == v2: {v1 == v2}")
    print(f"  v1 > v2:  {v1 > v2}")

    # Iteration and indexing
    print(f"\n  v1[0] = {v1[0]}, v1[1] = {v1[1]}")
    x, y = v1
    print(f"  Unpacked: x={x}, y={y}")
    print(f"  List: {list(v1)}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """Simple dataclass — auto-generates __init__, __repr__, __eq__."""
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass(order=True, frozen=True)
class Student:
    """Ordered, immutable dataclass."""
    sort_index: float = field(init=False, repr=False)
    name: str
    gpa: float
    courses: tuple = field(default_factory=tuple)

    def __post_init__(self):
        # frozen=True requires object.__setattr__ for post_init
        object.__setattr__(self, "sort_index", self.gpa)


def dataclass_demo():
    """Show dataclass features."""
    p1 = Point(1, 2)
    p2 = Point(4, 6)
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  Distance: {p1.distance_to(p2):.2f}")
    print(f"  Equal: {p1 == Point(1, 2)}")

    # Ordered and frozen
    students = [
        Student("Alice", 3.9, ("Math", "CS")),
        Student("Bob", 3.5, ("History",)),
        Student("Charlie", 3.8, ("Physics", "Math")),
    ]

    print("\n  Sorted students:")
    for s in sorted(students, reverse=True):
        print(f"    {s.name}: GPA {s.gpa}")

    # Frozen: immutable
    try:
        students[0].name = "Modified"
    except AttributeError as e:
        print(f"\n  Frozen prevents mutation: {e}")


# ---------------------------------------------------------------------------
# Composition over Inheritance
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False

    def start(self):
        self.running = True
        return f"Engine ({self.horsepower}hp) started"

    def stop(self):
        self.running = False
        return "Engine stopped"


class GPS:
    def __init__(self):
        self.location = (0.0, 0.0)

    def get_location(self):
        return f"Location: {self.location}"


class Car:
    """Car uses composition: HAS-A engine and GPS, not IS-A."""

    def __init__(self, model, horsepower):
        self.model = model
        self.engine = Engine(horsepower)   # Composition
        self.gps = GPS()                   # Composition

    def start(self):
        msg = self.engine.start()
        return f"{self.model}: {msg}"

    def locate(self):
        return f"{self.model}: {self.gps.get_location()}"


def composition_demo():
    """Composition vs inheritance."""
    car = Car("Tesla Model 3", 283)
    print(f"  {car.start()}")
    print(f"  {car.locate()}")

    # Components can be replaced or shared
    car.engine = Engine(450)
    print(f"  Engine swap: {car.start()}")

    print("\n  Composition advantages:")
    print("    - Flexible: swap components at runtime")
    print("    - Decoupled: Engine and GPS are independent")
    print("    - Testable: mock individual components")
    print("    - No diamond problem or deep hierarchies")


if __name__ == "__main__":
    sections = [
        ("Inheritance & ABC", inheritance_demo),
        ("MRO & Multiple Inheritance", mro_demo),
        ("Operator Overloading", operator_overloading_demo),
        ("Dataclasses", dataclass_demo),
        ("Composition", composition_demo),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

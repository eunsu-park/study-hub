"""
Example 13: Dataclasses and Modern OOP
Topic: Object-Oriented Programming

Demonstrates @dataclass, frozen dataclasses, field(), NamedTuple, and Protocol.
"""

from dataclasses import dataclass, field, asdict, astuple
from typing import NamedTuple, Protocol, runtime_checkable
import json


# =============================================================================
# BASIC DATACLASS
# =============================================================================

@dataclass
class Point:
    """Basic dataclass with auto-generated methods."""
    x: float
    y: float
    z: float = 0.0

    @property
    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


# =============================================================================
# FROZEN DATACLASS (IMMUTABLE)
# =============================================================================

@dataclass(frozen=True)
class Color:
    """Immutable color — can be used as dict key."""
    r: int
    g: int
    b: int

    def __post_init__(self):
        for name in ("r", "g", "b"):
            val = getattr(self, name)
            if not 0 <= val <= 255:
                raise ValueError(f"{name} must be 0-255, got {val}")

    def hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


# =============================================================================
# DATACLASS WITH field()
# =============================================================================

@dataclass
class Inventory:
    """Dataclass with advanced field configuration."""
    name: str
    items: list = field(default_factory=list)
    _count: int = field(default=0, repr=False, init=False)

    def add(self, item):
        self.items.append(item)
        self._count += 1

    def __len__(self):
        return self._count


# =============================================================================
# ORDERED DATACLASS
# =============================================================================

@dataclass(order=True)
class Student:
    """Sortable student — ordered by GPA then name."""
    gpa: float
    name: str = field(compare=False)  # Exclude from comparison

    def __post_init__(self):
        if not 0 <= self.gpa <= 4.0:
            raise ValueError(f"GPA must be 0-4, got {self.gpa}")


# =============================================================================
# NAMEDTUPLE
# =============================================================================

class Coordinate(NamedTuple):
    """Lightweight immutable record."""
    latitude: float
    longitude: float
    altitude: float = 0.0

    @property
    def is_northern(self):
        return self.latitude > 0


# =============================================================================
# PROTOCOL
# =============================================================================

@runtime_checkable
class Renderable(Protocol):
    def render(self) -> str: ...


class Paragraph:
    def __init__(self, text):
        self.text = text

    def render(self) -> str:
        return f"<p>{self.text}</p>"


class Header:
    def __init__(self, text, level=1):
        self.text = text
        self.level = level

    def render(self) -> str:
        tag = f"h{self.level}"
        return f"<{tag}>{self.text}</{tag}>"


def render_page(elements: list[Renderable]) -> str:
    return "\n".join(e.render() for e in elements)


# =============================================================================
# PRACTICAL: CONFIG SYSTEM
# =============================================================================

@dataclass(frozen=True)
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "mydb"

@dataclass
class AppConfig:
    name: str
    debug: bool = False
    db: DBConfig = field(default_factory=DBConfig)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


if __name__ == "__main__":
    # Basic dataclass
    print("=== Basic Dataclass ===")
    p1 = Point(3, 4)
    p2 = Point(3, 4)
    print(f"p1 = {p1}")
    print(f"p1 == p2? {p1 == p2}")
    print(f"Magnitude: {p1.magnitude:.2f}")

    # Frozen
    print("\n=== Frozen Dataclass ===")
    red = Color(255, 0, 0)
    blue = Color(0, 0, 255)
    print(f"{red} -> {red.hex()}")
    palette = {red: "red", blue: "blue"}
    print(f"Palette: {palette}")

    # field()
    print("\n=== field() ===")
    inv = Inventory("Warehouse")
    inv.add("Widget")
    inv.add("Gadget")
    print(f"{inv}")
    print(f"Count: {len(inv)}")

    # Ordered
    print("\n=== Ordered Dataclass ===")
    students = [Student(3.5, "Alice"), Student(3.8, "Bob"), Student(3.5, "Charlie")]
    for s in sorted(students, reverse=True):
        print(f"  {s.name}: {s.gpa}")

    # NamedTuple
    print("\n=== NamedTuple ===")
    nyc = Coordinate(40.7128, -74.0060, 10)
    print(f"NYC: {nyc}")
    print(f"Lat: {nyc.latitude}, Northern? {nyc.is_northern}")
    lat, lon, alt = nyc  # Unpacking
    print(f"Unpacked: ({lat}, {lon}, {alt})")

    # Protocol
    print("\n=== Protocol ===")
    elements = [Header("Hello", 1), Paragraph("World"), Header("Section", 2)]
    print(render_page(elements))
    print(f"\nParagraph is Renderable? {isinstance(Paragraph('x'), Renderable)}")

    # AppConfig
    print("\n=== Config System ===")
    config = AppConfig("MyApp", True, DBConfig("prod-db", 5433))
    print(config.to_json())

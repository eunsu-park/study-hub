"""
Exercise 13: Dataclasses and Modern OOP
Topic: Object-Oriented Programming

Practice @dataclass, frozen dataclasses, NamedTuple, and Protocol.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable


# =============================================================================
# Challenge 1: Dataclass — Inventory System
# =============================================================================

@dataclass
class Product:
    """Product dataclass.

    Fields:
        name (str): Product name.
        price (float): Product price.
        quantity (int): Stock quantity (default 0).
        tags (list[str]): Product tags (default empty list — use field!).

    Post-init validation:
        - price must be > 0 (raise ValueError)
        - quantity must be >= 0 (raise ValueError)

    Properties:
        total_value: price * quantity

    Methods:
        apply_discount(percent): Reduce price by percent (0-100).
    """

    # TODO: Implement
    pass


@dataclass(frozen=True, order=True)
class Rating:
    """Immutable, sortable rating.

    Fields:
        score (float): Rating score (used for ordering).
        reviewer (str): Reviewer name (excluded from comparison).
        comment (str): Review comment (excluded from comparison, default "").

    Post-init validation:
        - score must be 1.0 to 5.0

    The Rating should be hashable (frozen=True) and sortable by score.
    """

    # TODO: Implement
    pass


# =============================================================================
# Challenge 2: NamedTuple
# =============================================================================

class GeoPoint(NamedTuple):
    """Geographic point as NamedTuple.

    Fields:
        latitude (float)
        longitude (float)
        label (str, default "")

    Methods:
        distance_to(other): Approximate distance in km using Haversine formula.
            (You can use a simplified flat-earth formula for this exercise:
             dx = (other.longitude - self.longitude) * cos(avg_lat)
             dy = other.latitude - self.latitude
             distance = sqrt(dx^2 + dy^2) * 111.32  # km per degree)
    """

    # TODO: Implement
    pass


# =============================================================================
# Challenge 3: Protocol
# =============================================================================

@runtime_checkable
class Measurable(Protocol):
    """Protocol for anything with a measure() method."""

    def measure(self) -> float:
        ...


class Ruler:
    """Measures length in cm."""

    def __init__(self, length_cm: float):
        self.length_cm = length_cm

    def measure(self) -> float:
        return self.length_cm


class Scale:
    """Measures weight in kg."""

    def __init__(self, weight_kg: float):
        self.weight_kg = weight_kg

    def measure(self) -> float:
        return self.weight_kg


class Thermometer:
    """Measures temperature in Celsius."""

    def __init__(self, temp_c: float):
        self.temp_c = temp_c

    def measure(self) -> float:
        return self.temp_c


def average_measurement(instruments: list) -> float:
    """Calculate average of all measurements.

    Works with any object that has a measure() method (Protocol).

    TODO: Implement
    """
    pass


if __name__ == "__main__":
    # Test Product
    p = Product("Widget", 9.99, 100, ["sale", "new"])
    assert p.total_value == 999.0
    p.apply_discount(10)
    assert abs(p.price - 8.991) < 0.01

    try:
        Product("Bad", -5)
        assert False, "Should reject negative price"
    except ValueError:
        pass

    # Mutable default safety
    p1 = Product("A", 1.0)
    p2 = Product("B", 2.0)
    p1.tags.append("test")
    assert "test" not in p2.tags  # Independent lists!

    print(f"Product: {p}")

    # Test Rating
    r1 = Rating(4.5, "Alice", "Great!")
    r2 = Rating(3.0, "Bob")
    r3 = Rating(4.5, "Charlie", "Nice")

    assert r1 > r2
    assert r1 == r3  # Same score (reviewer excluded from comparison)

    ratings = sorted([r1, r2, r3])
    assert ratings[0].reviewer == "Bob"

    # Hashable (frozen)
    rating_set = {r1, r2, r3}
    assert len(rating_set) == 2  # r1 == r3

    print(f"Ratings sorted: {ratings}")

    # Test GeoPoint
    nyc = GeoPoint(40.7128, -74.0060, "NYC")
    la = GeoPoint(34.0522, -118.2437, "LA")

    dist = nyc.distance_to(la)
    assert 3000 < dist < 4500  # Approximate
    print(f"\n{nyc.label} to {la.label}: ~{dist:.0f} km")

    # Unpacking
    lat, lon, label = nyc
    assert lat == 40.7128

    # Test Protocol
    instruments = [Ruler(30), Scale(2.5), Thermometer(22)]
    avg = average_measurement(instruments)
    assert abs(avg - 18.167) < 0.01
    print(f"\nAverage measurement: {avg:.2f}")

    for inst in instruments:
        assert isinstance(inst, Measurable)

    print("\nAll tests passed!")

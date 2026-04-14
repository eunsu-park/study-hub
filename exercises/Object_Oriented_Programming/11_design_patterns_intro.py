"""
Exercise 11: Design Patterns Intro
Topic: Object-Oriented Programming

Implement Singleton, Factory, Observer, and Strategy patterns.
"""

from abc import ABC, abstractmethod


# =============================================================================
# Challenge 1: Singleton — Cache
# =============================================================================

class Cache:
    """Singleton cache with get/set/clear operations.

    Only ONE instance should ever exist.

    Methods:
        set(key, value): Store a key-value pair.
        get(key, default=None): Retrieve value by key.
        clear(): Remove all entries.
        __len__(): Return number of cached items.
        __contains__(key): Return True if key is cached.
    """

    # TODO: Implement as a Singleton
    pass


# =============================================================================
# Challenge 2: Factory — Shape Factory
# =============================================================================

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class CircleShape(Shape):
    """Circle with radius. area = pi * r^2."""

    # TODO: Implement
    pass


class RectangleShape(Shape):
    """Rectangle with width and height. area = w * h."""

    # TODO: Implement
    pass


class TriangleShape(Shape):
    """Triangle with base and height. area = 0.5 * b * h."""

    # TODO: Implement
    pass


class ShapeFactory:
    """Factory that creates shapes from type strings.

    create("circle", radius=5) -> CircleShape(5)
    create("rectangle", width=4, height=6) -> RectangleShape(4, 6)
    create("triangle", base=3, height=4) -> TriangleShape(3, 4)

    Raise ValueError for unknown shape types.
    """

    # TODO: Implement
    pass


# =============================================================================
# Challenge 3: Observer — Stock Price Monitor
# =============================================================================

class StockMonitor:
    """Observable stock price monitor.

    Methods:
        subscribe(callback): Register a callback function.
        unsubscribe(callback): Remove a callback.
        set_price(symbol, price): Update price and notify subscribers.
            Callbacks receive (symbol, price) as arguments.
        get_price(symbol): Return current price or None.
    """

    # TODO: Implement
    pass


if __name__ == "__main__":
    # Test Singleton Cache
    print("=== Singleton Cache ===")
    c1 = Cache()
    c1.set("key1", "value1")
    c1.set("key2", "value2")

    c2 = Cache()
    assert c1 is c2, "Must be singleton"
    assert c2.get("key1") == "value1"
    assert len(c2) == 2
    assert "key1" in c2

    c2.clear()
    assert len(c1) == 0
    print("Singleton: OK")

    # Test Shape Factory
    print("\n=== Shape Factory ===")
    factory = ShapeFactory()
    shapes = [
        factory.create("circle", radius=5),
        factory.create("rectangle", width=4, height=6),
        factory.create("triangle", base=3, height=4),
    ]

    for s in shapes:
        print(f"  {s}: area={s.area():.2f}")

    try:
        factory.create("hexagon")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("Factory: OK")

    # Test Observer
    print("\n=== Observer: Stock Monitor ===")
    monitor = StockMonitor()
    alerts = []

    def on_price_change(symbol, price):
        alerts.append(f"{symbol}: ${price:.2f}")

    monitor.subscribe(on_price_change)
    monitor.set_price("AAPL", 150.0)
    monitor.set_price("GOOG", 2800.0)
    monitor.set_price("AAPL", 155.0)

    assert len(alerts) == 3
    assert monitor.get_price("AAPL") == 155.0

    monitor.unsubscribe(on_price_change)
    monitor.set_price("AAPL", 160.0)
    assert len(alerts) == 3  # No more alerts

    print(f"Alerts: {alerts}")
    print("Observer: OK")

    print("\nAll tests passed!")

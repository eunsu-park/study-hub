"""
Exercises for Lesson 13: Dataclasses
Topic: Python

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import math
import json


# === Exercise 1: Immutable Config Class ===
# Problem: Write an immutable (frozen) configuration class.

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration.

    frozen=True makes all fields read-only after creation. This prevents
    accidental modification of config values during runtime. We use
    tuple instead of list for the settings field because frozen dataclasses
    require all fields to be hashable (lists are not).
    """
    app_name: str
    version: str
    debug: bool = False
    settings: tuple = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create an AppConfig from a dictionary.

        Converts the 'settings' list to a tuple to maintain immutability.
        """
        return cls(
            app_name=data["app_name"],
            version=data["version"],
            debug=data.get("debug", False),
            settings=tuple(data.get("settings", [])),
        )


def exercise_1():
    """Demonstrate immutable config class."""
    # Create from constructor
    config = AppConfig("MyApp", "1.0.0")
    print(f"Config: {config}")

    # Create from dictionary
    config_dict = {
        "app_name": "MyApp",
        "version": "1.0.0",
        "debug": True,
        "settings": ["verbose", "color"],
    }
    config2 = AppConfig.from_dict(config_dict)
    print(f"Config from dict: {config2}")

    # Verify immutability -- this should raise FrozenInstanceError
    try:
        config.debug = True
    except AttributeError as e:
        print(f"\nCannot modify frozen config: {e}")

    # Frozen dataclasses are hashable and can be used in sets
    configs = {config, config2}
    print(f"Unique configs in set: {len(configs)}")


# === Exercise 2: Auto-Calculated Fields ===
# Problem: Write a Circle class that auto-calculates area and circumference.

@dataclass
class Circle:
    """Circle with auto-calculated area and circumference.

    The radius is the only input parameter. Area and circumference are
    computed in __post_init__ and excluded from __init__ via init=False.
    Validation ensures radius is non-negative.
    """
    radius: float
    area: float = field(init=False, repr=True)
    circumference: float = field(init=False, repr=True)

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError(f"Radius cannot be negative, got {self.radius}")
        # Calculate derived fields after initialization
        self.area = math.pi * self.radius ** 2
        self.circumference = 2 * math.pi * self.radius


def exercise_2():
    """Demonstrate auto-calculated fields."""
    circle = Circle(5)
    print(f"Circle(5): {circle}")
    print(f"  Area: {circle.area:.4f}")
    print(f"  Circumference: {circle.circumference:.4f}")

    small = Circle(1)
    print(f"\nCircle(1): area={small.area:.4f}, circumference={small.circumference:.4f}")

    zero = Circle(0)
    print(f"Circle(0): area={zero.area:.4f}, circumference={zero.circumference:.4f}")

    # Negative radius raises ValueError
    try:
        Circle(-3)
    except ValueError as e:
        print(f"\nValueError: {e}")


# === Exercise 3: Nested Dataclasses and JSON ===
# Problem: Serialize/deserialize nested dataclasses to/from JSON.

@dataclass
class OrderItem:
    """Single item in an order."""
    product_name: str
    quantity: int
    price: float

    @property
    def subtotal(self) -> float:
        return self.quantity * self.price


@dataclass
class Order:
    """Order containing multiple items with JSON serialization.

    Uses dataclasses.asdict for serialization and a custom from_json
    classmethod for deserialization. The tricky part is reconstructing
    nested OrderItem instances from plain dicts during deserialization.
    """
    order_id: str
    customer: str
    items: List[OrderItem] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(item.subtotal for item in self.items)

    def to_json(self) -> str:
        """Serialize order to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Order":
        """Deserialize order from a JSON string.

        We need to manually reconstruct OrderItem instances because
        json.loads only produces plain dicts, not dataclass instances.
        """
        data = json.loads(json_str)
        items = [OrderItem(**item) for item in data.pop("items", [])]
        return cls(**data, items=items)

    def add_item(self, product_name: str, quantity: int, price: float):
        """Add an item to the order."""
        self.items.append(OrderItem(product_name, quantity, price))


def exercise_3():
    """Demonstrate nested dataclasses with JSON serialization."""
    # Create an order
    order = Order("ORD001", "Alice")
    order.add_item("Python Crash Course", 2, 29.99)
    order.add_item("Mechanical Keyboard", 1, 149.99)
    order.add_item("USB-C Cable", 3, 9.99)

    print(f"Order: {order.order_id} for {order.customer}")
    for item in order.items:
        print(f"  {item.product_name}: {item.quantity} x ${item.price:.2f} = ${item.subtotal:.2f}")
    print(f"  Total: ${order.total:.2f}")

    # Serialize to JSON
    json_str = order.to_json()
    print(f"\nJSON representation:\n{json_str}")

    # Deserialize from JSON
    restored = Order.from_json(json_str)
    print(f"\nRestored order: {restored.order_id} for {restored.customer}")
    print(f"  Items: {len(restored.items)}")
    print(f"  Total: ${restored.total:.2f}")

    # Verify round-trip correctness
    assert order.to_json() == restored.to_json(), "Round-trip serialization failed!"
    print("\nRound-trip JSON serialization verified!")


if __name__ == "__main__":
    print("=== Exercise 1: Immutable Config Class ===")
    exercise_1()

    print("\n=== Exercise 2: Auto-Calculated Fields ===")
    exercise_2()

    print("\n=== Exercise 3: Nested Dataclasses and JSON ===")
    exercise_3()

    print("\nAll exercises completed!")

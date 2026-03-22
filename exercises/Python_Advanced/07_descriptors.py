"""
Exercises for Lesson 07: Descriptors
Topic: Python

Solutions to practice problems from the lesson.
"""

import datetime


# === Exercise 1: Read-Only Attribute ===
# Problem: Write a descriptor that allows setting once but prevents modification.

class WriteOnce:
    """Descriptor that allows a single assignment, then becomes read-only.

    Uses a per-instance storage dict keyed by the instance's id to track
    whether the attribute has been set. This avoids polluting the
    instance's own __dict__ with internal state.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._values: dict[int, object] = {}

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if id(obj) not in self._values:
            raise AttributeError(f"'{self.name}' has not been set yet")
        return self._values[id(obj)]

    def __set__(self, obj, value):
        if id(obj) in self._values:
            raise AttributeError(
                f"'{self.name}' is read-only: already set to {self._values[id(obj)]!r}"
            )
        self._values[id(obj)] = value

    def __delete__(self, obj):
        raise AttributeError(f"Cannot delete read-only attribute '{self.name}'")


def exercise_1():
    """Demonstrate write-once descriptor."""

    class Config:
        db_host = WriteOnce()
        db_port = WriteOnce()

    config = Config()

    # First assignment succeeds
    config.db_host = "localhost"
    config.db_port = 5432
    print(f"db_host = {config.db_host}")
    print(f"db_port = {config.db_port}")

    # Second assignment raises
    try:
        config.db_host = "remote-server"
    except AttributeError as e:
        print(f"\nAttributeError: {e}")

    # Different instance works independently
    config2 = Config()
    config2.db_host = "production.db.example.com"
    print(f"\nconfig2.db_host = {config2.db_host}")


# === Exercise 2: Logging Descriptor ===
# Problem: Write a descriptor that logs all attribute access and modifications.

class LoggedAttribute:
    """Descriptor that logs every get, set, and delete operation.

    Useful for debugging when you need to trace exactly when and how
    an attribute changes. Stores the actual value in the instance's
    __dict__ to avoid infinite recursion.
    """

    def __init__(self):
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name
        # Use a mangled internal name to store the actual value
        self.internal_name = f"_logged_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = getattr(obj, self.internal_name, None)
        print(f"  [LOG] GET {obj.__class__.__name__}.{self.name} -> {value!r}")
        return value

    def __set__(self, obj, value):
        old = getattr(obj, self.internal_name, "<unset>")
        print(f"  [LOG] SET {obj.__class__.__name__}.{self.name}: {old!r} -> {value!r}")
        setattr(obj, self.internal_name, value)

    def __delete__(self, obj):
        value = getattr(obj, self.internal_name, "<unset>")
        print(f"  [LOG] DEL {obj.__class__.__name__}.{self.name} (was {value!r})")
        if hasattr(obj, self.internal_name):
            delattr(obj, self.internal_name)


def exercise_2():
    """Demonstrate logging descriptor."""

    class User:
        name = LoggedAttribute()
        email = LoggedAttribute()

    user = User()
    user.name = "Alice"
    user.email = "alice@example.com"

    print(f"\nAccessing user.name:")
    _ = user.name

    print(f"\nChanging user.email:")
    user.email = "alice@newdomain.com"


# === Exercise 3: Unit Conversion ===
# Problem: Write a descriptor that stores in base units but displays
# in different units. (e.g., store in meters, display in kilometers)

class UnitField:
    """Descriptor that stores values in base units and converts on access.

    Args:
        base_unit: Name of the base storage unit (e.g., "m", "kg")
        conversions: Dict mapping unit names to (multiplier, unit_label) pairs.
            The multiplier converts FROM base units TO the display unit.
    """

    def __init__(self, base_unit: str, conversions: dict[str, tuple[float, str]]):
        self.base_unit = base_unit
        self.conversions = conversions
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"_{name}_base"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, 0.0)

    def __set__(self, obj, value):
        """Store value in base units."""
        setattr(obj, self.storage_name, float(value))

    def convert(self, obj, unit: str) -> str:
        """Convert stored base value to the specified unit."""
        base_value = getattr(obj, self.storage_name, 0.0)
        if unit == self.base_unit:
            return f"{base_value:.4f} {self.base_unit}"

        if unit not in self.conversions:
            raise ValueError(
                f"Unknown unit '{unit}'. Available: {list(self.conversions.keys())}"
            )

        multiplier, label = self.conversions[unit]
        return f"{base_value * multiplier:.4f} {label}"


def exercise_3():
    """Demonstrate unit conversion descriptor."""

    class Distance:
        # Store in meters, convert to km, cm, miles, feet
        value = UnitField("m", {
            "km": (0.001, "km"),
            "cm": (100.0, "cm"),
            "mi": (0.000621371, "mi"),
            "ft": (3.28084, "ft"),
        })

    class Temperature:
        # Store in Celsius, convert to Fahrenheit, Kelvin
        value = UnitField("C", {
            "F": (1.8, "F"),  # partial -- needs offset, handled specially below
            "K": (1.0, "K"),  # partial -- needs offset
        })

    # Distance example
    d = Distance()
    d.value = 1500  # 1500 meters
    print(f"Stored: {d.value} m")
    print(f"In km: {Distance.value.convert(d, 'km')}")
    print(f"In cm: {Distance.value.convert(d, 'cm')}")
    print(f"In miles: {Distance.value.convert(d, 'mi')}")
    print(f"In feet: {Distance.value.convert(d, 'ft')}")

    # Another example: marathon distance
    d2 = Distance()
    d2.value = 42195  # Marathon in meters
    print(f"\nMarathon: {Distance.value.convert(d2, 'km')}")
    print(f"Marathon: {Distance.value.convert(d2, 'mi')}")


if __name__ == "__main__":
    print("=== Exercise 1: Read-Only Attribute ===")
    exercise_1()

    print("\n=== Exercise 2: Logging Descriptor ===")
    exercise_2()

    print("\n=== Exercise 3: Unit Conversion ===")
    exercise_3()

    print("\nAll exercises completed!")

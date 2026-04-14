"""
Example 06: Multiple Inheritance
Topic: Object-Oriented Programming

Demonstrates multiple inheritance, MRO, diamond problem resolution,
and practical mixin patterns.
"""

import json
from datetime import datetime


# =============================================================================
# MIXIN CLASSES
# =============================================================================

class SerializableMixin:
    """Adds JSON serialization."""

    def to_json(self):
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                data[key] = value
        return json.dumps(data, default=str, indent=2)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class LoggableMixin:
    """Adds logging capability."""

    def log(self, message, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] {self.__class__.__name__}: {message}")


class ValidatableMixin:
    """Adds validation capability."""

    def validate(self):
        """Override in subclasses to add validation rules."""
        errors = []
        for rule in self._get_rules():
            msg = rule()
            if msg:
                errors.append(msg)
        return errors

    def _get_rules(self):
        """Override to return list of validation callables."""
        return []

    def is_valid(self):
        return len(self.validate()) == 0


# =============================================================================
# COMBINING MIXINS
# =============================================================================

class Product(SerializableMixin, LoggableMixin, ValidatableMixin):
    """Product class combining multiple mixins."""

    def __init__(self, name, price, stock=0):
        self.name = name
        self.price = price
        self.stock = stock

    def _get_rules(self):
        return [
            lambda: "Name is required" if not self.name else None,
            lambda: "Price must be positive" if self.price <= 0 else None,
            lambda: "Stock cannot be negative" if self.stock < 0 else None,
        ]

    def __repr__(self):
        return f"Product({self.name!r}, ${self.price}, stock={self.stock})"


# =============================================================================
# DIAMOND PROBLEM WITH COOPERATIVE super()
# =============================================================================

class Base:
    """Root of the diamond."""

    def __init__(self, **kwargs):
        print(f"  Base.__init__({kwargs})")

    def greet(self):
        return "Hello from Base"


class Left(Base):
    """Left branch of the diamond."""

    def __init__(self, left_val=0, **kwargs):
        print(f"  Left.__init__(left_val={left_val})")
        super().__init__(**kwargs)
        self.left_val = left_val

    def greet(self):
        return f"Hello from Left (left_val={self.left_val})"


class Right(Base):
    """Right branch of the diamond."""

    def __init__(self, right_val=0, **kwargs):
        print(f"  Right.__init__(right_val={right_val})")
        super().__init__(**kwargs)
        self.right_val = right_val

    def greet(self):
        return f"Hello from Right (right_val={self.right_val})"


class Diamond(Left, Right):
    """Bottom of the diamond — inherits from both Left and Right."""

    def __init__(self, name, left_val=0, right_val=0):
        print(f"  Diamond.__init__(name={name!r})")
        super().__init__(left_val=left_val, right_val=right_val)
        self.name = name


if __name__ == "__main__":
    # Mixin demonstration
    print("=== Mixin Combination ===")
    laptop = Product("Laptop", 999.99, 50)

    laptop.log("Product created")
    print(f"\nJSON:\n{laptop.to_json()}")

    errors = laptop.validate()
    print(f"\nValid? {laptop.is_valid()}, errors: {errors}")

    # Invalid product
    bad = Product("", -10, -5)
    errors = bad.validate()
    print(f"\nBad product valid? {bad.is_valid()}, errors: {errors}")

    # Diamond problem
    print("\n=== Diamond Problem ===")
    print("Creating Diamond:")
    d = Diamond("test", left_val=10, right_val=20)
    print(f"\nAttributes: name={d.name}, left={d.left_val}, right={d.right_val}")
    print(f"Greeting: {d.greet()}")

    # MRO
    print(f"\nMRO: {[c.__name__ for c in Diamond.__mro__]}")

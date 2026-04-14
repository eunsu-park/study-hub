"""
Example 03: Constructors and Initialization
Topic: Object-Oriented Programming

Demonstrates __init__ patterns: validation, defaults, mutable default trap,
alternative constructors, builder pattern, and __new__ vs __init__.
"""


# =============================================================================
# VALIDATION IN __init__
# =============================================================================

class Temperature:
    """Temperature with validation and unit conversion."""

    ABSOLUTE_ZERO_C = -273.15

    def __init__(self, value, scale="C"):
        if scale not in ("C", "F", "K"):
            raise ValueError(f"Invalid scale: {scale}")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(value).__name__}")

        self._value = value
        self._scale = scale

        if self.celsius < self.ABSOLUTE_ZERO_C:
            raise ValueError(f"{value}{scale} is below absolute zero")

    @property
    def celsius(self):
        if self._scale == "C":
            return self._value
        elif self._scale == "F":
            return (self._value - 32) * 5 / 9
        else:
            return self._value - 273.15

    @property
    def fahrenheit(self):
        return self.celsius * 9 / 5 + 32

    @property
    def kelvin(self):
        return self.celsius + 273.15

    def __repr__(self):
        return f"Temperature({self._value}{self._scale})"


# =============================================================================
# MUTABLE DEFAULT TRAP
# =============================================================================

class TaskList:
    """Demonstrates the correct way to handle mutable defaults."""

    def __init__(self, name, tasks=None):
        self.name = name
        self.tasks = tasks if tasks is not None else []  # Correct!

    def add(self, task):
        self.tasks.append(task)

    def __repr__(self):
        return f"TaskList({self.name!r}, {len(self.tasks)} tasks)"


# =============================================================================
# ALTERNATIVE CONSTRUCTORS
# =============================================================================

class Color:
    """Color with multiple construction methods."""

    def __init__(self, r, g, b):
        for name, val in [("r", r), ("g", g), ("b", b)]:
            if not 0 <= val <= 255:
                raise ValueError(f"{name} must be 0-255, got {val}")
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_hex(cls, hex_str):
        """Create from hex string like '#FF8000'."""
        hex_str = hex_str.lstrip("#")
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return cls(r, g, b)

    @classmethod
    def red(cls):
        return cls(255, 0, 0)

    @classmethod
    def green(cls):
        return cls(0, 255, 0)

    @classmethod
    def blue(cls):
        return cls(0, 0, 255)

    def to_hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def __repr__(self):
        return f"Color({self.r}, {self.g}, {self.b})"


# =============================================================================
# BUILDER PATTERN
# =============================================================================

class Pizza:
    """Pizza with builder pattern (method chaining)."""

    def __init__(self, size):
        if size not in ("small", "medium", "large"):
            raise ValueError(f"Invalid size: {size}")
        self.size = size
        self.toppings = []
        self.extra_cheese = False
        self.extra_sauce = False

    def add_topping(self, topping):
        self.toppings.append(topping)
        return self  # Enable chaining

    def with_extra_cheese(self):
        self.extra_cheese = True
        return self

    def with_extra_sauce(self):
        self.extra_sauce = True
        return self

    def price(self):
        base = {"small": 8, "medium": 10, "large": 12}[self.size]
        total = base + len(self.toppings) * 1.5
        if self.extra_cheese:
            total += 2
        if self.extra_sauce:
            total += 1
        return total

    def __repr__(self):
        parts = [f"{self.size} pizza"]
        if self.toppings:
            parts.append(f"with {', '.join(self.toppings)}")
        if self.extra_cheese:
            parts.append("+ extra cheese")
        if self.extra_sauce:
            parts.append("+ extra sauce")
        return " ".join(parts)


# =============================================================================
# SINGLETON WITH __new__
# =============================================================================

class AppConfig:
    """Singleton configuration manager."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, **settings):
        if self._initialized:
            return
        self._settings = settings
        self._initialized = True

    def get(self, key, default=None):
        return self._settings.get(key, default)

    def __repr__(self):
        return f"AppConfig({self._settings})"


if __name__ == "__main__":
    # Temperature validation
    print("=== Temperature Validation ===")
    t1 = Temperature(100, "C")
    print(f"{t1} -> {t1.celsius:.1f}C = {t1.fahrenheit:.1f}F = {t1.kelvin:.1f}K")

    t2 = Temperature(32, "F")
    print(f"{t2} -> {t2.celsius:.1f}C")

    try:
        Temperature(-300, "C")
    except ValueError as e:
        print(f"Validation error: {e}")

    # Mutable defaults
    print("\n=== Mutable Defaults (Safe) ===")
    list1 = TaskList("Work")
    list1.add("Write code")
    list2 = TaskList("Home")
    print(f"{list1}, {list2}")  # Separate lists

    # Alternative constructors
    print("\n=== Alternative Constructors ===")
    c1 = Color(255, 128, 0)
    c2 = Color.from_hex("#FF8000")
    c3 = Color.red()
    print(f"{c1} -> {c1.to_hex()}")
    print(f"{c2} -> {c2.to_hex()}")
    print(f"{c3} -> {c3.to_hex()}")

    # Builder pattern
    print("\n=== Builder Pattern ===")
    pizza = (Pizza("large")
             .add_topping("pepperoni")
             .add_topping("mushrooms")
             .with_extra_cheese()
             .with_extra_sauce())
    print(f"{pizza} — ${pizza.price():.2f}")

    # Singleton
    print("\n=== Singleton ===")
    config1 = AppConfig(debug=True, port=8080)
    config2 = AppConfig(debug=False)  # Same instance, init skipped
    print(f"Same instance? {config1 is config2}")
    print(f"Config: {config1}")

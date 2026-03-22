"""
Exercises for Lesson 06: Metaclasses
Topic: Python

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Enforce Abstract Methods ===
# Problem: Write a metaclass that raises an error if abstract methods
# are not implemented.

class AbstractEnforcer(type):
    """Metaclass that enforces implementation of abstract methods.

    During class creation, it scans for methods marked with
    `_is_abstract = True` in base classes and raises TypeError if any
    concrete subclass fails to override them. This is a simplified
    version of what abc.ABCMeta does internally.
    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip enforcement for the base class itself
        if not bases:
            return cls

        # Collect abstract methods from all bases
        abstract_methods = set()
        for base in bases:
            for attr_name in dir(base):
                attr = getattr(base, attr_name, None)
                if callable(attr) and getattr(attr, "_is_abstract", False):
                    abstract_methods.add(attr_name)

        # Check if they are implemented in the new class
        missing = []
        for method_name in abstract_methods:
            method = namespace.get(method_name)
            if method is None or getattr(method, "_is_abstract", False):
                missing.append(method_name)

        if missing:
            raise TypeError(
                f"Cannot instantiate {name}: missing implementations for "
                f"{', '.join(sorted(missing))}"
            )

        return cls


def abstract(func):
    """Mark a method as abstract."""
    func._is_abstract = True
    return func


def exercise_1():
    """Demonstrate abstract method enforcement via metaclass."""

    class Animal(metaclass=AbstractEnforcer):
        @abstract
        def speak(self):
            pass

        @abstract
        def move(self):
            pass

    # Concrete class that implements all abstract methods
    class Dog(Animal):
        def speak(self):
            return "Woof!"

        def move(self):
            return "Running on four legs"

    dog = Dog()
    print(f"Dog.speak() = {dog.speak()}")
    print(f"Dog.move() = {dog.move()}")

    # Incomplete class -- should raise TypeError
    try:
        class BadAnimal(Animal):
            def speak(self):
                return "..."
            # move() is not implemented!
    except TypeError as e:
        print(f"\nTypeError: {e}")


# === Exercise 2: Attribute Transformation ===
# Problem: Write a metaclass that automatically logs all methods.

class AutoLogMeta(type):
    """Metaclass that wraps every user-defined method with logging.

    On class creation, it inspects the namespace for callables
    (excluding dunder methods) and replaces each with a wrapper that
    prints entry/exit information.
    """

    def __new__(mcs, name, bases, namespace):
        new_namespace = {}
        for attr_name, attr_value in namespace.items():
            # Only wrap regular methods, skip dunder and non-callables
            if callable(attr_value) and not attr_name.startswith("_"):
                new_namespace[attr_name] = mcs._wrap_with_logging(attr_name, attr_value)
            else:
                new_namespace[attr_name] = attr_value
        return super().__new__(mcs, name, bases, new_namespace)

    @staticmethod
    def _wrap_with_logging(method_name, method):
        """Create a wrapper that logs calls and returns."""
        def wrapper(*args, **kwargs):
            print(f"  [LOG] Entering {method_name}()")
            result = method(*args, **kwargs)
            print(f"  [LOG] Exiting {method_name}() -> {result}")
            return result
        wrapper.__name__ = method_name
        return wrapper


def exercise_2():
    """Demonstrate auto-logging metaclass."""

    class Calculator(metaclass=AutoLogMeta):
        def add(self, a, b):
            return a + b

        def multiply(self, a, b):
            return a * b

    calc = Calculator()
    print("Calling calc.add(3, 5):")
    calc.add(3, 5)

    print("\nCalling calc.multiply(4, 7):")
    calc.multiply(4, 7)


# === Exercise 3: Immutable Class ===
# Problem: Write a metaclass that prohibits attribute changes after instance creation.

class ImmutableMeta(type):
    """Metaclass that makes instances immutable after __init__ completes.

    Replaces __setattr__ and __delattr__ with versions that raise
    AttributeError. During __init__, a flag allows normal attribute
    setting; once __init__ returns, the instance becomes frozen.
    """

    def __call__(cls, *args, **kwargs):
        # Create the instance and run __init__ with mutability enabled
        instance = cls.__new__(cls)
        # Temporarily allow attribute setting during __init__
        object.__setattr__(instance, "_initializing", True)
        instance.__init__(*args, **kwargs)
        object.__setattr__(instance, "_initializing", False)
        return instance

    def __new__(mcs, name, bases, namespace):
        # Inject custom __setattr__ and __delattr__
        def frozen_setattr(self, key, value):
            if getattr(self, "_initializing", False):
                object.__setattr__(self, key, value)
            else:
                raise AttributeError(
                    f"Cannot modify attribute '{key}': {name} instances are immutable"
                )

        def frozen_delattr(self, key):
            raise AttributeError(
                f"Cannot delete attribute '{key}': {name} instances are immutable"
            )

        namespace["__setattr__"] = frozen_setattr
        namespace["__delattr__"] = frozen_delattr

        return super().__new__(mcs, name, bases, namespace)


def exercise_3():
    """Demonstrate immutable class metaclass."""

    class Point(metaclass=ImmutableMeta):
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

        def __repr__(self):
            return f"Point({self.x}, {self.y})"

    p = Point(3.0, 4.0)
    print(f"Created: {p}")
    print(f"p.x = {p.x}, p.y = {p.y}")

    # Try to modify -- should raise AttributeError
    try:
        p.x = 10.0
    except AttributeError as e:
        print(f"\nAttributeError: {e}")

    # Try to delete -- should also raise
    try:
        del p.y
    except AttributeError as e:
        print(f"AttributeError: {e}")


if __name__ == "__main__":
    print("=== Exercise 1: Enforce Abstract Methods ===")
    exercise_1()

    print("\n=== Exercise 2: Attribute Transformation ===")
    exercise_2()

    print("\n=== Exercise 3: Immutable Class ===")
    exercise_3()

    print("\nAll exercises completed!")

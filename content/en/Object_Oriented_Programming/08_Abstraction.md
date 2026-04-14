# Lesson 08: Abstraction

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain abstraction as an OOP pillar and differentiate it from encapsulation
2. Create abstract base classes (ABCs) using Python's `abc` module
3. Define abstract methods and abstract properties
4. Use ABCs to enforce contracts on subclasses
5. Implement interfaces using ABCs and Protocols
6. Register virtual subclasses for structural compatibility
7. Design effective abstractions for real-world systems

## What Is Abstraction?

Abstraction is the process of **hiding complex implementation details** and **exposing only the essential features** through a simplified interface. It answers: "What does this object do?" without revealing "How does it do it?"

```
┌─────────────────────────────────────────────────┐
│  Abstraction vs Encapsulation                   │
│                                                 │
│  Abstraction:    WHAT an object does             │
│                  Defining the interface           │
│                  "A car has start(), drive(),     │
│                   stop() methods"                │
│                                                 │
│  Encapsulation:  HOW it does it                  │
│                  Hiding the implementation        │
│                  "The engine internals are        │
│                   hidden behind start()"         │
└─────────────────────────────────────────────────┘
```

### Real-World Analogy

A TV remote is an abstraction:
- **Interface**: Power, Volume Up/Down, Channel Up/Down
- **Hidden complexity**: IR signals, circuit boards, signal processing
- You don't need to understand electronics to use a TV

```
┌──────────────┐
│  TV Remote   │  <-- Abstraction (interface)
├──────────────┤
│ [Power]      │
│ [Vol +] [-]  │
│ [Ch  +] [-]  │
└──────────────┘
       │
       │ hides
       ▼
┌──────────────┐
│ IR LED       │  <-- Implementation (hidden)
│ Encoder      │
│ Frequency    │
│ modulation   │
│ Battery mgmt │
└──────────────┘
```

## Abstract Base Classes (ABCs)

Python's `abc` module provides tools to create abstract classes — classes that cannot be instantiated and that enforce a contract on their subclasses.

```python
from abc import ABC, abstractmethod


class Shape(ABC):
    """Abstract base class for shapes.

    Any concrete subclass MUST implement area() and perimeter().
    """

    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the shape."""
        pass

    # Concrete method: available to all subclasses
    def describe(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"area={self.area():.2f}, perimeter={self.perimeter():.2f}"
        )


# Cannot instantiate an ABC
# shape = Shape()  -> TypeError: Can't instantiate abstract class Shape

# Must implement ALL abstract methods
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        from math import pi
        return pi * self.radius ** 2

    def perimeter(self):
        from math import pi
        return 2 * pi * self.radius


# Incomplete implementation also fails
# class BadShape(Shape):
#     def area(self):
#         return 0
#     # Missing perimeter() -> TypeError on instantiation

c = Circle(5)
print(c.describe())  # Circle: area=78.54, perimeter=31.42
```

## Abstract Methods and Properties

### Abstract Methods

```python
from abc import ABC, abstractmethod


class Database(ABC):
    """Abstract interface for database operations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection."""
        pass

    @abstractmethod
    def execute(self, query: str) -> list:
        """Execute a query and return results."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        pass

    # Concrete method using abstract methods
    def execute_and_close(self, query: str) -> list:
        """Execute a query and close the connection."""
        try:
            results = self.execute(query)
            return results
        finally:
            self.close()


class PostgresDB(Database):
    def connect(self):
        print("Connected to PostgreSQL")
        self.connected = True

    def execute(self, query):
        print(f"PostgreSQL executing: {query}")
        return [{"result": "data"}]

    def close(self):
        print("PostgreSQL connection closed")
        self.connected = False


class SQLiteDB(Database):
    def connect(self):
        print("Connected to SQLite")
        self.connected = True

    def execute(self, query):
        print(f"SQLite executing: {query}")
        return [{"result": "data"}]

    def close(self):
        print("SQLite connection closed")
        self.connected = False
```

### Abstract Properties

```python
from abc import ABC, abstractmethod


class Animal(ABC):
    """Abstract animal with required properties."""

    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def sound(self) -> str:
        """The sound this animal makes."""
        pass

    @property
    @abstractmethod
    def legs(self) -> int:
        """Number of legs."""
        pass

    def describe(self):
        return f"{self.name} ({self.__class__.__name__}): says '{self.sound}', has {self.legs} legs"


class Dog(Animal):
    @property
    def sound(self):
        return "Woof"

    @property
    def legs(self):
        return 4


class Snake(Animal):
    @property
    def sound(self):
        return "Hiss"

    @property
    def legs(self):
        return 0


print(Dog("Rex").describe())    # Rex (Dog): says 'Woof', has 4 legs
print(Snake("Sly").describe())  # Sly (Snake): says 'Hiss', has 0 legs
```

### Abstract Class Methods

```python
from abc import ABC, abstractmethod


class Serializable(ABC):
    """Abstract class requiring serialization support."""

    @abstractmethod
    def serialize(self) -> dict:
        """Convert object to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict):
        """Create object from dictionary."""
        pass


class User(Serializable):
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def serialize(self):
        return {"name": self.name, "email": self.email}

    @classmethod
    def deserialize(cls, data):
        return cls(data["name"], data["email"])


user = User("Alice", "alice@example.com")
data = user.serialize()       # {"name": "Alice", "email": "alice@example.com"}
user2 = User.deserialize(data)
print(user2.name)             # Alice
```

## Interfaces via ABCs

An **interface** is a purely abstract class — all methods are abstract, no implementation provided:

```python
from abc import ABC, abstractmethod


class Printable(ABC):
    """Interface: anything that can be printed."""

    @abstractmethod
    def to_string(self) -> str:
        pass


class Saveable(ABC):
    """Interface: anything that can be saved."""

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class Exportable(ABC):
    """Interface: anything that can be exported."""

    @abstractmethod
    def export(self, format: str) -> bytes:
        pass


class Report(Printable, Saveable, Exportable):
    """A report implementing multiple interfaces."""

    def __init__(self, title, content):
        self.title = title
        self.content = content

    def to_string(self):
        return f"=== {self.title} ===\n{self.content}"

    def save(self, path):
        print(f"Saving report to {path}")

    def load(self, path):
        print(f"Loading report from {path}")

    def export(self, format):
        return f"Exporting as {format}".encode()
```

## Virtual Subclasses with `register()`

You can declare that an existing class satisfies an ABC without modifying it:

```python
from abc import ABC, abstractmethod


class Iterable(ABC):
    @abstractmethod
    def __iter__(self):
        pass


# Register a class as a "virtual subclass" of Iterable
class NumberRange:
    """This class doesn't inherit from Iterable, but it IS iterable."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        current = self.start
        while current <= self.end:
            yield current
            current += 1


Iterable.register(NumberRange)

nr = NumberRange(1, 5)
print(isinstance(nr, Iterable))  # True (registered!)

for n in nr:
    print(n, end=" ")  # 1 2 3 4 5
```

## Template Method Pattern

ABCs often implement the **Template Method** pattern: the abstract class defines the algorithm skeleton, and subclasses fill in the steps:

```python
from abc import ABC, abstractmethod


class DataPipeline(ABC):
    """Template method pattern: define the pipeline, let subclasses fill in steps."""

    def run(self):
        """The template method — defines the algorithm skeleton."""
        data = self.extract()
        cleaned = self.transform(data)
        self.load(cleaned)
        print("Pipeline complete!")

    @abstractmethod
    def extract(self) -> list:
        """Step 1: Extract raw data."""
        pass

    @abstractmethod
    def transform(self, data: list) -> list:
        """Step 2: Transform/clean data."""
        pass

    @abstractmethod
    def load(self, data: list) -> None:
        """Step 3: Load processed data."""
        pass


class CSVPipeline(DataPipeline):
    def extract(self):
        print("Extracting from CSV file...")
        return [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]

    def transform(self, data):
        print("Converting age to int...")
        return [{**d, "age": int(d["age"])} for d in data]

    def load(self, data):
        print(f"Loaded {len(data)} records to database")


class APIPipeline(DataPipeline):
    def extract(self):
        print("Fetching from REST API...")
        return [{"user": "charlie", "score": 95}]

    def transform(self, data):
        print("Normalizing scores...")
        return [{**d, "score": d["score"] / 100} for d in data]

    def load(self, data):
        print(f"Loaded {len(data)} records to data warehouse")


# Same interface, different implementations
CSVPipeline().run()
APIPipeline().run()
```

## ABCs in the Standard Library

Python's `collections.abc` provides many useful ABCs:

```python
from collections.abc import Sequence, Mapping, Iterator, Callable

# Check if objects implement standard interfaces
print(isinstance([1, 2, 3], Sequence))     # True
print(isinstance({"a": 1}, Mapping))       # True
print(isinstance(iter([]), Iterator))       # True
print(isinstance(len, Callable))           # True

# Create custom collections by inheriting from ABCs
from collections.abc import MutableSequence

class ValidatedList(MutableSequence):
    """A list that validates items before adding."""

    def __init__(self, validator, initial=None):
        self._validator = validator
        self._data = []
        if initial:
            for item in initial:
                self.append(item)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._validator(value)
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def insert(self, index, value):
        self._validator(value)
        self._data.insert(index, value)


def positive_only(x):
    if x <= 0:
        raise ValueError(f"Value must be positive, got {x}")

nums = ValidatedList(positive_only, [1, 2, 3])
nums.append(4)      # OK
# nums.append(-1)   # ValueError!
print(list(nums))   # [1, 2, 3, 4]
```

## Summary

- Abstraction hides complexity behind simple interfaces — "what" not "how"
- Use `ABC` and `@abstractmethod` to create abstract classes that enforce contracts
- Abstract classes cannot be instantiated — subclasses must implement all abstract methods
- Abstract properties and class methods are also supported
- The Template Method pattern uses ABCs to define algorithm skeletons
- Python's `collections.abc` provides standard ABCs for common interfaces
- Use `register()` to make existing classes virtual subclasses of ABCs

## Next Steps

In [Lesson 09: Composition vs Inheritance](09_Composition_vs_Inheritance.md), we will explore when to use composition ("has-a") instead of inheritance ("is-a").

# Lesson 11: Design Patterns Intro

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what design patterns are and why they matter
2. Implement the Singleton pattern for single-instance classes
3. Use the Factory pattern to decouple object creation from usage
4. Apply the Observer pattern for event-driven communication
5. Implement the Strategy pattern for interchangeable algorithms
6. Classify patterns into Creational, Structural, and Behavioral categories
7. Choose the right pattern for common design problems

## What Are Design Patterns?

Design patterns are **reusable solutions to commonly recurring problems** in software design. They were popularized by the "Gang of Four" (GoF) book: *Design Patterns: Elements of Reusable Object-Oriented Software* (1994).

```
┌─────────────────────────────────────────────────┐
│            Design Pattern Categories            │
├─────────────────┬───────────────┬───────────────┤
│  Creational     │  Structural   │  Behavioral   │
│  (object        │  (object      │  (object      │
│   creation)     │   composition)│   interaction)│
├─────────────────┼───────────────┼───────────────┤
│ Singleton       │ Adapter       │ Observer      │
│ Factory Method  │ Decorator     │ Strategy      │
│ Abstract Factory│ Facade        │ Command       │
│ Builder         │ Proxy         │ Iterator      │
│ Prototype       │ Composite     │ State         │
└─────────────────┴───────────────┴───────────────┘
```

## Pattern 1: Singleton

**Intent**: Ensure a class has only **one instance** and provide a global access point.

**When to use**: Database connections, configuration managers, logging services — resources that must be shared across the entire application.

```
┌──────────────────────────────────┐
│         Singleton                │
├──────────────────────────────────┤
│ - _instance: Singleton           │
├──────────────────────────────────┤
│ + get_instance(): Singleton      │
│   (creates instance if needed,   │
│    returns existing one if not)  │
└──────────────────────────────────┘
         │
    Only ONE instance
    ever exists
```

### Implementation

```python
class DatabaseConnection:
    """Singleton database connection."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, host="localhost", port=5432):
        if self._initialized:
            return  # Don't re-initialize
        self.host = host
        self.port = port
        self.connected = False
        self._initialized = True

    def connect(self):
        self.connected = True
        return f"Connected to {self.host}:{self.port}"

    def query(self, sql):
        if not self.connected:
            raise RuntimeError("Not connected")
        return f"Executing: {sql}"


# Both variables point to the SAME instance
db1 = DatabaseConnection("prod-server", 5432)
db2 = DatabaseConnection()  # Same instance, __init__ skipped

print(db1 is db2)        # True
print(db1.host)          # prod-server
print(db2.host)          # prod-server (same object!)
```

### Thread-Safe Singleton

```python
import threading


class ThreadSafeSingleton:
    """Thread-safe singleton using a lock."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-check pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

## Pattern 2: Factory

**Intent**: Define an interface for creating objects, but let subclasses or factory functions decide **which class to instantiate**.

**When to use**: When the exact type of object to create depends on input data, configuration, or runtime conditions.

```
┌──────────────────┐
│  AnimalFactory    │
├──────────────────┤
│ + create(type)   │──── returns ──▶ Animal subclass
└──────────────────┘
         │
    ┌────┼─────┐
    │    │     │
   Dog  Cat  Bird
```

### Simple Factory

```python
class Animal:
    """Base class for animals."""

    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError


class Dog(Animal):
    def speak(self):
        return f"{self.name}: Woof!"


class Cat(Animal):
    def speak(self):
        return f"{self.name}: Meow!"


class Bird(Animal):
    def speak(self):
        return f"{self.name}: Tweet!"


class AnimalFactory:
    """Factory that creates animals based on type string."""

    _registry = {
        "dog": Dog,
        "cat": Cat,
        "bird": Bird,
    }

    @classmethod
    def create(cls, animal_type: str, name: str) -> Animal:
        """Create an animal by type name."""
        animal_class = cls._registry.get(animal_type.lower())
        if animal_class is None:
            raise ValueError(f"Unknown animal type: {animal_type}")
        return animal_class(name)

    @classmethod
    def register(cls, type_name: str, animal_class):
        """Register a new animal type (extensible!)."""
        cls._registry[type_name.lower()] = animal_class


# Usage: client doesn't need to know specific classes
dog = AnimalFactory.create("dog", "Rex")
cat = AnimalFactory.create("cat", "Whiskers")
print(dog.speak())  # Rex: Woof!
print(cat.speak())  # Whiskers: Meow!

# Extend without modifying factory code (OCP!)
class Fish(Animal):
    def speak(self):
        return f"{self.name}: Blub!"

AnimalFactory.register("fish", Fish)
fish = AnimalFactory.create("fish", "Nemo")
print(fish.speak())  # Nemo: Blub!
```

### Factory Method

```python
from abc import ABC, abstractmethod


class Document(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class PDFDocument(Document):
    def render(self):
        return "Rendering PDF document"


class HTMLDocument(Document):
    def render(self):
        return "Rendering HTML document"


class MarkdownDocument(Document):
    def render(self):
        return "Rendering Markdown document"


class DocumentCreator(ABC):
    """Abstract creator with factory method."""

    @abstractmethod
    def create_document(self) -> Document:
        """Factory method — subclasses decide what to create."""
        pass

    def open_document(self) -> str:
        """Template method that uses the factory method."""
        doc = self.create_document()
        return doc.render()


class PDFCreator(DocumentCreator):
    def create_document(self):
        return PDFDocument()


class HTMLCreator(DocumentCreator):
    def create_document(self):
        return HTMLDocument()


# Client code works with any creator
for creator in [PDFCreator(), HTMLCreator()]:
    print(creator.open_document())
```

## Pattern 3: Observer

**Intent**: Define a one-to-many dependency so that when one object changes state, all dependents are **notified automatically**.

**When to use**: Event systems, UI updates, notification services, pub/sub messaging.

```
┌──────────────┐     notify      ┌──────────────┐
│   Subject    │────────────────▶│  Observer 1  │
│  (Publisher) │                 └──────────────┘
│              │     notify      ┌──────────────┐
│  - observers │────────────────▶│  Observer 2  │
│  + attach()  │                 └──────────────┘
│  + detach()  │     notify      ┌──────────────┐
│  + notify()  │────────────────▶│  Observer 3  │
└──────────────┘                 └──────────────┘
```

```python
from abc import ABC, abstractmethod


class Observer(ABC):
    """Abstract observer."""

    @abstractmethod
    def update(self, event: str, data: dict) -> None:
        pass


class EventEmitter:
    """Subject that manages observers and emits events."""

    def __init__(self):
        self._observers: dict[str, list[Observer]] = {}

    def on(self, event: str, observer: Observer):
        """Subscribe to an event."""
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(observer)

    def off(self, event: str, observer: Observer):
        """Unsubscribe from an event."""
        if event in self._observers:
            self._observers[event].remove(observer)

    def emit(self, event: str, data: dict = None):
        """Notify all observers of an event."""
        for observer in self._observers.get(event, []):
            observer.update(event, data or {})


# Concrete observers
class Logger(Observer):
    def update(self, event, data):
        print(f"[LOG] {event}: {data}")


class EmailNotifier(Observer):
    def update(self, event, data):
        if event == "user_registered":
            print(f"[EMAIL] Welcome email sent to {data.get('email')}")


class Analytics(Observer):
    def __init__(self):
        self.events = []

    def update(self, event, data):
        self.events.append({"event": event, "data": data})
        print(f"[ANALYTICS] Tracked: {event} (total: {len(self.events)})")


# Wire up the system
app = EventEmitter()
logger = Logger()
emailer = EmailNotifier()
analytics = Analytics()

app.on("user_registered", logger)
app.on("user_registered", emailer)
app.on("user_registered", analytics)
app.on("order_placed", logger)
app.on("order_placed", analytics)

# Emit events — all subscribed observers are notified
app.emit("user_registered", {"email": "alice@example.com", "name": "Alice"})
app.emit("order_placed", {"order_id": 123, "total": 99.99})
```

## Pattern 4: Strategy

**Intent**: Define a family of algorithms, encapsulate each one, and make them **interchangeable** at runtime.

**When to use**: When you need different variations of an algorithm and want to switch between them without modifying client code.

```
┌──────────────┐     uses      ┌──────────────────┐
│   Context    │──────────────▶│  Strategy (ABC)  │
│              │               ├──────────────────┤
│ - strategy   │               │ + execute()      │
│ + do_work()  │               └────────┬─────────┘
└──────────────┘                   ┌────┼────┐
                                   │    │    │
                              StratA StratB StratC
```

```python
from abc import ABC, abstractmethod


class CompressionStrategy(ABC):
    """Abstract compression strategy."""

    @abstractmethod
    def compress(self, data: str) -> str:
        pass

    @abstractmethod
    def decompress(self, data: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ZipCompression(CompressionStrategy):
    @property
    def name(self):
        return "ZIP"

    def compress(self, data):
        return f"[ZIP compressed: {len(data)} -> {len(data)//2} bytes]"

    def decompress(self, data):
        return f"[ZIP decompressed]"


class GzipCompression(CompressionStrategy):
    @property
    def name(self):
        return "GZIP"

    def compress(self, data):
        return f"[GZIP compressed: {len(data)} -> {len(data)//3} bytes]"

    def decompress(self, data):
        return f"[GZIP decompressed]"


class NoCompression(CompressionStrategy):
    @property
    def name(self):
        return "None"

    def compress(self, data):
        return data

    def decompress(self, data):
        return data


class FileArchiver:
    """Context that uses a compression strategy."""

    def __init__(self, strategy: CompressionStrategy = None):
        self._strategy = strategy or NoCompression()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: CompressionStrategy):
        print(f"Switched compression: {self._strategy.name} -> {new_strategy.name}")
        self._strategy = new_strategy

    def archive(self, filename: str, data: str):
        compressed = self._strategy.compress(data)
        print(f"Archiving {filename} with {self._strategy.name}: {compressed}")


# Usage with runtime strategy switching
archiver = FileArchiver(ZipCompression())
archiver.archive("report.txt", "A" * 1000)

archiver.strategy = GzipCompression()  # Switch at runtime!
archiver.archive("data.csv", "B" * 5000)
```

## Pattern Comparison

```
┌────────────┬────────────────────┬──────────────────────┐
│  Pattern   │  Problem           │  Solution            │
├────────────┼────────────────────┼──────────────────────┤
│ Singleton  │ Need exactly one   │ Control instance     │
│            │ instance globally  │ creation in __new__  │
├────────────┼────────────────────┼──────────────────────┤
│ Factory    │ Don't know which   │ Delegate creation to │
│            │ class to create    │ factory method/class │
│            │ until runtime      │                      │
├────────────┼────────────────────┼──────────────────────┤
│ Observer   │ Objects need to    │ Subscribe/publish    │
│            │ react to changes   │ event system         │
│            │ in other objects   │                      │
├────────────┼────────────────────┼──────────────────────┤
│ Strategy   │ Need to switch     │ Encapsulate each     │
│            │ algorithms at      │ algorithm as an      │
│            │ runtime            │ interchangeable obj  │
└────────────┴────────────────────┴──────────────────────┘
```

## Anti-Patterns to Avoid

### Singleton Overuse

```python
# BAD: Using Singleton for everything
class ConfigSingleton:  # OK — config is naturally singular
    ...

class UserSingleton:  # BAD — multiple users exist!
    ...

class LoggerSingleton:  # DEBATABLE — one logger or many?
    ...
```

### God Factory

```python
# BAD: Factory that creates everything
class UniversalFactory:
    def create(self, type_name):
        if type_name == "user": ...
        elif type_name == "order": ...
        elif type_name == "product": ...
        elif type_name == "email": ...
        # 100 more elif branches... SRP violation!
```

## Summary

- Design patterns are reusable solutions to common OOP problems
- **Singleton**: Ensures one instance; use for shared resources (DB connections, config)
- **Factory**: Decouples creation from usage; extensible through registration
- **Observer**: Pub/sub event system; decouples event sources from handlers
- **Strategy**: Swappable algorithms via composition; eliminates conditional logic
- Patterns are guidelines, not rigid rules — apply them when they solve a real problem
- Over-engineering with patterns is itself an anti-pattern

## Next Steps

In [Lesson 12: Magic Methods](12_Magic_Methods.md), we will explore Python's special methods that let you customize how your objects behave with built-in operators and functions.

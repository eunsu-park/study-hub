# Lesson 09: Composition vs Inheritance

## Learning Objectives

By the end of this lesson, you will be able to:
1. Distinguish between "is-a" (inheritance) and "has-a" (composition) relationships
2. Apply composition to build flexible systems from smaller components
3. Implement delegation to forward method calls to composed objects
4. Recognize when inheritance is misused and refactor to composition
5. Use the Strategy pattern as a composition-based alternative to inheritance
6. Combine composition and inheritance effectively
7. Follow the principle "Favor composition over inheritance"

## Is-A vs Has-A

The fundamental question when designing relationships between classes:

```
┌─────────────────────────────────────────────┐
│  Inheritance (Is-A)                         │
│                                             │
│  "A Dog IS an Animal"                       │
│  "A Manager IS an Employee"                 │
│  class Dog(Animal): ...                     │
│                                             │
│  The subclass IS a specialized version of   │
│  the parent. It can be used anywhere the    │
│  parent is expected.                        │
├─────────────────────────────────────────────┤
│  Composition (Has-A)                        │
│                                             │
│  "A Car HAS an Engine"                      │
│  "A Computer HAS a CPU"                     │
│  class Car:                                 │
│      def __init__(self):                    │
│          self.engine = Engine()             │
│                                             │
│  The class CONTAINS other objects as parts. │
│  It delegates behavior to them.             │
└─────────────────────────────────────────────┘
```

## Composition in Action

```python
class Engine:
    """An engine component."""

    def __init__(self, horsepower, fuel_type="gasoline"):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.is_running = False

    def start(self):
        self.is_running = True
        return f"Engine ({self.horsepower}hp) started"

    def stop(self):
        self.is_running = False
        return "Engine stopped"


class Transmission:
    """A transmission component."""

    def __init__(self, type="automatic", gears=6):
        self.type = type
        self.gears = gears
        self.current_gear = 0  # 0 = Park

    def shift(self, gear):
        if 0 <= gear <= self.gears:
            self.current_gear = gear
            return f"Shifted to gear {gear}"
        raise ValueError(f"Invalid gear: {gear}")


class GPS:
    """A GPS navigation component."""

    def __init__(self):
        self.destination = None

    def navigate(self, destination):
        self.destination = destination
        return f"Navigating to {destination}"


class Car:
    """A car COMPOSED of components (not inheriting from them)."""

    def __init__(self, make, model, horsepower):
        self.make = make
        self.model = model
        # Composition: Car HAS these components
        self.engine = Engine(horsepower)
        self.transmission = Transmission()
        self.gps = GPS()

    def start(self):
        """Delegate to engine."""
        return self.engine.start()

    def drive(self, destination):
        """Coordinate multiple components."""
        if not self.engine.is_running:
            self.start()
        self.transmission.shift(1)
        return self.gps.navigate(destination)

    def describe(self):
        return (f"{self.make} {self.model}: "
                f"{self.engine.horsepower}hp {self.engine.fuel_type}, "
                f"{self.transmission.type} {self.transmission.gears}-speed")


car = Car("Toyota", "Camry", 203)
print(car.describe())   # Toyota Camry: 203hp gasoline, automatic 6-speed
print(car.start())      # Engine (203hp) started
print(car.drive("NYC")) # Navigating to NYC
```

### Why Not Inheritance Here?

```python
# BAD: A Car is NOT an Engine
class BadCar(Engine):  # Violates "is-a" — a car is not an engine!
    def __init__(self, make):
        super().__init__(200)
        self.make = make
    # Car inherits engine methods directly
    # What if we need TWO engines? Can't inherit twice!

# GOOD: A Car HAS an Engine (composition)
class GoodCar:
    def __init__(self, make, engine):
        self.make = make
        self.engine = engine  # Can swap engines, have multiple, etc.
```

## Delegation

Delegation is the mechanism by which a composed object forwards method calls to its components:

```python
class Logger:
    """A logging component."""

    def __init__(self, name):
        self.name = name
        self.entries = []

    def log(self, message, level="INFO"):
        entry = f"[{level}] {self.name}: {message}"
        self.entries.append(entry)
        print(entry)

    def get_logs(self):
        return list(self.entries)


class EmailService:
    """An email service component."""

    def send(self, to, subject, body):
        return f"Email sent to {to}: {subject}"


class UserService:
    """User service that delegates to logger and email."""

    def __init__(self):
        self._logger = Logger("UserService")
        self._email = EmailService()
        self._users = {}

    def register(self, username, email):
        """Register a new user — delegates logging and email."""
        if username in self._users:
            self._logger.log(f"Registration failed: {username} exists", "WARN")
            raise ValueError(f"User {username} already exists")

        self._users[username] = {"email": email}
        self._logger.log(f"User {username} registered")
        self._email.send(email, "Welcome!", f"Hello {username}!")
        return True

    def get_user(self, username):
        return self._users.get(username)


service = UserService()
service.register("alice", "alice@example.com")
# [INFO] UserService: User alice registered
```

## Strategy Pattern: Composition Over Inheritance

The Strategy pattern replaces inheritance hierarchies with interchangeable components:

```
┌─────────────────────────────────────────────────┐
│  Inheritance approach (rigid):                  │
│                                                 │
│       Sorter                                    │
│      /    \                                     │
│  BubbleSorter  QuickSorter  MergeSorter        │
│                                                 │
│  Problem: Can't change algorithm at runtime     │
├─────────────────────────────────────────────────┤
│  Composition approach (flexible):               │
│                                                 │
│  Sorter ──has──▶ SortStrategy                  │
│                   /    |    \                    │
│              Bubble  Quick  Merge               │
│                                                 │
│  Benefit: Swap strategies at runtime!           │
└─────────────────────────────────────────────────┘
```

```python
from abc import ABC, abstractmethod


class SortStrategy(ABC):
    """Abstract sorting strategy."""

    @abstractmethod
    def sort(self, data: list) -> list:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BubbleSort(SortStrategy):
    @property
    def name(self):
        return "Bubble Sort"

    def sort(self, data):
        arr = list(data)
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class QuickSort(SortStrategy):
    @property
    def name(self):
        return "Quick Sort"

    def sort(self, data):
        if len(data) <= 1:
            return list(data)
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class Sorter:
    """Sorter that uses a pluggable strategy (composition)."""

    def __init__(self, strategy: SortStrategy = None):
        self._strategy = strategy or BubbleSort()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: SortStrategy):
        """Change strategy at runtime!"""
        self._strategy = new_strategy

    def sort(self, data):
        print(f"Sorting with {self._strategy.name}")
        return self._strategy.sort(data)


data = [64, 34, 25, 12, 22, 11, 90]

sorter = Sorter(BubbleSort())
print(sorter.sort(data))  # Sorting with Bubble Sort -> sorted list

sorter.strategy = QuickSort()  # Swap strategy at runtime!
print(sorter.sort(data))  # Sorting with Quick Sort -> sorted list
```

## When to Use Which

```
┌──────────────────────────────────────────────────────────────┐
│  Use INHERITANCE when:                                       │
│  - There is a genuine "is-a" relationship                    │
│  - Subclass can substitute for parent everywhere (LSP)       │
│  - You want to reuse parent's interface + implementation     │
│  - The hierarchy is shallow (2-3 levels max)                 │
│                                                              │
│  Use COMPOSITION when:                                       │
│  - There is a "has-a" or "uses-a" relationship               │
│  - You need to combine behaviors from multiple sources       │
│  - You want to change behavior at runtime                    │
│  - You want loose coupling between components                │
│  - The inheritance hierarchy would be deep or complex        │
└──────────────────────────────────────────────────────────────┘
```

### Refactoring from Inheritance to Composition

```python
# BEFORE: Deep inheritance hierarchy
class Animal:
    def eat(self): pass

class FlyingAnimal(Animal):
    def fly(self): pass

class SwimmingAnimal(Animal):
    def swim(self): pass

# Problem: What about a duck that flies AND swims?
# class Duck(FlyingAnimal, SwimmingAnimal) -- diamond problem!


# AFTER: Composition with behavior objects
class FlyBehavior:
    def fly(self):
        return "Flying!"

class NoFlyBehavior:
    def fly(self):
        return "Can't fly"

class SwimBehavior:
    def swim(self):
        return "Swimming!"

class NoSwimBehavior:
    def swim(self):
        return "Can't swim"


class Animal:
    def __init__(self, name, fly_behavior=None, swim_behavior=None):
        self.name = name
        self._fly_behavior = fly_behavior or NoFlyBehavior()
        self._swim_behavior = swim_behavior or NoSwimBehavior()

    def fly(self):
        return self._fly_behavior.fly()

    def swim(self):
        return self._swim_behavior.swim()


duck = Animal("Duck", FlyBehavior(), SwimBehavior())
penguin = Animal("Penguin", NoFlyBehavior(), SwimBehavior())

print(f"{duck.name}: {duck.fly()}, {duck.swim()}")
# Duck: Flying!, Swimming!
print(f"{penguin.name}: {penguin.fly()}, {penguin.swim()}")
# Penguin: Can't fly, Swimming!
```

## Practical Example: Notification System

```python
class EmailSender:
    def send(self, recipient, message):
        return f"Email to {recipient}: {message}"

class SMSSender:
    def send(self, recipient, message):
        return f"SMS to {recipient}: {message}"

class SlackSender:
    def send(self, recipient, message):
        return f"Slack to #{recipient}: {message}"


class NotificationService:
    """Composed notification service — easy to extend."""

    def __init__(self):
        self._channels = {}

    def add_channel(self, name, sender):
        self._channels[name] = sender

    def notify(self, channel, recipient, message):
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")
        return self._channels[channel].send(recipient, message)

    def broadcast(self, recipient, message):
        """Send via ALL channels."""
        results = []
        for name, sender in self._channels.items():
            results.append(sender.send(recipient, message))
        return results


# Build the service by composing senders
notifier = NotificationService()
notifier.add_channel("email", EmailSender())
notifier.add_channel("sms", SMSSender())
notifier.add_channel("slack", SlackSender())

# Use individual channels
print(notifier.notify("email", "alice@mail.com", "Hello!"))

# Or broadcast to all
for msg in notifier.broadcast("alice", "Server is down!"):
    print(msg)
```

## Summary

- **Inheritance** = "is-a" relationship (Dog is an Animal)
- **Composition** = "has-a" relationship (Car has an Engine)
- Composition is more flexible: components can be swapped at runtime
- Delegation forwards method calls from the composed object to its components
- The Strategy pattern replaces inheritance hierarchies with pluggable components
- "Favor composition over inheritance" — use inheritance only for true "is-a" relationships with shallow hierarchies
- Deep inheritance hierarchies are a code smell — refactor to composition

## Next Steps

In [Lesson 10: SOLID Principles](10_SOLID_Principles.md), we will learn the five SOLID design principles for creating maintainable, extensible OOP systems.

# OOP Advanced

**Previous**: [OOP Basics](./08_OOP_Basics.md) | **Next**: [Modules and Packages](./10_Modules_and_Packages.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement single inheritance and use `super()` to call parent class methods
2. Override methods in subclasses and understand the method resolution order (MRO) in multiple inheritance
3. Define abstract base classes using the `abc` module to enforce interface contracts
4. Apply polymorphism and duck typing principles in Python
5. Overload operators (`__add__`, `__eq__`, `__lt__`, `__len__`, `__getitem__`, etc.) to make classes behave like built-in types
6. Choose between composition and inheritance for code reuse
7. Use the `@dataclass` decorator to reduce boilerplate in data-holding classes
8. Recognize and implement basic design patterns (Factory, Singleton) in Python

---

Building on the OOP foundations from the previous lesson, this lesson covers the techniques that make object-oriented Python truly powerful. Inheritance lets you build class hierarchies. Abstract classes define contracts. Operator overloading lets your objects work seamlessly with Python syntax. Composition provides flexible alternatives to deep inheritance trees. These tools, used well, produce code that is both expressive and maintainable.

## 1. Inheritance

Inheritance allows a new class (child/subclass) to acquire the attributes and methods of an existing class (parent/superclass).

### Basic Inheritance

```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def speak(self):
        return f"{self.name} makes a sound"

    def __str__(self):
        return f"{self.name} ({self.species})"


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent __init__
        self.breed = breed

    def speak(self):  # Override parent method
        return f"{self.name} barks: Woof!"

    def fetch(self, item):  # New method specific to Dog
        return f"{self.name} fetches the {item}"


class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Cat")
        self.indoor = indoor

    def speak(self):
        return f"{self.name} meows: Meow!"

    def purr(self):
        return f"{self.name} purrs contentedly"


# Usage
dog = Dog("Rex", "German Shepherd")
cat = Cat("Whiskers", indoor=True)

print(dog)             # Rex (Dog)
print(dog.speak())     # Rex barks: Woof!
print(dog.fetch("ball"))  # Rex fetches the ball

print(cat)             # Whiskers (Cat)
print(cat.speak())     # Whiskers meows: Meow!
print(cat.purr())      # Whiskers purrs contentedly

# Inheritance checks
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

### The `super()` Function

`super()` returns a proxy object that delegates method calls to the parent class. It is essential for proper initialization in class hierarchies.

```python
class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def describe(self):
        return f"{self.year} {self.make} {self.model}"


class ElectricVehicle(Vehicle):
    def __init__(self, make, model, year, battery_kwh):
        super().__init__(make, model, year)
        self.battery_kwh = battery_kwh

    def describe(self):
        base = super().describe()  # Call parent's describe
        return f"{base} (Electric, {self.battery_kwh} kWh)"


class Tesla(ElectricVehicle):
    def __init__(self, model, year, battery_kwh, autopilot=False):
        super().__init__("Tesla", model, year, battery_kwh)
        self.autopilot = autopilot

    def describe(self):
        base = super().describe()
        ap_status = "with Autopilot" if self.autopilot else "no Autopilot"
        return f"{base} - {ap_status}"


car = Tesla("Model 3", 2024, 75, autopilot=True)
print(car.describe())
# 2024 Tesla Model 3 (Electric, 75 kWh) - with Autopilot
```

### Extending Parent Methods

```python
class Logger:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def get_logs(self):
        return self.logs


class TimestampLogger(Logger):
    def log(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().log(f"[{timestamp}] {message}")  # Extend, not replace


class PriorityLogger(TimestampLogger):
    def log(self, message, priority="INFO"):
        super().log(f"({priority}) {message}")  # Further extend


logger = PriorityLogger()
logger.log("System started", "INFO")
logger.log("Disk full", "CRITICAL")

for entry in logger.get_logs():
    print(entry)
# [2024-01-15 10:30:00] (INFO) System started
# [2024-01-15 10:30:00] (CRITICAL) Disk full
```

---

## 2. Method Overriding

When a subclass defines a method with the same name as a parent method, the subclass version takes precedence.

```python
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement area()")

    def perimeter(self):
        raise NotImplementedError("Subclasses must implement perimeter()")

    def describe(self):
        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius


class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeter(self):
        return self.a + self.b + self.c


shapes = [Rectangle(5, 3), Circle(4), Triangle(3, 4, 5)]
for shape in shapes:
    print(shape.describe())
# Rectangle: area=15.00, perimeter=16.00
# Circle: area=50.27, perimeter=25.13
# Triangle: area=6.00, perimeter=12.00
```

---

## 3. Multiple Inheritance and MRO

Python supports multiple inheritance -- a class can inherit from more than one parent.

```python
class Flyable:
    def fly(self):
        return f"{self.__class__.__name__} is flying"

class Swimmable:
    def swim(self):
        return f"{self.__class__.__name__} is swimming"

class Walkable:
    def walk(self):
        return f"{self.__class__.__name__} is walking"


class Duck(Flyable, Swimmable, Walkable):
    def __init__(self, name):
        self.name = name

    def quack(self):
        return f"{self.name} says: Quack!"

duck = Duck("Donald")
print(duck.fly())    # Duck is flying
print(duck.swim())   # Duck is swimming
print(duck.walk())   # Duck is walking
print(duck.quack())  # Donald says: Quack!
```

### Method Resolution Order (MRO)

When multiple parents define the same method, Python uses the C3 linearization algorithm to determine which method to call.

```python
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    def greet(self):
        return "Hello from C"

class D(B, C):
    pass

d = D()
print(d.greet())  # Hello from B (B comes before C in MRO)

# Inspect the MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Or use mro() method
for cls in D.mro():
    print(cls.__name__)
# D -> B -> C -> A -> object
```

### Cooperative Multiple Inheritance with `super()`

```python
class Base:
    def __init__(self, **kwargs):
        # Absorb remaining kwargs so the chain does not break
        pass

class PowerSource(Base):
    def __init__(self, fuel_type="electric", **kwargs):
        super().__init__(**kwargs)
        self.fuel_type = fuel_type

class Navigation(Base):
    def __init__(self, gps=True, **kwargs):
        super().__init__(**kwargs)
        self.gps = gps

class Communication(Base):
    def __init__(self, radio_freq=None, **kwargs):
        super().__init__(**kwargs)
        self.radio_freq = radio_freq

class Drone(PowerSource, Navigation, Communication):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def status(self):
        return (f"Drone '{self.name}': fuel={self.fuel_type}, "
                f"GPS={'on' if self.gps else 'off'}, "
                f"radio={self.radio_freq or 'none'}")

drone = Drone("Scout", fuel_type="battery", gps=True, radio_freq="2.4GHz")
print(drone.status())
# Drone 'Scout': fuel=battery, GPS=on, radio=2.4GHz

print(Drone.__mro__)
# Drone -> PowerSource -> Navigation -> Communication -> Base -> object
```

### The Diamond Problem

```python
class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()

d = D()
# D.__init__
# B.__init__
# C.__init__
# A.__init__
# Each class's __init__ is called exactly once (thanks to C3 linearization)
```

---

## 4. Abstract Base Classes

Abstract base classes (ABCs) define interfaces that subclasses must implement. You cannot instantiate an abstract class directly.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes."""

    @abstractmethod
    def area(self):
        """Calculate the area of the shape."""
        pass

    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape."""
        pass

    def describe(self):
        """Non-abstract method -- inherited by all subclasses."""
        return f"{self.__class__.__name__}: area={self.area():.2f}"

# Cannot instantiate abstract class
# shape = Shape()  # TypeError: Can't instantiate abstract class Shape

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

    def perimeter(self):
        return 4 * self.side

# Must implement ALL abstract methods
class IncompleteShape(Shape):
    def area(self):
        return 0
    # Missing perimeter() -- cannot instantiate!

# incomplete = IncompleteShape()
# TypeError: Can't instantiate abstract class IncompleteShape
# with abstract method perimeter

s = Square(5)
print(s.describe())     # Square: area=25.00
print(s.perimeter())    # 20
```

### Abstract Properties

```python
from abc import ABC, abstractmethod

class DatabaseAdapter(ABC):
    """Abstract interface for database adapters."""

    @property
    @abstractmethod
    def connection_string(self):
        """Return the connection string."""
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def execute(self, query):
        pass

    @abstractmethod
    def close(self):
        pass


class PostgresAdapter(DatabaseAdapter):
    def __init__(self, host, port, database):
        self.host = host
        self.port = port
        self.database = database
        self._connected = False

    @property
    def connection_string(self):
        return f"postgresql://{self.host}:{self.port}/{self.database}"

    def connect(self):
        print(f"Connecting to {self.connection_string}")
        self._connected = True

    def execute(self, query):
        if not self._connected:
            raise RuntimeError("Not connected")
        print(f"Executing: {query}")
        return []

    def close(self):
        self._connected = False
        print("Connection closed")


db = PostgresAdapter("localhost", 5432, "mydb")
db.connect()
db.execute("SELECT * FROM users")
db.close()
```

---

## 5. Polymorphism and Duck Typing

### Polymorphism

Polymorphism means "many forms" -- different classes can be used interchangeably if they share the same interface.

```python
class PaymentProcessor:
    def process(self, amount):
        raise NotImplementedError

class CreditCardProcessor(PaymentProcessor):
    def __init__(self, card_number):
        self.card_number = card_number

    def process(self, amount):
        return f"Charged ${amount:.2f} to card ending in {self.card_number[-4:]}"

class PayPalProcessor(PaymentProcessor):
    def __init__(self, email):
        self.email = email

    def process(self, amount):
        return f"Sent ${amount:.2f} via PayPal to {self.email}"

class CryptoProcessor(PaymentProcessor):
    def __init__(self, wallet_address):
        self.wallet = wallet_address

    def process(self, amount):
        return f"Transferred ${amount:.2f} in crypto to {self.wallet[:8]}..."


def checkout(processor, amount):
    """Works with ANY payment processor -- polymorphism in action."""
    print(processor.process(amount))

# Same function, different behaviors
checkout(CreditCardProcessor("4111111111111111"), 99.99)
checkout(PayPalProcessor("alice@example.com"), 49.50)
checkout(CryptoProcessor("0xABCDEF1234567890"), 150.00)
# Charged $99.99 to card ending in 1111
# Sent $49.50 via PayPal to alice@example.com
# Transferred $150.00 in crypto to 0xABCDEF...
```

### Duck Typing

"If it walks like a duck and quacks like a duck, it is a duck." Python does not check types -- it checks behavior.

```python
# These classes share no common parent, but all have a write() method
class FileWriter:
    def write(self, data):
        print(f"Writing to file: {data}")

class NetworkSender:
    def write(self, data):
        print(f"Sending over network: {data}")

class Logger:
    def write(self, data):
        print(f"[LOG] {data}")

class NullWriter:
    def write(self, data):
        pass  # Silently discard

def save_report(writer, report):
    """Works with anything that has a write() method."""
    writer.write(f"Report: {report}")

# All work, no shared base class needed
save_report(FileWriter(), "Q4 Results")
save_report(NetworkSender(), "Q4 Results")
save_report(Logger(), "Q4 Results")
save_report(NullWriter(), "Q4 Results")
```

### Protocol Classes (Structural Subtyping)

Python 3.8+ offers `Protocol` for explicit duck typing with static type checking.

```python
from typing import Protocol

class Renderable(Protocol):
    def render(self) -> str:
        ...

class HTMLPage:
    def render(self) -> str:
        return "<html><body>Hello</body></html>"

class JSONResponse:
    def render(self) -> str:
        return '{"message": "Hello"}'

def display(item: Renderable) -> None:
    """Type checker verifies that item has render() method."""
    print(item.render())

# Both work -- they satisfy the Renderable protocol structurally
display(HTMLPage())
display(JSONResponse())
```

---

## 6. Operator Overloading

Operator overloading lets your objects work with Python operators (`+`, `-`, `==`, `<`, `[]`, etc.) by implementing special (dunder) methods.

### Arithmetic Operators

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """self + other"""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        """self - other"""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        """self * scalar"""
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        """scalar * self (reflected multiplication)"""
        return self.__mul__(scalar)

    def __neg__(self):
        """-self"""
        return Vector(-self.x, -self.y)

    def __abs__(self):
        """abs(self) -- magnitude"""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4, 6)
print(v1 - v2)      # Vector(2, 2)
print(v1 * 3)       # Vector(9, 12)
print(2 * v1)       # Vector(6, 8) -- uses __rmul__
print(-v1)           # Vector(-3, -4)
print(abs(v1))       # 5.0
```

### Comparison Operators

```python
class Student:
    def __init__(self, name, gpa):
        self.name = name
        self.gpa = gpa

    def __eq__(self, other):
        """self == other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa == other.gpa

    def __lt__(self, other):
        """self < other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa < other.gpa

    def __le__(self, other):
        """self <= other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa <= other.gpa

    def __gt__(self, other):
        """self > other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa > other.gpa

    def __ge__(self, other):
        """self >= other"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.gpa >= other.gpa

    def __repr__(self):
        return f"Student({self.name!r}, gpa={self.gpa})"

students = [
    Student("Alice", 3.8),
    Student("Bob", 3.5),
    Student("Charlie", 3.9),
    Student("Diana", 3.5),
]

# Sorting uses __lt__
print(sorted(students))
# [Student('Bob', gpa=3.5), Student('Diana', gpa=3.5),
#  Student('Alice', gpa=3.8), Student('Charlie', gpa=3.9)]

print(Student("Alice", 3.8) == Student("Bob", 3.8))  # True (same GPA)
print(Student("Alice", 3.8) > Student("Bob", 3.5))   # True
```

### Using `functools.total_ordering`

You only need to define `__eq__` and one ordering method; `total_ordering` derives the rest.

```python
from functools import total_ordering

@total_ordering
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __eq__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    def __repr__(self):
        return f"Temperature({self.celsius}°C)"

t1 = Temperature(100)
t2 = Temperature(37)

print(t1 > t2)    # True  (derived from __lt__ and __eq__)
print(t1 >= t2)   # True
print(t2 <= t1)   # True
```

### Container Operators: `__len__` and `__getitem__`

```python
class Playlist:
    def __init__(self, name):
        self.name = name
        self._songs = []

    def add(self, song):
        self._songs.append(song)
        return self  # For chaining

    def __len__(self):
        """len(playlist)"""
        return len(self._songs)

    def __getitem__(self, index):
        """playlist[index] and playlist[start:stop]"""
        return self._songs[index]

    def __setitem__(self, index, value):
        """playlist[index] = value"""
        self._songs[index] = value

    def __delitem__(self, index):
        """del playlist[index]"""
        del self._songs[index]

    def __contains__(self, song):
        """song in playlist"""
        return song in self._songs

    def __iter__(self):
        """for song in playlist"""
        return iter(self._songs)

    def __repr__(self):
        return f"Playlist({self.name!r}, {len(self)} songs)"


pl = Playlist("Road Trip")
pl.add("Hotel California").add("Bohemian Rhapsody").add("Stairway to Heaven")

print(len(pl))           # 3
print(pl[0])             # Hotel California
print(pl[-1])            # Stairway to Heaven
print(pl[0:2])           # ['Hotel California', 'Bohemian Rhapsody']
print("Bohemian Rhapsody" in pl)  # True

for song in pl:
    print(f"  Playing: {song}")

pl[1] = "We Will Rock You"
print(pl[1])             # We Will Rock You
```

### Callable Objects: `__call__`

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return self.n + x

add_five = Adder(5)
print(add_five(10))     # 15
print(add_five(20))     # 25
print(callable(add_five))  # True


class Polynomial:
    """Represent a polynomial and evaluate it at a point."""

    def __init__(self, *coefficients):
        # coefficients are in order: a0 + a1*x + a2*x^2 + ...
        self.coefficients = coefficients

    def __call__(self, x):
        return sum(c * x ** i for i, c in enumerate(self.coefficients))

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x")
            else:
                terms.append(f"{c}x^{i}")
        return " + ".join(terms) if terms else "0"

# p(x) = 1 + 2x + 3x^2
p = Polynomial(1, 2, 3)
print(p)         # 1 + 2x + 3x^2
print(p(0))      # 1
print(p(1))      # 6  (1 + 2 + 3)
print(p(2))      # 17 (1 + 4 + 12)
```

### Common Dunder Methods Reference

| Operator/Builtin | Method | Example |
|---------|--------|---------|
| `+` | `__add__` | `a + b` |
| `-` | `__sub__` | `a - b` |
| `*` | `__mul__` | `a * b` |
| `/` | `__truediv__` | `a / b` |
| `//` | `__floordiv__` | `a // b` |
| `%` | `__mod__` | `a % b` |
| `**` | `__pow__` | `a ** b` |
| `==` | `__eq__` | `a == b` |
| `!=` | `__ne__` | `a != b` |
| `<` | `__lt__` | `a < b` |
| `<=` | `__le__` | `a <= b` |
| `>` | `__gt__` | `a > b` |
| `>=` | `__ge__` | `a >= b` |
| `len()` | `__len__` | `len(a)` |
| `[]` | `__getitem__` | `a[key]` |
| `[]=` | `__setitem__` | `a[key] = val` |
| `del[]` | `__delitem__` | `del a[key]` |
| `in` | `__contains__` | `x in a` |
| `()` | `__call__` | `a()` |
| `str()` | `__str__` | `str(a)` |
| `repr()` | `__repr__` | `repr(a)` |
| `bool()` | `__bool__` | `if a:` |
| `hash()` | `__hash__` | `hash(a)` |
| `iter()` | `__iter__` | `for x in a:` |
| `next()` | `__next__` | `next(a)` |

---

## 7. Composition vs Inheritance

### Inheritance: "is-a" Relationship

```python
class Engine:
    def start(self):
        return "Engine started"

class ElectricEngine(Engine):
    def start(self):
        return "Electric engine humming"
```

A `ElectricEngine` **is an** `Engine`.

### Composition: "has-a" Relationship

```python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower

    def start(self):
        return f"{self.horsepower}HP engine started"

class Transmission:
    def __init__(self, type_name):
        self.type_name = type_name

    def shift(self, gear):
        return f"{self.type_name} shifting to gear {gear}"

class GPS:
    def navigate(self, destination):
        return f"Navigating to {destination}"


class Car:
    """Car is composed of engine, transmission, and optional GPS."""

    def __init__(self, make, model, engine, transmission, gps=None):
        self.make = make
        self.model = model
        self.engine = engine            # has-a Engine
        self.transmission = transmission # has-a Transmission
        self.gps = gps                  # has-a GPS (optional)

    def start(self):
        return f"{self.make} {self.model}: {self.engine.start()}"

    def drive(self, gear, destination=None):
        actions = [self.transmission.shift(gear)]
        if destination and self.gps:
            actions.append(self.gps.navigate(destination))
        return " | ".join(actions)


# Compose a car from parts
car = Car(
    "Toyota", "Camry",
    engine=Engine(203),
    transmission=Transmission("automatic"),
    gps=GPS()
)

print(car.start())
# Toyota Camry: 203HP engine started

print(car.drive(3, "Seoul"))
# automatic shifting to gear 3 | Navigating to Seoul
```

### When to Use Each

| Prefer Inheritance | Prefer Composition |
|---|---|
| True "is-a" relationship | "has-a" or "uses-a" relationship |
| Sharing behavior across a type hierarchy | Combining independent capabilities |
| Framework expects it (e.g., ABC, Django models) | Need to swap components at runtime |
| Shallow hierarchy (1-2 levels) | Deep hierarchies become fragile |

### The Composition Over Inheritance Principle

```python
# Instead of a complex inheritance tree:
# Animal -> FlyingAnimal -> FlyingSwimmingAnimal -> ...

# Use composition with capability objects:
class FlyAbility:
    def fly(self, owner):
        return f"{owner.name} soars through the sky"

class SwimAbility:
    def swim(self, owner):
        return f"{owner.name} glides through the water"

class RunAbility:
    def run(self, owner):
        return f"{owner.name} runs swiftly"


class Animal:
    def __init__(self, name, abilities=None):
        self.name = name
        self.abilities = abilities or []

    def perform(self, action):
        for ability in self.abilities:
            method = getattr(ability, action, None)
            if method:
                return method(self)
        return f"{self.name} cannot {action}"


duck = Animal("Duck", [FlyAbility(), SwimAbility(), RunAbility()])
eagle = Animal("Eagle", [FlyAbility()])
fish = Animal("Fish", [SwimAbility()])

print(duck.perform("fly"))    # Duck soars through the sky
print(duck.perform("swim"))   # Duck glides through the water
print(eagle.perform("swim"))  # Eagle cannot swim
print(fish.perform("swim"))   # Fish glides through the water
```

---

## 8. Dataclasses

The `@dataclass` decorator (Python 3.7+) automatically generates `__init__`, `__repr__`, `__eq__`, and optionally other methods.

### Basic Dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Automatically generates __init__, __repr__, __eq__
p1 = Point(3.0, 4.0)
p2 = Point(3.0, 4.0)

print(p1)          # Point(x=3.0, y=4.0)
print(p1 == p2)    # True (compares all fields)
print(p1.x)        # 3.0
```

### Default Values

```python
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int
    grade: float = 0.0
    courses: list = field(default_factory=list)  # Mutable default

s = Student("Alice", 20)
s.courses.append("Math")
print(s)  # Student(name='Alice', age=20, grade=0.0, courses=['Math'])

# Each instance gets its own list
s2 = Student("Bob", 22)
print(s2.courses)  # [] (independent)
```

### Frozen Dataclass (Immutable)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Color:
    red: int
    green: int
    blue: int

c = Color(255, 128, 0)
print(c)  # Color(red=255, green=128, blue=0)

# Cannot modify
# c.red = 200  # FrozenInstanceError

# Frozen dataclasses are hashable (can be dict keys or set elements)
colors = {Color(255, 0, 0): "red", Color(0, 255, 0): "green"}
print(colors[Color(255, 0, 0)])  # red
```

### Ordering with Dataclasses

```python
from dataclasses import dataclass

@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

versions = [Version(2, 1, 0), Version(1, 9, 5), Version(2, 0, 3)]
print(sorted(versions))
# [Version(major=1, minor=9, patch=5), Version(major=2, minor=0, patch=3),
#  Version(major=2, minor=1, patch=0)]
```

### Post-Init Processing

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not in __init__, computed

    def __post_init__(self):
        """Called after __init__."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")
        self.area = self.width * self.height

r = Rectangle(5, 3)
print(r)       # Rectangle(width=5, height=3, area=15)
print(r.area)  # 15
```

### Dataclass Inheritance

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: str
    department: str
    salary: float = 50000.0

emp = Employee("Alice", 30, "E001", "Engineering", 75000)
print(emp)
# Employee(name='Alice', age=30, employee_id='E001',
#          department='Engineering', salary=75000)
```

---

## 9. Common Design Patterns

### Factory Pattern

The Factory pattern creates objects without exposing the creation logic.

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message):
        pass

class EmailNotification(Notification):
    def __init__(self, email):
        self.email = email

    def send(self, message):
        return f"Email to {self.email}: {message}"

class SMSNotification(Notification):
    def __init__(self, phone):
        self.phone = phone

    def send(self, message):
        return f"SMS to {self.phone}: {message}"

class PushNotification(Notification):
    def __init__(self, device_id):
        self.device_id = device_id

    def send(self, message):
        return f"Push to {self.device_id}: {message}"


class NotificationFactory:
    """Factory class to create notification objects."""

    _registry = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, channel, destination):
        """Create a notification based on channel type."""
        notification_class = cls._registry.get(channel)
        if notification_class is None:
            raise ValueError(f"Unknown channel: {channel}")
        return notification_class(destination)

    @classmethod
    def register(cls, channel, notification_class):
        """Register a new notification type."""
        cls._registry[channel] = notification_class


# Usage
notif = NotificationFactory.create("email", "alice@example.com")
print(notif.send("Hello!"))
# Email to alice@example.com: Hello!

notif = NotificationFactory.create("sms", "+1-555-0100")
print(notif.send("Hello!"))
# SMS to +1-555-0100: Hello!

# Send to multiple channels
def broadcast(message, targets):
    """Send message to multiple channels."""
    for channel, destination in targets:
        notif = NotificationFactory.create(channel, destination)
        print(notif.send(message))

targets = [
    ("email", "alice@example.com"),
    ("sms", "+1-555-0100"),
    ("push", "device-abc-123"),
]
broadcast("System maintenance at 10 PM", targets)
```

### Singleton Pattern

The Singleton pattern ensures a class has only one instance.

```python
class Singleton:
    """Basic singleton using __new__."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        # __init__ is called every time, so guard against re-init
        if not hasattr(self, "_initialized"):
            self.value = value
            self._initialized = True


s1 = Singleton("first")
s2 = Singleton("second")

print(s1 is s2)       # True (same instance)
print(s1.value)        # first (not overwritten)
print(id(s1) == id(s2))  # True
```

### Singleton via Decorator

```python
def singleton(cls):
    """Decorator that turns a class into a singleton."""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class AppConfig:
    def __init__(self):
        self.settings = {}

    def set(self, key, value):
        self.settings[key] = value

    def get(self, key, default=None):
        return self.settings.get(key, default)


config1 = AppConfig()
config1.set("debug", True)

config2 = AppConfig()
print(config2.get("debug"))  # True (same instance)
print(config1 is config2)    # True
```

---

## 10. Putting It All Together

### Example: Plugin System Using ABC, Factory, and Duck Typing

```python
from abc import ABC, abstractmethod

class Plugin(ABC):
    """Abstract base for all plugins."""

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def execute(self, data):
        pass


class UpperCasePlugin(Plugin):
    @property
    def name(self):
        return "uppercase"

    def execute(self, data):
        return data.upper()


class ReversePlugin(Plugin):
    @property
    def name(self):
        return "reverse"

    def execute(self, data):
        return data[::-1]


class CensorPlugin(Plugin):
    def __init__(self, banned_words=None):
        self.banned_words = banned_words or []

    @property
    def name(self):
        return "censor"

    def execute(self, data):
        result = data
        for word in self.banned_words:
            result = result.replace(word, "*" * len(word))
        return result


class PluginManager:
    """Manage and run plugins in sequence."""

    def __init__(self):
        self._plugins = []

    def register(self, plugin):
        if not isinstance(plugin, Plugin):
            raise TypeError(f"Expected Plugin, got {type(plugin).__name__}")
        self._plugins.append(plugin)
        print(f"Registered plugin: {plugin.name}")

    def process(self, data):
        result = data
        for plugin in self._plugins:
            result = plugin.execute(result)
        return result

    def list_plugins(self):
        return [p.name for p in self._plugins]


# Build a processing pipeline
manager = PluginManager()
manager.register(CensorPlugin(["bad", "ugly"]))
manager.register(UpperCasePlugin())

text = "This is a bad and ugly example"
result = manager.process(text)
print(f"Original: {text}")
print(f"Processed: {result}")
# Original: This is a bad and ugly example
# Processed: THIS IS A *** AND **** EXAMPLE
```

---

## 11. Summary

| Concept | Key Points |
|---------|------------|
| Inheritance | `class Child(Parent):`; use `super()` for parent method calls |
| Method overriding | Subclass redefines parent method; MRO determines resolution order |
| Multiple inheritance | Multiple parents separated by commas; C3 linearization (MRO) |
| ABC (`abc` module) | `@abstractmethod` enforces interface; cannot instantiate directly |
| Polymorphism | Different classes, same interface; code works with any conforming object |
| Duck typing | Check behavior, not type; "if it has `.write()`, it is a writer" |
| Operator overloading | Dunder methods (`__add__`, `__eq__`, `__getitem__`, etc.) |
| Composition | "has-a" relationship; objects contain other objects |
| `@dataclass` | Auto-generates `__init__`, `__repr__`, `__eq__`; use `field()` for defaults |
| Factory pattern | Centralized object creation based on parameters |
| Singleton pattern | Ensures a single instance of a class |

---

## Exercises

1. Create an `Animal` hierarchy with `Dog`, `Cat`, and `Bird` subclasses. Each should have a `speak()` and `move()` method. Use a function that takes any `Animal` and calls both methods (polymorphism).
2. Define an abstract `Serializer` class with `serialize(data)` and `deserialize(text)` methods. Implement `JSONSerializer` and `CSVSerializer` subclasses.
3. Create a `Money` class with operator overloading: support `+`, `-`, `*` (scalar), `==`, `<`, and `str()`. Handle currency (e.g., `Money(10, "USD") + Money(5, "USD")`).
4. Build a `TaskQueue` using composition: it should contain a list of `Task` dataclasses (with name, priority, status) and support `add`, `pop_highest_priority`, `__len__`, and `__iter__`.
5. Implement a simple observer pattern: `EventEmitter` class that supports `on(event, callback)`, `off(event, callback)`, and `emit(event, *args)`. Test with multiple listeners.

---

**Previous**: [OOP Basics](./08_OOP_Basics.md) | **Next**: [Modules and Packages](./10_Modules_and_Packages.md)

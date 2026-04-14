# Lesson 10: SOLID Principles

## Learning Objectives

By the end of this lesson, you will be able to:
1. Name and explain each of the five SOLID principles
2. Identify violations of SOLID principles in existing code
3. Refactor code to follow the Single Responsibility Principle (SRP)
4. Design classes that are open for extension but closed for modification (OCP)
5. Apply the Liskov Substitution Principle to validate inheritance hierarchies
6. Split bloated interfaces following the Interface Segregation Principle (ISP)
7. Invert dependencies using the Dependency Inversion Principle (DIP)

## What Is SOLID?

SOLID is an acronym for five design principles introduced by Robert C. Martin ("Uncle Bob") that make object-oriented systems more maintainable, flexible, and understandable.

```
┌───┬──────────────────────────────────────────────┐
│ S │ Single Responsibility Principle              │
│   │ A class should have only ONE reason to change│
├───┼──────────────────────────────────────────────┤
│ O │ Open/Closed Principle                        │
│   │ Open for extension, closed for modification  │
├───┼──────────────────────────────────────────────┤
│ L │ Liskov Substitution Principle                │
│   │ Subtypes must be substitutable for supertypes│
├───┼──────────────────────────────────────────────┤
│ I │ Interface Segregation Principle              │
│   │ No client should depend on unused methods    │
├───┼──────────────────────────────────────────────┤
│ D │ Dependency Inversion Principle               │
│   │ Depend on abstractions, not concretions      │
└───┴──────────────────────────────────────────────┘
```

## S — Single Responsibility Principle (SRP)

> "A class should have only one reason to change."

Each class should do **one thing** and do it well.

```python
# BAD: God class with multiple responsibilities
class UserManager:
    """Handles user data, email, persistence, AND reporting."""

    def __init__(self, name, email):
        self.name = name
        self.email = email

    # Responsibility 1: User validation
    def validate_email(self):
        return "@" in self.email

    # Responsibility 2: Database operations
    def save_to_database(self):
        print(f"INSERT INTO users VALUES ('{self.name}', '{self.email}')")

    # Responsibility 3: Email sending
    def send_welcome_email(self):
        print(f"Sending welcome email to {self.email}")

    # Responsibility 4: Report generation
    def generate_report(self):
        return f"User Report: {self.name} ({self.email})"
```

```python
# GOOD: Each class has a single responsibility

class User:
    """Responsibility: Represent user data and validation."""

    def __init__(self, name, email):
        self.name = name
        self.email = email

    def validate_email(self):
        return "@" in self.email


class UserRepository:
    """Responsibility: Persist user data."""

    def save(self, user: User):
        print(f"Saving {user.name} to database")

    def find_by_email(self, email: str):
        print(f"Finding user with email {email}")


class EmailService:
    """Responsibility: Send emails."""

    def send_welcome(self, user: User):
        print(f"Welcome email sent to {user.email}")


class UserReportGenerator:
    """Responsibility: Generate reports."""

    def generate(self, user: User):
        return f"Report: {user.name} ({user.email})"
```

## O — Open/Closed Principle (OCP)

> "Software entities should be open for extension but closed for modification."

You should be able to add new behavior **without changing existing code**.

```python
# BAD: Must modify this function every time we add a new shape
def calculate_area_bad(shape):
    if shape["type"] == "circle":
        return 3.14 * shape["radius"] ** 2
    elif shape["type"] == "rectangle":
        return shape["width"] * shape["height"]
    elif shape["type"] == "triangle":  # Had to modify existing code!
        return 0.5 * shape["base"] * shape["height"]
    # Every new shape requires modifying this function
```

```python
# GOOD: Open for extension (add new shapes), closed for modification

from abc import ABC, abstractmethod
from math import pi


class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


# Adding a new shape requires NO changes to existing code
class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height


# This function NEVER needs to change
def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
```

## L — Liskov Substitution Principle (LSP)

> "Objects of a superclass should be replaceable with objects of a subclass without breaking the program."

If `Dog` extends `Animal`, then anywhere you use `Animal`, you should be able to use `Dog` without issues.

```python
# BAD: Violates LSP — Square breaks Rectangle's contract

class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    def area(self):
        return self._width * self._height


class Square(Rectangle):
    """Violates LSP: changing width also changes height!"""

    def __init__(self, side):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value):
        self._width = value
        self._height = value  # Unexpected side effect!

    @Rectangle.height.setter
    def height(self, value):
        self._width = value  # Unexpected side effect!
        self._height = value


# Code that works with Rectangle BREAKS with Square
def test_rectangle(rect):
    rect.width = 5
    rect.height = 10
    assert rect.area() == 50  # Fails for Square! area = 100
```

```python
# GOOD: Separate classes or use composition

from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2


# Both satisfy the Shape contract — LSP preserved
shapes = [Rectangle(5, 10), Square(7)]
for s in shapes:
    print(f"{s.__class__.__name__}: area = {s.area()}")
```

### LSP Checklist

```
┌─────────────────────────────────────────────────────┐
│  A subclass MUST:                                   │
│  1. Accept all inputs the parent accepts            │
│  2. Return compatible output types                  │
│  3. Not strengthen preconditions                    │
│  4. Not weaken postconditions                       │
│  5. Not throw unexpected exceptions                 │
│  6. Preserve the parent's invariants                │
└─────────────────────────────────────────────────────┘
```

## I — Interface Segregation Principle (ISP)

> "No client should be forced to depend on methods it does not use."

Split large interfaces into smaller, more specific ones.

```python
# BAD: Fat interface — not all workers need all methods

from abc import ABC, abstractmethod


class Worker(ABC):
    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass

    @abstractmethod
    def program(self):
        pass

    @abstractmethod
    def manage(self):
        pass


# A janitor doesn't program or manage!
class Janitor(Worker):
    def work(self):
        return "Cleaning"

    def eat(self):
        return "Eating lunch"

    def sleep(self):
        return "Sleeping"

    def program(self):
        raise NotImplementedError("Janitors don't program!")  # ISP violation

    def manage(self):
        raise NotImplementedError("Janitors don't manage!")  # ISP violation
```

```python
# GOOD: Segregated interfaces

from abc import ABC, abstractmethod


class Workable(ABC):
    @abstractmethod
    def work(self):
        pass


class Feedable(ABC):
    @abstractmethod
    def eat(self):
        pass


class Programmable(ABC):
    @abstractmethod
    def program(self):
        pass


class Manageable(ABC):
    @abstractmethod
    def manage(self):
        pass


# Each class implements ONLY the interfaces it needs
class Janitor(Workable, Feedable):
    def work(self):
        return "Cleaning"

    def eat(self):
        return "Eating lunch"


class Developer(Workable, Feedable, Programmable):
    def work(self):
        return "Developing software"

    def eat(self):
        return "Eating at desk"

    def program(self):
        return "Writing Python code"


class Manager(Workable, Feedable, Manageable):
    def work(self):
        return "Attending meetings"

    def eat(self):
        return "Business lunch"

    def manage(self):
        return "Managing team"
```

## D — Dependency Inversion Principle (DIP)

> "High-level modules should not depend on low-level modules. Both should depend on abstractions."

```
┌─────────────────────────────────────────────────┐
│  WITHOUT DIP:                                   │
│                                                 │
│  OrderService ──depends on──▶ MySQLDatabase     │
│  (high-level)                (low-level)        │
│                                                 │
│  Problem: Can't switch databases without        │
│  modifying OrderService                         │
├─────────────────────────────────────────────────┤
│  WITH DIP:                                      │
│                                                 │
│  OrderService ──depends on──▶ Database (ABC)    │
│  (high-level)                (abstraction)      │
│                                  ▲              │
│                                  │              │
│                        ┌─────────┼─────────┐    │
│                        │         │         │    │
│                   MySQL      Postgres   SQLite  │
│                   (low-level implementations)   │
└─────────────────────────────────────────────────┘
```

```python
# BAD: High-level module depends on low-level module

class MySQLDatabase:
    def query(self, sql):
        return f"MySQL: {sql}"

class OrderService:
    def __init__(self):
        self.db = MySQLDatabase()  # Tightly coupled!

    def get_orders(self):
        return self.db.query("SELECT * FROM orders")
```

```python
# GOOD: Both depend on an abstraction

from abc import ABC, abstractmethod


class Database(ABC):
    """Abstraction that both sides depend on."""

    @abstractmethod
    def query(self, sql: str) -> str:
        pass


class MySQLDatabase(Database):
    def query(self, sql):
        return f"MySQL: {sql}"


class PostgresDatabase(Database):
    def query(self, sql):
        return f"Postgres: {sql}"


class SQLiteDatabase(Database):
    def query(self, sql):
        return f"SQLite: {sql}"


class OrderService:
    """High-level module depends on abstraction, not concrete class."""

    def __init__(self, db: Database):  # Inject the dependency!
        self.db = db

    def get_orders(self):
        return self.db.query("SELECT * FROM orders")


# Easy to swap implementations
service_mysql = OrderService(MySQLDatabase())
service_postgres = OrderService(PostgresDatabase())
service_sqlite = OrderService(SQLiteDatabase())

print(service_mysql.get_orders())     # MySQL: SELECT * FROM orders
print(service_postgres.get_orders())  # Postgres: SELECT * FROM orders
```

### Dependency Injection

DIP is typically implemented through **dependency injection** — passing dependencies from outside rather than creating them inside:

```python
class NotificationService:
    """Depends on abstractions, receives concrete implementations."""

    def __init__(self, sender, logger, formatter):
        self._sender = sender      # Injected
        self._logger = logger      # Injected
        self._formatter = formatter # Injected

    def notify(self, user, message):
        formatted = self._formatter.format(message)
        self._sender.send(user, formatted)
        self._logger.log(f"Notified {user}")
```

## SOLID in Practice

```python
from abc import ABC, abstractmethod


# S: Each class has one responsibility
# O: New payment methods don't require changing existing code
# L: All payment methods are substitutable
# I: PaymentMethod interface is focused
# D: PaymentProcessor depends on abstraction

class PaymentMethod(ABC):
    """Focused interface (ISP) and abstraction (DIP)."""

    @abstractmethod
    def charge(self, amount: float) -> bool:
        pass

    @abstractmethod
    def refund(self, amount: float) -> bool:
        pass


class CreditCard(PaymentMethod):
    """Single responsibility: credit card payments."""

    def __init__(self, number):
        self.number = number[-4:]

    def charge(self, amount):
        print(f"Charged ${amount:.2f} to card ***{self.number}")
        return True

    def refund(self, amount):
        print(f"Refunded ${amount:.2f} to card ***{self.number}")
        return True


class PayPal(PaymentMethod):
    def __init__(self, email):
        self.email = email

    def charge(self, amount):
        print(f"Charged ${amount:.2f} via PayPal ({self.email})")
        return True

    def refund(self, amount):
        print(f"Refunded ${amount:.2f} via PayPal ({self.email})")
        return True


class PaymentProcessor:
    """Depends on PaymentMethod abstraction (DIP), not concrete classes."""

    def __init__(self, payment_method: PaymentMethod):
        self._method = payment_method

    def process(self, amount: float):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return self._method.charge(amount)

    def cancel(self, amount: float):
        return self._method.refund(amount)


# Open for extension: add new payment methods without modifying existing code
processor = PaymentProcessor(CreditCard("4111111111111234"))
processor.process(99.99)
```

## Summary

- **SRP**: One class, one responsibility, one reason to change
- **OCP**: Extend behavior by adding new classes, not modifying existing ones
- **LSP**: Subclasses must be fully substitutable for their parent classes
- **ISP**: Prefer many small, focused interfaces over one large interface
- **DIP**: Depend on abstractions (interfaces/ABCs), inject concrete implementations
- SOLID principles work together to create maintainable, testable, extensible systems
- Apply SOLID pragmatically — not every class needs all five principles

## Next Steps

In [Lesson 11: Design Patterns Intro](11_Design_Patterns_Intro.md), we will explore classic design patterns that embody SOLID principles.

"""
Example 10: SOLID Principles
Topic: Object-Oriented Programming

Demonstrates all five SOLID principles with before/after examples.
"""

from abc import ABC, abstractmethod


# =============================================================================
# SRP: Single Responsibility Principle
# =============================================================================

class User:
    """SRP: Only responsible for user data."""
    def __init__(self, name, email):
        self.name = name
        self.email = email


class UserRepository:
    """SRP: Only responsible for persistence."""
    def __init__(self):
        self._users = {}

    def save(self, user):
        self._users[user.email] = user
        print(f"  Saved {user.name}")

    def find(self, email):
        return self._users.get(email)


class EmailService:
    """SRP: Only responsible for sending emails."""
    def send(self, to, subject, body):
        print(f"  Email to {to}: [{subject}] {body}")


# =============================================================================
# OCP: Open/Closed Principle
# =============================================================================

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        from math import pi
        return pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h
    def area(self):
        return self.w * self.h

# OCP: This function never changes when new shapes are added
def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)


# =============================================================================
# DIP: Dependency Inversion Principle
# =============================================================================

class Database(ABC):
    @abstractmethod
    def query(self, sql: str) -> list:
        pass

class PostgresDB(Database):
    def query(self, sql):
        return [f"Postgres result for: {sql}"]

class SQLiteDB(Database):
    def query(self, sql):
        return [f"SQLite result for: {sql}"]

class OrderService:
    """DIP: Depends on Database abstraction, not concrete class."""

    def __init__(self, db: Database):
        self.db = db  # Injected!

    def get_orders(self):
        return self.db.query("SELECT * FROM orders")


if __name__ == "__main__":
    # SRP
    print("=== SRP: Single Responsibility ===")
    user = User("Alice", "alice@example.com")
    repo = UserRepository()
    email = EmailService()

    repo.save(user)
    email.send(user.email, "Welcome", f"Hello {user.name}!")

    # OCP
    print("\n=== OCP: Open/Closed ===")
    shapes = [Circle(5), Rectangle(4, 6)]
    print(f"Total area: {total_area(shapes):.2f}")

    # Adding new shape without modifying total_area
    class Hexagon(Shape):
        def __init__(self, side):
            self.side = side
        def area(self):
            from math import sqrt
            return (3 * sqrt(3) / 2) * self.side ** 2

    shapes.append(Hexagon(3))
    print(f"With hexagon: {total_area(shapes):.2f}")

    # DIP
    print("\n=== DIP: Dependency Inversion ===")
    pg_service = OrderService(PostgresDB())
    sqlite_service = OrderService(SQLiteDB())

    print(f"Postgres: {pg_service.get_orders()}")
    print(f"SQLite: {sqlite_service.get_orders()}")

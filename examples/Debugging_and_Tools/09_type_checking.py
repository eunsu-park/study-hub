"""
09 Type Checking
================
Demonstrates Python type hints, common patterns, and how
type checking catches bugs before runtime.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TypedDict


# --- Basic type hints ---

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def average(numbers: list[float]) -> float:
    """Calculate average of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot average empty list")
    return sum(numbers) / len(numbers)


# --- Optional and None handling ---

def find_user(user_id: int) -> str | None:
    """Look up a user by ID. Returns None if not found."""
    users = {1: "Alice", 2: "Bob", 3: "Charlie"}
    return users.get(user_id)


def safe_find_demo():
    """Show correct Optional handling."""
    print("=== Optional/None Handling ===")

    # Unsafe (mypy would flag this)
    user = find_user(999)
    # print(user.upper())  # Would crash if user is None!

    # Safe
    if user is not None:
        print(f"  Found: {user.upper()}")
    else:
        print("  User not found (handled safely)")

    # Safe with assertion
    user = find_user(1)
    assert user is not None, "User 1 should exist"
    print(f"  User 1: {user.upper()}")
    print()


# --- TypedDict ---

class UserProfile(TypedDict):
    """Typed dictionary for user profiles."""
    name: str
    age: int
    email: str


def format_user(user: UserProfile) -> str:
    """Format a user profile for display."""
    return f"{user['name']} (age {user['age']}, {user['email']})"


# --- Dataclass with types ---

@dataclass
class Point:
    """A 2D point with type annotations."""
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


# --- Type checking catches bugs ---

def type_bugs_demo():
    """Show bugs that type checking would catch."""
    print("=== Bugs Type Checking Catches ===")

    # Bug 1: Wrong return type
    def parse_age_buggy(text: str) -> int:
        if text.isdigit():
            return int(text)
        return None  # mypy: Incompatible return value type (got "None", expected "int")

    # Fixed:
    def parse_age(text: str) -> int:
        if text.isdigit():
            return int(text)
        raise ValueError(f"Invalid age: {text!r}")

    print("  Bug 1: Function returns None when declared to return int")
    print(f"    parse_age_buggy('abc') = {parse_age_buggy('abc')!r} (None, not int!)")
    try:
        parse_age("abc")
    except ValueError as e:
        print(f"    parse_age('abc') raises ValueError: {e}")

    # Bug 2: Wrong argument type
    def double(x: int) -> int:
        return x * 2

    result_ok = double(5)
    result_bug = double("5")  # mypy would flag: "str" instead of "int"
    print(f"\n  Bug 2: Wrong argument type")
    print(f"    double(5)   = {result_ok!r} (correct)")
    print(f"    double('5') = {result_bug!r} (string '55', not int 10!)")

    # Bug 3: Missing None check
    print(f"\n  Bug 3: Using Optional without None check")
    print(f"    find_user(999) = {find_user(999)!r}")
    print(f"    Calling .upper() on None would crash at runtime")
    print(f"    mypy catches: 'None has no attribute upper'")
    print()


def main():
    """Run all type checking demonstrations."""
    print("=== Basic Type Hints ===")
    print(f"  greet('Alice') = {greet('Alice')!r}")
    print(f"  add(2, 3) = {add(2, 3)}")
    print(f"  average([1.0, 2.0, 3.0]) = {average([1.0, 2.0, 3.0])}")
    print()

    safe_find_demo()

    print("=== TypedDict ===")
    user: UserProfile = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    print(f"  {format_user(user)}")
    print()

    print("=== Dataclass with Types ===")
    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)
    print(f"  {p1} to {p2}: distance = {p1.distance_to(p2):.1f}")
    print()

    type_bugs_demo()


if __name__ == "__main__":
    main()

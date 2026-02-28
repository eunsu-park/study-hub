"""
Exercises for Lesson 01: Type Hints
Topic: Python

Solutions to practice problems from the lesson.
"""

from typing import TypeVar, Callable, Optional, TypedDict


# === Exercise 1: Function Type Hints ===
# Problem: Add appropriate type hints to calculate_average

def calculate_average(numbers: list[float]) -> Optional[float]:
    """Return the mean of a list of numbers, or None if the list is empty.

    We use Optional[float] because the function legitimately returns None
    for empty input -- this makes the caller handle both cases explicitly.
    """
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def exercise_1():
    """Demonstrate typed calculate_average."""
    print(f"calculate_average([1, 2, 3, 4, 5]) = {calculate_average([1, 2, 3, 4, 5])}")
    print(f"calculate_average([]) = {calculate_average([])}")
    print(f"calculate_average([10.5, 20.3]) = {calculate_average([10.5, 20.3])}")


# === Exercise 2: Generic Function ===
# Problem: Write a generic function that finds the first element satisfying a condition

T = TypeVar("T")


def find_first(items: list[T], predicate: Callable[[T], bool]) -> Optional[T]:
    """Find the first element in items that satisfies predicate.

    We use TypeVar so the return type matches the element type of the input
    list -- the compiler knows find_first([1,2,3], ...) returns Optional[int].
    """
    for item in items:
        if predicate(item):
            return item
    return None


def exercise_2():
    """Demonstrate generic find_first."""
    result1 = find_first([1, 2, 3, 4], lambda x: x > 2)
    print(f"find_first([1, 2, 3, 4], x > 2) = {result1}")  # 3

    result2 = find_first(["a", "bb", "ccc"], lambda s: len(s) > 1)
    print(f'find_first(["a", "bb", "ccc"], len > 1) = {result2}')  # "bb"

    result3 = find_first([1, 2, 3], lambda x: x > 10)
    print(f"find_first([1, 2, 3], x > 10) = {result3}")  # None


# === Exercise 3: TypedDict ===
# Problem: Define a TypedDict representing a user profile
#   Required: id (int), username (str), email (str)
#   Optional: bio (str), avatar_url (str)

class UserProfileRequired(TypedDict):
    """Required fields for a user profile."""
    id: int
    username: str
    email: str


class UserProfile(UserProfileRequired, total=False):
    """Full user profile with optional fields.

    By inheriting from a total=True base (default) and setting total=False
    on this subclass, only the new fields become optional while the inherited
    fields remain required. This is the standard pattern for mixed
    required/optional TypedDicts.
    """
    bio: str
    avatar_url: str


def exercise_3():
    """Demonstrate UserProfile TypedDict."""
    # Minimal profile (required fields only)
    user1: UserProfile = {
        "id": 1,
        "username": "alice",
        "email": "alice@example.com",
    }
    print(f"User 1: {user1}")

    # Full profile (with optional fields)
    user2: UserProfile = {
        "id": 2,
        "username": "bob",
        "email": "bob@example.com",
        "bio": "Python developer",
        "avatar_url": "https://example.com/bob.png",
    }
    print(f"User 2: {user2}")


if __name__ == "__main__":
    print("=== Exercise 1: Function Type Hints ===")
    exercise_1()

    print("\n=== Exercise 2: Generic Function ===")
    exercise_2()

    print("\n=== Exercise 3: TypedDict ===")
    exercise_3()

    print("\nAll exercises completed!")

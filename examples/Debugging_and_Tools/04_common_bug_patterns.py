"""
04 Common Bug Patterns
======================
Demonstrates the most common Python bug patterns with
buggy code, explanations, and fixes.
"""
import copy
import math


def off_by_one_demos():
    """Demonstrate off-by-one error patterns and fixes."""
    print("=== Off-by-One Errors ===")

    # Range boundary
    print("Range boundary:")
    print(f"  range(1, 10) → {list(range(1, 10))}  (missing 10!)")
    print(f"  range(1, 11) → {list(range(1, 11))}  (correct)")

    # Fence post
    sections = 10
    print(f"\nFence post: {sections} sections need {sections + 1} posts")

    # Integer division for indexing
    data = [10, 20, 30, 40, 50]
    mid_wrong = len(data) / 2   # 2.5 (float!)
    mid_right = len(data) // 2  # 2 (int)
    print(f"\nMiddle index: / gives {mid_wrong} (float!), // gives {mid_right}")
    print()


def mutable_default_demo():
    """Demonstrate the mutable default argument bug."""
    print("=== Mutable Default Argument ===")

    # Buggy version
    def add_item_buggy(item, items=[]):
        items.append(item)
        return items

    r1 = add_item_buggy("a")
    r2 = add_item_buggy("b")
    r3 = add_item_buggy("c")
    print(f"Buggy: {r1}, {r2}, {r3}")
    print("  All return the same list! (shared default)")

    # Fixed version
    def add_item_fixed(item, items=None):
        if items is None:
            items = []
        items.append(item)
        return items

    r1 = add_item_fixed("a")
    r2 = add_item_fixed("b")
    r3 = add_item_fixed("c")
    print(f"Fixed: {r1}, {r2}, {r3}")
    print()


def aliasing_demo():
    """Demonstrate shared mutable state (aliasing) bugs."""
    print("=== Aliasing Bugs ===")

    # List aliasing
    original = [1, 2, 3]
    alias = original
    alias.append(4)
    print(f"List alias: original={original} (modified by alias!)")

    real_copy = original.copy()
    real_copy.append(5)
    print(f"Real copy:  original={original}, copy={real_copy}")

    # Nested list trap
    grid_buggy = [[0] * 3] * 3
    grid_buggy[0][0] = 1
    print(f"\nNested list bug: {grid_buggy}  (all rows changed!)")

    grid_fixed = [[0] * 3 for _ in range(3)]
    grid_fixed[0][0] = 1
    print(f"Nested list fix: {grid_fixed}  (only first row)")

    # Dict in loop
    users_buggy = []
    user = {}
    for name in ["Alice", "Bob", "Charlie"]:
        user["name"] = name
        users_buggy.append(user)
    print(f"\nDict reuse bug: {users_buggy}")

    users_fixed = []
    for name in ["Alice", "Bob", "Charlie"]:
        users_fixed.append({"name": name})
    print(f"Dict reuse fix: {users_fixed}")
    print()


def scope_demo():
    """Demonstrate variable scope issues."""
    print("=== Scope Issues ===")

    # UnboundLocalError
    counter = 0

    def increment_buggy():
        try:
            counter += 1
        except UnboundLocalError as e:
            return f"UnboundLocalError: {e}"

    print(f"UnboundLocalError: {increment_buggy()}")

    def increment_fixed(count):
        return count + 1

    counter = increment_fixed(counter)
    print(f"Fixed (pass & return): counter = {counter}")

    # Late binding closures
    functions_buggy = [lambda: i for i in range(5)]
    functions_fixed = [lambda i=i: i for i in range(5)]
    print(f"\nLate binding bug:   {[f() for f in functions_buggy]}")
    print(f"Late binding fix:   {[f() for f in functions_fixed]}")
    print()


def none_handling_demo():
    """Demonstrate None-related bugs and fixes."""
    print("=== None Handling ===")

    # Missing return
    def find_user(name, users):
        for user in users:
            if user["name"] == name:
                return user
        # Implicit return None

    users = [{"name": "Alice", "email": "a@test.com"}]
    result = find_user("Bob", users)
    print(f"find_user('Bob'): {result!r}")
    print(f"Safe access: {result['email'] if result else 'User not found'}")

    # Truthy/falsy confusion
    print("\nTruthy/falsy confusion:")
    for value in [0, "", [], None, False]:
        is_truthy = "truthy" if value else "falsy"
        is_none = "is None" if value is None else "not None"
        print(f"  {value!r:10s} → {is_truthy:6s}, {is_none}")
    print()


def numeric_pitfalls():
    """Demonstrate numeric comparison bugs."""
    print("=== Numeric Pitfalls ===")

    # Float comparison
    print(f"0.1 + 0.2 == 0.3 → {0.1 + 0.2 == 0.3} (False!)")
    print(f"0.1 + 0.2        → {0.1 + 0.2}")
    print(f"math.isclose()   → {math.isclose(0.1 + 0.2, 0.3)} (True)")

    # == vs is
    print(f"\n[1,2,3] == [1,2,3] → {[1,2,3] == [1,2,3]} (value equality)")
    print(f"[1,2,3] is [1,2,3] → {[1,2,3] is [1,2,3]} (identity)")
    print()


def iteration_pitfalls():
    """Demonstrate iteration-related bugs."""
    print("=== Iteration Pitfalls ===")

    # Modify during iteration
    numbers = [1, 2, 3, 4, 5, 6]
    numbers_copy = numbers.copy()
    for n in numbers_copy:
        if n % 2 == 0:
            numbers_copy.remove(n)
    print(f"Remove during iteration (buggy): {numbers_copy} (missed 6!)")

    numbers_fixed = [n for n in numbers if n % 2 != 0]
    print(f"List comprehension (fixed):      {numbers_fixed}")

    # Exhausted iterator
    gen = (x**2 for x in range(5))
    first = list(gen)
    second = list(gen)
    print(f"\nGenerator first use:  {first}")
    print(f"Generator second use: {second} (empty! exhausted)")
    print()


if __name__ == "__main__":
    off_by_one_demos()
    mutable_default_demo()
    aliasing_demo()
    scope_demo()
    none_handling_demo()
    numeric_pitfalls()
    iteration_pitfalls()

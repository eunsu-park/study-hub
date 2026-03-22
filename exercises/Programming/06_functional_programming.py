"""
Exercises for Lesson 06: Functional Programming Concepts
Topic: Programming

Solutions to practice problems from the lesson.
"""
import inspect
from functools import reduce


# === Exercise 1: Pure vs Impure ===
# Problem: Identify which functions are pure and why.

def exercise_1():
    """Solution: Classify functions as pure or impure with reasoning."""

    # The code to analyze (JavaScript, but logic is universal):
    # let count = 0;
    # function a(x) { return x * 2; }           -> PURE
    # function b(x) { count++; return x + count; } -> IMPURE
    # function c(x) { return Math.random() * x; }  -> IMPURE
    # function d(arr) { return arr.slice(1); }      -> PURE
    # function e(arr) { arr.push(0); return arr; }  -> IMPURE
    # function f(x) { console.log(x); return x; }   -> IMPURE

    analysis = {
        "a(x) = x * 2": {
            "pure": True,
            "reason": "Only depends on input x, no side effects, same input always gives same output",
        },
        "b(x) = count++; x + count": {
            "pure": False,
            "reason": "Reads AND modifies external state (count). Different result each call.",
        },
        "c(x) = Math.random() * x": {
            "pure": False,
            "reason": "Non-deterministic: Math.random() returns different values each call.",
        },
        "d(arr) = arr.slice(1)": {
            "pure": True,
            "reason": "slice() returns a NEW array, does not modify the original. Same input, same output.",
        },
        "e(arr) = arr.push(0); return arr": {
            "pure": False,
            "reason": "Mutates the input array (push modifies in place). Side effect on caller's data.",
        },
        "f(x) = console.log(x); return x": {
            "pure": False,
            "reason": "console.log is a side effect (I/O). Even though return value is deterministic.",
        },
    }

    for func, info in analysis.items():
        status = "PURE" if info["pure"] else "IMPURE"
        print(f"  {func}")
        print(f"    -> {status}: {info['reason']}")
        print()


# === Exercise 2: Implement map/filter/reduce ===
# Problem: Implement higher-order functions from scratch.

def exercise_2():
    """Solution: Custom implementations of map, filter, reduce."""

    def my_map(func, iterable):
        """
        Apply func to each element, return new list.
        Key: creates a NEW list (no mutation), applies transformation.
        """
        result = []
        for item in iterable:
            result.append(func(item))
        return result

    def my_filter(predicate, iterable):
        """
        Keep elements where predicate returns True.
        Key: predicate is a function that returns bool.
        """
        result = []
        for item in iterable:
            if predicate(item):
                result.append(item)
        return result

    def my_reduce(func, iterable, initial):
        """
        Fold a collection into a single value using a binary function.
        Key: accumulates result by applying func(acc, item) repeatedly.
        """
        accumulator = initial
        for item in iterable:
            accumulator = func(accumulator, item)
        return accumulator

    # Tests from the exercise
    assert my_map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert my_filter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
    assert my_reduce(lambda a, b: a + b, [1, 2, 3, 4], 0) == 10

    print("  my_map(x*2, [1,2,3]):", my_map(lambda x: x * 2, [1, 2, 3]))
    print("  my_filter(even, [1,2,3,4]):", my_filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))
    print("  my_reduce(+, [1,2,3,4], 0):", my_reduce(lambda a, b: a + b, [1, 2, 3, 4], 0))

    # Additional tests
    print("  my_map(str.upper, ['a','b']):", my_map(str.upper, ["a", "b"]))
    print("  my_reduce(max, [3,1,4,1,5], 0):", my_reduce(max, [3, 1, 4, 1, 5], 0))
    print("  All assertions passed!")


# === Exercise 3: Refactor to Functional Style ===
# Problem: Rewrite imperative processOrders in functional style.

def exercise_3():
    """Solution: Functional refactoring of order processing."""

    # Sample data (simulating the JavaScript objects)
    orders = [
        {"id": 1, "status": "completed", "amount": 100},
        {"id": 2, "status": "pending", "amount": 200},
        {"id": 3, "status": "completed", "amount": 300},
        {"id": 4, "status": "completed", "amount": 150},
        {"id": 5, "status": "cancelled", "amount": 50},
    ]

    # Functional approach: compose small, pure transformations.
    # Each step is a separate operation that can be tested independently.
    def process_orders(orders):
        """Process orders using functional composition."""
        # Step 1: Filter completed orders (pure: no mutation)
        valid_orders = list(filter(lambda o: o["status"] == "completed", orders))

        # Step 2: Calculate total and average (pure computation)
        total = sum(o["amount"] for o in valid_orders)
        avg = total / len(valid_orders) if valid_orders else 0

        # Step 3: Normalize amounts (pure: returns NEW dicts, no mutation)
        normalized = [
            {**order, "normalized": order["amount"] / avg}
            for order in valid_orders
        ]

        return normalized

    result = process_orders(orders)
    print("  Functional order processing:")
    for order in result:
        print(f"    Order {order['id']}: amount={order['amount']}, "
              f"normalized={order['normalized']:.2f}")

    # Verify original data is unchanged (no mutation)
    print(f"\n  Original orders unchanged: {'normalized' not in orders[0]}")


# === Exercise 4: Implement Curry ===
# Problem: Create a generic curry function.

def exercise_4():
    """Solution: Generic curry function using closures."""

    def curry(func):
        """
        Transform a multi-argument function into a chain of single-argument functions.

        Supports both full currying (f(1)(2)(3)) and partial application (f(1, 2)(3)).
        Uses inspect to determine the original function's arity (number of parameters).
        """
        # Get the number of parameters the function expects
        arity = len(inspect.signature(func).parameters)

        def curried(*args):
            if len(args) >= arity:
                # Enough arguments collected: call the original function
                return func(*args[:arity])
            else:
                # Not enough arguments: return a function that collects more
                def partial(*more_args):
                    return curried(*args, *more_args)
                return partial

        return curried

    # Test: curried addition
    def add(a, b, c):
        return a + b + c

    curried_add = curry(add)

    # All calling styles should work
    assert curried_add(1)(2)(3) == 6,    "Full currying failed"
    assert curried_add(1, 2)(3) == 6,    "Partial (2+1) failed"
    assert curried_add(1)(2, 3) == 6,    "Partial (1+2) failed"
    assert curried_add(1, 2, 3) == 6,    "All at once failed"

    print("  curried_add(1)(2)(3) =", curried_add(1)(2)(3))
    print("  curried_add(1, 2)(3) =", curried_add(1, 2)(3))
    print("  curried_add(1)(2, 3) =", curried_add(1)(2, 3))
    print("  curried_add(1, 2, 3) =", curried_add(1, 2, 3))

    # Practical use: create reusable partially-applied functions
    def multiply(a, b):
        return a * b

    curried_mul = curry(multiply)
    double = curried_mul(2)
    triple = curried_mul(3)

    print(f"\n  double = curry(multiply)(2)")
    print(f"  double(5) = {double(5)}")
    print(f"  triple(5) = {triple(5)}")
    print("  All assertions passed!")


# === Exercise 5: Build a Pipeline ===
# Problem: Create a data processing pipeline using composition.

def exercise_5():
    """Solution: Data pipeline using function composition."""

    # Utility: pipe function that chains operations left to right
    def pipe(*functions):
        """
        Compose functions left-to-right (data flows through each in order).
        pipe(f, g, h)(x) = h(g(f(x)))
        """
        def pipeline(data):
            result = data
            for func in functions:
                result = func(result)
            return result
        return pipeline

    # Sample data
    users = [
        {"name": "Alice", "email": "Alice@Example.com", "active": True},
        {"name": "Bob", "email": "BOB@test.com", "active": False},
        {"name": "Charlie", "email": "charlie@example.com", "active": True},
        {"name": "Diana", "email": "DIANA@Test.COM", "active": True},
        {"name": "Eve", "email": "Alice@Example.com", "active": True},  # Duplicate email
        {"name": "Frank", "email": "frank@demo.org", "active": True},
    ]

    # Step 1: Filter active users
    filter_active = lambda users: [u for u in users if u["active"]]

    # Step 2: Extract email addresses
    extract_emails = lambda users: [u["email"] for u in users]

    # Step 3: Normalize to lowercase
    normalize_lower = lambda emails: [e.lower() for e in emails]

    # Step 4: Remove duplicates (while preserving order)
    remove_duplicates = lambda emails: list(dict.fromkeys(emails))

    # Step 5: Sort alphabetically
    sort_alpha = lambda emails: sorted(emails)

    # Step 6: Join as comma-separated string
    join_csv = lambda emails: ", ".join(emails)

    # Compose the pipeline
    process_emails = pipe(
        filter_active,
        extract_emails,
        normalize_lower,
        remove_duplicates,
        sort_alpha,
        join_csv,
    )

    result = process_emails(users)
    print(f"  Input: {len(users)} users")
    print(f"  Pipeline result: {result}")

    # Show intermediate steps for clarity
    print("\n  Step-by-step:")
    data = users
    steps = [
        ("Filter active", filter_active),
        ("Extract emails", extract_emails),
        ("Normalize lowercase", normalize_lower),
        ("Remove duplicates", remove_duplicates),
        ("Sort alphabetically", sort_alpha),
        ("Join CSV", join_csv),
    ]
    for name, func in steps:
        data = func(data)
        print(f"    {name}: {data}")


# === Exercise 6: Implement Maybe Monad ===
# Problem: Implement a Maybe monad for safe null handling.

def exercise_6():
    """Solution: Maybe monad with map, flat_map, and get_or_else."""

    class Maybe:
        """
        Maybe monad for safe null handling.

        Wraps a value that might be None and provides chainable operations
        that automatically short-circuit on None (no NullPointerException).
        """

        def __init__(self, value):
            self._value = value

        @staticmethod
        def of(value):
            """Create a Maybe wrapping the given value (which may be None)."""
            return Maybe(value)

        def is_nothing(self):
            """Check if this Maybe contains no value."""
            return self._value is None

        def map(self, fn):
            """
            Apply fn to the contained value if present, wrap result in Maybe.
            If Nothing, returns Nothing without calling fn.
            This is the key monadic operation: it threads None safely.
            """
            if self.is_nothing():
                return Maybe.of(None)
            return Maybe.of(fn(self._value))

        def flat_map(self, fn):
            """
            Like map, but fn returns a Maybe. Avoids double-wrapping.
            Useful when the transformation itself might fail (return None).
            """
            if self.is_nothing():
                return Maybe.of(None)
            return fn(self._value)

        def get_or_else(self, default):
            """Return the value if present, otherwise return default."""
            if self.is_nothing():
                return default
            return self._value

        def __repr__(self):
            if self.is_nothing():
                return "Nothing"
            return f"Just({self._value!r})"

    # Demonstrate: safely navigate nested objects
    # Simulating: user.address.city.name
    user_data = {
        "name": "Alice",
        "address": {
            "city": {
                "name": "Seoul",
                "country": "Korea",
            }
        },
    }

    user_no_address = {"name": "Bob", "address": None}
    user_no_city = {"name": "Charlie", "address": {"city": None}}

    def get_city_name(user):
        """Safely extract city name using Maybe monad chaining."""
        return (
            Maybe.of(user)
            .map(lambda u: u.get("address"))
            .map(lambda a: a.get("city") if a else None)
            .map(lambda c: c.get("name") if c else None)
            .get_or_else("Unknown")
        )

    print("  Safe nested access with Maybe monad:")
    print(f"    Alice's city: {get_city_name(user_data)}")
    print(f"    Bob's city (no address): {get_city_name(user_no_address)}")
    print(f"    Charlie's city (no city): {get_city_name(user_no_city)}")
    print(f"    None user: {get_city_name(None)}")

    # flat_map example: chaining operations that might fail
    def safe_sqrt(x):
        """Returns Maybe: Some(sqrt) if non-negative, Nothing if negative."""
        if x < 0:
            return Maybe.of(None)
        return Maybe.of(x ** 0.5)

    print("\n  flat_map example (safe_sqrt):")
    print(f"    sqrt(25): {Maybe.of(25).flat_map(safe_sqrt)}")
    print(f"    sqrt(-4): {Maybe.of(-4).flat_map(safe_sqrt)}")
    print(f"    sqrt(None): {Maybe.of(None).flat_map(safe_sqrt)}")


if __name__ == "__main__":
    print("=== Exercise 1: Pure vs Impure ===")
    exercise_1()
    print("\n=== Exercise 2: Implement map/filter/reduce ===")
    exercise_2()
    print("\n=== Exercise 3: Refactor to Functional Style ===")
    exercise_3()
    print("\n=== Exercise 4: Implement Curry ===")
    exercise_4()
    print("\n=== Exercise 5: Build a Pipeline ===")
    exercise_5()
    print("\n=== Exercise 6: Implement Maybe Monad ===")
    exercise_6()
    print("\nAll exercises completed!")

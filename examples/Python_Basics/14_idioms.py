"""
14 Pythonic Idioms
==================
Demonstrates Pythonic patterns vs anti-patterns, comprehension tricks,
context managers, and idiomatic Python style.
"""

from contextlib import contextmanager
import time


def iteration_idioms():
    """Idiomatic iteration patterns."""
    colors = ["red", "green", "blue", "yellow"]

    # Anti-pattern: index-based loop
    print("Anti-pattern (C-style):")
    for i in range(len(colors)):
        print(f"  {colors[i]}")

    # Pythonic: direct iteration
    print("\nPythonic:")
    for color in colors:
        print(f"  {color}")

    # Anti-pattern: manual index tracking
    print("\nAnti-pattern (manual index):")
    i = 0
    for color in colors:
        print(f"  {i}: {color}")
        i += 1

    # Pythonic: enumerate
    print("\nPythonic (enumerate):")
    for i, color in enumerate(colors):
        print(f"  {i}: {color}")

    # Anti-pattern: parallel lists with index
    names = ["Alice", "Bob", "Charlie"]
    scores = [95, 82, 91]
    print("\nAnti-pattern (parallel index):")
    for i in range(len(names)):
        print(f"  {names[i]}: {scores[i]}")

    # Pythonic: zip
    print("\nPythonic (zip):")
    for name, score in zip(names, scores):
        print(f"  {name}: {score}")


def dict_idioms():
    """Idiomatic dictionary patterns."""
    # Anti-pattern: check then access
    d = {"name": "Alice", "age": 30}

    print("Anti-pattern (check then get):")
    if "email" in d:
        email = d["email"]
    else:
        email = "N/A"
    print(f"  email: {email}")

    # Pythonic: .get() with default
    print("\nPythonic (.get):")
    email = d.get("email", "N/A")
    print(f"  email: {email}")

    # Anti-pattern: build dict from lists
    keys = ["a", "b", "c"]
    values = [1, 2, 3]
    result = {}
    for i in range(len(keys)):
        result[keys[i]] = values[i]
    print(f"\nAnti-pattern: {result}")

    # Pythonic: dict(zip(...))
    result = dict(zip(keys, values))
    print(f"Pythonic:     {result}")

    # Counting with defaultdict vs manual
    words = "apple banana apple cherry banana apple".split()

    # Anti-pattern
    count = {}
    for w in words:
        if w in count:
            count[w] += 1
        else:
            count[w] = 1

    # Pythonic: Counter
    from collections import Counter
    count = Counter(words)
    print(f"\nCounter: {count}")

    # Dict merging (Python 3.9+)
    defaults = {"theme": "dark", "lang": "en", "font_size": 14}
    user = {"theme": "light", "font_size": 16}
    config = defaults | user  # user overrides defaults
    print(f"Merged: {config}")


def conditional_idioms():
    """Idiomatic conditional expressions."""
    # Anti-pattern: verbose boolean return
    def is_even_verbose(n):
        if n % 2 == 0:
            return True
        else:
            return False

    # Pythonic: direct boolean
    def is_even(n):
        return n % 2 == 0

    print(f"is_even(4): {is_even(4)}")

    # Anti-pattern: compare to True/False/None
    x = True
    print(f"\nAnti-pattern: x == True  -> {x == True}")   # noqa: E712
    print(f"Pythonic:     x          -> {x}")

    value = None
    print(f"Anti-pattern: value == None -> {value == None}")
    print(f"Pythonic:     value is None -> {value is None}")

    # Truthy/falsy checks
    items = []
    # Anti-pattern
    if len(items) == 0:
        print("\nAnti-pattern: len(items) == 0")
    # Pythonic
    if not items:
        print("Pythonic:     not items")

    # Ternary for assignment
    score = 85
    # Anti-pattern
    if score >= 60:
        result = "pass"
    else:
        result = "fail"
    # Pythonic
    result = "pass" if score >= 60 else "fail"
    print(f"\nTernary: {result}")


def string_idioms():
    """Idiomatic string operations."""
    words = ["Hello", "World", "from", "Python"]

    # Anti-pattern: concatenation in loop
    result = ""
    for w in words:
        result += w + " "
    print(f"Anti-pattern (+=): {result.strip()!r}")

    # Pythonic: join
    result = " ".join(words)
    print(f"Pythonic (join):   {result!r}")

    # Anti-pattern: manual string building
    name, age = "Alice", 30
    s = "Name: " + name + ", Age: " + str(age)
    print(f"\nAnti-pattern (concat): {s}")

    # Pythonic: f-string
    s = f"Name: {name}, Age: {age}"
    print(f"Pythonic (f-string):   {s}")

    # Check substring
    text = "Hello, World!"
    # Anti-pattern
    if text.find("World") != -1:
        print("\nAnti-pattern: .find() != -1")
    # Pythonic
    if "World" in text:
        print("Pythonic:     'World' in text")


def comprehension_tricks():
    """Advanced comprehension patterns."""
    # Flatten nested list
    nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flat = [x for sublist in nested for x in sublist]
    print(f"Flatten: {flat}")

    # Transform + filter in one pass
    words = ["Hello", "WORLD", "Python", "CODE", "test"]
    lower_long = [w.lower() for w in words if len(w) > 4]
    print(f"Transform+filter: {lower_long}")

    # Dict from pairs with filtering
    items = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    big = {k: v for k, v in items if v > 2}
    print(f"Dict filter: {big}")

    # Matrix operations
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Row sums
    row_sums = [sum(row) for row in matrix]
    print(f"\nRow sums:    {row_sums}")

    # Column sums
    col_sums = [sum(row[i] for row in matrix) for i in range(3)]
    print(f"Column sums: {col_sums}")

    # Diagonal
    diagonal = [matrix[i][i] for i in range(len(matrix))]
    print(f"Diagonal:    {diagonal}")

    # Conditional mapping
    nums = range(-5, 6)
    signs = ["+" if n > 0 else ("-" if n < 0 else "0") for n in nums]
    print(f"\nSigns: {signs}")

    # Using any() and all()
    data = [2, 4, 6, 8, 10]
    print(f"\nall even: {all(x % 2 == 0 for x in data)}")
    print(f"any > 5:  {any(x > 5 for x in data)}")


def unpacking_idioms():
    """Idiomatic unpacking patterns."""
    # Swap without temp
    a, b = 1, 2
    a, b = b, a
    print(f"Swap: a={a}, b={b}")

    # Star unpacking
    first, *rest = [1, 2, 3, 4, 5]
    print(f"first={first}, rest={rest}")

    *init, last = [1, 2, 3, 4, 5]
    print(f"init={init}, last={last}")

    # Ignore values with _
    _, name, _, score = ("ID001", "Alice", "Math", 95)
    print(f"\nIgnored: name={name}, score={score}")

    # Unpack in function calls
    def point_info(x, y, z):
        return f"({x}, {y}, {z})"

    coords = [3, 4, 5]
    print(f"Unpacked call: {point_info(*coords)}")


def context_manager_idioms():
    """Writing custom context managers."""

    # Class-based context manager
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  Elapsed: {self.elapsed:.6f}s")

    print("Class-based context manager:")
    with Timer():
        total = sum(range(1_000_000))

    # Generator-based context manager (simpler)
    @contextmanager
    def timer(label="Operation"):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        print(f"  {label}: {elapsed:.6f}s")

    print("\nGenerator-based context manager:")
    with timer("Sum"):
        total = sum(range(1_000_000))

    # Suppress exceptions
    @contextmanager
    def suppress(*exceptions):
        try:
            yield
        except exceptions:
            pass

    print("\nSuppress context manager:")
    with suppress(FileNotFoundError):
        open("/nonexistent/file.txt")
    print("  Continued after suppressed FileNotFoundError")


def general_tips():
    """Miscellaneous Pythonic tips."""
    # Use 'in' for membership testing
    valid = {"admin", "editor", "viewer"}
    role = "editor"
    assert role in valid

    # Chained comparisons
    x = 5
    assert 1 < x < 10

    # Multiple assignment
    x = y = z = 0
    print(f"Multiple assignment: x={x}, y={y}, z={z}")

    # Enumerate instead of range(len())
    items = ["a", "b", "c"]
    indexed = {i: v for i, v in enumerate(items)}
    print(f"Indexed dict: {indexed}")

    # Use collections.abc for type checking
    from collections.abc import Mapping, Sequence

    def process(data):
        if isinstance(data, Mapping):
            return f"dict-like with {len(data)} keys"
        elif isinstance(data, Sequence):
            return f"list-like with {len(data)} items"
        return "other"

    print(f"\nprocess(dict):  {process({'a': 1})}")
    print(f"process(list):  {process([1, 2, 3])}")
    print(f"process(tuple): {process((1, 2))}")

    # Avoid global mutable state; prefer function parameters
    # Avoid bare except; always catch specific exceptions
    # Use logging instead of print in production code
    print("\nKey principles:")
    print("  - Flat is better than nested")
    print("  - Explicit is better than implicit")
    print("  - Simple is better than complex")
    print("  - Readability counts")


if __name__ == "__main__":
    sections = [
        ("Iteration Idioms", iteration_idioms),
        ("Dict Idioms", dict_idioms),
        ("Conditional Idioms", conditional_idioms),
        ("String Idioms", string_idioms),
        ("Comprehension Tricks", comprehension_tricks),
        ("Unpacking Idioms", unpacking_idioms),
        ("Context Manager Idioms", context_manager_idioms),
        ("General Tips", general_tips),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

"""
Exercises for Lesson 15: Python Basics
Topic: Python

Solutions to practice problems from the lesson.
"""

import string
from statistics import median


# === Exercise 1: Data Types and Type Conversion ===
# Problem: Practice working with Python's core data types and conversion functions.

def exercise_1():
    """Solutions for data types and type conversion."""

    # 1. Create variables of each basic type and print their types
    print("--- 1. Basic types ---")
    i = 42
    f = 3.14
    s = "hello"
    b = True
    n = None
    for var in [i, f, s, b, n]:
        print(f"  {var!r:10s} -> {type(var).__name__}")

    # 2. safe_convert: attempt conversion, return None on failure
    print("\n--- 2. safe_convert ---")

    def safe_convert(value, target_type):
        """Attempt to convert value to target_type, return None on failure.

        Uses try/except because Python's duck typing means we can't predict
        all possible conversion failures -- better to ask forgiveness
        than permission (EAFP principle).
        """
        try:
            return target_type(value)
        except (ValueError, TypeError):
            return None

    print(f"  safe_convert('42', int) = {safe_convert('42', int)}")          # 42
    print(f"  safe_convert('hello', float) = {safe_convert('hello', float)}")  # None
    print(f"  safe_convert('3.14', float) = {safe_convert('3.14', float)}")    # 3.14

    # 3. Difference between == and is
    print("\n--- 3. == vs is ---")
    a = [1, 2, 3]
    b = [1, 2, 3]
    # == compares values, is compares identity (same object in memory)
    print(f"  a == b: {a == b}")   # True (same content)
    print(f"  a is b: {a is b}")   # False (different objects)
    c = a
    print(f"  a is c: {a is c}")   # True (c is an alias for a)

    # 4. Truthiness of various values
    print("\n--- 4. Truthiness ---")
    values = [0, 0.0, "", " ", [], [0], None, False]
    for v in values:
        print(f"  bool({v!r:8s}) = {bool(v)}")

    # 5. Count truthy values with one-liner
    print("\n--- 5. Count truthy values ---")
    items = [0, 1, "", "hello", None, [], [1]]
    truthy_count = sum(bool(x) for x in items)
    print(f"  Truthy values in {items}: {truthy_count}")  # 3 (1, "hello", [1])


# === Exercise 2: String Manipulation ===
# Problem: Build a small text-processing toolkit.

def exercise_2():
    """Solutions for string manipulation."""

    # 1. normalize: strip, lowercase, underscores
    print("--- 1. normalize ---")

    def normalize(s: str) -> str:
        """Strip whitespace, lowercase, and replace spaces with underscores."""
        return s.strip().lower().replace(" ", "_")

    print(f'  normalize("  Hello World  ") = "{normalize("  Hello World  ")}"')

    # 2. mask_email: hide middle of local part
    print("\n--- 2. mask_email ---")

    def mask_email(email: str) -> str:
        """Mask the middle characters of the local part of an email.

        Shows only the first and last character of the local part,
        replacing everything in between with asterisks.
        """
        local, domain = email.split("@")
        if len(local) <= 2:
            masked = local[0] + "*" * (len(local) - 1)
        else:
            masked = local[0] + "*" * (len(local) - 2) + local[-1]
        return f"{masked}@{domain}"

    print(f'  mask_email("alice@example.com") = "{mask_email("alice@example.com")}"')
    print(f'  mask_email("ab@example.com") = "{mask_email("ab@example.com")}"')

    # 3. word_count: frequency dictionary
    print("\n--- 3. word_count ---")

    def word_count(text: str) -> dict[str, int]:
        """Count word frequencies (case-insensitive)."""
        counts: dict[str, int] = {}
        for word in text.lower().split():
            counts[word] = counts.get(word, 0) + 1
        return counts

    result = word_count("the quick brown fox the fox")
    print(f"  word_count result: {result}")

    # 4. Formatted table with f-strings
    print("\n--- 4. Formatted table ---")
    data = [("Alice", 92.5), ("Bob", 78.0), ("Charlie", 85.3)]
    print(f"  | {'Name':<10} | {'Score':>6} |")
    print(f"  |{'-' * 12}|{'-' * 8}|")
    for name, score in data:
        print(f"  | {name:<10} | {score:>6.1f} |")

    # 5. Palindrome check
    print("\n--- 5. is_palindrome ---")

    def is_palindrome(s: str) -> bool:
        """Check if a string is a palindrome, ignoring case, spaces, punctuation.

        Filters to only alphanumeric characters before comparing,
        so "A man, a plan, a canal: Panama" passes.
        """
        cleaned = "".join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    test_strings = [
        "A man, a plan, a canal: Panama",
        "racecar",
        "hello",
        "Was it a car or a cat I saw?",
    ]
    for s in test_strings:
        print(f'  is_palindrome("{s}") = {is_palindrome(s)}')


# === Exercise 3: Control Flow and Comprehensions ===
# Problem: Practice conditionals, loops, and comprehensions.

def exercise_3():
    """Solutions for control flow and comprehensions."""

    # 1. FizzBuzz
    print("--- 1. fizzbuzz ---")

    def fizzbuzz(n: int) -> list[str]:
        """Classic FizzBuzz: multiples of 3 and 5 get special labels.

        Check the combined case (15) first because 15 is divisible by
        both 3 and 5 -- if we checked 3 or 5 first, we would miss FizzBuzz.
        """
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result

    print(f"  fizzbuzz(15) = {fizzbuzz(15)}")

    # 2. List comprehension replacement
    print("\n--- 2. List comprehension ---")
    # Original loop:
    # result = []
    # for x in range(20):
    #     if x % 2 == 0 and x % 3 != 0:
    #         result.append(x ** 2)
    result = [x ** 2 for x in range(20) if x % 2 == 0 and x % 3 != 0]
    print(f"  Squared (even, not divisible by 3): {result}")

    # 3. Dictionary comprehension: word -> length
    print("\n--- 3. Dict comprehension ---")
    sentence = "the quick brown fox"
    word_lengths = {word: len(word) for word in sentence.split()}
    print(f"  Word lengths: {word_lengths}")

    # 4. Multiplication table
    print("\n--- 4. Multiplication table ---")
    table = [[i * j for j in range(1, 6)] for i in range(1, 6)]
    for row in table:
        print("  " + " ".join(f"{x:4d}" for x in row))

    # 5. merge_ranked with zip and enumerate
    print("\n--- 5. merge_ranked ---")

    def merge_ranked(names: list[str], scores: list[int]) -> list[tuple]:
        """Combine names and scores, sort by score descending, add rank.

        zip pairs names with scores, sorted reorders by score,
        and enumerate adds the 1-based rank.
        """
        paired = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [(rank, name, score) for rank, (name, score) in enumerate(paired, 1)]

    names = ["Alice", "Bob", "Charlie"]
    scores = [85, 92, 78]
    ranked = merge_ranked(names, scores)
    print(f"  Ranked: {ranked}")


# === Exercise 4: Functions and Argument Handling ===
# Problem: Explore Python's flexible function argument mechanisms.

def exercise_4():
    """Solutions for functions and argument handling."""

    # 1. stats(*numbers) -> (min, max, mean, median)
    print("--- 1. stats ---")

    def stats(*numbers: float) -> tuple:
        """Return (min, max, mean, median) for the given numbers.

        Handles edge case of empty input by returning a tuple of Nones.
        Uses the statistics.median function for correct median calculation
        (handles both odd and even-length sequences).
        """
        if not numbers:
            return (None, None, None, None)
        return (
            min(numbers),
            max(numbers),
            sum(numbers) / len(numbers),
            median(numbers),
        )

    print(f"  stats(3, 1, 4, 1, 5, 9) = {stats(3, 1, 4, 1, 5, 9)}")
    print(f"  stats() = {stats()}")

    # 2. create_html_tag
    print("\n--- 2. create_html_tag ---")

    def create_html_tag(tag: str, content: str, **attributes: str) -> str:
        """Generate an HTML tag string with attributes.

        Trailing underscores in attribute names are stripped so Python
        reserved words like 'class' can be passed as 'class_'.
        """
        attrs = ""
        for key, value in attributes.items():
            # Strip trailing underscore (class_ -> class)
            clean_key = key.rstrip("_")
            attrs += f' {clean_key}="{value}"'
        return f"<{tag}{attrs}>{content}</{tag}>"

    html = create_html_tag("a", "Click here", href="https://example.com", class_="btn")
    print(f"  {html}")

    html2 = create_html_tag("div", "Hello", id="main", style="color: red")
    print(f"  {html2}")

    # 3. Multi-key sort with lambda
    print("\n--- 3. Multi-key sort ---")
    students = [
        {"name": "Alice", "grade": "B", "score": 85},
        {"name": "Bob", "grade": "A", "score": 92},
        {"name": "Charlie", "grade": "C", "score": 78},
        {"name": "Diana", "grade": "A", "score": 88},
    ]
    # Sort by grade ascending, then score descending
    # Negate score for descending within the same grade
    sorted_students = sorted(students, key=lambda s: (s["grade"], -s["score"]))
    for s in sorted_students:
        print(f"  {s['name']:8s} grade={s['grade']} score={s['score']}")

    # 4. Custom memoize decorator
    print("\n--- 4. Memoize decorator ---")

    def memoize(func):
        """Cache function results -- no functools.lru_cache allowed."""
        cache = {}
        call_count = [0]

        def wrapper(*args):
            call_count[0] += 1
            if args not in cache:
                cache[args] = func(*args)
            return cache[args]

        wrapper.call_count = call_count
        return wrapper

    # Without memoization
    naive_count = [0]

    def fib_naive(n):
        naive_count[0] += 1
        if n <= 1:
            return n
        return fib_naive(n - 1) + fib_naive(n - 2)

    naive_count[0] = 0
    result = fib_naive(30)
    print(f"  fib_naive(30) = {result}, calls = {naive_count[0]}")

    # With memoization
    @memoize
    def fib_memo(n):
        if n <= 1:
            return n
        return fib_memo(n - 1) + fib_memo(n - 2)

    fib_memo.call_count[0] = 0
    result = fib_memo(30)
    print(f"  fib_memo(30) = {result}, calls = {fib_memo.call_count[0]}")

    # 5. Mutable default argument bug
    print("\n--- 5. Mutable default argument bug ---")

    # THE BUG: all calls share the same default list object
    def append_to_buggy(item, lst=[]):
        lst.append(item)
        return lst

    print(f"  Buggy: {append_to_buggy(1)}")  # [1]
    print(f"  Buggy: {append_to_buggy(2)}")  # [1, 2] -- unexpected!
    print(f"  Buggy: {append_to_buggy(3)}")  # [1, 2, 3] -- list persists!

    # THE FIX: use None as default and create a new list in the body
    def append_to_fixed(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    print(f"  Fixed: {append_to_fixed(1)}")  # [1]
    print(f"  Fixed: {append_to_fixed(2)}")  # [2] -- fresh list each time
    print(f"  Fixed: {append_to_fixed(3)}")  # [3]


# === Exercise 5: Data Structures and Exception Handling ===
# Problem: Combine data structure operations with robust error handling.

def exercise_5():
    """Solutions for data structures and exception handling."""

    # 1. group_by
    print("--- 1. group_by ---")

    def group_by(items: list, key_func) -> dict:
        """Group items by the return value of key_func.

        Uses dict.setdefault to initialize new groups with an empty list,
        avoiding the need for collections.defaultdict.
        """
        groups: dict = {}
        for item in items:
            key = key_func(item)
            groups.setdefault(key, []).append(item)
        return groups

    result = group_by([1, 2, 3, 4, 5, 6], lambda x: "even" if x % 2 == 0 else "odd")
    print(f"  group_by (even/odd): {result}")

    words = ["apple", "banana", "avocado", "blueberry", "cherry"]
    by_letter = group_by(words, lambda w: w[0])
    print(f"  group_by (first letter): {by_letter}")

    # 2. Stack with custom exception
    print("\n--- 2. Stack ---")

    class StackUnderflowError(Exception):
        """Raised when pop() or peek() is called on an empty stack."""
        pass

    class Stack:
        """LIFO stack using a Python list as the underlying storage."""

        def __init__(self):
            self._items: list = []

        def push(self, item):
            self._items.append(item)

        def pop(self):
            if self.is_empty():
                raise StackUnderflowError("Cannot pop from an empty stack")
            return self._items.pop()

        def peek(self):
            if self.is_empty():
                raise StackUnderflowError("Cannot peek at an empty stack")
            return self._items[-1]

        def is_empty(self) -> bool:
            return len(self._items) == 0

        def __len__(self) -> int:
            return len(self._items)

        def __repr__(self) -> str:
            return f"Stack({self._items})"

    stack = Stack()
    stack.push(10)
    stack.push(20)
    stack.push(30)
    print(f"  Stack: {stack}")
    print(f"  peek: {stack.peek()}")
    print(f"  pop: {stack.pop()}")
    print(f"  Stack after pop: {stack}")

    try:
        empty_stack = Stack()
        empty_stack.pop()
    except StackUnderflowError as e:
        print(f"  StackUnderflowError: {e}")

    # 3. deep_merge
    print("\n--- 3. deep_merge ---")

    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base.

        For nested dicts, we recurse instead of replacing wholesale.
        For non-dict values, override takes precedence.
        Creates a new dict -- does not modify the originals.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    base = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    override = {"b": {"y": 99, "z": 30}, "d": 4}
    merged = deep_merge(base, override)
    print(f"  base:     {base}")
    print(f"  override: {override}")
    print(f"  merged:   {merged}")
    # {"a": 1, "b": {"x": 10, "y": 99, "z": 30}, "c": 3, "d": 4}

    # 4. read_csv_safe
    print("\n--- 4. read_csv_safe ---")

    import csv
    import os
    import tempfile

    def read_csv_safe(filename: str) -> list[dict]:
        """Read a CSV file safely with comprehensive error handling.

        - FileNotFoundError for missing files
        - Skips malformed rows (wrong column count)
        - Returns empty list for empty files
        - Uses finally to ensure file is always closed
        """
        f = None
        try:
            f = open(filename, "r", newline="")
            reader = csv.reader(f)

            headers = None
            rows = []

            for i, row in enumerate(reader):
                if i == 0:
                    headers = row
                    continue

                if headers and len(row) != len(headers):
                    print(f"    [WARNING] Skipping malformed row {i + 1}: {row}")
                    continue

                if headers:
                    rows.append(dict(zip(headers, row)))

            return rows

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        finally:
            if f is not None:
                f.close()

    # Create a test CSV with some malformed rows
    tmpfile = os.path.join(tempfile.gettempdir(), "test_safe.csv")
    with open(tmpfile, "w", newline="") as f:
        f.write("name,age,city\n")
        f.write("Alice,30,Seoul\n")
        f.write("Bob,25\n")  # Malformed: missing city
        f.write("Charlie,35,Tokyo\n")

    result = read_csv_safe(tmpfile)
    print(f"  Valid rows: {result}")
    os.remove(tmpfile)

    # Test with non-existent file
    try:
        read_csv_safe("/nonexistent/file.csv")
    except FileNotFoundError as e:
        print(f"  FileNotFoundError: {e}")

    # 5. Set operations
    print("\n--- 5. Set operations ---")
    active_users = {1, 2, 3, 4, 5, 6}
    premium_users = {4, 5, 6, 7, 8}

    both = active_users & premium_users
    active_only = active_users - premium_users
    exclusive = active_users ^ premium_users
    total = len(active_users | premium_users)

    print(f"  Active: {active_users}")
    print(f"  Premium: {premium_users}")
    print(f"  Both active AND premium (intersection): {both}")
    print(f"  Active but NOT premium (difference): {active_only}")
    print(f"  Exclusive (symmetric difference): {exclusive}")
    print(f"  Total distinct users (union size): {total}")


if __name__ == "__main__":
    print("=== Exercise 1: Data Types and Type Conversion ===")
    exercise_1()

    print("\n=== Exercise 2: String Manipulation ===")
    exercise_2()

    print("\n=== Exercise 3: Control Flow and Comprehensions ===")
    exercise_3()

    print("\n=== Exercise 4: Functions and Argument Handling ===")
    exercise_4()

    print("\n=== Exercise 5: Data Structures and Exception Handling ===")
    exercise_5()

    print("\nAll exercises completed!")

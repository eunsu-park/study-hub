"""
04 Control Flow
===============
Demonstrates if/elif/else, for/while loops, enumerate, zip,
break/continue, match-case, and comprehension patterns.
"""


def if_elif_else():
    """Conditional branching."""
    score = 85

    # Basic if/elif/else
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    print(f"Score {score} -> Grade {grade}")

    # Ternary (conditional expression)
    status = "pass" if score >= 60 else "fail"
    print(f"Status: {status}")

    # Nested ternary (use sparingly)
    label = "high" if score >= 80 else ("mid" if score >= 60 else "low")
    print(f"Label: {label}")


def for_loops():
    """Various for-loop patterns."""
    # Basic iteration
    fruits = ["apple", "banana", "cherry"]
    print("Fruits:")
    for fruit in fruits:
        print(f"  - {fruit}")

    # range() variations
    print("\nrange(5):", list(range(5)))
    print("range(2, 8):", list(range(2, 8)))
    print("range(0, 20, 3):", list(range(0, 20, 3)))
    print("range(10, 0, -2):", list(range(10, 0, -2)))

    # Iterating over a string
    print("\nCharacters in 'Python':", end=" ")
    for ch in "Python":
        print(ch, end=" ")
    print()

    # Iterating over a dictionary
    scores = {"Alice": 95, "Bob": 82, "Charlie": 91}
    print("\nScores:")
    for name, score in scores.items():
        print(f"  {name}: {score}")


def enumerate_and_zip():
    """enumerate() for indexed iteration, zip() for parallel iteration."""
    # enumerate
    languages = ["Python", "JavaScript", "Rust", "Go"]
    print("Languages (enumerate):")
    for i, lang in enumerate(languages):
        print(f"  [{i}] {lang}")

    # enumerate with custom start
    print("\nWith start=1:")
    for i, lang in enumerate(languages, start=1):
        print(f"  {i}. {lang}")

    # zip: parallel iteration
    names = ["Alice", "Bob", "Charlie"]
    ages = [30, 25, 35]
    cities = ["NYC", "LA", "Chicago"]

    print("\nPeople (zip):")
    for name, age, city in zip(names, ages, cities):
        print(f"  {name}, age {age}, from {city}")

    # zip stops at shortest — use itertools.zip_longest for padding
    short = [1, 2]
    long = [10, 20, 30, 40]
    print(f"\nzip({short}, {long}): {list(zip(short, long))}")

    # Unzip with zip(*)
    pairs = [(1, "a"), (2, "b"), (3, "c")]
    numbers, letters = zip(*pairs)
    print(f"Unzipped: numbers={numbers}, letters={letters}")


def while_loops():
    """while loop patterns."""
    # Basic while
    n = 5
    print(f"Countdown from {n}:")
    while n > 0:
        print(f"  {n}", end="")
        n -= 1
    print("  Launch!")

    # while with break
    print("\nFind first multiple of 7 > 50:")
    n = 51
    while True:
        if n % 7 == 0:
            print(f"  Found: {n}")
            break
        n += 1

    # while/else — else runs when loop ends normally (no break)
    print("\nSearch in list (while/else):")
    data = [2, 4, 6, 8, 10]
    target = 7
    i = 0
    while i < len(data):
        if data[i] == target:
            print(f"  Found {target} at index {i}")
            break
        i += 1
    else:
        print(f"  {target} not found in {data}")


def break_continue_pass():
    """Control flow within loops."""
    # continue: skip even numbers
    print("Odd numbers 1-10:")
    for n in range(1, 11):
        if n % 2 == 0:
            continue
        print(f"  {n}", end="")
    print()

    # break: stop at first negative
    data = [3, 7, 2, -1, 5, 8]
    print(f"\nProcess until negative ({data}):")
    for val in data:
        if val < 0:
            print(f"  Stopped at {val}")
            break
        print(f"  Processing {val}")

    # for/else: else runs only if no break
    print("\nPrime check with for/else:")
    for n in [17, 18, 19, 20]:
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                print(f"  {n} is composite ({d} x {n // d})")
                break
        else:
            print(f"  {n} is prime")

    # pass: placeholder
    class NotImplementedYet:
        pass


def match_case():
    """Structural pattern matching (Python 3.10+)."""
    def classify_http_status(status):
        match status:
            case 200:
                return "OK"
            case 301 | 302:
                return "Redirect"
            case 404:
                return "Not Found"
            case 500:
                return "Server Error"
            case code if 200 <= code < 300:
                return f"Success ({code})"
            case _:
                return f"Unknown ({status})"

    for code in [200, 201, 301, 404, 500, 418]:
        print(f"  HTTP {code}: {classify_http_status(code)}")

    # Pattern matching with structures
    def describe_point(point):
        match point:
            case (0, 0):
                return "origin"
            case (x, 0):
                return f"x-axis at {x}"
            case (0, y):
                return f"y-axis at {y}"
            case (x, y):
                return f"({x}, {y})"

    print("\nPoint descriptions:")
    for p in [(0, 0), (5, 0), (0, 3), (2, 7)]:
        print(f"  {p} -> {describe_point(p)}")


def comprehension_patterns():
    """List, dict, set, and generator comprehensions."""
    # List comprehension
    squares = [x ** 2 for x in range(10)]
    print(f"Squares: {squares}")

    # With condition
    evens = [x for x in range(20) if x % 2 == 0]
    print(f"Evens:   {evens}")

    # Dict comprehension
    word_lengths = {w: len(w) for w in ["hello", "world", "python"]}
    print(f"Lengths: {word_lengths}")

    # Set comprehension
    unique_remainders = {x % 5 for x in range(20)}
    print(f"Remainders mod 5: {sorted(unique_remainders)}")

    # Nested comprehension: flatten
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat = [val for row in matrix for val in row]
    print(f"Flattened: {flat}")

    # Generator expression (lazy — uses parentheses)
    total = sum(x ** 2 for x in range(1000))
    print(f"Sum of squares 0-999: {total}")

    # Conditional expression in comprehension
    labels = ["even" if x % 2 == 0 else "odd" for x in range(6)]
    print(f"Labels: {labels}")


if __name__ == "__main__":
    sections = [
        ("if/elif/else", if_elif_else),
        ("for Loops", for_loops),
        ("enumerate & zip", enumerate_and_zip),
        ("while Loops", while_loops),
        ("break/continue/pass", break_continue_pass),
        ("match-case (3.10+)", match_case),
        ("Comprehension Patterns", comprehension_patterns),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

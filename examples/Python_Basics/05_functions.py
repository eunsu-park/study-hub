"""
05 Functions
============
Demonstrates function definitions, default arguments, *args/**kwargs,
lambda expressions, map/filter/reduce, closures, and recursion.
"""

from functools import reduce


def basic_functions():
    """Function definition, return values, and docstrings."""

    def greet(name):
        """Return a greeting string."""
        return f"Hello, {name}!"

    print(greet("Alice"))
    print(f"Docstring: {greet.__doc__!r}")

    # Multiple return values (returns a tuple)
    def min_max(numbers):
        return min(numbers), max(numbers)

    lo, hi = min_max([3, 1, 4, 1, 5, 9])
    print(f"min={lo}, max={hi}")

    # No return statement returns None
    def do_nothing():
        pass

    result = do_nothing()
    print(f"do_nothing() returned: {result!r}")


def default_and_keyword_args():
    """Default values, positional, and keyword arguments."""

    def connect(host, port=5432, timeout=30, ssl=True):
        return f"{host}:{port} (timeout={timeout}, ssl={ssl})"

    # Positional
    print(connect("localhost"))
    # Keyword
    print(connect("db.example.com", port=3306, ssl=False))
    # Mixed
    print(connect("10.0.0.1", 8080, timeout=5))

    # WARNING: mutable default argument pitfall
    def bad_append(item, lst=[]):  # noqa: B006 — intentional demo
        lst.append(item)
        return lst

    print(f"\nbad_append(1): {bad_append(1)}")
    print(f"bad_append(2): {bad_append(2)}")  # Shared list!

    # Correct pattern
    def good_append(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    print(f"good_append(1): {good_append(1)}")
    print(f"good_append(2): {good_append(2)}")  # Fresh list


def args_and_kwargs():
    """Variable-length arguments with *args and **kwargs."""

    def sum_all(*args):
        """Accept any number of positional arguments."""
        print(f"  args = {args}")
        return sum(args)

    print(f"sum_all(1, 2, 3): {sum_all(1, 2, 3)}")

    def build_profile(**kwargs):
        """Accept any number of keyword arguments."""
        print(f"  kwargs = {kwargs}")
        return kwargs

    profile = build_profile(name="Alice", age=30, role="engineer")
    print(f"  Profile: {profile}")

    # Combining *args and **kwargs
    def log(level, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  [{level}] {msg}" + (f" ({extra})" if extra else ""))

    log("INFO", "Server started", port=8080)
    log("ERROR", "Connection", "failed", host="db", retry=3)

    # Unpacking when calling
    data = [1, 2, 3, 4, 5]
    print(f"\nsum_all(*{data}): {sum_all(*data)}")

    config = {"host": "localhost", "port": 8080, "timeout": 30}
    def show_config(host, port, timeout):
        return f"{host}:{port} (timeout={timeout})"
    print(f"show_config(**config): {show_config(**config)}")


def keyword_only_and_positional_only():
    """Keyword-only (after *) and positional-only (before /) parameters."""

    # Keyword-only: parameters after * must be passed by name
    def fetch(url, *, timeout=30, retries=3):
        return f"GET {url} (timeout={timeout}, retries={retries})"

    print(fetch("https://example.com", timeout=10))
    # fetch("url", 10) would raise TypeError

    # Positional-only (Python 3.8+): parameters before / must be passed by position
    def power(base, exp, /):
        return base ** exp

    print(f"power(2, 10) = {power(2, 10)}")
    # power(base=2, exp=10) would raise TypeError

    # Combined
    def hybrid(pos_only, /, normal, *, kw_only):
        return f"pos={pos_only}, normal={normal}, kw={kw_only}"

    print(hybrid(1, 2, kw_only=3))
    print(hybrid(1, normal=2, kw_only=3))


def lambda_and_higher_order():
    """Lambda expressions, map, filter, reduce."""
    # Lambda: anonymous single-expression function
    square = lambda x: x ** 2
    print(f"square(5) = {square(5)}")

    # Sorting with key function
    words = ["banana", "apple", "cherry", "date"]
    by_length = sorted(words, key=lambda w: len(w))
    print(f"Sorted by length: {by_length}")

    # map: apply function to each element
    numbers = [1, 2, 3, 4, 5]
    doubled = list(map(lambda x: x * 2, numbers))
    print(f"\nmap (double): {doubled}")

    # filter: keep elements where function returns True
    evens = list(filter(lambda x: x % 2 == 0, range(10)))
    print(f"filter (even): {evens}")

    # reduce: fold elements into single value
    product = reduce(lambda a, b: a * b, [1, 2, 3, 4, 5])
    print(f"reduce (product): {product}")

    # Prefer comprehensions over map/filter in most cases
    doubled_comp = [x * 2 for x in numbers]
    evens_comp = [x for x in range(10) if x % 2 == 0]
    print(f"\nComprehension equivalents: {doubled_comp}, {evens_comp}")


def closures_and_decorators():
    """Functions returning functions, closures, simple decorator."""

    # Closure: inner function captures outer variable
    def make_multiplier(factor):
        def multiply(x):
            return x * factor
        return multiply

    double = make_multiplier(2)
    triple = make_multiplier(3)
    print(f"double(5) = {double(5)}")
    print(f"triple(5) = {triple(5)}")

    # Simple decorator
    def timer(func):
        import time
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"  {func.__name__} took {elapsed:.6f}s")
            return result
        return wrapper

    @timer
    def slow_sum(n):
        return sum(range(n))

    result = slow_sum(1_000_000)
    print(f"  Result: {result}")


def recursion_examples():
    """Classic recursion: factorial, fibonacci, and practical example."""

    def factorial(n):
        """n! = n * (n-1) * ... * 1"""
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    for n in range(8):
        print(f"  {n}! = {factorial(n)}")

    # Fibonacci with memoization
    def fibonacci(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
        return memo[n]

    print("\nFibonacci sequence:")
    fibs = [fibonacci(i) for i in range(15)]
    print(f"  {fibs}")

    # Recursive directory-like structure flattening
    def flatten(nested):
        """Flatten arbitrarily nested lists."""
        result = []
        for item in nested:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result

    data = [1, [2, 3], [4, [5, 6, [7]]], 8]
    print(f"\nFlatten {data}:")
    print(f"  {flatten(data)}")


if __name__ == "__main__":
    sections = [
        ("Basic Functions", basic_functions),
        ("Default & Keyword Args", default_and_keyword_args),
        ("*args and **kwargs", args_and_kwargs),
        ("Keyword-only & Positional-only", keyword_only_and_positional_only),
        ("Lambda & Higher-Order", lambda_and_higher_order),
        ("Closures & Decorators", closures_and_decorators),
        ("Recursion", recursion_examples),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

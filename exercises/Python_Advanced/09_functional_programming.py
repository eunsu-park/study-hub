"""
Exercises for Lesson 09: Functional Programming
Topic: Python

Solutions to practice problems from the lesson.
"""

from typing import Callable, TypeVar, Any
from functools import reduce

T = TypeVar("T")
R = TypeVar("R")


# === Exercise 1: Function Composition ===
# Problem: Write a compose2 function that takes two functions and returns
# their composition.

def compose2(f: Callable, g: Callable) -> Callable:
    """Return the composition f(g(x)).

    compose2(f, g)(x) == f(g(x)). This is the mathematical definition
    of function composition. We apply g first, then f to the result.
    """
    def composed(*args, **kwargs):
        return f(g(*args, **kwargs))
    return composed


def compose(*funcs: Callable) -> Callable:
    """Generalized composition: compose(f, g, h)(x) == f(g(h(x))).

    Uses reduce to chain an arbitrary number of functions right-to-left.
    The rightmost function is applied first, matching mathematical notation.
    """
    return reduce(compose2, funcs)


def exercise_1():
    """Demonstrate function composition."""
    double = lambda x: x * 2
    add_one = lambda x: x + 1
    square = lambda x: x ** 2

    # compose2: two functions
    double_then_add = compose2(add_one, double)
    print(f"compose2(add_one, double)(5) = {double_then_add(5)}")  # (5*2)+1 = 11

    add_then_double = compose2(double, add_one)
    print(f"compose2(double, add_one)(5) = {add_then_double(5)}")  # (5+1)*2 = 12

    # compose: multiple functions -- applied right to left
    pipeline = compose(double, add_one, square)
    print(f"compose(double, add_one, square)(3) = {pipeline(3)}")  # (3^2 + 1) * 2 = 20

    # Practical example: text processing pipeline
    normalize = compose(str.strip, str.lower, lambda s: s.replace("  ", " "))
    print(f'normalize("  Hello  World  ") = "{normalize("  Hello  World  ")}"')


# === Exercise 2: Transducer ===
# Problem: Write a function that combines map and filter in a single iteration.

def map_filter(
    items: list[T],
    transform: Callable[[T], R],
    predicate: Callable[[T], bool],
) -> list[R]:
    """Apply filter then map in a single pass over the data.

    Traditional approach would be: list(map(transform, filter(predicate, items)))
    which creates an intermediate filtered list. This version avoids that
    by doing both operations in one loop.
    """
    result = []
    for item in items:
        if predicate(item):
            result.append(transform(item))
    return result


def transduce(
    xform: Callable,
    reducer: Callable,
    initial: Any,
    items: list,
) -> Any:
    """A more general transducer: compose transformations and reduce in one pass.

    xform: a function that takes a reducer and returns a new reducer
    reducer: the base reducing function (e.g., append, add)
    initial: the initial accumulator value
    items: the input sequence
    """
    composed_reducer = xform(reducer)
    result = initial
    for item in items:
        result = composed_reducer(result, item)
    return result


def mapping(transform):
    """Transducer that applies a transform to each item."""
    def xform(reducer):
        def new_reducer(acc, item):
            return reducer(acc, transform(item))
        return new_reducer
    return xform


def filtering(predicate):
    """Transducer that filters items based on a predicate."""
    def xform(reducer):
        def new_reducer(acc, item):
            if predicate(item):
                return reducer(acc, item)
            return acc
        return new_reducer
    return xform


def exercise_2():
    """Demonstrate transducer pattern."""
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Simple map_filter: square even numbers
    result = map_filter(numbers, lambda x: x ** 2, lambda x: x % 2 == 0)
    print(f"Squared evens: {result}")  # [4, 16, 36, 64, 100]

    # Transducer approach: compose transformations
    def append(lst, item):
        lst.append(item)
        return lst

    # Filter evens, then double -- composed as transducers
    xform = compose(filtering(lambda x: x % 2 == 0), mapping(lambda x: x * 2))
    result2 = transduce(xform, append, [], numbers)
    print(f"Transduced (filter even, double): {result2}")  # [4, 8, 12, 16, 20]


# === Exercise 3: Memoization ===
# Problem: Implement your own memoization decorator.

def memoize(func: Callable) -> Callable:
    """Decorator that caches function results based on arguments.

    Uses a dictionary keyed by (args, frozenset(kwargs.items())) so it
    works with both positional and keyword arguments. For recursive
    functions like Fibonacci, this reduces exponential time to linear.
    """
    cache: dict = {}

    def wrapper(*args, **kwargs):
        # Create a hashable key from args and kwargs
        key = (args, frozenset(kwargs.items())) if kwargs else args
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # Expose cache for inspection
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def exercise_3():
    """Demonstrate custom memoization."""
    call_count = 0

    # Without memoization: exponential calls
    def fib_naive(n):
        nonlocal call_count
        call_count += 1
        if n <= 1:
            return n
        return fib_naive(n - 1) + fib_naive(n - 2)

    call_count = 0
    result = fib_naive(25)
    print(f"fib_naive(25) = {result}, calls = {call_count}")

    # With memoization: linear calls
    call_count_memo = 0

    @memoize
    def fib_memo(n):
        nonlocal call_count_memo
        call_count_memo += 1
        if n <= 1:
            return n
        return fib_memo(n - 1) + fib_memo(n - 2)

    call_count_memo = 0
    result = fib_memo(25)
    print(f"fib_memo(25) = {result}, calls = {call_count_memo}")

    # Show dramatic difference for larger n
    fib_memo.cache_clear()
    call_count_memo = 0
    result = fib_memo(100)
    print(f"fib_memo(100) = {result}, calls = {call_count_memo}")

    # Demonstrate with keyword arguments
    @memoize
    def power(base, exp=2):
        return base ** exp

    print(f"\npower(3) = {power(3)}")
    print(f"power(3, exp=3) = {power(3, exp=3)}")
    print(f"Cache contents: {power.cache}")


if __name__ == "__main__":
    print("=== Exercise 1: Function Composition ===")
    exercise_1()

    print("\n=== Exercise 2: Transducer ===")
    exercise_2()

    print("\n=== Exercise 3: Memoization ===")
    exercise_3()

    print("\nAll exercises completed!")

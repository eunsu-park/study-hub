"""
13 Standard Library
===================
Demonstrates key modules: collections, itertools, functools,
datetime, and argparse.
"""

from collections import Counter, defaultdict, deque, OrderedDict, ChainMap
from itertools import (
    chain, islice, product, permutations, combinations,
    groupby, accumulate, repeat, starmap, zip_longest,
)
from functools import lru_cache, partial, reduce, wraps
from datetime import datetime, date, timedelta, timezone


def collections_demo():
    """collections module highlights."""
    # Counter: count occurrences
    words = "the cat sat on the mat the cat ate the rat".split()
    counter = Counter(words)
    print(f"Counter:      {counter}")
    print(f"Most common:  {counter.most_common(3)}")
    print(f"Total:        {sum(counter.values())}")

    # Counter arithmetic
    c1 = Counter(a=3, b=1)
    c2 = Counter(a=1, b=2)
    print(f"c1 + c2:      {c1 + c2}")
    print(f"c1 - c2:      {c1 - c2}")  # Only keeps positive

    # defaultdict: automatic default values
    grouped = defaultdict(list)
    students = [("Math", "Alice"), ("CS", "Bob"), ("Math", "Charlie"), ("CS", "Diana")]
    for dept, name in students:
        grouped[dept].append(name)
    print(f"\nGrouped: {dict(grouped)}")

    # deque: double-ended queue (O(1) append/pop on both ends)
    d = deque([1, 2, 3, 4, 5], maxlen=5)
    d.appendleft(0)   # Pushes out 5
    print(f"\nDeque (maxlen=5): {d}")
    d.rotate(2)        # Rotate right
    print(f"Rotated +2:       {d}")

    # ChainMap: merged dict view
    defaults = {"color": "blue", "size": "medium"}
    user_prefs = {"color": "red"}
    config = ChainMap(user_prefs, defaults)
    print(f"\nChainMap: color={config['color']}, size={config['size']}")


def itertools_demo():
    """itertools module highlights."""
    # chain: flatten iterables
    merged = list(chain([1, 2], [3, 4], [5, 6]))
    print(f"chain:         {merged}")

    # islice: slice any iterable (lazy)
    first_5_evens = list(islice((x for x in range(100) if x % 2 == 0), 5))
    print(f"islice:        {first_5_evens}")

    # product: cartesian product
    cards = list(product(["A", "K", "Q"], ["hearts", "spades"]))
    print(f"product:       {cards}")

    # permutations and combinations
    print(f"permutations:  {list(permutations('ABC', 2))}")
    print(f"combinations:  {list(combinations('ABCD', 2))}")

    # accumulate: running totals
    data = [1, 2, 3, 4, 5]
    running_sum = list(accumulate(data))
    print(f"\naccumulate:    {running_sum}")

    import operator
    running_prod = list(accumulate(data, operator.mul))
    print(f"accumulate(*): {running_prod}")

    # groupby: group consecutive items (must be sorted first)
    animals = [
        ("cat", "mammal"), ("dog", "mammal"), ("eagle", "bird"),
        ("parrot", "bird"), ("snake", "reptile"),
    ]
    print("\ngroupby:")
    for key, group in groupby(animals, key=lambda x: x[1]):
        items = [name for name, _ in group]
        print(f"  {key:>8}: {items}")

    # zip_longest: zip without truncation
    a = [1, 2, 3]
    b = ["a", "b"]
    print(f"\nzip_longest:   {list(zip_longest(a, b, fillvalue='?'))}")

    # starmap: unpack arguments from iterable
    pairs = [(2, 5), (3, 2), (10, 3)]
    powers = list(starmap(pow, pairs))
    print(f"starmap(pow):  {pairs} -> {powers}")


def functools_demo():
    """functools module highlights."""
    # lru_cache: memoization
    @lru_cache(maxsize=128)
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    fibs = [fibonacci(i) for i in range(20)]
    print(f"Fibonacci:     {fibs}")
    print(f"Cache info:    {fibonacci.cache_info()}")

    # partial: freeze some arguments
    def power(base, exponent):
        return base ** exponent

    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)
    print(f"\npartial square(5): {square(5)}")
    print(f"partial cube(3):   {cube(3)}")

    # reduce: fold sequence
    nums = [1, 2, 3, 4, 5]
    total = reduce(lambda a, b: a + b, nums)
    print(f"\nreduce(+, {nums}): {total}")

    # Find max using reduce
    data = [3, 7, 2, 9, 1, 8]
    maximum = reduce(lambda a, b: a if a > b else b, data)
    print(f"reduce(max, {data}): {maximum}")

    # wraps: preserve function metadata in decorators
    def my_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @my_decorator
    def example():
        """This is an example function."""
        pass

    print(f"\n@wraps preserves: {example.__name__}, doc={example.__doc__!r}")


def datetime_demo():
    """datetime module for date and time operations."""
    # Current date/time
    now = datetime.now()
    today = date.today()
    print(f"Now:   {now}")
    print(f"Today: {today}")

    # Creating specific dates
    birthday = date(1990, 6, 15)
    meeting = datetime(2025, 3, 15, 14, 30, 0)
    print(f"\nBirthday:  {birthday}")
    print(f"Meeting:   {meeting}")

    # Formatting (strftime)
    print(f"\nFormatting:")
    print(f"  ISO:      {meeting.isoformat()}")
    print(f"  Custom:   {meeting.strftime('%B %d, %Y at %I:%M %p')}")
    print(f"  Compact:  {meeting.strftime('%Y%m%d_%H%M')}")

    # Parsing (strptime)
    parsed = datetime.strptime("2025-03-15 14:30", "%Y-%m-%d %H:%M")
    print(f"  Parsed:   {parsed}")

    # Timedelta: arithmetic with dates
    week = timedelta(weeks=1)
    day = timedelta(days=1)
    print(f"\nTimedelta:")
    print(f"  Next week:   {today + week}")
    print(f"  Yesterday:   {today - day}")
    print(f"  90 days out: {today + timedelta(days=90)}")

    # Duration between dates
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    duration = end - start
    print(f"  {start} to {end}: {duration.days} days")

    # Timezone-aware
    utc = datetime.now(timezone.utc)
    est = timezone(timedelta(hours=-5))
    est_time = utc.astimezone(est)
    print(f"\n  UTC: {utc.strftime('%H:%M')}")
    print(f"  EST: {est_time.strftime('%H:%M')}")


def argparse_demo():
    """argparse module for CLI argument parsing (simulated)."""
    import argparse

    # Create parser
    parser = argparse.ArgumentParser(
        prog="myapp",
        description="A demo application",
        epilog="Example: myapp input.txt -o output.txt -v",
    )

    # Positional argument
    parser.add_argument("input", help="Input file path")

    # Optional arguments
    parser.add_argument("-o", "--output", default="result.txt",
                        help="Output file (default: result.txt)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="Number of iterations")
    parser.add_argument("--format", choices=["json", "csv", "txt"],
                        default="json", help="Output format")

    # Simulate parsing (instead of sys.argv)
    test_args = ["data.txt", "-o", "out.json", "-v", "-n", "5", "--format", "json"]
    args = parser.parse_args(test_args)

    print(f"Parsed arguments:")
    print(f"  input:   {args.input}")
    print(f"  output:  {args.output}")
    print(f"  verbose: {args.verbose}")
    print(f"  count:   {args.count}")
    print(f"  format:  {args.format}")

    # Show help text
    print(f"\nHelp output:")
    parser.print_help()


if __name__ == "__main__":
    sections = [
        ("collections", collections_demo),
        ("itertools", itertools_demo),
        ("functools", functools_demo),
        ("datetime", datetime_demo),
        ("argparse", argparse_demo),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

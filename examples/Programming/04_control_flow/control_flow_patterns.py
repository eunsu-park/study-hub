"""
Control Flow Patterns

Demonstrates the main tools for directing program execution:
1. Conditional branching — if/elif/else, guard clauses, match statement
2. Loops — for, while, break/continue, loop invariants
3. Recursion — base case + recursive case; converted to iteration
4. Iterators and generators — lazy, memory-efficient data processing
5. Exception vs. Result-style error flow — two ways to propagate failure

The thread: every non-trivial program needs decisions, repetition, and a
strategy for abnormal cases. Choosing the right tool keeps code readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, Union


# =============================================================================
# 1. CONDITIONAL BRANCHING
# =============================================================================

def classify_temperature_if(t: float) -> str:
    """Cascade of if/elif/else — readable for a small number of buckets."""
    if t < 0:
        return "freezing"
    elif t < 15:
        return "cold"
    elif t < 25:
        return "mild"
    elif t < 35:
        return "warm"
    else:
        return "hot"


def parse_command_guard(cmd: str) -> str:
    """
    Guard clauses flatten nesting. Each early-return handles a specific edge,
    and the 'happy path' stays at the top indentation level.
    """
    if not cmd:
        return "error: empty command"
    if cmd.startswith("#"):
        return "skipped: comment"
    if len(cmd) > 100:
        return "error: command too long"

    # Happy path — no extra indentation
    return f"executing: {cmd}"


@dataclass
class Shape:
    kind: str
    a: float
    b: float = 0.0


def describe_shape(shape: Shape) -> str:
    """
    Match statement (Python 3.10+) for structured dispatch. Each arm binds
    its own pattern; unmatched cases are caught by `_` or raise.
    """
    match shape:
        case Shape(kind="circle", a=r):
            return f"circle with radius {r}"
        case Shape(kind="square", a=side):
            return f"square with side {side}"
        case Shape(kind="rectangle", a=w, b=h):
            return f"rectangle {w}x{h}"
        case _:
            return f"unknown shape: {shape.kind}"


def demonstrate_conditionals() -> None:
    for t in [-5, 10, 22, 30, 40]:
        print(f"  {t}°C -> {classify_temperature_if(t)}")

    for cmd in ["", "ls -la", "# comment", "x" * 101]:
        print(f"  {cmd!r:<20} -> {parse_command_guard(cmd)}")

    for shape in [Shape("circle", 5), Shape("rectangle", 3, 4), Shape("triangle", 1)]:
        print(f"  {describe_shape(shape)}")


# =============================================================================
# 2. LOOPS AND INVARIANTS
# =============================================================================

def sum_squares_for(numbers: Iterable[int]) -> int:
    """
    Loop invariant: after processing the first k elements,
    `total` equals sum(n*n for n in processed).
    The invariant holds at entry, is preserved by each iteration,
    and yields the correct result when the loop exits.
    """
    total = 0  # invariant holds trivially for k=0
    for n in numbers:
        total += n * n  # preserves invariant for k+1
    return total  # at exit, k = len(numbers)


def first_match_while(text: str, target: str) -> int:
    """
    `while` with manual index — use when the loop condition depends on
    multiple variables that change in different ways.
    """
    i = 0
    while i <= len(text) - len(target):
        if text[i:i + len(target)] == target:
            return i
        i += 1
    return -1


def next_prime_with_break(start: int) -> int:
    """`break` short-circuits inner loops when the answer is found."""
    n = max(start, 2)
    while True:
        is_prime = True
        for divisor in range(2, int(n ** 0.5) + 1):
            if n % divisor == 0:
                is_prime = False
                break  # stop checking divisors
        if is_prime:
            return n
        n += 1


def demonstrate_loops() -> None:
    print(f"  sum of squares of [1..5] = {sum_squares_for(range(1, 6))}")
    print(f"  first 'lo' in 'hello world' = index {first_match_while('hello world', 'lo')}")
    print(f"  next prime >= 20 = {next_prime_with_break(20)}")


# =============================================================================
# 3. RECURSION — and the iterative equivalent
# =============================================================================

def factorial_recursive(n: int) -> int:
    """Classic recursive formulation: base case + recursive step."""
    if n <= 1:  # base case prevents infinite recursion
        return 1
    return n * factorial_recursive(n - 1)  # recursive case shrinks n


def factorial_iterative(n: int) -> int:
    """Same problem, iterative form — no stack depth concerns."""
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def demonstrate_recursion() -> None:
    for n in [0, 1, 5, 10]:
        r = factorial_recursive(n)
        i = factorial_iterative(n)
        assert r == i
        print(f"  {n}! = {r} (recursive == iterative)")


# =============================================================================
# 4. ITERATORS AND GENERATORS
# =============================================================================

def countdown(start: int) -> Iterator[int]:
    """
    Generator: produces values lazily, one at a time.
    Memory use is O(1) regardless of `start` — crucial for huge sequences.
    """
    while start > 0:
        yield start  # pause here; resume on next() call
        start -= 1


def fibonacci(limit: int) -> Generator[int, None, None]:
    """Generator for Fibonacci numbers up to `limit`. Lazy and composable."""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


def demonstrate_generators() -> None:
    print(f"  countdown(3) -> {list(countdown(3))}")
    print(f"  fib < 50     -> {list(fibonacci(50))}")
    # Generators compose with other iteration tools
    even_fibs = [x for x in fibonacci(100) if x % 2 == 0]
    print(f"  even fibs < 100 -> {even_fibs}")


# =============================================================================
# 5. EXCEPTIONS vs. RESULT-STYLE ERROR FLOW
# =============================================================================

def divide_exception(a: float, b: float) -> float:
    """Raises on error. Caller must wrap in try/except or the error propagates."""
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b


@dataclass(frozen=True)
class Ok:
    value: float


@dataclass(frozen=True)
class Err:
    message: str


Result = Union[Ok, Err]


def divide_result(a: float, b: float) -> Result:
    """
    Returns Ok or Err explicitly. The type signature makes failure visible,
    and callers must pattern-match — no accidental silent propagation.
    """
    if b == 0:
        return Err("cannot divide by zero")
    return Ok(a / b)


def demonstrate_error_styles() -> None:
    # Exception style — concise, but failure is invisible at the call site
    try:
        print(f"  divide_exception(10, 2) = {divide_exception(10, 2)}")
        divide_exception(10, 0)
    except ZeroDivisionError as e:
        print(f"  divide_exception(10, 0) raised: {e}")

    # Result style — failure explicit in the return type
    for (a, b) in [(10, 2), (10, 0)]:
        match divide_result(a, b):
            case Ok(value=v):
                print(f"  divide_result({a}, {b}) -> Ok({v})")
            case Err(message=m):
                print(f"  divide_result({a}, {b}) -> Err({m!r})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sections = [
        ("1. CONDITIONAL BRANCHING", demonstrate_conditionals),
        ("2. LOOPS AND INVARIANTS", demonstrate_loops),
        ("3. RECURSION (and iterative equivalent)", demonstrate_recursion),
        ("4. ITERATORS AND GENERATORS", demonstrate_generators),
        ("5. EXCEPTION vs. RESULT ERROR FLOW", demonstrate_error_styles),
    ]
    for title, fn in sections:
        print("=" * 70)
        print(title)
        print("=" * 70)
        fn()
        print()


if __name__ == "__main__":
    main()

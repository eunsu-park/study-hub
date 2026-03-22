"""
Python Closures and Scope

Demonstrates:
- LEGB scope resolution
- nonlocal keyword
- Factory function patterns (multiplier, counter)
- Loop variable capture pitfall and fix
- Closure vs callable class comparison
- Memoization as a closure pattern
"""

from typing import Callable


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# LEGB Scope Resolution
# =============================================================================

section("LEGB Scope Resolution")

# L — Local
# E — Enclosing
# G — Global
# B — Built-in

x = "global"  # Global scope


def outer():
    x = "enclosing"  # Enclosing scope

    def inner():
        x = "local"  # Local scope
        print(f"  inner sees: x = '{x}'")  # L wins

    def inner_no_local():
        print(f"  inner_no_local sees: x = '{x}'")  # E wins

    inner()
    inner_no_local()


print(f"Global x = '{x}'")
outer()
print(f"Global x after outer() = '{x}'")  # Unchanged


# Built-in scope example
def demonstrate_builtin():
    # 'len' resolved from built-in scope (B)
    result = len([1, 2, 3])
    print(f"  len([1,2,3]) via built-in scope = {result}")


demonstrate_builtin()


# =============================================================================
# nonlocal Keyword
# =============================================================================

section("nonlocal Keyword")


def make_counter_nonlocal():
    """Counter using nonlocal to modify enclosing variable."""
    count = 0

    def increment():
        # Why: 'nonlocal' lets the inner function rebind a variable in the
        # enclosing scope — without it, Python treats 'count' as a new local
        # variable and raises UnboundLocalError on 'count += 1'
        nonlocal count
        count += 1
        return count

    def reset():
        nonlocal count
        count = 0

    return increment, reset


inc, rst = make_counter_nonlocal()
print(f"  inc() = {inc()}")
print(f"  inc() = {inc()}")
print(f"  inc() = {inc()}")
rst()
print(f"  After reset, inc() = {inc()}")


def without_nonlocal_demo():
    """Show what happens without nonlocal (read-only access is fine)."""
    value = 10

    def read_only():
        # Reading the enclosing variable is allowed without nonlocal
        return value * 2

    print(f"  read_only() = {read_only()}")


without_nonlocal_demo()


# =============================================================================
# Factory Function Patterns
# =============================================================================

section("Factory Function Patterns")


# Multiplier factory
def make_multiplier(factor: int) -> Callable[[int], int]:
    """Return a closure that multiplies by factor."""
    def multiply(x: int) -> int:
        return x * factor  # 'factor' captured from enclosing scope
    return multiply


double = make_multiplier(2)
triple = make_multiplier(3)
times_ten = make_multiplier(10)

print("Multiplier factory:")
print(f"  double(5)    = {double(5)}")
print(f"  triple(5)    = {triple(5)}")
print(f"  times_ten(5) = {times_ten(5)}")

# Each closure captures its own 'factor'
print(f"\n  double.__closure__[0].cell_contents  = {double.__closure__[0].cell_contents}")
print(f"  triple.__closure__[0].cell_contents  = {triple.__closure__[0].cell_contents}")


# Counter factory
def make_counter(start: int = 0, step: int = 1) -> Callable[[], int]:
    """Return a closure that counts from start by step."""
    current = start

    def next_value() -> int:
        nonlocal current
        value = current
        current += step
        return value

    return next_value


evens = make_counter(start=0, step=2)
odds = make_counter(start=1, step=2)

print("\nCounter factory:")
print(f"  evens: {[evens() for _ in range(5)]}")
print(f"  odds:  {[odds() for _ in range(5)]}")


# Power factory
def make_power(exponent: int) -> Callable[[float], float]:
    """Return a closure that raises to exponent."""
    def power(base: float) -> float:
        return base ** exponent
    return power


square = make_power(2)
cube = make_power(3)

print("\nPower factory:")
print(f"  square(4) = {square(4)}")
print(f"  cube(3)   = {cube(3)}")


# =============================================================================
# Loop Variable Capture Pitfall and Fix
# =============================================================================

section("Loop Variable Capture Pitfall and Fix")

# PITFALL: all closures share the same 'i' variable
funcs_broken = []
for i in range(5):
    funcs_broken.append(lambda: i)  # captures 'i' by reference

print("Broken (all see final i=4):")
print(f"  {[f() for f in funcs_broken]}")  # [4, 4, 4, 4, 4]

# FIX 1: default argument captures value at definition time
funcs_fixed_default = []
for i in range(5):
    # Why: default arguments are evaluated at function definition time, not call time,
    # so each lambda gets its own copy of i's current value
    funcs_fixed_default.append(lambda i=i: i)

print("\nFixed with default argument:")
print(f"  {[f() for f in funcs_fixed_default]}")  # [0, 1, 2, 3, 4]

# FIX 2: factory function creates a new enclosing scope per iteration
def make_lambda(val: int) -> Callable[[], int]:
    return lambda: val


funcs_fixed_factory = [make_lambda(i) for i in range(5)]

print("\nFixed with factory function:")
print(f"  {[f() for f in funcs_fixed_factory]}")  # [0, 1, 2, 3, 4]


# =============================================================================
# Closure vs Callable Class Comparison
# =============================================================================

section("Closure vs Callable Class Comparison")


# Closure approach
def make_adder_closure(n: int) -> Callable[[int], int]:
    """Closure-based adder."""
    def add(x: int) -> int:
        return x + n
    return add


# Callable class approach
class Adder:
    """Class-based adder (equivalent to closure)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x: int) -> int:
        return x + self.n


add5_closure = make_adder_closure(5)
add5_class = Adder(5)

print("Adder comparison:")
print(f"  Closure:  add5_closure(10) = {add5_closure(10)}")
print(f"  Class:    add5_class(10)   = {add5_class(10)}")

# Introspection differences
print(f"\n  Closure type:  {type(add5_closure)}")
print(f"  Class type:    {type(add5_class)}")
print(f"  Closure captured n: {add5_closure.__closure__[0].cell_contents}")
print(f"  Class stored n:     {add5_class.n}")

# When to prefer each:
print("""
  Prefer closure when:
    - Simple, single-purpose callable
    - No need for inspection or mutation of state
    - Functional-style composition

  Prefer callable class when:
    - Multiple methods needed (e.g., reset, inspect)
    - State needs to be readable/writable externally
    - Inheritance or isinstance checks required
""")


# =============================================================================
# Memoization as a Closure Pattern
# =============================================================================

section("Memoization as a Closure Pattern")


def make_memoized(func: Callable) -> Callable:
    """Return a memoized version of func using a closure cache."""
    # Why: the cache dict lives in the enclosing scope of 'wrapper',
    # persisting across calls without any global state
    cache: dict = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
            print(f"  Cache miss  for {func.__name__}{args} -> {cache[args]}")
        else:
            print(f"  Cache hit   for {func.__name__}{args} -> {cache[args]}")
        return cache[args]

    wrapper.cache = cache  # expose cache for inspection
    wrapper.__name__ = func.__name__
    return wrapper


def fib(n: int) -> int:
    """Recursive Fibonacci (slow without memoization)."""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


@make_memoized
def fib_memo(n: int) -> int:
    """Recursive Fibonacci with memoization closure."""
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


print("Memoized Fibonacci:")
for n in [5, 5, 6, 7]:
    result = fib_memo(n)

print(f"\n  Cache after calls: {fib_memo.cache}")


# Memoization with max size (LRU-style, simple version)
def make_lru_cache(maxsize: int = 4) -> Callable:
    """Return a simple LRU-style memoization decorator."""
    def decorator(func: Callable) -> Callable:
        cache: dict = {}
        access_order: list = []

        def wrapper(*args):
            if args in cache:
                access_order.remove(args)
                access_order.append(args)
                return cache[args]

            result = func(*args)
            cache[args] = result
            access_order.append(args)

            if len(cache) > maxsize:
                oldest = access_order.pop(0)
                del cache[oldest]
                print(f"  Evicted {oldest} from cache")

            return result

        wrapper.cache = cache
        return wrapper

    return decorator


@make_lru_cache(maxsize=3)
def expensive(n: int) -> int:
    return n * n


print("\nSimple LRU cache (maxsize=3):")
for val in [1, 2, 3, 4, 5, 3, 2]:
    result = expensive(val)
    print(f"  expensive({val}) = {result}, cache keys: {list(expensive.cache.keys())}")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Closure patterns covered:
1. LEGB scope resolution
   - Local -> Enclosing -> Global -> Built-in lookup order

2. nonlocal keyword
   - Required to rebind (not just read) enclosing variables

3. Factory functions
   - make_multiplier(), make_counter(), make_power()
   - Each call creates an independent closure with its own captured state

4. Loop variable capture pitfall
   - Bug:  lambda: i  (all share the same 'i')
   - Fix1: lambda i=i: i  (default arg captures value)
   - Fix2: factory function per iteration

5. Closure vs callable class
   - Closures: concise, functional, no external state access
   - Classes: inspectable, mutable state, supports inheritance

6. Memoization via closure
   - Cache dict lives in enclosing scope across calls
   - No global state required
   - Foundation of functools.lru_cache
""")

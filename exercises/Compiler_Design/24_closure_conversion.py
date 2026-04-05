"""
Exercises for Lesson 24: Closure Conversion
Topic: Compiler_Design

Demonstrates lambda lifting, closure conversion, defunctionalization, and CPS.
"""

from dataclasses import dataclass
from typing import List, Set, Callable


# === Exercise 1: Free Variables ===

def exercise_1():
    """Compute free variables of nested lambda expressions."""
    print("Exercise 1: Free Variables")
    print()

    examples = [
        ("\\x -> x + y",        {"x"}, {"y"}),
        ("\\x -> \\y -> x + y + z", {"x", "y"}, {"z"}),
        ("\\f -> \\x -> f (f x)", {"f", "x"}, set()),
        ("\\x -> let y = x + 1 in y * z", {"x"}, {"z"}),
    ]

    for expr, bound, free in examples:
        print(f"  {expr}")
        print(f"    Bound: {bound}")
        print(f"    Free:  {free}")
    print()


# === Exercise 2: Lambda Lifting ===

def exercise_2():
    """Transform nested functions using lambda lifting."""
    print("Exercise 2: Lambda Lifting (3-level nesting)")
    print()

    print("  Original:")
    print("    def outer(a):")
    print("        def middle(b):")
    print("            def inner(c):")
    print("                return a + b + c")
    print("            return inner(10)")
    print("        return middle(20)")
    print()

    print("  After lambda lifting:")
    print("    def inner_lifted(a, b, c):")
    print("        return a + b + c")
    print()
    print("    def middle_lifted(a, b):")
    print("        return inner_lifted(a, b, 10)")
    print()
    print("    def outer(a):")
    print("        return middle_lifted(a, 20)")
    print()

    # Verify correctness
    def inner_lifted(a, b, c): return a + b + c
    def middle_lifted(a, b): return inner_lifted(a, b, 10)
    def outer(a): return middle_lifted(a, 20)

    result = outer(100)
    print(f"  outer(100) = {result}")
    assert result == 130, f"Expected 130, got {result}"
    print()


# === Exercise 3: Flat Closure Conversion ===

def exercise_3():
    """Implement flat closure conversion."""
    print("Exercise 3: Flat Closure Conversion")
    print()

    # Original: make_adder(n) returns a closure that adds n
    def make_adder(n):
        def adder(x):
            return x + n
        return adder

    # After closure conversion:
    def adder_code(env, x):
        return x + env['n']

    def make_adder_converted(n):
        env = {'n': n}
        return (adder_code, env)

    def call_closure(closure, *args):
        code, env = closure
        return code(env, *args)

    # Test
    add5 = make_adder_converted(5)
    add10 = make_adder_converted(10)

    print(f"  add5(3)  = {call_closure(add5, 3)}")
    print(f"  add10(3) = {call_closure(add10, 3)}")
    print(f"  add5 env = {add5[1]}")
    print(f"  add10 env = {add10[1]}")
    print()


# === Exercise 4: Defunctionalization ===

def exercise_4():
    """Defunctionalize a program using map/filter/fold."""
    print("Exercise 4: Defunctionalization")
    print()

    # Original higher-order program:
    # result = map(lambda x: x * 2, filter(lambda x: x > 0, [-1, 2, -3, 4]))

    # Defunctionalized:
    class Double: pass
    class Positive: pass

    def apply_func(f, x):
        if isinstance(f, Double):
            return x * 2
        raise ValueError(f"Unknown function: {f}")

    def apply_pred(p, x):
        if isinstance(p, Positive):
            return x > 0
        raise ValueError(f"Unknown predicate: {p}")

    def my_filter(pred, lst):
        return [x for x in lst if apply_pred(pred, x)]

    def my_map(func, lst):
        return [apply_func(func, x) for x in lst]

    data = [-1, 2, -3, 4]
    filtered = my_filter(Positive(), data)
    result = my_map(Double(), filtered)

    print(f"  Input: {data}")
    print(f"  After filter(Positive): {filtered}")
    print(f"  After map(Double): {result}")
    print()
    print("  No higher-order functions used -- all dispatch via apply_func/apply_pred")
    print()


# === Exercise 5: CPS Transformation ===

def exercise_5():
    """Convert recursive factorial to CPS."""
    print("Exercise 5: CPS Transformation")
    print()

    # Direct style
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)

    # CPS style
    def factorial_cps(n, k):
        if n == 0:
            return k(1)
        return factorial_cps(n - 1, lambda result: k(n * result))

    print("  Direct style:")
    print(f"    factorial(5) = {factorial(5)}")
    print()

    print("  CPS style:")
    print(f"    factorial_cps(5, id) = {factorial_cps(5, lambda x: x)}")
    print()

    # Trace execution
    print("  CPS execution trace for factorial_cps(4, id):")
    steps = [
        "factorial_cps(4, k0)  where k0 = id",
        "factorial_cps(3, k1)  where k1 = \\r -> k0(4 * r)",
        "factorial_cps(2, k2)  where k2 = \\r -> k1(3 * r)",
        "factorial_cps(1, k3)  where k3 = \\r -> k2(2 * r)",
        "factorial_cps(0, k4)  where k4 = \\r -> k3(1 * r)",
        "k4(1) = k3(1) = k2(2) = k1(6) = k0(24) = 24",
    ]
    for step in steps:
        print(f"    {step}")
    print()


def main():
    for i, ex in enumerate([exercise_1, exercise_2, exercise_3, exercise_4, exercise_5], 1):
        print(f"{'=' * 60}")
        print(f"Exercise {i}")
        print(f"{'=' * 60}")
        ex()


if __name__ == "__main__":
    main()

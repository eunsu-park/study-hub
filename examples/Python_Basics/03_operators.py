"""
03 Operators
============
Demonstrates all Python operator types: arithmetic, comparison, logical,
bitwise, assignment, membership, identity, short-circuit evaluation,
and the walrus operator.
"""


def arithmetic_operators():
    """Standard arithmetic operators."""
    a, b = 17, 5
    ops = [
        ("a + b",  a + b),
        ("a - b",  a - b),
        ("a * b",  a * b),
        ("a / b",  a / b),
        ("a // b", a // b),
        ("a % b",  a % b),
        ("a ** b", a ** b),
        ("-a",     -a),
    ]
    for expr, result in ops:
        print(f"  {expr:>8} = {result}")

    # Floor division with negatives (rounds toward negative infinity)
    print(f"\n  -17 // 5  = {-17 // 5}")   # -4, not -3
    print(f"  -17 %  5  = {-17 % 5}")      # 3, not -2


def comparison_operators():
    """Comparison and chained comparisons."""
    print("Basic comparisons:")
    pairs = [(1, 2), (2, 2), (3, 2), ("abc", "abd"), ([1], [1])]
    for a, b in pairs:
        print(f"  {a!r:>6} < {b!r:<6}: {a < b}  |  == : {a == b}  |  > : {a > b}")

    # Chained comparisons (unique to Python)
    x = 5
    print(f"\nChained comparisons (x = {x}):")
    print(f"  1 < x < 10:       {1 < x < 10}")
    print(f"  1 < x < 3:        {1 < x < 3}")
    print(f"  1 < x < 10 < 20:  {1 < x < 10 < 20}")
    print(f"  x == 5 == 5.0:    {x == 5 == 5.0}")


def logical_operators():
    """and, or, not with short-circuit evaluation."""
    print("Truth table for 'and':")
    for a in (True, False):
        for b in (True, False):
            print(f"  {a!s:>5} and {b!s:<5} = {a and b}")

    print("\nTruth table for 'or':")
    for a in (True, False):
        for b in (True, False):
            print(f"  {a!s:>5} or  {b!s:<5} = {a or b}")

    print(f"\n  not True  = {not True}")
    print(f"  not False = {not False}")


def short_circuit_evaluation():
    """Demonstrate short-circuit behavior and practical uses."""

    def side_effect(label, value):
        print(f"    evaluated: {label}")
        return value

    # 'and' stops at first falsy value
    print("Short-circuit 'and':")
    result = side_effect("A=True", True) and side_effect("B=False", False) and side_effect("C=True", True)
    print(f"  Result: {result}\n")

    # 'or' stops at first truthy value
    print("Short-circuit 'or':")
    result = side_effect("A=False", False) or side_effect("B=True", True) or side_effect("C=True", True)
    print(f"  Result: {result}\n")

    # Practical: default values
    name = "" or "Anonymous"
    print(f"  '' or 'Anonymous' = {name!r}")

    config = None
    timeout = config or 30
    print(f"  None or 30 = {timeout}")

    # Practical: guard clause
    data = [1, 2, 3]
    print(f"  data and data[0] = {data and data[0]}")
    empty = []
    print(f"  [] and [][0]     = {empty and 'never reached'}")


def bitwise_operators():
    """Bitwise operations on integers."""
    a, b = 0b1100, 0b1010  # 12, 10

    print(f"  a = {a:04b} ({a})")
    print(f"  b = {b:04b} ({b})")
    print(f"  a & b  = {a & b:04b} ({a & b})")    # AND
    print(f"  a | b  = {a | b:04b} ({a | b})")     # OR
    print(f"  a ^ b  = {a ^ b:04b} ({a ^ b})")     # XOR
    print(f"  ~a     = {~a} (bitwise NOT)")
    print(f"  a << 2 = {a << 2:08b} ({a << 2})")   # Left shift
    print(f"  a >> 1 = {a >> 1:04b} ({a >> 1})")    # Right shift

    # Practical: check if number is power of 2
    for n in [1, 2, 3, 4, 8, 10, 16, 32]:
        is_pow2 = n > 0 and (n & (n - 1)) == 0
        print(f"  {n:>3} is power of 2: {is_pow2}")


def assignment_operators():
    """Augmented assignment operators."""
    x = 100
    ops_and_results = []

    x += 10;  ops_and_results.append(("x += 10", x))
    x -= 5;   ops_and_results.append(("x -= 5",  x))
    x *= 2;   ops_and_results.append(("x *= 2",  x))
    x //= 3;  ops_and_results.append(("x //= 3", x))
    x %= 10;  ops_and_results.append(("x %= 10", x))
    x **= 3;  ops_and_results.append(("x **= 3", x))

    print(f"  Starting: x = 100")
    for expr, val in ops_and_results:
        print(f"  {expr:>10} -> x = {val}")


def membership_and_identity():
    """'in' and 'is' operators."""
    # Membership: in, not in
    fruits = ["apple", "banana", "cherry"]
    print(f"  'banana' in {fruits}: {'banana' in fruits}")
    print(f"  'grape' in {fruits}:  {'grape' in fruits}")

    text = "Hello, World!"
    print(f"  'World' in '{text}': {'World' in text}")

    d = {"a": 1, "b": 2}
    print(f"  'a' in {d}: {'a' in d}")       # Checks keys
    print(f"  1 in {d}:   {1 in d}")          # Keys, not values
    print(f"  1 in {d}.values(): {1 in d.values()}")


def walrus_operator():
    """Assignment expression := (Python 3.8+)."""
    # Without walrus — must compute twice or use temp variable
    data = [1, 5, 3, 8, 2, 9, 4, 7, 6]

    # With walrus in while loop
    print("Processing items > 5:")
    iterator = iter(data)
    results = []
    while (item := next(iterator, None)) is not None:
        if item > 5:
            results.append(item)
    print(f"  Found: {results}")

    # With walrus in list comprehension (filter + transform)
    raw = ["  hello  ", "", "  world  ", "   ", "  python  "]
    cleaned = [stripped for s in raw if (stripped := s.strip())]
    print(f"\n  Raw:     {raw}")
    print(f"  Cleaned: {cleaned}")

    # With walrus to avoid redundant computation
    import re
    text = "Contact: user@example.com or admin@test.org"
    pattern = r"[\w.]+@[\w.]+"
    if (match := re.search(pattern, text)):
        print(f"\n  Found email: {match.group()}")

    # In a condition with len()
    items = [1, 2, 3, 4, 5]
    if (n := len(items)) > 3:
        print(f"  List has {n} items (more than 3)")


def operator_precedence():
    """Demonstrate precedence rules with examples."""
    examples = [
        ("2 + 3 * 4",       2 + 3 * 4),
        ("(2 + 3) * 4",     (2 + 3) * 4),
        ("2 ** 3 ** 2",     2 ** 3 ** 2),      # Right-associative
        ("(2 ** 3) ** 2",   (2 ** 3) ** 2),
        ("-2 ** 2",         -(2 ** 2)),         # ** binds tighter than unary -
        ("not 1 == 2",      not 1 == 2),
        ("True or False and False", True or False and False),  # and before or
    ]
    print("  Precedence examples:")
    for expr_str, result in examples:
        print(f"    {expr_str:>30} = {result}")


if __name__ == "__main__":
    sections = [
        ("Arithmetic Operators", arithmetic_operators),
        ("Comparison Operators", comparison_operators),
        ("Logical Operators", logical_operators),
        ("Short-Circuit Evaluation", short_circuit_evaluation),
        ("Bitwise Operators", bitwise_operators),
        ("Assignment Operators", assignment_operators),
        ("Membership & Identity", membership_and_identity),
        ("Walrus Operator :=", walrus_operator),
        ("Operator Precedence", operator_precedence),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

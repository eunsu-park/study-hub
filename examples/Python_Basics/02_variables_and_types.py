"""
02 Variables and Types
======================
Demonstrates Python's dynamic typing, built-in types, type conversion,
isinstance checks, and numeric operations.
"""


def type_demonstrations():
    """Show Python's built-in types and dynamic typing."""
    # Python is dynamically typed: no type declaration needed
    x = 42
    print(f"x = {x!r:>20}  type = {type(x).__name__}")

    x = 3.14
    print(f"x = {x!r:>20}  type = {type(x).__name__}")

    x = "hello"
    print(f"x = {x!r:>20}  type = {type(x).__name__}")

    x = True
    print(f"x = {x!r:>20}  type = {type(x).__name__}")

    x = None
    print(f"x = {x!r:>20}  type = {type(x).__name__}")

    x = [1, 2, 3]
    print(f"x = {str(x):>20}  type = {type(x).__name__}")

    x = (1, 2, 3)
    print(f"x = {str(x):>20}  type = {type(x).__name__}")

    x = {"a": 1}
    print(f"x = {str(x):>20}  type = {type(x).__name__}")

    x = {1, 2, 3}
    print(f"x = {str(x):>20}  type = {type(x).__name__}")

    # Multiple assignment
    a, b, c = 1, 2.0, "three"
    print(f"\nMultiple assignment: a={a}, b={b}, c={c!r}")

    # Swap without temp variable
    a, b = 10, 20
    a, b = b, a
    print(f"After swap: a={a}, b={b}")


def numeric_operations():
    """Integer, float, and complex number operations."""
    # Integers have arbitrary precision
    big = 10 ** 50
    print(f"10^50 = {big}")
    print(f"Digits: {len(str(big))}")

    # Underscores for readability (Python 3.6+)
    million = 1_000_000
    hex_val = 0xFF_FF
    print(f"\n1_000_000 = {million}")
    print(f"0xFF_FF   = {hex_val}")

    # Float precision
    print(f"\n0.1 + 0.2 = {0.1 + 0.2}")
    print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")

    # Use decimal for exact arithmetic
    from decimal import Decimal
    d1 = Decimal("0.1") + Decimal("0.2")
    print(f"Decimal: 0.1 + 0.2 = {d1}")

    # Complex numbers
    z = 3 + 4j
    print(f"\nComplex: z = {z}")
    print(f"Real part:      {z.real}")
    print(f"Imaginary part: {z.imag}")
    print(f"Conjugate:      {z.conjugate()}")
    print(f"Magnitude:      {abs(z)}")

    # Number bases
    print(f"\nBinary:  {bin(255)}")
    print(f"Octal:   {oct(255)}")
    print(f"Hex:     {hex(255)}")
    print(f"From bin: {int('11111111', 2)}")


def type_conversion():
    """Explicit type casting between types."""
    # String to number
    s = "42"
    n = int(s)
    f = float(s)
    print(f"int('{s}')   = {n}  (type: {type(n).__name__})")
    print(f"float('{s}') = {f}  (type: {type(f).__name__})")

    # Number to string
    print(f"str(42)   = {str(42)!r}")
    print(f"str(3.14) = {str(3.14)!r}")

    # Float/int conversion
    print(f"\nint(3.7)   = {int(3.7)}")       # Truncates
    print(f"int(-3.7)  = {int(-3.7)}")        # Truncates toward zero
    print(f"round(3.7) = {round(3.7)}")       # Rounds

    # Bool conversions — truthy and falsy values
    falsy_values = [0, 0.0, "", [], {}, set(), None, False]
    truthy_values = [1, -1, 0.1, "hello", [0], {0: 0}, True]

    print("\nFalsy values:")
    for v in falsy_values:
        print(f"  bool({v!r:>10}) = {bool(v)}")

    print("Truthy values:")
    for v in truthy_values:
        print(f"  bool({str(v):>10}) = {bool(v)}")


def isinstance_checks():
    """Type checking with isinstance and type hierarchy."""
    values = [42, 3.14, "hello", True, [1, 2], (1,), {"a": 1}, None]

    print(f"{'Value':>12} | {'int':>5} | {'float':>5} | {'str':>5} | {'bool':>5}")
    print("-" * 50)
    for v in values:
        print(
            f"{str(v):>12} | "
            f"{str(isinstance(v, int)):>5} | "
            f"{str(isinstance(v, float)):>5} | "
            f"{str(isinstance(v, str)):>5} | "
            f"{str(isinstance(v, bool)):>5}"
        )

    # bool is a subclass of int
    print(f"\nbool is subclass of int: {issubclass(bool, int)}")
    print(f"True + True = {True + True}")
    print(f"True * 10   = {True * 10}")

    # Check multiple types at once
    x = 42
    print(f"\nisinstance(42, (int, float)): {isinstance(x, (int, float))}")


def identity_and_equality():
    """Difference between == (equality) and is (identity)."""
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a

    print(f"a = {a}, id = {id(a)}")
    print(f"b = {b}, id = {id(b)}")
    print(f"c = a,   id = {id(c)}")
    print(f"\na == b: {a == b}")   # Same value
    print(f"a is b: {a is b}")     # Different objects
    print(f"a is c: {a is c}")     # Same object

    # Small integer caching (-5 to 256)
    x = 256
    y = 256
    print(f"\n256 is 256: {x is y}")  # Cached

    # None should always be compared with 'is'
    val = None
    print(f"\nval is None: {val is None}")  # Correct
    print(f"val == None: {val == None}")    # Works but not idiomatic


if __name__ == "__main__":
    sections = [
        ("Type Demonstrations", type_demonstrations),
        ("Numeric Operations", numeric_operations),
        ("Type Conversion", type_conversion),
        ("isinstance Checks", isinstance_checks),
        ("Identity vs Equality", identity_and_equality),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

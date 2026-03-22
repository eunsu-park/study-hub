"""
Exercise 03: Operators

Practice expression evaluation, bitwise operations, and operator precedence.
"""


def evaluate_expression(a, b, c):
    """Evaluate: a**2 + 2*a*b + b**2 - c

    This is (a + b)**2 - c, but compute it using the expanded form.

    Args:
        a, b, c: Numeric values.

    Returns:
        Result of the expression.
    """
    # TODO: Implement this
    pass


def integer_division_info(dividend, divisor):
    """Return a dict with integer division results.

    Keys: "quotient" (//), "remainder" (%), "divmod" (tuple from divmod).

    Args:
        dividend: The number being divided.
        divisor: The number to divide by.

    Returns:
        Dict with quotient, remainder, and divmod result.
    """
    # TODO: Implement this
    pass


def bitwise_pack(r, g, b):
    """Pack three 8-bit color values into a single 24-bit integer.

    The red channel occupies bits 16-23, green bits 8-15, blue bits 0-7.
    Use bitwise shift (<<) and OR (|) operators.

    Args:
        r: Red value (0-255).
        g: Green value (0-255).
        b: Blue value (0-255).

    Returns:
        A single integer with all three channels packed.
    """
    # TODO: Implement this
    pass


def bitwise_unpack(color):
    """Unpack a 24-bit color integer into (r, g, b) tuple.

    Use bitwise shift (>>) and AND (&) operators.

    Args:
        color: A 24-bit integer representing an RGB color.

    Returns:
        A tuple (r, g, b) where each value is 0-255.
    """
    # TODO: Implement this
    pass


def is_power_of_two(n):
    """Check if a positive integer is a power of two using bitwise ops.

    Hint: A power of two in binary has exactly one '1' bit.
    Use the property: n & (n - 1) == 0 for powers of two.

    Args:
        n: A positive integer.

    Returns:
        True if n is a power of two, False otherwise.
    """
    # TODO: Implement this
    pass


def precedence_puzzle():
    """Return the results of tricky precedence expressions.

    Evaluate these WITHOUT running them first:
        expr1 = 2 + 3 * 4
        expr2 = (2 + 3) * 4
        expr3 = 2 ** 3 ** 2    (right-associative!)
        expr4 = -1 ** 2         (** binds tighter than unary -)
        expr5 = True or False and False

    Returns:
        A tuple (expr1, expr2, expr3, expr4, expr5).
    """
    # TODO: Evaluate each expression and return the tuple
    pass


# === Tests ===

assert evaluate_expression(3, 4, 5) == 44, "Expression (3+4)^2 - 5"
assert evaluate_expression(0, 0, 0) == 0, "Expression all zeros"
assert evaluate_expression(1, -1, 0) == 0, "Expression 1 + (-1) squared"

info = integer_division_info(17, 5)
assert info["quotient"] == 3, "17 // 5"
assert info["remainder"] == 2, "17 % 5"
assert info["divmod"] == (3, 2), "divmod(17, 5)"

assert bitwise_pack(255, 128, 0) == 0xFF8000, "Pack orange"
assert bitwise_pack(0, 0, 0) == 0x000000, "Pack black"
assert bitwise_pack(255, 255, 255) == 0xFFFFFF, "Pack white"

assert bitwise_unpack(0xFF8000) == (255, 128, 0), "Unpack orange"
assert bitwise_unpack(0x000000) == (0, 0, 0), "Unpack black"
assert bitwise_unpack(0xFFFFFF) == (255, 255, 255), "Unpack white"

assert is_power_of_two(1) is True, "2^0"
assert is_power_of_two(16) is True, "2^4"
assert is_power_of_two(15) is False, "15 is not"
assert is_power_of_two(1024) is True, "2^10"

assert precedence_puzzle() == (14, 20, 512, -1, True), "Precedence results"

print("All tests passed!")

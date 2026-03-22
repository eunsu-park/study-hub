"""
Exercise 02: Variables and Types

Practice type conversions, variable swapping, and type checking.
"""


def swap_variables(a, b):
    """Return a tuple with a and b swapped.

    Args:
        a: First value.
        b: Second value.

    Returns:
        A tuple (b, a).
    """
    # TODO: Implement this (use Pythonic swap)
    pass


def safe_int_convert(value):
    """Safely convert a value to an integer.

    If conversion fails, return None instead of raising an error.

    Args:
        value: Any value to convert.

    Returns:
        Integer value, or None if conversion is not possible.
    """
    # TODO: Implement this
    pass


def type_info(value):
    """Return a string describing the type and value.

    Format: "{type_name}: {value}"
    Example: type_info(42) -> "int: 42"

    Args:
        value: Any Python value.

    Returns:
        Formatted type info string.
    """
    # TODO: Implement this
    pass


def convert_to_types(string_number):
    """Convert a string number to int, float, and bool.

    Args:
        string_number: A string representing a number (e.g., "42").

    Returns:
        A dict with keys "int", "float", "bool" and converted values.
        Example: {"int": 42, "float": 42.0, "bool": True}
    """
    # TODO: Implement this
    pass


def is_numeric(value):
    """Check if a value is a numeric type (int, float, or complex).

    Args:
        value: Any Python value.

    Returns:
        True if the value is int, float, or complex; False otherwise.
    """
    # TODO: Implement this
    pass


def multi_assign():
    """Demonstrate multiple assignment.

    Create three variables x, y, z with values 10, 20, 30
    using a single assignment statement, then return them as a tuple.

    Returns:
        A tuple (10, 20, 30).
    """
    # TODO: Implement this
    pass


# === Tests ===

assert swap_variables(1, 2) == (2, 1), "Swap integers"
assert swap_variables("a", "b") == ("b", "a"), "Swap strings"

assert safe_int_convert("42") == 42, "Convert valid string"
assert safe_int_convert("hello") is None, "Convert invalid string"
assert safe_int_convert(3.7) == 3, "Convert float to int"

assert type_info(42) == "int: 42", "Int type info"
assert type_info("hello") == "str: hello", "Str type info"
assert type_info(3.14) == "float: 3.14", "Float type info"

result = convert_to_types("42")
assert result == {"int": 42, "float": 42.0, "bool": True}, "Convert 42"
result_zero = convert_to_types("0")
assert result_zero == {"int": 0, "float": 0.0, "bool": False}, "Convert 0"

assert is_numeric(42) is True, "Int is numeric"
assert is_numeric(3.14) is True, "Float is numeric"
assert is_numeric(1 + 2j) is True, "Complex is numeric"
assert is_numeric("42") is False, "String is not numeric"

assert multi_assign() == (10, 20, 30), "Multi assign"

print("All tests passed!")

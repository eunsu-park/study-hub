"""
Exercise 01: Reading Error Messages

Practice reading tracebacks, identifying error types,
and fixing code based on error messages.
"""


def fix_name_error():
    """Fix the NameError in this function.

    The function should print a greeting using the user's name.

    Returns:
        str: The greeting string.
    """
    # TODO: Fix the NameError
    username = "Alice"
    return f"Hello, {user_name}!"


def fix_type_error(age):
    """Fix the TypeError in this function.

    The function should return a string like "Age: 25".

    Args:
        age: The age as an integer.

    Returns:
        str: Formatted age string.
    """
    # TODO: Fix the TypeError (can't concatenate str and int)
    return "Age: " + age


def fix_index_error(items):
    """Fix the IndexError in this function.

    The function should return the last item in the list.

    Args:
        items: A non-empty list.

    Returns:
        The last item in the list.
    """
    # TODO: Fix the IndexError
    return items[len(items)]


def fix_key_error(data):
    """Fix the KeyError in this function.

    The function should return the user's email, or "N/A" if
    no email is provided.

    Args:
        data: A dictionary that may or may not contain "email".

    Returns:
        str: The email address or "N/A".
    """
    # TODO: Fix the KeyError (key might not exist)
    return data["email"]


def fix_value_error(text):
    """Fix the ValueError in this function.

    The function should convert text to an integer.
    If the text is not a valid integer, return 0.

    Args:
        text: A string that might contain a number.

    Returns:
        int: The parsed integer, or 0 if invalid.
    """
    # TODO: Fix the ValueError (handle invalid input)
    return int(text)


def classify_error(code_snippet):
    """Classify the type of error in the given code snippet.

    Args:
        code_snippet: A string describing the error.

    Returns:
        str: One of "syntax", "runtime", or "logical".
    """
    # TODO: Implement this function
    # Return "syntax" for parse-time errors
    # Return "runtime" for exceptions during execution
    # Return "logical" for wrong results without exceptions
    pass


if __name__ == "__main__":
    # Test fix_name_error
    try:
        result = fix_name_error()
        assert result == "Hello, Alice!", f"Expected 'Hello, Alice!', got {result!r}"
        print("fix_name_error: PASSED")
    except NameError as e:
        print(f"fix_name_error: FAILED - {e}")

    # Test fix_type_error
    try:
        result = fix_type_error(25)
        assert result == "Age: 25", f"Expected 'Age: 25', got {result!r}"
        print("fix_type_error: PASSED")
    except TypeError as e:
        print(f"fix_type_error: FAILED - {e}")

    # Test fix_index_error
    try:
        result = fix_index_error([10, 20, 30])
        assert result == 30, f"Expected 30, got {result}"
        print("fix_index_error: PASSED")
    except IndexError as e:
        print(f"fix_index_error: FAILED - {e}")

    # Test fix_key_error
    try:
        assert fix_key_error({"email": "a@b.com"}) == "a@b.com"
        assert fix_key_error({"name": "Alice"}) == "N/A"
        print("fix_key_error: PASSED")
    except KeyError as e:
        print(f"fix_key_error: FAILED - {e}")

    # Test fix_value_error
    try:
        assert fix_value_error("42") == 42
        assert fix_value_error("hello") == 0
        assert fix_value_error("") == 0
        print("fix_value_error: PASSED")
    except ValueError as e:
        print(f"fix_value_error: FAILED - {e}")

    # Test classify_error
    try:
        assert classify_error("missing colon after if") == "syntax"
        assert classify_error("division by zero") == "runtime"
        assert classify_error("function returns wrong value") == "logical"
        print("classify_error: PASSED")
    except (AssertionError, TypeError) as e:
        print(f"classify_error: FAILED - {e}")

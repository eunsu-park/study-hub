"""
Exercise 09: Type Checking

Practice adding type hints and fixing type-related bugs.
"""


def add_type_hints_basic(name, age, scores):
    """Add proper type hints to this function.

    TODO: Add type annotations to the parameters and return type.

    The function creates a summary string from a name (str),
    age (int), and scores (list of floats).

    Returns a formatted string.
    """
    avg = sum(scores) / len(scores) if scores else 0.0
    return f"{name} (age {age}): avg score = {avg:.1f}"


def add_type_hints_optional(user_id):
    """Add proper type hints including Optional/None.

    TODO: Add type annotations. This function takes an int
    and returns a str or None.
    """
    users = {1: "Alice", 2: "Bob", 3: "Charlie"}
    return users.get(user_id)


def fix_none_safety(user_id):
    """Fix the None-safety bug that mypy would catch.

    This function looks up a user and returns their name in
    uppercase. But find_user might return None.

    TODO: Add type hints AND fix the None-safety bug.

    Args:
        user_id: The user ID to look up.

    Returns:
        str: Uppercase name, or "UNKNOWN" if not found.
    """
    def find_user(uid):
        users = {1: "Alice", 2: "Bob"}
        return users.get(uid)

    user = find_user(user_id)
    return user.upper()  # BUG: crashes if user is None


def create_typed_dict():
    """Create a TypedDict for a configuration object.

    TODO: Define a TypedDict called AppConfig with:
    - host: str
    - port: int
    - debug: bool
    - database_url: str

    Then create and return a valid instance.

    Returns:
        dict: A dictionary matching the AppConfig type.
    """
    # TODO: Define TypedDict and create instance
    pass


def fix_return_type(text):
    """Fix the inconsistent return type.

    This function should always return an int.
    Currently it sometimes returns None.

    TODO: Add type hints and fix so it always returns int.

    Args:
        text: A string that might contain a number.

    Returns:
        int: The parsed number, or 0 if invalid.
    """
    if text.isdigit():
        return int(text)
    # BUG: missing return statement (returns None implicitly)


def add_type_hints_collection(items):
    """Add type hints to a function working with collections.

    TODO: Add proper type annotations.

    This function takes a list of strings and returns a dict
    mapping each unique string to its count.
    """
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


if __name__ == "__main__":
    # Test add_type_hints_basic
    result = add_type_hints_basic("Alice", 30, [90.0, 85.0, 92.0])
    assert "Alice" in result and "89.0" in result
    print("add_type_hints_basic: PASSED")

    # Test add_type_hints_optional
    assert add_type_hints_optional(1) == "Alice"
    assert add_type_hints_optional(999) is None
    print("add_type_hints_optional: PASSED")

    # Test fix_none_safety
    assert fix_none_safety(1) == "ALICE"
    assert fix_none_safety(999) == "UNKNOWN"
    print("fix_none_safety: PASSED")

    # Test create_typed_dict
    config = create_typed_dict()
    assert config is not None
    assert "host" in config
    assert isinstance(config["port"], int)
    assert isinstance(config["debug"], bool)
    print("create_typed_dict: PASSED")

    # Test fix_return_type
    assert fix_return_type("42") == 42
    assert fix_return_type("hello") == 0
    assert fix_return_type("") == 0
    print("fix_return_type: PASSED")

    # Test add_type_hints_collection
    result = add_type_hints_collection(["a", "b", "a", "c", "b", "a"])
    assert result == {"a": 3, "b": 2, "c": 1}
    print("add_type_hints_collection: PASSED")

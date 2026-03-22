"""
Exercise 14: Pythonic Idioms

Refactor non-Pythonic code and practice comprehension patterns.
"""


# --- Exercise 1: Refactor to List Comprehension ---

def squares_of_evens_unpythonic(numbers):
    """NON-PYTHONIC version. Do NOT modify this function."""
    result = []
    for n in numbers:
        if n % 2 == 0:
            result.append(n ** 2)
    return result


def squares_of_evens(numbers):
    """Return squares of even numbers from the input list.

    Rewrite squares_of_evens_unpythonic using a single list comprehension.

    Args:
        numbers: List of integers.

    Returns:
        List of squared even numbers.
    """
    # TODO: Implement in ONE line using a list comprehension
    pass


# --- Exercise 2: Dict Comprehension ---

def invert_dict(d):
    """Invert a dict (swap keys and values) using a dict comprehension.

    Assume all values are unique (1-to-1 mapping).

    Args:
        d: Input dictionary.

    Returns:
        Inverted dictionary.
    """
    # TODO: Implement in ONE line using a dict comprehension
    pass


# --- Exercise 3: Refactor Loop to Enumerate ---

def find_indices_unpythonic(lst, target):
    """NON-PYTHONIC version. Do NOT modify this function."""
    result = []
    i = 0
    while i < len(lst):
        if lst[i] == target:
            result.append(i)
        i += 1
    return result


def find_indices(lst, target):
    """Find all indices where target appears in lst.

    Rewrite find_indices_unpythonic using enumerate and list comprehension.

    Args:
        lst: Input list.
        target: Value to search for.

    Returns:
        List of indices.
    """
    # TODO: Implement using enumerate + list comprehension
    pass


# --- Exercise 4: Use zip for Parallel Iteration ---

def dot_product_unpythonic(vec_a, vec_b):
    """NON-PYTHONIC version. Do NOT modify this function."""
    total = 0
    for i in range(len(vec_a)):
        total += vec_a[i] * vec_b[i]
    return total


def dot_product(vec_a, vec_b):
    """Compute dot product of two vectors.

    Rewrite dot_product_unpythonic using zip and sum.

    Args:
        vec_a: First vector (list of numbers).
        vec_b: Second vector (list of numbers).

    Returns:
        Dot product (sum of element-wise products).
    """
    # TODO: Implement in ONE line using zip + sum + generator expression
    pass


# --- Exercise 5: Ternary and Truthiness ---

def classify_number_unpythonic(n):
    """NON-PYTHONIC version. Do NOT modify this function."""
    if n > 0:
        result = "positive"
    elif n < 0:
        result = "negative"
    else:
        result = "zero"
    return result


def classify_number(n):
    """Return "positive", "negative", or "zero".

    Rewrite classify_number_unpythonic using conditional expressions.

    Args:
        n: A number.

    Returns:
        Classification string.
    """
    # TODO: Implement using ternary (conditional) expressions
    pass


# --- Exercise 6: Comprehension Challenge ---

def matrix_flatten(matrix):
    """Flatten a 2D matrix using a nested list comprehension.

    Example: [[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6]

    Args:
        matrix: List of lists.

    Returns:
        Flat list.
    """
    # TODO: Implement in ONE line using nested list comprehension
    pass


def word_lengths(sentence):
    """Return a dict mapping each unique word to its length.

    Use a dict comprehension. Words split by whitespace, case-insensitive.

    Args:
        sentence: Input string.

    Returns:
        Dict mapping lowercase words to their lengths.
    """
    # TODO: Implement in ONE line using a dict comprehension
    pass


# === Tests ===

# Squares of evens
assert squares_of_evens([1, 2, 3, 4, 5, 6]) == [4, 16, 36], "Squares of evens"
assert squares_of_evens([1, 3, 5]) == [], "No evens"
assert squares_of_evens([]) == [], "Empty list"

# Invert dict
assert invert_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}, "Invert dict"
assert invert_dict({}) == {}, "Invert empty"

# Find indices
assert find_indices([1, 2, 3, 2, 4, 2], 2) == [1, 3, 5], "Find all 2s"
assert find_indices([1, 2, 3], 99) == [], "Not found"

# Dot product
assert dot_product([1, 2, 3], [4, 5, 6]) == 32, "Dot product"
assert dot_product([0, 0], [1, 1]) == 0, "Zero vector"

# Classify number
assert classify_number(5) == "positive", "Positive"
assert classify_number(-3) == "negative", "Negative"
assert classify_number(0) == "zero", "Zero"

# Matrix flatten
assert matrix_flatten([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6], "Flatten"
assert matrix_flatten([[], [1], []]) == [1], "Sparse flatten"

# Word lengths
wl = word_lengths("The quick brown fox")
assert wl == {"the": 3, "quick": 5, "brown": 5, "fox": 3}, "Word lengths"

print("All tests passed!")

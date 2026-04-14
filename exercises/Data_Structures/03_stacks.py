"""
Exercise 03: Stacks

Practice stack-based problems: balanced brackets,
postfix evaluation, and next greater element.
"""


def is_balanced(expression):
    """Check if brackets are balanced.

    Supports (, ), [, ], {, }.

    >>> is_balanced("({[]})")
    True
    >>> is_balanced("([)]")
    False
    >>> is_balanced("")
    True
    """
    # TODO: Implement this
    pass


def eval_postfix(expression):
    """Evaluate a postfix (RPN) expression.

    Supports +, -, *, / (integer division).

    >>> eval_postfix("3 4 +")
    7
    >>> eval_postfix("3 4 2 * +")
    11
    >>> eval_postfix("5 1 2 + 4 * + 3 -")
    14
    """
    # TODO: Implement this
    pass


def next_greater_elements(nums):
    """For each element, find the next element that is greater.

    Return -1 if no greater element exists.
    Use a monotonic stack for O(n) time.

    >>> next_greater_elements([4, 5, 2, 10, 8])
    [5, 10, 10, -1, -1]
    >>> next_greater_elements([3, 2, 1])
    [-1, -1, -1]
    """
    # TODO: Implement this
    pass


def decode_string(s):
    """Decode an encoded string like "3[a2[c]]".

    >>> decode_string("3[a]2[bc]")
    'aaabcbc'
    >>> decode_string("3[a2[c]]")
    'accaccacc'
    >>> decode_string("abc")
    'abc'
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    assert is_balanced("({[]})") is True
    assert is_balanced("([)]") is False
    assert is_balanced("") is True
    assert is_balanced("((())") is False
    print("is_balanced: PASSED")

    assert eval_postfix("3 4 +") == 7
    assert eval_postfix("3 4 2 * +") == 11
    assert eval_postfix("5 1 2 + 4 * + 3 -") == 14
    print("eval_postfix: PASSED")

    assert next_greater_elements([4, 5, 2, 10, 8]) == [5, 10, 10, -1, -1]
    assert next_greater_elements([3, 2, 1]) == [-1, -1, -1]
    assert next_greater_elements([1]) == [-1]
    print("next_greater_elements: PASSED")

    assert decode_string("3[a]2[bc]") == "aaabcbc"
    assert decode_string("3[a2[c]]") == "accaccacc"
    assert decode_string("abc") == "abc"
    print("decode_string: PASSED")

    print("\nAll tests passed!")

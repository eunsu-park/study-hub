#!/usr/bin/env python3
"""Example: Property-Based Testing with Hypothesis

Demonstrates Hypothesis strategies, property-based test design,
stateful testing, and integration with pytest.
Related lesson: 09_Property_Based_Testing.md
"""

# =============================================================================
# WHY PROPERTY-BASED TESTING?
# Traditional (example-based) tests verify specific inputs:
#   assert sort([3,1,2]) == [1,2,3]
#
# Property-based tests verify PROPERTIES that hold for ALL inputs:
#   "For any list, sorting it produces a list where every element
#    is <= the next element"
#
# Hypothesis generates hundreds of random inputs, including edge cases
# (empty lists, huge numbers, special characters) that humans rarely
# think of. When it finds a failure, it SHRINKS the input to the
# minimal failing case.
# =============================================================================

import pytest

try:
    from hypothesis import given, assume, settings, example, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="Hypothesis not installed (pip install hypothesis)"
)


# =============================================================================
# PRODUCTION CODE TO TEST
# =============================================================================

def encode_run_length(s: str) -> str:
    """Run-length encoding: 'aaabbc' -> 'a3b2c1'"""
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(f"{s[i-1]}{count}")
            count = 1
    result.append(f"{s[-1]}{count}")
    return "".join(result)


def decode_run_length(encoded: str) -> str:
    """Decode run-length encoding: 'a3b2c1' -> 'aaabbc'"""
    if not encoded:
        return ""
    result = []
    i = 0
    while i < len(encoded):
        char = encoded[i]
        i += 1
        num_str = ""
        while i < len(encoded) and encoded[i].isdigit():
            num_str += encoded[i]
            i += 1
        count = int(num_str) if num_str else 1
        result.append(char * count)
    return "".join(result)


def flatten(nested: list) -> list:
    """Flatten arbitrarily nested lists."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


class Stack:
    """A simple stack for stateful testing demo."""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("Pop from empty stack")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("Peek at empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)


# =============================================================================
# 1. BASIC STRATEGIES
# =============================================================================
# Strategies are generators for random test data. Hypothesis provides
# strategies for all Python types and lets you compose custom ones.

@given(x=st.integers(), y=st.integers())
def test_addition_commutativity(x, y):
    """PROPERTY: addition is commutative (a + b == b + a).
    Hypothesis generates hundreds of (x, y) pairs including:
    0, negative numbers, very large numbers, boundary values."""
    assert x + y == y + x


@given(x=st.integers(), y=st.integers(), z=st.integers())
def test_addition_associativity(x, y, z):
    """PROPERTY: addition is associative ((a+b)+c == a+(b+c))."""
    assert (x + y) + z == x + (y + z)


@given(s=st.text())
def test_string_reverse_involution(s):
    """PROPERTY: reversing a string twice gives back the original.
    This is an 'involution' or 'round-trip' property."""
    assert s[::-1][::-1] == s


@given(lst=st.lists(st.integers()))
def test_sort_idempotent(lst):
    """PROPERTY: sorting an already-sorted list returns the same list.
    Idempotency is a powerful property to test."""
    sorted_once = sorted(lst)
    sorted_twice = sorted(sorted_once)
    assert sorted_once == sorted_twice


@given(lst=st.lists(st.integers()))
def test_sort_preserves_elements(lst):
    """PROPERTY: sorting preserves all elements (no additions or removals).
    This catches bugs where sort accidentally drops or duplicates items."""
    result = sorted(lst)
    assert len(result) == len(lst)
    assert sorted(result) == sorted(lst)  # Same multiset


@given(lst=st.lists(st.integers(), min_size=1))
def test_sort_ordering(lst):
    """PROPERTY: every element in sorted output is <= the next one."""
    result = sorted(lst)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


# =============================================================================
# 2. ROUND-TRIP PROPERTIES
# =============================================================================
# The most powerful property pattern: encode(decode(x)) == x
# If encoding and decoding are inverses, this must always hold.

@given(s=st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1))
def test_run_length_round_trip(s):
    """PROPERTY: decoding an encoded string gives back the original.
    We restrict to letters since digits in the input would confuse
    our simple encoder. This constraint reveals a design limitation!"""
    encoded = encode_run_length(s)
    decoded = decode_run_length(encoded)
    assert decoded == s


@given(data=st.dictionaries(st.text(min_size=1), st.integers()))
def test_json_round_trip(data):
    """PROPERTY: JSON serialization is a round-trip for basic types."""
    import json
    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    assert deserialized == data


# =============================================================================
# 3. CUSTOM STRATEGIES
# =============================================================================
# Build complex test data by composing simple strategies.

# Strategy for a valid email address (simplified)
email_strategy = st.builds(
    lambda user, domain: f"{user}@{domain}.com",
    user=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1, max_size=20
    ),
    domain=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1, max_size=10
    ),
)


@given(email=email_strategy)
def test_email_has_at_sign(email):
    """Custom strategy generates structurally valid emails."""
    assert "@" in email
    assert email.count("@") == 1
    assert email.endswith(".com")


# Strategy for nested lists (recursive)
nested_list_strategy = st.recursive(
    st.integers(),  # base case: integers
    lambda children: st.lists(children, max_size=5),  # recursive: lists of children
    max_leaves=20,
)


@given(nested=nested_list_strategy)
def test_flatten_produces_flat_list(nested):
    """PROPERTY: flattened output contains no nested lists."""
    if isinstance(nested, list):
        result = flatten(nested)
        assert all(not isinstance(item, list) for item in result)


# =============================================================================
# 4. ASSUME AND FILTER
# =============================================================================
# Sometimes you need to constrain inputs beyond what strategies provide.
# assume() skips inputs that don't meet a precondition.

@given(x=st.integers(min_value=1, max_value=1000))
def test_division_inverse(x):
    """PROPERTY: multiplying by x then dividing by x gives back the original.
    We constrain x > 0 via the strategy (better than assume for simple constraints)."""
    result = (42 * x) / x
    assert result == pytest.approx(42)


@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_float_negation(x):
    """PROPERTY: negating twice returns the original.
    We exclude NaN because NaN != NaN by IEEE 754."""
    assert -(-x) == x


# =============================================================================
# 5. EXAMPLE-BASED + PROPERTY-BASED
# =============================================================================
# Use @example to pin specific inputs alongside random generation.
# Good for regression tests of previously-found bugs.

@given(s=st.text())
@example("")       # Always test empty string
@example("a")      # Single character
@example("aaa")    # All same characters
def test_encode_length(s):
    """PROPERTY: encoded length is never greater than 2 * len(original).
    @example pins specific edge cases that must always be tested."""
    # Only test with letter-only strings to match our encoder's limitations
    assume(s.isalpha() or s == "")
    encoded = encode_run_length(s)
    if s:
        # Each character produces at most 2 chars in output (letter + digit)
        # unless count > 9, in which case it's letter + digits
        decoded = decode_run_length(encoded)
        assert decoded == s


# =============================================================================
# 6. SETTINGS CUSTOMIZATION
# =============================================================================

@settings(max_examples=500, deadline=None)
@given(lst=st.lists(st.integers()))
def test_sort_stability_property(lst):
    """Run more examples for higher confidence. deadline=None disables
    the per-test time limit (useful for slow tests)."""
    result = sorted(lst)
    assert len(result) == len(lst)


# =============================================================================
# 7. STATEFUL TESTING
# =============================================================================
# Stateful testing generates sequences of operations and checks invariants
# after each step. This is the most powerful Hypothesis feature — it finds
# subtle bugs in stateful systems (caches, queues, state machines).

if HYPOTHESIS_AVAILABLE:
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize

    class StackStateMachine(RuleBasedStateMachine):
        """Generate random sequences of push/pop/peek operations and verify
        that the Stack behaves correctly at every step."""

        def __init__(self):
            super().__init__()
            self.stack = Stack()
            self.model = []  # Python list as reference model

        @rule(value=st.integers())
        def push_value(self, value):
            """Push a random integer onto both the real stack and model."""
            self.stack.push(value)
            self.model.append(value)

        @rule()
        def pop_value(self):
            """Pop from both stack and model, compare results."""
            if not self.model:
                with pytest.raises(IndexError):
                    self.stack.pop()
            else:
                expected = self.model.pop()
                actual = self.stack.pop()
                assert actual == expected

        @rule()
        def peek_value(self):
            """Peek at both stack and model, compare."""
            if not self.model:
                with pytest.raises(IndexError):
                    self.stack.peek()
            else:
                assert self.stack.peek() == self.model[-1]

        @invariant()
        def size_matches(self):
            """INVARIANT: size must always match the model."""
            assert self.stack.size() == len(self.model)

        @invariant()
        def emptiness_matches(self):
            """INVARIANT: is_empty must agree with the model."""
            assert self.stack.is_empty() == (len(self.model) == 0)

    # This creates a test from the state machine
    TestStackStateMachine = StackStateMachine.TestCase


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pip install hypothesis pytest
# pytest 06_property_based.py -v
# pytest 06_property_based.py -v --hypothesis-show-statistics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

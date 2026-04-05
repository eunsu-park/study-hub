#!/bin/bash
# Exercises for Lesson 10: Property-Based Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Writing Properties for a String Reversal Function ===
# Problem: Use Hypothesis to write property-based tests for string reversal.
exercise_1() {
    echo "=== Exercise 1: Writing Properties for a String Reversal Function ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from hypothesis import given, settings
from hypothesis import strategies as st


def reverse_string(s: str) -> str:
    """Simple string reversal."""
    return s[::-1]


@given(st.text())
def test_reverse_involution(s):
    """Reversing twice returns the original string."""
    assert reverse_string(reverse_string(s)) == s


@given(st.text())
def test_reverse_preserves_length(s):
    """Reversal does not change the length."""
    assert len(reverse_string(s)) == len(s)


@given(st.text())
def test_reverse_preserves_characters(s):
    """Reversal contains exactly the same characters."""
    assert sorted(reverse_string(s)) == sorted(s)


@given(st.text(), st.text())
def test_reverse_of_concatenation(a, b):
    """rev(a + b) == rev(b) + rev(a)."""
    assert reverse_string(a + b) == reverse_string(b) + reverse_string(a)


@given(st.text(min_size=1))
def test_reverse_first_last(s):
    """First char of reversed == last char of original."""
    assert reverse_string(s)[0] == s[-1]
    assert reverse_string(s)[-1] == s[0]
SOLUTION
}

# === Exercise 2: Using Hypothesis Strategies for Data Validation ===
# Problem: Validate an email-processing function using Hypothesis strategies.
exercise_2() {
    echo "=== Exercise 2: Using Hypothesis Strategies for Data Validation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import re
from hypothesis import given, assume, settings
from hypothesis import strategies as st


def normalize_email(email: str) -> str:
    """Lowercase and strip whitespace from an email address."""
    return email.strip().lower()


def is_valid_age(age: int) -> bool:
    """Check if age is within a valid range."""
    return 0 <= age <= 150


def parse_csv_row(row: str) -> list[str]:
    """Split a CSV row into fields."""
    return [field.strip() for field in row.split(",")]


# Strategy: generate plausible email-like strings
email_strategy = st.from_regex(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", fullmatch=True
)


@given(email_strategy)
def test_normalize_email_is_lowercase(email):
    """Normalized email is always lowercase."""
    result = normalize_email(email)
    assert result == result.lower()


@given(st.integers())
def test_valid_age_boundary(age):
    """is_valid_age accepts exactly 0..150."""
    if 0 <= age <= 150:
        assert is_valid_age(age)
    else:
        assert not is_valid_age(age)


@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=",")),
               min_size=1, max_size=10))
def test_parse_csv_round_trip(fields):
    """Joining then parsing recovers the original fields."""
    row = ", ".join(fields)
    parsed = parse_csv_row(row)
    assert parsed == [f.strip() for f in fields]


@given(st.text())
def test_normalize_email_idempotent(email):
    """Normalizing twice gives the same result as normalizing once."""
    once = normalize_email(email)
    twice = normalize_email(once)
    assert once == twice
SOLUTION
}

# === Exercise 3: Custom Strategies for Domain Objects ===
# Problem: Build custom Hypothesis strategies to generate domain-specific
# test data (e.g., Money, Product).
exercise_3() {
    echo "=== Exercise 3: Custom Strategies for Domain Objects ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass
from decimal import Decimal
from hypothesis import given, settings
from hypothesis import strategies as st


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __mul__(self, factor: int) -> "Money":
        return Money(self.amount * factor, self.currency)


@dataclass
class Product:
    name: str
    price: Money
    quantity: int


# -- Custom strategies --

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "KRW"]


@st.composite
def money_strategy(draw, currency=None):
    """Generate a Money object with a valid amount and currency."""
    amt = draw(st.decimals(min_value=Decimal("0.01"),
                           max_value=Decimal("99999.99"),
                           places=2,
                           allow_nan=False,
                           allow_infinity=False))
    cur = currency or draw(st.sampled_from(CURRENCIES))
    return Money(amount=amt, currency=cur)


@st.composite
def product_strategy(draw):
    """Generate a Product with realistic constraints."""
    name = draw(st.text(min_size=1, max_size=50,
                        alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))))
    price = draw(money_strategy(currency="USD"))
    qty = draw(st.integers(min_value=0, max_value=10000))
    return Product(name=name, price=price, quantity=qty)


@given(money_strategy(currency="USD"), money_strategy(currency="USD"))
def test_money_addition_commutative(a, b):
    """Addition of same-currency Money is commutative."""
    assert a + b == b + a


@given(money_strategy(currency="USD"))
def test_money_multiply_by_one(m):
    """Multiplying by 1 returns an equal Money."""
    assert m * 1 == m


@given(money_strategy(currency="USD"), st.integers(min_value=0, max_value=100))
def test_money_multiply_equivalent_to_repeated_add(m, n):
    """m * n == m added n times."""
    result = m * n
    total = Money(Decimal("0.00"), m.currency)
    for _ in range(n):
        total = total + m
    assert result == total


@given(product_strategy())
def test_product_total_non_negative(product):
    """Product total (price * quantity) is non-negative."""
    total = product.price * product.quantity
    assert total.amount >= 0
SOLUTION
}

# === Exercise 4: Stateful Testing with RuleBasedStateMachine ===
# Problem: Use Hypothesis stateful testing to verify a stack implementation
# against a list-based model.
exercise_4() {
    echo "=== Exercise 4: Stateful Testing with RuleBasedStateMachine ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from hypothesis import settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    precondition,
    invariant,
    initialize,
)
from hypothesis import strategies as st


class BoundedStack:
    """A stack with a maximum capacity."""

    def __init__(self, capacity: int):
        self._items: list = []
        self._capacity = capacity

    def push(self, item) -> None:
        if len(self._items) >= self._capacity:
            raise OverflowError("Stack is full")
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]

    def size(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def is_full(self) -> bool:
        return len(self._items) >= self._capacity


class StackStateMachine(RuleBasedStateMachine):
    """Compare BoundedStack against a simple list model."""

    def __init__(self):
        super().__init__()
        self.capacity = 5
        self.stack = BoundedStack(self.capacity)
        self.model: list = []  # Reference model

    @rule(value=st.integers())
    def push(self, value):
        if len(self.model) < self.capacity:
            self.stack.push(value)
            self.model.append(value)
        else:
            try:
                self.stack.push(value)
                assert False, "Should have raised OverflowError"
            except OverflowError:
                pass

    @precondition(lambda self: len(self.model) > 0)
    @rule()
    def pop(self):
        actual = self.stack.pop()
        expected = self.model.pop()
        assert actual == expected

    @precondition(lambda self: len(self.model) > 0)
    @rule()
    def peek(self):
        actual = self.stack.peek()
        expected = self.model[-1]
        assert actual == expected

    @invariant()
    def size_matches(self):
        assert self.stack.size() == len(self.model)

    @invariant()
    def empty_flag_correct(self):
        assert self.stack.is_empty() == (len(self.model) == 0)

    @invariant()
    def full_flag_correct(self):
        assert self.stack.is_full() == (len(self.model) >= self.capacity)


# Hypothesis discovers sequences of push/pop/peek that might break invariants
TestStackStateMachine = StackStateMachine.TestCase
SOLUTION
}

# === Exercise 5: Finding Edge Cases in a Sorting Algorithm ===
# Problem: Use property-based testing to verify sorting algorithm correctness
# and discover edge cases.
exercise_5() {
    echo "=== Exercise 5: Finding Edge Cases in a Sorting Algorithm ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from hypothesis import given, settings, example
from hypothesis import strategies as st


def insertion_sort(lst: list) -> list:
    """Simple insertion sort (returns a new sorted list)."""
    result = lst[:]
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result


# -- Property 1: Output is sorted --
@given(st.lists(st.integers()))
def test_result_is_sorted(lst):
    """Every adjacent pair in the output satisfies a <= b."""
    result = insertion_sort(lst)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


# -- Property 2: Output is a permutation of input --
@given(st.lists(st.integers()))
def test_result_is_permutation(lst):
    """Sorted output contains exactly the same elements as input."""
    result = insertion_sort(lst)
    assert sorted(result) == sorted(lst)
    assert len(result) == len(lst)


# -- Property 3: Idempotence --
@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    """Sorting an already-sorted list returns the same list."""
    once = insertion_sort(lst)
    twice = insertion_sort(once)
    assert once == twice


# -- Property 4: Minimum and maximum preservation --
@given(st.lists(st.integers(), min_size=1))
def test_first_is_min_last_is_max(lst):
    """First element of sorted list is min, last is max."""
    result = insertion_sort(lst)
    assert result[0] == min(lst)
    assert result[-1] == max(lst)


# -- Property 5: Stability (equal elements keep relative order) --
@given(st.lists(st.tuples(st.integers(min_value=0, max_value=5),
                           st.integers())))
def test_sort_stability(pairs):
    """For elements with equal sort keys, original order is preserved."""
    # Sort only by first element of tuple
    def sort_by_key(lst):
        result = lst[:]
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j][0] > key[0]:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result

    result = sort_by_key(pairs)
    # Group by first element and check that second elements
    # appear in original relative order
    from collections import defaultdict
    original_order = defaultdict(list)
    result_order = defaultdict(list)
    for k, v in pairs:
        original_order[k].append(v)
    for k, v in result:
        result_order[k].append(v)
    for k in original_order:
        assert original_order[k] == result_order[k]


# -- Edge case examples --
@given(st.just([]))
def test_empty_list(lst):
    assert insertion_sort(lst) == []


@given(st.just([42]))
def test_single_element(lst):
    assert insertion_sort(lst) == [42]


@example([3, 3, 3, 3])
@given(st.lists(st.integers()))
def test_all_equal_elements(lst):
    result = insertion_sort(lst)
    assert len(result) == len(lst)
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 10: Property-Based Testing"
echo "========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5

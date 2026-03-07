#!/bin/bash
# Exercises for Lesson 09: Property-Based Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Identifying Properties ===
# Problem: For each function below, identify testable properties
# and write property-based tests using Hypothesis.
exercise_1() {
    echo "=== Exercise 1: Identifying Properties ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Function 1: reverse a list
# Properties:
# - Reversing twice gives back original (involution)
# - Length is preserved
# - First element becomes last

@given(lst=st.lists(st.integers()))
def test_reverse_involution(lst):
    assert list(reversed(list(reversed(lst)))) == lst

@given(lst=st.lists(st.integers()))
def test_reverse_preserves_length(lst):
    assert len(list(reversed(lst))) == len(lst)

@given(lst=st.lists(st.integers(), min_size=1))
def test_reverse_swaps_first_and_last(lst):
    rev = list(reversed(lst))
    assert rev[0] == lst[-1]
    assert rev[-1] == lst[0]

# Function 2: abs() (absolute value)
# Properties:
# - Result is always >= 0
# - abs(abs(x)) == abs(x) (idempotent)
# - abs(x) == abs(-x) (symmetry)
# - abs(x) >= x for all x

@given(x=st.floats(allow_nan=False))
def test_abs_non_negative(x):
    assert abs(x) >= 0

@given(x=st.floats(allow_nan=False))
def test_abs_idempotent(x):
    assert abs(abs(x)) == abs(x)

@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_abs_symmetric(x):
    assert abs(x) == abs(-x)

# Function 3: set union
# Properties:
# - A | B contains all elements of A and all elements of B
# - A | B == B | A (commutative)
# - |A | B| <= |A| + |B|

@given(
    a=st.frozensets(st.integers()),
    b=st.frozensets(st.integers())
)
def test_union_contains_both(a, b):
    union = a | b
    assert a.issubset(union)
    assert b.issubset(union)

@given(
    a=st.frozensets(st.integers()),
    b=st.frozensets(st.integers())
)
def test_union_commutative(a, b):
    assert a | b == b | a
SOLUTION
}

# === Exercise 2: Custom Strategies ===
# Problem: Create custom Hypothesis strategies for domain-specific
# test data generation.
exercise_2() {
    echo "=== Exercise 2: Custom Strategies ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from hypothesis import given, assume
from hypothesis import strategies as st
from dataclasses import dataclass
from datetime import datetime, timedelta

# Strategy for valid ages
age_strategy = st.integers(min_value=0, max_value=150)

# Strategy for money amounts (cents precision)
money_strategy = st.decimals(
    min_value=0, max_value=999999.99,
    places=2, allow_nan=False, allow_infinity=False
)

# Strategy for coordinates
@st.composite
def coordinates(draw):
    """Custom composite strategy for (lat, lng) pairs."""
    lat = draw(st.floats(min_value=-90, max_value=90))
    lng = draw(st.floats(min_value=-180, max_value=180))
    return (lat, lng)

@given(coord=coordinates())
def test_coordinates_in_range(coord):
    lat, lng = coord
    assert -90 <= lat <= 90
    assert -180 <= lng <= 180

# Strategy for a valid user
@st.composite
def valid_user(draw):
    """Generate realistic user data."""
    name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",)),
        min_size=1, max_size=50
    ))
    age = draw(st.integers(min_value=13, max_value=120))
    email = draw(st.emails())
    return {"name": name, "age": age, "email": email}

@given(user=valid_user())
def test_valid_user_has_required_fields(user):
    assert "name" in user
    assert "age" in user
    assert "email" in user
    assert user["age"] >= 13

# Strategy for date ranges
@st.composite
def date_range(draw, min_days=1, max_days=365):
    """Generate a (start, end) date pair where start < end."""
    start = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    ))
    gap = draw(st.integers(min_value=min_days, max_value=max_days))
    end = start + timedelta(days=gap)
    return (start, end)

@given(dates=date_range())
def test_date_range_ordered(dates):
    start, end = dates
    assert start < end
SOLUTION
}

# === Exercise 3: Finding Bugs with Properties ===
# Problem: Use property-based testing to find a bug in a
# "unique elements" function.
exercise_3() {
    echo "=== Exercise 3: Finding Bugs with Properties ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Buggy implementation
def unique_elements_buggy(lst):
    """Return unique elements preserving order. BUG: fails for certain inputs."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

# This implementation actually works! Let's make a subtly buggy version:
def unique_case_insensitive_buggy(strings):
    """Unique strings, case-insensitive. BUG: doesn't preserve original case."""
    seen = set()
    result = []
    for s in strings:
        lower = s.lower()
        if lower not in seen:
            result.append(lower)  # BUG: should append 's', not 'lower'
            seen.add(lower)
    return result

# Property test that catches the bug:
@given(lst=st.lists(st.text(min_size=1, max_size=5)))
def test_unique_preserves_original_values(lst):
    """Property: every element in the result must exist in the original list."""
    result = unique_case_insensitive_buggy(lst)
    for item in result:
        # This assertion will FAIL when lst contains uppercase strings
        # because the buggy function lowercases them
        assert item in lst  # Catches the bug!

# Fixed implementation:
def unique_case_insensitive_fixed(strings):
    """Unique strings, case-insensitive. Preserves original case."""
    seen = set()
    result = []
    for s in strings:
        lower = s.lower()
        if lower not in seen:
            result.append(s)  # FIX: preserve original case
            seen.add(lower)
    return result

@given(lst=st.lists(st.text(min_size=1, max_size=5)))
def test_fixed_preserves_originals(lst):
    result = unique_case_insensitive_fixed(lst)
    for item in result:
        assert item in lst  # Now passes!

@given(lst=st.lists(st.text(min_size=1, max_size=5)))
def test_fixed_no_case_duplicates(lst):
    """Property: no two results should be equal when lowercased."""
    result = unique_case_insensitive_fixed(lst)
    lowers = [s.lower() for s in result]
    assert len(lowers) == len(set(lowers))
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 09: Property-Based Testing"
echo "==========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3

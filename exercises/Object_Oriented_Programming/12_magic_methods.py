"""
Exercise 12: Magic Methods
Topic: Object-Oriented Programming

Implement classes with various magic methods.
"""


class Vector:
    """A 2D vector with full operator support.

    Operators:
        v1 + v2 -> Vector (component-wise addition)
        v1 - v2 -> Vector (component-wise subtraction)
        v * scalar -> Vector (scalar multiplication)
        scalar * v -> Vector (reverse multiplication)
        -v -> Vector (negation)
        abs(v) -> float (magnitude)
        v1 == v2 -> bool (same components)
        hash(v) -> int (hashable)
        v1 @ v2 -> float (dot product via __matmul__)
        len(v) -> 2 (always)
        v[0] -> x, v[1] -> y (indexing)
        bool(v) -> False if zero vector
        repr(v) -> "Vector(x, y)"
        str(v) -> "(x, y)"
    """

    # TODO: Implement this class
    pass


class SortedSet:
    """A sorted collection of unique elements.

    Supports:
        len(s) -> number of elements
        x in s -> membership test
        s[i] -> element at index
        iter(s) -> iterate in sorted order
        s1 | s2 -> union (new SortedSet)
        s1 & s2 -> intersection (new SortedSet)
        repr(s) -> "SortedSet([1, 2, 3])"
        bool(s) -> False if empty

    Methods:
        add(item): Add item (maintains sorted order, ignores duplicates).
        remove(item): Remove item. Raise KeyError if not found.
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test Vector
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)

    assert v1 + v2 == Vector(4, 6)
    assert v1 - v2 == Vector(2, 2)
    assert v1 * 3 == Vector(9, 12)
    assert 3 * v1 == Vector(9, 12)
    assert -v1 == Vector(-3, -4)
    assert abs(v1) == 5.0
    assert (v1 @ v2) == 11  # dot product: 3*1 + 4*2
    assert len(v1) == 2
    assert v1[0] == 3
    assert v1[1] == 4
    assert bool(v1) is True
    assert bool(Vector(0, 0)) is False
    assert str(v1) == "(3, 4)"
    assert repr(v1) == "Vector(3, 4)"

    # Hashable
    vectors = {v1, v2, Vector(3, 4)}
    assert len(vectors) == 2

    print(f"v1 = {v1}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"|v1| = {abs(v1)}")
    print(f"v1 @ v2 = {v1 @ v2}")

    # Test SortedSet
    s1 = SortedSet([3, 1, 4, 1, 5, 9])
    assert list(s1) == [1, 3, 4, 5, 9]
    assert len(s1) == 5
    assert 4 in s1
    assert 2 not in s1
    assert s1[0] == 1
    assert s1[-1] == 9

    s2 = SortedSet([2, 4, 6, 8])

    union = s1 | s2
    assert list(union) == [1, 2, 3, 4, 5, 6, 8, 9]

    inter = s1 & s2
    assert list(inter) == [4]

    s1.add(2)
    assert 2 in s1
    s1.remove(2)
    assert 2 not in s1

    try:
        s1.remove(999)
        assert False, "Should raise KeyError"
    except KeyError:
        pass

    assert bool(SortedSet()) is False
    assert bool(s1) is True

    print(f"\ns1 = {s1}")
    print(f"s2 = {s2}")
    print(f"s1 | s2 = {union}")
    print(f"s1 & s2 = {inter}")

    print("\nAll tests passed!")

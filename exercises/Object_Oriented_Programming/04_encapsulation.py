"""
Exercise 04: Encapsulation
Topic: Object-Oriented Programming

Practice @property, validation, and protecting internal state.
"""


class PasswordManager:
    """A password manager with encapsulated storage.

    The password is stored internally but NEVER exposed directly.

    Methods:
        set_password(password): Set password. Must be >= 8 chars, contain
                                at least one digit and one uppercase letter.
                                Raise ValueError if invalid.
        check_password(password): Return True if password matches.
        password_strength(): Return "weak", "medium", or "strong":
            - weak: 8-11 chars
            - medium: 12-15 chars
            - strong: 16+ chars

    The stored password should NOT be accessible via any public attribute.
    """

    # TODO: Implement this class
    pass


class BoundedList:
    """A list with a maximum capacity.

    Properties:
        capacity (int, read-only): Maximum number of items.
        items (list, read-only): Return a COPY of the internal list.
        is_full (bool, read-only): True if at capacity.

    Methods:
        append(item): Add item. Raise OverflowError if full.
        pop(): Remove and return last item. Raise IndexError if empty.
        __len__(): Return number of items.
        __repr__(): Return "BoundedList(N/capacity)"
    """

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test PasswordManager
    pm = PasswordManager()

    try:
        pm.set_password("short")
        assert False, "Should reject short password"
    except ValueError:
        pass

    try:
        pm.set_password("alllowercase1")
        assert False, "Should reject no uppercase"
    except ValueError:
        pass

    try:
        pm.set_password("NoDigitsHere")
        assert False, "Should reject no digits"
    except ValueError:
        pass

    pm.set_password("GoodPass1")
    assert pm.check_password("GoodPass1") is True
    assert pm.check_password("wrong") is False
    assert pm.password_strength() == "weak"

    pm.set_password("VeryLongPassword123")
    assert pm.password_strength() == "strong"

    # Ensure password is not directly accessible
    assert not hasattr(pm, "password")
    print("PasswordManager: all checks passed")

    # Test BoundedList
    bl = BoundedList(3)
    bl.append("a")
    bl.append("b")
    bl.append("c")

    assert bl.is_full is True
    assert len(bl) == 3

    try:
        bl.append("d")
        assert False, "Should raise OverflowError"
    except OverflowError:
        pass

    # items should return a copy
    items = bl.items
    items.append("sneaky")
    assert len(bl) == 3  # Internal state unchanged

    val = bl.pop()
    assert val == "c"
    assert len(bl) == 2
    assert bl.is_full is False

    print(f"BoundedList: {bl}")

    print("\nAll tests passed!")

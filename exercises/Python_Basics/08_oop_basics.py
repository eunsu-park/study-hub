"""
Exercise 08: OOP Basics

Implement basic classes with methods, properties, and special methods.
"""


class BankAccount:
    """A simple bank account.

    Attributes:
        owner (str): Account owner's name.
        balance (float): Current balance (starts at 0).

    Methods:
        deposit(amount): Add money. Raise ValueError if amount <= 0.
        withdraw(amount): Remove money. Raise ValueError if amount <= 0
                         or if insufficient funds.
        __str__(): Return "BankAccount({owner}: ${balance:.2f})"
    """

    # TODO: Implement __init__, deposit, withdraw, __str__
    pass


class Rectangle:
    """A rectangle with width and height.

    Properties:
        area (read-only): width * height
        perimeter (read-only): 2 * (width + height)
        is_square (read-only): True if width == height

    Methods:
        scale(factor): Multiply both dimensions by factor.
        __eq__(other): Two rectangles are equal if same width and height.
        __repr__(): Return "Rectangle(width, height)"
    """

    # TODO: Implement with @property decorators
    pass


class Student:
    """A student with grades.

    Attributes:
        name (str): Student name.
        grades (list): List of numeric grades.

    Methods:
        add_grade(grade): Add a grade (0-100). Raise ValueError if invalid.
        average(): Return average grade, or 0.0 if no grades.
        highest(): Return highest grade, or 0 if no grades.
        passing(threshold=60): Return True if average >= threshold.
        __repr__(): Return "Student({name}, avg={average:.1f})"
    """

    # TODO: Implement this class
    pass


class Counter:
    """A simple counter with increment, decrement, and reset.

    Attributes:
        value (int): Current count, starts at given initial value.

    Methods:
        increment(n=1): Add n to counter.
        decrement(n=1): Subtract n from counter. Value should not go below 0.
        reset(): Reset to initial value.
        __int__(): Return current value as int.
        __add__(other): Return new Counter with sum of values.
    """

    # TODO: Implement this class
    pass


# === Tests ===

# BankAccount tests
acc = BankAccount("Alice")
assert acc.balance == 0, "Initial balance"
acc.deposit(100)
assert acc.balance == 100, "After deposit"
acc.withdraw(30)
assert acc.balance == 70, "After withdraw"
assert str(acc) == "BankAccount(Alice: $70.00)", "String repr"
try:
    acc.withdraw(200)
    assert False, "Should raise ValueError for insufficient funds"
except ValueError:
    pass
try:
    acc.deposit(-10)
    assert False, "Should raise ValueError for negative deposit"
except ValueError:
    pass

# Rectangle tests
r = Rectangle(5, 3)
assert r.area == 15, "Area"
assert r.perimeter == 16, "Perimeter"
assert r.is_square is False, "Not square"
r.scale(2)
assert r.area == 60, "Scaled area"
assert Rectangle(4, 4).is_square is True, "Is square"
assert repr(Rectangle(3, 4)) == "Rectangle(3, 4)", "Repr"
assert Rectangle(3, 4) == Rectangle(3, 4), "Equality"

# Student tests
s = Student("Bob")
s.add_grade(90)
s.add_grade(80)
s.add_grade(70)
assert s.average() == 80.0, "Average"
assert s.highest() == 90, "Highest"
assert s.passing() is True, "Passing"
assert s.passing(threshold=85) is False, "Not passing at 85"
assert repr(s) == "Student(Bob, avg=80.0)", "Repr"

# Counter tests
c = Counter(10)
c.increment()
assert int(c) == 11, "Increment"
c.decrement(5)
assert int(c) == 6, "Decrement"
c.reset()
assert int(c) == 10, "Reset"
c.decrement(100)
assert int(c) == 0, "Floor at 0"

print("All tests passed!")

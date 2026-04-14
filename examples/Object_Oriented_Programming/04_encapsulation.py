"""
Example 04: Encapsulation
Topic: Object-Oriented Programming

Demonstrates access control conventions, @property decorator,
enforcing invariants, and common encapsulation patterns.
"""

# =============================================================================
# ACCESS CONTROL CONVENTIONS
# =============================================================================

class BankAccount:
    """Bank account demonstrating Python's access control levels."""

    def __init__(self, owner, balance=0.0, pin="0000"):
        self.owner = owner           # Public
        self._balance = balance      # Protected (convention)
        self.__pin = pin             # Private (name-mangled)
        self._transactions = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._transactions.append(("deposit", amount))

    def withdraw(self, amount, pin):
        if pin != self.__pin:
            raise PermissionError("Invalid PIN")
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._transactions.append(("withdraw", amount))

    @property
    def balance(self):
        """Read-only balance."""
        return self._balance

    @property
    def transactions(self):
        """Return a copy to prevent external modification."""
        return list(self._transactions)

    def __repr__(self):
        return f"BankAccount({self.owner!r}, ${self._balance:.2f})"


# =============================================================================
# @PROPERTY WITH VALIDATION
# =============================================================================

class Circle:
    """Circle with property-based validation."""

    def __init__(self, radius):
        self.radius = radius  # Triggers setter!

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Radius must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"Radius must be positive, got {value}")
        self._radius = value

    @property
    def diameter(self):
        """Computed read-only property."""
        return self._radius * 2

    @property
    def area(self):
        """Computed read-only property."""
        from math import pi
        return pi * self._radius ** 2

    @property
    def circumference(self):
        from math import pi
        return 2 * pi * self._radius

    def __repr__(self):
        return f"Circle(radius={self._radius})"


# =============================================================================
# ENFORCING INVARIANTS
# =============================================================================

class DateRange:
    """Date range where start must always be <= end."""

    def __init__(self, start, end):
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if value > self._end:
            raise ValueError(f"start ({value}) must be <= end ({self._end})")
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if value < self._start:
            raise ValueError(f"end ({value}) must be >= start ({self._start})")
        self._end = value

    @property
    def duration(self):
        return self._end - self._start

    def __repr__(self):
        return f"DateRange({self._start} to {self._end})"


if __name__ == "__main__":
    # Access control
    print("=== Access Control ===")
    acct = BankAccount("Alice", 1000, "1234")
    acct.deposit(500)
    acct.withdraw(200, "1234")
    print(f"Balance: ${acct.balance:.2f}")
    print(f"Transactions: {acct.transactions}")

    # Name mangling demo
    print(f"\nName-mangled PIN access: {acct._BankAccount__pin}")

    # Property validation
    print("\n=== Property Validation ===")
    c = Circle(5)
    print(f"{c}: area={c.area:.2f}, circumference={c.circumference:.2f}")
    c.radius = 10
    print(f"After resize: {c}, area={c.area:.2f}")

    try:
        c.radius = -1
    except ValueError as e:
        print(f"Validation: {e}")

    # Invariant enforcement
    print("\n=== Invariant Enforcement ===")
    from datetime import date
    dr = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    print(f"{dr}, duration: {dr.duration.days} days")

    dr.end = date(2025, 6, 30)
    print(f"Extended: {dr}, duration: {dr.duration.days} days")

    try:
        dr.start = date(2026, 1, 1)
    except ValueError as e:
        print(f"Invariant violation: {e}")

"""
08 OOP Basics
=============
Demonstrates class definition, instance/class attributes, properties,
class methods, static methods, and encapsulation conventions.
"""


class BankAccount:
    """A simple bank account demonstrating core OOP concepts."""

    # Class attribute: shared across all instances
    bank_name = "Python National Bank"
    _total_accounts = 0

    def __init__(self, owner, balance=0.0):
        """Initialize a new account.

        Args:
            owner: Account holder name.
            balance: Initial balance (default 0).
        """
        self.owner = owner            # Public attribute
        self._balance = balance       # Protected by convention (single _)
        self.__account_id = BankAccount._total_accounts  # Name-mangled (double __)
        BankAccount._total_accounts += 1

    @property
    def balance(self):
        """Read-only access to balance."""
        return self._balance

    @property
    def account_id(self):
        """Read-only account identifier."""
        return self.__account_id

    def deposit(self, amount):
        """Add funds to the account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        return self._balance

    def withdraw(self, amount):
        """Remove funds from the account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return self._balance

    @classmethod
    def get_total_accounts(cls):
        """Class method: access class-level data."""
        return cls._total_accounts

    @classmethod
    def from_string(cls, account_string):
        """Alternative constructor from 'owner:balance' string."""
        owner, balance = account_string.split(":")
        return cls(owner, float(balance))

    @staticmethod
    def validate_amount(amount):
        """Static method: utility that doesn't need instance or class."""
        return isinstance(amount, (int, float)) and amount > 0

    def __str__(self):
        """Human-readable string representation."""
        return f"Account({self.owner}, ${self._balance:,.2f})"

    def __repr__(self):
        """Developer-friendly representation."""
        return f"BankAccount(owner={self.owner!r}, balance={self._balance})"


def basic_class_demo():
    """Demonstrate basic class usage."""
    # Creating instances
    acc1 = BankAccount("Alice", 1000)
    acc2 = BankAccount("Bob")

    print(f"acc1: {acc1}")
    print(f"acc2: {acc2}")
    print(f"repr: {acc1!r}")

    # Instance methods
    acc1.deposit(500)
    print(f"\nAfter deposit: {acc1}")
    acc1.withdraw(200)
    print(f"After withdrawal: {acc1}")

    # Property access (read-only)
    print(f"\nBalance: ${acc1.balance:,.2f}")
    print(f"Account ID: {acc1.account_id}")


def class_and_static_methods():
    """Demonstrate @classmethod and @staticmethod."""
    # Class method: access class data
    acc1 = BankAccount("Test1")
    acc2 = BankAccount("Test2")
    print(f"Total accounts created: {BankAccount.get_total_accounts()}")

    # Class method as alternative constructor
    acc3 = BankAccount.from_string("Charlie:5000")
    print(f"From string: {acc3}")

    # Static method: no self or cls needed
    print(f"\nValidate 100:   {BankAccount.validate_amount(100)}")
    print(f"Validate -50:   {BankAccount.validate_amount(-50)}")
    print(f"Validate 'abc': {BankAccount.validate_amount('abc')}")


def encapsulation_demo():
    """Show Python's naming conventions for access control."""

    class Person:
        def __init__(self, name, age, ssn):
            self.name = name          # Public
            self._age = age           # Protected (convention)
            self.__ssn = ssn          # Name-mangled -> _Person__ssn

        @property
        def age(self):
            return self._age

        @age.setter
        def age(self, value):
            if not isinstance(value, int) or value < 0:
                raise ValueError("Age must be a non-negative integer")
            self._age = value

        def get_masked_ssn(self):
            return f"***-**-{self.__ssn[-4:]}"

    p = Person("Alice", 30, "123-45-6789")

    # Public access
    print(f"Name: {p.name}")

    # Property with validation
    print(f"Age:  {p.age}")
    p.age = 31
    print(f"Updated age: {p.age}")

    try:
        p.age = -5
    except ValueError as e:
        print(f"Validation error: {e}")

    # Name-mangled attribute
    print(f"\nMasked SSN: {p.get_masked_ssn()}")
    # Direct access blocked: p.__ssn raises AttributeError
    # But name mangling allows: p._Person__ssn (not recommended)
    print(f"Name-mangled access: {p._Person__ssn}")

    # Show all attributes
    print(f"\nAttributes: {[a for a in dir(p) if not a.startswith('__')]}")


def property_patterns():
    """Advanced property usage patterns."""

    class Temperature:
        """Temperature with Celsius as internal storage."""

        def __init__(self, celsius=0.0):
            self._celsius = celsius

        @property
        def celsius(self):
            return self._celsius

        @celsius.setter
        def celsius(self, value):
            if value < -273.15:
                raise ValueError("Below absolute zero!")
            self._celsius = value

        @property
        def fahrenheit(self):
            """Computed property (read/write)."""
            return self._celsius * 9 / 5 + 32

        @fahrenheit.setter
        def fahrenheit(self, value):
            self.celsius = (value - 32) * 5 / 9

        @property
        def kelvin(self):
            return self._celsius + 273.15

        def __repr__(self):
            return f"Temperature({self._celsius:.1f}C)"

    t = Temperature(100)
    print(f"Boiling: {t.celsius}C = {t.fahrenheit}F = {t.kelvin}K")

    t.fahrenheit = 72
    print(f"Room temp: {t.celsius:.1f}C = {t.fahrenheit:.1f}F")

    t.celsius = -40
    print(f"Same in both: {t.celsius}C = {t.fahrenheit}F")


def special_methods_intro():
    """Common dunder methods for customizing behavior."""

    class Vector:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __repr__(self):
            return f"Vector({self.x}, {self.y})"

        def __str__(self):
            return f"({self.x}, {self.y})"

        def __len__(self):
            """Enable len() — returns integer magnitude."""
            return int((self.x ** 2 + self.y ** 2) ** 0.5)

        def __bool__(self):
            """Enable bool() — False for zero vector."""
            return self.x != 0 or self.y != 0

        def __eq__(self, other):
            if not isinstance(other, Vector):
                return NotImplemented
            return self.x == other.x and self.y == other.y

        def __hash__(self):
            return hash((self.x, self.y))

    v1 = Vector(3, 4)
    v2 = Vector(3, 4)
    v0 = Vector(0, 0)

    print(f"str:   {v1}")
    print(f"repr:  {v1!r}")
    print(f"len:   {len(v1)}")
    print(f"bool:  {bool(v1)}, {bool(v0)}")
    print(f"equal: {v1 == v2}")
    print(f"hash:  {hash(v1)} == {hash(v2)}: {hash(v1) == hash(v2)}")

    # Usable in sets and as dict keys thanks to __hash__ and __eq__
    vectors = {v1, v2, Vector(1, 1)}
    print(f"Set:   {vectors}")


if __name__ == "__main__":
    sections = [
        ("Basic Class Demo", basic_class_demo),
        ("Class & Static Methods", class_and_static_methods),
        ("Encapsulation", encapsulation_demo),
        ("Property Patterns", property_patterns),
        ("Special Methods Intro", special_methods_intro),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()

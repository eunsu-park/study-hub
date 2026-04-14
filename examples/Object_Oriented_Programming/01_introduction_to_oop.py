"""
Example 01: Introduction to OOP
Topic: Object-Oriented Programming

Demonstrates the difference between procedural and object-oriented approaches
to modeling a bank account system.
"""


# =============================================================================
# PROCEDURAL APPROACH
# =============================================================================

def procedural_bank_account():
    """Bank account modeled with procedural programming."""
    print("=== Procedural Approach ===")

    # Data is just variables
    account_owner = "Alice"
    account_balance = 1000.0

    def deposit(balance, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return balance + amount

    def withdraw(balance, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > balance:
            raise ValueError("Insufficient funds")
        return balance - amount

    def get_info(owner, balance):
        return f"Account({owner}: ${balance:.2f})"

    # Usage: must pass data explicitly, no protection
    account_balance = deposit(account_balance, 500)
    account_balance = withdraw(account_balance, 200)
    print(get_info(account_owner, account_balance))

    # Problem: nothing prevents this
    account_balance = -9999  # No protection!
    print(f"After direct modification: ${account_balance}")


# =============================================================================
# OBJECT-ORIENTED APPROACH
# =============================================================================

class BankAccount:
    """Bank account modeled with OOP.

    Data and behavior are bundled together, with validation
    and encapsulation protecting the internal state.
    """

    def __init__(self, owner, balance=0.0):
        """Initialize account with owner name and optional starting balance."""
        self.owner = owner
        self._balance = balance
        self._transactions = []

    def deposit(self, amount):
        """Deposit money into the account."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._transactions.append(("deposit", amount))
        return self._balance

    def withdraw(self, amount):
        """Withdraw money from the account."""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._transactions.append(("withdraw", amount))
        return self._balance

    @property
    def balance(self):
        """Read-only access to balance."""
        return self._balance

    def get_statement(self):
        """Return transaction history."""
        lines = [f"Statement for {self.owner}:"]
        for action, amount in self._transactions:
            sign = "+" if action == "deposit" else "-"
            lines.append(f"  {sign}${amount:.2f}")
        lines.append(f"  Balance: ${self._balance:.2f}")
        return "\n".join(lines)

    def __repr__(self):
        return f"BankAccount({self.owner!r}, balance={self._balance:.2f})"


def oop_bank_account():
    """Bank account modeled with OOP."""
    print("\n=== Object-Oriented Approach ===")

    alice = BankAccount("Alice", 1000.0)
    alice.deposit(500)
    alice.withdraw(200)

    print(repr(alice))
    print(alice.get_statement())

    # Multiple accounts are independent
    bob = BankAccount("Bob", 500.0)
    bob.deposit(100)
    print(f"\n{repr(bob)}")

    # Type checking
    print(f"\nalice is BankAccount? {isinstance(alice, BankAccount)}")


# =============================================================================
# FIRST CLASS: Dog
# =============================================================================

class Dog:
    """A simple Dog class demonstrating OOP basics."""

    species = "Canis familiaris"  # Class attribute

    def __init__(self, name, age, breed):
        self.name = name
        self.age = age
        self.breed = breed

    def bark(self):
        return f"{self.name} says: Woof!"

    def describe(self):
        return f"{self.name} is a {self.age}-year-old {self.breed}"

    def birthday(self):
        self.age += 1
        return f"Happy birthday, {self.name}! Now {self.age} years old."


def demo_dog_class():
    """Demonstrate the Dog class."""
    print("\n=== Dog Class Demo ===")

    rex = Dog("Rex", 3, "German Shepherd")
    bella = Dog("Bella", 5, "Golden Retriever")

    print(rex.bark())
    print(bella.describe())
    print(rex.birthday())

    # Class attribute is shared
    print(f"Rex's species: {rex.species}")
    print(f"Bella's species: {bella.species}")
    print(f"Same species object? {rex.species is bella.species}")


if __name__ == "__main__":
    procedural_bank_account()
    oop_bank_account()
    demo_dog_class()

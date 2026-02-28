"""
Exercises for Lesson 16: OOP Basics
Topic: Python

Solutions to practice problems from the lesson.
"""

import math
import uuid
import csv
import json
import os
import tempfile
from abc import ABC, abstractmethod


# === Exercise 1: Class Fundamentals ===
# Problem: Build a BankAccount class with deposit, withdraw, apply_interest,
#          class method from_dict, static method is_valid_amount.

class BankAccount:
    """Bank account with interest, validation, and factory method.

    Demonstrates: instance vs class variables, class methods,
    static methods, and validation in mutating methods.
    """
    interest_rate: float = 0.02  # Class variable shared by all instances

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if not self.is_valid_amount(amount):
            raise ValueError(f"Deposit amount must be positive, got {amount}")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if not self.is_valid_amount(amount):
            raise ValueError(f"Withdrawal amount must be positive, got {amount}")
        if amount > self.balance:
            raise ValueError(
                f"Insufficient funds: balance={self.balance:.2f}, requested={amount:.2f}"
            )
        self.balance -= amount

    def apply_interest(self) -> None:
        """Apply the class-level interest rate to the current balance."""
        self.balance *= (1 + self.interest_rate)

    @classmethod
    def from_dict(cls, data: dict) -> "BankAccount":
        """Factory method: create a BankAccount from a dictionary."""
        return cls(owner=data["owner"], balance=data.get("balance", 0.0))

    @staticmethod
    def is_valid_amount(amount: float) -> bool:
        """Check if an amount is positive."""
        return amount > 0

    def __str__(self) -> str:
        return f"BankAccount({self.owner}, balance=${self.balance:.2f})"


def exercise_1():
    """Demonstrate BankAccount class."""
    # Create accounts
    alice = BankAccount("Alice", 1000.0)
    bob = BankAccount.from_dict({"owner": "Bob", "balance": 500.0})
    print(f"  {alice}")
    print(f"  {bob}")

    # Operations
    alice.deposit(200)
    print(f"  After deposit: {alice}")

    alice.withdraw(150)
    print(f"  After withdraw: {alice}")

    alice.apply_interest()
    print(f"  After interest ({BankAccount.interest_rate:.0%}): {alice}")

    # Validation
    try:
        alice.withdraw(100000)
    except ValueError as e:
        print(f"  ValueError: {e}")

    try:
        alice.deposit(-50)
    except ValueError as e:
        print(f"  ValueError: {e}")

    # Class variable change affects all instances
    BankAccount.interest_rate = 0.03
    bob.apply_interest()
    print(f"  Bob after 3% interest: {bob}")
    BankAccount.interest_rate = 0.02  # Reset


# === Exercise 2: Inheritance Hierarchy ===
# Problem: Build Employee -> Manager, Contractor hierarchy.

class Employee:
    """Base employee with name, ID, and salary."""

    def __init__(self, name: str, emp_id: str, base_salary: float):
        self.name = name
        self.emp_id = emp_id
        self.base_salary = base_salary

    def calculate_pay(self) -> float:
        return self.base_salary

    def __str__(self) -> str:
        return f"Employee {self.name} ({self.emp_id}): ${self.calculate_pay():,.2f}/month"


class Manager(Employee):
    """Manager with a team bonus ($500 per team member)."""

    def __init__(self, name: str, emp_id: str, base_salary: float, team_size: int):
        super().__init__(name, emp_id, base_salary)
        self.team_size = team_size

    def calculate_pay(self) -> float:
        # Base salary + $500 bonus per team member
        return self.base_salary + (500 * self.team_size)

    def __str__(self) -> str:
        return (f"Manager {self.name} ({self.emp_id}): "
                f"${self.calculate_pay():,.2f}/month (team of {self.team_size})")


class Contractor(Employee):
    """Contractor paid by the hour, ignoring base_salary."""

    def __init__(self, name: str, emp_id: str, hourly_rate: float, hours_worked: float):
        # Pass 0 as base_salary since contractors are paid hourly
        super().__init__(name, emp_id, 0)
        self.hourly_rate = hourly_rate
        self.hours_worked = hours_worked

    def calculate_pay(self) -> float:
        return self.hourly_rate * self.hours_worked

    def __str__(self) -> str:
        return (f"Contractor {self.name} ({self.emp_id}): "
                f"${self.calculate_pay():,.2f} ({self.hours_worked}h @ ${self.hourly_rate}/h)")


def payroll_report(employees: list[Employee]) -> None:
    """Print a formatted pay summary for a mix of employee types.

    Polymorphism in action: calculate_pay() dispatches to the correct
    implementation regardless of the actual subclass.
    """
    total = 0.0
    print(f"  {'Name':<12} {'Type':<12} {'Pay':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for emp in employees:
        pay = emp.calculate_pay()
        total += pay
        emp_type = type(emp).__name__
        print(f"  {emp.name:<12} {emp_type:<12} ${pay:>10,.2f}")
    print(f"  {'':12} {'TOTAL':<12} ${total:>10,.2f}")


def exercise_2():
    """Demonstrate inheritance hierarchy."""
    employees = [
        Employee("Alice", "E001", 3000),
        Manager("Bob", "M001", 4000, team_size=5),
        Contractor("Charlie", "C001", hourly_rate=75, hours_worked=160),
        Manager("Diana", "M002", 3500, team_size=3),
    ]

    for emp in employees:
        print(f"  {emp}")

    print()
    payroll_report(employees)

    # Verify hierarchy
    print(f"\n  isinstance(Manager(), Employee): {isinstance(employees[1], Employee)}")
    print(f"  issubclass(Manager, Employee): {issubclass(Manager, Employee)}")
    print(f"  Manager MRO: {[c.__name__ for c in Manager.__mro__]}")


# === Exercise 3: Encapsulation with Properties ===
# Problem: Circle and Rectangle with property validation.

class Circle:
    """Circle with validated radius and computed area/circumference."""

    def __init__(self, radius: float):
        self.__id = str(uuid.uuid4())  # Private via name mangling
        self.radius = radius  # Goes through the setter for validation

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        if value < 0:
            raise ValueError(f"Radius cannot be negative, got {value}")
        self._radius = value

    @property
    def area(self) -> float:
        """Read-only computed property."""
        return math.pi * self._radius ** 2

    @property
    def circumference(self) -> float:
        """Read-only computed property."""
        return 2 * math.pi * self._radius

    def scale(self, factor: float):
        """Scale the radius by a factor. Validation triggers if factor < 0."""
        self.radius = self._radius * factor

    def __repr__(self) -> str:
        return f"Circle(radius={self._radius})"


class Rectangle:
    """Rectangle with validated width/height and computed aspect ratio."""

    def __init__(self, width: float, height: float):
        self.__id = str(uuid.uuid4())
        self.width = width
        self.height = height

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float):
        if value <= 0:
            raise ValueError(f"Width must be positive, got {value}")
        self._width = value

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        if value <= 0:
            raise ValueError(f"Height must be positive, got {value}")
        self._height = value

    @property
    def aspect_ratio(self) -> float:
        return self._width / self._height

    def scale(self, factor: float):
        self.width = self._width * factor
        self.height = self._height * factor

    def __repr__(self) -> str:
        return f"Rectangle(width={self._width}, height={self._height})"


def exercise_3():
    """Demonstrate encapsulation with properties."""
    c = Circle(5)
    print(f"  {c}: area={c.area:.2f}, circumference={c.circumference:.2f}")

    c.scale(2)
    print(f"  After scale(2): {c}, area={c.area:.2f}")

    # Validation in setter
    try:
        c.radius = -1
    except ValueError as e:
        print(f"  ValueError: {e}")

    r = Rectangle(16, 9)
    print(f"\n  {r}: aspect_ratio={r.aspect_ratio:.4f}")

    # Name mangling demonstration
    try:
        _ = c.__id  # This will fail
    except AttributeError:
        # Access via mangled name _ClassName__id
        print(f"\n  c.__id raises AttributeError (name mangling)")
        print(f"  c._Circle__id = {c._Circle__id[:8]}...")


# === Exercise 4: Magic Methods and Operator Overloading ===
# Problem: Implement a rich Fraction class.

class Fraction:
    """Fraction with arithmetic, comparison, and hashing.

    Always stored in reduced form using GCD. Negative fractions
    are normalized so the denominator is always positive.
    """

    def __init__(self, numerator: int, denominator: int):
        if denominator == 0:
            raise ZeroDivisionError("Fraction denominator cannot be zero")

        # Normalize sign: denominator is always positive
        if denominator < 0:
            numerator, denominator = -numerator, -denominator

        # Reduce to lowest terms
        g = math.gcd(abs(numerator), denominator)
        self.numerator = numerator // g
        self.denominator = denominator // g

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        return f"Fraction({self.numerator}, {self.denominator})"

    # Arithmetic operators -- all return new Fraction in reduced form
    def __add__(self, other: "Fraction") -> "Fraction":
        return Fraction(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __sub__(self, other: "Fraction") -> "Fraction":
        return Fraction(
            self.numerator * other.denominator - other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __mul__(self, other: "Fraction") -> "Fraction":
        return Fraction(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )

    def __truediv__(self, other: "Fraction") -> "Fraction":
        if other.numerator == 0:
            raise ZeroDivisionError("Cannot divide by zero fraction")
        return Fraction(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )

    # Comparison operators -- use cross-multiplication to avoid float imprecision
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fraction):
            return NotImplemented
        return self.numerator == other.numerator and self.denominator == other.denominator

    def __lt__(self, other: "Fraction") -> bool:
        return self.numerator * other.denominator < other.numerator * self.denominator

    def __le__(self, other: "Fraction") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Fraction") -> bool:
        return not self <= other

    def __ge__(self, other: "Fraction") -> bool:
        return not self < other

    # Unary operators
    def __abs__(self) -> "Fraction":
        return Fraction(abs(self.numerator), self.denominator)

    def __neg__(self) -> "Fraction":
        return Fraction(-self.numerator, self.denominator)

    def __float__(self) -> float:
        return self.numerator / self.denominator

    # Hashable -- required for use in sets and as dict keys
    def __hash__(self) -> int:
        # Since we always store in reduced form, equal fractions
        # produce the same (numerator, denominator) tuple
        return hash((self.numerator, self.denominator))


def exercise_4():
    """Demonstrate Fraction class with magic methods."""
    a = Fraction(1, 2)
    b = Fraction(1, 3)

    print(f"  a = {a}, b = {b}")
    print(f"  a + b = {a + b}")       # 5/6
    print(f"  a - b = {a - b}")       # 1/6
    print(f"  a * b = {a * b}")       # 1/6
    print(f"  a / b = {a / b}")       # 3/2

    print(f"\n  a < b: {a < b}")      # False
    print(f"  a > b: {a > b}")        # True
    print(f"  float(a): {float(a)}")  # 0.5
    print(f"  -a: {-a}")              # -1/2
    print(f"  abs(-a): {abs(-a)}")    # 1/2

    # Hashability: Fraction(1,2) and Fraction(2,4) are the same in a set
    s = {Fraction(1, 2), Fraction(2, 4), Fraction(3, 6)}
    print(f"\n  Set {{1/2, 2/4, 3/6}} has {len(s)} element(s): {{{', '.join(str(f) for f in s)}}}")

    # Auto-reduction
    print(f"  Fraction(6, 8) = {Fraction(6, 8)}")  # 3/4
    print(f"  repr: {repr(Fraction(6, 8))}")


# === Exercise 5: Abstract Base Classes and Polymorphism ===
# Problem: Design a plugin-style export system using ABCs.

class DataExporter(ABC):
    """Abstract base class for data exporters."""

    @abstractmethod
    def export(self, data: list, filepath: str) -> None:
        """Export data to a file."""
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Return the file extension for this exporter."""
        pass

    def validate(self, data) -> None:
        """Concrete method: validate that data is a list."""
        if not isinstance(data, list):
            raise TypeError(f"Data must be a list, got {type(data).__name__}")


class CSVExporter(DataExporter):
    """Export data (list of dicts) to CSV format."""

    def export(self, data: list, filepath: str) -> None:
        self.validate(data)
        if not data:
            return
        fieldnames = data[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def get_extension(self) -> str:
        return ".csv"


class JSONExporter(DataExporter):
    """Export data to JSON format with pretty printing."""

    def export(self, data: list, filepath: str) -> None:
        self.validate(data)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_extension(self) -> str:
        return ".json"


class MarkdownExporter:
    """Duck-typed exporter -- does NOT inherit from DataExporter.

    Demonstrates Python's duck typing: if it walks like a DataExporter
    and quacks like a DataExporter, it works as a DataExporter.
    """

    def export(self, data: list, filepath: str) -> None:
        if not data:
            return
        headers = list(data[0].keys())
        with open(filepath, "w") as f:
            # Header row
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join("---" for _ in headers) + " |\n")
            # Data rows
            for row in data:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

    def get_extension(self) -> str:
        return ".md"


def export_all(data: list, filepath_base: str, exporters: list) -> list[str]:
    """Export data using multiple exporters.

    Works with both ABC subclasses and duck-typed implementations
    because Python resolves method calls at runtime based on the
    actual object, not the declared type.
    """
    created_files = []
    for exporter in exporters:
        ext = exporter.get_extension()
        filepath = filepath_base + ext
        exporter.export(data, filepath)
        created_files.append(filepath)
    return created_files


def exercise_5():
    """Demonstrate ABCs and polymorphism."""
    data = [
        {"name": "Alice", "age": 30, "city": "Seoul"},
        {"name": "Bob", "age": 25, "city": "Tokyo"},
        {"name": "Charlie", "age": 35, "city": "London"},
    ]

    tmpdir = tempfile.mkdtemp()
    base_path = os.path.join(tmpdir, "export")

    exporters = [CSVExporter(), JSONExporter(), MarkdownExporter()]
    files = export_all(data, base_path, exporters)

    for filepath in files:
        print(f"  Created: {os.path.basename(filepath)}")
        with open(filepath) as f:
            content = f.read()
        # Show first 3 lines
        lines = content.strip().split("\n")
        for line in lines[:4]:
            print(f"    {line}")
        if len(lines) > 4:
            print(f"    ... ({len(lines) - 4} more lines)")
        print()

    # Check ABC vs duck typing
    print("  Type checking:")
    for exp in exporters:
        is_abc = isinstance(exp, DataExporter)
        name = type(exp).__name__
        print(f"    {name:<18} isinstance(DataExporter): {is_abc}")

    # Validate method only exists on ABC subclasses
    print("\n  validate() on ABC subclasses:")
    csv_exp = CSVExporter()
    csv_exp.validate(data)  # Works
    print("    CSVExporter.validate(list) -- OK")

    try:
        csv_exp.validate("not a list")
    except TypeError as e:
        print(f"    CSVExporter.validate(str) -- TypeError: {e}")

    # Clean up
    for f in files:
        os.remove(f)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    print("=== Exercise 1: Class Fundamentals ===")
    exercise_1()

    print("\n=== Exercise 2: Inheritance Hierarchy ===")
    exercise_2()

    print("\n=== Exercise 3: Encapsulation with Properties ===")
    exercise_3()

    print("\n=== Exercise 4: Magic Methods and Operator Overloading ===")
    exercise_4()

    print("\n=== Exercise 5: Abstract Base Classes and Polymorphism ===")
    exercise_5()

    print("\nAll exercises completed!")

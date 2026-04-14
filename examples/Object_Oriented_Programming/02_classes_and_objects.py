"""
Example 02: Classes and Objects
Topic: Object-Oriented Programming

Demonstrates class definition, instance creation, class vs instance attributes,
three method types, and the object lifecycle.
"""


# =============================================================================
# CLASS VS INSTANCE ATTRIBUTES
# =============================================================================

class Employee:
    """Demonstrates class vs instance attributes."""

    company = "Acme Corp"
    employee_count = 0

    def __init__(self, name, salary, department):
        self.name = name
        self.salary = salary
        self.department = department
        Employee.employee_count += 1
        self.employee_id = Employee.employee_count

    def __repr__(self):
        return f"Employee({self.name!r}, ${self.salary:,}, {self.department!r})"


def demo_attributes():
    """Demonstrate class vs instance attributes."""
    print("=== Class vs Instance Attributes ===")

    alice = Employee("Alice", 75000, "Engineering")
    bob = Employee("Bob", 80000, "Sales")

    print(f"Company (class attr): {Employee.company}")
    print(f"Alice's company: {alice.company}")
    print(f"Total employees: {Employee.employee_count}")
    print(f"Alice: {alice}")
    print(f"Bob: {bob}")

    # Instance attribute shadows class attribute
    alice.company = "Alice's Startup"
    print(f"\nAfter alice.company = 'Alice's Startup':")
    print(f"Alice's company: {alice.company}")
    print(f"Bob's company: {bob.company}")  # Unchanged
    print(f"Class company: {Employee.company}")  # Unchanged


# =============================================================================
# THREE METHOD TYPES
# =============================================================================

class Date:
    """Demonstrates instance, class, and static methods."""

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # Instance method
    def format(self, separator="-"):
        """Format the date as a string."""
        return f"{self.year}{separator}{self.month:02d}{separator}{self.day:02d}"

    # Class method (alternative constructor)
    @classmethod
    def from_string(cls, date_string):
        """Create a Date from 'YYYY-MM-DD' format."""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """Create a Date for today."""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)

    # Static method
    @staticmethod
    def is_leap_year(year):
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"


def demo_methods():
    """Demonstrate the three method types."""
    print("\n=== Three Method Types ===")

    # Instance method
    d1 = Date(2024, 3, 15)
    print(f"Instance: {d1.format()}")
    print(f"Custom format: {d1.format('/')}")

    # Class method (alternative constructor)
    d2 = Date.from_string("2024-12-25")
    print(f"From string: {d2}")

    d3 = Date.today()
    print(f"Today: {d3}")

    # Static method
    print(f"2024 is leap year? {Date.is_leap_year(2024)}")
    print(f"2023 is leap year? {Date.is_leap_year(2023)}")


# =============================================================================
# OBJECT LIFECYCLE
# =============================================================================

class Resource:
    """Demonstrates object lifecycle with __new__, __init__, __del__."""

    _count = 0

    def __new__(cls, name):
        print(f"  __new__: Allocating memory for '{name}'")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        print(f"  __init__: Initializing '{name}'")
        self.name = name
        Resource._count += 1

    def __del__(self):
        Resource._count -= 1
        print(f"  __del__: Destroying '{self.name}' (remaining: {Resource._count})")

    def __repr__(self):
        return f"Resource({self.name!r})"


def demo_lifecycle():
    """Demonstrate the object lifecycle."""
    print("\n=== Object Lifecycle ===")

    print("Creating r1:")
    r1 = Resource("alpha")
    print(f"  Count: {Resource._count}")

    print("\nCreating r2:")
    r2 = Resource("beta")
    print(f"  Count: {Resource._count}")

    print("\nDeleting r1:")
    del r1
    print(f"  Count: {Resource._count}")

    print("\nDeleting r2:")
    del r2


# =============================================================================
# PRACTICAL EXAMPLE: LIBRARY
# =============================================================================

class Book:
    """A book in the library."""

    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_checked_out = False
        self.borrower = None

    def check_out(self, borrower):
        if self.is_checked_out:
            raise RuntimeError(f"'{self.title}' is already checked out by {self.borrower}")
        self.is_checked_out = True
        self.borrower = borrower

    def return_book(self):
        self.is_checked_out = False
        self.borrower = None

    def __repr__(self):
        status = f"out:{self.borrower}" if self.is_checked_out else "available"
        return f"Book({self.title!r}, {status})"


class Library:
    """A library holding a collection of books."""

    def __init__(self, name):
        self.name = name
        self._books = []

    def add_book(self, book):
        self._books.append(book)

    def find_by_title(self, query):
        return [b for b in self._books if query.lower() in b.title.lower()]

    def available(self):
        return [b for b in self._books if not b.is_checked_out]

    def __len__(self):
        return len(self._books)

    def __repr__(self):
        return f"Library({self.name!r}, {len(self)} books)"


def demo_library():
    """Demonstrate the Library and Book classes."""
    print("\n=== Library System ===")

    lib = Library("City Library")
    lib.add_book(Book("Python Crash Course", "Eric Matthes", "978-1"))
    lib.add_book(Book("Clean Code", "Robert Martin", "978-2"))
    lib.add_book(Book("Design Patterns", "Gang of Four", "978-3"))

    print(lib)
    print(f"Available: {lib.available()}")

    book = lib.find_by_title("python")[0]
    book.check_out("Alice")
    print(f"\nAfter checkout: {book}")
    print(f"Available: {lib.available()}")

    book.return_book()
    print(f"\nAfter return: {book}")


if __name__ == "__main__":
    demo_attributes()
    demo_methods()
    demo_lifecycle()
    demo_library()

# Lesson 02: Classes and Objects

## Learning Objectives

By the end of this lesson, you will be able to:
1. Define classes with attributes and methods in Python
2. Distinguish between class attributes and instance attributes
3. Create and use object instances
4. Explain the relationship between a class and its instances
5. Use class methods and static methods appropriately
6. Understand the object lifecycle (creation, usage, destruction)
7. Apply namespace resolution rules for attribute lookups

## Classes: The Blueprint

A **class** is a blueprint that defines the structure and behavior of objects. Think of it as an architectural plan: the plan itself is not a house, but many houses can be built from the same plan.

```
┌────────────────────────────┐
│       CLASS: Car           │  <-- Blueprint (template)
├────────────────────────────┤
│  Attributes:               │
│    - make                  │
│    - model                 │
│    - year                  │
│    - mileage               │
├────────────────────────────┤
│  Methods:                  │
│    - start()               │
│    - drive(distance)       │
│    - stop()                │
│    - describe()            │
└─────────┬──────────────────┘
          │ instantiation
          ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Object #1   │  │  Object #2   │  │  Object #3   │
│  make: Toyota│  │  make: BMW   │  │  make: Ford  │
│  model: Camry│  │  model: X3   │  │  model: F150 │
│  year: 2022  │  │  year: 2023  │  │  year: 2021  │
│  mileage: 0  │  │  mileage: 0  │  │  mileage: 0  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Defining a Class

```python
class Car:
    """A class representing a car."""

    # Class attribute: shared by ALL instances
    wheel_count = 4

    def __init__(self, make, model, year):
        """Initialize a new Car instance.

        Args:
            make: Manufacturer name.
            model: Model name.
            year: Manufacturing year.
        """
        # Instance attributes: unique to EACH instance
        self.make = make
        self.model = model
        self.year = year
        self.mileage = 0.0
        self.is_running = False

    def start(self):
        """Start the car's engine."""
        if self.is_running:
            print(f"{self.make} {self.model} is already running.")
        else:
            self.is_running = True
            print(f"{self.make} {self.model} started.")

    def drive(self, distance):
        """Drive the car a certain distance.

        Args:
            distance: Distance to drive in miles.
        """
        if not self.is_running:
            print("Start the car first!")
            return
        if distance <= 0:
            raise ValueError("Distance must be positive")
        self.mileage += distance
        print(f"Drove {distance} miles. Total: {self.mileage}")

    def stop(self):
        """Stop the car's engine."""
        self.is_running = False
        print(f"{self.make} {self.model} stopped.")

    def describe(self):
        """Return a description of the car."""
        return f"{self.year} {self.make} {self.model} ({self.mileage:.0f} miles)"
```

## Objects: The Instances

An **object** is a specific instance created from a class. Each object has its own copy of instance attributes but shares class attributes.

```python
# Creating objects (instantiation)
car1 = Car("Toyota", "Camry", 2022)
car2 = Car("BMW", "X3", 2023)

# Each object is independent
car1.start()           # Toyota Camry started.
car1.drive(50)         # Drove 50 miles. Total: 50.0

car2.start()           # BMW X3 started.
car2.drive(30)         # Drove 30 miles. Total: 30.0

# Different mileage (instance attributes are independent)
print(car1.mileage)    # 50.0
print(car2.mileage)    # 30.0

# Same wheel count (class attribute is shared)
print(car1.wheel_count)  # 4
print(car2.wheel_count)  # 4
```

### Identity, Equality, and Type

Every object in Python has three fundamental properties:

```python
car1 = Car("Toyota", "Camry", 2022)
car2 = Car("Toyota", "Camry", 2022)

# Identity: unique ID in memory
print(id(car1))              # e.g., 140234567890
print(id(car2))              # different number
print(car1 is car2)          # False — different objects

# Type: which class it came from
print(type(car1))            # <class '__main__.Car'>
print(isinstance(car1, Car)) # True

# Equality: by default, objects are only equal to themselves
print(car1 == car2)          # False (default: same as `is`)
# (We'll customize equality in Lesson 12: Magic Methods)
```

## Class Attributes vs Instance Attributes

Understanding the difference is crucial:

```
┌──────────────────────────────────────────────────┐
│            Class: Employee                        │
│  ┌──────────────────────────────────┐            │
│  │ Class Attributes (shared)       │            │
│  │   company = "Acme Corp"         │            │
│  │   employee_count = 0            │            │
│  └──────────────────────────────────┘            │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ Instance #1 │  │ Instance #2 │               │
│  │ name: Alice │  │ name: Bob   │               │
│  │ salary: 75k │  │ salary: 80k │               │
│  │ dept: Eng   │  │ dept: Sales │               │
│  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────┘
```

```python
class Employee:
    """Demonstrates class vs instance attributes."""

    # Class attributes
    company = "Acme Corp"
    employee_count = 0

    def __init__(self, name, salary, department):
        # Instance attributes
        self.name = name
        self.salary = salary
        self.department = department

        # Modify class attribute via the class itself
        Employee.employee_count += 1

    def __del__(self):
        Employee.employee_count -= 1


# Class attributes are shared
alice = Employee("Alice", 75000, "Engineering")
bob = Employee("Bob", 80000, "Sales")

print(Employee.company)        # Acme Corp
print(alice.company)           # Acme Corp (looked up from class)
print(bob.company)             # Acme Corp
print(Employee.employee_count) # 2
```

### Attribute Lookup Order (Namespace Resolution)

When you access `obj.attr`, Python searches in this order:

```
1. Instance namespace  (obj.__dict__)
       │
       ▼ not found?
2. Class namespace     (type(obj).__dict__)
       │
       ▼ not found?
3. Parent class(es)    (via MRO)
       │
       ▼ not found?
4. AttributeError raised
```

```python
class Demo:
    x = "class attribute"

    def __init__(self):
        # No instance attribute 'x' set here
        pass

d = Demo()
print(d.x)            # "class attribute" (found in class namespace)

d.x = "instance attr"  # Creates instance attribute
print(d.x)            # "instance attr" (found in instance namespace)
print(Demo.x)         # "class attribute" (class namespace unchanged)

del d.x               # Remove instance attribute
print(d.x)            # "class attribute" (falls back to class)
```

## Methods: Instance, Class, and Static

Python supports three types of methods:

```python
class MathHelper:
    """Demonstrates the three method types."""

    precision = 2  # Class attribute

    def __init__(self, name):
        self.name = name  # Instance attribute

    # Instance method: has access to instance (self) and class
    def greet(self):
        return f"I'm {self.name}, precision={self.precision}"

    # Class method: has access to class (cls) but not instance
    @classmethod
    def set_precision(cls, value):
        cls.precision = value

    # Static method: no access to instance or class
    @staticmethod
    def add(a, b):
        return a + b
```

### When to Use Which

```
┌─────────────────────────────────────────────────────────┐
│  Method Type    │ Has self? │ Has cls? │ Use Case       │
├─────────────────┼──────────┼─────────┼────────────────┤
│ Instance method │   Yes    │  (via   │ Most methods:  │
│  def foo(self)  │          │  self)  │ operate on     │
│                 │          │         │ instance data  │
├─────────────────┼──────────┼─────────┼────────────────┤
│ Class method    │   No     │  Yes    │ Alternative    │
│  @classmethod   │          │         │ constructors,  │
│  def foo(cls)   │          │         │ class-wide ops │
├─────────────────┼──────────┼─────────┼────────────────┤
│ Static method   │   No     │  No     │ Utility funcs  │
│  @staticmethod  │          │         │ logically      │
│  def foo()      │          │         │ grouped w/class│
└─────────────────┴──────────┴─────────┴────────────────┘
```

### Class Methods as Alternative Constructors

A common pattern is using `@classmethod` to provide multiple ways to create objects:

```python
class Date:
    """A date class with multiple constructors."""

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

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

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"


# Multiple ways to create a Date
d1 = Date(2024, 3, 15)
d2 = Date.from_string("2024-03-15")
d3 = Date.today()
```

## The Object Lifecycle

```
    ┌────────────┐
    │  Class     │
    │  Definition│
    └──────┬─────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __new__()  │────▶│ Memory       │
    │ (allocate) │     │ Allocation   │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __init__() │────▶│ Initialize   │
    │ (initialize│     │ Attributes   │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐
    │  Object    │
    │  in use    │◄──── methods called, attributes accessed
    └──────┬─────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ No more    │────▶│ Reference    │
    │ references │     │ Count = 0    │
    └──────┬─────┘     └──────────────┘
           │
    ┌──────▼─────┐     ┌──────────────┐
    │ __del__()  │────▶│ Garbage      │
    │ (finalize) │     │ Collection   │
    └────────────┘     └──────────────┘
```

```python
class TrackedObject:
    """Demonstrates the object lifecycle."""

    def __new__(cls, name):
        """Called first: allocates memory."""
        print(f"1. __new__: Creating memory for '{name}'")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        """Called second: initializes the object."""
        print(f"2. __init__: Initializing '{name}'")
        self.name = name

    def __del__(self):
        """Called when the object is garbage collected."""
        print(f"3. __del__: Destroying '{self.name}'")


obj = TrackedObject("test")
# Output:
# 1. __new__: Creating memory for 'test'
# 2. __init__: Initializing 'test'

del obj
# Output:
# 3. __del__: Destroying 'test'
```

## Practical Example: Library System

```python
class Book:
    """A book in the library system."""

    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_checked_out = False
        self.borrower = None

    def check_out(self, borrower_name):
        """Check out the book to a borrower."""
        if self.is_checked_out:
            raise RuntimeError(f"'{self.title}' is already checked out")
        self.is_checked_out = True
        self.borrower = borrower_name

    def return_book(self):
        """Return the book to the library."""
        self.is_checked_out = False
        self.borrower = None

    def __repr__(self):
        status = f"(checked out by {self.borrower})" if self.is_checked_out else "(available)"
        return f"Book('{self.title}' by {self.author}) {status}"


class Library:
    """A library that holds a collection of books."""

    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        """Add a book to the library."""
        self.books.append(book)

    def find_by_title(self, title):
        """Find books whose title contains the search term."""
        return [b for b in self.books if title.lower() in b.title.lower()]

    def available_books(self):
        """Return a list of books that are not checked out."""
        return [b for b in self.books if not b.is_checked_out]

    def __len__(self):
        return len(self.books)


# Usage
library = Library("City Library")
library.add_book(Book("Python Crash Course", "Eric Matthes", "978-1593279288"))
library.add_book(Book("Clean Code", "Robert Martin", "978-0132350884"))
library.add_book(Book("Design Patterns", "Gang of Four", "978-0201633610"))

book = library.find_by_title("python")[0]
book.check_out("Alice")
print(book)            # Book('Python Crash Course' by Eric Matthes) (checked out by Alice)
print(len(library))    # 3
print(library.available_books())  # [Clean Code, Design Patterns]
```

## Summary

- A **class** is a blueprint; an **object** is an instance created from that blueprint
- **Class attributes** are shared by all instances; **instance attributes** are unique per object
- Python looks up attributes in order: instance -> class -> parent classes
- Three method types: **instance methods** (`self`), **class methods** (`cls`), **static methods** (neither)
- Class methods are commonly used as **alternative constructors**
- Objects go through a lifecycle: `__new__` (allocate) -> `__init__` (initialize) -> use -> `__del__` (finalize)

## Next Steps

In [Lesson 03: Constructors and Initialization](03_Constructors_and_Initialization.md), we will explore the `__init__` method in depth, including parameter validation, default values, and initialization patterns.

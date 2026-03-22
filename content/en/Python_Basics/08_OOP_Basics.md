# OOP Basics

**Previous**: [Strings and Text Processing](./07_Strings_and_Text_Processing.md) | **Next**: [OOP Advanced](./09_OOP_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define classes and create objects using the `__init__` constructor and `self` parameter
2. Distinguish between instance variables and class variables and know when to use each
3. Write instance methods, class methods (`@classmethod`), and static methods (`@staticmethod`)
4. Implement `__str__` and `__repr__` for readable object representation
5. Use the `@property` decorator to create managed attributes with getters and setters
6. Apply Python naming conventions for encapsulation (`public`, `_protected`, `__private`)
7. Understand the object lifecycle including construction and the `__del__` destructor
8. Design simple classes that model real-world entities with proper separation of concerns

---

Object-oriented programming (OOP) is a paradigm that organizes code around objects -- bundles of data (attributes) and behavior (methods) that model real-world entities. Instead of writing procedures that operate on separate data, you define classes that encapsulate both state and behavior. OOP promotes code reuse, modularity, and maintainability, and it is the dominant paradigm in large-scale Python applications, frameworks, and libraries.

## 1. Classes and Objects

A **class** is a blueprint; an **object** (or instance) is a concrete realization of that blueprint.

```python
# Define a class
class Dog:
    pass

# Create objects (instances)
dog1 = Dog()
dog2 = Dog()

print(type(dog1))       # <class '__main__.Dog'>
print(isinstance(dog1, Dog))  # True
print(dog1 is dog2)     # False (different objects)
```

### The `__init__` Constructor

The `__init__` method is called automatically when a new object is created. It initializes the object's attributes.

```python
class Dog:
    def __init__(self, name, breed, age):
        self.name = name
        self.breed = breed
        self.age = age

# Create objects with initial values
dog1 = Dog("Rex", "German Shepherd", 5)
dog2 = Dog("Buddy", "Golden Retriever", 3)

print(dog1.name)   # Rex
print(dog2.breed)  # Golden Retriever
```

### The `self` Parameter

`self` refers to the current instance. It is always the first parameter of instance methods, though you never pass it explicitly.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius  # self.radius is an instance attribute

    def area(self):
        return 3.14159 * self.radius ** 2

    def circumference(self):
        return 2 * 3.14159 * self.radius

c = Circle(5)
print(f"Area: {c.area():.2f}")             # Area: 78.54
print(f"Circumference: {c.circumference():.2f}")  # Circumference: 31.42

# Behind the scenes, Python translates c.area() to Circle.area(c)
print(f"Area: {Circle.area(c):.2f}")       # Area: 78.54
```

### Constructor with Default Values

```python
class Student:
    def __init__(self, name, grade=0, courses=None):
        self.name = name
        self.grade = grade
        self.courses = courses if courses is not None else []

    def enroll(self, course):
        if course not in self.courses:
            self.courses.append(course)

    def display(self):
        courses_str = ", ".join(self.courses) if self.courses else "None"
        print(f"{self.name} (Grade: {self.grade}) - Courses: {courses_str}")

s1 = Student("Alice", 95)
s1.enroll("Math")
s1.enroll("Physics")
s1.display()  # Alice (Grade: 95) - Courses: Math, Physics

s2 = Student("Bob")
s2.display()  # Bob (Grade: 0) - Courses: None
```

---

## 2. Instance Variables vs Class Variables

### Instance Variables

Instance variables belong to each object. They are defined inside `__init__` using `self`.

```python
class Player:
    def __init__(self, name, score=0):
        self.name = name      # instance variable
        self.score = score    # instance variable

p1 = Player("Alice", 100)
p2 = Player("Bob", 200)

# Each instance has its own copy
print(p1.score)  # 100
print(p2.score)  # 200

p1.score = 150
print(p1.score)  # 150
print(p2.score)  # 200 (unchanged)
```

### Class Variables

Class variables are shared across all instances. They are defined directly in the class body.

```python
class Player:
    # Class variable: shared by all instances
    game_name = "Adventure Quest"
    player_count = 0

    def __init__(self, name, score=0):
        self.name = name      # instance variable
        self.score = score    # instance variable
        Player.player_count += 1  # modify class variable

p1 = Player("Alice")
p2 = Player("Bob")
p3 = Player("Charlie")

# Access class variable through class or instance
print(Player.game_name)     # Adventure Quest
print(p1.game_name)         # Adventure Quest
print(Player.player_count)  # 3
```

### Shadowing Class Variables

Assigning to an attribute via an instance creates a new instance variable that shadows the class variable.

```python
class Config:
    debug = False
    version = "1.0"

c1 = Config()
c2 = Config()

# Both see the class variable
print(c1.debug)  # False
print(c2.debug)  # False

# Assigning via instance creates an instance variable
c1.debug = True
print(c1.debug)  # True  (instance variable)
print(c2.debug)  # False (still class variable)
print(Config.debug)  # False (class variable unchanged)

# Check where the attribute lives
print("debug" in c1.__dict__)  # True (instance has its own)
print("debug" in c2.__dict__)  # False (c2 uses class variable)
```

### Mutable Class Variable Pitfall

```python
# BAD: mutable class variable shared by all instances
class StudentBad:
    courses = []  # Shared mutable list!

    def __init__(self, name):
        self.name = name

s1 = StudentBad("Alice")
s2 = StudentBad("Bob")

s1.courses.append("Math")
print(s2.courses)  # ['Math'] -- both students share the same list!

# GOOD: initialize mutable attributes in __init__
class StudentGood:
    def __init__(self, name):
        self.name = name
        self.courses = []  # Each instance gets its own list

s1 = StudentGood("Alice")
s2 = StudentGood("Bob")

s1.courses.append("Math")
print(s2.courses)  # [] -- independent
```

---

## 3. Instance Methods

Instance methods operate on a specific instance and can access/modify its attributes through `self`.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        """Add funds to the account."""
        if amount <= 0:
            print("Deposit amount must be positive")
            return
        self.balance += amount
        self.transactions.append(("deposit", amount))
        print(f"Deposited ${amount:.2f}. Balance: ${self.balance:.2f}")

    def withdraw(self, amount):
        """Withdraw funds from the account."""
        if amount <= 0:
            print("Withdrawal amount must be positive")
            return
        if amount > self.balance:
            print(f"Insufficient funds. Balance: ${self.balance:.2f}")
            return
        self.balance -= amount
        self.transactions.append(("withdraw", amount))
        print(f"Withdrew ${amount:.2f}. Balance: ${self.balance:.2f}")

    def get_statement(self):
        """Print account statement."""
        print(f"\n--- Statement for {self.owner} ---")
        for action, amount in self.transactions:
            symbol = "+" if action == "deposit" else "-"
            print(f"  {symbol}${amount:.2f}")
        print(f"  Current Balance: ${self.balance:.2f}")
        print("---")

account = BankAccount("Alice", 1000)
account.deposit(500)       # Deposited $500.00. Balance: $1500.00
account.withdraw(200)      # Withdrew $200.00. Balance: $1300.00
account.withdraw(2000)     # Insufficient funds. Balance: $1300.00
account.deposit(100)       # Deposited $100.00. Balance: $1400.00
account.get_statement()
```

### Method Chaining

Return `self` from methods to enable chaining.

```python
class QueryBuilder:
    def __init__(self, table):
        self.table = table
        self._columns = "*"
        self._conditions = []
        self._order = None
        self._limit = None

    def select(self, *columns):
        self._columns = ", ".join(columns)
        return self  # Enable chaining

    def where(self, condition):
        self._conditions.append(condition)
        return self

    def order_by(self, column, desc=False):
        direction = "DESC" if desc else "ASC"
        self._order = f"{column} {direction}"
        return self

    def limit(self, n):
        self._limit = n
        return self

    def build(self):
        query = f"SELECT {self._columns} FROM {self.table}"
        if self._conditions:
            query += " WHERE " + " AND ".join(self._conditions)
        if self._order:
            query += f" ORDER BY {self._order}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query

# Fluent interface with method chaining
query = (QueryBuilder("users")
         .select("name", "email", "age")
         .where("age >= 18")
         .where("active = true")
         .order_by("name")
         .limit(10)
         .build())

print(query)
# SELECT name, email, age FROM users WHERE age >= 18 AND active = true ORDER BY name ASC LIMIT 10
```

---

## 4. Class Methods and Static Methods

### Class Methods (`@classmethod`)

Class methods receive the class (`cls`) as the first argument instead of an instance. They can access and modify class state.

```python
class Employee:
    raise_percentage = 1.05  # 5% raise
    employee_count = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1

    def apply_raise(self):
        self.salary *= self.raise_percentage

    @classmethod
    def set_raise_percentage(cls, percentage):
        """Set raise percentage for all employees."""
        cls.raise_percentage = percentage

    @classmethod
    def from_string(cls, employee_str):
        """Alternative constructor from a dash-separated string."""
        name, salary = employee_str.split("-")
        return cls(name, float(salary))

    @classmethod
    def get_count(cls):
        return cls.employee_count

# Regular construction
emp1 = Employee("Alice", 50000)

# Alternative constructor via classmethod
emp2 = Employee.from_string("Bob-60000")
print(emp2.name)     # Bob
print(emp2.salary)   # 60000.0

# Modify class variable
Employee.set_raise_percentage(1.10)
emp1.apply_raise()
print(f"Alice's salary: {emp1.salary:.2f}")  # 55000.00

print(f"Total employees: {Employee.get_count()}")  # 2
```

### Static Methods (`@staticmethod`)

Static methods do not receive `self` or `cls`. They are utility functions that logically belong to the class but do not need access to instance or class state.

```python
class MathUtils:
    @staticmethod
    def is_prime(n):
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def factorial(n):
        """Calculate factorial."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def gcd(a, b):
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

# Call without creating an instance
print(MathUtils.is_prime(17))     # True
print(MathUtils.factorial(5))     # 120
print(MathUtils.gcd(48, 18))     # 6
```

### Comparison: Instance vs Class vs Static Methods

```python
class MyClass:
    class_var = "I am a class variable"

    def __init__(self, value):
        self.instance_var = value

    def instance_method(self):
        """Access instance and class data via self."""
        return f"instance: {self.instance_var}, class: {self.class_var}"

    @classmethod
    def class_method(cls):
        """Access class data via cls. No instance access."""
        return f"class: {cls.class_var}"

    @staticmethod
    def static_method(x, y):
        """No access to instance or class data."""
        return x + y

obj = MyClass("hello")
print(obj.instance_method())   # instance: hello, class: I am a class variable
print(MyClass.class_method())  # class: I am a class variable
print(MyClass.static_method(3, 4))  # 7
```

| Feature | Instance Method | Class Method | Static Method |
|---------|----------------|--------------|---------------|
| First param | `self` | `cls` | None |
| Access instance? | Yes | No | No |
| Access class? | Yes (via `self.__class__`) | Yes (via `cls`) | No |
| Called on | Instance | Class or Instance | Class or Instance |
| Common use | Object behavior | Alternative constructors, class-level ops | Utility functions |

---

## 5. `__str__` and `__repr__`

These special methods control how objects are displayed.

- `__str__`: Human-readable representation (used by `print()` and `str()`)
- `__repr__`: Developer-oriented representation (used by `repr()`, debugger, and the interactive prompt)

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(3, 4)

print(p)           # (3, 4)         -- uses __str__
print(repr(p))     # Point(3, 4)    -- uses __repr__
print(f"Point: {p}")  # Point: (3, 4) -- uses __str__

# In a list, __repr__ is used for elements
points = [Point(1, 2), Point(3, 4)]
print(points)      # [Point(1, 2), Point(3, 4)]  -- uses __repr__
```

### Guidelines for `__repr__` and `__str__`

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __repr__(self):
        # Should ideally be a valid Python expression that recreates the object
        return f"Temperature({self.celsius})"

    def __str__(self):
        # Should be readable for end users
        return f"{self.celsius}°C ({self.celsius * 9/5 + 32:.1f}°F)"

t = Temperature(100)
print(repr(t))  # Temperature(100)
print(str(t))   # 100°C (212.0°F)

# If only __repr__ is defined, it is used as fallback for __str__
class Simple:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Simple({self.value!r})"

s = Simple("hello")
print(s)        # Simple('hello')  -- __repr__ used as fallback
print(repr(s))  # Simple('hello')
```

---

## 6. The `@property` Decorator

Properties let you define methods that are accessed like attributes. They enable computed attributes and controlled access.

### Basic Property (Getter)

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """Get the radius."""
        return self._radius

    @property
    def area(self):
        """Computed property: area of the circle."""
        return 3.14159 * self._radius ** 2

    @property
    def diameter(self):
        """Computed property: diameter of the circle."""
        return self._radius * 2

c = Circle(5)
print(c.radius)    # 5       (accessed like an attribute)
print(c.area)      # 78.53975
print(c.diameter)  # 10

# Cannot set read-only properties
# c.area = 100  # AttributeError: cannot set attribute
```

### Property with Setter

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius  # Store in private attribute

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5 / 9  # Reuses celsius setter validation

t = Temperature(25)
print(f"{t.celsius}°C = {t.fahrenheit}°F")  # 25°C = 77.0°F

t.fahrenheit = 212
print(f"{t.celsius}°C = {t.fahrenheit}°F")  # 100.0°C = 212.0°F

# Validation works
try:
    t.celsius = -300
except ValueError as e:
    print(e)  # Temperature below absolute zero is not possible
```

### Property with Deleter

```python
class CachedData:
    def __init__(self, source):
        self.source = source
        self._cache = None

    @property
    def data(self):
        if self._cache is None:
            print("Loading data from source...")
            self._cache = f"Data from {self.source}"
        return self._cache

    @data.deleter
    def data(self):
        print("Clearing cache...")
        self._cache = None

obj = CachedData("database")
print(obj.data)   # Loading data from source... / Data from database
print(obj.data)   # Data from database (cached, no loading message)

del obj.data      # Clearing cache...
print(obj.data)   # Loading data from source... / Data from database
```

---

## 7. Encapsulation and Naming Conventions

Python uses naming conventions (not strict access modifiers like Java/C++) to indicate the intended visibility of attributes and methods.

### Public Attributes

```python
class User:
    def __init__(self, name, email):
        self.name = name    # public: accessible from anywhere
        self.email = email  # public

user = User("Alice", "alice@example.com")
print(user.name)      # Alice
user.name = "Bob"     # OK, can modify directly
```

### Protected Attributes (`_single_underscore`)

A single leading underscore is a convention meaning "internal use." It signals that the attribute should not be accessed from outside the class, but Python does not enforce this.

```python
class Account:
    def __init__(self, owner, balance):
        self.owner = owner
        self._balance = balance  # protected by convention

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def get_balance(self):
        return self._balance

acc = Account("Alice", 1000)
print(acc.get_balance())  # 1000

# Still accessible (Python trusts the developer)
print(acc._balance)       # 1000 -- works, but discouraged
```

### Private Attributes (`__double_underscore`)

A double leading underscore triggers **name mangling**: Python renames the attribute to `_ClassName__attribute`, making accidental access harder.

```python
class SecureAccount:
    def __init__(self, owner, balance, pin):
        self.owner = owner
        self._balance = balance   # protected
        self.__pin = pin          # private (name-mangled)

    def verify_pin(self, pin):
        return self.__pin == pin

    def get_balance(self, pin):
        if self.verify_pin(pin):
            return self._balance
        return "Invalid PIN"

acc = SecureAccount("Alice", 5000, "1234")
print(acc.get_balance("1234"))  # 5000
print(acc.get_balance("0000"))  # Invalid PIN

# Direct access fails
# print(acc.__pin)  # AttributeError: 'SecureAccount' has no attribute '__pin'

# But name mangling can be bypassed (not truly private)
print(acc._SecureAccount__pin)  # 1234 -- possible but strongly discouraged
```

### Summary of Naming Conventions

| Convention | Example | Meaning |
|------------|---------|---------|
| `name` | `self.name` | Public -- free to use from anywhere |
| `_name` | `self._name` | Protected -- internal use, do not access externally |
| `__name` | `self.__name` | Private -- name-mangled to `_Class__name` |
| `__name__` | `self.__init__` | Dunder/magic -- special Python methods |

---

## 8. Object Lifecycle

### Construction: `__new__` and `__init__`

`__new__` creates the instance; `__init__` initializes it. In most cases, you only override `__init__`.

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print(f"1. __new__ called (creating instance of {cls.__name__})")
        instance = super().__new__(cls)
        return instance

    def __init__(self, value):
        print(f"2. __init__ called (initializing with {value})")
        self.value = value

obj = MyClass(42)
# 1. __new__ called (creating instance of MyClass)
# 2. __init__ called (initializing with 42)
```

### Destruction: `__del__`

`__del__` is called when an object is about to be garbage collected. It is rarely needed in practice.

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Resource '{self.name}' created")

    def __del__(self):
        print(f"Resource '{self.name}' destroyed")

# Normal lifecycle
r = Resource("file_handler")  # Resource 'file_handler' created
del r                          # Resource 'file_handler' destroyed

# Also triggered when reference count drops to zero
def demo():
    r = Resource("temp")  # Resource 'temp' created
    print("Inside function")
    # r goes out of scope when function returns

demo()
# Inside function
# Resource 'temp' destroyed (eventually, when garbage collected)
```

### Important Notes on `__del__`

```python
# __del__ is NOT guaranteed to be called immediately
# Prefer context managers (with statement) for cleanup

class FileHandler:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        print(f"Opened {filename}")

    def write(self, data):
        self.file.write(data)

    def __del__(self):
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()
            print(f"Closed {self.filename}")

# Better approach: context manager protocol
class BetterFileHandler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        print(f"Closed {self.filename}")
        return False

# Usage
# with BetterFileHandler("output.txt") as handler:
#     handler.file.write("Hello!")
# File is guaranteed to close here
```

---

## 9. Practical Examples

### Example: Student Grade Tracker

```python
class GradeTracker:
    """Track and analyze student grades."""

    # Class variable: grading scale
    GRADE_SCALE = {
        "A+": (97, 100), "A": (93, 96), "A-": (90, 92),
        "B+": (87, 89),  "B": (83, 86), "B-": (80, 82),
        "C+": (77, 79),  "C": (73, 76), "C-": (70, 72),
        "D+": (67, 69),  "D": (63, 66), "D-": (60, 62),
        "F": (0, 59),
    }

    def __init__(self, student_name):
        self.student_name = student_name
        self._grades = {}  # subject -> list of scores

    def add_grade(self, subject, score):
        """Add a grade for a subject."""
        if not 0 <= score <= 100:
            raise ValueError(f"Score must be 0-100, got {score}")
        if subject not in self._grades:
            self._grades[subject] = []
        self._grades[subject].append(score)

    @property
    def subjects(self):
        """List of all subjects."""
        return list(self._grades.keys())

    @property
    def gpa(self):
        """Calculate overall GPA on a 4.0 scale."""
        if not self._grades:
            return 0.0
        all_scores = []
        for scores in self._grades.values():
            all_scores.extend(scores)
        avg = sum(all_scores) / len(all_scores)
        return min(4.0, avg / 25)  # Simple 4.0 scale approximation

    def get_average(self, subject=None):
        """Get average score, optionally for a specific subject."""
        if subject:
            scores = self._grades.get(subject, [])
            return sum(scores) / len(scores) if scores else 0
        all_scores = [s for scores in self._grades.values() for s in scores]
        return sum(all_scores) / len(all_scores) if all_scores else 0

    @staticmethod
    def score_to_letter(score):
        """Convert numeric score to letter grade."""
        for letter, (low, high) in GradeTracker.GRADE_SCALE.items():
            if low <= score <= high:
                return letter
        return "N/A"

    def __str__(self):
        avg = self.get_average()
        letter = self.score_to_letter(avg) if avg else "N/A"
        return f"{self.student_name} - Average: {avg:.1f} ({letter})"

    def __repr__(self):
        return f"GradeTracker({self.student_name!r})"

# Usage
tracker = GradeTracker("Alice")
tracker.add_grade("Math", 95)
tracker.add_grade("Math", 88)
tracker.add_grade("Science", 92)
tracker.add_grade("English", 87)

print(tracker)  # Alice - Average: 90.5 (A-)
print(f"Math average: {tracker.get_average('Math'):.1f}")  # Math average: 91.5
print(f"Subjects: {tracker.subjects}")  # Subjects: ['Math', 'Science', 'English']
print(f"GPA: {tracker.gpa:.2f}")  # GPA: 3.62
```

### Example: Inventory Item with Properties

```python
class InventoryItem:
    """Represent an item in inventory with price and quantity validation."""

    _tax_rate = 0.10  # Class-level tax rate (10%)

    def __init__(self, name, price, quantity=0):
        self.name = name
        self.price = price        # Uses property setter
        self.quantity = quantity   # Uses property setter

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError(f"Price cannot be negative: {value}")
        self._price = round(value, 2)

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Quantity must be a non-negative integer: {value}")
        self._quantity = value

    @property
    def total_value(self):
        """Total value of this item in inventory."""
        return self._price * self._quantity

    @property
    def price_with_tax(self):
        """Price including tax."""
        return self._price * (1 + self._tax_rate)

    @classmethod
    def set_tax_rate(cls, rate):
        """Set tax rate for all items."""
        if not 0 <= rate <= 1:
            raise ValueError("Tax rate must be between 0 and 1")
        cls._tax_rate = rate

    @classmethod
    def from_dict(cls, data):
        """Create an InventoryItem from a dictionary."""
        return cls(
            name=data["name"],
            price=data["price"],
            quantity=data.get("quantity", 0),
        )

    def restock(self, amount):
        """Add items to inventory."""
        self.quantity += amount

    def sell(self, amount):
        """Remove items from inventory."""
        if amount > self._quantity:
            raise ValueError(f"Cannot sell {amount}, only {self._quantity} in stock")
        self.quantity -= amount

    def __str__(self):
        return f"{self.name}: ${self._price:.2f} x {self._quantity} = ${self.total_value:.2f}"

    def __repr__(self):
        return f"InventoryItem({self.name!r}, {self._price}, {self._quantity})"

# Usage
item = InventoryItem("Widget", 9.99, 100)
print(item)  # Widget: $9.99 x 100 = $999.00

item.sell(30)
print(item)  # Widget: $9.99 x 70 = $699.30

print(f"Price with tax: ${item.price_with_tax:.2f}")  # Price with tax: $10.99

# From dictionary
data = {"name": "Gadget", "price": 24.99, "quantity": 50}
item2 = InventoryItem.from_dict(data)
print(item2)  # Gadget: $24.99 x 50 = $1249.50

# Validation
try:
    item.price = -5
except ValueError as e:
    print(e)  # Price cannot be negative: -5
```

### Example: Simple Linked List Node

```python
class Node:
    """A node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data!r})"

    def __str__(self):
        parts = []
        current = self
        while current:
            parts.append(str(current.data))
            current = current.next
        return " -> ".join(parts) + " -> None"

# Build a linked list: 1 -> 2 -> 3 -> None
head = Node(1, Node(2, Node(3)))
print(head)  # 1 -> 2 -> 3 -> None

# Traverse
current = head
while current:
    print(f"Visiting: {current.data}")
    current = current.next
# Visiting: 1
# Visiting: 2
# Visiting: 3
```

---

## 10. Summary

| Concept | Key Points |
|---------|------------|
| Class | Blueprint defined with `class`; creates objects via `ClassName()` |
| `__init__` | Constructor; initializes instance attributes |
| `self` | Reference to the current instance; always first parameter |
| Instance variables | Per-object data (`self.attr`); defined in `__init__` |
| Class variables | Shared data; defined in class body |
| Instance methods | Operate on `self`; access instance and class data |
| `@classmethod` | Operates on `cls`; alternative constructors, class-level operations |
| `@staticmethod` | No `self` or `cls`; utility functions grouped in a class |
| `__str__`/`__repr__` | Human-readable / developer representation |
| `@property` | Managed attributes with getter, setter, deleter |
| Encapsulation | `public`, `_protected`, `__private` (name mangling) |
| `__del__` | Destructor; called before garbage collection (unreliable timing) |

---

## Exercises

1. Create a `Rectangle` class with `width` and `height` properties (validated to be positive), computed `area` and `perimeter` properties, and `__str__`/`__repr__` methods.
2. Build a `Playlist` class that manages a list of songs. Include `add_song`, `remove_song`, `shuffle`, `total_duration`, and a `from_file` classmethod that reads song data from a list of strings.
3. Implement a `Counter` class that tracks counts of items (similar to `collections.Counter`). Include `add`, `remove`, `most_common(n)`, and support `__str__` for display.
4. Create a `Matrix` class that stores a 2D grid of numbers. Add properties for `rows`, `cols`, and `shape`. Include a `@classmethod` factory `identity(n)` that creates an n x n identity matrix.
5. Design a `BankAccount` class with `__private` balance, `@property` for balance (read-only), `deposit`/`withdraw` methods with validation, and a transaction history.

---

**Previous**: [Strings and Text Processing](./07_Strings_and_Text_Processing.md) | **Next**: [OOP Advanced](./09_OOP_Advanced.md)

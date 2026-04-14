# Lesson 04: Encapsulation

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain encapsulation and why it is one of the four OOP pillars
2. Apply Python's naming conventions for access control (`_`, `__`)
3. Implement getters and setters using the `@property` decorator
4. Design clean public interfaces that hide implementation details
5. Use encapsulation to enforce invariants and validate state changes
6. Compare Python's approach to encapsulation with other languages
7. Avoid common encapsulation mistakes

## What Is Encapsulation?

Encapsulation is the principle of **bundling data and the methods that operate on that data together**, while **restricting direct access** to some of the object's components. It is about controlling how the outside world interacts with an object's internal state.

```
┌─────────────────────────────────────────────┐
│              Without Encapsulation          │
│                                             │
│  External Code ──── directly modifies ────▶ │
│                     obj.balance = -999      │
│                     obj.status = "invalid"  │
│                                             │
│  Result: Broken invariants, bugs, chaos     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              With Encapsulation             │
│                                             │
│  External Code ──── uses methods ──────────▶│
│                     obj.withdraw(100)       │
│                     obj.set_status("active")│
│                                             │
│  Methods validate, enforce rules, log       │
│  Result: Consistent state, fewer bugs       │
└─────────────────────────────────────────────┘
```

### The Capsule Analogy

Think of a capsule (pill). The medicine is enclosed inside a protective shell. You cannot reach in and rearrange the chemicals — you swallow the capsule and it releases the medicine in a controlled way. Similarly, encapsulation wraps data inside a protective layer of methods.

## Python's Access Control Conventions

Unlike Java or C++ which have `private`, `protected`, `public` keywords, Python relies on **naming conventions**. Python trusts developers with the phrase: "We are all consenting adults here."

```
┌──────────────┬─────────────┬──────────────────────────┐
│ Convention   │ Access      │ Meaning                  │
├──────────────┼─────────────┼──────────────────────────┤
│ name         │ Public      │ Free to use anywhere     │
│ _name        │ Protected   │ "Internal use" hint      │
│ __name       │ Private     │ Name-mangled by Python   │
│ __name__     │ Dunder/magic│ Reserved for Python      │
└──────────────┴─────────────┴──────────────────────────┘
```

### Public Attributes

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius  # Public — anyone can read/write

c = Circle(5)
print(c.radius)   # 5
c.radius = 10     # No restriction
c.radius = -999   # Oops! No validation
```

### Protected Attributes (Single Underscore `_`)

The single underscore is a **convention** — it tells other developers "this is internal, use at your own risk." Python does not enforce it.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance  # Convention: "don't touch directly"
        self._transactions = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount
        self._transactions.append(("deposit", amount))

    def get_balance(self):
        return self._balance


account = BankAccount("Alice", 1000)
# Works but signals "you probably shouldn't"
print(account._balance)  # 1000
```

### Private Attributes (Double Underscore `__`)

Double underscore triggers **name mangling**: Python renames `__attr` to `_ClassName__attr` to prevent accidental access from subclasses.

```python
class SecureAccount:
    def __init__(self, owner, pin):
        self.owner = owner
        self.__pin = pin       # Name-mangled to _SecureAccount__pin
        self.__balance = 0

    def verify_pin(self, pin):
        return pin == self.__pin

    def get_balance(self, pin):
        if not self.verify_pin(pin):
            raise PermissionError("Invalid PIN")
        return self.__balance


acct = SecureAccount("Bob", "1234")

# Direct access fails
# print(acct.__pin)  -> AttributeError

# But name-mangled version still accessible (not truly private!)
print(acct._SecureAccount__pin)  # "1234" — Python's "escape hatch"
```

```
┌─────────────────────────────────────────────────┐
│  Important: Python has no TRUE private access.  │
│  Name mangling prevents ACCIDENTAL access and   │
│  conflicts in subclasses, not INTENTIONAL.      │
└─────────────────────────────────────────────────┘
```

## The `@property` Decorator

The `@property` decorator is Python's elegant solution for encapsulation. It lets you define methods that look like attributes:

```python
class Temperature:
    """Temperature with automatic Fahrenheit conversion."""

    def __init__(self, celsius=0):
        self.celsius = celsius  # This triggers the setter!

    @property
    def celsius(self):
        """Get the temperature in Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """Set the temperature in Celsius with validation."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is impossible")
        self._celsius = value

    @property
    def fahrenheit(self):
        """Get the temperature in Fahrenheit (read-only computed property)."""
        return self._celsius * 9 / 5 + 32

    def __repr__(self):
        return f"Temperature({self._celsius:.1f}C / {self.fahrenheit:.1f}F)"


t = Temperature(100)
print(t.celsius)      # 100 (looks like attribute access, calls getter)
print(t.fahrenheit)   # 212.0 (computed on the fly)

t.celsius = 0         # Looks like assignment, calls setter with validation
print(t)              # Temperature(0.0C / 32.0F)

# t.celsius = -300    # ValueError: below absolute zero
# t.fahrenheit = 100  # AttributeError: can't set (read-only)
```

### Property Flow

```
    t.celsius = 25          t.celsius
         │                      │
         ▼                      ▼
  @celsius.setter          @property
  def celsius(self, val):  def celsius(self):
      validate(val)            return self._celsius
      self._celsius = val
```

### Read-Only Properties

```python
class Employee:
    """Employee with read-only computed properties."""

    def __init__(self, first_name, last_name, hourly_rate, hours_per_week):
        self.first_name = first_name
        self.last_name = last_name
        self.hourly_rate = hourly_rate
        self.hours_per_week = hours_per_week

    @property
    def full_name(self):
        """Full name is read-only and computed."""
        return f"{self.first_name} {self.last_name}"

    @property
    def weekly_pay(self):
        """Weekly pay is read-only and computed."""
        return self.hourly_rate * self.hours_per_week

    @property
    def annual_salary(self):
        """Estimated annual salary (52 weeks)."""
        return self.weekly_pay * 52


emp = Employee("Alice", "Smith", 50, 40)
print(emp.full_name)       # Alice Smith
print(emp.weekly_pay)      # 2000
print(emp.annual_salary)   # 104000

# emp.full_name = "Bob"    # AttributeError: can't set
```

### Property with Deleter

```python
class CachedData:
    """Demonstrates property with deleter for cache invalidation."""

    def __init__(self, raw_data):
        self._raw_data = raw_data
        self._processed = None

    @property
    def processed(self):
        """Lazy processing with caching."""
        if self._processed is None:
            print("Processing data...")
            self._processed = sorted(set(self._raw_data))
        return self._processed

    @processed.deleter
    def processed(self):
        """Clear the cache."""
        print("Cache cleared")
        self._processed = None


data = CachedData([3, 1, 4, 1, 5, 9, 2, 6, 5])
print(data.processed)  # Processing data... -> [1, 2, 3, 4, 5, 6, 9]
print(data.processed)  # No processing (cached) -> [1, 2, 3, 4, 5, 6, 9]
del data.processed     # Cache cleared
print(data.processed)  # Processing data... -> [1, 2, 3, 4, 5, 6, 9]
```

## Enforcing Invariants

Encapsulation's greatest power is maintaining **invariants** — conditions that must always be true:

```python
class DateRange:
    """A date range where start must be before end."""

    def __init__(self, start, end):
        if start > end:
            raise ValueError(f"Start ({start}) must be before end ({end})")
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if value > self._end:
            raise ValueError(f"Start ({value}) must be before end ({self._end})")
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if value < self._start:
            raise ValueError(f"End ({value}) must be after start ({self._start})")
        self._end = value

    @property
    def duration(self):
        return self._end - self._start


from datetime import date
dr = DateRange(date(2024, 1, 1), date(2024, 12, 31))
print(dr.duration)  # 365 days
dr.end = date(2025, 6, 30)  # OK
# dr.start = date(2026, 1, 1)  # ValueError: must be before end
```

## Encapsulation in Other Languages

```
┌──────────────┬────────────────────────────────────────┐
│  Language    │  Access Control Mechanism              │
├──────────────┼────────────────────────────────────────┤
│  Java/C#     │  private, protected, public keywords  │
│              │  Compiler-enforced                     │
├──────────────┼────────────────────────────────────────┤
│  C++         │  private, protected, public sections  │
│              │  friend classes can bypass             │
├──────────────┼────────────────────────────────────────┤
│  Python      │  Convention only (_single, __double)   │
│              │  No compiler enforcement               │
├──────────────┼────────────────────────────────────────┤
│  JavaScript  │  #private fields (ES2022)              │
│              │  Closures for older patterns            │
├──────────────┼────────────────────────────────────────┤
│  Go          │  Capitalization (exported vs unexported)│
│              │  Package-level, not class-level         │
└──────────────┴────────────────────────────────────────┘
```

## Common Mistakes

### Mistake 1: Over-Encapsulating

```python
# BAD: Pointless getters/setters with no logic
class OverEngineered:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value  # No validation, no logic — pointless!

# GOOD: Just use a public attribute
class Simple:
    def __init__(self, name):
        self.name = name  # Direct access is fine if no validation needed
```

### Mistake 2: Exposing Internal Mutable State

```python
# BAD: Returning internal list
class BadTeam:
    def __init__(self):
        self._members = []

    @property
    def members(self):
        return self._members  # Caller can mutate our internal list!


# GOOD: Return a copy
class GoodTeam:
    def __init__(self):
        self._members = []

    @property
    def members(self):
        return list(self._members)  # Return a copy

    def add_member(self, name):
        if name in self._members:
            raise ValueError(f"{name} is already a member")
        self._members.append(name)
```

## Summary

- Encapsulation bundles data and methods together while controlling access to internal state
- Python uses naming conventions: `public`, `_protected`, `__private` (name-mangled)
- `@property` provides attribute-like syntax with getter/setter/deleter logic
- Use properties to enforce invariants and validate state changes
- Prefer public attributes when no validation is needed — avoid trivial getters/setters
- Never expose internal mutable state directly — return copies

## Next Steps

In [Lesson 05: Inheritance](05_Inheritance.md), we will explore how classes can inherit attributes and methods from parent classes.

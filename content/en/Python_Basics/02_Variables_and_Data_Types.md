# Variables and Data Types

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Operators and Expressions](./03_Operators_and_Expressions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create variables using Python's assignment syntax and explain how names bind to objects
2. Distinguish between Python's core data types: `int`, `float`, `complex`, `str`, `bool`, and `None`
3. Use `type()` and `isinstance()` to inspect and verify object types at runtime
4. Convert between types using built-in functions (`int()`, `float()`, `str()`, `bool()`)
5. Apply Python's variable naming conventions and the `UPPER_CASE` constants convention
6. Use multiple assignment, tuple unpacking, and augmented assignment operators
7. Explain dynamic typing and how it differs from static typing
8. Perform numeric operations including integer division, modulo, and power

---

Every program manipulates data. In Python, data is stored in **objects**, and we use **variables** as names that refer to those objects. Understanding how variables work and what types of data Python provides is the foundation on which every subsequent lesson builds.

## Variables and Assignment

### What Is a Variable?

In Python, a variable is not a box that holds data. It is a **name** that refers to (or "points at") an object in memory. When you write:

```python
x = 42
```

Python does three things:

1. Creates an integer object with the value `42` in memory.
2. Creates the name `x` (if it does not already exist).
3. Binds the name `x` to the integer object.

This distinction matters. Multiple names can refer to the same object:

```python
a = [1, 2, 3]
b = a           # b now refers to the SAME list object

b.append(4)
print(a)        # [1, 2, 3, 4] -- a sees the change too!
print(b)        # [1, 2, 3, 4]

print(a is b)   # True -- same object in memory
print(id(a))    # Same memory address
print(id(b))    # Same memory address
```

### Assignment Syntax

```python
# Simple assignment
name = "Alice"
age = 30
pi = 3.14159

# Variables can be reassigned to different types (dynamic typing)
x = 10          # x is an int
x = "hello"     # x is now a str
x = [1, 2, 3]   # x is now a list
```

### Multiple Assignment

Python supports several forms of multiple assignment:

```python
# Assign the same value to multiple variables
a = b = c = 0
print(a, b, c)  # 0 0 0

# Assign different values in one line (tuple unpacking)
x, y, z = 1, 2, 3
print(x, y, z)  # 1 2 3

# Swap two variables (no temporary variable needed)
a = 10
b = 20
a, b = b, a
print(a, b)  # 20 10

# Extended unpacking with * (star expression)
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

*head, last = [1, 2, 3, 4, 5]
print(head)   # [1, 2, 3, 4]
print(last)   # 5

first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```

### Augmented Assignment

Augmented assignment operators combine an operation with assignment:

```python
count = 0
count += 1    # count = count + 1  ->  1
count += 5    # count = count + 5  ->  6
count -= 2    # count = count - 2  ->  4
count *= 3    # count = count * 3  ->  12
count //= 5   # count = count // 5 ->  2
count **= 3   # count = count ** 3 ->  8
count %= 5    # count = count % 5  ->  3

# Works with strings too
greeting = "Hello"
greeting += " World"
print(greeting)  # Hello World

# Works with lists
items = [1, 2]
items += [3, 4]
print(items)  # [1, 2, 3, 4]
```

All augmented assignment operators:

| Operator | Equivalent | Description |
|----------|-----------|-------------|
| `+=` | `x = x + y` | Add and assign |
| `-=` | `x = x - y` | Subtract and assign |
| `*=` | `x = x * y` | Multiply and assign |
| `/=` | `x = x / y` | Divide and assign |
| `//=` | `x = x // y` | Floor divide and assign |
| `%=` | `x = x % y` | Modulo and assign |
| `**=` | `x = x ** y` | Power and assign |
| `&=` | `x = x & y` | Bitwise AND and assign |
| `\|=` | `x = x \| y` | Bitwise OR and assign |
| `^=` | `x = x ^ y` | Bitwise XOR and assign |
| `>>=` | `x = x >> y` | Right shift and assign |
| `<<=` | `x = x << y` | Left shift and assign |

---

## Dynamic Typing

Python is a **dynamically typed** language. This means:

1. Variables do not have fixed types.
2. Types are associated with **objects**, not variables.
3. Type checking happens at **runtime**, not compile time.

```python
# The same variable can hold different types over its lifetime
value = 42          # int
print(type(value))  # <class 'int'>

value = "hello"     # str
print(type(value))  # <class 'str'>

value = [1, 2, 3]   # list
print(type(value))  # <class 'list'>
```

### Dynamic vs Static Typing

```python
# Python (dynamic typing)
x = 10        # No type declaration needed
x = "hello"   # Perfectly valid -- x changes type

# C (static typing) -- for comparison
# int x = 10;        // Must declare the type
# x = "hello";       // ERROR: cannot assign string to int
```

### Type Hints (Optional Annotations)

Python 3.5+ supports **type hints** — optional annotations that document expected types. They do not enforce types at runtime but help with readability and static analysis tools.

```python
# Type hints are suggestions, not constraints
name: str = "Alice"
age: int = 30
height: float = 5.8
is_student: bool = True

def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"

# Type hints do NOT prevent misuse at runtime
age: int = "not a number"  # No runtime error!
# But tools like mypy will flag this as an error
```

---

## Numeric Types

Python provides three built-in numeric types: `int`, `float`, and `complex`.

### Integers (`int`)

Python integers have **arbitrary precision** — they can be as large as your memory allows.

```python
# Integer literals
x = 42
negative = -17
zero = 0

# Large integers (no overflow!)
big = 2 ** 100
print(big)
# 1267650600228229401496703205376

# Underscores for readability (Python 3.6+)
population = 7_900_000_000
budget = 1_000_000
binary = 0b1010_0101
hex_value = 0xFF_FF

# Different bases
decimal = 255         # Base 10
binary = 0b11111111   # Base 2  (prefix 0b)
octal = 0o377         # Base 8  (prefix 0o)
hexadecimal = 0xFF    # Base 16 (prefix 0x)

print(decimal)       # 255
print(binary)        # 255
print(octal)         # 255
print(hexadecimal)   # 255

# Convert to string representations of different bases
print(bin(255))      # '0b11111111'
print(oct(255))      # '0o377'
print(hex(255))      # '0xff'
```

### Integer Operations

```python
# Basic arithmetic
print(10 + 3)    # 13  (addition)
print(10 - 3)    # 7   (subtraction)
print(10 * 3)    # 30  (multiplication)
print(10 / 3)    # 3.3333...  (true division -- always returns float)
print(10 // 3)   # 3   (floor division -- rounds down to int)
print(10 % 3)    # 1   (modulo -- remainder)
print(10 ** 3)   # 1000 (exponentiation)

# Floor division with negatives rounds toward negative infinity
print(-10 // 3)  # -4  (not -3!)
print(10 // -3)  # -4

# The divmod() function returns both quotient and remainder
quotient, remainder = divmod(17, 5)
print(quotient)    # 3
print(remainder)   # 2

# abs() for absolute value
print(abs(-42))    # 42
```

### Floating-Point Numbers (`float`)

Floats represent real numbers using IEEE 754 double-precision (64-bit) format.

```python
# Float literals
pi = 3.14159
negative = -2.5
small = 0.001

# Scientific notation
avogadro = 6.022e23      # 6.022 * 10^23
planck = 6.626e-34        # 6.626 * 10^-34
speed_of_light = 3.0e8    # 3.0 * 10^8

# Special float values
positive_inf = float("inf")
negative_inf = float("-inf")
not_a_number = float("nan")

print(positive_inf > 1e308)      # True
print(negative_inf < -1e308)     # True
print(not_a_number == not_a_number)  # False (NaN is never equal to itself!)

import math
print(math.isinf(positive_inf))  # True
print(math.isnan(not_a_number))  # True
```

### Floating-Point Precision Pitfalls

```python
# The classic floating-point surprise
print(0.1 + 0.2)
# 0.30000000000000004

print(0.1 + 0.2 == 0.3)
# False!

# Why? 0.1 cannot be represented exactly in binary floating-point.
# It is stored as 0.1000000000000000055511151231257827021181583404541015625

# Solution 1: Use math.isclose() for approximate comparison
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# Solution 2: Use the decimal module for exact decimal arithmetic
from decimal import Decimal
print(Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))  # True

# Solution 3: Use the fractions module for exact rational arithmetic
from fractions import Fraction
print(Fraction(1, 10) + Fraction(2, 10) == Fraction(3, 10))  # True
```

### Complex Numbers (`complex`)

Python has built-in support for complex numbers, using `j` (not `i`) for the imaginary part.

```python
# Complex number literals
z1 = 3 + 4j
z2 = complex(3, 4)   # Equivalent

print(z1.real)        # 3.0
print(z1.imag)        # 4.0
print(abs(z1))        # 5.0 (magnitude: sqrt(3^2 + 4^2))
print(z1.conjugate()) # (3-4j)

# Arithmetic with complex numbers
z3 = z1 + z2
print(z3)             # (6+8j)

z4 = z1 * z2
print(z4)             # (-7+24j)  because (3+4j)(3+4j) = 9+12j+12j+16j^2 = 9+24j-16

# For more complex operations, use the cmath module
import cmath
print(cmath.phase(z1))  # 0.9272... (angle in radians)
print(cmath.polar(z1))  # (5.0, 0.9272...) -- (magnitude, phase)
```

---

## Strings (`str`)

Strings are immutable sequences of Unicode characters. We cover strings briefly here and in depth in Lesson 07.

```python
# String literals
single = 'Hello'
double = "Hello"
triple_single = '''Multi-line
string'''
triple_double = """Also multi-line
string"""

# Strings are immutable
s = "Hello"
# s[0] = "h"  # TypeError: 'str' object does not support item assignment

# But you can create a new string
s = "h" + s[1:]
print(s)  # "hello"

# String length
print(len("Python"))  # 6

# String concatenation and repetition
greeting = "Hello" + " " + "World"
separator = "-" * 40
print(greeting)    # Hello World
print(separator)   # ----------------------------------------

# Escape sequences
newline = "Line 1\nLine 2"
tab = "Column1\tColumn2"
backslash = "Path: C:\\Users\\Alice"
quote = "She said \"hello\""

print(newline)
# Line 1
# Line 2

# Raw strings (ignore escape sequences)
raw = r"C:\Users\Alice\new_folder"
print(raw)  # C:\Users\Alice\new_folder

# f-strings (formatted string literals)
name = "Alice"
age = 30
print(f"{name} is {age} years old.")            # Alice is 30 years old.
print(f"In 5 years: {age + 5}")                 # In 5 years: 35
print(f"Pi is approximately {3.14159:.2f}")      # Pi is approximately 3.14
print(f"{'centered':^20}")                       #       centered
print(f"{1000000:,}")                            # 1,000,000
```

### Common String Escapes

| Escape | Meaning |
|--------|---------|
| `\n` | Newline |
| `\t` | Tab |
| `\\` | Backslash |
| `\'` | Single quote |
| `\"` | Double quote |
| `\0` | Null character |
| `\u00e9` | Unicode character (e.g., e with accent) |

---

## Booleans (`bool`)

Booleans represent truth values. There are exactly two: `True` and `False`.

```python
is_valid = True
is_empty = False

print(type(True))   # <class 'bool'>
print(type(False))  # <class 'bool'>

# Booleans are a subclass of int
print(isinstance(True, int))  # True
print(True == 1)   # True
print(False == 0)  # True
print(True + True)  # 2
print(True * 10)    # 10
```

### Truthiness and Falsiness

Every Python object has a boolean value. The following values are **falsy** (evaluate to `False`):

```python
# All of these are falsy
print(bool(False))     # False
print(bool(None))      # False
print(bool(0))         # False
print(bool(0.0))       # False
print(bool(0j))        # False
print(bool(""))        # False (empty string)
print(bool([]))        # False (empty list)
print(bool(()))        # False (empty tuple)
print(bool({}))        # False (empty dict)
print(bool(set()))     # False (empty set)
print(bool(range(0)))  # False (empty range)
```

Everything else is **truthy**:

```python
print(bool(1))           # True
print(bool(-1))          # True
print(bool(3.14))        # True
print(bool("hello"))     # True
print(bool([1, 2, 3]))   # True
print(bool({"a": 1}))    # True
```

This is extensively used in conditionals:

```python
name = ""
if name:
    print(f"Hello, {name}!")
else:
    print("Name is empty.")
# Output: Name is empty.

items = [1, 2, 3]
if items:
    print(f"Found {len(items)} items.")
else:
    print("No items.")
# Output: Found 3 items.
```

---

## None

`None` is Python's null value — it represents the absence of a value. There is exactly one `None` object.

```python
result = None

print(result)        # None
print(type(result))  # <class 'NoneType'>

# Always use 'is' (not ==) to check for None
if result is None:
    print("No result yet.")

if result is not None:
    print(f"Result: {result}")
```

### Common Uses of None

```python
# 1. Default return value of functions with no explicit return
def greet(name):
    """Print a greeting."""
    print(f"Hello, {name}!")

result = greet("Alice")
print(result)  # None

# 2. Default parameter values
def find_item(items, target, default=None):
    """Find an item in a list, returning default if not found."""
    for item in items:
        if item == target:
            return item
    return default

print(find_item([1, 2, 3], 4))         # None
print(find_item([1, 2, 3], 4, -1))     # -1

# 3. Placeholder for optional values
class User:
    def __init__(self, name, email=None):
        self.name = name
        self.email = email

user = User("Alice")
if user.email is None:
    print("No email on file.")
```

---

## Type Inspection

### `type()` — Get the Type of an Object

```python
print(type(42))           # <class 'int'>
print(type(3.14))         # <class 'float'>
print(type("hello"))      # <class 'str'>
print(type(True))         # <class 'bool'>
print(type(None))         # <class 'NoneType'>
print(type([1, 2, 3]))    # <class 'list'>
print(type({"a": 1}))     # <class 'dict'>

# You can compare types directly
print(type(42) == int)     # True
print(type("hi") == str)   # True
```

### `isinstance()` — Check Type with Inheritance

`isinstance()` is preferred over `type()` for type checking because it respects inheritance:

```python
# isinstance checks the inheritance chain
print(isinstance(True, bool))   # True
print(isinstance(True, int))    # True (bool is a subclass of int)
print(isinstance(42, int))      # True
print(isinstance(42, bool))     # False (int is NOT a subclass of bool)

# type() does NOT check inheritance
print(type(True) == int)   # False (type is exactly bool, not int)
print(type(True) == bool)  # True

# isinstance can check multiple types at once
def is_numeric(value):
    """Check if a value is numeric."""
    return isinstance(value, (int, float, complex))

print(is_numeric(42))       # True
print(is_numeric(3.14))     # True
print(is_numeric(2+3j))     # True
print(is_numeric("42"))     # False
```

### `id()` — Get the Memory Address

```python
a = 42
b = 42
c = 43

print(id(a))  # e.g., 140234866123456
print(id(b))  # Same as id(a) -- Python caches small integers
print(id(c))  # Different

# 'is' checks identity (same object), '==' checks equality (same value)
print(a is b)    # True (same cached object)
print(a == b)    # True (same value)

# But for larger integers
x = 1000
y = 1000
print(x == y)    # True  (same value)
print(x is y)    # May be False (different objects -- not cached)
```

> **Note**: Python caches small integers (typically -5 to 256) and short strings for performance. Do not rely on `is` for value comparisons — always use `==` for values and `is` only for `None`, `True`, `False`, and explicit identity checks.

---

## Type Conversion

Python provides built-in functions for converting between types.

### `int()` — Convert to Integer

```python
# From float (truncates toward zero)
print(int(3.7))      # 3
print(int(3.2))      # 3
print(int(-3.7))     # -3
print(int(-3.2))     # -3

# From string
print(int("42"))     # 42
print(int("-17"))    # -17
print(int("0xFF", 16))  # 255 (specify base)
print(int("0b1010", 2))  # 10
print(int("0o77", 8))    # 63

# From boolean
print(int(True))     # 1
print(int(False))    # 0

# Errors
# int("3.14")   # ValueError: invalid literal (use float() first)
# int("hello")  # ValueError: invalid literal
```

### `float()` — Convert to Float

```python
# From int
print(float(42))      # 42.0

# From string
print(float("3.14"))  # 3.14
print(float("-2.5"))  # -2.5
print(float("1e10"))  # 10000000000.0

# Special values
print(float("inf"))   # inf
print(float("-inf"))  # -inf
print(float("nan"))   # nan

# From boolean
print(float(True))    # 1.0
print(float(False))   # 0.0
```

### `str()` — Convert to String

```python
# From any type
print(str(42))         # '42'
print(str(3.14))       # '3.14'
print(str(True))       # 'True'
print(str(None))       # 'None'
print(str([1, 2, 3]))  # '[1, 2, 3]'

# repr() gives a more detailed string representation
print(repr("hello"))   # "'hello'"  (includes quotes)
print(repr(42))        # '42'
print(repr([1, 2]))    # '[1, 2]'
```

### `bool()` — Convert to Boolean

```python
# See the "Truthiness and Falsiness" section above
print(bool(0))      # False
print(bool(1))      # True
print(bool(""))     # False
print(bool("hi"))   # True
print(bool([]))     # False
print(bool([0]))    # True (non-empty list, even if it contains a falsy value)
```

### Safe Conversion Pattern

```python
def safe_int(value, default=0):
    """Convert a value to int safely, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

print(safe_int("42"))       # 42
print(safe_int("hello"))    # 0
print(safe_int(None))       # 0
print(safe_int("99", -1))   # 99
print(safe_int("abc", -1))  # -1
```

---

## Variable Naming Conventions

Python has strong naming conventions defined in PEP 8:

### Rules (Enforced by the Language)

```python
# Variable names MUST:
# - Start with a letter (a-z, A-Z) or underscore (_)
# - Contain only letters, digits (0-9), and underscores
# - Not be a Python keyword

# Valid names
name = "Alice"
_private = "internal"
count_2 = 42
__dunder__ = "special"

# Invalid names
# 2count = 42       # SyntaxError: cannot start with a digit
# my-var = 42       # SyntaxError: hyphens not allowed
# class = "hello"   # SyntaxError: 'class' is a keyword
```

### Conventions (Enforced by the Community)

```python
# Variables and functions: snake_case
user_name = "Alice"
item_count = 42
is_valid = True

def calculate_total(items):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
PI = 3.14159265358979
DATABASE_URL = "postgresql://localhost/mydb"

# Classes: PascalCase (covered in Lesson 08)
class UserAccount:
    pass

# Private variables: leading underscore
_internal_cache = {}

# Name-mangled variables: double leading underscore
# (used inside classes to avoid name conflicts)
class MyClass:
    __secret = 42   # Becomes _MyClass__secret

# Dunder (double-underscore) names: reserved for Python
# __init__, __str__, __repr__, __len__, etc.
# Never invent your own dunder names.
```

### Python Keywords

These names are reserved and cannot be used as variable names:

```python
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
#  'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
#  'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
#  'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
#  'try', 'while', 'with', 'yield']
```

### Naming Best Practices

```python
# GOOD: descriptive, clear intent
user_age = 25
total_price = 99.99
is_authenticated = True
max_connections = 100
file_path = "/tmp/data.csv"

# BAD: vague, single-letter (except in small loops), misleading
x = 25                  # What does x represent?
tp = 99.99              # Abbreviation unclear
flag = True             # What flag?
n = 100                 # n could mean anything

# ACCEPTABLE: single letters in limited contexts
for i in range(10):        # Loop counter
    pass

for x, y in coordinates:  # Mathematical convention
    pass

# Avoid shadowing built-in names
# BAD
list = [1, 2, 3]     # Shadows the built-in list() function
type = "admin"        # Shadows the built-in type() function
id = 42               # Shadows the built-in id() function

# GOOD
items = [1, 2, 3]
user_type = "admin"
user_id = 42
```

---

## Memory Model and Object Identity

Understanding Python's memory model helps prevent subtle bugs.

### Mutable vs Immutable Objects

| Type | Mutable? | Examples |
|------|----------|---------|
| `int` | No | `42`, `-7` |
| `float` | No | `3.14`, `-2.5` |
| `str` | No | `"hello"`, `""` |
| `bool` | No | `True`, `False` |
| `tuple` | No | `(1, 2, 3)` |
| `frozenset` | No | `frozenset({1, 2})` |
| `list` | **Yes** | `[1, 2, 3]` |
| `dict` | **Yes** | `{"a": 1}` |
| `set` | **Yes** | `{1, 2, 3}` |

```python
# Immutable: operations create new objects
a = "hello"
b = a.upper()    # Creates a NEW string
print(a)         # "hello" (unchanged)
print(b)         # "HELLO"
print(a is b)    # False (different objects)

# Mutable: operations can modify in place
x = [1, 2, 3]
y = x
y.append(4)      # Modifies the SAME list
print(x)         # [1, 2, 3, 4] (changed!)
print(x is y)    # True (same object)
```

### Copying Objects

```python
# Shallow copy (copies the outer container, not inner objects)
original = [1, [2, 3], 4]
shallow = original.copy()      # or: shallow = list(original)
                                # or: shallow = original[:]

shallow[0] = 99
print(original)  # [1, [2, 3], 4]  -- outer element unchanged

shallow[1].append(5)
print(original)  # [1, [2, 3, 5], 4]  -- inner list IS shared!

# Deep copy (copies everything recursively)
import copy
original = [1, [2, 3], 4]
deep = copy.deepcopy(original)

deep[1].append(5)
print(original)  # [1, [2, 3], 4]  -- completely independent
print(deep)      # [1, [2, 3, 5], 4]
```

---

## Practical Examples

### Example 1: Unit Conversion

```python
# Convert miles to kilometers
miles = 26.2  # Marathon distance
km_per_mile = 1.60934

kilometers = miles * km_per_mile
print(f"{miles} miles = {kilometers:.2f} km")
# 26.2 miles = 42.16 km
```

### Example 2: Circle Calculations

```python
import math

radius = 5.0

circumference = 2 * math.pi * radius
area = math.pi * radius ** 2

print(f"Radius: {radius}")
print(f"Circumference: {circumference:.4f}")
print(f"Area: {area:.4f}")
# Radius: 5.0
# Circumference: 31.4159
# Area: 78.5398
```

### Example 3: Data Validation

```python
def validate_age(value):
    """Validate and convert an age value."""
    if value is None:
        return None, "Age is required."

    if isinstance(value, str):
        if not value.strip():
            return None, "Age cannot be empty."
        try:
            value = int(value)
        except ValueError:
            return None, f"'{value}' is not a valid number."

    if not isinstance(value, int):
        return None, f"Expected int, got {type(value).__name__}."

    if value < 0 or value > 150:
        return None, f"Age {value} is out of range (0-150)."

    return value, None

# Test the validator
test_cases = [25, "30", "abc", "", None, -5, 200, 3.14]
for test in test_cases:
    age, error = validate_age(test)
    if error:
        print(f"  {test!r:>10} -> ERROR: {error}")
    else:
        print(f"  {test!r:>10} -> OK: {age}")
```

Output:

```
          25 -> OK: 25
        '30' -> OK: 30
       'abc' -> ERROR: 'abc' is not a valid number.
          '' -> ERROR: Age cannot be empty.
        None -> ERROR: Age is required.
          -5 -> ERROR: Age -5 is out of range (0-150).
         200 -> ERROR: Age 200 is out of range (0-150).
        3.14 -> ERROR: Expected int, got float.
```

### Example 4: Variable Introspection

```python
def describe_variable(name, value):
    """Print detailed information about a variable."""
    print(f"Variable: {name}")
    print(f"  Value:    {value!r}")
    print(f"  Type:     {type(value).__name__}")
    print(f"  Bool:     {bool(value)}")
    print(f"  ID:       {id(value)}")
    print()

describe_variable("count", 42)
describe_variable("ratio", 3.14)
describe_variable("name", "Alice")
describe_variable("flag", True)
describe_variable("empty", None)
describe_variable("items", [1, 2, 3])
```

---

## Exercises

1. **Type Explorer**: Write a script that creates variables of each basic type (`int`, `float`, `complex`, `str`, `bool`, `None`) and prints the type and value of each.

2. **Swap Challenge**: Swap three variables `a`, `b`, `c` so that `a` gets `b`'s value, `b` gets `c`'s value, and `c` gets `a`'s value. Do it in one line.

3. **Precision Test**: Calculate `0.1 + 0.1 + 0.1 - 0.3`. Explain why the result is not exactly zero. Write code that correctly compares the result to zero.

4. **Type Converter**: Write a function `smart_convert(value)` that takes a string and returns the most specific type it can be converted to (try `int` first, then `float`, then return the original string).

5. **Constants File**: Create a module `constants.py` with at least 10 named constants following the `UPPER_CASE` convention. Import and use them in another script.

6. **Memory Detective**: Create two lists with identical contents. Prove they are equal (`==`) but not identical (`is`). Then create a situation where two names point to the same list and prove they are identical.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Variables** | Names that reference objects, not boxes that hold values |
| **Dynamic Typing** | Variables can change type; types belong to objects |
| **int** | Arbitrary-precision integers; supports binary, octal, hex literals |
| **float** | IEEE 754 double-precision; beware of precision issues |
| **complex** | Built-in complex number support with `j` notation |
| **str** | Immutable Unicode strings; f-strings for formatting |
| **bool** | `True`/`False`; subclass of `int`; every object has a truth value |
| **None** | Python's null; check with `is None`, not `== None` |
| **type() / isinstance()** | Prefer `isinstance()` for type checking (respects inheritance) |
| **Type Conversion** | `int()`, `float()`, `str()`, `bool()` for explicit conversion |
| **Naming** | `snake_case` for variables/functions, `UPPER_CASE` for constants |
| **Mutability** | Immutable types create new objects; mutable types can change in place |

---

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Operators and Expressions](./03_Operators_and_Expressions.md)

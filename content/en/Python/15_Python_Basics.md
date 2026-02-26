# Python Basics

**Previous**: [Pattern Matching](./14_Pattern_Matching.md) | **Next**: [OOP Basics](./16_OOP_Basics.md)

> **Note**: This lesson is for reviewing prerequisite knowledge. If you lack foundational knowledge before starting advanced lessons, study this content first.

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify and use Python's basic data types (int, float, str, bool, None) and perform type conversions
2. Apply arithmetic, comparison, logical, and membership operators in expressions
3. Manipulate strings using indexing, slicing, common methods, and f-string formatting
4. Write conditional statements (`if`/`elif`/`else`) and ternary expressions for branching logic
5. Implement `for` and `while` loops with `range`, `enumerate`, `zip`, and comprehensions
6. Define functions with default parameters, variable arguments (`*args`, `**kwargs`), and lambda expressions
7. Use lists, tuples, dictionaries, and sets with their key operations and understand their trade-offs
8. Handle exceptions with `try`/`except`/`else`/`finally` and define custom exception classes

---

Every advanced Python technique rests on a solid understanding of the language fundamentals. Variables, data structures, control flow, functions, and exception handling are the building blocks you will use in every program you write. Reviewing these basics ensures you have a firm foundation before tackling topics like decorators, generators, and metaclasses.

## 1. Variables and Data Types

### 1.1 Basic Data Types

```python
# integer (int)
age = 25
count = -10
big_number = 1_000_000  # underscore for readability

# float (float)
pi = 3.14159
temperature = -40.5
scientific = 2.5e-3  # 0.0025

# string (str)
name = "Alice"
message = '안녕하세요'
multiline = """여러 줄
문자열입니다"""

# boolean (bool)
is_active = True
is_empty = False

# None (no value)
result = None

# type check
print(type(age))        # <class 'int'>
print(type(pi))         # <class 'float'>
print(type(name))       # <class 'str'>
print(type(is_active))  # <class 'bool'>
```

### 1.2 Type Conversion

```python
# string → int/float
num_str = "123"
num_int = int(num_str)      # 123
num_float = float(num_str)  # 123.0

# number → string
age = 25
age_str = str(age)  # "25"

# boolean conversion
bool(0)       # False
bool(1)       # True
bool("")      # False (empty string)
bool("hello") # True
bool([])      # False (empty list)
bool([1, 2])  # True

# type conversion error handling
try:
    invalid = int("hello")
except ValueError as e:
    print(f"Conversion error: {e}")
```

### 1.3 Operators

```python
# arithmetic operators
a, b = 10, 3
print(a + b)   # 13 (addition)
print(a - b)   # 7 (subtraction)
print(a * b)   # 30 (multiplication)
print(a / b)   # 3.333... (division, always float)
print(a // b)  # 3 (integer division)
print(a % b)   # 1 (remainder)
print(a ** b)  # 1000 (exponentiation)

# comparison operators
print(5 == 5)   # True
print(5 != 3)   # True
print(5 > 3)    # True
print(5 >= 5)   # True
print(3 < 5)    # True
print(3 <= 3)   # True

# logical operators
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# membership operators
fruits = ["apple", "banana"]
print("apple" in fruits)      # True
print("orange" not in fruits) # True

# identity operators (object comparison)
a = [1, 2, 3]
b = [1, 2, 3]
c = a
print(a == b)  # True (value comparison)
print(a is b)  # False (object comparison)
print(a is c)  # True (same object)
```

---

## 2. String Processing

### 2.1 String Basics

```python
# string creation
s1 = "Hello"
s2 = 'World'
s3 = """Multi
line"""

# string concatenation
greeting = s1 + " " + s2  # "Hello World"

# string repetition
dashes = "-" * 10  # "----------"

# indexing (starts from 0)
text = "Python"
print(text[0])   # 'P'
print(text[-1])  # 'n' (last)

# slicing
print(text[0:3])   # 'Pyt' (0~2)
print(text[2:])    # 'thon' (from 2 to end)
print(text[:3])    # 'Pyt' (start~2)
print(text[::2])   # 'Pto' (every 2 steps)
print(text[::-1])  # 'nohtyP' (reversed)
```

### 2.2 String Methods

```python
text = "  Hello, World!  "

# remove whitespace
print(text.strip())   # "Hello, World!"
print(text.lstrip())  # "Hello, World!  "
print(text.rstrip())  # "  Hello, World!"

# case conversion
s = "Hello World"
print(s.upper())       # "HELLO WORLD"
print(s.lower())       # "hello world"
print(s.capitalize())  # "Hello world"
print(s.title())       # "Hello World"

# search
print(s.find("World"))     # 6 (index)
print(s.find("Python"))    # -1 (not found)
print(s.count("o"))        # 2
print(s.startswith("He"))  # True
print(s.endswith("!"))     # False

# split and join
csv = "a,b,c,d"
parts = csv.split(",")     # ['a', 'b', 'c', 'd']
joined = "-".join(parts)   # 'a-b-c-d'

# replace
text = "I like Python"
new_text = text.replace("Python", "Java")  # "I like Java"
```

### 2.3 Formatting

```python
name = "Alice"
age = 25
score = 95.5

# f-string (recommended, Python 3.6+)
print(f"Name: {name}, Age: {age}")
print(f"Score: {score:.2f}")  # 2 decimal places
print(f"Binary: {age:08b}")   # 8-digit binary, zero-padded

# format() method
print("Name: {}, Age: {}".format(name, age))
print("Name: {n}, Age: {a}".format(n=name, a=age))

# % operator (old style)
print("Name: %s, Age: %d" % (name, age))

# alignment
text = "Python"
print(f"{text:>10}")   # "    Python" (right-align)
print(f"{text:<10}")   # "Python    " (left-align)
print(f"{text:^10}")   # "  Python  " (center-align)
print(f"{text:*^10}")  # "**Python**" (padding character)
```

---

## 3. Control Flow

### 3.1 Conditionals (if)

```python
# basic if-elif-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Grade: {grade}")  # Grade: B

# ternary operator
status = "Pass" if score >= 60 else "Fail"

# condition chaining
age = 25
if 18 <= age < 65:
    print("Working age")

# truthiness evaluation
items = [1, 2, 3]
if items:  # True if not empty
    print("List has items")

# combining logical operators
x, y = 5, 10
if x > 0 and y > 0:
    print("Both positive")

if x < 0 or y < 0:
    print("At least one negative")
```

### 3.2 Loops (for)

```python
# iterate over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# using range
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8):     # 2, 3, 4, 5, 6, 7
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)

# enumerate (index and value)
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# zip (parallel iteration over multiple sequences)
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# dictionary iteration
person = {"name": "Alice", "age": 25}
for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

# list comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### 3.3 Loops (while)

```python
# basic while
count = 0
while count < 5:
    print(count)
    count += 1

# break and continue
for i in range(10):
    if i == 3:
        continue  # skip 3
    if i == 7:
        break     # stop at 7
    print(i)  # 0, 1, 2, 4, 5, 6

# while-else (runs when loop ends without break)
n = 7
i = 2
while i < n:
    if n % i == 0:
        print(f"{n} is not prime")
        break
    i += 1
else:
    print(f"{n} is prime")

# infinite loop
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break
```

---

## 4. Functions

### 4.1 Function Definition

```python
# basic function
def greet(name):
    """Returns a greeting message."""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Hello, Alice!

# returning multiple values
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)
print(f"Quotient: {q}, Remainder: {r}")  # Quotient: 3, Remainder: 1

# default parameter values
def power(base, exp=2):
    return base ** exp

print(power(3))     # 9
print(power(3, 3))  # 27

# keyword arguments
def create_user(name, age, city="Seoul"):
    return {"name": name, "age": age, "city": city}

user = create_user(name="Bob", age=30, city="Busan")
```

### 4.2 Variable Arguments

```python
# *args (positional arguments)
def sum_all(*args):
    """Sum an arbitrary number of numbers."""
    return sum(args)

print(sum_all(1, 2, 3))       # 6
print(sum_all(1, 2, 3, 4, 5)) # 15

# **kwargs (keyword arguments)
def print_info(**kwargs):
    """Print an arbitrary number of key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Seoul")

# mixed usage
def mixed_func(required, *args, **kwargs):
    print(f"Required: {required}")
    print(f"Extra positional: {args}")
    print(f"Extra keyword: {kwargs}")

mixed_func("hello", 1, 2, 3, x=10, y=20)
```

### 4.3 Lambda Functions

```python
# basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# multiple parameters
add = lambda a, b: a + b
print(add(3, 4))  # 7

# sorting with lambda
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]

# sort by score
sorted_students = sorted(students, key=lambda x: x["score"], reverse=True)
for s in sorted_students:
    print(f"{s['name']}: {s['score']}")

# with map and filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

---

## 5. Data Structures

### 5.1 Lists

```python
# create a list
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# element access
print(fruits[0])   # "apple"
print(fruits[-1])  # "cherry"

# slicing
print(numbers[1:4])  # [2, 3, 4]
print(numbers[::2])  # [1, 3, 5]

# modify element
fruits[0] = "apricot"

# add elements
fruits.append("date")           # append to end
fruits.insert(1, "blueberry")   # insert at position
fruits.extend(["elderberry"])   # extend with multiple items

# remove elements
fruits.remove("banana")  # remove by value
del fruits[0]            # remove by index
popped = fruits.pop()    # remove and return last element
fruits.clear()           # remove all elements

# list operations
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b      # [1, 2, 3, 4, 5, 6]
d = a * 2      # [1, 2, 3, 1, 2, 3]

# useful methods
nums = [3, 1, 4, 1, 5, 9, 2]
print(len(nums))        # 7
print(nums.count(1))    # 2
print(nums.index(4))    # 2
nums.sort()             # sort in-place
nums.reverse()          # reverse in-place
```

### 5.2 Tuples

```python
# create a tuple (immutable)
point = (3, 4)
rgb = (255, 128, 0)
single = (42,)  # single element (trailing comma required)

# unpacking
x, y = point
print(f"x={x}, y={y}")

# returning multiple values from a function
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

minimum, maximum, total = get_stats([1, 2, 3, 4, 5])

# tuples are immutable
# point[0] = 5  # TypeError!

# but mutable objects inside can be modified
data = ([1, 2], [3, 4])
data[0].append(3)  # allowed! ([1, 2, 3], [3, 4])

# tuple ↔ list conversion
t = tuple([1, 2, 3])
l = list((1, 2, 3))
```

### 5.3 Dictionaries

```python
# create a dictionary
person = {
    "name": "Alice",
    "age": 25,
    "city": "Seoul"
}

# element access
print(person["name"])          # "Alice"
print(person.get("job"))       # None (key not found)
print(person.get("job", "N/A")) # "N/A" (default value)

# add/update elements
person["job"] = "Engineer"  # add
person["age"] = 26          # update

# delete elements
del person["city"]
job = person.pop("job")
person.clear()

# methods
person = {"name": "Alice", "age": 25}
print(person.keys())    # dict_keys(['name', 'age'])
print(person.values())  # dict_values(['Alice', 25])
print(person.items())   # dict_items([('name', 'Alice'), ('age', 25)])

# dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# merge (Python 3.9+)
a = {"x": 1, "y": 2}
b = {"y": 3, "z": 4}
c = a | b  # {"x": 1, "y": 3, "z": 4}

# nested dictionary
users = {
    "user1": {"name": "Alice", "age": 25},
    "user2": {"name": "Bob", "age": 30}
}
print(users["user1"]["name"])  # "Alice"
```

### 5.4 Sets

```python
# create a set (no duplicates, unordered)
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 3, 2, 1}  # {1, 2, 3}
empty = set()  # empty set ({} creates a dict!)

# add/remove elements
fruits.add("date")
fruits.remove("apple")    # KeyError if not found
fruits.discard("grape")   # no error if not found

# set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # union: {1, 2, 3, 4, 5, 6}
print(a & b)  # intersection: {3, 4}
print(a - b)  # difference: {1, 2}
print(a ^ b)  # symmetric difference: {1, 2, 5, 6}

# subset check
c = {1, 2}
print(c.issubset(a))    # True
print(a.issuperset(c))  # True

# remove duplicates from a list
numbers = [1, 2, 2, 3, 3, 3]
unique = list(set(numbers))  # [1, 2, 3]
```

---

## 6. Exception Handling

### 6.1 Basic Exception Handling

```python
# try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# handling multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Not a number")
except ZeroDivisionError:
    print("Cannot divide by zero")

# accessing exception info
try:
    x = int("hello")
except ValueError as e:
    print(f"Error occurred: {e}")

# else and finally
try:
    file = open("data.txt", "r")
except FileNotFoundError:
    print("File not found")
else:
    # runs when no exception occurs
    content = file.read()
    file.close()
finally:
    # always runs
    print("Operation complete")
```

### 6.2 Raising Exceptions

```python
# raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age is too large")
    return age

try:
    validate_age(-5)
except ValueError as e:
    print(f"Validation failed: {e}")

# custom exception
class InsufficientFundsError(Exception):
    """Exception for insufficient balance."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Balance {balance}, withdrawal requested {amount}")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(1000, 2000)
except InsufficientFundsError as e:
    print(f"Withdrawal failed: {e}")
```

---

## 7. File I/O

### 7.1 File Reading/Writing

```python
# write to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("Hello there\n")

# read from file
with open("output.txt", "r", encoding="utf-8") as f:
    content = f.read()  # read entire file
    print(content)

# read line by line
with open("output.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())

# append to file
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("Additional content\n")

# file modes
# "r"  read (default)
# "w"  write (overwrite)
# "a"  append
# "x"  create (error if already exists)
# "b"  binary mode (e.g., "rb", "wb")
```

### 7.2 JSON Processing

```python
import json

# Python object → JSON
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding"]
}

# convert to JSON string
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)

# save to JSON file
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# read from JSON file
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
    print(loaded["name"])
```

---

## Summary

### Core Syntax Overview

| Concept | Description | Example |
|---------|-------------|---------|
| Variables | Name to store values | `x = 10` |
| Data Types | int, float, str, bool, None | `type(x)` |
| Lists | Ordered mutable sequence | `[1, 2, 3]` |
| Tuples | Ordered immutable sequence | `(1, 2, 3)` |
| Dictionaries | Key-value pairs | `{"a": 1}` |
| Sets | Collection without duplicates | `{1, 2, 3}` |
| if/elif/else | Conditional branching | `if x > 0:` |
| for | Iterate over sequence | `for i in range(10):` |
| while | Conditional loop | `while x < 10:` |
| Functions | Reusable code block | `def func():` |
| Exception Handling | Error handling | `try/except` |

---

## Exercises

### Exercise 1: Data Types and Type Conversion

Practice working with Python's core data types and conversion functions.

1. Create variables of each basic type (int, float, str, bool, None) and print their types using `type()`.
2. Write a function `safe_convert(value, target_type)` that attempts to convert `value` to `target_type` and returns `None` (instead of raising an exception) if the conversion fails. Test it with:
   - `safe_convert("42", int)` → 42
   - `safe_convert("hello", float)` → None
   - `safe_convert("3.14", float)` → 3.14
3. Explain the difference between `==` and `is`. Demonstrate with a list: create two lists with identical content (`a = [1, 2, 3]` and `b = [1, 2, 3]`) and show that `a == b` is `True` but `a is b` is `False`.
4. What is the truthiness of each of the following? Predict, then verify: `0`, `0.0`, `""`, `" "`, `[]`, `[0]`, `None`, `False`.
5. Write a one-liner using `bool` and a list comprehension that counts how many values in `[0, 1, "", "hello", None, [], [1]]` are truthy.

### Exercise 2: String Manipulation

Build a small text-processing toolkit using string methods and f-strings.

1. Write a function `normalize(s)` that: strips leading/trailing whitespace, converts to lowercase, and replaces all spaces with underscores. Test: `normalize("  Hello World  ")` → `"hello_world"`.
2. Write a function `mask_email(email)` that partially hides an email address. For `"alice@example.com"` return `"a***e@example.com"` (show first and last character of the local part, mask the middle with `*`).
3. Write a function `word_count(text)` that returns a dictionary mapping each unique word (case-insensitive) to its frequency. For `"the quick brown fox the fox"` return `{'the': 2, 'quick': 1, 'brown': 1, 'fox': 2}`.
4. Using f-string formatting, print a formatted table for the following data:
   ```python
   data = [("Alice", 92.5), ("Bob", 78.0), ("Charlie", 85.3)]
   ```
   Each row should be formatted as: `| Name       |  Score |` with the name left-aligned in a 10-character field and the score right-aligned with one decimal place.
5. Write a function `is_palindrome(s)` that returns `True` if `s` (ignoring case, spaces, and punctuation) is a palindrome. Test with `"A man, a plan, a canal: Panama"`.

### Exercise 3: Control Flow and Comprehensions

Practice conditionals, loops, and comprehensions to replace verbose imperative code.

1. Write a function `fizzbuzz(n)` that returns a list of strings from 1 to n: "FizzBuzz" for multiples of both 3 and 5, "Fizz" for multiples of 3, "Buzz" for multiples of 5, and the number as a string otherwise.
2. Rewrite the following loop as a single list comprehension:
   ```python
   result = []
   for x in range(20):
       if x % 2 == 0 and x % 3 != 0:
           result.append(x ** 2)
   ```
3. Write a dictionary comprehension that maps each word in a sentence to its length. For `"the quick brown fox"` produce `{'the': 3, 'quick': 5, 'brown': 5, 'fox': 3}`.
4. Write a nested list comprehension that produces a multiplication table as a list of lists: `[[i*j for j in range(1, 6)] for i in range(1, 6)]`. Print it in a formatted grid.
5. Use `zip` and `enumerate` to write a function `merge_ranked(names, scores)` that returns a list of tuples `(rank, name, score)` sorted by score descending. For `names=["Alice","Bob","Charlie"]` and `scores=[85, 92, 78]` return `[(1, "Bob", 92), (2, "Alice", 85), (3, "Charlie", 78)]`.

### Exercise 4: Functions and Argument Handling

Explore Python's flexible function argument mechanisms.

1. Write a function `stats(*numbers)` that accepts any number of numeric arguments and returns a tuple `(min, max, mean, median)`. Handle the edge case of an empty input gracefully.
2. Write a function `create_html_tag(tag, content, **attributes)` that generates an HTML string. For example:
   ```python
   create_html_tag("a", "Click here", href="https://example.com", class_="btn")
   # → '<a href="https://example.com" class="btn">Click here</a>'
   ```
   Note: `class_` should be rendered as `class` (strip the trailing underscore).
3. Rewrite the following using a lambda and `sorted()`:
   ```python
   students = [{"name": "Alice", "grade": "B", "score": 85},
               {"name": "Bob",   "grade": "A", "score": 92},
               {"name": "Charlie", "grade": "C", "score": 78}]
   # Sort by grade first (alphabetically), then by score descending
   ```
4. Write a decorator `memoize` (without using `functools.lru_cache`) that caches the results of a function. Apply it to a recursive Fibonacci function and compare the call count with and without memoization for `fib(30)`.
5. Explain the difference between mutable and immutable default arguments. Demonstrate the classic bug with `def append_to(item, lst=[])` and show how to fix it correctly.

### Exercise 5: Data Structures and Exception Handling

Combine data structure operations with robust error handling.

1. Write a function `group_by(items, key_func)` that groups items from a list into a dictionary by the return value of `key_func`. For example:
   ```python
   group_by([1, 2, 3, 4, 5, 6], lambda x: "even" if x % 2 == 0 else "odd")
   # → {"even": [2, 4, 6], "odd": [1, 3, 5]}
   ```
2. Implement a simple stack class using a list that supports `push(item)`, `pop()`, `peek()` (view top without removing), and `is_empty()`. Raise a custom `StackUnderflowError` when `pop()` or `peek()` is called on an empty stack.
3. Write a function `deep_merge(base, override)` that recursively merges two nested dictionaries. Keys in `override` overwrite `base`; nested dicts are merged recursively rather than replaced wholesale.
4. Write a function `read_csv_safe(filename)` that reads a CSV file and returns a list of dictionaries. It should:
   - Raise `FileNotFoundError` with a clear message if the file does not exist.
   - Skip malformed rows (rows with wrong number of columns) and print a warning.
   - Return an empty list if the file exists but is empty.
   - Use `finally` to ensure the file is always closed even if an exception occurs mid-read.
5. Use set operations to solve: given two lists of user IDs, `active_users` and `premium_users`, compute and print:
   - Users who are both active and premium (intersection)
   - Active users who are NOT premium (difference)
   - Users who are either active or premium but not both (symmetric difference)
   - The total number of distinct users across both lists (union length)

---

## References

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [Python Cheat Sheet](https://www.pythoncheatsheet.org/)

---

**Previous**: [Pattern Matching](./14_Pattern_Matching.md) | **Next**: [OOP Basics](./16_OOP_Basics.md)

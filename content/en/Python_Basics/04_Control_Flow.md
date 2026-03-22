# Control Flow

**Previous**: [Operators and Expressions](./03_Operators_and_Expressions.md) | **Next**: [Functions](./05_Functions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write conditional logic using `if`, `elif`, and `else` to branch program execution
2. Use conditional expressions (ternary operator) for concise inline decisions
3. Iterate over sequences and ranges with `for` loops, including `enumerate()` and `zip()`
4. Build `while` loops for indefinite iteration and apply sentinel loop patterns
5. Control loop execution with `break`, `continue`, and `pass`
6. Use the `for/else` and `while/else` constructs for search-and-found patterns
7. Apply structural pattern matching (`match/case`) introduced in Python 3.10
8. Recognize and implement common control flow patterns: accumulators, flags, nested loops, and early returns

---

Control flow determines the order in which statements execute. Without it, a program would run top-to-bottom in a straight line. Conditional statements let programs make decisions, and loops let them repeat actions. Together, they form the backbone of every non-trivial program.

## Conditional Statements

### if Statement

The `if` statement executes a block of code only when a condition is true.

```python
temperature = 35

if temperature > 30:
    print("It's hot outside!")
    print("Stay hydrated.")

# Output:
# It's hot outside!
# Stay hydrated.
```

Key points:
- The condition can be any expression that evaluates to a truthy or falsy value.
- The colon (`:`) is required after the condition.
- The body is indented by 4 spaces (Python convention).

### if/else

```python
age = 16

if age >= 18:
    print("You can vote.")
else:
    print("You cannot vote yet.")
    years_left = 18 - age
    print(f"Wait {years_left} more year(s).")

# Output:
# You cannot vote yet.
# Wait 2 more year(s).
```

### if/elif/else

`elif` (short for "else if") lets you check multiple conditions in sequence. Python evaluates them top-to-bottom and executes the first matching block.

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score: {score}, Grade: {grade}")
# Score: 85, Grade: B
```

Important behavior:
- Only the **first** matching branch executes — subsequent `elif` and `else` blocks are skipped entirely.
- The `else` block is optional and acts as a catch-all.
- There is no limit to the number of `elif` branches.

### Nested Conditionals

```python
def classify_triangle(a, b, c):
    """Classify a triangle by its side lengths."""
    # First, check if it is a valid triangle
    if a + b <= c or b + c <= a or a + c <= b:
        return "Not a valid triangle"
    else:
        # Then classify by side equality
        if a == b == c:
            return "Equilateral"
        elif a == b or b == c or a == c:
            return "Isosceles"
        else:
            return "Scalene"

print(classify_triangle(3, 3, 3))   # Equilateral
print(classify_triangle(3, 3, 5))   # Isosceles
print(classify_triangle(3, 4, 5))   # Scalene
print(classify_triangle(1, 2, 10))  # Not a valid triangle
```

> **Tip**: Deeply nested conditionals (more than 2-3 levels) are a code smell. Refactor using early returns, guard clauses, or helper functions.

### Guard Clauses (Early Return Pattern)

Guard clauses flatten nested conditionals by handling edge cases first:

```python
# DEEP NESTING (harder to read)
def process_order(order):
    if order is not None:
        if order.items:
            if order.payment_valid:
                # ... process the order ...
                return "Order processed"
            else:
                return "Invalid payment"
        else:
            return "No items in order"
    else:
        return "No order provided"

# GUARD CLAUSES (flat and clear)
def process_order(order):
    if order is None:
        return "No order provided"
    if not order.items:
        return "No items in order"
    if not order.payment_valid:
        return "Invalid payment"

    # Happy path -- process the order
    return "Order processed"
```

### Conditional Expressions (Ternary Operator)

A compact way to choose between two values based on a condition:

```python
# Syntax: value_if_true if condition else value_if_false

age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # "adult"

# In function calls
print("even" if 42 % 2 == 0 else "odd")  # "even"

# In assignments
discount = 0.2 if is_member else 0.0
max_val = a if a > b else b

# Nested (use sparingly)
sign = "positive" if x > 0 else "zero" if x == 0 else "negative"
```

### Truthiness in Conditions

Python evaluates non-boolean values for truthiness (see Lesson 02):

```python
name = input("Enter your name: ")

# This works because empty strings are falsy
if name:
    print(f"Hello, {name}!")
else:
    print("You didn't enter a name.")

# Common truthiness checks
items = [1, 2, 3]
if items:          # True (non-empty list)
    print(f"{len(items)} items found")

data = {}
if not data:       # True (empty dict is falsy)
    print("No data available")
```

---

## for Loops

A `for` loop iterates over the items of any **iterable** (list, tuple, string, range, dict, set, file, generator, etc.).

### Basic for Loop

```python
# Iterate over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
# apple
# banana
# cherry

# Iterate over a string
for char in "Python":
    print(char, end=" ")
# P y t h o n

# Iterate over a dictionary (iterates over keys by default)
scores = {"Alice": 95, "Bob": 87, "Charlie": 92}
for name in scores:
    print(f"{name}: {scores[name]}")

# Iterate over dictionary items (key-value pairs)
for name, score in scores.items():
    print(f"{name}: {score}")

# Iterate over dictionary values only
for score in scores.values():
    print(score)
```

### The `range()` Function

`range()` generates a sequence of integers. It is commonly used with `for` loops when you need a numeric counter.

```python
# range(stop) -- 0 to stop-1
for i in range(5):
    print(i, end=" ")
# 0 1 2 3 4

# range(start, stop) -- start to stop-1
for i in range(2, 7):
    print(i, end=" ")
# 2 3 4 5 6

# range(start, stop, step)
for i in range(0, 20, 3):
    print(i, end=" ")
# 0 3 6 9 12 15 18

# Counting down
for i in range(10, 0, -1):
    print(i, end=" ")
# 10 9 8 7 6 5 4 3 2 1

# range is lazy -- it doesn't create a list in memory
r = range(1_000_000_000)  # Uses almost no memory
print(999_999 in r)        # True (O(1) membership test)
print(len(r))              # 1000000000
```

### `enumerate()` — Index and Value Together

When you need both the index and the value, use `enumerate()` instead of manual indexing:

```python
# BAD: manual indexing
fruits = ["apple", "banana", "cherry"]
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

# GOOD: enumerate
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Start counting from a different number
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry

# Practical: find the index of an item
def find_index(items, target):
    """Find the index of the first occurrence of target."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

print(find_index(["a", "b", "c", "d"], "c"))  # 2
print(find_index(["a", "b", "c", "d"], "z"))  # -1
```

### `zip()` — Iterate Over Multiple Sequences

`zip()` pairs up elements from two or more iterables:

```python
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
# Alice is 30 years old
# Bob is 25 years old
# Charlie is 35 years old

# zip stops at the shortest iterable
short = [1, 2]
long = [10, 20, 30, 40]
for a, b in zip(short, long):
    print(a, b)
# 1 10
# 2 20
# (30 and 40 are silently dropped)

# Use itertools.zip_longest to include all items
from itertools import zip_longest
for a, b in zip_longest(short, long, fillvalue=0):
    print(a, b)
# 1 10
# 2 20
# 0 30
# 0 40

# zip with three or more iterables
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]
cities = ["NYC", "LA", "Chicago"]

for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, from {city}")

# Transposing with zip
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
transposed = list(zip(*matrix))
print(transposed)
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

# Creating a dictionary from two lists
keys = ["name", "age", "city"]
values = ["Alice", 30, "NYC"]
person = dict(zip(keys, values))
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}
```

### Iterating Over Multiple Collections — Practical Example

```python
# Grade report for a class
students = ["Alice", "Bob", "Charlie", "Diana"]
midterms = [88, 76, 92, 85]
finals = [91, 82, 88, 90]

print(f"{'Student':<10} {'Midterm':>7} {'Final':>7} {'Average':>7} {'Grade':>5}")
print("-" * 42)

for i, (name, mid, final) in enumerate(zip(students, midterms, finals), 1):
    avg = (mid + final) / 2
    grade = "A" if avg >= 90 else "B" if avg >= 80 else "C" if avg >= 70 else "F"
    print(f"{name:<10} {mid:>7} {final:>7} {avg:>7.1f} {grade:>5}")

# Output:
# Student    Midterm   Final Average Grade
# ------------------------------------------
# Alice           88      91    89.5     B
# Bob             76      82    79.0     C
# Charlie         92      88    90.0     A
# Diana           85      90    87.5     B
```

---

## while Loops

A `while` loop repeats as long as its condition is true.

### Basic while Loop

```python
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1
# Count: 0
# Count: 1
# Count: 2
# Count: 3
# Count: 4
```

### Sentinel Loop (Loop Until a Special Value)

```python
# Read numbers until the user enters 0
total = 0
while True:
    value = int(input("Enter a number (0 to stop): "))
    if value == 0:
        break
    total += value

print(f"Total: {total}")
```

### Input Validation Loop

```python
# Keep asking until valid input is received
while True:
    try:
        age = int(input("Enter your age (1-120): "))
        if 1 <= age <= 120:
            break
        print("Age must be between 1 and 120.")
    except ValueError:
        print("Please enter a valid number.")

print(f"Your age is {age}.")
```

### Countdown Pattern

```python
def countdown(n):
    """Print a countdown from n to 1, then 'Go!'."""
    while n > 0:
        print(n, end=" ")
        n -= 1
    print("Go!")

countdown(5)
# 5 4 3 2 1 Go!
```

### Convergence Loop (Numerical Methods)

```python
# Newton's method for square root
def sqrt_newton(n, tolerance=1e-10):
    """Compute square root of n using Newton's method."""
    if n < 0:
        raise ValueError("Cannot compute square root of negative number")
    if n == 0:
        return 0.0

    guess = n / 2.0
    iterations = 0

    while True:
        new_guess = (guess + n / guess) / 2
        iterations += 1

        if abs(new_guess - guess) < tolerance:
            print(f"Converged in {iterations} iterations")
            return new_guess

        guess = new_guess

print(sqrt_newton(2))
# Converged in 36 iterations
# 1.4142135623730951

import math
print(math.sqrt(2))
# 1.4142135623730951
```

---

## Loop Control Statements

### `break` — Exit the Loop Immediately

```python
# Find the first negative number
numbers = [4, 7, 2, -3, 8, -1, 5]
for num in numbers:
    if num < 0:
        print(f"First negative number: {num}")
        break
else:
    # This else belongs to the for loop (see below)
    print("No negative numbers found")
# First negative number: -3

# break only exits the innermost loop
for i in range(3):
    for j in range(3):
        if j == 1:
            break        # Exits inner loop only
        print(f"i={i}, j={j}")
# i=0, j=0
# i=1, j=0
# i=2, j=0
```

### `continue` — Skip to the Next Iteration

```python
# Print only even numbers
for i in range(10):
    if i % 2 != 0:
        continue   # Skip odd numbers
    print(i, end=" ")
# 0 2 4 6 8

# Skip blank lines when processing text
lines = ["Hello", "", "World", "", "", "Python"]
for line in lines:
    if not line:
        continue
    print(f"Processing: {line}")
# Processing: Hello
# Processing: World
# Processing: Python
```

### `pass` — Do Nothing (Placeholder)

```python
# pass is a no-op statement used as a placeholder
for i in range(10):
    if i < 5:
        pass  # TODO: handle small numbers
    else:
        print(i)

# Common uses of pass
class MyError(Exception):
    pass  # Empty class body

def not_implemented_yet():
    pass  # Placeholder for future implementation

if condition:
    pass  # Deliberately empty block
```

### `break` vs `continue` vs `pass`

```python
# Comparison with the same loop
numbers = [1, 2, 3, 4, 5]

print("break:")
for n in numbers:
    if n == 3:
        break         # Stop the loop entirely
    print(n, end=" ")
# 1 2

print("\ncontinue:")
for n in numbers:
    if n == 3:
        continue      # Skip this iteration
    print(n, end=" ")
# 1 2 4 5

print("\npass:")
for n in numbers:
    if n == 3:
        pass          # Do nothing (print still executes)
    print(n, end=" ")
# 1 2 3 4 5
```

---

## Loop-Else Clause

Python's unique `for/else` and `while/else` construct runs the `else` block only if the loop completes **without hitting a `break`**.

### for/else — Search Pattern

```python
# Check if a list contains a prime number
def has_prime(numbers):
    """Check if any number in the list is prime."""
    for num in numbers:
        if num < 2:
            continue
        for divisor in range(2, int(num ** 0.5) + 1):
            if num % divisor == 0:
                break   # Not prime, break inner loop
        else:
            # Inner loop completed without break -> num is prime
            print(f"Found prime: {num}")
            return True
    return False

print(has_prime([4, 6, 8, 9, 11, 15]))  # Found prime: 11 -> True
print(has_prime([4, 6, 8, 9, 15]))      # False
```

### for/else — Finding an Item

```python
# Search for a target value
def find_user(users, target_name):
    """Find a user by name."""
    for user in users:
        if user["name"] == target_name:
            print(f"Found: {user}")
            break
    else:
        # Only runs if loop completed without break (not found)
        print(f"User '{target_name}' not found")

users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]

find_user(users, "Bob")      # Found: {'name': 'Bob', 'age': 25}
find_user(users, "Diana")    # User 'Diana' not found
```

### while/else

```python
# while/else works the same way
def find_factor(n):
    """Find the smallest factor of n greater than 1."""
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            print(f"Smallest factor of {n}: {divisor}")
            break
        divisor += 1
    else:
        # Loop completed without break -> n is prime
        print(f"{n} is prime")

find_factor(91)   # Smallest factor of 91: 7
find_factor(97)   # 97 is prime
```

### Understanding Loop-Else

```
for/while ...
    if condition:
        break        -> else block is SKIPPED
                     -> else block RUNS
else:
    ...

Mental model: "else" means "no break"
- If break executed  -> else is skipped
- If loop completed normally -> else runs
- If loop body never executed (empty iterable) -> else runs
```

```python
# Empty iterable: else still runs
for item in []:
    print("This never prints")
else:
    print("Else runs because loop body never executed")
# Output: Else runs because loop body never executed
```

---

## Nested Loops

Loops can be placed inside other loops.

### Basic Nested Loop

```python
# Multiplication table
print("Multiplication Table (1-5)")
print("   ", end="")
for j in range(1, 6):
    print(f"{j:4d}", end="")
print()
print("-" * 24)

for i in range(1, 6):
    print(f"{i:2d}|", end="")
    for j in range(1, 6):
        print(f"{i*j:4d}", end="")
    print()

# Output:
#      1   2   3   4   5
# ------------------------
#  1|   1   2   3   4   5
#  2|   2   4   6   8  10
#  3|   3   6   9  12  15
#  4|   4   8  12  16  20
#  5|   5  10  15  20  25
```

### Pattern: Finding Pairs

```python
# Find all pairs that sum to a target
def find_pairs(numbers, target):
    """Find all pairs in numbers that sum to target."""
    pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                pairs.append((numbers[i], numbers[j]))
    return pairs

nums = [1, 3, 5, 7, 9, 2, 4, 6, 8]
print(find_pairs(nums, 10))
# [(1, 9), (3, 7), (2, 8), (4, 6)]
```

### Pattern: Matrix Operations

```python
# Create and manipulate a 2D grid
rows, cols = 3, 4

# Create a matrix using nested loops
matrix = []
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(i * cols + j + 1)
    matrix.append(row)

# Print the matrix
for row in matrix:
    for val in row:
        print(f"{val:4d}", end="")
    print()
#    1   2   3   4
#    5   6   7   8
#    9  10  11  12

# Same thing with list comprehension (covered in Lesson 06)
matrix = [[i * cols + j + 1 for j in range(cols)] for i in range(rows)]
```

### Breaking Out of Nested Loops

```python
# Method 1: Use a flag variable
found = False
for i in range(10):
    for j in range(10):
        if i * j == 42:
            print(f"Found: {i} * {j} = 42")
            found = True
            break
    if found:
        break

# Method 2: Use a function with return (preferred)
def find_product(target):
    """Find two numbers whose product equals target."""
    for i in range(1, target + 1):
        for j in range(1, target + 1):
            if i * j == target:
                return i, j
    return None

result = find_product(42)
if result:
    print(f"Found: {result[0]} * {result[1]} = 42")

# Method 3: Use itertools.product
from itertools import product
for i, j in product(range(10), range(10)):
    if i * j == 42:
        print(f"Found: {i} * {j} = 42")
        break
```

---

## Structural Pattern Matching (match/case)

Introduced in Python 3.10, `match/case` provides powerful pattern matching that goes beyond simple value comparison.

### Basic Value Matching

```python
def http_status_message(code):
    """Return a human-readable message for an HTTP status code."""
    match code:
        case 200:
            return "OK"
        case 201:
            return "Created"
        case 301:
            return "Moved Permanently"
        case 400:
            return "Bad Request"
        case 401:
            return "Unauthorized"
        case 403:
            return "Forbidden"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:
            return f"Unknown status code: {code}"

print(http_status_message(200))  # OK
print(http_status_message(404))  # Not Found
print(http_status_message(999))  # Unknown status code: 999
```

### OR Patterns

```python
def classify_char(ch):
    """Classify a character."""
    match ch:
        case 'a' | 'e' | 'i' | 'o' | 'u':
            return "lowercase vowel"
        case 'A' | 'E' | 'I' | 'O' | 'U':
            return "uppercase vowel"
        case _ if ch.isalpha():
            return "consonant"
        case _ if ch.isdigit():
            return "digit"
        case _:
            return "other"

print(classify_char('a'))  # lowercase vowel
print(classify_char('B'))  # consonant
print(classify_char('5'))  # digit
print(classify_char('!'))  # other
```

### Sequence Patterns

```python
def process_command(command):
    """Process a command given as a list of strings."""
    match command:
        case ["quit"]:
            return "Exiting..."
        case ["hello", name]:
            return f"Hello, {name}!"
        case ["add", x, y]:
            return f"Result: {int(x) + int(y)}"
        case ["move", direction, distance]:
            return f"Moving {direction} by {distance} units"
        case ["move", direction]:
            return f"Moving {direction} by 1 unit"
        case [action, *args]:
            return f"Unknown command: {action} with args {args}"
        case _:
            return "Invalid command"

print(process_command(["hello", "Alice"]))      # Hello, Alice!
print(process_command(["add", "3", "5"]))        # Result: 8
print(process_command(["move", "north", "10"]))  # Moving north by 10 units
print(process_command(["move", "south"]))         # Moving south by 1 unit
print(process_command(["quit"]))                  # Exiting...
print(process_command(["dance", "fast", "now"]))  # Unknown command: dance with args ['fast', 'now']
```

### Mapping Patterns (Dictionaries)

```python
def process_event(event):
    """Process an event dictionary."""
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": "scroll", "direction": direction, "amount": amount}:
            return f"Scroll {direction} by {amount}"
        case {"type": event_type}:
            return f"Unknown event type: {event_type}"
        case _:
            return "Invalid event"

print(process_event({"type": "click", "x": 100, "y": 200}))
# Click at (100, 200)

print(process_event({"type": "keypress", "key": "Enter"}))
# Key pressed: Enter

print(process_event({"type": "resize", "width": 800, "height": 600}))
# Unknown event type: resize
```

### Guard Clauses in Patterns

```python
def categorize_number(n):
    """Categorize a number with guards."""
    match n:
        case n if n < 0:
            return "negative"
        case 0:
            return "zero"
        case n if n % 2 == 0:
            return "positive even"
        case _:
            return "positive odd"

print(categorize_number(-5))   # negative
print(categorize_number(0))    # zero
print(categorize_number(4))    # positive even
print(categorize_number(7))    # positive odd
```

### Class Patterns

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point

def describe_shape(shape):
    """Describe a geometric shape using pattern matching."""
    match shape:
        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle at origin with radius {r}"
        case Circle(center=Point(x=x, y=y), radius=r):
            return f"Circle at ({x}, {y}) with radius {r}"
        case Rectangle(
            top_left=Point(x=x1, y=y1),
            bottom_right=Point(x=x2, y=y2)
        ):
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return f"Rectangle {width}x{height}"
        case _:
            return "Unknown shape"

print(describe_shape(Circle(Point(0, 0), 5)))
# Circle at origin with radius 5

print(describe_shape(Circle(Point(3, 4), 2)))
# Circle at ({3}, {4}) with radius 2

print(describe_shape(Rectangle(Point(0, 10), Point(5, 0))))
# Rectangle 5x10
```

---

## Common Control Flow Patterns

### Pattern 1: Accumulator

Collect a result over multiple iterations.

```python
# Sum accumulator
numbers = [3, 7, 2, 8, 4, 1, 9, 5]
total = 0
for num in numbers:
    total += num
print(f"Sum: {total}")  # Sum: 39

# Better: use built-in sum()
print(f"Sum: {sum(numbers)}")  # Sum: 39

# String accumulator
words = ["Hello", "beautiful", "world"]
sentence = ""
for word in words:
    sentence += word + " "
print(sentence.strip())  # Hello beautiful world

# Better: use str.join()
print(" ".join(words))   # Hello beautiful world

# List accumulator
def get_evens(numbers):
    """Return a list of even numbers."""
    evens = []
    for num in numbers:
        if num % 2 == 0:
            evens.append(num)
    return evens

# Better: list comprehension
evens = [n for n in numbers if n % 2 == 0]

# Max/min accumulator
def find_max(numbers):
    """Find the maximum value in a list."""
    if not numbers:
        raise ValueError("Empty list")
    current_max = numbers[0]
    for num in numbers[1:]:
        if num > current_max:
            current_max = num
    return current_max

# Better: use built-in max()
print(max(numbers))
```

### Pattern 2: Flag Variable

Use a boolean to track whether a condition was ever met.

```python
def has_duplicate(items):
    """Check if a list contains any duplicates."""
    seen = set()
    for item in items:
        if item in seen:
            return True  # Early return (even better than a flag)
        seen.add(item)
    return False

print(has_duplicate([1, 2, 3, 4, 5]))     # False
print(has_duplicate([1, 2, 3, 2, 5]))     # True

# Flag pattern when you need to process ALL items
def validate_data(records):
    """Validate all records, collecting all errors."""
    all_valid = True
    errors = []

    for i, record in enumerate(records):
        if not record.get("name"):
            all_valid = False
            errors.append(f"Record {i}: missing name")
        if not isinstance(record.get("age"), int):
            all_valid = False
            errors.append(f"Record {i}: invalid age")

    return all_valid, errors

records = [
    {"name": "Alice", "age": 30},
    {"name": "", "age": 25},
    {"name": "Charlie", "age": "old"},
]
valid, errors = validate_data(records)
print(f"Valid: {valid}")
for err in errors:
    print(f"  {err}")
```

### Pattern 3: Sentinel Value

Use a special value to signal termination.

```python
# Reading until EOF marker
def read_until_eof(lines):
    """Process lines until EOF marker."""
    results = []
    for line in lines:
        if line.strip() == "EOF":
            break
        results.append(line.strip())
    return results

data = ["Hello", "World", "EOF", "This is ignored"]
print(read_until_eof(data))
# ['Hello', 'World']
```

### Pattern 4: Sliding Window

Process elements in overlapping groups.

```python
# Sliding window of size k
def sliding_window_max(numbers, k):
    """Find the maximum in each window of size k."""
    if len(numbers) < k:
        return []

    results = []
    for i in range(len(numbers) - k + 1):
        window = numbers[i:i + k]
        results.append(max(window))
    return results

data = [1, 3, -1, -3, 5, 3, 6, 7]
print(sliding_window_max(data, 3))
# [3, 3, 5, 5, 6, 7]
```

### Pattern 5: Two Pointers

Use two indices to traverse from both ends.

```python
def is_palindrome(s):
    """Check if a string is a palindrome (ignoring case and non-alpha)."""
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    left = 0
    right = len(cleaned) - 1

    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1

    return True

print(is_palindrome("racecar"))                # True
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("hello"))                  # False
```

### Pattern 6: State Machine

Use a variable to track the current state.

```python
def tokenize_csv_line(line):
    """Simple CSV tokenizer that handles quoted fields."""
    tokens = []
    current = ""
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            tokens.append(current.strip())
            current = ""
        else:
            current += char

    tokens.append(current.strip())  # Don't forget the last field
    return tokens

line = 'Alice,30,"New York, NY",Engineer'
print(tokenize_csv_line(line))
# ['Alice', '30', 'New York, NY', 'Engineer']
```

---

## Performance Tips

### Choose the Right Loop Construct

```python
# Use built-in functions when possible (implemented in C, much faster)
numbers = list(range(1000))

# BAD: manual loop
total = 0
for n in numbers:
    total += n

# GOOD: built-in function
total = sum(numbers)

# BAD: manual search
found = False
for n in numbers:
    if n == 500:
        found = True
        break

# GOOD: use 'in' operator
found = 500 in numbers

# BAD: building a list with a loop
squares = []
for n in numbers:
    squares.append(n ** 2)

# GOOD: list comprehension (faster and more readable)
squares = [n ** 2 for n in numbers]
```

### Avoid Repeated Attribute Lookups

```python
# BAD: repeated attribute lookup in loop
import math
for i in range(10000):
    x = math.sqrt(i)

# GOOD: cache the function reference
sqrt = math.sqrt
for i in range(10000):
    x = sqrt(i)
```

---

## Exercises

1. **FizzBuzz**: Print numbers from 1 to 100. For multiples of 3, print "Fizz"; for multiples of 5, print "Buzz"; for multiples of both, print "FizzBuzz".

2. **Number Guessing Game**: Generate a random number between 1 and 100. Let the user guess, providing "too high" or "too low" hints. Count the number of guesses.

3. **Prime Sieve**: Implement the Sieve of Eratosthenes to find all prime numbers up to N. Use nested loops and a boolean list.

4. **Triangle Printer**: Write a function that prints a right triangle of asterisks with a given height:
   ```
   *
   **
   ***
   ****
   *****
   ```

5. **Password Validator**: Write a loop that repeatedly asks for a password until it meets all criteria: at least 8 characters, at least one uppercase letter, at least one lowercase letter, at least one digit, and at least one special character.

6. **Collatz Conjecture**: For any positive integer n, if n is even divide by 2, if n is odd multiply by 3 and add 1. Repeat until you reach 1. Print the sequence and the number of steps.

7. **Pattern Matching Practice** (requires Python 3.10+): Write a function that processes a list of "commands" like `["move", "north"]`, `["attack", "dragon"]`, `["use", "potion", "health"]`, `["quit"]` using `match/case`.

8. **Matrix Spiral**: Given an NxN matrix, print its elements in spiral order (outer ring first, then inner rings).

---

## Summary

| Construct | Purpose | Key Points |
|-----------|---------|------------|
| **if/elif/else** | Branching | First matching branch wins; else is optional |
| **Ternary** | Inline conditional | `x if cond else y`; avoid deep nesting |
| **for** | Definite iteration | Iterates over any iterable; use `enumerate()` for indices |
| **while** | Indefinite iteration | Repeats while condition is true; watch for infinite loops |
| **range()** | Integer sequence | `range(stop)`, `range(start, stop)`, `range(start, stop, step)` |
| **enumerate()** | Index + value | `enumerate(iterable, start=0)` |
| **zip()** | Parallel iteration | Stops at shortest; use `zip_longest` for uneven lengths |
| **break** | Exit loop | Exits innermost loop only |
| **continue** | Skip iteration | Jumps to next iteration of innermost loop |
| **pass** | No-op placeholder | Empty block that does nothing |
| **for/else** | No-break detection | Else runs only if loop completed without break |
| **match/case** | Pattern matching | Python 3.10+; supports values, sequences, mappings, guards |
| **Guard clauses** | Early returns | Flatten nested conditionals for readability |
| **Accumulator** | Collect results | Prefer built-ins: `sum()`, `max()`, `min()`, `"".join()` |

---

**Previous**: [Operators and Expressions](./03_Operators_and_Expressions.md) | **Next**: [Functions](./05_Functions.md)

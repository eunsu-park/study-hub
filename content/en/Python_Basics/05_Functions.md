# Functions

**Previous**: [Control Flow](./04_Control_Flow.md) | **Next**: [Data Structures](./06_Data_Structures.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define functions using the `def` keyword with parameters, return values, and docstrings
2. Distinguish between positional arguments, keyword arguments, default parameters, and variable-length arguments (`*args`, `**kwargs`)
3. Explain scope rules in Python, including local, global, and enclosing scopes (LEGB rule)
4. Write lambda expressions for short anonymous functions and use them with higher-order functions
5. Apply essential built-in functions (`len`, `max`, `min`, `sum`, `sorted`, `map`, `filter`, `zip`, `enumerate`, `any`, `all`)
6. Implement basic recursive functions and understand their call stack behavior
7. Create nested functions and understand closure basics
8. Write well-documented functions following Python docstring conventions

---

Functions are the fundamental building blocks of organized, reusable Python code. Instead of writing the same logic over and over, you encapsulate it in a function, give it a meaningful name, and call it whenever you need it. Functions make programs easier to read, test, debug, and maintain. Every serious Python program -- from a simple script to a large web application -- is built out of functions working together.

## 1. Defining Functions

A function is created with the `def` keyword, followed by the function name, parentheses for parameters, and a colon. The indented block beneath is the function body.

```python
# Basic function definition
def greet():
    print("Hello, World!")

# Calling the function
greet()  # Output: Hello, World!
```

### Function with Parameters

Parameters are placeholders for values the function needs to do its work. When you call the function, you pass arguments that fill those placeholders.

```python
def greet_user(name):
    print(f"Hello, {name}!")

greet_user("Alice")   # Output: Hello, Alice!
greet_user("Bob")     # Output: Hello, Bob!
```

### Multiple Parameters

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(3, 5)       # Output: 3 + 5 = 8
add(10, 20)     # Output: 10 + 20 = 30
```

### The Difference Between Parameters and Arguments

- **Parameters** are the names listed in the function definition
- **Arguments** are the actual values passed when calling the function

```python
def multiply(x, y):    # x and y are parameters
    return x * y

result = multiply(3, 4) # 3 and 4 are arguments
print(result)            # Output: 12
```

---

## 2. Return Values

Functions can send results back to the caller using the `return` statement. Without an explicit `return`, a function returns `None`.

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)       # Output: 8

# A function without return gives None
def greet(name):
    print(f"Hello, {name}!")

value = greet("Alice")   # Output: Hello, Alice!
print(value)             # Output: None
```

### Returning Multiple Values

Python functions can return multiple values as a tuple.

```python
def divide_and_remainder(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide_and_remainder(17, 5)
print(f"Quotient: {q}, Remainder: {r}")  # Output: Quotient: 3, Remainder: 2

# The return is actually a tuple
result = divide_and_remainder(17, 5)
print(result)        # Output: (3, 2)
print(type(result))  # Output: <class 'tuple'>
```

### Early Return

`return` immediately exits the function. Code after `return` in the same block will not execute.

```python
def absolute_value(n):
    if n >= 0:
        return n
    return -n

print(absolute_value(5))    # Output: 5
print(absolute_value(-3))   # Output: 3


def find_first_negative(numbers):
    for num in numbers:
        if num < 0:
            return num
    return None   # No negative found

data = [3, 7, -2, 5, -8]
print(find_first_negative(data))  # Output: -2
print(find_first_negative([1, 2, 3]))  # Output: None
```

### Returning Different Types

While Python allows returning different types, it is best practice to keep return types consistent.

```python
def safe_divide(a, b):
    if b == 0:
        return None   # Signals an error condition
    return a / b

result = safe_divide(10, 3)
if result is not None:
    print(f"Result: {result:.2f}")  # Output: Result: 3.33
```

---

## 3. Default Parameters

Default parameter values let you call a function without providing every argument.

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")               # Output: Hello, Alice!
greet("Bob", "Good morning") # Output: Good morning, Bob!
```

### Multiple Defaults

```python
def create_profile(name, age=0, city="Unknown", active=True):
    return {
        "name": name,
        "age": age,
        "city": city,
        "active": active,
    }

print(create_profile("Alice"))
# Output: {'name': 'Alice', 'age': 0, 'city': 'Unknown', 'active': True}

print(create_profile("Bob", 25, "Seoul"))
# Output: {'name': 'Bob', 'age': 25, 'city': 'Seoul', 'active': True}
```

### Mutable Default Argument Pitfall

A common mistake is using a mutable object (like a list) as a default value. The default is created once and shared across all calls.

```python
# BAD: mutable default argument
def add_item_bad(item, items=[]):
    items.append(item)
    return items

print(add_item_bad("a"))  # Output: ['a']
print(add_item_bad("b"))  # Output: ['a', 'b']  -- Unexpected!

# GOOD: use None as default, create new list inside
def add_item_good(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_good("a"))  # Output: ['a']
print(add_item_good("b"))  # Output: ['b']  -- Correct!
```

---

## 4. Keyword Arguments

When calling a function, you can specify arguments by name. This makes calls clearer and allows you to pass arguments in any order.

```python
def describe_pet(animal, name, age):
    print(f"{name} is a {age}-year-old {animal}.")

# Positional arguments (order matters)
describe_pet("dog", "Rex", 5)

# Keyword arguments (order does not matter)
describe_pet(name="Whiskers", age=3, animal="cat")

# Mix of positional and keyword (positional must come first)
describe_pet("hamster", name="Pip", age=1)
```

### Enforcing Keyword-Only Arguments

Use `*` in the parameter list to force all subsequent parameters to be keyword-only.

```python
def connect(host, port, *, timeout=30, use_ssl=False):
    print(f"Connecting to {host}:{port}")
    print(f"  timeout={timeout}, ssl={use_ssl}")

connect("localhost", 8080)                          # OK
connect("localhost", 8080, timeout=60, use_ssl=True) # OK
# connect("localhost", 8080, 60, True)  # TypeError! timeout and use_ssl are keyword-only
```

### Positional-Only Parameters (Python 3.8+)

Use `/` to force parameters before it to be positional-only.

```python
def power(base, exp, /):
    return base ** exp

print(power(2, 10))    # Output: 1024
# power(base=2, exp=10)  # TypeError! base and exp are positional-only
```

### Combining Positional-Only, Regular, and Keyword-Only

```python
def example(pos_only, /, regular, *, kw_only):
    print(f"pos_only={pos_only}, regular={regular}, kw_only={kw_only}")

example(1, 2, kw_only=3)          # OK
example(1, regular=2, kw_only=3)  # OK
# example(pos_only=1, regular=2, kw_only=3)  # TypeError!
```

---

## 5. Variable-Length Arguments: `*args` and `**kwargs`

### `*args` -- Variable Positional Arguments

`*args` collects extra positional arguments into a tuple.

```python
def add_all(*args):
    print(f"args = {args}")
    return sum(args)

print(add_all(1, 2, 3))        # args = (1, 2, 3) -> Output: 6
print(add_all(10, 20, 30, 40)) # args = (10, 20, 30, 40) -> Output: 100
```

### `**kwargs` -- Variable Keyword Arguments

`**kwargs` collects extra keyword arguments into a dictionary.

```python
def print_info(**kwargs):
    print(f"kwargs = {kwargs}")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print_info(name="Alice", age=30, city="Seoul")
# kwargs = {'name': 'Alice', 'age': 30, 'city': 'Seoul'}
#   name: Alice
#   age: 30
#   city: Seoul
```

### Combining `*args` and `**kwargs`

```python
def universal_function(*args, **kwargs):
    print(f"Positional: {args}")
    print(f"Keyword: {kwargs}")

universal_function(1, 2, 3, name="Alice", active=True)
# Positional: (1, 2, 3)
# Keyword: {'name': 'Alice', 'active': True}
```

### Unpacking Arguments

You can unpack sequences and dictionaries when calling functions.

```python
def greet(first, last, greeting="Hello"):
    print(f"{greeting}, {first} {last}!")

# Unpack a list/tuple with *
names = ["Alice", "Smith"]
greet(*names)  # Output: Hello, Alice Smith!

# Unpack a dictionary with **
config = {"first": "Bob", "last": "Jones", "greeting": "Good morning"}
greet(**config)  # Output: Good morning, Bob Jones!
```

### Practical Example: Flexible Logger

```python
def log(level, message, *tags, **metadata):
    tag_str = ", ".join(tags) if tags else "none"
    meta_str = " | ".join(f"{k}={v}" for k, v in metadata.items())
    print(f"[{level.upper()}] {message}")
    print(f"  Tags: {tag_str}")
    if meta_str:
        print(f"  Meta: {meta_str}")

log("info", "User logged in", "auth", "security", user="alice", ip="192.168.1.1")
# [INFO] User logged in
#   Tags: auth, security
#   Meta: user=alice | ip=192.168.1.1
```

---

## 6. Docstrings

Docstrings are string literals that appear as the first statement in a function body. They describe what the function does and are accessible via the `help()` function and the `__doc__` attribute.

```python
def calculate_area(length, width):
    """Calculate the area of a rectangle.

    Args:
        length: The length of the rectangle (must be positive).
        width: The width of the rectangle (must be positive).

    Returns:
        The area as a float.

    Raises:
        ValueError: If length or width is negative.
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions must be non-negative")
    return length * width

# Accessing the docstring
help(calculate_area)
print(calculate_area.__doc__)
```

### Docstring Styles

```python
# Google style (shown above)
def func_google(param1, param2):
    """Summary line.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.
    """
    pass

# NumPy/SciPy style
def func_numpy(param1, param2):
    """Summary line.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
    pass

# reStructuredText style (Sphinx)
def func_rst(param1, param2):
    """Summary line.

    :param param1: Description of param1.
    :type param1: int
    :param param2: Description of param2.
    :type param2: str
    :returns: Description of return value.
    :rtype: bool
    """
    pass
```

---

## 7. Scope: Local vs Global (LEGB Rule)

Python resolves names using the LEGB rule: **L**ocal, **E**nclosing, **G**lobal, **B**uilt-in.

```python
# Global scope
x = "global"

def outer():
    # Enclosing scope
    x = "enclosing"

    def inner():
        # Local scope
        x = "local"
        print(f"inner: {x}")    # Output: local

    inner()
    print(f"outer: {x}")        # Output: enclosing

outer()
print(f"module: {x}")           # Output: global
```

### The `global` Keyword

The `global` keyword lets a function modify a global variable.

```python
counter = 0

def increment():
    global counter
    counter += 1

increment()
increment()
print(counter)  # Output: 2
```

### The `nonlocal` Keyword

The `nonlocal` keyword lets an inner function modify a variable in its enclosing scope.

```python
def make_counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment

counter = make_counter()
print(counter())  # Output: 1
print(counter())  # Output: 2
print(counter())  # Output: 3
```

### Scope Pitfall: UnboundLocalError

```python
x = 10

def broken():
    # Python sees x = ... below, so it treats x as local in the ENTIRE function
    # Trying to read x before assigning it causes an error
    # print(x)  # UnboundLocalError: local variable 'x' referenced before assignment
    x = 20
    print(x)

broken()       # Output: 20
print(x)       # Output: 10 (global x unchanged)
```

---

## 8. Nested Functions

Functions defined inside other functions. They can access variables from the enclosing scope.

```python
def greet_builder(greeting):
    def greet(name):
        return f"{greeting}, {name}!"
    return greet

hello = greet_builder("Hello")
hi = greet_builder("Hi")

print(hello("Alice"))  # Output: Hello, Alice!
print(hi("Bob"))       # Output: Hi, Bob!
```

### Practical Use: Validation Wrapper

```python
def validated_operation(operation_name):
    def validate_and_run(a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return f"Error: {operation_name} requires numeric inputs"
        if operation_name == "divide" and b == 0:
            return "Error: cannot divide by zero"

        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b,
        }
        return operations.get(operation_name, "Unknown operation")

    return validate_and_run

divide = validated_operation("divide")
print(divide(10, 3))    # Output: 3.3333333333333335
print(divide(10, 0))    # Output: Error: cannot divide by zero
print(divide(10, "a"))  # Output: Error: divide requires numeric inputs
```

### Closures

A closure is a nested function that remembers variables from its enclosing scope even after the outer function has finished executing.

```python
def create_multiplier(factor):
    def multiply(number):
        return number * factor   # factor is "closed over"
    return multiply

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))   # Output: 10
print(triple(5))   # Output: 15

# Inspect closure variables
print(double.__closure__[0].cell_contents)  # Output: 2
```

---

## 9. Lambda Expressions

Lambda expressions create small anonymous functions in a single line. They are useful when you need a short function for a brief period.

```python
# Regular function
def add(a, b):
    return a + b

# Equivalent lambda
add_lambda = lambda a, b: a + b

print(add(3, 5))         # Output: 8
print(add_lambda(3, 5))  # Output: 8
```

### Lambda with Sorting

```python
students = [
    {"name": "Alice", "grade": 88},
    {"name": "Bob", "grade": 95},
    {"name": "Charlie", "grade": 72},
    {"name": "Diana", "grade": 91},
]

# Sort by grade
by_grade = sorted(students, key=lambda s: s["grade"])
print([s["name"] for s in by_grade])
# Output: ['Charlie', 'Alice', 'Diana', 'Bob']

# Sort by name length
by_name_len = sorted(students, key=lambda s: len(s["name"]))
print([s["name"] for s in by_name_len])
# Output: ['Bob', 'Alice', 'Diana', 'Charlie']
```

### Lambda with Conditional Expression

```python
classify = lambda x: "positive" if x > 0 else ("negative" if x < 0 else "zero")

print(classify(5))    # Output: positive
print(classify(-3))   # Output: negative
print(classify(0))    # Output: zero
```

### When to Use Lambda vs def

| Use Lambda | Use def |
|------------|---------|
| Simple, single expression | Multiple statements or complex logic |
| As an argument to `sorted`, `map`, `filter` | Function needs a name for clarity |
| Throwaway, one-time use | Function will be reused or tested |
| The logic is immediately obvious | Needs a docstring |

---

## 10. Built-in Functions Overview

Python provides many useful built-in functions. Knowing them prevents reinventing the wheel.

### `len()` -- Length

```python
print(len("Hello"))       # Output: 5
print(len([1, 2, 3, 4]))  # Output: 4
print(len({"a": 1, "b": 2}))  # Output: 2
```

### `max()` and `min()` -- Maximum and Minimum

```python
print(max(3, 7, 2, 9))          # Output: 9
print(min(3, 7, 2, 9))          # Output: 2
print(max([10, 20, 5]))         # Output: 20

# With key function
words = ["apple", "hi", "banana", "cat"]
print(max(words, key=len))      # Output: banana
print(min(words, key=len))      # Output: hi
```

### `sum()` -- Sum of Iterable

```python
print(sum([1, 2, 3, 4, 5]))     # Output: 15
print(sum(range(1, 101)))       # Output: 5050

# With start value
print(sum([1, 2, 3], 10))       # Output: 16 (10 + 1 + 2 + 3)
```

### `sorted()` and `reversed()`

```python
numbers = [3, 1, 4, 1, 5, 9]

# sorted returns a new list
print(sorted(numbers))              # Output: [1, 1, 3, 4, 5, 9]
print(sorted(numbers, reverse=True)) # Output: [9, 5, 4, 3, 1, 1]

# reversed returns an iterator
print(list(reversed(numbers)))  # Output: [9, 5, 1, 4, 1, 3]

# Sort with custom key
names = ["Charlie", "alice", "Bob"]
print(sorted(names, key=str.lower))  # Output: ['alice', 'Bob', 'Charlie']
```

### `map()` -- Apply Function to Each Element

`map()` applies a function to every item in an iterable and returns an iterator.

```python
numbers = [1, 2, 3, 4, 5]

# Square each number
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # Output: [1, 4, 9, 16, 25]

# Convert strings to integers
str_numbers = ["10", "20", "30"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # Output: [10, 20, 30]

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # Output: [11, 22, 33]
```

### `filter()` -- Select Elements by Condition

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Keep even numbers only
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # Output: [2, 4, 6, 8, 10]

# Remove empty strings
words = ["hello", "", "world", "", "python"]
non_empty = list(filter(None, words))  # None removes falsy values
print(non_empty)  # Output: ['hello', 'world', 'python']
```

### `zip()` -- Combine Iterables

```python
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

# Pair elements together
pairs = list(zip(names, scores))
print(pairs)
# Output: [('Alice', 85), ('Bob', 92), ('Charlie', 78)]

# Common pattern: create a dictionary
score_dict = dict(zip(names, scores))
print(score_dict)
# Output: {'Alice': 85, 'Bob': 92, 'Charlie': 78}

# Unzipping with zip(*)
paired = [("a", 1), ("b", 2), ("c", 3)]
letters, nums = zip(*paired)
print(letters)  # Output: ('a', 'b', 'c')
print(nums)     # Output: (1, 2, 3)
```

### `enumerate()` -- Index + Element

```python
fruits = ["apple", "banana", "cherry"]

# Instead of manual indexing
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Custom start index
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry
```

### `any()` and `all()`

```python
numbers = [2, 4, 6, 8, 10]

# all: True if ALL elements are truthy (or iterable is empty)
print(all(n > 0 for n in numbers))    # Output: True
print(all(n % 2 == 0 for n in numbers))  # Output: True

# any: True if ANY element is truthy
print(any(n > 5 for n in numbers))    # Output: True
print(any(n > 100 for n in numbers))  # Output: False

# Practical example: validation
def validate_user(name, email, age):
    checks = [
        len(name) > 0,
        "@" in email,
        age >= 18,
    ]
    return all(checks)

print(validate_user("Alice", "alice@example.com", 25))  # Output: True
print(validate_user("", "alice@example.com", 25))        # Output: False
```

### Comparison: map/filter vs List Comprehension

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# map + filter approach
result1 = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# List comprehension approach (generally preferred in Python)
result2 = [x ** 2 for x in numbers if x % 2 == 0]

print(result1)  # Output: [4, 16, 36, 64, 100]
print(result2)  # Output: [4, 16, 36, 64, 100]
```

---

## 11. Recursion Basics

A recursive function calls itself. Every recursive function needs a **base case** (stopping condition) to avoid infinite recursion.

### Factorial

```python
def factorial(n):
    """Calculate n! recursively."""
    if n <= 1:       # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))   # Output: 120 (5 * 4 * 3 * 2 * 1)
print(factorial(0))   # Output: 1
```

### How the Call Stack Works

```
factorial(4)
  -> 4 * factorial(3)
       -> 3 * factorial(2)
            -> 2 * factorial(1)
                 -> return 1        # Base case
            -> return 2 * 1 = 2
       -> return 3 * 2 = 6
  -> return 4 * 6 = 24
```

### Fibonacci Sequence

```python
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(fibonacci(i), end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34
print()
```

### Sum of a List (Recursive)

```python
def recursive_sum(lst):
    """Sum all elements in a list recursively."""
    if not lst:        # Base case: empty list
        return 0
    return lst[0] + recursive_sum(lst[1:])

print(recursive_sum([1, 2, 3, 4, 5]))  # Output: 15
```

### Recursion vs Iteration

```python
# Recursive countdown
def countdown_recursive(n):
    if n <= 0:
        print("Go!")
        return
    print(n)
    countdown_recursive(n - 1)

# Iterative countdown
def countdown_iterative(n):
    while n > 0:
        print(n)
        n -= 1
    print("Go!")

countdown_recursive(3)
# 3
# 2
# 1
# Go!
```

### Recursion Limit

Python has a default recursion limit (usually 1000) to prevent stack overflow.

```python
import sys

print(sys.getrecursionlimit())  # Output: 1000 (default)

# You can change it, but be careful
# sys.setrecursionlimit(2000)

# For deep recursion, prefer iterative solutions
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(factorial_iterative(100))  # Works fine, no recursion limit issue
```

---

## 12. Putting It All Together

### Example: Text Analysis Function

```python
def analyze_text(text, *, case_sensitive=False, min_word_length=1):
    """Analyze text and return word statistics.

    Args:
        text: The input text to analyze.
        case_sensitive: Whether to treat uppercase and lowercase
            as different words. Defaults to False.
        min_word_length: Minimum word length to include. Defaults to 1.

    Returns:
        A dictionary containing word count, unique words,
        and word frequency.
    """
    if not case_sensitive:
        text = text.lower()

    words = text.split()
    words = [w for w in words if len(w) >= min_word_length]

    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    return {
        "total_words": len(words),
        "unique_words": len(frequency),
        "most_common": max(frequency, key=frequency.get) if frequency else None,
        "frequency": dict(sorted(frequency.items(), key=lambda x: -x[1])),
    }

sample = "the quick brown fox jumps over the lazy dog the fox"
result = analyze_text(sample, min_word_length=3)
print(f"Total words: {result['total_words']}")
print(f"Unique words: {result['unique_words']}")
print(f"Most common: {result['most_common']}")
print(f"Frequency: {result['frequency']}")
# Total words: 9
# Unique words: 7
# Most common: the
# Frequency: {'the': 3, 'fox': 2, 'quick': 1, 'brown': 1, 'jumps': 1, 'over': 1, 'lazy': 1}
```

### Example: Flexible Data Processor

```python
def process_data(data, *transformations, verbose=False):
    """Apply a series of transformations to data.

    Args:
        data: The input list of numbers.
        *transformations: Functions to apply in sequence.
        verbose: If True, print intermediate results.

    Returns:
        The transformed data as a list.
    """
    result = list(data)

    for i, transform in enumerate(transformations):
        result = list(map(transform, result))
        if verbose:
            print(f"  Step {i + 1} ({transform.__name__}): {result}")

    return result

def double(x):
    return x * 2

def add_one(x):
    return x + 1

def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
output = process_data(numbers, double, add_one, square, verbose=True)
#   Step 1 (double): [2, 4, 6, 8, 10]
#   Step 2 (add_one): [3, 5, 7, 9, 11]
#   Step 3 (square): [9, 25, 49, 81, 121]
print(f"Final: {output}")
# Final: [9, 25, 49, 81, 121]
```

---

## 13. Summary

| Concept | Key Points |
|---------|------------|
| `def` | Define a function with parameters and body |
| `return` | Send a value back; `None` if omitted |
| Default params | Provide fallback values; avoid mutable defaults |
| `*args` | Collect extra positional arguments as tuple |
| `**kwargs` | Collect extra keyword arguments as dict |
| Docstrings | First string in function body; use Google/NumPy style |
| Scope (LEGB) | Local > Enclosing > Global > Built-in |
| `global`/`nonlocal` | Modify variables in outer scopes |
| Lambda | Anonymous single-expression functions |
| Built-ins | `len`, `max`, `min`, `sum`, `sorted`, `map`, `filter`, `zip`, `enumerate`, `any`, `all` |
| Recursion | Function calls itself; always define a base case |

---

## Exercises

1. Write a function `power(base, exp)` that calculates `base ** exp` using recursion (no `**` operator).
2. Write a function `flatten(nested_list)` that takes a nested list like `[[1, 2], [3, [4, 5]], 6]` and returns `[1, 2, 3, 4, 5, 6]` using recursion.
3. Create a `make_validator(**rules)` function that returns a validation function. The validator should check a dictionary against the rules (e.g., `make_validator(name=str, age=int)` returns a function that checks types).
4. Use `map`, `filter`, and `zip` to: given two lists of names and ages, produce a list of names of people who are 18 or older.
5. Write a decorator-like function `retry(func, attempts=3)` that calls `func()` up to `attempts` times, returning the result on success or `None` on failure.

---

**Previous**: [Control Flow](./04_Control_Flow.md) | **Next**: [Data Structures](./06_Data_Structures.md)

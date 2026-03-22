# Python 관용구와 모범 사례

**이전**: [표준 라이브러리 핵심](./13_Standard_Library_Essentials.md)

> **주제**: Python 기초
> **수업**: 14개 중 14번째
> **선수 지식**: 이전 Python 기초 수업 전체

## 학습 목표

이 수업을 완료하면 다음을 할 수 있습니다:

1. PEP 8 스타일 가이드라인과 Python의 선(Zen of Python, PEP 20)을 따르는 파이썬다운 코드 작성하기
2. 리스트, 딕셔너리, 집합에 대한 고급 컴프리헨션 패턴을 사용하여 간결하고 읽기 쉬운 코드 작성하기
3. 할당, 함수 호출, 정의에서 `*`와 `**`를 사용한 언패킹 기법 적용하기
4. Python의 참/거짓(Truthiness/Falsiness) 규칙을 이해하고 조건문에서 효과적으로 사용하기
5. 버그나 성능 문제로 이어지는 일반적인 안티 패턴을 인식하고 수정하기
6. 초보자를 위한 실용적인 성능 팁 적용하기 (반복 연결 피하기, 제너레이터 사용, dict/set 조회 선호)
7. `print()`, `breakpoint()`, `pdb`를 사용하여 Python 코드를 효과적으로 디버깅하기
8. 더 고급 Python 주제의 미리보기로 기본 타입 힌트 읽고 쓰기

---

## 소개

작동하는 Python 코드를 작성하는 것은 한 가지입니다. **파이썬다운** — 깔끔하고, 관용적이며, 우아한 — 코드를 작성하는 것은 또 다른 문제입니다. Python에는 구문을 넘어서는 독특한 문화와 철학이 있습니다. Python 기초의 마지막 수업에서는 지금까지 배운 모든 것을 종합하고 경험 많은 Python 개발자들이 코드를 작성하는 방법을 보여줍니다.

우리는 초보자 코드와 전문 코드를 구분하는 관용구, 패턴, 모범 사례를 다루고 Python 여정의 다음 단계를 안내합니다.

---

## Python의 선 (PEP 20)

```python
import this
```

Python의 선은 Python의 안내 철학입니다. 가장 영향력 있는 원칙들:

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

### 선을 적용하기

```python
# "Beautiful is better than ugly"
# BAD
x = {'a':1,'b':2,'c':3}
y = [v for k,v in x.items() if v>1]

# GOOD
scores = {"alice": 1, "bob": 2, "charlie": 3}
high_scores = [score for name, score in scores.items() if score > 1]

# "Flat is better than nested"
# BAD
def process(data):
    if data:
        if data.get("items"):
            for item in data["items"]:
                if item.get("active"):
                    yield item

# GOOD: Use early returns and guard clauses
def process(data):
    if not data:
        return
    items = data.get("items", [])
    for item in items:
        if item.get("active"):
            yield item

# "Explicit is better than implicit"
# BAD
from os import *   # What did we just import?

# GOOD
from os import path, getcwd, listdir  # Clear and specific
```

---

## PEP 8 하이라이트

PEP 8은 Python의 스타일 가이드입니다. 가장 중요한 규칙은 다음과 같습니다:

### 명명 규칙

```python
# Variables and functions: snake_case
user_name = "Alice"
total_count = 42

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# Classes: PascalCase
class HttpClient:
    pass

class UserAuthentication:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# Private: leading underscore
_internal_cache = {}

def _helper_function():
    pass

# "Dunder" (double underscore): for Python special methods only
class MyClass:
    def __init__(self):
        pass
    def __repr__(self):
        return "MyClass()"
```

### 간격과 레이아웃

```python
# Indentation: 4 spaces (NEVER tabs)
def function():
    if True:
        for i in range(10):
            print(i)

# Line length: max 79 characters (99 for many teams)
# Break long lines with backslash or parentheses
result = (first_variable
          + second_variable
          - third_variable)

long_string = (
    "This is a very long string that "
    "spans multiple lines using "
    "implicit string concatenation"
)

# Blank lines:
# - 2 blank lines before/after top-level definitions
# - 1 blank line between methods in a class
class MyClass:

    def method_one(self):
        pass

    def method_two(self):
        pass


def standalone_function():
    pass
```

### 공백 규칙

```python
# YES: spaces around operators
x = 1 + 2
y = x * 3 - 1
result = value if condition else default

# NO: spaces inside brackets
# BAD: func( arg1, arg2 )
# GOOD:
func(arg1, arg2)

# NO: space before colon in slices
# BAD: my_list [1 : 3]
# GOOD:
my_list[1:3]

# YES: space after comma
items = [1, 2, 3]
point = (x, y)

# Keyword arguments: no spaces around =
def func(name="default", verbose=False):
    pass

func(name="Alice", verbose=True)
```

---

## 컴프리헨션 — 고급 패턴

### 리스트 컴프리헨션 (List Comprehension)

```python
# Basic
squares = [x ** 2 for x in range(10)]

# With condition (filter)
evens = [x for x in range(20) if x % 2 == 0]

# With transformation and condition
clean_names = [name.strip().title() for name in raw_names if name.strip()]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# Flatten nested lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Multiple conditions
filtered = [x for x in range(100) if x % 2 == 0 if x % 3 == 0]
# Same as: [x for x in range(100) if x % 2 == 0 and x % 3 == 0]
```

### 딕셔너리 컴프리헨션 (Dict Comprehension)

```python
# Basic: transform keys/values
names = ["alice", "bob", "charlie"]
name_lengths = {name: len(name) for name in names}
# {'alice': 5, 'bob': 3, 'charlie': 7}

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
# {1: 'a', 2: 'b', 3: 'c'}

# Filter items
scores = {"alice": 95, "bob": 67, "charlie": 82, "diana": 91}
passing = {name: score for name, score in scores.items() if score >= 70}
# {'alice': 95, 'charlie': 82, 'diana': 91}

# From two lists
keys = ["name", "age", "city"]
values = ["Alice", 30, "NYC"]
combined = {k: v for k, v in zip(keys, values)}
# {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Group items by category
words = ["apple", "ant", "banana", "avocado", "berry"]
by_first_letter = {}
for word in words:
    by_first_letter.setdefault(word[0], []).append(word)
# {'a': ['apple', 'ant', 'avocado'], 'b': ['banana', 'berry']}
```

### 집합 컴프리헨션 (Set Comprehension)

```python
# Unique values with transformation
sentence = "the quick brown fox jumps over the lazy dog"
unique_lengths = {len(word) for word in sentence.split()}
# {3, 4, 5}

# Unique first characters
first_chars = {word[0] for word in sentence.split()}
# {'t', 'q', 'b', 'f', 'j', 'o', 'l', 'd'}
```

### 제너레이터 표현식 (Generator Expression)

```python
# Generator expression: like list comprehension but lazy
# Uses parentheses instead of brackets
sum_of_squares = sum(x ** 2 for x in range(1000000))
# Does NOT create a million-element list in memory

# Pass directly to functions that accept iterables
any_negative = any(x < 0 for x in values)
all_positive = all(x > 0 for x in values)
max_length = max(len(s) for s in strings)
joined = ", ".join(str(x) for x in numbers)
```

### 컴프리헨션을 사용하지 말아야 할 때

```python
# BAD: Too complex (hard to read)
result = [
    transform(item)
    for sublist in data
    for item in sublist
    if isinstance(item, dict)
    if item.get("active")
    if item["score"] > threshold
]

# BETTER: Use a regular loop for complex logic
result = []
for sublist in data:
    for item in sublist:
        if not isinstance(item, dict):
            continue
        if not item.get("active"):
            continue
        if item["score"] <= threshold:
            continue
        result.append(transform(item))

# Rule of thumb: if the comprehension does not fit on one or two lines,
# use a regular loop
```

---

## 언패킹 (Unpacking)

### 기본 언패킹

```python
# Tuple unpacking
x, y = (10, 20)
name, age, city = ("Alice", 30, "NYC")

# Works with any iterable
first, second, third = [1, 2, 3]
a, b, c = "abc"
x, y = {10, 20}  # Order not guaranteed for sets

# Swap variables (no temp variable needed)
a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1
```

### `*`를 사용한 확장 언패킹

```python
# Capture remaining elements
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

*init, last = [1, 2, 3, 4, 5]
print(init)  # [1, 2, 3, 4]
print(last)  # 5

first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

# Ignoring values
_, *_, last = [1, 2, 3, 4, 5]
print(last)  # 5

# In for loops
pairs = [(1, "one"), (2, "two"), (3, "three")]
for number, name in pairs:
    print(f"{number}: {name}")
```

### 함수 호출과 정의에서의 `*`

```python
# Unpack a list into function arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # 6

# Merge lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged = [*list1, *list2]  # [1, 2, 3, 4, 5, 6]

# Merge sets
set1 = {1, 2, 3}
set2 = {3, 4, 5}
merged_set = {*set1, *set2}  # {1, 2, 3, 4, 5}
```

### 딕셔너리 언패킹을 위한 `**`

```python
# Merge dictionaries
defaults = {"color": "blue", "size": "medium", "quantity": 1}
user_prefs = {"color": "red", "quantity": 5}

config = {**defaults, **user_prefs}
print(config)  # {'color': 'red', 'size': 'medium', 'quantity': 5}

# Python 3.9+ alternative
config = defaults | user_prefs  # Same result

# Unpack dict into function keyword arguments
def create_user(name, age, city="Unknown"):
    return f"{name}, {age}, from {city}"

user_data = {"name": "Alice", "age": 30, "city": "NYC"}
print(create_user(**user_data))  # Alice, 30, from NYC

# Add items while unpacking
base = {"host": "localhost", "port": 5432}
full = {**base, "database": "myapp", "timeout": 30}
```

---

## 삼항 표현식 (Ternary Expression)

### 조건 표현식

```python
# Basic ternary: value_if_true if condition else value_if_false
age = 20
status = "adult" if age >= 18 else "minor"

# In assignments
x = 10
result = x if x > 0 else -x  # Absolute value

# In function calls
print("even" if x % 2 == 0 else "odd")

# In return statements
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# Nested ternary (avoid - hard to read)
# BAD
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"

# BETTER: use a function or dict
def get_grade(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    return "F"
```

---

## 참/거짓 (Truthiness and Falsiness)

### Python의 거짓 값

```python
# All of these are "falsy" (evaluate to False in boolean context)
falsy_values = [
    False,      # Boolean
    None,       # NoneType
    0,          # Integer zero
    0.0,        # Float zero
    0j,         # Complex zero
    "",          # Empty string
    [],          # Empty list
    (),          # Empty tuple
    {},          # Empty dict
    set(),       # Empty set
    frozenset(), # Empty frozenset
    range(0),    # Empty range
]

for val in falsy_values:
    assert not val, f"{val!r} should be falsy"
```

### 참/거짓을 효과적으로 사용하기

```python
# Check for empty collections
items = []

# BAD: Explicit length check
if len(items) == 0:
    print("Empty")

# GOOD: Use truthiness
if not items:
    print("Empty")

# Check for None or empty string
name = ""

# BAD
if name is not None and name != "":
    process(name)

# GOOD
if name:
    process(name)

# BUT be careful with numeric zero!
count = 0

# This skips zero, which might be a valid value:
if count:  # False for 0!
    print(count)

# If zero is valid, be explicit:
if count is not None:
    print(count)
```

### `or`와 `and` 단락 평가 패턴

```python
# 'or' returns the first truthy value (or the last value)
name = "" or "Anonymous"
print(name)  # "Anonymous"

name = "Alice" or "Anonymous"
print(name)  # "Alice"

# Useful for defaults
config_value = user_setting or default_setting

# 'and' returns the first falsy value (or the last value)
result = "Alice" and "Bob"
print(result)  # "Bob" (both truthy, returns last)

result = "" and "Bob"
print(result)  # "" (first is falsy, returns it)

# Common pattern: validate and use
data = get_data() and process(get_data())
```

### `is None` vs 참/거짓

```python
# Use 'is None' when you specifically want to check for None
def greet(name=None):
    # BAD: if not name - this also catches empty string ""
    # GOOD: explicitly check for None
    if name is None:
        name = "World"
    return f"Hello, {name}!"

print(greet())       # Hello, World!
print(greet(""))     # Hello, !  (empty string is a valid name)
print(greet("Bob"))  # Hello, Bob!
```

---

## EAFP vs LBYL (복습)

### EAFP: 허락보다 용서가 쉽다

```python
# Pythonic: try it and handle the exception
def get_user_age(users, name):
    try:
        return users[name]["age"]
    except KeyError:
        return None

# Access nested data safely
def deep_get(data, *keys, default=None):
    current = data
    for key in keys:
        try:
            current = current[key]
        except (KeyError, TypeError, IndexError):
            return default
    return current

config = {"db": {"primary": {"host": "localhost"}}}
print(deep_get(config, "db", "primary", "host"))     # localhost
print(deep_get(config, "db", "replica", "host"))     # None
```

### LBYL이 더 나은 경우

```python
# LBYL is better when:
# 1. The check is simple and cheap
if user.is_admin:
    show_admin_panel()

# 2. The operation has side effects
# BAD EAFP: might partially execute before failing
try:
    send_email(user)        # Sent!
    update_database(user)   # Fails - but email already sent
except DatabaseError:
    pass

# GOOD LBYL: validate everything first
if user.has_valid_email and database.is_connected:
    send_email(user)
    update_database(user)
```

---

## 컨텍스트 관리자 (`with` 문)

### 내장 컨텍스트 관리자

```python
# Files
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Multiple resources
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read().upper())

# Locks (threading)
import threading
lock = threading.Lock()
with lock:
    # Thread-safe operations
    shared_data.append(item)

# Suppress specific exceptions
from contextlib import suppress
with suppress(FileNotFoundError):
    os.remove("temp.txt")  # No error if file does not exist

# Temporary directory
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # tmpdir is automatically cleaned up
    pass

# Decimal precision
from decimal import Decimal, localcontext
with localcontext() as ctx:
    ctx.prec = 50
    result = Decimal(1) / Decimal(7)
```

### 자체 컨텍스트 관리자 작성

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label="Operation"):
    """Time a block of code."""
    start = time.perf_counter()
    yield  # Code in the 'with' block runs here
    elapsed = time.perf_counter() - start
    print(f"{label} took {elapsed:.4f}s")

with timer("Data processing"):
    data = [x ** 2 for x in range(1_000_000)]

@contextmanager
def working_directory(path):
    """Temporarily change the working directory."""
    import os
    original = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(original)

with working_directory("/tmp"):
    print(os.getcwd())  # /tmp
print(os.getcwd())      # Back to original

@contextmanager
def open_or_stdout(filepath=None):
    """Open a file or fall back to stdout."""
    import sys
    if filepath:
        f = open(filepath, "w", encoding="utf-8")
        try:
            yield f
        finally:
            f.close()
    else:
        yield sys.stdout

# Write to file or stdout
with open_or_stdout("output.txt") as f:
    f.write("Goes to file\n")

with open_or_stdout() as f:
    f.write("Goes to stdout\n")
```

---

## 문자열 형식 지정 모범 사례

### f-문자열 (Python 3.6+ 권장)

```python
name = "Alice"
age = 30
balance = 1234.5678

# Basic
print(f"Hello, {name}!")

# Expressions
print(f"{name} will be {age + 1} next year")

# Format specifications
print(f"Balance: ${balance:.2f}")         # $1234.57
print(f"Balance: ${balance:>12,.2f}")     # $    1,234.57
print(f"{'ID':>10} | {'Name':<15} | {'Score':>8}")

# Debugging (Python 3.8+)
x = 42
print(f"{x = }")        # x = 42
print(f"{x * 2 = }")    # x * 2 = 84
print(f"{name = !r}")   # name = 'Alice'

# Multiline f-strings
message = (
    f"User: {name}\n"
    f"Age: {age}\n"
    f"Balance: ${balance:.2f}"
)

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"Today is {now:%Y-%m-%d}")
print(f"Time: {now:%I:%M %p}")
```

### 다른 방법을 사용해야 할 때

```python
# str.format() — for templates defined elsewhere
template = "Hello, {name}! You have {count} new messages."
print(template.format(name="Alice", count=5))

# % formatting — in logging (lazy evaluation)
import logging
logger = logging.getLogger(__name__)
logger.debug("Processing item %s of %d", item_id, total)
# The string is only formatted if debug level is active

# Template strings — for untrusted input
from string import Template
t = Template("Hello, $name!")
# Safe: cannot execute arbitrary code
print(t.safe_substitute(name="Alice"))
```

---

## 일반적인 안티 패턴과 수정

### 안티 패턴 1: 변경 가능한 기본 인자

```python
# BAD: Mutable default argument is shared across calls
def append_to(item, target=[]):
    target.append(item)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2] !!  Expected [2]
print(append_to(3))  # [1, 2, 3] !!

# GOOD: Use None as default, create new list inside
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [2]
print(append_to(3))  # [3]
```

### 안티 패턴 2: `isinstance()` 대신 `type()` 사용

```python
# BAD: Does not handle subclasses
if type(x) == int:
    pass

# GOOD: Handles subclasses correctly
if isinstance(x, int):
    pass

# Check multiple types
if isinstance(x, (int, float)):
    pass
```

### 안티 패턴 3: 반복 문자열 연결

```python
# BAD: O(n^2) — creates a new string each iteration
result = ""
for i in range(10000):
    result += str(i) + ", "

# GOOD: O(n) — join is optimized
parts = [str(i) for i in range(10000)]
result = ", ".join(parts)

# Or with a generator
result = ", ".join(str(i) for i in range(10000))
```

### 안티 패턴 4: `enumerate()` 미사용

```python
# BAD: Manual index tracking
names = ["Alice", "Bob", "Charlie"]
i = 0
for name in names:
    print(f"{i}: {name}")
    i += 1

# ALSO BAD: Range + index
for i in range(len(names)):
    print(f"{i}: {names[i]}")

# GOOD: enumerate
for i, name in enumerate(names):
    print(f"{i}: {name}")

# With custom start
for i, name in enumerate(names, start=1):
    print(f"{i}: {name}")
```

### 안티 패턴 5: 불필요한 `dict.keys()` 사용

```python
data = {"a": 1, "b": 2, "c": 3}

# BAD
if "a" in data.keys():
    pass

# GOOD: dicts are iterable over keys by default
if "a" in data:
    pass

# BAD
for key in data.keys():
    print(key)

# GOOD (unless you need .keys() for set operations)
for key in data:
    print(key)
```

### 안티 패턴 6: `zip()` 미사용

```python
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]

# BAD
for i in range(len(names)):
    print(f"{names[i]} is {ages[i]}")

# GOOD
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# With strict mode (Python 3.10+) - ensures equal lengths
for name, age in zip(names, ages, strict=True):
    print(f"{name} is {age}")
```

### 안티 패턴 7: 불필요한 `if/else` 반환

```python
# BAD
def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False

# GOOD
def is_even(n):
    return n % 2 == 0

# BAD
def get_status(active):
    if active:
        return "active"
    return "inactive"
    # (This one is actually fine — but consider the ternary for simple cases)

# ALSO GOOD
def get_status(active):
    return "active" if active else "inactive"
```

---

## 초보자를 위한 성능 팁

### 반복 문자열 연결 피하기

```python
# BAD: O(n^2) time complexity
output = ""
for word in words:
    output += word + " "

# GOOD: O(n)
output = " ".join(words)
```

### 대용량 데이터에 제너레이터 사용

```python
# BAD: Creates a huge list in memory
total = sum([x ** 2 for x in range(10_000_000)])

# GOOD: Generator uses almost no memory
total = sum(x ** 2 for x in range(10_000_000))

# BAD: Reads entire file into memory
with open("huge.txt") as f:
    lines = f.readlines()
    for line in lines:
        process(line)

# GOOD: Reads one line at a time
with open("huge.txt") as f:
    for line in f:
        process(line)
```

### `list`보다 `dict`와 `set` 조회 선호

```python
import time

# Membership testing: list is O(n), set is O(1)
data = list(range(1_000_000))
data_set = set(data)

# Slow: O(n) lookup
start = time.perf_counter()
999_999 in data
list_time = time.perf_counter() - start

# Fast: O(1) lookup
start = time.perf_counter()
999_999 in data_set
set_time = time.perf_counter() - start

print(f"List lookup: {list_time:.6f}s")
print(f"Set lookup:  {set_time:.6f}s")
# Set is typically 1000x+ faster for large collections

# PRACTICAL: Convert to set for repeated lookups
valid_ids = set(load_valid_ids())  # Convert once
for record in records:
    if record["id"] in valid_ids:  # O(1) each time
        process(record)
```

### 수동 카운팅 대신 `collections.Counter` 사용

```python
# BAD: Manual counting
counts = {}
for item in items:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1

# GOOD
from collections import Counter
counts = Counter(items)
```

### 타이트한 루프에서 지역 변수 사용

```python
# Slightly slower: attribute lookup each iteration
for item in data:
    result.append(item * 2)

# Slightly faster: cache the method lookup
append = result.append
for item in data:
    append(item * 2)

# Even better: use a comprehension (optimized at C level)
result = [item * 2 for item in data]
```

### 단락 평가를 위한 `any()`와 `all()` 사용

```python
# BAD: Checks ALL items even if first matches
found = False
for item in items:
    if is_valid(item):
        found = True
        break

# GOOD: Stops at first match
found = any(is_valid(item) for item in items)

# BAD: Checks ALL items
all_valid = True
for item in items:
    if not is_valid(item):
        all_valid = False
        break

# GOOD
all_valid = all(is_valid(item) for item in items)
```

---

## 디버깅 기초

### `print()` 디버깅

가장 간단한 디버깅 기법:

```python
def calculate_discount(price, discount_pct, min_price=0):
    """Calculate discounted price."""
    print(f"DEBUG: price={price}, discount_pct={discount_pct}, min_price={min_price}")

    discount = price * (discount_pct / 100)
    print(f"DEBUG: discount={discount}")

    final = price - discount
    print(f"DEBUG: final before clamp={final}")

    final = max(final, min_price)
    print(f"DEBUG: final after clamp={final}")

    return final

# Better: use f-string debug format (Python 3.8+)
def calculate_discount(price, discount_pct, min_price=0):
    print(f"{price = }, {discount_pct = }, {min_price = }")  # Self-documenting
    discount = price * (discount_pct / 100)
    print(f"{discount = }")
    final = max(price - discount, min_price)
    print(f"{final = }")
    return final
```

### `breakpoint()`와 `pdb`

```python
def process_items(items):
    """Process items with a debugger breakpoint."""
    results = []
    for i, item in enumerate(items):
        if item.get("status") == "error":
            breakpoint()  # Drops into pdb debugger here
            # In the debugger, you can:
            #   p item        - print the item
            #   p i           - print the index
            #   n             - next line
            #   c             - continue execution
            #   l             - list surrounding code
            #   q             - quit debugger
        results.append(transform(item))
    return results
```

### 필수 `pdb` 명령어

| 명령어 | 축약 | 설명 |
|--------|------|------|
| `help` | `h` | 도움말 표시 |
| `next` | `n` | 다음 줄 실행 (건너뛰기) |
| `step` | `s` | 함수 호출 안으로 진입 |
| `continue` | `c` | 다음 중단점까지 계속 |
| `print expr` | `p expr` | 표현식 값 출력 |
| `pp expr` | | 표현식 예쁘게 출력 |
| `list` | `l` | 현재 코드 컨텍스트 표시 |
| `where` | `w` | 호출 스택 표시 |
| `up` | `u` | 호출 스택에서 위로 이동 |
| `down` | `d` | 호출 스택에서 아래로 이동 |
| `quit` | `q` | 디버거 종료 |
| `break N` | `b N` | N번 줄에 중단점 설정 |

### 실용적인 디버깅 예제

```python
import logging

# Set up logging for debugging (better than print for production code)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def find_duplicates(items):
    """Find duplicate items in a list."""
    logger.debug(f"Finding duplicates in {len(items)} items")

    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            logger.debug(f"Duplicate found: {item!r}")
            duplicates.add(item)
        seen.add(item)

    logger.info(f"Found {len(duplicates)} duplicate values")
    return duplicates

# Usage
data = [1, 2, 3, 2, 4, 5, 3, 6, 1]
dupes = find_duplicates(data)
print(f"Duplicates: {dupes}")
```

### 디버깅을 위한 `assert` 사용

```python
def divide(a, b):
    """Divide a by b."""
    assert b != 0, "Divisor must not be zero"
    assert isinstance(a, (int, float)), f"Expected number, got {type(a)}"
    return a / b

# Assertions can be disabled with python -O (optimize mode)
# Do NOT use assert for input validation in production code

# GOOD use of assert: internal invariants
def sort_and_deduplicate(items):
    result = sorted(set(items))
    assert result == sorted(result), "Result should be sorted"
    assert len(result) == len(set(result)), "Result should have no duplicates"
    return result
```

---

## 타입 힌트 입문

타입 힌트 (Type Hint, Python 3.5에서 도입, 3.9+에서 개선)는 코드에 선택적 타입 정보를 추가합니다. 런타임 동작에는 영향을 미치지 않지만 다음에 도움이 됩니다:
- IDE 자동 완성과 오류 감지
- `mypy` 같은 도구를 사용한 정적 분석
- 코드 문서화

### 기본 타입 힌트

```python
# Variable annotations
name: str = "Alice"
age: int = 30
balance: float = 1234.56
active: bool = True

# Function annotations
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process(data: list[str]) -> dict[str, int]:
    return {item: len(item) for item in data}
```

### 일반적인 타입 힌트 패턴

```python
from typing import Optional, Union

# Optional: can be the type or None
def find_user(user_id: int) -> Optional[str]:
    """Find a user by ID, returning None if not found."""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Union: can be one of several types (Python 3.10+ uses | instead)
def process(value: Union[str, int]) -> str:
    return str(value)

# Python 3.10+ syntax
def process(value: str | int) -> str:
    return str(value)

# Collections (Python 3.9+)
def mean(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

def word_count(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

# Callable
from typing import Callable

def apply_operation(
    x: float,
    y: float,
    operation: Callable[[float, float], float]
) -> float:
    return operation(x, y)

result = apply_operation(10, 3, lambda a, b: a + b)
```

### 클래스에서의 타입 힌트

```python
class User:
    name: str
    age: int
    scores: list[int]

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
        self.scores = []

    def add_score(self, score: int) -> None:
        self.scores.append(score)

    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __repr__(self) -> str:
        return f"User(name={self.name!r}, age={self.age})"
```

### 타입 검사 실행

```bash
# Install mypy
pip install mypy

# Check a file
mypy my_script.py

# Check with strict mode
mypy --strict my_script.py
```

> **참고:** 타입 힌트는 [Python 고급](../Python_Advanced/00_Overview.md)에서 훨씬 더 깊이 다룹니다. 이것은 시작을 위한 미리보기입니다.

---

## 모든 것을 종합하기

이 수업의 많은 패턴을 적용한 완전한 예제입니다:

```python
"""Student grade tracker — demonstrating Pythonic patterns."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class Student:
    """Represents a student with their grades."""

    name: str
    grades: list[float] = field(default_factory=list)

    @property
    def average(self) -> float:
        """Calculate the average grade."""
        return sum(self.grades) / len(self.grades) if self.grades else 0.0

    @property
    def letter_grade(self) -> str:
        """Convert average to a letter grade."""
        avg = self.average
        if avg >= 90:
            return "A"
        if avg >= 80:
            return "B"
        if avg >= 70:
            return "C"
        if avg >= 60:
            return "D"
        return "F"

    def add_grade(self, grade: float) -> None:
        """Add a grade, validating the value."""
        if not 0 <= grade <= 100:
            raise ValueError(f"Grade must be 0-100, got {grade}")
        self.grades.append(grade)


class GradeTracker:
    """Track grades for multiple students."""

    def __init__(self) -> None:
        self._students: dict[str, Student] = {}

    def add_student(self, name: str) -> Student:
        """Add a new student."""
        if name in self._students:
            raise ValueError(f"Student {name!r} already exists")
        student = Student(name=name)
        self._students[name] = student
        return student

    def get_student(self, name: str) -> Optional[Student]:
        """Get a student by name."""
        return self._students.get(name)

    def record_grade(self, name: str, grade: float) -> None:
        """Record a grade for a student."""
        student = self._students.get(name)
        if student is None:
            raise KeyError(f"Student {name!r} not found")
        student.add_grade(grade)

    @property
    def class_average(self) -> float:
        """Calculate the class average."""
        averages = [s.average for s in self._students.values() if s.grades]
        return sum(averages) / len(averages) if averages else 0.0

    def grade_distribution(self) -> dict[str, int]:
        """Get the distribution of letter grades."""
        return dict(Counter(
            s.letter_grade for s in self._students.values() if s.grades
        ))

    def top_students(self, n: int = 5) -> list[Student]:
        """Get the top n students by average grade."""
        with_grades = [s for s in self._students.values() if s.grades]
        return sorted(with_grades, key=lambda s: s.average, reverse=True)[:n]

    def save(self, filepath: str | Path) -> None:
        """Save tracker data to JSON."""
        data = {
            name: {"grades": student.grades}
            for name, student in self._students.items()
        }
        Path(filepath).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filepath: str | Path) -> GradeTracker:
        """Load tracker data from JSON."""
        path = Path(filepath)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        tracker = cls()
        for name, info in data.items():
            student = tracker.add_student(name)
            for grade in info.get("grades", []):
                student.add_grade(grade)
        return tracker

    def report(self) -> str:
        """Generate a formatted report."""
        lines = [
            "Grade Report",
            "=" * 50,
            f"Students: {len(self._students)}",
            f"Class Average: {self.class_average:.1f}",
            "",
            f"{'Name':<20} {'Avg':>6} {'Grade':>6} {'Count':>6}",
            "-" * 50,
        ]

        for student in sorted(
            self._students.values(),
            key=lambda s: s.average,
            reverse=True,
        ):
            lines.append(
                f"{student.name:<20} "
                f"{student.average:>6.1f} "
                f"{student.letter_grade:>6} "
                f"{len(student.grades):>6}"
            )

        lines.extend([
            "",
            "Distribution:",
            *[
                f"  {grade}: {count}"
                for grade, count in sorted(self.grade_distribution().items())
            ],
        ])

        return "\n".join(lines)


def main() -> None:
    """Demonstrate the grade tracker."""
    tracker = GradeTracker()

    # Add students and grades
    students_data = {
        "Alice": [95, 88, 92, 97, 90],
        "Bob": [78, 85, 72, 80, 88],
        "Charlie": [92, 96, 94, 91, 98],
        "Diana": [65, 70, 68, 72, 75],
        "Eve": [88, 91, 85, 93, 87],
    }

    for name, grades in students_data.items():
        tracker.add_student(name)
        for grade in grades:
            tracker.record_grade(name, grade)

    # Print report
    print(tracker.report())

    # Top students
    print("\nTop 3 students:")
    for i, student in enumerate(tracker.top_students(3), 1):
        print(f"  {i}. {student.name}: {student.average:.1f}")

    # Save and reload
    tracker.save("grades.json")
    loaded = GradeTracker.load("grades.json")
    print(f"\nLoaded {len(loaded._students)} students from file")


if __name__ == "__main__":
    main()
```

---

## 다음 단계

Python 기초를 완료하신 것을 축하합니다! 이제 Python 프로그래밍의 탄탄한 기반을 갖추셨습니다. 권장하는 다음 단계는 다음과 같습니다:

1. **Python 고급** — 데코레이터, 제너레이터, 메타클래스, 동시성 등 더 고급 주제를 위해 [Python 고급](../Python_Advanced/00_Overview.md)으로 계속하세요.

2. **연습** — 작은 프로젝트를 만들어 배운 것을 적용하세요:
   - 명령줄 할 일 관리 앱
   - 파일 정리 스크립트
   - 간단한 웹 스크레이퍼
   - 데이터 분석 파이프라인

3. **실제 코드 읽기** — GitHub에서 잘 작성된 Python 프로젝트를 학습하여 경험 많은 개발자들이 코드를 구성하는 방법을 살펴보세요.

4. **표준 라이브러리 탐색** — 200개 이상의 모듈이 있습니다. [Python 모듈 인덱스](https://docs.python.org/3/py-modindex.html)를 브라우징하여 유용한 도구를 발견하세요.

5. **테스트** — `pytest`로 테스트를 작성하여 코드를 신뢰할 수 있고 유지보수 가능하게 만드는 방법을 배우세요.

---

## 요약

| 주제 | 핵심 원칙 |
|------|----------|
| Python의 선 | 가독성이 중요하다; 명시적이 암시적보다 낫다 |
| PEP 8 | `snake_case` 함수, `PascalCase` 클래스, 4칸 들여쓰기 |
| 컴프리헨션 | 간단한 변환에 사용; 복잡한 로직은 루프 사용 |
| 언패킹 | 시퀀스에 `*`, 딕셔너리에 `**`; `a, b = b, a`로 교환 |
| 참/거짓 | 빈 컬렉션, `0`, `None`, `""`은 거짓 |
| EAFP | 먼저 시도하고 예외를 잡기 (파이썬다운 스타일) |
| 컨텍스트 관리자 | 정리가 필요한 리소스에 `with` 사용 |
| 문자열 형식 지정 | 대부분의 경우 f-문자열 사용 |
| 안티 패턴 | 변경 가능한 기본값, 단독 except, 수동 카운팅 |
| 성능 | 문자열 join, 제너레이터 사용, set/dict 조회 선호 |
| 디버깅 | `print(f"{x = }")`, `breakpoint()`, 로깅 |
| 타입 힌트 | 선택적이지만 문서화와 도구 지원에 가치 있음 |

---

## 추가 자료

- [PEP 8 — Python 코드 스타일 가이드](https://peps.python.org/pep-0008/)
- [PEP 20 — Python의 선](https://peps.python.org/pep-0020/)
- [PEP 257 — 독스트링 규칙](https://peps.python.org/pep-0257/)
- [PEP 484 — 타입 힌트](https://peps.python.org/pep-0484/)
- [Effective Python by Brett Slatkin](https://effectivepython.com/)
- [Fluent Python by Luciano Ramalho](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

---

**이전**: [표준 라이브러리 핵심](./13_Standard_Library_Essentials.md)

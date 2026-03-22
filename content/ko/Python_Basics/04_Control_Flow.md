# 제어 흐름

**이전**: [연산자와 표현식](./03_Operators_and_Expressions.md) | **다음**: [함수](./05_Functions.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. `if`, `elif`, `else`를 사용하여 프로그램 실행을 분기하는 조건 논리를 작성할 수 있다
2. 간결한 인라인 결정을 위해 조건 표현식 (삼항 연산자)을 사용할 수 있다
3. `enumerate()`와 `zip()`을 포함하여 `for` 루프로 시퀀스와 범위를 순회할 수 있다
4. 무한 반복을 위한 `while` 루프를 구축하고 감시 루프 패턴 (Sentinel Loop Pattern)을 적용할 수 있다
5. `break`, `continue`, `pass`로 루프 실행을 제어할 수 있다
6. 검색 및 발견 패턴에 `for/else`와 `while/else` 구조를 사용할 수 있다
7. Python 3.10에 도입된 구조적 패턴 매칭 (Structural Pattern Matching, `match/case`)을 적용할 수 있다
8. 누적기, 플래그, 중첩 루프, 조기 반환 등 일반적인 제어 흐름 패턴을 인식하고 구현할 수 있다

---

제어 흐름 (Control Flow)은 문장이 실행되는 순서를 결정합니다. 제어 흐름이 없으면 프로그램은 위에서 아래로 직선으로 실행됩니다. 조건문은 프로그램이 결정을 내릴 수 있게 하고, 루프는 동작을 반복할 수 있게 합니다. 이 둘이 함께 모든 비자명한 프로그램의 뼈대를 형성합니다.

## 조건문 (Conditional Statements)

### if 문

`if` 문은 조건이 참일 때만 코드 블록을 실행합니다.

```python
temperature = 35

if temperature > 30:
    print("It's hot outside!")
    print("Stay hydrated.")

# Output:
# It's hot outside!
# Stay hydrated.
```

핵심 사항:
- 조건은 참 또는 거짓 값으로 평가되는 모든 표현식이 될 수 있습니다.
- 조건 뒤에 콜론 (`:`)이 필수입니다.
- 본문은 4개의 공백으로 들여쓰기됩니다 (Python 관례).

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

`elif` ("else if"의 줄임말)를 사용하면 여러 조건을 순차적으로 검사할 수 있습니다. Python은 위에서 아래로 평가하고 첫 번째 일치하는 블록을 실행합니다.

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

중요한 동작:
- **첫 번째** 일치하는 분기만 실행됩니다 — 이후 `elif`와 `else` 블록은 완전히 건너뜁니다.
- `else` 블록은 선택적이며 포괄적인 대비책으로 작동합니다.
- `elif` 분기의 수에는 제한이 없습니다.

### 중첩 조건문 (Nested Conditionals)

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

> **팁**: 깊이 중첩된 조건문 (2-3단계 이상)은 코드 스멜입니다. 조기 반환 (Early Return), 가드 절 (Guard Clause), 또는 헬퍼 함수를 사용하여 리팩토링하세요.

### 가드 절 (Guard Clauses, 조기 반환 패턴)

가드 절은 엣지 케이스를 먼저 처리하여 중첩된 조건문을 평탄화합니다:

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

### 조건 표현식 (삼항 연산자, Ternary Operator)

조건에 따라 두 값 중 하나를 선택하는 간결한 방법:

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

### 조건에서의 참/거짓 (Truthiness)

Python은 비불리언 값을 참/거짓으로 평가합니다 (레슨 02 참조):

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

## for 루프

`for` 루프는 모든 **이터러블 (Iterable)** (리스트, 튜플, 문자열, range, dict, set, 파일, 제너레이터 등)의 항목을 순회합니다.

### 기본 for 루프

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

### `range()` 함수

`range()`는 정수 시퀀스를 생성합니다. 숫자 카운터가 필요한 `for` 루프에서 일반적으로 사용됩니다.

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

### `enumerate()` — 인덱스와 값을 함께

인덱스와 값이 모두 필요할 때, 수동 인덱싱 대신 `enumerate()`를 사용하세요:

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

### `zip()` — 여러 시퀀스 병렬 순회

`zip()`은 두 개 이상의 이터러블에서 요소를 쌍으로 묶습니다:

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

### 여러 컬렉션 순회 — 실용적인 예제

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

## while 루프

`while` 루프는 조건이 참인 동안 반복합니다.

### 기본 while 루프

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

### 감시 루프 (Sentinel Loop, 특정 값까지 반복)

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

### 입력 검증 루프 (Input Validation Loop)

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

### 카운트다운 패턴

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

### 수렴 루프 (Convergence Loop, 수치 해석)

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

## 루프 제어문 (Loop Control Statements)

### `break` — 즉시 루프 종료

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

### `continue` — 다음 반복으로 건너뛰기

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

### `pass` — 아무것도 하지 않음 (자리 표시자)

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

## 루프-Else 절

Python의 독특한 `for/else`와 `while/else` 구조는 루프가 `break`를 만나지 않고 완료된 경우에만 `else` 블록을 실행합니다.

### for/else — 검색 패턴

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

### for/else — 항목 찾기

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

### 루프-Else 이해하기

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

## 중첩 루프 (Nested Loops)

루프 안에 다른 루프를 넣을 수 있습니다.

### 기본 중첩 루프

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

### 패턴: 쌍 찾기

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

### 패턴: 행렬 연산

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

### 중첩 루프 탈출하기

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

## 구조적 패턴 매칭 (Structural Pattern Matching, match/case)

Python 3.10에 도입된 `match/case`는 단순한 값 비교를 넘어서는 강력한 패턴 매칭을 제공합니다.

### 기본 값 매칭

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

### OR 패턴

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

### 시퀀스 패턴

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

### 매핑 패턴 (딕셔너리)

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

### 패턴의 가드 절 (Guard Clauses in Patterns)

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

### 클래스 패턴

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

## 일반적인 제어 흐름 패턴

### 패턴 1: 누적기 (Accumulator)

여러 반복에 걸쳐 결과를 수집합니다.

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

### 패턴 2: 플래그 변수 (Flag Variable)

불리언을 사용하여 조건이 충족되었는지 추적합니다.

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

### 패턴 3: 감시 값 (Sentinel Value)

특수 값을 사용하여 종료를 신호합니다.

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

### 패턴 4: 슬라이딩 윈도우 (Sliding Window)

겹치는 그룹으로 요소를 처리합니다.

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

### 패턴 5: 투 포인터 (Two Pointers)

양쪽 끝에서 순회하기 위해 두 개의 인덱스를 사용합니다.

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

### 패턴 6: 상태 기계 (State Machine)

변수를 사용하여 현재 상태를 추적합니다.

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

## 성능 팁

### 올바른 루프 구조 선택

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

### 반복되는 속성 조회 피하기

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

## 연습문제

1. **FizzBuzz**: 1부터 100까지 숫자를 출력하세요. 3의 배수이면 "Fizz", 5의 배수이면 "Buzz", 둘 다의 배수이면 "FizzBuzz"를 출력하세요.

2. **숫자 추측 게임**: 1에서 100 사이의 난수를 생성하세요. 사용자가 추측하게 하고, "너무 높음" 또는 "너무 낮음" 힌트를 제공하세요. 추측 횟수를 세세요.

3. **소수 체 (Prime Sieve)**: 에라토스테네스의 체를 구현하여 N까지의 모든 소수를 찾으세요. 중첩 루프와 불리언 리스트를 사용하세요.

4. **삼각형 출력기**: 주어진 높이의 직각삼각형을 별표로 출력하는 함수를 작성하세요:
   ```
   *
   **
   ***
   ****
   *****
   ```

5. **비밀번호 검증기**: 모든 조건을 충족할 때까지 반복적으로 비밀번호를 묻는 루프를 작성하세요: 최소 8자, 대문자 최소 1개, 소문자 최소 1개, 숫자 최소 1개, 특수 문자 최소 1개.

6. **콜라츠 추측 (Collatz Conjecture)**: 양의 정수 n에 대해, n이 짝수이면 2로 나누고, 홀수이면 3을 곱하고 1을 더합니다. 1에 도달할 때까지 반복하세요. 시퀀스와 단계 수를 출력하세요.

7. **패턴 매칭 연습** (Python 3.10+ 필요): `["move", "north"]`, `["attack", "dragon"]`, `["use", "potion", "health"]`, `["quit"]`과 같은 "명령" 리스트를 `match/case`를 사용하여 처리하는 함수를 작성하세요.

8. **행렬 나선 (Matrix Spiral)**: NxN 행렬이 주어지면, 나선 순서 (바깥쪽 링부터, 그 다음 안쪽 링)로 요소를 출력하세요.

---

## 요약

| 구조 | 목적 | 핵심 요점 |
|------|------|----------|
| **if/elif/else** | 분기 | 첫 번째 일치하는 분기가 실행; else는 선택적 |
| **삼항** | 인라인 조건 | `x if cond else y`; 깊은 중첩 피하기 |
| **for** | 확정 반복 | 모든 이터러블을 순회; 인덱스에는 `enumerate()` 사용 |
| **while** | 무한 반복 | 조건이 참인 동안 반복; 무한 루프 주의 |
| **range()** | 정수 시퀀스 | `range(stop)`, `range(start, stop)`, `range(start, stop, step)` |
| **enumerate()** | 인덱스 + 값 | `enumerate(iterable, start=0)` |
| **zip()** | 병렬 순회 | 가장 짧은 것에서 멈춤; 길이가 다른 경우 `zip_longest` 사용 |
| **break** | 루프 종료 | 가장 안쪽 루프만 종료 |
| **continue** | 반복 건너뛰기 | 가장 안쪽 루프의 다음 반복으로 이동 |
| **pass** | 빈 자리 표시자 | 아무것도 하지 않는 빈 블록 |
| **for/else** | break 없음 감지 | else는 루프가 break 없이 완료된 경우에만 실행 |
| **match/case** | 패턴 매칭 | Python 3.10+; 값, 시퀀스, 매핑, 가드 지원 |
| **가드 절** | 조기 반환 | 가독성을 위해 중첩 조건문을 평탄화 |
| **누적기** | 결과 수집 | 내장 함수 선호: `sum()`, `max()`, `min()`, `"".join()` |

---

**이전**: [연산자와 표현식](./03_Operators_and_Expressions.md) | **다음**: [함수](./05_Functions.md)

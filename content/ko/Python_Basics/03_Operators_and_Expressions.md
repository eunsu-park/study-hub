# 연산자와 표현식

**이전**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md) | **다음**: [제어 흐름](./04_Control_Flow.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 내림 나눗셈 (Floor Division), 나머지 (Modulo), 거듭제곱 (Exponentiation)을 포함한 모든 산술 연산자를 사용할 수 있다
2. 비교 연산자를 적용하고 Python이 비교를 체이닝하는 방식을 이해할 수 있다
3. 논리 연산자 (`and`, `or`, `not`)로 조건을 결합하고 단락 평가 (Short-Circuit Evaluation)를 활용할 수 있다
4. 비트 연산자로 비트 수준의 조작을 수행할 수 있다
5. `in`과 `not in`으로 멤버십을, `is`와 `is not`으로 아이덴티티를 테스트할 수 있다
6. Python의 연산자 우선순위 규칙을 읽고 복잡한 표현식을 평가할 수 있다
7. 왈러스 연산자 (Walrus Operator, `:=`)를 사용하여 표현식 내에서 값을 할당할 수 있다
8. 일반적인 연산자 함정을 피하는 명확하고 관용적인 표현식을 작성할 수 있다

---

**표현식 (Expression)**은 값을 생성하는 코드 조각입니다. 연산자 (Operator)는 표현식의 구성 요소로 — 하나 이상의 피연산자를 받아 결과를 생성합니다. Python은 기본 산술을 넘어 체이닝 비교, 단락 논리, 왈러스 연산자를 포함하는 풍부한 연산자 세트를 제공합니다.

## 산술 연산자 (Arithmetic Operators)

산술 연산자는 숫자 피연산자에 대해 수학적 계산을 수행합니다.

### 기본 산술

```python
a = 15
b = 4

print(a + b)    # 19   Addition
print(a - b)    # 11   Subtraction
print(a * b)    # 60   Multiplication
print(a / b)    # 3.75 True division (always returns float)
print(a // b)   # 3    Floor division (rounds toward negative infinity)
print(a % b)    # 3    Modulo (remainder)
print(a ** b)   # 50625 Exponentiation (15^4)
print(-a)       # -15  Unary negation
print(+a)       # 15   Unary plus (rarely used)
```

### 진정한 나눗셈 vs 내림 나눗셈

진정한 나눗셈 (`/`)은 항상 float를 반환합니다. 내림 나눗셈 (`//`)은 가장 가까운 정수로 내림합니다.

```python
# True division always returns float
print(10 / 3)     # 3.3333333333333335
print(10 / 2)     # 5.0 (float, even when evenly divisible)
print(type(10/2))  # <class 'float'>

# Floor division rounds toward negative infinity
print(10 // 3)    # 3
print(-10 // 3)   # -4  (NOT -3! Rounds toward negative infinity)
print(10 // -3)   # -4

# Floor division with floats returns float
print(10.0 // 3)  # 3.0 (float result, but still floored)
print(7.5 // 2.5) # 3.0
```

### 나머지 연산자 (Modulo Operator)

나머지 연산자 `%`는 나눗셈의 나머지를 반환합니다. 내림 나눗셈 규칙을 따릅니다: `a == (a // b) * b + (a % b)`.

```python
# Basic modulo
print(10 % 3)    # 1   (10 = 3*3 + 1)
print(17 % 5)    # 2   (17 = 5*3 + 2)
print(20 % 4)    # 0   (evenly divisible)

# With negatives (follows floor division convention)
print(-10 % 3)   # 2   (because -10 // 3 == -4, and -4*3 + 2 == -10)
print(10 % -3)   # -2  (because 10 // -3 == -4, and -4*-3 + (-2) == 10)

# Common use cases
# 1. Check if a number is even or odd
number = 42
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")

# 2. Wrap around (circular indexing)
colors = ["red", "green", "blue"]
for i in range(10):
    print(f"Step {i}: {colors[i % len(colors)]}")
# Step 0: red, Step 1: green, Step 2: blue, Step 3: red, ...

# 3. Check divisibility
def is_leap_year(year):
    """Determine if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

print(is_leap_year(2024))  # True
print(is_leap_year(1900))  # False
print(is_leap_year(2000))  # True

# 4. Extract digits
number = 12345
ones = number % 10         # 5
tens = (number // 10) % 10 # 4
hundreds = (number // 100) % 10  # 3
```

### 거듭제곱 (Exponentiation)

```python
# Basic power
print(2 ** 10)    # 1024
print(3 ** 3)     # 27

# Fractional exponents (roots)
print(16 ** 0.5)  # 4.0  (square root)
print(27 ** (1/3))  # 3.0  (cube root)
print(2 ** -1)    # 0.5  (reciprocal)

# Large powers (Python handles big integers natively)
print(2 ** 100)
# 1267650600228229401496703205376

# Built-in pow() function with optional modulus
print(pow(2, 10))       # 1024 (same as 2 ** 10)
print(pow(2, 10, 1000)) # 24   (2^10 % 1000, computed efficiently)
# This is crucial for cryptography: modular exponentiation
```

### divmod() — 몫과 나머지를 함께

```python
# divmod returns (quotient, remainder)
q, r = divmod(17, 5)
print(f"17 / 5 = {q} remainder {r}")  # 17 / 5 = 3 remainder 2

# Practical: convert seconds to hours, minutes, seconds
total_seconds = 7384
hours, remaining = divmod(total_seconds, 3600)
minutes, seconds = divmod(remaining, 60)
print(f"{hours}h {minutes}m {seconds}s")  # 2h 3m 4s

# Practical: convert cents to dollars and cents
total_cents = 1599
dollars, cents = divmod(total_cents, 100)
print(f"${dollars}.{cents:02d}")  # $15.99
```

---

## 비교 연산자 (Comparison Operators)

비교 연산자는 두 값을 비교하고 불리언 (`True` 또는 `False`)을 반환합니다.

```python
a = 10
b = 20

print(a == b)    # False  Equal to
print(a != b)    # True   Not equal to
print(a < b)     # True   Less than
print(a > b)     # False  Greater than
print(a <= b)    # True   Less than or equal to
print(a >= b)    # False  Greater than or equal to
```

### 다른 타입 간 비교

```python
# Numeric types can be compared across types
print(1 == 1.0)      # True  (int vs float)
print(1 == True)      # True  (bool is a subclass of int)
print(0 == False)     # True
print(1 == 1+0j)      # True  (int vs complex)

# Strings compare lexicographically (Unicode code points)
print("apple" < "banana")  # True  (a < b)
print("Apple" < "apple")   # True  (A=65, a=97)
print("abc" < "abd")       # True  (first difference: c < d)
print("abc" < "abcd")      # True  (prefix is less than longer string)

# Comparing incompatible types raises TypeError (in most cases)
# print(1 < "2")  # TypeError: '<' not supported between instances of 'int' and 'str'
# Exception: == and != work across all types
print(1 == "1")   # False (different types, different values)
```

### 체이닝 비교 (Chained Comparisons)

Python은 독특하게 수학적 표기법처럼 읽히는 **체이닝 비교**를 지원합니다:

```python
x = 15

# These are equivalent
print(10 < x < 20)          # True
print(10 < x and x < 20)    # True (expanded form)

# Multiple chains
print(1 < 2 < 3 < 4 < 5)   # True
print(1 < 2 < 3 > 2 > 1)   # True  (each adjacent pair is compared)

# Practical use cases
age = 25
print(18 <= age < 65)  # True (working age)

grade = 85
print(80 <= grade < 90)  # True (B grade)

# Chained equality
a = b = c = 5
print(a == b == c)  # True

# Important: each operand is evaluated only once
# In `a < f() < b`, f() is called only once
```

### 객체 비교: == vs is

```python
# == checks VALUE equality
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True (same contents)

# 'is' checks IDENTITY (same object in memory)
print(a is b)   # False (different objects)

c = a
print(a is c)   # True (same object)

# Always use 'is' for None, True, False
x = None
print(x is None)       # True (correct)
print(x == None)       # True (works but not Pythonic)

# Why? Custom classes can override ==
class Weird:
    def __eq__(self, other):
        return True   # Claims to be equal to everything!

w = Weird()
print(w == None)     # True (misleading!)
print(w is None)     # False (correct)
```

---

## 논리 연산자 (Logical Operators)

논리 연산자는 불리언 표현식을 결합합니다.

### and, or, not

```python
# and: True if BOTH operands are true
print(True and True)    # True
print(True and False)   # False
print(False and True)   # False
print(False and False)  # False

# or: True if EITHER operand is true
print(True or True)     # True
print(True or False)    # True
print(False or True)    # True
print(False or False)   # False

# not: Inverts the boolean value
print(not True)     # False
print(not False)    # True
print(not 0)        # True
print(not "hello")  # False
```

### 단락 평가 (Short-Circuit Evaluation)

Python은 논리 표현식을 지연 평가합니다 — 결과가 결정되는 즉시 평가를 중단합니다.

```python
# 'and' stops at the first falsy value
# 'or' stops at the first truthy value

# and: returns the first falsy value, or the last value if all truthy
print(1 and 2 and 3)       # 3 (all truthy, returns last)
print(1 and 0 and 3)       # 0 (first falsy)
print(1 and "" and 3)      # '' (first falsy)
print([] and "hello")      # [] (first falsy)

# or: returns the first truthy value, or the last value if all falsy
print(0 or "" or "hello")  # 'hello' (first truthy)
print(0 or [] or None)     # None (all falsy, returns last)
print(1 or 2)              # 1 (first truthy)
print("hi" or "bye")       # 'hi' (first truthy)
```

### 실용적인 단락 평가 패턴

```python
# 1. Default values using 'or'
name = ""
display_name = name or "Anonymous"
print(display_name)  # "Anonymous"

config_timeout = 0  # Caution: 0 is falsy!
timeout = config_timeout or 30
print(timeout)  # 30 (probably NOT what you wanted if 0 is valid)
# Better: use 'if ... is None' or a conditional expression
timeout = config_timeout if config_timeout is not None else 30

# 2. Guard clauses using 'and'
data = {"users": [{"name": "Alice"}]}

# Short-circuit prevents KeyError/IndexError
users = data.get("users") and data["users"][0].get("name")
print(users)  # "Alice"

empty_data = {}
users = empty_data.get("users") and empty_data["users"][0].get("name")
print(users)  # None (short-circuited at get("users") returning None)

# 3. Conditional function calls
DEBUG = False
DEBUG and print("Debug message")  # print() is never called

DEBUG = True
DEBUG and print("Debug message")  # "Debug message"
```

### 드모르간 법칙 (De Morgan's Laws)

이 법칙들은 복잡한 불리언 표현식을 단순화하고 추론하는 데 도움이 됩니다:

```python
# De Morgan's Laws:
# not (A and B) == (not A) or (not B)
# not (A or B)  == (not A) and (not B)

a, b = True, False

# Verify first law
print(not (a and b))            # True
print((not a) or (not b))       # True

# Verify second law
print(not (a or b))             # False
print((not a) and (not b))      # False

# Practical: simplify a condition
# Instead of:
# if not (age >= 18 and has_id):
# Use:
# if age < 18 or not has_id:
```

---

## 비트 연산자 (Bitwise Operators)

비트 연산자는 정수의 이진 표현에 대해 작동합니다.

### 개요

```python
a = 0b1100   # 12 in decimal
b = 0b1010   # 10 in decimal

print(f"a     = {a:04b} ({a})")   # 1100 (12)
print(f"b     = {b:04b} ({b})")   # 1010 (10)
print(f"a & b = {a & b:04b} ({a & b})")   # 1000 (8)   AND
print(f"a | b = {a | b:04b} ({a | b})")   # 1110 (14)  OR
print(f"a ^ b = {a ^ b:04b} ({a ^ b})")   # 0110 (6)   XOR
print(f"~a    = {~a} (inverts all bits)")  # -13        NOT
print(f"a << 2 = {a << 2:08b} ({a << 2})")  # 110000 (48)  Left shift
print(f"a >> 2 = {a >> 2:04b} ({a >> 2})")  # 0011 (3)    Right shift
```

### 비트 진리표

```
 A | B | A & B | A | B | A ^ B
---|---|-------|-------|------
 0 | 0 |   0   |   0   |  0
 0 | 1 |   0   |   1   |  1
 1 | 0 |   0   |   1   |  1
 1 | 1 |   1   |   1   |  0
```

### 실용적인 비트 연산 예제

```python
# 1. Check if a number is even or odd (faster than modulo)
def is_even(n):
    """Check if n is even using bitwise AND."""
    return (n & 1) == 0

print(is_even(42))  # True
print(is_even(43))  # False

# 2. Bit flags (permissions system)
READ = 0b001     # 1
WRITE = 0b010    # 2
EXECUTE = 0b100  # 4

# Set permissions
user_perms = READ | WRITE  # 0b011 = 3

# Check permissions
can_read = bool(user_perms & READ)      # True
can_write = bool(user_perms & WRITE)    # True
can_execute = bool(user_perms & EXECUTE)  # False

# Add permission
user_perms |= EXECUTE  # 0b111 = 7

# Remove permission
user_perms &= ~WRITE   # 0b101 = 5

# Toggle permission
user_perms ^= READ     # 0b100 = 4 (READ was on, now off)
user_perms ^= READ     # 0b101 = 5 (READ was off, now on)

# 3. Powers of two
def is_power_of_two(n):
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0

print(is_power_of_two(16))   # True  (10000 & 01111 = 0)
print(is_power_of_two(18))   # False (10010 & 10001 = 10000)

# 4. Swap without temporary variable
a = 5
b = 3
a ^= b   # a = 5 ^ 3 = 6
b ^= a   # b = 3 ^ 6 = 5
a ^= b   # a = 6 ^ 5 = 3
print(a, b)  # 3 5

# (Note: Python's tuple swap `a, b = b, a` is preferred in practice)

# 5. Bit shifting for multiplication/division by powers of 2
x = 7
print(x << 1)  # 14  (multiply by 2)
print(x << 3)  # 56  (multiply by 8)
print(x >> 1)  # 3   (integer divide by 2)
```

---

## 멤버십 연산자 (Membership Operators)

멤버십 연산자는 값이 시퀀스에 포함되어 있는지 테스트합니다.

```python
# 'in' and 'not in' work with any iterable
numbers = [1, 2, 3, 4, 5]
print(3 in numbers)       # True
print(6 in numbers)       # False
print(6 not in numbers)   # True

# Strings
greeting = "Hello, World!"
print("World" in greeting)     # True
print("world" in greeting)     # False (case-sensitive)
print("xyz" not in greeting)   # True

# Tuples
coordinates = (3, 4, 5)
print(4 in coordinates)   # True

# Dictionaries (checks KEYS by default)
config = {"host": "localhost", "port": 5432, "debug": True}
print("host" in config)      # True (key exists)
print("localhost" in config)  # False (not a key, it's a value)
print("timeout" not in config)  # True

# Check dictionary values explicitly
print("localhost" in config.values())  # True
print(5432 in config.values())         # True

# Sets (O(1) lookup -- very fast)
valid_statuses = {"active", "pending", "archived"}
print("active" in valid_statuses)    # True
print("deleted" in valid_statuses)   # False

# range (O(1) for integers)
r = range(1_000_000)
print(999_999 in r)   # True (instant, does not iterate)
print(1_000_000 in r)  # False
```

### 성능 고려사항

```python
import time

# Membership testing performance comparison
large_list = list(range(1_000_000))
large_set = set(range(1_000_000))

# List: O(n) -- must scan sequentially
# Set:  O(1) -- hash lookup

# For frequent membership checks, convert to a set
data = ["apple", "banana", "cherry", "date", "elderberry"]
lookup_set = set(data)  # Convert once, check many times

target = "cherry"
if target in lookup_set:  # O(1) instead of O(n)
    print(f"Found {target}")
```

---

## 아이덴티티 연산자 (Identity Operators)

아이덴티티 연산자는 두 변수가 메모리에서 **동일한 객체**를 참조하는지 확인합니다.

```python
# 'is' and 'is not'
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)       # True  (same value)
print(a is b)       # False (different objects)
print(a is c)       # True  (same object)
print(a is not b)   # True  (different objects)
```

### `is`를 사용하는 경우

```python
# Use 'is' for singleton comparisons
x = None
if x is None:
    print("x is None")

if x is not None:
    print("x has a value")

# Use 'is' for True/False (rare -- usually just use the value directly)
flag = True
if flag is True:     # Explicit identity check (uncommon)
    pass
if flag:             # Preferred (Pythonic)
    pass

# Use 'is' for sentinel objects
_MISSING = object()  # Create a unique sentinel

def get_value(data, key, default=_MISSING):
    """Get a value from data, distinguishing None from missing."""
    value = data.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise KeyError(key)
        return default
    return value

data = {"name": "Alice", "age": None}
print(get_value(data, "name"))       # "Alice"
print(get_value(data, "age"))        # None (exists, value is None)
# get_value(data, "email")           # KeyError
print(get_value(data, "email", ""))  # "" (default provided)
```

### 정수 캐싱 주의사항

```python
# Python caches small integers (-5 to 256) for performance
a = 256
b = 256
print(a is b)  # True (cached)

a = 257
b = 257
print(a is b)  # May be False (not cached)
# NEVER use 'is' to compare integer values -- always use ==
```

---

## 연산자 우선순위 (Operator Precedence)

표현식에 여러 연산자가 포함된 경우, Python은 우선순위 계층에 따라 평가합니다. 높은 우선순위의 연산자가 더 강하게 바인딩됩니다.

### 우선순위 표 (높은 것에서 낮은 것 순)

| 우선순위 | 연산자 | 설명 |
|---------|--------|------|
| 1 (최고) | `()` | 괄호 (그룹화) |
| 2 | `**` | 거듭제곱 |
| 3 | `+x`, `-x`, `~x` | 단항 플러스, 마이너스, 비트 NOT |
| 4 | `*`, `/`, `//`, `%` | 곱셈, 나눗셈, 나머지 |
| 5 | `+`, `-` | 덧셈, 뺄셈 |
| 6 | `<<`, `>>` | 비트 시프트 |
| 7 | `&` | 비트 AND |
| 8 | `^` | 비트 XOR |
| 9 | `\|` | 비트 OR |
| 10 | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `is not`, `in`, `not in` | 비교 |
| 11 | `not` | 논리 NOT |
| 12 | `and` | 논리 AND |
| 13 | `or` | 논리 OR |
| 14 | `:=` | 왈러스 연산자 |
| 15 (최저) | `if ... else` | 조건 표현식 |

### 우선순위 예제

```python
# Exponentiation before negation
print(-2 ** 2)     # -4  (not 4!) -- equivalent to -(2 ** 2)
print((-2) ** 2)   # 4   (parentheses override)

# Multiplication before addition
print(2 + 3 * 4)   # 14 (not 20) -- equivalent to 2 + (3 * 4)
print((2 + 3) * 4) # 20 (parentheses override)

# Comparison before logical operators
print(1 < 2 and 3 < 4)   # True -- equivalent to (1 < 2) and (3 < 4)

# not before and, and before or
print(True or False and False)   # True -- equivalent to True or (False and False)
print((True or False) and False) # False (parentheses override)

# Right-to-left associativity for **
print(2 ** 3 ** 2)    # 512 -- equivalent to 2 ** (3 ** 2) = 2 ** 9
print((2 ** 3) ** 2)  # 64  (left-to-right would give this)
```

### 모범 사례: 명확성을 위해 괄호 사용

```python
# Even when not required, parentheses improve readability
# UNCLEAR
result = a + b * c // d ** e % f

# CLEAR
result = a + ((b * c) // (d ** e) % f)

# For boolean expressions, always use parentheses
if (age >= 18 and has_license) or is_emergency:
    pass

# For chained comparisons, parentheses are unnecessary but OK
if 0 <= index < len(items):
    pass
```

---

## 왈러스 연산자 (Walrus Operator, `:=`)

Python 3.8에서 도입된 **왈러스 연산자** (`:=`)는 표현식의 일부로 변수에 값을 할당합니다. 공식적으로 "할당 표현식 (Assignment Expression)" 또는 "이름 표현식 (Named Expression)"이라고 합니다.

### 기본 구문

```python
# Traditional approach: compute, then check
value = len("Hello, World!")
if value > 10:
    print(f"Long string: {value} characters")

# Walrus operator: assign and check in one step
if (n := len("Hello, World!")) > 10:
    print(f"Long string: {n} characters")
```

### 일반적인 사용 사례

#### 1. 입력이 있는 While 루프

```python
# Traditional
while True:
    line = input("Enter text (or 'quit'): ")
    if line == "quit":
        break
    print(f"You entered: {line}")

# With walrus operator
while (line := input("Enter text (or 'quit'): ")) != "quit":
    print(f"You entered: {line}")
```

#### 2. 계산이 포함된 필터링

```python
# Traditional: compute twice or use a separate variable
results = []
for x in range(20):
    y = x ** 2 + x + 1
    if y > 100:
        results.append(y)

# With walrus operator in a list comprehension
results = [y for x in range(20) if (y := x ** 2 + x + 1) > 100]
print(results)  # [111, 133, 157, 183, 211, 241, 273, 307, 343, 381]
```

#### 3. 정규 표현식 매칭

```python
import re

text = "Contact us at support@example.com for help"

# Traditional
match = re.search(r"[\w.]+@[\w.]+", text)
if match:
    email = match.group()
    print(f"Found email: {email}")

# With walrus operator
if (match := re.search(r"[\w.]+@[\w.]+", text)):
    print(f"Found email: {match.group()}")
```

#### 4. 청크 단위로 파일 읽기

```python
# Read a file in fixed-size chunks
with open("large_file.bin", "rb") as f:
    while (chunk := f.read(8192)):
        process(chunk)  # process each chunk
```

### 왈러스 연산자 가이드라인

```python
# DO use when it eliminates redundant computation
# DO use when it simplifies while-loop patterns
# DO use in comprehensions for expensive operations

# DO NOT use when it reduces readability
# BAD: too clever
if (x := (y := f(a)) + (z := g(b))) > 0:
    print(x, y, z)

# GOOD: keep it simple
y = f(a)
z = g(b)
x = y + z
if x > 0:
    print(x, y, z)
```

---

## 조건 표현식 (삼항 연산자, Ternary Operator)

Python의 조건 표현식은 한 줄에 간결한 `if/else`를 제공합니다.

```python
# Syntax: value_if_true if condition else value_if_false

age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # "adult"

# Equivalent to:
if age >= 18:
    status = "adult"
else:
    status = "minor"

# Can be used in any expression context
print("even" if 4 % 2 == 0 else "odd")  # "even"

x = 10
result = x * 2 if x > 0 else -x
print(result)  # 20

# Nested ternary (use sparingly -- readability suffers)
score = 85
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
print(grade)  # "B"

# Better as a function or if/elif chain for complex logic
def get_grade(score):
    """Convert a numeric score to a letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
```

---

## 실용적인 패턴

### 패턴 1: 값 클램핑 (Clamping)

```python
def clamp(value, minimum, maximum):
    """Restrict value to the range [minimum, maximum]."""
    return max(minimum, min(value, maximum))

print(clamp(15, 0, 10))   # 10
print(clamp(-5, 0, 10))   # 0
print(clamp(7, 0, 10))    # 7

# With chained comparisons for the check
value = 15
if not (0 <= value <= 10):
    print(f"Value {value} is out of range [0, 10]")
```

### 패턴 2: 안전한 나눗셈

```python
def safe_divide(a, b, default=0.0):
    """Divide a by b, returning default if b is zero."""
    return a / b if b != 0 else default

print(safe_divide(10, 3))     # 3.333...
print(safe_divide(10, 0))     # 0.0
print(safe_divide(10, 0, float("inf")))  # inf
```

### 패턴 3: 불리언 단순화

```python
# Unnecessary comparison to True/False
# BAD
if is_valid == True:
    pass

if is_empty == False:
    pass

# GOOD
if is_valid:
    pass

if not is_empty:
    pass

# Unnecessary bool() conversion
# BAD
if bool(items):
    pass

# GOOD
if items:
    pass

# Converting to bool explicitly (when you need a bool value, not truthiness)
has_items = bool(items)  # This is appropriate when storing as a boolean
```

### 패턴 4: 표현식 디버깅

```python
# Use parentheses and intermediate variables for debugging
# Instead of this hard-to-debug one-liner:
result = (a + b) * (c - d) / (e ** f) if g > 0 else default

# Break it apart:
numerator = (a + b) * (c - d)
denominator = e ** f
if g > 0:
    result = numerator / denominator
else:
    result = default
```

### 패턴 5: 순서를 유지하며 고유 값 수집

```python
# Using dict.fromkeys() preserves insertion order and removes duplicates
items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique_ordered = list(dict.fromkeys(items))
print(unique_ordered)  # [3, 1, 4, 5, 9, 2, 6]
```

---

## 일반적인 함정

### 1. 정수 나눗셈 놀라움

```python
# Python 3 always returns float for /
print(10 / 2)    # 5.0 (not 5!)
print(type(10 / 2))  # <class 'float'>

# Use // for integer result
print(10 // 2)   # 5
print(type(10 // 2))  # <class 'int'>
```

### 2. 체이닝 할당 vs 비교

```python
# This is assignment, not comparison
a = b = c = 0   # All three are 0

# This is chained comparison
print(a == b == c)  # True (0 == 0 == 0)

# This is NOT what you might expect
# a = b == c  means  a = (b == c)  which is  a = True
a = b == c
print(a)  # True (boolean result of comparison)
```

### 3. 가변 기본 인수 (Mutable Default Arguments)

```python
# BAD: mutable default argument
def append_to(element, target=[]):
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2]  -- Surprise! The list persists

# GOOD: use None as default
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [2]  -- Fresh list each time
```

### 4. 부동소수점 등가성

```python
# NEVER compare floats with ==
# BAD
if 0.1 + 0.2 == 0.3:
    print("Equal")

# GOOD
import math
if math.isclose(0.1 + 0.2, 0.3):
    print("Equal")

# With custom tolerance
if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
    print("Close enough")
```

---

## 연습문제

1. **표현식 평가기**: 코드를 실행하지 않고 각 표현식의 결과를 예측하세요. 그런 다음 REPL에서 확인하세요:
   - `2 ** 3 ** 2`
   - `-1 ** 2`
   - `True + True + False`
   - `"hello" * 3`
   - `15 // -4`
   - `not 0 or not 1`

2. **윤년 함수**: 논리 연산자만 사용하여 (if/else 없이) `is_leap_year(year)` 함수를 작성하세요.

3. **비트 카운터**: 양의 정수의 이진 표현에서 1-비트의 수를 세는 함수를 작성하세요 (`bin()`이나 문자열 메서드를 사용하지 않고).

4. **왈러스 연산자 연습**: 리스트에서 줄을 읽고 비어 있지 않은 줄만 처리하는 루프를 왈러스 연산자를 사용하여 다시 작성하세요.

5. **체이닝 비교**: 체이닝 비교를 사용하여 카테고리 문자열을 반환하는 `classify_bmi(bmi)` 함수를 작성하세요: "underweight" (< 18.5), "normal" (18.5-24.9), "overweight" (25-29.9), 또는 "obese" (>= 30).

6. **연산자 오버로드**: 두 리스트에 `+`를 사용할 때, 문자열과 정수에 `*`를 사용할 때, 두 딕셔너리에 `|`를 사용할 때 (Python 3.9+) 어떤 일이 일어나는지 조사하고 설명하세요.

---

## 요약

| 범주 | 연산자 | 핵심 요점 |
|------|--------|----------|
| **산술** | `+`, `-`, `*`, `/`, `//`, `%`, `**` | `/`는 항상 float 반환; `//`는 음의 무한대 방향으로 내림 |
| **비교** | `==`, `!=`, `<`, `>`, `<=`, `>=` | 체이닝 가능: `a < b < c` |
| **논리** | `and`, `or`, `not` | 단락 평가; 불리언뿐만 아니라 실제 피연산자 값을 반환 |
| **비트** | `&`, `\|`, `^`, `~`, `<<`, `>>` | 이진 표현에 대해 작동; 플래그와 저수준 작업에 유용 |
| **멤버십** | `in`, `not in` | 모든 이터러블과 작동; 집합과 딕셔너리에서 O(1) |
| **아이덴티티** | `is`, `is not` | 같은 값이 아닌 같은 객체를 확인; `None` 검사에 사용 |
| **왈러스** | `:=` | 표현식 내에서 할당; while 루프와 컴프리헨션에 유용 |
| **삼항** | `x if cond else y` | 간결한 조건; 깊게 중첩하지 않기 |
| **우선순위** | 괄호가 최우선 | 의심될 때 명확성을 위해 괄호 추가 |

---

**이전**: [변수와 데이터 타입](./02_Variables_and_Data_Types.md) | **다음**: [제어 흐름](./04_Control_Flow.md)

# 변수와 데이터 타입

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [연산자와 표현식](./03_Operators_and_Expressions.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. Python의 할당 구문을 사용하여 변수를 생성하고 이름이 객체에 바인딩되는 방식을 설명할 수 있다
2. Python의 핵심 데이터 타입인 `int`, `float`, `complex`, `str`, `bool`, `None`을 구별할 수 있다
3. `type()`과 `isinstance()`를 사용하여 런타임에 객체 타입을 검사하고 확인할 수 있다
4. 내장 함수 (`int()`, `float()`, `str()`, `bool()`)를 사용하여 타입 간 변환을 수행할 수 있다
5. Python의 변수 명명 규칙과 `UPPER_CASE` 상수 규칙을 적용할 수 있다
6. 다중 할당, 튜플 언패킹, 복합 할당 연산자를 사용할 수 있다
7. 동적 타이핑 (Dynamic Typing)과 정적 타이핑 (Static Typing)의 차이를 설명할 수 있다
8. 정수 나눗셈, 나머지, 거듭제곱을 포함한 숫자 연산을 수행할 수 있다

---

모든 프로그램은 데이터를 조작합니다. Python에서 데이터는 **객체 (Object)**에 저장되며, **변수 (Variable)**를 해당 객체를 가리키는 이름으로 사용합니다. 변수가 어떻게 작동하는지와 Python이 어떤 종류의 데이터를 제공하는지를 이해하는 것이 이후 모든 레슨의 기반이 됩니다.

## 변수와 할당 (Assignment)

### 변수란 무엇인가?

Python에서 변수는 데이터를 담는 상자가 아닙니다. 메모리에 있는 객체를 참조하는 (또는 "가리키는") **이름**입니다. 다음과 같이 작성하면:

```python
x = 42
```

Python은 세 가지를 수행합니다:

1. 값 `42`를 가진 정수 객체를 메모리에 생성합니다.
2. 이름 `x`를 생성합니다 (아직 존재하지 않는 경우).
3. 이름 `x`를 정수 객체에 바인딩합니다.

이 구분은 중요합니다. 여러 이름이 동일한 객체를 참조할 수 있습니다:

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

### 할당 구문

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

### 다중 할당 (Multiple Assignment)

Python은 여러 형태의 다중 할당을 지원합니다:

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

### 복합 할당 (Augmented Assignment)

복합 할당 연산자는 연산과 할당을 결합합니다:

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

모든 복합 할당 연산자:

| 연산자 | 동등 표현 | 설명 |
|--------|----------|------|
| `+=` | `x = x + y` | 더하고 할당 |
| `-=` | `x = x - y` | 빼고 할당 |
| `*=` | `x = x * y` | 곱하고 할당 |
| `/=` | `x = x / y` | 나누고 할당 |
| `//=` | `x = x // y` | 내림 나눗셈 후 할당 |
| `%=` | `x = x % y` | 나머지 연산 후 할당 |
| `**=` | `x = x ** y` | 거듭제곱 후 할당 |
| `&=` | `x = x & y` | 비트 AND 후 할당 |
| `\|=` | `x = x \| y` | 비트 OR 후 할당 |
| `^=` | `x = x ^ y` | 비트 XOR 후 할당 |
| `>>=` | `x = x >> y` | 오른쪽 시프트 후 할당 |
| `<<=` | `x = x << y` | 왼쪽 시프트 후 할당 |

---

## 동적 타이핑 (Dynamic Typing)

Python은 **동적으로 타이핑되는** 언어입니다. 이는 다음을 의미합니다:

1. 변수에는 고정된 타입이 없습니다.
2. 타입은 변수가 아닌 **객체**에 연결됩니다.
3. 타입 검사는 컴파일 타임이 아닌 **런타임**에 수행됩니다.

```python
# The same variable can hold different types over its lifetime
value = 42          # int
print(type(value))  # <class 'int'>

value = "hello"     # str
print(type(value))  # <class 'str'>

value = [1, 2, 3]   # list
print(type(value))  # <class 'list'>
```

### 동적 타이핑 vs 정적 타이핑

```python
# Python (dynamic typing)
x = 10        # No type declaration needed
x = "hello"   # Perfectly valid -- x changes type

# C (static typing) -- for comparison
# int x = 10;        // Must declare the type
# x = "hello";       // ERROR: cannot assign string to int
```

### 타입 힌트 (Type Hints) (선택적 어노테이션)

Python 3.5+에서는 **타입 힌트**를 지원합니다 — 예상되는 타입을 문서화하는 선택적 어노테이션입니다. 런타임에 타입을 강제하지는 않지만 가독성과 정적 분석 도구에 도움이 됩니다.

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

## 숫자 타입 (Numeric Types)

Python은 세 가지 내장 숫자 타입을 제공합니다: `int`, `float`, `complex`.

### 정수 (`int`)

Python 정수는 **임의 정밀도 (Arbitrary Precision)**를 가집니다 — 메모리가 허용하는 한 크기에 제한이 없습니다.

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

### 정수 연산

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

### 부동소수점 수 (`float`)

부동소수점 수는 IEEE 754 배정밀도 (64비트) 형식을 사용하여 실수를 표현합니다.

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

### 부동소수점 정밀도 함정

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

### 복소수 (`complex`)

Python은 복소수에 대한 내장 지원을 가지고 있으며, 허수부에 `j` (`i`가 아닌)를 사용합니다.

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

## 문자열 (`str`)

문자열은 유니코드 문자의 불변 시퀀스 (Immutable Sequence)입니다. 여기서는 간략히 다루며, 레슨 07에서 심도 있게 다룹니다.

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

### 일반적인 문자열 이스케이프

| 이스케이프 | 의미 |
|-----------|------|
| `\n` | 줄바꿈 |
| `\t` | 탭 |
| `\\` | 백슬래시 |
| `\'` | 작은따옴표 |
| `\"` | 큰따옴표 |
| `\0` | 널 문자 |
| `\u00e9` | 유니코드 문자 (예: 악센트가 있는 e) |

---

## 불리언 (`bool`)

불리언은 진리값을 표현합니다. 정확히 두 가지가 있습니다: `True`와 `False`.

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

### 참과 거짓 (Truthiness and Falsiness)

모든 Python 객체는 불리언 값을 가집니다. 다음 값들은 **거짓 (Falsy)**입니다 (`False`로 평가됨):

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

그 외 모든 것은 **참 (Truthy)**입니다:

```python
print(bool(1))           # True
print(bool(-1))          # True
print(bool(3.14))        # True
print(bool("hello"))     # True
print(bool([1, 2, 3]))   # True
print(bool({"a": 1}))    # True
```

이는 조건문에서 광범위하게 사용됩니다:

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

`None`은 Python의 널 값 (Null Value)으로 — 값의 부재를 나타냅니다. `None` 객체는 정확히 하나만 존재합니다.

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

### None의 일반적인 용도

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

## 타입 검사 (Type Inspection)

### `type()` — 객체의 타입 얻기

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

### `isinstance()` — 상속을 고려한 타입 검사

`isinstance()`는 상속을 존중하기 때문에 타입 검사에서 `type()`보다 선호됩니다:

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

### `id()` — 메모리 주소 얻기

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

> **참고**: Python은 성능을 위해 작은 정수 (일반적으로 -5에서 256)와 짧은 문자열을 캐시합니다. 값 비교에 `is`를 의존하지 마세요 — 값에는 항상 `==`을 사용하고, `is`는 `None`, `True`, `False` 및 명시적 아이덴티티 검사에만 사용하세요.

---

## 타입 변환 (Type Conversion)

Python은 타입 간 변환을 위한 내장 함수를 제공합니다.

### `int()` — 정수로 변환

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

### `float()` — 부동소수점 수로 변환

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

### `str()` — 문자열로 변환

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

### `bool()` — 불리언으로 변환

```python
# See the "Truthiness and Falsiness" section above
print(bool(0))      # False
print(bool(1))      # True
print(bool(""))     # False
print(bool("hi"))   # True
print(bool([]))     # False
print(bool([0]))    # True (non-empty list, even if it contains a falsy value)
```

### 안전한 변환 패턴

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

## 변수 명명 규칙

Python은 PEP 8에 정의된 강력한 명명 규칙을 가지고 있습니다:

### 규칙 (언어에 의해 강제됨)

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

### 관례 (커뮤니티에 의해 강제됨)

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

### Python 키워드

이 이름들은 예약되어 있으며 변수 이름으로 사용할 수 없습니다:

```python
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
#  'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
#  'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
#  'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
#  'try', 'while', 'with', 'yield']
```

### 명명 모범 사례

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

## 메모리 모델과 객체 아이덴티티

Python의 메모리 모델을 이해하면 미묘한 버그를 방지할 수 있습니다.

### 가변 (Mutable) vs 불변 (Immutable) 객체

| 타입 | 가변? | 예시 |
|------|-------|------|
| `int` | 아니오 | `42`, `-7` |
| `float` | 아니오 | `3.14`, `-2.5` |
| `str` | 아니오 | `"hello"`, `""` |
| `bool` | 아니오 | `True`, `False` |
| `tuple` | 아니오 | `(1, 2, 3)` |
| `frozenset` | 아니오 | `frozenset({1, 2})` |
| `list` | **예** | `[1, 2, 3]` |
| `dict` | **예** | `{"a": 1}` |
| `set` | **예** | `{1, 2, 3}` |

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

### 객체 복사하기

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

## 실용적인 예제

### 예제 1: 단위 변환

```python
# Convert miles to kilometers
miles = 26.2  # Marathon distance
km_per_mile = 1.60934

kilometers = miles * km_per_mile
print(f"{miles} miles = {kilometers:.2f} km")
# 26.2 miles = 42.16 km
```

### 예제 2: 원 계산

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

### 예제 3: 데이터 검증

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

출력:

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

### 예제 4: 변수 내성 (Variable Introspection)

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

## 연습문제

1. **타입 탐색기**: 각 기본 타입 (`int`, `float`, `complex`, `str`, `bool`, `None`)의 변수를 생성하고 각각의 타입과 값을 출력하는 스크립트를 작성하세요.

2. **교환 도전**: 세 변수 `a`, `b`, `c`를 교환하여 `a`가 `b`의 값을, `b`가 `c`의 값을, `c`가 `a`의 값을 가지도록 하세요. 한 줄로 작성하세요.

3. **정밀도 테스트**: `0.1 + 0.1 + 0.1 - 0.3`을 계산하세요. 결과가 정확히 0이 아닌 이유를 설명하세요. 결과를 0과 올바르게 비교하는 코드를 작성하세요.

4. **타입 변환기**: 문자열을 받아 변환 가능한 가장 구체적인 타입을 반환하는 함수 `smart_convert(value)`를 작성하세요 (먼저 `int`를 시도하고, 다음으로 `float`를 시도하고, 그 다음 원래 문자열을 반환).

5. **상수 파일**: `UPPER_CASE` 규칙을 따르는 최소 10개의 명명된 상수가 있는 `constants.py` 모듈을 만드세요. 다른 스크립트에서 임포트하여 사용하세요.

6. **메모리 탐정**: 동일한 내용을 가진 두 리스트를 만드세요. 같은 값이지만 (`==`) 동일한 객체는 아님을 (`is`) 증명하세요. 그런 다음 두 이름이 같은 리스트를 가리키는 상황을 만들고 동일한 객체임을 증명하세요.

---

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| **변수** | 값을 담는 상자가 아니라 객체를 참조하는 이름 |
| **동적 타이핑** | 변수는 타입을 변경할 수 있음; 타입은 객체에 속함 |
| **int** | 임의 정밀도 정수; 2진수, 8진수, 16진수 리터럴 지원 |
| **float** | IEEE 754 배정밀도; 정밀도 문제에 주의 |
| **complex** | `j` 표기법을 사용한 내장 복소수 지원 |
| **str** | 불변 유니코드 문자열; 포맷팅에 f-문자열 사용 |
| **bool** | `True`/`False`; `int`의 하위 클래스; 모든 객체에 진리값이 있음 |
| **None** | Python의 널; `== None`이 아닌 `is None`으로 검사 |
| **type() / isinstance()** | 타입 검사에는 `isinstance()` 선호 (상속 존중) |
| **타입 변환** | 명시적 변환에 `int()`, `float()`, `str()`, `bool()` 사용 |
| **명명** | 변수/함수에 `snake_case`, 상수에 `UPPER_CASE` |
| **가변성** | 불변 타입은 새 객체를 생성; 가변 타입은 제자리에서 변경 가능 |

---

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [연산자와 표현식](./03_Operators_and_Expressions.md)

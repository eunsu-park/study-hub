# 문자열과 텍스트 처리 (Strings and Text Processing)

**이전**: [자료구조](./06_Data_Structures.md) | **다음**: [OOP 기초](./08_OOP_Basics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 작은따옴표, 큰따옴표, 삼중따옴표, 원시 문자열 (Raw String)을 사용하여 문자열을 생성한다
2. 인덱싱과 슬라이싱을 사용하여 문자와 부분 문자열에 접근한다
3. 검색, 변환, 포매팅을 위한 필수 문자열 메서드를 적용한다
4. f-문자열 (f-string), `.format()`, `%` 포매팅을 사용하여 문자열 보간 (String Interpolation)을 수행한다
5. `re` 모듈을 사용하여 패턴 매칭과 치환을 위한 기본 정규 표현식 (Regular Expression)을 작성한다
6. 문자열의 불변성 (Immutability)과 그것이 성능에 미치는 영향을 이해한다
7. 이스케이프 문자, 여러 줄 문자열, 유니코드/UTF-8 인코딩을 다룬다
8. 문자열 메서드와 정규 표현식을 결합하여 실세계 텍스트 데이터를 처리한다

---

텍스트 처리는 거의 모든 프로그램의 핵심입니다. 설정 파일을 읽든, 사용자 입력을 파싱하든, 보고서를 생성하든, 웹 애플리케이션을 구축하든, 항상 문자열을 다루게 됩니다. 파이썬은 문자열을 풍부한 내장 메서드를 가진 일급 객체 (First-class Object)로 취급하며, `re` 모듈은 복잡한 패턴 매칭을 위한 강력한 정규 표현식 지원을 제공합니다.

## 1. 문자열 생성

### 따옴표

```python
# Single quotes
message = 'Hello, World!'

# Double quotes (identical behavior)
message = "Hello, World!"

# Use the other quote type to embed quotes
dialogue = 'She said, "Hello!"'
apostrophe = "It's a beautiful day"

# Escape quotes with backslash
mixed = "She said, \"It's fine.\""
print(mixed)  # She said, "It's fine."
```

### 삼중따옴표

삼중따옴표(`"""` 또는 `'''`)는 여러 줄 문자열을 생성합니다.

```python
poem = """Roses are red,
Violets are blue,
Python is great,
And so are you."""

print(poem)
# Roses are red,
# Violets are blue,
# Python is great,
# And so are you.

# Also used for docstrings
def greet(name):
    """Return a greeting message.

    Args:
        name: The person's name.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
```

### 원시 문자열

원시 문자열 (Raw String)은 백슬래시를 리터럴 문자로 취급합니다. 정규 표현식과 파일 경로에 특히 유용합니다.

```python
# Normal string: \n is a newline
print("Hello\nWorld")
# Hello
# World

# Raw string: \n is literal backslash + n
print(r"Hello\nWorld")
# Hello\nWorld

# Useful for Windows paths
path = r"C:\Users\alice\documents\file.txt"
print(path)  # C:\Users\alice\documents\file.txt

# Useful for regex patterns
import re
pattern = r"\d{3}-\d{4}"  # Match phone numbers like 555-1234
```

### 문자열 연결과 반복

```python
# Concatenation
first = "Hello"
last = "World"
full = first + ", " + last + "!"
print(full)  # Hello, World!

# Implicit concatenation (adjacent string literals)
message = ("This is a very long string "
           "that spans multiple lines "
           "in the source code.")
print(message)
# This is a very long string that spans multiple lines in the source code.

# Repetition
line = "-" * 40
print(line)  # ----------------------------------------

# Join (preferred for combining many strings)
words = ["Python", "is", "awesome"]
sentence = " ".join(words)
print(sentence)  # Python is awesome
```

---

## 2. 인덱싱과 슬라이싱

문자열은 문자의 시퀀스입니다. 리스트처럼 인덱싱과 슬라이싱을 지원합니다.

```python
text = "Hello, Python!"

# Indexing
print(text[0])     # H
print(text[7])     # P
print(text[-1])    # !
print(text[-7])    # P

# Length
print(len(text))   # 14

# Slicing
print(text[0:5])    # Hello
print(text[7:13])   # Python
print(text[:5])     # Hello
print(text[7:])     # Python!
print(text[::2])    # Hlo yhn
print(text[::-1])   # !nohtyP ,olleH

# Character membership
print("P" in text)      # True
print("Java" in text)   # False
print("Java" not in text)  # True
```

### 문자 순회

```python
word = "Python"

# Character by character
for char in word:
    print(char, end=" ")
print()  # P y t h o n

# With index
for i, char in enumerate(word):
    print(f"{i}: {char}")
# 0: P
# 1: y
# 2: t
# 3: h
# 4: o
# 5: n
```

---

## 3. 문자열 메서드

파이썬 문자열에는 40개 이상의 내장 메서드가 있습니다. 가장 자주 사용되는 것들을 소개합니다.

### 대소문자 변환

```python
text = "Hello, World!"

print(text.upper())       # HELLO, WORLD!
print(text.lower())       # hello, world!
print(text.title())       # Hello, World!
print(text.capitalize())  # Hello, world!
print(text.swapcase())    # hELLO, wORLD!

# Useful for case-insensitive comparison
user_input = "YES"
if user_input.lower() == "yes":
    print("User confirmed")
```

### 공백 처리

```python
text = "   Hello, World!   "

print(text.strip())    # "Hello, World!"        (both sides)
print(text.lstrip())   # "Hello, World!   "     (left only)
print(text.rstrip())   # "   Hello, World!"     (right only)

# Strip specific characters
data = "---Hello---"
print(data.strip("-"))   # "Hello"

# Center, left-justify, right-justify
word = "Python"
print(word.center(20, "-"))   # -------Python-------
print(word.ljust(20, "."))    # Python..............
print(word.rjust(20, "."))    # ..............Python
print(word.zfill(10))         # 0000Python
```

### 검색

```python
text = "Hello, World! Hello, Python!"

# find: returns index of first occurrence (-1 if not found)
print(text.find("Hello"))     # 0
print(text.find("Python"))    # 21
print(text.find("Java"))      # -1

# rfind: search from the right
print(text.rfind("Hello"))    # 14

# index: like find, but raises ValueError if not found
print(text.index("World"))    # 7
# text.index("Java")  # ValueError

# count: number of non-overlapping occurrences
print(text.count("Hello"))    # 2
print(text.count("l"))        # 4

# startswith and endswith
print(text.startswith("Hello"))     # True
print(text.endswith("Python!"))     # True
print(text.startswith(("Hello", "Hi", "Hey")))  # True (tuple of prefixes)
```

### 분할과 결합

```python
# split: break string into list
sentence = "Python is a great language"
words = sentence.split()
print(words)  # ['Python', 'is', 'a', 'great', 'language']

csv_data = "Alice,30,Seoul"
fields = csv_data.split(",")
print(fields)  # ['Alice', '30', 'Seoul']

# split with maxsplit
text = "one-two-three-four-five"
print(text.split("-", 2))  # ['one', 'two', 'three-four-five']

# rsplit: split from the right
print(text.rsplit("-", 2))  # ['one-two-three', 'four', 'five']

# splitlines: split by line boundaries
multiline = "Line 1\nLine 2\nLine 3"
print(multiline.splitlines())  # ['Line 1', 'Line 2', 'Line 3']

# join: combine list into string
words = ["Python", "is", "awesome"]
print(" ".join(words))       # Python is awesome
print(", ".join(words))      # Python, is, awesome
print("\n".join(words))
# Python
# is
# awesome

# partition: split into 3 parts (before, separator, after)
url = "https://www.example.com/path"
protocol, sep, rest = url.partition("://")
print(protocol)  # https
print(rest)      # www.example.com/path
```

### 치환

```python
text = "Hello, World! Hello, Python!"

# replace: replace all occurrences
print(text.replace("Hello", "Hi"))
# Hi, World! Hi, Python!

# replace with count limit
print(text.replace("Hello", "Hi", 1))
# Hi, World! Hello, Python!

# Chain replacements for simple cleanup
messy = "  Hello   World   "
clean = messy.strip().replace("   ", " ")
print(clean)  # "Hello World"
```

### 문자열 내용 검사

```python
# Alphabetic
print("Hello".isalpha())       # True
print("Hello123".isalpha())    # False

# Numeric
print("12345".isdigit())       # True
print("123.45".isdigit())      # False
print("12345".isnumeric())     # True

# Alphanumeric
print("Hello123".isalnum())    # True
print("Hello 123".isalnum())   # False (space)

# Whitespace
print("   ".isspace())         # True
print(" \t\n".isspace())       # True

# Case checks
print("HELLO".isupper())       # True
print("hello".islower())       # True
print("Hello World".istitle())  # True

# Identifier check (valid variable name)
print("my_var".isidentifier())    # True
print("2nd_var".isidentifier())   # False
print("class".isidentifier())     # True (but it is a keyword)

import keyword
print(keyword.iskeyword("class"))  # True
```

---

## 4. 문자열 포매팅

파이썬은 세 가지 주요 문자열 포매팅 접근 방식을 제공합니다.

### f-문자열 (포매팅된 문자열 리터럴) -- 권장

Python 3.6부터 사용 가능한 f-문자열은 가장 읽기 쉽고 성능이 좋은 옵션입니다.

```python
name = "Alice"
age = 30
score = 95.6789

# Basic interpolation
print(f"Name: {name}, Age: {age}")
# Name: Alice, Age: 30

# Expressions inside braces
print(f"Next year: {age + 1}")        # Next year: 31
print(f"Name upper: {name.upper()}")  # Name upper: ALICE

# Format specifiers
print(f"Score: {score:.2f}")          # Score: 95.68
print(f"Score: {score:10.2f}")        # Score:      95.68
print(f"Percentage: {score:.1f}%")    # Percentage: 95.7%

# Number formatting
big_number = 1234567890
print(f"With commas: {big_number:,}")         # With commas: 1,234,567,890
print(f"With underscores: {big_number:_}")    # With underscores: 1_234_567_890

# Padding and alignment
word = "hi"
print(f"|{word:<10}|")    # |hi        |  (left-aligned)
print(f"|{word:>10}|")    # |        hi|  (right-aligned)
print(f"|{word:^10}|")    # |    hi    |  (centered)
print(f"|{word:*^10}|")   # |****hi****|  (centered with fill)

# Integer formatting
num = 42
print(f"Decimal: {num:d}")       # Decimal: 42
print(f"Binary: {num:b}")        # Binary: 101010
print(f"Octal: {num:o}")         # Octal: 52
print(f"Hex: {num:x}")           # Hex: 2a
print(f"Hex upper: {num:X}")     # Hex upper: 2A
print(f"With prefix: {num:#x}")  # With prefix: 0x2a

# Date formatting
from datetime import datetime
now = datetime.now()
print(f"Date: {now:%Y-%m-%d %H:%M:%S}")
# Date: 2024-01-15 14:30:00 (example)

# Debugging with = (Python 3.8+)
x = 10
y = 20
print(f"{x = }, {y = }, {x + y = }")
# x = 10, y = 20, x + y = 30
```

### `.format()` 메서드

```python
# Positional arguments
print("Hello, {}! You are {} years old.".format("Alice", 30))
# Hello, Alice! You are 30 years old.

# Numbered arguments
print("{0} is {1}. {0} loves {2}.".format("Alice", 30, "Python"))
# Alice is 30. Alice loves Python.

# Named arguments
print("Name: {name}, Age: {age}".format(name="Alice", age=30))
# Name: Alice, Age: 30

# Format specifiers (same as f-strings)
print("{:.2f}".format(3.14159))     # 3.14
print("{:>10}".format("right"))     #      right
print("{:,}".format(1000000))       # 1,000,000

# Accessing object attributes and items
person = {"name": "Alice", "age": 30}
print("{p[name]} is {p[age]}".format(p=person))
# Alice is 30
```

### `%` 포매팅 (구 스타일)

```python
# Positional
print("Hello, %s! You are %d years old." % ("Alice", 30))
# Hello, Alice! You are 30 years old.

# Format specifiers
print("Pi is approximately %.4f" % 3.14159)  # Pi is approximately 3.1416
print("Hex: %x" % 255)                        # Hex: ff
print("Padded: %10s" % "right")               #     right

# Named (using dictionary)
print("%(name)s is %(age)d" % {"name": "Alice", "age": 30})
# Alice is 30
```

### 포매팅 방법 비교

| 기능 | f-문자열 | .format() | % |
|---------|----------|-----------|---|
| 파이썬 버전 | 3.6+ | 2.6+ | 전체 |
| 가독성 | 최고 | 좋음 | 보통 |
| 성능 | 가장 빠름 | 보통 | 보통 |
| 표현식 사용 | 가능 | 불가 | 불가 |
| 권장 | 예 | 호환성을 위해 | 레거시 전용 |

---

## 5. 이스케이프 문자

이스케이프 문자는 직접 타이핑할 수 없는 특수 문자를 나타냅니다.

```python
# Common escape characters
print("Hello\tWorld")    # Hello	World       (tab)
print("Hello\nWorld")    # Hello (newline) World
print("He said \"Hi\"")  # He said "Hi"
print('It\'s fine')      # It's fine
print("Backslash: \\")   # Backslash: \
print("Null: \0 end")    # Null:  end
print("Bell: \a")        # (system bell sound)

# Unicode escape
print("\u0041")           # A
print("\u00e9")           # e (e with acute accent)
print("\U0001F600")       # (grinning face emoji)
print("\N{GREEK SMALL LETTER ALPHA}")  # alpha

# Hex escape
print("\x41")             # A
print("\x48\x65\x6c\x6c\x6f")  # Hello
```

| 이스케이프 | 의미 |
|--------|---------|
| `\n` | 줄바꿈 |
| `\t` | 탭 |
| `\\` | 백슬래시 |
| `\'` | 작은따옴표 |
| `\"` | 큰따옴표 |
| `\r` | 캐리지 리턴 |
| `\0` | 널 문자 |
| `\uXXXX` | 유니코드 (16비트) |
| `\UXXXXXXXX` | 유니코드 (32비트) |
| `\xHH` | 16진수 값 |

---

## 6. 문자열 불변성

파이썬의 문자열은 불변 (Immutable)입니다 -- 한 번 생성되면 변경할 수 없습니다.

```python
text = "Hello"

# Cannot modify in place
# text[0] = "h"  # TypeError: 'str' object does not support item assignment

# Instead, create a new string
text = "h" + text[1:]
print(text)  # hello

# Or use replace
text = "Hello"
text = text.replace("H", "h")
print(text)  # hello
```

### 성능에 미치는 영향

```python
import time

# BAD: Concatenation in a loop creates a new string each time
# This is O(n^2) because each concatenation copies the entire string
def build_string_bad(n):
    result = ""
    for i in range(n):
        result += str(i)  # Creates a new string each iteration
    return result

# GOOD: Use a list and join at the end -- O(n)
def build_string_good(n):
    parts = []
    for i in range(n):
        parts.append(str(i))
    return "".join(parts)

# BEST: Use a list comprehension with join
def build_string_best(n):
    return "".join(str(i) for i in range(n))

# Timing comparison
n = 100000
start = time.time()
build_string_bad(n)
print(f"Concatenation: {time.time() - start:.3f}s")

start = time.time()
build_string_good(n)
print(f"List + join:   {time.time() - start:.3f}s")

start = time.time()
build_string_best(n)
print(f"Comprehension: {time.time() - start:.3f}s")
```

### 문자열 인터닝

파이썬은 성능을 위해 작은 문자열을 캐시합니다. 이것은 구현 세부사항으로 의존해서는 안 됩니다.

```python
a = "hello"
b = "hello"
print(a is b)    # True (interned -- same object)

a = "hello world!"
b = "hello world!"
print(a is b)    # May be True or False (implementation dependent)

# Always use == for string comparison, never 'is'
print(a == b)    # True (correct way to compare)
```

---

## 7. 정규 표현식

`re` 모듈은 정규 표현식 매칭을 제공합니다. 정규 표현식 (Regular Expression, regex)은 문자열 집합을 설명하는 패턴입니다.

### 기본 패턴 매칭

```python
import re

text = "My phone number is 555-1234 and my zip is 90210."

# search: find first match anywhere in string
match = re.search(r"\d{3}-\d{4}", text)
if match:
    print(f"Found: {match.group()}")      # Found: 555-1234
    print(f"Position: {match.start()}-{match.end()}")  # Position: 19-27

# match: match at the BEGINNING of string only
result = re.match(r"My", text)
print(result.group() if result else "No match")  # My

result = re.match(r"phone", text)
print(result if result else "No match")  # No match (not at beginning)

# fullmatch: match the ENTIRE string
print(re.fullmatch(r"\d+", "12345"))     # Match object
print(re.fullmatch(r"\d+", "123abc"))    # None
```

### 일반적인 정규식 패턴

| 패턴 | 의미 | 매칭 예시 |
|---------|---------|---------------|
| `\d` | 모든 숫자 | `5` |
| `\D` | 숫자가 아닌 문자 | `a` |
| `\w` | 단어 문자 (문자, 숫자, _) | `A`, `3`, `_` |
| `\W` | 단어가 아닌 문자 | `!`, ` ` |
| `\s` | 공백 문자 | ` `, `\t`, `\n` |
| `\S` | 공백이 아닌 문자 | `a`, `5` |
| `.` | 모든 문자 (줄바꿈 제외) | 모든 것 |
| `^` | 문자열의 시작 | |
| `$` | 문자열의 끝 | |
| `*` | 0회 이상 반복 | |
| `+` | 1회 이상 반복 | |
| `?` | 0회 또는 1회 | |
| `{n}` | 정확히 n회 반복 | |
| `{n,m}` | n회에서 m회 반복 | |
| `[abc]` | 문자 클래스 (a, b 또는 c) | |
| `[^abc]` | 부정 클래스 (a, b, c가 아닌) | |
| `(...)` | 캡처 그룹 | |
| `\|` | 대안 (또는) | |

### `findall` -- 모든 매칭 찾기

```python
import re

text = "Call 555-1234 or 555-5678. Emergency: 911."

# Find all phone numbers
phones = re.findall(r"\d{3}-\d{4}", text)
print(phones)  # ['555-1234', '555-5678']

# Find all numbers
numbers = re.findall(r"\d+", text)
print(numbers)  # ['555', '1234', '555', '5678', '911']

# Find all words
words = re.findall(r"[A-Za-z]+", text)
print(words)  # ['Call', 'or', 'Emergency']

# With groups: findall returns group contents
text = "alice@example.com, bob@test.org"
emails = re.findall(r"(\w+)@(\w+\.\w+)", text)
print(emails)  # [('alice', 'example.com'), ('bob', 'test.org')]
```

### `sub` -- 치환

```python
import re

text = "My phone is 555-1234 and fax is 555-5678"

# Replace all phone numbers
censored = re.sub(r"\d{3}-\d{4}", "XXX-XXXX", text)
print(censored)  # My phone is XXX-XXXX and fax is XXX-XXXX

# Replace with a function
def mask_phone(match):
    phone = match.group()
    return phone[:4] + "****"

masked = re.sub(r"\d{3}-\d{4}", mask_phone, text)
print(masked)  # My phone is 555-**** and fax is 555-****

# Clean up extra whitespace
messy = "Hello    World   Python    Rocks"
clean = re.sub(r"\s+", " ", messy)
print(clean)  # Hello World Python Rocks

# Remove HTML tags
html = "<h1>Title</h1><p>This is <b>bold</b> text.</p>"
plain = re.sub(r"<[^>]+>", "", html)
print(plain)  # TitleThis is bold text.
```

### 컴파일된 패턴

반복적으로 사용되는 패턴은 컴파일하면 성능이 향상됩니다.

```python
import re

# Compile the pattern once
email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

texts = [
    "Contact: alice@example.com",
    "Send to bob@test.org please",
    "No email here",
    "Multiple: a@b.com and c@d.org",
]

for text in texts:
    matches = email_pattern.findall(text)
    if matches:
        print(f"Found: {matches}")
# Found: ['alice@example.com']
# Found: ['bob@test.org']
# Found: ['a@b.com', 'c@d.org']
```

### 그룹과 이름이 있는 그룹

```python
import re

# Capturing groups with parentheses
date_text = "Today is 2024-01-15"
match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_text)
if match:
    print(f"Full match: {match.group()}")    # 2024-01-15
    print(f"Year: {match.group(1)}")          # 2024
    print(f"Month: {match.group(2)}")         # 01
    print(f"Day: {match.group(3)}")           # 15
    print(f"All groups: {match.groups()}")    # ('2024', '01', '15')

# Named groups with (?P<name>...)
pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
match = re.search(pattern, date_text)
if match:
    print(f"Year: {match.group('year')}")     # 2024
    print(f"Month: {match.group('month')}")   # 01
    print(f"Dict: {match.groupdict()}")
    # Dict: {'year': '2024', 'month': '01', 'day': '15'}
```

### 실용적인 정규식 예제

```python
import re

# Validate email (simple)
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

print(is_valid_email("user@example.com"))   # True
print(is_valid_email("invalid@.com"))       # False

# Extract URLs from text
text = "Visit https://example.com or http://test.org/page for details"
urls = re.findall(r"https?://[^\s]+", text)
print(urls)  # ['https://example.com', 'http://test.org/page']

# Parse log entries
log = "2024-01-15 14:30:22 ERROR Database connection failed"
pattern = r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<msg>.+)"
match = re.match(pattern, log)
if match:
    info = match.groupdict()
    print(f"[{info['level']}] {info['date']} - {info['msg']}")
    # [ERROR] 2024-01-15 - Database connection failed

# Password strength checker
def check_password(password):
    checks = {
        "length": len(password) >= 8,
        "uppercase": bool(re.search(r"[A-Z]", password)),
        "lowercase": bool(re.search(r"[a-z]", password)),
        "digit": bool(re.search(r"\d", password)),
        "special": bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)),
    }
    return checks

result = check_password("MyP@ss123")
for check, passed in result.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {check}: {status}")
# length: PASS
# uppercase: PASS
# lowercase: PASS
# digit: PASS
# special: PASS
```

### 정규식 플래그

```python
import re

# IGNORECASE (re.I)
print(re.findall(r"python", "Python PYTHON python", re.IGNORECASE))
# ['Python', 'PYTHON', 'python']

# MULTILINE (re.M): ^ and $ match start/end of each line
text = "Line 1\nLine 2\nLine 3"
print(re.findall(r"^Line", text, re.MULTILINE))
# ['Line', 'Line', 'Line']

# DOTALL (re.S): . matches newline too
text = "<div>\nHello\n</div>"
print(re.findall(r"<div>.*</div>", text))            # [] (no match)
print(re.findall(r"<div>.*</div>", text, re.DOTALL))  # ['<div>\nHello\n</div>']

# VERBOSE (re.X): allow comments and whitespace in pattern
phone_pattern = re.compile(r"""
    ^(\d{3})      # Area code
    [-.\s]?       # Optional separator
    (\d{3})       # First three digits
    [-.\s]?       # Optional separator
    (\d{4})$      # Last four digits
""", re.VERBOSE)

print(phone_pattern.match("555-123-4567"))   # Match
print(phone_pattern.match("555.123.4567"))   # Match
print(phone_pattern.match("555 123 4567"))   # Match
```

---

## 8. 여러 줄 문자열

### 삼중따옴표

```python
# Multi-line with triple quotes preserves all whitespace
text = """
This is line 1.
This is line 2.
    This is indented.
"""
print(text)
#
# This is line 1.
# This is line 2.
#     This is indented.
#

# textwrap.dedent removes common leading whitespace
import textwrap

def get_sql():
    query = textwrap.dedent("""\
        SELECT name, age
        FROM users
        WHERE active = true
        ORDER BY name""")
    return query

print(get_sql())
# SELECT name, age
# FROM users
# WHERE active = true
# ORDER BY name
```

### 줄 연속

```python
# Backslash continuation
long_string = "This is a very long string that " \
              "continues on the next line " \
              "and even the line after that."

# Parentheses continuation (preferred)
long_string = (
    "This is a very long string that "
    "continues on the next line "
    "and even the line after that."
)

print(long_string)
# This is a very long string that continues on the next line and even the line after that.
```

---

## 9. 문자열 인코딩 (UTF-8, encode/decode)

Python 3의 문자열은 기본적으로 유니코드입니다. 파일, 네트워크 또는 바이트를 다룰 때는 문자열과 바이트 사이의 변환이 필요합니다.

```python
# Encoding: str -> bytes
text = "Hello, World!"
encoded = text.encode("utf-8")
print(encoded)        # b'Hello, World!'
print(type(encoded))  # <class 'bytes'>

# Decoding: bytes -> str
decoded = encoded.decode("utf-8")
print(decoded)        # Hello, World!

# Unicode characters
korean = "안녕하세요"
utf8_bytes = korean.encode("utf-8")
print(utf8_bytes)     # b'\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94'
print(len(korean))    # 5 (characters)
print(len(utf8_bytes))  # 15 (bytes -- Korean chars are 3 bytes each in UTF-8)

# Different encodings
text = "cafe"
print(text.encode("utf-8"))     # b'cafe'
print(text.encode("ascii"))     # b'cafe'

text_accent = "cafe\u0301"      # cafe with combining acute accent
print(text_accent)              # cafe (with accent on e)
print(text_accent.encode("utf-8"))

# Handling encoding errors
text = "Hello \u00e9 World"  # e with acute accent
print(text.encode("ascii", errors="replace"))   # b'Hello ? World'
print(text.encode("ascii", errors="ignore"))    # b'Hello  World'
print(text.encode("ascii", errors="xmlcharrefreplace"))  # b'Hello &#233; World'
```

### ord()와 chr()

```python
# ord: character -> Unicode code point
print(ord("A"))     # 65
print(ord("a"))     # 97
print(ord("0"))     # 48
print(ord("\u00e9"))  # 233

# chr: code point -> character
print(chr(65))      # A
print(chr(97))      # a
print(chr(233))     # e (with accent)
print(chr(0x1F600)) # (grinning face emoji)
```

---

## 10. 실용적인 텍스트 처리 예제

### 예제: CSV 줄 파서

```python
def parse_csv_line(line, delimiter=","):
    """Parse a CSV line handling quoted fields.

    Args:
        line: A single CSV line string.
        delimiter: Field separator character.

    Returns:
        A list of field values.
    """
    fields = []
    current = []
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == delimiter and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    fields.append("".join(current).strip())
    return fields

line = 'Alice,30,"Seoul, Korea",Engineer'
print(parse_csv_line(line))
# ['Alice', '30', 'Seoul, Korea', 'Engineer']
```

### 예제: 텍스트 통계

```python
import re

def text_statistics(text):
    """Calculate various statistics about a text.

    Args:
        text: Input text to analyze.

    Returns:
        Dictionary with character, word, sentence, and paragraph counts.
    """
    # Character counts
    total_chars = len(text)
    non_space_chars = len(text.replace(" ", ""))

    # Word count
    words = text.split()
    word_count = len(words)

    # Sentence count (split on . ! ?)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Paragraph count (separated by blank lines)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraph_count = len([p for p in paragraphs if p.strip()])

    # Average word length
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

    # Most common words
    word_freq = {}
    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        if clean:
            word_freq[clean] = word_freq.get(clean, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:5]

    return {
        "characters": total_chars,
        "characters_no_spaces": non_space_chars,
        "words": word_count,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "avg_word_length": round(avg_word_len, 1),
        "top_words": top_words,
    }

sample = """Python is a great programming language. It is easy to learn.
Many developers love Python for its clean syntax.

Python supports multiple paradigms. You can write object-oriented,
functional, or procedural code. Python is very versatile!"""

stats = text_statistics(sample)
for key, value in stats.items():
    print(f"  {key}: {value}")
```

### 예제: 템플릿 엔진

```python
import re

def render_template(template, context):
    """Simple template engine replacing {{variable}} with values.

    Args:
        template: String with {{variable}} placeholders.
        context: Dictionary of variable names to values.

    Returns:
        Rendered string.
    """
    def replacer(match):
        key = match.group(1).strip()
        return str(context.get(key, match.group(0)))

    return re.sub(r"\{\{(.+?)\}\}", replacer, template)

template = """Dear {{name}},

Thank you for your order #{{order_id}}.
Your total is ${{total}}.

Best regards,
{{company}}"""

context = {
    "name": "Alice",
    "order_id": "12345",
    "total": "99.99",
    "company": "Python Shop",
}

print(render_template(template, context))
# Dear Alice,
#
# Thank you for your order #12345.
# Your total is $99.99.
#
# Best regards,
# Python Shop
```

---

## 11. 요약

| 주제 | 핵심 포인트 |
|-------|------------|
| 생성 | 작은따옴표, 큰따옴표, 삼중따옴표; `r""`로 원시 문자열 |
| 인덱싱/슬라이싱 | 0 기반; 음수 인덱싱; `[start:stop:step]` |
| 메서드 | `upper`, `lower`, `strip`, `split`, `join`, `replace`, `find`, `count` |
| f-문자열 | `f"{expr}"` -- 가장 읽기 쉽고 빠른 포매팅 |
| 이스케이프 문자 | `\n`, `\t`, `\\`, `\"`, 유니코드 이스케이프 |
| 불변성 | 제자리 수정 불가; 문자열 구축에는 `join` 사용 |
| 정규식 | `re.search`, `re.findall`, `re.sub`, `re.compile` |
| 인코딩 | `str.encode()`로 바이트로, `bytes.decode()`로 문자열로; 기본 UTF-8 |

---

## 연습문제

1. 문자열에서 모음(a, e, i, o, u)의 수를 대소문자 구분 없이 반환하는 함수 `count_vowels(text)`를 작성하세요.
2. 각 단어의 첫 글자를 대문자로 만드는 함수 `title_case(text)`를 작성하세요. 단, "the", "a", "an", "in", "on", "at", "of" 같은 짧은 단어는 첫 번째 단어가 아닌 한 제외합니다.
3. 정규식을 사용하여 텍스트에서 `YYYY-MM-DD` 형식의 모든 날짜를 추출하고 `(year, month, day)` 튜플의 리스트로 반환하세요.
4. 금지된 단어를 같은 길이의 별표로 대체하는 (대소문자 무시) `censor(text, banned_words)` 함수를 작성하세요.
5. `**볼드**`, `*이탤릭*`, `` `코드` ``, `# 제목`을 처리하는 간단한 마크다운-HTML 변환기를 구축하세요.

---

**이전**: [자료구조](./06_Data_Structures.md) | **다음**: [OOP 기초](./08_OOP_Basics.md)

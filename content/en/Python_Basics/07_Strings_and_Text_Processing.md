# Strings and Text Processing

**Previous**: [Data Structures](./06_Data_Structures.md) | **Next**: [OOP Basics](./08_OOP_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create strings using single quotes, double quotes, triple quotes, and raw strings
2. Access characters and substrings using indexing and slicing
3. Apply essential string methods for searching, transforming, and formatting text
4. Use f-strings, `.format()`, and `%` formatting for string interpolation
5. Write basic regular expressions using the `re` module for pattern matching and substitution
6. Understand string immutability and its implications for performance
7. Work with escape characters, multi-line strings, and Unicode/UTF-8 encoding
8. Process real-world text data by combining string methods and regular expressions

---

Text processing is at the heart of almost every program. Whether you are reading configuration files, parsing user input, generating reports, or building web applications, you work with strings constantly. Python treats strings as first-class objects with a rich set of built-in methods, and its `re` module provides powerful regular expression support for complex pattern matching.

## 1. Creating Strings

### Quotes

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

### Triple Quotes

Triple quotes (`"""` or `'''`) create multi-line strings.

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

### Raw Strings

Raw strings treat backslashes as literal characters. They are especially useful for regular expressions and file paths.

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

### String Concatenation and Repetition

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

## 2. Indexing and Slicing

Strings are sequences of characters. Like lists, they support indexing and slicing.

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

### Iterating Over Characters

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

## 3. String Methods

Python strings have over 40 built-in methods. Here are the most commonly used ones.

### Case Conversion

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

### Whitespace Handling

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

### Searching

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

### Splitting and Joining

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

### Replacing

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

### Testing String Content

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

## 4. String Formatting

Python offers three main approaches to string formatting.

### f-strings (Formatted String Literals) -- Recommended

Available since Python 3.6, f-strings are the most readable and performant option.

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

### `.format()` Method

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

### `%` Formatting (Old Style)

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

### Comparison of Formatting Methods

| Feature | f-string | .format() | % |
|---------|----------|-----------|---|
| Python version | 3.6+ | 2.6+ | All |
| Readability | Best | Good | Fair |
| Performance | Fastest | Medium | Medium |
| Expressions | Yes | No | No |
| Recommended | Yes | For compatibility | Legacy only |

---

## 5. Escape Characters

Escape characters represent special characters that cannot be typed directly.

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

| Escape | Meaning |
|--------|---------|
| `\n` | Newline |
| `\t` | Tab |
| `\\` | Backslash |
| `\'` | Single quote |
| `\"` | Double quote |
| `\r` | Carriage return |
| `\0` | Null character |
| `\uXXXX` | Unicode (16-bit) |
| `\UXXXXXXXX` | Unicode (32-bit) |
| `\xHH` | Hex value |

---

## 6. String Immutability

Strings in Python are immutable -- once created, they cannot be changed.

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

### Performance Implications

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

### String Interning

Python caches small strings for performance. This is an implementation detail you should not rely on.

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

## 7. Regular Expressions

The `re` module provides regular expression matching. Regular expressions (regex) are patterns that describe sets of strings.

### Basic Pattern Matching

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

### Common Regex Patterns

| Pattern | Meaning | Example Match |
|---------|---------|---------------|
| `\d` | Any digit | `5` |
| `\D` | Any non-digit | `a` |
| `\w` | Word character (letter, digit, _) | `A`, `3`, `_` |
| `\W` | Non-word character | `!`, ` ` |
| `\s` | Whitespace | ` `, `\t`, `\n` |
| `\S` | Non-whitespace | `a`, `5` |
| `.` | Any character (except newline) | anything |
| `^` | Start of string | |
| `$` | End of string | |
| `*` | 0 or more repetitions | |
| `+` | 1 or more repetitions | |
| `?` | 0 or 1 repetitions | |
| `{n}` | Exactly n repetitions | |
| `{n,m}` | Between n and m repetitions | |
| `[abc]` | Character class (a, b, or c) | |
| `[^abc]` | Negated class (not a, b, or c) | |
| `(...)` | Capturing group | |
| `\|` | Alternation (or) | |

### `findall` -- Find All Matches

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

### `sub` -- Substitution

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

### Compiled Patterns

For patterns used repeatedly, compile them for better performance.

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

### Groups and Named Groups

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

### Practical Regex Examples

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

### Regex Flags

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

## 8. Multi-line Strings

### Triple Quotes

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

### Line Continuation

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

## 9. String Encoding (UTF-8, encode/decode)

Strings in Python 3 are Unicode by default. When working with files, networks, or bytes, you need to convert between strings and bytes.

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

### ord() and chr()

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

## 10. Practical Text Processing Examples

### Example: CSV Line Parser

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

### Example: Text Statistics

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

### Example: Template Engine

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

## 11. Summary

| Topic | Key Points |
|-------|------------|
| Creation | Single, double, triple quotes; raw strings with `r""` |
| Indexing/Slicing | Zero-based; negative indexing; `[start:stop:step]` |
| Methods | `upper`, `lower`, `strip`, `split`, `join`, `replace`, `find`, `count` |
| f-strings | `f"{expr}"` -- most readable and fastest formatting |
| Escape chars | `\n`, `\t`, `\\`, `\"`, Unicode escapes |
| Immutability | Cannot modify in place; use `join` for building strings |
| Regex | `re.search`, `re.findall`, `re.sub`, `re.compile` |
| Encoding | `str.encode()` to bytes, `bytes.decode()` to string; default UTF-8 |

---

## Exercises

1. Write a function `count_vowels(text)` that returns the number of vowels (a, e, i, o, u) in a string, case-insensitive.
2. Write a function `title_case(text)` that capitalizes the first letter of each word, except for small words like "the", "a", "an", "in", "on", "at", "of" (unless it is the first word).
3. Use regex to extract all dates in `YYYY-MM-DD` format from a text and return them as a list of tuples `(year, month, day)`.
4. Write a `censor(text, banned_words)` function that replaces banned words with asterisks of the same length (case-insensitive).
5. Build a simple Markdown-to-HTML converter that handles `**bold**`, `*italic*`, `` `code` ``, and `# headings`.

---

**Previous**: [Data Structures](./06_Data_Structures.md) | **Next**: [OOP Basics](./08_OOP_Basics.md)

# What Are Regular Expressions

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what regular expressions are and their historical origins
2. Identify common use cases for regex in software development
3. Understand the relationship between regex and finite automata
4. Use Python's `re` module to perform basic pattern matching
5. Distinguish between `re.match()`, `re.search()`, and `re.findall()`
6. Work with match objects to extract matched text and positions
7. Use raw strings (`r""`) to avoid backslash escaping issues

---

## 1. What Is a Regular Expression?

A **regular expression** (regex or regexp) is a sequence of characters that defines a search pattern. This pattern can be used to match, find, or replace text within strings.

Think of regex as a mini-language specifically designed for describing text patterns:

```
Pattern:  \d{3}-\d{4}
          ───┬── ─┬─ ───┬── ─┬─
             |    |     |    |
             |    |     |    └── Exactly 4 digits
             |    |     └────── A literal hyphen
             |    └──────────── Exactly 3 digits
             └───────────────── \d means "any digit"

Matches:  "555-1234"  "800-5678"  "123-4567"
No match: "55-1234"   "5555-123"  "abc-defg"
```

---

## 2. A Brief History

Regular expressions have deep roots in computer science:

| Year | Event |
|------|-------|
| 1951 | Stephen Kleene describes "regular sets" using mathematical notation |
| 1968 | Ken Thompson implements regex in the QED text editor |
| 1973 | `grep` (Global Regular Expression Print) is born in Unix |
| 1986 | POSIX standardizes regex syntax (BRE and ERE) |
| 1987 | Larry Wall creates Perl, which popularizes advanced regex features |
| 1997 | Philip Hazel creates PCRE (Perl Compatible Regular Expressions) |
| 2003 | Python's `re` module stabilizes with the feature set we use today |

The theoretical foundation comes from **formal language theory**: regular expressions describe exactly the class of languages that can be recognized by **finite automata** (state machines).

```
Finite Automaton for the pattern "ab+c":

    ┌───┐  a   ┌───┐  b   ┌───┐  b   ┌───┐  c   ╔═══╗
    │ S │ ───> │ 1 │ ───> │ 2 │ ───> │ 2 │ ───> ║ ✓ ║
    └───┘      └───┘      └───┘      └───┘      ╚═══╝
                                       ↑    │
                                       └────┘
                                      (loop on 'b')

    S = Start state
    ✓ = Accept state (match found)
```

---

## 3. Why Learn Regular Expressions?

Regular expressions appear everywhere in software development:

### Validation
```
Is "user@example.com" a valid email?
Does "2024-01-15" match a date format?
Is "P@ssw0rd!" a strong password?
```

### Search and Replace
```
Find all phone numbers in a document
Replace "color" with "colour" (but not "Colorado")
Extract all URLs from HTML
```

### Data Extraction
```
Parse log files for error messages
Extract fields from CSV with mixed delimiters
Pull version numbers from release notes
```

### Text Processing
```
Tokenize source code
Split text into sentences
Clean user input (strip HTML tags, normalize whitespace)
```

---

## 4. Python's `re` Module

Python provides the `re` module in the standard library for regex operations. No installation required.

```python
import re
```

### 4.1 Your First Pattern Match

```python
import re

text = "The year is 2024 and the month is 12."

# Search for a 4-digit number
match = re.search(r'\d{4}', text)

if match:
    print(f"Found: {match.group()}")     # Found: 2024
    print(f"Position: {match.start()}")  # Position: 12
```

### 4.2 Raw Strings: Why `r""` Matters

In regular expressions, backslashes have special meaning. Python strings also use backslashes for escape sequences. This creates a conflict:

```
Without raw string:
    "\n"   -> Python interprets as newline character
    "\\n"  -> Python interprets as literal \n (what regex sees)

With raw string:
    r"\n"  -> Python keeps it as \n (what regex sees)
```

**Always use raw strings (`r""`) for regex patterns:**

```python
# BAD: Without raw string
pattern = "\d+"          # \d might be misinterpreted
pattern = "\\d+"         # Works but ugly and error-prone

# GOOD: With raw string
pattern = r"\d+"         # Clear and correct
```

Visual comparison:

```
Python String    What Regex Engine Sees
─────────────    ──────────────────────
"\d+"            \d+      (works by luck -- \d has no Python escape)
"\b"             ←BELL→   (Python backspace, NOT word boundary!)
r"\b"            \b       (correct: word boundary)
"\\"             \        (one backslash)
r"\\"            \\       (two backslashes -- probably not what you want)
```

---

## 5. Core Functions in `re`

### 5.1 `re.search()` -- Find First Match

Scans the entire string and returns the first match, or `None`:

```python
import re

text = "Error 404: Page not found at 15:30:00"

match = re.search(r'\d+', text)
if match:
    print(match.group())  # "404"
```

### 5.2 `re.match()` -- Match at the Start

Only matches at the **beginning** of the string:

```python
import re

# match() only checks the start of the string
print(re.match(r'\d+', "404 error"))     # <Match '404'>
print(re.match(r'\d+', "Error 404"))     # None (doesn't start with digits)
print(re.search(r'\d+', "Error 404"))    # <Match '404'> (search finds it)
```

```
re.match() vs re.search():

    String: "Error 404: Not Found"
             ^
             |
    match()  checks HERE only
    search() scans ──────────────>
```

### 5.3 `re.findall()` -- Find All Matches

Returns a list of all non-overlapping matches:

```python
import re

text = "Prices: $10.50, $23.99, $5.00"

prices = re.findall(r'\$\d+\.\d{2}', text)
print(prices)  # ['$10.50', '$23.99', '$5.00']
```

### 5.4 `re.finditer()` -- Iterate Over Matches

Returns an iterator of match objects (more detailed than `findall`):

```python
import re

text = "2024-01-15 Error: Connection failed\n2024-01-16 Info: Retry succeeded"

for match in re.finditer(r'\d{4}-\d{2}-\d{2}', text):
    print(f"Date: {match.group()} at position {match.start()}")

# Date: 2024-01-15 at position 0
# Date: 2024-01-16 at position 36
```

### 5.5 `re.fullmatch()` -- Match the Entire String

The entire string must match the pattern (useful for validation):

```python
import re

# Validate a date format
print(re.fullmatch(r'\d{4}-\d{2}-\d{2}', "2024-01-15"))  # Match
print(re.fullmatch(r'\d{4}-\d{2}-\d{2}', "Date: 2024-01-15"))  # None
```

---

## 6. The Match Object

When a regex matches, Python returns a `Match` object with useful methods:

```python
import re

text = "My phone number is 555-867-5309."
match = re.search(r'(\d{3})-(\d{3})-(\d{4})', text)

if match:
    print(match.group())     # '555-867-5309'  (entire match)
    print(match.group(0))    # '555-867-5309'  (same as group())
    print(match.group(1))    # '555'           (first capture group)
    print(match.group(2))    # '867'           (second capture group)
    print(match.group(3))    # '5309'          (third capture group)
    print(match.groups())    # ('555', '867', '5309')
    print(match.start())     # 19              (start position)
    print(match.end())       # 31              (end position)
    print(match.span())      # (19, 31)        (start, end tuple)
```

```
Match Object Anatomy:
                                    
    text = "My phone number is 555-867-5309."
                               ↑           ↑
                          start=19      end=31

    .group()  -> "555-867-5309"    Full match
    .group(1) -> "555"             ──┐
    .group(2) -> "867"               ├── Capture groups
    .group(3) -> "5309"            ──┘
    .span()   -> (19, 31)         Position in original text
```

---

## 7. Compiling Patterns

If you use a pattern multiple times, compile it for better performance:

```python
import re

# Compile once, use many times
phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

texts = [
    "Call 555-867-5309 for info",
    "Fax: 555-123-4567",
    "No phone here",
]

for text in texts:
    match = phone_pattern.search(text)
    if match:
        print(f"Found: {match.group()}")
```

Benefits of `re.compile()`:
- **Performance**: Pattern is compiled once, not on every call
- **Readability**: Give the pattern a descriptive variable name
- **Reuse**: Use the same compiled pattern in multiple places

---

## 8. A Simple Real-World Example

Let's combine what we've learned to extract data from a log line:

```python
import re

log_line = "[2024-01-15 08:30:45] ERROR server.py:142 - Connection timeout after 30s"

# Pattern breakdown:
#   \[(\d{4}-\d{2}-\d{2})\s+  - Date in brackets
#   (\d{2}:\d{2}:\d{2})\]\s+  - Time in brackets
#   (\w+)\s+                   - Log level
#   (\S+):(\d+)\s+-\s+        - File:line
#   (.+)                       - Message

pattern = re.compile(
    r'\[(\d{4}-\d{2}-\d{2})\s+'   # Date
    r'(\d{2}:\d{2}:\d{2})\]\s+'   # Time
    r'(\w+)\s+'                     # Level
    r'(\S+):(\d+)\s+-\s+'          # File:Line
    r'(.+)'                         # Message
)

match = pattern.search(log_line)
if match:
    date, time, level, file, line, message = match.groups()
    print(f"Date:    {date}")     # 2024-01-15
    print(f"Time:    {time}")     # 08:30:45
    print(f"Level:   {level}")    # ERROR
    print(f"File:    {file}")     # server.py
    print(f"Line:    {line}")     # 142
    print(f"Message: {message}")  # Connection timeout after 30s
```

---

## 9. Regex vs String Methods

Not everything needs regex. Python's string methods are faster for simple operations:

| Task | String Method | Regex |
|------|--------------|-------|
| Check if starts with "http" | `s.startswith("http")` | `re.match(r'http', s)` |
| Replace exact word | `s.replace("old", "new")` | `re.sub(r'old', 'new', s)` |
| Split on single delimiter | `s.split(",")` | `re.split(r',', s)` |
| Check if contains substring | `"hello" in s` | `re.search(r'hello', s)` |

**Rule of thumb**: Use string methods for fixed text operations. Use regex when you need **patterns** -- variable text, optional parts, alternatives, or repetition.

---

## Summary

| Concept | Description |
|---------|-------------|
| Regular Expression | A pattern language for matching text |
| `re` module | Python's built-in regex library |
| Raw strings (`r""`) | Prevents Python from interpreting backslashes |
| `re.search()` | Find first match anywhere in string |
| `re.match()` | Match only at the start of string |
| `re.findall()` | Return list of all matches |
| `re.finditer()` | Return iterator of match objects |
| `re.fullmatch()` | Match the entire string |
| `re.compile()` | Pre-compile pattern for reuse |
| Match object | Contains matched text, groups, and positions |

---

## Next Lesson

In [02_Literal_Matching_and_Metacharacters](./02_Literal_Matching_and_Metacharacters.md), we'll dive into the building blocks of regex patterns: literal characters and metacharacters.

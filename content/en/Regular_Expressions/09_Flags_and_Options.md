# Flags and Options

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `re.IGNORECASE` (`re.I`) for case-insensitive matching
2. Apply `re.MULTILINE` (`re.M`) for line-by-line anchor behavior
3. Use `re.DOTALL` (`re.S`) to make `.` match newlines
4. Write readable patterns with `re.VERBOSE` (`re.X`)
5. Combine multiple flags using the bitwise OR operator
6. Use inline flags within patterns (`(?i)`, `(?m)`, `(?s)`, `(?x)`)
7. Understand `re.ASCII` vs default Unicode matching
8. Apply flags effectively in real-world patterns

---

## 1. Overview of Regex Flags

Python's `re` module provides flags that modify pattern behavior:

```
Flag              Short    Effect
────              ─────    ──────
re.IGNORECASE     re.I     Case-insensitive matching
re.MULTILINE      re.M     ^ and $ match line boundaries
re.DOTALL         re.S     . matches newline characters too
re.VERBOSE        re.X     Allow comments and whitespace in patterns
re.ASCII          re.A     \w, \d, \s match ASCII only
re.LOCALE         re.L     Locale-dependent matching (rarely used)
re.UNICODE        re.U     Unicode matching (default in Python 3)
```

---

## 2. `re.IGNORECASE` (`re.I`)

Makes the pattern match regardless of case:

```python
import re

text = "Python PYTHON python PyThOn"

# Without flag: case-sensitive (default)
print(re.findall(r'python', text))
# ['python']

# With IGNORECASE: matches all cases
print(re.findall(r'python', text, re.IGNORECASE))
# ['Python', 'PYTHON', 'python', 'PyThOn']

# Short form
print(re.findall(r'python', text, re.I))
# ['Python', 'PYTHON', 'python', 'PyThOn']
```

### Character Ranges with IGNORECASE

```python
import re

text = "ABC abc 123 XYZ xyz"

# [a-z] with IGNORECASE also matches uppercase
print(re.findall(r'[a-z]+', text, re.I))
# ['ABC', 'abc', 'XYZ', 'xyz']

# Without flag
print(re.findall(r'[a-z]+', text))
# ['abc', 'xyz']
```

### Practical Example

```python
import re

# Case-insensitive search in text
article = """
Python is a programming language.
PYTHON was created by Guido van Rossum.
Learning python is fun!
"""

# Find all mentions of "python"
mentions = re.findall(r'\bpython\b', article, re.I)
print(f"Found {len(mentions)} mentions of Python")
# Found 3 mentions of Python
```

---

## 3. `re.MULTILINE` (`re.M`)

Changes `^` and `$` to match at line boundaries, not just string boundaries:

```python
import re

text = """Line 1: Hello
Line 2: World
Line 3: Python"""

# Without MULTILINE: ^ matches only string start
print(re.findall(r'^Line \d', text))
# ['Line 1']

# With MULTILINE: ^ matches each line start
print(re.findall(r'^Line \d', text, re.MULTILINE))
# ['Line 1', 'Line 2', 'Line 3']

# $ also changes behavior
print(re.findall(r'\w+$', text, re.M))
# ['Hello', 'World', 'Python']
```

```
Without re.MULTILINE:
    "Line 1\nLine 2\nLine 3"
     ^                       $
     (only one ^ and one $)

With re.MULTILINE:
    "Line 1\nLine 2\nLine 3"
     ^      $^      $^      $
     (^ and $ at each line boundary)
```

### Practical Example

```python
import re

config = """# Database settings
host=localhost
port=5432
# Connection pool
max_connections=10
timeout=30"""

# Remove comment lines
clean = re.sub(r'^#.*$\n?', '', config, flags=re.M)
print(clean)
# host=localhost
# port=5432
# max_connections=10
# timeout=30
```

---

## 4. `re.DOTALL` (`re.S`)

Makes `.` match **any** character, including newlines:

```python
import re

text = """<div>
Hello
World
</div>"""

# Without DOTALL: . doesn't match \n
match = re.search(r'<div>(.+)</div>', text)
print(match)  # None (can't match across lines)

# With DOTALL: . matches \n too
match = re.search(r'<div>(.+)</div>', text, re.DOTALL)
print(match.group(1))
# "\nHello\nWorld\n"
```

```
Without re.DOTALL:
    . matches: a b c 1 2 3 ! @ # (any char except \n)
    
With re.DOTALL:
    . matches: a b c 1 2 3 ! @ # \n (truly any character)

Alternative without DOTALL: use [\s\S] instead of .
    [\s\S] always matches any character including newline
```

### Practical Example

```python
import re

html = """<script>
function hello() {
    alert("Hi!");
}
</script>
<p>Content here</p>"""

# Extract script content (spans multiple lines)
match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
if match:
    print(match.group(1).strip())
    # function hello() {
    #     alert("Hi!");
    # }
```

---

## 5. `re.VERBOSE` (`re.X`)

Allows you to write patterns with whitespace and comments for readability:

```python
import re

# Without VERBOSE: hard to read
pattern_compact = r'^(?:(?P<scheme>\w+)://)(?P<host>[^/:]+)(?::(?P<port>\d+))?(?P<path>/[^\s?]*)?(?:\?(?P<query>\S+))?$'

# With VERBOSE: readable and documented
pattern_verbose = re.compile(r"""
    ^                           # Start of string
    (?:(?P<scheme>\w+)://)      # Scheme (http, https, ftp)
    (?P<host>[^/:]+)            # Hostname
    (?::(?P<port>\d+))?         # Optional port number
    (?P<path>/[^\s?]*)?         # Optional path
    (?:\?(?P<query>\S+))?       # Optional query string
    $                           # End of string
""", re.VERBOSE)

url = "https://example.com:8080/api/users?page=1"
match = pattern_verbose.search(url)
if match:
    print(match.groupdict())
```

### Rules for VERBOSE Mode

```
In VERBOSE mode:
- Whitespace is IGNORED (except inside character classes or when escaped)
- # starts a comment until end of line
- To match a literal space, use \s, [ ], or \ (escaped space)
- To match a literal #, use \# or [#]
```

```python
import re

# Matching spaces in VERBOSE mode
pattern = re.compile(r"""
    Hello       # match "Hello"
    [ ]         # match a literal space (inside character class)
    World       # match "World"
""", re.VERBOSE)

print(pattern.search("Hello World"))  # Match!
```

### Practical Example: Email Validation

```python
import re

email_pattern = re.compile(r"""
    ^                       # Start
    [\w.+-]+                # Local part: word chars, dots, plus, hyphen
    @                       # @ symbol
    [a-zA-Z0-9-]+          # Domain name
    (?:\.[a-zA-Z0-9-]+)*   # Subdomains
    \.[a-zA-Z]{2,}         # TLD (at least 2 letters)
    $                       # End
""", re.VERBOSE)

test_emails = [
    "user@example.com",
    "first.last+tag@sub.domain.org",
    "invalid@",
    "@no-local.com",
    "user@.com",
]

for email in test_emails:
    result = "Valid" if email_pattern.match(email) else "Invalid"
    print(f"{email:35s} -> {result}")
```

---

## 6. Combining Flags

Use the bitwise OR operator `|` to combine multiple flags:

```python
import re

text = """Hello World
hello python
HELLO REGEX"""

# Combine IGNORECASE and MULTILINE
pattern = r'^hello\b'
matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
print(matches)  # ['Hello', 'hello', 'HELLO']

# Combine three flags
html = """<div>
Content with
multiple lines
</div>"""

match = re.search(
    r"""
    <div>       # opening tag
    \s*(.+?)    # content (lazy)
    \s*</div>   # closing tag
    """,
    html,
    re.VERBOSE | re.DOTALL | re.IGNORECASE
)
print(match.group(1))
# "Content with\nmultiple lines"
```

---

## 7. Inline Flags

You can embed flags directly in the pattern using `(?flags)`:

```python
import re

# Inline IGNORECASE
print(re.findall(r'(?i)python', "Python PYTHON python"))
# ['Python', 'PYTHON', 'python']

# Inline MULTILINE
text = "Line 1\nLine 2\nLine 3"
print(re.findall(r'(?m)^Line \d', text))
# ['Line 1', 'Line 2', 'Line 3']

# Inline DOTALL
text = "start\nmiddle\nend"
print(re.search(r'(?s)start(.+)end', text).group(1))
# "\nmiddle\n"
```

### Inline Flag Characters

```
Inline    Equivalent Flag
──────    ───────────────
(?i)      re.IGNORECASE
(?m)      re.MULTILINE
(?s)      re.DOTALL
(?x)      re.VERBOSE
(?a)      re.ASCII
(?imsx)   Combine multiple flags
```

### Scoped Inline Flags

```python
import re

# Apply flag to part of the pattern only
# (?i:...) makes only that group case-insensitive
pattern = r'Hello (?i:world)'  # "world" is case-insensitive
print(re.search(pattern, "Hello WORLD"))   # Match
print(re.search(pattern, "HELLO WORLD"))   # None (Hello is still case-sensitive)
```

---

## 8. `re.ASCII` (`re.A`)

Controls whether `\w`, `\d`, `\s`, `\b` match Unicode or ASCII only:

```python
import re

# Python 3 default: Unicode matching
text = "hello cafe 42"

# Default: \w matches Unicode word characters
print(re.findall(r'\w+', text))
# ['hello', 'cafe', '42']  -- includes accented characters in Unicode context

# With ASCII: restrict to [a-zA-Z0-9_]
print(re.findall(r'\w+', text, re.ASCII))
# ['hello', 'cafe', '42']  -- ASCII letters only
```

```
Unicode vs ASCII matching:

    Character    \w (Unicode, default)    \w (ASCII, re.A)
    ─────────    ────────────────────     ────────────────
    a-z, A-Z     ✓                        ✓
    0-9          ✓                        ✓
    _            ✓                        ✓
    Accented     ✓ (depends)              ✗
```

---

## 9. Practical Flag Combinations

### Log Processing (MULTILINE + VERBOSE)

```python
import re

log = """2024-01-15 08:30:45 ERROR Connection failed
2024-01-15 08:30:46 INFO Retrying
2024-01-15 08:30:47 ERROR Timeout expired"""

pattern = re.compile(r"""
    ^                       # Start of line
    (\d{4}-\d{2}-\d{2})    # Date
    \s+
    (\d{2}:\d{2}:\d{2})    # Time
    \s+
    (ERROR|WARN)            # Level (only errors and warnings)
    \s+
    (.+)                    # Message
    $                       # End of line
""", re.VERBOSE | re.MULTILINE)

for match in pattern.finditer(log):
    date, time, level, msg = match.groups()
    print(f"[{level}] {date} {time}: {msg}")
```

### Multi-line HTML Parsing (DOTALL + IGNORECASE)

```python
import re

html = """
<DIV class="content">
  <P>First paragraph</P>
  <P>Second paragraph</P>
</DIV>
"""

# Extract content between div tags (case-insensitive, multi-line)
match = re.search(
    r'<div[^>]*>(.*?)</div>',
    html,
    re.DOTALL | re.IGNORECASE
)
if match:
    # Extract all paragraph contents
    paragraphs = re.findall(r'<p>(.*?)</p>', match.group(1), re.I)
    print(paragraphs)  # ['First paragraph', 'Second paragraph']
```

### Configuration Parser (VERBOSE + MULTILINE)

```python
import re

config = """
# Server configuration
server.host = localhost
server.port = 8080

# Database configuration
db.host = 192.168.1.100
db.port = 5432
db.name = myapp
"""

pattern = re.compile(r"""
    ^                   # Start of line
    (?!\s*\#)           # Not a comment line
    (\w+(?:\.\w+)*)     # Key (dotted notation)
    \s*=\s*             # Equals with optional whitespace
    (.+?)               # Value
    \s*$                # End of line (trim trailing whitespace)
""", re.VERBOSE | re.MULTILINE)

config_dict = dict(pattern.findall(config))
for key, value in config_dict.items():
    print(f"{key} = {value}")
```

---

## 10. Flag Decision Guide

```
Need case-insensitive matching?
└── Yes -> re.IGNORECASE (re.I)

Working with multi-line text and need ^ $ per line?
└── Yes -> re.MULTILINE (re.M)

Need . to match across line breaks?
└── Yes -> re.DOTALL (re.S)

Pattern is complex and needs documentation?
└── Yes -> re.VERBOSE (re.X)

Processing only ASCII text (no Unicode)?
└── Yes -> re.ASCII (re.A)
```

---

## Summary

| Flag | Short | Inline | Effect |
|------|-------|--------|--------|
| `re.IGNORECASE` | `re.I` | `(?i)` | Case-insensitive matching |
| `re.MULTILINE` | `re.M` | `(?m)` | `^`/`$` match line boundaries |
| `re.DOTALL` | `re.S` | `(?s)` | `.` matches newline too |
| `re.VERBOSE` | `re.X` | `(?x)` | Allow comments and whitespace |
| `re.ASCII` | `re.A` | `(?a)` | ASCII-only for `\w`, `\d`, `\s` |

Key points:
- Combine flags with `|`: `re.I | re.M | re.S`
- Use `re.VERBOSE` for any pattern longer than ~30 characters
- Inline flags `(?i)` affect the entire pattern (or use scoped `(?i:...)`)
- `re.DOTALL` and `re.MULTILINE` serve different purposes and are often used together

---

## Next Lesson

In [10_Common_Patterns](./10_Common_Patterns.md), we'll build and analyze patterns for common validation and extraction tasks.

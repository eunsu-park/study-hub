# Anchors and Boundaries

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `^` and `$` anchors in single-line and multiline modes
2. Apply word boundary `\b` to match whole words
3. Use non-word boundary `\B` for partial word matches
4. Understand the difference between `\A`, `\Z`, and `^`, `$`
5. Combine anchors with quantifiers and character classes
6. Apply multiline mode (`re.MULTILINE`) for line-by-line matching
7. Solve practical problems using boundary assertions

---

## 1. What Are Anchors?

Anchors match **positions** in a string, not characters. They have zero width -- they don't consume any text:

```
String:  H e l l o ,   W o r l d !
         ↑                         ↑
         ^                         $
     start of string           end of string

         ↑   ↑ ↑   ↑ ↑ ↑   ↑   ↑ ↑ ↑ ↑ ↑
         \b  \b\b   \b\b\b  \b  \b\b \b\b\b
         word boundaries (between \w and \W)
```

---

## 2. Start and End Anchors: `^` and `$`

### `^` -- Start of String

```python
import re

lines = [
    "Python is great",
    "I love Python",
    "Python rocks",
]

for line in lines:
    if re.search(r'^Python', line):
        print(f"Starts with Python: {line}")

# Starts with Python: Python is great
# Starts with Python: Python rocks
```

### `$` -- End of String

```python
import re

files = ["report.pdf", "data.csv", "image.pdf", "notes.txt"]

for f in files:
    if re.search(r'\.pdf$', f):
        print(f"PDF file: {f}")

# PDF file: report.pdf
# PDF file: image.pdf
```

### Combining `^` and `$` for Full String Validation

```python
import re

def validate_username(username):
    """Username: 3-16 alphanumeric chars, underscores allowed."""
    return bool(re.fullmatch(r'^[a-zA-Z]\w{2,15}$', username))

test_cases = [
    ("alice", True),
    ("Bob_99", True),
    ("ab", False),        # too short
    ("1alice", False),    # starts with digit
    ("a" * 17, False),   # too long
]

for username, expected in test_cases:
    result = validate_username(username)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] '{username}' -> {result}")
```

---

## 3. Multiline Mode: `re.MULTILINE`

Without `re.MULTILINE`, `^` and `$` match only the start and end of the **entire string**. With `re.MULTILINE`, they match the start and end of **each line**:

```python
import re

text = """Line 1: Hello
Line 2: World
Line 3: Python"""

# Without MULTILINE: ^ matches only the start of the string
print(re.findall(r'^Line \d', text))
# ['Line 1']

# With MULTILINE: ^ matches the start of each line
print(re.findall(r'^Line \d', text, re.MULTILINE))
# ['Line 1', 'Line 2', 'Line 3']
```

```
Without re.MULTILINE:

    "Line 1: Hello\nLine 2: World\nLine 3: Python"
     ^                                             $
     only here                              only here

With re.MULTILINE:

    "Line 1: Hello\nLine 2: World\nLine 3: Python"
     ^             ^              ^                $
     ^             $^             $^               $
     Each \n creates a new line boundary
```

### Extracting Lines That Match a Pattern

```python
import re

log = """2024-01-15 INFO: Server started
2024-01-15 ERROR: Connection failed
2024-01-16 INFO: Request received
2024-01-16 ERROR: Timeout expired
2024-01-16 WARN: High memory usage"""

# Find all ERROR lines
errors = re.findall(r'^.*ERROR.*$', log, re.MULTILINE)
for e in errors:
    print(e)

# 2024-01-15 ERROR: Connection failed
# 2024-01-16 ERROR: Timeout expired
```

---

## 4. `\A` and `\Z` -- Absolute Anchors

`\A` and `\Z` always match the start/end of the **entire string**, even in multiline mode:

```python
import re

text = """First line
Second line
Third line"""

# ^ with MULTILINE matches each line start
print(re.findall(r'^\w+', text, re.MULTILINE))
# ['First', 'Second', 'Third']

# \A always matches only the absolute start
print(re.findall(r'\A\w+', text, re.MULTILINE))
# ['First']

# \Z always matches only the absolute end
print(re.findall(r'\w+\Z', text, re.MULTILINE))
# ['line']
```

```
Anchor Comparison:

    Mode          ^/$ match              \A/\Z match
    ────          ─────────              ───────────
    Default       String start/end       String start/end
    MULTILINE     Line start/end         String start/end (unchanged)
```

---

## 5. Word Boundaries: `\b`

The `\b` anchor matches the **boundary between a word character and a non-word character**:

```
Word characters (\w):     a-z A-Z 0-9 _
Non-word characters (\W): spaces, punctuation, start/end of string

A word boundary exists:
    - Between \w and \W
    - Between \w and start/end of string
    - Between \W and \w
```

```python
import re

text = "cat scatter category caterpillar"

# Without \b: matches "cat" anywhere, including inside words
print(re.findall(r'cat', text))
# ['cat', 'cat', 'cat', 'cat']

# With \b: matches "cat" only as a whole word
print(re.findall(r'\bcat\b', text))
# ['cat']
```

```
Word Boundary Visualization:

    c a t   s c a t t e r   c a t e g o r y
    ↑     ↑ ↑             ↑ ↑
    \b    \b \b            \b \b

    \bcat\b  matches only here:
    [cat]  scatter  category  caterpillar
     ───
     ✓       ✗        ✗         ✗
```

### Common `\b` Patterns

```python
import re

# Match whole words only
text = "I love JavaScript, not just Java"
print(re.findall(r'\bJava\b', text))
# ['Java']  (not "JavaScript")

# Match words starting with a prefix
text = "preview, preprocess, present, compress"
print(re.findall(r'\bpre\w+', text))
# ['preview', 'preprocess', 'present']

# Match words ending with a suffix
text = "running, jumping, sing, nothing"
print(re.findall(r'\w+ing\b', text))
# ['running', 'jumping', 'sing', 'nothing']
```

---

## 6. Non-Word Boundary: `\B`

`\B` matches where `\b` does NOT match -- positions inside or outside words:

```python
import re

text = "cat scatter category caterpillar"

# \B: NOT a word boundary -- match "cat" inside other words
print(re.findall(r'\Bcat\B', text))
# ['cat']  (only "scatter" has "cat" fully inside)

print(re.findall(r'\Bcat', text))
# ['cat', 'cat']  (scatter, category)
```

```
\B Visualization:

    s c a t t e r
      ↑     ↑
      \B    \B   (NOT word boundaries -- inside the word)

    \Bcat\B matches "cat" in "scatter":
    s[cat]ter  -- both sides of "cat" are word characters
```

### Practical Use: Match Substrings

```python
import re

# Find "ever" only when it's part of a larger word
text = "ever whatever however forever"
print(re.findall(r'\Bever\b', text))
# ['ever', 'ever', 'ever']  (whatever, however, forever)
# NOT the standalone "ever"
```

---

## 7. Combining Anchors with Other Features

### Validate Formats

```python
import re

# Email-like validation (simplified)
emails = ["user@example.com", "@invalid", "user@", "a@b.c"]
for email in emails:
    if re.fullmatch(r'^[\w.+-]+@[\w-]+\.[\w.]+$', email):
        print(f"Valid:   {email}")
    else:
        print(f"Invalid: {email}")
```

### Extract First/Last Words

```python
import re

text = "The quick brown fox jumps over the lazy dog"

# First word
print(re.search(r'^\w+', text).group())  # "The"

# Last word
print(re.search(r'\w+$', text).group())  # "dog"

# First word of each line
multiline = """Hello World
Goodbye Moon
Greetings Star"""
print(re.findall(r'^\w+', multiline, re.MULTILINE))
# ['Hello', 'Goodbye', 'Greetings']
```

### Match Blank Lines

```python
import re

text = """Line 1

Line 3

Line 5"""

# Find empty lines
blank_lines = re.findall(r'^$', text, re.MULTILINE)
print(f"Number of blank lines: {len(blank_lines)}")  # 2
```

---

## 8. Anchors in Substitution

```python
import re

# Add line numbers to each line
text = """apple
banana
cherry"""

def add_line_numbers(text):
    lines = text.split('\n')
    return '\n'.join(f"{i+1}. {line}" for i, line in enumerate(lines))

# Or using regex with a counter
counter = [0]
def number_line(match):
    counter[0] += 1
    return f"{counter[0]}. {match.group()}"

result = re.sub(r'^.+', number_line, text, flags=re.MULTILINE)
print(result)
# 1. apple
# 2. banana
# 3. cherry
```

### Strip Trailing Whitespace

```python
import re

code = "def hello():   \n    pass  \n    return True   \n"

# Remove trailing whitespace from each line
clean = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
print(repr(clean))
# 'def hello():\n    pass\n    return True\n'
```

---

## 9. Practical Examples

### Example 1: Validate IP Address Format

```python
import re

def is_valid_ip_format(ip):
    """Check if string looks like an IPv4 address (basic format check)."""
    return bool(re.fullmatch(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ip))

tests = ["192.168.1.1", "10.0.0.1", "999.999.999.999", "1.2.3", "a.b.c.d"]
for ip in tests:
    print(f"{ip:20s} -> {is_valid_ip_format(ip)}")
```

### Example 2: Find Sentences

```python
import re

text = "Hello world. How are you? I'm fine! Thanks."

# Find sentences (ending with . ? or !)
sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
for s in sentences:
    print(s)
# Hello world.
# How are you?
# I'm fine!
# Thanks.
```

### Example 3: Match Variable Names

```python
import re

code = "count = 0; _private = True; 2bad = False; my_var = 42"

# Valid Python identifiers: start with letter or underscore
identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
print(identifiers)
# ['count', '_private', 'bad', 'False', 'my_var']
```

---

## Summary

| Anchor | Meaning | Multiline Behavior |
|--------|---------|-------------------|
| `^` | Start of string | Start of each line (with `re.MULTILINE`) |
| `$` | End of string | End of each line (with `re.MULTILINE`) |
| `\A` | Absolute start of string | Always start of string |
| `\Z` | Absolute end of string | Always end of string |
| `\b` | Word boundary | N/A (not affected by multiline) |
| `\B` | Non-word boundary | N/A (not affected by multiline) |

Key points:
- Anchors match **positions**, not characters
- `\b` is essential for matching **whole words**
- Use `re.MULTILINE` when processing text line by line
- `\A` and `\Z` are immune to multiline mode

---

## Next Lesson

In [06_Groups_and_Capturing](./06_Groups_and_Capturing.md), we'll learn how to group parts of patterns and extract captured text.

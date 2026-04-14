# Lookahead and Lookbehind

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand zero-width assertions and how they differ from regular matching
2. Use positive lookahead `(?=...)` to assert what follows
3. Use negative lookahead `(?!...)` to assert what does NOT follow
4. Use positive lookbehind `(?<=...)` to assert what precedes
5. Use negative lookbehind `(?<!...)` to assert what does NOT precede
6. Combine multiple lookaround assertions in a single pattern
7. Understand lookbehind length limitations in Python
8. Apply lookarounds to solve practical text processing problems

---

## 1. What Are Lookaround Assertions?

Lookaround assertions check whether a pattern exists before or after the current position, **without consuming any characters**. They are zero-width, like anchors:

```
Type                 Syntax         Meaning
────                 ──────         ───────
Positive Lookahead   (?=...)        What follows MUST match
Negative Lookahead   (?!...)        What follows must NOT match
Positive Lookbehind  (?<=...)       What precedes MUST match
Negative Lookbehind  (?<!...)       What precedes must NOT match
```

```
Zero-Width Concept:

    Regular match:     f o o b a r
                       ─────────── consumed (moves the cursor)

    Lookahead:         f o o | b a r
                              ↑
                       cursor stays here, just "peeks ahead"
```

---

## 2. Positive Lookahead: `(?=...)`

Matches a position where the pattern inside `(?=...)` **can** be matched ahead:

```python
import re

# Match "foo" only if followed by "bar"
text = "foobar foobaz foo"
print(re.findall(r'foo(?=bar)', text))
# ['foo']  -- only the foo before bar

# Note: "bar" is NOT included in the match!
match = re.search(r'foo(?=bar)', text)
print(match.group())  # "foo" (not "foobar")
print(match.span())   # (0, 3)
```

```
Positive Lookahead Visualization:

    Pattern: \w+(?=\.)

    "Hello World. How are you?"
     ─────────
     match    ↑
              └── lookahead checks for "." but doesn't consume it

    Result: "World" (the word before the period)
```

### Practical Examples

```python
import re

# Find words followed by a comma
text = "apple, banana, cherry and grape"
print(re.findall(r'\w+(?=,)', text))
# ['apple', 'banana']

# Find numbers followed by a unit
text = "100px, 50em, 200px, 75%"
print(re.findall(r'\d+(?=px)', text))
# ['100', '200']  -- only numbers with px
```

---

## 3. Negative Lookahead: `(?!...)`

Matches a position where the pattern inside `(?!...)` **cannot** be matched ahead:

```python
import re

# Match "foo" only if NOT followed by "bar"
text = "foobar foobaz foo"
print(re.findall(r'foo(?!bar)', text))
# ['foo', 'foo']  -- foobaz and standalone foo
```

```
Negative Lookahead Visualization:

    Pattern: \d+(?!px)

    "100px 50em 200px 75%"
          ─── 
          50    <- not followed by "px"
               ─── 
               75   <- not followed by "px"

    Wait -- this is tricky! Let's see what actually happens:
    "100" -- the engine tries "100" not followed by "px" -> "100" IS followed by "px" -> no
          -- but "10" not followed by "px" -> "10" is followed by "0" -> yes!
    
    Solution: be more specific with \d+(?!\d|px)
```

### Common Negative Lookahead Patterns

```python
import re

# Match "java" but not "javascript"
text = "java javascript javafx java"
print(re.findall(r'\bjava\b(?!\s*script)', text))
# Note: \b already prevents matching inside "javascript"
print(re.findall(r'\bjava(?!script)\b', text))
# ['java', 'java']

# Match lines NOT starting with a comment
code = """# This is a comment
print("hello")
# Another comment
x = 42"""
non_comments = re.findall(r'^(?!#).*$', code, re.MULTILINE)
print(non_comments)
# ['print("hello")', 'x = 42']
```

---

## 4. Positive Lookbehind: `(?<=...)`

Matches a position where the pattern inside `(?<=...)` **can** be matched behind:

```python
import re

# Match a number only if preceded by "$"
text = "Price: $50, Quantity: 10, Total: $500"
print(re.findall(r'(?<=\$)\d+', text))
# ['50', '500']  -- only numbers after $
```

```
Positive Lookbehind Visualization:

    Pattern: (?<=\$)\d+

    "Price: $50, Quantity: 10"
            ↑──
            └── lookbehind checks for "$" before the digits
            
    "$50" -> (?<=\$) checks: is there a $ before? YES -> match "50"
    "10"  -> (?<=\$) checks: is there a $ before? NO  -> skip
```

### More Examples

```python
import re

# Extract values after "="
text = "name=John age=30 city=NYC"
print(re.findall(r'(?<=\=)\w+', text))
# ['John', '30', 'NYC']

# Extract text inside parentheses (using lookbehind + lookahead)
text = "Hello (World) and (Python)"
print(re.findall(r'(?<=\()\w+(?=\))', text))
# ['World', 'Python']
```

---

## 5. Negative Lookbehind: `(?<!...)`

Matches a position where the pattern inside `(?<!...)` **cannot** be matched behind:

```python
import re

# Match numbers NOT preceded by "$"
text = "Price: $50, Quantity: 10, Total: $500"
print(re.findall(r'(?<!\$)\b\d+', text))
# ['10']

# Match "port" not preceded by "air" or "trans"
text = "port airport transport port export"
print(re.findall(r'(?<!air)(?<!trans)\bport\b', text))
# ['port', 'port']  -- standalone ports only
```

```
Negative Lookbehind Visualization:

    Pattern: (?<!air)port

    "airport"   -> (?<!air) checks: is "air" before "port"? YES -> skip
    "transport" -> (?<!air) checks: is "air" before "port"? NO  -> check next
                   (but we also need (?<!trans))
    "port"      -> (?<!air) checks: is "air" before? NO -> match!
```

---

## 6. Lookbehind Length Limitations

In Python's `re` module, lookbehind patterns must have a **fixed length**:

```python
import re

# VALID: fixed-length lookbehind
re.findall(r'(?<=abc)\w+', "abcdef")      # OK: "abc" is 3 chars
re.findall(r'(?<=\d{3})\w+', "123abc")    # OK: \d{3} is 3 chars

# INVALID: variable-length lookbehind
try:
    re.findall(r'(?<=\d+)\w+', "123abc")  # ERROR!
except re.error as e:
    print(f"Error: {e}")
    # Error: look-behind requires fixed-width pattern
```

### Workarounds for Variable-Length Lookbehind

```python
import re

# Instead of variable lookbehind, capture what you need
text = "price: $1234"

# Can't do: (?<=\$\d*)\d+
# Solution 1: Capture group
match = re.search(r'\$(\d+)', text)
print(match.group(1))  # "1234"

# Solution 2: Use the `regex` third-party module (supports variable lookbehind)
# pip install regex
# import regex
# regex.findall(r'(?<=\$\d*)\d+', text)  # Works!
```

**Alternation in lookbehind** is allowed if all alternatives have the same length:

```python
import re

# OK: both alternatives are 3 characters
print(re.findall(r'(?<=abc|def)\w+', "abcXYZ defGHI"))
# ['XYZ', 'GHI']

# ERROR: alternatives have different lengths
try:
    re.findall(r'(?<=ab|defg)\w+', "abcXYZ defgGHI")
except re.error as e:
    print(f"Error: {e}")
```

---

## 7. Combining Multiple Lookarounds

You can chain multiple lookaround assertions:

```python
import re

# Password validation: at least one digit AND one uppercase
# (We'll check if the whole string passes multiple conditions)
def validate_password(pwd):
    checks = [
        (r'(?=.*[A-Z])', "uppercase letter"),
        (r'(?=.*[a-z])', "lowercase letter"),
        (r'(?=.*\d)', "digit"),
        (r'(?=.*[!@#$%])', "special character"),
        (r'.{8,}', "8+ characters"),
    ]
    for pattern, desc in checks:
        if not re.search(pattern, pwd):
            print(f"  Missing: {desc}")
            return False
    return True

passwords = ["P@ssw0rd", "password", "P@SS", "12345678"]
for pwd in passwords:
    result = validate_password(pwd)
    print(f"'{pwd}' -> {'Valid' if result else 'Invalid'}\n")
```

### All-in-One Password Pattern

```python
import re

# Single pattern with multiple lookaheads
pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%]).{8,}$'

print(bool(re.match(pattern, "P@ssw0rd")))   # True
print(bool(re.match(pattern, "password")))    # False
print(bool(re.match(pattern, "SHORT1!")))     # False
```

```
Multiple Lookahead at Same Position:

    Pattern: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%]).{8,}$

    Position ^:
    ├── (?=.*[A-Z])    peek ahead: is there an uppercase letter? ✓
    ├── (?=.*[a-z])    peek ahead: is there a lowercase letter? ✓
    ├── (?=.*\d)       peek ahead: is there a digit? ✓
    ├── (?=.*[!@#$%])  peek ahead: is there a special char? ✓
    └── .{8,}$         now actually match 8+ characters to end

    All lookaheads check from the SAME position (start of string).
    They don't consume characters, so they all start at ^.
```

---

## 8. Lookarounds in Substitution

Lookarounds are powerful in `re.sub()` because they don't consume text:

```python
import re

# Add commas to large numbers: 1234567 -> 1,234,567
def add_commas(n):
    return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', str(n))

print(add_commas(1234567))      # "1,234,567"
print(add_commas(1000000000))   # "1,000,000,000"
print(add_commas(42))           # "42"
```

```
Comma Insertion Pattern: (?<=\d)(?=(\d{3})+(?!\d))

    Number: 1234567

    Check each position:
    1 | 2 3 4 5 6 7    <- after 1: 234567 = 2 groups of 3? Yes -> insert comma
    1 , 2 | 3 4 5 6 7  <- after 2: 34567 = not multiple of 3? No
    1 , 2 3 | 4 5 6 7  <- after 3: 4567 = not multiple of 3? No
    1 , 2 3 4 | 5 6 7  <- after 4: 567 = 1 group of 3? Yes -> insert comma
    ...

    Result: 1,234,567
```

### More Substitution Examples

```python
import re

# Insert space before uppercase letters (camelCase -> camel Case)
text = "camelCaseToSeparateWords"
result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
print(result)  # "camel Case To Separate Words"

# Highlight search terms without changing them
text = "Python is a programming language. Python is great."
result = re.sub(r'(?<=\b)Python(?=\b)', '[PYTHON]', text)
print(result)  # "[PYTHON] is a programming language. [PYTHON] is great."
```

---

## 9. Practical Examples

### Example 1: Extract Prices Without Currency Symbol

```python
import re

text = "Items: $19.99, EUR29.50, $5.00, JPY1500"

# Get numbers after $ sign
usd_prices = re.findall(r'(?<=\$)\d+\.?\d*', text)
print(f"USD: {usd_prices}")  # ['19.99', '5.00']

# Get numbers after EUR
eur_prices = re.findall(r'(?<=EUR)\d+\.?\d*', text)
print(f"EUR: {eur_prices}")  # ['29.50']
```

### Example 2: Match Words Not Inside Quotes

```python
import re

text = 'The "quick brown" fox "jumped" over'

# Find words NOT preceded by a quote context
# Simple approach: split on quoted sections
parts = re.split(r'"[^"]*"', text)
words = []
for part in parts:
    words.extend(re.findall(r'\w+', part))
print(words)  # ['The', 'fox', 'over']
```

### Example 3: Validate File Extensions

```python
import re

files = ["report.pdf", "data.csv", "image.exe", "script.py", "notes.txt"]

# Match filenames that don't end with .exe
safe_files = [f for f in files if re.search(r'^.*(?<!\.exe)$', f)]
print(safe_files)  # ['report.pdf', 'data.csv', 'script.py', 'notes.txt']
```

### Example 4: Find Consecutive Capitalized Words (Proper Nouns)

```python
import re

text = "I visited New York City and saw the Statue of Liberty"

# Find sequences of capitalized words (2+ words)
# Use lookahead to check the pattern exists
proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', text)
print(proper_nouns)  # ['New York City', 'Statue of Liberty'] -- 
# Hmm, "of" is lowercase. Let's adjust:
proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s(?:of\s)?[A-Z][a-z]+)+', text)
print(proper_nouns)  # ['New York City']
```

---

## 10. Lookaround Cheat Sheet

```
Assertion          Direction    Positive/Negative    Example
─────────          ─────────    ────────────────     ───────
(?=pattern)        Forward →    Positive (must)      \w+(?=\.)
(?!pattern)        Forward →    Negative (must not)  \d+(?!px)
(?<=pattern)       Backward ←   Positive (must)      (?<=\$)\d+
(?<!pattern)       Backward ←   Negative (must not)  (?<!\d)\w+

Key Rules:
- Lookarounds are ZERO-WIDTH (don't consume characters)
- Lookbehinds must have FIXED LENGTH in Python's re
- Multiple lookarounds can be chained at the same position
- Lookarounds can contain any pattern (groups, quantifiers, etc.)
```

---

## Summary

| Assertion | Syntax | Meaning |
|-----------|--------|---------|
| Positive lookahead | `(?=...)` | What follows must match |
| Negative lookahead | `(?!...)` | What follows must not match |
| Positive lookbehind | `(?<=...)` | What precedes must match |
| Negative lookbehind | `(?<!...)` | What precedes must not match |

Key points:
- Lookarounds are zero-width -- they don't consume characters
- Python lookbehinds require fixed-width patterns
- Chain multiple lookaheads for complex validation (e.g., passwords)
- Lookarounds in `re.sub()` enable insertions without replacing text

---

## Next Lesson

In [08_Substitution_and_Splitting](./08_Substitution_and_Splitting.md), we'll master `re.sub()` and `re.split()` for transforming text.

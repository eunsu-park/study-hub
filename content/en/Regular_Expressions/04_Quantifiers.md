# Quantifiers

## Learning Objectives

After completing this lesson, you will be able to:

1. Use basic quantifiers (`*`, `+`, `?`) to control repetition
2. Specify exact repetition counts with `{n}`, `{n,}`, and `{n,m}`
3. Distinguish between greedy and lazy (non-greedy) matching
4. Understand how the regex engine backtracks with quantifiers
5. Apply quantifiers to character classes and groups
6. Choose the right quantifier for each situation
7. Avoid common quantifier pitfalls

---

## 1. The Three Basic Quantifiers

Quantifiers specify how many times the preceding element should occur:

```
Quantifier   Meaning              Matches
──────────   ──────────────       ─────────────────────────
*            Zero or more         "", "a", "aa", "aaa", ...
+            One or more          "a", "aa", "aaa", ...
?            Zero or one          "", "a"
```

### `*` -- Zero or More

```python
import re

# ab*c: b can appear 0 or more times
print(re.findall(r'ab*c', "ac abc abbc abbbc"))
# ['ac', 'abc', 'abbc', 'abbbc']
```

```
Pattern: ab*c

    "ac"     ->  a[]c       ✓  (zero b's)
    "abc"    ->  a[b]c      ✓  (one b)
    "abbc"   ->  a[bb]c     ✓  (two b's)
    "abbbc"  ->  a[bbb]c    ✓  (three b's)
    "adc"    ->  a[d]c      ✗  (d is not b)
```

### `+` -- One or More

```python
import re

# ab+c: b must appear at least once
print(re.findall(r'ab+c', "ac abc abbc abbbc"))
# ['abc', 'abbc', 'abbbc']
# Note: "ac" is NOT matched (needs at least one b)
```

```
Pattern: ab+c

    "ac"     ->  a[]c       ✗  (zero b's -- need at least one)
    "abc"    ->  a[b]c      ✓  (one b)
    "abbc"   ->  a[bb]c     ✓  (two b's)
    "abbbc"  ->  a[bbb]c    ✓  (three b's)
```

### `?` -- Zero or One

```python
import re

# colou?r: u is optional
print(re.findall(r'colou?r', "color and colour"))
# ['color', 'colour']

# https?: s is optional
print(re.findall(r'https?://', "http://a.com and https://b.com"))
# ['http://', 'https://']
```

```
Pattern: colou?r

    "color"  ->  colo[]r    ✓  (zero u's)
    "colour" ->  colo[u]r   ✓  (one u)
    "colouur"->  colo[uu]r  ✗  (two u's -- only 0 or 1 allowed)
```

---

## 2. Exact Repetition: `{n}`, `{n,}`, `{n,m}`

For precise control over repetition counts:

```
Syntax      Meaning
──────      ───────
{n}         Exactly n times
{n,}        n or more times
{n,m}       Between n and m times (inclusive)
{,m}        Up to m times (0 to m)
```

```python
import re

# {n}: Exactly n times
print(re.findall(r'\d{4}', "12 123 1234 12345"))
# ['1234', '1234']  (12345 contains two overlapping 4-digit sequences)

# {n,}: n or more
print(re.findall(r'\d{3,}', "1 12 123 1234 12345"))
# ['123', '1234', '12345']

# {n,m}: between n and m
print(re.findall(r'\d{2,4}', "1 12 123 1234 12345"))
# ['12', '123', '1234', '1234']  (greedy: takes as many as possible)
```

### Practical Examples

```python
import re

# US ZIP code: exactly 5 digits
zip_pattern = r'^\d{5}$'
print(re.fullmatch(zip_pattern, "12345"))   # Match
print(re.fullmatch(zip_pattern, "1234"))    # None

# US ZIP+4: 5 digits, optional dash and 4 digits
zip4_pattern = r'^\d{5}(-\d{4})?$'
print(re.fullmatch(zip4_pattern, "12345"))       # Match
print(re.fullmatch(zip4_pattern, "12345-6789"))  # Match

# Password: 8 to 20 characters
pwd_pattern = r'^.{8,20}$'
print(re.fullmatch(pwd_pattern, "short"))           # None
print(re.fullmatch(pwd_pattern, "longenough123"))   # Match
```

---

## 3. Greedy vs Lazy Matching

### Greedy (Default)

By default, quantifiers are **greedy** -- they match as much text as possible:

```python
import re

text = "<b>bold</b> and <i>italic</i>"

# Greedy: .* matches as much as possible
print(re.search(r'<.*>', text).group())
# '<b>bold</b> and <i>italic</i>'   -- matched everything!
```

```
Greedy Matching: <.*>

    Input: <b>bold</b> and <i>italic</i>
           ─────────────────────────────────
           <                               >
           ↑ start match     matched all ↑

    The .* gobbles up EVERYTHING, then backtracks
    until it finds the last >
```

### Lazy (Non-Greedy)

Add `?` after a quantifier to make it **lazy** -- match as little as possible:

```python
import re

text = "<b>bold</b> and <i>italic</i>"

# Lazy: .*? matches as little as possible
print(re.findall(r'<.*?>', text))
# ['<b>', '</b>', '<i>', '</i>']
```

```
Lazy Matching: <.*?>

    Input: <b>bold</b> and <i>italic</i>
           ───
           < >
           ↑ stops at FIRST >

    The .*? takes as FEW characters as possible,
    stopping at the first >
```

### Complete Lazy Quantifier Table

```
Greedy    Lazy       Meaning
──────    ────       ───────
*         *?         Zero or more (prefer fewer)
+         +?         One or more (prefer fewer)
?         ??         Zero or one (prefer zero)
{n,m}     {n,m}?     n to m times (prefer fewer)
{n,}      {n,}?      n or more (prefer fewer)
```

```python
import re

text = "aabab"

# Greedy vs lazy comparison
print(re.search(r'a.*b', text).group())    # "aabab" (greedy)
print(re.search(r'a.*?b', text).group())   # "aab"   (lazy)

print(re.search(r'a.+b', text).group())    # "aabab" (greedy)
print(re.search(r'a.+?b', text).group())   # "aab"   (lazy)
```

---

## 4. How Backtracking Works

Understanding backtracking is key to understanding greedy vs lazy:

### Greedy Backtracking

```
Pattern: ".*" matching: He said "hello" and "world"

Step 1: " matches the first "
Step 2: .* matches EVERYTHING to the end of string
        He said "hello" and "world"
                ──────────────────────  <- .* matches all this

Step 3: Engine needs to match closing "
        Backtracks one character at a time:
        "world"    <- tries here, but last char isn't "
        "world"   <- tries here, " found! But wrong one...
        Actually, regex finds " and stops (greedy finds LAST possible match)

Result: "hello" and "world"
```

### Lazy Backtracking

```
Pattern: ".*?" matching: He said "hello" and "world"

Step 1: " matches the first "
Step 2: .*? starts with ZERO characters
        Engine tries to match closing " immediately
Step 3: h is not " -- expand .*? by one character
Step 4: e is not " -- expand by one more
Step 5: l is not " -- continue...
Step 6: Eventually reaches " -- match found!

Result: "hello"
```

---

## 5. Quantifiers with Character Classes

Quantifiers apply to the element immediately before them:

```python
import re

# \d+ : one or more digits
print(re.findall(r'\d+', "Price: $12.99, Qty: 5"))
# ['12', '99', '5']

# [a-z]+ : one or more lowercase letters
print(re.findall(r'[a-z]+', "Hello World"))
# ['ello', 'orld']

# \w{3,} : word characters, 3 or more
print(re.findall(r'\w{3,}', "I am a developer at Google"))
# ['developer', 'Google']

# [aeiou]* : zero or more vowels (matches empty strings too!)
print(re.findall(r'[aeiou]+', "beautiful"))
# ['eau', 'i', 'u']
```

### Quantifiers with Groups

Quantifiers can apply to groups (covered in detail in Lesson 6):

```python
import re

# (ab)+ : the sequence "ab" one or more times
print(re.findall(r'(?:ab)+', "ababab cd ab"))
# ['ababab', 'ab']

# (ha){2,} : "ha" repeated 2+ times
print(re.findall(r'(?:ha){2,}', "ha haha hahaha"))
# ['haha', 'hahaha']
```

---

## 6. Practical Quantifier Patterns

### Matching Optional Sections

```python
import re

# Optional area code: (\d{3})?\s*\d{3}-\d{4}
phones = ["555-1234", "(800) 555-1234", "555-5678"]
pattern = r'\(?\d{3}\)?\s*\d{3}-\d{4}'

for phone in phones:
    match = re.fullmatch(pattern, phone)
    print(f"{phone:20s} -> {'Match' if match else 'No match'}")
```

### Matching CSV Fields

```python
import re

# Match quoted or unquoted CSV fields
csv_line = 'John,"Smith, Jr.",42,"New York"'
fields = re.findall(r'"[^"]*"|[^,]+', csv_line)
print(fields)  # ['John', '"Smith, Jr."', '42', '"New York"']
```

### Matching Numbers

```python
import re

text = "Integers: 42, -7. Floats: 3.14, -0.5, .99. Scientific: 1e10, 2.5E-3"

# Integer
print(re.findall(r'-?\d+', text))
# ['-42', '-7', '3', '14', '-0', '5', '99', '1', '10', '2', '5', '-3']

# Float (simple)
print(re.findall(r'-?\d*\.?\d+', text))
# ['42', '-7', '3.14', '-0.5', '.99', '1', '10', '2.5', '-3']

# Scientific notation
print(re.findall(r'-?\d+\.?\d*[eE][+-]?\d+', text))
# ['1e10', '2.5E-3']
```

---

## 7. Common Mistakes

### Mistake 1: `*` Matches Empty Strings

```python
import re

# * allows zero matches -- creates empty strings
print(re.findall(r'\d*', "abc"))
# ['', '', '', '']  -- matches empty string at each position!

# Use + instead to require at least one match
print(re.findall(r'\d+', "abc"))
# []  -- no digits found (clear result)
```

### Mistake 2: Greedy Matching Too Much

```python
import re

html = '<span class="name">John</span>'

# WRONG: greedy eats through multiple tags
print(re.search(r'<.*>', html).group())
# '<span class="name">John</span>'

# RIGHT: use lazy quantifier
print(re.search(r'<.*?>', html).group())
# '<span class="name">'

# BETTER: use negated character class (more efficient)
print(re.search(r'<[^>]+>', html).group())
# '<span class="name">'
```

### Mistake 3: Forgetting Quantifier Scope

```python
import re

# Quantifier applies to the element IMMEDIATELY before it
text = "abc abbc abbbc"

# Wrong thinking: ab+ means "ab" one or more times
# Correct: a followed by b+ (one or more b's)
print(re.findall(r'ab+c', text))
# ['abc', 'abbc', 'abbbc']

# To repeat "ab", use a group: (ab)+
print(re.findall(r'(?:ab)+c', text))
# ['abc', 'abbc']  -- different behavior!
```

---

## 8. Quantifier Decision Guide

```
Do you need the element to appear?
│
├── No (optional) ─────────────────> Use ? (zero or one)
│   └── Can it repeat? ───> Yes ──> Use * (zero or more)
│
└── Yes (required) ────────────────> Use + (one or more)
    └── Exact count? ──────> Yes ──> Use {n} or {n,m}

Do you want to match as little as possible?
└── Yes ──> Add ? after quantifier (lazy)
```

---

## Summary

| Quantifier | Meaning | Greedy Example | Lazy Example |
|-----------|---------|----------------|--------------|
| `*` | Zero or more | `a*` matches "" and "aaa" | `a*?` prefers "" |
| `+` | One or more | `a+` matches "a" and "aaa" | `a+?` prefers "a" |
| `?` | Zero or one | `a?` matches "" and "a" | `a??` prefers "" |
| `{n}` | Exactly n | `a{3}` matches "aaa" | N/A |
| `{n,}` | n or more | `a{2,}` matches "aa", "aaa" | `a{2,}?` prefers "aa" |
| `{n,m}` | n to m | `a{2,4}` matches "aa" to "aaaa" | `a{2,4}?` prefers "aa" |

Key takeaways:
- **Greedy** (default): match as much as possible, backtrack if needed
- **Lazy** (`?` suffix): match as little as possible, expand if needed
- Use `[^X]*` instead of `.*?` when possible (more efficient)

---

## Next Lesson

In [05_Anchors_and_Boundaries](./05_Anchors_and_Boundaries.md), we'll learn about position-based matching with anchors and word boundaries.

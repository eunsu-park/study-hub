# Literal Matching and Metacharacters

## Learning Objectives

After completing this lesson, you will be able to:

1. Match literal text in regex patterns
2. Identify and use the core metacharacters (`. ^ $ | \ ( ) [ ] { } * + ?`)
3. Escape metacharacters to match them literally
4. Use the dot (`.`) to match any character
5. Apply the alternation operator (`|`) for multiple alternatives
6. Understand anchors `^` and `$` at a basic level
7. Build simple patterns combining literals and metacharacters

---

## 1. Literal Matching

The simplest regex pattern is just plain text. Each character matches itself:

```python
import re

text = "The cat sat on the mat"

# Literal match: find "cat"
match = re.search(r'cat', text)
print(match.group())  # "cat"

# findall: find all occurrences of "at"
print(re.findall(r'at', text))  # ['at', 'at', 'at']
```

```
Pattern: "cat"

    T h e   c a t   s a t   o n   t h e   m a t
                ─────
                match

Each character in the pattern must match exactly:
    c -> c  ✓
    a -> a  ✓
    t -> t  ✓
```

### Case Sensitivity

By default, regex matching is **case-sensitive**:

```python
import re

text = "Python is great. PYTHON is powerful."

print(re.findall(r'Python', text))   # ['Python']
print(re.findall(r'PYTHON', text))   # ['PYTHON']
print(re.findall(r'python', text))   # []
```

---

## 2. The Metacharacters

Regex has 12 characters with special meaning. These are the **metacharacters**:

```
. ^ $ * + ? { } [ ] \ | ( )
```

Each metacharacter serves a specific purpose:

```
Metacharacter   Purpose                     Example
─────────────   ───────                     ───────
.               Any character (except \n)   a.c  matches "abc", "a1c"
^               Start of string/line        ^Hello  matches "Hello world"
$               End of string/line          world$  matches "Hello world"
*               Zero or more                ab*c  matches "ac", "abc", "abbc"
+               One or more                 ab+c  matches "abc", "abbc"
?               Zero or one                 colou?r  matches "color", "colour"
{n,m}           Repeat n to m times         \d{2,4}  matches "12", "123", "1234"
[...]           Character class             [aeiou]  matches any vowel
\               Escape / special sequence   \.  matches literal dot
|               Alternation (OR)            cat|dog  matches "cat" or "dog"
(...)           Grouping / capture          (ab)+  matches "ab", "abab"
```

---

## 3. The Dot (`.`) -- Any Character

The dot matches **any single character** except a newline (`\n`):

```python
import re

# . matches any single character
print(re.findall(r'c.t', "cat cot cut c1t c_t c\nt"))
# ['cat', 'cot', 'cut', 'c1t', 'c_t']
# Note: "c\nt" is NOT matched (dot doesn't match newline by default)
```

```
Pattern: c.t

    "cat"  ->  c[a]t  ✓   (a matches .)
    "cot"  ->  c[o]t  ✓   (o matches .)
    "cut"  ->  c[u]t  ✓   (u matches .)
    "c1t"  ->  c[1]t  ✓   (1 matches .)
    "c t"  ->  c[ ]t  ✓   (space matches .)
    "ct"   ->  c[]t   ✗   (. requires exactly one character)
    "c\nt" ->  c[\n]t ✗   (. doesn't match newline by default)
```

### Multiple Dots

```python
import re

# Two dots: match any two characters between 'a' and 'd'
print(re.findall(r'a..d', "abcd a12d a  d aXYd"))
# ['abcd', 'a12d', 'a  d']
```

---

## 4. Anchors: `^` and `$`

Anchors don't match characters -- they match **positions** in the string:

### `^` -- Start of String

```python
import re

text = "Python is great"

print(re.search(r'^Python', text))   # Match: "Python" at start
print(re.search(r'^is', text))       # None: "is" is not at start
```

### `$` -- End of String

```python
import re

text = "Python is great"

print(re.search(r'great$', text))    # Match: "great" at end
print(re.search(r'Python$', text))   # None: "Python" is not at end
```

### Combining `^` and `$`

```python
import re

# Match the entire string
print(re.search(r'^Python$', "Python"))          # Match
print(re.search(r'^Python$', "Python is great")) # None

# Validate a simple format
print(re.search(r'^\d{5}$', "12345"))   # Match (US ZIP code)
print(re.search(r'^\d{5}$', "1234"))    # None (too short)
print(re.search(r'^\d{5}$', "123456"))  # None (too long)
```

```
Anchor Visualization:

    ^ P y t h o n   i s   g r e a t $
    ↑                                 ↑
    ^= start position                 $= end position

    ^Python  -> matches (Python is at the start)
    great$   -> matches (great is at the end)
    ^great   -> no match (great is not at the start)
```

---

## 5. The Pipe (`|`) -- Alternation

The pipe operator means **OR** -- match the pattern on the left or the right:

```python
import re

text = "I have a cat and a dog and a bird"

# Match "cat" or "dog"
print(re.findall(r'cat|dog', text))   # ['cat', 'dog']

# Match "cat" or "dog" or "bird"
print(re.findall(r'cat|dog|bird', text))   # ['cat', 'dog', 'bird']
```

### Alternation Scope

The `|` operator has low precedence -- it splits the **entire** expression:

```python
import re

# "gray|grey" matches "gray" OR "grey"
print(re.findall(r'gray|grey', "gray and grey"))   # ['gray', 'grey']

# Equivalent with grouping: "gr(a|e)y"
print(re.findall(r'gr(a|e)y', "gray and grey"))    # ['a', 'e']
# Note: findall returns the captured group, not the full match!

# To get full match with groups, use finditer:
for m in re.finditer(r'gr(a|e)y', "gray and grey"):
    print(m.group())  # "gray", "grey"
```

```
Pattern: cat|dog

    Alternative 1: "cat"
                      OR
    Alternative 2: "dog"

Pattern: ^I (like|love|enjoy) (cats|dogs)$

    Start with "I "
    Then one of: "like", "love", "enjoy"
    Then " "
    Then one of: "cats", "dogs"
    End of string
```

---

## 6. The Backslash (`\`) -- Escape Character

The backslash has two roles:

### Role 1: Escape Metacharacters

To match a metacharacter literally, precede it with `\`:

```python
import re

# Without escaping: . matches any character
print(re.findall(r'.', "a.b"))     # ['a', '.', 'b']

# With escaping: \. matches only a literal dot
print(re.findall(r'\.', "a.b"))    # ['.']

# Match a literal dollar sign
price = "The price is $9.99"
print(re.search(r'\$\d+\.\d{2}', price).group())  # "$9.99"
```

### Escaping All Metacharacters

```
To match literally     Use in regex
──────────────────     ────────────
.                      \.
^                      \^
$                      \$
*                      \*
+                      \+
?                      \?
{                      \{
}                      \}
[                      \[
]                      \]
\                      \\
|                      \|
(                      \(
)                      \)
```

### Role 2: Special Sequences

The backslash also creates special character sequences (covered in later lessons):

```
\d    Any digit (0-9)
\w    Any word character (a-z, A-Z, 0-9, _)
\s    Any whitespace (space, tab, newline)
\b    Word boundary
\n    Newline
\t    Tab
```

### Using `re.escape()` for Dynamic Patterns

When building regex from user input, escape special characters:

```python
import re

user_input = "Is this real? (yes/no)"

# re.escape() adds backslashes before all metacharacters
escaped = re.escape(user_input)
print(escaped)  # 'Is\\ this\\ real\\?\\ \\(yes/no\\)'

# Safe to use in a regex
text = "Question: Is this real? (yes/no)"
match = re.search(re.escape(user_input), text)
print(match.group())  # "Is this real? (yes/no)"
```

---

## 7. Combining Literals and Metacharacters

Let's build increasingly complex patterns:

### Example 1: Match a Date-like Pattern

```python
import re

text = "Events on 2024-01-15 and 2024-12-31"

# \d = any digit (shorthand we'll cover next lesson)
dates = re.findall(r'\d\d\d\d-\d\d-\d\d', text)
print(dates)  # ['2024-01-15', '2024-12-31']
```

### Example 2: Match Simple File Extensions

```python
import re

files = "report.pdf, data.csv, image.png, script.py"

# Match filenames ending with .pdf or .csv
matches = re.findall(r'\w+\.(?:pdf|csv)', files)
print(matches)  # ['report.pdf', 'data.csv']
```

### Example 3: Match Lines Starting or Ending with Specific Text

```python
import re

log = """INFO: Server started
ERROR: Connection failed
INFO: Request received
ERROR: Timeout"""

# Find lines that start with "ERROR"
errors = re.findall(r'^ERROR.*', log, re.MULTILINE)
print(errors)
# ['ERROR: Connection failed', 'ERROR: Timeout']
```

---

## 8. Common Mistakes with Metacharacters

### Mistake 1: Forgetting to Escape

```python
import re

# WRONG: . matches any character
re.search(r'192.168.1.1', "192X168Y1Z1")  # Matches! (not intended)

# RIGHT: Escape the dots
re.search(r'192\.168\.1\.1', "192X168Y1Z1")  # None (correct)
```

### Mistake 2: Forgetting Raw Strings

```python
import re

# WRONG: \b is Python's backspace character
re.findall('\bword\b', "a word here")     # [] -- doesn't work!

# RIGHT: Use raw string
re.findall(r'\bword\b', "a word here")    # ['word']
```

### Mistake 3: Alternation Scope

```python
import re

# WRONG: Matches "cat" OR "dogs" (not "cats" or "dogs")
re.findall(r'cat|dogs', "cats and dogs")  # ['cat', 'dogs']

# RIGHT: Use grouping for shared parts
re.findall(r'(?:cat|dog)s', "cats and dogs")  # ['cats', 'dogs']
```

---

## 9. Pattern Building Strategy

When creating a regex pattern, follow this approach:

```
Step 1: Write out examples of what you want to match
        "192.168.1.1", "10.0.0.1", "172.16.0.1"

Step 2: Identify the fixed parts (literals)
        Numbers separated by dots

Step 3: Identify the variable parts (need metacharacters)
        1-3 digits in each position

Step 4: Build the pattern piece by piece
        \d+\.\d+\.\d+\.\d+

Step 5: Test with both matching and non-matching examples
        "192.168.1.1"  -> should match
        "999.999.999.999" -> matches (we'll refine later)
        "abc.def.ghi.jkl" -> should not match ✓
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Literal match | Characters match themselves (`abc` matches "abc") |
| `.` (dot) | Matches any single character except newline |
| `^` (caret) | Matches the start of string (or line in multiline mode) |
| `$` (dollar) | Matches the end of string (or line in multiline mode) |
| `\|` (pipe) | Alternation -- match left side OR right side |
| `\` (backslash) | Escapes metacharacters or creates special sequences |
| `re.escape()` | Automatically escapes all metacharacters in a string |
| Raw strings | Always use `r""` for regex patterns |

---

## Next Lesson

In [03_Character_Classes](./03_Character_Classes.md), we'll learn how to match specific sets of characters using character classes and shorthand notations.

# Groups and Capturing

## Learning Objectives

After completing this lesson, you will be able to:

1. Use parentheses `()` to create capture groups
2. Extract captured text using `match.group()` and `match.groups()`
3. Apply backreferences (`\1`, `\2`) to match repeated text
4. Create named groups with `(?P<name>...)` syntax
5. Use non-capturing groups `(?:...)` for grouping without capture
6. Nest groups and understand group numbering
7. Apply groups with quantifiers and alternation
8. Use `re.findall()` behavior with groups

---

## 1. Basic Capture Groups

Parentheses `()` create **capture groups** that serve two purposes:
1. **Grouping**: Treat multiple characters as a unit
2. **Capturing**: Save the matched text for later use

```python
import re

text = "Date: 2024-01-15"

# Without groups: get the entire match
match = re.search(r'\d{4}-\d{2}-\d{2}', text)
print(match.group())  # "2024-01-15"

# With groups: capture individual parts
match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
print(match.group())   # "2024-01-15"  (entire match)
print(match.group(1))  # "2024"        (year)
print(match.group(2))  # "01"          (month)
print(match.group(3))  # "15"          (day)
print(match.groups())  # ('2024', '01', '15')
```

```
Pattern: (\d{4})-(\d{2})-(\d{2})

    2  0  2  4  -  0  1  -  1  5
    ──────────     ─────     ─────
    Group 1        Group 2   Group 3
    \d{4}          \d{2}     \d{2}

    group(0) = "2024-01-15"   <- entire match
    group(1) = "2024"         <- first (...)
    group(2) = "01"           <- second (...)
    group(3) = "15"           <- third (...)
```

---

## 2. Group Numbering

Groups are numbered by their **opening parenthesis** position, from left to right:

```python
import re

text = "John Smith (age 30)"
pattern = r'((\w+)\s(\w+))\s\(age\s(\d+)\)'
match = re.search(pattern, text)

print(match.group(0))  # "John Smith (age 30)" - entire match
print(match.group(1))  # "John Smith"          - outer group
print(match.group(2))  # "John"                - first inner group
print(match.group(3))  # "Smith"               - second inner group
print(match.group(4))  # "30"                  - age group
```

```
Group Numbering (count opening parentheses left to right):

    ( ( \w+ ) \s ( \w+ ) ) \s \( age \s ( \d+ ) \)
    ↑ ↑           ↑               ↑
    1 2           3               4

    Group 0: entire match
    Group 1: ((\w+)\s(\w+))     = "John Smith"
    Group 2: (\w+) first        = "John"
    Group 3: (\w+) second       = "Smith"
    Group 4: (\d+)              = "30"
```

---

## 3. `re.findall()` with Groups

When `findall()` encounters groups, it returns the captured groups, NOT the full match:

```python
import re

text = "2024-01-15 and 2024-12-31"

# Without groups: returns full matches
print(re.findall(r'\d{4}-\d{2}-\d{2}', text))
# ['2024-01-15', '2024-12-31']

# With ONE group: returns list of group contents
print(re.findall(r'(\d{4})-\d{2}-\d{2}', text))
# ['2024', '2024']  -- only the captured year!

# With MULTIPLE groups: returns list of tuples
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# [('2024', '01', '15'), ('2024', '12', '31')]
```

This is a common source of confusion. To get the full match when using groups, either:
- Use `finditer()` and call `.group()` on each match
- Use non-capturing groups `(?:...)`

---

## 4. Named Groups: `(?P<name>...)`

Named groups make patterns more readable and maintainable:

```python
import re

text = "2024-01-15 08:30:45"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\s+(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'

match = re.search(pattern, text)
if match:
    print(match.group('year'))    # "2024"
    print(match.group('month'))   # "01"
    print(match.group('day'))     # "15"
    print(match.group('hour'))    # "08"
    print(match.groupdict())
    # {'year': '2024', 'month': '01', 'day': '15',
    #  'hour': '08', 'min': '30', 'sec': '45'}
```

### Named Groups in Substitution

```python
import re

# Reformat date: YYYY-MM-DD -> DD/MM/YYYY
text = "Date: 2024-01-15"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
result = re.sub(pattern, r'\g<day>/\g<month>/\g<year>', text)
print(result)  # "Date: 15/01/2024"
```

```
Named Group Syntax:

    Define:    (?P<name>pattern)
    Reference: \g<name> (in replacement)
    Access:    match.group('name')
    Dict:      match.groupdict()
```

---

## 5. Non-Capturing Groups: `(?:...)`

When you need grouping but don't need to capture:

```python
import re

# Capturing group: findall returns captured text
print(re.findall(r'(https?)://(\S+)', "http://a.com https://b.com"))
# [('http', 'a.com'), ('https', 'b.com')]

# Non-capturing group: findall returns full match
print(re.findall(r'(?:https?)://\S+', "http://a.com https://b.com"))
# ['http://a.com', 'https://b.com']
```

### When to Use Non-Capturing Groups

```python
import re

# Use (?:...) for grouping with alternation
text = "gray grey"
print(re.findall(r'gr(?:a|e)y', text))    # ['gray', 'grey']
print(re.findall(r'gr(a|e)y', text))      # ['a', 'e']  -- oops!

# Use (?:...) for grouping with quantifiers
text = "ababab cd abab"
print(re.findall(r'(?:ab)+', text))        # ['ababab', 'abab']
print(re.findall(r'(ab)+', text))          # ['ab', 'ab']  -- only last capture
```

---

## 6. Backreferences: `\1`, `\2`, ...

Backreferences match the **same text** that was captured by a previous group:

```python
import re

# \1 refers back to whatever group 1 matched
# Find doubled words
text = "the the cat sat sat on the the mat"
print(re.findall(r'\b(\w+)\s+\1\b', text))
# ['the', 'sat', 'the']
```

```
Backreference Visualization:

    Pattern: \b(\w+)\s+\1\b

    "the the" -> (\w+) captures "the", \1 checks for "the" again
                  ───                        ───
                 group 1            must match group 1 exactly

    "the cat" -> (\w+) captures "the", \1 checks for "the"
                  ───                        ───
                 "the"               "cat" != "the"  ✗
```

### More Backreference Examples

```python
import re

# Match HTML tags with matching closing tags
html = "<b>bold</b> <i>italic</i> <b>broken</i>"
pattern = r'<(\w+)>.*?</\1>'
print(re.findall(pattern, html))
# ['b', 'i']  -- the broken tag doesn't match

# Match repeated characters (like "aa", "bb")
text = "aardvark bookkeeper"
print(re.findall(r'(.)\1', text))
# ['a', 'o', 'k', 'e']

# Match quoted strings (same quote on both sides)
text = '''She said "hello" and 'goodbye' but not "mixed'"""
pattern = r'''(["'])(.*?)\1'''
print(re.findall(pattern, text))
# [('"', 'hello'), ("'", 'goodbye')]
```

### Named Backreferences

```python
import re

# Use (?P=name) to reference a named group
pattern = r'(?P<tag>\w+)>.*?</(?P=tag)>'
html = "<b>bold</b> <i>italic</i>"
print(re.findall(pattern, html))
# ['b', 'i']
```

---

## 7. Groups with Quantifiers

When a group is repeated with a quantifier, only the **last** capture is saved:

```python
import re

# (ab)+ captures only the last "ab"
match = re.search(r'(ab)+', "ababab")
print(match.group())   # "ababab"  (full match)
print(match.group(1))  # "ab"     (only last capture)

# To capture ALL repetitions, use findall on the group itself
text = "ababab"
print(re.findall(r'ab', text))  # ['ab', 'ab', 'ab']
```

```
Repeated Group Behavior:

    Pattern: (\d+,)*\d+    (comma-separated numbers)
    Input:   "1,2,3,4"

    Iteration 1: (\d+,) captures "1,"
    Iteration 2: (\d+,) captures "2,"  (overwrites "1,")
    Iteration 3: (\d+,) captures "3,"  (overwrites "2,")
    Final \d+:   matches "4"

    group(1) = "3,"  (only the last capture is saved)
```

---

## 8. Alternation in Groups

Groups are often used with `|` for alternation:

```python
import re

# Match different date formats
text = "Dates: 2024-01-15, 01/15/2024, Jan 15 2024"
pattern = r'(\d{4})-(\d{2})-(\d{2})|(\d{2})/(\d{2})/(\d{4})'

for match in re.finditer(pattern, text):
    print(f"Full: {match.group()}, Groups: {match.groups()}")

# Full: 2024-01-15, Groups: ('2024', '01', '15', None, None, None)
# Full: 01/15/2024, Groups: (None, None, None, '01', '15', '2024')
```

Note: unmatched groups return `None`. Named groups make this cleaner:

```python
import re

pattern = r'(?P<year1>\d{4})-(?P<month1>\d{2})-(?P<day1>\d{2})|(?P<month2>\d{2})/(?P<day2>\d{2})/(?P<year2>\d{4})'

text = "2024-01-15 and 01/15/2024"
for match in re.finditer(pattern, text):
    d = match.groupdict()
    year = d['year1'] or d['year2']
    month = d['month1'] or d['month2']
    day = d['day1'] or d['day2']
    print(f"{year}-{month}-{day}")
# 2024-01-15
# 2024-01-15
```

---

## 9. Practical Examples

### Example 1: Parse Key-Value Pairs

```python
import re

config = """
host=localhost
port=5432
database=mydb
user=admin
password=s3cret
"""

pairs = re.findall(r'^(\w+)=(.+)$', config, re.MULTILINE)
config_dict = dict(pairs)
print(config_dict)
# {'host': 'localhost', 'port': '5432', 'database': 'mydb',
#  'user': 'admin', 'password': 's3cret'}
```

### Example 2: Extract URL Components

```python
import re

url = "https://www.example.com:8080/path/to/page?q=search&lang=en"
pattern = r'(?P<scheme>\w+)://(?P<host>[^/:]+)(?::(?P<port>\d+))?(?P<path>/[^?]*)?(?:\?(?P<query>.+))?'

match = re.search(pattern, url)
if match:
    for key, value in match.groupdict().items():
        print(f"{key:10s}: {value}")

# scheme    : https
# host      : www.example.com
# port      : 8080
# path      : /path/to/page
# query     : q=search&lang=en
```

### Example 3: Swap First and Last Names

```python
import re

names = "Smith, John\nDoe, Jane\nPark, Eunsu"
# "Last, First" -> "First Last"
result = re.sub(r'(\w+),\s*(\w+)', r'\2 \1', names)
print(result)
# John Smith
# Jane Doe
# Eunsu Park
```

### Example 4: Find Repeated Words

```python
import re

text = "The the quick brown fox fox jumped over the lazy lazy dog"
dupes = re.findall(r'\b(\w+)\s+\1\b', text, re.IGNORECASE)
print(f"Repeated words: {dupes}")
# Repeated words: ['The', 'fox', 'lazy']
```

---

## 10. Group Reference Cheat Sheet

```
Syntax              Purpose                 Example
──────              ───────                 ───────
(...)               Capture group           (\d+) captures digits
(?:...)             Non-capturing group     (?:ab)+ groups without capture
(?P<name>...)       Named group             (?P<year>\d{4})
\1, \2              Backreference           (\w+)\s+\1
(?P=name)           Named backreference     (?P=tag)
\g<1>, \g<name>     Replacement reference   re.sub(r'(...)', r'\g<1>', ...)
match.group(n)      Access by number        match.group(1)
match.group('name') Access by name          match.group('year')
match.groups()      All groups as tuple     ('2024', '01', '15')
match.groupdict()   Named groups as dict    {'year': '2024', ...}
```

---

## Summary

| Concept | Syntax | Purpose |
|---------|--------|---------|
| Capture group | `(...)` | Group and capture matched text |
| Non-capturing group | `(?:...)` | Group without capturing |
| Named group | `(?P<name>...)` | Capture with a name |
| Backreference | `\1`, `\2` | Match same text as a previous group |
| Named backreference | `(?P=name)` | Match same text as a named group |
| Group access | `.group(n)` | Get captured text by number |
| All groups | `.groups()` | Tuple of all captured groups |
| Group dict | `.groupdict()` | Dict of named groups |

---

## Next Lesson

In [07_Lookahead_and_Lookbehind](./07_Lookahead_and_Lookbehind.md), we'll learn about zero-width assertions that check what comes before or after a position without consuming text.

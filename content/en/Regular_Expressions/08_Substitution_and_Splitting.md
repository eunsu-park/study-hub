# Substitution and Splitting

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `re.sub()` for pattern-based text replacement
2. Reference capture groups in replacement strings
3. Write callback functions for dynamic replacements
4. Use `re.subn()` to count replacements
5. Split strings on complex patterns with `re.split()`
6. Control splitting behavior with `maxsplit` and groups
7. Handle edge cases in splitting (empty strings, consecutive delimiters)

---

## 1. Basic Substitution with `re.sub()`

`re.sub(pattern, replacement, string, count=0, flags=0)` replaces all occurrences of a pattern:

```python
import re

text = "I have 3 cats and 2 dogs"

# Replace all digits with "#"
result = re.sub(r'\d', '#', text)
print(result)  # "I have # cats and # dogs"

# Replace sequences of digits
result = re.sub(r'\d+', 'N', text)
print(result)  # "I have N cats and N dogs"
```

### The `count` Parameter

Limit the number of replacements:

```python
import re

text = "aaa bbb ccc aaa bbb"

# Replace only the first 2 occurrences
result = re.sub(r'aaa|bbb', 'XXX', text, count=2)
print(result)  # "XXX XXX ccc aaa bbb"

# Replace only the first occurrence
result = re.sub(r'aaa|bbb', 'XXX', text, count=1)
print(result)  # "XXX bbb ccc aaa bbb"
```

---

## 2. Group References in Replacements

Use `\1`, `\2`, or `\g<name>` in the replacement string to reference captured groups:

```python
import re

# Swap first and last names
text = "Smith, John"
result = re.sub(r'(\w+), (\w+)', r'\2 \1', text)
print(result)  # "John Smith"

# Duplicate a word
text = "hello world"
result = re.sub(r'(\w+)', r'\1-\1', text)
print(result)  # "hello-hello world-world"

# Reformat dates: MM/DD/YYYY -> YYYY-MM-DD
text = "01/15/2024 and 12/31/2024"
result = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\3-\1-\2', text)
print(result)  # "2024-01-15 and 2024-12-31"
```

### Named Group References

```python
import re

# Using \g<name> for named groups
text = "2024-01-15"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
result = re.sub(pattern, r'\g<day>/\g<month>/\g<year>', text)
print(result)  # "15/01/2024"
```

```
Group Reference Syntax in Replacement Strings:

    \1, \2, \3      Numbered groups
    \g<1>, \g<2>    Numbered groups (explicit, avoids ambiguity)
    \g<name>        Named groups

    Why \g<1> instead of \1?
    Pattern: r'(\d)0'  replacement: r'\g<1>5' -> clear: group 1 + "5"
                                    r'\15'    -> ambiguous: group 1 + "5" or group 15?
```

---

## 3. Callback Functions in `re.sub()`

Pass a **function** as the replacement to perform dynamic transformations:

```python
import re

text = "I have 3 cats and 12 dogs"

# Double all numbers
def double_number(match):
    num = int(match.group())
    return str(num * 2)

result = re.sub(r'\d+', double_number, text)
print(result)  # "I have 6 cats and 24 dogs"
```

The callback receives a **Match object** and must return a **string**:

```python
import re

# Convert temperatures: "72F" -> "22.2C"
text = "Today: 72F, Tomorrow: 85F, Record: 104F"

def fahrenheit_to_celsius(match):
    f = int(match.group(1))
    c = (f - 32) * 5 / 9
    return f"{c:.1f}C"

result = re.sub(r'(\d+)F', fahrenheit_to_celsius, text)
print(result)  # "Today: 22.2C, Tomorrow: 29.4C, Record: 40.0C"
```

### Lambda Callbacks

```python
import re

# Uppercase all words starting with 'p'
text = "python programming is powerful and practical"
result = re.sub(r'\bp\w+', lambda m: m.group().upper(), text)
print(result)  # "PYTHON PROGRAMMING is POWERFUL and PRACTICAL"

# Censor bad words
text = "This is damn bad and hell ugly"
bad_words = {'damn', 'hell'}
result = re.sub(
    r'\b\w+\b',
    lambda m: '***' if m.group().lower() in bad_words else m.group(),
    text
)
print(result)  # "This is *** bad and *** ugly"
```

---

## 4. `re.subn()` -- Substitution with Count

`re.subn()` returns a tuple of `(new_string, number_of_substitutions)`:

```python
import re

text = "cat bat rat cat mat"

result, count = re.subn(r'[cbr]at', 'dog', text)
print(f"Result: {result}")    # "dog dog dog dog mat"
print(f"Replacements: {count}")  # 4

# Useful for checking if any replacement was made
text = "no matches here"
result, count = re.subn(r'\d+', 'NUM', text)
if count == 0:
    print("No replacements made")
```

---

## 5. Advanced Substitution Patterns

### Conditional Replacement

```python
import re

# Pluralize words: add "s" or "es"
words = ["cat", "box", "bus", "dog", "dish"]
for word in words:
    plural = re.sub(r'(s|x|sh|ch)$', r'\1es', word)
    if plural == word:  # no substitution happened
        plural = word + 's'
    print(f"{word} -> {plural}")
# cat -> cats
# box -> boxes
# bus -> buses
# dog -> dogs
# dish -> dishes
```

### Remove Duplicate Whitespace

```python
import re

text = "Hello    World   !    How    are   you?"
clean = re.sub(r'\s+', ' ', text)
print(clean)  # "Hello World ! How are you?"
```

### Wrap Matches with Tags

```python
import re

text = "Important: read the manual carefully"
# Wrap capitalized words in bold tags
result = re.sub(r'\b([A-Z]\w*)', r'<b>\1</b>', text)
print(result)  # "<b>Important</b>: read the manual carefully"
```

---

## 6. Basic Splitting with `re.split()`

`re.split(pattern, string, maxsplit=0, flags=0)` splits a string by pattern matches:

```python
import re

# Split on any whitespace
text = "Hello   World\tPython\nRegex"
print(re.split(r'\s+', text))
# ['Hello', 'World', 'Python', 'Regex']

# Split on comma with optional whitespace
text = "apple, banana,cherry ,  grape"
print(re.split(r'\s*,\s*', text))
# ['apple', 'banana', 'cherry', 'grape']

# Split on multiple delimiters
text = "one;two,three:four|five"
print(re.split(r'[;,:|]', text))
# ['one', 'two', 'three', 'four', 'five']
```

### The `maxsplit` Parameter

```python
import re

text = "one,two,three,four,five"

# Split into at most 3 parts
print(re.split(r',', text, maxsplit=2))
# ['one', 'two', 'three,four,five']

# Split only the first delimiter
print(re.split(r',', text, maxsplit=1))
# ['one', 'two,three,four,five']
```

---

## 7. Splitting with Capture Groups

When the split pattern contains capture groups, the captured text is **included** in the result:

```python
import re

text = "one1two2three3four"

# Without groups: delimiters are removed
print(re.split(r'\d', text))
# ['one', 'two', 'three', 'four']

# With groups: delimiters are included
print(re.split(r'(\d)', text))
# ['one', '1', 'two', '2', 'three', '3', 'four']
```

```
Split with Capture Group Visualization:

    Input:   "one1two2three"
    Pattern: (\d)

    Split positions:
    "one" | 1 | "two" | 2 | "three"
           ↑            ↑
           delimiter     delimiter

    Without (): ['one', 'two', 'three']    (delimiters discarded)
    With ():    ['one', '1', 'two', '2', 'three']  (delimiters kept)
```

### Practical Use: Keep Delimiters

```python
import re

# Split sentences but keep the punctuation
text = "Hello! How are you? I'm fine. Thanks!"
parts = re.split(r'([.!?])\s*', text)
print(parts)
# ['Hello', '!', 'How are you', '?', "I'm fine", '.', 'Thanks', '!', '']

# Reconstruct sentences with punctuation
sentences = []
for i in range(0, len(parts) - 1, 2):
    if parts[i]:
        sentences.append(parts[i] + parts[i+1])
print(sentences)
# ['Hello!', 'How are you?', "I'm fine.", 'Thanks!']
```

---

## 8. Edge Cases in Splitting

### Empty Strings from Consecutive Delimiters

```python
import re

text = "one,,two,,,three"
print(re.split(r',', text))
# ['one', '', 'two', '', '', 'three']

# Filter out empty strings
parts = [p for p in re.split(r',', text) if p]
print(parts)  # ['one', 'two', 'three']
```

### Pattern at Start or End

```python
import re

text = ",one,two,three,"
print(re.split(r',', text))
# ['', 'one', 'two', 'three', '']
# Note: empty strings at start and end
```

### Splitting on Lookaround (Zero-Width)

```python
import re

# Split camelCase into words (split before uppercase)
text = "camelCaseToSeparateWords"
parts = re.split(r'(?=[A-Z])', text)
print(parts)  # ['camel', 'Case', 'To', 'Separate', 'Words']

# Split on word boundaries
text = "hello-world_foo bar"
parts = re.split(r'[-_\s]', text)
print(parts)  # ['hello', 'world', 'foo', 'bar']
```

---

## 9. Practical Examples

### Example 1: Template Variable Replacement

```python
import re

template = "Hello, {{name}}! You have {{count}} new messages."
variables = {"name": "Alice", "count": "5"}

def replace_var(match):
    var_name = match.group(1)
    return variables.get(var_name, match.group())

result = re.sub(r'\{\{(\w+)\}\}', replace_var, template)
print(result)  # "Hello, Alice! You have 5 new messages."
```

### Example 2: Clean HTML Tags

```python
import re

html = "<p>Hello <b>World</b>! Click <a href='url'>here</a>.</p>"

# Remove all HTML tags
clean = re.sub(r'<[^>]+>', '', html)
print(clean)  # "Hello World! Click here."

# Remove specific tags but keep content
clean = re.sub(r'</?(?:b|i|em|strong)>', '', html)
print(clean)  # "<p>Hello World! Click <a href='url'>here</a>.</p>"
```

### Example 3: Normalize Whitespace

```python
import re

text = "  Hello   World  \n\n  How   are   you?  \n  "

# Collapse whitespace, trim
clean = re.sub(r'\s+', ' ', text).strip()
print(f"'{clean}'")
# 'Hello World How are you?'

# Normalize line-by-line (preserve line breaks)
clean = re.sub(r'[ \t]+', ' ', text)      # collapse spaces/tabs
clean = re.sub(r'^ | $', '', clean, flags=re.MULTILINE)  # trim lines
clean = re.sub(r'\n{3,}', '\n\n', clean)  # max 2 newlines
print(clean)
```

### Example 4: Parse and Transform a Log File

```python
import re

log = """[2024-01-15 08:30:45] ERROR: Connection refused
[2024-01-15 08:30:46] INFO: Retrying connection
[2024-01-15 08:30:47] ERROR: Connection timeout
[2024-01-15 08:30:50] INFO: Connection established"""

# Extract timestamps from ERROR lines
errors = re.findall(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ERROR: (.+)',
    log
)
for timestamp, message in errors:
    print(f"{timestamp} - {message}")
# 2024-01-15 08:30:45 - Connection refused
# 2024-01-15 08:30:47 - Connection timeout

# Convert log format: [timestamp] LEVEL: msg -> timestamp|LEVEL|msg
result = re.sub(
    r'\[(.+?)\] (\w+): (.+)',
    r'\1|\2|\3',
    log
)
print(result)
```

### Example 5: Smart Split on Commas (Respecting Quotes)

```python
import re

# Split on commas, but not commas inside quotes
line = 'John,"Smith, Jr.",42,"New York, NY"'

# Use findall instead of split for this case
fields = re.findall(r'"[^"]*"|[^,]+', line)
print(fields)
# ['John', '"Smith, Jr."', '42', '"New York, NY"']

# Strip quotes
fields = [f.strip('"') for f in fields]
print(fields)
# ['John', 'Smith, Jr.', '42', 'New York, NY']
```

---

## 10. Performance Considerations

```python
import re

# Compile patterns used in loops
pattern = re.compile(r'\b\w+\b')

# Use subn to check if replacement was needed
text = "some text"
result, n = re.subn(r'pattern', 'replacement', text)
if n > 0:
    print(f"Made {n} replacements")

# Avoid unnecessary regex -- use str methods for simple cases
text = "hello world"
# GOOD: text.replace("hello", "hi")     <- faster
# OK:   re.sub(r'hello', 'hi', text)    <- regex overhead

# Use re.sub only when you need PATTERNS
# BAD:  re.sub(r'hello', 'hi', text)    <- overkill
# GOOD: re.sub(r'\bhello\b', 'hi', text) <- needs word boundary
```

---

## Summary

| Function | Purpose | Returns |
|----------|---------|---------|
| `re.sub(pat, repl, s)` | Replace all matches | New string |
| `re.sub(pat, func, s)` | Replace with callback | New string |
| `re.subn(pat, repl, s)` | Replace and count | (new_string, count) |
| `re.split(pat, s)` | Split on pattern | List of strings |

Key replacement syntax:
- `\1`, `\2` -- numbered group references
- `\g<1>`, `\g<name>` -- explicit group references
- Callback function receives Match object, returns string

---

## Next Lesson

In [09_Flags_and_Options](./09_Flags_and_Options.md), we'll explore regex flags that modify how patterns are interpreted.

# Character Classes

## Learning Objectives

After completing this lesson, you will be able to:

1. Define custom character classes using square brackets `[...]`
2. Use character ranges (`[a-z]`, `[0-9]`) for concise patterns
3. Negate character classes with `[^...]`
4. Use shorthand character classes (`\d`, `\w`, `\s`) and their negations
5. Understand the difference between character classes inside and outside brackets
6. Combine character classes with other regex features
7. Handle Unicode characters in character classes

---

## 1. What Is a Character Class?

A **character class** (or character set) matches **one character** from a defined set. It is enclosed in square brackets `[...]`:

```python
import re

text = "bag big bog bug"

# [aeiou] matches any single vowel
print(re.findall(r'b[aeiou]g', text))
# ['bag', 'big', 'bog', 'bug']
```

```
Pattern: b[aeiou]g

    Breakdown:
    b         - literal 'b'
    [aeiou]   - ANY ONE character from the set {a, e, i, o, u}
    g         - literal 'g'

    "bag" -> b[a]g ✓  (a is in [aeiou])
    "beg" -> b[e]g ✓  (e is in [aeiou])
    "big" -> b[i]g ✓  (i is in [aeiou])
    "bog" -> b[o]g ✓  (o is in [aeiou])
    "bug" -> b[u]g ✓  (u is in [aeiou])
    "byg" -> b[y]g ✗  (y is NOT in [aeiou])
```

---

## 2. Character Ranges

Use a hyphen inside brackets to specify a range:

```python
import re

# [a-z] matches any lowercase letter
print(re.findall(r'[a-z]+', "Hello World 123"))
# ['ello', 'orld']

# [A-Z] matches any uppercase letter
print(re.findall(r'[A-Z]+', "Hello World 123"))
# ['H', 'W']

# [0-9] matches any digit
print(re.findall(r'[0-9]+', "Hello World 123"))
# ['123']

# [a-zA-Z] matches any letter (upper or lower)
print(re.findall(r'[a-zA-Z]+', "Hello World 123"))
# ['Hello', 'World']

# [a-zA-Z0-9] matches any alphanumeric character
print(re.findall(r'[a-zA-Z0-9]+', "user_name@email.com"))
# ['user', 'name', 'email', 'com']
```

```
Range Mechanics (based on ASCII/Unicode code points):

    [a-z]   = a b c d e f g h i j k l m n o p q r s t u v w x y z
               97 ────────────────────────────────────────────> 122

    [A-Z]   = A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
               65 ────────────────────────────────────────────> 90

    [0-9]   = 0 1 2 3 4 5 6 7 8 9
               48 ──────────────> 57

    [a-f]   = a b c d e f   (partial range)
    [3-7]   = 3 4 5 6 7     (partial range)
```

### Multiple Ranges

```python
import re

# Hex digits: [0-9a-fA-F]
hex_pattern = r'[0-9a-fA-F]+'
print(re.findall(hex_pattern, "Color: #FF8C00, value=0x1A3F"))
# ['FF8C00', '0', '1A3F']
```

---

## 3. Negated Character Classes

Place `^` immediately after `[` to match any character **NOT** in the set:

```python
import re

# [^aeiou] matches any character that is NOT a vowel
text = "Hello World"
print(re.findall(r'[^aeiou]', text))
# ['H', 'l', 'l', ' ', 'W', 'r', 'l', 'd']

# [^0-9] matches any non-digit character
print(re.findall(r'[^0-9]+', "abc123def456"))
# ['abc', 'def']

# [^a-zA-Z] matches any non-letter character
print(re.findall(r'[^a-zA-Z]+', "Hello, World! 123"))
# [', ', '! ', '123']
```

```
Negation Visualization:

    [aeiou]    Matches: a e i o u
               Rejects: everything else

    [^aeiou]   Matches: everything else
               Rejects: a e i o u

    Think of [^...] as "anything EXCEPT these characters"
```

**Important**: `^` only means negation when it's the **first** character after `[`. Elsewhere inside brackets, it's a literal `^`:

```python
import re

# ^ as first char: negation
print(re.findall(r'[^abc]', "a^bc"))   # ['^']

# ^ not as first char: literal ^
print(re.findall(r'[a^bc]', "a^bc"))   # ['a', '^', 'b', 'c']
```

---

## 4. Special Characters Inside Character Classes

Most metacharacters lose their special meaning inside `[...]`:

```python
import re

# Inside brackets, most metacharacters are literal
text = "a.b a+b a*b a?b"
print(re.findall(r'a[.+*?]b', text))
# ['a.b', 'a+b', 'a*b', 'a?b']
```

Characters that ARE special inside `[...]`:

```
Character   Behavior Inside [...]          How to Use Literally
─────────   ────────────────────           ────────────────────
]           Closes the bracket             Place first: []abc] or escape: [\]]
\           Escape character               Escape it: [\\]
^           Negation (if first)            Don't place first: [a^b]
-           Range separator                Place first/last: [-abc] or [abc-]
```

```python
import re

# Match literal ], \, ^, -
text = "a] b\\ c^ d- e"

# ] first, ^ not first, - last
print(re.findall(r'[]\^\\-]', text))  # [']', '\\', '^', '-']

# Or escape them
print(re.findall(r'[\]\^\\-]', text))  # [']', '\\', '^', '-']
```

---

## 5. Shorthand Character Classes

Python regex provides shorthand notations for common character classes:

```
Shorthand   Equivalent         Meaning
─────────   ──────────         ───────
\d          [0-9]              Any digit
\D          [^0-9]             Any non-digit
\w          [a-zA-Z0-9_]      Any word character
\W          [^a-zA-Z0-9_]     Any non-word character
\s          [ \t\n\r\f\v]     Any whitespace
\S          [^ \t\n\r\f\v]    Any non-whitespace
```

```python
import re

text = "User: alice_99, Age: 25, Email: alice@test.com"

# \d - digits
print(re.findall(r'\d+', text))
# ['99', '25']

# \w - word characters (letters, digits, underscore)
print(re.findall(r'\w+', text))
# ['User', 'alice_99', 'Age', '25', 'Email', 'alice', 'test', 'com']

# \s - whitespace
print(re.findall(r'\s+', text))
# [' ', ' ', ' ', ' ', ' ']

# \D - non-digits
print(re.findall(r'\D+', text))
# ['User: alice_', ', Age: ', ', Email: alice@test.com']
```

### Uppercase = Negation

The pattern is simple: lowercase shorthand matches a class, uppercase matches the opposite.

```
\d  (digit)      <-->  \D  (non-digit)
\w  (word char)  <-->  \W  (non-word char)
\s  (whitespace) <-->  \S  (non-whitespace)
```

```python
import re

text = "Hello, World! 123"

# \W matches non-word characters (punctuation, spaces)
print(re.findall(r'\W+', text))
# [', ', '! ']

# \S matches non-whitespace
print(re.findall(r'\S+', text))
# ['Hello,', 'World!', '123']
```

---

## 6. Combining Shorthands with Character Classes

You can mix shorthands and literal characters inside brackets:

```python
import re

# Match digits or hyphens (for phone numbers)
text = "Call 555-867-5309"
print(re.findall(r'[\d-]+', text))
# ['555-867-5309']

# Match word characters or dots (for filenames)
text = "file.txt image.png README"
print(re.findall(r'[\w.]+', text))
# ['file.txt', 'image.png', 'README']

# Match letters, digits, or common symbols
text = "price = $19.99 + tax"
print(re.findall(r'[\w$+.]+', text))
# ['price', '$19.99', '+', 'tax']
```

---

## 7. The Dot vs Character Classes

The dot (`.`) matches almost any character. Character classes give you precise control:

```
Comparison: . vs [...]

    .         Matches ANY character except \n
    [.]       Matches ONLY a literal dot
    [^\n]     Equivalent to . (matches any character except \n)
    [\s\S]    Matches ANY character INCLUDING \n
```

```python
import re

text = "a.b acb a\nb"

# . matches any character (except newline)
print(re.findall(r'a.b', text))     # ['a.b', 'acb']

# [.] matches only a literal dot
print(re.findall(r'a[.]b', text))   # ['a.b']

# [\s\S] matches anything including newline
print(re.findall(r'a[\s\S]b', text))  # ['a.b', 'acb', 'a\nb']
```

---

## 8. POSIX-like Classes (Limited in Python)

Some regex flavors support POSIX classes like `[:alpha:]`. Python's `re` module does NOT support them directly, but you can achieve the same with Unicode-aware patterns:

```python
import re

# Python doesn't support [:alpha:], use these instead:
# Letters:      [a-zA-Z] or \w (includes digits and _)
# Digits:       [0-9] or \d
# Alphanumeric: [a-zA-Z0-9] or [\w] (includes _)
# Whitespace:   \s

# For Unicode letters, use re.UNICODE flag (default in Python 3)
text = "Hello cafe resume"
print(re.findall(r'[a-zA-Z]+', text))  # ASCII letters only
# ['Hello', 'cafe', 'resume']
```

---

## 9. Unicode and Character Classes

In Python 3, `\w`, `\d`, and `\s` are Unicode-aware by default:

```python
import re

# \w matches Unicode word characters
text = "hello world"
print(re.findall(r'\w+', text))
# ['hello', 'world']

# \d matches Unicode digits (not just 0-9)
text = "123 numbers"
print(re.findall(r'\d+', text))
# ['123']

# To restrict to ASCII only, use re.ASCII flag
print(re.findall(r'\w+', text, re.ASCII))
# ['123', 'numbers']
```

---

## 10. Practical Examples

### Example 1: Extract Words Without Vowels

```python
import re

text = "my gym by cry fly rhythm"
# Words with only consonants (and y)
words = re.findall(r'\b[^aeiouAEIOU\W]+\b', text)
print(words)  # ['my', 'gym', 'by', 'cry', 'fly', 'rhythm']
```

### Example 2: Match Hex Color Codes

```python
import re

css = "colors: #FF0000, #00ff00, #0000FF, #abc, not #xyz"
hex_colors = re.findall(r'#[0-9a-fA-F]{3,6}', css)
print(hex_colors)  # ['#FF0000', '#00ff00', '#0000FF', '#abc']
```

### Example 3: Clean Up Whitespace

```python
import re

text = "  Hello   World   !  "
# Replace multiple whitespace with single space
clean = re.sub(r'\s+', ' ', text).strip()
print(f"'{clean}'")  # 'Hello World !'
```

### Example 4: Extract Initials

```python
import re

name = "John Michael Smith"
initials = re.findall(r'[A-Z]', name)
print("".join(initials))  # "JMS"
```

---

## Summary

| Concept | Syntax | Example |
|---------|--------|---------|
| Character class | `[abc]` | Match a, b, or c |
| Range | `[a-z]` | Match lowercase letter |
| Negation | `[^abc]` | Match anything except a, b, c |
| Digit | `\d` / `[0-9]` | Match any digit |
| Non-digit | `\D` / `[^0-9]` | Match any non-digit |
| Word char | `\w` / `[a-zA-Z0-9_]` | Match letter, digit, underscore |
| Non-word char | `\W` / `[^a-zA-Z0-9_]` | Match non-word character |
| Whitespace | `\s` | Match space, tab, newline, etc. |
| Non-whitespace | `\S` | Match non-whitespace |
| Literal dot in class | `[.]` | Match only a dot |

---

## Next Lesson

In [04_Quantifiers](./04_Quantifiers.md), we'll learn how to specify how many times a character or group should repeat.

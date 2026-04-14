# Regular Expressions Learning Guide

## Introduction

This folder contains materials for learning Regular Expressions (regex) -- one of the most powerful tools for text processing available to programmers. From simple pattern matching to complex text transformations, regex is an essential skill for tasks like data validation, log analysis, search-and-replace operations, and data cleaning.

**Target Audience**: Developers who want to master pattern matching and text processing

---

## Learning Roadmap

```
[Foundations]           [Core Syntax]           [Advanced]              [Practical]
    |                       |                       |                       |
    v                       v                       v                       v
What Are Regex -------> Char Classes ---------> Lookahead/ -----------> Common
    |                       |                   Lookbehind              Patterns
    v                       v                       |                       |
Literals & -----------> Quantifiers                 v                       v
Metacharacters              |                   Substitution &          Performance &
    |                       v                   Splitting               Pitfalls
    |                   Anchors &                   |                       |
    |                   Boundaries                  v                       v
    |                       |                   Flags &                 Real-World
    |                       v                   Options                 Applications
    +-----------------> Groups &
                        Capturing
```

---

## Prerequisites

- Basic Python programming ([Programming](../Programming/00_Overview.md))
- Familiarity with strings and string methods

---

## What You'll Learn

- How regular expressions work internally (finite automata concepts)
- Matching literal text and using metacharacters
- Character classes, shorthand notations, and Unicode support
- Quantifiers (greedy, lazy, possessive) and their behavior
- Anchors, word boundaries, and multiline matching
- Capturing groups, backreferences, and named groups
- Lookahead and lookbehind assertions
- Text substitution, splitting, and callback-based replacements
- Regex flags and inline modifiers
- Battle-tested patterns for common validation tasks
- Performance optimization and catastrophic backtracking prevention
- Real-world applications in log parsing, data cleaning, and refactoring

---

## File List

| Filename | Difficulty | Key Topics |
|----------|-----------|------------|
| [01_What_Are_Regular_Expressions.md](./01_What_Are_Regular_Expressions.md) | ⭐ | History, use cases, Python re module basics |
| [02_Literal_Matching_and_Metacharacters.md](./02_Literal_Matching_and_Metacharacters.md) | ⭐ | Literal text, `.`, `^`, `$`, `\|`, escaping |
| [03_Character_Classes.md](./03_Character_Classes.md) | ⭐ | `[abc]`, ranges, `\d`, `\w`, `\s`, negation |
| [04_Quantifiers.md](./04_Quantifiers.md) | ⭐⭐ | `*`, `+`, `?`, `{n,m}`, greedy vs lazy |
| [05_Anchors_and_Boundaries.md](./05_Anchors_and_Boundaries.md) | ⭐⭐ | `^`, `$`, `\b`, `\B`, multiline anchors |
| [06_Groups_and_Capturing.md](./06_Groups_and_Capturing.md) | ⭐⭐ | `()`, backreferences, named groups, non-capturing |
| [07_Lookahead_and_Lookbehind.md](./07_Lookahead_and_Lookbehind.md) | ⭐⭐⭐ | `(?=)`, `(?!)`, `(?<=)`, `(?<!)` |
| [08_Substitution_and_Splitting.md](./08_Substitution_and_Splitting.md) | ⭐⭐ | `re.sub`, `re.split`, callback functions |
| [09_Flags_and_Options.md](./09_Flags_and_Options.md) | ⭐⭐ | `IGNORECASE`, `MULTILINE`, `DOTALL`, `VERBOSE` |
| [10_Common_Patterns.md](./10_Common_Patterns.md) | ⭐⭐ | Email, URL, IP, date, phone validation |
| [11_Performance_and_Pitfalls.md](./11_Performance_and_Pitfalls.md) | ⭐⭐⭐ | Catastrophic backtracking, optimization techniques |
| [12_Real_World_Applications.md](./12_Real_World_Applications.md) | ⭐⭐⭐ | Log parsing, data cleaning, code refactoring |

---

## Recommended Learning Path

### Stage 1: Foundations (Lessons 1-3)
1. What Are Regular Expressions -> Literal Matching -> Character Classes

### Stage 2: Core Syntax (Lessons 4-6)
2. Quantifiers -> Anchors and Boundaries -> Groups and Capturing

### Stage 3: Advanced Patterns (Lessons 7-9)
3. Lookahead and Lookbehind -> Substitution and Splitting -> Flags and Options

### Stage 4: Practical Mastery (Lessons 10-12)
4. Common Patterns -> Performance and Pitfalls -> Real-World Applications

---

## Quick Start

### Test a Regex in Python

```python
import re

# Find all email addresses in text
text = "Contact us at support@example.com or sales@example.com"
emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
print(emails)  # ['support@example.com', 'sales@example.com']
```

### Interactive Practice

```python
import re

pattern = r'\d{3}-\d{4}'
text = "Call 555-1234 or 555-5678"

for match in re.finditer(pattern, text):
    print(f"Found: {match.group()} at position {match.start()}-{match.end()}")
```

---

## Related Materials

- [Python Basics](../Python_Basics/00_Overview.md) - Python fundamentals
- [Shell Script](../Shell_Script/00_Overview.md) - grep and sed use regex extensively
- [Data Science](../Data_Science/00_Overview.md) - Text preprocessing with regex

# Performance and Pitfalls

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand how regex engines work (NFA vs DFA)
2. Identify and prevent catastrophic backtracking
3. Recognize dangerous pattern constructs
4. Use atomic grouping concepts for optimization
5. Apply `re.compile()` effectively for performance
6. Choose between regex and string methods appropriately
7. Benchmark and profile regex performance
8. Write efficient patterns using best practices

---

## 1. How Regex Engines Work

Python uses an **NFA (Non-deterministic Finite Automaton)** engine, which means it uses **backtracking** to try different matching possibilities:

```
NFA Engine Process (simplified):

    Pattern: a(b|c)*d
    Input:   "abcd"

    Step 1: 'a' matches 'a' ✓
    Step 2: (b|c)* tries 'b' -> matches 'b' ✓
    Step 3: (b|c)* tries 'c' -> matches 'c' ✓
    Step 4: (b|c)* tries 'd' -> 'd' is not b or c
            Exit (b|c)* loop
    Step 5: 'd' matches 'd' ✓ -> MATCH FOUND

    The engine tries options and BACKTRACKS when a path fails.
```

### NFA vs DFA

```
NFA (Python, Perl, Java, .NET):
    + Supports backreferences, lookaround, lazy quantifiers
    - Can have exponential worst-case performance
    - Uses backtracking

DFA (grep, awk, most database engines):
    + Guaranteed linear time O(n)
    + No backtracking issues
    - No backreferences or lookaround support
```

---

## 2. Catastrophic Backtracking

The most dangerous regex pitfall. Certain patterns cause **exponential** time complexity:

### The Classic Example

```python
import re
import time

# DANGEROUS pattern: nested quantifiers
pattern = r'^(a+)+$'

# Short input: fast
start = time.time()
re.search(pattern, "aaaaaaaaaaaaaaa!")  # 15 a's + !
elapsed = time.time() - start
print(f"15 chars: {elapsed:.4f}s")

# DON'T TRY THIS with long input:
# re.search(pattern, "a" * 30 + "!")  # Would take MINUTES
# re.search(pattern, "a" * 40 + "!")  # Would take HOURS
```

```
Why (a+)+$ causes catastrophic backtracking:

    Input: "aaaa!" (4 a's followed by !)

    The engine tries all ways to divide "aaaa" among the groups:
    (aaaa)         -> fail at !
    (aaa)(a)       -> fail at !
    (aa)(aa)       -> fail at !
    (aa)(a)(a)     -> fail at !
    (a)(aaa)       -> fail at !
    (a)(aa)(a)     -> fail at !
    (a)(a)(aa)     -> fail at !
    (a)(a)(a)(a)   -> fail at !

    For n characters: 2^(n-1) combinations!

    n=4:    8 combinations
    n=10:   512 combinations
    n=20:   524,288 combinations
    n=30:   536,870,912 combinations  (takes minutes)
    n=40:   549,755,813,888 combinations (takes hours)
```

### More Dangerous Patterns

```
DANGEROUS (nested quantifiers with overlap):
    (a+)+          (a*)*          (a+)*
    (a|b)+         (\w+\s*)*     (.*)*

    These patterns allow the same characters to be matched
    by multiple iterations of the outer quantifier.

SAFE alternatives:
    a+             a*             a+
    [ab]+          (?:\w+\s*)*    .*

    Key: eliminate the redundant nesting.
```

---

## 3. Identifying Dangerous Patterns

### Rule of Thumb

A pattern is dangerous when:
1. There's a **quantifier** inside another **quantifier**
2. The inner and outer quantifiers can match the **same characters**
3. The overall match can **fail** (triggering exhaustive backtracking)

```
Pattern Analysis:

    (\w+)+     DANGEROUS: \w+ inside ()+ -- both match word chars
    (\d+\w+)+  DANGEROUS: overlapping -- \d is a subset of \w
    (\d+[,-])+  SAFE: \d+ and [,-] match different characters
    (\w+\s)+   SAFE: \w+ and \s match different characters
    [^"]*      SAFE: no nesting, single quantifier
```

### Testing for Vulnerability

```python
import re
import time

def test_pattern_safety(pattern, char='a', max_len=25):
    """Test if a pattern is vulnerable to catastrophic backtracking."""
    print(f"Testing pattern: {pattern}")
    for n in range(5, max_len + 1, 5):
        text = char * n + '!'  # Input that will FAIL to match
        start = time.time()
        re.search(pattern, text)
        elapsed = time.time() - start
        print(f"  n={n:3d}: {elapsed:.4f}s")
        if elapsed > 1.0:
            print("  WARNING: Pattern may be vulnerable!")
            break
    print()

# Safe pattern
test_pattern_safety(r'^a+$')

# Dangerous pattern (be careful with large n)
test_pattern_safety(r'^(a+)+$', max_len=25)
```

---

## 4. Fixing Dangerous Patterns

### Strategy 1: Remove Redundant Quantifiers

```python
# BEFORE (dangerous):
r'(a+)+'       # Nested quantifiers on same character

# AFTER (safe):
r'a+'           # Equivalent but no backtracking risk
```

### Strategy 2: Use Atomic Groups (Conceptual)

Python's `re` module doesn't support atomic groups `(?>...)`, but you can simulate them:

```python
import re

# The regex module (third-party) supports atomic groups:
# import regex
# pattern = regex.compile(r'(?>a+)b')  # Atomic: don't backtrack into a+

# In standard re, restructure the pattern instead:
# Instead of: (a+)+$
# Use:        a+$

# Instead of: (\w+\s*)*
# Use:        (?:\w+\s)*  or  [\w\s]*
```

### Strategy 3: Use Negated Character Classes

```python
import re

# SLOW: greedy + backtracking
html = '<div class="name">value</div>'
# r'<.*>'  -- .* matches everything, then backtracks

# FAST: negated character class (no backtracking needed)
# r'<[^>]*>'  -- [^>]* stops at first >
tags = re.findall(r'<[^>]*>', html)
print(tags)  # ['<div class="name">', '</div>']
```

### Strategy 4: Be Specific

```python
import re

# SLOW: vague pattern
pattern_slow = r'".*"'  # .* matches everything then backtracks

# FAST: specific pattern
pattern_fast = r'"[^"]*"'  # [^"]* can't overshoot the closing quote

# The fast pattern never needs to backtrack because [^"]
# will naturally stop at the first " character.
```

---

## 5. `re.compile()` Performance

### When to Compile

```python
import re
import time

pattern_str = r'\b\w{3,}\b'

# Method 1: Compile once
compiled = re.compile(pattern_str)

# Method 2: Use string directly
# re.findall(pattern_str, text)  # Compiles internally each time

# Benchmark
text = "The quick brown fox jumps over the lazy dog " * 1000

# Compiled pattern
start = time.time()
for _ in range(1000):
    compiled.findall(text)
compiled_time = time.time() - start

# String pattern (Python caches recently used patterns)
start = time.time()
for _ in range(1000):
    re.findall(pattern_str, text)
string_time = time.time() - start

print(f"Compiled: {compiled_time:.3f}s")
print(f"String:   {string_time:.3f}s")
# Compiled is slightly faster due to avoided cache lookup
```

### Python's Internal Cache

```
Python caches the most recently used regex patterns (up to 512).
So re.findall(r'\d+', text) doesn't recompile every time.

But re.compile() is still recommended when:
- You use the pattern in a loop
- You want a descriptive variable name
- The pattern is complex
- Performance is critical
```

---

## 6. String Methods vs Regex

String methods are faster for simple operations:

```python
import re
import time

text = "Hello World Hello World " * 10000

# Simple substring check
start = time.time()
for _ in range(1000):
    "Hello" in text
str_time = time.time() - start

start = time.time()
for _ in range(1000):
    re.search(r'Hello', text)
re_time = time.time() - start

print(f"'in' operator: {str_time:.4f}s")
print(f"re.search():   {re_time:.4f}s")
# 'in' is typically 5-10x faster
```

### Decision Guide

```
Use string methods when:
    ✓ Fixed text (no patterns)
    ✓ Simple contains/starts/ends checks
    ✓ Split on single delimiter
    ✓ Replace exact text

Use regex when:
    ✓ Pattern matching (variable text)
    ✓ Complex splitting rules
    ✓ Capture groups needed
    ✓ Lookaround assertions
    ✓ Case-insensitive with patterns
    ✓ Multiple alternative matches
```

---

## 7. Common Pitfalls

### Pitfall 1: Forgetting `re.escape()`

```python
import re

# User input might contain metacharacters
user_search = "price (USD)"

# WRONG: parentheses are interpreted as groups
try:
    re.search(user_search, "The price (USD) is $10")
except re.error:
    print("Pattern error!")

# RIGHT: escape user input
safe_pattern = re.escape(user_search)
match = re.search(safe_pattern, "The price (USD) is $10")
print(match.group())  # "price (USD)"
```

### Pitfall 2: `.` Doesn't Match Newlines

```python
import re

text = "line1\nline2\nline3"

# WRONG: . doesn't match \n
print(re.search(r'line1.*line3', text))  # None

# RIGHT: use re.DOTALL
print(re.search(r'line1.*line3', text, re.DOTALL).group())
# "line1\nline2\nline3"

# Or use [\s\S]
print(re.search(r'line1[\s\S]*line3', text).group())
```

### Pitfall 3: `re.match()` vs `re.search()`

```python
import re

text = "Error: file not found"

# WRONG: match() only checks the start
print(re.match(r'file', text))    # None!

# RIGHT: search() checks anywhere
print(re.search(r'file', text))   # Match!
```

### Pitfall 4: Greedy vs Lazy in the Wrong Context

```python
import re

# Extracting HTML content
html = "<p>First</p><p>Second</p>"

# WRONG: greedy matches too much
print(re.search(r'<p>(.+)</p>', html).group(1))
# "First</p><p>Second"

# RIGHT: lazy matches correctly
print(re.search(r'<p>(.+?)</p>', html).group(1))
# "First"

# BEST: negated character class
print(re.search(r'<p>([^<]+)</p>', html).group(1))
# "First"
```

### Pitfall 5: Unintended Capture Groups

```python
import re

# findall with groups returns captured groups, not full matches
text = "color or colour"

# Unexpected: returns group contents
print(re.findall(r'colo(u?)r', text))   # ['', 'u']

# Fixed: use non-capturing group
print(re.findall(r'colou?r', text))      # ['color', 'colour']
```

---

## 8. Optimization Techniques

### Technique 1: Anchor Your Patterns

```python
import re

# Without anchor: engine tries at every position
re.search(r'\d{4}-\d{2}-\d{2}', "no dates here at all")
# Tries at position 0, 1, 2, ... (wastes time)

# With anchor: engine fails fast
re.search(r'^\d{4}-\d{2}-\d{2}', "no dates here at all")
# Tries only at position 0, fails immediately
```

### Technique 2: Put Likely Matches First in Alternation

```python
import re

log = "INFO: message"  # Most common case

# If 90% of logs are INFO, put it first
pattern_optimized = r'INFO|ERROR|WARN|DEBUG'
pattern_suboptimal = r'DEBUG|WARN|ERROR|INFO'

# The engine tries alternatives left to right
# Putting common cases first reduces average tries
```

### Technique 3: Use Non-Capturing Groups

```python
import re

# Capturing groups have overhead (saving state)
re.findall(r'(abc)+(def)+', text)       # Captures (slower)
re.findall(r'(?:abc)+(?:def)+', text)   # Non-capturing (faster)
```

### Technique 4: Avoid Excessive Backtracking Points

```python
import re

# More specific = fewer backtracking points
# BAD:  r'.*\d+.*@.*\..*'
# GOOD: r'[\w.]+\d+[\w.]*@[\w.]+\.\w+'
```

---

## 9. Profiling Regex Performance

```python
import re
import time

def benchmark_pattern(pattern, text, n=10000):
    """Benchmark a regex pattern."""
    compiled = re.compile(pattern)
    start = time.time()
    for _ in range(n):
        compiled.search(text)
    elapsed = time.time() - start
    print(f"Pattern: {pattern:30s} Time: {elapsed:.4f}s ({n} iterations)")
    return elapsed

text = "The quick brown fox jumps over the lazy dog, and the dog barks."

# Compare different approaches for the same task
print("Finding words starting with 'th':")
benchmark_pattern(r'\bth\w+', text)
benchmark_pattern(r'th[a-z]+', text)
benchmark_pattern(r'(?i)\bth\w+', text)

print("\nFinding quoted text:")
quoted_text = 'She said "hello" and "world" and "python"'
benchmark_pattern(r'".*?"', quoted_text)     # Lazy
benchmark_pattern(r'"[^"]*"', quoted_text)   # Negated class (usually faster)
```

---

## 10. Best Practices Summary

```
DO:
    ✓ Use raw strings (r"")
    ✓ Compile patterns used repeatedly
    ✓ Use non-capturing groups (?:...) when you don't need captures
    ✓ Use character classes [^X] instead of lazy .*?
    ✓ Anchor patterns when possible (^, $, \b)
    ✓ Use re.VERBOSE for complex patterns
    ✓ Test with both matching and non-matching inputs
    ✓ Use re.escape() for user-provided text
    ✓ Put common alternatives first in |

DON'T:
    ✗ Nest quantifiers on overlapping patterns: (a+)+
    ✗ Use regex for parsing HTML, JSON, or XML
    ✗ Use regex when string methods suffice
    ✗ Forget to test with edge cases (empty strings, very long inputs)
    ✗ Use .* when a more specific pattern exists
    ✗ Ignore backtracking behavior
```

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| Catastrophic backtracking | Nested quantifiers on overlapping chars = exponential time |
| Prevention | Use negated classes `[^X]*` instead of lazy `.*?` |
| `re.compile()` | Marginal speedup; mainly for readability and reuse |
| String methods | Faster for simple fixed-text operations |
| Anchoring | `^`, `$`, `\b` help the engine fail fast |
| Testing | Always test with failing inputs (worst case for NFA) |

---

## Next Lesson

In [12_Real_World_Applications](./12_Real_World_Applications.md), we'll apply everything we've learned to real-world text processing tasks.

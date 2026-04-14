# Print Debugging

**Previous**: [Reading Error Messages](./01_Reading_Error_Messages.md) | **Next**: [Using a Debugger](./03_Using_a_Debugger.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `print()` strategically to trace program execution flow
2. Apply f-string formatting to produce clear, informative debug output
3. Use `repr()` vs `str()` to distinguish between similar-looking values
4. Add contextual labels to print statements for quick identification
5. Print data structures, types, and intermediate values effectively
6. Know when print debugging is appropriate and when to switch to other tools
7. Clean up debug prints before committing code
8. Use `sys.stderr` to separate debug output from program output

---

Print debugging is the oldest and most universal debugging technique. Despite the availability of sophisticated debuggers and logging frameworks, strategic use of `print()` remains one of the fastest ways to understand what your code is actually doing. The key word is *strategic* -- scattering random print statements is ineffective. This lesson teaches you how to print the right information at the right places.

> **Philosophy:** Print debugging isn't primitive -- it's pragmatic. Even experienced developers use it daily. The difference between a beginner and an expert is *what* they choose to print and *where* they place the statements.

---

## 1. The Art of Strategic Printing

### 1.1 Don't Print Everything -- Print at Decision Points

Bad approach (too much noise):
```python
def process_orders(orders):
    print(orders)           # Wall of data
    results = []
    for order in orders:
        print(order)        # Still noisy
        total = 0
        for item in order["items"]:
            print(item)     # Even more noise
            total += item["price"] * item["quantity"]
            print(total)    # Constant output
        results.append(total)
    print(results)
    return results
```

Strategic approach (print at decision points):
```python
def process_orders(orders):
    print(f"[process_orders] Received {len(orders)} orders")
    results = []
    for i, order in enumerate(orders):
        total = 0
        for item in order["items"]:
            total += item["price"] * item["quantity"]
        print(f"  Order {i}: total={total}, items={len(order['items'])}")
        results.append(total)
    print(f"[process_orders] Results: {results}")
    return results
```

### 1.2 Label Everything

Never print a bare value. Always include context:

```python
# BAD -- which number is this?
print(x)
print(len(data))
print(result)

# GOOD -- instantly identifiable
print(f"x = {x}")
print(f"len(data) = {len(data)}")
print(f"result = {result}")
```

Python 3.8+ has a shortcut using `=` in f-strings:

```python
x = 42
data = [1, 2, 3]
print(f"{x = }")          # x = 42
print(f"{len(data) = }")  # len(data) = 3
print(f"{x * 2 = }")      # x * 2 = 84
```

---

## 2. Essential Print Patterns

### 2.1 Function Entry/Exit

```python
def calculate_tax(income, deductions):
    print(f">>> calculate_tax(income={income}, deductions={deductions})")
    taxable = income - deductions
    if taxable <= 0:
        print(f"<<< calculate_tax -> 0 (no taxable income)")
        return 0
    rate = 0.3 if taxable > 50000 else 0.2
    tax = taxable * rate
    print(f"<<< calculate_tax -> {tax} (rate={rate}, taxable={taxable})")
    return tax
```

### 2.2 Loop Iteration Tracking

```python
def find_duplicates(items):
    seen = set()
    duplicates = []
    for i, item in enumerate(items):
        if item in seen:
            print(f"  [dup] index={i}, item={item!r}")
            duplicates.append(item)
        seen.add(item)
    print(f"[find_duplicates] {len(duplicates)} duplicates in {len(items)} items")
    return duplicates
```

For long loops, print every Nth iteration:

```python
for i, record in enumerate(records):
    if i % 1000 == 0:
        print(f"  Processing record {i}/{len(records)}...")
    process(record)
```

### 2.3 Conditional Branch Tracking

```python
def classify_score(score):
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    else:
        grade = "F"
    print(f"[classify] score={score} -> grade={grade!r}")
    return grade
```

### 2.4 Data Flow Tracking

When a value passes through several transformations:

```python
def clean_username(raw_input):
    print(f"[clean] step 0 (raw):     {raw_input!r}")
    stripped = raw_input.strip()
    print(f"[clean] step 1 (strip):   {stripped!r}")
    lowered = stripped.lower()
    print(f"[clean] step 2 (lower):   {lowered!r}")
    cleaned = "".join(c for c in lowered if c.isalnum() or c == "_")
    print(f"[clean] step 3 (filter):  {cleaned!r}")
    return cleaned
```

---

## 3. `repr()` vs `str()`: See What's Really There

`str()` produces human-readable output. `repr()` shows the precise representation including type information. For debugging, **always use `repr()`**.

```python
a = "hello"
b = "hello "     # trailing space
c = "hello\t"    # trailing tab
d = ""            # empty string
e = None

# str() hides differences
print(str(a))    # hello
print(str(b))    # hello    (can you see the space?)
print(str(c))    # hello    (looks similar!)
print(str(d))    #          (blank -- is it empty or None?)
print(str(e))    # None     (string "None" or actual None?)

# repr() reveals truth
print(repr(a))   # 'hello'
print(repr(b))   # 'hello '   (trailing space visible!)
print(repr(c))   # 'hello\t'  (tab character visible!)
print(repr(d))   # ''         (clearly empty)
print(repr(e))   # None       (clearly None, not a string)
```

In f-strings, use `!r` for repr:

```python
value = "hello "
print(f"value = {value!r}")   # value = 'hello '
```

---

## 4. Printing Types and Structure

### 4.1 Type Checking

```python
def debug_value(name, value):
    print(f"{name}: value={value!r}, type={type(value).__name__}")

debug_value("count", "5")     # count: value='5', type=str     ← Bug! It's a string
debug_value("count", 5)       # count: value=5, type=int       ← Correct
debug_value("items", None)    # items: value=None, type=NoneType
```

### 4.2 Collection Contents

```python
import pprint

# For small collections, f-string is fine
data = {"name": "Alice", "age": 30}
print(f"data = {data}")

# For large/nested structures, use pprint
large_data = {
    "users": [
        {"name": "Alice", "scores": [90, 85, 92]},
        {"name": "Bob", "scores": [78, 88, 95]},
    ],
    "metadata": {"version": 2, "count": 2},
}
pprint.pprint(large_data, width=60)
```

### 4.3 Object Attributes

```python
# See all attributes of an object
print(dir(obj))

# See instance variables
print(vars(obj))

# Focused inspection
print(f"obj.name={obj.name!r}, obj.status={obj.status!r}")
```

---

## 5. Separating Debug Output

### 5.1 Using stderr

Debug output and program output should not be mixed:

```python
import sys

def process(data):
    print(f"[DEBUG] Processing: {data!r}", file=sys.stderr)
    result = data.upper()
    print(result)  # This is the actual program output
    return result
```

```bash
# Now you can separate them:
python script.py > output.txt  # Only program output goes to file
# Debug messages still appear on screen (stderr)
```

### 5.2 Using a Debug Flag

```python
DEBUG = True  # Set to False before committing

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def calculate(x, y):
    debug_print(f"calculate({x}, {y})")
    result = x + y
    debug_print(f"result = {result}")
    return result
```

### 5.3 Using Environment Variables

```python
import os

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs, file=__import__("sys").stderr)
```

```bash
DEBUG=1 python script.py    # Debug output enabled
python script.py            # Debug output disabled
```

---

## 6. The `icecream` Library

The `icecream` library (third-party) automates much of what we've covered:

```python
from icecream import ic

x = 42
ic(x)           # ic| x: 42
ic(len([1,2]))  # ic| len([1, 2]): 2

def add(a, b):
    ic(a, b)       # ic| a: 3, b: 4
    result = a + b
    ic(result)     # ic| result: 7
    return result
```

Install with: `pip install icecream`

---

## 7. Common Mistakes with Print Debugging

### 7.1 Forgetting to Remove Debug Prints

```python
# BAD: Debug prints left in production code
def get_user(user_id):
    print(f"LOOKING UP USER {user_id}")  # Oops, left in
    user = db.query(user_id)
    print(f"FOUND: {user}")              # Oops, left in
    return user
```

**Prevention**: Use `grep -rn "print(" .` before committing, or use a linter rule.

### 7.2 Print Inside Tight Loops

```python
# BAD: 1 million print calls will destroy performance
for i in range(1_000_000):
    print(f"i = {i}")  # Extremely slow
    data[i] = process(i)

# BETTER: Sample
for i in range(1_000_000):
    if i % 100_000 == 0:
        print(f"Progress: {i:,} / 1,000,000")
    data[i] = process(i)
```

### 7.3 Printing Without repr

```python
# Misleading
value = ""
print(f"value = {value}")   # value =      (is it empty? None? whitespace?)

# Clear
print(f"value = {value!r}") # value = ''   (clearly empty string)
```

---

## 8. When to Stop Print Debugging

Print debugging works best for:
- Quick checks during development
- Understanding data flow in unfamiliar code
- Verifying assumptions about values

Switch to a **debugger** (next lesson) when:
- You need to inspect many variables at a specific point
- The bug requires stepping through code line by line
- You need to modify values during execution
- The control flow is complex (many branches, recursion)

Switch to **logging** (Lesson 6) when:
- You need permanent, structured diagnostic output
- You need different verbosity levels
- The application runs in production
- You need timestamps, source locations, or structured data

---

## Summary

- Label every print statement with context -- never print bare values
- Use `repr()` (or `!r` in f-strings) to see exact values including whitespace and types
- Print at decision points: function entry/exit, branches, loop milestones
- Separate debug output from program output using `stderr` or flags
- Always clean up debug prints before committing
- Print debugging is effective but has limits -- know when to use other tools

---

## Exercises

1. Add strategic print statements to a buggy function to find the error
2. Use `!r` formatting to identify a whitespace bug
3. Implement a `debug_print()` function with an enable/disable flag
4. Trace data flow through a pipeline using labeled prints

**Previous**: [Reading Error Messages](./01_Reading_Error_Messages.md) | **Next**: [Using a Debugger](./03_Using_a_Debugger.md)

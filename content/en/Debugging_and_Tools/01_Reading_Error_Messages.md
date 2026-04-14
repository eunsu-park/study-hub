# Reading Error Messages

**Next**: [Print Debugging](./02_Print_Debugging.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Read a Python traceback from bottom to top and identify the error type and message
2. Locate the exact file, line number, and function where an error occurred
3. Distinguish between syntax errors, runtime errors, and logical errors
4. Recognize the 10 most common Python exception types and their causes
5. Interpret chained exceptions and understand the `__cause__` chain
6. Use the traceback structure to navigate to the source of a bug
7. Read error messages from external libraries and frameworks

---

Every developer's debugging journey begins with a single skill: reading error messages. When Python encounters a problem it cannot handle, it produces a **traceback** -- a detailed report showing exactly where things went wrong and why. Beginners often panic when they see a wall of red text, but a traceback is actually your best friend. It tells you the file, the line, the function, and the type of error. Learning to read this information calmly and systematically is the single most important debugging skill you will ever develop.

> **Key Insight:** Always read tracebacks from the **bottom up**. The last line tells you *what* went wrong. The lines above tell you *where*.

---

## 1. Anatomy of a Traceback

When Python raises an exception, it prints a traceback. Here is a simple example:

```python
# file: calculator.py
def divide(a, b):
    return a / b

def calculate():
    result = divide(10, 0)
    return result

calculate()
```

Running this produces:

```
Traceback (most recent call last):
  File "calculator.py", line 8, in <module>
    calculate()
  File "calculator.py", line 5, in calculate
    result = divide(10, 0)
  File "calculator.py", line 2, in divide
    return a / b
ZeroDivisionError: division by zero
```

### Reading Order: Bottom to Top

```
Step 1 (BOTTOM) → ZeroDivisionError: division by zero
                   "What went wrong: divided by zero"

Step 2           → File "calculator.py", line 2, in divide
                   "Where it crashed: line 2, inside divide()"

Step 3           → File "calculator.py", line 5, in calculate
                   "Who called divide(): calculate() on line 5"

Step 4 (TOP)     → File "calculator.py", line 8, in <module>
                   "Entry point: line 8 at module level"
```

### Traceback Structure Diagram

```
┌─────────────────────────────────────────────┐
│  Traceback (most recent call last):         │  ← Header
├─────────────────────────────────────────────┤
│  File "X", line N, in <module>              │  ← Oldest call
│    code_line_here                           │     (entry point)
│  File "X", line N, in func_a               │  ← Middle call
│    code_line_here                           │
│  File "X", line N, in func_b               │  ← Newest call
│    code_line_here                           │     (crash site)
├─────────────────────────────────────────────┤
│  ExceptionType: error message               │  ← Error summary
└─────────────────────────────────────────────┘
         ↑ START READING HERE ↑
```

---

## 2. Three Categories of Errors

### 2.1 Syntax Errors

Syntax errors occur **before** your code runs. Python cannot parse the code at all.

```python
# Missing colon
def greet(name)
    print(f"Hello, {name}")
```

```
  File "greet.py", line 1
    def greet(name)
                   ^
SyntaxError: expected ':'
```

Key characteristics:
- No "Traceback (most recent call last):" header
- The `^` caret points to where Python got confused
- The code never executes -- this is a **parse-time** error

Common syntax errors:

| Error | Example | Fix |
|-------|---------|-----|
| Missing colon | `if x == 5` | `if x == 5:` |
| Unmatched parenthesis | `print("hello"` | `print("hello")` |
| Invalid assignment | `5 = x` | `x = 5` |
| Missing quotes | `print(hello)` | `print("hello")` |
| Indentation | Mixed tabs/spaces | Use consistent 4-space indent |

### 2.2 Runtime Errors (Exceptions)

Runtime errors occur **during** execution. The syntax is valid, but an operation fails.

```python
numbers = [1, 2, 3]
print(numbers[10])  # IndexError: list index out of range
```

These produce the full traceback with call stack.

### 2.3 Logical Errors

Logical errors are the hardest: the code runs without any error, but produces the **wrong result**.

```python
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) + 1  # Bug: the +1 shouldn't be here

print(average([10, 20, 30]))  # Prints 21.0 instead of 20.0
```

Python cannot detect logical errors -- you must find them through testing, code review, or debugging.

---

## 3. The 10 Most Common Python Exceptions

### 3.1 NameError

```python
print(user_name)
# NameError: name 'user_name' is not defined
```

**Cause**: Using a variable that hasn't been defined. Check for typos.

### 3.2 TypeError

```python
"age: " + 25
# TypeError: can only concatenate str (not "int") to str
```

**Cause**: Performing an operation on incompatible types.

```python
len(42)
# TypeError: object of type 'int' has no len()
```

### 3.3 ValueError

```python
int("hello")
# ValueError: invalid literal for int() with base 10: 'hello'
```

**Cause**: Right type, wrong value. The function received a value it cannot process.

### 3.4 IndexError

```python
items = [1, 2, 3]
print(items[5])
# IndexError: list index out of range
```

**Cause**: Accessing a list/tuple index that does not exist.

### 3.5 KeyError

```python
data = {"name": "Alice"}
print(data["age"])
# KeyError: 'age'
```

**Cause**: Accessing a dictionary key that does not exist. Use `.get()` for safe access.

### 3.6 AttributeError

```python
x = 42
x.append(1)
# AttributeError: 'int' object has no attribute 'append'
```

**Cause**: Calling a method or accessing an attribute that doesn't exist on the object.

### 3.7 FileNotFoundError

```python
with open("nonexistent.txt") as f:
    data = f.read()
# FileNotFoundError: [Errno 2] No such file or directory: 'nonexistent.txt'
```

**Cause**: Trying to open a file that doesn't exist. Check path and working directory.

### 3.8 ZeroDivisionError

```python
100 / 0
# ZeroDivisionError: division by zero
```

**Cause**: Dividing by zero. Always validate denominators.

### 3.9 ImportError / ModuleNotFoundError

```python
import nonexistent_module
# ModuleNotFoundError: No module named 'nonexistent_module'

from os import nonexistent_func
# ImportError: cannot import name 'nonexistent_func' from 'os'
```

**Cause**: The module doesn't exist or isn't installed, or the name doesn't exist in the module.

### 3.10 IndentationError

```python
def greet():
print("hello")
# IndentationError: expected an indented block after function definition
```

**Cause**: Incorrect indentation. This is technically a subclass of `SyntaxError`.

---

## 4. Multi-File Tracebacks

Real projects spread code across many files. Tracebacks reflect this:

```
Traceback (most recent call last):
  File "main.py", line 12, in <module>
    app.run()
  File "/project/app.py", line 45, in run
    result = self.processor.process(data)
  File "/project/processor.py", line 23, in process
    validated = self.validator.check(item)
  File "/project/validator.py", line 8, in check
    return int(item["count"])
ValueError: invalid literal for int() with base 10: 'three'
```

Reading strategy:
1. **Bottom**: `ValueError` -- someone tried to convert `'three'` to an integer
2. **Crash site**: `validator.py`, line 8 -- `int(item["count"])` is the culprit
3. **Caller**: `processor.py`, line 23 -- this is where `item` came from
4. **Trace upward**: Follow the data to find where `'three'` was introduced

---

## 5. Chained Exceptions

Python 3 supports exception chaining with `from`:

```python
def load_config(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError as e:
        raise RuntimeError(f"Config missing: {path}") from e
```

```
Traceback (most recent call last):
  File "config.py", line 3, in load_config
    with open(path) as f:
FileNotFoundError: [Errno 2] No such file or directory: 'app.conf'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "main.py", line 5, in <module>
    config = load_config("app.conf")
  File "config.py", line 5, in load_config
    raise RuntimeError(f"Config missing: {path}") from e
RuntimeError: Config missing: app.conf
```

Read strategy:
1. Start at the **bottom traceback** -- this is the exception that actually propagated
2. Look at the **top traceback** to see the original root cause
3. The phrase "The above exception was the direct cause" links them

---

## 6. Understanding Library Error Messages

When errors come from third-party libraries, the traceback can be long. Focus on:

1. **Your code frames**: Look for file paths in *your* project, not in `site-packages/`
2. **The error message**: The bottom line is still the most important
3. **The boundary**: Find where your code calls the library

```
Traceback (most recent call last):
  File "app.py", line 15, in handle_request        ← YOUR CODE
    response = requests.get(url, timeout=5)
  File ".../site-packages/requests/api.py", ...     ← LIBRARY
    return request('GET', url, **kwargs)
  File ".../site-packages/requests/api.py", ...     ← LIBRARY
    ...
  File ".../site-packages/urllib3/...", ...          ← LIBRARY
    raise ConnectTimeoutError(...)
requests.exceptions.ConnectTimeout: ...              ← ERROR
```

**Tip**: You can usually ignore library-internal frames. Focus on what *you* passed to the library and what it complained about.

---

## 7. Practical Tips for Error Messages

### 7.1 Copy-Paste the Error

When searching for help, copy the **last line** of the traceback (the `ExceptionType: message` part) into a search engine. Remove any project-specific paths or variable values first.

### 7.2 Read the Full Message

Error messages in Python are usually descriptive. Don't just read the exception type -- the message after the colon contains crucial details:

```python
# Bad: "I got a TypeError"
# Good: "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
#        → This tells me I'm trying to add an int and a str
```

### 7.3 Check the Line Number

The traceback points to exact lines. Open the file and look at that line. The bug might be on that line or in the data flowing into it.

### 7.4 Watch for "Did You Mean?"

Python 3.10+ includes helpful suggestions:

```python
import colection
# ModuleNotFoundError: No module named 'colection'. Did you mean: 'collection'?
```

```python
name = "Alice"
print(nme)
# NameError: name 'nme' is not defined. Did you mean: 'name'?
```

### 7.5 Use `python -v` for Import Debugging

```bash
python -v script.py  # Shows every import attempt
```

---

## 8. Building Your Error Vocabulary

Keep a mental (or written) map of error patterns:

| When you see... | Think... |
|----------------|----------|
| `NameError` | Typo? Variable not defined yet? Wrong scope? |
| `TypeError: ... NoneType` | A function returned None unexpectedly |
| `TypeError: ... argument` | Wrong number or type of arguments to a function |
| `KeyError` | Dictionary doesn't have that key. Print the dict. |
| `IndexError` | List is shorter than expected. Print its length. |
| `AttributeError: 'NoneType'` | Something is None that shouldn't be. Trace back. |
| `RecursionError` | Missing base case or infinite loop in recursion |
| `UnicodeDecodeError` | File encoding mismatch. Try `encoding='utf-8'` |

---

## Summary

- Tracebacks are read **bottom-to-top**: error type first, then trace the call stack upward
- Python errors fall into three categories: syntax, runtime, and logical
- The 10 most common exceptions cover the vast majority of beginner errors
- Multi-file tracebacks show the full call chain -- focus on *your* code frames
- Chained exceptions use `from` to link cause and effect
- Always read the full error message, not just the exception type
- Search engines are more effective when you paste the exact error line

---

## Exercises

1. Given a traceback, identify the error type, the file, and the line number
2. Classify a set of errors as syntax, runtime, or logical
3. Fix code snippets based on their error messages
4. Read a multi-file traceback and locate the root cause

**Next**: [Print Debugging](./02_Print_Debugging.md)

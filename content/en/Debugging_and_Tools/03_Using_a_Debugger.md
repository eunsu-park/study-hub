# Using a Debugger

**Previous**: [Print Debugging](./02_Print_Debugging.md) | **Next**: [Common Bug Patterns](./04_Common_Bug_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Start the Python debugger using `pdb`, `breakpoint()`, and command-line invocation
2. Navigate code using step, next, continue, and return commands
3. Inspect variables, evaluate expressions, and modify state during execution
4. Set conditional breakpoints to stop only when specific conditions are met
5. Use `pdb.post_mortem()` to debug after an exception has occurred
6. Navigate the call stack with `up`, `down`, and `where` commands
7. Understand the basics of IDE debuggers (VS Code, PyCharm)

---

A debugger lets you **pause** your program at any point, **inspect** every variable, **step** through code line by line, and **modify** values on the fly. While print debugging is quick, a debugger gives you a full X-ray of your program's state. Learning to use `pdb` (Python's built-in debugger) is a fundamental skill that will save you hours of frustration.

> **Analogy:** Print debugging is like adding windows to a building to peek inside. A debugger is like being able to walk through every room, open every drawer, and rearrange the furniture -- all while time is frozen.

---

## 1. Starting the Debugger

### 1.1 Using `breakpoint()` (Python 3.7+, Recommended)

```python
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    breakpoint()  # Execution pauses here
    return total / len(numbers)

result = calculate_average([10, 20, 30])
```

When Python hits `breakpoint()`, it drops into the interactive `pdb` prompt:

```
> /path/to/script.py(5)calculate_average()
-> return total / len(numbers)
(Pdb) 
```

### 1.2 Using `import pdb; pdb.set_trace()`

The pre-3.7 equivalent (still works):

```python
import pdb; pdb.set_trace()  # Same as breakpoint()
```

### 1.3 Running from Command Line

```bash
# Start the script under debugger control from the beginning
python -m pdb script.py

# The debugger will stop at the first line
> /path/to/script.py(1)<module>()
-> import sys
(Pdb) 
```

### 1.4 Disabling breakpoint()

```bash
# Disable all breakpoints (useful in production)
PYTHONBREAKPOINT=0 python script.py

# Use a different debugger
PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

---

## 2. Essential pdb Commands

### Command Reference

```
┌──────────────────────────────────────────────────────────────┐
│  Navigation                                                  │
├──────────────────────────────────────────────────────────────┤
│  n (next)       Execute current line, step OVER function     │
│  s (step)       Execute current line, step INTO function     │
│  c (continue)   Resume execution until next breakpoint       │
│  r (return)     Execute until current function returns       │
│  unt (until) N  Execute until line N is reached              │
├──────────────────────────────────────────────────────────────┤
│  Inspection                                                  │
├──────────────────────────────────────────────────────────────┤
│  p expr         Print the value of an expression             │
│  pp expr        Pretty-print the value of an expression      │
│  l (list)       Show source code around current line         │
│  ll (longlist)  Show the full source of current function     │
│  a (args)       Show arguments of current function           │
│  w (where)      Show the call stack (traceback)              │
│  whatis expr    Show the type of an expression               │
├──────────────────────────────────────────────────────────────┤
│  Breakpoints                                                 │
├──────────────────────────────────────────────────────────────┤
│  b N            Set breakpoint at line N                     │
│  b func         Set breakpoint at function entry             │
│  b N, cond      Conditional breakpoint (stops if cond=True)  │
│  cl N           Clear breakpoint number N                    │
│  bl             List all breakpoints                         │
│  disable N      Disable breakpoint N (keep but don't stop)   │
│  enable N       Re-enable breakpoint N                       │
├──────────────────────────────────────────────────────────────┤
│  Stack Navigation                                            │
├──────────────────────────────────────────────────────────────┤
│  u (up)         Move up one frame in the call stack          │
│  d (down)       Move down one frame in the call stack        │
├──────────────────────────────────────────────────────────────┤
│  Control                                                     │
├──────────────────────────────────────────────────────────────┤
│  q (quit)       Quit the debugger and program                │
│  restart        Restart the program                          │
│  h (help)       Show help; h <command> for details           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Debugging Walkthrough

### Sample Buggy Code

```python
# buggy_stats.py
def compute_stats(data):
    """Compute mean and standard deviation."""
    n = len(data)
    mean = sum(data) / n
    
    variance = sum((x - mean) for x in data) / n  # BUG: missing ** 2
    std_dev = variance ** 0.5
    
    return {"mean": mean, "std_dev": std_dev}

result = compute_stats([2, 4, 4, 4, 5, 5, 7, 9])
print(f"Mean: {result['mean']}, Std Dev: {result['std_dev']}")
```

### Step-by-Step Debugging Session

```
$ python -m pdb buggy_stats.py
> buggy_stats.py(1)<module>()
-> def compute_stats(data):
(Pdb) b 6            # Set breakpoint at line 6 (variance calc)
Breakpoint 1 at buggy_stats.py:6
(Pdb) c               # Continue to the breakpoint
> buggy_stats.py(6)compute_stats()
-> variance = sum((x - mean) for x in data) / n
(Pdb) p mean           # Check the mean value
5.0
(Pdb) p data           # Check input data
[2, 4, 4, 4, 5, 5, 7, 9]
(Pdb) p [(x - mean) for x in data]    # Check the terms
[-3.0, -1.0, -1.0, -1.0, 0.0, 0.0, 2.0, 4.0]
(Pdb) p sum([(x - mean) for x in data])
0.0                     # Sum of deviations is always 0! 
(Pdb) # AHA! We need (x - mean)**2, not (x - mean)
(Pdb) p sum([(x - mean)**2 for x in data]) / n
4.0                     # Correct variance
(Pdb) q
```

---

## 4. Conditional Breakpoints

Stop only when a specific condition is true:

```python
def process_records(records):
    for i, record in enumerate(records):
        breakpoint()  # This would stop 1000 times!
        result = transform(record)
```

Instead, use conditional breakpoints:

```
(Pdb) b 3, i == 500           # Stop only at iteration 500
(Pdb) b 3, record["status"] == "error"  # Stop on error records
(Pdb) b 3, len(record) > 10   # Stop on oversized records
```

Or in code:

```python
def process_records(records):
    for i, record in enumerate(records):
        if record.get("status") == "error":
            breakpoint()  # Only stop on error records
        result = transform(record)
```

---

## 5. Post-Mortem Debugging

When an exception occurs, `pdb.post_mortem()` lets you inspect the state at the moment of the crash:

```python
import pdb

try:
    result = buggy_function()
except Exception:
    pdb.post_mortem()  # Opens debugger at the crash point
```

From the command line:

```bash
# Automatically enter debugger on any unhandled exception
python -m pdb script.py
# When it crashes, pdb offers: (Pdb) prompt at the crash site
```

### Using `pdb.pm()` in Interactive Python

```python
>>> import my_module
>>> my_module.buggy_function()
Traceback (most recent call last):
  ...
ValueError: invalid value
>>> import pdb; pdb.pm()   # Debug the LAST exception
> my_module.py(10)buggy_function()
-> result = int(value)
(Pdb) p value
'not_a_number'
```

---

## 6. Stack Navigation

When stopped in a deeply nested call, use `up` and `down` to move through the call stack:

```python
def level_3(x):
    breakpoint()
    return x * 2

def level_2(x):
    return level_3(x + 10)

def level_1(x):
    return level_2(x + 5)

level_1(1)
```

```
(Pdb) w                     # Show full call stack
  /path/script.py(10)<module>()
-> level_1(1)
  /path/script.py(8)level_1()
-> return level_2(x + 5)
  /path/script.py(5)level_2()
-> return level_3(x + 10)
> /path/script.py(2)level_3()    ← Current frame
-> return x * 2
(Pdb) p x                   # x in level_3
16
(Pdb) u                     # Move up to level_2
> /path/script.py(5)level_2()
(Pdb) p x                   # x in level_2
6
(Pdb) u                     # Move up to level_1
> /path/script.py(8)level_1()
(Pdb) p x                   # x in level_1
1
(Pdb) d                     # Back down to level_2
```

---

## 7. Modifying Values During Debugging

You can change variable values during a debug session:

```python
def divide(a, b):
    breakpoint()
    return a / b

divide(10, 0)
```

```
(Pdb) p b
0
(Pdb) !b = 5         # Use ! prefix to distinguish from pdb commands
(Pdb) p b
5
(Pdb) c               # Continue -- now it divides 10/5 instead of 10/0
```

**Warning**: Use `!` before assignments to avoid conflicts with pdb commands (e.g., `!n = 5` instead of `n = 5`, since `n` is the "next" command).

---

## 8. Enhanced Debuggers

### 8.1 `ipdb` -- IPython-Powered Debugger

```bash
pip install ipdb
```

```python
import ipdb; ipdb.set_trace()
# Or: PYTHONBREAKPOINT=ipdb.set_trace python script.py
```

Benefits: Tab completion, syntax highlighting, better `?` help.

### 8.2 `pdb++` (`pdbpp`)

```bash
pip install pdbpp
```

Automatically replaces `pdb` with an enhanced version:
- Syntax highlighting
- `sticky` mode (always shows surrounding code)
- Tab completion

---

## 9. IDE Debuggers

### VS Code

1. Open the Python file
2. Click the gutter (left of line numbers) to set a red breakpoint dot
3. Press `F5` to start debugging (or `Run > Start Debugging`)
4. Use the debug toolbar: Continue (`F5`), Step Over (`F10`), Step Into (`F11`)
5. Inspect variables in the "Variables" panel on the left
6. Add expressions to the "Watch" panel
7. Use the "Debug Console" to evaluate expressions

### PyCharm

1. Click the gutter to set breakpoints
2. Right-click and select "Debug" (or use the bug icon)
3. Use `F8` (Step Over), `F7` (Step Into), `F9` (Resume)
4. The "Variables" pane shows all local variables automatically
5. "Evaluate Expression" (`Alt+F8`) lets you run arbitrary code

### Common IDE Features

```
┌──────────────────────────────────────────────┐
│  Feature              Shortcut (VS Code)     │
├──────────────────────────────────────────────┤
│  Toggle breakpoint    F9 / click gutter      │
│  Start debugging      F5                     │
│  Stop debugging       Shift+F5               │
│  Step over            F10                     │
│  Step into            F11                     │
│  Step out             Shift+F11              │
│  Continue             F5                     │
│  Restart              Ctrl+Shift+F5          │
└──────────────────────────────────────────────┘
```

---

## 10. Debugging Tips and Best Practices

### 10.1 Start Wide, Narrow Down

1. Set a breakpoint before the suspected bug area
2. Use `n` to step over until you spot the wrong value
3. On the next run, set a breakpoint at the exact problem line
4. Use `s` to step into the function call

### 10.2 Use `commands` for Automated Actions

```
(Pdb) b 15                    # Breakpoint at line 15
(Pdb) commands 1              # When breakpoint 1 hits...
(com) p x, y, total           # ...print these values
(com) c                       # ...and continue
(com) end                     # End of commands
```

Now every time line 15 is reached, it prints the values and continues automatically -- like adding a print statement without modifying the code.

### 10.3 Debugging Recursive Functions

```python
def factorial(n):
    breakpoint()
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Use conditional breakpoints:
```
(Pdb) b 2, n == 1             # Only stop at base case
```

Or use `r` (return) to quickly skip through recursive calls.

---

## Summary

- `breakpoint()` is the modern way to invoke the debugger (Python 3.7+)
- `n` steps over, `s` steps into, `c` continues, `r` returns from the current function
- `p` and `pp` inspect values; `!` prefix lets you modify variables
- Conditional breakpoints avoid stopping at every iteration
- `pdb.post_mortem()` lets you debug after a crash
- `up`/`down` let you navigate the call stack
- IDE debuggers provide a visual interface for the same concepts
- Start with `pdb` to understand the fundamentals, then use IDE debuggers for convenience

---

## Exercises

1. Use `breakpoint()` to pause inside a function and inspect variables
2. Step through a loop using `n` and track how values change
3. Set a conditional breakpoint to stop only on a specific condition
4. Use post-mortem debugging to investigate a crash
5. Navigate the call stack using `up` and `down`

**Previous**: [Print Debugging](./02_Print_Debugging.md) | **Next**: [Common Bug Patterns](./04_Common_Bug_Patterns.md)

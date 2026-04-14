# Linters and Formatters

**Previous**: [Testing Basics](./07_Testing_Basics.md) | **Next**: [Type Checking](./09_Type_Checking.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between linters (bug finders) and formatters (style enforcers)
2. Use `pylint` to catch bugs, code smells, and style violations
3. Use `flake8` as a lightweight, fast linting tool
4. Use `ruff` as a modern, ultra-fast all-in-one linter
5. Use `black` to automatically format code to a consistent style
6. Configure linters with configuration files (`pyproject.toml`, `.flake8`)
7. Set up pre-commit hooks to run linters automatically before each commit
8. Interpret linter output and fix common warnings

---

Linters and formatters are automated tools that catch bugs and enforce code style **before** your code ever runs. A linter reads your source code and flags potential problems: unused variables, unreachable code, style violations, and even some logical errors. A formatter automatically rewrites your code to follow a consistent style. Together, they eliminate entire categories of bugs and style debates.

> **Prevention vs Cure:** Debugging finds bugs after they happen. Linters find bugs before they happen. The best bug is one that never reaches your test suite.

---

## 1. Linters vs Formatters

```
┌──────────────────────────────────────────────────────────┐
│                Your Source Code                           │
│                                                          │
│  ┌─────────────┐         ┌─────────────┐                │
│  │   Linter    │         │  Formatter  │                │
│  │  (analyze)  │         │  (rewrite)  │                │
│  └──────┬──────┘         └──────┬──────┘                │
│         ▼                       ▼                        │
│  "Line 15: unused      Automatically fixes               │
│   variable 'x'"       indentation, quotes,               │
│  "Line 23: possible   trailing commas, etc.              │
│   bug (== None)"                                         │
│                                                          │
│  Reports problems      Changes the file                  │
│  (you fix manually)    (automatically)                   │
└──────────────────────────────────────────────────────────┘
```

| Aspect | Linter | Formatter |
|--------|--------|-----------|
| Purpose | Find bugs and code smells | Enforce consistent style |
| Action | Reports warnings | Rewrites files |
| Examples | pylint, flake8, ruff | black, autopep8, yapf |
| Focus | Correctness + style | Style only |

---

## 2. pylint: The Comprehensive Linter

### 2.1 Installation and Basic Usage

```bash
pip install pylint
pylint my_script.py
```

### 2.2 Example

```python
# file: example.py
import os
import sys

def calculate(x, y):
    z = x + y
    result = x * y
    return result

def unused_function():
    pass

value = calculate(1, 2)
```

```bash
$ pylint example.py
************* Module example
example.py:1:0: C0114: Missing module docstring (missing-module-docstring)
example.py:1:0: W0611: Unused import os (unused-import)
example.py:2:0: W0611: Unused import sys (unused-import)
example.py:4:0: C0116: Missing function or method docstring (missing-function-docstring)
example.py:6:4: W0612: Unused variable 'z' (unused-variable)
example.py:10:0: C0116: Missing function or method docstring (missing-function-docstring)

Your code has been rated at 3.33/10
```

### 2.3 Understanding pylint Message Codes

```
C0114 → Convention (style)
W0611 → Warning (possible problem)
E0001 → Error (definite bug)
R0903 → Refactor (code smell)
F0001 → Fatal (pylint can't process the file)

Format: [TYPE][NNNN]
C = Convention    (style rules)
W = Warning       (possible bugs)
E = Error         (definite bugs)
R = Refactor      (design improvements)
F = Fatal         (processing errors)
```

### 2.4 Useful pylint Checks

| Check | What It Catches |
|-------|----------------|
| `W0611` | Unused imports |
| `W0612` | Unused variables |
| `W0613` | Unused function arguments |
| `E1101` | Accessing non-existent attribute |
| `E0602` | Undefined variable |
| `W0104` | Statement has no effect |
| `R0903` | Too few public methods (possible data class) |
| `C0301` | Line too long |
| `W0621` | Redefining name from outer scope |

### 2.5 Disabling Specific Checks

```python
# Disable for a specific line
x = unused_value  # pylint: disable=unused-variable

# Disable for a block
# pylint: disable=missing-docstring
def helper():
    pass
# pylint: enable=missing-docstring
```

In `pyproject.toml`:
```toml
[tool.pylint.messages_control]
disable = [
    "C0114",  # missing-module-docstring
    "C0116",  # missing-function-docstring
]
```

---

## 3. flake8: The Lightweight Linter

### 3.1 Installation and Usage

```bash
pip install flake8
flake8 my_script.py
```

### 3.2 Key Differences from pylint

- Faster and simpler
- Combines three tools: `pycodestyle` (style), `pyflakes` (errors), `mccabe` (complexity)
- Less opinionated -- fewer false positives
- No scoring system

### 3.3 Example Output

```bash
$ flake8 example.py
example.py:1:1: F401 'os' imported but unused
example.py:2:1: F401 'sys' imported but unused
example.py:6:5: F841 local variable 'z' is assigned to but never used
example.py:14:1: W292 no newline at end of file
```

### 3.4 Configuration (`.flake8` or `setup.cfg`)

```ini
# .flake8
[flake8]
max-line-length = 100
ignore = E203, W503
per-file-ignores =
    __init__.py: F401
exclude =
    .git,
    __pycache__,
    build,
```

### 3.5 flake8 Error Codes

```
E1xx / E2xx / E3xx / E4xx / E5xx → pycodestyle (style)
W1xx / W2xx / W3xx / W5xx / W6xx → pycodestyle (warnings)
F4xx / F8xx                       → pyflakes (logic errors)
C901                              → mccabe (complexity)
```

---

## 4. ruff: The Modern, Ultra-Fast Linter

### 4.1 Installation and Usage

```bash
pip install ruff
ruff check my_script.py      # Lint
ruff format my_script.py     # Format (like black)
```

### 4.2 Why ruff?

- Written in Rust -- **10-100x faster** than pylint or flake8
- Replaces pylint, flake8, isort, pyupgrade, and more in a single tool
- Auto-fix capabilities for many rules
- Active development and growing rule set

### 4.3 Example

```bash
$ ruff check example.py
example.py:1:8: F401 [*] `os` imported but unused
example.py:2:8: F401 [*] `sys` imported but unused
example.py:6:5: F841 Local variable `z` is assigned to but never used
Found 3 errors.
[*] 2 fixable with the `--fix` option.
```

```bash
# Auto-fix what's fixable
ruff check --fix example.py
```

### 4.4 Configuration (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]
ignore = ["E501"]  # line too long

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
```

---

## 5. black: The Code Formatter

### 5.1 Installation and Usage

```bash
pip install black
black my_script.py          # Format file
black --check my_script.py  # Check without modifying
black --diff my_script.py   # Show changes without modifying
```

### 5.2 What black Does

black is **opinionated** -- it has very few configuration options on purpose.

Before:
```python
x = {  'a':37,'b':42,
'c':927}
y = 'hello ''world'
z = 'hello '+'world'
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
```

After:
```python
x = {"a": 37, "b": 42, "c": 927}
y = "hello " "world"
z = "hello " + "world"
a = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
]
```

### 5.3 Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ["py312"]
```

### 5.4 ruff format: black-Compatible Formatting

`ruff format` is a drop-in replacement for `black` that's much faster:

```bash
ruff format my_script.py     # Same output as black, but faster
```

---

## 6. isort: Import Sorting

### 6.1 Usage

```bash
pip install isort
isort my_script.py
```

### 6.2 Before and After

Before:
```python
import os
from collections import OrderedDict
import sys
from pathlib import Path
import json
```

After:
```python
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
```

### 6.3 Compatibility with black

```toml
# pyproject.toml
[tool.isort]
profile = "black"
```

Note: `ruff` includes isort functionality built-in (`ruff check --select I`).

---

## 7. Pre-Commit Hooks

### 7.1 What Are Pre-Commit Hooks?

Pre-commit hooks run linters and formatters **automatically** before every `git commit`. If any check fails, the commit is rejected until you fix the issues.

### 7.2 Setup with `pre-commit`

```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

```bash
# Install the hooks
pre-commit install

# Now every git commit automatically runs the checks
git commit -m "Add feature"
# ruff..........................................................Passed
# ruff-format...................................................Passed
# Trim Trailing Whitespace......................................Passed
```

### 7.3 Running Manually

```bash
# Run on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff
```

---

## 8. Comparing Linters

| Feature | pylint | flake8 | ruff |
|---------|--------|--------|------|
| Speed | Slow | Medium | Very fast |
| Checks | Most comprehensive | Moderate | Comprehensive |
| Auto-fix | No | No | Yes (many rules) |
| Formatting | No | No | Yes (`ruff format`) |
| Import sorting | No | No (plugin) | Yes (built-in) |
| Configuration | Many options | Simple | Many options |
| False positives | More | Fewer | Fewer |
| Best for | Deep analysis | Quick checks | Everything |

### Recommendation for Beginners

Start with **ruff**: it's fast, comprehensive, has auto-fix, includes formatting and import sorting, and has good defaults.

```bash
# One tool to rule them all
pip install ruff
ruff check --fix .     # Lint and fix
ruff format .          # Format
```

---

## 9. Interpreting and Fixing Common Warnings

### 9.1 Unused Import (F401)

```python
import os  # F401: imported but unused

# Fix: Remove the import or use it
```

### 9.2 Unused Variable (F841)

```python
result = expensive_calculation()  # F841: assigned but never used

# Fix: Use the variable or prefix with _
_ = expensive_calculation()  # Convention for intentionally unused
```

### 9.3 Comparison to None (E711)

```python
if x == None:    # E711: comparison to None

if x is None:    # Correct
```

### 9.4 Bare Except (E722)

```python
try:
    process()
except:          # E722: bare except (catches everything, even SystemExit)
    pass

try:
    process()
except Exception:  # Better: only catches actual errors
    pass
```

### 9.5 Line Too Long (E501)

```python
# E501: line too long (120 > 79 characters)
result = some_function(argument_one, argument_two, argument_three, argument_four, argument_five)

# Fix: break across lines
result = some_function(
    argument_one, argument_two, argument_three,
    argument_four, argument_five,
)
```

---

## 10. Integrating Linters with Your Editor

### VS Code

Install the "Ruff" extension, then in `settings.json`:

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
        "source.fixAll.ruff": "explicit",
        "source.organizeImports.ruff": "explicit"
    }
}
```

### PyCharm

- Built-in inspections cover many linter rules
- Install the "Ruff" plugin for ruff integration
- Configure in `Settings > Tools > Ruff`

---

## Summary

- Linters find bugs and code smells; formatters enforce consistent style
- `pylint` is comprehensive but slow; `flake8` is faster but less thorough
- `ruff` is the modern choice: fast, comprehensive, with auto-fix and formatting
- `black` (or `ruff format`) eliminates style debates by auto-formatting code
- Pre-commit hooks run checks automatically before every commit
- Start with `ruff` -- it replaces pylint, flake8, isort, and black in one tool
- Always fix linter warnings -- they often reveal real bugs

---

## Exercises

1. Run `ruff check` on a provided Python file and fix all warnings
2. Configure a `pyproject.toml` with ruff settings
3. Set up `ruff format` to auto-format code
4. Create a pre-commit configuration with ruff hooks

**Previous**: [Testing Basics](./07_Testing_Basics.md) | **Next**: [Type Checking](./09_Type_Checking.md)

# Getting Started with Python

**Next**: [Variables and Data Types](./02_Variables_and_Data_Types.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Install Python on macOS, Linux, and Windows using multiple methods including pyenv
2. Use the Python REPL for interactive exploration and rapid prototyping
3. Write and run Python scripts from the command line
4. Create and manage virtual environments with `venv`
5. Install, upgrade, and remove packages with `pip`
6. Choose and configure an appropriate editor or IDE for Python development
7. Identify key differences between Python 3.10+ and earlier versions
8. Write, save, and execute your first complete Python program

---

Python is one of the most widely used programming languages in the world. Its clear syntax, vast ecosystem, and gentle learning curve make it an excellent first language and a powerful tool for professionals. This lesson walks you through everything you need to go from zero to running code.

## Why Python?

Before diving into installation, it helps to understand why Python has become the lingua franca of programming education and a dominant force in industry.

### Key Strengths

- **Readability**: Python's syntax is designed to be close to natural English. Indentation-based blocks force clean formatting.
- **Versatility**: Web development (Django, Flask), data science (pandas, NumPy), machine learning (PyTorch, scikit-learn), automation, scripting, and more.
- **Massive Ecosystem**: Over 500,000 packages on PyPI (the Python Package Index).
- **Community**: One of the largest and most welcoming developer communities.
- **Cross-Platform**: Runs on macOS, Linux, Windows, and many embedded platforms.

### Python by the Numbers

```
Stack Overflow Developer Survey 2024:
  - 3rd most popular language overall
  - 1st most wanted language (for the 7th consecutive year)
  - Primary language for AI/ML, data science, and scientific computing

GitHub Octoverse 2024:
  - 2nd most used language on GitHub
  - Fastest growing among top-10 languages
```

---

## Installing Python

Python comes pre-installed on macOS and most Linux distributions, but the system version is often outdated. We recommend installing a current version (3.12+) through one of the methods below.

### Method 1: pyenv (Recommended for Developers)

`pyenv` lets you install and switch between multiple Python versions effortlessly. This is the preferred approach for developers who work on multiple projects.

#### Installing pyenv

```bash
# macOS (using Homebrew)
brew update
brew install pyenv

# Linux (using the automatic installer)
curl https://pyenv.run | bash
```

#### Configuring Your Shell

Add the following to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
# pyenv initialization
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart your shell or run `source ~/.zshrc` (or the appropriate file).

#### Installing a Python Version

```bash
# List available versions
pyenv install --list | grep "^  3\."

# Install Python 3.12.7
pyenv install 3.12.7

# Set it as the global default
pyenv global 3.12.7

# Verify
python --version
# Python 3.12.7
```

#### Managing Multiple Versions

```bash
# Install another version
pyenv install 3.11.10

# Set a version for a specific project directory
cd ~/projects/my-legacy-app
pyenv local 3.11.10

# This creates a .python-version file in the directory
cat .python-version
# 3.11.10

# Outside this directory, the global version is used
cd ~
python --version
# Python 3.12.7
```

### Method 2: Official Installer

Download the installer from [python.org/downloads](https://www.python.org/downloads/):

1. Choose the latest stable release (3.12.x or newer).
2. Run the installer.
3. **Important (Windows)**: Check "Add Python to PATH" during installation.
4. Verify the installation.

```bash
# macOS / Linux
python3 --version

# Windows (after adding to PATH)
python --version
```

### Method 3: System Package Manager

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip

# macOS (Homebrew)
brew install python@3.12

# Arch Linux
sudo pacman -S python python-pip
```

### Verifying Your Installation

Regardless of the method you chose, confirm everything works:

```bash
# Check the Python version
python3 --version
# Python 3.12.7

# Check pip (the package installer)
python3 -m pip --version
# pip 24.2 from /usr/lib/python3.12/site-packages/pip (python 3.12)

# Quick sanity check
python3 -c "print('Python is working!')"
# Python is working!
```

---

## The Python REPL

REPL stands for **Read-Eval-Print Loop**. It is an interactive environment where you type Python expressions, and the interpreter immediately evaluates and displays the result.

### Starting the REPL

```bash
python3
```

You will see something like:

```
Python 3.12.7 (main, Oct  1 2024, 08:00:00) [Clang 16.0.0] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

The `>>>` prompt indicates the REPL is waiting for your input.

### Basic REPL Usage

```python
>>> 2 + 3
5

>>> "Hello" + " " + "World"
'Hello World'

>>> len("Python")
6

>>> type(42)
<class 'int'>

>>> import math
>>> math.sqrt(144)
12.0
```

### Multi-Line Statements

When you enter a statement that requires continuation (like an `if` block or a function definition), the prompt changes to `...`:

```python
>>> for i in range(3):
...     print(f"Count: {i}")
...
Count: 0
Count: 1
Count: 2
```

Press Enter on an empty `...` line to execute the block.

### Useful REPL Features

```python
>>> # The underscore _ holds the last result
>>> 7 * 8
56
>>> _ + 4
60

>>> # Get help on any object or function
>>> help(len)
Help on built-in function len in module builtins:
len(obj, /)
    Return the number of items in a container.

>>> # List attributes and methods of an object
>>> dir(str)
['__add__', '__class__', ..., 'upper', 'zfill']

>>> # Exit the REPL
>>> exit()
```

### Enhanced REPLs

The standard REPL is functional but minimal. Consider these alternatives for a richer experience:

| Tool | Description | Install |
|------|-------------|---------|
| **IPython** | Syntax highlighting, tab completion, magic commands | `pip install ipython` |
| **bpython** | Real-time autocomplete, inline documentation | `pip install bpython` |
| **ptpython** | Multi-line editing, vi/emacs key bindings | `pip install ptpython` |

```bash
# Using IPython
pip install ipython
ipython
```

```python
In [1]: import sys

In [2]: sys.version
Out[2]: '3.12.7 (main, Oct  1 2024, 08:00:00) [Clang 16.0.0]'

In [3]: # Tab completion works on objects
In [3]: sys.pl<TAB>
         sys.platform   sys.platlibdir
```

---

## Running Python Scripts

While the REPL is great for experimentation, real programs live in files.

### Creating a Script

Create a file named `hello.py` with any text editor:

```python
# hello.py
# A simple greeting program

def greet(name):
    """Return a greeting message for the given name."""
    return f"Hello, {name}! Welcome to Python."

def main():
    """Entry point of the program."""
    message = greet("World")
    print(message)

    # Greet multiple people
    names = ["Alice", "Bob", "Charlie"]
    for name in names:
        print(greet(name))

if __name__ == "__main__":
    main()
```

### Running the Script

```bash
python3 hello.py
```

Output:

```
Hello, World! Welcome to Python.
Hello, Alice! Welcome to Python.
Hello, Bob! Welcome to Python.
Hello, Charlie! Welcome to Python.
```

### The `if __name__ == "__main__"` Idiom

This pattern is fundamental in Python. Every Python file has a special variable `__name__`:

- When a file is **run directly** (e.g., `python3 hello.py`), `__name__` is set to `"__main__"`.
- When a file is **imported** as a module (e.g., `import hello`), `__name__` is set to the module name (`"hello"`).

```python
# demonstrate_name.py
print(f"__name__ is: {__name__}")

if __name__ == "__main__":
    print("This file was run directly.")
else:
    print("This file was imported as a module.")
```

```bash
# Run directly
python3 demonstrate_name.py
# __name__ is: __main__
# This file was run directly.
```

```python
# Import from REPL
>>> import demonstrate_name
# __name__ is: demonstrate_name
# This file was imported as a module.
```

### Command-Line Arguments

Python provides `sys.argv` for basic argument handling:

```python
# greet_cli.py
import sys

def main():
    """Greet a user by name from the command line."""
    if len(sys.argv) < 2:
        print("Usage: python3 greet_cli.py <name>")
        sys.exit(1)

    name = sys.argv[1]
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main()
```

```bash
python3 greet_cli.py Alice
# Hello, Alice!

python3 greet_cli.py
# Usage: python3 greet_cli.py <name>
```

For more complex argument parsing, use the `argparse` module (covered in Lesson 13).

### Making Scripts Executable (Unix/macOS)

```bash
# Add a shebang line as the first line of your script
#!/usr/bin/env python3

# Make the file executable
chmod +x hello.py

# Run without specifying python3
./hello.py
```

---

## Virtual Environments

A virtual environment is an isolated Python installation that has its own set of packages. This prevents conflicts between projects that require different versions of the same library.

### Why Virtual Environments Matter

Consider this scenario:

```
Project A requires requests==2.28.0
Project B requires requests==2.31.0

Without virtual environments:
  - Installing one version breaks the other project

With virtual environments:
  - Each project has its own isolated copy of requests
  - No conflicts, no headaches
```

### Creating a Virtual Environment

```bash
# Navigate to your project directory
mkdir ~/projects/my-project
cd ~/projects/my-project

# Create a virtual environment named .venv
python3 -m venv .venv
```

This creates a `.venv` directory containing:

```
.venv/
├── bin/               # Executables (python, pip, activate)
│   ├── activate       # Shell activation script
│   ├── pip
│   └── python -> python3.12
├── include/           # C header files (for compiled extensions)
├── lib/               # Installed packages
│   └── python3.12/
│       └── site-packages/
└── pyvenv.cfg         # Configuration file
```

### Activating and Deactivating

```bash
# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Your prompt changes to show the active environment
(.venv) $ python --version
Python 3.12.7

# Deactivate when done
(.venv) $ deactivate
$
```

### Best Practices for Virtual Environments

```bash
# Always add .venv to .gitignore
echo ".venv/" >> .gitignore

# Use a consistent name (.venv is the convention)
python3 -m venv .venv

# Upgrade pip inside the venv immediately after creation
source .venv/bin/activate
pip install --upgrade pip

# Freeze dependencies for reproducibility
pip freeze > requirements.txt

# Recreate an environment from a requirements file
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## pip — The Package Installer

`pip` is Python's standard package manager. It installs packages from [PyPI](https://pypi.org/) (Python Package Index).

### Basic pip Commands

```bash
# Install a package
pip install requests

# Install a specific version
pip install requests==2.31.0

# Install minimum version
pip install "requests>=2.28"

# Upgrade a package
pip install --upgrade requests

# Uninstall a package
pip uninstall requests

# List installed packages
pip list

# Show details about a package
pip show requests
```

### Requirements Files

A `requirements.txt` file lists all project dependencies:

```
# requirements.txt
requests==2.31.0
flask>=3.0,<4.0
python-dotenv~=1.0.0
```

Version specifiers:

| Specifier | Meaning | Example |
|-----------|---------|---------|
| `==` | Exact version | `requests==2.31.0` |
| `>=` | Minimum version | `flask>=3.0` |
| `<=` | Maximum version | `numpy<=1.26.0` |
| `~=` | Compatible release | `django~=5.0.0` (allows 5.0.x but not 5.1) |
| `!=` | Exclude version | `setuptools!=69.0.0` |

```bash
# Install all dependencies from the file
pip install -r requirements.txt

# Generate a requirements file from current environment
pip freeze > requirements.txt
```

### pip vs pipx

- **pip**: Installs packages into the current environment (virtual or global).
- **pipx**: Installs command-line tools in isolated environments. Use it for tools like `black`, `flake8`, and `mypy`.

```bash
# Install pipx
pip install --user pipx
pipx ensurepath

# Install tools with pipx
pipx install black
pipx install flake8
pipx install mypy

# These are now available globally without polluting any venv
black --version
```

---

## IDE and Editor Setup

A good editor makes a significant difference in productivity. Here are the most popular choices for Python development.

### Visual Studio Code (Recommended)

VS Code with the Python extension is the most popular setup:

1. Install VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Install the **Python** extension (by Microsoft).
3. Install the **Pylance** extension for type checking and IntelliSense.

Key settings for Python development (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.tabSize": 4,
        "editor.insertSpaces": true
    },
    "python.linting.enabled": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

JetBrains PyCharm is a full-featured Python IDE:

- **Community Edition**: Free, open-source, covers most needs.
- **Professional Edition**: Paid, adds web framework support, database tools, and remote development.

PyCharm automatically detects virtual environments and provides:
- Intelligent code completion
- Built-in debugger with visual breakpoints
- Integrated testing and profiling
- Refactoring tools

### Terminal-Based Editors

For those who prefer working in the terminal:

```bash
# Neovim with Python support
# Install python3 provider
pip install pynvim

# Vim with ALE (Asynchronous Lint Engine)
# Add to .vimrc:
# Plug 'dense-analysis/ale'
# let g:ale_linters = {'python': ['flake8', 'mypy']}
# let g:ale_fixers = {'python': ['black', 'isort']}
```

### Recommended Extensions and Tools

| Tool | Purpose | Install |
|------|---------|---------|
| **black** | Code formatter (PEP 8 compliant) | `pip install black` |
| **isort** | Import sorter | `pip install isort` |
| **flake8** | Linter (style + logic) | `pip install flake8` |
| **mypy** | Static type checker | `pip install mypy` |
| **ruff** | Fast linter + formatter (Rust-based) | `pip install ruff` |
| **pytest** | Testing framework | `pip install pytest` |

---

## Python Version Differences

Python evolves with each release. Knowing what features are available in which version helps you write modern, clean code.

### Python 3.10+ Features

#### Structural Pattern Matching (3.10)

```python
# match/case — Python's version of switch/case
def classify_status(status_code):
    """Classify an HTTP status code."""
    match status_code:
        case 200:
            return "OK"
        case 301 | 302:
            return "Redirect"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"

print(classify_status(200))   # OK
print(classify_status(302))   # Redirect
print(classify_status(999))   # Unknown
```

#### Parenthesized Context Managers (3.10)

```python
# Before 3.10 — awkward line continuation
with open("input.txt", "r") as infile, \
     open("output.txt", "w") as outfile:
    outfile.write(infile.read())

# Python 3.10+ — clean parenthesized form
with (
    open("input.txt", "r") as infile,
    open("output.txt", "w") as outfile,
):
    outfile.write(infile.read())
```

#### Better Error Messages (3.10+)

Python 3.10 and later provide much more helpful error messages:

```python
# Before 3.10
# SyntaxError: invalid syntax

# Python 3.10+
# SyntaxError: expected ':' after 'if' statement
# SyntaxError: '{' was never closed
# SyntaxError: did you forget a comma?
```

### Python 3.11+ Features

#### Exception Groups (3.11)

```python
# Raise and catch multiple exceptions simultaneously
try:
    raise ExceptionGroup("validation errors", [
        ValueError("name is required"),
        TypeError("age must be an integer"),
    ])
except* ValueError as eg:
    print(f"Value errors: {eg.exceptions}")
except* TypeError as eg:
    print(f"Type errors: {eg.exceptions}")
```

#### Performance (3.11)

Python 3.11 is 10-60% faster than 3.10 for most workloads thanks to the Faster CPython project.

### Python 3.12+ Features

#### Improved f-strings (3.12)

```python
# Nested quotes and expressions in f-strings
names = ["Alice", "Bob"]
print(f"Users: {", ".join(names)}")  # Previously required workarounds
# Users: Alice, Bob

# Backslashes in f-strings
print(f"Newline: {"\n".join(names)}")
```

#### Type Parameter Syntax (3.12)

```python
# New syntax for generic classes and functions
def first[T](items: list[T]) -> T:
    return items[0]

class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()
```

### Version Compatibility Table

| Feature | Minimum Version |
|---------|----------------|
| f-strings | 3.6 |
| Data classes | 3.7 |
| Walrus operator (`:=`) | 3.8 |
| Dictionary union (`\|`) | 3.9 |
| Pattern matching (`match/case`) | 3.10 |
| Exception groups (`except*`) | 3.11 |
| Improved f-strings (nested quotes) | 3.12 |

---

## Your First Program

Let us bring everything together with a small but complete program.

### The Temperature Converter

Create a file called `temperature.py`:

```python
#!/usr/bin/env python3
"""
Temperature Converter
Converts temperatures between Celsius, Fahrenheit, and Kelvin.
"""

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def display_conversions(value, unit):
    """Display all conversions for a given temperature."""
    if unit == "C":
        f = celsius_to_fahrenheit(value)
        k = celsius_to_kelvin(value)
        print(f"  {value:.2f} C = {f:.2f} F = {k:.2f} K")
    elif unit == "F":
        c = fahrenheit_to_celsius(value)
        k = celsius_to_kelvin(c)
        print(f"  {value:.2f} F = {c:.2f} C = {k:.2f} K")
    elif unit == "K":
        c = kelvin_to_celsius(value)
        f = celsius_to_fahrenheit(c)
        print(f"  {value:.2f} K = {c:.2f} C = {f:.2f} F")
    else:
        print(f"  Unknown unit: {unit}")

def main():
    """Main entry point."""
    print("Temperature Converter")
    print("=" * 40)

    # Convert some well-known temperatures
    conversions = [
        (0, "C"),       # Freezing point of water
        (100, "C"),     # Boiling point of water
        (212, "F"),     # Boiling point of water in Fahrenheit
        (98.6, "F"),    # Normal body temperature
        (0, "K"),       # Absolute zero
        (293.15, "K"),  # Room temperature
    ]

    for value, unit in conversions:
        display_conversions(value, unit)

    print("=" * 40)
    print("Done!")

if __name__ == "__main__":
    main()
```

### Running It

```bash
python3 temperature.py
```

Expected output:

```
Temperature Converter
========================================
  0.00 C = 32.00 F = 273.15 K
  100.00 C = 212.00 F = 373.15 K
  212.00 F = 100.00 C = 373.15 K
  98.60 F = 37.00 C = 310.15 K
  0.00 K = -273.15 C = -459.67 F
  293.15 K = 20.00 C = 68.00 F
========================================
Done!
```

### What This Program Demonstrates

| Concept | Where It Appears |
|---------|-----------------|
| Shebang line | Line 1 (`#!/usr/bin/env python3`) |
| Module docstring | Lines 2-5 (triple-quoted string) |
| Function definitions | `def celsius_to_fahrenheit(celsius):` |
| Function docstrings | `"""Convert Celsius to Fahrenheit."""` |
| f-string formatting | `f"{value:.2f} C = {f:.2f} F"` |
| Conditional logic | `if unit == "C":` |
| List of tuples | `conversions = [(0, "C"), ...]` |
| Tuple unpacking in loop | `for value, unit in conversions:` |
| Main guard | `if __name__ == "__main__":` |
| String methods | `"=" * 40` (repetition) |

---

## Project Structure Best Practices

Even for small projects, establishing a good structure pays dividends. Here is a minimal recommended layout:

```
my-project/
├── .venv/                # Virtual environment (never commit)
├── .gitignore            # Excludes .venv, __pycache__, etc.
├── README.md             # Project description
├── requirements.txt      # Pinned dependencies
├── src/                  # Source code
│   └── my_project/
│       ├── __init__.py
│       └── main.py
└── tests/                # Test files
    ├── __init__.py
    └── test_main.py
```

A good `.gitignore` for Python projects:

```gitignore
# Virtual environments
.venv/
venv/
env/

# Python bytecode
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db
```

---

## Common Pitfalls for Beginners

### 1. Using the System Python

```bash
# BAD: Installing packages into the system Python
sudo pip install requests   # Do not do this

# GOOD: Always use a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install requests
```

### 2. Python 2 vs Python 3

Python 2 reached end-of-life on January 1, 2020. Always use Python 3.

```bash
# Check which python you are using
python --version    # Might be Python 2 on some systems!
python3 --version   # Always Python 3

# On some systems, ensure 'python' points to Python 3
# pyenv handles this automatically
```

### 3. Forgetting to Activate the Virtual Environment

```bash
# Symptom: packages not found even though you installed them
pip install requests
python3 -c "import requests"  # ModuleNotFoundError

# Fix: activate the correct virtual environment
source .venv/bin/activate
pip install requests
python3 -c "import requests"  # Works!
```

### 4. Indentation Errors

Python uses indentation (spaces) to define code blocks. Mixing tabs and spaces causes errors.

```python
# BAD: mixed indentation
def greet():
    print("Hello")      # 4 spaces
	print("World")      # 1 tab -- SyntaxError!

# GOOD: consistent 4-space indentation
def greet():
    print("Hello")
    print("World")
```

Configure your editor to insert 4 spaces when you press Tab.

---

## Exercises

1. **Install Python**: Install Python 3.12+ using pyenv or your preferred method. Verify with `python3 --version`.

2. **REPL Exploration**: Open the Python REPL and:
   - Calculate `2 ** 100` (2 to the power of 100).
   - Find the type of `3.14`.
   - Use `help()` to read the documentation for the `print` function.

3. **First Script**: Create a script that asks the user for their name (using `input()`) and prints a personalized greeting.

4. **Virtual Environment**: Create a virtual environment, activate it, install the `requests` package, and verify the import works.

5. **Temperature Converter Enhancement**: Modify the temperature converter to accept user input:
   - Prompt for a temperature value.
   - Prompt for the unit (C, F, or K).
   - Display all conversions.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Installation** | Use pyenv for version management; always use Python 3.12+ |
| **REPL** | Use `>>>` for quick experiments; try IPython for a better experience |
| **Scripts** | Save code in `.py` files; use the `__name__` guard |
| **Virtual Environments** | Always isolate project dependencies with `python3 -m venv .venv` |
| **pip** | Install packages with `pip install`; pin versions in `requirements.txt` |
| **IDE** | VS Code + Python extension is the most popular choice |
| **Modern Python** | Target 3.10+ for pattern matching, better errors, and performance |

---

**Next**: [Variables and Data Types](./02_Variables_and_Data_Types.md)

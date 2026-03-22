# Modules and Packages

**Previous**: [OOP Advanced](./09_OOP_Advanced.md) | **Next**: [File I/O](./11_File_IO.md)

> **Topic**: Python Basics
> **Lesson**: 10 of 14
> **Prerequisites**: Functions, Classes and Objects, OOP Fundamentals

## Learning Objectives

After completing this lesson, you will be able to:

1. Import modules and specific objects using `import`, `from...import`, and `import as` syntax
2. Explain how Python resolves module names using `sys.path` and the module search order
3. Create your own reusable modules and understand the `__name__ == "__main__"` pattern
4. Organize code into packages using `__init__.py` and understand package structure
5. Distinguish between relative and absolute imports and know when to use each
6. Navigate the Python standard library and identify key modules for common tasks
7. Install and manage third-party packages using `pip` and `requirements.txt`
8. Set up virtual environments to isolate project dependencies

---

## Introduction

As your programs grow beyond a single file, you need a way to organize code into manageable, reusable pieces. Python's module and package system provides exactly this. A **module** is simply a `.py` file containing Python definitions and statements. A **package** is a directory of modules with a special `__init__.py` file. Together, they form the backbone of code organization in Python.

Every Python file you have written so far has been a module. In this lesson, you will learn how to split your code across files, import functionality between them, use the vast Python standard library, and tap into the rich ecosystem of third-party packages.

---

## The `import` Statement

### Basic Import

The simplest way to use code from another module is with the `import` statement:

```python
import math

print(math.pi)          # 3.141592653589793
print(math.sqrt(16))    # 4.0
print(math.ceil(4.2))   # 5
print(math.floor(4.8))  # 4
```

When you write `import math`, Python:
1. Searches for the `math` module
2. Executes all the code in that module
3. Creates a module object named `math` in your current namespace

You access the module's contents using dot notation: `math.pi`, `math.sqrt()`.

### Importing Multiple Modules

You can import several modules on separate lines (preferred style) or on one line:

```python
# Preferred: one import per line
import os
import sys
import json

# Also valid but less readable
import os, sys, json
```

PEP 8 recommends one import per line for better readability and cleaner version control diffs.

### Import Order Convention

PEP 8 specifies import ordering:

```python
# 1. Standard library imports
import os
import sys
from collections import defaultdict

# 2. Related third-party imports
import requests
import numpy as np

# 3. Local application/library specific imports
from mypackage import mymodule
from mypackage.utils import helper
```

Separate each group with a blank line.

---

## The `from...import` Statement

### Importing Specific Names

Instead of importing the entire module, you can import specific objects:

```python
from math import pi, sqrt, ceil

print(pi)        # 3.141592653589793
print(sqrt(25))  # 5.0
print(ceil(4.1)) # 5

# Note: math itself is NOT available
# math.floor(4.8)  # NameError: name 'math' is not defined
```

### Importing Everything (Avoid This)

The wildcard import `from module import *` imports all public names:

```python
from math import *

# Now ALL math functions are available directly
print(sin(0))    # 0.0
print(cos(0))    # 1.0
print(tan(0))    # 0.0
```

**Why you should avoid `import *`:**

```python
from math import *
from cmath import *  # Complex math

# Which 'sqrt' do we have now?
# cmath.sqrt overwrote math.sqrt silently!
print(sqrt(4))  # (2+0j)  -- complex number, probably not what you wanted
```

Wildcard imports:
- Pollute the namespace with unknown names
- Can silently shadow existing names
- Make it impossible to determine where a name came from
- Break static analysis tools

### Controlling `import *` with `__all__`

Module authors can control what `import *` exports:

```python
# mymodule.py
__all__ = ["public_func", "PublicClass"]

def public_func():
    return "I am exported with import *"

def _private_func():
    return "I am never exported with import *"

class PublicClass:
    pass

class _InternalClass:
    pass
```

---

## The `import as` Statement (Aliasing)

### Aliasing Modules

You can give a module a shorter or different name:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Now use the aliases
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({"values": data})
```

Common community conventions:
- `numpy` → `np`
- `pandas` → `pd`
- `matplotlib.pyplot` → `plt`
- `tensorflow` → `tf`
- `seaborn` → `sns`

### Aliasing Imported Names

```python
from datetime import datetime as dt
from collections import OrderedDict as OD

now = dt.now()
config = OD([("host", "localhost"), ("port", 8080)])
```

### Resolving Name Conflicts

Aliases are useful when two modules export the same name:

```python
from json import dumps as json_dumps
from yaml import dumps as yaml_dumps

data = {"name": "Alice", "age": 30}
print(json_dumps(data))
print(yaml_dumps(data))
```

---

## Module Search Path

When you write `import mymodule`, Python searches for it in a specific order:

### The Search Order

1. **`sys.modules` cache** — Already imported modules (cached for performance)
2. **Built-in modules** — Modules compiled into the Python interpreter (e.g., `sys`, `builtins`)
3. **`sys.path`** — A list of directories to search

### Inspecting `sys.path`

```python
import sys

for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
```

Typical output:

```
0: /home/user/project          # Current directory (or script directory)
1: /usr/lib/python3.12         # Standard library
2: /usr/lib/python3.12/lib-dynload
3: /home/user/.local/lib/python3.12/site-packages  # User packages
4: /usr/lib/python3.12/site-packages                # System packages
```

### Key Points About `sys.path`

```python
import sys

# The first entry is usually the current directory or script directory
print(sys.path[0])

# You can modify sys.path at runtime (but usually should not)
sys.path.insert(0, "/path/to/my/modules")

# Now Python will search /path/to/my/modules first
import my_custom_module
```

### The `PYTHONPATH` Environment Variable

You can add directories to `sys.path` using the `PYTHONPATH` environment variable:

```bash
# In your shell
export PYTHONPATH="/home/user/my_libs:/home/user/other_libs"
python my_script.py
```

### Checking Where a Module Comes From

```python
import os
import json
import math

print(os.__file__)    # /usr/lib/python3.12/os.py
print(json.__file__)  # /usr/lib/python3.12/json/__init__.py
print(math.__file__)  # /usr/lib/python3.12/lib-dynload/math.cpython-312-x86_64-linux-gnu.so
```

---

## Creating Your Own Modules

### A Simple Module

Any Python file is a module. Create a file called `geometry.py`:

```python
# geometry.py
"""Geometry utility functions for 2D shapes."""

import math

PI = math.pi

def circle_area(radius):
    """Calculate the area of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return PI * radius ** 2

def circle_circumference(radius):
    """Calculate the circumference of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 2 * PI * radius

def rectangle_area(width, height):
    """Calculate the area of a rectangle."""
    if width < 0 or height < 0:
        raise ValueError("Dimensions cannot be negative")
    return width * height

def triangle_area(base, height):
    """Calculate the area of a triangle."""
    if base < 0 or height < 0:
        raise ValueError("Dimensions cannot be negative")
    return 0.5 * base * height

class Circle:
    """Represents a circle with a given radius."""

    def __init__(self, radius):
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        self.radius = radius

    @property
    def area(self):
        """The area of the circle."""
        return circle_area(self.radius)

    @property
    def circumference(self):
        """The circumference of the circle."""
        return circle_circumference(self.radius)

    def __repr__(self):
        return f"Circle(radius={self.radius})"
```

### Using Your Module

In another file in the same directory:

```python
# main.py
import geometry

print(geometry.circle_area(5))        # 78.53981633974483
print(geometry.rectangle_area(4, 6))  # 24

c = geometry.Circle(10)
print(c.area)           # 314.1592653589793
print(c.circumference)  # 62.83185307179586
```

Or import specific items:

```python
# main.py
from geometry import circle_area, Circle

print(circle_area(5))  # 78.53981633974483

c = Circle(3)
print(c)  # Circle(radius=3)
```

---

## The `__name__ == "__main__"` Pattern

### Understanding `__name__`

Every module has a built-in attribute `__name__`:
- When a module is **run directly**: `__name__` is set to `"__main__"`
- When a module is **imported**: `__name__` is set to the module's name

```python
# greet.py
print(f"__name__ is: {__name__}")

def greet(name):
    """Return a greeting string."""
    return f"Hello, {name}!"
```

```bash
# Running directly
$ python greet.py
__name__ is: __main__
```

```python
# Importing in another file
import greet  # Prints: __name__ is: greet
```

### The Guard Pattern

Use the `if __name__ == "__main__"` guard to write code that runs only when the file is executed directly:

```python
# converter.py
"""Temperature conversion utilities."""

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

if __name__ == "__main__":
    # This block only runs when the file is executed directly
    # It does NOT run when the file is imported as a module
    print("Temperature Converter")
    print("=" * 30)

    test_temps = [0, 20, 37, 100]
    for temp in test_temps:
        f = celsius_to_fahrenheit(temp)
        k = celsius_to_kelvin(temp)
        print(f"{temp}C = {f:.1f}F = {k:.2f}K")
```

```bash
# Run directly: the test code executes
$ python converter.py
Temperature Converter
==============================
0C = 32.0F = 273.15K
20C = 68.0F = 293.15K
37C = 98.6F = 310.15K
100C = 212.0F = 373.15K
```

```python
# Import as module: only functions are available, test code does not run
from converter import celsius_to_fahrenheit
print(celsius_to_fahrenheit(100))  # 212.0
```

### Why This Pattern Matters

1. **Dual-purpose files**: The module works both as an importable library and a standalone script
2. **Testing convenience**: You can include quick tests or demos in the guard block
3. **Entry points**: CLI tools often use this pattern as their main entry point

```python
# database.py
"""Database connection utility."""

import sqlite3

def connect(db_path):
    """Connect to a SQLite database and return the connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(conn):
    """Create application tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

if __name__ == "__main__":
    # Quick setup and verification when run directly
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else ":memory:"
    print(f"Connecting to: {db_path}")

    conn = connect(db_path)
    create_tables(conn)
    print("Tables created successfully")

    # Verify
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row["name"] for row in cursor]
    print(f"Tables: {tables}")
    conn.close()
```

---

## Packages

### What Is a Package?

A **package** is a directory containing Python modules and a special `__init__.py` file. Packages let you organize related modules under a common namespace.

### Basic Package Structure

```
myproject/
├── main.py
└── shapes/
    ├── __init__.py
    ├── circle.py
    ├── rectangle.py
    └── triangle.py
```

```python
# shapes/__init__.py
"""Shapes package for geometric calculations."""

# This file can be empty, or it can define package-level imports
```

```python
# shapes/circle.py
"""Circle calculations."""

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

    def __repr__(self):
        return f"Circle(radius={self.radius})"
```

```python
# shapes/rectangle.py
"""Rectangle calculations."""

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    @property
    def perimeter(self):
        return 2 * (self.width + self.height)

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"
```

```python
# shapes/triangle.py
"""Triangle calculations."""

import math

class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @property
    def area(self):
        # Heron's formula
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    @property
    def perimeter(self):
        return self.a + self.b + self.c

    def __repr__(self):
        return f"Triangle(a={self.a}, b={self.b}, c={self.c})"
```

### Importing from Packages

```python
# main.py

# Import a module from the package
from shapes import circle
c = circle.Circle(5)
print(c.area)  # 78.53981633974483

# Import a class directly
from shapes.rectangle import Rectangle
r = Rectangle(4, 6)
print(r.area)  # 24

# Import the package itself
import shapes.triangle
t = shapes.triangle.Triangle(3, 4, 5)
print(t.area)  # 6.0
```

### Using `__init__.py` for Convenience Imports

Make commonly used classes available at the package level:

```python
# shapes/__init__.py
"""Shapes package for geometric calculations."""

from shapes.circle import Circle
from shapes.rectangle import Rectangle
from shapes.triangle import Triangle

__all__ = ["Circle", "Rectangle", "Triangle"]
```

Now users can import directly from the package:

```python
# main.py
from shapes import Circle, Rectangle, Triangle

c = Circle(5)
r = Rectangle(4, 6)
t = Triangle(3, 4, 5)

print(c.area)  # 78.53981633974483
print(r.area)  # 24
print(t.area)  # 6.0
```

### Subpackages

Packages can contain other packages:

```
myproject/
├── main.py
└── shapes/
    ├── __init__.py
    ├── two_d/
    │   ├── __init__.py
    │   ├── circle.py
    │   └── rectangle.py
    └── three_d/
        ├── __init__.py
        ├── sphere.py
        └── cube.py
```

```python
from shapes.two_d.circle import Circle
from shapes.three_d.sphere import Sphere
```

---

## Relative vs Absolute Imports

### Absolute Imports

Absolute imports use the full path from the project root:

```python
# Inside shapes/circle.py, importing from shapes/utils.py
from shapes.utils import validate_positive
from shapes.constants import PI
```

### Relative Imports

Relative imports use dots to refer to the current and parent packages:

```python
# Inside shapes/circle.py
from . import utils                    # Import utils module from same package
from .utils import validate_positive   # Import specific name from sibling module
from .constants import PI              # Import from sibling module

# Inside shapes/two_d/circle.py
from .. import constants               # Import from parent package
from ..constants import PI             # Import specific name from parent package
from . import rectangle                # Import sibling module
```

**Dot notation:**
- `.` — current package
- `..` — parent package
- `...` — grandparent package

### When to Use Which

| Aspect | Absolute | Relative |
|--------|----------|----------|
| Readability | Clear, explicit | Shorter but less obvious |
| Refactoring | Must update if package moves | Survives package renaming |
| PEP 8 | Preferred | Acceptable for internal packages |
| Standalone scripts | Always works | Only works inside packages |

**General guideline:** Use absolute imports for public APIs and relative imports for internal package organization.

### Important Note on Relative Imports

Relative imports only work inside packages. You cannot use them in a standalone script:

```python
# This will FAIL if run directly: python shapes/circle.py
from . import utils  # ImportError: attempted relative import with no known parent package

# This WORKS when imported as part of a package
# python -c "from shapes.circle import Circle"
```

---

## The Standard Library Overview

Python's "batteries included" philosophy means the standard library is extensive:

### Text and Data Processing

```python
import string
print(string.ascii_lowercase)  # abcdefghijklmnopqrstuvwxyz
print(string.digits)           # 0123456789
print(string.punctuation)      # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

import re
pattern = re.compile(r"\b\w+@\w+\.\w+\b")
emails = pattern.findall("Contact us at info@example.com or help@test.org")
print(emails)  # ['info@example.com', 'help@test.org']

import textwrap
long_text = "This is a very long string that should be wrapped at a certain width."
print(textwrap.fill(long_text, width=30))
```

### File and OS Operations

```python
import os
import shutil
import pathlib
import tempfile
import glob

# Current directory
print(os.getcwd())

# List files matching a pattern
for f in glob.glob("*.py"):
    print(f)

# Path manipulation with pathlib
p = pathlib.Path("data/output/results.csv")
print(p.parent)    # data/output
print(p.stem)      # results
print(p.suffix)    # .csv
```

### Data Formats

```python
import json
import csv
import configparser
import xml.etree.ElementTree as ET
import sqlite3

# JSON
data = {"name": "Alice", "scores": [95, 87, 91]}
json_str = json.dumps(data, indent=2)
print(json_str)
```

### Mathematics and Numbers

```python
import math
import statistics
import random
import decimal
import fractions

# Statistics
scores = [85, 90, 78, 92, 88, 76, 95]
print(statistics.mean(scores))    # 86.28571428571429
print(statistics.median(scores))  # 88
print(statistics.stdev(scores))   # 7.1572...

# Precise decimal arithmetic
from decimal import Decimal
price = Decimal("19.99")
tax = Decimal("0.08")
total = price * (1 + tax)
print(total)  # 21.5892
```

### Date and Time

```python
import datetime
import time
import calendar

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Calculate time difference
future = now + datetime.timedelta(days=30)
print(f"30 days from now: {future.date()}")
```

### Networking and Internet

```python
import urllib.request
import urllib.parse
import http.server
import email
import socket

# URL parsing
url = "https://example.com/path?key=value&lang=en"
parsed = urllib.parse.urlparse(url)
print(parsed.scheme)    # https
print(parsed.netloc)    # example.com
print(parsed.path)      # /path
params = urllib.parse.parse_qs(parsed.query)
print(params)           # {'key': ['value'], 'lang': ['en']}
```

### Debugging and Testing

```python
import logging
import unittest
import pdb
import timeit
import profile

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Application started")
logger.warning("Disk space low")

# Timing code
elapsed = timeit.timeit("sum(range(1000))", number=10000)
print(f"Time: {elapsed:.4f} seconds")
```

---

## Installing Third-Party Packages

### Using `pip`

`pip` is Python's package installer:

```bash
# Install a package
pip install requests

# Install a specific version
pip install requests==2.31.0

# Install minimum version
pip install "requests>=2.28.0"

# Upgrade a package
pip install --upgrade requests

# Uninstall a package
pip uninstall requests

# Show package info
pip show requests

# List all installed packages
pip list

# Search for outdated packages
pip list --outdated
```

### `requirements.txt`

A `requirements.txt` file lists all project dependencies:

```text
# requirements.txt
requests==2.31.0
flask==3.0.0
sqlalchemy>=2.0.0,<3.0.0
pytest>=7.0.0
python-dotenv~=1.0.0
```

Version specifiers:
- `==2.31.0` — Exact version
- `>=2.28.0` — Minimum version
- `<3.0.0` — Maximum version
- `~=1.0.0` — Compatible release (>=1.0.0, <2.0.0)
- `>=2.28.0,<3.0.0` — Range

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Generate requirements.txt from current environment
pip freeze > requirements.txt
```

### `pip freeze` vs Curated `requirements.txt`

```bash
# pip freeze outputs EVERYTHING installed (including transitive dependencies)
$ pip freeze
certifi==2023.11.17
charset-normalizer==3.3.2
idna==3.6
requests==2.31.0
urllib3==2.1.0

# A curated requirements.txt lists only YOUR direct dependencies
# requests==2.31.0
```

For production projects, consider using `pip-tools` or `pip freeze` with careful curation.

---

## Virtual Environments

### Why Virtual Environments?

Without virtual environments, all projects share the same Python packages:

```
Project A needs requests==2.28.0
Project B needs requests==2.31.0
Conflict! Only one version can be installed globally.
```

Virtual environments create isolated Python installations for each project.

### Creating and Using Virtual Environments

```bash
# Create a virtual environment
python -m venv myproject_env

# Activate it (Linux/macOS)
source myproject_env/bin/activate

# Activate it (Windows)
myproject_env\Scripts\activate

# Your prompt changes to show the active environment
(myproject_env) $ python --version
Python 3.12.0

# Install packages (isolated to this environment)
(myproject_env) $ pip install requests flask

# See where packages are installed
(myproject_env) $ pip show requests
Location: /home/user/myproject_env/lib/python3.12/site-packages

# Deactivate when done
(myproject_env) $ deactivate
```

### Project Workflow

```bash
# Standard project setup
mkdir myproject && cd myproject
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install requests flask pytest

# Save dependencies
pip freeze > requirements.txt

# Later, on another machine or after cloning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `.gitignore` for Virtual Environments

Always exclude virtual environments from version control:

```gitignore
# .gitignore
venv/
env/
.venv/
__pycache__/
*.pyc
```

---

## Popular Third-Party Packages

### HTTP Requests — `requests`

```python
import requests

# GET request
response = requests.get("https://api.github.com/users/python")
print(response.status_code)  # 200
data = response.json()
print(data["name"])           # Python

# POST request with JSON body
response = requests.post(
    "https://httpbin.org/post",
    json={"username": "alice", "action": "login"}
)
print(response.json()["json"])

# With headers and timeout
response = requests.get(
    "https://api.example.com/data",
    headers={"Authorization": "Bearer token123"},
    timeout=10
)
```

### Testing — `pytest`

```python
# test_math_utils.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(0, 0) == 0

def test_add_mixed():
    assert add(-1, 1) == 0
```

```bash
# Run tests
$ pytest test_math_utils.py -v
test_math_utils.py::test_add_positive PASSED
test_math_utils.py::test_add_negative PASSED
test_math_utils.py::test_add_zero PASSED
test_math_utils.py::test_add_mixed PASSED
```

### Data Validation — `pydantic`

```python
from pydantic import BaseModel, EmailStr, field_validator

class User(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v

# Valid data
user = User(name="Alice", age=30, email="alice@example.com")
print(user.model_dump())  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# Invalid data raises ValidationError
try:
    user = User(name="Bob", age=-5, email="bob@test.com")
except Exception as e:
    print(e)  # Validation error for age
```

### Environment Variables — `python-dotenv`

```python
# .env file:
# DATABASE_URL=postgresql://localhost/mydb
# SECRET_KEY=super-secret-key-123
# DEBUG=true

from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file into environment variables

db_url = os.getenv("DATABASE_URL")
secret = os.getenv("SECRET_KEY")
debug = os.getenv("DEBUG", "false").lower() == "true"

print(f"Database: {db_url}")
print(f"Debug mode: {debug}")
```

### Command-Line Interfaces — `click`

```python
import click

@click.command()
@click.option("--name", prompt="Your name", help="The person to greet")
@click.option("--count", default=1, help="Number of greetings")
def hello(name, count):
    """Simple program that greets NAME for COUNT times."""
    for _ in range(count):
        click.echo(f"Hello, {name}!")

if __name__ == "__main__":
    hello()
```

```bash
$ python hello.py --name Alice --count 3
Hello, Alice!
Hello, Alice!
Hello, Alice!
```

---

## Module Reloading and Caching

### The Module Cache

Python caches imported modules in `sys.modules`:

```python
import sys

import json
print("json" in sys.modules)  # True

# Subsequent imports use the cache (no re-execution)
import json  # Instant, uses cached version
```

### Reloading Modules

During development, you might want to reload a modified module:

```python
import importlib
import mymodule

# After modifying mymodule.py
importlib.reload(mymodule)
```

**Caveats:**
- Objects created from the old module are not updated
- `from mymodule import func` bindings are not updated
- Use primarily during interactive development, not in production

---

## Module Attributes and Introspection

### Special Module Attributes

```python
import json

print(json.__name__)     # json
print(json.__file__)     # /usr/lib/python3.12/json/__init__.py
print(json.__doc__[:50]) # JSON (JavaScript Object Notation) <http://json.org
print(json.__package__)  # json
print(json.__spec__)     # ModuleSpec(name='json', ...)

# List all names in a module
print(dir(json))
# ['JSONDecodeError', 'JSONDecoder', 'JSONEncoder', 'dump', 'dumps', 'load', 'loads', ...]
```

### Using `dir()` to Explore Modules

```python
import math

# Find all names that contain "log"
log_funcs = [name for name in dir(math) if "log" in name.lower()]
print(log_funcs)  # ['log', 'log10', 'log1p', 'log2']

# Get help on a specific function
help(math.log)
```

### The `inspect` Module

```python
import inspect
import json

# Get the source file
print(inspect.getfile(json))

# Check if something is a function, class, etc.
print(inspect.isfunction(json.dumps))  # True
print(inspect.isclass(json.JSONEncoder))  # True

# Get function signature
sig = inspect.signature(json.dumps)
print(sig)
# (obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, ...)
```

---

## Common Patterns and Best Practices

### Lazy Imports

Import expensive modules only when needed:

```python
def process_data(data):
    """Process data, importing numpy only if needed."""
    import numpy as np  # Imported only when function is called
    return np.array(data).mean()
```

### Conditional Imports

Handle optional dependencies gracefully:

```python
try:
    import ujson as json  # Faster JSON library
except ImportError:
    import json           # Fall back to standard library

# Use whichever was imported
data = json.dumps({"key": "value"})
```

### Circular Import Prevention

Circular imports occur when module A imports module B, and module B imports module A:

```python
# BAD: circular imports
# a.py
from b import func_b
def func_a():
    return func_b() + 1

# b.py
from a import func_a    # ImportError! a hasn't finished loading yet
def func_b():
    return func_a() + 1
```

**Solutions:**

```python
# Solution 1: Import inside the function
# a.py
def func_a():
    from b import func_b  # Deferred import
    return func_b() + 1

# Solution 2: Restructure - move shared code to a third module
# common.py
def shared_logic():
    return 42

# a.py
from common import shared_logic

# b.py
from common import shared_logic
```

### Module-Level Configuration

```python
# config.py
"""Application configuration module."""

import os

# Read from environment with defaults
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "10"))

# Computed settings
if DEBUG:
    LOG_LEVEL = "DEBUG"
```

```python
# app.py
import config

print(config.DATABASE_URL)
print(config.DEBUG)
```

---

## Practical Exercise: Building a Package

Let us build a complete package step by step:

```
calculator/
├── __init__.py
├── basic.py
├── scientific.py
└── history.py
```

```python
# calculator/__init__.py
"""A simple calculator package."""

from calculator.basic import add, subtract, multiply, divide
from calculator.scientific import power, sqrt, log

__version__ = "1.0.0"
__all__ = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log"]
```

```python
# calculator/basic.py
"""Basic arithmetic operations."""

def add(a, b):
    """Return the sum of a and b."""
    return a + b

def subtract(a, b):
    """Return the difference of a and b."""
    return a - b

def multiply(a, b):
    """Return the product of a and b."""
    return a * b

def divide(a, b):
    """Return the quotient of a divided by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

```python
# calculator/scientific.py
"""Scientific calculator operations."""

import math

def power(base, exponent):
    """Return base raised to the power of exponent."""
    return base ** exponent

def sqrt(n):
    """Return the square root of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Cannot take square root of negative number")
    return math.sqrt(n)

def log(n, base=math.e):
    """Return the logarithm of n with the given base.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return math.log(n, base)
```

```python
# calculator/history.py
"""Calculation history tracking."""

class History:
    """Tracks a history of calculations."""

    def __init__(self):
        self._entries = []

    def record(self, expression, result):
        """Record a calculation."""
        self._entries.append({"expression": expression, "result": result})

    def last(self, n=1):
        """Return the last n entries."""
        return self._entries[-n:]

    def clear(self):
        """Clear all history."""
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f"History({len(self._entries)} entries)"
```

```python
# main.py — Using the calculator package
from calculator import add, subtract, multiply, divide, power, sqrt
from calculator.history import History

history = History()

# Basic operations
result = add(10, 5)
history.record("10 + 5", result)
print(f"10 + 5 = {result}")  # 15

result = divide(20, 4)
history.record("20 / 4", result)
print(f"20 / 4 = {result}")  # 5.0

# Scientific operations
result = power(2, 10)
history.record("2^10", result)
print(f"2^10 = {result}")  # 1024

result = sqrt(144)
history.record("sqrt(144)", result)
print(f"sqrt(144) = {result}")  # 12.0

# History
print(f"\nHistory: {history}")
for entry in history.last(3):
    print(f"  {entry['expression']} = {entry['result']}")
```

---

## Summary

| Concept | Syntax | Use Case |
|---------|--------|----------|
| Import module | `import math` | Access all names via `math.name` |
| Import name | `from math import sqrt` | Direct access to `sqrt` |
| Alias | `import numpy as np` | Shorter name |
| Module guard | `if __name__ == "__main__":` | Code that runs only when script is executed directly |
| Package | Directory with `__init__.py` | Organize related modules |
| Relative import | `from . import sibling` | Within a package |
| pip install | `pip install package` | Install third-party packages |
| Virtual env | `python -m venv env` | Isolated dependencies |

Key takeaways:
- **Modules** are `.py` files; **packages** are directories with `__init__.py`
- Use **absolute imports** by default; **relative imports** for internal package structure
- Always use `if __name__ == "__main__":` to make modules both importable and runnable
- Use **virtual environments** for every project
- The **standard library** is vast — check it before reaching for third-party packages
- Keep your `requirements.txt` up to date

---

## Further Reading

- [Python Import System (Official Docs)](https://docs.python.org/3/reference/import.html)
- [Python Module Index](https://docs.python.org/3/py-modindex.html)
- [PEP 8 — Import Conventions](https://peps.python.org/pep-0008/#imports)
- [PEP 328 — Imports: Multi-Line and Absolute/Relative](https://peps.python.org/pep-0328/)
- [Python Packaging User Guide](https://packaging.python.org/)

---

**Previous**: [OOP Advanced](./09_OOP_Advanced.md) | **Next**: [File I/O](./11_File_IO.md)

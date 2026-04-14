# Type Checking

**Previous**: [Linters and Formatters](./08_Linters_and_Formatters.md) | **Next**: [Profiling Basics](./10_Profiling_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Add type hints to functions, variables, and class attributes
2. Use built-in types (`list`, `dict`, `tuple`) and `typing` module types (`Optional`, `Union`)
3. Run `mypy` to check type correctness statically
4. Interpret mypy error messages and fix type-related bugs
5. Apply gradual typing to add types to an existing codebase incrementally
6. Use `TypeAlias`, `Literal`, `TypedDict`, and other advanced type constructs
7. Understand the limits of type checking and when it catches bugs
8. Configure mypy with `pyproject.toml`

---

Python is dynamically typed: variables can hold any type at any time. This flexibility is powerful but dangerous -- type-related bugs are among the most common errors in Python code. Type hints let you annotate your code with expected types, and tools like `mypy` check those annotations **before** your code runs. This catches an entire category of bugs at development time rather than at runtime.

> **Key Insight:** Type hints don't change how Python runs your code. They are annotations for humans and tools. Python ignores them at runtime, but `mypy` reads them and warns you about type mismatches.

---

## 1. Why Type Hints?

### The Problem

```python
def calculate_discount(price, discount):
    return price * (1 - discount)

# These all "work" but are they correct?
calculate_discount(100, 0.1)      # 90.0 ✓
calculate_discount("100", 0.1)    # TypeError at runtime!
calculate_discount(100, "10%")    # TypeError at runtime!
calculate_discount(100, None)     # TypeError at runtime!
```

### The Solution

```python
def calculate_discount(price: float, discount: float) -> float:
    return price * (1 - discount)
```

Now `mypy` will catch errors **before runtime**:

```
error: Argument 1 to "calculate_discount" has incompatible type "str"; expected "float"
```

---

## 2. Basic Type Hints

### 2.1 Function Annotations

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def is_valid(value: float) -> bool:
    return 0.0 <= value <= 1.0

def process(data: list) -> None:  # None return type
    for item in data:
        print(item)
```

### 2.2 Variable Annotations

```python
name: str = "Alice"
age: int = 30
scores: list[float] = [95.5, 87.0, 92.3]
settings: dict[str, int] = {"width": 800, "height": 600}
is_active: bool = True
```

### 2.3 Collection Types (Python 3.9+)

```python
# Built-in generics (Python 3.9+)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, float] = {"Alice": 95.0, "Bob": 87.5}
coordinates: tuple[float, float] = (3.14, 2.71)
unique_ids: set[int] = {1, 2, 3}

# Variable-length tuple
values: tuple[int, ...] = (1, 2, 3, 4, 5)

# Nested types
matrix: list[list[int]] = [[1, 2], [3, 4]]
users: dict[str, dict[str, str]] = {
    "alice": {"email": "alice@example.com", "role": "admin"}
}
```

For Python 3.7-3.8, import from `typing`:
```python
from typing import List, Dict, Tuple, Set
names: List[str] = ["Alice", "Bob"]
```

---

## 3. Special Types

### 3.1 Optional (None-able values)

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    """Return username or None if not found."""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Python 3.10+ syntax
def find_user(user_id: int) -> str | None:
    ...
```

### 3.2 Union (multiple possible types)

```python
from typing import Union

def normalize(value: Union[int, float, str]) -> float:
    if isinstance(value, str):
        return float(value)
    return float(value)

# Python 3.10+ syntax
def normalize(value: int | float | str) -> float:
    ...
```

### 3.3 Any (escape hatch)

```python
from typing import Any

def log_value(value: Any) -> None:
    """Accept any type -- avoid using this when possible."""
    print(value)
```

### 3.4 Literal (specific values)

```python
from typing import Literal

def set_direction(direction: Literal["north", "south", "east", "west"]) -> None:
    print(f"Moving {direction}")

set_direction("north")   # OK
set_direction("up")      # mypy error: "up" is not a valid value
```

### 3.5 Callable

```python
from typing import Callable

def apply_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

apply_function(lambda x, y: x + y, 3, 4)  # OK
```

---

## 4. Type Hints for Classes

```python
class Student:
    name: str
    age: int
    grades: list[float]
    
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
        self.grades = []
    
    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)
    
    def average(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)
    
    def is_passing(self) -> bool:
        return self.average() >= 60.0
```

### TypedDict for Structured Dictionaries

```python
from typing import TypedDict

class UserProfile(TypedDict):
    name: str
    age: int
    email: str

def process_user(user: UserProfile) -> str:
    return f"{user['name']} ({user['age']})"

# mypy will check that the dict has the right keys and types
user: UserProfile = {"name": "Alice", "age": 30, "email": "alice@example.com"}
process_user(user)  # OK

bad_user: UserProfile = {"name": "Bob", "age": "thirty", "email": "bob@example.com"}
# mypy error: incompatible type "str"; expected "int"
```

---

## 5. Running mypy

### 5.1 Installation and Basic Usage

```bash
pip install mypy
mypy my_script.py
```

### 5.2 Example Session

```python
# file: demo.py
def add(a: int, b: int) -> int:
    return a + b

result: str = add(1, 2)   # Bug: assigning int to str
print(result.upper())     # This would crash at runtime
```

```bash
$ mypy demo.py
demo.py:4: error: Incompatible types in assignment
    (expression has type "int", variable has type "str")  [assignment]
Found 1 error in 1 file (checked 1 source file)
```

### 5.3 Common mypy Errors

| Error | Meaning |
|-------|---------|
| `Incompatible types in assignment` | Assigning wrong type to variable |
| `Incompatible return value type` | Function returns wrong type |
| `Argument ... has incompatible type` | Passing wrong type to function |
| `Item "None" of "Optional[X]" has no attribute` | Using Optional without None check |
| `"X" has no attribute "Y"` | Calling method that doesn't exist on type |
| `Missing return statement` | Function declares return type but doesn't always return |

### 5.4 Handling Optional Correctly

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice"}
    return users.get(user_id)

# BAD: mypy catches this!
name = find_user(1)
print(name.upper())  # error: Item "None" of "str | None" has no attribute "upper"

# GOOD: Check for None first
name = find_user(1)
if name is not None:
    print(name.upper())  # OK: mypy knows name is str here

# ALTERNATIVE: assertion
name = find_user(1)
assert name is not None
print(name.upper())  # OK after assertion
```

---

## 6. Gradual Typing

You don't have to add types everywhere at once. Add them gradually:

### 6.1 Strategy

```
1. Start with new code → Always add type hints
2. Add types to public function signatures
3. Add types to frequently-changed files
4. Use mypy in CI to prevent regressions
5. Gradually increase strictness
```

### 6.2 Silencing Specific Lines

```python
result = sketchy_function()  # type: ignore[no-any-return]
```

### 6.3 mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false    # Start permissive
check_untyped_defs = true

# Strict mode for new modules
[[tool.mypy.overrides]]
module = "myapp.new_module"
disallow_untyped_defs = true

# Ignore third-party libraries without stubs
[[tool.mypy.overrides]]
module = "some_library.*"
ignore_missing_imports = true
```

---

## 7. Type Aliases

```python
from typing import TypeAlias

# Simple alias
Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[list[float]]
UserId: TypeAlias = int

def dot_product(v1: Vector, v2: Vector) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def add_vectors(v1: Vector, v2: Vector) -> Vector:
    return [a + b for a, b in zip(v1, v2)]

# Complex nested types become readable
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
```

---

## 8. Type Checking Catches Real Bugs

### 8.1 None-Safety

```python
def get_config(key: str) -> str | None:
    config = {"host": "localhost", "port": "8080"}
    return config.get(key)

# Without type checking, this bug goes unnoticed until runtime:
port = get_config("port")
url = f"http://localhost:{port + 1}"  # TypeError if port is None!

# With mypy:
# error: Unsupported operand types for + ("str | None" and "int")
```

### 8.2 Wrong Return Type

```python
def parse_age(text: str) -> int:
    if text.isdigit():
        return int(text)
    # Missing return! mypy catches: Missing return statement

# Fix:
def parse_age(text: str) -> int:
    if text.isdigit():
        return int(text)
    raise ValueError(f"Invalid age: {text!r}")
```

### 8.3 Container Type Mismatch

```python
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

scores: list[str] = ["90", "85", "78"]
average(scores)
# mypy error: Argument 1 has incompatible type "list[str]"; expected "list[float]"
```

---

## 9. Common Patterns

### 9.1 Return Self from Methods

```python
from typing import Self  # Python 3.11+

class Builder:
    def __init__(self) -> None:
        self.items: list[str] = []
    
    def add(self, item: str) -> Self:
        self.items.append(item)
        return self

# Enables fluent interface
result = Builder().add("a").add("b").add("c")
```

### 9.2 Dataclasses with Types

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
```

### 9.3 Protocol (Structural Typing)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()  # Works for any object with a draw() method
```

---

## 10. Integrating Type Checking into Your Workflow

### Editor Integration

```bash
# VS Code: Install "Mypy Type Checker" extension
# PyCharm: Built-in type checking support
```

### CI/CD

```yaml
# GitHub Actions
- name: Type check
  run: mypy src/
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

---

## Summary

- Type hints annotate expected types without changing runtime behavior
- `mypy` checks type correctness statically, catching bugs before code runs
- Use `Optional[X]` (or `X | None`) for values that might be `None`
- Type checking catches None-safety bugs, wrong argument types, and missing returns
- Gradual typing lets you add types incrementally to existing codebases
- Configure mypy in `pyproject.toml` for project-wide settings
- `TypedDict`, `Literal`, and `Protocol` provide advanced type constructs
- Type checking works best when combined with linters and formatters

---

## Exercises

1. Add type hints to a set of untyped functions
2. Fix mypy errors in a typed codebase
3. Use `Optional` correctly to handle None-returning functions
4. Create a `TypedDict` for a structured configuration dictionary

**Previous**: [Linters and Formatters](./08_Linters_and_Formatters.md) | **Next**: [Profiling Basics](./10_Profiling_Basics.md)

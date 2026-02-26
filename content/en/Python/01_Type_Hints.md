# Type Hints

**Next**: [Decorators](./02_Decorators.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why type hints improve code quality despite Python being dynamically typed
2. Write basic type annotations for variables, function parameters, and return values
3. Apply collection types (`list`, `dict`, `set`, `tuple`) with parameterized type arguments
4. Distinguish between `Optional`, `Union`, `Any`, `Callable`, and `TypeVar` from the `typing` module
5. Implement generic classes using `Generic[T]` and structural subtyping using `Protocol`
6. Define complex type aliases and use `TypedDict` for structured dictionary types
7. Apply static type checking with `mypy` and understand runtime vs. static type enforcement
8. Write overloaded function signatures using `@overload`

---

Type hints bridge the gap between Python's dynamic flexibility and the safety guarantees found in statically typed languages. In production codebases with dozens of contributors, type annotations serve as machine-checkable documentation that catches entire categories of bugs -- wrong argument types, mismatched return values, forgotten `None` checks -- before the code ever runs. Modern Python tooling (IDEs, linters, CI pipelines) leverages type hints to deliver autocompletion, refactoring support, and automated verification at scale.

## 1. What are Type Hints?

Type hints are a feature introduced in Python 3.5+ that allow you to explicitly specify the types of variables and functions. While Python remains a dynamically typed language, type hints can improve code readability and stability.

```python
# Without type hints
def greet(name):
    return f"Hello, {name}"

# With type hints
def greet(name: str) -> str:
    return f"Hello, {name}"
```

### Advantages of Type Hints

| Advantage | Description |
|-----------|-------------|
| Readability | Clearly shows function input/output types |
| IDE Support | Enables autocomplete and type error warnings |
| Documentation | Code serves as its own documentation |
| Bug Prevention | Detect errors before runtime through static analysis |

---

## 2. Basic Type Hints

### Primitive Types

```python
# Variable type hints
name: str = "Python"
age: int = 30
price: float = 19.99
is_active: bool = True
data: bytes = b"hello"

# Function type hints
def add(a: int, b: int) -> int:
    return a + b

def say_hello(name: str) -> None:
    print(f"Hello, {name}")
```

### Collection Types

```python
# Python 3.9+ (direct use of built-in types)
numbers: list[int] = [1, 2, 3]
names: set[str] = {"Alice", "Bob"}
scores: dict[str, int] = {"math": 95, "english": 88}
point: tuple[int, int] = (10, 20)
coordinates: tuple[float, ...] = (1.0, 2.0, 3.0)  # Variable length

# Python 3.8 and below (using typing module)
from typing import List, Set, Dict, Tuple

numbers: List[int] = [1, 2, 3]
names: Set[str] = {"Alice", "Bob"}
scores: Dict[str, int] = {"math": 95}
point: Tuple[int, int] = (10, 20)
```

---

## 3. typing Module

### Optional

Indicates that a value can be None.

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    """Find user or return None"""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Python 3.10+ alternative
def find_user(user_id: int) -> str | None:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)
```

### Union

Allows one of multiple types.

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    """Accept int or str and convert to string"""
    return str(value)

# Python 3.10+ alternative
def process(value: int | str) -> str:
    return str(value)
```

### Any

Allows any type (disables type checking).

```python
from typing import Any

def log(message: Any) -> None:
    print(message)

# Any value is allowed
log("hello")
log(123)
log([1, 2, 3])
```

### Callable

Specifies function types.

```python
from typing import Callable

# Callable[[argument_types], return_type]
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def add(x: int, y: int) -> int:
    return x + y

result = apply(add, 3, 4)  # 7
```

### TypeVar

Defines generic type variables.

```python
from typing import TypeVar, List

T = TypeVar('T')

def first(items: List[T]) -> T:
    """Return first element of list"""
    return items[0]

# Type is automatically inferred
num = first([1, 2, 3])      # int
name = first(["a", "b"])    # str
```

---

## 4. Advanced Type Hints

### Generic Classes

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0

# Usage
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")
```

### TypedDict

Defines dictionary key and value types.

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

# Requires exact keys and types
user: User = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}

# Optional keys
class UserOptional(TypedDict, total=False):
    name: str
    age: int
    nickname: str  # Optional
```

### Protocol (Structural Subtyping)

Express duck typing with type hints.

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

# Circle and Square don't inherit from Drawable
# but they can be used because they have draw() method
render(Circle())  # OK
render(Square())  # OK
```

### Literal

Allows only specific literal values.

```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode: {mode}")

set_mode("read")    # OK
set_mode("write")   # OK
# set_mode("delete")  # Type error!
```

### Final

Indicates constants (no reassignment).

```python
from typing import Final

MAX_SIZE: Final = 100
PI: Final[float] = 3.14159

# MAX_SIZE = 200  # Type checker warns
```

---

## 5. Type Aliases

Give names to complex types.

```python
from typing import Dict, List, Tuple

# Type aliases
UserId = int
UserData = Dict[str, str]
UserList = List[Tuple[UserId, UserData]]

def get_users() -> UserList:
    return [
        (1, {"name": "Alice", "email": "alice@example.com"}),
        (2, {"name": "Bob", "email": "bob@example.com"}),
    ]

# Python 3.10+ TypeAlias
from typing import TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]
```

---

## 6. Runtime vs Static Analysis

### Type Hints Are Not Enforced at Runtime

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

# Runs without issues at runtime!
result = greet(123)  # "Hello, 123"
```

### Static Type Checker (mypy)

```bash
# Install
pip install mypy

# Run type check
mypy your_script.py
```

```python
# example.py
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # mypy detects error
```

```
$ mypy example.py
example.py:4: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

### Runtime Type Checking (Optional)

```python
from typing import get_type_hints

def validate_types(func):
    """Runtime type validation decorator"""
    hints = get_type_hints(func)

    def wrapper(*args, **kwargs):
        # Validate argument types
        for (name, value), expected_type in zip(
            kwargs.items(), hints.values()
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type}")
        return func(*args, **kwargs)
    return wrapper
```

---

## 7. Practical Patterns

### API Response Type Definition

```python
from typing import TypedDict, List, Optional

class APIResponse(TypedDict):
    success: bool
    data: Optional[dict]
    error: Optional[str]

class User(TypedDict):
    id: int
    name: str
    email: str

class UsersResponse(TypedDict):
    users: List[User]
    total: int
    page: int

def fetch_users(page: int = 1) -> UsersResponse:
    # API call logic
    return {
        "users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}],
        "total": 100,
        "page": page
    }
```

### Configuration Class

```python
from typing import Final, ClassVar

class Config:
    # Class variable (shared across instances)
    DEBUG: ClassVar[bool] = False
    VERSION: ClassVar[str] = "1.0.0"

    # Constants
    MAX_CONNECTIONS: Final = 100
    TIMEOUT: Final[int] = 30
```

### Overload

Define multiple signatures for the same function.

```python
from typing import overload, Union

@overload
def process(value: int) -> int: ...

@overload
def process(value: str) -> str: ...

def process(value: Union[int, str]) -> Union[int, str]:
    if isinstance(value, int):
        return value * 2
    return value.upper()

# Type checker infers correct return type
num: int = process(5)       # int
text: str = process("hi")   # str
```

---

## 8. Commonly Used Types Summary

| Type | Description | Example |
|------|-------------|---------|
| `int`, `str`, `float`, `bool` | Basic types | `x: int = 1` |
| `list[T]` | List | `nums: list[int]` |
| `dict[K, V]` | Dictionary | `data: dict[str, int]` |
| `set[T]` | Set | `ids: set[int]` |
| `tuple[T, ...]` | Tuple | `point: tuple[int, int]` |
| `Optional[T]` | T or None | `name: Optional[str]` |
| `Union[T1, T2]` | T1 or T2 | `value: Union[int, str]` |
| `Callable[[Args], R]` | Function type | `fn: Callable[[int], str]` |
| `Any` | Any type | `data: Any` |
| `TypeVar` | Generic variable | `T = TypeVar('T')` |
| `Protocol` | Structural typing | `class Sized(Protocol)` |

---

## 9. Practice Problems

### Exercise 1: Function Type Hints

Add appropriate type hints to the following function.

```python
def calculate_average(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
```

### Exercise 2: Generic Function

Write a generic function that finds the first element satisfying a condition in a list.

```python
# find_first([1, 2, 3, 4], lambda x: x > 2) -> 3
# find_first(["a", "bb", "ccc"], lambda s: len(s) > 1) -> "bb"
```

### Exercise 3: TypedDict

Define a TypedDict representing a user profile.
- Required: id (int), username (str), email (str)
- Optional: bio (str), avatar_url (str)

---

**Next**: [Decorators](./02_Decorators.md)

# Context Managers

**Previous**: [Decorators](./02_Decorators.md) | **Next**: [Iterators & Generators](./04_Iterators_and_Generators.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of the `with` statement and how it guarantees resource cleanup
2. Implement a context manager class with `__enter__` and `__exit__` methods
3. Describe the role of each `__exit__` parameter (`exc_type`, `exc_val`, `exc_tb`) and control exception propagation
4. Write generator-based context managers using `@contextmanager` from `contextlib`
5. Apply `contextlib` utilities: `suppress`, `redirect_stdout`, `closing`, and `ExitStack`
6. Compose multiple context managers in a single `with` statement
7. Implement practical patterns such as timers, temporary directory changes, and transactions
8. Write async context managers using `__aenter__` / `__aexit__` and `@asynccontextmanager`

---

Resource leaks -- unclosed files, unreleased locks, dangling database connections -- are among the most insidious bugs in long-running applications. Python's `with` statement and the context manager protocol provide a deterministic, exception-safe way to acquire and release resources. Whether you are opening files, managing database transactions, or temporarily changing global state, context managers ensure that cleanup code always runs, even when exceptions interrupt the happy path.

## 1. What are Context Managers?

Context managers are used with the `with` statement to automatically handle setup and cleanup of resources.

```python
# Without context manager
file = open("example.txt", "w")
try:
    file.write("Hello")
finally:
    file.close()

# Using context manager
with open("example.txt", "w") as file:
    file.write("Hello")
# file.close() is automatically called
```

### Execution Flow

```
with expression as variable:
    │
    ▼
┌─────────────────────┐
│  __enter__() called │ ← Resource setup
│  Return value → var │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Execute with body │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  __exit__() called  │ ← Resource cleanup (runs even on exception)
└─────────────────────┘
```

---

## 2. Implementing with Classes

Implement `__enter__` and `__exit__` methods.

### Basic Structure

```python
class MyContextManager:
    def __enter__(self):
        print("Resource setup")
        return self  # Value bound to 'as' clause

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Resource cleanup")
        return False  # Re-raise exception

with MyContextManager() as cm:
    print("Performing operation")
```

Output:
```
Resource setup
Performing operation
Resource cleanup
```

### File Manager Example

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
```

### Database Connection Example

```python
class DatabaseConnection:
    def __init__(self, host, database):
        self.host = host
        self.database = database
        self.connection = None

    def __enter__(self):
        print(f"Connecting: {self.host}/{self.database}")
        self.connection = {"host": self.host, "db": self.database}
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.connection = None
        return False

with DatabaseConnection("localhost", "mydb") as conn:
    print(f"Using: {conn}")
```

---

## 3. Exception Handling in __exit__

The `__exit__` method can receive and handle exception information.

### Parameters

| Parameter | Description |
|-----------|-------------|
| exc_type | Exception class (e.g., `ValueError`) |
| exc_val | Exception instance |
| exc_tb | Traceback object |

All are `None` if no exception occurred.

### Exception Handling Example

```python
class ErrorHandler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            # Return True to suppress exception (don't propagate)
            return True
        return False

with ErrorHandler():
    raise ValueError("Test error")

print("This line executes (exception was suppressed)")
```

Output:
```
Exception occurred: ValueError: Test error
This line executes (exception was suppressed)
```

### Handling Specific Exceptions Only

```python
class IgnoreValueError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Suppress only ValueError
        if exc_type is ValueError:
            print(f"ValueError ignored: {exc_val}")
            return True
        return False  # Propagate other exceptions

with IgnoreValueError():
    raise ValueError("This error is ignored")

# with IgnoreValueError():
#     raise TypeError("This error propagates")  # Program stops
```

---

## 4. contextlib Module

### @contextmanager Decorator

Easily create context managers using generator functions.

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Setup")       # __enter__ part
    yield "resource"     # Value bound to 'as' clause
    print("Cleanup")     # __exit__ part

with my_context() as value:
    print(f"Using: {value}")
```

Output:
```
Setup
Using: resource
Cleanup
```

### Including Exception Handling

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Acquiring resource")
    try:
        yield "resource"
    except Exception as e:
        print(f"Handling exception: {e}")
        raise  # Re-raise exception (remove to suppress)
    finally:
        print("Releasing resource")

with managed_resource() as r:
    print(f"Using: {r}")
    # raise ValueError("Test")
```

### File Manager (contextmanager version)

```python
from contextlib import contextmanager

@contextmanager
def open_file(path, mode):
    f = open(path, mode)
    try:
        yield f
    finally:
        f.close()

with open_file("test.txt", "w") as f:
    f.write("Hello!")
```

---

## 5. contextlib Utilities

### suppress - Suppress Exceptions

```python
from contextlib import suppress

# Traditional way
try:
    import json
    data = json.loads("invalid")
except json.JSONDecodeError:
    pass

# Using suppress
with suppress(json.JSONDecodeError):
    data = json.loads("invalid")
# Exception is ignored if it occurs
```

### redirect_stdout - Redirect Output

```python
from contextlib import redirect_stdout
import io

# Capture output to string
f = io.StringIO()
with redirect_stdout(f):
    print("This output is captured")

output = f.getvalue()
print(f"Captured content: {output}")
```

### closing - Auto-call close()

```python
from contextlib import closing
from urllib.request import urlopen

# urlopen is not a context manager (for Python 2 compatibility)
with closing(urlopen("https://example.com")) as page:
    content = page.read()
```

### ExitStack - Dynamic Context Management

Manage multiple context managers dynamically.

```python
from contextlib import ExitStack

files = ["file1.txt", "file2.txt", "file3.txt"]

with ExitStack() as stack:
    file_objects = [
        stack.enter_context(open(f, "w"))
        for f in files
    ]
    # Write to all files
    for f in file_objects:
        f.write("Hello\n")
# All files are automatically closed
```

---

## 6. Nested Context Managers

### Multiple with Statements

```python
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        outfile.write(infile.read())
```

### Single Line

```python
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())
```

### Multi-line with Parentheses

```python
# Python 3.10+
with (
    open("file1.txt") as f1,
    open("file2.txt") as f2,
    open("file3.txt") as f3,
):
    # Use all files
    pass
```

---

## 7. Practical Patterns

### Timer

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name="Task"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")

with timer("Data processing"):
    # Time-consuming task
    time.sleep(0.5)
```

### Temporary Directory Change

```python
from contextlib import contextmanager
import os

@contextmanager
def change_dir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

with change_dir("/tmp"):
    print(f"Current: {os.getcwd()}")
# Automatically restored to original directory
```

### Temporary Environment Variables

```python
from contextlib import contextmanager
import os

@contextmanager
def temp_env(**kwargs):
    old_env = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

with temp_env(DEBUG="true", API_KEY="test"):
    print(os.environ["DEBUG"])  # true
# Original environment restored
```

### Lock

```python
from contextlib import contextmanager
import threading

@contextmanager
def locked(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# Actually, Lock itself is a context manager
lock = threading.Lock()
with lock:
    # Critical section
    pass
```

### Transaction Pattern

```python
from contextlib import contextmanager

class Transaction:
    def __init__(self):
        self.operations = []

    def add(self, op):
        self.operations.append(op)

    def commit(self):
        for op in self.operations:
            print(f"Executing: {op}")
        self.operations.clear()

    def rollback(self):
        print("Rolling back!")
        self.operations.clear()

@contextmanager
def transaction(tx):
    try:
        yield tx
        tx.commit()
    except Exception:
        tx.rollback()
        raise

tx = Transaction()
with transaction(tx):
    tx.add("INSERT INTO users VALUES (1, 'Alice')")
    tx.add("UPDATE accounts SET balance = 100")
    # raise ValueError("Error!")  # Uncomment to rollback
```

---

## 8. Async Context Managers

Use `async with` by implementing `__aenter__` and `__aexit__`.

```python
class AsyncResource:
    async def __aenter__(self):
        print("Async setup")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Async cleanup")
        return False

async def main():
    async with AsyncResource() as r:
        print("Async operation")

import asyncio
asyncio.run(main())
```

### contextlib's asynccontextmanager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer(name):
    import time
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.4f}s")

async def main():
    async with async_timer("Async task"):
        await asyncio.sleep(0.5)
```

---

## 9. Summary

| Method | When to Use |
|--------|-------------|
| Class (`__enter__`, `__exit__`) | When state management is needed |
| `@contextmanager` | Simple setup/cleanup logic |
| `suppress` | Ignore specific exceptions |
| `redirect_stdout` | Redirect output |
| `ExitStack` | Dynamic context management |
| `closing` | Auto-call close() method |

---

## 10. Practice Problems

### Exercise 1: Timeout Context Manager

Create a context manager that raises TimeoutError after a specified time.

### Exercise 2: Log Level Change

Create a context manager that temporarily changes the logging level and then restores it.

### Exercise 3: Test Double

Create a context manager that temporarily replaces a function for testing purposes.

---

**Previous**: [Decorators](./02_Decorators.md) | **Next**: [Iterators & Generators](./04_Iterators_and_Generators.md)

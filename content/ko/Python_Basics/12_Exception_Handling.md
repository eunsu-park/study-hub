# 예외 처리

**이전**: [파일 입출력](./11_File_IO.md) | **다음**: [표준 라이브러리 핵심](./13_Standard_Library_Essentials.md)

> **주제**: Python 기초
> **수업**: 14개 중 12번째
> **선수 지식**: 함수, 파일 입출력, 클래스와 객체

## 학습 목표

이 수업을 완료하면 다음을 할 수 있습니다:

1. `try`/`except`/`else`/`finally` 블록을 작성하여 오류를 우아하게 처리하기
2. 특정 예외를 잡고 여러 예외 유형을 다르게 처리하기
3. `BaseException`에서 특정 예외까지 Python 예외 계층 구조를 탐색하기
4. `raise`로 예외를 발생시키고 `raise ... from ...`으로 예외를 연쇄하기
5. 애플리케이션의 오류 시나리오에 맞는 사용자 정의 예외 클래스 정의하기
6. LBYL과 EAFP 프로그래밍 스타일을 구분하고 각각을 적절히 적용하기
7. 가장 일반적인 내장 예외(`ValueError`, `TypeError`, `KeyError` 등)를 파악하고 처리하기
8. 예외 처리 모범 사례를 적용하여 견고하고 유지보수 가능한 코드 작성하기

---

## 소개

오류는 발생합니다. 사용자가 잘못된 데이터를 입력하고, 파일이 사라지고, 네트워크 연결이 끊어지고, 서버의 메모리가 부족해집니다. 예외 처리 (Exception Handling)는 Python 프로그램이 충돌 대신 이러한 상황에 우아하게 대응하는 방법입니다.

**예외**는 프로그램의 정상적인 흐름을 방해하는 이벤트입니다. Python이 오류를 만나면 예외 객체를 생성하고 "발생(raise)"시킵니다. 예외가 잡히지(처리되지) 않으면 프로그램은 트레이스백과 함께 종료됩니다. 예외 처리를 통해 이러한 오류를 가로채고, 적절하게 대응하며, 프로그램을 계속 실행할 수 있습니다.

---

## 예외 처리 없이 일어나는 일

```python
# This program crashes when given bad input
def divide(a, b):
    return a / b

# Normal usage works fine
print(divide(10, 3))  # 3.3333333333333335

# But this crashes the entire program
print(divide(10, 0))
# ZeroDivisionError: division by zero
# Program terminates here - no code below this runs
```

트레이스백은 다음을 보여줍니다:
```
Traceback (most recent call last):
  File "example.py", line 7, in <module>
    print(divide(10, 0))
  File "example.py", line 2, in divide
    return a / b
ZeroDivisionError: division by zero
```

---

## `try`/`except` 블록

### 기본 구문

```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    # Code that runs if ZeroDivisionError occurs
    print("Cannot divide by zero!")
```

### 예외 객체 잡기

```python
try:
    result = int("not_a_number")
except ValueError as e:
    print(f"Conversion failed: {e}")
    # Output: Conversion failed: invalid literal for int() with base 10: 'not_a_number'
```

### 실용적인 예제

```python
def safe_divide(a, b):
    """Divide a by b, returning None if division is impossible."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Warning: Division by zero, returning None")
        return None
    except TypeError as e:
        print(f"Warning: Invalid types for division: {e}")
        return None

# Normal case
print(safe_divide(10, 3))     # 3.3333333333333335

# Division by zero
print(safe_divide(10, 0))     # None (with warning)

# Type error
print(safe_divide("10", 3))   # None (with warning)
```

---

## 다중 `except` 절

### 다른 예외를 다르게 처리하기

```python
def process_data(data, index):
    """Access and convert data at a given index."""
    try:
        value = data[index]
        return int(value)
    except IndexError:
        print(f"Index {index} is out of range (list has {len(data)} items)")
        return None
    except ValueError:
        print(f"Cannot convert '{data[index]}' to integer")
        return None
    except TypeError:
        print(f"Invalid data type: {type(data)} is not subscriptable")
        return None

data = ["10", "20", "abc", "40"]
print(process_data(data, 0))   # 10
print(process_data(data, 2))   # None (cannot convert 'abc')
print(process_data(data, 10))  # None (index out of range)
print(process_data(None, 0))   # None (not subscriptable)
```

### 하나의 절에서 여러 예외 잡기

```python
def parse_value(text):
    """Parse a numeric value from text."""
    try:
        return float(text)
    except (ValueError, TypeError) as e:
        print(f"Cannot parse value: {e}")
        return 0.0

print(parse_value("3.14"))   # 3.14
print(parse_value("hello"))  # 0.0
print(parse_value(None))     # 0.0
```

### 순서가 중요합니다: 가장 구체적인 것부터

```python
# WRONG: The broad exception catches everything
try:
    value = int("abc")
except Exception:
    print("Something went wrong")  # This always runs
except ValueError:
    print("Invalid value")  # This NEVER runs (unreachable)

# CORRECT: Most specific first, broadest last
try:
    value = int("abc")
except ValueError:
    print("Invalid value")         # This catches ValueError
except Exception:
    print("Something went wrong")  # This catches everything else
```

---

## `else` 절

`else` 블록은 `try` 블록이 예외를 발생시키지 **않고** 완료될 때만 실행됩니다:

```python
def read_config(filepath):
    """Read a configuration file and return its contents."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Config file not found: {filepath}")
        return {}
    except PermissionError:
        print(f"No permission to read: {filepath}")
        return {}
    else:
        # Only runs if no exception was raised
        print(f"Successfully read {len(content)} characters from {filepath}")
        import json
        return json.loads(content)
```

### 왜 `else`를 사용하나요?

```python
# Without else: the parse step is inside try, but we only want
# to catch errors from the file reading, not from JSON parsing
try:
    with open("data.json", "r", encoding="utf-8") as f:
        content = f.read()
    data = json.loads(content)  # Bug: FileNotFoundError handler
                                 # won't help with JSON errors
except FileNotFoundError:
    data = {}

# With else: clearly separates "risky" code from "follow-up" code
try:
    with open("data.json", "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    data = {}
else:
    # JSON errors will propagate normally (not caught by FileNotFoundError handler)
    data = json.loads(content)
```

---

## `finally` 절

`finally` 블록은 예외 발생 여부에 관계없이 **항상** 실행됩니다:

```python
def process_file(filepath):
    """Process a file, ensuring cleanup happens."""
    resource = None
    try:
        resource = acquire_resource()
        data = resource.read(filepath)
        return transform(data)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # This ALWAYS runs, no matter what
        if resource is not None:
            resource.release()
            print("Resource released")
```

### `finally` 실행 보장

```python
def demonstrate_finally():
    """Show that finally always runs."""
    try:
        print("1. Inside try")
        return "from try"
    finally:
        print("2. Inside finally")  # Runs even though try returned!

result = demonstrate_finally()
# Output:
# 1. Inside try
# 2. Inside finally
print(result)  # from try
```

```python
def another_example():
    """Finally runs even with unhandled exceptions."""
    try:
        print("Trying...")
        raise ValueError("Something broke")
    finally:
        print("Cleaning up...")  # Runs before the exception propagates

# another_example()
# Output:
# Trying...
# Cleaning up...
# Then: ValueError: Something broke
```

### 완전한 `try`/`except`/`else`/`finally`

```python
import json
from pathlib import Path

def load_user_data(filepath):
    """Load user data with complete error handling."""
    print(f"Attempting to load: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"  File not found: {filepath}")
        return None
    except PermissionError:
        print(f"  Permission denied: {filepath}")
        return None
    else:
        # Only runs if file was read successfully
        try:
            data = json.loads(raw)
            print(f"  Loaded {len(data)} records")
            return data
        except json.JSONDecodeError as e:
            print(f"  Invalid JSON: {e}")
            return None
    finally:
        # Always runs
        print(f"  Finished processing: {filepath}")
```

---

## 예외 발생시키기

### `raise` 문

`raise`를 사용하여 오류 조건이 발생했음을 알립니다:

```python
def set_age(age):
    """Set a person's age with validation."""
    if not isinstance(age, int):
        raise TypeError(f"Age must be an integer, got {type(age).__name__}")
    if age < 0:
        raise ValueError(f"Age cannot be negative, got {age}")
    if age > 150:
        raise ValueError(f"Age {age} is unrealistically large")
    return age

# Valid
print(set_age(25))  # 25

# Invalid
try:
    set_age(-5)
except ValueError as e:
    print(f"Error: {e}")  # Error: Age cannot be negative, got -5

try:
    set_age("twenty")
except TypeError as e:
    print(f"Error: {e}")  # Error: Age must be an integer, got str
```

### 예외 다시 발생시키기

때로는 예외를 잡고, 무언가(로깅 등)를 한 다음, 다시 발생시키고 싶을 때가 있습니다:

```python
import logging

logger = logging.getLogger(__name__)

def process_payment(amount, card_number):
    """Process a payment, logging any errors."""
    try:
        validate_card(card_number)
        charge(amount, card_number)
    except PaymentError as e:
        logger.error(f"Payment failed for card ending {card_number[-4:]}: {e}")
        raise  # Re-raise the same exception with original traceback
```

### 단독 `raise`

인자 없이 `raise`를 사용하면 현재 예외를 다시 발생시킵니다:

```python
try:
    result = risky_operation()
except Exception:
    cleanup()
    raise  # Re-raise whatever exception was caught
```

---

## 예외 연쇄

### 암시적 연쇄

`except` 블록 내에서 예외가 발생하면 Python이 자동으로 연쇄합니다:

```python
try:
    result = 1 / 0
except ZeroDivisionError:
    raise ValueError("Calculation failed")
# Output includes:
# ZeroDivisionError: division by zero
#
# During handling of the above exception, another exception occurred:
#
# ValueError: Calculation failed
```

### `from`을 사용한 명시적 연쇄

`raise ... from ...`을 사용하여 인과 관계를 명시적으로 나타냅니다:

```python
def load_config(filepath):
    """Load configuration from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            import json
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file missing: {filepath}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid config format in {filepath}") from e
```

출력:

```
FileNotFoundError: [Errno 2] No such file or directory: 'config.json'

The above exception was the direct cause of the following exception:

ConfigError: Config file missing: config.json
```

### `from None`으로 연쇄 억제하기

```python
def get_setting(config, key):
    """Get a setting from config, raising a clear error if missing."""
    try:
        return config[key]
    except KeyError:
        raise SettingError(f"Required setting '{key}' not found") from None
        # 'from None' suppresses the original KeyError in the traceback
```

---

## 예외 계층 구조

Python 예외는 클래스 계층 구조를 형성합니다:

```
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── AttributeError
    ├── EOFError
    ├── ImportError
    │   └── ModuleNotFoundError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── NameError
    │   └── UnboundLocalError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── FileExistsError
    │   ├── PermissionError
    │   ├── IsADirectoryError
    │   ├── NotADirectoryError
    │   └── TimeoutError
    ├── RuntimeError
    │   ├── NotImplementedError
    │   └── RecursionError
    ├── StopIteration
    ├── TypeError
    ├── ValueError
    │   └── UnicodeError
    │       ├── UnicodeDecodeError
    │       └── UnicodeEncodeError
    └── Warning
        ├── DeprecationWarning
        ├── UserWarning
        └── FutureWarning
```

### 핵심 사항

```python
# Catching Exception catches most errors but NOT:
# - SystemExit (raised by sys.exit())
# - KeyboardInterrupt (raised by Ctrl+C)
# - GeneratorExit (raised when generator is closed)

try:
    dangerous_operation()
except Exception as e:
    # Catches ValueError, TypeError, OSError, etc.
    # Does NOT catch KeyboardInterrupt or SystemExit
    print(f"Error: {e}")

# NEVER catch BaseException (it catches Ctrl+C!)
# BAD:
try:
    long_operation()
except BaseException:  # User cannot Ctrl+C to stop!
    pass
```

### 부모 클래스 잡기

```python
# Catching a parent class catches all its children
try:
    value = my_list[100]
except LookupError:
    # Catches both IndexError and KeyError
    print("Lookup failed")

try:
    with open("missing.txt") as f:
        pass
except OSError:
    # Catches FileNotFoundError, PermissionError, IsADirectoryError, etc.
    print("OS error occurred")
```

---

## 일반적인 내장 예외

### `ValueError`

함수가 올바른 타입이지만 잘못된 값의 인자를 받을 때 발생합니다:

```python
# Common causes of ValueError
int("abc")           # invalid literal for int()
int("3.14")          # invalid literal for int() (use float first)
float("not_a_num")   # could not convert string to float
list.remove("missing_item")  # x not in list

# Handling ValueError
def parse_int(text, default=0):
    """Parse an integer from text, returning default on failure."""
    try:
        return int(text)
    except ValueError:
        return default

print(parse_int("42"))      # 42
print(parse_int("hello"))   # 0
print(parse_int("", -1))    # -1
```

### `TypeError`

부적절한 타입의 객체에 연산이 적용될 때 발생합니다:

```python
# Common causes of TypeError
"hello" + 5           # can only concatenate str to str
len(42)               # object of type 'int' has no len()
"hello"[1.5]          # string indices must be integers
sum("abc")            # unsupported operand type

# Handling TypeError
def safe_add(a, b):
    """Add two values, handling type mismatches."""
    try:
        return a + b
    except TypeError:
        return str(a) + str(b)

print(safe_add(1, 2))        # 3
print(safe_add("hi", " there"))  # hi there
print(safe_add("count: ", 5))    # count: 5
```

### `KeyError`

딕셔너리 키를 찾을 수 없을 때 발생합니다:

```python
data = {"name": "Alice", "age": 30}

# KeyError
# print(data["email"])  # KeyError: 'email'

# Solution 1: Use get() with default
email = data.get("email", "not provided")

# Solution 2: Check first (LBYL)
if "email" in data:
    email = data["email"]

# Solution 3: Handle the exception (EAFP)
try:
    email = data["email"]
except KeyError:
    email = "not provided"
```

### `IndexError`

시퀀스 인덱스가 범위를 벗어날 때 발생합니다:

```python
items = [10, 20, 30]

# IndexError
# print(items[5])  # IndexError: list index out of range

# Safe access
def safe_get(lst, index, default=None):
    """Safely get an item from a list by index."""
    try:
        return lst[index]
    except IndexError:
        return default

print(safe_get(items, 0))    # 10
print(safe_get(items, 5))    # None
print(safe_get(items, -1))   # 30
```

### `FileNotFoundError`

파일이나 디렉토리가 요청되었지만 존재하지 않을 때 발생합니다:

```python
from pathlib import Path

def read_file_safe(filepath):
    """Read a file, returning None if it does not exist."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except PermissionError:
        print(f"No permission to read: {filepath}")
        return None
```

### `AttributeError`

객체에 요청된 속성이 없을 때 발생합니다:

```python
class User:
    def __init__(self, name):
        self.name = name

user = User("Alice")

# AttributeError
# print(user.email)  # 'User' object has no attribute 'email'

# Safe attribute access
email = getattr(user, "email", "not set")
print(email)  # not set

# Check first
if hasattr(user, "email"):
    print(user.email)
else:
    print("No email attribute")
```

### `ImportError`와 `ModuleNotFoundError`

```python
# ModuleNotFoundError (subclass of ImportError)
try:
    import nonexistent_package
except ModuleNotFoundError:
    print("Package not installed. Install with: pip install nonexistent_package")

# ImportError for specific names
try:
    from math import nonexistent_function
except ImportError as e:
    print(f"Import error: {e}")
```

### `StopIteration`

이터레이터가 소진되었을 때 `next()`에 의해 발생합니다:

```python
my_iter = iter([1, 2, 3])

print(next(my_iter))  # 1
print(next(my_iter))  # 2
print(next(my_iter))  # 3

# StopIteration
try:
    print(next(my_iter))
except StopIteration:
    print("Iterator exhausted")

# Use default value to avoid the exception
my_iter2 = iter([1])
print(next(my_iter2, "done"))  # 1
print(next(my_iter2, "done"))  # done (no exception)
```

---

## 사용자 정의 예외 클래스

### 기본 사용자 정의 예외

```python
class AppError(Exception):
    """Base exception for our application."""
    pass

class ValidationError(AppError):
    """Raised when data validation fails."""
    pass

class DatabaseError(AppError):
    """Raised when a database operation fails."""
    pass

class AuthenticationError(AppError):
    """Raised when authentication fails."""
    pass

# Usage
def validate_email(email):
    if "@" not in email:
        raise ValidationError(f"Invalid email address: {email}")
    return email

try:
    validate_email("invalid-email")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### 추가 데이터를 가진 사용자 정의 예외

```python
class HTTPError(Exception):
    """Exception for HTTP errors with status code and details."""

    def __init__(self, status_code, message, url=None):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"HTTP {status_code}: {message}")

    @property
    def is_client_error(self):
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self):
        return 500 <= self.status_code < 600

# Usage
def fetch_data(url):
    """Simulate fetching data from a URL."""
    # Simulated response
    status = 404
    if status == 404:
        raise HTTPError(404, "Resource not found", url=url)

try:
    fetch_data("https://api.example.com/users/999")
except HTTPError as e:
    print(f"Error: {e}")
    print(f"Status: {e.status_code}")
    print(f"URL: {e.url}")
    print(f"Client error? {e.is_client_error}")
```

### 예외 계층 구조 구축

```python
class PaymentError(Exception):
    """Base class for payment-related errors."""
    pass

class InsufficientFundsError(PaymentError):
    """Raised when account balance is too low."""
    def __init__(self, required, available):
        self.required = required
        self.available = available
        self.shortfall = required - available
        super().__init__(
            f"Insufficient funds: need ${required:.2f}, "
            f"have ${available:.2f} (short ${self.shortfall:.2f})"
        )

class CardDeclinedError(PaymentError):
    """Raised when a card is declined."""
    def __init__(self, reason="Unknown"):
        self.reason = reason
        super().__init__(f"Card declined: {reason}")

class PaymentTimeoutError(PaymentError):
    """Raised when payment processing times out."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Payment timed out after {timeout_seconds}s")

# Usage
def process_payment(amount, balance):
    if amount > balance:
        raise InsufficientFundsError(required=amount, available=balance)
    return balance - amount

try:
    new_balance = process_payment(100.00, 50.00)
except InsufficientFundsError as e:
    print(f"Payment failed: {e}")
    print(f"You need ${e.shortfall:.2f} more")
except PaymentError as e:
    # Catches any other payment error
    print(f"Payment issue: {e}")
```

---

## LBYL vs EAFP

### LBYL: Look Before You Leap (뛰기 전에 살펴보기)

작업을 수행하기 전에 조건을 확인합니다:

```python
# LBYL style
def get_value_lbyl(data, key):
    if isinstance(data, dict) and key in data:
        return data[key]
    return None

# LBYL: Check before division
def divide_lbyl(a, b):
    if b == 0:
        return None
    return a / b

# LBYL: Check before file read
import os
def read_file_lbyl(filepath):
    if os.path.exists(filepath) and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return None
```

### EAFP: Easier to Ask Forgiveness than Permission (허락보다 용서가 쉽다)

작업을 시도하고 예외를 처리합니다:

```python
# EAFP style
def get_value_eafp(data, key):
    try:
        return data[key]
    except (KeyError, TypeError):
        return None

# EAFP: Just try the division
def divide_eafp(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

# EAFP: Just try to read
def read_file_eafp(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, PermissionError):
        return None
```

### 언제 어떤 것을 사용할지

| 요소 | LBYL | EAFP |
|------|------|------|
| 파이썬다운가? | 덜 | 더 (Python 스타일) |
| 경쟁 조건 | 취약함 | 안전함 |
| 성능 (성공 시) | 느림 (확인 + 작업) | 빠름 (작업만) |
| 성능 (실패 시) | 빠름 (예외 없음) | 느림 (예외 오버헤드) |
| 가독성 | 명확한 전제 조건 | 명확한 의도 |

**Python 관례:** EAFP를 선호합니다. 이것이 표준 Python 스타일이며, 경쟁 조건을 피하고 (예: 확인과 읽기 사이에 파일이 삭제될 수 있음), 자연스럽게 읽힙니다.

```python
# LBYL has a race condition:
if os.path.exists("data.txt"):
    # Another process could delete the file RIGHT HERE
    with open("data.txt") as f:  # Could still fail!
        data = f.read()

# EAFP is race-condition-free:
try:
    with open("data.txt") as f:
        data = f.read()
except FileNotFoundError:
    data = None
```

---

## `warnings` 모듈

오류는 아니지만 주의가 필요한 상황에 사용합니다:

```python
import warnings

def deprecated_function():
    """An old function that still works but should not be used."""
    warnings.warn(
        "deprecated_function() is deprecated, use new_function() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return "old result"

def risky_calculation(x):
    """A calculation that might produce unreliable results."""
    if x > 1000:
        warnings.warn(
            f"Large input ({x}) may produce inaccurate results",
            RuntimeWarning,
            stacklevel=2
        )
    return x ** 0.5

# Usage
result = deprecated_function()
# Warning: deprecated_function() is deprecated, use new_function() instead

result = risky_calculation(1_000_000)
# Warning: Large input (1000000) may produce inaccurate results
```

### 경고 제어하기

```python
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Show only DeprecationWarnings
warnings.filterwarnings("default", category=DeprecationWarning)

# Turn warnings into errors (useful for testing)
warnings.filterwarnings("error", category=DeprecationWarning)

try:
    deprecated_function()
except DeprecationWarning as e:
    print(f"Caught deprecated usage: {e}")

# Context manager for temporary warning control
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = deprecated_function()  # Warning suppressed
```

---

## 예외 처리 모범 사례

### 1. 구체적으로 — 너무 광범위하게 잡지 마세요

```python
# BAD: Catches everything, hides bugs
try:
    process_data(data)
except Exception:
    pass  # Silent failure - you will never know what went wrong

# BAD: Also catches KeyboardInterrupt
try:
    long_running_task()
except:  # Bare except catches EVERYTHING including Ctrl+C
    pass

# GOOD: Catch specific exceptions
try:
    process_data(data)
except ValueError as e:
    logger.warning(f"Invalid data: {e}")
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    retry_connection()
```

### 2. 흐름 제어에 예외를 사용하지 마세요

```python
# BAD: Using exceptions for normal flow
def find_item_bad(items, target):
    try:
        return items.index(target)
    except ValueError:
        return -1

# BETTER: Direct approach for expected outcomes
def find_item_good(items, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# ACCEPTABLE: EAFP for dict/list access where missing keys are common
def get_nested(data, *keys, default=None):
    """Safely get a nested value from dictionaries."""
    current = data
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError, IndexError):
        return default

config = {"database": {"host": "localhost", "port": 5432}}
print(get_nested(config, "database", "host"))         # localhost
print(get_nested(config, "database", "password"))     # None
print(get_nested(config, "missing", "key"))            # None
```

### 3. 리소스를 정리하세요

```python
# GOOD: Use context managers
with open("data.txt", "r", encoding="utf-8") as f:
    data = f.read()
# File is closed automatically

# GOOD: Use finally for non-context-manager resources
connection = None
try:
    connection = create_connection()
    result = connection.execute(query)
except DatabaseError as e:
    logger.error(f"Query failed: {e}")
    result = None
finally:
    if connection is not None:
        connection.close()
```

### 4. 예외를 올바르게 로깅하세요

```python
import logging
import traceback

logger = logging.getLogger(__name__)

def process_request(request):
    """Process a request with proper error logging."""
    try:
        result = handle(request)
        return result
    except ValueError as e:
        # Log with context but handle gracefully
        logger.warning(f"Invalid request data: {e}")
        return {"error": "Invalid input", "details": str(e)}
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        return {"error": "Internal server error"}
```

### 5. 유용한 오류 메시지를 제공하세요

```python
# BAD: Vague error
def connect_bad(host, port):
    if not host:
        raise ValueError("Invalid input")

# GOOD: Specific, actionable error
def connect_good(host, port):
    if not host:
        raise ValueError(
            f"Host cannot be empty. Got: {host!r}. "
            "Provide a hostname like 'localhost' or '192.168.1.1'"
        )
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(
            f"Port must be an integer between 1 and 65535, got: {port!r}"
        )
```

### 6. 애플리케이션 로직에 사용자 정의 예외를 사용하세요

```python
# BAD: Reusing built-in exceptions for business logic
def withdraw(account, amount):
    if amount > account.balance:
        raise ValueError("Not enough money")  # Ambiguous

# GOOD: Custom exceptions are clear and catchable
class InsufficientBalanceError(Exception):
    def __init__(self, required, available):
        self.required = required
        self.available = available
        super().__init__(
            f"Cannot withdraw ${required:.2f}: "
            f"only ${available:.2f} available"
        )

def withdraw(account, amount):
    if amount > account.balance:
        raise InsufficientBalanceError(amount, account.balance)
```

### 7. 예외를 조용히 무시하지 마세요

```python
# BAD: Silent pass
try:
    send_notification(user, message)
except Exception:
    pass  # If this fails, nobody will ever know

# BETTER: At minimum, log it
try:
    send_notification(user, message)
except NotificationError as e:
    logger.warning(f"Failed to send notification to {user}: {e}")
    # Continue execution - notification is non-critical
```

---

## 실용 예제: 견고한 데이터 처리

```python
"""A robust data processing pipeline with proper exception handling."""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass

class DataLoadError(DataProcessingError):
    """Error loading data from a source."""
    pass

class DataValidationError(DataProcessingError):
    """Error validating data."""
    def __init__(self, record_number, field, message):
        self.record_number = record_number
        self.field = field
        super().__init__(
            f"Record #{record_number}, field '{field}': {message}"
        )

class DataExportError(DataProcessingError):
    """Error exporting processed data."""
    pass

def load_data(filepath):
    """Load data from a JSON or CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise DataLoadError(f"File not found: {filepath}")

    try:
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            raise DataLoadError(f"Unsupported file format: {path.suffix}")
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {filepath}: {e}") from e
    except csv.Error as e:
        raise DataLoadError(f"Invalid CSV in {filepath}: {e}") from e

def validate_record(record, index):
    """Validate a single data record."""
    required_fields = ["name", "email", "age"]
    for field in required_fields:
        if field not in record:
            raise DataValidationError(index, field, "Missing required field")

    if not isinstance(record.get("name"), str) or not record["name"].strip():
        raise DataValidationError(index, "name", "Name must be a non-empty string")

    if "@" not in str(record.get("email", "")):
        raise DataValidationError(
            index, "email",
            f"Invalid email: {record.get('email')}"
        )

    try:
        age = int(record["age"])
        if age < 0 or age > 150:
            raise DataValidationError(
                index, "age",
                f"Age must be between 0 and 150, got {age}"
            )
    except (ValueError, TypeError):
        raise DataValidationError(
            index, "age",
            f"Age must be a number, got {record['age']!r}"
        )

def process_pipeline(input_file, output_file):
    """Run the complete data processing pipeline."""
    errors = []
    valid_records = []

    # Step 1: Load data
    try:
        raw_data = load_data(input_file)
        logger.info(f"Loaded {len(raw_data)} records from {input_file}")
    except DataLoadError as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Step 2: Validate records
    for i, record in enumerate(raw_data, 1):
        try:
            validate_record(record, i)
            valid_records.append(record)
        except DataValidationError as e:
            errors.append(str(e))
            logger.warning(f"Validation error: {e}")

    logger.info(
        f"Validation complete: {len(valid_records)} valid, "
        f"{len(errors)} errors"
    )

    # Step 3: Export results
    try:
        output = Path(output_file)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({
                "processed_at": datetime.now().isoformat(),
                "total_records": len(raw_data),
                "valid_records": len(valid_records),
                "errors": len(errors),
                "data": valid_records,
                "error_details": errors,
            }, f, indent=2)
        logger.info(f"Results written to {output_file}")
        return True
    except OSError as e:
        raise DataExportError(f"Cannot write to {output_file}: {e}") from e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = process_pipeline("input.json", "output.json")
    print(f"Pipeline {'succeeded' if success else 'failed'}")
```

---

## 요약

| 개념 | 목적 | 구문 |
|------|------|------|
| `try/except` | 예외를 잡고 처리 | `try: ... except ExcType as e: ...` |
| `else` | 성공 시에만 실행되는 코드 | `try: ... except: ... else: ...` |
| `finally` | 항상 실행되는 코드 | `try: ... finally: ...` |
| `raise` | 예외 던지기 | `raise ValueError("msg")` |
| `raise from` | 예외 연쇄 | `raise NewError() from original` |
| 사용자 정의 예외 | 애플리케이션별 오류 | `class MyError(Exception): ...` |
| EAFP | 먼저 시도하고 오류 처리 | Python의 선호 스타일 |
| LBYL | 먼저 확인하고 실행 | 확인이 저렴할 때 사용 |

핵심 내용:
- **특정 예외를 잡으세요** — 단독 `except:`나 `BaseException` 잡기를 절대 사용하지 마세요
- **`else`와 `finally`를 사용하여** 오류 처리를 명확하게 구성하세요
- 파이썬다운 코드를 위해 LBYL보다 **EAFP를 선호**하세요
- 애플리케이션의 도메인별 오류를 위한 **사용자 정의 예외를 만드세요**
- `with` 문이나 `finally`를 사용하여 **항상 리소스를 정리**하세요
- 문제 진단에 충분한 컨텍스트와 함께 **예외를 로깅**하세요
- 무엇이 잘못되었고 어떻게 수정하는지 알려주는 **유용한 오류 메시지를 제공**하세요

---

## 추가 자료

- [Python 오류와 예외 (공식 튜토리얼)](https://docs.python.org/3/tutorial/errors.html)
- [내장 예외 (공식 문서)](https://docs.python.org/3/library/exceptions.html)
- [PEP 3134 — 예외 연쇄](https://peps.python.org/pep-3134/)
- [Effective Python: 예외의 차이점 알기](https://effectivepython.com/)

---

**이전**: [파일 입출력](./11_File_IO.md) | **다음**: [표준 라이브러리 핵심](./13_Standard_Library_Essentials.md)

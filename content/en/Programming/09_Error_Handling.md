# Error Handling Strategies

**Previous**: [Clean Code & Code Smells](./08_Clean_Code.md) | **Next**: [Testing Fundamentals](./10_Testing_Fundamentals.md)

> **Topic**: Programming
> **Lesson**: 9 of 16
> **Prerequisites**: Functions and Methods, Data Structures, Program Flow Control

## Learning Objectives

After completing this lesson, you will be able to:

1. Classify errors by type (syntax, runtime, logic, resource, network, user input) and select appropriate handling strategies for each
2. Implement try/catch/finally exception handling in Python, JavaScript, Java, and C++
3. Design custom exception hierarchies for domain-specific error scenarios
4. Compare exception-based and return-value-based error handling (Go, Rust) and choose the right approach for a given context
5. Apply defensive programming techniques including input validation, preconditions, postconditions, and invariants
6. Implement the fail-fast principle and graceful degradation patterns such as circuit breakers and retry with exponential backoff
7. Write clear, actionable error messages tailored to both developer and end-user audiences
8. Use structured logging with appropriate severity levels (DEBUG through CRITICAL) to support production debugging

---

## Introduction

Error handling is one of the most critical aspects of software development, yet it's often treated as an afterthought. Poor error handling leads to crashes, security vulnerabilities, data corruption, and frustrated users. Good error handling makes software robust, maintainable, and user-friendly.

This lesson explores error handling strategies used across different programming languages and paradigms, from traditional exceptions to modern Result types, and teaches you when and how to apply each approach.

---

## Why Error Handling Matters

### Robustness

Software encounters errors constantly: network failures, invalid user input, missing files, out-of-memory conditions. Robust software anticipates and handles these errors gracefully rather than crashing.

### User Experience

When errors occur, users need clear feedback about what went wrong and what they can do about it. Compare these two experiences:

**Poor error handling:**
```
ERROR: Exception in thread "main" java.lang.NullPointerException
    at com.example.App.main(App.java:42)
```

**Good error handling:**
```
Unable to save document: The file "report.pdf" is currently open in another program.
Please close it and try again.
```

### Debugging and Maintenance

Informative error messages and proper logging make debugging exponentially easier. When something goes wrong in production, comprehensive error context is invaluable.

### System Integrity

Proper error handling prevents errors from cascading through the system, corrupting data, or leaving resources (files, connections, locks) in inconsistent states.

---

## Types of Errors

Understanding the categories of errors helps you choose appropriate handling strategies:

### 1. Syntax Errors

Errors in the code structure that prevent compilation or interpretation. These are caught before runtime:

```python
# Python syntax error
if x > 5
    print("Greater")  # Missing colon
```

### 2. Runtime Errors

Errors that occur during program execution:

```python
# Division by zero
result = 10 / 0  # ZeroDivisionError

# File not found
file = open("missing.txt")  # FileNotFoundError
```

### 3. Logic Errors

The program runs without crashing, but produces incorrect results:

```python
# Intended to calculate average, but has logic error
def average(numbers):
    return sum(numbers) / len(numbers) + 1  # Should not add 1!
```

### 4. Resource Errors

Memory exhaustion, disk full, too many open files:

```python
# Opening too many files without closing
files = []
for i in range(10000):
    files.append(open(f"file{i}.txt", "w"))  # May exhaust file descriptors
```

### 5. Network Errors

Timeouts, connection refused, DNS failures, packet loss.

### 6. User Input Errors

Invalid data from users: wrong format, out-of-range values, missing required fields.

---

## Exception-Based Error Handling

Most modern languages (Python, Java, C++, JavaScript, C#) use exceptions as their primary error handling mechanism.

### Try/Catch/Finally Mechanism

The basic structure separates normal code from error handling code:

**Python:**
```python
try:
    file = open("config.json", "r")
    data = file.read()
    config = json.loads(data)
    print(f"Loaded {len(config)} settings")
except FileNotFoundError:
    print("Configuration file not found, using defaults")
    config = default_config()
except json.JSONDecodeError as e:
    print(f"Invalid JSON in config file: {e}")
    config = default_config()
finally:
    if 'file' in locals():
        file.close()  # Always executed, even if exception occurs
```

**JavaScript:**
```javascript
try {
    const response = await fetch('https://api.example.com/data');
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log(`Received ${data.length} items`);
} catch (error) {
    if (error instanceof TypeError) {
        console.error('Network error:', error.message);
    } else {
        console.error('Failed to fetch data:', error.message);
    }
} finally {
    console.log('Request completed');
}
```

**Java:**
```java
FileReader reader = null;
try {
    reader = new FileReader("data.txt");
    int character;
    while ((character = reader.read()) != -1) {
        System.out.print((char) character);
    }
} catch (FileNotFoundException e) {
    System.err.println("File not found: " + e.getMessage());
} catch (IOException e) {
    System.err.println("Error reading file: " + e.getMessage());
} finally {
    if (reader != null) {
        try {
            reader.close();
        } catch (IOException e) {
            System.err.println("Error closing file: " + e.getMessage());
        }
    }
}
```

**C++:**
```cpp
#include <iostream>
#include <fstream>
#include <stdexcept>

try {
    std::ifstream file("data.txt");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }

    file.close();
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Exception Hierarchy

Languages organize exceptions in hierarchies:

**Python exception hierarchy (simplified):**
```
BaseException
├── Exception
│   ├── ArithmeticError
│   │   ├── ZeroDivisionError
│   │   └── OverflowError
│   ├── LookupError
│   │   ├── IndexError
│   │   └── KeyError
│   ├── OSError
│   │   ├── FileNotFoundError
│   │   └── PermissionError
│   └── ValueError
└── KeyboardInterrupt
```

**Java: Checked vs Unchecked Exceptions**

Java distinguishes between:
- **Checked exceptions**: Must be declared in method signature or caught (IOException, SQLException)
- **Unchecked exceptions**: Runtime exceptions that don't require explicit handling (NullPointerException, IllegalArgumentException)

```java
// Checked exception - must declare or catch
public void readFile(String path) throws IOException {
    FileReader reader = new FileReader(path);
    // ...
}

// Unchecked exception - optional handling
public int divide(int a, int b) {
    if (b == 0) {
        throw new IllegalArgumentException("Divisor cannot be zero");
    }
    return a / b;
}
```

### Custom Exceptions

Create domain-specific exceptions for clarity:

**Python:**
```python
class InsufficientFundsError(Exception):
    """Raised when account balance is insufficient for withdrawal"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Cannot withdraw ${amount}: balance is only ${balance}")

class Account:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return amount

# Usage
account = Account(100)
try:
    account.withdraw(150)
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
    print(f"Available balance: ${e.balance}")
```

**JavaScript:**
```javascript
class ValidationError extends Error {
    constructor(field, message) {
        super(message);
        this.name = 'ValidationError';
        this.field = field;
    }
}

function validateEmail(email) {
    if (!email.includes('@')) {
        throw new ValidationError('email', 'Email must contain @ symbol');
    }
}

try {
    validateEmail('invalid-email');
} catch (error) {
    if (error instanceof ValidationError) {
        console.error(`Validation failed for ${error.field}: ${error.message}`);
    } else {
        throw error;  // Re-throw unknown errors
    }
}
```

### When to Catch, When to Propagate

**Catch when:**
- You can handle the error meaningfully
- You can provide a fallback/default value
- You need to log the error locally
- You're at a system boundary (API endpoint, UI layer)

**Propagate when:**
- You can't do anything useful with the error
- A higher layer has better context for handling
- The error represents a programming bug (should crash)

```python
# BAD: Catching without handling
def read_user_data(user_id):
    try:
        return database.query(f"SELECT * FROM users WHERE id={user_id}")
    except Exception:
        pass  # Silent failure - data loss!

# GOOD: Propagate to caller
def read_user_data(user_id):
    return database.query(f"SELECT * FROM users WHERE id={user_id}")
    # Let caller decide how to handle database errors

# GOOD: Catch and handle meaningfully
def get_user_or_default(user_id):
    try:
        return read_user_data(user_id)
    except UserNotFoundError:
        return create_guest_user()
```

### Anti-Pattern: Pokemon Exception Handling

**"Gotta catch 'em all!"** – catching every exception indiscriminately is dangerous:

```python
# TERRIBLE: Catches everything, hides bugs
try:
    result = process_data(user_input)
    save_to_database(result)
    send_notification(user_email)
except Exception:
    print("Something went wrong")  # Which operation failed? What error?
```

**Better approach:**
```python
try:
    result = process_data(user_input)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    return {"error": "Invalid input format"}

try:
    save_to_database(result)
except DatabaseError as e:
    logger.error(f"Database save failed: {e}")
    return {"error": "Unable to save data, please try again"}

try:
    send_notification(user_email)
except EmailError as e:
    # Non-critical: log but don't fail the request
    logger.warning(f"Failed to send notification: {e}")
```

### Anti-Pattern: Exceptions for Control Flow

Don't use exceptions for normal program flow:

```python
# BAD: Using exceptions for control flow
def find_user(username):
    try:
        return users[username]
    except KeyError:
        raise UserNotFoundError()

# GOOD: Use normal conditional logic
def find_user(username):
    if username in users:
        return users[username]
    else:
        return None  # or raise UserNotFoundError if this is exceptional
```

Exceptions should represent **exceptional conditions**, not expected outcomes. They're expensive (in performance) and make code flow harder to follow.

### Best Practice: Catch Specific Exceptions

```python
# BAD: Too broad
try:
    data = json.loads(text)
except Exception:
    print("Error parsing JSON")

# GOOD: Specific exception
try:
    data = json.loads(text)
except json.JSONDecodeError as e:
    print(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")
```

---

## Return-Value Based Error Handling

Some languages prefer explicit error returns over exceptions.

### C-Style Error Codes

Traditional C uses return codes and `errno`:

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main() {
    FILE *file = fopen("missing.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s (errno=%d)\n",
                strerror(errno), errno);
        return 1;
    }

    char buffer[256];
    if (fgets(buffer, sizeof(buffer), file) == NULL) {
        if (ferror(file)) {
            fprintf(stderr, "Error reading file\n");
        }
        fclose(file);
        return 1;
    }

    printf("Read: %s", buffer);
    fclose(file);
    return 0;
}
```

**Drawbacks:**
- Easy to ignore error codes
- Error handling code mixed with normal code
- No automatic propagation (must check every call)

### Go-Style Multiple Return Values

Go returns `(value, error)` pairs:

```go
package main

import (
    "fmt"
    "os"
    "strconv"
)

func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

func parseAndDivide(numStr, denStr string) (int, error) {
    numerator, err := strconv.Atoi(numStr)
    if err != nil {
        return 0, fmt.Errorf("invalid numerator: %w", err)
    }

    denominator, err := strconv.Atoi(denStr)
    if err != nil {
        return 0, fmt.Errorf("invalid denominator: %w", err)
    }

    result, err := divide(numerator, denominator)
    if err != nil {
        return 0, fmt.Errorf("division failed: %w", err)
    }

    return result, nil
}

func main() {
    result, err := parseAndDivide("10", "2")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        os.Exit(1)
    }
    fmt.Printf("Result: %d\n", result)
}
```

**Advantages:**
- Errors are explicit in the type signature
- Impossible to ignore errors (compiler forces checking)
- Clear separation of success and error paths

### Rust Result<T, E> and Option<T>

Rust uses algebraic data types for error handling:

```rust
use std::fs::File;
use std::io::{self, Read};

// Result<T, E>: either Ok(value) or Err(error)
fn read_username_from_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;  // ? operator propagates errors
    let mut username = String::new();
    file.read_to_string(&mut username)?;
    Ok(username)
}

// Option<T>: either Some(value) or None
fn find_user(id: u32) -> Option<String> {
    let users = vec![
        (1, "Alice"),
        (2, "Bob"),
        (3, "Charlie"),
    ];

    users.iter()
        .find(|(user_id, _)| *user_id == id)
        .map(|(_, name)| name.to_string())
}

fn main() {
    // Handling Result
    match read_username_from_file("user.txt") {
        Ok(username) => println!("Username: {}", username),
        Err(e) => eprintln!("Error reading file: {}", e),
    }

    // Handling Option
    match find_user(2) {
        Some(name) => println!("Found: {}", name),
        None => println!("User not found"),
    }

    // Using unwrap_or for defaults
    let username = find_user(99).unwrap_or_else(|| "Guest".to_string());
    println!("Logged in as: {}", username);
}
```

**The ? operator:**
```rust
fn calculate() -> Result<i32, String> {
    let x = parse_number("42")?;  // If Err, return early
    let y = parse_number("10")?;  // If Err, return early
    Ok(x + y)                     // If both Ok, return result
}
```

---

## Comparison: Exceptions vs Result Types

| Aspect | Exceptions | Result Types |
|--------|-----------|--------------|
| **Visibility** | Hidden in control flow | Explicit in type signature |
| **Enforcement** | Can be ignored | Compiler forces handling (Rust) |
| **Performance** | Slower (stack unwinding) | Zero-cost (Rust) |
| **Propagation** | Automatic | Manual (but aided by ?, ??, etc.) |
| **Mixing with normal flow** | Clean separation | Can clutter code with checks |
| **Best for** | Rare, exceptional failures | Expected errors, data validation |

**When to use exceptions:**
- Unexpected, exceptional conditions
- Errors that rarely happen
- When you want automatic propagation up the call stack
- Languages where they're idiomatic (Java, Python, C#)

**When to use Result types:**
- Expected errors (validation, parsing)
- When error handling should be explicit
- Performance-critical code
- Languages that support them well (Rust, Go, Haskell)

---

## Defensive Programming

Defensive programming means writing code that anticipates and handles unexpected conditions.

### Input Validation at System Boundaries

Validate all external input: user input, network data, file contents, environment variables.

```python
def create_user(username, email, age):
    # Validate at the boundary
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters")

    if not email or '@' not in email:
        raise ValueError("Invalid email address")

    if not isinstance(age, int) or age < 0 or age > 150:
        raise ValueError("Age must be between 0 and 150")

    # Now we can trust the data internally
    user = User(username=username, email=email, age=age)
    return user.save()
```

### Preconditions, Postconditions, Invariants

**Preconditions:** What must be true before a function executes
**Postconditions:** What must be true after a function executes
**Invariants:** What must always be true (e.g., for a data structure)

```python
class BankAccount:
    def __init__(self, initial_balance):
        # Precondition
        assert initial_balance >= 0, "Initial balance cannot be negative"
        self.balance = initial_balance
        # Invariant: balance >= 0 must always hold

    def withdraw(self, amount):
        # Preconditions
        assert amount > 0, "Withdrawal amount must be positive"
        assert amount <= self.balance, "Insufficient funds"

        self.balance -= amount

        # Postcondition: balance decreased by amount
        # Invariant: balance still >= 0
        assert self.balance >= 0

        return amount
```

### Design by Contract (Bertrand Meyer)

Formalize the relationship between a class and its clients:
- **Client obligations** (preconditions): What the client must ensure
- **Class obligations** (postconditions): What the class guarantees
- **Class invariants**: What always holds for the class

### Assertions: Development vs Production

**Assertions are for catching programming bugs, not handling runtime errors.**

```python
# Development: assertions enabled
def binary_search(arr, target):
    assert len(arr) > 0, "Array must not be empty"
    assert all(arr[i] <= arr[i+1] for i in range(len(arr)-1)), "Array must be sorted"
    # ... binary search implementation

# Production: assertions may be disabled (python -O)
# Use exceptions for runtime validation
def process_payment(amount):
    if amount <= 0:
        raise ValueError("Payment amount must be positive")  # NOT assert
```

**Guidelines:**
- **Use assertions for:** Internal consistency checks, verifying assumptions, debugging
- **Use exceptions for:** User input validation, external system failures, runtime errors

---

## Fail-Fast Principle

Detect errors as early as possible and fail immediately, rather than continuing with invalid state.

```python
# BAD: Failing slowly
def process_orders(orders):
    results = []
    for order in orders:
        if order.is_valid():
            results.append(process(order))
        else:
            results.append(None)  # Continues with invalid data
    return results

# GOOD: Fail fast
def process_orders(orders):
    # Validate all orders first
    for order in orders:
        if not order.is_valid():
            raise InvalidOrderError(f"Invalid order: {order.id}")

    # All orders are valid, process them
    return [process(order) for order in orders]
```

**Benefits:**
- Errors are caught close to their source
- Prevents cascading failures
- Makes debugging easier
- Prevents data corruption

---

## Graceful Degradation

When components fail, provide fallback behavior rather than complete system failure.

```javascript
// Load configuration with fallbacks
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        return await response.json();
    } catch (error) {
        console.warn('Failed to load remote config, using defaults:', error);
        return DEFAULT_CONFIG;
    }
}

// Retry with exponential backoff
async function fetchWithRetry(url, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fetch(url);
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            const delay = Math.pow(2, i) * 1000;  // 1s, 2s, 4s
            console.log(`Retry ${i + 1}/${maxRetries} after ${delay}ms`);
            await sleep(delay);
        }
    }
}
```

**Circuit Breaker Pattern:**

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise e

# Usage
breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def unreliable_api_call():
    breaker.call(requests.get, 'https://flaky-api.com/data')
```

---

## Error Messages

### For Developers: Context and Debugging

Include enough information to debug the problem:

```python
class DatabaseError(Exception):
    def __init__(self, query, params, original_error):
        self.query = query
        self.params = params
        self.original_error = original_error

        message = f"""
Database query failed:
  Query: {query}
  Parameters: {params}
  Original error: {original_error}
  Timestamp: {datetime.now()}
"""
        super().__init__(message)

# Usage
try:
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
except Exception as e:
    raise DatabaseError(
        query="SELECT * FROM users WHERE id = %s",
        params=(user_id,),
        original_error=e
    )
```

### For Users: Clear and Actionable

```python
# BAD: Technical jargon
"HTTP 504 Gateway Timeout"

# GOOD: User-friendly explanation
"We couldn't load your data right now. Please check your internet connection and try again."

# BETTER: Actionable guidance
"We couldn't connect to the server. Here's what you can try:
 • Check your internet connection
 • Try refreshing the page
 • If the problem persists, contact support@example.com"
```

---

## Logging Levels

Modern logging frameworks provide severity levels:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEBUG: Detailed information for diagnosing problems
logger.debug(f"Processing user {user_id}, attempt {attempt_num}")

# INFO: General informational messages
logger.info(f"User {user_id} logged in successfully")

# WARNING: Something unexpected but not an error
logger.warning(f"Cache miss for key {key}, fetching from database")

# ERROR: An error occurred, but the application continues
logger.error(f"Failed to send email to {email}: {error}")

# CRITICAL: Serious error, application may not continue
logger.critical("Database connection pool exhausted, shutting down")
```

**When to log:**
- **DEBUG:** Function entry/exit, variable values, detailed flow
- **INFO:** Significant events (user actions, system state changes)
- **WARNING:** Unexpected but handled situations, deprecation notices
- **ERROR:** Caught exceptions, failed operations
- **CRITICAL:** System failures, unrecoverable errors

---

## Retry Patterns

### Exponential Backoff

```python
import time
import random

def exponential_backoff(func, max_retries=5, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, give up

            # Calculate delay: base * 2^attempt + jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            time.sleep(delay)

# Usage
result = exponential_backoff(lambda: requests.get('https://api.example.com/data'))
```

### Jitter

Add randomness to prevent thundering herd problem:

```python
def backoff_with_jitter(attempt, base=1, max_delay=60):
    # Full jitter: randomize between 0 and calculated delay
    delay = min(max_delay, base * (2 ** attempt))
    return random.uniform(0, delay)
```

---

## Summary

**Key Principles:**
1. **Anticipate errors** – They will happen
2. **Fail fast** – Detect early, fail loudly
3. **Be specific** – Catch specific exceptions, provide detailed context
4. **Separate concerns** – Don't mix error handling with business logic
5. **Provide context** – Include enough information for debugging
6. **Think about users** – Make error messages helpful and actionable
7. **Don't silently fail** – Log errors, even if you handle them
8. **Validate at boundaries** – Check all external input
9. **Choose the right tool** – Exceptions for exceptional cases, Results for expected errors

---

## Exercises

### Exercise 1: Refactor Poor Error Handling

Refactor this code with better error handling:

```python
def process_file(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    result = json.loads(data)
    return result['value'] * 2

result = process_file('data.json')
print(result)
```

**Issues to fix:**
- No error handling
- File not closed if exception occurs
- Assumes 'value' key exists
- Assumes 'value' is numeric

### Exercise 2: Design an Error Strategy

You're building a REST API for a payment system. Design an error handling strategy that addresses:
- Invalid request data (missing fields, wrong types)
- Database connection failures
- External payment gateway errors (timeout, rejection)
- Insufficient funds
- Logging requirements
- Error responses to clients

Write example code showing how you'd handle each scenario.

### Exercise 3: Implement Retry Logic

Write a function `retry_with_circuit_breaker` that:
- Retries a function up to N times with exponential backoff
- Opens a circuit breaker after consecutive failures
- Provides meaningful logging at each step
- Returns a Result type or raises an informative exception

Test it with a mock function that fails randomly.

### Exercise 4: Error Message Improvement

Improve these error messages:

1. `"Error 42"`
2. `"Invalid input"`
3. `"NullPointerException at line 127"`
4. `"Cannot process request"`

For each, write:
- A developer-friendly version (detailed, technical)
- A user-friendly version (clear, actionable)

### Exercise 5: Exception vs Result Type

Implement a function `parse_date(text: str)` that parses a date string in the format "YYYY-MM-DD":
1. Once using exceptions
2. Once using a Result type (using a library or your own implementation)

Compare the two approaches. When would you use each?

---

**Previous**: [Clean Code & Code Smells](./08_Clean_Code.md) | **Next**: [Testing Fundamentals](./10_Testing_Fundamentals.md)

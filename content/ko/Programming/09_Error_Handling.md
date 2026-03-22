# 에러 처리 전략(Error Handling Strategies)

> **주제**: Programming
> **레슨**: 16 중 9
> **선수 지식**: 함수와 메서드, 자료 구조, 프로그램 흐름 제어
> **목표**: 프로그래밍 패러다임 전반에 걸친 에러 처리 전략을 이해하고, 예외와 반환 값을 언제 사용할지 배우며, 방어적 프로그래밍 기법을 적용하여 견고한 소프트웨어를 구축합니다.

---

## 소개

에러 처리는 소프트웨어 개발에서 가장 중요한 측면 중 하나이지만, 종종 나중에 고려되곤 합니다. 잘못된 에러 처리는 충돌, 보안 취약점, 데이터 손상, 그리고 좌절한 사용자로 이어집니다. 좋은 에러 처리는 소프트웨어를 견고하고, 유지보수 가능하며, 사용자 친화적으로 만듭니다.

이 레슨은 다양한 프로그래밍 언어와 패러다임에서 사용되는 에러 처리 전략을 탐구하며, 전통적인 예외부터 현대적인 Result 타입까지 다루고, 각 접근 방식을 언제 어떻게 적용할지 가르칩니다.

---

## 에러 처리가 중요한 이유

### 견고성(Robustness)

소프트웨어는 끊임없이 에러를 만납니다: 네트워크 장애, 잘못된 사용자 입력, 파일 누락, 메모리 부족 상황. 견고한 소프트웨어는 충돌하기보다 이러한 에러를 예상하고 우아하게 처리합니다.

### 사용자 경험(User Experience)

에러가 발생할 때, 사용자는 무엇이 잘못되었고 어떻게 해야 하는지에 대한 명확한 피드백이 필요합니다. 다음 두 경험을 비교해보세요:

**잘못된 에러 처리:**
```
ERROR: Exception in thread "main" java.lang.NullPointerException
    at com.example.App.main(App.java:42)
```

**좋은 에러 처리:**
```
문서를 저장할 수 없습니다: "report.pdf" 파일이 현재 다른 프로그램에서 열려 있습니다.
파일을 닫고 다시 시도해주세요.
```

### 디버깅과 유지보수(Debugging and Maintenance)

유익한 에러 메시지와 적절한 로깅은 디버깅을 기하급수적으로 쉽게 만듭니다. 프로덕션에서 문제가 발생할 때, 포괄적인 에러 컨텍스트는 매우 귀중합니다.

### 시스템 무결성(System Integrity)

적절한 에러 처리는 에러가 시스템 전체로 확산되어 데이터를 손상시키거나 리소스(파일, 연결, 락)를 일관되지 않은 상태로 남기는 것을 방지합니다.

---

## 에러의 유형

에러의 범주를 이해하면 적절한 처리 전략을 선택하는 데 도움이 됩니다:

### 1. 구문 에러(Syntax Errors)

코드 구조의 에러로 컴파일이나 해석을 방해합니다. 런타임 전에 발견됩니다:

```python
# Python syntax error
if x > 5
    print("Greater")  # Missing colon
```

### 2. 런타임 에러(Runtime Errors)

프로그램 실행 중에 발생하는 에러:

```python
# Division by zero
result = 10 / 0  # ZeroDivisionError

# File not found
file = open("missing.txt")  # FileNotFoundError
```

### 3. 논리 에러(Logic Errors)

프로그램이 충돌하지 않고 실행되지만, 잘못된 결과를 생성합니다:

```python
# Intended to calculate average, but has logic error
def average(numbers):
    return sum(numbers) / len(numbers) + 1  # Should not add 1!
```

### 4. 리소스 에러(Resource Errors)

메모리 소진, 디스크 가득 참, 너무 많은 열린 파일:

```python
# Opening too many files without closing
files = []
for i in range(10000):
    files.append(open(f"file{i}.txt", "w"))  # May exhaust file descriptors
```

### 5. 네트워크 에러(Network Errors)

타임아웃, 연결 거부, DNS 실패, 패킷 손실.

### 6. 사용자 입력 에러(User Input Errors)

사용자로부터의 잘못된 데이터: 잘못된 형식, 범위를 벗어난 값, 필수 필드 누락.

---

## 예외 기반 에러 처리

대부분의 현대 언어(Python, Java, C++, JavaScript, C#)는 예외를 기본 에러 처리 메커니즘으로 사용합니다.

### Try/Catch/Finally 메커니즘

기본 구조는 정상 코드와 에러 처리 코드를 분리합니다:

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

### 예외 계층(Exception Hierarchy)

언어는 예외를 계층 구조로 조직합니다:

**Python 예외 계층(간소화):**
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

**Java: 체크 vs 언체크 예외(Checked vs Unchecked Exceptions)**

Java는 구분합니다:
- **체크 예외(Checked exceptions)**: 메서드 시그니처에 선언하거나 catch해야 함 (IOException, SQLException)
- **언체크 예외(Unchecked exceptions)**: 명시적 처리가 필요 없는 런타임 예외 (NullPointerException, IllegalArgumentException)

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

### 커스텀 예외(Custom Exceptions)

명확성을 위해 도메인별 예외를 생성합니다:

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

### 언제 잡고, 언제 전파할지

**잡아야 할 때:**
- 에러를 의미 있게 처리할 수 있을 때
- 대체/기본값을 제공할 수 있을 때
- 에러를 로컬에서 로깅해야 할 때
- 시스템 경계에 있을 때 (API 엔드포인트, UI 레이어)

**전파해야 할 때:**
- 에러로 유용한 작업을 할 수 없을 때
- 상위 레이어가 처리를 위한 더 나은 컨텍스트를 가지고 있을 때
- 에러가 프로그래밍 버그를 나타낼 때 (충돌해야 함)

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

### 안티패턴: 포켓몬 예외 처리(Pokemon Exception Handling)

**"모두 다 잡아라!"** – 모든 예외를 무분별하게 잡는 것은 위험합니다:

```python
# TERRIBLE: Catches everything, hides bugs
try:
    result = process_data(user_input)
    save_to_database(result)
    send_notification(user_email)
except Exception:
    print("Something went wrong")  # Which operation failed? What error?
```

**더 나은 접근:**
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

### 안티패턴: 제어 흐름용 예외(Exceptions for Control Flow)

정상적인 프로그램 흐름에 예외를 사용하지 마세요:

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

예외는 **예외적인 조건**을 나타내야 하며, 예상된 결과가 아닙니다. 예외는 (성능 측면에서) 비용이 높고 코드 흐름을 따라가기 어렵게 만듭니다.

### 모범 사례: 특정 예외 잡기

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

## 반환값 기반 에러 처리

일부 언어는 예외보다 명시적 에러 반환을 선호합니다.

### C 스타일 에러 코드

전통적인 C는 반환 코드와 `errno`를 사용합니다:

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

**단점:**
- 에러 코드를 무시하기 쉬움
- 에러 처리 코드가 정상 코드와 섞임
- 자동 전파 없음 (모든 호출을 확인해야 함)

### Go 스타일 다중 반환값

Go는 `(값, 에러)` 쌍을 반환합니다:

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

**장점:**
- 에러가 타입 시그니처에 명시적임
- 에러를 무시할 수 없음 (컴파일러가 확인 강제)
- 성공과 에러 경로의 명확한 분리

### Rust Result<T, E>와 Option<T>

Rust는 에러 처리를 위해 대수적 데이터 타입을 사용합니다:

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

**? 연산자:**
```rust
fn calculate() -> Result<i32, String> {
    let x = parse_number("42")?;  // If Err, return early
    let y = parse_number("10")?;  // If Err, return early
    Ok(x + y)                     // If both Ok, return result
}
```

---

## 비교: 예외 vs Result 타입

| 측면 | 예외(Exceptions) | Result 타입 |
|--------|-----------|--------------|
| **가시성(Visibility)** | 제어 흐름에 숨겨짐 | 타입 시그니처에 명시적 |
| **강제성(Enforcement)** | 무시 가능 | 컴파일러가 처리 강제 (Rust) |
| **성능(Performance)** | 느림 (스택 풀기) | 제로 비용 (Rust) |
| **전파(Propagation)** | 자동 | 수동 (하지만 ?, ?? 등으로 지원) |
| **정상 흐름과 섞임** | 깔끔한 분리 | 확인으로 코드가 복잡해질 수 있음 |
| **최적 용도** | 드물고 예외적인 실패 | 예상된 에러, 데이터 검증 |

**예외를 사용할 때:**
- 예상치 못한, 예외적인 조건
- 거의 발생하지 않는 에러
- 호출 스택 위로 자동 전파를 원할 때
- 관용적인 언어 (Java, Python, C#)

**Result 타입을 사용할 때:**
- 예상된 에러 (검증, 파싱)
- 에러 처리가 명시적이어야 할 때
- 성능이 중요한 코드
- 잘 지원하는 언어 (Rust, Go, Haskell)

---

## 방어적 프로그래밍(Defensive Programming)

방어적 프로그래밍은 예상치 못한 조건을 예상하고 처리하는 코드를 작성하는 것을 의미합니다.

### 시스템 경계에서의 입력 검증

모든 외부 입력을 검증합니다: 사용자 입력, 네트워크 데이터, 파일 내용, 환경 변수.

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

### 전제조건, 후조건, 불변조건(Preconditions, Postconditions, Invariants)

**전제조건(Preconditions):** 함수 실행 전에 참이어야 하는 것
**후조건(Postconditions):** 함수 실행 후에 참이어야 하는 것
**불변조건(Invariants):** 항상 참이어야 하는 것 (예: 데이터 구조)

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

### 계약에 의한 설계(Design by Contract) (Bertrand Meyer)

클래스와 클라이언트 간의 관계를 공식화합니다:
- **클라이언트 의무** (전제조건): 클라이언트가 보장해야 하는 것
- **클래스 의무** (후조건): 클래스가 보장하는 것
- **클래스 불변조건**: 클래스에 대해 항상 유지되는 것

### 단언문: 개발 vs 프로덕션

**단언문은 프로그래밍 버그를 잡기 위한 것이지, 런타임 에러를 처리하기 위한 것이 아닙니다.**

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

**가이드라인:**
- **단언문 사용:** 내부 일관성 확인, 가정 검증, 디버깅
- **예외 사용:** 사용자 입력 검증, 외부 시스템 실패, 런타임 에러

---

## 빠른 실패 원칙(Fail-Fast Principle)

잘못된 상태로 계속하기보다, 에러를 가능한 한 빨리 감지하고 즉시 실패합니다.

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

**이점:**
- 에러가 원인에 가까운 곳에서 발견됨
- 연쇄 실패 방지
- 디버깅이 쉬워짐
- 데이터 손상 방지

---

## 우아한 성능 저하(Graceful Degradation)

컴포넌트가 실패할 때, 완전한 시스템 실패보다 대체 동작을 제공합니다.

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

**서킷 브레이커 패턴(Circuit Breaker Pattern):**

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

## 에러 메시지

### 개발자용: 컨텍스트와 디버깅

문제를 디버깅하기에 충분한 정보를 포함합니다:

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

### 사용자용: 명확하고 실행 가능

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

## 로깅 레벨(Logging Levels)

현대 로깅 프레임워크는 심각도 레벨을 제공합니다:

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

**언제 로깅할지:**
- **DEBUG:** 함수 진입/종료, 변수 값, 상세한 흐름
- **INFO:** 중요한 이벤트 (사용자 액션, 시스템 상태 변경)
- **WARNING:** 예상치 못했지만 처리된 상황, 폐기 예정 알림
- **ERROR:** 잡힌 예외, 실패한 작업
- **CRITICAL:** 시스템 실패, 복구 불가능한 에러

---

## 재시도 패턴(Retry Patterns)

### 지수 백오프(Exponential Backoff)

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

### 지터(Jitter)

썬더링 허드 문제를 방지하기 위해 무작위성을 추가합니다:

```python
def backoff_with_jitter(attempt, base=1, max_delay=60):
    # Full jitter: randomize between 0 and calculated delay
    delay = min(max_delay, base * (2 ** attempt))
    return random.uniform(0, delay)
```

---

## 요약

**핵심 원칙:**
1. **에러 예상** – 에러는 발생할 것입니다
2. **빠르게 실패** – 일찍 감지하고, 크게 실패하기
3. **구체적으로** – 특정 예외를 잡고, 상세한 컨텍스트 제공
4. **관심사 분리** – 에러 처리와 비즈니스 로직을 섞지 않기
5. **컨텍스트 제공** – 디버깅을 위한 충분한 정보 포함
6. **사용자 생각** – 에러 메시지를 유용하고 실행 가능하게 만들기
7. **조용히 실패하지 않기** – 처리하더라도 에러 로깅
8. **경계에서 검증** – 모든 외부 입력 확인
9. **올바른 도구 선택** – 예외적인 경우에는 예외, 예상된 에러에는 Results

---

## 연습 문제

### 연습 문제 1: 잘못된 에러 처리 리팩토링

이 코드를 더 나은 에러 처리로 리팩토링하세요:

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

**수정할 문제:**
- 에러 처리 없음
- 예외 발생 시 파일이 닫히지 않음
- 'value' 키 존재 가정
- 'value'가 숫자라고 가정

### 연습 문제 2: 에러 전략 설계

결제 시스템용 REST API를 구축하고 있습니다. 다음을 다루는 에러 처리 전략을 설계하세요:
- 잘못된 요청 데이터 (누락된 필드, 잘못된 타입)
- 데이터베이스 연결 실패
- 외부 결제 게이트웨이 에러 (타임아웃, 거부)
- 잔액 부족
- 로깅 요구사항
- 클라이언트에 대한 에러 응답

각 시나리오를 처리하는 방법을 보여주는 예제 코드를 작성하세요.

### 연습 문제 3: 재시도 로직 구현

다음을 수행하는 `retry_with_circuit_breaker` 함수를 작성하세요:
- 지수 백오프로 함수를 N번까지 재시도
- 연속 실패 후 서킷 브레이커 열기
- 각 단계에서 의미 있는 로깅 제공
- Result 타입을 반환하거나 유익한 예외 발생

무작위로 실패하는 모의 함수로 테스트하세요.

### 연습 문제 4: 에러 메시지 개선

이 에러 메시지를 개선하세요:

1. `"Error 42"`
2. `"Invalid input"`
3. `"NullPointerException at line 127"`
4. `"Cannot process request"`

각각에 대해 작성하세요:
- 개발자 친화적 버전 (상세하고 기술적)
- 사용자 친화적 버전 (명확하고 실행 가능)

### 연습 문제 5: 예외 vs Result 타입

"YYYY-MM-DD" 형식의 날짜 문자열을 파싱하는 함수 `parse_date(text: str)`를 구현하세요:
1. 예외를 사용하여 한 번
2. Result 타입을 사용하여 한 번 (라이브러리 사용 또는 직접 구현)

두 접근 방식을 비교하세요. 각각을 언제 사용하시겠습니까?

---

## 내비게이션

**이전 레슨**: [08_Design_Patterns.md](08_Design_Patterns.md)
**다음 레슨**: [테스팅 기초(Testing Fundamentals)](10_Testing_Fundamentals.md)

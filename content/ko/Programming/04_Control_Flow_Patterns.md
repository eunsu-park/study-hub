# 제어 흐름 패턴

> **토픽**: Programming
> **레슨**: 4 of 16
> **선수 지식**: What Is Programming, Programming Paradigms, Data Types & Abstraction
> **목표**: 제어 흐름 메커니즘 — 분기, 루프, 재귀, 반복자, 에러 처리 — 을 마스터하고 각 패턴을 언제 사용할지 학습합니다.

---

## 순차 실행(Sequential Execution)

기본적으로 프로그램은 **순차적**으로 실행됩니다 — 한 문장씩, 위에서 아래로.

**Python:**
```python
print("Step 1")
x = 10
print("Step 2")
y = x + 5
print("Step 3")
print(f"Result: {y}")

# Output:
# Step 1
# Step 2
# Step 3
# Result: 15
```

**JavaScript:**
```javascript
console.log("Step 1");
let x = 10;
console.log("Step 2");
let y = x + 5;
console.log("Step 3");
console.log("Result:", y);
```

**Java:**
```java
System.out.println("Step 1");
int x = 10;
System.out.println("Step 2");
int y = x + 5;
System.out.println("Step 3");
System.out.println("Result: " + y);
```

이것이 가장 간단한 제어 흐름 형태입니다. 하지만 실제 프로그램은 **결정**과 **반복**이 필요합니다.

---

## 조건 분기(Conditional Branching)

조건에 따라 다른 코드 경로를 실행합니다.

### If/Else

**Python:**
```python
age = 18

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")
```

**JavaScript:**
```javascript
let age = 18;

if (age >= 18) {
    console.log("Adult");
} else if (age >= 13) {
    console.log("Teenager");
} else {
    console.log("Child");
}
```

**Java:**
```java
int age = 18;

if (age >= 18) {
    System.out.println("Adult");
} else if (age >= 13) {
    System.out.println("Teenager");
} else {
    System.out.println("Child");
}
```

**C++:**
```cpp
int age = 18;

if (age >= 18) {
    std::cout << "Adult" << std::endl;
} else if (age >= 13) {
    std::cout << "Teenager" << std::endl;
} else {
    std::cout << "Child" << std::endl;
}
```

### Switch/Match

여러 개별 케이스를 위한 구문.

**Java (switch):**
```java
int day = 3;
String dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
        break;
}

System.out.println(dayName);  // Wednesday
```

**JavaScript (switch):**
```javascript
let day = 3;
let dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
}

console.log(dayName);  // Wednesday
```

**C++ (switch):**
```cpp
int day = 3;
std::string dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
        break;
}

std::cout << dayName << std::endl;  // Wednesday
```

### 패턴 매칭(Pattern Matching) (현대 언어)

**Python (3.10+):**
```python
def describe(value):
    match value:
        case 0:
            return "zero"
        case 1 | 2 | 3:
            return "small"
        case int(x) if x < 0:
            return "negative"
        case int():
            return "positive integer"
        case str():
            return "string"
        case _:
            return "unknown"

print(describe(2))     # small
print(describe(-5))    # negative
print(describe("hi"))  # string
```

**Rust (match):**
```rust
fn describe(value: i32) -> &'static str {
    match value {
        0 => "zero",
        1..=3 => "small",
        x if x < 0 => "negative",
        _ => "positive",
    }
}

println!("{}", describe(2));   // small
println!("{}", describe(-5));  // negative
```

**패턴 매칭의 이점**: 더 표현력 있고, 완전성 검사 (컴파일러가 모든 케이스가 커버되었는지 확인).

### 삼항 연산자(Ternary Operator)

간결한 조건 표현식.

**Python:**
```python
age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # adult
```

**JavaScript:**
```javascript
let age = 20;
let status = age >= 18 ? "adult" : "minor";
console.log(status);  // adult
```

**Java:**
```java
int age = 20;
String status = age >= 18 ? "adult" : "minor";
System.out.println(status);  // adult
```

**C++:**
```cpp
int age = 20;
std::string status = age >= 18 ? "adult" : "minor";
std::cout << status << std::endl;  // adult
```

### 가드 절(Guard Clauses)

중첩을 줄이고 가독성을 향상시키는 **조기 반환(early returns)**.

**전 (중첩됨):**
```python
def process_user(user):
    if user is not None:
        if user.is_active:
            if user.has_permission("admin"):
                print("Processing admin user")
                # ... complex logic ...
            else:
                print("No permission")
        else:
            print("Inactive user")
    else:
        print("No user")
```

**후 (가드 절):**
```python
def process_user(user):
    if user is None:
        print("No user")
        return

    if not user.is_active:
        print("Inactive user")
        return

    if not user.has_permission("admin"):
        print("No permission")
        return

    print("Processing admin user")
    # ... complex logic ...
```

**이점**: 중첩 감소, 더 명확한 에러 처리, 주요 로직이 마지막에 위치.

**JavaScript:**
```javascript
function processUser(user) {
    if (!user) {
        console.log("No user");
        return;
    }

    if (!user.isActive) {
        console.log("Inactive user");
        return;
    }

    if (!user.hasPermission("admin")) {
        console.log("No permission");
        return;
    }

    console.log("Processing admin user");
    // ... main logic ...
}
```

---

## 루프(Loops)

반복은 프로그래밍의 기본입니다.

### For 루프

고정된 횟수만큼 또는 컬렉션을 순회합니다.

**Python:**
```python
# Range-based
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Iterating over collection
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# With index
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

**JavaScript:**
```javascript
// Traditional for loop
for (let i = 0; i < 5; i++) {
    console.log(i);  // 0, 1, 2, 3, 4
}

// For-of (ES6)
let fruits = ["apple", "banana", "cherry"];
for (let fruit of fruits) {
    console.log(fruit);
}

// For-in (iterates over keys/indices)
for (let index in fruits) {
    console.log(index, fruits[index]);
}
```

**Java:**
```java
// Traditional for loop
for (int i = 0; i < 5; i++) {
    System.out.println(i);  // 0, 1, 2, 3, 4
}

// Enhanced for loop (for-each)
String[] fruits = {"apple", "banana", "cherry"};
for (String fruit : fruits) {
    System.out.println(fruit);
}
```

**C++:**
```cpp
// Traditional for loop
for (int i = 0; i < 5; i++) {
    std::cout << i << std::endl;  // 0, 1, 2, 3, 4
}

// Range-based for loop (C++11)
std::vector<std::string> fruits = {"apple", "banana", "cherry"};
for (const auto& fruit : fruits) {
    std::cout << fruit << std::endl;
}
```

### While 루프

조건이 참인 동안 반복합니다.

**Python:**
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

**JavaScript:**
```javascript
let count = 0;
while (count < 5) {
    console.log(count);
    count++;
}
```

**Java:**
```java
int count = 0;
while (count < 5) {
    System.out.println(count);
    count++;
}
```

### Do-While 루프

최소 한 번 실행하고, 조건이 참인 동안 반복합니다.

**Java:**
```java
int count = 0;
do {
    System.out.println(count);
    count++;
} while (count < 5);

// Executes at least once even if condition is false
int x = 10;
do {
    System.out.println("Runs once");
} while (x < 5);  // Still runs once
```

**JavaScript:**
```javascript
let count = 0;
do {
    console.log(count);
    count++;
} while (count < 5);
```

**C++:**
```cpp
int count = 0;
do {
    std::cout << count << std::endl;
    count++;
} while (count < 5);
```

**참고**: Python에는 do-while 루프가 없습니다. `while True`와 `break`를 사용하세요:
```python
count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break
```

### 루프 제어: Break와 Continue

**Break**: 즉시 루프를 종료합니다.

**Python:**
```python
for i in range(10):
    if i == 5:
        break  # Exit loop when i is 5
    print(i)  # 0, 1, 2, 3, 4
```

**Continue**: 현재 반복의 나머지를 건너뛰고 다음으로 진행합니다.

**Python:**
```python
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)  # 1, 3, 5, 7, 9
```

**JavaScript:**
```javascript
for (let i = 0; i < 10; i++) {
    if (i === 5) break;  // Exit at 5
    console.log(i);  // 0, 1, 2, 3, 4
}

for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) continue;  // Skip even
    console.log(i);  // 1, 3, 5, 7, 9
}
```

### 루프 불변량(Loop Invariants)

**루프 불변량**은 각 반복 전후에 참인 조건입니다. 정확성에 대한 추론에 유용합니다.

**예시: 배열에서 최댓값 찾기**

**불변량**: 각 반복 시작 시, `max`는 지금까지 검사한 모든 요소의 최댓값을 담고 있습니다.

**Python:**
```python
def find_max(numbers):
    if not numbers:
        return None

    max_val = numbers[0]  # Invariant: max_val is max of numbers[0:0+1]

    for i in range(1, len(numbers)):
        # Invariant: max_val is max of numbers[0:i]
        if numbers[i] > max_val:
            max_val = numbers[i]
        # Invariant maintained: max_val is max of numbers[0:i+1]

    return max_val
```

불변량을 이해하면 올바른 루프를 작성하고 문제가 발생했을 때 디버그하는 데 도움이 됩니다.

---

## 재귀(Recursion)

자기 자신을 호출하는 함수. 모든 재귀 함수는 다음이 필요합니다:
1. **기저 사례(Base case)**: 재귀를 멈추는 조건
2. **재귀 사례(Recursive case)**: 더 간단한 입력으로 자신을 호출

### 팩토리얼(Factorial)

**수학적 정의**:
- `factorial(0) = 1` (기저 사례)
- `factorial(n) = n × factorial(n-1)` (재귀 사례)

**Python:**
```python
def factorial(n):
    if n == 0:
        return 1  # Base case
    else:
        return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120 (5 × 4 × 3 × 2 × 1)
```

**JavaScript:**
```javascript
function factorial(n) {
    if (n === 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

console.log(factorial(5));  // 120
```

**Java:**
```java
public static int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

System.out.println(factorial(5));  // 120
```

**C++:**
```cpp
int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

std::cout << factorial(5) << std::endl;  // 120
```

### 피보나치(Fibonacci)

**정의**:
- `fib(0) = 0`, `fib(1) = 1` (기저 사례)
- `fib(n) = fib(n-1) + fib(n-2)` (재귀 사례)

**Python:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(6))  # 8 (0, 1, 1, 2, 3, 5, 8)
```

**참고**: 이것은 반복 계산으로 인해 비효율적입니다 (지수 시간). 더 나은 성능을 위해 메모이제이션이나 반복을 사용하세요.

### 트리 순회(Tree Traversal)

재귀는 트리 구조에서 빛을 발합니다.

**Python:**
```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(node):
    """Left → Root → Right"""
    if node is None:
        return  # Base case

    inorder_traversal(node.left)
    print(node.value)
    inorder_traversal(node.right)

# Example tree:
#       1
#      / \
#     2   3
#    / \
#   4   5

root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)

inorder_traversal(root)  # Output: 4, 2, 5, 1, 3
```

### 꼬리 재귀(Tail Recursion)

재귀 호출이 함수의 마지막 연산이면 **꼬리 재귀**입니다. 일부 컴파일러는 꼬리 재귀를 루프로 최적화합니다 (스택 성장 없음).

**꼬리 재귀가 아님 (팩토리얼):**
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)  # Multiplication AFTER recursive call
```

**꼬리 재귀 (누산기가 있는 팩토리얼):**
```python
def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    else:
        return factorial_tail(n - 1, n * acc)  # Recursive call is last operation

print(factorial_tail(5))  # 120
```

**꼬리 호출 최적화가 있는 언어**: Scheme, Scala, 일부 Rust, 일부 JavaScript 엔진.

### 재귀 vs 반복 언제 사용할까

**재귀가 더 나을 때**:
- 문제가 자연스럽게 재귀적 (트리, 그래프, 분할 정복)
- 코드가 더 명확하고 우아함

**반복이 더 나을 때**:
- 성능이 중요 (스택 오버헤드 피함)
- 문제가 자연스럽게 반복적 (단순 루프)

**예시: 팩토리얼을 반복적으로**

**Python:**
```python
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial_iterative(5))  # 120
```

더 효율적이지만 (재귀 호출 없음), 복잡한 문제에서는 재귀가 종종 더 명확합니다.

---

## 반복자와 제너레이터(Iterators and Generators)

### 반복자(Iterators)

한 번에 하나씩 값 시퀀스를 생성하는 객체.

**Python:**
```python
# Lists are iterable
numbers = [1, 2, 3, 4, 5]
iterator = iter(numbers)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
```

**JavaScript:**
```javascript
let numbers = [1, 2, 3, 4, 5];
let iterator = numbers[Symbol.iterator]();

console.log(iterator.next().value);  // 1
console.log(iterator.next().value);  // 2
console.log(iterator.next().value);  // 3
```

**이점**: 메모리 효율적 (전체 컬렉션을 메모리에 둘 필요 없음), 지연 평가.

### 제너레이터(Generators)

값을 한 번에 하나씩 **yield**하며, yield 사이에 일시 중지하는 함수.

**Python:**
```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count  # Pause here, return count
        count += 1

# Create generator
gen = count_up_to(5)

print(next(gen))  # 1
print(next(gen))  # 2

# Or use in loop
for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5
```

**JavaScript:**
```javascript
function* countUpTo(n) {
    let count = 1;
    while (count <= n) {
        yield count;
        count++;
    }
}

let gen = countUpTo(5);
console.log(gen.next().value);  // 1
console.log(gen.next().value);  // 2

// Or use in loop
for (let num of countUpTo(5)) {
    console.log(num);  // 1, 2, 3, 4, 5
}
```

**이점**:
- **지연 평가**: 필요할 때 값 계산
- **메모리 효율적**: 전체 시퀀스를 저장하지 않음
- **무한 시퀀스**: 무한 스트림 표현 가능

**예시: 무한 시퀀스**

**Python:**
```python
def infinite_count():
    count = 0
    while True:
        yield count
        count += 1

# Only compute as needed
gen = infinite_count()
print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 2
# ... can continue forever
```

---

## 코루틴과 Async/Await(Coroutines and Async/Await)

**코루틴(Coroutines)**: 일시 중지하고 재개할 수 있는 함수로, 협력적 멀티태스킹을 가능하게 합니다.

### Async/Await 패턴

**Python:**
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(2)  # Simulate network delay
    print(f"Done fetching {url}")
    return f"Data from {url}"

async def main():
    # Run concurrently
    task1 = fetch_data("https://api1.com")
    task2 = fetch_data("https://api2.com")

    result1, result2 = await asyncio.gather(task1, task2)
    print(result1, result2)

# Run
asyncio.run(main())
```

**JavaScript:**
```javascript
async function fetchData(url) {
    console.log(`Fetching ${url}...`);
    await new Promise(resolve => setTimeout(resolve, 2000));  // Simulate delay
    console.log(`Done fetching ${url}`);
    return `Data from ${url}`;
}

async function main() {
    let task1 = fetchData("https://api1.com");
    let task2 = fetchData("https://api2.com");

    let [result1, result2] = await Promise.all([task1, task2]);
    console.log(result1, result2);
}

main();
```

**스레드와의 주요 차이점**: 코루틴은 **협력적**(명시적으로 제어를 양보)이지, **선점적**(OS가 언제든 스레드를 중단 가능)이 아닙니다.

---

## 에러 흐름(Error Flow)

오류와 예외 상황을 어떻게 처리할까요?

### 예외(Exceptions)

**Try/Catch/Finally**

**Python:**
```python
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    finally:
        print("Cleanup (always runs)")

divide(10, 2)   # 5.0, "Cleanup"
divide(10, 0)   # "Error", None, "Cleanup"
```

**JavaScript:**
```javascript
function divide(a, b) {
    try {
        if (b === 0) {
            throw new Error("Cannot divide by zero");
        }
        return a / b;
    } catch (error) {
        console.log("Error:", error.message);
        return null;
    } finally {
        console.log("Cleanup (always runs)");
    }
}

divide(10, 2);  // 5, "Cleanup"
divide(10, 0);  // "Error: Cannot divide by zero", null, "Cleanup"
```

**Java:**
```java
public static Double divide(int a, int b) {
    try {
        return (double) a / b;
    } catch (ArithmeticException e) {
        System.out.println("Error: " + e.getMessage());
        return null;
    } finally {
        System.out.println("Cleanup (always runs)");
    }
}
```

**C++:**
```cpp
double divide(int a, int b) {
    try {
        if (b == 0) {
            throw std::runtime_error("Cannot divide by zero");
        }
        return static_cast<double>(a) / b;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 0.0;
    }
}
```

### Result/Either 타입

함수형 접근법: 성공 또는 실패를 나타내는 타입 반환.

**Rust:**
```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}

match divide(10, 2) {
    Ok(result) => println!("Result: {}", result),
    Err(error) => println!("Error: {}", error),
}
```

**이점**: 에러가 타입 시그니처에 명시적. 컴파일러가 처리를 강제합니다.

### 에러 전파(Error Propagation)

**Rust의 `?` 연산자**:
```rust
fn read_file_length(path: &str) -> Result<usize, std::io::Error> {
    let contents = std::fs::read_to_string(path)?;  // Propagate error if it occurs
    Ok(contents.len())
}
```

`read_to_string`이 `Err`를 반환하면 즉시 `read_file_length`에서 반환됩니다. 그렇지 않으면 계속합니다.

**Java의 `throws`**:
```java
public static String readFile(String path) throws IOException {
    return new String(Files.readAllBytes(Paths.get(path)));
}

// Caller must handle
try {
    String content = readFile("file.txt");
} catch (IOException e) {
    System.out.println("Error reading file");
}
```

---

## 단락 평가(Short-Circuit Evaluation)

논리 연산자 `&&` (AND)와 `||` (OR)는 **단락 평가**를 사용합니다: 결과가 결정되는 즉시 평가를 멈춥니다.

**Python:**
```python
def is_positive(x):
    print(f"Checking {x}")
    return x > 0

# AND: stops at first false
result = is_positive(5) and is_positive(10) and is_positive(-3)
# Output: Checking 5, Checking 10, Checking -3
# Result: False

# OR: stops at first true
result = is_positive(-5) or is_positive(10) or is_positive(20)
# Output: Checking -5, Checking 10
# Result: True (doesn't check 20)
```

**JavaScript:**
```javascript
function check(x) {
    console.log(`Checking ${x}`);
    return x > 0;
}

let result = check(5) && check(10) && check(-3);
// Logs: Checking 5, Checking 10, Checking -3

let result2 = check(-5) || check(10) || check(20);
// Logs: Checking -5, Checking 10
// Doesn't log Checking 20 (short-circuited)
```

**사용 사례: null/undefined 에러 피하기**

**JavaScript:**
```javascript
let user = getUser();
if (user && user.isActive && user.hasPermission("admin")) {
    console.log("Admin user");
}
// If user is null, doesn't try to access user.isActive (would error)
```

---

## 구조적 프로그래밍(Structured Programming)

**구조적 프로그래밍** (1960년대-70년대)이 주장한 것:
- **`goto` 금지**: 대신 루프와 함수 사용
- **단일 진입, 단일 종료**: 함수는 하나의 진입점과 하나의 반환 (하지만 지금은 여러 반환이 일반적)
- **하향식 설계**: 문제를 더 작은 프로시저로 나눔

**나쁨 (goto로 비구조화):**
```c
// Don't do this
int i = 0;
start:
    printf("%d\n", i);
    i++;
    if (i < 5) goto start;
```

**좋음 (루프로 구조화):**
```c
for (int i = 0; i < 5; i++) {
    printf("%d\n", i);
}
```

**현대적 합의**: 구조적 프로그래밍 원칙은 좋지만, 조기 반환과 `break`/`continue`의 실용적 사용은 가독성을 향상시킵니다.

---

## 연습 문제

### 연습 문제 1: 가드 절로 리팩토링

이 중첩된 코드를 가드 절을 사용해 리팩토링하세요:

**Python:**
```python
def process_order(order):
    if order is not None:
        if order.is_valid():
            if order.total > 0:
                if order.user.is_verified:
                    print("Processing order")
                else:
                    print("User not verified")
            else:
                print("Order total must be positive")
        else:
            print("Invalid order")
    else:
        print("No order")
```

### 연습 문제 2: 재귀 vs 반복

**배열의 합계**를 재귀적으로와 반복적으로 모두 구현하세요:

**재귀:**
```python
def sum_recursive(numbers):
    # Base case: empty array
    # Recursive case: first element + sum of rest
    pass
```

**반복:**
```python
def sum_iterative(numbers):
    # Use a loop
    pass
```

어느 것이 더 명확한가요? 어느 것이 더 효율적인가요?

### 연습 문제 3: 제너레이터

**피보나치 수열**을 무한정 생성하는 제너레이터를 작성하세요:

```python
def fibonacci_gen():
    # Yield 0, 1, 1, 2, 3, 5, 8, 13, ...
    pass

# Usage
gen = fibonacci_gen()
for i in range(10):
    print(next(gen))  # First 10 Fibonacci numbers
```

### 연습 문제 4: 패턴 매칭

언어가 패턴 매칭을 지원하면 (Python 3.10+, Rust, Scala), 값을 분류하는 함수를 작성하세요:
- 0이면: "zero"
- 1-10이면: "small"
- 11-100이면: "medium"
- 100보다 크면: "large"
- 음수이면: "negative"
- 그 외: "unknown"

### 연습 문제 5: 에러 처리

`a / b`의 결과를 반환하는 `safe_divide(a, b)` 함수를 작성하세요:
- 0으로 나누기를 우아하게 처리
- 한 구현에서는 예외 사용 (try/catch)
- 다른 구현에서는 Result 타입 (또는 유사) 사용

**Python:**
```python
# Exception version
def safe_divide_exception(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

# Result version (using a tuple)
def safe_divide_result(a, b):
    if b == 0:
        return (False, "Cannot divide by zero")
    else:
        return (True, a / b)

# Usage
success, value = safe_divide_result(10, 2)
if success:
    print(f"Result: {value}")
else:
    print(f"Error: {value}")
```

### 연습 문제 6: 꼬리 재귀

**피보나치 함수**를 꼬리 재귀로 다시 작성하세요:

**힌트**: 누산기가 있는 헬퍼 함수 사용.

```python
def fibonacci_tail(n, a=0, b=1):
    # Base case: n == 0
    # Recursive case: call with updated accumulators
    pass
```

---

## 요약

제어 흐름은 **실행 순서**를 결정합니다:

- **순차적**: 기본, 위에서 아래로
- **분기**:
  - If/else, switch/match, 삼항 연산자
  - 가드 절: 명확성을 위한 조기 반환
- **루프**:
  - For, while, do-while
  - 제어를 위한 Break/continue
  - 정확성에 대한 추론을 위한 루프 불변량
- **재귀**:
  - 기저 사례 + 재귀 사례
  - 최적화를 위한 꼬리 재귀
  - 문제가 자연스럽게 재귀적일 때 사용
- **반복자 & 제너레이터**:
  - 지연 평가, 메모리 효율적
  - 필요에 따라 값 생성
- **코루틴 & Async/Await**:
  - 협력적 멀티태스킹
  - 실행 일시 중지/재개
- **에러 처리**:
  - 예외: try/catch/finally
  - Result 타입: 명시적 에러 값
  - 에러 전파: `?` 연산자, `throws`
- **단락 평가**: `&&`와 `||`로 로직 최적화
- **구조적 프로그래밍**: goto 피하기, 구조화된 구조 사용

**핵심 통찰**: 문제에 맞는 올바른 제어 흐름 패턴을 선택하세요. 가드 절은 중첩을 줄입니다. 재귀는 트리와 그래프에서 빛을 발합니다. 제너레이터는 무한 시퀀스를 가능하게 합니다. 예외는 예외적인 경우를 처리합니다. 각 패턴에는 제자리가 있습니다.

---

## 탐색

[← 이전: Data Types & Abstraction](03_Data_Types_and_Abstraction.md)

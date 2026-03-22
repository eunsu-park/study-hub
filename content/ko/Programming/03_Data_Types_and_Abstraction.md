# 데이터 타입과 추상화

> **토픽**: Programming
> **레슨**: 3 of 16
> **선수 지식**: What Is Programming, Programming Paradigms
> **목표**: 데이터 타입, 타입 시스템, 추상 데이터 타입, 추상화가 복잡성을 관리하는 방법을 이해합니다.

---

## 타입이란 무엇인가?

**타입(type)**은 다음을 결정하는 데이터의 분류입니다:
- 데이터가 가질 수 있는 값
- 데이터에 수행할 수 있는 연산
- 차지하는 메모리 크기
- 컴퓨터가 해석하는 방법

**비유**: 타입은 컨테이너와 같습니다 — 유리병은 액체를 담고, 판지 상자는 고체를 담습니다. 판지 상자에 물을 부을 수 없습니다 (또는 부어서는 안 됩니다). 타입은 코드에서 유사한 제약을 강제합니다.

### 타입이 중요한 이유

```python
# Without types (conceptually):
x = "42"
y = 10
z = x + y  # What should this mean? "4210" or 52? Error?

# With types:
x: str = "42"
y: int = 10
# z = x + y  # Type error: can't add string and int
z = int(x) + y  # Explicit conversion: 52
```

타입은 어떤 연산이 유효한지에 대한 **제약**을 강제하여 오류를 방지합니다.

---

## 기본 타입(Primitive Types)

기본 타입은 언어가 제공하는 **구성 요소**입니다. 일반적으로 하드웨어 표현에 직접 매핑됩니다.

### 정수(Integers)

소수 부분이 없는 정수.

**Python:**
```python
age = 25
population = 7_800_000_000  # Python allows underscores for readability
negative = -42
```

**Java:**
```java
byte smallNumber = 127;       // 8-bit: -128 to 127
short mediumNumber = 32000;   // 16-bit: -32,768 to 32,767
int standardNumber = 100000;  // 32-bit: ~-2B to 2B
long largeNumber = 10000000000L;  // 64-bit
```

**C++:**
```cpp
int x = 42;
unsigned int y = 100;  // Only positive values
long long z = 9223372036854775807LL;  // 64-bit
```

### 부동소수점 수(Floating-Point Numbers)

소수 부분이 있는 숫자. 이진 인코딩으로 인한 근사 표현.

**Python:**
```python
pi = 3.14159
scientific = 6.022e23  # 6.022 × 10^23 (Avogadro's number)
```

**JavaScript:**
```javascript
let price = 19.99;
let tiny = 0.0000001;
let notExact = 0.1 + 0.2;  // 0.30000000000000004 (floating-point precision issue)
```

**Java:**
```java
float f = 3.14f;     // 32-bit, single precision
double d = 3.14159;  // 64-bit, double precision (default)
```

### 불리언(Booleans)

논리 연산을 위한 참 또는 거짓 값.

**Python:**
```python
is_active = True
has_permission = False

if is_active and has_permission:
    print("Access granted")
```

**JavaScript:**
```javascript
let isLoggedIn = true;
let isAdmin = false;
console.log(isLoggedIn && !isAdmin);  // true
```

**C++:**
```cpp
bool flag = true;
bool result = (5 > 3);  // true
```

### 문자(Characters)

단일 문자, 종종 정수(ASCII/Unicode 코드 포인트)로 표현됩니다.

**Java:**
```java
char letter = 'A';  // Single quotes for char
char unicode = '\u0041';  // Unicode: also 'A'
```

**C++:**
```cpp
char c = 'x';
char newline = '\n';
```

**Python:**
```python
# Python has no separate char type; single-character strings
letter = 'A'
```

### 문자열(Strings)

문자의 시퀀스. 일부 언어는 문자열을 기본형으로, 다른 언어는 객체로 취급합니다.

**Python:**
```python
name = "Alice"
message = 'Hello, World!'
multiline = """This is
a multi-line
string"""
```

**JavaScript:**
```javascript
let greeting = "Hello";
let template = `Hello, ${name}!`;  // Template literals
```

**Java:**
```java
String text = "Hello, World!";  // String is an object, not primitive
```

**C++:**
```cpp
#include <string>
std::string message = "Hello, C++!";
```

---

## 복합 타입(Composite Types)

기본 타입으로부터 구축된 타입.

### 배열(Arrays)

같은 타입의 요소로 이루어진 고정 크기의 정렬된 컬렉션.

**Python:**
```python
# Python lists are dynamic, not fixed-size, but conceptually similar
numbers = [1, 2, 3, 4, 5]
```

**JavaScript:**
```javascript
let numbers = [1, 2, 3, 4, 5];  // Dynamic arrays
```

**Java:**
```java
int[] numbers = {1, 2, 3, 4, 5};  // Fixed size
int[] array = new int[10];  // Allocate size 10, initialized to 0
```

**C++:**
```cpp
#include <array>
std::array<int, 5> numbers = {1, 2, 3, 4, 5};  // Fixed size 5
```

### 레코드/구조체(Records/Structs)

서로 다른 타입의 관련 데이터를 그룹화합니다.

**C:**
```c
struct Person {
    char name[50];
    int age;
    double salary;
};

struct Person alice = {"Alice", 30, 75000.0};
printf("%s is %d years old\n", alice.name, alice.age);
```

**C++:**
```cpp
struct Point {
    int x;
    int y;
};

Point p = {10, 20};
std::cout << "x: " << p.x << ", y: " << p.y << std::endl;
```

### 튜플(Tuples)

가능하게는 다른 타입의 요소로 이루어진 정렬된 고정 크기 컬렉션.

**Python:**
```python
person = ("Alice", 30, "Engineer")  # (name, age, job)
name, age, job = person  # Unpacking
```

**JavaScript (배열 사용):**
```javascript
let person = ["Alice", 30, "Engineer"];
let [name, age, job] = person;  // Destructuring
```

---

## 타입 시스템(Type Systems)

### 정적 vs 동적 타이핑

**정적 타이핑(Static Typing)**: 타입을 **컴파일 시점**에 확인합니다. 변수는 고정된 타입을 갖습니다.

**언어**: Java, C++, C, Rust, Go, TypeScript

**Java 예시:**
```java
int x = 10;
// x = "hello";  // Compile error: incompatible types
x = 20;  // OK
```

**C++ 예시:**
```cpp
int count = 5;
// count = "text";  // Compile error
count = 10;  // OK
```

**이점**:
- 오류를 일찍 포착 (프로그램 실행 전)
- 더 나은 도구 (자동완성, 리팩토링)
- 성능 최적화 (컴파일러가 타입을 알고 있음)

**절충안**:
- 더 장황함 (타입 주석)
- 덜 유연함

---

**동적 타이핑(Dynamic Typing)**: 타입을 **런타임**에 확인합니다. 변수는 어떤 타입이든 담을 수 있습니다.

**언어**: Python, JavaScript, Ruby, PHP

**Python 예시:**
```python
x = 10       # x is an int
x = "hello"  # Now x is a string — no error
x = [1, 2]   # Now x is a list
```

**JavaScript 예시:**
```javascript
let x = 10;
x = "hello";  // OK
x = {key: "value"};  // OK
```

**이점**:
- 덜 상용구적
- 더 유연함
- 더 빠른 프로토타이핑

**절충안**:
- 런타임에 오류 포착 (프로덕션에서 충돌 가능)
- 대규모 프로젝트에서 코드에 대한 추론 어려움
- 느린 성능 (런타임 타입 검사)

---

### 강한 vs 약한 타이핑

**강한 타이핑(Strong Typing)**: 타입 규칙의 엄격한 강제. 호환되지 않는 타입 간의 암시적 변환 없음.

**Python (강함):**
```python
x = "5"
y = 10
# z = x + y  # TypeError: can't add string and int
z = int(x) + y  # Must explicitly convert: 15
```

**약한 타이핑(Weak Typing)**: 암시적 타입 변환(타입 강제) 허용.

**JavaScript (약함):**
```javascript
let x = "5";
let y = 10;
let z = x + y;  // "510" — string concatenation (implicit conversion)
let w = x - y;  // -5 — subtraction (implicit conversion to number)
console.log(z, w);  // "510", -5
```

**강한 타이핑의 이점**: 더 적은 놀라움, 더 명확한 의도
**약한 타이핑의 이점**: 더 허용적, 덜 장황함 (하지만 더 많은 버그)

---

### 타입 추론(Type Inference)

컴파일러/인터프리터가 자동으로 타입을 추론합니다.

**Kotlin:**
```kotlin
val x = 10  // Inferred as Int
val name = "Alice"  // Inferred as String
// x = "text"  // Error: type mismatch
```

**Rust:**
```rust
let x = 10;  // Inferred as i32 (32-bit integer)
let y = 3.14;  // Inferred as f64 (64-bit float)
```

**TypeScript:**
```javascript
let count = 5;  // Inferred as number
// count = "text";  // Error: Type 'string' is not assignable to type 'number'
```

**이점**: 동적 타이핑의 간결함 + 정적 타이핑의 안전성.

---

## 추상 데이터 타입(Abstract Data Types, ADTs)

**추상 데이터 타입**은 **인터페이스**(어떤 연산이 가능한지)와 **구현**(그 연산이 어떻게 수행되는지)을 분리합니다.

**핵심 아이디어**: 사용자는 내부 세부사항을 알 필요 없이 잘 정의된 인터페이스를 통해 ADT와 상호작용합니다.

### 스택(Stack) ADT

**인터페이스 (연산)**:
- `push(item)`: 맨 위에 항목 추가
- `pop()`: 맨 위 항목 제거 및 반환
- `peek()`: 제거하지 않고 맨 위 항목 보기
- `is_empty()`: 스택이 비어있는지 확인

**구현 1: 배열 사용**

**Python:**
```python
class ArrayStack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0


# Usage (same interface regardless of implementation)
stack = ArrayStack()
stack.push(10)
stack.push(20)
print(stack.pop())  # 20
```

**구현 2: 연결 리스트 사용**

**Python:**
```python
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkedStack:
    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, item):
        self._top = Node(item, self._top)
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        value = self._top.value
        self._top = self._top.next
        self._size -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._top.value

    def is_empty(self):
        return self._top is None


# Usage (same interface!)
stack = LinkedStack()
stack.push(10)
stack.push(20)
print(stack.pop())  # 20
```

**핵심 포인트**: **인터페이스**는 동일합니다. 사용자는 배열 기반인지 연결 리스트 기반인지 알 필요가 없습니다. 이것이 **추상화**입니다.

---

### 큐(Queue) ADT

**인터페이스**:
- `enqueue(item)`: 뒤쪽에 추가
- `dequeue()`: 앞쪽에서 제거 및 반환
- `is_empty()`: 비어있는지 확인

**Java 구현:**
```java
import java.util.LinkedList;

public interface Queue<T> {
    void enqueue(T item);
    T dequeue();
    boolean isEmpty();
}

public class LinkedQueue<T> implements Queue<T> {
    private LinkedList<T> data = new LinkedList<>();

    public void enqueue(T item) {
        data.addLast(item);
    }

    public T dequeue() {
        if (isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        return data.removeFirst();
    }

    public boolean isEmpty() {
        return data.isEmpty();
    }
}
```

---

### 맵/딕셔너리(Map/Dictionary) ADT

**인터페이스**:
- `put(key, value)`: 키-값 쌍 저장
- `get(key)`: 키로 값 검색
- `remove(key)`: 키-값 쌍 삭제
- `contains(key)`: 키 존재 여부 확인

**Python (내장 dict 사용):**
```python
# Python's dict is an implementation of the Map ADT
phonebook = {}
phonebook["Alice"] = "555-1234"
phonebook["Bob"] = "555-5678"

print(phonebook["Alice"])  # "555-1234"
print("Alice" in phonebook)  # True
```

**JavaScript (Map 사용):**
```javascript
let map = new Map();
map.set("Alice", "555-1234");
map.set("Bob", "555-5678");

console.log(map.get("Alice"));  // "555-1234"
console.log(map.has("Alice"));  // true
```

---

## 제네릭과 템플릿(Generics and Templates)

**제네릭(Generics)** (Java, C#, TypeScript)과 **템플릿(Templates)** (C++)은 **모든 타입**과 작동하는 코드를 작성할 수 있게 합니다.

### Java 제네릭

**제네릭 없이:**
```java
// Must use Object, lose type safety
public class Box {
    private Object item;

    public void set(Object item) {
        this.item = item;
    }

    public Object get() {
        return item;
    }
}

Box box = new Box();
box.set("Hello");
String s = (String) box.get();  // Explicit cast needed
```

**제네릭 사용:**
```java
public class Box<T> {
    private T item;

    public void set(T item) {
        this.item = item;
    }

    public T get() {
        return item;
    }
}

Box<String> stringBox = new Box<>();
stringBox.set("Hello");
String s = stringBox.get();  // No cast needed, type-safe

Box<Integer> intBox = new Box<>();
intBox.set(42);
// intBox.set("text");  // Compile error
```

### C++ 템플릿

```cpp
template <typename T>
class Box {
private:
    T item;

public:
    void set(T value) {
        item = value;
    }

    T get() const {
        return item;
    }
};

// Usage
Box<int> intBox;
intBox.set(42);
std::cout << intBox.get() << std::endl;  // 42

Box<std::string> stringBox;
stringBox.set("Hello");
std::cout << stringBox.get() << std::endl;  // Hello
```

### TypeScript 제네릭

```javascript
class Box<T> {
    private item: T;

    set(value: T): void {
        this.item = value;
    }

    get(): T {
        return this.item;
    }
}

let stringBox = new Box<string>();
stringBox.set("Hello");
console.log(stringBox.get());  // Hello

let numberBox = new Box<number>();
numberBox.set(42);
console.log(numberBox.get());  // 42
```

**이점**: 코드 재사용, 타입 안전성, 런타임 오버헤드 없음 (Java는 타입 소거, C++는 템플릿 인스턴스화).

---

## 대수적 데이터 타입(Algebraic Data Types)

### 합 타입(Sum Types) (열거형, 태그 유니온)

값은 **여러 변형 중 하나**가 될 수 있습니다.

**Rust enum:**
```rust
enum Status {
    Success,
    Error(String),
    Loading,
}

let result = Status::Error("Network timeout".to_string());

match result {
    Status::Success => println!("Success!"),
    Status::Error(msg) => println!("Error: {}", msg),
    Status::Loading => println!("Loading..."),
}
```

**TypeScript 식별 유니온:**
```javascript
type Status =
    | { kind: "success"; data: string }
    | { kind: "error"; message: string }
    | { kind: "loading" };

function handleStatus(status: Status) {
    switch (status.kind) {
        case "success":
            console.log("Data:", status.data);
            break;
        case "error":
            console.log("Error:", status.message);
            break;
        case "loading":
            console.log("Loading...");
            break;
    }
}
```

### 곱 타입(Product Types) (튜플, 레코드)

값은 **여러 필드를 함께** 포함합니다.

**Rust struct:**
```rust
struct Point {
    x: i32,
    y: i32,
}

let p = Point { x: 10, y: 20 };
println!("({}, {})", p.x, p.y);
```

**TypeScript:**
```javascript
type Point = {
    x: number;
    y: number;
};

let p: Point = { x: 10, y: 20 };
console.log(`(${p.x}, ${p.y})`);
```

---

## Null과 그 문제들

**10억 달러 실수** — Tony Hoare (null 참조 발명자):

> "나는 그것을 나의 10억 달러 실수라고 부릅니다. 1965년에 null 참조를 발명한 것이었습니다... 이것은 셀 수 없이 많은 오류, 취약점, 시스템 충돌을 초래했습니다."

### 문제

```java
String name = getUserName();
int length = name.length();  // NullPointerException if name is null
```

많은 언어에서 변수가 `null`일 수 있어 런타임 충돌로 이어집니다.

### 해결책: Option/Maybe 타입

**Rust의 Option<T>:**
```rust
fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

let result = divide(10, 2);
match result {
    Some(value) => println!("Result: {}", value),
    None => println!("Cannot divide by zero"),
}

// Or use combinators
let result = divide(10, 2).unwrap_or(0);  // Default to 0 if None
```

**Java의 Optional<T>:**
```java
import java.util.Optional;

public Optional<String> findUserName(int id) {
    if (id == 1) {
        return Optional.of("Alice");
    } else {
        return Optional.empty();
    }
}

Optional<String> name = findUserName(1);
name.ifPresent(n -> System.out.println("Name: " + n));

String result = name.orElse("Unknown");  // Default value
```

**TypeScript:**
```javascript
function divide(a: number, b: number): number | null {
    return b === 0 ? null : a / b;
}

let result = divide(10, 2);
if (result !== null) {
    console.log("Result:", result);
} else {
    console.log("Cannot divide by zero");
}
```

**이점**: 값의 부재를 명시적으로 처리하도록 강제합니다. 더 이상 `NullPointerException` 놀라움이 없습니다.

---

## 타입 주석과 문서화

동적 타입 언어에서도 타입을 문서화할 수 있고 (해야 합니다).

### Python 타입 힌트

```python
def greet(name: str) -> str:
    """
    Greet a person by name.

    Args:
        name: The person's name

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"

# Type checker (mypy) can catch errors:
# greet(42)  # Error: Argument 1 has incompatible type "int"; expected "str"
```

### JavaScript with JSDoc

```javascript
/**
 * Calculate the area of a rectangle
 * @param {number} width - The width
 * @param {number} height - The height
 * @returns {number} The area
 */
function area(width, height) {
    return width * height;
}
```

### TypeScript

```javascript
function area(width: number, height: number): number {
    return width * height;
}

// area("5", 10);  // Error: Argument of type 'string' is not assignable to 'number'
```

---

## 연습 문제

### 연습 문제 1: 타입 시스템 분석

이 JavaScript 코드가 주어졌을 때:
```javascript
let x = "10";
let y = 5;
console.log(x + y);  // "105"
console.log(x - y);  // 5
```

1. 왜 `+`는 `"105"`를 생성하지만 `-`는 `5`를 생성하나요?
2. Python에서 작동할까요? 왜 또는 왜 안 될까요?
3. 이것은 강한 타이핑인가요 약한 타이핑인가요? 정적인가요 동적인가요?

### 연습 문제 2: 스택 ADT 구현

선택한 언어로 스택 ADT를 구현하세요:
- 배열을 기본 자료구조로 사용
- `push`, `pop`, `peek`, `is_empty` 구현
- 정수로 테스트하고, 그 다음 문자열로 테스트 (같은 코드가 작동해야 함)

### 연습 문제 3: 제네릭

잠재적으로 다른 타입의 두 값을 담는 제네릭 `Pair<T, U>` 클래스 구현:
- 생성자: `Pair(T first, U second)`
- 메서드: `getFirst()`, `getSecond()`, `setFirst(T)`, `setSecond(U)`
- 테스트: ("Alice", 30)에 대한 `Pair<String, Integer>`

### 연습 문제 4: Option 타입

간단한 `Option<T>` 타입 구현 (Rust나 Java와 유사):
- `Some(value)`: 값 포함
- `None`: 비어있음
- 메서드:
  - `isSome()`: Some이면 true 반환
  - `isNone()`: None이면 true 반환
  - `unwrap()`: 값 반환 또는 None이면 오류 던짐
  - `unwrapOr(default)`: 값 반환 또는 None이면 기본값

### 연습 문제 5: ADT 설계

**도서관 시스템**을 위한 ADT 설계:
- 어떤 연산을 지원해야 하나요?
  - 책 추가
  - 책 제거
  - 제목, 저자, ISBN으로 검색
  - 책 대출
  - 책 반납
- 인터페이스 정의 (아직 구현하지 않음)
- 고려사항: 어떤 자료구조가 이것을 구현할 수 있나요?

### 연습 문제 6: Null 안전성

이 Java 코드를 `Optional`을 사용하도록 리팩토링하세요:

```java
public String getUserEmail(int userId) {
    if (userId == 1) {
        return "alice@example.com";
    }
    return null;
}

String email = getUserEmail(1);
System.out.println(email.toUpperCase());  // Potential NullPointerException
```

`Optional<String>`을 사용해 안전하게 만드세요.

---

## 요약

- **타입은 데이터를 분류**: 기본형 (int, float, bool, char, string), 복합형 (배열, 구조체, 튜플)
- **타입 시스템**:
  - **정적 vs 동적**: 컴파일 시점 vs 런타임 검사
  - **강한 vs 약한**: 엄격한 vs 허용적 타입 규칙
  - **타입 추론**: 자동 타입 추론
- **추상 데이터 타입**: 인터페이스 (무엇) vs 구현 (어떻게)
  - 스택, 큐, 맵/딕셔너리
- **제네릭/템플릿**: 재사용성과 타입 안전성을 위한 타입 매개변수화 코드
- **대수적 데이터 타입**: 합 타입 (열거형), 곱 타입 (튜플/레코드)
- **Null 안전성**: `Option`/`Maybe`/`Optional` 타입으로 null 관련 충돌 방지
- **문서화**: 타입 주석으로 코드를 더 명확하게 하고 오류를 일찍 포착

**핵심 통찰**: 추상화는 복잡성 관리에 관한 것입니다. ADT는 구현 세부사항을 걱정하지 않고 더 높은 수준에서 생각할 수 있게 합니다. 타입은 오류를 포착하고 코드를 더 유지보수 가능하게 만드는 데 도움이 됩니다.

---

## 탐색

[← 이전: Programming Paradigms](02_Programming_Paradigms.md) | [다음: Control Flow Patterns →](04_Control_Flow_Patterns.md)

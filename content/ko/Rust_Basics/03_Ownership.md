# 03. 소유권

**이전**: [변수와 타입](./02_Variables_and_Types.md) | **다음**: [빌림과 참조](./04_Borrowing_and_References.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 세 가지 소유권(Ownership) 규칙과 Rust가 이를 강제하는 이유를 설명할 수 있다
2. 스택(Stack) 할당과 힙(Heap) 할당을 구별할 수 있다
3. 이동 의미론(Move Semantics)이 힙 할당 데이터의 소유권을 어떻게 이전하는지 추적할 수 있다
4. 이동(Move) 대신 `Copy` 트레이트가 사용되는 경우를 파악할 수 있다
5. 명시적 깊은 복사(Deep Copy)를 위해 `Clone`을 구현할 수 있다

---

소유권(Ownership)은 Rust의 핵심 혁신입니다 — 가비지 컬렉터(Garbage Collector) 없이 메모리 안전성을 보장하는 컴파일 타임 시스템입니다. 다른 모든 Rust 개념(빌림(Borrowing), 라이프타임(Lifetime), 스마트 포인터(Smart Pointer))은 다음 세 가지 규칙 위에 구축됩니다:

1. **각 값은 정확히 하나의 소유자(Owner)를 가집니다.**
2. **한 번에 하나의 소유자만 존재할 수 있습니다.**
3. **소유자가 스코프(Scope)를 벗어나면, 값은 해제(drop)됩니다.**

## 목차
1. [스택 vs 힙](#1-스택-vs-힙)
2. [세 가지 소유권 규칙](#2-세-가지-소유권-규칙)
3. [이동 의미론](#3-이동-의미론)
4. [Copy와 Clone](#4-copy와-clone)
5. [소유권과 함수](#5-소유권과-함수)
6. [연습 문제](#6-연습-문제)

---

## 1. 스택 vs 힙

데이터가 어디에 위치하는지 이해하는 것은 소유권을 이해하는 데 필수적입니다.

```
STACK (fast, fixed-size)              HEAP (flexible, dynamic-size)
┌──────────────────────┐              ┌──────────────────────────┐
│ Push/pop operations   │              │ Allocator finds free     │
│ Last-in, first-out    │              │ space, returns pointer   │
│ Size known at compile │              │ Size determined at       │
│ time                  │              │ runtime                  │
│                       │  pointer     │                          │
│  x: i32 = 42         │──────────┐   │                          │
│  ptr ────────────────────────── │──>│  "hello world" (bytes)   │
│  len: 11              │         │   │                          │
│  capacity: 11         │         │   │                          │
└──────────────────────┘              └──────────────────────────┘
```

| 속성 | 스택(Stack) | 힙(Heap) |
|------|-------|------|
| 속도 | 매우 빠름 (포인터 이동) | 느림 (할당자 탐색) |
| 크기 | 컴파일 타임에 결정 | 런타임에 결정 |
| 접근 | 직접 접근 | 포인터를 통한 간접 접근 |
| 예시 | `i32`, `f64`, `bool`, `[i32; 5]` | `String`, `Vec<T>`, `Box<T>` |
| 정리 | 자동 (스코프 종료 시) | `drop` 필요 (Rust) 또는 GC/수동 해제 |

---

## 2. 세 가지 소유권 규칙

### 규칙 1: 각 값은 정확히 하나의 소유자를 가진다

```rust
fn main() {
    let s = String::from("hello");  // s owns the String
    // The String data lives on the heap
    // The variable s (on the stack) holds: pointer, length, capacity
}
```

### 규칙 2: 한 번에 하나의 소유자만 존재

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // Ownership MOVES from s1 to s2
    // println!("{s1}");  // ERROR: s1 is no longer valid
    println!("{s2}");     // OK: s2 is the owner now
}
```

### 규칙 3: 소유자가 스코프를 벗어나면 값이 해제된다

```rust
fn main() {
    {
        let s = String::from("hello");
        // s is valid here
    }
    // s is out of scope — Rust calls drop(s), freeing heap memory
    // This is like C++'s RAII (Resource Acquisition Is Initialization)
}
```

---

## 3. 이동 의미론

힙 할당 값을 다른 변수에 대입하면, Rust는 그것을 **이동(Move)**합니다 — 원래 변수는 무효화됩니다:

```rust
fn main() {
    let s1 = String::from("hello");

    // BEFORE move:
    // Stack:                    Heap:
    // s1: [ptr, len=5, cap=5] → "hello"

    let s2 = s1;

    // AFTER move:
    // Stack:                    Heap:
    // s1: [INVALID]
    // s2: [ptr, len=5, cap=5] → "hello"

    // Why not copy? Because two owners would mean double-free:
    // If both s1 and s2 owned the same heap data, dropping both
    // would free the same memory twice → undefined behavior.
    // Move prevents this at compile time.
}
```

다음 다이어그램은 메모리에서 발생하는 일을 보여줍니다:

```
Before: let s1 = String::from("hello");

  s1 ─────────┐
  [ptr|5|5]   │
              ▼
         ┌─────────┐
         │ h e l l o│
         └─────────┘

After: let s2 = s1;

  s1 ─────────╳  (invalidated)
  s2 ─────────┐
  [ptr|5|5]   │
              ▼
         ┌─────────┐
         │ h e l l o│
         └─────────┘
```

---

## 4. Copy와 Clone

### 4.1 Copy 트레이트 — 암묵적 비트 단위 복사

`Copy`를 구현한 타입은 이동 대신 대입 시 복제됩니다. 이것들은 복사 비용이 저렴한 소규모 스택 전용 타입입니다:

```rust
fn main() {
    let x: i32 = 42;
    let y = x;      // Copy, not move — x is still valid
    println!("x={x}, y={y}");  // x=42, y=42

    // Types that implement Copy:
    // - All integer types (i8, u8, i32, u64, etc.)
    // - Floating-point types (f32, f64)
    // - bool
    // - char
    // - Tuples of Copy types: (i32, f64) is Copy, (i32, String) is NOT
    // - Fixed-size arrays of Copy types: [i32; 5] is Copy
}
```

타입이 `Copy`를 구현하려면 모든 필드가 `Copy`여야 **하고** `Drop` 트레이트(커스텀 소멸자 트레이트)를 구현하지 않아야 합니다.

### 4.2 Clone 트레이트 — 명시적 깊은 복사

힙 할당 타입의 경우, `.clone()`을 사용하여 명시적 깊은 복사(Deep Copy)를 만듭니다:

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();  // Deep copy — new heap allocation
    println!("s1={s1}, s2={s2}");  // Both valid

    // After clone:
    // s1 → heap: "hello"  (original)
    // s2 → heap: "hello"  (copy — different allocation)
}
```

> **지침**: 클론(Clone)보다 빌림(Borrowing, 레슨 04)을 선호하세요. 진정으로 독립적인 소유권이 필요할 때만 클론하세요. 과도한 클론은 Rust의 제로 비용(zero-cost) 철학에 위배됩니다.

### 4.3 Copy와 Clone 파생하기

```rust
// Your own types can be Copy if all fields are Copy
#[derive(Debug, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

// This CANNOT be Copy because String is not Copy
#[derive(Debug, Clone)]
struct Person {
    name: String,  // Heap-allocated → no Copy
    age: u32,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = p1;  // Copy — p1 is still valid
    println!("{p1:?}");

    let alice = Person { name: String::from("Alice"), age: 30 };
    let bob = alice.clone();  // Must explicitly clone
    // let bob = alice;       // This would MOVE, invalidating alice
    println!("{alice:?}");
}
```

---

## 5. 소유권과 함수

### 5.1 함수에 소유권 전달하기

```rust
fn take_ownership(s: String) {
    println!("Got: {s}");
}   // s is dropped here — heap memory freed

fn make_copy(n: i32) {
    println!("Got: {n}");
}   // n is dropped, but it was just a stack copy

fn main() {
    let greeting = String::from("hello");
    take_ownership(greeting);
    // println!("{greeting}");  // ERROR: greeting was moved into the function

    let num = 42;
    make_copy(num);
    println!("{num}");  // OK: i32 implements Copy
}
```

### 5.2 소유권 반환하기

```rust
fn create_string() -> String {
    let s = String::from("created");
    s   // Ownership moves to the caller
}

fn take_and_give_back(s: String) -> String {
    println!("Borrowed briefly: {s}");
    s   // Return ownership to caller
}

fn main() {
    let s1 = create_string();       // s1 owns "created"
    let s2 = take_and_give_back(s1); // s1 → function → s2
    println!("{s2}");
}
```

> 소유권을 넘겼다가 다시 받는 것은 번거롭습니다. 다음 레슨에서는 **빌림(Borrowing)**을 소개합니다 — 소유권을 가져가지 않고도 함수가 데이터에 접근할 수 있는 방법입니다.

### 5.3 여러 값 반환하기

```rust
fn calculate_length(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)  // Return both the String AND the length
}

fn main() {
    let s = String::from("hello");
    let (s, len) = calculate_length(s);
    println!("'{s}' has length {len}");
}
```

---

## 6. 연습 문제

### 연습 1: 이동 예측
코드를 실행하지 말고, 어떤 `println!` 구문이 컴파일되고 어떤 것이 오류를 일으킬지 예측하세요. 그런 다음 `cargo check`로 검증하세요.

```rust
fn main() {
    let a = String::from("hello");
    let b = a;
    let c = b;
    println!("{a}");  // ?
    println!("{b}");  // ?
    println!("{c}");  // ?
}
```

### 연습 2: Copy vs Move
`x * 2`를 반환하는 `double(x: i32) -> i32` 함수를 작성하세요. 변수와 함께 호출하고 원래 변수를 여전히 사용할 수 있음을 확인하세요. 그런 다음 "!"를 추가하는 `exclaim(s: String) -> String`을 작성하여 — 문자열을 계속 사용하려면 반환 값을 캡처해야 함을 보여주세요.

### 연습 3: 소유권 이전
String의 소유권을 받아, 첫 번째 단어를 추출하고, 첫 번째 단어와 나머지 텍스트를 모두 반환하는 `first_word(s: String) -> (String, String)` 함수를 작성하세요. `main`에서 사용법을 보여주세요.

### 연습 4: Clone의 트레이드오프
1000개의 원소를 가진 `Vec<String>`이 있을 때, `let v2 = v1;` (이동)과 `let v2 = v1.clone();` (깊은 복사)의 성능 차이를 설명하세요. 클론이 정당화되는 경우는 언제인가요?

### 연습 5: 커스텀 Copy 타입
`r: u8, g: u8, b: u8`를 가진 `Color` 구조체를 정의하고 `Copy`와 `Clone`을 파생하세요. `Color`를 값으로 받는 함수를 작성하고 호출 후에도 원래 값에 접근 가능한지 확인하세요.

---

## 참고 자료
- [The Rust Book — Ownership](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
- [Rust by Example — Ownership](https://doc.rust-lang.org/rust-by-example/scope/move.html)

---

**이전**: [변수와 타입](./02_Variables_and_Types.md) | **다음**: [빌림과 참조](./04_Borrowing_and_References.md)

# 04. 빌림과 참조(Borrowing and References)

**이전**: [소유권(Ownership)](./03_Ownership.md) | **다음**: [슬라이스(Slices)](./05_Slices.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 불변(`&T`) 및 가변(`&mut T`) 참조(Reference)를 생성한다
2. 두 가지 빌림(Borrowing) 규칙을 설명하고, 그것이 데이터 경쟁을 막는 이유를 이해한다
3. 댕글링 참조(Dangling Reference) 오류를 식별하고 수정한다
4. 참조 전달, 값 전달, 가변 참조 전달 중 적절한 방식을 선택한다

---

이전 레슨에서는 함수에 소유권(Ownership)을 넘기면 불편하다는 점을 보여줬습니다 — 접근권을 되찾으려면 값을 반환해야 합니다. **빌림(Borrowing)**은 이 문제를 해결합니다: 소유권을 이전하지 않고도 참조(Reference)를 통해 함수가 데이터를 *읽거나* *수정*할 수 있게 합니다. 컴파일러는 두 가지 규칙을 통해 데이터 경쟁을 컴파일 시점에 제거합니다.

## 목차
1. [참조와 빌림](#1-참조와-빌림)
2. [두 가지 빌림 규칙](#2-두-가지-빌림-규칙)
3. [가변 참조](#3-가변-참조)
4. [댕글링 참조](#4-댕글링-참조)
5. [참조 패턴](#5-참조-패턴)
6. [연습 문제](#6-연습-문제)

---

## 1. 참조와 빌림

**참조(Reference)**는 값을 소유하지 않고 빌리는 포인터입니다. 참조가 가리키는 값은 참조가 스코프를 벗어나도 해제(drop)되지 않습니다.

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
}   // s goes out of scope, but it doesn't own the String — nothing is dropped

fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // &s1 creates a reference
    println!("'{s1}' has length {len}");  // s1 is still valid!
}
```

```
Ownership vs Borrowing:

  Ownership (move):                 Borrowing (reference):
  main → fn: s moved in            main → fn: &s lent temporarily
  main loses access                 main keeps ownership
  fn must return to give back       fn returns, reference expires
```

---

## 2. 두 가지 빌림 규칙

Rust는 컴파일 시점에 다음 규칙을 강제합니다:

> **규칙 1**: 가변 참조(Mutable Reference)는 **하나만** 존재하거나, 불변 참조(Immutable Reference)는 **개수에 제한 없이** 존재할 수 있습니다 — 단, 동시에 둘 다는 불가능합니다.
>
> **규칙 2**: 참조는 항상 **유효해야** 합니다(댕글링 포인터 불가).

```rust
fn main() {
    let mut s = String::from("hello");

    // Multiple immutable references — OK
    let r1 = &s;
    let r2 = &s;
    println!("{r1}, {r2}");  // Both valid

    // Mutable reference after immutable ones are done — OK
    // (r1 and r2 are no longer used after this point — NLL)
    let r3 = &mut s;
    r3.push_str(", world");
    println!("{r3}");

    // Simultaneous mutable + immutable — ERROR
    // let r4 = &s;
    // let r5 = &mut s;
    // println!("{r4}, {r5}");  // ERROR: cannot borrow as mutable
}
```

### 왜 이런 규칙이 필요한가?

이 규칙들은 **데이터 경쟁(Data Race)**을 컴파일 시점에 방지합니다. 데이터 경쟁은 다음 조건이 충족될 때 발생합니다:
1. 두 개 이상의 포인터가 동일한 데이터에 동시에 접근
2. 그 중 적어도 하나가 쓰기 작업
3. 동기화 메커니즘 없음

Rust의 규칙(쓰기는 하나 또는 읽기는 여러 개)은 조건 1+2를 불가능하게 만듭니다.

### 비어휘적 생명주기(NLL, Non-Lexical Lifetimes)

컴파일러는 참조가 스코프를 벗어나는 시점이 아니라, **마지막으로 사용되는** 시점을 추적합니다:

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;
    let r2 = &s;
    println!("{r1}, {r2}");
    // r1 and r2 are not used after this line — their lifetimes end here

    let r3 = &mut s;   // OK! No conflict because r1, r2 are "dead"
    r3.push_str("!");
    println!("{r3}");
}
```

---

## 3. 가변 참조

### 3.1 기본 가변 빌림

```rust
fn append_greeting(s: &mut String) {
    s.push_str(", world!");
}

fn main() {
    let mut s = String::from("hello");  // Variable must be mut
    append_greeting(&mut s);             // Pass mutable reference
    println!("{s}");  // hello, world!
}
```

### 3.2 가변 참조는 한 번에 하나만

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &mut s;
    // let r2 = &mut s;  // ERROR: cannot borrow s as mutable more than once
    r1.push_str("!");
    println!("{r1}");

    // After r1 is no longer used, we can create a new mutable reference
    let r2 = &mut s;
    r2.push_str("!");
    println!("{r2}");
}
```

### 3.3 함수에서의 가변 참조

```rust
fn swap_values(a: &mut i32, b: &mut i32) {
    let temp = *a;  // Dereference to get the value
    *a = *b;
    *b = temp;
}

fn main() {
    let mut x = 1;
    let mut y = 2;
    swap_values(&mut x, &mut y);
    println!("x={x}, y={y}");  // x=2, y=1
}
```

---

## 4. 댕글링 참조

댕글링 참조(Dangling Reference)는 이미 해제된 메모리를 가리킵니다. Rust는 이를 컴파일 시점에 방지합니다:

```rust
// This does NOT compile:
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // ERROR: s is dropped at end of function, reference would dangle
// }

// Solution: return the owned value instead
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // Ownership moves to caller — no dangling reference
}

fn main() {
    let s = no_dangle();
    println!("{s}");
}
```

---

## 5. 참조 패턴

### 5.1 올바른 매개변수 타입 선택

```rust
// Takes ownership — caller loses access
fn consume(s: String) { /* ... */ }

// Immutable borrow — caller keeps ownership, function can read
fn inspect(s: &String) { /* ... */ }

// Mutable borrow — caller keeps ownership, function can modify
fn modify(s: &mut String) { /* ... */ }

// General guideline:
// Use &T    when you only need to read
// Use &mut T when you need to modify
// Use T      when you need ownership (storing in a struct, spawning a thread, etc.)
```

### 5.2 역참조(Dereference) 연산자

```rust
fn main() {
    let x = 5;
    let r = &x;

    // Explicit dereference
    assert_eq!(*r, 5);

    // Rust auto-dereferences in many contexts (dot operator, comparisons)
    let s = String::from("hello");
    let r = &s;
    println!("{}", r.len());  // Auto-deref: same as (*r).len()
}
```

### 5.3 참조의 참조

```rust
fn main() {
    let x = 42;
    let r1 = &x;       // &i32
    let r2 = &r1;      // &&i32
    let r3 = &r2;      // &&&i32

    // Rust auto-dereferences through multiple levels
    assert_eq!(***r3, 42);
    assert_eq!(**r2, 42);
    println!("{r3}");  // 42 — Display auto-derefs
}
```

---

## 6. 연습 문제

### 연습 1: 빌림 검사기 오류 수정
다음 코드가 컴파일되도록 수정하세요:

```rust
fn main() {
    let mut s = String::from("hello");
    let r1 = &s;
    let r2 = &mut s;
    println!("{r1}, {r2}");
}
```

### 연습 2: 문자열 수정자
문자열을 제자리에서 대문자로 변환하는 `make_uppercase(s: &mut String)` 함수를 작성하세요. `main`에서 호출하고 결과를 출력하세요.

### 연습 3: 개수 세기 함수
소유권을 가져가지 않고 `s`에서 `c`의 등장 횟수를 세는 `count_char(s: &str, c: char) -> usize`를 작성하세요. 함수 호출 후에도 원본 문자열을 여전히 사용할 수 있음을 시연하세요.

### 연습 4: 참조 생명주기
다음 함수 시그니처는 유효하지만 아래 구현이 유효하지 않은 이유를 설명하세요:

```rust
fn longest(a: &str, b: &str) -> &str {
    if a.len() > b.len() { a } else { b }
}
```

컴파일러가 요구하는 추가 어노테이션은 무엇인가요? (레슨 11의 미리보기.)

### 연습 5: 소유권 결정
각 시나리오에 대해 함수가 `T`, `&T`, `&mut T` 중 어떤 타입을 받아야 할지 결정하세요:
1. 사용자 이름을 출력하는 함수
2. 장바구니에 항목을 추가하는 함수
3. 전역 레지스트리에 설정 객체를 저장하는 함수
4. 바이트 슬라이스의 해시를 계산하는 함수

---

## 참고 자료
- [The Rust Book — References and Borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- [Rust by Example — Borrowing](https://doc.rust-lang.org/rust-by-example/scope/borrow.html)

---

**이전**: [소유권(Ownership)](./03_Ownership.md) | **다음**: [슬라이스(Slices)](./05_Slices.md)

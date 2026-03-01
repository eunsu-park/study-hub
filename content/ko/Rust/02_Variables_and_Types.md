# 02. 변수와 타입

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [소유권](./03_Ownership.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `let`과 `let mut`을 사용하여 불변(immutable) 및 가변(mutable) 변수를 선언할 수 있다
2. 섀도잉(Shadowing)을 사용하여 새로운 타입이나 값으로 변수를 재바인딩(rebind)할 수 있다
3. 스칼라 타입(정수, 부동소수점, bool, char)과 복합 타입(튜플, 배열)을 구별할 수 있다
4. 타입 추론(Type Inference)과 명시적 타입 어노테이션을 적절히 적용할 수 있다
5. Rust가 기본적으로 불변성(Immutability)을 택하는 이유와 `const`가 `let`과 어떻게 다른지 설명할 수 있다

---

Rust의 타입 시스템은 가장 큰 강점 중 하나입니다. 모든 값은 컴파일 타임에 타입이 결정되지만, 타입을 명시적으로 작성할 필요는 거의 없습니다 — 컴파일러가 추론해 줍니다. 기본 불변성과 결합하여, 이는 간결하면서도 우발적인 변이(Mutation) 버그에 강한 코드를 만들어 냅니다.

## 목차
1. [변수와 가변성](#1-변수와-가변성)
2. [스칼라 타입](#2-스칼라-타입)
3. [복합 타입](#3-복합-타입)
4. [타입 변환](#4-타입-변환)
5. [상수와 정적 변수](#5-상수와-정적-변수)
6. [연습 문제](#6-연습-문제)

---

## 1. 변수와 가변성

### 1.1 기본적으로 불변

```rust
fn main() {
    let x = 5;
    // x = 6;  // ERROR: cannot assign twice to immutable variable
    println!("x = {x}");
}
```

왜 불변이 기본값일까요? 변수가 절대 변하지 않으면, 코드를 지역적으로 추론할 수 있습니다 — 프로그램 전체에서 발생 가능한 모든 변이를 추적할 필요가 없습니다. 컴파일러가 이 보장을 강제합니다.

### 1.2 가변 변수

```rust
fn main() {
    let mut counter = 0;  // mut keyword opts into mutability
    counter += 1;
    counter += 1;
    println!("counter = {counter}");  // 2
}
```

### 1.3 섀도잉(Shadowing)

섀도잉은 같은 이름으로 변수를 재선언합니다. 변이(Mutation)와 달리, 섀도잉은 타입을 변경할 수 있습니다:

```rust
fn main() {
    let x = 5;          // x is i32
    let x = x + 1;      // New x shadows the old one (still i32)
    let x = x * 2;      // Shadows again
    println!("x = {x}"); // 12

    // Shadowing can change the type
    let spaces = "   ";         // &str
    let spaces = spaces.len();  // usize — different type, same name
    println!("spaces = {spaces}"); // 3

    // With mut, you CANNOT change the type:
    // let mut s = "hello";
    // s = s.len();  // ERROR: expected &str, found usize
}
```

섀도잉은 의미 있는 이름을 유지하면서 값을 단계적으로 변환하는 파이프라인에서 Rust의 관용적(idiomatic) 패턴입니다.

### 1.4 타입 어노테이션

```rust
fn main() {
    // Explicit type annotation
    let x: i32 = 42;
    let pi: f64 = 3.14159;
    let active: bool = true;

    // Type inference — compiler deduces the type
    let y = 42;        // inferred as i32 (default integer type)
    let z = 3.14;      // inferred as f64 (default float type)
    let name = "Rust";  // inferred as &str
}
```

---

## 2. 스칼라 타입

스칼라 타입은 하나의 값을 나타냅니다.

### 2.1 정수 타입

| 크기 | 부호 있음(Signed) | 부호 없음(Unsigned) |
|------|--------|----------|
| 8비트 | `i8` | `u8` |
| 16비트 | `i16` | `u16` |
| 32비트 | `i32` (기본값) | `u32` |
| 64비트 | `i64` | `u64` |
| 128비트 | `i128` | `u128` |
| 포인터 크기 | `isize` | `usize` |

```rust
fn main() {
    let decimal = 98_222;      // Underscores for readability
    let hex = 0xff;            // Hexadecimal
    let octal = 0o77;          // Octal
    let binary = 0b1111_0000;  // Binary
    let byte = b'A';           // Byte literal (u8 only)

    // Integer overflow behavior:
    // - Debug mode:   panics at runtime
    // - Release mode: wraps around (two's complement)
    // Use wrapping_*, checked_*, overflowing_*, saturating_* for explicit control
    let max: u8 = 255;
    let wrapped = max.wrapping_add(1);  // 0
    let saturated = max.saturating_add(1);  // 255
    let checked = max.checked_add(1);  // None
}
```

### 2.2 부동소수점 타입

```rust
fn main() {
    let x = 2.0;      // f64 (default, double precision)
    let y: f32 = 3.0;  // f32 (single precision)

    // Arithmetic
    let sum = 5.0 + 10.0;
    let difference = 95.5 - 4.3;
    let product = 4.0 * 30.0;
    let quotient = 56.7 / 32.2;
    let remainder = 43.0 % 5.0;  // 3.0

    // f64 is generally preferred — same speed as f32 on modern CPUs
    // but more precision (15-17 significant digits vs 6-9)
}
```

### 2.3 불리언(Boolean)과 문자(Character)

```rust
fn main() {
    // Boolean — 1 byte
    let t: bool = true;
    let f = false;

    // Character — 4 bytes (Unicode scalar value)
    let c = 'z';
    let emoji = '🦀';
    let hangul = '가';

    // char represents a Unicode scalar value (U+0000 to U+D7FF, U+E000 to U+10FFFF)
    // This is NOT the same as a byte — a char is always 4 bytes in Rust
    println!("size of char: {} bytes", std::mem::size_of::<char>());  // 4
}
```

---

## 3. 복합 타입

### 3.1 튜플(Tuples)

튜플은 서로 다른 타입의 값들을 하나의 복합 값으로 묶습니다. 고정 길이를 가집니다.

```rust
fn main() {
    // Creating a tuple
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    // Destructuring
    let (x, y, z) = tup;
    println!("y = {y}");  // 6.4

    // Index access (zero-based)
    let five_hundred = tup.0;
    let six_point_four = tup.1;
    let one = tup.2;

    // Unit tuple — the empty tuple () is Rust's "void"
    let unit: () = ();
    // Functions without a return value implicitly return ()
}
```

### 3.2 배열(Arrays)

배열은 고정 길이를 가지며 같은 타입의 원소를 저장합니다. **스택(Stack)**에 위치합니다.

```rust
fn main() {
    // Array declaration
    let a: [i32; 5] = [1, 2, 3, 4, 5];

    // Initialize with same value
    let zeros = [0; 10];  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    // Access elements
    let first = a[0];
    let second = a[1];

    // Rust checks array bounds at runtime
    // let invalid = a[10];  // Panics: index out of bounds

    // Array length
    println!("length: {}", a.len());  // 5

    // Iteration
    for element in &a {
        println!("{element}");
    }

    // Arrays are useful when you know the exact size at compile time
    // For dynamic sizes, use Vec<T> (covered in Lesson 08)
}
```

### 3.3 문자열(Strings) (미리 보기)

Rust에는 두 가지 주요 문자열 타입이 있습니다. 이것은 간략한 소개이며 — 레슨 05 (슬라이스)에서 심층적으로 다룹니다.

```rust
fn main() {
    // &str — string slice, immutable reference to string data
    let greeting: &str = "Hello, world!";  // Stored in binary

    // String — heap-allocated, growable, owned string
    let mut name = String::from("Rust");
    name.push_str(" Programming");
    println!("{name}");  // Rust Programming

    // Converting between them
    let s: String = greeting.to_string();
    let slice: &str = &s;
}
```

---

## 4. 타입 변환

Rust는 숫자 타입 간에 암묵적 타입 변환(강제 변환)이 없습니다. 명시적으로 해야 합니다:

```rust
fn main() {
    // `as` keyword for primitive casts
    let x: i32 = 42;
    let y: f64 = x as f64;
    let z: u8 = x as u8;  // May truncate!

    // Safer conversions with From/Into traits
    let a: i32 = 5;
    let b: i64 = i64::from(a);  // Infallible widening conversion
    let c: i64 = a.into();      // Same thing, using Into trait

    // TryFrom for fallible conversions
    let big: i64 = 1_000_000;
    let small: Result<i32, _> = i32::try_from(big);  // Ok(1000000)

    let too_big: i64 = 5_000_000_000;
    let fail: Result<i32, _> = i32::try_from(too_big);  // Err(...)

    // String to number
    let parsed: i32 = "42".parse().expect("not a number");
    let pi: f64 = "3.14".parse().unwrap();
}
```

---

## 5. 상수와 정적 변수

### 5.1 상수(Constants)

```rust
// Constants must have a type annotation and be known at compile time
const MAX_POINTS: u32 = 100_000;
const PI: f64 = 3.141_592_653_589_793;

fn main() {
    // Constants are inlined at each usage site
    println!("Max: {MAX_POINTS}");
}
```

### 5.2 정적 변수(Static Variables)

```rust
// Static variables have a fixed memory address for the entire program
static LANGUAGE: &str = "Rust";
static mut COUNTER: u32 = 0;  // Mutable statics require unsafe to access

fn main() {
    println!("{LANGUAGE}");

    // Mutable statics are inherently unsafe (data races possible)
    unsafe {
        COUNTER += 1;
        println!("COUNTER = {COUNTER}");
    }
}
```

| 특징 | `const` | `static` | `let` |
|------|---------|----------|-------|
| 스코프 | 모든 곳 | 모든 곳 | 블록 내 |
| 메모리 | 인라인(Inlined) | 고정 주소 | 스택(Stack) |
| 가변성 | 불가 | `static mut` (unsafe) | `let mut` |
| 타입 어노테이션 | 필수 | 필수 | 선택 |
| 계산 시점 | 컴파일 타임 | 컴파일 타임 | 런타임 |

---

## 6. 연습 문제

### 연습 1: 변수 바인딩
`36.6` 값을 가진 변수 `temperature`를 선언한 후, `as`를 사용하여 정수 부분만 가진 값으로 섀도잉하세요. 원래 개념과 섀도잉된 값 모두 출력하세요.

### 연습 2: 튜플 구조 분해
세 수의 최솟값과 최댓값을 반환하는 `min_max(a: i32, b: i32, c: i32) -> (i32, i32)` 함수를 작성하세요. `main`에서 결과를 구조 분해(destructure)하세요.

### 연습 3: 배열 연산
12개월의 강수량 값(임의의 f64 값)으로 배열을 만드세요. 연간 총 강수량과 월평균 강수량을 계산하는 루프를 작성하세요.

### 연습 4: 타입 변환
`.parse()`를 사용하여 문자열 `"255"`를 `u8`로 변환하는 코드를 작성하세요. 그런 다음 `"256"` 파싱을 시도하고 `match`를 사용하여 오류를 우아하게 처리하세요. `.parse::<u8>()`이 실패할 때 어떤 오류 타입을 반환하나요?

### 연습 5: 오버플로우 동작
`u8::MAX`에서 `wrapping_add`, `checked_add`, `saturating_add`의 차이를 보여주세요. 네트워크 패킷 시퀀스 카운터에는 어떤 메서드를 사용하겠습니까? 그 이유는?

---

## 참고 자료
- [The Rust Book — Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
- [The Rust Book — Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html)
- [Rust Reference — Types](https://doc.rust-lang.org/reference/types.html)

---

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [소유권](./03_Ownership.md)

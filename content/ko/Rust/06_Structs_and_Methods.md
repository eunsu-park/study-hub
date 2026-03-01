# 06. 구조체와 메서드(Structs and Methods)

**이전**: [슬라이스(Slices)](./05_Slices.md) | **다음**: [열거형과 패턴 매칭(Enums and Pattern Matching)](./07_Enums_and_Pattern_Matching.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 이름 있는 필드(Named Field), 튜플 구조체(Tuple Struct), 유닛 구조체(Unit Struct)를 정의한다
2. `impl` 블록을 사용하여 메서드(Method)와 연관 함수(Associated Function)를 구현한다
3. `#[derive]`를 사용하여 공통 트레이트(Trait)를 자동으로 구현한다
4. 수정된 복사본을 만들기 위해 구조체 업데이트 문법(Struct Update Syntax)을 적용한다
5. 구조체 소유권(Ownership)이 빌림(Borrowing)과 어떻게 상호작용하는지 이해한다

---

구조체(Struct)는 Rust에서 사용자 정의 데이터 타입을 만드는 주요 도구입니다. `impl` 블록과 함께 사용하면 OOP 언어의 클래스와 동일한 캡슐화를 제공합니다 — 단, 상속은 없습니다. Rust는 다형성을 위해 상속 대신 트레이트(레슨 10)를 사용하며 **컴포지션(Composition)을 상속(Inheritance)보다 선호**합니다.

## 목차
1. [구조체 정의](#1-구조체-정의)
2. [메서드와 연관 함수](#2-메서드와-연관-함수)
3. [파생 트레이트](#3-파생-트레이트)
4. [구조체 패턴](#4-구조체-패턴)
5. [연습 문제](#5-연습-문제)

---

## 1. 구조체 정의

### 1.1 이름 있는 필드 구조체

```rust
struct User {
    username: String,
    email: String,
    active: bool,
    sign_in_count: u64,
}

fn main() {
    // Creating an instance
    let user1 = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        active: true,
        sign_in_count: 1,
    };

    // Accessing fields
    println!("{}", user1.username);

    // Mutable struct — the entire struct must be mut
    let mut user2 = User {
        username: String::from("bob"),
        email: String::from("bob@example.com"),
        active: true,
        sign_in_count: 0,
    };
    user2.sign_in_count += 1;
}
```

### 1.2 필드 초기화 축약 문법과 업데이트 문법

```rust
fn build_user(username: String, email: String) -> User {
    User {
        username,        // Shorthand: field name matches parameter name
        email,
        active: true,
        sign_in_count: 1,
    }
}

fn main() {
    let user1 = build_user(String::from("alice"), String::from("alice@ex.com"));

    // Struct update syntax — create from existing, overriding some fields
    let user2 = User {
        email: String::from("bob@ex.com"),
        ..user1  // Remaining fields from user1
    };
    // Note: user1.username was MOVED into user2
    // user1.email was overridden, but user1.username is now invalid
    // user1.active and user1.sign_in_count are Copy, so they're fine
}
```

### 1.3 튜플 구조체

```rust
// Named tuples — useful for type distinction
struct Color(u8, u8, u8);
struct Point(f64, f64, f64);

fn main() {
    let red = Color(255, 0, 0);
    let origin = Point(0.0, 0.0, 0.0);

    // Access by index
    println!("R={}, G={}, B={}", red.0, red.1, red.2);

    // Destructuring
    let Point(x, y, z) = origin;
    println!("({x}, {y}, {z})");
}
```

### 1.4 유닛 구조체

```rust
// No fields — useful as markers or for trait implementations
struct AlwaysEqual;

fn main() {
    let _subject = AlwaysEqual;
}
```

---

## 2. 메서드와 연관 함수

### 2.1 메서드 정의

```rust
#[derive(Debug)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Method: takes &self (immutable borrow of the instance)
    fn area(&self) -> f64 {
        self.width * self.height
    }

    // Method with mutable self
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }

    // Method that takes ownership (rare)
    fn into_square(self) -> Rectangle {
        let side = self.width.max(self.height);
        Rectangle { width: side, height: side }
    }
}

fn main() {
    let mut rect = Rectangle { width: 30.0, height: 50.0 };
    println!("Area: {}", rect.area());

    rect.scale(2.0);
    println!("Scaled: {rect:?}");

    let square = rect.into_square();  // rect is consumed
    println!("Square: {square:?}");
}
```

### 2.2 연관 함수 (생성자)

```rust
impl Rectangle {
    // Associated function — no self parameter, called with ::
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn square(size: f64) -> Self {
        Self { width: size, height: size }
    }
}

fn main() {
    let rect = Rectangle::new(10.0, 20.0);
    let sq = Rectangle::square(15.0);
}
```

### 2.3 여러 impl 블록

```rust
// You can split methods across multiple impl blocks
// Useful for organizing code or conditional compilation
impl Rectangle {
    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

impl Rectangle {
    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }
}
```

### 2.4 메서드 호출 문법과 자동 참조

```rust
fn main() {
    let rect = Rectangle::new(10.0, 20.0);

    // These are equivalent — Rust adds &, &mut, or * automatically
    rect.area();       // Auto-borrows: (&rect).area()
    (&rect).area();    // Explicit borrow

    // This auto-referencing is why Rust doesn't have -> (like C++)
    // The compiler determines &self, &mut self, or self from the method signature
}
```

---

## 3. 파생 트레이트

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };

    // Debug — enables {:?} formatting
    println!("{p1:?}");           // Point { x: 1.0, y: 2.0 }
    println!("{p1:#?}");          // Pretty-printed

    // Clone — explicit deep copy
    let p2 = p1.clone();

    // PartialEq — enables == and !=
    assert_eq!(p1, p2);
}
```

파생 가능한 공통 트레이트:

| 트레이트(Trait) | 목적 | 활성화 기능 |
|----------------|------|------------|
| `Debug` | 디버그 포매팅 | `{:?}`, `{:#?}` |
| `Clone` | 명시적 깊은 복사 | `.clone()` |
| `Copy` | 암묵적 비트 단위 복사 | 이동 없이 대입 |
| `PartialEq` | 동등성 비교 | `==`, `!=` |
| `Eq` | 전체 동등성 (NaN 없음) | HashMap 키에 필요 |
| `Hash` | 해싱 | HashMap/HashSet 키 |
| `Default` | 기본값 | `Type::default()` |
| `PartialOrd` | 부분 순서 | `<`, `>`, `<=`, `>=` |
| `Ord` | 전체 순서 | 컬렉션에서 `.sort()` |

---

## 4. 구조체 패턴

### 4.1 빌더 패턴(Builder Pattern)

```rust
#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_secs: u64,
}

impl Config {
    fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

#[derive(Default)]
struct ConfigBuilder {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_secs: u64,
}

impl ConfigBuilder {
    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn max_connections(mut self, n: u32) -> Self {
        self.max_connections = n;
        self
    }

    fn build(self) -> Config {
        Config {
            host: if self.host.is_empty() { "localhost".to_string() } else { self.host },
            port: if self.port == 0 { 8080 } else { self.port },
            max_connections: if self.max_connections == 0 { 100 } else { self.max_connections },
            timeout_secs: if self.timeout_secs == 0 { 30 } else { self.timeout_secs },
        }
    }
}

fn main() {
    let config = Config::builder()
        .host("example.com")
        .port(3000)
        .max_connections(500)
        .build();
    println!("{config:#?}");
}
```

### 4.2 뉴타입 패턴(Newtype Pattern)

```rust
// Wrap a primitive to create a distinct type with domain meaning
struct Meters(f64);
struct Seconds(f64);

impl Meters {
    fn new(value: f64) -> Self {
        Self(value)
    }

    fn value(&self) -> f64 {
        self.0
    }
}

fn speed(distance: Meters, time: Seconds) -> f64 {
    distance.0 / time.0
}

fn main() {
    let d = Meters::new(100.0);
    let t = Seconds(9.58);
    // Cannot accidentally pass Seconds as Meters — the type system prevents it
    println!("Speed: {:.2} m/s", speed(d, t));
}
```

---

## 5. 연습 문제

### 연습 1: 원(Circle)
`radius: f64`를 가진 `Circle` 구조체를 정의하세요. `area()`, `circumference()` 메서드와 연관 함수 `new(radius: f64) -> Self`를 구현하세요.

### 연습 2: 학생 기록
`name: String`, `grades: Vec<f64>`를 가진 `Student` 구조체를 정의하세요. `average()`, `highest()`, `is_passing(threshold: f64) -> bool`을 구현하세요.

### 연습 3: 구조체 업데이트 문법
`ServerConfig` 구조체를 만드세요. 구조체 업데이트 문법을 사용하여 개발 설정으로부터 프로덕션 설정을 만드는 과정을 시연하고, 어떤 필드가 이동(Move)되고 어떤 필드가 복사(Copy)되는지 설명하세요.

### 연습 4: 뉴타입 강제
`Celsius(f64)`와 `Fahrenheit(f64)` 뉴타입을 생성하세요. 두 타입 간의 변환 메서드를 구현하세요. 컴파일러가 두 타입의 혼용을 막는다는 것을 보여주세요.

### 연습 5: 빌더 구현
`from`, `to`, `subject`, `body` 필드를 가진 `Email` 구조체의 빌더 패턴을 구현하세요. 빌더는 빌드 전에 `from`과 `to`가 비어 있지 않은지 검증해야 합니다.

---

## 참고 자료
- [The Rust Book — Structs](https://doc.rust-lang.org/book/ch05-00-structs.html)
- [Rust by Example — Structures](https://doc.rust-lang.org/rust-by-example/custom_types/structs.html)

---

**이전**: [슬라이스(Slices)](./05_Slices.md) | **다음**: [열거형과 패턴 매칭(Enums and Pattern Matching)](./07_Enums_and_Pattern_Matching.md)

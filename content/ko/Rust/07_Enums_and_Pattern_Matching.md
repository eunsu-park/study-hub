# 07. 열거형과 패턴 매칭(Enums and Pattern Matching)

**이전**: [구조체와 메서드(Structs and Methods)](./06_Structs_and_Methods.md) | **다음**: [컬렉션(Collections)](./08_Collections.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 유닛(Unit), 튜플(Tuple), 구조체(Struct) 배리언트(Variant)를 가진 열거형(Enum)을 정의한다
2. `Option<T>`를 사용하여 값의 부재를 안전하게 처리한다
3. 제어 흐름을 위해 완전한(Exhaustive) `match` 표현식을 작성한다
4. 간결한 단일 패턴 매칭을 위해 `if let`과 `let else`를 적용한다

---

열거형(Enum)과 패턴 매칭(Pattern Matching)은 Rust의 타입 주도 설계의 핵심입니다. 다른 언어에서 null 포인터, 예외 계층 구조, 타입 태그를 사용하는 곳에서, Rust는 열거형을 사용하여 가능한 모든 상태를 타입 시스템에 인코딩하고 — `match`를 통해 각 경우를 빠짐없이 처리합니다.

## 목차
1. [열거형 정의](#1-열거형-정의)
2. [Option 타입](#2-option-타입)
3. [match를 이용한 패턴 매칭](#3-match를-이용한-패턴-매칭)
4. [if let과 let else](#4-if-let과-let-else)
5. [실제 활용 열거형 패턴](#5-실제-활용-열거형-패턴)
6. [연습 문제](#6-연습-문제)

---

## 1. 열거형 정의

### 1.1 기본 열거형

```rust
#[derive(Debug)]
enum Direction {
    North,
    South,
    East,
    West,
}

fn describe(dir: &Direction) -> &str {
    match dir {
        Direction::North => "heading north",
        Direction::South => "heading south",
        Direction::East => "heading east",
        Direction::West => "heading west",
    }
}

fn main() {
    let dir = Direction::North;
    println!("{}: {}", dir, describe(&dir));
}
```

### 1.2 데이터를 가진 열거형

각 배리언트(Variant)는 서로 다른 타입과 양의 데이터를 가질 수 있습니다:

```rust
#[derive(Debug)]
enum Message {
    Quit,                       // Unit variant (no data)
    Move { x: i32, y: i32 },   // Struct variant (named fields)
    Write(String),              // Tuple variant (one field)
    ChangeColor(u8, u8, u8),   // Tuple variant (multiple fields)
}

impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quitting"),
            Message::Move { x, y } => println!("Moving to ({x}, {y})"),
            Message::Write(text) => println!("Writing: {text}"),
            Message::ChangeColor(r, g, b) => println!("Color: ({r}, {g}, {b})"),
        }
    }
}

fn main() {
    let messages = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("hello")),
        Message::ChangeColor(255, 0, 128),
    ];

    for msg in &messages {
        msg.process();
    }
}
```

### 1.3 열거형 vs 구조체

```
Struct: one shape, many instances
  struct Point { x: f64, y: f64 }  → Every Point has x and y

Enum: many shapes (variants), one type
  enum Shape { Circle(f64), Rect(f64, f64) }  → A Shape is EITHER Circle OR Rect
```

---

## 2. Option 타입

Rust에는 `null`이 없습니다. 대신, `Option<T>` 열거형이 값의 부재 가능성을 인코딩합니다:

```rust
// Defined in the standard library:
// enum Option<T> {
//     Some(T),   // A value is present
//     None,      // No value
// }

fn divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None
    } else {
        Some(a / b)
    }
}

fn find_first_even(numbers: &[i32]) -> Option<i32> {
    for &n in numbers {
        if n % 2 == 0 {
            return Some(n);
        }
    }
    None
}

fn main() {
    // Using match
    match divide(10.0, 3.0) {
        Some(result) => println!("10 / 3 = {result:.2}"),
        None => println!("Cannot divide by zero"),
    }

    // Useful Option methods
    let x: Option<i32> = Some(42);
    let y: Option<i32> = None;

    println!("{}", x.unwrap());              // 42 (panics if None!)
    println!("{}", y.unwrap_or(0));           // 0 (safe default)
    println!("{}", y.unwrap_or_default());    // 0 (uses Default trait)
    println!("{}", x.is_some());             // true
    println!("{}", y.is_none());             // true

    // map — transform the inner value
    let doubled = x.map(|v| v * 2);  // Some(84)

    // and_then — chain operations that return Option (flatmap)
    let result = x
        .map(|v| v as f64)
        .and_then(|v| divide(v, 7.0));  // Some(6.0)
}
```

---

## 3. match를 이용한 패턴 매칭

### 3.1 완전한 매칭(Exhaustive Matching)

```rust
fn describe_number(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1..=9 => "single digit",
        10..=99 => "double digit",
        100..=999 => "triple digit",
        _ => "large number",  // _ is the wildcard, catches everything else
    }
}
```

### 3.2 match에서 구조 분해(Destructuring)

```rust
#[derive(Debug)]
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle { base: f64, height: f64 },
}

fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(radius) => std::f64::consts::PI * radius * radius,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle { base, height } => 0.5 * base * height,
    }
}
```

### 3.3 매치 가드(Match Guard)

```rust
fn classify(n: i32) -> &'static str {
    match n {
        n if n < 0 => "negative",
        0 => "zero",
        n if n % 2 == 0 => "positive even",
        _ => "positive odd",
    }
}
```

### 3.4 @를 이용한 바인딩

```rust
fn check_age(age: u32) {
    match age {
        0 => println!("newborn"),
        a @ 1..=12 => println!("child, age {a}"),
        a @ 13..=17 => println!("teenager, age {a}"),
        a => println!("adult, age {a}"),
    }
}
```

### 3.5 match는 표현식이다

```rust
fn main() {
    let n = 42;
    let description = match n % 3 {
        0 => "divisible by 3",
        1 => "remainder 1",
        2 => "remainder 2",
        _ => unreachable!(),
    };
    println!("{n} is {description}");
}
```

---

## 4. if let과 let else

### 4.1 if let — 단일 패턴

하나의 배리언트에만 관심이 있을 때:

```rust
fn main() {
    let config_max: Option<u32> = Some(3);

    // Verbose match
    match config_max {
        Some(max) => println!("Max: {max}"),
        None => {},
    }

    // Concise if let
    if let Some(max) = config_max {
        println!("Max: {max}");
    }

    // if let with else
    if let Some(max) = config_max {
        println!("Max: {max}");
    } else {
        println!("No max configured");
    }
}
```

### 4.2 let else — 조기 반환(Early Return) 패턴

```rust
fn parse_config(input: &str) -> Option<u32> {
    // let else: bind or diverge (return, break, continue, panic)
    let Some(value) = input.strip_prefix("port=") else {
        return None;
    };

    let Ok(port) = value.parse::<u32>() else {
        return None;
    };

    Some(port)
}

fn main() {
    println!("{:?}", parse_config("port=8080"));  // Some(8080)
    println!("{:?}", parse_config("host=local"));  // None
}
```

### 4.3 while let

```rust
fn main() {
    let mut stack = vec![1, 2, 3];

    // Pop elements until the stack is empty
    while let Some(top) = stack.pop() {
        println!("Popped: {top}");
    }
}
```

---

## 5. 실제 활용 열거형 패턴

### 5.1 상태 머신(State Machine)

```rust
#[derive(Debug)]
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32 },
    Connected { session_id: String },
    Error(String),
}

impl ConnectionState {
    fn next(self) -> Self {
        match self {
            Self::Disconnected => Self::Connecting { attempt: 1 },
            Self::Connecting { attempt } if attempt < 3 => {
                Self::Connecting { attempt: attempt + 1 }
            }
            Self::Connecting { attempt } if attempt >= 3 => {
                Self::Error(format!("Failed after {attempt} attempts"))
            }
            _ => self,
        }
    }
}
```

### 5.2 커맨드 패턴(Command Pattern)

```rust
enum Command {
    Add(String),
    Remove(usize),
    List,
    Quit,
}

fn execute(cmd: &Command, items: &mut Vec<String>) {
    match cmd {
        Command::Add(item) => {
            items.push(item.clone());
            println!("Added: {item}");
        }
        Command::Remove(index) => {
            if *index < items.len() {
                let removed = items.remove(*index);
                println!("Removed: {removed}");
            }
        }
        Command::List => {
            for (i, item) in items.iter().enumerate() {
                println!("{i}: {item}");
            }
        }
        Command::Quit => println!("Goodbye!"),
    }
}
```

---

## 6. 연습 문제

### 연습 1: 신호등
`Red`, `Yellow`, `Green` 배리언트를 가진 `TrafficLight` 열거형을 정의하세요. 각 신호의 지속 시간을 반환하는 `duration(&self) -> u32` 메서드와 다음 신호를 반환하는 `next(&self) -> Self` 메서드를 구현하세요.

### 연습 2: 표현식 평가기
`Num(f64)`, `Add(Box<Expr>, Box<Expr>)`, `Mul(Box<Expr>, Box<Expr>)`을 가진 `Expr` 열거형을 정의하세요. match를 사용하여 `fn eval(expr: &Expr) -> f64`를 구현하세요.

### 연습 3: 안전한 나눗셈
0으로 나누면 `None`을 반환하는 `fn safe_div(a: i32, b: i32) -> Option<i32>`를 작성하세요. `.and_then()`을 사용하여 세 번의 나눗셈을 체이닝하세요.

### 연습 4: 커맨드 파싱
"add milk", "remove 2", "list", "quit"과 같은 문자열을 `Command` 열거형으로 파싱하는 함수를 작성하세요. `Option<Command>`를 반환하세요.

### 연습 5: 중첩 패턴
`Option<Option<i32>>`를 받아 내부 값을 반환하고, `Some(None)`은 0을, `None`은 -1을 반환하는 중첩 패턴 매칭 함수를 작성하세요.

---

## 참고 자료
- [The Rust Book — Enums](https://doc.rust-lang.org/book/ch06-00-enums.html)
- [The Rust Book — match](https://doc.rust-lang.org/book/ch06-02-match.html)
- [Rust by Example — enum](https://doc.rust-lang.org/rust-by-example/custom_types/enum.html)

---

**이전**: [구조체와 메서드(Structs and Methods)](./06_Structs_and_Methods.md) | **다음**: [컬렉션(Collections)](./08_Collections.md)

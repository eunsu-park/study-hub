# 20. 선언적 매크로

**이전**: [빌드 시스템 심층 분석](./03_Build_System.md) | **다음**: [절차적 매크로](./05_Procedural_Macros.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `macro_rules!`를 사용하여 여러 매치 암(arm)이 있는 선언적 매크로 작성하기
2. 반복 연산자(`*`, `+`, `?`)로 가변 길이 입력 매칭하기
3. 프래그먼트 지정자(`expr`, `ty`, `ident`, `tt` 등) 이해하기
4. `cargo expand`와 `trace_macros!`로 매크로 확장 디버깅하기
5. 코드 생성, DSL, 보일러플레이트 감소를 위한 실용적 매크로 패턴 적용하기

---

Rust의 매크로 시스템은 가장 강력한 기능 중 하나입니다. C/C++ 텍스트 치환 매크로와 달리, Rust 매크로는 **추상 구문 트리(AST)**에서 작동합니다 — 위생적(hygienic)이고, 타입을 인식하며, 컴파일 타임에 검사됩니다. 이 레슨은 `macro_rules!`로 정의하는 **선언적 매크로**(매크로 바이 예제라고도 함)를 다룹니다.

## 목차
1. [매크로가 필요한 이유](#1-매크로가-필요한-이유)
2. [macro_rules! 기초](#2-macro_rules-기초)
3. [프래그먼트 지정자](#3-프래그먼트-지정자)
4. [반복](#4-반복)
5. [다중 매치 암](#5-다중-매치-암)
6. [실용적 패턴](#6-실용적-패턴)
7. [매크로 위생성](#7-매크로-위생성)
8. [매크로 디버깅](#8-매크로-디버깅)
9. [스코핑과 내보내기](#9-스코핑과-내보내기)
10. [흔한 함정](#10-흔한-함정)
11. [실제 예제](#11-실제-예제)
12. [연습문제](#12-연습문제)

---

## 1. 매크로가 필요한 이유

함수로는 모든 것을 할 수 없습니다. 다음의 제약 사항들을 고려해 보세요:

```rust
// 가변 개수의 인수를 받는 함수를 작성할 수 없습니다
// println!이 매크로인 이유가 바로 이것입니다
println!("one: {}", 1);
println!("two: {} {}", 1, 2);
println!("three: {} {} {}", 1, 2, 3);

// 구조체 정의를 생성하거나 트레이트를 자동 구현하는
// 함수를 작성할 수 없습니다

// 새로운 식별자를 만드는 함수를 작성할 수 없습니다
```

매크로는 이러한 빈 틈을 채웁니다. 매크로는 **컴파일 타임**에 실행되어, 타입 검사와 빌림 검사가 일어나기 전에 코드를 생성합니다.

### 함수 vs 매크로

| 특성 | 함수 | 매크로 |
|------|------|--------|
| 평가 시점 | 런타임 | 컴파일 타임 |
| 가변 인수 | 불가 (슬라이스 사용) | 가능 |
| 코드 생성 | 불가 | 가능 |
| 위생성 | 해당 없음 | 있음 (스코프) |
| 타입 검사 | 함수 자체에서 | *확장된* 코드에서 |
| 디버깅 | 표준 | 확장 도구 필요 |

---

## 2. macro_rules! 기초

선언적 매크로는 입력 토큰에 대해 패턴을 매칭하고 대체 코드를 생성합니다:

```rust
// 가장 간단한 매크로 — 인수 없음
macro_rules! say_hello {
    () => {
        println!("Hello from a macro!");
    };
}

fn main() {
    say_hello!();  // 확장: println!("Hello from a macro!");
}
```

### 호출 스타일

매크로는 소괄호, 대괄호, 또는 중괄호로 호출할 수 있습니다:

```rust
macro_rules! my_macro {
    () => { 42 };
}

fn main() {
    let a = my_macro!();   // 소괄호 — 표현식 스타일 매크로에 가장 일반적
    let b = my_macro![];   // 대괄호 — vec![] 같은 배열 스타일 매크로에 관례적
    let c = my_macro!{};   // 중괄호 — 아이템 정의 매크로에 관례적
    assert_eq!(a, b);
    assert_eq!(b, c);
}
```

관례: 함수 형태 호출에는 `()`, 리터럴 형태 호출(`vec![]`)에는 `[]`, 아이템 수준 매크로에는 `{}`를 사용합니다.

### 인수 캡처

`$name:specifier`를 사용하여 입력 토큰을 캡처합니다:

```rust
macro_rules! create_greeting {
    ($name:expr) => {
        format!("Hello, {}!", $name)
    };
}

fn main() {
    let greeting = create_greeting!("Rust");
    println!("{greeting}");  // Hello, Rust!

    let user = String::from("Alice");
    let greeting = create_greeting!(user);
    println!("{greeting}");  // Hello, Alice!
}
```

---

## 3. 프래그먼트 지정자

프래그먼트 지정자는 매크로 파서에 어떤 종류의 구문을 기대하는지 알려줍니다:

| 지정자 | 매칭 대상 | 예시 |
|--------|-----------|------|
| `expr` | 임의의 표현식 | `1 + 2`, `foo()`, `if x { 1 } else { 2 }` |
| `ty` | 타입 | `i32`, `Vec<String>`, `&'a str` |
| `ident` | 식별자 | `foo`, `MyStruct`, `x` |
| `pat` | 패턴 | `Some(x)`, `(a, b)`, `_` |
| `path` | 경로 | `std::io::Result`, `crate::module` |
| `stmt` | 문장 | `let x = 1`, `x += 1` |
| `block` | 블록 `{ ... }` | `{ let x = 1; x + 1 }` |
| `item` | 아이템 | `fn foo() {}`, `struct Bar;` |
| `meta` | 속성 내용 | `derive(Debug)`, `cfg(test)` |
| `tt` | 단일 토큰 트리 | 단일 토큰 또는 `(...)` / `[...]` / `{...}` 그룹 |
| `literal` | 리터럴 값 | `42`, `"hello"`, `true` |
| `lifetime` | 라이프타임 | `'a`, `'static` |
| `vis` | 가시성 한정자 | `pub`, `pub(crate)`, (비어있음) |

### 타입 지정자 사용

```rust
macro_rules! declare_pair {
    ($name:ident, $t:ty) => {
        struct $name {
            first: $t,
            second: $t,
        }
    };
}

declare_pair!(IntPair, i32);
declare_pair!(StringPair, String);

fn main() {
    let pair = IntPair { first: 1, second: 2 };
    println!("Pair: ({}, {})", pair.first, pair.second);

    let sp = StringPair {
        first: "hello".into(),
        second: "world".into(),
    };
    println!("Pair: ({}, {})", sp.first, sp.second);
}
```

### `tt` 지정자 (토큰 트리)

`tt`는 가장 유연한 지정자입니다 — 단일 토큰이나 구분자로 균형을 이루는 토큰 그룹을 매칭합니다:

```rust
macro_rules! apply {
    ($func:ident, $($arg:tt)*) => {
        $func($($arg)*)
    };
}

fn add(a: i32, b: i32) -> i32 { a + b }

fn main() {
    let result = apply!(add, 3, 4);
    println!("apply!(add, 3, 4) = {result}");  // 7
}
```

---

## 4. 반복

반복이야말로 매크로를 진정으로 강력하게 만드는 것입니다. 구문은 `$(...) 구분자 반복_연산자`입니다:

- `*` — 0번 이상
- `+` — 1번 이상
- `?` — 0번 또는 1번

```rust
// vec![] 복제 — 요소 목록에서 Vec 생성
macro_rules! my_vec {
    // 쉼표로 구분된 표현식 목록 매칭
    ( $( $element:expr ),* ) => {
        {
            let mut v = Vec::new();
            $( v.push($element); )*
            v
        }
    };
    // 후행 쉼표 처리
    ( $( $element:expr ),+ , ) => {
        my_vec![ $( $element ),* ]
    };
}

fn main() {
    let v = my_vec![1, 2, 3, 4, 5];
    println!("{v:?}");  // [1, 2, 3, 4, 5]

    let v = my_vec!["hello", "world",];  // 후행 쉼표 가능
    println!("{v:?}");  // ["hello", "world"]
}
```

### HashMap 매크로

```rust
// key => value 쌍에서 HashMap 생성
macro_rules! hash_map {
    ( $( $key:expr => $value:expr ),* $(,)? ) => {
        {
            let mut map = std::collections::HashMap::new();
            $( map.insert($key, $value); )*
            map
        }
    };
}

fn main() {
    let scores = hash_map! {
        "Alice" => 95,
        "Bob" => 87,
        "Charlie" => 92,
    };

    for (name, score) in &scores {
        println!("{name}: {score}");
    }
}
```

---

## 5. 다중 매치 암

`match` 표현식과 마찬가지로 매크로도 여러 암을 가질 수 있습니다. 매크로는 각 암을 순서대로 시도합니다:

```rust
macro_rules! calculate {
    // 단일 값 — 항등
    ($x:expr) => { $x };

    // 연산자를 가진 두 값
    ($x:expr, +, $y:expr) => { $x + $y };
    ($x:expr, -, $y:expr) => { $x - $y };
    ($x:expr, *, $y:expr) => { $x * $y };
    ($x:expr, /, $y:expr) => { $x / $y };
}

fn main() {
    println!("{}", calculate!(5));           // 5
    println!("{}", calculate!(10, +, 20));   // 30
    println!("{}", calculate!(100, /, 4));   // 25
}
```

### 형태별 오버로딩

```rust
macro_rules! log_message {
    // 인수 없음 — 구분선만
    () => {
        println!("---");
    };

    // 메시지만
    ($msg:expr) => {
        println!("[LOG] {}", $msg);
    };

    // 레벨과 메시지
    ($level:ident, $msg:expr) => {
        println!("[{:>5}] {}", stringify!($level).to_uppercase(), $msg);
    };

    // 레벨, 메시지, 키-값 컨텍스트
    ($level:ident, $msg:expr, $( $key:ident = $val:expr ),+ ) => {
        print!("[{:>5}] {}", stringify!($level).to_uppercase(), $msg);
        $( print!(" {}={}", stringify!($key), $val); )+
        println!();
    };
}

fn main() {
    log_message!();
    log_message!("Application started");
    log_message!(info, "Request received");
    log_message!(error, "Connection failed", host = "db.example.com", retries = 3);
}
```

---

## 6. 실용적 패턴

### 패턴 1: 구조체와 Display 자동 생성

```rust
macro_rules! make_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $field_vis:vis $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $( $field_vis $field: $ty, )*
        }

        impl $name {
            $vis fn new( $( $field: $ty ),* ) -> Self {
                Self { $( $field, )* }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}(", stringify!($name))?;
                let fields = vec![
                    $( format!("{}: {:?}", stringify!($field), self.$field), )*
                ];
                write!(f, "{})", fields.join(", "))
            }
        }
    };
}

make_struct! {
    #[derive(Debug, Clone)]
    pub struct Config {
        pub host: String,
        pub port: u16,
        pub debug: bool,
    }
}

fn main() {
    let config = Config::new("localhost".into(), 8080, true);
    println!("{config}");  // Config(host: "localhost", port: 8080, debug: true)
}
```

### 패턴 2: 문자열 변환이 있는 열거형

```rust
macro_rules! enum_with_str {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $variant:ident => $str:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $( $variant, )*
        }

        impl $name {
            pub fn as_str(&self) -> &'static str {
                match self {
                    $( $name::$variant => $str, )*
                }
            }

            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $( $str => Some($name::$variant), )*
                    _ => None,
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.as_str())
            }
        }
    };
}

enum_with_str! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Color {
        Red => "red",
        Green => "green",
        Blue => "blue",
        Yellow => "yellow",
    }
}

fn main() {
    let c = Color::Red;
    println!("Color: {c}");                    // red
    println!("From str: {:?}", Color::from_str("blue"));  // Some(Blue)
}
```

### 패턴 3: 테스트 생성

```rust
macro_rules! test_cases {
    ( $func:ident : $( ($input:expr) => $expected:expr );+ $(;)? ) => {
        $(
            paste::paste! {
                #[test]
                fn [< test_ $func _ $input >]() {
                    assert_eq!($func($input), $expected);
                }
            }
        )+
    };
}

// paste 크레이트 없이 사용하는 더 간단한 버전:
macro_rules! test_suite {
    ($name:ident, $func:expr, $( ($input:expr, $expected:expr) ),+ $(,)? ) => {
        #[cfg(test)]
        mod $name {
            use super::*;

            $(
                #[test]
                fn test() {
                    assert_eq!($func($input), $expected,
                        "Failed for input: {:?}", $input);
                }
            )+
        }
    };
}

fn double(n: i32) -> i32 { n * 2 }

fn is_even(n: i32) -> bool { n % 2 == 0 }

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_doubles {
        ( $( $input:expr => $expected:expr ),* $(,)? ) => {
            $(
                assert_eq!(double($input), $expected,
                    "double({}) should be {}", $input, $expected);
            )*
        };
    }

    #[test]
    fn test_double() {
        assert_doubles! {
            0 => 0,
            1 => 2,
            5 => 10,
            -3 => -6,
        }
    }
}
```

---

## 7. 매크로 위생성

Rust 매크로는 **위생적(hygienic)**입니다 — 매크로 내부에서 정의된 변수가 호출자의 스코프로 누출되지 않고, 호출자의 변수가 매크로 내부를 실수로 가리지(shadow) 않습니다:

```rust
macro_rules! using_x {
    ($body:expr) => {
        {
            let x = 42;  // 매크로 스코프의 'x'
            $body         // 호출자의 'x'를 사용 (매크로의 것이 아님)
        }
    };
}

fn main() {
    let x = 10;
    let result = using_x!(x + 1);
    println!("result: {result}");  // 11 (43이 아님)
}
```

### 위생성 깨기 (필요한 경우)

때로는 의도적으로 매크로가 호출자에게 보이는 바인딩을 도입하기를 원할 수 있습니다. 관용적인 접근 방식은 호출자가 변수 이름을 직접 지정하게 하는 것입니다:

```rust
macro_rules! let_binding {
    ($name:ident = $value:expr) => {
        let $name = $value;
    };
}

fn main() {
    let_binding!(x = 42);
    println!("x = {x}");  // 42 — 호출자가 이름을 선택했으므로 동작
}
```

---

## 8. 매크로 디버깅

### cargo expand

매크로 디버깅에 가장 유용한 도구입니다:

```bash
cargo install cargo-expand

# 크레이트의 모든 매크로 확장
cargo expand

# 특정 모듈의 매크로 확장
cargo expand module_name
```

### trace_macros! (나이틀리 전용)

```rust
#![feature(trace_macros)]

macro_rules! my_add {
    ($a:expr, $b:expr) => { $a + $b };
}

fn main() {
    trace_macros!(true);
    let x = my_add!(1, 2);
    trace_macros!(false);
    println!("{x}");
}
```

컴파일 중 출력:
```
note: trace_macro
  --> src/main.rs:8:13
   |
8  |     let x = my_add!(1, 2);
   |             ^^^^^^^^^^^^^^
   |
   = note: expanding `my_add! { 1, 2 }`
   = note: to `1 + 2`
```

### stringify!로 검사

```rust
macro_rules! debug_expand {
    ($($tokens:tt)*) => {
        println!("Input tokens: {}", stringify!($($tokens)*));
        $($tokens)*
    };
}

fn main() {
    debug_expand! {
        let x = 1 + 2;
        println!("x = {x}");
    }
}
```

### 컴파일 에러 메시지

`compile_error!`를 사용하여 매크로에서 명확한 에러를 생성합니다:

```rust
macro_rules! validated_enum {
    ( $name:ident { $( $variant:ident ),+ $(,)? } ) => {
        enum $name { $( $variant, )+ }
    };
    ( $name:ident { } ) => {
        compile_error!("Enum must have at least one variant");
    };
}
```

---

## 9. 스코핑과 내보내기

### #[macro_export]

`#[macro_export]`는 매크로를 크레이트 루트에서 사용 가능하게 만듭니다:

```rust
#[macro_export]
macro_rules! my_assert {
    ($cond:expr) => {
        if !$cond {
            panic!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("Assertion failed: {} — {}", stringify!($cond), $msg);
        }
    };
}
```

---

## 10. 흔한 함정

### 함정 1: 연산자 우선순위

```rust
macro_rules! double {
    ($x:expr) => { $x * 2 };
}

fn main() {
    println!("{}", double!(3 + 1));  // 8 (7이 아님)
    // $x:expr가 전체 표현식 "3 + 1"을 캡처하므로 (3 + 1) * 2 = 8
    // C 매크로와 달리 Rust의 위생성이 이를 올바르게 처리합니다!
}
```

### 함정 2: 재귀 확장 한도

```rust
macro_rules! count {
    () => { 0usize };
    ($head:tt $($tail:tt)*) => { 1usize + count!($($tail)*) };
}

fn main() {
    let n = count!(a b c d e);
    println!("Count: {n}");  // 5
}

// 기본 재귀 한도는 128입니다. 깊은 재귀 매크로의 경우:
// #![recursion_limit = "256"]
```

---

## 11. 실제 예제

### 설정 DSL

```rust
use std::collections::HashMap;

macro_rules! config {
    (
        $( section [$section:ident] {
            $( $key:ident : $value:expr ),* $(,)?
        } )*
    ) => {
        {
            let mut sections: HashMap<&str, HashMap<&str, String>> = HashMap::new();
            $(
                let mut section_map = HashMap::new();
                $( section_map.insert(stringify!($key), format!("{}", $value)); )*
                sections.insert(stringify!($section), section_map);
            )*
            sections
        }
    };
}

fn main() {
    let cfg = config! {
        section [database] {
            host: "localhost",
            port: 5432,
            name: "mydb",
        }
        section [server] {
            host: "0.0.0.0",
            port: 8080,
            workers: 4,
        }
    };

    for (section, values) in &cfg {
        println!("[{section}]");
        for (key, value) in values {
            println!("  {key} = {value}");
        }
    }
}
```

---

## 12. 연습문제

1. **해시 셋 매크로**: `vec![]`과 유사하게 쉼표로 구분된 요소 목록에서 `HashSet`을 만드는 `hash_set!`을 작성하세요.

2. **가변 인수 min/max**: 2개 이상의 인수를 받아 최솟값을 반환하는 `min!` 매크로를 작성하세요. 예: `min!(5, 3, 8, 1)`은 `1`을 반환합니다.

3. **Display가 있는 구조체**: 구조체 정의를 생성하고 각 필드 이름과 값을 출력하는 `Display`를 자동으로 구현하는 매크로를 작성하세요.

4. **백오프가 있는 재시도**: 지수 백오프(매 시도마다 지연 시간 두 배)를 지원하도록 `retry!` 매크로를 확장하세요.

5. **JSON 스타일 DSL**: JSON 스타일 구문에서 중첩 구조를 만드는 `json!` 매크로를 작성하세요.

---

## 참고 자료

- [The Rust Reference: Macros by Example](https://doc.rust-lang.org/reference/macros-by-example.html)
- [The Little Book of Rust Macros](https://veykril.github.io/tlborm/)
- [Rust by Example: Macros](https://doc.rust-lang.org/rust-by-example/macros.html)
- [cargo-expand](https://github.com/dtolnay/cargo-expand)

---

**이전**: [빌드 시스템 심층 분석](./03_Build_System.md) | **다음**: [절차적 매크로](./05_Procedural_Macros.md)

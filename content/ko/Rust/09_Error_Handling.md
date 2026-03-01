# 09. 에러 처리(Error Handling)

**이전**: [컬렉션](./08_Collections.md) | **다음**: [트레이트와 제네릭](./10_Traits_and_Generics.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 복구 가능한 에러(`Result`)와 복구 불가능한 에러(`panic!`)를 구분하고 적절한 것을 선택할 수 있다
2. `?` 연산자를 사용하여 에러를 간결하게 전파하고, 내부적으로 어떻게 동작하는지 설명할 수 있다
3. `std::error::Error`를 구현하는 커스텀 에러 타입을 정의할 수 있다
4. 라이브러리와 애플리케이션에서 인체공학적인 에러 처리를 위해 `thiserror`와 `anyhow` 크레이트를 사용할 수 있다
5. `From` 트레이트를 사용하여 에러를 변환할 수 있다

---

에러 처리는 Rust가 기존에 알던 언어들과 가장 크게 다른 부분입니다. 예외(exception)도, try/catch 블록도, null 포인터 서프라이즈도 없습니다. 대신 Rust는 실패 가능성을 타입 시스템에 직접 인코딩합니다 — 함수가 실패할 수 있다면, 반환 타입이 그 사실을 알려줍니다. 이렇게 하면 에러를 실수로 무시하는 일이 불가능해지고, 실패 경로를 미리 생각하도록 강제합니다.

레스토랑 주방에 비유하자면: 화재가 번지도록 두고 누군가 알아채길 바라는 것(예외) 대신, Rust는 모든 스테이션에 소화기를 건네주고 갖고 있다는 사실을 인정하도록 요구합니다(Result).

## 목차
1. [복구 불가능한 에러: panic!](#1-복구-불가능한-에러-panic)
2. [Result<T, E> — 복구 가능한 에러](#2-resultt-e--복구-가능한-에러)
3. [? 연산자](#3--연산자)
4. [커스텀 에러 타입](#4-커스텀-에러-타입)
5. [thiserror — 에러 타입 파생](#5-thiserror--에러-타입-파생)
6. [anyhow — 애플리케이션 수준 에러](#6-anyhow--애플리케이션-수준-에러)
7. [panic vs Result 선택 기준](#7-panic-vs-result-선택-기준)
8. [From을 이용한 에러 변환](#8-from을-이용한-에러-변환)
9. [연습 문제](#9-연습-문제)

---

## 1. 복구 불가능한 에러: panic!

프로그램이 합리적으로 계속 진행할 수 없을 만큼 심각한 문제가 발생하면 `panic!`을 사용합니다. 이는 즉시 실행을 중단하고, (기본값으로) 스택을 되감으며(unwind), 에러 메시지를 출력합니다.

```rust
fn main() {
    // Explicit panic
    // panic!("something went terribly wrong");

    // Implicit panics from the standard library
    let v = vec![1, 2, 3];
    // v[99]; // panics: index out of bounds

    // Panics are for bugs, not for expected failures
    // Examples: violated invariants, unreachable code, failed assertions
    let x = 5;
    assert!(x > 0, "x must be positive, got {}", x);
    assert_eq!(2 + 2, 4);
}
```

### 1.1 스택 되감기(Unwinding) vs 즉시 종료(Abort)

패닉이 발생하면 Rust는 두 가지 전략 중 하나를 사용합니다:

```
Unwinding (default):                  Abort:
┌──────────────────────┐              ┌──────────────────────┐
│ Walks back up the    │              │ Immediately kills    │
│ call stack, running  │              │ the process.         │
│ Drop for each frame. │              │ No cleanup, no Drop. │
│                      │              │                      │
│ Pro: memory cleaned  │              │ Pro: smaller binary  │
│ Con: larger binary   │              │ Con: no cleanup      │
│      slower panic    │              │      OS reclaims mem │
└──────────────────────┘              └──────────────────────┘
```

즉시 종료 모드를 사용하려면 `Cargo.toml`에 다음을 추가하세요:
```toml
[profile.release]
panic = "abort"
```

### 1.2 백트레이스(Backtrace)

패닉이 발생한 위치를 확인하려면 `RUST_BACKTRACE=1`을 설정하세요:

```bash
$ RUST_BACKTRACE=1 cargo run
thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 99'
stack backtrace:
   0: std::panicking::begin_panic_handler
   ...
   8: playground::main
             at ./src/main.rs:3:5
```

---

## 2. Result<T, E> — 복구 가능한 에러

파일 없음, 잘못된 입력, 네트워크 타임아웃처럼 **발생할 것으로 예상**되는 에러에는 Rust가 `Result<T, E>`를 사용합니다:

```rust
enum Result<T, E> {
    Ok(T),   // Success, contains the value
    Err(E),  // Failure, contains the error
}
```

```rust
use std::fs;

fn main() {
    // fs::read_to_string returns Result<String, io::Error>
    let result = fs::read_to_string("hello.txt");

    match result {
        Ok(contents) => println!("File contents:\n{}", contents),
        Err(error) => println!("Failed to read file: {}", error),
    }
}
```

### 2.1 Result 다루기

```rust
use std::num::ParseIntError;

fn parse_age(input: &str) -> Result<u32, ParseIntError> {
    input.trim().parse::<u32>()
}

fn main() {
    // Pattern matching — the most explicit approach
    match parse_age("25") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Parse error: {}", e),
    }

    // unwrap() — panics on Err (use only in prototypes or tests)
    let age = parse_age("30").unwrap();
    println!("age = {}", age);

    // expect() — panics with a custom message (better than unwrap)
    let age = parse_age("30").expect("failed to parse age");
    println!("age = {}", age);

    // unwrap_or() — provide a fallback value
    let age = parse_age("not_a_number").unwrap_or(0);
    println!("age with fallback = {}", age);

    // unwrap_or_else() — compute fallback lazily
    let age = parse_age("not_a_number").unwrap_or_else(|_| {
        println!("Using default age");
        18
    });
    println!("age = {}", age);
}
```

### 2.2 매핑과 체이닝

```rust
fn double_parse(input: &str) -> Result<i32, std::num::ParseIntError> {
    input
        .trim()
        .parse::<i32>()
        .map(|n| n * 2) // transform the Ok value
}

fn main() {
    println!("{:?}", double_parse("21")); // Ok(42)
    println!("{:?}", double_parse("xx")); // Err(ParseIntError)

    // map_err transforms the Err variant
    let result: Result<i32, String> = "abc"
        .parse::<i32>()
        .map_err(|e| format!("Custom error: {}", e));
    println!("{:?}", result);

    // and_then chains operations that each return Result
    let result = "42"
        .parse::<i32>()
        .and_then(|n| {
            if n > 0 {
                Ok(n)
            } else {
                Err("must be positive".parse::<i32>().unwrap_err())
            }
        });
    println!("{:?}", result);
}
```

---

## 3. ? 연산자

`?` 연산자는 깔끔한 에러 처리를 위한 Rust의 비밀 무기입니다. 장황한 `match` 보일러플레이트를 단 한 글자로 대체합니다:

```rust
use std::fs;
use std::io;

// WITHOUT ? — verbose and deeply nested
fn read_username_verbose() -> Result<String, io::Error> {
    let result = fs::read_to_string("username.txt");
    match result {
        Ok(contents) => Ok(contents.trim().to_string()),
        Err(e) => Err(e),
    }
}

// WITH ? — concise and readable
fn read_username() -> Result<String, io::Error> {
    let contents = fs::read_to_string("username.txt")?;
    // If read_to_string returns Err, the function returns that Err immediately
    // If it returns Ok, the inner value is extracted and assigned to contents
    Ok(contents.trim().to_string())
}

fn main() {
    match read_username() {
        Ok(name) => println!("Username: {}", name),
        Err(e) => println!("Error: {}", e),
    }
}
```

### 3.1 ? 연산자의 내부 동작

`?` 연산자는 대략 다음과 같이 동작합니다:

```
expression?

// desugars to approximately:

match expression {
    Ok(val)  => val,
    Err(err) => return Err(From::from(err)),
                       ^^^^^^^^^^^^^^^^^^^^
                       Note: calls From::from() to convert
                       the error type if needed!
}
```

`?`가 하는 일은 **두 가지**입니다:
1. `Err`시 조기 반환(early return)
2. `From`을 사용한 에러 타입 변환

### 3.2 여러 ? 호출 연결하기

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_first_line(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;       // might fail
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;    // might fail
    let first_line = contents
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    Ok(first_line)
}

// Even more concise with method chaining
fn read_first_line_short(path: &str) -> Result<String, io::Error> {
    let contents = std::fs::read_to_string(path)?;
    Ok(contents.lines().next().unwrap_or("").to_string())
}

fn main() {
    match read_first_line("/etc/hostname") {
        Ok(line) => println!("First line: {}", line),
        Err(e) => println!("Error: {}", e),
    }
}
```

### 3.3 main()에서 ? 사용하기

```rust
use std::fs;
use std::io;

// main() can return Result — errors are printed automatically
fn main() -> Result<(), io::Error> {
    let contents = fs::read_to_string("config.txt")?;
    println!("Config: {}", contents);
    Ok(())
}
```

---

## 4. 커스텀 에러 타입

실제 애플리케이션은 여러 가지 실패 모드를 가집니다. 커스텀 에러 타입은 호출자가 각 경우를 구조적으로 처리할 수 있게 해줍니다:

```rust
use std::fmt;
use std::num::ParseIntError;
use std::io;

#[derive(Debug)]
enum AppError {
    Io(io::Error),
    Parse(ParseIntError),
    Validation(String),
}

// Display: human-readable error messages
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {}", e),
            AppError::Parse(e) => write!(f, "Parse error: {}", e),
            AppError::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

// Error trait: connects to Rust's error ecosystem
impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // source() returns the underlying cause, enabling error chains
        match self {
            AppError::Io(e) => Some(e),
            AppError::Parse(e) => Some(e),
            AppError::Validation(_) => None,
        }
    }
}

// From impls enable automatic conversion with ?
impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::Parse(e)
    }
}

fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = std::fs::read_to_string(path)?; // io::Error → AppError via From
    let port: u32 = contents.trim().parse()?;       // ParseIntError → AppError via From
    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is reserved (must be >= 1024)", port),
        ));
    }
    Ok(port)
}

fn main() {
    match load_config("port.txt") {
        Ok(port) => println!("Listening on port {}", port),
        Err(AppError::Io(e)) => eprintln!("Could not read config: {}", e),
        Err(AppError::Parse(e)) => eprintln!("Invalid port number: {}", e),
        Err(AppError::Validation(msg)) => eprintln!("{}", msg),
    }
}
```

이 방법은 동작하지만 보일러플레이트가 번거롭습니다. 이때 `thiserror`가 등장합니다.

---

## 5. thiserror — 에러 타입 파생

`thiserror` 크레이트는 파생 매크로를 통해 `Display`, `Error`, `From` 구현을 자동으로 생성합니다. **라이브러리** 에러 타입에 표준적으로 사용됩니다:

```toml
# Cargo.toml
[dependencies]
thiserror = "2"
```

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    // #[error(...)] generates the Display impl
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),  // #[from] generates From<io::Error>

    #[error("Parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),

    #[error("Validation: {0}")]
    Validation(String),
}

// That's it! ~10 lines replaces ~40 lines of manual impls.

fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = std::fs::read_to_string(path)?;
    let port: u32 = contents.trim().parse()?;
    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is reserved", port),
        ));
    }
    Ok(port)
}

fn main() {
    if let Err(e) = load_config("port.txt") {
        eprintln!("Error: {}", e);
        // Can inspect the source chain
        if let Some(source) = std::error::Error::source(&e) {
            eprintln!("  caused by: {}", source);
        }
    }
}
```

---

## 6. anyhow — 애플리케이션 수준 에러

`thiserror`가 라이브러리(구조화된, 타입이 있는 에러)를 위한 것이라면, `anyhow`는 어떤 에러든 컨텍스트와 함께 전파하고 싶은 **애플리케이션** 용입니다:

```toml
# Cargo.toml
[dependencies]
anyhow = "1"
```

```rust
use anyhow::{Context, Result};

// anyhow::Result<T> is an alias for Result<T, anyhow::Error>
// anyhow::Error can hold any error type that implements std::error::Error

fn load_config(path: &str) -> Result<u32> {
    let contents = std::fs::read_to_string(path)
        .context(format!("Failed to read config from '{}'", path))?;
        //       ^^^^^^^ adds human-readable context to the error chain

    let port: u32 = contents
        .trim()
        .parse()
        .context("Config file does not contain a valid port number")?;

    anyhow::ensure!(port >= 1024, "Port {} is reserved (must be >= 1024)", port);
    //      ^^^^^^^ like assert!, but returns Err instead of panicking

    Ok(port)
}

fn main() -> Result<()> {
    let port = load_config("port.txt")?;
    println!("Server running on port {}", port);
    Ok(())
    // If an error occurs, anyhow prints the full error chain:
    //   Error: Failed to read config from 'port.txt'
    //
    //   Caused by:
    //       No such file or directory (os error 2)
}
```

### thiserror vs anyhow — 언제 무엇을 사용할까

```
                 thiserror                    anyhow
              ┌─────────────────┐         ┌─────────────────┐
  Use for:    │ Libraries       │         │ Applications    │
              │ Public APIs     │         │ CLI tools       │
              │                 │         │ Scripts         │
              ├─────────────────┤         ├─────────────────┤
  Errors:     │ Enum variants   │         │ Opaque, boxed   │
              │ Typed, matchable│         │ Context strings  │
              ├─────────────────┤         ├─────────────────┤
  Callers:    │ Can match on    │         │ Just print or   │
              │ specific errors │         │ propagate       │
              └─────────────────┘         └─────────────────┘

  Rule of thumb: if your code is consumed by others → thiserror
                  if your code is the final binary  → anyhow
```

---

## 7. panic vs Result 선택 기준

| 상황 | 사용 | 이유 |
|------|------|------|
| 배열 인덱스 초과 (버그) | `panic!` | 프로그래밍 오류 — 발생해서는 안 됨 |
| 파일을 찾을 수 없음 | `Result` | 사용자가 해결 가능한 예상된 실패 |
| 잘못된 사용자 입력 | `Result` | 일반적인 런타임 조건 |
| 불변 조건 위반 (예: 음수 배열 크기) | `panic!` | 호출 코드의 버그 |
| 네트워크 타임아웃 | `Result` | 예상됨, 재시도 가능 |
| 프로토타입 / 예제 코드 | `unwrap()` | 빠른 실험에서 허용 가능 |
| 테스트 | `unwrap()` / `?` | 테스트 실패는 패닉이어야 함 |

**가이드라인:**
1. 실패 가능한 함수에는 기본적으로 `Result`를 사용하세요
2. `panic!`은 진정으로 복구 불가능한 상황이나 프로그래밍 버그에만 사용하세요
3. 라이브러리 코드에서는 거의 `panic!`을 사용하지 마세요 — 호출자가 결정하도록 하세요
4. 프로덕션 코드에서 `unwrap()`은 코드 스멜(code smell)입니다 — 최소한 `expect()`를 사용하세요

---

## 8. From을 이용한 에러 변환

`From` 트레이트는 `?`가 여러 에러 타입에 걸쳐 동작하게 해주는 접착제입니다. `Result<T, MyError>`를 반환하는 함수 안에서 `result?`를 작성하면, Rust는 자동으로 `MyError::from(original_error)`를 호출합니다:

```rust
use std::num::ParseIntError;

#[derive(Debug)]
struct AgeError {
    message: String,
}

// This From impl lets ? convert ParseIntError → AgeError
impl From<ParseIntError> for AgeError {
    fn from(e: ParseIntError) -> Self {
        AgeError {
            message: format!("Invalid age: {}", e),
        }
    }
}

fn parse_age(input: &str) -> Result<u32, AgeError> {
    let age: u32 = input.trim().parse()?;
    // Without the From impl above, this ? would fail to compile
    // because parse() returns ParseIntError, not AgeError

    if age > 150 {
        return Err(AgeError {
            message: format!("Age {} is unrealistic", age),
        });
    }
    Ok(age)
}

fn main() {
    match parse_age("abc") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Error: {}", e.message),
    }

    match parse_age("200") {
        Ok(age) => println!("Age: {}", age),
        Err(e) => println!("Error: {}", e.message),
    }
}
```

### 8.1 에러 변환 흐름

```
Your function returns: Result<T, YourError>

Code inside the function:
    let val = some_call()?;
              ^^^^^^^^^^^
              Returns Result<V, OtherError>

Compiler checks:
    Is there an impl From<OtherError> for YourError?
    ├── Yes → calls YourError::from(other_error), returns Err(converted)
    └── No  → compile error!

This is why thiserror's #[from] attribute is so useful:
    #[error("...")]
    Io(#[from] std::io::Error)
    // generates: impl From<io::Error> for YourError
```

---

## 9. 연습 문제

### 문제 1: 안전한 나눗셈(Safe Division)
`safe_divide(a: f64, b: f64) -> Result<f64, String>` 함수를 작성하세요. 0으로 나눌 때 `inf`나 `NaN`을 반환하는 대신 에러를 반환합니다. 그런 다음 `?`를 사용하여 여러 나눗셈을 연결하는 `calculate` 함수를 작성하세요.

### 문제 2: 설정 파서(Config Parser)
`parse_config(input: &str) -> Result<Config, ConfigError>` 함수를 작성하세요:
- `Config`는 `host: String`과 `port: u16` 필드를 가집니다
- 입력 형식은 `host:port`입니다 (예: `"localhost:8080"`)
- `ConfigError`는 `MissingPort`, `InvalidPort(ParseIntError)`, `EmptyHost` 변형을 가집니다
- `ConfigError`에 대해 `Display`와 `Error`를 수동으로 구현하세요 (`thiserror` 없이)

### 문제 3: 파일 처리 파이프라인(File Processing Pipeline)
숫자가 한 줄씩 적힌 파일을 읽고, 각 줄을 `f64`로 파싱하여 평균을 계산하고 반환하는 함수를 작성하세요. `thiserror`를 사용하여 I/O 에러와 파싱 에러를 모두 표현할 수 있는 커스텀 에러 타입을 만드세요. 에러에는 파싱이 실패한 줄 번호가 포함되어야 합니다.

### 문제 4: 에러 컨텍스트 체인(Error Context Chain)
`anyhow`를 사용하여 `process_user_data(path: &str) -> anyhow::Result<UserSummary>` 함수를 작성하세요:
1. JSON 유사 파일 읽기 (간단한 형식으로 시뮬레이션 가능)
2. 사용자 이름과 나이 파싱
3. 데이터 검증 (나이는 0-150이어야 함)
4. 각 단계마다 `.context(...)`를 추가하여 최종 에러 메시지에 전체 체인이 표시되도록 하기

### 문제 5: 백오프를 이용한 재시도(Retry with Backoff)
제네릭 재시도 함수를 작성하세요:
```rust
fn retry<T, E, F>(max_attempts: u32, mut operation: F) -> Result<T, E>
where F: FnMut() -> Result<T, E>
```
`operation()`을 최대 `max_attempts`번 호출하고, 첫 번째 `Ok`나 마지막 `Err`를 반환합니다. 무작위로 실패하는 함수로 테스트하세요.

---

## 참고 자료

- [The Rust Programming Language, Ch. 9: Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [Rust by Example: Error Handling](https://doc.rust-lang.org/rust-by-example/error.html)
- [thiserror crate documentation](https://docs.rs/thiserror)
- [anyhow crate documentation](https://docs.rs/anyhow)
- [Error Handling in a Correctness-Critical Rust Project (BurntSushi)](https://blog.burntsushi.net/rust-error-handling/)

---

**이전**: [컬렉션](./08_Collections.md) | **다음**: [트레이트와 제네릭](./10_Traits_and_Generics.md)

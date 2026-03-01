# 11. 라이프타임(Lifetimes)

**이전**: [트레이트와 제네릭](./10_Traits_and_Generics.md) | **다음**: [클로저와 이터레이터](./12_Closures_and_Iterators.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 라이프타임(Lifetime)이 무엇이며 댕글링 참조(dangling reference)를 방지하기 위해 Rust가 왜 라이프타임을 필요로 하는지 설명할 수 있다
2. 함수 시그니처와 구조체 정의에 라이프타임 파라미터를 표기할 수 있다
3. 세 가지 라이프타임 생략(elision) 규칙을 적용하여 명시적 표기가 불필요한 경우를 판별할 수 있다
4. 문자열 리터럴과 소유된 데이터에 `'static` 라이프타임을 올바르게 사용할 수 있다
5. 일반적인 라이프타임 패턴을 파악하고 빌림 검사기(borrow checker)와 싸우는 안티패턴을 피할 수 있다

---

라이프타임은 **참조가 얼마나 오래 유효한가?** 라는 근본적인 질문에 대한 Rust의 답입니다. C에서는 지역 변수에 대한 포인터를 반환하면 런타임에 충돌이 발생합니다. 가비지 컬렉션 언어에서는 런타임이 참조가 남아있는 동안 모든 것을 살려둡니다. Rust는 세 번째 길을 선택합니다: 컴파일러가 모든 참조가 사용 기간보다 오래 살아남는지를 정적으로 검증하며, 런타임 비용은 제로입니다.

소유권이 "이 데이터를 누가 소유하는가?"에 관한 것이라면, 라이프타임은 "이 데이터에 대한 빌린 뷰가 얼마나 오래 지속되는가?"에 관한 것입니다. 도서관 책에 비유하자면: 소유권은 책을 산 사람이고, 라이프타임은 대출 카드의 반납 기한입니다. 컴파일러는 이미 연체된 책을 대출해주지 않는 사서 역할을 합니다.

라이프타임은 Rust에서 개념적으로 가장 도전적인 부분입니다. 천천히 이해하세요 — 한번 이해되면 빌림 검사기가 적에서 동반자로 바뀝니다.

## 목차
1. [라이프타임이 존재하는 이유](#1-라이프타임이-존재하는-이유)
2. [라이프타임 표기 문법](#2-라이프타임-표기-문법)
3. [함수 시그니처에서의 라이프타임](#3-함수-시그니처에서의-라이프타임)
4. [전형적인 예제: longest()](#4-전형적인-예제-longest)
5. [라이프타임 생략 규칙](#5-라이프타임-생략-규칙)
6. [구조체 정의에서의 라이프타임](#6-구조체-정의에서의-라이프타임)
7. ['static 라이프타임](#7-static-라이프타임)
8. [제네릭 타입의 라이프타임 바운드](#8-제네릭-타입의-라이프타임-바운드)
9. [일반 패턴과 안티패턴](#9-일반-패턴과-안티패턴)
10. [연습 문제](#10-연습-문제)

---

## 1. 라이프타임이 존재하는 이유

라이프타임은 **댕글링 참조(dangling reference)** — 해제된 메모리를 가리키는 포인터 — 를 방지합니다:

```rust
fn main() {
    let r;                // declare r (no value yet)
    {
        let x = 5;
        r = &x;          // borrow x
    }                     // x is dropped here
    // println!("{}", r); // ERROR: r is a dangling reference to dropped x
}
```

컴파일러는 `x`의 라이프타임이 `r`의 라이프타임보다 짧기 때문에 이 오류를 잡아냅니다:

```
Scope diagram:

    let r;              ─────────────────────────── r lives here
    {                       │
        let x = 5;         ├── x lives here
        r = &x;            │   r borrows x
    }                       │   x dropped ← PROBLEM: r still exists
                        ────┘
    // r would be dangling
```

라이프타임 검사가 없다면 이 코드는 컴파일되어 정의되지 않은 동작(undefined behavior)을 일으킵니다(C에서처럼). Rust는 이것이 컴파일되는 것을 거부합니다.

---

## 2. 라이프타임 표기 문법

라이프타임 표기는 어떤 값이 얼마나 오래 살아있는지를 **변경하지 않습니다**. 컴파일러가 안전성을 검증할 수 있도록 참조들 사이의 **관계**를 설명합니다:

```rust
&i32        // a reference (implicit lifetime)
&'a i32     // a reference with explicit lifetime 'a
&'a mut i32 // a mutable reference with explicit lifetime 'a
```

`'a`("틱 에이"라고 읽음)는 **라이프타임 파라미터** — 스코프(scope)에 붙이는 이름입니다. 관례적으로 라이프타임은 짧은 소문자 이름을 사용합니다: `'a`, `'b`, `'c`.

**핵심 통찰:** 라이프타임을 생성하는 것이 아닙니다. 이미 존재하는 스코프에 이름표를 붙여서 컴파일러가 관계가 올바른지 확인할 수 있게 할 뿐입니다. 이미 존재하는 스코프에 이름표를 붙이는 것과 같습니다.

---

## 3. 함수 시그니처에서의 라이프타임

함수가 참조를 받아 참조를 반환할 때, 컴파일러는 출력의 라이프타임이 입력과 어떻게 연관되는지 알아야 합니다:

```rust
// This won't compile — the compiler doesn't know which input's
// lifetime the return value is tied to
// fn first_word(s: &str) -> &str { ... }
// Actually, this DOES compile due to elision rules (covered in Section 5).
// But let's see a case where elision doesn't help:

// Two input references — which one does the return value borrow from?
// fn pick(a: &str, b: &str) -> &str { ... }
// ERROR: missing lifetime specifier

// Solution: annotate to express the relationship
fn pick<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() >= b.len() { a } else { b }
}

fn main() {
    let s1 = String::from("long string");
    let result;
    {
        let s2 = String::from("short");
        result = pick(&s1, &s2);
        println!("result: {}", result); // OK: both s1 and s2 are alive
    }
    // println!("{}", result); // ERROR if uncommented:
    // s2 was dropped, and result might reference s2
}
```

여기서 `'a` 표기의 의미: "반환된 참조는 **두** 입력 참조가 모두 유효한 동안만 유효합니다." 컴파일러는 이것을 사용하여 어느 한 입력이 드롭(drop)된 후에는 결과를 사용할 수 없도록 보장합니다.

---

## 4. 전형적인 예제: longest()

Rust Book의 표준적인 라이프타임 교육 예제입니다:

```rust
// Returns a reference to the longer of two string slices
// 'a means: the returned reference lives at least as long as
// the shorter of the two input lifetimes
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("abcdef");

    // Case 1: both references live long enough ✓
    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result); // OK
    }

    // Case 2: result tries to outlive one of the inputs ✗
    // let result;
    // {
    //     let string2 = String::from("xyz");
    //     result = longest(string1.as_str(), string2.as_str());
    // }
    // println!("Longest: {}", result); // ERROR: string2 doesn't live long enough
}
```

### 4.1 'a가 실제로 나타내는 것

```
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str

The lifetime 'a is the OVERLAP (intersection) of x's and y's lifetimes:

    string1: ──────────────────────────────────
    string2:        ─────────────────
                    ▲               ▲
                    │    'a = this  │
                    │    overlap    │
                    └───────────────┘

The return value is valid only within this overlap region.
```

### 4.2 라이프타임 표기가 필요 없는 경우

함수가 하나의 입력에서만 빌린다면 컴파일러가 알아낼 수 있습니다:

```rust
// Only one input reference → compiler knows the output borrows from it
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

// Returning an owned value → no borrow relationship at all
fn make_greeting(name: &str) -> String {
    format!("Hello, {}!", name) // returns an owned String, not a reference
}
```

---

## 5. 라이프타임 생략 규칙

초기 Rust는 모든 곳에 라이프타임 표기를 요구했습니다. 번거로워서 컴파일러가 세 가지 **생략(elision) 규칙**을 학습했습니다 — 너무 흔한 패턴이라 컴파일러가 자동으로 적용합니다:

```
Rule 1: Each input reference gets its own lifetime parameter.

    fn foo(x: &str)            → fn foo<'a>(x: &'a str)
    fn foo(x: &str, y: &str)   → fn foo<'a, 'b>(x: &'a str, y: &'b str)

Rule 2: If there is exactly ONE input lifetime, it is assigned
         to ALL output references.

    fn foo(x: &str) -> &str    → fn foo<'a>(x: &'a str) -> &'a str
    (This is why first_word works without annotations!)

Rule 3: If one of the inputs is &self or &mut self, the lifetime
         of self is assigned to all output references.

    impl Foo {
        fn bar(&self, x: &str) -> &str
        // becomes:
        fn bar<'a, 'b>(&'a self, x: &'b str) -> &'a str
        // output borrows from self, not from x
    }
```

### 5.1 규칙 단계별 적용

```
Example: fn longest(x: &str, y: &str) -> &str

Step 1 (Rule 1): fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &str
Step 2 (Rule 2): TWO input lifetimes → Rule 2 does not apply
Step 3 (Rule 3): No &self → Rule 3 does not apply

Result: output lifetime is STILL ambiguous → compiler error!
         You MUST add explicit annotations.
```

```
Example: fn first_word(s: &str) -> &str

Step 1 (Rule 1): fn first_word<'a>(s: &'a str) -> &str
Step 2 (Rule 2): ONE input lifetime → assign to output
                  fn first_word<'a>(s: &'a str) -> &'a str

Result: fully resolved → no annotations needed!
```

---

## 6. 구조체 정의에서의 라이프타임

구조체가 참조를 보유할 때, 참조된 데이터가 구조체보다 오래 살아남도록 라이프타임 표기가 필요합니다:

```rust
// ImportantExcerpt borrows from a string — it cannot outlive that string
#[derive(Debug)]
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    // Rule 3 applies: output borrows from &self
    fn level(&self) -> i32 {
        3
    }

    // Rule 3: returned &str borrows from &self (not from announcement)
    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");

    // The excerpt borrows from novel — it must not outlive novel
    let first_sentence;
    {
        let i = novel.find('.').unwrap_or(novel.len());
        first_sentence = ImportantExcerpt {
            part: &novel[..i],
        };
    }
    // first_sentence is still valid because novel is still alive
    println!("{:?}", first_sentence);
}
```

### 6.1 여러 라이프타임을 가진 구조체

```rust
// Sometimes fields borrow from different sources
#[derive(Debug)]
struct Highlight<'text, 'query> {
    text: &'text str,
    query: &'query str,
}

impl<'text, 'query> Highlight<'text, 'query> {
    fn new(text: &'text str, query: &'query str) -> Self {
        Highlight { text, query }
    }

    fn matches(&self) -> bool {
        self.text.contains(self.query)
    }
}

fn main() {
    let document = String::from("Rust is a systems programming language");
    let search = String::from("systems");

    let highlight = Highlight::new(&document, &search);
    println!("Match: {}", highlight.matches()); // true
}
```

---

## 7. 'static 라이프타임

`'static`은 가능한 가장 긴 라이프타임 — 프로그램 전체 실행 기간입니다:

```rust
fn main() {
    // String literals are 'static — they're embedded in the binary
    let s: &'static str = "I live forever";

    // Type-level constants are also 'static
    static GREETING: &str = "Hello, world!";
    println!("{}", GREETING);
}
```

### 7.1 'static이 항상 "리터럴"을 의미하지는 않음

소유된 타입은 아무것도 빌리지 않으므로 `'static` 바운드를 만족합니다:

```rust
use std::fmt::Display;

// T: Display + 'static means T implements Display AND
// does not contain any non-'static references
fn print_static<T: Display + 'static>(item: T) {
    println!("{}", item);
}

fn main() {
    print_static(42);                    // i32 is 'static (no references)
    print_static(String::from("hello")); // String is 'static (owns its data)
    // print_static(&String::from("hello")); // temporary reference — NOT 'static

    let s = String::from("hello");
    // print_static(&s); // &String has lifetime of s, which is not 'static
}
```

### 7.2 에러 메시지에서 'static을 보게 되는 경우

흔한 혼란의 원인입니다:

```
error: `x` does not live long enough
  --> src/main.rs:5:5
   |
   = note: ...borrowed value must be valid for the static lifetime...
```

이것은 보통 참조를 소유된 값이 필요한 곳에 저장하려 한다는 의미입니다. 해결책은 거의 항상 `'static`을 추가하는 것이 아니라 **데이터를 복제하거나 소유**하는 것입니다:

```rust
// BAD: fighting the compiler with 'static
// fn get_name() -> &'static str {
//     let name = String::from("Rust");
//     &name // ERROR: returning reference to local
// }

// GOOD: return an owned String
fn get_name() -> String {
    String::from("Rust")
}
```

---

## 8. 제네릭 타입의 라이프타임 바운드

라이프타임을 제네릭 및 트레이트 바운드와 결합할 수 있습니다:

```rust
use std::fmt::Display;

// T must outlive 'a AND implement Display
fn announce<'a, T: Display>(text: &'a str, extra: T) -> &'a str {
    println!("Announcement: {}", extra);
    text
}

// Lifetime bound on a generic: T: 'a means "T outlives 'a"
// This is needed when T might contain references
struct Wrapper<'a, T: 'a> {
    value: &'a T,
}

fn main() {
    let x = 42;
    let w = Wrapper { value: &x };
    println!("wrapped: {}", w.value);

    let text = String::from("hello");
    let result = announce(&text, 42);
    println!("{}", result);
}
```

### 8.1 실전에서의 라이프타임 바운드

```rust
// A cache that borrows its data source
struct Cache<'src, T>
where
    T: AsRef<str> + 'src, // T must outlive 'src and be convertible to &str
{
    source: &'src T,
    cached_len: usize,
}

impl<'src, T: AsRef<str> + 'src> Cache<'src, T> {
    fn new(source: &'src T) -> Self {
        let len = source.as_ref().len();
        Cache {
            source,
            cached_len: len,
        }
    }

    fn get(&self) -> &str {
        self.source.as_ref()
    }
}

fn main() {
    let data = String::from("Hello, lifetime bounds!");
    let cache = Cache::new(&data);
    println!("Cached ({} bytes): {}", cache.cached_len, cache.get());
}
```

---

## 9. 일반 패턴과 안티패턴

### 9.1 패턴: 참조 대신 소유된 데이터 반환

라이프타임 표기가 복잡해지면 자문해보세요: "그냥 소유된 값을 반환할 수 있지 않을까?"

```rust
// Complex: lifetime annotations, borrow checker gymnastics
fn get_display_name<'a>(first: &'a str, last: &'a str, use_full: bool) -> &'a str {
    if use_full {
        // PROBLEM: can't return a newly created string as a reference
        // let full = format!("{} {}", first, last);
        // &full  // ERROR: returning reference to local variable
        first // workaround — but not what we wanted
    } else {
        first
    }
}

// Simple: just return a String
fn get_display_name_owned(first: &str, last: &str, use_full: bool) -> String {
    if use_full {
        format!("{} {}", first, last) // no lifetime issues!
    } else {
        first.to_string()
    }
}
```

### 9.2 패턴: 생성자 인수에서 빌리는 구조체

```rust
// Good pattern: struct borrows data that outlives it
struct Parser<'input> {
    input: &'input str,
    position: usize,
}

impl<'input> Parser<'input> {
    fn new(input: &'input str) -> Self {
        Parser { input, position: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.position += c.len_utf8();
        }
    }

    fn remaining(&self) -> &'input str {
        &self.input[self.position..]
    }
}

fn main() {
    let source = String::from("hello world");
    let mut parser = Parser::new(&source);

    while let Some(c) = parser.peek() {
        print!("[{}]", c);
        parser.advance();
    }
    println!();
    // [h][e][l][l][o][ ][w][o][r][l][d]
}
```

### 9.3 안티패턴: 자기 참조 구조체

```rust
// This CANNOT work in safe Rust:
// struct SelfRef {
//     data: String,
//     slice: &str, // wants to borrow from data above
// }
//
// Why? If SelfRef moves in memory, slice becomes a dangling pointer.
// The borrow checker prevents this at compile time.

// Solutions:
// 1. Store an index instead of a reference
struct TextWithHighlight {
    text: String,
    highlight_start: usize,
    highlight_end: usize,
}

impl TextWithHighlight {
    fn highlighted(&self) -> &str {
        &self.text[self.highlight_start..self.highlight_end]
    }
}

// 2. Use separate structs with explicit lifetime relationship
// 3. Use Pin + unsafe (advanced — beyond this lesson)
// 4. Use crates like ouroboros or self_cell
```

### 9.4 안티패턴: 불필요한 라이프타임 표기

```rust
// Unnecessary: the compiler applies elision Rule 2 automatically
// fn first_char<'a>(s: &'a str) -> &'a str {
//     &s[..1]
// }

// Better: let elision do its job
fn first_char(s: &str) -> &str {
    &s[..1]
}

// Only add lifetime annotations when the compiler asks for them
// or when you need to express a relationship between multiple references
```

### 9.5 결정 흐름도

```
Do I need lifetime annotations?

  Does my function take references AND return a reference?
  ├── No  → You don't need them
  └── Yes → Does it take exactly ONE reference input?
            ├── Yes → Elision handles it (Rule 2)
            └── No  → Is one of the inputs &self?
                      ├── Yes → Elision handles it (Rule 3)
                      └── No  → You need explicit lifetimes!

  Does my struct hold references?
  ├── No  → You don't need them
  └── Yes → You MUST annotate the struct with lifetimes
```

---

## 10. 연습 문제

### 문제 1: 첫 번째와 마지막 요소
`first_and_last<'a>(words: &'a [String]) -> (&'a str, &'a str)` 함수를 작성하세요. 슬라이스의 첫 번째와 마지막 문자열에 대한 참조를 반환합니다. 빈 슬라이스에 대해서는 `("", "")`를 반환합니다. 함수 호출 이후에도 반환된 참조를 사용하여 유효성을 검증하세요.

### 문제 2: 가장 긴 줄
`longest_line<'a>(text: &'a str) -> &'a str` 함수를 작성하세요. 여러 줄로 구성된 문자열을 받아 가장 긴 줄에 대한 참조를 반환합니다. `.lines()`를 사용하여 순회하세요. 길이가 다양한 여러 줄이 있는 문자열로 테스트하세요.

### 문제 3: 라이프타임이 있는 구조체
`&'a str`을 보유하고 한 번에 한 단어씩 순회하는 `WordIterator<'a>` 구조체를 만드세요(`split_whitespace`의 수동 구현과 유사합니다). `Item = &'a str`로 `Iterator`를 구현하세요. `for` 루프와 `.count()`, `.collect::<Vec<_>>()`같은 이터레이터 메서드로 테스트하세요.

### 문제 4: 여러 라이프타임
**서로 다른** 라이프타임을 가진 두 문자열 슬라이스를 보유하는 `Merger<'a, 'b>` 구조체를 작성하세요. 두 슬라이스를 연결하는(소유된 데이터 반환 — 라이프타임 문제 없음) `merge(&self) -> String` 메서드를 구현하세요. 그런 다음 올바른 라이프타임을 가진 개별 참조를 반환하는 `first(&self) -> &'a str`와 `second(&self) -> &'b str`를 구현하세요. 두 소스 문자열이 서로 다른 스코프를 가질 수 있음을 증명하세요.

### 문제 5: 라이프타임 디버깅
다음 코드는 컴파일되지 않습니다. 라이프타임 문제를 파악하고, 실패하는 이유를 설명하고, 수정하세요(유효한 수정 방법이 여러 가지일 수 있음):

```rust
fn longest_with_announcement(x: &str, y: &str, ann: &str) -> &str {
    println!("Announcement: {}", ann);
    if x.len() > y.len() { x } else { y }
}
```

그런 다음 `ann`이 `x`와 `y`와 **다른** 라이프타임을 갖는 버전을 작성하여, announcement가 반환값만큼 오래 살아있을 필요가 없음을 보여주세요.

---

## 참고 자료

- [The Rust Programming Language, Ch. 10.3: Validating References with Lifetimes](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)
- [Rust by Example: Lifetimes](https://doc.rust-lang.org/rust-by-example/scope/lifetime.html)
- [The Rustonomicon: Lifetimes](https://doc.rust-lang.org/nomicon/lifetimes.html)
- [Common Rust Lifetime Misconceptions (pretzelhammer)](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md)
- [Lifetime Elision Rules (Rust Reference)](https://doc.rust-lang.org/reference/lifetime-elision.html)

---

**이전**: [트레이트와 제네릭](./10_Traits_and_Generics.md) | **다음**: [클로저와 이터레이터](./12_Closures_and_Iterators.md)

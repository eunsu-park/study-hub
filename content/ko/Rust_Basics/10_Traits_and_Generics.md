# 10. 트레이트와 제네릭(Traits and Generics)

**이전**: [에러 처리](./09_Error_Handling.md) | **다음**: [라이프타임](./11_Lifetimes.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 메서드와 기본 구현을 포함한 트레이트(Trait)를 정의하고, 커스텀 타입에 구현할 수 있다
2. 인라인 문법과 `where` 절을 모두 사용하여 트레이트 바운드(trait bound)가 있는 제네릭(Generic) 함수와 구조체를 작성할 수 있다
3. 정적 디스패치(static dispatch, 제네릭)와 동적 디스패치(dynamic dispatch, `dyn Trait`)를 구분하고 적절하게 선택할 수 있다
4. 자신의 타입에 주요 표준 라이브러리 트레이트(`Display`, `From`, `Iterator`, `Default`)를 구현할 수 있다
5. 슈퍼트레이트(supertrait)를 사용하여 트레이트 계층 구조를 표현할 수 있다

---

트레이트(Trait)와 제네릭(Generic)은 Rust의 추상화 접근 방식의 두 기둥입니다. 트레이트는 공유 동작을 정의합니다 — Java의 인터페이스나 Swift의 프로토콜과 비슷합니다 — 반면 제네릭은 성능을 희생하지 않고 여러 타입에 걸쳐 동작하는 코드를 작성할 수 있게 해줍니다. 함께 사용하면 동적 타입 언어의 유연성과 정적 타입 언어의 속도를 모두 얻을 수 있습니다.

비유하자면: 트레이트는 직무 설명서("정렬, 비교, 출력이 가능해야 함")이고, 제네릭은 "이 직무 설명을 충족하는 누구든 이 역할을 맡을 수 있다"는 채용 정책입니다. 그러면 컴파일러가 역할을 맡는 각 구체적인 타입에 대해 특화된 코드를 생성합니다 — 런타임 오버헤드가 없습니다.

## 목차
1. [트레이트 정의](#1-트레이트-정의)
2. [트레이트 구현](#2-트레이트-구현)
3. [트레이트 바운드](#3-트레이트-바운드)
4. [impl Trait 문법](#4-impl-trait-문법)
5. [where 절](#5-where-절)
6. [제네릭 구조체와 열거형](#6-제네릭-구조체와-열거형)
7. [표준 라이브러리 트레이트](#7-표준-라이브러리-트레이트)
8. [트레이트 객체: 동적 디스패치](#8-트레이트-객체-동적-디스패치)
9. [슈퍼트레이트와 트레이트 상속](#9-슈퍼트레이트와-트레이트-상속)
10. [연습 문제](#10-연습-문제)

---

## 1. 트레이트 정의

트레이트는 타입이 구현할 수 있는 메서드 집합을 정의합니다. 일부 메서드는 기본 구현을 가질 수 있습니다:

```rust
trait Summary {
    // Required method — implementors MUST define this
    fn summarize_author(&self) -> String;

    // Default method — implementors CAN override this
    fn summarize(&self) -> String {
        // Default implementation calls the required method
        format!("(Read more from {}...)", self.summarize_author())
    }
}
```

### 1.1 여러 메서드를 가진 트레이트

```rust
trait Shape {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;

    // Default: uses other methods defined in the same trait
    fn describe(&self) -> String {
        format!(
            "Shape with area {:.2} and perimeter {:.2}",
            self.area(),
            self.perimeter()
        )
    }
}
```

### 1.2 계약으로서의 트레이트

```
Trait: Summary
┌──────────────────────────────┐
│  Required:                   │
│    fn summarize_author(&self)│
│                              │
│  Provided (default):         │
│    fn summarize(&self)       │
│      → calls summarize_author│
└──────────────────────────────┘
        │
        │ impl Summary for ...
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  NewsArticle  │  │   Tweet      │  │   BlogPost   │
│  ✓ author()   │  │  ✓ author()  │  │  ✓ author()  │
│  ✓ summarize()│  │  (default)   │  │  ✓ summarize│
│    (custom)   │  │              │  │    (custom)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 2. 트레이트 구현

`impl TraitName for TypeName`으로 트레이트를 구현합니다:

```rust
trait Summary {
    fn summarize_author(&self) -> String;
    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }
}

struct NewsArticle {
    title: String,
    author: String,
    content: String,
}

impl Summary for NewsArticle {
    fn summarize_author(&self) -> String {
        self.author.clone()
    }

    // Override the default summarize()
    fn summarize(&self) -> String {
        format!("{}, by {} — {}", self.title, self.author, &self.content[..50.min(self.content.len())])
    }
}

struct Tweet {
    username: String,
    text: String,
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
    // Uses the default summarize() — no need to override
}

fn main() {
    let article = NewsArticle {
        title: String::from("Rust 2025 Edition Released"),
        author: String::from("Niko Matsakis"),
        content: String::from("The Rust team announced the 2025 edition today with exciting new features..."),
    };

    let tweet = Tweet {
        username: String::from("rustlang"),
        text: String::from("Rust 2025 is here!"),
    };

    println!("{}", article.summarize());
    // "Rust 2025 Edition Released, by Niko Matsakis — The Rust team announced..."

    println!("{}", tweet.summarize());
    // "(Read more from @rustlang...)"
}
```

### 2.1 고아 규칙(Orphan Rule)

트레이트를 타입에 구현하려면 **둘 중 하나 이상**이 현재 크레이트(crate)에 있어야 합니다. 이것은 충돌하는 구현을 방지합니다:

```rust
// In YOUR crate:

// OK: your trait on a foreign type
// impl MyTrait for Vec<i32> { ... }

// OK: a foreign trait on your type
// impl Display for MyStruct { ... }

// NOT OK: a foreign trait on a foreign type
// impl Display for Vec<i32> { ... }  // compiler error!
```

---

## 3. 트레이트 바운드

트레이트 바운드(Trait Bound)는 제네릭 타입을 제한합니다. "T는 이 트레이트를 구현하는 한 어떤 타입이든 될 수 있다"는 의미입니다:

```rust
use std::fmt::Display;

// T must implement both Display and PartialOrd
fn largest<T: Display + PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    println!("The largest is {}", largest); // Display lets us print
    largest
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    largest(&numbers); // T = i32, which is Display + PartialOrd

    let chars = vec!['y', 'm', 'a', 'q'];
    largest(&chars); // T = char, also Display + PartialOrd
}
```

### 3.1 여러 트레이트 바운드

`+`를 사용하여 여러 트레이트를 요구합니다:

```rust
use std::fmt::{Display, Debug};

fn print_both<T: Display + Debug>(item: &T) {
    println!("Display: {}", item);    // uses Display
    println!("Debug:   {:?}", item);  // uses Debug
}

fn main() {
    print_both(&42);
    print_both(&"hello");
}
```

---

## 4. impl Trait 문법

`impl Trait`는 두 위치에서 동작하는 문법적 설탕(syntactic sugar)입니다:

### 4.1 인수 위치 (트레이트 바운드의 설탕)

```rust
use std::fmt::Display;

// These two signatures are equivalent:
fn notify_verbose<T: Display>(item: &T) {
    println!("Breaking: {}", item);
}

fn notify(item: &impl Display) {
    println!("Breaking: {}", item);
}
// "impl Display" in argument position means "some type that implements Display"

fn main() {
    notify(&"earthquake");
    notify(&42);
}
```

### 4.2 반환 위치 (불투명 반환 타입)

```rust
fn make_greeting(name: &str) -> impl std::fmt::Display {
    // The caller knows the return type implements Display,
    // but NOT which concrete type it is (it's String here)
    format!("Hello, {}!", name)
}

fn main() {
    let greeting = make_greeting("Rust");
    println!("{}", greeting); // works because it implements Display
    // But you can't call String-specific methods on greeting
}
```

반환 위치의 `impl Trait`는 이름을 붙일 수 없는 타입을 가진 클로저와 이터레이터에 특히 유용합니다:

```rust
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    // Each closure has a unique, anonymous type — impl Fn lets us return it
    move |y| x + y
}

fn even_numbers(limit: i32) -> impl Iterator<Item = i32> {
    (0..limit).filter(|n| n % 2 == 0)
}

fn main() {
    let add_five = make_adder(5);
    println!("{}", add_five(3)); // 8

    for n in even_numbers(10) {
        print!("{} ", n); // 0 2 4 6 8
    }
    println!();
}
```

---

## 5. where 절

트레이트 바운드가 복잡해지면 `where` 절로 가독성을 높일 수 있습니다:

```rust
use std::fmt::{Debug, Display};
use std::hash::Hash;

// Cluttered: hard to read with inline bounds
fn process_cluttered<T: Display + Debug + Clone + Hash, U: Debug + Default>(t: &T, u: &U) {
    println!("{:?} {:?}", t, u);
}

// Clean: where clause separates bounds from the signature
fn process<T, U>(t: &T, u: &U)
where
    T: Display + Debug + Clone + Hash,
    U: Debug + Default,
{
    println!("{:?} {:?}", t, u);
}

fn main() {
    process(&42, &String::new());
}
```

`where` 절은 인라인 문법으로는 표현할 수 없는 바운드도 나타낼 수 있습니다:

```rust
use std::fmt::Debug;

// Bound on an associated type — only possible with where
fn print_pairs<T>(items: &T)
where
    T: IntoIterator + Clone,
    T::Item: Debug, // bound on the associated type Item
{
    for item in items.clone() {
        println!("{:?}", item);
    }
}

fn main() {
    print_pairs(&vec![1, 2, 3]);
}
```

---

## 6. 제네릭 구조체와 열거형

제네릭은 함수에만 국한되지 않습니다 — 구조체와 열거형도 제네릭이 될 수 있습니다:

### 6.1 제네릭 구조체

```rust
#[derive(Debug)]
struct Point<T> {
    x: T,
    y: T,
}

// Implement methods for ALL Point<T>
impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

// Implement methods ONLY for Point<f64>
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

fn main() {
    let int_point = Point::new(5, 10);
    let float_point = Point::new(1.0, 5.0);

    println!("{:?}", int_point);
    println!("distance: {:.2}", float_point.distance_from_origin());
    // int_point.distance_from_origin(); // ERROR: only defined for f64
}
```

### 6.2 여러 타입 파라미터

```rust
#[derive(Debug)]
struct Pair<A, B> {
    first: A,
    second: B,
}

impl<A, B> Pair<A, B> {
    fn new(first: A, second: B) -> Self {
        Pair { first, second }
    }

    // Method that mixes type parameters from two different Pairs
    fn mix<C, D>(self, other: Pair<C, D>) -> Pair<A, D> {
        Pair {
            first: self.first,
            second: other.second,
        }
    }
}

fn main() {
    let p1 = Pair::new("hello", 42);
    let p2 = Pair::new(3.14, true);
    let mixed = p1.mix(p2);
    println!("{:?}", mixed); // Pair { first: "hello", second: true }
}
```

### 6.3 제네릭 열거형

가장 유명한 두 가지 제네릭 열거형을 이미 사용해보셨을 것입니다:

```rust
// From the standard library:
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Your own generic enum
#[derive(Debug)]
enum Tree<T> {
    Leaf(T),
    Node {
        value: T,
        left: Box<Tree<T>>,
        right: Box<Tree<T>>,
    },
}

fn main() {
    let tree = Tree::Node {
        value: 10,
        left: Box::new(Tree::Leaf(5)),
        right: Box::new(Tree::Node {
            value: 15,
            left: Box::new(Tree::Leaf(12)),
            right: Box::new(Tree::Leaf(20)),
        }),
    };
    println!("{:#?}", tree);
}
```

---

## 7. 표준 라이브러리 트레이트

Rust 코드 전반에서 등장하는 트레이트들입니다. 자신의 타입에 구현하면 강력한 기능이 활성화됩니다:

### 7.1 Display — 사람이 읽기 쉬운 출력

```rust
use std::fmt;

struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

fn main() {
    let red = Color { r: 255, g: 0, b: 0 };
    println!("{}", red);   // #FF0000
    // println!("{:?}", red); // would need #[derive(Debug)]
}
```

### 7.2 From과 Into — 타입 변환

```rust
struct Celsius(f64);
struct Fahrenheit(f64);

// Implementing From<X> for Y automatically gives you Into<Y> for X
impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

impl From<Fahrenheit> for Celsius {
    fn from(f: Fahrenheit) -> Self {
        Celsius((f.0 - 32.0) * 5.0 / 9.0)
    }
}

fn main() {
    let boiling = Celsius(100.0);

    // Using From explicitly
    let f = Fahrenheit::from(boiling);
    println!("{}F", f.0); // 212F

    // Using Into (available automatically)
    let body_temp = Fahrenheit(98.6);
    let c: Celsius = body_temp.into();
    println!("{:.1}C", c.0); // 37.0C
}
```

### 7.3 Iterator — 타입을 순회 가능하게 만들기

```rust
struct Countdown {
    value: i32,
}

impl Countdown {
    fn new(start: i32) -> Self {
        Countdown { value: start }
    }
}

impl Iterator for Countdown {
    type Item = i32; // the associated type — what the iterator yields

    fn next(&mut self) -> Option<Self::Item> {
        if self.value > 0 {
            let current = self.value;
            self.value -= 1;
            Some(current)
        } else {
            None // signals the end of iteration
        }
    }
}

fn main() {
    // Now Countdown works with for loops and all iterator adaptors
    for n in Countdown::new(5) {
        print!("{} ", n);
    }
    println!(); // 5 4 3 2 1

    // And with iterator methods
    let sum: i32 = Countdown::new(10).filter(|n| n % 2 == 0).sum();
    println!("Sum of even numbers 1-10: {}", sum); // 2+4+6+8+10 = 30
}
```

### 7.4 Default — 합리적인 기본값

```rust
#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    max_connections: usize,
    verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: String::from("localhost"),
            port: 8080,
            max_connections: 100,
            verbose: false,
        }
    }
}

fn main() {
    // Create with all defaults
    let config = Config::default();
    println!("{:?}", config);

    // Override specific fields using struct update syntax
    let custom = Config {
        port: 3000,
        verbose: true,
        ..Config::default() // fill the rest from default
    };
    println!("{:?}", custom);
}
```

---

## 8. 트레이트 객체: 동적 디스패치

제네릭은 **정적 디스패치(static dispatch)**를 사용합니다 — 컴파일러가 각 구체적인 타입에 대해 별도의 함수 버전을 생성합니다(단형화, monomorphization). 빠르지만 바이너리 크기가 커집니다.

**트레이트 객체(Trait Object)**는 **동적 디스패치(dynamic dispatch)**를 사용합니다 — 런타임에 vtable 포인터를 통해 단일 함수가 모든 타입을 처리합니다:

```
Static Dispatch (generics):            Dynamic Dispatch (dyn Trait):

  fn draw<T: Shape>(s: &T)              fn draw(s: &dyn Shape)
           │                                      │
    ┌──────┼──────┐                    ┌──────────┴──────────┐
    ▼      ▼      ▼                    ▼                     │
draw_Circle  draw_Rect  draw_Tri     draw(...) single fn     │
(specialized  (specialized            │                      │
 for each      for each)              uses vtable            │
 type)                                at runtime             │
                                      ┌──────────────┐       │
                                      │ vtable       │       │
                                      │  area → ...  │       │
                                      │  draw → ...  │       │
                                      └──────────────┘       │
```

```rust
trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &str;
}

struct Circle { radius: f64 }
struct Rectangle { width: f64, height: f64 }

impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
    fn name(&self) -> &str { "Circle" }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn name(&self) -> &str { "Rectangle" }
}

// Dynamic dispatch: accepts ANY type that implements Shape
fn print_area(shape: &dyn Shape) {
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

fn main() {
    let circle = Circle { radius: 5.0 };
    let rect = Rectangle { width: 3.0, height: 4.0 };

    print_area(&circle);
    print_area(&rect);

    // Heterogeneous collection — different types in one Vec!
    // This is only possible with trait objects
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 1.0 }),
        Box::new(Rectangle { width: 2.0, height: 3.0 }),
        Box::new(Circle { radius: 4.0 }),
    ];

    let total_area: f64 = shapes.iter().map(|s| s.area()).sum();
    println!("Total area: {:.2}", total_area);
}
```

### 8.1 언제 어느 것을 사용할까

| 특징 | 제네릭 (정적) | 트레이트 객체 (동적) |
|------|--------------|---------------------|
| 성능 | 빠름 (간접 참조 없음) | 약간의 오버헤드 (vtable 조회) |
| 바이너리 크기 | 큼 (코드 중복) | 작음 (하나의 복사본) |
| 이종 컬렉션 | 불가 | 가능 |
| 컴파일 타임에 알 수 있음 | 예 (단형화됨) | 아니오 (런타임 다형성) |
| 연관 타입 사용 가능 | 예 | 제한적 |

**원칙:** 제네릭으로 시작하세요. 이종 컬렉션이 필요하거나 런타임까지 구체적인 타입을 알 수 없을 때(예: 플러그인 시스템, GUI 프레임워크) `dyn Trait`을 사용하세요.

---

## 9. 슈퍼트레이트와 트레이트 상속

슈퍼트레이트(Supertrait)는 다른 트레이트를 먼저 구현해야 하는 트레이트입니다:

```rust
use std::fmt;

// Printable requires Display — it's a supertrait relationship
// Any type implementing Printable MUST also implement Display
trait Printable: fmt::Display {
    fn print(&self) {
        println!("{}", self); // we can use {} because Display is guaranteed
    }

    fn print_heading(&self) {
        println!("=== {} ===", self);
    }
}

struct Document {
    title: String,
    content: String,
}

// Must implement Display first (the supertrait)
impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.title, self.content)
    }
}

// Then we can implement Printable
impl Printable for Document {}
// Uses default implementations of print() and print_heading()

fn main() {
    let doc = Document {
        title: String::from("Rust Guide"),
        content: String::from("Traits are powerful!"),
    };

    doc.print();         // "Rust Guide: Traits are powerful!"
    doc.print_heading(); // "=== Rust Guide: Traits are powerful! ==="
}
```

### 9.1 여러 슈퍼트레이트

```rust
use std::fmt::{Debug, Display};

// Requires BOTH Debug and Display
trait Loggable: Debug + Display {
    fn log(&self) {
        // Debug for log files (structured), Display for users (human-readable)
        println!("[LOG] Display: {} | Debug: {:?}", self, self);
    }
}

#[derive(Debug)]
struct Event {
    name: String,
    severity: u8,
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Event '{}' (severity: {})", self.name, self.severity)
    }
}

impl Loggable for Event {}

fn main() {
    let event = Event {
        name: String::from("disk_full"),
        severity: 9,
    };
    event.log();
    // [LOG] Display: Event 'disk_full' (severity: 9) | Debug: Event { name: "disk_full", severity: 9 }
}
```

---

## 10. 연습 문제

### 문제 1: 넓이 트레이트(Area Trait)
`area(&self) -> f64`와 `perimeter(&self) -> f64` 메서드를 가진 `Measurable` 트레이트를 정의하세요. `Circle`, `Rectangle`, `Triangle`에 구현하세요. 슬라이스에서 가장 큰 넓이를 반환하는 제네릭 함수 `largest_area<T: Measurable>(shapes: &[T]) -> f64`를 작성하세요.

### 문제 2: From 변환
`Celsius(f64)`, `Fahrenheit(f64)`, `Kelvin(f64)` 변형을 가진 `Temperature` 열거형을 만드세요. `From<Celsius> for Fahrenheit`, `From<Fahrenheit> for Celsius` 등의 변환을 구현하세요. 또한 단위 기호를 포함하여 온도를 출력하도록 `Display`를 구현하세요.

### 문제 3: 커스텀 이터레이터(Custom Iterator)
`Iterator<Item = u64>`를 구현하는 `FibIterator` 구조체를 만드세요. 피보나치 수를 무한히 생성해야 합니다. 이터레이터 메서드와 함께 사용하세요: 처음 10개의 피보나치 수를 수집하고, 1000보다 큰 첫 번째 수를 찾고, 처음 20개의 합을 계산하세요.

### 문제 4: 트레이트 객체를 이용한 플러그인 시스템(Plugin System with Trait Objects)
`name(&self) -> &str`와 `execute(&self, input: &str) -> String` 메서드를 가진 `Plugin` 트레이트를 설계하세요. 세 가지 플러그인 구조체를 만드세요(예: `UppercasePlugin`, `ReversePlugin`, `CensorPlugin`). `Vec<Box<dyn Plugin>>`을 저장하고 주어진 입력에 대해 모든 플러그인을 순서대로 실행하는 `PluginRunner`를 작성하세요.

### 문제 5: 제네릭 스택(Generic Stack)
`push`, `pop -> Option<T>`, `peek -> Option<&T>`, `is_empty`, `size` 메서드를 가진 제네릭 `Stack<T>`를 구현하세요. 그런 다음 트레이트 바운드 `where T: Ord`를 가진 `min() -> Option<&T>` 메서드를 추가하세요. `i32`, `String`, 커스텀 구조체로 스택을 사용하는 테스트를 작성하세요.

---

## 참고 자료

- [The Rust Programming Language, Ch. 10: Generic Types, Traits, and Lifetimes](https://doc.rust-lang.org/book/ch10-00-generics.html)
- [Rust by Example: Traits](https://doc.rust-lang.org/rust-by-example/trait.html)
- [Rust by Example: Generics](https://doc.rust-lang.org/rust-by-example/generics.html)
- [The Rust Reference: Trait Objects](https://doc.rust-lang.org/reference/types/trait-object.html)
- [std::fmt::Display](https://doc.rust-lang.org/std/fmt/trait.Display.html)
- [std::convert::From](https://doc.rust-lang.org/std/convert/trait.From.html)

---

**이전**: [에러 처리](./09_Error_Handling.md) | **다음**: [라이프타임](./11_Lifetimes.md)

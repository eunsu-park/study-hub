# 22. 고급 트레이트

**이전**: [절차적 매크로](./05_Procedural_Macros.md) | **다음**: [고급 비동기](./07_Advanced_Async.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트레이트 객체(`dyn Trait`)와 제네릭(`impl Trait`)의 장단점을 파악하여 선택하기
2. 대여 이터레이터(Lending Iterator)와 비동기 트레이트를 위한 GATs(제네릭 연관 타입) 사용하기
3. 블랭킷 구현(Blanket Implementation) 작성 및 고아 규칙(Orphan Rule) 이해하기
4. 타입 수준 상태 머신을 포함한 고급 연관 타입 패턴 적용하기
5. 슈퍼트레이트, 마커 트레이트, 봉인된 트레이트(Sealed Trait)로 트레이트 계층 설계하기

---

레슨 10에서 트레이트와 제네릭의 기초를 소개했습니다. 이 레슨은 고급 영역을 다룹니다: 트레이트 객체 vs 단형성화(monomorphization)의 트레이드오프, GATs(Rust 1.65에서 안정화), 일관성(coherence)과 고아 규칙, 프로덕션 Rust 라이브러리에서 사용되는 패턴.

## 목차
1. [정적 vs 동적 디스패치](#1-정적-vs-동적-디스패치)
2. [트레이트 객체 상세](#2-트레이트-객체-상세)
3. [객체 안전성](#3-객체-안전성)
4. [연관 타입 vs 제네릭 파라미터](#4-연관-타입-vs-제네릭-파라미터)
5. [제네릭 연관 타입(GATs)](#5-제네릭-연관-타입gats)
6. [블랭킷 구현](#6-블랭킷-구현)
7. [일관성과 고아 규칙](#7-일관성과-고아-규칙)
8. [슈퍼트레이트](#8-슈퍼트레이트)
9. [마커 트레이트](#9-마커-트레이트)
10. [봉인된 트레이트](#10-봉인된-트레이트)
11. [고급 패턴](#11-고급-패턴)
12. [연습문제](#12-연습문제)

---

## 1. 정적 vs 동적 디스패치

Rust는 트레이트 기반 다형성을 위해 두 가지 디스패치 메커니즘을 제공합니다:

### 정적 디스패치 (제네릭 / `impl Trait`)

컴파일러가 각 구체 타입에 대해 특수화된 코드를 생성합니다 — **단형성화(monomorphization)**:

```rust
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing circle r={}", self.radius); }
}
impl Drawable for Square {
    fn draw(&self) { println!("Drawing square s={}", self.side); }
}

// 정적 디스패치 — 컴파일러가 draw_shape::<Circle>과 draw_shape::<Square>를 생성
fn draw_shape(shape: &impl Drawable) {
    shape.draw();
}
```

**장점**: 제로 비용 추상화, 인라이닝 가능, 힙 할당 없음.
**단점**: 더 큰 바이너리(코드 중복), 타입이 컴파일 타임에 알려져야 함.

### 동적 디스패치 (`dyn Trait`)

런타임에 vtable 조회 — 컴파일러가 함수의 **단일** 복사본을 생성:

```rust
// 동적 디스패치 — 단일 함수가 vtable을 통해 모든 Drawable 처리
fn draw_any(shape: &dyn Drawable) {
    shape.draw();  // 런타임에 vtable 조회
}

// 이종(heterogeneous) 컬렉션 저장 가능
fn draw_all(shapes: &[Box<dyn Drawable>]) {
    for shape in shapes {
        shape.draw();
    }
}
```

**장점**: 더 작은 바이너리, 이종 컬렉션, 런타임 다형성.
**단점**: vtable 간접 비용, 인라이닝 불가, 소유 값에 힙 할당 필요.

### 비교 표

| 측면 | 정적 (`impl Trait`) | 동적 (`dyn Trait`) |
|------|---------------------|---------------------|
| 디스패치 | 컴파일 타임 | 런타임 (vtable) |
| 성능 | 제로 비용 | 약간의 오버헤드 |
| 바이너리 크기 | 더 큼 (단형성화) | 더 작음 |
| 이종 컬렉션 | 불가 | 가능 |
| 객체 안전성 필요 | 아니오 | 예 |

---

## 2. 트레이트 객체 상세

트레이트 객체(`dyn Trait`)는 두 개의 포인터로 구성된 **팻 포인터(fat pointer)**입니다:

```
┌──────────────────────┐
│  data ptr  │ vtable ptr │
│  (8 bytes) │ (8 bytes)  │
└──────────────────────┘
```

**데이터 포인터**는 구체 값을 가리킵니다. **vtable 포인터**는 함수 포인터 테이블을 가리킵니다:

```
Circle의 Drawable 구현을 위한 vtable:
┌─────────────────────────────┐
│ drop_in_place: fn(*mut ())  │
│ size: usize                 │
│ align: usize                │
│ draw: fn(*const ())         │
└─────────────────────────────┘
```

### 트레이트 객체 라이프타임

트레이트 객체에는 암묵적 라이프타임 바운드가 있습니다:

```rust
// Box<dyn Trait>은 실제로 Box<dyn Trait + 'static>
// &'a dyn Trait은 &'a (dyn Trait + 'a)

trait Logger {
    fn log(&self, msg: &str);
}

// 트레이트 객체의 명시적 라이프타임
fn get_logger<'a>(loggers: &'a [Box<dyn Logger>]) -> &'a dyn Logger {
    &*loggers[0]
}

// 소유된 트레이트 객체 — 기본적으로 'static
fn create_logger() -> Box<dyn Logger> {
    struct StdoutLogger;
    impl Logger for StdoutLogger {
        fn log(&self, msg: &str) { println!("{msg}"); }
    }
    Box::new(StdoutLogger)
}
```

### 트레이트 객체의 다중 트레이트 바운드

```rust
use std::fmt::{Debug, Display};

// 불가: dyn Debug + Display (비-자동 트레이트는 하나만 허용)
// 하지만 슈퍼트레이트를 만들 수 있습니다:
trait DebugDisplay: Debug + Display {}
impl<T: Debug + Display> DebugDisplay for T {}

fn print_thing(thing: &dyn DebugDisplay) {
    println!("Debug: {:?}", thing);
    println!("Display: {}", thing);
}
```

---

## 3. 객체 안전성

모든 트레이트가 트레이트 객체로 사용될 수 있는 것은 아닙니다. 트레이트가 **객체 안전(object-safe)**하려면:

1. 모든 메서드에 리시버가 있어야 합니다 (`self`, `&self`, `&mut self` 등)
2. 메서드가 `Self`를 반환하지 않아야 합니다
3. 메서드에 제네릭 타입 파라미터가 없어야 합니다
4. 트레이트가 `Self: Sized`를 요구하지 않아야 합니다

```rust
// 객체 안전
trait Draw {
    fn draw(&self);
}

// 객체 안전하지 않음 — Self를 반환
trait Clonable {
    fn clone_self(&self) -> Self;  // 런타임에 크기를 결정할 수 없음
}

// 해결책: 객체 안전하지 않은 메서드에 Sized 요구
trait MixedTrait {
    fn object_safe_method(&self);

    fn non_object_safe_method(&self) -> Self
    where
        Self: Sized;  // 트레이트 객체에서 제외됨
}
```

---

## 4. 연관 타입 vs 제네릭 파라미터

연관 타입은 **일대일** 매핑을 강제합니다: 각 구현 타입이 정확히 하나의 연관 타입을 선택합니다:

```rust
// Iterator는 연관 타입을 가집니다 — 각 이터레이터가 하나의 아이템 타입을 생성
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// 제네릭 파라미터는 동일한 타입에 대한 여러 구현을 허용합니다:
trait GenericIterator<Item> {
    fn next(&mut self) -> Option<Item>;
}

// 연관 타입의 경우: Vec<i32> -> Item = i32 (하나의 선택)
// 제네릭의 경우: Vec<i32>는 GenericIterator<i32>와 GenericIterator<String> 모두 구현 가능
```

### 연관 타입 사용 시점

| 상황 | 연관 타입 | 제네릭 파라미터 |
|------|-----------|-----------------|
| 타입당 하나의 구현 | 선호 | 가능하지만 다루기 어려움 |
| 타입당 여러 구현 | 불가 | 사용 |
| 호출자가 타입 선택 | 아니오 | 예 |
| 타입 추론 | 더 나음 | 종종 더 명시적 |

### 타입 수준 상태 머신

연관 타입은 컴파일 타임 상태 머신을 가능하게 합니다:

```rust
trait ConnectionState {}
struct Disconnected;
struct Connected;
struct Authenticated;

impl ConnectionState for Disconnected {}
impl ConnectionState for Connected {}
impl ConnectionState for Authenticated {}

struct Connection<S: ConnectionState> {
    host: String,
    _state: std::marker::PhantomData<S>,
}

impl Connection<Disconnected> {
    fn new(host: &str) -> Self {
        Connection {
            host: host.to_string(),
            _state: std::marker::PhantomData,
        }
    }

    fn connect(self) -> Connection<Connected> {
        println!("Connecting to {}", self.host);
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

impl Connection<Connected> {
    fn authenticate(self, _password: &str) -> Connection<Authenticated> {
        println!("Authenticating...");
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }

    fn disconnect(self) -> Connection<Disconnected> {
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

impl Connection<Authenticated> {
    fn query(&self, sql: &str) -> String {
        format!("Running '{}' on {}", sql, self.host)
    }

    fn disconnect(self) -> Connection<Disconnected> {
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

fn main() {
    let conn = Connection::new("db.example.com");
    // conn.query("SELECT 1");  // 오류: Connection<Disconnected>에 `query` 메서드 없음

    let conn = conn.connect();
    // conn.query("SELECT 1");  // 오류: Connection<Connected>에 `query` 메서드 없음

    let conn = conn.authenticate("secret");
    let result = conn.query("SELECT 1");  // OK!
    println!("{result}");

    let _conn = conn.disconnect();
}
```

---

## 5. 제네릭 연관 타입(GATs)

GATs(Rust 1.65에서 안정화)는 연관 타입이 자체 제네릭 파라미터를 가질 수 있게 합니다:

```rust
trait LendingIterator {
    type Item<'a> where Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>>;
}
```

---

## 6. 블랭킷 구현

블랭킷 구현은 특정 바운드를 만족하는 모든 타입에 대해 트레이트를 구현합니다:

```rust
// 표준 라이브러리에서: Display를 구현하는 모든 타입이 ToString도 구현
impl<T: std::fmt::Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{self}")
    }
}
```

### 확장 트레이트

블랭킷 구현은 외부 타입에 메서드를 추가하는 **확장 트레이트(Extension Trait)** 패턴을 구동합니다:

```rust
trait IteratorExt: Iterator {
    fn take_every(self, n: usize) -> TakeEvery<Self>
    where
        Self: Sized,
    {
        TakeEvery { iter: self, n, count: 0 }
    }
}

// 블랭킷 구현 — 모든 Iterator가 이 메서드를 얻습니다
impl<I: Iterator> IteratorExt for I {}

struct TakeEvery<I> {
    iter: I,
    n: usize,
    count: usize,
}

impl<I: Iterator> Iterator for TakeEvery<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.iter.next()?;
            self.count += 1;
            if self.count % self.n == 0 {
                return Some(item);
            }
        }
    }
}

fn main() {
    let evens: Vec<_> = (1..=20).take_every(2).collect();
    println!("{evens:?}");  // [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
}
```

---

## 7. 일관성과 고아 규칙

Rust의 **일관성 규칙**은 모든 타입 + 트레이트 조합에 최대 하나의 구현만 있도록 보장합니다. **고아 규칙**이 핵심 제약입니다:

> 트레이트 또는 타입 중 **하나가 자신의 크레이트에 로컬**인 경우에만 타입에 대해 트레이트를 구현할 수 있습니다.

### 뉴타입 패턴 (해결 방법)

```rust
// 외부 타입을 래핑
struct PrettyVec<T>(Vec<T>);

impl<T: std::fmt::Display> std::fmt::Display for PrettyVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let items: Vec<String> = self.0.iter().map(|x| x.to_string()).collect();
        write!(f, "[{}]", items.join(", "))
    }
}
```

---

## 8. 슈퍼트레이트

슈퍼트레이트는 다른 트레이트의 구현을 요구하는 트레이트입니다:

```rust
use std::fmt;

trait Describable: fmt::Display + fmt::Debug {
    fn description(&self) -> String {
        format!("Display: {} | Debug: {:?}", self, self)
    }
}
```

---

## 9. 마커 트레이트

마커 트레이트는 메서드를 가지지 않습니다 — 컴파일러나 다른 코드에 의미론적 정보를 전달합니다:

```rust
// 표준 라이브러리 마커:
// Send — 스레드 간 전송 안전
// Sync — 스레드 간 참조 공유 안전
// Sized — 컴파일 타임에 알려진 크기
// Unpin — 고정(pin) 후 이동 안전
```

---

## 10. 봉인된 트레이트

봉인된 트레이트(Sealed Trait)는 정의된 크레이트 내에서만 구현할 수 있습니다. 이를 통해 하위 코드를 손상시키지 않고 나중에 메서드를 추가할 수 있습니다:

```rust
mod private {
    pub trait Sealed {}
}

// 비공개 슈퍼트레이트가 있는 공개 트레이트
pub trait DatabaseDriver: private::Sealed {
    fn connect(&self, url: &str) -> String;
    fn query(&self, sql: &str) -> Vec<String>;
}

// 이 크레이트의 타입만 Sealed를 구현할 수 있고, 따라서 DatabaseDriver도 구현 가능
pub struct PostgresDriver;
impl private::Sealed for PostgresDriver {}
impl DatabaseDriver for PostgresDriver {
    fn connect(&self, url: &str) -> String {
        format!("Connected to Postgres at {url}")
    }
    fn query(&self, sql: &str) -> Vec<String> {
        vec![format!("Postgres result for: {sql}")]
    }
}

pub struct SqliteDriver;
impl private::Sealed for SqliteDriver {}
impl DatabaseDriver for SqliteDriver {
    fn connect(&self, url: &str) -> String {
        format!("Connected to SQLite at {url}")
    }
    fn query(&self, sql: &str) -> Vec<String> {
        vec![format!("SQLite result for: {sql}")]
    }
}

// 외부 크레이트는 다음을 할 수 없습니다:
// impl private::Sealed for MyDriver {}  // 오류: Sealed는 비공개 모듈에 있음
// impl DatabaseDriver for MyDriver {}   // 오류: Sealed 바운드 누락
```

---

## 11. 고급 패턴

### 패턴: 타입 수준 숫자

```rust
trait TypeNum {
    const VALUE: usize;
}

struct Zero;
struct Succ<N>(std::marker::PhantomData<N>);

impl TypeNum for Zero {
    const VALUE: usize = 0;
}

impl<N: TypeNum> TypeNum for Succ<N> {
    const VALUE: usize = N::VALUE + 1;
}

type One = Succ<Zero>;
type Two = Succ<One>;
type Three = Succ<Two>;

fn print_type_num<N: TypeNum>() {
    println!("Type-level number: {}", N::VALUE);
}

fn main() {
    print_type_num::<Zero>();   // 0
    print_type_num::<One>();    // 1
    print_type_num::<Three>();  // 3
}
```

### 전략(Strategy) 패턴과 트레이트

```rust
trait SortStrategy {
    fn sort<T: Ord>(data: &mut [T]);
}

struct BubbleSort;
struct QuickSort;

impl SortStrategy for BubbleSort {
    fn sort<T: Ord>(data: &mut [T]) {
        let n = data.len();
        for i in 0..n {
            for j in 0..n - 1 - i {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                }
            }
        }
    }
}

impl SortStrategy for QuickSort {
    fn sort<T: Ord>(data: &mut [T]) {
        data.sort();  // 표준 라이브러리 퀵정렬 사용
    }
}

struct Sorter<S: SortStrategy> {
    _strategy: std::marker::PhantomData<S>,
}

impl<S: SortStrategy> Sorter<S> {
    fn new() -> Self {
        Sorter { _strategy: std::marker::PhantomData }
    }

    fn sort<T: Ord>(&self, data: &mut [T]) {
        S::sort(data);
    }
}

fn main() {
    let mut data = vec![5, 3, 8, 1, 9, 2];

    let sorter = Sorter::<BubbleSort>::new();
    sorter.sort(&mut data);
    println!("버블 정렬: {data:?}");

    let mut data = vec![5, 3, 8, 1, 9, 2];
    let sorter = Sorter::<QuickSort>::new();
    sorter.sort(&mut data);
    println!("퀵 정렬: {data:?}");
}
```

---

## 12. 연습문제

1. **플러그인 시스템**: 플러그인을 동적으로 로드(`Box<dyn Plugin>`)할 수 있는 트레이트 기반 플러그인 시스템을 설계하세요.

2. **GAT 컬렉션**: `LinkedList` 타입에 대해 `Collection` GAT 트레이트를 구현하세요.

3. **완전한 위임의 뉴타입**: 정렬된 순서를 유지하는 `SortedVec<T>` 뉴타입을 만드세요. `Deref`, `Display`, `IntoIterator`, `FromIterator`를 구현하세요.

4. **봉인된 트레이트 계층**: `encode`/`decode` 메서드가 있는 봉인된 `Codec` 트레이트를 설계하세요. `Json`, `Toml`, `Yaml` 타입에 대해 구현하세요.

5. **타입 상태 빌더**: `url`, `method`, `body`가 설정되어야만 `send()`를 호출할 수 있도록 타입 상태 패턴을 사용하는 `RequestBuilder`를 만드세요.

---

## 참고 자료

- [The Rust Reference: Trait Objects](https://doc.rust-lang.org/reference/types/trait-object.html)
- [Rust Blog: GATs Stabilization](https://blog.rust-lang.org/2022/10/28/gats-stabilization.html)
- [Coherence and Orphan Rules](https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules)

---

**이전**: [절차적 매크로](./05_Procedural_Macros.md) | **다음**: [고급 비동기](./07_Advanced_Async.md)

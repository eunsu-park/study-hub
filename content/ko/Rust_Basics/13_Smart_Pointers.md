# 13. 스마트 포인터(Smart Pointer)

**이전**: [클로저와 이터레이터](./12_Closures_and_Iterators.md) | **다음**: [동시성](./14_Concurrency.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 스마트 포인터(Smart Pointer)와 일반 참조의 차이를 설명하고, 힙 할당이 필요한 경우를 파악한다
- `Box<T>`를 힙 할당, 재귀적 데이터 구조, 트레이트 객체(trait object)에 활용한다
- `Deref`와 `Drop` 트레이트를 구현하여 포인터 동작과 리소스 정리를 커스터마이즈한다
- `Rc<T>`와 `RefCell<T>`를 적용하여 단일 스레드 환경에서 공유 소유권과 내부 가변성(interior mutability)을 모델링한다
- `Weak<T>`를 사용하여 참조 순환(reference cycle)을 감지하고 끊는다

## 목차

1. [스마트 포인터 vs 참조](#1-스마트-포인터-vs-참조)
2. [Box: 힙 할당](#2-boxt-힙-할당)
3. [Deref 트레이트](#3-deref-트레이트)
4. [Drop 트레이트](#4-drop-트레이트)
5. [Rc: 참조 카운팅](#5-rct-참조-카운팅)
6. [RefCell: 내부 가변성](#6-refcellt-내부-가변성)
7. [Rc RefCell 패턴](#7-rcrefcellt-패턴)
8. [Arc: 원자적 참조 카운팅](#8-arct-원자적-참조-카운팅)
9. [Weak: 참조 순환 끊기](#9-weakt-참조-순환-끊기)
10. [스마트 포인터 비교 표](#10-스마트-포인터-비교-표)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 스마트 포인터 vs 참조

**참조**(`&T` 또는 `&mut T`)는 데이터를 소유하지 않고 빌리는 경량 포인터다. **스마트 포인터(Smart Pointer)**는 포인터처럼 동작하는 구조체이지만, 가리키는 데이터를 **소유**하며 추가적인 기능(자동 정리, 참조 카운팅, 내부 가변성 등)을 제공한다.

차이를 이렇게 생각해도 좋다: 참조는 친구의 책을 빌리는 것 — 읽을 수 있지만 반납해야 한다. 스마트 포인터는 자신의 책을 사는 것 — 소유하며, 다 쓰면 처분 방식을 스스로 결정한다.

```
참조 (&T)                       스마트 포인터 (예: Box<T>)
──────────────                  ────────────────────────────
데이터를 빌림                   데이터를 소유
메타데이터 없음                 메타데이터를 가질 수 있음 (참조 카운트 등)
소멸자 없음                     정리를 위한 Drop 구현
항상 유효 (빌림 규칙)           자체 유효성을 관리
스택 전용 (포인터 자체)         데이터는 힙에 있음
```

Rust의 스마트 포인터는 두 가지 핵심 트레이트를 구현한다:
- **`Deref`** — 스마트 포인터가 참조처럼 동작하도록 함 (`*` 연산자 사용 가능)
- **`Drop`** — 스마트 포인터가 스코프를 벗어날 때 일어나는 일을 정의

---

## 2. Box<T>: 힙 할당

`Box<T>`는 가장 단순한 스마트 포인터다. 데이터를 **힙**에 할당하고 스택에 포인터를 저장한다. `Box`가 스코프를 벗어나면 힙 데이터와 포인터가 모두 해제된다.

```
스택                  힙
┌──────────┐          ┌──────────┐
│ Box<i32> │ ───────► │  42      │
│ (8 bytes)│          │ (4 bytes)│
└──────────┘          └──────────┘
```

### 기본 사용법

```rust
fn main() {
    // 정수를 힙에 할당
    let boxed = Box::new(42);
    println!("boxed = {boxed}"); // Deref 덕분에 투명하게 동작

    // Box는 크기를 알 수 없는 타입에 알려진 크기를 제공할 때 유용
    let boxed_slice: Box<[i32]> = vec![1, 2, 3].into_boxed_slice();
    println!("slice len = {}", boxed_slice.len());
}
```

### 재귀적 타입

`Box` 없이는 재귀적 타입의 크기가 무한해진다. `Box`는 알려진 포인터 크기로 간접 참조를 제공한다:

```rust
// 링크드 리스트: 각 노드는 값과 다음 노드를 가리키는 포인터를 가짐
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),  // Box가 고정 크기 포인터를 제공
    Nil,
}

use List::{Cons, Nil};

fn main() {
    // 리스트 구성: 1 -> 2 -> 3 -> Nil
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("{list:?}");
}
```

`Box` 없이는 컴파일러가 다음과 같이 오류를 낸다:

```
error: recursive type `List` has infinite size
  --> help: insert some indirection (e.g., a `Box`) to break the cycle
```

### 트레이트 객체

`Box<dyn Trait>`는 동적 디스패치(dynamic dispatch)를 가능하게 한다 — 공통 인터페이스 뒤에 서로 다른 구체적 타입을 저장할 수 있다:

```rust
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing circle with r={}", self.radius); }
}
impl Drawable for Square {
    fn draw(&self) { println!("Drawing square with s={}", self.side); }
}

fn main() {
    // 이질적인 컬렉션: 공통 트레이트 뒤에 서로 다른 타입
    let shapes: Vec<Box<dyn Drawable>> = vec![
        Box::new(Circle { radius: 3.0 }),
        Box::new(Square { side: 4.0 }),
        Box::new(Circle { radius: 1.5 }),
    ];

    for shape in &shapes {
        shape.draw(); // vtable을 통한 동적 디스패치
    }
}
```

---

## 3. Deref 트레이트

`Deref` 트레이트는 역참조 연산자 `*`의 동작을 커스터마이즈할 수 있게 한다. 이를 통해 **역참조 강제 변환(deref coercion)**이 가능해진다 — 컴파일러가 스마트 포인터의 참조를 내부 데이터의 참조로 자동 변환한다.

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(val: T) -> MyBox<T> {
        MyBox(val)
    }
}

// Deref를 구현하면 *MyBox<T>가 &T를 반환
impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0 // 내부 값에 대한 참조 반환
    }
}

fn greet(name: &str) {
    println!("Hello, {name}!");
}

fn main() {
    let boxed_name = MyBox::new(String::from("Rust"));

    // 역참조 강제 변환 체인:
    //   &MyBox<String> → &String → &str
    // 컴파일러가 이 변환들을 자동으로 삽입
    greet(&boxed_name);

    // 명시적 역참조 (강제 변환 덕분에 거의 필요하지 않음)
    let inner: &String = &*boxed_name;
    println!("inner = {inner}");
}
```

역참조 강제 변환의 규칙:
- `T: Deref<Target = U>`이면 `&T` → `&U` (불변)
- `T: DerefMut<Target = U>`이면 `&mut T` → `&mut U` (가변)
- `T: Deref<Target = U>`이면 `&mut T` → `&U` (가변에서 불변으로의 변환은 항상 안전)

---

## 4. Drop 트레이트

`Drop` 트레이트는 값이 스코프를 벗어날 때 실행되는 커스텀 정리 로직을 정의한다. 이것이 Rust의 소멸자(destructor)에 해당하며, 리소스 해제(파일 닫기, 메모리 해제, 락 해제)를 처리한다.

```rust
struct DatabaseConnection {
    name: String,
}

impl DatabaseConnection {
    fn new(name: &str) -> Self {
        println!("[{name}] Connection opened");
        DatabaseConnection { name: name.to_string() }
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        // 커넥션이 스코프를 벗어날 때 자동 정리
        println!("[{}] Connection closed (cleaned up)", self.name);
    }
}

fn main() {
    let _conn1 = DatabaseConnection::new("primary");
    let _conn2 = DatabaseConnection::new("replica");

    println!("Doing work...");

    // main()이 끝날 때 선언의 역순으로 Drop이 실행됨:
    // [replica] Connection closed (cleaned up)
    // [primary] Connection closed (cleaned up)
}
```

### `std::mem::drop`으로 조기 해제

`.drop()`을 직접 호출하는 것은 불가능하다(컴파일러가 이중 해제를 방지하기 위해 금지). 대신 `std::mem::drop()`을 사용한다:

```rust
fn main() {
    let conn = DatabaseConnection::new("temp");
    println!("Using connection...");

    drop(conn); // 명시적으로 조기 해제 — Drop::drop() 트리거
    // `conn`은 여기서부터 더 이상 유효하지 않음

    println!("Connection already cleaned up, continuing...");
}

// (위의 DatabaseConnection 정의)
struct DatabaseConnection { name: String }
impl DatabaseConnection {
    fn new(name: &str) -> Self {
        println!("[{name}] Connection opened");
        DatabaseConnection { name: name.to_string() }
    }
}
impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("[{}] Connection closed", self.name);
    }
}
```

---

## 5. Rc<T>: 참조 카운팅

때로는 하나의 값에 **여러 소유자**가 필요하다. 예를 들어 그래프에서 여러 엣지가 같은 노드를 가리킬 수 있다. `Rc<T>`(Reference Counted)는 데이터를 참조하는 `Rc` 포인터의 수를 추적하고, 카운트가 0이 되면 데이터를 해제한다.

```
   owner_a ──────►┌──────────────────┐
                  │ Rc<String>       │
   owner_b ──────►│ ref_count: 3     │
                  │ data: "shared"   │
   owner_c ──────►└──────────────────┘
```

```rust
use std::rc::Rc;

fn main() {
    let original = Rc::new(String::from("shared data"));
    println!("ref count after creation: {}", Rc::strong_count(&original)); // 1

    let clone_a = Rc::clone(&original); // 참조 카운트 증가, 딥 카피 아님
    println!("ref count after clone_a: {}", Rc::strong_count(&original)); // 2

    {
        let clone_b = Rc::clone(&original);
        println!("ref count with clone_b: {}", Rc::strong_count(&original)); // 3
    } // clone_b가 여기서 해제됨

    println!("ref count after clone_b dropped: {}", Rc::strong_count(&original)); // 2
    println!("data: {original}");
}
```

`Rc<T>`의 **중요한 제한사항**:
- **단일 스레드 전용** — 스레드 간에 안전하게 보낼 수 없음 (대신 `Arc<T>` 사용)
- **불변 데이터 전용** — `Rc<T>`는 공유 `&T` 참조만 제공하며 `&mut T`는 제공하지 않음
- `Rc`와 함께 가변성을 얻으려면 `RefCell<T>`와 조합한다 (아래 참고)

---

## 6. RefCell<T>: 내부 가변성

보통 Rust는 빌림 규칙을 **컴파일 타임**에 적용한다: `&mut T`가 하나이거나 `&T`가 여러 개이거나, 둘 다는 안 된다. `RefCell<T>`는 이 검사를 **런타임**으로 이동시켜, 불변 참조만 있어도 데이터를 변경할 수 있게 한다.

```
컴파일 타임 빌림 (&T / &mut T):      런타임 빌림 (RefCell<T>):
──────────────────────────────────    ──────────────────────────────────
오류가 컴파일 시점에 잡힘              오류가 런타임에 패닉 발생
런타임 비용 없음                       추적을 위한 소소한 런타임 오버헤드
제한적이지만 안전                      유연하지만 런타임 패닉 위험
```

```rust
use std::cell::RefCell;

fn main() {
    let data = RefCell::new(vec![1, 2, 3]);

    // borrow()는 Ref<T>를 반환 — 런타임에 추적되는 불변 빌림
    println!("data = {:?}", data.borrow());

    // borrow_mut()는 RefMut<T>를 반환 — 런타임에 추적되는 가변 빌림
    data.borrow_mut().push(4);
    println!("after push: {:?}", data.borrow());

    // 빌림 규칙 위반 시 런타임 패닉:
    // let r1 = data.borrow();
    // let r2 = data.borrow_mut(); // 패닉: 이미 불변으로 빌림
}
```

실용적인 사용 사례로 테스트의 **목 객체(mock object)** 패턴이 있다 — 불변 참조로 메서드 호출을 기록하고 싶을 때:

```rust
use std::cell::RefCell;

struct MockLogger {
    messages: RefCell<Vec<String>>,
}

impl MockLogger {
    fn new() -> Self {
        MockLogger { messages: RefCell::new(Vec::new()) }
    }

    // 주목: &self (불변)이지만 RefCell을 통해 메시지를 기록 가능
    fn log(&self, msg: &str) {
        self.messages.borrow_mut().push(msg.to_string());
    }

    fn get_messages(&self) -> Vec<String> {
        self.messages.borrow().clone()
    }
}

fn main() {
    let logger = MockLogger::new();

    // log()는 &self를 받지만 내부적으로 RefCell을 통해 변경
    logger.log("Starting process");
    logger.log("Process complete");

    println!("Logged: {:?}", logger.get_messages());
}
```

---

## 7. Rc<RefCell<T>> 패턴

`Rc`와 `RefCell`을 조합하면 **공유 가변 소유권**을 얻을 수 있다: 여러 소유자가 모두 내부 데이터를 변경할 수 있다.

```
   owner_a ──────►┌──────────────────────────┐
                  │ Rc<RefCell<Vec<i32>>>     │
   owner_b ──────►│ ref_count: 2             │
                  │ RefCell { data: [1,2,3] }│
                  └──────────────────────────┘
                  두 소유자 모두 Vec에 borrow_mut() 가능
```

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<RefCell<Node>> {
        Rc::new(RefCell::new(Node {
            value,
            children: Vec::new(),
        }))
    }
}

fn main() {
    let root = Node::new(1);
    let child_a = Node::new(2);
    let child_b = Node::new(3);

    // 루트에 자식 추가 — RefCell을 통해 가변으로 빌림
    root.borrow_mut().children.push(Rc::clone(&child_a));
    root.borrow_mut().children.push(Rc::clone(&child_b));

    // child_a는 공유됨: 루트가 소유하지만 직접 접근도 가능
    child_a.borrow_mut().value = 20;

    // 트리 출력
    let root_ref = root.borrow();
    println!("Root: {}", root_ref.value);
    for child in &root_ref.children {
        println!("  Child: {}", child.borrow().value);
    }
}
```

**이 패턴을 사용할 때**: 트리 구조, 그래프, 옵저버 패턴 — 여러 소유자와 변경이 모두 필요한 경우. 하지만 `Rc<RefCell<T>>`는 컴파일 타임 안전성을 런타임 검사와 맞바꾸므로, 가능하면 더 단순한 대안(`Vec`의 인덱스, `HashMap`)을 선호한다.

---

## 8. Arc<T>: 원자적 참조 카운팅

`Arc<T>`(Atomic Reference Counted)는 `Rc<T>`의 스레드 안전 버전이다. 원자적 연산으로 참조 카운트를 업데이트하여 스레드 간 안전한 공유가 가능하다.

```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);

    let mut handles = vec![];

    for i in 0..3 {
        let data_clone = Arc::clone(&data); // 원자적으로 참조 카운트 증가
        let handle = thread::spawn(move || {
            // 각 스레드는 동일한 데이터를 가리키는 자체 Arc를 가짐
            let sum: i32 = data_clone.iter().sum();
            println!("Thread {i}: sum = {sum}");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final ref count: {}", Arc::strong_count(&data)); // 1
}
```

`Arc` vs `Rc`:

| 기능 | `Rc<T>` | `Arc<T>` |
|------|---------|----------|
| 스레드 안전 | 아니오 (`!Send`, `!Sync`) | 예 |
| 성능 | 빠름 (원자적 연산 없음) | 약간의 오버헤드 (원자적 연산) |
| 사용 사례 | 단일 스레드 공유 | 멀티 스레드 공유 |

스레드 간 가변 공유 상태를 위해서는 `Mutex<T>`와 조합한다: `Arc<Mutex<T>>` (레슨 14에서 자세히 다룸).

---

## 9. Weak<T>: 참조 순환 끊기

`Rc`와 `Arc`는 **참조 순환(reference cycle)**을 만들 수 있다 — 두 값이 서로를 가리켜 참조 카운트가 절대 0에 도달하지 않는 상황이다. 이는 메모리 누수를 유발한다. `Weak<T>`는 강한 카운트를 증가시키지 않는 비소유 참조를 제공하여 이 문제를 해결한다.

```
강한 순환 (메모리 누수):             Weak 사용 (누수 없음):
┌────────┐     ┌────────┐          ┌────────┐      ┌────────┐
│ Node A │────►│ Node B │          │ Node A │─────►│ Node B │
│ rc: 2  │◄────│ rc: 2  │          │ rc: 1  │◄─ ─ ─│ rc: 1  │
└────────┘     └────────┘          └────────┘weak  └────────┘
rc가 절대 0에 도달하지 않음!         Weak 참조는 해제를 막지 않음
```

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Person {
    name: String,
    // 친한 친구 — Weak로 참조 순환 방지
    best_friend: RefCell<Option<Weak<Person>>>,
}

impl Person {
    fn new(name: &str) -> Rc<Person> {
        Rc::new(Person {
            name: name.to_string(),
            best_friend: RefCell::new(None),
        })
    }
}

impl Drop for Person {
    fn drop(&mut self) {
        println!("{} is being dropped", self.name);
    }
}

fn main() {
    let alice = Person::new("Alice");
    let bob = Person::new("Bob");

    // Weak 참조를 이용한 상호 친구 관계 설정
    *alice.best_friend.borrow_mut() = Some(Rc::downgrade(&bob));
    *bob.best_friend.borrow_mut() = Some(Rc::downgrade(&alice));

    // Weak::upgrade()로 접근 — Option<Rc<T>>를 반환
    if let Some(friend) = alice.best_friend.borrow().as_ref().unwrap().upgrade() {
        println!("Alice's best friend is {}", friend.name);
    }

    println!("Alice strong={}, weak={}", Rc::strong_count(&alice), Rc::weak_count(&alice));
    println!("Bob   strong={}, weak={}", Rc::strong_count(&bob), Rc::weak_count(&bob));

    // Alice와 Bob 모두 제대로 해제됨 — 메모리 누수 없음
}
```

**경험 법칙**: `Weak`는 "역방향 포인터"에 사용한다 — 트리의 부모 참조, 그래프의 역방향 관계, 항목이 할당 해제를 막으면 안 되는 캐시.

---

## 10. 스마트 포인터 비교 표

| 타입 | 소유권 | 가변성 | 스레드 안전 | 사용 사례 |
|------|--------|--------|-------------|-----------|
| `Box<T>` | 단일 소유자 | 소유자가 가변이면 가변 | 예 (`T`가 그렇다면 `Send + Sync`) | 힙 할당, 재귀적 타입, 트레이트 객체 |
| `Rc<T>` | 복수 소유자 | 불변 | 아니오 | 공유 소유권, 단일 스레드 |
| `Arc<T>` | 복수 소유자 | 불변 | 예 | 공유 소유권, 멀티 스레드 |
| `RefCell<T>` | 단일 소유자 | 내부 가변성 | 아니오 | 런타임 빌림 검사 |
| `Mutex<T>` | 단일 소유자 | 내부 가변성 | 예 | 스레드 안전 내부 가변성 |
| `Rc<RefCell<T>>` | 복수 + 가변 | 내부 가변성 | 아니오 | 공유 가변 상태, 단일 스레드 |
| `Arc<Mutex<T>>` | 복수 + 가변 | 내부 가변성 | 예 | 공유 가변 상태, 멀티 스레드 |
| `Weak<T>` | 비소유 | 불변 (upgrade를 통해) | 부모와 동일 | 참조 순환 끊기 |

결정 플로우차트:

```
힙 할당이 필요한가?
├── 예, 단일 소유자 → Box<T>
├── 예, 복수 소유자, 단일 스레드 → Rc<T>
│   └── 변경이 필요한가? → Rc<RefCell<T>>
├── 예, 복수 소유자, 멀티 스레드 → Arc<T>
│   └── 변경이 필요한가? → Arc<Mutex<T>>
└── 아니오 → 스택 할당 사용 (기본)
```

---

## 11. 연습 문제

### 문제 1: 수식 트리 평가기

`Box<T>`를 이용해 이진 수식 트리를 구축하라. `Num(f64)`, `Add(Box<Expr>, Box<Expr>)`, `Mul(Box<Expr>, Box<Expr>)`, `Neg(Box<Expr>)` 변형을 가진 `Expr` 열거형을 정의한다. 수식을 재귀적으로 평가하는 `eval()` 메서드를 구현한다. `(3 + 4) * -(2 + 5)`로 테스트한다.

### 문제 2: 참조 카운팅 문서 모델

`Rc<String>` 콘텐츠를 가진 `Document` 구조체를 만들어라. `Snapshot` 시스템을 구현한다: `snapshot()`을 호출하면 콘텐츠를 공유하는 경량 복사본(`Rc::clone`)이 반환된다. 문서가 수정되면 새로운 `String`을 할당한다(카피-온-라이트 의미론). 이전 스냅샷이 원래 콘텐츠를 유지하는지 확인한다.

### 문제 3: 이중 연결 리스트

`Rc`, `Weak`, `RefCell`을 사용하여 이중 연결 리스트를 구현하라. 각 노드는 `next: Option<Rc<RefCell<Node>>>`와 `prev: Option<Weak<RefCell<Node>>>`를 가져야 한다. `push_back`, `push_front`, `display_forward` 메서드를 구현한다. 모든 노드가 제대로 해제되는지(메모리 누수 없음) 확인한다.

### 문제 4: 로깅 기능이 있는 커스텀 스마트 포인터

모든 역참조와 해제를 로깅하는 `TrackedBox<T>` 스마트 포인터를 만들어라. `Deref`, `DerefMut`, `Drop`을 구현한다. 총 할당 및 해제 횟수를 추적하기 위해 `static AtomicUsize` 카운터를 사용한다. 모든 할당이 해제되었는지 요약을 출력한다.

### 문제 5: 옵저버 패턴

`Rc<RefCell<T>>`와 `Weak<RefCell<T>>`를 사용하여 옵저버 패턴을 구현하라. `Subject`는 옵저버에 대한 `Weak` 참조 목록을 유지한다. 옵저버는 스스로 등록하고 알림을 받는다. 옵저버를 해제해도 주제(Subject)가 정상적으로 동작하는 것을 보여준다(`Weak` 참조가 단순히 upgrade에 실패함).

---

## 12. 참고 자료

- [The Rust Programming Language, Ch. 15: Smart Pointers](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html)
- [Rust by Example: Box, Rc, Arc](https://doc.rust-lang.org/rust-by-example/std/box.html)
- [std::rc::Rc Documentation](https://doc.rust-lang.org/std/rc/struct.Rc.html)
- [std::cell::RefCell Documentation](https://doc.rust-lang.org/std/cell/struct.RefCell.html)
- [std::sync::Arc Documentation](https://doc.rust-lang.org/std/sync/struct.Arc.html)

---

**이전**: [클로저와 이터레이터](./12_Closures_and_Iterators.md) | **다음**: [동시성](./14_Concurrency.md)

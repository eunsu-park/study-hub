# 12. 클로저(Closure)와 이터레이터(Iterator)

**이전**: [라이프타임](./11_Lifetimes.md) | **다음**: [스마트 포인터](./13_Smart_Pointers.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 다양한 문법 형태로 클로저(Closure)를 정의하고, Rust가 매개변수와 반환 타입을 어떻게 추론하는지 설명한다
- `Fn`, `FnMut`, `FnOnce` 트레이트를 구별하고, 주어진 클로저가 어떤 트레이트를 구현하는지 예측한다
- `move` 클로저를 사용하여 캡처된 변수의 소유권을 클로저 내부로 이전한다
- 이터레이터 어댑터(`map`, `filter`, `zip` 등)를 체이닝하여 선언적 데이터 처리 파이프라인을 구축한다
- 커스텀 타입에 `Iterator` 트레이트를 구현하고, Rust 이터레이터가 왜 제로 비용 추상화(zero-cost abstraction)인지 설명한다

## 목차

1. [클로저란 무엇인가?](#1-클로저란-무엇인가)
2. [클로저 문법](#2-클로저-문법)
3. [타입 추론과 캡처 모드](#3-타입-추론과-캡처-모드)
4. [Fn, FnMut, FnOnce 트레이트](#4-fn-fnmut-fnonce-트레이트)
5. [move 클로저](#5-move-클로저)
6. [Iterator 트레이트](#6-iterator-트레이트)
7. [이터레이터 어댑터](#7-이터레이터-어댑터)
8. [소비 어댑터](#8-소비-어댑터)
9. [커스텀 이터레이터 만들기](#9-커스텀-이터레이터-만들기)
10. [성능: 이터레이터 vs 루프](#10-성능-이터레이터-vs-루프)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 클로저란 무엇인가?

**클로저(Closure)**는 주변 스코프의 변수를 캡처할 수 있는 익명 함수다. 클로저를 "배낭을 멘 함수"로 생각해도 좋다 — 필요한 외부 데이터를 함께 들고 다닌다.

많은 언어에서 클로저는 람다(lambda) 또는 익명 함수(anonymous function)라고 불린다. Rust의 클로저가 특별한 이유는, 컴파일러가 각 변수의 캡처 방식(불변 참조, 가변 참조, 값에 의한 소유권 이전)을 정적으로 결정하기 때문에 클로저가 직접 작성한 코드만큼 효율적이다.

```
일반 함수:          fn add(a: i32, b: i32) -> i32 { a + b }
                    ^^ 이름이 있으며, 캡처 없음

클로저:             |a, b| a + b
                    ^^ 익명이며, 스코프에서 캡처 가능
```

---

## 2. 클로저 문법

Rust는 간결한 형태부터 자세한 형태까지 여러 클로저 문법을 제공한다:

```rust
fn main() {
    // 1. 단일 표현식 — 중괄호 없음, 타입 어노테이션 없음
    let add = |a, b| a + b;
    println!("3 + 4 = {}", add(3, 4));

    // 2. 블록 본문 — 여러 구문이 있으면 중괄호 필요
    let greet = |name: &str| {
        let message = format!("Hello, {}!", name);
        println!("{}", message);
        message // 마지막 표현식이 반환값
    };
    greet("Rustacean");

    // 3. 완전한 어노테이션 — 매개변수와 반환 타입을 명시적으로 지정
    let multiply = |x: i32, y: i32| -> i32 { x * y };
    println!("5 * 6 = {}", multiply(5, 6));

    // 4. 매개변수 없음
    let say_hi = || println!("Hi!");
    say_hi();
}
```

일반 함수와 달리 클로저는 타입 어노테이션이 **필수가 아니다**. 컴파일러가 첫 번째 호출 시점에서 타입을 추론한다. 그러나 한번 추론된 타입은 고정된다:

```rust
fn main() {
    let identity = |x| x;

    let s = identity("hello"); // 추론됨: |&str| -> &str
    // let n = identity(42);   // 오류: &str이 와야 하는데 정수가 왔음
    println!("{}", s);
}
```

---

## 3. 타입 추론과 캡처 모드

클로저가 주변 스코프의 변수를 참조하면, Rust는 클로저 본문을 만족하는 **가장 제한이 적은** 캡처 모드를 선택한다:

```
캡처 모드          일어나는 일              구현되는 트레이트
─────────────────────────────────────────────────────────────
불변 참조          &T                        Fn
가변 참조          &mut T                    FnMut
소유권             T (값이 이동됨)           FnOnce
```

컴파일러는 항상 가장 가벼운 모드를 먼저 선택한다:

```rust
fn main() {
    let name = String::from("Alice");

    // 불변 참조로 캡처 — 클로저가 `name`을 읽기만 함
    let greet = || println!("Hello, {name}!");
    greet();
    greet(); // 여러 번 호출 가능
    println!("name is still accessible: {name}");

    let mut counter = 0;

    // 가변 참조로 캡처 — 클로저가 `counter`를 수정함
    let mut increment = || {
        counter += 1; // &mut counter가 필요
        println!("counter = {counter}");
    };
    increment();
    increment();
    // `increment`가 존재하는 동안 `counter`는 가변으로 빌려진 상태
    // println!("{counter}"); // `increment`의 마지막 사용 전에 주석 해제하면 오류

    let ticket = String::from("VIP-001");

    // 값으로 캡처 — 클로저가 `ticket`을 소비함
    let consume = || {
        let _moved = ticket; // 소유권을 가져감
        println!("Ticket consumed");
    };
    consume();
    // consume(); // 오류: 클로저가 FnOnce를 구현하며 이미 호출됨
    // println!("{ticket}"); // 오류: ticket이 이동됨
}
```

---

## 4. Fn, FnMut, FnOnce 트레이트

Rust의 모든 클로저는 세 가지 트레이트 중 하나 이상을 구현하며, 이 트레이트들은 계층 구조를 형성한다:

```
        FnOnce          ← 모든 클로저가 구현 (최소 한 번은 호출 가능)
          ▲
          │
        FnMut           ← 캡처를 소비하지 않는 클로저 (여러 번 호출 가능)
          ▲
          │
         Fn             ← 캡처를 변경하지 않는 클로저 (동시 호출 가능)
```

즉, 모든 `Fn`은 `FnMut`이기도 하며, 모든 `FnMut`은 `FnOnce`이기도 하다.

클로저를 매개변수로 받는 함수를 작성할 때는 동작하는 가장 관대한 트레이트를 선택한다:

```rust
// 변경 없이 여러 번 호출 가능한 모든 클로저를 받음
fn apply_twice<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(f(x))
}

// 캡처된 상태를 변경할 수 있는 클로저를 받음
fn call_n_times<F: FnMut()>(mut f: F, n: usize) {
    for _ in 0..n {
        f(); // 각 호출이 캡처된 변수를 변경할 수 있음
    }
}

// 캡처된 값을 소비할 수 있는 클로저를 받음 (정확히 한 번 호출)
fn call_once<F: FnOnce() -> String>(f: F) -> String {
    f() // 클로저가 캡처에서 값을 이동시킬 수 있음
}

fn main() {
    // Fn 예시: 순수 변환, 캡처 변경 없음
    let double = |x| x * 2;
    println!("apply_twice(double, 3) = {}", apply_twice(double, 3)); // 12

    // FnMut 예시: 클로저가 캡처된 카운터를 변경함
    let mut total = 0;
    call_n_times(|| { total += 1; }, 5);
    println!("total = {total}"); // 5

    // FnOnce 예시: 클로저가 String을 이동시킴
    let greeting = String::from("Hello, world!");
    let result = call_once(|| greeting); // `greeting`이 반환값으로 이동
    println!("{result}");
}
```

**가이드라인**: 가능하면 `Fn`을 사용하고, 클로저가 상태를 변경해야 하면 `FnMut`을 사용하며, 캡처를 소비하는 클로저에만 `FnOnce`를 사용한다.

---

## 5. move 클로저

기본적으로 클로저는 가장 가벼운 모드로 변수를 빌린다. `move` 키워드는 사용 방식에 관계없이 클로저가 캡처된 모든 변수의 **소유권**을 가져오도록 강제한다:

```rust
use std::thread;

fn main() {
    let name = String::from("Alice");

    // `move` 없이는 `name`을 빌리려 할 것이다.
    // 그러나 스레드는 현재 스코프보다 오래 살 수 있으므로, Rust는 소유권을 요구한다.
    let handle = thread::spawn(move || {
        // `name`은 이제 이 클로저가 소유 — 다른 스레드에서 안전하게 사용 가능
        println!("Hello from thread: {name}");
    });

    // println!("{name}"); // 오류: `name`이 클로저로 이동됨

    handle.join().unwrap();
}
```

정수처럼 `Copy` 트레이트를 구현하는 타입의 경우, `move`는 소유권 이전 대신 복사를 수행한다:

```rust
fn main() {
    let x = 42; // i32는 Copy를 구현

    let closure = move || println!("x = {x}");
    closure();

    // i32가 이동이 아닌 복사되었으므로 x는 여전히 사용 가능
    println!("x in main = {x}");
}
```

---

## 6. Iterator 트레이트

`Iterator` 트레이트는 `std::iter`에 정의되어 있으며, 단 하나의 메서드만 구현하면 된다:

```rust
trait Iterator {
    type Item;                        // 각 원소의 타입
    fn next(&mut self) -> Option<Self::Item>; // Some(item) 또는 None을 반환
}
```

`next()`를 호출할 때마다 이터레이터가 한 단계 진행된다. 시퀀스가 소진되면 `None`을 반환한다. 이터레이터를 책의 북마크로 생각하면 된다 — `next()`를 호출할 때마다 한 페이지씩 넘어간다.

```rust
fn main() {
    let numbers = vec![10, 20, 30];

    // `iter()`는 원소를 빌림 — &i32를 반환
    let mut iter = numbers.iter();
    assert_eq!(iter.next(), Some(&10));
    assert_eq!(iter.next(), Some(&20));
    assert_eq!(iter.next(), Some(&30));
    assert_eq!(iter.next(), None); // 소진됨

    // 컬렉션에서 이터레이터를 만드는 세 가지 방법:
    //   iter()      → 빌림 (&T)
    //   iter_mut()  → 가변 빌림 (&mut T)
    //   into_iter() → 소유권 이전 (T)

    // `for` 루프는 자동으로 `into_iter()`를 호출
    for num in &numbers {
        // num은 &i32
        print!("{num} ");
    }
    println!();
}
```

`size_hint()` 메서드는 남은 길이에 대한 힌트를 제공하며, `collect()` 같은 메서드가 메모리를 미리 할당하는 데 활용한다:

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    let iter = v.iter();
    let (lower, upper) = iter.size_hint();
    println!("At least {lower} elements, at most {upper:?} elements");
    // 출력: At least 5 elements, at most Some(5) elements
}
```

---

## 7. 이터레이터 어댑터

이터레이터 어댑터는 **지연 평가(lazy)** 방식으로 동작한다 — 소비 어댑터(consuming adaptor)가 실행을 주도하기 전까지는 아무 작업도 수행하지 않고, 원소를 필요할 때만 변환하는 새로운 이터레이터를 생성한다. 자유롭게 체이닝할 수 있다:

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map: 각 원소를 변환
    let squares: Vec<i32> = numbers.iter().map(|&x| x * x).collect();
    println!("squares: {squares:?}");

    // filter: 조건을 만족하는 원소만 유지
    let evens: Vec<&i32> = numbers.iter().filter(|&&x| x % 2 == 0).collect();
    println!("evens: {evens:?}");

    // enumerate: 인덱스 (i, 값) 쌍을 붙임
    for (i, val) in numbers.iter().enumerate() {
        if i < 3 {
            print!("[{i}]={val} ");
        }
    }
    println!();

    // zip: 두 이터레이터를 원소별로 쌍으로 묶음
    let names = vec!["Alice", "Bob", "Charlie"];
    let scores = vec![95, 87, 92];
    let results: Vec<_> = names.iter().zip(scores.iter()).collect();
    println!("results: {results:?}");

    // take와 skip: 이터레이터를 슬라이스
    let first_three: Vec<&i32> = numbers.iter().take(3).collect();
    let after_seven: Vec<&i32> = numbers.iter().skip(7).collect();
    println!("first 3: {first_three:?}, after 7: {after_seven:?}");

    // chain: 두 이터레이터를 연결
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let combined: Vec<&i32> = a.iter().chain(b.iter()).collect();
    println!("chained: {combined:?}");
}
```

**핵심 인사이트**: 어댑터는 지연 평가되므로, 이 코드는 파이프라인에 대한 설명만 구성한다. 실제 작업은 소비 어댑터(`collect` 등)가 이터레이션을 주도할 때 비로소 실행된다:

```
numbers.iter()        → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  .filter(|x| even)  → [   2,    4,    6,    8,    10 ]    ← 지연 평가
  .map(|x| x * x)    → [   4,   16,   36,   64,   100]    ← 지연 평가
  .collect()          → Vec[4, 16, 36, 64, 100]            ← 이터레이션 주도
```

---

## 8. 소비 어댑터

소비 어댑터(Consuming Adaptor)는 `next()`를 반복 호출하여 최종 결과를 생성한다:

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];

    // sum: 모든 원소를 더함
    let total: i32 = numbers.iter().sum();
    println!("sum = {total}"); // 15

    // fold: 누산기(accumulator)를 이용한 일반화된 리덕션
    let product = numbers.iter().fold(1, |acc, &x| acc * x);
    println!("product = {product}"); // 120

    // any: 조건을 만족하는 원소가 하나라도 있는가?
    let has_even = numbers.iter().any(|&x| x % 2 == 0);
    println!("has even? {has_even}"); // true

    // all: 모든 원소가 조건을 만족하는가?
    let all_positive = numbers.iter().all(|&x| x > 0);
    println!("all positive? {all_positive}"); // true

    // find: 조건을 만족하는 첫 번째 원소 (Option 반환)
    let first_even = numbers.iter().find(|&&x| x % 2 == 0);
    println!("first even = {first_even:?}"); // Some(2)

    // position: 첫 번째 매치의 인덱스 (Option<usize> 반환)
    let pos = numbers.iter().position(|&x| x == 3);
    println!("position of 3 = {pos:?}"); // Some(2)

    // 타입 터보피시와 함께 collect — 다른 컬렉션으로 수집
    let as_vec: Vec<i32> = (1..=5).collect();
    let as_string: String = vec!['R', 'u', 's', 't'].into_iter().collect();
    println!("{as_vec:?}, {as_string}");
}
```

어댑터와 소비자를 하나의 파이프라인으로 조합하는 강력한 패턴:

```rust
fn main() {
    let text = "hello world, hello rust, goodbye world";

    // 'h'로 시작하는 단어 수 세기
    let h_count = text.split_whitespace()
        .filter(|word| word.starts_with('h'))
        .count();
    println!("Words starting with 'h': {h_count}"); // 2

    // 길이가 4 초과인 단어를 대문자로 변환하여 쉼표로 연결
    let result: String = text.split_whitespace()
        .filter(|w| w.len() > 4)
        .map(|w| w.to_uppercase())
        .collect::<Vec<_>>()
        .join(", ");
    println!("{result}"); // HELLO, WORLD,, HELLO, RUST,, GOODBYE, WORLD
}
```

---

## 9. 커스텀 이터레이터 만들기

어떤 타입이든 이터러블(iterable)하게 만들려면 `Iterator` 트레이트를 구현하면 된다. 다음은 피보나치 수열을 생성하는 카운터 예시다:

```rust
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.a;
        let new_next = self.a + self.b;
        self.a = self.b;
        self.b = new_next;
        Some(current) // 무한 이터레이터 — None을 반환하지 않음
    }
}

fn main() {
    // 처음 10개의 피보나치 수 가져오기
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("First 10 Fibonacci: {fibs:?}");
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    // 100 미만의 피보나치 수의 합
    let sum: u64 = Fibonacci::new()
        .take_while(|&n| n < 100)
        .sum();
    println!("Sum of Fibs below 100: {sum}");

    // 다른 어댑터와 자연스럽게 조합
    let even_fibs: Vec<u64> = Fibonacci::new()
        .take(20)
        .filter(|n| n % 2 == 0)
        .collect();
    println!("Even Fibonacci (first 20): {even_fibs:?}");
}
```

`IntoIterator`를 구현하면 컬렉션 타입에서 `for` 루프를 사용할 수 있다:

```rust
struct Countdown {
    remaining: u32,
}

impl Countdown {
    fn from(start: u32) -> Self {
        Countdown { remaining: start }
    }
}

impl Iterator for Countdown {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None // 이터레이션 종료 신호
        } else {
            self.remaining -= 1;
            Some(self.remaining + 1) // 감소 전의 값 반환
        }
    }
}

fn main() {
    // for 루프에서 바로 사용 (Iterator는 자동으로 IntoIterator를 제공)
    for n in Countdown::from(5) {
        print!("{n}... ");
    }
    println!("Liftoff!");
    // 5... 4... 3... 2... 1... Liftoff!
}
```

---

## 10. 성능: 이터레이터 vs 루프

이터레이터 체인이 오버헤드를 추가하는지 걱정하는 경우가 있다. Rust의 답변은 **제로 비용 추상화(zero-cost abstraction)** — 컴파일러가 이터레이터 체인을 직접 작성한 루프와 동일한 기계어 코드로 최적화한다.

```rust
fn sum_of_squares_loop(numbers: &[i32]) -> i32 {
    let mut total = 0;
    for &n in numbers {
        if n % 2 == 0 {
            total += n * n;
        }
    }
    total
}

fn sum_of_squares_iter(numbers: &[i32]) -> i32 {
    numbers.iter()
        .filter(|&&n| n % 2 == 0)
        .map(|&n| n * n)
        .sum()
}

fn main() {
    let data: Vec<i32> = (1..=1000).collect();

    let result_loop = sum_of_squares_loop(&data);
    let result_iter = sum_of_squares_iter(&data);
    assert_eq!(result_loop, result_iter);

    println!("Sum of squares of even numbers 1..=1000: {result_loop}");
}
```

두 함수는 사실상 동일한 어셈블리로 컴파일된다. 이터레이터 버전이 자주 **선호**되는 이유:

- 인덱스 관련 오프-바이-원(off-by-one) 오류의 가능성을 없앤다
- 의도가 더 명확하다 (선언적 vs 명령적)
- 컴파일러가 SIMD 자동 벡터화를 더 안정적으로 적용할 수 있다

Rust 문서는 이터레이터를 "Rust의 제로 비용 추상화 중 하나"라고 부른다 — 더 높은 수준의 추상화를 사용하더라도 런타임 페널티가 없다.

---

## 11. 연습 문제

### 문제 1: 단어 빈도 카운터

`word_frequencies(text: &str) -> Vec<(String, usize)>` 함수를 작성하라. 텍스트를 소문자 단어로 분리하고, 각 단어의 등장 횟수를 세어 빈도 순(높은 것부터)으로 정렬된 결과를 반환한다. 이터레이터 어댑터와 `HashMap`을 활용한다.

### 문제 2: 커스텀 범위 이터레이터

`start`에서 `end`(미포함)까지 커스텀 `step` 크기로 순회하는 `StepRange` 구조체를 만들어라. `Iterator` 트레이트를 구현한다. 예를 들어, `StepRange::new(0, 20, 3)`은 `0, 3, 6, 9, 12, 15, 18`을 생성해야 한다.

### 문제 3: 클로저 기반 이벤트 시스템

핸들러 클로저를 등록하고 이벤트를 디스패치할 수 있는 간단한 이벤트 시스템을 설계하라. `on(event_name, callback)`과 `emit(event_name, data)` 메서드를 가진 `Dispatcher` 구조체를 만든다. 콜백에 어떤 `Fn` 트레이트를 사용해야 하는지 결정하고 그 이유를 설명한다.

### 문제 4: 병렬 파이프라인

이터레이터 어댑터와 소비자만을 사용하여, 단일 표현식으로 다음을 수행하라:
1. CSV 형식(`"name,score"`)의 `Vec<String>` 라인을 받는다
2. 각 라인을 `(String, u32)` 튜플로 파싱한다
3. 점수가 50 미만인 항목을 필터링한다
4. 남은 항목의 평균 점수를 계산한다

### 문제 5: 무한 이터레이터 조합

커스텀 무한 이터레이터 두 개를 만들어라: `Naturals` (1, 2, 3, ...)와 `Powers` (밑수를 받아 밑수^0, 밑수^1, 밑수^2, ...을 반환). 그런 다음 `zip`과 `take_while`로 조합하여 `2^n < 1_000_000`을 만족하는 모든 쌍 `(n, 2^n)`을 찾아라.

---

## 12. 참고 자료

- [The Rust Programming Language, Ch. 13: Functional Language Features](https://doc.rust-lang.org/book/ch13-00-functional-features.html)
- [Rust by Example: Closures](https://doc.rust-lang.org/rust-by-example/fn/closures.html)
- [Rust by Example: Iterators](https://doc.rust-lang.org/rust-by-example/trait/iter.html)
- [std::iter Module Documentation](https://doc.rust-lang.org/std/iter/index.html)
- [Iterator trait API Reference](https://doc.rust-lang.org/std/iter/trait.Iterator.html)

---

**이전**: [라이프타임](./11_Lifetimes.md) | **다음**: [스마트 포인터](./13_Smart_Pointers.md)

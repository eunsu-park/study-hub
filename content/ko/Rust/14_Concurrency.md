# 14. 동시성(Concurrency)

**이전**: [스마트 포인터](./13_Smart_Pointers.md) | **다음**: [비동기와 Await](./15_Async_Await.md)

**난이도**: ⭐⭐⭐⭐

## 학습 목표

- `thread::spawn`으로 스레드를 생성하고 `JoinHandle`을 사용하여 결과를 안전하게 수집한다
- `move` 클로저를 사용해 데이터를 스레드에 전달하고 `mpsc` 채널(Channel)로 스레드 간 통신한다
- `Mutex<T>`와 `Arc<Mutex<T>>`로 공유 가변 상태를 보호하고 `RwLock<T>`와의 트레이드오프를 설명한다
- `Send`와 `Sync` 마커 트레이트가 Rust의 컴파일 타임 스레드 안전성 보장을 어떻게 가능하게 하는지 설명한다
- 범위가 지정된 스레드(scoped thread)와 데이터 병렬 라이브러리를 적용하여 라이프타임 문제 없이 안전한 동시성 코드를 작성한다

## 목차

1. [동시성이 중요한 이유](#1-동시성이-중요한-이유)
2. [스레드 기초](#2-스레드-기초)
3. [스레드와 move 클로저](#3-스레드와-move-클로저)
4. [채널을 이용한 메시지 전달](#4-채널을-이용한-메시지-전달)
5. [다중 생산자](#5-다중-생산자)
6. [뮤텍스로 상태 공유](#6-뮤텍스로-상태-공유)
7. [RwLock vs Mutex](#7-rwlock-vs-mutex)
8. [Send와 Sync 트레이트](#8-send와-sync-트레이트)
9. [데드락 방지](#9-데드락-방지)
10. [범위 스레드](#10-범위-스레드)
11. [Rayon으로 데이터 병렬 처리](#11-rayon으로-데이터-병렬-처리)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. 동시성이 중요한 이유

현대 CPU는 여러 코어를 가지고 있으며, 하나의 코어만 사용하는 프로그램은 성능을 낭비하는 것이다. 동시성(Concurrency)을 사용하면 여러 작업을 동시에 실행할 수 있지만, 데이터 레이스(data race), 데드락(deadlock), 경쟁 조건(race condition)이라는 위험이 따른다.

대부분의 언어는 프로그래머가 런타임에 이러한 버그를 직접 방지해야 한다. Rust는 다른 접근 방식을 취한다: **타입 시스템**이 **컴파일 타임**에 데이터 레이스를 방지한다. Rust의 동시성 프로그램이 컴파일되면, 데이터 레이스가 없다는 것이 보장된다. 이것이 Rust의 "두려움 없는 동시성(fearless concurrency)" 보장이다.

```
단일 스레드:                  멀티 스레드:
┌──────────────┐             ┌──────────┐  ┌──────────┐
│ Task A       │             │ Task A   │  │ Task B   │
│ Task B       │             │          │  │          │
│ Task C       │             └──────────┘  └──────────┘
└──────────────┘             ┌──────────┐
총 시간: A + B + C            │ Task C   │
                             └──────────┘
                             총 시간: max(A+C, B) (겹치는 부분 있음)
```

---

## 2. 스레드 기초

`std::thread::spawn`은 새로운 OS 스레드를 생성한다. 스레드가 끝날 때까지 기다리고 반환값을 얻기 위해 `JoinHandle`을 사용한다:

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // spawn()은 클로저를 받아 새 스레드에서 실행
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("  spawned thread: count {i}");
            thread::sleep(Duration::from_millis(100));
        }
        42 // 스레드의 반환값
    });

    // 메인 스레드는 동시에 계속 실행
    for i in 1..=3 {
        println!("main thread: count {i}");
        thread::sleep(Duration::from_millis(150));
    }

    // join()은 생성된 스레드가 끝날 때까지 블록
    // Result<T, Box<dyn Any>>를 반환 — Ok(T) 또는 스레드가 패닉하면 Err
    let result = handle.join().unwrap();
    println!("Spawned thread returned: {result}");
}
```

```
실행 타임라인:
Main:     ──[1]────[2]────[3]────[join/wait]──►
Spawned:  ──[1]──[2]──[3]──[4]──[5]──────────►
                                    ▲
                                    └─ join 후 main이 재개
```

---

## 3. 스레드와 move 클로저

생성된 스레드는 생성한 스코프보다 오래 살 수 있다. 따라서 `thread::spawn`에 전달되는 클로저는 사용하는 모든 데이터를 소유해야 한다. `move` 키워드가 소유권을 이전한다:

```rust
use std::thread;

fn main() {
    let name = String::from("Alice");
    let age = 30; // Copy 타입 — 클로저로 복사됨

    let handle = thread::spawn(move || {
        // `name`은 이동됨 (String은 Copy를 구현하지 않음)
        // `age`는 복사됨 (i32는 Copy를 구현)
        println!("{name} is {age} years old");
    });

    // println!("{name}"); // 오류: `name`이 스레드로 이동됨
    println!("age is still accessible: {age}"); // OK: i32는 복사됨

    handle.join().unwrap();
}
```

`move` 없이는, 컴파일러가 빌린 데이터가 충분히 오래 살아있음을 보장할 수 없으므로 코드를 거부한다:

```
error[E0373]: closure may outlive the current function, but it borrows `name`
  --> help: to force the closure to take ownership, use the `move` keyword
```

---

## 4. 채널을 이용한 메시지 전달

채널(Channel)은 "통신으로 공유하라"는 철학을 구현한다. Rust는 `mpsc`(multiple producer, single consumer) 채널을 제공한다:

```
생산자 (tx)              채널               소비자 (rx)
┌──────────┐       ┌──────────────┐       ┌──────────┐
│ thread 1 │──tx──►│  [ A, B, C ] │──rx──►│  main    │
└──────────┘       └──────────────┘       └──────────┘
   send(A)            버퍼 큐               recv() → A
   send(B)                                  recv() → B
   send(C)                                  recv() → C
```

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // 채널 생성: tx (transmitter/sender), rx (receiver)
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let messages = vec![
            String::from("hello"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for msg in messages {
            tx.send(msg).unwrap(); // send()는 `msg`의 소유권을 이전
            thread::sleep(Duration::from_millis(200));
        }
        // tx가 여기서 해제됨 — 스트림 종료 신호
    });

    // recv()는 메시지가 도착하거나 채널이 닫힐 때까지 블록
    // rx를 이터레이터로 사용하면 자동으로 루프에서 recv()를 호출
    for received in rx {
        println!("Got: {received}");
    }
    // 모든 송신자가 해제되면 루프 종료
}
```

**중요**: `send()`는 소유권을 이전한다. 값을 보낸 후에는 송신자가 더 이상 사용할 수 없다. 이것이 설계상 데이터 레이스를 방지한다.

---

## 5. 다중 생산자

송신자를 클론하면 여러 스레드가 같은 채널로 메시지를 보낼 수 있다:

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    // 각 생산자 스레드를 위해 송신자를 클론
    for id in 0..3 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            for i in 0..3 {
                let msg = format!("Thread {id}, message {i}");
                tx_clone.send(msg).unwrap();
                thread::sleep(Duration::from_millis(100));
            }
        });
    }

    // 원래 송신자를 해제하여 rx가 모든 생산자가 끝났음을 알 수 있게 함
    drop(tx);

    // 모든 메시지 수집
    for msg in rx {
        println!("{msg}");
    }
    println!("All producers finished");
}
```

```
Thread 0 ──tx0──►┐
Thread 1 ──tx1──►├──[ channel ]──rx──► main
Thread 2 ──tx2──►┘
                  스레드별 전송 순서는 유지되지만
                  스레드 간에는 교차될 수 있음
```

---

## 6. 뮤텍스로 상태 공유

여러 스레드가 같은 데이터를 읽고 써야 할 때는 `Mutex`(뮤텍스, 상호 배제 잠금)를 사용한다. `Mutex<T>`는 호출자가 데이터에 접근하기 전에 `lock()`을 요구하여 내부 가변성을 제공한다:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Arc는 스레드 간 공유 소유권을 허용
    // Mutex는 안전한 가변 접근을 제공
    let counter = Arc::new(Mutex::new(0));

    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            // lock()은 MutexGuard를 반환 — 해제될 때 자동으로 잠금 해제
            let mut num = counter.lock().unwrap();
            *num += 1;
            // MutexGuard가 여기서 해제됨 → 잠금 해제
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final counter: {}", *counter.lock().unwrap()); // 10
}
```

```
Thread 1: ──[lock]──[*num += 1]──[unlock]──────────────────►
Thread 2: ──────────[wait...]──[lock]──[*num += 1]──[unlock]►
Thread 3: ──────────────────────────────[wait..]──[lock]──..►

Mutex는 한 번에 하나의 스레드만 데이터에 접근하도록 보장
```

**왜 `Rc`가 아닌 `Arc`인가?** `Rc`는 스레드 안전하지 않다(비원자적 참조 카운팅을 사용). `Arc`는 원자적 연산을 사용하며, 약간의 오버헤드가 있지만 스레드 간 정확성을 보장한다.

---

## 7. RwLock vs Mutex

`RwLock<T>`는 **다수의 동시 읽기** 또는 **하나의 쓰기**를 허용하여 읽기 빈번한 워크로드에서 처리량을 향상시킬 수 있다:

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let config = Arc::new(RwLock::new(String::from("initial")));

    let mut handles = vec![];

    // 5개의 읽기 스레드 생성
    for i in 0..5 {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            // read()는 여러 동시 독자를 허용
            let val = config.read().unwrap();
            println!("Reader {i}: {val}");
        }));
    }

    // 1개의 쓰기 스레드 생성
    {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            // write()는 독점적 접근이 필요
            let mut val = config.write().unwrap();
            *val = String::from("updated");
            println!("Writer: updated config");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final: {}", config.read().unwrap());
}
```

| 기능 | `Mutex<T>` | `RwLock<T>` |
|------|------------|-------------|
| 동시 읽기 | 아니오 | 예 |
| 쓰기 접근 | 독점적 | 독점적 |
| 성능 (읽기 빈번) | 낮음 (직렬화) | 높음 (병렬 읽기) |
| 성능 (쓰기 빈번) | 유사 | 약간 낮음 (오버헤드) |
| 쓰기 기아 위험 | 없음 | 가능 |

**가이드라인**: 기본적으로 `Mutex`를 사용한다. 프로파일링에서 직렬화된 읽기로 인한 경합이 확인될 때만 `RwLock`으로 전환한다.

---

## 8. Send와 Sync 트레이트

Rust는 두 가지 **마커 트레이트**(메서드가 없음)를 통해 스레드 안전성을 보장한다:

- **`Send`**: 타입을 다른 스레드로 **이전**할 수 있음 (소유권이 스레드 경계를 넘을 수 있음)
- **`Sync`**: 여러 스레드에서 타입을 **참조**할 수 있음 (`&T`가 안전하게 공유 가능)

```
Send: "나는 다른 스레드로 이동될 수 있다"
  - 대부분의 타입은 Send
  - Send 아님: Rc<T> (비원자적 참조 카운트), 원시 포인터

Sync: "여러 스레드가 동시에 &T를 가질 수 있다"
  - T가 Sync이면 &T가 Send
  - Sync 아님: RefCell<T> (런타임 빌림 검사는 스레드 안전하지 않음),
              Cell<T>, Rc<T>
```

```rust
use std::rc::Rc;
use std::sync::Arc;

fn must_be_send<T: Send>(_val: T) {}
fn must_be_sync<T: Sync>(_val: &T) {}

fn main() {
    let arc = Arc::new(42);
    must_be_send(arc.clone()); // Arc<i32>는 Send
    must_be_sync(&arc);        // Arc<i32>는 Sync

    // let rc = Rc::new(42);
    // must_be_send(rc);       // 오류: Rc<i32>는 Send가 아님
    // must_be_sync(&rc);      // 오류: Rc<i32>는 Sync가 아님

    println!("Arc is both Send and Sync!");
}
```

이 트레이트들은 컴파일러에 의해 자동으로 구현된다. 구조체의 모든 필드가 `Send`이면 그 구조체도 `Send`다. 이 조합 가능성 덕분에 Rust는 런타임 오버헤드 없이 컴파일 타임에 데이터 레이스를 방지할 수 있다.

---

## 9. 데드락 방지

**데드락(Deadlock)**은 둘 이상의 스레드가 각각 잠금을 보유하면서 서로의 잠금을 기다릴 때 발생하는 순환 의존성이다:

```
데드락:
Thread A: Lock 1 보유, Lock 2 대기
Thread B: Lock 2 보유, Lock 1 대기

Thread A ──[lock1]──────────[wait lock2...]──► 멈춤
Thread B ──────────[lock2]──[wait lock1...]──► 멈춤
```

방지 전략:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let resource_a = Arc::new(Mutex::new("Resource A"));
    let resource_b = Arc::new(Mutex::new("Resource B"));

    // 전략 1: 항상 동일한 전역 순서로 잠금 획득
    // 두 스레드 모두 A를 먼저 잠금, 그 다음 B — 순환 의존성 없음

    let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
    let t1 = thread::spawn(move || {
        let _a = ra.lock().unwrap(); // 항상 A 먼저 잠금
        let _b = rb.lock().unwrap(); // 그 다음 B 잠금
        println!("Thread 1: got both locks");
    });

    let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
    let t2 = thread::spawn(move || {
        let _a = ra.lock().unwrap(); // 동일한 순서: A 먼저
        let _b = rb.lock().unwrap(); // 그 다음 B
        println!("Thread 2: got both locks");
    });

    t1.join().unwrap();
    t2.join().unwrap();

    // 전략 2: try_lock()으로 블로킹 방지
    let ra = Arc::clone(&resource_a);
    let lock_result = ra.try_lock();
    match lock_result {
        Ok(guard) => println!("Got lock: {}", *guard),
        Err(_) => println!("Could not acquire lock, doing something else"),
    }

    // 전략 3: 잠금 범위 최소화 — 가능한 짧은 시간 동안만 잠금 유지
    {
        let data = resource_a.lock().unwrap();
        let result = data.len(); // 잠금 하에서 최소한의 작업만
        drop(data); // 더 많은 작업 전에 명시적으로 잠금 해제
        println!("Length was: {result}");
    }
}
```

---

## 10. 범위 스레드

`std::thread::scope`(Rust 1.63에서 안정화)는 스코프가 반환되기 전에 모든 스레드가 종료됨을 보장하기 때문에, 지역 변수를 빌릴 수 있는 스레드를 생성할 수 있다:

```rust
use std::thread;

fn main() {
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let chunk_size = data.len() / 2;

    // 범위 스레드는 `move` 없이 `data`를 빌릴 수 있음
    thread::scope(|s| {
        let (left, right) = data.split_at_mut(chunk_size);

        // 스레드가 `left`를 빌림 — 스코프가 끝나기 전에 종료 보장
        s.spawn(|| {
            for val in left.iter_mut() {
                *val *= 2;
            }
            println!("Left doubled: {left:?}");
        });

        // 스레드가 `right`를 빌림 — 슬라이스가 겹치지 않으므로 데이터 레이스 없음
        s.spawn(|| {
            for val in right.iter_mut() {
                *val *= 3;
            }
            println!("Right tripled: {right:?}");
        });
    }); // 모든 범위 스레드가 여기서 자동으로 join됨

    println!("Final data: {data:?}");
    // [2, 4, 6, 8, 15, 18, 21, 24]
}
```

```
스코프 경계:
┌──────────────────────────────────┐
│  s.spawn(|| { ... left ... });   │
│  s.spawn(|| { ... right ... });  │
│                                  │
│  ← 모든 스레드가 여기서 join됨   │
└──────────────────────────────────┘
스코프 이후 data에 다시 접근 가능
```

범위 스레드는 `Arc`와 채널의 오버헤드 없이 작업을 병렬화하고 싶을 때 이상적이다.

---

## 11. Rayon으로 데이터 병렬 처리

`rayon` 크레이트는 병렬 이터레이터를 통해 손쉬운 데이터 병렬 처리를 제공한다. 스레드 풀을 관리하고 자동으로 작업을 분배한다:

```rust
// Cargo.toml에 추가: rayon = "1.10"

use rayon::prelude::*;

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

fn main() {
    let numbers: Vec<u64> = (1..100_000).collect();

    // 순차 처리
    let count_seq = numbers.iter().filter(|&&n| is_prime(n)).count();

    // 병렬 처리 — iter()를 par_iter()로만 변경
    let count_par = numbers.par_iter().filter(|&&n| is_prime(n)).count();

    assert_eq!(count_seq, count_par);
    println!("Primes below 100,000: {count_par}");

    // 병렬 정렬
    let mut data = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    data.par_sort();
    println!("Sorted: {data:?}");

    // 병렬 맵-리듀스
    let sum_of_squares: u64 = (1..=1_000_000u64)
        .into_par_iter()
        .map(|n| n * n)
        .sum();
    println!("Sum of squares 1..=1M: {sum_of_squares}");
}
```

Rayon은 **작업 훔치기(work stealing)** 방식을 사용한다: 유휴 스레드가 바쁜 스레드의 큐에서 작업을 가져와 자동으로 부하를 분산한다. 대부분의 데이터 병렬 워크로드에서 Rayon은 수동 스레드 관리보다 간단하고 빠른 경우가 많다.

---

## 12. 연습 문제

### 문제 1: 병렬 단어 카운터

여러 텍스트 파일을 동시에 읽고(파일당 하나의 스레드), 각 파일의 단어 수를 세어 채널을 통해 메인 스레드로 결과를 보내는 프로그램을 작성하라. 파일당 단어 수와 총 합계를 출력한다.

### 문제 2: 생산자-소비자 파이프라인

채널을 사용하는 3단계 파이프라인을 구축하라:
1. **1단계**: 1부터 100까지 숫자 생성
2. **2단계**: 소수가 아닌 숫자 필터링
3. **3단계**: 결과 수집 및 출력

각 단계는 자체 스레드에서 실행된다. `mpsc::channel`로 각 단계를 연결한다.

```
[Generator] ──ch1──► [Filter] ──ch2──► [Collector]
```

### 문제 3: 스레드 안전 캐시

`Arc<RwLock<HashMap<K, V>>>`를 사용하여 동시성 캐시를 구현하라. `get()`(읽기 잠금)과 `insert()`(쓰기 잠금) 연산을 지원한다. 여러 읽기 및 쓰기 스레드를 생성하고 정확성을 검증한다. `Arc<Mutex<HashMap>>` 버전과 성능을 비교한다.

### 문제 4: 식사하는 철학자

5명의 철학자와 5개의 포크로 고전적인 식사하는 철학자 문제를 구현하라. 각 포크에 `Arc<Mutex<()>>`를 사용한다. 순서가 있는 잠금 획득으로 데드락 없는 솔루션을 구현한다. 각 철학자가 생각 중, 식사 중, 완료 상태일 때 출력한다.

### 문제 5: 병렬 행렬 곱셈

두 행렬(`Vec<Vec<f64>>`로 표현)의 곱을 범위 스레드를 사용하여 계산하라. 결과 행렬의 행을 사용 가능한 스레드에 나눈다. 큰 행렬(예: 500x500)에 대해 순차 구현과 실행 시간을 비교한다.

---

## 13. 참고 자료

- [The Rust Programming Language, Ch. 16: Fearless Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html)
- [std::thread Module Documentation](https://doc.rust-lang.org/std/thread/index.html)
- [std::sync Module Documentation](https://doc.rust-lang.org/std/sync/index.html)
- [Rayon: Data Parallelism in Rust](https://docs.rs/rayon/latest/rayon/)
- [Rust Atomics and Locks (Mara Bos)](https://marabos.nl/atomics/)

---

**이전**: [스마트 포인터](./13_Smart_Pointers.md) | **다음**: [비동기와 Await](./15_Async_Await.md)

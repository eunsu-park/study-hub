# 15. 비동기(Async)와 Await

**이전**: [동시성](./14_Concurrency.md) | **다음**: [모듈과 Cargo](./16_Modules_and_Cargo.md)

**난이도**: ⭐⭐⭐⭐

## 학습 목표

- 비동기(async) 프로그래밍과 OS 스레드를 비교하고, I/O 바운드 워크로드에서 async가 선호되는 이유를 설명한다
- `Future` 트레이트의 폴링(poll) 기반 실행 모델과 자기 참조적(self-referential) 퓨처에서 `Pin`의 역할을 설명한다
- `async fn` / `.await`으로 비동기 함수를 작성하고 Tokio 런타임에서 실행한다
- `tokio::select!`, 비동기 채널(async channel), 비동기 I/O를 사용하여 동시성 네트워크 애플리케이션을 구축한다
- `.await` 포인트 사이에서 잠금을 유지하거나 async 컨텍스트 내에서 블로킹하는 등의 일반적인 함정을 식별한다

## 목차

1. [Async vs 스레딩](#1-async-vs-스레딩)
2. [Future 트레이트](#2-future-트레이트)
3. [Async Fn과 Await 문법](#3-async-fn과-await-문법)
4. [Tokio 런타임](#4-tokio-런타임)
5. [select!로 퓨처 경합시키기](#5-select로-퓨처-경합시키기)
6. [비동기 채널](#6-비동기-채널)
7. [비동기 I/O](#7-비동기-io)
8. [비동기에서의 에러 처리](#8-비동기에서의-에러-처리)
9. [스트림: 비동기 이터레이터](#9-스트림-비동기-이터레이터)
10. [일반적인 함정](#10-일반적인-함정)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. Async vs 스레딩

OS 스레드와 async 태스크는 같은 문제(동시에 실행하기)를 서로 다른 트레이드오프로 해결한다:

```
OS 스레드:                               Async 태스크:
┌───────┐ ┌───────┐ ┌───────┐      ┌──────────────────────────────┐
│Thread1│ │Thread2│ │Thread3│      │        단일 스레드             │
│ stack │ │ stack │ │ stack │      │  ┌──┐  ┌──┐  ┌──┐  ┌──┐     │
│ ~8MB  │ │ ~8MB  │ │ ~8MB  │      │  │T1│  │T2│  │T1│  │T3│ ... │
└───────┘ └───────┘ └───────┘      │  └──┘  └──┘  └──┘  └──┘     │
OS가 스케줄링 관리                   └──────────────────────────────┘
스레드당 높은 메모리                  런타임이 스케줄링 관리
CPU 바운드 작업에 적합                태스크당 낮은 메모리 (~수 KB)
                                      I/O 바운드 작업에 탁월
```

| 측면 | OS 스레드 | Async 태스크 |
|------|-----------|-------------|
| 단위당 메모리 | ~8 MB 스택 | ~수 KB |
| 확장성 | 수백~수천 | 수십만 |
| 컨텍스트 전환 | OS 커널 (느림) | 사용자 공간 (빠름) |
| 최적 용도 | CPU 바운드 병렬성 | I/O 바운드 동시성 (네트워크, 디스크) |
| 선점 | 예 (OS가 중단 가능) | 아니오 (협력적, `.await`에서 양보) |

**경험 법칙**: 태스크가 대부분의 시간을 기다리는 데 쓴다면(네트워크 요청, 데이터베이스 쿼리, 파일 I/O) async를 사용한다. 대부분의 시간을 연산에 쓴다면 스레드를 사용한다.

---

## 2. Future 트레이트

Rust에서 비동기 연산은 `Future` 트레이트로 표현된다:

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

enum Poll<T> {
    Ready(T),   // 퓨처가 값 T로 완료됨
    Pending,    // 아직 준비되지 않음 — 준비되면 executor를 깨울 것
}
```

`Future`를 레시피 카드로 생각해도 좋다. 레시피는 생성할 때 실행되지 않는다. 대신 **executor**(Tokio 등)가 퓨처가 완료될 때까지 반복적으로 확인("poll")한다:

```
Executor가 퓨처를 폴링:

poll() → Pending  (준비 안 됨, I/O 진행 중)
  ... executor가 다른 작업 수행 ...
poll() → Pending  (여전히 대기 중)
  ... executor가 다른 작업 수행 ...
poll() → Ready(value)  (완료! 결과 반환)
```

### Pin이 필요한 이유

일부 퓨처는 자기 참조적 데이터를 포함한다(같은 구조체 내의 다른 필드를 가리키는 필드). 그런 구조체를 메모리에서 이동시키면 내부 포인터가 무효화된다. `Pin<&mut Self>`는 퓨처가 처음 폴링된 후 이동되지 않음을 보장하여 자기 참조를 안전하게 만든다.

`Pin`을 직접 사용할 일은 거의 없다 — `async fn`과 `.await`이 자동으로 처리한다. `Pin`이 폴링 기반 모델을 견고하게 만들기 위해 존재한다는 것만 알면 된다.

---

## 3. Async Fn과 Await 문법

`async fn`은 함수 본문을 `Future`를 구현하는 상태 머신(state machine)으로 변환하는 문법 설탕이다:

```rust
// 이 async 함수는...
async fn fetch_data(url: &str) -> String {
    format!("Data from {url}")
}

// ...대략 이것과 동일:
// fn fetch_data(url: &str) -> impl Future<Output = String> { ... }

// .await는 퓨처가 완료될 때까지 실행을 일시 중단
async fn process() {
    let data = fetch_data("https://example.com").await;
    println!("Got: {data}");
}

// async 블록은 익명 퓨처를 생성
async fn demo() {
    let future = async {
        let x = 1 + 2;
        x * 10
    };
    let result = future.await; // 퓨처를 완료까지 구동
    println!("result = {result}"); // 30
}
```

**핵심 포인트**: `async fn`을 호출하는 것은 실행하지 **않는다**. `Future`를 반환할 뿐이다. 퓨처는 폴링될 때만 실행되며, `.await`가 이를 트리거한다:

```rust
async fn greet() -> String {
    println!("Inside greet");
    String::from("Hello!")
}

async fn example() {
    let future = greet(); // 아직 아무것도 출력되지 않음 — 퓨처만 생성
    println!("Future created but not started");

    let result = future.await; // 이제 "Inside greet"가 출력됨
    println!("{result}");
}
```

---

## 4. Tokio 런타임

Rust의 표준 라이브러리에는 async 런타임이 포함되지 않는다. **Tokio**는 가장 널리 사용되는 런타임으로, 멀티 스레드 태스크 스케줄러, 비동기 I/O, 타이머, 채널을 제공한다.

```rust
// Cargo.toml에 추가:
// [dependencies]
// tokio = { version = "1", features = ["full"] }

use tokio::time::{sleep, Duration};

// #[tokio::main]은 런타임을 설정하고 async main 함수를 블록 실행
#[tokio::main]
async fn main() {
    println!("Starting...");

    // 동시성 태스크 생성 — async용 thread::spawn
    let task1 = tokio::spawn(async {
        sleep(Duration::from_millis(200)).await;
        println!("Task 1 complete");
        1
    });

    let task2 = tokio::spawn(async {
        sleep(Duration::from_millis(100)).await;
        println!("Task 2 complete");
        2
    });

    // 두 태스크를 await — 동시에 실행됨
    let (r1, r2) = (task1.await.unwrap(), task2.await.unwrap());
    println!("Results: {r1} + {r2} = {}", r1 + r2);
}
```

```
tokio::spawn을 사용한 타임라인:

Task 1: ──[spawn]──────────[sleep 200ms]──[done]──►
Task 2: ──[spawn]──[sleep 100ms]──[done]──────────►
Main:   ──[spawn tasks]──[await t1]──[await t2]──[print]──►

두 태스크가 Tokio 스레드 풀에서 동시에 실행됨.
Task 2가 먼저 끝나지만 main은 t1을 먼저 await함.
```

### Spawning vs Awaiting

- **`task.await`**: 퓨처를 순서대로 하나씩 실행
- **`tokio::spawn(task)`**: 스레드 풀에서 퓨처를 실행, 진정한 동시성

```rust
use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() {
    let start = Instant::now();

    // 순차 실행: 총 ~300ms
    sleep(Duration::from_millis(100)).await;
    sleep(Duration::from_millis(200)).await;
    println!("Sequential: {:?}", start.elapsed());

    let start = Instant::now();

    // tokio::join!을 사용한 동시 실행: 총 ~200ms (둘 중 최대값)
    let ((), ()) = tokio::join!(
        sleep(Duration::from_millis(100)),
        sleep(Duration::from_millis(200)),
    );
    println!("Concurrent: {:?}", start.elapsed());
}
```

---

## 5. select!로 퓨처 경합시키기

`tokio::select!`는 여러 퓨처를 기다리다가 **가장 먼저 완료되는** 브랜치를 실행한다. 나머지 퓨처는 드롭(취소)된다:

```rust
use tokio::time::{sleep, Duration};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let (tx, rx) = oneshot::channel::<String>();

    // 150ms 후 응답이 도착하는 시뮬레이션
    tokio::spawn(async move {
        sleep(Duration::from_millis(150)).await;
        let _ = tx.send(String::from("Response received"));
    });

    // 응답과 200ms 타임아웃을 경합
    tokio::select! {
        result = rx => {
            match result {
                Ok(msg) => println!("Got message: {msg}"),
                Err(_) => println!("Sender dropped"),
            }
        }
        _ = sleep(Duration::from_millis(200)) => {
            println!("Timeout! No response within 200ms");
        }
    }
    // 출력: "Got message: Response received" (150ms < 200ms)
}
```

```
select!는 먼저 완료되는 것을 선택:

rx future:      ──────[150ms]──[Ready("Response")]  ← 선택됨
timeout future: ──────[200ms]──[Ready(())]          ← 취소됨

결과: rx 브랜치가 실행됨
```

`select!`는 타임아웃, 취소, 정상 종료, I/O 소스 다중화를 구현하는 데 필수적이다.

---

## 6. 비동기 채널

Tokio는 보내거나 받을 때 `.await`하는 async 인식 채널을 제공한다:

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // 바운드 채널: 버퍼가 최대 32개의 메시지를 보유
    // 버퍼가 가득 차면 송신자가 블록(await) — 배압(backpressure)
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // 3개의 생산자 태스크 생성
    for id in 0..3 {
        let tx = tx.clone();
        tokio::spawn(async move {
            for i in 0..3 {
                let msg = format!("Producer {id}, msg {i}");
                // 버퍼가 가득 차면 send().await가 양보
                tx.send(msg).await.unwrap();
            }
        });
    }

    // 원래 송신자를 해제하여 수신자가 모든 생산자가 끝났음을 알 수 있게 함
    drop(tx);

    // 메시지가 도착할 때마다 소비
    while let Some(msg) = rx.recv().await {
        println!("Received: {msg}");
    }
    println!("All producers finished");
}
```

Tokio는 다음 채널도 제공한다:
- **`oneshot`**: 단일 값, 단일 송신자, 단일 수신자
- **`broadcast`**: 다수의 소비자가 각자 모든 메시지를 받음
- **`watch`**: 많은 수신자가 관찰할 수 있는 단일 값 (최신 값만 유지)

```rust
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(16);

    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    tx.send("Hello everyone!".to_string()).unwrap();

    // 두 수신자 모두 동일한 메시지를 받음
    println!("rx1: {}", rx1.recv().await.unwrap());
    println!("rx2: {}", rx2.recv().await.unwrap());
}
```

---

## 7. 비동기 I/O

Tokio는 표준 I/O 연산의 비동기 버전을 제공한다. 이 연산들은 OS를 기다리는 동안 제어권을 양보하여 다른 태스크가 실행될 수 있게 한다:

```rust
use tokio::fs;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> io::Result<()> {
    // 비동기 파일 쓰기
    let mut file = fs::File::create("/tmp/async_demo.txt").await?;
    file.write_all(b"Hello from async Rust!\n").await?;
    file.write_all(b"Second line\n").await?;
    // 모든 데이터가 디스크에 플러시되도록 보장
    file.flush().await?;

    // 비동기 파일 읽기 — 줄 단위
    let file = fs::File::open("/tmp/async_demo.txt").await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        println!("Read: {line}");
    }

    // 파일 전체를 한 번에 읽기
    let contents = fs::read_to_string("/tmp/async_demo.txt").await?;
    println!("Full contents:\n{contents}");

    // 정리
    fs::remove_file("/tmp/async_demo.txt").await?;

    Ok(())
}
```

### 비동기 TCP 서버 예시

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Server listening on 127.0.0.1:8080");

    loop {
        // 새 연결 수락
        let (mut socket, addr) = listener.accept().await?;
        println!("New connection from {addr}");

        // 각 연결마다 태스크 생성 — 동시 처리
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];

            loop {
                // 클라이언트로부터 데이터 읽기
                let n = match socket.read(&mut buf).await {
                    Ok(0) => return, // 연결 종료
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("Read error: {e}");
                        return;
                    }
                };

                // 데이터를 클라이언트에 에코
                if let Err(e) = socket.write_all(&buf[..n]).await {
                    eprintln!("Write error: {e}");
                    return;
                }
            }
        });
    }
}
```

```
TCP 에코 서버:

Client A ──[connect]──[send "Hi"]──[recv "Hi"]──►
Client B ──[connect]──[send "Hey"]──[recv "Hey"]──►
                          │
              ┌───────────┴───────────┐
              │ Tokio Runtime         │
              │ ┌─────┐  ┌─────┐     │
              │ │TaskA│  │TaskB│     │
              │ └─────┘  └─────┘     │
              │ 각 연결이 자체 스레드 │
              │ 에서 실행됨           │
              └───────────────────────┘
```

---

## 8. 비동기에서의 에러 처리

비동기 함수는 Rust의 `Result` 타입과 완벽하게 호환된다. 동기 코드와 마찬가지로 `?`로 오류를 전파한다:

```rust
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    Io(tokio::io::Error),
    Parse(ParseIntError),
    NotFound(String),
}

impl From<tokio::io::Error> for AppError {
    fn from(e: tokio::io::Error) -> Self { AppError::Io(e) }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self { AppError::Parse(e) }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O error: {e}"),
            AppError::Parse(e) => write!(f, "Parse error: {e}"),
            AppError::NotFound(key) => write!(f, "Not found: {key}"),
        }
    }
}

// 커스텀 에러를 반환하는 비동기 함수
async fn load_config(path: &str) -> Result<u32, AppError> {
    let contents = tokio::fs::read_to_string(path).await?; // ?가 io::Error를 변환
    let port: u32 = contents.trim().parse()?; // ?가 ParseIntError를 변환
    if port == 0 {
        return Err(AppError::NotFound("valid port".to_string()));
    }
    Ok(port)
}

#[tokio::main]
async fn main() {
    match load_config("/tmp/nonexistent.conf").await {
        Ok(port) => println!("Port: {port}"),
        Err(e) => println!("Error: {e}"),
    }
}
```

더 간단한 경우에는 `Box<dyn std::error::Error>` 또는 `anyhow` 크레이트를 사용한다:

```rust
// anyhow 사용 (Cargo.toml에 추가: anyhow = "1"):
// use anyhow::{Context, Result};
//
// async fn load_config(path: &str) -> Result<u32> {
//     let contents = tokio::fs::read_to_string(path).await
//         .context("failed to read config file")?;
//     let port: u32 = contents.trim().parse()
//         .context("config file does not contain a valid port number")?;
//     Ok(port)
// }
```

---

## 9. 스트림: 비동기 이터레이터

`Stream`은 `Iterator`의 비동기 버전이다. `Iterator::next()`가 `Option<T>`를 반환하는 반면, 스트림의 `next()`는 `Option<T>`로 resolve되는 퓨처를 반환한다:

```rust
// Cargo.toml에 추가:
// tokio-stream = "0.1"

use tokio_stream::StreamExt;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() {
    // 인터벌 타이머에서 스트림 생성
    let mut tick_stream = tokio_stream::wrappers::IntervalStream::new(
        interval(Duration::from_millis(200))
    );

    // 처음 5개의 틱 가져오기
    let mut count = 0;
    while let Some(instant) = tick_stream.next().await {
        count += 1;
        println!("Tick {count} at {instant:?}");
        if count >= 5 {
            break;
        }
    }

    // 스트림도 이터레이터와 유사한 어댑터 메서드를 지원
    let numbers = tokio_stream::iter(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let sum: i32 = numbers
        .filter(|&x| x % 2 == 0)     // 짝수만 유지
        .map(|x| x * x)              // 제곱
        .fold(0, |acc, x| acc + x)   // 합계
        .await;
    println!("Sum of squares of even numbers: {sum}"); // 4+16+36+64+100 = 220
}
```

```
Iterator vs Stream:

Iterator:   next() → Option<T>             (동기, 블로킹)
Stream:     next() → Future<Option<T>>      (비동기, 논블로킹)

Stream:  ──[poll]──Pending──[poll]──Ready(Some(1))──[poll]──Ready(Some(2))──...──Ready(None)
```

---

## 10. 일반적인 함정

### 함정 1: `.await` 포인트에서 뮤텍스 잠금 유지

```rust
use tokio::sync::Mutex;
use std::sync::Arc;

async fn bad_example(data: Arc<Mutex<Vec<i32>>>) {
    let mut guard = data.lock().await;
    guard.push(1);

    // 나쁜 예: await 포인트에서 MutexGuard를 유지
    // 태스크가 여기서 일시 중단될 수 있으며, 다른 태스크가
    // 잠금을 획득하려 할 때 잠재적 데드락 또는 기아 상태 유발
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    guard.push(2); // 위의 sleep 동안에도 잠금이 유지됨
}

async fn good_example(data: Arc<Mutex<Vec<i32>>>) {
    // 좋은 예: 잠금 범위를 제한
    {
        let mut guard = data.lock().await;
        guard.push(1);
    } // 여기서 잠금 해제, await 이전에

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    {
        let mut guard = data.lock().await;
        guard.push(2);
    } // 즉시 잠금 해제
}

#[tokio::main]
async fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    good_example(data).await;
}
```

### 함정 2: Async 컨텍스트에서 블로킹

동기 블로킹 호출(무거운 연산, 블로킹 I/O, `std::thread::sleep`)을 async 태스크 내에서 사용하면 전체 executor 스레드를 블록하여 다른 태스크를 굶긴다:

```rust
#[tokio::main]
async fn main() {
    // 나쁜 예: std::thread::sleep이 런타임 스레드를 블록
    // std::thread::sleep(std::time::Duration::from_secs(5));

    // 좋은 예: tokio의 비동기 sleep 사용
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // CPU 집약적인 작업은 블로킹 스레드 풀로 이동
    let result = tokio::task::spawn_blocking(|| {
        // 이것은 async executor가 아닌 전용 스레드에서 실행
        let mut sum: u64 = 0;
        for i in 0..10_000_000 {
            sum += i;
        }
        sum
    }).await.unwrap();

    println!("CPU result: {result}");
}
```

### 함정 3: 퓨처가 지연 평가됨을 잊는 것

```rust
async fn important_work() -> i32 {
    println!("Doing important work");
    42
}

#[tokio::main]
async fn main() {
    // 이것은 퓨처를 생성하지만 실행하지 않음
    let future = important_work(); // 아직 아무것도 출력되지 않음

    // 퓨처는 await되거나 spawn되어야 실행됨
    println!("Before await");
    let result = future.await; // 이제 "Doing important work"가 출력됨
    println!("Result: {result}");
}
```

### 함정 요약

```
함정                              해결책
───────────────────────────────────────────────────────────
.await에서 잠금 유지              잠금 범위를 지정, .await 전에 해제
async에서 블로킹 호출             spawn_blocking() 또는 비동기 대안 사용
퓨처의 지연 평가를 잊음           항상 .await 또는 tokio::spawn()
!Send 퓨처를 spawn                non-Send 퓨처에 tokio::task::LocalSet 사용
```

---

## 11. 연습 문제

### 문제 1: 동시성 HTTP 페처

`reqwest`(비동기 HTTP 클라이언트)와 `tokio`를 사용하여 URL 목록을 받아 동시에 페치하고(`tokio::spawn` 사용), 각 URL의 응답 상태 코드와 콘텐츠 길이를 출력하는 프로그램을 작성하라. `tokio::time::timeout`을 사용하여 요청당 5초 타임아웃을 추가한다.

### 문제 2: 채팅 서버

`tokio::net::TcpListener`를 사용하여 간단한 TCP 채팅 서버를 구축하라. 클라이언트가 연결하면 줄 단위로 메시지를 보낸다. 서버는 각 메시지를 다른 모든 연결된 클라이언트에 브로드캐스트한다. 메시지 배포에 `tokio::sync::broadcast`를 사용한다. 클라이언트 연결 해제를 정상적으로 처리한다.

```
Client A ──"Hi"──►┐
                  │ Server ──"[A]: Hi"──► Client B
Client B ──"Hey"──►│        ──"[B]: Hey"──► Client A
```

### 문제 3: 속도 제한기

초당 최대 N개의 요청을 허용하는 비동기 속도 제한기를 구현하라. `tokio::time::Interval`과 `tokio::sync::Semaphore`를 사용한다. 각각 "요청하기"를 원하는 50개의 태스크를 생성하고, 1초 내에 N개를 초과하여 실행되지 않는지 검증한다.

### 문제 4: 비동기 파일 처리 파이프라인

다음을 수행하는 파이프라인을 구축하라:
1. 비동기 채널에서 파일 이름을 읽는다
2. 각 파일의 내용을 비동기적으로 읽는다
3. 내용을 처리한다 (예: 줄 수 세기, 패턴 찾기)
4. 결과를 출력 파일에 쓴다

단계 간에 `tokio::sync::mpsc` 채널을 사용하고, 파일 작업에 `tokio::fs`를 사용한다. 설정 가능한 동시성 제한으로 여러 파일을 동시에 처리한다 (힌트: `tokio::sync::Semaphore` 사용).

### 문제 5: 정상 종료

섹션 7과 유사한 TCP 에코 서버를 작성하되, `Ctrl+C`를 정상적으로 처리하라. 신호를 받으면:
1. 새 연결 수락을 중단한다
2. 모든 활성 연결이 완료될 때까지 기다린다 (10초 타임아웃)
3. 요약을 출력한다 (처리된 총 연결 수, 전송된 바이트 수)

종료 로직을 구현하기 위해 `tokio::signal::ctrl_c()`와 `tokio::select!`를 사용한다.

---

## 12. 참고 자료

- [The Rust Async Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Tokio API Documentation](https://docs.rs/tokio/latest/tokio/)
- [tokio-stream Documentation](https://docs.rs/tokio-stream/latest/tokio_stream/)
- [Pin and Unpin Explained](https://doc.rust-lang.org/std/pin/index.html)
- [Asynchronous Programming in Rust (Jon Gjengset)](https://www.youtube.com/watch?v=ThjvMReOXYM)

---

**이전**: [동시성](./14_Concurrency.md) | **다음**: [모듈과 Cargo](./16_Modules_and_Cargo.md)

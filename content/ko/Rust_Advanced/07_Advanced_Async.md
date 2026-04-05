# 23. 고급 비동기

**이전**: [고급 트레이트](./06_Advanced_Traits.md) | **다음**: [FFI와 상호운용](./08_FFI_and_Interop.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Tokio 런타임 내부 구조(태스크 스케줄링, 작업 훔치기, 스레드 풀) 이해하기
2. `tokio::select!`로 동시 브랜치 실행 및 취소 안전성(Cancel Safety) 처리하기
3. 비동기 스트림(Async Stream)으로 비동기 값 시퀀스 처리하기
4. Tower 프레임워크로 미들웨어 스택 구축하기
5. 취소, 타임아웃, 우아한 셧다운(Graceful Shutdown) 처리하기

---

레슨 15에서 `async`/`await` 기초를 소개했습니다. 이 레슨은 프로덕션 비동기 Rust를 심층적으로 다룹니다: Tokio 런타임의 아키텍처, `select!`로 퓨처 합성, 스트림 처리, Tower 미들웨어 에코시스템, 그리고 취소 안전성이라는 중요한 주제.

## 목차
1. [Tokio 런타임 내부 구조](#1-tokio-런타임-내부-구조)
2. [태스크 스폰과 JoinHandle](#2-태스크-스폰과-joinhandle)
3. [tokio::select!](#3-tokioselect)
4. [취소 안전성](#4-취소-안전성)
5. [비동기 스트림](#5-비동기-스트림)
6. [비동기 코드의 채널](#6-비동기-코드의-채널)
7. [타임아웃과 데드라인](#7-타임아웃과-데드라인)
8. [우아한 셧다운](#8-우아한-셧다운)
9. [Tower 미들웨어](#9-tower-미들웨어)
10. [비동기 패턴](#10-비동기-패턴)
11. [성능 고려사항](#11-성능-고려사항)
12. [연습문제](#12-연습문제)

---

## 1. Tokio 런타임 내부 구조

Tokio는 Rust에서 가장 널리 사용되는 비동기 런타임입니다. 아키텍처를 이해하면 올바르고 성능 좋은 비동기 코드를 작성하는 데 도움이 됩니다.

### 런타임 종류

```rust
// 멀티 스레드 런타임 (기본) — 작업 훔치기(work-stealing) 스케줄러
#[tokio::main]
async fn main() {
    // 기본적으로 CPU 코어 수만큼의 스레드 사용
    println!("멀티 스레드 런타임에서 실행 중");
}

// 스레드 수 커스터마이즈
#[tokio::main(worker_threads = 4)]
async fn main() {
    println!("4개 워커 스레드에서 실행 중");
}

// 현재 스레드 런타임 — 단일 스레드, 협력적 스케줄링
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 모든 태스크가 하나의 스레드에서 실행 — 경량 앱에 적합
    println!("현재 스레드 런타임에서 실행 중");
}

// 수동 런타임 구성
fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("my-worker")
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        println!("커스텀 런타임에서 실행 중");
    });
}
```

### 작업 훔치기 스케줄러

멀티 스레드 런타임은 **작업 훔치기(work-stealing)** 스케줄러를 사용합니다:

```
스레드 1: [태스크 A] [태스크 C] [태스크 E]     ← 로컬 큐
스레드 2: [태스크 B] [태스크 D]              ← 로컬 큐
스레드 3: []                             ← 유휴, 스레드 1에서 훔침
스레드 4: [태스크 F]                       ← 로컬 큐

글로벌 주입 큐: [태스크 G, 태스크 H]    ← 새 태스크가 처음 여기 위치
```

- 각 워커 스레드는 **로컬 태스크 큐**(256슬롯 링 버퍼)를 가짐
- 새 태스크는 **글로벌 주입 큐**에 먼저 위치
- 유휴 스레드는 다른 스레드의 로컬 큐에서 **훔침**
- 이로써 모든 코어를 바쁘게 유지하면서 경합을 최소화

### 협력적 스케줄링

Tokio 태스크는 협력적으로 스케줄됩니다 — `.await` 포인트에서 제어를 양보해야 합니다:

```rust
#[tokio::main]
async fn main() {
    // 나쁜 예: 전체 스레드를 차단 — 다른 태스크가 실행될 수 없음
    tokio::spawn(async {
        loop {
            // 양보 없이 CPU 집약적 작업
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    // 좋은 예: CPU 바운드 작업에 tokio::task::yield_now() 사용
    tokio::spawn(async {
        for i in 0..1_000_000 {
            if i % 1000 == 0 {
                tokio::task::yield_now().await;  // 다른 태스크에 기회 양보
            }
            // ... 작업 ...
        }
    });

    // 최선: 진정한 CPU 바운드 작업에는 spawn_blocking 사용
    let result = tokio::task::spawn_blocking(|| {
        // 별도 스레드 풀에서 실행
        (0..1_000_000).sum::<u64>()
    }).await.unwrap();

    println!("결과: {result}");
}
```

---

## 2. 태스크 스폰과 JoinHandle

### 태스크 스폰

```rust
use tokio::task::JoinHandle;

#[tokio::main]
async fn main() {
    // spawn은 JoinHandle을 반환
    let handle: JoinHandle<String> = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        "태스크 완료".to_string()
    });

    // 결과 대기
    let result = handle.await.unwrap();
    println!("{result}");

    // 여러 태스크 스폰 후 결과 수집
    let mut handles = Vec::new();
    for i in 0..5 {
        handles.push(tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50 * i)).await;
            i * 10
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }
    println!("결과: {results:?}");  // [0, 10, 20, 30, 40]
}
```

### 태스크 그룹 관리를 위한 JoinSet

```rust
use tokio::task::JoinSet;

#[tokio::main]
async fn main() {
    let mut set = JoinSet::new();

    for i in 0..5 {
        set.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100 - i * 10)).await;
            format!("태스크 {i} 완료")
        });
    }

    // 완료 순서대로 결과 수집 (스폰 순서가 아님!)
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => println!("{msg}"),
            Err(e) => eprintln!("태스크 실패: {e}"),
        }
    }

    // 남은 태스크 모두 중단
    // set.abort_all();
}
```

---

## 3. tokio::select!

`select!`는 여러 퓨처를 동시에 대기하고 가장 먼저 완료되는 것에 반응합니다:

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    tokio::select! {
        _ = sleep(Duration::from_secs(3)) => {
            println!("3초 타이머 발동");
        }
        _ = sleep(Duration::from_secs(5)) => {
            println!("5초 타이머 발동");  // 도달하지 않음
        }
    }
    // 3초 브랜치만 실행; 5초 퓨처는 드롭됨
}
```

### 패턴 매칭을 사용한 select!

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx1, mut rx1) = mpsc::channel::<String>(32);
    let (tx2, mut rx2) = mpsc::channel::<i32>(32);

    // 생산자 시뮬레이션
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        tx1.send("hello".into()).await.unwrap();
    });
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        tx2.send(42).await.unwrap();
    });

    tokio::select! {
        Some(msg) = rx1.recv() => {
            println!("문자열 수신: {msg}");
        }
        Some(num) = rx2.recv() => {
            println!("숫자 수신: {num}");
        }
    }
}
```

### 루프 내 select!

```rust
use tokio::sync::mpsc;
use tokio::signal;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(32);

    // 생산자
    tokio::spawn(async move {
        for i in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if tx.send(format!("메시지 {i}")).await.is_err() {
                break;
            }
        }
    });

    let mut count = 0;
    loop {
        tokio::select! {
            Some(msg) = rx.recv() => {
                println!("수신: {msg}");
                count += 1;
            }
            _ = signal::ctrl_c() => {
                println!("\n{count}개 메시지 후 종료");
                break;
            }
            else => {
                println!("모든 채널 닫힘");
                break;
            }
        }
    }
}
```

---

## 4. 취소 안전성

`select!`에서 선택되지 않은 브랜치의 퓨처는 **드롭(취소)**됩니다. 퓨처가 작업 도중이었다면 상태가 불일치할 수 있습니다:

```rust
use tokio::sync::mpsc;

// 취소 안전하지 않음: recv() 완료 후 처리 전에 취소되면 메시지 손실
async fn process_messages(rx: &mut mpsc::Receiver<String>) {
    // recv()가 반환된 후 처리 완료 전에 이 퓨처가 드롭되면
    // 메시지는 소비되었지만 처리되지 않음!
    if let Some(msg) = rx.recv().await {
        println!("처리 중: {msg}");
        // ... 비용이 드는 작업 ...
    }
}

// 취소 안전한 대안
async fn process_messages_safe(
    rx: &mut mpsc::Receiver<String>,
    buffer: &mut Option<String>,
) {
    // 이전 취소에서 버퍼에 저장된 메시지가 있는지 확인
    let msg = if let Some(msg) = buffer.take() {
        msg
    } else {
        match rx.recv().await {
            Some(msg) => msg,
            None => return,
        }
    };

    // 메시지 처리
    println!("처리 중: {msg}");
    // 여기서 취소되어도 msg가 드롭되지만 괜찮음 —
    // 이미 의도적으로 채널에서 소비했음
}
```

### 취소 안전 연산

| 연산 | 취소 안전? | 비고 |
|------|-----------|------|
| `mpsc::Receiver::recv()` | 예 | 완료 전 취소되면 메시지가 채널에 남음 |
| `oneshot::Receiver::recv()` | 예 | 값이 채널에 남음 |
| `TcpStream::read()` | 아니오 | 부분 읽기가 손실될 수 있음 |
| `tokio::io::AsyncReadExt::read_exact()` | 아니오 | 부분 진행 손실 |
| `tokio::time::sleep()` | 예 | 손상될 상태 없음 |
| `JoinHandle::await` | 아니오 | 태스크는 계속 실행되지만 결과 손실 |

### 연산을 취소 안전하게 만들기

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct SafeReader {
    stream: TcpStream,
    buffer: Vec<u8>,
    bytes_read: usize,
    target_len: usize,
}

impl SafeReader {
    fn new(stream: TcpStream, target_len: usize) -> Self {
        Self {
            stream,
            buffer: vec![0u8; target_len],
            bytes_read: 0,
            target_len,
        }
    }

    /// 취소 안전 읽기: 진행 상황이 self에 저장되므로
    /// 취소해도 부분 데이터가 손실되지 않음
    async fn read_exact(&mut self) -> std::io::Result<&[u8]> {
        while self.bytes_read < self.target_len {
            let n = self.stream
                .read(&mut self.buffer[self.bytes_read..])
                .await?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "연결 종료",
                ));
            }
            self.bytes_read += n;
        }
        Ok(&self.buffer[..self.target_len])
    }
}
```

---

## 5. 비동기 스트림

비동기 스트림은 비동기 이터레이터와 같습니다 — 시간에 걸쳐 값의 시퀀스를 생성합니다:

```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    // 이터레이터로부터 스트림 생성
    let mut s = stream::iter(vec![1, 2, 3, 4, 5]);

    while let Some(value) = s.next().await {
        println!("수신: {value}");
    }

    // 스트림 컴비네이터 (이터레이터와 유사하지만 비동기)
    let doubled: Vec<_> = stream::iter(1..=5)
        .map(|x| x * 2)
        .collect()
        .await;
    println!("2배: {doubled:?}");

    // 필터와 take
    let result: Vec<_> = stream::iter(1..=100)
        .filter(|x| x % 7 == 0)
        .take(5)
        .collect()
        .await;
    println!("7의 배수 첫 5개: {result:?}");
}
```

### 커스텀 스트림 생성

```rust
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;

struct Counter {
    current: u64,
    max: u64,
}

impl Stream for Counter {
    type Item = u64;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.current < self.max {
            let val = self.current;
            self.current += 1;
            Poll::Ready(Some(val))
        } else {
            Poll::Ready(None)
        }
    }
}

// async_stream 크레이트로 더 쉬운 스트림 생성
use async_stream::stream;

fn countdown(from: u32) -> impl Stream<Item = u32> {
    stream! {
        for i in (0..=from).rev() {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            yield i;
        }
    }
}

#[tokio::main]
async fn main() {
    use tokio_stream::StreamExt;

    let mut s = countdown(5);
    while let Some(n) = s.next().await {
        println!("카운트다운: {n}");
    }
    println!("발사!");
}
```

### 스트림 동시성

```rust
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    // buffer_unordered로 스트림 항목을 동시에 처리
    let urls = vec![
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ];

    let results: Vec<_> = tokio_stream::iter(urls)
        .map(|url| async move {
            // HTTP 요청 시뮬레이션
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            format!("{url}에서 응답")
        })
        .buffer_unordered(3)  // 최대 3개 동시 처리
        .collect()
        .await;

    for r in results {
        println!("{r}");
    }
}
```

---

## 6. 비동기 코드의 채널

### mpsc — 다중 생산자, 단일 소비자

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    // 바운드 채널 — 가득 차면 백프레셔 적용
    let (tx, mut rx) = mpsc::channel::<String>(100);

    for i in 0..5 {
        let tx = tx.clone();
        tokio::spawn(async move {
            tx.send(format!("태스크 {i}의 메시지")).await.unwrap();
        });
    }

    // 원본 송신자를 드롭하여 수신자가 모든 송신자 완료를 알 수 있게 함
    drop(tx);

    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
}
```

### broadcast — 다중 생산자, 다중 소비자

```rust
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(16);

    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    tokio::spawn(async move {
        tx.send("안녕하세요!".into()).unwrap();
        tx.send("안녕히 계세요!".into()).unwrap();
    });

    tokio::spawn(async move {
        while let Ok(msg) = rx1.recv().await {
            println!("[구독자 1] {msg}");
        }
    });

    tokio::spawn(async move {
        while let Ok(msg) = rx2.recv().await {
            println!("[구독자 2] {msg}");
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}
```

### watch — 단일 생산자, 다중 소비자 (최신 값)

```rust
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = watch::channel("초기값".to_string());

    // 소비자 — 변경 감시
    let mut rx2 = rx.clone();
    tokio::spawn(async move {
        while rx2.changed().await.is_ok() {
            println!("[감시자] 값 변경: {}", *rx2.borrow());
        }
    });

    // 생산자 — 값 업데이트
    tx.send("업데이트 1".into()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    tx.send("업데이트 2".into()).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // 현재 값은 항상 사용 가능
    println!("현재: {}", *rx.borrow());
}
```

---

## 7. 타임아웃과 데드라인

```rust
use tokio::time::{timeout, Duration, Instant};

async fn slow_operation() -> String {
    tokio::time::sleep(Duration::from_secs(5)).await;
    "완료".into()
}

#[tokio::main]
async fn main() {
    // 간단한 타임아웃
    match timeout(Duration::from_secs(1), slow_operation()).await {
        Ok(result) => println!("수신: {result}"),
        Err(_) => println!("작업 타임아웃"),
    }

    // 데드라인 기반 타임아웃
    let deadline = Instant::now() + Duration::from_secs(2);
    match tokio::time::timeout_at(deadline, slow_operation()).await {
        Ok(result) => println!("수신: {result}"),
        Err(_) => println!("데드라인 초과"),
    }

    // 타임아웃을 사용한 재시도
    let result = retry_with_timeout(3, Duration::from_millis(500)).await;
    println!("재시도 결과: {result:?}");
}

async fn retry_with_timeout(
    max_retries: u32,
    per_attempt_timeout: Duration,
) -> Result<String, String> {
    for attempt in 1..=max_retries {
        match timeout(per_attempt_timeout, slow_operation()).await {
            Ok(result) => return Ok(result),
            Err(_) => {
                eprintln!("시도 {attempt}/{max_retries} 타임아웃");
            }
        }
    }
    Err("모든 시도 타임아웃".into())
}
```

---

## 8. 우아한 셧다운

```rust
use tokio::sync::{broadcast, mpsc};
use tokio::signal;

#[tokio::main]
async fn main() {
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let (done_tx, mut done_rx) = mpsc::channel::<()>(10);

    // 워커 태스크 스폰
    for id in 0..3 {
        let mut shutdown_rx = shutdown_tx.subscribe();
        let done_tx = done_tx.clone();

        tokio::spawn(async move {
            println!("[워커 {id}] 시작");

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        println!("[워커 {id}] 종료 중...");
                        // 정리 작업 수행
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        println!("[워커 {id}] 정리 완료");
                        drop(done_tx);  // 완료 신호
                        return;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                        println!("[워커 {id}] 작업 중...");
                    }
                }
            }
        });
    }

    // done_tx 복사본 드롭
    drop(done_tx);

    // 셧다운 신호 대기
    signal::ctrl_c().await.expect("Ctrl+C 리스너 설치 실패");
    println!("\nCtrl+C 수신, 셧다운 시작...");

    // 모든 워커에 셧다운 신호 전송
    let _ = shutdown_tx.send(());

    // 모든 워커가 완료될 때까지 대기
    let _ = done_rx.recv().await;
    println!("모든 워커 종료. 안녕히!");
}
```

---

## 9. Tower 미들웨어

Tower는 비동기 서비스를 위한 미들웨어 프레임워크입니다. axum, tonic, hyper 등 Rust 네트워킹 라이브러리에서 사용됩니다:

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

// 핵심 Tower 트레이트 (간략화)
trait Service<Request> {
    type Response;
    type Error;
    type Future: Future<Output = Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>>;
    fn call(&mut self, req: Request) -> Self::Future;
}

// 간단한 에코 서비스
struct EchoService;

impl Service<String> for EchoService {
    type Response = String;
    type Error = std::convert::Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<String, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: String) -> Self::Future {
        Box::pin(async move {
            Ok(format!("에코: {req}"))
        })
    }
}

// 타이밍 미들웨어 (Layer)
struct TimingLayer;

struct TimingService<S> {
    inner: S,
}

impl<S, Req> Service<Req> for TimingService<S>
where
    S: Service<Req>,
    S::Future: Send + 'static,
    S::Response: std::fmt::Debug + Send + 'static,
    S::Error: Send + 'static,
    Req: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let start = Instant::now();
        let future = self.inner.call(req);
        Box::pin(async move {
            let result = future.await;
            println!("요청 처리 시간: {:?}", start.elapsed());
            result
        })
    }
}
```

### Axum과 함께 Tower 사용 (실용적)

```rust
use axum::{
    Router,
    routing::get,
    middleware::{self, Next},
    extract::Request,
    response::Response,
};
use std::time::Instant;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    timeout::TimeoutLayer,
};

async fn timing_middleware(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let path = req.uri().path().to_string();
    let response = next.run(req).await;
    println!("{path} 처리 시간: {:?}", start.elapsed());
    response
}

async fn hello() -> &'static str {
    "안녕하세요!"
}

fn app() -> Router {
    Router::new()
        .route("/", get(hello))
        .layer(
            ServiceBuilder::new()
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .layer(middleware::from_fn(timing_middleware))
        )
}
```

---

## 10. 비동기 패턴

### 팬아웃 / 팬인

```rust
use tokio::task::JoinSet;

async fn fetch_url(url: &str) -> Result<String, String> {
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(format!("{url}의 콘텐츠"))
}

#[tokio::main]
async fn main() {
    let urls = vec![
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
    ];

    // 팬아웃: 동시 요청 스폰
    let mut set = JoinSet::new();
    for url in &urls {
        let url = url.to_string();
        set.spawn(async move { fetch_url(&url).await });
    }

    // 팬인: 결과 수집
    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok(content)) => results.push(content),
            Ok(Err(e)) => eprintln!("요청 오류: {e}"),
            Err(e) => eprintln!("태스크 패닉: {e}"),
        }
    }

    println!("{}개 결과 수집", results.len());
}
```

### 속도 제한

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let semaphore = Arc::new(Semaphore::new(3));  // 최대 3개 동시 작업
    let mut handles = Vec::new();

    for i in 0..10 {
        let sem = semaphore.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            println!("[{i}] 시작 (활성 퍼밋: {})", 3 - sem.available_permits());
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            println!("[{i}] 완료");
            // _permit이 여기서 드롭되어 세마포어 해제
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}
```

---

## 11. 성능 고려사항

### 런타임 블록 방지

```rust
// 나쁜 예: 전체 워커 스레드를 차단
async fn bad_hash(data: &[u8]) -> Vec<u8> {
    // CPU 집약적이며 비동기 런타임을 차단
    expensive_hash_function(data)
}

// 좋은 예: CPU 집약적 작업을 블로킹 스레드 풀로 이동
async fn good_hash(data: Vec<u8>) -> Vec<u8> {
    tokio::task::spawn_blocking(move || {
        expensive_hash_function(&data)
    }).await.unwrap()
}

fn expensive_hash_function(data: &[u8]) -> Vec<u8> {
    // CPU 집약적 작업 시뮬레이션
    std::thread::sleep(std::time::Duration::from_millis(100));
    data.to_vec()
}
```

### 태스크 크기와 할당

```rust
// 나쁜 예: 큰 퓨처 — 스폰 시 힙에 저장
async fn large_task() {
    let buf = [0u8; 1_000_000];  // 퓨처 스택에 1MB!
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    println!("버퍼 길이: {}", buf.len());
}

// 좋은 예: 대규모 데이터를 힙에 할당
async fn small_task() {
    let buf = vec![0u8; 1_000_000];  // 힙 할당, 퓨처는 작음
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    println!("버퍼 길이: {}", buf.len());
}
```

### 경합 감소

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

// 읽기가 많은 워크로드에는 Mutex 대신 RwLock 사용
struct Cache {
    data: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl Cache {
    async fn get(&self, key: &str) -> Option<String> {
        let data = self.data.read().await;  // 다수 읽기 허용
        data.get(key).cloned()
    }

    async fn set(&self, key: String, value: String) {
        let mut data = self.data.write().await;  // 독점 접근
        data.insert(key, value);
    }
}
```

---

## 12. 연습문제

1. **팬아웃 가져오기**: URL 목록을 받아 구성 가능한 동시성 제한(세마포어 사용)으로 동시에 가져오는 비동기 함수를 작성하세요. 결과를 원래 순서대로 반환하세요.

2. **비동기 파이프라인**: 스트림 처리 파이프라인을 구축하세요: 숫자 생성 → 짝수 필터 → 제곱 매핑 → 10개씩 청크로 버퍼링 → 청크 출력. `tokio_stream` 사용.

3. **우아한 셧다운 서버**: Ctrl+C를 처리하는 간단한 TCP 에코 서버를 작성하세요: (1) 새 연결 중단, (2) 기존 연결이 완료될 때까지 대기(5초 데드라인), (3) 남은 연결 강제 종료.

4. **속도 제한 미들웨어**: 토큰 버킷 알고리즘을 사용하여 클라이언트 IP당 초당 N개 요청을 허용하는 Tower 호환 속도 제한 레이어를 구현하세요.

5. **취소 안전 상태 머신**: 채널에서 메시지를 읽고 처리된 결과를 파일에 쓰는 취소 안전 메시지 프로세서를 구현하세요. 작업이 중간에 취소되어도 메시지가 손실되지 않도록 하세요.

---

## 참고 자료

- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Tokio: select!](https://tokio.rs/tokio/tutorial/select)
- [Tower documentation](https://docs.rs/tower/latest/tower/)
- [async-stream crate](https://docs.rs/async-stream/latest/async_stream/)
- [Alice Ryhl: Actors with Tokio](https://ryhl.io/blog/actors-with-tokio/)
- [Jon Gjengset: Decrusting Tokio](https://www.youtube.com/watch?v=o2ob8zkeq2s)

---

**이전**: [고급 트레이트](./06_Advanced_Traits.md) | **다음**: [FFI와 상호운용](./08_FFI_and_Interop.md)

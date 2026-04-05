# 27. 네트워크 프로그래밍

**이전**: [임베디드 Rust](./10_Embedded_Rust.md) | **다음**: [고급 에러 처리](./12_Advanced_Error_Handling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Tokio의 비동기 네트워킹으로 TCP/UDP 서버 및 클라이언트 빌드하기
2. Axum 프레임워크를 사용하여 HTTP 서버와 API 생성하기
3. 실시간 애플리케이션을 위한 WebSocket 통신 구현하기
4. `rustls`를 사용하여 보안 연결을 위한 TLS 구성하기
5. 커넥션 풀링, 백프레셔, 우아한 셧다운 등 프로덕션 네트워킹 패턴 적용하기

---

Rust의 비동기 에코시스템은 네트워크 프로그래밍을 위한 강력한 기본 요소를 제공합니다. Tokio가 I/O 이벤트 루프를 처리하고, Axum과 Hyper 같은 프레임워크가 그 위에 HTTP를 구축합니다. 이 레슨은 원시 TCP 소켓부터 프로덕션 HTTP API까지 네트워킹을 다룹니다.

## 목차
1. [Tokio로 TCP](#1-tokio로-tcp)
2. [Tokio로 UDP](#2-tokio로-udp)
3. [Axum으로 HTTP](#3-axum으로-http)
4. [요청 라우팅과 추출기](#4-요청-라우팅과-추출기)
5. [미들웨어와 레이어](#5-미들웨어와-레이어)
6. [JSON API](#6-json-api)
7. [WebSocket](#7-websocket)
8. [rustls로 TLS](#8-rustls로-tls)
9. [reqwest로 HTTP 클라이언트](#9-reqwest로-http-클라이언트)
10. [프로덕션 패턴](#10-프로덕션-패턴)
11. [커넥션 풀링과 데이터베이스](#11-커넥션-풀링과-데이터베이스)
12. [연습문제](#12-연습문제)

---

## 1. Tokio로 TCP

### 에코 서버

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("TCP 서버 :8080에서 리스닝");

    loop {
        let (mut socket, addr) = listener.accept().await?;
        println!("{addr}에서 새 연결");

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];

            loop {
                let n = match socket.read(&mut buf).await {
                    Ok(0) => {
                        println!("{addr} 연결 종료");
                        return;
                    }
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("{addr}에서 읽기 오류: {e}");
                        return;
                    }
                };

                if let Err(e) = socket.write_all(&buf[..n]).await {
                    eprintln!("{addr}에 쓰기 오류: {e}");
                    return;
                }
            }
        });
    }
}
```

### 코덱을 이용한 라인 기반 프로토콜

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (socket, addr) = listener.accept().await?;
        let (reader, mut writer) = socket.into_split();
        let mut lines = BufReader::new(reader).lines();

        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                let response = match line.trim() {
                    "PING" => "PONG\n".to_string(),
                    "TIME" => format!("{:?}\n", std::time::SystemTime::now()),
                    "QUIT" => {
                        let _ = writer.write_all(b"BYE\n").await;
                        return;
                    }
                    cmd => format!("UNKNOWN: {cmd}\n"),
                };

                if writer.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
            }
            println!("{addr} 연결 종료");
        });
    }
}
```

### TCP 클라이언트

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect("127.0.0.1:8080").await?;
    println!("서버에 연결됨");

    stream.write_all(b"Hello, server!\n").await?;

    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;
    println!("응답: {}", String::from_utf8_lossy(&buf[..n]));

    Ok(())
}
```

---

## 2. Tokio로 UDP

```rust
use tokio::net::UdpSocket;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 서버
    let server = UdpSocket::bind("127.0.0.1:9090").await?;
    println!("UDP 서버 :9090에서 실행 중");

    tokio::spawn(async move {
        let mut buf = [0u8; 1024];
        loop {
            let (len, addr) = server.recv_from(&mut buf).await.unwrap();
            let msg = String::from_utf8_lossy(&buf[..len]);
            println!("{addr}에서 받은 메시지: {msg}");

            let response = format!("에코: {msg}");
            server.send_to(response.as_bytes(), addr).await.unwrap();
        }
    });

    // 클라이언트
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let client = UdpSocket::bind("127.0.0.1:0").await?;  // 랜덤 포트
    client.send_to(b"Hello UDP!", "127.0.0.1:9090").await?;

    let mut buf = [0u8; 1024];
    let (len, _) = client.recv_from(&mut buf).await?;
    println!("클라이언트 수신: {}", String::from_utf8_lossy(&buf[..len]));

    Ok(())
}
```

---

## 3. Axum으로 HTTP

Axum은 Tokio와 Tower 위에 구축된 권장 Rust HTTP 프레임워크입니다:

```toml
[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower-http = { version = "0.6", features = ["cors", "compression-full", "trace"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

### 최소 서버

```rust
use axum::{routing::get, Router};

async fn hello() -> &'static str {
    "Hello, World!"
}

async fn health() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(hello))
        .route("/health", get(health));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("서버 실행 중: http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
```

---

## 4. 요청 라우팅과 추출기

Axum은 추출기(extractor)를 사용하여 요청 데이터를 파싱합니다:

```rust
use axum::{
    extract::{Path, Query, State, Json},
    routing::{get, post, put, delete},
    Router,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

// 애플리케이션 상태
type AppState = Arc<RwLock<HashMap<u64, User>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct CreateUser {
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct ListParams {
    page: Option<u64>,
    limit: Option<u64>,
}

// 경로 파라미터 추출
async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<User>, StatusCode> {
    let users = state.read().await;
    users.get(&id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

// 쿼리 파라미터 추출
async fn list_users(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> Json<Vec<User>> {
    let page = params.page.unwrap_or(1);
    let limit = params.limit.unwrap_or(10);

    let users = state.read().await;
    let result: Vec<User> = users.values()
        .skip(((page - 1) * limit) as usize)
        .take(limit as usize)
        .cloned()
        .collect();

    Json(result)
}

// JSON 본문 추출
async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUser>,
) -> (StatusCode, Json<User>) {
    let mut users = state.write().await;
    let id = users.len() as u64 + 1;

    let user = User {
        id,
        name: payload.name,
        email: payload.email,
    };

    users.insert(id, user.clone());
    (StatusCode::CREATED, Json(user))
}

// 삭제 핸들러
async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> StatusCode {
    let mut users = state.write().await;
    if users.remove(&id).is_some() {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(RwLock::new(HashMap::new()));

    let app = Router::new()
        .route("/users", get(list_users).post(create_user))
        .route("/users/{id}", get(get_user).delete(delete_user))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 5. 미들웨어와 레이어

```rust
use axum::{
    Router,
    routing::get,
    middleware::{self, Next},
    extract::Request,
    response::Response,
    http::{HeaderMap, StatusCode},
};
use std::time::Instant;
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    trace::TraceLayer,
};

// 커스텀 미들웨어: 요청 타이밍
async fn timing_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;

    let elapsed = start.elapsed();
    println!("{method} {uri} — {:?} — {}", elapsed, response.status());

    response
}

// 커스텀 미들웨어: API 키 인증
async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let api_key = headers
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some("secret-key-123") => Ok(next.run(request).await),
        Some(_) => Err(StatusCode::UNAUTHORIZED),
        None => Err(StatusCode::UNAUTHORIZED),
    }
}

async fn public_handler() -> &'static str {
    "공개 엔드포인트"
}

async fn protected_handler() -> &'static str {
    "보호된 엔드포인트"
}

#[tokio::main]
async fn main() {
    let public_routes = Router::new()
        .route("/public", get(public_handler));

    let protected_routes = Router::new()
        .route("/protected", get(protected_handler))
        .layer(middleware::from_fn(auth_middleware));

    let app = Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn(timing_middleware))
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 6. JSON API

### 구조화된 에러 응답

```rust
use axum::{
    response::{IntoResponse, Response},
    http::StatusCode,
    Json,
};
use serde::Serialize;

#[derive(Serialize)]
struct ApiError {
    error: String,
    code: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.code)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

impl ApiError {
    fn not_found(msg: impl Into<String>) -> Self {
        ApiError {
            error: msg.into(),
            code: 404,
            details: None,
        }
    }

    fn bad_request(msg: impl Into<String>, details: impl Into<String>) -> Self {
        ApiError {
            error: msg.into(),
            code: 400,
            details: Some(details.into()),
        }
    }

    fn internal(msg: impl Into<String>) -> Self {
        ApiError {
            error: msg.into(),
            code: 500,
            details: None,
        }
    }
}

// Result를 반환하는 핸들러
async fn get_user(Path(id): Path<u64>) -> Result<Json<User>, ApiError> {
    if id == 0 {
        return Err(ApiError::bad_request(
            "잘못된 ID",
            "사용자 ID는 0보다 커야 합니다",
        ));
    }

    // 조회 시뮬레이션
    if id > 100 {
        return Err(ApiError::not_found(format!("사용자 {id}를 찾을 수 없음")));
    }

    Ok(Json(User {
        id,
        name: "Alice".into(),
        email: "alice@example.com".into(),
    }))
}

use axum::extract::Path;
use serde::Deserialize;

#[derive(Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}
```

### 페이지네이션

```rust
use serde::{Serialize, Deserialize};

#[derive(Deserialize)]
struct Pagination {
    #[serde(default = "default_page")]
    page: u64,
    #[serde(default = "default_per_page")]
    per_page: u64,
}

fn default_page() -> u64 { 1 }
fn default_per_page() -> u64 { 20 }

#[derive(Serialize)]
struct PaginatedResponse<T: Serialize> {
    data: Vec<T>,
    page: u64,
    per_page: u64,
    total: u64,
    total_pages: u64,
}

impl<T: Serialize> PaginatedResponse<T> {
    fn new(data: Vec<T>, page: u64, per_page: u64, total: u64) -> Self {
        Self {
            data,
            page,
            per_page,
            total,
            total_pages: (total + per_page - 1) / per_page,
        }
    }
}
```

---

## 7. WebSocket

```rust
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::broadcast;

async fn ws_handler(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    // 환영 메시지 전송
    if socket.send(Message::Text("환영합니다!".into())).await.is_err() {
        return;
    }

    // 에코 루프
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                let response = format!("보낸 메시지: {text}");
                if socket.send(Message::Text(response)).await.is_err() {
                    break;
                }
            }
            Message::Ping(data) => {
                if socket.send(Message::Pong(data)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    println!("WebSocket 연결 종료");
}

// 브로드캐스트를 이용한 채팅방
async fn chat_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(tx): axum::extract::State<broadcast::Sender<String>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_chat(socket, tx))
}

async fn handle_chat(socket: WebSocket, tx: broadcast::Sender<String>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = tx.subscribe();

    use axum::extract::ws::Message;
    use futures_util::{SinkExt, StreamExt};

    // 브로드캐스트를 이 클라이언트에게 전달
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // 클라이언트 메시지를 브로드캐스트로 전달
    let tx_clone = tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            let _ = tx_clone.send(text);
        }
    });

    // 어느 쪽 태스크가 먼저 끝나는지 대기
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(100);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/chat", get(chat_handler))
        .with_state(tx);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 8. rustls로 TLS

```rust
use axum::{routing::get, Router};
use axum_server::tls_rustls::RustlsConfig;
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let config = RustlsConfig::from_pem_file(
        "certs/cert.pem",
        "certs/key.pem",
    )
    .await
    .unwrap();

    let app = Router::new().route("/", get(|| async { "Hello, TLS!" }));

    let addr = SocketAddr::from(([0, 0, 0, 0], 443));
    println!("HTTPS 서버 실행 중: https://localhost");

    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

---

## 9. reqwest로 HTTP 클라이언트

```rust
use reqwest;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
struct GithubUser {
    login: String,
    name: Option<String>,
    public_repos: u32,
}

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    // 간단한 GET 요청
    let body = reqwest::get("https://httpbin.org/get")
        .await?
        .text()
        .await?;
    println!("응답: {}", &body[..100]);

    // JSON GET 요청
    let user: GithubUser = reqwest::Client::new()
        .get("https://api.github.com/users/rust-lang")
        .header("User-Agent", "rust-example")
        .send()
        .await?
        .json()
        .await?;
    println!("사용자: {} ({:?})", user.login, user.name);

    // JSON 본문과 함께 POST 요청
    #[derive(Serialize)]
    struct Payload {
        title: String,
        body: String,
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client
        .post("https://httpbin.org/post")
        .json(&Payload {
            title: "Hello".into(),
            body: "World".into(),
        })
        .send()
        .await?;

    println!("상태: {}", resp.status());

    Ok(())
}
```

---

## 10. 프로덕션 패턴

### 우아한 셧다운 (Graceful Shutdown)

```rust
use axum::{routing::get, Router};
use tokio::signal;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(|| async { "Hello" }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("핸들러 설치 실패");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("핸들러 설치 실패")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => println!("\nCtrl+C 수신"),
        _ = terminate => println!("\nSIGTERM 수신"),
    }
}
```

### 요청 타임아웃과 속도 제한

```rust
use tower::ServiceBuilder;
use tower_http::timeout::TimeoutLayer;
use std::time::Duration;

let app = Router::new()
    .route("/", get(|| async { "Hello" }))
    .layer(
        ServiceBuilder::new()
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
    );
```

---

## 11. 커넥션 풀링과 데이터베이스

```rust
use axum::{routing::get, Router, extract::State, Json};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

#[derive(sqlx::FromRow, serde::Serialize)]
struct User {
    id: i64,
    name: String,
    email: String,
}

async fn list_users(State(pool): State<PgPool>) -> Json<Vec<User>> {
    let users = sqlx::query_as::<_, User>("SELECT id, name, email FROM users LIMIT 100")
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

    Json(users)
}

#[tokio::main]
async fn main() {
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .min_connections(5)
        .acquire_timeout(Duration::from_secs(5))
        .connect("postgres://user:pass@localhost/mydb")
        .await
        .expect("풀 생성 실패");

    // 마이그레이션 실행
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("마이그레이션 실패");

    let app = Router::new()
        .route("/users", get(list_users))
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

use std::time::Duration;
```

---

## 12. 연습문제

1. **채팅 서버**: 다중 채팅방 TCP 서버를 빌드하세요. 클라이언트가 `/join room_name`을 보내면 방에 참가하고, 메시지는 해당 방의 모든 멤버에게 브로드캐스트됩니다.

2. **REST API**: 게시물과 댓글이 있는 블로그의 완전한 REST API를 빌드하세요. CRUD 작업, 페이지네이션, 검색, 올바른 에러 응답을 포함하세요.

3. **WebSocket 대시보드**: 매초마다 연결된 WebSocket 클라이언트에 시스템 메트릭(CPU, 메모리)을 스트리밍하는 실시간 대시보드를 만드세요.

4. **HTTP 프록시**: 모든 요청을 로깅하고, 타이밍 헤더를 추가하고, 구성 가능한 차단 도메인 목록을 지원하는 간단한 HTTP 포워드 프록시를 빌드하세요.

5. **파일 업로드 서버**: 멀티파트 파일 업로드를 받아 중복 제거(해시 기반 이름)와 함께 파일을 저장하고, 적절한 MIME 타입으로 다시 제공하는 HTTP 서버를 빌드하세요.

---

## 참고 자료

- [Axum documentation](https://docs.rs/axum/latest/axum/)
- [Tokio networking tutorial](https://tokio.rs/tokio/tutorial)
- [tower-http documentation](https://docs.rs/tower-http/latest/tower_http/)
- [reqwest documentation](https://docs.rs/reqwest/latest/reqwest/)
- [sqlx documentation](https://docs.rs/sqlx/latest/sqlx/)
- [rustls documentation](https://docs.rs/rustls/latest/rustls/)

---

**이전**: [임베디드 Rust](./10_Embedded_Rust.md) | **다음**: [고급 에러 처리](./12_Advanced_Error_Handling.md)

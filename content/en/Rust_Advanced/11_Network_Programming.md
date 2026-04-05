# 27. Network Programming

**Previous**: [Embedded Rust](./10_Embedded_Rust.md) | **Next**: [Advanced Error Handling](./12_Advanced_Error_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build TCP and UDP servers and clients with Tokio's async networking
2. Create HTTP servers and APIs using the Axum framework
3. Implement WebSocket communication for real-time applications
4. Configure TLS for secure connections using `rustls`
5. Apply production networking patterns: connection pooling, backpressure, and graceful shutdown

---

Rust's async ecosystem provides powerful primitives for network programming. Tokio handles the I/O event loop, while frameworks like Axum and Hyper build on top for HTTP. This lesson covers networking from raw TCP sockets to production HTTP APIs.

## Table of Contents
1. [TCP with Tokio](#1-tcp-with-tokio)
2. [UDP with Tokio](#2-udp-with-tokio)
3. [HTTP with Axum](#3-http-with-axum)
4. [Request Routing and Extractors](#4-request-routing-and-extractors)
5. [Middleware and Layers](#5-middleware-and-layers)
6. [JSON APIs](#6-json-apis)
7. [WebSocket](#7-websocket)
8. [TLS with rustls](#8-tls-with-rustls)
9. [HTTP Client with reqwest](#9-http-client-with-reqwest)
10. [Production Patterns](#10-production-patterns)
11. [Connection Pooling and Database](#11-connection-pooling-and-database)
12. [Exercises](#12-exercises)

---

## 1. TCP with Tokio

### Echo Server

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("TCP server listening on :8080");

    loop {
        let (mut socket, addr) = listener.accept().await?;
        println!("New connection from {addr}");

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];

            loop {
                let n = match socket.read(&mut buf).await {
                    Ok(0) => {
                        println!("{addr} disconnected");
                        return;
                    }
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("Read error from {addr}: {e}");
                        return;
                    }
                };

                if let Err(e) = socket.write_all(&buf[..n]).await {
                    eprintln!("Write error to {addr}: {e}");
                    return;
                }
            }
        });
    }
}
```

### Line-Based Protocol with Codec

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
            println!("{addr} disconnected");
        });
    }
}
```

### TCP Client

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect("127.0.0.1:8080").await?;
    println!("Connected to server");

    stream.write_all(b"Hello, server!\n").await?;

    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;
    println!("Response: {}", String::from_utf8_lossy(&buf[..n]));

    Ok(())
}
```

---

## 2. UDP with Tokio

```rust
use tokio::net::UdpSocket;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Server
    let server = UdpSocket::bind("127.0.0.1:9090").await?;
    println!("UDP server on :9090");

    tokio::spawn(async move {
        let mut buf = [0u8; 1024];
        loop {
            let (len, addr) = server.recv_from(&mut buf).await.unwrap();
            let msg = String::from_utf8_lossy(&buf[..len]);
            println!("Received from {addr}: {msg}");

            let response = format!("Echo: {msg}");
            server.send_to(response.as_bytes(), addr).await.unwrap();
        }
    });

    // Client
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let client = UdpSocket::bind("127.0.0.1:0").await?;  // Random port
    client.send_to(b"Hello UDP!", "127.0.0.1:9090").await?;

    let mut buf = [0u8; 1024];
    let (len, _) = client.recv_from(&mut buf).await?;
    println!("Client got: {}", String::from_utf8_lossy(&buf[..len]));

    Ok(())
}
```

---

## 3. HTTP with Axum

Axum is the recommended Rust HTTP framework, built on Tokio and Tower:

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

### Minimal Server

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
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
```

---

## 4. Request Routing and Extractors

Axum uses extractors to parse request data:

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

// Application state
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

// Path parameter extraction
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

// Query parameter extraction
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

// JSON body extraction
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

// Delete handler
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

## 5. Middleware and Layers

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

// Custom middleware: request timing
async fn timing_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;

    let elapsed = start.elapsed();
    println!("{method} {uri} — {:?} — {}", elapsed, response.status());

    response
}

// Custom middleware: API key authentication
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
    "Public endpoint"
}

async fn protected_handler() -> &'static str {
    "Protected endpoint"
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

## 6. JSON APIs

### Structured Error Responses

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

// Handler returning Result
async fn get_user(Path(id): Path<u64>) -> Result<Json<User>, ApiError> {
    if id == 0 {
        return Err(ApiError::bad_request(
            "Invalid ID",
            "User ID must be greater than 0",
        ));
    }

    // Simulate lookup
    if id > 100 {
        return Err(ApiError::not_found(format!("User {id} not found")));
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

### Pagination

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
    // Send welcome message
    if socket.send(Message::Text("Welcome!".into())).await.is_err() {
        return;
    }

    // Echo loop
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                let response = format!("You said: {text}");
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
    println!("WebSocket connection closed");
}

// Chat room with broadcast
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

    // Forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Forward client messages to broadcast
    let tx_clone = tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            let _ = tx_clone.send(text);
        }
    });

    // Wait for either task to finish
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

## 8. TLS with rustls

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
    println!("HTTPS server on https://localhost");

    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

---

## 9. HTTP Client with reqwest

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
    // Simple GET
    let body = reqwest::get("https://httpbin.org/get")
        .await?
        .text()
        .await?;
    println!("Response: {}", &body[..100]);

    // JSON GET
    let user: GithubUser = reqwest::Client::new()
        .get("https://api.github.com/users/rust-lang")
        .header("User-Agent", "rust-example")
        .send()
        .await?
        .json()
        .await?;
    println!("User: {} ({:?})", user.login, user.name);

    // POST with JSON body
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

    println!("Status: {}", resp.status());

    Ok(())
}
```

---

## 10. Production Patterns

### Graceful Shutdown

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
        signal::ctrl_c().await.expect("failed to install handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => println!("\nReceived Ctrl+C"),
        _ = terminate => println!("\nReceived SIGTERM"),
    }
}
```

### Request Timeout and Rate Limiting

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

## 11. Connection Pooling and Database

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
        .expect("Failed to create pool");

    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Migration failed");

    let app = Router::new()
        .route("/users", get(list_users))
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

use std::time::Duration;
```

---

## 12. Exercises

1. **Chat server**: Build a multi-room TCP chat server. Clients send `/join room_name` to join a room, and messages are broadcast to all members of that room.

2. **REST API**: Build a complete REST API for a blog with posts and comments. Include CRUD operations, pagination, search, and proper error responses.

3. **WebSocket dashboard**: Create a real-time dashboard that streams system metrics (CPU, memory) to connected WebSocket clients every second.

4. **HTTP proxy**: Build a simple HTTP forward proxy that logs all requests, adds timing headers, and supports a configurable blocklist of domains.

5. **File upload server**: Build an HTTP server that accepts multipart file uploads, stores files with deduplication (hash-based naming), and serves them back with proper MIME types.

---

## References

- [Axum documentation](https://docs.rs/axum/latest/axum/)
- [Tokio networking tutorial](https://tokio.rs/tokio/tutorial)
- [tower-http documentation](https://docs.rs/tower-http/latest/tower_http/)
- [reqwest documentation](https://docs.rs/reqwest/latest/reqwest/)
- [sqlx documentation](https://docs.rs/sqlx/latest/sqlx/)
- [rustls documentation](https://docs.rs/rustls/latest/rustls/)

---

**Previous**: [Embedded Rust](./10_Embedded_Rust.md) | **Next**: [Advanced Error Handling](./12_Advanced_Error_Handling.md)

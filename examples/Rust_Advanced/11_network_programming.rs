// 11_network_programming.rs — TCP/UDP patterns and HTTP concepts
//
// Run: rustc 11_network_programming.rs && ./11_network_programming
//
// Demonstrates network programming concepts without requiring
// a running server. Uses in-memory simulation for portability.

use std::collections::HashMap;
use std::fmt;

fn main() {
    println!("=== HTTP Request/Response ===");
    http_demo();

    println!("\n=== URL Parser ===");
    url_parser_demo();

    println!("\n=== Router Pattern ===");
    router_demo();

    println!("\n=== Connection Pool ===");
    connection_pool_demo();

    println!("\n=== Protocol Framing ===");
    framing_demo();
}

// --- HTTP request/response modeling ---

#[derive(Debug)]
enum Method { Get, Post, Put, Delete }

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Method::Get => write!(f, "GET"),
            Method::Post => write!(f, "POST"),
            Method::Put => write!(f, "PUT"),
            Method::Delete => write!(f, "DELETE"),
        }
    }
}

struct Request {
    method: Method,
    path: String,
    headers: HashMap<String, String>,
    body: Option<String>,
}

impl Request {
    fn get(path: &str) -> Self {
        Request { method: Method::Get, path: path.to_string(), headers: HashMap::new(), body: None }
    }

    fn post(path: &str, body: &str) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        Request { method: Method::Post, path: path.to_string(), headers, body: Some(body.into()) }
    }

    fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} HTTP/1.1", self.method, self.path)?;
        for (k, v) in &self.headers {
            write!(f, "\n{k}: {v}")?;
        }
        if let Some(body) = &self.body {
            write!(f, "\n\n{body}")?;
        }
        Ok(())
    }
}

struct Response {
    status: u16,
    status_text: String,
    body: String,
}

impl Response {
    fn ok(body: &str) -> Self {
        Response { status: 200, status_text: "OK".into(), body: body.into() }
    }
    fn not_found() -> Self {
        Response { status: 404, status_text: "Not Found".into(), body: "404 Not Found".into() }
    }
    fn json(body: &str) -> Self {
        Response { status: 200, status_text: "OK".into(), body: body.into() }
    }
}

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP/1.1 {} {}\n{}", self.status, self.status_text, self.body)
    }
}

fn http_demo() {
    let req = Request::get("/api/users")
        .header("Accept", "application/json")
        .header("Authorization", "Bearer token123");
    println!("{req}");
    println!();

    let req = Request::post("/api/users", r#"{"name": "Alice"}"#);
    println!("{req}");
}

// --- Simple URL parser ---

#[derive(Debug)]
struct Url {
    scheme: String,
    host: String,
    port: Option<u16>,
    path: String,
    query: HashMap<String, String>,
}

fn parse_url(url: &str) -> Option<Url> {
    let (scheme, rest) = url.split_once("://")?;
    let (authority, path_query) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = if let Some((h, p)) = authority.split_once(':') {
        (h.to_string(), p.parse().ok())
    } else {
        (authority.to_string(), None)
    };

    let (path, query_str) = path_query.split_once('?').unwrap_or((path_query, ""));

    let mut query = HashMap::new();
    if !query_str.is_empty() {
        for pair in query_str.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                query.insert(k.into(), v.into());
            }
        }
    }

    Some(Url {
        scheme: scheme.into(),
        host,
        port,
        path: format!("/{path}"),
        query,
    })
}

fn url_parser_demo() {
    let urls = [
        "https://example.com/api/users?page=1&limit=10",
        "http://localhost:8080/health",
        "https://db.example.com:5432/mydb",
    ];

    for url_str in &urls {
        if let Some(url) = parse_url(url_str) {
            println!("  {}://{}:{} {} {:?}",
                url.scheme, url.host,
                url.port.map(|p| p.to_string()).unwrap_or("(default)".into()),
                url.path, url.query
            );
        }
    }
}

// --- Router pattern ---

type Handler = fn(&Request) -> Response;

struct Router {
    routes: Vec<(Method, String, Handler)>,
}

impl Router {
    fn new() -> Self { Router { routes: Vec::new() } }

    fn get(&mut self, path: &str, handler: Handler) {
        self.routes.push((Method::Get, path.to_string(), handler));
    }

    fn post(&mut self, path: &str, handler: Handler) {
        self.routes.push((Method::Post, path.to_string(), handler));
    }

    fn dispatch(&self, req: &Request) -> Response {
        for (method, path, handler) in &self.routes {
            let method_match = matches!(
                (&req.method, method),
                (Method::Get, Method::Get) | (Method::Post, Method::Post) |
                (Method::Put, Method::Put) | (Method::Delete, Method::Delete)
            );
            if method_match && &req.path == path {
                return handler(req);
            }
        }
        Response::not_found()
    }
}

fn handle_index(_req: &Request) -> Response { Response::ok("Welcome!") }
fn handle_users(_req: &Request) -> Response { Response::json(r#"[{"name":"Alice"},{"name":"Bob"}]"#) }
fn handle_create_user(_req: &Request) -> Response { Response { status: 201, status_text: "Created".into(), body: r#"{"id": 1}"#.into() } }

fn router_demo() {
    let mut router = Router::new();
    router.get("/", handle_index);
    router.get("/users", handle_users);
    router.post("/users", handle_create_user);

    let test_requests = vec![
        Request::get("/"),
        Request::get("/users"),
        Request::post("/users", r#"{"name":"Charlie"}"#),
        Request::get("/unknown"),
    ];

    for req in &test_requests {
        let resp = router.dispatch(req);
        println!("  {} {} → {} {}", req.method, req.path, resp.status, resp.body);
    }
}

// --- Connection pool pattern ---

struct Connection {
    id: u32,
    host: String,
    in_use: bool,
}

struct ConnectionPool {
    connections: Vec<Connection>,
    max_size: usize,
    next_id: u32,
}

impl ConnectionPool {
    fn new(host: &str, max_size: usize) -> Self {
        let connections = (0..max_size as u32)
            .map(|id| Connection { id, host: host.to_string(), in_use: false })
            .collect();
        ConnectionPool { connections, max_size, next_id: max_size as u32 }
    }

    fn acquire(&mut self) -> Option<u32> {
        for conn in &mut self.connections {
            if !conn.in_use {
                conn.in_use = true;
                return Some(conn.id);
            }
        }
        None // Pool exhausted
    }

    fn release(&mut self, id: u32) {
        if let Some(conn) = self.connections.iter_mut().find(|c| c.id == id) {
            conn.in_use = false;
        }
    }

    fn active_count(&self) -> usize {
        self.connections.iter().filter(|c| c.in_use).count()
    }
}

fn connection_pool_demo() {
    let mut pool = ConnectionPool::new("db.example.com", 3);

    let c1 = pool.acquire().unwrap();
    let c2 = pool.acquire().unwrap();
    let c3 = pool.acquire().unwrap();
    println!("  Acquired: {c1}, {c2}, {c3} (active: {})", pool.active_count());

    let c4 = pool.acquire();
    println!("  Pool exhausted: {c4:?}");

    pool.release(c2);
    println!("  Released {c2} (active: {})", pool.active_count());

    let c5 = pool.acquire().unwrap();
    println!("  Reacquired: {c5} (active: {})", pool.active_count());
}

// --- Protocol framing (length-prefixed messages) ---

fn encode_message(msg: &str) -> Vec<u8> {
    let len = msg.len() as u32;
    let mut frame = Vec::with_capacity(4 + msg.len());
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(msg.as_bytes());
    frame
}

fn decode_messages(data: &[u8]) -> Vec<String> {
    let mut messages = Vec::new();
    let mut offset = 0;

    while offset + 4 <= data.len() {
        let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > data.len() { break; }
        let msg = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
        messages.push(msg);
        offset += len;
    }

    messages
}

fn framing_demo() {
    // Encode multiple messages into a byte stream
    let mut stream = Vec::new();
    for msg in ["hello", "world", "rust networking"] {
        stream.extend_from_slice(&encode_message(msg));
    }
    println!("  Encoded {} bytes", stream.len());

    // Decode back
    let messages = decode_messages(&stream);
    for msg in &messages {
        println!("  Decoded: \"{msg}\"");
    }
}

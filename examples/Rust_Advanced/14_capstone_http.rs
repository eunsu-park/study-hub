// 14_capstone_http.rs — Minimal HTTP server framework (no external deps)
//
// Run: rustc 14_capstone_http.rs && ./14_capstone_http
//
// Demonstrates: routing, middleware, JSON handling, and request processing
// patterns used in production HTTP servers. Uses in-memory simulation.

use std::collections::HashMap;
use std::fmt;

fn main() {
    println!("=== HTTP Framework Demo ===\n");

    let mut app = App::new();

    // Register middleware
    app.use_middleware(logging_middleware);
    app.use_middleware(cors_middleware);

    // Register routes
    app.get("/", handle_index);
    app.get("/api/users", handle_list_users);
    app.post("/api/users", handle_create_user);
    app.get("/api/users/:id", handle_get_user);
    app.delete("/api/users/:id", handle_delete_user);
    app.get("/health", handle_health);

    // Simulate requests
    let requests = vec![
        Request::new("GET", "/"),
        Request::new("GET", "/api/users"),
        Request::new("POST", "/api/users").with_body(r#"{"name":"Alice","email":"alice@example.com"}"#),
        Request::new("GET", "/api/users/1"),
        Request::new("DELETE", "/api/users/1"),
        Request::new("GET", "/health"),
        Request::new("GET", "/not-found"),
    ];

    for req in requests {
        println!("--- {} {} ---", req.method, req.path);
        let resp = app.handle(req);
        println!("  → {} {}", resp.status, resp.status_text);
        if !resp.body.is_empty() {
            println!("  Body: {}", resp.body);
        }
        println!();
    }
}

// === Core Types ===

#[derive(Debug, Clone)]
struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
    params: HashMap<String, String>,
}

impl Request {
    fn new(method: &str, path: &str) -> Self {
        Request {
            method: method.to_string(),
            path: path.to_string(),
            headers: HashMap::new(),
            body: String::new(),
            params: HashMap::new(),
        }
    }

    fn with_body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self.headers.insert("Content-Type".into(), "application/json".into());
        self
    }

    fn param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }
}

struct Response {
    status: u16,
    status_text: String,
    headers: HashMap<String, String>,
    body: String,
}

impl Response {
    fn ok() -> Self {
        Response { status: 200, status_text: "OK".into(), headers: HashMap::new(), body: String::new() }
    }

    fn json(body: &str) -> Self {
        let mut r = Self::ok();
        r.headers.insert("Content-Type".into(), "application/json".into());
        r.body = body.to_string();
        r
    }

    fn status(code: u16, text: &str) -> Self {
        Response { status: code, status_text: text.into(), headers: HashMap::new(), body: String::new() }
    }

    fn with_body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }
}

// === Router ===

type HandlerFn = fn(&Request) -> Response;
type MiddlewareFn = fn(&mut Request, &mut Response);

struct Route {
    method: String,
    pattern: String,
    handler: HandlerFn,
}

impl Route {
    fn matches(&self, method: &str, path: &str) -> Option<HashMap<String, String>> {
        if self.method != method {
            return None;
        }

        let pattern_parts: Vec<&str> = self.pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        if pattern_parts.len() != path_parts.len() {
            return None;
        }

        let mut params = HashMap::new();
        for (pat, actual) in pattern_parts.iter().zip(path_parts.iter()) {
            if pat.starts_with(':') {
                params.insert(pat[1..].to_string(), actual.to_string());
            } else if pat != actual {
                return None;
            }
        }

        Some(params)
    }
}

// === Application ===

struct App {
    routes: Vec<Route>,
    middleware: Vec<MiddlewareFn>,
}

impl App {
    fn new() -> Self {
        App { routes: Vec::new(), middleware: Vec::new() }
    }

    fn get(&mut self, pattern: &str, handler: HandlerFn) {
        self.routes.push(Route { method: "GET".into(), pattern: pattern.into(), handler });
    }

    fn post(&mut self, pattern: &str, handler: HandlerFn) {
        self.routes.push(Route { method: "POST".into(), pattern: pattern.into(), handler });
    }

    fn delete(&mut self, pattern: &str, handler: HandlerFn) {
        self.routes.push(Route { method: "DELETE".into(), pattern: pattern.into(), handler });
    }

    fn use_middleware(&mut self, mw: MiddlewareFn) {
        self.middleware.push(mw);
    }

    fn handle(&self, mut req: Request) -> Response {
        // Find matching route
        for route in &self.routes {
            if let Some(params) = route.matches(&req.method, &req.path) {
                req.params = params;

                // Run pre-middleware
                let mut resp = (route.handler)(&req);

                // Run post-middleware
                for mw in &self.middleware {
                    mw(&mut req, &mut resp);
                }

                return resp;
            }
        }

        Response::status(404, "Not Found").with_body(r#"{"error":"not found"}"#)
    }
}

// === Middleware ===

fn logging_middleware(req: &mut Request, resp: &mut Response) {
    println!("  [LOG] {} {} → {}", req.method, req.path, resp.status);
}

fn cors_middleware(_req: &mut Request, resp: &mut Response) {
    resp.headers.insert("Access-Control-Allow-Origin".into(), "*".into());
}

// === Handlers ===

fn handle_index(_req: &Request) -> Response {
    Response::json(r#"{"message":"Welcome to the API"}"#)
}

fn handle_list_users(_req: &Request) -> Response {
    Response::json(r#"{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]}"#)
}

fn handle_create_user(req: &Request) -> Response {
    if req.body.is_empty() {
        return Response::status(400, "Bad Request")
            .with_body(r#"{"error":"body required"}"#);
    }
    Response { status: 201, status_text: "Created".into(), headers: HashMap::new(),
        body: format!(r#"{{"id":3,"created":true,"received":{}}}"#, req.body) }
}

fn handle_get_user(req: &Request) -> Response {
    let id = req.param("id").unwrap_or("0");
    Response::json(&format!(r#"{{"id":{},"name":"User {}","email":"user{}@example.com"}}"#, id, id, id))
}

fn handle_delete_user(req: &Request) -> Response {
    let id = req.param("id").unwrap_or("0");
    Response::json(&format!(r#"{{"deleted":true,"id":{}}}"#, id))
}

fn handle_health(_req: &Request) -> Response {
    Response::json(r#"{"status":"healthy","uptime":"42s"}"#)
}

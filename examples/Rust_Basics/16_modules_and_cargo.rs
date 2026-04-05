// 16_modules_and_cargo.rs — Modules, visibility, and use
//
// Run: rustc 16_modules_and_cargo.rs && ./16_modules_and_cargo

fn main() {
    println!("=== Inline Modules ===");
    inline_modules();

    println!("\n=== Nested Modules ===");
    nested_modules();

    println!("\n=== Visibility Rules ===");
    visibility_demo();

    println!("\n=== Use and Re-exports ===");
    use_demo();
}

// --- Inline modules ---

mod math {
    pub fn add(a: i32, b: i32) -> i32 { a + b }
    pub fn multiply(a: i32, b: i32) -> i32 { a * b }

    // Private — only accessible within this module
    fn _validate(n: i32) -> bool { n >= 0 }
}

fn inline_modules() {
    // Access via full path
    let sum = math::add(3, 4);
    let product = math::multiply(3, 4);
    println!("add(3,4) = {sum}, multiply(3,4) = {product}");
}

// --- Nested modules with super and self ---

mod network {
    pub mod server {
        pub fn start() -> String {
            let config = super::config::default_port();
            format!("Server started on port {config}")
        }
    }

    mod config {
        pub(super) fn default_port() -> u16 {
            8080
        }
    }

    pub mod client {
        pub fn connect(host: &str, port: u16) -> String {
            format!("Connected to {host}:{port}")
        }
    }
}

fn nested_modules() {
    println!("{}", network::server::start());
    println!("{}", network::client::connect("localhost", 3000));
    // network::config::default_port(); // ERROR: config is private
}

// --- Visibility levels ---

mod api {
    // pub — visible everywhere
    pub struct Request {
        pub method: String,
        pub path: String,
        headers: Vec<(String, String)>, // private field
    }

    impl Request {
        // Public constructor needed because headers is private
        pub fn new(method: &str, path: &str) -> Self {
            Request {
                method: method.to_string(),
                path: path.to_string(),
                headers: Vec::new(),
            }
        }

        pub fn add_header(&mut self, key: &str, value: &str) {
            self.headers.push((key.to_string(), value.to_string()));
        }

        pub fn header_count(&self) -> usize {
            self.headers.len()
        }
    }

    // pub(crate) — visible within this crate only
    pub(crate) fn internal_helper() -> &'static str {
        "crate-internal"
    }

    // pub(super) — visible to the parent module only
    pub mod handlers {
        pub fn index() -> String {
            "200 OK: index".to_string()
        }

        pub(super) fn health_check() -> String {
            "200 OK: healthy".to_string()
        }
    }
}

fn visibility_demo() {
    let mut req = api::Request::new("GET", "/index");
    req.add_header("Accept", "text/html");
    println!("Request: {} {} ({} headers)", req.method, req.path, req.header_count());
    // println!("{:?}", req.headers); // ERROR: field is private

    println!("Helper: {}", api::internal_helper());
    println!("Handler: {}", api::handlers::index());
    // api::handlers::health_check(); // ERROR: pub(super) — only visible within api
}

// --- use keyword ---

mod shapes {
    pub struct Circle {
        pub radius: f64,
    }

    impl Circle {
        pub fn new(radius: f64) -> Self {
            Circle { radius }
        }

        pub fn area(&self) -> f64 {
            std::f64::consts::PI * self.radius * self.radius
        }
    }

    pub struct Square {
        pub side: f64,
    }

    impl Square {
        pub fn new(side: f64) -> Self {
            Square { side }
        }

        pub fn area(&self) -> f64 {
            self.side * self.side
        }
    }
}

fn use_demo() {
    // Bring items into scope
    use shapes::{Circle, Square};

    let c = Circle::new(5.0);
    let s = Square::new(4.0);
    println!("Circle area: {:.2}", c.area());
    println!("Square area: {:.2}", s.area());

    // Alias with `as`
    use shapes::Circle as Circ;
    let c2 = Circ::new(3.0);
    println!("Circ alias area: {:.2}", c2.area());
}

// 06_structs_and_methods.rs — Struct definitions, methods, and patterns
//
// Run: rustc 06_structs_and_methods.rs && ./06_structs_and_methods

fn main() {
    println!("=== Named-Field Structs ===");
    named_field_demo();

    println!("\n=== Methods and Associated Functions ===");
    methods_demo();

    println!("\n=== Tuple and Unit Structs ===");
    tuple_and_unit_structs();

    println!("\n=== Struct Update Syntax ===");
    update_syntax();

    println!("\n=== Builder Pattern ===");
    builder_demo();
}

// --- Named-Field Struct ---

#[derive(Debug, Clone)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Associated function (constructor) — no &self
    fn new(width: f64, height: f64) -> Self {
        Rectangle { width, height }
    }

    fn square(size: f64) -> Self {
        Rectangle { width: size, height: size }
    }

    // Methods — take &self (immutable borrow)
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }

    // Method taking &mut self
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }
}

// Multiple impl blocks are allowed
impl std::fmt::Display for Rectangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}×{}", self.width, self.height)
    }
}

fn named_field_demo() {
    let rect = Rectangle::new(10.0, 5.0);
    println!("Rectangle: {rect}");
    println!("Area: {}", rect.area());
    println!("Perimeter: {}", rect.perimeter());
    println!("Debug: {rect:?}");
}

fn methods_demo() {
    let big = Rectangle::new(20.0, 15.0);
    let small = Rectangle::new(10.0, 5.0);
    let sq = Rectangle::square(10.0);

    println!("{big} can hold {small}: {}", big.can_hold(&small));
    println!("{sq} is_square: {}", sq.is_square());

    // Mutable method
    let mut r = Rectangle::new(5.0, 3.0);
    println!("Before scale: {r}");
    r.scale(2.0);
    println!("After scale(2.0): {r}");
}

// --- Tuple Structs ---

#[derive(Debug)]
struct Color(u8, u8, u8);

#[derive(Debug)]
struct Point(f64, f64, f64);

// --- Unit Struct ---

#[derive(Debug)]
struct Marker;

fn tuple_and_unit_structs() {
    let red = Color(255, 0, 0);
    let origin = Point(0.0, 0.0, 0.0);
    let _m = Marker;

    println!("Color: ({}, {}, {})", red.0, red.1, red.2);
    println!("Point: {:?}", origin);

    // Destructuring
    let Color(r, g, b) = red;
    println!("Destructured: r={r}, g={g}, b={b}");

    let Point(x, y, _z) = origin;
    println!("x={x}, y={y}");
}

// --- Struct Update Syntax ---

#[derive(Debug, Clone)]
struct Config {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_secs: u64,
}

fn update_syntax() {
    let default_config = Config {
        host: String::from("localhost"),
        port: 8080,
        max_connections: 100,
        timeout_secs: 30,
    };

    // Create a new Config, overriding only some fields
    let production = Config {
        host: String::from("0.0.0.0"),
        port: 443,
        ..default_config.clone()
    };

    println!("Default: {:?}", default_config);
    println!("Production: {:?}", production);
}

// --- Builder Pattern ---

#[derive(Debug)]
struct Server {
    host: String,
    port: u16,
    workers: usize,
}

struct ServerBuilder {
    host: String,
    port: u16,
    workers: usize,
}

impl ServerBuilder {
    fn new() -> Self {
        ServerBuilder {
            host: String::from("127.0.0.1"),
            port: 8080,
            workers: 4,
        }
    }

    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    fn build(self) -> Server {
        Server {
            host: self.host,
            port: self.port,
            workers: self.workers,
        }
    }
}

fn builder_demo() {
    let server = ServerBuilder::new()
        .host("0.0.0.0")
        .port(3000)
        .workers(8)
        .build();

    println!("Server: {server:?}");

    // Using defaults
    let default_server = ServerBuilder::new().build();
    println!("Default: {default_server:?}");
}

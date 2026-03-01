// 07_enums_patterns.rs — Enums and pattern matching
//
// Run: rustc 07_enums_patterns.rs && ./07_enums_patterns

#[derive(Debug)]
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle(r) => std::f64::consts::PI * r * r,
            Shape::Rectangle(w, h) => w * h,
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }

    fn describe(&self) -> String {
        match self {
            Shape::Circle(r) => format!("Circle with radius {r:.1}"),
            Shape::Rectangle(w, h) => format!("Rectangle {w:.1}x{h:.1}"),
            Shape::Triangle { base, height } => {
                format!("Triangle base={base:.1} height={height:.1}")
            }
        }
    }
}

// State machine using enums
#[derive(Debug)]
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

impl TrafficLight {
    fn duration_secs(&self) -> u32 {
        match self {
            TrafficLight::Red => 60,
            TrafficLight::Yellow => 5,
            TrafficLight::Green => 45,
        }
    }

    fn next(&self) -> Self {
        match self {
            TrafficLight::Red => TrafficLight::Green,
            TrafficLight::Green => TrafficLight::Yellow,
            TrafficLight::Yellow => TrafficLight::Red,
        }
    }
}

// Option<T> usage — safe division
fn safe_divide(a: f64, b: f64) -> Option<f64> {
    if b.abs() < f64::EPSILON {
        None
    } else {
        Some(a / b)
    }
}

// Chaining Options with and_then
fn parse_and_double(input: &str) -> Option<i64> {
    input
        .parse::<i64>()
        .ok() // Convert Result to Option
        .and_then(|n| n.checked_mul(2)) // Chain with another Option-returning op
}

fn main() {
    println!("=== Shape Enum ===");
    let shapes: Vec<Shape> = vec![
        Shape::Circle(5.0),
        Shape::Rectangle(4.0, 6.0),
        Shape::Triangle {
            base: 3.0,
            height: 8.0,
        },
    ];

    for shape in &shapes {
        println!("{}: area = {:.2}", shape.describe(), shape.area());
    }

    println!("\n=== Traffic Light State Machine ===");
    let mut light = TrafficLight::Red;
    for _ in 0..6 {
        println!("{:?} ({} seconds)", light, light.duration_secs());
        light = light.next();
    }

    println!("\n=== Option<T> ===");
    let results = vec![
        ("10 / 3", safe_divide(10.0, 3.0)),
        ("10 / 0", safe_divide(10.0, 0.0)),
    ];
    for (expr, result) in results {
        match result {
            Some(v) => println!("{expr} = {v:.4}"),
            None => println!("{expr} = undefined"),
        }
    }

    println!("\n=== Option Chaining ===");
    for input in ["42", "abc", "4611686018427387904"] {
        match parse_and_double(input) {
            Some(v) => println!("parse_and_double(\"{input}\") = {v}"),
            None => println!("parse_and_double(\"{input}\") = None"),
        }
    }

    println!("\n=== Match Guards and Binding ===");
    for n in [-5, 0, 3, 7, 42] {
        let desc = match n {
            n if n < 0 => format!("{n} is negative"),
            0 => "zero".to_string(),
            x @ 1..=10 => format!("{x} is small positive"),
            x => format!("{x} is large"),
        };
        println!("{desc}");
    }

    println!("\n=== if let ===");
    let config_max: Option<u32> = Some(100);
    if let Some(max) = config_max {
        println!("Max connections: {max}");
    }

    let missing: Option<u32> = None;
    if let Some(val) = missing {
        println!("Found: {val}");
    } else {
        println!("No value configured — using default");
    }
}

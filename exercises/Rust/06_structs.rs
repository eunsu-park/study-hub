// Exercise: Structs and Methods
// Implement structs with methods and associated functions.
//
// Run: rustc 06_structs.rs && ./06_structs

fn main() {
    // Exercise 1: Circle
    let c = Circle::new(5.0);
    println!("Circle: area={:.2}, circumference={:.2}", c.area(), c.circumference());
    assert!((c.area() - 78.54).abs() < 0.01);

    // Exercise 2: Student
    let s = Student::new("Alice", vec![85.0, 92.0, 78.0, 95.0, 88.0]);
    println!("Student {}: avg={:.1}, highest={:.1}, passing={}",
        s.name, s.average(), s.highest(), s.is_passing(70.0));
    assert!(s.is_passing(70.0));
    assert!(!s.is_passing(95.0));

    // Exercise 3: Celsius/Fahrenheit newtypes
    let c = Celsius(100.0);
    let f = c.to_fahrenheit();
    println!("{:.1}°C = {:.1}°F", c.0, f.0);
    assert!((f.0 - 212.0).abs() < 0.01);
    let back = f.to_celsius();
    assert!((back.0 - 100.0).abs() < 0.01);

    println!("\nAll exercises passed!");
}

// TODO: Implement Circle
struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Self {
        todo!()
    }
    fn area(&self) -> f64 {
        todo!()
    }
    fn circumference(&self) -> f64 {
        todo!()
    }
}

// TODO: Implement Student
struct Student {
    name: String,
    grades: Vec<f64>,
}

impl Student {
    fn new(name: &str, grades: Vec<f64>) -> Self {
        todo!()
    }
    fn average(&self) -> f64 {
        todo!()
    }
    fn highest(&self) -> f64 {
        todo!()
    }
    fn is_passing(&self, threshold: f64) -> bool {
        todo!()
    }
}

// TODO: Implement Celsius and Fahrenheit newtypes
struct Celsius(f64);
struct Fahrenheit(f64);

impl Celsius {
    fn to_fahrenheit(&self) -> Fahrenheit {
        todo!()
    }
}

impl Fahrenheit {
    fn to_celsius(&self) -> Celsius {
        todo!()
    }
}

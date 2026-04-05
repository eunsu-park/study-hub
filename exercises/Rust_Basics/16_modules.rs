// Exercise: Modules and Cargo
// Practice with module organization and visibility.
//
// Run: rustc 16_modules.rs && ./16_modules

// Exercise 1: Create a module hierarchy
// Organize the math functions into a `math` module with submodules

mod math {
    pub mod arithmetic {
        // TODO: Implement add, subtract, multiply, divide
        pub fn add(a: f64, b: f64) -> f64 {
            todo!()
        }
        pub fn multiply(a: f64, b: f64) -> f64 {
            todo!()
        }
    }

    pub mod geometry {
        // TODO: Implement circle_area, rectangle_area
        pub fn circle_area(radius: f64) -> f64 {
            todo!()
        }
        pub fn rectangle_area(width: f64, height: f64) -> f64 {
            todo!()
        }
    }

    // Exercise 2: Re-export for convenience
    // TODO: Add `pub use` statements so users can access functions directly
    // e.g., math::add instead of math::arithmetic::add
}

fn main() {
    // Using fully qualified path
    let sum = math::arithmetic::add(3.0, 4.0);
    println!("3 + 4 = {sum}");

    let area = math::geometry::circle_area(5.0);
    println!("Circle area (r=5): {area:.2}");

    // Exercise 3: Explain why this function is accessible
    // but private helpers inside the module are not
    // math::geometry::_internal_helper(); // Should NOT compile

    println!("\nAll exercises passed!");
}

// Exercise 4: Design a module structure for a library project
// Describe (in comments) how you would organize:
// - A web server with routes, middleware, and database modules
// - Using mod.rs vs named file approach
// - What should be pub, pub(crate), and private

// Your answer here:
// TODO: Write your module design as comments

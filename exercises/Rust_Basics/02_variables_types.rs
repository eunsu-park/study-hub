// Exercise: Variables and Types
// Fix the compilation errors and implement the missing functions.
//
// Run: rustc 02_variables_types.rs && ./02_variables_types

fn main() {
    // Exercise 1: Fix the immutability error
    let x = 5;
    // x = 10;  // TODO: Fix this line so x can be reassigned
    println!("x = {x}");

    // Exercise 2: Use shadowing to convert string to its length
    let data = "hello world";
    // TODO: Shadow `data` to be the length of the string (usize)
    // println!("data length = {data}");

    // Exercise 3: Implement min_max
    let (min, max) = min_max(5, 2, 8);
    println!("min={min}, max={max}");
    assert_eq!(min, 2);
    assert_eq!(max, 8);

    // Exercise 4: Array average
    let rainfall = [3.2, 1.5, 4.8, 2.1, 5.5, 3.0, 2.8, 4.2, 3.9, 1.8, 2.5, 3.7];
    let avg = average(&rainfall);
    println!("Average rainfall: {avg:.2}");

    // Exercise 5: Safe parsing
    safe_parse("255");
    safe_parse("256");
    safe_parse("abc");

    println!("\nAll exercises passed!");
}

fn min_max(a: i32, b: i32, c: i32) -> (i32, i32) {
    // TODO: Return (minimum, maximum) of three numbers
    todo!()
}

fn average(data: &[f64]) -> f64 {
    // TODO: Compute and return the average
    todo!()
}

fn safe_parse(input: &str) {
    // TODO: Parse input as u8, print the value or "Error: ..." using match
    todo!()
}

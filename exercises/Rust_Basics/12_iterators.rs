// Exercise: Closures and Iterators
// Implement solutions using closures and iterator chains.
//
// Run: rustc 12_iterators.rs && ./12_iterators

fn main() {
    // Exercise 1: Apply a closure
    let double = |x: i32| x * 2;
    let triple = |x: i32| x * 3;
    assert_eq!(apply(5, &double), 10);
    assert_eq!(apply(5, &triple), 15);
    println!("Exercise 1 passed!");

    // Exercise 2: Iterator pipeline
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // Sum of squares of odd numbers
    let result = sum_odd_squares(&data);
    assert_eq!(result, 1 + 9 + 25 + 49 + 81); // 165
    println!("Exercise 2 passed!");

    // Exercise 3: Custom iterator — Count up by step
    let by_threes: Vec<i32> = StepCounter::new(0, 3).take(5).collect();
    assert_eq!(by_threes, vec![0, 3, 6, 9, 12]);
    println!("Exercise 3 passed!");

    // Exercise 4: Flatten and process
    let nested = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
    let result = flatten_and_double(&nested);
    assert_eq!(result, vec![2, 4, 6, 8, 10, 12, 14, 16, 18]);
    println!("Exercise 4 passed!");

    println!("\nAll exercises passed!");
}

fn apply(x: i32, f: &dyn Fn(i32) -> i32) -> i32 {
    // TODO: Apply f to x
    todo!()
}

fn sum_odd_squares(data: &[i32]) -> i32 {
    // TODO: Filter odd numbers, square them, and sum
    todo!()
}

// TODO: Implement the StepCounter iterator
struct StepCounter {
    current: i32,
    step: i32,
}

impl StepCounter {
    fn new(start: i32, step: i32) -> Self {
        todo!()
    }
}

impl Iterator for StepCounter {
    type Item = i32;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn flatten_and_double(nested: &[Vec<i32>]) -> Vec<i32> {
    // TODO: Flatten the nested vecs and double each element
    todo!()
}

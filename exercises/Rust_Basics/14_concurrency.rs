// Exercise: Concurrency
// Practice with threads, channels, and shared state.
//
// Run: rustc 14_concurrency.rs && ./14_concurrency

use std::sync::{mpsc, Arc, Mutex};
use std::thread;

fn main() {
    // Exercise 1: Parallel sum
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let result = parallel_sum(&data, 2); // Split into 2 threads
    assert_eq!(result, 55);
    println!("Exercise 1 passed! Parallel sum = {result}");

    // Exercise 2: Channel message collection
    let messages = channel_collect(5);
    assert_eq!(messages.len(), 5);
    println!("Exercise 2 passed! Got {} messages", messages.len());

    // Exercise 3: Shared counter
    let count = shared_counter(10, 100); // 10 threads, each increments 100 times
    assert_eq!(count, 1000);
    println!("Exercise 3 passed! Counter = {count}");

    println!("\nAll exercises passed!");
}

fn parallel_sum(data: &[i32], num_threads: usize) -> i32 {
    // TODO: Split data into num_threads chunks, sum each in a thread,
    // then add partial sums together.
    // Hint: Use thread::scope for easy borrowing
    todo!()
}

fn channel_collect(n: usize) -> Vec<String> {
    // TODO: Spawn n threads, each sends "Hello from thread {i}" via channel.
    // Collect all messages into a Vec and return.
    todo!()
}

fn shared_counter(num_threads: usize, increments_per_thread: usize) -> usize {
    // TODO: Use Arc<Mutex<usize>>. Spawn num_threads threads, each
    // incrementing the counter increments_per_thread times. Return final count.
    todo!()
}

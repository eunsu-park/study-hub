// 14_concurrency.rs — Threads, channels, and Mutex
//
// Run: rustc 14_concurrency.rs && ./14_concurrency

use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== Basic Threads ===");
    basic_threads();

    println!("\n=== Message Passing (Channels) ===");
    channels_demo();

    println!("\n=== Shared State (Mutex) ===");
    mutex_demo();

    println!("\n=== Scoped Threads ===");
    scoped_threads();
}

fn basic_threads() {
    // Spawn a thread — returns a JoinHandle
    let handle = thread::spawn(|| {
        for i in 1..=3 {
            println!("  [spawned] count: {i}");
            thread::sleep(Duration::from_millis(10));
        }
        42 // Return a value from the thread
    });

    for i in 1..=3 {
        println!("  [main] count: {i}");
        thread::sleep(Duration::from_millis(10));
    }

    // Wait for thread to finish and get its return value
    let result = handle.join().expect("Thread panicked");
    println!("Thread returned: {result}");

    // move closure — transfer ownership to the thread
    let data = vec![1, 2, 3];
    let handle = thread::spawn(move || {
        // data is moved into this closure — the thread owns it
        println!("  Thread got data: {data:?}");
    });
    handle.join().unwrap();
    // println!("{data:?}"); // ERROR: data was moved
}

fn channels_demo() {
    // Create a multi-producer, single-consumer channel
    let (tx, rx) = mpsc::channel();

    // Spawn a producer thread
    let tx1 = tx.clone(); // Clone sender for another producer
    thread::spawn(move || {
        let messages = vec!["hello", "from", "thread 1"];
        for msg in messages {
            tx1.send(format!("[T1] {msg}")).unwrap();
            thread::sleep(Duration::from_millis(20));
        }
    });

    // Second producer
    thread::spawn(move || {
        let messages = vec!["hi", "from", "thread 2"];
        for msg in messages {
            tx.send(format!("[T2] {msg}")).unwrap();
            thread::sleep(Duration::from_millis(30));
        }
    });

    // Receive all messages — rx.iter() blocks until all senders are dropped
    for received in rx {
        println!("  Got: {received}");
    }
}

fn mutex_demo() {
    // Arc (Atomic Reference Count) + Mutex for shared mutable state
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for i in 0..5 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            // lock() returns a MutexGuard — automatically unlocks when dropped
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("  Thread {i} incremented counter to {num}");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final counter: {}", *counter.lock().unwrap());
}

fn scoped_threads() {
    // Scoped threads can borrow from the parent stack — no move needed
    let mut data = vec![1, 2, 3, 4, 5];

    thread::scope(|s| {
        // Immutable borrow in one thread
        s.spawn(|| {
            let sum: i32 = data.iter().sum();
            println!("  Sum: {sum}");
        });

        // Another immutable borrow — multiple readers are fine
        s.spawn(|| {
            let max = data.iter().max().unwrap();
            println!("  Max: {max}");
        });
    });
    // All scoped threads have joined by here

    // Now we can mutate again
    data.push(6);
    println!("After scoped threads: {data:?}");
}

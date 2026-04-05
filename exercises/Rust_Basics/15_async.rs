// Exercise: Async/Await
// Practice with async functions and Tokio.
// Note: This requires tokio as a dependency to actually run.
//
// Cargo.toml:
//   [dependencies]
//   tokio = { version = "1", features = ["full"] }
//
// Run: cargo run (in a Cargo project)

// Uncomment these to run with tokio:
// use tokio::time::{sleep, Duration};

fn main() {
    println!("This exercise requires a Cargo project with tokio.");
    println!("See the comments for exercise descriptions.");
}

// Exercise 1: Basic async function
// Write an async function `fetch_data(id: u32) -> String` that:
// - Sleeps for 100ms (simulating a network call)
// - Returns "Data for id {id}"
// Call it from main using #[tokio::main]

// Exercise 2: Concurrent tasks
// Write a function that spawns 5 tokio tasks, each calling fetch_data
// with a different id, and collects all results using JoinSet or join_all.
// All 5 should run concurrently (not sequentially).

// Exercise 3: Select
// Write two async functions: one that resolves after 1 second returning "slow"
// and one after 100ms returning "fast". Use tokio::select! to get whichever
// finishes first.

// Exercise 4: Async channel
// Create a tokio::sync::mpsc channel. Spawn a producer that sends numbers 1-10
// with a 50ms delay between each. In the receiver, sum all values and verify = 55.

/*
// Skeleton for Exercise 1-4 (uncomment in a Cargo project):

#[tokio::main]
async fn main() {
    // Exercise 1
    let data = fetch_data(42).await;
    println!("{data}");

    // Exercise 2
    let results = fetch_all(5).await;
    println!("All results: {results:?}");

    // Exercise 3
    let winner = race().await;
    println!("Winner: {winner}");

    // Exercise 4
    let sum = channel_sum().await;
    assert_eq!(sum, 55);
    println!("Channel sum: {sum}");
}

async fn fetch_data(id: u32) -> String {
    todo!()
}

async fn fetch_all(count: u32) -> Vec<String> {
    todo!()
}

async fn race() -> String {
    todo!()
}

async fn channel_sum() -> i32 {
    todo!()
}
*/

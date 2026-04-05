// Exercise: Advanced Async
// These exercises require tokio. Add to Cargo.toml:
// tokio = { version = "1", features = ["full"] }
// tokio-stream = "0.1"
//
// This file contains conceptual exercises. Build each as a separate binary.

// Exercise 1: Fan-out fetcher
// Write an async function that takes a Vec<String> of URLs and a concurrency
// limit. Fetch all URLs concurrently (using Semaphore) and return results
// in the original order.

// Exercise 2: Async pipeline
// Build: generate numbers 1..=100 → filter evens → map to squares
// → buffer into chunks of 10 → print each chunk
// Use tokio_stream and StreamExt combinators.

// Exercise 3: Graceful shutdown TCP echo server
// On Ctrl+C: stop accepting, wait for active connections (5s deadline),
// force-close remaining.

// Exercise 4: select! with multiple channels
// Create 3 mpsc channels. Each producer sends at different rates.
// Use select! in a loop to process all messages and track per-channel counts.

// Exercise 5: Cancel-safe message processor
// Read from mpsc, process, write to file. Ensure no messages lost on cancel.
// Hint: buffer the current message in a struct field.

fn main() {
    println!("Advanced async exercises require tokio runtime.");
    println!("Create a Cargo project and implement each exercise as a binary.");
    println!();
    println!("Example Cargo.toml:");
    println!("  [dependencies]");
    println!("  tokio = {{ version = \"1\", features = [\"full\"] }}");
    println!("  tokio-stream = \"0.1\"");
}

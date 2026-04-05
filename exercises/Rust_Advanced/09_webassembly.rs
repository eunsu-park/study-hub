// Exercise: WebAssembly (Conceptual)
// Wasm exercises require wasm-pack. This file outlines the exercises.
//
// Setup: cargo new --lib wasm-exercises && cd wasm-exercises
// Add to Cargo.toml: wasm-bindgen = "0.2"

// Exercise 1: Markdown renderer
// Build a Wasm module using pulldown-cmark that converts Markdown to HTML.
// Create an HTML page with textarea input and live preview.

// Exercise 2: Game of Life
// Implement Conway's Game of Life. Render on HTML canvas.
// Add start/stop/step controls and adjustable speed.

// Exercise 3: JSON formatter
// Accept JSON string input, validate it, and output pretty-printed JSON.
// Highlight syntax errors with line/column info.

// Exercise 4: WASI CLI tool
// Target: wasm32-wasip1
// Read a CSV file, compute per-column statistics, output summary.
// Test with: wasmtime --dir=. your_tool.wasm input.csv

// Exercise 5: Yew TODO app
// Full TODO app: add/remove/toggle, filter, localStorage persistence.
// Build with: trunk serve

fn main() {
    println!("WebAssembly exercises require wasm-pack or trunk.");
    println!("Create a dedicated project for each exercise.");
    println!();
    println!("Quick start:");
    println!("  cargo new --lib my-wasm-lib");
    println!("  # Add wasm-bindgen to Cargo.toml");
    println!("  wasm-pack build --target web");
}

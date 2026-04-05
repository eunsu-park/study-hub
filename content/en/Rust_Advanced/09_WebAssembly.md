# 25. WebAssembly

**Previous**: [FFI and Interop](./08_FFI_and_Interop.md) | **Next**: [Embedded Rust](./10_Embedded_Rust.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Compile Rust to WebAssembly using `wasm-pack` and `wasm-bindgen`
2. Interact with JavaScript APIs from Rust via `wasm-bindgen` and `web-sys`
3. Build applications targeting both browser and WASI (server-side) environments
4. Use the Yew framework to build full web front-end applications in Rust
5. Optimize Wasm binary size and debug Wasm applications

---

WebAssembly (Wasm) is a binary instruction format that runs in web browsers alongside JavaScript, and increasingly in server-side environments via WASI. Rust is one of the best languages for targeting Wasm due to its small runtime, no garbage collector, and excellent tooling.

## Table of Contents
1. [WebAssembly Overview](#1-webassembly-overview)
2. [Setup and Tooling](#2-setup-and-tooling)
3. [wasm-bindgen Basics](#3-wasm-bindgen-basics)
4. [Interacting with JavaScript](#4-interacting-with-javascript)
5. [web-sys and js-sys](#5-web-sys-and-js-sys)
6. [DOM Manipulation](#6-dom-manipulation)
7. [WASI: Server-Side Wasm](#7-wasi-server-side-wasm)
8. [The Yew Framework](#8-the-yew-framework)
9. [Binary Size Optimization](#9-binary-size-optimization)
10. [Debugging Wasm](#10-debugging-wasm)
11. [Practical Patterns](#11-practical-patterns)
12. [Exercises](#12-exercises)

---

## 1. WebAssembly Overview

```
┌──────────────────────────────────────┐
│            Web Browser               │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ JavaScript  │◄►│  Wasm Module │  │
│  │   Engine    │  │  (from Rust) │  │
│  └─────────────┘  └──────────────┘  │
│         │                  │         │
│         └──────┬───────────┘         │
│                ▼                     │
│           Web APIs                   │
│  (DOM, fetch, Canvas, WebGL, etc.)   │
└──────────────────────────────────────┘
```

Key characteristics:
- **Compact binary format** — smaller than JavaScript for equivalent code
- **Near-native speed** — ahead-of-time compiled, predictable performance
- **Sandboxed** — runs in the same security sandbox as JavaScript
- **Language-agnostic** — target from Rust, C, C++, Go, and more

### Wasm vs JavaScript Trade-offs

| Aspect | JavaScript | Wasm (Rust) |
|--------|-----------|-------------|
| Startup | Fast (JIT) | Fast (AOT) |
| Peak performance | Good (JIT optimized) | Excellent (near-native) |
| DOM access | Direct | Via JS bridge |
| Bundle size | Small (text) | Small (binary) |
| GC pauses | Yes | No |
| Best for | UI, DOM, glue | Compute, codecs, games |

---

## 2. Setup and Tooling

```bash
# Install the Wasm target
rustup target add wasm32-unknown-unknown

# Install wasm-pack (builds, tests, and publishes Wasm packages)
cargo install wasm-pack

# For WASI target
rustup target add wasm32-wasip1

# Optional: wasmtime runtime for WASI
cargo install wasmtime-cli
```

### Project Setup

```bash
# Create a new library project
cargo new --lib my-wasm-lib
cd my-wasm-lib
```

```toml
# Cargo.toml
[package]
name = "my-wasm-lib"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = "s"     # Optimize for size
lto = true          # Link-time optimization
```

---

## 3. wasm-bindgen Basics

`wasm-bindgen` bridges Rust and JavaScript, handling type conversions automatically:

```rust
use wasm_bindgen::prelude::*;

// Export a function to JavaScript
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

// Export a struct
#[wasm_bindgen]
pub struct Calculator {
    value: f64,
}

#[wasm_bindgen]
impl Calculator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Calculator {
        Calculator { value: 0.0 }
    }

    pub fn add(&mut self, n: f64) {
        self.value += n;
    }

    pub fn subtract(&mut self, n: f64) {
        self.value -= n;
    }

    pub fn multiply(&mut self, n: f64) {
        self.value *= n;
    }

    pub fn result(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}
```

Build and use:

```bash
wasm-pack build --target web
```

```html
<!DOCTYPE html>
<html>
<body>
<script type="module">
  import init, { greet, Calculator } from './pkg/my_wasm_lib.js';

  async function main() {
    await init();

    console.log(greet("World"));  // "Hello, World!"

    const calc = new Calculator();
    calc.add(10);
    calc.multiply(3);
    calc.subtract(5);
    console.log(`Result: ${calc.result()}`);  // 25
    calc.free();  // Free Wasm memory
  }

  main();
</script>
</body>
</html>
```

### Build Targets

```bash
# For use with bundlers (webpack, vite, etc.)
wasm-pack build --target bundler

# For direct use in browsers (ES modules)
wasm-pack build --target web

# For Node.js
wasm-pack build --target nodejs
```

---

## 4. Interacting with JavaScript

### Importing JavaScript Functions

```rust
use wasm_bindgen::prelude::*;

// Import JavaScript functions
#[wasm_bindgen]
extern "C" {
    // console.log
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // console.warn
    #[wasm_bindgen(js_namespace = console, js_name = warn)]
    fn console_warn(s: &str);

    // window.alert
    fn alert(s: &str);

    // Import a custom JS function
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;

    // Import a JS class
    type Date;
    #[wasm_bindgen(constructor)]
    fn new() -> Date;
    #[wasm_bindgen(method, js_name = toISOString)]
    fn to_iso_string(this: &Date) -> String;
}

#[wasm_bindgen]
pub fn demo() {
    log("Hello from Rust!");
    console_warn("This is a warning");

    let date = Date::new();
    log(&format!("Current time: {}", date.to_iso_string()));
    log(&format!("Random number: {}", random()));
}
```

### Passing Complex Types

```rust
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct UserData {
    pub name: String,
    pub age: u32,
    pub scores: Vec<f64>,
}

// Use serde for complex type conversion
#[wasm_bindgen]
pub fn process_user(val: JsValue) -> Result<JsValue, JsValue> {
    let user: UserData = serde_wasm_bindgen::from_value(val)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let avg_score: f64 = user.scores.iter().sum::<f64>() / user.scores.len() as f64;

    let result = serde_json::json!({
        "name": user.name,
        "average_score": avg_score,
        "grade": if avg_score >= 90.0 { "A" } else if avg_score >= 80.0 { "B" } else { "C" }
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

---

## 5. web-sys and js-sys

`web-sys` provides bindings to Web APIs. `js-sys` provides bindings to JavaScript built-ins:

```toml
[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
    "Document", "Element", "HtmlElement", "Window",
    "console", "HtmlCanvasElement", "CanvasRenderingContext2d",
    "Request", "RequestInit", "Response", "Headers",
] }
js-sys = "0.3"
```

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, Window};

fn window() -> Window {
    web_sys::window().expect("no global `window`")
}

fn document() -> Document {
    window().document().expect("no `document`")
}

#[wasm_bindgen]
pub fn create_paragraph(text: &str) -> Result<(), JsValue> {
    let document = document();
    let body = document.body().expect("no body");

    let p = document.create_element("p")?;
    p.set_text_content(Some(text));
    p.set_attribute("class", "rust-paragraph")?;
    body.append_child(&p)?;

    Ok(())
}

// Using js-sys for JavaScript built-in types
use js_sys::{Array, Date, Map, Promise};

#[wasm_bindgen]
pub fn js_types_demo() {
    // Create JS Array
    let arr = Array::new();
    arr.push(&JsValue::from(1));
    arr.push(&JsValue::from(2));
    arr.push(&JsValue::from(3));

    web_sys::console::log_1(&format!("Array length: {}", arr.length()).into());

    // JS Date
    let now = Date::new_0();
    web_sys::console::log_1(&format!("Time: {}", now.to_iso_string()).into());

    // JS Map
    let map = Map::new();
    map.set(&"key".into(), &"value".into());
}
```

---

## 6. DOM Manipulation

### Canvas Drawing

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use std::f64::consts::PI;

#[wasm_bindgen]
pub fn draw_chart(canvas_id: &str, data: &[f64]) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()?;

    let ctx = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;

    let width = canvas.width() as f64;
    let height = canvas.height() as f64;

    // Clear canvas
    ctx.clear_rect(0.0, 0.0, width, height);

    // Draw bar chart
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bar_width = width / data.len() as f64 * 0.8;
    let gap = width / data.len() as f64 * 0.2;

    let colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"];

    for (i, &value) in data.iter().enumerate() {
        let bar_height = (value / max_val) * (height * 0.8);
        let x = i as f64 * (bar_width + gap) + gap;
        let y = height - bar_height;

        ctx.set_fill_style_str(colors[i % colors.len()]);
        ctx.fill_rect(x, y, bar_width, bar_height);

        // Label
        ctx.set_fill_style_str("#333");
        ctx.set_font("14px sans-serif");
        ctx.set_text_align("center");
        ctx.fill_text(
            &format!("{:.0}", value),
            x + bar_width / 2.0,
            y - 5.0,
        )?;
    }

    Ok(())
}
```

### Event Handling

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
pub fn setup_click_handler(button_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let button = document.get_element_by_id(button_id).unwrap();

    let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let x = event.client_x();
        let y = event.client_y();
        web_sys::console::log_1(&format!("Clicked at ({x}, {y})").into());
    }) as Box<dyn FnMut(_)>);

    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;

    // IMPORTANT: Prevent the closure from being dropped
    closure.forget();

    Ok(())
}
```

---

## 7. WASI: Server-Side Wasm

WASI (WebAssembly System Interface) allows Wasm to run outside the browser with controlled access to system resources:

```rust
// Simple WASI program — file I/O and environment
use std::env;
use std::fs;

fn main() {
    // Environment variables
    for (key, value) in env::vars() {
        println!("{key}={value}");
    }

    // Command-line arguments
    let args: Vec<String> = env::args().collect();
    println!("Args: {args:?}");

    // File I/O (sandboxed)
    let content = "Hello from WASI Rust!\n";
    fs::write("output.txt", content).expect("Failed to write");

    let read_back = fs::read_to_string("output.txt").expect("Failed to read");
    println!("Read: {read_back}");

    // Current time
    let now = std::time::SystemTime::now();
    println!("Time: {now:?}");
}
```

Build and run:

```bash
# Build for WASI
cargo build --target wasm32-wasip1 --release

# Run with wasmtime
wasmtime target/wasm32-wasip1/release/my-wasi-app.wasm

# With directory access (sandboxed)
wasmtime --dir=./data target/wasm32-wasip1/release/my-wasi-app.wasm

# With environment variables
wasmtime --env FOO=bar target/wasm32-wasip1/release/my-wasi-app.wasm
```

### WASI HTTP Server (Component Model)

```rust
// Using wasi-http (experimental, component model)
// This demonstrates the direction WASI is heading

use std::io::Write;

fn main() {
    // WASI provides standardized interfaces for:
    // - Filesystem (wasi:filesystem)
    // - Sockets (wasi:sockets)
    // - HTTP (wasi:http)
    // - Clocks (wasi:clocks)
    // - Random (wasi:random)

    println!("WASI enables portable server-side Wasm");
    println!("Run the same binary on wasmtime, wasmer, or WasmEdge");
}
```

---

## 8. The Yew Framework

Yew is a Rust framework for building web front-end applications, inspired by React:

```toml
[dependencies]
yew = { version = "0.21", features = ["csr"] }
```

### Basic Component

```rust
use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    let counter = use_state(|| 0);

    let increment = {
        let counter = counter.clone();
        Callback::from(move |_| counter.set(*counter + 1))
    };

    let decrement = {
        let counter = counter.clone();
        Callback::from(move |_| counter.set(*counter - 1))
    };

    html! {
        <div class="app">
            <h1>{ "Yew Counter" }</h1>
            <p>{ format!("Count: {}", *counter) }</p>
            <button onclick={increment}>{ "+1" }</button>
            <button onclick={decrement}>{ "-1" }</button>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
```

### Component with Props and State

```rust
use yew::prelude::*;

#[derive(Properties, PartialEq)]
struct TodoItemProps {
    text: String,
    done: bool,
    on_toggle: Callback<()>,
}

#[function_component(TodoItem)]
fn todo_item(props: &TodoItemProps) -> Html {
    let style = if props.done { "text-decoration: line-through" } else { "" };

    html! {
        <li style={style} onclick={props.on_toggle.reform(|_| ())}>
            { &props.text }
        </li>
    }
}

#[derive(Clone, PartialEq)]
struct Todo {
    text: String,
    done: bool,
}

#[function_component(TodoApp)]
fn todo_app() -> Html {
    let todos = use_state(|| vec![
        Todo { text: "Learn Rust".into(), done: true },
        Todo { text: "Learn Yew".into(), done: false },
        Todo { text: "Build something".into(), done: false },
    ]);

    let input_ref = use_node_ref();

    let on_add = {
        let todos = todos.clone();
        let input_ref = input_ref.clone();
        Callback::from(move |_| {
            if let Some(input) = input_ref.cast::<web_sys::HtmlInputElement>() {
                let text = input.value();
                if !text.is_empty() {
                    let mut new_todos = (*todos).clone();
                    new_todos.push(Todo { text, done: false });
                    todos.set(new_todos);
                    input.set_value("");
                }
            }
        })
    };

    html! {
        <div>
            <h1>{ "Todo App" }</h1>
            <div>
                <input ref={input_ref} placeholder="New todo..." />
                <button onclick={on_add}>{ "Add" }</button>
            </div>
            <ul>
                { for todos.iter().enumerate().map(|(i, todo)| {
                    let todos = todos.clone();
                    let on_toggle = Callback::from(move |_| {
                        let mut new_todos = (*todos).clone();
                        new_todos[i].done = !new_todos[i].done;
                        todos.set(new_todos);
                    });
                    html! {
                        <TodoItem
                            text={todo.text.clone()}
                            done={todo.done}
                            on_toggle={on_toggle}
                        />
                    }
                })}
            </ul>
        </div>
    }
}
```

Build with Trunk:

```bash
cargo install trunk
trunk serve  # Dev server with hot reload
trunk build --release  # Production build
```

---

## 9. Binary Size Optimization

Wasm binary size directly affects load time:

```toml
# Cargo.toml
[profile.release]
opt-level = "z"         # Optimize for size (aggressive)
lto = true              # Link-time optimization
codegen-units = 1       # Single codegen unit (slower build, better optimization)
strip = true            # Strip debug symbols
panic = "abort"         # No unwinding code
```

### wasm-opt Post-Processing

```bash
# Install binaryen tools
# brew install binaryen (macOS)
# apt install binaryen (Ubuntu)

# Optimize the Wasm binary further
wasm-opt -Oz -o optimized.wasm original.wasm

# Typical size reduction pipeline:
# 1. Cargo release build:  ~200KB
# 2. wasm-opt:             ~150KB
# 3. gzip compression:     ~50KB
```

### Size Analysis

```bash
# Analyze what's taking up space in the Wasm binary
cargo install twiggy

twiggy top target/wasm32-unknown-unknown/release/my_lib.wasm
twiggy dominators target/wasm32-unknown-unknown/release/my_lib.wasm
```

### Common Size Reducers

```rust
// 1. Use #[wasm_bindgen] only on what's needed
// 2. Avoid format! and println! when possible (pulls in formatting machinery)
// 3. Use no_std where possible for libraries

// 4. Replace String with &str where possible
#[wasm_bindgen]
pub fn process(input: &str) -> String {  // &str input avoids allocation
    input.to_uppercase()
}

// 5. Avoid pulling in large dependencies
// Use web-sys feature flags to include only what you need
```

---

## 10. Debugging Wasm

### Console Logging

```rust
// Simple logging macro for Wasm
macro_rules! console_log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into())
    };
}

#[wasm_bindgen]
pub fn debug_demo() {
    console_log!("Debug value: {}", 42);
    console_log!("Complex: {:?}", vec![1, 2, 3]);
}
```

### Panic Hook

```rust
use wasm_bindgen::prelude::*;

// Show Rust panics as console errors with stack traces
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
```

```toml
[dependencies]
console_error_panic_hook = "0.1"
```

### Browser DevTools

```
1. Build with debug info: wasm-pack build --dev
2. Open browser DevTools → Sources tab
3. Rust source maps enable stepping through Rust code
4. Memory tab shows Wasm linear memory
```

### Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_greet() {
        assert_eq!(greet("Rust"), "Hello, Rust!");
    }

    #[wasm_bindgen_test]
    fn test_calculator() {
        let mut calc = Calculator::new();
        calc.add(10.0);
        calc.multiply(3.0);
        assert_eq!(calc.result(), 30.0);
    }
}
```

```bash
wasm-pack test --chrome --headless
```

---

## 11. Practical Patterns

### Shared Memory with JavaScript

```rust
use wasm_bindgen::prelude::*;

// Expose raw memory for zero-copy data sharing
#[wasm_bindgen]
pub struct ImageProcessor {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[wasm_bindgen]
impl ImageProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height * 4) as usize;  // RGBA
        ImageProcessor {
            width,
            height,
            pixels: vec![0; size],
        }
    }

    // Return a pointer to the pixel buffer
    // JavaScript can create a Uint8Array view into Wasm memory
    pub fn pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    pub fn pixels_len(&self) -> usize {
        self.pixels.len()
    }

    // Process the image (e.g., grayscale conversion)
    pub fn grayscale(&mut self) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            let gray = (0.299 * chunk[0] as f64
                      + 0.587 * chunk[1] as f64
                      + 0.114 * chunk[2] as f64) as u8;
            chunk[0] = gray;
            chunk[1] = gray;
            chunk[2] = gray;
            // chunk[3] (alpha) unchanged
        }
    }
}
```

JavaScript usage:

```javascript
const processor = new ImageProcessor(800, 600);

// Get a view into Wasm memory (zero-copy!)
const pixels = new Uint8Array(
  wasm.memory.buffer,
  processor.pixels_ptr(),
  processor.pixels_len()
);

// Copy image data from canvas into Wasm memory
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, 800, 600);
pixels.set(imageData.data);

// Process in Rust (fast!)
processor.grayscale();

// Copy back to canvas
imageData.data.set(pixels);
ctx.putImageData(imageData, 0, 0);
```

---

## 12. Exercises

1. **Markdown renderer**: Build a Wasm module that converts Markdown text to HTML. Use the `pulldown-cmark` crate. Create a simple web page with a textarea input and live preview.

2. **Game of Life**: Implement Conway's Game of Life in Rust + Wasm. Render the grid on an HTML canvas. Add start/stop/step controls.

3. **JSON formatter**: Build a Wasm-powered JSON formatter/validator. Input a JSON string, output pretty-printed JSON with syntax highlighting.

4. **WASI CLI tool**: Write a WASI command-line tool that reads a CSV file, computes statistics (mean, median, std dev per column), and outputs a summary. Test with wasmtime.

5. **Yew TODO app**: Build a full TODO application with Yew including: add/remove/toggle items, filter (all/active/completed), local storage persistence, and keyboard shortcuts.

---

## References

- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
- [web-sys documentation](https://rustwasm.github.io/wasm-bindgen/api/web_sys/)
- [WASI documentation](https://wasi.dev/)
- [Yew documentation](https://yew.rs/)
- [Trunk documentation](https://trunkrs.dev/)

---

**Previous**: [FFI and Interop](./08_FFI_and_Interop.md) | **Next**: [Embedded Rust](./10_Embedded_Rust.md)

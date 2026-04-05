// 09_webassembly.rs — WebAssembly concepts and patterns
//
// Run: rustc 09_webassembly.rs && ./09_webassembly
//
// Demonstrates Wasm-compatible patterns. In real projects, use wasm-pack
// and wasm-bindgen to compile to actual WebAssembly.

use std::fmt;

fn main() {
    println!("=== Wasm-friendly Functions ===");
    wasm_functions();

    println!("\n=== Linear Memory Simulation ===");
    linear_memory_demo();

    println!("\n=== DOM Manipulation Pattern ===");
    dom_pattern();

    println!("\n=== Game of Life (Classic Wasm Example) ===");
    game_of_life_demo();
}

// --- Wasm-friendly: simple types, no_std compatible ---

// Functions exported to Wasm must use simple types (i32, f64, etc.)
// Complex types go through linear memory

#[no_mangle]
pub extern "C" fn wasm_add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn wasm_fibonacci(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 2..=n {
        let tmp = a + b;
        a = b;
        b = tmp;
    }
    b
}

#[no_mangle]
pub extern "C" fn wasm_is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 { return false; }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

fn wasm_functions() {
    println!("  add(3, 4) = {}", wasm_add(3, 4));
    println!("  fibonacci(10) = {}", wasm_fibonacci(10));
    println!("  fibonacci(50) = {}", wasm_fibonacci(50));
    println!("  is_prime(97) = {}", wasm_is_prime(97));
    println!("  is_prime(100) = {}", wasm_is_prime(100));
}

// --- Linear memory: how Wasm shares data with JavaScript ---

struct LinearMemory {
    buffer: Vec<u8>,
}

impl LinearMemory {
    fn new(size: usize) -> Self {
        LinearMemory { buffer: vec![0; size] }
    }

    fn write_string(&mut self, offset: usize, s: &str) -> usize {
        let bytes = s.as_bytes();
        let end = offset + bytes.len();
        if end <= self.buffer.len() {
            self.buffer[offset..end].copy_from_slice(bytes);
        }
        bytes.len()
    }

    fn read_string(&self, offset: usize, len: usize) -> String {
        let end = (offset + len).min(self.buffer.len());
        String::from_utf8_lossy(&self.buffer[offset..end]).to_string()
    }

    fn write_i32(&mut self, offset: usize, value: i32) {
        let bytes = value.to_le_bytes();
        self.buffer[offset..offset + 4].copy_from_slice(&bytes);
    }

    fn read_i32(&self, offset: usize) -> i32 {
        let bytes: [u8; 4] = self.buffer[offset..offset + 4].try_into().unwrap();
        i32::from_le_bytes(bytes)
    }
}

fn linear_memory_demo() {
    let mut mem = LinearMemory::new(1024);

    // Write and read a string
    let len = mem.write_string(0, "Hello from Wasm!");
    let s = mem.read_string(0, len);
    println!("  String at offset 0: \"{s}\"");

    // Write and read integers
    mem.write_i32(256, 42);
    mem.write_i32(260, -1);
    println!("  i32 at 256: {}", mem.read_i32(256));
    println!("  i32 at 260: {}", mem.read_i32(260));
}

// --- DOM manipulation pattern ---

#[derive(Debug)]
enum DomEvent {
    Click { x: i32, y: i32 },
    Input { value: String },
    KeyPress { key: char },
}

struct VirtualDom {
    elements: Vec<(String, String)>, // (tag, content)
    log: Vec<String>,
}

impl VirtualDom {
    fn new() -> Self {
        VirtualDom { elements: Vec::new(), log: Vec::new() }
    }

    fn create_element(&mut self, tag: &str, content: &str) -> usize {
        let id = self.elements.len();
        self.elements.push((tag.to_string(), content.to_string()));
        self.log.push(format!("Created <{tag}> #{id}: \"{content}\""));
        id
    }

    fn update_element(&mut self, id: usize, content: &str) {
        if let Some(elem) = self.elements.get_mut(id) {
            elem.1 = content.to_string();
            self.log.push(format!("Updated #{id}: \"{}\"", content));
        }
    }

    fn handle_event(&mut self, event: DomEvent) {
        match event {
            DomEvent::Click { x, y } => {
                self.log.push(format!("Click at ({x}, {y})"));
            }
            DomEvent::Input { value } => {
                self.log.push(format!("Input: \"{value}\""));
            }
            DomEvent::KeyPress { key } => {
                self.log.push(format!("Key: '{key}'"));
            }
        }
    }
}

fn dom_pattern() {
    let mut dom = VirtualDom::new();
    let heading = dom.create_element("h1", "Welcome");
    let _para = dom.create_element("p", "Click the button");
    let _btn = dom.create_element("button", "Click me");

    dom.handle_event(DomEvent::Click { x: 100, y: 200 });
    dom.update_element(heading, "Clicked!");
    dom.handle_event(DomEvent::Input { value: "hello".into() });

    for entry in &dom.log {
        println!("  {entry}");
    }
}

// --- Conway's Game of Life (classic Wasm demo) ---

struct Universe {
    width: u32,
    height: u32,
    cells: Vec<bool>,
}

impl Universe {
    fn new(width: u32, height: u32) -> Self {
        let cells = (0..width * height)
            .map(|i| i % 2 == 0 || i % 7 == 0) // Deterministic seed
            .collect();
        Universe { width, height, cells }
    }

    fn get_index(&self, row: u32, col: u32) -> usize {
        (row * self.width + col) as usize
    }

    fn live_neighbor_count(&self, row: u32, col: u32) -> u8 {
        let mut count = 0;
        for dr in [self.height - 1, 0, 1] {
            for dc in [self.width - 1, 0, 1] {
                if dr == 0 && dc == 0 { continue; }
                let r = (row + dr) % self.height;
                let c = (col + dc) % self.width;
                count += self.cells[self.get_index(r, c)] as u8;
            }
        }
        count
    }

    fn tick(&mut self) {
        let mut next = self.cells.clone();
        for row in 0..self.height {
            for col in 0..self.width {
                let idx = self.get_index(row, col);
                let neighbors = self.live_neighbor_count(row, col);
                next[idx] = match (self.cells[idx], neighbors) {
                    (true, 2) | (true, 3) => true,
                    (true, _) => false,
                    (false, 3) => true,
                    (otherwise, _) => otherwise,
                };
            }
        }
        self.cells = next;
    }

    fn alive_count(&self) -> usize {
        self.cells.iter().filter(|&&c| c).count()
    }
}

impl fmt::Display for Universe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.height {
            for col in 0..self.width {
                let symbol = if self.cells[self.get_index(row, col)] { "■" } else { "□" };
                write!(f, "{symbol}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

fn game_of_life_demo() {
    let mut universe = Universe::new(10, 6);

    println!("  Generation 0 ({} alive):", universe.alive_count());
    print!("{universe}");

    for gen in 1..=3 {
        universe.tick();
        println!("  Generation {gen} ({} alive):", universe.alive_count());
        print!("{universe}");
    }
}

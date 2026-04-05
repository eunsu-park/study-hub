// 02_cli_tool.rs — Building a CLI application pattern (without external crates)
//
// Run: rustc 02_cli_tool.rs && ./02_cli_tool
//
// Demonstrates: argument parsing, command dispatch, and structured output
// In real projects, use clap for parsing and serde for serialization.

use std::collections::HashMap;
use std::fmt;

fn main() {
    println!("=== Manual Argument Parsing ===");
    arg_parsing_demo();

    println!("\n=== Command Dispatch ===");
    command_dispatch();

    println!("\n=== Structured Output ===");
    structured_output();

    println!("\n=== Todo App Simulation ===");
    todo_app_demo();
}

// --- Manual argument parser ---

#[derive(Debug)]
struct Args {
    command: String,
    flags: HashMap<String, String>,
    positional: Vec<String>,
}

fn parse_args(input: &[&str]) -> Args {
    let mut flags = HashMap::new();
    let mut positional = Vec::new();
    let command = input.first().map(|s| s.to_string()).unwrap_or_default();

    let mut i = 1;
    while i < input.len() {
        if input[i].starts_with("--") {
            let key = input[i].trim_start_matches("--").to_string();
            if i + 1 < input.len() && !input[i + 1].starts_with("--") {
                flags.insert(key, input[i + 1].to_string());
                i += 1;
            } else {
                flags.insert(key, "true".to_string());
            }
        } else {
            positional.push(input[i].to_string());
        }
        i += 1;
    }

    Args { command, flags, positional }
}

fn arg_parsing_demo() {
    let input = ["add", "--priority", "high", "--verbose", "Buy groceries"];
    let args = parse_args(&input);
    println!("  Command: {}", args.command);
    println!("  Flags: {:?}", args.flags);
    println!("  Positional: {:?}", args.positional);
}

// --- Command dispatch ---

type CmdFn = fn(&[String]) -> Result<String, String>;

fn cmd_greet(args: &[String]) -> Result<String, String> {
    let name = args.first().map(|s| s.as_str()).unwrap_or("World");
    Ok(format!("Hello, {name}!"))
}

fn cmd_add(args: &[String]) -> Result<String, String> {
    if args.len() < 2 {
        return Err("Usage: add <a> <b>".to_string());
    }
    let a: f64 = args[0].parse().map_err(|e| format!("Bad number: {e}"))?;
    let b: f64 = args[1].parse().map_err(|e| format!("Bad number: {e}"))?;
    Ok(format!("{a} + {b} = {}", a + b))
}

fn command_dispatch() {
    let mut commands: HashMap<&str, CmdFn> = HashMap::new();
    commands.insert("greet", cmd_greet);
    commands.insert("add", cmd_add);

    let test_cases: Vec<(&str, Vec<String>)> = vec![
        ("greet", vec!["Alice".into()]),
        ("add", vec!["3.14".into(), "2.86".into()]),
        ("add", vec!["bad".into(), "input".into()]),
        ("unknown", vec![]),
    ];

    for (cmd, args) in &test_cases {
        let result = match commands.get(cmd) {
            Some(f) => f(args),
            None => Err(format!("Unknown command: {cmd}")),
        };
        match result {
            Ok(msg) => println!("  [{cmd}] {msg}"),
            Err(e) => println!("  [{cmd}] ERROR: {e}"),
        }
    }
}

// --- Structured output ---

#[derive(Debug)]
struct Task {
    id: u32,
    title: String,
    done: bool,
    priority: Priority,
}

#[derive(Debug)]
enum Priority {
    Low,
    Medium,
    High,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Priority::Low => write!(f, "LOW"),
            Priority::Medium => write!(f, "MED"),
            Priority::High => write!(f, "HIGH"),
        }
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let check = if self.done { "✓" } else { " " };
        write!(f, "[{check}] #{:03} [{}] {}", self.id, self.priority, self.title)
    }
}

fn structured_output() {
    let tasks = vec![
        Task { id: 1, title: "Design API".into(), done: true, priority: Priority::High },
        Task { id: 2, title: "Write tests".into(), done: false, priority: Priority::Medium },
        Task { id: 3, title: "Update docs".into(), done: false, priority: Priority::Low },
    ];

    for task in &tasks {
        println!("  {task}");
    }
}

// --- Todo app simulation ---

struct TodoApp {
    tasks: Vec<Task>,
    next_id: u32,
}

impl TodoApp {
    fn new() -> Self {
        TodoApp { tasks: Vec::new(), next_id: 1 }
    }

    fn add(&mut self, title: &str, priority: Priority) -> u32 {
        let id = self.next_id;
        self.tasks.push(Task { id, title: title.to_string(), done: false, priority });
        self.next_id += 1;
        id
    }

    fn complete(&mut self, id: u32) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.done = true;
            true
        } else {
            false
        }
    }

    fn list(&self) -> &[Task] {
        &self.tasks
    }

    fn pending_count(&self) -> usize {
        self.tasks.iter().filter(|t| !t.done).count()
    }
}

fn todo_app_demo() {
    let mut app = TodoApp::new();

    app.add("Learn Rust", Priority::High);
    app.add("Build CLI tool", Priority::Medium);
    app.add("Publish crate", Priority::Low);

    app.complete(1);

    for task in app.list() {
        println!("  {task}");
    }
    println!("  Pending: {}", app.pending_count());
}

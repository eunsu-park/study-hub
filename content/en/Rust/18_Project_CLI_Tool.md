# 18. Project: CLI Tool

**Previous**: [Unsafe Rust](./17_Unsafe_Rust.md)

## Learning Objectives

After completing this project, you will be able to:

1. Structure a real-world Rust CLI application with multiple modules
2. Parse command-line arguments using `clap` with derive macros
3. Serialize and deserialize data structures with `serde` and `serde_json`
4. Handle errors idiomatically using the `anyhow` crate
5. Combine file I/O, data modeling, and user interaction into a cohesive application

---

This is a capstone project. Instead of learning individual concepts, you will build a complete **todo** command-line application from scratch. The tool lets users add tasks, list them, mark them as done, and remove them — all persisted to a JSON file on disk. Along the way, you will use patterns and crates that appear in virtually every production Rust CLI tool.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Setup](#2-project-setup)
3. [CLI Argument Parsing with Clap](#3-cli-argument-parsing-with-clap)
4. [Data Model](#4-data-model)
5. [File Persistence](#5-file-persistence)
6. [Implementing Commands](#6-implementing-commands)
7. [Error Handling with Anyhow](#7-error-handling-with-anyhow)
8. [Colored Output](#8-colored-output)
9. [Putting It All Together](#9-putting-it-all-together)
10. [Extension Ideas](#10-extension-ideas)

---

## 1. Project Overview

Here is what the finished tool looks like in action:

```
$ todo add "Buy groceries"
Added: Buy groceries (id: 1)

$ todo add "Write Rust project"
Added: Write Rust project (id: 2)

$ todo add "Call dentist"
Added: Call dentist (id: 3)

$ todo list
 ID  Status  Description
  1  [ ]     Buy groceries
  2  [ ]     Write Rust project
  3  [ ]     Call dentist

$ todo done 2
Completed: Write Rust project

$ todo list
 ID  Status  Description
  1  [ ]     Buy groceries
  2  [x]     Write Rust project
  3  [ ]     Call dentist

$ todo remove 1
Removed: Buy groceries

$ todo list
 ID  Status  Description
  2  [x]     Write Rust project
  3  [ ]     Call dentist
```

### Architecture

```
todo-cli/
├── Cargo.toml
└── src/
    ├── main.rs       ← entry point: parse args, dispatch commands
    ├── cli.rs         ← clap argument definitions
    ├── model.rs       ← Todo and TodoList data structures
    ├── storage.rs     ← JSON file read/write
    └── commands.rs    ← business logic for each command
```

---

## 2. Project Setup

Create the project and add dependencies:

```bash
cargo new todo-cli
cd todo-cli

cargo add clap --features derive     # argument parsing with derive macros
cargo add serde --features derive    # serialization framework
cargo add serde_json                 # JSON format support
cargo add anyhow                     # ergonomic error handling
cargo add colored                    # terminal colors (optional)
```

Your `Cargo.toml` should look like this:

```toml
[package]
name = "todo-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
colored = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

---

## 3. CLI Argument Parsing with Clap

Clap's derive macros let you define your CLI interface as plain Rust structs and enums. The library generates all the parsing, help text, and error messages automatically.

```rust
// src/cli.rs
use clap::{Parser, Subcommand};

/// A simple command-line todo manager.
///
/// Stores tasks in a JSON file and supports adding, listing,
/// completing, and removing items.
#[derive(Parser)]
#[command(name = "todo", version, about)]
pub struct Cli {
    /// Path to the todo storage file
    #[arg(short, long, default_value = "todos.json")]
    pub file: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Add a new todo item
    Add {
        /// Description of the task
        #[arg(required = true)]
        description: Vec<String>,
    },

    /// List all todo items
    List,

    /// Mark a todo item as completed
    Done {
        /// ID of the todo to complete
        id: u64,
    },

    /// Remove a todo item
    Remove {
        /// ID of the todo to remove
        id: u64,
    },
}
```

**Why clap derive?** Writing argument parsers by hand is tedious and error-prone. The derive approach keeps your CLI definition close to your data types, and clap generates `--help` output, error messages for invalid input, and shell completions for free.

---

## 4. Data Model

The data model is straightforward: a `Todo` holds one task, and a `TodoList` manages a collection of them. We derive `Serialize` and `Deserialize` so serde can convert these types to and from JSON automatically.

```rust
// src/model.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Todo {
    pub id: u64,
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct TodoList {
    // The next ID to assign — monotonically increasing so IDs
    // are never reused even after deletions
    next_id: u64,
    pub items: Vec<Todo>,
}

impl TodoList {
    pub fn new() -> Self {
        TodoList {
            next_id: 1,
            items: Vec::new(),
        }
    }

    /// Add a task and return the assigned ID.
    pub fn add(&mut self, description: String) -> u64 {
        let id = self.next_id;
        self.items.push(Todo {
            id,
            description,
            completed: false,
        });
        // Increment after use so the next task gets a fresh ID
        self.next_id += 1;
        id
    }

    /// Mark a task as completed. Returns the description if found.
    pub fn mark_done(&mut self, id: u64) -> Option<String> {
        // .iter_mut() gives mutable references so we can modify in place
        self.items.iter_mut().find(|t| t.id == id).map(|todo| {
            todo.completed = true;
            todo.description.clone()
        })
    }

    /// Remove a task by ID. Returns the description if found.
    pub fn remove(&mut self, id: u64) -> Option<String> {
        // Find the position first, then remove by index.
        // .position() returns Option<usize>.
        let pos = self.items.iter().position(|t| t.id == id)?;
        let todo = self.items.remove(pos);
        Some(todo.description)
    }

    /// Return true if the list has no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_find() {
        let mut list = TodoList::new();
        let id = list.add("Test task".into());
        assert_eq!(id, 1);
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].description, "Test task");
        assert!(!list.items[0].completed);
    }

    #[test]
    fn mark_done_existing() {
        let mut list = TodoList::new();
        list.add("Task".into());
        let result = list.mark_done(1);
        assert_eq!(result, Some("Task".into()));
        assert!(list.items[0].completed);
    }

    #[test]
    fn mark_done_nonexistent() {
        let mut list = TodoList::new();
        assert_eq!(list.mark_done(99), None);
    }

    #[test]
    fn remove_existing() {
        let mut list = TodoList::new();
        list.add("Task".into());
        let result = list.remove(1);
        assert_eq!(result, Some("Task".into()));
        assert!(list.is_empty());
    }

    #[test]
    fn ids_never_reuse() {
        let mut list = TodoList::new();
        let id1 = list.add("First".into());
        let id2 = list.add("Second".into());
        list.remove(id1);
        let id3 = list.add("Third".into());
        // id3 should be 3, not 1 (IDs are never recycled)
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }
}
```

---

## 5. File Persistence

Tasks are stored as a JSON file. We read the file on startup and write it back after every modification. The `anyhow` crate provides `Context` for adding descriptive messages to errors without defining custom error types.

```rust
// src/storage.rs
use std::path::Path;
use anyhow::{Context, Result};

use crate::model::TodoList;

/// Load a TodoList from a JSON file.
/// Returns an empty TodoList if the file does not exist.
pub fn load(path: &str) -> Result<TodoList> {
    let path = Path::new(path);

    if !path.exists() {
        // No file yet — start fresh. This is the expected state
        // on first run, so we do not treat it as an error.
        return Ok(TodoList::new());
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let list: TodoList = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;

    Ok(list)
}

/// Save a TodoList to a JSON file.
/// Uses pretty-printing so the file is human-readable.
pub fn save(path: &str, list: &TodoList) -> Result<()> {
    let contents = serde_json::to_string_pretty(list)
        .context("Failed to serialize TodoList to JSON")?;

    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path))?;

    Ok(())
}
```

The resulting JSON file looks like this:

```json
{
  "next_id": 4,
  "items": [
    {
      "id": 1,
      "description": "Buy groceries",
      "completed": false
    },
    {
      "id": 2,
      "description": "Write Rust project",
      "completed": true
    }
  ]
}
```

---

## 6. Implementing Commands

Each command is a function that takes the mutable `TodoList`, performs an operation, and prints feedback. Separating commands from the CLI parsing keeps the code testable — you can call these functions in tests without simulating command-line arguments.

```rust
// src/commands.rs
use colored::Colorize;
use crate::model::TodoList;

/// Add a new todo item. The description parts are joined with spaces
/// so users do not need to quote multi-word descriptions.
pub fn add(list: &mut TodoList, description: Vec<String>) {
    let desc = description.join(" ");
    let id = list.add(desc.clone());
    println!("{} {} (id: {})", "Added:".green().bold(), desc, id);
}

/// Display all todo items in a formatted table.
pub fn list_todos(list: &TodoList) {
    if list.is_empty() {
        println!("{}", "No todos yet. Add one with `todo add <description>`".dimmed());
        return;
    }

    // Print header
    println!(
        " {:>3}  {:6}  {}",
        "ID".bold(),
        "Status".bold(),
        "Description".bold()
    );

    for todo in &list.items {
        let status = if todo.completed {
            "[x]".green().to_string()
        } else {
            "[ ]".to_string()
        };

        let desc = if todo.completed {
            // Dim completed tasks so pending ones stand out
            todo.description.dimmed().to_string()
        } else {
            todo.description.clone()
        };

        println!(" {:>3}  {:6}  {}", todo.id, status, desc);
    }
}

/// Mark a todo as completed by ID.
pub fn done(list: &mut TodoList, id: u64) {
    match list.mark_done(id) {
        Some(desc) => println!("{} {}", "Completed:".green().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}

/// Remove a todo by ID.
pub fn remove(list: &mut TodoList, id: u64) {
    match list.remove(id) {
        Some(desc) => println!("{} {}", "Removed:".yellow().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}
```

---

## 7. Error Handling with Anyhow

In application code (as opposed to library code), `anyhow::Result` is the standard choice. It erases error types into a single `anyhow::Error`, which:

- Captures backtraces automatically
- Supports `.context("msg")` for adding human-readable descriptions
- Works with the `?` operator on any `std::error::Error` type
- Prints the full chain of causes when displayed

```rust
// The flow of errors in our application:
//
//   storage::load() → might fail (IO error, JSON parse error)
//        ↓
//   commands::*()   → infallible (print output directly)
//        ↓
//   storage::save() → might fail (IO error, JSON serialize error)
//        ↓
//   main()          → catches all errors, prints them, exits with code 1

// Example error output:
//   Error: Failed to parse JSON from todos.json
//
//   Caused by:
//       expected value at line 1 column 1
```

For **library** crates (code others depend on), prefer `thiserror` to define structured error enums so callers can match on specific variants. For **application** crates (binaries like this one), `anyhow` is simpler and sufficient.

---

## 8. Colored Output

The `colored` crate extends string types with methods like `.green()`, `.bold()`, and `.dimmed()`. It respects the `NO_COLOR` environment variable and detects non-TTY output automatically.

```rust
use colored::Colorize;

fn demo() {
    println!("{}", "Success!".green().bold());
    println!("{}", "Warning!".yellow());
    println!("{}", "Error!".red().bold());
    println!("{}", "Muted text".dimmed());

    // Chaining styles
    println!("{}", "Important".red().bold().underline());
}
```

If you prefer not to add this dependency, replace the colored calls with plain `println!`. The application works identically either way.

---

## 9. Putting It All Together

### 9.1 Main Entry Point

```rust
// src/main.rs
mod cli;
mod commands;
mod model;
mod storage;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load existing todos (or start fresh)
    let mut list = storage::load(&cli.file)?;

    // Dispatch to the appropriate command
    match cli.command {
        Command::Add { description } => commands::add(&mut list, description),
        Command::List => commands::list_todos(&list),
        Command::Done { id } => commands::done(&mut list, id),
        Command::Remove { id } => commands::remove(&mut list, id),
    }

    // Persist changes back to disk
    storage::save(&cli.file, &list)?;

    Ok(())
}
```

### 9.2 Complete File Listing

Here is every file in the project for reference. If you have been following along, you already have each one from the sections above.

**`src/cli.rs`** — CLI argument definitions (Section 3)

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "todo", version, about = "A simple command-line todo manager")]
pub struct Cli {
    #[arg(short, long, default_value = "todos.json")]
    pub file: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Add a new todo item
    Add {
        #[arg(required = true)]
        description: Vec<String>,
    },
    /// List all todo items
    List,
    /// Mark a todo item as completed
    Done { id: u64 },
    /// Remove a todo item
    Remove { id: u64 },
}
```

**`src/model.rs`** — Data structures (Section 4)

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Todo {
    pub id: u64,
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct TodoList {
    next_id: u64,
    pub items: Vec<Todo>,
}

impl TodoList {
    pub fn new() -> Self {
        TodoList {
            next_id: 1,
            items: Vec::new(),
        }
    }

    pub fn add(&mut self, description: String) -> u64 {
        let id = self.next_id;
        self.items.push(Todo {
            id,
            description,
            completed: false,
        });
        self.next_id += 1;
        id
    }

    pub fn mark_done(&mut self, id: u64) -> Option<String> {
        self.items.iter_mut().find(|t| t.id == id).map(|todo| {
            todo.completed = true;
            todo.description.clone()
        })
    }

    pub fn remove(&mut self, id: u64) -> Option<String> {
        let pos = self.items.iter().position(|t| t.id == id)?;
        let todo = self.items.remove(pos);
        Some(todo.description)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
```

**`src/storage.rs`** — File persistence (Section 5)

```rust
use std::path::Path;
use anyhow::{Context, Result};
use crate::model::TodoList;

pub fn load(path: &str) -> Result<TodoList> {
    let path = Path::new(path);
    if !path.exists() {
        return Ok(TodoList::new());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let list: TodoList = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;
    Ok(list)
}

pub fn save(path: &str, list: &TodoList) -> Result<()> {
    let contents = serde_json::to_string_pretty(list)
        .context("Failed to serialize TodoList to JSON")?;
    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path))?;
    Ok(())
}
```

**`src/commands.rs`** — Command implementations (Section 6)

```rust
use colored::Colorize;
use crate::model::TodoList;

pub fn add(list: &mut TodoList, description: Vec<String>) {
    let desc = description.join(" ");
    let id = list.add(desc.clone());
    println!("{} {} (id: {})", "Added:".green().bold(), desc, id);
}

pub fn list_todos(list: &TodoList) {
    if list.is_empty() {
        println!(
            "{}",
            "No todos yet. Add one with `todo add <description>`".dimmed()
        );
        return;
    }
    println!(
        " {:>3}  {:6}  {}",
        "ID".bold(),
        "Status".bold(),
        "Description".bold()
    );
    for todo in &list.items {
        let status = if todo.completed {
            "[x]".green().to_string()
        } else {
            "[ ]".to_string()
        };
        let desc = if todo.completed {
            todo.description.dimmed().to_string()
        } else {
            todo.description.clone()
        };
        println!(" {:>3}  {:6}  {}", todo.id, status, desc);
    }
}

pub fn done(list: &mut TodoList, id: u64) {
    match list.mark_done(id) {
        Some(desc) => println!("{} {}", "Completed:".green().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}

pub fn remove(list: &mut TodoList, id: u64) {
    match list.remove(id) {
        Some(desc) => println!("{} {}", "Removed:".yellow().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}
```

**`src/main.rs`** — Entry point

```rust
mod cli;
mod commands;
mod model;
mod storage;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut list = storage::load(&cli.file)?;

    match cli.command {
        Command::Add { description } => commands::add(&mut list, description),
        Command::List => commands::list_todos(&list),
        Command::Done { id } => commands::done(&mut list, id),
        Command::Remove { id } => commands::remove(&mut list, id),
    }

    storage::save(&cli.file, &list)?;
    Ok(())
}
```

### 9.3 Build and Test

```bash
# Build the project
cargo build

# Run tests (the model tests from section 4)
cargo test

# Try it out
cargo run -- add "Learn Rust"
cargo run -- add "Build a CLI tool"
cargo run -- list
cargo run -- done 1
cargo run -- list
cargo run -- remove 2
cargo run -- list

# Use a custom file path
cargo run -- --file work.json add "Ship feature"

# View help
cargo run -- --help
cargo run -- add --help

# Build a release binary and install it
cargo build --release
# The binary is at target/release/todo-cli
# Copy it to a directory in your PATH to use it globally
```

---

## 10. Extension Ideas

Once the basic tool works, try adding these features to deepen your Rust skills:

**Due dates** — Add an `Option<String>` (or `Option<NaiveDate>` with the `chrono` crate) to `Todo`. Add a `--due` flag to the `add` command. Highlight overdue tasks in red.

**Priority levels** — Add a `Priority` enum (`Low`, `Medium`, `High`) to `Todo`. Sort the list display by priority. Add `--priority` to the `add` command.

**Search and filter** — Add a `search` subcommand that filters todos by substring match. Add `--done` / `--pending` flags to `list` to show only completed or pending items.

**Export formats** — Add an `export` subcommand that outputs in CSV, Markdown table, or plain text formats. Use the `--format` flag to choose.

**Multiple lists** — Support named lists (`todo --list work add "Ship feature"`). Store them as separate JSON files or as a single file with a top-level map.

**Interactive mode** — Add a `tui` feature that launches an interactive terminal UI using the `ratatui` crate. Users navigate with arrow keys and toggle completion with Enter.

**Undo** — Keep a history of operations and support `todo undo` to reverse the last change. Store the history in the JSON file alongside the items.

---

## References
- [clap documentation](https://docs.rs/clap/latest/clap/)
- [serde documentation](https://serde.rs/)
- [anyhow documentation](https://docs.rs/anyhow/latest/anyhow/)
- [colored documentation](https://docs.rs/colored/latest/colored/)
- [Command Line Applications in Rust (book)](https://rust-cli.github.io/book/)

---

**Previous**: [Unsafe Rust](./17_Unsafe_Rust.md)

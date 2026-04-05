// Exercise: CLI Tool Project (Conceptual)
// Building a CLI tool requires a Cargo project with external crates.
// This file contains conceptual stubs and guided TODOs.
//
// To practice, create a new project as described in Lesson 18:
//   cargo new my-cli-tool && cd my-cli-tool

// ============================================================
// Project Setup Instructions
// ============================================================
// Add to Cargo.toml:
//   [dependencies]
//   clap = { version = "4", features = ["derive"] }
//   serde = { version = "1", features = ["derive"] }
//   serde_json = "1"
//   anyhow = "1"

// ============================================================
// Exercise 1: CLI Argument Parsing with clap Derive
// ============================================================
// In a real project, define your CLI struct like this:
//
// use clap::{Parser, Subcommand};
//
// #[derive(Parser)]
// #[command(name = "my-tool", version, about = "A sample CLI tool")]
// struct Cli {
//     /// Path to the configuration file
//     #[arg(short, long, default_value = "config.json")]
//     config: String,
//
//     /// Enable verbose output
//     #[arg(short, long)]
//     verbose: bool,
//
//     #[command(subcommand)]
//     command: Commands,
// }
//
// TODO: Add a --dry-run flag to Cli that prevents side effects.
// TODO: Add a --output-format flag that accepts "plain" or "json".

// ============================================================
// Exercise 2: Subcommands
// ============================================================
// In a real project, define subcommands like this:
//
// #[derive(Subcommand)]
// enum Commands {
//     /// Add a new item
//     Add {
//         #[arg(short, long)]
//         name: String,
//         #[arg(short, long)]
//         value: String,
//     },
//     /// List all items
//     List {
//         #[arg(short, long)]
//         filter: Option<String>,
//     },
//     /// Remove an item by name
//     Remove { name: String },
// }
//
// TODO: Add a Search subcommand with a `query: String` positional argument.
// TODO: Add an Export subcommand that accepts a file path and a format flag.

// ============================================================
// Exercise 3: Config File Handling
// ============================================================
// In a real project, read and write a JSON config like this:
//
// use serde::{Deserialize, Serialize};
// use anyhow::{Context, Result};
// use std::{fs, path::Path};
//
// #[derive(Debug, Serialize, Deserialize, Default)]
// struct Config {
//     items: Vec<Item>,
// }
//
// #[derive(Debug, Serialize, Deserialize)]
// struct Item {
//     name: String,
//     value: String,
// }
//
// fn load_config(path: &str) -> Result<Config> {
//     if !Path::new(path).exists() {
//         return Ok(Config::default());
//     }
//     let data = fs::read_to_string(path)
//         .with_context(|| format!("Failed to read config: {path}"))?;
//     serde_json::from_str(&data)
//         .with_context(|| format!("Failed to parse config: {path}"))
// }
//
// fn save_config(path: &str, config: &Config) -> Result<()> {
//     let json = serde_json::to_string_pretty(config)?;
//     fs::write(path, json)
//         .with_context(|| format!("Failed to write config: {path}"))?;
//     Ok(())
// }
//
// TODO: Add a `fn merge_configs(base: Config, override_: Config) -> Config` that
//       combines two configs, with override_ entries taking precedence by name.

// ============================================================
// Exercise 4: Main Dispatch Loop
// ============================================================
// In a real project, connect args, config, and subcommands:
//
// fn main() -> Result<()> {
//     let cli = Cli::parse();
//     let mut config = load_config(&cli.config)?;
//
//     match cli.command {
//         Commands::Add { name, value } => {
//             // TODO: Push a new Item into config.items and save the config.
//         }
//         Commands::List { filter } => {
//             // TODO: Print all items (filter by name substring if filter is Some).
//         }
//         Commands::Remove { name } => {
//             // TODO: Remove the item with the matching name; error if not found.
//         }
//     }
//     Ok(())
// }
//
// TODO: Respect the --verbose flag by printing extra diagnostics.
// TODO: Respect the --dry-run flag by skipping save_config calls.
// TODO: Respect the --output-format flag when printing list output.

// ============================================================
// Exercise 5: Conceptual Questions
// ============================================================
// Answer the following as comments:
//
// Q1: What is the difference between `#[arg]` and `#[command]` in clap's derive API?
// TODO: A1:
//
// Q2: Why is `anyhow::Result` preferred over `std::result::Result<(), Box<dyn Error>>`
//     for binary crates, but not usually recommended for library crates?
// TODO: A2:
//
// Q3: How would you add shell completion generation (e.g., bash, zsh) to a clap CLI?
// TODO: A3:

fn main() {
    println!("CLI Tool exercises are project-based.");
    println!("Create a Cargo project and implement the stubs above.");
    println!("See Lesson 18 for step-by-step instructions.");
}

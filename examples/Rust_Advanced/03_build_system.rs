// 03_build_system.rs — Cargo workspaces, features, and build configuration
//
// Run: rustc 03_build_system.rs && ./03_build_system
//
// Demonstrates: workspace patterns, feature flag simulation, and conditional compilation

fn main() {
    println!("=== Conditional Compilation ===");
    conditional_compilation();

    println!("\n=== Feature Flag Pattern ===");
    feature_flags();

    println!("\n=== Build Metadata ===");
    build_metadata();

    println!("\n=== Workspace Structure Example ===");
    workspace_layout();
}

// --- Conditional compilation with cfg ---

fn conditional_compilation() {
    // Platform detection
    #[cfg(target_os = "macos")]
    println!("  Running on macOS");

    #[cfg(target_os = "linux")]
    println!("  Running on Linux");

    #[cfg(target_os = "windows")]
    println!("  Running on Windows");

    // Architecture
    #[cfg(target_arch = "x86_64")]
    println!("  Architecture: x86_64");

    #[cfg(target_arch = "aarch64")]
    println!("  Architecture: aarch64");

    // Debug vs Release
    #[cfg(debug_assertions)]
    println!("  Build mode: debug");

    #[cfg(not(debug_assertions))]
    println!("  Build mode: release");

    // cfg! macro returns bool — usable in expressions
    let ptr_size = if cfg!(target_pointer_width = "64") { "64-bit" } else { "32-bit" };
    println!("  Pointer width: {ptr_size}");
}

// --- Feature flag simulation ---

mod logger {
    pub enum Level {
        Error,
        Warn,
        Info,
        Debug,
    }

    impl Level {
        pub fn label(&self) -> &str {
            match self {
                Level::Error => "ERROR",
                Level::Warn => "WARN",
                Level::Info => "INFO",
                Level::Debug => "DEBUG",
            }
        }
    }

    pub struct Logger {
        min_level: u8,
        use_color: bool,
    }

    impl Logger {
        /// Simulates a logger configured by feature flags:
        /// - "verbose" feature → show debug messages
        /// - "color" feature → colored output
        pub fn new(verbose: bool, color: bool) -> Self {
            Logger {
                min_level: if verbose { 0 } else { 2 },
                use_color: color,
            }
        }

        pub fn log(&self, level: Level, message: &str) {
            let level_num = match level {
                Level::Debug => 0,
                Level::Info => 1,
                Level::Warn => 2,
                Level::Error => 3,
            };

            if level_num < self.min_level {
                return;
            }

            let label = level.label();
            if self.use_color {
                let color = match level_num {
                    3 => "\x1b[31m", // red
                    2 => "\x1b[33m", // yellow
                    1 => "\x1b[32m", // green
                    _ => "\x1b[36m", // cyan
                };
                println!("  {color}[{label}]\x1b[0m {message}");
            } else {
                println!("  [{label}] {message}");
            }
        }
    }
}

fn feature_flags() {
    use logger::{Level, Logger};

    println!("  --- Default (no verbose, no color) ---");
    let log = Logger::new(false, false);
    log.log(Level::Debug, "This won't show");
    log.log(Level::Info, "This won't show either");
    log.log(Level::Warn, "Warning: disk space low");
    log.log(Level::Error, "Failed to connect");

    println!("  --- Verbose + Color ---");
    let log = Logger::new(true, true);
    log.log(Level::Debug, "Connecting to database...");
    log.log(Level::Info, "Server started on :8080");
    log.log(Level::Warn, "Deprecated API called");
    log.log(Level::Error, "Connection timeout");
}

// --- Build metadata ---

fn build_metadata() {
    // These environment variables are set by Cargo at build time
    // When compiling with rustc directly, they won't be set
    let pkg_name = option_env!("CARGO_PKG_NAME").unwrap_or("(rustc build)");
    let pkg_version = option_env!("CARGO_PKG_VERSION").unwrap_or("0.0.0");

    println!("  Package: {pkg_name}");
    println!("  Version: {pkg_version}");
    println!("  Rust version: {}", env!("CARGO_PKG_RUST_VERSION", "unknown"));

    // Compile-time assertions
    #[cfg(target_pointer_width = "64")]
    const _: () = assert!(std::mem::size_of::<usize>() == 8);
}

// --- Workspace layout documentation ---

fn workspace_layout() {
    let layout = r#"
    Cargo Workspace Structure:
    ┌─ Cargo.toml (workspace root)
    │   [workspace]
    │   members = ["core", "cli", "server"]
    │   [workspace.dependencies]
    │   serde = { version = "1", features = ["derive"] }
    │
    ├─ core/
    │   ├─ Cargo.toml        # [dependencies] serde.workspace = true
    │   └─ src/lib.rs         # Shared business logic
    │
    ├─ cli/
    │   ├─ Cargo.toml         # [dependencies] core = { path = "../core" }
    │   └─ src/main.rs        # CLI binary
    │
    └─ server/
        ├─ Cargo.toml         # [dependencies] core = { path = "../core" }
        └─ src/main.rs        # HTTP server binary

    Commands:
      cargo build -p cli       # Build just the CLI
      cargo test --workspace   # Test all members
      cargo run -p server      # Run the server
    "#;
    println!("{layout}");
}

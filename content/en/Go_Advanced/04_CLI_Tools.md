# 15. CLI Tools

**Previous**: [Database Access](./03_Database_Access.md) | **Next**: [Advanced Types](./05_Advanced_Types.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Parse command-line flags with the `flag` package
2. Build multi-command CLIs with `cobra`
3. Handle stdin/stdout for Unix-style piping
4. Create interactive terminal applications
5. Implement progress indicators and colored output

---

Go is the language of choice for CLI tools — Docker, Kubernetes (kubectl), Terraform, and Hugo are all Go CLI applications. Go's fast compilation, static binaries, and excellent cross-compilation make it ideal.

## Table of Contents
1. [flag Package](#1-flag-package)
2. [Cobra Framework](#2-cobra-framework)
3. [stdin/stdout and Piping](#3-stdinstdout-and-piping)
4. [Environment and Configuration](#4-environment-and-configuration)
5. [Terminal UI](#5-terminal-ui)
6. [Distribution](#6-distribution)
7. [Summary](#7-summary)

---

## 1. flag Package

### 1.1 Basic Flags

```go
package main

import (
    "flag"
    "fmt"
)

func main() {
    // Define flags
    name := flag.String("name", "World", "name to greet")
    count := flag.Int("count", 1, "number of greetings")
    verbose := flag.Bool("verbose", false, "enable verbose output")
    port := flag.Int("port", 8080, "server port")

    // Parse command-line arguments
    flag.Parse()

    // Use flags
    for i := 0; i < *count; i++ {
        fmt.Printf("Hello, %s!\n", *name)
    }

    if *verbose {
        fmt.Printf("Port: %d\n", *port)
    }

    // Remaining arguments (not flags)
    args := flag.Args()
    fmt.Println("Extra args:", args)
}
```

```bash
go run main.go -name Alice -count 3 -verbose extra1 extra2
```

### 1.2 Subcommands with FlagSet

```go
func main() {
    // Subcommand flag sets
    serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
    servePort := serveCmd.Int("port", 8080, "server port")
    serveHost := serveCmd.String("host", "localhost", "server host")

    initCmd := flag.NewFlagSet("init", flag.ExitOnError)
    initTemplate := initCmd.String("template", "default", "project template")

    if len(os.Args) < 2 {
        fmt.Println("Usage: app <command> [flags]")
        fmt.Println("Commands: serve, init")
        os.Exit(1)
    }

    switch os.Args[1] {
    case "serve":
        serveCmd.Parse(os.Args[2:])
        fmt.Printf("Serving on %s:%d\n", *serveHost, *servePort)
    case "init":
        initCmd.Parse(os.Args[2:])
        fmt.Printf("Initializing with template: %s\n", *initTemplate)
    default:
        fmt.Printf("Unknown command: %s\n", os.Args[1])
        os.Exit(1)
    }
}
```

---

## 2. Cobra Framework

### 2.1 Project Structure

```
myapp/
├── cmd/
│   ├── root.go
│   ├── serve.go
│   ├── init.go
│   └── version.go
├── internal/
│   └── ...
├── main.go
└── go.mod
```

### 2.2 Root Command

```go
// cmd/root.go
package cmd

import (
    "fmt"
    "os"

    "github.com/spf13/cobra"
)

var (
    cfgFile string
    verbose bool
)

var rootCmd = &cobra.Command{
    Use:   "myapp",
    Short: "A brief description of your application",
    Long:  `A longer description that spans multiple lines.`,
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}

func init() {
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
}
```

### 2.3 Subcommands

```go
// cmd/serve.go
package cmd

import (
    "fmt"
    "github.com/spf13/cobra"
)

var (
    port int
    host string
)

var serveCmd = &cobra.Command{
    Use:   "serve",
    Short: "Start the HTTP server",
    Long:  `Start the HTTP server with the specified configuration.`,
    RunE: func(cmd *cobra.Command, args []string) error {
        fmt.Printf("Starting server on %s:%d\n", host, port)
        return startServer(host, port)
    },
}

func init() {
    rootCmd.AddCommand(serveCmd)
    serveCmd.Flags().IntVarP(&port, "port", "p", 8080, "server port")
    serveCmd.Flags().StringVar(&host, "host", "localhost", "server host")
}
```

```go
// main.go
package main

import "github.com/username/myapp/cmd"

func main() {
    cmd.Execute()
}
```

---

## 3. stdin/stdout and Piping

### 3.1 Reading stdin

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

func main() {
    // Check if stdin is a pipe or terminal
    stat, _ := os.Stdin.Stat()
    isPipe := (stat.Mode() & os.ModeCharDevice) == 0

    if isPipe {
        // Reading from pipe: cat file.txt | myapp
        scanner := bufio.NewScanner(os.Stdin)
        lineNum := 0
        for scanner.Scan() {
            lineNum++
            line := scanner.Text()
            fmt.Printf("%4d: %s\n", lineNum, strings.ToUpper(line))
        }
        if err := scanner.Err(); err != nil {
            fmt.Fprintln(os.Stderr, "error reading stdin:", err)
            os.Exit(1)
        }
    } else {
        // Interactive mode
        fmt.Println("Enter text (Ctrl+D to finish):")
        scanner := bufio.NewScanner(os.Stdin)
        for scanner.Scan() {
            fmt.Println("You said:", scanner.Text())
        }
    }
}
```

### 3.2 stdout vs stderr

```go
func main() {
    // Regular output → stdout (piped)
    fmt.Println("This goes to stdout")
    fmt.Fprintln(os.Stdout, "Also stdout")

    // Errors and diagnostics → stderr (not piped)
    fmt.Fprintln(os.Stderr, "This goes to stderr")
    log.Println("Log goes to stderr by default")

    // Usage: myapp 2>/dev/null  — hide errors
    // Usage: myapp > output.txt — redirect stdout only
    // Usage: myapp 2>&1         — merge stderr into stdout
}
```

### 3.3 Exit Codes

```go
func main() {
    if err := run(); err != nil {
        fmt.Fprintln(os.Stderr, "error:", err)
        os.Exit(1) // Non-zero = failure
    }
    // os.Exit(0) is implicit on normal return
}

func run() error {
    // Actual application logic
    return nil
}

// Common exit codes:
// 0 — success
// 1 — general error
// 2 — usage error (wrong arguments)
// 126 — permission denied
// 127 — command not found
```

---

## 4. Environment and Configuration

### 4.1 Environment Variables

```go
func main() {
    // Read environment variables
    dbURL := os.Getenv("DATABASE_URL")
    if dbURL == "" {
        dbURL = "postgres://localhost:5432/myapp"
    }

    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    // LookupEnv distinguishes empty from unset
    val, exists := os.LookupEnv("API_KEY")
    if !exists {
        fmt.Println("API_KEY not set")
    } else if val == "" {
        fmt.Println("API_KEY is empty")
    }
}
```

### 4.2 Configuration Precedence

```go
type Config struct {
    Host     string
    Port     int
    LogLevel string
    DBUrl    string
}

func LoadConfig() *Config {
    cfg := &Config{
        Host:     "localhost",
        Port:     8080,
        LogLevel: "info",
    }

    // 1. Config file (lowest priority)
    if data, err := os.ReadFile("config.json"); err == nil {
        json.Unmarshal(data, cfg)
    }

    // 2. Environment variables (medium priority)
    if host := os.Getenv("APP_HOST"); host != "" {
        cfg.Host = host
    }
    if port := os.Getenv("APP_PORT"); port != "" {
        cfg.Port, _ = strconv.Atoi(port)
    }

    // 3. Command-line flags (highest priority)
    flag.StringVar(&cfg.Host, "host", cfg.Host, "server host")
    flag.IntVar(&cfg.Port, "port", cfg.Port, "server port")
    flag.Parse()

    return cfg
}
```

---

## 5. Terminal UI

### 5.1 Progress Indicator

```go
func spinner(done <-chan struct{}) {
    chars := `|/-\`
    i := 0
    for {
        select {
        case <-done:
            fmt.Print("\r \r") // Clear spinner
            return
        default:
            fmt.Printf("\r%c Processing...", chars[i%len(chars)])
            i++
            time.Sleep(100 * time.Millisecond)
        }
    }
}

func main() {
    done := make(chan struct{})
    go spinner(done)

    // Simulate work
    time.Sleep(3 * time.Second)

    close(done)
    fmt.Println("Done!")
}
```

### 5.2 Progress Bar

```go
func progressBar(current, total int, width int) string {
    percent := float64(current) / float64(total)
    filled := int(percent * float64(width))
    bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
    return fmt.Sprintf("\r[%s] %3.0f%% (%d/%d)", bar, percent*100, current, total)
}

func main() {
    total := 100
    for i := 0; i <= total; i++ {
        fmt.Print(progressBar(i, total, 40))
        time.Sleep(50 * time.Millisecond)
    }
    fmt.Println()
}
```

### 5.3 ANSI Colors

```go
const (
    Reset  = "\033[0m"
    Red    = "\033[31m"
    Green  = "\033[32m"
    Yellow = "\033[33m"
    Blue   = "\033[34m"
    Bold   = "\033[1m"
)

func success(msg string) {
    fmt.Printf("%s%s✓ %s%s\n", Green, Bold, msg, Reset)
}

func warning(msg string) {
    fmt.Printf("%s⚠ %s%s\n", Yellow, msg, Reset)
}

func errorMsg(msg string) {
    fmt.Printf("%s✗ %s%s\n", Red, msg, Reset)
}
```

---

## 6. Distribution

### 6.1 Cross-Compilation

```bash
# Build for Linux
GOOS=linux GOARCH=amd64 go build -o myapp-linux-amd64

# Build for macOS (Apple Silicon)
GOOS=darwin GOARCH=arm64 go build -o myapp-darwin-arm64

# Build for Windows
GOOS=windows GOARCH=amd64 go build -o myapp-windows-amd64.exe

# Common targets
# GOOS: linux, darwin, windows, freebsd
# GOARCH: amd64, arm64, 386, arm
```

### 6.2 Build Information

```go
// Set at build time with ldflags
var (
    version = "dev"
    commit  = "unknown"
    date    = "unknown"
)

func main() {
    if os.Args[1] == "version" {
        fmt.Printf("Version: %s\nCommit: %s\nDate: %s\n", version, commit, date)
        return
    }
}
```

```bash
go build -ldflags "-X main.version=1.0.0 -X main.commit=$(git rev-parse HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" -o myapp
```

---

## 7. Summary

### Key Takeaways

1. **`flag` for simple CLIs** — built-in, no dependencies. Good for single-command tools.
2. **Cobra for complex CLIs** — subcommands, auto-completion, help generation.
3. **Unix conventions** — stdout for data, stderr for diagnostics, exit codes for status.
4. **Configuration hierarchy** — defaults < config file < env vars < flags.
5. **Static binaries** — `CGO_ENABLED=0 go build` produces fully self-contained binaries.
6. **Cross-compilation** — `GOOS`/`GOARCH` environment variables. No cross-compiler needed.

---

## Exercises

### Exercise 1: File Utility
Build a CLI tool with subcommands: `count` (lines/words/chars), `find` (search files), `replace` (find and replace in files).

### Exercise 2: Task Manager
Create a task manager CLI with cobra: `add`, `list`, `done`, `delete` commands with JSON file persistence.

### Exercise 3: Unix Filter
Build a text processing filter that reads stdin, applies transformations (uppercase, lowercase, trim, wrap), and outputs to stdout.

### Exercise 4: Release Automation
Create a build script that cross-compiles for 3 platforms, includes version info via ldflags, and creates a release archive.

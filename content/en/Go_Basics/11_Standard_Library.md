# 11. Standard Library

**Previous**: [Testing](./10_Testing.md) | **Next**: [HTTP Server](../Go_Advanced/01_HTTP_Server.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `io` package interfaces for composable I/O operations
2. Parse and produce JSON with `encoding/json`
3. Work with files and directories using `os` and `filepath`
4. Make HTTP requests with `net/http`
5. Use `time`, `strings`, `bytes`, `regexp`, and `log/slog`

---

Go's standard library is famously comprehensive. It includes everything from HTTP servers to cryptography, without requiring any third-party packages. This lesson covers the most commonly used packages.

## Table of Contents
1. [io Package](#1-io-package)
2. [encoding/json](#2-encodingjson)
3. [os and filepath](#3-os-and-filepath)
4. [net/http Client](#4-nethttp-client)
5. [time Package](#5-time-package)
6. [strings, bytes, regexp](#6-strings-bytes-regexp)
7. [Summary](#7-summary)

---

## 1. io Package

### 1.1 Reader and Writer

```go
package main

import (
    "bytes"
    "fmt"
    "io"
    "os"
    "strings"
)

func main() {
    // strings.NewReader → io.Reader
    r := strings.NewReader("Hello, io!")

    // Copy reader to stdout (io.Writer)
    io.Copy(os.Stdout, r)
    fmt.Println()

    // ReadAll
    r2 := strings.NewReader("read everything")
    data, _ := io.ReadAll(r2)
    fmt.Println(string(data))

    // bytes.Buffer — implements both Reader and Writer
    var buf bytes.Buffer
    buf.WriteString("hello ")
    buf.WriteString("buffer")
    fmt.Println(buf.String())

    // MultiReader — concatenate readers
    r1 := strings.NewReader("part1 ")
    r3 := strings.NewReader("part2")
    multi := io.MultiReader(r1, r3)
    io.Copy(os.Stdout, multi)
    fmt.Println()

    // TeeReader — read and copy simultaneously
    original := strings.NewReader("tee data")
    var captured bytes.Buffer
    tee := io.TeeReader(original, &captured)
    io.Copy(os.Stdout, tee)
    fmt.Println()
    fmt.Println("Captured:", captured.String())

    // LimitReader — read only N bytes
    big := strings.NewReader("a very long string that we only want part of")
    limited := io.LimitReader(big, 10)
    data, _ = io.ReadAll(limited)
    fmt.Println(string(data)) // "a very lon"
}
```

### 1.2 Pipe

```go
func main() {
    pr, pw := io.Pipe()

    // Writer goroutine
    go func() {
        defer pw.Close()
        for i := 0; i < 5; i++ {
            fmt.Fprintf(pw, "line %d\n", i)
        }
    }()

    // Reader
    data, _ := io.ReadAll(pr)
    fmt.Println(string(data))
}
```

---

## 2. encoding/json

### 2.1 Marshal and Unmarshal

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email,omitempty"`
    CreatedAt time.Time `json:"created_at"`
    Password  string    `json:"-"` // Never included
}

func main() {
    // Marshal: Go struct → JSON bytes
    user := User{
        ID:        1,
        Name:      "Alice",
        Email:     "alice@example.com",
        CreatedAt: time.Now(),
        Password:  "secret",
    }

    data, err := json.Marshal(user)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println(string(data))

    // Pretty print
    pretty, _ := json.MarshalIndent(user, "", "  ")
    fmt.Println(string(pretty))

    // Unmarshal: JSON bytes → Go struct
    jsonStr := `{"id": 2, "name": "Bob", "created_at": "2024-01-15T10:30:00Z"}`
    var user2 User
    err = json.Unmarshal([]byte(jsonStr), &user2)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Printf("%+v\n", user2)
}
```

### 2.2 Streaming JSON

```go
func main() {
    // Encoder — write JSON to io.Writer
    encoder := json.NewEncoder(os.Stdout)
    encoder.SetIndent("", "  ")
    encoder.Encode(User{ID: 1, Name: "Alice"})

    // Decoder — read JSON from io.Reader
    jsonStream := `{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}`

    decoder := json.NewDecoder(strings.NewReader(jsonStream))
    for decoder.More() {
        var user User
        if err := decoder.Decode(&user); err != nil {
            fmt.Println("Error:", err)
            break
        }
        fmt.Printf("%+v\n", user)
    }
}
```

### 2.3 Dynamic JSON

```go
func main() {
    // Unmarshal into map for dynamic JSON
    jsonStr := `{"name": "Alice", "age": 30, "scores": [95, 87, 92]}`
    var data map[string]any
    json.Unmarshal([]byte(jsonStr), &data)

    fmt.Println(data["name"])   // "Alice"
    fmt.Println(data["age"])    // 30 (float64!)
    fmt.Println(data["scores"]) // [95 87 92]

    // json.RawMessage — defer parsing
    type Event struct {
        Type    string          `json:"type"`
        Payload json.RawMessage `json:"payload"`
    }

    eventJSON := `{"type": "user_created", "payload": {"id": 1, "name": "Alice"}}`
    var event Event
    json.Unmarshal([]byte(eventJSON), &event)

    // Parse payload based on type
    switch event.Type {
    case "user_created":
        var user User
        json.Unmarshal(event.Payload, &user)
        fmt.Printf("Created user: %+v\n", user)
    }
}
```

---

## 3. os and filepath

### 3.1 File Operations

```go
package main

import (
    "fmt"
    "os"
)

func main() {
    // Write file
    err := os.WriteFile("example.txt", []byte("Hello, file!\n"), 0644)
    if err != nil {
        fmt.Println("Write error:", err)
        return
    }

    // Read file
    data, err := os.ReadFile("example.txt")
    if err != nil {
        fmt.Println("Read error:", err)
        return
    }
    fmt.Println(string(data))

    // File with explicit open/close
    f, err := os.Create("output.txt")
    if err != nil {
        fmt.Println(err)
        return
    }
    defer f.Close()

    f.WriteString("line 1\n")
    f.WriteString("line 2\n")
    fmt.Fprintf(f, "line %d\n", 3)

    // Append to file
    f2, _ := os.OpenFile("output.txt", os.O_APPEND|os.O_WRONLY, 0644)
    defer f2.Close()
    f2.WriteString("line 4 (appended)\n")

    // Check existence
    if _, err := os.Stat("example.txt"); os.IsNotExist(err) {
        fmt.Println("File does not exist")
    }

    // Remove
    os.Remove("example.txt")
    os.Remove("output.txt")
}
```

### 3.2 Directory Operations

```go
func main() {
    // Create directory
    os.Mkdir("testdir", 0755)
    os.MkdirAll("nested/deep/dir", 0755)

    // Read directory entries
    entries, _ := os.ReadDir(".")
    for _, entry := range entries {
        info, _ := entry.Info()
        fmt.Printf("%-30s %10d %s\n", entry.Name(), info.Size(), info.ModTime().Format("2006-01-02"))
    }

    // Walk directory tree
    filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
        if err != nil {
            return err
        }
        fmt.Println(path)
        return nil
    })

    // Cleanup
    os.RemoveAll("testdir")
    os.RemoveAll("nested")
}
```

### 3.3 filepath Package

```go
import "path/filepath"

func main() {
    // Join paths (OS-aware)
    p := filepath.Join("home", "user", "documents", "file.txt")
    fmt.Println(p) // "home/user/documents/file.txt" on Unix

    // Extract components
    fmt.Println(filepath.Dir(p))   // "home/user/documents"
    fmt.Println(filepath.Base(p))  // "file.txt"
    fmt.Println(filepath.Ext(p))   // ".txt"

    // Clean path
    fmt.Println(filepath.Clean("a/b/../c")) // "a/c"

    // Absolute path
    abs, _ := filepath.Abs(".")
    fmt.Println(abs)

    // Glob pattern matching
    matches, _ := filepath.Glob("*.go")
    fmt.Println(matches)
}
```

---

## 4. net/http Client

### 4.1 Simple Requests

```go
func main() {
    // GET request
    resp, err := http.Get("https://httpbin.org/get")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    fmt.Println("Status:", resp.StatusCode)
    fmt.Println(string(body))

    // POST with JSON
    payload := bytes.NewBufferString(`{"name": "Alice"}`)
    resp, err = http.Post("https://httpbin.org/post", "application/json", payload)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer resp.Body.Close()
    body, _ = io.ReadAll(resp.Body)
    fmt.Println(string(body))
}
```

### 4.2 Custom Client

```go
func main() {
    client := &http.Client{
        Timeout: 10 * time.Second,
        Transport: &http.Transport{
            MaxIdleConns:        100,
            IdleConnTimeout:     90 * time.Second,
            MaxConnsPerHost:     10,
            MaxIdleConnsPerHost: 10,
        },
    }

    req, err := http.NewRequest("GET", "https://api.example.com/data", nil)
    if err != nil {
        fmt.Println(err)
        return
    }

    req.Header.Set("Authorization", "Bearer token123")
    req.Header.Set("Accept", "application/json")

    resp, err := client.Do(req)
    if err != nil {
        fmt.Println(err)
        return
    }
    defer resp.Body.Close()

    var result map[string]any
    json.NewDecoder(resp.Body).Decode(&result)
    fmt.Println(result)
}
```

---

## 5. time Package

### 5.1 Time Operations

```go
func main() {
    now := time.Now()
    fmt.Println("Now:", now)
    fmt.Println("Unix:", now.Unix())

    // Create specific time
    t := time.Date(2024, time.March, 15, 10, 30, 0, 0, time.UTC)
    fmt.Println(t)

    // Duration
    d := 2*time.Hour + 30*time.Minute
    fmt.Println(d)

    future := now.Add(d)
    fmt.Println("Future:", future)

    elapsed := time.Since(now)
    fmt.Println("Elapsed:", elapsed)

    // Formatting — Go uses reference time: Mon Jan 2 15:04:05 MST 2006
    fmt.Println(now.Format("2006-01-02 15:04:05"))
    fmt.Println(now.Format(time.RFC3339))
    fmt.Println(now.Format("January 2, 2006"))

    // Parsing
    parsed, _ := time.Parse("2006-01-02", "2024-03-15")
    fmt.Println(parsed)

    // Timer and Ticker
    timer := time.NewTimer(2 * time.Second)
    <-timer.C
    fmt.Println("Timer fired")

    ticker := time.NewTicker(500 * time.Millisecond)
    defer ticker.Stop()
    for i := 0; i < 3; i++ {
        <-ticker.C
        fmt.Println("Tick")
    }
}
```

---

## 6. strings, bytes, regexp

### 6.1 strings Package

```go
import "strings"

func main() {
    s := "Hello, World!"

    fmt.Println(strings.Contains(s, "World"))     // true
    fmt.Println(strings.HasPrefix(s, "Hello"))     // true
    fmt.Println(strings.HasSuffix(s, "!"))         // true
    fmt.Println(strings.Index(s, "World"))         // 7
    fmt.Println(strings.Count(s, "l"))             // 3
    fmt.Println(strings.Repeat("Go ", 3))          // "Go Go Go "
    fmt.Println(strings.TrimSpace("  hello  "))    // "hello"
    fmt.Println(strings.ReplaceAll(s, "l", "L"))   // "HeLLo, WorLd!"

    // Split and Join
    parts := strings.Split("a,b,c,d", ",")
    fmt.Println(parts) // [a b c d]
    fmt.Println(strings.Join(parts, " | ")) // "a | b | c | d"

    // strings.Builder
    var b strings.Builder
    for i := 0; i < 1000; i++ {
        fmt.Fprintf(&b, "%d ", i)
    }
    result := b.String()
    _ = result
}
```

### 6.2 regexp Package

```go
import "regexp"

func main() {
    // Compile once, reuse many times
    re := regexp.MustCompile(`\b\w+@\w+\.\w+\b`)

    text := "Contact alice@example.com or bob@test.org"

    // Find all matches
    matches := re.FindAllString(text, -1)
    fmt.Println(matches) // [alice@example.com bob@test.org]

    // Replace
    masked := re.ReplaceAllString(text, "[EMAIL]")
    fmt.Println(masked)

    // Submatch (capture groups)
    re2 := regexp.MustCompile(`(\w+)@(\w+)\.(\w+)`)
    parts := re2.FindStringSubmatch("alice@example.com")
    fmt.Println(parts) // [alice@example.com alice example com]

    // Named groups
    re3 := regexp.MustCompile(`(?P<user>\w+)@(?P<domain>\w+\.\w+)`)
    match := re3.FindStringSubmatch("alice@example.com")
    for i, name := range re3.SubexpNames() {
        if name != "" {
            fmt.Printf("%s: %s\n", name, match[i])
        }
    }
}
```

### 6.3 log/slog (Go 1.21+)

```go
import "log/slog"

func main() {
    // Structured logging
    slog.Info("user logged in",
        "user_id", 123,
        "ip", "192.168.1.1",
    )

    // JSON handler
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
    }))

    logger.Debug("debug message", "key", "value")
    logger.Info("request processed",
        "method", "GET",
        "path", "/api/users",
        "duration_ms", 42,
    )
    logger.Error("database error",
        "err", fmt.Errorf("connection refused"),
        "retry", 3,
    )

    // Logger with default fields
    childLogger := logger.With("service", "auth", "version", "1.0")
    childLogger.Info("token created", "user_id", 42)
}
```

---

## 7. Summary

### Key Takeaways

1. **io.Reader and io.Writer** — the foundation of composable I/O. Most packages accept these interfaces.
2. **encoding/json** — struct tags control marshaling. Use `json.RawMessage` for deferred parsing.
3. **os.ReadFile / os.WriteFile** — simple file operations. Use `os.Open` for streaming large files.
4. **net/http.Client** — always set timeouts. Reuse clients for connection pooling.
5. **time layout uses reference time** — `2006-01-02 15:04:05` is Go's unique formatting approach.
6. **strings.Builder** — efficient string concatenation. Never use `+` in loops.
7. **log/slog** — structured logging built into Go 1.21+. Use instead of `log` package.

---

## Exercises

### Exercise 1: JSON Config Loader
Write a config loader that reads JSON from a file, validates required fields, and returns a typed Config struct. Support environment variable overrides.

### Exercise 2: File Watcher
Create a utility that watches a directory for changes (new/modified/deleted files) and prints events with timestamps.

### Exercise 3: HTTP Client Wrapper
Build a reusable HTTP client with retry logic, timeout, custom headers, and JSON response decoding.

### Exercise 4: Log Analyzer
Write a program that reads structured log files (JSON lines), filters by level and time range, and produces a summary report.

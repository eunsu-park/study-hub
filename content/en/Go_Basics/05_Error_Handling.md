# 05. Error Handling

**Previous**: [Interfaces](./04_Interfaces.md) | **Next**: [Packages and Modules](./06_Packages_and_Modules.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the `error` interface and create custom error types
2. Wrap and unwrap errors with `fmt.Errorf` and `%w`
3. Use `errors.Is` and `errors.As` for error inspection
4. Define sentinel errors and error type hierarchies
5. Apply `panic` and `recover` appropriately

---

Go famously eschews exceptions in favor of explicit error values. Every function that can fail returns an `error` as its last return value. This approach is verbose but makes error paths visible and forces developers to think about failure at every step.

## Table of Contents
1. [The error Interface](#1-the-error-interface)
2. [Creating Errors](#2-creating-errors)
3. [Error Wrapping](#3-error-wrapping)
4. [Sentinel Errors](#4-sentinel-errors)
5. [Custom Error Types](#5-custom-error-types)
6. [Panic and Recover](#6-panic-and-recover)
7. [Summary](#7-summary)

---

## 1. The error Interface

### 1.1 Error Basics

```go
package main

import (
    "errors"
    "fmt"
    "os"
    "strconv"
)

func main() {
    // Most functions return (result, error)
    f, err := os.Open("nonexistent.txt")
    if err != nil {
        fmt.Println("Error:", err)
        // Don't use f — it's invalid when err != nil
    } else {
        defer f.Close()
        fmt.Println("Opened:", f.Name())
    }

    // Always check errors — don't use _ unless you have a reason
    n, err := strconv.Atoi("not-a-number")
    if err != nil {
        fmt.Println("Parse error:", err)
        return
    }
    fmt.Println("Parsed:", n)
}
```

### 1.2 Error Patterns

```go
// Pattern 1: Return early on error (guard clause)
func readConfig(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }

    var cfg Config
    err = json.Unmarshal(data, &cfg)
    if err != nil {
        return Config{}, err
    }

    return cfg, nil
}

// Pattern 2: Multiple error checks in sequence
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("open: %w", err)
    }
    defer f.Close()

    data, err := io.ReadAll(f)
    if err != nil {
        return fmt.Errorf("read: %w", err)
    }

    if err := validate(data); err != nil {
        return fmt.Errorf("validate: %w", err)
    }

    return nil
}

// Pattern 3: Error variable with defer
func writeFile(path string, data []byte) (err error) {
    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer func() {
        closeErr := f.Close()
        if err == nil {
            err = closeErr
        }
    }()

    _, err = f.Write(data)
    return err
}
```

---

## 2. Creating Errors

### 2.1 Simple Errors

```go
import (
    "errors"
    "fmt"
)

func divide(a, b float64) (float64, error) {
    if b == 0 {
        // errors.New — simple static error message
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func sqrt(x float64) (float64, error) {
    if x < 0 {
        // fmt.Errorf — formatted error message
        return 0, fmt.Errorf("cannot take sqrt of negative number: %g", x)
    }
    return math.Sqrt(x), nil
}

func main() {
    result, err := divide(10, 0)
    if err != nil {
        fmt.Println(err) // "division by zero"
    }

    result, err = sqrt(-1)
    if err != nil {
        fmt.Println(err) // "cannot take sqrt of negative number: -1"
    }
    _ = result
}
```

### 2.2 Error Message Conventions

```go
// Good: lowercase, no punctuation, include context
return fmt.Errorf("open config: %w", err)
return fmt.Errorf("parse port %q: %w", portStr, err)
return errors.New("missing required field: email")

// Bad: uppercase, punctuation, vague
return fmt.Errorf("Error opening file.")        // BAD
return errors.New("Something went wrong!")       // BAD
return fmt.Errorf("Failed to process request")   // BAD

// Error messages form chains: "open config: parse port "abc": strconv.Atoi: invalid syntax"
```

---

## 3. Error Wrapping

### 3.1 Wrapping with %w

Go 1.13 introduced error wrapping with `%w` in `fmt.Errorf`.

```go
func readFile(path string) ([]byte, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("readFile(%s): %w", path, err)
    }
    return data, nil
}

func loadConfig(path string) (*Config, error) {
    data, err := readFile(path)
    if err != nil {
        return nil, fmt.Errorf("loadConfig: %w", err)
    }

    var cfg Config
    if err := json.Unmarshal(data, &cfg); err != nil {
        return nil, fmt.Errorf("loadConfig: parse json: %w", err)
    }
    return &cfg, nil
}

func main() {
    cfg, err := loadConfig("/nonexistent/config.json")
    if err != nil {
        fmt.Println(err)
        // "loadConfig: readFile(/nonexistent/config.json): open /nonexistent/config.json: no such file or directory"
    }
    _ = cfg
}
```

### 3.2 errors.Is — Checking Error Identity

```go
func main() {
    _, err := os.Open("/nonexistent")

    // errors.Is traverses the wrap chain
    if errors.Is(err, os.ErrNotExist) {
        fmt.Println("File not found!")
    }

    // Works through multiple layers of wrapping
    wrapped := fmt.Errorf("config: %w",
        fmt.Errorf("read: %w", os.ErrNotExist))
    fmt.Println(errors.Is(wrapped, os.ErrNotExist)) // true
}
```

### 3.3 errors.As — Extracting Error Types

```go
type PathError struct {
    Op   string
    Path string
    Err  error
}

func (e *PathError) Error() string {
    return fmt.Sprintf("%s %s: %s", e.Op, e.Path, e.Err)
}

func (e *PathError) Unwrap() error {
    return e.Err
}

func main() {
    _, err := os.Open("/nonexistent")

    // Extract the specific error type
    var pathErr *os.PathError
    if errors.As(err, &pathErr) {
        fmt.Println("Op:", pathErr.Op)
        fmt.Println("Path:", pathErr.Path)
        fmt.Println("Underlying:", pathErr.Err)
    }
}
```

### 3.4 Multiple Wrapping (Go 1.20+)

```go
// Join multiple errors
func validateForm(name, email string) error {
    var errs []error

    if name == "" {
        errs = append(errs, errors.New("name is required"))
    }
    if email == "" {
        errs = append(errs, errors.New("email is required"))
    }
    if !strings.Contains(email, "@") {
        errs = append(errs, errors.New("email must contain @"))
    }

    if len(errs) > 0 {
        return errors.Join(errs...)
    }
    return nil
}

func main() {
    err := validateForm("", "invalid")
    if err != nil {
        fmt.Println(err)
        // "name is required\nemail must contain @"
    }
}
```

---

## 4. Sentinel Errors

### 4.1 Defining Sentinel Errors

Sentinel errors are predefined error values used for comparison.

```go
package mypackage

import "errors"

// Sentinel errors — package-level variables
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrConflict     = errors.New("conflict: resource already exists")
    ErrInternal     = errors.New("internal server error")
)

type UserStore struct {
    users map[string]*User
}

func (s *UserStore) Get(id string) (*User, error) {
    u, ok := s.users[id]
    if !ok {
        return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
    }
    return u, nil
}

// Caller checks with errors.Is
func main() {
    store := &UserStore{users: make(map[string]*User)}

    _, err := store.Get("abc123")
    if errors.Is(err, ErrNotFound) {
        fmt.Println("User not found, creating...")
    }

    switch {
    case errors.Is(err, ErrNotFound):
        // Handle 404
    case errors.Is(err, ErrUnauthorized):
        // Handle 401
    case err != nil:
        // Handle unexpected error
    }
}
```

### 4.2 Standard Library Sentinels

```go
import (
    "io"
    "os"
    "database/sql"
)

// Common sentinel errors from standard library:
// io.EOF            — end of input
// os.ErrNotExist    — file not found
// os.ErrPermission  — permission denied
// sql.ErrNoRows     — query returned no rows

func readUntilEOF(r io.Reader) error {
    buf := make([]byte, 1024)
    for {
        _, err := r.Read(buf)
        if errors.Is(err, io.EOF) {
            return nil // Normal end of input
        }
        if err != nil {
            return err // Actual error
        }
    }
}
```

---

## 5. Custom Error Types

### 5.1 Struct-Based Errors

```go
type HTTPError struct {
    Code    int
    Message string
    Details map[string]string
}

func (e *HTTPError) Error() string {
    return fmt.Sprintf("HTTP %d: %s", e.Code, e.Message)
}

func NewHTTPError(code int, msg string) *HTTPError {
    return &HTTPError{Code: code, Message: msg}
}

func fetchUser(id string) (*User, error) {
    if id == "" {
        return nil, &HTTPError{
            Code:    400,
            Message: "user ID is required",
            Details: map[string]string{"field": "id"},
        }
    }
    // ... fetch from database
    return nil, &HTTPError{Code: 404, Message: "user not found"}
}

func main() {
    _, err := fetchUser("")
    if err != nil {
        var httpErr *HTTPError
        if errors.As(err, &httpErr) {
            fmt.Printf("Status %d: %s\n", httpErr.Code, httpErr.Message)
            fmt.Println("Details:", httpErr.Details)
        }
    }
}
```

### 5.2 Error Type with Unwrap

```go
type QueryError struct {
    Query string
    Err   error
}

func (e *QueryError) Error() string {
    return fmt.Sprintf("query %q: %v", e.Query, e.Err)
}

func (e *QueryError) Unwrap() error {
    return e.Err
}

func runQuery(q string) error {
    // Simulate a connection error
    return &QueryError{
        Query: q,
        Err:   fmt.Errorf("connection refused"),
    }
}
```

### 5.3 Error Handling Strategy

```go
// Layer 1: Low-level — return wrapped errors
func readFromDB(id string) ([]byte, error) {
    row := db.QueryRow("SELECT data FROM items WHERE id = ?", id)
    var data []byte
    if err := row.Scan(&data); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("item %s: %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("query item %s: %w", id, err)
    }
    return data, nil
}

// Layer 2: Business logic — add context
func GetItem(id string) (*Item, error) {
    data, err := readFromDB(id)
    if err != nil {
        return nil, fmt.Errorf("GetItem: %w", err)
    }
    return parseItem(data)
}

// Layer 3: HTTP handler — map to status codes
func handleGetItem(w http.ResponseWriter, r *http.Request) {
    item, err := GetItem(r.URL.Query().Get("id"))
    if err != nil {
        switch {
        case errors.Is(err, ErrNotFound):
            http.Error(w, "Not Found", 404)
        case errors.Is(err, ErrUnauthorized):
            http.Error(w, "Unauthorized", 401)
        default:
            log.Printf("unexpected error: %v", err)
            http.Error(w, "Internal Server Error", 500)
        }
        return
    }
    json.NewEncoder(w).Encode(item)
}
```

---

## 6. Panic and Recover

### 6.1 Panic

`panic` is for unrecoverable errors — programming bugs, not operational errors.

```go
func main() {
    // Appropriate uses of panic:
    // 1. Truly impossible states (programming bugs)
    // 2. Initialization failures in init() or main()

    // Example: impossible state
    switch dayOfWeek {
    case 0: // Sunday
    case 1: // Monday
    // ...
    default:
        panic(fmt.Sprintf("invalid day: %d", dayOfWeek))
    }

    // Example: must-succeed initialization
    re := regexp.MustCompile(`\d+`) // panics if regex is invalid
    template.Must(template.New("t").Parse("{{.Name}}")) // panics on parse error
}

// Don't use panic for:
// - File not found → return error
// - Network timeout → return error
// - Invalid user input → return error
```

### 6.2 Recover

```go
func safeDiv(a, b int) (result int, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()
    return a / b, nil // panics if b == 0
}

func main() {
    result, err := safeDiv(10, 0)
    if err != nil {
        fmt.Println("Recovered:", err)
    } else {
        fmt.Println("Result:", result)
    }
}
```

### 6.3 Recover in Servers

```go
// Middleware to prevent one panic from crashing the whole server
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("PANIC: %v\n%s", err, debug.Stack())
                http.Error(w, "Internal Server Error", 500)
            }
        }()
        next.ServeHTTP(w, r)
    })
}
```

---

## 7. Summary

### Key Takeaways

1. **Errors are values** — the `error` interface has one method: `Error() string`. Return errors explicitly.
2. **Check every error** — `if err != nil` is Go's most common pattern. Never ignore errors silently.
3. **Wrap with context** — `fmt.Errorf("operation: %w", err)` preserves the chain and adds meaning.
4. **`errors.Is` for sentinels** — compare through wrap chains: `errors.Is(err, ErrNotFound)`.
5. **`errors.As` for types** — extract typed errors: `errors.As(err, &httpErr)`.
6. **Sentinel errors for expected failures** — `ErrNotFound`, `io.EOF`, `sql.ErrNoRows`.
7. **Panic only for bugs** — use `panic` for impossible states, never for operational errors.

### Error Decision Tree

```
Is this a programming bug?
├── Yes → panic (nil map access, impossible enum value)
└── No → return error
     │
     Is this an expected failure?
     ├── Yes → sentinel error (ErrNotFound, ErrTimeout)
     └── No → wrap with context: fmt.Errorf("op: %w", err)
```

---

## Exercises

### Exercise 1: Validation Library
Create a `ValidationError` type that collects multiple field errors. Implement `Error()`, `Unwrap()`, and a `HasField(name string) bool` method.

### Exercise 2: File Processor
Write a file processing function chain (open, read, parse, validate) where each step wraps errors with context. Test with `errors.Is` and `errors.As`.

### Exercise 3: Retry with Errors
Implement a `Retry(attempts int, delay time.Duration, fn func() error) error` function that retries on failure and returns the last error with all attempt information.

### Exercise 4: Error Middleware
Create an HTTP error handling middleware that converts custom error types to appropriate HTTP status codes and JSON error responses.

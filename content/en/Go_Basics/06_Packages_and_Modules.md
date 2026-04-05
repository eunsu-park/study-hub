# 06. Packages and Modules

**Previous**: [Error Handling](./05_Error_Handling.md) | **Next**: [Concurrency: Goroutines](./07_Concurrency_Goroutines.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and organize Go packages following standard conventions
2. Use Go modules for dependency management
3. Understand visibility rules (exported vs unexported)
4. Apply package design principles for clean APIs
5. Use `go mod` commands for dependency management and versioning

---

Go's module system provides reproducible builds and explicit dependency management. Packages organize code into logical units, while modules group related packages and manage their versions.

## Table of Contents
1. [Package Fundamentals](#1-package-fundamentals)
2. [Visibility and Naming](#2-visibility-and-naming)
3. [Go Modules](#3-go-modules)
4. [Dependency Management](#4-dependency-management)
5. [Package Design Patterns](#5-package-design-patterns)
6. [Internal Packages and Workspaces](#6-internal-packages-and-workspaces)
7. [Summary](#7-summary)

---

## 1. Package Fundamentals

### 1.1 Package Declaration

Every Go file starts with a package declaration. Files in the same directory must use the same package name.

```go
// file: mathutil/math.go
package mathutil

func Add(a, b int) int { return a + b }
func Sub(a, b int) int { return a - b }

// unexported helper — only visible within this package
func abs(n int) int {
    if n < 0 {
        return -n
    }
    return n
}
```

```go
// file: mathutil/stats.go
package mathutil // Same package — same directory

func Mean(nums []float64) float64 {
    if len(nums) == 0 {
        return 0
    }
    sum := 0.0
    for _, n := range nums {
        sum += n
    }
    return sum / float64(len(nums))
}
```

### 1.2 Importing Packages

```go
package main

import (
    "fmt"
    "math/rand"
    "strings"

    // Third-party packages
    "github.com/gorilla/mux"

    // Local packages
    "github.com/username/myproject/mathutil"
    "github.com/username/myproject/internal/config"
)

func main() {
    fmt.Println(mathutil.Add(3, 4))
    fmt.Println(mathutil.Mean([]float64{1, 2, 3, 4, 5}))
}
```

### 1.3 Import Grouping Convention

```go
import (
    // Group 1: Standard library
    "fmt"
    "net/http"
    "os"

    // Group 2: Third-party packages (blank line separator)
    "github.com/gorilla/mux"
    "go.uber.org/zap"

    // Group 3: Internal/local packages (blank line separator)
    "github.com/myorg/myproject/internal/config"
    "github.com/myorg/myproject/pkg/auth"
)
```

### 1.4 Import Aliases and Special Imports

```go
import (
    "fmt"

    // Alias — resolve name conflicts
    crand "crypto/rand"
    mrand "math/rand"

    // Dot import — imports into current namespace (avoid in production)
    . "math"

    // Blank import — execute init() only (side effects)
    _ "github.com/lib/pq"           // Register PostgreSQL driver
    _ "image/png"                     // Register PNG decoder
    _ "net/http/pprof"                // Register pprof HTTP handlers
)

func main() {
    fmt.Println(Sqrt(2))            // math.Sqrt via dot import
    fmt.Println(mrand.Intn(100))    // math/rand
    // crand.Read(...)              // crypto/rand
}
```

---

## 2. Visibility and Naming

### 2.1 Export Rules

```go
package user

// EXPORTED (uppercase first letter) — visible to other packages
type User struct {
    ID    int     // Exported field
    Name  string  // Exported field
    email string  // unexported field
}

func NewUser(name, email string) *User {  // Exported function
    return &User{
        ID:    generateID(), // Can call unexported function internally
        Name:  name,
        email: email,
    }
}

func (u *User) Email() string { return u.email }  // Exported method (getter)

// unexported — only visible within this package
func generateID() int {
    // ...
    return 0
}

type validator struct { // unexported type
    rules []Rule
}
```

### 2.2 Package Naming Conventions

```go
// GOOD: Short, lowercase, single word
package http
package json
package user
package auth
package config

// BAD: Verbose, mixed case, generic
package httpHelpers    // BAD: camelCase
package common         // BAD: too generic
package utils          // BAD: too generic — what utils?
package base           // BAD: meaningless

// Package name is part of the qualified name
// GOOD:
http.Get(url)          // reads as "HTTP get"
json.Marshal(v)        // reads as "JSON marshal"
auth.NewToken()        // reads as "auth new token"

// BAD: stuttering
http.HTTPGet(url)      // "HTTP HTTP get"
user.UserCreate()      // "user user create"
```

### 2.3 init() Functions

```go
package config

import (
    "log"
    "os"
)

var (
    DatabaseURL string
    Port        int
)

// init runs automatically when the package is imported
// Multiple init() functions per file are allowed (but discouraged)
func init() {
    DatabaseURL = os.Getenv("DATABASE_URL")
    if DatabaseURL == "" {
        DatabaseURL = "postgres://localhost:5432/mydb"
    }

    portStr := os.Getenv("PORT")
    if portStr == "" {
        Port = 8080
    } else {
        var err error
        Port, err = strconv.Atoi(portStr)
        if err != nil {
            log.Fatalf("invalid PORT: %s", portStr)
        }
    }
}
```

---

## 3. Go Modules

### 3.1 Module Initialization

```bash
# Create a new module
mkdir myproject && cd myproject
go mod init github.com/username/myproject

# go.mod is created
cat go.mod
```

```
module github.com/username/myproject

go 1.22
```

### 3.2 go.mod File

```
module github.com/username/myproject

go 1.22

require (
    github.com/gorilla/mux v1.8.1
    go.uber.org/zap v1.27.0
    golang.org/x/sync v0.6.0
)

require (
    // Indirect dependencies (managed automatically)
    go.uber.org/multierr v1.11.0 // indirect
)
```

### 3.3 go.sum File

```bash
# go.sum contains cryptographic hashes for verification
# NEVER edit manually — managed by go tools
# ALWAYS commit go.sum to version control

cat go.sum
# github.com/gorilla/mux v1.8.1 h1:TuMoUvkRETex...
# github.com/gorilla/mux v1.8.1/go.mod h1:DVbg23sW...
```

### 3.4 Module Commands

```bash
# Add a dependency
go get github.com/gorilla/mux@latest
go get github.com/gorilla/mux@v1.8.1     # Specific version
go get github.com/gorilla/mux@v1.8       # Latest patch

# Update dependencies
go get -u ./...                            # Update all direct dependencies
go get -u=patch ./...                      # Patch updates only

# Tidy — remove unused, add missing
go mod tidy

# Vendor — copy dependencies locally
go mod vendor
go build -mod=vendor ./...

# Download dependencies (for CI caching)
go mod download

# Check for module graph issues
go mod verify

# Show dependency graph
go mod graph

# Show why a dependency is needed
go mod why github.com/some/package

# Edit go.mod programmatically
go mod edit -require github.com/foo/bar@v1.0.0
go mod edit -droprequire github.com/foo/bar
```

---

## 4. Dependency Management

### 4.1 Semantic Versioning

```
v1.2.3
│ │ └── Patch: bug fixes, no API changes
│ └──── Minor: new features, backward compatible
└────── Major: breaking changes

v0.x.y — Pre-1.0: no stability guarantees
v2.0.0 — Major version in import path: github.com/user/pkg/v2
```

### 4.2 Version Selection

```go
// Go uses Minimum Version Selection (MVS)
// If A requires X v1.2.0 and B requires X v1.3.0,
// Go selects X v1.3.0 (the minimum version satisfying both)

// Major version suffixes for v2+
import "github.com/user/pkg/v2"        // v2.x.x
import "github.com/user/pkg/v3"        // v3.x.x
// v0 and v1 have no suffix
import "github.com/user/pkg"           // v0.x.x or v1.x.x
```

### 4.3 Replace and Exclude Directives

```
// go.mod

// Replace a dependency with a local copy (for development)
replace github.com/user/pkg => ../pkg

// Replace with a fork
replace github.com/original/pkg => github.com/myfork/pkg v1.2.3

// Exclude a buggy version
exclude github.com/user/pkg v1.2.0
```

### 4.4 Private Modules

```bash
# For private repositories
export GOPRIVATE=github.com/mycompany/*
export GONOSUMDB=github.com/mycompany/*
export GONOPROXY=github.com/mycompany/*

# Or in go.env
go env -w GOPRIVATE=github.com/mycompany/*
```

---

## 5. Package Design Patterns

### 5.1 Standard Project Layout

```
myproject/
├── go.mod
├── go.sum
├── main.go              # Entry point (package main)
├── cmd/                 # Multiple entry points
│   ├── server/
│   │   └── main.go      # go run ./cmd/server
│   └── cli/
│       └── main.go      # go run ./cmd/cli
├── internal/            # Private packages (compiler-enforced)
│   ├── config/
│   ├── database/
│   └── middleware/
├── pkg/                 # Public library packages
│   ├── auth/
│   └── models/
├── api/                 # API definitions (protobuf, OpenAPI)
├── web/                 # Static assets, templates
└── scripts/             # Build and deployment scripts
```

### 5.2 Functional Options Pattern

```go
package server

type Server struct {
    host    string
    port    int
    timeout time.Duration
    logger  *log.Logger
    tls     bool
}

// Option is a function that configures Server
type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func WithLogger(l *log.Logger) Option {
    return func(s *Server) { s.logger = l }
}

func WithTLS(enable bool) Option {
    return func(s *Server) { s.tls = enable }
}

func New(host string, opts ...Option) *Server {
    s := &Server{
        host:    host,
        port:    8080,
        timeout: 30 * time.Second,
        logger:  log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage:
// s := server.New("localhost",
//     server.WithPort(9090),
//     server.WithTimeout(60*time.Second),
//     server.WithTLS(true),
// )
```

### 5.3 Package Documentation

```go
// Package auth provides authentication and authorization utilities
// for the myproject application.
//
// Basic usage:
//
//	token, err := auth.NewToken(userID, auth.WithExpiry(24*time.Hour))
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	claims, err := auth.Validate(token)
package auth

// User represents an authenticated user.
// The zero value is not useful; use NewUser to create instances.
type User struct {
    ID    string
    Email string
    Roles []string
}
```

---

## 6. Internal Packages and Workspaces

### 6.1 Internal Packages

```
myproject/
├── internal/
│   └── secret/          # Only importable by myproject and its subpackages
│       └── secret.go
├── pkg/
│   └── public/          # Importable by anyone
│       └── public.go
└── cmd/
    └── app/
        └── main.go      # Can import internal/secret
```

```go
// This import works from within myproject:
import "github.com/user/myproject/internal/secret"

// This import FAILS from outside myproject:
// import "github.com/user/myproject/internal/secret"
// Error: use of internal package not allowed
```

### 6.2 Go Workspaces (Go 1.18+)

```bash
# For multi-module development
mkdir workspace && cd workspace
go work init ./module-a ./module-b

# go.work file:
cat go.work
```

```
go 1.22

use (
    ./module-a
    ./module-b
)
```

```bash
# Modules can reference each other without replace directives
go work sync  # Sync go.sum files across modules
```

### 6.3 Build Tags

```go
//go:build linux
// +build linux

package mypackage

// This file is only compiled on Linux
func platformSpecific() {
    // Linux-specific implementation
}

// file: mypackage_windows.go
//go:build windows

func platformSpecific() {
    // Windows-specific implementation
}
```

```bash
# Build with custom tags
go build -tags integration ./...
```

---

## 7. Summary

### Key Takeaways

1. **Packages organize code** — one directory = one package. File names don't matter, only package declaration.
2. **Uppercase = exported** — the only visibility rule. No public/private/protected keywords.
3. **Go modules manage dependencies** — `go.mod` declares module path and requirements.
4. **`go mod tidy` is your friend** — run it after adding/removing imports to keep `go.mod` clean.
5. **Internal packages are enforced** — `internal/` directories restrict importability.
6. **Small, focused packages** — avoid `util`, `common`, `helper`. Package name should describe its purpose.
7. **Functional options for configuration** — the `WithXxx` pattern avoids parameter explosion.

### Quick Reference

```bash
go mod init MODULE_PATH    # Initialize module
go mod tidy                # Sync dependencies
go get PKG@VERSION         # Add/update dependency
go mod vendor              # Vendor dependencies
go build ./...             # Build all packages
go test ./...              # Test all packages
go vet ./...               # Static analysis
go doc PKG                 # View documentation
```

---

## Exercises

### Exercise 1: Package Organization
Create a project with three packages: `models` (User, Product structs), `store` (in-memory CRUD), and `main` (CLI interface). Practice proper visibility and naming.

### Exercise 2: Functional Options
Implement a `Logger` with functional options: `WithLevel`, `WithOutput`, `WithFormat`, `WithTimestamp`. Write clean package documentation.

### Exercise 3: Multi-Module Workspace
Create a workspace with two modules: `mathlib` (math utilities) and `calculator` (CLI using mathlib). Use `go work` for local development.

### Exercise 4: Build Tags
Create a `storage` package with two implementations: `storage_file.go` (file-based) and `storage_memory.go` (memory-based), selected via build tags.

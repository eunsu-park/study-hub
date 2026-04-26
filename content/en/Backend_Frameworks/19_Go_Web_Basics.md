# 19. Go Web Basics

**Previous**: [Project: REST API](./18_Project_REST_API.md) | **Next**: [Redis Caching Patterns](./20_Redis_Caching_Patterns.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Understand why Go is a strong choice for backend services (concurrency, performance, simplicity)
- Build HTTP servers using Go's `net/http` standard library
- Use the Gin framework for routing, middleware, and request binding
- Compare Gin with the Echo framework to make informed framework choices
- Handle JSON requests and responses idiomatically
- Implement middleware patterns for logging, authentication, and recovery
- Integrate a SQL database using GORM
- Apply Go-idiomatic error handling patterns in web applications
- Follow established Go project structure conventions

## Table of Contents

Before the framework reference, read [**Theory & Principles**](#theory--principles) — `net/http` and ServeMux routing, the goroutine + channel CSP model that powers Go's concurrency, and `context.Context` as the universal cancellation propagation mechanism.

1. [Why Go for Backend Development](#1-why-go-for-backend-development)
2. [The net/http Standard Library](#2-the-nethttp-standard-library)
3. [Gin Framework Basics](#3-gin-framework-basics)
4. [Echo Framework Comparison](#4-echo-framework-comparison)
5. [Request Handling and JSON Responses](#5-request-handling-and-json-responses)
6. [Middleware Patterns](#6-middleware-patterns)
7. [Database Integration with GORM](#7-database-integration-with-gorm)
8. [Error Handling Patterns](#8-error-handling-patterns)
9. [Project Structure Conventions](#9-project-structure-conventions)
10. [Testing Go Web Applications](#10-testing-go-web-applications)
11. [Practice Exercises](#11-practice-exercises)

---

## Theory & Principles

Go's web stack looks similar to other languages on the surface — handlers, middleware, ORMs — but the underlying concurrency and request-cancellation models are different in ways that change how you write the code. Three concepts cover the differences.

- **(A) `net/http` and the ServeMux** — Go's stdlib HTTP server, the building block every framework wraps.
- **(B) Goroutines and channels: CSP-style concurrency** — what "go func()" actually does and why Go has no event loop.
- **(C) `context.Context` for cancellation propagation** — the chain that ties an HTTP request to every downstream call.

### A. net/http and the ServeMux

Unlike Python (where you need Flask/FastAPI/Django) or Node (where you need Express), Go's standard library ships a production-grade HTTP server. Every Go web framework is just a layer over `net/http`.

#### A.1 The Handler interface

The whole HTTP server is built on one interface:

```go
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}
```

A handler is anything that implements `ServeHTTP`. `http.HandlerFunc` is a convenience wrapper that lets you use plain functions:

```go
http.HandleFunc("/hello", func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "hello")
})
http.ListenAndServe(":8080", nil)
```

That's a complete production-capable HTTP server. No framework, no decorators, just the standard library.

#### A.2 ServeMux: pattern-based routing

`http.ServeMux` is the built-in router. It matches URL paths against registered patterns:

```go
mux := http.NewServeMux()
mux.HandleFunc("/users/", listUsers)        // matches /users/, /users/foo, etc
mux.HandleFunc("GET /users/{id}", getUser)  // Go 1.22+: method + path with {id}
http.ListenAndServe(":8080", mux)
```

Pre-Go 1.22, ServeMux was very basic — no method matching, no path parameters. That gap is why third-party routers (gorilla/mux, chi, gin's tree) became popular. Go 1.22 added method-aware patterns and path parameters to the stdlib, narrowing the gap. Gin and Echo still win on middleware ergonomics, binding, and error helpers — but stdlib `net/http` is now sufficient for many APIs.

#### A.3 The request lifecycle

For each accepted connection, Go's HTTP server runs:

1. Read and parse the request headers.
2. Look up the handler in the mux.
3. Call `handler.ServeHTTP(w, r)` *in a new goroutine* — that's the magic.
4. Handler reads the body if needed, writes the response.
5. The goroutine ends; the connection may be reused (HTTP/1.1 keep-alive) or closed.

The "new goroutine per request" is the entire concurrency model. There is no thread pool to size, no event loop to share — the runtime multiplexes goroutines onto OS threads transparently.

### B. Goroutines and Channels: CSP-Style Concurrency

The model is Communicating Sequential Processes (CSP, Tony Hoare 1978). Goroutines are the processes, channels are the communication primitive.

#### B.1 What a goroutine actually is

`go func() { ... }()` starts a new goroutine. Costs:

- ~2 KB initial stack (grows on demand to MB).
- A few microseconds of scheduler overhead per spawn.
- One slot in the runtime's goroutine table.

A Go server can have *millions* of goroutines blocked on I/O simultaneously. The runtime multiplexes them across `GOMAXPROCS` OS threads (defaults to CPU count). Blocking on a syscall doesn't block the OS thread — the runtime moves other goroutines to other threads.

This is why Go has no event loop and no `async`/`await`. The runtime *is* the event loop, hidden under syntactic sync code.

#### B.2 Channels: typed pipes between goroutines

Channels are how goroutines communicate without sharing memory. Two operations: `ch <- value` (send), `<-ch` (receive). Both block until the other side is ready (for unbuffered channels).

```go
ch := make(chan int)
go func() { ch <- 42 }()
val := <-ch  // blocks until the goroutine sends
```

The Go mantra: **don't communicate by sharing memory; share memory by communicating**. Instead of locks around a shared variable, pass the variable on a channel; whoever holds the value is the one allowed to mutate it.

For web servers, channels appear in:

- **Worker pools.** N goroutines pull jobs from one channel.
- **Fan-in/fan-out.** Spawn N goroutines for parallel work, collect results on a channel.
- **Cancellation.** `done := make(chan struct{}); close(done)` signals every listener.
- **Rate limiting.** A buffered channel of size N is a semaphore.

#### B.3 The race detector

Go ships with a built-in race detector: `go run -race ./...`. It catches data races (concurrent unsynchronized access to the same memory) at runtime. CI for any concurrent Go code should run with `-race`. The cost: 2-10× slower, ~5× more memory. Worth it.

### C. context.Context: Cancellation Propagation

A web request typically calls a database, which calls a cache, which might call a downstream service. If the client disconnects, every one of those calls should stop. `context.Context` is how Go propagates that cancellation signal.

#### C.1 The Context interface

```go
type Context interface {
    Deadline() (time.Time, bool)  // when this context expires
    Done() <-chan struct{}        // closed when canceled
    Err() error                   // why it was canceled
    Value(key any) any            // request-scoped values
}
```

Every HTTP handler in Go gets a context: `r.Context()`. That context is canceled when the client disconnects or the request times out. Pass it to *every* downstream call:

```go
func handler(w http.ResponseWriter, r *http.Request) {
    rows, err := db.QueryContext(r.Context(), "SELECT ...")
    // if r.Context() is canceled, the query is canceled too
}
```

Database drivers (`database/sql`), HTTP clients (`http.Client.Do`), Redis clients, gRPC clients — all accept a `context.Context` and respect cancellation. This is what makes graceful shutdown work end-to-end.

#### C.2 Deriving contexts: WithCancel, WithTimeout, WithDeadline

You can derive a child context with extra constraints:

```go
ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
defer cancel()
result, err := slowAPI.Call(ctx)
```

`ctx` is canceled when *either* the parent is canceled *or* the 2-second timeout fires. The `defer cancel()` is mandatory hygiene — it releases resources even on the success path.

#### C.3 The propagation discipline

The rule: **every function that can block accepts `ctx context.Context` as its first parameter**. This is the convention across the Go standard library and ecosystem. If a function can take seconds and does not accept a context, it cannot be cooperatively canceled.

The anti-pattern: `context.Background()` deep in your code. That ignores any cancellation from above, decoupling your code from the request lifecycle. Use `context.Background()` only in `main()`, tests, or true root contexts.

#### C.4 Context values: limited, with care

`ctx.Value(key)` lets you attach request-scoped values (request ID, user ID, tracing span). Useful but easy to misuse:

- **Do** put plumbing data: trace IDs, request IDs, auth principal.
- **Do not** put business data — pass it as explicit function arguments. Context values are untyped and invisible to the type system.

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (Why Go) sells the §B concurrency story and §C cancellation discipline as core advantages.
- §2 (`net/http` standard library) is §A — the foundation every framework rests on.
- §3 (Gin framework) wraps §A.2 with friendlier routing, request binding, and middleware DSL.
- §4 (Echo comparison) compares Gin and Echo on the same axes — performance, ergonomics, ecosystem.
- §5 (Request handling and JSON) is the same idea as Lesson 02's Pydantic story, with Go struct tags driving binding/validation.
- §6 (Middleware patterns) is `func(next http.Handler) http.Handler` — Go's middleware idiom, simpler than the framework variants.
- §7 (GORM) is the §A repository abstraction over `database/sql`, with its own N+1 considerations from Lesson 04 §C.
- §8 (Error handling) is Go's `if err != nil` discipline — explicit errors, no exceptions, with `errors.Is/As` for matching.
- §9 (Project structure) is the convention `cmd/`, `internal/`, `pkg/` that gives Go projects predictable shape.
- §10 (Testing) is the stdlib `testing` package plus `httptest.NewServer` for integration tests of §A handlers.

---

## 1. Why Go for Backend Development

Go (Golang) has become one of the most popular languages for building backend services. Companies like Google, Uber, Dropbox, and Twitch rely on Go for high-performance, concurrent systems.

### Key Advantages

**Concurrency model**: Go's goroutines and channels make concurrent programming accessible. A goroutine costs roughly 2 KB of stack space (compared to ~1 MB per OS thread), enabling millions of concurrent operations.

```
OS Thread Model:              Goroutine Model:
┌────────────────┐           ┌────────────────┐
│  OS Thread 1   │           │  OS Thread 1   │
│  (1-8 MB stack)│           │  (manages many) │
│  ┌──────────┐  │           │  ┌──┐┌──┐┌──┐  │
│  │ 1 handler│  │           │  │g1││g2││g3│  │
│  └──────────┘  │           │  └──┘└──┘└──┘  │
├────────────────┤           │  ┌──┐┌──┐┌──┐  │
│  OS Thread 2   │           │  │g4││g5││g6│  │
│  ┌──────────┐  │           │  └──┘└──┘└──┘  │
│  │ 1 handler│  │           └────────────────┘
│  └──────────┘  │           Each goroutine: ~2KB
└────────────────┘           Thousands on one thread
```

**Performance**: Go compiles to native machine code with no virtual machine overhead. Typical Go web services handle 10-50x the throughput of equivalent Python/Node.js services.

**Simplicity**: Go has a small language specification (~50 keywords). The standard library is extensive, and the ecosystem favors explicit code over "magic."

**Fast compilation**: Large Go projects compile in seconds, enabling rapid development cycles.

**Single binary deployment**: Go produces statically linked binaries with no runtime dependencies. Deploy by copying a single file.

**Built-in tooling**: `go fmt`, `go vet`, `go test`, and `go doc` are part of the standard toolchain.

### When Go Excels

| Use Case | Why Go Fits |
|---|---|
| Microservices | Small binary, fast startup, low memory |
| API gateways | High concurrency, low latency |
| CLI tools | Single binary, cross-compilation |
| Data pipelines | Goroutines for parallel processing |
| DevOps tooling | Docker, Kubernetes, Terraform are Go |

### When to Consider Alternatives

- Rapid prototyping (Python/Node.js may be faster to iterate)
- Heavy ORM-driven CRUD apps (Django/Rails have more batteries)
- Data science and ML (Python ecosystem is stronger)

---

## 2. The net/http Standard Library

Go's `net/http` package is production-grade. Many teams use it without any framework at all.

### Basic HTTP Server

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

func main() {
    mux := http.NewServeMux()

    mux.HandleFunc("GET /", handleHome)
    mux.HandleFunc("GET /health", handleHealth)
    mux.HandleFunc("GET /api/users/{id}", handleGetUser)
    mux.HandleFunc("POST /api/users", handleCreateUser)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    log.Printf("Server starting on %s", server.Addr)
    if err := server.ListenAndServe(); err != nil {
        log.Fatalf("Server failed: %v", err)
    }
}

func handleHome(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "Welcome to the Go API")
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "status": "healthy",
        "time":   time.Now().UTC().Format(time.RFC3339),
    })
}
```

### Path Parameters (Go 1.22+)

Go 1.22 introduced pattern matching in `http.ServeMux`:

```go
func handleGetUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "id":   id,
        "name": "User " + id,
    })
}
```

### Reading Request Body

```go
type CreateUserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

func handleCreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, `{"error": "invalid JSON"}`, http.StatusBadRequest)
        return
    }
    defer r.Body.Close()

    if req.Name == "" || req.Email == "" {
        http.Error(w, `{"error": "name and email required"}`, http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "id":    42,
        "name":  req.Name,
        "email": req.Email,
    })
}
```

### Custom Middleware with net/http

```go
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        log.Printf("Started %s %s", r.Method, r.URL.Path)

        next.ServeHTTP(w, r)

        log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(start))
    })
}

func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic recovered: %v", err)
                http.Error(w, `{"error": "internal server error"}`, http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// Chain middleware
func main() {
    mux := http.NewServeMux()
    // ... register handlers ...

    handler := loggingMiddleware(recoveryMiddleware(mux))

    http.ListenAndServe(":8080", handler)
}
```

---

## 3. Gin Framework Basics

[Gin](https://github.com/gin-gonic/gin) is the most popular Go web framework, known for its speed and minimal API surface.

### Installation and Hello World

```bash
go mod init myapp
go get -u github.com/gin-gonic/gin
```

```go
package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default() // includes Logger and Recovery middleware

    r.GET("/", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "message": "Hello, World!",
        })
    })

    r.Run(":8080")
}
```

### Routing

```go
func setupRoutes(r *gin.Engine) {
    // Basic routes
    r.GET("/ping", handlePing)
    r.POST("/users", createUser)
    r.GET("/users/:id", getUser)
    r.PUT("/users/:id", updateUser)
    r.DELETE("/users/:id", deleteUser)

    // Route groups
    api := r.Group("/api/v1")
    {
        api.GET("/products", listProducts)
        api.POST("/products", createProduct)

        // Nested groups with middleware
        admin := api.Group("/admin")
        admin.Use(authMiddleware())
        {
            admin.GET("/stats", getStats)
            admin.DELETE("/users/:id", deleteUser)
        }
    }

    // Query parameters: /search?q=go&page=1
    r.GET("/search", func(c *gin.Context) {
        query := c.DefaultQuery("q", "")
        page := c.DefaultQuery("page", "1")
        c.JSON(http.StatusOK, gin.H{"query": query, "page": page})
    })
}
```

### Request Binding and Validation

Gin uses struct tags for automatic request binding and validation:

```go
type CreateProductRequest struct {
    Name     string  `json:"name" binding:"required,min=1,max=200"`
    Price    float64 `json:"price" binding:"required,gt=0"`
    Category string  `json:"category" binding:"required,oneof=electronics books clothing"`
    SKU      string  `json:"sku" binding:"required,alphanum,len=8"`
}

func createProduct(c *gin.Context) {
    var req CreateProductRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    // req is now validated and populated
    c.JSON(http.StatusCreated, gin.H{
        "id":       1,
        "name":     req.Name,
        "price":    req.Price,
        "category": req.Category,
    })
}

// Bind path params and query params
type GetProductsQuery struct {
    Page     int    `form:"page" binding:"min=1"`
    PageSize int    `form:"page_size" binding:"min=1,max=100"`
    Sort     string `form:"sort" binding:"oneof=name price created_at"`
}

type URIParams struct {
    ID uint `uri:"id" binding:"required,min=1"`
}

func getUser(c *gin.Context) {
    var uri URIParams
    if err := c.ShouldBindUri(&uri); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user ID"})
        return
    }
    // use uri.ID ...
}
```

### Gin Context

The `gin.Context` is the core of every handler, carrying request data, response methods, and flow control:

```go
func exampleHandler(c *gin.Context) {
    // Request data
    id := c.Param("id")                          // path parameter
    name := c.Query("name")                       // query parameter
    token := c.GetHeader("Authorization")          // request header

    // Set values for downstream middleware/handlers
    c.Set("user_id", 42)
    userID, exists := c.Get("user_id")

    // Response methods
    c.JSON(http.StatusOK, gin.H{"id": id})         // JSON response
    c.String(http.StatusOK, "Hello %s", name)       // plain text
    c.Data(http.StatusOK, "text/csv", csvBytes)     // raw bytes
    c.File("./path/to/file.pdf")                    // serve file

    // Abort the request chain
    c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "access denied"})
}
```

---

## 4. Echo Framework Comparison

[Echo](https://echo.labstack.com/) is another popular Go web framework. Here is a side-by-side comparison with Gin.

### Echo Hello World

```go
package main

import (
    "net/http"
    "github.com/labstack/echo/v4"
    "github.com/labstack/echo/v4/middleware"
)

func main() {
    e := echo.New()
    e.Use(middleware.Logger())
    e.Use(middleware.Recover())

    e.GET("/", func(c echo.Context) error {
        return c.JSON(http.StatusOK, map[string]string{
            "message": "Hello, World!",
        })
    })

    e.Logger.Fatal(e.Start(":8080"))
}
```

### Feature Comparison

| Feature | Gin | Echo |
|---|---|---|
| Handler signature | `func(c *gin.Context)` | `func(c echo.Context) error` |
| Error handling | Manual (no return) | Return `error` (centralized) |
| Binding | `c.ShouldBindJSON(&s)` | `c.Bind(&s)` |
| Path params | `c.Param("id")` | `c.Param("id")` |
| Middleware | `r.Use(fn)` | `e.Use(fn)` |
| Route groups | `r.Group("/api")` | `e.Group("/api")` |
| Performance | ~65,000 req/s | ~63,000 req/s |
| GitHub stars | ~80k | ~30k |

### Echo Error Handling (Key Differentiator)

Echo's handler signature returns `error`, enabling centralized error handling:

```go
// Echo custom error handler
func customErrorHandler(err error, c echo.Context) {
    code := http.StatusInternalServerError
    message := "Internal Server Error"

    if he, ok := err.(*echo.HTTPError); ok {
        code = he.Code
        message = he.Message.(string)
    }

    c.JSON(code, map[string]string{"error": message})
}

func main() {
    e := echo.New()
    e.HTTPErrorHandler = customErrorHandler

    e.GET("/users/:id", func(c echo.Context) error {
        id := c.Param("id")
        user, err := findUser(id)
        if err != nil {
            return echo.NewHTTPError(http.StatusNotFound, "user not found")
        }
        return c.JSON(http.StatusOK, user)
    })
}
```

### Which to Choose?

- **Gin**: Larger community, more third-party middleware, well-established
- **Echo**: Cleaner error handling, slightly better documentation, built-in HTTPS
- Both are excellent; pick based on team preference

---

## 5. Request Handling and JSON Responses

### Consistent Response Envelope

Define a standard response format across your API:

```go
// response.go
package response

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

type APIResponse struct {
    Success bool        `json:"success"`
    Data    interface{} `json:"data,omitempty"`
    Error   *APIError   `json:"error,omitempty"`
    Meta    *Meta       `json:"meta,omitempty"`
}

type APIError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

type Meta struct {
    Page       int `json:"page"`
    PageSize   int `json:"page_size"`
    TotalCount int `json:"total_count"`
    TotalPages int `json:"total_pages"`
}

func Success(c *gin.Context, status int, data interface{}) {
    c.JSON(status, APIResponse{Success: true, Data: data})
}

func SuccessWithMeta(c *gin.Context, data interface{}, meta Meta) {
    c.JSON(http.StatusOK, APIResponse{
        Success: true,
        Data:    data,
        Meta:    &meta,
    })
}

func Error(c *gin.Context, status int, code, message string) {
    c.JSON(status, APIResponse{
        Success: false,
        Error:   &APIError{Code: code, Message: message},
    })
}
```

### Handling Different Content Types

```go
func handleUpload(c *gin.Context) {
    // Multipart form
    file, err := c.FormFile("avatar")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "no file uploaded"})
        return
    }

    dst := fmt.Sprintf("./uploads/%s", file.Filename)
    if err := c.SaveUploadedFile(file, dst); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save file"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "filename": file.Filename,
        "size":     file.Size,
    })
}

func handleFormData(c *gin.Context) {
    // URL-encoded form data
    name := c.PostForm("name")
    email := c.PostForm("email")
    c.JSON(http.StatusOK, gin.H{"name": name, "email": email})
}
```

### Streaming Responses

```go
func handleStream(c *gin.Context) {
    c.Stream(func(w io.Writer) bool {
        for i := 0; i < 10; i++ {
            c.SSEvent("message", gin.H{"count": i})
            time.Sleep(1 * time.Second)
        }
        return false
    })
}
```

---

## 6. Middleware Patterns

### Authentication Middleware

```go
func authMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "authorization header required",
            })
            return
        }

        // Strip "Bearer " prefix
        if len(token) > 7 && token[:7] == "Bearer " {
            token = token[7:]
        }

        claims, err := validateJWT(token)
        if err != nil {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "invalid or expired token",
            })
            return
        }

        // Store user info for downstream handlers
        c.Set("user_id", claims.UserID)
        c.Set("user_role", claims.Role)
        c.Next()
    }
}

// Role-based authorization
func requireRole(roles ...string) gin.HandlerFunc {
    return func(c *gin.Context) {
        userRole, exists := c.Get("user_role")
        if !exists {
            c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "no role found"})
            return
        }

        for _, role := range roles {
            if userRole == role {
                c.Next()
                return
            }
        }

        c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "insufficient permissions"})
    }
}
```

### Rate Limiting Middleware

```go
import "golang.org/x/time/rate"

func rateLimitMiddleware(rps float64, burst int) gin.HandlerFunc {
    limiter := rate.NewLimiter(rate.Limit(rps), burst)
    return func(c *gin.Context) {
        if !limiter.Allow() {
            c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                "error": "rate limit exceeded",
            })
            return
        }
        c.Next()
    }
}

// Per-client rate limiting with a map
func perClientRateLimiter(rps float64, burst int) gin.HandlerFunc {
    clients := make(map[string]*rate.Limiter)
    var mu sync.Mutex

    return func(c *gin.Context) {
        ip := c.ClientIP()

        mu.Lock()
        limiter, exists := clients[ip]
        if !exists {
            limiter = rate.NewLimiter(rate.Limit(rps), burst)
            clients[ip] = limiter
        }
        mu.Unlock()

        if !limiter.Allow() {
            c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                "error": "rate limit exceeded",
            })
            return
        }
        c.Next()
    }
}
```

### CORS Middleware

```go
func corsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("Access-Control-Allow-Origin", "*")
        c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        c.Header("Access-Control-Max-Age", "86400")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(http.StatusNoContent)
            return
        }

        c.Next()
    }
}
```

### Request ID Middleware

```go
import "github.com/google/uuid"

func requestIDMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        requestID := c.GetHeader("X-Request-ID")
        if requestID == "" {
            requestID = uuid.New().String()
        }
        c.Set("request_id", requestID)
        c.Header("X-Request-ID", requestID)
        c.Next()
    }
}
```

---

## 7. Database Integration with GORM

[GORM](https://gorm.io/) is Go's most popular ORM. It supports PostgreSQL, MySQL, SQLite, and SQL Server.

### Setup

```bash
go get -u gorm.io/gorm
go get -u gorm.io/driver/postgres
```

### Model Definition

```go
package models

import (
    "time"
    "gorm.io/gorm"
)

type User struct {
    ID        uint           `gorm:"primaryKey" json:"id"`
    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"-"`
    Name      string         `gorm:"size:100;not null" json:"name"`
    Email     string         `gorm:"uniqueIndex;size:255;not null" json:"email"`
    Password  string         `gorm:"size:255;not null" json:"-"`
    Role      string         `gorm:"size:20;default:user" json:"role"`
    Posts     []Post         `gorm:"foreignKey:AuthorID" json:"posts,omitempty"`
}

type Post struct {
    ID        uint           `gorm:"primaryKey" json:"id"`
    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"-"`
    Title     string         `gorm:"size:200;not null" json:"title"`
    Content   string         `gorm:"type:text;not null" json:"content"`
    Published bool           `gorm:"default:false" json:"published"`
    AuthorID  uint           `gorm:"not null" json:"author_id"`
    Author    User           `gorm:"foreignKey:AuthorID" json:"author,omitempty"`
    Tags      []Tag          `gorm:"many2many:post_tags;" json:"tags,omitempty"`
}

type Tag struct {
    ID   uint   `gorm:"primaryKey" json:"id"`
    Name string `gorm:"uniqueIndex;size:50" json:"name"`
}
```

### Database Connection

```go
package database

import (
    "fmt"
    "log"
    "os"
    "time"

    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/logger"
)

var DB *gorm.DB

func Connect() {
    dsn := fmt.Sprintf(
        "host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
        os.Getenv("DB_HOST"),
        os.Getenv("DB_USER"),
        os.Getenv("DB_PASSWORD"),
        os.Getenv("DB_NAME"),
        os.Getenv("DB_PORT"),
    )

    var err error
    DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
    })
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }

    // Connection pool settings
    sqlDB, _ := DB.DB()
    sqlDB.SetMaxIdleConns(10)
    sqlDB.SetMaxOpenConns(100)
    sqlDB.SetConnMaxLifetime(time.Hour)

    log.Println("Database connected successfully")
}

func Migrate() {
    DB.AutoMigrate(&User{}, &Post{}, &Tag{})
    log.Println("Database migration completed")
}
```

### CRUD Operations

```go
package repository

import "gorm.io/gorm"

type PostRepository struct {
    db *gorm.DB
}

func NewPostRepository(db *gorm.DB) *PostRepository {
    return &PostRepository{db: db}
}

func (r *PostRepository) Create(post *Post) error {
    return r.db.Create(post).Error
}

func (r *PostRepository) FindByID(id uint) (*Post, error) {
    var post Post
    err := r.db.Preload("Author").Preload("Tags").First(&post, id).Error
    if err != nil {
        return nil, err
    }
    return &post, nil
}

func (r *PostRepository) List(page, pageSize int, published *bool) ([]Post, int64, error) {
    var posts []Post
    var total int64

    query := r.db.Model(&Post{})
    if published != nil {
        query = query.Where("published = ?", *published)
    }

    query.Count(&total)

    err := query.
        Preload("Author").
        Preload("Tags").
        Offset((page - 1) * pageSize).
        Limit(pageSize).
        Order("created_at DESC").
        Find(&posts).Error

    return posts, total, err
}

func (r *PostRepository) Update(post *Post) error {
    return r.db.Save(post).Error
}

func (r *PostRepository) Delete(id uint) error {
    return r.db.Delete(&Post{}, id).Error // soft delete
}

// Transaction example
func (r *PostRepository) CreateWithTags(post *Post, tagNames []string) error {
    return r.db.Transaction(func(tx *gorm.DB) error {
        if err := tx.Create(post).Error; err != nil {
            return err
        }

        for _, name := range tagNames {
            var tag Tag
            tx.FirstOrCreate(&tag, Tag{Name: name})
            if err := tx.Model(post).Association("Tags").Append(&tag); err != nil {
                return err
            }
        }

        return nil
    })
}
```

---

## 8. Error Handling Patterns

### Custom Error Types

```go
package apperror

import "fmt"

type AppError struct {
    Code    int    `json:"-"`
    Type    string `json:"type"`
    Message string `json:"message"`
    Detail  string `json:"detail,omitempty"`
}

func (e *AppError) Error() string {
    return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func NotFound(resource string, id interface{}) *AppError {
    return &AppError{
        Code:    404,
        Type:    "NOT_FOUND",
        Message: fmt.Sprintf("%s not found", resource),
        Detail:  fmt.Sprintf("No %s with ID %v", resource, id),
    }
}

func BadRequest(message string) *AppError {
    return &AppError{Code: 400, Type: "BAD_REQUEST", Message: message}
}

func Unauthorized(message string) *AppError {
    return &AppError{Code: 401, Type: "UNAUTHORIZED", Message: message}
}

func Forbidden(message string) *AppError {
    return &AppError{Code: 403, Type: "FORBIDDEN", Message: message}
}

func Internal(err error) *AppError {
    return &AppError{
        Code:    500,
        Type:    "INTERNAL_ERROR",
        Message: "An internal error occurred",
        Detail:  err.Error(),
    }
}
```

### Error Handling Middleware

```go
func errorHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()

        // Check for errors set during request handling
        if len(c.Errors) > 0 {
            err := c.Errors.Last().Err
            switch e := err.(type) {
            case *apperror.AppError:
                c.JSON(e.Code, gin.H{
                    "success": false,
                    "error":   e,
                })
            default:
                c.JSON(http.StatusInternalServerError, gin.H{
                    "success": false,
                    "error": gin.H{
                        "type":    "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                    },
                })
            }
        }
    }
}

// Usage in handler
func getPost(c *gin.Context) {
    id, err := strconv.ParseUint(c.Param("id"), 10, 32)
    if err != nil {
        c.Error(apperror.BadRequest("invalid post ID"))
        return
    }

    post, err := postRepo.FindByID(uint(id))
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            c.Error(apperror.NotFound("post", id))
        } else {
            c.Error(apperror.Internal(err))
        }
        return
    }

    c.JSON(http.StatusOK, gin.H{"success": true, "data": post})
}
```

### Wrapping Errors with Context

```go
import "fmt"

func (s *PostService) Publish(postID, userID uint) error {
    post, err := s.repo.FindByID(postID)
    if err != nil {
        return fmt.Errorf("publish: finding post %d: %w", postID, err)
    }

    if post.AuthorID != userID {
        return fmt.Errorf("publish: user %d is not author of post %d: %w",
            userID, postID, apperror.Forbidden("not the author"))
    }

    post.Published = true
    if err := s.repo.Update(post); err != nil {
        return fmt.Errorf("publish: updating post %d: %w", postID, err)
    }

    return nil
}
```

---

## 9. Project Structure Conventions

### Standard Go Project Layout

```
myapp/
├── cmd/
│   └── server/
│       └── main.go              # Application entry point
├── internal/                    # Private application code
│   ├── config/
│   │   └── config.go            # Configuration loading
│   ├── database/
│   │   └── database.go          # Database connection
│   ├── handler/
│   │   ├── user_handler.go      # HTTP handlers (controllers)
│   │   └── post_handler.go
│   ├── middleware/
│   │   ├── auth.go
│   │   ├── cors.go
│   │   └── logging.go
│   ├── model/
│   │   ├── user.go              # Database models
│   │   └── post.go
│   ├── repository/
│   │   ├── user_repo.go         # Data access layer
│   │   └── post_repo.go
│   ├── service/
│   │   ├── user_service.go      # Business logic
│   │   └── post_service.go
│   └── router/
│       └── router.go            # Route definitions
├── pkg/                         # Public, reusable packages
│   ├── apperror/
│   │   └── errors.go
│   └── response/
│       └── response.go
├── migrations/                  # SQL migration files
├── docs/                        # API documentation
├── go.mod
├── go.sum
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### Configuration Management

```go
// internal/config/config.go
package config

import (
    "log"
    "os"
    "strconv"
)

type Config struct {
    Port        string
    DatabaseURL string
    JWTSecret   string
    LogLevel    string
    Environment string
}

func Load() *Config {
    return &Config{
        Port:        getEnv("PORT", "8080"),
        DatabaseURL: getEnv("DATABASE_URL", "postgres://localhost:5432/myapp?sslmode=disable"),
        JWTSecret:   getEnv("JWT_SECRET", "change-me-in-production"),
        LogLevel:    getEnv("LOG_LEVEL", "info"),
        Environment: getEnv("ENVIRONMENT", "development"),
    }
}

func getEnv(key, fallback string) string {
    if value, ok := os.LookupEnv(key); ok {
        return value
    }
    return fallback
}

func getEnvInt(key string, fallback int) int {
    if value, ok := os.LookupEnv(key); ok {
        if i, err := strconv.Atoi(value); err == nil {
            return i
        }
    }
    return fallback
}
```

### Makefile for Common Tasks

```makefile
.PHONY: run build test lint migrate

run:
	go run cmd/server/main.go

build:
	CGO_ENABLED=0 go build -o bin/server cmd/server/main.go

test:
	go test ./... -v -cover

lint:
	golangci-lint run

migrate:
	go run cmd/migrate/main.go

docker-build:
	docker build -t myapp .

docker-run:
	docker-compose up -d
```

### Dockerfile

```dockerfile
# Build stage
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server cmd/server/main.go

# Run stage
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /server .
EXPOSE 8080
CMD ["./server"]
```

---

## 10. Testing Go Web Applications

Go's standard library includes everything needed to test HTTP handlers without external frameworks.

### Table-Driven Tests

The idiomatic Go testing pattern groups related cases in a slice of structs:

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name   string
        a, b   int
        want   int
    }{
        {"positive", 1, 2, 3},
        {"negative", -1, -2, -3},
        {"zero", 0, 5, 5},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := Add(tt.a, tt.b); got != tt.want {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### httptest Package

`net/http/httptest` lets you call handlers directly without starting a real server:

```go
import (
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
)

func TestHandleHealth(t *testing.T) {
    req := httptest.NewRequest(http.MethodGet, "/health", nil)
    rec := httptest.NewRecorder()

    handleHealth(rec, req)

    if rec.Code != http.StatusOK {
        t.Fatalf("expected 200, got %d", rec.Code)
    }
    var body map[string]string
    json.NewDecoder(rec.Body).Decode(&body)
    if body["status"] != "healthy" {
        t.Errorf("unexpected status: %s", body["status"])
    }
}
```

For Gin handlers, wrap the engine in `httptest.NewRecorder()`:

```go
func TestCreateProduct(t *testing.T) {
    gin.SetMode(gin.TestMode)
    r := gin.New()
    r.POST("/products", createProduct)

    body := `{"name":"Widget","price":9.99,"category":"electronics","sku":"ABCD1234"}`
    req := httptest.NewRequest(http.MethodPost, "/products", strings.NewReader(body))
    req.Header.Set("Content-Type", "application/json")
    rec := httptest.NewRecorder()

    r.ServeHTTP(rec, req)

    if rec.Code != http.StatusCreated {
        t.Fatalf("expected 201, got %d: %s", rec.Code, rec.Body.String())
    }
}
```

### testify for Assertions

[testify](https://github.com/stretchr/testify) reduces boilerplate with `assert` and `require`:

```go
import "github.com/stretchr/testify/assert"

func TestUserService(t *testing.T) {
    svc := NewUserService(setupTestDB(t))
    user, err := svc.Create("alice@example.com", "Alice")

    assert.NoError(t, err)
    assert.Equal(t, "Alice", user.Name)
    assert.NotZero(t, user.ID)
}
```

Use `require` instead of `assert` when a failure should stop the test immediately (e.g., nil pointer check before accessing fields).

---

## 11. Practice Exercises

### Exercise 1: Basic API with net/http

Build a bookmark manager API using only Go's standard library:
- `GET /bookmarks` — list all bookmarks (in-memory slice)
- `POST /bookmarks` — add a bookmark (title, URL, tags)
- `GET /bookmarks/{id}` — get a single bookmark
- `DELETE /bookmarks/{id}` — remove a bookmark
- Add a logging middleware that prints method, path, and duration

```go
// Starter code
package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sync"
    "time"
)

type Bookmark struct {
    ID        int       `json:"id"`
    Title     string    `json:"title"`
    URL       string    `json:"url"`
    Tags      []string  `json:"tags"`
    CreatedAt time.Time `json:"created_at"`
}

var (
    bookmarks = make([]Bookmark, 0)
    nextID    = 1
    mu        sync.Mutex
)

func main() {
    mux := http.NewServeMux()

    // TODO: Register routes
    // TODO: Add logging middleware
    // TODO: Start server on :8080

    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### Exercise 2: Gin CRUD API with Validation

Build a task management API with Gin:
- Implement full CRUD for tasks (title, description, status, priority, due date)
- Use struct binding with validation tags
- Add route groups: public routes and authenticated routes
- Implement a simple token-based auth middleware
- Return consistent JSON responses using a response helper

### Exercise 3: GORM Repository Pattern

Create a repository for a library system:
- Models: `Book`, `Author`, `Genre` (with many-to-many relationships)
- Repository methods: `Create`, `FindByID`, `Search` (by title/author), `ListByGenre`, `Delete`
- Use transactions for creating a book with a new author
- Add pagination to list operations
- Write table-driven tests using an in-memory SQLite database

```go
// Test starter
func TestPostRepository_Create(t *testing.T) {
    db := setupTestDB(t)
    repo := NewPostRepository(db)

    tests := []struct {
        name    string
        post    Post
        wantErr bool
    }{
        {
            name:    "valid post",
            post:    Post{Title: "Test", Content: "Body", AuthorID: 1},
            wantErr: false,
        },
        {
            name:    "missing title",
            post:    Post{Content: "Body", AuthorID: 1},
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := repo.Create(&tt.post)
            if (err != nil) != tt.wantErr {
                t.Errorf("Create() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Exercise 4: Middleware Chain

Build a middleware chain for a Gin application that includes:
1. Request ID generation (UUID)
2. Structured JSON logging (method, path, status, duration, request ID)
3. Panic recovery with error logging
4. CORS with configurable origins
5. Rate limiting (10 req/s per IP)

Test the chain by sending concurrent requests with `curl` or a Go test client.

---

## Further Reading

- [Go Documentation](https://go.dev/doc/)
- [Effective Go](https://go.dev/doc/effective_go)
- [Gin Documentation](https://gin-gonic.com/docs/)
- [Echo Documentation](https://echo.labstack.com/docs)
- [GORM Documentation](https://gorm.io/docs/)
- [Standard Go Project Layout](https://github.com/golang-standards/project-layout)

---

**Previous**: [Project: REST API](./18_Project_REST_API.md) | **Next**: [Redis Caching Patterns](./20_Redis_Caching_Patterns.md)

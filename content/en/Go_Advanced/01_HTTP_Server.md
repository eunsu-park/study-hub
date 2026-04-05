# 12. HTTP Server

**Previous**: [Standard Library](../Go_Basics/11_Standard_Library.md) | **Next**: [REST API](./02_REST_API.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build HTTP servers using `net/http`
2. Implement request routing with `http.ServeMux` (Go 1.22+)
3. Write middleware for logging, recovery, and authentication
4. Handle static files and templates
5. Implement graceful shutdown

---

Go's `net/http` package provides a production-ready HTTP server out of the box — no framework required. With Go 1.22's enhanced routing, the standard library handles path parameters and method-based routing that previously required third-party routers.

## Table of Contents
1. [Basic Server](#1-basic-server)
2. [Enhanced Routing (Go 1.22+)](#2-enhanced-routing-go-122)
3. [Handlers and HandlerFunc](#3-handlers-and-handlerfunc)
4. [Middleware](#4-middleware)
5. [Templates and Static Files](#5-templates-and-static-files)
6. [Graceful Shutdown](#6-graceful-shutdown)
7. [Summary](#7-summary)

---

## 1. Basic Server

### 1.1 Hello World Server

```go
package main

import (
    "fmt"
    "log"
    "net/http"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
    })

    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusOK)
        fmt.Fprint(w, `{"status": "ok"}`)
    })

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 1.2 http.Server Configuration

```go
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/", homeHandler)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
        MaxHeaderBytes: 1 << 20, // 1 MB
    }

    log.Println("Server starting on :8080")
    log.Fatal(server.ListenAndServe())
}
```

---

## 2. Enhanced Routing (Go 1.22+)

### 2.1 Method-Based and Path Parameters

```go
func main() {
    mux := http.NewServeMux()

    // Method-based routing
    mux.HandleFunc("GET /users", listUsers)
    mux.HandleFunc("POST /users", createUser)

    // Path parameters with {name}
    mux.HandleFunc("GET /users/{id}", getUser)
    mux.HandleFunc("PUT /users/{id}", updateUser)
    mux.HandleFunc("DELETE /users/{id}", deleteUser)

    // Wildcard — matches remaining path
    mux.HandleFunc("GET /files/{path...}", serveFile)

    // Exact match with trailing slash
    mux.HandleFunc("GET /api/", apiIndex)      // Matches /api/ only
    mux.HandleFunc("GET /api/{rest...}", apiCatchAll)

    log.Fatal(http.ListenAndServe(":8080", mux))
}

func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id") // Extract path parameter
    fmt.Fprintf(w, "User ID: %s", id)
}

func serveFile(w http.ResponseWriter, r *http.Request) {
    path := r.PathValue("path") // Wildcard value
    fmt.Fprintf(w, "File path: %s", path)
}
```

### 2.2 Precedence Rules

```go
mux := http.NewServeMux()

// More specific patterns take precedence
mux.HandleFunc("GET /users/me", getCurrentUser)    // Matches first for /users/me
mux.HandleFunc("GET /users/{id}", getUser)         // Matches /users/123

// Method-specific beats catch-all
mux.HandleFunc("GET /items", listItems)            // GET only
mux.HandleFunc("/items", handleItems)              // All other methods
```

---

## 3. Handlers and HandlerFunc

### 3.1 The Handler Interface

```go
// http.Handler interface
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}

// Struct-based handler
type APIHandler struct {
    db     *sql.DB
    logger *slog.Logger
}

func (h *APIHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    h.logger.Info("request",
        "method", r.Method,
        "path", r.URL.Path,
    )
    // Handle request...
}

func main() {
    handler := &APIHandler{
        db:     connectDB(),
        logger: slog.Default(),
    }
    http.Handle("/api/", handler)
}
```

### 3.2 Reading Requests

```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    // Method
    fmt.Println("Method:", r.Method)

    // URL components
    fmt.Println("Path:", r.URL.Path)
    fmt.Println("Query:", r.URL.Query().Get("page"))

    // Headers
    fmt.Println("Content-Type:", r.Header.Get("Content-Type"))
    fmt.Println("User-Agent:", r.Header.Get("User-Agent"))

    // Body (for POST/PUT)
    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Bad request", http.StatusBadRequest)
        return
    }
    defer r.Body.Close()
    fmt.Println("Body:", string(body))

    // Form data
    r.ParseForm()
    fmt.Println("Name:", r.FormValue("name"))

    // JSON body
    var data struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }
    if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
}
```

### 3.3 Writing Responses

```go
func jsonResponse(w http.ResponseWriter, status int, data any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

func handleUser(w http.ResponseWriter, r *http.Request) {
    user := User{ID: 1, Name: "Alice"}
    jsonResponse(w, http.StatusOK, user)
}

func handleError(w http.ResponseWriter, r *http.Request) {
    // http.Error — simple text error response
    http.Error(w, "Not Found", http.StatusNotFound)

    // JSON error
    jsonResponse(w, http.StatusBadRequest, map[string]string{
        "error": "invalid request",
    })
}
```

---

## 4. Middleware

### 4.1 Middleware Pattern

```go
type Middleware func(http.Handler) http.Handler

// Logging middleware
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        slog.Info("request",
            "method", r.Method,
            "path", r.URL.Path,
            "duration", time.Since(start),
        )
    })
}

// Recovery middleware
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                slog.Error("panic recovered", "error", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// CORS middleware
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        next.ServeHTTP(w, r)
    })
}
```

### 4.2 Chaining Middleware

```go
func chain(handler http.Handler, middlewares ...Middleware) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /api/users", listUsers)

    // Apply middleware chain
    handler := chain(mux,
        recoveryMiddleware,
        loggingMiddleware,
        corsMiddleware,
    )

    http.ListenAndServe(":8080", handler)
}
```

### 4.3 Response Wrapper for Status Capture

```go
type statusRecorder struct {
    http.ResponseWriter
    statusCode int
}

func (r *statusRecorder) WriteHeader(code int) {
    r.statusCode = code
    r.ResponseWriter.WriteHeader(code)
}

func loggingWithStatus(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        recorder := &statusRecorder{ResponseWriter: w, statusCode: 200}
        start := time.Now()
        next.ServeHTTP(recorder, r)
        slog.Info("request",
            "method", r.Method,
            "path", r.URL.Path,
            "status", recorder.statusCode,
            "duration", time.Since(start),
        )
    })
}
```

---

## 5. Templates and Static Files

### 5.1 HTML Templates

```go
import "html/template"

var templates = template.Must(template.ParseGlob("templates/*.html"))

type PageData struct {
    Title string
    Users []User
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
    data := PageData{
        Title: "User List",
        Users: []User{
            {ID: 1, Name: "Alice"},
            {ID: 2, Name: "Bob"},
        },
    }
    templates.ExecuteTemplate(w, "home.html", data)
}
```

```html
<!-- templates/home.html -->
<!DOCTYPE html>
<html>
<head><title>{{.Title}}</title></head>
<body>
    <h1>{{.Title}}</h1>
    <ul>
    {{range .Users}}
        <li>{{.Name}} (ID: {{.ID}})</li>
    {{end}}
    </ul>
</body>
</html>
```

### 5.2 Static Files

```go
func main() {
    mux := http.NewServeMux()

    // Serve static files
    fs := http.FileServer(http.Dir("static"))
    mux.Handle("GET /static/", http.StripPrefix("/static/", fs))

    // Embedded files (Go 1.16+)
    //go:embed static/*
    // var staticFS embed.FS
    // mux.Handle("/static/", http.FileServer(http.FS(staticFS)))

    http.ListenAndServe(":8080", mux)
}
```

---

## 6. Graceful Shutdown

```go
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /", homeHandler)

    server := &http.Server{
        Addr:    ":8080",
        Handler: mux,
    }

    // Start server in goroutine
    go func() {
        slog.Info("server starting", "addr", server.Addr)
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            slog.Error("server error", "err", err)
        }
    }()

    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    slog.Info("shutting down server...")

    // Give active connections 30 seconds to finish
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := server.Shutdown(ctx); err != nil {
        slog.Error("forced shutdown", "err", err)
    }

    slog.Info("server stopped")
}
```

---

## 7. Summary

### Key Takeaways

1. **`net/http` is production-ready** — no framework needed for most applications.
2. **Go 1.22 enhanced routing** — method-based routing and path parameters in the standard library.
3. **Middleware pattern** — `func(http.Handler) http.Handler` for cross-cutting concerns.
4. **Always set timeouts** — `ReadTimeout`, `WriteTimeout`, `IdleTimeout` prevent resource exhaustion.
5. **Graceful shutdown** — `server.Shutdown(ctx)` waits for active connections to complete.
6. **Response headers before body** — call `w.Header().Set()` and `w.WriteHeader()` before writing body.

---

## Exercises

### Exercise 1: File Upload Server
Build a server that accepts file uploads via POST, stores them in a directory, and serves them back via GET.

### Exercise 2: Middleware Suite
Create a middleware package with: request ID injection, rate limiting, basic auth, and request/response logging.

### Exercise 3: SSE Server
Implement a Server-Sent Events endpoint that streams real-time updates to connected clients.

### Exercise 4: Reverse Proxy
Build a simple reverse proxy using `httputil.ReverseProxy` with custom header injection and logging.

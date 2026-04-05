# 21. Cloud Native Patterns

**Previous**: [Network Programming](./09_Network_Programming.md) | **Next**: [Capstone: Microservice](./11_Capstone_Microservice.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement health check endpoints (liveness and readiness)
2. Use `context.Context` for request lifecycle management
3. Implement graceful shutdown with signal handling
4. Apply the 12-factor app methodology
5. Add observability: structured logging, metrics, and tracing

---

Cloud native development is about building applications that leverage cloud infrastructure effectively. Go is the default language for cloud native tools (Docker, Kubernetes, Prometheus), and these patterns make your Go services production-ready.

## Table of Contents
1. [Health Checks](#1-health-checks)
2. [Graceful Shutdown](#2-graceful-shutdown)
3. [Configuration (12-Factor)](#3-configuration-12-factor)
4. [Observability](#4-observability)
5. [Resilience Patterns](#5-resilience-patterns)
6. [Dependency Injection](#6-dependency-injection)
7. [Summary](#7-summary)

---

## 1. Health Checks

### 1.1 Liveness and Readiness

```go
type HealthChecker struct {
    mu     sync.RWMutex
    checks map[string]func() error
    ready  bool
}

func NewHealthChecker() *HealthChecker {
    return &HealthChecker{
        checks: make(map[string]func() error),
    }
}

func (h *HealthChecker) AddCheck(name string, check func() error) {
    h.mu.Lock()
    defer h.mu.Unlock()
    h.checks[name] = check
}

func (h *HealthChecker) SetReady(ready bool) {
    h.mu.Lock()
    defer h.mu.Unlock()
    h.ready = ready
}

// Liveness: Is the process alive? (restart if no)
func (h *HealthChecker) LivenessHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "alive"})
}

// Readiness: Can the process serve traffic? (remove from LB if no)
func (h *HealthChecker) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
    h.mu.RLock()
    defer h.mu.RUnlock()

    if !h.ready {
        w.WriteHeader(http.StatusServiceUnavailable)
        json.NewEncoder(w).Encode(map[string]string{"status": "not ready"})
        return
    }

    results := make(map[string]string)
    allHealthy := true

    for name, check := range h.checks {
        if err := check(); err != nil {
            results[name] = err.Error()
            allHealthy = false
        } else {
            results[name] = "ok"
        }
    }

    status := http.StatusOK
    if !allHealthy {
        status = http.StatusServiceUnavailable
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]any{
        "status": map[bool]string{true: "ready", false: "not ready"}[allHealthy],
        "checks": results,
    })
}

func main() {
    health := NewHealthChecker()

    // Register dependency checks
    health.AddCheck("database", func() error {
        return db.PingContext(context.Background())
    })
    health.AddCheck("redis", func() error {
        return redis.Ping(context.Background()).Err()
    })

    mux := http.NewServeMux()
    mux.HandleFunc("GET /healthz", health.LivenessHandler)
    mux.HandleFunc("GET /readyz", health.ReadinessHandler)

    // Mark ready after initialization
    health.SetReady(true)
}
```

---

## 2. Graceful Shutdown

### 2.1 Complete Shutdown Pattern

```go
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // Initialize dependencies
    db, err := setupDatabase(ctx)
    if err != nil {
        log.Fatal(err)
    }

    health := NewHealthChecker()
    health.AddCheck("db", func() error { return db.PingContext(ctx) })

    // Setup HTTP server
    mux := http.NewServeMux()
    mux.HandleFunc("GET /healthz", health.LivenessHandler)
    mux.HandleFunc("GET /readyz", health.ReadinessHandler)
    mux.HandleFunc("GET /api/users", listUsersHandler)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
    }

    // Start server
    go func() {
        health.SetReady(true)
        slog.Info("server starting", "addr", server.Addr)
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            slog.Error("server error", "err", err)
            cancel()
        }
    }()

    // Wait for shutdown signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

    select {
    case sig := <-quit:
        slog.Info("received shutdown signal", "signal", sig)
    case <-ctx.Done():
        slog.Info("context cancelled")
    }

    // Graceful shutdown sequence
    slog.Info("starting graceful shutdown...")

    // 1. Stop accepting new traffic
    health.SetReady(false)

    // 2. Wait for load balancer to detect (Kubernetes needs this)
    time.Sleep(5 * time.Second)

    // 3. Shutdown HTTP server (waits for active requests)
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()

    if err := server.Shutdown(shutdownCtx); err != nil {
        slog.Error("server shutdown error", "err", err)
    }

    // 4. Close other resources
    db.Close()

    slog.Info("server stopped")
}
```

---

## 3. Configuration (12-Factor)

### 3.1 Environment-Based Configuration

```go
type Config struct {
    Server   ServerConfig
    Database DatabaseConfig
    Log      LogConfig
}

type ServerConfig struct {
    Host            string        `env:"SERVER_HOST" default:"0.0.0.0"`
    Port            int           `env:"SERVER_PORT" default:"8080"`
    ReadTimeout     time.Duration `env:"SERVER_READ_TIMEOUT" default:"5s"`
    WriteTimeout    time.Duration `env:"SERVER_WRITE_TIMEOUT" default:"10s"`
    ShutdownTimeout time.Duration `env:"SERVER_SHUTDOWN_TIMEOUT" default:"30s"`
}

type DatabaseConfig struct {
    URL             string        `env:"DATABASE_URL" required:"true"`
    MaxOpenConns    int           `env:"DB_MAX_OPEN_CONNS" default:"25"`
    MaxIdleConns    int           `env:"DB_MAX_IDLE_CONNS" default:"5"`
    ConnMaxLifetime time.Duration `env:"DB_CONN_MAX_LIFETIME" default:"5m"`
}

type LogConfig struct {
    Level  string `env:"LOG_LEVEL" default:"info"`
    Format string `env:"LOG_FORMAT" default:"json"`
}

func LoadConfig() (*Config, error) {
    cfg := &Config{}
    // Load from environment using reflection or a library like envconfig
    if err := loadFromEnv(cfg); err != nil {
        return nil, err
    }
    return cfg, nil
}
```

### 3.2 12-Factor Principles for Go

```go
// I. Codebase: One codebase tracked in version control
// II. Dependencies: Explicitly declared via go.mod
// III. Config: Store in environment variables

// IV. Backing services: Treat as attached resources
type App struct {
    db    *sql.DB           // Database
    cache *redis.Client     // Cache
    queue *amqp.Channel     // Message queue
}

// V. Build, release, run: Strict separation
// go build → Docker image → Kubernetes deployment

// VI. Processes: Stateless, share-nothing
// Store state in database/Redis, not in process memory

// VII. Port binding: Export service via port
server.ListenAndServe(":"+cfg.Port, handler)

// VIII. Concurrency: Scale via goroutines and horizontal scaling

// IX. Disposability: Fast startup, graceful shutdown
// Start in <1s, shutdown with Shutdown(ctx)

// X. Dev/prod parity: Keep environments similar
// Same Docker image, different environment variables

// XI. Logs: Treat as event streams
slog.Info("event", "key", "value") // Write to stdout

// XII. Admin processes: Run as one-off commands
// go run ./cmd/migrate
// go run ./cmd/seed
```

---

## 4. Observability

### 4.1 Structured Logging

```go
func setupLogger(cfg LogConfig) *slog.Logger {
    var handler slog.Handler
    opts := &slog.HandlerOptions{
        Level: parseLevel(cfg.Level),
    }

    switch cfg.Format {
    case "json":
        handler = slog.NewJSONHandler(os.Stdout, opts)
    default:
        handler = slog.NewTextHandler(os.Stdout, opts)
    }

    logger := slog.New(handler)
    slog.SetDefault(logger)
    return logger
}

// Request logging middleware
func requestLogger(logger *slog.Logger) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            start := time.Now()
            recorder := &statusRecorder{ResponseWriter: w, statusCode: 200}

            next.ServeHTTP(recorder, r)

            logger.Info("request",
                "method", r.Method,
                "path", r.URL.Path,
                "status", recorder.statusCode,
                "duration_ms", time.Since(start).Milliseconds(),
                "remote_addr", r.RemoteAddr,
                "request_id", r.Header.Get("X-Request-ID"),
            )
        })
    }
}
```

### 4.2 Metrics with expvar

```go
import "expvar"

var (
    requestCount = expvar.NewInt("requests_total")
    errorCount   = expvar.NewInt("errors_total")
    requestDur   = expvar.NewFloat("request_duration_seconds")
)

func metricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        requestCount.Add(1)

        recorder := &statusRecorder{ResponseWriter: w, statusCode: 200}
        next.ServeHTTP(recorder, r)

        duration := time.Since(start).Seconds()
        requestDur.Set(duration)

        if recorder.statusCode >= 500 {
            errorCount.Add(1)
        }
    })
}

// expvar automatically exposes /debug/vars as JSON
// For Prometheus-style metrics, use the prometheus/client_golang package
```

---

## 5. Resilience Patterns

### 5.1 Retry with Backoff

```go
type RetryConfig struct {
    MaxAttempts int
    InitialWait time.Duration
    MaxWait     time.Duration
    Multiplier  float64
}

func Retry(ctx context.Context, cfg RetryConfig, fn func() error) error {
    wait := cfg.InitialWait

    for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
        err := fn()
        if err == nil {
            return nil
        }

        if attempt == cfg.MaxAttempts {
            return fmt.Errorf("all %d attempts failed: %w", cfg.MaxAttempts, err)
        }

        slog.Warn("retry",
            "attempt", attempt,
            "error", err,
            "next_wait", wait,
        )

        select {
        case <-time.After(wait):
            wait = time.Duration(float64(wait) * cfg.Multiplier)
            if wait > cfg.MaxWait {
                wait = cfg.MaxWait
            }
        case <-ctx.Done():
            return ctx.Err()
        }
    }
    return nil
}
```

### 5.2 Timeout Wrapper

```go
func withTimeout[T any](ctx context.Context, timeout time.Duration, fn func(context.Context) (T, error)) (T, error) {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    type result struct {
        val T
        err error
    }

    ch := make(chan result, 1)
    go func() {
        val, err := fn(ctx)
        ch <- result{val, err}
    }()

    select {
    case r := <-ch:
        return r.val, r.err
    case <-ctx.Done():
        var zero T
        return zero, ctx.Err()
    }
}
```

---

## 6. Dependency Injection

### 6.1 Constructor Injection

```go
type UserService struct {
    repo   UserRepository
    cache  Cache
    logger *slog.Logger
}

func NewUserService(repo UserRepository, cache Cache, logger *slog.Logger) *UserService {
    return &UserService{
        repo:   repo,
        cache:  cache,
        logger: logger,
    }
}

type UserRepository interface {
    FindByID(ctx context.Context, id string) (*User, error)
    Save(ctx context.Context, u *User) error
}

type Cache interface {
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
}

// Wire up in main
func main() {
    logger := setupLogger()
    db := setupDatabase()
    redis := setupRedis()

    userRepo := postgres.NewUserRepository(db)
    userCache := redis.NewCache(redis)
    userService := NewUserService(userRepo, userCache, logger)

    handler := NewAPIHandler(userService)
    // ...
}
```

---

## 7. Summary

### Key Takeaways

1. **Health checks are mandatory** — liveness for restart, readiness for traffic routing.
2. **Graceful shutdown** — stop accepting traffic, finish requests, close resources, exit.
3. **12-factor configuration** — environment variables, no config files in containers.
4. **Structured logging to stdout** — let the platform handle log aggregation.
5. **Retry with exponential backoff** — handle transient failures gracefully.
6. **Dependency injection via constructors** — interfaces for testability, concrete types for production.

---

## Exercises

### Exercise 1: Production Server Template
Create a server template with health checks, graceful shutdown, structured logging, and configuration from environment variables.

### Exercise 2: Circuit Breaker
Implement a circuit breaker that tracks failure rates and opens the circuit when failures exceed a threshold.

### Exercise 3: Request Tracing
Add distributed tracing using request IDs. Propagate through HTTP headers and log with every operation.

### Exercise 4: Kubernetes Deployment
Create Kubernetes manifests (Deployment, Service, ConfigMap) for a Go service with proper health checks and resource limits.

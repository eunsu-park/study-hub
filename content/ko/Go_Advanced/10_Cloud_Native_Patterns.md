# 21. 클라우드 네이티브 패턴

**이전**: [네트워크 프로그래밍](./09_Network_Programming.md) | **다음**: [캡스톤: 마이크로서비스](./11_Capstone_Microservice.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 헬스 체크 엔드포인트(라이브니스 및 레디니스)를 구현한다
2. `context.Context`를 요청 수명 주기 관리에 사용한다
3. 시그널 처리를 통한 그레이스풀 셧다운을 구현한다
4. 12-팩터 앱 방법론을 적용한다
5. 관측성을 추가한다: 구조화된 로깅, 메트릭, 추적

---

클라우드 네이티브 개발은 클라우드 인프라를 효과적으로 활용하는 애플리케이션을 구축하는 것이다. Go는 클라우드 네이티브 도구(Docker, Kubernetes, Prometheus)의 기본 언어이며, 이러한 패턴은 Go 서비스를 프로덕션에 적합하게 만든다.

## 목차
1. [헬스 체크](#1-헬스-체크)
2. [그레이스풀 셧다운](#2-그레이스풀-셧다운)
3. [설정 (12-팩터)](#3-설정-12-팩터)
4. [관측성](#4-관측성)
5. [복원력 패턴](#5-복원력-패턴)
6. [의존성 주입](#6-의존성-주입)
7. [요약](#7-요약)

---

## 1. 헬스 체크

### 1.1 라이브니스와 레디니스

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

## 2. 그레이스풀 셧다운

### 2.1 완전한 셧다운 패턴

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

## 3. 설정 (12-팩터)

### 3.1 환경 기반 설정

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

### 3.2 Go를 위한 12-팩터 원칙

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

## 4. 관측성

### 4.1 구조화된 로깅

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

### 4.2 expvar을 사용한 메트릭

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

## 5. 복원력 패턴

### 5.1 백오프를 사용한 재시도

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

### 5.2 타임아웃 래퍼

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

## 6. 의존성 주입

### 6.1 생성자 주입

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

## 7. 요약

### 핵심 포인트

1. **헬스 체크는 필수이다** — 라이브니스는 재시작용, 레디니스는 트래픽 라우팅용이다.
2. **그레이스풀 셧다운** — 트래픽 수신 중단, 요청 완료, 리소스 해제, 종료 순으로 진행한다.
3. **12-팩터 설정** — 환경 변수를 사용하고, 컨테이너에 설정 파일을 두지 않는다.
4. **구조화된 로깅을 stdout으로** — 로그 집계는 플랫폼이 처리하도록 한다.
5. **지수 백오프를 사용한 재시도** — 일시적 장애를 우아하게 처리한다.
6. **생성자를 통한 의존성 주입** — 테스트 가능성을 위해 인터페이스를, 프로덕션에는 구체 타입을 사용한다.

---

## 연습 문제

### 연습 1: 프로덕션 서버 템플릿
헬스 체크, 그레이스풀 셧다운, 구조화된 로깅, 환경 변수 기반 설정이 포함된 서버 템플릿을 만든다.

### 연습 2: 서킷 브레이커
실패율을 추적하고 임계값을 초과하면 서킷을 여는 서킷 브레이커를 구현한다.

### 연습 3: 요청 추적
요청 ID를 사용한 분산 추적을 추가한다. HTTP 헤더를 통해 전파하고 모든 작업에서 로그와 함께 기록한다.

### 연습 4: Kubernetes 배포
적절한 헬스 체크와 리소스 제한이 포함된 Go 서비스를 위한 Kubernetes 매니페스트(Deployment, Service, ConfigMap)를 만든다.

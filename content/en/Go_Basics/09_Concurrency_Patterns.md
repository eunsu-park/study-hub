# 09. Concurrency Patterns

**Previous**: [Channels](./08_Channels.md) | **Next**: [Testing](./10_Testing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement fan-out/fan-in patterns for parallel processing
2. Build data processing pipelines with cancellation
3. Use `context.Context` for timeouts, deadlines, and cancellation
4. Create worker pools for bounded concurrency
5. Apply the errgroup pattern for coordinated error handling

---

This lesson brings together goroutines and channels into production-ready patterns. These patterns solve real problems: processing items in parallel with bounded concurrency, building cancellable pipelines, and managing the lifecycle of concurrent operations.

## Table of Contents
1. [Context Package](#1-context-package)
2. [Fan-Out / Fan-In](#2-fan-out--fan-in)
3. [Worker Pool](#3-worker-pool)
4. [Pipeline Pattern](#4-pipeline-pattern)
5. [errgroup for Coordinated Concurrency](#5-errgroup-for-coordinated-concurrency)
6. [Advanced Patterns](#6-advanced-patterns)
7. [Summary](#7-summary)

---

## 1. Context Package

### 1.1 Context Basics

`context.Context` carries deadlines, cancellation signals, and request-scoped values across API boundaries.

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func main() {
    // Background — root context (never cancelled)
    ctx := context.Background()

    // WithCancel — manual cancellation
    ctx, cancel := context.WithCancel(ctx)
    defer cancel()

    go func(ctx context.Context) {
        for {
            select {
            case <-ctx.Done():
                fmt.Println("Cancelled:", ctx.Err())
                return
            default:
                fmt.Println("Working...")
                time.Sleep(200 * time.Millisecond)
            }
        }
    }(ctx)

    time.Sleep(1 * time.Second)
    cancel() // Signal cancellation
    time.Sleep(100 * time.Millisecond)
}
```

### 1.2 Timeout and Deadline

```go
func fetchData(ctx context.Context, url string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

func main() {
    // WithTimeout — auto-cancels after duration
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel() // Always call cancel to release resources

    data, err := fetchData(ctx, "https://api.example.com/data")
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("Request timed out")
        } else {
            fmt.Println("Error:", err)
        }
        return
    }
    fmt.Println(string(data))

    // WithDeadline — auto-cancels at specific time
    deadline := time.Now().Add(10 * time.Second)
    ctx2, cancel2 := context.WithDeadline(context.Background(), deadline)
    defer cancel2()
    _ = ctx2
}
```

### 1.3 Context Values

```go
type contextKey string

const (
    requestIDKey contextKey = "requestID"
    userIDKey    contextKey = "userID"
)

func WithRequestID(ctx context.Context, id string) context.Context {
    return context.WithValue(ctx, requestIDKey, id)
}

func RequestID(ctx context.Context) string {
    if id, ok := ctx.Value(requestIDKey).(string); ok {
        return id
    }
    return ""
}

func handler(ctx context.Context) {
    fmt.Println("Request ID:", RequestID(ctx))
}

func main() {
    ctx := context.Background()
    ctx = WithRequestID(ctx, "req-12345")
    handler(ctx)
}
```

---

## 2. Fan-Out / Fan-In

### 2.1 Fan-Out

Distribute work to multiple goroutines.

```go
func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = process(input) // Each worker reads from same input
    }
    return channels
}

func process(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for val := range input {
            output <- val * val // Some expensive computation
        }
    }()
    return output
}
```

### 2.2 Fan-In

Merge multiple channels into one.

```go
func fanIn(ctx context.Context, channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    output := func(ch <-chan int) {
        defer wg.Done()
        for val := range ch {
            select {
            case merged <- val:
            case <-ctx.Done():
                return
            }
        }
    }

    wg.Add(len(channels))
    for _, ch := range channels {
        go output(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

### 2.3 Complete Fan-Out/Fan-In Example

```go
func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Generate work
    input := make(chan int)
    go func() {
        defer close(input)
        for i := 0; i < 100; i++ {
            select {
            case input <- i:
            case <-ctx.Done():
                return
            }
        }
    }()

    // Fan-out to 4 workers
    workers := fanOut(input, 4)

    // Fan-in results
    results := fanIn(ctx, workers...)

    // Consume
    for val := range results {
        fmt.Println(val)
    }
}
```

---

## 3. Worker Pool

### 3.1 Fixed Worker Pool

```go
type Job struct {
    ID      int
    Payload string
}

type Result struct {
    JobID  int
    Output string
    Err    error
}

func workerPool(ctx context.Context, numWorkers int, jobs <-chan Job) <-chan Result {
    results := make(chan Result)
    var wg sync.WaitGroup

    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done():
                    return
                default:
                    result := processJob(workerID, job)
                    results <- result
                }
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    return results
}

func processJob(workerID int, job Job) Result {
    time.Sleep(100 * time.Millisecond) // Simulate work
    return Result{
        JobID:  job.ID,
        Output: fmt.Sprintf("worker-%d processed: %s", workerID, job.Payload),
    }
}

func main() {
    ctx := context.Background()
    jobs := make(chan Job, 100)

    // Submit jobs
    go func() {
        for i := 0; i < 50; i++ {
            jobs <- Job{ID: i, Payload: fmt.Sprintf("task-%d", i)}
        }
        close(jobs)
    }()

    // Process with 5 workers
    results := workerPool(ctx, 5, jobs)

    for result := range results {
        fmt.Println(result.Output)
    }
}
```

### 3.2 Dynamic Worker Pool with Semaphore

```go
func processAll(ctx context.Context, items []string, maxConcurrent int) []error {
    sem := make(chan struct{}, maxConcurrent)
    errs := make([]error, len(items))
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, it string) {
            defer wg.Done()

            // Acquire semaphore
            select {
            case sem <- struct{}{}:
                defer func() { <-sem }()
            case <-ctx.Done():
                errs[idx] = ctx.Err()
                return
            }

            // Process
            if err := process(ctx, it); err != nil {
                errs[idx] = err
            }
        }(i, item)
    }

    wg.Wait()
    return errs
}
```

---

## 4. Pipeline Pattern

### 4.1 Stage-Based Pipeline

```go
type Stage func(ctx context.Context, in <-chan any) <-chan any

func pipeline(ctx context.Context, source <-chan any, stages ...Stage) <-chan any {
    current := source
    for _, stage := range stages {
        current = stage(ctx, current)
    }
    return current
}

// Stage: parse CSV lines
func parseStage(ctx context.Context, in <-chan any) <-chan any {
    out := make(chan any)
    go func() {
        defer close(out)
        for val := range in {
            line := val.(string)
            record := parseCSVLine(line)
            select {
            case out <- record:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

// Stage: validate records
func validateStage(ctx context.Context, in <-chan any) <-chan any {
    out := make(chan any)
    go func() {
        defer close(out)
        for val := range in {
            record := val.(Record)
            if record.IsValid() {
                select {
                case out <- record:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return out
}

// Stage: transform
func transformStage(ctx context.Context, in <-chan any) <-chan any {
    out := make(chan any)
    go func() {
        defer close(out)
        for val := range in {
            record := val.(Record)
            transformed := record.Transform()
            select {
            case out <- transformed:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}
```

### 4.2 Type-Safe Pipeline with Generics

```go
func pipelineStage[In, Out any](
    ctx context.Context,
    in <-chan In,
    fn func(In) (Out, error),
) <-chan Out {
    out := make(chan Out)
    go func() {
        defer close(out)
        for val := range in {
            result, err := fn(val)
            if err != nil {
                continue // or log, or send to error channel
            }
            select {
            case out <- result:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}
```

---

## 5. errgroup for Coordinated Concurrency

### 5.1 Basic errgroup

```go
import "golang.org/x/sync/errgroup"

func fetchAll(ctx context.Context, urls []string) ([]string, error) {
    g, ctx := errgroup.WithContext(ctx)
    results := make([]string, len(urls))

    for i, url := range urls {
        i, url := i, url // Capture loop variables
        g.Go(func() error {
            body, err := fetchURL(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            results[i] = body
            return nil
        })
    }

    // Wait for all goroutines to complete
    // Returns first non-nil error (and cancels context)
    if err := g.Wait(); err != nil {
        return nil, err
    }
    return results, nil
}
```

### 5.2 errgroup with Concurrency Limit

```go
func processItems(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(10) // Max 10 concurrent goroutines

    for _, item := range items {
        item := item
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }

    return g.Wait()
}
```

### 5.3 errgroup with Multiple Stages

```go
func startServices(ctx context.Context) error {
    g, ctx := errgroup.WithContext(ctx)

    // Start HTTP server
    g.Go(func() error {
        return httpServer.ListenAndServe()
    })

    // Start gRPC server
    g.Go(func() error {
        return grpcServer.Serve(listener)
    })

    // Start background worker
    g.Go(func() error {
        return runWorker(ctx)
    })

    // Wait for context cancellation, then shut down
    g.Go(func() error {
        <-ctx.Done()
        httpServer.Shutdown(context.Background())
        grpcServer.GracefulStop()
        return nil
    })

    return g.Wait()
}
```

---

## 6. Advanced Patterns

### 6.1 Or-Done Channel

```go
func orDone(ctx context.Context, c <-chan any) <-chan any {
    out := make(chan any)
    go func() {
        defer close(out)
        for {
            select {
            case <-ctx.Done():
                return
            case v, ok := <-c:
                if !ok {
                    return
                }
                select {
                case out <- v:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return out
}
```

### 6.2 Tee Channel

```go
func tee(ctx context.Context, in <-chan any) (<-chan any, <-chan any) {
    out1, out2 := make(chan any), make(chan any)
    go func() {
        defer close(out1)
        defer close(out2)
        for val := range orDone(ctx, in) {
            // Shadow to allow nil after send
            o1, o2 := out1, out2
            for i := 0; i < 2; i++ {
                select {
                case o1 <- val:
                    o1 = nil
                case o2 <- val:
                    o2 = nil
                }
            }
        }
    }()
    return out1, out2
}
```

### 6.3 Rate-Limited Worker

```go
func rateLimitedWorker(ctx context.Context, jobs <-chan Job, rps int) <-chan Result {
    results := make(chan Result)
    limiter := time.NewTicker(time.Second / time.Duration(rps))

    go func() {
        defer close(results)
        defer limiter.Stop()

        for job := range jobs {
            select {
            case <-limiter.C:
                result := processJob(0, job)
                select {
                case results <- result:
                case <-ctx.Done():
                    return
                }
            case <-ctx.Done():
                return
            }
        }
    }()

    return results
}
```

### 6.4 Circuit Breaker

```go
type CircuitBreaker struct {
    mu          sync.Mutex
    failures    int
    threshold   int
    resetAfter  time.Duration
    lastFailure time.Time
    state       string // "closed", "open", "half-open"
}

func NewCircuitBreaker(threshold int, resetAfter time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        threshold:  threshold,
        resetAfter: resetAfter,
        state:      "closed",
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()
    if cb.state == "open" {
        if time.Since(cb.lastFailure) > cb.resetAfter {
            cb.state = "half-open"
        } else {
            cb.mu.Unlock()
            return fmt.Errorf("circuit breaker is open")
        }
    }
    cb.mu.Unlock()

    err := fn()

    cb.mu.Lock()
    defer cb.mu.Unlock()

    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.threshold {
            cb.state = "open"
        }
        return err
    }

    cb.failures = 0
    cb.state = "closed"
    return nil
}
```

---

## 7. Summary

### Key Takeaways

1. **Context is essential** — pass `context.Context` as the first parameter for cancellation, timeouts, and deadlines.
2. **Fan-out/fan-in for parallelism** — distribute work across workers, merge results back.
3. **Worker pools bound concurrency** — prevent resource exhaustion with fixed or semaphore-based pools.
4. **Pipelines compose** — chain stages for data processing with clean cancellation.
5. **errgroup coordinates** — run concurrent tasks with first-error-cancels-all semantics.
6. **Always handle cancellation** — every goroutine should check `ctx.Done()` in select statements.
7. **Prefer errgroup over WaitGroup** — it handles errors and cancellation together.

### Pattern Selection Guide

| Problem | Pattern |
|---------|---------|
| Process N items concurrently | Worker pool |
| Transform data through stages | Pipeline |
| Fetch multiple resources | Fan-out/fan-in or errgroup |
| Limit request rate | Rate limiter |
| Handle cascading failures | Circuit breaker |
| Timeout operations | context.WithTimeout |
| Wait for first of N | select |

---

## Exercises

### Exercise 1: Image Processing Pipeline
Build a pipeline: read file paths → load images → resize → apply filter → save. Use context for cancellation and worker pools for the CPU-intensive stages.

### Exercise 2: Concurrent Web Crawler
Build a web crawler that visits pages concurrently with a worker pool. Respect rate limits, avoid revisiting URLs, and stop after a timeout.

### Exercise 3: MapReduce
Implement a simple MapReduce framework: distribute map operations across workers, shuffle results, then reduce. Test with word counting.

### Exercise 4: Service Orchestrator
Use errgroup to start multiple services (HTTP server, background worker, health checker). Implement graceful shutdown when any service fails or context is cancelled.

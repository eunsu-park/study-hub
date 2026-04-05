# 18. Performance Profiling

**Previous**: [Reflection and Code Generation](./06_Reflection_and_Codegen.md) | **Next**: [Build and Deploy](./08_Build_and_Deploy.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Profile CPU and memory usage with `pprof`
2. Use execution tracing to find latency issues
3. Write and interpret benchmarks
4. Apply common memory optimization techniques
5. Use `runtime` metrics for production monitoring

---

Go provides built-in profiling and tracing tools that work in both development and production. Unlike languages that require external profilers, Go's `pprof` and `trace` tools are part of the standard library and designed for low-overhead production use.

## Table of Contents
1. [pprof: CPU Profiling](#1-pprof-cpu-profiling)
2. [Memory Profiling](#2-memory-profiling)
3. [HTTP pprof Endpoints](#3-http-pprof-endpoints)
4. [Execution Tracing](#4-execution-tracing)
5. [Memory Optimization](#5-memory-optimization)
6. [Runtime Metrics](#6-runtime-metrics)
7. [Summary](#7-summary)

---

## 1. pprof: CPU Profiling

### 1.1 Programmatic Profiling

```go
package main

import (
    "os"
    "runtime/pprof"
    "log"
)

func main() {
    // CPU profile
    f, err := os.Create("cpu.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()

    // Your application code here
    doWork()
}

func doWork() {
    // Simulate CPU-intensive work
    result := 0
    for i := 0; i < 100_000_000; i++ {
        result += i * i
    }
}
```

```bash
# Run the program
go run main.go

# Analyze with pprof
go tool pprof cpu.prof

# Interactive commands:
# top10          — show top 10 functions by CPU time
# top -cum       — sort by cumulative time
# list doWork    — show annotated source for function
# web            — open flame graph in browser (requires graphviz)
# svg            — generate SVG flame graph
```

### 1.2 Benchmark Profiling

```bash
# Profile from benchmarks
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
go tool pprof -http=:8080 cpu.prof
```

### 1.3 Reading pprof Output

```
Type: cpu
Duration: 5.23s, Total samples: 4890ms (93.49%)
Showing nodes accounting for 4780ms, 97.75% of 4890ms total
      flat  flat%   sum%        cum   cum%
    2100ms 42.94% 42.94%     2100ms 42.94%  main.doWork
    1500ms 30.67% 73.62%     1500ms 30.67%  runtime.mallocgc
     680ms 13.91% 87.52%      680ms 13.91%  runtime.memmove
     500ms 10.22% 97.75%     4780ms 97.75%  main.processData
```

---

## 2. Memory Profiling

### 2.1 Heap Profile

```go
func main() {
    // Run your application
    doWork()

    // Write heap profile
    f, _ := os.Create("mem.prof")
    defer f.Close()
    pprof.WriteHeapProfile(f)
}
```

```bash
go tool pprof mem.prof

# Useful commands:
# top            — show top memory allocators
# top -inuse_space  — current memory usage
# top -alloc_space  — total allocations (including freed)
# list functionName — annotated source
```

### 2.2 Allocation Profiling in Benchmarks

```go
func BenchmarkAlloc(b *testing.B) {
    b.ReportAllocs() // Report allocations

    for i := 0; i < b.N; i++ {
        s := make([]byte, 1024)
        _ = s
    }
}
```

```bash
go test -bench=BenchmarkAlloc -benchmem
# BenchmarkAlloc-8    5000000    240 ns/op    1024 B/op    1 allocs/op
```

---

## 3. HTTP pprof Endpoints

### 3.1 Adding to Servers

```go
import (
    "net/http"
    _ "net/http/pprof" // Register pprof handlers
)

func main() {
    // pprof endpoints are registered on DefaultServeMux:
    // /debug/pprof/          — index
    // /debug/pprof/profile   — CPU profile (30s default)
    // /debug/pprof/heap      — heap profile
    // /debug/pprof/goroutine — goroutine stacks
    // /debug/pprof/trace     — execution trace

    // For production, serve on a separate port
    go func() {
        log.Println("pprof on :6060")
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()

    // Your main server
    mux := http.NewServeMux()
    mux.HandleFunc("/", handler)
    http.ListenAndServe(":8080", mux)
}
```

### 3.2 Collecting Profiles Remotely

```bash
# CPU profile (30 seconds)
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Goroutine dump
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Web UI
go tool pprof -http=:8081 http://localhost:6060/debug/pprof/heap

# Compare two profiles
go tool pprof -diff_base=before.prof after.prof
```

---

## 4. Execution Tracing

### 4.1 Collecting Traces

```go
import "runtime/trace"

func main() {
    f, _ := os.Create("trace.out")
    defer f.Close()

    trace.Start(f)
    defer trace.Stop()

    // Your application code
    doWork()
}
```

```bash
go test -trace=trace.out ./...
go tool trace trace.out
# Opens browser with interactive timeline showing:
# - Goroutine creation/blocking/unblocking
# - System call timing
# - GC pauses
# - Network blocking
```

### 4.2 Custom Trace Regions

```go
func processRequest(ctx context.Context, req Request) {
    // Create a trace region for this operation
    ctx, task := trace.NewTask(ctx, "processRequest")
    defer task.End()

    // Sub-regions
    trace.WithRegion(ctx, "validate", func() {
        validate(req)
    })

    trace.WithRegion(ctx, "database", func() {
        queryDB(ctx, req)
    })

    trace.WithRegion(ctx, "render", func() {
        renderResponse(req)
    })
}
```

---

## 5. Memory Optimization

### 5.1 Common Optimization Techniques

```go
// 1. Pre-allocate slices when size is known
// BAD
var results []int
for i := 0; i < 10000; i++ {
    results = append(results, i) // Causes multiple reallocations
}

// GOOD
results := make([]int, 0, 10000) // Single allocation
for i := 0; i < 10000; i++ {
    results = append(results, i)
}

// 2. Use strings.Builder instead of concatenation
// BAD
s := ""
for i := 0; i < 1000; i++ {
    s += "x" // O(n²) — creates new string each time
}

// GOOD
var b strings.Builder
b.Grow(1000)
for i := 0; i < 1000; i++ {
    b.WriteString("x") // O(n) — writes to internal buffer
}
s := b.String()

// 3. Use sync.Pool for frequently allocated objects
var bufPool = sync.Pool{
    New: func() any {
        return new(bytes.Buffer)
    },
}

func processRequest(data []byte) string {
    buf := bufPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufPool.Put(buf)
    }()
    buf.Write(data)
    return buf.String()
}

// 4. Struct field ordering — minimize padding
// BAD: 24 bytes (with padding)
type BadLayout struct {
    a bool    // 1 byte + 7 padding
    b int64   // 8 bytes
    c bool    // 1 byte + 7 padding
}

// GOOD: 16 bytes (minimal padding)
type GoodLayout struct {
    b int64   // 8 bytes
    a bool    // 1 byte
    c bool    // 1 byte + 6 padding
}
```

### 5.2 Escape Analysis

```bash
# See what escapes to heap
go build -gcflags='-m' ./...
# ./main.go:10: new(User) escapes to heap
# ./main.go:15: s does not escape
```

```go
// Stack allocation (fast) — value doesn't escape
func sum(nums []int) int {
    total := 0 // Stays on stack
    for _, n := range nums {
        total += n
    }
    return total
}

// Heap allocation (slow) — pointer escapes
func newUser(name string) *User {
    u := &User{Name: name} // Escapes to heap — returned pointer
    return u
}
```

---

## 6. Runtime Metrics

### 6.1 runtime.MemStats

```go
func printMemStats() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc:      %d MB\n", m.Alloc/1024/1024)
    fmt.Printf("TotalAlloc: %d MB\n", m.TotalAlloc/1024/1024)
    fmt.Printf("Sys:        %d MB\n", m.Sys/1024/1024)
    fmt.Printf("NumGC:      %d\n", m.NumGC)
    fmt.Printf("GCPause:    %v\n", time.Duration(m.PauseNs[(m.NumGC+255)%256]))
    fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())
}
```

### 6.2 GOGC and Memory Limit

```bash
# GOGC controls GC aggressiveness (default: 100)
GOGC=50 ./myapp    # More frequent GC, lower memory
GOGC=200 ./myapp   # Less frequent GC, higher throughput
GOGC=off ./myapp   # Disable GC (use with GOMEMLIMIT)

# GOMEMLIMIT (Go 1.19+) — soft memory limit
GOMEMLIMIT=512MiB ./myapp
```

---

## 7. Summary

### Key Takeaways

1. **pprof for CPU and memory** — `go tool pprof` is the Swiss army knife of Go profiling.
2. **HTTP pprof for production** — add `_ "net/http/pprof"` on a separate port for live profiling.
3. **`-benchmem` for allocation tracking** — always include in benchmark runs.
4. **Trace for latency** — `go tool trace` shows goroutine scheduling and GC pauses.
5. **Pre-allocate when possible** — `make([]T, 0, n)` avoids repeated allocations.
6. **`sync.Pool` for temporary objects** — reduces GC pressure for frequently allocated/freed objects.
7. **Escape analysis** — `go build -gcflags='-m'` shows what allocates on the heap.

### Profiling Workflow

```
1. Write benchmarks
2. Profile: go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
3. Analyze: go tool pprof -http=:8080 cpu.prof
4. Identify bottleneck
5. Optimize
6. Re-benchmark and compare
```

---

## Exercises

### Exercise 1: Profile and Optimize
Write a deliberately slow program (string concatenation, excessive allocation). Profile it, identify bottlenecks, and optimize. Show before/after benchmarks.

### Exercise 2: Memory Leak Detection
Create a program with an intentional goroutine leak. Use pprof goroutine profiling to find and fix it.

### Exercise 3: Struct Layout Optimizer
Write a tool that analyzes struct definitions and suggests reordered field layouts to minimize padding.

### Exercise 4: Production Monitoring
Add pprof endpoints to an HTTP server. Write a script that periodically collects profiles and alerts on anomalies.

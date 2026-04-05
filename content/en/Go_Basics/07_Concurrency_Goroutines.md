# 07. Concurrency: Goroutines

**Previous**: [Packages and Modules](./06_Packages_and_Modules.md) | **Next**: [Channels](./08_Channels.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Launch goroutines and understand their lightweight nature
2. Synchronize goroutines with `sync.WaitGroup`
3. Protect shared data with `sync.Mutex` and `sync.RWMutex`
4. Use `sync.Once`, `sync.Map`, and atomic operations
5. Identify and prevent common concurrency bugs

---

Concurrency is Go's defining feature. Goroutines — lightweight threads managed by the Go runtime — make concurrent programming accessible. Where OS threads cost megabytes of stack space and require expensive context switches, goroutines start with kilobytes and are multiplexed onto a small pool of OS threads.

## Table of Contents
1. [Goroutine Basics](#1-goroutine-basics)
2. [sync.WaitGroup](#2-syncwaitgroup)
3. [Mutex: Protecting Shared Data](#3-mutex-protecting-shared-data)
4. [RWMutex and Once](#4-rwmutex-and-once)
5. [Atomic Operations](#5-atomic-operations)
6. [Race Conditions and Detection](#6-race-conditions-and-detection)
7. [Summary](#7-summary)

---

## 1. Goroutine Basics

### 1.1 Launching Goroutines

```go
package main

import (
    "fmt"
    "time"
)

func sayHello(name string) {
    for i := 0; i < 3; i++ {
        fmt.Printf("[%s] Hello #%d\n", name, i+1)
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    // Launch a goroutine — prefix with 'go' keyword
    go sayHello("goroutine-1")
    go sayHello("goroutine-2")

    // main() is itself a goroutine
    sayHello("main")

    // WARNING: if main() exits, all goroutines are killed
    // We'll fix this with WaitGroup shortly
}
```

### 1.2 Goroutine Lifecycle

```go
func main() {
    // Goroutines are extremely lightweight
    // You can easily launch thousands
    for i := 0; i < 10000; i++ {
        go func(id int) {
            // Each goroutine does some work
            _ = id * id
        }(i) // Pass i as argument to avoid closure capture bug
    }

    // Goroutine characteristics:
    // - Initial stack: ~2-8 KB (grows as needed, up to 1 GB)
    // - Scheduled by Go runtime (M:N scheduling)
    // - No goroutine ID accessible (by design)
    // - Cannot be forcibly killed from outside
    // - Garbage collected when function returns

    time.Sleep(time.Second)
    fmt.Println("Done")
}
```

### 1.3 Anonymous Goroutines

```go
func main() {
    // Anonymous function as goroutine
    go func() {
        fmt.Println("I'm anonymous!")
    }()

    // With parameters
    message := "hello"
    go func(msg string) {
        fmt.Println(msg)
    }(message) // Pass value — don't capture mutable variable

    time.Sleep(100 * time.Millisecond)
}
```

---

## 2. sync.WaitGroup

### 2.1 Basic WaitGroup

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done() // Decrement counter when goroutine completes

    fmt.Printf("Worker %d starting\n", id)
    time.Sleep(time.Duration(id) * 100 * time.Millisecond)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup

    for i := 1; i <= 5; i++ {
        wg.Add(1) // Increment counter BEFORE launching goroutine
        go worker(i, &wg)
    }

    wg.Wait() // Block until counter reaches zero
    fmt.Println("All workers completed")
}
```

### 2.2 WaitGroup Best Practices

```go
// GOOD: Add before launching goroutine
func good() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait()
}

// BAD: Add inside goroutine — race condition!
func bad() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        go func(id int) {
            wg.Add(1) // BAD: might not execute before Wait()
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait() // Might return before all goroutines even start
}

// Pattern: Collect results from goroutines
func fetchAll(urls []string) []string {
    var (
        wg      sync.WaitGroup
        mu      sync.Mutex
        results []string
    )

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()
            result := fetch(u)
            mu.Lock()
            results = append(results, result)
            mu.Unlock()
        }(url)
    }

    wg.Wait()
    return results
}
```

### 2.3 Parallel Processing Pattern

```go
func processItems(items []Item) []Result {
    results := make([]Result, len(items))
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, it Item) {
            defer wg.Done()
            results[idx] = process(it) // Safe: each goroutine writes to unique index
        }(i, item)
    }

    wg.Wait()
    return results
}
```

---

## 3. Mutex: Protecting Shared Data

### 3.1 The Problem: Data Race

```go
// WITHOUT mutex — data race!
func unsafeCounter() {
    count := 0
    var wg sync.WaitGroup

    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            count++ // DATA RACE: multiple goroutines read-modify-write
        }()
    }

    wg.Wait()
    fmt.Println(count) // Not 1000! Undefined behavior.
}
```

### 3.2 sync.Mutex

```go
package main

import (
    "fmt"
    "sync"
)

// SafeCounter is safe for concurrent use
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}

func main() {
    counter := &SafeCounter{}
    var wg sync.WaitGroup

    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }

    wg.Wait()
    fmt.Println(counter.Value()) // Always 1000
}
```

### 3.3 Mutex Patterns

```go
// Thread-safe map
type SafeMap struct {
    mu   sync.Mutex
    data map[string]int
}

func NewSafeMap() *SafeMap {
    return &SafeMap{data: make(map[string]int)}
}

func (m *SafeMap) Set(key string, value int) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
}

func (m *SafeMap) Get(key string) (int, bool) {
    m.mu.Lock()
    defer m.mu.Unlock()
    v, ok := m.data[key]
    return v, ok
}

func (m *SafeMap) Delete(key string) {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.data, key)
}

// Rule: mutex should be unexported and close to the data it protects
type Cache struct {
    mu    sync.Mutex // Guards items
    items map[string]*CacheItem

    // statsMu sync.Mutex // Separate mutex for independent data
    // hits    int
    // misses  int
}
```

---

## 4. RWMutex and Once

### 4.1 sync.RWMutex

For read-heavy workloads, `RWMutex` allows multiple concurrent readers.

```go
type Config struct {
    mu       sync.RWMutex
    settings map[string]string
}

func NewConfig() *Config {
    return &Config{settings: make(map[string]string)}
}

// Multiple goroutines can read simultaneously
func (c *Config) Get(key string) string {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.settings[key]
}

// Only one goroutine can write (blocks all readers)
func (c *Config) Set(key, value string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.settings[key] = value
}

// When to use RWMutex vs Mutex:
// - RWMutex: many readers, few writers (config, caches)
// - Mutex: frequent writes, or critical section is very short
```

### 4.2 sync.Once

Execute a function exactly once, regardless of how many goroutines call it.

```go
type Database struct {
    once sync.Once
    conn *sql.DB
}

func (db *Database) Connection() *sql.DB {
    db.once.Do(func() {
        // This runs exactly once, even if called from many goroutines
        var err error
        db.conn, err = sql.Open("postgres", "...")
        if err != nil {
            log.Fatal(err)
        }
    })
    return db.conn
}

// Singleton pattern
var (
    instance *Service
    once     sync.Once
)

func GetService() *Service {
    once.Do(func() {
        instance = &Service{}
        instance.init()
    })
    return instance
}
```

### 4.3 sync.Map

```go
// sync.Map is optimized for two common patterns:
// 1. Key is written once and read many times
// 2. Multiple goroutines read/write disjoint sets of keys

func main() {
    var m sync.Map

    // Store
    m.Store("key1", "value1")
    m.Store("key2", 42)

    // Load
    if val, ok := m.Load("key1"); ok {
        fmt.Println(val.(string))
    }

    // LoadOrStore — load if exists, store if not
    actual, loaded := m.LoadOrStore("key3", "default")
    fmt.Println(actual, loaded) // "default" false

    // Delete
    m.Delete("key1")

    // Range
    m.Range(func(key, value any) bool {
        fmt.Println(key, value)
        return true // return false to stop iteration
    })

    // When to use sync.Map vs Mutex+map:
    // sync.Map: long-lived caches, many goroutines, disjoint keys
    // Mutex+map: known key set, need complex operations (len, iterate-modify)
}
```

---

## 5. Atomic Operations

### 5.1 sync/atomic Package

For simple counters and flags, atomic operations are faster than mutex.

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

func main() {
    var counter int64
    var wg sync.WaitGroup

    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            atomic.AddInt64(&counter, 1) // Atomic increment
        }()
    }

    wg.Wait()
    fmt.Println(atomic.LoadInt64(&counter)) // 1000

    // Atomic operations
    var val int64
    atomic.StoreInt64(&val, 42)                    // Set
    fmt.Println(atomic.LoadInt64(&val))             // Get: 42
    atomic.AddInt64(&val, 10)                       // Add: 52
    old := atomic.SwapInt64(&val, 100)              // Swap: old=52
    swapped := atomic.CompareAndSwapInt64(&val, 100, 200) // CAS
    fmt.Println(old, swapped, atomic.LoadInt64(&val))
}
```

### 5.2 atomic.Value (Go 1.4+) and atomic Types (Go 1.19+)

```go
// atomic.Value — for any type
var config atomic.Value

func loadConfig() {
    cfg := readConfigFromFile()
    config.Store(cfg) // Atomic store
}

func getConfig() *Config {
    return config.Load().(*Config) // Atomic load
}

// Go 1.19+ typed atomics
var (
    counter atomic.Int64
    flag    atomic.Bool
    ptr     atomic.Pointer[Config]
)

func main() {
    counter.Add(1)
    counter.Add(1)
    fmt.Println(counter.Load()) // 2

    flag.Store(true)
    fmt.Println(flag.Load()) // true

    cfg := &Config{Port: 8080}
    ptr.Store(cfg)
    fmt.Println(ptr.Load().Port) // 8080
}
```

---

## 6. Race Conditions and Detection

### 6.1 Common Race Conditions

```go
// Race 1: Shared variable without synchronization
func race1() {
    shared := 0
    go func() { shared = 1 }()
    go func() { shared = 2 }()
    // Who wins? Undefined!
}

// Race 2: Check-then-act
func race2(cache map[string]int, key string) int {
    // NOT safe — another goroutine could modify between check and act
    if val, ok := cache[key]; ok {
        return val
    }
    cache[key] = compute(key) // Race!
    return cache[key]
}

// Race 3: Slice append from multiple goroutines
func race3() {
    var results []int
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(n int) {
            defer wg.Done()
            results = append(results, n) // Race! append is not safe
        }(i)
    }
    wg.Wait()
}
```

### 6.2 Race Detector

```bash
# Build and run with race detector
go run -race main.go
go test -race ./...
go build -race -o myapp

# Output example:
# WARNING: DATA RACE
# Write at 0x00c0000b4010 by goroutine 7:
#   main.main.func1()
#       /path/main.go:15 +0x38
# Previous write at 0x00c0000b4010 by goroutine 6:
#   main.main.func1()
#       /path/main.go:15 +0x38
```

### 6.3 Debugging Goroutine Leaks

```go
import "runtime"

func main() {
    // Monitor goroutine count
    fmt.Println("Goroutines:", runtime.NumGoroutine())

    // After running your program for a while, this number should be stable
    // If it keeps growing, you have a goroutine leak

    // Common leak: goroutine blocked on channel forever
    ch := make(chan int)
    go func() {
        val := <-ch // Blocks forever if nothing sends
        fmt.Println(val)
    }()
    // If we never send on ch, the goroutine leaks

    // Fix: use context for cancellation (covered in Lesson 09)
}
```

---

## 7. Summary

### Key Takeaways

1. **Goroutines are cheap** — launch thousands without worry. They start with small stacks that grow as needed.
2. **`go` keyword launches goroutines** — the calling function continues immediately.
3. **`sync.WaitGroup` for coordination** — `Add` before launch, `Done` in goroutine, `Wait` to block.
4. **`sync.Mutex` for shared data** — lock before access, unlock with `defer`.
5. **`sync.RWMutex` for read-heavy** — multiple readers, one writer.
6. **`sync/atomic` for simple values** — faster than mutex for counters and flags.
7. **Always run `-race`** — the race detector finds data races at runtime.

### Concurrency Primitives Summary

| Primitive | Use Case | Performance |
|-----------|----------|-------------|
| `sync.Mutex` | Protect shared data | Good for short critical sections |
| `sync.RWMutex` | Read-heavy workloads | Better when reads >> writes |
| `sync.WaitGroup` | Wait for goroutines | Zero overhead when done |
| `sync.Once` | One-time initialization | Near-zero after first call |
| `sync.Map` | Concurrent map | Good for append-only patterns |
| `atomic.Int64` | Counters, flags | Fastest for simple operations |

---

## Exercises

### Exercise 1: Parallel Web Scraper
Write a function that fetches N URLs concurrently using goroutines and WaitGroup. Collect results in a thread-safe way. Limit concurrency to M goroutines.

### Exercise 2: Thread-Safe Cache
Implement a thread-safe cache with `Get`, `Set`, `Delete`, and `Size` methods using `sync.RWMutex`. Add TTL (time-to-live) support.

### Exercise 3: Race Detector Practice
Write three programs with intentional data races. Run them with `-race` and fix each one using the appropriate synchronization primitive.

### Exercise 4: Concurrent Counter Benchmark
Benchmark three counter implementations: Mutex, RWMutex, and atomic. Compare performance with varying read/write ratios.

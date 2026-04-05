# 08. Channels

**Previous**: [Concurrency: Goroutines](./07_Concurrency_Goroutines.md) | **Next**: [Concurrency Patterns](./09_Concurrency_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and use unbuffered and buffered channels
2. Use directional channel types for safety
3. Multiplex channels with `select`
4. Implement common channel patterns: done, timeout, ticker
5. Avoid channel pitfalls: deadlocks, leaks, and panics

---

Channels are Go's primary mechanism for communication between goroutines. They embody the Go proverb: "Don't communicate by sharing memory; share memory by communicating." Channels provide both data transfer and synchronization.

## Table of Contents
1. [Channel Basics](#1-channel-basics)
2. [Buffered Channels](#2-buffered-channels)
3. [Directional Channels](#3-directional-channels)
4. [Select Statement](#4-select-statement)
5. [Channel Patterns](#5-channel-patterns)
6. [Channel Pitfalls](#6-channel-pitfalls)
7. [Summary](#7-summary)

---

## 1. Channel Basics

### 1.1 Creating and Using Channels

```go
package main

import "fmt"

func main() {
    // Create an unbuffered channel of int
    ch := make(chan int)

    // Send and receive in separate goroutines
    go func() {
        ch <- 42 // Send — blocks until someone receives
    }()

    value := <-ch // Receive — blocks until someone sends
    fmt.Println(value) // 42

    // Channel for strings
    msgCh := make(chan string)
    go func() {
        msgCh <- "hello"
        msgCh <- "world"
    }()

    fmt.Println(<-msgCh) // "hello"
    fmt.Println(<-msgCh) // "world"
}
```

### 1.2 Unbuffered Channels (Synchronous)

```go
func main() {
    ch := make(chan string) // Unbuffered — capacity 0

    go func() {
        fmt.Println("Sending...")
        ch <- "data" // Blocks until receiver is ready
        fmt.Println("Sent!")
    }()

    time.Sleep(time.Second) // Simulate delay
    fmt.Println("Receiving...")
    val := <-ch // Both sender and receiver synchronize here
    fmt.Println("Received:", val)

    // Output order is guaranteed:
    // Sending...
    // (1 second pause)
    // Receiving...
    // Sent!
    // Received: data
}
```

### 1.3 Closing Channels

```go
func producer(ch chan<- int) {
    for i := 0; i < 5; i++ {
        ch <- i
    }
    close(ch) // Signal that no more values will be sent
}

func main() {
    ch := make(chan int)
    go producer(ch)

    // Range over channel — stops when channel is closed
    for val := range ch {
        fmt.Println(val) // 0, 1, 2, 3, 4
    }

    // Check if channel is closed
    ch2 := make(chan int)
    close(ch2)
    val, ok := <-ch2
    fmt.Println(val, ok) // 0 false (zero value, not ok)

    // Rules:
    // - Only the SENDER should close a channel
    // - Sending on a closed channel PANICS
    // - Receiving from a closed channel returns zero value immediately
    // - Closing an already-closed channel PANICS
}
```

---

## 2. Buffered Channels

### 2.1 Buffered Channel Basics

```go
func main() {
    // Buffered channel — capacity 3
    ch := make(chan int, 3)

    // Can send without a receiver (up to buffer size)
    ch <- 1 // Doesn't block
    ch <- 2 // Doesn't block
    ch <- 3 // Doesn't block
    // ch <- 4 // Would block! Buffer is full

    fmt.Println(len(ch), cap(ch)) // 3 3

    // Receive
    fmt.Println(<-ch) // 1 (FIFO)
    fmt.Println(<-ch) // 2
    fmt.Println(<-ch) // 3
}
```

### 2.2 When to Use Buffered Channels

```go
// 1. Decouple producer and consumer speeds
func logAsync(messages <-chan string) {
    for msg := range messages {
        writeToFile(msg) // Slow I/O
    }
}

func main() {
    logCh := make(chan string, 100) // Buffer absorbs bursts
    go logAsync(logCh)

    for i := 0; i < 1000; i++ {
        logCh <- fmt.Sprintf("event %d", i) // Fast producer
    }
    close(logCh)
}

// 2. Semaphore — limit concurrency
func processWithLimit(items []Item, maxConcurrent int) {
    sem := make(chan struct{}, maxConcurrent) // Buffered as semaphore
    var wg sync.WaitGroup

    for _, item := range items {
        wg.Add(1)
        sem <- struct{}{} // Acquire — blocks when buffer full

        go func(it Item) {
            defer wg.Done()
            defer func() { <-sem }() // Release
            process(it)
        }(item)
    }
    wg.Wait()
}

// 3. Channel of size 1 — mutex alternative
func main() {
    mu := make(chan struct{}, 1)

    mu <- struct{}{}   // Lock
    // critical section
    <-mu               // Unlock
}
```

---

## 3. Directional Channels

### 3.1 Send-Only and Receive-Only

```go
// chan<- T — send-only channel
// <-chan T — receive-only channel

func producer(out chan<- int) {
    for i := 0; i < 10; i++ {
        out <- i
    }
    close(out)
}

func consumer(in <-chan int) {
    for val := range in {
        fmt.Println("Got:", val)
    }
}

func main() {
    ch := make(chan int, 5)

    // Bidirectional channel is implicitly converted
    go producer(ch) // chan int → chan<- int (OK)
    consumer(ch)    // chan int → <-chan int (OK)

    // Cannot convert back:
    // var bidir chan int = sendOnly // COMPILE ERROR
}
```

### 3.2 Generator Pattern

```go
// Generator returns a receive-only channel
func fibonacci(n int) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        a, b := 0, 1
        for i := 0; i < n; i++ {
            ch <- a
            a, b = b, a+b
        }
    }()
    return ch
}

func main() {
    for val := range fibonacci(10) {
        fmt.Println(val) // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    }
}
```

---

## 4. Select Statement

### 4.1 Basic Select

`select` lets a goroutine wait on multiple channel operations.

```go
func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(100 * time.Millisecond)
        ch1 <- "from ch1"
    }()

    go func() {
        time.Sleep(200 * time.Millisecond)
        ch2 <- "from ch2"
    }()

    // Wait for whichever is ready first
    select {
    case msg := <-ch1:
        fmt.Println(msg)
    case msg := <-ch2:
        fmt.Println(msg)
    }
    // Prints "from ch1" (it's faster)
}
```

### 4.2 Select with Default (Non-Blocking)

```go
func main() {
    ch := make(chan int, 1)

    // Non-blocking receive
    select {
    case val := <-ch:
        fmt.Println("received:", val)
    default:
        fmt.Println("no value ready") // This executes
    }

    // Non-blocking send
    ch <- 42
    select {
    case ch <- 100:
        fmt.Println("sent 100")
    default:
        fmt.Println("channel full") // This executes (buffer is 1, already has 42)
    }
}
```

### 4.3 Timeout Pattern

```go
func fetchWithTimeout(url string, timeout time.Duration) (string, error) {
    resultCh := make(chan string, 1)
    errCh := make(chan error, 1)

    go func() {
        result, err := http.Get(url)
        if err != nil {
            errCh <- err
            return
        }
        defer result.Body.Close()
        body, _ := io.ReadAll(result.Body)
        resultCh <- string(body)
    }()

    select {
    case result := <-resultCh:
        return result, nil
    case err := <-errCh:
        return "", err
    case <-time.After(timeout):
        return "", fmt.Errorf("timeout after %v", timeout)
    }
}
```

### 4.4 Ticker and Done

```go
func periodicTask(done <-chan struct{}) {
    ticker := time.NewTicker(500 * time.Millisecond)
    defer ticker.Stop()

    for {
        select {
        case t := <-ticker.C:
            fmt.Println("Tick at", t.Format("15:04:05.000"))
        case <-done:
            fmt.Println("Stopping periodic task")
            return
        }
    }
}

func main() {
    done := make(chan struct{})

    go periodicTask(done)

    time.Sleep(2 * time.Second)
    close(done) // Signal all goroutines to stop
    time.Sleep(100 * time.Millisecond)
}
```

---

## 5. Channel Patterns

### 5.1 Done Channel

```go
func doWork(done <-chan struct{}) <-chan int {
    results := make(chan int)
    go func() {
        defer close(results)
        for i := 0; ; i++ {
            select {
            case <-done:
                return // Clean shutdown
            case results <- i:
                time.Sleep(100 * time.Millisecond)
            }
        }
    }()
    return results
}

func main() {
    done := make(chan struct{})
    results := doWork(done)

    for i := 0; i < 5; i++ {
        fmt.Println(<-results)
    }
    close(done) // Signal goroutine to stop
}
```

### 5.2 Pipeline

```go
func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            out <- n
        }
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            out <- n * n
        }
    }()
    return out
}

func filter(in <-chan int, pred func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if pred(n) {
                out <- n
            }
        }
    }()
    return out
}

func main() {
    // Pipeline: generate → square → filter (even)
    nums := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(nums)
    evens := filter(squared, func(n int) bool { return n%2 == 0 })

    for val := range evens {
        fmt.Println(val) // 4, 16, 36, 64, 100
    }
}
```

### 5.3 Fan-Out / Fan-In

```go
// Fan-out: multiple goroutines read from same channel
// Fan-in: multiple channels merged into one

func fanIn(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    // Start a goroutine for each input channel
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for val := range c {
                merged <- val
            }
        }(ch)
    }

    // Close merged when all input channels are done
    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

---

## 6. Channel Pitfalls

### 6.1 Deadlocks

```go
// Deadlock 1: Send with no receiver on unbuffered channel
func deadlock1() {
    ch := make(chan int)
    ch <- 42 // Blocks forever — no goroutine to receive
    // fatal error: all goroutines are asleep - deadlock!
}

// Deadlock 2: Circular wait
func deadlock2() {
    ch1 := make(chan int)
    ch2 := make(chan int)

    go func() {
        val := <-ch1 // Wait for ch1
        ch2 <- val   // Send to ch2
    }()

    val := <-ch2 // Wait for ch2 — but goroutine waits for ch1!
    ch1 <- val
}

// Deadlock 3: Range over unclosed channel
func deadlock3() {
    ch := make(chan int, 3)
    ch <- 1
    ch <- 2
    ch <- 3
    // close(ch) // MISSING! Range will block forever after 3 values
    for val := range ch {
        fmt.Println(val)
    }
}
```

### 6.2 Goroutine Leaks

```go
// Leak: goroutine blocked on send, no one receives
func leak() <-chan int {
    ch := make(chan int)
    go func() {
        result := expensiveComputation()
        ch <- result // If caller doesn't receive, goroutine leaks
    }()
    return ch
}

// Fix: use buffered channel of size 1
func noLeak() <-chan int {
    ch := make(chan int, 1) // Buffered — goroutine can send and exit
    go func() {
        result := expensiveComputation()
        ch <- result // Won't block even if no one receives
    }()
    return ch
}

// Fix: use done channel for cancellation
func cancelable(done <-chan struct{}) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        result := expensiveComputation()
        select {
        case ch <- result:
        case <-done: // Caller can cancel
        }
    }()
    return ch
}
```

### 6.3 Channel Operation Summary

| Operation | nil channel | Closed channel | Active channel |
|-----------|-------------|----------------|----------------|
| Send `ch <- v` | Block forever | **PANIC** | Block or succeed |
| Receive `<-ch` | Block forever | Zero value, `ok=false` | Block or succeed |
| Close `close(ch)` | **PANIC** | **PANIC** | Succeed |
| Range `for v := range ch` | Block forever | Exits loop | Iterate values |

---

## 7. Summary

### Key Takeaways

1. **Unbuffered channels synchronize** — sender and receiver rendezvous. Use for coordination.
2. **Buffered channels decouple** — producer can run ahead of consumer. Use for performance.
3. **Directional types enforce safety** — `chan<- T` and `<-chan T` prevent misuse at compile time.
4. **`select` multiplexes** — wait on multiple channels, implement timeouts and cancellation.
5. **Close signals completion** — only sender closes. Range over channel for clean iteration.
6. **Avoid nil channel operations** — they block forever. Use this intentionally in select to disable a case.
7. **Watch for leaks** — every goroutine must have a way to exit. Use done channels or context.

---

## Exercises

### Exercise 1: Chat System
Build a simple chat system where multiple goroutines (users) send messages through channels. Implement a central hub that broadcasts messages to all connected users.

### Exercise 2: Pipeline Processing
Create a data processing pipeline: `readCSV → parseRows → filterValid → transform → writeOutput`. Each stage is a goroutine connected by channels.

### Exercise 3: Rate Limiter
Implement a token-bucket rate limiter using channels. Allow N requests per second with burst capacity B.

### Exercise 4: Timeout Orchestrator
Write a function that makes 5 concurrent API calls and returns the results of whichever 3 finish first, canceling the remaining 2.

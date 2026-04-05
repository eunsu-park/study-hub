# 18. 성능 프로파일링

**이전**: [리플렉션과 코드 생성](./06_Reflection_and_Codegen.md) | **다음**: [빌드와 배포](./08_Build_and_Deploy.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `pprof`로 CPU 및 메모리 사용량을 프로파일링한다
2. 실행 추적을 사용하여 지연 시간 문제를 찾는다
3. 벤치마크를 작성하고 해석한다
4. 일반적인 메모리 최적화 기법을 적용한다
5. 프로덕션 모니터링에 `runtime` 메트릭을 사용한다

---

Go는 개발과 프로덕션 환경 모두에서 동작하는 내장 프로파일링 및 추적 도구를 제공한다. 외부 프로파일러가 필요한 다른 언어와 달리, Go의 `pprof`와 `trace` 도구는 표준 라이브러리의 일부이며 낮은 오버헤드로 프로덕션에서 사용할 수 있도록 설계되었다.

## 목차
1. [pprof: CPU 프로파일링](#1-pprof-cpu-프로파일링)
2. [메모리 프로파일링](#2-메모리-프로파일링)
3. [HTTP pprof 엔드포인트](#3-http-pprof-엔드포인트)
4. [실행 추적](#4-실행-추적)
5. [메모리 최적화](#5-메모리-최적화)
6. [런타임 메트릭](#6-런타임-메트릭)
7. [요약](#7-요약)

---

## 1. pprof: CPU 프로파일링

### 1.1 프로그래밍 방식 프로파일링

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

### 1.2 벤치마크 프로파일링

```bash
# Profile from benchmarks
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
go tool pprof -http=:8080 cpu.prof
```

### 1.3 pprof 출력 읽기

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

## 2. 메모리 프로파일링

### 2.1 힙 프로파일

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

### 2.2 벤치마크에서의 할당 프로파일링

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

## 3. HTTP pprof 엔드포인트

### 3.1 서버에 추가하기

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

### 3.2 원격으로 프로파일 수집

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

## 4. 실행 추적

### 4.1 추적 수집

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

### 4.2 커스텀 추적 영역

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

## 5. 메모리 최적화

### 5.1 일반적인 최적화 기법

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

### 5.2 이스케이프 분석

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

## 6. 런타임 메트릭

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

### 6.2 GOGC와 메모리 제한

```bash
# GOGC controls GC aggressiveness (default: 100)
GOGC=50 ./myapp    # More frequent GC, lower memory
GOGC=200 ./myapp   # Less frequent GC, higher throughput
GOGC=off ./myapp   # Disable GC (use with GOMEMLIMIT)

# GOMEMLIMIT (Go 1.19+) — soft memory limit
GOMEMLIMIT=512MiB ./myapp
```

---

## 7. 요약

### 핵심 포인트

1. **CPU와 메모리에는 pprof** — `go tool pprof`는 Go 프로파일링의 만능 도구이다.
2. **프로덕션에는 HTTP pprof** — 별도 포트에 `_ "net/http/pprof"`를 추가하여 실시간 프로파일링을 한다.
3. **할당 추적에는 `-benchmem`** — 벤치마크 실행 시 항상 포함한다.
4. **지연 시간에는 Trace** — `go tool trace`로 고루틴 스케줄링과 GC 일시 정지를 확인한다.
5. **가능하면 사전 할당한다** — `make([]T, 0, n)`으로 반복 할당을 방지한다.
6. **임시 객체에는 `sync.Pool`** — 자주 할당/해제되는 객체의 GC 부하를 줄인다.
7. **이스케이프 분석** — `go build -gcflags='-m'`으로 힙에 할당되는 항목을 확인한다.

### 프로파일링 워크플로우

```
1. 벤치마크를 작성한다
2. 프로파일링한다: go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
3. 분석한다: go tool pprof -http=:8080 cpu.prof
4. 병목 지점을 식별한다
5. 최적화한다
6. 재벤치마크하고 비교한다
```

---

## 연습 문제

### 연습 1: 프로파일링 및 최적화
의도적으로 느린 프로그램(문자열 연결, 과도한 할당)을 작성한다. 프로파일링하여 병목 지점을 식별하고 최적화한다. 최적화 전후의 벤치마크를 보여준다.

### 연습 2: 메모리 누수 탐지
의도적인 고루틴 누수가 있는 프로그램을 만든다. pprof 고루틴 프로파일링을 사용하여 찾아 수정한다.

### 연습 3: 구조체 레이아웃 최적화기
구조체 정의를 분석하고 패딩을 최소화하기 위한 필드 재배치를 제안하는 도구를 작성한다.

### 연습 4: 프로덕션 모니터링
HTTP 서버에 pprof 엔드포인트를 추가한다. 주기적으로 프로파일을 수집하고 이상 징후에 대해 경고하는 스크립트를 작성한다.

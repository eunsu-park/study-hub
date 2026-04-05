# 09. 동시성 패턴

**이전**: [채널](./08_Channels.md) | **다음**: [테스팅](./10_Testing.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 병렬 처리를 위한 fan-out/fan-in 패턴을 구현한다
2. 취소가 가능한 데이터 처리 파이프라인을 구축한다
3. 타임아웃, 데드라인, 취소를 위해 `context.Context`를 사용한다
4. 제한된 동시성을 위한 워커 풀을 생성한다
5. 조율된 에러 처리를 위한 errgroup 패턴을 적용한다

---

이 레슨은 고루틴과 채널을 프로덕션에서 사용할 수 있는 패턴으로 결합한다. 이 패턴들은 실제 문제를 해결한다: 제한된 동시성으로 항목을 병렬 처리하고, 취소 가능한 파이프라인을 구축하고, 동시 연산의 생명주기를 관리한다.

## 목차
1. [Context 패키지](#1-context-패키지)
2. [Fan-Out / Fan-In](#2-fan-out--fan-in)
3. [워커 풀](#3-워커-풀)
4. [파이프라인 패턴](#4-파이프라인-패턴)
5. [조율된 동시성을 위한 errgroup](#5-조율된-동시성을-위한-errgroup)
6. [고급 패턴](#6-고급-패턴)
7. [요약](#7-요약)

---

## 1. Context 패키지

### 1.1 Context 기초

`context.Context`는 API 경계를 넘어 데드라인, 취소 신호, 요청 범위 값을 전달한다.

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func main() {
    // Background — 루트 context (절대 취소되지 않음)
    ctx := context.Background()

    // WithCancel — 수동 취소
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
    cancel() // 취소 신호
    time.Sleep(100 * time.Millisecond)
}
```

### 1.2 타임아웃과 데드라인

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
    // WithTimeout — 지정 시간 후 자동 취소
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel() // 항상 cancel을 호출하여 리소스 해제

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

    // WithDeadline — 특정 시각에 자동 취소
    deadline := time.Now().Add(10 * time.Second)
    ctx2, cancel2 := context.WithDeadline(context.Background(), deadline)
    defer cancel2()
    _ = ctx2
}
```

### 1.3 Context 값

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

여러 고루틴에 작업을 분배한다.

```go
func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = process(input) // 각 워커가 같은 입력에서 읽기
    }
    return channels
}

func process(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for val := range input {
            output <- val * val // 비용이 큰 연산
        }
    }()
    return output
}
```

### 2.2 Fan-In

여러 채널을 하나로 병합한다.

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

### 2.3 완전한 Fan-Out/Fan-In 예제

```go
func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 작업 생성
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

    // 4개 워커로 fan-out
    workers := fanOut(input, 4)

    // 결과를 fan-in
    results := fanIn(ctx, workers...)

    // 소비
    for val := range results {
        fmt.Println(val)
    }
}
```

---

## 3. 워커 풀

### 3.1 고정 워커 풀

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
    time.Sleep(100 * time.Millisecond) // 작업 시뮬레이션
    return Result{
        JobID:  job.ID,
        Output: fmt.Sprintf("worker-%d processed: %s", workerID, job.Payload),
    }
}

func main() {
    ctx := context.Background()
    jobs := make(chan Job, 100)

    // 작업 제출
    go func() {
        for i := 0; i < 50; i++ {
            jobs <- Job{ID: i, Payload: fmt.Sprintf("task-%d", i)}
        }
        close(jobs)
    }()

    // 5개 워커로 처리
    results := workerPool(ctx, 5, jobs)

    for result := range results {
        fmt.Println(result.Output)
    }
}
```

### 3.2 세마포어를 사용한 동적 워커 풀

```go
func processAll(ctx context.Context, items []string, maxConcurrent int) []error {
    sem := make(chan struct{}, maxConcurrent)
    errs := make([]error, len(items))
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, it string) {
            defer wg.Done()

            // 세마포어 획득
            select {
            case sem <- struct{}{}:
                defer func() { <-sem }()
            case <-ctx.Done():
                errs[idx] = ctx.Err()
                return
            }

            // 처리
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

## 4. 파이프라인 패턴

### 4.1 단계 기반 파이프라인

```go
type Stage func(ctx context.Context, in <-chan any) <-chan any

func pipeline(ctx context.Context, source <-chan any, stages ...Stage) <-chan any {
    current := source
    for _, stage := range stages {
        current = stage(ctx, current)
    }
    return current
}

// 단계: CSV 라인 파싱
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

// 단계: 레코드 유효성 검사
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

// 단계: 변환
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

### 4.2 제네릭을 사용한 타입 안전 파이프라인

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
                continue // 또는 로그, 또는 에러 채널로 전송
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

## 5. 조율된 동시성을 위한 errgroup

### 5.1 기본 errgroup

```go
import "golang.org/x/sync/errgroup"

func fetchAll(ctx context.Context, urls []string) ([]string, error) {
    g, ctx := errgroup.WithContext(ctx)
    results := make([]string, len(urls))

    for i, url := range urls {
        i, url := i, url // 루프 변수 캡처
        g.Go(func() error {
            body, err := fetchURL(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            results[i] = body
            return nil
        })
    }

    // 모든 고루틴이 완료될 때까지 대기
    // 첫 번째 non-nil 에러를 반환 (그리고 context를 취소)
    if err := g.Wait(); err != nil {
        return nil, err
    }
    return results, nil
}
```

### 5.2 동시성 제한이 있는 errgroup

```go
func processItems(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(10) // 최대 10개의 동시 고루틴

    for _, item := range items {
        item := item
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }

    return g.Wait()
}
```

### 5.3 여러 단계의 errgroup

```go
func startServices(ctx context.Context) error {
    g, ctx := errgroup.WithContext(ctx)

    // HTTP 서버 시작
    g.Go(func() error {
        return httpServer.ListenAndServe()
    })

    // gRPC 서버 시작
    g.Go(func() error {
        return grpcServer.Serve(listener)
    })

    // 백그라운드 워커 시작
    g.Go(func() error {
        return runWorker(ctx)
    })

    // context 취소를 기다린 후 종료
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

## 6. 고급 패턴

### 6.1 Or-Done 채널

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

### 6.2 Tee 채널

```go
func tee(ctx context.Context, in <-chan any) (<-chan any, <-chan any) {
    out1, out2 := make(chan any), make(chan any)
    go func() {
        defer close(out1)
        defer close(out2)
        for val := range orDone(ctx, in) {
            // 송신 후 nil을 허용하기 위해 섀도잉
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

### 6.3 속도 제한 워커

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

### 6.4 서킷 브레이커

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

## 7. 요약

### 핵심 포인트

1. **Context는 필수이다** — 취소, 타임아웃, 데드라인을 위해 `context.Context`를 첫 번째 매개변수로 전달한다.
2. **병렬 처리를 위한 Fan-out/fan-in** — 워커들에게 작업을 분배하고, 결과를 다시 병합한다.
3. **워커 풀은 동시성을 제한한다** — 고정 또는 세마포어 기반 풀로 리소스 고갈을 방지한다.
4. **파이프라인은 조합한다** — 깨끗한 취소와 함께 데이터 처리를 위한 단계를 체인한다.
5. **errgroup은 조율한다** — 첫 번째 에러가 모두를 취소하는 의미론으로 동시 태스크를 실행한다.
6. **항상 취소를 처리한다** — 모든 고루틴은 select 문에서 `ctx.Done()`을 확인해야 한다.
7. **WaitGroup보다 errgroup을 선호한다** — 에러와 취소를 함께 처리한다.

### 패턴 선택 가이드

| 문제 | 패턴 |
|------|------|
| N개 항목을 동시에 처리 | 워커 풀 |
| 단계를 거쳐 데이터 변환 | 파이프라인 |
| 여러 리소스 가져오기 | Fan-out/fan-in 또는 errgroup |
| 요청 속도 제한 | 속도 제한기 |
| 연쇄 실패 처리 | 서킷 브레이커 |
| 연산 타임아웃 | context.WithTimeout |
| N개 중 첫 번째 대기 | select |

---

## 연습 문제

### 연습 1: 이미지 처리 파이프라인
파이프라인을 구축한다: 파일 경로 읽기 → 이미지 로드 → 리사이즈 → 필터 적용 → 저장. 취소를 위해 context를 사용하고 CPU 집약적 단계에 워커 풀을 사용한다.

### 연습 2: 동시 웹 크롤러
워커 풀을 사용하여 페이지를 동시에 방문하는 웹 크롤러를 구축한다. 속도 제한을 준수하고, URL 재방문을 피하고, 타임아웃 후 중지한다.

### 연습 3: MapReduce
간단한 MapReduce 프레임워크를 구현한다: 워커들에게 map 연산을 분배하고, 결과를 셔플한 다음 reduce한다. 단어 세기로 테스트한다.

### 연습 4: 서비스 오케스트레이터
errgroup을 사용하여 여러 서비스(HTTP 서버, 백그라운드 워커, 헬스 체커)를 시작한다. 서비스가 실패하거나 context가 취소될 때 우아한 종료를 구현한다.

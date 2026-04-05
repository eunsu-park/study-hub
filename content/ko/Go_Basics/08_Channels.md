# 08. 채널

**이전**: [동시성: 고루틴](./07_Concurrency_Goroutines.md) | **다음**: [동시성 패턴](./09_Concurrency_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 비버퍼 채널과 버퍼 채널을 생성하고 사용한다
2. 안전성을 위해 방향성 채널 타입을 사용한다
3. `select`로 채널을 다중화한다
4. 일반적인 채널 패턴(done, timeout, ticker)을 구현한다
5. 채널 함정(데드락, 누수, 패닉)을 피한다

---

채널은 고루틴 간 통신을 위한 Go의 핵심 메커니즘이다. Go 격언인 "메모리를 공유하여 통신하지 말고, 통신하여 메모리를 공유하라"를 구현한다. 채널은 데이터 전송과 동기화를 모두 제공한다.

## 목차
1. [채널 기초](#1-채널-기초)
2. [버퍼 채널](#2-버퍼-채널)
3. [방향성 채널](#3-방향성-채널)
4. [Select 문](#4-select-문)
5. [채널 패턴](#5-채널-패턴)
6. [채널 함정](#6-채널-함정)
7. [요약](#7-요약)

---

## 1. 채널 기초

### 1.1 채널 생성과 사용

```go
package main

import "fmt"

func main() {
    // int 타입의 비버퍼 채널 생성
    ch := make(chan int)

    // 별도의 고루틴에서 송신과 수신
    go func() {
        ch <- 42 // 송신 — 누군가 수신할 때까지 블록
    }()

    value := <-ch // 수신 — 누군가 송신할 때까지 블록
    fmt.Println(value) // 42

    // 문자열용 채널
    msgCh := make(chan string)
    go func() {
        msgCh <- "hello"
        msgCh <- "world"
    }()

    fmt.Println(<-msgCh) // "hello"
    fmt.Println(<-msgCh) // "world"
}
```

### 1.2 비버퍼 채널 (동기식)

```go
func main() {
    ch := make(chan string) // 비버퍼 — 용량 0

    go func() {
        fmt.Println("Sending...")
        ch <- "data" // 수신자가 준비될 때까지 블록
        fmt.Println("Sent!")
    }()

    time.Sleep(time.Second) // 지연 시뮬레이션
    fmt.Println("Receiving...")
    val := <-ch // 송신자와 수신자가 여기서 동기화
    fmt.Println("Received:", val)

    // 출력 순서가 보장됨:
    // Sending...
    // (1초 대기)
    // Receiving...
    // Sent!
    // Received: data
}
```

### 1.3 채널 닫기

```go
func producer(ch chan<- int) {
    for i := 0; i < 5; i++ {
        ch <- i
    }
    close(ch) // 더 이상 값을 보내지 않겠다는 신호
}

func main() {
    ch := make(chan int)
    go producer(ch)

    // 채널에 대해 range — 채널이 닫히면 중단
    for val := range ch {
        fmt.Println(val) // 0, 1, 2, 3, 4
    }

    // 채널이 닫혔는지 확인
    ch2 := make(chan int)
    close(ch2)
    val, ok := <-ch2
    fmt.Println(val, ok) // 0 false (제로 값, ok 아님)

    // 규칙:
    // - 송신자만 채널을 닫아야 한다
    // - 닫힌 채널에 송신하면 패닉 발생
    // - 닫힌 채널에서 수신하면 즉시 제로 값 반환
    // - 이미 닫힌 채널을 다시 닫으면 패닉 발생
}
```

---

## 2. 버퍼 채널

### 2.1 버퍼 채널 기초

```go
func main() {
    // 버퍼 채널 — 용량 3
    ch := make(chan int, 3)

    // 수신자 없이 송신 가능 (버퍼 크기까지)
    ch <- 1 // 블록하지 않음
    ch <- 2 // 블록하지 않음
    ch <- 3 // 블록하지 않음
    // ch <- 4 // 블록됨! 버퍼가 가득 참

    fmt.Println(len(ch), cap(ch)) // 3 3

    // 수신
    fmt.Println(<-ch) // 1 (FIFO)
    fmt.Println(<-ch) // 2
    fmt.Println(<-ch) // 3
}
```

### 2.2 버퍼 채널을 사용해야 할 때

```go
// 1. 생산자와 소비자 속도 분리
func logAsync(messages <-chan string) {
    for msg := range messages {
        writeToFile(msg) // 느린 I/O
    }
}

func main() {
    logCh := make(chan string, 100) // 버퍼가 버스트를 흡수
    go logAsync(logCh)

    for i := 0; i < 1000; i++ {
        logCh <- fmt.Sprintf("event %d", i) // 빠른 생산자
    }
    close(logCh)
}

// 2. 세마포어 — 동시성 제한
func processWithLimit(items []Item, maxConcurrent int) {
    sem := make(chan struct{}, maxConcurrent) // 세마포어로 사용하는 버퍼 채널
    var wg sync.WaitGroup

    for _, item := range items {
        wg.Add(1)
        sem <- struct{}{} // 획득 — 버퍼가 가득 차면 블록

        go func(it Item) {
            defer wg.Done()
            defer func() { <-sem }() // 해제
            process(it)
        }(item)
    }
    wg.Wait()
}

// 3. 크기 1의 채널 — 뮤텍스 대안
func main() {
    mu := make(chan struct{}, 1)

    mu <- struct{}{}   // 잠금
    // 임계 구역
    <-mu               // 잠금 해제
}
```

---

## 3. 방향성 채널

### 3.1 송신 전용과 수신 전용

```go
// chan<- T — 송신 전용 채널
// <-chan T — 수신 전용 채널

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

    // 양방향 채널이 암시적으로 변환됨
    go producer(ch) // chan int → chan<- int (OK)
    consumer(ch)    // chan int → <-chan int (OK)

    // 역변환은 불가:
    // var bidir chan int = sendOnly // 컴파일 오류
}
```

### 3.2 제너레이터 패턴

```go
// 제너레이터는 수신 전용 채널을 반환한다
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

## 4. Select 문

### 4.1 기본 Select

`select`는 고루틴이 여러 채널 연산을 대기할 수 있게 한다.

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

    // 먼저 준비된 것을 대기
    select {
    case msg := <-ch1:
        fmt.Println(msg)
    case msg := <-ch2:
        fmt.Println(msg)
    }
    // "from ch1" 출력 (더 빠르므로)
}
```

### 4.2 Default가 있는 Select (논블로킹)

```go
func main() {
    ch := make(chan int, 1)

    // 논블로킹 수신
    select {
    case val := <-ch:
        fmt.Println("received:", val)
    default:
        fmt.Println("no value ready") // 이것이 실행됨
    }

    // 논블로킹 송신
    ch <- 42
    select {
    case ch <- 100:
        fmt.Println("sent 100")
    default:
        fmt.Println("channel full") // 이것이 실행됨 (버퍼가 1이고, 이미 42가 있음)
    }
}
```

### 4.3 타임아웃 패턴

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

### 4.4 Ticker와 Done

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
    close(done) // 모든 고루틴에 중지 신호
    time.Sleep(100 * time.Millisecond)
}
```

---

## 5. 채널 패턴

### 5.1 Done 채널

```go
func doWork(done <-chan struct{}) <-chan int {
    results := make(chan int)
    go func() {
        defer close(results)
        for i := 0; ; i++ {
            select {
            case <-done:
                return // 깨끗한 종료
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
    close(done) // 고루틴에 중지 신호
}
```

### 5.2 파이프라인

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
    // 파이프라인: generate → square → filter (짝수)
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
// Fan-out: 여러 고루틴이 같은 채널에서 읽기
// Fan-in: 여러 채널을 하나로 병합

func fanIn(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    // 각 입력 채널에 대해 고루틴 시작
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for val := range c {
                merged <- val
            }
        }(ch)
    }

    // 모든 입력 채널이 완료되면 merged 닫기
    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

---

## 6. 채널 함정

### 6.1 데드락

```go
// 데드락 1: 비버퍼 채널에서 수신자 없이 송신
func deadlock1() {
    ch := make(chan int)
    ch <- 42 // 영원히 블록 — 수신할 고루틴이 없음
    // fatal error: all goroutines are asleep - deadlock!
}

// 데드락 2: 순환 대기
func deadlock2() {
    ch1 := make(chan int)
    ch2 := make(chan int)

    go func() {
        val := <-ch1 // ch1 대기
        ch2 <- val   // ch2로 송신
    }()

    val := <-ch2 // ch2 대기 — 그런데 고루틴은 ch1을 대기 중!
    ch1 <- val
}

// 데드락 3: 닫히지 않은 채널에 대한 range
func deadlock3() {
    ch := make(chan int, 3)
    ch <- 1
    ch <- 2
    ch <- 3
    // close(ch) // 누락! 3개 값 후에 range가 영원히 블록
    for val := range ch {
        fmt.Println(val)
    }
}
```

### 6.2 고루틴 누수

```go
// 누수: 고루틴이 송신에서 블록되어, 아무도 수신하지 않음
func leak() <-chan int {
    ch := make(chan int)
    go func() {
        result := expensiveComputation()
        ch <- result // 호출자가 수신하지 않으면, 고루틴 누수
    }()
    return ch
}

// 수정: 크기 1의 버퍼 채널 사용
func noLeak() <-chan int {
    ch := make(chan int, 1) // 버퍼 — 고루틴이 송신하고 종료 가능
    go func() {
        result := expensiveComputation()
        ch <- result // 아무도 수신하지 않아도 블록되지 않음
    }()
    return ch
}

// 수정: 취소용 done 채널 사용
func cancelable(done <-chan struct{}) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        result := expensiveComputation()
        select {
        case ch <- result:
        case <-done: // 호출자가 취소 가능
        }
    }()
    return ch
}
```

### 6.3 채널 연산 요약

| 연산 | nil 채널 | 닫힌 채널 | 활성 채널 |
|------|----------|-----------|-----------|
| 송신 `ch <- v` | 영원히 블록 | **패닉** | 블록 또는 성공 |
| 수신 `<-ch` | 영원히 블록 | 제로 값, `ok=false` | 블록 또는 성공 |
| 닫기 `close(ch)` | **패닉** | **패닉** | 성공 |
| Range `for v := range ch` | 영원히 블록 | 루프 종료 | 값 반복 |

---

## 7. 요약

### 핵심 포인트

1. **비버퍼 채널은 동기화한다** — 송신자와 수신자가 만나는 지점이다. 조율에 사용한다.
2. **버퍼 채널은 분리한다** — 생산자가 소비자보다 앞서 실행할 수 있다. 성능에 사용한다.
3. **방향성 타입은 안전성을 강제한다** — `chan<- T`와 `<-chan T`는 컴파일 시점에 오용을 방지한다.
4. **`select`는 다중화한다** — 여러 채널을 대기하고, 타임아웃과 취소를 구현한다.
5. **Close는 완료를 알린다** — 송신자만 닫는다. 깨끗한 반복을 위해 채널에 range를 사용한다.
6. **nil 채널 연산을 피한다** — 영원히 블록한다. select에서 case를 비활성화하기 위해 의도적으로 사용할 수 있다.
7. **누수를 주의한다** — 모든 고루틴은 종료할 방법이 있어야 한다. done 채널이나 context를 사용한다.

---

## 연습 문제

### 연습 1: 채팅 시스템
여러 고루틴(사용자)이 채널을 통해 메시지를 보내는 간단한 채팅 시스템을 구축한다. 연결된 모든 사용자에게 메시지를 브로드캐스트하는 중앙 허브를 구현한다.

### 연습 2: 파이프라인 처리
데이터 처리 파이프라인을 생성한다: `readCSV → parseRows → filterValid → transform → writeOutput`. 각 단계는 채널로 연결된 고루틴이다.

### 연습 3: 속도 제한기
채널을 사용하여 토큰 버킷 속도 제한기를 구현한다. 버스트 용량 B로 초당 N개의 요청을 허용한다.

### 연습 4: 타임아웃 오케스트레이터
5개의 동시 API 호출을 수행하고 먼저 완료된 3개의 결과를 반환하며, 나머지 2개를 취소하는 함수를 작성한다.

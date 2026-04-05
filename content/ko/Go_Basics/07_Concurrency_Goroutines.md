# 07. 동시성: 고루틴

**이전**: [패키지와 모듈](./06_Packages_and_Modules.md) | **다음**: [채널](./08_Channels.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 고루틴(goroutine)을 실행하고 경량 특성을 이해할 수 있다
2. `sync.WaitGroup`으로 고루틴을 동기화할 수 있다
3. `sync.Mutex`와 `sync.RWMutex`로 공유 데이터를 보호할 수 있다
4. `sync.Once`, `sync.Map`, 원자적 연산(atomic operation)을 사용할 수 있다
5. 일반적인 동시성 버그를 식별하고 방지할 수 있다

---

동시성(concurrency)은 Go의 정의적 특성이다. 고루틴 — Go 런타임이 관리하는 경량 스레드 — 은 동시성 프로그래밍을 접근하기 쉽게 만든다. OS 스레드가 메가바이트의 스택 공간을 필요로 하고 비싼 컨텍스트 스위칭이 필요한 반면, 고루틴은 킬로바이트로 시작하며 소수의 OS 스레드 풀에 다중화(multiplex)된다.

## 목차
1. [고루틴 기초](#1-고루틴-기초)
2. [sync.WaitGroup](#2-syncwaitgroup)
3. [뮤텍스: 공유 데이터 보호](#3-뮤텍스-공유-데이터-보호)
4. [RWMutex와 Once](#4-rwmutex와-once)
5. [원자적 연산](#5-원자적-연산)
6. [경쟁 조건과 탐지](#6-경쟁-조건과-탐지)
7. [요약](#7-요약)

---

## 1. 고루틴 기초

### 1.1 고루틴 실행

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
    // 고루틴 실행 — 'go' 키워드를 앞에 붙인다
    go sayHello("goroutine-1")
    go sayHello("goroutine-2")

    // main()도 그 자체로 고루틴이다
    sayHello("main")

    // 경고: main()이 종료되면 모든 고루틴이 종료된다
    // 이것은 곧 WaitGroup으로 해결할 것이다
}
```

### 1.2 고루틴 생명 주기

```go
func main() {
    // 고루틴은 극도로 경량이다
    // 수천 개를 쉽게 실행할 수 있다
    for i := 0; i < 10000; i++ {
        go func(id int) {
            // 각 고루틴이 작업을 수행한다
            _ = id * id
        }(i) // 클로저 캡처 버그를 피하기 위해 i를 인자로 전달한다
    }

    // 고루틴 특성:
    // - 초기 스택: ~2-8 KB (필요에 따라 최대 1 GB까지 증가)
    // - Go 런타임에 의해 스케줄링됨 (M:N 스케줄링)
    // - 고루틴 ID에 접근할 수 없음 (의도적 설계)
    // - 외부에서 강제로 종료할 수 없음
    // - 함수가 반환되면 가비지 컬렉션됨

    time.Sleep(time.Second)
    fmt.Println("완료")
}
```

### 1.3 익명 고루틴

```go
func main() {
    // 익명 함수를 고루틴으로 실행
    go func() {
        fmt.Println("나는 익명이다!")
    }()

    // 매개변수와 함께
    message := "hello"
    go func(msg string) {
        fmt.Println(msg)
    }(message) // 값을 전달한다 — 가변 변수를 캡처하지 마라

    time.Sleep(100 * time.Millisecond)
}
```

---

## 2. sync.WaitGroup

### 2.1 기본 WaitGroup

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done() // 고루틴이 완료되면 카운터를 감소시킨다

    fmt.Printf("워커 %d 시작\n", id)
    time.Sleep(time.Duration(id) * 100 * time.Millisecond)
    fmt.Printf("워커 %d 완료\n", id)
}

func main() {
    var wg sync.WaitGroup

    for i := 1; i <= 5; i++ {
        wg.Add(1) // 고루틴 실행 전에 카운터를 증가시킨다
        go worker(i, &wg)
    }

    wg.Wait() // 카운터가 0이 될 때까지 블록한다
    fmt.Println("모든 워커 완료")
}
```

### 2.2 WaitGroup 모범 사례

```go
// 좋음: 고루틴 실행 전에 Add를 호출한다
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

// 나쁨: 고루틴 내부에서 Add — 경쟁 조건!
func bad() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        go func(id int) {
            wg.Add(1) // 나쁨: Wait() 전에 실행되지 않을 수 있다
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait() // 모든 고루틴이 시작되기 전에 반환될 수 있다
}

// 패턴: 고루틴에서 결과를 수집한다
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

### 2.3 병렬 처리 패턴

```go
func processItems(items []Item) []Result {
    results := make([]Result, len(items))
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, it Item) {
            defer wg.Done()
            results[idx] = process(it) // 안전: 각 고루틴이 고유한 인덱스에 쓴다
        }(i, item)
    }

    wg.Wait()
    return results
}
```

---

## 3. 뮤텍스: 공유 데이터 보호

### 3.1 문제: 데이터 경쟁

```go
// 뮤텍스 없이 — 데이터 경쟁(data race)!
func unsafeCounter() {
    count := 0
    var wg sync.WaitGroup

    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            count++ // 데이터 경쟁: 여러 고루틴이 읽기-수정-쓰기를 수행
        }()
    }

    wg.Wait()
    fmt.Println(count) // 1000이 아니다! 정의되지 않은 동작.
}
```

### 3.2 sync.Mutex

```go
package main

import (
    "fmt"
    "sync"
)

// SafeCounter는 동시 사용에 안전하다
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
    fmt.Println(counter.Value()) // 항상 1000
}
```

### 3.3 뮤텍스 패턴

```go
// 스레드 안전 맵
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

// 규칙: 뮤텍스는 내보내지 않아야 하며 보호하는 데이터에 가까이 있어야 한다
type Cache struct {
    mu    sync.Mutex // items를 보호한다
    items map[string]*CacheItem

    // statsMu sync.Mutex // 독립적인 데이터를 위한 별도 뮤텍스
    // hits    int
    // misses  int
}
```

---

## 4. RWMutex와 Once

### 4.1 sync.RWMutex

읽기가 많은 워크로드에서 `RWMutex`는 여러 동시 읽기를 허용한다.

```go
type Config struct {
    mu       sync.RWMutex
    settings map[string]string
}

func NewConfig() *Config {
    return &Config{settings: make(map[string]string)}
}

// 여러 고루틴이 동시에 읽을 수 있다
func (c *Config) Get(key string) string {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.settings[key]
}

// 하나의 고루틴만 쓸 수 있다 (모든 읽기를 차단한다)
func (c *Config) Set(key, value string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.settings[key] = value
}

// RWMutex vs Mutex를 사용할 때:
// - RWMutex: 읽기가 많고, 쓰기가 적은 경우 (설정, 캐시)
// - Mutex: 쓰기가 빈번하거나, 임계 영역이 매우 짧은 경우
```

### 4.2 sync.Once

몇 개의 고루틴이 호출하든 함수를 정확히 한 번만 실행한다.

```go
type Database struct {
    once sync.Once
    conn *sql.DB
}

func (db *Database) Connection() *sql.DB {
    db.once.Do(func() {
        // 이것은 많은 고루틴에서 호출되더라도 정확히 한 번만 실행된다
        var err error
        db.conn, err = sql.Open("postgres", "...")
        if err != nil {
            log.Fatal(err)
        }
    })
    return db.conn
}

// 싱글턴 패턴
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
// sync.Map은 두 가지 일반적인 패턴에 최적화되어 있다:
// 1. 키가 한 번 쓰이고 여러 번 읽히는 경우
// 2. 여러 고루틴이 분리된 키 집합을 읽고/쓰는 경우

func main() {
    var m sync.Map

    // 저장
    m.Store("key1", "value1")
    m.Store("key2", 42)

    // 로드
    if val, ok := m.Load("key1"); ok {
        fmt.Println(val.(string))
    }

    // LoadOrStore — 있으면 로드, 없으면 저장
    actual, loaded := m.LoadOrStore("key3", "default")
    fmt.Println(actual, loaded) // "default" false

    // 삭제
    m.Delete("key1")

    // 순회
    m.Range(func(key, value any) bool {
        fmt.Println(key, value)
        return true // false를 반환하면 순회를 중단한다
    })

    // sync.Map vs Mutex+map을 사용할 때:
    // sync.Map: 오래 살아남는 캐시, 많은 고루틴, 분리된 키
    // Mutex+map: 알려진 키 집합, 복잡한 연산이 필요한 경우 (len, 순회-수정)
}
```

---

## 5. 원자적 연산

### 5.1 sync/atomic 패키지

간단한 카운터와 플래그에는 원자적 연산(atomic operation)이 뮤텍스보다 빠르다.

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
            atomic.AddInt64(&counter, 1) // 원자적 증가
        }()
    }

    wg.Wait()
    fmt.Println(atomic.LoadInt64(&counter)) // 1000

    // 원자적 연산
    var val int64
    atomic.StoreInt64(&val, 42)                    // 설정
    fmt.Println(atomic.LoadInt64(&val))             // 가져오기: 42
    atomic.AddInt64(&val, 10)                       // 더하기: 52
    old := atomic.SwapInt64(&val, 100)              // 교환: old=52
    swapped := atomic.CompareAndSwapInt64(&val, 100, 200) // CAS
    fmt.Println(old, swapped, atomic.LoadInt64(&val))
}
```

### 5.2 atomic.Value (Go 1.4+)와 원자적 타입 (Go 1.19+)

```go
// atomic.Value — 모든 타입을 위한
var config atomic.Value

func loadConfig() {
    cfg := readConfigFromFile()
    config.Store(cfg) // 원자적 저장
}

func getConfig() *Config {
    return config.Load().(*Config) // 원자적 로드
}

// Go 1.19+ 타입이 있는 원자적 타입
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

## 6. 경쟁 조건과 탐지

### 6.1 일반적인 경쟁 조건

```go
// 경쟁 1: 동기화 없는 공유 변수
func race1() {
    shared := 0
    go func() { shared = 1 }()
    go func() { shared = 2 }()
    // 누가 이기는가? 정의되지 않음!
}

// 경쟁 2: 확인 후 행동
func race2(cache map[string]int, key string) int {
    // 안전하지 않음 — 확인과 행동 사이에 다른 고루틴이 수정할 수 있다
    if val, ok := cache[key]; ok {
        return val
    }
    cache[key] = compute(key) // 경쟁!
    return cache[key]
}

// 경쟁 3: 여러 고루틴에서 슬라이스 append
func race3() {
    var results []int
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(n int) {
            defer wg.Done()
            results = append(results, n) // 경쟁! append는 안전하지 않다
        }(i)
    }
    wg.Wait()
}
```

### 6.2 경쟁 탐지기

```bash
# 경쟁 탐지기로 빌드하고 실행한다
go run -race main.go
go test -race ./...
go build -race -o myapp

# 출력 예:
# WARNING: DATA RACE
# Write at 0x00c0000b4010 by goroutine 7:
#   main.main.func1()
#       /path/main.go:15 +0x38
# Previous write at 0x00c0000b4010 by goroutine 6:
#   main.main.func1()
#       /path/main.go:15 +0x38
```

### 6.3 고루틴 누수 디버깅

```go
import "runtime"

func main() {
    // 고루틴 수를 모니터링한다
    fmt.Println("고루틴:", runtime.NumGoroutine())

    // 프로그램을 한동안 실행한 후, 이 수가 안정적이어야 한다
    // 계속 증가하면 고루틴 누수가 있다

    // 일반적인 누수: 채널에서 영원히 블록되는 고루틴
    ch := make(chan int)
    go func() {
        val := <-ch // 아무것도 보내지 않으면 영원히 블록된다
        fmt.Println(val)
    }()
    // ch에 절대 보내지 않으면 고루틴이 누수된다

    // 해결: 취소를 위해 context를 사용한다 (레슨 09에서 다룸)
}
```

---

## 7. 요약

### 핵심 포인트

1. **고루틴은 저렴하다** — 걱정 없이 수천 개를 실행할 수 있다. 필요에 따라 증가하는 작은 스택으로 시작한다.
2. **`go` 키워드가 고루틴을 실행한다** — 호출하는 함수는 즉시 계속된다.
3. **`sync.WaitGroup`으로 조정** — 실행 전에 `Add`, 고루틴에서 `Done`, 블록하려면 `Wait`를 사용한다.
4. **`sync.Mutex`로 공유 데이터 보호** — 접근 전에 잠그고, `defer`로 해제한다.
5. **`sync.RWMutex`는 읽기가 많을 때** — 여러 읽기, 하나의 쓰기.
6. **`sync/atomic`은 간단한 값에** — 카운터와 플래그에는 뮤텍스보다 빠르다.
7. **항상 `-race`를 실행하라** — 경쟁 탐지기가 런타임에 데이터 경쟁을 찾는다.

### 동시성 프리미티브 요약

| 프리미티브 | 사용 사례 | 성능 |
|-----------|----------|------|
| `sync.Mutex` | 공유 데이터 보호 | 짧은 임계 영역에 좋음 |
| `sync.RWMutex` | 읽기가 많은 워크로드 | 읽기 >> 쓰기일 때 더 좋음 |
| `sync.WaitGroup` | 고루틴 대기 | 완료 시 오버헤드 없음 |
| `sync.Once` | 일회성 초기화 | 첫 호출 후 거의 오버헤드 없음 |
| `sync.Map` | 동시성 맵 | 추가 전용 패턴에 좋음 |
| `atomic.Int64` | 카운터, 플래그 | 간단한 연산에 가장 빠름 |

---

## 연습 문제

### 연습 1: 병렬 웹 스크레이퍼
고루틴과 WaitGroup을 사용하여 N개의 URL을 동시에 가져오는 함수를 작성하라. 스레드 안전한 방식으로 결과를 수집하라. 동시성을 M개의 고루틴으로 제한하라.

### 연습 2: 스레드 안전 캐시
`sync.RWMutex`를 사용하여 `Get`, `Set`, `Delete`, `Size` 메서드를 가진 스레드 안전 캐시를 구현하라. TTL(time-to-live) 지원을 추가하라.

### 연습 3: 경쟁 탐지기 연습
의도적인 데이터 경쟁이 있는 세 개의 프로그램을 작성하라. `-race`로 실행하고 적절한 동기화 프리미티브를 사용하여 각각을 수정하라.

### 연습 4: 동시성 카운터 벤치마크
Mutex, RWMutex, atomic의 세 가지 카운터 구현을 벤치마크하라. 다양한 읽기/쓰기 비율로 성능을 비교하라.

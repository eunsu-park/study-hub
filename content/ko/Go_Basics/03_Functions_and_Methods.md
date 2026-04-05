# 03. 함수와 메서드

**이전**: [복합 타입](./02_Composite_Types.md) | **다음**: [인터페이스](./04_Interfaces.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 함수 타입을 매개변수와 반환값으로 사용할 수 있다
2. 값 리시버(value receiver)와 포인터 리시버(pointer receiver)로 메서드를 정의할 수 있다
3. 포인터 리시버와 값 리시버를 언제 사용해야 하는지 이해할 수 있다
4. 상태를 캡처하고 변경하는 클로저(closure)를 작성할 수 있다
5. map, filter, reduce 같은 함수형 패턴을 적용할 수 있다

---

Go는 함수를 일급 시민(first-class citizen)으로 취급한다 — 변수에 저장하고, 인자로 전달하고, 다른 함수에서 반환할 수 있다. 메서드(method)는 클래스 없이 타입에 동작을 붙이며, 간단한 리시버 구문을 사용하여 Go만의 독특한 객체 지향적 특성을 부여한다.

## 목차
1. [함수 타입](#1-함수-타입)
2. [고차 함수](#2-고차-함수)
3. [클로저](#3-클로저)
4. [메서드](#4-메서드)
5. [포인터 vs 값 리시버](#5-포인터-vs-값-리시버)
6. [메서드 집합과 임베딩](#6-메서드-집합과-임베딩)
7. [요약](#7-요약)

---

## 1. 함수 타입

### 1.1 타입으로서의 함수 시그니처

모든 함수는 매개변수와 반환 타입으로 정의되는 타입을 가진다.

```go
package main

import "fmt"

// 이름이 있는 함수 타입
type MathFunc func(float64, float64) float64

func add(a, b float64) float64 { return a + b }
func mul(a, b float64) float64 { return a * b }

func main() {
    // 함수 타입의 변수
    var op MathFunc
    op = add
    fmt.Println(op(3, 4)) // 7

    op = mul
    fmt.Println(op(3, 4)) // 12

    // 맵에 함수 저장
    ops := map[string]MathFunc{
        "+": add,
        "*": mul,
        "-": func(a, b float64) float64 { return a - b },
    }

    for symbol, fn := range ops {
        fmt.Printf("10 %s 3 = %.0f\n", symbol, fn(10, 3))
    }
}
```

### 1.2 명확성을 위한 타입 정의

```go
// 술어(predicate) 함수 타입
type Predicate func(int) bool

// 변환기(transformer) 함수 타입
type Transformer func(string) string

// 정렬을 위한 비교기(comparator)
type Comparator func(a, b interface{}) int

// 미들웨어 패턴 (HTTP에서 일반적)
type Middleware func(http.Handler) http.Handler
```

---

## 2. 고차 함수

### 2.1 매개변수로서의 함수

```go
package main

import (
    "fmt"
    "strings"
)

// 각 요소에 함수를 적용한다
func mapStrings(ss []string, f func(string) string) []string {
    result := make([]string, len(ss))
    for i, s := range ss {
        result[i] = f(s)
    }
    return result
}

// 술어와 일치하는 요소를 필터링한다
func filterInts(nums []int, pred func(int) bool) []int {
    var result []int
    for _, n := range nums {
        if pred(n) {
            result = append(result, n)
        }
    }
    return result
}

// 단일 값으로 축소한다
func reduce(nums []int, initial int, f func(int, int) int) int {
    acc := initial
    for _, n := range nums {
        acc = f(acc, n)
    }
    return acc
}

func main() {
    // Map
    words := []string{"hello", "world", "go"}
    upper := mapStrings(words, strings.ToUpper)
    fmt.Println(upper) // [HELLO WORLD GO]

    // Filter
    nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    evens := filterInts(nums, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4 6 8 10]

    // Reduce
    sum := reduce(nums, 0, func(a, b int) int { return a + b })
    fmt.Println(sum) // 55

    product := reduce([]int{1, 2, 3, 4, 5}, 1, func(a, b int) int { return a * b })
    fmt.Println(product) // 120
}
```

### 2.2 함수를 반환하는 함수

```go
// multiplier는 n을 곱하는 함수를 반환한다
func multiplier(n int) func(int) int {
    return func(x int) int {
        return x * n
    }
}

// 두 함수를 합성한다: f(g(x))
func compose(f, g func(int) int) func(int) int {
    return func(x int) int {
        return f(g(x))
    }
}

// 술어 결합자(predicate combinator)
func not(pred func(int) bool) func(int) bool {
    return func(n int) bool { return !pred(n) }
}

func and(p1, p2 func(int) bool) func(int) bool {
    return func(n int) bool { return p1(n) && p2(n) }
}

func main() {
    double := multiplier(2)
    triple := multiplier(3)
    fmt.Println(double(5))  // 10
    fmt.Println(triple(5))  // 15

    // 합성: triple(double(x))
    sixTimes := compose(triple, double)
    fmt.Println(sixTimes(5)) // 30

    // 술어 합성
    isPositive := func(n int) bool { return n > 0 }
    isEven := func(n int) bool { return n%2 == 0 }
    isPositiveEven := and(isPositive, isEven)
    isOdd := not(isEven)

    fmt.Println(isPositiveEven(4))  // true
    fmt.Println(isPositiveEven(-2)) // false
    fmt.Println(isOdd(3))           // true
}
```

---

## 3. 클로저

### 3.1 변수 캡처

클로저(closure)는 감싸는 스코프의 변수를 **참조(reference)**로 캡처한다 — 변경을 감지할 수 있다.

```go
func main() {
    // 카운터 클로저
    count := 0
    increment := func() int {
        count++
        return count
    }
    fmt.Println(increment()) // 1
    fmt.Println(increment()) // 2
    fmt.Println(count)       // 2 — 외부 변수가 변경됨

    // 클로저 팩토리
    newCounter := func() func() int {
        n := 0
        return func() int {
            n++
            return n
        }
    }

    c1 := newCounter()
    c2 := newCounter()
    fmt.Println(c1(), c1(), c1()) // 1 2 3
    fmt.Println(c2(), c2())       // 1 2 — 독립적인 카운터
}
```

### 3.2 클로저의 함정

```go
func main() {
    // 버그: 루프 변수가 참조로 캡처됨
    funcs := make([]func(), 5)
    for i := 0; i < 5; i++ {
        funcs[i] = func() {
            fmt.Println(i) // 모두 5를 출력한다 (Go < 1.22에서)
        }
    }
    for _, f := range funcs {
        f()
    }

    // 수정 1: 변수를 복사한다 (Go < 1.22)
    for i := 0; i < 5; i++ {
        i := i // 새 변수로 섀도잉
        funcs[i] = func() {
            fmt.Println(i) // 올바름: 0, 1, 2, 3, 4
        }
    }

    // 수정 2: Go 1.22+ — 루프 변수가 기본적으로 반복당 하나씩이다
    // 위 코드는 Go 1.22+에서 섀도잉 없이도 올바르게 동작한다

    // 수정 3: 매개변수로 전달한다
    for i := 0; i < 5; i++ {
        funcs[i] = func(n int) func() {
            return func() { fmt.Println(n) }
        }(i)
    }
}
```

### 3.3 실용적인 클로저 패턴

```go
// 메모이제이션(memoization)
func memoize(f func(int) int) func(int) int {
    cache := make(map[int]int)
    return func(n int) int {
        if v, ok := cache[n]; ok {
            return v
        }
        result := f(n)
        cache[n] = result
        return result
    }
}

// 속도 제한기(rate limiter)
func rateLimiter(maxCalls int, period time.Duration) func() bool {
    calls := 0
    lastReset := time.Now()
    return func() bool {
        if time.Since(lastReset) > period {
            calls = 0
            lastReset = time.Now()
        }
        if calls >= maxCalls {
            return false
        }
        calls++
        return true
    }
}

// Once — 한 번만 실행한다 (단순화된 sync.Once)
func once(f func()) func() {
    done := false
    return func() {
        if !done {
            done = true
            f()
        }
    }
}

func main() {
    // 메모이제이션된 피보나치
    var fib func(int) int
    fib = memoize(func(n int) int {
        if n <= 1 {
            return n
        }
        return fib(n-1) + fib(n-2)
    })
    fmt.Println(fib(40)) // 메모이제이션 덕분에 즉시 반환

    // Once
    init := once(func() { fmt.Println("초기화됨!") })
    init() // "초기화됨!" 출력
    init() // 출력 없음
    init() // 출력 없음
}
```

---

## 4. 메서드

### 4.1 메서드 기초

메서드(method)는 특별한 **리시버(receiver)** 인자를 가진 함수이다. 타입에 동작을 붙인다.

```go
package main

import (
    "fmt"
    "math"
)

type Point struct {
    X, Y float64
}

// 값 리시버 — 복사본에서 동작한다
func (p Point) Distance(q Point) float64 {
    dx := p.X - q.X
    dy := p.Y - q.Y
    return math.Sqrt(dx*dx + dy*dy)
}

// String 메서드 — fmt.Stringer 인터페이스를 암시적으로 구현한다
func (p Point) String() string {
    return fmt.Sprintf("(%g, %g)", p.X, p.Y)
}

func main() {
    a := Point{3, 4}
    b := Point{0, 0}
    fmt.Println(a.Distance(b)) // 5
    fmt.Println(a)              // (3, 4) — fmt가 String()을 호출한다
}
```

### 4.2 모든 이름이 있는 타입에 대한 메서드

구조체뿐만 아니라 모든 이름이 있는 타입에 메서드를 정의할 수 있다.

```go
type Celsius float64
type Fahrenheit float64

func (c Celsius) ToFahrenheit() Fahrenheit {
    return Fahrenheit(c*9/5 + 32)
}

func (f Fahrenheit) ToCelsius() Celsius {
    return Celsius((f - 32) * 5 / 9)
}

type StringSlice []string

func (ss StringSlice) Contains(target string) bool {
    for _, s := range ss {
        if s == target {
            return true
        }
    }
    return false
}

func (ss StringSlice) Join(sep string) string {
    return strings.Join(ss, sep)
}

func main() {
    temp := Celsius(100)
    fmt.Printf("%.1f°C = %.1f°F\n", temp, temp.ToFahrenheit())

    colors := StringSlice{"red", "green", "blue"}
    fmt.Println(colors.Contains("green")) // true
    fmt.Println(colors.Join(", "))        // "red, green, blue"
}
```

---

## 5. 포인터 vs 값 리시버

### 5.1 포인터 리시버

메서드가 **리시버를 수정**해야 하거나 구조체가 큰 경우 포인터 리시버를 사용한다.

```go
type Account struct {
    Owner   string
    Balance float64
}

// 포인터 리시버 — 구조체를 수정할 수 있다
func (a *Account) Deposit(amount float64) {
    a.Balance += amount
}

func (a *Account) Withdraw(amount float64) error {
    if amount > a.Balance {
        return fmt.Errorf("insufficient funds: have %.2f, want %.2f", a.Balance, amount)
    }
    a.Balance -= amount
    return nil
}

// 값 리시버 — 수정할 수 없다 (복사본에서 동작한다)
func (a Account) String() string {
    return fmt.Sprintf("%s: $%.2f", a.Owner, a.Balance)
}

func main() {
    acc := Account{Owner: "Alice", Balance: 100}
    acc.Deposit(50)
    fmt.Println(acc) // Alice: $150.00

    err := acc.Withdraw(200)
    if err != nil {
        fmt.Println("에러:", err)
    }

    // Go는 값에서 포인터 메서드를 호출할 때 자동으로 주소를 취한다
    acc.Deposit(10) // (&acc).Deposit(10)과 동일하다
}
```

### 5.2 리시버 타입 선택

```go
/*
포인터 리시버(*T)를 사용하는 경우:
  1. 메서드가 리시버를 수정할 때
  2. 구조체가 클 때 (복사 방지)
  3. 일관성 — 하나의 메서드라도 포인터를 사용하면 모두 포인터를 사용한다

값 리시버(T)를 사용하는 경우:
  1. 메서드가 리시버를 수정하지 않을 때
  2. 타입이 작을 때 (int, 작은 구조체)
  3. 설계상 불변인 타입 (time.Time 같은)
*/

// 경험 법칙: 확신이 없으면 포인터 리시버를 사용한다
type LargeStruct struct {
    Data [1024]byte
    // ... 많은 필드
}

// 포인터 — 호출마다 1KB+ 복사를 방지한다
func (ls *LargeStruct) Process() {}

type SmallPoint struct {
    X, Y int
}

// 값 — 16바이트뿐이므로 복사해도 괜찮다
func (p SmallPoint) Distance() float64 {
    return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}
```

### 5.3 nil 리시버

```go
type IntList struct {
    Value int
    Next  *IntList
}

// 메서드는 nil 리시버를 우아하게 처리할 수 있다
func (l *IntList) Sum() int {
    if l == nil {
        return 0
    }
    return l.Value + l.Next.Sum()
}

func (l *IntList) String() string {
    if l == nil {
        return "nil"
    }
    return fmt.Sprintf("%d -> %s", l.Value, l.Next.String())
}

func main() {
    list := &IntList{1, &IntList{2, &IntList{3, nil}}}
    fmt.Println(list)      // 1 -> 2 -> 3 -> nil
    fmt.Println(list.Sum()) // 6

    var empty *IntList
    fmt.Println(empty.Sum()) // 0 — 패닉 없음!
}
```

---

## 6. 메서드 집합과 임베딩

### 6.1 메서드 집합

타입의 **메서드 집합(method set)**은 어떤 인터페이스를 만족하는지를 결정한다.

```go
// 값 타입 T:    값 리시버를 가진 메서드
// 포인터 타입 *T: 값 또는 포인터 리시버를 가진 메서드

type Rect struct {
    Width, Height float64
}

func (r Rect) Area() float64 {
    return r.Width * r.Height
}

func (r *Rect) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

func main() {
    r := Rect{10, 5}
    r.Area()   // OK — 값에 대한 값 리시버
    r.Scale(2) // OK — Go가 자동으로 주소를 취한다: (&r).Scale(2)

    p := &Rect{10, 5}
    p.Area()   // OK — 포인터는 값 메서드를 호출할 수 있다
    p.Scale(2) // OK — 포인터에 대한 포인터 리시버
}
```

### 6.2 임베딩을 통한 메서드 승격

```go
type Logger struct {
    Prefix string
}

func (l Logger) Log(msg string) {
    fmt.Printf("[%s] %s\n", l.Prefix, msg)
}

type Server struct {
    Logger           // 임베딩 — Logger.Log가 승격된다
    Host   string
    Port   int
}

func (s *Server) Start() {
    s.Log(fmt.Sprintf("Starting on %s:%d", s.Host, s.Port))
    // s.Log는 s.Logger.Log를 호출한다 — 승격된 메서드
}

func main() {
    s := Server{
        Logger: Logger{Prefix: "SERVER"},
        Host:   "localhost",
        Port:   8080,
    }
    s.Start()  // [SERVER] Starting on localhost:8080
    s.Log("직접 호출도 가능하다") // 승격된 메서드
}
```

### 6.3 메서드 값과 표현식

```go
func main() {
    p := Point{3, 4}

    // 메서드 값(method value) — 특정 리시버에 바인딩된다
    distFromP := p.Distance
    fmt.Println(distFromP(Point{0, 0})) // 5

    // 메서드 표현식(method expression) — 바인딩되지 않은, 리시버가 첫 번째 인자이다
    dist := Point.Distance
    fmt.Println(dist(p, Point{0, 0})) // 5

    // 고차 함수에 전달할 때 유용하다
    points := []Point{{1, 2}, {3, 4}, {5, 6}}
    origin := Point{0, 0}
    for _, pt := range points {
        fmt.Printf("%s: %.2f\n", pt, pt.Distance(origin))
    }
}
```

---

## 7. 요약

### 핵심 포인트

1. **함수는 타입이다** — `func(int) int`은 구체적인 타입이다. 명확성을 위해 함수 타입에 이름을 붙여라.
2. **고차 함수** — map/filter/reduce 같은 재사용 가능한 패턴을 위해 함수를 매개변수로 전달한다.
3. **클로저는 참조로 캡처한다** — 외부 변수의 변경이 보인다. 루프 변수 캡처 버그에 주의하라.
4. **메서드는 리시버 구문을 사용한다** — `func (t T) Method()`는 값 리시버, `func (t *T) Method()`는 포인터 리시버이다.
5. **변경을 위한 포인터 리시버** — 메서드가 리시버를 수정하거나 구조체가 크면 `*T`를 사용한다.
6. **일관성 규칙** — 타입의 어떤 메서드라도 포인터 리시버를 사용하면, 모든 메서드가 그래야 한다.
7. **임베딩은 메서드를 승격시킨다** — 임베딩된 타입의 메서드는 외부 타입에서 직접 호출할 수 있다.

---

## 연습 문제

### 연습 1: 파이프라인 빌더
문자열 변환 함수를 체이닝하는 `Pipeline` 타입을 만들라. `pipeline.Add(strings.ToUpper).Add(strings.TrimSpace).Execute("  hello  ")`가 `"HELLO"`를 반환해야 한다.

### 연습 2: 기하학
`Circle`과 `Rectangle` 구조체를 정의하라. `Area()`, `Perimeter()`, `Scale(factor)` 메서드를 추가하라. 적절한 리시버 타입을 사용하라.

### 연습 3: 함수형 도구 모음
`[]int`에 대한 제네릭 스타일의 `Map`, `Filter`, `Reduce`를 구현하라. 그런 다음 이를 사용하여 리스트에서 짝수의 제곱합을 구하라.

### 연습 4: 연결 리스트
`Append`, `Prepend`, `Delete`, `Find`, `Len`, `String` 메서드를 가진 단일 연결 리스트를 구현하라. 적절한 곳에 포인터 리시버를 사용하라.

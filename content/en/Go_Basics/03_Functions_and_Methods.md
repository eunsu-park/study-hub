# 03. Functions and Methods

**Previous**: [Composite Types](./02_Composite_Types.md) | **Next**: [Interfaces](./04_Interfaces.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use function types as parameters and return values
2. Define methods with value and pointer receivers
3. Understand when to use pointer vs value receivers
4. Write closures that capture and mutate state
5. Apply functional patterns: map, filter, reduce

---

Go treats functions as first-class citizens — they can be stored in variables, passed as arguments, and returned from other functions. Methods attach behavior to types without classes, using a simple receiver syntax that gives Go its distinctive object-oriented flavor.

## Table of Contents
1. [Function Types](#1-function-types)
2. [Higher-Order Functions](#2-higher-order-functions)
3. [Closures](#3-closures)
4. [Methods](#4-methods)
5. [Pointer vs Value Receivers](#5-pointer-vs-value-receivers)
6. [Method Sets and Embedding](#6-method-sets-and-embedding)
7. [Summary](#7-summary)

---

## 1. Function Types

### 1.1 Function Signatures as Types

Every function has a type defined by its parameter and return types.

```go
package main

import "fmt"

// Named function type
type MathFunc func(float64, float64) float64

func add(a, b float64) float64 { return a + b }
func mul(a, b float64) float64 { return a * b }

func main() {
    // Variable of function type
    var op MathFunc
    op = add
    fmt.Println(op(3, 4)) // 7

    op = mul
    fmt.Println(op(3, 4)) // 12

    // Function in a map
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

### 1.2 Type Definitions for Clarity

```go
// Predicate function type
type Predicate func(int) bool

// Transformer function type
type Transformer func(string) string

// Comparator for sorting
type Comparator func(a, b interface{}) int

// Middleware pattern (common in HTTP)
type Middleware func(http.Handler) http.Handler
```

---

## 2. Higher-Order Functions

### 2.1 Functions as Parameters

```go
package main

import (
    "fmt"
    "strings"
)

// Apply a function to each element
func mapStrings(ss []string, f func(string) string) []string {
    result := make([]string, len(ss))
    for i, s := range ss {
        result[i] = f(s)
    }
    return result
}

// Filter elements matching a predicate
func filterInts(nums []int, pred func(int) bool) []int {
    var result []int
    for _, n := range nums {
        if pred(n) {
            result = append(result, n)
        }
    }
    return result
}

// Reduce to a single value
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

### 2.2 Functions Returning Functions

```go
// Multiplier returns a function that multiplies by n
func multiplier(n int) func(int) int {
    return func(x int) int {
        return x * n
    }
}

// Compose two functions: f(g(x))
func compose(f, g func(int) int) func(int) int {
    return func(x int) int {
        return f(g(x))
    }
}

// Predicate combinators
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

    // Compose: triple(double(x))
    sixTimes := compose(triple, double)
    fmt.Println(sixTimes(5)) // 30

    // Predicate composition
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

## 3. Closures

### 3.1 Capturing Variables

Closures capture variables from their enclosing scope by **reference** — they see mutations.

```go
func main() {
    // Counter closure
    count := 0
    increment := func() int {
        count++
        return count
    }
    fmt.Println(increment()) // 1
    fmt.Println(increment()) // 2
    fmt.Println(count)       // 2 — outer variable mutated

    // Closure factory
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
    fmt.Println(c2(), c2())       // 1 2 — independent counter
}
```

### 3.2 Closure Pitfalls

```go
func main() {
    // BUG: loop variable captured by reference
    funcs := make([]func(), 5)
    for i := 0; i < 5; i++ {
        funcs[i] = func() {
            fmt.Println(i) // All print 5 (in Go < 1.22)
        }
    }
    for _, f := range funcs {
        f()
    }

    // FIX 1: Copy the variable (Go < 1.22)
    for i := 0; i < 5; i++ {
        i := i // Shadow with new variable
        funcs[i] = func() {
            fmt.Println(i) // Correct: 0, 1, 2, 3, 4
        }
    }

    // FIX 2: Go 1.22+ — loop variables are per-iteration by default
    // The above code works correctly without shadowing in Go 1.22+

    // FIX 3: Pass as parameter
    for i := 0; i < 5; i++ {
        funcs[i] = func(n int) func() {
            return func() { fmt.Println(n) }
        }(i)
    }
}
```

### 3.3 Practical Closure Patterns

```go
// Memoization
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

// Rate limiter
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

// Once — execute only once (simplified sync.Once)
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
    // Memoized fibonacci
    var fib func(int) int
    fib = memoize(func(n int) int {
        if n <= 1 {
            return n
        }
        return fib(n-1) + fib(n-2)
    })
    fmt.Println(fib(40)) // Instant, thanks to memoization

    // Once
    init := once(func() { fmt.Println("initialized!") })
    init() // prints "initialized!"
    init() // no output
    init() // no output
}
```

---

## 4. Methods

### 4.1 Method Basics

Methods are functions with a special **receiver** argument. They attach behavior to types.

```go
package main

import (
    "fmt"
    "math"
)

type Point struct {
    X, Y float64
}

// Value receiver — works on a copy
func (p Point) Distance(q Point) float64 {
    dx := p.X - q.X
    dy := p.Y - q.Y
    return math.Sqrt(dx*dx + dy*dy)
}

// String method — implements fmt.Stringer interface implicitly
func (p Point) String() string {
    return fmt.Sprintf("(%g, %g)", p.X, p.Y)
}

func main() {
    a := Point{3, 4}
    b := Point{0, 0}
    fmt.Println(a.Distance(b)) // 5
    fmt.Println(a)              // (3, 4) — String() called by fmt
}
```

### 4.2 Methods on Any Named Type

You can define methods on any named type, not just structs.

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

## 5. Pointer vs Value Receivers

### 5.1 Pointer Receivers

Use a pointer receiver when the method needs to **modify the receiver** or when the struct is large.

```go
type Account struct {
    Owner   string
    Balance float64
}

// Pointer receiver — can modify the struct
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

// Value receiver — cannot modify (works on a copy)
func (a Account) String() string {
    return fmt.Sprintf("%s: $%.2f", a.Owner, a.Balance)
}

func main() {
    acc := Account{Owner: "Alice", Balance: 100}
    acc.Deposit(50)
    fmt.Println(acc) // Alice: $150.00

    err := acc.Withdraw(200)
    if err != nil {
        fmt.Println("Error:", err)
    }

    // Go automatically takes address when calling pointer method on value
    acc.Deposit(10) // Same as (&acc).Deposit(10)
}
```

### 5.2 Choosing Receiver Type

```go
/*
Use POINTER receiver (*T) when:
  1. Method modifies the receiver
  2. Struct is large (avoids copying)
  3. Consistency — if any method uses pointer, use pointer for all

Use VALUE receiver (T) when:
  1. Method doesn't modify the receiver
  2. Type is small (int, small struct)
  3. Type is immutable by design (like time.Time)
*/

// Rule of thumb: if in doubt, use pointer receiver
type LargeStruct struct {
    Data [1024]byte
    // ... many fields
}

// Pointer — avoids copying 1KB+ on each call
func (ls *LargeStruct) Process() {}

type SmallPoint struct {
    X, Y int
}

// Value — only 16 bytes, copying is fine
func (p SmallPoint) Distance() float64 {
    return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}
```

### 5.3 Nil Receivers

```go
type IntList struct {
    Value int
    Next  *IntList
}

// Methods can handle nil receivers gracefully
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
    fmt.Println(empty.Sum()) // 0 — no panic!
}
```

---

## 6. Method Sets and Embedding

### 6.1 Method Sets

The **method set** of a type determines which interfaces it satisfies.

```go
// Value type T:    methods with value receiver
// Pointer type *T: methods with value OR pointer receiver

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
    r.Area()   // OK — value receiver on value
    r.Scale(2) // OK — Go auto-takes address: (&r).Scale(2)

    p := &Rect{10, 5}
    p.Area()   // OK — pointer can call value methods
    p.Scale(2) // OK — pointer receiver on pointer
}
```

### 6.2 Promoted Methods via Embedding

```go
type Logger struct {
    Prefix string
}

func (l Logger) Log(msg string) {
    fmt.Printf("[%s] %s\n", l.Prefix, msg)
}

type Server struct {
    Logger           // Embedded — Logger.Log is promoted
    Host   string
    Port   int
}

func (s *Server) Start() {
    s.Log(fmt.Sprintf("Starting on %s:%d", s.Host, s.Port))
    // s.Log calls s.Logger.Log — promoted method
}

func main() {
    s := Server{
        Logger: Logger{Prefix: "SERVER"},
        Host:   "localhost",
        Port:   8080,
    }
    s.Start()  // [SERVER] Starting on localhost:8080
    s.Log("Direct call also works") // Promoted method
}
```

### 6.3 Method Value and Expression

```go
func main() {
    p := Point{3, 4}

    // Method value — bound to specific receiver
    distFromP := p.Distance
    fmt.Println(distFromP(Point{0, 0})) // 5

    // Method expression — unbound, receiver is first argument
    dist := Point.Distance
    fmt.Println(dist(p, Point{0, 0})) // 5

    // Useful for passing to higher-order functions
    points := []Point{{1, 2}, {3, 4}, {5, 6}}
    origin := Point{0, 0}
    for _, pt := range points {
        fmt.Printf("%s: %.2f\n", pt, pt.Distance(origin))
    }
}
```

---

## 7. Summary

### Key Takeaways

1. **Functions are types** — `func(int) int` is a concrete type. Name function types for clarity.
2. **Higher-order functions** — pass functions as parameters for reusable patterns like map/filter/reduce.
3. **Closures capture by reference** — mutations to outer variables are visible. Watch for loop-variable capture bugs.
4. **Methods use receiver syntax** — `func (t T) Method()` for value, `func (t *T) Method()` for pointer.
5. **Pointer receivers for mutation** — if a method modifies the receiver or the struct is large, use `*T`.
6. **Consistency rule** — if any method on a type uses a pointer receiver, all methods should.
7. **Embedding promotes methods** — embedded types' methods are callable directly on the outer type.

---

## Exercises

### Exercise 1: Pipeline Builder
Create a `Pipeline` type that chains string transformation functions. `pipeline.Add(strings.ToUpper).Add(strings.TrimSpace).Execute("  hello  ")` should return `"HELLO"`.

### Exercise 2: Geometry
Define `Circle` and `Rectangle` structs. Add `Area()`, `Perimeter()`, and `Scale(factor)` methods. Use appropriate receiver types.

### Exercise 3: Functional Toolkit
Implement generic-style `Map`, `Filter`, and `Reduce` for `[]int`. Then use them to: find the sum of squares of even numbers in a list.

### Exercise 4: Linked List
Implement a singly linked list with methods: `Append`, `Prepend`, `Delete`, `Find`, `Len`, and `String`. Use pointer receivers where appropriate.

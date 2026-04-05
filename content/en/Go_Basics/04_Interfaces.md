# 04. Interfaces

**Previous**: [Functions and Methods](./03_Functions_and_Methods.md) | **Next**: [Error Handling](./05_Error_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define and implement interfaces using Go's implicit satisfaction model
2. Use the empty interface and type assertions safely
3. Apply common interface patterns: Stringer, Reader, Writer
4. Design small, composable interfaces following Go conventions
5. Use type switches for polymorphic behavior

---

Interfaces are Go's central mechanism for abstraction. Unlike Java or C# where classes explicitly declare which interfaces they implement, Go uses **implicit satisfaction** — if a type has the right methods, it implements the interface automatically. This decoupling is what makes Go interfaces so powerful and flexible.

## Table of Contents
1. [Interface Basics](#1-interface-basics)
2. [Implicit Satisfaction](#2-implicit-satisfaction)
3. [Common Standard Interfaces](#3-common-standard-interfaces)
4. [Empty Interface and Type Assertions](#4-empty-interface-and-type-assertions)
5. [Interface Composition](#5-interface-composition)
6. [Interface Design Principles](#6-interface-design-principles)
7. [Summary](#7-summary)

---

## 1. Interface Basics

### 1.1 Defining Interfaces

An interface defines a set of method signatures. Any type that implements all methods satisfies the interface.

```go
package main

import (
    "fmt"
    "math"
)

// Interface definition
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Circle implements Shape
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

// Rectangle implements Shape
type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// Function accepting interface
func printShape(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    c := Circle{Radius: 5}
    r := Rectangle{Width: 10, Height: 3}

    printShape(c) // Area: 78.54, Perimeter: 31.42
    printShape(r) // Area: 30.00, Perimeter: 26.00

    // Slice of interface type — polymorphism
    shapes := []Shape{c, r, Circle{Radius: 1}}
    totalArea := 0.0
    for _, s := range shapes {
        totalArea += s.Area()
    }
    fmt.Printf("Total area: %.2f\n", totalArea)
}
```

### 1.2 Interface Values

An interface value consists of two components: a **type** and a **value**.

```go
func main() {
    var s Shape
    fmt.Println(s)         // <nil>
    fmt.Println(s == nil)  // true

    s = Circle{Radius: 5}
    fmt.Printf("Type: %T, Value: %v\n", s, s)
    // Type: main.Circle, Value: {5}

    s = Rectangle{Width: 3, Height: 4}
    fmt.Printf("Type: %T, Value: %v\n", s, s)
    // Type: main.Rectangle, Value: {3 4}

    // GOTCHA: nil pointer in interface is NOT nil interface
    var c *Circle // nil pointer
    s = c
    fmt.Println(s == nil) // false! Interface holds (*Circle, nil)
}
```

---

## 2. Implicit Satisfaction

### 2.1 No "implements" Keyword

```go
// This interface exists in the fmt package
// type Stringer interface {
//     String() string
// }

type Temperature struct {
    Celsius float64
}

// Temperature implements fmt.Stringer implicitly
func (t Temperature) String() string {
    return fmt.Sprintf("%.1f°C", t.Celsius)
}

func main() {
    t := Temperature{Celsius: 36.6}
    fmt.Println(t) // "36.6°C" — fmt.Println calls String()

    // No declaration needed — Temperature satisfies Stringer
    // because it has a String() string method
}
```

### 2.2 Compile-Time Verification

```go
// Verify at compile time that a type satisfies an interface
var _ Shape = Circle{}     // Compile error if Circle doesn't implement Shape
var _ Shape = (*Circle)(nil) // Check with pointer receiver

// This is a common Go idiom used in libraries
var _ io.Reader = (*MyReader)(nil)
var _ io.Writer = (*MyWriter)(nil)
```

### 2.3 Decoupling Producer and Consumer

```go
// producer.go — defines concrete type
type FileStore struct {
    BasePath string
}

func (fs FileStore) Save(key string, data []byte) error {
    return os.WriteFile(filepath.Join(fs.BasePath, key), data, 0644)
}

func (fs FileStore) Load(key string) ([]byte, error) {
    return os.ReadFile(filepath.Join(fs.BasePath, key))
}

// consumer.go — defines the interface it needs
type Store interface {
    Save(key string, data []byte) error
    Load(key string) ([]byte, error)
}

// Consumer doesn't know about FileStore, MemoryStore, S3Store, etc.
type App struct {
    store Store
}

func (a *App) SaveConfig(config []byte) error {
    return a.store.Save("config.json", config)
}
```

---

## 3. Common Standard Interfaces

### 3.1 fmt.Stringer

```go
type Point struct {
    X, Y float64
}

func (p Point) String() string {
    return fmt.Sprintf("(%g, %g)", p.X, p.Y)
}

// Now fmt.Println(p) uses this method
```

### 3.2 io.Reader and io.Writer

The most important interfaces in Go's standard library.

```go
import (
    "bytes"
    "io"
    "os"
    "strings"
)

// io.Reader: Read(p []byte) (n int, err error)
// io.Writer: Write(p []byte) (n int, err error)

func main() {
    // strings.Reader implements io.Reader
    r := strings.NewReader("Hello, World!")

    // Copy from reader to writer (stdout)
    io.Copy(os.Stdout, r)
    fmt.Println()

    // bytes.Buffer implements both Reader and Writer
    var buf bytes.Buffer
    buf.WriteString("Hello ")
    buf.WriteString("Buffer!")
    fmt.Println(buf.String())

    // Any function accepting io.Reader works with files, network, strings, etc.
    data, _ := io.ReadAll(strings.NewReader("read all of this"))
    fmt.Println(string(data))
}

// Write a function that works with any io.Reader
func countLines(r io.Reader) (int, error) {
    scanner := bufio.NewScanner(r)
    count := 0
    for scanner.Scan() {
        count++
    }
    return count, scanner.Err()
}

// Works with files, strings, network connections, etc.
// lines, _ := countLines(os.Stdin)
// lines, _ := countLines(strings.NewReader("a\nb\nc"))
// lines, _ := countLines(file)
```

### 3.3 sort.Interface

```go
import "sort"

type Person struct {
    Name string
    Age  int
}

// ByAge implements sort.Interface
type ByAge []Person

func (a ByAge) Len() int           { return len(a) }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func main() {
    people := []Person{
        {"Alice", 30},
        {"Bob", 25},
        {"Carol", 28},
    }

    sort.Sort(ByAge(people))
    fmt.Println(people) // Sorted by age: Bob, Carol, Alice

    // Modern alternative: sort.Slice (Go 1.8+)
    sort.Slice(people, func(i, j int) bool {
        return people[i].Name < people[j].Name
    })
    fmt.Println(people) // Sorted by name: Alice, Bob, Carol
}
```

### 3.4 error Interface

```go
// The error interface is simple:
// type error interface {
//     Error() string
// }

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed on %s: %s", e.Field, e.Message)
}

func validateAge(age int) error {
    if age < 0 || age > 150 {
        return &ValidationError{
            Field:   "age",
            Message: fmt.Sprintf("must be 0-150, got %d", age),
        }
    }
    return nil
}
```

---

## 4. Empty Interface and Type Assertions

### 4.1 Empty Interface (any)

```go
// any is an alias for interface{} (Go 1.18+)
func printAnything(val any) {
    fmt.Printf("Type: %T, Value: %v\n", val, val)
}

func main() {
    printAnything(42)
    printAnything("hello")
    printAnything([]int{1, 2, 3})
    printAnything(nil)

    // Slice of any — like Java's Object[]
    mixed := []any{1, "two", 3.0, true}
    for _, v := range mixed {
        fmt.Println(v)
    }
}
```

### 4.2 Type Assertions

```go
func main() {
    var val any = "hello"

    // Type assertion — extracts concrete type
    s := val.(string)
    fmt.Println(s) // "hello"

    // PANIC if wrong type:
    // n := val.(int) // panic: interface conversion

    // Safe assertion with "comma ok"
    s, ok := val.(string)
    if ok {
        fmt.Println("It's a string:", s)
    }

    n, ok := val.(int)
    if !ok {
        fmt.Println("Not an int") // This prints
    }
    fmt.Println(n) // 0 (zero value)
}
```

### 4.3 Type Switches

```go
func describe(val any) string {
    switch v := val.(type) {
    case nil:
        return "nil"
    case int:
        return fmt.Sprintf("integer: %d", v)
    case float64:
        return fmt.Sprintf("float: %.2f", v)
    case string:
        return fmt.Sprintf("string: %q (len=%d)", v, len(v))
    case bool:
        return fmt.Sprintf("bool: %t", v)
    case []int:
        return fmt.Sprintf("int slice: %v (len=%d)", v, len(v))
    case Shape:
        return fmt.Sprintf("shape with area %.2f", v.Area())
    default:
        return fmt.Sprintf("unknown: %T", v)
    }
}

func main() {
    values := []any{42, 3.14, "hello", true, nil, []int{1, 2}}
    for _, v := range values {
        fmt.Println(describe(v))
    }
}
```

---

## 5. Interface Composition

### 5.1 Embedding Interfaces

```go
// Small, focused interfaces
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

// Composed interfaces
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// Real-world example from standard library:
// type ReadWriteCloser interface {
//     io.Reader
//     io.Writer
//     io.Closer
// }
```

### 5.2 Accept Interfaces, Return Structs

```go
// GOOD: Accept interface — flexible for callers
func Process(r io.Reader) error {
    data, err := io.ReadAll(r)
    if err != nil {
        return err
    }
    fmt.Println(string(data))
    return nil
}

// GOOD: Return concrete type — callers get full functionality
func NewBuffer() *bytes.Buffer {
    return &bytes.Buffer{}
}

// The caller can use it as io.Writer, io.Reader, or *bytes.Buffer
```

### 5.3 Interface Segregation

```go
// BAD: Large interface — hard to implement, test, and mock
type Repository interface {
    Create(item Item) error
    Read(id string) (Item, error)
    Update(id string, item Item) error
    Delete(id string) error
    List() ([]Item, error)
    Search(query string) ([]Item, error)
    Count() (int, error)
    Export(w io.Writer) error
}

// GOOD: Small, focused interfaces
type ItemReader interface {
    Read(id string) (Item, error)
}

type ItemWriter interface {
    Create(item Item) error
    Update(id string, item Item) error
    Delete(id string) error
}

type ItemLister interface {
    List() ([]Item, error)
    Search(query string) ([]Item, error)
}

// Compose when you need more
type ItemStore interface {
    ItemReader
    ItemWriter
}
```

---

## 6. Interface Design Principles

### 6.1 Go Proverbs for Interfaces

```go
// "The bigger the interface, the weaker the abstraction." — Rob Pike

// 1. Keep interfaces small (1-3 methods)
type Sizer interface {
    Size() int64
}

// 2. Define interfaces where they are USED, not where they are implemented
// consumer defines what it needs:
type UserService struct {
    repo UserRepository
}

type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(u *User) error
}
// Different packages can implement this without importing UserService

// 3. Use standard interfaces when possible (io.Reader, fmt.Stringer, etc.)

// 4. Don't export interfaces prematurely
// Start with concrete types; extract interface when you need polymorphism
```

### 6.2 Interface Testing Pattern

```go
// Interface enables easy testing with mocks
type EmailSender interface {
    Send(to, subject, body string) error
}

// Production implementation
type SMTPSender struct {
    Host string
    Port int
}

func (s *SMTPSender) Send(to, subject, body string) error {
    // Actually send email via SMTP
    return nil
}

// Test mock
type MockSender struct {
    SentEmails []struct{ To, Subject, Body string }
}

func (m *MockSender) Send(to, subject, body string) error {
    m.SentEmails = append(m.SentEmails, struct{ To, Subject, Body string }{to, subject, body})
    return nil
}

// Usage in tests:
// sender := &MockSender{}
// service := NewNotificationService(sender)
// service.NotifyUser("alice@example.com", "Hello")
// assert(len(sender.SentEmails) == 1)
```

### 6.3 The Stringer Contract

```go
// Implementing String() makes your types print-friendly everywhere
type Duration struct {
    Hours   int
    Minutes int
    Seconds int
}

func (d Duration) String() string {
    return fmt.Sprintf("%02d:%02d:%02d", d.Hours, d.Minutes, d.Seconds)
}

func main() {
    d := Duration{1, 30, 45}
    fmt.Println(d)                // 01:30:45
    fmt.Sprintf("Duration: %s", d) // "Duration: 01:30:45"
    s := "Meeting: " + d.String()  // "Meeting: 01:30:45"
}
```

---

## 7. Summary

### Key Takeaways

1. **Implicit satisfaction** — no `implements` keyword. If a type has the methods, it satisfies the interface.
2. **Interfaces are satisfied by method sets** — value receivers on value types, all receivers on pointer types.
3. **Small interfaces are powerful** — `io.Reader` (1 method) is the most widely used interface in Go.
4. **Accept interfaces, return structs** — functions should take interfaces for flexibility and return concrete types for usability.
5. **Type assertions and switches** — safely extract concrete types from interfaces when needed.
6. **Composition over inheritance** — embed small interfaces to build larger ones.
7. **Define interfaces at the consumer** — the package that uses the interface should define it, not the implementor.

### Interface Cheat Sheet

| Pattern | When to Use |
|---------|-------------|
| `any` / `interface{}` | Truly generic containers (prefer generics in Go 1.18+) |
| Type assertion `v.(T)` | When you know the concrete type |
| Type switch `v.(type)` | When handling multiple possible types |
| Small interface (1-2 methods) | Abstractions like Reader, Writer, Stringer |
| Interface embedding | Building larger interfaces from smaller ones |

---

## Exercises

### Exercise 1: Shape Calculator
Define a `Shape` interface with `Area()` and `Perimeter()`. Implement `Circle`, `Rectangle`, and `Triangle`. Write a `LargestShape(shapes []Shape) Shape` function.

### Exercise 2: Custom Sorter
Implement `sort.Interface` for a `[]Employee` type that can sort by name, salary, or hire date.

### Exercise 3: Plugin System
Design a simple plugin system using interfaces. Define a `Plugin` interface with `Name() string`, `Init() error`, and `Execute(data any) (any, error)`. Create two concrete plugins.

### Exercise 4: Mock Testing
Write a `NotificationService` that depends on a `Notifier` interface. Create a real `EmailNotifier` and a `MockNotifier` for testing. Write a test that verifies notifications are sent.

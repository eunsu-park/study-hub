# 01. Go Fundamentals

**Previous**: [Overview](./00_Overview.md) | **Next**: [Composite Types](./02_Composite_Types.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Install Go and configure the development environment
2. Declare variables using `var`, `:=`, and understand type inference
3. Use all basic types: integers, floats, strings, booleans, runes
4. Write control flow with `if`, `for`, `switch`, and `defer`
5. Define and call functions with multiple return values

---

Go is a language that rewards simplicity. Where other languages offer five ways to do something, Go typically offers one — and that one way is clear, fast, and predictable. This lesson covers the building blocks: types, variables, control flow, and functions.

## Table of Contents
1. [Hello, Go!](#1-hello-go)
2. [Variables and Declarations](#2-variables-and-declarations)
3. [Basic Types](#3-basic-types)
4. [Type Conversions and Constants](#4-type-conversions-and-constants)
5. [Control Flow](#5-control-flow)
6. [Functions](#6-functions)
7. [Summary](#7-summary)

---

## 1. Hello, Go!

### 1.1 Your First Program

Every Go program begins with a package declaration and an import block. The `main` package with a `main()` function is the entry point for executables.

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, Go!")
}
```

```bash
# Run directly (compiles to temp dir and executes)
go run main.go

# Build a binary
go build -o hello main.go
./hello

# Format code
go fmt main.go
```

### 1.2 Program Structure

```go
package main          // Package declaration — every file belongs to exactly one package

import (              // Import block — grouped with parentheses
    "fmt"             // Standard library: formatted I/O
    "math"            // Standard library: math functions
    "strings"         // Standard library: string manipulation
)

// main is the entry point — no arguments, no return value
func main() {
    fmt.Println("Pi is approximately", math.Pi)
    fmt.Println(strings.ToUpper("hello"))
}
```

Key observations:
- **No semicolons** — the lexer inserts them automatically
- **Unused imports are compile errors** — Go enforces cleanliness
- **Exported names start with uppercase** — `fmt.Println` is exported; `fmt.println` would not be
- **`gofmt`** enforces a single formatting style — no style debates

### 1.3 Go Workspace

```bash
# Modern Go uses modules (Go 1.11+)
mkdir myproject && cd myproject
go mod init github.com/username/myproject

# This creates go.mod — the module definition file
cat go.mod
# module github.com/username/myproject
# go 1.22
```

---

## 2. Variables and Declarations

### 2.1 var Declaration

The `var` keyword declares variables with explicit type or inferred from initial value.

```go
package main

import "fmt"

func main() {
    // Explicit type
    var name string = "Alice"
    var age int = 30
    var height float64 = 5.9

    // Type inference — compiler deduces the type
    var city = "Seoul"           // string
    var population = 9_700_000   // int (underscores for readability)

    // Zero values — every type has a default
    var count int       // 0
    var rate float64    // 0.0
    var label string    // "" (empty string)
    var active bool     // false

    fmt.Println(name, age, height)
    fmt.Println(city, population)
    fmt.Println(count, rate, label, active)

    // Block declaration
    var (
        x int    = 10
        y int    = 20
        z string = "result"
    )
    fmt.Println(x, y, z)
}
```

### 2.2 Short Variable Declaration

Inside functions, the `:=` operator declares and initializes in one step. This is the most common form.

```go
func main() {
    // Short declaration — type is inferred
    name := "Bob"          // string
    age := 25              // int
    pi := 3.14159          // float64
    active := true         // bool

    fmt.Println(name, age, pi, active)

    // Multiple assignment
    x, y := 10, 20
    fmt.Println(x, y)

    // Swap values — no temp variable needed
    x, y = y, x
    fmt.Println(x, y) // 20, 10

    // := requires at least one NEW variable on the left
    x, z := 100, "hello"  // OK: z is new
    fmt.Println(x, z)
}
```

### 2.3 Naming Conventions

```go
// camelCase for local variables and unexported identifiers
userName := "alice"
maxRetries := 3

// PascalCase for exported identifiers (visible outside the package)
func ProcessOrder() {}
type HttpClient struct {}

// Acronyms stay uppercase
var httpURL string
var xmlParser *Parser
var userID int
```

---

## 3. Basic Types

### 3.1 Integer Types

```go
package main

import (
    "fmt"
    "math"
)

func main() {
    // Signed integers
    var i8 int8 = 127                    // -128 to 127
    var i16 int16 = 32767                // -32768 to 32767
    var i32 int32 = math.MaxInt32        // -2^31 to 2^31-1
    var i64 int64 = math.MaxInt64        // -2^63 to 2^63-1

    // Platform-dependent: 32-bit on 32-bit systems, 64-bit on 64-bit
    var i int = 42

    // Unsigned integers
    var u8 uint8 = 255                   // 0 to 255 (alias: byte)
    var u16 uint16 = 65535
    var u32 uint32 = math.MaxUint32
    var u64 uint64 = math.MaxUint64

    // byte and rune are aliases
    var b byte = 'A'        // alias for uint8
    var r rune = '가'        // alias for int32 (Unicode code point)

    fmt.Println(i8, i16, i32, i64, i)
    fmt.Println(u8, u16, u32, u64)
    fmt.Println(b, r)

    // Number literals
    decimal := 42
    hex := 0xFF
    octal := 0o77
    binary := 0b1010_1100
    withSep := 1_000_000

    fmt.Printf("dec=%d hex=%d oct=%d bin=%d sep=%d\n",
        decimal, hex, octal, binary, withSep)
}
```

### 3.2 Floating-Point and Complex

```go
func main() {
    var f32 float32 = 3.14       // ~7 decimal digits precision
    var f64 float64 = 3.14159265 // ~15 decimal digits precision

    // Default float literal type is float64
    pi := 3.14159265358979

    // Scientific notation
    avogadro := 6.022e23
    planck := 6.626e-34

    fmt.Println(f32, f64, pi)
    fmt.Println(avogadro, planck)

    // Complex numbers (built-in!)
    c1 := complex(3, 4)           // 3+4i
    c2 := 2 + 5i                  // literal syntax
    fmt.Println(c1 + c2)          // (5+9i)
    fmt.Println(real(c1), imag(c1)) // 3, 4
}
```

### 3.3 Strings and Runes

```go
package main

import (
    "fmt"
    "strings"
    "unicode/utf8"
)

func main() {
    // Strings are immutable byte sequences
    greeting := "Hello, 世界"
    fmt.Println(len(greeting))                    // 13 bytes (not characters!)
    fmt.Println(utf8.RuneCountInString(greeting)) // 9 runes (characters)

    // Raw strings — no escape processing
    path := `C:\Users\alice\documents`
    multiline := `
        This is a
        multi-line string
    `
    fmt.Println(path)
    fmt.Println(multiline)

    // String operations
    fmt.Println(strings.ToUpper("hello"))          // "HELLO"
    fmt.Println(strings.Contains("hello", "ell"))  // true
    fmt.Println(strings.Replace("aaa", "a", "b", 2)) // "bba"
    fmt.Println(strings.Split("a,b,c", ","))       // [a b c]
    fmt.Println(strings.Join([]string{"a", "b"}, "-")) // "a-b"

    // Iterating over runes (characters)
    for i, r := range "Go 한국어" {
        fmt.Printf("byte %d: rune %c (U+%04X)\n", i, r, r)
    }

    // String builder for efficient concatenation
    var builder strings.Builder
    for i := 0; i < 5; i++ {
        fmt.Fprintf(&builder, "item %d ", i)
    }
    fmt.Println(builder.String())
}
```

### 3.4 Booleans

```go
func main() {
    a := true
    b := false

    fmt.Println(a && b)  // false (AND)
    fmt.Println(a || b)  // true (OR)
    fmt.Println(!a)      // false (NOT)

    // Comparison operators return bools
    x, y := 10, 20
    fmt.Println(x == y)  // false
    fmt.Println(x < y)   // true
    fmt.Println(x != y)  // true
}
```

---

## 4. Type Conversions and Constants

### 4.1 Explicit Type Conversions

Go has no implicit type conversions. Every conversion must be explicit.

```go
func main() {
    // int → float64
    x := 42
    y := float64(x)

    // float64 → int (truncates)
    pi := 3.99
    n := int(pi) // 3, not 4

    // int → string (Unicode code point, NOT digit conversion!)
    r := rune(65)
    fmt.Println(string(r)) // "A"

    // Number to string — use fmt or strconv
    import "strconv"
    s := strconv.Itoa(42)        // "42"
    f := strconv.FormatFloat(3.14, 'f', 2, 64) // "3.14"

    // String to number
    n, err := strconv.Atoi("42")
    if err != nil {
        fmt.Println("Parse error:", err)
    }
    fmt.Println(n) // 42

    fmt.Println(y, n, s, f)
}
```

### 4.2 Constants

```go
package main

import "fmt"

// Typed constants
const Pi float64 = 3.14159265358979
const MaxRetries int = 3

// Untyped constants — flexible, adapt to context
const (
    Hello  = "Hello"
    Answer = 42        // Can be used as int, float64, etc.
    E      = 2.71828
)

// iota — auto-incrementing constant generator
type Weekday int

const (
    Sunday    Weekday = iota // 0
    Monday                   // 1
    Tuesday                  // 2
    Wednesday                // 3
    Thursday                 // 4
    Friday                   // 5
    Saturday                 // 6
)

// Bit flags with iota
type Permission uint8

const (
    Read    Permission = 1 << iota // 1
    Write                          // 2
    Execute                        // 4
)

func main() {
    fmt.Println(Pi, MaxRetries)
    fmt.Println(Monday, Friday)
    fmt.Printf("Read=%d Write=%d Execute=%d\n", Read, Write, Execute)

    // Combine permissions with OR
    perm := Read | Write
    fmt.Printf("perm=%d hasRead=%t hasExec=%t\n",
        perm, perm&Read != 0, perm&Execute != 0)
}
```

---

## 5. Control Flow

### 5.1 if/else

```go
func main() {
    x := 42

    // Standard if/else
    if x > 0 {
        fmt.Println("positive")
    } else if x < 0 {
        fmt.Println("negative")
    } else {
        fmt.Println("zero")
    }

    // if with init statement — variable scoped to if/else block
    if err := doSomething(); err != nil {
        fmt.Println("error:", err)
    }
    // err is not accessible here

    // Common pattern: error checking
    if data, err := fetchData(); err != nil {
        fmt.Println("failed:", err)
    } else {
        fmt.Println("got:", data)
    }
}
```

### 5.2 for Loops

Go has only `for` — no `while` or `do-while`. It covers all cases.

```go
func main() {
    // Classic three-component for
    for i := 0; i < 5; i++ {
        fmt.Println(i)
    }

    // While-style (condition only)
    n := 1
    for n < 100 {
        n *= 2
    }
    fmt.Println(n) // 128

    // Infinite loop
    count := 0
    for {
        count++
        if count > 3 {
            break
        }
    }

    // Range over slice
    fruits := []string{"apple", "banana", "cherry"}
    for index, value := range fruits {
        fmt.Printf("%d: %s\n", index, value)
    }

    // Range — ignore index
    for _, fruit := range fruits {
        fmt.Println(fruit)
    }

    // Range over string (iterates runes, not bytes)
    for i, r := range "Go언어" {
        fmt.Printf("byte %d: %c\n", i, r)
    }

    // Range over map
    ages := map[string]int{"Alice": 30, "Bob": 25}
    for name, age := range ages {
        fmt.Printf("%s is %d\n", name, age)
    }

    // continue and labeled loops
    outer:
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if i == j {
                continue outer
            }
            fmt.Println(i, j)
        }
    }
}
```

### 5.3 switch

```go
func main() {
    // Expression switch
    day := "Monday"
    switch day {
    case "Monday":
        fmt.Println("Start of work week")
    case "Friday":
        fmt.Println("TGIF!")
    case "Saturday", "Sunday":
        fmt.Println("Weekend!")
    default:
        fmt.Println("Midweek")
    }

    // Switch with no condition (cleaner than if/else chains)
    score := 85
    switch {
    case score >= 90:
        fmt.Println("A")
    case score >= 80:
        fmt.Println("B")
    case score >= 70:
        fmt.Println("C")
    default:
        fmt.Println("F")
    }

    // Switch with init statement
    switch os := runtime.GOOS; os {
    case "linux":
        fmt.Println("Linux")
    case "darwin":
        fmt.Println("macOS")
    default:
        fmt.Println(os)
    }

    // Type switch
    var val interface{} = 42
    switch v := val.(type) {
    case int:
        fmt.Printf("int: %d\n", v)
    case string:
        fmt.Printf("string: %s\n", v)
    case bool:
        fmt.Printf("bool: %t\n", v)
    default:
        fmt.Printf("unknown: %T\n", v)
    }

    // fallthrough — explicit, unlike C
    switch 3 {
    case 3:
        fmt.Println("three")
        fallthrough
    case 4:
        fmt.Println("four (via fallthrough)")
    case 5:
        fmt.Println("five (not reached)")
    }
}
```

### 5.4 defer

`defer` schedules a function call to run when the surrounding function returns. Deferred calls execute in LIFO (last-in, first-out) order.

```go
package main

import (
    "fmt"
    "os"
)

func main() {
    // defer runs when main() returns
    fmt.Println("start")
    defer fmt.Println("deferred 1")
    defer fmt.Println("deferred 2")
    fmt.Println("end")
    // Output: start, end, deferred 2, deferred 1

    // Common pattern: cleanup
    f, err := os.Create("test.txt")
    if err != nil {
        fmt.Println(err)
        return
    }
    defer f.Close() // Guaranteed to close, even on error

    f.WriteString("hello\n")

    // defer captures values at defer time
    x := 10
    defer fmt.Println("deferred x =", x) // prints 10, not 20
    x = 20
    fmt.Println("current x =", x)
}

// defer is critical for resource cleanup
func readFile(path string) (string, error) {
    f, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()

    // Even if an error occurs here, f.Close() still runs
    data, err := io.ReadAll(f)
    if err != nil {
        return "", err
    }
    return string(data), nil
}
```

---

## 6. Functions

### 6.1 Basic Functions

```go
package main

import (
    "fmt"
    "math"
)

// Simple function
func greet(name string) string {
    return "Hello, " + name + "!"
}

// Multiple parameters of the same type
func add(a, b int) int {
    return a + b
}

// Multiple return values — idiomatic Go
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Named return values
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return // "naked return" — returns named values
}

// Variadic function
func sum(nums ...int) int {
    total := 0
    for _, n := range nums {
        total += n
    }
    return total
}

func main() {
    fmt.Println(greet("Go"))          // "Hello, Go!"
    fmt.Println(add(3, 4))            // 7

    result, err := divide(10, 3)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Printf("10 / 3 = %.2f\n", result)
    }

    x, y := split(17)
    fmt.Println(x, y)

    fmt.Println(sum(1, 2, 3, 4, 5))  // 15

    // Spread a slice into variadic
    numbers := []int{10, 20, 30}
    fmt.Println(sum(numbers...))      // 60
}
```

### 6.2 Functions as Values

```go
func main() {
    // Function variable
    operation := add
    fmt.Println(operation(3, 4)) // 7

    // Anonymous function
    double := func(x int) int {
        return x * 2
    }
    fmt.Println(double(5)) // 10

    // Immediately invoked
    result := func(a, b int) int {
        return a * b
    }(3, 4)
    fmt.Println(result) // 12

    // Higher-order function
    apply := func(f func(int) int, val int) int {
        return f(val)
    }
    fmt.Println(apply(double, 10)) // 20

    // Closures — capture outer variables
    counter := makeCounter()
    fmt.Println(counter()) // 1
    fmt.Println(counter()) // 2
    fmt.Println(counter()) // 3
}

func makeCounter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}
```

### 6.3 Formatting Output

```go
func main() {
    name := "Alice"
    age := 30
    height := 5.75

    // Printf verbs
    fmt.Printf("Name: %s\n", name)         // string
    fmt.Printf("Age: %d\n", age)           // integer
    fmt.Printf("Height: %.1f ft\n", height) // float
    fmt.Printf("Hex: %x\n", 255)           // hex
    fmt.Printf("Binary: %b\n", 42)         // binary
    fmt.Printf("Type: %T\n", name)         // type name
    fmt.Printf("Value: %v\n", name)        // default format
    fmt.Printf("Quoted: %q\n", name)       // quoted string
    fmt.Printf("Pointer: %p\n", &name)     // pointer address

    // Sprintf returns a string (doesn't print)
    msg := fmt.Sprintf("%s is %d years old", name, age)
    fmt.Println(msg)

    // Fprintln writes to any io.Writer
    fmt.Fprintln(os.Stderr, "This goes to stderr")
}
```

---

## 7. Summary

### Key Takeaways

1. **Go is opinionated** — one formatting style, no unused imports, explicit type conversions. This eliminates bike-shedding and keeps codebases consistent.

2. **Zero values are useful** — every type has a meaningful default (`0`, `""`, `false`, `nil`). This eliminates the need for constructors in many cases.

3. **`:=` is idiomatic** — use short declaration inside functions. Reserve `var` for package-level declarations and zero-value initialization.

4. **`for` is the only loop** — it handles classic iteration, while-loops, infinite loops, and range-based iteration over collections.

5. **Multiple return values** — Go uses `(value, error)` pairs instead of exceptions. Always check errors.

6. **`defer` for cleanup** — guarantees resource cleanup regardless of how a function exits.

7. **Functions are first-class** — they can be assigned to variables, passed as arguments, and returned from other functions.

### Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `=` instead of `:=` for new variables | `:=` declares + assigns; `=` only assigns |
| Ignoring errors from functions | Always handle the `error` return value |
| String concatenation in loops | Use `strings.Builder` instead |
| `len(s)` on UTF-8 strings for character count | Use `utf8.RuneCountInString(s)` |

---

## Exercises

### Exercise 1: Temperature Converter
Write a function `celsiusToFahrenheit(c float64) float64` and its inverse. Print a conversion table from 0°C to 100°C in steps of 10.

### Exercise 2: FizzBuzz
Write a FizzBuzz program for numbers 1-100. Print "Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both, and the number otherwise.

### Exercise 3: String Analysis
Write a function that takes a string and returns: (a) the number of words, (b) the number of characters (runes), (c) the reversed string. Handle Unicode correctly.

### Exercise 4: Simple Calculator
Write a calculator function `calc(a float64, op string, b float64) (float64, error)` that supports `+`, `-`, `*`, `/`. Return an error for division by zero and unknown operators.

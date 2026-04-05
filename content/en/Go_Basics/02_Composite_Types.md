# 02. Composite Types

**Previous**: [Go Fundamentals](./01_Go_Fundamentals.md) | **Next**: [Functions and Methods](./03_Functions_and_Methods.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and manipulate arrays with fixed sizes
2. Use slices for dynamic collections, understanding capacity and length
3. Create and operate on maps for key-value storage
4. Define and instantiate structs with named fields
5. Understand the difference between value and reference semantics for each type

---

Go's composite types build on the basic types from Lesson 01 to create powerful data structures. Arrays provide fixed-size sequences, slices add dynamic sizing with efficient memory management, maps offer hash-table performance for key-value lookups, and structs let you define your own aggregate types.

## Table of Contents
1. [Arrays](#1-arrays)
2. [Slices](#2-slices)
3. [Slice Internals](#3-slice-internals)
4. [Maps](#4-maps)
5. [Structs](#5-structs)
6. [Struct Embedding and Tags](#6-struct-embedding-and-tags)
7. [Summary](#7-summary)

---

## 1. Arrays

### 1.1 Array Basics

Arrays in Go have a **fixed size** that is part of the type. `[3]int` and `[5]int` are different types.

```go
package main

import "fmt"

func main() {
    // Declaration with explicit size
    var numbers [5]int
    fmt.Println(numbers) // [0 0 0 0 0] — zero values

    // Initialize with values
    primes := [5]int{2, 3, 5, 7, 11}
    fmt.Println(primes)

    // Let compiler count
    vowels := [...]string{"a", "e", "i", "o", "u"}
    fmt.Println(vowels, len(vowels)) // 5

    // Access and modify
    primes[0] = 1
    fmt.Println(primes[0]) // 1

    // Partial initialization
    sparse := [10]int{1: 10, 5: 50, 9: 90}
    fmt.Println(sparse) // [0 10 0 0 0 50 0 0 0 90]
}
```

### 1.2 Array Iteration

```go
func main() {
    colors := [4]string{"red", "green", "blue", "yellow"}

    // Range-based
    for i, color := range colors {
        fmt.Printf("%d: %s\n", i, color)
    }

    // Classic index-based
    for i := 0; i < len(colors); i++ {
        fmt.Println(colors[i])
    }
}
```

### 1.3 Arrays Are Values

Arrays are **value types** — assigning or passing an array copies the entire thing.

```go
func main() {
    a := [3]int{1, 2, 3}
    b := a     // b is a COPY
    b[0] = 99
    fmt.Println(a) // [1 2 3] — unchanged
    fmt.Println(b) // [99 2 3]

    // Arrays of the same type and size can be compared
    x := [3]int{1, 2, 3}
    y := [3]int{1, 2, 3}
    fmt.Println(x == y) // true
}

// Passing array to function copies it (expensive for large arrays)
func sum(arr [1000]int) int {
    total := 0
    for _, v := range arr {
        total += v
    }
    return total
}
```

---

## 2. Slices

### 2.1 Slice Basics

Slices are Go's answer to dynamic arrays. They are far more common than arrays in practice.

```go
package main

import "fmt"

func main() {
    // Slice literal (no size specified)
    fruits := []string{"apple", "banana", "cherry"}
    fmt.Println(fruits)
    fmt.Println(len(fruits)) // 3
    fmt.Println(cap(fruits)) // 3

    // make — create a slice with length and capacity
    nums := make([]int, 5)     // len=5, cap=5, all zeros
    buf := make([]int, 0, 10)  // len=0, cap=10, empty but pre-allocated

    fmt.Println(nums)
    fmt.Println(buf, len(buf), cap(buf))

    // Append — grows the slice as needed
    buf = append(buf, 1, 2, 3)
    fmt.Println(buf) // [1 2 3]

    // Append multiple
    more := []int{4, 5, 6}
    buf = append(buf, more...)
    fmt.Println(buf) // [1 2 3 4 5 6]

    // nil slice vs empty slice
    var nilSlice []int          // nil, len=0, cap=0
    emptySlice := []int{}       // not nil, len=0, cap=0
    fmt.Println(nilSlice == nil) // true
    fmt.Println(emptySlice == nil) // false
    // Both work identically with append, len, cap, range
}
```

### 2.2 Slicing Operations

```go
func main() {
    s := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    // s[low:high] — elements from low to high-1
    fmt.Println(s[2:5])  // [2 3 4]
    fmt.Println(s[:3])   // [0 1 2]     — from start
    fmt.Println(s[7:])   // [7 8 9]     — to end
    fmt.Println(s[:])    // full slice   — copy of reference

    // Slicing from an array
    arr := [5]int{10, 20, 30, 40, 50}
    slice := arr[1:4] // [20 30 40]
    fmt.Println(slice)

    // WARNING: slice shares underlying array!
    slice[0] = 999
    fmt.Println(arr) // [10 999 30 40 50] — array modified!

    // Three-index slice: s[low:high:cap] — limits capacity
    limited := s[2:5:5] // len=3, cap=3 (instead of 8)
    fmt.Println(len(limited), cap(limited))

    // Copy — creates independent copy
    src := []int{1, 2, 3}
    dst := make([]int, len(src))
    copied := copy(dst, src)
    fmt.Println(dst, copied)
    dst[0] = 99
    fmt.Println(src) // [1 2 3] — unaffected
}
```

### 2.3 Common Slice Patterns

```go
func main() {
    // Remove element at index i
    s := []int{0, 1, 2, 3, 4}
    i := 2
    s = append(s[:i], s[i+1:]...)
    fmt.Println(s) // [0 1 3 4]

    // Insert element at index i
    s = []int{0, 1, 3, 4}
    i = 2
    s = append(s[:i], append([]int{2}, s[i:]...)...)
    fmt.Println(s) // [0 1 2 3 4]

    // Filter — create new slice with matching elements
    nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    evens := filter(nums, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4 6 8 10]

    // Deduplicate (sorted slice)
    sorted := []int{1, 1, 2, 2, 3, 3, 3}
    unique := deduplicate(sorted)
    fmt.Println(unique) // [1 2 3]
}

func filter(s []int, pred func(int) bool) []int {
    var result []int
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

func deduplicate(s []int) []int {
    if len(s) == 0 {
        return s
    }
    result := []int{s[0]}
    for _, v := range s[1:] {
        if v != result[len(result)-1] {
            result = append(result, v)
        }
    }
    return result
}
```

---

## 3. Slice Internals

### 3.1 Memory Layout

A slice is a three-word struct: **pointer**, **length**, and **capacity**.

```
Slice header (24 bytes on 64-bit):
┌──────────┬────────┬──────────┐
│ pointer  │ length │ capacity │
└──────────┴────────┴──────────┘
     │
     ▼
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ . │ . │ . │  Underlying array
└───┴───┴───┴───┴───┴───┴───┴───┘
```

```go
func main() {
    s := make([]int, 3, 8)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s)

    // Append within capacity — no reallocation
    s = append(s, 1)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s) // same pointer

    // Append beyond capacity — reallocates (new pointer)
    s = append(s, 2, 3, 4, 5, 6)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s) // new pointer!
}
```

### 3.2 Growth Strategy

```go
func main() {
    var s []int
    prev := cap(s)
    for i := 0; i < 20; i++ {
        s = append(s, i)
        if cap(s) != prev {
            fmt.Printf("len=%-3d cap changed: %d → %d\n", len(s), prev, cap(s))
            prev = cap(s)
        }
    }
    // Output shows capacity roughly doubles: 0→1→2→4→8→16→32
}
```

### 3.3 Avoiding Memory Leaks

```go
// BAD: returned slice holds reference to entire original array
func getFirstThree(data []int) []int {
    return data[:3] // Still references the large underlying array
}

// GOOD: copy to release the original
func getFirstThreeSafe(data []int) []int {
    result := make([]int, 3)
    copy(result, data[:3])
    return result
}
```

---

## 4. Maps

### 4.1 Map Basics

```go
package main

import "fmt"

func main() {
    // Map literal
    ages := map[string]int{
        "Alice": 30,
        "Bob":   25,
        "Carol": 28,
    }
    fmt.Println(ages)

    // make
    scores := make(map[string]int)

    // Set values
    scores["math"] = 95
    scores["science"] = 88

    // Get value
    fmt.Println(scores["math"]) // 95

    // Check existence — the "comma ok" idiom
    val, ok := ages["Dave"]
    if ok {
        fmt.Println("Dave:", val)
    } else {
        fmt.Println("Dave not found") // This prints
    }

    // Delete
    delete(ages, "Bob")
    fmt.Println(ages)

    // Length
    fmt.Println(len(ages)) // 2

    // Iteration (order is NOT guaranteed)
    for name, age := range ages {
        fmt.Printf("%s: %d\n", name, age)
    }

    // nil map — reads return zero values, writes PANIC
    var m map[string]int // nil
    fmt.Println(m["x"]) // 0 (safe)
    // m["x"] = 1        // PANIC: assignment to entry in nil map
}
```

### 4.2 Map Patterns

```go
func main() {
    // Word frequency counter
    text := "the quick brown fox jumps over the lazy dog the fox"
    freq := wordFrequency(text)
    for word, count := range freq {
        fmt.Printf("%-10s %d\n", word, count)
    }

    // Set using map[T]struct{}
    set := make(map[string]struct{})
    set["apple"] = struct{}{}
    set["banana"] = struct{}{}
    if _, exists := set["apple"]; exists {
        fmt.Println("apple is in the set")
    }

    // Grouping
    students := []struct {
        Name  string
        Grade string
    }{
        {"Alice", "A"}, {"Bob", "B"}, {"Carol", "A"}, {"Dave", "B"},
    }
    groups := make(map[string][]string)
    for _, s := range students {
        groups[s.Grade] = append(groups[s.Grade], s.Name)
    }
    fmt.Println(groups) // map[A:[Alice Carol] B:[Bob Dave]]
}

func wordFrequency(text string) map[string]int {
    freq := make(map[string]int)
    for _, word := range strings.Fields(text) {
        freq[word]++
    }
    return freq
}
```

### 4.3 Maps Are Reference Types

```go
func main() {
    m1 := map[string]int{"a": 1}
    m2 := m1   // m2 points to the SAME underlying hash table
    m2["a"] = 99
    fmt.Println(m1["a"]) // 99 — m1 is affected!

    // Maps cannot be compared with ==
    // Use reflect.DeepEqual or manual comparison
}

func modifyMap(m map[string]int) {
    m["new"] = 42 // Modifies the original
}
```

---

## 5. Structs

### 5.1 Struct Basics

```go
package main

import "fmt"

// Define a struct type
type Person struct {
    Name string
    Age  int
    City string
}

func main() {
    // Literal initialization
    alice := Person{
        Name: "Alice",
        Age:  30,
        City: "Seoul",
    }
    fmt.Println(alice)

    // Positional (not recommended — fragile)
    bob := Person{"Bob", 25, "Tokyo"}
    fmt.Println(bob)

    // Zero value — all fields are zero values
    var empty Person
    fmt.Println(empty) // { 0 }

    // Access and modify fields
    alice.Age = 31
    fmt.Println(alice.Age)

    // Pointer to struct
    p := &alice
    p.City = "Busan"        // Automatic dereferencing
    fmt.Println(alice.City)  // "Busan"

    // new returns a pointer to a zero-value struct
    carol := new(Person)
    carol.Name = "Carol"
    fmt.Println(*carol)

    // Anonymous struct
    point := struct {
        X, Y float64
    }{3.0, 4.0}
    fmt.Println(point)
}
```

### 5.2 Struct Comparison and Copying

```go
func main() {
    // Structs are value types — assignment copies
    a := Person{Name: "Alice", Age: 30, City: "Seoul"}
    b := a
    b.Age = 31
    fmt.Println(a.Age) // 30 — unchanged
    fmt.Println(b.Age) // 31

    // Comparable structs (all fields must be comparable)
    c := Person{Name: "Alice", Age: 30, City: "Seoul"}
    fmt.Println(a == c) // true

    // Structs with slice/map fields are NOT comparable with ==
    type Team struct {
        Name    string
        Members []string // slices cannot be compared
    }
    // t1 == t2 would be a compile error
}
```

### 5.3 Constructor Pattern

```go
type Server struct {
    Host    string
    Port    int
    Timeout time.Duration
    TLS     bool
}

// "Constructor" function — Go convention is NewXxx
func NewServer(host string, port int) *Server {
    return &Server{
        Host:    host,
        Port:    port,
        Timeout: 30 * time.Second, // sensible defaults
        TLS:     true,
    }
}

func main() {
    s := NewServer("localhost", 8080)
    fmt.Printf("%+v\n", s)
}
```

---

## 6. Struct Embedding and Tags

### 6.1 Embedding (Composition)

Go uses embedding instead of inheritance. Embedded fields are "promoted" — their methods and fields are accessible directly.

```go
type Address struct {
    Street string
    City   string
    Zip    string
}

type Employee struct {
    Name    string
    Address          // Embedded — fields promoted
    Company string
}

func main() {
    emp := Employee{
        Name:    "Alice",
        Address: Address{Street: "123 Main", City: "Seoul", Zip: "04500"},
        Company: "Acme",
    }

    // Access promoted fields directly
    fmt.Println(emp.City)        // "Seoul" — promoted from Address
    fmt.Println(emp.Address.City) // "Seoul" — explicit access also works
}
```

### 6.2 Struct Tags

Tags are metadata attached to struct fields, used by packages like `encoding/json`.

```go
import "encoding/json"

type User struct {
    ID        int    `json:"id"`
    FirstName string `json:"first_name"`
    LastName  string `json:"last_name"`
    Email     string `json:"email,omitempty"` // omit if empty
    Password  string `json:"-"`               // never include in JSON
}

func main() {
    u := User{
        ID:        1,
        FirstName: "Alice",
        LastName:  "Kim",
        Email:     "",
        Password:  "secret",
    }

    data, _ := json.MarshalIndent(u, "", "  ")
    fmt.Println(string(data))
    // {
    //   "id": 1,
    //   "first_name": "Alice",
    //   "last_name": "Kim"
    // }
    // Note: Email omitted (empty), Password omitted (-)
}
```

---

## 7. Summary

### Key Takeaways

1. **Arrays have fixed size** — they are value types and rarely used directly. Prefer slices.
2. **Slices are reference types** — they point to an underlying array. Be aware of shared memory when slicing.
3. **`append` may allocate** — when capacity is exceeded, a new larger array is allocated and data is copied.
4. **Maps are unordered** — iteration order is randomized. Use sorted keys if order matters.
5. **Structs are value types** — assignment copies all fields. Use pointers for large structs or when mutation is needed.
6. **Embedding over inheritance** — Go uses composition. Embed structs to promote fields and methods.
7. **Struct tags** — metadata for serialization, validation, and ORM mapping. The `json` tag is the most common.

### Type Summary

| Type | Value/Ref | Zero Value | Comparable | Mutable |
|------|-----------|------------|------------|---------|
| Array | Value | `[0, 0, ...]` | Yes | Yes |
| Slice | Reference | `nil` | No | Yes |
| Map | Reference | `nil` | No | Yes |
| Struct | Value | All fields zero | If all fields are | Yes |

---

## Exercises

### Exercise 1: Matrix Operations
Create a 3x3 matrix type using `[3][3]float64`. Implement `add`, `transpose`, and `multiply` functions.

### Exercise 2: Stack Implementation
Implement a stack using a slice with `Push`, `Pop`, and `Peek` operations. Handle empty stack gracefully.

### Exercise 3: Student Records
Define a `Student` struct with name, grades (slice), and calculate GPA. Create a function to find the top N students from a slice of students.

### Exercise 4: Word Index
Build a word index from a text: `map[string][]int` where keys are words and values are line numbers where each word appears.

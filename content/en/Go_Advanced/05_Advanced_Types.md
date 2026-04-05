# 16. Advanced Types

**Previous**: [CLI Tools](./04_CLI_Tools.md) | **Next**: [Reflection and Code Generation](./06_Reflection_and_Codegen.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use generics with type parameters and constraints (Go 1.18+)
2. Define custom type constraints
3. Implement generic data structures: stack, queue, set, ordered map
4. Apply generics appropriately — when to use and when to avoid
5. Use the `constraints` and `slices` packages

---

Go 1.18 introduced generics (type parameters), the most significant language change since Go 1.0. Generics allow writing type-safe, reusable code without sacrificing Go's simplicity. This lesson covers generics from basic syntax through real-world data structure implementation.

## Table of Contents
1. [Generic Functions](#1-generic-functions)
2. [Type Constraints](#2-type-constraints)
3. [Generic Data Structures](#3-generic-data-structures)
4. [Standard Library Generics](#4-standard-library-generics)
5. [When to Use Generics](#5-when-to-use-generics)
6. [Advanced Constraint Patterns](#6-advanced-constraint-patterns)
7. [Summary](#7-summary)

---

## 1. Generic Functions

### 1.1 Basic Syntax

```go
package main

import "fmt"

// Type parameter in square brackets
func Min[T int | float64 | string](a, b T) T {
    if a < b {
        return a
    }
    return b
}

// Multiple type parameters
func Map[T any, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

func Filter[T any](s []T, pred func(T) bool) []T {
    var result []T
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

func Reduce[T any, U any](s []T, initial U, f func(U, T) U) U {
    acc := initial
    for _, v := range s {
        acc = f(acc, v)
    }
    return acc
}

func main() {
    fmt.Println(Min(3, 5))         // 3
    fmt.Println(Min(3.14, 2.71))   // 2.71
    fmt.Println(Min("abc", "xyz")) // "abc"

    nums := []int{1, 2, 3, 4, 5}
    doubled := Map(nums, func(n int) int { return n * 2 })
    fmt.Println(doubled) // [2 4 6 8 10]

    strs := Map(nums, func(n int) string { return fmt.Sprintf("#%d", n) })
    fmt.Println(strs) // [#1 #2 #3 #4 #5]

    evens := Filter(nums, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4]

    sum := Reduce(nums, 0, func(acc, n int) int { return acc + n })
    fmt.Println(sum) // 15
}
```

### 1.2 Type Inference

```go
func Contains[T comparable](s []T, target T) bool {
    for _, v := range s {
        if v == target {
            return true
        }
    }
    return false
}

func main() {
    // Type is inferred — no need to specify
    Contains([]int{1, 2, 3}, 2)           // T = int
    Contains([]string{"a", "b"}, "b")     // T = string

    // Explicit type parameter (rarely needed)
    Contains[int]([]int{1, 2, 3}, 2)
}
```

---

## 2. Type Constraints

### 2.1 Built-in Constraints

```go
import "cmp"

// comparable — supports == and != (built-in)
func IndexOf[T comparable](s []T, target T) int {
    for i, v := range s {
        if v == target {
            return i
        }
    }
    return -1
}

// cmp.Ordered — supports <, <=, >, >= (int, float, string)
func Max[T cmp.Ordered](vals ...T) T {
    m := vals[0]
    for _, v := range vals[1:] {
        if v > m {
            m = v
        }
    }
    return m
}

// any — no constraint (like interface{})
func Identity[T any](v T) T {
    return v
}
```

### 2.2 Custom Constraints

```go
// Constraint as interface
type Number interface {
    int | int8 | int16 | int32 | int64 |
    float32 | float64
}

func Sum[T Number](nums []T) T {
    var total T
    for _, n := range nums {
        total += n
    }
    return total
}

// Constraint with method
type Stringer interface {
    String() string
}

func JoinStrings[T Stringer](items []T, sep string) string {
    var parts []string
    for _, item := range items {
        parts = append(parts, item.String())
    }
    return strings.Join(parts, sep)
}

// Constraint with underlying type (tilde ~)
type Integer interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64
}

type UserID int64
type ProductID int64

// Works with UserID, ProductID, and plain int64
func Abs[T Integer](n T) T {
    if n < 0 {
        return -n
    }
    return n
}
```

---

## 3. Generic Data Structures

### 3.1 Stack

```go
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    return s.items[len(s.items)-1], true
}

func (s *Stack[T]) Len() int { return len(s.items) }

func main() {
    s := &Stack[int]{}
    s.Push(1)
    s.Push(2)
    s.Push(3)
    val, _ := s.Pop()
    fmt.Println(val) // 3
}
```

### 3.2 Set

```go
type Set[T comparable] struct {
    items map[T]struct{}
}

func NewSet[T comparable](items ...T) *Set[T] {
    s := &Set[T]{items: make(map[T]struct{})}
    for _, item := range items {
        s.Add(item)
    }
    return s
}

func (s *Set[T]) Add(item T)              { s.items[item] = struct{}{} }
func (s *Set[T]) Remove(item T)           { delete(s.items, item) }
func (s *Set[T]) Contains(item T) bool    { _, ok := s.items[item]; return ok }
func (s *Set[T]) Len() int                { return len(s.items) }

func (s *Set[T]) Union(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        result.Add(item)
    }
    for item := range other.items {
        result.Add(item)
    }
    return result
}

func (s *Set[T]) Intersection(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        if other.Contains(item) {
            result.Add(item)
        }
    }
    return result
}

func main() {
    a := NewSet(1, 2, 3, 4)
    b := NewSet(3, 4, 5, 6)
    fmt.Println(a.Union(b).Len())        // 6
    fmt.Println(a.Intersection(b).Len()) // 2
}
```

### 3.3 Result Type

```go
type Result[T any] struct {
    value T
    err   error
}

func Ok[T any](value T) Result[T] {
    return Result[T]{value: value}
}

func Err[T any](err error) Result[T] {
    return Result[T]{err: err}
}

func (r Result[T]) IsOk() bool  { return r.err == nil }
func (r Result[T]) IsErr() bool { return r.err != nil }

func (r Result[T]) Unwrap() T {
    if r.err != nil {
        panic(fmt.Sprintf("unwrap on error: %v", r.err))
    }
    return r.value
}

func (r Result[T]) UnwrapOr(defaultVal T) T {
    if r.err != nil {
        return defaultVal
    }
    return r.value
}

func MapResult[T, U any](r Result[T], f func(T) U) Result[U] {
    if r.err != nil {
        return Err[U](r.err)
    }
    return Ok(f(r.value))
}
```

---

## 4. Standard Library Generics

### 4.1 slices Package

```go
import "slices"

func main() {
    s := []int{3, 1, 4, 1, 5, 9}

    slices.Sort(s)
    fmt.Println(s) // [1 1 3 4 5 9]

    idx, found := slices.BinarySearch(s, 4)
    fmt.Println(idx, found) // 3 true

    fmt.Println(slices.Contains(s, 5))    // true
    fmt.Println(slices.Index(s, 4))       // 3
    fmt.Println(slices.Min(s))            // 1
    fmt.Println(slices.Max(s))            // 9

    compact := slices.Compact(s) // Remove consecutive duplicates
    fmt.Println(compact) // [1 3 4 5 9]

    reversed := slices.Clone(s)
    slices.Reverse(reversed)
    fmt.Println(reversed)
}
```

### 4.2 maps Package

```go
import "maps"

func main() {
    m := map[string]int{"a": 1, "b": 2, "c": 3}

    keys := maps.Keys(m)   // Unsorted
    vals := maps.Values(m)
    fmt.Println(keys, vals)

    // Clone
    m2 := maps.Clone(m)
    m2["d"] = 4
    fmt.Println(m)  // Original unchanged
    fmt.Println(m2)

    // Equal
    fmt.Println(maps.Equal(m, maps.Clone(m))) // true

    // Delete by predicate
    maps.DeleteFunc(m2, func(k string, v int) bool {
        return v > 2
    })
}
```

---

## 5. When to Use Generics

### 5.1 Good Use Cases

```go
// 1. Collections and data structures
type Queue[T any] struct { /* ... */ }
type OrderedMap[K cmp.Ordered, V any] struct { /* ... */ }

// 2. Utility functions that work across types
func Keys[K comparable, V any](m map[K]V) []K { /* ... */ }
func Values[K comparable, V any](m map[K]V) []V { /* ... */ }

// 3. Algorithm implementations
func Sort[T cmp.Ordered](s []T) { /* ... */ }
func BinarySearch[T cmp.Ordered](s []T, target T) int { /* ... */ }

// 4. Functional patterns
func Map[T, U any](s []T, f func(T) U) []U { /* ... */ }
func Filter[T any](s []T, f func(T) bool) []T { /* ... */ }
```

### 5.2 When NOT to Use Generics

```go
// DON'T: When interfaces work fine
// BAD — unnecessarily generic
func PrintAll[T fmt.Stringer](items []T) {
    for _, item := range items {
        fmt.Println(item.String())
    }
}

// GOOD — interface is simpler
func PrintAll(items []fmt.Stringer) {
    for _, item := range items {
        fmt.Println(item.String())
    }
}

// DON'T: When the type is known
// BAD — just use []string
func JoinGeneric[T ~string](items []T, sep T) T { /* ... */ }

// GOOD — concrete type is clearer
func Join(items []string, sep string) string { /* ... */ }
```

---

## 6. Advanced Constraint Patterns

### 6.1 Interface with Type and Method Constraints

```go
type Addable interface {
    ~int | ~float64
    String() string
}

// Constraint requiring method on pointer receiver
type Validator interface {
    Validate() error
}

func ValidateAll[T Validator](items []T) error {
    for _, item := range items {
        if err := item.Validate(); err != nil {
            return err
        }
    }
    return nil
}
```

### 6.2 Self-Referential Constraints

```go
type Comparable[T any] interface {
    CompareTo(other T) int
}

func SortCustom[T Comparable[T]](s []T) {
    sort.Slice(s, func(i, j int) bool {
        return s[i].CompareTo(s[j]) < 0
    })
}
```

---

## 7. Summary

### Key Takeaways

1. **Generics use `[T constraint]` syntax** — type parameters in square brackets after function/type name.
2. **`comparable`** — built-in constraint for `==` and `!=`.
3. **`cmp.Ordered`** — constraint for `<`, `>`, `<=`, `>=` (numbers and strings).
4. **Tilde `~`** — matches underlying types (e.g., `~int` includes `type MyInt int`).
5. **Use generics for data structures and algorithms** — where type-safety matters across types.
6. **Prefer interfaces when behavior matters more than type** — generics are for type parameters, not replacing interfaces.
7. **`slices` and `maps` packages** — generic utilities for common operations.

---

## Exercises

### Exercise 1: Generic Linked List
Implement a generic doubly-linked list with `PushFront`, `PushBack`, `PopFront`, `PopBack`, `Find`, and `ForEach`.

### Exercise 2: Generic Cache
Build a generic LRU cache `Cache[K comparable, V any]` with `Get`, `Set`, `Delete`, and `Len`. Use a map and a doubly-linked list.

### Exercise 3: Functional Pipeline
Create a generic pipeline: `Pipe(data, Transform1, Transform2, Filter, Reduce)` where each stage is type-safe.

### Exercise 4: Binary Search Tree
Implement a generic BST `Tree[T cmp.Ordered]` with `Insert`, `Search`, `Delete`, `InOrder`, `Min`, and `Max`.

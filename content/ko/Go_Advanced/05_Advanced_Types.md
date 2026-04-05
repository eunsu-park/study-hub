# 16. 고급 타입

**이전**: [CLI 도구](./04_CLI_Tools.md) | **다음**: [리플렉션과 코드 생성](./06_Reflection_and_Codegen.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 타입 매개변수와 제약 조건으로 제네릭을 사용한다 (Go 1.18+)
2. 커스텀 타입 제약 조건을 정의한다
3. 제네릭 자료구조를 구현한다: 스택, 큐, 집합, 정렬 맵
4. 제네릭을 적절히 적용한다 — 사용해야 할 때와 피해야 할 때를 판단한다
5. `constraints`와 `slices` 패키지를 사용한다

---

Go 1.18에서 제네릭(타입 매개변수)이 도입되었다. 이는 Go 1.0 이후 가장 중요한 언어 변경 사항이다. 제네릭을 사용하면 Go의 단순함을 유지하면서 타입 안전하고 재사용 가능한 코드를 작성할 수 있다. 이 레슨에서는 기본 문법부터 실제 자료구조 구현까지 제네릭을 다룬다.

## 목차
1. [제네릭 함수](#1-제네릭-함수)
2. [타입 제약 조건](#2-타입-제약-조건)
3. [제네릭 자료구조](#3-제네릭-자료구조)
4. [표준 라이브러리 제네릭](#4-표준-라이브러리-제네릭)
5. [제네릭을 사용해야 할 때](#5-제네릭을-사용해야-할-때)
6. [고급 제약 조건 패턴](#6-고급-제약-조건-패턴)
7. [요약](#7-요약)

---

## 1. 제네릭 함수

### 1.1 기본 문법

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

### 1.2 타입 추론

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

## 2. 타입 제약 조건

### 2.1 내장 제약 조건

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

### 2.2 커스텀 제약 조건

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

## 3. 제네릭 자료구조

### 3.1 스택

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

### 3.2 집합

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

### 3.3 Result 타입

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

## 4. 표준 라이브러리 제네릭

### 4.1 slices 패키지

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

### 4.2 maps 패키지

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

## 5. 제네릭을 사용해야 할 때

### 5.1 좋은 사용 사례

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

### 5.2 제네릭을 사용하지 말아야 할 때

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

## 6. 고급 제약 조건 패턴

### 6.1 타입과 메서드 제약 조건을 가진 인터페이스

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

### 6.2 자기 참조 제약 조건

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

## 7. 요약

### 핵심 포인트

1. **제네릭은 `[T constraint]` 문법을 사용한다** — 함수/타입 이름 뒤에 대괄호로 타입 매개변수를 지정한다.
2. **`comparable`** — `==`와 `!=`를 지원하는 내장 제약 조건이다.
3. **`cmp.Ordered`** — `<`, `>`, `<=`, `>=`를 지원하는 제약 조건이다(숫자와 문자열).
4. **틸드 `~`** — 기반 타입을 매칭한다(예: `~int`는 `type MyInt int`도 포함한다).
5. **자료구조와 알고리즘에 제네릭을 사용한다** — 여러 타입에서 타입 안전성이 중요한 경우에 적합하다.
6. **동작이 타입보다 중요할 때는 인터페이스를 선호한다** — 제네릭은 인터페이스를 대체하는 것이 아니라 타입 매개변수를 위한 것이다.
7. **`slices`와 `maps` 패키지** — 일반적인 연산을 위한 제네릭 유틸리티를 제공한다.

---

## 연습 문제

### 연습 1: 제네릭 연결 리스트
`PushFront`, `PushBack`, `PopFront`, `PopBack`, `Find`, `ForEach`를 가진 제네릭 이중 연결 리스트를 구현한다.

### 연습 2: 제네릭 캐시
`Get`, `Set`, `Delete`, `Len`을 가진 제네릭 LRU 캐시 `Cache[K comparable, V any]`를 구축한다. 맵과 이중 연결 리스트를 사용한다.

### 연습 3: 함수형 파이프라인
각 단계가 타입 안전한 제네릭 파이프라인을 만든다: `Pipe(data, Transform1, Transform2, Filter, Reduce)`.

### 연습 4: 이진 탐색 트리
`Insert`, `Search`, `Delete`, `InOrder`, `Min`, `Max`를 가진 제네릭 BST `Tree[T cmp.Ordered]`를 구현한다.

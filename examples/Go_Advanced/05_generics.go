// 16_generics.go — Generic data structures (Go 1.18+)
//
// Run: go run 16_generics.go

package main

import (
	"cmp"
	"fmt"
)

// Generic Stack
type Stack[T any] struct {
	items []T
}

func (s *Stack[T]) Push(item T)      { s.items = append(s.items, item) }
func (s *Stack[T]) Len() int         { return len(s.items) }

func (s *Stack[T]) Pop() (T, bool) {
	if len(s.items) == 0 {
		var zero T
		return zero, false
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return item, true
}

// Generic Set
type Set[T comparable] struct {
	items map[T]struct{}
}

func NewSet[T comparable](vals ...T) *Set[T] {
	s := &Set[T]{items: make(map[T]struct{})}
	for _, v := range vals {
		s.Add(v)
	}
	return s
}

func (s *Set[T]) Add(v T)           { s.items[v] = struct{}{} }
func (s *Set[T]) Contains(v T) bool { _, ok := s.items[v]; return ok }
func (s *Set[T]) Len() int          { return len(s.items) }

// Generic utility functions
func Map[T, U any](s []T, f func(T) U) []U {
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

func Min[T cmp.Ordered](vals ...T) T {
	m := vals[0]
	for _, v := range vals[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func main() {
	fmt.Println("=== Generic Stack ===")
	s := &Stack[int]{}
	s.Push(1)
	s.Push(2)
	s.Push(3)
	for s.Len() > 0 {
		v, _ := s.Pop()
		fmt.Printf("%d ", v)
	}
	fmt.Println()

	fmt.Println("\n=== Generic Set ===")
	set := NewSet(1, 2, 3, 4, 5)
	fmt.Println("Contains 3:", set.Contains(3))
	fmt.Println("Contains 6:", set.Contains(6))

	fmt.Println("\n=== Generic Map/Filter ===")
	nums := []int{1, 2, 3, 4, 5}
	doubled := Map(nums, func(n int) int { return n * 2 })
	fmt.Println("Doubled:", doubled)

	evens := Filter(nums, func(n int) bool { return n%2 == 0 })
	fmt.Println("Evens:", evens)

	fmt.Println("\n=== Min ===")
	fmt.Println("Min(3,1,4,1,5):", Min(3, 1, 4, 1, 5))
	fmt.Println("Min(\"c\",\"a\",\"b\"):", Min("c", "a", "b"))
}

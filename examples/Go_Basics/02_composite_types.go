// 02_composite_types.go — Arrays, slices, maps, structs
//
// Run: go run 02_composite_types.go

package main

import (
	"fmt"
	"sort"
	"strings"
)

type Student struct {
	Name   string
	Grades []int
}

func (s Student) GPA() float64 {
	if len(s.Grades) == 0 {
		return 0
	}
	sum := 0
	for _, g := range s.Grades {
		sum += g
	}
	return float64(sum) / float64(len(s.Grades))
}

func main() {
	fmt.Println("=== Slices ===")
	sliceDemo()

	fmt.Println("\n=== Maps ===")
	mapDemo()

	fmt.Println("\n=== Structs ===")
	structDemo()
}

func sliceDemo() {
	nums := []int{5, 3, 8, 1, 9, 2, 7}
	fmt.Println("Original:", nums)

	sorted := make([]int, len(nums))
	copy(sorted, nums)
	sort.Ints(sorted)
	fmt.Println("Sorted:", sorted)

	var evens []int
	for _, n := range nums {
		if n%2 == 0 {
			evens = append(evens, n)
		}
	}
	fmt.Println("Evens:", evens)

	s := []int{0, 1, 2, 3, 4, 5}
	fmt.Println("s[2:4]:", s[2:4])
	fmt.Printf("len=%d cap=%d\n", len(s), cap(s))
}

func mapDemo() {
	text := "the quick brown fox jumps over the lazy dog the fox"
	freq := make(map[string]int)
	for _, word := range strings.Fields(text) {
		freq[word]++
	}
	fmt.Println("Word frequencies:")
	for word, count := range freq {
		fmt.Printf("  %-10s %d\n", word, count)
	}
}

func structDemo() {
	students := []Student{
		{Name: "Alice", Grades: []int{95, 87, 92, 88}},
		{Name: "Bob", Grades: []int{78, 82, 90, 85}},
		{Name: "Carol", Grades: []int{92, 95, 98, 91}},
	}

	for _, s := range students {
		fmt.Printf("%s: GPA=%.1f\n", s.Name, s.GPA())
	}
}

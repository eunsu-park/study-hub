// 03_functions_methods.go — Function types, methods, receivers, closures
//
// Run: go run 03_functions_methods.go

package main

import (
	"fmt"
	"math"
	"strings"
)

type Point struct {
	X, Y float64
}

func (p Point) Distance(q Point) float64 {
	dx := p.X - q.X
	dy := p.Y - q.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func (p Point) String() string {
	return fmt.Sprintf("(%g, %g)", p.X, p.Y)
}

type Circle struct {
	Center Point
	Radius float64
}

func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

func (c *Circle) Scale(factor float64) {
	c.Radius *= factor
}

func main() {
	fmt.Println("=== Higher-Order Functions ===")
	words := []string{"hello", "world", "go"}
	upper := mapStrings(words, strings.ToUpper)
	fmt.Println(upper)

	nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	evens := filterInts(nums, func(n int) bool { return n%2 == 0 })
	fmt.Println("Evens:", evens)

	sum := reduce(nums, 0, func(a, b int) int { return a + b })
	fmt.Println("Sum:", sum)

	fmt.Println("\n=== Methods ===")
	a := Point{3, 4}
	b := Point{0, 0}
	fmt.Printf("Distance from %s to %s: %.2f\n", a, b, a.Distance(b))

	c := Circle{Center: Point{0, 0}, Radius: 5}
	fmt.Printf("Area: %.2f\n", c.Area())
	c.Scale(2)
	fmt.Printf("After scale(2), Area: %.2f\n", c.Area())

	fmt.Println("\n=== Closures ===")
	counter := makeCounter()
	fmt.Println(counter(), counter(), counter())

	adder := makeAdder(10)
	fmt.Println(adder(5), adder(3))
}

func mapStrings(ss []string, f func(string) string) []string {
	result := make([]string, len(ss))
	for i, s := range ss {
		result[i] = f(s)
	}
	return result
}

func filterInts(nums []int, pred func(int) bool) []int {
	var result []int
	for _, n := range nums {
		if pred(n) {
			result = append(result, n)
		}
	}
	return result
}

func reduce(nums []int, initial int, f func(int, int) int) int {
	acc := initial
	for _, n := range nums {
		acc = f(acc, n)
	}
	return acc
}

func makeCounter() func() int {
	n := 0
	return func() int {
		n++
		return n
	}
}

func makeAdder(base int) func(int) int {
	return func(n int) int {
		return base + n
	}
}

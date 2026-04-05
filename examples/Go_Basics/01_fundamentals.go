// 01_fundamentals.go — Go fundamentals: variables, types, control flow, functions
//
// Run: go run 01_fundamentals.go

package main

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

func main() {
	fmt.Println("=== Variables ===")
	variablesDemo()

	fmt.Println("\n=== Types ===")
	typesDemo()

	fmt.Println("\n=== Control Flow ===")
	controlFlowDemo()

	fmt.Println("\n=== Functions ===")
	functionsDemo()
}

func variablesDemo() {
	var name string = "Alice"
	var age int = 30
	city := "Seoul"
	pi := 3.14159

	var count int
	var rate float64
	var label string
	var active bool

	fmt.Printf("Name: %s, Age: %d, City: %s, Pi: %.2f\n", name, age, city, pi)
	fmt.Printf("Zeros: count=%d, rate=%.1f, label=%q, active=%t\n", count, rate, label, active)

	x, y := 10, 20
	x, y = y, x
	fmt.Printf("After swap: x=%d, y=%d\n", x, y)
}

func typesDemo() {
	fmt.Printf("MaxInt64: %d\n", math.MaxInt64)
	fmt.Printf("Hex: %x, Oct: %o, Bin: %b\n", 255, 255, 42)

	greeting := "Hello, 세계"
	fmt.Printf("Bytes: %d, Runes: %d\n", len(greeting), utf8.RuneCountInString(greeting))

	for i, r := range greeting {
		fmt.Printf("  byte %d: %c (U+%04X)\n", i, r, r)
	}

	var b strings.Builder
	for i := 0; i < 5; i++ {
		fmt.Fprintf(&b, "%d ", i)
	}
	fmt.Println("Built:", b.String())
}

func controlFlowDemo() {
	sum := 0
	for i := 1; i <= 10; i++ {
		sum += i
	}
	fmt.Println("Sum 1-10:", sum)

	score := 85
	var grade string
	switch {
	case score >= 90:
		grade = "A"
	case score >= 80:
		grade = "B"
	case score >= 70:
		grade = "C"
	default:
		grade = "F"
	}
	fmt.Printf("Score %d = Grade %s\n", score, grade)
}

func functionsDemo() {
	result, err := divide(10, 3)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("10 / 3 = %.4f\n", result)
	}

	fmt.Println("Sum:", variadic(1, 2, 3, 4, 5))

	counter := makeCounter()
	fmt.Println(counter(), counter(), counter())
}

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, fmt.Errorf("division by zero")
	}
	return a / b, nil
}

func variadic(nums ...int) int {
	total := 0
	for _, n := range nums {
		total += n
	}
	return total
}

func makeCounter() func() int {
	n := 0
	return func() int {
		n++
		return n
	}
}

// 10_testing_example.go — Testing patterns demonstration
//
// Run: go run 10_testing_example.go
// Note: This shows test patterns. In practice, use _test.go files.

package main

import "fmt"

// Functions to be tested
func Add(a, b int) int { return a + b }

func Divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, fmt.Errorf("division by zero")
	}
	return a / b, nil
}

func Reverse(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

func main() {
	fmt.Println("=== Table-Driven Test Pattern ===")

	tests := []struct {
		name string
		a, b int
		want int
	}{
		{"positive", 2, 3, 5},
		{"negative", -1, -2, -3},
		{"zero", 0, 0, 0},
		{"mixed", -5, 10, 5},
	}

	for _, tt := range tests {
		got := Add(tt.a, tt.b)
		status := "PASS"
		if got != tt.want {
			status = "FAIL"
		}
		fmt.Printf("  %s: Add(%d, %d) = %d (want %d) [%s]\n",
			tt.name, tt.a, tt.b, got, tt.want, status)
	}

	fmt.Println("\n=== Reverse Tests ===")
	reverseTests := []struct {
		input, want string
	}{
		{"hello", "olleh"},
		{"", ""},
		{"한국어", "어국한"},
		{"a", "a"},
	}

	for _, tt := range reverseTests {
		got := Reverse(tt.input)
		status := "PASS"
		if got != tt.want {
			status = "FAIL"
		}
		fmt.Printf("  Reverse(%q) = %q (want %q) [%s]\n",
			tt.input, got, tt.want, status)
	}
}

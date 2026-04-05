// 04_interfaces.go — Interface design, duck typing, type assertions
//
// Run: go run 04_interfaces.go

package main

import (
	"fmt"
	"math"
)

type Shape interface {
	Area() float64
	Perimeter() float64
}

type Circle struct {
	Radius float64
}

func (c Circle) Area() float64      { return math.Pi * c.Radius * c.Radius }
func (c Circle) Perimeter() float64 { return 2 * math.Pi * c.Radius }

type Rectangle struct {
	Width, Height float64
}

func (r Rectangle) Area() float64      { return r.Width * r.Height }
func (r Rectangle) Perimeter() float64 { return 2 * (r.Width + r.Height) }

func printShape(s Shape) {
	fmt.Printf("  Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func largestShape(shapes []Shape) Shape {
	largest := shapes[0]
	for _, s := range shapes[1:] {
		if s.Area() > largest.Area() {
			largest = s
		}
	}
	return largest
}

func describe(val any) string {
	switch v := val.(type) {
	case int:
		return fmt.Sprintf("integer: %d", v)
	case string:
		return fmt.Sprintf("string: %q", v)
	case Shape:
		return fmt.Sprintf("shape with area %.2f", v.Area())
	default:
		return fmt.Sprintf("unknown: %T", v)
	}
}

func main() {
	shapes := []Shape{
		Circle{Radius: 5},
		Rectangle{Width: 10, Height: 3},
		Circle{Radius: 1},
		Rectangle{Width: 4, Height: 4},
	}

	fmt.Println("=== Shapes ===")
	for _, s := range shapes {
		fmt.Printf("%T:\n", s)
		printShape(s)
	}

	fmt.Printf("\nLargest: %T with area %.2f\n", largestShape(shapes), largestShape(shapes).Area())

	fmt.Println("\n=== Type Switch ===")
	values := []any{42, "hello", Circle{Radius: 3}, true}
	for _, v := range values {
		fmt.Println(describe(v))
	}
}

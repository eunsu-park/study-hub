// 18_profiling.go — Benchmarking and optimization techniques
//
// Run: go run 18_profiling.go

package main

import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

func main() {
	fmt.Println("=== Memory Stats ===")
	printMemStats()

	fmt.Println("\n=== String Concatenation Benchmark ===")
	sizes := []int{100, 1000, 10000}
	for _, size := range sizes {
		d1 := benchmark(func() { concatPlus(size) })
		d2 := benchmark(func() { concatBuilder(size) })
		fmt.Printf("Size %5d: plus=%v builder=%v speedup=%.1fx\n",
			size, d1, d2, float64(d1)/float64(d2))
	}

	fmt.Println("\n=== Preallocate vs Append ===")
	n := 100000
	d1 := benchmark(func() { appendGrow(n) })
	d2 := benchmark(func() { appendPrealloc(n) })
	fmt.Printf("Append:     %v\n", d1)
	fmt.Printf("Preallocate: %v (%.1fx faster)\n", d2, float64(d1)/float64(d2))
}

func concatPlus(n int) string {
	s := ""
	for i := 0; i < n; i++ {
		s += "x"
	}
	return s
}

func concatBuilder(n int) string {
	var b strings.Builder
	b.Grow(n)
	for i := 0; i < n; i++ {
		b.WriteString("x")
	}
	return b.String()
}

func appendGrow(n int) []int {
	var s []int
	for i := 0; i < n; i++ {
		s = append(s, i)
	}
	return s
}

func appendPrealloc(n int) []int {
	s := make([]int, 0, n)
	for i := 0; i < n; i++ {
		s = append(s, i)
	}
	return s
}

func benchmark(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc:      %d KB\n", m.Alloc/1024)
	fmt.Printf("TotalAlloc: %d KB\n", m.TotalAlloc/1024)
	fmt.Printf("Sys:        %d KB\n", m.Sys/1024)
	fmt.Printf("NumGC:      %d\n", m.NumGC)
	fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())
}

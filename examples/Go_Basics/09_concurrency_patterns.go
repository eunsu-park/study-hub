// 09_concurrency_patterns.go — Worker pool, pipeline, context
//
// Run: go run 09_concurrency_patterns.go

package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	fmt.Println("=== Worker Pool ===")
	workerPoolDemo()

	fmt.Println("\n=== Pipeline ===")
	pipelineDemo()

	fmt.Println("\n=== Context Cancellation ===")
	contextDemo()
}

func workerPoolDemo() {
	jobs := make(chan int, 100)
	results := make(chan int, 100)

	// Start 3 workers
	var wg sync.WaitGroup
	for w := 1; w <= 3; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := range jobs {
				time.Sleep(50 * time.Millisecond)
				results <- j * j
			}
		}(w)
	}

	// Submit 10 jobs
	for j := 1; j <= 10; j++ {
		jobs <- j
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		fmt.Printf("Result: %d\n", r)
	}
}

func pipelineDemo() {
	nums := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	squared := square(nums)
	evens := filter(squared, func(n int) bool { return n%2 == 0 })

	for val := range evens {
		fmt.Printf("%d ", val)
	}
	fmt.Println()
}

func generate(nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _, n := range nums {
			out <- n
		}
	}()
	return out
}

func square(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			out <- n * n
		}
	}()
	return out
}

func filter(in <-chan int, pred func(int) bool) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			if pred(n) {
				out <- n
			}
		}
	}()
	return out
}

func contextDemo() {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	ch := make(chan string, 1)
	go func() {
		time.Sleep(200 * time.Millisecond)
		ch <- "result"
	}()

	select {
	case result := <-ch:
		fmt.Println("Got:", result)
	case <-ctx.Done():
		fmt.Println("Cancelled:", ctx.Err())
	}
}

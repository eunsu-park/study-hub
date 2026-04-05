// 08_channels.go — Channel types, select, patterns
//
// Run: go run 08_channels.go

package main

import (
	"fmt"
	"sync"
	"time"
)

func main() {
	fmt.Println("=== Basic Channel ===")
	basicChannel()

	fmt.Println("\n=== Generator Pattern ===")
	for val := range fibonacci(10) {
		fmt.Printf("%d ", val)
	}
	fmt.Println()

	fmt.Println("\n=== Select with Timeout ===")
	selectDemo()

	fmt.Println("\n=== Fan-In ===")
	fanInDemo()
}

func basicChannel() {
	ch := make(chan string)

	go func() {
		ch <- "Hello from goroutine!"
	}()

	msg := <-ch
	fmt.Println(msg)
}

func fibonacci(n int) <-chan int {
	ch := make(chan int)
	go func() {
		defer close(ch)
		a, b := 0, 1
		for i := 0; i < n; i++ {
			ch <- a
			a, b = b, a+b
		}
	}()
	return ch
}

func selectDemo() {
	ch := make(chan string, 1)

	go func() {
		time.Sleep(100 * time.Millisecond)
		ch <- "data"
	}()

	select {
	case msg := <-ch:
		fmt.Println("Received:", msg)
	case <-time.After(200 * time.Millisecond):
		fmt.Println("Timeout!")
	}
}

func fanInDemo() {
	ch1 := produce("A", 3)
	ch2 := produce("B", 3)

	merged := fanIn(ch1, ch2)
	for val := range merged {
		fmt.Println(val)
	}
}

func produce(prefix string, count int) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		for i := 0; i < count; i++ {
			ch <- fmt.Sprintf("%s-%d", prefix, i)
			time.Sleep(50 * time.Millisecond)
		}
	}()
	return ch
}

func fanIn(channels ...<-chan string) <-chan string {
	var wg sync.WaitGroup
	merged := make(chan string)

	for _, ch := range channels {
		wg.Add(1)
		go func(c <-chan string) {
			defer wg.Done()
			for val := range c {
				merged <- val
			}
		}(ch)
	}

	go func() {
		wg.Wait()
		close(merged)
	}()

	return merged
}

/*
Exercise: Go Web Basics
Practice with HTTP handlers, middleware, JSON encoding, and routing.

Run: go run 19_go_web_basics.go
Test: go test -v -run TestExercise
*/

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"
)

// ========== Exercise 1: JSON Response Helper ==========
// Implement a helper that writes JSON responses with proper headers.

func WriteJSON(w http.ResponseWriter, status int, data any) error {
	// TODO: Implement
	// 1. Set Content-Type header to "application/json"
	// 2. Write the status code
	// 3. Encode data as JSON to the response writer
	// 4. Return any encoding error
	return nil
}

// Test:
func TestExercise1() {
	w := httptest.NewRecorder()
	WriteJSON(w, http.StatusOK, map[string]string{"msg": "hello"})
	if w.Code != 200 {
		fmt.Println("FAIL: expected status 200, got", w.Code)
	} else if !strings.Contains(w.Header().Get("Content-Type"), "application/json") {
		fmt.Println("FAIL: Content-Type not set")
	} else {
		fmt.Println("PASS: Exercise 1")
	}
}

// ========== Exercise 2: Request Logger Middleware ==========
// Implement middleware that logs method, path, status, and duration.

type StatusRecorder struct {
	http.ResponseWriter
	StatusCode int
}

func (r *StatusRecorder) WriteHeader(code int) {
	r.StatusCode = code
	r.ResponseWriter.WriteHeader(code)
}

func LoggingMiddleware(next http.Handler) http.Handler {
	// TODO: Implement
	// 1. Record the start time
	// 2. Wrap ResponseWriter in StatusRecorder to capture status code
	// 3. Call next.ServeHTTP
	// 4. Print: "[METHOD] /path -> STATUS (DURATIONms)"
	return next // replace this
}

// Test:
func TestExercise2() {
	handler := LoggingMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	}))
	req := httptest.NewRequest("POST", "/api/items", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != 201 {
		fmt.Println("FAIL: expected 201, got", w.Code)
	} else {
		fmt.Println("PASS: Exercise 2")
	}
}

// ========== Exercise 3: In-Memory CRUD Store ==========
// Implement a thread-safe key-value store with HTTP handlers.

type Item struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type Store struct {
	mu    sync.RWMutex
	items map[string]Item
}

func NewStore() *Store {
	return &Store{items: make(map[string]Item)}
}

func (s *Store) Set(item Item) {
	// TODO: Implement (thread-safe write)
}

func (s *Store) Get(id string) (Item, bool) {
	// TODO: Implement (thread-safe read)
	return Item{}, false
}

func (s *Store) Delete(id string) bool {
	// TODO: Implement (thread-safe delete, return true if existed)
	return false
}

func (s *Store) List() []Item {
	// TODO: Implement (return all items as a slice)
	return nil
}

// Test:
func TestExercise3() {
	store := NewStore()
	store.Set(Item{ID: "1", Name: "Widget"})
	store.Set(Item{ID: "2", Name: "Gadget"})
	item, ok := store.Get("1")
	if !ok || item.Name != "Widget" {
		fmt.Println("FAIL: Get failed")
		return
	}
	deleted := store.Delete("1")
	if !deleted {
		fmt.Println("FAIL: Delete returned false")
		return
	}
	_, ok = store.Get("1")
	if ok {
		fmt.Println("FAIL: item still exists after delete")
		return
	}
	if len(store.List()) != 1 {
		fmt.Println("FAIL: expected 1 item after delete")
		return
	}
	fmt.Println("PASS: Exercise 3")
}

// ========== Exercise 4: Rate Limiter Middleware ==========
// Implement a per-IP token bucket rate limiter.

type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	rate    float64 // tokens per second
	burst   int     // max tokens
}

type bucket struct {
	tokens    float64
	lastCheck time.Time
}

func NewRateLimiter(rate float64, burst int) *RateLimiter {
	return &RateLimiter{
		buckets: make(map[string]*bucket),
		rate:    rate,
		burst:   burst,
	}
}

func (rl *RateLimiter) Allow(ip string) bool {
	// TODO: Implement token bucket algorithm
	// 1. Get or create bucket for IP
	// 2. Refill tokens based on elapsed time (capped at burst)
	// 3. If tokens >= 1, consume one and return true
	// 4. Otherwise return false
	return true // replace this
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	// TODO: Implement
	// Return 429 Too Many Requests if Allow() returns false
	// Use r.RemoteAddr as the IP key
	return next // replace this
}

// Test:
func TestExercise4() {
	rl := NewRateLimiter(1.0, 2) // 1 token/sec, burst of 2
	// Should allow first 2 (burst), deny 3rd
	if !rl.Allow("127.0.0.1") {
		fmt.Println("FAIL: first request denied")
		return
	}
	if !rl.Allow("127.0.0.1") {
		fmt.Println("FAIL: second request denied")
		return
	}
	if rl.Allow("127.0.0.1") {
		fmt.Println("FAIL: third request should be denied")
		return
	}
	fmt.Println("PASS: Exercise 4")
}

// ========== Run All Tests ==========

func main() {
	fmt.Println("=== Go Web Basics Exercises ===")
	TestExercise1()
	TestExercise2()
	TestExercise3()
	TestExercise4()
}

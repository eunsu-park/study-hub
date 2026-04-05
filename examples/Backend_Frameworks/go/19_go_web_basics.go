/*
Go Web Basics — net/http and Basic Routing
Demonstrates: HTTP server, handler functions, middleware, JSON encoding,
              path parameters, and graceful shutdown.

Run: go run 19_go_web_basics.go
Test: curl http://localhost:8080/api/books
*/

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

// --- 1. Data Models ---

type Book struct {
	ID     int    `json:"id"`
	Title  string `json:"title"`
	Author string `json:"author"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Status  int    `json:"status"`
}

var books = []Book{
	{ID: 1, Title: "The Go Programming Language", Author: "Donovan & Kernighan"},
	{ID: 2, Title: "Concurrency in Go", Author: "Katherine Cox-Buday"},
	{ID: 3, Title: "Go Web Programming", Author: "Sau Sheong Chang"},
}

var requestCount atomic.Int64

// --- 2. JSON Helper ---

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// --- 3. Middleware ---

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestCount.Add(1)
		log.Printf("[%s] %s %s", r.Method, r.URL.Path, r.RemoteAddr)
		next.ServeHTTP(w, r)
		log.Printf("[%s] %s completed in %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// --- 4. Handlers ---

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":   "healthy",
		"requests": requestCount.Load(),
	})
}

func handleListBooks(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, books)
}

func handleGetBook(w http.ResponseWriter, r *http.Request) {
	// Manual path parameter extraction (Go 1.22+ has r.PathValue)
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/books/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "missing book id", Status: 400})
		return
	}
	var id int
	fmt.Sscanf(parts[0], "%d", &id)
	for _, b := range books {
		if b.ID == id {
			writeJSON(w, http.StatusOK, b)
			return
		}
	}
	writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "book not found", Status: 404})
}

func handleCreateBook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, ErrorResponse{Error: "method not allowed", Status: 405})
		return
	}
	var b Book
	if err := json.NewDecoder(r.Body).Decode(&b); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON", Status: 400})
		return
	}
	b.ID = len(books) + 1
	books = append(books, b)
	writeJSON(w, http.StatusCreated, b)
}

// --- 5. Router Setup ---

func setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", handleHealth)
	mux.HandleFunc("/api/books", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			handleListBooks(w, r)
		case http.MethodPost:
			handleCreateBook(w, r)
		default:
			writeJSON(w, http.StatusMethodNotAllowed, ErrorResponse{Error: "method not allowed", Status: 405})
		}
	})
	mux.HandleFunc("/api/books/", handleGetBook)
	return mux
}

// --- 6. Graceful Shutdown ---

func main() {
	mux := setupRoutes()
	handler := loggingMiddleware(corsMiddleware(mux))

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Server listening on %s", srv.Addr)
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down gracefully...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Forced shutdown: %v", err)
	}
	log.Println("Server stopped")
}

// 22_microservice.go — Complete microservice example (URL shortener core)
//
// Run: go run 22_microservice.go
// Test: curl -X POST -d '{"url":"https://go.dev"}' localhost:8080/api/shorten

package main

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

type URL struct {
	Code        string    `json:"code"`
	OriginalURL string    `json:"original_url"`
	Clicks      int64     `json:"clicks"`
	CreatedAt   time.Time `json:"created_at"`
}

type Store struct {
	mu   sync.RWMutex
	urls map[string]*URL
}

func NewStore() *Store {
	return &Store{urls: make(map[string]*URL)}
}

func (s *Store) Create(originalURL string) *URL {
	s.mu.Lock()
	defer s.mu.Unlock()
	code := generateCode(6)
	u := &URL{Code: code, OriginalURL: originalURL, CreatedAt: time.Now()}
	s.urls[code] = u
	return u
}

func (s *Store) Get(code string) (*URL, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.urls[code]
	return u, ok
}

func (s *Store) IncrClicks(code string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if u, ok := s.urls[code]; ok {
		u.Clicks++
	}
}

func generateCode(length int) string {
	b := make([]byte, length)
	rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)[:length]
}

func main() {
	store := NewStore()
	mux := http.NewServeMux()

	mux.HandleFunc("POST /api/shorten", func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil || input.URL == "" {
			http.Error(w, `{"error":"url required"}`, 400)
			return
		}
		u := store.Create(input.URL)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(201)
		json.NewEncoder(w).Encode(u)
	})

	mux.HandleFunc("GET /{code}", func(w http.ResponseWriter, r *http.Request) {
		code := r.PathValue("code")
		u, ok := store.Get(code)
		if !ok {
			http.NotFound(w, r)
			return
		}
		store.IncrClicks(code)
		http.Redirect(w, r, u.OriginalURL, http.StatusMovedPermanently)
	})

	mux.HandleFunc("GET /api/urls/{code}", func(w http.ResponseWriter, r *http.Request) {
		code := r.PathValue("code")
		u, ok := store.Get(code)
		if !ok {
			http.Error(w, `{"error":"not found"}`, 404)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(u)
	})

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"status":"ok"}`)
	})

	handler := loggingMiddleware(mux)
	log.Println("URL Shortener on :8080")
	log.Fatal(http.ListenAndServe(":8080", handler))
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

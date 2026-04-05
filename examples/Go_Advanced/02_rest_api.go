// 13_rest_api.go — REST API with in-memory CRUD
//
// Run: go run 13_rest_api.go
// Test: curl -X POST -d '{"name":"Alice","email":"a@b.com"}' localhost:8080/api/users

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

type User struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	CreatedAt time.Time `json:"created_at"`
}

var (
	users  = make(map[string]*User)
	mu     sync.RWMutex
	nextID = 1
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/users", listUsers)
	mux.HandleFunc("POST /api/users", createUser)
	mux.HandleFunc("GET /api/users/{id}", getUser)
	mux.HandleFunc("DELETE /api/users/{id}", deleteUser)

	log.Println("REST API server on :8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}

func listUsers(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	defer mu.RUnlock()
	var list []*User
	for _, u := range users {
		list = append(list, u)
	}
	writeJSON(w, 200, list)
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Name  string `json:"name"`
		Email string `json:"email"`
	}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		writeJSON(w, 400, map[string]string{"error": "invalid json"})
		return
	}
	mu.Lock()
	defer mu.Unlock()
	id := fmt.Sprintf("%d", nextID)
	nextID++
	u := &User{ID: id, Name: input.Name, Email: input.Email, CreatedAt: time.Now()}
	users[id] = u
	writeJSON(w, 201, u)
}

func getUser(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	mu.RLock()
	defer mu.RUnlock()
	u, ok := users[id]
	if !ok {
		writeJSON(w, 404, map[string]string{"error": "not found"})
		return
	}
	writeJSON(w, 200, u)
}

func deleteUser(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	mu.Lock()
	defer mu.Unlock()
	if _, ok := users[id]; !ok {
		writeJSON(w, 404, map[string]string{"error": "not found"})
		return
	}
	delete(users, id)
	w.WriteHeader(204)
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

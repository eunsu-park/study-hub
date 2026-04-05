// 14_database.go — Database access patterns (uses in-memory simulation)
//
// Run: go run 14_database.go

package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type User struct {
	ID        int
	Name      string
	Email     string
	CreatedAt time.Time
}

// Simulated database
type MemDB struct {
	mu     sync.RWMutex
	users  map[int]*User
	nextID int
}

func NewMemDB() *MemDB {
	return &MemDB{users: make(map[int]*User), nextID: 1}
}

func (db *MemDB) Insert(ctx context.Context, name, email string) (*User, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	u := &User{ID: db.nextID, Name: name, Email: email, CreatedAt: time.Now()}
	db.users[db.nextID] = u
	db.nextID++
	return u, nil
}

func (db *MemDB) FindByID(ctx context.Context, id int) (*User, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	u, ok := db.users[id]
	if !ok {
		return nil, fmt.Errorf("user %d: not found", id)
	}
	return u, nil
}

func (db *MemDB) List(ctx context.Context, limit, offset int) ([]*User, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	var result []*User
	for _, u := range db.users {
		result = append(result, u)
	}
	if offset >= len(result) {
		return nil, nil
	}
	end := offset + limit
	if end > len(result) {
		end = len(result)
	}
	return result[offset:end], nil
}

func main() {
	ctx := context.Background()
	db := NewMemDB()

	fmt.Println("=== Database Patterns ===")

	// Insert
	u1, _ := db.Insert(ctx, "Alice", "alice@example.com")
	u2, _ := db.Insert(ctx, "Bob", "bob@example.com")
	db.Insert(ctx, "Carol", "carol@example.com")
	fmt.Printf("Inserted: %+v\n", u1)
	fmt.Printf("Inserted: %+v\n", u2)

	// Find by ID
	found, err := db.FindByID(ctx, 1)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("Found: %+v\n", found)
	}

	// List with pagination
	users, _ := db.List(ctx, 2, 0)
	fmt.Println("\nPage 1:")
	for _, u := range users {
		fmt.Printf("  %d: %s <%s>\n", u.ID, u.Name, u.Email)
	}
}

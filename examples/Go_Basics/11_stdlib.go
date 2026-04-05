// 11_stdlib.go — Standard library: io, json, os, time
//
// Run: go run 11_stdlib.go

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type User struct {
	ID        int       `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

func main() {
	fmt.Println("=== io Package ===")
	ioDemo()

	fmt.Println("\n=== JSON ===")
	jsonDemo()

	fmt.Println("\n=== Time ===")
	timeDemo()

	fmt.Println("\n=== filepath ===")
	filepathDemo()
}

func ioDemo() {
	r := strings.NewReader("Hello, io!")
	io.Copy(os.Stdout, r)
	fmt.Println()

	var buf bytes.Buffer
	buf.WriteString("Hello ")
	buf.WriteString("Buffer!")
	fmt.Println(buf.String())

	data, _ := io.ReadAll(strings.NewReader("read all"))
	fmt.Println(string(data))
}

func jsonDemo() {
	user := User{
		ID:        1,
		Name:      "Alice",
		Email:     "alice@example.com",
		CreatedAt: time.Now(),
	}

	data, _ := json.MarshalIndent(user, "", "  ")
	fmt.Println(string(data))

	jsonStr := `{"id": 2, "name": "Bob", "created_at": "2024-01-15T10:30:00Z"}`
	var user2 User
	json.Unmarshal([]byte(jsonStr), &user2)
	fmt.Printf("Parsed: %+v\n", user2)
}

func timeDemo() {
	now := time.Now()
	fmt.Println("Now:", now.Format("2006-01-02 15:04:05"))
	fmt.Println("RFC3339:", now.Format(time.RFC3339))

	d := 2*time.Hour + 30*time.Minute
	future := now.Add(d)
	fmt.Println("Future:", future.Format("15:04:05"))

	parsed, _ := time.Parse("2006-01-02", "2024-03-15")
	fmt.Println("Parsed:", parsed)
}

func filepathDemo() {
	p := filepath.Join("home", "user", "documents", "file.txt")
	fmt.Println("Path:", p)
	fmt.Println("Dir:", filepath.Dir(p))
	fmt.Println("Base:", filepath.Base(p))
	fmt.Println("Ext:", filepath.Ext(p))
	fmt.Println("Clean:", filepath.Clean("a/b/../c"))
}

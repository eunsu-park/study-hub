// 05_error_handling.go — Error types, wrapping, sentinel errors
//
// Run: go run 05_error_handling.go

package main

import (
	"errors"
	"fmt"
)

var (
	ErrNotFound     = errors.New("not found")
	ErrUnauthorized = errors.New("unauthorized")
)

type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation: %s — %s", e.Field, e.Message)
}

func findUser(id int) (string, error) {
	if id <= 0 {
		return "", &ValidationError{Field: "id", Message: "must be positive"}
	}
	if id > 100 {
		return "", fmt.Errorf("findUser(%d): %w", id, ErrNotFound)
	}
	return fmt.Sprintf("User-%d", id), nil
}

func main() {
	fmt.Println("=== Error Handling Patterns ===")

	ids := []int{-1, 50, 200}
	for _, id := range ids {
		name, err := findUser(id)
		if err != nil {
			var valErr *ValidationError
			switch {
			case errors.As(err, &valErr):
				fmt.Printf("ID %d: Validation error on %s: %s\n", id, valErr.Field, valErr.Message)
			case errors.Is(err, ErrNotFound):
				fmt.Printf("ID %d: Not found\n", id)
			default:
				fmt.Printf("ID %d: Unexpected error: %v\n", id, err)
			}
		} else {
			fmt.Printf("ID %d: Found %s\n", id, name)
		}
	}

	fmt.Println("\n=== Error Wrapping ===")
	_, err := findUser(999)
	fmt.Println("Error:", err)
	fmt.Println("Is ErrNotFound:", errors.Is(err, ErrNotFound))

	wrapped := fmt.Errorf("service: %w", fmt.Errorf("repo: %w", ErrNotFound))
	fmt.Println("Deep wrap:", wrapped)
	fmt.Println("Is ErrNotFound:", errors.Is(wrapped, ErrNotFound))
}

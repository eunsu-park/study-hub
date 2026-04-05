# 13. REST API

**Previous**: [HTTP Server](./01_HTTP_Server.md) | **Next**: [Database Access](./03_Database_Access.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design RESTful API endpoints following HTTP conventions
2. Handle JSON request/response encoding with validation
3. Implement CRUD operations with proper status codes
4. Add pagination, filtering, and error responses
5. Structure API code for maintainability

---

Building REST APIs is one of Go's most common use cases. This lesson covers idiomatic patterns for request handling, validation, error responses, and API structure.

## Table of Contents
1. [REST API Design](#1-rest-api-design)
2. [Request Handling](#2-request-handling)
3. [Response Patterns](#3-response-patterns)
4. [CRUD Implementation](#4-crud-implementation)
5. [Validation](#5-validation)
6. [Pagination and Filtering](#6-pagination-and-filtering)
7. [Summary](#7-summary)

---

## 1. REST API Design

### 1.1 Resource-Based URLs

```go
// Standard REST conventions:
// GET    /api/users          — List users
// POST   /api/users          — Create user
// GET    /api/users/{id}     — Get user
// PUT    /api/users/{id}     — Update user (full)
// PATCH  /api/users/{id}     — Update user (partial)
// DELETE /api/users/{id}     — Delete user

func main() {
    mux := http.NewServeMux()

    api := &API{store: NewInMemoryStore()}

    mux.HandleFunc("GET /api/users", api.ListUsers)
    mux.HandleFunc("POST /api/users", api.CreateUser)
    mux.HandleFunc("GET /api/users/{id}", api.GetUser)
    mux.HandleFunc("PUT /api/users/{id}", api.UpdateUser)
    mux.HandleFunc("DELETE /api/users/{id}", api.DeleteUser)

    // Nested resources
    mux.HandleFunc("GET /api/users/{userID}/posts", api.ListUserPosts)
    mux.HandleFunc("POST /api/users/{userID}/posts", api.CreateUserPost)

    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### 1.2 API Structure

```go
type API struct {
    store  Store
    logger *slog.Logger
}

type Store interface {
    ListUsers(ctx context.Context, opts ListOptions) ([]User, int, error)
    GetUser(ctx context.Context, id string) (*User, error)
    CreateUser(ctx context.Context, u *User) error
    UpdateUser(ctx context.Context, id string, u *User) error
    DeleteUser(ctx context.Context, id string) error
}

type User struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}
```

---

## 2. Request Handling

### 2.1 JSON Decoding with Limits

```go
func decodeJSON[T any](r *http.Request) (T, error) {
    var v T
    // Limit body size to prevent abuse
    r.Body = http.MaxBytesReader(nil, r.Body, 1<<20) // 1 MB

    decoder := json.NewDecoder(r.Body)
    decoder.DisallowUnknownFields() // Strict parsing

    if err := decoder.Decode(&v); err != nil {
        return v, fmt.Errorf("decode json: %w", err)
    }

    // Check for extra data
    if decoder.More() {
        return v, fmt.Errorf("body must contain a single JSON object")
    }
    return v, nil
}

func (api *API) CreateUser(w http.ResponseWriter, r *http.Request) {
    var input struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }

    input, err := decodeJSON[struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }](r)
    if err != nil {
        writeError(w, http.StatusBadRequest, err.Error())
        return
    }

    user := &User{
        ID:        generateID(),
        Name:      input.Name,
        Email:     input.Email,
        CreatedAt: time.Now(),
    }

    if err := api.store.CreateUser(r.Context(), user); err != nil {
        writeError(w, http.StatusInternalServerError, "failed to create user")
        return
    }

    writeJSON(w, http.StatusCreated, user)
}
```

### 2.2 Path and Query Parameters

```go
func (api *API) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id") // Go 1.22+

    user, err := api.store.GetUser(r.Context(), id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            writeError(w, http.StatusNotFound, "user not found")
            return
        }
        writeError(w, http.StatusInternalServerError, "internal error")
        return
    }

    writeJSON(w, http.StatusOK, user)
}

func (api *API) ListUsers(w http.ResponseWriter, r *http.Request) {
    q := r.URL.Query()
    page, _ := strconv.Atoi(q.Get("page"))
    if page < 1 {
        page = 1
    }
    perPage, _ := strconv.Atoi(q.Get("per_page"))
    if perPage < 1 || perPage > 100 {
        perPage = 20
    }
    search := q.Get("search")
    sortBy := q.Get("sort")

    opts := ListOptions{
        Page:    page,
        PerPage: perPage,
        Search:  search,
        SortBy:  sortBy,
    }

    users, total, err := api.store.ListUsers(r.Context(), opts)
    if err != nil {
        writeError(w, http.StatusInternalServerError, "failed to list users")
        return
    }

    writeJSON(w, http.StatusOK, PagedResponse{
        Data:    users,
        Total:   total,
        Page:    page,
        PerPage: perPage,
    })
}
```

---

## 3. Response Patterns

### 3.1 Standard Response Helpers

```go
type ErrorResponse struct {
    Error   string            `json:"error"`
    Details map[string]string `json:"details,omitempty"`
}

type PagedResponse struct {
    Data    any `json:"data"`
    Total   int `json:"total"`
    Page    int `json:"page"`
    PerPage int `json:"per_page"`
}

func writeJSON(w http.ResponseWriter, status int, data any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    if err := json.NewEncoder(w).Encode(data); err != nil {
        slog.Error("failed to encode response", "err", err)
    }
}

func writeError(w http.ResponseWriter, status int, message string) {
    writeJSON(w, status, ErrorResponse{Error: message})
}

func writeErrorWithDetails(w http.ResponseWriter, status int, message string, details map[string]string) {
    writeJSON(w, status, ErrorResponse{Error: message, Details: details})
}
```

### 3.2 Status Code Guide

```go
// 2xx Success
// 200 OK          — GET, PUT, PATCH success
// 201 Created     — POST success (include Location header)
// 204 No Content  — DELETE success (no body)

// 4xx Client Error
// 400 Bad Request     — Invalid input, malformed JSON
// 401 Unauthorized    — Not authenticated
// 403 Forbidden       — Authenticated but not authorized
// 404 Not Found       — Resource doesn't exist
// 409 Conflict        — Duplicate resource
// 422 Unprocessable   — Valid JSON but fails validation

// 5xx Server Error
// 500 Internal Server Error — Unexpected failure
// 503 Service Unavailable   — Temporary overload
```

---

## 4. CRUD Implementation

### 4.1 In-Memory Store

```go
type InMemoryStore struct {
    mu    sync.RWMutex
    users map[string]*User
}

func NewInMemoryStore() *InMemoryStore {
    return &InMemoryStore{users: make(map[string]*User)}
}

func (s *InMemoryStore) GetUser(ctx context.Context, id string) (*User, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    user, ok := s.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return user, nil
}

func (s *InMemoryStore) CreateUser(ctx context.Context, u *User) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    if _, exists := s.users[u.ID]; exists {
        return ErrConflict
    }
    s.users[u.ID] = u
    return nil
}

func (s *InMemoryStore) UpdateUser(ctx context.Context, id string, u *User) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    if _, exists := s.users[id]; !exists {
        return ErrNotFound
    }
    u.ID = id
    u.UpdatedAt = time.Now()
    s.users[id] = u
    return nil
}

func (s *InMemoryStore) DeleteUser(ctx context.Context, id string) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    if _, exists := s.users[id]; !exists {
        return ErrNotFound
    }
    delete(s.users, id)
    return nil
}

func (s *InMemoryStore) ListUsers(ctx context.Context, opts ListOptions) ([]User, int, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    var users []User
    for _, u := range s.users {
        if opts.Search != "" && !strings.Contains(strings.ToLower(u.Name), strings.ToLower(opts.Search)) {
            continue
        }
        users = append(users, *u)
    }

    total := len(users)
    start := (opts.Page - 1) * opts.PerPage
    end := start + opts.PerPage
    if start > len(users) {
        return nil, total, nil
    }
    if end > len(users) {
        end = len(users)
    }

    return users[start:end], total, nil
}
```

---

## 5. Validation

### 5.1 Manual Validation

```go
type CreateUserInput struct {
    Name  string `json:"name"`
    Email string `json:"email"`
    Age   int    `json:"age"`
}

func (input *CreateUserInput) Validate() map[string]string {
    errors := make(map[string]string)

    if strings.TrimSpace(input.Name) == "" {
        errors["name"] = "name is required"
    } else if len(input.Name) > 100 {
        errors["name"] = "name must be 100 characters or less"
    }

    if input.Email == "" {
        errors["email"] = "email is required"
    } else if !isValidEmail(input.Email) {
        errors["email"] = "email is invalid"
    }

    if input.Age < 0 || input.Age > 150 {
        errors["age"] = "age must be between 0 and 150"
    }

    if len(errors) > 0 {
        return errors
    }
    return nil
}

var emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)

func isValidEmail(email string) bool {
    return emailRegex.MatchString(email)
}

func (api *API) CreateUser(w http.ResponseWriter, r *http.Request) {
    var input CreateUserInput
    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        writeError(w, http.StatusBadRequest, "invalid JSON")
        return
    }

    if errs := input.Validate(); errs != nil {
        writeErrorWithDetails(w, http.StatusUnprocessableEntity, "validation failed", errs)
        return
    }

    // Create user...
}
```

---

## 6. Pagination and Filtering

### 6.1 Pagination

```go
type ListOptions struct {
    Page    int
    PerPage int
    Search  string
    SortBy  string
    Order   string // "asc" or "desc"
}

type PagedResponse struct {
    Data       any  `json:"data"`
    Total      int  `json:"total"`
    Page       int  `json:"page"`
    PerPage    int  `json:"per_page"`
    TotalPages int  `json:"total_pages"`
    HasNext    bool `json:"has_next"`
    HasPrev    bool `json:"has_prev"`
}

func NewPagedResponse(data any, total, page, perPage int) PagedResponse {
    totalPages := (total + perPage - 1) / perPage
    return PagedResponse{
        Data:       data,
        Total:      total,
        Page:       page,
        PerPage:    perPage,
        TotalPages: totalPages,
        HasNext:    page < totalPages,
        HasPrev:    page > 1,
    }
}
```

### 6.2 Cursor-Based Pagination

```go
type CursorResponse struct {
    Data       any    `json:"data"`
    NextCursor string `json:"next_cursor,omitempty"`
    HasMore    bool   `json:"has_more"`
}

func (api *API) ListUsersWithCursor(w http.ResponseWriter, r *http.Request) {
    cursor := r.URL.Query().Get("cursor")
    limit := 20

    users, nextCursor, err := api.store.ListUsersAfterCursor(r.Context(), cursor, limit+1)
    if err != nil {
        writeError(w, http.StatusInternalServerError, "failed to list users")
        return
    }

    hasMore := len(users) > limit
    if hasMore {
        users = users[:limit]
    }

    writeJSON(w, http.StatusOK, CursorResponse{
        Data:       users,
        NextCursor: nextCursor,
        HasMore:    hasMore,
    })
}
```

---

## 7. Summary

### Key Takeaways

1. **Resource-based URLs** — nouns not verbs. HTTP methods define the action.
2. **Consistent response format** — always return JSON with proper Content-Type and status codes.
3. **Validate input strictly** — use `MaxBytesReader`, `DisallowUnknownFields`, custom validation.
4. **Error responses are structured** — include error message and field-level details.
5. **Pagination is essential** — offset-based for simple cases, cursor-based for large datasets.
6. **Separate concerns** — handler (HTTP), service (business logic), store (persistence).

---

## Exercises

### Exercise 1: Todo API
Build a complete TODO API with CRUD operations, status filtering, due date sorting, and pagination.

### Exercise 2: Validation Framework
Create a reusable validation framework with rules like `required`, `min_length`, `max_length`, `email`, `regex`.

### Exercise 3: API Versioning
Implement API versioning via URL path (`/v1/users`, `/v2/users`) with different response formats.

### Exercise 4: Batch Operations
Add batch endpoints: `POST /api/users/batch` (create many) and `DELETE /api/users/batch` (delete many) with partial success handling.

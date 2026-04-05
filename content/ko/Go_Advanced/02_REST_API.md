# 13. REST API

**이전**: [HTTP 서버](./01_HTTP_Server.md) | **다음**: [데이터베이스 접근](./03_Database_Access.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. HTTP 규칙을 따르는 RESTful API 엔드포인트를 설계한다
2. 유효성 검사와 함께 JSON 요청/응답 인코딩을 처리한다
3. 적절한 상태 코드로 CRUD 연산을 구현한다
4. 페이지네이션, 필터링, 에러 응답을 추가한다
5. 유지보수성을 위해 API 코드를 구조화한다

---

REST API 구축은 Go의 가장 일반적인 사용 사례 중 하나이다. 이 레슨은 요청 처리, 유효성 검사, 에러 응답, API 구조에 대한 관용적 패턴을 다룬다.

## 목차
1. [REST API 설계](#1-rest-api-설계)
2. [요청 처리](#2-요청-처리)
3. [응답 패턴](#3-응답-패턴)
4. [CRUD 구현](#4-crud-구현)
5. [유효성 검사](#5-유효성-검사)
6. [페이지네이션과 필터링](#6-페이지네이션과-필터링)
7. [요약](#7-요약)

---

## 1. REST API 설계

### 1.1 리소스 기반 URL

```go
// 표준 REST 규칙:
// GET    /api/users          — 사용자 목록
// POST   /api/users          — 사용자 생성
// GET    /api/users/{id}     — 사용자 조회
// PUT    /api/users/{id}     — 사용자 수정 (전체)
// PATCH  /api/users/{id}     — 사용자 수정 (부분)
// DELETE /api/users/{id}     — 사용자 삭제

func main() {
    mux := http.NewServeMux()

    api := &API{store: NewInMemoryStore()}

    mux.HandleFunc("GET /api/users", api.ListUsers)
    mux.HandleFunc("POST /api/users", api.CreateUser)
    mux.HandleFunc("GET /api/users/{id}", api.GetUser)
    mux.HandleFunc("PUT /api/users/{id}", api.UpdateUser)
    mux.HandleFunc("DELETE /api/users/{id}", api.DeleteUser)

    // 중첩 리소스
    mux.HandleFunc("GET /api/users/{userID}/posts", api.ListUserPosts)
    mux.HandleFunc("POST /api/users/{userID}/posts", api.CreateUserPost)

    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### 1.2 API 구조

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

## 2. 요청 처리

### 2.1 제한이 있는 JSON 디코딩

```go
func decodeJSON[T any](r *http.Request) (T, error) {
    var v T
    // 남용을 방지하기 위해 바디 크기 제한
    r.Body = http.MaxBytesReader(nil, r.Body, 1<<20) // 1 MB

    decoder := json.NewDecoder(r.Body)
    decoder.DisallowUnknownFields() // 엄격한 파싱

    if err := decoder.Decode(&v); err != nil {
        return v, fmt.Errorf("decode json: %w", err)
    }

    // 추가 데이터 확인
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

### 2.2 경로와 쿼리 매개변수

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

## 3. 응답 패턴

### 3.1 표준 응답 헬퍼

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

### 3.2 상태 코드 가이드

```go
// 2xx 성공
// 200 OK          — GET, PUT, PATCH 성공
// 201 Created     — POST 성공 (Location 헤더 포함)
// 204 No Content  — DELETE 성공 (바디 없음)

// 4xx 클라이언트 에러
// 400 Bad Request     — 유효하지 않은 입력, 잘못된 JSON
// 401 Unauthorized    — 인증되지 않음
// 403 Forbidden       — 인증되었지만 권한 없음
// 404 Not Found       — 리소스가 존재하지 않음
// 409 Conflict        — 중복 리소스
// 422 Unprocessable   — 유효한 JSON이지만 유효성 검사 실패

// 5xx 서버 에러
// 500 Internal Server Error — 예상치 못한 실패
// 503 Service Unavailable   — 일시적 과부하
```

---

## 4. CRUD 구현

### 4.1 인메모리 저장소

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

## 5. 유효성 검사

### 5.1 수동 유효성 검사

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

    // 사용자 생성...
}
```

---

## 6. 페이지네이션과 필터링

### 6.1 페이지네이션

```go
type ListOptions struct {
    Page    int
    PerPage int
    Search  string
    SortBy  string
    Order   string // "asc" 또는 "desc"
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

### 6.2 커서 기반 페이지네이션

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

## 7. 요약

### 핵심 포인트

1. **리소스 기반 URL** — 동사가 아니라 명사를 사용한다. HTTP 메서드가 동작을 정의한다.
2. **일관된 응답 형식** — 항상 적절한 Content-Type과 상태 코드로 JSON을 반환한다.
3. **입력을 엄격하게 검증한다** — `MaxBytesReader`, `DisallowUnknownFields`, 커스텀 유효성 검사를 사용한다.
4. **에러 응답은 구조화한다** — 에러 메시지와 필드 수준 상세 정보를 포함한다.
5. **페이지네이션은 필수이다** — 단순한 경우에는 오프셋 기반, 대규모 데이터셋에는 커서 기반을 사용한다.
6. **관심사를 분리한다** — 핸들러(HTTP), 서비스(비즈니스 로직), 저장소(영속성)로 나눈다.

---

## 연습 문제

### 연습 1: Todo API
CRUD 연산, 상태 필터링, 마감일 정렬, 페이지네이션이 포함된 완전한 TODO API를 구축한다.

### 연습 2: 유효성 검사 프레임워크
`required`, `min_length`, `max_length`, `email`, `regex` 같은 규칙이 포함된 재사용 가능한 유효성 검사 프레임워크를 생성한다.

### 연습 3: API 버전 관리
다른 응답 형식을 가진 URL 경로(`/v1/users`, `/v2/users`)를 통한 API 버전 관리를 구현한다.

### 연습 4: 배치 연산
배치 엔드포인트를 추가한다: `POST /api/users/batch` (다수 생성)와 `DELETE /api/users/batch` (다수 삭제)에 부분 성공 처리를 포함한다.

# 14. 데이터베이스 접근

**이전**: [REST API](./02_REST_API.md) | **다음**: [CLI 도구](./04_CLI_Tools.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 적절한 커넥션 풀링으로 `database/sql`을 사용하여 데이터베이스 연산을 수행한다
2. 준비된 구문과 매개변수화된 쿼리로 안전하게 쿼리를 실행한다
3. 트랜잭션을 처리하고 레포지토리 패턴을 구현한다
4. 보일러플레이트를 줄이기 위해 `sqlx`를 사용한다
5. 데이터베이스 마이그레이션을 관리한다

---

Go의 `database/sql` 패키지는 SQL 데이터베이스를 위한 범용 인터페이스를 제공한다. 드라이버(예: PostgreSQL용 `lib/pq`, MySQL용 `go-sql-driver/mysql`)와 결합하여 커넥션 풀링, 준비된 구문, 트랜잭션을 처리한다.

## 목차
1. [database/sql 기초](#1-databasesql-기초)
2. [CRUD 연산](#2-crud-연산)
3. [트랜잭션](#3-트랜잭션)
4. [커넥션 풀 구성](#4-커넥션-풀-구성)
5. [보일러플레이트를 줄이기 위한 sqlx](#5-보일러플레이트를-줄이기-위한-sqlx)
6. [마이그레이션과 레포지토리 패턴](#6-마이그레이션과-레포지토리-패턴)
7. [요약](#7-요약)

---

## 1. database/sql 기초

### 1.1 데이터베이스에 연결

```go
package main

import (
    "database/sql"
    "fmt"
    "log"

    _ "github.com/lib/pq" // PostgreSQL 드라이버 (부작용을 위한 빈 import)
)

func main() {
    connStr := "host=localhost port=5432 user=postgres password=secret dbname=myapp sslmode=disable"

    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatal("open:", err)
    }
    defer db.Close()

    // sql.Open은 실제로 연결하지 않음 — Ping이 연결함
    if err := db.Ping(); err != nil {
        log.Fatal("ping:", err)
    }

    fmt.Println("Connected to database!")
}
```

### 1.2 테이블 생성

```go
func createTables(db *sql.DB) error {
    query := `
    CREATE TABLE IF NOT EXISTS users (
        id          SERIAL PRIMARY KEY,
        name        VARCHAR(100) NOT NULL,
        email       VARCHAR(255) UNIQUE NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS posts (
        id          SERIAL PRIMARY KEY,
        user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
        title       VARCHAR(255) NOT NULL,
        body        TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    `
    _, err := db.Exec(query)
    return err
}
```

---

## 2. CRUD 연산

### 2.1 삽입

```go
type User struct {
    ID        int
    Name      string
    Email     string
    CreatedAt time.Time
}

func insertUser(db *sql.DB, name, email string) (int, error) {
    var id int
    err := db.QueryRow(
        `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id`,
        name, email,
    ).Scan(&id)
    return id, err
}
```

### 2.2 단일 행 쿼리

```go
func getUserByID(db *sql.DB, id int) (*User, error) {
    user := &User{}
    err := db.QueryRow(
        `SELECT id, name, email, created_at FROM users WHERE id = $1`,
        id,
    ).Scan(&user.ID, &user.Name, &user.Email, &user.CreatedAt)

    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("user %d: %w", id, ErrNotFound)
    }
    if err != nil {
        return nil, fmt.Errorf("query user %d: %w", id, err)
    }
    return user, nil
}
```

### 2.3 다중 행 쿼리

```go
func listUsers(db *sql.DB, limit, offset int) ([]User, error) {
    rows, err := db.Query(
        `SELECT id, name, email, created_at FROM users ORDER BY id LIMIT $1 OFFSET $2`,
        limit, offset,
    )
    if err != nil {
        return nil, err
    }
    defer rows.Close() // 항상 rows를 닫아야 함

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt); err != nil {
            return nil, err
        }
        users = append(users, u)
    }

    // 반복에서 발생한 에러 확인
    if err := rows.Err(); err != nil {
        return nil, err
    }
    return users, nil
}
```

### 2.4 수정과 삭제

```go
func updateUser(db *sql.DB, id int, name, email string) error {
    result, err := db.Exec(
        `UPDATE users SET name = $1, email = $2, updated_at = NOW() WHERE id = $3`,
        name, email, id,
    )
    if err != nil {
        return err
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    if rowsAffected == 0 {
        return ErrNotFound
    }
    return nil
}

func deleteUser(db *sql.DB, id int) error {
    result, err := db.Exec(`DELETE FROM users WHERE id = $1`, id)
    if err != nil {
        return err
    }

    rowsAffected, _ := result.RowsAffected()
    if rowsAffected == 0 {
        return ErrNotFound
    }
    return nil
}
```

---

## 3. 트랜잭션

### 3.1 기본 트랜잭션

```go
func transferFunds(db *sql.DB, fromID, toID int, amount float64) error {
    tx, err := db.Begin()
    if err != nil {
        return fmt.Errorf("begin tx: %w", err)
    }
    defer tx.Rollback() // 커밋되지 않으면 롤백 (커밋 후에는 no-op)

    // 출금
    result, err := tx.Exec(
        `UPDATE accounts SET balance = balance - $1 WHERE id = $2 AND balance >= $1`,
        amount, fromID,
    )
    if err != nil {
        return fmt.Errorf("debit: %w", err)
    }
    rows, _ := result.RowsAffected()
    if rows == 0 {
        return fmt.Errorf("insufficient funds or account not found")
    }

    // 입금
    _, err = tx.Exec(
        `UPDATE accounts SET balance = balance + $1 WHERE id = $2`,
        amount, toID,
    )
    if err != nil {
        return fmt.Errorf("credit: %w", err)
    }

    return tx.Commit()
}
```

### 3.2 트랜잭션 헬퍼

```go
func withTransaction(db *sql.DB, fn func(tx *sql.Tx) error) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }

    defer func() {
        if p := recover(); p != nil {
            tx.Rollback()
            panic(p) // 롤백 후 재패닉
        }
    }()

    if err := fn(tx); err != nil {
        tx.Rollback()
        return err
    }

    return tx.Commit()
}

// 사용법
func createUserWithPosts(db *sql.DB, user *User, posts []Post) error {
    return withTransaction(db, func(tx *sql.Tx) error {
        var userID int
        err := tx.QueryRow(
            `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id`,
            user.Name, user.Email,
        ).Scan(&userID)
        if err != nil {
            return err
        }

        for _, post := range posts {
            _, err := tx.Exec(
                `INSERT INTO posts (user_id, title, body) VALUES ($1, $2, $3)`,
                userID, post.Title, post.Body,
            )
            if err != nil {
                return err
            }
        }
        return nil
    })
}
```

---

## 4. 커넥션 풀 구성

```go
func setupDB(connStr string) (*sql.DB, error) {
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        return nil, err
    }

    // 풀 구성
    db.SetMaxOpenConns(25)                  // 최대 열린 연결 수
    db.SetMaxIdleConns(5)                   // 최대 유휴 연결 수
    db.SetConnMaxLifetime(5 * time.Minute)  // 최대 연결 수명
    db.SetConnMaxIdleTime(1 * time.Minute)  // 최대 유휴 시간

    // 연결 확인
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        db.Close()
        return nil, fmt.Errorf("ping: %w", err)
    }

    return db, nil
}
```

---

## 5. 보일러플레이트를 줄이기 위한 sqlx

```go
import "github.com/jmoiron/sqlx"

type User struct {
    ID        int       `db:"id"`
    Name      string    `db:"name"`
    Email     string    `db:"email"`
    CreatedAt time.Time `db:"created_at"`
}

func main() {
    db, err := sqlx.Connect("postgres", connStr)
    if err != nil {
        log.Fatal(err)
    }

    // 단일 행을 구조체로 가져오기
    var user User
    err = db.Get(&user, `SELECT * FROM users WHERE id = $1`, 1)

    // 다중 행을 슬라이스로 선택
    var users []User
    err = db.Select(&users, `SELECT * FROM users ORDER BY id LIMIT 10`)

    // 이름 있는 쿼리
    _, err = db.NamedExec(
        `INSERT INTO users (name, email) VALUES (:name, :email)`,
        User{Name: "Alice", Email: "alice@example.com"},
    )

    // map을 사용한 이름 있는 쿼리
    rows, err := db.NamedQuery(
        `SELECT * FROM users WHERE name = :name`,
        map[string]any{"name": "Alice"},
    )
    defer rows.Close()
}
```

---

## 6. 마이그레이션과 레포지토리 패턴

### 6.1 간단한 마이그레이션 시스템

```go
type Migration struct {
    Version int
    Name    string
    Up      string
    Down    string
}

var migrations = []Migration{
    {
        Version: 1,
        Name:    "create_users",
        Up: `CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )`,
        Down: `DROP TABLE users`,
    },
    {
        Version: 2,
        Name:    "add_user_role",
        Up:      `ALTER TABLE users ADD COLUMN role VARCHAR(50) DEFAULT 'user'`,
        Down:    `ALTER TABLE users DROP COLUMN role`,
    },
}

func runMigrations(db *sql.DB) error {
    db.Exec(`CREATE TABLE IF NOT EXISTS schema_migrations (version INT PRIMARY KEY)`)

    for _, m := range migrations {
        var exists bool
        db.QueryRow(`SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version = $1)`, m.Version).Scan(&exists)
        if exists {
            continue
        }

        log.Printf("Running migration %d: %s", m.Version, m.Name)
        if _, err := db.Exec(m.Up); err != nil {
            return fmt.Errorf("migration %d: %w", m.Version, err)
        }
        db.Exec(`INSERT INTO schema_migrations (version) VALUES ($1)`, m.Version)
    }
    return nil
}
```

### 6.2 레포지토리 패턴

```go
type UserRepository struct {
    db *sql.DB
}

func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) FindByID(ctx context.Context, id int) (*User, error) {
    user := &User{}
    err := r.db.QueryRowContext(ctx,
        `SELECT id, name, email, created_at FROM users WHERE id = $1`, id,
    ).Scan(&user.ID, &user.Name, &user.Email, &user.CreatedAt)

    if errors.Is(err, sql.ErrNoRows) {
        return nil, ErrNotFound
    }
    return user, err
}

func (r *UserRepository) Create(ctx context.Context, u *User) error {
    return r.db.QueryRowContext(ctx,
        `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id, created_at`,
        u.Name, u.Email,
    ).Scan(&u.ID, &u.CreatedAt)
}
```

---

## 7. 요약

### 핵심 포인트

1. **`sql.Open`은 연결하지 않는다** — 연결을 확인하려면 `db.Ping()`을 사용한다.
2. **항상 rows를 닫는다** — `defer rows.Close()`가 커넥션 누수를 방지한다.
3. **매개변수화된 쿼리를 사용한다** — `$1, $2` (PostgreSQL) 또는 `?, ?` (MySQL)을 사용한다. SQL 문자열을 절대 연결하지 않는다.
4. **`sql.ErrNoRows`를 확인한다** — "찾을 수 없음"과 실제 에러를 구분한다.
5. **커넥션 풀을 구성한다** — `MaxOpenConns`, `MaxIdleConns`, 타임아웃을 설정한다.
6. **원자성을 위한 트랜잭션** — 안전을 위해 `defer tx.Rollback()`을 사용한다.
7. **context를 사용한다** — 타임아웃/취소 지원을 위해 `QueryRowContext`, `ExecContext`를 사용한다.

---

## 연습 문제

### 연습 1: PostgreSQL을 사용한 User CRUD
검색과 페이지네이션을 포함한 데이터베이스 기반 CRUD가 있는 완전한 사용자 관리 API를 구축한다.

### 연습 2: 트랜잭션 안전성
적절한 트랜잭션으로 은행 이체 시스템을 구현하고, 동시 접근과 롤백을 처리한다.

### 연습 3: 커넥션 풀 튜닝
부하 테스트를 작성하고 다른 풀 설정을 실험한다. 처리량과 지연 시간을 측정한다.

### 연습 4: 마이그레이션 도구
디렉토리에서 `.sql` 파일을 읽고 순서대로 적용하는 독립형 마이그레이션 도구를 구축한다.

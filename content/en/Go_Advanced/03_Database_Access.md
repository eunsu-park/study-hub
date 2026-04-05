# 14. Database Access

**Previous**: [REST API](./02_REST_API.md) | **Next**: [CLI Tools](./04_CLI_Tools.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `database/sql` for database operations with proper connection pooling
2. Execute queries safely with prepared statements and parameterized queries
3. Handle transactions and implement repository patterns
4. Use `sqlx` for reduced boilerplate
5. Manage database migrations

---

Go's `database/sql` package provides a generic interface for SQL databases. Combined with a driver (e.g., `lib/pq` for PostgreSQL, `go-sql-driver/mysql` for MySQL), it handles connection pooling, prepared statements, and transactions.

## Table of Contents
1. [database/sql Basics](#1-databasesql-basics)
2. [CRUD Operations](#2-crud-operations)
3. [Transactions](#3-transactions)
4. [Connection Pool Configuration](#4-connection-pool-configuration)
5. [sqlx for Reduced Boilerplate](#5-sqlx-for-reduced-boilerplate)
6. [Migrations and Repository Pattern](#6-migrations-and-repository-pattern)
7. [Summary](#7-summary)

---

## 1. database/sql Basics

### 1.1 Connecting to a Database

```go
package main

import (
    "database/sql"
    "fmt"
    "log"

    _ "github.com/lib/pq" // PostgreSQL driver (blank import for side effects)
)

func main() {
    connStr := "host=localhost port=5432 user=postgres password=secret dbname=myapp sslmode=disable"

    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatal("open:", err)
    }
    defer db.Close()

    // sql.Open doesn't actually connect — Ping does
    if err := db.Ping(); err != nil {
        log.Fatal("ping:", err)
    }

    fmt.Println("Connected to database!")
}
```

### 1.2 Creating Tables

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

## 2. CRUD Operations

### 2.1 Insert

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

### 2.2 Query Single Row

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

### 2.3 Query Multiple Rows

```go
func listUsers(db *sql.DB, limit, offset int) ([]User, error) {
    rows, err := db.Query(
        `SELECT id, name, email, created_at FROM users ORDER BY id LIMIT $1 OFFSET $2`,
        limit, offset,
    )
    if err != nil {
        return nil, err
    }
    defer rows.Close() // ALWAYS close rows

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt); err != nil {
            return nil, err
        }
        users = append(users, u)
    }

    // Check for errors from iteration
    if err := rows.Err(); err != nil {
        return nil, err
    }
    return users, nil
}
```

### 2.4 Update and Delete

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

## 3. Transactions

### 3.1 Basic Transaction

```go
func transferFunds(db *sql.DB, fromID, toID int, amount float64) error {
    tx, err := db.Begin()
    if err != nil {
        return fmt.Errorf("begin tx: %w", err)
    }
    defer tx.Rollback() // Rollback if not committed (no-op after commit)

    // Debit
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

    // Credit
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

### 3.2 Transaction Helper

```go
func withTransaction(db *sql.DB, fn func(tx *sql.Tx) error) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }

    defer func() {
        if p := recover(); p != nil {
            tx.Rollback()
            panic(p) // Re-panic after rollback
        }
    }()

    if err := fn(tx); err != nil {
        tx.Rollback()
        return err
    }

    return tx.Commit()
}

// Usage
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

## 4. Connection Pool Configuration

```go
func setupDB(connStr string) (*sql.DB, error) {
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        return nil, err
    }

    // Pool configuration
    db.SetMaxOpenConns(25)                  // Max open connections
    db.SetMaxIdleConns(5)                   // Max idle connections
    db.SetConnMaxLifetime(5 * time.Minute)  // Max connection age
    db.SetConnMaxIdleTime(1 * time.Minute)  // Max idle time

    // Verify connection
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

## 5. sqlx for Reduced Boilerplate

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

    // Get single row into struct
    var user User
    err = db.Get(&user, `SELECT * FROM users WHERE id = $1`, 1)

    // Select multiple rows into slice
    var users []User
    err = db.Select(&users, `SELECT * FROM users ORDER BY id LIMIT 10`)

    // Named queries
    _, err = db.NamedExec(
        `INSERT INTO users (name, email) VALUES (:name, :email)`,
        User{Name: "Alice", Email: "alice@example.com"},
    )

    // Named query with map
    rows, err := db.NamedQuery(
        `SELECT * FROM users WHERE name = :name`,
        map[string]any{"name": "Alice"},
    )
    defer rows.Close()
}
```

---

## 6. Migrations and Repository Pattern

### 6.1 Simple Migration System

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

### 6.2 Repository Pattern

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

## 7. Summary

### Key Takeaways

1. **`sql.Open` doesn't connect** — use `db.Ping()` to verify connectivity.
2. **Always close rows** — `defer rows.Close()` prevents connection leaks.
3. **Use parameterized queries** — `$1, $2` (PostgreSQL) or `?, ?` (MySQL). Never concatenate SQL strings.
4. **Check `sql.ErrNoRows`** — distinguish "not found" from actual errors.
5. **Configure connection pools** — set `MaxOpenConns`, `MaxIdleConns`, and timeouts.
6. **Transactions for atomicity** — use `defer tx.Rollback()` for safety.
7. **Use context** — `QueryRowContext`, `ExecContext` for timeout/cancellation support.

---

## Exercises

### Exercise 1: User CRUD with PostgreSQL
Build a complete user management API with database-backed CRUD, including search and pagination.

### Exercise 2: Transaction Safety
Implement a bank transfer system with proper transactions, handling concurrent access and rollbacks.

### Exercise 3: Connection Pool Tuning
Write a load test and experiment with different pool settings. Measure throughput and latency.

### Exercise 4: Migration Tool
Build a standalone migration tool that reads `.sql` files from a directory and applies them in order.

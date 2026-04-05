# 10. Testing

**Previous**: [Concurrency Patterns](./09_Concurrency_Patterns.md) | **Next**: [Standard Library](./11_Standard_Library.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write unit tests using the `testing` package
2. Apply table-driven test patterns for comprehensive coverage
3. Write benchmarks and interpret results
4. Use test helpers, subtests, and test fixtures
5. Apply fuzzing for automated edge case discovery

---

Go's testing philosophy matches its language philosophy: simple, explicit, and built-in. The `testing` package and `go test` command provide everything you need — unit tests, benchmarks, fuzzing, and examples — without any external framework.

## Table of Contents
1. [Test Basics](#1-test-basics)
2. [Table-Driven Tests](#2-table-driven-tests)
3. [Test Helpers and Fixtures](#3-test-helpers-and-fixtures)
4. [Benchmarks](#4-benchmarks)
5. [Fuzzing](#5-fuzzing)
6. [Test Organization](#6-test-organization)
7. [Summary](#7-summary)

---

## 1. Test Basics

### 1.1 Your First Test

```go
// file: math.go
package mathutil

func Add(a, b int) int {
    return a + b
}

func Divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}
```

```go
// file: math_test.go
package mathutil

import "testing"

func TestAdd(t *testing.T) {
    got := Add(2, 3)
    want := 5
    if got != want {
        t.Errorf("Add(2, 3) = %d, want %d", got, want)
    }
}

func TestDivide(t *testing.T) {
    got, err := Divide(10, 2)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if got != 5.0 {
        t.Errorf("Divide(10, 2) = %f, want 5.0", got)
    }
}

func TestDivideByZero(t *testing.T) {
    _, err := Divide(10, 0)
    if err == nil {
        t.Fatal("expected error for division by zero")
    }
}
```

```bash
# Run tests
go test ./...
go test -v ./...          # Verbose output
go test -run TestAdd      # Run specific test
go test -count=1 ./...    # Disable test caching
go test -cover ./...      # Show coverage percentage
go test -coverprofile=coverage.out ./...  # Coverage file
go tool cover -html=coverage.out          # HTML coverage report
```

### 1.2 t.Error vs t.Fatal

```go
func TestErrorVsFatal(t *testing.T) {
    // t.Error — reports failure but continues test
    if 1+1 != 2 {
        t.Error("math is broken")
    }
    // This still runs
    t.Log("after Error")

    // t.Fatal — reports failure and STOPS test immediately
    if true {
        t.Fatal("stopping here")
    }
    // This does NOT run
    t.Log("after Fatal")

    // Rule of thumb:
    // - t.Fatal/Fatalf: when remaining test would be meaningless (nil check, setup failure)
    // - t.Error/Errorf: when you want to report multiple failures
}
```

### 1.3 Subtests

```go
func TestMath(t *testing.T) {
    t.Run("Add", func(t *testing.T) {
        if Add(1, 2) != 3 {
            t.Error("1+2 should be 3")
        }
    })

    t.Run("Divide", func(t *testing.T) {
        t.Run("valid", func(t *testing.T) {
            result, err := Divide(10, 2)
            if err != nil {
                t.Fatal(err)
            }
            if result != 5.0 {
                t.Errorf("got %f, want 5.0", result)
            }
        })

        t.Run("by zero", func(t *testing.T) {
            _, err := Divide(10, 0)
            if err == nil {
                t.Fatal("expected error")
            }
        })
    })
}
```

```bash
# Run specific subtest
go test -run "TestMath/Divide/by_zero"
```

---

## 2. Table-Driven Tests

### 2.1 Basic Pattern

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive", 2, 3, 5},
        {"negative", -1, -2, -3},
        {"zero", 0, 0, 0},
        {"mixed", -5, 10, 5},
        {"large", 1000000, 2000000, 3000000},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            if got != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.expected)
            }
        })
    }
}
```

### 2.2 Table-Driven with Errors

```go
func TestDivide(t *testing.T) {
    tests := []struct {
        name      string
        a, b      float64
        want      float64
        wantErr   bool
        errString string
    }{
        {"valid division", 10, 2, 5, false, ""},
        {"float result", 7, 3, 2.3333333333, false, ""},
        {"divide by zero", 10, 0, 0, true, "division by zero"},
        {"zero numerator", 0, 5, 0, false, ""},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := Divide(tt.a, tt.b)

            if tt.wantErr {
                if err == nil {
                    t.Fatal("expected error, got nil")
                }
                if !strings.Contains(err.Error(), tt.errString) {
                    t.Errorf("error = %q, want containing %q", err, tt.errString)
                }
                return
            }

            if err != nil {
                t.Fatalf("unexpected error: %v", err)
            }

            if math.Abs(got-tt.want) > 1e-9 {
                t.Errorf("Divide(%g, %g) = %g, want %g", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### 2.3 Parallel Table Tests

```go
func TestFetchURL(t *testing.T) {
    tests := []struct {
        name string
        url  string
        want int
    }{
        {"google", "https://google.com", 200},
        {"github", "https://github.com", 200},
    }

    for _, tt := range tests {
        tt := tt // Capture for parallel
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel() // Run subtests concurrently
            resp, err := http.Get(tt.url)
            if err != nil {
                t.Fatal(err)
            }
            defer resp.Body.Close()
            if resp.StatusCode != tt.want {
                t.Errorf("status = %d, want %d", resp.StatusCode, tt.want)
            }
        })
    }
}
```

---

## 3. Test Helpers and Fixtures

### 3.1 Test Helpers

```go
func assertNoError(t *testing.T, err error) {
    t.Helper() // Marks this as helper — error reports show caller's line
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
}

func assertEqual[T comparable](t *testing.T, got, want T) {
    t.Helper()
    if got != want {
        t.Errorf("got %v, want %v", got, want)
    }
}

func TestWithHelpers(t *testing.T) {
    result, err := Divide(10, 2)
    assertNoError(t, err)
    assertEqual(t, result, 5.0)
}
```

### 3.2 TestMain

```go
func TestMain(m *testing.M) {
    // Setup before all tests
    fmt.Println("Setting up...")
    db := setupTestDB()

    // Run all tests
    code := m.Run()

    // Teardown after all tests
    fmt.Println("Cleaning up...")
    db.Close()

    os.Exit(code)
}
```

### 3.3 Temporary Files and Directories

```go
func TestWriteFile(t *testing.T) {
    // t.TempDir() — cleaned up automatically after test
    dir := t.TempDir()
    path := filepath.Join(dir, "test.txt")

    err := os.WriteFile(path, []byte("hello"), 0644)
    if err != nil {
        t.Fatal(err)
    }

    data, err := os.ReadFile(path)
    if err != nil {
        t.Fatal(err)
    }
    if string(data) != "hello" {
        t.Errorf("got %q, want %q", string(data), "hello")
    }
    // dir and its contents are removed automatically
}
```

### 3.4 Test Fixtures (testdata)

```go
// Files in testdata/ are ignored by the build system but available to tests
func TestParseConfig(t *testing.T) {
    data, err := os.ReadFile("testdata/valid_config.json")
    if err != nil {
        t.Fatal(err)
    }

    cfg, err := ParseConfig(data)
    if err != nil {
        t.Fatal(err)
    }
    if cfg.Port != 8080 {
        t.Errorf("port = %d, want 8080", cfg.Port)
    }
}

// Golden file pattern
func TestRender(t *testing.T) {
    got := Render(input)

    golden := filepath.Join("testdata", t.Name()+".golden")

    if *update { // -update flag
        os.WriteFile(golden, []byte(got), 0644)
    }

    want, _ := os.ReadFile(golden)
    if got != string(want) {
        t.Errorf("output mismatch:\ngot:\n%s\nwant:\n%s", got, string(want))
    }
}
```

---

## 4. Benchmarks

### 4.1 Writing Benchmarks

```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(42, 58)
    }
}

func BenchmarkDivide(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Divide(355.0, 113.0)
    }
}

// Benchmark with setup
func BenchmarkSort(b *testing.B) {
    data := make([]int, 10000)
    for i := range data {
        data[i] = rand.Intn(10000)
    }

    b.ResetTimer() // Don't count setup time

    for i := 0; i < b.N; i++ {
        d := make([]int, len(data))
        copy(d, data)
        sort.Ints(d)
    }
}
```

```bash
go test -bench=. -benchmem
# BenchmarkAdd-8       1000000000     0.25 ns/op    0 B/op    0 allocs/op
# BenchmarkDivide-8     500000000     2.38 ns/op    0 B/op    0 allocs/op
# BenchmarkSort-8           10000   105432 ns/op    81920 B/op  1 allocs/op

# Compare benchmarks
go test -bench=. -benchmem -count=5 > old.txt
# ... make changes ...
go test -bench=. -benchmem -count=5 > new.txt
benchstat old.txt new.txt
```

### 4.2 Sub-Benchmarks

```go
func BenchmarkConcat(b *testing.B) {
    sizes := []int{10, 100, 1000, 10000}

    for _, size := range sizes {
        b.Run(fmt.Sprintf("plus/%d", size), func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                s := ""
                for j := 0; j < size; j++ {
                    s += "a"
                }
            }
        })

        b.Run(fmt.Sprintf("builder/%d", size), func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                var builder strings.Builder
                for j := 0; j < size; j++ {
                    builder.WriteString("a")
                }
                _ = builder.String()
            }
        })
    }
}
```

---

## 5. Fuzzing

### 5.1 Fuzz Testing (Go 1.18+)

```go
func FuzzReverse(f *testing.F) {
    // Seed corpus — initial test cases
    f.Add("hello")
    f.Add("world")
    f.Add("")
    f.Add("한국어")

    f.Fuzz(func(t *testing.T, s string) {
        reversed := Reverse(s)
        doubleReversed := Reverse(reversed)

        // Property: reversing twice gives original
        if s != doubleReversed {
            t.Errorf("Reverse(Reverse(%q)) = %q", s, doubleReversed)
        }

        // Property: length preserved
        if utf8.RuneCountInString(s) != utf8.RuneCountInString(reversed) {
            t.Errorf("length changed: %d → %d", len(s), len(reversed))
        }
    })
}

func FuzzParseJSON(f *testing.F) {
    f.Add([]byte(`{"name": "test"}`))
    f.Add([]byte(`{}`))
    f.Add([]byte(`[]`))

    f.Fuzz(func(t *testing.T, data []byte) {
        var v any
        err := json.Unmarshal(data, &v)
        if err != nil {
            return // Invalid JSON is fine — just shouldn't panic
        }

        // Re-marshal and verify roundtrip
        encoded, err := json.Marshal(v)
        if err != nil {
            t.Fatalf("Marshal failed after successful Unmarshal: %v", err)
        }

        var v2 any
        if err := json.Unmarshal(encoded, &v2); err != nil {
            t.Fatalf("Unmarshal of re-marshaled data failed: %v", err)
        }
    })
}
```

```bash
go test -fuzz=FuzzReverse -fuzztime=30s
# Crashes saved to testdata/fuzz/FuzzReverse/
```

---

## 6. Test Organization

### 6.1 Package-Level vs External Tests

```go
// math_test.go — same package (white-box testing)
package mathutil

func TestInternalHelper(t *testing.T) {
    // Can access unexported functions
    result := internalHelper(42)
    if result != 84 {
        t.Error("unexpected")
    }
}

// math_external_test.go — external package (black-box testing)
package mathutil_test

import "github.com/user/project/mathutil"

func TestPublicAPI(t *testing.T) {
    // Can only access exported functions
    result := mathutil.Add(1, 2)
    if result != 3 {
        t.Error("unexpected")
    }
}
```

### 6.2 Integration Tests with Build Tags

```go
//go:build integration

package mypackage

func TestDatabaseIntegration(t *testing.T) {
    // Only runs with: go test -tags integration
    db := connectToTestDB()
    defer db.Close()
    // ...
}
```

### 6.3 Example Tests

```go
func ExampleAdd() {
    fmt.Println(Add(2, 3))
    // Output: 5
}

func ExampleDivide() {
    result, err := Divide(10, 3)
    if err != nil {
        fmt.Println("error:", err)
        return
    }
    fmt.Printf("%.4f\n", result)
    // Output: 3.3333
}
```

---

## 7. Summary

### Key Takeaways

1. **Test files end in `_test.go`** — automatically excluded from production builds.
2. **Table-driven tests** — the standard Go pattern. Cover many cases with minimal code.
3. **`t.Helper()`** — essential for test helper functions to show correct line numbers.
4. **`t.Parallel()`** — run subtests concurrently for faster test suites.
5. **Benchmarks with `b.N`** — the framework determines iteration count automatically.
6. **Fuzzing finds edge cases** — property-based testing discovers bugs you wouldn't think to test.
7. **No external framework needed** — the standard `testing` package covers unit tests, benchmarks, fuzzing, and examples.

### Testing Commands

```bash
go test ./...                    # All tests
go test -v -run TestName        # Specific test, verbose
go test -bench=. -benchmem      # Benchmarks with memory
go test -cover -coverprofile=c.out  # Coverage
go test -race ./...              # Race detection
go test -fuzz=FuzzName -fuzztime=1m # Fuzzing
go test -short ./...             # Skip long tests
```

---

## Exercises

### Exercise 1: Table-Driven Tests
Write table-driven tests for a `ParseURL` function that handles valid URLs, invalid URLs, missing schemes, and edge cases.

### Exercise 2: Benchmark Comparison
Benchmark three string concatenation methods (+ operator, fmt.Sprintf, strings.Builder) across sizes 10, 100, 1000, 10000.

### Exercise 3: Fuzz Testing
Write a fuzz test for a URL parser that checks: (a) no panics on any input, (b) parse/unparse roundtrip consistency.

### Exercise 4: Test Doubles
Write a service that depends on a database interface. Create a mock implementation and test the service logic independently.

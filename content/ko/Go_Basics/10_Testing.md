# 10. 테스팅

**이전**: [동시성 패턴](./09_Concurrency_Patterns.md) | **다음**: [표준 라이브러리](./11_Standard_Library.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `testing` 패키지를 사용하여 단위 테스트를 작성한다
2. 포괄적인 커버리지를 위한 테이블 구동 테스트 패턴을 적용한다
3. 벤치마크를 작성하고 결과를 해석한다
4. 테스트 헬퍼, 서브테스트, 테스트 픽스처를 사용한다
5. 자동 엣지 케이스 발견을 위한 퍼징을 적용한다

---

Go의 테스트 철학은 언어 철학과 일치한다: 단순하고, 명시적이며, 내장되어 있다. `testing` 패키지와 `go test` 명령은 단위 테스트, 벤치마크, 퍼징, 예제 등 필요한 모든 것을 외부 프레임워크 없이 제공한다.

## 목차
1. [테스트 기초](#1-테스트-기초)
2. [테이블 구동 테스트](#2-테이블-구동-테스트)
3. [테스트 헬퍼와 픽스처](#3-테스트-헬퍼와-픽스처)
4. [벤치마크](#4-벤치마크)
5. [퍼징](#5-퍼징)
6. [테스트 구성](#6-테스트-구성)
7. [요약](#7-요약)

---

## 1. 테스트 기초

### 1.1 첫 번째 테스트

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
# 테스트 실행
go test ./...
go test -v ./...          # 상세 출력
go test -run TestAdd      # 특정 테스트 실행
go test -count=1 ./...    # 테스트 캐시 비활성화
go test -cover ./...      # 커버리지 비율 표시
go test -coverprofile=coverage.out ./...  # 커버리지 파일
go tool cover -html=coverage.out          # HTML 커버리지 보고서
```

### 1.2 t.Error vs t.Fatal

```go
func TestErrorVsFatal(t *testing.T) {
    // t.Error — 실패를 보고하지만 테스트 계속 진행
    if 1+1 != 2 {
        t.Error("math is broken")
    }
    // 이것은 여전히 실행됨
    t.Log("after Error")

    // t.Fatal — 실패를 보고하고 테스트 즉시 중단
    if true {
        t.Fatal("stopping here")
    }
    // 이것은 실행되지 않음
    t.Log("after Fatal")

    // 경험 법칙:
    // - t.Fatal/Fatalf: 나머지 테스트가 의미 없을 때 (nil 검사, 설정 실패)
    // - t.Error/Errorf: 여러 실패를 보고하고 싶을 때
}
```

### 1.3 서브테스트

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
# 특정 서브테스트 실행
go test -run "TestMath/Divide/by_zero"
```

---

## 2. 테이블 구동 테스트

### 2.1 기본 패턴

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

### 2.2 에러가 있는 테이블 구동 테스트

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

### 2.3 병렬 테이블 테스트

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
        tt := tt // 병렬을 위한 캡처
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel() // 서브테스트를 동시에 실행
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

## 3. 테스트 헬퍼와 픽스처

### 3.1 테스트 헬퍼

```go
func assertNoError(t *testing.T, err error) {
    t.Helper() // 헬퍼로 표시 — 에러 보고가 호출자의 줄 번호를 표시
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
    // 모든 테스트 전 설정
    fmt.Println("Setting up...")
    db := setupTestDB()

    // 모든 테스트 실행
    code := m.Run()

    // 모든 테스트 후 정리
    fmt.Println("Cleaning up...")
    db.Close()

    os.Exit(code)
}
```

### 3.3 임시 파일과 디렉토리

```go
func TestWriteFile(t *testing.T) {
    // t.TempDir() — 테스트 후 자동으로 정리
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
    // dir과 그 내용은 자동으로 제거됨
}
```

### 3.4 테스트 픽스처 (testdata)

```go
// testdata/의 파일은 빌드 시스템에서 무시되지만 테스트에서 사용 가능
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

// 골든 파일 패턴
func TestRender(t *testing.T) {
    got := Render(input)

    golden := filepath.Join("testdata", t.Name()+".golden")

    if *update { // -update 플래그
        os.WriteFile(golden, []byte(got), 0644)
    }

    want, _ := os.ReadFile(golden)
    if got != string(want) {
        t.Errorf("output mismatch:\ngot:\n%s\nwant:\n%s", got, string(want))
    }
}
```

---

## 4. 벤치마크

### 4.1 벤치마크 작성

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

// 설정이 포함된 벤치마크
func BenchmarkSort(b *testing.B) {
    data := make([]int, 10000)
    for i := range data {
        data[i] = rand.Intn(10000)
    }

    b.ResetTimer() // 설정 시간을 측정하지 않음

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

# 벤치마크 비교
go test -bench=. -benchmem -count=5 > old.txt
# ... 변경 작업 ...
go test -bench=. -benchmem -count=5 > new.txt
benchstat old.txt new.txt
```

### 4.2 서브 벤치마크

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

## 5. 퍼징

### 5.1 퍼즈 테스트 (Go 1.18+)

```go
func FuzzReverse(f *testing.F) {
    // 시드 코퍼스 — 초기 테스트 케이스
    f.Add("hello")
    f.Add("world")
    f.Add("")
    f.Add("한국어")

    f.Fuzz(func(t *testing.T, s string) {
        reversed := Reverse(s)
        doubleReversed := Reverse(reversed)

        // 속성: 두 번 뒤집으면 원본을 얻음
        if s != doubleReversed {
            t.Errorf("Reverse(Reverse(%q)) = %q", s, doubleReversed)
        }

        // 속성: 길이 보존
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
            return // 유효하지 않은 JSON은 괜찮음 — 패닉만 발생하지 않으면 됨
        }

        // 재마샬링하고 라운드트립 검증
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
# 크래시가 testdata/fuzz/FuzzReverse/에 저장됨
```

---

## 6. 테스트 구성

### 6.1 패키지 수준 vs 외부 테스트

```go
// math_test.go — 같은 패키지 (화이트박스 테스트)
package mathutil

func TestInternalHelper(t *testing.T) {
    // 비공개 함수에 접근 가능
    result := internalHelper(42)
    if result != 84 {
        t.Error("unexpected")
    }
}

// math_external_test.go — 외부 패키지 (블랙박스 테스트)
package mathutil_test

import "github.com/user/project/mathutil"

func TestPublicAPI(t *testing.T) {
    // 공개 함수만 접근 가능
    result := mathutil.Add(1, 2)
    if result != 3 {
        t.Error("unexpected")
    }
}
```

### 6.2 빌드 태그를 사용한 통합 테스트

```go
//go:build integration

package mypackage

func TestDatabaseIntegration(t *testing.T) {
    // go test -tags integration 으로만 실행됨
    db := connectToTestDB()
    defer db.Close()
    // ...
}
```

### 6.3 예제 테스트

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

## 7. 요약

### 핵심 포인트

1. **테스트 파일은 `_test.go`로 끝난다** — 프로덕션 빌드에서 자동으로 제외된다.
2. **테이블 구동 테스트** — 표준 Go 패턴이다. 최소한의 코드로 많은 케이스를 커버한다.
3. **`t.Helper()`** — 정확한 줄 번호를 표시하기 위해 테스트 헬퍼 함수에 필수이다.
4. **`t.Parallel()`** — 더 빠른 테스트 스위트를 위해 서브테스트를 동시에 실행한다.
5. **`b.N`을 사용한 벤치마크** — 프레임워크가 자동으로 반복 횟수를 결정한다.
6. **퍼징은 엣지 케이스를 찾는다** — 속성 기반 테스트가 생각하지 못한 버그를 발견한다.
7. **외부 프레임워크가 필요 없다** — 표준 `testing` 패키지가 단위 테스트, 벤치마크, 퍼징, 예제를 모두 포함한다.

### 테스트 명령어

```bash
go test ./...                    # 모든 테스트
go test -v -run TestName        # 특정 테스트, 상세 출력
go test -bench=. -benchmem      # 메모리 포함 벤치마크
go test -cover -coverprofile=c.out  # 커버리지
go test -race ./...              # 경쟁 조건 감지
go test -fuzz=FuzzName -fuzztime=1m # 퍼징
go test -short ./...             # 긴 테스트 건너뛰기
```

---

## 연습 문제

### 연습 1: 테이블 구동 테스트
유효한 URL, 유효하지 않은 URL, 누락된 스킴, 엣지 케이스를 처리하는 `ParseURL` 함수에 대한 테이블 구동 테스트를 작성한다.

### 연습 2: 벤치마크 비교
세 가지 문자열 연결 방법(+ 연산자, fmt.Sprintf, strings.Builder)을 크기 10, 100, 1000, 10000에 걸쳐 벤치마크한다.

### 연습 3: 퍼즈 테스트
URL 파서에 대한 퍼즈 테스트를 작성한다: (a) 어떤 입력에도 패닉이 발생하지 않음, (b) 파싱/역파싱 라운드트립 일관성을 확인한다.

### 연습 4: 테스트 더블
데이터베이스 인터페이스에 의존하는 서비스를 작성한다. 목 구현을 생성하고 서비스 로직을 독립적으로 테스트한다.

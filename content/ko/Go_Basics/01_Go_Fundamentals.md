# 01. Go 기초

**이전**: [개요](./00_Overview.md) | **다음**: [복합 타입](./02_Composite_Types.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. Go를 설치하고 개발 환경을 구성할 수 있다
2. `var`, `:=`를 사용하여 변수를 선언하고 타입 추론(type inference)을 이해할 수 있다
3. 정수, 실수, 문자열, 불리언, 룬(rune) 등 모든 기본 타입을 사용할 수 있다
4. `if`, `for`, `switch`, `defer`로 제어 흐름을 작성할 수 있다
5. 다중 반환값을 가진 함수를 정의하고 호출할 수 있다

---

Go는 단순함에 보상을 주는 언어이다. 다른 언어가 무언가를 하는 다섯 가지 방법을 제공하는 반면, Go는 보통 하나의 방법만 제공한다 — 그리고 그 하나의 방법은 명확하고, 빠르고, 예측 가능하다. 이 레슨에서는 기본 구성 요소인 타입, 변수, 제어 흐름, 함수를 다룬다.

## 목차
1. [Hello, Go!](#1-hello-go)
2. [변수와 선언](#2-변수와-선언)
3. [기본 타입](#3-기본-타입)
4. [타입 변환과 상수](#4-타입-변환과-상수)
5. [제어 흐름](#5-제어-흐름)
6. [함수](#6-함수)
7. [요약](#7-요약)

---

## 1. Hello, Go!

### 1.1 첫 번째 프로그램

모든 Go 프로그램은 패키지(package) 선언과 import 블록으로 시작한다. `main` 패키지와 `main()` 함수가 실행 파일의 진입점(entry point)이다.

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, Go!")
}
```

```bash
# 직접 실행 (임시 디렉토리에 컴파일 후 실행)
go run main.go

# 바이너리 빌드
go build -o hello main.go
./hello

# 코드 포맷팅
go fmt main.go
```

### 1.2 프로그램 구조

```go
package main          // 패키지 선언 — 모든 파일은 정확히 하나의 패키지에 속한다
import (              // import 블록 — 괄호로 묶어서 그룹화
    "fmt"             // 표준 라이브러리: 형식화된 I/O
    "math"            // 표준 라이브러리: 수학 함수
    "strings"         // 표준 라이브러리: 문자열 조작
)

// main은 진입점 — 인자 없음, 반환값 없음
func main() {
    fmt.Println("Pi is approximately", math.Pi)
    fmt.Println(strings.ToUpper("hello"))
}
```

핵심 사항:
- **세미콜론 없음** — 렉서(lexer)가 자동으로 삽입한다
- **사용하지 않는 import는 컴파일 에러** — Go는 깔끔함을 강제한다
- **내보내는 이름은 대문자로 시작** — `fmt.Println`은 내보내짐; `fmt.println`은 불가능하다
- **`gofmt`** — 단일 포맷팅 스타일을 강제한다. 스타일 논쟁이 없다

### 1.3 Go 워크스페이스

```bash
# 현대 Go는 모듈 사용 (Go 1.11+)
mkdir myproject && cd myproject
go mod init github.com/username/myproject

# go.mod 파일 생성 — 모듈 정의 파일
cat go.mod
# module github.com/username/myproject
# go 1.22
```

---

## 2. 변수와 선언

### 2.1 var 선언

`var` 키워드는 명시적 타입 또는 초기값에서 추론된 타입으로 변수를 선언한다.

```go
package main

import "fmt"

func main() {
    // 명시적 타입
    var name string = "Alice"
    var age int = 30
    var height float64 = 5.9

    // 타입 추론 — 컴파일러가 타입을 추론한다
    var city = "Seoul"           // string
    var population = 9_700_000   // int (가독성을 위한 밑줄)

    // 영값(zero value) — 모든 타입에 기본값이 있다
    var count int       // 0
    var rate float64    // 0.0
    var label string    // "" (빈 문자열)
    var active bool     // false

    fmt.Println(name, age, height)
    fmt.Println(city, population)
    fmt.Println(count, rate, label, active)

    // 블록 선언
    var (
        x int    = 10
        y int    = 20
        z string = "result"
    )
    fmt.Println(x, y, z)
}
```

### 2.2 짧은 변수 선언

함수 내부에서 `:=` 연산자는 선언과 초기화를 한 번에 수행한다. 가장 일반적인 형태이다.

```go
func main() {
    // 짧은 선언 — 타입 추론
    name := "Bob"          // string
    age := 25              // int
    pi := 3.14159          // float64
    active := true         // bool

    fmt.Println(name, age, pi, active)

    // 다중 할당
    x, y := 10, 20
    fmt.Println(x, y)

    // 값 교환 — 임시 변수 불필요
    x, y = y, x
    fmt.Println(x, y) // 20, 10

    // := 는 왼쪽에 최소 하나의 새 변수가 필요하다
    x, z := 100, "hello"  // OK: z는 새 변수
    fmt.Println(x, z)
}
```

### 2.3 네이밍 규칙

```go
// 로컬 변수와 내보내지 않는 식별자에는 camelCase
userName := "alice"
maxRetries := 3

// 내보내는 식별자(패키지 외부에서 볼 수 있음)에는 PascalCase
func ProcessOrder() {}
type HttpClient struct {}

// 약어는 대문자 유지
var httpURL string
var xmlParser *Parser
var userID int
```

---

## 3. 기본 타입

### 3.1 정수 타입

```go
package main

import (
    "fmt"
    "math"
)

func main() {
    // 부호 있는 정수(signed integer)
    var i8 int8 = 127                    // -128 ~ 127
    var i16 int16 = 32767                // -32768 ~ 32767
    var i32 int32 = math.MaxInt32        // -2^31 ~ 2^31-1
    var i64 int64 = math.MaxInt64        // -2^63 ~ 2^63-1

    // 플랫폼 의존: 32비트 시스템에서 32비트, 64비트 시스템에서 64비트
    var i int = 42

    // 부호 없는 정수(unsigned integer)
    var u8 uint8 = 255                   // 0 ~ 255 (별칭: byte)
    var u16 uint16 = 65535
    var u32 uint32 = math.MaxUint32
    var u64 uint64 = math.MaxUint64

    // byte와 rune은 별칭이다
    var b byte = 'A'        // uint8의 별칭
    var r rune = '가'        // int32의 별칭 (유니코드 코드 포인트)

    fmt.Println(i8, i16, i32, i64, i)
    fmt.Println(u8, u16, u32, u64)
    fmt.Println(b, r)

    // 숫자 리터럴
    decimal := 42
    hex := 0xFF
    octal := 0o77
    binary := 0b1010_1100
    withSep := 1_000_000

    fmt.Printf("dec=%d hex=%d oct=%d bin=%d sep=%d\n",
        decimal, hex, octal, binary, withSep)
}
```

### 3.2 부동소수점과 복소수

```go
func main() {
    var f32 float32 = 3.14       // ~7자리 소수점 정밀도
    var f64 float64 = 3.14159265 // ~15자리 소수점 정밀도

    // 기본 실수 리터럴 타입은 float64
    pi := 3.14159265358979

    // 과학 표기법
    avogadro := 6.022e23
    planck := 6.626e-34

    fmt.Println(f32, f64, pi)
    fmt.Println(avogadro, planck)

    // 복소수 (내장!)
    c1 := complex(3, 4)           // 3+4i
    c2 := 2 + 5i                  // 리터럴 구문
    fmt.Println(c1 + c2)          // (5+9i)
    fmt.Println(real(c1), imag(c1)) // 3, 4
}
```

### 3.3 문자열과 룬(Rune)

```go
package main

import (
    "fmt"
    "strings"
    "unicode/utf8"
)

func main() {
    // 문자열은 불변 바이트 시퀀스이다
    greeting := "Hello, 世界"
    fmt.Println(len(greeting))                    // 13 바이트 (문자가 아니다!)
    fmt.Println(utf8.RuneCountInString(greeting)) // 9 룬 (문자)

    // 원시 문자열(raw string) — 이스케이프 처리 없음
    path := `C:\Users\alice\documents`
    multiline := `
        여러 줄
        문자열이다
    `
    fmt.Println(path)
    fmt.Println(multiline)

    // 문자열 연산
    fmt.Println(strings.ToUpper("hello"))          // "HELLO"
    fmt.Println(strings.Contains("hello", "ell"))  // true
    fmt.Println(strings.Replace("aaa", "a", "b", 2)) // "bba"
    fmt.Println(strings.Split("a,b,c", ","))       // [a b c]
    fmt.Println(strings.Join([]string{"a", "b"}, "-")) // "a-b"

    // 룬(문자) 순회
    for i, r := range "Go 한국어" {
        fmt.Printf("바이트 %d: 룬 %c (U+%04X)\n", i, r, r)
    }

    // 효율적인 연결을 위한 문자열 빌더(string builder)
    var builder strings.Builder
    for i := 0; i < 5; i++ {
        fmt.Fprintf(&builder, "항목 %d ", i)
    }
    fmt.Println(builder.String())
}
```

### 3.4 불리언

```go
func main() {
    a := true
    b := false

    fmt.Println(a && b)  // false (AND)
    fmt.Println(a || b)  // true (OR)
    fmt.Println(!a)      // false (NOT)

    // 비교 연산자는 bool을 반환한다
    x, y := 10, 20
    fmt.Println(x == y)  // false
    fmt.Println(x < y)   // true
    fmt.Println(x != y)  // true
}
```

---

## 4. 타입 변환과 상수

### 4.1 명시적 타입 변환

Go에는 암시적 타입 변환(implicit type conversion)이 없다. 모든 변환은 명시적이어야 한다.

```go
func main() {
    // int → float64
    x := 42
    y := float64(x)

    // float64 → int (절삭)
    pi := 3.99
    n := int(pi) // 3, 4가 아니다

    // int → string (유니코드 코드 포인트, 숫자 변환이 아니다!)
    r := rune(65)
    fmt.Println(string(r)) // "A"

    // 숫자를 문자열로 — fmt 또는 strconv 사용
    import "strconv"
    s := strconv.Itoa(42)        // "42"
    f := strconv.FormatFloat(3.14, 'f', 2, 64) // "3.14"

    // 문자열을 숫자로
    n, err := strconv.Atoi("42")
    if err != nil {
        fmt.Println("파싱 에러:", err)
    }
    fmt.Println(n) // 42

    fmt.Println(y, n, s, f)
}
```

### 4.2 상수

```go
package main

import "fmt"

// 타입이 있는 상수
const Pi float64 = 3.14159265358979
const MaxRetries int = 3

// 타입이 없는 상수 — 유연하게 컨텍스트에 적응한다
const (
    Hello  = "Hello"
    Answer = 42        // int, float64 등으로 사용 가능
    E      = 2.71828
)

// iota — 자동 증가 상수 생성기
type Weekday int

const (
    Sunday    Weekday = iota // 0
    Monday                   // 1
    Tuesday                  // 2
    Wednesday                // 3
    Thursday                 // 4
    Friday                   // 5
    Saturday                 // 6
)

// iota를 사용한 비트 플래그(bit flag)
type Permission uint8

const (
    Read    Permission = 1 << iota // 1
    Write                          // 2
    Execute                        // 4
)

func main() {
    fmt.Println(Pi, MaxRetries)
    fmt.Println(Monday, Friday)
    fmt.Printf("Read=%d Write=%d Execute=%d\n", Read, Write, Execute)

    // OR로 권한 결합
    perm := Read | Write
    fmt.Printf("perm=%d hasRead=%t hasExec=%t\n",
        perm, perm&Read != 0, perm&Execute != 0)
}
```

---

## 5. 제어 흐름

### 5.1 if/else

```go
func main() {
    x := 42

    // 표준 if/else
    if x > 0 {
        fmt.Println("양수")
    } else if x < 0 {
        fmt.Println("음수")
    } else {
        fmt.Println("영")
    }

    // 초기화 문이 있는 if — 변수는 if/else 블록에 한정된다
    if err := doSomething(); err != nil {
        fmt.Println("에러:", err)
    }
    // err는 여기서 접근 불가

    // 일반적인 패턴: 에러 확인
    if data, err := fetchData(); err != nil {
        fmt.Println("실패:", err)
    } else {
        fmt.Println("결과:", data)
    }
}
```

### 5.2 for 루프

Go에는 `for`만 있다 — `while`이나 `do-while`은 없다. 모든 경우를 처리한다.

```go
func main() {
    // 전통적인 3요소 for
    for i := 0; i < 5; i++ {
        fmt.Println(i)
    }

    // while 스타일 (조건만)
    n := 1
    for n < 100 {
        n *= 2
    }
    fmt.Println(n) // 128

    // 무한 루프
    count := 0
    for {
        count++
        if count > 3 {
            break
        }
    }

    // 슬라이스(slice) 순회
    fruits := []string{"apple", "banana", "cherry"}
    for index, value := range fruits {
        fmt.Printf("%d: %s\n", index, value)
    }

    // range — 인덱스 무시
    for _, fruit := range fruits {
        fmt.Println(fruit)
    }

    // 문자열 순회 (바이트가 아닌 룬 단위로 순회한다)
    for i, r := range "Go언어" {
        fmt.Printf("byte %d: %c\n", i, r)
    }

    // 맵(map) 순회
    ages := map[string]int{"Alice": 30, "Bob": 25}
    for name, age := range ages {
        fmt.Printf("%s is %d\n", name, age)
    }

    // continue와 레이블이 붙은 루프
    outer:
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if i == j {
                continue outer
            }
            fmt.Println(i, j)
        }
    }
}
```

### 5.3 switch

```go
func main() {
    // 표현식 switch
    day := "Monday"
    switch day {
    case "Monday":
        fmt.Println("한 주의 시작")
    case "Friday":
        fmt.Println("불금!")
    case "Saturday", "Sunday":
        fmt.Println("주말!")
    default:
        fmt.Println("평일")
    }

    // 조건 없는 switch (if/else 체인보다 깔끔하다)
    score := 85
    switch {
    case score >= 90:
        fmt.Println("A")
    case score >= 80:
        fmt.Println("B")
    case score >= 70:
        fmt.Println("C")
    default:
        fmt.Println("F")
    }

    // 초기화 문이 있는 switch
    switch os := runtime.GOOS; os {
    case "linux":
        fmt.Println("Linux")
    case "darwin":
        fmt.Println("macOS")
    default:
        fmt.Println(os)
    }

    // 타입 switch
    var val interface{} = 42
    switch v := val.(type) {
    case int:
        fmt.Printf("정수: %d\n", v)
    case string:
        fmt.Printf("문자열: %s\n", v)
    case bool:
        fmt.Printf("불리언: %t\n", v)
    default:
        fmt.Printf("알 수 없음: %T\n", v)
    }

    // fallthrough — C와 달리 명시적이다
    switch 3 {
    case 3:
        fmt.Println("three")
        fallthrough
    case 4:
        fmt.Println("four (fallthrough를 통해)")
    case 5:
        fmt.Println("five (도달하지 않음)")
    }
}
```

### 5.4 defer

`defer`는 감싸는 함수가 반환될 때 실행할 함수 호출을 예약한다. 지연된 호출은 LIFO(후입선출) 순서로 실행된다.

```go
package main

import (
    "fmt"
    "os"
)

func main() {
    // defer는 main()이 반환될 때 실행된다
    fmt.Println("시작")
    defer fmt.Println("지연 1")
    defer fmt.Println("지연 2")
    fmt.Println("끝")
    // 출력: 시작, 끝, 지연 2, 지연 1

    // 일반적인 패턴: 리소스 정리
    f, err := os.Create("test.txt")
    if err != nil {
        fmt.Println(err)
        return
    }
    defer f.Close() // 에러가 발생해도 닫기가 보장된다

    f.WriteString("hello\n")

    // defer는 defer 시점의 값을 캡처한다
    x := 10
    defer fmt.Println("deferred x =", x) // 20이 아닌 10을 출력한다
    x = 20
    fmt.Println("현재 x =", x)
}

// defer는 리소스 정리에 필수적이다
func readFile(path string) (string, error) {
    f, err := os.Open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()

    // 여기서 에러가 발생해도 f.Close()는 여전히 실행된다
    data, err := io.ReadAll(f)
    if err != nil {
        return "", err
    }
    return string(data), nil
}
```

---

## 6. 함수

### 6.1 기본 함수

```go
package main

import (
    "fmt"
    "math"
)

// 간단한 함수
func greet(name string) string {
    return "Hello, " + name + "!"
}

// 같은 타입의 여러 매개변수
func add(a, b int) int {
    return a + b
}

// 다중 반환값 — Go의 관용적 패턴
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// 이름이 있는 반환값
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return // "벌거벗은 return" — 이름 있는 값을 반환한다
}

// 가변 인자 함수(variadic function)
func sum(nums ...int) int {
    total := 0
    for _, n := range nums {
        total += n
    }
    return total
}

func main() {
    fmt.Println(greet("Go"))          // "Hello, Go!"
    fmt.Println(add(3, 4))            // 7

    result, err := divide(10, 3)
    if err != nil {
        fmt.Println("에러:", err)
    } else {
        fmt.Printf("10 / 3 = %.2f\n", result)
    }

    x, y := split(17)
    fmt.Println(x, y)

    fmt.Println(sum(1, 2, 3, 4, 5))  // 15

    // 슬라이스를 가변 인자로 전개
    numbers := []int{10, 20, 30}
    fmt.Println(sum(numbers...))      // 60
}
```

### 6.2 값으로서의 함수

```go
func main() {
    // 함수 변수
    operation := add
    fmt.Println(operation(3, 4)) // 7

    // 익명 함수(anonymous function)
    double := func(x int) int {
        return x * 2
    }
    fmt.Println(double(5)) // 10

    // 즉시 호출
    result := func(a, b int) int {
        return a * b
    }(3, 4)
    fmt.Println(result) // 12

    // 고차 함수(higher-order function)
    apply := func(f func(int) int, val int) int {
        return f(val)
    }
    fmt.Println(apply(double, 10)) // 20

    // 클로저(closure) — 외부 변수를 캡처한다
    counter := makeCounter()
    fmt.Println(counter()) // 1
    fmt.Println(counter()) // 2
    fmt.Println(counter()) // 3
}

func makeCounter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}
```

### 6.3 출력 포맷팅

```go
func main() {
    name := "Alice"
    age := 30
    height := 5.75

    // Printf 동사(verb)
    fmt.Printf("이름: %s\n", name)         // 문자열
    fmt.Printf("나이: %d\n", age)           // 정수
    fmt.Printf("키: %.1f ft\n", height)    // 실수
    fmt.Printf("16진수: %x\n", 255)        // 16진수
    fmt.Printf("2진수: %b\n", 42)          // 2진수
    fmt.Printf("타입: %T\n", name)         // 타입 이름
    fmt.Printf("값: %v\n", name)           // 기본 형식
    fmt.Printf("따옴표: %q\n", name)       // 따옴표로 감싼 문자열
    fmt.Printf("포인터: %p\n", &name)      // 포인터 주소

    // Sprintf는 문자열을 반환한다 (출력하지 않는다)
    msg := fmt.Sprintf("%s는 %d세입니다", name, age)
    fmt.Println(msg)

    // Fprintln은 모든 io.Writer에 쓴다
    fmt.Fprintln(os.Stderr, "이것은 stderr로 출력된다")
}
```

---

## 7. 요약

### 핵심 포인트

1. **Go는 주관적이다** — 하나의 포맷팅 스타일, 사용하지 않는 import 금지, 명시적 타입 변환. 이를 통해 자전거 헛간 논쟁을 없애고 코드베이스를 일관되게 유지한다.

2. **영값이 유용하다** — 모든 타입에 의미 있는 기본값(`0`, `""`, `false`, `nil`)이 있다. 많은 경우 생성자가 필요 없다.

3. **`:=`가 관용적이다** — 함수 내부에서는 짧은 선언을 사용한다. `var`는 패키지 수준 선언과 영값 초기화에 사용한다.

4. **`for`가 유일한 루프이다** — 전통적 반복, while 루프, 무한 루프, 컬렉션에 대한 range 기반 반복을 모두 처리한다.

5. **다중 반환값** — Go는 예외 대신 `(값, error)` 쌍을 사용한다. 항상 에러를 확인해야 한다.

6. **`defer`로 정리** — 함수 종료 방식에 관계없이 리소스 정리를 보장한다.

7. **함수는 일급 시민(first-class citizen)이다** — 변수에 할당하고, 인자로 전달하고, 다른 함수에서 반환할 수 있다.

### 흔한 실수

| 실수 | 해결 |
|------|------|
| 새 변수에 `:=` 대신 `=` 사용 | `:=`는 선언+할당; `=`는 할당만 |
| 함수의 에러 무시 | 항상 `error` 반환값을 처리한다 |
| 루프에서 문자열 연결 | 대신 `strings.Builder`를 사용한다 |
| UTF-8 문자열에서 문자 수에 `len(s)` 사용 | `utf8.RuneCountInString(s)`를 사용한다 |

---

## 연습 문제

### 연습 1: 온도 변환기
`celsiusToFahrenheit(c float64) float64` 함수와 그 역함수를 작성하라. 0°C부터 100°C까지 10단위로 변환 표를 출력하라.

### 연습 2: FizzBuzz
1-100까지 FizzBuzz 프로그램을 작성하라. 3의 배수는 "Fizz", 5의 배수는 "Buzz", 둘 다면 "FizzBuzz", 아니면 숫자를 출력하라.

### 연습 3: 문자열 분석
문자열을 받아 (a) 단어 수, (b) 문자(룬) 수, (c) 뒤집힌 문자열을 반환하는 함수를 작성하라. 유니코드를 올바르게 처리하라.

### 연습 4: 간단한 계산기
`calc(a float64, op string, b float64) (float64, error)` 함수를 작성하라. `+`, `-`, `*`, `/`를 지원하고, 0으로 나누기와 알 수 없는 연산자에 대해 에러를 반환하라.

# 04. 인터페이스

**이전**: [함수와 메서드](./03_Functions_and_Methods.md) | **다음**: [에러 처리](./05_Error_Handling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. Go의 암시적 만족(implicit satisfaction) 모델을 사용하여 인터페이스를 정의하고 구현할 수 있다
2. 빈 인터페이스(empty interface)와 타입 단언(type assertion)을 안전하게 사용할 수 있다
3. Stringer, Reader, Writer 같은 일반적인 인터페이스 패턴을 적용할 수 있다
4. Go 규칙을 따르는 작고 합성 가능한 인터페이스를 설계할 수 있다
5. 다형적 동작을 위한 타입 스위치(type switch)를 사용할 수 있다

---

인터페이스(interface)는 Go의 핵심 추상화 메커니즘이다. Java나 C#에서 클래스가 구현하는 인터페이스를 명시적으로 선언하는 것과 달리, Go는 **암시적 만족**을 사용한다 — 타입이 올바른 메서드를 가지고 있으면 자동으로 인터페이스를 구현한다. 이러한 디커플링이 Go 인터페이스를 강력하고 유연하게 만드는 요소이다.

## 목차
1. [인터페이스 기초](#1-인터페이스-기초)
2. [암시적 만족](#2-암시적-만족)
3. [일반적인 표준 인터페이스](#3-일반적인-표준-인터페이스)
4. [빈 인터페이스와 타입 단언](#4-빈-인터페이스와-타입-단언)
5. [인터페이스 합성](#5-인터페이스-합성)
6. [인터페이스 설계 원칙](#6-인터페이스-설계-원칙)
7. [요약](#7-요약)

---

## 1. 인터페이스 기초

### 1.1 인터페이스 정의

인터페이스는 메서드 시그니처의 집합을 정의한다. 모든 메서드를 구현하는 모든 타입이 인터페이스를 만족한다.

```go
package main

import (
    "fmt"
    "math"
)

// 인터페이스 정의
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Circle은 Shape를 구현한다
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

// Rectangle은 Shape를 구현한다
type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// 인터페이스를 받는 함수
func printShape(s Shape) {
    fmt.Printf("넓이: %.2f, 둘레: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    c := Circle{Radius: 5}
    r := Rectangle{Width: 10, Height: 3}

    printShape(c) // 넓이: 78.54, 둘레: 31.42
    printShape(r) // 넓이: 30.00, 둘레: 26.00

    // 인터페이스 타입의 슬라이스 — 다형성(polymorphism)
    shapes := []Shape{c, r, Circle{Radius: 1}}
    totalArea := 0.0
    for _, s := range shapes {
        totalArea += s.Area()
    }
    fmt.Printf("총 넓이: %.2f\n", totalArea)
}
```

### 1.2 인터페이스 값

인터페이스 값은 두 가지 구성 요소로 이루어진다: **타입**과 **값**.

```go
func main() {
    var s Shape
    fmt.Println(s)         // <nil>
    fmt.Println(s == nil)  // true

    s = Circle{Radius: 5}
    fmt.Printf("타입: %T, 값: %v\n", s, s)
    // 타입: main.Circle, 값: {5}

    s = Rectangle{Width: 3, Height: 4}
    fmt.Printf("타입: %T, 값: %v\n", s, s)
    // 타입: main.Rectangle, 값: {3 4}

    // 주의: 인터페이스 안의 nil 포인터는 nil 인터페이스가 아니다
    var c *Circle // nil 포인터
    s = c
    fmt.Println(s == nil) // false! 인터페이스는 (*Circle, nil)을 가지고 있다
}
```

---

## 2. 암시적 만족

### 2.1 "implements" 키워드 없음

```go
// 이 인터페이스는 fmt 패키지에 존재한다
// type Stringer interface {
//     String() string
// }

type Temperature struct {
    Celsius float64
}

// Temperature는 fmt.Stringer를 암시적으로 구현한다
func (t Temperature) String() string {
    return fmt.Sprintf("%.1f°C", t.Celsius)
}

func main() {
    t := Temperature{Celsius: 36.6}
    fmt.Println(t) // "36.6°C" — fmt.Println이 String()을 호출한다

    // 선언이 필요 없다 — Temperature는 String() string 메서드가 있으므로
    // Stringer를 만족한다
}
```

### 2.2 컴파일 타임 검증

```go
// 컴파일 타임에 타입이 인터페이스를 만족하는지 확인한다
var _ Shape = Circle{}     // Circle이 Shape를 구현하지 않으면 컴파일 에러
var _ Shape = (*Circle)(nil) // 포인터 리시버로 확인

// 이것은 라이브러리에서 사용되는 일반적인 Go 관용구이다
var _ io.Reader = (*MyReader)(nil)
var _ io.Writer = (*MyWriter)(nil)
```

### 2.3 생산자와 소비자의 디커플링

```go
// producer.go — 구체적 타입을 정의한다
type FileStore struct {
    BasePath string
}

func (fs FileStore) Save(key string, data []byte) error {
    return os.WriteFile(filepath.Join(fs.BasePath, key), data, 0644)
}

func (fs FileStore) Load(key string) ([]byte, error) {
    return os.ReadFile(filepath.Join(fs.BasePath, key))
}

// consumer.go — 필요한 인터페이스를 정의한다
type Store interface {
    Save(key string, data []byte) error
    Load(key string) ([]byte, error)
}

// 소비자는 FileStore, MemoryStore, S3Store 등을 알 필요가 없다
type App struct {
    store Store
}

func (a *App) SaveConfig(config []byte) error {
    return a.store.Save("config.json", config)
}
```

---

## 3. 일반적인 표준 인터페이스

### 3.1 fmt.Stringer

```go
type Point struct {
    X, Y float64
}

func (p Point) String() string {
    return fmt.Sprintf("(%g, %g)", p.X, p.Y)
}

// 이제 fmt.Println(p)가 이 메서드를 사용한다
```

### 3.2 io.Reader와 io.Writer

Go 표준 라이브러리에서 가장 중요한 인터페이스이다.

```go
import (
    "bytes"
    "io"
    "os"
    "strings"
)

// io.Reader: Read(p []byte) (n int, err error)
// io.Writer: Write(p []byte) (n int, err error)

func main() {
    // strings.Reader는 io.Reader를 구현한다
    r := strings.NewReader("Hello, World!")

    // reader에서 writer(stdout)로 복사한다
    io.Copy(os.Stdout, r)
    fmt.Println()

    // bytes.Buffer는 Reader와 Writer를 모두 구현한다
    var buf bytes.Buffer
    buf.WriteString("Hello ")
    buf.WriteString("Buffer!")
    fmt.Println(buf.String())

    // io.Reader를 받는 모든 함수는 파일, 네트워크, 문자열 등과 동작한다
    data, _ := io.ReadAll(strings.NewReader("이 모두를 읽는다"))
    fmt.Println(string(data))
}

// 모든 io.Reader와 동작하는 함수를 작성한다
func countLines(r io.Reader) (int, error) {
    scanner := bufio.NewScanner(r)
    count := 0
    for scanner.Scan() {
        count++
    }
    return count, scanner.Err()
}

// 파일, 문자열, 네트워크 연결 등과 동작한다
// lines, _ := countLines(os.Stdin)
// lines, _ := countLines(strings.NewReader("a\nb\nc"))
// lines, _ := countLines(file)
```

### 3.3 sort.Interface

```go
import "sort"

type Person struct {
    Name string
    Age  int
}

// ByAge는 sort.Interface를 구현한다
type ByAge []Person

func (a ByAge) Len() int           { return len(a) }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func main() {
    people := []Person{
        {"Alice", 30},
        {"Bob", 25},
        {"Carol", 28},
    }

    sort.Sort(ByAge(people))
    fmt.Println(people) // 나이순 정렬: Bob, Carol, Alice

    // 현대적 대안: sort.Slice (Go 1.8+)
    sort.Slice(people, func(i, j int) bool {
        return people[i].Name < people[j].Name
    })
    fmt.Println(people) // 이름순 정렬: Alice, Bob, Carol
}
```

### 3.4 error 인터페이스

```go
// error 인터페이스는 간단하다:
// type error interface {
//     Error() string
// }

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed on %s: %s", e.Field, e.Message)
}

func validateAge(age int) error {
    if age < 0 || age > 150 {
        return &ValidationError{
            Field:   "age",
            Message: fmt.Sprintf("must be 0-150, got %d", age),
        }
    }
    return nil
}
```

---

## 4. 빈 인터페이스와 타입 단언

### 4.1 빈 인터페이스 (any)

```go
// any는 interface{}의 별칭이다 (Go 1.18+)
func printAnything(val any) {
    fmt.Printf("타입: %T, 값: %v\n", val, val)
}

func main() {
    printAnything(42)
    printAnything("hello")
    printAnything([]int{1, 2, 3})
    printAnything(nil)

    // any의 슬라이스 — Java의 Object[]와 비슷하다
    mixed := []any{1, "two", 3.0, true}
    for _, v := range mixed {
        fmt.Println(v)
    }
}
```

### 4.2 타입 단언

```go
func main() {
    var val any = "hello"

    // 타입 단언(type assertion) — 구체적 타입을 추출한다
    s := val.(string)
    fmt.Println(s) // "hello"

    // 잘못된 타입이면 패닉:
    // n := val.(int) // 패닉: interface conversion

    // "comma ok"를 사용한 안전한 단언
    s, ok := val.(string)
    if ok {
        fmt.Println("문자열이다:", s)
    }

    n, ok := val.(int)
    if !ok {
        fmt.Println("정수가 아니다") // 이것이 출력됨
    }
    fmt.Println(n) // 0 (영값)
}
```

### 4.3 타입 스위치

```go
func describe(val any) string {
    switch v := val.(type) {
    case nil:
        return "nil"
    case int:
        return fmt.Sprintf("정수: %d", v)
    case float64:
        return fmt.Sprintf("실수: %.2f", v)
    case string:
        return fmt.Sprintf("문자열: %q (길이=%d)", v, len(v))
    case bool:
        return fmt.Sprintf("불리언: %t", v)
    case []int:
        return fmt.Sprintf("정수 슬라이스: %v (길이=%d)", v, len(v))
    case Shape:
        return fmt.Sprintf("넓이가 %.2f인 도형", v.Area())
    default:
        return fmt.Sprintf("알 수 없음: %T", v)
    }
}

func main() {
    values := []any{42, 3.14, "hello", true, nil, []int{1, 2}}
    for _, v := range values {
        fmt.Println(describe(v))
    }
}
```

---

## 5. 인터페이스 합성

### 5.1 인터페이스 임베딩

```go
// 작고 집중된 인터페이스
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

// 합성된 인터페이스
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// 표준 라이브러리의 실제 예:
// type ReadWriteCloser interface {
//     io.Reader
//     io.Writer
//     io.Closer
// }
```

### 5.2 인터페이스를 받고 구조체를 반환하라

```go
// 좋음: 인터페이스를 받는다 — 호출자에게 유연하다
func Process(r io.Reader) error {
    data, err := io.ReadAll(r)
    if err != nil {
        return err
    }
    fmt.Println(string(data))
    return nil
}

// 좋음: 구체적 타입을 반환한다 — 호출자가 전체 기능을 얻는다
func NewBuffer() *bytes.Buffer {
    return &bytes.Buffer{}
}

// 호출자는 이를 io.Writer, io.Reader, 또는 *bytes.Buffer로 사용할 수 있다
```

### 5.3 인터페이스 분리

```go
// 나쁨: 큰 인터페이스 — 구현, 테스트, 모킹이 어렵다
type Repository interface {
    Create(item Item) error
    Read(id string) (Item, error)
    Update(id string, item Item) error
    Delete(id string) error
    List() ([]Item, error)
    Search(query string) ([]Item, error)
    Count() (int, error)
    Export(w io.Writer) error
}

// 좋음: 작고 집중된 인터페이스
type ItemReader interface {
    Read(id string) (Item, error)
}

type ItemWriter interface {
    Create(item Item) error
    Update(id string, item Item) error
    Delete(id string) error
}

type ItemLister interface {
    List() ([]Item, error)
    Search(query string) ([]Item, error)
}

// 더 필요할 때 합성한다
type ItemStore interface {
    ItemReader
    ItemWriter
}
```

---

## 6. 인터페이스 설계 원칙

### 6.1 인터페이스를 위한 Go 격언

```go
// "인터페이스가 클수록 추상화가 약해진다." — Rob Pike

// 1. 인터페이스를 작게 유지한다 (1-3개 메서드)
type Sizer interface {
    Size() int64
}

// 2. 인터페이스를 구현하는 곳이 아닌 사용하는 곳에서 정의한다
// 소비자가 필요한 것을 정의한다:
type UserService struct {
    repo UserRepository
}

type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(u *User) error
}
// 다른 패키지에서 UserService를 import하지 않고도 이를 구현할 수 있다

// 3. 가능하면 표준 인터페이스를 사용한다 (io.Reader, fmt.Stringer 등)

// 4. 인터페이스를 성급하게 내보내지 않는다
// 구체적 타입으로 시작하고, 다형성이 필요할 때 인터페이스를 추출한다
```

### 6.2 인터페이스 테스트 패턴

```go
// 인터페이스는 모킹(mocking)으로 쉬운 테스트를 가능하게 한다
type EmailSender interface {
    Send(to, subject, body string) error
}

// 프로덕션 구현
type SMTPSender struct {
    Host string
    Port int
}

func (s *SMTPSender) Send(to, subject, body string) error {
    // 실제로 SMTP를 통해 이메일을 보낸다
    return nil
}

// 테스트 모킹
type MockSender struct {
    SentEmails []struct{ To, Subject, Body string }
}

func (m *MockSender) Send(to, subject, body string) error {
    m.SentEmails = append(m.SentEmails, struct{ To, Subject, Body string }{to, subject, body})
    return nil
}

// 테스트에서의 사용:
// sender := &MockSender{}
// service := NewNotificationService(sender)
// service.NotifyUser("alice@example.com", "Hello")
// assert(len(sender.SentEmails) == 1)
```

### 6.3 Stringer 계약

```go
// String()을 구현하면 타입이 어디서나 출력하기 좋아진다
type Duration struct {
    Hours   int
    Minutes int
    Seconds int
}

func (d Duration) String() string {
    return fmt.Sprintf("%02d:%02d:%02d", d.Hours, d.Minutes, d.Seconds)
}

func main() {
    d := Duration{1, 30, 45}
    fmt.Println(d)                // 01:30:45
    fmt.Sprintf("Duration: %s", d) // "Duration: 01:30:45"
    s := "Meeting: " + d.String()  // "Meeting: 01:30:45"
}
```

---

## 7. 요약

### 핵심 포인트

1. **암시적 만족** — `implements` 키워드가 없다. 타입이 메서드를 가지고 있으면 인터페이스를 만족한다.
2. **인터페이스는 메서드 집합으로 만족된다** — 값 타입은 값 리시버, 포인터 타입은 모든 리시버를 가진다.
3. **작은 인터페이스가 강력하다** — `io.Reader`(1개 메서드)는 Go에서 가장 널리 사용되는 인터페이스이다.
4. **인터페이스를 받고 구조체를 반환하라** — 함수는 유연성을 위해 인터페이스를 받고, 사용성을 위해 구체적 타입을 반환해야 한다.
5. **타입 단언과 스위치** — 필요할 때 인터페이스에서 구체적 타입을 안전하게 추출한다.
6. **상속보다 합성** — 작은 인터페이스를 임베딩하여 더 큰 인터페이스를 만든다.
7. **소비자에서 인터페이스를 정의하라** — 인터페이스를 사용하는 패키지가 이를 정의해야 하며, 구현자가 아니다.

### 인터페이스 치트 시트

| 패턴 | 사용 시기 |
|------|----------|
| `any` / `interface{}` | 진정한 제네릭 컨테이너 (Go 1.18+에서는 제네릭 선호) |
| 타입 단언 `v.(T)` | 구체적 타입을 알고 있을 때 |
| 타입 스위치 `v.(type)` | 여러 가능한 타입을 처리할 때 |
| 작은 인터페이스 (1-2개 메서드) | Reader, Writer, Stringer 같은 추상화 |
| 인터페이스 임베딩 | 작은 인터페이스로부터 큰 인터페이스를 구성할 때 |

---

## 연습 문제

### 연습 1: 도형 계산기
`Area()`와 `Perimeter()`를 가진 `Shape` 인터페이스를 정의하라. `Circle`, `Rectangle`, `Triangle`을 구현하라. `LargestShape(shapes []Shape) Shape` 함수를 작성하라.

### 연습 2: 커스텀 정렬기
이름, 급여, 또는 입사일로 정렬할 수 있는 `[]Employee` 타입에 대해 `sort.Interface`를 구현하라.

### 연습 3: 플러그인 시스템
인터페이스를 사용한 간단한 플러그인 시스템을 설계하라. `Name() string`, `Init() error`, `Execute(data any) (any, error)`를 가진 `Plugin` 인터페이스를 정의하라. 두 개의 구체적 플러그인을 만들라.

### 연습 4: 모킹 테스트
`Notifier` 인터페이스에 의존하는 `NotificationService`를 작성하라. 실제 `EmailNotifier`와 테스트용 `MockNotifier`를 만들라. 알림이 전송되는지 확인하는 테스트를 작성하라.

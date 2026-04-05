# 02. 복합 타입

**이전**: [Go 기초](./01_Go_Fundamentals.md) | **다음**: [함수와 메서드](./03_Functions_and_Methods.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 고정 크기의 배열(array)을 선언하고 조작할 수 있다
2. 동적 컬렉션을 위한 슬라이스(slice)를 사용하고, 용량(capacity)과 길이(length)를 이해할 수 있다
3. 키-값 저장을 위한 맵(map)을 생성하고 조작할 수 있다
4. 이름이 있는 필드를 가진 구조체(struct)를 정의하고 인스턴스화할 수 있다
5. 각 타입의 값 의미론(value semantics)과 참조 의미론(reference semantics)의 차이를 이해할 수 있다

---

Go의 복합 타입(composite type)은 레슨 01의 기본 타입을 기반으로 강력한 데이터 구조를 만든다. 배열은 고정 크기 시퀀스를 제공하고, 슬라이스는 효율적인 메모리 관리를 갖춘 동적 크기 조절을 추가하며, 맵은 키-값 조회를 위한 해시 테이블 성능을 제공하고, 구조체는 자신만의 집합 타입을 정의할 수 있게 한다.

## 목차
1. [배열](#1-배열)
2. [슬라이스](#2-슬라이스)
3. [슬라이스 내부 구조](#3-슬라이스-내부-구조)
4. [맵](#4-맵)
5. [구조체](#5-구조체)
6. [구조체 임베딩과 태그](#6-구조체-임베딩과-태그)
7. [요약](#7-요약)

---

## 1. 배열

### 1.1 배열 기초

Go의 배열은 타입의 일부인 **고정 크기**를 가진다. `[3]int`와 `[5]int`는 다른 타입이다.

```go
package main

import "fmt"

func main() {
    // 명시적 크기로 선언
    var numbers [5]int
    fmt.Println(numbers) // [0 0 0 0 0] — 영값

    // 값으로 초기화
    primes := [5]int{2, 3, 5, 7, 11}
    fmt.Println(primes)

    // 컴파일러가 크기를 세도록 한다
    vowels := [...]string{"a", "e", "i", "o", "u"}
    fmt.Println(vowels, len(vowels)) // 5

    // 접근 및 수정
    primes[0] = 1
    fmt.Println(primes[0]) // 1

    // 부분 초기화
    sparse := [10]int{1: 10, 5: 50, 9: 90}
    fmt.Println(sparse) // [0 10 0 0 0 50 0 0 0 90]
}
```

### 1.2 배열 순회

```go
func main() {
    colors := [4]string{"red", "green", "blue", "yellow"}

    // range 기반
    for i, color := range colors {
        fmt.Printf("%d: %s\n", i, color)
    }

    // 전통적 인덱스 기반
    for i := 0; i < len(colors); i++ {
        fmt.Println(colors[i])
    }
}
```

### 1.3 배열은 값 타입이다

배열은 **값 타입(value type)**이다 — 할당하거나 전달하면 전체가 복사된다.

```go
func main() {
    a := [3]int{1, 2, 3}
    b := a     // b는 복사본이다
    b[0] = 99
    fmt.Println(a) // [1 2 3] — 변경되지 않음
    fmt.Println(b) // [99 2 3]

    // 같은 타입과 크기의 배열은 비교할 수 있다
    x := [3]int{1, 2, 3}
    y := [3]int{1, 2, 3}
    fmt.Println(x == y) // true
}

// 배열을 함수에 전달하면 복사된다 (큰 배열에서는 비용이 크다)
func sum(arr [1000]int) int {
    total := 0
    for _, v := range arr {
        total += v
    }
    return total
}
```

---

## 2. 슬라이스

### 2.1 슬라이스 기초

슬라이스(slice)는 Go의 동적 배열에 대한 답이다. 실제로 배열보다 훨씬 많이 사용된다.

```go
package main

import "fmt"

func main() {
    // 슬라이스 리터럴 (크기 지정 없음)
    fruits := []string{"apple", "banana", "cherry"}
    fmt.Println(fruits)
    fmt.Println(len(fruits)) // 3
    fmt.Println(cap(fruits)) // 3

    // make — 길이와 용량을 지정하여 슬라이스 생성
    nums := make([]int, 5)     // len=5, cap=5, 모두 0
    buf := make([]int, 0, 10)  // len=0, cap=10, 비어 있지만 미리 할당됨

    fmt.Println(nums)
    fmt.Println(buf, len(buf), cap(buf))

    // append — 필요에 따라 슬라이스를 확장한다
    buf = append(buf, 1, 2, 3)
    fmt.Println(buf) // [1 2 3]

    // 여러 요소 추가
    more := []int{4, 5, 6}
    buf = append(buf, more...)
    fmt.Println(buf) // [1 2 3 4 5 6]

    // nil 슬라이스 vs 빈 슬라이스
    var nilSlice []int          // nil, len=0, cap=0
    emptySlice := []int{}       // nil이 아님, len=0, cap=0
    fmt.Println(nilSlice == nil) // true
    fmt.Println(emptySlice == nil) // false
    // 둘 다 append, len, cap, range와 동일하게 동작한다
}
```

### 2.2 슬라이싱 연산

```go
func main() {
    s := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    // s[low:high] — low부터 high-1까지의 요소
    fmt.Println(s[2:5])  // [2 3 4]
    fmt.Println(s[:3])   // [0 1 2]     — 처음부터
    fmt.Println(s[7:])   // [7 8 9]     — 끝까지
    fmt.Println(s[:])    // 전체 슬라이스 — 참조의 복사

    // 배열에서 슬라이싱
    arr := [5]int{10, 20, 30, 40, 50}
    slice := arr[1:4] // [20 30 40]
    fmt.Println(slice)

    // 경고: 슬라이스는 기본 배열을 공유한다!
    slice[0] = 999
    fmt.Println(arr) // [10 999 30 40 50] — 배열이 수정됨!

    // 3-인덱스 슬라이스: s[low:high:cap] — 용량을 제한한다
    limited := s[2:5:5] // len=3, cap=3 (8 대신)
    fmt.Println(len(limited), cap(limited))

    // copy — 독립적인 복사본을 생성한다
    src := []int{1, 2, 3}
    dst := make([]int, len(src))
    copied := copy(dst, src)
    fmt.Println(dst, copied)
    dst[0] = 99
    fmt.Println(src) // [1 2 3] — 영향 없음
}
```

### 2.3 일반적인 슬라이스 패턴

```go
func main() {
    // 인덱스 i의 요소 제거
    s := []int{0, 1, 2, 3, 4}
    i := 2
    s = append(s[:i], s[i+1:]...)
    fmt.Println(s) // [0 1 3 4]

    // 인덱스 i에 요소 삽입
    s = []int{0, 1, 3, 4}
    i = 2
    s = append(s[:i], append([]int{2}, s[i:]...)...)
    fmt.Println(s) // [0 1 2 3 4]

    // 필터 — 조건에 맞는 요소로 새 슬라이스를 생성한다
    nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    evens := filter(nums, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4 6 8 10]

    // 중복 제거 (정렬된 슬라이스)
    sorted := []int{1, 1, 2, 2, 3, 3, 3}
    unique := deduplicate(sorted)
    fmt.Println(unique) // [1 2 3]
}

func filter(s []int, pred func(int) bool) []int {
    var result []int
    for _, v := range s {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

func deduplicate(s []int) []int {
    if len(s) == 0 {
        return s
    }
    result := []int{s[0]}
    for _, v := range s[1:] {
        if v != result[len(result)-1] {
            result = append(result, v)
        }
    }
    return result
}
```

---

## 3. 슬라이스 내부 구조

### 3.1 메모리 레이아웃

슬라이스는 세 단어 구조체이다: **포인터(pointer)**, **길이(length)**, **용량(capacity)**.

```
슬라이스 헤더 (64비트에서 24바이트):
┌──────────┬────────┬──────────┐
│ 포인터   │ 길이   │ 용량     │
└──────────┴────────┴──────────┘
     │
     ▼
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ . │ . │ . │  기본 배열
└───┴───┴───┴───┴───┴───┴───┴───┘
```

```go
func main() {
    s := make([]int, 3, 8)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s)

    // 용량 내에서 추가 — 재할당 없음
    s = append(s, 1)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s) // 같은 포인터

    // 용량 초과 추가 — 재할당 (새 포인터)
    s = append(s, 2, 3, 4, 5, 6)
    fmt.Printf("len=%d cap=%d ptr=%p\n", len(s), cap(s), s) // 새 포인터!
}
```

### 3.2 증가 전략

```go
func main() {
    var s []int
    prev := cap(s)
    for i := 0; i < 20; i++ {
        s = append(s, i)
        if cap(s) != prev {
            fmt.Printf("len=%-3d cap 변경: %d → %d\n", len(s), prev, cap(s))
            prev = cap(s)
        }
    }
    // 출력에서 용량이 대략 2배로 증가하는 것을 볼 수 있다: 0→1→2→4→8→16→32
}
```

### 3.3 메모리 누수 방지

```go
// 나쁨: 반환된 슬라이스가 전체 원본 배열에 대한 참조를 유지한다
func getFirstThree(data []int) []int {
    return data[:3] // 여전히 큰 기본 배열을 참조한다
}

// 좋음: 복사하여 원본을 해제한다
func getFirstThreeSafe(data []int) []int {
    result := make([]int, 3)
    copy(result, data[:3])
    return result
}
```

---

## 4. 맵

### 4.1 맵 기초

```go
package main

import "fmt"

func main() {
    // 맵 리터럴
    ages := map[string]int{
        "Alice": 30,
        "Bob":   25,
        "Carol": 28,
    }
    fmt.Println(ages)

    // make
    scores := make(map[string]int)

    // 값 설정
    scores["math"] = 95
    scores["science"] = 88

    // 값 가져오기
    fmt.Println(scores["math"]) // 95

    // 존재 확인 — "comma ok" 관용구
    val, ok := ages["Dave"]
    if ok {
        fmt.Println("Dave:", val)
    } else {
        fmt.Println("Dave를 찾을 수 없음") // 이것이 출력됨
    }

    // 삭제
    delete(ages, "Bob")
    fmt.Println(ages)

    // 길이
    fmt.Println(len(ages)) // 2

    // 순회 (순서가 보장되지 않는다)
    for name, age := range ages {
        fmt.Printf("%s: %d\n", name, age)
    }

    // nil 맵 — 읽기는 영값을 반환하지만, 쓰기는 패닉(PANIC)을 발생시킨다
    var m map[string]int // nil
    fmt.Println(m["x"]) // 0 (안전)
    // m["x"] = 1        // 패닉: nil 맵의 항목에 할당
}
```

### 4.2 맵 패턴

```go
func main() {
    // 단어 빈도 카운터
    text := "the quick brown fox jumps over the lazy dog the fox"
    freq := wordFrequency(text)
    for word, count := range freq {
        fmt.Printf("%-10s %d\n", word, count)
    }

    // map[T]struct{}를 사용한 집합(set)
    set := make(map[string]struct{})
    set["apple"] = struct{}{}
    set["banana"] = struct{}{}
    if _, exists := set["apple"]; exists {
        fmt.Println("apple이 집합에 있다")
    }

    // 그룹화
    students := []struct {
        Name  string
        Grade string
    }{
        {"Alice", "A"}, {"Bob", "B"}, {"Carol", "A"}, {"Dave", "B"},
    }
    groups := make(map[string][]string)
    for _, s := range students {
        groups[s.Grade] = append(groups[s.Grade], s.Name)
    }
    fmt.Println(groups) // map[A:[Alice Carol] B:[Bob Dave]]
}

func wordFrequency(text string) map[string]int {
    freq := make(map[string]int)
    for _, word := range strings.Fields(text) {
        freq[word]++
    }
    return freq
}
```

### 4.3 맵은 참조 타입이다

```go
func main() {
    m1 := map[string]int{"a": 1}
    m2 := m1   // m2는 같은 기본 해시 테이블을 가리킨다
    m2["a"] = 99
    fmt.Println(m1["a"]) // 99 — m1이 영향을 받는다!

    // 맵은 ==로 비교할 수 없다
    // reflect.DeepEqual 또는 수동 비교를 사용한다
}

func modifyMap(m map[string]int) {
    m["new"] = 42 // 원본을 수정한다
}
```

---

## 5. 구조체

### 5.1 구조체 기초

```go
package main

import "fmt"

// 구조체 타입 정의
type Person struct {
    Name string
    Age  int
    City string
}

func main() {
    // 리터럴 초기화
    alice := Person{
        Name: "Alice",
        Age:  30,
        City: "Seoul",
    }
    fmt.Println(alice)

    // 위치 기반 (권장하지 않음 — 깨지기 쉽다)
    bob := Person{"Bob", 25, "Tokyo"}
    fmt.Println(bob)

    // 영값 — 모든 필드가 영값이다
    var empty Person
    fmt.Println(empty) // { 0 }

    // 필드 접근 및 수정
    alice.Age = 31
    fmt.Println(alice.Age)

    // 구조체에 대한 포인터
    p := &alice
    p.City = "Busan"        // 자동 역참조
    fmt.Println(alice.City)  // "Busan"

    // new는 영값 구조체에 대한 포인터를 반환한다
    carol := new(Person)
    carol.Name = "Carol"
    fmt.Println(*carol)

    // 익명 구조체
    point := struct {
        X, Y float64
    }{3.0, 4.0}
    fmt.Println(point)
}
```

### 5.2 구조체 비교와 복사

```go
func main() {
    // 구조체는 값 타입이다 — 할당하면 복사된다
    a := Person{Name: "Alice", Age: 30, City: "Seoul"}
    b := a
    b.Age = 31
    fmt.Println(a.Age) // 30 — 변경되지 않음
    fmt.Println(b.Age) // 31

    // 비교 가능한 구조체 (모든 필드가 비교 가능해야 한다)
    c := Person{Name: "Alice", Age: 30, City: "Seoul"}
    fmt.Println(a == c) // true

    // 슬라이스/맵 필드가 있는 구조체는 ==로 비교할 수 없다
    type Team struct {
        Name    string
        Members []string // 슬라이스는 비교할 수 없다
    }
    // t1 == t2는 컴파일 에러가 된다
}
```

### 5.3 생성자 패턴

```go
type Server struct {
    Host    string
    Port    int
    Timeout time.Duration
    TLS     bool
}

// "생성자" 함수 — Go 규칙은 NewXxx이다
func NewServer(host string, port int) *Server {
    return &Server{
        Host:    host,
        Port:    port,
        Timeout: 30 * time.Second, // 합리적인 기본값
        TLS:     true,
    }
}

func main() {
    s := NewServer("localhost", 8080)
    fmt.Printf("%+v\n", s)
}
```

---

## 6. 구조체 임베딩과 태그

### 6.1 임베딩 (합성)

Go는 상속(inheritance) 대신 임베딩(embedding)을 사용한다. 임베딩된 필드는 "승격(promoted)"된다 — 메서드와 필드에 직접 접근할 수 있다.

```go
type Address struct {
    Street string
    City   string
    Zip    string
}

type Employee struct {
    Name    string
    Address          // 임베딩 — 필드가 승격된다
    Company string
}

func main() {
    emp := Employee{
        Name:    "Alice",
        Address: Address{Street: "123 Main", City: "Seoul", Zip: "04500"},
        Company: "Acme",
    }

    // 승격된 필드에 직접 접근
    fmt.Println(emp.City)        // "Seoul" — Address에서 승격됨
    fmt.Println(emp.Address.City) // "Seoul" — 명시적 접근도 가능하다
}
```

### 6.2 구조체 태그

태그(tag)는 구조체 필드에 첨부되는 메타데이터로, `encoding/json` 같은 패키지가 사용한다.

```go
import "encoding/json"

type User struct {
    ID        int    `json:"id"`
    FirstName string `json:"first_name"`
    LastName  string `json:"last_name"`
    Email     string `json:"email,omitempty"` // 비어 있으면 생략
    Password  string `json:"-"`               // JSON에 절대 포함하지 않음
}

func main() {
    u := User{
        ID:        1,
        FirstName: "Alice",
        LastName:  "Kim",
        Email:     "",
        Password:  "secret",
    }

    data, _ := json.MarshalIndent(u, "", "  ")
    fmt.Println(string(data))
    // {
    //   "id": 1,
    //   "first_name": "Alice",
    //   "last_name": "Kim"
    // }
    // 참고: Email은 생략됨 (비어 있음), Password는 생략됨 (-)
}
```

---

## 7. 요약

### 핵심 포인트

1. **배열은 고정 크기이다** — 값 타입이며 직접 사용하는 경우가 드물다. 슬라이스를 선호하라.
2. **슬라이스는 참조 타입이다** — 기본 배열을 가리킨다. 슬라이싱 시 공유 메모리에 주의하라.
3. **`append`는 할당할 수 있다** — 용량 초과 시 새로운 더 큰 배열이 할당되고 데이터가 복사된다.
4. **맵은 정렬되지 않는다** — 순회 순서가 무작위이다. 순서가 필요하면 정렬된 키를 사용하라.
5. **구조체는 값 타입이다** — 할당하면 모든 필드가 복사된다. 큰 구조체나 변경이 필요하면 포인터를 사용하라.
6. **상속보다 임베딩** — Go는 합성(composition)을 사용한다. 구조체를 임베딩하여 필드와 메서드를 승격시킨다.
7. **구조체 태그** — 직렬화(serialization), 검증(validation), ORM 매핑을 위한 메타데이터이다. `json` 태그가 가장 일반적이다.

### 타입 요약

| 타입 | 값/참조 | 영값 | 비교 가능 | 가변 |
|------|---------|------|----------|------|
| 배열 | 값 | `[0, 0, ...]` | 예 | 예 |
| 슬라이스 | 참조 | `nil` | 아니오 | 예 |
| 맵 | 참조 | `nil` | 아니오 | 예 |
| 구조체 | 값 | 모든 필드 영값 | 모든 필드가 비교 가능하면 | 예 |

---

## 연습 문제

### 연습 1: 행렬 연산
`[3][3]float64`를 사용한 3x3 행렬 타입을 만들라. `add`, `transpose`, `multiply` 함수를 구현하라.

### 연습 2: 스택 구현
슬라이스를 사용하여 `Push`, `Pop`, `Peek` 연산을 가진 스택을 구현하라. 빈 스택을 우아하게 처리하라.

### 연습 3: 학생 기록
이름, 성적(슬라이스)을 가진 `Student` 구조체를 정의하고 GPA를 계산하라. 학생 슬라이스에서 상위 N명을 찾는 함수를 만들라.

### 연습 4: 단어 인덱스
텍스트에서 단어 인덱스를 구축하라: `map[string][]int` — 키는 단어이고 값은 각 단어가 나타나는 줄 번호이다.

# 레슨 21: C++23 기능

**이전**: [CMake와 빌드 시스템](./20_CMake_and_Build_Systems.md) | **다음**: [외부 라이브러리](./22_External_Libraries.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 예외 없이 `std::expected`를 사용하여 오류를 처리하고, 임시방편적인 오류 코드와 `std::optional` 우회책을 대체한다
2. 추론 `this`(deducing `this`)를 적용하여 CRTP 패턴을 단순화하고, 값 범주(value category)를 인식하는 멤버 함수를 작성한다
3. `std::print`와 `std::println`을 사용하여 형식화된 출력을 작성하고, `printf`와 `iostream`의 타입 안전한 고성능 대안으로 활용한다
4. `std::mdspan`을 사용하여 데이터를 소유하지 않는 다차원 배열 뷰를 만들고, C 배열 및 라이브러리와 제로 복사(zero-copy) 상호운용을 가능하게 한다
5. `std::generator`로 지연 생성기(lazy generator)를 만들고, 코루틴(coroutine) 문법으로 시퀀스를 온디맨드(on-demand) 생성한다

---

## 목차

1. [C++23 한눈에 보기](#1-c23-한눈에-보기)
2. [`std::expected` — 올바른 오류 처리](#2-stdexpected--올바른-오류-처리)
3. [추론 `this`(Deducing `this`)](#3-추론-thisdeducing-this)
4. [`std::print`와 `std::println`](#4-stdprint와-stdprintln)
5. [`std::mdspan` — 다차원 뷰](#5-stdmdspan--다차원-뷰)
6. [`std::generator` — 지연 코루틴 시퀀스](#6-stdgenerator--지연-코루틴-시퀀스)
7. [기타 주목할 만한 C++23 기능](#7-기타-주목할-만한-c23-기능)
8. [연습문제(Exercises)](#8-연습문제exercises)

---

## 1. C++23 한눈에 보기

C++23은 최신 출판된 ISO 표준(ISO/IEC 14882:2024)입니다. C++20이 "빅 포(big four)"(개념, 범위, 코루틴, 모듈)를 도입했다면, C++23은 거친 부분을 다듬고 실용적인 유틸리티를 추가합니다.

| 범주 | 주요 추가 사항 |
|------|---------------|
| 오류 처리 | `std::expected` |
| 언어 | 추론 `this`, `if consteval`, `static operator()` |
| 입출력 | `std::print`, `std::println` |
| 컨테이너 | `std::flat_map`, `std::flat_set`, `std::mdspan` |
| 범위(Ranges) | `std::views::zip`, `chunk`, `slide`, `cartesian_product`, `enumerate` |
| 코루틴(Coroutines) | `std::generator` |
| 유틸리티 | `std::stacktrace`, `std::move_only_function` |

**컴파일러 지원**: GCC 14+, Clang 18+, MSVC 19.38+이 대부분의 기능을 지원합니다. 자세한 내용은 [cppreference 컴파일러 지원](https://en.cppreference.com/w/cpp/compiler_support) 페이지를 확인하세요.

---

## 2. `std::expected` — 올바른 오류 처리

### 문제점

C++에는 여러 오류 처리 방식이 있으며, 각각 절충점이 있습니다:

| 방식 | 단점 |
|------|------|
| 예외(Exceptions) | 런타임 오버헤드, 제어 흐름 추론이 어려움 |
| 오류 코드(Error codes) | 무시하기 쉬움, 타입 안전한 오류 페이로드 없음 |
| `std::optional` | 오류 정보 없음 — "값 또는 없음"만 표현 가능 |
| `std::variant` | 장황하고, 인덱스 기반 접근 방식 |

### `std::expected<T, E>`

`std::expected`는 타입 `T`의 값 또는 타입 `E`의 오류를 보유합니다:

```cpp
#include <expected>
#include <string>
#include <charconv>

enum class ParseError { empty_input, invalid_format, overflow };

std::expected<int, ParseError> parse_int(std::string_view sv) {
    if (sv.empty())
        return std::unexpected(ParseError::empty_input);

    int result{};
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);

    if (ec == std::errc::result_out_of_range)
        return std::unexpected(ParseError::overflow);
    if (ec != std::errc{} || ptr != sv.data() + sv.size())
        return std::unexpected(ParseError::invalid_format);

    return result;  // implicitly wraps in expected
}

void demo() {
    auto result = parse_int("42");
    if (result) {
        // Access value with * or .value()
        std::println("Parsed: {}", *result);
    }

    auto err = parse_int("abc");
    if (!err) {
        // Access error with .error()
        std::println("Error code: {}", static_cast<int>(err.error()));
    }
}
```

### 모나드 연산(Monadic Operations)

`std::expected`는 연쇄(chaining)를 위한 `and_then`, `or_else`, `transform`을 지원합니다:

```cpp
auto read_config(std::string_view path)
    -> std::expected<Config, Error>;

auto validate(Config cfg)
    -> std::expected<Config, Error>;

auto apply(Config cfg)
    -> std::expected<void, Error>;

// Chain operations — each step propagates errors automatically
auto result = read_config("/etc/app.conf")
    .and_then(validate)
    .and_then(apply);

// Equivalent to nested if-else without the indentation
```

---

## 3. 추론 `this`(Deducing `this`)

### 문제점

C++23 이전에는 값 범주(lvalue vs rvalue)에 따라 다르게 동작하는 멤버 함수를 작성하려면 코드를 중복해야 했습니다:

```cpp
// Pre-C++23: two overloads for const/non-const
class Widget {
    std::string name_;
public:
    const std::string& name() const& { return name_; }
    std::string name() && { return std::move(name_); }
};
```

### 명시적 객체 매개변수(Explicit Object Parameter)

C++23은 첫 번째 매개변수를 명시적인 `this`로 선언할 수 있게 합니다:

```cpp
class Widget {
    std::string name_;
public:
    // Single function handles all value categories
    template<typename Self>
    auto&& name(this Self&& self) {
        return std::forward<Self>(self).name_;
    }
};

// Usage:
Widget w{"hello"};
auto& ref = w.name();           // lvalue: returns const string&
auto val = std::move(w).name(); // rvalue: returns string&&
```

### 단순화된 CRTP

CRTP(Curiously Recurring Template Pattern, 기묘하게 반복되는 템플릿 패턴)가 훨씬 간단해집니다:

```cpp
// Pre-C++23 CRTP
template<typename Derived>
class Addable {
public:
    Derived operator+(const Derived& other) const {
        Derived result = static_cast<const Derived&>(*this);
        result += other;
        return result;
    }
};

class Vec2 : public Addable<Vec2> { /* ... */ };

// C++23: no template parameter needed
class Addable23 {
public:
    template<typename Self>
    Self operator+(this Self self, const Self& other) {
        self += other;
        return self;
    }
};

class Vec2 : public Addable23 { /* ... */ };
```

### 재귀 람다(Recursive Lambdas)

추론 `this`는 재귀 람다를 자연스럽게 만들어줍니다:

```cpp
auto fibonacci = [](this auto self, int n) -> int {
    if (n <= 1) return n;
    return self(n - 1) + self(n - 2);
};

std::println("{}", fibonacci(10));  // 55
```

---

## 4. `std::print`와 `std::println`

`std::print`는 `std::format`을 표준 출력에 직접 연결하여, 타입이 안전하지 않은 `printf`와 장황한 `iostream` 둘 다를 대체합니다:

```cpp
#include <print>

void demo() {
    int x = 42;
    double pi = 3.14159;
    std::string name = "C++23";

    // Type-safe, compile-time checked format strings
    std::println("Hello, {}!", name);
    std::println("x = {}, pi = {:.2f}", x, pi);

    // Alignment and fill
    std::println("{:>10}", "right");    //      right
    std::println("{:*^10}", "center");  //  **center**

    // Print to any FILE* or ostream
    std::print(stderr, "Error: {}\n", "something went wrong");
}
```

**왜 `std::format` + `std::cout`를 쓰지 않나요?**

`std::print`는 더 효율적입니다 — 중간 `std::string`을 생성하지 않고 출력 스트림에 직접 씁니다. 또한 유니코드(Unicode)를 올바르게 처리하고 플러시(flush)도 적절히 수행합니다.

---

## 5. `std::mdspan` — 다차원 뷰

`std::mdspan`은 연속된 메모리에 대한 비소유(non-owning) 다차원 뷰를 제공합니다. 다차원 `std::span`이라고 생각하면 됩니다.

```cpp
#include <mdspan>
#include <vector>

void demo() {
    std::vector<double> data(12);
    std::iota(data.begin(), data.end(), 1.0);

    // View as 3×4 matrix (row-major by default)
    std::mdspan mat(data.data(), 3, 4);

    // Access with multidimensional indexing
    for (std::size_t i = 0; i < mat.extent(0); ++i) {
        for (std::size_t j = 0; j < mat.extent(1); ++j) {
            std::print("{:4.0f}", mat[i, j]);  // C++23 multi-subscript
        }
        std::println();
    }
    // Output:
    //    1   2   3   4
    //    5   6   7   8
    //    9  10  11  12
}
```

### 레이아웃 정책(Layout Policies)

```cpp
// Column-major (Fortran style) — for BLAS/LAPACK interop
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_left> col_major(data.data(), 3, 4);

// Custom stride
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_stride> strided(
    data.data(),
    std::layout_stride::mapping(
        std::dextents<size_t, 2>(3, 4),
        std::array<size_t, 2>{4, 1}  // row stride=4, col stride=1
    )
);
```

### C와의 제로 복사 상호운용(Zero-Copy Interop with C)

```cpp
// Wrap a C array without copying
extern "C" void legacy_compute(double* matrix, int rows, int cols);

void modern_wrapper(std::mdspan<double, std::dextents<size_t, 2>> mat) {
    // Pass underlying pointer to C code
    legacy_compute(mat.data_handle(),
                   static_cast<int>(mat.extent(0)),
                   static_cast<int>(mat.extent(1)));
}
```

---

## 6. `std::generator` — 지연 코루틴 시퀀스

`std::generator<T>`는 표준 라이브러리의 코루틴 기반 지연 시퀀스 생성기입니다:

```cpp
#include <generator>
#include <ranges>

// Infinite sequence of Fibonacci numbers
std::generator<long long> fibonacci() {
    long long a = 0, b = 1;
    while (true) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

void demo() {
    // Take first 10 Fibonacci numbers
    for (auto n : fibonacci() | std::views::take(10)) {
        std::print("{} ", n);
    }
    // Output: 0 1 1 2 3 5 8 13 21 34
}
```

### 트리 순회(Tree Traversal)

```cpp
struct TreeNode {
    int value;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
};

std::generator<int> inorder(TreeNode* node) {
    if (!node) co_return;
    co_yield std::ranges::elements_of(inorder(node->left));
    co_yield node->value;
    co_yield std::ranges::elements_of(inorder(node->right));
}
```

### 반복자와의 비교(Compared to Iterators)

```cpp
// Traditional iterator: ~50 lines of boilerplate
// std::generator: 5 lines of clear, sequential logic
// Both produce the same lazy, on-demand sequence
```

---

## 7. 기타 주목할 만한 C++23 기능

### `std::flat_map`과 `std::flat_set`

연속 배열로 지원되는 캐시 친화적(cache-friendly) 정렬 컨테이너:

```cpp
#include <flat_map>
std::flat_map<std::string, int> scores;
scores["Alice"] = 95;
// Internally: sorted vector<pair<string,int>>
// Better cache locality than std::map (red-black tree)
```

### 새로운 범위 어댑터(New Range Adaptors)

```cpp
#include <ranges>

std::vector v = {1, 2, 3, 4, 5};

// zip: combine multiple ranges
for (auto [a, b] : std::views::zip(v, v | std::views::reverse)) {
    std::println("{} {}", a, b);  // 1 5, 2 4, 3 3, ...
}

// enumerate: index + value (like Python's enumerate)
for (auto [i, val] : std::views::enumerate(v)) {
    std::println("[{}] = {}", i, val);
}

// chunk: split into groups of N
for (auto chunk : v | std::views::chunk(2)) {
    // chunk = {1,2}, {3,4}, {5}
}

// slide: sliding window
for (auto window : v | std::views::slide(3)) {
    // window = {1,2,3}, {2,3,4}, {3,4,5}
}
```

### `if consteval`

```cpp
consteval int compile_time_only(int x) { return x * 2; }

constexpr int flexible(int x) {
    if consteval {
        return compile_time_only(x);  // only at compile time
    } else {
        return x * 2;  // fallback at runtime
    }
}
```

### `static operator()`와 `static operator[]`

```cpp
struct Multiply {
    static int operator()(int a, int b) { return a * b; }
};
// No implicit 'this' pointer → potentially more efficient
```

---

## 8. 연습문제(Exercises)

### 연습문제 1: `std::expected`를 이용한 오류 파이프라인

`std::expected`를 사용하여 데이터 처리 파이프라인을 구축하세요:
1. `read_file(path) → expected<string, Error>` — "파일" 읽기 (모의)
2. `parse_json(str) → expected<Config, Error>` — "JSON" 파싱 (모의)
3. `validate(Config) → expected<Config, Error>` — 필수 필드 확인

`and_then`으로 연쇄하고, 모든 오류 케이스를 처리하세요.

### 연습문제 2: 추론 `this`로 CRTP 교체

CRTP 기반 `Printable<Derived>` 믹스인(mixin)을 추론 `this`를 사용하도록 리팩터링하세요. 믹스인은 파생 클래스의 `to_string()`을 호출하는 `print()` 메서드를 제공해야 합니다. 두 개의 서로 다른 파생 클래스로 테스트하세요.

### 연습문제 3: `std::mdspan`을 이용한 행렬 연산

`std::mdspan`을 사용하여 두 행렬을 곱하는 함수를 작성하세요:
- `multiply(mdspan<double, dextents<size_t,2>> A, mdspan<double, dextents<size_t,2>> B, mdspan<double, dextents<size_t,2>> C)`
- 행 우선(row-major)과 열 우선(column-major) 레이아웃 모두 처리하세요

### 연습문제 4: 지연 시퀀스 생성기

`std::generator`를 사용하여 다음 생성기를 구현하세요:
1. `primes()` — 무한 소수 시퀀스
2. `flatten(vector<vector<int>>)` — 중첩 컨테이너 평탄화(flatten)
3. `interleave(gen1, gen2)` — 두 생성기를 교대로 전환

### 연습문제 5: 범위 파이프라인(Range Pipeline)

C++23 범위 어댑터를 사용하여 단일 파이프라인으로 다음을 해결하세요:
- 문자열 벡터를 받아, 열거(enumerate)하고, 길이가 3보다 큰 것을 필터링하고, 2개씩 청크(chunk)로 묶은 뒤, 각 청크를 `"[idx] word, [idx] word"` 형식으로 포맷하세요.

---

## 내비게이션

**이전**: [CMake와 빌드 시스템](./20_CMake_and_Build_Systems.md) | **다음**: [외부 라이브러리](./22_External_Libraries.md)

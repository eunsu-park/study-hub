# C++23 기능

**이전**: [디자인 패턴: 행위 및 C++ 이디엄](./15_Design_Patterns_Behavioral_Idioms.md) | **다음**: [외부 라이브러리와 고급 빌드 시스템](./17_External_Libraries_and_Build.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 예외 없이 오류를 처리하기 위해 `std::expected`를 사용하여 임시 오류 코드와 `std::optional` 우회를 대체할 수 있다
2. CRTP 패턴을 단순화하고 값 카테고리 인식 멤버 함수를 작성하기 위해 추론 `this`(deducing this)를 적용할 수 있다
3. `printf`와 `iostream`을 대체하는 타입 안전하고 효율적인 `std::print`와 `std::println`으로 포맷된 출력을 작성할 수 있다
4. 데이터를 소유하지 않는 다차원 배열 뷰를 위해 `std::mdspan`을 사용하여 C 배열 및 라이브러리와 제로 카피 상호운용을 할 수 있다
5. 코루틴 구문을 사용하여 요청 시 시퀀스를 생성하는 `std::generator`로 지연 제너레이터를 만들 수 있다

---

C++23은 최신 게시된 ISO 표준(ISO/IEC 14882:2024)입니다. C++20이 "빅 포"(컨셉, 레인지, 코루틴, 모듈)를 도입했다면, C++23은 거친 부분을 다듬고 일상적인 코딩을 더 쉽고 안전하게 만드는 실용적인 유틸리티를 추가합니다.

---

## 목차

1. [C++23 한눈에 보기](#1-c23-한눈에-보기)
2. [`std::expected` -- 올바른 오류 처리](#2-stdexpected----올바른-오류-처리)
3. [추론 `this`](#3-추론-this)
4. [`std::print`와 `std::println`](#4-stdprint와-stdprintln)
5. [`std::mdspan` -- 다차원 뷰](#5-stdmdspan----다차원-뷰)
6. [`std::generator` -- 지연 코루틴 시퀀스](#6-stdgenerator----지연-코루틴-시퀀스)
7. [기타 주목할 만한 C++23 기능](#7-기타-주목할-만한-c23-기능)

---

## 1. C++23 한눈에 보기

| 카테고리 | 주요 추가 사항 |
|---------|---------------|
| 오류 처리 | `std::expected` |
| 언어 | 추론 `this`, `if consteval`, `static operator()` |
| I/O | `std::print`, `std::println` |
| 컨테이너 | `std::flat_map`, `std::flat_set`, `std::mdspan` |
| 레인지 | `std::views::zip`, `chunk`, `slide`, `cartesian_product`, `enumerate` |
| 코루틴 | `std::generator` |
| 유틸리티 | `std::stacktrace`, `std::move_only_function` |

**컴파일러 지원**: GCC 14+, Clang 18+, MSVC 19.38+가 대부분의 기능을 지원합니다. 자세한 내용은 [cppreference 컴파일러 지원](https://en.cppreference.com/w/cpp/compiler_support)을 확인하세요.

---

## 2. `std::expected` -- 올바른 오류 처리

### 문제점

C++에는 여러 오류 처리 접근법이 있으며, 각각 트레이드오프가 있습니다:

| 접근법 | 단점 |
|--------|------|
| 예외 | 런타임 오버헤드, 제어 흐름 추론 어려움 |
| 오류 코드 | 무시하기 쉬움, 타입 안전 오류 페이로드 없음 |
| `std::optional` | 오류 정보 없음 -- "값 또는 없음"만 |
| `std::variant` | 장황함, 인덱스 기반 접근 |

### `std::expected<T, E>`

`std::expected`는 타입 `T`의 값 또는 타입 `E`의 오류를 보유합니다:

```cpp
#include <expected>
#include <string>
#include <charconv>
#include <print>

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

    return result;
}

void demo() {
    auto result = parse_int("42");
    if (result) {
        std::println("Parsed: {}", *result);
    }

    auto err = parse_int("abc");
    if (!err) {
        std::println("Error code: {}", static_cast<int>(err.error()));
    }
}
```

### 모나딕 연산

`std::expected`는 체이닝을 위한 `and_then`, `or_else`, `transform`을 지원합니다:

```cpp
auto read_config(std::string_view path)
    -> std::expected<Config, Error>;

auto validate(Config cfg)
    -> std::expected<Config, Error>;

auto apply(Config cfg)
    -> std::expected<void, Error>;

// 연산 체이닝 -- 각 단계가 자동으로 오류를 전파
auto result = read_config("/etc/app.conf")
    .and_then(validate)
    .and_then(apply);
```

---

## 3. 추론 `this`

### 문제점

C++23 이전에는 값 카테고리에 따라 다르게 동작하는 멤버 함수를 작성하려면 코드를 중복해야 했습니다:

```cpp
// Pre-C++23: two overloads
class Widget {
    std::string name_;
public:
    const std::string& name() const& { return name_; }
    std::string name() && { return std::move(name_); }
};
```

### 명시적 객체 매개변수

C++23은 첫 번째 매개변수를 명시적 `this`로 허용합니다:

```cpp
class Widget {
    std::string name_;
public:
    template<typename Self>
    auto&& name(this Self&& self) {
        return std::forward<Self>(self).name_;
    }
};

Widget w{"hello"};
auto& ref = w.name();           // lvalue: returns const string&
auto val = std::move(w).name(); // rvalue: returns string&&
```

### 단순화된 CRTP

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

// C++23: 템플릿 매개변수 불필요
class Addable23 {
public:
    template<typename Self>
    Self operator+(this Self self, const Self& other) {
        self += other;
        return self;
    }
};
```

### 재귀 람다

추론 `this`는 재귀 람다를 자연스럽게 만듭니다:

```cpp
auto fibonacci = [](this auto self, int n) -> int {
    if (n <= 1) return n;
    return self(n - 1) + self(n - 2);
};

std::println("{}", fibonacci(10));  // 55
```

---

## 4. `std::print`와 `std::println`

`std::print`는 `std::format`을 표준 출력으로 직접 가져와, `printf`(타입 안전하지 않음)와 `iostream`(장황함) 모두를 대체합니다:

```cpp
#include <print>

void demo() {
    int x = 42;
    double pi = 3.14159;
    std::string name = "C++23";

    std::println("Hello, {}!", name);
    std::println("x = {}, pi = {:.2f}", x, pi);

    // 정렬과 채움
    std::println("{:>10}", "right");    //      right
    std::println("{:*^10}", "center");  //  **center**

    // stderr로 출력
    std::print(stderr, "Error: {}\n", "something went wrong");
}
```

**왜 `std::format` + `std::cout`가 아닌가?**

`std::print`는 더 효율적입니다 -- 중간 `std::string`을 생성하지 않고 출력 스트림에 직접 씁니다. 또한 유니코드를 올바르게 처리하고 적절하게 플러시합니다.

---

## 5. `std::mdspan` -- 다차원 뷰

`std::mdspan`은 연속 메모리에 대한 비소유 다차원 뷰를 제공합니다. 다차원 `std::span`으로 생각하면 됩니다.

```cpp
#include <mdspan>
#include <vector>
#include <print>

void demo() {
    std::vector<double> data(12);
    std::iota(data.begin(), data.end(), 1.0);

    // 3x4 행렬로 보기 (기본 행 우선)
    std::mdspan mat(data.data(), 3, 4);

    for (std::size_t i = 0; i < mat.extent(0); ++i) {
        for (std::size_t j = 0; j < mat.extent(1); ++j) {
            std::print("{:4.0f}", mat[i, j]);
        }
        std::println();
    }
}
```

### 레이아웃 정책

```cpp
// 열 우선 (Fortran 스타일) -- BLAS/LAPACK 상호운용용
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_left> col_major(data.data(), 3, 4);

// 커스텀 스트라이드
std::mdspan<double, std::dextents<size_t, 2>,
            std::layout_stride> strided(
    data.data(),
    std::layout_stride::mapping(
        std::dextents<size_t, 2>(3, 4),
        std::array<size_t, 2>{4, 1}
    )
);
```

### C와의 제로 카피 상호운용

```cpp
extern "C" void legacy_compute(double* matrix, int rows, int cols);

void modern_wrapper(std::mdspan<double, std::dextents<size_t, 2>> mat) {
    legacy_compute(mat.data_handle(),
                   static_cast<int>(mat.extent(0)),
                   static_cast<int>(mat.extent(1)));
}
```

---

## 6. `std::generator` -- 지연 코루틴 시퀀스

`std::generator<T>`는 표준 라이브러리의 코루틴 기반 지연 시퀀스 제너레이터입니다:

```cpp
#include <generator>
#include <ranges>
#include <print>

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
    for (auto n : fibonacci() | std::views::take(10)) {
        std::print("{} ", n);
    }
    // Output: 0 1 1 2 3 5 8 13 21 34
}
```

### 트리 순회

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

### 수동 반복자와 비교

```cpp
// 수동 반복자: ~50줄의 보일러플레이트 (begin, end, operator++, 등)
// std::generator: 5줄의 명확하고 순차적인 로직
// 둘 다 동일한 지연, 온디맨드 시퀀스를 생성
```

---

## 7. 기타 주목할 만한 C++23 기능

### `std::flat_map`과 `std::flat_set`

연속 배열로 뒷받침되는 캐시 친화적 정렬 컨테이너:

```cpp
#include <flat_map>
std::flat_map<std::string, int> scores;
scores["Alice"] = 95;
// Internally: sorted vector<pair<string,int>>
// std::map (레드-블랙 트리)보다 캐시 지역성이 더 좋음
```

### 새로운 레인지 어댑터

```cpp
#include <ranges>

std::vector v = {1, 2, 3, 4, 5};

// zip: 여러 레인지 결합
for (auto [a, b] : std::views::zip(v, v | std::views::reverse)) {
    std::println("{} {}", a, b);
}

// enumerate: 인덱스 + 값
for (auto [i, val] : std::views::enumerate(v)) {
    std::println("[{}] = {}", i, val);
}

// chunk: N개씩 그룹으로 분할
for (auto chunk : v | std::views::chunk(2)) {
    // {1,2}, {3,4}, {5}
}

// slide: 슬라이딩 윈도우
for (auto window : v | std::views::slide(3)) {
    // {1,2,3}, {2,3,4}, {3,4,5}
}
```

### `if consteval`

```cpp
consteval int compile_time_only(int x) { return x * 2; }

constexpr int flexible(int x) {
    if consteval {
        return compile_time_only(x);
    } else {
        return x * 2;
    }
}
```

### `static operator()`와 `static operator[]`

```cpp
struct Multiply {
    static int operator()(int a, int b) { return a * b; }
};
// 암시적 'this' 포인터 없음 -- 잠재적으로 더 효율적
```

---

## 연습 문제

### 연습 1: `std::expected`를 이용한 오류 파이프라인

`std::expected`를 사용하여 데이터 처리 파이프라인을 구축하세요:
1. `read_file(path) -> expected<string, Error>` -- "파일" 읽기 (모의)
2. `parse_json(str) -> expected<Config, Error>` -- "JSON" 파싱 (모의)
3. `validate(Config) -> expected<Config, Error>` -- 필수 필드 확인

`and_then`으로 체이닝하세요. 모든 오류 케이스를 처리하세요.

### 연습 2: 추론 `this`를 이용한 CRTP 대체

CRTP 기반 `Printable<Derived>` 믹스인을 추론 `this`를 사용하도록 리팩토링하세요. 믹스인은 파생 클래스의 `to_string()`을 호출하는 `print()` 메서드를 제공해야 합니다. 두 개의 다른 파생 클래스로 테스트하세요.

### 연습 3: `std::mdspan`을 이용한 행렬 연산

`std::mdspan`을 사용하여 두 행렬을 곱하는 함수를 작성하세요:
- `multiply(mdspan<double, dextents<size_t,2>> A, mdspan<double, dextents<size_t,2>> B, mdspan<double, dextents<size_t,2>> C)`
- 행 우선과 열 우선 레이아웃 모두 처리

### 연습 4: 지연 시퀀스 제너레이터

`std::generator`를 사용하여 다음 제너레이터를 구현하세요:
1. `primes()` -- 무한 소수 수열
2. `flatten(vector<vector<int>>)` -- 중첩 컨테이너 평탄화
3. `interleave(gen1, gen2)` -- 두 제너레이터 번갈아 가기

### 연습 5: 레인지 파이프라인

C++23 레인지 어댑터를 사용하여 단일 파이프라인으로 해결하세요:
- 문자열 벡터가 주어지면, enumerate하고, 길이 > 3으로 필터링하고, 2개씩 chunk로 나누고, 각 chunk를 `"[idx] word, [idx] word"` 형식으로 포맷하세요.

---

## 다음 단계

C++23은 최신 언어 및 라이브러리 기능을 제공합니다. 마지막 레슨은 외부 라이브러리와 고급 빌드 시스템을 다룹니다 -- 코드를 출하 가능한 소프트웨어로 만드는 실용적인 기술입니다.

- [외부 라이브러리와 고급 빌드 시스템](./17_External_Libraries_and_Build.md)

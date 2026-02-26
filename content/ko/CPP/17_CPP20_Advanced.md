# C++20 심화

**이전**: [멀티스레딩과 동시성](./16_Multithreading_Concurrency.md) | **다음**: [C++ 디자인 패턴](./18_Design_Patterns.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 커스텀 컨셉(Concept)을 정의하고 표준 라이브러리 컨셉을 적용하여 템플릿 매개변수를 제약할 수 있습니다
2. 뷰(Views)(filter, transform, take, drop)를 사용하여 지연 데이터 처리를 위한 레인지(Range) 파이프라인을 구성할 수 있습니다
3. `co_yield`와 `co_return`으로 코루틴(Coroutine)을 구현하여 제너레이터(Generator)와 비동기 태스크를 빌드할 수 있습니다
4. 전통적인 헤더/소스 모델의 대안으로 모듈(Module)을 사용하여 코드를 구성할 수 있습니다
5. 삼방향 비교(Three-way Comparison) 연산자(스페이스십 연산자)를 사용하여 모든 관계 연산자를 자동 생성할 수 있습니다
6. `std::span`, `std::format`, 지정 초기화자(Designated Initializer), `consteval` 등 C++20 유틸리티를 적용할 수 있습니다
7. C++23 기능(`std::expected`, `std::print`, `std::generator`)과 그 용도를 파악할 수 있습니다

---

C++20은 C++11 이후 가장 혁신적인 표준으로, 템플릿, 이터레이션, 코드 구성 방식을 근본적으로 바꿉니다. 컨셉(Concepts)은 난해한 SFINAE 오류를 읽기 쉬운 제약 조건으로 대체합니다. 레인지(Ranges)는 이터레이터 쌍 보일러플레이트를 제거하고 조합 가능한 파이프라인으로 대체합니다. 코루틴(Coroutines)은 스레드 오버헤드 없이 협력적 멀티태스킹을 가능하게 합니다. 이 기능들을 이해하는 것은 단순히 최신 문법을 사용하는 것이 아니라 더 명확하고 안전하며 유지보수하기 쉬운 코드를 작성하는 것입니다.

**난이도**: ⭐⭐⭐⭐⭐

**선수 지식**: 템플릿, 람다, 스마트 포인터

---

## 목차

1. [Concepts](#concepts)
2. [Ranges](#ranges)
3. [Coroutines](#coroutines)
4. [Modules](#modules)
5. [기타 C++20 기능](#기타-c20-기능)
6. [C++23 미리보기](#c23-미리보기)

---

## Concepts

### Concepts란?

템플릿 매개변수에 대한 **제약 조건(Constraint)**을 정의하는 기능입니다. 이전의 SFINAE(Substitution Failure Is Not An Error)보다 훨씬 가독성이 좋습니다.

### 기본 사용법

```cpp
#include <concepts>
#include <iostream>

// concept 정의
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// concept 사용
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// 또는 requires 절 사용
template<typename T>
    requires Numeric<T>
T multiply(T a, T b) {
    return a * b;
}

// 또는 후행 requires
template<typename T>
T divide(T a, T b) requires Numeric<T> {
    return a / b;
}

int main() {
    std::cout << add(1, 2) << "\n";        // OK
    std::cout << add(1.5, 2.5) << "\n";    // OK
    // add("hello", "world");              // 컴파일 에러!
    return 0;
}
```

### 표준 Concepts

```cpp
#include <concepts>

// 타입 관련
std::same_as<T, U>           // T와 U가 같은 타입
std::derived_from<D, B>      // D가 B의 파생 클래스
std::convertible_to<From, To>// From이 To로 변환 가능

// 산술 관련
std::integral<T>             // 정수 타입
std::floating_point<T>       // 부동소수점 타입
std::signed_integral<T>      // 부호 있는 정수
std::unsigned_integral<T>    // 부호 없는 정수

// 비교 관련
std::equality_comparable<T>  // == 연산 가능
std::totally_ordered<T>      // <, >, <=, >= 연산 가능

// 호출 관련
std::invocable<F, Args...>   // F(Args...)가 호출 가능
std::predicate<F, Args...>   // F(Args...)가 bool 반환
```

### 커스텀 Concept 정의

```cpp
#include <concepts>
#include <string>

// 문자열처럼 동작하는 타입
template<typename T>
concept StringLike = requires(T t) {
    { t.length() } -> std::convertible_to<std::size_t>;
    { t.c_str() } -> std::same_as<const char*>;
    { t[0] } -> std::convertible_to<char>;
};

// 컨테이너 concept
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

// 사용
template<Container C>
void printContainer(const C& container) {
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << "\n";
}
```

### requires 표현식

```cpp
// 단순 요구사항
template<typename T>
concept Addable = requires(T a, T b) {
    a + b;  // 이 표현식이 유효해야 함
};

// 타입 요구사항
template<typename T>
concept HasValueType = requires {
    typename T::value_type;
};

// 복합 요구사항
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// 중첩 요구사항
template<typename T>
concept Sortable = requires(T t) {
    requires std::totally_ordered<typename T::value_type>;
    { t.begin() } -> std::random_access_iterator;
};
```

### Concept을 이용한 오버로딩

```cpp
#include <concepts>
#include <iostream>

template<std::integral T>
void print(T value) {
    std::cout << "Integer: " << value << "\n";
}

template<std::floating_point T>
void print(T value) {
    std::cout << "Float: " << value << "\n";
}

template<typename T>
void print(T value) {
    std::cout << "Other: " << value << "\n";
}

int main() {
    print(42);       // Integer: 42
    print(3.14);     // Float: 3.14
    print("hello");  // Other: hello
    return 0;
}
```

---

## Ranges

### Ranges란?

컨테이너와 알고리즘을 더 우아하게 다루는 라이브러리입니다. 파이프라인(Pipeline) 스타일의 연산을 지원합니다.

### 기본 사용법

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 기존 방식
    // for (auto it = nums.begin(); it != nums.end(); ++it) { ... }

    // Ranges 방식
    for (int n : nums | std::views::filter([](int x) { return x % 2 == 0; })
                      | std::views::transform([](int x) { return x * x; })) {
        std::cout << n << " ";  // 4 16 36 64 100
    }

    return 0;
}
```

### Views (뷰)

뷰는 지연 평가(Lazy Evaluation)되며, 원본 데이터를 복사하지 않습니다.

```cpp
#include <ranges>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // filter: 조건에 맞는 요소만
    auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });

    // transform: 변환
    auto squared = v | std::views::transform([](int x) { return x * x; });

    // take: 처음 n개
    auto first3 = v | std::views::take(3);

    // drop: 처음 n개 제외
    auto afterFirst3 = v | std::views::drop(3);

    // reverse: 역순
    auto reversed = v | std::views::reverse;

    // 조합
    auto result = v | std::views::filter([](int x) { return x > 3; })
                    | std::views::transform([](int x) { return x * 2; })
                    | std::views::take(3);

    for (int n : result) {
        std::cout << n << " ";  // 8 10 12
    }

    return 0;
}
```

### 주요 Views

```cpp
#include <ranges>
namespace views = std::views;

// 생성 뷰
auto r1 = views::iota(1, 10);        // 1, 2, ..., 9
auto r2 = views::iota(1) | views::take(10);  // 무한 시퀀스에서 10개

// 변환 뷰
auto r3 = v | views::transform(func);
auto r4 = v | views::filter(pred);
auto r5 = v | views::take(n);
auto r6 = v | views::drop(n);
auto r7 = v | views::take_while(pred);
auto r8 = v | views::drop_while(pred);
auto r9 = v | views::reverse;

// 분할 뷰
auto r10 = str | views::split(' ');  // 공백으로 분할

// 접합 뷰
auto r11 = nested | views::join;     // 중첩 range 평탄화

// 요소 뷰
auto r12 = pairs | views::elements<0>;  // 튜플의 첫 번째 요소
auto r13 = pairs | views::keys;         // map의 키
auto r14 = pairs | views::values;       // map의 값
```

### Range 알고리즘

```cpp
#include <ranges>
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 범위 기반 알고리즘
    std::ranges::sort(v);
    std::ranges::reverse(v);

    auto it = std::ranges::find(v, 5);
    bool found = std::ranges::contains(v, 5);

    int count = std::ranges::count_if(v, [](int x) { return x > 3; });

    auto [min, max] = std::ranges::minmax(v);

    // 프로젝션(Projection)
    struct Person {
        std::string name;
        int age;
    };

    std::vector<Person> people = {{"Alice", 30}, {"Bob", 25}};
    std::ranges::sort(people, {}, &Person::age);  // age로 정렬

    return 0;
}
```

---

## Coroutines

### Coroutines란?

실행을 **일시 중단(Suspend)**하고 나중에 **재개(Resume)**할 수 있는 함수입니다.

### 기본 구조

```cpp
#include <coroutine>
#include <iostream>

// 코루틴 반환 타입
template<typename T>
struct Generator {
    struct promise_type {
        T current_value;

        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }

        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }

        void return_void() {}
    };

    std::coroutine_handle<promise_type> handle;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    Generator(Generator&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    bool next() {
        if (!handle.done()) {
            handle.resume();
        }
        return !handle.done();
    }

    T value() const {
        return handle.promise().current_value;
    }
};

// 코루틴 함수
Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;  // 값을 yield하고 일시 중단
    }
}

int main() {
    auto gen = range(1, 5);

    while (gen.next()) {
        std::cout << gen.value() << " ";  // 1 2 3 4
    }

    return 0;
}
```

### co_await, co_yield, co_return

```cpp
// co_yield: 값을 산출하고 일시 중단
Generator<int> numbers() {
    co_yield 1;
    co_yield 2;
    co_yield 3;
}

// co_return: 코루틴 종료
Task<int> compute() {
    // 비동기 작업...
    co_return 42;
}

// co_await: awaitable 객체 대기
Task<void> asyncWork() {
    auto result = co_await asyncOperation();
    // result 사용...
}
```

### 실용적인 예: 간단한 Task

```cpp
#include <coroutine>
#include <optional>
#include <iostream>

template<typename T>
struct Task {
    struct promise_type {
        std::optional<T> result;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { std::terminate(); }

        void return_value(T value) {
            result = value;
        }
    };

    std::coroutine_handle<promise_type> handle;

    Task(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Task() { if (handle) handle.destroy(); }

    T get() {
        return *handle.promise().result;
    }
};

Task<int> asyncAdd(int a, int b) {
    co_return a + b;
}

int main() {
    auto task = asyncAdd(10, 20);
    std::cout << "Result: " << task.get() << "\n";
    return 0;
}
```

---

## Modules

### Modules란?

헤더 파일의 단점을 해결하는 새로운 코드 구성 방식입니다.

### 모듈 정의

```cpp
// math.cppm (모듈 인터페이스)
export module math;

export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// 내부 구현 (export 안 함)
int helper() {
    return 42;
}
```

### 모듈 사용

```cpp
// main.cpp
import math;
import <iostream>;

int main() {
    std::cout << add(1, 2) << "\n";
    std::cout << multiply(3, 4) << "\n";
    return 0;
}
```

### 컴파일 (GCC 예)

```bash
# 모듈 컴파일
g++ -std=c++20 -fmodules-ts -c math.cppm

# 메인 컴파일 및 링크
g++ -std=c++20 -fmodules-ts main.cpp math.o -o main
```

### 모듈 장점

| 기존 헤더 | 모듈 |
|-----------|------|
| 매번 파싱 | 한 번만 컴파일 |
| 매크로 오염 | 격리됨 |
| 포함 순서 중요 | 순서 무관 |
| 느린 빌드 | 빠른 빌드 |

---

## 기타 C++20 기능

### 삼방향 비교 연산자(Spaceship Operator)

```cpp
#include <compare>

struct Point {
    int x, y;

    auto operator<=>(const Point&) const = default;
    // ==, !=, <, >, <=, >= 자동 생성
};

int main() {
    Point p1{1, 2}, p2{1, 3};

    if (p1 < p2) { /* ... */ }
    if (p1 == p2) { /* ... */ }

    auto result = p1 <=> p2;
    if (result < 0) { /* p1 < p2 */ }

    return 0;
}
```

### 지정 초기화자(Designated Initializers)

```cpp
struct Config {
    int width = 800;
    int height = 600;
    bool fullscreen = false;
    const char* title = "App";
};

int main() {
    // C++20 지정 초기화
    Config cfg{
        .width = 1920,
        .height = 1080,
        .fullscreen = true
        // title은 기본값 사용
    };

    return 0;
}
```

### consteval과 constinit

```cpp
// consteval: 반드시 컴파일 타임에 평가
consteval int square(int n) {
    return n * n;
}

constexpr int a = square(5);  // OK
// int b = square(x);         // 에러! x가 상수가 아님

// constinit: 정적 초기화 강제
constinit int global = 42;
// constinit int bad = foo();  // 에러! foo()가 constexpr 아님
```

### std::span

```cpp
#include <span>
#include <vector>
#include <array>

void process(std::span<int> data) {
    for (int& n : data) {
        n *= 2;
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::array<int, 5> stdArr = {1, 2, 3, 4, 5};

    process(arr);      // OK
    process(vec);      // OK
    process(stdArr);   // OK

    return 0;
}
```

### std::format

```cpp
#include <format>
#include <iostream>

int main() {
    std::string s = std::format("Hello, {}!", "World");
    std::cout << s << "\n";

    std::cout << std::format("{:>10}", 42) << "\n";      // 오른쪽 정렬
    std::cout << std::format("{:08x}", 255) << "\n";     // 16진수, 0 채움
    std::cout << std::format("{:.2f}", 3.14159) << "\n"; // 소수점 2자리

    return 0;
}
```

### std::source_location

```cpp
#include <source_location>
#include <iostream>

void log(const std::string& msg,
         const std::source_location& loc = std::source_location::current()) {
    std::cout << loc.file_name() << ":"
              << loc.line() << " "
              << loc.function_name() << ": "
              << msg << "\n";
}

int main() {
    log("Hello!");  // main.cpp:15 main: Hello!
    return 0;
}
```

---

## C++23 미리보기

### std::expected

```cpp
#include <expected>
#include <string>

std::expected<int, std::string> divide(int a, int b) {
    if (b == 0) {
        return std::unexpected("Division by zero");
    }
    return a / b;
}

int main() {
    auto result = divide(10, 2);
    if (result) {
        std::cout << "Result: " << *result << "\n";
    } else {
        std::cout << "Error: " << result.error() << "\n";
    }
    return 0;
}
```

### std::print

```cpp
#include <print>

int main() {
    std::print("Hello, {}!\n", "World");
    std::println("Value: {}", 42);  // 자동 줄바꿈
    return 0;
}
```

### std::generator (C++23)

```cpp
#include <generator>

std::generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

int main() {
    for (int n : range(1, 10)) {
        std::cout << n << " ";
    }
    return 0;
}
```

---

## 연습 문제

### 문제 1: Concept 정의

"출력 가능한" 타입을 나타내는 Concept을 정의하세요 (operator<< 지원).

<details>
<summary>정답 보기</summary>

```cpp
template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print(const T& value) {
    std::cout << value << "\n";
}
```

</details>

### 문제 2: Range 파이프라인

1부터 100까지 숫자 중 3의 배수이면서 5의 배수가 아닌 수의 제곱 합을 구하세요.

<details>
<summary>정답 보기</summary>

```cpp
#include <ranges>
#include <numeric>
#include <iostream>

int main() {
    auto result = std::views::iota(1, 101)
        | std::views::filter([](int x) { return x % 3 == 0 && x % 5 != 0; })
        | std::views::transform([](int x) { return x * x; });

    int sum = std::accumulate(result.begin(), result.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    return 0;
}
```

</details>

---

## 다음 단계

- [C++ 디자인 패턴](./18_Design_Patterns.md) - C++ 디자인 패턴

---

## 참고 자료

- [cppreference C++20](https://en.cppreference.com/w/cpp/20)
- [C++20 완벽 가이드](https://leanpub.com/cpp20)
- [Ranges 라이브러리](https://en.cppreference.com/w/cpp/ranges)

---

**이전**: [멀티스레딩과 동시성](./16_Multithreading_Concurrency.md) | **다음**: [C++ 디자인 패턴](./18_Design_Patterns.md)

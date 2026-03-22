# C++20 모듈과 유틸리티

**이전**: [C++20 코루틴](./10_CPP20_Coroutines.md) | **다음**: [C++ 멀티스레딩](./12_Multithreading.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모듈이 헤더 파일을 대체하는 이유와 컴파일 모델의 차이를 설명할 수 있다
2. `export module`, `import`, 모듈 파티션을 사용하여 모듈 인터페이스 유닛을 작성할 수 있다
3. 헤더 유닛(`import <header>`)을 사용하여 헤더에서 모듈로 점진적으로 전환할 수 있다
4. 위치 인수와 커스텀 포매터를 포함하여 `std::format`으로 문자열을 포매팅할 수 있다
5. `std::span`을 연속 데이터에 대한 비소유 뷰로 적용하고 서브스팬을 안전하게 추출할 수 있다
6. 삼중 비교(우주선) 연산자를 사용하여 모든 관계 연산자를 자동 생성할 수 있다
7. `std::jthread`로 자동 조인 스레드를 만들고 `stop_token`으로 협력적 취소를 구현할 수 있다

---

C++20은 컨셉, 레인지, 코루틴 외에도 일상적인 코딩을 현대화하는 여러 기능을 도입했습니다. 모듈은 수십 년간 느린 빌드와 매크로 오염을 야기한 전처리기 기반 포함 모델을 제거합니다. `std::format`은 Python 스타일의 포매팅을 C++에 가져옵니다. `std::span`은 연속 메모리에 대한 안전하고 비소유적인 뷰를 제공합니다. 우주선 연산자는 비교 코드의 보일러플레이트를 제거합니다. `std::jthread`는 스레드 join을 잊는 위험을 수정합니다. 이 유틸리티들이 함께 C++20 코드를 더 짧고, 안전하고, 컴파일이 빠르게 만듭니다.

---

## 목차

1. [모듈 소개](#1-모듈-소개)
2. [모듈 문법](#2-모듈-문법)
3. [헤더 유닛](#3-헤더-유닛)
4. [std::format](#4-stdformat)
5. [std::span](#5-stdspan)
6. [삼중 비교](#6-삼중-비교)
7. [std::jthread](#7-stdjthread)
8. [기타 C++20 기능](#8-기타-c20-기능)

---

## 1. 모듈 소개

### 모듈이 필요한 이유

`#include` 모델에는 근본적인 문제가 있습니다:

| 문제 | 설명 |
|------|------|
| 반복 파싱 | `<vector>`가 포함하는 **모든** 번역 단위에서 파싱됨 |
| 매크로 누출 | 한 헤더의 `#define`이 이후 모든 헤더에 영향 |
| 포함 순서 | 다른 순서가 다른 결과를 생성할 수 있음 |
| 캡슐화 없음 | 헤더의 모든 것이 보임 ("private" 섹션 없음) |
| 느린 빌드 | 대규모 프로젝트가 대부분의 컴파일 시간을 헤더 재파싱에 사용 |

모듈은 이 모든 것을 해결합니다: 각 모듈은 바이너리 모듈 인터페이스(BMI)로 **한 번만** 컴파일되고, 매크로는 모듈 경계를 넘어 누출되지 않으며, import 순서는 중요하지 않습니다.

### 헤더 vs 모듈

| 기존 헤더 | C++20 모듈 |
|-----------|-----------|
| 텍스트 포함 (`#include`) | 의미론적 임포트 (`import`) |
| 모든 TU에서 파싱 | 한 번 컴파일, 캐시됨 |
| 매크로 오염 | 매크로 누출 없음 |
| 순서 의존적 | 순서 독립적 |
| 가시성 제어 없음 | `export`가 API를 제어 |
| 느린 증분 빌드 | 빠른 증분 빌드 |

---

## 2. 모듈 문법

### 모듈 인터페이스 유닛

```cpp
// math.cppm (or math.ixx on MSVC)
export module math;  // Declares this file as the interface of module "math"

// Exported: visible to importers
export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// Not exported: internal to the module
int helper_function() {
    return 42;
}
```

### 모듈 구현 유닛

```cpp
// math_impl.cpp
module math;  // Implements the "math" module (no 'export' keyword)

// Can access all names in the module, including non-exported ones
int internal_compute(int x) {
    return helper_function() + x;
}
```

### 모듈 임포트

```cpp
// main.cpp
import math;
import <iostream>;  // Header unit (see Section 3)

int main() {
    std::cout << add(1, 2) << "\n";       // OK: exported
    std::cout << multiply(3, 4) << "\n";  // OK: exported
    // helper_function();                  // Error: not exported
    return 0;
}
```

### 모듈 파티션

큰 모듈은 파티션으로 분할할 수 있습니다:

```cpp
// math-arithmetic.cppm
export module math:arithmetic;

export int add(int a, int b) { return a + b; }
export int sub(int a, int b) { return a - b; }

// math-trig.cppm
export module math:trig;

import <cmath>;

export double sine(double x) { return std::sin(x); }
export double cosine(double x) { return std::cos(x); }

// math.cppm (primary module interface)
export module math;

export import :arithmetic;  // Re-export partition
export import :trig;
```

### 컴파일

```bash
# GCC
g++ -std=c++20 -fmodules-ts -c math.cppm -o math.o
g++ -std=c++20 -fmodules-ts main.cpp math.o -o main

# Clang
clang++ -std=c++20 --precompile math.cppm -o math.pcm
clang++ -std=c++20 -fmodule-file=math=math.pcm main.cpp math.o -o main

# MSVC
cl /std:c++20 /c math.ixx
cl /std:c++20 main.cpp math.obj
```

---

## 3. 헤더 유닛

헤더 유닛을 사용하면 기존 헤더를 모듈처럼 `import`할 수 있어, 코드를 다시 작성하지 않고도 일부 컴파일 속도 향상을 얻을 수 있습니다.

```cpp
// Instead of:
#include <iostream>
#include <vector>
#include <string>

// Write:
import <iostream>;
import <vector>;
import <string>;
```

### 전환 전략

1. **1단계**: 표준 헤더의 `#include`를 `import`로 교체
2. **2단계**: 자체 유틸리티 헤더를 모듈로 변환
3. **3단계**: 애플리케이션 코드를 모듈로 변환
4. 서드파티 헤더는 모듈을 제공할 때까지 `#include`로 **유지**

### 임포트 가능한 헤더

모든 헤더가 임포트 가능한 것은 아닙니다. 표준은 모든 C++ 표준 라이브러리 헤더가 임포트 가능함을 보장합니다. C 헤더(`<cstdio>`, `<cmath>`)는 컴파일러에 따라 임포트 가능 여부가 달라질 수 있습니다.

```cpp
import <vector>;     // Always importable
import <iostream>;   // Always importable
import <cmath>;      // Implementation-defined
// import "mylib.h"; // Importable only if the build system supports it
```

---

## 4. std::format

### 기본 포매팅

`std::format`은 Python 스타일의 문자열 포매팅을 C++에 가져옵니다:

```cpp
#include <format>
#include <iostream>
#include <string>

int main() {
    // Basic replacement
    std::string s = std::format("Hello, {}!", "World");
    std::cout << s << "\n";  // Hello, World!

    // Multiple arguments
    std::cout << std::format("{} + {} = {}", 1, 2, 3) << "\n";

    // Type is inferred automatically
    std::cout << std::format("int={}, double={}, bool={}, str={}",
                             42, 3.14, true, "hello") << "\n";

    return 0;
}
```

### 포맷 지정자

```cpp
#include <format>
#include <iostream>

int main() {
    // 너비와 정렬
    std::cout << std::format("{:>10}", "right") << "\n";    //      right
    std::cout << std::format("{:<10}", "left") << "\n";     // left
    std::cout << std::format("{:^10}", "center") << "\n";   //   center

    // 채움 문자
    std::cout << std::format("{:*>10}", 42) << "\n";        // ********42
    std::cout << std::format("{:0>8}", 42) << "\n";         // 00000042

    // 숫자 포매팅
    std::cout << std::format("{:d}", 255) << "\n";          // 255 (decimal)
    std::cout << std::format("{:x}", 255) << "\n";          // ff (hex)
    std::cout << std::format("{:o}", 255) << "\n";          // 377 (octal)
    std::cout << std::format("{:b}", 255) << "\n";          // 11111111 (binary)
    std::cout << std::format("{:#x}", 255) << "\n";         // 0xff (with prefix)

    // 부동소수점
    std::cout << std::format("{:.2f}", 3.14159) << "\n";    // 3.14
    std::cout << std::format("{:.4e}", 12345.6) << "\n";    // 1.2346e+04

    return 0;
}
```

### 위치 인수

```cpp
#include <format>

// 위치로 인수 참조
auto s1 = std::format("{0} scored {1} points. {0} wins!", "Alice", 95);
// "Alice scored 95 points. Alice wins!"

// 인수 재사용
auto s2 = std::format("{0}{1}{0}", "abra", "cad");
// "abracadabra"
```

### 커스텀 포매터

```cpp
#include <format>
#include <iostream>

struct Point {
    double x, y;
};

template<>
struct std::formatter<Point> {
    // Parse format spec (e.g., {:f} for fixed)
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();  // No custom spec
    }

    // Format the Point
    auto format(const Point& p, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "({:.2f}, {:.2f})", p.x, p.y);
    }
};

int main() {
    Point p{1.5, 2.7};
    std::cout << std::format("Point: {}", p) << "\n";
    // Point: (1.50, 2.70)
    return 0;
}
```

---

## 5. std::span

### 연속 데이터에 대한 비소유 뷰

`std::span`은 요소의 연속 시퀀스에 대한 가볍고 비소유적인 참조입니다. `(포인터, 크기)` 패턴을 대체하며 배열, 벡터, C 배열을 하나의 타입으로 통합합니다.

```cpp
#include <span>
#include <vector>
#include <array>
#include <iostream>

void print(std::span<const int> data) {
    for (int n : data) {
        std::cout << n << " ";
    }
    std::cout << "\n";
}

void double_values(std::span<int> data) {
    for (int& n : data) {
        n *= 2;
    }
}

int main() {
    int c_arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {10, 20, 30};
    std::array<int, 4> std_arr = {100, 200, 300, 400};

    print(c_arr);     // 1 2 3 4 5
    print(vec);       // 10 20 30
    print(std_arr);   // 100 200 300 400

    double_values(c_arr);
    print(c_arr);     // 2 4 6 8 10

    return 0;
}
```

### 서브스팬

```cpp
#include <span>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::span<int> s(v);

    // 처음 N개 요소
    auto first3 = s.first(3);   // 0 1 2

    // 마지막 N개 요소
    auto last3 = s.last(3);     // 7 8 9

    // 서브스팬: 오프셋, 개수
    auto mid = s.subspan(3, 4); // 3 4 5 6

    // 크기와 비어있음
    std::cout << "Size: " << s.size() << "\n";      // 10
    std::cout << "Empty: " << s.empty() << "\n";     // 0 (false)

    // 요소 접근
    std::cout << "Front: " << s.front() << "\n";     // 0
    std::cout << "Back: " << s.back() << "\n";       // 9
    std::cout << "s[4]: " << s[4] << "\n";           // 4

    return 0;
}
```

### 정적 vs 동적 범위

```cpp
#include <span>

// 동적 범위: 런타임에 크기 결정 (기본)
void dynamic_span(std::span<int> s);          // s.size() varies

// 정적 범위: 컴파일 타임에 크기 결정
void fixed_span(std::span<int, 4> s);         // s.size() == 4 always

// 정적 범위로 컴파일 타임 검사 가능
std::array<int, 4> arr = {1, 2, 3, 4};
fixed_span(arr);          // OK
// std::vector<int> v(4);
// fixed_span(v);          // Error: vector has dynamic extent
```

---

## 6. 삼중 비교

### 우주선 연산자 (`<=>`)

삼중 비교 연산자는 단일 선언으로 여섯 개의 관계 연산자를 모두 생성합니다.

```cpp
#include <compare>
#include <iostream>

struct Point {
    int x, y;

    // Default: lexicographic comparison of members in declaration order
    auto operator<=>(const Point&) const = default;
};

int main() {
    Point a{1, 2}, b{1, 3}, c{1, 2};

    // 여섯 개 연산자가 모두 동작:
    std::cout << (a < b) << "\n";   // 1 (true)
    std::cout << (a > b) << "\n";   // 0 (false)
    std::cout << (a <= c) << "\n";  // 1 (true)
    std::cout << (a >= c) << "\n";  // 1 (true)
    std::cout << (a == c) << "\n";  // 1 (true)
    std::cout << (a != b) << "\n";  // 1 (true)

    return 0;
}
```

### 비교 카테고리

```cpp
#include <compare>

// strong_ordering: <, ==, > 중 정확히 하나가 성립; 동등은 동일을 의미
struct IntWrapper {
    int value;
    std::strong_ordering operator<=>(const IntWrapper&) const = default;
};

// weak_ordering: 동등한 객체가 동일하지 않을 수 있음
struct CaseInsensitiveString {
    std::string str;
    std::weak_ordering operator<=>(const CaseInsensitiveString& other) const {
        // Case-insensitive comparison
        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        };
        return to_lower(str) <=> to_lower(other.str);
    }
    bool operator==(const CaseInsensitiveString& other) const {
        return (*this <=> other) == 0;
    }
};

// partial_ordering: 일부 값이 순서 없을 수 있음 (예: NaN)
struct FloatWrapper {
    float value;
    std::partial_ordering operator<=>(const FloatWrapper&) const = default;
};
```

### 커스텀 우주선 연산자

```cpp
#include <compare>
#include <string>

struct Student {
    std::string name;
    double gpa;

    // GPA 우선 비교 (내림차순), 그 다음 이름 (오름차순)
    std::strong_ordering operator<=>(const Student& other) const {
        // Higher GPA first
        if (auto cmp = other.gpa <=> gpa; cmp != 0) return cmp;
        // Then alphabetical name
        return name <=> other.name;
    }

    bool operator==(const Student& other) const {
        return name == other.name && gpa == other.gpa;
    }
};
```

---

## 7. std::jthread

### 자동 조인 스레드

`std::jthread`는 소멸자에서 **자동으로 join**하는 `std::thread`로, 조인 가능한 스레드가 스코프를 벗어날 때 발생하는 `std::terminate` 호출 버그를 제거합니다.

```cpp
#include <thread>
#include <iostream>

void work() {
    std::cout << "Working...\n";
}

int main() {
    {
        std::jthread t(work);
        // No need to call t.join()!
    }
    // t's destructor calls join automatically

    // Compare with std::thread:
    // {
    //     std::thread t(work);
    //     // Forgetting t.join() here causes std::terminate!
    // }

    return 0;
}
```

### stop_token을 이용한 협력적 취소

`std::jthread`는 `stop_token`을 통한 내장 취소 메커니즘을 제공합니다:

```cpp
#include <thread>
#include <iostream>
#include <chrono>

void worker(std::stop_token stoken) {
    int counter = 0;
    while (!stoken.stop_requested()) {
        std::cout << "Working... " << ++counter << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "Worker received stop request. Cleaning up.\n";
}

int main() {
    std::jthread t(worker);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 협력적 중단 요청
    t.request_stop();

    // 소멸자가 자동으로 join
    return 0;
}
```

### 중단 콜백

```cpp
#include <thread>
#include <iostream>

void demo() {
    std::jthread t([](std::stop_token stoken) {
        // Register a callback that runs when stop is requested
        std::stop_callback cb(stoken, [] {
            std::cout << "Stop callback invoked!\n";
        });

        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    t.request_stop();  // Triggers the callback + loop exit
}
```

---

## 8. 기타 C++20 기능

### consteval

`consteval` 함수는 반드시 컴파일 타임에 평가되어야 합니다 (`constexpr`는 컴파일 타임에 평가될 *수 있는* 것과 다름):

```cpp
consteval int square(int n) {
    return n * n;
}

constexpr int a = square(5);   // OK: compile-time
// int x = 5;
// int b = square(x);          // Error: x is not a constant expression
```

### constinit

`constinit`은 변수가 컴파일 타임에 초기화되도록 보장하여 "정적 초기화 순서 문제(static initialization order fiasco)"를 방지합니다:

```cpp
constinit int global_counter = 0;       // OK: zero-initialized at compile time
// constinit int bad = some_function();  // Error if some_function() isn't constexpr
```

### 지정 초기화 (Designated Initializers)

```cpp
struct Config {
    int width = 800;
    int height = 600;
    bool fullscreen = false;
    const char* title = "App";
};

int main() {
    Config cfg{
        .width = 1920,
        .height = 1080,
        .fullscreen = true
        // .title uses default
    };
    return 0;
}
```

### [[likely]]와 [[unlikely]]

옵티마이저를 위한 분기 예측 힌트:

```cpp
int process(int value) {
    if (value > 0) [[likely]] {
        return value * 2;
    } else [[unlikely]] {
        throw std::runtime_error("Negative value");
    }
}
```

### std::source_location

`__FILE__`, `__LINE__`, `__func__` 매크로를 대체합니다:

```cpp
#include <source_location>
#include <iostream>

void log(const std::string& msg,
         const std::source_location& loc = std::source_location::current()) {
    std::cout << loc.file_name() << ":"
              << loc.line() << " ["
              << loc.function_name() << "] "
              << msg << "\n";
}

int main() {
    log("Application started");
    // main.cpp:42 [main] Application started
    return 0;
}
```

---

## 연습 문제

### 연습 1: 모듈 라이브러리

두 개의 파티션을 가진 `geometry` 모듈을 만드세요: `:shapes` (Circle, Rectangle 클래스)와 `:algorithms` (area, perimeter 함수). 모듈을 임포트하고 두 파티션을 사용하는 `main.cpp`를 작성하세요.

### 연습 2: 커스텀 포매터

`Duration` 구조체(시, 분, 초)에 대한 `std::formatter` 특수화를 구현하세요. 두 가지 포맷 스펙을 지원하세요: `{:short}`는 "2h30m15s", `{:long}`은 "2 hours, 30 minutes, 15 seconds".

### 연습 3: Span 유틸리티

`split_at(std::span<int>, size_t pos)`를 작성하여 `std::pair<std::span<int>, std::span<int>>`를 반환하세요. 두 번째 함수 `sliding_window(std::span<const int>, size_t window_size)`를 작성하여 겹치는 윈도우의 `vector<span<const int>>`를 반환하세요.

### 연습 4: 우주선 연산자

기본 `<=>`를 가진 `Version` 구조체(major, minor, patch)를 정의하세요. 그런 다음 순서에서는 patch 필드를 무시하지만 동등성에서는 포함하는 `SemanticVersion`을 정의하세요. 양쪽에 대한 테스트를 작성하세요.

### 연습 5: Jthread 워커 풀

`std::jthread`와 `stop_token`을 사용하여 간단한 워커 풀을 만드세요. 풀은 스레드 안전 큐를 통해 태스크를 받고 `request_stop()`이 호출되면 우아하게 종료해야 합니다. 4개의 워커와 20개의 태스크로 테스트하세요.

---

## 다음 단계

모듈, 포매팅, 스팬, jthread를 익혔으니, 이제 멀티스레딩을 깊이 다룰 준비가 되었습니다 -- 고성능 동시성 C++ 애플리케이션 작성의 기반입니다.

- [C++ 멀티스레딩](./12_Multithreading.md)

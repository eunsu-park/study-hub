# C++20 개념(Concepts)

**이전**: [모던 C++ (C++17)](./07_Modern_CPP_17.md) | **다음**: [C++20 범위(Ranges)](./09_CPP20_Ranges.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 개념이 도입된 이유와 해결하는 문제 설명하기
2. `<concepts>` 헤더의 표준 개념을 사용하여 템플릿 제약하기
3. 단순, 복합, 타입, 중첩 요구사항으로 `requires` 절 작성하기
4. `concept` 키워드를 사용하여 커스텀 개념 정의하기
5. 약식 함수 템플릿을 위한 제약된 `auto` 적용하기
6. 개념 기반 오버로딩과 포섭(subsumption) 규칙 활용하기

---

개념(Concepts)은 일상적인 템플릿 프로그래밍에 가장 큰 영향을 미치는 C++20 기능이라고 할 수 있습니다. 개념 이전에는 템플릿의 오용이 수백 줄에 걸친, 내부 구현 세부사항을 참조하는 에러 메시지를 생성했습니다. SFINAE 기반 제약은 강력했지만 난해했습니다. 개념은 이 두 문제를 단일하고 가독성 좋은 메커니즘으로 대체합니다: 타입이 무엇을 할 수 있어야 하는지 선언하면 컴파일러가 명확한 진단으로 이를 강제합니다. 이 레슨은 기본 원리부터 고급 패턴까지 개념을 다룹니다.

## 1. 왜 개념인가?

### 문제: 템플릿 에러 메시지

```cpp
#include <algorithm>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};
    std::sort(lst.begin(), lst.end());
    // 개념 없이: RandomAccessIterator 요구사항에 대한
    // 이해하기 어려운 에러 메시지 수백 줄이 <algorithm> 깊은 곳에서 발생

    // 개념 사용 시 (C++20): 명확한 에러 메시지:
    // "error: std::list<int>::iterator does not satisfy random_access_iterator"

    return 0;
}
```

### 문제: SFINAE의 복잡성

```cpp
// C++20 이전: 템플릿 제약을 위한 SFINAE
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
T doubleValue(T x) { return x * 2; }

// C++20: 개념으로 가독성 확보
template<std::integral T>
T doubleValue(T x) { return x * 2; }
```

---

## 2. 표준 개념 사용

`<concepts>` 헤더는 바로 사용할 수 있는 풍부한 개념 세트를 제공합니다.

### 핵심 언어 개념

```cpp
#include <concepts>
#include <iostream>

// same_as: 타입이 동일
template<std::same_as<int> T>
void intOnly(T x) {
    std::cout << "int: " << x << "\n";
}

// derived_from: 상속 확인
struct Base {};
struct Derived : Base {};

template<std::derived_from<Base> T>
void acceptBase(T& obj) {
    std::cout << "Derived from Base\n";
}

// convertible_to: 암묵적 변환 존재
template<std::convertible_to<double> T>
double toDouble(T x) {
    return static_cast<double>(x);
}

int main() {
    intOnly(42);       // OK
    // intOnly(3.14);  // 에러: double은 same_as<int>가 아님

    Derived d;
    acceptBase(d);     // OK

    std::cout << toDouble(42) << "\n";    // OK: int -> double
    std::cout << toDouble(3.14f) << "\n"; // OK: float -> double

    return 0;
}
```

### 산술 개념

```cpp
#include <concepts>
#include <iostream>

template<std::integral T>
T bitwiseOr(T a, T b) {
    return a | b;  // 비트 연산은 정수에만 의미가 있음
}

template<std::floating_point T>
T interpolate(T a, T b, T t) {
    return a + t * (b - a);
}

template<std::signed_integral T>
T absolute(T x) {
    return x < 0 ? -x : x;
}

template<std::unsigned_integral T>
T safeDivide(T a, T b) {
    return b == 0 ? 0 : a / b;
}

int main() {
    std::cout << bitwiseOr(0b1010, 0b0101) << "\n";  // 15
    std::cout << interpolate(0.0, 10.0, 0.5) << "\n"; // 5.0
    std::cout << absolute(-42) << "\n";                // 42
    std::cout << safeDivide(10u, 3u) << "\n";          // 3

    return 0;
}
```

### 비교 개념

```cpp
#include <concepts>
#include <iostream>
#include <string>

template<std::equality_comparable T>
bool areSame(const T& a, const T& b) {
    return a == b;
}

template<std::totally_ordered T>
const T& clamp(const T& val, const T& lo, const T& hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

int main() {
    std::cout << std::boolalpha;
    std::cout << areSame(42, 42) << "\n";             // true
    std::cout << areSame(std::string("a"), std::string("b")) << "\n"; // false

    std::cout << clamp(15, 0, 10) << "\n";  // 10
    std::cout << clamp(5, 0, 10) << "\n";   // 5

    return 0;
}
```

### 호출 가능 개념

```cpp
#include <concepts>
#include <iostream>
#include <functional>

// invocable: F를 Args로 호출 가능
template<std::invocable<int, int> F>
int apply(F func, int a, int b) {
    return func(a, b);
}

// predicate: F(Args...)가 bool 유사 타입 반환
template<typename T, std::predicate<T> F>
int countMatching(const std::vector<T>& vec, F pred) {
    int count = 0;
    for (const auto& elem : vec) {
        if (pred(elem)) ++count;
    }
    return count;
}

int main() {
    std::cout << apply([](int a, int b) { return a + b; }, 3, 4) << "\n";
    std::cout << apply(std::plus<>{}, 3, 4) << "\n";

    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    auto isEven = [](int x) { return x % 2 == 0; };
    std::cout << "Even count: " << countMatching(v, isEven) << "\n";

    return 0;
}
```

### 표준 개념 전체 참조

| 개념 | 헤더 | 설명 |
|------|------|------|
| `same_as<T, U>` | `<concepts>` | T와 U가 같은 타입 |
| `derived_from<D, B>` | `<concepts>` | D가 B에서 파생 |
| `convertible_to<From, To>` | `<concepts>` | 암묵적 변환 존재 |
| `integral<T>` | `<concepts>` | 정수 타입 (int, long 등) |
| `signed_integral<T>` | `<concepts>` | 부호 있는 정수 |
| `unsigned_integral<T>` | `<concepts>` | 부호 없는 정수 |
| `floating_point<T>` | `<concepts>` | float, double, long double |
| `equality_comparable<T>` | `<concepts>` | `==`와 `!=` 지원 |
| `totally_ordered<T>` | `<concepts>` | `<`, `>`, `<=`, `>=` 지원 |
| `movable<T>` | `<concepts>` | 이동 생성 및 대입 가능 |
| `copyable<T>` | `<concepts>` | 복사 생성 및 대입 가능 |
| `semiregular<T>` | `<concepts>` | 복사 가능 + 기본 생성 가능 |
| `regular<T>` | `<concepts>` | 준정규 + 동등 비교 가능 |
| `invocable<F, Args...>` | `<concepts>` | F(Args...)가 유효 |
| `predicate<F, Args...>` | `<concepts>` | F(Args...)가 bool 반환 |

---

## 3. requires 절

`requires` 절은 템플릿 매개변수에 대한 제약을 지정합니다. 여러 구문 형태가 있습니다.

### 단순 제약

```cpp
#include <concepts>
#include <iostream>

// 후행 requires 절
template<typename T>
T add(T a, T b) requires std::integral<T> {
    return a + b;
}

// 본문 앞 requires (같은 의미, 다른 위치)
template<typename T>
    requires std::integral<T>
T multiply(T a, T b) {
    return a * b;
}

// 논리곱 (AND)
template<typename T>
    requires std::integral<T> && std::signed_integral<T>
T negate(T x) {
    return -x;
}

// 논리합 (OR)
template<typename T>
    requires std::integral<T> || std::floating_point<T>
T square(T x) {
    return x * x;
}

int main() {
    std::cout << add(1, 2) << "\n";       // OK
    // add(1.0, 2.0);                      // 에러: integral이 아님

    std::cout << square(3) << "\n";        // OK: integral
    std::cout << square(3.14) << "\n";     // OK: floating_point

    return 0;
}
```

---

## 4. requires 표현식

`requires` 표현식은 특정 표현식이 유효한지 확인하는 컴파일 시간 술어입니다.

### 단순 요구사항

```cpp
#include <iostream>
#include <concepts>

// 표현식이 유효한지 확인
template<typename T>
concept Addable = requires(T a, T b) {
    a + b;      // 표현식이 유효해야 함
    a - b;      // 이것도
    a += b;     // 이것도
};

template<Addable T>
T combine(T a, T b) {
    return a + b;
}

int main() {
    std::cout << combine(1, 2) << "\n";      // OK
    std::cout << combine(1.5, 2.5) << "\n";  // OK

    return 0;
}
```

### 타입 요구사항

```cpp
#include <concepts>
#include <vector>
#include <iostream>

// 연관 타입이 존재하는지 확인
template<typename T>
concept HasValueType = requires {
    typename T::value_type;     // value_type이 있어야 함
    typename T::iterator;       // iterator가 있어야 함
    typename T::size_type;      // size_type이 있어야 함
};

template<HasValueType C>
void printInfo(const C& container) {
    std::cout << "Container with " << container.size() << " elements\n";
}

int main() {
    std::vector<int> v = {1, 2, 3};
    printInfo(v);  // OK: vector에 value_type, iterator, size_type이 있음

    // printInfo(42);  // 에러: int에 이러한 타입이 없음

    return 0;
}
```

### 복합 요구사항

```cpp
#include <concepts>
#include <string>
#include <iostream>

// 표현식의 유효성 AND 반환 타입 확인
template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// StringLike: 특정 메서드가 특정 반환 타입을 가져야 함
template<typename T>
concept StringLike = requires(T t, std::size_t i) {
    { t.length() } -> std::convertible_to<std::size_t>;
    { t[i] } -> std::convertible_to<char>;
    { t.substr(i, i) } -> std::same_as<T>;
};

template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void println(const T& value) {
    std::cout << value << "\n";
}

int main() {
    println(42);            // OK
    println("Hello");       // OK
    println(3.14);          // OK

    static_assert(Hashable<int>);
    static_assert(Hashable<std::string>);
    static_assert(StringLike<std::string>);

    return 0;
}
```

### 중첩 요구사항

```cpp
#include <concepts>
#include <vector>

// 중첩 요구사항: requires 안에 requires 사용
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    { t.begin() } -> std::input_or_output_iterator;
    { t.end() } -> std::input_or_output_iterator;
    { t.size() } -> std::convertible_to<std::size_t>;
    // 중첩: value_type이 equality_comparable이어야 함
    requires std::equality_comparable<typename T::value_type>;
};

// SortableContainer: 값이 정렬 가능해야 함
template<typename T>
concept SortableContainer = Container<T> &&
    requires {
        requires std::totally_ordered<typename T::value_type>;
    };

template<SortableContainer C>
void sortContainer(C& c) {
    std::sort(c.begin(), c.end());
}

int main() {
    std::vector<int> v = {3, 1, 4};
    sortContainer(v);  // OK

    return 0;
}
```

---

## 5. 커스텀 개념 정의

### 기본 커스텀 개념

```cpp
#include <concepts>
#include <iostream>
#include <string>
#include <cmath>

// 산술 개념 (정수 또는 부동소수점)
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// 특정 연산을 가진 숫자
template<typename T>
concept Number = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

// 개념 합성
template<typename T>
concept OrderedNumber = Number<T> && std::totally_ordered<T>;

// 이터레이터 개념
template<typename I>
concept ForwardIterable = requires(I it) {
    { *it };         // 역참조 가능
    { ++it } -> std::same_as<I&>;  // 증가 가능
    { it != it } -> std::convertible_to<bool>;  // 비교 가능
};

// 실용적: 직렬화 가능 개념
template<typename T>
concept Serializable = requires(T t, std::ostream& os, std::istream& is) {
    { os << t } -> std::same_as<std::ostream&>;
    { is >> t } -> std::same_as<std::istream&>;
};

template<OrderedNumber T>
T median(T a, T b, T c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

int main() {
    std::cout << median(3, 1, 2) << "\n";      // 2
    std::cout << median(3.0, 1.0, 2.0) << "\n"; // 2.0

    static_assert(Number<int>);
    static_assert(Number<double>);
    static_assert(!Number<std::string>);  // string은 /가 없음

    return 0;
}
```

---

## 6. 제약된 auto

`auto` 키워드를 개념으로 제약하여 약식 함수 템플릿을 만들 수 있습니다.

```cpp
#include <concepts>
#include <iostream>
#include <string>

// 제약된 auto 매개변수 (약식 템플릿)
void printNumber(std::integral auto n) {
    std::cout << "Integer: " << n << "\n";
}

void printNumber(std::floating_point auto n) {
    std::cout << "Float: " << n << "\n";
}

// 각 auto는 독립적으로 제약
auto multiply(std::integral auto a, std::floating_point auto b) {
    return a * b;
}

// 제약된 auto 반환 타입
std::integral auto computeSize(int width, int height) {
    return width * height;
}

// 변수 선언에서의 제약된 auto
void example() {
    std::integral auto x = 42;        // OK: int은 integral
    // std::integral auto y = 3.14;   // 에러: double은 integral이 아님
}

// 커스텀 개념과 함께
template<typename T>
concept Printable = requires(std::ostream& os, T t) {
    { os << t } -> std::same_as<std::ostream&>;
};

void display(Printable auto const& value) {
    std::cout << value << "\n";
}

int main() {
    printNumber(42);      // Integer: 42
    printNumber(3.14);    // Float: 3.14

    std::cout << multiply(3, 2.5) << "\n";  // 7.5
    std::cout << computeSize(10, 20) << "\n";  // 200

    display(42);
    display("Hello");
    display(std::string("World"));

    return 0;
}
```

---

## 7. 개념 기반 오버로딩

여러 오버로드가 호출을 만족할 때, 컴파일러는 **포섭(subsumption)**을 사용하여 가장 제약된 오버로드를 선택합니다.

```cpp
#include <concepts>
#include <iostream>
#include <string>

// 제약 없음 (가장 덜 구체적)
template<typename T>
void describe(const T& value) {
    std::cout << "Unknown type\n";
}

// integral 제약
template<std::integral T>
void describe(const T& value) {
    std::cout << "Integral: " << value << "\n";
}

// signed_integral은 integral을 포섭 (더 구체적)
template<std::signed_integral T>
void describe(const T& value) {
    std::cout << "Signed integral: " << value << "\n";
}

// 커스텀 개념 계층
template<typename T>
concept Animal = requires(T t) {
    { t.name() } -> std::convertible_to<std::string>;
};

template<typename T>
concept Pet = Animal<T> && requires(T t) {
    { t.owner() } -> std::convertible_to<std::string>;
};

// Pet은 Animal을 포섭 (더 제약됨)
template<Animal T>
void greet(const T& a) {
    std::cout << "Hello, animal " << a.name() << "\n";
}

template<Pet T>
void greet(const T& p) {
    std::cout << "Hello, " << p.name() << " (owned by " << p.owner() << ")\n";
}

int main() {
    describe("hello");     // Unknown type (const char*)
    describe(42u);         // Integral (unsigned int)
    describe(42);          // Signed integral (int) -- 가장 제약된 것이 승리

    // 포섭: signed_integral => integral => 제약 없음
    // 컴파일러가 가장 구체적인 매치를 선택

    return 0;
}
```

### 포섭 규칙

```
더 제약됨 (선호)
         |
  signed_integral<T>      -- integral<T>를 함의
         |
     integral<T>           -- 기본 타입 검사를 함의
         |
   (제약 없음)
         |
덜 제약됨 (대체)
```

컴파일러는 **가장 제약된** 사용 가능 오버로드를 선호합니다. 개념 C1이 C2를 **포섭**하는 것은 C1의 제약이 논리적으로 C2의 제약을 함의하는 경우입니다.

---

## 8. 실용적 패턴

### 컨테이너 타입 제약

```cpp
#include <concepts>
#include <ranges>
#include <vector>
#include <list>
#include <iostream>

template<typename C>
concept RandomAccessContainer = requires(C c, typename C::size_type i) {
    typename C::value_type;
    { c[i] } -> std::same_as<typename C::reference>;
    { c.size() } -> std::convertible_to<std::size_t>;
    requires std::random_access_iterator<typename C::iterator>;
};

template<RandomAccessContainer C>
auto binarySearch(const C& container, const typename C::value_type& target)
    -> std::optional<typename C::size_type> {
    typename C::size_type lo = 0;
    typename C::size_type hi = container.size();
    while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (container[mid] == target) return mid;
        if (container[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return std::nullopt;
}

int main() {
    std::vector<int> v = {1, 3, 5, 7, 9};
    if (auto idx = binarySearch(v, 5)) {
        std::cout << "Found at index " << *idx << "\n";
    }

    // std::list<int> l = {1, 2, 3};
    // binarySearch(l, 2);  // 에러: list는 RandomAccessContainer가 아님

    return 0;
}
```

### 연산을 가진 산술 개념

```cpp
#include <concepts>
#include <iostream>

template<typename T>
concept ArithmeticLike = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { -a } -> std::same_as<T>;
} && std::totally_ordered<T> && std::regular<T>;

template<ArithmeticLike T>
class Matrix2x2 {
    T data_[2][2];

public:
    Matrix2x2(T a, T b, T c, T d)
        : data_{{a, b}, {c, d}} {}

    Matrix2x2 operator+(const Matrix2x2& other) const {
        return {
            data_[0][0] + other.data_[0][0],
            data_[0][1] + other.data_[0][1],
            data_[1][0] + other.data_[1][0],
            data_[1][1] + other.data_[1][1]
        };
    }

    T determinant() const {
        return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
    }

    void print() const {
        std::cout << "[" << data_[0][0] << " " << data_[0][1] << "]\n"
                  << "[" << data_[1][0] << " " << data_[1][1] << "]\n";
    }
};

int main() {
    Matrix2x2<int> m1(1, 2, 3, 4);
    Matrix2x2<double> m2(1.0, 0.0, 0.0, 1.0);

    m1.print();
    std::cout << "det = " << m1.determinant() << "\n";

    auto m3 = m2 + Matrix2x2<double>(0.5, 0.5, 0.5, 0.5);
    m3.print();

    return 0;
}
```

### 다중 개념으로 제약

```cpp
#include <concepts>
#include <iostream>
#include <string>

// 여러 요구사항 결합
template<typename T>
concept Displayable =
    std::copyable<T> &&
    requires(std::ostream& os, const T& t) {
        { os << t } -> std::same_as<std::ostream&>;
    };

template<typename T>
concept Parseable =
    std::default_initializable<T> &&
    requires(std::istream& is, T& t) {
        { is >> t } -> std::same_as<std::istream&>;
    };

template<typename T>
concept Serializable = Displayable<T> && Parseable<T>;

template<Serializable T>
class ConfigValue {
    std::string key_;
    T value_;

public:
    ConfigValue(std::string key, T value)
        : key_(std::move(key)), value_(std::move(value)) {}

    friend std::ostream& operator<<(std::ostream& os, const ConfigValue& cv) {
        return os << cv.key_ << "=" << cv.value_;
    }
};

int main() {
    ConfigValue<int> port("port", 8080);
    ConfigValue<std::string> host("host", std::string("localhost"));

    std::cout << port << "\n";
    std::cout << host << "\n";

    return 0;
}
```

---

## 요약

| 기능 | 문법 | 설명 |
|------|------|------|
| 표준 개념 | `std::integral<T>` | 사전 정의된 타입 제약 |
| requires 절 | `requires C<T>` | 템플릿 매개변수 제약 |
| requires 표현식 | `requires(T t) { ... }` | 표현식 유효성 확인 |
| 커스텀 개념 | `concept C = ...` | 재사용 가능한 제약 정의 |
| 제약된 auto | `std::integral auto x` | 약식 템플릿 문법 |
| 포섭 | 더 제약된 것이 승리 | 오버로드 해석 규칙 |

### 개념 문법 치트시트

```cpp
// 개념을 적용하는 네 가지 방법:
template<std::integral T>             // 1. 개념을 템플릿 매개변수로
void f1(T x);

template<typename T>
    requires std::integral<T>         // 2. requires 절
void f2(T x);

template<typename T>
void f3(T x) requires std::integral<T>;  // 3. 후행 requires

void f4(std::integral auto x);       // 4. 제약된 auto (약식)
```

---

## 연습문제

### 연습문제 1: Printable 개념

타입이 `std::ostream`에 대해 `operator<<`를 지원하는지 확인하는 `Printable` 개념을 정의하세요. `Printable` 타입만 받는 `print` 함수를 작성하세요.

### 연습문제 2: 숫자 계층

개념 계층을 만드세요: `Number` -> `Integral` -> `SignedIntegral`, 각 수준이 더 많은 제약을 추가합니다. 포섭을 보여주는 오버로드된 `describe()` 함수를 작성하세요.

### 연습문제 3: Container 개념

`begin()`, `end()`, `size()`, `value_type`, `iterator`를 요구하는 `Container` 개념을 정의하세요. 그런 다음 `random_access_iterator`와 `totally_ordered` 값을 추가로 요구하는 `SortableContainer`를 정의하세요.

### 연습문제 4: 제약된 제네릭 알고리즘

값 타입이 커스텀 `Addable` 개념을 만족하도록 요구하는 `myAccumulate` 함수를 작성하세요. 정수, 부동소수점, `std::string`으로 테스트하세요.

### 연습문제 5: 개념 기반 디스패치

개념 오버로딩을 사용하여 다음을 처리하는 `serialize` 함수를 만드세요:
- 산술 타입 (문자열로 변환)
- 문자열 유사 타입 (인용 부호와 이스케이프)
- 컨테이너 타입 (각 요소를 재귀적으로 직렬화)
- 나머지 (static_assert 실패)

---

## 다음 단계

C++20 범위(Ranges)는 데이터 처리를 위한 합성 가능하고 지연 평가되는 파이프라인을 제공합니다. [09_CPP20_Ranges.md](./09_CPP20_Ranges.md)에서 뷰, 어댑터, 범위 알고리즘을 탐구해 봅시다.

# 템플릿 메타프로그래밍

**이전**: [템플릿](./02_Templates.md) | **다음**: [스마트 포인터와 RAII](./04_Smart_Pointers_and_RAII.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 매개변수 팩과 폴드 표현식을 사용하여 가변 인수 템플릿 작성하기
2. 컴파일 시간 오버로드 해석을 위해 SFINAE와 `std::enable_if` 적용하기
3. `<type_traits>`의 타입 특성을 사용하여 타입을 쿼리하고 변환하기
4. `constexpr`과 `consteval`로 컴파일 시간 계산 구현하기
5. 템플릿 정책과 태그 디스패치 패턴 설계하기

---

템플릿 메타프로그래밍(TMP)은 C++ 템플릿 시스템을 사용하여 컴파일 시간에 계산을 수행하는 기법입니다. C++ 템플릿이 튜링 완전 언어를 형성한다는 우연한 발견에서 시작하여, 더 제네릭하고 더 효율적인 코드를 작성하기 위한 체계적인 기법으로 발전했습니다. 모던 C++(C++17과 C++20)는 `if constexpr`, 폴드 표현식, 개념(concepts)으로 TMP의 역사적 복잡성을 많이 완화했지만, 기저 메커니즘의 이해는 라이브러리 설계와 성능이 중요한 코드에 필수적입니다.

## 1. 가변 인수 템플릿

가변 인수 템플릿은 **매개변수 팩(parameter packs)**을 사용하여 임의 개수의 템플릿 인수를 받습니다.

### 매개변수 팩과 확장

```cpp
#include <iostream>
#include <string>

// sizeof...는 팩의 요소 수를 반환
template<typename... Args>
void countArgs(Args... args) {
    std::cout << "Type pack size: " << sizeof...(Args) << "\n";
    std::cout << "Value pack size: " << sizeof...(args) << "\n";
}

// 재귀 언패킹 (C++17 이전 스타일)
// 기저 사례
void printAll() {
    std::cout << "\n";
}

// 재귀 사례
template<typename T, typename... Rest>
void printAll(T first, Rest... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
    }
    printAll(rest...);
}

// 다양한 컨텍스트에서의 팩 확장
template<typename... Ts>
void expandDemo(Ts... args) {
    // 함수 호출에서의 확장
    // args의 각 요소가 함수를 통해 전달됨
    auto dummy = {(std::cout << args << " ", 0)...};
    (void)dummy;
    std::cout << "\n";
}

int main() {
    countArgs(1, 2.0, "three", 'f');  // 4, 4
    printAll(1, "hello", 3.14, true); // 1, hello, 3.14, 1
    expandDemo(10, 20, 30);           // 10 20 30

    return 0;
}
```

### 컴파일 시간 인덱스 시퀀스

```cpp
#include <iostream>
#include <tuple>
#include <utility>

// 튜플의 모든 요소 출력
template<typename Tuple, std::size_t... Is>
void printTupleImpl(const Tuple& t, std::index_sequence<Is...>) {
    ((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
    std::cout << "\n";
}

template<typename... Args>
void printTuple(const std::tuple<Args...>& t) {
    printTupleImpl(t, std::index_sequence_for<Args...>{});
}

int main() {
    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14

    return 0;
}
```

---

## 2. 폴드 표현식

C++17 폴드 표현식은 매개변수 팩에 대한 연산을 간소화하여, 많은 경우 재귀 템플릿의 필요성을 제거합니다.

### 폴드의 네 가지 형태

```cpp
#include <iostream>
#include <string>

// 단항 우측 폴드: (args op ...)
// 확장: a1 op (a2 op (a3 op a4))
template<typename... Args>
auto sumRight(Args... args) {
    return (args + ...);
}

// 단항 좌측 폴드: (... op args)
// 확장: ((a1 op a2) op a3) op a4
template<typename... Args>
auto sumLeft(Args... args) {
    return (... + args);
}

// 이항 우측 폴드: (args op ... op init)
// 확장: a1 op (a2 op (a3 op init))
template<typename... Args>
auto sumWithInit(Args... args) {
    return (args + ... + 0);  // 빈 팩 처리
}

// 이항 좌측 폴드: (init op ... op args)
// 확장: ((init op a1) op a2) op a3
template<typename... Args>
auto concatStrings(Args... args) {
    return (std::string{} + ... + args);
}

int main() {
    std::cout << sumRight(1, 2, 3, 4) << "\n";     // 10
    std::cout << sumLeft(1, 2, 3, 4) << "\n";       // 10
    std::cout << sumWithInit() << "\n";               // 0 (빈 팩)

    std::cout << concatStrings("Hello", " ", "World") << "\n";

    return 0;
}
```

### 일반적인 폴드 패턴

```cpp
#include <iostream>

// 쉼표로 구분하여 모두 출력
template<typename... Args>
void printComma(Args... args) {
    std::size_t n = 0;
    ((std::cout << (n++ ? ", " : "") << args), ...);
    std::cout << "\n";
}

// 모든 조건 확인
template<typename... Args>
bool allPositive(Args... args) {
    return ((args > 0) && ...);
}

// 임의 조건 확인
template<typename... Args>
bool anyNegative(Args... args) {
    return ((args < 0) || ...);
}

// 모든 인수에 함수 적용
template<typename F, typename... Args>
void forEachArg(F func, Args&&... args) {
    (func(std::forward<Args>(args)), ...);
}

int main() {
    printComma(1, "two", 3.0, '4');   // 1, two, 3, 4
    std::cout << std::boolalpha;
    std::cout << allPositive(1, 2, 3) << "\n";   // true
    std::cout << allPositive(1, -2, 3) << "\n";  // false
    std::cout << anyNegative(1, -2, 3) << "\n";  // true

    forEachArg([](auto x) { std::cout << x << " "; },
               1, "hello", 3.14);  // 1 hello 3.14
    std::cout << "\n";

    return 0;
}
```

---

## 3. SFINAE

SFINAE(Substitution Failure Is Not An Error)는 치환 중 유효하지 않은 타입이 생성되면 컴파일러가 해당 템플릿 오버로드를 조용히 버리도록 합니다.

### std::enable_if

```cpp
#include <iostream>
#include <type_traits>
#include <string>

// 방법 1: 반환 타입 SFINAE
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, std::string>
describe(T value) {
    return "integer: " + std::to_string(value);
}

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, std::string>
describe(T value) {
    return "float: " + std::to_string(value);
}

// 방법 2: 템플릿 매개변수 SFINAE (더 깔끔함)
template<typename T, std::enable_if_t<std::is_pointer_v<T>, int> = 0>
void check(T ptr) {
    std::cout << "Pointer: " << (ptr ? "non-null" : "null") << "\n";
}

template<typename T, std::enable_if_t<!std::is_pointer_v<T>, int> = 0>
void check(T value) {
    std::cout << "Value: " << value << "\n";
}

int main() {
    std::cout << describe(42) << "\n";      // integer: 42
    std::cout << describe(3.14) << "\n";    // float: 3.140000

    int x = 10;
    check(&x);    // Pointer: non-null
    check(42);    // Value: 42

    return 0;
}
```

### SFINAE로 멤버 함수 감지

```cpp
#include <iostream>
#include <type_traits>

// 타입에 .size() 메서드가 있는지 감지
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// 타입에 operator<<가 있는지 감지
template<typename T, typename = void>
struct is_printable : std::false_type {};

template<typename T>
struct is_printable<T, std::void_t<
    decltype(std::declval<std::ostream&>() << std::declval<T>())
>> : std::true_type {};

// 사용
template<typename T>
void smartPrint(const T& obj) {
    if constexpr (has_size<T>::value) {
        std::cout << "Container with " << obj.size() << " elements\n";
    } else if constexpr (is_printable<T>::value) {
        std::cout << obj << "\n";
    } else {
        std::cout << "(unprintable type)\n";
    }
}

#include <vector>
#include <string>

int main() {
    std::cout << std::boolalpha;
    std::cout << "vector has size: " << has_size<std::vector<int>>::value << "\n";  // true
    std::cout << "int has size: " << has_size<int>::value << "\n";                   // false

    smartPrint(std::vector<int>{1, 2, 3});  // Container with 3 elements
    smartPrint(42);                           // 42
    smartPrint(std::string("hello"));        // Container with 5 elements

    return 0;
}
```

---

## 4. 타입 특성(Type Traits)

`<type_traits>` 헤더는 풍부한 컴파일 시간 타입 쿼리 및 변환 세트를 제공합니다.

### 기본 타입 카테고리

```cpp
#include <iostream>
#include <type_traits>
#include <vector>

template<typename T>
void analyzeType() {
    std::cout << std::boolalpha;
    std::cout << "  is_void:            " << std::is_void_v<T> << "\n";
    std::cout << "  is_integral:        " << std::is_integral_v<T> << "\n";
    std::cout << "  is_floating_point:  " << std::is_floating_point_v<T> << "\n";
    std::cout << "  is_array:           " << std::is_array_v<T> << "\n";
    std::cout << "  is_pointer:         " << std::is_pointer_v<T> << "\n";
    std::cout << "  is_reference:       " << std::is_reference_v<T> << "\n";
    std::cout << "  is_class:           " << std::is_class_v<T> << "\n";
    std::cout << "  is_enum:            " << std::is_enum_v<T> << "\n";
}

int main() {
    std::cout << "--- int ---\n";
    analyzeType<int>();

    std::cout << "--- double* ---\n";
    analyzeType<double*>();

    std::cout << "--- std::vector<int> ---\n";
    analyzeType<std::vector<int>>();

    return 0;
}
```

### 타입 변환

```cpp
#include <iostream>
#include <type_traits>

int main() {
    std::cout << std::boolalpha;

    // 한정자 제거
    using A = std::remove_const_t<const int>;        // int
    using B = std::remove_reference_t<int&>;         // int
    using C = std::remove_pointer_t<int*>;           // int
    using D = std::decay_t<const int&>;              // int
    using E = std::decay_t<int[10]>;                 // int*
    using F = std::decay_t<int(double)>;             // int(*)(double)

    std::cout << std::is_same_v<A, int> << "\n";  // true
    std::cout << std::is_same_v<B, int> << "\n";  // true
    std::cout << std::is_same_v<C, int> << "\n";  // true
    std::cout << std::is_same_v<D, int> << "\n";  // true

    // 한정자 추가
    using G = std::add_const_t<int>;                 // const int
    using H = std::add_lvalue_reference_t<int>;      // int&
    using I = std::add_pointer_t<int>;               // int*

    // 조건부 타입 선택
    using J = std::conditional_t<(sizeof(int) > 4), long, int>;

    // 공통 타입
    using K = std::common_type_t<int, double>;       // double
    std::cout << std::is_same_v<K, double> << "\n";  // true

    return 0;
}
```

### 타입 관계

```cpp
#include <iostream>
#include <type_traits>

class Base {};
class Derived : public Base {};
class Unrelated {};

int main() {
    std::cout << std::boolalpha;

    // is_same
    std::cout << std::is_same_v<int, int> << "\n";          // true
    std::cout << std::is_same_v<int, unsigned> << "\n";      // false

    // is_base_of
    std::cout << std::is_base_of_v<Base, Derived> << "\n";   // true
    std::cout << std::is_base_of_v<Base, Unrelated> << "\n"; // false

    // is_convertible
    std::cout << std::is_convertible_v<Derived*, Base*> << "\n";  // true
    std::cout << std::is_convertible_v<int, double> << "\n";      // true

    // is_constructible
    std::cout << std::is_constructible_v<std::string, const char*> << "\n"; // true

    // is_assignable
    std::cout << std::is_assignable_v<int&, double> << "\n";  // true

    return 0;
}
```

---

## 5. if constexpr

C++17의 `if constexpr`는 컴파일 시간에 조건을 평가하고 거짓 분기를 완전히 버려서 인스턴스화 에러를 방지합니다.

### SFINAE 대체

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// 이전: SFINAE (복잡하고 읽기 어려움)
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, std::string>
toStringOld(T value) { return std::to_string(value); }

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, std::string>
toStringOld(T value) { return std::to_string(value); }

// 이후: if constexpr (깔끔하고 가독성 좋음)
template<typename T>
std::string toString(T value) {
    if constexpr (std::is_integral_v<T>) {
        return "int:" + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float:" + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string:" + value;
    } else {
        return "(unknown type)";
    }
}

int main() {
    std::cout << toString(42) << "\n";
    std::cout << toString(3.14) << "\n";
    std::cout << toString(std::string("hello")) << "\n";

    return 0;
}
```

### if constexpr를 사용한 컴파일 시간 재귀

```cpp
#include <iostream>
#include <tuple>

// if constexpr를 사용하여 튜플 요소 출력
template<std::size_t I = 0, typename... Ts>
void printTuple(const std::tuple<Ts...>& t) {
    if constexpr (I < sizeof...(Ts)) {
        if constexpr (I > 0) std::cout << ", ";
        std::cout << std::get<I>(t);
        printTuple<I + 1>(t);
    }
}

// 컴파일 시간 팩토리얼
template<int N>
constexpr int factorial() {
    if constexpr (N <= 1) {
        return 1;
    } else {
        return N * factorial<N - 1>();
    }
}

int main() {
    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14
    std::cout << "\n";

    constexpr int f5 = factorial<5>();
    std::cout << "5! = " << f5 << "\n";  // 120

    return 0;
}
```

---

## 6. constexpr과 consteval

### constexpr 함수

`constexpr` 함수는 상수 인수가 주어지면 컴파일 시간에 평가될 수 있고, 그렇지 않으면 런타임에 평가됩니다.

```cpp
#include <iostream>
#include <array>

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

constexpr int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// constexpr 클래스
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int manhattanDistance() const { return x + y; }
    constexpr Point operator+(const Point& other) const {
        return {x + other.x, y + other.y};
    }
};

int main() {
    // 컴파일 시간 평가
    constexpr int fib10 = fibonacci(10);
    static_assert(fib10 == 55, "fibonacci(10) should be 55");

    constexpr int p = power(2, 10);
    static_assert(p == 1024);

    // 배열 크기로 사용
    std::array<int, fibonacci(6)> arr;  // 크기 8

    constexpr Point a(3, 4);
    constexpr Point b(1, 2);
    constexpr Point c = a + b;
    static_assert(c.x == 4 && c.y == 6);

    // 런타임 평가도 가능
    int n;
    std::cin >> n;
    std::cout << "fib(" << n << ") = " << fibonacci(n) << "\n";

    return 0;
}
```

### consteval (C++20)

`consteval` 함수는 반드시 컴파일 시간에 평가되어야 합니다. "즉시 함수(immediate functions)"라고 불립니다.

```cpp
#include <iostream>

// 반드시 컴파일 시간에 평가되어야 함
consteval int square(int n) {
    return n * n;
}

// consteval은 constexpr을 호출할 수 있지만, 반대는 불가
consteval int cube(int n) {
    return n * n * n;
}

int main() {
    constexpr int s = square(5);  // OK: 컴파일 시간
    std::cout << s << "\n";       // 25

    // int x = 5;
    // int bad = square(x);  // 에러: x는 상수 표현식이 아님

    // 컴파일 시간 검증에 유용
    constexpr int result = cube(3);
    static_assert(result == 27);

    return 0;
}
```

---

## 7. 태그 디스패치

태그 디스패치는 빈 타입(태그)을 사용하여 타입 속성에 기반한 오버로드를 선택하며, SFINAE에 대한 깔끔한 대안을 제공합니다.

```cpp
#include <iostream>
#include <type_traits>
#include <iterator>
#include <vector>
#include <list>

// 이터레이터 카테고리에 대한 태그 디스패치
namespace detail {

template<typename Iter>
void advanceImpl(Iter& it, int n, std::random_access_iterator_tag) {
    std::cout << "Random access advance (O(1))\n";
    it += n;
}

template<typename Iter>
void advanceImpl(Iter& it, int n, std::bidirectional_iterator_tag) {
    std::cout << "Bidirectional advance (O(n))\n";
    while (n > 0) { ++it; --n; }
    while (n < 0) { --it; ++n; }
}

template<typename Iter>
void advanceImpl(Iter& it, int n, std::input_iterator_tag) {
    std::cout << "Input advance (O(n), forward only)\n";
    while (n > 0) { ++it; --n; }
}

} // namespace detail

template<typename Iter>
void myAdvance(Iter& it, int n) {
    // 이터레이터 카테고리 태그에 기반한 디스패치
    detail::advanceImpl(it, n,
        typename std::iterator_traits<Iter>::iterator_category{});
}

// true_type/false_type를 사용한 태그 디스패치
template<typename T>
void processImpl(T value, std::true_type /* is_integral */) {
    std::cout << "Processing integer: " << value << "\n";
}

template<typename T>
void processImpl(T value, std::false_type /* is_integral */) {
    std::cout << "Processing non-integer: " << value << "\n";
}

template<typename T>
void process(T value) {
    processImpl(value, std::is_integral<T>{});
}

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto vit = v.begin();
    myAdvance(vit, 3);  // Random access advance (O(1))

    std::list<int> l = {1, 2, 3, 4, 5};
    auto lit = l.begin();
    myAdvance(lit, 3);  // Bidirectional advance (O(n))

    process(42);    // Processing integer: 42
    process(3.14);  // Processing non-integer: 3.14

    return 0;
}
```

---

## 8. 정책 기반 설계

정책 기반 설계는 템플릿 매개변수를 사용하여 컴파일 시간에 동작을 주입하여, 유연하면서도 효율적인 클래스를 만듭니다.

```cpp
#include <iostream>
#include <string>
#include <mutex>

// 로깅 정책
struct ConsoleLog {
    static void log(const std::string& msg) {
        std::cout << "[Console] " << msg << "\n";
    }
};

struct NullLog {
    static void log(const std::string&) {
        // 아무것도 하지 않음
    }
};

// 스레딩 정책
struct SingleThreaded {
    struct Lock {
        Lock() {}  // No-op
    };
};

struct MultiThreaded {
    struct Lock {
        Lock() { /* mutex.lock(); */ }
        ~Lock() { /* mutex.unlock(); */ }
    };
};

// 정책을 사용하는 클래스
template<typename LogPolicy = ConsoleLog,
         typename ThreadPolicy = SingleThreaded>
class DataStore {
    int data_ = 0;

public:
    void set(int value) {
        typename ThreadPolicy::Lock lock;
        LogPolicy::log("Setting value to " + std::to_string(value));
        data_ = value;
    }

    int get() const {
        return data_;
    }
};

int main() {
    // 상세 출력, 단일 스레드 저장소
    DataStore<ConsoleLog, SingleThreaded> verbose;
    verbose.set(42);  // [Console] Setting value to 42

    // 무출력 저장소 (로깅이 완전히 컴파일에서 제거됨)
    DataStore<NullLog, SingleThreaded> silent;
    silent.set(42);   // 출력 없음, 오버헤드 없음

    // 스레드 안전 저장소
    DataStore<ConsoleLog, MultiThreaded> safe;
    safe.set(42);     // [Console] Setting value to 42

    return 0;
}
```

### CRTP (Curiously Recurring Template Pattern, 기묘하게 재귀하는 템플릿 패턴)

```cpp
#include <iostream>

// CRTP를 통한 정적 다형성
template<typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->areaImpl();
    }

    void describe() const {
        std::cout << "Shape with area: " << area() << "\n";
    }
};

class Circle : public Shape<Circle> {
    double radius_;
public:
    Circle(double r) : radius_(r) {}
    double areaImpl() const { return 3.14159 * radius_ * radius_; }
};

class Rectangle : public Shape<Rectangle> {
    double w_, h_;
public:
    Rectangle(double w, double h) : w_(w), h_(h) {}
    double areaImpl() const { return w_ * h_; }
};

// 어떤 Shape<Derived>와도 동작 -- 가상 디스패치 없음
template<typename T>
void printArea(const Shape<T>& shape) {
    shape.describe();
}

int main() {
    Circle c(5.0);
    Rectangle r(3.0, 4.0);

    printArea(c);  // Shape with area: 78.5397
    printArea(r);  // Shape with area: 12

    return 0;
}
```

---

## 9. 요약

| 기법 | 목적 | 도입 시기 |
|------|------|----------|
| 가변 인수 템플릿 | 임의 인수 수용 | C++11 |
| 폴드 표현식 | 팩 연산 간소화 | C++17 |
| SFINAE / enable_if | 조건부 오버로드 | C++11 |
| 타입 특성 | 컴파일 시간 타입 쿼리 | C++11 |
| if constexpr | 컴파일 시간 분기 | C++17 |
| constexpr / consteval | 컴파일 시간 계산 | C++11/C++20 |
| 태그 디스패치 | 깔끔한 오버로드 선택 | C++98+ |
| 정책 기반 설계 | 컴파일 시간 전략 | C++98+ |
| CRTP | 정적 다형성 | C++98+ |

---

## 연습문제

### 연습문제 1: 타입 안전 printf

가변 인수 템플릿을 사용하여 `%` 자리표시자를 타입 안전 인수로 대체하는 `safePrintf(format, args...)` 함수를 구현하세요. 인수 개수 불일치 시 예외를 던지세요.

### 연습문제 2: 컴파일 시간 문자열 해시

컴파일 시간에 문자열 리터럴의 해시를 계산하는 `constexpr` 함수를 작성하세요. `switch` 문에서 사용하세요.

### 연습문제 3: has_method 감지기

SFINAE와 `std::void_t`를 사용하여 이름이 있는 어떤 메서드든 감지할 수 있는 제네릭 `has_method` 감지기를 만드세요 (예: `has_push_back`, `has_reserve`).

### 연습문제 4: 컴파일 시간 피보나치 수열

전체적으로 컴파일 시간에 계산되는 처음 N개의 피보나치 수를 포함하는 `constexpr std::array`를 구현하세요.

### 연습문제 5: 정책 기반 로거

교체 가능한 출력 정책(콘솔, 파일, null)과 포매팅 정책(일반, 타임스탬프, JSON)을 가진 정책 기반 설계로 Logger 클래스를 설계하세요.

---

## 다음 단계

스마트 포인터와 RAII는 모던 C++의 자원 관리의 핵심입니다. [04_Smart_Pointers_and_RAII.md](./04_Smart_Pointers_and_RAII.md)에서 `unique_ptr`, `shared_ptr`, `weak_ptr`를 심층적으로 탐구해 봅시다.

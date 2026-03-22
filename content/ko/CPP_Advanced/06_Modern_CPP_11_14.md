# 모던 C++ -- C++11과 C++14

**이전**: [에러 처리 패턴](./05_Error_Handling_Patterns.md) | **다음**: [모던 C++ (C++17)](./07_Modern_CPP_17.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 타입 추론과 후행 반환 타입을 위해 `auto`와 `decltype` 사용하기
2. 중괄호를 사용한 균일 초기화로 축소 변환 방지하기
3. 제네릭 람다(C++14) 포함 다양한 캡처 모드로 람다 표현식 작성하기
4. 컴파일 시간 평가를 위해 `constexpr` 사용하고 C++14의 완화된 `constexpr` 이해하기
5. `nullptr`과 `NULL`을 구분하고 `enum class`로 범위 지정 열거형 적용하기
6. 설명적 메시지와 함께 컴파일 시간 검증을 위해 `static_assert` 사용하기

---

C++11은 언어 역사상 가장 변혁적인 업데이트로, 수동 메모리 관리와 장황한 문법의 부담을 가진 C++을 표현적이고 안전하며 효율적인 모던 코드를 작성할 수 있는 언어로 변모시켰습니다. C++14는 이러한 기능들을 품질 향상으로 다듬었습니다. 함께 이들은 모든 모던 C++ 프로그래머가 마스터해야 하는 기준선을 형성합니다. 이 레슨은 C++11/14 기능들을 추출하고 확장하여 깊이 있게 다룹니다.

## 1. auto와 decltype

### auto 타입 추론

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <memory>

int main() {
    // 기본 타입 추론
    auto x = 42;          // int
    auto y = 3.14;        // double
    auto s = "Hello";     // const char*
    auto b = true;        // bool

    // 복잡한 타입 간소화
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();  // std::vector<int>::iterator

    std::map<std::string, std::vector<int>> data;
    auto& ref = data;  // std::map<std::string, std::vector<int>>&

    // auto와 const, 참조
    const auto cx = 42;     // const int
    auto& ry = y;           // double&
    const auto& cry = y;    // const double&
    auto&& urx = 42;        // int&& (rvalue 참조)
    auto&& ury = y;         // double& (lvalue 참조, 참조 축소)

    // auto 반환 타입 (C++14)
    auto makeVec = []() {
        return std::vector<int>{1, 2, 3};
    };
    auto v = makeVec();

    return 0;
}
```

### decltype

```cpp
#include <iostream>

int x = 10;
decltype(x) y = 20;        // int (x와 같은 타입)
decltype(x + 0.5) z = 1.5; // double

// decltype은 참조와 const를 보존
int& rx = x;
decltype(rx) ry = x;       // int& (참조 보존)

const int cx = 42;
decltype(cx) cy = 10;      // const int

// 표현식에서의 decltype
// decltype(x)는 int (변수)
// decltype((x))는 int& (lvalue를 산출하는 표현식)

// 후행 반환 타입
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: 후행 반환 타입 불필요
template<typename T, typename U>
auto addModern(T a, U b) {
    return a + b;
}
```

### decltype(auto) (C++14)

```cpp
#include <iostream>
#include <string>

// decltype(auto)는 참조를 포함한 정확한 타입을 보존
int x = 10;

auto getValue1() { return x; }          // int (감쇠)
decltype(auto) getValue2() { return x; }  // int (동일)
decltype(auto) getRef() { return (x); }  // int& (표현식이 lvalue)

// 완벽한 반환 타입 전달에 유용
template<typename F, typename... Args>
decltype(auto) callAndReturn(F&& f, Args&&... args) {
    return f(std::forward<Args>(args)...);
}
```

---

## 2. 균일 초기화

중괄호 초기화(`{}`)는 어디서나 동작하는 균일한 문법을 제공하며 축소 변환을 방지합니다.

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <initializer_list>

// initializer_list를 사용하는 커스텀 클래스
class Matrix {
    std::vector<std::vector<int>> data_;

public:
    Matrix(std::initializer_list<std::initializer_list<int>> init) {
        for (auto& row : init) {
            data_.emplace_back(row);
        }
    }

    void print() const {
        for (auto& row : data_) {
            for (int val : row) std::cout << val << " ";
            std::cout << "\n";
        }
    }
};

int main() {
    // 직접 초기화
    int a{42};
    double b{3.14};
    std::string c{"Hello"};

    // 축소 방지
    // int narrow{3.14};     // 에러: 축소 변환
    // char small{1000};     // 에러: 축소 변환
    int ok{static_cast<int>(3.14)};  // OK: 명시적 캐스트

    // 컨테이너 초기화
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // 구조체 초기화
    struct Point { int x, y; };
    Point p{10, 20};

    // 커스텀 클래스
    Matrix m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    m.print();

    // 가장 성가신 파싱 회피
    // Widget w();   // 함수를 선언! (most vexing parse)
    // Widget w{};   // Widget 객체를 생성

    return 0;
}
```

### initializer_list 함정

```cpp
#include <iostream>
#include <vector>

class Widget {
public:
    Widget(int size, double value) {
        std::cout << "size=" << size << ", value=" << value << "\n";
    }
    Widget(std::initializer_list<double> list) {
        std::cout << "initializer_list with " << list.size() << " elements\n";
    }
};

int main() {
    Widget w1(10, 5.0);    // (int, double) 생성자 호출
    Widget w2{10, 5.0};    // initializer_list 생성자 호출!
    Widget w3(10, 5.0);    // 비목록 생성자를 원하면 () 사용

    // std::vector에도 같은 문제
    std::vector<int> v1(5, 10);   // 5개 요소, 각각 10
    std::vector<int> v2{5, 10};   // 2개 요소: 5와 10

    return 0;
}
```

---

## 3. 범위 기반 for 루프

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 값으로 복사 (안전하지만 복사함)
    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // 참조 (수정 가능)
    for (int& x : vec) {
        x *= 2;
    }

    // const 참조 (읽기 전용, 복사 없음)
    for (const int& x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // auto 추론 (권장)
    for (const auto& x : vec) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // map과 함께 사용
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}};
    for (const auto& [name, score] : scores) {  // C++17 구조적 바인딩
        std::cout << name << ": " << score << "\n";
    }

    // C 배열과 함께 사용
    int arr[] = {10, 20, 30};
    for (int x : arr) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // initializer list와 함께 사용
    for (int x : {100, 200, 300}) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

## 4. nullptr

```cpp
#include <iostream>

void foo(int n) {
    std::cout << "int: " << n << std::endl;
}

void foo(int* p) {
    std::cout << "pointer: " << (p ? "non-null" : "null") << std::endl;
}

int main() {
    // NULL은 0으로 정의되어 모호성 발생
    // foo(NULL);  // 모호: int인가 포인터인가?

    // nullptr은 타입 안전
    foo(nullptr);  // 포인터 오버로드 호출
    foo(0);        // int 오버로드 호출

    // 타입: std::nullptr_t
    std::nullptr_t np = nullptr;

    // 불리언 컨텍스트에서 동작
    int* p = nullptr;
    if (!p) {
        std::cout << "p is null" << std::endl;
    }

    // 템플릿 컨텍스트
    auto lambda = [](auto* ptr) {
        if (ptr) std::cout << *ptr << "\n";
    };

    int x = 42;
    lambda(&x);     // OK
    // lambda(NULL);  // 템플릿에서 에러
    // lambda(nullptr);  // OK: std::nullptr_t 추론

    return 0;
}
```

---

## 5. 람다 표현식

### 기본 문법

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // [캡처](매개변수) -> 반환타입 { 본문 }

    // 기본 람다
    auto hello = []() { std::cout << "Hello!\n"; };
    hello();

    // 매개변수와 반환값
    auto add = [](int a, int b) -> int { return a + b; };
    std::cout << add(3, 4) << "\n";  // 7

    // 반환 타입 추론 (보통 불필요)
    auto multiply = [](double a, double b) { return a * b; };

    // 즉시 실행 람다 (IIFE)
    int result = [](int x) { return x * x; }(5);
    std::cout << "IIFE: " << result << "\n";  // 25

    return 0;
}
```

### 캡처 모드

```cpp
#include <iostream>
#include <string>

int main() {
    int x = 10;
    int y = 20;
    std::string name = "Alice";

    // 값으로 캡처 (람다 생성 시점에 복사)
    auto byValue = [x, y]() {
        std::cout << x + y << "\n";
        // x = 100;  // 에러: 값으로 캡처된 것은 const
    };

    // 참조로 캡처
    auto byRef = [&x, &y]() {
        x += 10;
        y += 10;
    };
    byRef();
    std::cout << x << ", " << y << "\n";  // 20, 30

    // 모두 값으로 캡처
    auto allVal = [=]() {
        std::cout << x << " " << name << "\n";
    };

    // 모두 참조로 캡처
    auto allRef = [&]() {
        x = 100;
        name = "Bob";
    };

    // 혼합 캡처
    auto mixed = [=, &x]() {  // y,name은 값으로; x는 참조로
        x = 50;
        std::cout << y << " " << name << "\n";
    };

    // mutable 람다 (값으로 캡처된 변수 수정)
    int counter = 0;
    auto increment = [counter]() mutable {
        return ++counter;  // 람다의 내부 복사본 수정
    };
    std::cout << increment() << "\n";  // 1
    std::cout << increment() << "\n";  // 2
    std::cout << counter << "\n";      // 0 (원본 변경 없음)

    // this 포인터 캡처 (멤버 함수에서)
    // auto lambda = [this]() { memberFunc(); };
    // auto lambda = [*this]() { /* *this의 복사본 캡처 */ };  // C++17

    return 0;
}
```

### STL과 람다

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};

    // 내림차순 정렬
    std::sort(vec.begin(), vec.end(),
        [](int a, int b) { return a > b; });

    // 첫 번째 짝수 찾기
    auto it = std::find_if(vec.begin(), vec.end(),
        [](int x) { return x % 2 == 0; });
    if (it != vec.end()) {
        std::cout << "First even: " << *it << "\n";
    }

    // 3보다 큰 요소 세기
    int count = std::count_if(vec.begin(), vec.end(),
        [](int x) { return x > 3; });
    std::cout << "Count > 3: " << count << "\n";

    // 변환
    std::vector<int> squared(vec.size());
    std::transform(vec.begin(), vec.end(), squared.begin(),
        [](int x) { return x * x; });

    // for_each
    std::for_each(vec.begin(), vec.end(),
        [](int x) { std::cout << x << " "; });
    std::cout << "\n";

    // 람다로 누적
    int sum = std::accumulate(vec.begin(), vec.end(), 0,
        [](int acc, int x) { return acc + x; });
    std::cout << "Sum: " << sum << "\n";

    return 0;
}
```

### 제네릭 람다 (C++14)

```cpp
#include <iostream>
#include <string>
#include <vector>

int main() {
    // C++14: auto 매개변수로 람다를 제네릭하게 (템플릿과 유사)
    auto print = [](const auto& x) {
        std::cout << x << "\n";
    };

    print(42);          // int
    print(3.14);        // double
    print("Hello");     // const char*

    // 다중 auto 매개변수
    auto add = [](auto a, auto b) {
        return a + b;
    };
    std::cout << add(1, 2) << "\n";            // 3
    std::cout << add(1.5, 2.5) << "\n";        // 4.0
    std::cout << add(std::string("Hello, "),
                     std::string("World!")) << "\n";

    // 완벽한 전달을 사용하는 제네릭 람다
    auto wrapper = [](auto&& func, auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };

    auto result = wrapper([](int a, int b) { return a + b; }, 3, 4);
    std::cout << "Wrapped: " << result << "\n";  // 7

    return 0;
}
```

---

## 6. constexpr

### C++11 constexpr

```cpp
#include <iostream>
#include <array>

// C++11: constexpr 함수는 단일 return 문이어야 함
constexpr int square(int x) {
    return x * x;
}

constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// constexpr 변수
constexpr double PI = 3.14159265358979;
constexpr int MAX_SIZE = 1024;

// constexpr 생성자
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int manhattanDist() const { return x + y; }
};

int main() {
    // 컴파일 시간에 평가
    constexpr int s = square(5);
    static_assert(s == 25, "square(5) should be 25");

    constexpr int f = factorial(5);
    static_assert(f == 120, "5! should be 120");

    // 템플릿 인수나 배열 크기로 사용
    std::array<int, factorial(4)> arr;  // 크기 24

    constexpr Point p(3, 4);
    static_assert(p.manhattanDist() == 7);

    // 런타임에서도 동작
    int n;
    std::cin >> n;
    std::cout << "square(" << n << ") = " << square(n) << "\n";

    return 0;
}
```

### 완화된 constexpr (C++14)

```cpp
#include <iostream>
#include <array>

// C++14: constexpr 함수가 여러 문장을 가질 수 있음
constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// 루프와 지역 변수를 사용하는 constexpr
constexpr int sumOfSquares(int n) {
    int sum = 0;
    for (int i = 1; i <= n; ++i) {
        sum += i * i;
    }
    return sum;
}

// constexpr 배열 생성
template<std::size_t N>
constexpr std::array<int, N> generateFibArray() {
    std::array<int, N> arr{};
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = fibonacci(i);
    }
    return arr;
}

int main() {
    constexpr int fib10 = fibonacci(10);
    static_assert(fib10 == 55);

    constexpr int ss = sumOfSquares(5);
    static_assert(ss == 55);  // 1+4+9+16+25

    constexpr auto fibs = generateFibArray<10>();
    for (int f : fibs) {
        std::cout << f << " ";  // 0 1 1 2 3 5 8 13 21 34
    }
    std::cout << "\n";

    return 0;
}
```

---

## 7. enum class

범위 지정 열거형은 이름 충돌과 암묵적 변환을 방지합니다.

```cpp
#include <iostream>

// 구식 enum (C++03): 네임스페이스를 오염시키고 암묵적 변환
enum OldColor { RED, GREEN, BLUE };
// enum TrafficLight { RED, YELLOW, GREEN };  // 에러: RED와 GREEN 충돌

// 범위 지정 enum (C++11)
enum class Color { Red, Green, Blue };
enum class TrafficLight { Red, Yellow, Green };  // 충돌 없음!

// 기저 타입 지정
enum class ErrorCode : uint8_t {
    Success = 0,
    NotFound = 1,
    Timeout = 2,
    Internal = 255
};

// 전방 선언 (범위 지정 enum이나 명시적 타입에서만 가능)
enum class Direction : int;

void handleError(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success:
            std::cout << "OK\n";
            break;
        case ErrorCode::NotFound:
            std::cout << "Not found\n";
            break;
        default:
            std::cout << "Error: " << static_cast<int>(code) << "\n";
    }
}

int main() {
    Color c = Color::Red;
    TrafficLight t = TrafficLight::Red;

    // int로의 암묵적 변환 없음
    // int x = c;              // 에러
    int x = static_cast<int>(c);  // OK: 명시적 캐스트

    // 타입 안전 비교
    // if (c == t) {}          // 에러: 다른 타입
    if (c == Color::Red) {
        std::cout << "It's red\n";
    }

    handleError(ErrorCode::NotFound);

    // 기저 타입
    std::cout << "Size of ErrorCode: " << sizeof(ErrorCode) << "\n";  // 1

    return 0;
}
```

---

## 8. static_assert

명확한 에러 메시지를 생성하는 컴파일 시간 검증입니다.

```cpp
#include <iostream>
#include <type_traits>

// 컴파일 시간에 크기 확인
static_assert(sizeof(int) >= 4, "int must be at least 4 bytes");
static_assert(sizeof(void*) == 8, "64-bit platform required");

// 타입 속성 확인
template<typename T>
class NumericContainer {
    static_assert(std::is_arithmetic_v<T>,
                  "NumericContainer requires an arithmetic type");

    T value_;

public:
    NumericContainer(T v) : value_(v) {}
    T get() const { return value_; }
};

// 정렬 확인
template<typename T>
class AlignedStorage {
    static_assert(alignof(T) <= 64,
                  "Type alignment must not exceed 64 bytes");
    alignas(T) char storage[sizeof(T)];
};

// C++17: 메시지 없는 static_assert
// static_assert(sizeof(int) == 4);

int main() {
    NumericContainer<int> ic(42);      // OK
    NumericContainer<double> dc(3.14); // OK
    // NumericContainer<std::string> sc("hi"); // 컴파일 시간 에러

    return 0;
}
```

---

## 9. 이동 의미론 통합

이동 의미론이 C++11/14 기능과 통합되는 방식의 간략한 요약입니다.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>

int main() {
    // 실전에서의 std::move
    std::string s = "Hello, World!";
    std::vector<std::string> vec;

    vec.push_back(s);              // 복사
    vec.push_back(std::move(s));   // 이동 (s는 이제 비어있음)

    // emplace_back: 제자리 생성 (이동조차 피함)
    vec.emplace_back("Constructed in-place");

    // unique_ptr와 이동 의미론
    auto ptr = std::make_unique<int>(42);
    auto ptr2 = std::move(ptr);   // 소유권 전달

    // 값으로 반환: 컴파일러가 RVO 적용
    auto makeVec = []() -> std::vector<int> {
        std::vector<int> v = {1, 2, 3};
        return v;  // NRVO 또는 이동
    };
    auto result = makeVec();

    return 0;
}
```

---

## 10. C++14 추가 기능

### std::make_unique

```cpp
#include <memory>
#include <iostream>

int main() {
    // C++11: make_unique 없음
    std::unique_ptr<int> p1(new int(42));

    // C++14: make_unique (선호)
    auto p2 = std::make_unique<int>(42);
    auto arr = std::make_unique<int[]>(10);

    std::cout << *p2 << "\n";  // 42

    return 0;
}
```

### 변수 템플릿

```cpp
#include <iostream>

// 변수 템플릿 (C++14)
template<typename T>
constexpr T pi = T(3.14159265358979323846L);

template<typename T>
constexpr T e = T(2.71828182845904523536L);

// 표준 라이브러리에서 사용: std::is_integral_v<T>는 변수 템플릿
// template<typename T>
// inline constexpr bool is_integral_v = is_integral<T>::value;

int main() {
    std::cout << "float pi:  " << pi<float> << "\n";
    std::cout << "double pi: " << pi<double> << "\n";
    std::cout << "double e:  " << e<double> << "\n";

    return 0;
}
```

### 반환 타입 추론

```cpp
#include <iostream>
#include <vector>

// C++14: 컴파일러가 반환 타입 추론
auto multiply(int a, int b) {
    return a * b;  // int로 추론
}

auto getString() {
    return std::string("Hello");
}

// 참고: 모든 반환 경로가 같은 타입을 반환해야 함
auto conditional(bool flag) {
    if (flag) return 1;
    return 2;
    // return 1.0;  // 에러: 일관되지 않은 반환 타입
}
```

### 이진 리터럴과 자릿수 구분자

```cpp
#include <iostream>

int main() {
    // 이진 리터럴 (C++14)
    int bits = 0b1010'1010;       // 170
    int mask = 0b1111'0000;       // 240

    // 자릿수 구분자 (모든 숫자 리터럴)
    int million = 1'000'000;
    double pi = 3.141'592'653;
    int hex = 0xFF'FF;
    long long big = 1'000'000'000'000LL;

    std::cout << "bits: " << bits << "\n";
    std::cout << "million: " << million << "\n";
    std::cout << "pi: " << pi << "\n";

    return 0;
}
```

---

## 요약

| 기능 | 버전 | 핵심 이점 |
|------|------|----------|
| `auto` / `decltype` | C++11 | 타입 장황함 감소 |
| 균일 초기화 | C++11 | 축소 방지, 일관된 문법 |
| 범위 기반 for | C++11 | 더 깔끔한 반복 |
| `nullptr` | C++11 | 타입 안전 널 포인터 |
| 람다 표현식 | C++11 | 인라인 함수, 클로저 |
| `constexpr` | C++11 | 컴파일 시간 평가 |
| `enum class` | C++11 | 범위 지정, 타입 안전 열거형 |
| `static_assert` | C++11 | 컴파일 시간 검증 |
| 이동 의미론 | C++11 | 효율적인 자원 전달 |
| 제네릭 람다 | C++14 | 템플릿과 유사한 람다 |
| `decltype(auto)` | C++14 | 정확한 타입 추론 |
| 완화된 `constexpr` | C++14 | 다중 문장 컴파일 시간 함수 |
| `std::make_unique` | C++14 | 안전한 unique_ptr 생성 |
| 변수 템플릿 | C++14 | 타입 매개변수화된 상수 |
| 이진 리터럴 | C++14 | `0b1010` 표기법 |
| 자릿수 구분자 | C++14 | 가독성 높은 숫자 리터럴 |

---

## 연습문제

### 연습문제 1: 람다 누적기

호출 간 값을 누적하는 람다를 반환하는 함수를 작성하세요 (mutable 캡처를 사용하는 상태 유지 람다).

### 연습문제 2: constexpr 룩업 테이블

컴파일 시간에 룩업 테이블(예: 정수 각도에서의 사인 값)을 생성하고 `std::array`에 저장하는 `constexpr` 함수를 만드세요.

### 연습문제 3: 제네릭 컨테이너 출력

구분자와 접두사/접미사를 설정할 수 있는, 범위 기반 for를 사용하여 어떤 컨테이너든 출력하는 제네릭 람다를 작성하세요.

### 연습문제 4: 타입 안전 빌더 패턴

`auto` 반환 타입과 이동 의미론을 사용하여, 각 `.set_X()` 호출이 이동으로 빌더를 반환하고 최종 `.build()`가 생성된 객체를 반환하는 빌더 패턴을 구현하세요.

### 연습문제 5: Enum 기반 상태 머신

상태와 이벤트에 `enum class`를 사용하고, `static_assert`로 상태 전이 테이블을 컴파일 시간에 검증하는 간단한 상태 머신을 구현하세요.

---

## 다음 단계

C++17은 구조적 바인딩, `std::optional`, `std::variant`, `std::filesystem` 등 많은 기능을 가져왔습니다. [07_Modern_CPP_17.md](./07_Modern_CPP_17.md)에서 탐구해 봅시다.

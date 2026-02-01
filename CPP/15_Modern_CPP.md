# 모던 C++ (C++11/14/17/20)

## 1. C++ 버전 변화

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ 표준 발전 역사                         │
├─────────────────────────────────────────────────────────────┤
│  1998   2003   2011        2014    2017    2020    2023     │
│   │      │      │           │       │       │       │       │
│   ▼      ▼      ▼           ▼       ▼       ▼       ▼       │
│ C++98  C++03  C++11       C++14   C++17   C++20   C++23     │
│        (버그   (대규모     (개선)  (대규모 (대규모          │
│        수정)   업데이트)          업데이트) 업데이트)         │
└─────────────────────────────────────────────────────────────┘
```

### 컴파일 옵션

```bash
# C++11
g++ -std=c++11 main.cpp -o main

# C++14
g++ -std=c++14 main.cpp -o main

# C++17
g++ -std=c++17 main.cpp -o main

# C++20
g++ -std=c++20 main.cpp -o main
```

---

## 2. C++11 주요 기능

### auto 키워드

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    // 타입 자동 추론
    auto x = 42;          // int
    auto y = 3.14;        // double
    auto s = "Hello";     // const char*

    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 긴 타입명 대신 auto 사용
    auto it = vec.begin();  // std::vector<int>::iterator

    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // 복잡한 타입도 auto로 간단히
    for (auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // 함수 반환 타입 추론
    auto add = [](int a, int b) { return a + b; };
    std::cout << add(3, 4) << std::endl;  // 7

    return 0;
}
```

### 범위 기반 for 루프

```cpp
#include <iostream>
#include <vector>
#include <array>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 값 복사
    for (int x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 참조 (수정 가능)
    for (int& x : vec) {
        x *= 2;
    }

    // const 참조 (읽기만)
    for (const int& x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 배열에도 사용 가능
    int arr[] = {10, 20, 30};
    for (int x : arr) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 초기화 리스트와 함께
    for (int x : {100, 200, 300}) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### nullptr

```cpp
#include <iostream>

void foo(int n) {
    std::cout << "int: " << n << std::endl;
}

void foo(int* p) {
    std::cout << "pointer" << std::endl;
}

int main() {
    // C++11 이전: NULL은 0으로 정의됨
    // foo(NULL);  // 모호함!

    // C++11: nullptr 사용
    foo(nullptr);  // pointer
    foo(0);        // int: 0

    int* p = nullptr;
    if (p == nullptr) {
        std::cout << "p is null" << std::endl;
    }

    return 0;
}
```

### 초기화 리스트 (Initializer List)

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <initializer_list>

class MyContainer {
private:
    std::vector<int> data;

public:
    MyContainer(std::initializer_list<int> list)
        : data(list) {
        std::cout << "생성자 호출" << std::endl;
    }

    void print() const {
        for (int x : data) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    // 균일 초기화 (Uniform Initialization)
    int a{42};
    double b{3.14};
    std::string c{"Hello"};

    // 컨테이너 초기화
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    // 커스텀 클래스
    MyContainer mc = {10, 20, 30, 40};
    mc.print();  // 10 20 30 40

    // 배열
    int arr[] = {1, 2, 3};

    // 중괄호로 narrowing 방지
    // int x{3.14};  // 에러! narrowing conversion

    return 0;
}
```

### 람다 표현식

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // 기본 람다
    auto hello = []() {
        std::cout << "Hello, Lambda!" << std::endl;
    };
    hello();

    // 매개변수와 반환
    auto add = [](int a, int b) -> int {
        return a + b;
    };
    std::cout << add(3, 4) << std::endl;  // 7

    // 캡처 (외부 변수 접근)
    int x = 10;
    int y = 20;

    // 값 캡처
    auto byValue = [x, y]() {
        std::cout << x + y << std::endl;
    };

    // 참조 캡처
    auto byRef = [&x, &y]() {
        x++;
        y++;
    };

    // 모든 변수 값 캡처
    auto allByValue = [=]() {
        std::cout << x + y << std::endl;
    };

    // 모든 변수 참조 캡처
    auto allByRef = [&]() {
        x = 100;
        y = 200;
    };

    // 혼합 캡처
    auto mixed = [=, &x]() {  // y는 값, x는 참조
        x = 50;
        std::cout << y << std::endl;
    };

    // STL 알고리즘과 사용
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};

    // 정렬
    std::sort(vec.begin(), vec.end(),
        [](int a, int b) { return a > b; });  // 내림차순

    // 출력
    std::for_each(vec.begin(), vec.end(),
        [](int x) { std::cout << x << " "; });
    std::cout << std::endl;

    return 0;
}
```

### 이동 시맨틱 (Move Semantics)

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>

class Buffer {
private:
    int* data;
    size_t size;

public:
    // 생성자
    Buffer(size_t n) : data(new int[n]), size(n) {
        std::cout << "생성자" << std::endl;
    }

    // 소멸자
    ~Buffer() {
        delete[] data;
        std::cout << "소멸자" << std::endl;
    }

    // 복사 생성자
    Buffer(const Buffer& other)
        : data(new int[other.size]), size(other.size) {
        std::copy(other.data, other.data + size, data);
        std::cout << "복사 생성자" << std::endl;
    }

    // 이동 생성자 (C++11)
    Buffer(Buffer&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "이동 생성자" << std::endl;
    }

    // 복사 대입 연산자
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
        }
        std::cout << "복사 대입" << std::endl;
        return *this;
    }

    // 이동 대입 연산자 (C++11)
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        std::cout << "이동 대입" << std::endl;
        return *this;
    }
};

Buffer createBuffer() {
    return Buffer(100);  // 이동 최적화
}

int main() {
    Buffer b1(10);

    Buffer b2 = b1;              // 복사 생성자
    Buffer b3 = std::move(b1);   // 이동 생성자
    // b1은 이제 빈 상태

    Buffer b4 = createBuffer();  // 이동 생성자 (RVO로 생략될 수 있음)

    return 0;
}
```

### 스마트 포인터

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "생성" << std::endl; }
    ~Resource() { std::cout << "소멸" << std::endl; }
};

int main() {
    // unique_ptr: 단독 소유
    std::unique_ptr<Resource> p1(new Resource());
    auto p2 = std::make_unique<Resource>();  // C++14

    // shared_ptr: 공유 소유
    auto p3 = std::make_shared<Resource>();
    auto p4 = p3;  // 참조 카운트 증가
    std::cout << "참조 카운트: " << p3.use_count() << std::endl;

    // weak_ptr: 약한 참조
    std::weak_ptr<Resource> weak = p3;
    if (auto sp = weak.lock()) {
        std::cout << "객체 접근 가능" << std::endl;
    }

    return 0;
}
```

### constexpr

```cpp
#include <iostream>
#include <array>

// 컴파일 타임 상수 함수
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int square(int x) {
    return x * x;
}

// 컴파일 타임 상수 클래스
class Point {
public:
    int x, y;
    constexpr Point(int x, int y) : x(x), y(y) {}
    constexpr int getX() const { return x; }
    constexpr int getY() const { return y; }
};

int main() {
    // 컴파일 타임에 계산
    constexpr int fact5 = factorial(5);  // 120
    constexpr int sq10 = square(10);     // 100

    // 배열 크기로 사용 가능
    std::array<int, factorial(4)> arr;  // 크기 24

    // 컴파일 타임 상수 객체
    constexpr Point p(3, 4);
    static_assert(p.getX() == 3, "X should be 3");

    std::cout << "5! = " << fact5 << std::endl;
    std::cout << "10^2 = " << sq10 << std::endl;

    return 0;
}
```

---

## 3. C++14 주요 기능

### 제네릭 람다

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    // C++14: auto 매개변수
    auto print = [](auto x) {
        std::cout << x << std::endl;
    };

    print(42);          // int
    print(3.14);        // double
    print("Hello");     // const char*

    // 여러 타입 매개변수
    auto add = [](auto a, auto b) {
        return a + b;
    };

    std::cout << add(1, 2) << std::endl;       // 3
    std::cout << add(1.5, 2.5) << std::endl;   // 4
    std::cout << add(std::string("Hello, "), std::string("World!")) << std::endl;

    return 0;
}
```

### 변수 템플릿

```cpp
#include <iostream>

// 변수 템플릿 (C++14)
template<typename T>
constexpr T pi = T(3.141592653589793238462643383);

template<typename T>
constexpr T e = T(2.718281828459045235360287471);

int main() {
    std::cout << "float pi: " << pi<float> << std::endl;
    std::cout << "double pi: " << pi<double> << std::endl;
    std::cout << "double e: " << e<double> << std::endl;

    return 0;
}
```

### [[deprecated]] 속성

```cpp
#include <iostream>

// 함수 사용 중단 경고
[[deprecated("Use newFunction() instead")]]
void oldFunction() {
    std::cout << "Old function" << std::endl;
}

void newFunction() {
    std::cout << "New function" << std::endl;
}

// 클래스도 가능
class [[deprecated("Use NewClass instead")]] OldClass {};

int main() {
    // oldFunction();  // 컴파일러 경고 발생
    newFunction();

    return 0;
}
```

### 이진 리터럴과 자릿수 구분자

```cpp
#include <iostream>

int main() {
    // 이진 리터럴 (C++14)
    int binary = 0b1010'1010;  // 170
    int hex = 0xFF'FF;         // 65535

    // 자릿수 구분자
    int million = 1'000'000;
    double pi = 3.141'592'653;
    int binary2 = 0b1111'0000'1111'0000;

    std::cout << "binary: " << binary << std::endl;
    std::cout << "million: " << million << std::endl;
    std::cout << "pi: " << pi << std::endl;

    return 0;
}
```

### 반환 타입 추론

```cpp
#include <iostream>
#include <vector>

// C++14: 반환 타입 자동 추론
auto multiply(int a, int b) {
    return a * b;  // int 반환으로 추론
}

auto getString() {
    return std::string("Hello");  // std::string 반환으로 추론
}

// 재귀에도 사용 가능
auto factorial(int n) -> int {  // 재귀는 명시 필요
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    auto result = multiply(3, 4);
    std::cout << result << std::endl;  // 12

    auto str = getString();
    std::cout << str << std::endl;  // Hello

    return 0;
}
```

---

## 4. C++17 주요 기능

### 구조적 바인딩

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <array>

std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}

int main() {
    // 튜플 분해
    auto [num, pi, str] = getData();
    std::cout << num << ", " << pi << ", " << str << std::endl;

    // pair 분해
    std::pair<int, std::string> p = {1, "Alice"};
    auto [id, name] = p;
    std::cout << id << ": " << name << std::endl;

    // 배열 분해
    int arr[] = {1, 2, 3};
    auto [a, b, c] = arr;
    std::cout << a << ", " << b << ", " << c << std::endl;

    // map 순회
    std::map<std::string, int> ages = {
        {"Alice", 25},
        {"Bob", 30}
    };

    for (auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    // 구조체 분해
    struct Point { int x, y; };
    Point pt = {10, 20};
    auto [x, y] = pt;
    std::cout << "Point: " << x << ", " << y << std::endl;

    return 0;
}
```

### if/switch 초기화 구문

```cpp
#include <iostream>
#include <map>
#include <mutex>

std::map<int, std::string> database = {
    {1, "Alice"},
    {2, "Bob"}
};

std::mutex mtx;

int main() {
    // if with initializer
    if (auto it = database.find(1); it != database.end()) {
        std::cout << "Found: " << it->second << std::endl;
    }

    // switch with initializer
    switch (auto x = 2 * 3; x) {
        case 6:
            std::cout << "x is 6" << std::endl;
            break;
        default:
            std::cout << "x is " << x << std::endl;
    }

    // lock과 함께 사용
    if (std::lock_guard<std::mutex> lock(mtx); true) {
        // 뮤텍스 잠금 상태에서 작업
        std::cout << "Protected section" << std::endl;
    }

    return 0;
}
```

### if constexpr

```cpp
#include <iostream>
#include <type_traits>
#include <string>

template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "정수: " << value * 2 << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "실수: " << value / 2 << std::endl;
    } else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "문자열: " << value.length() << "글자" << std::endl;
    } else {
        std::cout << "기타: " << value << std::endl;
    }
}

int main() {
    process(10);                    // 정수: 20
    process(3.14);                  // 실수: 1.57
    process(std::string("Hello"));  // 문자열: 5글자
    process("Hello");               // 기타: Hello

    return 0;
}
```

### std::optional

```cpp
#include <iostream>
#include <optional>
#include <string>

std::optional<int> divide(int a, int b) {
    if (b == 0) {
        return std::nullopt;
    }
    return a / b;
}

std::optional<std::string> findUser(int id) {
    if (id == 1) return "Alice";
    if (id == 2) return "Bob";
    return std::nullopt;
}

int main() {
    // 기본 사용
    auto result = divide(10, 2);
    if (result) {
        std::cout << "결과: " << *result << std::endl;
    }

    auto result2 = divide(10, 0);
    std::cout << "has_value: " << result2.has_value() << std::endl;  // 0

    // value_or로 기본값 제공
    std::cout << divide(10, 3).value_or(-1) << std::endl;  // 3
    std::cout << divide(10, 0).value_or(-1) << std::endl;  // -1

    // 문자열
    auto user = findUser(1);
    if (user) {
        std::cout << "User: " << *user << std::endl;
    }

    std::cout << findUser(3).value_or("Unknown") << std::endl;  // Unknown

    return 0;
}
```

### std::variant

```cpp
#include <iostream>
#include <variant>
#include <string>

int main() {
    // 여러 타입 중 하나를 저장
    std::variant<int, double, std::string> v;

    v = 42;
    std::cout << std::get<int>(v) << std::endl;

    v = 3.14;
    std::cout << std::get<double>(v) << std::endl;

    v = "Hello";
    std::cout << std::get<std::string>(v) << std::endl;

    // 현재 타입 확인
    if (std::holds_alternative<std::string>(v)) {
        std::cout << "문자열입니다" << std::endl;
    }

    // 인덱스로 접근
    std::cout << "index: " << v.index() << std::endl;  // 2

    // visit 패턴
    std::variant<int, double, std::string> values[] = {
        42, 3.14, std::string("Hello")
    };

    for (auto& val : values) {
        std::visit([](auto&& arg) {
            std::cout << arg << std::endl;
        }, val);
    }

    return 0;
}
```

### std::filesystem

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // 경로 다루기
    fs::path p = "/home/user/documents/file.txt";

    std::cout << "filename: " << p.filename() << std::endl;
    std::cout << "stem: " << p.stem() << std::endl;
    std::cout << "extension: " << p.extension() << std::endl;
    std::cout << "parent_path: " << p.parent_path() << std::endl;

    // 현재 디렉토리
    std::cout << "현재 경로: " << fs::current_path() << std::endl;

    // 파일/디렉토리 존재 확인
    fs::path testPath = ".";
    if (fs::exists(testPath)) {
        std::cout << testPath << " 존재함" << std::endl;
    }

    // 디렉토리 순회
    std::cout << "\n=== 현재 디렉토리 내용 ===" << std::endl;
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path().filename();
        if (fs::is_directory(entry)) {
            std::cout << " [DIR]";
        } else {
            std::cout << " [" << fs::file_size(entry) << " bytes]";
        }
        std::cout << std::endl;
    }

    // 경로 조합
    fs::path dir = "/home/user";
    fs::path file = "document.txt";
    fs::path full = dir / file;
    std::cout << "조합된 경로: " << full << std::endl;

    return 0;
}
```

### std::string_view

```cpp
#include <iostream>
#include <string>
#include <string_view>

// 문자열을 복사하지 않고 참조
void printView(std::string_view sv) {
    std::cout << "View: " << sv << std::endl;
    std::cout << "Length: " << sv.length() << std::endl;
}

int main() {
    // 다양한 문자열 타입에서 생성
    std::string str = "Hello, World!";
    const char* cstr = "Hello from C!";
    char arr[] = "Hello from array!";

    printView(str);
    printView(cstr);
    printView(arr);
    printView("Literal string");

    // 서브스트링 (복사 없음)
    std::string_view sv = "Hello, World!";
    std::string_view sub = sv.substr(0, 5);
    std::cout << "Substring: " << sub << std::endl;  // Hello

    // 주의: 원본이 사라지면 댕글링!
    // std::string_view bad;
    // {
    //     std::string temp = "temporary";
    //     bad = temp;
    // }
    // std::cout << bad << std::endl;  // 정의되지 않은 동작!

    return 0;
}
```

---

## 5. C++20 주요 기능

### Concepts

```cpp
#include <iostream>
#include <concepts>
#include <vector>

// concept 정의
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept Printable = requires(T t) {
    { std::cout << t };
};

template<typename T>
concept Container = requires(T t) {
    t.begin();
    t.end();
    t.size();
};

// concept 사용
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// requires 절
template<typename T>
requires Printable<T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// 축약형
void printSize(Container auto& c) {
    std::cout << "Size: " << c.size() << std::endl;
}

int main() {
    std::cout << add(1, 2) << std::endl;       // 3
    std::cout << add(1.5, 2.5) << std::endl;   // 4
    // add("a", "b");  // 에러: Numeric 제약 불만족

    print(42);
    print("Hello");

    std::vector<int> vec = {1, 2, 3};
    printSize(vec);  // Size: 3

    return 0;
}
```

### Ranges

```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 파이프라인 스타일
    auto result = numbers
        | std::views::filter([](int n) { return n % 2 == 0; })  // 짝수
        | std::views::transform([](int n) { return n * n; })     // 제곱
        | std::views::take(3);                                    // 3개만

    std::cout << "결과: ";
    for (int n : result) {
        std::cout << n << " ";  // 4 16 36
    }
    std::cout << std::endl;

    // 다양한 view
    // iota: 숫자 범위 생성
    for (int n : std::views::iota(1, 6)) {
        std::cout << n << " ";  // 1 2 3 4 5
    }
    std::cout << std::endl;

    // reverse
    for (int n : std::views::reverse(numbers)) {
        std::cout << n << " ";  // 10 9 8 7 6 5 4 3 2 1
    }
    std::cout << std::endl;

    // drop: 처음 n개 건너뛰기
    for (int n : numbers | std::views::drop(5)) {
        std::cout << n << " ";  // 6 7 8 9 10
    }
    std::cout << std::endl;

    return 0;
}
```

### 삼중 비교 연산자 (Spaceship Operator)

```cpp
#include <iostream>
#include <compare>
#include <string>

class Version {
public:
    int major, minor, patch;

    Version(int ma, int mi, int pa)
        : major(ma), minor(mi), patch(pa) {}

    // 삼중 비교 연산자 (C++20)
    auto operator<=>(const Version& other) const = default;

    // == 연산자도 자동 생성
};

int main() {
    // 기본 타입
    int a = 5, b = 10;
    auto result = a <=> b;

    if (result < 0) {
        std::cout << "a < b" << std::endl;
    } else if (result > 0) {
        std::cout << "a > b" << std::endl;
    } else {
        std::cout << "a == b" << std::endl;
    }

    // 커스텀 클래스
    Version v1{1, 2, 3};
    Version v2{1, 3, 0};

    if (v1 < v2) {
        std::cout << "v1 < v2" << std::endl;
    }

    if (v1 == Version{1, 2, 3}) {
        std::cout << "v1 == 1.2.3" << std::endl;
    }

    return 0;
}
```

### 모듈 (Modules)

```cpp
// math.cppm (모듈 인터페이스)
export module math;

export int add(int a, int b) {
    return a + b;
}

export int multiply(int a, int b) {
    return a * b;
}

// main.cpp
import math;
import <iostream>;

int main() {
    std::cout << add(3, 4) << std::endl;       // 7
    std::cout << multiply(3, 4) << std::endl;  // 12
    return 0;
}

// 컴파일 (컴파일러마다 다름)
// g++ -std=c++20 -fmodules-ts -c math.cppm
// g++ -std=c++20 -fmodules-ts main.cpp math.o -o main
```

### 코루틴 (Coroutines)

```cpp
#include <iostream>
#include <coroutine>

// 간단한 Generator 예제
template<typename T>
struct Generator {
    struct promise_type {
        T value;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T v) {
            value = v;
            return {};
        }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    bool next() {
        if (!handle.done()) {
            handle.resume();
        }
        return !handle.done();
    }

    T value() { return handle.promise().value; }
};

Generator<int> range(int start, int end) {
    for (int i = start; i < end; i++) {
        co_yield i;
    }
}

int main() {
    auto gen = range(1, 5);

    while (gen.next()) {
        std::cout << gen.value() << " ";  // 1 2 3 4
    }
    std::cout << std::endl;

    return 0;
}
```

### std::format (C++20)

```cpp
#include <iostream>
#include <format>
#include <string>

int main() {
    // 기본 사용
    std::string s1 = std::format("Hello, {}!", "World");
    std::cout << s1 << std::endl;  // Hello, World!

    // 인덱스 지정
    std::string s2 = std::format("{1} + {0} = {2}", 10, 20, 30);
    std::cout << s2 << std::endl;  // 20 + 10 = 30

    // 정렬과 너비
    std::string s3 = std::format("|{:>10}|", 42);    // 오른쪽 정렬
    std::string s4 = std::format("|{:<10}|", 42);    // 왼쪽 정렬
    std::string s5 = std::format("|{:^10}|", 42);    // 가운데 정렬
    std::cout << s3 << std::endl;  // |        42|
    std::cout << s4 << std::endl;  // |42        |
    std::cout << s5 << std::endl;  // |    42    |

    // 숫자 형식
    std::string s6 = std::format("{:b}", 42);   // 이진수
    std::string s7 = std::format("{:x}", 255);  // 16진수
    std::string s8 = std::format("{:.2f}", 3.14159);  // 소수점 2자리
    std::cout << s6 << std::endl;  // 101010
    std::cout << s7 << std::endl;  // ff
    std::cout << s8 << std::endl;  // 3.14

    return 0;
}
```

---

## 6. 모범 사례

### 코드 스타일 권장사항

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

// 1. auto를 적절히 사용
auto calculate() {
    return 42;  // 명확한 경우 auto 사용
}

// 2. const 적극 사용
void printVector(const std::vector<int>& vec) {
    for (const auto& x : vec) {
        std::cout << x << " ";
    }
}

// 3. 스마트 포인터 사용
class Resource {
public:
    void use() { std::cout << "사용" << std::endl; }
};

void goodMemoryManagement() {
    auto ptr = std::make_unique<Resource>();
    ptr->use();
    // 자동 해제
}

// 4. 범위 기반 for 사용
void iterateModern(const std::vector<int>& vec) {
    for (const auto& item : vec) {
        std::cout << item << std::endl;
    }
}

// 5. 초기화 리스트 사용
class Person {
private:
    std::string name;
    int age;

public:
    Person(std::string n, int a)
        : name(std::move(n)), age(a) {}  // 이동 사용
};

// 6. noexcept 적절히 사용
void safeFunction() noexcept {
    // 예외를 던지지 않는 함수
}

// 7. constexpr 활용
constexpr int maxSize = 100;
constexpr int square(int x) { return x * x; }

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    printVector(vec);
    std::cout << std::endl;

    goodMemoryManagement();

    iterateModern(vec);

    constexpr int result = square(10);
    std::cout << "10^2 = " << result << std::endl;

    return 0;
}
```

---

## 7. 요약

| 버전 | 주요 기능 |
|------|-----------|
| C++11 | auto, 람다, 이동 시맨틱, 스마트 포인터, nullptr, constexpr |
| C++14 | 제네릭 람다, 변수 템플릿, 반환 타입 추론 |
| C++17 | 구조적 바인딩, if constexpr, optional, variant, filesystem |
| C++20 | Concepts, Ranges, 삼중 비교, 모듈, 코루틴, std::format |

---

## 8. 연습 문제

### 연습 1: 모던 C++ 리팩토링

기존 C++98/03 스타일 코드를 C++17 이상으로 리팩토링하세요.

### 연습 2: 타입 안전 설정 시스템

`std::variant`와 `std::optional`을 사용하여 타입 안전한 설정 관리 시스템을 구현하세요.

### 연습 3: 파이프라인 처리기

C++20 Ranges를 활용하여 데이터 처리 파이프라인을 구현하세요.

---

## 학습 완료

C++ 입문부터 모던 C++까지의 학습을 완료했습니다!

### 복습 추천 순서

1. 기초 복습: 01~06 (변수, 함수, 포인터)
2. OOP 복습: 07~09 (클래스, 상속)
3. STL 복습: 10~11 (컨테이너, 알고리즘)
4. 고급 복습: 12~15 (템플릿, 스마트 포인터, 모던 C++)

### 다음 학습 추천

- 디자인 패턴
- 멀티스레딩
- 네트워크 프로그래밍
- 실제 프로젝트 진행

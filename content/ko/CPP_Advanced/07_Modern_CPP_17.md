# 모던 C++ -- C++17

**이전**: [모던 C++ (C++11/14)](./06_Modern_CPP_11_14.md) | **다음**: [C++20 개념(Concepts)](./08_CPP20_Concepts.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 구조체, 쌍, 튜플, 배열, 맵에서 구조적 바인딩을 사용하여 구조화된 데이터 분해하기
2. SFINAE 패턴을 대체하는 컴파일 시간 분기를 위해 `if constexpr` 적용하기
3. 표현적인 데이터 모델링을 위해 어휘 타입 `std::optional`, `std::variant`, `std::any` 사용하기
4. 제로 복사 문자열 참조를 위해 `std::string_view` 활용하기
5. `std::filesystem`으로 파일 시스템 연산 수행하기
6. CTAD, 폴드 표현식, inline 변수 등 기타 C++17 기능 적용하기

---

C++17은 일상적인 C++ 코딩을 극적으로 개선하는 기능 모음을 제공했습니다. 구조적 바인딩은 `std::get`과 `std::tie`의 번거로움을 제거했습니다. 어휘 타입(`optional`, `variant`, `any`)은 임시방편 패턴을 표준적이고 잘 테스트된 대안으로 대체했습니다. `std::filesystem`은 이식 가능한 파일 연산을 표준 라이브러리에 도입했습니다. 그리고 `if constexpr`는 템플릿 코드를 다시 읽기 쉽게 만들었습니다. 이 레슨은 각 기능을 실용적인 예제와 함께 심층적으로 다룹니다.

## 1. 구조적 바인딩

구조적 바인딩을 사용하면 집합체, 쌍, 튜플, 배열을 이름이 있는 변수로 분해할 수 있습니다.

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <array>

std::tuple<int, double, std::string> getData() {
    return {42, 3.14, "Hello"};
}

struct Point { double x, y, z; };

int main() {
    // 튜플 분해
    auto [num, pi, str] = getData();
    std::cout << num << ", " << pi << ", " << str << "\n";

    // 쌍 분해
    std::pair<int, std::string> p = {1, "Alice"};
    auto [id, name] = p;
    std::cout << id << ": " << name << "\n";

    // 배열 분해
    int arr[] = {10, 20, 30};
    auto [a, b, c] = arr;
    std::cout << a << ", " << b << ", " << c << "\n";

    // 구조체 분해
    Point pt{1.0, 2.5, 3.7};
    auto [x, y, z] = pt;
    std::cout << "Point: " << x << ", " << y << ", " << z << "\n";

    // 맵 반복 (가장 일반적인 사용)
    std::map<std::string, int> ages = {
        {"Alice", 25}, {"Bob", 30}, {"Carol", 28}
    };
    for (const auto& [name, age] : ages) {
        std::cout << name << " is " << age << "\n";
    }

    // 참조로 (수정 가능)
    auto& [rx, ry, rz] = pt;
    rx = 100.0;  // pt.x를 수정
    std::cout << "Modified: " << pt.x << "\n";  // 100

    // if 문에서
    std::map<int, std::string> lookup = {{1, "one"}, {2, "two"}};
    if (auto [it, inserted] = lookup.insert({3, "three"}); inserted) {
        std::cout << "Inserted: " << it->second << "\n";
    }

    return 0;
}
```

---

## 2. if constexpr

거짓 분기를 완전히 버리는 컴파일 시간 분기로, 인스턴스화 에러를 방지합니다.

```cpp
#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// 복잡한 SFINAE 패턴을 대체
template<typename T>
std::string stringify(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        return "int:" + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float:" + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string:" + value;
    } else if constexpr (std::is_same_v<T, const char*>) {
        return "cstr:" + std::string(value);
    } else {
        // 이 분기는 도달할 때만 컴파일됨
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

// 컴파일 시간 재귀 튜플 처리
template<typename Tuple, std::size_t I = 0>
void printTuple(const Tuple& t) {
    if constexpr (I < std::tuple_size_v<Tuple>) {
        if constexpr (I > 0) std::cout << ", ";
        std::cout << std::get<I>(t);
        printTuple<Tuple, I + 1>(t);
    }
}

// 이종 컨테이너 연산
template<typename T>
auto getSize(const T& container) {
    if constexpr (requires { container.size(); }) {
        return container.size();
    } else if constexpr (std::is_array_v<T>) {
        return sizeof(T) / sizeof(T[0]);
    } else {
        return 1;  // 스칼라
    }
}

int main() {
    std::cout << stringify(42) << "\n";
    std::cout << stringify(3.14) << "\n";
    std::cout << stringify(std::string("hello")) << "\n";
    std::cout << stringify("world") << "\n";

    auto t = std::make_tuple(1, "hello", 3.14);
    printTuple(t);  // 1, hello, 3.14
    std::cout << "\n";

    return 0;
}
```

---

## 3. std::optional

존재하거나 존재하지 않을 수 있는 값을 나타냅니다. 센티넬 값과 포인터 기반 패턴을 대체합니다.

```cpp
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>

// 값을 반환하지 않을 수 있는 함수
std::optional<int> divide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

std::optional<std::string> findUser(int id) {
    if (id == 1) return "Alice";
    if (id == 2) return "Bob";
    return std::nullopt;
}

// 클래스 멤버로서의 optional
class Config {
    std::optional<int> port_;
    std::optional<std::string> host_;

public:
    void setPort(int p) { port_ = p; }
    void setHost(const std::string& h) { host_ = h; }

    int port() const { return port_.value_or(8080); }
    std::string host() const { return host_.value_or("localhost"); }
};

int main() {
    // 기본 사용법
    auto result = divide(10, 3);
    if (result) {
        std::cout << "Result: " << *result << "\n";  // 3
    }

    auto bad = divide(10, 0);
    std::cout << "has_value: " << bad.has_value() << "\n";  // false

    // value_or로 기본값 제공
    std::cout << divide(10, 3).value_or(-1) << "\n";  // 3
    std::cout << divide(10, 0).value_or(-1) << "\n";  // -1

    // value()는 비어있으면 std::bad_optional_access를 던짐
    try {
        auto v = bad.value();
    } catch (const std::bad_optional_access& e) {
        std::cout << "No value: " << e.what() << "\n";
    }

    // 문자열과 함께
    auto user = findUser(1);
    std::cout << "User: " << user.value_or("Unknown") << "\n";

    // 제자리 생성
    std::optional<std::vector<int>> ov(std::in_place, {1, 2, 3});
    std::cout << "Size: " << ov->size() << "\n";

    // Config 사용
    Config cfg;
    std::cout << cfg.host() << ":" << cfg.port() << "\n";  // localhost:8080
    cfg.setPort(3000);
    std::cout << cfg.host() << ":" << cfg.port() << "\n";  // localhost:3000

    return 0;
}
```

---

## 4. std::variant

언제든 정확히 하나의 대안 타입을 보유하는 타입 안전 공용체입니다.

```cpp
#include <iostream>
#include <variant>
#include <string>
#include <vector>

// 타입 안전 공용체
using Value = std::variant<int, double, std::string>;

// std::visit를 사용한 방문자 패턴
struct ValuePrinter {
    void operator()(int i) const { std::cout << "int: " << i; }
    void operator()(double d) const { std::cout << "double: " << d; }
    void operator()(const std::string& s) const { std::cout << "string: " << s; }
};

// 오버로드 패턴 (C++17 관용구)
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// JSON 유사 값 타입
using JsonValue = std::variant<
    std::nullptr_t, bool, int, double, std::string,
    std::vector<int>  // 간소화
>;

int main() {
    Value v = 42;
    std::cout << std::get<int>(v) << "\n";  // 42

    v = 3.14;
    std::cout << std::get<double>(v) << "\n";  // 3.14

    v = std::string("Hello");

    // 현재 타입 확인
    if (std::holds_alternative<std::string>(v)) {
        std::cout << "It's a string: " << std::get<std::string>(v) << "\n";
    }

    // get_if는 포인터 반환 (잘못된 타입이면 nullptr)
    if (auto* sp = std::get_if<std::string>(&v)) {
        std::cout << "String value: " << *sp << "\n";
    }

    // index()는 0 기반 타입 인덱스 반환
    std::cout << "Active index: " << v.index() << "\n";  // 2

    // 방문자 구조체를 사용한 std::visit
    Value values[] = {42, 3.14, std::string("Hello")};
    for (const auto& val : values) {
        std::visit(ValuePrinter{}, val);
        std::cout << "\n";
    }

    // 오버로드된 람다 패턴을 사용한 std::visit
    for (const auto& val : values) {
        std::visit(overloaded{
            [](int i) { std::cout << "int: " << i << "\n"; },
            [](double d) { std::cout << "double: " << d << "\n"; },
            [](const std::string& s) { std::cout << "str: " << s << "\n"; }
        }, val);
    }

    return 0;
}
```

---

## 5. std::any

어떤 단일 값이든 보유할 수 있는 타입 소거된 컨테이너입니다.

```cpp
#include <iostream>
#include <any>
#include <string>
#include <vector>

int main() {
    std::any a = 42;
    std::cout << std::any_cast<int>(a) << "\n";  // 42

    a = 3.14;
    std::cout << std::any_cast<double>(a) << "\n";

    a = std::string("Hello");
    std::cout << std::any_cast<std::string>(a) << "\n";

    // 타입 확인
    std::cout << "type: " << a.type().name() << "\n";
    std::cout << "has_value: " << a.has_value() << "\n";

    // 포인터로 안전한 캐스트 (타입 불일치 시 nullptr 반환)
    if (auto* p = std::any_cast<std::string>(&a)) {
        std::cout << "String: " << *p << "\n";
    }

    // 잘못된 타입은 std::bad_any_cast를 던짐
    try {
        auto val = std::any_cast<int>(a);  // a는 string을 보유!
    } catch (const std::bad_any_cast& e) {
        std::cout << "Bad cast: " << e.what() << "\n";
    }

    // 리셋
    a.reset();
    std::cout << "has_value after reset: " << a.has_value() << "\n";

    // 실용적 사용: 이종 컨테이너
    std::vector<std::any> config = {
        42,
        std::string("localhost"),
        true,
        3.14
    };

    return 0;
}
```

### optional vs variant vs any 사용 시기

| 타입 | 사용 시기 |
|------|----------|
| `std::optional<T>` | 값이 없을 수 있는 경우 (널 가능 단일 타입) |
| `std::variant<T, U, ...>` | 값이 알려진 타입 중 하나인 경우 (타입 안전 공용체) |
| `std::any` | 값 타입이 완전히 알 수 없는 경우 (타입 소거, 최후의 수단) |

---

## 6. std::string_view

문자열에 대한 비소유 참조입니다. 제로 복사, 경량, `std::string`과 `const char*` 모두와 호환됩니다.

```cpp
#include <iostream>
#include <string>
#include <string_view>

// 복사 없이 모든 문자열 타입을 받음
void printView(std::string_view sv) {
    std::cout << "View: " << sv
              << " (length: " << sv.length() << ")\n";
}

// 효율적인 부분 문자열
std::string_view getExtension(std::string_view filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string_view::npos) return "";
    return filename.substr(pos + 1);
}

// 토큰 파싱
void parseCSV(std::string_view line) {
    while (!line.empty()) {
        auto pos = line.find(',');
        auto token = line.substr(0, pos);
        std::cout << "[" << token << "] ";
        if (pos == std::string_view::npos) break;
        line.remove_prefix(pos + 1);
    }
    std::cout << "\n";
}

int main() {
    // 모든 문자열 타입과 동작
    std::string str = "Hello, World!";
    const char* cstr = "Hello from C!";

    printView(str);
    printView(cstr);
    printView("Literal string");

    // 부분 문자열 (복사 없음!)
    std::string_view sv = "Hello, World!";
    auto sub = sv.substr(0, 5);
    std::cout << "Substring: " << sub << "\n";  // Hello

    // 확장자 파싱
    std::cout << "Extension: " << getExtension("main.cpp") << "\n";
    std::cout << "Extension: " << getExtension("archive.tar.gz") << "\n";

    // CSV 파싱
    parseCSV("Alice,25,New York");

    // 주의: 댕글링 string_view
    // std::string_view bad;
    // {
    //     std::string temp = "temporary";
    //     bad = temp;  // bad가 temp의 버퍼를 가리킴
    // }
    // std::cout << bad;  // 미정의 동작: temp가 파괴됨

    return 0;
}
```

---

## 7. std::filesystem

Boost.Filesystem에서 표준화된 이식 가능한 파일 시스템 연산입니다.

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    // 경로 연산
    fs::path p = "/home/user/documents/file.txt";
    std::cout << "filename:   " << p.filename() << "\n";
    std::cout << "stem:       " << p.stem() << "\n";
    std::cout << "extension:  " << p.extension() << "\n";
    std::cout << "parent:     " << p.parent_path() << "\n";
    std::cout << "root:       " << p.root_path() << "\n";

    // 경로 연결
    fs::path dir = "/home/user";
    fs::path file = "document.txt";
    fs::path full = dir / file;
    std::cout << "Combined: " << full << "\n";

    // 현재 디렉토리
    std::cout << "CWD: " << fs::current_path() << "\n";

    // 존재 여부와 타입 확인
    fs::path testPath = ".";
    std::cout << "exists: " << fs::exists(testPath) << "\n";
    std::cout << "is_dir: " << fs::is_directory(testPath) << "\n";

    // 디렉토리 반복
    std::cout << "\n=== Current directory ===\n";
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path().filename();
        if (entry.is_directory()) {
            std::cout << " [DIR]";
        } else {
            std::cout << " [" << entry.file_size() << " bytes]";
        }
        std::cout << "\n";
    }

    // 재귀 디렉토리 반복
    // for (const auto& entry : fs::recursive_directory_iterator(".")) { ... }

    // 파일 연산 (예상된 실패에 에러 코드 사용)
    std::error_code ec;
    fs::create_directories("/tmp/test/subdir", ec);
    if (!ec) {
        std::cout << "Directories created\n";
    }

    // 복사, 이름 변경, 삭제
    // fs::copy("source.txt", "dest.txt", ec);
    // fs::rename("old.txt", "new.txt", ec);
    // fs::remove("file.txt", ec);
    // auto removed = fs::remove_all("/tmp/test", ec);  // 재귀

    // 파일 크기와 마지막 수정 시간
    // auto size = fs::file_size("file.txt");
    // auto time = fs::last_write_time("file.txt");

    return 0;
}
```

---

## 8. 폴드 표현식

C++17 폴드 표현식은 매개변수 팩에 이항 연산자를 적용합니다.

```cpp
#include <iostream>

// 모든 인수 합산
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

// 공백으로 구분하여 출력
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << " "), ...);
    std::cout << "\n";
}

// 모든 조건 확인
template<typename... Args>
bool all(Args... args) {
    return (args && ...);
}

// 모두 vector에 push
#include <vector>
template<typename T, typename... Args>
void pushAll(std::vector<T>& vec, Args&&... args) {
    (vec.push_back(std::forward<Args>(args)), ...);
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << "\n";  // 15
    print(1, "hello", 3.14);  // 1 hello 3.14

    std::vector<int> v;
    pushAll(v, 1, 2, 3, 4, 5);
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
```

---

## 9. 클래스 템플릿 인수 추론 (CTAD)

C++17은 컴파일러가 생성자 인수에서 클래스 템플릿 인수를 추론하도록 합니다.

```cpp
#include <iostream>
#include <vector>
#include <tuple>
#include <mutex>
#include <optional>

int main() {
    // C++17 이전: 명시적 템플릿 인수
    std::pair<int, double> p1(1, 3.14);
    std::tuple<int, double, std::string> t1(1, 3.14, "hello");

    // C++17: CTAD가 타입을 추론
    std::pair p2(1, 3.14);                  // pair<int, double>
    std::tuple t2(1, 3.14, "hello");        // tuple<int, double, const char*>
    std::optional o(42);                    // optional<int>
    std::vector v{1, 2, 3, 4};             // vector<int>

    // 커스텀 클래스를 위한 추론 가이드
    // template<typename T>
    // class MyContainer {
    //     T value;
    // public:
    //     MyContainer(T v) : value(v) {}
    // };
    // // 생성자로부터의 암묵적 추론 가이드
    // MyContainer mc(42);  // MyContainer<int>

    // lock_guard CTAD
    std::mutex mtx;
    std::lock_guard lock(mtx);  // lock_guard<std::mutex>

    std::cout << p2.first << ", " << p2.second << "\n";

    return 0;
}
```

---

## 10. 기타 C++17 기능

### inline 변수

```cpp
// header.h
// C++17: inline 변수를 헤더에서 정의 가능
struct Config {
    static inline int maxRetries = 3;
    static inline std::string defaultHost = "localhost";
};

// 네임스페이스 범위에서도 동작
inline constexpr int VERSION = 17;
```

### 중첩 네임스페이스

```cpp
// C++17 이전
namespace A { namespace B { namespace C {
    void func() {}
}}}

// C++17
namespace A::B::C {
    void func() {}
}
```

### [[nodiscard]]

```cpp
#include <iostream>

[[nodiscard]] int computeValue() {
    return 42;
}

[[nodiscard("Error codes must not be ignored")]]
int openFile(const char* path) {
    return 0;  // 성공
}

class [[nodiscard]] ErrorCode {
    int code_;
public:
    ErrorCode(int c) : code_(c) {}
};

int main() {
    // computeValue();  // 경고: 반환값 무시
    int v = computeValue();  // OK

    // openFile("test.txt");  // 커스텀 메시지와 함께 경고
    int err = openFile("test.txt");  // OK

    std::cout << v << "\n";
    return 0;
}
```

### [[maybe_unused]]와 [[fallthrough]]

```cpp
#include <iostream>

void example([[maybe_unused]] int debugValue) {
    // 릴리스 빌드에서 debugValue가 사용되지 않아도 경고 없음
    #ifdef DEBUG
    std::cout << debugValue << "\n";
    #endif
}

void handleStatus(int status) {
    switch (status) {
        case 0:
            std::cout << "Success\n";
            break;
        case 1:
            std::cout << "Warning: ";
            [[fallthrough]];  // 의도적 폴스루
        case 2:
            std::cout << "Continuing...\n";
            break;
    }
}
```

### 초기화를 동반한 if/switch

```cpp
#include <iostream>
#include <map>

int main() {
    std::map<int, std::string> db = {{1, "Alice"}, {2, "Bob"}};

    // 초기화를 동반한 if
    if (auto it = db.find(1); it != db.end()) {
        std::cout << "Found: " << it->second << "\n";
    }
    // 'it'은 여기서 보이지 않음

    // 초기화를 동반한 switch
    switch (auto val = 2 * 3; val) {
        case 6: std::cout << "Six\n"; break;
        default: std::cout << "Other: " << val << "\n";
    }

    return 0;
}
```

---

## 요약

| 기능 | 카테고리 | 핵심 이점 |
|------|----------|----------|
| 구조적 바인딩 | 문법 | 깔끔한 분해 |
| `if constexpr` | 템플릿 | 가독성 높은 컴파일 시간 분기 |
| `std::optional` | 어휘 | 널 가능 값 |
| `std::variant` | 어휘 | 타입 안전 공용체 |
| `std::any` | 어휘 | 타입 소거된 컨테이너 |
| `std::string_view` | 성능 | 제로 복사 문자열 참조 |
| `std::filesystem` | 라이브러리 | 이식 가능한 파일 연산 |
| 폴드 표현식 | 템플릿 | 간소화된 팩 연산 |
| CTAD | 템플릿 | 템플릿 인수 상용구 감소 |
| inline 변수 | 링크 | 헤더에서 정의된 변수 |
| `[[nodiscard]]` | 안전성 | 무시된 반환값 방지 |

---

## 연습문제

### 연습문제 1: 설정 파서

`std::variant`, `std::optional`, `std::string_view`를 사용하여 문자열, 정수, 부동소수점, 불리언 값을 처리하는 설정 파서를 구현하세요.

### 연습문제 2: 파일 검색

`std::filesystem`을 사용하여 디렉토리에서 패턴(확장자 또는 이름 부분 문자열)에 맞는 파일을 재귀적으로 검색하는 함수를 작성하세요.

### 연습문제 3: Variant 계산기

피연산자가 `std::variant<int, double>`인 계산기를 구현하고, `std::visit`를 사용하여 올바른 결과 타입을 생성하는 연산을 구현하세요.

### 연습문제 4: 문자열 토크나이저

메모리 할당 없이 구분자로 문자열을 분할하는 `std::string_view`를 사용한 토크나이저를 작성하세요.

### 연습문제 5: 타입 안전 설정

각 설정이 허용된 타입의 `std::variant`인 `Settings` 클래스를 만드세요. 오버로드된 패턴과 함께 `std::visit`를 사용하여 설정을 문자열 형식으로 직렬화하세요.

---

## 다음 단계

C++20은 템플릿을 제약하는 혁명적인 접근 방식인 개념(Concepts)을 도입했습니다. [08_CPP20_Concepts.md](./08_CPP20_Concepts.md)에서 탐구해 봅시다.

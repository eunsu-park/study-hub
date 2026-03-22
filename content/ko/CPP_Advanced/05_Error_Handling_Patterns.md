# 에러 처리 패턴

**이전**: [스마트 포인터와 RAII](./04_Smart_Pointers_and_RAII.md) | **다음**: [모던 C++ (C++11/14)](./06_Modern_CPP_11_14.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 세 가지 수준의 예외 안전성 보장(기본, 강력, 무던짐) 적용하기
2. `noexcept` 지정을 사용하여 컴파일러 최적화 활성화하기
3. 에러 코드, 예외, `std::expected`를 사용한 에러 처리 전략 설계하기
4. 예외 안전한 정리를 위한 범위 가드(scope guard) 패턴 구현하기
5. 생성자, 소멸자, 이동 연산에서의 에러 처리하기

---

에러 처리는 모든 C++ 코드베이스에서 가장 중요한 설계 결정 중 하나입니다. 잘못된 전략은 자원 누수, 조용한 실패, 또는 에러 검사로 가득 찬 읽기 어려운 코드로 이어집니다. C++는 예외, 에러 코드, 그리고 새로운 `std::expected` 등 여러 메커니즘을 제공하며, 각각 고유한 트레이드오프가 있습니다. 이 레슨은 이들 중 선택하기 위한 프레임워크를 구축하고, 일이 잘못되더라도 올바르게 유지되는 코드를 작성하는 방법을 보여줍니다.

## 1. 예외 안전성 보장

C++ 커뮤니티는 세 가지 수준의 예외 안전성을 정의합니다. 작성하는 모든 함수는 최소한 기본 보장을 제공해야 합니다.

| 보장 | 약속 | 예시 |
|------|------|------|
| **무던짐(Nothrow)** | 연산이 절대 던지지 않음 | `std::swap`, 소멸자, 이동 연산 |
| **강력(Strong)** | 예외가 던져지면 상태가 변하지 않음 (커밋 또는 롤백) | `std::vector::push_back` |
| **기본(Basic)** | 예외가 던져지면 불변식이 보존되고 누수 없음 | 대부분의 표준 라이브러리 연산 |

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class Account {
    double balance_;
    std::string owner_;

public:
    Account(std::string owner, double balance)
        : balance_(balance), owner_(std::move(owner)) {}

    // 무던짐 보장
    double balance() const noexcept { return balance_; }

    // 강력 보장: 이체가 성공하거나 아무것도 변하지 않음
    void transfer(Account& to, double amount) {
        if (amount > balance_) {
            throw std::runtime_error("Insufficient funds");
        }
        // 두 연산 모두 noexcept (double 뺄셈/덧셈)
        balance_ -= amount;
        to.balance_ += amount;
    }

    // 기본 보장: 객체는 유효하지만 상태가 변경될 수 있음
    void addTransaction(std::vector<std::string>& log, double amount) {
        balance_ += amount;  // noexcept
        log.push_back(owner_ + ": " + std::to_string(amount));
        // push_back이 던지면 balance_는 이미 변경됨
        // 객체는 여전히 유효하지만 상태가 부분적으로 갱신됨
    }
};

int main() {
    Account alice("Alice", 1000.0);
    Account bob("Bob", 500.0);

    try {
        alice.transfer(bob, 2000.0);  // 던짐: 잔액 부족
    } catch (const std::runtime_error& e) {
        std::cout << "Error: " << e.what() << "\n";
        // 강력 보장: 잔액 변경 없음
        std::cout << "Alice: " << alice.balance() << "\n";  // 1000
        std::cout << "Bob: " << bob.balance() << "\n";      // 500
    }

    return 0;
}
```

---

## 2. noexcept

`noexcept` 지정자는 함수가 예외를 던지지 않음을 선언합니다. 이는 중요한 컴파일러 최적화를 가능하게 하며 특정 STL 연산에 필요합니다.

### 기본 noexcept

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

class Widget {
    int* data_;
    size_t size_;

public:
    Widget(size_t n) : data_(new int[n]()), size_(n) {}
    ~Widget() noexcept { delete[] data_; }

    // 이동 연산은 vector 재할당 최적화를 위해 반드시 noexcept여야 함
    Widget(Widget&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Widget& operator=(Widget&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 복사 (할당으로 인해 던질 수 있음)
    Widget(const Widget& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
};

int main() {
    // noexcept 연산자: 컴파일 시간에 쿼리
    std::cout << std::boolalpha;
    std::cout << "Widget move is noexcept: "
              << std::is_nothrow_move_constructible_v<Widget> << "\n";  // true
    std::cout << "Widget copy is noexcept: "
              << std::is_nothrow_copy_constructible_v<Widget> << "\n";  // false

    // vector는 noexcept일 때 이동, 아니면 복사 사용
    std::vector<Widget> vec;
    vec.reserve(1);
    vec.emplace_back(100);
    vec.emplace_back(200);  // 재할당: 이동 사용 (noexcept)

    return 0;
}
```

### 조건부 noexcept

```cpp
#include <type_traits>

// noexcept가 포함된 연산의 던짐 여부에 따라 결정
template<typename T>
void swapValues(T& a, T& b)
    noexcept(std::is_nothrow_move_constructible_v<T> &&
             std::is_nothrow_move_assignable_v<T>) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

// 호출된 함수로부터 noexcept 전파
template<typename F, typename... Args>
decltype(auto) callNoexcept(F&& f, Args&&... args)
    noexcept(noexcept(f(std::forward<Args>(args)...))) {
    return f(std::forward<Args>(args)...);
}
```

---

## 3. 예외 안전 코드

### 복사 후 교환을 통한 강력 보장

```cpp
#include <iostream>
#include <algorithm>

class StrongSafe {
    int* data_;
    size_t size_;

public:
    StrongSafe(size_t n) : data_(new int[n]()), size_(n) {}
    ~StrongSafe() { delete[] data_; }

    friend void swap(StrongSafe& a, StrongSafe& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
    }

    // 복사 생성자 (던질 수 있음)
    StrongSafe(const StrongSafe& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // 강력 보장: temp의 복사 생성이 실패하면
    // *this는 완전히 변하지 않음
    StrongSafe& operator=(StrongSafe other) noexcept {
        swap(*this, other);
        return *this;
    }

    // 이동 생성자 (noexcept)
    StrongSafe(StrongSafe&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    size_t size() const noexcept { return size_; }
};
```

### RAII + 예외

```cpp
#include <iostream>
#include <memory>
#include <fstream>
#include <stdexcept>

// 나쁨: 예외 시 수동 정리
void badExample() {
    int* data = new int[1000];
    // readFile이 던지면 data가 누수됨!
    // readFile(data);
    delete[] data;
}

// 좋음: RAII가 자동으로 정리 처리
void goodExample() {
    auto data = std::make_unique<int[]>(1000);
    // readFile이 던져도 unique_ptr 소멸자가 data를 해제
    // readFile(data.get());
}

// 다중 자원: 모두 RAII로 보호
void multiResource() {
    auto file = std::fstream("data.txt", std::ios::out);
    auto buffer = std::make_unique<char[]>(4096);
    auto connection = std::make_unique<int>(42);  // 시뮬레이션
    // 어떤 연산이 던져도 모든 자원이 생성 역순으로 정리됨
}
```

---

## 4. 에러 코드 vs 예외

### 각각 사용할 때

| 기준 | 에러 코드 | 예외 |
|------|-----------|------|
| 예상된 실패 | 선호 | 과도함 |
| 예상치 못한 실패 | 쉽게 무시됨 | 선호 |
| 성능이 중요한 핫 경로 | 선호 | 던짐 시 오버헤드 |
| 생성자 | 코드 반환 불가 | 선호 |
| 깊은 호출 체인 | 전파가 번거로움 | 자동 전파 |

### std::error_code와 std::system_error

```cpp
#include <iostream>
#include <system_error>
#include <fstream>
#include <cerrno>
#include <cstring>

// 예상된 실패에 error_code 사용
std::error_code openFile(const std::string& path, std::fstream& out) {
    out.open(path, std::ios::in);
    if (!out.is_open()) {
        return std::make_error_code(std::errc::no_such_file_or_directory);
    }
    return {};  // 에러 없음
}

// 예상치 못한 실패에 예외 사용
void processFile(const std::string& path) {
    std::fstream file;
    if (auto ec = openFile(path, file)) {
        // 실패가 예상되지 않을 때 예외로 변환
        throw std::system_error(ec, "Cannot process " + path);
    }
    // 파일 처리...
}

int main() {
    // 에러 코드: 호출자가 처리 방법 결정
    std::fstream file;
    if (auto ec = openFile("missing.txt", file)) {
        std::cout << "Error: " << ec.message() << "\n";
        // 카테고리와 코드 확인 가능
        if (ec == std::errc::no_such_file_or_directory) {
            std::cout << "File not found, using defaults\n";
        }
    }

    // 예외: 자동 전파
    try {
        processFile("missing.txt");
    } catch (const std::system_error& e) {
        std::cout << "System error: " << e.what() << "\n";
        std::cout << "Code: " << e.code() << "\n";
    }

    return 0;
}
```

---

## 5. std::expected (C++23)

`std::expected<T, E>`는 모나딕 에러 처리를 제공합니다: 예상 결과이거나 에러인 값입니다. 에러 코드의 명시성과 예외의 합성 가능성을 결합합니다.

```cpp
#include <iostream>
#include <string>
#include <cmath>

// C++23 이전을 위한 std::expected 시뮬레이션
// C++23에서는 #include <expected> 사용
#if __cplusplus >= 202302L
#include <expected>
using std::expected;
using std::unexpected;
#else
// 데모용 간소화된 폴리필
template<typename T, typename E>
class expected {
    bool has_val_;
    union { T val_; E err_; };
public:
    expected(T val) : has_val_(true), val_(std::move(val)) {}
    expected(E err, bool) : has_val_(false), err_(std::move(err)) {}
    bool has_value() const { return has_val_; }
    T& value() { return val_; }
    E& error() { return err_; }
    T value_or(T default_val) { return has_val_ ? val_ : default_val; }
    explicit operator bool() const { return has_val_; }
    T& operator*() { return val_; }
    ~expected() { if (has_val_) val_.~T(); else err_.~E(); }
};

template<typename E>
auto make_unexpected(E e) { return expected<int, E>(std::move(e), false); }
#endif

// 에러 타입
enum class MathError {
    DivisionByZero,
    NegativeSqrt,
    Overflow
};

std::string to_string(MathError e) {
    switch (e) {
        case MathError::DivisionByZero: return "division by zero";
        case MathError::NegativeSqrt: return "negative sqrt";
        case MathError::Overflow: return "overflow";
    }
    return "unknown";
}

// expected를 반환하는 함수
expected<double, MathError> safeDivide(double a, double b) {
    if (b == 0.0) return expected<double, MathError>(MathError::DivisionByZero, false);
    return a / b;
}

expected<double, MathError> safeSqrt(double x) {
    if (x < 0.0) return expected<double, MathError>(MathError::NegativeSqrt, false);
    return std::sqrt(x);
}

int main() {
    auto result = safeDivide(10.0, 3.0);
    if (result) {
        std::cout << "10 / 3 = " << *result << "\n";
    }

    auto bad = safeDivide(10.0, 0.0);
    if (!bad) {
        std::cout << "Error: " << to_string(bad.error()) << "\n";
    }

    // value_or로 기본값 제공
    std::cout << "Result: " << safeDivide(10.0, 0.0).value_or(-1.0) << "\n";

    return 0;
}
```

### 모나딕 연산 (C++23)

```cpp
// C++23 std::expected는 모나딕 체이닝을 지원:
// auto result = getData(id)
//     .and_then(validate)      // 값이 있으면 체이닝
//     .transform(serialize)    // 값을 매핑
//     .or_else(handleError);   // 에러 처리

// 예시 (C++23):
// std::expected<double, MathError> compute(double x) {
//     return safeSqrt(x)
//         .and_then([](double v) { return safeDivide(1.0, v); })
//         .transform([](double v) { return v * 100; });
// }
```

---

## 6. 범위 가드(Scope Guards)

범위 가드는 범위를 벗어날 때 정리 코드를 실행하여, RAII 래퍼 클래스 없이도 예외 안전한 정리를 제공합니다.

```cpp
#include <iostream>
#include <functional>
#include <exception>

// 간단한 범위 가드
class ScopeGuard {
    std::function<void()> cleanup_;
    bool dismissed_ = false;

public:
    explicit ScopeGuard(std::function<void()> cleanup)
        : cleanup_(std::move(cleanup)) {}

    ~ScopeGuard() {
        if (!dismissed_ && cleanup_) {
            cleanup_();
        }
    }

    void dismiss() { dismissed_ = true; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

// 편의 매크로
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define SCOPE_EXIT auto CONCAT(scope_guard_, __LINE__) = ScopeGuard

// 사용
void processTransaction() {
    std::cout << "Begin transaction\n";

    ScopeGuard rollback([&]() {
        std::cout << "Rolling back transaction\n";
    });

    // 던질 수 있는 작업 수행...
    std::cout << "Doing work...\n";

    // 여기까지 오면 커밋 성공
    rollback.dismiss();
    std::cout << "Transaction committed\n";
}

void fileOperation() {
    FILE* f = fopen("/tmp/test.txt", "w");
    if (!f) return;

    // 어떤 종료에서든 파일이 닫히도록 보장
    ScopeGuard closeFile([&]() {
        std::cout << "Closing file\n";
        fclose(f);
    });

    fprintf(f, "Hello\n");
    // 예외가 발생하더라도 파일은 여전히 닫힘
}

int main() {
    processTransaction();
    std::cout << "---\n";
    fileOperation();

    return 0;
}
```

---

## 7. 특수 멤버에서의 에러 처리

### 생성자

생성자는 C++에서 생성 실패를 알리는 유일한 신뢰할 수 있는 방법입니다.

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>

class Connection {
    int fd_;

public:
    // 생성자는 실패 시 예외를 던질 수 있고 던져야 함
    Connection(const std::string& host, int port) {
        fd_ = -1;  // 시뮬레이션된 연결
        if (host.empty()) {
            throw std::invalid_argument("Empty host");
        }
        // 이것이 던지면 소멸자가 실행되지 않음 (객체가 완전히 생성되지 않음)
        // 하지만 이미 생성된 멤버의 소멸자는 실행됨
        fd_ = 42;  // 시뮬레이션된 성공적인 연결
        std::cout << "Connected to " << host << ":" << port << "\n";
    }

    ~Connection() {
        if (fd_ >= 0) {
            std::cout << "Disconnecting\n";
            // close(fd_);
        }
    }
};

// 다중 자원 생성자: 스마트 포인터 사용
class Server {
    std::unique_ptr<Connection> db_;
    std::unique_ptr<Connection> cache_;

public:
    Server() {
        db_ = std::make_unique<Connection>("db.local", 5432);
        // 이것이 던지면 db_가 자동으로 정리됨
        cache_ = std::make_unique<Connection>("cache.local", 6379);
    }
};
```

### 소멸자

소멸자는 **절대** 던지지 말아야 합니다. 다른 예외가 이미 진행 중인 상태에서 던지면 `std::terminate`가 호출됩니다.

```cpp
class SafeCleanup {
public:
    ~SafeCleanup() noexcept {
        try {
            // 위험한 정리 작업
        } catch (...) {
            // 로깅하되 다시 던지지 않음
            // std::cerr << "Cleanup failed\n";
        }
    }
};
```

### 이동 연산

이동 연산은 가능한 한 `noexcept`여야 합니다. 던지는 이동은 많은 STL 컨텍스트에서 이동 의미론의 목적을 무효화합니다.

```cpp
class Buffer {
    int* data_;
    size_t size_;

public:
    // 이동: noexcept, 포인터 교환만
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 이동이 반드시 던질 수 있는 작업을 해야 하는 경우,
    // 명확히 문서화하고 결과를 수용
};
```

---

## 8. 모범 사례

### 에러 처리 결정 매트릭스

```
에러가 예상되고 일반적인가?
  ├── 예 → 에러 코드 또는 std::expected 사용
  └── 아니오 → 프로그래밍 에러(버그)인가?
        ├── 예 → assert() / std::terminate 사용
        └── 아니오 → 예외 사용
```

### 요약 표

| 패턴 | 사용 시기 | 예시 |
|------|----------|------|
| RAII | 항상 자원 관리에 | 스마트 포인터, lock_guard |
| 예외 | 예상치 못한 에러, 생성자 | 파일 없음, 메모리 부족 |
| 에러 코드 | 예상된 실패, 핫 경로 | 네트워크 타임아웃, 파싱 실패 |
| `std::expected` | 합성 가능한 에러 처리 | 데이터 파이프라인 단계 |
| `noexcept` | 이동 연산, 소멸자, swap | 이동 생성자 |
| 범위 가드 | RAII 클래스 없는 임시 정리 | 트랜잭션 롤백 |
| `assert` | 프로그래밍 에러 (디버그 전용) | 전제조건 위반 |

### 가이드라인

1. **어디서나 RAII 사용** -- 모든 자원은 RAII 객체가 소유해야 함
2. **이동 연산에 noexcept 표시** -- STL 최적화 활성화
3. **소멸자에서 절대 던지지 않기** -- `std::terminate` 야기 가능
4. **합성 가능한 에러에는 `std::expected` 선호** (C++23)
5. **예외는 예외적 상황에만 사용**, 제어 흐름에는 사용하지 않기
6. **모든 함수의 예외 안전성 보장을 문서화**

---

## 연습문제

### 연습문제 1: 예외 안전 스택

`push`에 대해 강력한 예외 안전성 보장을 제공하는 스택을 구현하세요. 내부 할당이 실패하면 스택이 변하지 않아야 합니다.

### 연습문제 2: 범위 가드 구현

`std::uncaught_exceptions()`를 사용하여 `ScopeExit`, `ScopeSuccess`(정상 종료 시에만 실행), `ScopeFail`(예외 시에만 실행)을 구현하세요.

### 연습문제 3: Expected 파이프라인

`std::expected`(또는 폴리필)를 사용하여 데이터 처리 파이프라인을 구현하세요: `readFile -> parseCSV -> validate -> transform`. 각 단계는 설명적인 에러로 실패할 수 있습니다.

### 연습문제 4: 트랜잭션 클래스

일련의 연산을 수집하는 `Transaction` 클래스를 작성하세요. `commit()` 시 모든 연산이 실행됩니다. 하나라도 던지면 이전에 실행된 모든 연산이 롤백됩니다.

### 연습문제 5: noexcept 감사

이동 연산이 있는 기존 클래스를 가져와서 `noexcept`로 표시할 수 있는지 분석하세요. 이동 중 불필요하게 할당하거나 던지는 연산을 수정하세요.

---

## 다음 단계

모던 C++11과 C++14는 C++ 작성 방식을 바꾼 풍부한 기능을 도입했습니다. [06_Modern_CPP_11_14.md](./06_Modern_CPP_11_14.md)에서 `auto`, 람다, `constexpr` 등을 탐구해 봅시다.

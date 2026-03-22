# 이동 의미론 심화

**이전**: [C++ 고급](./00_Overview.md) | **다음**: [템플릿](./02_Templates.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 표현식 컨텍스트에서 lvalue, rvalue, xvalue, prvalue, glvalue를 구분하기
2. 이동 생성자와 이동 대입 연산자를 올바르게 구현하기
3. 효율적인 자원 전달을 위해 `std::move`와 `std::forward` 적용하기
4. 5의 법칙(Rule of Five)과 0의 법칙(Rule of Zero)에 따라 클래스 설계하기
5. 복사 생략(Copy Elision), RVO, NRVO 최적화 설명하기

---

이동 의미론(Move Semantics)은 C++11에서 도입되어 C++의 자원 소유권 처리 방식을 근본적으로 변화시켰습니다. 이동 의미론 이전에는 함수에서 큰 객체를 반환하면 소스가 곧 파괴될 때조차 모든 바이트를 복사해야 했습니다. 안전하게 "빼앗을 수 있는" 객체(rvalue)와 보존해야 하는 객체(lvalue)를 구분함으로써, 컴파일러와 프로그래머가 협력하여 불필요한 복사를 제거할 수 있습니다. 값 카테고리와 이동 연산을 마스터하는 것은 고성능 모던 C++을 작성하는 데 필수적입니다.

## 1. 값 카테고리(Value Categories)

모든 C++ 표현식은 두 가지 독립적인 속성을 가집니다: **타입**과 **값 카테고리**. C++11은 세분화된 값 카테고리 분류 체계를 도입했습니다.

```
          expression
          /        \
       glvalue    rvalue
       /    \     /    \
    lvalue  xvalue  prvalue
```

| 카테고리 | 식별성(Identity)? | 이동 가능? | 예시 |
|----------|:---:|:---:|---------|
| **lvalue** | 있음 | 아니오 | 변수, 문자열 리터럴, `*ptr`, `arr[i]` |
| **xvalue** | 있음 | 예 | `std::move(x)`, `T&&`로 캐스트 |
| **prvalue** | 없음 | 예 | 리터럴(`42`, `true`), `x + y`, 값으로 반환하는 함수 |
| **glvalue** | 있음 | -- | lvalue 또는 xvalue |
| **rvalue** | -- | 예 | xvalue 또는 prvalue |

```cpp
#include <iostream>
#include <string>

int getValue() { return 42; }
int& getRef(int& x) { return x; }

int main() {
    int x = 10;

    // lvalue 예시
    int& ref = x;            // x는 lvalue
    int* ptr = &x;           // lvalue의 주소를 가져올 수 있음

    // prvalue 예시
    // int& bad = 42;        // 에러: lvalue 참조에 prvalue를 바인딩할 수 없음
    const int& ok = 42;      // OK: const lvalue 참조는 수명을 연장
    int&& rref = 42;         // OK: rvalue 참조는 prvalue에 바인딩

    // xvalue 예시
    int&& moved = std::move(x);  // std::move(x)는 xvalue

    // 카테고리 구분
    // &getValue();           // 에러: prvalue는 주소가 없음
    &getRef(x);              // OK: lvalue는 주소가 있음

    return 0;
}
```

---

## 2. Rvalue 참조

rvalue 참조(`T&&`)는 rvalue에 바인딩되는 참조입니다. 이를 통해 컴파일러가 복사(lvalue로부터)와 이동(rvalue로부터)을 구분할 수 있습니다.

```cpp
#include <iostream>

void process(int& x)  { std::cout << "lvalue: " << x << "\n"; }
void process(int&& x) { std::cout << "rvalue: " << x << "\n"; }

int main() {
    int a = 10;

    process(a);             // lvalue 오버로드
    process(20);            // rvalue 오버로드
    process(std::move(a));  // rvalue 오버로드 (a가 xvalue로 캐스트됨)

    // rvalue 참조는 임시 객체의 수명을 연장
    int&& rref = 42;       // 임시 객체는 rref만큼 존재
    rref = 100;             // rvalue 참조를 통해 수정 가능
    std::cout << rref << "\n";  // 100

    // 중요: 이름이 있는 rvalue 참조는 그 자체가 lvalue!
    int&& r = 5;
    // process(r);          // lvalue 오버로드 호출! r은 이름이 있는 변수
    process(std::move(r));  // rvalue 오버로드 호출

    return 0;
}
```

### 바인딩 규칙

| 참조 타입 | lvalue에 바인딩? | rvalue에 바인딩? |
|-----------|:---:|:---:|
| `T&` | 예 | 아니오 |
| `const T&` | 예 | 예 |
| `T&&` | 아니오 | 예 |
| `const T&&` | 아니오 | 예 (드물게 사용) |

---

## 3. 이동 생성자

이동 생성자는 소스 객체로부터 자원의 소유권을 전달하며, 소스를 유효하지만 미지정 상태로 남깁니다.

```cpp
#include <iostream>
#include <cstring>
#include <utility>

class String {
private:
    char* data_;
    size_t size_;

public:
    // 생성자
    String(const char* str = "") {
        size_ = std::strlen(str);
        data_ = new char[size_ + 1];
        std::strcpy(data_, str);
        std::cout << "Constructed: \"" << data_ << "\"\n";
    }

    // 복사 생성자
    String(const String& other)
        : size_(other.size_), data_(new char[other.size_ + 1]) {
        std::strcpy(data_, other.data_);
        std::cout << "Copied: \"" << data_ << "\"\n";
    }

    // 이동 생성자
    String(String&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        // 자원을 빼앗음
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "Moved: \"" << data_ << "\"\n";
    }

    // 소멸자
    ~String() {
        std::cout << "Destroyed: \""
                  << (data_ ? data_ : "null") << "\"\n";
        delete[] data_;
    }

    const char* c_str() const { return data_ ? data_ : ""; }
    size_t size() const { return size_; }
};

int main() {
    String s1("Hello");
    String s2 = s1;              // 복사 생성자
    String s3 = std::move(s1);   // 이동 생성자
    // s1은 이제 이동된 상태 (data_ == nullptr)

    std::cout << "s2: " << s2.c_str() << "\n";
    std::cout << "s3: " << s3.c_str() << "\n";
    std::cout << "s1: " << s1.c_str() << " (moved-from)\n";

    return 0;
}
```

### noexcept가 중요한 이유

이동 연산에서 `noexcept` 지정자는 매우 중요합니다. `std::vector`와 같은 STL 컨테이너는 재할당 시 `noexcept`로 표시된 경우에만 이동 생성자를 사용합니다. 그렇지 않으면 예외 안전성을 위해 복사로 대체합니다.

```cpp
#include <vector>
#include <iostream>

class Widget {
public:
    // noexcept 없으면, vector::push_back이 이동 대신 복사함
    Widget(Widget&& other) noexcept { /* ... */ }
    Widget& operator=(Widget&& other) noexcept { /* ... */ }
};
```

---

## 4. 이동 대입

이동 대입 연산자는 하나의 기존 객체에서 다른 객체로 자원을 전달합니다.

```cpp
#include <iostream>
#include <algorithm>
#include <cstring>

class Buffer {
private:
    int* data_;
    size_t size_;

public:
    Buffer(size_t n) : data_(new int[n]()), size_(n) {}

    ~Buffer() { delete[] data_; }

    // 복사 대입
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    // 이동 대입
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data_;       // 현재 자원 해제
            data_ = other.data_;  // 자원을 빼앗음
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    size_t size() const { return size_; }
};

int main() {
    Buffer b1(100);
    Buffer b2(50);

    b2 = std::move(b1);  // 이동 대입
    std::cout << "b2 size: " << b2.size() << "\n";  // 100

    return 0;
}
```

### 이동된 후 상태(Moved-From State)

`std::move` 이후, 소스 객체는 **유효하지만 미지정 상태(valid but unspecified)**로 남습니다. C++ 표준은 안전하게 재대입하거나 소멸시킬 수 있음만 보장합니다 — 하지만 대부분의 타입에서 값을 읽는 것은 미정의 동작(undefined behavior)입니다.

```cpp
#include <string>

int main() {
    std::string s = "hello";
    std::string t = std::move(s);

    // s는 유효하지만 미지정 상태 — 읽지 말 것
    // std::cout << s;  // 미정의 동작: "" 출력될 수도, 크래시될 수도 있음

    s = "world";       // OK: 이동된 객체에 재대입은 항상 안전
    std::cout << s;    // "world" -- 완전히 재사용 가능
}
```

### 자기 이동(Self-Move)

`std::move`를 통한 자기 대입 — `x = std::move(x)` — 은 표준 라이브러리 타입에서 기술적으로 미정의 동작입니다. 표준은 자기 이동이 객체를 "유효한" (그러나 미지정) 상태로 남겨야 한다고 요구할 뿐입니다; 실제로는 종종 데이터를 조용히 손상시킵니다.

```cpp
// 방어적 이동 대입: 자기 대입 방지
Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {          // <-- 핵심 검사
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}
// 또는 복사 후 교환 관용구는 설계상 자기 이동을 안전하게 처리합니다.
```

---

### 복사 후 교환(Copy-and-Swap) 관용구

단일 함수로 복사와 이동 대입을 모두 제공하는 대안적 접근 방식:

```cpp
class String {
    char* data_;
    size_t size_;

public:
    // ... 생성자, 소멸자 ...

    friend void swap(String& a, String& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
    }

    // 통합 대입: 복사와 이동 모두 처리
    // lvalue로 호출 시: 매개변수가 복사 생성됨
    // rvalue로 호출 시: 매개변수가 이동 생성됨
    String& operator=(String other) noexcept {
        swap(*this, other);
        return *this;
    }
};
```

---

## 5. std::move

`std::move`는 실제로 아무것도 **이동하지 않습니다**. 단순히 rvalue 참조로의 무조건적 캐스트로, 객체가 이동될 수 있음을 알립니다.

```cpp
#include <iostream>
#include <utility>
#include <string>
#include <vector>

int main() {
    std::string s = "Hello, World!";

    // std::move는 단지 static_cast<std::string&&>(s)
    std::string&& rref = std::move(s);
    // 아직 아무것도 이동되지 않음! s는 여전히 그대로.

    // rvalue 참조가 다른 객체의 초기화나 대입에 사용될 때 이동이 발생:
    std::string s2 = std::move(s);  // 이제 이동 발생
    std::cout << "s2: " << s2 << "\n";   // "Hello, World!"
    std::cout << "s: \"" << s << "\"\n";  // "" (이동됨)

    // 실용적 사용: 컨테이너에 이동
    std::vector<std::string> vec;
    std::string name = "Alice";
    vec.push_back(std::move(name));  // 복사 대신 이동
    // name은 이제 비어있음

    // std::move를 사용하지 말아야 할 때:
    // 1. const 객체에 (이동이 아닌 복사가 됨)
    const std::string cs = "constant";
    std::string s3 = std::move(cs);  // 복사됨! const가 이동을 방지

    // 2. 반환값에 (RVO를 방해)
    // return std::move(local);  // 나쁨: 복사 생략을 방해

    return 0;
}
```

---

## 6. 완벽한 전달(Perfect Forwarding)

템플릿 코드에서 **전달 참조**(forwarding reference, `T&&`에서 `T`가 추론되는 템플릿 매개변수)는 lvalue와 rvalue 모두에 바인딩될 수 있습니다. `std::forward`는 원래의 값 카테고리를 보존합니다.

```cpp
#include <iostream>
#include <utility>
#include <string>

void process(const std::string& s) {
    std::cout << "lvalue: " << s << "\n";
}

void process(std::string&& s) {
    std::cout << "rvalue: " << s << "\n";
}

// 완벽한 전달 없이: 항상 lvalue 오버로드 호출
template<typename T>
void wrapperBad(T&& arg) {
    process(arg);  // arg는 항상 lvalue (이름이 있으므로)
}

// 완벽한 전달으로: 값 카테고리 보존
template<typename T>
void wrapperGood(T&& arg) {
    process(std::forward<T>(arg));
}

// 참조 축소(reference collapsing) 작동 방식:
// T = std::string&   -> T&& = std::string& && = std::string&  (lvalue)
// T = std::string    -> T&& = std::string&&                    (rvalue)

// 완벽한 전달을 사용하는 팩토리 함수
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

int main() {
    std::string s = "Hello";

    wrapperBad(s);              // lvalue (올바름)
    wrapperBad(std::move(s));   // lvalue (잘못됨! rvalue 속성 상실)

    std::string s2 = "World";
    wrapperGood(s2);            // lvalue (올바름)
    wrapperGood(std::move(s2)); // rvalue (올바름!)

    return 0;
}
```

### 참조 축소 규칙

| 템플릿 매개변수 `T` | `T&&`가 되는 형태 | 전달되는 형태 |
|----------------------|-------------------|-------------|
| `X&` | `X& &&` = `X&` | lvalue |
| `X&&` | `X&& &&` = `X&&` | rvalue |
| `X` | `X&&` | rvalue |

---

## 7. 5의 법칙과 0의 법칙

### 5의 법칙(Rule of Five)

클래스가 자원을 관리하고 다음 5개의 특수 멤버 함수 중 **하나라도** 정의하면, **모두** 정의해야 합니다:

```cpp
class ResourceOwner {
    int* data_;
    size_t size_;

public:
    // 1. 소멸자
    ~ResourceOwner() {
        delete[] data_;
    }

    // 2. 복사 생성자
    ResourceOwner(const ResourceOwner& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // 3. 복사 대입 연산자
    ResourceOwner& operator=(const ResourceOwner& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    // 4. 이동 생성자
    ResourceOwner(ResourceOwner&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 5. 이동 대입 연산자
    ResourceOwner& operator=(ResourceOwner&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 생성자
    ResourceOwner(size_t n) : data_(new int[n]()), size_(n) {}
};
```

### 0의 법칙(Rule of Zero)

클래스가 직접 자원을 관리하지 **않는** 경우(스마트 포인터나 표준 컨테이너를 사용하는 경우), 5개의 특수 멤버를 **하나도** 정의하지 말아야 합니다. 컴파일러가 생성하는 기본값이 올바르게 동작합니다.

```cpp
#include <memory>
#include <vector>
#include <string>

// 0의 법칙: 특수 멤버 불필요
class Employee {
    std::string name_;
    int id_;
    std::vector<std::string> projects_;
    std::unique_ptr<int[]> scores_;

public:
    Employee(std::string name, int id)
        : name_(std::move(name)), id_(id),
          scores_(std::make_unique<int[]>(10)) {}

    // 컴파일러가 생성하는 것:
    // - 소멸자 (unique_ptr가 정리 처리)
    // - 이동 생성자와 이동 대입 (unique_ptr는 이동 전용)
    // - 복사는 암묵적으로 삭제됨 (unique_ptr는 복사 불가)
};
```

### 어떤 것을 언제 사용할까

| 시나리오 | 법칙 |
|----------|------|
| 클래스가 원시 자원을 소유 (원시 포인터, 파일 핸들) | 5의 법칙 |
| 스마트 포인터와 STL 컨테이너만 사용 | 0의 법칙 |
| 커스텀 복사가 필요하지만 기본 이동은 괜찮은 경우 | 이동 연산에 `= default` |

---

## 8. 복사 생략(Copy Elision)

복사 생략은 불필요한 복사/이동 연산을 제거하는 컴파일러 최적화입니다.

### 반환 값 최적화(RVO)

```cpp
#include <iostream>

class Heavy {
public:
    Heavy() { std::cout << "Constructed\n"; }
    Heavy(const Heavy&) { std::cout << "Copied\n"; }
    Heavy(Heavy&&) noexcept { std::cout << "Moved\n"; }
};

// 이름 있는 반환 값 최적화 (NRVO)
Heavy createNamed() {
    Heavy h;       // 호출자의 메모리에 직접 생성
    return h;      // NRVO: 복사나 이동 없음
}

// 반환 값 최적화 (RVO) - C++17부터 보장
Heavy createUnnamed() {
    return Heavy();  // 보장: C++17에서 복사나 이동 없음
}

int main() {
    std::cout << "--- NRVO ---\n";
    Heavy h1 = createNamed();    // 일반적으로: "Constructed"만 출력

    std::cout << "--- RVO (C++17 보장) ---\n";
    Heavy h2 = createUnnamed();  // 항상: "Constructed"만 출력

    return 0;
}
```

### 보장된 복사 생략 (C++17)

C++17은 prvalue 초기화에 대해 복사 생략을 의무화합니다. 이는 다음을 의미합니다:

```cpp
// C++17에서 복사/이동 생성자를 호출하지 않음이 보장:
Heavy h = Heavy();                 // 직접 초기화
Heavy h = Heavy(Heavy(Heavy()));   // 중첩된 임시 객체도
auto h = createUnnamed();          // prvalue 반환으로부터

// NRVO는 보장되지 않음 (하지만 일반적으로 적용):
auto h = createNamed();            // 생략될 수도 아닐 수도 있음
```

### 복사 생략이 적용되지 않는 경우

```cpp
Heavy selectOne(bool flag) {
    Heavy a, b;
    if (flag) return a;  // NRVO 적용 불가 (여러 반환 경로)
    return b;            // 컴파일러가 대신 이동 사용 가능
}

Heavy passThrough(Heavy h) {
    return h;  // 매개변수: 생략이 아닌 이동
}
```

| 최적화 | C++17에서 보장? | 조건 |
|--------|:---:|---------|
| RVO (무명 반환) | 예 | prvalue 반환 시 |
| NRVO (유명 반환) | 아니오 | 단일 지역 변수 반환 시 |
| 매개변수 전달 | 아니오 | 함수 매개변수 |

---

## 연습문제

### 연습문제 1: 값 카테고리 퀴즈

각 표현식의 값 카테고리(lvalue, xvalue, prvalue)를 식별하세요:
- `x` (지역 변수)
- `std::move(x)`
- `42`
- `x + y`
- `*ptr`

### 연습문제 2: 이동 지원 String 클래스 구현

동적으로 할당된 문자 배열을 관리하는 `MyString` 클래스를 작성하세요. 5의 법칙에 따라 5개의 특수 멤버 함수를 모두 구현하세요. 적절한 경우 복사 대신 이동이 사용되는지 확인하세요.

### 연습문제 3: 완벽한 전달 팩토리

완벽한 전달을 사용하여 임의의 인수로 어떤 타입 `T`든 생성하는 `create<T>(args...)` 함수 템플릿을 작성하세요. 서로 다른 생성자 시그니처를 가진 타입으로 테스트하세요.

### 연습문제 4: 복사 vs 이동 벤치마크

100만 개의 요소를 가진 `std::vector<int>`를 포함하는 `LargeObject` 클래스를 만드세요. 1000개의 이러한 객체를 복사하는 것과 이동하는 것의 시간 차이를 측정하세요.

### 연습문제 5: 생략 감지

RVO, NRVO, 이동 생성이 발생하는 시점을 보여주는 프로그램을 작성하세요. 모든 특수 멤버 함수에서 메시지를 출력하는 클래스를 사용하세요. `-fno-elide-constructors`로 차이를 확인하세요.

---

## 다음 단계

템플릿은 C++의 제네릭 프로그래밍의 기초입니다. [02_Templates.md](./02_Templates.md)에서 함수 템플릿, 클래스 템플릿, 특수화를 탐구해 봅시다.

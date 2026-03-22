# 스마트 포인터와 RAII

**이전**: [템플릿 메타프로그래밍](./03_Template_Metaprogramming.md) | **다음**: [에러 처리 패턴](./05_Error_Handling_Patterns.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 메모리 관리의 일반적인 함정 식별하기 (누수, 이중 해제, 댕글링 포인터)
2. RAII 원칙을 적용하여 자원의 수명을 객체 범위에 연결하기
3. 독점 소유권을 위한 `unique_ptr` 사용 및 `std::move`로 소유권 전달하기
4. 공유 소유권을 위한 `shared_ptr` 사용 및 참조 카운팅 작동 방식 설명하기
5. `weak_ptr`로 순환 참조 끊기 및 `lock()`으로 안전하게 승격하기
6. 주어진 소유권 시나리오에 적합한 스마트 포인터 타입 선택하기
7. 모던 C++ 모범 사례에 따라 함수와 스마트 포인터 주고받기

---

수동 `new`/`delete`는 전통적인 C++ 코드에서 가장 큰 버그 원인입니다: 메모리 누수, 이중 해제, 댕글링 포인터는 수많은 프로덕션 장애와 보안 취약점을 야기했습니다. 스마트 포인터는 소유권 의미론을 타입 시스템에 직접 인코딩하여 이러한 전체 버그 클래스를 제거합니다. `unique_ptr`, `shared_ptr`, `weak_ptr`를 언제 사용해야 하는지 내재화하면, 원시 포인터가 허용하는 것보다 더 안전하고 더 추론하기 쉬운 코드를 작성할 수 있습니다.

## 1. 메모리 관리의 과제

C++에서의 수동 메모리 관리는 여러 문제를 야기할 수 있습니다.

```cpp
#include <iostream>

// 메모리 누수 예시
void memoryLeak() {
    int* p = new int(42);
    // delete를 잊음 - 메모리 누수!
}

// 이중 해제 예시
void doubleFree() {
    int* p = new int(42);
    delete p;
    // delete p;  // 이중 해제 - 미정의 동작!
}

// 댕글링 포인터 예시
int* danglingPointer() {
    int* p = new int(42);
    delete p;
    return p;  // 해제된 메모리를 가리킴 - 위험!
}

// 예외 시 메모리 누수
void exceptionLeak() {
    int* p = new int(42);
    // throw std::runtime_error("Error!");  // delete가 실행되지 않음
    delete p;
}
```

### 문제 요약

| 문제 | 설명 |
|------|------|
| 메모리 누수 | delete 호출을 잊음 |
| 이중 해제 | 같은 메모리를 두 번 해제 |
| 댕글링 포인터 | 해제된 메모리에 접근 |
| 예외 안전성 | 예외 발생 시 메모리 누수 |

---

## 2. RAII (Resource Acquisition Is Initialization)

자원 획득은 초기화이다: 객체 생성 시 자원을 획득하고, 소멸 시 자동으로 해제합니다.

```cpp
#include <iostream>

// RAII 원칙을 적용한 클래스
class IntPtr {
private:
    int* ptr;

public:
    // 생성자에서 자원 획득
    explicit IntPtr(int value) : ptr(new int(value)) {
        std::cout << "Memory allocated" << std::endl;
    }

    // 소멸자에서 자원 해제
    ~IntPtr() {
        delete ptr;
        std::cout << "Memory freed" << std::endl;
    }

    int& operator*() { return *ptr; }
    int* get() { return ptr; }

    // 복사 비활성화 (간소화)
    IntPtr(const IntPtr&) = delete;
    IntPtr& operator=(const IntPtr&) = delete;
};

void useRAII() {
    IntPtr p(42);
    std::cout << "Value: " << *p << std::endl;
    // 함수 종료 시 자동으로 메모리 해제
}

int main() {
    std::cout << "=== RAII Start ===" << std::endl;
    useRAII();
    std::cout << "=== RAII End ===" << std::endl;
    return 0;
}
```

출력:
```
=== RAII Start ===
Memory allocated
Value: 42
Memory freed
=== RAII End ===
```

---

## 3. unique_ptr

독점 소유권을 가진 스마트 포인터입니다. 하나의 `unique_ptr`만이 객체를 소유할 수 있습니다.

### 기본 사용법

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
    void use() { std::cout << "Resource used" << std::endl; }
};

int main() {
    // unique_ptr 생성
    std::unique_ptr<Resource> p1(new Resource());
    p1->use();

    // make_unique 사용 (C++14, 권장)
    auto p2 = std::make_unique<Resource>();
    p2->use();

    // 기본 타입
    auto num = std::make_unique<int>(42);
    std::cout << "Value: " << *num << std::endl;

    // 배열
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    std::cout << "Array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;  // 모든 메모리 자동 해제
}
```

### 소유권 전달 (move)

```cpp
#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> p) {
    std::cout << "Inside function: " << *p << std::endl;
}  // p가 여기서 파괴됨

std::unique_ptr<int> createResource() {
    return std::make_unique<int>(100);
}

int main() {
    auto p1 = std::make_unique<int>(42);

    // 복사 불가
    // auto p2 = p1;  // 컴파일 에러!

    // 이동은 허용
    auto p2 = std::move(p1);
    std::cout << "p2: " << *p2 << std::endl;

    // p1은 이제 nullptr
    if (p1 == nullptr) {
        std::cout << "p1 is empty" << std::endl;
    }

    // 함수에 전달 (소유권 전달)
    auto p3 = std::make_unique<int>(200);
    takeOwnership(std::move(p3));
    // p3는 이제 nullptr

    // 함수에서 반환 (소유권 전달)
    auto p4 = createResource();
    std::cout << "p4: " << *p4 << std::endl;

    return 0;
}
```

### unique_ptr 메서드

```cpp
#include <iostream>
#include <memory>

int main() {
    auto p = std::make_unique<int>(42);

    // get(): 원시 포인터 얻기 (소유권 유지)
    int* raw = p.get();
    std::cout << "raw: " << *raw << std::endl;

    // release(): 소유권 포기하고 원시 포인터 반환
    int* released = p.release();
    if (p == nullptr) {
        std::cout << "p is empty" << std::endl;
    }
    delete released;  // 수동 삭제 필요

    // reset(): 기존 객체 해제하고 새 객체 설정
    auto p2 = std::make_unique<int>(100);
    std::cout << "Before reset: " << *p2 << std::endl;
    p2.reset(new int(200));
    std::cout << "After reset: " << *p2 << std::endl;
    p2.reset();  // nullptr로 설정

    // swap(): 두 포인터 교환
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    a.swap(b);
    std::cout << "After swap: a=" << *a << ", b=" << *b << std::endl;

    return 0;
}
```

### 커스텀 삭제자

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

// 함수 삭제자
void customDeleter(int* p) {
    std::cout << "Custom deleter called" << std::endl;
    delete p;
}

// 람다 삭제자를 사용한 FILE* 관리
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "Closing file" << std::endl;
        fclose(f);
    }
};

// C API 래퍼 패턴
auto make_file(const char* path, const char* mode) {
    return std::unique_ptr<FILE, decltype(fileDeleter)>(
        fopen(path, mode), fileDeleter
    );
}

int main() {
    // 람다 삭제자
    auto deleter = [](int* p) {
        std::cout << "Lambda deleter" << std::endl;
        delete p;
    };
    std::unique_ptr<int, decltype(deleter)> p(new int(100), deleter);

    // shared_ptr는 커스텀 삭제자에 더 간단한 문법
    auto sp = std::shared_ptr<FILE>(
        fopen("/dev/null", "w"),
        [](FILE* f) { if (f) fclose(f); }
    );

    return 0;
}
```

---

## 4. shared_ptr

공유 소유권을 가진 스마트 포인터입니다. 여러 `shared_ptr`이 같은 객체를 공유할 수 있습니다.

> **비유 -- 공유 도서관 책**: `shared_ptr`는 도서관 대출 시스템과 같습니다. 여러 독자(소유자)가 같은 책을 대출할 수 있습니다. 숨겨진 카운터가 아직 책을 가진 독자 수를 추적합니다. 마지막 독자가 책을 반납해야(카운터가 0이 되어야) 도서관이 메모리를 해제합니다.

### 기본 사용법

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
};

int main() {
    std::shared_ptr<Resource> p1 = std::make_shared<Resource>();
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    {
        std::shared_ptr<Resource> p2 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 2

        std::shared_ptr<Resource> p3 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 3
    }
    // p2, p3 파괴됨
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    return 0;  // 참조 카운트가 0이 되면 Resource 파괴
}
```

### make_shared의 장점

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int data[100];
};

int main() {
    // 방법 1: new 사용 (메모리 할당 2회)
    std::shared_ptr<Widget> p1(new Widget());

    // 방법 2: make_shared 사용 (메모리 할당 1회, 권장)
    auto p2 = std::make_shared<Widget>();

    /*
    make_shared의 장점:
    1. 단일 메모리 할당 (객체 + 제어 블록)
    2. 예외 안전성
    3. 더 깔끔한 코드
    */

    return 0;
}
```

### make_shared 제어 블록(control block)과 weak_ptr 수명

`make_shared`는 관리 대상 객체와 제어 블록(참조 카운트 + 약한 카운트)을 모두 담는 단일 할당을 수행합니다. 이는 효율적이지만 미묘한 수명 관련 함의가 있습니다: 객체 자체가 소멸된 이후에도 마지막 `weak_ptr`이 소멸될 때까지 **전체 할당** 영역이 살아 있습니다.

```cpp
auto sp = std::make_shared<Widget>();  // 할당 1회: [Widget | 제어 블록]
std::weak_ptr<Widget> wp = sp;

sp.reset();  // Widget 소멸 (강한 참조 카운트 = 0)
             // 그러나: 메모리가 아직 해제되지 않음 — weak_ptr가
             // 제어 블록의 약한 카운트를 읽기 위해 할당 영역을 살려둠
wp.reset();  // 이제 메모리 해제 (약한 카운트 = 0)
```

`Widget`이 크고 객체보다 오래 사는 `weak_ptr`를 많이 보유한다면, 객체 메모리가 제어 블록과 독립적으로 해제되도록 `std::shared_ptr<Widget>(new Widget())`를 사용하는 것을 고려하세요.

### shared_ptr과 컨테이너

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Person {
public:
    std::string name;
    Person(const std::string& n) : name(n) {
        std::cout << name << " created" << std::endl;
    }
    ~Person() {
        std::cout << name << " destroyed" << std::endl;
    }
};

int main() {
    std::vector<std::shared_ptr<Person>> people;

    auto alice = std::make_shared<Person>("Alice");
    auto bob = std::make_shared<Person>("Bob");

    people.push_back(alice);
    people.push_back(bob);
    people.push_back(alice);  // Alice 공유

    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 3

    people.clear();
    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 1

    return 0;
}
```

---

## 5. weak_ptr

`shared_ptr`의 순환 참조 문제를 해결합니다. 참조 카운트를 증가시키지 않습니다.

### 순환 참조 문제

```cpp
#include <iostream>
#include <memory>

class B;  // 전방 선언

class A {
public:
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::shared_ptr<A> a_ptr;  // 순환 참조!
    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // 순환 참조 발생

        std::cout << "a ref count: " << a.use_count() << std::endl;  // 2
        std::cout << "b ref count: " << b.use_count() << std::endl;  // 2
    }
    // 메모리 누수! A도 B도 파괴되지 않음
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### weak_ptr를 사용한 해결

```cpp
#include <iostream>
#include <memory>

class B;

class A {
public:
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::weak_ptr<A> a_ptr;  // weak_ptr 사용!
    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // weak_ptr는 참조 카운트를 증가시키지 않음

        std::cout << "a ref count: " << a.use_count() << std::endl;  // 1
        std::cout << "b ref count: " << b.use_count() << std::endl;  // 2
    }
    // 올바르게 파괴됨!
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### weak_ptr 사용법

```cpp
#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> weak;

    {
        auto shared = std::make_shared<int>(42);
        weak = shared;

        std::cout << "Inside block:" << std::endl;
        std::cout << "  expired: " << weak.expired() << std::endl;  // false
        std::cout << "  use_count: " << weak.use_count() << std::endl;  // 1

        // weak_ptr 접근: lock()으로 shared_ptr 얻기
        if (auto sp = weak.lock()) {
            std::cout << "  Value: " << *sp << std::endl;
        }
    }
    // shared가 파괴됨

    std::cout << "Outside block:" << std::endl;
    std::cout << "  expired: " << weak.expired() << std::endl;  // true

    if (auto sp = weak.lock()) {
        std::cout << "  Value: " << *sp << std::endl;
    } else {
        std::cout << "  Object is destroyed" << std::endl;
    }

    return 0;
}
```

### 캐시 구현 예시

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Image {
public:
    std::string filename;

    Image(const std::string& fn) : filename(fn) {
        std::cout << "Loading image: " << filename << std::endl;
    }
    ~Image() {
        std::cout << "Releasing image: " << filename << std::endl;
    }
};

class ImageCache {
private:
    std::map<std::string, std::weak_ptr<Image>> cache;

public:
    std::shared_ptr<Image> getImage(const std::string& filename) {
        auto it = cache.find(filename);

        if (it != cache.end()) {
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << std::endl;
                return sp;
            }
        }

        std::cout << "Cache miss: " << filename << std::endl;
        auto image = std::make_shared<Image>(filename);
        cache[filename] = image;
        return image;
    }
};

int main() {
    ImageCache cache;

    {
        auto img1 = cache.getImage("photo.jpg");
        auto img2 = cache.getImage("photo.jpg");  // 캐시 히트
        auto img3 = cache.getImage("icon.png");
    }
    // 모든 이미지 해제됨

    auto img = cache.getImage("photo.jpg");  // 다시 로드

    return 0;
}
```

---

## 6. enable_shared_from_this

클래스 내부에서 자신의 `shared_ptr`을 안전하게 얻습니다.

> **함정 — 생성자에서 절대 `shared_from_this()`를 호출하지 마세요.** 생성자가 실행될 때는 아직 어떤 `shared_ptr`도 객체를 소유하지 않습니다. 이 시점에 `shared_from_this()`를 호출하면 `std::bad_weak_ptr`가 던져집니다 (또는 이전 구현에서는 미정의 동작이 발생합니다). 항상 객체가 `shared_ptr`에 의해 관리된 이후 일반 멤버 함수에서 호출하세요.
>
> ```cpp
> class Bad : public std::enable_shared_from_this<Bad> {
> public:
>     Bad() {
>         auto self = shared_from_this();  // 예외 발생: 아직 shared_ptr이 없음
>     }
> };
>
> // 올바른 패턴: 팩토리 함수(factory function) 사용
> class Good : public std::enable_shared_from_this<Good> {
>     Good() = default;  // private 생성자
> public:
>     static std::shared_ptr<Good> create() {
>         return std::shared_ptr<Good>(new Good());  // 먼저 shared_ptr 생성
>     }
>     std::shared_ptr<Good> getPtr() {
>         return shared_from_this();  // 안전: 생성 이후 호출됨
>     }
> };
> ```

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Task : public std::enable_shared_from_this<Task> {
public:
    std::string name;

    Task(const std::string& n) : name(n) {}

    // 자신에 대한 shared_ptr를 안전하게 반환
    std::shared_ptr<Task> getPtr() {
        return shared_from_this();
    }

    void addToQueue(std::vector<std::shared_ptr<Task>>& queue) {
        queue.push_back(shared_from_this());
    }
};

int main() {
    std::vector<std::shared_ptr<Task>> taskQueue;

    {
        auto task = std::make_shared<Task>("Task1");
        std::cout << "Ref count: " << task.use_count() << std::endl;  // 1

        task->addToQueue(taskQueue);
        std::cout << "Ref count: " << task.use_count() << std::endl;  // 2
    }
    // task 변수는 파괴되었지만 taskQueue에 남아있음

    for (const auto& t : taskQueue) {
        std::cout << t->name << std::endl;
    }

    return 0;
}
```

---

## 7. 스마트 포인터 선택 가이드

| 상황 | 선택 |
|------|------|
| 단일 소유자 | `unique_ptr` |
| 다중 소유자 | `shared_ptr` |
| 순환 참조 방지 | `weak_ptr` |
| 캐시, 옵저버 | `weak_ptr` |
| 팩토리 함수 반환 | `unique_ptr` |
| 컨테이너 저장 | `shared_ptr` 또는 `unique_ptr` |

---

## 8. 스마트 포인터와 함수

### 함수 매개변수

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int value;
    Widget(int v) : value(v) {}
};

// 소유권 전달
void takeOwnership(std::unique_ptr<Widget> w) {
    std::cout << "Ownership received: " << w->value << std::endl;
}

// 소유권 공유
void shareOwnership(std::shared_ptr<Widget> w) {
    std::cout << "Shared: " << w->value
              << " (count: " << w.use_count() << ")" << std::endl;
}

// 소유권 없이 사용 (비소유 접근에 선호)
void useOnly(Widget& w) {
    std::cout << "Use only: " << w.value << std::endl;
}

// 소유권 없이 사용 (nullable)
void useOnlyPtr(Widget* w) {
    if (w) {
        std::cout << "Pointer use: " << w->value << std::endl;
    }
}

int main() {
    auto up = std::make_unique<Widget>(1);
    useOnly(*up);
    useOnlyPtr(up.get());
    takeOwnership(std::move(up));

    auto sp = std::make_shared<Widget>(2);
    useOnly(*sp);
    shareOwnership(sp);

    return 0;
}
```

### 함수 반환

```cpp
#include <iostream>
#include <memory>

class Product {
public:
    std::string name;
    Product(const std::string& n) : name(n) {}
};

// 팩토리 함수: unique_ptr 반환
std::unique_ptr<Product> createProduct(const std::string& name) {
    return std::make_unique<Product>(name);
}

// 캐시된 객체: shared_ptr 반환
std::shared_ptr<Product> getCachedProduct() {
    static auto cached = std::make_shared<Product>("Cached");
    return cached;
}

int main() {
    auto p1 = createProduct("Widget");
    std::cout << p1->name << std::endl;

    auto p2 = getCachedProduct();
    auto p3 = getCachedProduct();
    std::cout << "Cache count: " << p2.use_count() << std::endl;  // 3

    return 0;
}
```

---

## 9. unique_ptr를 사용한 Pimpl 관용구

Pimpl(Pointer to Implementation) 관용구는 구현 세부사항을 포인터 뒤에 숨겨 컴파일 시간 의존성을 줄입니다.

**widget.h** (헤더):
```cpp
#ifndef WIDGET_H
#define WIDGET_H

#include <memory>
#include <string>

class Widget {
public:
    Widget(const std::string& name, int value);
    ~Widget();  // 여기서 선언, .cpp에서 정의

    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;

    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;

    void doWork();
    std::string getName() const;

private:
    struct Impl;                      // 전방 선언만
    std::unique_ptr<Impl> pImpl_;     // "컴파일 방화벽"
};

#endif
```

**widget.cpp** (구현):
```cpp
#include "widget.h"
#include <iostream>
#include <vector>

struct Widget::Impl {
    std::string name;
    int value;
    std::vector<int> history;

    Impl(const std::string& n, int v) : name(n), value(v) {}
};

Widget::Widget(const std::string& name, int value)
    : pImpl_(std::make_unique<Impl>(name, value)) {}

Widget::~Widget() = default;

Widget::Widget(Widget&& other) noexcept = default;
Widget& Widget::operator=(Widget&& other) noexcept = default;

void Widget::doWork() {
    pImpl_->history.push_back(pImpl_->value);
    std::cout << "Widget '" << pImpl_->name << "' doing work\n";
}

std::string Widget::getName() const {
    return pImpl_->name;
}
```

---

## 10. 성능 고려사항

### unique_ptr vs shared_ptr

| | `unique_ptr` | `shared_ptr` |
|---|---|---|
| **크기** | 원시 포인터와 동일 | 포인터 2개 (객체 + 제어 블록) |
| **오버헤드** | 제로 | 참조 카운팅 (원자적 연산) |
| **할당** | 1회 (객체만) | `make_shared`로 1회, `new`로 2회 |
| **스레드 안전성** | 없음 (단일 소유자) | 참조 카운트는 원자적 |

> **메모리 구조**
>
> unique_ptr: `ptr --> Object` (포인터 하나만)
>
> shared_ptr: `ptr --> Object`, `control --> Control Block` (참조 카운트, 약한 카운트, 삭제자)

### 핵심 원칙

1. **직접적인 new/delete 피하기** - `make_unique`, `make_shared` 사용
2. **unique_ptr를 기본으로** - 필요할 때만 shared_ptr 사용
3. **순환 참조 주의** - weak_ptr로 해결
4. **RAII 원칙 따르기** - 자원 관리 자동화
5. **비소유 접근에는 참조로 전달** - 불필요하게 스마트 포인터를 전달하지 않기

---

## 연습문제

### 연습문제 1: 리소스 매니저

`unique_ptr`와 커스텀 삭제자를 사용하여 다양한 자원(파일 핸들, 네트워크 연결)을 관리하는 클래스를 구현하세요.

### 연습문제 2: 그래프 자료구조

`shared_ptr`과 `weak_ptr`를 사용하여 순환 참조를 피하면서 노드가 서로 연결된 그래프를 구현하세요.

### 연습문제 3: 객체 풀

스마트 포인터를 사용하여 재사용 가능한 객체 풀을 구현하세요. 객체가 더 이상 사용되지 않을 때 풀로 반환되어야 합니다 (힌트: `shared_ptr`의 커스텀 삭제자).

### 연습문제 4: Pimpl 리팩토링

헤더 파일에 무거운 헤더가 여러 개 있는 클래스를 가져와 `unique_ptr`를 사용한 Pimpl 관용구로 리팩토링하세요.

### 연습문제 5: 옵저버 패턴

옵저버가 주제에 대해 `weak_ptr`를 보유하고, 주제가 옵저버 콜백에 `shared_ptr`를 보유하는 옵저버 패턴을 구현하세요. 만료된 옵저버가 자동으로 정리되는 것을 보여주세요.

---

## 다음 단계

에러 처리는 견고한 C++ 프로그램의 중요한 측면입니다. [05_Error_Handling_Patterns.md](./05_Error_Handling_Patterns.md)에서 예외 안전성 보장, `noexcept`, 모던 에러 처리 패턴을 탐구해 봅시다.

# 스마트 포인터와 메모리 관리

## 1. 메모리 관리의 어려움

C++에서 수동 메모리 관리는 여러 문제를 일으킬 수 있습니다.

```cpp
#include <iostream>

// 메모리 누수 예제
void memoryLeak() {
    int* p = new int(42);
    // delete를 잊음 - 메모리 누수!
}

// 이중 해제 예제
void doubleFree() {
    int* p = new int(42);
    delete p;
    // delete p;  // 이중 해제 - 정의되지 않은 동작!
}

// 댕글링 포인터 예제
int* danglingPointer() {
    int* p = new int(42);
    delete p;
    return p;  // 해제된 메모리를 가리킴 - 위험!
}

// 예외 시 메모리 누수
void exceptionLeak() {
    int* p = new int(42);
    // throw std::runtime_error("Error!");  // delete 실행 안 됨
    delete p;
}
```

### 문제점 정리

| 문제 | 설명 |
|------|------|
| 메모리 누수 | delete를 호출하지 않음 |
| 이중 해제 | 같은 메모리를 두 번 해제 |
| 댕글링 포인터 | 해제된 메모리 접근 |
| 예외 안전성 | 예외 발생 시 메모리 누수 |

---

## 2. RAII (Resource Acquisition Is Initialization)

자원 획득은 초기화다: 객체 생성 시 자원 획득, 소멸 시 자동 해제.

```cpp
#include <iostream>

// RAII 원칙 적용 클래스
class IntPtr {
private:
    int* ptr;

public:
    // 생성자에서 자원 획득
    explicit IntPtr(int value) : ptr(new int(value)) {
        std::cout << "메모리 할당" << std::endl;
    }

    // 소멸자에서 자원 해제
    ~IntPtr() {
        delete ptr;
        std::cout << "메모리 해제" << std::endl;
    }

    int& operator*() { return *ptr; }
    int* get() { return ptr; }

    // 복사 금지 (단순화)
    IntPtr(const IntPtr&) = delete;
    IntPtr& operator=(const IntPtr&) = delete;
};

void useRAII() {
    IntPtr p(42);
    std::cout << "값: " << *p << std::endl;
    // 함수 종료 시 자동으로 메모리 해제
}

int main() {
    std::cout << "=== RAII 시작 ===" << std::endl;
    useRAII();
    std::cout << "=== RAII 끝 ===" << std::endl;
    return 0;
}
```

출력:
```
=== RAII 시작 ===
메모리 할당
값: 42
메모리 해제
=== RAII 끝 ===
```

---

## 3. unique_ptr

단독 소유권을 가지는 스마트 포인터입니다. 하나의 `unique_ptr`만 객체를 소유할 수 있습니다.

### 기본 사용법

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource 생성" << std::endl; }
    ~Resource() { std::cout << "Resource 소멸" << std::endl; }
    void use() { std::cout << "Resource 사용" << std::endl; }
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
    std::cout << "값: " << *num << std::endl;

    // 배열
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    std::cout << "배열: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;  // 자동으로 모든 메모리 해제
}
```

### 소유권 이전 (move)

```cpp
#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> p) {
    std::cout << "함수 내부: " << *p << std::endl;
}  // p가 여기서 소멸

std::unique_ptr<int> createResource() {
    return std::make_unique<int>(100);
}

int main() {
    auto p1 = std::make_unique<int>(42);

    // 복사 불가
    // auto p2 = p1;  // 컴파일 에러!

    // 이동은 가능
    auto p2 = std::move(p1);
    std::cout << "p2: " << *p2 << std::endl;

    // p1은 이제 nullptr
    if (p1 == nullptr) {
        std::cout << "p1은 비어있음" << std::endl;
    }

    // 함수에 전달 (소유권 이전)
    auto p3 = std::make_unique<int>(200);
    takeOwnership(std::move(p3));
    // p3는 이제 nullptr

    // 함수에서 반환 (소유권 이전)
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
        std::cout << "p는 비어있음" << std::endl;
    }
    delete released;  // 수동 해제 필요

    // reset(): 기존 객체 해제하고 새 객체 설정
    auto p2 = std::make_unique<int>(100);
    std::cout << "reset 전: " << *p2 << std::endl;
    p2.reset(new int(200));
    std::cout << "reset 후: " << *p2 << std::endl;
    p2.reset();  // nullptr로 설정
    if (!p2) {
        std::cout << "p2는 비어있음" << std::endl;
    }

    // swap(): 두 포인터 교환
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    a.swap(b);
    std::cout << "swap 후: a=" << *a << ", b=" << *b << std::endl;

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
    std::cout << "커스텀 삭제자 호출" << std::endl;
    delete p;
}

// FILE* 용 삭제자
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "파일 닫기" << std::endl;
        fclose(f);
    }
};

int main() {
    // 함수 포인터 삭제자
    std::unique_ptr<int, void(*)(int*)> p1(
        new int(42), customDeleter
    );

    // 람다 삭제자
    auto deleter = [](int* p) {
        std::cout << "람다 삭제자" << std::endl;
        delete p;
    };
    std::unique_ptr<int, decltype(deleter)> p2(
        new int(100), deleter
    );

    // FILE 관리
    std::unique_ptr<FILE, decltype(fileDeleter)> file(
        fopen("test.txt", "w"), fileDeleter
    );
    if (file) {
        fprintf(file.get(), "Hello, World!\n");
    }

    return 0;
}
```

---

## 4. shared_ptr

공유 소유권을 가지는 스마트 포인터입니다. 여러 `shared_ptr`이 같은 객체를 공유할 수 있습니다.

### 기본 사용법

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource 생성" << std::endl; }
    ~Resource() { std::cout << "Resource 소멸" << std::endl; }
};

int main() {
    // shared_ptr 생성
    std::shared_ptr<Resource> p1 = std::make_shared<Resource>();
    std::cout << "참조 카운트: " << p1.use_count() << std::endl;  // 1

    {
        // 공유
        std::shared_ptr<Resource> p2 = p1;
        std::cout << "참조 카운트: " << p1.use_count() << std::endl;  // 2

        std::shared_ptr<Resource> p3 = p1;
        std::cout << "참조 카운트: " << p1.use_count() << std::endl;  // 3
    }
    // p2, p3 소멸
    std::cout << "참조 카운트: " << p1.use_count() << std::endl;  // 1

    return 0;  // 참조 카운트가 0이 되면 Resource 소멸
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
    // 방법 1: new 사용 (메모리 할당 2번)
    std::shared_ptr<Widget> p1(new Widget());

    // 방법 2: make_shared 사용 (메모리 할당 1번, 권장)
    auto p2 = std::make_shared<Widget>();

    /*
    make_shared 장점:
    1. 메모리 할당 1번 (객체 + 제어 블록)
    2. 예외 안전성
    3. 코드 간결
    */

    std::cout << "p1 use_count: " << p1.use_count() << std::endl;
    std::cout << "p2 use_count: " << p2.use_count() << std::endl;

    return 0;
}
```

### shared_ptr과 컨테이너

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Person {
public:
    std::string name;
    Person(const std::string& n) : name(n) {
        std::cout << name << " 생성" << std::endl;
    }
    ~Person() {
        std::cout << name << " 소멸" << std::endl;
    }
};

int main() {
    std::vector<std::shared_ptr<Person>> people;

    auto alice = std::make_shared<Person>("Alice");
    auto bob = std::make_shared<Person>("Bob");

    people.push_back(alice);
    people.push_back(bob);
    people.push_back(alice);  // Alice 공유

    std::cout << "Alice 참조 카운트: " << alice.use_count() << std::endl;  // 3

    std::cout << "\n=== 목록 ===" << std::endl;
    for (const auto& p : people) {
        std::cout << p->name << std::endl;
    }

    people.clear();
    std::cout << "\n=== clear 후 ===" << std::endl;
    std::cout << "Alice 참조 카운트: " << alice.use_count() << std::endl;  // 1

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

    ~A() { std::cout << "A 소멸" << std::endl; }
};

class B {
public:
    std::shared_ptr<A> a_ptr;  // 순환 참조!

    ~B() { std::cout << "B 소멸" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // 순환 참조 발생

        std::cout << "a 참조 카운트: " << a.use_count() << std::endl;  // 2
        std::cout << "b 참조 카운트: " << b.use_count() << std::endl;  // 2
    }
    // 메모리 누수! A, B 모두 소멸되지 않음
    std::cout << "블록 종료" << std::endl;

    return 0;
}
```

### weak_ptr로 해결

```cpp
#include <iostream>
#include <memory>

class B;

class A {
public:
    std::shared_ptr<B> b_ptr;

    ~A() { std::cout << "A 소멸" << std::endl; }
};

class B {
public:
    std::weak_ptr<A> a_ptr;  // weak_ptr 사용!

    ~B() { std::cout << "B 소멸" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // weak_ptr는 참조 카운트 증가 안 함

        std::cout << "a 참조 카운트: " << a.use_count() << std::endl;  // 1
        std::cout << "b 참조 카운트: " << b.use_count() << std::endl;  // 2
    }
    // 정상적으로 소멸!
    std::cout << "블록 종료" << std::endl;

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

        std::cout << "블록 내부:" << std::endl;
        std::cout << "  expired: " << weak.expired() << std::endl;  // false
        std::cout << "  use_count: " << weak.use_count() << std::endl;  // 1

        // weak_ptr 접근: lock()으로 shared_ptr 얻기
        if (auto sp = weak.lock()) {
            std::cout << "  값: " << *sp << std::endl;
        }
    }
    // shared가 소멸됨

    std::cout << "블록 외부:" << std::endl;
    std::cout << "  expired: " << weak.expired() << std::endl;  // true
    std::cout << "  use_count: " << weak.use_count() << std::endl;  // 0

    if (auto sp = weak.lock()) {
        std::cout << "  값: " << *sp << std::endl;
    } else {
        std::cout << "  객체가 소멸됨" << std::endl;
    }

    return 0;
}
```

### 캐시 구현 예제

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Image {
public:
    std::string filename;

    Image(const std::string& fn) : filename(fn) {
        std::cout << "이미지 로딩: " << filename << std::endl;
    }
    ~Image() {
        std::cout << "이미지 해제: " << filename << std::endl;
    }
};

class ImageCache {
private:
    std::map<std::string, std::weak_ptr<Image>> cache;

public:
    std::shared_ptr<Image> getImage(const std::string& filename) {
        auto it = cache.find(filename);

        if (it != cache.end()) {
            // 캐시에 있으면 weak_ptr에서 shared_ptr 얻기 시도
            if (auto sp = it->second.lock()) {
                std::cout << "캐시 히트: " << filename << std::endl;
                return sp;
            }
        }

        // 캐시 미스: 새로 로딩
        std::cout << "캐시 미스: " << filename << std::endl;
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

        std::cout << "img1 use_count: " << img1.use_count() << std::endl;
    }
    // 모든 이미지 해제됨

    std::cout << "\n=== 다시 요청 ===" << std::endl;
    auto img = cache.getImage("photo.jpg");  // 다시 로딩

    return 0;
}
```

---

## 6. enable_shared_from_this

클래스 내부에서 자신의 `shared_ptr`을 안전하게 얻습니다.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Task : public std::enable_shared_from_this<Task> {
public:
    std::string name;

    Task(const std::string& n) : name(n) {
        std::cout << name << " 생성" << std::endl;
    }

    ~Task() {
        std::cout << name << " 소멸" << std::endl;
    }

    // 자신의 shared_ptr를 안전하게 반환
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
        std::cout << "참조 카운트: " << task.use_count() << std::endl;  // 1

        task->addToQueue(taskQueue);
        std::cout << "참조 카운트: " << task.use_count() << std::endl;  // 2
    }
    // task 변수는 소멸, 하지만 taskQueue에 남아있음

    std::cout << "\n=== Queue 내용 ===" << std::endl;
    for (const auto& t : taskQueue) {
        std::cout << t->name << std::endl;
    }

    return 0;
}
```

주의사항:
```cpp
// 잘못된 사용 - 반드시 shared_ptr로 관리되어야 함
// Task t("Direct");
// t.getPtr();  // 런타임 에러!
```

---

## 7. 스마트 포인터 선택 가이드

```
┌─────────────────────────────────────────────────────┐
│              스마트 포인터 선택                        │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    단독 소유?      공유 필요?     약한 참조?
          │             │             │
          ▼             ▼             ▼
    unique_ptr    shared_ptr     weak_ptr
```

| 상황 | 선택 |
|------|------|
| 하나의 소유자 | `unique_ptr` |
| 여러 소유자 | `shared_ptr` |
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

// 소유권 전달 (unique_ptr)
void takeOwnership(std::unique_ptr<Widget> w) {
    std::cout << "소유권 받음: " << w->value << std::endl;
}

// 소유권 공유 (shared_ptr 복사)
void shareOwnership(std::shared_ptr<Widget> w) {
    std::cout << "공유 받음: " << w->value
              << " (count: " << w.use_count() << ")" << std::endl;
}

// 소유권 없이 사용 (참조)
void useOnly(Widget& w) {
    std::cout << "사용만: " << w.value << std::endl;
}

// 소유권 없이 사용 (원시 포인터)
void useOnlyPtr(Widget* w) {
    if (w) {
        std::cout << "포인터 사용: " << w->value << std::endl;
    }
}

int main() {
    // unique_ptr
    auto up = std::make_unique<Widget>(1);
    useOnly(*up);
    useOnlyPtr(up.get());
    takeOwnership(std::move(up));  // 소유권 이전

    // shared_ptr
    auto sp = std::make_shared<Widget>(2);
    useOnly(*sp);
    useOnlyPtr(sp.get());
    shareOwnership(sp);  // 공유
    std::cout << "원본 count: " << sp.use_count() << std::endl;

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
    std::cout << "캐시 count: " << p2.use_count() << std::endl;  // 3

    return 0;
}
```

---

## 9. 일반적인 실수와 해결

### 실수 1: 같은 원시 포인터로 여러 스마트 포인터 생성

```cpp
#include <iostream>
#include <memory>

int main() {
    int* raw = new int(42);

    // 잘못된 코드 - 절대 하지 말 것!
    // std::shared_ptr<int> p1(raw);
    // std::shared_ptr<int> p2(raw);  // 이중 해제 발생!

    // 올바른 코드
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;  // 공유

    return 0;
}
```

### 실수 2: this를 shared_ptr로 변환

```cpp
#include <iostream>
#include <memory>

class Bad {
public:
    // 잘못된 방법
    std::shared_ptr<Bad> getShared() {
        // return std::shared_ptr<Bad>(this);  // 위험!
        return nullptr;
    }
};

class Good : public std::enable_shared_from_this<Good> {
public:
    // 올바른 방법
    std::shared_ptr<Good> getShared() {
        return shared_from_this();
    }
};
```

### 실수 3: 순환 참조

```cpp
// 위의 weak_ptr 섹션 참조
// shared_ptr만 사용하면 순환 참조로 메모리 누수
// weak_ptr로 한쪽 연결을 약한 참조로 변경
```

### 실수 4: unique_ptr 복사 시도

```cpp
#include <memory>

void processWidget(std::unique_ptr<int> p) {}

int main() {
    auto p = std::make_unique<int>(42);

    // 잘못된 코드
    // processWidget(p);  // 컴파일 에러

    // 올바른 코드 (소유권 이전)
    processWidget(std::move(p));

    return 0;
}
```

---

## 10. 성능 고려사항

### unique_ptr vs shared_ptr

```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

int main() {
    const int N = 1000000;

    // unique_ptr (오버헤드 거의 없음)
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_unique<int>(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // shared_ptr (참조 카운트 관리 오버헤드)
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_shared<int>(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    std::cout << "unique_ptr: " << dur1.count() << " us" << std::endl;
    std::cout << "shared_ptr: " << dur2.count() << " us" << std::endl;

    return 0;
}
```

### 메모리 구조

```
unique_ptr:
┌─────────────────┐
│  ptr → 객체     │  (포인터 하나만)
└─────────────────┘

shared_ptr:
┌─────────────────┐     ┌─────────────────┐
│  ptr ─────────────┬──▶│     객체         │
│  control ───┐    │    └─────────────────┘
└─────────────│───┘
              ▼
        ┌─────────────────┐
        │  참조 카운트      │
        │  weak 카운트      │
        │  삭제자          │
        └─────────────────┘
```

---

## 11. 요약

| 스마트 포인터 | 소유권 | 복사 | 참조 카운트 | 용도 |
|--------------|--------|------|------------|------|
| `unique_ptr` | 단독 | X | X | 단일 소유자 |
| `shared_ptr` | 공유 | O | O | 공유 소유권 |
| `weak_ptr` | 없음 | O | X | 순환 참조 방지 |

### 핵심 원칙

1. **new/delete 직접 사용 피하기** - `make_unique`, `make_shared` 사용
2. **기본은 unique_ptr** - 필요할 때만 shared_ptr
3. **순환 참조 주의** - weak_ptr로 해결
4. **RAII 원칙 준수** - 자원 관리 자동화

---

## 12. 연습 문제

### 연습 1: 리소스 매니저

파일, 네트워크 연결 등 다양한 리소스를 관리하는 클래스를 `unique_ptr`로 구현하세요.

### 연습 2: 그래프 자료구조

노드들이 서로 연결된 그래프를 `shared_ptr`와 `weak_ptr`로 구현하세요.

### 연습 3: 객체 풀

재사용 가능한 객체 풀을 스마트 포인터로 구현하세요.

---

## 다음 단계

[15_Modern_CPP.md](./15_Modern_CPP.md)에서 C++11/14/17/20의 주요 기능을 배워봅시다!

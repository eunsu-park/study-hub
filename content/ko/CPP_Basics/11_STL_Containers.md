# STL 컨테이너

**이전**: [상속과 다형성](./10_Inheritance_and_Polymorphism.md) | **다음**: [STL 알고리즘과 반복자](./12_STL_Algorithms_and_Iterators.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. STL이 무엇인지 설명하고 4가지 주요 구성요소를 식별한다: 컨테이너, 반복자, 알고리즘, 함수 객체
2. `std::vector`로 동적 배열을 구현하고 요소 접근, 삽입, 삭제, 반복자 순회를 적용한다
3. 순서 컨테이너(`vector`, `array`, `deque`, `list`)를 비교하고 사용 사례에 맞는 것을 선택한다
4. 연관 컨테이너(`set`, `map`)와 비정렬 컨테이너를 정렬 및 해시 기반 저장에 적용한다
5. `stack`, `queue`, `priority_queue` 어댑터로 LIFO, FIFO, 우선순위 기반 로직을 구현한다
6. 정렬 컨테이너와 비정렬 컨테이너의 내부 구조, 시간 복잡도, 순회 순서를 구분한다
7. `std::pair`와 `std::tuple`로 복합 데이터를 설계하고 C++17 구조적 바인딩으로 분해한다

---

표준 템플릿 라이브러리(STL)는 C++이 진정으로 빛나는 부분입니다. 모든 프로젝트마다 연결 리스트, 해시 맵, 정렬 알고리즘을 다시 만드는 대신, 수십 년에 걸쳐 정제된 검증되고 최적화된 컨테이너와 알고리즘에 의존할 수 있습니다. 어떤 컨테이너를 선택할지 -- 그리고 각 선택 뒤의 Big-O 트레이드오프를 이해하는 것 -- 은 성능이 좋고 관용적인 C++ 코드를 작성하는 가장 영향력 있는 기술 중 하나입니다.

## 1. STL이란?

STL(Standard Template Library)은 C++ 표준 라이브러리의 핵심으로, 데이터 구조와 알고리즘을 제공합니다.

### STL 구성요소

| 구성요소 | 설명 |
|----------|------|
| 컨테이너(Container) | 데이터를 저장하는 자료구조 |
| 반복자(Iterator) | 컨테이너 요소 순회 |
| 알고리즘(Algorithm) | 정렬, 검색 등 범용 함수 |
| 함수 객체(Function Object) | 함수처럼 동작하는 객체 |

---

## 2. vector

동적 크기 배열. 가장 많이 사용됩니다.

### 기본 사용법

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v1;                  // 빈 벡터
    std::vector<int> v2(5);               // 크기 5, 0으로 초기화
    std::vector<int> v3(5, 10);           // 크기 5, 10으로 초기화
    std::vector<int> v4 = {1, 2, 3, 4, 5}; // 초기화 리스트

    v1.push_back(10);
    v1.push_back(20);
    v1.push_back(30);

    for (int num : v1) { std::cout << num << " "; }
    std::cout << std::endl;  // 10 20 30
    return 0;
}
```

### 요소 접근

```cpp
std::vector<int> v = {10, 20, 30, 40, 50};
std::cout << v[0] << std::endl;      // 10
std::cout << v.at(2) << std::endl;   // 30 (범위 검사)
std::cout << v.front() << std::endl;  // 10
std::cout << v.back() << std::endl;   // 50
std::cout << "Size: " << v.size() << std::endl;
```

### 삽입과 삭제

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
v.push_back(6);                        // 끝에 추가
v.pop_back();                          // 끝에서 제거
v.insert(v.begin() + 2, 100);         // 중간에 삽입
v.erase(v.begin() + 2);               // 중간에서 삭제
v.erase(v.begin(), v.begin() + 2);    // 범위 삭제
v.clear();                             // 전체 삭제
```

### 반복자

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};

// auto 사용 (권장)
for (auto it = v.begin(); it != v.end(); ++it) {
    std::cout << *it << " ";
}

// 역방향 반복자
for (auto it = v.rbegin(); it != v.rend(); ++it) {
    std::cout << *it << " ";
}
// 5 4 3 2 1
```

---

## 3. array

고정 크기 배열입니다.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // 접근
    std::cout << arr[0] << std::endl;
    std::cout << arr.at(2) << std::endl;
    std::cout << arr.front() << std::endl;
    std::cout << arr.back() << std::endl;

    // 크기
    std::cout << "Size: " << arr.size() << std::endl;

    // 순회
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 채우기
    arr.fill(0);

    return 0;
}
```

---

## 4. deque

양쪽 끝에서 빠른 삽입/삭제가 가능한 컨테이너입니다.

```cpp
#include <iostream>
#include <deque>

int main() {
    std::deque<int> dq;

    // 앞/뒤에 추가
    dq.push_back(1);
    dq.push_back(2);
    dq.push_front(0);
    dq.push_front(-1);

    // {-1, 0, 1, 2}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 앞/뒤에서 제거
    dq.pop_front();
    dq.pop_back();

    // {0, 1}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 5. list

이중 연결 리스트입니다.

> **비유 — 포스트잇 체인**: `std::list`는 각 포스트잇에 "다음 노트는 X 페이지에 있다"고 적힌 포스트잇 체인과 같습니다. 포인터 하나만 수정하면 어디서든 즉시 삽입하거나 제거할 수 있지만, 50번째 노트를 찾으려면 처음부터 체인을 따라가야 합니다 — 지름길이 없습니다.

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // 앞/뒤에 추가
    lst.push_front(0);
    lst.push_back(9);

    // 정렬 (멤버 메서드)
    lst.sort();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 1 3 4 5 9

    // 중복 제거 (연속된 것만)
    lst.unique();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 3 4 5 9

    // 삽입
    auto it = lst.begin();
    std::advance(it, 2);  // 2칸 이동
    lst.insert(it, 100);  // 그 위치에 삽입

    return 0;
}
```

---

## 6. set

정렬된 고유 요소 컬렉션입니다.

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;
    s.insert(30); s.insert(10); s.insert(20); s.insert(10);  // 중복 무시

    for (int num : s) { std::cout << num << " "; }
    // 10 20 30 (자동 정렬)

    if (s.find(20) != s.end()) { std::cout << "20 found" << std::endl; }
    s.erase(20);
    return 0;
}
```

---

## 7. map

정렬된 키-값 쌍 컨테이너입니다.

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> ages;
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages.insert({"Charlie", 35});

    // 구조적 바인딩 (C++17)으로 순회
    for (const auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    if (ages.find("Alice") != ages.end()) {
        std::cout << "Alice found" << std::endl;
    }

    ages.erase("Bob");
    return 0;
}
```

---

## 8. unordered_set / unordered_map

해시 테이블 기반 컨테이너로 평균 O(1) 접근입니다.

### set vs unordered_set

| 기능 | set | unordered_set |
|------|-----|---------------|
| 내부 구조 | 레드-블랙 트리 | 해시 테이블 |
| 정렬 | 예 | 아니오 |
| 삽입/검색 | O(log n) | O(1) 평균 |
| 순회 순서 | 정렬됨 | 미정의 |

---

## 9. stack과 queue

### stack (LIFO)

```cpp
#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;
    s.push(10); s.push(20); s.push(30);
    while (!s.empty()) {
        std::cout << s.top() << " "; s.pop();
    }
    // 30 20 10
    return 0;
}
```

### queue (FIFO)

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;
    q.push(10); q.push(20); q.push(30);
    while (!q.empty()) {
        std::cout << q.front() << " "; q.pop();
    }
    // 10 20 30
    return 0;
}
```

### priority_queue

```cpp
#include <iostream>
#include <queue>

int main() {
    std::priority_queue<int> pq;  // 최대 힙 (기본)
    pq.push(30); pq.push(10); pq.push(20);
    while (!pq.empty()) { std::cout << pq.top() << " "; pq.pop(); }
    // 30 20 10

    // 최소 힙
    std::priority_queue<int, std::vector<int>, std::greater<int>> minPq;
    minPq.push(30); minPq.push(10); minPq.push(20);
    while (!minPq.empty()) { std::cout << minPq.top() << " "; minPq.pop(); }
    // 10 20 30
    return 0;
}
```

---

## 10. pair와 tuple

```cpp
#include <iostream>
#include <tuple>

int main() {
    // pair
    std::pair<std::string, int> p1("Alice", 25);
    std::cout << p1.first << ": " << p1.second << std::endl;

    // tuple + 구조적 바인딩
    std::tuple<std::string, int, double> t("Alice", 25, 165.5);
    auto [name, age, height] = t;
    std::cout << name << ", " << age << ", " << height << std::endl;
    return 0;
}
```

---

## 11. 컨테이너 선택 가이드

| 요구사항 | 권장 컨테이너 |
|---------|--------------|
| 순차 접근 + 끝 삽입/삭제 | `vector` |
| 양쪽 끝 삽입/삭제 | `deque` |
| 빈번한 중간 삽입/삭제 | `list` |
| 고유 요소 + 정렬 | `set` |
| 고유 요소 + 빠른 검색 | `unordered_set` |
| 키-값 + 정렬 | `map` |
| 키-값 + 빠른 검색 | `unordered_map` |
| LIFO | `stack` |
| FIFO | `queue` |
| 우선순위 | `priority_queue` |

---

## 12. 커스텀 할당자와 해시 커스터마이징

### 커스텀 할당자가 필요한 이유?

모든 STL 컨테이너는 선택적 할당자 템플릿 파라미터를 받습니다. 기본적으로 `std::allocator<T>`는 `new`/`delete`를 사용하지만, 커스텀 할당자를 사용하면 다음이 가능합니다:

- **메모리 풀**: 큰 블록을 미리 할당하고 고정 크기 청크를 나눠줌 (시스템 호출 오버헤드 제거)
- **아레나 할당**: 연속된 영역에 많은 객체를 할당하고 한 번에 해제 (게임 엔진, 컴파일러, 요청 범위 서버에 유용)
- **추적**: 할당 횟수 세기, 누수 감지, 메모리 사용량 로깅
- **정렬**: SIMD 또는 하드웨어 요구사항을 위한 특정 정렬 보장

### 최소한의 커스텀 할당자 (C++17)

C++17은 할당자 요구사항을 대폭 단순화했습니다. `allocate`, `deallocate`, 몇 가지 타입 별칭만 있으면 됩니다:

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>

/* 몇 바이트가 할당되었는지 세는 추적 할당자.
 * 이유: 디버깅, 프로파일링, 또는 메모리 예산 적용에 유용합니다. */
template <typename T>
struct TrackingAllocator {
    using value_type = T;

    /* 이 할당자의 모든 리바운드 복사본에 걸친 공유 카운터.
     * shared_ptr를 사용하는 이유: 컨테이너가 내부 노드용 할당자를 리바운드할 때,
     * 하나의 카운터로 전체 메모리를 추적하기 위해서입니다. */
    std::shared_ptr<std::size_t> total_allocated;

    TrackingAllocator()
        : total_allocated(std::make_shared<std::size_t>(0)) {}

    /* 리바인딩 생성자: 다른 타입에 사용할 수 있게 해줍니다.
     * 필요한 이유: std::vector<T, Alloc>은 내부적으로 Alloc<SomeInternalType>이 필요하므로,
     * 컨테이너가 이 생성자를 통해 할당자를 변환합니다. */
    template <typename U>
    TrackingAllocator(const TrackingAllocator<U>& other)
        : total_allocated(other.total_allocated) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated += bytes;
        std::cout << "[alloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        return static_cast<T*>(std::malloc(bytes));
    }

    void deallocate(T* ptr, std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        *total_allocated -= bytes;
        std::cout << "[dealloc] " << bytes << " bytes (total: "
                  << *total_allocated << ")\n";
        std::free(ptr);
    }

    /* 컨테이너 동등성 검사에 필요합니다. 두 할당자가 "동등"하다는 것은
     * 한 쪽에서 할당한 메모리를 다른 쪽에서 해제할 수 있음을 의미합니다. */
    template <typename U>
    bool operator==(const TrackingAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const TrackingAllocator<U>&) const { return false; }
};

int main() {
    /* 추적 할당자를 std::vector와 함께 사용 */
    std::vector<int, TrackingAllocator<int>> v;

    v.push_back(1);   // 초기 버퍼 할당
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    v.push_back(5);   // 재할당 발생 가능 (용량 두 배 증가)

    std::cout << "Vector contents: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

### 아레나 할당자 개념

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

/* 단순한 아레나(범프) 할당자: 고정 크기 버퍼에서 할당합니다.
 * 이유: 매우 빠른 할당 (포인터 증가만으로 가능), 아레나 소멸 시 메모리 전부 해제.
 * 객체별 해제 없음. */
template <typename T>
struct ArenaAllocator {
    using value_type = T;

    /* 공유 아레나 상태 */
    struct Arena {
        std::uint8_t* buffer;
        std::size_t   capacity;
        std::size_t   offset;

        Arena(std::size_t cap)
            : buffer(new std::uint8_t[cap]), capacity(cap), offset(0) {}
        ~Arena() { delete[] buffer; }
    };

    std::shared_ptr<Arena> arena;

    explicit ArenaAllocator(std::size_t capacity)
        : arena(std::make_shared<Arena>(capacity)) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other)
        : arena(other.arena) {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        /* alignof(T)에 맞게 정렬 */
        std::size_t aligned = (arena->offset + alignof(T) - 1) & ~(alignof(T) - 1);
        if (aligned + bytes > arena->capacity) {
            throw std::bad_alloc();
        }
        T* result = reinterpret_cast<T*>(arena->buffer + aligned);
        arena->offset = aligned + bytes;
        return result;
    }

    /* 아레나 할당자: 해제는 아무것도 하지 않음. 메모리는 한 번에 해제됩니다. */
    void deallocate(T*, std::size_t) { /* 의도적으로 비어 있음 */ }

    template <typename U> bool operator==(const ArenaAllocator<U>&) const { return true; }
    template <typename U> bool operator!=(const ArenaAllocator<U>&) const { return false; }
};

int main() {
    ArenaAllocator<int> alloc(4096);  // 4KB 아레나
    std::vector<int, ArenaAllocator<int>> v(alloc);

    for (int i = 0; i < 100; i++) {
        v.push_back(i);
    }
    std::cout << "Arena used: " << alloc.arena->offset << " bytes\n";

    return 0;  // Arena 소멸자에서 모든 아레나 메모리 한 번에 해제
}
```

### `unordered_map`을 위한 커스텀 해시

기본적으로 `std::unordered_map`과 `std::unordered_set`은 내장 타입과 `std::string`에만 작동하는 `std::hash<Key>`를 사용합니다. 커스텀 타입에는 해시 함수를 직접 제공해야 합니다.

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

struct Point {
    int x, y;

    /* operator==은 비정렬 컨테이너에 필수입니다.
     * 이유: 해싱 후 컨테이너가 충돌을 처리하기 위해 동등성이 필요합니다. */
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

/* 방법 1: std::hash 특수화 (널리 사용되는 타입에 권장) */
template <>
struct std::hash<Point> {
    std::size_t operator()(const Point& p) const {
        /* 해시 결합 패턴: 각 필드의 해시를 혼합합니다.
         * 이 공식을 쓰는 이유? 소수를 곱하고 XOR하면
         * (1,2)와 (2,1)이 같은 해시를 생성하지 않게 됩니다.
         * 시프트와 황금비 상수(0x9e3779b9)가 비트를 고르게 분산합니다. */
        std::size_t h1 = std::hash<int>{}(p.x);
        std::size_t h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 * 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

int main() {
    /* std::hash 특수화 후 Point를 키로 직접 사용 가능 */
    std::unordered_map<Point, std::string> labels;
    labels[{0, 0}] = "origin";
    labels[{1, 2}] = "point A";
    labels[{3, 4}] = "point B";

    for (const auto& [pt, label] : labels) {
        std::cout << "(" << pt.x << ", " << pt.y << "): "
                  << label << "\n";
    }

    return 0;
}
```

### 복합 키를 위한 해시 결합

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <functional>

/* 재사용 가능한 hash_combine 유틸리티.
 * 이유: 여러 필드 해시를 하나로 합치는 것은 반복적인 필요입니다.
 * boost::hash_combine을 모델로 합니다. */
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Employee {
    std::string department;
    std::string name;
    int id;

    bool operator==(const Employee& o) const {
        return department == o.department && name == o.name && id == o.id;
    }
};

/* 방법 2: 템플릿 인수로 전달하는 펑터 (로컬/특수 용도에 적합) */
struct EmployeeHash {
    std::size_t operator()(const Employee& e) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<std::string>{}(e.department));
        hash_combine(seed, std::hash<std::string>{}(e.name));
        hash_combine(seed, std::hash<int>{}(e.id));
        return seed;
    }
};

int main() {
    /* 해시 펑터를 세 번째 템플릿 인수로 전달 */
    std::unordered_map<Employee, double, EmployeeHash> salaries;

    salaries[{"Engineering", "Alice", 1001}] = 95000.0;
    salaries[{"Marketing",   "Bob",   2001}] = 85000.0;

    for (const auto& [emp, salary] : salaries) {
        std::cout << emp.name << " (" << emp.department << "): $"
                  << salary << "\n";
    }

    return 0;
}
```

### 커스터마이징 시기

| 시나리오 | 해결책 |
|----------|--------|
| 커스텀 타입을 `unordered_map` 키로 사용 | `std::hash` 특수화 또는 해시 펑터 전달 |
| 복합 키 (여러 필드) | `hash_combine` 패턴 사용 |
| 결정론적 메모리 할당 필요 | 아레나/풀 커스텀 할당자 |
| 메모리 사용량 추적 또는 예산 | 추적 할당자 |
| 고빈도, 동일 크기 할당 | 풀 할당자 |

---

## 13. 요약

| 컨테이너 | 특성 |
|----------|------|
| `vector` | 동적 배열, 끝에서 O(1) |
| `array` | 고정 배열 |
| `deque` | 양쪽 끝에서 O(1) |
| `list` | 이중 연결 리스트 |
| `set` | 정렬 + 고유 |
| `map` | 키-값 + 정렬 |
| `unordered_set` | 해시 + 고유 |
| `unordered_map` | 해시 + 키-값 |
| `stack` | LIFO |
| `queue` | FIFO |
| `priority_queue` | 힙 |

---

## 연습문제

### 연습문제 1: 컨테이너 선택 근거
각 시나리오에 가장 적합한 STL 컨테이너를 선택하고 시간 복잡도와 사용 사례 적합성으로 근거를 제시하세요.

### 연습문제 2: Vector 증가 관찰
빈 `std::vector<int>`에 20개의 정수를 하나씩 push하고 각 `push_back` 후 `size()`와 `capacity()`를 출력하세요.

### 연습문제 3: 단어 빈도 카운터
`std::map<std::string, int>`을 사용하여 각 단어의 출현 횟수를 세는 프로그램을 작성하세요.

### 연습문제 4: stack으로 괄호 매칭
`std::stack<char>`를 사용하여 괄호 문자열이 균형잡혀 있는지 확인하는 함수를 작성하세요.

### 연습문제 5: 복합 키를 위한 커스텀 해시
`Point3D { int x, y, z; }`에 대해 `std::hash`를 특수화하고 `std::unordered_map`에서 사용하세요.

---

## 다음 단계

[STL 알고리즘과 반복자](./12_STL_Algorithms_and_Iterators.md)에서 STL 알고리즘에 대해 알아봅시다!
